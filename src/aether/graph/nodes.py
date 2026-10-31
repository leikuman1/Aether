"""状态机节点。每个节点都是 AgentState 上的异步函数，可脱离图单独调用。"""

from collections.abc import Mapping
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import interrupt

from aether.graph.citations import extract_chunk_ids, fabricated_citation_ids, is_extractive
from aether.graph.grading import accept_support, constrain_query
from aether.graph.llm import LLMUnavailable, chat_json
from aether.graph.state import AgentState, Hit
from aether.retrieval.types import RetrievedChunk

ABSTAIN_TEXT = {
    "human_rejected": "本轮未获确认，已中止，不作答。",
    "provider_error": "模型调用失败，已弃答，不根据语料硬答。",
    "empty_question": "问题为空，无法作答。",
    "citation_integrity": "引用未通过完整性校验，拒绝作答。",
    "ungrounded": "回答超出检索原文，拒绝作答。",
    "no_citation": "回答缺少可核对引用，拒绝作答。",
    "budget_exhausted": "证据不足且改写预算已用尽，无法根据当前语料回答。",
    "insufficient_evidence": "证据不足，无法根据当前语料回答。",
}


class Searcher(Protocol):
    async def search(self, query: str, k: int = 5) -> list[RetrievedChunk]: ...


def hit_from_retrieved(item: RetrievedChunk) -> Hit:
    chunk = item.chunk
    return {
        "chunk_id": chunk.chunk_id.lower(),
        "doc_id": chunk.doc_id,
        "source": chunk.source,
        "heading": chunk.heading,
        "text": chunk.text,
        "score": float(item.score),
    }


def merge_hits(existing: list[Hit], new_hits: list[Hit]) -> list[Hit]:
    merged = {hit["chunk_id"]: hit for hit in existing}
    for hit in new_hits:
        merged.setdefault(hit["chunk_id"], hit)
    return list(merged.values())


def _messages(state: AgentState, answer: str, decision: str) -> list[dict]:
    return [
        {"role": "user", "content": state.get("question", ""), "decision": ""},
        {"role": "assistant", "content": answer, "decision": decision},
    ]


def infer_abstain_reason(state: AgentState) -> str:
    if state.get("human_decision") == "rejected":
        return "human_rejected"
    if state.get("error_code") in {"llm_unavailable", "empty_question"}:
        return "provider_error" if state["error_code"] == "llm_unavailable" else "empty_question"
    verify_reason = state.get("verify_reason") or ""
    if verify_reason in ABSTAIN_TEXT:
        return verify_reason
    budget = int(state.get("rewrite_budget", 0))
    if state.get("grade") == "insufficient" and int(state.get("rewrite_count", 0)) >= budget:
        return "budget_exhausted"
    return "insufficient_evidence"


async def plan(state: AgentState, llm: BaseChatModel) -> dict:
    question = str(state.get("question", "")).strip()
    if not question:
        return {
            "question": "",
            "plan": "",
            "query": "",
            "grade": "error",
            "error_code": "empty_question",
            "abstain_reason": "empty_question",
            "attempts": [],
            "citation_violations": [],
        }
    try:
        data = await chat_json(
            llm,
            "plan",
            {"question": question, "messages": list(state.get("messages") or [])[-6:]},
        )
    except LLMUnavailable:
        return {
            "question": question,
            "plan": "",
            "query": "",
            "grade": "error",
            "error_code": "llm_unavailable",
            "abstain_reason": "provider_error",
            "attempts": [],
            "citation_violations": [],
            "human_decision": "",
            "verify_reason": "",
            "verified": False,
        }
    proposed = question if not data else str(data.get("query") or question)
    query = constrain_query(question, proposed, [])
    return {
        "question": question,
        "plan": query,
        "query": query,
        "grade": "",
        "grade_reason": "",
        "error_code": "",
        "abstain_reason": "",
        "verified": False,
        "verify_reason": "",
        "decision": "",
        "answer": "",
        "draft": "",
        "citations": [],
        "citation_violations": [],
        "supporting": [],
        "attempts": [],
        "human_decision": "",
    }


async def retrieve(state: AgentState, searcher: Searcher, *, k: int) -> dict:
    if k <= 0:
        raise ValueError("k must be positive")
    query = str(state.get("query") or state.get("question") or "").strip()
    found = await searcher.search(query, k) if query else []
    hits = [hit_from_retrieved(item) for item in found]
    previous = list(state.get("previous_queries") or [])
    if query and query not in previous:
        previous.append(query)
    return {
        "query": query,
        "hits": hits,
        "turn_hits": merge_hits(list(state.get("turn_hits") or []), hits),
        "previous_queries": previous,
    }


async def grade(state: AgentState, llm: BaseChatModel) -> dict:
    question = str(state.get("question", ""))
    hits = list(state.get("hits") or [])
    min_terms = int(state.get("min_support_terms", 4))
    min_quote = int(state.get("min_quote_chars", 12))
    error_code = ""
    proposal: dict | None = {"sufficient": False, "supporting": [], "reason": "no-hits"}
    if hits:
        try:
            proposal = await chat_json(
                llm,
                "grade",
                {"question": question, "hits": hits},
            )
        except LLMUnavailable:
            error_code = "llm_unavailable"
            proposal = None
    supporting = (
        []
        if error_code
        else accept_support(
            question,
            hits,
            proposal,
            min_terms=min_terms,
            min_quote_chars=min_quote,
        )
    )
    model_yes = isinstance(proposal, dict) and proposal.get("sufficient") is True
    if error_code:
        decision = "error"
        reason = "llm_unavailable"
        supporting = []
    elif supporting and model_yes:
        decision = "sufficient"
        reason = str(proposal.get("reason") or "supported") if isinstance(proposal, dict) else ""
    else:
        decision = "insufficient"
        supporting = []
        if model_yes:
            reason = "support-rejected"
        elif isinstance(proposal, dict):
            reason = str(proposal.get("reason") or "insufficient")
        else:
            reason = "unparseable-grade"
    attempt = {
        "query": str(state.get("query") or question),
        "hit_ids": [hit["chunk_id"] for hit in hits],
        "grade": decision,
        "reason": reason,
    }
    return {
        "grade": decision,
        "grade_reason": reason,
        "supporting": supporting,
        "attempts": [*list(state.get("attempts") or []), attempt],
        "error_code": error_code,
        "abstain_reason": "provider_error" if error_code else "",
    }


async def rewrite(state: AgentState, llm: BaseChatModel) -> dict:
    question = str(state.get("question", ""))
    previous = list(state.get("previous_queries") or [])
    try:
        data = await chat_json(
            llm,
            "rewrite",
            {
                "question": question,
                "previous_queries": previous,
                "grade_reason": state.get("grade_reason", ""),
            },
        )
    except LLMUnavailable:
        return {
            "error_code": "llm_unavailable",
            "abstain_reason": "provider_error",
            "grade": "error",
        }
    proposed = "" if not data else str(data.get("query") or "")
    query = constrain_query(question, proposed, previous)
    return {
        "query": query,
        "rewrite_count": int(state.get("rewrite_count", 0)) + 1,
        "error_code": "",
    }


async def review(state: AgentState) -> dict:
    """足够的证据在生成前可以暂停，等待同一 thread 的 resume。"""
    if not state.get("require_review"):
        return {"human_decision": "skipped"}
    decision = interrupt(
        {
            "question": state.get("question", ""),
            "supporting_ids": [item["chunk_id"] for item in state.get("supporting") or []],
            "grade": state.get("grade", ""),
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, Mapping) else bool(decision)
    return {"human_decision": "approved" if approved else "rejected"}


async def generate(state: AgentState, llm: BaseChatModel) -> dict:
    supporting = list(state.get("supporting") or [])
    try:
        data = await chat_json(
            llm,
            "generate",
            {
                "question": state.get("question", ""),
                "supporting": supporting,
            },
        )
    except LLMUnavailable:
        return {
            "draft": "",
            "error_code": "llm_unavailable",
            "abstain_reason": "provider_error",
        }
    draft = "" if not data else str(data.get("answer") or "")
    return {"draft": draft, "error_code": ""}


async def verify(state: AgentState) -> dict:
    """确定性校验。失败时不把草稿写进 answer，避免编造引用流出。"""
    if state.get("error_code") == "llm_unavailable":
        return {
            "verified": False,
            "verify_reason": "provider_error",
            "abstain_reason": "provider_error",
            "citations": [],
        }
    draft = str(state.get("draft") or "")
    allowed = {hit["chunk_id"] for hit in state.get("turn_hits") or []}
    cited = extract_chunk_ids(draft)
    violations = fabricated_citation_ids(draft, allowed)
    if violations:
        return {
            "verified": False,
            "verify_reason": "citation_integrity",
            "citation_violations": violations,
            "citations": [],
            "abstain_reason": "citation_integrity",
        }
    if not cited:
        return {
            "verified": False,
            "verify_reason": "no_citation",
            "citation_violations": [],
            "citations": [],
            "abstain_reason": "no_citation",
        }
    accepted = {(item["chunk_id"], item["quote"]) for item in state.get("supporting") or []}
    supported_ids = {chunk_id for chunk_id, _ in accepted}
    quotes = [quote for chunk_id, quote in accepted if chunk_id in cited]
    if not set(cited) <= supported_ids or not is_extractive(draft, quotes):
        return {
            "verified": False,
            "verify_reason": "ungrounded",
            "citation_violations": [],
            "citations": [],
            "abstain_reason": "ungrounded",
        }
    return {
        "verified": True,
        "verify_reason": "",
        "decision": "answer",
        "abstain_reason": "",
        "answer": draft,
        "citations": cited,
        "citation_violations": [],
        "error_code": "",
        "messages": _messages(state, draft, "answer"),
    }


async def abstain(state: AgentState) -> dict:
    reason = infer_abstain_reason(state)
    answer = ABSTAIN_TEXT[reason]
    return {
        "decision": "abstain",
        "abstain_reason": reason,
        "answer": answer,
        "citations": [],
        "verified": False,
        "messages": _messages(state, answer, "abstain"),
    }


def route_after_plan(state: AgentState) -> str:
    failed = state.get("error_code") or state.get("grade") == "error"
    if failed or not str(state.get("query") or ""):
        return "abstain"
    return "retrieve"


def route_after_grade(state: AgentState) -> str:
    if state.get("error_code") or state.get("grade") == "error":
        return "abstain"
    if state.get("grade") == "sufficient":
        return "review"
    if int(state.get("rewrite_count", 0)) >= int(state.get("rewrite_budget", 0)):
        return "abstain"
    return "rewrite"


def route_after_rewrite(state: AgentState) -> str:
    if state.get("error_code"):
        return "abstain"
    return "retrieve"


def route_after_review(state: AgentState) -> str:
    if state.get("human_decision") == "rejected":
        return "abstain"
    return "generate"


def route_after_verify(state: AgentState) -> str:
    if state.get("verified"):
        return "end"
    return "abstain"
