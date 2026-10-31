"""每个节点都是 AgentState 上的可单测函数，引用校验不靠模型自觉。"""

import inspect
import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from aether.graph.citations import (
    extract_chunk_ids,
    extractive_remainder,
    fabricated_citation_ids,
    is_extractive,
)
from aether.graph.grading import accept_support, constrain_query, content_terms, quote_supports
from aether.graph.llm import LLMUnavailable
from aether.graph.nodes import (
    ABSTAIN_TEXT,
    abstain,
    generate,
    grade,
    plan,
    retrieve,
    review,
    rewrite,
    route_after_grade,
    route_after_plan,
    route_after_verify,
    verify,
)
from aether.retrieval.types import Chunk, RetrievedChunk

CID = "0123456789abcdef0123"
FAKE_CID = "ffffffffffffffffffffffffffff"
QUOTE = "quasar rollback pointer switches blue green traffic after checks"


def _hit(text: str = QUOTE, chunk_id: str = CID) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "quasar",
        "source": "quasar.md",
        "heading": "Release",
        "text": text,
        "score": 1.0,
    }


def _state(**kwargs) -> dict:
    state = {
        "question": "quasar rollback pointer switches blue green traffic",
        "rewrite_count": 0,
        "rewrite_budget": 2,
        "min_support_terms": 2,
        "min_quote_chars": 12,
        "require_review": False,
        "hits": [_hit()],
        "turn_hits": [_hit()],
        "attempts": [],
        "supporting": [],
        "previous_queries": [],
        "messages": [],
        "query": "quasar rollback pointer switches blue green traffic",
    }
    state.update(kwargs)
    return state


class _Searcher:
    def __init__(self, hits):
        self.hits = hits
        self.queries: list[str] = []

    async def search(self, query: str, k: int = 5):
        self.queries.append(query)
        return self.hits[:k]


def _retrieved(text: str = QUOTE, chunk_id: str = CID) -> RetrievedChunk:
    chunk = Chunk(chunk_id, "quasar", "quasar.md", 0, "Release", text)
    return RetrievedChunk(chunk, 0.5, {"bm25": 0.5})


def test_nodes_are_async_functions_of_state() -> None:
    for fn in (plan, retrieve, grade, rewrite, review, generate, verify, abstain):
        assert inspect.iscoroutinefunction(fn)
        assert "state" in inspect.signature(fn).parameters


def test_citation_ids_must_come_from_retrieved_set() -> None:
    answer = f"{QUOTE} [{CID}] and also [{FAKE_CID}]"
    assert extract_chunk_ids(answer) == [CID, FAKE_CID]
    assert fabricated_citation_ids(answer, {CID}) == [FAKE_CID]
    assert fabricated_citation_ids(f"{QUOTE} [{CID}]", {CID}) == []
    assert is_extractive(f"{QUOTE} [{CID}]", [QUOTE])
    assert extractive_remainder(f"Paris. {QUOTE} [{CID}]", [QUOTE]) == "paris"
    assert not is_extractive(f"Paris. {QUOTE} [{CID}]", [QUOTE])


def test_accept_support_drops_fabricated_ids_and_missing_anchors() -> None:
    hits = [_hit()]
    question = "quasar rollback pointer switches in version k1"
    forged = {
        "sufficient": True,
        "supporting": [
            {"chunk_id": FAKE_CID, "quote": QUOTE},
            {"chunk_id": CID, "quote": "not in the chunk"},
            {"chunk_id": CID, "quote": QUOTE},
        ],
    }
    # k1 不在原文里，整段引用也不能算支持。
    assert accept_support(question, hits, forged, min_terms=2, min_quote_chars=12) == []
    assert quote_supports(question, QUOTE, min_terms=2, min_quote_chars=12) is None
    clean = accept_support(
        "quasar rollback pointer switches blue green traffic",
        hits,
        {"sufficient": True, "supporting": [{"chunk_id": CID, "quote": QUOTE}]},
        min_terms=2,
        min_quote_chars=12,
    )
    assert clean[0]["chunk_id"] == CID
    assert constrain_query("bm25 k1 saturation", "bm25 elasticsearch", []) == " ".join(
        sorted({"bm25", "k1", "saturation"})
    )


@pytest.mark.asyncio
async def test_plan_retrieve_grade_rewrite_are_independent() -> None:
    listed = FakeListChatModel(
        responses=[
            json.dumps({"query": "quasar elasticsearch"}),
            json.dumps(
                {
                    "sufficient": True,
                    "supporting": [{"chunk_id": FAKE_CID, "quote": "invented quote text here"}],
                    "reason": "nope",
                }
            ),
            json.dumps({"query": "quasar rollback"}),
        ]
    )
    planned = await plan(_state(), listed)
    question = _state()["question"]
    assert "elasticsearch" not in planned["query"]
    assert content_terms(planned["query"]) <= content_terms(question)
    searcher = _Searcher([_retrieved()])
    retrieved = await retrieve({**_state(), "query": planned["query"]}, searcher, k=3)
    assert retrieved["hits"][0]["chunk_id"] == CID
    assert searcher.queries == [planned["query"]]
    graded = await grade({**_state(), **retrieved}, listed)
    assert graded["grade"] == "insufficient"
    assert graded["supporting"] == []
    assert graded["grade_reason"] == "support-rejected"
    rewritten = await rewrite({**_state(), "previous_queries": [planned["query"]]}, listed)
    assert rewritten["rewrite_count"] == 1
    assert rewritten["query"] == "quasar rollback"
    skipped = await review(_state(require_review=False))
    assert skipped == {"human_decision": "skipped"}


@pytest.mark.asyncio
async def test_grade_accepts_verbatim_quote_and_verify_rejects_extra_claims() -> None:
    listed = FakeListChatModel(
        responses=[
            json.dumps(
                {
                    "sufficient": True,
                    "supporting": [{"chunk_id": CID, "quote": QUOTE}],
                    "reason": "quoted",
                }
            ),
            json.dumps({"answer": f"extra claim {QUOTE} [{CID}]"}),
        ]
    )
    graded = await grade(_state(), listed)
    assert graded["grade"] == "sufficient"
    assert graded["supporting"][0]["quote"] == QUOTE
    drafted = await generate({**_state(), "supporting": graded["supporting"]}, listed)
    checked = await verify({**_state(), **graded, **drafted})
    assert checked["verified"] is False
    assert checked["verify_reason"] == "ungrounded"
    assert "answer" not in checked
    honest = await verify(
        {
            **_state(),
            **graded,
            "draft": f"{QUOTE} [{CID}]",
            "error_code": "",
        }
    )
    assert honest["verified"] is True
    assert honest["citations"] == [CID]
    assert honest["decision"] == "answer"
    rejected = await abstain({**_state(), **checked, "grade": "sufficient"})
    assert rejected["decision"] == "abstain"
    assert rejected["abstain_reason"] == "ungrounded"
    assert fabricated_citation_ids(rejected["answer"], {CID}) == []
    assert CID not in rejected["answer"]
    assert rejected["answer"] == ABSTAIN_TEXT["ungrounded"]


@pytest.mark.asyncio
async def test_plan_provider_error_resets_previous_turn_decisions(monkeypatch) -> None:
    monkeypatch.setattr(
        "aether.graph.nodes.chat_json", AsyncMock(side_effect=LLMUnavailable("llm_unavailable"))
    )
    state = _state(human_decision="rejected", verify_reason="ungrounded", verified=True)
    planned = await plan(state, FakeListChatModel(responses=["unused"]))
    assert planned["human_decision"] == ""
    assert planned["verify_reason"] == ""
    assert planned["verified"] is False
    assert planned["grade"] == "error"
    assert planned["error_code"] == "llm_unavailable"
    result = await abstain({**state, **planned})
    assert result["abstain_reason"] == "provider_error"
    assert result["answer"] == ABSTAIN_TEXT["provider_error"]


@pytest.mark.asyncio
async def test_verify_requires_accepted_quote_for_every_cited_id() -> None:
    other_id = "abcdef0123456789abcd"
    other_quote = "quasar rollback checks deployment health before switching traffic"
    checked = await verify(
        _state(
            turn_hits=[_hit(), _hit(other_quote, other_id)],
            supporting=[{"chunk_id": CID, "quote": QUOTE}],
            draft=f"{QUOTE} [{CID}]\n{other_quote} [{other_id}]",
        )
    )
    assert checked["verified"] is False
    assert checked["verify_reason"] == "ungrounded"
    assert checked["citations"] == []


@pytest.mark.asyncio
async def test_verify_rejects_unaccepted_text_from_a_cited_chunk() -> None:
    full_text = f"{QUOTE}. The release also purges the deployment cache."
    checked = await verify(
        _state(
            turn_hits=[_hit(full_text)],
            supporting=[{"chunk_id": CID, "quote": QUOTE}],
            draft=f"{full_text} [{CID}]",
        )
    )
    assert checked["verified"] is False
    assert checked["verify_reason"] == "ungrounded"


def test_routes_cover_budget_review_and_verify() -> None:
    assert route_after_plan({"error_code": "llm_unavailable"}) == "abstain"
    assert route_after_plan({"query": "bm25"}) == "retrieve"
    assert route_after_grade({"grade": "sufficient"}) == "review"
    exhausted = {"grade": "insufficient", "rewrite_count": 2, "rewrite_budget": 2}
    assert route_after_grade(exhausted) == "abstain"
    retry = {"grade": "insufficient", "rewrite_count": 1, "rewrite_budget": 2}
    assert route_after_grade(retry) == "rewrite"
    assert route_after_grade({"grade": "error", "error_code": "llm_unavailable"}) == "abstain"
    assert route_after_verify({"verified": True}) == "end"
    assert route_after_verify({"verified": False}) == "abstain"


@pytest.mark.asyncio
async def test_abstain_on_budget_has_no_citations() -> None:
    result = await abstain(
        _state(grade="insufficient", rewrite_count=2, rewrite_budget=2, verify_reason="")
    )
    assert result["abstain_reason"] == "budget_exhausted"
    assert extract_chunk_ids(result["answer"]) == []
    assert result["messages"][1]["decision"] == "abstain"
