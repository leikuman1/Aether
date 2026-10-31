"""一次问答的运行入口。检查点固定用 SqliteSaver，按 thread_id 恢复。"""

import asyncio
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from aether.config import Settings
from aether.graph.builder import build_graph
from aether.graph.llm import create_chat_model
from aether.graph.nodes import Searcher
from aether.graph.state import TurnResult
from aether.retrieval.embeddings import create_embedder
from aether.retrieval.hybrid import HybridSearcher
from aether.retrieval.ingest import open_searcher
from aether.retrieval.types import Embedder


def configure_searcher(searcher: HybridSearcher, settings: Settings) -> HybridSearcher:
    searcher.rrf_k = settings.retrieval_rrf_k
    searcher.route_depth = settings.retrieval_route_depth
    searcher.sparse_weight = settings.retrieval_sparse_weight
    searcher.dense_weight = settings.retrieval_dense_weight
    return searcher


async def open_index(settings: Settings, index_path: Path) -> tuple[Embedder, HybridSearcher]:
    embedder: Embedder = create_embedder(settings)
    searcher = await asyncio.to_thread(
        open_searcher,
        index_path,
        embedder,
        rrf_k=settings.retrieval_rrf_k,
        route_depth=settings.retrieval_route_depth,
    )
    return embedder, configure_searcher(searcher, settings)


def _result(thread_id: str, snapshot) -> TurnResult:
    values = dict(snapshot.values or {})
    interrupts = tuple(snapshot.interrupts or ())
    payload = interrupts[0].value if interrupts else None
    if payload is not None and not isinstance(payload, dict):
        payload = {"value": payload}
    hits = list(values.get("turn_hits") or [])
    return {
        "thread_id": thread_id,
        "interrupted": bool(snapshot.next),
        "interrupt": payload,
        "next": list(snapshot.next),
        "decision": str(values.get("decision") or ""),
        "answer": str(values.get("answer") or ""),
        "abstain_reason": str(values.get("abstain_reason") or ""),
        "citations": list(values.get("citations") or []),
        "citation_violations": list(values.get("citation_violations") or []),
        "retrieved_ids": [hit["chunk_id"] for hit in hits],
        "rewrite_count": int(values.get("rewrite_count") or 0),
        "attempts": list(values.get("attempts") or []),
        "messages": list(values.get("messages") or []),
        "grade": str(values.get("grade") or ""),
        "error_code": str(values.get("error_code") or ""),
        "verified": bool(values.get("verified")),
    }


async def run_turn(
    *,
    searcher: Searcher,
    llm: BaseChatModel,
    checkpoint_path: str | Path,
    thread_id: str,
    rewrite_budget: int,
    min_support_terms: int,
    min_quote_chars: int,
    retrieval_k: int,
    question: str | None = None,
    require_review: bool = False,
    resume: dict | None = None,
) -> TurnResult:
    """执行一轮，或在同一 thread_id 上提交中断后的人工决定。"""
    if not thread_id.strip():
        raise ValueError("thread_id is required")
    if rewrite_budget < 0 or min_support_terms < 1 or min_quote_chars < 1 or retrieval_k < 1:
        raise ValueError("invalid agent limits")
    path = Path(checkpoint_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 80,
    }
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        compiled = build_graph(llm, searcher, retrieval_k=retrieval_k).compile(checkpointer=saver)
        snapshot = await compiled.aget_state(config)
        pending = bool(snapshot.next)
        if resume is None:
            if pending:
                raise ValueError("thread is interrupted; resume before starting a new question")
            if question is None or not str(question).strip():
                raise ValueError("question is required")
            await compiled.ainvoke(
                {
                    "question": question.strip(),
                    "rewrite_count": 0,
                    "rewrite_budget": rewrite_budget,
                    "min_support_terms": min_support_terms,
                    "min_quote_chars": min_quote_chars,
                    "require_review": require_review,
                    "previous_queries": [],
                    "hits": [],
                    "turn_hits": [],
                    "attempts": [],
                    "supporting": [],
                    "citations": [],
                    "citation_violations": [],
                    "messages": [],
                },
                config,
            )
        else:
            if not pending:
                raise ValueError("thread has no pending interrupt")
            await compiled.ainvoke(Command(resume=resume), config)
        snapshot = await compiled.aget_state(config)
    return _result(thread_id, snapshot)


def chat_model_for(settings: Settings) -> BaseChatModel:
    return create_chat_model(settings)


async def read_thread(
    *,
    searcher: Searcher,
    llm: BaseChatModel,
    checkpoint_path: str | Path,
    thread_id: str,
    retrieval_k: int,
) -> TurnResult:
    """只读检查点，不推进图。用于跨进程确认同一 thread 的消息还在。"""
    if not thread_id.strip():
        raise ValueError("thread_id is required")
    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise ValueError("checkpoint file does not exist")
    config = {"configurable": {"thread_id": thread_id}}
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        compiled = build_graph(llm, searcher, retrieval_k=retrieval_k).compile(checkpointer=saver)
        snapshot = await compiled.aget_state(config)
        if snapshot.created_at is None:
            raise ValueError(f"unknown thread: {thread_id}")
    return _result(thread_id, snapshot)
