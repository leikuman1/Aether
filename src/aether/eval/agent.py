"""对抗集弃答率与可回答集的引用完整性。门禁不依赖模型自称弃答。"""

import asyncio
import json
import tempfile
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from aether.config import Settings
from aether.graph.citations import extract_chunk_ids, fabricated_citation_ids
from aether.graph.llm import create_chat_model
from aether.graph.runner import configure_searcher, run_turn
from aether.retrieval.chunking import get_chunker
from aether.retrieval.embeddings import create_embedder
from aether.retrieval.ingest import ingest_directory, read_meta

MIN_ADVERSARIAL = 20
MIN_ANSWERABLE = 10
ABSTAIN_RATE_MIN = 0.80
ANSWER_RATE_MIN = 0.80


def load_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"empty or duplicate ids: {path}")
    for row in rows:
        if row.get("expect") not in {"abstain", "answer"} or not str(row.get("query", "")).strip():
            raise ValueError(f"invalid agent golden row: {row.get('id')}")
    return rows


def _integrity(result: dict) -> tuple[list[str], bool]:
    allowed = set(result["retrieved_ids"])
    fabricated = fabricated_citation_ids(result["answer"], allowed)
    citations = [str(item).lower() for item in result["citations"]]
    if result["decision"] == "answer":
        ok = (
            not fabricated
            and bool(citations)
            and set(citations) <= allowed
            and citations == extract_chunk_ids(result["answer"])
        )
    elif result["decision"] == "abstain":
        ok = not fabricated and not extract_chunk_ids(result["answer"])
    else:
        ok = False
    return fabricated, ok


def _item(row: dict, result: dict) -> dict:
    fabricated, integrity_ok = _integrity(result)
    return {
        "id": row["id"],
        "expect": row["expect"],
        "query": row["query"],
        "relevant_chunk_ids": [str(item).lower() for item in row.get("relevant_chunk_ids") or []],
        "decision": result["decision"],
        "abstain_reason": result["abstain_reason"],
        "answer": result["answer"],
        "citations": result["citations"],
        "retrieved_ids": result["retrieved_ids"],
        "fabricated": fabricated,
        "integrity_ok": integrity_ok,
        "rewrite_count": result["rewrite_count"],
        "attempts": len(result["attempts"]),
        "grade": result["grade"],
    }


def summarize(items: list[dict], *, meta: dict, settings: Settings) -> dict:
    adversarial = [item for item in items if item["expect"] == "abstain"]
    answerable = [item for item in items if item["expect"] == "answer"]
    abstain_rate = (
        sum(item["decision"] == "abstain" for item in adversarial) / len(adversarial)
        if adversarial
        else 0.0
    )
    answer_rate = (
        sum(item["decision"] == "answer" for item in answerable) / len(answerable)
        if answerable
        else 0.0
    )
    fabricated = sum(len(item["fabricated"]) for item in items)
    integrity_ok = all(item["integrity_ok"] for item in items) if items else False
    exhausted = (
        sum(item["abstain_reason"] == "budget_exhausted" for item in adversarial) / len(adversarial)
        if adversarial
        else 0.0
    )
    labeled = [
        item for item in answerable if item["relevant_chunk_ids"] and item["decision"] == "answer"
    ]
    gold_hits = sum(
        bool(set(item["citations"]) & set(item["relevant_chunk_ids"])) for item in labeled
    )
    gold_citation_rate = gold_hits / len(labeled) if labeled else 0.0
    gate_passed = (
        len(adversarial) >= MIN_ADVERSARIAL
        and len(answerable) >= MIN_ANSWERABLE
        and abstain_rate >= ABSTAIN_RATE_MIN
        and answer_rate >= ANSWER_RATE_MIN
        and fabricated == 0
        and integrity_ok
        and exhausted >= ABSTAIN_RATE_MIN
    )
    return {
        "embedder": meta.get("embedder"),
        "corpus_fingerprint": meta.get("corpus_fingerprint"),
        "chunks": int(meta.get("num_chunks", 0)),
        "dense_backend": meta.get("dense_backend"),
        "rewrite_budget": settings.agent_rewrite_budget,
        "min_support_terms": settings.agent_min_support_terms,
        "retrieval_k": settings.agent_retrieval_k,
        "adversarial": len(adversarial),
        "answerable": len(answerable),
        "abstain_rate": abstain_rate,
        "answer_rate": answer_rate,
        "budget_exhausted_rate": exhausted,
        "gold_citation_rate": gold_citation_rate,
        "fabricated_citations": fabricated,
        "integrity_ok": integrity_ok,
        "gate_passed": gate_passed,
        "thresholds": {
            "abstain_rate": ABSTAIN_RATE_MIN,
            "answer_rate": ANSWER_RATE_MIN,
            "min_adversarial": MIN_ADVERSARIAL,
            "min_answerable": MIN_ANSWERABLE,
        },
        "items": items,
    }


async def evaluate_agent(
    settings: Settings,
    corpus: Path,
    adversarial_path: Path,
    answerable_path: Path,
    *,
    llm: BaseChatModel | None = None,
) -> dict:
    adversarial = await asyncio.to_thread(load_jsonl, adversarial_path)
    answerable = await asyncio.to_thread(load_jsonl, answerable_path)
    model = llm or create_chat_model(settings)
    embedder = create_embedder(settings)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        index_path = root / "index.db"
        checkpoint = root / "sessions.sqlite"
        searcher = await ingest_directory(
            corpus,
            get_chunker("markdown"),
            embedder,
            index_path,
            dense_backend=settings.dense_backend,
        )
        configure_searcher(searcher, settings)
        meta = await asyncio.to_thread(read_meta, index_path)
        items: list[dict] = []
        try:
            for row in [*adversarial, *answerable]:
                result = await run_turn(
                    question=row["query"],
                    searcher=searcher,
                    llm=model,
                    checkpoint_path=checkpoint,
                    thread_id=f"agent-{row['id']}",
                    rewrite_budget=settings.agent_rewrite_budget,
                    min_support_terms=settings.agent_min_support_terms,
                    min_quote_chars=settings.agent_min_quote_chars,
                    retrieval_k=settings.agent_retrieval_k,
                )
                items.append(_item(row, result))
        finally:
            await searcher.aclose()
            if hasattr(embedder, "aclose"):
                await embedder.aclose()
    return summarize(items, meta=meta, settings=settings)
