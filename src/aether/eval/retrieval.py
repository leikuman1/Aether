"""冻结相关性标签；报告三路、两种分块的 recall@k / MRR@k 与逐查询证据。"""

import asyncio
import hashlib
import json
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from aether.config import Settings
from aether.retrieval.chunking import MarkdownChunker, get_chunker
from aether.retrieval.embeddings import create_embedder
from aether.retrieval.ingest import corpus_fingerprint, ingest_directory
from aether.retrieval.loaders import load_directory
from aether.retrieval.types import Chunk


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        raise ValueError("k must be positive and relevant IDs cannot be empty")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        raise ValueError("k must be positive and relevant IDs cannot be empty")
    return next((1 / i for i, cid in enumerate(ranked[:k], 1) if cid in relevant), 0.0)


def evidence_ids(row: dict, chunks: list[Chunk]) -> set[str]:
    """证据相关片段：包含至少一条证据的一半连续原文（>=40字），与检索结果无关。

    完整小节证据跨窗口时允许部分覆盖；不按 doc_id 粗放放宽标签。
    """
    ids = set()
    for evidence in row["evidence"]:
        text = evidence["text"]
        width = max(40, (len(text) + 1) // 2)
        matched = {
            c.chunk_id
            for c in chunks
            if c.doc_id == evidence["doc_id"]
            and any(text[i : i + width] in c.text for i in range(len(text) - width + 1))
        }
        if not matched:
            raise ValueError(f"unmapped evidence in {row['id']}: {evidence['doc_id']}")
        ids.update(matched)
    if not ids:
        raise ValueError(f"empty relevance labels: {row['id']}")
    return ids


def load_golden(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("golden set is empty or contains duplicate IDs")
    for row in rows:
        if not row["query"].strip() or not row["evidence"]:
            raise ValueError("golden query/evidence cannot be empty")
    return rows


def _aggregate(records: list[dict]) -> dict:
    return {
        "queries": len(records),
        "recall@1": statistics.mean(r["recall@1"] for r in records),
        "recall@5": statistics.mean(r["recall@5"] for r in records),
        "recall@10": statistics.mean(r["recall@10"] for r in records),
        "mrr@10": statistics.mean(r["mrr@10"] for r in records),
    }


async def evaluate(settings: Settings, corpus: Path, golden_path: Path) -> dict:
    rows = await asyncio.to_thread(load_golden, golden_path)
    docs = await load_directory(corpus)
    canonical = [c for doc in docs for c in MarkdownChunker().split(doc)]
    for row in rows:
        expected = evidence_ids(row, canonical)
        if set(row["relevant_chunk_ids"]) != expected:
            raise ValueError(f"stale canonical relevant_chunk_ids: {row['id']}")
        for evidence in row["evidence"]:
            if not any(d.doc_id == evidence["doc_id"] and evidence["text"] in d.text for d in docs):
                raise ValueError(f"evidence missing from corpus: {row['id']}")
    embedder = create_embedder(settings)
    report = {
        "schema_version": 1,
        "queries": len(rows),
        "embedder": embedder.name,
        "golden_sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
        "configuration": {
            "backend": settings.dense_backend,
            "rrf_k": settings.retrieval_rrf_k,
            "route_depth": settings.retrieval_route_depth,
            "sparse_weight": settings.retrieval_sparse_weight,
            "dense_weight": settings.retrieval_dense_weight,
        },
        "relevance_rule": "same doc, >= half continuous evidence text (minimum 40 chars)",
        "strategies": [],
    }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for strategy in ("fixed", "markdown"):
                chunker = get_chunker(strategy)
                searcher = await ingest_directory(
                    corpus,
                    chunker,
                    embedder,
                    Path(tmp) / f"{strategy}.db",
                    dense_backend=settings.dense_backend,
                )
                searcher.rrf_k = settings.retrieval_rrf_k
                searcher.route_depth = settings.retrieval_route_depth
                searcher.sparse_weight = settings.retrieval_sparse_weight
                searcher.dense_weight = settings.retrieval_dense_weight
                chunks = list(searcher.chunks.values())
                labels = {r["id"]: evidence_ids(r, chunks) for r in rows}
                entry = {
                    "chunker": chunker.name,
                    "chunks": len(chunks),
                    "corpus_fingerprint": corpus_fingerprint(chunks),
                    "routes": {},
                }
                try:
                    for mode in ("bm25", "dense", "hybrid"):
                        records = []
                        for row in rows:
                            hits, trace = await searcher.search_with_trace(row["query"], 10, mode)
                            ranked = [h.chunk.chunk_id for h in hits]
                            relevant = labels[row["id"]]
                            records.append(
                                {
                                    "id": row["id"],
                                    "query": row["query"],
                                    "category": row["category"],
                                    "relevant_chunk_ids": sorted(relevant),
                                    "retrieved_chunk_ids": ranked,
                                    **{
                                        f"recall@{k}": recall_at_k(ranked, relevant, k)
                                        for k in (1, 5, 10)
                                    },
                                    "mrr@10": reciprocal_rank(ranked, relevant, 10),
                                    "hits": [
                                        {
                                            "chunk_id": h.chunk.chunk_id,
                                            "score": h.score,
                                            "route_scores": h.route_scores,
                                        }
                                        for h in hits
                                    ],
                                    "trace": trace,
                                }
                            )
                        grouped = defaultdict(list)
                        for record in records:
                            grouped[record["category"]].append(record)
                        times = sorted(r["trace"]["total_ms"] for r in records)
                        entry["routes"][mode] = {
                            **_aggregate(records),
                            "latency_p50_ms": statistics.median(times),
                            "latency_p95_ms": times[max(0, int(len(times) * 0.95) - 1)],
                            "categories": {
                                key: _aggregate(values) for key, values in grouped.items()
                            },
                            "per_query": records,
                        }
                finally:
                    await searcher.aclose()
                routes = entry["routes"]
                entry["gate_passed"] = routes["hybrid"]["recall@5"] + 1e-12 >= max(
                    routes["bm25"]["recall@5"],
                    routes["dense"]["recall@5"],
                )
                report["strategies"].append(entry)
    finally:
        if hasattr(embedder, "aclose"):
            await embedder.aclose()
    report["gate_passed"] = len(rows) >= 50 and all(s["gate_passed"] for s in report["strategies"])
    return report
