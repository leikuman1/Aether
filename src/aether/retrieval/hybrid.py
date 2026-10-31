"""BM25 / dense 独立召回与 RRF 融合；保留每路得分、阶段时延、重排接口。"""

import asyncio
import time
from dataclasses import dataclass
from typing import Literal

from aether.retrieval.dense import NumpyDenseIndex, SqliteVecDenseIndex
from aether.retrieval.sparse import Bm25Index
from aether.retrieval.types import Chunk, Embedder, Reranker, RetrievedChunk

SearchMode = Literal["bm25", "dense", "hybrid"]


class IdentityReranker:
    @property
    def name(self) -> str:
        return "identity"

    async def rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        k: int,
    ) -> list[RetrievedChunk]:
        return results[:k]


def rrf_fuse(
    routes: dict[str, list[tuple[str, float]]],
    *,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
    k: int = 10,
) -> list[tuple[str, float]]:
    if rrf_k <= 0 or k <= 0:
        raise ValueError("rrf_k and k must be positive")
    weights = weights or {}
    if any(w < 0 for w in weights.values()):
        raise ValueError("RRF weights cannot be negative")
    scores: dict[str, float] = {}
    for route, hits in routes.items():
        weight = weights.get(route, 1.0)
        if weight == 0:
            continue
        seen = set()
        for rank, (cid, _) in enumerate(hits, 1):
            if cid not in seen:
                scores[cid] = scores.get(cid, 0) + weight / (rrf_k + rank)
                seen.add(cid)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]


@dataclass
class HybridSearcher:
    chunks: dict[str, Chunk]
    bm25: Bm25Index
    dense: NumpyDenseIndex | SqliteVecDenseIndex
    embedder: Embedder
    rrf_k: int = 60
    route_depth: int = 50
    sparse_weight: float = 1.0
    dense_weight: float = 1.0
    reranker: Reranker | None = None

    async def search(
        self, query: str, k: int = 5, mode: SearchMode = "hybrid"
    ) -> list[RetrievedChunk]:
        results, _ = await self.search_with_trace(query, k, mode)
        return results

    async def search_with_trace(
        self,
        query: str,
        k: int = 5,
        mode: SearchMode = "hybrid",
    ) -> tuple[list[RetrievedChunk], dict]:
        if k <= 0 or self.route_depth <= 0:
            raise ValueError("k and route_depth must be positive")
        if mode not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"unknown search mode: {mode}")
        if not query.strip():
            return [], {"mode": mode, "total_ms": 0.0, "candidates": {}}
        started = time.perf_counter()
        timings = {}
        routes = {}
        depth = max(self.route_depth, k)
        if mode in ("bm25", "hybrid"):
            tick = time.perf_counter()
            routes["bm25"] = await asyncio.to_thread(self.bm25.search, query, depth)
            timings["bm25_ms"] = (time.perf_counter() - tick) * 1000
        if mode in ("dense", "hybrid"):
            tick = time.perf_counter()
            qvec = await self.embedder.embed_query(query)
            timings["embedding_ms"] = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            routes["dense"] = await asyncio.to_thread(self.dense.search, qvec, depth)
            timings["dense_ms"] = (time.perf_counter() - tick) * 1000
        if mode == "hybrid":
            hits = rrf_fuse(
                routes,
                rrf_k=self.rrf_k,
                weights={"bm25": self.sparse_weight, "dense": self.dense_weight},
                k=depth,
            )
        else:
            hits = routes[mode]
        route_maps = {route: dict(items) for route, items in routes.items()}
        results = [
            RetrievedChunk(
                self.chunks[cid],
                score,
                {route: values[cid] for route, values in route_maps.items() if cid in values},
            )
            for cid, score in hits
        ]
        if self.reranker is not None:
            results = await self.reranker.rerank(query, results, k)
        results = results[:k]
        return results, {
            "mode": mode,
            **timings,
            "total_ms": (time.perf_counter() - started) * 1000,
            "candidates": {route: len(items) for route, items in routes.items()},
            "reranker": self.reranker.name if self.reranker else None,
        }

    async def aclose(self) -> None:
        await asyncio.to_thread(self.dense.close)
