"""命令行摄取/查询/索引信息；检索配置统一读取 AETHER_ 环境变量。"""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from aether.config import Settings
from aether.retrieval.chunking import get_chunker
from aether.retrieval.embeddings import create_embedder
from aether.retrieval.ingest import index_info, ingest_directory, open_searcher


async def run(args: argparse.Namespace) -> dict:
    settings = Settings()
    if args.command == "info":
        return await asyncio.to_thread(index_info, args.index)
    embedder = create_embedder(settings)
    searcher = None
    try:
        if args.command == "ingest":
            searcher = await ingest_directory(
                args.corpus,
                get_chunker(args.chunker),
                embedder,
                args.index,
                dense_backend=settings.dense_backend,
            )
            return await asyncio.to_thread(index_info, args.index)
        searcher = await asyncio.to_thread(open_searcher, args.index, embedder)
        searcher.rrf_k = settings.retrieval_rrf_k
        searcher.route_depth = settings.retrieval_route_depth
        searcher.sparse_weight = settings.retrieval_sparse_weight
        searcher.dense_weight = settings.retrieval_dense_weight
        hits, trace = await searcher.search_with_trace(args.query, args.k, args.mode)
        return {"hits": [asdict(hit) for hit in hits], "trace": trace}
    finally:
        if searcher is not None:
            await searcher.aclose()
        if hasattr(embedder, "aclose"):
            await embedder.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("corpus", type=Path)
    ingest.add_argument("--chunker", default="markdown")
    query = sub.add_parser("query")
    query.add_argument("query")
    query.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    query.add_argument("--k", type=int, default=5)
    info = sub.add_parser("info")
    for child in (ingest, query, info):
        child.add_argument("--index", type=Path, default=Path("data/index.db"))
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
