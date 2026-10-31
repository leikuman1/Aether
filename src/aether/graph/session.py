"""thread_id 会话的启动、中断恢复与查看。进程退出后可用同一 Sqlite 文件接着跑。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from aether.config import Settings
from aether.graph.llm import create_chat_model
from aether.graph.runner import open_index, read_thread, run_turn


def _settings(args: argparse.Namespace) -> Settings:
    updates = {}
    if args.min_support_terms is not None:
        updates["agent_min_support_terms"] = args.min_support_terms
    if args.rewrite_budget is not None:
        updates["agent_rewrite_budget"] = args.rewrite_budget
    if args.k is not None:
        updates["agent_retrieval_k"] = args.k
    settings = Settings()
    if updates:
        settings = settings.model_copy(update=updates)
    return settings


async def _run(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    embedder, searcher = await open_index(settings, args.index)
    llm = create_chat_model(settings)
    try:
        if args.command == "show":
            return await read_thread(
                searcher=searcher,
                llm=llm,
                checkpoint_path=args.checkpoint,
                thread_id=args.thread_id,
                retrieval_k=settings.agent_retrieval_k,
            )
        if args.command == "resume":
            approved = bool(args.approve)
            return await run_turn(
                question=None,
                searcher=searcher,
                llm=llm,
                checkpoint_path=args.checkpoint,
                thread_id=args.thread_id,
                rewrite_budget=settings.agent_rewrite_budget,
                min_support_terms=settings.agent_min_support_terms,
                min_quote_chars=settings.agent_min_quote_chars,
                retrieval_k=settings.agent_retrieval_k,
                resume={"approved": approved},
            )
        return await run_turn(
            question=args.question,
            searcher=searcher,
            llm=llm,
            checkpoint_path=args.checkpoint,
            thread_id=args.thread_id,
            rewrite_budget=settings.agent_rewrite_budget,
            min_support_terms=settings.agent_min_support_terms,
            min_quote_chars=settings.agent_min_quote_chars,
            retrieval_k=settings.agent_retrieval_k,
            require_review=args.require_review,
        )
    finally:
        await searcher.aclose()
        if hasattr(embedder, "aclose"):
            await embedder.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--thread-id", required=True)
    common.add_argument("--checkpoint", type=Path, required=True)
    common.add_argument("--index", type=Path, required=True)
    common.add_argument("--min-support-terms", type=int)
    common.add_argument("--rewrite-budget", type=int)
    common.add_argument("--k", type=int)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start", parents=[common])
    start.add_argument("--question", required=True)
    start.add_argument("--require-review", action="store_true")
    resume = sub.add_parser("resume", parents=[common])
    choice = resume.add_mutually_exclusive_group(required=True)
    choice.add_argument("--approve", action="store_true")
    choice.add_argument("--reject", action="store_true")
    sub.add_parser("show", parents=[common])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = asyncio.run(_run(args))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
