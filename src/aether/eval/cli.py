"""aether-eval：检索三路指标，以及纠错 Agent 的弃答/引用门禁。"""

import argparse
import asyncio
import json
from pathlib import Path

from aether.config import Settings
from aether.eval.agent import evaluate_agent
from aether.eval.retrieval import evaluate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["retrieval", "agent"])
    parser.add_argument("--corpus", type=Path, default=Path("data/golden/corpus"))
    parser.add_argument("--golden", type=Path, default=Path("data/golden/retrieval.jsonl"))
    parser.add_argument(
        "--adversarial",
        type=Path,
        default=Path("data/golden/agent_adversarial.jsonl"),
    )
    parser.add_argument(
        "--answerable",
        type=Path,
        default=Path("data/golden/agent_answerable.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", action="store_true")
    return parser


def _print_retrieval(report: dict, output: Path) -> None:
    print(f"embedder={report['embedder']} queries={report['queries']}")
    print(f"{'chunker':<20} {'route':<8} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR@10':>8}")
    for strategy in report["strategies"]:
        for route, scores in strategy["routes"].items():
            print(
                f"{strategy['chunker']:<20} {route:<8} "
                f"{scores['recall@1']:7.4f} {scores['recall@5']:7.4f} "
                f"{scores['recall@10']:7.4f} {scores['mrr@10']:8.4f}"
            )
    print(f"gate={'PASS' if report['gate_passed'] else 'FAIL'} JSON={output}")


def _print_agent(report: dict, output: Path) -> None:
    print(
        f"embedder={report['embedder']} adversarial={report['adversarial']} "
        f"answerable={report['answerable']}"
    )
    print(
        f"abstain_rate={report['abstain_rate']:.4f} "
        f"answer_rate={report['answer_rate']:.4f} "
        f"budget_exhausted_rate={report['budget_exhausted_rate']:.4f} "
        f"gold_citation_rate={report['gold_citation_rate']:.4f} "
        f"fabricated_citations={report['fabricated_citations']}"
    )
    print(f"gate={'PASS' if report['gate_passed'] else 'FAIL'} JSON={output}")


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    settings = Settings()
    if args.command == "agent":
        output = args.output or settings.data_dir / "eval" / "agent.json"
        try:
            report = asyncio.run(
                evaluate_agent(settings, args.corpus, args.adversarial, args.answerable)
            )
        except (ValueError, OSError) as exc:
            parser.exit(2, f"evaluation failed: {exc}\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _print_agent(report, output)
    else:
        output = args.output or settings.data_dir / "eval" / "retrieval.json"
        try:
            report = asyncio.run(evaluate(settings, args.corpus, args.golden))
        except (ValueError, OSError) as exc:
            parser.exit(2, f"evaluation failed: {exc}\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _print_retrieval(report, output)
    if args.gate and not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
