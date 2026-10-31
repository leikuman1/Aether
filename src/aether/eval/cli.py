"""aether-eval retrieval：三路 × 两种分块，JSON 报告与终端表格。"""

import argparse
import asyncio
import json
from pathlib import Path

from aether.config import Settings
from aether.eval.retrieval import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["retrieval"])
    parser.add_argument("--corpus", type=Path, default=Path("data/golden/corpus"))
    parser.add_argument("--golden", type=Path, default=Path("data/golden/retrieval.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    output = args.output or settings.data_dir / "eval" / "retrieval.json"
    try:
        report = asyncio.run(evaluate(settings, args.corpus, args.golden))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"evaluation failed: {exc}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    if args.gate and not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
