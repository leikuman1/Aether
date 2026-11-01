"""aether-eval：检索三路指标、纠错 Agent 门禁，以及答案质量评测（judge + 引用真实性）。"""

import argparse
import asyncio
import json
from pathlib import Path

from aether.config import Settings
from aether.eval.agent import evaluate_agent
from aether.eval.answer import evaluate_answers
from aether.eval.judge import (
    JudgeError,
    RecordingJudge,
    ReplayingJudge,
    create_judge,
)
from aether.eval.retrieval import evaluate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["retrieval", "agent", "answer"])
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
    parser.add_argument(
        "--answer-golden",
        type=Path,
        default=Path("data/golden/answer.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", action="store_true")
    # answer 评测：judge 后端与 fixtures 录制/回放
    parser.add_argument("--judge", choices=["fake", "openai", "replay"], default="fake")
    parser.add_argument("--record", action="store_true", help="录制 judge 响应到 --fixtures")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/golden/answer_judge_fixtures.jsonl"),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="答案评测历史 JSONL（默认 data/eval/answer_history.jsonl）",
    )
    parser.add_argument("--baseline", type=Path, default=None, help="对比基准，覆盖历史")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="judge 同输入重复次数（稳定性实验）",
    )
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


def _print_answer(report: dict, output: Path) -> None:
    metrics = report["metrics"]
    print(
        f"embedder={report['embedder']} judge={report['judge']['model']} "
        f"items={metrics['items']} answered={metrics['answered']}"
    )
    print(
        f"answer_rate={metrics['answer_rate']:.4f} "
        f"citation_pass_rate={metrics['citation_pass_rate']:.4f} "
        f"gold_citation_rate={metrics['gold_citation_rate']:.4f} "
        f"fabricated_citations={metrics['fabricated_citations']}"
    )
    thresholds = report["thresholds"]
    print(
        f"faithfulness={metrics['faithfulness']:.4f} "
        f"answer_relevance={metrics['answer_relevance']:.4f} "
        f"(thresholds {thresholds['faithfulness']:.2f}/{thresholds['answer_relevance']:.2f})"
    )
    stability = report["stability"]
    print(
        f"stability repeats={stability['repeats']} "
        f"faithfulness_dev={stability['faithfulness_max_deviation']:.4f} "
        f"relevance_dev={stability['answer_relevance_max_deviation']:.4f}"
    )
    comparison = report["comparison"]
    if comparison is None:
        print("comparison: first run - no previous report")
    else:
        deltas = " ".join(f"{key}={comparison['delta'][key]:+.4f}" for key in comparison["delta"])
        print(f"comparison: {comparison['source']}@{comparison['previous_recorded_at']} {deltas}")
        if comparison["regressed"]:
            print(f"regressed: {' '.join(comparison['regressed'])}")
    print(f"gate={'PASS' if report['gate_passed'] else 'FAIL'} JSON={output}")


def _write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_answer(args: argparse.Namespace, settings: Settings) -> dict:
    if args.judge == "replay" and args.record:
        raise SystemExit("--record replays nothing; record with --judge fake or openai")
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.judge == "replay":
        # 回放模式也固定生成/embedding 为 Fake，已有本机密钥不能意外触发外部请求。
        settings = settings.model_copy(update={"llm_api_key": "", "embedding_provider": "fake"})
        judge = ReplayingJudge(args.fixtures)
    else:
        judge = create_judge(settings, args.judge, record=args.record)
    history = args.history or settings.data_dir / "eval" / "answer_history.jsonl"
    report = asyncio.run(
        evaluate_answers(
            settings,
            args.corpus,
            args.answer_golden,
            judge=judge,
            repeat=args.repeat,
            history_path=history,
            baseline_path=args.baseline,
        )
    )
    if args.record and isinstance(judge, RecordingJudge):
        args.fixtures.parent.mkdir(parents=True, exist_ok=True)
        args.fixtures.write_text("\n".join(judge.fixture_lines()) + "\n", encoding="utf-8")
        print(f"fixtures={args.fixtures} entries={len(judge.recordings)}")
    return report


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
        _write_report(report, output)
        _print_agent(report, output)
    elif args.command == "answer":
        output = args.output or settings.data_dir / "eval" / "answer.json"
        try:
            report = _run_answer(args, settings)
        except (JudgeError, ValueError, OSError) as exc:
            parser.exit(2, f"evaluation failed: {exc}\n")
        _write_report(report, output)
        _print_answer(report, output)
    else:
        output = args.output or settings.data_dir / "eval" / "retrieval.json"
        try:
            report = asyncio.run(evaluate(settings, args.corpus, args.golden))
        except (ValueError, OSError) as exc:
            parser.exit(2, f"evaluation failed: {exc}\n")
        _write_report(report, output)
        _print_retrieval(report, output)
    if args.gate and not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
