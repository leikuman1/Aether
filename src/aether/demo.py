"""一键离线演示：摄取黄金语料 → 提问 → 评测。

默认不读密钥、强制 Fake embedding。``--live`` 才使用当前 Settings 里的模型。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
from langsmith import tracing_context

from aether.api.app import create_app
from aether.api.observe import configure_logging
from aether.config import Settings
from aether.eval.agent import evaluate_agent
from aether.eval.answer import evaluate_answers
from aether.eval.judge import JudgeError, create_judge
from aether.eval.retrieval import evaluate

_REPO = Path(__file__).resolve().parents[2]
_QUESTION = "BM25 的 k1 和 b 分别控制什么，切块方式变化后要注意哪些统计量？"


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=_REPO / "data" / "golden" / "corpus")
    parser.add_argument("--data-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--question", default=_QUESTION)
    parser.add_argument("--thread-id", default="demo")
    parser.add_argument(
        "--eval",
        choices=["retrieval", "agent", "answer", "all", "skip"],
        default="retrieval",
        help="默认跑检索门禁；all 再加 agent 与答案门禁（更慢）；skip 只做摄取和提问",
    )
    parser.add_argument(
        "--judge",
        choices=["fake", "openai"],
        default="fake",
        help="答案评测的 judge。openai 需要密钥，离线演示请保持 fake",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="使用已配置的 AETHER_LLM_API_KEY 与 embedding；缺密钥则退出",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="演示摘要 JSON，默认写到 data-dir"
    )
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    settings = Settings(_env_file=None)
    corpus = Path(args.corpus).resolve()
    update: dict[str, Any] = {
        "data_dir": Path(args.data_dir),
        "api_ingest_root": corpus,
    }
    if not args.live:
        update["llm_api_key"] = ""
        update["embedding_provider"] = "fake"
    return settings.model_copy(update=update)


def _sse(text: str) -> list[tuple[str, dict]]:
    frames: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event, data = "message", []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data.append(line.removeprefix("data:").strip())
        if not data:
            continue
        payload = json.loads("\n".join(data))
        if not isinstance(payload, dict):
            raise RuntimeError("SSE data must be a JSON object")
        frames.append((event, payload))
    return frames


def _retrieval_summary(report: dict) -> dict:
    strategies = []
    for strategy in report["strategies"]:
        strategies.append(
            {
                "chunker": strategy["chunker"],
                "gate_passed": strategy["gate_passed"],
                "recall@5": {name: route["recall@5"] for name, route in strategy["routes"].items()},
            }
        )
    return {
        "gate_passed": report["gate_passed"],
        "queries": report["queries"],
        "strategies": strategies,
    }


def _agent_summary(report: dict) -> dict:
    return {
        "gate_passed": report["gate_passed"],
        "adversarial": report["adversarial"],
        "answerable": report["answerable"],
        "abstain_rate": report["abstain_rate"],
        "answer_rate": report["answer_rate"],
        "budget_exhausted_rate": report["budget_exhausted_rate"],
        "gold_citation_rate": report["gold_citation_rate"],
        "fabricated_citations": report["fabricated_citations"],
    }


def _answer_summary(report: dict) -> dict:
    metrics = report["metrics"]
    return {
        "gate_passed": report["gate_passed"],
        "items": metrics["items"],
        "answer_rate": metrics["answer_rate"],
        "citation_pass_rate": metrics["citation_pass_rate"],
        "gold_citation_rate": metrics["gold_citation_rate"],
        "faithfulness": metrics["faithfulness"],
        "answer_relevance": metrics["answer_relevance"],
        "judge": report["judge"]["model"],
    }


async def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    await asyncio.to_thread(path.write_text, text, "utf-8")


async def run_demo(
    settings: Settings,
    *,
    corpus: Path,
    question: str,
    thread_id: str,
    eval_mode: str,
    judge_mode: str,
    output: Path,
) -> dict[str, Any]:
    corpus = corpus.resolve()
    if not corpus.is_dir():
        raise RuntimeError(f"corpus not found: {corpus}")
    if not question.strip():
        raise RuntimeError("question is empty")
    started = time.perf_counter()
    app = create_app(settings)
    ask: dict[str, Any] = {}
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://aether-demo") as client:
            ingested = await client.post(
                "/v1/ingest",
                json={"corpus_dir": str(corpus), "chunker": "markdown"},
            )
            if ingested.status_code != 200:
                raise RuntimeError(f"ingest failed: {ingested.status_code} {ingested.text[:240]}")
            chat = await client.post(
                "/v1/chat",
                json={"question": question, "thread_id": thread_id},
            )
            if chat.status_code != 200:
                raise RuntimeError(f"chat failed: {chat.status_code} {chat.text[:240]}")
            frames = _sse(chat.text)
            done = frames[-1][1]
            trace_id = chat.headers["x-aether-trace-id"]
            if done.get("trace_id") != trace_id:
                raise RuntimeError("SSE trace_id differs from X-Aether-Trace-Id")
            traced = await client.get(f"/v1/traces/{trace_id}")
            if traced.status_code != 200:
                raise RuntimeError(f"trace lookup failed: {traced.status_code}")
            trace = traced.json()
            if trace["slowest_stage"] != done["slowest_stage"]:
                raise RuntimeError("trace endpoint slowest_stage differs from SSE")
            stage_rows = [payload for name, payload in frames if name == "stage"]
            ask = {
                "thread_id": done["thread_id"],
                "trace_id": trace_id,
                "decision": done["decision"],
                "verified": done["verified"],
                "abstain_reason": done.get("abstain_reason") or "",
                "answer": done["answer"],
                "usage": done["usage"],
                "slowest_stage": trace["slowest_stage"],
                "slowest_elapsed_ms": trace["slowest_elapsed_ms"],
                "stages": [
                    {
                        "stage": row["stage"],
                        "elapsed_ms": row["elapsed_ms"],
                        "cost_usd": row["cost_usd"],
                    }
                    for row in stage_rows
                ],
            }
    sections: dict[str, Any] = {}
    modes = ["retrieval", "agent", "answer"] if eval_mode == "all" else []
    if eval_mode not in {"all", "skip"}:
        modes = [eval_mode]
    golden = corpus.parent
    if "retrieval" in modes:
        report = await evaluate(settings, corpus, golden / "retrieval.jsonl")
        await _write(settings.data_dir / "eval" / "retrieval.json", report)
        sections["retrieval"] = _retrieval_summary(report)
    if "agent" in modes:
        report = await evaluate_agent(
            settings,
            corpus,
            golden / "agent_adversarial.jsonl",
            golden / "agent_answerable.jsonl",
        )
        await _write(settings.data_dir / "eval" / "agent.json", report)
        sections["agent"] = _agent_summary(report)
    if "answer" in modes:
        judge = create_judge(settings, judge_mode)
        report = await evaluate_answers(
            settings,
            corpus,
            golden / "answer.jsonl",
            judge=judge,
            history_path=settings.data_dir / "eval" / "answer_history.jsonl",
        )
        await _write(settings.data_dir / "eval" / "answer.json", report)
        sections["answer"] = _answer_summary(report)
    summary = {
        "schema_version": 1,
        "mode": "live" if settings.llm_api_key else "offline",
        "llm_configured": bool(settings.llm_api_key),
        "embedding_provider": settings.embedding_provider,
        "corpus": _portable(corpus),
        "question": question,
        "ingest": ingested.json(),
        "ask": ask,
        "eval": sections,
        "gate_passed": all(item["gate_passed"] for item in sections.values()) if sections else True,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    await _write(output, summary)
    return summary


def _print(summary: dict, output: Path) -> None:
    ask = summary["ask"]
    print(
        f"mode={summary['mode']} embedding={summary['embedding_provider']} "
        f"llm_configured={str(summary['llm_configured']).lower()}"
    )
    ingest = summary["ingest"]
    print(f"ingest docs={ingest.get('docs')} chunks={ingest.get('chunks')}")
    print(
        f"ask decision={ask['decision']} verified={str(ask['verified']).lower()} "
        f"thread_id={ask['thread_id']} trace_id={ask['trace_id']}"
    )
    print(
        f"slowest_stage={ask['slowest_stage']} "
        f"slowest_elapsed_ms={ask['slowest_elapsed_ms']} "
        f"cost_usd={ask['usage']['cost_usd']} tokens={ask['usage']['total_tokens']}"
    )
    stages = " ".join(f"{row['stage']}={row['elapsed_ms']}" for row in ask["stages"])
    print(f"stages {stages}")
    print(f"answer {ask['answer']}")
    for name, section in summary["eval"].items():
        if name == "retrieval":
            for strategy in section["strategies"]:
                recall = strategy["recall@5"]
                print(
                    f"eval retrieval chunker={strategy['chunker']} "
                    f"hybrid_recall@5={recall['hybrid']:.4f} "
                    f"bm25_recall@5={recall['bm25']:.4f} "
                    f"dense_recall@5={recall['dense']:.4f}"
                )
        elif name == "agent":
            print(
                f"eval agent abstain_rate={section['abstain_rate']:.4f} "
                f"answer_rate={section['answer_rate']:.4f} "
                f"fabricated_citations={section['fabricated_citations']}"
            )
        else:
            print(
                f"eval answer faithfulness={section['faithfulness']:.4f} "
                f"answer_relevance={section['answer_relevance']:.4f} "
                f"citation_pass_rate={section['citation_pass_rate']:.4f}"
            )
        print(f"eval {name} gate={'PASS' if section['gate_passed'] else 'FAIL'}")
    if not summary["eval"]:
        print("eval skip")
    print(f"gate={'PASS' if summary['gate_passed'] else 'FAIL'} elapsed_s={summary['elapsed_s']}")
    print(f"report={output}")


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.live and not Settings(_env_file=None).llm_api_key:
        parser.exit(2, "aether-demo --live requires AETHER_LLM_API_KEY\n")
    settings = build_settings(args)
    if args.judge == "openai" and not settings.llm_api_key:
        parser.exit(2, "--judge openai requires AETHER_LLM_API_KEY\n")
    output = args.output or (settings.data_dir / "demo-report.json")
    configure_logging()
    try:
        with tracing_context(enabled=None if args.live else False):
            summary = asyncio.run(
                run_demo(
                    settings,
                    corpus=args.corpus,
                    question=args.question,
                    thread_id=args.thread_id,
                    eval_mode=args.eval,
                    judge_mode=args.judge,
                    output=output,
                )
            )
    except (JudgeError, RuntimeError, OSError, ValueError) as exc:
        parser.exit(2, f"demo failed: {exc}\n")
    _print(summary, output)
    if not summary["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
