"""答案评测：确定性引用真实性（=100%）+ LLM-as-judge + 历史对比。

与 agent 评测的分工：agent 评测管「该不该答」（弃答语义），这里管「答得怎么样」——
引用必须存在于语料且属于本轮检索（纯确定性，无 LLM，必须 100%）；faithfulness 与
answer-relevance 由 judge 打分（固定 rubric、温度 0，CI 回放 fixtures）。报告写入
历史 JSONL，并与上一次运行对比关键指标。
"""

import asyncio
import hashlib
import json
import statistics
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from aether.config import Settings
from aether.eval.judge import BaseJudge, rubric_fingerprint
from aether.graph.citations import extract_chunk_ids, fabricated_citation_ids
from aether.graph.llm import create_chat_model
from aether.graph.runner import configure_searcher, run_turn
from aether.retrieval.chunking import MarkdownChunker, get_chunker
from aether.retrieval.embeddings import create_embedder
from aether.retrieval.ingest import ingest_directory, read_meta
from aether.retrieval.loaders import load_directory

MIN_ITEMS = 30
ANSWER_RATE_MIN = 0.80
CITATION_PASS_RATE_MIN = 1.0
GOLD_CITATION_RATE_MIN = 0.80
FAITHFULNESS_MIN = 0.80
RELEVANCE_MIN = 0.80
COMPARISON_METRICS = (
    "answer_rate",
    "citation_pass_rate",
    "gold_citation_rate",
    "faithfulness",
    "answer_relevance",
)


def load_answer_golden(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"empty or duplicate ids: {path}")
    for row in rows:
        if not str(row.get("query", "")).strip():
            raise ValueError(f"invalid answer golden row: {row.get('id')}")
        if not str(row.get("reference_answer", "")).strip():
            raise ValueError(f"missing reference answer: {row.get('id')}")
        gold = [str(item).lower() for item in row.get("relevant_chunk_ids") or []]
        if not gold or len(set(gold)) != len(gold):
            raise ValueError(f"invalid relevant_chunk_ids: {row.get('id')}")
        row["relevant_chunk_ids"] = gold
    return rows


def citation_flags(result: dict, canonical_ids: set[str]) -> dict:
    """确定性引用真实性：chunk 存在、属于本轮检索、与正文一致。不依赖任何模型。"""
    answer = result["answer"]
    citations = [str(item).lower() for item in result["citations"]]
    allowed = {item.lower() for item in result["retrieved_ids"]}
    fabricated = fabricated_citation_ids(answer, allowed)
    ids_in_answer = extract_chunk_ids(answer)
    citations_exist = set(citations) <= canonical_ids
    citations_retrieved = set(citations) <= allowed
    if result["decision"] == "answer":
        consistent = bool(citations) and ids_in_answer == citations
        citation_ok = not fabricated and citations_exist and citations_retrieved and consistent
    elif result["decision"] == "abstain":
        citation_ok = not fabricated and not ids_in_answer and not citations
    else:
        citation_ok = False
    return {
        "fabricated": fabricated,
        "citations_exist": citations_exist,
        "citations_retrieved": citations_retrieved,
        "citation_ok": citation_ok,
    }


def _item(row: dict, result: dict, canonical_ids: set[str]) -> dict:
    flags = citation_flags(result, canonical_ids)
    citations = [str(item).lower() for item in result["citations"]]
    return {
        "id": row["id"],
        "category": row.get("category", ""),
        "query": row["query"],
        "reference_answer": row["reference_answer"],
        "relevant_chunk_ids": row["relevant_chunk_ids"],
        "decision": result["decision"],
        "abstain_reason": result["abstain_reason"],
        "answer": result["answer"],
        "citations": citations,
        "retrieved_ids": [item.lower() for item in result["retrieved_ids"]],
        "gold_covered": set(row["relevant_chunk_ids"]) <= set(citations),
        "rewrite_count": result["rewrite_count"],
        "attempts": len(result["attempts"]),
        "grade": result["grade"],
        "faithfulness": None,
        "relevance": None,
    } | flags


async def _judge_one(judge: BaseJudge, item: dict, canonical_texts: dict[str, str]) -> None:
    if not item["citation_ok"]:
        verdict = {"score": 0.0, "reason": "deterministic citation validation failed"}
        item["faithfulness"] = dict(verdict)
        item["relevance"] = dict(verdict)
        return
    cited = [
        {"chunk_id": chunk_id, "text": canonical_texts[chunk_id]} for chunk_id in item["citations"]
    ]
    faithfulness, faith_reason = await judge.score(
        "faithfulness",
        {
            "question": item["query"],
            "answer": item["answer"],
            "citations": cited,
        },
    )
    relevance, relevance_reason = await judge.score(
        "relevance",
        {
            "question": item["query"],
            "answer": item["answer"],
            "reference_answer": item["reference_answer"],
        },
    )
    item["faithfulness"] = {"score": faithfulness, "reason": faith_reason}
    item["relevance"] = {"score": relevance, "reason": relevance_reason}


async def judge_items(
    items: list[dict], judge: BaseJudge, canonical_texts: dict[str, str], *, repeat: int = 1
) -> dict:
    """首轮打分写进 items；--repeat N 额外重跑 N-1 轮，只度量同输入波动。"""
    repeats = max(1, repeat)
    answered = [item for item in items if item["decision"] == "answer"]
    for item in answered:
        await _judge_one(judge, item, canonical_texts)
    deviations = {"faithfulness": 0.0, "answer_relevance": 0.0}
    for _ in range(repeats - 1):
        for item in answered:
            first_f = item["faithfulness"]["score"]
            first_r = item["relevance"]["score"]
            snapshot = (item["faithfulness"], item["relevance"])
            await _judge_one(judge, item, canonical_texts)
            deviations["faithfulness"] = max(
                deviations["faithfulness"], abs(item["faithfulness"]["score"] - first_f)
            )
            deviations["answer_relevance"] = max(
                deviations["answer_relevance"], abs(item["relevance"]["score"] - first_r)
            )
            item["faithfulness"], item["relevance"] = snapshot
    return {
        "repeats": repeats,
        "faithfulness_max_deviation": round(deviations["faithfulness"], 6),
        "answer_relevance_max_deviation": round(deviations["answer_relevance"], 6),
    }


def summarize(
    items: list[dict], *, meta: dict, settings: Settings, judge_name: str, stability: dict
) -> dict:
    answered = [item for item in items if item["decision"] == "answer"]
    judged = [item for item in answered if item["faithfulness"]]
    metrics = {
        "items": len(items),
        "answered": len(answered),
        "answer_rate": len(answered) / len(items) if items else 0.0,
        "citation_pass_rate": (
            statistics.mean(1.0 if item["citation_ok"] else 0.0 for item in items) if items else 0.0
        ),
        "fabricated_citations": sum(len(item["fabricated"]) for item in items),
        "gold_citation_rate": (
            statistics.mean(1.0 if item["gold_covered"] else 0.0 for item in answered)
            if answered
            else 0.0
        ),
        "faithfulness": statistics.mean(i["faithfulness"]["score"] for i in judged)
        if judged
        else 0.0,
        "answer_relevance": statistics.mean(i["relevance"]["score"] for i in judged)
        if judged
        else 0.0,
    }
    thresholds = {
        "min_items": MIN_ITEMS,
        "answer_rate": ANSWER_RATE_MIN,
        "citation_pass_rate": CITATION_PASS_RATE_MIN,
        "gold_citation_rate": GOLD_CITATION_RATE_MIN,
        "faithfulness": FAITHFULNESS_MIN,
        "answer_relevance": RELEVANCE_MIN,
    }
    gate_passed = (
        metrics["items"] >= MIN_ITEMS
        and metrics["answer_rate"] >= ANSWER_RATE_MIN
        and metrics["citation_pass_rate"] >= CITATION_PASS_RATE_MIN
        and metrics["gold_citation_rate"] >= GOLD_CITATION_RATE_MIN
        and metrics["faithfulness"] >= FAITHFULNESS_MIN
        and metrics["answer_relevance"] >= RELEVANCE_MIN
    )
    return {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedder": meta.get("embedder"),
        "corpus_fingerprint": meta.get("corpus_fingerprint"),
        "chunks": int(meta.get("num_chunks", 0)),
        "dense_backend": meta.get("dense_backend"),
        "rewrite_budget": settings.agent_rewrite_budget,
        "retrieval_k": settings.agent_retrieval_k,
        "judge": {
            "model": judge_name,
            "rubric_sha256": rubric_fingerprint(),
            "temperature": 0.0,
        },
        "metrics": metrics,
        "stability": stability,
        "thresholds": thresholds,
        "gate_passed": gate_passed,
        "items": items,
    }


def _history_summary(report: dict, golden_sha: str) -> dict:
    """历史行只留指标与指纹，不留逐条 items。"""
    return {
        "recorded_at": report["recorded_at"],
        "golden_sha256": golden_sha,
        "judge": report["judge"],
        "metrics": report["metrics"],
        "thresholds": report["thresholds"],
        "gate_passed": report["gate_passed"],
    }


def read_previous(history_path: Path | None, baseline_path: Path | None) -> dict | None:
    """对比基准优先用 --baseline，否则取历史最后一行。"""
    if baseline_path is not None:
        text = baseline_path.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"baseline is empty: {baseline_path}")
        source = "baseline"
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            row = json.loads(lines[-1])
    elif history_path is not None and history_path.is_file():
        lines = [
            line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if not lines:
            return None
        source = "history"
        row = json.loads(lines[-1])
    else:
        return None
    previous_metrics = {
        key: row["metrics"][key] for key in COMPARISON_METRICS if key in row.get("metrics", {})
    }
    return {
        "source": source,
        "recorded_at": row.get("recorded_at"),
        "metrics": previous_metrics,
    }


def build_comparison(report: dict, previous: dict | None) -> dict | None:
    if previous is None:
        return None
    current = report["metrics"]
    shared = [key for key in COMPARISON_METRICS if key in previous["metrics"]]
    return {
        "source": previous["source"],
        "previous_recorded_at": previous["recorded_at"],
        "previous": previous["metrics"],
        "delta": {key: round(current[key] - previous["metrics"][key], 6) for key in shared},
        "regressed": [key for key in shared if current[key] < previous["metrics"][key]],
    }


def append_history(history_path: Path, summary: dict) -> int:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    total = sum(1 for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


async def evaluate_answers(
    settings: Settings,
    corpus: Path,
    golden_path: Path,
    *,
    judge: BaseJudge,
    llm: BaseChatModel | None = None,
    repeat: int = 1,
    history_path: Path | None = None,
    baseline_path: Path | None = None,
) -> dict:
    rows = await asyncio.to_thread(load_answer_golden, golden_path)
    golden_sha = hashlib.sha256(
        (await asyncio.to_thread(golden_path.read_text, encoding="utf-8")).encode()
    ).hexdigest()
    docs = await load_directory(corpus)
    canonical = {chunk.chunk_id: chunk for doc in docs for chunk in MarkdownChunker().split(doc)}
    doc_texts = {doc.doc_id: doc.text for doc in docs}
    for row in rows:
        if not set(row["relevant_chunk_ids"]) <= canonical.keys():
            raise ValueError(f"gold chunk missing from corpus: {row['id']}")
        for evidence in row.get("evidence") or []:
            text = evidence.get("text", "")
            if text and text not in doc_texts.get(evidence.get("doc_id", ""), ""):
                raise ValueError(f"evidence not verbatim in corpus: {row['id']}")
    model = llm or create_chat_model(settings)
    embedder = create_embedder(settings)
    temporary = await asyncio.to_thread(tempfile.TemporaryDirectory)
    searcher = None
    items: list[dict] = []
    try:
        root = Path(temporary.name)
        searcher = await ingest_directory(
            corpus,
            get_chunker("markdown"),
            embedder,
            root / "index.db",
            dense_backend=settings.dense_backend,
        )
        configure_searcher(searcher, settings)
        meta = await asyncio.to_thread(read_meta, root / "index.db")
        for row in rows:
            result = await run_turn(
                question=row["query"],
                searcher=searcher,
                llm=model,
                checkpoint_path=root / "sessions.sqlite",
                thread_id=f"answer-{row['id']}",
                rewrite_budget=settings.agent_rewrite_budget,
                min_support_terms=settings.agent_min_support_terms,
                min_quote_chars=settings.agent_min_quote_chars,
                retrieval_k=settings.agent_retrieval_k,
            )
            items.append(_item(row, result, set(canonical)))
    finally:
        try:
            if searcher is not None:
                await searcher.aclose()
        finally:
            try:
                if hasattr(embedder, "aclose"):
                    await embedder.aclose()
            finally:
                await asyncio.to_thread(temporary.cleanup)
    canonical_texts = {chunk_id: chunk.text for chunk_id, chunk in canonical.items()}
    stability = await judge_items(items, judge, canonical_texts, repeat=repeat)
    report = summarize(
        items, meta=meta, settings=settings, judge_name=judge.name, stability=stability
    )
    report["golden_sha256"] = golden_sha
    previous = await asyncio.to_thread(read_previous, history_path, baseline_path)
    report["comparison"] = build_comparison(report, previous)
    report["history"] = {"path": str(history_path) if history_path else None, "runs": None}
    if history_path is not None:
        runs = await asyncio.to_thread(
            append_history, history_path, _history_summary(report, golden_sha)
        )
        report["history"]["runs"] = runs
    return report
