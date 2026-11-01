"""M4 答案门禁、协议错误、无网络 fixtures 与版本对比。"""

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from aether.config import Settings
from aether.eval.answer import (
    FAITHFULNESS_MIN,
    RELEVANCE_MIN,
    citation_flags,
    evaluate_answers,
    load_answer_golden,
)
from aether.eval.judge import (
    JudgeError,
    LexicalJudge,
    LLMJudge,
    RecordingJudge,
    ReplayingJudge,
    create_judge,
    parse_verdict,
    prompt_key,
    render_judge_prompt,
    rubric_fingerprint,
)
from aether.graph.citations import extract_chunk_ids
from aether.graph.llm import EvidenceBoundChatModel
from aether.retrieval.chunking import MarkdownChunker
from aether.retrieval.loaders import load_directory

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/golden/corpus"
GOLDEN = ROOT / "data/golden/answer.jsonl"
FIXTURES = ROOT / "data/golden/answer_judge_fixtures.jsonl"


async def test_answer_golden_integrity() -> None:
    docs = await load_directory(CORPUS)
    chunks = {c.chunk_id: c for doc in docs for c in MarkdownChunker().split(doc)}
    doc_texts = {doc.doc_id: doc.text for doc in docs}
    rows = load_answer_golden(GOLDEN)
    assert len(rows) >= 30
    categories: dict[str, int] = {}
    for row in rows:
        assert set(row["relevant_chunk_ids"]) <= chunks.keys()
        categories[row["category"]] = categories.get(row["category"], 0) + 1
        assert row["reference_answer"] == "\n".join(e["text"] for e in row["evidence"])
        for evidence in row["evidence"]:
            assert evidence["text"] in doc_texts[evidence["doc_id"]]
    assert len(categories) == 4 and min(categories.values()) >= 5


async def test_committed_fixtures_gate_and_stability(settings: Settings) -> None:
    report = await evaluate_answers(
        settings, CORPUS, GOLDEN, judge=ReplayingJudge(FIXTURES), repeat=3
    )
    metrics = report["metrics"]
    assert metrics["items"] >= 30
    assert metrics["answer_rate"] == 1.0
    assert metrics["citation_pass_rate"] == 1.0
    assert metrics["fabricated_citations"] == 0
    assert metrics["gold_citation_rate"] == 1.0
    assert metrics["faithfulness"] >= FAITHFULNESS_MIN
    assert metrics["answer_relevance"] >= RELEVANCE_MIN
    assert report["gate_passed"] is True
    assert report["stability"] == {
        "repeats": 3,
        "faithfulness_max_deviation": 0.0,
        "answer_relevance_max_deviation": 0.0,
    }
    for item in report["items"]:
        assert item["citation_ok"] is True
        assert set(item["citations"]) <= set(item["retrieved_ids"])
        assert extract_chunk_ids(item["answer"]) == item["citations"]


@pytest.mark.parametrize(
    ("cited", "retrieved", "canonical", "decision", "expected"),
    [
        ("a", "a", "a", "answer", True),
        ("c", "a", "a", "answer", False),
        ("b", "a", "ab", "answer", False),
        ("a", "a", "b", "answer", False),
        ("a", "a", "a", "abstain", False),
    ],
)
def test_citations_must_exist_and_be_retrieved(cited, retrieved, canonical, decision, expected):
    result = {
        "decision": decision,
        "answer": f"span [{cited * 20}]",
        "citations": [cited.upper() * 20],
        "retrieved_ids": [retrieved.upper() * 20],
    }
    assert citation_flags(result, {c * 20 for c in canonical})["citation_ok"] is expected
    result["citations"] = []
    assert citation_flags(result, {c * 20 for c in canonical})["citation_ok"] is False


async def test_lexical_judge_rejects_hallucination_and_irrelevance() -> None:
    judge = LexicalJudge()
    text = "BM25 controls term saturation and document length normalization."
    payload = {
        "question": "What does BM25 control?",
        "answer": f"{text} [{'a' * 20}]",
        "citations": [{"chunk_id": "a" * 20, "text": text}],
    }
    assert (await judge.score("faithfulness", payload))[0] == 1.0
    payload["answer"] += "\nThis system launched in 2019."
    assert (await judge.score("faithfulness", payload))[0] < FAITHFULNESS_MIN
    assert (
        await judge.score(
            "relevance",
            {
                "question": "What does BM25 control?",
                "answer": "Photosynthesis converts sunlight into chemical energy.",
                "reference_answer": text,
            },
        )
    )[0] < RELEVANCE_MIN


async def test_real_judge_protocol_recording_and_replay(tmp_path: Path) -> None:
    # 真实 LLMJudge 路径使用 FakeListChatModel，验证固定 prompt、解析和回放。
    llm = FakeListChatModel(responses=['{"score": 0.25, "reason": "unsupported"}'])
    recorder = RecordingJudge(LLMJudge(llm, "fixture-model"))
    payload = {"question": "q", "answer": "a", "reference_answer": "r"}
    verdict = await recorder.score("relevance", payload)
    assert verdict == (0.25, "unsupported")
    # 稳定性实验允许后续响应变化，但录制必须保留报告使用的首轮响应。
    llm.responses = ['{"score": 0.9, "reason": "changed"}']
    assert (await recorder.score("relevance", payload))[0] == 0.9
    fixtures = tmp_path / "judge.jsonl"
    fixtures.write_text("\n".join(recorder.fixture_lines()) + "\n", encoding="utf-8")
    replay = ReplayingJudge(fixtures)
    assert await replay.score("relevance", payload) == verdict
    with pytest.raises(JudgeError, match="missing judge fixture"):
        await replay.score("relevance", dict(payload, answer="changed"))
    key = prompt_key(render_judge_prompt("relevance", payload))
    fixtures.write_text(json.dumps({"key": key, "response": "not json"}), encoding="utf-8")
    with pytest.raises(JudgeError):
        await ReplayingJudge(fixtures).score("relevance", payload)
    fixtures.write_text(
        "\n".join(json.dumps({"key": key, "response": response}) for response in ["x", "y"]),
        encoding="utf-8",
    )
    with pytest.raises(JudgeError, match="conflicting"):
        ReplayingJudge(fixtures)


def test_judge_zero_temperature_and_pinned_rubric(settings: Settings) -> None:
    with pytest.raises(JudgeError, match="API_KEY"):
        create_judge(settings, "openai")
    live = create_judge(
        settings.model_copy(update={"llm_api_key": "sk-test", "llm_temperature": 0.9}), "openai"
    )
    assert live.llm.temperature == 0.0
    pinned = "a1c9608196fa7cc17d141ca4ac9bd911f4b1b95d3d9a317f4ecb4e3d08c3b11d"
    assert rubric_fingerprint() == pinned


@pytest.mark.parametrize("score", ["NaN", "Infinity", "-Infinity", "true", '"0.9"', "1.5", "-0.1"])
def test_invalid_judge_scores_fail_closed(score):
    with pytest.raises(JudgeError):
        parse_verdict(f'{{"score": {score}, "reason": "invalid"}}')


def test_cli_report_history_replay_and_low_scores(tmp_path, capsys, monkeypatch):
    from aether.eval.cli import main

    output = tmp_path / "answer.json"
    history = tmp_path / "history.jsonl"
    common = ["aether-eval", "answer", "--gate", "--output", str(output), "--history", str(history)]
    monkeypatch.setattr("sys.argv", [*common, "--judge", "replay"])
    # 回放必须忽略本机已有 provider 配置；网络拦截夹具会抓到任何意外调用。
    monkeypatch.setenv("AETHER_LLM_API_KEY", "offline-must-ignore")
    monkeypatch.setenv("AETHER_EMBEDDING_PROVIDER", "openai")
    main()
    assert "gate=PASS" in capsys.readouterr().out
    first = json.loads(output.read_text(encoding="utf-8"))
    assert first["comparison"] is None
    main()
    second = json.loads(output.read_text(encoding="utf-8"))
    assert second["history"]["runs"] == 2
    assert set(second["comparison"]["delta"].values()) == {0.0}
    assert "comparison: history@" in capsys.readouterr().out
    # 直接降低录制分数：输入、引用、样本数不变，必须因为质量阈值退出 1。
    low = tmp_path / "low.jsonl"
    rows = [json.loads(line) for line in FIXTURES.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["response"] = json.dumps({"score": 0.25, "reason": "regression"})
    low.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", [*common, "--judge", "replay", "--fixtures", str(low)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    failed = json.loads(output.read_text(encoding="utf-8"))
    assert failed["metrics"]["citation_pass_rate"] == 1.0
    assert failed["metrics"]["faithfulness"] == 0.25
    assert failed["metrics"]["answer_relevance"] == 0.25
    assert set(failed["comparison"]["regressed"]) == {"faithfulness", "answer_relevance"}
    assert "gate=FAIL" in capsys.readouterr().out
    # 缺失 fixture -> 协议错误退出 2，不会静默回退 fake。
    low.write_text("\n".join(json.dumps(row) for row in rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "missing judge fixture" in capsys.readouterr().err


async def test_invalid_citation_produces_failing_report(settings, monkeypatch):
    import aether.eval.answer as answer_eval

    real_run = answer_eval.run_turn

    async def corrupt_result(**kwargs):
        result = await real_run(**kwargs)
        result["answer"] = f"fabricated [{'f' * 20}]"
        result["citations"] = ["f" * 20]
        return result

    monkeypatch.setattr(answer_eval, "run_turn", corrupt_result)
    report = await evaluate_answers(
        settings, CORPUS, GOLDEN, judge=LexicalJudge(), llm=EvidenceBoundChatModel()
    )
    assert report["gate_passed"] is False
    assert report["metrics"]["citation_pass_rate"] == 0.0
    assert report["metrics"]["faithfulness"] == 0.0


async def test_baseline_override_and_stale_gold(tmp_path, settings):
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(
        json.dumps(
            {"recorded_at": "prior-version", "metrics": {"answer_relevance": 1.0}}, indent=2
        ),
        encoding="utf-8",
    )
    report = await evaluate_answers(
        settings, CORPUS, GOLDEN, judge=ReplayingJudge(FIXTURES), baseline_path=baseline
    )
    assert report["comparison"]["source"] == "baseline"
    assert report["comparison"]["delta"]["answer_relevance"] < 0
    rows = load_answer_golden(GOLDEN)
    rows[0]["relevant_chunk_ids"] = ["f" * 20]
    stale = tmp_path / "stale.jsonl"
    stale.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="gold chunk missing"):
        await evaluate_answers(settings, CORPUS, stale, judge=LexicalJudge())
