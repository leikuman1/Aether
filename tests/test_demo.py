"""离线 aether-demo：不带密钥完成摄取、提问，并用 trace 标出最慢阶段。"""

import asyncio
from pathlib import Path

import pytest
from langsmith import tracing_context
from langsmith.utils import tracing_is_enabled

from aether import demo
from aether.demo import _parser, build_settings, main, run_demo
from aether.eval.judge import JudgeError

CORPUS = Path("data/golden/corpus")


def test_default_command_is_offline_retrieval() -> None:
    args = _parser().parse_args([])
    assert args.eval == "retrieval"
    assert args.live is False
    assert args.judge == "fake"


def test_live_flag_requires_an_api_key(tmp_path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--live", "--eval", "skip", "--data-dir", str(tmp_path)])
    assert caught.value.code == 2


def test_offline_demo_asks_and_names_the_slowest_stage(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AETHER_LLM_API_KEY", "secret-should-not-be-used")
    monkeypatch.setenv("AETHER_EMBEDDING_PROVIDER", "openai")
    args = _parser().parse_args(
        ["--eval", "skip", "--data-dir", str(tmp_path), "--corpus", str(CORPUS)]
    )
    settings = build_settings(args)
    assert settings.llm_api_key == ""
    assert settings.embedding_provider == "fake"
    summary = asyncio.run(
        run_demo(
            settings,
            corpus=args.corpus,
            question=args.question,
            thread_id="demo-test",
            eval_mode="skip",
            judge_mode="fake",
            output=tmp_path / "report.json",
        )
    )
    assert summary["mode"] == "offline"
    assert summary["llm_configured"] is False
    assert summary["gate_passed"] is True
    ask = summary["ask"]
    assert ask["decision"] == "answer"
    assert ask["verified"] is True
    assert ask["trace_id"]
    slowest = max(enumerate(ask["stages"]), key=lambda pair: (pair[1]["elapsed_ms"], pair[0]))[1]
    assert ask["slowest_stage"] == slowest["stage"]
    assert "secret-should-not-be-used" not in (tmp_path / "report.json").read_text(encoding="utf-8")


def test_offline_cli_disables_external_tracing(monkeypatch):
    async def inspect_tracing(*args, **kwargs):
        assert tracing_is_enabled() is False
        return {"gate_passed": True}

    monkeypatch.setattr(demo, "run_demo", inspect_tracing)
    monkeypatch.setattr(demo, "_print", lambda *args: None)
    with tracing_context(enabled=True):
        main([])
        assert tracing_is_enabled() is True


def test_demo_judge_protocol_error_exits_two(monkeypatch, capsys):
    async def failing_judge(*args, **kwargs):
        raise JudgeError("invalid judge verdict")

    monkeypatch.setattr(demo, "run_demo", failing_judge)
    with pytest.raises(SystemExit) as caught:
        main(["--eval", "answer"])
    assert caught.value.code == 2
    assert "demo failed: invalid judge verdict" in capsys.readouterr().err
