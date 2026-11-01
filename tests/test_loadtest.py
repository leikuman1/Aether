"""k6 摘要解析与分位计算。不启动服务、不访问网络。"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from aether.api import loadtest
from aether.api.loadtest import load_k6_summary, parse_k6_summary, percentile


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([10, 20, 30, 40], 0.50) == 25
    assert percentile([10, 20, 30, 40], 0.95) == 38.5
    assert percentile([], 0.95) is None


def test_parse_k6_summary_exposes_p50_p95_and_error_rate() -> None:
    # 字段名与 k6 1.3 handleSummary 的 metrics.*.values 一致。
    # Rate.passes 是失败次数（http_req_failed 的非零样本）。
    summary = {
        "metrics": {
            "http_req_duration": {
                "values": {
                    "avg": 10.0,
                    "min": 1.0,
                    "med": 8.0,
                    "max": 30.0,
                    "p(50)": 8.0,
                    "p(95)": 20.0,
                }
            },
            "http_req_failed": {"values": {"rate": 0.25, "passes": 1, "fails": 3}},
            "http_reqs": {"values": {"count": 4, "rate": 1.0}},
            "checks": {"values": {"rate": 0.75, "passes": 3, "fails": 1}},
        }
    }
    parsed = parse_k6_summary(summary)
    assert parsed["requests"] == 4
    assert parsed["errors"] == 1
    assert parsed["error_rate"] == 0.25
    assert parsed["latency_ms"]["p50"] == 8.0
    assert parsed["latency_ms"]["p95"] == 20.0
    wrapped = load_k6_summary("Grafana banner\n" + json.dumps(summary) + "\nrunning\n")
    assert wrapped["metrics"]["http_reqs"]["values"]["count"] == 4


@pytest.mark.parametrize("exit_code", [0, 99])
async def test_load_report_requires_successful_driver_exit(monkeypatch, exit_code) -> None:
    process = Mock()
    process.stdout.readline = AsyncMock(return_value=b'{"port": 12345}\n')
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(loadtest, "_k6_binary", lambda _: "k6")
    monkeypatch.setattr(loadtest, "_ready", AsyncMock())
    monkeypatch.setattr(loadtest, "_stop", AsyncMock())
    monkeypatch.setattr(loadtest.httpx.AsyncClient, "post", AsyncMock(return_value=Mock()))
    monkeypatch.setattr(
        loadtest,
        "_run_k6",
        AsyncMock(
            return_value=(
                {"requests": 1, "errors": 0, "error_rate": 0.0, "exit_code": exit_code},
                "",
                exit_code,
            )
        ),
    )
    monkeypatch.setattr(
        loadtest,
        "_sample_trace",
        AsyncMock(return_value={"log_matches_endpoint": True, "slowest_stage": "verify"}),
    )
    report = await loadtest.run_load(
        concurrency=1,
        iterations=1,
        timeout=30,
        driver="k6",
        k6=None,
        question="question",
    )
    assert report["ok"] is (exit_code == 0)


@pytest.mark.parametrize("ok, expected_exit", [(True, 0), (False, 1)])
def test_load_cli_propagates_report_status(monkeypatch, capsys, ok, expected_exit) -> None:
    monkeypatch.setattr(loadtest, "run_load", AsyncMock(return_value={"ok": ok}))
    with pytest.raises(SystemExit) as caught:
        loadtest.main(["--driver", "builtin"])
    assert caught.value.code == expected_exit
    assert json.loads(capsys.readouterr().out)["ok"] is ok
