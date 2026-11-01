"""k6 摘要解析与分位计算。不启动服务、不访问网络。"""

import json

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
