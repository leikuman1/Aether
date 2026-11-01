"""每请求结构化日志与进程内 trace 索引。

不引入 APM。每条日志是一行 JSON，字段含 trace_id、阶段时延和成本。
进程内环形缓冲只服务 GET /v1/traces/{trace_id}；重启后以日志为准。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections import OrderedDict
from decimal import Decimal
from typing import Any

TRACE_CAPACITY = 256
_SUMMARY_EVENTS = frozenset({"chat_complete", "request_complete", "http_access"})

logger = logging.getLogger(__name__)


def zero_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": "0",
        "source": "none",
    }


def usage_delta(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """两个累计 snapshot 之间的 token 与 USD。负差记 0，避免估算回退污染阶段成本。"""
    cost = Decimal(str(current["cost_usd"])) - Decimal(str(previous["cost_usd"]))
    if not cost.is_finite() or cost < 0:
        cost = Decimal(0)
    return {
        "input_tokens": max(0, int(current["input_tokens"]) - int(previous["input_tokens"])),
        "output_tokens": max(0, int(current["output_tokens"]) - int(previous["output_tokens"])),
        "cost_usd": format(cost, "f"),
    }


def annotate_stage(name: str, elapsed_ms: float, cost: dict[str, Any]) -> dict[str, Any]:
    elapsed = float(elapsed_ms)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed_ms must be finite and nonnegative")
    return {
        "stage": name,
        "elapsed_ms": round(elapsed, 3),
        "input_tokens": int(cost["input_tokens"]),
        "output_tokens": int(cost["output_tokens"]),
        "cost_usd": str(cost["cost_usd"]),
    }


def slowest_stage(stages: list[dict[str, Any]]) -> tuple[str | None, float]:
    """时延最大的阶段。并列时取更晚的一条，结果稳定。"""
    if not stages:
        return None, 0.0
    _index, stage = max(enumerate(stages), key=lambda pair: (pair[1]["elapsed_ms"], pair[0]))
    return str(stage["stage"]), float(stage["elapsed_ms"])


def make_trace(
    *,
    trace_id: str,
    method: str,
    path: str,
    status: str,
    stages: list[dict[str, Any]],
    usage: dict[str, Any],
    thread_id: str | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    name, elapsed = slowest_stage(stages)
    return {
        "trace_id": trace_id,
        "method": method,
        "path": path,
        "thread_id": thread_id,
        "status": status,
        "error_code": error_code,
        "stages": stages,
        "slowest_stage": name,
        "slowest_elapsed_ms": elapsed,
        "usage": usage,
    }


class TraceStore:
    """单进程 FIFO。容量满时丢掉最旧的 trace，不阻塞请求。"""

    def __init__(self, capacity: int = TRACE_CAPACITY):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._records: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def put(self, record: dict[str, Any]) -> None:
        trace_id = str(record["trace_id"])
        self._records.pop(trace_id, None)
        self._records[trace_id] = record
        while len(self._records) > self.capacity:
            self._records.popitem(last=False)

    def get(self, trace_id: str) -> dict[str, Any] | None:
        record = self._records.get(trace_id)
        if record is None:
            return None
        return {
            **record,
            "stages": [dict(stage) for stage in record["stages"]],
            "usage": dict(record["usage"]),
        }

    def __len__(self) -> int:
        return len(self._records)


def log_event(
    target: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    payload = {"event": event, **fields}
    target.log(level, "%s", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def configure_logging() -> None:
    """把 aether.* 打成一行一个 JSON 对象。重复调用不叠加 handler。"""
    target = logging.getLogger("aether")
    target.setLevel(logging.INFO)
    if not any(isinstance(handler, _JsonStreamHandler) for handler in target.handlers):
        target.addHandler(_JsonStreamHandler())
    target.propagate = False


class _JsonStreamHandler(logging.StreamHandler):
    """标记用的 stderr handler。消息本身已是 JSON，不再套 level 前缀。"""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter("%(message)s"))


def json_logs_enabled(settings_flag: bool) -> bool:
    raw = os.environ.get("AETHER_JSON_LOGS")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return settings_flag


class TraceMiddleware:
    """纯 ASGI：分配 trace_id、补响应头，不缓冲 SSE 正文。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        trace_id = _new_trace_id()
        state = scope.setdefault("state", {})
        if not isinstance(state, dict):
            state = {}
            scope["state"] = state
        state["trace_id"] = trace_id
        started = time.perf_counter()
        status_holder = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["code"] = int(message["status"])
                headers = message.setdefault("headers", [])
                if not any(key.lower() == b"x-aether-trace-id" for key, _value in headers):
                    headers.append((b"x-aether-trace-id", trace_id.encode("ascii")))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            record = _ensure_http_trace(scope, trace_id, status_holder["code"], elapsed_ms)
            usage = record["usage"] if record else zero_usage()
            log_event(
                logger,
                "http_access",
                trace_id=trace_id,
                method=scope.get("method"),
                path=scope.get("path"),
                status_code=status_holder["code"],
                elapsed_ms=elapsed_ms,
                cost_usd=usage["cost_usd"],
                total_tokens=usage["total_tokens"],
                usage_source=usage["source"],
                slowest_stage=None if record is None else record["slowest_stage"],
                slowest_elapsed_ms=0.0 if record is None else record["slowest_elapsed_ms"],
            )


def _new_trace_id() -> str:
    from uuid import uuid4

    return uuid4().hex


def _service(scope):
    app = scope.get("app")
    state = getattr(app, "state", None)
    return getattr(state, "service", None)


def _ensure_http_trace(scope, trace_id: str, status_code: int, elapsed_ms: float) -> dict | None:
    service = _service(scope)
    if service is None:
        return None
    existing = service.traces.get(trace_id)
    if existing is not None:
        return existing
    stage = annotate_stage(
        "http",
        elapsed_ms,
        {"input_tokens": 0, "output_tokens": 0, "cost_usd": "0"},
    )
    record = make_trace(
        trace_id=trace_id,
        method=str(scope.get("method") or ""),
        path=str(scope.get("path") or ""),
        status="success" if status_code < 400 else "error",
        stages=[stage],
        usage=zero_usage(),
        error_code=None if status_code < 400 else f"http_{status_code}",
    )
    service.traces.put(record)
    log_event(
        logger,
        "stage",
        trace_id=trace_id,
        stage=stage["stage"],
        elapsed_ms=stage["elapsed_ms"],
        input_tokens=0,
        output_tokens=0,
        cost_usd="0",
    )
    return record


def iter_log_events(text: str):
    for line in text.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            payload = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("trace_id") and payload.get("event"):
            yield payload


def slowest_from_log(text: str, trace_id: str) -> dict[str, Any]:
    """从 JSON 日志里按 trace_id 找出最慢阶段。有 stage 行时以它们为准。"""
    stages: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for payload in iter_log_events(text):
        if payload.get("trace_id") != trace_id:
            continue
        event = payload.get("event")
        if event == "stage":
            stages.append(
                {
                    "stage": payload["stage"],
                    "elapsed_ms": float(payload["elapsed_ms"]),
                    "input_tokens": int(payload.get("input_tokens") or 0),
                    "output_tokens": int(payload.get("output_tokens") or 0),
                    "cost_usd": str(payload.get("cost_usd", "0")),
                }
            )
        elif event in _SUMMARY_EVENTS:
            summaries.append(payload)
    if not stages and not summaries:
        raise KeyError(trace_id)
    if stages:
        name, elapsed = slowest_stage(stages)
    else:
        summary = summaries[-1]
        name = summary.get("slowest_stage")
        elapsed = float(summary.get("slowest_elapsed_ms") or 0)
    return {
        "trace_id": trace_id,
        "slowest_stage": name,
        "slowest_elapsed_ms": elapsed,
        "stages": stages,
        "cost_usd": _summary_cost(summaries, stages),
    }


def _summary_cost(summaries: list[dict[str, Any]], stages: list[dict[str, Any]]) -> str:
    for summary in reversed(summaries):
        if summary.get("cost_usd") is not None:
            return str(summary["cost_usd"])
    total = sum((Decimal(stage["cost_usd"]) for stage in stages), Decimal(0))
    return format(total, "f")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="从 JSON 日志按 trace_id 定位最慢阶段（不访问服务进程）。"
    )
    parser.add_argument("--log", required=True, help="aether JSON 日志文件")
    parser.add_argument("--trace-id", required=True)
    args = parser.parse_args(argv)
    from pathlib import Path

    path = Path(args.log)
    if not path.is_file():
        parser.exit(2, f"log not found: {path}\n")
    try:
        found = slowest_from_log(path.read_text(encoding="utf-8"), args.trace_id)
    except KeyError:
        parser.exit(1, f"trace not found: {args.trace_id}\n")
    print(json.dumps(found, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
