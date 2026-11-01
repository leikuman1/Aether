"""Offline SSE smoke over real TCP: python -m aether.api.smoke --concurrency 20."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import re
import socket
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

_QUESTION = "BM25 的 k1 和 b 分别控制什么，切块方式变化后要注意哪些统计量？"
_STAGES = ("retrieve", "grade", "generate", "verify")
_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+.*)?$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _cost(value: Any) -> Decimal:
    _check(isinstance(value, (str, int, float)) and not isinstance(value, bool), "invalid cost")
    result = Decimal(str(value))
    _check(result.is_finite() and result >= 0, "cost must be finite and nonnegative")
    return result


async def _events(response: httpx.Response) -> AsyncIterator[tuple[str, dict]]:
    """Handle split UTF-8/lines, CRLF, comments and multiline SSE data fields."""
    event, data = "message", []
    first = True
    async for line in response.aiter_lines():
        if first:
            line = line.removeprefix("﻿")
            first = False
        if not line:
            if data:
                payload = json.loads("\n".join(data))
                _check(isinstance(payload, dict), "SSE data must be a JSON object")
                yield event, payload
            event, data = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
        # id/retry/extension fields do not affect this one-shot consumer.
    if data:
        payload = json.loads("\n".join(data))
        _check(isinstance(payload, dict), "SSE data must be a JSON object")
        yield event, payload


def _validate_done(done: dict, headers: httpx.Headers, tokens: list[str]) -> dict:
    _check(done.get("decision") == "answer", "decision is not answer")
    _check(done.get("verified") is True, "answer was not verified")
    answer = done.get("answer")
    _check(isinstance(answer, str) and bool(answer), "answer is empty")
    _check(bool(tokens) and "".join(tokens) == answer, "token concatenation differs from answer")
    thread_id = headers.get("x-aether-thread-id", "")
    trace_id = headers.get("x-aether-trace-id", "")
    _check(bool(thread_id.strip()) and bool(trace_id.strip()), "missing thread/trace headers")
    _check(done.get("thread_id") == thread_id, "thread header differs from done")
    usage = done.get("usage")
    _check(isinstance(usage, dict), "done is missing usage")
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        _check(type(usage.get(key)) is int and usage[key] >= 0, f"invalid usage.{key}")
    _check(
        usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"],
        "usage token counts do not add up",
    )
    _check(
        isinstance(usage.get("source"), str) and bool(usage["source"].strip()),
        "usage source is missing",
    )
    _check(int(headers["x-aether-tokens"]) == usage["total_tokens"], "token header mismatch")
    cost = _cost(usage.get("cost_usd"))
    _check(_cost(headers["x-aether-cost-usd"]) == cost, "cost header mismatch")
    _check(cost == 0, "offline fake provider reported a nonzero cost")
    return {**usage, "cost_usd": str(cost)}


async def _request(
    client: httpx.AsyncClient, barrier: asyncio.Barrier, number: int, timeout: float
) -> dict:
    await barrier.wait()
    started = time.perf_counter()
    result: dict[str, Any] = {"request": number, "status": "error", "stages": []}
    try:
        async with asyncio.timeout(timeout):
            async with client.stream("POST", "/v1/chat", json={"question": _QUESTION}) as response:
                result["headers_ms"] = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                _check(
                    response.headers.get("content-type", "").split(";")[0].strip()
                    == "text/event-stream",
                    "response is not text/event-stream",
                )
                tokens: list[str] = []
                done = None
                async for event, payload in _events(response):
                    _check(done is None, "received event after done")
                    if event == "stage":
                        stage, elapsed = payload.get("stage"), payload.get("elapsed_ms")
                        _check(isinstance(stage, str) and bool(stage), "invalid stage name")
                        _check(
                            type(elapsed) in (int, float)
                            and math.isfinite(elapsed)
                            and elapsed >= 0,
                            "invalid stage elapsed_ms",
                        )
                        result["stages"].append(stage)
                    elif event == "token":
                        _check(isinstance(payload.get("text"), str), "invalid token text")
                        tokens.append(payload["text"])
                    elif event == "done":
                        done = payload
                    else:
                        raise ValueError(f"unexpected SSE event: {event}")
                _check(done is not None, "stream ended without done")
                position = -1
                for stage in _STAGES:
                    position = result["stages"].index(stage, position + 1)
                result["usage"] = _validate_done(done, response.headers, tokens)
                result["thread_id"] = response.headers["x-aether-thread-id"]
                result["trace_id"] = response.headers["x-aether-trace-id"]
                result["status"] = "success"
    except (TimeoutError, httpx.TimeoutException) as exc:
        result.update(status="timeout", error=f"{type(exc).__name__}: request exceeded timeout")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000
    return result


async def _metrics(client: httpx.AsyncClient, *, require_all: bool = False) -> dict[str, Decimal]:
    response = await client.get("/metrics")
    response.raise_for_status()
    counters = dict.fromkeys(("success", "error", "input", "output", "cost_usd"), Decimal(0))
    found: set[str] = set()
    for line in response.text.splitlines():
        match = _SAMPLE.fullmatch(line.strip())
        if not match:
            continue
        name, raw_labels, value = match.groups()
        labels = dict(_LABEL.findall(raw_labels or ""))
        key = None
        if name == "aether_chat_requests_total":
            key = labels.get("status")
            _check(key in ("success", "error"), "invalid chat metric status")
        elif name == "aether_tokens_total":
            key = labels.get("direction")
            _check(key in ("input", "output"), "invalid token metric direction")
        elif name == "aether_cost_usd_total":
            key = "cost_usd"
        if key is not None:
            counters[key] += _cost(value)
            found.add(key)
    if require_all:
        _check(
            {"success", "input", "output", "cost_usd"} <= found,
            "metrics response is missing required Aether counters",
        )
    return counters


def _accounting(before: dict, after: dict, results: list[dict]) -> dict:
    successes = [item for item in results if item["status"] == "success"]
    expected = {
        "success": Decimal(len(successes)),
        "error": Decimal(0),
        "input": Decimal(sum(item["usage"]["input_tokens"] for item in successes)),
        "output": Decimal(sum(item["usage"]["output_tokens"] for item in successes)),
        "cost_usd": sum((_cost(item["usage"]["cost_usd"]) for item in successes), Decimal(0)),
    }
    delta = {key: after[key] - before[key] for key in expected}
    _check(delta == expected, f"metrics accounting mismatch: observed={delta}, expected={expected}")
    return {key: str(value) for key, value in delta.items()}


async def _ready(client: httpx.AsyncClient, process: asyncio.subprocess.Process) -> None:
    while True:
        if process.returncode is not None:
            raise RuntimeError(f"server exited during startup ({process.returncode})")
        try:
            response = await client.get("/readyz", timeout=0.5)
            if response.is_success:
                return
        except httpx.TransportError:
            pass
        await asyncio.sleep(0.05)


async def _stop(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
    else:
        await process.wait()


def _serve() -> None:
    """Child entrypoint; cwd is empty and dotenv loading is explicitly disabled."""
    import uvicorn

    from aether.api.app import create_app
    from aether.config import Settings

    settings = Settings(_env_file=None)
    if settings.llm_api_key or settings.embedding_provider != "fake":
        raise RuntimeError("smoke server must use offline providers")
    app = create_app(settings)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(socket.SOMAXCONN)
        print(json.dumps({"port": listener.getsockname()[1]}), flush=True)
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="warning", access_log=False, timeout_graceful_shutdown=3)
        )
        server.run(sockets=[listener])


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)


async def _run(concurrency: int, timeout: float) -> dict:
    started = time.perf_counter()
    source_root = Path(__file__).resolve().parents[2]
    corpus = source_root.parent / "data" / "golden" / "corpus"
    report: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "llm_provider": "fake",
            "embedding_provider": "fake",
            "env_file": None,
            "corpus_dir": str(corpus),
        },
        "transport": "real_tcp",
        "concurrency": concurrency,
        "requests": concurrency,
        "completed": 0,
        "end_to_end_ms": None,
        "errors": [],
        "timeouts": 0,
        "results": [],
    }
    try:
        _check(corpus.is_dir(), f"golden corpus not found: {corpus}")
        with tempfile.TemporaryDirectory(prefix="aether-smoke-") as temporary:
            root = Path(temporary)
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.upper().startswith("AETHER_")
            }
            env.update(
                AETHER_LLM_API_KEY="",
                AETHER_LLM_BASE_URL="http://127.0.0.1:1/v1",
                AETHER_EMBEDDING_PROVIDER="fake",
                AETHER_DATA_DIR=str(root / "data"),
                AETHER_API_INGEST_ROOT=str(corpus),
                PYTHONPATH=str(source_root),
                PYTHONIOENCODING="utf-8",
                LANGSMITH_TRACING="false",
                LANGCHAIN_TRACING_V2="false",
                LANGCHAIN_TRACING="false",
            )
            with (root / "server.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "aether.api.smoke",
                    "--_serve",
                    cwd=root,
                    env=env,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=log,
                )
                try:
                    async with asyncio.timeout(timeout):
                        announcement = await process.stdout.readline()
                        _check(
                            bool(announcement), "server exited before announcing its loopback port"
                        )
                        port = json.loads(announcement)["port"]
                        _check(type(port) is int and 0 < port < 65536, "invalid server port")
                        report["base_url"] = f"http://127.0.0.1:{port}"
                        client = httpx.AsyncClient(
                            base_url=report["base_url"],
                            timeout=timeout,
                            trust_env=False,
                            limits=httpx.Limits(
                                max_connections=concurrency, max_keepalive_connections=concurrency
                            ),
                        )
                    async with client:
                        async with asyncio.timeout(timeout):
                            await _ready(client, process)
                        async with asyncio.timeout(timeout):
                            response = await client.post(
                                "/v1/ingest",
                                json={"corpus_dir": str(corpus), "chunker": "markdown"},
                            )
                            response.raise_for_status()
                            before = await _metrics(client)
                        barrier = asyncio.Barrier(concurrency)
                        batch_started = time.perf_counter()
                        results = await asyncio.gather(
                            *(
                                _request(client, barrier, number, timeout)
                                for number in range(concurrency)
                            )
                        )
                        report["results"] = results
                        report["end_to_end_ms"] = (time.perf_counter() - batch_started) * 1000
                        report["completed"] = sum(item["status"] == "success" for item in results)
                        report["timeouts"] = sum(item["status"] == "timeout" for item in results)
                        report["errors"].extend(
                            f"request {item['request']}: {item['error']}"
                            for item in results
                            if item["status"] == "error"
                        )
                        async with asyncio.timeout(timeout):
                            after = await _metrics(client, require_all=True)
                        report["metrics_delta"] = _accounting(before, after, results)
                        for filename in ("index.db", "sessions.sqlite"):
                            _check(
                                (root / "data" / filename).is_file(),
                                f"missing persisted {filename}",
                            )
                finally:
                    await _stop(process)
                    log.flush()
                    if report["completed"] != concurrency:
                        report["server_log"] = await asyncio.to_thread(
                            (root / "server.log").read_text, encoding="utf-8", errors="replace"
                        )
    except (TimeoutError, httpx.TimeoutException):
        report["timeouts"] += 1
        report["errors"].append("startup, ingest or metrics exceeded timeout")
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    results = report["results"]
    latencies = [item["elapsed_ms"] for item in results]
    report["latency_ms"] = {
        "p50": _percentile(latencies, 0.50),
        "p95": _percentile(latencies, 0.95),
        "max": max(latencies, default=None),
    }
    usages = [item["usage"] for item in results if item["status"] == "success"]
    report["tokens"] = {
        key: sum(usage[key] for usage in usages)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    report["cost_usd"] = str(sum((_cost(usage["cost_usd"]) for usage in usages), Decimal(0)))
    report["elapsed_ms"] = (time.perf_counter() - started) * 1000
    report["ok"] = (
        report["completed"] == concurrency and not report["errors"] and not report["timeouts"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="also write the JSON report to this file")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--timeout", type=float, default=30, help="seconds per startup/operation/request"
    )
    parser.add_argument("--_serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._serve:
        _serve()
        return
    if args.concurrency < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("concurrency and timeout must be positive and timeout must be finite")
    report = asyncio.run(_run(args.concurrency, args.timeout))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
