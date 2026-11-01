"""离线聊天压测：优先 k6（每轮独立 thread_id），没有 k6 时用内置 httpx。

测量并发、p50/p95 与错误率。默认 Fake 模型，拒绝密钥和外部 embedding。
hey 不适合：它不能为每次请求生成不同的 thread_id，第二条会 409。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import shutil
import socket
import sys
import tempfile
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from aether.api.observe import slowest_from_log

_SOURCE = Path(__file__).resolve().parents[2]
_REPO = _SOURCE.parent
_SCRIPT = _REPO / "scripts" / "k6" / "chat.js"
_QUESTION = "BM25 的 k1 和 b 分别控制什么，切块方式变化后要注意哪些统计量？"


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_k6_summary(text: str) -> dict[str, Any]:
    """k6 仍会先把启动横幅打到 stdout；摘要是其中一行 JSON。"""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "metrics" in payload:
            return payload
    raise ValueError("k6 summary JSON not found")


def parse_k6_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """读取 k6 handleSummary 的 metrics。Rate.passes 是失败请求数。"""
    metrics = summary["metrics"]
    duration = metrics["http_req_duration"]["values"]
    failed = metrics["http_req_failed"]["values"]
    requests = int(metrics["http_reqs"]["values"]["count"])
    error_rate = float(failed["rate"])
    errors = int(failed["passes"]) if "passes" in failed else round(error_rate * requests)
    p50 = duration.get("p(50)", duration["med"])
    return {
        "requests": requests,
        "errors": errors,
        "error_rate": error_rate,
        "latency_ms": {
            "min": float(duration["min"]),
            "avg": float(duration["avg"]),
            "p50": float(p50),
            "p95": float(duration["p(95)"]),
            "max": float(duration["max"]),
        },
        "k6_metrics": {
            "http_req_duration": duration,
            "http_req_failed": failed,
            "http_reqs": metrics["http_reqs"]["values"],
            "checks": (metrics.get("checks") or {}).get("values"),
        },
    }


def _k6_binary(explicit: str | None) -> str | None:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(explicit)
        return found
    return shutil.which("k6")


def _serve() -> None:
    import uvicorn

    from aether.api.app import create_app
    from aether.api.observe import configure_logging
    from aether.config import Settings

    os.environ["AETHER_JSON_LOGS"] = "1"
    settings = Settings(_env_file=None)
    if settings.llm_api_key or settings.embedding_provider != "fake":
        raise RuntimeError("load test server must use offline providers")
    app = create_app(settings)
    configure_logging()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(socket.SOMAXCONN)
        print(json.dumps({"port": listener.getsockname()[1]}), flush=True)
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="warning", access_log=False, timeout_graceful_shutdown=3)
        )
        server.run(sockets=[listener])


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


def _child_env(data_dir: Path, corpus: Path, *, concurrency: int, timeout: float) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("AETHER_")}
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env.update(
        AETHER_LLM_API_KEY="",
        AETHER_LLM_BASE_URL="http://127.0.0.1:1/v1",
        AETHER_EMBEDDING_PROVIDER="fake",
        AETHER_JSON_LOGS="1",
        AETHER_DATA_DIR=str(data_dir),
        AETHER_API_INGEST_ROOT=str(corpus),
        AETHER_API_MAX_CONCURRENCY=str(max(concurrency, 1)),
        AETHER_API_REQUEST_TIMEOUT=str(timeout),
        PYTHONPATH=str(_SOURCE),
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
        LANGCHAIN_TRACING="false",
    )
    return env


async def _sample_trace(client: httpx.AsyncClient, question: str, log_path: Path) -> dict[str, Any]:
    response = await client.post(
        "/v1/chat",
        json={"question": question, "thread_id": "load-sample"},
    )
    response.raise_for_status()
    trace_id = response.headers["x-aether-trace-id"]
    traced = await client.get(f"/v1/traces/{trace_id}")
    traced.raise_for_status()
    body = traced.json()
    # 子进程 stderr 可能刚 flush；短等再读，避免样本行还在块缓冲里。
    await asyncio.sleep(0.05)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    from_log = slowest_from_log(text, trace_id)
    return {
        "trace_id": trace_id,
        "slowest_stage": body["slowest_stage"],
        "slowest_elapsed_ms": body["slowest_elapsed_ms"],
        "cost_usd": body["usage"]["cost_usd"],
        "log_slowest_stage": from_log["slowest_stage"],
        "log_slowest_elapsed_ms": from_log["slowest_elapsed_ms"],
        "log_matches_endpoint": from_log["slowest_stage"] == body["slowest_stage"],
    }


async def _run_k6(
    binary: str,
    *,
    base_url: str,
    concurrency: int,
    iterations: int,
    timeout: float,
    question: str,
) -> tuple[dict[str, Any], str, int]:
    env = _child_env(Path("."), Path("."), concurrency=concurrency, timeout=timeout)
    descriptor, summary_name = tempfile.mkstemp(prefix="aether-k6-", suffix=".json")
    os.close(descriptor)
    summary_path = Path(summary_name)
    env.update(
        AETHER_BASE_URL=base_url,
        AETHER_VUS=str(concurrency),
        AETHER_ITERATIONS=str(iterations),
        AETHER_MAX_DURATION="180s",
        AETHER_TIMEOUT=f"{timeout}s",
        AETHER_QUESTION=question,
        AETHER_K6_SUMMARY=str(summary_path),
    )
    process = await asyncio.create_subprocess_exec(
        binary,
        "run",
        str(_SCRIPT),
        "--summary-trend-stats",
        "avg,min,med,max,p(50),p(90),p(95),p(99)",
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    stderr_text = stderr.decode("utf-8", errors="replace")
    try:
        if summary_path.is_file() and summary_path.stat().st_size:
            text = summary_path.read_text(encoding="utf-8")
        else:
            text = stdout.decode("utf-8", errors="replace")
    finally:
        summary_path.unlink(missing_ok=True)
    version = await asyncio.create_subprocess_exec(
        binary,
        "version",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    version_out, _ = await version.communicate()
    if process.returncode != 0 and '"metrics"' not in text:
        raise RuntimeError(stderr_text[-2000:] or "k6 exited without a summary")
    try:
        summary = load_k6_summary(text)
    except ValueError as exc:
        raise RuntimeError(f"{exc}: {stderr_text[-1000:]}") from exc
    parsed = parse_k6_summary(summary)
    parsed["tool_version"] = version_out.decode("utf-8", errors="replace").strip()
    parsed["exit_code"] = process.returncode
    return parsed, stderr_text[-2000:], process.returncode or 0


async def _run_builtin(
    client: httpx.AsyncClient,
    *,
    concurrency: int,
    iterations: int,
    question: str,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    async def one(number: int) -> None:
        started = time.perf_counter()
        status = "error"
        try:
            response = await client.post(
                "/v1/chat",
                json={"question": question, "thread_id": f"builtin-{number}"},
            )
            status = "success" if response.status_code == 200 else "error"
            code = response.status_code
        except httpx.HTTPError:
            code = 0
        results.append(
            {
                "status": status,
                "status_code": code,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
        )

    # 按并发窗口发送，避免一次堆出超过服务容量的连接。
    for offset in range(0, iterations, concurrency):
        batch = range(offset, min(offset + concurrency, iterations))
        await asyncio.gather(*(one(number) for number in batch))
    latencies = [item["elapsed_ms"] for item in results]
    errors = sum(item["status"] != "success" for item in results)
    count = len(results)
    return {
        "requests": count,
        "errors": errors,
        "error_rate": (errors / count) if count else 1.0,
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "avg": (sum(latencies) / count) if count else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "tool_version": f"python {platform.python_version()}",
        "exit_code": 0 if errors == 0 else 1,
    }


async def run_load(
    *,
    concurrency: int,
    iterations: int,
    timeout: float,
    driver: str,
    k6: str | None,
    question: str,
) -> dict[str, Any]:
    corpus = _REPO / "data" / "golden" / "corpus"
    if not corpus.is_dir():
        raise RuntimeError(f"golden corpus not found: {corpus}")
    if not _SCRIPT.is_file():
        raise RuntimeError(f"k6 script not found: {_SCRIPT}")
    binary = _k6_binary(k6)
    chosen = driver
    if driver == "auto":
        chosen = "k6" if binary else "builtin"
    if chosen == "k6" and not binary:
        raise RuntimeError("k6 binary not found")
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "tool": chosen,
        "concurrency": concurrency,
        "iterations": iterations,
        "question": question,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "llm_provider": "fake",
            "embedding_provider": "fake",
        },
        "errors_detail": [],
        "sample_trace": None,
        "ok": False,
    }
    with tempfile.TemporaryDirectory(prefix="aether-load-") as temporary:
        root = Path(temporary)
        log_path = root / "server.log"
        env = _child_env(root / "data", corpus, concurrency=concurrency, timeout=timeout)
        with log_path.open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "aether.api.loadtest",
                "--_serve",
                cwd=root,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
            )
            try:
                assert process.stdout is not None
                announcement = await asyncio.wait_for(process.stdout.readline(), timeout=timeout)
                if not announcement:
                    raise RuntimeError("server exited before announcing its port")
                port = json.loads(announcement)["port"]
                base_url = f"http://127.0.0.1:{port}"
                report["base_url"] = base_url
                async with httpx.AsyncClient(
                    base_url=base_url, timeout=timeout, trust_env=False
                ) as client:
                    await asyncio.wait_for(_ready(client, process), timeout=timeout)
                    ingest = await client.post(
                        "/v1/ingest",
                        json={"corpus_dir": str(corpus), "chunker": "markdown"},
                    )
                    ingest.raise_for_status()
                    if chosen == "k6":
                        parsed, k6_stderr, _code = await _run_k6(
                            binary or "k6",
                            base_url=base_url,
                            concurrency=concurrency,
                            iterations=iterations,
                            timeout=timeout,
                            question=question,
                        )
                        if k6_stderr.strip():
                            report["k6_stderr_tail"] = k6_stderr
                    else:
                        parsed = await _run_builtin(
                            client,
                            concurrency=concurrency,
                            iterations=iterations,
                            question=question,
                        )
                    report.update(parsed)
                    report["sample_trace"] = await _sample_trace(client, question, log_path)
            finally:
                await _stop(process)
    sample = report["sample_trace"] or {}
    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    report["ok"] = (
        report.get("exit_code") == 0
        and report.get("errors") == 0
        and report.get("requests") == iterations
        and bool(sample.get("log_matches_endpoint"))
        and bool(sample.get("slowest_stage"))
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--driver", choices=["auto", "k6", "builtin"], default="auto")
    parser.add_argument("--k6", default=None, help="k6 可执行文件路径")
    parser.add_argument("--question", default=_QUESTION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._serve:
        _serve()
        return
    if (
        args.concurrency < 1
        or args.iterations < 1
        or args.timeout <= 0
        or not math.isfinite(args.timeout)
    ):
        parser.error("concurrency, iterations and timeout must be positive")
    if args.concurrency > 512:
        parser.error("concurrency must be <= 512")
    try:
        report = asyncio.run(
            run_load(
                concurrency=args.concurrency,
                iterations=args.iterations,
                timeout=args.timeout,
                driver=args.driver,
                k6=args.k6,
                question=args.question,
            )
        )
    except (RuntimeError, OSError, json.JSONDecodeError, KeyError, httpx.HTTPError) as exc:
        parser.exit(2, f"load test failed: {exc}\n")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
