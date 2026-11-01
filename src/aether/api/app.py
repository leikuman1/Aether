"""异步服务：受引用门保护的 SSE、SqliteSaver 会话和请求级模型核算。"""

import asyncio
import json
import logging
import os
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aether.api.accounting import Metrics, RequestUsage
from aether.api.observe import (
    TraceMiddleware,
    TraceStore,
    annotate_stage,
    json_logs_enabled,
    log_event,
    make_trace,
    usage_delta,
    zero_usage,
)
from aether.api.observe import (
    configure_logging as configure_json_logging,
)
from aether.config import Settings, get_settings
from aether.graph.llm import create_chat_model
from aether.graph.runner import open_index, read_thread, run_turn
from aether.retrieval.chunking import FixedWindowChunker, MarkdownChunker
from aether.retrieval.embeddings import EmbeddingError, create_embedder
from aether.retrieval.ingest import index_info, ingest_directory

logger = logging.getLogger(__name__)
ThreadId = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,128}$")]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str | None = Field(default=None, min_length=1, max_length=16000)
    thread_id: ThreadId = Field(default_factory=lambda: uuid4().hex)
    require_review: bool = False
    resume: "ReviewDecision | None" = None

    @model_validator(mode="after")
    def validate_turn(self):
        if (self.question is None) == (self.resume is None):
            raise ValueError("provide exactly one of question or resume")
        if self.resume is not None and self.require_review:
            raise ValueError("require_review applies only to a new question")
        return self


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool = Field(strict=True)


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus_dir: str = Field(min_length=1, max_length=4096)
    chunker: Literal["markdown", "fixed"] = "markdown"


class ServiceError(Exception):
    def __init__(self, status: int, code: str):
        self.status = status
        self.code = code


def _check_storage(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path) as probe:
        probe.write(b"ready")
        probe.flush()


def _corpus_path(body: IngestRequest, settings: Settings) -> Path:
    root = (settings.api_ingest_root or settings.data_dir / "corpus").resolve()
    corpus = Path(body.corpus_dir).resolve()
    if not corpus.is_relative_to(root):
        raise ServiceError(422, "corpus_outside_ingest_root")
    if not corpus.is_dir():
        raise ServiceError(422, "corpus_not_found")
    # load_directory 跟随文件符号链接，限制所有实际来源也在允许的根目录内。
    if any(not item.resolve().is_relative_to(root) for item in corpus.rglob("*")):
        raise ServiceError(422, "corpus_outside_ingest_root")
    return corpus


class Service:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = create_chat_model(settings)
        self.metrics = Metrics()
        self.traces = TraceStore()
        self.active_threads: set[str] = set()
        self.ingesting = False

    @property
    def index(self) -> Path:
        return self.settings.data_dir / "index.db"

    @property
    def checkpoint(self) -> Path:
        return self.settings.data_dir / "sessions.sqlite"

    @asynccontextmanager
    async def session(self, thread_id: str):
        # 检查与占用之间无 await，单事件循环内原子；SQLite 的多进程写协调不由此承担。
        if self.ingesting:
            raise ServiceError(503, "index_updating")
        if thread_id in self.active_threads:
            raise ServiceError(409, "thread_busy")
        if len(self.active_threads) >= self.settings.api_max_concurrency:
            raise ServiceError(429, "service_busy")
        self.active_threads.add(thread_id)
        embedder = searcher = None
        try:
            if not await asyncio.to_thread(self.index.is_file):
                raise ServiceError(503, "index_not_ready")
            embedder, searcher = await open_index(self.settings, self.index)
            yield searcher
        finally:
            try:
                if searcher is not None:
                    await searcher.aclose()
                if embedder is not None and hasattr(embedder, "aclose"):
                    await embedder.aclose()
            finally:
                self.active_threads.discard(thread_id)

    async def ingest(self, body: IngestRequest) -> dict:
        if self.ingesting or self.active_threads:
            raise ServiceError(409, "index_busy")
        self.ingesting = True
        embedder = searcher = None
        try:
            corpus = await asyncio.to_thread(_corpus_path, body, self.settings)
            embedder = create_embedder(self.settings)
            chunker = MarkdownChunker() if body.chunker == "markdown" else FixedWindowChunker()
            searcher = await ingest_directory(
                corpus, chunker, embedder, self.index, dense_backend=self.settings.dense_backend
            )
            return await asyncio.to_thread(index_info, self.index)
        except EmbeddingError as exc:
            raise ServiceError(503, "provider_unavailable") from exc
        except ValueError as exc:
            raise ServiceError(422, "invalid_corpus") from exc
        finally:
            try:
                if searcher is not None:
                    await searcher.aclose()
                if embedder is not None and hasattr(embedder, "aclose"):
                    await embedder.aclose()
            finally:
                self.ingesting = False


def _headers(usage: dict, trace_id: str, thread_id: str) -> dict[str, str]:
    return {
        "X-Aether-Tokens": str(usage["total_tokens"]),
        "X-Aether-Cost-Usd": usage["cost_usd"],
        "X-Aether-Usage-Source": usage["source"],
        "X-Aether-Trace-Id": trace_id,
        "X-Aether-Thread-Id": thread_id,
    }


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if json_logs_enabled(app.state.settings.json_logs):
        # uvicorn 会在导入后重置 logging；放到 lifespan 里，handler 才能留下来。
        configure_json_logging()
    await asyncio.to_thread(_check_storage, app.state.settings.data_dir)
    app.state.service = Service(app.state.settings)
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Aether", version=settings.version, lifespan=lifespan)
    app.state.settings = settings

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        trace_id = _request_trace_id(request)
        log_event(
            logger,
            "service_error",
            level=logging.WARNING,
            trace_id=trace_id,
            code=exc.code,
            status_code=exc.status,
            cost_usd="0",
        )
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"code": exc.code, "trace_id": trace_id}},
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        trace_id = _request_trace_id(request)
        log_event(
            logger,
            "service_error",
            level=logging.ERROR,
            trace_id=trace_id,
            code="service_unavailable",
            error_type=type(exc).__name__,
            cost_usd="0",
        )
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "service_unavailable", "trace_id": trace_id}},
        )

    @app.get("/healthz", tags=["meta"])
    async def healthz() -> dict:
        return {
            "status": "ok",
            "version": settings.version,
            "llm_configured": bool(settings.llm_api_key),
        }

    @app.get("/readyz", tags=["meta"], responses={503: {"description": "存储不可用"}})
    async def readyz() -> dict:
        try:
            await asyncio.to_thread(_check_storage, settings.data_dir)
        except OSError as exc:
            raise ServiceError(503, "storage_unavailable") from exc
        return {"status": "ready"}

    @app.get("/metrics", tags=["meta"], response_class=PlainTextResponse)
    async def metrics(request: Request) -> PlainTextResponse:
        return PlainTextResponse(
            request.app.state.service.metrics.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post(
        "/v1/ingest",
        tags=["retrieval"],
        responses={
            409: {"description": "摄取或聊天正在使用索引"},
            422: {"description": "语料无效或超出摄取根目录"},
            503: {"description": "provider 或存储不可用"},
        },
    )
    async def ingest(body: IngestRequest, request: Request) -> dict:
        """全量原子重建索引；corpus_dir 必须位于 AETHER_API_INGEST_ROOT 下。"""
        service: Service = request.app.state.service
        trace_id = _request_trace_id(request)
        started = time.perf_counter()
        error_code: str | None = None
        succeeded = False
        try:
            result = await service.ingest(body)
            succeeded = True
            return result
        except ServiceError as exc:
            error_code = exc.code
            raise
        finally:
            _store_simple_trace(
                service,
                trace_id=trace_id,
                method="POST",
                path="/v1/ingest",
                stage="ingest",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                status="success" if succeeded else "error",
                error_code=error_code,
            )

    @app.get("/v1/threads/{thread_id}", tags=["chat"])
    async def thread(thread_id: ThreadId, request: Request) -> dict:
        service = request.app.state.service
        try:
            async with service.session(thread_id) as searcher:
                return await read_thread(
                    searcher=searcher,
                    llm=service.llm,
                    checkpoint_path=service.checkpoint,
                    thread_id=thread_id,
                    retrieval_k=settings.agent_retrieval_k,
                )
        except ValueError as exc:
            raise ServiceError(404, "thread_not_found") from exc

    @app.post(
        "/v1/chat",
        tags=["chat"],
        response_class=StreamingResponse,
        response_model=None,
        responses={
            200: {
                "headers": {
                    "X-Aether-Tokens": {
                        "description": "本请求累计模型输入与输出 tokens",
                        "schema": {"type": "integer"},
                    },
                    "X-Aether-Cost-Usd": {
                        "description": "按配置价格核算的 USD",
                        "schema": {"type": "string"},
                    },
                    "X-Aether-Usage-Source": {
                        "description": "provider 或 estimated",
                        "schema": {"type": "string"},
                    },
                    "X-Aether-Thread-Id": {"schema": {"type": "string"}},
                    "X-Aether-Trace-Id": {"schema": {"type": "string"}},
                },
                "content": {"text/event-stream": {}},
                "description": "SSE: stage {stage,elapsed_ms,input_tokens,output_tokens,cost_usd}; "
                "token {text}; done {TurnResult,usage,trace_id,slowest_stage,slowest_elapsed_ms}。"
                "图执行及引用校验完成后发送，响应头为最终 usage；token 是验证后回答的文本分块。",
            },
            409: {"description": "thread 正忙或中断状态冲突"},
            429: {"description": "并发已满"},
            503: {"description": "provider/索引不可用，结构化 error.code，无异常原文"},
            504: {"description": "请求执行超时"},
        },
    )
    async def chat(body: ChatRequest, request: Request) -> StreamingResponse | JSONResponse:
        service: Service = request.app.state.service
        usage = RequestUsage(settings)
        trace_id = _request_trace_id(request)
        stages: list[dict] = []
        previous = zero_usage()
        started = time.perf_counter()

        async def stage(name: str, elapsed_ms: float) -> None:
            nonlocal previous
            current = usage.snapshot()
            item = annotate_stage(name, elapsed_ms, usage_delta(previous, current))
            previous = current
            stages.append(item)
            service.metrics.stage(name, elapsed_ms)
            log_event(logger, "stage", trace_id=trace_id, thread_id=body.thread_id, **item)

        def commit(success: bool, error_code: str | None) -> tuple[dict, dict]:
            final_usage = usage.snapshot()
            recorded = list(stages)
            if not recorded:
                item = annotate_stage(
                    "request",
                    (time.perf_counter() - started) * 1000,
                    usage_delta(previous, final_usage),
                )
                recorded.append(item)
                log_event(logger, "stage", trace_id=trace_id, thread_id=body.thread_id, **item)
            service.metrics.record(final_usage, success=success)
            record = make_trace(
                trace_id=trace_id,
                method="POST",
                path="/v1/chat",
                status="success" if success else "error",
                stages=recorded,
                usage=final_usage,
                thread_id=body.thread_id,
                error_code=error_code,
            )
            service.traces.put(record)
            log_event(
                logger,
                "chat_complete",
                level=logging.INFO if success else logging.ERROR,
                trace_id=trace_id,
                thread_id=body.thread_id,
                status=record["status"],
                error_code=error_code,
                slowest_stage=record["slowest_stage"],
                slowest_elapsed_ms=record["slowest_elapsed_ms"],
                input_tokens=final_usage["input_tokens"],
                output_tokens=final_usage["output_tokens"],
                total_tokens=final_usage["total_tokens"],
                cost_usd=final_usage["cost_usd"],
                usage_source=final_usage["source"],
            )
            return final_usage, record

        error: ServiceError | None = None
        try:
            async with asyncio.timeout(settings.api_request_timeout):
                async with service.session(body.thread_id) as searcher:
                    result = await run_turn(
                        searcher=searcher,
                        llm=service.llm,
                        checkpoint_path=service.checkpoint,
                        thread_id=body.thread_id,
                        question=body.question,
                        require_review=body.require_review,
                        resume=body.resume.model_dump() if body.resume is not None else None,
                        rewrite_budget=settings.agent_rewrite_budget,
                        min_support_terms=settings.agent_min_support_terms,
                        min_quote_chars=settings.agent_min_quote_chars,
                        retrieval_k=settings.agent_retrieval_k,
                        callbacks=[usage],
                        on_stage=stage,
                    )
                    if result["error_code"] == "llm_unavailable":
                        raise ServiceError(503, "provider_unavailable")
        except ServiceError as exc:
            error = exc
        except EmbeddingError:
            error = ServiceError(503, "provider_unavailable")
        except TimeoutError:
            error = ServiceError(504, "request_timeout")
        except ValueError:
            error = ServiceError(409, "thread_state_conflict")
        except asyncio.CancelledError:
            commit(False, "cancelled")
            raise
        except Exception as exc:
            log_event(
                logger,
                "chat_error",
                level=logging.ERROR,
                trace_id=trace_id,
                error_type=type(exc).__name__,
                cost_usd=usage.snapshot()["cost_usd"],
            )
            error = ServiceError(503, "service_unavailable")
        final_usage, record = commit(error is None, None if error is None else error.code)
        headers = _headers(final_usage, trace_id, body.thread_id)
        if error is not None:
            return JSONResponse(
                status_code=error.status,
                headers=headers,
                content={"error": {"code": error.code, "trace_id": trace_id}, "usage": final_usage},
            )

        async def events():
            for item in stages:
                yield _event("stage", item)
            answer = result["answer"]
            size = settings.api_token_chunk_chars
            for offset in range(0, len(answer), size):
                yield _event("token", {"text": answer[offset : offset + size]})
                await asyncio.sleep(0)
            yield _event(
                "done",
                {
                    **result,
                    "usage": final_usage,
                    "trace_id": trace_id,
                    "slowest_stage": record["slowest_stage"],
                    "slowest_elapsed_ms": record["slowest_elapsed_ms"],
                },
            )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get(
        "/v1/traces/{trace_id}",
        tags=["meta"],
        responses={404: {"description": "trace 不在本进程缓冲中（已重启或被挤出）"}},
    )
    async def read_trace(trace_id: str, request: Request) -> dict:
        """按 trace_id 返回阶段时延与成本，并标出最慢阶段。缓冲仅在本进程。"""
        if len(trace_id) != 32 or any(char not in "0123456789abcdef" for char in trace_id):
            raise ServiceError(404, "trace_not_found")
        record = request.app.state.service.traces.get(trace_id)
        if record is None:
            raise ServiceError(404, "trace_not_found")
        return record

    app.add_middleware(TraceMiddleware)
    return app


app = create_app()


def _request_trace_id(request: Request) -> str:
    trace_id = getattr(request.state, "trace_id", None)
    if isinstance(trace_id, str) and trace_id:
        return trace_id
    return uuid4().hex


def _store_simple_trace(
    service: Service,
    *,
    trace_id: str,
    method: str,
    path: str,
    stage: str,
    elapsed_ms: float,
    status: str,
    error_code: str | None,
) -> None:
    item = annotate_stage(
        stage,
        elapsed_ms,
        {"input_tokens": 0, "output_tokens": 0, "cost_usd": "0"},
    )
    record = make_trace(
        trace_id=trace_id,
        method=method,
        path=path,
        status=status,
        stages=[item],
        usage=zero_usage(),
        error_code=error_code,
    )
    service.traces.put(record)
    log_event(logger, "stage", trace_id=trace_id, **item)
    log_event(
        logger,
        "request_complete",
        level=logging.INFO if status == "success" else logging.WARNING,
        trace_id=trace_id,
        path=path,
        status=status,
        error_code=error_code,
        slowest_stage=record["slowest_stage"],
        slowest_elapsed_ms=record["slowest_elapsed_ms"],
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cost_usd="0",
        usage_source="none",
    )


def main() -> None:
    # 模块导入时已经 create_app()；用环境变量让随后的 lifespan 打开 JSON 日志。
    os.environ["AETHER_JSON_LOGS"] = "1"
    uvicorn.run("aether.api.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
