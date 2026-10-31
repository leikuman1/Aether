"""异步服务：受引用门保护的 SSE、SqliteSaver 会话和请求级模型核算。"""

import asyncio
import json
import logging
import tempfile
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
    await asyncio.to_thread(_check_storage, app.state.settings.data_dir)
    app.state.service = Service(app.state.settings)
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Aether", version=settings.version, lifespan=lifespan)
    app.state.settings = settings

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"code": exc.code, "trace_id": uuid4().hex}},
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        trace_id = uuid4().hex
        logger.error("service_error trace_id=%s type=%s", trace_id, type(exc).__name__)
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
        return await request.app.state.service.ingest(body)

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
                "description": "SSE: stage {stage,elapsed_ms}; token {text}; "
                "done {TurnResult,usage,trace_id}。"
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
        trace_id = uuid4().hex
        stages: list[dict] = []

        async def stage(name: str, elapsed_ms: float) -> None:
            stages.append({"stage": name, "elapsed_ms": round(elapsed_ms, 3)})
            service.metrics.stage(name, elapsed_ms)
            logger.info("stage trace_id=%s stage=%s elapsed_ms=%.3f", trace_id, name, elapsed_ms)

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
            service.metrics.record(usage.snapshot(), success=False)
            raise
        except Exception as exc:
            logger.error("chat_error trace_id=%s type=%s", trace_id, type(exc).__name__)
            error = ServiceError(503, "service_unavailable")
        final_usage = usage.snapshot()
        service.metrics.record(final_usage, success=error is None)
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
            yield _event("done", {**result, "usage": final_usage, "trace_id": trace_id})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()


def main() -> None:
    uvicorn.run("aether.api.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
