"""FastAPI 入口：M0 骨架（健康检查 + 可写数据目录探测），chat/ingest 在 M3 落地。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from aether.config import Settings, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.llm_configured = bool(app.state.settings.llm_api_key)
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Aether", version=settings.version, lifespan=lifespan)
    app.state.settings = settings

    @app.get("/healthz", tags=["meta"])
    async def healthz() -> dict:
        return {
            "status": "ok",
            "version": settings.version,
            "llm_configured": bool(settings.llm_api_key),
        }

    @app.get("/readyz", tags=["meta"])
    async def readyz() -> dict:
        # M3 起此目录承载索引/会话/评测产物；就绪检查顺带验证可写
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "ready"}

    return app


app = create_app()


def main() -> None:
    uvicorn.run("aether.api.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
