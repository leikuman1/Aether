"""线程中的索引打开不能随请求取消而丢失资源所有权。"""

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from aether.graph import runner


@pytest.mark.parametrize("cancel", [True, False])
async def test_index_open_closes_embedder_and_late_searcher(settings, monkeypatch, cancel):
    entered, release = threading.Event(), threading.Event()
    embedder = AsyncMock()
    searcher = AsyncMock()

    def opening(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        if not cancel:
            raise ValueError("invalid index")
        return searcher

    monkeypatch.setattr(runner, "create_embedder", lambda settings: embedder)
    monkeypatch.setattr(runner, "open_searcher", opening)
    task = asyncio.create_task(runner.open_index(settings, settings.data_dir / "index.db"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()  # 仍持有生命周期直到线程归还连接。
        release.set()
        with pytest.raises(asyncio.CancelledError if cancel else ValueError):
            await task
        embedder.aclose.assert_awaited_once()
        if cancel:
            searcher.aclose.assert_awaited_once()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
