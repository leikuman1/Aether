"""共享测试夹具：隔离 .env 与本机环境变量，测试永不触网。"""

import os
import socket
from pathlib import Path

import pytest

from aether.config import Settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for key in list(os.environ):
        if key.startswith("AETHER_"):
            monkeypatch.delenv(key)

    def deny_network(*args, **kwargs):
        raise AssertionError("tests must not access the network")

    original_connect = socket.socket.connect
    original_socketpair = socket.socketpair

    def local_socketpair(*args, **kwargs):
        # Windows asyncio 用内部 loopback socketpair 唤醒事件循环。
        with monkeypatch.context() as local:
            local.setattr(socket.socket, "connect", original_connect)
            return original_socketpair(*args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "socketpair", local_socketpair)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, llm_api_key="", data_dir=tmp_path)
