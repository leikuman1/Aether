"""共享测试夹具：隔离 .env 与本机环境变量，测试永不触网。"""

from pathlib import Path

import pytest

from aether.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, llm_api_key="", data_dir=tmp_path)
