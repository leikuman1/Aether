"""集中配置：所有可变项走环境变量（AETHER_ 前缀），密钥绝不入库。"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AETHER_", env_file=".env", extra="ignore")

    app_name: str = "aether"
    version: str = "0.1.0"

    # OpenAI 兼容协议：智谱 GLM / OpenAI / DeepSeek / 本地 vLLM 均可
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_model: str = "glm-4.6"
    llm_api_key: str = ""  # 留空 = 不发起任何外部调用（测试/演练模式）
    embedding_model: str = "embedding-3"
    embedding_provider: Literal["fake", "openai"] = "fake"
    embedding_dim: int = Field(default=2048, gt=0)
    fake_embedding_dim: int = Field(default=256, gt=0)
    embedding_timeout: float = Field(default=30.0, gt=0)
    embedding_batch_size: int = Field(default=64, gt=0)
    embedding_max_retries: int = Field(default=3, ge=0, le=10)
    embedding_retry_delay: float = Field(default=0.5, ge=0)
    dense_backend: Literal["numpy", "sqlite-vec"] = "numpy"
    retrieval_rrf_k: int = Field(default=60, gt=0)
    retrieval_route_depth: int = Field(default=50, gt=0, le=4096)
    # 离线 FakeEmbedder 的保守配置；真实模型需在独立标注集重新校准。
    retrieval_sparse_weight: float = Field(default=16.0, gt=0)
    retrieval_dense_weight: float = Field(default=1.0, gt=0)

    data_dir: Path = Path("data")


@lru_cache
def get_settings() -> Settings:
    return Settings()
