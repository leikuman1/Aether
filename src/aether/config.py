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
    dense_backend: Literal["numpy", "sqlite-vec"] = "sqlite-vec"
    retrieval_rrf_k: int = Field(default=60, gt=0)
    retrieval_route_depth: int = Field(default=50, gt=0, le=4096)
    # 离线 FakeEmbedder 的保守配置；真实模型需在独立标注集重新校准。
    retrieval_sparse_weight: float = Field(default=16.0, gt=0)
    retrieval_dense_weight: float = Field(default=1.0, gt=0)

    # 纠错 Agent：改写次数、检索条数、证据门禁。温度 0 便于复现。
    agent_rewrite_budget: int = Field(default=2, ge=0, le=8)
    agent_retrieval_k: int = Field(default=5, gt=0, le=20)
    agent_min_support_terms: int = Field(default=4, ge=1, le=30)
    agent_min_quote_chars: int = Field(default=12, ge=8, le=200)
    llm_timeout: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=8)
    llm_temperature: float = 0.0

    # 服务采用单 worker；同 thread 串行，不同 thread 并发，摄取期间拒绝新聊天。
    # json_logs 默认关，避免测试把日志从根 logger 上摘掉；uv run aether-api 会打开。
    json_logs: bool = False
    api_request_timeout: float = Field(default=60.0, gt=0)
    api_max_concurrency: int = Field(default=32, ge=1, le=512)
    api_token_chunk_chars: int = Field(default=16, ge=1, le=256)
    api_ingest_root: Path | None = None  # 默认 data_dir/corpus；仅允许此根目录内摄取
    llm_input_cost_per_million: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    llm_output_cost_per_million: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    data_dir: Path = Path("data")


@lru_cache
def get_settings() -> Settings:
    return Settings()
