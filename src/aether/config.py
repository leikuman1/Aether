"""集中配置：所有可变项走环境变量（AETHER_ 前缀），密钥绝不入库。"""

from functools import lru_cache
from pathlib import Path

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

    data_dir: Path = Path("data")


@lru_cache
def get_settings() -> Settings:
    return Settings()
