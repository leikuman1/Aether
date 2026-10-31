"""离线确定性子词哈希与 OpenAI 兼容异步 embedding 客户端。"""

import asyncio
import hashlib
import math

import httpx

from aether.config import Settings
from aether.retrieval.tokenizer import tokenize_subwords


class EmbeddingError(Exception):
    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class FakeEmbedder:
    """仅模拟子词相似度，不能作为真实语义质量的证明。"""

    def __init__(self, dim: int = 256):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def name(self) -> str:
        return f"fake-hash-v1-{self._dim}d"

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize_subwords(text):
            digest = hashlib.sha256(token.encode()).digest()
            vec[int.from_bytes(digest[:4], "big") % self.dim] += 1 if digest[4] % 2 else -1
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0], norm = 1.0, 1.0
        return [v / norm for v in vec]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(lambda: [self._embed(t) for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAICompatEmbedder:
    """所有 provider 配置来自 Settings；429/5xx/transport error 有界退避。"""

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None):
        if not settings.llm_api_key:
            raise EmbeddingError("AETHER_LLM_API_KEY is required for production embeddings")
        self._settings = settings
        self._owned = client is None
        self._client = client or httpx.AsyncClient(timeout=settings.embedding_timeout)

    @property
    def name(self) -> str:
        endpoint = self._settings.llm_base_url.rstrip("/")
        fingerprint = hashlib.sha256(endpoint.encode()).hexdigest()[:12]
        return f"openai-compat:{fingerprint}:{self._settings.embedding_model}:{self.dim}d"

    @property
    def dim(self) -> int:
        return self._settings.embedding_dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = []
        for start in range(0, len(texts), self._settings.embedding_batch_size):
            batch = texts[start : start + self._settings.embedding_batch_size]
            result.extend(await self._request(batch))
        return result

    async def embed_query(self, text: str) -> list[float]:
        return (await self._request([text]))[0]

    async def _request(self, texts: list[str]) -> list[list[float]]:
        settings = self._settings
        last_status = None
        for attempt in range(settings.embedding_max_retries + 1):
            try:
                response = await self._client.post(
                    settings.llm_base_url.rstrip("/") + "/embeddings",
                    headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                    json={"model": settings.embedding_model, "input": texts},
                    timeout=settings.embedding_timeout,
                )
            except httpx.TransportError:
                last_status = None
            else:
                last_status = response.status_code
                if response.is_success:
                    try:
                        return self._parse(response.json(), len(texts))
                    except (ValueError, TypeError, KeyError, AttributeError) as exc:
                        raise EmbeddingError("malformed embeddings response") from exc
                if last_status != 429 and last_status < 500:
                    raise EmbeddingError("embeddings request rejected", status=last_status)
            if attempt < settings.embedding_max_retries:
                await asyncio.sleep(settings.embedding_retry_delay * 2**attempt)
        # 不携带 provider 响应/请求 URL，避免日志泄漏密钥或输入语料。
        raise EmbeddingError("embeddings retry budget exhausted", status=last_status)

    def _parse(self, body: dict, count: int) -> list[list[float]]:
        data = body["data"]
        if not isinstance(data, list) or len(data) != count:
            raise ValueError("embedding count mismatch")
        values = {}
        for item in data:
            index = item["index"]
            vector = item["embedding"]
            if type(index) is not int or index in values or not 0 <= index < count:
                raise ValueError("invalid embedding index")
            if not isinstance(vector, list) or len(vector) != self.dim:
                raise ValueError("embedding dimension mismatch")
            if any(type(v) not in (float, int) or not math.isfinite(v) for v in vector):
                raise ValueError("nonfinite embedding")
            norm = math.hypot(*vector)
            if norm == 0 or not math.isfinite(norm):
                raise ValueError("invalid embedding norm")
            values[index] = [float(v) for v in vector]
        return [values[i] for i in range(count)]

    async def aclose(self) -> None:
        if self._owned:
            await self._client.aclose()


def create_embedder(settings: Settings) -> FakeEmbedder | OpenAICompatEmbedder:
    if settings.embedding_provider == "fake":
        return FakeEmbedder(settings.fake_embedding_dim)
    return OpenAICompatEmbedder(settings)
