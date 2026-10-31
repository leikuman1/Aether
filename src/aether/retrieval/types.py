"""检索核心数据类型与协议：chunk、检索结果、Embedder/DenseIndex/Reranker 协议。"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Document:
    """一篇待摄取的文档（markdown/txt），doc_id 默认为相对路径去扩展名。"""

    doc_id: str
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """检索的最小单元；chunk_id 由 doc_id + 分块策略 + 序号哈希派生，确定性可复现。"""

    chunk_id: str
    doc_id: str
    source: str
    ordinal: int
    heading: str
    text: str

    @property
    def display(self) -> str:
        return f"{self.heading} | {self.text[:60]}" if self.heading else self.text[:60]


@dataclass(frozen=True)
class RetrievedChunk:
    """一路检索命中；scores 记录该 chunk 在各检索路（bm25/dense）的得分，便于观测。"""

    chunk: Chunk
    score: float
    route_scores: dict[str, float] = field(default_factory=dict)


class Embedder(Protocol):
    """文本 → 向量。生产走 OpenAI 兼容协议，测试用 FakeEmbedder，二者可替换。"""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class DenseIndex(Protocol):
    """稠密向量索引：add 后可按向量近邻检索，返回 (chunk_id, 相似度分数)。"""

    @property
    def name(self) -> str: ...

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None: ...

    def search(self, query: list[float], k: int) -> list[tuple[str, float]]: ...


class Reranker(Protocol):
    """重排接口预留（M2+ 接入 cross-encoder）；M1 仅提供恒等实现。"""

    @property
    def name(self) -> str: ...

    async def rerank(
        self, query: str, results: list[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]: ...
