"""稀疏检索：rank-bm25（Okapi BM25）封装，CJK 感知分词。

rank-bm25 无持久化：chunk 文本存 sqlite chunks 表，索引在加载时重建——
当前规模（<10 万 chunk）重建毫秒级，实测数字见 docs/decisions.md。
"""

from rank_bm25 import BM25Okapi

from aether.retrieval.tokenizer import cached_tokenize_words
from aether.retrieval.types import Chunk


class Bm25Index:
    """词级 BM25：add 批量建索引，search 返回 (chunk_id, 分数) 按分数降序。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._bm25: BM25Okapi | None = None

    @property
    def name(self) -> str:
        return f"bm25okapi-k{self._k1}-b{self._b}"

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._ids.extend(c.chunk_id for c in chunks)
        # heading 已前置进 chunk.text（markdown 策略），无需重复拼接
        self._texts.extend(c.text for c in chunks)
        self._rebuild()

    def _rebuild(self) -> None:
        if not self._ids:
            self._bm25 = None
            return
        corpus = [cached_tokenize_words(t) or ["__empty_document__"] for t in self._texts]
        self._bm25 = BM25Okapi(corpus, k1=self._k1, b=self._b)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or k <= 0:
            return []
        scores = self._bm25.get_scores(cached_tokenize_words(query))
        ranked = sorted(
            ((cid, float(s)) for cid, s in zip(self._ids, scores, strict=True) if s > 0),
            key=lambda x: (-x[1], x[0]),  # 分数相同按 id 稳定排序（可复现）
        )
        return ranked[:k]
