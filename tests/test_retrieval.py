"""Offline retrieval contracts, using temporary corpora and both dense backends."""

import asyncio
from dataclasses import replace
from itertools import pairwise

import pytest

from aether.retrieval.chunking import FixedWindowChunker, MarkdownChunker
from aether.retrieval.dense import _open_conn, create_dense_index
from aether.retrieval.embeddings import EmbeddingError, FakeEmbedder
from aether.retrieval.hybrid import HybridSearcher, rrf_fuse
from aether.retrieval.ingest import (
    index_info,
    ingest_directory,
    load_chunks,
    open_searcher,
    read_meta,
)
from aether.retrieval.tokenizer import tokenize_subwords, tokenize_words
from aether.retrieval.types import Chunk, Document, RetrievedChunk


@pytest.fixture(params=["numpy", "sqlite-vec"])
def backend(request):
    return request.param


@pytest.fixture
def dense(backend):
    conn = _open_conn(":memory:", extension=backend == "sqlite-vec")
    index = create_dense_index(backend, conn, 3)
    try:
        yield index
    finally:
        index.close()


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "guide.md").write_text(
        "# Deployment\nquasar deployment rollback release\n", encoding="utf-8"
    )
    (root / "nested").mkdir()
    (root / "nested" / "storage.txt").write_text(
        "sqlite transactions durable storage", encoding="utf-8"
    )
    (root / "network.markdown").write_text(
        "# Networking\nrouter firewall packet routing", encoding="utf-8"
    )
    (root / "ignored.json").write_text("{}", encoding="utf-8")
    return root


@pytest.mark.parametrize("chunker", [FixedWindowChunker(12, 3), MarkdownChunker(100)])
def test_chunkers_cover_body_with_bounded_chunks_and_stable_content_ids(chunker):
    body = "abcdefghijklmnopqrstuvwxyz" * 13
    doc = Document("guide", "guide.md", body)
    chunks = chunker.split(doc)
    limit = chunker.size if isinstance(chunker, FixedWindowChunker) else chunker.max_size
    assert len(chunks) > 1
    assert all(0 < len(c.text) <= limit for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.doc_id == doc.doc_id and c.source == doc.source for c in chunks)
    assert chunks == chunker.split(doc)
    assert [c.chunk_id for c in chunks] == [
        c.chunk_id for c in chunker.split(replace(doc, source="moved/guide.md"))
    ]
    changed = chunker.split(replace(doc, text="Z" + body[1:]))
    assert changed[0].chunk_id != chunks[0].chunk_id
    other_strategy = (
        FixedWindowChunker(13, 3)
        if isinstance(chunker, FixedWindowChunker)
        else MarkdownChunker(101)
    )
    assert other_strategy.split(doc)[0].chunk_id != chunks[0].chunk_id
    if isinstance(chunker, FixedWindowChunker):
        assert all(left.text[-3:] == right.text[:3] for left, right in pairwise(chunks))
        assert chunks[0].text + "".join(c.text[3:] for c in chunks[1:]) == body
    else:
        assert "".join(c.text for c in chunks) == body


def test_fixed_window_sentence_boundary_and_exact_size():
    chunker = FixedWindowChunker(10, 2)
    chunks = chunker.split(Document("a", "a.txt", "abcdef。ghijklmnop"))
    assert chunks[0].text == "abcdef。"
    assert chunks[1].text.startswith("f。")
    assert [c.text for c in chunker.split(Document("a", "a.txt", "0123456789"))] == ["0123456789"]
    assert chunker.split(Document("a", "a.txt", " \r\n\t")) == []
    assert chunker.split(Document("a", "a.txt", "abc\r\ndef")) == chunker.split(
        Document("a", "a.txt", "abc\ndef")
    )


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_markdown_heading_hierarchy_ignores_fenced_headings(fence):
    text = (
        f"intro\n# Parent\nfirst\n{fence}\n# literal heading\n"
        f"{fence}python\n## also literal\n{fence}\n## Child\nsecond\n# Peer\nthird"
    )
    chunks = MarkdownChunker(200).split(Document("a", "a.md", text))
    assert [c.heading for c in chunks] == ["", "Parent", "Parent > Child", "Peer"]
    assert chunks[0].text == "intro"
    assert f"{fence}\n# literal heading\n{fence}python\n## also literal\n{fence}" in chunks[1].text
    assert chunks[2].text == "Parent > Child\nsecond"
    assert chunks[3].text == "Peer\nthird"


def test_markdown_long_heading_budget_preserves_all_body():
    heading, body = "H" * 140, "x" * 249
    chunks = MarkdownChunker(100).split(Document("a", "a.md", f"# {heading}\n{body}"))
    assert all(c.heading == heading and len(c.text) <= 100 for c in chunks)
    assert "".join(c.text.split("\n", 1)[1] for c in chunks) == body
    assert MarkdownChunker(100).split(Document("a", "a.md", "# Heading\n \n")) == []


def test_chunker_invalid_boundaries():
    for factory in [
        lambda: FixedWindowChunker(0, 0),
        lambda: FixedWindowChunker(10, 10),
        lambda: FixedWindowChunker(10, -1),
        lambda: MarkdownChunker(99),
    ]:
        with pytest.raises(ValueError):
            factory()


def test_tokenizer_cjk_bigrams_stop_at_punctuation():
    for punctuation in [chr(0xFF0C), ".", "\n", chr(0xFF01)]:
        text = f"中文{punctuation}检索 API_42"
        words = tokenize_words(text)
        assert words == ["中", "文", "中文", "检", "索", "检索", "api", "42"]
        subwords = tokenize_subwords(text)
        assert "文检" not in words and "文检" not in subwords
        assert "<ap" in subwords and "pi>" in subwords
    assert tokenize_words("かな。한국") == ["か", "な", "かな", "한", "국", "한국"]
    assert tokenize_words("BM25算法") == ["bm25", "算", "法", "算法"]
    assert tokenize_words("2024年") == ["2024", "年"]
    assert tokenize_words("算法BM25") == ["算", "法", "算法", "bm25"]


def test_dense_cosine_ranking_upsert_and_empty(dense):
    assert dense.search([1, 0, 0], 5) == []
    dense.add([], [])
    dense.add(["exact", "diagonal", "opposite"], [[2, 0, 0], [1, 1, 0], [-3, 0, 0]])
    hits = dense.search([10, 0, 0], 10)
    assert [cid for cid, _ in hits] == ["exact", "diagonal", "opposite"]
    assert [score for _, score in hits] == pytest.approx([1, 2**-0.5, -1], abs=1e-6)
    assert dense.search([1, 0, 0], 1) == hits[:1]
    dense.add(["exact"], [[0, 0, 1]])
    assert len(dense) == 3
    updated = dense.search([0, 0, 4], 1)
    assert updated[0][0] == "exact"
    assert updated[0][1] == pytest.approx(1)


def test_dense_rejects_bad_vectors_without_mutation(dense):
    dense.add(["old"], [[1, 0, 0]])
    for vector in [[1, 0], [float("nan"), 1, 0], [0, 0, 0]]:
        with pytest.raises(ValueError):
            dense.add(["old"], [vector])
        with pytest.raises(ValueError):
            dense.search(vector, 1)
        assert len(dense) == 1
        assert dense.search([1, 0, 0], 1)[0][1] == pytest.approx(1)


def test_dense_validates_ids_and_k(dense, backend):
    for ids, vectors in [(["x", "x"], [[1, 0, 0]] * 2), (["x"], [])]:
        with pytest.raises(ValueError):
            dense.add(ids, vectors)
    for k in [0, -1]:
        with pytest.raises(ValueError, match="k"):
            dense.search([1, 0, 0], k)
    if backend == "sqlite-vec":
        with pytest.raises(ValueError, match="4096"):
            dense.search([1, 0, 0], 4097)
    assert len(dense) == 0


def test_dense_persistence_after_connection_reopen(tmp_path, backend):
    path = str(tmp_path / "dense.db")
    index = create_dense_index(backend, _open_conn(path, extension=backend == "sqlite-vec"), 3)
    try:
        index.add(["a", "b"], [[1, 0, 0], [0, 1, 0]])
        expected = index.search([1, 0, 0], 5)
    finally:
        index.close()
    restored = create_dense_index(backend, _open_conn(path, extension=backend == "sqlite-vec"), 3)
    try:
        assert len(restored) == 2
        assert restored.search([1, 0, 0], 5) == expected
        restored.add(["a"], [[0, 0, 1]])
        assert len(restored) == 2
        assert restored.search([0, 0, 1], 1)[0][0] == "a"
    finally:
        restored.close()


def test_rrf_weighted_math_and_score_independence():
    routes = {"sparse": [("a", 100), ("b", 0.01)], "dense": [("b", -9), ("c", -10)]}
    hits = rrf_fuse(routes, rrf_k=10, weights={"sparse": 2, "dense": 1}, k=3)
    assert [cid for cid, _ in hits] == ["b", "a", "c"]
    assert [score for _, score in hits] == pytest.approx([2 / 12 + 1 / 11, 2 / 11, 1 / 12])
    assert rrf_fuse({"sparse": routes["sparse"], "dense": []}, rrf_k=10) == [
        ("a", 1 / 11),
        ("b", 1 / 12),
    ]
    assert rrf_fuse(routes, weights={"dense": 0}) == [("a", 1 / 61), ("b", 1 / 62)]
    assert rrf_fuse({}) == []


def test_rrf_duplicates_contribute_once_and_ties_are_stable():
    hits = rrf_fuse({"sparse": [("a", 3), ("a", 2), ("b", 1)]}, rrf_k=10)
    assert hits == [("a", 1 / 11), ("b", 1 / 13)]
    assert rrf_fuse({"s": [("b", 1)], "d": [("a", 900)]}, k=1) == [("a", 1 / 61)]
    for kwargs in [{"k": 0}, {"rrf_k": 0}, {"weights": {"s": -1}}]:
        with pytest.raises(ValueError):
            rrf_fuse({"s": [("a", 1)]}, **kwargs)


async def test_reranker_receives_candidates_beyond_final_k():
    class Route:
        def __init__(self, hits):
            self.hits = hits
            self.depths = []

        def search(self, query, k):
            self.depths.append(k)
            return self.hits[:k]

    class PromoteLast:
        name = "promote-last"

        async def rerank(self, query, results, k):
            assert query == "query" and k == 1
            assert len(results) == 4
            assert all(set(r.route_scores) == {"bm25", "dense"} for r in results)
            return results[-1:]

    chunks = {str(i): Chunk(str(i), "doc", "doc.txt", i, "", str(i)) for i in range(4)}
    sparse = Route([(str(i), 4 - i) for i in range(4)])
    dense = Route([(str(i), 1 - i / 4) for i in range(4)])
    searcher = HybridSearcher(
        chunks, sparse, dense, FakeEmbedder(3), route_depth=4, reranker=PromoteLast()
    )
    results, trace = await searcher.search_with_trace("query", k=1)
    assert [r.chunk.chunk_id for r in results] == ["3"]
    assert sparse.depths == dense.depths == [4]
    assert trace["candidates"] == {"bm25": 4, "dense": 4}
    assert trace["reranker"] == "promote-last"
    assert trace["total_ms"] >= 0


async def test_ingest_restore_and_repeat_are_equivalent(corpus, tmp_path, backend):
    db = tmp_path / "index.db"
    embedder = FakeEmbedder(32)
    searcher = await ingest_directory(
        corpus, MarkdownChunker(100), embedder, db, dense_backend=backend
    )
    try:
        original = {}
        for mode in ["bm25", "dense", "hybrid"]:
            hits, trace = await searcher.search_with_trace("quasar deployment", k=3, mode=mode)
            assert hits and hits[0].chunk.doc_id == "guide"
            assert all(isinstance(hit, RetrievedChunk) for hit in hits)
            assert all(hit.route_scores for hit in hits)
            assert trace["mode"] == mode and trace["total_ms"] >= 0
            original[mode] = hits
        assert await searcher.search(" \n ") == []
    finally:
        await searcher.aclose()
    chunks, meta = await asyncio.to_thread(lambda: (load_chunks(db), read_meta(db)))
    assert {c.doc_id for c in chunks} == {"guide", "nested/storage", "network"}
    assert meta["dense_backend"] == backend
    assert meta["embedder"] == embedder.name
    assert index_info(db)["chunks"] == len(chunks) == 3
    assert index_info(db)["docs"] == 3
    restored = await asyncio.to_thread(open_searcher, db, embedder)
    try:
        for mode, expected in original.items():
            assert await restored.search("quasar deployment", 3, mode) == expected
    finally:
        await restored.aclose()
    repeated = await ingest_directory(
        corpus, MarkdownChunker(100), embedder, db, dense_backend=backend
    )
    try:
        assert await repeated.search("quasar deployment", 3) == original["hybrid"]
        assert await asyncio.to_thread(load_chunks, db) == chunks
        assert await asyncio.to_thread(read_meta, db) == meta
    finally:
        await repeated.aclose()


async def test_ingest_embedding_failure_preserves_previous_database(corpus, tmp_path, backend):
    class BrokenEmbedder(FakeEmbedder):
        async def embed_documents(self, texts):
            raise EmbeddingError("fixture embedding failure")

    db = tmp_path / "index.db"
    embedder = FakeEmbedder(32)
    original = await ingest_directory(
        corpus, MarkdownChunker(100), embedder, db, dense_backend=backend
    )
    try:
        expected = await original.search("quasar deployment", 3)
    finally:
        await original.aclose()
    before = await asyncio.to_thread(db.read_bytes)
    await asyncio.to_thread((corpus / "guide.md").write_text, "changed body", encoding="utf-8")
    with pytest.raises(EmbeddingError, match="fixture"):
        await ingest_directory(
            corpus, MarkdownChunker(100), BrokenEmbedder(32), db, dense_backend=backend
        )
    assert await asyncio.to_thread(db.read_bytes) == before
    assert not list(tmp_path.glob(".aether-*.db"))
    restored = await asyncio.to_thread(open_searcher, db, embedder)
    try:
        assert await restored.search("quasar deployment", 3) == expected
    finally:
        await restored.aclose()


async def test_ingest_rejects_embedder_identity_and_dimension_mismatch(corpus, tmp_path, backend):
    class RenamedEmbedder(FakeEmbedder):
        @property
        def name(self):
            return "different-model"

    class WrongDimensionEmbedder(FakeEmbedder):
        @property
        def name(self):
            return FakeEmbedder(32).name

    db = tmp_path / "index.db"
    searcher = await ingest_directory(
        corpus, MarkdownChunker(100), FakeEmbedder(32), db, dense_backend=backend
    )
    await searcher.aclose()
    with pytest.raises(ValueError, match="identity mismatch"):
        await asyncio.to_thread(open_searcher, db, RenamedEmbedder(32))
    with pytest.raises(ValueError, match="dimension mismatch"):
        await asyncio.to_thread(open_searcher, db, WrongDimensionEmbedder(16))
