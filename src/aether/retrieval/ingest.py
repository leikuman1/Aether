"""异步摄取与恢复：临时 SQLite 建库、校验成功后原子替换，失败保留旧索引。"""

import asyncio
import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path

from aether.retrieval.chunking import Chunker
from aether.retrieval.dense import _matrix, _open_conn, create_dense_index
from aether.retrieval.hybrid import HybridSearcher
from aether.retrieval.loaders import load_directory
from aether.retrieval.sparse import Bm25Index
from aether.retrieval.types import Chunk, Embedder


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    items = [(c.doc_id, c.chunk_id, c.text) for c in sorted(chunks, key=lambda c: c.chunk_id)]
    return hashlib.sha256(json.dumps(items, ensure_ascii=False).encode()).hexdigest()


def _build(db_path: Path, chunks: list[Chunk], vectors: list[list[float]], meta: dict) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".aether-", suffix=".db", dir=db_path.parent)
    os.close(fd)
    conn = None
    try:
        conn = _open_conn(temporary, extension=meta["dense_backend"] == "sqlite-vec")
        conn.executescript(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT, source TEXT, "
            "ordinal INTEGER, heading TEXT, text TEXT);"
        )
        conn.executemany("INSERT INTO meta VALUES (?, ?)", sorted(meta.items()))
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
            [tuple(asdict(c).values()) for c in chunks],
        )
        conn.commit()
        dense = create_dense_index(meta["dense_backend"], conn, int(meta["dim"]))
        dense.add([c.chunk_id for c in chunks], vectors)
        conn.close()
        conn = None
        os.replace(temporary, db_path)
    finally:
        if conn is not None:
            conn.close()
        Path(temporary).unlink(missing_ok=True)


async def ingest_directory(
    corpus_dir: Path,
    chunker: Chunker,
    embedder: Embedder,
    db_path: Path,
    *,
    dense_backend: str = "numpy",
) -> HybridSearcher:
    docs = await load_directory(corpus_dir)
    chunks = [c for doc in docs for c in chunker.split(doc)]
    if not chunks:
        raise ValueError("corpus has no nonempty chunks")
    if len({doc.doc_id for doc in docs}) != len(docs):
        raise ValueError("duplicate document IDs (same relative path with different extensions)")
    vectors = await embedder.embed_documents([c.text for c in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("embedder returned wrong vector count")
    _matrix(vectors, embedder.dim)
    meta = {
        "schema_version": "1",
        "chunker": chunker.name,
        "embedder": embedder.name,
        "dim": str(embedder.dim),
        "dense_backend": dense_backend,
        "num_chunks": str(len(chunks)),
        "num_docs": str(len(docs)),
        "corpus_fingerprint": corpus_fingerprint(chunks),
    }
    await asyncio.to_thread(_build, db_path, chunks, vectors, meta)
    return await asyncio.to_thread(open_searcher, db_path, embedder)


def read_meta(db_path: Path) -> dict[str, str]:
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return dict(conn.execute("SELECT key, value FROM meta").fetchall())
    finally:
        conn.close()


def load_chunks(db_path: Path) -> list[Chunk]:
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return [
            Chunk(*row)
            for row in conn.execute(
                "SELECT chunk_id, doc_id, source, ordinal, heading, text "
                "FROM chunks ORDER BY doc_id, ordinal"
            ).fetchall()
        ]
    finally:
        conn.close()


def open_searcher(
    db_path: Path,
    embedder: Embedder,
    *,
    rrf_k: int = 60,
    route_depth: int = 50,
) -> HybridSearcher:
    meta = read_meta(db_path)
    if meta["embedder"] != embedder.name:
        raise ValueError("index/embedder identity mismatch")
    if int(meta["dim"]) != embedder.dim:
        raise ValueError("index/embedder dimension mismatch")
    chunks = load_chunks(db_path)
    conn = _open_conn(str(db_path), extension=meta["dense_backend"] == "sqlite-vec")
    try:
        dense = create_dense_index(meta["dense_backend"], conn, int(meta["dim"]))
        bm25 = Bm25Index()
        bm25.add(chunks)
        return HybridSearcher(
            {c.chunk_id: c for c in chunks},
            bm25,
            dense,
            embedder,
            rrf_k=rrf_k,
            route_depth=route_depth,
        )
    except Exception:
        conn.close()
        raise


def index_info(db_path: Path) -> dict:
    meta = read_meta(db_path)
    return {**meta, "chunks": int(meta["num_chunks"]), "docs": int(meta["num_docs"])}
