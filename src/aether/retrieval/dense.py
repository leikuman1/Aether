"""精确余弦检索双后端，持久化于 SQLite；调用方在 worker thread 执行 I/O。"""

import sqlite3
import threading

import numpy as np


def _open_conn(path: str, *, extension: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    if extension:
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            try:
                conn.load_extension(sqlite_vec.loadable_path())
            finally:
                conn.enable_load_extension(False)
        except Exception:
            conn.close()
            raise
    return conn


def _matrix(vectors: list[list[float]], dim: int) -> np.ndarray:
    mat = np.asarray(vectors, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[1] != dim or not np.isfinite(mat).all():
        raise ValueError(f"expected finite vectors of dimension {dim}")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("zero or overflowing vectors cannot use cosine distance")
    return mat / norms


def _validate_ids(ids: list[str], vectors: list[list[float]]) -> None:
    if len(ids) != len(vectors) or len(set(ids)) != len(ids):
        raise ValueError("ids must be unique and match vector count")


class NumpyDenseIndex:
    def __init__(self, conn: sqlite3.Connection, dim: int):
        self._conn, self._dim = conn, dim
        self._lock = threading.RLock()
        conn.execute("CREATE TABLE IF NOT EXISTS vectors (chunk_id TEXT PRIMARY KEY, vec BLOB)")
        conn.commit()
        self._reload()

    @property
    def name(self) -> str:
        return "numpy"

    def _reload(self) -> None:
        rows = self._conn.execute("SELECT chunk_id, vec FROM vectors ORDER BY chunk_id").fetchall()
        self._ids = [r[0] for r in rows]
        self._mat = (
            np.stack([np.frombuffer(r[1], dtype="<f4") for r in rows])
            if rows
            else np.empty((0, self._dim), dtype=np.float32)
        )
        if self._mat.shape[1] != self._dim:
            raise ValueError("stored vector dimension mismatch")

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        _validate_ids(chunk_ids, vectors)
        if not chunk_ids:
            return
        mat = _matrix(vectors, self._dim)
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                [(cid, v.astype("<f4").tobytes()) for cid, v in zip(chunk_ids, mat, strict=True)],
            )
            self._reload()

    def search(self, query: list[float], k: int) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        q = _matrix([query], self._dim)[0]
        with self._lock:
            scores = self._mat @ q
            top = np.argsort(-scores, kind="stable")[:k]
            return [(self._ids[i], float(scores[i])) for i in top]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class SqliteVecDenseIndex:
    def __init__(self, conn: sqlite3.Connection, dim: int):
        self._conn, self._dim = conn, dim
        self._lock = threading.RLock()
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"chunk_id TEXT PRIMARY KEY, embedding float[{int(dim)}] distance_metric=cosine)"
        )
        conn.commit()

    @property
    def name(self) -> str:
        return "sqlite-vec"

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        _validate_ids(chunk_ids, vectors)
        if not chunk_ids:
            return
        mat = _matrix(vectors, self._dim)
        with self._lock, self._conn:
            # vec0 不支持可靠的 UPSERT；显式删除后插入，同一事务内完成。
            self._conn.executemany(
                "DELETE FROM vec_chunks WHERE chunk_id = ?",
                [(cid,) for cid in chunk_ids],
            )
            self._conn.executemany(
                "INSERT INTO vec_chunks VALUES (?, ?)",
                [(cid, v.astype("<f4").tobytes()) for cid, v in zip(chunk_ids, mat, strict=True)],
            )

    def search(self, query: list[float], k: int) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        q = _matrix([query], self._dim)[0].astype("<f4").tobytes()
        with self._lock:
            # vec0 的 KNN 上限为 4096；超限明确失败，避免静默少召回。
            if k > 4096:
                raise ValueError("sqlite-vec requires k <= 4096")
            rows = self._conn.execute(
                "SELECT chunk_id, distance FROM vec_chunks "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (q, k),
            ).fetchall()
        return sorted(
            [(cid, 1.0 - float(distance)) for cid, distance in rows],
            key=lambda item: (-item[1], item[0]),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def create_dense_index(
    backend: str,
    conn: sqlite3.Connection,
    dim: int,
) -> NumpyDenseIndex | SqliteVecDenseIndex:
    if dim <= 0:
        raise ValueError("dimension must be positive")
    if backend == "numpy":
        return NumpyDenseIndex(conn, dim)
    if backend == "sqlite-vec":
        return SqliteVecDenseIndex(conn, dim)
    raise ValueError(f"unknown dense backend: {backend}")
