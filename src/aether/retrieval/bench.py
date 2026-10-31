"""python -m aether.retrieval.bench：固定种子、精确召回、双后端延迟与落盘体积。"""

import argparse
import json
import platform
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np

from aether.retrieval.dense import _open_conn, create_dense_index


def benchmark(n: int = 10000, dim: int = 256, queries: int = 100) -> dict:
    if min(n, dim, queries) <= 0:
        raise ValueError("n, dim, queries must be positive")
    rng = np.random.default_rng(2026)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    qvecs = rng.normal(size=(queries, dim)).astype(np.float32)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True)
    ids = [f"v-{i:07d}" for i in range(n)]
    k = min(10, n)
    truth = [set(np.argsort(-(vectors @ q), kind="stable")[:k].tolist()) for q in qvecs]
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "seed": 2026,
        "n": n,
        "dim": dim,
        "queries": queries,
        "k": k,
        "thresholds": {"recall@10": 1.0, "query_p95_ms": 100, "build_ms": 30000},
        "backends": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        for backend in ("numpy", "sqlite-vec"):
            path = Path(tmp) / f"{backend}.db"
            index = None
            try:
                started = time.perf_counter()
                conn = _open_conn(str(path), extension=backend == "sqlite-vec")
                index = create_dense_index(backend, conn, dim)
                index.add(ids, vectors.tolist())
                build_ms = (time.perf_counter() - started) * 1000
                index.search(qvecs[0].tolist(), k)  # warmup
                times, recalls = [], []
                for q, expected in zip(qvecs, truth, strict=True):
                    started = time.perf_counter()
                    hits = index.search(q.tolist(), k)
                    times.append((time.perf_counter() - started) * 1000)
                    got = {int(cid.removeprefix("v-")) for cid, _ in hits}
                    recalls.append(len(got & expected) / k)
                index.close()
                index = None
                recall = float(np.mean(recalls))
                p95 = float(np.percentile(times, 95))
                report["backends"][backend] = {
                    "build_ms": round(build_ms, 3),
                    "query_p50_ms": round(float(np.median(times)), 3),
                    "query_p95_ms": round(p95, 3),
                    "recall@10": recall,
                    "disk_bytes": path.stat().st_size,
                    "passed": recall == 1.0 and p95 <= 100 and build_ms <= 30000,
                }
            except Exception as exc:
                report["backends"][backend] = {"passed": False, "error": str(exc)}
            finally:
                if index is not None:
                    index.close()
    report["selected"] = "sqlite-vec" if report["backends"]["sqlite-vec"]["passed"] else "numpy"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args.n, args.dim, args.queries)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    # numpy 是明确退路，但双后端都失败时必须红。
    if not any(r["passed"] for r in report["backends"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
