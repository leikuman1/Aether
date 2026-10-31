"""Benchmark cleanup and the configured-backend success gate."""

import json
import sqlite3

import pytest

from aether.retrieval.bench import benchmark, main
from aether.retrieval.dense import _open_conn


def test_benchmark_closes_connection_when_index_construction_fails(monkeypatch):
    opened: list[sqlite3.Connection] = []

    def open_conn(path: str, *, extension: bool = False) -> sqlite3.Connection:
        conn = _open_conn(path, extension=extension)
        opened.append(conn)
        return conn

    def fail_index(*args, **kwargs):
        raise RuntimeError("index construction failed")

    monkeypatch.setattr("aether.retrieval.bench._open_conn", open_conn)
    monkeypatch.setattr("aether.retrieval.bench.create_dense_index", fail_index)
    report = benchmark(n=1, dim=2, queries=1)
    assert opened
    assert {name: row["passed"] for name, row in report["backends"].items()} == {
        "numpy": False,
        "sqlite-vec": False,
    }
    assert "selected" not in report
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


def test_tiny_benchmark_passes_both_backends():
    report = benchmark(n=2, dim=2, queries=1)
    assert report["backends"]["numpy"]["passed"] is True
    assert report["backends"]["sqlite-vec"]["recall@10"] == 1.0
    assert report["backends"]["sqlite-vec"]["passed"] is True


def _report(*, numpy: bool, sqlite_vec: bool) -> dict:
    return {"backends": {"numpy": {"passed": numpy}, "sqlite-vec": {"passed": sqlite_vec}}}


def test_main_requires_configured_sqlite_vec_even_if_numpy_passes(monkeypatch, capsys):
    report = _report(numpy=True, sqlite_vec=False)
    monkeypatch.setattr("aether.retrieval.bench.benchmark", lambda *args, **kwargs: report)
    monkeypatch.setattr("sys.argv", ["aether.retrieval.bench"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert report["selected"] == "sqlite-vec"
    printed = json.loads(capsys.readouterr().out)
    assert printed["selected"] == "sqlite-vec"
    assert printed["backends"]["numpy"]["passed"] is True


def test_main_accepts_configured_backend_and_follows_numpy_override(monkeypatch, capsys):
    report = _report(numpy=False, sqlite_vec=True)
    monkeypatch.setattr("aether.retrieval.bench.benchmark", lambda *args, **kwargs: report)
    monkeypatch.setattr("sys.argv", ["aether.retrieval.bench"])
    main()
    assert report["selected"] == "sqlite-vec"
    capsys.readouterr()

    monkeypatch.setenv("AETHER_DENSE_BACKEND", "numpy")
    report = _report(numpy=False, sqlite_vec=True)
    monkeypatch.setattr("aether.retrieval.bench.benchmark", lambda *args, **kwargs: report)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert report["selected"] == "numpy"
