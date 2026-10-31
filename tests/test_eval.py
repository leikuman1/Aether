import copy
import json
import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest

from aether.config import Settings
from aether.eval.retrieval import evaluate, load_golden, recall_at_k, reciprocal_rank
from aether.retrieval.chunking import MarkdownChunker
from aether.retrieval.hybrid import HybridSearcher
from aether.retrieval.loaders import load_directory

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_known_rankings_and_duplicate_hits():
    ranked = ["wrong", "b", "b", "a"]
    assert recall_at_k(ranked, {"a", "b"}, 3) == 0.5
    assert recall_at_k(ranked, {"a", "b"}, 4) == 1
    assert reciprocal_rank(ranked, {"a", "b"}, 4) == 0.5
    assert reciprocal_rank(["wrong"], {"a"}, 10) == 0
    with pytest.raises(ValueError):
        recall_at_k([], set(), 5)


async def test_golden_gate_reproducibility_and_label_integrity(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, llm_api_key="", data_dir=tmp_path)
    corpus = ROOT / "data/golden/corpus"
    golden = ROOT / "data/golden/retrieval.jsonl"
    main_thread = threading.get_ident()
    created: list[int] = []
    cleaned: list[tuple[int, str]] = []
    original_init = tempfile.TemporaryDirectory.__init__
    original_cleanup = tempfile.TemporaryDirectory.cleanup

    def init(self, *args, **kwargs):
        created.append(threading.get_ident())
        original_init(self, *args, **kwargs)

    def cleanup(self):
        cleaned.append((threading.get_ident(), self.name))
        original_cleanup(self)

    monkeypatch.setattr(tempfile.TemporaryDirectory, "__init__", init)
    monkeypatch.setattr(tempfile.TemporaryDirectory, "cleanup", cleanup)
    first = await evaluate(settings, corpus, golden)
    assert created and all(ident != main_thread for ident in created)
    assert cleaned
    assert all(ident != main_thread and not Path(name).exists() for ident, name in cleaned)
    second = await evaluate(settings, corpus, golden)
    assert first["queries"] >= 50
    assert first["gate_passed"] is True
    assert {s["chunker"] for s in first["strategies"]} == {"fixed-400-64", "markdown-480"}
    for left, right in zip(first["strategies"], second["strategies"], strict=True):
        assert left["corpus_fingerprint"] == right["corpus_fingerprint"]
        for mode in ("bm25", "dense", "hybrid"):
            assert left["routes"][mode]["recall@5"] == right["routes"][mode]["recall@5"]
            assert left["routes"][mode]["mrr@10"] == right["routes"][mode]["mrr@10"]
            assert [r["retrieved_chunk_ids"] for r in left["routes"][mode]["per_query"]] == [
                r["retrieved_chunk_ids"] for r in right["routes"][mode]["per_query"]
            ]
    corrupted = copy.deepcopy(load_golden(golden))
    corrupted[0]["relevant_chunk_ids"] = ["invented"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(json.dumps(row) for row in corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="stale canonical"):
        await evaluate(settings, corpus, bad)


async def test_label_mapping_failure_closes_searcher(tmp_path, monkeypatch):
    import aether.eval.retrieval as retrieval

    settings = Settings(_env_file=None, llm_api_key="", data_dir=tmp_path)
    canonical_ids: set[str] | None = None
    closed: list[HybridSearcher] = []
    real_ids = retrieval.evidence_ids
    real_close = HybridSearcher.aclose

    def mapped(row, chunks):
        nonlocal canonical_ids
        ids = {chunk.chunk_id for chunk in chunks}
        if canonical_ids is None:
            canonical_ids = ids
        if ids != canonical_ids:
            raise ValueError(f"unmapped evidence in {row['id']}: forced")
        return real_ids(row, chunks)

    async def tracking_close(self):
        closed.append(self)
        await real_close(self)

    monkeypatch.setattr(retrieval, "evidence_ids", mapped)
    monkeypatch.setattr(HybridSearcher, "aclose", tracking_close)
    with pytest.raises(ValueError, match="unmapped evidence"):
        await evaluate(settings, ROOT / "data/golden/corpus", ROOT / "data/golden/retrieval.jsonl")
    assert closed
    with pytest.raises(sqlite3.ProgrammingError):
        closed[0].dense._conn.execute("SELECT 1")


async def test_equal_weight_regression_is_visible(tmp_path):
    report = await evaluate(
        Settings(_env_file=None, retrieval_sparse_weight=1, data_dir=tmp_path),
        ROOT / "data/golden/corpus",
        ROOT / "data/golden/retrieval.jsonl",
    )
    assert report["gate_passed"] is False
    assert any(not strategy["gate_passed"] for strategy in report["strategies"])


async def test_all_canonical_golden_ids_exist():
    docs = await load_directory(ROOT / "data/golden/corpus")
    chunks = {c.chunk_id for doc in docs for c in MarkdownChunker().split(doc)}
    rows = load_golden(ROOT / "data/golden/retrieval.jsonl")
    assert all(
        row["relevant_chunk_ids"] and set(row["relevant_chunk_ids"]) <= chunks for row in rows
    )


def test_offline_cli_returns_json_and_failing_gate(tmp_path, capsys, monkeypatch):
    from aether.eval.cli import main

    monkeypatch.setattr(
        "sys.argv",
        ["aether-eval", "retrieval", "--gate", "--output", str(tmp_path / "report.json")],
    )
    main()
    assert "gate=PASS" in capsys.readouterr().out
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["gate_passed"]
    monkeypatch.setenv("AETHER_RETRIEVAL_SPARSE_WEIGHT", "1")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


async def test_loader_failures_and_utf8_txt(tmp_path):
    from aether.retrieval.loaders import LoaderError, load_file

    with pytest.raises(LoaderError):
        await load_directory(tmp_path)
    unsupported = tmp_path / "data.pdf"
    unsupported.write_bytes(b"invalid")
    with pytest.raises(LoaderError):
        await load_file(unsupported, tmp_path)
    unsupported.with_suffix(".txt").write_bytes(b"\xff\xfe")
    with pytest.raises(LoaderError):
        await load_directory(tmp_path)
    unsupported.with_suffix(".txt").write_text("异步文本", encoding="utf-8-sig")
    docs = await load_directory(tmp_path)
    assert docs[0].text == "异步文本"
    assert docs[0].source == "data.txt"


def test_network_access_is_blocked():
    import socket

    with pytest.raises(AssertionError, match="network"):
        socket.create_connection(("example.com", 443))
    left, right = socket.socketpair()
    left.close()
    right.close()
