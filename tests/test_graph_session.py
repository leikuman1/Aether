"""同一 thread_id 经 SqliteSaver 跨进程恢复：中断、继续、再问一轮。"""

import json
import os
import subprocess
import sys

import pytest

from aether.config import Settings
from aether.retrieval.chunking import MarkdownChunker
from aether.retrieval.embeddings import create_embedder
from aether.retrieval.ingest import ingest_directory

QUESTION = "quasar rollback pointer switches blue green traffic"
CORPUS = (
    "# Quasar\n\n"
    "## Release pointer\n\n"
    "quasar rollback pointer switches blue green traffic after the health check "
    "passes and keeps the previous deployment available for rollback.\n"
)


def _run(args: list[str], env: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "aether.graph.session", *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.asyncio
async def test_thread_restores_across_processes(tmp_path, settings: Settings) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "quasar.md").write_text(CORPUS, encoding="utf-8")
    index = tmp_path / "index.db"
    checkpoint = tmp_path / "sessions.sqlite"
    embedder = create_embedder(settings)
    searcher = await ingest_directory(
        corpus,
        MarkdownChunker(),
        embedder,
        index,
        dense_backend=settings.dense_backend,
    )
    await searcher.aclose()
    if hasattr(embedder, "aclose"):
        await embedder.aclose()

    env = os.environ.copy()
    env.update(
        {
            "AETHER_EMBEDDING_PROVIDER": "fake",
            "AETHER_LLM_API_KEY": "",
            "AETHER_DENSE_BACKEND": settings.dense_backend,
            "AETHER_AGENT_MIN_SUPPORT_TERMS": "4",
            "AETHER_AGENT_REWRITE_BUDGET": "2",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONWARNINGS": "ignore",
        }
    )
    common = [
        "--thread-id",
        "thread-restore",
        "--checkpoint",
        str(checkpoint),
        "--index",
        str(index),
        "--min-support-terms",
        "4",
    ]
    started = _run(["start", *common, "--question", QUESTION, "--require-review"], env)
    assert started["interrupted"] is True
    assert started["next"] == ["review"]
    assert started["interrupt"]["supporting_ids"]
    cited = started["interrupt"]["supporting_ids"][0]

    shown = _run(["show", *common], env)
    assert shown["interrupted"] is True
    assert shown["messages"] == []
    assert shown["thread_id"] == "thread-restore"

    resumed = _run(["resume", *common, "--approve"], env)
    assert resumed["interrupted"] is False
    assert resumed["decision"] == "answer"
    assert cited in resumed["citations"]
    assert cited in resumed["answer"]
    assert resumed["messages"][0]["content"] == QUESTION
    assert resumed["messages"][1]["decision"] == "answer"

    persisted = _run(["show", *common], env)
    assert persisted["messages"] == resumed["messages"]
    assert persisted["answer"] == resumed["answer"]

    follow = _run(
        ["start", *common, "--question", "赤壁之战的交战双方是谁，发生于哪一年？"],
        env,
    )
    assert follow["decision"] == "abstain"
    assert follow["abstain_reason"] == "budget_exhausted"
    assert len(follow["messages"]) == 4
    assert follow["messages"][0]["content"] == QUESTION
    assert follow["messages"][2]["content"].startswith("赤壁之战")
    assert follow["rewrite_count"] == 2
    assert len(follow["attempts"]) == 3
