"""整张图由 FakeChatModel / FakeListChatModel 驱动：预算耗尽、编造引用、提供者故障。"""

import json
from typing import override

import pytest
from langchain_core.language_models.fake_chat_models import FakeChatModel, FakeListChatModel

from aether.graph.citations import extract_chunk_ids, fabricated_citation_ids
from aether.graph.llm import EvidenceBoundChatModel
from aether.graph.runner import run_turn
from aether.retrieval.types import Chunk, RetrievedChunk

CID = "0123456789abcdef0123"
FAKE = "ffffffffffffffffffffffffffff"
QUOTE = "quasar rollback pointer switches blue green traffic after the health check"
QUESTION = "quasar rollback pointer switches blue green traffic"


class CountingList(FakeListChatModel):
    calls: int = 0

    @override
    def _call(self, *args, **kwargs) -> str:
        self.calls += 1
        return super()._call(*args, **kwargs)


class DownModel(FakeChatModel):
    @override
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("provider 500")

    @override
    def _call(self, *args, **kwargs) -> str:
        raise RuntimeError("provider 500")


class _Searcher:
    def __init__(self, hits):
        self.hits = hits
        self.queries: list[str] = []

    async def search(self, query: str, k: int = 5):
        self.queries.append(query)
        return list(self.hits)[:k]


def _retrieved() -> RetrievedChunk:
    chunk = Chunk(CID, "quasar", "quasar.md", 0, "Release", QUOTE)
    return RetrievedChunk(chunk, 1.0, {"bm25": 1.0})


def _limits(tmp_path, name: str) -> dict:
    return {
        "checkpoint_path": tmp_path / f"{name}.sqlite",
        "thread_id": name,
        "rewrite_budget": 2,
        "min_support_terms": 2,
        "min_quote_chars": 12,
        "retrieval_k": 4,
    }


@pytest.mark.asyncio
async def test_fake_list_model_answers_only_with_real_citation(tmp_path) -> None:
    model = CountingList(
        responses=[
            json.dumps({"query": QUESTION}),
            json.dumps(
                {
                    "sufficient": True,
                    "supporting": [{"chunk_id": CID, "quote": QUOTE}],
                    "reason": "quoted",
                }
            ),
            json.dumps({"answer": f"{QUOTE} [{CID}]"}),
        ]
    )
    result = await run_turn(
        question=QUESTION,
        searcher=_Searcher([_retrieved()]),
        llm=model,
        require_review=False,
        **_limits(tmp_path, "answer"),
    )
    assert result["decision"] == "answer"
    assert result["citations"] == [CID]
    assert result["citations"] == extract_chunk_ids(result["answer"])
    assert fabricated_citation_ids(result["answer"], set(result["retrieved_ids"])) == []
    assert result["rewrite_count"] == 0
    assert model.calls == 3
    assert result["messages"][-1]["decision"] == "answer"


@pytest.mark.asyncio
async def test_fake_list_model_answers_with_subspan_quote(tmp_path) -> None:
    text = f"Deployment runbook. {QUOTE}. Keep the previous deployment available."
    hit = RetrievedChunk(Chunk(CID, "quasar", "quasar.md", 0, "Release", text), 1.0, {})
    model = FakeListChatModel(
        responses=[
            json.dumps({"query": QUESTION}),
            json.dumps(
                {
                    "sufficient": True,
                    "supporting": [{"chunk_id": CID, "quote": QUOTE}],
                }
            ),
            json.dumps({"answer": f"{QUOTE} [{CID}]"}),
        ]
    )
    result = await run_turn(
        question=QUESTION,
        searcher=_Searcher([hit]),
        llm=model,
        **_limits(tmp_path, "subspan"),
    )
    assert result["decision"] == "answer"
    assert result["verified"] is True
    assert result["answer"] == f"{QUOTE} [{CID}]"
    assert result["citations"] == [CID]


@pytest.mark.asyncio
async def test_offline_model_retains_multiple_supporting_chunks(tmp_path) -> None:
    other_id = "abcdef0123456789abcd"
    other_quote = "quasar rollback checks deployment health before switching traffic"
    other_hit = RetrievedChunk(
        Chunk(other_id, "quasar", "quasar.md", 1, "Health", other_quote), 0.9, {}
    )
    result = await run_turn(
        question=QUESTION,
        searcher=_Searcher([_retrieved(), other_hit]),
        llm=EvidenceBoundChatModel(min_support_terms=2),
        **_limits(tmp_path, "multi-evidence"),
    )
    assert result["decision"] == "answer"
    assert set(result["citations"]) == {CID, other_id}
    assert QUOTE in result["answer"]
    assert other_quote in result["answer"]


@pytest.mark.asyncio
async def test_rewrite_budget_exhausted_abstains_without_answering(tmp_path) -> None:
    insufficient = json.dumps({"sufficient": False, "supporting": [], "reason": "no-overlap"})
    model = CountingList(
        responses=[
            json.dumps({"query": QUESTION}),
            insufficient,
            json.dumps({"query": "quasar rollback pointer"}),
            insufficient,
            json.dumps({"query": "blue green traffic"}),
            insufficient,
        ]
    )
    searcher = _Searcher([_retrieved()])
    result = await run_turn(
        question=QUESTION,
        searcher=searcher,
        llm=model,
        **_limits(tmp_path, "budget"),
    )
    assert result["decision"] == "abstain"
    assert result["abstain_reason"] == "budget_exhausted"
    assert result["rewrite_count"] == 2
    assert len(result["attempts"]) == 3
    assert {item["grade"] for item in result["attempts"]} == {"insufficient"}
    assert len(searcher.queries) == 3
    assert model.calls == 6
    assert extract_chunk_ids(result["answer"]) == []
    assert result["verified"] is False


@pytest.mark.asyncio
async def test_zero_budget_abstains_on_first_insufficient_grade(tmp_path) -> None:
    model = CountingList(
        responses=[
            json.dumps({"query": QUESTION}),
            json.dumps({"sufficient": False, "supporting": [], "reason": "no"}),
        ]
    )
    limits = _limits(tmp_path, "zero")
    limits["rewrite_budget"] = 0
    result = await run_turn(
        question=QUESTION,
        searcher=_Searcher([_retrieved()]),
        llm=model,
        **limits,
    )
    assert result["abstain_reason"] == "budget_exhausted"
    assert result["rewrite_count"] == 0
    assert len(result["attempts"]) == 1
    assert model.calls == 2


@pytest.mark.asyncio
async def test_fabricated_citation_is_not_emitted(tmp_path) -> None:
    model = FakeListChatModel(
        responses=[
            json.dumps({"query": QUESTION}),
            json.dumps(
                {
                    "sufficient": True,
                    "supporting": [{"chunk_id": CID, "quote": QUOTE}],
                    "reason": "quoted",
                }
            ),
            json.dumps({"answer": f"The secret code is 42 [{FAKE}]"}),
        ]
    )
    result = await run_turn(
        question=QUESTION,
        searcher=_Searcher([_retrieved()]),
        llm=model,
        **_limits(tmp_path, "fake-id"),
    )
    assert result["decision"] == "abstain"
    assert result["abstain_reason"] == "citation_integrity"
    assert result["citation_violations"] == [FAKE]
    assert FAKE not in result["answer"]
    assert extract_chunk_ids(result["answer"]) == []
    assert fabricated_citation_ids(result["answer"], set(result["retrieved_ids"])) == []


@pytest.mark.asyncio
async def test_real_id_with_extra_claim_abstains(tmp_path) -> None:
    model = FakeListChatModel(
        responses=[
            json.dumps({"query": QUESTION}),
            json.dumps(
                {
                    "sufficient": True,
                    "supporting": [{"chunk_id": CID, "quote": QUOTE}],
                    "reason": "quoted",
                }
            ),
            json.dumps({"answer": f"Paris is the capital. {QUOTE} [{CID}]"}),
        ]
    )
    result = await run_turn(
        question=QUESTION,
        searcher=_Searcher([_retrieved()]),
        llm=model,
        **_limits(tmp_path, "ungrounded"),
    )
    assert result["abstain_reason"] == "ungrounded"
    assert CID not in result["answer"]


@pytest.mark.asyncio
async def test_provider_error_abstains(tmp_path) -> None:
    result = await run_turn(
        question=QUESTION,
        searcher=_Searcher([_retrieved()]),
        llm=DownModel(),
        **_limits(tmp_path, "down"),
    )
    assert result["decision"] == "abstain"
    assert result["abstain_reason"] == "provider_error"
    assert "500" not in result["answer"]
    assert extract_chunk_ids(result["answer"]) == []


def test_down_model_is_fake_chat_model() -> None:
    assert isinstance(DownModel(), FakeChatModel)
