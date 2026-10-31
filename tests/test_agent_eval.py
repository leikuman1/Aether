"""对抗黄金集：语料内没有答案时弃答，可回答集必须引用真实 chunk。"""

import json
from pathlib import Path
from typing import override

import pytest
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aether.config import Settings
from aether.eval.agent import ABSTAIN_RATE_MIN, evaluate_agent, load_jsonl
from aether.graph.citations import extract_chunk_ids
from aether.graph.grading import propose_support
from aether.graph.llm import EvidenceBoundChatModel, create_chat_model, parse_prompt
from aether.retrieval.chunking import MarkdownChunker
from aether.retrieval.loaders import load_directory

ROOT = Path("data/golden")
ADVERSARIAL = ROOT / "agent_adversarial.jsonl"
ANSWERABLE = ROOT / "agent_answerable.jsonl"
CORPUS = ROOT / "corpus"
FAKE_ID = "ffffffffffffffffffffffffffff"


class HostileChat(FakeChatModel):
    """故意声称证据足够，并编造 chunk id。门禁必须丢掉这些引用。"""

    @override
    def _call(self, messages, stop=None, run_manager=None, **kwargs) -> str:
        task, payload = parse_prompt(messages[-1].content)
        if task == "plan":
            return json.dumps({"query": payload.get("question", "")}, ensure_ascii=False)
        if task == "rewrite":
            return json.dumps(
                {"query": f"{payload.get('question', '')} elasticsearch 8.11"},
                ensure_ascii=False,
            )
        if task == "grade":
            return json.dumps(
                {
                    "sufficient": True,
                    "supporting": [
                        {
                            "chunk_id": FAKE_ID,
                            "quote": "this sentence was never retrieved from the corpus",
                        }
                    ],
                    "reason": "trust-me",
                },
                ensure_ascii=False,
            )
        if task == "generate":
            return json.dumps({"answer": f"Paris is the capital [{FAKE_ID}]"})
        raise AssertionError(task)

    @override
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = AIMessage(content=self._call(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.mark.asyncio
async def test_adversarial_items_have_no_corpus_support() -> None:
    docs = await load_directory(CORPUS)
    hits = [
        {"chunk_id": chunk.chunk_id, "text": chunk.text}
        for doc in docs
        for chunk in MarkdownChunker().split(doc)
    ]
    adversarial = load_jsonl(ADVERSARIAL)
    answerable = load_jsonl(ANSWERABLE)
    assert len(adversarial) >= 20
    assert len(answerable) >= 10
    for row in adversarial:
        assert row["expect"] == "abstain"
        assert propose_support(row["query"], hits, min_terms=4, min_quote_chars=12) == []
    for row in answerable:
        assert row["expect"] == "answer"
        gold_ids = set(row["relevant_chunk_ids"])
        gold = [hit for hit in hits if hit["chunk_id"] in gold_ids]
        assert propose_support(row["query"], gold, min_terms=4, min_quote_chars=12)


@pytest.mark.asyncio
async def test_agent_gate_abstains_and_cites_real_chunks(settings: Settings) -> None:
    report = await evaluate_agent(settings, CORPUS, ADVERSARIAL, ANSWERABLE)
    assert report["adversarial"] >= 20
    assert report["answerable"] >= 10
    assert report["abstain_rate"] >= ABSTAIN_RATE_MIN
    assert report["answer_rate"] >= ABSTAIN_RATE_MIN
    assert report["budget_exhausted_rate"] >= ABSTAIN_RATE_MIN
    assert report["fabricated_citations"] == 0
    assert report["integrity_ok"] is True
    assert report["gate_passed"] is True
    assert report["gold_citation_rate"] >= ABSTAIN_RATE_MIN
    adversarial = [item for item in report["items"] if item["expect"] == "abstain"]
    assert {item["rewrite_count"] for item in adversarial} == {settings.agent_rewrite_budget}
    assert {item["attempts"] for item in adversarial} == {settings.agent_rewrite_budget + 1}
    for item in report["items"]:
        assert extract_chunk_ids(item["answer"]) == (
            item["citations"] if item["decision"] == "answer" else []
        )
        if item["decision"] == "answer":
            assert set(item["citations"]) <= set(item["retrieved_ids"])
            assert set(item["citations"]) & set(item["relevant_chunk_ids"])


@pytest.mark.asyncio
async def test_hostile_fake_cannot_emit_fabricated_citations(settings: Settings) -> None:
    report = await evaluate_agent(
        settings,
        CORPUS,
        ADVERSARIAL,
        ANSWERABLE,
        llm=HostileChat(),
    )
    assert isinstance(report, dict)
    adversarial = [item for item in report["items"] if item["expect"] == "abstain"]
    assert adversarial
    assert all(item["decision"] == "abstain" for item in adversarial)
    assert report["fabricated_citations"] == 0
    for item in report["items"]:
        assert FAKE_ID not in item["answer"]
        assert extract_chunk_ids(item["answer"]) == []
    # 可回答题在敌对模型下也不能靠编造 id 蒙混过关。
    assert all(item["decision"] == "abstain" for item in report["items"])


def test_offline_chat_model_is_fake_without_api_key(settings: Settings) -> None:
    model = create_chat_model(settings)
    assert isinstance(model, EvidenceBoundChatModel)
    assert isinstance(model, FakeChatModel)
    live = create_chat_model(
        settings.model_copy(
            update={
                "llm_api_key": "sk-test",
                "llm_base_url": "https://example.invalid/v1",
            }
        )
    )
    assert type(live).__name__ == "ChatOpenAI"
    assert live.openai_api_base == "https://example.invalid/v1"
