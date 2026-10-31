"""Embedding behavior with deterministic fakes and an in-process HTTP transport."""

import json
import math
import traceback

import httpx
import pytest

from aether.config import Settings
from aether.retrieval.embeddings import EmbeddingError, FakeEmbedder, OpenAICompatEmbedder


def embedding_settings(**overrides):
    values = {
        "llm_api_key": "test",
        "llm_base_url": "https://embedding.invalid/v1/",
        "embedding_model": "fixture-model",
        "embedding_dim": 3,
        "embedding_batch_size": 2,
        "embedding_retry_delay": 0,
        "embedding_max_retries": 2,
        "embedding_timeout": 0.25,
    }
    return Settings(_env_file=None, **(values | overrides))


async def test_fake_embedder_determinism_norm_empty_text_and_dimensions():
    embedder = FakeEmbedder(32)
    texts = ["中文检索 API", "unrelated storage", "", "  \n，!?", "中文检索 API"]  # noqa: RUF001
    vectors = await embedder.embed_documents(texts)
    assert vectors == await FakeEmbedder(32).embed_documents(texts)
    assert vectors[0] == vectors[-1] == await embedder.embed_query(texts[0])
    assert vectors[0] != vectors[1]
    assert vectors[2] == vectors[3] == [1.0] + [0.0] * 31
    assert all(len(v) == 32 and math.hypot(*v) == pytest.approx(1) for v in vectors)
    assert await embedder.embed_documents([]) == []
    assert embedder.dim == 32 and embedder.name == FakeEmbedder(32).name
    assert embedder.name != FakeEmbedder(16).name
    for dim in [0, -1]:
        with pytest.raises(ValueError):
            FakeEmbedder(dim)


async def test_openai_batching_reorders_response_indices_and_uses_settings():
    batches = []
    texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
    expected = {text: [float(i + 1), 2.0, 3.0] for i, text in enumerate(texts)}

    def respond(request):
        assert request.method == "POST"
        assert str(request.url) == "https://embedding.invalid/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer test"
        assert request.extensions["timeout"]["read"] == 0.25
        payload = json.loads(request.content)
        assert payload["model"] == "fixture-model"
        batch = payload["input"]
        batches.append(batch)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": expected[text]}
                    for i, text in reversed(list(enumerate(batch)))
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        embedder = OpenAICompatEmbedder(embedding_settings(), client=client)
        assert await embedder.embed_documents([]) == []
        assert batches == []
        assert await embedder.embed_documents(texts) == [expected[t] for t in texts]
        assert batches == [["alpha", "beta"], ["gamma", "delta"], ["epsilon"]]
        assert await embedder.embed_query("alpha") == expected["alpha"]
        assert batches[-1] == ["alpha"]
        assert embedder.dim == 3
        assert "fixture-model" in embedder.name
        await embedder.aclose()
        assert not client.is_closed


async def test_openai_retries_429_then_succeeds():
    calls = []

    def respond(request):
        calls.append(request.content)
        if len(calls) == 1:
            return httpx.Response(429, text="private response content")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 2, 3]}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        embedder = OpenAICompatEmbedder(embedding_settings(), client=client)
        assert await embedder.embed_query("fixture input") == [1, 2, 3]
    assert len(calls) == 2 and calls[0] == calls[1]


@pytest.mark.parametrize(
    "failure, expected_calls, expected_status",
    [
        (401, 1, 401),
        (500, 3, 500),
        ("timeout", 3, None),
    ],
)
async def test_openai_retry_budget_and_errors_do_not_leak_payload(
    failure, expected_calls, expected_status, caplog
):
    secret = "PRIVATE_PROVIDER_RESPONSE_7293"
    private_input = "PRIVATE_CORPUS_INPUT_9241"
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        if failure == "timeout":
            raise httpx.ReadTimeout(secret, request=request)
        return httpx.Response(failure, text=secret)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        embedder = OpenAICompatEmbedder(embedding_settings(), client=client)
        with pytest.raises(EmbeddingError) as caught:
            await embedder.embed_query(private_input)
    assert calls == expected_calls
    assert caught.value.status == expected_status
    expected_message = "request rejected" if failure == 401 else "retry budget exhausted"
    assert expected_message in str(caught.value)
    rendered = "".join(traceback.format_exception(caught.value)) + caplog.text
    assert secret not in rendered and private_input not in rendered
    assert "embedding.invalid" not in str(caught.value)


@pytest.mark.parametrize(
    "malformed",
    [
        "invalid-json",
        "missing-index",
        "duplicate-index",
        "out-of-range-index",
        "nan",
        "zero",
        "dimension",
        "boolean-index",
    ],
)
async def test_openai_rejects_malformed_success_without_retry_or_content_leak(malformed, caplog):
    secret = "PRIVATE_PROVIDER_RESPONSE_7293"
    items = [{"index": 0, "embedding": [1, 0, 0]}, {"index": 1, "embedding": [0, 1, 0]}]
    if malformed == "missing-index":
        del items[0]["index"]
    elif malformed == "duplicate-index":
        items[1]["index"] = 0
    elif malformed == "out-of-range-index":
        items[1]["index"] = 2
    elif malformed == "boolean-index":
        items[0]["index"] = False
    elif malformed == "nan":
        items[0]["embedding"] = [float("nan"), 0, 0]
    elif malformed == "zero":
        items[0]["embedding"] = [0, 0, 0]
    elif malformed == "dimension":
        items[0]["embedding"] = [1, 0]
    content = (
        secret
        if malformed == "invalid-json"
        else json.dumps({"data": items, "provider_detail": secret})
    )
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=content, headers={"Content-Type": "application/json"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        embedder = OpenAICompatEmbedder(embedding_settings(), client=client)
        with pytest.raises(EmbeddingError, match="malformed embeddings response") as caught:
            await embedder.embed_documents(["first", "second"])
    assert calls == 1
    rendered = "".join(traceback.format_exception(caught.value)) + caplog.text
    assert secret not in rendered
    assert caught.value.status is None


async def test_openai_requires_key_and_identity_tracks_model_endpoint_and_dimension():
    with pytest.raises(EmbeddingError, match="AETHER_LLM_API_KEY"):
        OpenAICompatEmbedder(embedding_settings(llm_api_key=""))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(500))
    ) as client:
        original = OpenAICompatEmbedder(embedding_settings(), client=client).name
        assert (
            original
            == OpenAICompatEmbedder(
                embedding_settings(llm_base_url="https://embedding.invalid/v1"), client=client
            ).name
        )
        for changes in [
            {"embedding_model": "other-model"},
            {"embedding_dim": 4},
            {"llm_base_url": "https://other.invalid/v1"},
        ]:
            assert (
                original != OpenAICompatEmbedder(embedding_settings(**changes), client=client).name
            )
