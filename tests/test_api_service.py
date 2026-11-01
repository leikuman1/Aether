"""M3：完整图驱动 SSE、独立进程重启、provider 500、超时及核算回归。"""

import asyncio
import json
import os
import subprocess
import sys
from decimal import Decimal

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from aether.api.app import create_app
from aether.graph.llm import EvidenceBoundChatModel, parse_prompt

QUESTION = "quasar rollback pointer switches blue green traffic"
CORPUS = (
    "# Quasar\n\n## Release pointer\n\n"
    "quasar rollback pointer switches blue green traffic after the health check "
    "passes and keeps the previous deployment available for rollback.\n"
)


def events(response):
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    result = []
    for frame in response.text.strip().split("\n\n"):
        name, data = frame.split("\n", 1)
        result.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return result


def done(response):
    return events(response)[-1][1]


def metric(text, key):
    return Decimal(
        next(line.split()[-1] for line in text.splitlines() if line.startswith(key + " "))
    )


@pytest.fixture
def api_settings(settings):
    root = settings.data_dir / "corpus"
    root.mkdir()
    (root / "quasar.md").write_text(CORPUS, encoding="utf-8")
    return settings


def ingest(client, settings):
    response = client.post("/v1/ingest", json={"corpus_dir": str(settings.data_dir / "corpus")})
    assert response.status_code == 200, response.text
    assert response.json()["chunks"] > 0


class MeteredModel(EvidenceBoundChatModel):
    async def _agenerate(self, messages, **kwargs):
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=self._respond(messages),
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 25,
                            "total_tokens": 125,
                        },
                    )
                )
            ]
        )


def test_sse_usage_headers_metrics_and_docs(api_settings):
    # 非零价格 + provider usage；不只断言 Fake 的零账单。
    api_settings.llm_api_key = "local-fixture-key"
    api_settings.llm_input_cost_per_million = 2
    api_settings.llm_output_cost_per_million = 8
    with TestClient(create_app(api_settings)) as client:
        client.app.state.service.llm = MeteredModel()
        ingest(client, api_settings)
        responses = [client.post("/v1/chat", json={"question": QUESTION}) for _ in range(2)]
        for response in responses:
            frames = events(response)
            final = frames[-1][1]
            stages = [data["stage"] for name, data in frames if name == "stage"]
            assert stages == ["plan", "retrieve", "grade", "review", "generate", "verify"]
            assert all(data["elapsed_ms"] >= 0 for name, data in frames if name == "stage")
            assert final["decision"] == "answer"
            assert final["verified"]
            assert (
                "".join(data["text"] for name, data in frames if name == "token") == final["answer"]
            )
            assert final["usage"] == {
                "input_tokens": 300,
                "output_tokens": 75,
                "total_tokens": 375,
                "cost_usd": "0.0012",
                "source": "provider",
            }
            assert int(response.headers["x-aether-tokens"]) == final["usage"]["total_tokens"]
            assert response.headers["x-aether-cost-usd"] == final["usage"]["cost_usd"]
            assert response.headers["x-aether-thread-id"] == final["thread_id"]
            assert response.headers["x-aether-trace-id"] == final["trace_id"]
        metrics = client.get("/metrics").text
        assert metric(metrics, 'aether_chat_requests_total{status="success"}') == 2
        assert metric(metrics, 'aether_tokens_total{direction="input"}') == 600
        assert metric(metrics, 'aether_tokens_total{direction="output"}') == 150
        assert metric(metrics, "aether_cost_usd_total") == sum(
            Decimal(response.headers["x-aether-cost-usd"]) for response in responses
        )
        assert metric(metrics, 'aether_stage_duration_seconds_count{stage="generate"}') == 2
        for path in ("/healthz", "/readyz", "/metrics", "/docs", "/openapi.json"):
            assert client.get(path).status_code == 200
        schema = client.get("/openapi.json").json()
        operation = schema["paths"]["/v1/chat"]["post"]
        assert "text/event-stream" in operation["responses"]["200"]["content"]
        assert "503" in operation["responses"]


def test_rewrite_and_abstain_usage_counts_all_calls(api_settings):
    api_settings.llm_api_key = "local-fixture-key"
    api_settings.llm_input_cost_per_million = 2
    api_settings.llm_output_cost_per_million = 8
    with TestClient(create_app(api_settings)) as client:
        client.app.state.service.llm = MeteredModel()
        ingest(client, api_settings)
        response = client.post("/v1/chat", json={"question": "quasar rollback alien 9999"})
        final = done(response)
        assert final["abstain_reason"] == "budget_exhausted"
        assert final["rewrite_count"] == 2
        # plan + 3 grade + 2 rewrite，检索无语义支持但仍有命中。
        assert final["usage"]["input_tokens"] == 600
        assert final["usage"]["output_tokens"] == 150
        assert Decimal(response.headers["x-aether-cost-usd"]) == Decimal("0.0024")
        assert metric(client.get("/metrics").text, "aether_cost_usd_total") == Decimal("0.0024")


async def test_real_inflight_thread_conflict_and_request_usage_isolation(api_settings):
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockingMeteredModel(MeteredModel):
        async def _agenerate(self, messages, **kwargs):
            entered.set()
            await release.wait()
            return await super()._agenerate(messages, **kwargs)

    api_settings.llm_api_key = "local-fixture-key"
    app = create_app(api_settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a") as client,
    ):
        await client.post("/v1/ingest", json={"corpus_dir": str(api_settings.data_dir / "corpus")})
        app.state.service.llm = BlockingMeteredModel()
        first = asyncio.create_task(
            client.post("/v1/chat", json={"question": QUESTION, "thread_id": "inflight"})
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            conflict = await client.post(
                "/v1/chat", json={"question": QUESTION, "thread_id": "inflight"}
            )
            assert conflict.status_code == 409
            assert conflict.json()["error"]["code"] == "thread_busy"
            assert conflict.headers["x-aether-tokens"] == "0"
            release.set()
            second = await client.post("/v1/chat", json={"question": QUESTION})
            completed = await asyncio.wait_for(first, timeout=10)
        finally:
            release.set()
            if not first.done():
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
        assert done(completed)["usage"]["total_tokens"] == 375
        assert done(second)["usage"]["total_tokens"] == 375
        text = (await client.get("/metrics")).text
        assert metric(text, 'aether_chat_requests_total{status="success"}') == 2
        assert metric(text, 'aether_chat_requests_total{status="error"}') == 1
        assert metric(text, 'aether_tokens_total{direction="input"}') == 600


@pytest.mark.parametrize("failed_task", ["plan", "grade", "generate"])
def test_provider_http_500_is_structured_503_without_leak(api_settings, failed_task):
    reference = EvidenceBoundChatModel()

    def provider(request):
        payload = json.loads(request.content)
        messages = payload["messages"]
        task, _ = parse_prompt(messages[-1]["content"])
        if task == failed_task:
            return httpx.Response(
                500, json={"error": {"message": "SECRET stack trace provider body"}}
            )
        from langchain_core.messages import HumanMessage

        content = reference._respond([HumanMessage(content=messages[-1]["content"])])
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "mock",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as upstream:
            app = create_app(api_settings)
            async with app.router.lifespan_context(app):
                app.state.service.llm = ChatOpenAI(
                    model="mock",
                    api_key="test",
                    base_url="https://provider.invalid/v1",
                    http_async_client=upstream,
                    max_retries=0,
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://aether"
                ) as client:
                    built = await client.post(
                        "/v1/ingest", json={"corpus_dir": str(api_settings.data_dir / "corpus")}
                    )
                    assert built.status_code == 200
                    response = await client.post("/v1/chat", json={"question": QUESTION})
                    assert response.status_code == 503
                    assert response.json()["error"]["code"] == "provider_unavailable"
                    assert response.headers["content-type"].startswith("application/json")
                    assert "SECRET" not in response.text
                    assert "Traceback" not in response.text
                    assert "provider.invalid" not in response.text
                    assert "stack trace" not in response.text
                    assert "event: token" not in response.text
                    usage = response.json()["usage"]
                    assert int(response.headers["x-aether-tokens"]) == usage["total_tokens"]
                    metrics = (await client.get("/metrics")).text
                    assert metric(metrics, 'aether_chat_requests_total{status="error"}') == 1
                    assert (
                        metric(metrics, 'aether_tokens_total{direction="input"}')
                        == usage["input_tokens"]
                    )
                    assert not app.state.service.active_threads

    asyncio.run(run())


def test_missing_index_validation_and_conflicts(api_settings):
    with TestClient(create_app(api_settings)) as client:
        response = client.post("/v1/chat", json={"question": QUESTION})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "index_not_ready"
        for body in (
            {},
            {"question": "   "},
            {"question": QUESTION, "resume": {"approved": True}},
            {"resume": {"approved": "false"}},
            {"question": QUESTION, "thread_id": "../x"},
        ):
            assert client.post("/v1/chat", json=body).status_code == 422
        outside = client.post("/v1/ingest", json={"corpus_dir": str(api_settings.data_dir.parent)})
        assert outside.status_code == 422
        ingest(client, api_settings)
        assert client.get("/v1/threads/unknown").status_code == 404
        assert (
            client.post(
                "/v1/chat", json={"thread_id": "x", "resume": {"approved": True}}
            ).status_code
            == 409
        )
        paused = done(
            client.post(
                "/v1/chat", json={"question": QUESTION, "thread_id": "x", "require_review": True}
            )
        )
        assert paused["interrupted"]
        assert (
            client.post("/v1/chat", json={"question": QUESTION, "thread_id": "x"}).status_code
            == 409
        )
        refused = done(
            client.post("/v1/chat", json={"thread_id": "x", "resume": {"approved": False}})
        )
        assert refused["abstain_reason"] == "human_rejected"


async def test_timeout_releases_thread_and_busy_guards(api_settings):
    class SlowModel(EvidenceBoundChatModel):
        async def _agenerate(self, messages, **kwargs):
            await asyncio.sleep(5)
            return await super()._agenerate(messages, **kwargs)

    app = create_app(api_settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a") as client,
    ):
        await client.post("/v1/ingest", json={"corpus_dir": str(api_settings.data_dir / "corpus")})
        service = app.state.service
        service.llm = SlowModel()
        api_settings.api_request_timeout = 0.1
        failed = await client.post("/v1/chat", json={"question": QUESTION, "thread_id": "slow"})
        assert failed.status_code == 504
        assert failed.json()["error"]["code"] == "request_timeout"
        assert not service.active_threads
        service.llm = EvidenceBoundChatModel()
        api_settings.api_request_timeout = 10
        previous = await client.get("/v1/threads/slow")
        assert previous.json()["interrupted"] is False
        retried = await client.post("/v1/chat", json={"question": QUESTION, "thread_id": "slow"})
        assert done(retried)["decision"] == "answer"
        service.active_threads.add("busy")
        conflict = await client.post("/v1/chat", json={"question": QUESTION, "thread_id": "busy"})
        assert conflict.status_code == 409
        api_settings.api_max_concurrency = 1
        full = await client.post("/v1/chat", json={"question": QUESTION})
        assert full.status_code == 429
        ingest_busy = await client.post(
            "/v1/ingest", json={"corpus_dir": str(api_settings.data_dir / "corpus")}
        )
        assert ingest_busy.status_code == 409
        service.active_threads.clear()
        service.ingesting = True
        updating = await client.post("/v1/chat", json={"question": QUESTION})
        assert updating.status_code == 503
        assert updating.json()["error"]["code"] == "index_updating"


def test_ingest_failure_keeps_existing_index(api_settings):
    with TestClient(create_app(api_settings)) as client:
        ingest(client, api_settings)
        (api_settings.data_dir / "corpus" / "quasar.md").write_text("", encoding="utf-8")
        failed = client.post(
            "/v1/ingest", json={"corpus_dir": str(api_settings.data_dir / "corpus")}
        )
        assert failed.status_code == 422
        assert not client.app.state.service.ingesting
        final = done(client.post("/v1/chat", json={"question": QUESTION}))
        assert final["decision"] == "answer"
        assert final["usage"]["source"] == "estimated"
        assert Decimal(final["usage"]["cost_usd"]) == 0


# 每次调用启动全新的解释器、ASGI 服务 lifespan 与 SQLite 连接。子进程同样阻断网络。
WORKER = """
import json, socket, sys
from unittest.mock import patch
from fastapi.testclient import TestClient
from aether.api.app import create_app
from aether.config import Settings
original_pair = socket.socketpair
original_connect = socket.socket.connect
def deny(*a, **k):
    raise AssertionError("no network")
def pair(*a, **k):
    with patch.object(socket.socket, "connect", original_connect):
        return original_pair(*a, **k)
socket.socket.connect = deny
socket.create_connection = deny
socket.socketpair = pair
settings = Settings(_env_file=None, data_dir=sys.argv[1], llm_api_key="", embedding_provider="fake")
with TestClient(create_app(settings)) as client:
    action = sys.argv[2]
    question = sys.argv[3]
    if action == "start":
        built = client.post("/v1/ingest", json={
            "corpus_dir": str(settings.data_dir / "corpus")})
        assert built.status_code == 200
        body = {"question": question, "thread_id": "restart", "require_review": True}
    elif action == "resume":
        previous = client.get("/v1/threads/restart").json()
        assert previous["interrupted"]
        body = {"thread_id": "restart", "resume": {"approved": True}}
    else:
        previous = client.get("/v1/threads/restart").json()
        assert len(previous["messages"]) == 2
        body = {"question": question, "thread_id": "restart"}
    response = client.post("/v1/chat", json=body)
    assert response.status_code == 200, response.text
    frame = response.text.strip().split("\\n\\n")[-1]
    print(frame.split("data: ", 1)[1])
"""


def test_service_restores_thread_across_process_restarts(api_settings):
    env = {key: value for key, value in os.environ.items() if not key.startswith("AETHER_")}
    env["PYTHONIOENCODING"] = "utf-8"
    results = []
    for action in ("start", "resume", "follow"):
        proc = subprocess.run(
            [sys.executable, "-c", WORKER, str(api_settings.data_dir), action, QUESTION],
            env=env,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        results.append(json.loads(proc.stdout))
    paused, resumed, followed = results
    assert paused["interrupted"]
    assert resumed["decision"] == "answer"
    assert paused["interrupt"]["supporting_ids"][0] in resumed["citations"]
    assert followed["decision"] == "answer"
    assert len(followed["messages"]) == 4
    assert followed["messages"][:2] == resumed["messages"]
