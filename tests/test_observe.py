"""trace_id、分阶段时延和成本：日志与 GET /v1/traces 指向同一最慢阶段。"""

import asyncio
import json
import logging
from decimal import Decimal

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aether.api.app import create_app
from aether.api.observe import TraceStore, annotate_stage, main, slowest_from_log, slowest_stage
from aether.graph.llm import EvidenceBoundChatModel

QUESTION = "quasar rollback pointer switches blue green traffic"
CORPUS = (
    "# Quasar\n\n## Release pointer\n\n"
    "quasar rollback pointer switches blue green traffic after the health check "
    "passes and keeps the previous deployment available for rollback.\n"
)


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


def test_slowest_stage_breaks_ties_toward_the_later_row() -> None:
    stages = [
        annotate_stage("plan", 1, {"input_tokens": 0, "output_tokens": 0, "cost_usd": "0"}),
        annotate_stage("retrieve", 1, {"input_tokens": 0, "output_tokens": 0, "cost_usd": "0"}),
    ]
    assert slowest_stage(stages) == ("retrieve", 1.0)


def test_trace_store_drops_the_oldest_record() -> None:
    store = TraceStore(capacity=2)
    for number in range(3):
        store.put(
            {
                "trace_id": f"{number:032x}",
                "stages": [{"stage": "http", "elapsed_ms": number}],
                "usage": {"cost_usd": "0"},
            }
        )
    assert store.get(f"{0:032x}") is None
    latest = store.get(f"{2:032x}")
    assert latest is not None
    latest["stages"][0]["stage"] = "mutated"
    assert store.get(f"{2:032x}")["stages"][0]["stage"] == "http"


def test_slowest_from_log_uses_stage_lines_for_one_trace(tmp_path) -> None:
    wanted = "ab" * 16
    other = "cd" * 16
    lines = [
        "INFO uvicorn started",
        json.dumps(
            {
                "event": "stage",
                "trace_id": wanted,
                "stage": "plan",
                "elapsed_ms": 1.5,
                "input_tokens": 100,
                "output_tokens": 25,
                "cost_usd": "0.0004",
            }
        ),
        json.dumps(
            {
                "event": "stage",
                "trace_id": other,
                "stage": "retrieve",
                "elapsed_ms": 50,
                "cost_usd": "0",
            }
        ),
        "prefix "
        + json.dumps(
            {
                "event": "stage",
                "trace_id": wanted,
                "stage": "retrieve",
                "elapsed_ms": 9.25,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": "0",
            }
        ),
        json.dumps(
            {
                "event": "chat_complete",
                "trace_id": wanted,
                "slowest_stage": "retrieve",
                "slowest_elapsed_ms": 9.25,
                "cost_usd": "0.0004",
            }
        ),
    ]
    path = tmp_path / "server.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    found = slowest_from_log(path.read_text(encoding="utf-8"), wanted)
    assert found["slowest_stage"] == "retrieve"
    assert found["slowest_elapsed_ms"] == 9.25
    assert found["cost_usd"] == "0.0004"
    assert main(["--log", str(path), "--trace-id", wanted]) is None


def test_health_and_missing_index_share_the_response_trace_id(settings) -> None:
    with TestClient(create_app(settings)) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        trace_id = health.headers["x-aether-trace-id"]
        body = client.get(f"/v1/traces/{trace_id}")
        assert body.status_code == 200
        assert body.json()["slowest_stage"] == "http"
        assert body.json()["usage"]["cost_usd"] == "0"
        missing = client.post("/v1/chat", json={"question": QUESTION})
        assert missing.status_code == 503
        assert missing.json()["error"]["code"] == "index_not_ready"
        assert missing.json()["error"]["trace_id"] == missing.headers["x-aether-trace-id"]
        traced = client.get(f"/v1/traces/{missing.headers['x-aether-trace-id']}")
        assert traced.json()["slowest_stage"] == "request"
        assert client.get(f"/v1/traces/{'ab' * 16}").status_code == 404
        assert client.get("/v1/traces/not-a-trace").status_code == 404


def test_chat_trace_names_the_slowest_stage_and_splits_cost(settings, caplog) -> None:
    root = settings.data_dir / "corpus"
    root.mkdir()
    (root / "quasar.md").write_text(CORPUS, encoding="utf-8")
    settings.llm_api_key = "local-fixture-key"
    settings.llm_input_cost_per_million = 2
    settings.llm_output_cost_per_million = 8
    caplog.set_level(logging.INFO, logger="aether")
    with TestClient(create_app(settings)) as client:
        client.app.state.service.llm = MeteredModel()
        assert client.post("/v1/ingest", json={"corpus_dir": str(root)}).status_code == 200
        response = client.post("/v1/chat", json={"question": QUESTION, "thread_id": "cost"})
        assert response.status_code == 200
        frames = []
        for block in response.text.strip().split("\n\n"):
            name, data = block.split("\n", 1)
            frames.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
        stages = [data for name, data in frames if name == "stage"]
        expected = max(enumerate(stages), key=lambda pair: (pair[1]["elapsed_ms"], pair[0]))[1]
        done = frames[-1][1]
        trace_id = response.headers["x-aether-trace-id"]
        assert done["trace_id"] == trace_id
        assert done["slowest_stage"] == expected["stage"]
        traced = client.get(f"/v1/traces/{trace_id}").json()
        assert traced["slowest_stage"] == expected["stage"]
        assert traced["slowest_elapsed_ms"] == expected["elapsed_ms"]
        assert sum(Decimal(stage["cost_usd"]) for stage in traced["stages"]) == Decimal(
            traced["usage"]["cost_usd"]
        )
        assert Decimal(traced["usage"]["cost_usd"]) == Decimal("0.0012")
        logged = [
            json.loads(record.message)
            for record in caplog.records
            if record.name.startswith("aether") and record.message.startswith("{")
        ]
        stage_logs = [
            item
            for item in logged
            if item.get("event") == "stage" and item.get("trace_id") == trace_id
        ]
        assert stage_logs
        assert {"elapsed_ms", "cost_usd", "input_tokens", "output_tokens"} <= stage_logs[0].keys()
        from_log = slowest_from_log("\n".join(json.dumps(item) for item in logged), trace_id)
        assert from_log["slowest_stage"] == traced["slowest_stage"]


def test_unexpected_error_keeps_trace_header_and_access_status(settings, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="aether")
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:

        async def broken_ingest(body):
            raise OSError("private storage details")

        monkeypatch.setattr(client.app.state.service, "ingest", broken_ingest)
        response = client.post("/v1/ingest", json={"corpus_dir": str(settings.data_dir)})
        assert response.status_code == 503
        trace_id = response.json()["error"]["trace_id"]
        assert response.headers["x-aether-trace-id"] == trace_id
        access = [
            json.loads(record.message)
            for record in caplog.records
            if record.name.startswith("aether") and '"event":"http_access"' in record.message
        ]
        assert access[-1]["status_code"] == 503
        assert access[-1]["trace_id"] == trace_id
        assert "private storage details" not in caplog.text + response.text


@pytest.mark.parametrize("failure", ["timeout", "error", "cancel"])
async def test_failed_retrieve_is_in_trace_after_completed_plan(
    settings, monkeypatch, caplog, failure
):
    from aether.retrieval.hybrid import HybridSearcher

    root = settings.data_dir / "corpus"
    root.mkdir()
    (root / "quasar.md").write_text(CORPUS, encoding="utf-8")
    app = create_app(settings)
    caplog.set_level(logging.INFO, logger="aether")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a") as client,
    ):
        assert (await client.post("/v1/ingest", json={"corpus_dir": str(root)})).status_code == 200
        timeout_scope = asyncio.timeout(None)

        async def stalled_search(self, *args, **kwargs):
            if failure == "timeout":
                timeout_scope.reschedule(asyncio.get_running_loop().time() + 0.1)
                await asyncio.Event().wait()
            await asyncio.sleep(0.1)
            if failure == "cancel":
                request_task.cancel()
                await asyncio.Event().wait()
            raise RuntimeError("private retrieval details")

        monkeypatch.setattr(HybridSearcher, "search", stalled_search)
        with monkeypatch.context() as timing:
            timing.setattr(asyncio, "timeout", lambda _: timeout_scope)
            if failure == "cancel":
                request_task = asyncio.create_task(
                    client.post("/v1/chat", json={"question": QUESTION})
                )
                with pytest.raises(asyncio.CancelledError):
                    await request_task
            else:
                response = await client.post("/v1/chat", json={"question": QUESTION})
                assert response.status_code == (504 if failure == "timeout" else 503)
        completed = [
            json.loads(record.message)
            for record in caplog.records
            if record.name.startswith("aether") and '"event":"chat_complete"' in record.message
        ][-1]
        trace_id = completed["trace_id"]
        traced = app.state.service.traces.get(trace_id)
        assert [row["stage"] for row in traced["stages"]] == ["plan", "retrieve"]
        assert traced["slowest_stage"] == "retrieve"
        assert traced["slowest_elapsed_ms"] >= 90
        logs = "\n".join(record.message for record in caplog.records)
        assert slowest_from_log(logs, trace_id)["slowest_stage"] == "retrieve"
        assert "private retrieval details" not in logs
        assert not app.state.service.active_threads
