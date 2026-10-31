from fastapi.testclient import TestClient

from aether.api.app import create_app
from aether.config import Settings


def make_client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def test_healthz_reports_version_and_llm_flag(settings: Settings) -> None:
    with make_client(settings) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == settings.version
    assert body["llm_configured"] is False


def test_readyz_ensures_data_dir_writable(settings: Settings) -> None:
    with make_client(settings) as client:
        resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}
    assert settings.data_dir.exists()
