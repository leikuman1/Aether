from aether.config import Settings


def test_settings_ignores_working_directory_dotenv(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text(
        "AETHER_LLM_MODEL=from-dotenv\nAETHER_LLM_API_KEY=secret\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.llm_model == "glm-4.6"
    assert settings.llm_api_key == ""


def test_settings_env_prefix(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AETHER_LLM_MODEL", "test-model")
    monkeypatch.setenv("AETHER_AGENT_REWRITE_BUDGET", "1")
    monkeypatch.delenv("AETHER_LLM_API_KEY", raising=False)
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert s.llm_model == "test-model"
    assert s.llm_api_key == ""  # 无密钥默认 = 演练模式，不触网
    assert s.agent_rewrite_budget == 1
    assert s.llm_temperature == 0.0
