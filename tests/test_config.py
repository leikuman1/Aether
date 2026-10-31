from aether.config import Settings


def test_settings_env_prefix(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AETHER_LLM_MODEL", "test-model")
    monkeypatch.delenv("AETHER_LLM_API_KEY", raising=False)
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert s.llm_model == "test-model"
    assert s.llm_api_key == ""  # 无密钥默认 = 演练模式，不触网
