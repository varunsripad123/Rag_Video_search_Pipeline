from pathlib import Path

from src.config import load_config


def test_load_config(tmp_path: Path, monkeypatch):
    config_content = """
project:
  name: Test
logging:
  json: false
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(config_content)
    monkeypatch.setenv("RAG_CONFIG_PATH", str(cfg_path))
    load_config.cache_clear()
    config = load_config()
    assert config.project.name == "Test"
    assert config.logging.json_output is False
    monkeypatch.setenv("RAG_LOGGING__LEVEL", "\"DEBUG\"")
    load_config.cache_clear()
    config = load_config()
    assert config.logging.level == "DEBUG"
