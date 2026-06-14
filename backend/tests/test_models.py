"""Tests for model API key lookup."""

from __future__ import annotations

from simple_agent.models import get_api_key


def test_get_api_key_reads_project_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=from_file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert get_api_key("deepseek") == "from_file"


def test_get_api_key_prefers_process_env_over_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=from_file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from_process")

    assert get_api_key("deepseek") == "from_process"


def test_get_api_key_strips_quotes_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('DEEPSEEK_API_KEY="quoted_value"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert get_api_key("deepseek") == "quoted_value"
