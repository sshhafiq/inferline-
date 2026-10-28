"""Tests for configuration loading."""

import os

import pytest
import yaml

from inferline.config import Settings


def test_defaults():
    s = Settings()
    assert s.server.port == 8000
    assert s.scheduler.max_tokens_per_batch == 2048
    assert s.rate_limit.enabled is True
    assert s.api_keys == []


def test_yaml_loading(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "server": {"port": 9000},
                "scheduler": {"max_tokens_per_batch": 4096},
                "api_keys": ["k1", "k2"],
            }
        )
    )
    s = Settings.load(path)
    assert s.server.port == 9000
    assert s.scheduler.max_tokens_per_batch == 4096
    assert s.api_keys == ["k1", "k2"]


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("INFERLINE_SERVER__PORT", "9001")
    monkeypatch.setenv("INFERLINE_SCHEDULER__MAX_TOKENS_PER_BATCH", "512")
    monkeypatch.setenv("INFERLINE_RATE_LIMIT__ENABLED", "false")
    monkeypatch.setenv("INFERLINE_API_KEYS", "a,b,c")
    s = Settings.load()
    assert s.server.port == 9001
    assert s.scheduler.max_tokens_per_batch == 512
    assert s.rate_limit.enabled is False
    assert s.api_keys == ["a", "b", "c"]


def test_env_beats_yaml(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"server": {"port": 9000}}))
    monkeypatch.setenv("INFERLINE_SERVER__PORT", "9002")
    assert Settings.load(path).server.port == 9002


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        Settings.load("/does/not/exist.yaml")
