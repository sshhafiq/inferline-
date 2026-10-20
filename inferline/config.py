"""Configuration loading for inferline.

Settings come from three layers, later layers winning:

1. Built-in defaults (see the models below).
2. A YAML file, e.g. ``config.yaml``.
3. Environment variables prefixed with ``INFERLINE_``. Nested keys use a
   double underscore, e.g. ``INFERLINE_SCHEDULER__MAX_TOKENS_PER_BATCH=4096``.
   Comma-separated values are parsed as lists (handy for ``INFERLINE_API_KEYS``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ServerSettings(BaseModel):
    """HTTP server settings."""

    host: str = "0.0.0.0"
    port: int = 8000


class SchedulerSettings(BaseModel):
    """Continuous-batching scheduler settings."""

    max_tokens_per_batch: int = Field(
        default=2048,
        description="Token budget per micro-batch (prompt + completion tokens).",
    )
    max_wait_ms: int = Field(
        default=20,
        description="How long the scheduler waits to fill a batch before dispatching.",
    )
    max_queue_depth: int = Field(
        default=1024,
        description="Requests beyond this are rejected with 429.",
    )


class RateLimitSettings(BaseModel):
    """Per-API-key token-bucket settings."""

    enabled: bool = True
    requests_per_second: float = 10.0
    burst: int = 20


class MockBackendSettings(BaseModel):
    """Deterministic stand-in backend settings (no GPU required)."""

    latency_per_token_ms: float = Field(
        default=5.0, description="Simulated per-token decode latency."
    )
    vocab_size: int = 512


class Settings(BaseModel):
    """Top-level inferline settings."""

    server: ServerSettings = Field(default_factory=ServerSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    mock: MockBackendSettings = Field(default_factory=MockBackendSettings)
    model_name: str = "inferline-mock-1b"
    api_keys: list[str] = Field(
        default_factory=list,
        description="Accepted Bearer API keys. Empty means auth is disabled.",
    )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        """Load settings from an optional YAML file plus environment overrides."""
        data: dict[str, Any] = {}
        if path is not None:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        data = _apply_env_overrides(data)
        return cls.model_validate(data)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``INFERLINE_*`` environment variables onto a settings dict."""
    import os

    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in data.items()}
    prefix = "INFERLINE_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = [p.lower() for p in env_key[len(prefix) :].split("__")]
        _set_nested(out, parts, _coerce(env_val))
    return out


def _set_nested(data: dict[str, Any], parts: list[str], value: Any) -> None:
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _coerce(value: str) -> Any:
    """Best-effort conversion of an env string to int/float/bool/list."""
    if "," in value:
        return [_coerce_scalar(v) for v in value.split(",")]
    return _coerce_scalar(value)


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
