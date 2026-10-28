"""Tests for the FastAPI server: OpenAI schema compat, SSE, auth, limits."""

import json

import pytest
from fastapi.testclient import TestClient

from inferline.config import Settings
from inferline.server import create_app, render_prompt
from inferline.scheduler import estimate_tokens


def make_settings(**overrides):
    base = dict(
        mock={"latency_per_token_ms": 0.0},
        rate_limit={"enabled": True, "requests_per_second": 1000.0, "burst": 1000},
    )
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base:
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return Settings.model_validate(base)


@pytest.fixture
def client():
    app = create_app(make_settings())
    with TestClient(app) as c:
        yield c


def chat_payload(**overrides):
    payload = {
        "model": "inferline-mock-1b",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 8,
    }
    payload.update(overrides)
    return payload


def test_chat_completions_schema(client):
    resp = client.post("/v1/chat/completions", json=chat_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert len(body["choices"][0]["message"]["content"].split()) == 8
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens"] > 0


def test_chat_completions_streaming_format(client):
    with client.stream(
        "POST", "/v1/chat/completions", json=chat_payload(stream=True)
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = [line for line in resp.iter_lines() if line]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[len("data: "):]) for line in lines[:-1]]
    # first chunk carries the role, the rest carry content
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[0]["object"] == "chat.completion.chunk"
    for chunk in chunks[1:-1]:
        assert "content" in chunk["choices"][0]["delta"]
        assert chunk["choices"][0]["finish_reason"] is None
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    streamed = " ".join(
        c["choices"][0]["delta"]["content"]
        for c in chunks
        if "content" in c["choices"][0]["delta"]
    )
    # deterministic backend: streaming matches non-streaming output
    plain = client.post("/v1/chat/completions", json=chat_payload()).json()
    assert streamed == plain["choices"][0]["message"]["content"]


def test_completions_endpoint(client):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Once upon a time", "max_tokens": 6},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "text_completion"
    assert len(body["choices"][0]["text"].split()) == 6
    assert body["choices"][0]["finish_reason"] == "stop"


def test_completions_streaming_ends_with_done(client):
    with client.stream(
        "POST",
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 4, "stream": True},
    ) as resp:
        lines = [line for line in resp.iter_lines() if line]
    assert lines[-1] == "data: [DONE]"
    assert len(lines) == 4 + 1 + 1  # 4 tokens + final chunk + DONE


def test_models_endpoint(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "inferline-mock-1b"


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["queue_depth"], int)


def test_metrics_endpoint(client):
    client.post("/v1/chat/completions", json=chat_payload())
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "inferline_requests_total" in text
    assert "inferline_tokens_total" in text
    assert "# TYPE inferline_batch_size histogram" in text


def test_auth_rejects_bad_key():
    app = create_app(make_settings(api_keys=["good-key"]))
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 401
        resp = client.post(
            "/v1/chat/completions",
            json=chat_payload(),
            headers={"Authorization": "Bearer good-key"},
        )
        assert resp.status_code == 200


def test_rate_limiter_returns_429():
    app = create_app(
        make_settings(rate_limit={"requests_per_second": 1.0, "burst": 1})
    )
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 200
        assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 429


def test_validation_errors():
    app = create_app(make_settings())
    with TestClient(app) as client:
        # empty messages rejected by pydantic
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [], "max_tokens": 8},
        )
        assert resp.status_code == 422


def test_render_prompt():
    from inferline.server import ChatMessage

    prompt = render_prompt(
        [
            ChatMessage(role="system", content="Be nice"),
            ChatMessage(role="user", content="Hi"),
        ]
    )
    assert prompt == "system: Be nice\nuser: Hi"
