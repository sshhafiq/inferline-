"""OpenAI-compatible HTTP API.

Endpoints:

* ``POST /v1/chat/completions`` — chat completions, streaming or not.
* ``POST /v1/completions`` — legacy completions, streaming or not.
* ``GET /v1/models`` — model listing.
* ``GET /health`` — liveness probe (includes queue depth).
* ``GET /metrics`` — Prometheus exposition.

Streaming uses server-sent events in the exact OpenAI chunk format, terminated
by ``data: [DONE]``.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from inferline.auth import ANONYMOUS, make_auth_dependency
from inferline.batching import BatchingPolicy, Priority
from inferline.config import Settings
from inferline.metrics import MetricsRegistry
from inferline.ratelimit import RateLimiter
from inferline.scheduler import (
    GenerationRequest,
    MockBackend,
    Scheduler,
    TokenEvent,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# OpenAI-compatible schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "inferline-mock-1b"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


class CompletionRequest(BaseModel):
    model: str = "inferline-mock-1b"
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def render_prompt(messages: list[ChatMessage]) -> str:
    """Flatten chat messages into a single prompt string for the backend."""
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _sse(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    settings = settings or Settings()
    metrics = MetricsRegistry()
    backend = MockBackend(
        latency_per_token_ms=settings.mock.latency_per_token_ms,
        vocab_size=settings.mock.vocab_size,
    )
    policy = BatchingPolicy(
        max_tokens_per_batch=settings.scheduler.max_tokens_per_batch,
        max_wait_ms=settings.scheduler.max_wait_ms,
    )
    scheduler_holder: dict[str, Scheduler] = {}
    limiter = RateLimiter(
        rate=settings.rate_limit.requests_per_second,
        burst=settings.rate_limit.burst,
        enabled=settings.rate_limit.enabled,
    )
    verify_key = make_auth_dependency(settings.api_keys)

    async def rate_limit(key: str = Depends(verify_key)) -> str:
        if not limiter.allow(key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )
        return key

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler = Scheduler(backend=backend, policy=policy, metrics=metrics)
        await scheduler.start()
        scheduler_holder["scheduler"] = scheduler
        app.state.scheduler = scheduler
        app.state.metrics = metrics
        app.state.settings = settings
        yield
        await scheduler.aclose()

    app = FastAPI(title="inferline", version="0.1.0", lifespan=lifespan)

    def get_scheduler() -> Scheduler:
        return scheduler_holder["scheduler"]

    def check_capacity(scheduler: Scheduler) -> None:
        if scheduler.queue_depth >= settings.scheduler.max_queue_depth:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server overloaded, try again later",
            )

    async def _chat_stream(
        completion_id: str,
        model: str,
        created: int,
        events: AsyncIterator[TokenEvent],
    ) -> AsyncIterator[bytes]:
        import json

        def chunk(delta: dict, finish_reason: str | None) -> bytes:
            return _sse(
                json.dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                )
            )

        # Role announcement first; no token is consumed for it.
        yield chunk({"role": "assistant"}, None)
        async for event in events:
            if event.is_last:
                break
            yield chunk({"content": event.token}, None)
        yield chunk({}, "stop")
        yield b"data: [DONE]\n\n"

    async def _completion_stream(
        completion_id: str,
        model: str,
        created: int,
        events: AsyncIterator[TokenEvent],
    ) -> AsyncIterator[bytes]:
        import json

        async for event in events:
            if event.is_last:
                break
            chunk = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "text": event.token, "finish_reason": None}
                ],
            }
            yield _sse(json.dumps(chunk))
        final = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
        }
        yield _sse(json.dumps(final))
        yield b"data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
        key: str = Depends(rate_limit),
    ):
        scheduler = get_scheduler()
        check_capacity(scheduler)
        prompt = render_prompt(body.messages)
        gen_request = GenerationRequest(
            request_id=_new_id("req"),
            prompt=prompt,
            prompt_tokens=estimate_tokens(prompt),
            max_new_tokens=body.max_tokens,
            priority=Priority.INTERACTIVE if body.stream else Priority.BATCH,
        )
        metrics.requests_total.inc(1, "chat_completions", "accepted")
        created = int(time.time())
        completion_id = _new_id("chatcmpl")
        model = settings.model_name
        if body.stream:
            events = scheduler.stream(gen_request)
            return StreamingResponse(
                _chat_stream(completion_id, model, created, events),
                media_type="text/event-stream",
            )
        text = await scheduler.generate(gen_request)
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(text)
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model,
            choices=[ChatChoice(message=ChoiceMessage(content=text))],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    @app.post("/v1/completions")
    async def completions(
        body: CompletionRequest,
        request: Request,
        key: str = Depends(rate_limit),
    ):
        scheduler = get_scheduler()
        check_capacity(scheduler)
        gen_request = GenerationRequest(
            request_id=_new_id("req"),
            prompt=body.prompt,
            prompt_tokens=estimate_tokens(body.prompt),
            max_new_tokens=body.max_tokens,
            priority=Priority.INTERACTIVE if body.stream else Priority.BATCH,
        )
        metrics.requests_total.inc(1, "completions", "accepted")
        created = int(time.time())
        completion_id = _new_id("cmpl")
        model = settings.model_name
        if body.stream:
            events = scheduler.stream(gen_request)
            return StreamingResponse(
                _completion_stream(completion_id, model, created, events),
                media_type="text/event-stream",
            )
        text = await scheduler.generate(gen_request)
        prompt_tokens = estimate_tokens(body.prompt)
        completion_tokens = estimate_tokens(text)
        return CompletionResponse(
            id=completion_id,
            created=created,
            model=model,
            choices=[CompletionChoice(text=text)],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    @app.get("/v1/models")
    async def list_models(key: str = Depends(rate_limit)):
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "inferline",
                }
            ],
        }

    @app.get("/health")
    async def health():
        scheduler = get_scheduler()
        return {
            "status": "ok",
            "model": settings.model_name,
            "queue_depth": scheduler.queue_depth,
        }

    @app.get("/metrics")
    async def prometheus_metrics():
        return PlainTextResponse(
            metrics.render(), media_type="text/plain; version=0.0.4"
        )

    # expose internals for tests and operability
    app.state._metrics = metrics
    app.state._limiter = limiter
    app.state._settings = settings

    return app


def main() -> None:  # pragma: no cover - entry point
    import uvicorn
    from pathlib import Path

    config_path = Path("config.yaml")
    settings = Settings.load(config_path if config_path.exists() else None)
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
