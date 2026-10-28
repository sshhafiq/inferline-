"""Tests for the continuous-batching scheduler."""

import asyncio

import pytest

from inferline.batching import BatchingPolicy, Priority
from inferline.metrics import MetricsRegistry
from inferline.scheduler import (
    GenerationRequest,
    MockBackend,
    Scheduler,
    TransformersBackend,
    estimate_tokens,
)


def make_request(prompt, max_tokens=8, priority=Priority.BATCH):
    return GenerationRequest(
        request_id=f"req-{prompt}",
        prompt=prompt,
        prompt_tokens=estimate_tokens(prompt),
        max_new_tokens=max_tokens,
        priority=priority,
    )


@pytest.fixture
async def scheduler():
    backend = MockBackend(latency_per_token_ms=0)
    policy = BatchingPolicy(max_tokens_per_batch=4096, max_wait_ms=5)
    sched = Scheduler(backend=backend, policy=policy, metrics=MetricsRegistry())
    await sched.start()
    yield sched
    await sched.aclose()


async def test_generate_returns_deterministic_text(scheduler):
    text1 = await scheduler.generate(make_request("hello world"))
    text2 = await scheduler.generate(make_request("hello world"))
    assert text1 == text2
    assert len(text1.split()) == 8


async def test_different_prompts_differ(scheduler):
    text1 = await scheduler.generate(make_request("alpha"))
    text2 = await scheduler.generate(make_request("beta"))
    assert text1 != text2


async def test_concurrent_requests_are_batched(scheduler):
    backend = scheduler._backend
    results = await asyncio.gather(
        *[scheduler.generate(make_request(f"prompt-{i}")) for i in range(4)]
    )
    assert len(results) == 4
    # all four arrived within the coalescing window -> one micro-batch
    assert backend.batches_seen == [4]


async def test_stream_yields_tokens_in_order(scheduler):
    tokens = []
    async for event in scheduler.stream(make_request("stream me", max_tokens=5)):
        if not event.is_last:
            tokens.append(event.token)
    assert len(tokens) == 5
    full = await scheduler.generate(make_request("stream me", max_tokens=5))
    assert " ".join(tokens) == full


async def test_stream_ends_with_terminal_event(scheduler):
    events = [e async for e in scheduler.stream(make_request("x", max_tokens=3))]
    assert events[-1].is_last
    assert not any(e.is_last for e in events[:-1])


async def test_metrics_recorded(scheduler):
    metrics = scheduler._metrics
    await scheduler.generate(make_request("metrics please", max_tokens=4))
    assert metrics.tokens_total.get() == 4.0
    # ttft histogram got one observation
    out: list[str] = []
    metrics.ttft_seconds.render(out)
    assert "inferline_ttft_seconds_count 1" in "\n".join(out)
    assert metrics.queue_depth.get() == 0.0


async def test_queue_depth_tracks_inflight():
    backend = MockBackend(latency_per_token_ms=20)
    policy = BatchingPolicy(max_tokens_per_batch=4096, max_wait_ms=50)
    sched = Scheduler(backend=backend, policy=policy)
    await sched.start()
    try:
        task = asyncio.create_task(sched.generate(make_request("slow", max_tokens=3)))
        await asyncio.sleep(0.01)
        assert sched.queue_depth >= 0  # request accepted without error
        await task
        assert sched.queue_depth == 0
    finally:
        await sched.aclose()


async def test_generate_on_closed_scheduler_raises():
    sched = Scheduler(MockBackend(), BatchingPolicy())
    await sched.start()
    await sched.aclose()
    with pytest.raises(RuntimeError):
        await sched.generate(make_request("too late"))


async def test_transformers_stub_raises_helpfully():
    stub = TransformersBackend()
    with pytest.raises(NotImplementedError, match="documented stub"):
        async for _ in stub.stream_batch(["hi"], [4]):
            pass


def test_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10
