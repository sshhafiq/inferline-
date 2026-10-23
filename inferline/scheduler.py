"""Continuous-batching scheduler.

The scheduler owns a single background task that:

1. waits for queued requests,
2. coalesces arrivals for up to ``max_wait_ms``,
3. packs a micro-batch with :class:`inferline.batching.BatchingPolicy`,
4. runs one forward pass per decode step through a pluggable
   :class:`Backend`, fanning tokens out to each request's stream,
5. records TTFT / throughput / batch-size metrics.

This is the same architecture vLLM and TensorRT-LLM use: many concurrent
requests share each decode step instead of each request occupying the
"GPU" alone.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from typing import AsyncIterator, Protocol

from inferline.batching import BatchingPolicy, Priority, QueuedRequest
from inferline.metrics import MetricsRegistry


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used for batch budgeting."""
    return max(1, len(text) // 4)


@dataclass
class GenerationRequest(QueuedRequest):
    """A generation request; ``prompt`` rides along with the queue metadata."""

    prompt: str = ""


@dataclass
class TokenEvent:
    """One streamed token (or the terminal event) for a request."""

    token: str
    is_last: bool = False


class Backend(Protocol):
    """Pluggable model backend.

    Implementations receive a micro-batch of prompts and yield, per decode
    step, the ``(batch_index, token)`` pairs produced in that step. Batch
    indices refer to positions in the ``prompts`` list passed in.
    """

    name: str

    def stream_batch(
        self, prompts: list[str], max_tokens: list[int]
    ) -> AsyncIterator[list[tuple[int, str]]]:
        """Yield per-step ``(batch_index, token)`` lists until all done."""
        ...  # pragma: no cover - protocol


class MockBackend:
    """Deterministic stand-in backend — no GPU or weights required.

    For each prompt it derives a repeatable pseudo-token sequence from the
    SHA-256 of the prompt text, then streams one decode step at a time with
    ``latency_per_token_ms`` of ``asyncio.sleep`` injected per step to emulate
    GPU time. Two identical prompts always produce identical output, which
    makes the scheduler and the HTTP layer fully testable.
    """

    name = "mock"

    _WORDS = (
        "the quick brown fox jumps over lazy dogs while models dream of "
        "electric tokens streaming through silicon valleys of thought and "
        "quiet inference under starlit servers".split()
    )

    def __init__(
        self, latency_per_token_ms: float = 5.0, vocab_size: int = 512
    ) -> None:
        self.latency_per_token_ms = latency_per_token_ms
        self.vocab_size = vocab_size
        #: Batch sizes seen, in dispatch order — useful for tests/demos.
        self.batches_seen: list[int] = []

    async def stream_batch(
        self, prompts: list[str], max_tokens: list[int]
    ) -> AsyncIterator[list[tuple[int, str]]]:
        sequences = [
            self._sequence_for(prompt, n) for prompt, n in zip(prompts, max_tokens)
        ]
        self.batches_seen.append(len(prompts))
        steps = max((len(s) for s in sequences), default=0)
        for step in range(steps):
            if self.latency_per_token_ms > 0:
                await asyncio.sleep(self.latency_per_token_ms / 1000.0)
            yield [
                (i, seq[step]) for i, seq in enumerate(sequences) if step < len(seq)
            ]

    def _sequence_for(self, prompt: str, n: int) -> list[str]:
        seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8], "big")
        rng = random.Random(seed)
        words = self._WORDS
        return [rng.choice(words) for _ in range(max(0, n))]


class TransformersBackend:
    """Stub for a real Hugging Face ``transformers`` backend.

    To wire up real weights, install ``torch`` and ``transformers`` and replace
    this stub with an implementation shaped like::

        from transformers import AutoModelForCausalLM, AutoTokenizer

        class TransformersBackend:
            name = "hf"

            def __init__(self, model_id: str):
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype="auto", device_map="auto"
                )

            async def stream_batch(self, prompts, max_tokens):
                # NOTE: real continuous batching needs paged KV-cache
                # management (see vLLM). A faithful minimal version runs one
                # batched `model.generate` per micro-batch here and replays
                # the tokens step-by-step; true per-step batching is left as
                # an exercise.
                ...

    The rest of inferline (scheduler, server, metrics) works unchanged
    against any :class:`Backend`.
    """

    name = "transformers-stub"

    def __init__(self, model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> None:
        self.model_id = model_id

    async def stream_batch(
        self, prompts: list[str], max_tokens: list[int]
    ) -> AsyncIterator[list[tuple[int, str]]]:
        raise NotImplementedError(
            "TransformersBackend is a documented stub: install torch and "
            "transformers and implement stream_batch() per the class docstring."
        )
        yield  # make this an async generator


@dataclass
class _Inflight:
    request: GenerationRequest
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    first_token_sent: bool = False
    tokens_streamed: int = 0


class Scheduler:
    """Async continuous-batching scheduler over a :class:`Backend`."""

    def __init__(
        self,
        backend: Backend,
        policy: BatchingPolicy,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._backend = backend
        self._policy = policy
        self._metrics = metrics or MetricsRegistry()
        self._pending: deque[_Inflight] = deque()
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> None:
        """Launch the background scheduling loop."""
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="inferline-scheduler")

    async def aclose(self) -> None:
        """Stop the loop and fail any requests still waiting."""
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        while self._pending:
            inflight = self._pending.popleft()
            inflight.queue.put_nowait(TokenEvent(token="", is_last=True))

    @property
    def queue_depth(self) -> int:
        return len(self._pending)

    def _enqueue(self, request: GenerationRequest) -> _Inflight:
        if self._closed:
            raise RuntimeError("scheduler is closed")
        inflight = _Inflight(request=request)
        self._pending.append(inflight)
        self._metrics.queue_depth.inc()
        self._wakeup.set()
        return inflight

    async def generate(self, request: GenerationRequest) -> str:
        """Generate a full completion (non-streaming)."""
        parts: list[str] = []
        async for event in self.stream(request):
            if not event.is_last:
                parts.append(event.token)
        # The mock backend emits word tokens; real backends emit subwords
        # that already carry their own spacing.
        return " ".join(parts)

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[TokenEvent]:
        """Stream tokens for one request as they are decoded."""
        inflight = self._enqueue(request)
        while True:
            event = await inflight.queue.get()
            yield event
            if event.is_last:
                return

    async def _loop(self) -> None:
        wait_s = self._policy.max_wait_ms / 1000.0
        while True:
            if not self._pending:
                await self._wakeup.wait()
                self._wakeup.clear()
            else:
                # Coalescing window: let near-simultaneous arrivals join.
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=wait_s)
                except asyncio.TimeoutError:
                    pass
                finally:
                    self._wakeup.clear()
            batch = self._policy.form_batch(
                [inf.request for inf in self._pending]
            )
            if not batch:
                continue
            wanted = {id(r) for r in batch}
            inflight_batch = [inf for inf in self._pending if id(inf.request) in wanted]
            for inf in inflight_batch:
                self._pending.remove(inf)
            await self._dispatch(inflight_batch)

    async def _dispatch(self, batch: list[_Inflight]) -> None:
        prompts = [inf.request.prompt for inf in batch]
        limits = [inf.request.max_new_tokens for inf in batch]
        self._metrics.batch_size.observe(len(batch))
        started = monotonic()
        total_tokens = 0
        try:
            async for step_events in self._backend.stream_batch(prompts, limits):
                for index, token in step_events:
                    inflight = batch[index]
                    if not inflight.first_token_sent:
                        inflight.first_token_sent = True
                        self._metrics.ttft_seconds.observe(
                            monotonic() - inflight.request.arrival_time
                        )
                    inflight.tokens_streamed += 1
                    total_tokens += 1
                    inflight.queue.put_nowait(TokenEvent(token=token))
        finally:
            elapsed = monotonic() - started
            if elapsed > 0:
                self._metrics.tokens_per_second.set(total_tokens / elapsed)
            self._metrics.tokens_total.inc(total_tokens)
            for inflight in batch:
                inflight.queue.put_nowait(TokenEvent(token="", is_last=True))
                self._metrics.request_latency_seconds.observe(
                    monotonic() - inflight.request.arrival_time
                )
                self._metrics.queue_depth.dec()
