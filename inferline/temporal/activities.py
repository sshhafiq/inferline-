"""Activities for the batch-inference workflow.

One activity runs one shard: it admits the shard's requests through a
per-tenant :class:`~inferline.ratelimit.RateLimiter`, drives them through
a real :class:`~inferline.scheduler.Scheduler` (continuous batching with
the project's :class:`~inferline.batching.BatchingPolicy`), and reports
per-request results plus the shard's slice of the
:class:`~inferline.metrics.MetricsRegistry`.

The activity heartbeats while generation is in flight, so the workflow's
``heartbeat_timeout`` can distinguish a slow shard from a dead worker.
"""

from __future__ import annotations

import asyncio
import time

from temporalio import activity

from inferline.batching import BatchingPolicy, Priority
from inferline.metrics import MetricsRegistry
from inferline.ratelimit import RateLimiter
from inferline.scheduler import GenerationRequest, MockBackend, Scheduler
from inferline.temporal.models import (
    RequestResult,
    RunShardInput,
    ShardResult,
)

_HEARTBEAT_INTERVAL_S = 0.25


class InferenceActivities:
    """Shard-execution activities. Stateless: safe to share per worker."""

    @activity.defn
    async def run_inference_shard(self, input: RunShardInput) -> ShardResult:
        started = time.monotonic()
        shard = input.shard

        policy = BatchingPolicy(
            max_tokens_per_batch=input.policy.max_tokens_per_batch,
            max_wait_ms=input.policy.max_wait_ms,
            max_batch_requests=input.policy.max_batch_requests,
        )
        limiter = RateLimiter(
            rate=input.rate_limit_rate, burst=input.rate_limit_burst
        )
        metrics = MetricsRegistry()
        backend = MockBackend(
            latency_per_token_ms=input.backend_latency_per_token_ms
        )
        scheduler = Scheduler(backend=backend, policy=policy, metrics=metrics)

        # Admission control first: requests the tenant's bucket rejects
        # never reach the scheduler.
        results: dict[str, RequestResult] = {}
        admitted: list[GenerationRequest] = []
        for spec in shard.requests:
            if not limiter.allow(input.tenant, cost=1):
                results[spec.request_id] = RequestResult(
                    request_id=spec.request_id, status="rate_limited",
                    error="tenant rate limit exceeded",
                )
                continue
            admitted.append(
                GenerationRequest(
                    request_id=spec.request_id,
                    prompt_tokens=max(1, len(spec.prompt) // 4),
                    max_new_tokens=spec.max_new_tokens,
                    priority=Priority[spec.priority],
                    prompt=spec.prompt,
                )
            )

        done = 0
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(input.job_id, shard.shard_index,
                                 lambda: done)
        )
        try:
            await scheduler.start()
            outcomes = await asyncio.gather(
                *(self._generate_one(scheduler, req) for req in admitted),
                return_exceptions=True,
            )
        finally:
            heartbeat_task.cancel()
            await scheduler.aclose()

        for req, outcome in zip(admitted, outcomes):
            done += 1
            if isinstance(outcome, BaseException):
                results[req.request_id] = RequestResult(
                    request_id=req.request_id, status="failed",
                    error=f"{type(outcome).__name__}: {outcome}",
                )
            else:
                text, tokens, latency = outcome
                results[req.request_id] = RequestResult(
                    request_id=req.request_id, status="completed",
                    output_text=text, tokens_generated=tokens,
                    latency_s=latency,
                )

        return ShardResult(
            shard_index=shard.shard_index,
            results=[results[spec.request_id] for spec in shard.requests],
            batch_sizes=list(backend.batches_seen),
            tokens_total=int(metrics.tokens_total.total()),
            wall_time_s=time.monotonic() - started,
        )

    # ------------------------------------------------------------------
    @staticmethod
    async def _generate_one(
        scheduler: Scheduler, request: GenerationRequest
    ) -> tuple[str, int, float]:
        """Stream one request to completion, counting tokens + latency."""
        started = time.monotonic()
        parts: list[str] = []
        tokens = 0
        async for event in scheduler.stream(request):
            if not event.is_last:
                parts.append(event.token)
                tokens += 1
        return " ".join(parts), tokens, time.monotonic() - started

    @staticmethod
    async def _heartbeat_loop(job_id: str, shard_index: int, done_fn) -> None:
        while True:
            try:
                activity.heartbeat(
                    {"job_id": job_id, "shard": shard_index, "done": done_fn()}
                )
            except Exception:
                # No activity context (direct unit-test invocation):
                # heartbeats are best-effort progress signals, not data.
                return
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)

    def all(self) -> list:
        return [self.run_inference_shard]
