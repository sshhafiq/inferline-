"""The batch-inference workflow.

``BatchInferenceWorkflow`` turns a list of prompts into a durable job:

1. **Shard** — requests are packed into shards under a request-count and
   a token budget (the same worst-case cost model as
   :class:`inferline.batching.BatchingPolicy`).
2. **Fan out** — shard activities run in parallel waves of
   ``shard_parallelism``; each activity drives a real continuous-batching
   :class:`~inferline.scheduler.Scheduler`.
3. **Cancel** — the ``cancel_job`` signal is honoured between waves:
   shards not yet started never run, their requests are reported as
   ``cancelled``, and the job returns the partial results gathered so
   far instead of failing.
4. **Aggregate** — per-request results are reassembled in input order and
   job-level totals (tokens, mean latency, batch count) are computed
   from the shard reports.

Progress is observable at any time through the ``progress`` query.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from inferline.temporal.activities import InferenceActivities
    from inferline.temporal.models import (
        BatchInferenceInput,
        BatchInferenceResult,
        BatchProgress,
        InferenceRequestSpec,
        RequestResult,
        RunShardInput,
        ShardResult,
        ShardSpec,
    )

_SHARD_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
)


def shard_requests(
    requests: list[InferenceRequestSpec],
    max_requests: int,
    token_budget: int,
) -> list[ShardSpec]:
    """Greedily pack requests into shards (pure, deterministic).

    A shard closes when adding the next request would exceed either the
    request-count cap or the token budget; a single oversized request
    gets a shard of its own so it can never starve.
    """
    shards: list[ShardSpec] = []
    current: list[InferenceRequestSpec] = []
    current_cost = 0
    for spec in requests:
        cost = spec.estimated_token_cost()
        if current and (
            len(current) >= max_requests or current_cost + cost > token_budget
        ):
            shards.append(ShardSpec(shard_index=len(shards), requests=current))
            current, current_cost = [], 0
        current.append(spec)
        current_cost += cost
    if current:
        shards.append(ShardSpec(shard_index=len(shards), requests=current))
    return shards


@workflow.defn
class BatchInferenceWorkflow:
    """Durable fan-out of inference shards with cancellation."""

    def __init__(self) -> None:
        self._job_id = ""
        self._total = 0
        self._completed = 0
        self._shards_total = 0
        self._shards_completed = 0
        self._cancelled = False

    @workflow.signal
    def cancel_job(self) -> None:
        """Request cancellation; honoured between shard waves."""
        self._cancelled = True

    @workflow.query
    def progress(self) -> BatchProgress:
        return BatchProgress(
            job_id=self._job_id,
            total_requests=self._total,
            completed_requests=self._completed,
            shards_total=self._shards_total,
            shards_completed=self._shards_completed,
            cancelled=self._cancelled,
        )

    @workflow.run
    async def run(self, input: BatchInferenceInput) -> BatchInferenceResult:
        self._job_id = input.job_id
        self._total = len(input.requests)
        shards = shard_requests(
            input.requests, input.shard_max_requests, input.shard_token_budget
        )
        self._shards_total = len(shards)

        shard_results: list[ShardResult] = []
        waves = [
            shards[i : i + input.shard_parallelism]
            for i in range(0, len(shards), input.shard_parallelism)
        ]
        for wave in waves:
            if self._cancelled:
                break
            wave_results = await asyncio.gather(
                *(
                    workflow.execute_activity(
                        InferenceActivities.run_inference_shard,
                        RunShardInput(
                            job_id=input.job_id,
                            shard=shard,
                            policy=input.policy,
                            tenant=input.tenant,
                            rate_limit_rate=input.rate_limit_rate,
                            rate_limit_burst=input.rate_limit_burst,
                            backend_latency_per_token_ms=(
                                input.backend_latency_per_token_ms
                            ),
                        ),
                        start_to_close_timeout=timedelta(minutes=10),
                        heartbeat_timeout=timedelta(seconds=30),
                        retry_policy=_SHARD_RETRY,
                    )
                    for shard in wave
                )
            )
            shard_results.extend(wave_results)
            self._shards_completed += len(wave_results)
            self._completed += sum(len(r.results) for r in wave_results)

        # Reassemble in the caller's request order; requests in shards
        # that never ran (cancellation) are reported as cancelled.
        by_id: dict[str, RequestResult] = {}
        tokens_total = 0
        batches = 0
        for shard_result in shard_results:
            tokens_total += shard_result.tokens_total
            batches += len(shard_result.batch_sizes)
            for result in shard_result.results:
                by_id[result.request_id] = result
        ordered: list[RequestResult] = []
        for spec in input.requests:
            ordered.append(
                by_id.get(
                    spec.request_id,
                    RequestResult(request_id=spec.request_id,
                                  status="cancelled"),
                )
            )

        completed = [r for r in ordered if r.status == "completed"]
        mean_latency = (
            sum(r.latency_s for r in completed) / len(completed)
            if completed else 0.0
        )
        return BatchInferenceResult(
            job_id=input.job_id,
            status="cancelled" if self._cancelled else "completed",
            results=ordered,
            requests_completed=len(completed),
            requests_rate_limited=sum(
                1 for r in ordered if r.status == "rate_limited"
            ),
            requests_failed=sum(1 for r in ordered if r.status == "failed"),
            requests_cancelled=sum(
                1 for r in ordered if r.status == "cancelled"
            ),
            tokens_total=tokens_total,
            mean_latency_s=mean_latency,
            batches_dispatched=batches,
        )
