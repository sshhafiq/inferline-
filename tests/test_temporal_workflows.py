"""Workflow tests against Temporal's time-skipping test environment.

Skipped automatically if the test-server binary cannot be downloaded
(offline sandbox); the activity tests cover the shard logic directly.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from inferline.temporal.activities import InferenceActivities
from inferline.temporal.models import (
    BatchInferenceInput,
    BatchPolicySpec,
    InferenceRequestSpec,
)
from inferline.temporal.workflows import BatchInferenceWorkflow

pytestmark = pytest.mark.asyncio

TASK_QUEUE = "test-inferline"


async def _start_env() -> WorkflowEnvironment:
    try:
        return await asyncio.wait_for(
            WorkflowEnvironment.start_time_skipping(), timeout=90
        )
    except Exception as exc:
        pytest.skip(f"Temporal test server unavailable: {exc}")


def _worker(env: WorkflowEnvironment) -> Worker:
    return Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[BatchInferenceWorkflow],
        activities=InferenceActivities().all(),
    )


def _requests(n: int, tokens: int = 8):
    return [
        InferenceRequestSpec(
            request_id=f"r{i}", prompt=f"prompt number {i}",
            max_new_tokens=tokens,
        )
        for i in range(n)
    ]


async def test_batch_job_completes_and_aggregates():
    env = await _start_env()
    async with env:
        async with _worker(env):
            job_id = f"t-{uuid.uuid4().hex[:8]}"
            result = await env.client.execute_workflow(
                BatchInferenceWorkflow.run,
                BatchInferenceInput(
                    job_id=job_id,
                    requests=_requests(10),
                    policy=BatchPolicySpec(max_tokens_per_batch=512,
                                           max_wait_ms=5),
                    shard_max_requests=4,
                ),
                id=job_id,
                task_queue=TASK_QUEUE,
            )
            assert result.status == "completed"
            assert result.requests_completed == 10
            assert result.tokens_total == 80
            assert result.batches_dispatched >= 3  # 3 shards, >=1 batch each
            assert [r.request_id for r in result.results] == [
                f"r{i}" for i in range(10)
            ]


async def test_batch_job_progress_query():
    env = await _start_env()
    async with env:
        async with _worker(env):
            job_id = f"t-{uuid.uuid4().hex[:8]}"
            handle = await env.client.start_workflow(
                BatchInferenceWorkflow.run,
                BatchInferenceInput(
                    job_id=job_id,
                    requests=_requests(6),
                    policy=BatchPolicySpec(max_tokens_per_batch=512,
                                           max_wait_ms=5),
                    shard_max_requests=3,
                ),
                id=job_id,
                task_queue=TASK_QUEUE,
            )
            result = await handle.result()
            assert result.status == "completed"
            progress = await handle.query(BatchInferenceWorkflow.progress)
            assert progress.total_requests == 6
            assert progress.shards_total == 2
            assert progress.shards_completed == 2


async def test_batch_job_cancellation_between_waves():
    env = await _start_env()
    async with env:
        async with _worker(env):
            job_id = f"t-{uuid.uuid4().hex[:8]}"
            handle = await env.client.start_workflow(
                BatchInferenceWorkflow.run,
                BatchInferenceInput(
                    job_id=job_id,
                    # Slow shards (20ms x 48 tokens) in 4 waves of 1, so
                    # the cancel signal lands while later waves pend.
                    requests=_requests(8, tokens=48),
                    policy=BatchPolicySpec(max_tokens_per_batch=512,
                                           max_wait_ms=5),
                    shard_max_requests=2,
                    shard_parallelism=1,
                    backend_latency_per_token_ms=20.0,
                ),
                id=job_id,
                task_queue=TASK_QUEUE,
            )
            await asyncio.sleep(0.2)
            await handle.signal(BatchInferenceWorkflow.cancel_job)
            result = await handle.result()
            assert result.status == "cancelled"
            assert result.requests_cancelled > 0
            assert result.requests_completed + result.requests_cancelled == 8
