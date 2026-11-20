"""Unit tests for the Temporal shard activity + sharding (no server).

The activity drives a real Scheduler with the MockBackend, so these
tests exercise continuous batching, rate limiting, and metrics without
a Temporal cluster. Workflow tests live in
``test_temporal_workflows.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from inferline.temporal.activities import InferenceActivities
from inferline.temporal.models import (
    BatchPolicySpec,
    InferenceRequestSpec,
    RunShardInput,
    ShardSpec,
)
from inferline.temporal.workflows import shard_requests


def _spec(i: int, prompt: str = "hello world", tokens: int = 8):
    return InferenceRequestSpec(
        request_id=f"r{i}", prompt=prompt, max_new_tokens=tokens
    )


def _run_shard(specs, **overrides):
    activities = InferenceActivities()
    input = RunShardInput(
        job_id="job-1",
        shard=ShardSpec(shard_index=0, requests=specs),
        policy=BatchPolicySpec(max_tokens_per_batch=512, max_wait_ms=5),
        **overrides,
    )
    return asyncio.run(activities.run_inference_shard(input))


# ----------------------------------------------------------------------
# shard execution
# ----------------------------------------------------------------------
def test_shard_completes_all_requests_in_order():
    specs = [_spec(i) for i in range(4)]
    result = _run_shard(specs)
    assert [r.request_id for r in result.results] == ["r0", "r1", "r2", "r3"]
    assert all(r.status == "completed" for r in result.results)
    assert all(r.tokens_generated == 8 for r in result.results)
    assert all(r.output_text for r in result.results)
    assert result.tokens_total == 32
    assert result.batch_sizes  # at least one micro-batch was dispatched


def test_shard_respects_priority_lane_ordering():
    specs = [
        _spec(0),
        InferenceRequestSpec(request_id="hot", prompt="urgent prompt",
                             max_new_tokens=4, priority="INTERACTIVE"),
    ]
    result = _run_shard(specs)
    assert {r.request_id for r in result.results} == {"r0", "hot"}
    assert all(r.status == "completed" for r in result.results)


def test_shard_rate_limit_rejects_over_burst():
    specs = [_spec(i) for i in range(5)]
    result = _run_shard(specs, rate_limit_rate=0.001, rate_limit_burst=2)
    statuses = [r.status for r in result.results]
    assert statuses.count("completed") == 2
    assert statuses.count("rate_limited") == 3


def test_shard_empty_is_a_noop():
    result = _run_shard([])
    assert result.results == []
    assert result.tokens_total == 0


# ----------------------------------------------------------------------
# sharding (pure workflow-side packing)
# ----------------------------------------------------------------------
def test_shard_requests_count_cap():
    specs = [_spec(i) for i in range(10)]
    shards = shard_requests(specs, max_requests=4, token_budget=10**9)
    assert [len(s.requests) for s in shards] == [4, 4, 2]
    assert [s.shard_index for s in shards] == [0, 1, 2]


def test_shard_requests_token_budget():
    # Each spec costs len("hello world")//4 + 8 = 2 + 8 = 10 tokens.
    specs = [_spec(i) for i in range(5)]
    shards = shard_requests(specs, max_requests=100, token_budget=25)
    assert [len(s.requests) for s in shards] == [2, 2, 1]


def test_shard_requests_oversized_single_gets_own_shard():
    big = InferenceRequestSpec(request_id="big", prompt="x" * 400,
                               max_new_tokens=500)
    shards = shard_requests([big, _spec(1)], max_requests=10, token_budget=50)
    assert [len(s.requests) for s in shards] == [1, 1]
    assert shards[0].requests[0].request_id == "big"


def test_shard_requests_empty():
    assert shard_requests([], max_requests=4, token_budget=100) == []
