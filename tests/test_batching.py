"""Tests for the batching policy."""

import pytest

from inferline.batching import BatchingPolicy, Priority, QueuedRequest


def make_request(rid, prompt_tokens=10, max_new_tokens=10, priority=Priority.BATCH):
    return QueuedRequest(
        request_id=rid,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        priority=priority,
    )


def test_empty_queue_gives_empty_batch():
    assert BatchingPolicy().form_batch([]) == []


def test_batch_respects_token_budget():
    policy = BatchingPolicy(max_tokens_per_batch=100)
    queue = [make_request(f"r{i}", prompt_tokens=30, max_new_tokens=20) for i in range(5)]
    batch = policy.form_batch(queue)
    # each request costs 50 tokens; budget 100 -> 2 requests
    assert [r.request_id for r in batch] == ["r0", "r1"]


def test_priority_lane_goes_first():
    policy = BatchingPolicy(max_tokens_per_batch=60)
    queue = [
        make_request("batch-1", prompt_tokens=25, max_new_tokens=25),
        make_request("interactive-1", prompt_tokens=25, max_new_tokens=25,
                     priority=Priority.INTERACTIVE),
    ]
    batch = policy.form_batch(queue)
    # budget 60 fits only one 50-token request; interactive wins
    assert [r.request_id for r in batch] == ["interactive-1"]


def test_fifo_within_lane():
    policy = BatchingPolicy(max_tokens_per_batch=1000)
    queue = [make_request(f"r{i}") for i in range(3)]
    # reverse arrival order artificially
    queue[0].arrival_time = 3.0
    queue[1].arrival_time = 1.0
    queue[2].arrival_time = 2.0
    batch = policy.form_batch(queue)
    assert [r.request_id for r in batch] == ["r1", "r2", "r0"]


def test_oversized_request_admitted_alone_no_starvation():
    policy = BatchingPolicy(max_tokens_per_batch=100)
    queue = [make_request("huge", prompt_tokens=500, max_new_tokens=500)]
    batch = policy.form_batch(queue)
    assert [r.request_id for r in batch] == ["huge"]


def test_does_not_mutate_queue():
    policy = BatchingPolicy(max_tokens_per_batch=50)
    queue = [make_request("r0", prompt_tokens=25, max_new_tokens=25)]
    policy.form_batch(queue)
    assert len(queue) == 1


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        BatchingPolicy(max_tokens_per_batch=0)
    with pytest.raises(ValueError):
        BatchingPolicy(max_wait_ms=-1)
