"""Serializable data model for the batch-inference workflow.

Plain dataclasses with JSON-friendly types only, so Temporal's default
data converter can carry them. ``priority`` is the *name* of an
:class:`inferline.batching.Priority` member (``"BATCH"`` /
``"INTERACTIVE"``); activities resolve it to the enum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceRequestSpec:
    """One generation request in a batch job."""

    request_id: str
    prompt: str
    max_new_tokens: int = 64
    priority: str = "BATCH"

    def estimated_token_cost(self) -> int:
        """Worst-case tokens, mirroring ``QueuedRequest.token_cost``.

        Uses the same ~4 chars/token heuristic as
        :func:`inferline.scheduler.estimate_tokens`. Pure arithmetic, so
        the workflow can shard on it deterministically.
        """
        return max(1, len(self.prompt) // 4) + self.max_new_tokens


@dataclass
class BatchPolicySpec:
    """Serializable mirror of :class:`inferline.batching.BatchingPolicy`."""

    max_tokens_per_batch: int = 2048
    max_wait_ms: int = 20
    max_batch_requests: int = 64


@dataclass
class BatchInferenceInput:
    """Input to :class:`BatchInferenceWorkflow`."""

    job_id: str
    requests: list[InferenceRequestSpec]
    policy: BatchPolicySpec = field(default_factory=BatchPolicySpec)
    #: Sharding: a shard closes when either limit is hit.
    shard_max_requests: int = 8
    shard_token_budget: int = 16384
    #: How many shard activities run concurrently (one wave).
    shard_parallelism: int = 4
    #: Tenant key for the per-shard RateLimiter.
    tenant: str = "default"
    rate_limit_rate: float = 1000.0
    rate_limit_burst: int = 10000
    #: MockBackend knob, forwarded to shard activities (ms per token).
    backend_latency_per_token_ms: float = 0.0


@dataclass
class ShardSpec:
    shard_index: int
    requests: list[InferenceRequestSpec] = field(default_factory=list)


@dataclass
class RunShardInput:
    job_id: str
    shard: ShardSpec
    policy: BatchPolicySpec = field(default_factory=BatchPolicySpec)
    tenant: str = "default"
    rate_limit_rate: float = 1000.0
    rate_limit_burst: int = 10000
    backend_latency_per_token_ms: float = 0.0


@dataclass
class RequestResult:
    request_id: str
    status: str  # completed | rate_limited | failed | cancelled
    output_text: str = ""
    tokens_generated: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None


@dataclass
class ShardResult:
    shard_index: int
    results: list[RequestResult] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    tokens_total: int = 0
    wall_time_s: float = 0.0


@dataclass
class BatchProgress:
    """Live progress, returned by the workflow's ``progress`` query."""

    job_id: str
    total_requests: int = 0
    completed_requests: int = 0
    shards_total: int = 0
    shards_completed: int = 0
    cancelled: bool = False


@dataclass
class BatchInferenceResult:
    job_id: str
    status: str  # completed | cancelled
    results: list[RequestResult] = field(default_factory=list)
    requests_completed: int = 0
    requests_rate_limited: int = 0
    requests_failed: int = 0
    requests_cancelled: int = 0
    tokens_total: int = 0
    mean_latency_s: float = 0.0
    batches_dispatched: int = 0
