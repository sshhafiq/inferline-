"""Temporal integration: batch inference as a durable workflow.

:mod:`inferline.scheduler` serves requests inside one process. This
subpackage lifts whole *batch jobs* — hundreds of prompts with a shared
policy and tenant — onto Temporal:

- :class:`~inferline.temporal.workflows.BatchInferenceWorkflow` shards
  the job, fans shards out as parallel activities in waves, honours a
  cancellation signal between waves, and aggregates results + metrics.
- :class:`~inferline.temporal.activities.InferenceActivities` runs one
  shard through a real :class:`~inferline.scheduler.Scheduler` (with the
  project's :class:`BatchingPolicy`, :class:`RateLimiter`, and
  :class:`MetricsRegistry`), heartbeating while it works.
- :mod:`inferline.temporal.worker` — a runnable worker + demo client.

Requires the ``temporal`` extra: ``pip install inferline[temporal]``.
"""

from inferline.temporal.models import (
    BatchInferenceInput,
    BatchInferenceResult,
    BatchPolicySpec,
    BatchProgress,
    InferenceRequestSpec,
    RequestResult,
    ShardResult,
)

__all__ = [
    "BatchInferenceInput",
    "BatchInferenceResult",
    "BatchPolicySpec",
    "BatchProgress",
    "InferenceRequestSpec",
    "RequestResult",
    "ShardResult",
]
