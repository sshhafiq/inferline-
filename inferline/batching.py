"""Batching policy for the continuous-batching scheduler.

A micro-batch is formed greedily under a token budget: requests are ordered
by priority lane (interactive traffic first) then FIFO, and admitted while
``prompt_tokens + max_new_tokens`` fits the remaining budget. This mirrors how
production inference servers (vLLM, TensorRT-LLM) pack requests — the budget
bounds worst-case KV-cache usage per forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from time import monotonic


class Priority(IntEnum):
    """Scheduling lanes. Higher value = scheduled sooner."""

    BATCH = 0  #: throughput-oriented background traffic
    INTERACTIVE = 1  #: latency-sensitive traffic (e.g. streaming chat)


@dataclass
class QueuedRequest:
    """A request waiting in the scheduler queue."""

    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    priority: Priority = Priority.BATCH
    arrival_time: float = field(default_factory=monotonic)

    @property
    def token_cost(self) -> int:
        """Worst-case tokens this request can occupy in a batch."""
        return self.prompt_tokens + self.max_new_tokens


@dataclass
class BatchingPolicy:
    """Knobs controlling micro-batch formation."""

    max_tokens_per_batch: int = 2048
    max_wait_ms: int = 20
    max_batch_requests: int = 64

    def __post_init__(self) -> None:
        if self.max_tokens_per_batch <= 0:
            raise ValueError("max_tokens_per_batch must be positive")
        if self.max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        if self.max_batch_requests <= 0:
            raise ValueError("max_batch_requests must be positive")

    def order(self, queue: list[QueuedRequest]) -> list[QueuedRequest]:
        """Order waiting requests: priority lane first, then FIFO."""
        return sorted(queue, key=lambda r: (-r.priority.value, r.arrival_time))

    def form_batch(self, queue: list[QueuedRequest]) -> list[QueuedRequest]:
        """Greedily pack the highest-priority requests into one micro-batch.

        The queue itself is not mutated; the caller removes the returned
        requests. Guarantees a non-empty batch whenever the queue is
        non-empty — an oversized head-of-line request is admitted alone so it
        can never starve.
        """
        if not queue:
            return []
        batch: list[QueuedRequest] = []
        budget = self.max_tokens_per_batch
        for request in self.order(queue):
            if len(batch) >= self.max_batch_requests:
                break
            if request.token_cost <= budget:
                batch.append(request)
                budget -= request.token_cost
            elif not batch:
                # Oversized head-of-line: admit alone to avoid starvation.
                batch.append(request)
                break
        return batch
