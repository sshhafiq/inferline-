"""inferline — a lightweight high-throughput LLM inference server.

The serving stack is split into small, testable pieces:

* :mod:`inferline.config` — YAML + environment configuration.
* :mod:`inferline.auth` — API-key authentication.
* :mod:`inferline.ratelimit` — per-key token-bucket rate limiting.
* :mod:`inferline.metrics` — in-process Prometheus exposition.
* :mod:`inferline.batching` — the batching policy (token budgets, priority lanes).
* :mod:`inferline.scheduler` — the continuous-batching scheduler and backends.
* :mod:`inferline.server` — the FastAPI app (OpenAI-compatible API).
"""

from inferline.batching import BatchingPolicy, Priority, QueuedRequest
from inferline.config import Settings
from inferline.metrics import MetricsRegistry
from inferline.ratelimit import RateLimiter
from inferline.scheduler import (
    Backend,
    GenerationRequest,
    MockBackend,
    Scheduler,
    TokenEvent,
    TransformersBackend,
)
from inferline.server import create_app

__all__ = [
    "Backend",
    "BatchingPolicy",
    "GenerationRequest",
    "MetricsRegistry",
    "MockBackend",
    "Priority",
    "QueuedRequest",
    "RateLimiter",
    "Scheduler",
    "Settings",
    "TokenEvent",
    "TransformersBackend",
    "create_app",
]

__version__ = "0.1.0"
