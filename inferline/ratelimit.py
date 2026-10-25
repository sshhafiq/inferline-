"""Per-API-key token-bucket rate limiting.

Each key gets its own bucket that refills at ``rate`` tokens per second up to
``capacity`` (the burst size). A request costs one token. The implementation
is thread-safe so it can be shared across worker threads.
"""

from __future__ import annotations

import threading
import time


class TokenBucket:
    """A single token bucket with monotonic-clock refills."""

    def __init__(self, rate: float, capacity: int) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def allow(self, cost: int = 1) -> bool:
        """Consume ``cost`` tokens if available; return whether it succeeded."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._updated
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._updated = now
            if self._tokens >= cost:
                self._tokens -= cost
                return True
            return False

    @property
    def tokens(self) -> float:
        """Current token count (refilled as of now)."""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self.capacity, self._tokens + (now - self._updated) * self.rate
            )
            self._updated = now
            return self._tokens


class RateLimiter:
    """Manages one :class:`TokenBucket` per key, created lazily."""

    def __init__(self, rate: float, burst: int, enabled: bool = True) -> None:
        self.rate = rate
        self.burst = burst
        self.enabled = enabled
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Return True if ``key`` may proceed right now."""
        if not self.enabled:
            return True
        bucket = self._bucket_for(key)
        return bucket.allow()

    def _bucket_for(self, key: str) -> TokenBucket:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(rate=self.rate, capacity=self.burst)
                self._buckets[key] = bucket
            return bucket
