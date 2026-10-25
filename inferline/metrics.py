"""In-process metrics with Prometheus text exposition.

This module deliberately avoids a hard dependency on ``prometheus_client``:
the registry below implements just enough of the Prometheus exposition format
(counters, gauges, histograms) for the ``/metrics`` endpoint and for tests.
"""

from __future__ import annotations

import threading
from typing import Iterable


class Counter:
    """Monotonically increasing counter, optionally labelled."""

    def __init__(self, name: str, documentation: str) -> None:
        self.name = name
        self.documentation = documentation
        self._values: dict[tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, *labels: str) -> None:
        with self._lock:
            self._values[labels] = self._values.get(labels, 0.0) + amount

    def get(self, *labels: str) -> float:
        with self._lock:
            return self._values.get(labels, 0.0)

    def render(
        self, label_names: Iterable[str], out: list[str]
    ) -> None:
        out.append(f"# HELP {self.name} {self.documentation}")
        out.append(f"# TYPE {self.name} counter")
        with self._lock:
            items = list(self._values.items())
        for labels, value in sorted(items):
            out.append(f"{self.name}{_format_labels(label_names, labels)} {value}")


class Gauge:
    """A value that can go up and down."""

    def __init__(self, name: str, documentation: str) -> None:
        self.name = name
        self.documentation = documentation
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    def get(self) -> float:
        with self._lock:
            return self._value

    def render(self, out: list[str]) -> None:
        out.append(f"# HELP {self.name} {self.documentation}")
        out.append(f"# TYPE {self.name} gauge")
        out.append(f"{self.name} {self.get()}")


class Histogram:
    """Cumulative histogram with configurable buckets."""

    def __init__(
        self, name: str, documentation: str, buckets: Iterable[float]
    ) -> None:
        self.name = name
        self.documentation = documentation
        self.buckets = sorted(buckets)
        self._counts = [0] * (len(self.buckets) + 1)  # last slot is +Inf
        self._sum = 0.0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    self._counts[i] += 1
                    break
            self._counts[-1] += 1  # the +Inf bucket always counts
            self._sum += value

    def render(self, out: list[str]) -> None:
        out.append(f"# HELP {self.name} {self.documentation}")
        out.append(f"# TYPE {self.name} histogram")
        with self._lock:
            counts = list(self._counts)
            total = self._sum
        cumulative = 0
        for bound, count in zip(self.buckets, counts[:-1]):
            cumulative += count
            out.append(f'{self.name}_bucket{{le="{bound}"}} {cumulative}')
        out.append(f'{self.name}_bucket{{le="+Inf"}} {counts[-1]}')
        out.append(f"{self.name}_sum {total}")
        out.append(f"{self.name}_count {counts[-1]}")


def _format_labels(names: Iterable[str], values: tuple[str, ...]) -> str:
    names = list(names)
    if not names:
        return ""
    pairs = ",".join(f'{n}="{v}"' for n, v in zip(names, values))
    return "{" + pairs + "}"


class MetricsRegistry:
    """The set of metrics inferline tracks, exposed at ``/metrics``."""

    def __init__(self) -> None:
        self.requests_total = Counter(
            "inferline_requests_total", "Total generation requests."
        )
        self.request_labels = ("endpoint", "status")
        self.tokens_total = Counter(
            "inferline_tokens_total", "Total generated tokens."
        )
        self.queue_depth = Gauge(
            "inferline_queue_depth", "Requests currently waiting or in flight."
        )
        self.batch_size = Histogram(
            "inferline_batch_size",
            "Number of requests per dispatched micro-batch.",
            buckets=(1, 2, 4, 8, 16, 32, 64),
        )
        self.ttft_seconds = Histogram(
            "inferline_ttft_seconds",
            "Time to first token per request.",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        )
        self.request_latency_seconds = Histogram(
            "inferline_request_latency_seconds",
            "End-to-end request latency.",
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )
        self.tokens_per_second = Gauge(
            "inferline_tokens_per_second",
            "Generation throughput of the most recent batch.",
        )

    def render(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        out: list[str] = []
        self.requests_total.render(self.request_labels, out)
        self.tokens_total.render((), out)
        self.queue_depth.render(out)
        self.batch_size.render(out)
        self.ttft_seconds.render(out)
        self.request_latency_seconds.render(out)
        self.tokens_per_second.render(out)
        return "\n".join(out) + "\n"
