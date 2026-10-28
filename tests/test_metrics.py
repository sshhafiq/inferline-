"""Tests for the Prometheus metrics registry."""

from inferline.metrics import Counter, Gauge, Histogram, MetricsRegistry


def test_counter_rendering():
    c = Counter("my_counter", "A test counter.")
    c.inc(2, "a", "b")
    c.inc(1, "a", "b")
    out = []
    c.render(("l1", "l2"), out)
    text = "\n".join(out)
    assert "# TYPE my_counter counter" in text
    assert 'my_counter{l1="a",l2="b"} 3.0' in text


def test_gauge_up_down():
    g = Gauge("my_gauge", "A test gauge.")
    g.inc(2)
    g.dec(1)
    assert g.get() == 1.0
    g.set(5)
    assert g.get() == 5.0


def test_histogram_buckets_cumulative():
    h = Histogram("my_hist", "A test histogram.", buckets=(1.0, 2.0))
    h.observe(0.5)
    h.observe(1.5)
    h.observe(5.0)
    out = []
    h.render(out)
    text = "\n".join(out)
    assert 'my_hist_bucket{le="1.0"} 1' in text
    assert 'my_hist_bucket{le="2.0"} 2' in text
    assert 'my_hist_bucket{le="+Inf"} 3' in text
    assert "my_hist_sum 7.0" in text
    assert "my_hist_count 3" in text


def test_registry_renders_all_metrics():
    reg = MetricsRegistry()
    reg.requests_total.inc(1, "chat_completions", "accepted")
    reg.queue_depth.set(4)
    reg.batch_size.observe(3)
    text = reg.render()
    assert "inferline_requests_total" in text
    assert 'inferline_requests_total{endpoint="chat_completions",status="accepted"} 1.0' in text
    assert "inferline_queue_depth 4" in text
    assert "inferline_batch_size_count 1" in text
    assert text.endswith("\n")
