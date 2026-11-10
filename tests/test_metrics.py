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


def test_histogram_count_and_mean():
    h = Histogram("my_hist2", "Another test histogram.", buckets=(1.0, 2.0))
    assert h.count() == 0
    assert h.mean() == 0.0
    h.observe(0.5)
    h.observe(1.5)
    h.observe(5.0)
    assert h.count() == 3
    assert h.mean() == 7.0 / 3


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


def test_counter_total_sums_label_sets():
    c = Counter("total_counter", "A test counter.")
    assert c.total() == 0.0
    c.inc(2, "a")
    c.inc(3, "b")
    c.inc(1, "a")
    assert c.total() == 6.0
    c.reset("b")
    assert c.total() == 3.0


def test_counter_reset_drops_one_label_set():
    c = Counter("reset_counter", "A test counter.")
    c.inc(3, "a")
    c.inc(1, "b")
    c.reset("a")
    assert c.get("a") == 0.0
    assert c.get("b") == 1.0
    c.reset("missing")
    assert c.get("b") == 1.0


def test_counter_label_set_count():
    c = Counter("label_counter", "A test counter.")
    assert c.label_set_count() == 0
    c.inc(2, "a")
    c.inc(3, "b")
    c.inc(1, "a")
    assert c.label_set_count() == 2
    c.reset("b")
    assert c.label_set_count() == 1


def test_histogram_sum_tracks_observed_values():
    h = Histogram("sum_hist", "Histogram for sum checks.", buckets=(1.0, 2.0))
    assert h.sum() == 0.0
    h.observe(0.5)
    h.observe(1.5)
    h.observe(5.0)
    assert h.sum() == 7.0
    assert h.sum() == h.mean() * h.count()


def test_gauge_reset_returns_to_zero():
    g = Gauge("reset_gauge", "A test gauge.")
    assert g.get() == 0.0
    g.set(7)
    g.inc(3)
    assert g.get() == 10.0
    g.reset()
    assert g.get() == 0.0
    g.dec(1)
    assert g.get() == -1.0
    g.reset()
    assert g.get() == 0.0


def test_histogram_reset_clears_counts_and_sum():
    h = Histogram("reset_hist", "Histogram for reset checks.", buckets=(1.0, 2.0))
    h.observe(0.5)
    h.observe(5.0)
    assert h.count() == 2
    h.reset()
    assert h.count() == 0
    assert h.sum() == 0.0
    assert h.mean() == 0.0
    h.observe(1.5)
    assert h.count() == 1
    assert h.sum() == 1.5


def test_counter_max_value_picks_largest_label_set():
    c = Counter("max_counter", "A test counter.")
    assert c.max_value() == 0.0
    c.inc(2, "a")
    c.inc(5, "b")
    c.inc(1, "a")
    assert c.max_value() == 5.0
    c.reset("b")
    assert c.max_value() == 3.0
    c.reset("a")
    assert c.max_value() == 0.0


def test_counter_min_value_picks_smallest_label_set():
    c = Counter("min_counter", "A test counter.")
    assert c.min_value() == 0.0
    c.inc(2, "a")
    c.inc(5, "b")
    c.inc(1, "a")
    assert c.min_value() == 3.0
    c.reset("a")
    assert c.min_value() == 5.0
    c.reset("b")
    assert c.min_value() == 0.0


def test_counter_mean_value_averages_label_sets():
    c = Counter("mean_counter", "A test counter.")
    assert c.mean_value() == 0.0
    c.inc(2, "a")
    c.inc(6, "b")
    assert c.mean_value() == 4.0
    c.inc(2, "a")
    assert c.mean_value() == 5.0
    c.reset("b")
    assert c.mean_value() == 4.0
    c.reset("a")
    assert c.mean_value() == 0.0


def test_histogram_bucket_counts_track_each_bucket():
    h = Histogram("bucket_hist", "Histogram for bucket checks.", buckets=(1.0, 2.0))
    assert h.bucket_counts() == [0, 0, 0]
    h.observe(0.5)
    h.observe(1.5)
    h.observe(5.0)
    # Each observation lands in the first bucket that covers it; +Inf counts all.
    assert h.bucket_counts() == [1, 1, 3]
    # The returned list is a copy, so mutating it cannot corrupt the histogram.
    counts = h.bucket_counts()
    counts[0] = 99
    assert h.bucket_counts() == [1, 1, 3]


def test_histogram_overflow_count_counts_values_above_top_bucket():
    h = Histogram("overflow_hist", "Histogram for overflow checks.", buckets=(1.0, 2.0))
    assert h.overflow_count() == 0
    h.observe(0.5)
    h.observe(1.5)
    assert h.overflow_count() == 0
    h.observe(5.0)
    h.observe(9.0)
    assert h.overflow_count() == 2
    h.reset()
    assert h.overflow_count() == 0


def test_histogram_lowest_bucket_count_tracks_fast_observations():
    h = Histogram("lowest_hist", "Histogram for lowest bucket checks.", buckets=(1.0, 2.0))
    assert h.lowest_bucket_count() == 0
    h.observe(0.5)
    h.observe(1.5)
    h.observe(0.25)
    assert h.lowest_bucket_count() == 2
    h.observe(5.0)
    assert h.lowest_bucket_count() == 2
    h.reset()
    assert h.lowest_bucket_count() == 0


def test_histogram_highest_finite_bucket_count_tracks_slow_observations():
    h = Histogram("highest_hist", "Histogram for highest bucket checks.", buckets=(1.0, 2.0))
    assert h.highest_finite_bucket_count() == 0
    h.observe(0.5)
    h.observe(1.5)
    assert h.highest_finite_bucket_count() == 1
    h.observe(5.0)
    assert h.highest_finite_bucket_count() == 1
    h.reset()
    assert h.highest_finite_bucket_count() == 0


def test_histogram_empty_bucket_count_tracks_unfilled_buckets():
    h = Histogram("empty_hist", "Histogram for empty bucket checks.", buckets=(1.0, 2.0, 4.0))
    assert h.empty_bucket_count() == 3
    h.observe(0.5)
    assert h.empty_bucket_count() == 2
    h.observe(5.0)
    # Overflow lands in no finite bucket, so the count is unchanged.
    assert h.empty_bucket_count() == 2
    h.observe(1.5)
    h.observe(3.0)
    assert h.empty_bucket_count() == 0
    h.reset()
    assert h.empty_bucket_count() == 3
