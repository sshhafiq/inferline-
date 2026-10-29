"""Tests for the token-bucket rate limiter."""

import time

from inferline.ratelimit import RateLimiter, TokenBucket


def test_bucket_allows_burst_then_denies():
    bucket = TokenBucket(rate=10.0, capacity=3)
    assert bucket.allow()
    assert bucket.allow()
    assert bucket.allow()
    assert not bucket.allow()


def test_bucket_refills_over_time():
    bucket = TokenBucket(rate=100.0, capacity=1)
    assert bucket.allow()
    assert not bucket.allow()
    time.sleep(0.02)  # 20ms * 100/s = 2 tokens, capped at 1
    assert bucket.allow()


def test_bucket_never_exceeds_capacity():
    bucket = TokenBucket(rate=1000.0, capacity=2)
    time.sleep(0.05)
    assert bucket.tokens == 2.0


def test_limiter_is_per_key():
    limiter = RateLimiter(rate=1.0, burst=1)
    assert limiter.allow("alice")
    assert not limiter.allow("alice")
    assert limiter.allow("bob")  # separate bucket


def test_limiter_disabled_allows_everything():
    limiter = RateLimiter(rate=1.0, burst=1, enabled=False)
    for _ in range(10):
        assert limiter.allow("alice")


def test_bucket_multi_token_cost():
    bucket = TokenBucket(rate=1.0, capacity=3)
    assert bucket.allow(cost=2)
    assert not bucket.allow(cost=2)
    assert bucket.allow(cost=1)
    assert not bucket.allow()


def test_invalid_parameters():
    import pytest

    with pytest.raises(ValueError):
        TokenBucket(rate=0, capacity=1)
    with pytest.raises(ValueError):
        TokenBucket(rate=1, capacity=0)
