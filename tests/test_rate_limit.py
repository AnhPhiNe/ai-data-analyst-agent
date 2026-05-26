from backend.core.rate_limit import InMemoryRateLimiter


def test_in_memory_rate_limiter_blocks_after_limit() -> None:
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)

    assert limiter.allow("client:/chat/query") is True
    assert limiter.allow("client:/chat/query") is True
    assert limiter.allow("client:/chat/query") is False


def test_in_memory_rate_limiter_reset() -> None:
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)

    assert limiter.allow("client:/datasets/upload") is True
    assert limiter.allow("client:/datasets/upload") is False
    limiter.reset()
    assert limiter.allow("client:/datasets/upload") is True
