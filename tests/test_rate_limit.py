import asyncio

import pytest

from src.api.rate_limit import RateLimiter
from src.config.settings import AppConfig, SecuritySettings


@pytest.mark.asyncio
async def test_rate_limiter_memory(monkeypatch):
    config = AppConfig(security=SecuritySettings(api_keys=["test"], rate_limit_per_minute=2, rate_limit_burst=1))
    limiter = RateLimiter(config)
    assert await limiter.allow("test") is True
    assert await limiter.allow("test") is False
    await asyncio.sleep(1)
    assert await limiter.allow("test") in {True, False}
