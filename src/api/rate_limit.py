"""Rate limiting middleware."""
from __future__ import annotations

import asyncio
import time
from typing import Dict

import redis.asyncio as redis
from fastapi import Depends, HTTPException, status

from src.config import AppConfig, load_config
from .auth import verify_api_key


class RateLimiter:
    """Simple token bucket rate limiter supporting Redis or in-memory storage."""

    def __init__(self, config: AppConfig):
        self.limit = config.security.rate_limit_per_minute
        self.burst = config.security.rate_limit_burst
        self.redis_url = config.security.redis_url
        self.window = 60
        self._memory_buckets: Dict[str, tuple[float, float]] = {}
        self._redis: redis.Redis | None = None

    async def _get_redis(self) -> redis.Redis | None:
        if not self.redis_url:
            return None
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url)
        return self._redis

    async def allow(self, key: str) -> bool:
        client = await self._get_redis()
        if client:
            return await self._allow_redis(client, key)
        return self._allow_memory(key)

    async def _allow_redis(self, client: redis.Redis, key: str) -> bool:
        script = """
        local tokens_key = KEYS[1]
        local timestamp_key = KEYS[2]
        local rate = tonumber(ARGV[1])
        local burst = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local window = tonumber(ARGV[4])

        local tokens = tonumber(redis.call('get', tokens_key) or burst)
        local timestamp = tonumber(redis.call('get', timestamp_key) or now)
        local delta = math.max(0, now - timestamp)
        tokens = math.min(burst, tokens + delta * rate / window)
        if tokens < 1 then
            redis.call('set', tokens_key, tokens)
            redis.call('set', timestamp_key, now)
            return 0
        else
            tokens = tokens - 1
            redis.call('set', tokens_key, tokens)
            redis.call('set', timestamp_key, now)
            redis.call('expire', tokens_key, window)
            redis.call('expire', timestamp_key, window)
            return 1
        end
        """
        now = time.time()
        result = await client.eval(
            script,
            keys=[f"ratelimit:{key}:tokens", f"ratelimit:{key}:ts"],
            args=[self.limit, self.burst, now, self.window],
        )
        return bool(result)

    def _allow_memory(self, key: str) -> bool:
        now = time.time()
        tokens, timestamp = self._memory_buckets.get(key, (self.burst, now))
        delta = max(0.0, now - timestamp)
        tokens = min(self.burst, tokens + delta * self.limit / self.window)
        if tokens < 1:
            self._memory_buckets[key] = (tokens, now)
            return False
        tokens -= 1
        self._memory_buckets[key] = (tokens, now)
        return True


_limiter: RateLimiter | None = None


def get_rate_limiter(config: AppConfig = Depends(load_config)) -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(config)
    return _limiter


async def enforce_rate_limit(api_key: str, limiter: RateLimiter | None = None) -> None:
    limiter = limiter or get_rate_limiter(load_config())
    allowed = await limiter.allow(api_key)
    if not allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")


async def rate_limit_dependency(
    api_key: str = Depends(verify_api_key),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    await enforce_rate_limit(api_key, limiter)
