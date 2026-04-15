"""Sliding-window rate limiter using Redis sorted sets."""

import time

from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter keyed by client IP (or API key).

    Algorithm: Redis Sorted Set — each request stores a timestamp member.
    Window cleanup removes entries older than (now - window_sec).
    """

    def __init__(self, app: object, redis: Redis, rpm: int, burst: int) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._redis = redis
        self._rpm = rpm
        self._burst = burst
        self._window_sec = 60

    async def dispatch(self, request: Request, call_next: object) -> Response:
        # Health/metrics endpoints are exempt from rate limiting.
        if request.url.path in {"/health", "/ready", "/metrics"}:
            return await call_next(request)  # type: ignore[misc]

        identifier = self._get_identifier(request)
        key = f"ratelimit:{identifier}"
        now = time.time()
        window_start = now - self._window_sec

        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, "-inf", window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, self._window_sec + 1)
        results = await pipe.execute()

        count: int = results[2]
        remaining = max(0, self._rpm - count)
        reset_at = int(now) + self._window_sec

        if count > self._rpm + self._burst:
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limit_exceeded", "retry_after": self._window_sec},
                headers={
                    "X-RateLimit-Limit": str(self._rpm),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "Retry-After": str(self._window_sec),
                },
            )

        response: Response = await call_next(request)  # type: ignore[misc]
        response.headers["X-RateLimit-Limit"] = str(self._rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)
        return response

    @staticmethod
    def _get_identifier(request: Request) -> str:
        # Prefer API key header for per-key limiting; fall back to IP.
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key[:16]}"  # truncate to avoid huge keys
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        return f"ip:{request.client.host if request.client else 'unknown'}"
