"""Sliding-window in-memory rate-limiting middleware."""

import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter: max `calls` requests per `period` seconds
    per client IP address.

    Args:
        calls  : maximum allowed requests in the window (default 60)
        period : window length in seconds (default 60)
    """

    def __init__(self, app, calls: int = 60, period: int = 60) -> None:
        super().__init__(app)
        self.calls = calls
        self.period = period
        self._store: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "0.0.0.0"
        now = time.time()
        window_start = now - self.period

        # Drop timestamps that have fallen outside the window
        self._store[client_ip] = [t for t in self._store[client_ip] if t > window_start]

        if len(self._store[client_ip]) >= self.calls:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please retry later."},
                headers={"Retry-After": str(self.period)},
            )

        self._store[client_ip].append(now)
        return await call_next(request)
