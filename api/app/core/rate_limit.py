from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request, status


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def enforce(self, *, key: str, limit: int, window_seconds: int) -> None:
        now = monotonic()
        cutoff = now - window_seconds
        with self._lock:
            bucket = self._events[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests")
            bucket.append(now)


rate_limiter = SlidingWindowRateLimiter()


def enforce_rate_limit(request: Request, *, scope: str, limit: int, window_seconds: int) -> None:
    client_host = request.client.host if request.client else "unknown"
    rate_limiter.enforce(key=f"{scope}:{client_host}", limit=limit, window_seconds=window_seconds)
