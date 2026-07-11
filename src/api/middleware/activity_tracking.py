"""Per-request user activity tracking (admin feature, spec 2026-07-11).

ActivityBuffer is a BOUNDED in-memory aggregator: (user, endpoint_group,
method, minute) -> count. Bounded because this box has OOM history — past the
cap, NEW buckets are dropped (counted in .dropped); existing buckets keep
incrementing. Flush drains to the record_user_activity RPC (additive upsert),
fired as a background task from the middleware. Everything here is fail-open:
tracking must never block or break a request.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class ActivityBuffer:
    """Bounded aggregation of per-request activity into per-minute buckets."""

    def __init__(
        self,
        max_buckets: int = 2048,
        flush_interval_s: float = 30.0,
        flush_threshold: int = 500,
    ):
        self.max_buckets = max_buckets
        self.flush_interval_s = flush_interval_s
        self.flush_threshold = flush_threshold
        self.dropped = 0
        self._buckets: Dict[Tuple[str, Optional[str], str, str, str], int] = {}
        self._last_flush = time.monotonic()

    def record(
        self,
        user_id: str,
        user_email: Optional[str],
        endpoint_group: str,
        http_method: str,
        bucket_minute_iso: str,
    ) -> bool:
        """Record one request. Returns True when the caller should flush."""
        key = (user_id, user_email, endpoint_group, http_method, bucket_minute_iso)
        if key not in self._buckets and len(self._buckets) >= self.max_buckets:
            self.dropped += 1
        else:
            self._buckets[key] = self._buckets.get(key, 0) + 1
        return self.should_flush()

    def should_flush(self) -> bool:
        if not self._buckets:
            return False
        if len(self._buckets) >= self.flush_threshold:
            return True
        return (time.monotonic() - self._last_flush) >= self.flush_interval_s

    def drain(self) -> List[Dict[str, Any]]:
        """Return accumulated rows (RPC payload shape) and reset the buffer."""
        rows = [
            {
                "user_id": k[0],
                "user_email": k[1],
                "endpoint_group": k[2],
                "http_method": k[3],
                "bucket_minute": k[4],
                "request_count": count,
            }
            for k, count in self._buckets.items()
        ]
        self._buckets = {}
        self._last_flush = time.monotonic()
        return rows


async def flush_rows(rows: List[Dict[str, Any]]) -> None:
    """Persist drained rows via the additive-upsert RPC. Fail-open."""
    if not rows:
        return
    try:
        from src.api.dependencies.supabase_client import get_supabase

        client = get_supabase()
        if client is None:
            return
        await asyncio.to_thread(
            lambda: client.rpc("record_user_activity", {"p_rows": rows}).execute()
        )
    except Exception:
        logger.warning(
            "activity flush failed (fail-open, %d rows lost)", len(rows), exc_info=True
        )


# Strong references to in-flight flush tasks: asyncio only weakly references
# tasks, so an unreferenced fire-and-forget task can be garbage-collected
# before it runs — silently dropping rows (fail-open would hide it).
_INFLIGHT_FLUSHES: set = set()


def schedule_flush(rows: List[Dict[str, Any]]) -> "asyncio.Task[None]":
    task = asyncio.get_running_loop().create_task(flush_rows(rows))
    _INFLIGHT_FLUSHES.add(task)
    task.add_done_callback(_INFLIGHT_FLUSHES.discard)
    return task


def _endpoint_group(path: str) -> Optional[str]:
    """'/api/causal/estimate' -> 'causal'. Bounded cardinality by design."""
    parts = path.split("/")
    if len(parts) >= 3 and parts[1] == "api" and parts[2]:
        return parts[2]
    return None


class ActivityTrackingMiddleware(BaseHTTPMiddleware):
    """Records (user, endpoint_group, minute) for authenticated /api requests.

    Must be INNER to JWTAuthMiddleware (added BEFORE it in main.py — Starlette
    add_middleware prepends, so earlier-added = inner = sees request.state.user).
    Fail-open everywhere; flushes fire-and-forget so requests never wait on DB.
    """

    def __init__(self, app, buffer: Optional[ActivityBuffer] = None):
        super().__init__(app)
        self.buffer = buffer or ActivityBuffer()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        try:
            user = getattr(request.state, "user", None)
            group = _endpoint_group(request.url.path)
            if user and user.get("id") and group:
                try:
                    uuid.UUID(str(user["id"]))  # skip TESTING_MODE's non-UUID user
                except ValueError:
                    return response
                minute = (
                    datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
                )
                if self.buffer.record(
                    str(user["id"]), user.get("email"), group, request.method, minute
                ):
                    rows = self.buffer.drain()
                    schedule_flush(rows)
        except Exception:
            logger.warning("activity tracking failed (fail-open)", exc_info=True)
        return response
