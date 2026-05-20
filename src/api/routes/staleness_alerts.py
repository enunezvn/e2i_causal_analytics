"""Server-Sent Events bridge from ``e2i:alerts`` Redis pub/sub to
CopilotKit dashboards (issue #390).

Endpoint
--------
``GET /api/alerts/stream?brand=<brand>``  AUTH — single-brand
                                                subscription. Returns
                                                ``text/event-stream``
                                                with one SSE ``event:
                                                alert`` per matching
                                                alert publish on
                                                :data:`ALERTS_CHANNEL`.

Wiring
------
The sentinel-action handlers in :mod:`src.tasks.sentinel_actions`
publish JSON-serialized alert payloads to the Redis pub/sub channel
``e2i:alerts`` (constant
:data:`src.tasks.sentinel_actions.ALERTS_CHANNEL`). This module owns
the SUBSCRIBER side: a per-connection async task that opens a Redis
pubsub on that channel, filters by the connecting client's requested
brand, and pushes matching events into a bounded per-connection queue
that the SSE response drains.

Design choices (issue #390, locked by the parent agent's brief)
---------------------------------------------------------------
* **SSE, not WebSocket.** One-way broadcast shape, simpler client
  semantics, matches the issue body recommendation. Implemented via
  ``sse_starlette.sse.EventSourceResponse`` (already a transitive dep
  via langgraph/copilotkit).
* **Single-brand subscription only.** ``?brand=<name>`` is required.
  Comma-separated multi-brand is OUT OF SCOPE for V1; file a follow-up.
* **No replay-on-connect.** Live-only. Subscribers connecting AFTER an
  alert was published do NOT receive it (pub/sub semantics, not
  Streams). If a client drops mid-stream and reconnects, the gap is
  permanent for this connection.
* **Backpressure: drop oldest.** The per-connection queue caps at
  :data:`MAX_QUEUE_DEPTH` (100). Once full, the oldest event is evicted
  on each new arrival, and a WARNING is logged. The connection MUST
  remain open — disconnecting a slow client would defeat the alerting
  contract.
* **Auth: ``Depends(require_auth)``** mirroring the executive_insights
  route's pattern. Any authenticated role is sufficient; brand-level
  authorization is enforced at signal layer not at the SSE bridge.

Lifecycle
---------
Each HTTP connection spawns:

1. A single Redis client (``redis.asyncio.from_url`` via the canonical
   :func:`src.memory.services.factories.get_redis_client` factory).
2. A pubsub subscription on :data:`ALERTS_CHANNEL`.
3. A background subscriber task that pumps incoming pubsub messages
   through the brand filter into the per-connection queue.
4. A foreground SSE generator that drains the queue.

When the client disconnects (the SSE generator is closed), the bridge:

* Calls ``cancel()`` on the subscriber task.
* Awaits it.
* Calls ``pubsub.aclose()`` to release the Redis subscription.

The Redis client itself is the shared cached singleton from the
factory — we do NOT close it here (other subscribers/publishers may be
using it). This mirrors the publisher side at
:func:`src.tasks.sentinel_actions.publish_alert`.

Out of scope (filed if needed)
------------------------------
* Comma-separated multi-brand subscription
* Acknowledgment-based delivery
* Historical alert replay UI
* Frontend wiring (separate task)
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Any, AsyncIterator, Awaitable, Callable, Deque, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies.auth import require_auth
from src.memory.services.factories import get_redis_client
from src.mlops.lifecycle_monitoring import record_alert_latency_cluster
from src.tasks.sentinel_actions import ALERTS_CHANNEL

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["Alerts"])


# Backpressure: max events buffered per connection before drop-oldest
# kicks in. Issue #390 spec is "drop oldest at depth >100, log a
# warning, do not disconnect".
MAX_QUEUE_DEPTH: int = 100


# Type alias for the redis-client factory the bridge uses. The default
# `get_redis_client` is sync (returns a cached client); the bridge
# wraps it in an async factory so unit tests can substitute an async
# stand-in. The protocol is intentionally minimal: just `.pubsub()`.
RedisFactory = Callable[[], Awaitable[Any]]


async def _default_redis_factory() -> Any:
    """Async wrapper around the sync :func:`get_redis_client` cache.

    The factory module returns a cached client on call — wrapping it
    in an async fn keeps the bridge's call-site await-able and lets
    tests inject a fully-async fake.
    """
    return get_redis_client()


class AlertBridge:
    """Per-connection bridge: Redis pub/sub → bounded queue → SSE.

    Instances are NOT shared across connections; each HTTP GET creates
    its own bridge so the queue depth, brand filter, and subscriber
    task are scoped to that client.

    The bridge exposes a single public method :meth:`stream` returning
    an async iterator of SSE-shaped dicts that ``EventSourceResponse``
    consumes.
    """

    def __init__(
        self,
        brand: str,
        *,
        redis_factory: Optional[RedisFactory] = None,
    ) -> None:
        if not brand:
            raise ValueError("brand is required for AlertBridge")
        self.brand = brand
        self._redis_factory: RedisFactory = redis_factory or _default_redis_factory
        # Bounded queue with explicit drop-oldest semantics. We use a
        # deque (not asyncio.Queue with maxsize) because asyncio.Queue's
        # default behavior on put-when-full is to BLOCK, not drop. We
        # want drop-oldest, which requires manual eviction.
        self._queue: Deque[Dict[str, Any]] = deque()
        # Event consumers wait on this to be set when a new item is
        # enqueued; the consumer clears it when the queue drains.
        self._item_ready: asyncio.Event = asyncio.Event()
        self._subscriber_task: Optional[asyncio.Task[None]] = None
        self._pubsub: Optional[Any] = None
        self._closed: bool = False
        # Test/observability hook: counts events that were dropped due
        # to backpressure. Lifetime = single connection.
        self._dropped_for_backpressure: int = 0

    async def _subscribe_and_pump(self) -> None:
        """Open the pubsub subscription and pump matching alerts into
        the per-connection queue. Runs as the background task.

        Liveness contract (codex iter-1 H1)
        -----------------------------------
        ON EVERY EXIT — success, exception, OR cancellation — the
        ``finally`` clause sets :attr:`self._item_ready` so the SSE
        generator's blocking ``await self._item_ready.wait()`` returns
        immediately and the generator can re-check
        ``self._subscriber_task.done()`` and exit. Without this
        guarantee, a factory-raise / subscribe-raise / listen-loop-
        raise that happened AFTER the generator entered the wait
        would leave the connection parked forever (until the client
        eventually times out).

        Exception propagation (codex iter-1 H1, iter-2 doc trim)
        --------------------------------------------------------
        Unexpected exceptions from the factory, ``pubsub.subscribe()``,
        or the ``listen()`` loop are LOGGED here, then RE-RAISED so the
        exception surfaces on ``task.exception()``.
        ``asyncio.CancelledError`` is re-raised unchanged so the task
        is marked cancelled rather than failed.

        NOTE: The SSE :meth:`stream` path itself does NOT inspect
        ``task.exception()`` — it exits cleanly on ``task.done()``
        regardless of exception state. The re-raise contract is purely
        for testability + external observability: unit tests assert
        that a simulated broker outage surfaces via
        ``bridge._subscriber_task.exception()``, and future
        observability hooks (e.g. error-tracking sidecars that
        introspect completed tasks) get the failure for free. Runtime
        branching on the exception type inside the stream loop is
        intentionally out of scope for V1 — would require its own
        design + test (e.g. emitting a final ``event: error`` SSE
        frame before close).
        """
        try:
            try:
                client = await self._redis_factory()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Log + re-raise so the task carries the exception.
                # The finally on the outer try wakes the generator.
                logger.exception(
                    "staleness-alerts bridge: redis client factory failed; subscriber task exiting"
                )
                raise

            try:
                self._pubsub = client.pubsub()
                await self._pubsub.subscribe(ALERTS_CHANNEL)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    f"staleness-alerts bridge: failed to subscribe to "
                    f"{ALERTS_CHANNEL}; subscriber task exiting"
                )
                raise

            try:
                async for raw_msg in self._pubsub.listen():
                    if self._closed:
                        return
                    if raw_msg is None:
                        # End-of-stream sentinel from a fake (or a real
                        # pubsub that was closed beneath us).
                        return
                    # Only forward real channel messages, not subscribe-
                    # confirmation control frames.
                    if raw_msg.get("type") != "message":
                        continue
                    channel = raw_msg.get("channel")
                    if channel != ALERTS_CHANNEL:
                        continue
                    raw_data = raw_msg.get("data")
                    if raw_data is None:
                        continue
                    try:
                        payload = json.loads(raw_data)
                    except (TypeError, ValueError):
                        logger.warning(
                            "staleness-alerts bridge: skipping non-JSON "
                            "alert payload on channel=%s",
                            ALERTS_CHANNEL,
                        )
                        continue
                    # #391 monitoring box 3 + #404 cluster-wide dedup:
                    # record publish→receive latency BEFORE the brand
                    # filter so cross-brand alerts also contribute to
                    # the delivery-latency histogram. Uses the
                    # cluster-wide helper so multi-worker uvicorn /
                    # multi-pod K8s deployments emit at-most-one
                    # latency sample per alert_id globally (Redis SETNX
                    # claim with TTL); falls back to per-process LRU
                    # when Redis is unavailable. Best-effort;
                    # transport errors swallowed inside the helper.
                    await record_alert_latency_cluster(payload)
                    # Per-brand filter at the bridge layer. Multi-brand
                    # subscription is out of scope (issue #390 V1).
                    if not self._matches_brand(payload):
                        continue
                    self._enqueue_with_backpressure(payload)
            except asyncio.CancelledError:
                # Normal disconnect path — re-raise so the task is properly
                # marked cancelled and the lifecycle owner can await us.
                raise
            except Exception:
                # An unexpected error in the listen loop. Log + re-raise
                # so the exception is exposed via ``task.exception()`` for
                # tests and observability hooks; the stream loop itself
                # does NOT inspect it (exits on ``task.done()`` regardless).
                logger.exception(
                    f"staleness-alerts bridge: subscriber loop failed for brand={self.brand}"
                )
                raise
        finally:
            # Codex iter-1 H1: wake the SSE generator ON EVERY EXIT
            # (success, exception, cancellation). The generator's
            # ``while True`` loop re-checks ``self._subscriber_task.done()``
            # after waking and returns cleanly. Without this finally
            # clause, a subscriber failure that happened AFTER the
            # generator entered ``await self._item_ready.wait()`` would
            # leave the connection hung indefinitely.
            self._item_ready.set()

    def _matches_brand(self, payload: Dict[str, Any]) -> bool:
        """True if the alert payload is in scope for this connection's
        brand subscription.

        Match policy:
        * ``payload['brands']`` is a list — match if ``self.brand``
          appears in it (case-sensitive; brands are exact tokens).
        * Cross-brand alerts (``"all"`` in brands) also match every
          brand subscriber. This mirrors the
          :mod:`src.memory.lifecycle.invalidator` ``"all"``-brand
          convention used elsewhere in the codebase.
        * If ``brands`` is missing or empty, the alert does NOT match
          any single-brand subscriber. (A broadcast-without-brands is
          treated as out-of-scope — the publisher should populate the
          field.)
        """
        brands = payload.get("brands")
        if not isinstance(brands, list) or not brands:
            return False
        if self.brand in brands:
            return True
        if "all" in brands:
            return True
        return False

    def _enqueue_with_backpressure(self, payload: Dict[str, Any]) -> None:
        """Push ``payload`` onto the per-connection queue, dropping
        oldest events if the queue is at :data:`MAX_QUEUE_DEPTH`.
        """
        if len(self._queue) >= MAX_QUEUE_DEPTH:
            # Drop oldest. We log at WARNING (not ERROR) — backpressure
            # is a degraded mode, not a crash. The connection stays open.
            try:
                dropped = self._queue.popleft()
                dropped_sentinel_id = dropped.get("sentinel_id", "<unknown>")
            except IndexError:
                dropped_sentinel_id = "<empty>"
            self._dropped_for_backpressure += 1
            # Log only once every 25 drops to avoid log-spam under
            # sustained backpressure; the first drop ALWAYS logs so
            # operators see the signal immediately.
            if self._dropped_for_backpressure == 1 or self._dropped_for_backpressure % 25 == 0:
                logger.warning(
                    "staleness-alerts bridge: backpressure dropping oldest "
                    "alert for brand=%s; per-connection queue at depth=%d "
                    "max=%d total_dropped=%d dropped_sentinel_id=%s",
                    self.brand,
                    len(self._queue),
                    MAX_QUEUE_DEPTH,
                    self._dropped_for_backpressure,
                    dropped_sentinel_id,
                )
        self._queue.append(payload)
        self._item_ready.set()

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """Drain the per-connection queue as SSE-shaped dicts.

        Yields dicts of the form ``{"event": "alert", "data": <json>}``
        which ``sse_starlette.EventSourceResponse`` consumes and frames
        as the on-the-wire SSE protocol.

        Lifecycle ownership: this generator OWNS the subscriber task
        and the pubsub subscription. When the consumer closes the
        generator (e.g. client disconnect), the ``finally`` block
        cancels the subscriber and closes the pubsub.
        """
        # Spawn the subscriber task as a child of the current event
        # loop. We deliberately do NOT use asyncio.gather here — the
        # SSE generator's caller (uvicorn/starlette) is the lifecycle
        # owner; we just need a fire-and-forget background task we can
        # cancel on close.
        loop = asyncio.get_running_loop()
        self._subscriber_task = loop.create_task(self._subscribe_and_pump())

        try:
            while True:
                # Wait for new items.
                if not self._queue:
                    # If the subscriber task is done AND queue is empty,
                    # we're at end-of-stream (e.g. fake pubsub closed).
                    if self._subscriber_task is not None and self._subscriber_task.done():
                        return
                    await self._item_ready.wait()
                    # Loop back; in the meantime queue may have been
                    # drained by a previous yield, so always re-check.
                    self._item_ready.clear()
                    continue
                payload = self._queue.popleft()
                # Re-arm the event only after the queue drains fully.
                if not self._queue:
                    self._item_ready.clear()
                yield {
                    "event": "alert",
                    "data": json.dumps(payload),
                }
        except asyncio.CancelledError:
            # Normal disconnect — propagate so uvicorn finishes cleanly.
            raise
        finally:
            self._closed = True
            # Cancel subscriber task if still running.
            if self._subscriber_task is not None and not self._subscriber_task.done():
                self._subscriber_task.cancel()
                try:
                    await self._subscriber_task
                except (asyncio.CancelledError, Exception):
                    # Cancellation OR any exception during cleanup is
                    # acceptable — we already logged inside the task.
                    pass
            # Close the pubsub subscription. Best-effort — Redis client
            # itself is shared so we do NOT call its aclose().
            if self._pubsub is not None:
                try:
                    await self._pubsub.aclose()
                except Exception:
                    logger.exception(
                        "staleness-alerts bridge: pubsub.aclose() failed "
                        "during cleanup (best-effort; ignoring)"
                    )


@router.get("/stream")
async def alerts_stream(
    brand: str = Query(
        ...,
        description=(
            "Brand to subscribe to. Required; single-brand only for V1. "
            "Cross-brand alerts (publisher includes 'all' in brands) "
            "also reach every subscriber."
        ),
        min_length=1,
        max_length=64,
    ),
    user: Dict[str, Any] = Depends(require_auth),
) -> EventSourceResponse:
    """Subscribe to staleness alerts for ``brand`` via Server-Sent Events.

    Returns a ``text/event-stream`` response that stays open until the
    client disconnects. Each event:

    .. code-block:: text

        event: alert
        data: {"type": "staleness_alert", "sentinel_id": "...", "brands": ["<brand>"], "findings": [...]}

    Heartbeat: ``sse_starlette`` emits a ``: ping`` comment line every
    15 seconds by default to keep intermediaries from closing idle
    connections. Clients should ignore comment lines (per the SSE
    spec).
    """
    if not brand or not brand.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="brand query param is required and must be non-empty",
        )

    bridge = AlertBridge(brand=brand.strip())
    return EventSourceResponse(
        bridge.stream(),
        # 15s ping keeps proxies from idle-killing the connection.
        ping=15,
    )
