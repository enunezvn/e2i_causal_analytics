"""Unit tests for the staleness-alerts SSE bridge (#390).

The bridge subscribes to the Redis pub/sub channel ``e2i:alerts`` and
forwards alerts to connected CopilotKit clients via Server-Sent Events
at ``GET /api/alerts/stream``.

The five tests below pin the load-bearing behaviors stated in issue
#390's acceptance criteria:

* ``test_alerts_stream_requires_auth``                — 401 without bearer
* ``test_alerts_stream_yields_published_alert``       — single-brand alert
                                                        flows through
                                                        Redis pub/sub
                                                        within the SSE
                                                        response payload
* ``test_alerts_stream_filters_by_brand``             — per-brand subscription
                                                        drops other-brand
                                                        alerts at the
                                                        bridge layer
* ``test_alerts_stream_drops_oldest_under_backpressure``
                                                      — depth >100 → oldest
                                                        events dropped +
                                                        warning logged
                                                        (no disconnect)
* ``test_alerts_stream_handles_subscriber_disconnect_cleanly``
                                                      — client disconnect
                                                        cancels the bridge
                                                        task; Redis pubsub
                                                        ``aclose()`` invoked
                                                        on teardown

The Redis pub/sub subscription is injected via a fake-pubsub fixture so
the test never opens a real Redis connection. The integration test at
``tests/integration/test_staleness_alerts_e2e.py`` exercises a real
broker.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Fake pub/sub used by every test
# ---------------------------------------------------------------------------


class _FakePubSub:
    """Stand-in for ``redis.asyncio.client.PubSub`` used by the bridge.

    Tests push messages via :meth:`inject` (synchronous); :meth:`listen`
    yields them in FIFO order until the queue is closed via
    :meth:`stop`.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        self.subscribed_channels: List[str] = []
        self.closed: bool = False
        self.unsubscribed: bool = False

    async def subscribe(self, channel: str) -> None:
        self.subscribed_channels.append(channel)

    async def unsubscribe(self, *_channels: str) -> None:
        self.unsubscribed = True

    async def aclose(self) -> None:
        self.closed = True
        # Wake any pending `listen` call so the iterator can exit.
        await self._queue.put(None)

    async def listen(self) -> AsyncIterator[Dict[str, Any]]:
        while True:
            msg = await self._queue.get()
            if msg is None:
                return
            yield msg

    async def inject(self, payload: Dict[str, Any]) -> None:
        """Async helper to push an ``e2i:alerts`` message."""
        await self._queue.put(
            {
                "type": "message",
                "channel": "e2i:alerts",
                "data": json.dumps(payload),
            }
        )


class _FakeRedis:
    """Minimal stand-in for ``redis.asyncio.Redis`` that returns a fake
    pubsub. The bridge consumes the pubsub but never publishes back.
    """

    def __init__(self) -> None:
        self.pubsub_instance = _FakePubSub()
        self.closed: bool = False

    def pubsub(self) -> _FakePubSub:
        return self.pubsub_instance

    async def aclose(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Helper to drive the SSE generator
# ---------------------------------------------------------------------------


async def _collect_events(
    generator: AsyncIterator[Any],
    *,
    expected: int,
    timeout: float = 2.0,
) -> List[Any]:
    """Drive ``generator`` until ``expected`` events have been yielded or
    ``timeout`` elapses. Closes the generator on exit.
    """
    collected: List[Any] = []

    async def _drain() -> None:
        async for evt in generator:
            collected.append(evt)
            if len(collected) >= expected:
                return

    try:
        await asyncio.wait_for(_drain(), timeout=timeout)
    finally:
        await generator.aclose()
    return collected


# ===========================================================================
# Test 1: auth rejection
# ===========================================================================


def test_alerts_stream_requires_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unauthenticated GET → 401 from ``require_auth``.

    Testing mode is bypassed here (we monkeypatch ``TESTING_MODE`` False)
    so the auth path actually runs and rejects the request.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.dependencies import auth as auth_module
    from src.api.routes.staleness_alerts import router

    # Flip testing-mode off so require_auth actually runs.
    monkeypatch.setattr(auth_module, "TESTING_MODE", False)
    monkeypatch.setattr(auth_module, "SUPABASE_URL", "")
    monkeypatch.setattr(auth_module, "SUPABASE_ANON_KEY", "")

    app = FastAPI()
    app.include_router(router, prefix="/api")

    with TestClient(app) as client:
        response = client.get("/api/alerts/stream?brand=e2i")

    assert response.status_code == 401, (
        f"Expected 401 from unauthenticated SSE connect, got "
        f"{response.status_code}: {response.text}"
    )


# ===========================================================================
# Test 2: bridge yields published alert
# ===========================================================================


@pytest.mark.asyncio
async def test_alerts_stream_yields_published_alert() -> None:
    """A single ``e2i:alerts`` publish reaches the SSE consumer.

    We drive the bridge generator directly (it's a normal async iterator)
    rather than going through the FastAPI ``EventSourceResponse`` wrapper
    — the wrapper's framing is exercised in the integration test; this
    unit pins the SUBSCRIBER → QUEUE → CLIENT semantics.
    """
    from src.api.routes import staleness_alerts as mod

    fake_redis = _FakeRedis()

    async def _factory() -> _FakeRedis:
        return fake_redis  # type: ignore[return-value]

    payload = {
        "type": "staleness_alert",
        "sentinel_id": "s-1",
        "brands": ["e2i"],
        "findings": [{"finding_id": "f-1"}],
    }

    bridge = mod.AlertBridge(brand="e2i", redis_factory=_factory)
    gen = bridge.stream()
    # Yield once so the bridge's background subscriber starts.
    await asyncio.sleep(0)
    # Inject after subscribe so the FIFO order is deterministic.
    await asyncio.sleep(0)
    await fake_redis.pubsub_instance.inject(payload)

    events = await _collect_events(gen, expected=1, timeout=2.0)

    assert len(events) == 1
    body = events[0]
    # The SSE shape exposes the JSON payload as the event's data field.
    # Bridge yields dicts {"event": "alert", "data": <json-str>}; the
    # EventSourceResponse renders both into the SSE wire format.
    assert body["event"] == "alert"
    assert json.loads(body["data"]) == payload


# ===========================================================================
# Test 3: per-brand filter
# ===========================================================================


@pytest.mark.asyncio
async def test_alerts_stream_filters_by_brand() -> None:
    """Only alerts whose payload ``brands`` includes the subscriber's
    brand reach the client. Multi-brand subscription is out of scope
    (issue #390); single-brand is the contract.
    """
    from src.api.routes import staleness_alerts as mod

    fake_redis = _FakeRedis()

    async def _factory() -> _FakeRedis:
        return fake_redis  # type: ignore[return-value]

    bridge = mod.AlertBridge(brand="e2i", redis_factory=_factory)
    gen = bridge.stream()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Two alerts: only the e2i one should pass the brand filter.
    other_brand_payload = {
        "type": "staleness_alert",
        "sentinel_id": "s-other",
        "brands": ["FabhaltaUS"],
        "findings": [],
    }
    matching_payload = {
        "type": "staleness_alert",
        "sentinel_id": "s-match",
        "brands": ["e2i"],
        "findings": [],
    }
    await fake_redis.pubsub_instance.inject(other_brand_payload)
    await fake_redis.pubsub_instance.inject(matching_payload)

    events = await _collect_events(gen, expected=1, timeout=2.0)

    assert len(events) == 1
    body = events[0]
    assert json.loads(body["data"]) == matching_payload


# ===========================================================================
# Test 4: backpressure drops oldest
# ===========================================================================


@pytest.mark.asyncio
async def test_alerts_stream_drops_oldest_under_backpressure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If a slow consumer falls behind, the queue caps at
    :data:`MAX_QUEUE_DEPTH` (100); oldest events are dropped, and a
    warning is logged. The connection MUST remain open.
    """
    from src.api.routes import staleness_alerts as mod

    fake_redis = _FakeRedis()

    async def _factory() -> _FakeRedis:
        return fake_redis  # type: ignore[return-value]

    bridge = mod.AlertBridge(brand="e2i", redis_factory=_factory)

    # Drive the subscriber loop a few times so the bridge starts
    # subscribing and consuming. We do NOT pull from `gen` here — the
    # whole point is to simulate a slow consumer that never drains.
    gen = bridge.stream()
    # Yield repeatedly to let the bridge subscriber task start.
    for _ in range(5):
        await asyncio.sleep(0)

    # Inject MAX_QUEUE_DEPTH + 25 events with sequential ids. The
    # subscriber's queue is bounded at MAX_QUEUE_DEPTH (100), so 25
    # events should be evicted from the head.
    total_published = mod.MAX_QUEUE_DEPTH + 25
    for i in range(total_published):
        await fake_redis.pubsub_instance.inject(
            {
                "type": "staleness_alert",
                "sentinel_id": f"s-{i}",
                "brands": ["e2i"],
                "findings": [],
            }
        )
        # Yield so the bridge task can move messages into its queue
        # before the queue overflow happens. Without this the
        # injection-side queue (unbounded) absorbs them all in one go.
        await asyncio.sleep(0)

    # Give the bridge a moment to drain the injection queue into the
    # per-connection bounded queue and trigger the drop-oldest path.
    for _ in range(10):
        await asyncio.sleep(0)

    # Now drain the SSE side: should see exactly MAX_QUEUE_DEPTH events,
    # and the FIRST should be one of the LATER published ones (oldest
    # 25 were dropped).
    with caplog.at_level(logging.WARNING, logger="src.api.routes.staleness_alerts"):
        events = await _collect_events(gen, expected=mod.MAX_QUEUE_DEPTH, timeout=5.0)

    assert len(events) == mod.MAX_QUEUE_DEPTH, (
        f"Expected {mod.MAX_QUEUE_DEPTH} events after drop-oldest, got {len(events)}"
    )

    # The first surviving event MUST NOT be sentinel-id s-0 — the
    # earliest events were evicted under backpressure. Specifically, at
    # least 25 events at the head should have been dropped.
    first_id = json.loads(events[0]["data"])["sentinel_id"]
    first_index = int(first_id.split("-", 1)[1])
    assert first_index >= 25, (
        f"Expected oldest 25 events to be dropped under backpressure; "
        f"first surviving event id was s-{first_index}"
    )

    # A WARNING about backpressure must have been logged at least once.
    backpressure_warns = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "backpressure" in r.getMessage().lower()
    ]
    assert backpressure_warns, (
        "Expected a WARNING-level log mentioning 'backpressure' when "
        "the per-connection queue overflowed; got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


# ===========================================================================
# Test 5: clean disconnect
# ===========================================================================


@pytest.mark.asyncio
async def test_alerts_stream_handles_subscriber_disconnect_cleanly() -> None:
    """When the client closes the connection mid-stream, the bridge
    cancels its background subscriber task and calls ``pubsub.aclose()``.
    No orphan tasks should remain.
    """
    from src.api.routes import staleness_alerts as mod

    fake_redis = _FakeRedis()

    async def _factory() -> _FakeRedis:
        return fake_redis  # type: ignore[return-value]

    bridge = mod.AlertBridge(brand="e2i", redis_factory=_factory)
    gen = bridge.stream()

    # Drive the bridge so it subscribes.
    for _ in range(5):
        await asyncio.sleep(0)

    # Push a single event, consume it, then close the generator to
    # simulate the client disconnect.
    await fake_redis.pubsub_instance.inject(
        {
            "type": "staleness_alert",
            "sentinel_id": "s-disconnect",
            "brands": ["e2i"],
            "findings": [],
        }
    )

    # First event arrives.
    first = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
    assert json.loads(first["data"])["sentinel_id"] == "s-disconnect"

    # Close the generator — this simulates the client disconnect path.
    await gen.aclose()

    # Yield repeatedly to let the bridge's cleanup task settle.
    for _ in range(10):
        await asyncio.sleep(0)

    # The fake pubsub should have been closed.
    assert fake_redis.pubsub_instance.closed, (
        "Bridge MUST call pubsub.aclose() on client disconnect to "
        "release the Redis subscription; instead the fake pubsub was "
        "not closed."
    )

    # The subscriber background task MUST be done (no orphan).
    task = bridge._subscriber_task  # type: ignore[attr-defined]
    assert task is not None
    assert task.done(), (
        "Subscriber background task MUST be cancelled / completed when "
        "the SSE generator is closed; instead it is still pending."
    )
