"""#1999 item 3: request-path Redis helpers never run ``init_redis()``'s backoff.

``get_redis()`` runs ``init_redis()`` (tenacity: 5 attempts, exponential wait)
when the module client is unset, i.e. when the lifespan started in
Redis-degraded mode. Measured before this fix with the client unset and a
refused REDIS_URL: ``get_redis()`` raised after 16.0 s, and so did every
``DurableJobStore.get()`` on the default factory. The same default factory
(``await get_redis()``) backs the segments and resource-optimizer durable
stores. ``InflightLock`` avoided it by reading the private
``redis_client._redis_client``.

The fix: ``redis_client.current_client()`` (the initialised client or None,
never connects) and ``request_path_client()`` (that client, or schedule ONE
background ``init_redis()`` for the process and raise ``RuntimeError``, which
every store already degrades on). The background reconnect keeps what
``get_redis()`` gave these stores: they become durable again once Redis is back
(``DurableJobStore``'s documented "transient outage" mode), without a request
waiting for it.

No Redis double: the "down" Redis is a real redis-py client against a closed
local port (connect refused instantly). The recovery half needs a reachable
Redis and lives in tests/integration/test_redis_reconnect_1999.py.
"""

from __future__ import annotations

import asyncio
import socket
import time
from typing import List

import pytest

from src.api.dependencies import redis_client

# Generous against a 16 s backoff, tight against any retry wait (min 2 s).
FAST_SECONDS = 1.0
# Socket timeout for the hung-server tests: far longer than any test, so a parked
# PING ends only when the test releases the peer (never by a wall-clock race).
HUNG_TIMEOUT_SECONDS = 20.0


def _refused_url() -> str:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"redis://127.0.0.1:{port}"


@pytest.fixture
async def redis_down(monkeypatch):
    """Lifespan started in degraded mode: no module client, Redis refusing."""
    monkeypatch.setattr(redis_client, "REDIS_URL", _refused_url())
    monkeypatch.setattr(redis_client, "_redis_client", None)
    try:
        yield
    finally:
        # Cancels a background reconnect this test scheduled (and closes nothing
        # else: the client is unset).
        await redis_client.close_redis()


async def _timed(coro):
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    result = await coro
    return result, loop.time() - t0


@pytest.mark.unit
def test_current_client_is_the_initialised_client_or_none_and_never_connects(monkeypatch):
    monkeypatch.setattr(redis_client, "_redis_client", None)
    assert redis_client.current_client() is None

    client = redis_client.aioredis.from_url(_refused_url())  # no connection is made
    monkeypatch.setattr(redis_client, "_redis_client", client)
    assert redis_client.current_client() is client


@pytest.mark.unit
async def test_durable_job_store_degrades_without_waiting_on_the_init_backoff(redis_down):
    from src.api.dependencies.durable_job_store import DurableJobStore
    from src.api.schemas.causal import DiscoverEffectsResponse

    store = DurableJobStore("test:1999", DiscoverEffectsResponse)
    result, elapsed = await _timed(store.get("missing"))

    assert result is None
    assert elapsed < FAST_SECONDS, f"request path waited {elapsed:.1f}s on Redis init"
    assert store._last_durable is False


@pytest.mark.unit
async def test_segments_store_degrades_without_waiting_on_the_init_backoff(redis_down):
    from src.api.routes.segments import _DurableAnalysesStore

    durable, elapsed = await _timed(_DurableAnalysesStore().is_durable())

    assert durable is False
    assert elapsed < FAST_SECONDS, f"request path waited {elapsed:.1f}s on Redis init"


@pytest.mark.unit
async def test_resource_optimizer_store_degrades_without_waiting_on_the_init_backoff(redis_down):
    from src.api.routes.resource_optimizer import _DurableOptimizationsStore

    durable, elapsed = await _timed(_DurableOptimizationsStore().is_durable())

    assert durable is False
    assert elapsed < FAST_SECONDS, f"request path waited {elapsed:.1f}s on Redis init"


@pytest.mark.unit
async def test_inflight_lock_degrades_to_local_without_waiting(redis_down):
    from src.api.dependencies.inflight_lock import InflightLock

    lock = InflightLock("test:1999:inflight")
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    async with lock.hold("rid") as lease:
        elapsed = loop.time() - t0
        assert lease.mode == "local"
    assert elapsed < FAST_SECONDS


class _HungPeer:
    """A TCP peer that accepts and never replies until ``release()`` closes its
    connections (the parked PING then fails at once with a connection error)."""

    def __init__(self) -> None:
        self.accepted: List[asyncio.StreamWriter] = []

    def __len__(self) -> int:
        return len(self.accepted)

    async def release(self) -> None:
        for writer in self.accepted:
            writer.close()
        await asyncio.sleep(0.05)


@pytest.fixture
async def redis_hung(monkeypatch):
    """Lifespan started degraded, and Redis now ACCEPTS connections but never
    answers (a hung server): an attempt parks inside its PING until the test
    releases the peer, so a test acts while the attempt is provably in flight.
    A real TCP peer, not a Redis double: it never sends a byte."""
    peer = _HungPeer()

    async def _accept(reader, writer):
        peer.accepted.append(writer)

    server = await asyncio.start_server(_accept, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(redis_client, "REDIS_URL", f"redis://127.0.0.1:{port}")
    monkeypatch.setattr(redis_client, "REDIS_SOCKET_TIMEOUT", HUNG_TIMEOUT_SECONDS)
    monkeypatch.setattr(redis_client, "_redis_client", None)
    try:
        yield peer
    finally:
        await redis_client.close_redis()
        await peer.release()
        server.close()
        await server.wait_closed()


async def _reconnect_inside_ping(peer: _HungPeer) -> "asyncio.Task[None]":
    """Trigger a degraded call and return the reconnect task once the hung
    server has accepted its connection (the round is now awaiting PING)."""
    with pytest.raises(RuntimeError):
        await redis_client.request_path_client()
    task = redis_client._reconnect_task
    assert task is not None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + FAST_SECONDS
    while not peer:
        assert loop.time() < deadline, "the reconnect never connected"
        await asyncio.sleep(0.01)
    assert not task.done()
    return task


@pytest.mark.unit
async def test_a_reconnect_in_flight_publishes_nothing_and_requests_still_degrade_at_once(
    redis_hung,
):
    """Codex round 1 MED: an unverified client must not be visible. While the
    round waits on PING, request paths see no client, degrade immediately and
    report non-durable, and further degraded calls start no second round."""
    from src.api.dependencies.durable_job_store import DurableJobStore
    from src.api.schemas.causal import DiscoverEffectsResponse

    task = await _reconnect_inside_ping(redis_hung)
    store = DurableJobStore("test:1999", DiscoverEffectsResponse)

    assert redis_client.current_client() is None
    result, elapsed = await _timed(store.get("missing"))
    assert result is None and elapsed < 0.5, elapsed
    durable, elapsed = await _timed(store.is_durable())
    assert durable is False and elapsed < 0.5, elapsed
    assert redis_client._reconnect_task is task  # single-flight while in flight
    assert len(redis_hung) == 1


@pytest.mark.unit
async def test_close_redis_cancels_a_reconnect_that_is_inside_its_ping(redis_hung):
    """Shutdown cancels an IN-FLIGHT round (not one that never started), and no
    client appears afterwards."""
    task = await _reconnect_inside_ping(redis_hung)

    await redis_client.close_redis()

    assert task.cancelled()
    assert redis_client._reconnect_task is None
    await redis_hung.release()  # whatever was still pending on the peer ends now
    assert redis_client.current_client() is None


@pytest.mark.unit
async def test_a_failed_attempt_does_not_clear_a_client_published_during_its_ping(redis_hung):
    """Codex round 1 MED: two initialisers (the background round and /ready's
    ``get_redis()``) must not undo each other. A client published while this
    attempt waits on PING survives the attempt's failure, and the attempt's own
    candidate is never published."""
    attempt = asyncio.create_task(redis_client._connect_once())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + FAST_SECONDS
    while not redis_hung:
        assert loop.time() < deadline, "the attempt never connected"
        await asyncio.sleep(0.01)

    assert not attempt.done()  # parked on the peer until released
    winner = redis_client.aioredis.from_url(_refused_url())  # published by "another initialiser"
    redis_client._redis_client = winner
    await redis_hung.release()  # now the attempt's PING fails

    with pytest.raises(ConnectionError):
        await asyncio.wait_for(attempt, timeout=FAST_SECONDS)
    assert redis_client.current_client() is winner
    redis_client._redis_client = None  # never connected; nothing to close
    await winner.aclose()


@pytest.mark.unit
async def test_a_failed_reconnect_is_not_retried_inside_the_cooldown(redis_down, monkeypatch):
    """No storm while Redis stays down: a background round is ONE connection
    attempt (the refused port fails it at once), and after it fails, degraded
    calls within the cooldown start no new round; after the cooldown, one does."""
    monkeypatch.setattr(redis_client, "RECONNECT_COOLDOWN_SECONDS", 0.3)

    with pytest.raises(RuntimeError):
        await redis_client.request_path_client()
    first = redis_client._reconnect_task
    assert first is not None
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await asyncio.wait_for(asyncio.shield(first), timeout=FAST_SECONDS)
    assert loop.time() - t0 < FAST_SECONDS  # one attempt, not the 16 s backoff
    assert redis_client.current_client() is None

    with pytest.raises(RuntimeError):
        await redis_client.request_path_client()
    assert redis_client._reconnect_task is first  # inside the cooldown

    await asyncio.sleep(0.35)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        await redis_client.request_path_client()
    assert time.monotonic() - t0 < FAST_SECONDS
    assert redis_client._reconnect_task is not first
