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

import pytest

from src.api.dependencies import redis_client

# Generous against a 16 s backoff, tight against any retry wait (min 2 s).
FAST_SECONDS = 1.0


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


@pytest.mark.unit
async def test_a_degraded_call_schedules_one_background_reconnect_per_process(redis_down):
    """Recovery is not lost: the first degraded call starts ONE ``init_redis()``
    in the background; further degraded calls while it runs start no other."""
    from src.api.dependencies.durable_job_store import DurableJobStore
    from src.api.schemas.causal import DiscoverEffectsResponse

    store = DurableJobStore("test:1999", DiscoverEffectsResponse)
    await store.get("a")
    task = redis_client._reconnect_task
    assert task is not None

    await store.get("b")
    await store.get("c")
    assert redis_client._reconnect_task is task


@pytest.mark.unit
async def test_close_redis_cancels_a_pending_reconnect(redis_down):
    """Shutdown must not leave a reconnect that would install a client after
    ``close_redis()`` (a leaked pool on a closing loop)."""
    with pytest.raises(RuntimeError):
        await redis_client.request_path_client()
    task = redis_client._reconnect_task
    assert task is not None and not task.done()

    await redis_client.close_redis()

    assert task.cancelled()
    assert redis_client._reconnect_task is None
    assert redis_client.current_client() is None


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
