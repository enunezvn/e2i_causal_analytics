"""#1999 item 3, recovery half, against a REAL Redis.

The request-path helpers stopped calling ``get_redis()`` (whose
``init_redis()`` backoff blocked a request 16 s when the lifespan had started
in Redis-degraded mode). What ``get_redis()`` also gave the durable stores was
recovery: once Redis came back, the next call reconnected. This pins that the
background reconnect keeps it: a store that degraded while Redis was down
becomes durable again, readable by a separate client (i.e. by another worker),
with no request waiting on the reconnect.

"Down" is a closed local port; "back" is the real Redis this lane runs with
(``REDIS_URL``, as tests/conftest.py resolves it for ``requires_redis``).
"""

from __future__ import annotations

import asyncio
import os
import socket
import uuid

import pytest
import redis.asyncio as aioredis

from src.api.dependencies import redis_client
from src.api.dependencies.durable_job_store import DurableJobStore
from src.api.schemas.causal import DiscoveredEffect, DiscoverEffectsResponse

pytestmark = [pytest.mark.integration, pytest.mark.requires_redis]

_PASSWORD = os.getenv("REDIS_PASSWORD")
REAL_REDIS_URL = os.getenv("REDIS_URL") or (
    f"redis://:{_PASSWORD}@localhost:6382" if _PASSWORD else "redis://localhost:6382"
)


def _refused_url() -> str:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"redis://127.0.0.1:{port}"


def _job() -> DiscoverEffectsResponse:
    return DiscoverEffectsResponse(
        job_id="j",
        status="running",
        dataset="patient_journeys",
        total=1,
        completed=0,
        effects=[DiscoveredEffect(treatment="t", outcome="o", status="pending")],
    )


async def _settle(task) -> None:
    assert task is not None
    await asyncio.wait_for(asyncio.shield(task), timeout=10)


async def test_a_store_degraded_at_startup_becomes_durable_once_redis_is_back(monkeypatch):
    prefix = f"test:1999:recover:{uuid.uuid4().hex}"
    monkeypatch.setattr(redis_client, "_redis_client", None)
    monkeypatch.setattr(redis_client, "RECONNECT_COOLDOWN_SECONDS", 0.2)
    monkeypatch.setattr(redis_client, "REDIS_URL", _refused_url())
    store = DurableJobStore(prefix, DiscoverEffectsResponse)
    reader = aioredis.from_url(REAL_REDIS_URL, decode_responses=True)
    try:
        # Redis down: the write degrades to memory at once; one reconnect round runs and fails.
        await store.set("while-down", _job())
        assert store._last_durable is False
        await _settle(redis_client._reconnect_task)
        assert redis_client.current_client() is None

        # Redis back (after the cooldown): the next degraded call schedules a round that succeeds.
        monkeypatch.setattr(redis_client, "REDIS_URL", REAL_REDIS_URL)
        await asyncio.sleep(0.25)
        await store.set("still-degraded", _job())
        assert store._last_durable is False  # this call did not wait for the reconnect
        await _settle(redis_client._reconnect_task)
        assert redis_client.current_client() is not None

        # Durable again: another client (another worker) reads what this store writes.
        await store.set("after-recovery", _job())
        assert store._last_durable is True
        assert await reader.get(f"{prefix}:after-recovery") is not None
        assert await reader.get(f"{prefix}:while-down") is None  # memory-only, as designed
    finally:
        for key in await reader.keys(f"{prefix}:*"):
            await reader.delete(key)
        await reader.aclose()
        await redis_client.close_redis()


# --- Codex round 2: competing SUCCESSFUL initialisers (needs pings that succeed) ---


async def _get_redis_losing_publication_while(monkeypatch, swap_to) -> tuple:
    """``get_redis()`` whose init attempt pings OK but finds a client already
    published (``winner``); while it discards its own candidate, another
    initialiser replaces the module client with ``swap_to``. The hook wraps the
    real ``_discard`` only to place that interleaving deterministically."""
    monkeypatch.setattr(redis_client, "REDIS_URL", REAL_REDIS_URL)
    monkeypatch.setattr(redis_client, "_redis_client", None)
    winner = aioredis.from_url(REAL_REDIS_URL)
    real_discard = redis_client._discard

    async def _discard_while_another_initialiser_publishes(client):
        redis_client._redis_client = swap_to
        await real_discard(client)

    monkeypatch.setattr(redis_client, "_discard", _discard_while_another_initialiser_publishes)
    task = asyncio.create_task(redis_client.get_redis())
    await asyncio.sleep(0)  # the attempt is now inside its candidate's PING (socket I/O)
    assert not task.done()
    redis_client._redis_client = winner
    returned = await asyncio.wait_for(task, timeout=10)
    return winner, returned


async def test_get_redis_never_returns_none_when_the_client_is_cleared_during_cleanup(
    monkeypatch,
):
    winner, returned = await _get_redis_losing_publication_while(monkeypatch, swap_to=None)
    try:
        assert returned is winner  # a validated client, never None
    finally:
        redis_client._redis_client = None
        await winner.aclose()


async def test_get_redis_does_not_overwrite_a_client_published_during_cleanup(monkeypatch):
    replacement = aioredis.from_url(REAL_REDIS_URL)
    winner, returned = await _get_redis_losing_publication_while(monkeypatch, swap_to=replacement)
    try:
        assert returned is winner
        assert redis_client.current_client() is replacement  # not overwritten by get_redis
    finally:
        redis_client._redis_client = None
        await winner.aclose()
        await replacement.aclose()


async def test_a_cancelled_attempt_still_closes_the_candidate_it_lost_with(monkeypatch):
    """Codex round 3: an attempt whose candidate pinged OK but lost publication
    is cancelled (close_redis() at shutdown) WHILE it closes that candidate. The
    close must still complete: the candidate's connection ends disconnected."""
    monkeypatch.setattr(redis_client, "REDIS_URL", REAL_REDIS_URL)
    monkeypatch.setattr(redis_client, "_redis_client", None)
    winner = aioredis.from_url(REAL_REDIS_URL)
    real_discard = redis_client._discard
    lost: list = []

    async def _discard_cancelled_midway(client):
        lost.append(client)
        asyncio.current_task().cancel()  # delivered at the first await inside the close
        await real_discard(client)

    monkeypatch.setattr(redis_client, "_discard", _discard_cancelled_midway)
    task = asyncio.create_task(redis_client._connect_once())
    await asyncio.sleep(0)  # the attempt is now inside its candidate's PING (socket I/O)
    redis_client._redis_client = winner
    try:
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=10)
        assert task.cancelled()
        assert redis_client.current_client() is winner
        (candidate,) = lost
        pool = candidate.connection_pool
        connections = list(pool._available_connections) + list(pool._in_use_connections)
        assert connections, "the candidate never connected (interleaving not exercised)"
        assert not any(c.is_connected for c in connections)
    finally:
        redis_client._redis_client = None
        await winner.aclose()
