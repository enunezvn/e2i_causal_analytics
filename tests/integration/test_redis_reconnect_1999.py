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
