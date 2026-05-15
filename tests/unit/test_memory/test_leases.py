"""Unit tests for AgentLease (subsystem 5)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

from src.memory.coordination.leases import AgentLease, LeaseAcquisitionError


class FakeRedis:
    """In-memory fake that implements just enough of redis-asyncio for AgentLease."""

    def __init__(self) -> None:
        self.store: Dict[str, str] = {}

    async def set(
        self,
        key: str,
        value: str,
        nx: bool = False,
        px: Optional[int] = None,
    ) -> Optional[bool]:
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def get(self, key: str) -> Optional[str]:
        return self.store.get(key)

    async def eval(self, script: str, numkeys: int, *args: Any) -> int:
        # Both helper scripts (renew/release) check that GET(KEYS[1]) == ARGV[1].
        key = args[0]
        holder = args[1]
        current = self.store.get(key)
        if current != holder:
            return 0
        if "del" in script:
            del self.store[key]
            return 1
        if "pexpire" in script:
            return 1
        return 0


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture(autouse=True)
def patch_redis(fake_redis):
    with patch("src.memory.coordination.leases.get_redis_client", return_value=fake_redis):
        yield


@pytest.mark.asyncio
async def test_acquire_succeeds_when_free(fake_redis: FakeRedis):
    lease = AgentLease("cohort", "abc", ttl_seconds=60)
    assert await lease.acquire() is True
    assert fake_redis.store["lease:cohort:abc"] == lease.holder_id


@pytest.mark.asyncio
async def test_acquire_fails_when_held(fake_redis: FakeRedis):
    held = AgentLease("cohort", "abc")
    assert await held.acquire() is True

    contender = AgentLease("cohort", "abc")
    assert await contender.acquire() is False  # SET NX returns nil


@pytest.mark.asyncio
async def test_release_only_by_holder(fake_redis: FakeRedis):
    held = AgentLease("cohort", "abc")
    await held.acquire()

    # Different holder cannot release.
    impostor = AgentLease("cohort", "abc")
    # impostor doesn't think it's held, so release short-circuits to False:
    assert await impostor.release() is False
    # but even if impostor lies about ownership, the Lua-script branch
    # checks GET == ARGV[1] and returns 0 -- emulate by manually setting _held.
    impostor._held = True
    assert await impostor.release() is False

    # Owner can release.
    assert await held.release() is True
    assert "lease:cohort:abc" not in fake_redis.store


@pytest.mark.asyncio
async def test_renew_only_by_holder(fake_redis: FakeRedis):
    lease = AgentLease("cohort", "abc")
    await lease.acquire()
    assert await lease.renew(ttl_seconds=120) is True

    # Impostor renew is rejected.
    impostor = AgentLease("cohort", "abc")
    impostor._held = True  # lie about holding it
    assert await impostor.renew() is False


@pytest.mark.asyncio
async def test_acquire_waits_until_released(fake_redis: FakeRedis):
    held = AgentLease("cohort", "abc")
    await held.acquire()

    async def releaser():
        await asyncio.sleep(0.05)
        await held.release()

    contender = AgentLease("cohort", "abc")
    asyncio.create_task(releaser())
    ok = await contender.acquire(wait_seconds=1.0, poll_interval=0.01)
    assert ok is True
    assert fake_redis.store["lease:cohort:abc"] == contender.holder_id


@pytest.mark.asyncio
async def test_context_manager_raises_on_contention(fake_redis: FakeRedis):
    blocker = AgentLease("cohort", "abc")
    await blocker.acquire()

    with pytest.raises(LeaseAcquisitionError):
        async with AgentLease("cohort", "abc", ttl_seconds=60):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_current_holder_returns_holder_id(fake_redis: FakeRedis):
    lease = AgentLease("model_version", "m42")
    await lease.acquire()
    assert await lease.current_holder() == lease.holder_id
