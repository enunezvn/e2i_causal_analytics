"""Unit coverage for DurableJobStore — the cross-worker job store that fixes the
submit-on-worker-A / poll-on-worker-B 404 under gunicorn --workers 2.

The decisive test is ``test_cross_worker_shared_redis``: two *separate* store
instances (standing in for two worker processes) sharing one Redis must see each
other's writes — exactly what a module-level dict could not do.
"""

import pytest

from src.api.dependencies.durable_job_store import DurableJobStore
from src.api.schemas.causal import DiscoveredEffect, DiscoverEffectsResponse


def _resp(job_id: str, status: str = "pending") -> DiscoverEffectsResponse:
    return DiscoverEffectsResponse(
        job_id=job_id,
        status=status,
        dataset="patient_journeys",
        total=1,
        completed=0,
        effects=[DiscoveredEffect(treatment="t", outcome="o", status="pending")],
    )


class _FakeRedis:
    """Minimal async Redis stand-in shared across 'workers'."""

    def __init__(self) -> None:
        self.kv: dict = {}

    async def set(self, key, value, ex=None):
        self.kv[key] = value

    async def get(self, key):
        return self.kv.get(key)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cross_worker_shared_redis_sees_each_others_writes():
    fake = _FakeRedis()

    async def factory():
        return fake

    worker_a = DurableJobStore("test:cw", DiscoverEffectsResponse, redis_factory=factory)
    worker_b = DurableJobStore("test:cw", DiscoverEffectsResponse, redis_factory=factory)

    # POST handled by worker A.
    await worker_a.set("job1", _resp("job1", "completed"))
    # Poll handled by worker B (a different process) -> must find it via Redis.
    got = await worker_b.get("job1")
    assert got is not None
    assert got.job_id == "job1"
    assert got.status == "completed"
    assert await worker_b.is_durable() is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_fallback_when_redis_unavailable():
    async def boom():
        raise RuntimeError("redis not initialised")

    store = DurableJobStore("test:mem", DiscoverEffectsResponse, redis_factory=boom)
    await store.set("j", _resp("j", "running"))
    got = await store.get("j")
    assert got is not None and got.status == "running"  # served from memory mirror
    assert await store.get("missing") is None
    assert await store.is_durable() is False  # degraded mode surfaced for /health


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redis_command_failure_degrades_to_memory():
    class _FlakyRedis:
        async def set(self, *a, **k):
            raise ConnectionError("down")

        async def get(self, *a, **k):
            raise ConnectionError("down")

    async def factory():
        return _FlakyRedis()

    store = DurableJobStore("test:flaky", DiscoverEffectsResponse, redis_factory=factory)
    await store.set("j", _resp("j", "running"))  # Redis SET raises -> memory mirror
    got = await store.get("j")  # Redis GET raises -> memory fallback
    assert got is not None and got.status == "running"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_is_fifo_bounded():
    async def boom():
        raise RuntimeError("no redis")

    store = DurableJobStore(
        "test:bound", DiscoverEffectsResponse, redis_factory=boom, max_fallback=2
    )
    await store.set("a", _resp("a"))
    await store.set("b", _resp("b"))
    await store.set("c", _resp("c"))  # evicts oldest ("a")
    assert await store.get("a") is None
    assert (await store.get("b")) is not None
    assert (await store.get("c")) is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_undecodable_record_degrades_not_500():
    """A corrupted / schema-drifted Redis value (e.g. old-format record read by
    new code during a rolling deploy) must degrade, never 500."""

    class _CorruptRedis:
        async def set(self, *a, **k):
            pass  # no-op; the store mirrors a valid copy to memory

        async def get(self, key):
            return b'{"truncated":'  # invalid JSON for ANY key

    async def factory():
        return _CorruptRedis()

    store = DurableJobStore("test:corrupt", DiscoverEffectsResponse, redis_factory=factory)
    # set() mirrors a valid copy to memory; the Redis read returns garbage ->
    # undecodable -> must fall through to the valid memory copy (no exception).
    await store.set("j", _resp("j", "running"))
    got = await store.get("j")
    assert got is not None and got.status == "running"
    # Unknown key: garbage from Redis + nothing in memory -> None, never raises.
    assert await store.get("nope") is None


# ---------------------------------------------------------------------------
# Markers: a tiny sidecar flag per job (e.g. "cancel") that another worker can
# raise without a read-modify-write race against the task's own row publishes.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_marker_is_visible_across_workers():
    """The cancel POST lands on worker B while the task runs on worker A: the
    marker must be raised in Redis, not in B's process memory."""
    fake = _FakeRedis()

    async def factory():
        return fake

    worker_a = DurableJobStore("test:mk", DiscoverEffectsResponse, redis_factory=factory)
    worker_b = DurableJobStore("test:mk", DiscoverEffectsResponse, redis_factory=factory)
    assert await worker_a.has_marker("job1", "cancel") is False
    await worker_b.set_marker("job1", "cancel")
    assert await worker_a.has_marker("job1", "cancel") is True
    # Namespaced per job AND per marker name.
    assert await worker_a.has_marker("job2", "cancel") is False
    assert await worker_a.has_marker("job1", "other") is False
    assert "test:mk:job1:cancel" in fake.kv


@pytest.mark.unit
@pytest.mark.asyncio
async def test_marker_degrades_to_memory_without_redis():
    async def boom():
        raise RuntimeError("no redis")

    store = DurableJobStore("test:mk-mem", DiscoverEffectsResponse, redis_factory=boom)
    assert await store.has_marker("j", "cancel") is False
    await store.set_marker("j", "cancel")
    assert await store.has_marker("j", "cancel") is True


# ---------------------------------------------------------------------------
# Timestamped markers (liveness heartbeats): a task refreshes one periodically;
# a poll on ANY worker reads its age to tell a live job from an orphaned one.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_touched_marker_age_is_visible_across_workers(monkeypatch):
    """The task on worker A touches the heartbeat; the poll on worker B reads
    its age from Redis. Absent marker -> None (never a fake 0)."""
    import src.api.dependencies.durable_job_store as mod

    fake = _FakeRedis()

    async def factory():
        return fake

    worker_a = DurableJobStore("test:hb", DiscoverEffectsResponse, redis_factory=factory)
    worker_b = DurableJobStore("test:hb", DiscoverEffectsResponse, redis_factory=factory)
    assert await worker_b.marker_age_seconds("job1", "alive") is None

    now = {"t": 1_000.0}
    monkeypatch.setattr(mod.time, "time", lambda: now["t"])
    await worker_a.touch_marker("job1", "alive", ttl_seconds=120)
    assert "test:hb:job1:alive" in fake.kv
    now["t"] += 42.0
    age = await worker_b.marker_age_seconds("job1", "alive")
    assert age == pytest.approx(42.0)
    # A fresh touch resets the age.
    await worker_a.touch_marker("job1", "alive", ttl_seconds=120)
    assert await worker_b.marker_age_seconds("job1", "alive") == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_touched_marker_age_degrades_to_memory_without_redis(monkeypatch):
    """Without Redis the age still ages (memory has no TTL; the caller compares
    the age against its own budget) so a same-worker orphan check still works."""
    import src.api.dependencies.durable_job_store as mod

    async def boom():
        raise RuntimeError("no redis")

    store = DurableJobStore("test:hb-mem", DiscoverEffectsResponse, redis_factory=boom)
    now = {"t": 5_000.0}
    monkeypatch.setattr(mod.time, "time", lambda: now["t"])
    assert await store.marker_age_seconds("j", "alive") is None
    await store.touch_marker("j", "alive", ttl_seconds=120)
    now["t"] += 130.0
    assert await store.marker_age_seconds("j", "alive") == pytest.approx(130.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_marker_without_a_timestamp_reads_as_absent():
    """A tampered / legacy value must read as 'no heartbeat', never crash the poll."""
    fake = _FakeRedis()

    async def factory():
        return fake

    store = DurableJobStore("test:hb-bad", DiscoverEffectsResponse, redis_factory=factory)
    fake.kv["test:hb-bad:k:alive"] = "not-a-number"
    assert await store.marker_age_seconds("k", "alive") is None
