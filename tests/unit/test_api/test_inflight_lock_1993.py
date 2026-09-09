"""#1993: ``InflightLock`` -- the cross-worker in-flight lock behind
``POST /expert-reviews/{review_id}/assessment``.

Redis path: ``SET key <uuid-token> NX PX <ttl>``; a loser polls until the key is
gone (holder released or TTL expired), bounded by the TTL; release is a
compare-and-delete Lua script so a lease that outlived its TTL can never delete
the NEXT holder's key. Degraded path: any Redis error / hang falls back to a
bounded, process-local ``asyncio.Lock`` per id, warns ONCE, and stops probing
Redis for a cooldown so a dead Redis costs one bounded attempt, not a retry
storm per poll. The default client getter reads the app's already-initialised
client and never triggers ``init_redis()``'s 5-attempt backoff on a request.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

import src.api.dependencies.redis_client as redis_client_mod
from src.api.dependencies import inflight_lock as mod
from src.api.dependencies.inflight_lock import InflightLock

PREFIX = "test:inflight"


class _FakeRedis:
    def __init__(
        self,
        *,
        fail_set: bool = False,
        hang: bool = False,
        fail_set_with: Optional[BaseException] = None,
        fail_eval: bool = False,
        get_delay: float = 0.0,
        eval_gate: Optional[asyncio.Event] = None,
    ) -> None:
        self.store: Dict[str, Tuple[str, Optional[float]]] = {}
        self.ops: List[str] = []
        self.fail_set = fail_set
        self.hang = hang
        self.fail_set_with = fail_set_with
        self.fail_eval = fail_eval
        self.get_delay = get_delay  # a slow server: each GET takes this long
        self.set_times: List[float] = []  # loop time of every SET attempt
        self.cancelled_gets = 0  # GETs cut short by the caller's timeout
        # When set, EVAL blocks on this gate; a cancellation while blocked means
        # the command never reached the server (worst case: no delete happens).
        self.eval_gate = eval_gate

    def _live(self, name: str) -> Optional[str]:
        item = self.store.get(name)
        if item is None:
            return None
        value, expires = item
        if expires is not None and time.monotonic() >= expires:
            del self.store[name]
            return None
        return value

    def live_keys(self) -> List[str]:
        return [k for k in list(self.store) if self._live(k) is not None]

    def preset(self, name: str, value: str, px: Optional[int] = None) -> None:
        self.store[name] = (value, time.monotonic() + px / 1000.0 if px else None)

    async def set(self, name: str, value: str, nx: bool = False, px: Optional[int] = None):
        self.ops.append("set")
        self.set_times.append(asyncio.get_running_loop().time())
        if self.hang:
            await asyncio.Event().wait()  # never returns
        if self.fail_set:
            raise ConnectionError("redis down")
        if self.fail_set_with is not None:
            raise self.fail_set_with
        if nx and self._live(name) is not None:
            return None
        self.store[name] = (value, time.monotonic() + px / 1000.0 if px else None)
        return True

    async def get(self, name: str) -> Optional[str]:
        self.ops.append("get")
        if self.get_delay:
            try:
                await asyncio.sleep(self.get_delay)
            except asyncio.CancelledError:
                self.cancelled_gets += 1
                raise
        return self._live(name)

    async def delete(self, name: str) -> int:
        self.ops.append("delete")
        return 1 if self.store.pop(name, None) is not None else 0

    async def eval(self, script: str, numkeys: int, *args: Any) -> int:
        self.ops.append("eval")
        if self.eval_gate is not None:
            await self.eval_gate.wait()
        if self.fail_eval:
            raise ConnectionError("redis down at release")
        key, token = args[0], args[1]
        if self._live(key) == token:
            del self.store[key]
            return 1
        return 0


def _lock(redis: Optional[_FakeRedis], **kw: Any) -> InflightLock:
    async def _factory():
        if redis is None:
            raise ConnectionError("redis down")
        return redis

    return InflightLock(PREFIX, redis_factory=_factory, **kw)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_winner_sets_nx_px_and_releases_its_own_token():
    redis = _FakeRedis()
    lock = _lock(redis, ttl_ms=5_000)
    async with lock.hold("r1") as lease:
        assert lease.mode == "redis" and lease.waited is False
        assert lease.key == f"{PREFIX}:r1"
        value, expires = redis.store[f"{PREFIX}:r1"]
        assert value == lease.token and len(value) >= 32
        assert expires is not None and 0 < expires - time.monotonic() <= 5.0
    assert redis.live_keys() == []
    assert redis.ops == ["set", "eval"]  # compare-and-delete, not a blind DEL


@pytest.mark.unit
@pytest.mark.asyncio
async def test_loser_waits_for_the_holder_then_acquires():
    redis = _FakeRedis()
    lock = _lock(redis, poll_seconds=0.02)
    order: List[str] = []

    async def holder():
        async with lock.hold("r1") as lease:
            order.append(f"A-in:{lease.waited}")
            await asyncio.sleep(0.25)
            order.append("A-out")

    async def follower():
        await asyncio.sleep(0.05)  # arrive while A holds the key
        async with lock.hold("r1") as lease:
            order.append(f"B-in:{lease.waited}")
        order.append("B-out")

    await asyncio.gather(holder(), follower())
    assert order == ["A-in:False", "A-out", "B-in:True", "B-out"]
    assert redis.live_keys() == []
    # A same-worker waiter queues on the local mutex and never polls Redis
    # (round-4); polling is only for a key held by ANOTHER worker, covered by
    # the dead-holder / never-clearing foreign-key tests below.
    assert redis.ops.count("get") == 0
    assert redis.ops.count("set") == 2  # one SET per holder, none while waiting


@pytest.mark.unit
@pytest.mark.asyncio
async def test_different_ids_do_not_serialise():
    redis = _FakeRedis()
    lock = _lock(redis)
    async with lock.hold("r1") as a, lock.hold("r2") as b:
        assert a.waited is False and b.waited is False
        assert sorted(redis.live_keys()) == [f"{PREFIX}:r1", f"{PREFIX}:r2"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_release_is_compare_and_delete_never_the_next_holders_key():
    """A lease that outlived its TTL: the key now belongs to another holder;
    our release must leave it alone."""
    redis = _FakeRedis()
    lock = _lock(redis)
    async with lock.hold("r1"):
        redis.preset(f"{PREFIX}:r1", "someone-else", px=60_000)
    assert redis._live(f"{PREFIX}:r1") == "someone-else"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_loser_acquires_after_a_dead_holders_key_expires():
    redis = _FakeRedis()
    redis.preset(f"{PREFIX}:r1", "dead-worker", px=100)
    lock = _lock(redis, ttl_ms=5_000, poll_seconds=0.02)
    t0 = time.monotonic()
    async with lock.hold("r1") as lease:
        assert lease.mode == "redis" and lease.waited is True
        assert redis.store[f"{PREFIX}:r1"][0] == lease.token
    assert time.monotonic() - t0 < 3.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wait_is_bounded_by_the_ttl_then_reports_exhaustion(caplog):
    """A key that never clears (a never-expiring foreign holder): after ~TTL
    ``hold`` yields ``mode == "none"`` -- NO lock held, no local fallback taken
    -- and the caller decides (the route answers 409; codex HIGH 2)."""
    redis = _FakeRedis()
    redis.preset(f"{PREFIX}:r1", "stuck", px=None)
    lock = _lock(redis, ttl_ms=100, poll_seconds=0.02)
    t0 = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        async with lock.hold("r1") as lease:
            assert lease.mode == "none" and lease.waited is True
    assert time.monotonic() - t0 < 5.0
    assert redis._live(f"{PREFIX}:r1") == "stuck"  # not ours; untouched
    assert lock._local == {}  # the stage-1 mutex is dropped with the lease
    assert any("exhausted" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_budget_exhaustion_trims_the_last_command_and_attempts_no_late_set(caplog):
    """Codex round-5: a waiter behind another worker's holder, on a SLOW Redis
    (each GET takes 2 s), with a 0.3 s budget. Untrimmed, the GET alone would
    overrun the budget by ~1.7 s and a SET could follow; trimmed, the GET is
    cut at the deadline, no SET is attempted afterwards, the outcome is
    ``mode == "none"`` and it is NOT a degrade event (no cooldown, no outage
    warning). Wall-clock bound 1.0 s: far below the untrimmed 2 s, far above
    the trimmed ~0.32 s, so CI load cannot flip it."""
    redis = _FakeRedis(get_delay=2.0)
    redis.preset(f"{PREFIX}:r1", "held-by-another-worker", px=None)
    lock = _lock(redis, ttl_ms=300, poll_seconds=0.02, op_timeout_seconds=1.0)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        async with lock.hold("r1") as lease:
            assert lease.mode == "none" and lease.waited is True
    elapsed = loop.time() - t0
    assert elapsed < 1.0, elapsed
    assert redis.cancelled_gets == 1  # the last GET was trimmed to the budget
    assert redis.set_times == [pytest.approx(t0, abs=0.1)]  # one SET, none after the deadline
    assert all(t < t0 + lock._wait_seconds for t in redis.set_times)
    assert lock._degraded_until == 0.0  # exhaustion never degrades
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(msgs) == 1 and "budget exhausted" in msgs[0] and "unavailable" not in msgs[0]
    # Redis is still used by the next request (no cooldown was entered)
    redis.store.clear()
    async with lock.hold("r1") as lease:
        assert lease.mode == "redis"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redis_failure_degrades_to_local_lock_with_one_warning_and_a_cooldown(caplog):
    calls = {"factory": 0}

    async def _factory():
        calls["factory"] += 1
        raise ConnectionError("redis down")

    lock = InflightLock(PREFIX, redis_factory=_factory, degrade_cooldown_seconds=30.0)
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        for _ in range(3):
            async with lock.hold("r1") as lease:
                assert lease.mode == "local"
    assert calls["factory"] == 1  # the cooldown stops the per-request probing
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "fallback" in warnings[0].getMessage()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cooldown_expiry_probes_redis_again():
    redis = _FakeRedis(fail_set=True)
    lock = _lock(redis, degrade_cooldown_seconds=0.05)
    async with lock.hold("r1") as lease:
        assert lease.mode == "local"
    await asyncio.sleep(0.08)
    redis.fail_set = False
    async with lock.hold("r1") as lease:
        assert lease.mode == "redis"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_failure_mid_request_degrades_to_local():
    redis = _FakeRedis(fail_set=True)
    lock = _lock(redis)
    async with lock.hold("r1") as lease:
        assert lease.mode == "local"
    assert redis.ops == ["set"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_release_failure_warns_and_does_not_enter_the_cooldown(caplog):
    """A lost DEL leaks one key until its TTL; it must not send the whole
    worker to the local lock for the cooldown (quality review item 3)."""
    redis = _FakeRedis(fail_eval=True)
    lock = _lock(redis)
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        async with lock.hold("r1") as lease:
            assert lease.mode == "redis"
    assert redis.live_keys() == [f"{PREFIX}:r1"]  # left to the TTL
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "release failed" in warnings[0].getMessage()

    redis.fail_eval = False
    async with lock.hold("r2") as lease:
        assert lease.mode == "redis"  # a cooldown would have made this "local"
    assert redis.ops.count("set") == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancellation_during_the_release_eval_never_strands_the_local_mutex():
    """Codex round-6: the release EVAL is awaited inside ``finally``; a
    cancellation landing there must still release/unref the per-id local
    mutex (a stranded entry would 409 every later same-worker request for the
    id until the worker recycled). The cancellation still propagates. The
    Redis key, whose EVAL never landed, clears by the TTL path and the next
    same-worker request acquires in redis mode without exhaustion."""
    redis = _FakeRedis(eval_gate=asyncio.Event())  # EVAL blocks; never opened
    lock = _lock(redis, ttl_ms=200, poll_seconds=0.02)
    body_done = asyncio.Event()

    async def holder():
        async with lock.hold("r1") as lease:
            assert lease.mode == "redis"
            body_done.set()
        # not reached: the release EVAL blocks and the task is cancelled there

    task = asyncio.create_task(holder())
    await asyncio.wait_for(body_done.wait(), timeout=5.0)
    deadline = asyncio.get_running_loop().time() + 5.0
    while redis.ops.count("eval") < 1:  # the release is in flight, blocked on the gate
        assert asyncio.get_running_loop().time() < deadline, "release EVAL never started"
        await asyncio.sleep(0.005)
    task.cancel()
    (outcome,) = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(outcome, asyncio.CancelledError)  # propagation preserved

    assert lock._local == {}  # the fix: released and unref'd despite the cancel
    assert redis.live_keys() == [f"{PREFIX}:r1"]  # the EVAL never landed: TTL path

    redis.eval_gate = None
    async with lock.hold("r1") as lease:  # same worker, next request
        assert lease.mode == "redis" and lease.waited is True  # waited out the TTL, no 409
    assert redis.live_keys() == [] and lock._local == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unexpected_error_still_degrades_but_is_logged_as_a_bug_not_an_outage(caplog):
    """Only transport errors are outages (warning once). Anything else still
    degrades (never fail the request) but is an ERROR with a traceback."""
    redis = _FakeRedis(fail_set_with=TypeError("bad call"))
    lock = _lock(redis)
    with caplog.at_level(logging.DEBUG, logger=mod.__name__):
        async with lock.hold("r1") as lease:
            assert lease.mode == "local"
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "unexpected TypeError" in errors[0].getMessage()
    assert errors[0].exc_info is not None
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redis_recovery_does_not_bypass_an_active_local_holder():
    """Codex round-4: A acquired in LOCAL mode during an outage and is still
    holding when Redis recovers and the cooldown lifts. B on the same worker
    must queue on the local mutex (never SET while A holds), then acquire in
    redis mode after A releases."""
    redis = _FakeRedis()
    state = {"healthy": False}

    async def _factory():
        if not state["healthy"]:
            raise ConnectionError("redis down")
        return redis

    lock = InflightLock(PREFIX, redis_factory=_factory, poll_seconds=0.02)
    order: List[str] = []
    a_holding = asyncio.Event()
    b_waiting = asyncio.Event()

    async def holder_a():
        async with lock.hold("r1") as lease:
            order.append(f"A-in:{lease.mode}")
            a_holding.set()
            await asyncio.wait_for(b_waiting.wait(), timeout=5.0)
            assert redis.ops == []  # B has not touched Redis while A holds
            order.append("A-out")

    async def follower_b():
        await asyncio.wait_for(a_holding.wait(), timeout=5.0)
        state["healthy"] = True  # Redis recovers...
        lock._degraded_until = 0.0  # ...and the cooldown has lifted
        b_waiting.set()
        async with lock.hold("r1") as lease:
            order.append(f"B-in:{lease.mode}:waited={lease.waited}")

    # b_waiting is set just before B blocks on the mutex; A then verifies no
    # Redis traffic happened and releases, which is the only way B proceeds.
    await asyncio.gather(holder_a(), follower_b())
    assert order == ["A-in:local", "A-out", "B-in:redis:waited=True"]
    assert redis.ops == ["set", "eval"]  # B's own acquire/release only
    assert redis.live_keys() == [] and lock._local == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hanging_redis_op_is_bounded_by_the_op_timeout():
    redis = _FakeRedis(hang=True)
    lock = _lock(redis, op_timeout_seconds=0.05)
    t0 = time.monotonic()
    async with lock.hold("r1") as lease:
        assert lease.mode == "local"
    assert time.monotonic() - t0 < 3.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_fallback_serialises_same_id_in_process():
    lock = _lock(None, poll_seconds=0.01)
    order: List[str] = []

    async def holder():
        async with lock.hold("r1") as lease:
            order.append(f"A-in:{lease.waited}:{lease.mode}")
            await asyncio.sleep(0.15)
            order.append("A-out")

    async def follower():
        await asyncio.sleep(0.03)
        async with lock.hold("r1") as lease:
            order.append(f"B-in:{lease.waited}:{lease.mode}")

    await asyncio.gather(holder(), follower())
    assert order == ["A-in:False:local", "A-out", "B-in:True:local"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_table_holds_only_in_flight_ids_and_never_drops_a_held_one():
    """Bounded by concurrency: an id's entry lives while it has a holder or a
    waiter and is dropped when the last one leaves; a held entry is never
    evicted (that would hand the next request a fresh, unlocked lock)."""
    lock = _lock(None)
    async with lock.hold("held"):
        for i in range(10):
            async with lock.hold(f"k{i}"):
                assert set(lock._local) == {"held", f"k{i}"}
            assert set(lock._local) == {"held"}
    assert lock._local == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_default_factory_uses_the_initialised_client_and_never_calls_init_redis(
    monkeypatch,
):
    async def _init_redis_must_not_run():
        raise AssertionError("init_redis() must not be triggered from a request")

    monkeypatch.setattr(redis_client_mod, "init_redis", _init_redis_must_not_run)

    # Not initialised (startup degraded mode): local fallback, no retry storm.
    monkeypatch.setattr(redis_client_mod, "_redis_client", None)
    lock = InflightLock(PREFIX)
    async with lock.hold("r1") as lease:
        assert lease.mode == "local"

    # Initialised: the same client object is used.
    redis = _FakeRedis()
    monkeypatch.setattr(redis_client_mod, "_redis_client", redis)
    lock = InflightLock(PREFIX)
    async with lock.hold("r1") as lease:
        assert lease.mode == "redis"
        assert redis.live_keys() == [f"{PREFIX}:r1"]
    assert redis.live_keys() == []


@pytest.mark.unit
def test_defaults_match_the_gunicorn_request_timeout():
    """TTL ~ nginx ``proxy_read_timeout 120s`` for /api/, the longest a caller
    waits on a build (gunicorn --timeout does not cap a UvicornWorker build)."""
    assert mod.DEFAULT_TTL_MS == 120_000  # nginx proxy_read_timeout 120s for /api/
    assert mod.DEFAULT_POLL_SECONDS == pytest.approx(0.2)
    assert mod.DEFAULT_OP_TIMEOUT_SECONDS <= 1.0
    lock = InflightLock("expert_review:assessment:inflight")
    assert lock.key("abc") == "expert_review:assessment:inflight:abc"
