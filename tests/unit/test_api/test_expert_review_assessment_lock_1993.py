"""#1993: ``POST /expert-reviews/{review_id}/assessment`` holds a cross-worker
in-flight lock so two concurrent UNCACHED requests for one review build the
LLM assessment ONCE (one bill, one persisted value) instead of twice with the
last write winning.

Hermetic, in-process bare app (pattern of test_expert_review_detail_route.py;
no ``src.api.main`` import, no lifespan, no Redis service): the repo, the
validation-row read, the LLM build and the lock's Redis client are all test
doubles. ``require_operator`` is overridden with an explicit operator.

Interleaving is DETERMINISTIC, not sleep-based (codex MED 4): the build stub
blocks on a ``threading.Event`` gate (bounded at 5 s so a broken test fails
instead of hanging the xdist worker). A test launches the first request,
waits until its build has STARTED (it now holds the lock), launches the
second, waits until it has OBSERVED that request waiting (a GET poll on the
fake Redis, or a second ref on the local lock entry), and only then opens the
gate. Concurrency runs on ONE event loop via ``httpx.ASGITransport``.

Positive control on the "assessments are equal" assertion: every build returns
its own sequence number captured BEFORE the gate, so a double build yields two
DIFFERENT payloads (measured on main before the fix: builds=2, both
``cached=False``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import pytest
from fastapi import FastAPI

import src.api.routes.expert_review as route_mod
from src.api.dependencies.auth import require_operator
from src.api.dependencies.inflight_lock import InflightLock
from src.api.errors import SAFE_503_DETAIL_PREFIX

RID = "1c8f3d6a-5b7e-4c21-9f0a-2d4e6b8a0c13"
RID_UNKNOWN = "9e8d7c6b-5a49-4382-b1c0-d9e8f7a6b5c4"
LOCK_KEY = f"expert_review:assessment:inflight:{RID}"
URL = f"/api/expert-reviews/{RID}/assessment"
OPERATOR = {"id": "op-1", "email": "operator@example.com", "app_metadata": {"role": "admin"}}
GATE_TIMEOUT = 5.0  # bounded: a broken test fails, never hangs the worker
T0 = "2026-09-09T12:00:00+00:00"
IDENTICAL = {"items": [{"id": "q1", "verdict": "insufficient"}], "is_fallback": True}
# A stored payload the route stamped on an earlier build (round-3 MED-B).
STORED_IDENTICAL = {**IDENTICAL, "generation_id": "gen-0", "generated_at": T0}
STAMP_KEYS = {"generation_id", "generated_at"}

ROW: Dict[str, Any] = {
    "review_id": RID,
    "review_type": "dag_approval",
    "dag_version_hash": "h" * 64,
    "brand": "Kisqali",
    "approval_status": "pending",
    "dag_structure_json": json.dumps({"nodes": ["t", "y"], "edges": [["t", "y"]]}),
    "related_validation_ids": [],
    "updated_at": T0,
}


class _Repo:
    """Row store whose ``update_agent_assessment`` makes the write VISIBLE to the
    next reader (the JSONB-string form the live write path produces)."""

    def __init__(self, row: Dict[str, Any], *, persist: bool = True) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {row["review_id"]: dict(row)}
        self.persist = persist
        self.writes: List[Dict[str, Any]] = []
        self.reads = 0
        # When set, the FIRST read returns its snapshot only after this opens:
        # the snapshot is taken now, the caller proceeds later (round-2 HIGH).
        self.first_read_gate: Optional[asyncio.Event] = None
        # When set, every read numbered above it raises (a store outage that
        # starts after N successful reads; round-3 MED-A).
        self.fail_reads_after: Optional[int] = None

    async def get_by_id(self, review_id: str) -> Optional[Dict[str, Any]]:
        self.reads += 1
        if self.fail_reads_after is not None and self.reads > self.fail_reads_after:
            raise RuntimeError("connection refused")
        row = self.rows.get(review_id)
        snapshot = dict(row) if row is not None else None
        if self.reads == 1 and self.first_read_gate is not None:
            await asyncio.wait_for(self.first_read_gate.wait(), timeout=GATE_TIMEOUT)
        return snapshot

    def stored(self) -> Dict[str, Any]:
        """The stored assessment as a dict: a seeded row holds a dict, a
        persisted one the JSON string (both forms the route decodes)."""
        raw = self.rows[RID]["agent_assessment_json"]
        return raw if isinstance(raw, dict) else json.loads(raw)

    async def update_agent_assessment(self, review_id: str, assessment: Dict[str, Any]) -> bool:
        self.writes.append(dict(assessment))
        if not self.persist:
            return False
        self.rows[review_id]["agent_assessment_json"] = json.dumps(assessment)
        return True


def _unstamped(assessment: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in assessment.items() if k not in STAMP_KEYS}


class _FakeRedis:
    """In-memory subset the lock uses: SET NX PX, GET, DELETE, EVAL (compare-and-
    delete release script). PX expiry is honoured on read."""

    def __init__(
        self, *, fail_set: bool = False, factory_down: bool = False, get_delay: float = 0.0
    ) -> None:
        self.store: Dict[str, Tuple[str, Optional[float]]] = {}
        self.ops: List[str] = []
        self.fail_set = fail_set
        self.factory_down = factory_down  # the client getter itself fails (outage)
        self.get_delay = get_delay  # a slow server: each GET takes this long
        self.set_times: List[float] = []
        self.cancelled_gets = 0

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
        if self.fail_set:
            raise ConnectionError("redis down")
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
        key, token = args[0], args[1]
        if self._live(key) == token:
            del self.store[key]
            return 1
        return 0


class _Harness:
    def __init__(self, app: FastAPI, lock: InflightLock, redis: _FakeRedis) -> None:
        self.app, self.lock, self.redis = app, lock, redis
        self.builds = 0
        self.gate = threading.Event()

    def redis_waiting(self) -> bool:
        """A request is polling a key held by ANOTHER worker (foreign key)."""
        return self.redis.ops.count("get") >= 1

    def local_waiting(self) -> bool:
        """The second request is queued on the per-id local mutex -- where every
        same-worker waiter queues, in redis AND local mode (round-4)."""
        entry = self.lock._local.get(RID)
        return entry is not None and entry.refs >= 2

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://testserver"
        )


def _harness(
    monkeypatch, repo: _Repo, redis: _FakeRedis, *, identical: bool = False, **lock_kwargs: Any
) -> _Harness:
    async def _repo_factory():
        return repo

    async def _no_validation_rows(ids):
        return []

    async def _redis_factory():
        if redis.factory_down:
            raise ConnectionError("redis down")
        return redis

    # poll fast so a waiter is observed quickly; the default is pinned in the helper tests
    lock_kwargs.setdefault("poll_seconds", 0.02)
    lock = InflightLock(
        "expert_review:assessment:inflight", redis_factory=_redis_factory, **lock_kwargs
    )
    app = FastAPI()
    app.include_router(route_mod.router, prefix="/api")
    app.dependency_overrides[require_operator] = lambda: OPERATOR
    h = _Harness(app, lock, redis)

    def _gated_build(review: Dict[str, Any], validations: List[Dict[str, Any]]) -> Dict[str, Any]:
        h.builds += 1
        n = h.builds  # captured BEFORE the gate: a double build is visible
        if not h.gate.wait(timeout=GATE_TIMEOUT):
            raise RuntimeError("test gate never opened")
        if identical:
            return dict(IDENTICAL)  # a deterministic regeneration (codex MED 3)
        return {"items": [{"id": "q1", "verdict": "supports"}], "is_fallback": False, "n": n}

    monkeypatch.setattr(route_mod, "_get_expert_review_repo", _repo_factory)
    monkeypatch.setattr(route_mod, "_get_validation_rows", _no_validation_rows)
    monkeypatch.setattr(route_mod, "_build_assessment", _gated_build)
    monkeypatch.setattr(route_mod, "_ASSESSMENT_LOCK", lock)
    return h


async def _until(pred: Callable[[], bool], what: str, timeout: float = GATE_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting until: {what}")
        await asyncio.sleep(0.01)


async def _contended_pair(
    h: _Harness,
    client: httpx.AsyncClient,
    *,
    force: bool = False,
    waiting: Optional[Callable[[], bool]] = None,
    cancel_first: bool = False,
) -> Tuple[Optional[httpx.Response], httpx.Response]:
    """Two requests that PROVABLY contend: the second is launched only after the
    first is inside its build (holding the lock) and the gate opens only after
    the second has been observed waiting."""
    params = {"force": "true"} if force else None
    t1 = asyncio.create_task(client.post(URL, params=params))
    await _until(lambda: h.builds >= 1, "the first request entered the build (lock held)")
    if cancel_first:
        t1.cancel()
        (outcome,) = await asyncio.gather(t1, return_exceptions=True)
        assert isinstance(outcome, asyncio.CancelledError)
    t2 = asyncio.create_task(client.post(URL, params=params))
    await _until(waiting or h.local_waiting, "the second request observed waiting on the lock")
    h.gate.set()
    r2 = await t2
    r1 = None if cancel_first else await t1
    return r1, r2


async def _single(h: _Harness, *, force: bool = False, path: str = URL) -> httpx.Response:
    h.gate.set()
    async with h.client() as client:
        return await client.post(path, params={"force": "true"} if force else None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_two_concurrent_uncached_requests_build_once(monkeypatch, caplog):
    repo, redis = _Repo(ROW), _FakeRedis()
    h = _harness(monkeypatch, repo, redis)

    with caplog.at_level(logging.INFO, logger=route_mod.__name__):
        async with h.client() as client:
            r1, r2 = await _contended_pair(h, client)

    assert (r1.status_code, r2.status_code) == (200, 200), (r1.text, r2.text)
    assert h.builds == 1
    b1, b2 = r1.json(), r2.json()
    assert b1["assessment"] == b2["assessment"]
    assert _unstamped(b1["assessment"]) == {
        "items": [{"id": "q1", "verdict": "supports"}],
        "is_fallback": False,
        "n": 1,
    }
    # the build is stamped before it is persisted, and the replay carries the stamp
    assert STAMP_KEYS <= set(b1["assessment"]) and len(b1["assessment"]["generation_id"]) == 32
    assert repo.stored()["generation_id"] == b1["assessment"]["generation_id"]
    # the first request is the winner (fresh); the second replayed its write
    assert (b1["cached"], b2["cached"]) == (False, True)
    assert b1["persisted"] is b2["persisted"] is True
    assert len(repo.writes) == 1
    # the lock is released, not left to expire
    assert redis.live_keys() == []
    assert "set" in redis.ops and "eval" in redis.ops
    # observability: the build's elapsed seconds and the waiter's outcome are logged
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert sum("assessment built in" in m and "lock mode=redis" in m for m in msgs) == 1
    assert sum("waited=True" in m and "replaying the stored result" in m for m in msgs) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lock_is_released_so_a_later_force_builds_again(monkeypatch):
    repo, redis = _Repo(ROW), _FakeRedis()
    h = _harness(monkeypatch, repo, redis)
    async with h.client() as client:
        await _contended_pair(h, client)
    assert h.builds == 1
    assert redis.live_keys() == []

    r = await _single(h, force=True)

    assert r.status_code == 200, r.text
    assert h.builds == 2
    assert r.json()["cached"] is False and r.json()["persisted"] is True
    assert r.json()["assessment"]["n"] == 2
    assert redis.live_keys() == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_winner_persist_failure_makes_the_loser_build(monkeypatch, caplog):
    """The loser waits, re-reads, finds NO stored assessment (the winner's
    write failed) and builds itself: one extra build, never an error."""
    repo, redis = _Repo(ROW, persist=False), _FakeRedis()
    h = _harness(monkeypatch, repo, redis)

    with caplog.at_level(logging.INFO, logger=route_mod.__name__):
        async with h.client() as client:
            r1, r2 = await _contended_pair(h, client)

    assert (r1.status_code, r2.status_code) == (200, 200), (r1.text, r2.text)
    assert h.builds == 2
    assert [r1.json()["cached"], r2.json()["cached"]] == [False, False]
    assert [r1.json()["persisted"], r2.json()["persisted"]] == [False, False]
    assert (r1.json()["assessment"]["n"], r2.json()["assessment"]["n"]) == (1, 2)
    assert redis.live_keys() == []
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert sum("no new stored result appeared" in m for m in msgs) == 1
    assert sum("assessment built in" in m for m in msgs) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_force_loser_replays_the_winners_fresh_assessment_not_the_stale_one(monkeypatch):
    """Two concurrent ``force=true`` calls on a row that already holds an OLD
    assessment: the loser must replay the winner's NEW value (``cached=True``),
    regardless of ``force`` -- the winner just regenerated it."""
    stale = {"items": [], "is_fallback": True, "n": 0}
    repo, redis = _Repo({**ROW, "agent_assessment_json": json.dumps(stale)}), _FakeRedis()
    h = _harness(monkeypatch, repo, redis)

    async with h.client() as client:
        r1, r2 = await _contended_pair(h, client, force=True)

    assert h.builds == 1
    fresh, replay = r1.json(), r2.json()
    assert fresh["cached"] is False and replay["cached"] is True
    assert fresh["assessment"] == replay["assessment"]
    assert replay["assessment"]["n"] == 1  # the new one, not ``stale``


@pytest.mark.unit
@pytest.mark.asyncio
async def test_force_loser_does_not_replay_a_stale_row_when_the_winner_failed_to_persist(
    monkeypatch,
):
    """Same as above but the winner's write FAILS: the row still holds the OLD
    assessment with the OLD ``updated_at``. A ``force`` loser must not present
    that as the regenerated result -- it builds."""
    stale = {"items": [], "is_fallback": True, "n": 0}
    repo = _Repo({**ROW, "agent_assessment_json": json.dumps(stale)}, persist=False)
    h = _harness(monkeypatch, repo, _FakeRedis())

    async with h.client() as client:
        r1, r2 = await _contended_pair(h, client, force=True)

    assert (r1.status_code, r2.status_code) == (200, 200)
    assert h.builds == 2
    assert [r1.json()["cached"], r2.json()["cached"]] == [False, False]
    assert (r1.json()["assessment"]["n"], r2.json()["assessment"]["n"]) == (1, 2)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_force_loser_replays_a_byte_identical_regeneration_via_generation_id(monkeypatch):
    """Codex MED 3 / round-3 MED-B: a forced regeneration can be byte-identical
    to the stored one (deterministic fallbacks exist). The loser replays ONLY
    because the winner's fresh ``generation_id`` differs from the snapshot's."""
    repo = _Repo({**ROW, "agent_assessment_json": dict(STORED_IDENTICAL)})
    h = _harness(monkeypatch, repo, _FakeRedis(), identical=True)

    async with h.client() as client:
        r1, r2 = await _contended_pair(h, client, force=True)

    assert (r1.status_code, r2.status_code) == (200, 200)
    assert h.builds == 1
    assert len(repo.writes) == 1
    assert (r1.json()["cached"], r2.json()["cached"]) == (False, True)
    fresh, replay = r1.json()["assessment"], r2.json()["assessment"]
    assert _unstamped(fresh) == _unstamped(replay) == IDENTICAL  # byte-identical content
    assert fresh["generation_id"] == replay["generation_id"] != "gen-0"  # the signal
    assert fresh["generated_at"] != T0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_force_loser_builds_when_identical_and_the_winner_failed_to_persist(monkeypatch):
    """The negative twin: the persist failed, so the stored ``generation_id``
    is still the snapshot's -- not a winner's result; the loser builds."""
    repo = _Repo({**ROW, "agent_assessment_json": dict(STORED_IDENTICAL)}, persist=False)
    h = _harness(monkeypatch, repo, _FakeRedis(), identical=True)

    async with h.client() as client:
        r1, r2 = await _contended_pair(h, client, force=True)

    assert (r1.status_code, r2.status_code) == (200, 200)
    assert h.builds == 2
    assert (r1.json()["cached"], r2.json()["cached"]) == (False, False)
    assert repo.stored()["generation_id"] == "gen-0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_force_builds_after_an_unrelated_row_update_landed_before_an_idle_acquire(
    monkeypatch,
):
    """Codex round-3 MED-B: a forced request snapshots the OLD assessment, an
    UNRELATED update (resolve / DAG backfill) moves ``updated_at`` before it
    acquires an idle lock. An ``updated_at`` signal would replay the OLD value
    and skip the requested regeneration; the ``generation_id`` is unchanged, so
    it BUILDS."""
    repo = _Repo({**ROW, "agent_assessment_json": dict(STORED_IDENTICAL)})
    repo.first_read_gate = asyncio.Event()
    h = _harness(monkeypatch, repo, _FakeRedis())
    h.gate.set()

    async with h.client() as client:
        t = asyncio.create_task(client.post(URL, params={"force": "true"}))
        await _until(lambda: repo.reads == 1, "the forced request took its snapshot")
        repo.rows[RID]["approval_status"] = "approved"  # an unrelated write...
        repo.rows[RID]["updated_at"] = "2026-09-09T12:00:59+00:00"  # ...moved updated_at
        repo.first_read_gate.set()  # the request now acquires an IDLE lock
        r = await t

    assert r.status_code == 200, r.text
    assert r.json()["cached"] is False and r.json()["persisted"] is True
    assert h.builds == 1 and len(repo.writes) == 1
    assert repo.stored()["generation_id"] != "gen-0"
    assert _unstamped(r.json()["assessment"])["n"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_legacy_stored_payload_without_generation_id_builds_under_force(monkeypatch):
    """Two legacy payloads without the key compare as NOT new: a forced request
    on a pre-stamp row builds (and the row is stamped from then on)."""
    legacy = {"items": [{"id": "q1", "verdict": "supports"}], "is_fallback": False}
    repo = _Repo({**ROW, "agent_assessment_json": dict(legacy)})
    h = _harness(monkeypatch, repo, _FakeRedis())

    r = await _single(h, force=True)

    assert r.status_code == 200, r.text
    assert r.json()["cached"] is False and h.builds == 1
    assert STAMP_KEYS <= set(repo.stored())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failed_reread_after_acquire_is_a_safe_503_with_no_build(monkeypatch):
    """Codex round-3 MED-A: A persists and releases; B acquires and its re-read
    raises twice (one retry). B must NOT build from its stale snapshot over A's
    fresh result: it answers the SAFE 503 with zero builds and releases the lock."""
    repo, redis = _Repo(ROW), _FakeRedis()
    h = _harness(monkeypatch, repo, redis)

    a = await _single(h)
    assert a.status_code == 200 and a.json()["cached"] is False and h.builds == 1
    assert repo.reads == 2  # A: initial read + re-read under the lock
    stored_by_a = repo.stored()

    repo.fail_reads_after = 3  # B's initial read succeeds; its re-reads fail
    b = await _single(h, force=True)

    assert b.status_code == 503, b.text
    assert b.json()["detail"].startswith(SAFE_503_DETAIL_PREFIX)
    assert repo.reads == 5  # B: initial + 2 re-read attempts
    assert h.builds == 1 and len(repo.writes) == 1  # zero builds by B
    assert repo.stored() == stored_by_a  # A's result intact
    assert redis.live_keys() == []  # the lease was released


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_request_keeps_the_lock_until_its_build_persists(monkeypatch):
    """Codex HIGH 1: cancelling the request (client gone, nginx 504) does not
    stop the builder thread. The build runs in a shielded task, so the lock is
    held until the persist; a second caller waits and replays -- one build."""
    repo, redis = _Repo(ROW), _FakeRedis()
    h = _harness(monkeypatch, repo, redis)

    async with h.client() as client:
        r1, r2 = await _contended_pair(h, client, cancel_first=True)

    assert r1 is None and r2.status_code == 200, r2.text
    await _until(lambda: not route_mod._INFLIGHT_BUILDS, "the orphaned build task finished")
    assert h.builds == 1
    assert len(repo.writes) == 1  # the cancelled request's build still persisted
    assert r2.json()["cached"] is True and r2.json()["persisted"] is True
    assert r2.json()["assessment"]["n"] == 1
    assert redis.live_keys() == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_request_that_snapshotted_before_a_concurrent_build_replays_after_a_late_acquire(
    monkeypatch, caplog
):
    """Codex round-2 HIGH: r2 reads the UNCACHED row before r1 acquires; r1
    builds, persists and releases; r2 then acquires an idle lock
    (``waited=False``). The row is re-read after EVERY acquisition, so r2
    replays instead of building again."""
    repo, redis = _Repo(ROW), _FakeRedis()
    repo.first_read_gate = asyncio.Event()
    h = _harness(monkeypatch, repo, redis)

    with caplog.at_level(logging.INFO, logger=route_mod.__name__):
        async with h.client() as client:
            t2 = asyncio.create_task(client.post(URL))
            await _until(lambda: repo.reads == 1, "the late request took its uncached snapshot")
            t1 = asyncio.create_task(client.post(URL))
            await _until(lambda: h.builds >= 1, "the other request entered the build")
            h.gate.set()
            r1 = await t1  # built, persisted, released
            assert redis.live_keys() == []
            repo.first_read_gate.set()  # the late request now acquires an IDLE lock
            r2 = await t2

    assert (r1.status_code, r2.status_code) == (200, 200), (r1.text, r2.text)
    assert h.builds == 1 and len(repo.writes) == 1
    assert (r1.json()["cached"], r2.json()["cached"]) == (False, True)
    assert r1.json()["assessment"] == r2.json()["assessment"]
    assert r2.json()["assessment"]["n"] == 1 and r2.json()["persisted"] is True
    assert redis.ops.count("get") == 0  # r2 never polled: it did not wait
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert sum("waited=False" in m and "replaying the stored result" in m for m in msgs) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_holder_exceeding_the_wait_bound_is_409_too(monkeypatch):
    """Codex round-2 MED: the guarantee is "Redis being DOWN never fails a
    request", not "the local lock never fails one". A second request that
    exhausts the bound behind a LOCAL holder is 409 (that client has already
    been 504'd by nginx); no build, and the local table is empty afterwards."""
    repo, redis = _Repo(ROW), _FakeRedis(fail_set=True)
    h = _harness(monkeypatch, repo, redis, ttl_ms=100)  # wait bound ~0.12 s

    async with h.client() as client:
        t1 = asyncio.create_task(client.post(URL))
        await _until(lambda: h.builds >= 1, "the first request holds the LOCAL lock")
        r2 = await client.post(URL)  # exhausts the bound behind the local holder
        assert r2.status_code == 409, r2.text
        assert r2.headers.get("retry-after") == "5"
        assert r2.json()["detail"] == route_mod._BUILD_IN_PROGRESS_DETAIL
        assert h.builds == 1
        h.gate.set()
        r1 = await t1

    assert r1.status_code == 200 and r1.json()["cached"] is False
    assert h.builds == 1 and len(repo.writes) == 1
    assert h.lock._local == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wait_exhaustion_is_409_with_retry_after_and_no_build(monkeypatch):
    """Codex HIGH 2: a key that never clears (a foreign holder) exhausts the
    bounded wait; the route answers 409 + Retry-After instead of building
    unlocked, and touches neither the builder nor the foreign key."""
    repo, redis = _Repo(ROW), _FakeRedis()
    redis.preset(LOCK_KEY, "stuck-elsewhere", px=None)
    h = _harness(monkeypatch, repo, redis, ttl_ms=100)

    r = await _single(h)

    assert r.status_code == 409, r.text
    assert r.headers.get("retry-after") == "5"
    assert r.json()["detail"] == route_mod._BUILD_IN_PROGRESS_DETAIL
    assert h.builds == 0 and repo.writes == []
    assert redis._live(LOCK_KEY) == "stuck-elsewhere"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_budget_exhaustion_on_a_slow_redis_is_409_within_the_budget(monkeypatch, caplog):
    """Codex round-5, end to end: behind another worker's holder on a SLOW
    Redis (GET 2 s) with a 0.3 s budget, the waiter answers 409 in ~0.3 s (not
    ~2 s), attempts no SET after the deadline, builds nothing, and the lock
    stays in redis mode for the next request (exhaustion is not an outage)."""
    repo, redis = _Repo(ROW), _FakeRedis(get_delay=2.0)
    redis.preset(LOCK_KEY, "held-by-another-worker", px=None)
    h = _harness(monkeypatch, repo, redis, ttl_ms=300, op_timeout_seconds=1.0)
    loop = asyncio.get_running_loop()

    t0 = loop.time()
    with caplog.at_level(logging.WARNING, logger="src.api.dependencies.inflight_lock"):
        r = await _single(h)
    elapsed = loop.time() - t0

    assert r.status_code == 409, r.text
    assert r.headers.get("retry-after") == "5"
    assert elapsed < 1.0, elapsed
    assert h.builds == 0 and repo.writes == []
    assert redis.cancelled_gets == 1
    assert len(redis.set_times) == 1 and redis.set_times[0] < t0 + h.lock._wait_seconds
    assert h.lock._degraded_until == 0.0
    assert not [r for r in caplog.records if "unavailable" in r.getMessage()]

    del redis.store[LOCK_KEY]  # the other worker released
    r2 = await _single(h)
    assert r2.status_code == 200 and r2.json()["cached"] is False and h.builds == 1
    assert redis.live_keys() == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redis_failure_falls_back_to_a_process_local_lock(monkeypatch, caplog):
    """Redis ``SET`` raising must never fail the request; within one process
    the fallback ``asyncio.Lock`` still serialises the build. The cost is
    bounded: ONE Redis attempt (the second request is inside the cooldown) and
    ONE warning."""
    repo, redis = _Repo(ROW), _FakeRedis(fail_set=True)
    h = _harness(monkeypatch, repo, redis)

    with caplog.at_level(logging.WARNING, logger="src.api.dependencies.inflight_lock"):
        async with h.client() as client:
            r1, r2 = await _contended_pair(h, client)

    assert (r1.status_code, r2.status_code) == (200, 200), (r1.text, r2.text)
    assert h.builds == 1
    assert (r1.json()["cached"], r2.json()["cached"]) == (False, True)
    assert redis.ops.count("set") == 1, redis.ops
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert "fallback" in warnings[0].getMessage()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redis_recovery_mid_build_does_not_bypass_the_local_holder(monkeypatch, caplog):
    """Codex round-4 gated regression: A holds in LOCAL mode (client getter
    failing) mid-build; Redis recovers and the cooldown lifts; B on the same
    worker waits on the local mutex and never acquires Redis while A holds;
    A persists and releases; B replays cached=True -- one build."""
    repo, redis = _Repo(ROW), _FakeRedis(factory_down=True)
    h = _harness(monkeypatch, repo, redis)

    with caplog.at_level(logging.INFO, logger=route_mod.__name__):
        async with h.client() as client:
            t1 = asyncio.create_task(client.post(URL))
            await _until(lambda: h.builds >= 1, "A entered the build holding the LOCAL mutex")
            redis.factory_down = False  # Redis recovers...
            h.lock._degraded_until = 0.0  # ...and the cooldown has lifted
            t2 = asyncio.create_task(client.post(URL))
            await _until(h.local_waiting, "B queued on the local mutex")
            assert redis.ops == []  # B never touched Redis while A holds
            h.gate.set()
            r1, r2 = await t1, await t2

    assert (r1.status_code, r2.status_code) == (200, 200), (r1.text, r2.text)
    assert h.builds == 1 and len(repo.writes) == 1
    assert (r1.json()["cached"], r2.json()["cached"]) == (False, True)
    assert r1.json()["assessment"] == r2.json()["assessment"]
    assert redis.ops == ["set", "eval"] and redis.live_keys() == []  # B's own lease only
    assert h.lock._local == {}
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert sum("assessment built in" in m and "lock mode=local" in m for m in msgs) == 1
    assert sum("lock mode=redis" in m and "replaying the stored result" in m for m in msgs) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cached_fast_path_and_404_take_no_lock(monkeypatch):
    """Unchanged semantics: a stored assessment without ``force`` replays with
    no lock traffic, and an unknown review is 404 with no lock traffic."""
    stored = {"items": [], "is_fallback": False, "n": 7}
    repo, redis = _Repo({**ROW, "agent_assessment_json": stored}), _FakeRedis()
    h = _harness(monkeypatch, repo, redis)

    r = await _single(h)
    assert r.status_code == 200
    assert r.json() == {"review_id": RID, "assessment": stored, "cached": True, "persisted": True}
    assert h.builds == 0
    assert redis.ops == []

    r404 = await _single(h, path=f"/api/expert-reviews/{RID_UNKNOWN}/assessment")
    assert r404.status_code == 404
    assert redis.ops == []
