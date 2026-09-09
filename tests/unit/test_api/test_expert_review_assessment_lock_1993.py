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

RID = "1c8f3d6a-5b7e-4c21-9f0a-2d4e6b8a0c13"
RID_UNKNOWN = "9e8d7c6b-5a49-4382-b1c0-d9e8f7a6b5c4"
LOCK_KEY = f"expert_review:assessment:inflight:{RID}"
URL = f"/api/expert-reviews/{RID}/assessment"
OPERATOR = {"id": "op-1", "email": "operator@example.com", "app_metadata": {"role": "admin"}}
GATE_TIMEOUT = 5.0  # bounded: a broken test fails, never hangs the worker
T0 = "2026-09-09T12:00:00+00:00"
IDENTICAL = {"items": [{"id": "q1", "verdict": "insufficient"}], "is_fallback": True}

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
    next reader (the JSONB-string form the live write path produces) and bumps
    ``updated_at`` like the live ``trg_er_updated_at`` BEFORE UPDATE trigger."""

    def __init__(self, row: Dict[str, Any], *, persist: bool = True) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {row["review_id"]: dict(row)}
        self.persist = persist
        self.writes: List[Dict[str, Any]] = []
        self.version = 0

    async def get_by_id(self, review_id: str) -> Optional[Dict[str, Any]]:
        row = self.rows.get(review_id)
        return dict(row) if row is not None else None

    async def update_agent_assessment(self, review_id: str, assessment: Dict[str, Any]) -> bool:
        self.writes.append(dict(assessment))
        if not self.persist:
            return False
        self.version += 1
        self.rows[review_id]["agent_assessment_json"] = json.dumps(assessment)
        self.rows[review_id]["updated_at"] = f"2026-09-09T12:00:{self.version:02d}+00:00"
        return True


class _FakeRedis:
    """In-memory subset the lock uses: SET NX PX, GET, DELETE, EVAL (compare-and-
    delete release script). PX expiry is honoured on read."""

    def __init__(self, *, fail_set: bool = False) -> None:
        self.store: Dict[str, Tuple[str, Optional[float]]] = {}
        self.ops: List[str] = []
        self.fail_set = fail_set

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
        if self.fail_set:
            raise ConnectionError("redis down")
        if nx and self._live(name) is not None:
            return None
        self.store[name] = (value, time.monotonic() + px / 1000.0 if px else None)
        return True

    async def get(self, name: str) -> Optional[str]:
        self.ops.append("get")
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
        """The second request is polling the held key."""
        return self.redis.ops.count("get") >= 1

    def local_waiting(self) -> bool:
        """The second request is queued on the process-local lock entry."""
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
    await _until(waiting or h.redis_waiting, "the second request observed waiting on the lock")
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
    assert (
        b1["assessment"]
        == b2["assessment"]
        == {"items": [{"id": "q1", "verdict": "supports"}], "is_fallback": False, "n": 1}
    )
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
    assert sum("replaying the winner's stored result" in m for m in msgs) == 1


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
async def test_force_loser_replays_a_byte_identical_regeneration_via_updated_at(monkeypatch):
    """Codex MED 3: a forced regeneration can be byte-identical to the stored
    one (deterministic fallbacks exist). Payload equality alone would make the
    loser rebuild; the trigger-maintained ``updated_at`` moved, so it replays."""
    repo = _Repo({**ROW, "agent_assessment_json": dict(IDENTICAL)})
    h = _harness(monkeypatch, repo, _FakeRedis(), identical=True)

    async with h.client() as client:
        r1, r2 = await _contended_pair(h, client, force=True)

    assert (r1.status_code, r2.status_code) == (200, 200)
    assert h.builds == 1
    assert len(repo.writes) == 1
    assert (r1.json()["cached"], r2.json()["cached"]) == (False, True)
    assert r1.json()["assessment"] == r2.json()["assessment"] == IDENTICAL
    assert repo.rows[RID]["updated_at"] != T0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_force_loser_builds_when_identical_and_the_winner_failed_to_persist(monkeypatch):
    """The negative twin: identical payload AND unchanged ``updated_at`` (the
    persist failed) is not a winner's result -- the loser builds."""
    repo = _Repo({**ROW, "agent_assessment_json": dict(IDENTICAL)}, persist=False)
    h = _harness(monkeypatch, repo, _FakeRedis(), identical=True)

    async with h.client() as client:
        r1, r2 = await _contended_pair(h, client, force=True)

    assert (r1.status_code, r2.status_code) == (200, 200)
    assert h.builds == 2
    assert (r1.json()["cached"], r2.json()["cached"]) == (False, False)
    assert repo.rows[RID]["updated_at"] == T0


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
async def test_redis_failure_falls_back_to_a_process_local_lock(monkeypatch, caplog):
    """Redis ``SET`` raising must never fail the request; within one process
    the fallback ``asyncio.Lock`` still serialises the build. The cost is
    bounded: ONE Redis attempt (the second request is inside the cooldown) and
    ONE warning."""
    repo, redis = _Repo(ROW), _FakeRedis(fail_set=True)
    h = _harness(monkeypatch, repo, redis)

    with caplog.at_level(logging.WARNING, logger="src.api.dependencies.inflight_lock"):
        async with h.client() as client:
            r1, r2 = await _contended_pair(h, client, waiting=h.local_waiting)

    assert (r1.status_code, r2.status_code) == (200, 200), (r1.text, r2.text)
    assert h.builds == 1
    assert (r1.json()["cached"], r2.json()["cached"]) == (False, True)
    assert redis.ops.count("set") == 1, redis.ops
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert "fallback" in warnings[0].getMessage()


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
