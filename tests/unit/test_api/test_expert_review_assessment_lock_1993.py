"""#1993: ``POST /expert-reviews/{review_id}/assessment`` holds a cross-worker
in-flight lock so two concurrent UNCACHED requests for one review build the
LLM assessment ONCE (one bill, one persisted value) instead of twice with the
last write winning.

Hermetic, in-process bare app (pattern of test_expert_review_detail_route.py;
no ``src.api.main`` import, no lifespan, no Redis service): the repo, the
validation-row read, the slow LLM build and the lock's Redis client are all
test doubles. ``require_operator`` is overridden with an explicit operator.
Concurrency is driven on ONE event loop with ``httpx.ASGITransport`` +
``asyncio.gather``; the build stub sleeps in a worker thread (the route runs it
via ``asyncio.to_thread``) so the second request genuinely overlaps the first.

Positive control on the "assessments are equal" assertion: every build returns
its own sequence number captured BEFORE the sleep, so a double build yields two
DIFFERENT payloads (measured on main before the fix: builds=2, both
``cached=False``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

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

ROW: Dict[str, Any] = {
    "review_id": RID,
    "review_type": "dag_approval",
    "dag_version_hash": "h" * 64,
    "brand": "Kisqali",
    "approval_status": "pending",
    "dag_structure_json": json.dumps({"nodes": ["t", "y"], "edges": [["t", "y"]]}),
    "related_validation_ids": [],
}


class _Repo:
    """Row store whose ``update_agent_assessment`` makes the write VISIBLE to the
    next reader (the JSONB-string form the live write path produces)."""

    def __init__(self, row: Dict[str, Any], *, persist: bool = True) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {row["review_id"]: dict(row)}
        self.persist = persist
        self.writes: List[Dict[str, Any]] = []
        self.reads: List[str] = []

    async def get_by_id(self, review_id: str) -> Optional[Dict[str, Any]]:
        self.reads.append(review_id)
        row = self.rows.get(review_id)
        return dict(row) if row is not None else None

    async def update_agent_assessment(self, review_id: str, assessment: Dict[str, Any]) -> bool:
        self.writes.append(dict(assessment))
        if not self.persist:
            return False
        self.rows[review_id]["agent_assessment_json"] = json.dumps(assessment)
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


def _app(
    monkeypatch, repo: _Repo, redis: _FakeRedis, *, build_sleep: float = 0.3
) -> Tuple[FastAPI, Dict[str, int]]:
    async def _repo_factory():
        return repo

    async def _no_validation_rows(ids):
        return []

    counter = {"builds": 0}

    def _slow_build(review: Dict[str, Any], validations: List[Dict[str, Any]]) -> Dict[str, Any]:
        counter["builds"] += 1
        n = counter["builds"]  # captured BEFORE the sleep: a double build is visible
        time.sleep(build_sleep)
        return {"items": [{"id": "q1", "verdict": "supports"}], "is_fallback": False, "n": n}

    async def _redis_factory():
        return redis

    monkeypatch.setattr(route_mod, "_get_expert_review_repo", _repo_factory)
    monkeypatch.setattr(route_mod, "_get_validation_rows", _no_validation_rows)
    monkeypatch.setattr(route_mod, "_build_assessment", _slow_build)
    monkeypatch.setattr(
        route_mod,
        "_ASSESSMENT_LOCK",
        InflightLock("expert_review:assessment:inflight", redis_factory=_redis_factory),
    )
    app = FastAPI()
    app.include_router(route_mod.router, prefix="/api")
    app.dependency_overrides[require_operator] = lambda: OPERATOR
    return app, counter


async def _post(app: FastAPI, n: int = 1, *, force: bool = False) -> List[httpx.Response]:
    params = {"force": "true"} if force else None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        return list(await asyncio.gather(*(client.post(URL, params=params) for _ in range(n))))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_two_concurrent_uncached_requests_build_once(monkeypatch):
    repo, redis = _Repo(ROW), _FakeRedis()
    app, counter = _app(monkeypatch, repo, redis)

    r1, r2 = await _post(app, 2)

    assert (r1.status_code, r2.status_code) == (200, 200), (r1.text, r2.text)
    assert counter["builds"] == 1
    b1, b2 = r1.json(), r2.json()
    assert (
        b1["assessment"]
        == b2["assessment"]
        == {
            "items": [{"id": "q1", "verdict": "supports"}],
            "is_fallback": False,
            "n": 1,
        }
    )
    # exactly one winner (fresh) and one loser (replayed the winner's write)
    assert sorted([b1["cached"], b2["cached"]]) == [False, True]
    assert b1["persisted"] is b2["persisted"] is True
    assert len(repo.writes) == 1
    # the lock is released, not left to expire
    assert redis.live_keys() == []
    assert "set" in redis.ops and "eval" in redis.ops


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lock_is_released_so_a_later_force_builds_again(monkeypatch):
    repo, redis = _Repo(ROW), _FakeRedis()
    app, counter = _app(monkeypatch, repo, redis, build_sleep=0.05)
    await _post(app, 2)
    assert counter["builds"] == 1
    assert redis.live_keys() == []

    (r,) = await _post(app, 1, force=True)

    assert r.status_code == 200, r.text
    assert counter["builds"] == 2
    assert r.json()["cached"] is False and r.json()["persisted"] is True
    assert r.json()["assessment"]["n"] == 2
    assert redis.live_keys() == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_winner_persist_failure_makes_the_loser_build(monkeypatch):
    """The loser waits, re-reads, finds NO stored assessment (the winner's
    write failed) and builds itself: one extra build, never an error."""
    repo, redis = _Repo(ROW, persist=False), _FakeRedis()
    app, counter = _app(monkeypatch, repo, redis)

    r1, r2 = await _post(app, 2)

    assert (r1.status_code, r2.status_code) == (200, 200), (r1.text, r2.text)
    assert counter["builds"] == 2
    assert [r1.json()["cached"], r2.json()["cached"]] == [False, False]
    assert [r1.json()["persisted"], r2.json()["persisted"]] == [False, False]
    assert {r1.json()["assessment"]["n"], r2.json()["assessment"]["n"]} == {1, 2}
    assert redis.live_keys() == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_force_loser_returns_the_winners_fresh_assessment_not_the_stale_one(monkeypatch):
    """Two concurrent ``force=true`` calls on a row that already holds an OLD
    assessment: the loser must replay the winner's NEW value (``cached=True``),
    regardless of ``force`` -- the winner just regenerated it."""
    stale = {"items": [], "is_fallback": True, "n": 0}
    repo, redis = _Repo({**ROW, "agent_assessment_json": json.dumps(stale)}), _FakeRedis()
    app, counter = _app(monkeypatch, repo, redis)

    r1, r2 = await _post(app, 2, force=True)

    assert counter["builds"] == 1
    fresh, replay = sorted((r1.json(), r2.json()), key=lambda b: b["cached"])
    assert fresh["cached"] is False and replay["cached"] is True
    assert fresh["assessment"] == replay["assessment"]
    assert replay["assessment"]["n"] == 1  # the new one, not ``stale``


@pytest.mark.unit
@pytest.mark.asyncio
async def test_force_loser_does_not_replay_a_stale_row_when_the_winner_failed_to_persist(
    monkeypatch,
):
    """Same as above but the winner's write FAILS: the row still holds the OLD
    assessment. A ``force`` loser must not present that as the regenerated
    result -- it builds (the value it re-reads is unchanged from before it waited)."""
    stale = {"items": [], "is_fallback": True, "n": 0}
    repo = _Repo({**ROW, "agent_assessment_json": json.dumps(stale)}, persist=False)
    redis = _FakeRedis()
    app, counter = _app(monkeypatch, repo, redis)

    r1, r2 = await _post(app, 2, force=True)

    assert (r1.status_code, r2.status_code) == (200, 200)
    assert counter["builds"] == 2
    assert [r1.json()["cached"], r2.json()["cached"]] == [False, False]
    assert {r1.json()["assessment"]["n"], r2.json()["assessment"]["n"]} == {1, 2}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redis_failure_falls_back_to_a_process_local_lock(monkeypatch, caplog):
    """Redis ``SET`` raising must never fail the request; within one process
    the fallback ``asyncio.Lock`` still serialises the build. The cost is
    bounded: ONE Redis attempt (the second request is inside the cooldown) and
    ONE warning."""
    repo, redis = _Repo(ROW), _FakeRedis(fail_set=True)
    app, counter = _app(monkeypatch, repo, redis)

    with caplog.at_level(logging.WARNING, logger="src.api.dependencies.inflight_lock"):
        r1, r2 = await _post(app, 2)

    assert (r1.status_code, r2.status_code) == (200, 200), (r1.text, r2.text)
    assert counter["builds"] == 1
    assert sorted([r1.json()["cached"], r2.json()["cached"]]) == [False, True]
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
    app, counter = _app(monkeypatch, repo, redis)

    (r,) = await _post(app, 1)
    assert r.status_code == 200
    assert r.json() == {"review_id": RID, "assessment": stored, "cached": True, "persisted": True}
    assert counter["builds"] == 0
    assert redis.ops == []

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        r404 = await client.post(f"/api/expert-reviews/{RID_UNKNOWN}/assessment")
    assert r404.status_code == 404
    assert redis.ops == []
