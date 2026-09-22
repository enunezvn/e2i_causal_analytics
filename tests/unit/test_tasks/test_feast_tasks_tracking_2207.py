"""#2207: the three Feast beat tasks record what they did into the Feast tracking tables.

Census finding (2026-09-22): ``ml_feast_feature_views``, ``ml_feast_materialization_jobs``
and ``ml_feast_feature_freshness`` had writers (``src/repositories/feast_tracking.py``,
b6ca3f308) that no live producer ever called — 0 rows each while the beat tasks in
``src/tasks/feast_tasks.py`` ran every 6 h / 4 h on worker_medium. Owner decision (a):
wire the repositories from those tasks.

What the tables must say after this change, measured against the LIVE outcome of
those tasks (worker logs 2026-09-22 10:01 UTC): the app/worker image cannot
``import feast`` (#307), so every scheduled materialize returns
``{"status": "failed", "error": "Failed to initialize Feast client"}`` — that outcome
is recorded as a ``failed`` job row per targeted view, never dropped. A freshness run
that could not verify a view records that view as ``unknown`` (#556: unverifiable is
not fresh). The recording is best-effort: a broken tracking client must never fail the
parent beat task.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.feature_store.feast_views import FEAST_FEATURE_VIEW_SOURCE_TABLES
from src.tasks import feast_tasks
from src.workers.celery_app import celery_app

REPO = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# In-memory fake of the SYNC supabase client the feast_tracking repositories drive
# (``client.table(t).insert(row).execute()`` — no await). Stores rows per table so a
# test asserts on what LANDED, not on call shapes.
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, store: Dict[str, List[Dict[str, Any]]], table: str):
        self._store = store
        self._table = table
        self._op = "select"
        self._payload: Any = None
        self._filters: List[tuple] = []

    # builders -------------------------------------------------------------
    def select(self, *_a, **_k):
        self._op = "select"
        return self

    def insert(self, data):
        self._op = "insert"
        self._payload = data
        return self

    def update(self, data):
        self._op = "update"
        self._payload = data
        return self

    def eq(self, col, val):
        self._filters.append(("eq", col, val))
        return self

    def is_(self, col, val):
        self._filters.append(("is", col, val))
        return self

    def limit(self, *_a):
        return self

    def order(self, *_a, **_k):
        return self

    # execution ------------------------------------------------------------
    def _match(self, row: Dict[str, Any]) -> bool:
        for kind, col, val in self._filters:
            if kind == "eq" and str(row.get(col)) != str(val):
                return False
            if kind == "is" and val == "null" and row.get(col) is not None:
                return False
        return True

    def execute(self):
        rows = self._store.setdefault(self._table, [])
        if self._op == "insert":
            payload = self._payload if isinstance(self._payload, list) else [self._payload]
            rows.extend(dict(p) for p in payload)
            return SimpleNamespace(data=[dict(p) for p in payload])
        if self._op == "update":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                r.update(self._payload)
            return SimpleNamespace(data=[dict(r) for r in hit])
        return SimpleNamespace(data=[dict(r) for r in rows if self._match(r)])


class FakeSupabase:
    def __init__(self):
        self.store: Dict[str, List[Dict[str, Any]]] = {}

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self.store, name)


class _ExplodingSupabase:
    def table(self, name: str):
        raise RuntimeError("tracking backend down")


class _FakeJob:
    """Stands in for ``scripts.feast_materialize.MaterializationJob``."""

    outcomes: Dict[str, Any] = {}

    def __init__(self, *_a, **_k):
        pass

    async def run_incremental_materialization(self, **_k):
        return dict(self.outcomes["incremental"])

    async def run_full_materialization(self, **_k):
        return dict(self.outcomes["full"])

    async def check_feature_freshness(self, **_k):
        return dict(self.outcomes["freshness"])

    async def close(self):
        pass


FAILED_INIT = {"status": "failed", "error": "Failed to initialize Feast client"}


@pytest.fixture()
def fake_db():
    db = FakeSupabase()
    with patch.object(feast_tasks, "_tracking_client", return_value=db):
        yield db


@pytest.fixture()
def fake_job():
    with patch("scripts.feast_materialize.MaterializationJob", _FakeJob):
        yield _FakeJob


# ---------------------------------------------------------------------------
# materialization jobs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_failed_incremental_materialize_lands_one_failed_job_row_per_view(fake_db, fake_job):
    """The live outcome on the worker image — recorded, not dropped."""
    fake_job.outcomes = {"incremental": FAILED_INIT, "full": FAILED_INIT}

    # Amended 2026-09-22 (#2207 follow-up): a failed run is RED in Celery (fail loud) and an
    # init failure is not doubled by the auto-recovery branch — see
    # test_feast_tasks_fail_loud_2207.py. The rows still land first.
    with pytest.raises(RuntimeError, match="Failed to initialize Feast client"):
        feast_tasks.materialize_incremental_features()

    jobs = fake_db.store.get("ml_feast_materialization_jobs", [])
    incremental = [j for j in jobs if j["job_type"] == "incremental"]
    recovery = [j for j in jobs if j["job_type"] == "full"]
    assert {j["feature_view_name"] for j in incremental} == set(FEAST_FEATURE_VIEW_SOURCE_TABLES), (
        "one row per real Feast feature view when the run targeted all views"
    )
    assert all(j["status"] == "failed" for j in incremental)
    assert all(j["error_message"] == "Failed to initialize Feast client" for j in incremental)
    assert recovery == []  # no recovery attempt for an init failure
    # every job row links to a registry row for its view
    views = {v["name"]: v for v in fake_db.store.get("ml_feast_feature_views", [])}
    assert set(views) == set(FEAST_FEATURE_VIEW_SOURCE_TABLES)
    for j in incremental:
        assert j["feature_view_id"] == views[j["feature_view_name"]]["id"]
        assert j["start_time"] and j["end_time"]


@pytest.mark.unit
def test_successful_full_materialize_records_success_with_window_and_duration(fake_db, fake_job):
    fake_job.outcomes = {
        "full": {
            "status": "completed",
            "feature_views": ["hcp_profile_features"],
            "rows_materialized": 160,
            "duration_seconds": 3.5,
            "mode": "full",
        }
    }
    start = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()

    feast_tasks.materialize_features(start_date=start, feature_views=["hcp_profile_features"])

    jobs = fake_db.store["ml_feast_materialization_jobs"]
    assert len(jobs) == 1
    (job,) = jobs
    assert job["feature_view_name"] == "hcp_profile_features"
    assert job["job_type"] == "full"
    assert job["status"] == "success"
    assert job["rows_materialized"] == 160
    assert job["duration_seconds"] == 3.5
    assert job["start_time"].startswith(start[:19])
    # registry row is created once and reused
    assert len(fake_db.store["ml_feast_feature_views"]) == 1
    assert fake_db.store["ml_feast_feature_views"][0]["source_name"] == "hcp_profiles"


@pytest.mark.unit
def test_registry_row_is_reused_across_runs(fake_db, fake_job):
    fake_job.outcomes = {"full": {"status": "completed", "feature_views": ["triggers_x"]}}
    feast_tasks.materialize_features(feature_views=["trigger_response_features"])
    feast_tasks.materialize_features(feature_views=["trigger_response_features"])
    assert len(fake_db.store["ml_feast_feature_views"]) == 1
    assert len(fake_db.store["ml_feast_materialization_jobs"]) == 2


@pytest.mark.unit
def test_dry_run_records_nothing(fake_db, fake_job):
    fake_job.outcomes = {"full": {"status": "validated", "feature_views": []}}
    feast_tasks.materialize_features(dry_run=True)
    assert "ml_feast_materialization_jobs" not in fake_db.store


# ---------------------------------------------------------------------------
# freshness
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_completed_freshness_check_records_fresh_stale_and_unknown(fake_db, fake_job):
    now = datetime.now(timezone.utc)
    fake_job.outcomes = {
        "freshness": {
            "status": "completed",
            "fresh": False,
            "fresh_features": [
                {
                    "feature_view": "hcp_profile_features",
                    "last_updated": (now - timedelta(hours=2)).isoformat(),
                    "age_hours": 2.0,
                }
            ],
            "stale_features": [
                {
                    "feature_view": "patient_journey_features",
                    "last_updated": (now - timedelta(hours=99)).isoformat(),
                    "age_hours": 99.0,
                }
            ],
            "errors": [
                {"feature_view": "hcp_engagement_features", "error": "No statistics available"}
            ],
            "max_staleness_hours": 24.0,
        }
    }

    feast_tasks.check_feature_freshness(max_staleness_hours=24.0, alert_on_stale=False)

    rows = {r["feature_view_name"]: r for r in fake_db.store["ml_feast_feature_freshness"]}
    assert set(rows) == {
        "hcp_profile_features",
        "patient_journey_features",
        "hcp_engagement_features",
    }
    assert rows["hcp_profile_features"]["freshness_status"] == "fresh"
    assert rows["hcp_profile_features"]["staleness_seconds"] == 7200
    assert rows["patient_journey_features"]["freshness_status"] == "stale"
    assert rows["patient_journey_features"]["staleness_seconds"] == 99 * 3600
    assert rows["hcp_engagement_features"]["freshness_status"] == "unknown"
    assert "staleness_seconds" not in rows["hcp_engagement_features"]
    for r in rows.values():
        assert r["staleness_threshold_seconds"] == 24 * 3600


@pytest.mark.unit
def test_freshness_check_that_cannot_run_records_every_targeted_view_as_unknown(fake_db, fake_job):
    """The live outcome on the worker image: initialize() fails before any view is
    probed. #556 says unverifiable is not fresh — the table must say so too."""
    fake_job.outcomes = {"freshness": FAILED_INIT}

    # Amended 2026-09-22 (#2207 follow-up): the beat is RED when it probed nothing; the
    # unknown rows land before it raises.
    with pytest.raises(RuntimeError, match="Failed to initialize Feast client"):
        feast_tasks.check_feature_freshness(feature_views=None, alert_on_stale=False)

    rows = fake_db.store["ml_feast_feature_freshness"]
    assert {r["feature_view_name"] for r in rows} == set(FEAST_FEATURE_VIEW_SOURCE_TABLES)
    assert all(r["freshness_status"] == "unknown" for r in rows)


# ---------------------------------------------------------------------------
# the recording never fails the parent task
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tracking_backend_failure_does_not_fail_the_beat_task(fake_job, caplog):
    # Amended 2026-09-22 (#2207 follow-up): a FAILED run now raises on its own merits
    # (fail loud), so the "never fail the parent" contract is asserted on runs that
    # succeeded: a broken tracking backend must not turn them into failed tasks.
    fake_job.outcomes = {
        "incremental": {"status": "completed", "feature_views": ["hcp_profile_features"]},
        "freshness": {
            "status": "completed",
            "fresh": True,
            "fresh_features": [],
            "stale_features": [],
            "errors": [],
        },
    }
    with patch.object(feast_tasks, "_tracking_client", return_value=_ExplodingSupabase()):
        r1 = feast_tasks.materialize_incremental_features()
        r2 = feast_tasks.check_feature_freshness(alert_on_stale=False)
    assert r1["status"] == "completed" and r2["status"] == "completed"


@pytest.mark.unit
def test_unconfigured_tracking_client_does_not_fail_the_beat_task(fake_job):
    # Amended 2026-09-22 (#2207 follow-up): asserted on a successful run (a failed run
    # raises on its own merits now).
    fake_job.outcomes = {"incremental": {"status": "completed", "feature_views": ["x"]}}
    with patch.object(
        feast_tasks, "_tracking_client", side_effect=RuntimeError("SUPABASE_URL unset")
    ):
        assert feast_tasks.materialize_incremental_features()["status"] == "completed"


# ---------------------------------------------------------------------------
# routing: every Feast task must land on a queue a running worker consumes
# ---------------------------------------------------------------------------


def _compose_queues(service: str) -> set:
    text = (REPO / "docker" / "docker-compose.yml").read_text()
    block = text.split(f"\n  {service}:\n", 1)[1]
    m = re.search(r"--queues=([a-z_,]+)", block)
    assert m, f"{service} has no --queues in docker/docker-compose.yml"
    return set(m.group(1).split(","))


@pytest.mark.unit
def test_weekly_full_materialize_no_longer_targets_the_unconsumed_ml_queue():
    """Measured 2026-09-22: 15 ``src.tasks.materialize_features`` messages sat undelivered
    on the ``ml`` queue in Redis (worker_heavy replicas: 0, #705) — 15 weeks of the weekly
    beat that never ran. The entry must target a queue worker_medium consumes."""
    entry = celery_app.conf.beat_schedule["feast-materialize-full-weekly"]
    assert entry["task"] == "src.tasks.materialize_features"
    assert entry["options"]["queue"] in _compose_queues("worker_medium")


@pytest.mark.unit
def test_every_feast_task_has_an_explicit_route_to_a_consumed_queue():
    routes = celery_app.conf.task_routes
    medium = _compose_queues("worker_medium")
    for task in (
        "src.tasks.materialize_features",
        "src.tasks.materialize_incremental_features",
        "src.tasks.check_feature_freshness",
        "src.tasks.materialize_feature_view",
    ):
        assert routes.get(task, {}).get("queue") in medium, task


# ---------------------------------------------------------------------------
# codex r3: a job row is counted only when it LANDED, and it lands atomically with
# its terminal status — never a `pending` row whose close-out failed separately
# ---------------------------------------------------------------------------


class _JobsInsertRefusingSupabase(FakeSupabase):
    """Registry inserts work; every ml_feast_materialization_jobs insert fails."""

    def table(self, name: str) -> _FakeQuery:
        q = super().table(name)
        if name == "ml_feast_materialization_jobs":
            q.execute = lambda: (_ for _ in ()).throw(RuntimeError("insert refused"))  # type: ignore[method-assign]
        return q


@pytest.mark.unit
def test_a_job_row_that_did_not_land_is_not_counted(fake_job, caplog):
    import asyncio

    from src.tasks.feast_tracking import record_materialization_jobs

    db = _JobsInsertRefusingSupabase()
    written = asyncio.run(
        record_materialization_jobs(
            db,
            job_type="incremental",
            requested_start=None,
            requested_end=datetime.now(timezone.utc),
            feature_views=["hcp_profile_features"],
            result=FAILED_INIT,
        )
    )
    assert written == 0
    assert "ml_feast_materialization_jobs" not in db.store  # nothing pending, nothing at all
    assert len(db.store["ml_feast_feature_views"]) == 1  # the registry row still landed


@pytest.mark.unit
def test_job_rows_land_with_their_terminal_status_in_one_insert(fake_db, fake_job):
    """No create-then-update pair: the single inserted row already carries the outcome."""
    fake_job.outcomes = {"incremental": FAILED_INIT, "full": FAILED_INIT}
    # Amended 2026-09-22 (#2207 follow-up): fail loud + no recovery on an init failure.
    with pytest.raises(RuntimeError):
        feast_tasks.materialize_incremental_features(feature_views=["hcp_profile_features"])
    rows = fake_db.store["ml_feast_materialization_jobs"]
    assert len(rows) == 1  # the incremental run only (no recovery for an init failure)
    for r in rows:
        assert r["status"] == "failed"
        assert r["error_message"] == "Failed to initialize Feast client"
        assert r["completed_at"]
    assert not any(r["status"] == "pending" for r in rows)
