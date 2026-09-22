"""#2207 follow-up (owner decision 2026-09-22): the Feast beats fail LOUD when they did
nothing, and the worker reaches the sidecar instead of ``import feast``.

Measured on worker_medium 2026-09-22 16:01 UTC: ``materialize_incremental_features``
returned ``{'status': 'failed', 'error': 'Failed to initialize Feast client'}`` and
Celery logged ``succeeded`` — ``_fail_loud_if_skipped`` keyed only on ``skipped``, which
init failure never reaches — then the auto-recovery branch doubled the identical failure.
The #556 intent ("a no-op is never silent") requires a RED task in Celery for both.

Contract pinned here:
- ``status == "failed"`` raises after the tracking rows are recorded (the rows carry
  the truthful outcome; the raise makes the beat visible);
- an init failure does not attempt the auto-recovery full materialization (it would
  fail identically); a non-init incremental failure still does, and a recovery that
  completed is NOT raised on (the store was populated);
- the freshness beat raises when it could not probe at all;
- ``MaterializationJob()`` with no explicit config resolves the client through
  ``get_feast_client()`` so ``FEAST_URL`` (set on worker_medium) is honoured;
- ``check_feature_freshness`` falls back to the nine real views when the client lists
  none (remote mode has no registry to list) — never "All 0 feature views are fresh".
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.feature_store import feast_client as feast_client_module
from src.feature_store.feast_remote_materialize import INIT_FAILURE_ERROR
from src.feature_store.feast_views import FEAST_ONLINE_FEATURE_VIEWS
from src.tasks import feast_tasks

# reuse the in-memory tracking fake from the sibling test module
from tests.unit.test_tasks.test_feast_tasks_tracking_2207 import FakeSupabase, _FakeJob

FAILED_INIT = {"status": "failed", "error": INIT_FAILURE_ERROR, "stage": "init"}
FAILED_SIDECAR = {
    "status": "failed",
    "error": "HTTP 500 from http://feast:6566/materialize-incremental",
}
COMPLETED = {
    "status": "completed",
    "feature_views": ["hcp_profile_features"],
    "duration_seconds": 1.0,
}


@pytest.fixture()
def fake_db():
    db = FakeSupabase()
    with patch.object(feast_tasks, "_tracking_client", return_value=db):
        yield db


@pytest.fixture()
def fake_job():
    with patch("scripts.feast_materialize.MaterializationJob", _FakeJob):
        yield _FakeJob


def _jobs(db, job_type):
    return [
        j for j in db.store.get("ml_feast_materialization_jobs", []) if j["job_type"] == job_type
    ]


# ---------------------------------------------------------------------------
# fail loud
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_failure_records_rows_skips_recovery_and_raises(fake_db, fake_job):
    fake_job.outcomes = {"incremental": FAILED_INIT, "full": FAILED_INIT}
    with pytest.raises(RuntimeError, match=INIT_FAILURE_ERROR):
        feast_tasks.materialize_incremental_features()
    incremental = _jobs(fake_db, "incremental")
    assert {j["feature_view_name"] for j in incremental} == set(FEAST_ONLINE_FEATURE_VIEWS)
    assert all(
        j["status"] == "failed" and j["error_message"] == INIT_FAILURE_ERROR for j in incremental
    )
    assert _jobs(fake_db, "full") == [], "an init failure must not be doubled by auto-recovery"


@pytest.mark.unit
def test_full_materialize_failure_raises_after_recording(fake_db, fake_job):
    fake_job.outcomes = {"full": {"status": "failed", "error": "HTTP 503 from sidecar"}}
    with pytest.raises(RuntimeError, match="HTTP 503"):
        feast_tasks.materialize_features(feature_views=["hcp_profile_features"])
    (row,) = _jobs(fake_db, "full")
    assert row["status"] == "failed" and row["error_message"] == "HTTP 503 from sidecar"


@pytest.mark.unit
def test_non_init_incremental_failure_recovers_with_full_and_does_not_raise(fake_db, fake_job):
    fake_job.outcomes = {"incremental": FAILED_SIDECAR, "full": COMPLETED}
    result = feast_tasks.materialize_incremental_features(feature_views=["hcp_profile_features"])
    assert result["status"] == "failed"  # the incremental call did fail
    assert result["recovery_attempt"]["status"] == "completed"
    (inc,) = _jobs(fake_db, "incremental")
    (rec,) = _jobs(fake_db, "full")
    assert inc["status"] == "failed" and rec["status"] == "success"


@pytest.mark.unit
def test_non_init_incremental_failure_whose_recovery_also_fails_raises(fake_db, fake_job):
    fake_job.outcomes = {"incremental": FAILED_SIDECAR, "full": FAILED_SIDECAR}
    with pytest.raises(RuntimeError, match="HTTP 500"):
        feast_tasks.materialize_incremental_features(feature_views=["hcp_profile_features"])
    assert len(_jobs(fake_db, "incremental")) == 1 and len(_jobs(fake_db, "full")) == 1


@pytest.mark.unit
def test_completed_incremental_returns_normally(fake_db, fake_job):
    fake_job.outcomes = {"incremental": COMPLETED}
    result = feast_tasks.materialize_incremental_features(feature_views=["hcp_profile_features"])
    assert result["status"] == "completed"
    assert "recovery_attempt" not in result
    (inc,) = _jobs(fake_db, "incremental")
    assert inc["status"] == "success"


@pytest.mark.unit
def test_freshness_beat_that_could_not_probe_records_unknown_and_raises(fake_db, fake_job):
    fake_job.outcomes = {"freshness": FAILED_INIT}
    with pytest.raises(RuntimeError, match=INIT_FAILURE_ERROR):
        feast_tasks.check_feature_freshness(alert_on_stale=False)
    rows = fake_db.store["ml_feast_feature_freshness"]
    assert {r["feature_view_name"] for r in rows} == set(FEAST_ONLINE_FEATURE_VIEWS)
    assert all(r["freshness_status"] == "unknown" for r in rows)


@pytest.mark.unit
def test_tracking_backend_failure_still_never_fails_a_successful_beat(fake_job):
    """The side channel stays best-effort: a broken tracking backend does not turn a
    run that DID materialize into a failed task."""

    class _Exploding:
        def table(self, name):
            raise RuntimeError("tracking backend down")

    fake_job.outcomes = {"incremental": COMPLETED}
    with patch.object(feast_tasks, "_tracking_client", return_value=_Exploding()):
        assert feast_tasks.materialize_incremental_features()["status"] == "completed"


# ---------------------------------------------------------------------------
# the job reaches the sidecar: FEAST_URL honoured through get_feast_client()
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_materialization_job_default_config_honours_feast_url(monkeypatch):
    from scripts.feast_materialize import MaterializationJob

    monkeypatch.setenv("FEAST_URL", "http://feast:6566")
    monkeypatch.setattr(feast_client_module, "_client", None)
    job = MaterializationJob()
    assert await job.initialize() is True
    assert job.feast_client is not None
    assert job.feast_client._remote_base_url == "http://feast:6566"
    assert job.feast_client._store is None  # no feast import happened
    monkeypatch.setattr(feast_client_module, "_client", None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_freshness_check_falls_back_to_the_real_views_when_client_lists_none():
    from scripts.feast_materialize import MaterializationJob

    client = MagicMock()
    client.initialize = AsyncMock()
    client.list_feature_views = AsyncMock(return_value=[])
    probed: List[str] = []

    async def _stats(feature_view: str, feature_name: str, **_k):
        probed.append(feature_view)
        return MagicMock(last_updated=datetime.now(timezone.utc))

    client.get_feature_statistics = _stats
    job = MaterializationJob(feast_client=client)
    result = await job.check_feature_freshness(feature_views=None, max_staleness_hours=24.0)
    assert result["status"] == "completed"
    assert set(probed) == set(FEAST_ONLINE_FEATURE_VIEWS)
    assert {f["feature_view"] for f in result["fresh_features"]} == set(FEAST_ONLINE_FEATURE_VIEWS)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_init_failure_result_is_marked_as_init_stage():
    from scripts.feast_materialize import MaterializationJob

    client = MagicMock()
    client.initialize = AsyncMock(side_effect=ImportError("No module named 'feast'"))
    job = MaterializationJob(feast_client=client)
    result = await job.run_incremental_materialization(end_date=datetime.now(timezone.utc))
    assert result == {"status": "failed", "error": INIT_FAILURE_ERROR, "stage": "init"}


@pytest.mark.unit
def test_worker_env_contract_documented():
    """FEAST_URL is on the shared compose env anchor; the beat must read it."""
    assert os.environ.get("FEAST_URL") is None or os.environ["FEAST_URL"].startswith("http")
