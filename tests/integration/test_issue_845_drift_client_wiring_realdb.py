"""Faithful (real docker supabase-db, NO mocks) wiring tests for #845.

#842 realigned the drift-monitoring record models to the live ``ml_*`` schema but
left the *reachability* bug open: every call site constructed the repositories
**without a client** (``DriftHistoryRepository()`` …), so ``BaseRepository``'s
``if not self.client`` guards silently no-op'd — the Celery drift sweep persisted
nothing, the monitoring API read empty, performance metrics were dropped.

#845 wires a real async client into those call sites via
``get_drift_monitoring_client``. Where #842's real-DB suite passes a client
*explicitly* to the repos to prove the schema layer, THIS suite drives the
**wired call sites themselves** (the Celery task, the API route function, the
performance tracker, and a resolver-built alert repo) end-to-end and asserts that
real rows land / are read back — i.e. the dormant subsystem is now active.

Opt-in (real docker supabase-db required), skipped in CI by default:
    E2I_DB_INTEGRATION=1 .venv/bin/pytest \
        tests/integration/test_issue_845_drift_client_wiring_realdb.py -p no:cacheprovider -n0
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.skipif(
        os.getenv("E2I_DB_INTEGRATION") != "1",
        reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
    ),
    # One shared event loop for the whole module so the cached async Supabase
    # client (a process-global singleton) is created and torn down on a single
    # loop — avoids "Event loop is closed" across function-scoped loops.
    pytest.mark.asyncio(loop_scope="module"),
]

# Local faithful runs need the Supabase env; the worktree has no .env so let
# python-dotenv walk up to the main repo's .env. Gated so CI (skipped) never
# depends on it.
if os.getenv("E2I_DB_INTEGRATION") == "1":
    from dotenv import load_dotenv

    load_dotenv()

import src.repositories.drift_monitoring as dm  # noqa: E402
from src.memory.services.factories import get_async_supabase_client  # noqa: E402

_NOW = datetime.now(timezone.utc)
_BASELINE = {"start": _NOW - timedelta(days=14), "end": _NOW - timedelta(days=7)}
_CURRENT = {"start": _NOW - timedelta(days=7), "end": _NOW}

_GUARDED_TABLES = (
    "ml_drift_history",
    "ml_monitoring_alerts",
    "ml_monitoring_runs",
    "ml_performance_metrics",
    "ml_retraining_history",
)


def _handle() -> str:
    return f"issue845_{uuid.uuid4().hex[:10]}"


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def client():
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    c = await get_async_supabase_client()
    assert c is not None, "real async supabase client required"
    yield c
    factories._async_supabase_client = None


@pytest_asyncio.fixture(loop_scope="module", scope="module", autouse=True)
async def _no_leak_guard(client):
    """Every test must self-clean: no inserted row may survive the module."""

    async def _count(table: str) -> int:
        res = await client.table(table).select("id", count="exact").limit(1).execute()
        return res.count or 0

    before = {t: await _count(t) for t in _GUARDED_TABLES}
    yield
    after = {t: await _count(t) for t in _GUARDED_TABLES}
    assert after == before, f"row leak — before={before} after={after}"


async def _delete_by_handle(client, table: str, jsonb_col: str, handle: str) -> None:
    await client.table(table).delete().eq(f"{jsonb_col}->>_model_version", handle).execute()


# ---------------------------------------------------------------------------
# The resolver returns a live client (not None, queryable).
# ---------------------------------------------------------------------------
async def test_resolver_returns_live_client(client):
    resolved = await dm.get_drift_monitoring_client()
    assert resolved is not None
    # A trivial real query proves it is a live client bound to the real DB.
    res = await resolved.table("ml_monitoring_runs").select("id").limit(1).execute()
    assert res is not None


# ---------------------------------------------------------------------------
# Acceptance: a drift-detection task RUN persists a real ml_monitoring_runs row.
# Drives the real Celery task (checks off + features supplied so no upstream
# nodes/connector data are needed) — start_run + complete_run go through the
# wired client.
# ---------------------------------------------------------------------------
async def test_run_drift_detection_task_persists_real_run(client):
    from src.tasks.drift_monitoring_tasks import run_drift_detection

    handle = _handle()
    try:
        result = run_drift_detection(
            model_id=handle,
            time_window="7d",
            features=["unit_probe_feature"],
            check_data_drift=False,
            check_model_drift=False,
            check_concept_drift=False,
        )
        assert isinstance(result, dict)
        assert result.get("status") in ("completed", "success")

        rows = (
            await client.table("ml_monitoring_runs")
            .select("*")
            .eq("config->>_model_version", handle)
            .execute()
        ).data
        assert len(rows) == 1, f"expected exactly one persisted run, got {rows}"
        assert rows[0]["status"] == "completed"
        # The app handle is preserved + the run is linked to no fabricated model.
        assert rows[0]["config"]["_model_version"] == handle
    finally:
        await _delete_by_handle(client, "ml_monitoring_runs", "config", handle)


# ---------------------------------------------------------------------------
# Acceptance: GET /monitoring/drift/latest/{model_id} returns persisted drift.
# Drives the real route function, which resolves the client internally (#845).
# ---------------------------------------------------------------------------
async def test_drift_latest_endpoint_reads_persisted_drift(client):
    from src.api.routes.monitoring import get_latest_drift_status
    from src.repositories.drift_monitoring import DriftHistoryRepository

    handle = _handle()
    drift_ids: list = []
    try:
        # Seed real drift via the repo (the task's persistence path).
        seeded = await DriftHistoryRepository(client).record_drift_results(
            handle,
            [
                {
                    "feature": "days_since_last_visit",
                    "drift_type": "data",
                    "test_statistic": 0.21,
                    "p_value": 0.004,
                    "drift_detected": True,
                    "severity": "high",
                    "drift_score": 0.55,
                }
            ],
            _BASELINE,
            _CURRENT,
        )
        drift_ids = [r.id for r in seeded]
        assert len(drift_ids) == 1

        # The WIRED endpoint (resolves its own client) returns the persisted row.
        resp = await get_latest_drift_status(handle, limit=10)
        assert resp.features_checked >= 1
        assert any(r.feature == "days_since_last_visit" for r in resp.results)
        assert resp.overall_drift_score > 0.0
    finally:
        # An AFTER-INSERT trigger auto-creates an alert referencing the drift row;
        # delete dependents first to satisfy the FK, then the drift rows.
        if drift_ids:
            await (
                client.table("ml_monitoring_alerts")
                .delete()
                .in_("drift_history_id", drift_ids)
                .execute()
            )
        await _delete_by_handle(client, "ml_drift_history", "raw_results", handle)


# ---------------------------------------------------------------------------
# Acceptance: alerts persist through the wired path (resolver-built repo —
# exactly what the Celery task uses to create alerts).
# ---------------------------------------------------------------------------
async def test_alerts_persist_via_resolved_client(client):
    from src.repositories.drift_monitoring import MonitoringAlertRepository

    repo = MonitoringAlertRepository(await dm.get_drift_monitoring_client())
    handle = _handle()
    try:
        alerts = await repo.create_alerts_from_drift(
            handle, [{"feature": "rx_count", "drift_type": "data", "severity": "critical"}]
        )
        assert len(alerts) == 1
        rows = (
            await client.table("ml_monitoring_alerts")
            .select("*")
            .eq("metadata->>_model_version", handle)
            .execute()
        ).data
        assert len(rows) == 1
        assert rows[0]["severity"] == "critical"
    finally:
        await _delete_by_handle(client, "ml_monitoring_alerts", "metadata", handle)


# ---------------------------------------------------------------------------
# Performance tracker persists a real ml_performance_metrics row (the wired
# PerformanceTracker.record_performance resolves its own client now).
# ---------------------------------------------------------------------------
async def test_performance_tracker_persists_real_metric(client):
    from src.services.performance_tracking import PerformanceTracker

    handle = _handle()
    try:
        snap = await PerformanceTracker().record_performance(
            model_version=handle,
            predictions=np.array([1, 0, 1, 0, 1]),
            actuals=np.array([1, 0, 0, 0, 1]),
        )
        assert snap is not None and snap.metrics  # metrics computed

        rows = (
            await client.table("ml_performance_metrics")
            .select("*")
            .eq("metadata->>_model_version", handle)
            .execute()
        ).data
        assert len(rows) >= 1, "performance metrics must persist a real row"
        assert all(r["metadata"]["_model_version"] == handle for r in rows)
    finally:
        await _delete_by_handle(client, "ml_performance_metrics", "metadata", handle)
