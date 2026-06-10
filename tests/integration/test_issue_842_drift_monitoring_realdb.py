"""Faithful (real docker supabase-db, NO mocks) round-trip tests for #842.

Before the realignment, every drift_monitoring repository serialised its Pydantic
model straight into the table, so a write 42703'd on the FIRST column that did
not exist (model_version/detected_at/sample_size_*/metadata/training_config/…)
or violated a NOT-NULL column with no default (test_type/title/run_type/
trigger_type). These tests exercise each repository's real write+read methods
against the live schema and assert the round-trip succeeds, that the
``model_version`` handle is preserved/reconstructed, and that the rows address
only real columns. Every inserted row is cleaned up.

Reachability note (out of #842 scope, surfaced not silenced — filed as #845): in
production the API/tasks construct these repos as ``DriftHistoryRepository()``
with NO client, so ``self.client`` is None and the methods no-op — the schema
42703 is currently *masked* by that no-op. This realignment is the precondition
that makes wiring a real async client (the #820/#821 FAILS-OPEN family, tracked
in #845) safe. These tests pass a real client (the repo's designed contract) to
prove the schema layer end-to-end.

Opt-in (real docker supabase-db required), skipped in CI by default:
    E2I_DB_INTEGRATION=1 .venv/bin/pytest \
        tests/integration/test_issue_842_drift_monitoring_realdb.py -p no:cacheprovider -n0
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

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

# Local faithful runs need the Supabase env; the worktree has no .env so search
# upward (find_dotenv walks parent dirs -> the main repo .env). Gated so CI (where
# the module is skipped) never depends on it.
if os.getenv("E2I_DB_INTEGRATION") == "1":
    from dotenv import load_dotenv

    load_dotenv()

from src.memory.services.factories import get_async_supabase_client  # noqa: E402
from src.repositories.drift_monitoring import (  # noqa: E402
    DriftHistoryRepository,
    MonitoringAlertRepository,
    MonitoringRunRepository,
    PerformanceMetricRepository,
    RetrainingHistoryRepository,
)

_NOW = datetime.now(timezone.utc)
_BASELINE = {"start": _NOW - timedelta(days=14), "end": _NOW - timedelta(days=7)}
_CURRENT = {"start": _NOW - timedelta(days=7), "end": _NOW}


def _handle() -> str:
    return f"issue842_{uuid.uuid4().hex[:10]}"


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def client():
    # Bind a fresh singleton to THIS module's loop, then clear it so a
    # closed-loop client never leaks to another test module.
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    c = await get_async_supabase_client()
    assert c is not None, "real async supabase client required"
    yield c
    factories._async_supabase_client = None


_GUARDED_TABLES = (
    "ml_drift_history",
    "ml_monitoring_alerts",
    "ml_monitoring_runs",
    "ml_performance_metrics",
    "ml_retraining_history",
)


async def _count(client, table: str) -> int:
    res = await client.table(table).select("id", count="exact").limit(1).execute()
    return res.count or 0


@pytest_asyncio.fixture(loop_scope="module", scope="module", autouse=True)
async def _no_leak_guard(client):
    """Snapshot the five tables' rowcounts and assert each test self-cleans:
    no inserted row may survive the module (Finding 4 / anti-pollution)."""
    before = {t: await _count(client, t) for t in _GUARDED_TABLES}
    yield
    after = {t: await _count(client, t) for t in _GUARDED_TABLES}
    assert after == before, f"row leak — before={before} after={after}"


async def _delete_by_handle(client, table: str, jsonb_col: str, handle: str) -> None:
    await client.table(table).delete().eq(f"{jsonb_col}->>_model_version", handle).execute()


# ---------------------------------------------------------------------------
# ml_drift_history
# ---------------------------------------------------------------------------
async def test_drift_history_write_read_roundtrip(client):
    repo = DriftHistoryRepository(client)
    mv = _handle()
    drift_results = [
        {
            "feature": "age",
            "drift_type": "data",
            "test_statistic": 0.156,
            "p_value": 0.023,
            "drift_detected": True,
            "severity": "high",
            "drift_score": 0.42,
        },
        {
            "feature": "risk_score",
            "drift_type": "data",
            "test_statistic": 0.04,
            "p_value": 0.8,
            "drift_detected": False,
            "severity": "none",
        },
    ]
    drift_ids = []
    try:
        created = await repo.record_drift_results(mv, drift_results, _BASELINE, _CURRENT)
        assert len(created) == 2  # no 42703
        drift_ids = [r.id for r in created]

        latest = await repo.get_latest_drift_status(mv, limit=10)
        assert {r.feature_name for r in latest} == {"age", "risk_score"}
        assert all(r.model_version == mv for r in latest)  # handle reconstructed
        assert all(r.created_at is not None for r in latest)

        trend = await repo.get_drift_trend(mv, "age", days=30)
        assert len(trend) == 1
        assert trend[0].severity == "high"
    finally:
        # An AFTER-INSERT trigger (trigger_create_drift_alert) auto-creates an
        # alert referencing the drift row via drift_history_id; delete those
        # dependents first to satisfy the FK, then the drift rows.
        if drift_ids:
            await (
                client.table("ml_monitoring_alerts")
                .delete()
                .in_("drift_history_id", drift_ids)
                .execute()
            )
        await _delete_by_handle(client, "ml_drift_history", "raw_results", mv)


# ---------------------------------------------------------------------------
# ml_monitoring_alerts
# ---------------------------------------------------------------------------
async def test_alerts_write_read_ack_resolve(client):
    repo = MonitoringAlertRepository(client)
    mv = _handle()
    drift_results = [
        {"feature": "age", "drift_type": "data", "severity": "critical"},
        {"feature": "rx_count", "drift_type": "data", "severity": "high"},
    ]
    try:
        alerts = await repo.create_alerts_from_drift(mv, drift_results)
        assert len(alerts) == 2  # no 42703 / no 'warning' enum 22P02

        active = await repo.get_active_alerts(mv, limit=10)
        assert len(active) == 2
        assert all(a.model_version == mv for a in active)
        assert all(a.title for a in active)  # NOT NULL title populated
        assert {a.severity for a in active} == {"critical", "high"}

        ack = await repo.acknowledge_alert(active[0].id, "tester")
        assert ack is not None and ack.status == "acknowledged"
        assert ack.acknowledged_by == "tester"
        assert ack.acknowledged_at is not None

        res = await repo.resolve_alert(active[1].id, "tester")
        assert res is not None and res.status == "resolved"
        assert res.resolved_by == "tester"
    finally:
        await _delete_by_handle(client, "ml_monitoring_alerts", "metadata", mv)


# ---------------------------------------------------------------------------
# ml_monitoring_runs
# ---------------------------------------------------------------------------
async def test_runs_start_complete_roundtrip(client):
    repo = MonitoringRunRepository(client)
    mv = _handle()
    try:
        run = await repo.start_run(mv, run_type="manual", config={"time_window": "7d"})
        assert run.id  # inserted, no 42703

        updated = await repo.complete_run(
            run_id=run.id,
            features_checked=25,
            drift_detected_count=2,
            alerts_generated=1,
            duration_ms=12500,
        )
        assert updated is not None
        assert updated.status == "completed"
        assert updated.total_checks == 25
        assert float(updated.duration_seconds) == 12.5  # ms -> seconds round-trip

        recent = await repo.get_recent_runs(model_version=mv, limit=10)
        assert len(recent) == 1
        assert recent[0].model_version == mv
        assert recent[0].total_checks == 25
    finally:
        await _delete_by_handle(client, "ml_monitoring_runs", "config", mv)


# ---------------------------------------------------------------------------
# ml_performance_metrics
# ---------------------------------------------------------------------------
async def test_performance_metrics_write_read(client):
    repo = PerformanceMetricRepository(client)
    mv = _handle()
    try:
        created = await repo.record_metrics(
            model_version=mv,
            metrics={"roc_auc": 0.85, "precision": 0.71},
            sample_size=5000,
            window_start=_CURRENT["start"],
            window_end=_CURRENT["end"],
        )
        assert len(created) == 2  # no 42703

        trend = await repo.get_metric_trend(mv, "roc_auc", days=30)
        assert len(trend) == 1
        assert trend[0].metric_name == "roc_auc"
        assert float(trend[0].metric_value) == 0.85
        assert trend[0].model_version == mv
    finally:
        await _delete_by_handle(client, "ml_performance_metrics", "metadata", mv)


# ---------------------------------------------------------------------------
# ml_retraining_history
# ---------------------------------------------------------------------------
async def test_retraining_trigger_complete_roundtrip(client):
    repo = RetrainingHistoryRepository(client)
    old_v, new_v = _handle(), _handle()
    created_ids = []
    try:
        rec = await repo.trigger_retraining(
            old_model_version=old_v,
            new_model_version=new_v,
            trigger_reason="data_drift",
            drift_score_before=0.42,
            performance_before=0.85,
            training_config={"data_source": "optum/initiation", "target_outcome": "y"},
        )
        created_ids.append(rec.id)
        assert rec.id  # inserted, no 42703 (trigger_type NOT NULL derived='drift')

        fetched = await repo.get_by_id(rec.id)
        assert fetched is not None
        assert fetched.trigger_type == "drift"
        assert float(fetched.performance_before) == 0.85  # compat property <- old_metric_value
        assert float(fetched.drift_score_before) == 0.42  # preserved in config
        assert fetched.training_config["data_source"] == "optum/initiation"

        done = await repo.complete_retraining(rec.id, performance_after=0.90, success=True)
        assert done is not None
        assert done.status == "completed"
        assert float(done.performance_after) == 0.90  # compat property <- new_metric_value

        # mark_failed writes the reason to notes (no error_message column)
        rec2 = await repo.trigger_retraining(
            old_model_version=old_v,
            new_model_version=_handle(),
            trigger_reason="manual",
            drift_score_before=0.0,
            performance_before=0.80,
            training_config={"data_source": "x", "target_outcome": "y"},
        )
        created_ids.append(rec2.id)
        failed = await repo.mark_failed(rec2.id, "pipeline raised ValueError: boom")
        assert failed is not None
        assert failed.status == "failed"
        assert failed.notes == "pipeline raised ValueError: boom"
    finally:
        for rid in created_ids:
            await client.table("ml_retraining_history").delete().eq("id", rid).execute()
