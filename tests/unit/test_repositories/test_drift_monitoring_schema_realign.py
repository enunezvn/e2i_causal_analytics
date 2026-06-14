"""RED-first unit tests for #842 — drift_monitoring.py record models realigned
to the live ml_* monitoring schema (database/ml/017_model_monitoring_tables.sql).

These tests pin the column mapping each record serializes to. They are pure
(no DB): they exercise ``to_db_row()`` (write mapping), ``from_db_row()`` (read
mapping), the pure mapping helpers, and the backward-compat accessors the
service layer reads.

Design: **record fields == the real DB column names** (single source of truth);
the only non-column field is ``model_version`` (the app-level handle, preserved
inside the table's jsonb column because there is no model_version column — the
real linkage column is ``model_id`` uuid FK). Repository *methods* keep the
app-vocabulary parameters and map them; the retraining record exposes compat
properties so ``retraining_trigger.py`` stays unchanged.

A write/read currently 42703s on the FIRST mismatched column; these tests assert
the realigned mapping so inserts/selects address only real columns.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pytest

from src.repositories.drift_monitoring import (
    _MV_KEY,
    DriftHistoryRecord,
    MonitoringAlertRecord,
    MonitoringRunRecord,
    MonitoringRunRepository,
    PerformanceMetricRecord,
    RetrainingHistoryRecord,
    _derive_trigger_type,
    _ms_to_seconds,
    _normalize_alert_status,
    _normalize_drift_type,
    _normalize_severity,
    _normalize_test_type,
)

_T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
_T1 = datetime(2026, 6, 8, tzinfo=timezone.utc)
_T2 = datetime(2026, 6, 8, tzinfo=timezone.utc)
_T3 = datetime(2026, 6, 15, tzinfo=timezone.utc)

_DRIFT_COLS = {
    "id",
    "model_id",
    "experiment_id",
    "deployment_id",
    "drift_type",
    "feature_name",
    "test_type",
    "test_statistic",
    "p_value",
    "threshold",
    "drift_detected",
    "severity",
    "baseline_start",
    "baseline_end",
    "current_start",
    "current_end",
    "baseline_mean",
    "baseline_std",
    "baseline_min",
    "baseline_max",
    "baseline_count",
    "current_mean",
    "current_std",
    "current_min",
    "current_max",
    "current_count",
    "drift_score",
    "contribution_to_overall",
    "raw_results",
    "detected_by",
    "created_at",
}
_ALERT_COLS = {
    "id",
    "alert_type",
    "title",
    "severity",
    "status",
    "model_id",
    "experiment_id",
    "deployment_id",
    "drift_history_id",
    "message",
    "affected_features",
    "drift_type",
    "composite_drift_score",
    "recommended_action",
    "recommended_priority",
    "auto_action_taken",
    "auto_action_details",
    "notified_channels",
    "notification_sent_at",
    "notification_error",
    "acknowledged_at",
    "acknowledged_by",
    "acknowledgement_notes",
    "resolved_at",
    "resolved_by",
    "resolution_notes",
    "resolution_action",
    "triggered_retraining",
    "retraining_job_id",
    "metadata",
    "created_at",
    "updated_at",
}
_RUN_COLS = {
    "id",
    "run_type",
    "trigger_type",
    "model_ids",
    "deployment_ids",
    "started_at",
    "completed_at",
    "duration_seconds",
    "total_checks",
    "drift_detected_count",
    "alerts_generated",
    "critical_count",
    "high_count",
    "medium_count",
    "low_count",
    "overall_health_score",
    "requires_attention",
    "status",
    "error_message",
    "config",
    "summary",
    "created_by",
    "created_at",
}
_PERF_COLS = {
    "id",
    "model_id",
    "experiment_id",
    "deployment_id",
    "metric_name",
    "metric_value",
    "data_split",
    "segment",
    "sample_size",
    "positive_rate",
    "ci_lower",
    "ci_upper",
    "ci_level",
    "baseline_value",
    "delta",
    "delta_pct",
    "is_degraded",
    "measured_at",
    "measurement_window_start",
    "measurement_window_end",
    "source",
    "metadata",
    "created_at",
}
_RETRAIN_COLS = {
    "id",
    "trigger_type",
    "alert_id",
    "monitoring_run_id",
    "model_id",
    "old_model_version",
    "new_model_version",
    "trigger_reason",
    "drift_severity",
    "performance_delta",
    "training_run_id",
    "status",
    "old_metric_value",
    "new_metric_value",
    "improvement",
    "auto_deployed",
    "deployed_at",
    "deployment_id",
    "triggered_at",
    "completed_at",
    "duration_seconds",
    "config",
    "notes",
    "created_by",
    "created_at",
}


# ---------------------------------------------------------------------------
# Pure mapping helpers
# ---------------------------------------------------------------------------
def test_normalize_severity_coerces_to_valid_enum():
    # drift_severity_enum = none/low/medium/high/critical — 'warning' is NOT valid
    assert _normalize_severity("warning") == "high"
    assert _normalize_severity("critical") == "critical"
    assert _normalize_severity("none") == "none"
    assert _normalize_severity("bogus") in {"none", "low", "medium", "high", "critical"}


def test_normalize_test_type_coerces_to_valid_enum():
    # statistical_test_enum = psi/ks/chi_square/wasserstein/js_divergence/importance_correlation
    assert _normalize_test_type("psi") == "psi"
    assert _normalize_test_type("KS") == "ks"
    assert _normalize_test_type(None) in {
        "psi",
        "ks",
        "chi_square",
        "wasserstein",
        "js_divergence",
        "importance_correlation",
    }
    assert _normalize_test_type("not_a_test") == "psi"


def test_derive_trigger_type_maps_reason_to_enumlike():
    # trigger_type column is NOT NULL with no default: 'drift'/'performance'/'scheduled'/'manual'
    assert _derive_trigger_type("data_drift") == "drift"
    assert _derive_trigger_type("model_drift") == "drift"
    assert _derive_trigger_type("concept_drift") == "drift"
    assert _derive_trigger_type("performance_degradation") == "performance"
    assert _derive_trigger_type("scheduled") == "scheduled"
    assert _derive_trigger_type("manual") == "manual"


def test_ms_to_seconds():
    assert _ms_to_seconds(12500) == 12.5
    assert _ms_to_seconds(0) == 0.0


def test_normalize_drift_type_coerces_to_valid_enum():
    # drift_type_enum = data/model/concept — an unknown literal would 22P02
    assert _normalize_drift_type("data") == "data"
    assert _normalize_drift_type("MODEL") == "model"
    assert _normalize_drift_type("concept") == "concept"
    assert _normalize_drift_type("covariate") == "data"  # unknown -> NOT-NULL default
    assert _normalize_drift_type("covariate", default=None) is None  # nullable column
    assert _normalize_drift_type(None, default=None) is None


def test_normalize_alert_status_coerces_to_valid_enum():
    # alert_status_enum = active/acknowledged/investigating/resolved/dismissed
    for valid in ("active", "acknowledged", "investigating", "resolved", "dismissed"):
        assert _normalize_alert_status(valid) == valid
    assert _normalize_alert_status("ACKNOWLEDGED") == "acknowledged"
    assert _normalize_alert_status("closed") == "active"  # unknown -> safe default
    assert _normalize_alert_status(None) == "active"


# ---------------------------------------------------------------------------
# DriftHistoryRecord -> ml_drift_history
# ---------------------------------------------------------------------------
def _drift_record() -> DriftHistoryRecord:
    return DriftHistoryRecord(
        model_version="propensity_v2.1.0",
        feature_name="age",
        drift_type="data",
        test_type="psi",
        test_statistic=0.156,
        p_value=0.023,
        drift_detected=True,
        severity="medium",
        baseline_start=_T0,
        baseline_end=_T1,
        current_start=_T2,
        current_end=_T3,
        baseline_count=1000,
        current_count=1200,
        drift_score=0.42,
    )


def test_drift_history_to_db_row_only_real_columns():
    row = _drift_record().to_db_row()
    assert set(row).issubset(_DRIFT_COLS), f"phantom columns: {set(row) - _DRIFT_COLS}"
    for legacy in (
        "model_version",
        "detected_at",
        "sample_size_baseline",
        "sample_size_current",
        "metadata",
    ):
        assert legacy not in row, f"{legacy} leaked into the insert row"


def test_drift_history_column_value_mapping():
    row = _drift_record().to_db_row()
    assert row["baseline_count"] == 1000
    assert row["current_count"] == 1200
    assert "created_at" in row  # was detected_at
    assert row["test_type"] == "psi"  # NOT NULL enum supplied
    assert row.get("model_id") is None  # uuid FK, NULL when unresolved
    assert row["raw_results"]["_model_version"] == "propensity_v2.1.0"  # handle preserved


def test_drift_history_from_db_row_reconstructs_model_version():
    row = {
        "id": "11111111-1111-1111-1111-111111111111",
        "model_id": None,
        "drift_type": "data",
        "feature_name": "age",
        "test_type": "psi",
        "severity": "low",
        "drift_detected": False,
        "baseline_start": _T0.isoformat(),
        "baseline_end": _T1.isoformat(),
        "current_start": _T2.isoformat(),
        "current_end": _T3.isoformat(),
        "created_at": _T3.isoformat(),
        "raw_results": {"_model_version": "propensity_v2.1.0"},
    }
    rec = DriftHistoryRecord.from_db_row(row)
    assert rec.model_version == "propensity_v2.1.0"
    assert rec.created_at is not None


def test_drift_history_invalid_drift_type_coerced_to_valid_enum():
    """drift_type is a NOT-NULL drift_type_enum (data/model/concept); an unknown
    value passed through verbatim would 22P02 the insert."""
    rec = DriftHistoryRecord(
        drift_type="covariate",  # not a valid drift_type_enum value
        feature_name="age",
        baseline_start=_T0,
        baseline_end=_T1,
        current_start=_T2,
        current_end=_T3,
    )
    assert rec.to_db_row()["drift_type"] in {"data", "model", "concept"}


# ---------------------------------------------------------------------------
# MonitoringAlertRecord -> ml_monitoring_alerts
# ---------------------------------------------------------------------------
def test_alert_to_db_row_only_real_columns_and_title_present():
    rec = MonitoringAlertRecord(
        model_version="propensity_v2.1.0",
        alert_type="data_drift",
        severity="critical",
        message="CRITICAL data drift detected in: age, rx_count",
        affected_features=["age", "rx_count"],
        recommended_action="Retrain",
    )
    row = rec.to_db_row()
    assert set(row).issubset(_ALERT_COLS), f"phantom columns: {set(row) - _ALERT_COLS}"
    assert "model_version" not in row
    assert "triggered_at" not in row  # no such column; created_at is the fire time
    assert row["title"]  # NOT NULL, derived from message
    assert row["metadata"]["_model_version"] == "propensity_v2.1.0"


def test_alert_severity_warning_maps_to_valid_enum():
    rec = MonitoringAlertRecord(
        model_version="m",
        alert_type="data_drift",
        severity="warning",
        message="HIGH data drift",
        recommended_action="Monitor",
    )
    assert rec.to_db_row()["severity"] == "high"


def test_alert_invalid_drift_type_coerced_to_valid_or_none():
    """ml_monitoring_alerts.drift_type is a nullable drift_type_enum; an unknown
    value must be coerced (to None here, since the column is nullable) rather
    than written verbatim and 22P02'ing."""
    rec = MonitoringAlertRecord(
        model_version="m",
        alert_type="covariate_drift",
        severity="high",
        drift_type="covariate",
        message="HIGH covariate drift",
    )
    dt = rec.to_db_row()["drift_type"]
    assert dt is None or dt in {"data", "model", "concept"}


def test_alert_invalid_status_coerced_to_valid_enum():
    """ml_monitoring_alerts.status is alert_status_enum; an unknown status must
    be coerced (to the safe 'active' default) rather than 22P02'ing."""
    rec = MonitoringAlertRecord(
        alert_type="data_drift", severity="high", message="x", status="closed"
    )
    assert rec.to_db_row()["status"] in {
        "active",
        "acknowledged",
        "investigating",
        "resolved",
        "dismissed",
    }


# ---------------------------------------------------------------------------
# MonitoringRunRecord -> ml_monitoring_runs   (fields == DB columns)
# ---------------------------------------------------------------------------
def test_run_to_db_row_real_columns():
    rec = MonitoringRunRecord(
        model_version="propensity_v2.1.0",
        run_type="full",
        trigger_type="scheduled",
        total_checks=25,
        drift_detected_count=2,
        alerts_generated=1,
        duration_seconds=12.5,
    )
    row = rec.to_db_row()
    assert set(row).issubset(_RUN_COLS), f"phantom columns: {set(row) - _RUN_COLS}"
    assert "model_version" not in row
    assert "features_checked" not in row
    assert "duration_ms" not in row
    assert row["run_type"]  # NOT NULL
    assert row["trigger_type"]  # NOT NULL
    assert row["total_checks"] == 25
    assert row["duration_seconds"] == 12.5
    assert row["config"]["_model_version"] == "propensity_v2.1.0"


# ---------------------------------------------------------------------------
# PerformanceMetricRecord -> ml_performance_metrics  (fields == DB columns)
# ---------------------------------------------------------------------------
def test_perf_to_db_row_real_columns():
    rec = PerformanceMetricRecord(
        model_version="propensity_v2.1.0",
        metric_name="roc_auc",
        metric_value=0.85,
        sample_size=5000,
        measurement_window_start=_T2,
        measurement_window_end=_T3,
    )
    row = rec.to_db_row()
    assert set(row).issubset(_PERF_COLS), f"phantom columns: {set(row) - _PERF_COLS}"
    assert "model_version" not in row
    assert "window_start" not in row
    assert "window_end" not in row
    assert "recorded_at" not in row
    assert "measurement_window_start" in row
    assert "measurement_window_end" in row
    assert "measured_at" in row  # was recorded_at
    assert row["metadata"]["_model_version"] == "propensity_v2.1.0"


# ---------------------------------------------------------------------------
# RetrainingHistoryRecord -> ml_retraining_history  (fields == DB columns)
# ---------------------------------------------------------------------------
def test_retrain_to_db_row_real_columns():
    rec = RetrainingHistoryRecord(
        trigger_type="drift",
        old_model_version="propensity_v2.1.0",
        new_model_version="propensity_v2.2.0",
        trigger_reason="data_drift",
        old_metric_value=0.85,
        config={"data_source": "optum/initiation", "drift_score_before": 0.42},
        status="pending",
    )
    row = rec.to_db_row()
    assert set(row).issubset(_RETRAIN_COLS), f"phantom columns: {set(row) - _RETRAIN_COLS}"
    for legacy in (
        "training_config",
        "performance_before",
        "performance_after",
        "drift_score_before",
        "error_message",
    ):
        assert legacy not in row, f"{legacy} leaked into the insert row"
    assert row["trigger_type"] == "drift"  # NOT NULL
    assert row["old_metric_value"] == 0.85
    assert row["config"]["data_source"] == "optum/initiation"
    assert row["config"]["drift_score_before"] == 0.42


def test_retrain_compat_properties_read_from_real_columns():
    """retraining_trigger.py reads record.performance_before/performance_after/
    drift_score_before/training_config — these derive from the real columns so
    that consumer stays unchanged."""
    row = {
        "id": "22222222-2222-2222-2222-222222222222",
        "trigger_type": "drift",
        "old_model_version": "propensity_v2.1.0",
        "new_model_version": "propensity_v2.2.0",
        "trigger_reason": "data_drift",
        "status": "completed",
        "old_metric_value": 0.85,
        "new_metric_value": 0.90,
        "config": {"data_source": "optum", "drift_score_before": 0.42},
        "triggered_at": _T3.isoformat(),
        "created_at": _T3.isoformat(),
    }
    rec = RetrainingHistoryRecord.from_db_row(row)
    assert rec.performance_before == 0.85
    assert rec.performance_after == 0.90
    assert rec.drift_score_before == 0.42
    assert rec.training_config["data_source"] == "optum"


# ---------------------------------------------------------------------------
# MonitoringRunRepository.get_recent_runs model filter
# ---------------------------------------------------------------------------
# RED-first regression for the live 500 on
#   GET /monitoring/runs?model_id=<name>&days=30   and
#   GET /monitoring/health/<name>
# (error 42703: ``column ml_monitoring_runs.model_id does not exist``).
#
# Root cause: ``ml_monitoring_runs`` links models via the ``model_ids UUID[]``
# array column (database/ml/017_model_monitoring_tables.sql line 291) — it has
# NO scalar ``model_id`` column (unlike ml_drift_history / ml_performance_metrics
# / ml_monitoring_alerts, which do). ``get_recent_runs`` reused the scalar
# ``_apply_model_filter`` helper, which on a resolved-uuid handle emits
# ``.eq("model_id", uuid)`` → PostgREST 42703 → the repo raises → the route's
# ``_log_and_500`` returns HTTP 500. The unfiltered path never filters, so it
# worked. The 2 live seed rows store the model both in ``model_ids`` (e.g.
# ``{5fd7826b-...}``) and in ``config->>_model_version``.


class _FakeQuery:
    """Minimal async PostgREST query-builder double.

    Records every filter operation issued so the test can assert WHICH column +
    operator the repository targets (the bug is a phantom ``model_id`` column).
    Chainable + awaitable-execute, mirroring the real client's surface used by
    ``get_recent_runs`` (select/order/limit/eq/contains/gte/execute).
    """

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        # list of (op, column, value)
        self.ops: List[Tuple[str, str, Any]] = []

    def select(self, *_a, **_k) -> "_FakeQuery":
        return self

    def order(self, *_a, **_k) -> "_FakeQuery":
        return self

    def limit(self, *_a, **_k) -> "_FakeQuery":
        return self

    def eq(self, column: str, value: Any) -> "_FakeQuery":
        self.ops.append(("eq", column, value))
        return self

    def contains(self, column: str, value: Any) -> "_FakeQuery":
        self.ops.append(("contains", column, value))
        return self

    def gte(self, column: str, value: Any) -> "_FakeQuery":
        self.ops.append(("gte", column, value))
        return self

    async def execute(self) -> Any:
        # Honor the recorded filters faithfully so the returned rows reflect what
        # the column/operator combination would actually match.
        result = list(self._rows)
        for op, column, value in self.ops:
            if column == "model_id":
                # The phantom column: in the real DB this is the 42703. Modelling
                # it as "matches nothing" still makes the assertions below catch
                # the bug (no real row is ever returned via this filter).
                result = []
            elif op == "contains" and column == "model_ids":
                result = [r for r in result if value[0] in (r.get("model_ids") or [])]
            elif op == "eq" and _MV_KEY in column:
                # jsonb handle fallback, e.g. "config->>_model_version"
                result = [r for r in result if (r.get("config") or {}).get(_MV_KEY) == value]
        return type("Res", (), {"data": result})()


class _FakeTable:
    def __init__(self, q: _FakeQuery):
        self._q = q

    def select(self, *a, **k) -> _FakeQuery:
        return self._q.select(*a, **k)


class _FakeClient:
    """Resolves the model name -> uuid (registry) and serves the runs table."""

    def __init__(self, q: _FakeQuery, *, registry: Dict[str, str]):
        self._q = q
        self._registry = registry  # model_name -> id

    def table(self, name: str) -> Any:
        if name == "ml_model_registry":
            return _FakeRegistry(self._registry)
        return _FakeTable(self._q)


class _FakeRegistry:
    """Mimics the ml_model_registry select(.eq(col, val)).limit(1).execute()
    used by ``_resolve_model_id`` (tries model_version then model_name)."""

    def __init__(self, registry: Dict[str, str]):
        self._registry = registry
        self._pending: Tuple[str, str] | None = None

    def select(self, *_a, **_k) -> "_FakeRegistry":
        return self

    def eq(self, column: str, value: str) -> "_FakeRegistry":
        self._pending = (column, value)
        return self

    def limit(self, *_a, **_k) -> "_FakeRegistry":
        return self

    async def execute(self) -> Any:
        data: List[Dict[str, Any]] = []
        if self._pending is not None:
            column, value = self._pending
            if column == "model_name" and value in self._registry:
                data = [{"id": self._registry[value]}]
        return type("Res", (), {"data": data})()


# The live fixtures (verified against the prod DB on 2026-06-14):
_FULL_NAME = "csu_treatment_initiation_lr_full_v1"
_FULL_ID = "5fd7826b-28d7-491b-b9b1-8b5494dbe1ff"
_BAL_NAME = "csu_treatment_initiation_lr_balanced_v1"
_BAL_ID = "d765b451-12df-46df-955f-63359b506b52"

_LIVE_RUNS = [
    {
        "id": "09ba6fc6-5170-439f-85a2-c3da8ff61db9",
        "model_ids": [_FULL_ID],
        "config": {"_model_version": _FULL_ID},
        "run_type": "full",
        "trigger_type": "scheduled",
        "status": "completed",
        "started_at": "2026-06-14T00:23:58.583875+00:00",
        "completed_at": "2026-06-14T00:23:59.278485+00:00",
        "total_checks": 0,
        "drift_detected_count": 0,
        "alerts_generated": 0,
        "duration_seconds": 0.81,
        "error_message": None,
    },
    {
        "id": "1bf0b9b1-a9cc-4cd7-a950-3e61d321b15b",
        "model_ids": [_BAL_ID],
        "config": {"_model_version": _BAL_ID},
        "run_type": "full",
        "trigger_type": "scheduled",
        "status": "completed",
        "started_at": "2026-06-14T00:23:58.540911+00:00",
        "completed_at": "2026-06-14T00:23:59.285629+00:00",
        "total_checks": 0,
        "drift_detected_count": 0,
        "alerts_generated": 0,
        "duration_seconds": 0.85,
        "error_message": None,
    },
]


@pytest.mark.asyncio
async def test_recent_runs_filter_never_targets_phantom_model_id_column():
    """RED-first: ``get_recent_runs(model_version=<name>)`` must NOT emit a filter
    against ``ml_monitoring_runs.model_id`` (no such column → 42703 → HTTP 500).

    The handle resolves to a uuid via the registry; the run table links via the
    ``model_ids`` array, so the resolved uuid must be filtered with an
    array-contains on ``model_ids`` — never ``.eq("model_id", uuid)``.
    """
    q = _FakeQuery([dict(r) for r in _LIVE_RUNS])
    client = _FakeClient(q, registry={_FULL_NAME: _FULL_ID, _BAL_NAME: _BAL_ID})
    repo = MonitoringRunRepository(client)

    await repo.get_recent_runs(model_version=_FULL_NAME, limit=50)

    phantom = [op for op in q.ops if op[1] == "model_id"]
    assert not phantom, (
        "get_recent_runs filtered ml_monitoring_runs on the phantom scalar "
        f"'model_id' column (42703 in prod): {phantom!r}"
    )


@pytest.mark.asyncio
async def test_recent_runs_returns_real_row_for_resolved_uuid():
    """The model-name-filtered query returns the REAL run row whose ``model_ids``
    array contains the resolved uuid — the same row the unfiltered endpoint shows.
    """
    q = _FakeQuery([dict(r) for r in _LIVE_RUNS])
    client = _FakeClient(q, registry={_FULL_NAME: _FULL_ID, _BAL_NAME: _BAL_ID})
    repo = MonitoringRunRepository(client)

    runs = await repo.get_recent_runs(model_version=_FULL_NAME, limit=50)

    assert len(runs) == 1, f"expected exactly the full_v1 run, got {len(runs)}"
    assert str(runs[0].id) == "09ba6fc6-5170-439f-85a2-c3da8ff61db9"
    # array-contains is the operator that addresses the model_ids[] column
    assert ("contains", "model_ids", [_FULL_ID]) in q.ops


@pytest.mark.asyncio
async def test_recent_runs_falls_back_to_jsonb_handle_when_unresolved():
    """When the handle does NOT resolve to a registry uuid (unregistered model),
    filter by the preserved ``config->>_model_version`` handle — never by a
    phantom scalar column, and never silently returning everything."""
    q = _FakeQuery([dict(r) for r in _LIVE_RUNS])
    # Empty registry → no resolution; the run rows preserve a uuid handle, but an
    # unresolved *name* handle is filtered by the jsonb fallback.
    client = _FakeClient(q, registry={})
    repo = MonitoringRunRepository(client)

    await repo.get_recent_runs(model_version="totally_unregistered_model", limit=50)

    assert not [op for op in q.ops if op[1] == "model_id"]
    jsonb_ops = [op for op in q.ops if op[0] == "eq" and _MV_KEY in op[1]]
    assert jsonb_ops, f"expected a config->>{_MV_KEY} fallback filter, got {q.ops!r}"
