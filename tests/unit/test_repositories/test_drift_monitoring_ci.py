"""RED-first unit tests for B2 (goldstd slope uncertainty) — ci_lower/ci_upper
passthrough on PerformanceMetricRecord / PerformanceMetricRepository.

The ml_performance_metrics table ALREADY has nullable ``ci_lower`` /
``ci_upper`` DECIMAL(12,6) columns (database/ml/017_model_monitoring_tables.sql)
plus ``ci_level`` defaulting to 0.95 — NO migration involved. This file proves
the new optional fields flow through to_db_row() only when set, and that
``record_metrics(cis=...)`` attaches the CI to the matching metric row ONLY,
without breaking callers that omit them.
"""

import asyncio
import datetime as dt
from unittest.mock import AsyncMock, MagicMock

from src.repositories.drift_monitoring import (
    PerformanceMetricRecord,
    PerformanceMetricRepository,
)

# ---------------------------------------------------------------------------
# Pure to_db_row tests (no DB)
# ---------------------------------------------------------------------------


def test_record_ci_fields_flow_to_db_row():
    """ci_lower/ci_upper set on the record must be emitted in the DB row."""
    rec = PerformanceMetricRecord(
        model_id="m1",
        metric_name="calibration_slope",
        metric_value=1.4455,
        sample_size=415,
        source="holdout",
        ci_lower=1.22,
        ci_upper=1.67,
    )
    row = rec.to_db_row()
    assert row["ci_lower"] == 1.22
    assert row["ci_upper"] == 1.67
    assert row["sample_size"] == 415


def test_record_ci_none_omits_keys():
    """When no CI is set, the keys must be ABSENT so the columns stay NULL
    (pre-B2 rows and non-slope metrics are unchanged)."""
    rec = PerformanceMetricRecord(
        metric_name="auc_roc",
        metric_value=0.83,
        source="holdout",
    )
    row = rec.to_db_row()
    assert "ci_lower" not in row
    assert "ci_upper" not in row


# ---------------------------------------------------------------------------
# record_metrics(cis=...) keyword-argument passthrough (async, no DB)
# ---------------------------------------------------------------------------


def _make_repo() -> PerformanceMetricRepository:
    """Return a repo with a mocked async client that captures insert calls."""
    client = MagicMock()
    fake_result = MagicMock()
    fake_result.data = []
    execute_mock = AsyncMock(return_value=fake_result)
    insert_mock = MagicMock()
    insert_mock.execute = execute_mock
    table_mock = MagicMock()
    table_mock.insert = MagicMock(return_value=insert_mock)
    client.table = MagicMock(return_value=table_mock)
    return PerformanceMetricRepository(client)


def test_record_metrics_cis_attaches_to_matching_metric_only():
    """cis={'calibration_slope': (lo, hi)} sets the CI on the calibration_slope
    record and leaves every other metric's CI unset."""
    repo = _make_repo()
    now = dt.datetime(2026, 7, 21, tzinfo=dt.timezone.utc)

    records = asyncio.run(
        repo.record_metrics(
            model_version="m1",
            metrics={"calibration_slope": 1.4455, "auc_roc": 0.66},
            sample_size=415,
            window_start=now,
            window_end=now,
            measured_at=now,
            source="holdout",
            cis={"calibration_slope": (1.22, 1.67)},
        )
    )
    by_name = {r.metric_name: r for r in records}
    slope = by_name["calibration_slope"]
    assert slope.ci_lower == 1.22
    assert slope.ci_upper == 1.67
    auc = by_name["auc_roc"]
    assert auc.ci_lower is None
    assert auc.ci_upper is None


def test_record_metrics_no_cis_backward_compat():
    """Existing callers that omit cis= continue to work with NULL CI columns."""
    repo = _make_repo()
    now = dt.datetime(2026, 7, 21, tzinfo=dt.timezone.utc)

    records = asyncio.run(
        repo.record_metrics(
            model_version="m2",
            metrics={"auc_roc": 0.75},
            sample_size=100,
            window_start=now,
            window_end=now,
        )
    )
    assert len(records) == 1
    assert records[0].ci_lower is None
    assert records[0].ci_upper is None
    row = records[0].to_db_row()
    assert "ci_lower" not in row and "ci_upper" not in row
