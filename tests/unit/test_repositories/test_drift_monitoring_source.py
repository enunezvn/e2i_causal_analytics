"""RED-first unit tests for Task 6 — source + measured_at passthrough on
PerformanceMetricRecord / PerformanceMetricRepository (additive, NO migration).

The ml_performance_metrics table already has a ``source`` column (default
'mlflow') and ``measured_at``; this file proves the new fields flow through
to_db_row() and that record_metrics() accepts them without breaking callers
that omit them.
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


def test_record_carries_source_and_measured_at():
    """T6: new source field flows through to_db_row; measured_at is preserved."""
    rec = PerformanceMetricRecord(
        model_id="m1",
        metric_name="auc_roc",
        metric_value=0.83,
        measured_at=dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc),
        source="backtest_wf",
        metadata={"split_version": "e2i_pilot_v3"},
    )
    row = rec.to_db_row()
    assert row["source"] == "backtest_wf"
    assert row["measured_at"].startswith("2026-05-01")
    assert row["metadata"]["split_version"] == "e2i_pilot_v3"


def test_record_source_none_omits_key():
    """When source is not set, the key must be absent so the DB default 'mlflow' applies."""
    rec = PerformanceMetricRecord(
        metric_name="auc_roc",
        metric_value=0.75,
    )
    row = rec.to_db_row()
    assert "source" not in row, "source=None should NOT emit a key (lets DB default 'mlflow' apply)"


def test_record_source_explicit_none_omits_key():
    """Explicitly passing source=None must also omit the key."""
    rec = PerformanceMetricRecord(
        metric_name="f1",
        metric_value=0.70,
        source=None,
    )
    row = rec.to_db_row()
    assert "source" not in row


def test_record_holdout_source():
    """Other source tags (e.g. 'holdout') pass through unchanged."""
    rec = PerformanceMetricRecord(
        metric_name="precision",
        metric_value=0.90,
        source="holdout",
    )
    row = rec.to_db_row()
    assert row["source"] == "holdout"


def test_record_measured_at_default_is_recent():
    """Without explicit measured_at, the default is close to now (not epoch/None)."""
    before = dt.datetime.now(dt.timezone.utc)
    rec = PerformanceMetricRecord(metric_name="auc_roc", metric_value=0.80)
    after = dt.datetime.now(dt.timezone.utc)
    row = rec.to_db_row()
    # measured_at is serialised as ISO string; parse it back
    ts = dt.datetime.fromisoformat(row["measured_at"])
    assert before <= ts <= after


# ---------------------------------------------------------------------------
# record_metrics() keyword-argument passthrough (async, no DB)
# ---------------------------------------------------------------------------


def _make_repo() -> PerformanceMetricRepository:
    """Return a repo with a mocked async client that captures insert calls."""
    client = MagicMock()
    # Chain: client.table().insert().execute() -> coroutine returning fake result
    fake_result = MagicMock()
    fake_result.data = []
    execute_mock = AsyncMock(return_value=fake_result)
    insert_mock = MagicMock()
    insert_mock.execute = execute_mock
    table_mock = MagicMock()
    table_mock.insert = MagicMock(return_value=insert_mock)
    client.table = MagicMock(return_value=table_mock)
    repo = PerformanceMetricRepository(client)
    return repo


def test_record_metrics_passes_source_and_measured_at():
    """record_metrics accepts source= and measured_at= and embeds them in every record."""
    repo = _make_repo()
    now = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)
    window_start = now
    window_end = now + dt.timedelta(days=30)

    records = asyncio.run(
        repo.record_metrics(
            model_version="m1",
            metrics={"auc_roc": 0.83, "f1": 0.78},
            sample_size=500,
            window_start=window_start,
            window_end=window_end,
            measured_at=now,
            source="backtest_wf",
        )
    )
    assert len(records) == 2
    for rec in records:
        assert rec.source == "backtest_wf"
        assert rec.measured_at == now


def test_record_metrics_backward_compat_no_source():
    """Existing callers that omit source= and measured_at= continue to work."""
    repo = _make_repo()
    window = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)

    records = asyncio.run(
        repo.record_metrics(
            model_version="m2",
            metrics={"auc_roc": 0.75},
            sample_size=100,
            window_start=window,
            window_end=window + dt.timedelta(days=7),
        )
    )
    assert len(records) == 1
    rec = records[0]
    assert rec.source is None
    # measured_at defaults to now (not the supplied window_start)
    assert rec.measured_at is not None


def test_record_metrics_measured_at_none_uses_default():
    """Passing measured_at=None explicitly should still use the now() default."""
    repo = _make_repo()
    window = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)
    before = dt.datetime.now(dt.timezone.utc)
    records = asyncio.run(
        repo.record_metrics(
            model_version="m3",
            metrics={"precision": 0.88},
            sample_size=200,
            window_start=window,
            window_end=window + dt.timedelta(days=7),
            measured_at=None,
        )
    )
    after = dt.datetime.now(dt.timezone.utc)
    assert len(records) == 1
    assert before <= records[0].measured_at <= after


# ---------------------------------------------------------------------------
# delete_metrics() unit test (no DB — pure argument-routing check)
# ---------------------------------------------------------------------------


def test_delete_metrics_no_client_returns_zero():
    """delete_metrics must return 0 when the client is absent (safe no-op)."""
    repo = PerformanceMetricRepository(None)
    result = asyncio.run(repo.delete_metrics(model_id="m1", source="backtest_wf"))
    assert result == 0


def test_delete_metrics_builds_correct_query():
    """delete_metrics issues a .delete().eq('model_id').eq('source') chain."""
    # We need to capture the chain so we can assert what was called.
    client = MagicMock()

    # Build a chain mock: .delete().eq().eq().execute()
    execute_mock = AsyncMock(return_value=MagicMock(data=["row1", "row2"]))

    eq2 = MagicMock()
    eq2.execute = execute_mock

    eq1 = MagicMock()
    eq1.eq = MagicMock(return_value=eq2)

    delete_mock = MagicMock()
    delete_mock.eq = MagicMock(return_value=eq1)

    table_mock = MagicMock()
    table_mock.delete = MagicMock(return_value=delete_mock)

    client.table = MagicMock(return_value=table_mock)

    repo = PerformanceMetricRepository(client)
    count = asyncio.run(repo.delete_metrics(model_id="m1", source="backtest_wf"))

    client.table.assert_called_once_with("ml_performance_metrics")
    table_mock.delete.assert_called_once()
    delete_mock.eq.assert_called_once_with("model_id", "m1")
    eq1.eq.assert_called_once_with("source", "backtest_wf")
    assert count == 2


def test_delete_metrics_with_split_version():
    """With split_version= supplied, a third .eq() on metadata->>split_version is chained."""
    client = MagicMock()

    execute_mock = AsyncMock(return_value=MagicMock(data=["row1"]))

    eq3 = MagicMock()
    eq3.execute = execute_mock

    eq2 = MagicMock()
    eq2.eq = MagicMock(return_value=eq3)

    eq1 = MagicMock()
    eq1.eq = MagicMock(return_value=eq2)

    delete_mock = MagicMock()
    delete_mock.eq = MagicMock(return_value=eq1)

    table_mock = MagicMock()
    table_mock.delete = MagicMock(return_value=delete_mock)

    client.table = MagicMock(return_value=table_mock)

    repo = PerformanceMetricRepository(client)
    count = asyncio.run(
        repo.delete_metrics(model_id="m1", source="backtest_wf", split_version="e2i_pilot_v3")
    )

    # Third eq should use the JSONB ->> form mirroring _apply_model_filter
    eq2.eq.assert_called_once_with("metadata->>split_version", "e2i_pilot_v3")
    assert count == 1
