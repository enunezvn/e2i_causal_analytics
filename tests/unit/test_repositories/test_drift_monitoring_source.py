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


# ---------------------------------------------------------------------------
# record_curve() + get_latest_curve() — structured eval artifacts in metadata
# ---------------------------------------------------------------------------


def test_record_curve_stores_payload_in_metadata(monkeypatch):
    """record_curve writes ONE row: metric_name=kind, metric_value=value, metadata=payload."""
    import src.repositories.drift_monitoring as D

    async def _mid(client, model_version):
        return "mid"

    monkeypatch.setattr(D, "_resolve_model_id", _mid)

    repo = _make_repo()
    ts = dt.datetime(2026, 6, 10, tzinfo=dt.timezone.utc)
    rec = asyncio.run(
        repo.record_curve(
            "mv",
            "confusion_matrix",
            0.68,
            {"tn": 1, "fp": 2, "fn": 3, "tp": 4, "threshold": 0.5},
            500,
            ts,
            ts,
            measured_at=ts,
            source="holdout_curve",
        )
    )
    assert rec is not None
    assert rec.metric_name == "confusion_matrix"
    assert rec.metric_value == 0.68
    assert rec.metadata["tp"] == 4
    assert rec.source == "holdout_curve"
    # Payload survives serialisation into the row's metadata jsonb.
    row = rec.to_db_row()
    assert row["metadata"]["fn"] == 3
    assert row["metric_name"] == "confusion_matrix"


def test_record_curve_no_client_returns_none():
    repo = PerformanceMetricRepository(None)
    ts = dt.datetime(2026, 6, 10, tzinfo=dt.timezone.utc)
    rec = asyncio.run(repo.record_curve("mv", "roc_curve", 0.7, {"points": []}, 10, ts, ts))
    assert rec is None


def _make_read_repo(rows):
    """Repo whose select chain (eq/order/limit/execute) resolves to ``rows``."""
    client = MagicMock()
    chain = MagicMock()
    chain.select = MagicMock(return_value=chain)
    chain.eq = MagicMock(return_value=chain)
    chain.order = MagicMock(return_value=chain)
    chain.limit = MagicMock(return_value=chain)
    chain.execute = AsyncMock(return_value=MagicMock(data=rows))
    client.table = MagicMock(return_value=chain)
    return PerformanceMetricRepository(client)


def test_get_latest_curve_returns_latest_record(monkeypatch):
    import src.repositories.drift_monitoring as D

    async def _mid(client, model_version):
        return "mid"

    monkeypatch.setattr(D, "_resolve_model_id", _mid)

    row = {
        "model_id": "mid",
        "metric_name": "roc_curve",
        "metric_value": 0.67,
        "metadata": {"points": [{"fpr": 0.0, "tpr": 0.0, "threshold": 1.0}]},
        "measured_at": "2026-06-10T00:00:00+00:00",
    }
    repo = _make_read_repo([row])
    rec = asyncio.run(repo.get_latest_curve("mv", "roc_curve"))
    assert rec is not None
    assert rec.metric_name == "roc_curve"
    assert rec.metadata["points"][0]["fpr"] == 0.0


def test_get_latest_curve_none_when_empty(monkeypatch):
    import src.repositories.drift_monitoring as D

    async def _mid(client, model_version):
        return "mid"

    monkeypatch.setattr(D, "_resolve_model_id", _mid)

    repo = _make_read_repo([])
    rec = asyncio.run(repo.get_latest_curve("mv", "confusion_matrix"))
    assert rec is None


# ---------------------------------------------------------------------------
# get_metric_trend() — a TREND is a walk-forward time series. The point-in-time
# 'holdout' snapshot (the headline eval, recorded once) is NOT a trend point;
# mixing it in made current=holdout vs baseline=mean(walk-forward) — a
# cross-source comparison that fabricated recall/F1 degradation on the
# model-performance alert AND grafted a mislabeled point onto the Time Series
# chart. It must be excluded from the trend read.
# ---------------------------------------------------------------------------


def _make_trend_repo(rows):
    """Repo whose trend chain (select/eq/gte/order/limit/execute) resolves to rows."""
    client = MagicMock()
    chain = MagicMock()
    for m in ("select", "eq", "gte", "order", "limit"):
        setattr(chain, m, MagicMock(return_value=chain))
    chain.execute = AsyncMock(return_value=MagicMock(data=rows))
    client.table = MagicMock(return_value=chain)
    return PerformanceMetricRepository(client)


def _acc_row(value, measured_at, source):
    return {
        "model_id": "mid",
        "metric_name": "accuracy",
        "metric_value": value,
        "measured_at": measured_at,
        "source": source,
    }


def test_get_metric_trend_excludes_holdout_snapshot(monkeypatch):
    """The point-in-time 'holdout' row must NOT appear in the trend series."""
    import src.repositories.drift_monitoring as D

    async def _mid(client, model_version):
        return "mid"

    monkeypatch.setattr(D, "_resolve_model_id", _mid)

    rows = [
        _acc_row(0.7050, "2026-06-10T00:00:00+00:00", "holdout"),  # snapshot (latest)
        _acc_row(0.6929, "2026-06-01T00:00:00+00:00", "backtest_wf"),
        _acc_row(0.7065, "2026-05-01T00:00:00+00:00", "backtest_wf"),
    ]
    repo = _make_trend_repo(rows)
    recs = asyncio.run(repo.get_metric_trend("mv", "accuracy", days=365))

    assert len(recs) == 2, "holdout snapshot must be dropped from the trend"
    assert all(r.source != "holdout" for r in recs)
    # The holdout value (0.7050) must not be present as a trend point.
    assert all(abs(r.metric_value - 0.7050) > 1e-9 for r in recs)


def test_get_metric_trend_keeps_non_holdout_sources(monkeypatch):
    """backtest_wf / mlflow / null-source rows are all retained (only holdout drops)."""
    import src.repositories.drift_monitoring as D

    async def _mid(client, model_version):
        return "mid"

    monkeypatch.setattr(D, "_resolve_model_id", _mid)

    rows = [
        _acc_row(0.70, "2026-06-01T00:00:00+00:00", "backtest_wf"),
        _acc_row(0.72, "2026-05-01T00:00:00+00:00", "mlflow"),
        _acc_row(0.71, "2026-04-01T00:00:00+00:00", None),
    ]
    repo = _make_trend_repo(rows)
    recs = asyncio.run(repo.get_metric_trend("mv", "accuracy"))
    assert len(recs) == 3, "only the holdout snapshot is excluded; other sources stay"
