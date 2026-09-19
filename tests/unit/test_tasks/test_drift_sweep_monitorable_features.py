"""The drift sweep must monitor features that have data, and say what it compared.

Measured 2026-09-19. The ``features`` registry grew from 15 rows to 382
(2026-07-16 onward), and only the original 15 have any ``feature_values``.
``run_drift_detection`` took ``get_available_features()[:50]`` from an unordered
registry select, so none of the 15 were ever picked again: the last
drift-history row is from 2026-07-15, while every run since still recorded
``total_checks=50`` and 0 drift — a run that compared nothing, reported as a
healthy one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PATCHES = [
    "src.repositories.drift_monitoring.MonitoringRunRepository",
    "src.repositories.drift_monitoring.DriftHistoryRepository",
    "src.repositories.drift_monitoring.MonitoringAlertRepository",
    "src.agents.drift_monitor.nodes.alert_aggregator.AlertAggregatorNode",
    "src.agents.drift_monitor.nodes.concept_drift.ConceptDriftNode",
    "src.agents.drift_monitor.nodes.model_drift.ModelDriftNode",
    "src.agents.drift_monitor.nodes.data_drift.DataDriftNode",
    "src.agents.drift_monitor.connectors.get_connector",
    "src.repositories.drift_monitoring.get_drift_monitoring_client",
]


def _run(data_drift_update: Dict[str, Any], available: List[str], **kwargs: Any):
    from src.tasks.drift_monitoring_tasks import run_drift_detection

    mocks = {}
    patchers = [patch(target) for target in _PATCHES]
    try:
        for target, patcher in zip(_PATCHES, patchers, strict=True):
            mocks[target.rsplit(".", 1)[-1]] = patcher.start()
        connector = MagicMock()
        connector.get_available_features = AsyncMock(return_value=available)
        mocks["get_connector"].return_value = connector
        mocks["get_drift_monitoring_client"].return_value = MagicMock()
        seen: Dict[str, Any] = {}

        async def data_drift(state):
            seen["features"] = list(state["features_to_monitor"])
            return data_drift_update

        mocks["DataDriftNode"].return_value.execute = AsyncMock(side_effect=data_drift)
        mocks["ModelDriftNode"].return_value.execute = AsyncMock(
            return_value={"warnings": ["Model drift: no labeled predictions for model m1"]}
        )
        mocks["ConceptDriftNode"].return_value.execute = AsyncMock(return_value={})
        mocks["AlertAggregatorNode"].return_value.execute = AsyncMock(return_value={})
        run_repo = mocks["MonitoringRunRepository"].return_value
        run_repo.start_run = AsyncMock(return_value=MagicMock(id="run-1"))
        run_repo.complete_run = AsyncMock()
        mocks["DriftHistoryRepository"].return_value.record_drift_results = AsyncMock()
        alerts = mocks["MonitoringAlertRepository"].return_value
        alerts.create_alerts_from_drift = AsyncMock(return_value=[])
        alerts.auto_resolve_cleared = AsyncMock(return_value=0)

        result = run_drift_detection(model_id="m1", time_window="28d", **kwargs)
        seen["data_drift_node_call"] = mocks["DataDriftNode"].call_args
        return result, connector, run_repo.complete_run, seen
    finally:
        for patcher in patchers:
            patcher.stop()


def _drift_result(feature: str) -> Dict[str, Any]:
    return {
        "feature": feature,
        "drift_type": "data",
        "test_statistic": 0.1,
        "p_value": 0.5,
        "drift_detected": False,
        "severity": "none",
        "baseline_period": "b",
        "current_period": "c",
    }


def test_sweep_asks_for_features_with_values_across_both_windows() -> None:
    before = datetime.now(timezone.utc)
    _, connector, _, seen = _run({}, ["trx_30d", "nrx_30d"])

    kwargs = connector.get_available_features.await_args.kwargs
    since = kwargs["with_values_since"]
    # The data-drift node compares [now-2w, now-w] against [now-w, now]; the
    # baseline start is the earliest value either window can use.
    assert before - timedelta(days=56, minutes=1) <= since <= before - timedelta(days=55)
    assert seen["features"] == ["trx_30d", "nrx_30d"]


def test_explicit_features_skip_discovery() -> None:
    _, connector, _, seen = _run({}, ["unused"], features=["a", "b"])

    connector.get_available_features.assert_not_awaited()
    assert seen["features"] == ["a", "b"]


def test_run_records_features_compared_not_features_requested() -> None:
    update = {"data_drift_results": [_drift_result("trx_30d"), _drift_result("nrx_30d")]}
    _, _, complete_run, _ = _run(update, ["trx_30d", "nrx_30d", "no_data_feature"])

    kwargs = complete_run.await_args.kwargs
    assert kwargs["features_checked"] == 2
    summary = kwargs["summary"]
    assert summary["features_requested"] == 3
    assert summary["features_compared"] == 2
    assert "Model drift: no labeled predictions for model m1" in summary["warnings"]


def test_a_run_that_compared_nothing_says_so() -> None:
    _, _, complete_run, _ = _run({}, ["no_data_1", "no_data_2"])

    kwargs = complete_run.await_args.kwargs
    assert kwargs["features_checked"] == 0
    assert kwargs["summary"]["features_compared"] == 0
    assert kwargs["summary"]["features_requested"] == 2


@pytest.mark.asyncio
async def test_supabase_connector_filters_to_features_with_values() -> None:
    """Query shape only; the live check (15 of 382 returned, 0 for a future
    ``since``) is recorded in the PR. ``!inner`` is what drops value-less rows."""
    from src.agents.drift_monitor.connectors.supabase_connector import SupabaseDataConnector

    connector = SupabaseDataConnector.__new__(SupabaseDataConnector)
    connector._ensure_initialized = AsyncMock()
    query = MagicMock()
    for method in ("select", "eq", "gte", "limit"):
        getattr(query, method).return_value = query
    query.execute.return_value = MagicMock(data=[{"name": "trx_30d"}, {"name": "nrx_30d"}])
    connector._client = MagicMock()
    connector._client.table.return_value = query
    since = datetime(2026, 7, 25, tzinfo=timezone.utc)

    names = await connector.get_available_features(with_values_since=since)

    assert names == ["trx_30d", "nrx_30d"]
    assert "feature_values!inner" in query.select.call_args.args[0]
    query.gte.assert_any_call("feature_values.event_timestamp", since.isoformat())


def test_sweep_uses_the_calibrated_sample_floor_and_names_what_it_skipped() -> None:
    """The PSI/KS detector flags 99% of no-drift features at n=100 per window and
    3-7% from n=1000 (simulation, 2026-09-19); the live windows hold ~75 values
    per feature. The sweep must not alert on samples the detector cannot judge,
    and must say which features it did not compare."""
    from src.tasks.drift_monitoring_tasks import SWEEP_DATA_DRIFT_MIN_SAMPLES

    update = {"data_drift_results": [_drift_result("trx_30d")]}
    _, _, complete_run, seen = _run(update, ["trx_30d", "nrx_30d"])

    assert SWEEP_DATA_DRIFT_MIN_SAMPLES == 1000
    assert seen["data_drift_node_call"].kwargs["min_samples"] == SWEEP_DATA_DRIFT_MIN_SAMPLES
    summary = complete_run.await_args.kwargs["summary"]
    assert summary["min_samples_per_window"] == SWEEP_DATA_DRIFT_MIN_SAMPLES
    assert summary["features_not_compared"] == ["nrx_30d"]


def test_data_drift_node_floor_is_a_parameter_with_the_old_default() -> None:
    from src.agents.drift_monitor.nodes.data_drift import DataDriftNode

    assert DataDriftNode(connector=MagicMock())._min_samples == 30
    assert DataDriftNode(connector=MagicMock(), min_samples=1000)._min_samples == 1000


@pytest.mark.asyncio
async def test_below_the_floor_yields_no_verdict() -> None:
    import numpy as np

    from src.agents.drift_monitor.nodes.data_drift import DataDriftNode

    node = DataDriftNode(connector=MagicMock(), min_samples=1000)
    rng = np.random.default_rng(0)
    small = await node._detect_feature_drift(
        "f", rng.normal(0, 1, 75), rng.normal(0, 1, 75), 0.05, 0.1
    )
    large = await node._detect_feature_drift(
        "f", rng.normal(0, 1, 1500), rng.normal(0, 1, 1500), 0.05, 0.1
    )
    assert small is None
    assert large is not None
