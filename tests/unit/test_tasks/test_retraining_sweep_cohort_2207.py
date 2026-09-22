"""#2207 follow-up (owner decision 2026-09-22): the daily retraining sweep reads the
per-model cohort contract off the registry row and passes it through to the trigger.

``check_retraining_for_all_models`` -> connector projection (id, model_name, ..., the
three migration-150 columns) -> ``evaluate_retraining_need.delay(cohort=...)`` ->
``evaluate_and_trigger_retraining(cohort=...)``. The existing guard
(``has_cohort_contract``) then passes for contracted models and still blocks — same
reason string — for the rest. The connector is faked at its seam; the task functions
are the real ones.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.drift_monitor.connectors.supabase_connector import SupabaseDataConnector
from src.tasks import drift_monitoring_tasks


class _Query:
    """Records the projection; 42703s when asked for the 150 columns if configured."""

    def __init__(self, rows: List[Dict[str, Any]], missing_columns: bool = False):
        self.rows = rows
        self.missing_columns = missing_columns
        self.selects: List[str] = []
        self.eq_calls: List[tuple] = []

    def table(self, _name):
        return self

    def select(self, cols):
        self.selects.append(cols)
        self._current = cols
        return self

    def eq(self, col, val):
        self.eq_calls.append((col, val))
        return self

    def in_(self, *_a):
        return self

    def order(self, *_a, **_k):
        return self

    def execute(self):
        if self.missing_columns and "cohort_data_source" in self._current:
            raise RuntimeError("column ml_model_registry.cohort_data_source does not exist (42703)")
        return MagicMock(data=self.rows)


def _connector(rows, missing_columns=False):
    conn = SupabaseDataConnector.__new__(SupabaseDataConnector)
    q = _Query(rows, missing_columns)
    conn._client = q
    conn._initialized = True
    return conn, q


ROWS = [
    {
        "id": "id-1",
        "model_name": "initiation_kisqali_goldstd_lr_v1",
        "model_version": "1.0",
        "stage": "staging",
        "registered_at": "2026-09-01T00:00:00+00:00",
        "cohort_data_source": "patient_journeys",
        "cohort_target_outcome": "treatment_initiated",
        "cohort_feature_manifest_source": None,
    },
    {
        "id": "id-2",
        "model_name": "persistence_kisqali_goldstd_lr_v1",
        "model_version": "1.0",
        "stage": "staging",
        "registered_at": "2026-09-01T00:00:00+00:00",
        "cohort_data_source": None,
        "cohort_target_outcome": "persistence_kisqali",
        "cohort_feature_manifest_source": None,
    },
]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_connector_projects_the_contract_columns_and_keeps_the_synthetic_guard():
    conn, q = _connector(ROWS)
    models = await conn.get_available_models(stages=["production", "staging"])
    assert len(models) == 2
    assert all(c in q.selects[0] for c in ("cohort_data_source", "cohort_target_outcome"))
    assert ("is_synthetic", False) in q.eq_calls


@pytest.mark.unit
@pytest.mark.asyncio
async def test_connector_falls_back_to_the_narrow_projection_before_migration_150():
    """The 6-hourly drift sweep shares this projection: a missing column must not blank it."""
    conn, q = _connector(ROWS, missing_columns=True)
    models = await conn.get_available_models(stages=["production", "staging"])
    assert len(models) == 2
    assert len(q.selects) == 2 and "cohort_data_source" not in q.selects[1]


@pytest.mark.unit
def test_sweep_passes_the_row_contract_to_the_evaluation_task():
    connector = MagicMock()
    connector.get_available_models = AsyncMock(return_value=ROWS)
    with (
        patch("src.agents.drift_monitor.connectors.get_connector", return_value=connector),
        patch.object(drift_monitoring_tasks, "evaluate_retraining_need") as task,
    ):
        task.delay = MagicMock(return_value=MagicMock(id="t"))
        out = drift_monitoring_tasks.check_retraining_for_all_models()
    assert out["tasks_queued"] == 2
    by_model = {c.kwargs["model_id"]: c.kwargs for c in task.delay.call_args_list}
    assert by_model["id-1"]["cohort"] == {
        "data_source": "patient_journeys",
        "target_outcome": "treatment_initiated",
    }
    assert by_model["id-2"]["cohort"] == {"target_outcome": "persistence_kisqali"}


@pytest.mark.unit
def test_sweep_decodes_a_json_file_source():
    files = {"type": "file_dir", "path": "data/rwd/optum/initiation"}
    row = {**ROWS[0], "cohort_data_source": json.dumps(files, sort_keys=True)}
    connector = MagicMock()
    connector.get_available_models = AsyncMock(return_value=[row])
    with (
        patch("src.agents.drift_monitor.connectors.get_connector", return_value=connector),
        patch.object(drift_monitoring_tasks, "evaluate_retraining_need") as task,
    ):
        task.delay = MagicMock(return_value=MagicMock(id="t"))
        drift_monitoring_tasks.check_retraining_for_all_models()
    assert task.delay.call_args.kwargs["cohort"]["data_source"] == files


@pytest.mark.unit
def test_evaluation_task_forwards_the_cohort_to_the_helper():
    seen: Dict[str, Any] = {}

    async def _helper(model_version, auto_approve=False, *, cohort=None):
        seen.update(model_version=model_version, auto_approve=auto_approve, cohort=cohort)
        return {"model_version": model_version, "retraining_triggered": False}

    cohort = {"data_source": "patient_journeys", "target_outcome": "treatment_initiated"}
    with patch("src.services.retraining_trigger.evaluate_and_trigger_retraining", _helper):
        out = drift_monitoring_tasks.evaluate_retraining_need(
            model_id="id-1", auto_approve=False, cohort=cohort
        )
    assert seen == {"model_version": "id-1", "auto_approve": False, "cohort": cohort}
    assert out["retraining_triggered"] is False


@pytest.mark.unit
def test_evaluation_task_without_cohort_is_unchanged():
    seen: Dict[str, Any] = {}

    async def _helper(model_version, auto_approve=False, *, cohort=None):
        seen.update(cohort=cohort)
        return {"model_version": model_version}

    with patch("src.services.retraining_trigger.evaluate_and_trigger_retraining", _helper):
        drift_monitoring_tasks.evaluate_retraining_need(model_id="id-2")
    assert seen == {"cohort": None}
