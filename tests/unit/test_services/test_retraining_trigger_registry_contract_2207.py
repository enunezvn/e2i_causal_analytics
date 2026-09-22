"""#2207 follow-up (owner decision 2026-09-22): ``RetrainingTriggerService.trigger_retraining``
is self-healing through the registry contract.

- a request that omits ``data_source`` / ``target_outcome`` falls back to the registry
  row's contract (migration 150 columns); explicit request values win;
- a complete contract (from either source) is persisted onto the registry row ONCE —
  only the columns that are NULL;
- ``ml_retraining_history.model_id`` is set to the registry row's id (it was never
  written before: 0 rows carried it);
- a request with no contract anywhere behaves exactly as today (no data_source in the
  training config, ``model_id`` NULL, the job fails closed at execution).

The service's repositories are the real facades over an in-memory async supabase fake
(``tests/unit/_fakes/async_supabase.py``); only the drift-history read, the performance
tracker and the Celery ``.delay`` are patched, exactly like the Phase-D test.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.services.retraining_trigger import RetrainingTriggerService, TriggerReason
from tests.unit._fakes.async_supabase import FakeAsyncSupabase

MODEL_NAME = "initiation_kisqali_goldstd_lr_v1"


def _db(**contract_cols: Any) -> tuple[FakeAsyncSupabase, str]:
    rid = str(uuid4())
    row: Dict[str, Any] = {
        "id": rid,
        "model_name": MODEL_NAME,
        "model_version": "1.0",
        "is_synthetic": False,
        "cohort_data_source": None,
        "cohort_target_outcome": None,
        "cohort_feature_manifest_source": None,
    }
    row.update(contract_cols)
    return FakeAsyncSupabase({"ml_model_registry": [row], "ml_retraining_history": []}), rid


async def _trigger(
    db: FakeAsyncSupabase, *, cohort: Dict[str, Any] | None, handle: str = MODEL_NAME
):
    service = RetrainingTriggerService()
    drift_repo = MagicMock()
    drift_repo.get_latest_drift_status = AsyncMock(return_value=[])
    tracker = MagicMock()
    tracker.get_performance_trend = AsyncMock(side_effect=Exception("no perf"))
    with (
        patch(
            "src.repositories.drift_monitoring.get_drift_monitoring_client",
            AsyncMock(return_value=db),
        ),
        patch("src.repositories.drift_monitoring.DriftHistoryRepository", return_value=drift_repo),
        patch("src.services.performance_tracking.get_performance_tracker", return_value=tracker),
        patch("src.tasks.drift_monitoring_tasks.execute_model_retraining") as mock_task,
    ):
        mock_task.delay = MagicMock(return_value=MagicMock(id="task-1"))
        job = await service.trigger_retraining(
            model_version=handle, reason=TriggerReason.MANUAL, cohort=cohort
        )
        queued = mock_task.delay.call_args.kwargs
    return job, queued


@pytest.mark.unit
@pytest.mark.asyncio
async def test_request_without_contract_falls_back_to_the_registry_row():
    db, rid = _db(
        cohort_data_source="patient_journeys",
        cohort_target_outcome="treatment_initiated",
        cohort_feature_manifest_source="synthetic",
    )
    job, queued = await _trigger(db, cohort=None)
    tc = queued["training_config"]
    assert tc["data_source"] == "patient_journeys"
    assert tc["target_outcome"] == "treatment_initiated"
    assert tc["feature_manifest_source"] == "synthetic"
    (history,) = db.rows("ml_retraining_history")
    assert history["id"] == job.job_id
    assert history["config"]["data_source"] == "patient_journeys"
    assert history["model_id"] == rid


@pytest.mark.unit
@pytest.mark.asyncio
async def test_explicit_request_values_win_over_the_registry_row():
    db, _ = _db(cohort_data_source="patient_journeys", cohort_target_outcome="treatment_initiated")
    _, queued = await _trigger(
        db, cohort={"data_source": "data/rwd/optum/initiation", "brand": "Kisqali"}
    )
    tc = queued["training_config"]
    assert tc["data_source"] == "data/rwd/optum/initiation"  # explicit
    assert tc["target_outcome"] == "treatment_initiated"  # filled from the row
    assert tc["brand"] == "Kisqali"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_complete_contract_is_persisted_onto_null_columns_once():
    db, rid = _db(cohort_target_outcome="initiation_kisqali")  # 150 backfill; source NULL
    contract = {
        "data_source": "patient_journeys",
        "target_outcome": "treatment_initiated",
        "feature_manifest_source": "synthetic",
    }
    await _trigger(db, cohort=contract)
    (row,) = db.rows("ml_model_registry")
    assert row["cohort_data_source"] == "patient_journeys"
    assert row["cohort_feature_manifest_source"] == "synthetic"
    assert row["cohort_target_outcome"] == "initiation_kisqali"  # NOT overwritten (was set)

    # a second trigger with a different explicit source does not rewrite the row
    await _trigger(db, cohort={"data_source": "other_table", "target_outcome": "x"})
    (row,) = db.rows("ml_model_registry")
    assert row["cohort_data_source"] == "patient_journeys"
    assert len(db.rows("ml_retraining_history")) == 2
    assert all(h["model_id"] == rid for h in db.rows("ml_retraining_history"))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_incomplete_contract_is_not_persisted():
    db, _ = _db()
    await _trigger(db, cohort={"data_source": "patient_journeys"})  # no target anywhere
    (row,) = db.rows("ml_model_registry")
    assert row["cohort_data_source"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_contract_anywhere_behaves_as_today():
    db, rid = _db()
    job, queued = await _trigger(db, cohort=None)
    tc = queued["training_config"]
    assert "data_source" not in tc and "target_outcome" not in tc
    (history,) = db.rows("ml_retraining_history")
    assert history["status"] == "pending"
    assert history["model_id"] == rid  # the id is still resolved and recorded
    assert job.job_id == history["id"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unregistered_model_handle_leaves_model_id_null():
    db, _ = _db()
    job, queued = await _trigger(
        db, cohort={"data_source": "t", "target_outcome": "y"}, handle="ghost_v9"
    )
    (history,) = db.rows("ml_retraining_history")
    assert history["model_id"] is None
    assert history["old_model_version"] == "ghost_v9"
    assert queued["training_config"]["data_source"] == "t"
