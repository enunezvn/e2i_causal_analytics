"""#2207 follow-up (owner decision 2026-09-22): ``RetrainingTriggerService.trigger_retraining``
is self-healing through the registry contract.

- a request that omits ``data_source`` / ``target_outcome`` falls back to the registry
  row's contract (migration 150 columns); explicit request values win;
- the trigger NEVER writes the registry row: healing happens only when the job has
  produced a promotable model (``execute_model_retraining`` on completion, see
  test_retraining_execute_heals_contract_2207.py — codex r1 HIGH-2);
- ``ml_retraining_history.model_id`` is set to the registry row's id (it was never
  written before: 0 rows carried it);
- a request with no contract anywhere used to be recorded and enqueued, then fail closed
  at execution; since #2335 the trigger refuses it (422, nothing recorded) because it
  declares no required features. A column-less contract is completed by the request's
  ``candidate_features``.

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
    rid, exp_id = str(uuid4()), str(uuid4())
    row: Dict[str, Any] = {
        "id": rid,
        # #2242: every live registry row has its experiment (1454/1454 on 2026-09-23);
        # the trigger refuses a registered row whose identity it cannot read.
        "experiment_id": exp_id,
        "model_name": MODEL_NAME,
        "model_version": "1.0",
        "is_synthetic": False,
        "cohort_data_source": None,
        "cohort_target_outcome": None,
        "cohort_feature_manifest_source": None,
    }
    row.update(contract_cols)
    experiment = {
        "id": exp_id,
        "experiment_name": "initiation_kisqali_goldstd_eval_v1",
        "prediction_target": "initiation_kisqali",
    }
    store = {
        "ml_model_registry": [row],
        "ml_experiments": [experiment],
        "ml_retraining_history": [],
    }
    return FakeAsyncSupabase(store), rid


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
    # #2335: a bare table name declares no columns, so the request declares the features.
    job, queued = await _trigger(db, cohort={"candidate_features": ["disease_severity"]})
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
        db,
        cohort={
            "data_source": "data/rwd/optum/initiation",
            "brand": "Kisqali",
            "candidate_features": ["age_at_index"],  # #2335: file route declares none
        },
    )
    tc = queued["training_config"]
    assert tc["data_source"] == "data/rwd/optum/initiation"  # explicit
    assert tc["target_outcome"] == "treatment_initiated"  # filled from the row
    assert tc["brand"] == "Kisqali"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_trigger_never_writes_the_registry_row():
    """codex r1 HIGH-2: a contract persisted at trigger time could heal wrongly (the job
    has not run yet); healing is the completed job's business."""
    db, rid = _db(cohort_target_outcome="initiation_kisqali")  # set by hand; source NULL
    contract = {
        "data_source": "patient_journeys",
        "target_outcome": "treatment_initiated",
        "feature_manifest_source": "synthetic",
        "candidate_features": ["disease_severity"],  # #2335: declared requirement
    }
    _, queued = await _trigger(db, cohort=contract)
    assert queued["training_config"]["data_source"] == "patient_journeys"  # the job gets it
    (row,) = db.rows("ml_model_registry")
    assert row["cohort_data_source"] is None
    assert row["cohort_feature_manifest_source"] is None
    assert row["cohort_target_outcome"] == "initiation_kisqali"
    (history,) = db.rows("ml_retraining_history")
    assert history["model_id"] == rid


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_contract_anywhere_is_refused_before_anything_is_recorded():
    """#2335: this job used to be recorded ``pending`` and enqueued only to fail closed at
    execution. It declares no required features, so the trigger now refuses it."""
    from src.services.retraining_trigger import RetrainRefusedError

    db, _ = _db()
    with pytest.raises(RetrainRefusedError) as exc:
        await _trigger(db, cohort=None)
    assert exc.value.reason == "undeclared_required_features"
    assert exc.value.http_status == 422
    assert db.rows("ml_retraining_history") == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unregistered_model_handle_is_refused_without_a_history_row():
    """#2319 item 2: before, an unregistered handle wrote a history row with a NULL
    ``model_id`` and enqueued a job that could not attach its candidate to any registered
    model. The trigger now refuses it (no row, no task)."""
    from src.services.retraining_trigger import RetrainRefusedError

    db, _ = _db()
    with pytest.raises(RetrainRefusedError, match="ghost_v9"):
        await _trigger(db, cohort={"data_source": "t", "target_outcome": "y"}, handle="ghost_v9")
    assert db.rows("ml_retraining_history") == []
