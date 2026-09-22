"""#2207 follow-up (codex r1 HIGH-2): the registry row's cohort contract is healed ONLY
by a contract that just produced a promotable model — in ``_execute_real_retraining``
after ``complete_retraining`` — never at trigger time, and only as a consistent unit.

The history repository and the registry read/write run over the in-memory async
supabase fake; the pipeline and the trigger service are patched as in
``test_execute_model_retraining_real.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.tasks.drift_monitoring_tasks import _execute_real_retraining
from tests.unit._fakes.async_supabase import FakeAsyncSupabase

MODEL = "initiation_kisqali_goldstd_lr_v1"
CONTRACT: Dict[str, Any] = {
    "data_source": "patient_journeys",
    "target_outcome": "treatment_initiated",
    "feature_manifest_source": "synthetic",
    "problem_description": "p",
    "business_objective": "b",
}
GOOD = SimpleNamespace(
    status="completed",
    training_result={"validation_metrics": {"roc_auc": 0.71}, "success_criteria_met": True},
    deployment_result={"model_version": "v2"},
)
BAD = SimpleNamespace(status="failed", training_result={}, deployment_result={})


def _db(**cols: Any) -> tuple[FakeAsyncSupabase, str]:
    rid = str(uuid4())
    row = {
        "id": rid,
        "model_name": MODEL,
        "model_version": "1.0",
        "is_synthetic": False,
        "cohort_data_source": None,
        "cohort_target_outcome": None,
        "cohort_feature_manifest_source": None,
    }
    row.update(cols)
    history = {"id": "rt-1", "status": "pending", "old_model_version": MODEL, "model_id": rid}
    return FakeAsyncSupabase({"ml_model_registry": [row], "ml_retraining_history": [history]}), rid


async def _run(db: FakeAsyncSupabase, result: Any, training_config: Dict[str, Any]):
    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value=result)
    service = MagicMock()
    service.complete_retraining = AsyncMock()
    with (
        patch("src.agents.tier_0.pipeline.MLFoundationPipeline", MagicMock(return_value=pipeline)),
        patch(
            "src.repositories.drift_monitoring.get_drift_monitoring_client",
            AsyncMock(return_value=db),
        ),
        patch(
            "src.services.retraining_trigger.get_retraining_trigger_service", return_value=service
        ),
    ):
        return await _execute_real_retraining("rt-1", MODEL, "v2", dict(training_config)), service


@pytest.mark.unit
@pytest.mark.asyncio
async def test_completed_retrain_heals_null_columns():
    db, _ = _db(cohort_target_outcome="treatment_initiated")  # consistent with the contract
    out, service = await _run(db, GOOD, CONTRACT)
    assert out["status"] == "completed"
    service.complete_retraining.assert_awaited_once()
    (row,) = db.rows("ml_model_registry")
    assert row["cohort_data_source"] == "patient_journeys"
    assert row["cohort_feature_manifest_source"] == "synthetic"
    assert row["cohort_target_outcome"] == "treatment_initiated"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conflicting_row_value_blocks_the_whole_heal():
    """The migration-150 backfill put the experiment LABEL (initiation_kisqali) on the
    row; a retrain on the real column (treatment_initiated) must not compose the pair
    {patient_journeys, initiation_kisqali} that nobody ever ran."""
    db, _ = _db(cohort_target_outcome="initiation_kisqali")
    out, _ = await _run(db, GOOD, CONTRACT)
    assert out["status"] == "completed"  # the retrain itself is unaffected
    (row,) = db.rows("ml_model_registry")
    assert row["cohort_data_source"] is None
    assert row["cohort_feature_manifest_source"] is None
    assert row["cohort_target_outcome"] == "initiation_kisqali"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failed_retrain_heals_nothing():
    db, _ = _db()
    out, service = await _run(db, BAD, CONTRACT)
    assert out["status"] == "failed"
    service.complete_retraining.assert_not_awaited()
    (row,) = db.rows("ml_model_registry")
    assert row["cohort_data_source"] is None and row["cohort_target_outcome"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unregistered_model_completes_without_healing():
    db = FakeAsyncSupabase(
        {"ml_model_registry": [], "ml_retraining_history": [{"id": "rt-1", "status": "pending"}]}
    )
    out, _ = await _run(db, GOOD, CONTRACT)
    assert out["status"] == "completed"
    assert db.rows("ml_model_registry") == []
