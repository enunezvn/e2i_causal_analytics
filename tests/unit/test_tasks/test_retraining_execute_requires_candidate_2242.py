"""#2242 (codex r1 #1): a retrain completes only when its candidate is linked.

``_execute_real_retraining`` marks a retrain ``completed`` on a promotable pipeline
result. For a retrain that carries ``retrain_of`` the promised link is the
``ml_model_registry`` row (retrain_of.model_name, retrain_of.new_model_version); when
the deployer could not write it (registry failure, fail-closed provenance guard) the
job must be recorded ``failed`` — never ``completed`` with nothing to point at.

Fixture pattern of test_retraining_execute_heals_contract_2207.py: the history
repository and the registry read run over the in-memory async supabase fake; the
pipeline and the trigger service are patched.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.tasks.drift_monitoring_tasks import _execute_real_retraining
from tests.unit._fakes.async_supabase import FakeAsyncSupabase

pytestmark = pytest.mark.unit

MODEL = "initiation_kisqali_goldstd_lr_v1"
NEW_VERSION = "1.0_retrained_20260923_0632_ab12cd"
CANDIDATE_ID = str(uuid4())


def _result(model_registry_id: Any) -> SimpleNamespace:
    """A promotable pipeline result whose deployer reported ``model_registry_id``."""
    return SimpleNamespace(
        status="completed",
        training_result={"validation_metrics": {"roc_auc": 0.71}, "success_criteria_met": True},
        deployment_result={"model_version": "1", "model_registry_id": model_registry_id},
    )


GOOD = _result(CANDIDATE_ID)


def _db(with_candidate: bool) -> tuple[FakeAsyncSupabase, Dict[str, Any]]:
    parent_id, exp_id = str(uuid4()), str(uuid4())
    rows = [{"id": parent_id, "model_name": MODEL, "model_version": "1.0", "experiment_id": exp_id}]
    if with_candidate:
        rows.append(
            {
                "id": CANDIDATE_ID,
                "model_name": MODEL,
                "model_version": NEW_VERSION,
                "experiment_id": exp_id,
            }
        )
    history = {"id": "rt-1", "status": "pending", "model_id": parent_id}
    retrain_of = {
        "model_id": parent_id,
        "model_name": MODEL,
        "model_version": "1.0",
        "experiment_id": exp_id,
        "experiment_name": "initiation_kisqali_goldstd_eval_v1",
        "new_model_version": NEW_VERSION,
    }
    db = FakeAsyncSupabase({"ml_model_registry": rows, "ml_retraining_history": [history]})
    return db, {
        "data_source": "patient_journeys",
        "target_outcome": "treatment_initiated",
        "retrain_of": retrain_of,
    }


async def _run(db: FakeAsyncSupabase, training_config: Dict[str, Any], result: Any = GOOD):
    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value=result)
    service = MagicMock()
    service.complete_retraining = AsyncMock(return_value=None)
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
        out = await _execute_real_retraining("rt-1", MODEL, NEW_VERSION, training_config)
    return out, service


@pytest.mark.asyncio
async def test_retrain_without_a_registered_candidate_is_marked_failed():
    db, tc = _db(with_candidate=False)
    out, service = await _run(db, tc)
    assert out["status"] == "failed"
    assert NEW_VERSION in out["error"]
    service.complete_retraining.assert_not_awaited()
    (history,) = db.rows("ml_retraining_history")
    assert history["status"] == "failed"


@pytest.mark.asyncio
async def test_retrain_with_its_registered_candidate_completes():
    db, tc = _db(with_candidate=True)
    out, service = await _run(db, tc)
    assert out["status"] == "completed"
    service.complete_retraining.assert_awaited_once()


@pytest.mark.asyncio
async def test_codex_r2_a_row_this_run_did_not_register_does_not_complete_it():
    """Celery redelivery: the first execution wrote (MODEL, NEW_VERSION); the second's
    registry write is refused (run-provenance conflict) so its deployer reports no id.
    The pre-existing row must not complete the second execution with its metric."""
    db, tc = _db(with_candidate=True)
    out, service = await _run(db, tc, result=_result(None))
    assert out["status"] == "failed"
    service.complete_retraining.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_r2_a_reported_row_that_is_not_the_candidate_does_not_complete():
    db, tc = _db(with_candidate=True)
    parent_id = db.rows("ml_model_registry")[0]["id"]  # a real row, wrong version
    out, service = await _run(db, tc, result=_result(parent_id))
    assert out["status"] == "failed"
    service.complete_retraining.assert_not_awaited()
