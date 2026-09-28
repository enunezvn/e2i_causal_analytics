"""The retrain job records what the deployer actually did (Part of #2157).

Live, 2026-09-28 (retrain 073c38eb): the deployer wrote ml_model_registry row c524db0f
and then raised on bento validation. The pipeline keeps a raised deployer's error in
``result.errors`` and leaves ``deployment_result`` None, so the job failed with
"no ml_model_registry row ... written by this run (deployer reported None)". That is
false: a row was written, and the real cause was the deployment failure. The job still
fails closed, but its reason must name that failure.

A completed retrain is register + promote only (no endpoint). Its history notes and
the task result must say so instead of reporting ``deployed``.

Fixture pattern of test_retraining_execute_requires_candidate_2242.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.test_tasks.test_retraining_execute_requires_candidate_2242 import (
    CANDIDATE_ID,
    NEW_VERSION,
    _db,
    _run,
)

pytestmark = pytest.mark.unit

DEPLOYER_ERROR = (
    "bento_validation_error: Bento validation failed: ['Bento not found: Invalid Tag "
    "tag=\"exp_kisq_al_x_deployment:v3'] (ml_model_registry row c524db0f-1df1-4f21-806f-"
    "3b857c5245b9 was written before this failure)"
)


def _raised_deployer_result() -> SimpleNamespace:
    return SimpleNamespace(
        status="completed",
        training_result={"validation_metrics": {"roc_auc": 0.836}, "success_criteria_met": True},
        deployment_result=None,
        errors=[
            {"stage": "model_deployment", "error": DEPLOYER_ERROR, "error_type": "DeploymentError"}
        ],
    )


@pytest.mark.asyncio
async def test_a_raised_deployer_fails_the_job_with_its_own_error():
    db, tc = _db(with_candidate=True)
    out, service = await _run(db, tc, result=_raised_deployer_result())
    assert out["status"] == "failed"
    service.complete_retraining.assert_not_awaited()
    assert "model_deployer failed" in out["error"]
    assert DEPLOYER_ERROR in out["error"]
    assert "no ml_model_registry row" not in out["error"]
    (history,) = db.rows("ml_retraining_history")
    assert history["status"] == "failed"
    assert DEPLOYER_ERROR in history["notes"]


def _promote_only_result(reason: Any) -> SimpleNamespace:
    return SimpleNamespace(
        status="completed",
        training_result={"validation_metrics": {"roc_auc": 0.836}, "success_criteria_met": True},
        deployment_result={
            "model_version": "3",
            "model_registry_id": CANDIDATE_ID,
            "deployment_successful": True,
            "deployment_skipped_reason": reason,
        },
        errors=[],
    )


@pytest.mark.asyncio
async def test_a_register_only_retrain_completes_and_says_no_endpoint_was_deployed():
    reason = "deployment_action='promote': registered and promoted only, no endpoint deployed"
    db, tc = _db(with_candidate=True)
    out, service = await _run(db, tc, result=_promote_only_result(reason))
    assert out["status"] == "completed"
    assert out["deployed"] is False
    assert out["deployment_skipped_reason"] == reason
    notes = service.complete_retraining.await_args.kwargs["notes"]
    assert reason in notes and NEW_VERSION in notes and CANDIDATE_ID in notes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment_result",
    [
        # MLflow stage transition failed: the graph returns normally, promotion_successful
        # False, and the agent reports status="failed" but still carries the row it wrote.
        {"status": "failed", "deployment_successful": False},
        # #2259: the registry gate refused the promotion in _store_to_database.
        {
            "status": "failed",
            "deployment_successful": False,
            "promotion_refused_reason": "training_provenance is NULL",
        },
    ],
)
async def test_codex_r1_a_normally_returned_failed_promotion_is_not_completed(deployment_result):
    """codex r1 HIGH: the linked row alone is not a completed retrain; the deployer
    must also report success."""
    db, tc = _db(with_candidate=True)
    result = _promote_only_result("deployment_action='promote': no endpoint")
    result.deployment_result.update(deployment_result)
    out, service = await _run(db, tc, result=result)
    assert out["status"] == "failed"
    service.complete_retraining.assert_not_awaited()
    assert "did not promote" in out["error"]
    assert CANDIDATE_ID in out["error"]
    if deployment_result.get("promotion_refused_reason"):
        assert "training_provenance is NULL" in out["error"]


@pytest.mark.asyncio
async def test_repository_completion_persists_the_notes():
    from src.repositories.drift_monitoring import RetrainingHistoryRepository

    db, _ = _db(with_candidate=True)
    repo = RetrainingHistoryRepository(db)
    await repo.complete_retraining("rt-1", 0.836, True, notes="no endpoint deployed")
    (history,) = db.rows("ml_retraining_history")
    assert history["status"] == "completed"
    assert history["notes"] == "no endpoint deployed"
