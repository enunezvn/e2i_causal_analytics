"""#2310 / #2311: what a COMPLETED retrain means, and what it reports.

A completed retrain is the candidate row THIS run wrote -- its id + name + version + the
retrained model's experiment, linked to the parent (``retrain_of_id``) and at stage
``'candidate'`` -- plus the deployer reporting success. Every other outcome fails closed.
The job records the exact ml_deployments record (``ml_retraining_history.deployment_id``,
never written before), reports the MLflow version the deployer registered (it reported
None on every run: job 836578bf returned None while MLflow created v4), and says the
candidate was registered, not promoted, with no endpoint.

Fixture pattern of test_retraining_execute_requires_candidate_2242.py (history + registry
over the in-memory async supabase fake; the pipeline and the trigger service patched).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from uuid import uuid4

import pytest

from tests.unit.test_tasks.test_retraining_execute_requires_candidate_2242 import (
    CANDIDATE_ID,
    NEW_VERSION,
    _db,
    _run,
)

pytestmark = pytest.mark.unit

RECORD_ID = str(uuid4())
SKIPPED = "deployment_action='promote': register/promote only — no Bento packaged"


def _result(**deployment: Any) -> SimpleNamespace:
    return SimpleNamespace(
        status="completed",
        training_result={"validation_metrics": {"roc_auc": 0.836}, "success_criteria_met": True},
        deployment_result={
            "model_registry_id": CANDIDATE_ID,
            "deployment_successful": True,
            "mlflow_model_version": 4,
            "deployment_record_id": RECORD_ID,
            "deployment_skipped_reason": SKIPPED,
            **deployment,
        },
        errors=[],
    )


def _candidate(db) -> Dict[str, Any]:
    return next(r for r in db.rows("ml_model_registry") if r["id"] == CANDIDATE_ID)


@pytest.mark.asyncio
async def test_a_linked_candidate_completes_and_records_its_exact_ids():
    db, tc = _db(with_candidate=True)
    out, service = await _run(db, tc, result=_result())
    assert out["status"] == "completed"
    assert out["mlflow_model_version"] == 4
    assert out["deployment_record_id"] == RECORD_ID
    assert out["deployed"] is False
    (history,) = db.rows("ml_retraining_history")
    assert history["deployment_id"] == RECORD_ID
    notes = service.complete_retraining.await_args.kwargs["notes"]
    assert "registered as candidate, not promoted, no endpoint" in notes
    assert CANDIDATE_ID in notes and NEW_VERSION in notes and RECORD_ID in notes
    assert "MLflow version 4" in notes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field, value",
    [
        ("stage", "staging"),  # the pre-#2310 shape: a peer of the reference row
        ("stage", "development"),
        ("retrain_of_id", None),  # unlinked
        ("retrain_of_id", "other"),  # linked to another model
    ],
)
async def test_a_row_that_is_not_this_parents_candidate_does_not_complete(field, value):
    db, tc = _db(with_candidate=True)
    _candidate(db)[field] = str(uuid4()) if value == "other" else value
    out, service = await _run(db, tc, result=_result())
    assert out["status"] == "failed"
    service.complete_retraining.assert_not_awaited()
    assert "not linked to the retrained model" in out["error"]
    (history,) = db.rows("ml_retraining_history")
    assert history["status"] == "failed"
    assert history.get("deployment_id") is None


@pytest.mark.asyncio
async def test_an_identity_without_the_parent_id_does_not_complete():
    db, tc = _db(with_candidate=True)
    tc["retrain_of"].pop("model_id")
    out, service = await _run(db, tc, result=_result())
    assert out["status"] == "failed"
    service.complete_retraining.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_deployer_that_did_not_report_success_does_not_complete():
    db, tc = _db(with_candidate=True)
    out, service = await _run(db, tc, result=_result(deployment_successful=False))
    assert out["status"] == "failed"
    service.complete_retraining.assert_not_awaited()
