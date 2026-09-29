"""#2310 / #2311 / #2308: what a COMPLETED retrain means, and what it reports.

A completed retrain requires all of the following, and every other outcome fails closed:
* the candidate row THIS run wrote: its id, name and version, and the retrained model's
  experiment;
* that row linked to the parent (``retrain_of_id``) and at stage ``'candidate'``;
* the exact MLflow version, recorded on the row and equal to what the deployer reports;
* the deployer's confirmed ml_deployments record for that row, kept as ``'registered'``
  with no endpoint;
* the deployer reporting success.

On completion the job records that deployment in ``ml_retraining_history.deployment_id``,
which was never written before. It returns the real MLflow version: on every earlier run it
returned None (job 836578bf returned None while MLflow created v4). Its notes say the
candidate was registered, not promoted, with no endpoint.

Fixture pattern of test_retraining_execute_requires_candidate_2242.py (history + registry +
deployments over the in-memory async supabase fake; the pipeline and the trigger service
patched).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from uuid import uuid4

import pytest

from tests.unit.test_tasks.test_retraining_execute_requires_candidate_2242 import (
    CANDIDATE_ID,
    DEPLOY_RECORD_ID,
    NEW_VERSION,
    _db,
    _run,
)

pytestmark = pytest.mark.unit

SKIPPED = "deployment_action='promote': register/promote only — no Bento packaged"


def _result(**deployment: Any) -> SimpleNamespace:
    return SimpleNamespace(
        status="completed",
        training_result={"validation_metrics": {"roc_auc": 0.836}, "success_criteria_met": True},
        deployment_result={
            "model_registry_id": CANDIDATE_ID,
            "deployment_successful": True,
            "mlflow_model_version": 4,
            "deployment_record_id": DEPLOY_RECORD_ID,
            "db_persisted": True,
            "deployment_skipped_reason": SKIPPED,
            **deployment,
        },
        errors=[],
    )


def _candidate(db) -> Dict[str, Any]:
    return next(r for r in db.rows("ml_model_registry") if r["id"] == CANDIDATE_ID)


def _deployment(db) -> Dict[str, Any]:
    return next(r for r in db.rows("ml_deployments") if r["id"] == DEPLOY_RECORD_ID)


async def _assert_failed(db, tc, result, *needles: str) -> None:
    out, service = await _run(db, tc, result=result)
    assert out["status"] == "failed"
    service.complete_retraining.assert_not_awaited()
    for needle in needles:
        assert needle in out["error"]
    (history,) = db.rows("ml_retraining_history")
    assert history["status"] == "failed"
    assert history.get("deployment_id") is None


@pytest.mark.asyncio
async def test_a_linked_candidate_completes_and_records_its_exact_ids():
    db, tc = _db(with_candidate=True)
    out, service = await _run(db, tc, result=_result())
    assert out["status"] == "completed"
    assert out["mlflow_model_version"] == 4
    assert out["deployment_record_id"] == DEPLOY_RECORD_ID
    assert out["deployed"] is False
    (history,) = db.rows("ml_retraining_history")
    assert history["deployment_id"] == DEPLOY_RECORD_ID
    notes = service.complete_retraining.await_args.kwargs["notes"]
    assert "registered as candidate, not promoted, no endpoint" in notes
    assert CANDIDATE_ID in notes and NEW_VERSION in notes and DEPLOY_RECORD_ID in notes
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
    await _assert_failed(db, tc, _result(), "not linked to the retrained model")


@pytest.mark.asyncio
async def test_an_identity_without_the_parent_id_does_not_complete():
    db, tc = _db(with_candidate=True)
    tc["retrain_of"].pop("model_id")
    await _assert_failed(db, tc, _result())


@pytest.mark.asyncio
async def test_a_deployer_that_did_not_report_success_does_not_complete():
    db, tc = _db(with_candidate=True)
    await _assert_failed(db, tc, _result(deployment_successful=False))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reported, recorded, needle",
    [
        (None, 4, "no MLflow version"),
        (4, None, "no MLflow version"),
        (5, 4, "row records MLflow v4, deployer v5"),
    ],
)
async def test_the_mlflow_version_must_be_recorded_and_match(reported, recorded, needle):
    db, tc = _db(with_candidate=True)
    _candidate(db)["mlflow_model_version"] = recorded
    await _assert_failed(db, tc, _result(mlflow_model_version=reported), needle)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment, mutate",
    [
        ({"deployment_record_id": None}, None),  # the agent reported no record
        ({"db_persisted": False}, None),  # the agent could not confirm the record
        ({"deployment_record_id": str(uuid4())}, None),  # a record that does not exist
        ({}, {"model_registry_id": str(uuid4())}),  # another row's record
        ({}, {"status": "active"}),  # the #2308 defect
        ({}, {"status": "pending"}),  # the status write never landed
        ({}, {"endpoint_url": "http://svc:3000/predict"}),
    ],
)
async def test_the_deploy_record_must_be_this_candidates_registered_record(deployment, mutate):
    db, tc = _db(with_candidate=True)
    if mutate:
        _deployment(db).update(mutate)
    await _assert_failed(db, tc, _result(**deployment), "registered but not completed")
