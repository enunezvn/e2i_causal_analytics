"""#2311 (codex r2 on PR #2315): a new MLflow version needs a PROVEN absence of an earlier one.

* No run id, or MLflow unable to answer the version search: fail closed before registering
  (a retry must not register a second version on an unanswered question).
* A linked candidate row that records no MLflow version cannot say which version is its
  candidate: fail closed, never healed by guessing.

Real registry code over the in-memory async supabase fake.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from tests.unit.test_agents.test_ml_foundation.test_model_deployer.test_retrain_candidate_write_2310 import (  # noqa: E501
    MLFLOW_EXP,
    MODEL,
    _db,
)

pytestmark = pytest.mark.unit


async def _register(db, retrain_of, model_uri: str, find):
    from src.agents.ml_foundation.model_deployer.nodes import registry_manager
    from src.agents.ml_foundation.model_deployer.state import ModelDeployerState

    state = ModelDeployerState(
        audit_workflow_id=uuid4(),
        model_uri=model_uri,
        experiment_id=MLFLOW_EXP,
        deployment_name="d",
        validation_metrics={},
        success_criteria_met=True,
        retrain_of=retrain_of,
    )
    register = AsyncMock(return_value=(MODEL, 9, "None"))
    with (
        patch.object(registry_manager, "_register_model_mlflow", register),
        patch.object(registry_manager, "_model_versions_for_run", find),
        patch.object(
            registry_manager, "_get_async_supabase_client_or_none", AsyncMock(return_value=db)
        ),
    ):
        out = await registry_manager.register_model(state)
    return out, register


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["promote", "deploy"])
async def test_a_deploy_record_whose_status_write_did_not_land_is_not_a_success(action):
    """codex r3: a zero-row status update (RLS, a concurrent delete) must not leave a
    'registered' success reported over a record that still reads 'pending'."""
    from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent
    from src.repositories.deployment import MLDeploymentRepository

    db, retrain_of = _db()
    first, _ = await _register(db, retrain_of, "runs:/run-a/model", AsyncMock(return_value=[]))
    output = {"deployment_successful": True, "status": "completed"}

    async def _no_row(self, **_kw):
        return True  # the repository reports True whatever the update matched

    with (
        patch(
            "src.memory.services.factories.get_async_supabase_client", AsyncMock(return_value=db)
        ),
        patch.object(MLDeploymentRepository, "update_status", _no_row),
    ):
        await ModelDeployerAgent()._store_to_database(
            output,
            {
                "model_registry_id": first["model_registry_id"],
                "deployment_name": "d",
                "target_environment": "candidate" if action == "promote" else "staging",
                "promotion_successful": action == "promote",
                "current_stage": "None",
                "promotion_target_stage": "Candidate" if action == "promote" else None,
                "deployment_action": action,
                "endpoint_url": None if action == "promote" else "http://svc:3000/predict",
            },
        )
    assert output["db_persisted"] is False
    assert output["deployment_record_id"] is None
    assert output["deployment_successful"] is False
    assert output["status"] == ("failed" if action == "promote" else "partial")
    assert "pending" in output["db_persist_skipped_reason"]


@pytest.mark.asyncio
async def test_an_answered_empty_search_is_the_only_licence_to_register():
    db, retrain_of = _db()
    out, register = await _register(db, retrain_of, "runs:/run-a/model", AsyncMock(return_value=[]))
    register.assert_awaited_once()
    assert out["registration_successful"] is True


@pytest.mark.asyncio
async def test_an_unanswered_version_search_fails_closed_before_registering():
    db, retrain_of = _db()
    find = AsyncMock(return_value=None)
    out, register = await _register(db, retrain_of, "runs:/run-a/model", find)
    register.assert_not_awaited()
    assert out["registration_successful"] is False
    assert "cannot prove" in out["error"]


@pytest.mark.asyncio
async def test_a_retrain_without_a_run_id_fails_closed_before_registering():
    db, retrain_of = _db()
    find = AsyncMock(return_value=[])
    out, register = await _register(db, retrain_of, "models:/m-deadbeef", find)
    register.assert_not_awaited()
    find.assert_not_awaited()
    assert out["registration_successful"] is False
    assert "cannot prove" in out["error"]


@pytest.mark.asyncio
async def test_a_linked_row_without_an_mlflow_version_is_not_healed_by_guessing():
    db, retrain_of = _db()
    row = {
        "id": str(uuid4()),
        "experiment_id": retrain_of["experiment_id"],
        "model_name": MODEL,
        "model_version": retrain_of["new_model_version"],
        "mlflow_run_id": "run-a",
        "stage": "candidate",
        "retrain_of_id": retrain_of["model_id"],
        "mlflow_model_version": None,
        "is_synthetic": False,
    }
    db.rows("ml_model_registry").append(row)
    find = AsyncMock(return_value=[3])
    out, register = await _register(db, retrain_of, "runs:/run-a/model", find)
    register.assert_not_awaited()
    assert out["registration_successful"] is False
    assert "records no MLflow version" in out["error"]
    assert row["mlflow_model_version"] is None
