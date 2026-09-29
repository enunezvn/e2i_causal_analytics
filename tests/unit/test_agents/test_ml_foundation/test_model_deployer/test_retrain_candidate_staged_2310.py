"""#2310 (codex r5 on PR #2315): a candidate's MLflow version must BE at stage None.

A reused version (the one its row records, or the one found by run) is not assumed to be
at stage None: live v3/v4/v5 of initiation_kisqali_goldstd_lr_v1 sit at Staging. The
candidate step reads the real stage and fails closed on anything else -- it never moves
a version -- and tags only a version that is at None.

Real MLflow on a sqlite store under tmp_path; the registry over the in-memory fake.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from tests.unit.test_agents.test_ml_foundation.test_model_deployer.test_retrain_candidate_write_2310 import (  # noqa: E501
    MLFLOW_EXP,
    MODEL,
    _db,
    local_mlflow,  # noqa: F401 (fixture)
)

pytestmark = pytest.mark.unit


def _run(client, tmp_path) -> str:
    exp = client.create_experiment("lane_2310_r5", artifact_location=(tmp_path / "a").as_uri())
    return client.create_run(exp).info.run_id


async def _register_then_promote(db, retrain_of, run_id):
    """The deployer's own register -> validate -> promote chain for a retrain."""
    from src.agents.ml_foundation.model_deployer.nodes import registry_manager
    from src.agents.ml_foundation.model_deployer.state import ModelDeployerState

    state = ModelDeployerState(
        audit_workflow_id=uuid4(),
        model_uri=f"runs:/{run_id}/model",
        experiment_id=MLFLOW_EXP,
        deployment_name="d",
        validation_metrics={},
        success_criteria_met=True,
        retrain_of=retrain_of,
        target_environment="candidate",
    ).model_dump()
    register = AsyncMock(side_effect=AssertionError("a reused version is not re-registered"))
    with (
        patch.object(registry_manager, "_register_model_mlflow", register),
        patch.object(
            registry_manager, "_get_async_supabase_client_or_none", AsyncMock(return_value=db)
        ),
    ):
        state.update(await registry_manager.register_model(state))
        state.update(await registry_manager.validate_promotion(state))
        return state, await registry_manager.promote_stage(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("reused_via", ["row", "run_search"])
@pytest.mark.parametrize("stage", ["None", "Staging"])
async def test_only_a_version_at_stage_none_becomes_a_candidate(
    local_mlflow,  # noqa: F811
    tmp_path,
    reused_via,
    stage,
):
    client, _ = local_mlflow
    db, retrain_of = _db()
    run_id = _run(client, tmp_path)
    db.rows("ml_training_runs")[0]["mlflow_run_id"] = run_id
    version = int(client.create_model_version(MODEL, source="x", run_id=run_id).version)
    if stage != "None":
        client.transition_model_version_stage(MODEL, str(version), stage)
    if reused_via == "row":
        db.rows("ml_model_registry").append(
            {
                "id": str(uuid4()),
                "experiment_id": retrain_of["experiment_id"],
                "model_name": MODEL,
                "model_version": retrain_of["new_model_version"],
                "mlflow_run_id": run_id,
                "stage": "candidate",
                "retrain_of_id": retrain_of["model_id"],
                "mlflow_model_version": version,
                "is_synthetic": False,
            }
        )

    state, out = await _register_then_promote(db, retrain_of, run_id)

    assert state["model_version"] == version
    mv = client.get_model_version(MODEL, str(version))
    assert mv.current_stage == stage  # never moved, either way
    if stage == "None":
        assert out["promotion_successful"] is True
        assert mv.tags["e2i.role"] == "candidate"
    else:
        assert out["promotion_successful"] is False
        assert out["current_stage"] == stage
        assert "not 'None'" in out["promotion_reason"]
        assert "e2i.role" not in mv.tags
