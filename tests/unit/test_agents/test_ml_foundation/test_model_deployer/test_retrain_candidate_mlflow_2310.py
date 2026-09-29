"""#2311 (codex r1 on PR #2315): a retrain never leaves an orphan MLflow version, and the
real ``runs:/`` registration path can become a candidate.

* A version an earlier delivery registered but never recorded (its database write failed
  after MLflow registered) is found by run and reused, not registered again.
* Two versions from one run are ambiguous: fail closed, nothing registered.
* A version THIS delivery created but could not link is tagged ``e2i.role=unlinked`` and
  named in the error: it is not left behind silently.
* The connector path reported stage "development", which no promotion path knows, so a
  ``runs:/`` retrain wrote its row and then failed before it was tagged.

Real MLflow on a sqlite store under tmp_path (never the tracking server; artifacts under
tmp_path, never ./mlruns); the registry over the in-memory async supabase fake.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from tests.unit.test_agents.test_ml_foundation.test_model_deployer.test_retrain_candidate_write_2310 import (  # noqa: E501
    MLFLOW_EXP,
    MODEL,
    _db,
    _row,
    local_mlflow,  # noqa: F401 (fixture)
)

pytestmark = pytest.mark.unit


def _mlflow_run(client, tmp_path) -> str:
    exp = client.create_experiment("lane_2310", artifact_location=(tmp_path / "art").as_uri())
    return client.create_run(exp).info.run_id


async def _register(db, retrain_of, run_id, register):
    from src.agents.ml_foundation.model_deployer.nodes import registry_manager
    from src.agents.ml_foundation.model_deployer.state import ModelDeployerState

    state = ModelDeployerState(
        audit_workflow_id=uuid4(),
        model_uri=f"runs:/{run_id}/model",
        experiment_id=MLFLOW_EXP,
        deployment_name=f"{MLFLOW_EXP}_deployment",
        validation_metrics={"roc_auc": 0.836},
        success_criteria_met=True,
        retrain_of=retrain_of,
    )
    with (
        patch.object(registry_manager, "_register_model_mlflow", register),
        patch.object(
            registry_manager, "_get_async_supabase_client_or_none", AsyncMock(return_value=db)
        ),
    ):
        return await registry_manager.register_model(state)


@pytest.mark.asyncio
async def test_a_version_an_earlier_delivery_registered_but_never_recorded_is_reused(
    local_mlflow,  # noqa: F811
    tmp_path,
):
    client, _ = local_mlflow
    db, retrain_of = _db()
    run_id = _mlflow_run(client, tmp_path)
    db.rows("ml_training_runs")[0]["mlflow_run_id"] = run_id
    earlier = int(client.create_model_version(MODEL, source="x", run_id=run_id).version)
    register = AsyncMock(side_effect=AssertionError("must not register a second version"))
    out = await _register(db, retrain_of, run_id, register)
    register.assert_not_awaited()
    assert out["registration_successful"] is True
    assert out["model_version"] == earlier
    assert _row(db, out["model_registry_id"])["mlflow_model_version"] == earlier


@pytest.mark.asyncio
async def test_two_versions_from_one_run_are_ambiguous_and_fail_closed(
    local_mlflow,  # noqa: F811
    tmp_path,
):
    client, _ = local_mlflow
    db, retrain_of = _db()
    run_id = _mlflow_run(client, tmp_path)
    for _ in range(2):
        client.create_model_version(MODEL, source="x", run_id=run_id)
    register = AsyncMock()
    out = await _register(db, retrain_of, run_id, register)
    register.assert_not_awaited()
    assert out["registration_successful"] is False
    assert "not determinable" in out["error"]
    assert len(db.rows("ml_model_registry")) == 1  # only the parent


@pytest.mark.asyncio
async def test_a_version_this_delivery_could_not_link_is_tagged_unlinked(
    local_mlflow,  # noqa: F811
    tmp_path,
):
    client, _ = local_mlflow
    db, retrain_of = _db()
    db.rows("ml_training_runs").clear()  # the row cannot be sourced: the DB write fails
    run_id = _mlflow_run(client, tmp_path)

    async def _create(uri, name):
        mv = client.create_model_version(name, source="x", run_id=run_id)
        return name, int(mv.version), "None"

    out = await _register(db, retrain_of, run_id, _create)
    assert out["registration_successful"] is False
    created = out["unlinked_mlflow_version"]
    assert created is not None and f"MLflow version {created}" in out["error"]
    mv = client.get_model_version(MODEL, str(created))
    assert mv.tags == {"e2i.role": "unlinked", "e2i.retrain_of": retrain_of["model_id"]}
    assert mv.current_stage == "None"


@pytest.mark.asyncio
async def test_a_real_runs_uri_registration_reports_stage_none_and_can_become_a_candidate(
    local_mlflow,  # noqa: F811
    tmp_path,
):
    import mlflow
    import mlflow.sklearn
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
        _register_model_mlflow,
        promote_stage,
        validate_promotion,
    )

    client, _ = local_mlflow
    run_id = _mlflow_run(client, tmp_path)
    model = LogisticRegression().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    with mlflow.start_run(run_id=run_id):
        mlflow.sklearn.log_model(model, name="model")
    name, version, stage = await _register_model_mlflow(f"runs:/{run_id}/model", MODEL)
    assert (name, stage) == (MODEL, "None")
    state = {
        "registered_model_name": MODEL,
        "model_version": version,
        "current_stage": stage,
        "target_environment": "candidate",
        "retrain_of": {"model_id": str(uuid4())},
        "validation_metrics": {},
    }
    state.update(await validate_promotion(state))
    assert state["promotion_allowed"] is True
    out = await promote_stage(state)
    assert out["promotion_successful"] is True
    assert client.get_model_version(MODEL, str(version)).current_stage == "None"
