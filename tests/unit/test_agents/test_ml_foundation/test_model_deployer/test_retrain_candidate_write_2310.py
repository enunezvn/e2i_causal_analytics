"""#2310 / #2311 / #2308: a retrain registers a linked CANDIDATE, not a staging peer.

Before (live, 2026-09-28): retrain jobs 836578bf / 36cd579b wrote registry rows at
stage 'staging', the stage the gold-standard v1.0 reference rows use, with no link to
the row they retrain; MLflow moved each new version to Staging; the job reported
``mlflow_model_version=None`` although MLflow created v4; and the register-only run
recorded an 'active' ml_deployments row with no endpoint.

After:
* R1 lineage: the row carries ``retrain_of_id`` = the parent; every reuse path requires
  the same parent (NULL or another parent fails closed, never healed); the MLflow version
  is tagged ``e2i.retrain_of`` / ``e2i.role=candidate``.
* R2 role: the row is inserted at stage 'candidate'; MLflow keeps the version at stage
  None (never Staging).
* R4 exact ids: the row records the MLflow version the deployer registered; a retry
  reuses the version its row records instead of registering an orphan, and a row that
  records a different version is refused.
* R5: a register-only deploy keeps its ml_deployments row, at 'registered', not 'active'.

Real code over the in-memory async supabase fake (``tests/unit/_fakes/async_supabase``)
and a real MLflow on a sqlite store under tmp_path (never the tracking server). The
real-Postgres versions (enum, FK, trigger, PostgREST) are in
``tests/unit/test_database/learning_loop/test_retrain_candidate_write_realdb_2310.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from tests.unit._fakes.async_supabase import FakeAsyncSupabase

pytestmark = pytest.mark.unit

MODEL = "initiation_kisqali_goldstd_lr_v1"
NEW_VERSION = "1.0_retrained_20260928_1934_725a94"
MLFLOW_EXP = "exp_kisq_al_20260928193409_725a94"


def _db() -> Tuple[FakeAsyncSupabase, Dict[str, Any]]:
    exp_id, parent_id = str(uuid4()), str(uuid4())
    db = FakeAsyncSupabase(
        {
            "ml_experiments": [
                {
                    "id": exp_id,
                    "experiment_name": "initiation_kisqali_goldstd_eval_v1",
                    "mlflow_experiment_id": MLFLOW_EXP,
                    "is_synthetic": False,
                }
            ],
            "ml_model_registry": [
                {
                    "id": parent_id,
                    "experiment_id": exp_id,
                    "model_name": MODEL,
                    "model_version": "1.0",
                    "stage": "staging",
                    "is_synthetic": False,
                }
            ],
            "ml_training_runs": [
                {
                    "id": str(uuid4()),
                    "experiment_id": exp_id,
                    "run_name": "train",
                    "mlflow_run_id": "run-a",
                    "algorithm": "LogisticRegression",
                    "hyperparameters": {},
                    "status": "completed",
                    "is_synthetic": False,
                }
            ],
            "ml_deployments": [],
        }
    )
    retrain_of = {
        "model_id": parent_id,
        "model_name": MODEL,
        "model_version": "1.0",
        "experiment_id": exp_id,
        "experiment_name": "initiation_kisqali_goldstd_eval_v1",
        "new_model_version": NEW_VERSION,
    }
    return db, retrain_of


def _row(db: FakeAsyncSupabase, row_id: str) -> Dict[str, Any]:
    return next(r for r in db.rows("ml_model_registry") if r["id"] == row_id)


async def _register(
    db: FakeAsyncSupabase,
    retrain_of: Optional[Dict[str, Any]],
    *,
    run_id: str = "run-a",
    mlflow_version: int = 4,
    deployment_name: str = f"{MLFLOW_EXP}_deployment",
):
    from src.agents.ml_foundation.model_deployer.nodes import registry_manager
    from src.agents.ml_foundation.model_deployer.state import ModelDeployerState

    state = ModelDeployerState(
        audit_workflow_id=uuid4(),
        model_uri=f"runs:/{run_id}/model",
        experiment_id=MLFLOW_EXP,
        deployment_name=deployment_name,
        validation_metrics={"roc_auc": 0.836},
        success_criteria_met=True,
        retrain_of=retrain_of,
    )
    mlflow = AsyncMock(side_effect=lambda uri, name: (name, mlflow_version, "None"))
    with (
        patch.object(registry_manager, "_register_model_mlflow", mlflow),
        # #2311: MLflow is replaced here; it answers "no earlier version from this run".
        patch.object(registry_manager, "_model_versions_for_run", AsyncMock(return_value=[])),
        patch.object(
            registry_manager, "_get_async_supabase_client_or_none", AsyncMock(return_value=db)
        ),
    ):
        out = await registry_manager.register_model(state)
    return out, mlflow


# ------------------------------------------------------------------ R1 + R2 + R4: the row


@pytest.mark.asyncio
async def test_a_retrain_registers_a_candidate_linked_to_its_parent_with_the_mlflow_version():
    db, retrain_of = _db()
    out, _ = await _register(db, retrain_of, mlflow_version=4)
    assert out["registration_successful"] is True
    row = _row(db, out["model_registry_id"])
    assert row["model_name"] == MODEL and row["model_version"] == NEW_VERSION
    assert row["stage"] == "candidate"
    assert row["retrain_of_id"] == retrain_of["model_id"]
    assert row["mlflow_model_version"] == 4
    assert out["model_version"] == 4


@pytest.mark.asyncio
async def test_a_non_retrain_registration_keeps_development_and_records_its_mlflow_version():
    db, _ = _db()
    out, mlflow = await _register(db, None, mlflow_version=2)
    row = _row(db, out["model_registry_id"])
    assert row["stage"] == "development"
    assert row.get("retrain_of_id") is None
    assert row["mlflow_model_version"] == 2
    assert mlflow.await_args.args[1] == f"{MLFLOW_EXP}_deployment"


@pytest.mark.asyncio
async def test_a_retry_reuses_the_recorded_mlflow_version_instead_of_registering_an_orphan():
    """#2311: registration used to run before the DB idempotency check, so a redelivered
    job created a second MLflow version for the same run and left it unlinked."""
    db, retrain_of = _db()
    first, mlflow_1 = await _register(db, retrain_of, mlflow_version=4)
    again, mlflow_2 = await _register(db, retrain_of, mlflow_version=5)
    assert mlflow_1.await_count == 1
    mlflow_2.assert_not_awaited()
    assert again["registration_successful"] is True
    assert again["model_registry_id"] == first["model_registry_id"]
    assert again["model_version"] == 4
    assert len(db.rows("ml_model_registry")) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_parent", [None, "other"])
async def test_reuse_refuses_a_row_whose_lineage_is_not_the_requested_parent(existing_parent):
    """NULL or another parent fails closed BEFORE MLflow is touched; never healed."""
    db, retrain_of = _db()
    parent = str(uuid4()) if existing_parent == "other" else None
    squatter = {
        "id": str(uuid4()),
        "experiment_id": retrain_of["experiment_id"],
        "model_name": MODEL,
        "model_version": NEW_VERSION,
        "mlflow_run_id": "run-a",
        "stage": "staging",
        "retrain_of_id": parent,
        "mlflow_model_version": 4,
        "is_synthetic": False,
    }
    db.rows("ml_model_registry").append(squatter)
    out, mlflow = await _register(db, retrain_of)
    assert out["registration_successful"] is False
    assert out.get("model_registry_id") is None
    assert "lineage" in out["error"]
    mlflow.assert_not_awaited()
    assert _row(db, squatter["id"])["retrain_of_id"] == parent  # not healed


@pytest.mark.asyncio
async def test_the_writer_refuses_a_row_that_records_another_mlflow_version():
    """The unique-race reuse: a concurrent delivery wrote the row for MLflow v4; this
    delivery registered v5. Reusing would leave v5 an unlinked orphan: refuse."""
    from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
        _persist_model_registry_row,
    )

    db, retrain_of = _db()
    first, _ = await _register(db, retrain_of, mlflow_version=4)
    rid = await _persist_model_registry_row(
        db,
        experiment_id_str=MLFLOW_EXP,
        model_uri="runs:/run-a/model",
        registered_model_name=MODEL,
        model_version=5,
        validation_metrics=None,
        version_label=NEW_VERSION,
        expected_experiment_id=retrain_of["experiment_id"],
        retrain_of_id=retrain_of["model_id"],
    )
    assert rid is None
    assert _row(db, first["model_registry_id"])["mlflow_model_version"] == 4


@pytest.mark.asyncio
async def test_a_non_retrain_registration_never_adopts_a_candidate_row():
    from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
        _persist_model_registry_row,
    )

    db, retrain_of = _db()
    first, _ = await _register(db, retrain_of, mlflow_version=4)
    rid = await _persist_model_registry_row(
        db,
        experiment_id_str=MLFLOW_EXP,
        model_uri="runs:/run-a/model",
        registered_model_name=MODEL,
        model_version=4,
        validation_metrics=None,
        version_label=NEW_VERSION,
    )
    assert rid is None
    assert _row(db, first["model_registry_id"])["stage"] == "candidate"


@pytest.mark.asyncio
async def test_a_retrain_identity_without_the_parent_id_is_refused_before_mlflow():
    db, retrain_of = _db()
    retrain_of.pop("model_id")
    out, mlflow = await _register(db, retrain_of)
    assert out["registration_successful"] is False
    assert out["error_type"] == "incomplete_retrain_identity"
    mlflow.assert_not_awaited()


# ------------------------------------------------------ R2: promotion = role, not a stage


@pytest.mark.asyncio
async def test_the_candidate_environment_validates_to_the_candidate_target():
    from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
        validate_promotion,
    )

    out = await validate_promotion(
        {
            "current_stage": "None",
            "target_environment": "candidate",
            "retrain_of": {"model_id": str(uuid4())},
            "validation_metrics": {},
        }
    )
    assert out["promotion_allowed"] is True
    assert out["promotion_target_stage"] == "Candidate"


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """A real MLflow registry on a sqlite store under tmp_path (never the tracking server)."""
    from mlflow.tracking import MlflowClient

    from src.mlops.mlflow_connector import MLflowConnector

    import mlflow

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", uri)
    monkeypatch.setattr(MLflowConnector, "_instance", None)
    # mlflow's module-level URIs win over the environment once any earlier test in the
    # process set them (CI xdist worker: "Run ... not found" on another store). Pin both.
    saved = (mlflow.get_tracking_uri(), mlflow.get_registry_uri())
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_registered_model(MODEL)
    version = client.create_model_version(MODEL, source=str(tmp_path / "artifact")).version
    try:
        yield client, int(version)
    finally:
        mlflow.set_tracking_uri(saved[0])
        mlflow.set_registry_uri(saved[1])
        monkeypatch.setattr(MLflowConnector, "_instance", None)


@pytest.mark.asyncio
async def test_promoting_a_candidate_tags_the_version_and_never_moves_its_mlflow_stage(
    local_mlflow,
):
    from src.agents.ml_foundation.model_deployer.nodes.registry_manager import promote_stage

    client, version = local_mlflow
    parent = str(uuid4())
    out = await promote_stage(
        {
            "registered_model_name": MODEL,
            "model_version": version,
            "promotion_target_stage": "Candidate",
            "current_stage": "None",
            "retrain_of": {"model_id": parent},
            "validation_metrics": {},
        }
    )
    mv = client.get_model_version(MODEL, str(version))
    assert mv.current_stage == "None"
    assert mv.tags == {"e2i.role": "candidate", "e2i.retrain_of": parent}
    assert out["promotion_successful"] is True
    assert out["current_stage"] == "None"  # the MLflow stage; the role is the tag
    assert out["mlflow_transition_success"] is False
    assert "not promoted" in out["promotion_reason"]


@pytest.mark.asyncio
async def test_a_candidate_whose_version_cannot_be_tagged_is_not_a_success(local_mlflow):
    from src.agents.ml_foundation.model_deployer.nodes.registry_manager import promote_stage

    out = await promote_stage(
        {
            "registered_model_name": MODEL,
            "model_version": 99,  # no such version
            "promotion_target_stage": "Candidate",
            "current_stage": "None",
            "retrain_of": {"model_id": str(uuid4())},
            "validation_metrics": {},
        }
    )
    assert out["promotion_successful"] is False
    assert out["current_stage"] == "None"


@pytest.mark.asyncio
async def test_the_candidate_target_requires_a_retrain(local_mlflow):
    from src.agents.ml_foundation.model_deployer.nodes.registry_manager import promote_stage

    client, version = local_mlflow
    out = await promote_stage(
        {
            "registered_model_name": MODEL,
            "model_version": version,
            "promotion_target_stage": "Candidate",
            "current_stage": "None",
            "validation_metrics": {},
        }
    )
    assert out["promotion_successful"] is False
    assert client.get_model_version(MODEL, str(version)).tags == {}


# ------------------------------------------------------------- planner + agent invariants


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retrain, action, ok",
    [(True, "promote", True), (True, "register", True), (False, "promote", False),
     (True, "deploy", False)],
)  # fmt: skip
async def test_the_candidate_environment_is_register_only_and_retrain_only(retrain, action, ok):
    from src.agents.ml_foundation.model_deployer.nodes.deployment_planner import plan_deployment

    state = {
        "target_environment": "candidate",
        "deployment_action": action,
        "deployment_name": "d",
        "retrain_of": {"model_id": str(uuid4())} if retrain else None,
    }
    out = await plan_deployment(state)
    assert out.get("deployment_plan_created", False) is ok


def _agent_input(**extra: Any) -> Dict[str, Any]:
    return {
        "model_uri": "runs:/run-a/model",
        "experiment_id": MLFLOW_EXP,
        "validation_metrics": {"roc_auc": 0.8},
        "success_criteria_met": True,
        "deployment_name": "d",
        **extra,
    }


@pytest.mark.asyncio
async def test_the_agent_registers_every_retrain_in_the_candidate_environment():
    from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent

    agent = ModelDeployerAgent()
    seen: Dict[str, Any] = {}

    async def _ainvoke(state):
        seen.update(state)
        return {**state, "error": "stop here", "error_type": "test_stop"}

    with patch.object(agent.graph, "ainvoke", _ainvoke):
        with pytest.raises(RuntimeError, match="test_stop"):
            await agent.run(
                _agent_input(
                    retrain_of={"model_id": str(uuid4())},
                    deployment_action="promote",
                    target_environment="staging",
                )
            )
    assert seen["target_environment"] == "candidate"


@pytest.mark.asyncio
async def test_the_agent_refuses_a_retrain_with_an_endpoint_deploy_action():
    from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent

    with pytest.raises(ValueError, match="register-only"):
        await ModelDeployerAgent().run(
            _agent_input(retrain_of={"model_id": str(uuid4())}, deployment_action="deploy")
        )


# ------------------------------------------------------------------- R5: the deploy record


async def _store(db: FakeAsyncSupabase, state: Dict[str, Any]) -> Dict[str, Any]:
    from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent

    output: Dict[str, Any] = {"deployment_successful": True, "status": "completed"}
    with patch(
        "src.memory.services.factories.get_async_supabase_client", AsyncMock(return_value=db)
    ):
        await ModelDeployerAgent()._store_to_database(output, state)
    return output


@pytest.mark.asyncio
async def test_a_candidate_keeps_its_stage_and_its_deploy_record_is_registered_not_active():
    db, retrain_of = _db()
    first, _ = await _register(db, retrain_of, mlflow_version=4)
    rid = first["model_registry_id"]
    output = await _store(
        db,
        {
            "model_registry_id": rid,
            "deployment_name": "d",
            "target_environment": "candidate",
            "promotion_successful": True,
            "current_stage": "None",
            "promotion_target_stage": "Candidate",
            "deployment_action": "promote",
        },
    )
    assert output["db_persisted"] is True
    assert output["deployment_successful"] is True
    (dep,) = db.rows("ml_deployments")
    assert dep["status"] == "registered"
    assert dep["environment"] == "candidate"
    assert output["deployment_record_id"] == dep["id"]
    assert _row(db, rid)["stage"] == "candidate"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint_url, action, status",
    [(None, "promote", "registered"), ("http://svc:3000", "deploy", "active")],
)
async def test_only_a_deploy_with_an_endpoint_is_active(endpoint_url, action, status):
    db, _ = _db()
    first, _ = await _register(db, None, mlflow_version=2)
    await _store(
        db,
        {
            "model_registry_id": first["model_registry_id"],
            "deployment_name": "d",
            "target_environment": "staging",
            "promotion_successful": False,
            "current_stage": "None",
            "deployment_action": action,
            "endpoint_url": endpoint_url,
        },
    )
    (dep,) = db.rows("ml_deployments")
    assert dep["status"] == status
