"""#2207 follow-up (owner decision 2026-09-22): the registry writer at training time
persists the per-model cohort contract (migration 150 columns).

``MLFoundationPipeline.run`` receives ``data_source`` / ``target_outcome`` /
``feature_manifest_source`` in ``input_data``; the deploy stage threads them into the
model_deployer's input, ``ModelDeployerAgent.run`` onto the initial state (declared
fields — LangGraph drops undeclared keys at channel boundaries), the ``register_model``
node into ``_persist_model_registry_row``, and ``MLModelRegistryRepository.register_model``
onto the row. A dict data_source is stored as JSON.

No mocks on the persistence path: the real repositories run over an in-memory async
supabase fake; only the MLflow registration call is patched (no MLflow here).
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.agents.ml_foundation.model_deployer.nodes import registry_manager
from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
    _persist_model_registry_row,
    register_model,
)
from src.agents.ml_foundation.model_deployer.state import ModelDeployerState
from src.repositories.ml_experiment import MLModelRegistry, MLModelRegistryRepository
from tests.unit._fakes.async_supabase import FakeAsyncSupabase

MLFLOW_EXP = "7"
RUN_ID = "abc123run"


def _db() -> tuple[FakeAsyncSupabase, str]:
    exp_id = str(uuid4())
    db = FakeAsyncSupabase(
        {
            "ml_experiments": [
                {
                    "id": exp_id,
                    "experiment_name": "initiation_kisqali_retrain",
                    "mlflow_experiment_id": MLFLOW_EXP,
                    "prediction_target": "treatment_initiated",
                    "is_synthetic": False,
                }
            ],
            "ml_training_runs": [
                {
                    "id": str(uuid4()),
                    "experiment_id": exp_id,
                    "run_name": "r",
                    "mlflow_run_id": RUN_ID,
                    "algorithm": "logistic_regression",
                    "hyperparameters": {"C": 1.0},
                    "status": "finished",
                    "test_metrics": {"auc": 0.8},
                    "is_synthetic": False,
                }
            ],
            "ml_model_registry": [],
        }
    )
    return db, exp_id


CONTRACT: Dict[str, Any] = {
    "data_source": "patient_journeys",
    "target_outcome": "treatment_initiated",
    "feature_manifest_source": "synthetic",
}


@pytest.mark.unit
def test_state_declares_the_cohort_fields():
    s = ModelDeployerState(
        audit_workflow_id=uuid4(),
        data_source={"type": "file_dir", "path": "x"},
        target_outcome="treatment_initiated",
    )
    assert s.data_source == {"type": "file_dir", "path": "x"}
    assert s.target_outcome == "treatment_initiated"
    assert ModelDeployerState(audit_workflow_id=uuid4()).data_source is None


@pytest.mark.unit
def test_registry_model_carries_and_emits_the_columns_only_when_set():
    m = MLModelRegistry(model_name="m", model_version="1", algorithm="lr")
    assert "cohort_data_source" not in m.to_dict()  # pre-150 schemas keep inserting
    m2 = MLModelRegistry(
        model_name="m",
        model_version="1",
        algorithm="lr",
        cohort_data_source="patient_journeys",
        cohort_target_outcome="treatment_initiated",
    )
    d = m2.to_dict()
    assert d["cohort_data_source"] == "patient_journeys"
    assert d["cohort_target_outcome"] == "treatment_initiated"
    assert "cohort_feature_manifest_source" not in d
    back = MLModelRegistry.from_dict({**d, "id": str(uuid4())})
    assert back.cohort_data_source == "patient_journeys"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_repository_register_model_writes_the_contract_json_for_dict_sources():
    db, exp_id = _db()
    repo = MLModelRegistryRepository(supabase_client=db)
    files = {"type": "file_dir", "path": "data/rwd/optum/initiation"}
    model = await repo.register_model(
        experiment_id=exp_id,
        model_name="m",
        model_version="1",
        mlflow_run_id=RUN_ID,
        mlflow_model_uri="runs:/abc/model",
        algorithm="lr",
        hyperparameters={},
        metrics={},
        cohort_data_source=files,
        cohort_target_outcome="initiated_biologic_180d",
        cohort_feature_manifest_source="optum",
    )
    (row,) = db.rows("ml_model_registry")
    assert row["cohort_data_source"] == json.dumps(files, sort_keys=True)
    assert row["cohort_target_outcome"] == "initiated_biologic_180d"
    assert row["cohort_feature_manifest_source"] == "optum"
    assert model.cohort_data_source == json.dumps(files, sort_keys=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_persist_row_threads_the_contract_and_heals_a_reused_row():
    db, exp_id = _db()
    rid = await _persist_model_registry_row(
        db,
        experiment_id_str=MLFLOW_EXP,
        model_uri=f"runs:/{RUN_ID}/model",
        registered_model_name="initiation_kisqali_retrain",
        model_version=1,
        validation_metrics={"auc_roc": 0.81},
        cohort=CONTRACT,
    )
    (row,) = db.rows("ml_model_registry")
    assert rid == row["id"]
    assert row["cohort_data_source"] == "patient_journeys"
    assert row["cohort_target_outcome"] == "treatment_initiated"
    assert row["cohort_feature_manifest_source"] == "synthetic"

    # idempotent re-deploy of the same name+version: the row is reused, and NULL
    # contract columns are healed (never overwritten)
    row["cohort_feature_manifest_source"] = None
    row["cohort_data_source"] = "kept_as_is"
    rid2 = await _persist_model_registry_row(
        db,
        experiment_id_str=MLFLOW_EXP,
        model_uri=f"runs:/{RUN_ID}/model",
        registered_model_name="initiation_kisqali_retrain",
        model_version=1,
        validation_metrics={"auc_roc": 0.81},
        cohort={**CONTRACT, "feature_manifest_source": "csu"},
    )
    assert rid2 == rid
    (row,) = db.rows("ml_model_registry")
    assert row["cohort_data_source"] == "kept_as_is"
    assert row["cohort_feature_manifest_source"] == "csu"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_model_node_hands_the_state_contract_to_the_repository():
    """A real deployer state (the shape ModelDeployerAgent.run builds) -> the row."""
    db, exp_id = _db()
    state = ModelDeployerState(
        audit_workflow_id=uuid4(),
        model_uri=f"runs:/{RUN_ID}/model",
        experiment_id=MLFLOW_EXP,
        deployment_name="initiation_kisqali_retrain",
        validation_metrics={"auc_roc": 0.81},
        success_criteria_met=True,
        data_source=CONTRACT["data_source"],
        target_outcome=CONTRACT["target_outcome"],
        scope_spec={"feature_manifest_source": "synthetic"},
    )
    with (
        patch.object(
            registry_manager,
            "_register_model_mlflow",
            AsyncMock(return_value=("initiation_kisqali_retrain", 1, "None")),
        ),
        patch.object(
            registry_manager, "_get_async_supabase_client_or_none", AsyncMock(return_value=db)
        ),
    ):
        out = await register_model(state)
    assert out["registration_successful"] is True
    (row,) = db.rows("ml_model_registry")
    assert out["model_registry_id"] == row["id"]
    assert row["cohort_data_source"] == "patient_journeys"
    assert row["cohort_target_outcome"] == "treatment_initiated"
    assert row["cohort_feature_manifest_source"] == "synthetic"  # from scope_spec


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_model_node_without_contract_writes_null_columns():
    db, _ = _db()
    state = ModelDeployerState(
        audit_workflow_id=uuid4(),
        model_uri=f"runs:/{RUN_ID}/model",
        experiment_id=MLFLOW_EXP,
        deployment_name="initiation_kisqali_retrain",
        validation_metrics={},
        success_criteria_met=True,
    )
    with (
        patch.object(
            registry_manager,
            "_register_model_mlflow",
            AsyncMock(return_value=("initiation_kisqali_retrain", 1, "None")),
        ),
        patch.object(
            registry_manager, "_get_async_supabase_client_or_none", AsyncMock(return_value=db)
        ),
    ):
        out = await register_model(state)
    (row,) = db.rows("ml_model_registry")
    assert out["model_registry_id"] == row["id"]
    assert "cohort_data_source" not in row  # nothing fabricated, nothing sent


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_threads_the_contract_onto_the_initial_state():
    from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent

    captured: Dict[str, Any] = {}

    class _Graph:
        async def ainvoke(self, initial_state, *_a, **_k):
            captured.update(dict(initial_state))
            return {**initial_state, "deployment_successful": False}

    agent = ModelDeployerAgent.__new__(ModelDeployerAgent)
    agent.agent_name = "model_deployer"
    agent.tier = 0
    agent.graph = _Graph()
    with patch(
        "src.agents.ml_foundation.model_deployer.agent._get_opik_connector", return_value=None
    ):
        try:
            await agent.run(
                {
                    "model_uri": "runs:/x/model",
                    "experiment_id": MLFLOW_EXP,
                    "validation_metrics": {},
                    "success_criteria_met": True,
                    "deployment_name": "d",
                    "data_source": {"type": "files", "paths": {}},
                    "target_outcome": "treatment_initiated",
                    "feature_manifest_source": "csu",
                }
            )
        except Exception:
            pass  # the stub graph's output need not satisfy the agent's post-processing
    assert captured["data_source"] == {"type": "files", "paths": {}}
    assert captured["target_outcome"] == "treatment_initiated"
    assert captured["feature_manifest_source"] == "csu"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pipeline_deploy_stage_passes_the_contract_to_the_deployer():
    from src.agents.tier_0.pipeline import MLFoundationPipeline, PipelineResult

    captured: Dict[str, Any] = {}

    class _Deployer:
        async def run(self, deployer_input):
            captured.update(deployer_input)
            return {"deployment_successful": False}

    from src.agents.tier_0.pipeline import PipelineConfig, PipelineStage

    pipeline = MLFoundationPipeline.__new__(MLFoundationPipeline)
    pipeline.config = PipelineConfig()
    pipeline._agents = {}
    result = PipelineResult(
        pipeline_run_id="run-1", status="running", current_stage=PipelineStage.MODEL_DEPLOYMENT
    )
    result.experiment_id = MLFLOW_EXP
    result.scope_spec = {"feature_manifest_source": "synthetic"}
    result.training_result = {"model_artifact_uri": "runs:/x/model", "validation_metrics": {}}
    with patch.object(MLFoundationPipeline, "_get_agent", return_value=_Deployer()):
        await pipeline._run_model_deployment(
            {
                "problem_description": "p",
                "business_objective": "b",
                "target_outcome": "treatment_initiated",
                "data_source": "patient_journeys",
                "feature_manifest_source": "synthetic",
            },
            result,
            None,
        )
    assert captured["data_source"] == "patient_journeys"
    assert captured["target_outcome"] == "treatment_initiated"
    assert captured["feature_manifest_source"] == "synthetic"
