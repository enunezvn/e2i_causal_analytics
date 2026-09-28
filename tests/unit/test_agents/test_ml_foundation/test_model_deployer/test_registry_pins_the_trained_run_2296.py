"""#2296 codex r1 HIGH: an MLflow 3 ``models:/m-<id>`` URI carries no run id, so the
registry writer fell back to the experiment's "best" completed run. Once runs are
finalised (this issue), a retrain's parent experiment holds several completed runs and
"best by AUC" can source a SIBLING run's algorithm / hyperparameters / mlflow_run_id for
THIS artifact — plausible-wrong provenance. The trainer knows its exact MLflow run; the
pipeline hands it to the deployer and the registry writer pins it (validated like a
``runs:/`` pin: it must exist and belong to the resolved experiment, else fail closed).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from tests.unit._fakes.async_supabase import FakeAsyncSupabase

EXP_KEY = "exp_kisq_al_1"


def _db() -> FakeAsyncSupabase:
    exp = str(uuid4())
    run = {"experiment_id": exp, "status": "completed", "is_synthetic": False}
    return FakeAsyncSupabase(
        {
            "ml_experiments": [
                {
                    "id": exp,
                    "experiment_name": "Kisqali - treatment_initiated",
                    "mlflow_experiment_id": EXP_KEY,
                    "prediction_target": "treatment_initiated",
                    "is_synthetic": False,
                }
            ],
            "ml_model_registry": [],
            "ml_training_runs": [
                # an earlier sibling retrain: HIGHER test AUC, different algorithm
                dict(
                    run,
                    id=str(uuid4()),
                    run_name="sibling",
                    mlflow_run_id="run-sibling",
                    algorithm="XGBoost",
                    hyperparameters={"max_depth": 3},
                    test_metrics={"roc_auc": 0.99},
                ),
                # the run that produced THIS artifact
                dict(
                    run,
                    id=str(uuid4()),
                    run_name="this",
                    mlflow_run_id="run-this",
                    algorithm="LogisticRegression",
                    hyperparameters={"C": 1.36},
                    test_metrics={"roc_auc": 0.835},
                ),
            ],
        }
    )


async def _register(
    db: FakeAsyncSupabase,
    mlflow_run_id: Optional[str],
    model_uri: str = "models:/m-64661e179ccb40058caed24f0b44fe1d",
) -> Dict[str, Any]:
    from src.agents.ml_foundation.model_deployer.nodes import registry_manager
    from src.agents.ml_foundation.model_deployer.state import ModelDeployerState

    state = ModelDeployerState(
        audit_workflow_id=uuid4(),
        model_uri=model_uri,
        mlflow_run_id=mlflow_run_id,
        experiment_id=EXP_KEY,
        deployment_name=f"{EXP_KEY}_deployment",
        validation_metrics={"roc_auc": 0.8375},
        success_criteria_met=True,
        target_outcome="treatment_initiated",
    )
    with (
        patch.object(
            registry_manager,
            "_register_model_mlflow",
            AsyncMock(side_effect=lambda uri, name: (name, 1, "None")),
        ),
        patch.object(
            registry_manager, "_get_async_supabase_client_or_none", AsyncMock(return_value=db)
        ),
    ):
        return await registry_manager.register_model(state)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_registry_row_is_sourced_from_the_run_that_trained_the_artifact():
    db = _db()
    out = await _register(db, "run-this")
    (row,) = db.rows("ml_model_registry")
    assert out["model_registry_id"] == row["id"]
    assert row["mlflow_run_id"] == "run-this"
    assert row["algorithm"] == "LogisticRegression"
    assert row["hyperparameters"] == {"C": 1.36}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_pinned_run_missing_from_the_table_fails_closed():
    db = _db()
    out = await _register(db, "run-never-persisted")
    assert out.get("model_registry_id") is None
    assert db.rows("ml_model_registry") == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_runs_uri_that_contradicts_the_trainers_run_fails_closed():
    """codex r2 MED: two provenance sources that disagree are never silently resolved."""
    db = _db()
    out = await _register(db, "run-sibling", model_uri="runs:/run-this/model")
    assert out.get("model_registry_id") is None
    assert db.rows("ml_model_registry") == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_runs_uri_that_agrees_with_the_trainers_run_registers():
    db = _db()
    out = await _register(db, "run-this", model_uri="runs:/run-this/model")
    assert out["model_registry_id"] and db.rows("ml_model_registry")[0]["mlflow_run_id"] == (
        "run-this"
    )


# ---------------------------------------------------------------------------
# the handoff: trainer output -> pipeline -> deployer state (a declared channel)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_pipeline_hands_the_deployer_the_trainers_mlflow_run():
    from src.agents.tier_0.pipeline import (
        MLFoundationPipeline,
        PipelineConfig,
        PipelineResult,
        PipelineStage,
    )

    pipeline = MLFoundationPipeline(config=PipelineConfig(skip_mlflow=True, enable_hpo=False))
    result = PipelineResult(
        pipeline_run_id="run", status="running", current_stage=PipelineStage.MODEL_DEPLOYMENT
    )
    result.experiment_id = EXP_KEY
    result.scope_spec = {"problem_type": "binary_classification"}
    result.training_result = {
        "model_artifact_uri": "models:/m-64661e17",
        "mlflow_run_id": "run-this",
        "validation_metrics": {"roc_auc": 0.8375},
        "success_criteria_met": True,
    }
    captured: Dict[str, Any] = {}
    fake = MagicMock()

    async def _run(deployer_input):
        captured.update(deployer_input)
        return {"deployment_successful": False}

    fake.run = AsyncMock(side_effect=_run)
    with patch.object(pipeline, "_get_agent", return_value=fake):
        await pipeline._run_model_deployment(
            {"data_source": "patient_journeys", "target_outcome": "treatment_initiated"},
            result,
            None,
        )
    assert captured["mlflow_run_id"] == "run-this"


@pytest.mark.unit
def test_mlflow_run_id_is_a_deployer_graph_channel():
    from src.agents.ml_foundation.model_deployer.graph import create_model_deployer_graph

    assert "mlflow_run_id" in create_model_deployer_graph().channels
