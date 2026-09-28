"""A retrain candidate's deliverable ends at register + stage transition (Part of #2157).

Live, 2026-09-28 (retrain 073c38eb): the deployer registered the candidate
(ml_model_registry row c524db0f, MLflow v3 -> staging) and then ran the endpoint path.
Packaging failed, and no Celery worker has docker, so ``bentoml containerize`` cannot
work there either ("Backend docker is not healthy"). The deployer raised, the pipeline
dropped its output, and the job failed with "no ml_model_registry row ... written"
although one was. #2242's contract only needs the registry row written by this run.
Serving goes through the e2i_bentoml sidecar (scripts/sync_goldstd_serving.py), not a
per-model image.

So a retrain uses the deployer's existing ``deployment_action="promote"`` path: the graph
ends after promote_stage. The output says why no endpoint was deployed, and the episodic
row is still written.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.unit

RETRAIN_OF: Dict[str, Any] = {
    "model_id": "4ec55d13-46c8-4df4-9ec8-7723fad67fb3",
    "model_name": "initiation_kisqali_goldstd_lr_v1",
    "model_version": "1.0",
    "experiment_id": "35a2cd41-4d85-4034-b1a0-1c2a61e3766e",
    "experiment_name": "initiation_kisqali_goldstd_eval_v1",
    "prediction_target": "initiation_kisqali",
    "new_model_version": "1.0_retrained_20260928_1617_d18ac8",
}


async def _deployer_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    from src.agents.tier_0.pipeline import (
        MLFoundationPipeline,
        PipelineConfig,
        PipelineResult,
        PipelineStage,
    )

    captured: Dict[str, Any] = {}

    class _Deployer:
        async def run(self, deployer_input):
            captured.update(deployer_input)
            return {"deployment_successful": False}

    pipeline = MLFoundationPipeline(PipelineConfig())
    result = PipelineResult(
        pipeline_run_id="run-1", status="running", current_stage=PipelineStage.MODEL_DEPLOYMENT
    )
    result.experiment_id = "exp_kisq_al_x"
    result.scope_spec = {}
    result.training_result = {"model_artifact_uri": "runs:/x/model", "validation_metrics": {}}
    with (
        patch.object(MLFoundationPipeline, "_get_agent", return_value=_Deployer()),
        patch.object(MLFoundationPipeline, "_get_audit_service", return_value=None),
    ):
        await pipeline._run_model_deployment(input_data, result, None)
    return captured


BASE_INPUT = {
    "problem_description": "p",
    "business_objective": "b",
    "target_outcome": "treatment_initiated",
    "data_source": "patient_journeys",
}


@pytest.mark.asyncio
async def test_a_retrain_asks_the_deployer_to_register_and_promote_only():
    captured = await _deployer_input({**BASE_INPUT, "retrain_of": RETRAIN_OF})
    assert captured["deployment_action"] == "promote"


@pytest.mark.asyncio
async def test_a_non_retrain_run_keeps_the_full_deploy_path():
    captured = await _deployer_input(dict(BASE_INPUT))
    assert captured.get("deployment_action", "deploy") == "deploy"


# --- the deployer, real graph: promote ends before packaging -----------------------


def _real_graph_patches():
    """MLflow succeeds; BentoML is AVAILABLE and build_bento would blow up if reached."""
    return (
        patch(
            "src.agents.ml_foundation.model_deployer.nodes.registry_manager._register_model_mlflow",
            return_value=("initiation_kisqali_goldstd_lr_v1", 3, "None"),
        ),
        patch(
            "src.agents.ml_foundation.model_deployer.nodes.registry_manager._transition_stage_mlflow",
            return_value=True,
        ),
        patch(
            "src.agents.ml_foundation.model_deployer.nodes.deployment_orchestrator.BENTOML_AVAILABLE",
            True,
        ),
        patch(
            "src.agents.ml_foundation.model_deployer.nodes.deployment_orchestrator.build_bento",
            side_effect=AssertionError("package_model reached on a promote-only run"),
        ),
    )


async def _run_agent(deployment_action: str) -> tuple[Dict[str, Any], AsyncMock]:
    from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent

    p1, p2, p3, p4 = _real_graph_patches()
    episodic = AsyncMock()
    with (
        p1,
        p2,
        p3,
        p4,
        patch.object(ModelDeployerAgent, "_store_to_database", AsyncMock()),
        patch.object(ModelDeployerAgent, "_update_procedural_memory", AsyncMock()),
        patch.object(ModelDeployerAgent, "_update_semantic_memory", AsyncMock()),
        patch.object(ModelDeployerAgent, "_update_episodic_memory", episodic),
    ):
        out = await ModelDeployerAgent().run(
            {
                "model_uri": "runs:/abc/model",
                "experiment_id": "exp_2157",
                "validation_metrics": {"roc_auc": 0.8363},
                "success_criteria_met": True,
                "deployment_name": "exp_2157_deployment",
                "target_environment": "staging",
                "deployment_action": deployment_action,
            }
        )
    return out, episodic


@pytest.mark.asyncio
async def test_promote_only_never_packages_and_says_no_endpoint_was_deployed():
    out, episodic = await _run_agent("promote")
    assert out["status"] == "completed"
    assert out["bentoml_tag"] == ""  # a reached package_model would have set a tag
    reason = out["deployment_skipped_reason"]
    assert "no endpoint" in reason and "promote" in reason
    # #2120/#2157: the deployer's episodic row is written on the register-only path too
    episodic.assert_awaited_once()
    assert episodic.await_args.args[0]["deployment_skipped_reason"] == reason


@pytest.mark.asyncio
async def test_the_full_deploy_path_still_packages_and_containerizes():
    """``deploy`` keeps the endpoint path: package_model runs (build_bento raises, and it
    falls back to simulated), then containerize_model, which is stopped here so that no
    real ``bentoml containerize`` runs."""
    stop = patch(
        "src.agents.ml_foundation.model_deployer.nodes.deployment_orchestrator.containerize_bento",
        side_effect=RuntimeError("containerize reached"),
    )
    with stop, pytest.raises(RuntimeError, match="containerization_error: .*containerize reached"):
        await _run_agent("deploy")


@pytest.mark.asyncio
async def test_episodic_row_records_why_no_endpoint_was_deployed():
    from src.agents.ml_foundation.model_deployer.memory_hooks import ModelDeployerMemoryHooks

    insert = AsyncMock(return_value="mem-1")
    with patch("src.memory.episodic_memory.insert_episodic_memory", insert):
        await ModelDeployerMemoryHooks().store_deployment(
            session_id=None,
            result={"deployment_skipped_reason": "deployment_action='promote': no endpoint"},
            state={
                "experiment_id": "exp_2157",
                "audit_workflow_id": "6f1c0e1e-8a43-4d43-9c55-2b8f1e0d7a11",
                "deployment_action": "promote",
            },
        )
    content = insert.await_args.kwargs["raw_content"]
    assert content["deployment_skipped_reason"] == "deployment_action='promote': no endpoint"
    assert content["deployment_action"] == "promote"
    summary = insert.await_args.kwargs["summary"]
    assert "no endpoint" in summary
    assert "Health:" not in summary  # no endpoint -> no health check, not a FAILED one


# --- a failure after registration names the row it left behind ---------------------


@pytest.mark.asyncio
async def test_a_failure_after_registration_names_the_registry_row_it_wrote():
    from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent

    class _Graph:
        async def ainvoke(self, initial_state, *_a, **_k):
            return {
                **initial_state,
                "model_registry_id": "c524db0f-1df1-4f21-806f-3b857c5245b9",
                "error": "Bento validation failed: ['Bento not found: Invalid Tag ...']",
                "error_type": "bento_validation_error",
            }

    agent = ModelDeployerAgent.__new__(ModelDeployerAgent)
    agent.agent_name = "model_deployer"
    agent.tier = 0
    agent.graph = _Graph()
    with (
        patch(
            "src.agents.ml_foundation.model_deployer.agent._get_opik_connector", return_value=None
        ),
        pytest.raises(RuntimeError) as err,
    ):
        await agent.run(
            {
                "model_uri": "runs:/x/model",
                "experiment_id": "exp_x",
                "validation_metrics": {},
                "success_criteria_met": True,
                "deployment_name": "d",
            }
        )
    msg = str(err.value)
    assert msg.startswith("bento_validation_error: Bento validation failed")
    assert "c524db0f-1df1-4f21-806f-3b857c5245b9" in msg
