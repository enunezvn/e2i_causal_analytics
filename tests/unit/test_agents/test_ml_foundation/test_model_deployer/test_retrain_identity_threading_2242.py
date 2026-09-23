"""#2242: the retrained model's identity reaches the register_model node.

``MLFoundationPipeline._run_model_deployment`` -> deployer input -> ``ModelDeployerAgent.run``
-> the declared ``ModelDeployerState.retrain_of`` field (LangGraph drops undeclared keys at
channel boundaries, see the #2207 cohort-contract fields). The node's behaviour on that
field is covered end to end in tests/unit/test_services/test_retrain_registry_linkage_2242.py.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.agents.ml_foundation.model_deployer.state import ModelDeployerState

pytestmark = pytest.mark.unit

RETRAIN_OF: Dict[str, Any] = {
    "model_id": "4ec55d13-46c8-4df4-9ec8-7723fad67fb3",
    "model_name": "initiation_kisqali_goldstd_lr_v1",
    "model_version": "1.0",
    "experiment_id": "35a2cd41-4d85-4034-b1a0-1c2a61e3766e",
    "experiment_name": "initiation_kisqali_goldstd_eval_v1",
    "prediction_target": "initiation_kisqali",
    "new_model_version": "1.0_retrained_20260923_0542",
}


def test_state_declares_retrain_of():
    s = ModelDeployerState(audit_workflow_id=uuid4(), retrain_of=RETRAIN_OF)
    assert s.retrain_of == RETRAIN_OF
    assert ModelDeployerState(audit_workflow_id=uuid4()).retrain_of is None


@pytest.mark.asyncio
async def test_agent_threads_retrain_of_onto_the_initial_state():
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
                    "experiment_id": "exp_x",
                    "validation_metrics": {},
                    "success_criteria_met": True,
                    "deployment_name": "d",
                    "retrain_of": RETRAIN_OF,
                }
            )
        except Exception:
            pass  # the stub graph's output need not satisfy the agent's post-processing
    assert captured["retrain_of"] == RETRAIN_OF


@pytest.mark.asyncio
async def test_pipeline_deploy_stage_passes_retrain_of_to_the_deployer():
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
        await pipeline._run_model_deployment(
            {
                "problem_description": "p",
                "business_objective": "b",
                "target_outcome": "treatment_initiated",
                "data_source": "patient_journeys",
                "retrain_of": RETRAIN_OF,
            },
            result,
            None,
        )
    assert captured["retrain_of"] == RETRAIN_OF
    # the physical label still names the cohort contract persisted on the candidate
    assert captured["target_outcome"] == "treatment_initiated"


@pytest.mark.asyncio
async def test_agent_output_reports_the_registry_row_it_wrote():
    """codex r2: the retrain completion check needs THIS run's ml_model_registry id."""
    from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent

    class _Graph:
        async def ainvoke(self, initial_state, *_a, **_k):
            return {
                **initial_state,
                "model_registry_id": "rid-1",
                "promotion_successful": True,
                "deployment_action": "register",
            }

    agent = ModelDeployerAgent.__new__(ModelDeployerAgent)
    agent.agent_name = "model_deployer"
    agent.tier = 0
    agent.sla_seconds = 600
    agent.graph = _Graph()
    with (
        patch(
            "src.agents.ml_foundation.model_deployer.agent._get_opik_connector", return_value=None
        ),
        patch.object(ModelDeployerAgent, "_store_to_database", AsyncMock()),
        patch.object(ModelDeployerAgent, "_update_procedural_memory", AsyncMock()),
        patch.object(ModelDeployerAgent, "_update_semantic_memory", AsyncMock()),
        patch.object(ModelDeployerAgent, "_update_episodic_memory", AsyncMock()),
    ):
        out = await agent.run(
            {
                "model_uri": "runs:/x/model",
                "experiment_id": "exp_x",
                "validation_metrics": {},
                "success_criteria_met": True,
                "deployment_name": "d",
                "deployment_action": "register",
                "retrain_of": RETRAIN_OF,
            }
        )
    assert out["model_registry_id"] == "rid-1"
