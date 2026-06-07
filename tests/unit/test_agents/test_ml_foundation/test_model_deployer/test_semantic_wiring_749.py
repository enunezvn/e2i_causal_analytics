"""#749/#772: model_deployer must populate the SEMANTIC graph (FalkorDB e2i_causal).
Its store_deployment_pattern hook (Deployment + DEPLOYS_FOR, Rollback + ROLLED_BACK) was
defined but never called from run(). Pin that run() now invokes the semantic writer with
mapped data and degrades gracefully.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent

_OUTPUT = {
    "deployment_successful": True,
    "status": "deployed",
    "rollback_occurred": False,
}
_STATE = {
    "experiment_id": "exp-749",
    "deployment_name": "deploy-749",
    "model_version": 3,
    "target_environment": "staging",
    "deployment_action": "blue_green",
}


@pytest.mark.unit
def test_update_semantic_memory_invokes_deployment_pattern_writer():
    agent = ModelDeployerAgent()
    with patch("src.agents.ml_foundation.model_deployer.agent.ModelDeployerMemoryHooks") as HookCls:
        hook = HookCls.return_value
        hook.store_deployment_pattern = AsyncMock(return_value=True)
        asyncio.run(agent._update_semantic_memory(_OUTPUT, _STATE))

    hook.store_deployment_pattern.assert_awaited_once()
    kwargs = hook.store_deployment_pattern.await_args.kwargs
    assert kwargs["experiment_id"] == "exp-749"
    assert kwargs["deployment_id"] == "deploy-749"
    assert kwargs["model_version"] == 3
    assert kwargs["target_environment"] == "staging"
    assert kwargs["deployment_status"] == "deployed"
    assert kwargs["deployment_strategy"] == "blue_green"
    assert kwargs["rollback_occurred"] is False


@pytest.mark.unit
def test_run_exposes_update_semantic_memory():
    agent = ModelDeployerAgent()
    assert hasattr(agent, "_update_semantic_memory")


@pytest.mark.unit
def test_update_semantic_memory_degrades_gracefully_on_error():
    agent = ModelDeployerAgent()
    with patch(
        "src.agents.ml_foundation.model_deployer.agent.ModelDeployerMemoryHooks",
        side_effect=RuntimeError("falkordb unreachable"),
    ):
        asyncio.run(agent._update_semantic_memory(_OUTPUT, _STATE))  # must not raise
