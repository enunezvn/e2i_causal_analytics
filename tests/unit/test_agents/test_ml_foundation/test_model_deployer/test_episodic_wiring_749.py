"""#749: model_deployer must record its run to EPISODIC memory. Its store_deployment hook was defined but
(a) never called from run() and (b) called a non-existent insert_episodic_memory signature
(fixed by the compat shim + migration 039). Pin that run() now invokes the episodic writer
with a valid UUID session_id and degrades gracefully.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent

_DATA = {
    "experiment_id": "exp-749",
    "audit_workflow_id": "11111111-1111-1111-1111-111111111111",
    "training_run_id": "run-749",
    "model_id": "m1",
    "algorithm_name": "logistic_regression",
    "test_metrics": {"auc_roc": 0.71},
    "success_criteria_met": True,
    "model_candidate": {
        "algorithm_name": "lr",
        "algorithm_family": "linear",
        "selection_score": 0.8,
    },
    "qc_status": "passed",
    "overall_score": 0.86,
    "data_source": "optum_mart_discontinuation",
    "deployment_name": "deploy-749",
    "deployment_successful": True,
    "global_importance_ranked": [("age", 0.5)],
}


@pytest.mark.unit
def test_update_episodic_memory_invokes_store_deployment_with_uuid_session():
    agent = ModelDeployerAgent()
    with patch("src.agents.ml_foundation.model_deployer.agent.ModelDeployerMemoryHooks") as HookCls:
        hook = HookCls.return_value
        hook.store_deployment = AsyncMock(return_value="mem-1")
        asyncio.run(agent._update_episodic_memory(_DATA, _DATA))

    hook.store_deployment.assert_awaited_once()
    kwargs = hook.store_deployment.await_args.kwargs
    # session_id MUST be a valid UUID (the column is uuid) — uses audit_workflow_id
    uuid.UUID(kwargs["session_id"])
    assert kwargs["session_id"] == "11111111-1111-1111-1111-111111111111"
    assert "result" in kwargs and "state" in kwargs


@pytest.mark.unit
def test_update_episodic_memory_degrades_gracefully_on_error():
    agent = ModelDeployerAgent()
    with patch(
        "src.agents.ml_foundation.model_deployer.agent.ModelDeployerMemoryHooks",
        side_effect=RuntimeError("supabase unreachable"),
    ):
        asyncio.run(agent._update_episodic_memory(_DATA, _DATA))  # must not raise
