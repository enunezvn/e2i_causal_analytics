"""#773 cluster B-3: simulated/failed MLflow registration must NOT crash the workflow.

PR #830 (audit F4) changed ``register_model``'s simulated-fallback branch to emit
``deployment_status="degraded"`` — but ``ModelDeployerState.deployment_status``
is ``Literal["pending", "deploying", "healthy", "unhealthy", "failed"]``: there
is no "degraded" member. Since LangGraph validates node updates against the
pydantic state schema, ANY failed/simulated MLflow registration now raises

    1 validation error for ModelDeployerState
    deployment_status: Input should be 'pending', 'deploying', 'healthy',
    'unhealthy' or 'failed' [type=literal_error, input_value='degraded', ...]

i.e. the workflow CRASHES instead of failing closed — the exact opposite of
PR #830's fail-closed intent (nightly tier0 e2e red since 2026-06-10).

Intent investigation (#830 body + consumer grep): the graph ENDS immediately
after a failed registration (``_should_continue_after_registration`` returns
"end" on ``registration_successful=False``), so nothing is ever deployed on
this path — there is no "deployed but impaired" tri-state for "degraded" to
describe. No consumer reads "degraded" (health_checker branches only on
"failed"/"healthy"; deployment_orchestrator emits only "failed"/"healthy"; the
DB ``deployment_status_enum`` has neither "degraded" nor "healthy"). The truth
signal already lives in ``registration_successful=False`` +
``registration_simulated=True`` + ``mlflow_available=False``. So the failure
case maps to the existing Literal member **"unhealthy"** rather than widening
the vocabulary with a member that has one producer and zero readers.

Faithful simulated path (same idiom as test_f4_fail_closed_simulation.py):
``model_uri="simulated://model"`` is an invalid MLflow URI, so the REAL
``MLflowConnector`` registration fails and the simulation fallback runs
exactly as in the failing nightly tests.
"""

from uuid import uuid4

import pytest

from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent
from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
    register_model,
)
from src.agents.ml_foundation.model_deployer.state import ModelDeployerState


@pytest.mark.asyncio
async def test_simulated_registration_emits_valid_state_literal():
    """The simulated-fallback update must validate against ModelDeployerState
    (LangGraph applies it to the pydantic state schema) and must map the
    failure to the existing "unhealthy" member — fail closed, not crash."""
    result = await register_model(
        {
            "model_uri": "simulated://model",
            "deployment_name": "i773_literal_test",
            "experiment_id": "e773",
        }
    )

    # The node still reports the truth (PR #830's intent, preserved).
    assert result.get("registration_successful") is False
    assert result.get("registration_simulated") is True

    # RED before the fix: pydantic literal_error on 'degraded'.
    # (extra="ignore" on BaseAgentSchema drops non-state keys;
    # audit_workflow_id is required-no-default, supplied at the boundary
    # exactly as agent.run() does.)
    state = ModelDeployerState(audit_workflow_id=uuid4(), **result)
    assert state.deployment_status == "unhealthy", (
        "failed/simulated registration must map to the existing 'unhealthy' "
        "Literal member (no consumer distinguishes 'degraded'; the graph ends "
        "before any deployment on this path)"
    )


@pytest.mark.asyncio
async def test_register_workflow_completes_failed_closed_on_simulated_mlflow():
    """End-to-end through the real compiled graph: a register-only run whose
    MLflow registration fails must COMPLETE with an honest failed status —
    not raise RuntimeError(literal_error). This is the exact shape of the
    nightly tests/integration/test_tier0_e2e.py::TestModelDeployer failures."""
    agent = ModelDeployerAgent()

    # RED before the fix: RuntimeError("Model deployment workflow failed:
    # 1 validation error for ModelDeployerState deployment_status ...").
    result = await agent.run(
        {
            "experiment_id": "exp_i773",
            "model_uri": "simulated://model",
            "validation_metrics": {"auc_roc": 0.75, "f1_score": 0.68},
            "success_criteria_met": True,
            "deployment_name": "i773_register_only",
            "deployment_action": "register",
        }
    )

    # Fail CLOSED: workflow completes and reports the truth.
    assert result["status"] == "failed"
    assert result["deployment_successful"] is False
    assert result["deployment_manifest"]["status"] == "unhealthy"
