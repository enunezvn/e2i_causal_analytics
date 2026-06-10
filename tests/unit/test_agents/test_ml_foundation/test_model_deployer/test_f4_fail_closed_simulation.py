"""F4 (21-agent audit, HIGH): model_deployer must FAIL CLOSED on simulated/failed
MLflow operations — never fabricate ``registration_successful`` /
``promotion_successful = True`` or a random rollback id.

The "try real MLflow, fall back to simulation" path is an intentional dev
pattern (commit 214890aa) — REASON-BEFORE-RULES: we KEEP the simulation values
(registered name/version) for dev inspection, but the success flags must report
the TRUTH (``False`` when simulated), so the graph fails closed
(``registration_successful=False`` -> graph ends; ``promotion_successful``
drives overall status) and ``ml_deployments``/``ml_model_registry`` are never
populated by a fabricated success.

Faithful, NO mocks: ``model_uri="simulated://model"`` is an invalid MLflow URI,
so the REAL ``MLflowConnector`` registration fails and the simulation fallback
is exercised exactly as in prod-degraded conditions.
"""

import pytest

from src.agents.ml_foundation.model_deployer.nodes.deployment_orchestrator import (
    check_rollback_availability,
)
from src.agents.ml_foundation.model_deployer.nodes.registry_manager import (
    promote_stage,
    register_model,
)


@pytest.mark.asyncio
async def test_register_model_simulated_does_not_fabricate_success():
    """register_model on a simulated/failed MLflow registration must report
    registration_successful=False (was hardcoded True), and mlflow_available
    must reflect the REAL MLflow result (was computed after the fallback
    overwrote registered_model_name -> always True)."""
    result = await register_model(
        {
            "model_uri": "simulated://model",
            "deployment_name": "f4_reg_test",
            "experiment_id": "e1",
        }
    )

    assert result.get("registration_successful") is False, (
        "simulated registration must NOT claim success"
    )
    assert result.get("mlflow_available") is False, (
        "mlflow_available must reflect real MLflow availability, not the "
        "post-fallback (always-non-None) registered_model_name"
    )


@pytest.mark.asyncio
async def test_promote_stage_simulated_does_not_fabricate_success():
    """promote_stage when the real MLflow transition fails must report
    promotion_successful=False (was hardcoded True) consistent with the
    already-correct mlflow_transition_success=False signal."""
    result = await promote_stage(
        {
            "current_stage": "None",
            "promotion_target_stage": "Staging",
            "registered_model_name": "f4_nonexistent_model",
            "model_version": 1,
        }
    )

    assert result.get("mlflow_transition_success") is False, (
        "precondition: the real MLflow transition fails for an unregistered model"
    )
    assert result.get("promotion_successful") is False, (
        "simulated/failed promotion must NOT claim success"
    )
    # current_stage must NOT advance to the target on a failed transition,
    # otherwise the version_record would claim the promoted stage.
    assert result.get("current_stage") == "None", (
        "failed promotion must keep the previous stage, not claim the target"
    )


@pytest.mark.asyncio
async def test_rollback_does_not_fabricate_previous_deployment_id():
    """check_rollback_availability must not fabricate a random
    ``deploy_prev_<uuid>`` previous_deployment_id when no real previous
    deployment exists; it must fail closed (rollback_available=False)."""
    result = await check_rollback_availability(
        {"experiment_id": "e1", "current_stage": "Production"}
    )

    pid = result.get("previous_deployment_id")
    assert not (pid and str(pid).startswith("deploy_prev_")), (
        f"must NOT fabricate a random rollback id, got {pid!r}"
    )
    assert result.get("rollback_available") is False, (
        "with no real previous deployment (none in state, no history query), "
        "rollback must fail closed"
    )


@pytest.mark.asyncio
async def test_memory_hooks_skips_non_completed_deployment():
    """F4 (audit): contribute_to_memory must NOT persist a failed/partial
    deployment as a completed deployment memory. The gate previously read
    ``overall_status`` while the agent emits ``status`` — so the skip never
    fired. A 'partial' result must early-return with zero writes."""
    from src.agents.ml_foundation.model_deployer.memory_hooks import contribute_to_memory

    counts = await contribute_to_memory(
        {
            "status": "partial",
            "deployment_manifest": {"environment": "staging"},
            "deployment_successful": False,
        },
        {"deployment_name": "f4_partial", "experiment_id": "e1"},
    )

    assert counts.get("episodic_stored", 0) == 0
    assert counts.get("semantic_stored", 0) == 0
    assert counts.get("working_cached", 0) == 0


@pytest.mark.asyncio
async def test_rollback_uses_real_previous_deployment_id_when_present():
    """When a REAL previous_deployment_id is supplied, rollback is available and
    the real id is used verbatim (no fabrication)."""
    result = await check_rollback_availability(
        {
            "experiment_id": "e1",
            "current_stage": "Production",
            "previous_deployment_id": "deploy_realprev_001",
        }
    )

    assert result.get("rollback_available") is True
    assert result.get("previous_deployment_id") == "deploy_realprev_001"
