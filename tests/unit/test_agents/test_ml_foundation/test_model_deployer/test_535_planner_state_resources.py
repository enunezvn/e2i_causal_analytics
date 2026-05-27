"""Regression tests for #535 — ModelDeployerState.resources type alignment.

``deployment_planner.plan_deployment`` writes ``ResourceProfile.to_dict()`` back
into ``state["resources"]`` (deployment_planner.py:486). That dict is *mixed
type*: ``cpu``/``memory`` are ``str``, ``gpu`` is ``None`` (``Optional[str]``),
and ``replicas``/``min_replicas``/``max_replicas``/``target_cpu_utilization`` are
``int``. Downstream consumers rely on the ints (e.g.
``deployment_orchestrator.py:248`` does ``replicas = resources.get("replicas", 1)``
and passes it as a numeric ``replicas=``), so the ints are intentional.

Before the fix, ``ModelDeployerState.resources`` was typed
``Optional[Dict[str, str]]``. Because the graph is built with
``StateGraph(ModelDeployerState)`` and ``BaseAgentSchema`` sets
``validate_assignment=True``, LangGraph re-validates every channel write — so the
planner's int/None values raised 5 Pydantic ``ValidationError``s on
``resources.*`` and Step-7 deployment-planning failed on *every* run
(#535, observed in tier0 scenario_a pre- and post-#531).

These tests construct ``ModelDeployerState`` from real planner output and assert
the mixed-type dict validates with its types preserved.
"""

from uuid import uuid4

import pytest

from src.agents.ml_foundation.model_deployer.nodes.deployment_planner import (
    ResourceProfile,
    plan_deployment,
)
from src.agents.ml_foundation.model_deployer.state import ModelDeployerState


class TestResourcesStateTypeAlignment:
    """#535: state.resources must accept the planner's mixed-type profile dict."""

    @pytest.mark.asyncio
    async def test_plan_deployment_resources_validate_against_state(self):
        """End-to-end regression guard the issue asks for: the ``resources`` dict
        that ``plan_deployment`` returns must validate when merged into
        ``ModelDeployerState`` (the LangGraph channel boundary)."""
        state = {
            "target_environment": "staging",
            "model_type": "classification",
            "experiment_id": "exp_535",
        }
        result = await plan_deployment(state)
        resources = result["resources"]

        # Guard against a vacuous test: the planner must really emit non-str values.
        assert isinstance(resources["replicas"], int)
        assert resources["gpu"] is None

        # The #535 regression: this construction raised pydantic.ValidationError
        # (5 errors on resources.*) before the type was widened.
        deployer_state = ModelDeployerState(
            audit_workflow_id=uuid4(),
            resources=resources,
        )

        # Types must survive validation (not be coerced/dropped).
        assert deployer_state.resources is not None
        assert deployer_state.resources["replicas"] == resources["replicas"]
        assert isinstance(deployer_state.resources["replicas"], int)
        assert deployer_state.resources["gpu"] is None
        assert deployer_state.resources["cpu"] == resources["cpu"]

    def test_resource_profile_to_dict_preserves_int_and_none_in_state(self):
        """Focused unit: ``ResourceProfile.to_dict()`` (int replicas, None gpu)
        validates into ``ModelDeployerState.resources`` with types intact."""
        profile_dict = ResourceProfile(
            cpu="4",
            memory="8Gi",
            gpu=None,
            replicas=2,
            min_replicas=2,
            max_replicas=10,
            target_cpu_utilization=60,
        ).to_dict()

        deployer_state = ModelDeployerState(
            audit_workflow_id=uuid4(),
            resources=profile_dict,
        )

        assert deployer_state.resources == profile_dict
        assert isinstance(deployer_state.resources["replicas"], int)
        assert deployer_state.resources["replicas"] == 2
        assert deployer_state.resources["target_cpu_utilization"] == 60
        assert deployer_state.resources["gpu"] is None

    def test_state_still_accepts_string_only_resources_input(self):
        """Backward-compat guard: the original input shape
        (``{"cpu": "2", "memory": "4Gi"}``, str→str) must keep validating after
        the widening — guards against over-narrowing to ``Dict[str, int]``."""
        deployer_state = ModelDeployerState(
            audit_workflow_id=uuid4(),
            resources={"cpu": "2", "memory": "4Gi"},
        )
        assert deployer_state.resources == {"cpu": "2", "memory": "4Gi"}
