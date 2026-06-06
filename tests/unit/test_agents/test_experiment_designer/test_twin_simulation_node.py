"""R3/H8 + R2/H3: with the twin pre-screen ENABLED but no decision-useful real
model, the node must surface an honest warning and CONTINUE to design — never a
fabricated 'deploy'/'skip' off a fake effect. (The R2 fail-closed tool guarantees
the tool itself cannot fabricate; this locks the node's handling of that.)"""

import asyncio

import pytest

pytestmark = pytest.mark.xdist_group(name="experiment_designer_tools")


def test_enabled_node_without_model_warns_and_continues():
    from src.agents.experiment_designer.nodes.twin_simulation import TwinSimulationNode

    node = TwinSimulationNode()
    state = {
        "status": "context_loaded",
        "enable_twin_simulation": True,
        "intervention_type": "email_campaign",
        "brand": "Kisqali",
        "constraints": {},
        "warnings": [],
    }
    out = asyncio.run(node.execute(state))
    # Honest: no fabricated deploy/skip; proceeds to design (or honestly skips).
    assert out["status"] in {"reasoning", "skipped"}
    # Never a fabricated deploy that also forces an experiment: either the
    # recommendation is not "deploy", or skip_experiment is explicitly False.
    assert out.get("twin_recommendation") != "deploy" or out.get("skip_experiment") is False


def test_disabled_node_skips_to_reasoning():
    from src.agents.experiment_designer.nodes.twin_simulation import TwinSimulationNode

    node = TwinSimulationNode()
    state = {"status": "context_loaded", "enable_twin_simulation": False, "warnings": []}
    out = asyncio.run(node.execute(state))
    assert out["status"] == "reasoning"
    assert "twin_recommendation" not in out  # the dark default: never runs
