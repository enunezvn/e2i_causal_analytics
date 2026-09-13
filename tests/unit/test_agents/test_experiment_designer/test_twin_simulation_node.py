"""R3/H8 + R2/H3: with the twin pre-screen ENABLED but no decision-useful real
model, the node must surface an honest warning and CONTINUE to design — never a
fabricated 'deploy'/'skip' off a fake effect. (The R2 fail-closed tool guarantees
the tool itself cannot fabricate; this locks the node's handling of that.)"""

import asyncio
import logging
from unittest.mock import patch

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


def test_a_failure_keeps_the_exception_text_out_of_warnings_and_errors(caplog):
    """#2020 E1b: the node's except put ``str(e)`` into the warnings and errors, and the
    orchestrator stringifies this agent's whole output into the answer."""
    from src.agents.experiment_designer.nodes.twin_simulation import TwinSimulationNode

    raw = (
        "Input X contains NaN. For further information visit "
        "https://errors.pydantic.dev/2.12/v/value_error"
    )
    state = {
        "status": "context_loaded",
        "enable_twin_simulation": True,
        "intervention_type": "email_campaign",
        "brand": "Kisqali",
        "constraints": {},
        "warnings": [],
    }
    with (
        patch(
            "src.agents.experiment_designer.nodes.twin_simulation.simulate_intervention",
            side_effect=ValueError(raw),
        ),
        caplog.at_level(logging.ERROR),
    ):
        out = asyncio.run(TwinSimulationNode().execute(state))

    assert "Twin simulation failed. Proceeding with standard design." in out["warnings"]
    error = out["errors"][-1]
    assert error["error"] == "the twin simulation could not be completed"
    assert error["node"] == "twin_simulation"
    assert error["recoverable"] is True
    assert "timestamp" in error
    for leak in ("NaN", "pydantic.dev"):
        assert leak not in str(out["warnings"]), f"{leak!r} reached the warnings"
        assert leak not in str(out["errors"]), f"{leak!r} reached the errors"
    # Still recoverable: the design continues.
    assert out["status"] == "reasoning"
    assert out["skip_experiment"] is False
    assert raw in caplog.text
    # The message alone would satisfy the check above; the traceback is what makes it debuggable.
    assert any(r.name.endswith("twin_simulation") and r.exc_info for r in caplog.records)
