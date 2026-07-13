# tests/unit/test_agents/test_causal_impact/test_refutation_randomized.py
"""RefutationNode must thread the state's ``randomized_design`` flag into
``RefutationRunner.run_all_tests`` (post-#1217 e-value RCT-gate follow-up).

The flag is DESIGN knowledge declared by the API layer (dataset spec); the node
is a pure conduit. Default is fail-closed: absent flag → False → the
observational unmeasured-confounding gate stays fully armed.
"""

from __future__ import annotations

import time

import pytest

import src.agents.causal_impact.nodes.refutation as refutation_mod
from src.agents.causal_impact.nodes.refutation import RefutationNode
from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)


def _proceed_suite() -> RefutationSuite:
    return RefutationSuite(
        passed=True,
        confidence_score=1.0,
        tests=[
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.08,
                refuted_effect=0.01,
            )
        ],
        gate_decision=GateDecision.PROCEED,
    )


class _RecorderRunner:
    """Kwargs-capturing stand-in for RefutationRunner (no DoWhy refits)."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def run_all_tests(self, **kwargs):
        self.kwargs = kwargs
        return _proceed_suite()


def _state(**overrides) -> dict:
    state = {
        "query": "rct question",
        "query_id": "",  # no persistence path
        "treatment_var": "control_group_flag",
        "outcome_var": "action_taken",
        "confounders": [],
        "estimation_result": {
            "ate": 0.08,
            "ate_ci_lower": 0.06,
            "ate_ci_upper": 0.10,
            "method": "linear_regression",
            "selected_estimator": "ols",
        },
        "compute_deadline": time.monotonic() + 60,
        "status": "in_progress",
    }
    state.update(overrides)
    return state


@pytest.fixture
def node(monkeypatch):
    n = RefutationNode()
    recorder = _RecorderRunner()
    n.runner = recorder
    monkeypatch.setattr(
        refutation_mod,
        "_reconstruct_dowhy_artifacts",
        lambda **kwargs: (object(), object(), object()),
    )

    async def _no_signal(outcome):
        return None

    monkeypatch.setattr(n, "_log_validation_outcome_signal", _no_signal)
    return n


@pytest.mark.asyncio
async def test_randomized_design_true_is_threaded_to_runner(node):
    result = await node.execute(_state(randomized_design=True))
    assert node.runner.kwargs is not None
    assert node.runner.kwargs.get("randomized_design") is True
    assert result["gate_decision"] == "proceed"


@pytest.mark.asyncio
async def test_absent_flag_defaults_to_observational_false(node):
    """Fail-closed: a state without the flag must arm the full gate."""
    await node.execute(_state())
    assert node.runner.kwargs is not None
    assert node.runner.kwargs.get("randomized_design") is False
