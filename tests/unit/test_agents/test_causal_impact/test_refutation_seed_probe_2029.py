"""#2029: the per-refit calibration probe passes the run's seed to DoWhy.

The seed is derived by the MODULE-level ``seed_for_estimate`` in
``src.causal_engine.refutation_runner`` -- not by an attribute of the runner
INSTANCE. Tests that replace ``node.runner`` with a lightweight double (no DoWhy
refits) must keep working, and the probe must never raise on a runner double.
"""

from __future__ import annotations

import inspect
import time

import pytest

import src.agents.causal_impact.nodes.refutation as refutation_mod
from src.agents.causal_impact.nodes import refutation as node
from src.agents.causal_impact.nodes.refutation import RefutationNode
from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationRunner,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
    seed_for_estimate,
)


def _probe_window() -> str:
    """The source window around the throwaway 1-sim calibration refute.

    Anchored on the ``logger.debug`` message that follows the probe (its first
    occurrence in the module); the window reaches back far enough to contain
    the ``refute_estimate`` call itself.
    """
    src = inspect.getsource(node)
    anchor = src.index("per-refit calibration failed")
    return src[anchor - 2000 : anchor + 400]


def test_probe_window_contains_the_calibration_refute():
    # Negative control: the window provably holds the 1-sim probe, not some
    # other refute call in the module.
    probe = _probe_window()
    assert 'method_name="placebo_treatment_refuter"' in probe
    assert "num_simulations=1," in probe


def test_probe_source_passes_random_state():
    probe = _probe_window()
    assert "random_state=" in probe, "the calibration probe must be seeded like the scored refits"


def test_probe_seed_comes_from_the_module_function_not_the_runner_instance():
    """Regression: ``self.runner._seed_for`` made the probe a contract on the
    runner OBJECT -- every runner double in the suite raised AttributeError
    inside the node's try/except and the whole refutation failed."""
    probe = _probe_window()
    assert "seed_for_estimate(" in probe
    assert "self.runner._seed_for" not in probe


# --- the one derivation ------------------------------------------------------


def test_seed_for_estimate_is_the_runner_derivation():
    assert seed_for_estimate("estimate-A") == RefutationRunner._seed_for("estimate-A")
    assert seed_for_estimate("estimate-A") != seed_for_estimate("estimate-B")
    seed = seed_for_estimate("estimate-A")
    assert isinstance(seed, int) and 0 <= seed <= 0x7FFFFFFF


def test_seed_for_estimate_none_and_empty_are_unseeded():
    assert seed_for_estimate(None) is None
    assert seed_for_estimate("") is None


# --- behavioural: a runner double without ``_seed_for`` must not break execute


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


class _BareRunner:
    """Minimal runner double: ONLY ``run_all_tests`` -- deliberately no
    ``_seed_for`` (mirrors ``_RecorderRunner`` / ``_Runner`` in sibling tests)."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def run_all_tests(self, **kwargs):
        self.kwargs = kwargs
        return _proceed_suite()


class _RecordingModel:
    """DoWhy CausalModel double that records the probe's ``random_state``."""

    def __init__(self) -> None:
        self.refute_kwargs: list[dict] = []

    def refute_estimate(self, identified_estimand, estimate, **kwargs):
        self.refute_kwargs.append(kwargs)
        return object()


def _state(**overrides) -> dict:
    state = {
        "query": "seeded probe",
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
def bare_node(monkeypatch):
    n = RefutationNode()
    n.runner = _BareRunner()
    model = _RecordingModel()
    monkeypatch.setattr(
        refutation_mod,
        "_reconstruct_dowhy_artifacts",
        lambda **kwargs: (model, object(), object()),
    )

    async def _no_signal(outcome):
        return None

    monkeypatch.setattr(n, "_log_validation_outcome_signal", _no_signal)
    return n, model


@pytest.mark.asyncio
async def test_execute_with_runner_double_lacking_seed_for_does_not_fail(bare_node):
    n, _model = bare_node
    result = await n.execute(_state())
    assert n.runner.kwargs is not None, "run_all_tests was never reached"
    assert result["status"] != "failed"
    assert result["gate_decision"] == "proceed"


@pytest.mark.asyncio
async def test_probe_random_state_matches_module_seed_for_the_query_id(bare_node):
    n, model = bare_node
    await n.execute(_state(query_id="query-2029-seeded"))
    probes = [k for k in model.refute_kwargs if k.get("num_simulations") == 1]
    assert probes, "the 1-sim calibration probe did not run"
    assert probes[0]["random_state"] == seed_for_estimate("query-2029-seeded")
    assert probes[0]["random_state"] is not None
