"""Regression tests for #2127: the confidence of a causal_impact estimate in
the audit chain is the refutation-suite score, recorded on the ``refutation``
row; the ``estimation`` row stays NULL by definition.

Measured 2026-09-15 (live, 7 d): ``audit_chain_entries.confidence_score`` was
NULL on 177/177 ``estimation`` rows and 155/155 ``refutation`` rows of
``agent_name='causal_impact'``. Two mechanisms in ``traced_node``
(``src/agents/causal_impact/graph.py``):

* the estimation branch read ``est.get("confidence")`` — a key
  ``EstimationResult`` (``state.py``) never has and ``nodes/estimation.py``
  never writes — so it forwarded ``None`` on every row;
* the refutation branch never set ``confidence_score`` at all, although the
  node's return dict carries the suite score twice: ``refutation_confidence``
  (``nodes/refutation.py``, sibling of ``refutation_results``) and
  ``refutation_results["confidence_adjustment"]`` (``RefutationSuite
  .to_legacy_format``).

Owner decision (issue comment, 2026-09-15): the platform already defines the
number — ``RefutationRunner._calculate_confidence_score`` is what gates
PROCEED / REVIEW / BLOCK — so the ``refutation`` row carries it as returned,
and the ``estimation`` row stays NULL on purpose (an estimate has precision,
not confidence, before refutation; the chain is append-only so the row cannot
receive the score later). The dead read goes.

Fixtures are imported from the #2123 harness
(``test_audit_interpretation_confidence_2123.py``): the same recording double
and the real ``traced_node``.
"""

from __future__ import annotations

import inspect
import logging

import pytest

from src.agents.causal_impact.graph import traced_node
from src.agents.causal_impact.nodes.refutation import RefutationNode
from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationRunner,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)
from src.utils.audit_chain import RefutationResults
from tests.unit.test_agents.test_causal_impact.test_audit_interpretation_confidence_2123 import (  # noqa: F401  (pytest fixtures, resolved by name)
    _reset_audit_service,
    base_state,
    mock_audit_service,
    mock_opik,
)

GRAPH_LOGGER = "src.agents.causal_impact.graph"


def _legacy_refutation_results(confidence_adjustment: float) -> dict:
    """The dict shape ``RefutationSuite.to_legacy_format`` returns (the value
    the node stores under ``refutation_results``) — three tests, two passed."""
    return {
        "tests_passed": 2,
        "tests_failed": 1,
        "total_tests": 3,
        "overall_robust": True,
        "individual_tests": {
            "placebo_treatment": {"passed": True, "status": "passed"},
            "random_common_cause": {"passed": True, "status": "passed"},
            "data_subset": {"passed": False, "status": "failed"},
        },
        "confidence_adjustment": confidence_adjustment,
        "gate_decision": "proceed",
        "needs_review": False,
        "skipped_tests": {},
    }


def _refutation_node(result_extra: dict):
    """A fake refutation node returning a completed result plus ``result_extra``
    — the shape ``RefutationNode.execute`` returns on success (``refutation_results``
    beside ``refutation_confidence``) or on early failure (``refutation_error``)."""

    @traced_node("refutation")
    async def _node(state):
        return {
            "status": "completed",
            "current_phase": "sensitivity",
            "refutation_latency_ms": 12.0,
            **result_extra,
        }

    return _node


def _estimation_node(estimation_result: dict):
    """A fake estimation node returning the ``EstimationResult`` shape."""

    @traced_node("estimation")
    async def _node(state):
        return {
            "status": "completed",
            "current_phase": "refutation",
            "estimation_result": estimation_result,
        }

    return _node


def _full_estimation_result() -> dict:
    return {
        "ate": 0.12,
        "ate_ci_lower": 0.05,
        "ate_ci_upper": 0.19,
        "ate_std_error": 0.035,
        "p_value": 0.001,
        "statistical_significance": True,
        "sample_size": 1200,
        "method": "backdoor.linear_regression",
    }


def _warnings(caplog, fragment: str):
    return [
        r
        for r in caplog.records
        if r.name == GRAPH_LOGGER and r.levelno == logging.WARNING and fragment in r.getMessage()
    ]


# ---------------------------------------------------------------------------
# T1 — the refutation row carries the suite score AS RETURNED by the node.
# RED on base: the branch never sets confidence_score → None.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refutation_row_carries_the_suite_score(mock_audit_service, mock_opik, base_state):
    node = _refutation_node(
        {
            "refutation_results": _legacy_refutation_results(0.875),
            "refutation_confidence": 0.875,
            "gate_decision": "proceed",
        }
    )
    await node(base_state)

    assert mock_audit_service.add_entry.call_count == 1
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "refutation"
    cs = kwargs["confidence_score"]
    assert isinstance(cs, float) and cs == 0.875, (
        "the refutation audit row must carry the refutation-suite score the node "
        f"returned as refutation_confidence (0.875); got {cs!r} (#2127)"
    )
    # Untouched by this lane: verdict + individual-test wrapping + output payload.
    assert kwargs["validation_passed"] is True
    rr = kwargs["refutation_results"]
    assert isinstance(rr, RefutationResults)
    assert rr.placebo_treatment is True
    assert rr.random_common_cause is True
    assert rr.data_subset is False
    assert rr.unobserved_confound is None
    assert kwargs["output_data"]["tests_passed"] == 2
    assert kwargs["output_data"]["overall_robust"] is True
    assert kwargs["output_data"]["gate_decision"] == "proceed"


# ---------------------------------------------------------------------------
# T2 (pin, green on base) — early failure: no suite ran, no key, NULL silently.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refutation_early_failure_row_is_null_without_warning(
    mock_audit_service, mock_opik, base_state, caplog
):
    node = _refutation_node(
        {
            "status": "failed",
            "refutation_error": "DoWhy model unavailable",
            "error_message": "DoWhy model unavailable",
        }
    )
    with caplog.at_level(logging.WARNING, logger=GRAPH_LOGGER):
        await node(base_state)

    assert mock_audit_service.add_entry.call_count == 1
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "refutation_error"
    assert kwargs["confidence_score"] is None
    assert kwargs["validation_passed"] is False
    assert kwargs["refutation_results"] is None
    assert not _warnings(caplog, "refutation_confidence"), (
        "an absent refutation_confidence means no suite ran — NULL, no warning"
    )


# ---------------------------------------------------------------------------
# T3 — a value that is not a number in [0, 1] is a runner bug, not a value to
# store: None + exactly one WARNING. On base the value is None either way
# (never read), so the None half is a pin; the WARNING half is RED on base.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [True, 1.7, -0.25, "0.9"])
@pytest.mark.asyncio
async def test_refutation_out_of_range_or_non_numeric_score_is_null_with_one_warning(
    bad_value, mock_audit_service, mock_opik, base_state, caplog
):
    node = _refutation_node(
        {
            "refutation_results": _legacy_refutation_results(0.875),
            "refutation_confidence": bad_value,
        }
    )
    with caplog.at_level(logging.WARNING, logger=GRAPH_LOGGER):
        await node(base_state)

    assert mock_audit_service.add_entry.call_count == 1
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "refutation"
    assert kwargs["confidence_score"] is None, (
        f"refutation_confidence={bad_value!r} must not be stored, clamped or "
        f"rescaled; got {kwargs['confidence_score']!r}"
    )
    # Everything else on the row is unaffected by the bad score.
    assert kwargs["validation_passed"] is True
    assert isinstance(kwargs["refutation_results"], RefutationResults)

    warned = _warnings(caplog, "refutation_confidence")
    assert len(warned) == 1, (
        "expected exactly one WARNING on the graph logger naming "
        f"refutation_confidence; got {[r.getMessage() for r in warned]}"
    )
    assert repr(bad_value) in warned[0].getMessage()


# ---------------------------------------------------------------------------
# T4 — the estimation row is NULL by definition. The full-result variant is a
# pin (green on base); the stray-key variant is the teeth: RED on base, where
# the dead ``est.get("confidence")`` read forwards 0.9.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimation_row_is_null_and_output_summary_unchanged(
    mock_audit_service, mock_opik, base_state
):
    await _estimation_node(_full_estimation_result())(base_state)

    assert mock_audit_service.add_entry.call_count == 1
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "estimation"
    assert kwargs["confidence_score"] is None
    assert kwargs["validation_passed"] is None
    assert kwargs["refutation_results"] is None
    out = kwargs["output_data"]
    assert out["ate"] == 0.12
    assert out["p_value"] == 0.001
    assert out["statistical_significance"] is True
    assert set(out) == {
        "current_phase",
        "status",
        "has_error",
        "ate",
        "p_value",
        "statistical_significance",
    }


@pytest.mark.asyncio
async def test_estimation_row_ignores_a_stray_confidence_key(
    mock_audit_service, mock_opik, base_state
):
    await _estimation_node({**_full_estimation_result(), "confidence": 0.9})(base_state)

    assert mock_audit_service.add_entry.call_count == 1
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "estimation"
    assert kwargs["confidence_score"] is None, (
        "the estimation row is NULL by definition — before refutation an estimate "
        "has precision (SE/CI/p), not confidence; the wrapper must not read a "
        f"``confidence`` key off the estimate (#2127); got {kwargs['confidence_score']!r}"
    )
    assert "confidence" not in kwargs["output_data"]


# ---------------------------------------------------------------------------
# T5 (agreement) — the number on the row IS the gate score: the real
# ``_calculate_confidence_score`` → ``RefutationSuite.confidence_score`` →
# both carriers on the node's return (``refutation_confidence`` and
# ``refutation_results["confidence_adjustment"]``, ``nodes/refutation.py``
# ``execute``) → the audit row.
# ---------------------------------------------------------------------------


def _real_suite() -> RefutationSuite:
    tests = [
        RefutationResult(
            test_name=RefutationTestType.PLACEBO_TREATMENT,
            status=RefutationStatus.PASSED,
            original_effect=0.12,
            refuted_effect=0.004,
        ),
        RefutationResult(
            test_name=RefutationTestType.RANDOM_COMMON_CAUSE,
            status=RefutationStatus.WARNING,
            original_effect=0.12,
            refuted_effect=0.10,
        ),
        RefutationResult(
            test_name=RefutationTestType.DATA_SUBSET,
            status=RefutationStatus.PASSED,
            original_effect=0.12,
            refuted_effect=0.118,
        ),
        RefutationResult(
            test_name=RefutationTestType.SENSITIVITY_E_VALUE,
            status=RefutationStatus.SKIPPED,
            original_effect=0.12,
            refuted_effect=0.12,
        ),
    ]
    score = RefutationRunner()._calculate_confidence_score(tests)
    return RefutationSuite(
        passed=True, confidence_score=score, tests=tests, gate_decision=GateDecision.PROCEED
    )


def test_suite_score_is_the_documented_weighted_mean():
    # placebo 0.25×1.0 + random_common_cause 0.25×0.6 + data_subset 0.125×1.0,
    # over 0.625 of weight (the SKIPPED reading is excluded) = 0.84.
    assert _real_suite().confidence_score == pytest.approx(0.84)


def test_legacy_format_carries_the_same_score_as_the_suite():
    suite = _real_suite()
    legacy = suite.to_legacy_format()
    assert legacy["confidence_adjustment"] == suite.confidence_score
    assert legacy["gate_decision"] == "proceed"
    assert legacy["tests_passed"] == 2 and legacy["total_tests"] == 3


def test_node_return_shape_pins_both_carriers_to_suite_confidence_score():
    """``RefutationNode.execute`` returns ``refutation_confidence`` as
    ``suite.confidence_score`` beside ``refutation_results`` (the legacy dict
    whose ``confidence_adjustment`` is the same attribute). Running the node
    needs DoWhy + persistence, so the return shape is pinned on its source."""
    src = inspect.getsource(RefutationNode.execute)
    assert '"refutation_confidence": suite.confidence_score,' in src
    assert '"refutation_results": refutation_results,' in src
    assert "refutation_results = cast(RefutationResults, suite.to_legacy_format())" in src


@pytest.mark.asyncio
async def test_audit_row_equals_the_gate_score_end_to_end(
    mock_audit_service, mock_opik, base_state
):
    suite = _real_suite()
    node = _refutation_node(
        {
            "refutation_results": suite.to_legacy_format(),
            "refutation_confidence": suite.confidence_score,
            "gate_decision": suite.gate_decision.value,
        }
    )
    await node(base_state)

    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["confidence_score"] == suite.confidence_score
    assert kwargs["confidence_score"] == pytest.approx(0.84)
    assert kwargs["output_data"]["gate_decision"] == "proceed"
    assert kwargs["validation_passed"] is True
