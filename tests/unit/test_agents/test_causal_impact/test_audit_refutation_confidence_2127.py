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
import re

import pytest

import tests.unit.test_agents.test_causal_impact.test_refutation_block_never_mints_1991 as harness_1991
from src.agents.causal_impact.graph import traced_node
from src.agents.causal_impact.nodes.refutation import RefutationNode
from src.causal_engine.expert_review_gate import ExpertReviewGate
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
# T2b (round 3) — an EXCEPTION return must not inherit a score. The node's
# two ``except`` returns spread the INPUT state, so a ``refutation_confidence``
# already in that state rides along on the error return although THIS
# invocation ran no suite. Both of those returns set the ``refutation_error``
# KEY (its value may be empty — the generic handler stores ``str(e)``, "" for
# a bare exception), and the completed return never does, so the wrapper
# gates the score on key PRESENCE — not on the value's truthiness (round 5),
# and not on the failed-closed verdict (round 4: a completed BLOCK also
# fails closed but carries a fresh score). Reachable
# only by invoking the traced node directly with such a dict:
# ``CausalImpactState`` declares no ``refutation_confidence`` channel, so the
# compiled graph never carries one — this pins the intent, not a live defect.
# RED on bdfeeaac8: 0.91 recorded on the refutation_error row.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refutation_failed_closed_row_does_not_inherit_a_stale_score(
    mock_audit_service, mock_opik, base_state, caplog
):
    node = _refutation_node(
        {
            "status": "failed",
            "refutation_error": "DoWhy model unavailable",
            "error_message": "DoWhy model unavailable",
            # Stale score spread from the input state by the node's error return.
            "refutation_confidence": 0.91,
        }
    )
    with caplog.at_level(logging.WARNING, logger=GRAPH_LOGGER):
        await node(base_state)

    assert mock_audit_service.add_entry.call_count == 1
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "refutation_error"
    assert kwargs["confidence_score"] is None, (
        "this invocation ran no suite — the refutation_error row must be NULL, not "
        f"the score inherited from the input state; got {kwargs['confidence_score']!r}"
    )
    assert kwargs["validation_passed"] is False
    assert kwargs["refutation_results"] is None
    assert not _warnings(caplog, "refutation_confidence")


# ---------------------------------------------------------------------------
# T2b' (round 5) — the marker is key PRESENCE, not truthiness. The generic
# handler stores ``"refutation_error": str(e)``, which is "" for a bare
# exception (``str(RuntimeError()) == ""``). A truthiness gate treats that
# row as a completed return and records the stale inherited score. The row
# is still named ``refutation_error``: ``node_failed_closed`` falls through
# the falsy marker to ``status == "failed"`` + ``error_message``
# (``_FAILURE_PAYLOAD_KEYS``, audit_chain_mixin.py). RED on 41fa61f75:
# 0.91 recorded.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_refutation_error_marker_still_blocks_the_inherited_score(
    mock_audit_service, mock_opik, base_state, caplog
):
    node = _refutation_node(
        {
            "status": "failed",
            "refutation_error": "",
            "error_message": "Refutation failed: ",
            "refutation_confidence": 0.91,
        }
    )
    with caplog.at_level(logging.WARNING, logger=GRAPH_LOGGER):
        await node(base_state)

    assert mock_audit_service.add_entry.call_count == 1
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    # Named by the failed status + error_message, not by the (empty) marker.
    assert kwargs["action_type"] == "refutation_error"
    assert kwargs["confidence_score"] is None, (
        "an empty refutation_error is still the exception marker (str(e) of a bare "
        f"exception) — NULL, not the inherited score; got {kwargs['confidence_score']!r}"
    )
    assert kwargs["validation_passed"] is False
    assert kwargs["refutation_results"] is None
    assert not _warnings(caplog, "refutation_confidence")


class _RaisingRunner:
    """Stand-in runner whose suite raises a BARE exception (empty message)."""

    def run_all_tests(self, **kwargs):
        raise RuntimeError()


@pytest.mark.asyncio
async def test_real_node_bare_exception_return_does_not_inherit_a_stale_score(
    monkeypatch, mock_audit_service, mock_opik, base_state, caplog
):
    """The REAL ``RefutationNode.execute`` body: the stand-in runner raises
    ``RuntimeError()`` inside the node's ``try``, the generic handler returns
    ``refutation_error: ""`` and spreads the input state — which carries a
    stale ``refutation_confidence: 0.91`` — through the real ``traced_node``."""
    gate = ExpertReviewGate(repository=harness_1991._ReviewRepo(rows=[]), auto_create_review=True)
    real_node = harness_1991._node(monkeypatch, GateDecision.PROCEED, gate)
    real_node.runner = _RaisingRunner()
    traced = traced_node("refutation")(real_node.execute)

    with caplog.at_level(logging.WARNING, logger=GRAPH_LOGGER):
        result = await traced(
            {**harness_1991._state(), **base_state, "refutation_confidence": 0.91}
        )

    # Positive control: the exception reached the GENERIC handler.
    assert result["refutation_error"] == ""
    assert result["error_message"] == "Refutation failed: "
    assert result["status"] == "failed"
    assert result["refutation_confidence"] == 0.91  # inherited by the spread

    assert mock_audit_service.add_entry.call_count == 1
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "refutation_error"
    assert kwargs["confidence_score"] is None, (
        f"real bare-exception return: expected NULL, got {kwargs['confidence_score']!r}"
    )
    assert not _warnings(caplog, "refutation_confidence")


# ---------------------------------------------------------------------------
# T2c / T2d (round 4) — provenance, not outcome. A COMPLETED suite that
# BLOCKs, or is withheld on the expert-review gate, returns ``status: failed``
# + ``error_message`` (so the row is a ``refutation_error`` row) but carries
# ITS OWN fresh ``refutation_confidence`` and ``refutation_results`` — that
# score is the one the row most needs and must be recorded. Only the two
# ``except`` returns set the ``refutation_error`` key. RED on 6326b347c: the
# ``None if failed_closed`` gate dropped the fresh score (None).
# ---------------------------------------------------------------------------


def _legacy_block_results(confidence_adjustment: float) -> dict:
    return {
        **_legacy_refutation_results(confidence_adjustment),
        "tests_passed": 1,
        "tests_failed": 2,
        "overall_robust": False,
        "individual_tests": {
            "placebo_treatment": {"passed": False, "status": "failed"},
            "random_common_cause": {"passed": True, "status": "passed"},
            "data_subset": {"passed": False, "status": "failed"},
        },
        "gate_decision": "block",
    }


@pytest.mark.asyncio
async def test_completed_block_row_keeps_its_fresh_suite_score(
    mock_audit_service, mock_opik, base_state, caplog
):
    node = _refutation_node(
        {
            "status": "failed",
            "current_phase": "failed",
            "error_message": "Estimate blocked: confidence 0.30 below 0.50",
            "refutation_results": _legacy_block_results(0.30),
            "refutation_confidence": 0.30,
            "gate_decision": "block",
            # NO refutation_error key: the suite ran and produced this score.
        }
    )
    with caplog.at_level(logging.WARNING, logger=GRAPH_LOGGER):
        await node(base_state)

    assert mock_audit_service.add_entry.call_count == 1
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "refutation_error"
    assert kwargs["confidence_score"] == 0.30, (
        "a completed BLOCK carries its own fresh suite score — the refutation_error "
        f"row must record it, not NULL; got {kwargs['confidence_score']!r} (#2127)"
    )
    assert kwargs["validation_passed"] is False
    rr = kwargs["refutation_results"]
    assert isinstance(rr, RefutationResults)
    assert rr.placebo_treatment is False and rr.random_common_cause is True
    assert kwargs["output_data"]["gate_decision"] == "block"
    assert not _warnings(caplog, "refutation_confidence")


@pytest.mark.asyncio
async def test_expert_review_halt_row_keeps_its_fresh_suite_score(
    mock_audit_service, mock_opik, base_state, caplog
):
    node = _refutation_node(
        {
            "status": "failed",
            "current_phase": "awaiting_expert_review",
            "error_message": "withheld: the DAG was rejected by expert review",
            "refutation_results": _legacy_refutation_results(0.72),
            "refutation_confidence": 0.72,
            "gate_decision": "proceed",
            "expert_review_halt": True,
            "expert_review_decision": "rejected",
        }
    )
    with caplog.at_level(logging.WARNING, logger=GRAPH_LOGGER):
        await node(base_state)

    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "refutation_error"
    assert kwargs["confidence_score"] == 0.72
    assert kwargs["validation_passed"] is False
    assert isinstance(kwargs["refutation_results"], RefutationResults)
    assert not _warnings(caplog, "refutation_confidence")


# Positive control for the marker: the REAL node, driven into BLOCK with the
# #1991 harness (stand-in runner, no DoWhy), through the REAL traced_node.
# Proves the completed BLOCK return sets status failed + error_message WITHOUT
# refutation_error and carries the suite's score, and that the row records it.


@pytest.mark.asyncio
async def test_real_node_block_return_through_traced_node_records_the_score(
    monkeypatch, mock_audit_service, mock_opik, base_state
):
    gate = ExpertReviewGate(repository=harness_1991._ReviewRepo(rows=[]), auto_create_review=True)
    real_node = harness_1991._node(monkeypatch, GateDecision.BLOCK, gate)
    traced = traced_node("refutation")(real_node.execute)

    result = await traced({**harness_1991._state(), **base_state})

    # The node's completed BLOCK return: failed closed, fresh score, no marker.
    assert result["status"] == "failed" and result["error_message"]
    assert "refutation_error" not in result
    assert result["refutation_confidence"] == 0.3
    assert result["refutation_results"]["confidence_adjustment"] == 0.3
    assert result["refutation_results"]["gate_decision"] == "block"

    assert mock_audit_service.add_entry.call_count == 1
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "refutation_error"
    assert kwargs["confidence_score"] == 0.3, (
        f"real BLOCK return: expected 0.3 on the refutation_error row, got "
        f"{kwargs['confidence_score']!r}"
    )
    assert kwargs["validation_passed"] is False


def test_refutation_error_marker_is_set_only_by_the_except_returns():
    """Both ``except`` blocks of ``RefutationNode.execute`` return
    ``"refutation_error": ...``; the completed-return literal never does."""
    src = inspect.getsource(RefutationNode.execute)
    head, sep, tail = src.partition("except RefutationError")
    assert sep, "expected an `except RefutationError` block in RefutationNode.execute"
    assert not re.search(r'"refutation_error":', head), (
        "the completed return must not set refutation_error — it is the marker "
        "of an exception return"
    )
    assert len(re.findall(r'"refutation_error":\s*', tail)) == 2
    assert re.search(r'"refutation_confidence":\s*suite\.confidence_score', head)


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
    needs DoWhy + persistence, so the return shape is pinned on its source
    (whitespace-tolerant, so a reformat does not trip it)."""
    src = inspect.getsource(RefutationNode.execute)
    assert re.search(r'"refutation_confidence":\s*suite\.confidence_score', src)
    assert re.search(r'"refutation_results":\s*refutation_results', src)
    assert re.search(r"suite\.to_legacy_format\(\)", src)


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
