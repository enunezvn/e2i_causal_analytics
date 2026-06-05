"""P3 — Refutation fail-open seam + REVIEW wiring + memory integrity (H1, H2).

Findings (causal-validation-pipeline-review-20260605):
- H1: an errored/blocked refutation still yielded status="completed" with no
  confidence penalty (router defaulted gate→"proceed"; _build_output keyed
  status on ATE presence only and defaulted confidence_adjustment=1.0).
- H2: a REVIEW-band gate (confidence 0.50-0.70) was surfaced as
  passed/overall_robust=True, flowed identically to PROCEED, the
  ExpertReviewGate was never consulted, and memory persisted it as
  refutation_passed=True into semantic memory (RAG amplification).

These tests assert the fail-closed contract at the orchestration seam.
"""

from __future__ import annotations

import time
import types

import pytest

from src.agents.causal_impact.agent import CausalImpactAgent
from src.agents.causal_impact.graph import should_continue_after_refutation
from src.agents.causal_impact.nodes.refutation import RefutationNode
from src.causal_engine.refutation_runner import GateDecision, RefutationSuite


def _review_suite(confidence: float = 0.6) -> RefutationSuite:
    return RefutationSuite(
        passed=True,  # legacy "not blocked" — includes REVIEW
        confidence_score=confidence,
        tests=[],
        gate_decision=GateDecision.REVIEW,
    )


def _proceed_suite(confidence: float = 0.85) -> RefutationSuite:
    return RefutationSuite(
        passed=True,
        confidence_score=confidence,
        tests=[],
        gate_decision=GateDecision.PROCEED,
    )


# =============================================================================
# H1 — router must NOT fail open on a refutation error/failure
# =============================================================================


class TestRouterFailsClosed:
    def test_refutation_error_routes_to_error_handler(self):
        state = {"refutation_error": "DoWhy unavailable", "status": "failed"}
        assert should_continue_after_refutation(state) == "error_handler"

    def test_failed_status_routes_to_error_handler(self):
        # A failed refutation with NO refutation_results must not default to
        # "proceed" → sensitivity (the H1 fail-open).
        state = {"status": "failed"}
        assert should_continue_after_refutation(state) == "error_handler"

    def test_block_routes_to_error_handler(self):
        state = {"refutation_results": {"gate_decision": "block"}, "gate_decision": "block"}
        assert should_continue_after_refutation(state) == "error_handler"

    def test_review_routes_to_sensitivity(self):
        state = {"refutation_results": {"gate_decision": "review"}, "gate_decision": "review"}
        assert should_continue_after_refutation(state) == "sensitivity"

    def test_proceed_routes_to_sensitivity(self):
        state = {"refutation_results": {"gate_decision": "proceed"}, "gate_decision": "proceed"}
        assert should_continue_after_refutation(state) == "sensitivity"


# =============================================================================
# H1 + H2 — _build_output must consult the refutation outcome
# =============================================================================


def _base_state(**overrides):
    state = {
        "query_id": "q1",
        "estimation_result": {
            "ate": 0.5,
            "ate_ci_lower": 0.3,
            "ate_ci_upper": 0.7,
            "statistical_significance": True,
            "method": "linear_regression",
        },
        "interpretation": {},
        "sensitivity_analysis": {},
        "causal_graph": {},
    }
    state.update(overrides)
    return state


class TestBuildOutputFailsClosed:
    def test_refutation_error_yields_failed_penalized(self):
        agent = CausalImpactAgent()
        # Refutation errored: no refutation_results, status=failed (router would
        # have sent this to error_handler, but _build_output is authoritative).
        state = _base_state(refutation_error="DoWhy unavailable", status="failed")
        out = agent._build_output(state, time.time())
        assert out["status"] == "failed", "errored refutation must NOT be 'completed'"
        assert out["refutation_passed"] is False
        # H1: confidence must be PENALIZED, not silently 1.0 (no penalty).
        assert out["confidence"] < 0.3, f"expected hard penalty, got {out['confidence']}"
        assert out["requires_further_analysis"] is True

    def test_review_band_flagged_needs_review_not_passed(self):
        agent = CausalImpactAgent()
        suite = _review_suite()
        state = _base_state(
            refutation_results=suite.to_legacy_format(),
            gate_decision="review",
        )
        out = agent._build_output(state, time.time())
        # REVIEW completes (it is not a hard failure) but must NOT be 'passed'.
        assert out["status"] == "completed"
        assert out["refutation_passed"] is False, "REVIEW must not be surfaced as passed/robust"
        assert out.get("needs_review") is True

    def test_proceed_band_passes(self):
        agent = CausalImpactAgent()
        suite = _proceed_suite()
        state = _base_state(
            refutation_results=suite.to_legacy_format(),
            gate_decision="proceed",
        )
        out = agent._build_output(state, time.time())
        assert out["status"] == "completed"
        assert out["refutation_passed"] is True
        assert out.get("needs_review") in (False, None)

    def test_block_band_is_failed_and_low_confidence(self):
        """BLOCK contract (codex follow-up): a blocked estimate is status=failed,
        not passed, not needs_review, and its confidence is the real (low) suite
        score — by gate definition < 0.50, so well below the 0.70 threshold. The
        refutation DID run, so this is the informative-low case, distinct from the
        unrun hard penalty."""
        agent = CausalImpactAgent()
        suite = RefutationSuite(
            passed=False,
            confidence_score=0.30,  # BLOCK band (< 0.50)
            tests=[],
            gate_decision=GateDecision.BLOCK,
        )
        state = _base_state(
            refutation_results=suite.to_legacy_format(),
            gate_decision="block",
        )
        out = agent._build_output(state, time.time())
        assert out["status"] == "failed"
        assert out["refutation_passed"] is False
        assert out.get("needs_review") is False
        assert out["confidence"] < 0.5, (
            f"blocked estimate must be low-confidence, got {out['confidence']}"
        )
        assert out["requires_further_analysis"] is True


# =============================================================================
# H2 — runner surfaces a distinct needs_review signal
# =============================================================================


class TestRunnerNeedsReview:
    def test_review_suite_needs_review(self):
        suite = _review_suite()
        assert suite.needs_review is True
        legacy = suite.to_legacy_format()
        assert legacy["needs_review"] is True
        assert legacy["gate_decision"] == "review"

    def test_proceed_suite_not_needs_review(self):
        suite = _proceed_suite()
        assert suite.needs_review is False
        assert suite.to_legacy_format()["needs_review"] is False


# =============================================================================
# H2 — REVIEW band consults the ExpertReviewGate (built control)
# =============================================================================


class _SpyGate:
    def __init__(self):
        self.calls = []

    async def check_approval(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(decision=types.SimpleNamespace(value="pending_review"))


class TestReviewGateWiring:
    @pytest.mark.asyncio
    async def test_review_consults_expert_review_gate(self):
        spy = _SpyGate()
        node = RefutationNode(expert_review_gate=spy)
        state = {
            "treatment_var": "t",
            "outcome_var": "y",
            "brand": "Kisqali",
            "dag_hash": "abc123",
            "query_id": "q1",
        }
        fields = await node._consult_review_gate(state, _review_suite())
        assert fields["needs_review"] is True
        assert spy.calls, "ExpertReviewGate.check_approval was not invoked on REVIEW"
        assert fields["expert_review_decision"] == "pending_review"
        assert fields.get("review_caveat")

    @pytest.mark.asyncio
    async def test_no_repo_gate_degrades_gracefully(self):
        # With no injected gate (no repository), the default gate bypasses but we
        # still flag needs_review — graceful, not crash, not silently-robust.
        node = RefutationNode()
        fields = await node._consult_review_gate(
            {"treatment_var": "t", "outcome_var": "y"}, _review_suite()
        )
        assert fields["needs_review"] is True


# =============================================================================
# H2 — memory must NOT persist a REVIEW-grade estimate as validated
# =============================================================================


class _MockHooks:
    def __init__(self):
        self.semantic_called = False

    async def cache_causal_analysis(self, *a, **k):
        return True

    async def store_causal_analysis(self, *a, **k):
        return "mem-1"

    async def store_causal_path(self, *a, **k):
        self.semantic_called = True
        return True


class TestInterpretationDowngradesReview:
    """H2 — a REVIEW result must NOT be described as robust in the user narrative."""

    @pytest.mark.asyncio
    async def test_review_narrative_is_cautionary_not_robust(self):
        from src.agents.causal_impact.nodes.interpretation import InterpretationNode

        node = InterpretationNode()
        refutation_results = _review_suite().to_legacy_format()
        refutation_results["tests_passed"] = 3
        refutation_results["total_tests"] = 4

        state = {
            "query": "impact of t on y",
            "query_id": "t-review",
            "treatment_var": "t",
            "outcome_var": "y",
            "confounders": ["region"],
            "data_source": "synthetic",
            "causal_graph": {
                "nodes": ["t", "y", "region"],
                "edges": [("region", "t"), ("t", "y")],
                "treatment_nodes": ["t"],
                "outcome_nodes": ["y"],
                "adjustment_sets": [["region"]],
                "dag_dot": "digraph { }",
                "confidence": 0.85,
            },
            "estimation_result": {
                "method": "CausalForestDML",
                "ate": 0.5,
                "ate_ci_lower": 0.4,
                "ate_ci_upper": 0.6,
                "standard_error": 0.05,
                "effect_size": "medium",
                "statistical_significance": True,
                "p_value": 0.01,
            },
            "refutation_results": refutation_results,
            "sensitivity_analysis": {
                "e_value": 2.5,
                "robust_to_confounding": True,
                "interpretation": "moderate",
            },
            "interpretation_depth": "standard",
            "user_context": {"expertise": "analyst"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }
        result = await node.execute(state)
        narrative = result["interpretation"]["narrative"].lower()
        # REVIEW (needs_review) must take the caution branch, not the robust one.
        assert "genuine and not spurious" not in narrative, (
            "REVIEW must not be described as robust/genuine in the narrative"
        )
        assert "caution" in narrative


class TestMemoryDoesNotAmplifyReview:
    @pytest.mark.asyncio
    async def test_review_result_not_written_to_semantic_memory(self):
        from src.agents.causal_impact.memory_hooks import contribute_to_memory

        hooks = _MockHooks()
        result = {
            "status": "completed",
            "ate_estimate": 0.5,
            "confidence": 0.6,
            "refutation_passed": False,  # REVIEW → not passed (H2 fix)
            "gate_decision": "review",
            "needs_review": True,
        }
        state = {"treatment_var": "t", "outcome_var": "y", "gate_decision": "review"}
        counts = await contribute_to_memory(
            result=result, state=state, memory_hooks=hooks, session_id="s1"
        )
        assert counts["semantic_stored"] == 0
        assert hooks.semantic_called is False, "REVIEW must NOT be written to semantic memory"

    @pytest.mark.asyncio
    async def test_proceed_result_written_to_semantic_memory(self):
        from src.agents.causal_impact.memory_hooks import contribute_to_memory

        hooks = _MockHooks()
        result = {
            "status": "completed",
            "ate_estimate": 0.5,
            "confidence": 0.85,
            "refutation_passed": True,
            "gate_decision": "proceed",
        }
        state = {"treatment_var": "t", "outcome_var": "y", "gate_decision": "proceed"}
        counts = await contribute_to_memory(
            result=result, state=state, memory_hooks=hooks, session_id="s1"
        )
        assert counts["semantic_stored"] == 1
        assert hooks.semantic_called is True
