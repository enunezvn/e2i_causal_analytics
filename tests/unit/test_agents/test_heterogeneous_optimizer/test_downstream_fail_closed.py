"""Tests for #437: heterogeneous_optimizer downstream nodes fail-closed on absent CATE.

Pins the disambiguation between:
- ``overall_ate is None`` and ``cate_by_segment is None`` (upstream cate_estimator
  produced no fields; downstream MUST fail-close).
- ``overall_ate == 0.0`` and populated ``cate_by_segment`` (legitimate honest
  zero; downstream MUST run normally).

Disambiguation matrix (4 upstream states × 3 downstream nodes = 12 cases):

| Row | Upstream state                                          | Expected downstream behavior                              |
|-----|---------------------------------------------------------|------------------------------------------------------------|
| 1   | status=failed                                           | Skip (existing behavior preserved)                         |
| 2   | non-failed AND overall_ate populated (incl. 0.0)        | Run normally; honest zero output ok                        |
|     | AND cate_by_segment populated                            |                                                            |
| 3   | non-failed AND overall_ate is None                      | **Fail-closed**: status=failed,                            |
|     | AND cate_by_segment is None                              | error="upstream cate_estimator produced no CATE fields"    |
| 4   | non-failed AND overall_ate is real float                | Run with heterogeneity_score=0 honestly; warning emitted   |
|     | AND cate_by_segment is empty dict                        |                                                            |

Wave-3 patterns honored:
- #2 LABEL-disguised-as-REWIRE: empty paths emit ``status=failed``, not a
  ``completed_with_no_cate`` flag while synthesizing fake content.
- #4 silent-substitute SKIPPED-not-substituted: row 4 emits an explicit warning
  rather than substituting a different signal.
- #5 pre-existing tests pinning OLD behavior: existing tests' OLD-default
  expectations (e.g. ``ate=0`` synthesized when upstream is None) are updated
  in companion edits, not pinned here.
"""

import pytest

from src.agents.heterogeneous_optimizer.nodes.policy_learner import PolicyLearnerNode
from src.agents.heterogeneous_optimizer.nodes.profile_generator import ProfileGeneratorNode
from src.agents.heterogeneous_optimizer.nodes.segment_analyzer import SegmentAnalyzerNode
from src.agents.heterogeneous_optimizer.state import (
    CATEResult,
    HeterogeneousOptimizerState,
    SegmentProfile,
)

# ---------------------------------------------------------------------------
# Shared fixtures for the disambiguation matrix.
# ---------------------------------------------------------------------------


def _cate_result(
    segment_name: str, segment_value: str, cate: float, sample_size: int = 100
) -> CATEResult:
    return {
        "segment_name": segment_name,
        "segment_value": segment_value,
        "cate_estimate": cate,
        "cate_ci_lower": cate - 0.1,
        "cate_ci_upper": cate + 0.1,
        "sample_size": sample_size,
        "statistical_significance": True,
    }


def _segment_profile(
    segment_id: str, responder_type: str, cate: float, size: int = 100
) -> SegmentProfile:
    return {
        "segment_id": segment_id,
        "responder_type": responder_type,  # type: ignore[typeddict-item]
        "cate_estimate": cate,
        "defining_features": [{"variable": "x", "value": "y", "effect_size": 1.0}],
        "size": size,
        "size_percentage": 10.0,
        "recommendation": "test",
    }


def _base_state() -> HeterogeneousOptimizerState:
    """Minimum HeterogeneousOptimizerState satisfying TypedDict shape.

    Required-field discipline: keep only required fields here; tests mutate
    optional fields (``overall_ate``, ``cate_by_segment``, ``status``,
    ``high_responders``, ``low_responders``) per row.
    """
    state: HeterogeneousOptimizerState = {  # type: ignore[typeddict-item]
        "query": "test",
        "treatment_var": "treatment",
        "outcome_var": "outcome",
        "segment_vars": ["seg"],
        "effect_modifiers": [],
        "data_source": "test",
        "filters": None,
        "tier0_data": None,
        "n_estimators": 100,
        "min_samples_leaf": 10,
        "significance_level": 0.05,
        "top_segments_count": 10,
        "cate_by_segment": None,
        "overall_ate": None,
        "heterogeneity_score": None,
        "feature_importance": None,
        "uplift_by_segment": None,
        "overall_auuc": None,
        "overall_qini": None,
        "targeting_efficiency": None,
        "model_type_used": None,
        "uplift_latency_ms": None,
        "hierarchical_segment_results": None,
        "nested_ci": None,
        "segment_heterogeneity_score": None,
        "overall_hierarchical_ate": None,
        "overall_hierarchical_ci_lower": None,
        "overall_hierarchical_ci_upper": None,
        "n_segments_analyzed": None,
        "segmentation_method_used": None,
        "hierarchical_estimator_type": None,
        "hierarchical_latency_ms": None,
        "high_responders": None,
        "low_responders": None,
        "segment_comparison": None,
        "policy_recommendations": None,
        "expected_total_lift": None,
        "optimal_allocation_summary": None,
        "cate_plot_data": None,
        "segment_grid_data": None,
        "executive_summary": None,
        "key_insights": None,
        "strategic_interpretation": None,
        "estimation_latency_ms": 0,
        "analysis_latency_ms": 0,
        "total_latency_ms": 0,
        "errors": [],
        "warnings": [],
        "status": "analyzing",
        "confidence": None,
        "requires_further_analysis": None,
        "suggested_next_agent": None,
        "session_id": None,
        "working_memory_context": None,
        "episodic_context": None,
        "library_execution_plan": None,
        "library_execution_mode": None,
        "primary_library": None,
        "libraries_executed": None,
        "libraries_skipped": None,
        "econml_cate_result": None,
        "econml_model_used": None,
        "econml_latency_ms": None,
        "causalml_uplift_result": None,
        "causalml_model_used": None,
        "causalml_latency_ms": None,
        "cross_library_validation": None,
        "econml_causalml_agreement": None,
        "dowhy_validation_result": None,
        "validation_passed": None,
        "library_consensus_effect": None,
        "library_agreement_score": None,
        "effect_estimate_variance": None,
        "question_type": None,
        "routing_confidence": None,
        "routing_rationale": None,
        "audit_workflow_id": None,
        "discovered_dag_adjacency": None,
        "discovered_dag_nodes": None,
        "discovered_dag_edge_types": None,
        "discovery_gate_decision": None,
        "discovery_gate_confidence": None,
        "dag_validated_segments": None,
        "dag_invalid_segments": None,
        "latent_confounder_segments": None,
        "dag_validation_warnings": None,
    }
    return state


def _state_row1_failed() -> HeterogeneousOptimizerState:
    """Row 1: upstream set status=failed. Downstream must skip unchanged."""
    state = _base_state()
    state["status"] = "failed"
    state["overall_ate"] = None
    state["cate_by_segment"] = None
    state["errors"] = [{"node": "cate_estimator", "error": "upstream synthetic error"}]
    return state


def _state_row2_populated_zero() -> HeterogeneousOptimizerState:
    """Row 2: upstream succeeded with legitimate zero ATE and populated segments.

    This must NOT be confused with row 3 — overall_ate=0.0 is a valid honest
    zero (the treatment genuinely had no effect on average), not absence.
    """
    state = _base_state()
    state["status"] = "analyzing"
    state["overall_ate"] = 0.0
    state["cate_by_segment"] = {
        "seg": [
            _cate_result("seg", "v1", 0.0),
            _cate_result("seg", "v2", 0.0),
        ]
    }
    state["heterogeneity_score"] = 0.0
    # Pre-populate downstream-only inputs so policy_learner and
    # profile_generator can exercise their happy path past segment_analyzer.
    state["high_responders"] = []
    state["low_responders"] = []
    state["segment_comparison"] = {
        "overall_ate": 0.0,
        "high_responder_avg_cate": 0.0,
        "low_responder_avg_cate": 0.0,
        "effect_ratio": 1.0,
        "high_responder_count": 0,
        "low_responder_count": 0,
    }
    return state


def _state_row3_missing() -> HeterogeneousOptimizerState:
    """Row 3: upstream did NOT fail but produced no CATE fields. Must fail-close."""
    state = _base_state()
    state["status"] = "analyzing"  # NOT "failed"; upstream silently swallowed
    state["overall_ate"] = None
    state["cate_by_segment"] = None
    return state


def _state_row5_ate_absent_segments_present() -> HeterogeneousOptimizerState:
    """Row 5: overall_ate is None but cate_by_segment has populated segments.

    Codex iter-0 MEDIUM: previously untested branch of the V-A1/V-A2 matrix.
    Without a baseline ATE, heterogeneity analysis / policy scoring / executive
    summary are undefined — cate_by_segment alone does not redeem missing ATE.
    Must fail-close to prevent the helpers' historical `or 0` conflation from
    silently treating absent ATE as a legitimate zero baseline.
    """
    state = _base_state()
    state["status"] = "analyzing"  # NOT "failed"; upstream silently swallowed ATE
    state["overall_ate"] = None
    state["cate_by_segment"] = {
        "seg": [
            _cate_result("seg", "v1", 0.12),
            _cate_result("seg", "v2", -0.05),
        ]
    }
    # Pre-populate downstream-only inputs so policy_learner / profile_generator
    # would have everything they need EXCEPT the missing overall_ate. This
    # isolates the test to the ATE-absence branch.
    state["high_responders"] = []
    state["low_responders"] = []
    state["segment_comparison"] = {
        "overall_ate": 0.0,
        "high_responder_avg_cate": 0.12,
        "low_responder_avg_cate": -0.05,
        "effect_ratio": 1.0,
        "high_responder_count": 1,
        "low_responder_count": 1,
    }
    return state


def _state_row4_partial() -> HeterogeneousOptimizerState:
    """Row 4: real float ATE but empty cate_by_segment dict.

    Must run with heterogeneity_score=0 (honest empty result) + emit a warning.
    """
    state = _base_state()
    state["status"] = "analyzing"
    state["overall_ate"] = 0.15
    state["cate_by_segment"] = {}
    # Pre-populate so policy_learner / profile_generator can run.
    state["high_responders"] = []
    state["low_responders"] = []
    state["segment_comparison"] = {
        "overall_ate": 0.15,
        "high_responder_avg_cate": 0.0,
        "low_responder_avg_cate": 0.0,
        "effect_ratio": 1.0,
        "high_responder_count": 0,
        "low_responder_count": 0,
    }
    return state


# ---------------------------------------------------------------------------
# SegmentAnalyzerNode disambiguation (rows 1-4)
# ---------------------------------------------------------------------------


class TestSegmentAnalyzerFailClosed:
    """SegmentAnalyzerNode disambiguation matrix."""

    @pytest.mark.asyncio
    async def test_row1_failed_status_preserved(self):
        """Row 1: status=failed → existing skip path preserved (unchanged)."""
        node = SegmentAnalyzerNode()
        state = _state_row1_failed()

        result = await node.execute(state)

        assert result["status"] == "failed"
        # Existing behavior: passes through state untouched (no downstream synthesis).
        assert result.get("high_responders") is None
        assert result.get("low_responders") is None

    @pytest.mark.asyncio
    async def test_row2_populated_zero_runs_honestly(self):
        """Row 2: overall_ate=0.0 (legitimate honest zero) + populated segments runs normally."""
        node = SegmentAnalyzerNode()
        state = _state_row2_populated_zero()

        result = await node.execute(state)

        # Must NOT fail-close on a legitimate honest zero.
        assert result["status"] != "failed"
        # high/low_responders are produced (likely empty given ATE=0, but the
        # node ran rather than fail-closed).
        assert "high_responders" in result
        assert "low_responders" in result

    @pytest.mark.asyncio
    async def test_row3_absent_cate_fails_closed(self):
        """Row 3: overall_ate is None AND cate_by_segment is None → fail-closed."""
        node = SegmentAnalyzerNode()
        state = _state_row3_missing()

        result = await node.execute(state)

        assert result["status"] == "failed", (
            "Must fail-close when upstream produced no CATE fields; "
            "do NOT synthesize neutral output"
        )
        errors = result.get("errors") or []
        assert any(
            "cate_estimator" in (e.get("error", "") or "") or "CATE" in (e.get("error", "") or "")
            for e in errors
        ), f"Error must reference upstream CATE absence; got {errors!r}"
        # Must not have synthesized neutral content.
        assert not result.get("high_responders"), (
            "LABEL-disguised-as-REWIRE: still synthesized empty responders"
        )
        assert not result.get("low_responders"), (
            "LABEL-disguised-as-REWIRE: still synthesized empty responders"
        )

    @pytest.mark.asyncio
    async def test_row4_partial_runs_with_warning(self):
        """Row 4: real ATE + empty cate_by_segment → runs honestly + warning."""
        node = SegmentAnalyzerNode()
        state = _state_row4_partial()

        result = await node.execute(state)

        # Must NOT fail-close (we have a real ATE).
        assert result["status"] != "failed"
        # An honest warning must be emitted that segments are absent.
        warnings = result.get("warnings") or []
        assert any(
            "segment" in (w or "").lower() or "cate" in (w or "").lower() for w in warnings
        ), f"Expected warning about empty cate_by_segment; got {warnings!r}"

    @pytest.mark.asyncio
    async def test_row5_ate_absent_segments_present_fails_closed(self):
        """Row 5: overall_ate=None + populated cate_by_segment → MUST fail-close.

        Previously untested matrix branch (codex iter-0 MEDIUM). The historical
        ``state.get("overall_ate") or 0`` conflation would have silently treated
        missing ATE as 0.0 here, producing a "successful" heterogeneity report.
        """
        node = SegmentAnalyzerNode()
        state = _state_row5_ate_absent_segments_present()

        result = await node.execute(state)

        assert result["status"] == "failed", (
            "Must fail-close when overall_ate is None even if cate_by_segment "
            "is populated; baseline ATE is required for heterogeneity analysis"
        )
        # Must not synthesize responders from segments-without-ATE.
        assert not result.get("high_responders"), (
            "LABEL-disguised-as-REWIRE: still produced responders without baseline ATE"
        )
        assert not result.get("low_responders"), (
            "LABEL-disguised-as-REWIRE: still produced responders without baseline ATE"
        )


# ---------------------------------------------------------------------------
# PolicyLearnerNode disambiguation (rows 1-5)
# ---------------------------------------------------------------------------


class TestPolicyLearnerFailClosed:
    """PolicyLearnerNode disambiguation matrix."""

    @pytest.mark.asyncio
    async def test_row1_failed_status_preserved(self):
        node = PolicyLearnerNode()
        state = _state_row1_failed()

        result = await node.execute(state)

        assert result["status"] == "failed"
        assert result.get("policy_recommendations") is None

    @pytest.mark.asyncio
    async def test_row2_populated_zero_runs_honestly(self):
        node = PolicyLearnerNode()
        state = _state_row2_populated_zero()

        result = await node.execute(state)

        assert result["status"] != "failed"
        assert "policy_recommendations" in result

    @pytest.mark.asyncio
    async def test_row3_absent_cate_fails_closed(self):
        node = PolicyLearnerNode()
        state = _state_row3_missing()

        result = await node.execute(state)

        assert result["status"] == "failed", (
            "PolicyLearner must fail-close when upstream produced no CATE fields"
        )
        errors = result.get("errors") or []
        assert any(
            "cate_estimator" in (e.get("error", "") or "") or "CATE" in (e.get("error", "") or "")
            for e in errors
        ), f"Error must reference CATE absence; got {errors!r}"
        # Must NOT synthesize recommendations.
        recs = result.get("policy_recommendations")
        assert not recs, f"LABEL-disguised-as-REWIRE: still synthesized recommendations: {recs!r}"

    @pytest.mark.asyncio
    async def test_row4_partial_runs_with_warning(self):
        node = PolicyLearnerNode()
        state = _state_row4_partial()

        result = await node.execute(state)

        assert result["status"] != "failed"
        warnings = result.get("warnings") or []
        assert any(
            "segment" in (w or "").lower() or "cate" in (w or "").lower() for w in warnings
        ), f"Expected warning about empty cate_by_segment; got {warnings!r}"

    @pytest.mark.asyncio
    async def test_row5_ate_absent_segments_present_fails_closed(self):
        """Row 5 (codex iter-0 MEDIUM): policy scoring requires baseline ATE."""
        node = PolicyLearnerNode()
        state = _state_row5_ate_absent_segments_present()

        result = await node.execute(state)

        assert result["status"] == "failed", (
            "PolicyLearner must fail-close when overall_ate=None even with "
            "populated cate_by_segment; lift estimates require baseline ATE"
        )
        recs = result.get("policy_recommendations")
        assert not recs, "LABEL-disguised-as-REWIRE: produced recommendations without baseline ATE"


# ---------------------------------------------------------------------------
# ProfileGeneratorNode disambiguation (rows 1-5)
# ---------------------------------------------------------------------------


class TestProfileGeneratorFailClosed:
    """ProfileGeneratorNode disambiguation matrix."""

    @pytest.mark.asyncio
    async def test_row1_failed_status_preserved(self):
        node = ProfileGeneratorNode()
        state = _state_row1_failed()

        result = await node.execute(state)

        assert result["status"] == "failed"
        assert result.get("executive_summary") is None
        assert result.get("key_insights") is None
        assert result.get("strategic_interpretation") is None

    @pytest.mark.asyncio
    async def test_row2_populated_zero_runs_honestly(self):
        node = ProfileGeneratorNode()
        state = _state_row2_populated_zero()

        result = await node.execute(state)

        assert result["status"] != "failed"
        # Honest content emitted (may describe a uniform-effect scenario).
        assert result.get("executive_summary"), "expected executive_summary on honest zero"
        assert result.get("strategic_interpretation"), (
            "expected strategic_interpretation on honest zero"
        )

    @pytest.mark.asyncio
    async def test_row3_absent_cate_fails_closed(self):
        node = ProfileGeneratorNode()
        state = _state_row3_missing()

        result = await node.execute(state)

        assert result["status"] == "failed", (
            "ProfileGenerator must fail-close when upstream produced no CATE fields"
        )
        errors = result.get("errors") or []
        assert any(
            "cate_estimator" in (e.get("error", "") or "") or "CATE" in (e.get("error", "") or "")
            for e in errors
        ), f"Error must reference CATE absence; got {errors!r}"
        # Must NOT synthesize neutral content for the user-visible fields.
        assert not result.get("executive_summary"), (
            "LABEL-disguised-as-REWIRE: synthesized executive_summary"
        )
        assert not result.get("key_insights"), "LABEL-disguised-as-REWIRE: synthesized key_insights"
        assert not result.get("strategic_interpretation"), (
            "LABEL-disguised-as-REWIRE: synthesized strategic_interpretation"
        )

    @pytest.mark.asyncio
    async def test_row4_partial_runs_with_warning(self):
        node = ProfileGeneratorNode()
        state = _state_row4_partial()

        result = await node.execute(state)

        assert result["status"] != "failed"
        warnings = result.get("warnings") or []
        assert any(
            "segment" in (w or "").lower() or "cate" in (w or "").lower() for w in warnings
        ), f"Expected warning about empty cate_by_segment; got {warnings!r}"

    @pytest.mark.asyncio
    async def test_row5_ate_absent_segments_present_fails_closed(self):
        """Row 5 (codex iter-0 MEDIUM): user-visible content requires baseline ATE."""
        node = ProfileGeneratorNode()
        state = _state_row5_ate_absent_segments_present()

        result = await node.execute(state)

        assert result["status"] == "failed", (
            "ProfileGenerator must fail-close when overall_ate=None even with "
            "populated cate_by_segment; executive summary requires baseline ATE"
        )
        assert not result.get("executive_summary"), (
            "LABEL-disguised-as-REWIRE: produced executive_summary without baseline ATE"
        )
        assert not result.get("strategic_interpretation"), (
            "LABEL-disguised-as-REWIRE: produced strategic_interpretation without baseline ATE"
        )
