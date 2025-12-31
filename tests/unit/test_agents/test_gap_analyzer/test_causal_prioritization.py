"""Tests for Causal Prioritization in Gap Analyzer (V4.4).

Tests the causal evidence filtering methods in PrioritizerNode:
- _build_causal_feature_lookup
- _apply_causal_evidence_adjustments
- _has_causal_evidence
- Integration with execute method
"""

import pytest

from src.agents.gap_analyzer.nodes.prioritizer import (
    DIRECT_CAUSE_BOOST,
    HIGH_CAUSAL_SCORE_THRESHOLD,
    NO_CAUSAL_EVIDENCE_PENALTY,
    PrioritizerNode,
)
from src.agents.gap_analyzer.state import (
    GapAnalyzerState,
    PerformanceGap,
    ROIEstimate,
)


class TestCausalFeatureLookup:
    """Test _build_causal_feature_lookup method."""

    def test_build_lookup_from_rankings(self):
        """Test building feature lookup from causal rankings."""
        node = PrioritizerNode()

        causal_rankings = [
            {
                "feature_name": "trx",
                "causal_rank": 1,
                "predictive_rank": 2,
                "causal_score": 0.85,
                "predictive_score": 0.70,
                "rank_difference": -1,
                "is_direct_cause": True,
                "path_length": 1,
            },
            {
                "feature_name": "market_share",
                "causal_rank": 2,
                "predictive_rank": 1,
                "causal_score": 0.65,
                "predictive_score": 0.90,
                "rank_difference": 1,
                "is_direct_cause": False,
                "path_length": 2,
            },
        ]

        lookup = node._build_causal_feature_lookup(causal_rankings)

        assert "trx" in lookup
        assert "market_share" in lookup
        assert lookup["trx"]["causal_rank"] == 1
        assert lookup["trx"]["is_direct_cause"] is True
        assert lookup["market_share"]["causal_score"] == 0.65

    def test_build_lookup_empty_rankings(self):
        """Test building lookup from empty rankings."""
        node = PrioritizerNode()

        lookup = node._build_causal_feature_lookup([])

        assert lookup == {}

    def test_build_lookup_missing_feature_name(self):
        """Test that rankings without feature_name are skipped."""
        node = PrioritizerNode()

        causal_rankings = [
            {"feature_name": "trx", "causal_score": 0.8},
            {"causal_score": 0.5},  # Missing feature_name
            {"feature_name": "", "causal_score": 0.3},  # Empty feature_name
        ]

        lookup = node._build_causal_feature_lookup(causal_rankings)

        assert len(lookup) == 1
        assert "trx" in lookup


class TestCausalEvidenceAdjustments:
    """Test _apply_causal_evidence_adjustments method."""

    def _create_test_gap(self, metric: str = "trx") -> PerformanceGap:
        """Create test gap."""
        return {
            "gap_id": f"region_Northeast_{metric}_vs_target",
            "metric": metric,
            "segment": "region",
            "segment_value": "Northeast",
            "current_value": 400.0,
            "target_value": 500.0,
            "gap_size": 100.0,
            "gap_percentage": 20.0,
            "gap_type": "vs_target",
        }

    def _create_test_roi(self, gap_id: str, roi: float = 3.0) -> ROIEstimate:
        """Create test ROI estimate."""
        return {
            "gap_id": gap_id,
            "estimated_revenue_impact": 40000.0,
            "estimated_cost_to_close": 10000.0,
            "expected_roi": roi,
            "payback_period_months": 6,
            "confidence": 0.8,
            "assumptions": ["Test assumption"],
        }

    def _create_opportunity(self, metric: str = "trx", roi: float = 3.0):
        """Create test opportunity."""
        gap = self._create_test_gap(metric=metric)
        roi_estimate = self._create_test_roi(gap["gap_id"], roi=roi)
        return {
            "rank": 1,
            "gap": gap,
            "roi_estimate": roi_estimate,
            "recommended_action": "Test action",
            "implementation_difficulty": "medium",
            "time_to_impact": "3-6 months",
        }

    def test_direct_cause_boost(self):
        """Test that direct causes get ROI boost."""
        node = PrioritizerNode()

        opportunities = [self._create_opportunity(metric="trx", roi=3.0)]

        causal_lookup = {
            "trx": {
                "causal_score": 0.9,
                "is_direct_cause": True,
            }
        }

        adjusted, warnings = node._apply_causal_evidence_adjustments(
            opportunities,
            causal_lookup,
            direct_cause_features=["trx"],
            predictive_only_features=[],
        )

        assert len(adjusted) == 1
        adjusted_roi = adjusted[0]["roi_estimate"]["expected_roi"]
        assert adjusted_roi == 3.0 * DIRECT_CAUSE_BOOST
        assert adjusted[0]["roi_estimate"]["causal_adjustment_reason"] == "direct_cause_boost"
        assert len(warnings) == 0

    def test_predictive_only_penalty(self):
        """Test that predictive-only features get ROI penalty."""
        node = PrioritizerNode()

        opportunities = [self._create_opportunity(metric="market_share", roi=4.0)]

        causal_lookup = {
            "market_share": {
                "causal_score": 0.3,  # Low causal score
                "is_direct_cause": False,
            }
        }

        adjusted, warnings = node._apply_causal_evidence_adjustments(
            opportunities,
            causal_lookup,
            direct_cause_features=[],
            predictive_only_features=["market_share"],
        )

        assert len(adjusted) == 1
        adjusted_roi = adjusted[0]["roi_estimate"]["expected_roi"]
        assert adjusted_roi == 4.0 * NO_CAUSAL_EVIDENCE_PENALTY
        assert adjusted[0]["roi_estimate"]["causal_adjustment_reason"] == "predictive_only_penalty"
        assert len(warnings) == 1
        assert "lacks causal evidence" in warnings[0]

    def test_high_causal_score_boost(self):
        """Test that high causal score features get boost."""
        node = PrioritizerNode()

        opportunities = [self._create_opportunity(metric="nrx", roi=2.0)]

        # High causal score but not direct cause
        causal_lookup = {
            "nrx": {
                "causal_score": 0.8,  # Above threshold
                "is_direct_cause": False,
            }
        }

        adjusted, warnings = node._apply_causal_evidence_adjustments(
            opportunities,
            causal_lookup,
            direct_cause_features=[],
            predictive_only_features=[],
        )

        assert len(adjusted) == 1
        adjusted_roi = adjusted[0]["roi_estimate"]["expected_roi"]
        # Boost = 1.0 + (0.8 - 0.6) * 0.5 = 1.1
        expected_boost = 1.0 + (0.8 - HIGH_CAUSAL_SCORE_THRESHOLD) * 0.5
        assert adjusted_roi == pytest.approx(2.0 * expected_boost)
        assert adjusted[0]["roi_estimate"]["causal_adjustment_reason"] == "high_causal_score"

    def test_no_causal_info_warning(self):
        """Test warning when no causal info available for feature."""
        node = PrioritizerNode()

        opportunities = [self._create_opportunity(metric="conversion_rate", roi=2.5)]

        causal_lookup = {}  # No causal info

        adjusted, warnings = node._apply_causal_evidence_adjustments(
            opportunities,
            causal_lookup,
            direct_cause_features=[],
            predictive_only_features=[],
        )

        assert len(adjusted) == 1
        # ROI should be unchanged (no adjustment)
        assert adjusted[0]["roi_estimate"]["expected_roi"] == 2.5
        assert len(warnings) == 1
        assert "no causal analysis available" in warnings[0]

    def test_multiple_opportunities_mixed_adjustments(self):
        """Test multiple opportunities with different adjustments."""
        node = PrioritizerNode()

        opportunities = [
            self._create_opportunity(metric="trx", roi=3.0),  # Direct cause
            self._create_opportunity(metric="market_share", roi=4.0),  # Predictive only
            self._create_opportunity(metric="nrx", roi=2.0),  # High causal score
        ]

        causal_lookup = {
            "trx": {"causal_score": 0.9, "is_direct_cause": True},
            "market_share": {"causal_score": 0.3, "is_direct_cause": False},
            "nrx": {"causal_score": 0.75, "is_direct_cause": False},
        }

        adjusted, warnings = node._apply_causal_evidence_adjustments(
            opportunities,
            causal_lookup,
            direct_cause_features=["trx"],
            predictive_only_features=["market_share"],
        )

        assert len(adjusted) == 3

        # Check trx got boost
        trx_opp = next(o for o in adjusted if o["gap"]["metric"] == "trx")
        assert trx_opp["roi_estimate"]["expected_roi"] == 3.0 * DIRECT_CAUSE_BOOST

        # Check market_share got penalty
        ms_opp = next(o for o in adjusted if o["gap"]["metric"] == "market_share")
        assert ms_opp["roi_estimate"]["expected_roi"] == 4.0 * NO_CAUSAL_EVIDENCE_PENALTY

        # Check nrx got high causal score boost
        nrx_opp = next(o for o in adjusted if o["gap"]["metric"] == "nrx")
        expected_boost = 1.0 + (0.75 - HIGH_CAUSAL_SCORE_THRESHOLD) * 0.5
        assert nrx_opp["roi_estimate"]["expected_roi"] == pytest.approx(2.0 * expected_boost)


class TestHasCausalEvidence:
    """Test _has_causal_evidence method."""

    def _create_base_state(self) -> GapAnalyzerState:
        """Create base state without causal fields."""
        return {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "time_period": "current_quarter",
            "filters": None,
            "gap_type": "vs_target",
            "min_gap_threshold": 5.0,
            "max_opportunities": 10,
            "gaps_detected": [],
            "gaps_by_segment": None,
            "total_gap_value": 0.0,
            "roi_estimates": [],
            "total_addressable_value": 0.0,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            "detection_latency_ms": 0,
            "roi_latency_ms": 0,
            "total_latency_ms": 0,
            "segments_analyzed": 0,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

    def test_has_causal_evidence_accept(self):
        """Test causal evidence available with accept decision."""
        node = PrioritizerNode()
        state = self._create_base_state()
        state["causal_rankings"] = [{"feature_name": "trx", "causal_score": 0.8}]
        state["discovery_gate_decision"] = "accept"

        assert node._has_causal_evidence(state) is True

    def test_has_causal_evidence_review(self):
        """Test causal evidence available with review decision."""
        node = PrioritizerNode()
        state = self._create_base_state()
        state["causal_rankings"] = [{"feature_name": "trx", "causal_score": 0.8}]
        state["discovery_gate_decision"] = "review"

        assert node._has_causal_evidence(state) is True

    def test_no_causal_evidence_reject(self):
        """Test no causal evidence with reject decision."""
        node = PrioritizerNode()
        state = self._create_base_state()
        state["causal_rankings"] = [{"feature_name": "trx", "causal_score": 0.8}]
        state["discovery_gate_decision"] = "reject"

        assert node._has_causal_evidence(state) is False

    def test_no_causal_evidence_empty_rankings(self):
        """Test no causal evidence with empty rankings."""
        node = PrioritizerNode()
        state = self._create_base_state()
        state["causal_rankings"] = []
        state["discovery_gate_decision"] = "accept"

        assert node._has_causal_evidence(state) is False

    def test_no_causal_evidence_missing_fields(self):
        """Test no causal evidence when fields are missing."""
        node = PrioritizerNode()
        state = self._create_base_state()
        # No causal_rankings or discovery_gate_decision

        assert node._has_causal_evidence(state) is False


class TestCausalPrioritizationIntegration:
    """Integration tests for causal prioritization in execute method."""

    def _create_test_gap(self, metric: str = "trx") -> PerformanceGap:
        """Create test gap."""
        return {
            "gap_id": f"region_Northeast_{metric}_vs_target",
            "metric": metric,
            "segment": "region",
            "segment_value": "Northeast",
            "current_value": 400.0,
            "target_value": 500.0,
            "gap_size": 100.0,
            "gap_percentage": 20.0,
            "gap_type": "vs_target",
        }

    def _create_test_roi(self, gap_id: str, roi: float = 3.0) -> ROIEstimate:
        """Create test ROI estimate."""
        return {
            "gap_id": gap_id,
            "estimated_revenue_impact": 40000.0,
            "estimated_cost_to_close": 10000.0,
            "expected_roi": roi,
            "payback_period_months": 6,
            "confidence": 0.8,
            "assumptions": ["Test assumption"],
        }

    def _create_test_state_with_causal(
        self,
        gaps: list,
        roi_estimates: list,
        causal_rankings: list = None,
        discovery_gate_decision: str = None,
        direct_cause_features: list = None,
        predictive_only_features: list = None,
    ) -> GapAnalyzerState:
        """Create test state with causal discovery fields."""
        return {
            "query": "test",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "kisqali",
            "time_period": "current_quarter",
            "filters": None,
            "gap_type": "vs_target",
            "min_gap_threshold": 5.0,
            "max_opportunities": 10,
            "gaps_detected": gaps,
            "gaps_by_segment": None,
            "total_gap_value": 1000.0,
            "roi_estimates": roi_estimates,
            "total_addressable_value": 100000.0,
            "prioritized_opportunities": None,
            "quick_wins": None,
            "strategic_bets": None,
            "executive_summary": None,
            "key_insights": None,
            "detection_latency_ms": 100,
            "roi_latency_ms": 50,
            "total_latency_ms": 0,
            "segments_analyzed": 1,
            "errors": [],
            "warnings": [],
            "status": "prioritizing",
            # V4.4 causal discovery fields
            "causal_rankings": causal_rankings,
            "discovery_gate_decision": discovery_gate_decision,
            "direct_cause_features": direct_cause_features or [],
            "predictive_only_features": predictive_only_features or [],
        }

    @pytest.mark.asyncio
    async def test_execute_with_causal_evidence(self):
        """Test execute applies causal adjustments when evidence available."""
        node = PrioritizerNode()

        gap = self._create_test_gap(metric="trx")
        roi = self._create_test_roi(gap["gap_id"], roi=3.0)

        causal_rankings = [
            {
                "feature_name": "trx",
                "causal_rank": 1,
                "causal_score": 0.9,
                "is_direct_cause": True,
            }
        ]

        state = self._create_test_state_with_causal(
            gaps=[gap],
            roi_estimates=[roi],
            causal_rankings=causal_rankings,
            discovery_gate_decision="accept",
            direct_cause_features=["trx"],
        )

        result = await node.execute(state)

        assert result["status"] == "completed"
        assert len(result["prioritized_opportunities"]) == 1

        # Check that ROI was boosted
        opp = result["prioritized_opportunities"][0]
        assert opp["roi_estimate"]["expected_roi"] == 3.0 * DIRECT_CAUSE_BOOST
        assert opp["roi_estimate"]["causal_adjustment_factor"] == DIRECT_CAUSE_BOOST

    @pytest.mark.asyncio
    async def test_execute_without_causal_evidence(self):
        """Test execute skips causal adjustments when no evidence."""
        node = PrioritizerNode()

        gap = self._create_test_gap(metric="trx")
        roi = self._create_test_roi(gap["gap_id"], roi=3.0)

        state = self._create_test_state_with_causal(
            gaps=[gap],
            roi_estimates=[roi],
            causal_rankings=None,
            discovery_gate_decision=None,
        )

        result = await node.execute(state)

        assert result["status"] == "completed"
        assert len(result["prioritized_opportunities"]) == 1

        # ROI should be unchanged
        opp = result["prioritized_opportunities"][0]
        assert opp["roi_estimate"]["expected_roi"] == 3.0
        assert "causal_adjustment_factor" not in opp["roi_estimate"]

    @pytest.mark.asyncio
    async def test_execute_reranks_after_causal_adjustment(self):
        """Test that opportunities are re-ranked after causal adjustment."""
        node = PrioritizerNode()

        gap1 = self._create_test_gap(metric="trx")
        gap1["gap_id"] = "region_Northeast_trx_vs_target"
        roi1 = self._create_test_roi(gap1["gap_id"], roi=3.0)  # Will be boosted to 3.6

        gap2 = self._create_test_gap(metric="market_share")
        gap2["gap_id"] = "region_Northeast_market_share_vs_target"
        roi2 = self._create_test_roi(gap2["gap_id"], roi=4.0)  # Will be penalized to 2.8

        causal_rankings = [
            {"feature_name": "trx", "causal_score": 0.9, "is_direct_cause": True},
            {"feature_name": "market_share", "causal_score": 0.3, "is_direct_cause": False},
        ]

        state = self._create_test_state_with_causal(
            gaps=[gap1, gap2],
            roi_estimates=[roi1, roi2],
            causal_rankings=causal_rankings,
            discovery_gate_decision="accept",
            direct_cause_features=["trx"],
            predictive_only_features=["market_share"],
        )

        result = await node.execute(state)

        opps = result["prioritized_opportunities"]
        assert len(opps) == 2

        # After adjustment: trx = 3.6, market_share = 2.8
        # trx should be ranked higher
        assert opps[0]["gap"]["metric"] == "trx"
        assert opps[0]["rank"] == 1
        assert opps[1]["gap"]["metric"] == "market_share"
        assert opps[1]["rank"] == 2

    @pytest.mark.asyncio
    async def test_execute_returns_causal_warnings(self):
        """Test that causal evidence warnings are returned."""
        node = PrioritizerNode()

        gap = self._create_test_gap(metric="market_share")
        roi = self._create_test_roi(gap["gap_id"], roi=4.0)

        causal_rankings = [
            {"feature_name": "market_share", "causal_score": 0.3, "is_direct_cause": False}
        ]

        state = self._create_test_state_with_causal(
            gaps=[gap],
            roi_estimates=[roi],
            causal_rankings=causal_rankings,
            discovery_gate_decision="accept",
            predictive_only_features=["market_share"],
        )

        result = await node.execute(state)

        assert "causal_evidence_warnings" in result
        assert len(result["causal_evidence_warnings"]) == 1
        assert "lacks causal evidence" in result["causal_evidence_warnings"][0]

    @pytest.mark.asyncio
    async def test_execute_with_rejected_discovery(self):
        """Test that rejected discovery gate prevents causal adjustments."""
        node = PrioritizerNode()

        gap = self._create_test_gap(metric="trx")
        roi = self._create_test_roi(gap["gap_id"], roi=3.0)

        causal_rankings = [
            {"feature_name": "trx", "causal_score": 0.9, "is_direct_cause": True}
        ]

        state = self._create_test_state_with_causal(
            gaps=[gap],
            roi_estimates=[roi],
            causal_rankings=causal_rankings,
            discovery_gate_decision="reject",  # Rejected
            direct_cause_features=["trx"],
        )

        result = await node.execute(state)

        # ROI should be unchanged due to rejected gate
        opp = result["prioritized_opportunities"][0]
        assert opp["roi_estimate"]["expected_roi"] == 3.0
        assert "causal_adjustment_factor" not in opp["roi_estimate"]
