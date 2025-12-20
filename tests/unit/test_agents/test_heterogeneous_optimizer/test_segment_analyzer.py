"""Tests for Segment Analyzer Node."""

import pytest
from src.agents.heterogeneous_optimizer.nodes.segment_analyzer import SegmentAnalyzerNode
from src.agents.heterogeneous_optimizer.state import HeterogeneousOptimizerState, CATEResult


class TestSegmentAnalyzerNode:
    """Test SegmentAnalyzerNode."""

    def _create_cate_result(
        self, segment_name: str, segment_value: str, cate: float, sample_size: int = 100
    ) -> CATEResult:
        """Create test CATE result."""
        return {
            "segment_name": segment_name,
            "segment_value": segment_value,
            "cate_estimate": cate,
            "cate_ci_lower": cate - 0.1,
            "cate_ci_upper": cate + 0.1,
            "sample_size": sample_size,
            "statistical_significance": True,
        }

    def _create_test_state(self, overall_ate: float = 0.25) -> HeterogeneousOptimizerState:
        """Create test state with CATE results."""
        cate_by_segment = {
            "hcp_specialty": [
                self._create_cate_result("hcp_specialty", "Oncology", 0.50, 200),  # High
                self._create_cate_result("hcp_specialty", "Cardiology", 0.30, 150),
                self._create_cate_result("hcp_specialty", "Primary Care", 0.10, 300),  # Low
            ],
            "region": [
                self._create_cate_result("region", "Northeast", 0.40, 250),  # High
                self._create_cate_result("region", "West", 0.25, 200),
                self._create_cate_result("region", "Southeast", 0.12, 180),  # Low
            ],
        }

        return {
            "query": "test",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "segment_vars": ["hcp_specialty", "region"],
            "effect_modifiers": ["modifier1"],
            "data_source": "test",
            "filters": None,
            "n_estimators": 100,
            "min_samples_leaf": 10,
            "significance_level": 0.05,
            "top_segments_count": 10,
            "cate_by_segment": cate_by_segment,
            "overall_ate": overall_ate,
            "heterogeneity_score": 0.6,
            "feature_importance": {"modifier1": 0.5},
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
            "estimation_latency_ms": 100,
            "analysis_latency_ms": 0,
            "total_latency_ms": 0,
            "errors": [],
            "warnings": [],
            "status": "analyzing",
        }

    @pytest.mark.asyncio
    async def test_identify_high_responders(self):
        """Test identification of high responder segments."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state(overall_ate=0.25)

        result = await node.execute(state)

        assert "high_responders" in result
        assert len(result["high_responders"]) > 0

        # Check high responders meet threshold (>= 1.5x ATE)
        for responder in result["high_responders"]:
            assert responder["responder_type"] == "high"
            assert responder["cate_estimate"] >= 0.25 * 1.5

    @pytest.mark.asyncio
    async def test_identify_low_responders(self):
        """Test identification of low responder segments."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state(overall_ate=0.25)

        result = await node.execute(state)

        assert "low_responders" in result
        assert len(result["low_responders"]) > 0

        # Check low responders meet threshold (<= 0.5x ATE)
        for responder in result["low_responders"]:
            assert responder["responder_type"] == "low"
            assert responder["cate_estimate"] <= 0.25 * 0.5

    @pytest.mark.asyncio
    async def test_segment_profile_structure(self):
        """Test structure of segment profiles."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        # Check high responder structure
        if result["high_responders"]:
            profile = result["high_responders"][0]
            assert "segment_id" in profile
            assert "responder_type" in profile
            assert "cate_estimate" in profile
            assert "defining_features" in profile
            assert "size" in profile
            assert "size_percentage" in profile
            assert "recommendation" in profile

    @pytest.mark.asyncio
    async def test_segment_comparison(self):
        """Test segment comparison creation."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "segment_comparison" in result
        comparison = result["segment_comparison"]

        assert "overall_ate" in comparison
        assert "high_responder_avg_cate" in comparison
        assert "low_responder_avg_cate" in comparison
        assert "effect_ratio" in comparison
        assert "high_responder_count" in comparison
        assert "low_responder_count" in comparison

    @pytest.mark.asyncio
    async def test_effect_ratio_calculation(self):
        """Test effect ratio calculation."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        comparison = result["segment_comparison"]
        high_avg = comparison["high_responder_avg_cate"]
        low_avg = comparison["low_responder_avg_cate"]

        if low_avg != 0:
            expected_ratio = high_avg / low_avg
            assert abs(comparison["effect_ratio"] - expected_ratio) < 0.01

    @pytest.mark.asyncio
    async def test_top_segments_limit(self):
        """Test top segments count limit."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state()
        state["top_segments_count"] = 2

        result = await node.execute(state)

        # Should limit to top 2
        assert len(result["high_responders"]) <= 2
        assert len(result["low_responders"]) <= 2

    @pytest.mark.asyncio
    async def test_high_responders_sorted(self):
        """Test high responders sorted by CATE."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        high_responders = result["high_responders"]
        for i in range(len(high_responders) - 1):
            assert high_responders[i]["cate_estimate"] >= high_responders[i + 1]["cate_estimate"]

    @pytest.mark.asyncio
    async def test_low_responders_sorted(self):
        """Test low responders sorted by CATE."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        low_responders = result["low_responders"]
        for i in range(len(low_responders) - 1):
            assert low_responders[i]["cate_estimate"] <= low_responders[i + 1]["cate_estimate"]

    @pytest.mark.asyncio
    async def test_size_percentage_calculation(self):
        """Test size percentage calculation."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        # Total size from all segments
        total_size = sum(
            r["sample_size"]
            for results in state["cate_by_segment"].values()
            for r in results
        )

        # Check percentages sum to ~100% (accounting for filtering)
        for profile in result["high_responders"] + result["low_responders"]:
            expected_pct = profile["size"] / total_size * 100
            assert abs(profile["size_percentage"] - expected_pct) < 0.01

    @pytest.mark.asyncio
    async def test_recommendation_generation_high(self):
        """Test recommendation generation for high responders."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        for profile in result["high_responders"]:
            rec = profile["recommendation"]
            assert "Prioritize treatment" in rec
            assert "High response expected" in rec

    @pytest.mark.asyncio
    async def test_recommendation_generation_low(self):
        """Test recommendation generation for low responders."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        for profile in result["low_responders"]:
            rec = profile["recommendation"]
            assert "De-prioritize treatment" in rec
            assert "alternative interventions" in rec

    @pytest.mark.asyncio
    async def test_analysis_latency(self):
        """Test latency measurement."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "analysis_latency_ms" in result
        assert result["analysis_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_status_update(self):
        """Test status update to optimizing."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert result["status"] == "optimizing"

    @pytest.mark.asyncio
    async def test_failed_status_passthrough(self):
        """Test failed status is passed through."""
        node = SegmentAnalyzerNode()
        state = self._create_test_state()
        state["status"] = "failed"

        result = await node.execute(state)

        assert result["status"] == "failed"


class TestSegmentAnalyzerEdgeCases:
    """Test edge cases for segment analyzer."""

    def _create_cate_result(self, segment_name: str, segment_value: str, cate: float):
        """Create test CATE result."""
        return {
            "segment_name": segment_name,
            "segment_value": segment_value,
            "cate_estimate": cate,
            "cate_ci_lower": cate - 0.1,
            "cate_ci_upper": cate + 0.1,
            "sample_size": 100,
            "statistical_significance": True,
        }

    def _create_test_state(self, cate_by_segment, overall_ate=0.25):
        """Create test state."""
        return {
            "query": "test",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "segment_vars": list(cate_by_segment.keys()),
            "effect_modifiers": ["modifier1"],
            "data_source": "test",
            "filters": None,
            "n_estimators": 100,
            "min_samples_leaf": 10,
            "significance_level": 0.05,
            "top_segments_count": 10,
            "cate_by_segment": cate_by_segment,
            "overall_ate": overall_ate,
            "heterogeneity_score": 0.6,
            "feature_importance": {},
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
            "estimation_latency_ms": 100,
            "analysis_latency_ms": 0,
            "total_latency_ms": 0,
            "errors": [],
            "warnings": [],
            "status": "analyzing",
        }

    @pytest.mark.asyncio
    async def test_no_high_responders(self):
        """Test when no segments qualify as high responders."""
        node = SegmentAnalyzerNode()

        # All CATE values below high threshold
        cate_by_segment = {
            "segment1": [
                self._create_cate_result("segment1", "value1", 0.20),
                self._create_cate_result("segment1", "value2", 0.15),
            ]
        }
        state = self._create_test_state(cate_by_segment, overall_ate=0.25)

        result = await node.execute(state)

        assert len(result["high_responders"]) == 0

    @pytest.mark.asyncio
    async def test_no_low_responders(self):
        """Test when no segments qualify as low responders."""
        node = SegmentAnalyzerNode()

        # All CATE values above low threshold
        cate_by_segment = {
            "segment1": [
                self._create_cate_result("segment1", "value1", 0.40),
                self._create_cate_result("segment1", "value2", 0.35),
            ]
        }
        state = self._create_test_state(cate_by_segment, overall_ate=0.25)

        result = await node.execute(state)

        assert len(result["low_responders"]) == 0

    @pytest.mark.asyncio
    async def test_zero_ate(self):
        """Test with zero ATE."""
        node = SegmentAnalyzerNode()

        cate_by_segment = {
            "segment1": [
                self._create_cate_result("segment1", "value1", 0.10),
            ]
        }
        state = self._create_test_state(cate_by_segment, overall_ate=0.0)

        result = await node.execute(state)

        # With zero ATE, no segments should qualify
        assert len(result["high_responders"]) == 0
        assert len(result["low_responders"]) == 0

    @pytest.mark.asyncio
    async def test_negative_ate(self):
        """Test with negative ATE."""
        node = SegmentAnalyzerNode()

        cate_by_segment = {
            "segment1": [
                self._create_cate_result("segment1", "value1", -0.10),
            ]
        }
        state = self._create_test_state(cate_by_segment, overall_ate=-0.25)

        result = await node.execute(state)

        # With negative ATE, logic should still work
        # (though no segments will qualify since ATE > 0 is required)
        assert len(result["high_responders"]) == 0
        assert len(result["low_responders"]) == 0
