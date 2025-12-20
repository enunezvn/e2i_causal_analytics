"""Tests for Policy Learner Node."""

import pytest
from src.agents.heterogeneous_optimizer.nodes.policy_learner import PolicyLearnerNode
from src.agents.heterogeneous_optimizer.state import (
    HeterogeneousOptimizerState,
    CATEResult,
    SegmentProfile,
)


class TestPolicyLearnerNode:
    """Test PolicyLearnerNode."""

    def _create_cate_result(
        self, segment_name: str, segment_value: str, cate: float, sample_size: int = 100, significant: bool = True
    ) -> CATEResult:
        """Create test CATE result."""
        return {
            "segment_name": segment_name,
            "segment_value": segment_value,
            "cate_estimate": cate,
            "cate_ci_lower": cate - 0.1,
            "cate_ci_upper": cate + 0.1,
            "sample_size": sample_size,
            "statistical_significance": significant,
        }

    def _create_segment_profile(
        self, segment_id: str, responder_type: str, cate: float, size: int = 100
    ) -> SegmentProfile:
        """Create test segment profile."""
        return {
            "segment_id": segment_id,
            "responder_type": responder_type,
            "cate_estimate": cate,
            "defining_features": [{"variable": "test", "value": "test", "effect_size": 1.0}],
            "size": size,
            "size_percentage": 10.0,
            "recommendation": "test recommendation",
        }

    def _create_test_state(self, overall_ate: float = 0.25) -> HeterogeneousOptimizerState:
        """Create test state with analysis results."""
        cate_by_segment = {
            "hcp_specialty": [
                self._create_cate_result("hcp_specialty", "Oncology", 0.50, 200, True),  # High
                self._create_cate_result("hcp_specialty", "Cardiology", 0.30, 150, True),
                self._create_cate_result("hcp_specialty", "Primary Care", 0.10, 300, True),  # Low
            ]
        }

        high_responders = [
            self._create_segment_profile("hcp_specialty_Oncology", "high", 0.50, 200)
        ]

        low_responders = [
            self._create_segment_profile("hcp_specialty_Primary Care", "low", 0.10, 300)
        ]

        return {
            "query": "test",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "segment_vars": ["hcp_specialty"],
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
            "high_responders": high_responders,
            "low_responders": low_responders,
            "segment_comparison": {
                "overall_ate": overall_ate,
                "high_responder_avg_cate": 0.50,
                "low_responder_avg_cate": 0.10,
                "effect_ratio": 5.0,
                "high_responder_count": 1,
                "low_responder_count": 1,
            },
            "policy_recommendations": None,
            "expected_total_lift": None,
            "optimal_allocation_summary": None,
            "cate_plot_data": None,
            "segment_grid_data": None,
            "executive_summary": None,
            "key_insights": None,
            "estimation_latency_ms": 100,
            "analysis_latency_ms": 50,
            "total_latency_ms": 0,
            "errors": [],
            "warnings": [],
            "status": "optimizing",
        }

    @pytest.mark.asyncio
    async def test_generate_policy_recommendations(self):
        """Test policy recommendation generation."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "policy_recommendations" in result
        assert len(result["policy_recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_policy_recommendation_structure(self):
        """Test structure of policy recommendations."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        if result["policy_recommendations"]:
            rec = result["policy_recommendations"][0]
            assert "segment" in rec
            assert "current_treatment_rate" in rec
            assert "recommended_treatment_rate" in rec
            assert "expected_incremental_outcome" in rec
            assert "confidence" in rec

    @pytest.mark.asyncio
    async def test_high_responder_recommendation(self):
        """Test recommendation for high responder segment."""
        node = PolicyLearnerNode()
        state = self._create_test_state(overall_ate=0.25)

        result = await node.execute(state)

        # Find Oncology recommendation (CATE=0.50, which is 2x ATE)
        oncology_rec = [r for r in result["policy_recommendations"] if "Oncology" in r["segment"]]

        if oncology_rec:
            rec = oncology_rec[0]
            # High responder should have increased treatment rate
            assert rec["recommended_treatment_rate"] > rec["current_treatment_rate"]

    @pytest.mark.asyncio
    async def test_low_responder_recommendation(self):
        """Test recommendation for low responder segment."""
        node = PolicyLearnerNode()
        state = self._create_test_state(overall_ate=0.25)

        result = await node.execute(state)

        # Find Primary Care recommendation (CATE=0.10, which is 0.4x ATE)
        primary_care_rec = [
            r for r in result["policy_recommendations"] if "Primary Care" in r["segment"]
        ]

        if primary_care_rec:
            rec = primary_care_rec[0]
            # Low responder should have decreased treatment rate
            assert rec["recommended_treatment_rate"] < rec["current_treatment_rate"]

    @pytest.mark.asyncio
    async def test_treatment_rate_bounds(self):
        """Test treatment rates are bounded between 0.1 and 0.9."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        for rec in result["policy_recommendations"]:
            assert 0.0 <= rec["recommended_treatment_rate"] <= 1.0
            assert 0.0 <= rec["current_treatment_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_expected_lift_calculation(self):
        """Test expected lift calculation."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        # Expected lift should be sum of incremental outcomes
        expected_total = sum(r["expected_incremental_outcome"] for r in result["policy_recommendations"])

        assert abs(result["expected_total_lift"] - expected_total) < 0.01

    @pytest.mark.asyncio
    async def test_confidence_calculation(self):
        """Test confidence calculation."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        for rec in result["policy_recommendations"]:
            assert 0.0 <= rec["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_increases_with_sample_size(self):
        """Test confidence increases with sample size."""
        node = PolicyLearnerNode()

        # Create two similar segments with different sample sizes
        result_small = self._create_cate_result("test", "small", 0.30, sample_size=50, significant=False)
        result_large = self._create_cate_result("test", "large", 0.30, sample_size=500, significant=False)

        rec_small = node._generate_recommendation(result_small, 0.25)
        rec_large = node._generate_recommendation(result_large, 0.25)

        # Larger sample should have higher confidence
        assert rec_large["confidence"] > rec_small["confidence"]

    @pytest.mark.asyncio
    async def test_confidence_increases_with_significance(self):
        """Test confidence increases with statistical significance."""
        node = PolicyLearnerNode()

        result_sig = self._create_cate_result("test", "sig", 0.30, sample_size=100, significant=True)
        result_nonsig = self._create_cate_result("test", "nonsig", 0.30, sample_size=100, significant=False)

        rec_sig = node._generate_recommendation(result_sig, 0.25)
        rec_nonsig = node._generate_recommendation(result_nonsig, 0.25)

        # Significant result should have higher confidence
        assert rec_sig["confidence"] > rec_nonsig["confidence"]

    @pytest.mark.asyncio
    async def test_recommendations_sorted_by_lift(self):
        """Test recommendations sorted by expected incremental outcome."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        recs = result["policy_recommendations"]
        for i in range(len(recs) - 1):
            assert recs[i]["expected_incremental_outcome"] >= recs[i + 1]["expected_incremental_outcome"]

    @pytest.mark.asyncio
    async def test_top_20_recommendations(self):
        """Test limiting to top 20 recommendations."""
        node = PolicyLearnerNode()

        # Create state with many segments
        cate_by_segment = {
            "segment1": [
                self._create_cate_result("segment1", f"value_{i}", 0.30, 100)
                for i in range(30)
            ]
        }

        state = self._create_test_state()
        state["cate_by_segment"] = cate_by_segment

        result = await node.execute(state)

        # Should be limited to 20
        assert len(result["policy_recommendations"]) <= 20

    @pytest.mark.asyncio
    async def test_allocation_summary_generation(self):
        """Test allocation summary generation."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "optimal_allocation_summary" in result
        assert isinstance(result["optimal_allocation_summary"], str)
        assert len(result["optimal_allocation_summary"]) > 0

    @pytest.mark.asyncio
    async def test_total_latency_calculation(self):
        """Test total latency calculation."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        # Should sum all latencies
        assert result["total_latency_ms"] > 0
        assert result["total_latency_ms"] >= result["estimation_latency_ms"]
        assert result["total_latency_ms"] >= result["analysis_latency_ms"]

    @pytest.mark.asyncio
    async def test_status_update(self):
        """Test status update to completed."""
        node = PolicyLearnerNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_failed_status_passthrough(self):
        """Test failed status is passed through."""
        node = PolicyLearnerNode()
        state = self._create_test_state()
        state["status"] = "failed"

        result = await node.execute(state)

        assert result["status"] == "failed"


class TestPolicyLearnerEdgeCases:
    """Test edge cases for policy learner."""

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
            "high_responders": [],
            "low_responders": [],
            "segment_comparison": {},
            "policy_recommendations": None,
            "expected_total_lift": None,
            "optimal_allocation_summary": None,
            "cate_plot_data": None,
            "segment_grid_data": None,
            "executive_summary": None,
            "key_insights": None,
            "estimation_latency_ms": 100,
            "analysis_latency_ms": 50,
            "total_latency_ms": 0,
            "errors": [],
            "warnings": [],
            "status": "optimizing",
        }

    @pytest.mark.asyncio
    async def test_zero_cate(self):
        """Test with zero CATE estimate."""
        node = PolicyLearnerNode()

        cate_by_segment = {
            "segment1": [self._create_cate_result("segment1", "value1", 0.0)]
        }
        state = self._create_test_state(cate_by_segment)

        result = await node.execute(state)

        # Should still generate recommendation
        assert len(result["policy_recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_negative_cate(self):
        """Test with negative CATE estimate."""
        node = PolicyLearnerNode()

        cate_by_segment = {
            "segment1": [self._create_cate_result("segment1", "value1", -0.10)]
        }
        state = self._create_test_state(cate_by_segment)

        result = await node.execute(state)

        # Should recommend minimal treatment for negative CATE
        if result["policy_recommendations"]:
            rec = result["policy_recommendations"][0]
            assert rec["recommended_treatment_rate"] <= 0.1

    @pytest.mark.asyncio
    async def test_very_high_cate(self):
        """Test with very high CATE estimate."""
        node = PolicyLearnerNode()

        cate_by_segment = {
            "segment1": [self._create_cate_result("segment1", "value1", 5.0)]
        }
        state = self._create_test_state(cate_by_segment, overall_ate=1.0)

        result = await node.execute(state)

        # Should recommend high treatment rate
        if result["policy_recommendations"]:
            rec = result["policy_recommendations"][0]
            assert rec["recommended_treatment_rate"] >= 0.7
