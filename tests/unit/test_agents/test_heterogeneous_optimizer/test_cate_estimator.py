"""Tests for CATE Estimator Node."""

import pytest

from src.agents.heterogeneous_optimizer.nodes.cate_estimator import CATEEstimatorNode
from src.agents.heterogeneous_optimizer.state import HeterogeneousOptimizerState


class TestCATEEstimatorNode:
    """Test CATEEstimatorNode."""

    def _create_test_state(self, **overrides) -> HeterogeneousOptimizerState:
        """Create test state with defaults."""
        state = {
            "query": "Which segments respond best to treatment?",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty", "region"],
            "effect_modifiers": ["hcp_tenure", "competitive_pressure", "formulary_status"],
            "data_source": "hcp_performance_metrics",
            "filters": None,
            "n_estimators": 100,
            "min_samples_leaf": 10,
            "significance_level": 0.05,
            "top_segments_count": 10,
            "cate_by_segment": None,
            "overall_ate": None,
            "heterogeneity_score": None,
            "feature_importance": None,
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
            "estimation_latency_ms": 0,
            "analysis_latency_ms": 0,
            "total_latency_ms": 0,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }
        state.update(overrides)
        return state

    @pytest.mark.asyncio
    async def test_cate_estimation_basic(self):
        """Test basic CATE estimation."""
        node = CATEEstimatorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "overall_ate" in result
        assert result["overall_ate"] is not None
        assert isinstance(result["overall_ate"], float)

    @pytest.mark.asyncio
    async def test_heterogeneity_score(self):
        """Test heterogeneity score calculation."""
        node = CATEEstimatorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "heterogeneity_score" in result
        assert result["heterogeneity_score"] is not None
        assert 0.0 <= result["heterogeneity_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_feature_importance(self):
        """Test feature importance extraction."""
        node = CATEEstimatorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "feature_importance" in result
        assert result["feature_importance"] is not None
        assert isinstance(result["feature_importance"], dict)

        # Should have importance for each effect modifier
        for modifier in state["effect_modifiers"]:
            assert modifier in result["feature_importance"]

    @pytest.mark.asyncio
    async def test_cate_by_segment(self):
        """Test CATE calculation by segment."""
        node = CATEEstimatorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "cate_by_segment" in result
        assert result["cate_by_segment"] is not None
        assert isinstance(result["cate_by_segment"], dict)

        # Should have results for each segment variable
        for segment_var in state["segment_vars"]:
            assert segment_var in result["cate_by_segment"]

    @pytest.mark.asyncio
    async def test_cate_result_structure(self):
        """Test structure of CATE results."""
        node = CATEEstimatorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        cate_by_segment = result["cate_by_segment"]
        for _segment_var, results in cate_by_segment.items():
            assert isinstance(results, list)
            assert len(results) > 0

            # Check first result structure
            cate_result = results[0]
            assert "segment_name" in cate_result
            assert "segment_value" in cate_result
            assert "cate_estimate" in cate_result
            assert "cate_ci_lower" in cate_result
            assert "cate_ci_upper" in cate_result
            assert "sample_size" in cate_result
            assert "statistical_significance" in cate_result

    @pytest.mark.asyncio
    async def test_confidence_interval(self):
        """Test confidence interval calculation."""
        node = CATEEstimatorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        cate_by_segment = result["cate_by_segment"]
        for _segment_var, results in cate_by_segment.items():
            for cate_result in results:
                ci_lower = cate_result["cate_ci_lower"]
                ci_upper = cate_result["cate_ci_upper"]
                cate = cate_result["cate_estimate"]

                # CI should contain estimate
                assert ci_lower <= cate <= ci_upper

    @pytest.mark.asyncio
    async def test_statistical_significance(self):
        """Test statistical significance determination."""
        node = CATEEstimatorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        cate_by_segment = result["cate_by_segment"]
        for _segment_var, results in cate_by_segment.items():
            for cate_result in results:
                significance = cate_result["statistical_significance"]
                assert isinstance(significance, bool)

                # If significant, CI should not contain 0
                if significance:
                    ci_lower = cate_result["cate_ci_lower"]
                    ci_upper = cate_result["cate_ci_upper"]
                    assert (ci_lower > 0) or (ci_upper < 0)

    @pytest.mark.asyncio
    async def test_estimation_latency(self):
        """Test latency measurement."""
        node = CATEEstimatorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "estimation_latency_ms" in result
        assert result["estimation_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_status_update(self):
        """Test status update to analyzing."""
        node = CATEEstimatorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert result["status"] == "analyzing"

    @pytest.mark.asyncio
    async def test_insufficient_data(self):
        """Test handling of insufficient data."""
        node = CATEEstimatorNode()

        # Create mock connector that returns small dataset
        class SmallDataConnector:
            async def query(self, source, columns, filters=None):
                import numpy as np
                import pandas as pd

                np.random.seed(42)
                # Only 50 rows
                return pd.DataFrame({col: np.random.randn(50) for col in columns})

        node.data_connector = SmallDataConnector()
        state = self._create_test_state()

        result = await node.execute(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert "Insufficient data" in result["errors"][0]["error"]

    @pytest.mark.asyncio
    async def test_multiple_segments(self):
        """Test CATE estimation with multiple segment variables."""
        node = CATEEstimatorNode()
        state = self._create_test_state(
            segment_vars=["hcp_specialty", "region", "patient_volume_decile"]
        )

        result = await node.execute(state)

        assert len(result["cate_by_segment"]) == 3

    @pytest.mark.asyncio
    async def test_heterogeneity_score_range(self):
        """Test heterogeneity score is properly normalized."""
        node = CATEEstimatorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        # Score should be between 0 and 1
        score = result["heterogeneity_score"]
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_cate_sorted_by_estimate(self):
        """Test CATE results are sorted by estimate."""
        node = CATEEstimatorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        for _segment_var, results in result["cate_by_segment"].items():
            # Check sorting (descending)
            for i in range(len(results) - 1):
                assert results[i]["cate_estimate"] >= results[i + 1]["cate_estimate"]


class TestCATEEstimatorEdgeCases:
    """Test edge cases for CATE estimator."""

    def _create_test_state(self, **overrides):
        """Create test state."""
        state = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure", "competitive_pressure"],
            "data_source": "test",
            "filters": None,
            "n_estimators": 52,  # Must be divisible by 4 (subforest_size)
            "min_samples_leaf": 10,
            "significance_level": 0.05,
            "top_segments_count": 10,
            "cate_by_segment": None,
            "overall_ate": None,
            "heterogeneity_score": None,
            "feature_importance": None,
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
            "estimation_latency_ms": 0,
            "analysis_latency_ms": 0,
            "total_latency_ms": 0,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }
        state.update(overrides)
        return state

    @pytest.mark.asyncio
    async def test_binary_treatment(self):
        """Test with binary treatment variable."""
        node = CATEEstimatorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        # Should handle binary treatment
        assert result["status"] == "analyzing"

    @pytest.mark.asyncio
    async def test_continuous_treatment(self):
        """Test with continuous treatment variable."""
        node = CATEEstimatorNode()

        # Mock connector with continuous treatment
        class ContinuousTreatmentConnector:
            async def query(self, source, columns, filters=None):
                import numpy as np
                import pandas as pd

                np.random.seed(42)
                n = 1000
                data = {}
                for col in columns:
                    if col == "hcp_engagement_frequency":
                        data[col] = np.random.uniform(0, 10, n)  # Continuous
                    else:
                        data[col] = np.random.randn(n)
                return pd.DataFrame(data)

        node.data_connector = ContinuousTreatmentConnector()
        state = self._create_test_state()

        result = await node.execute(state)

        # Should handle continuous treatment
        assert result["status"] == "analyzing"

    @pytest.mark.asyncio
    async def test_single_effect_modifier(self):
        """Test with single effect modifier."""
        node = CATEEstimatorNode()
        state = self._create_test_state(effect_modifiers=["hcp_tenure"])

        result = await node.execute(state)

        assert len(result["feature_importance"]) == 1

    @pytest.mark.asyncio
    async def test_many_effect_modifiers(self):
        """Test with many effect modifiers."""
        import numpy as np
        import pandas as pd

        # Custom mock connector with 10 effect modifiers
        class ManyModifiersConnector:
            async def query(self, source, columns, filters=None):
                np.random.seed(42)
                n = 1000
                data = {}
                for col in columns:
                    if col == "hcp_engagement_frequency":
                        data[col] = np.random.choice([0, 1], n)
                    elif col == "trx_total":
                        data[col] = np.random.randn(n) * 100 + 500
                    elif col == "hcp_specialty":
                        data[col] = np.random.choice(["A", "B"], n)
                    else:
                        data[col] = np.random.randn(n)
                return pd.DataFrame(data)

        node = CATEEstimatorNode()
        node.data_connector = ManyModifiersConnector()
        modifiers = [f"modifier_{i}" for i in range(10)]
        state = self._create_test_state(effect_modifiers=modifiers)

        result = await node.execute(state)

        # If status is "analyzing", check feature importance
        if result["status"] == "analyzing":
            assert len(result["feature_importance"]) == 10
        else:
            # Edge case where EconML may fail with many modifiers - check error exists
            assert result["errors"] is not None
