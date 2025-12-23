"""Tests for estimation node."""

import pytest

from src.agents.causal_impact.nodes.estimation import EstimationNode
from src.agents.causal_impact.state import CausalGraph, CausalImpactState


class TestEstimationNode:
    """Test EstimationNode."""

    def _create_test_graph(self) -> CausalGraph:
        """Create test causal graph."""
        return {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("geographic_region", "hcp_engagement_level"),
                ("geographic_region", "patient_conversion_rate"),
                ("hcp_engagement_level", "patient_conversion_rate"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "digraph { ... }",
            "confidence": 0.85,
        }

    @pytest.mark.asyncio
    async def test_estimate_with_causal_forest(self):
        """Test estimation using CausalForestDML."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-1",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "parameters": {"method": "CausalForestDML"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        assert "estimation_result" in result
        est = result["estimation_result"]

        assert est["method"] == "CausalForestDML"
        assert "ate" in est
        assert "ate_ci_lower" in est
        assert "ate_ci_upper" in est
        assert est["effect_size"] in ["small", "medium", "large"]
        assert isinstance(est["statistical_significance"], bool)
        assert result["current_phase"] == "refuting"

    @pytest.mark.asyncio
    async def test_estimate_with_linear_dml(self):
        """Test estimation using LinearDML."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-2",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "parameters": {"method": "LinearDML"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]
        assert est["method"] == "LinearDML"
        assert "ate" in est

    @pytest.mark.asyncio
    async def test_estimate_with_linear_regression(self):
        """Test estimation using linear regression."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-3",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "parameters": {"method": "linear_regression"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]
        assert est["method"] == "linear_regression"

    @pytest.mark.asyncio
    async def test_estimate_with_propensity_weighting(self):
        """Test estimation using propensity score weighting."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-4",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "parameters": {"method": "propensity_score_weighting"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]
        assert est["method"] == "propensity_score_weighting"

    @pytest.mark.asyncio
    async def test_confidence_interval_validity(self):
        """Test that confidence intervals are valid."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-5",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]
        ate = est["ate"]
        ci_lower = est["ate_ci_lower"]
        ci_upper = est["ate_ci_upper"]

        # CI should contain point estimate
        assert ci_lower <= ate <= ci_upper, "ATE not within confidence interval"

    @pytest.mark.asyncio
    async def test_effect_size_classification(self):
        """Test effect size classification."""
        node = EstimationNode()

        # Test small effect
        assert node._classify_effect_size(0.1) == "small"

        # Test medium effect
        assert node._classify_effect_size(0.3) == "medium"

        # Test large effect
        assert node._classify_effect_size(0.8) == "large"

        # Test negative effects
        assert node._classify_effect_size(-0.1) == "small"
        assert node._classify_effect_size(-0.6) == "large"

    @pytest.mark.asyncio
    async def test_heterogeneity_detection(self):
        """Test CATE heterogeneity detection."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-6",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "parameters": {"method": "CausalForestDML"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]

        # CausalForestDML should detect heterogeneity
        if est["method"] == "CausalForestDML":
            assert est["heterogeneity_detected"] is True
            assert "cate_segments" in est
            assert len(est["cate_segments"]) > 0

    @pytest.mark.asyncio
    async def test_covariates_adjusted(self):
        """Test that adjustment set is recorded."""
        node = EstimationNode()

        graph = self._create_test_graph()
        graph["adjustment_sets"] = [["geographic_region", "hcp_specialty"]]

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-7",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region", "hcp_specialty"],
            "data_source": "synthetic",
            "causal_graph": graph,
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]
        assert "covariates_adjusted" in est
        assert "geographic_region" in est["covariates_adjusted"]

    @pytest.mark.asyncio
    async def test_sample_size_recorded(self):
        """Test that sample size is recorded."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-8",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]
        assert "sample_size" in est
        assert est["sample_size"] > 0

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test that estimation latency is measured."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-9",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        assert "estimation_latency_ms" in result
        assert result["estimation_latency_ms"] >= 0
        assert result["estimation_latency_ms"] < 30000  # Should be < 30s

    @pytest.mark.asyncio
    async def test_error_handling_missing_graph(self):
        """Test error handling when causal graph is missing."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-10",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        assert "estimation_error" in result
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_error_handling_unknown_method(self):
        """Test error handling for unknown estimation method."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-11",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": self._create_test_graph(),
            "parameters": {"method": "unknown_method"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        assert "estimation_error" in result
        assert result["status"] == "failed"


class TestCATESegments:
    """Test CATE segment analysis."""

    @pytest.mark.asyncio
    async def test_cate_segments_structure(self):
        """Test CATE segments have correct structure."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-12",
            "treatment_var": "T",
            "outcome_var": "O",
            "confounders": [],
            "data_source": "synthetic",
            "causal_graph": {
                "nodes": ["T", "O"],
                "edges": [("T", "O")],
                "treatment_nodes": ["T"],
                "outcome_nodes": ["O"],
                "adjustment_sets": [[]],
                "dag_dot": "...",
                "confidence": 0.8,
            },
            "parameters": {"method": "CausalForestDML"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]

        if "cate_segments" in est:
            for segment in est["cate_segments"]:
                assert "segment" in segment
                assert "cate" in segment
                assert "size" in segment
                assert "description" in segment


class TestStatisticalSignificance:
    """Test statistical significance logic."""

    @pytest.mark.asyncio
    async def test_significance_threshold(self):
        """Test that significance is based on 95% CI."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-13",
            "treatment_var": "T",
            "outcome_var": "O",
            "confounders": [],
            "data_source": "synthetic",
            "causal_graph": {
                "nodes": ["T", "O"],
                "edges": [("T", "O")],
                "treatment_nodes": ["T"],
                "outcome_nodes": ["O"],
                "adjustment_sets": [[]],
                "dag_dot": "...",
                "confidence": 0.8,
            },
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]

        # If significant, CI should not contain 0
        if est["statistical_significance"]:
            ci_lower = est["ate_ci_lower"]
            ci_upper = est["ate_ci_upper"]

            # Either both positive or both negative
            assert (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)

    @pytest.mark.asyncio
    async def test_p_value_consistency(self):
        """Test that p-value is consistent with significance."""
        node = EstimationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-14",
            "treatment_var": "T",
            "outcome_var": "O",
            "confounders": [],
            "data_source": "synthetic",
            "causal_graph": {
                "nodes": ["T", "O"],
                "edges": [("T", "O")],
                "treatment_nodes": ["T"],
                "outcome_nodes": ["O"],
                "adjustment_sets": [[]],
                "dag_dot": "...",
                "confidence": 0.8,
            },
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        est = result["estimation_result"]

        # If significant, p-value should be < 0.05
        if est["statistical_significance"]:
            assert est["p_value"] < 0.05
