"""Tests for HierarchicalAnalyzerNode.

B9.4: Integration tests for hierarchical analysis in heterogeneous_optimizer agent.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.heterogeneous_optimizer.nodes.hierarchical_analyzer import (
    HierarchicalAnalyzerNode,
    HierarchicalAnalyzerOutput,
    HierarchicalCATEResult,
    NestedCIResult,
)
from src.agents.heterogeneous_optimizer.state import HeterogeneousOptimizerState


class TestHierarchicalAnalyzerNode:
    """Test HierarchicalAnalyzerNode class."""

    @pytest.fixture
    def base_state(self) -> HeterogeneousOptimizerState:
        """Create base state for testing."""
        return {
            "query": "Analyze heterogeneous treatment effects",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "segment_vars": ["segment_a"],
            "effect_modifiers": ["feature_1", "feature_2"],
            "data_source": "mock_data",
            "filters": None,
            "n_estimators": 50,
            "min_samples_leaf": 10,
            "significance_level": 0.05,
            "top_segments_count": 5,
            "status": "analyzing",
            "errors": [],
            "warnings": [],
            # Simulated prior step results
            "cate_by_segment": None,
            "overall_ate": 0.15,
            "heterogeneity_score": 0.5,
            "uplift_by_segment": None,
            "overall_auuc": 0.65,
        }

    @pytest.fixture
    def node(self) -> HierarchicalAnalyzerNode:
        """Create HierarchicalAnalyzerNode instance."""
        return HierarchicalAnalyzerNode(
            n_segments=3,
            segmentation_method="quantile",
            estimator_type="ols",  # Fast for testing
            min_segment_size=30,
            confidence_level=0.95,
            aggregation_method="variance_weighted",
            timeout_seconds=60,
        )

    def test_init_default(self) -> None:
        """Test default initialization."""
        node = HierarchicalAnalyzerNode()

        assert node.n_segments == 3
        assert node.segmentation_method == "quantile"
        assert node.estimator_type == "causal_forest"
        assert node.min_segment_size == 50
        assert node.confidence_level == 0.95
        assert node.aggregation_method == "variance_weighted"
        assert node.timeout_seconds == 180

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        node = HierarchicalAnalyzerNode(
            n_segments=5,
            segmentation_method="kmeans",
            estimator_type="linear_dml",
            min_segment_size=100,
            confidence_level=0.99,
            aggregation_method="bootstrap",
            timeout_seconds=300,
        )

        assert node.n_segments == 5
        assert node.segmentation_method == "kmeans"
        assert node.estimator_type == "linear_dml"
        assert node.min_segment_size == 100
        assert node.confidence_level == 0.99
        assert node.aggregation_method == "bootstrap"
        assert node.timeout_seconds == 300

    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        node: HierarchicalAnalyzerNode,
        base_state: HeterogeneousOptimizerState,
    ) -> None:
        """Test successful execution."""
        result = await node.execute(base_state)

        # Should have hierarchical results
        assert "hierarchical_segment_results" in result
        assert "nested_ci" in result
        assert "hierarchical_latency_ms" in result
        assert result.get("segmentation_method") == "quantile"
        assert result.get("estimator_type") == "ols"

    @pytest.mark.asyncio
    async def test_execute_skips_on_failure(
        self,
        node: HierarchicalAnalyzerNode,
        base_state: HeterogeneousOptimizerState,
    ) -> None:
        """Test that execution is skipped when previous step failed."""
        base_state["status"] = "failed"

        result = await node.execute(base_state)

        assert "warnings" in result
        assert any("skipped" in w.lower() for w in result.get("warnings", []))
        assert "hierarchical_segment_results" not in result

    @pytest.mark.asyncio
    async def test_execute_with_different_estimators(
        self,
        base_state: HeterogeneousOptimizerState,
    ) -> None:
        """Test with different EconML estimators."""
        estimators = ["ols", "linear_dml", "causal_forest"]

        for estimator in estimators:
            node = HierarchicalAnalyzerNode(
                n_segments=2,
                estimator_type=estimator,
                min_segment_size=30,
                timeout_seconds=60,
            )

            result = await node.execute(base_state)

            # Should complete (may have warnings but not fail completely)
            assert "hierarchical_latency_ms" in result or "warnings" in result

    @pytest.mark.asyncio
    async def test_execute_with_different_segmentation_methods(
        self,
        base_state: HeterogeneousOptimizerState,
    ) -> None:
        """Test with different segmentation methods."""
        methods = ["quantile", "kmeans", "threshold"]

        for method in methods:
            node = HierarchicalAnalyzerNode(
                n_segments=2,
                segmentation_method=method,
                estimator_type="ols",
                min_segment_size=30,
                timeout_seconds=60,
            )

            result = await node.execute(base_state)

            # Should complete
            assert "hierarchical_latency_ms" in result or "warnings" in result

    @pytest.mark.asyncio
    async def test_execute_produces_segment_results(
        self,
        node: HierarchicalAnalyzerNode,
        base_state: HeterogeneousOptimizerState,
    ) -> None:
        """Test that segment results are produced."""
        result = await node.execute(base_state)

        segment_results = result.get("hierarchical_segment_results", [])

        if len(segment_results) > 0:
            seg = segment_results[0]
            assert "segment_id" in seg
            assert "segment_name" in seg
            assert "n_samples" in seg
            assert "success" in seg

    @pytest.mark.asyncio
    async def test_execute_produces_nested_ci(
        self,
        node: HierarchicalAnalyzerNode,
        base_state: HeterogeneousOptimizerState,
    ) -> None:
        """Test that nested CI is computed."""
        result = await node.execute(base_state)

        nested_ci = result.get("nested_ci")

        if nested_ci is not None:
            assert "aggregate_ate" in nested_ci
            assert "aggregate_ci_lower" in nested_ci
            assert "aggregate_ci_upper" in nested_ci
            assert "aggregation_method" in nested_ci
            assert "segment_contributions" in nested_ci

    @pytest.mark.asyncio
    async def test_execute_with_bootstrap_aggregation(
        self,
        base_state: HeterogeneousOptimizerState,
    ) -> None:
        """Test with bootstrap aggregation method."""
        node = HierarchicalAnalyzerNode(
            n_segments=2,
            estimator_type="ols",
            min_segment_size=30,
            aggregation_method="bootstrap",
            timeout_seconds=60,
        )

        result = await node.execute(base_state)

        # Should complete
        assert "hierarchical_latency_ms" in result or "warnings" in result

    def test_generate_mock_data(
        self,
        node: HierarchicalAnalyzerNode,
        base_state: HeterogeneousOptimizerState,
    ) -> None:
        """Test mock data generation."""
        df = node._generate_mock_data(base_state)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 500  # Default mock size
        assert base_state["treatment_var"] in df.columns
        assert base_state["outcome_var"] in df.columns
        assert "feature_1" in df.columns
        assert "feature_2" in df.columns

    def test_prepare_data(
        self,
        node: HierarchicalAnalyzerNode,
        base_state: HeterogeneousOptimizerState,
    ) -> None:
        """Test data preparation."""
        df = node._generate_mock_data(base_state)

        X, treatment, y = node._prepare_data(df, base_state)

        assert isinstance(X, pd.DataFrame)
        assert len(X.columns) == 2  # feature_1, feature_2
        assert len(treatment) == len(df)
        assert len(y) == len(df)
        assert set(np.unique(treatment)).issubset({0, 1})

    def test_get_uplift_scores_none(
        self,
        node: HierarchicalAnalyzerNode,
        base_state: HeterogeneousOptimizerState,
    ) -> None:
        """Test uplift score retrieval returns None when not available."""
        scores = node._get_uplift_scores(base_state, 100)

        # Should return None to let hierarchical analyzer compute internally
        assert scores is None


class TestHierarchicalAnalyzerTypedDicts:
    """Test TypedDict structures."""

    def test_hierarchical_cate_result_success(self) -> None:
        """Test HierarchicalCATEResult for successful estimation."""
        result = HierarchicalCATEResult(
            segment_id=0,
            segment_name="low_uplift",
            n_samples=200,
            uplift_range=(0.0, 0.1),
            cate_mean=0.12,
            cate_std=0.03,
            cate_ci_lower=0.06,
            cate_ci_upper=0.18,
            success=True,
            error_message=None,
        )

        assert result["segment_id"] == 0
        assert result["success"] is True
        assert result["cate_mean"] == 0.12

    def test_hierarchical_cate_result_failure(self) -> None:
        """Test HierarchicalCATEResult for failed estimation."""
        result = HierarchicalCATEResult(
            segment_id=1,
            segment_name="medium_uplift",
            n_samples=10,
            uplift_range=(0.1, 0.2),
            cate_mean=None,
            cate_std=None,
            cate_ci_lower=None,
            cate_ci_upper=None,
            success=False,
            error_message="Insufficient samples",
        )

        assert result["success"] is False
        assert result["error_message"] == "Insufficient samples"
        assert result["cate_mean"] is None

    def test_nested_ci_result(self) -> None:
        """Test NestedCIResult structure."""
        result = NestedCIResult(
            aggregate_ate=0.15,
            aggregate_ci_lower=0.10,
            aggregate_ci_upper=0.20,
            aggregate_std=0.025,
            confidence_level=0.95,
            aggregation_method="variance_weighted",
            segment_contributions={"low": 0.4, "medium": 0.35, "high": 0.25},
            i_squared=45.2,
            tau_squared=0.001,
            n_segments_included=3,
            total_sample_size=500,
        )

        assert result["aggregate_ate"] == 0.15
        assert result["aggregation_method"] == "variance_weighted"
        assert len(result["segment_contributions"]) == 3
        assert abs(sum(result["segment_contributions"].values()) - 1.0) < 0.01


class TestGraphIntegration:
    """Test integration with agent graph."""

    @pytest.mark.asyncio
    async def test_graph_with_hierarchical(self) -> None:
        """Test graph creation with hierarchical enabled."""
        from src.agents.heterogeneous_optimizer.graph import (
            create_heterogeneous_optimizer_graph,
        )

        graph = create_heterogeneous_optimizer_graph(enable_hierarchical=True)

        # Graph should be compiled
        assert graph is not None

    @pytest.mark.asyncio
    async def test_graph_without_hierarchical(self) -> None:
        """Test graph creation with hierarchical disabled."""
        from src.agents.heterogeneous_optimizer.graph import (
            create_heterogeneous_optimizer_graph,
        )

        graph = create_heterogeneous_optimizer_graph(enable_hierarchical=False)

        # Graph should be compiled
        assert graph is not None

    def test_node_import(self) -> None:
        """Test that node can be imported from nodes package."""
        from src.agents.heterogeneous_optimizer.nodes import HierarchicalAnalyzerNode

        node = HierarchicalAnalyzerNode()
        assert node is not None

    def test_state_has_hierarchical_fields(self) -> None:
        """Test that state TypedDict has hierarchical fields."""
        from src.agents.heterogeneous_optimizer.state import (
            HeterogeneousOptimizerState,
        )

        # Check annotations for hierarchical fields
        annotations = HeterogeneousOptimizerState.__annotations__

        assert "hierarchical_segment_results" in annotations
        assert "nested_ci" in annotations
        assert "segment_heterogeneity_score" in annotations
        assert "overall_hierarchical_ate" in annotations
        assert "overall_hierarchical_ci_lower" in annotations
        assert "overall_hierarchical_ci_upper" in annotations
        assert "n_segments_analyzed" in annotations
        assert "segmentation_method_used" in annotations
        assert "hierarchical_estimator_type" in annotations
        assert "hierarchical_latency_ms" in annotations

    def test_output_has_hierarchical_fields(self) -> None:
        """Test that output TypedDict has hierarchical fields."""
        from src.agents.heterogeneous_optimizer.state import (
            HeterogeneousOptimizerOutput,
        )

        annotations = HeterogeneousOptimizerOutput.__annotations__

        assert "hierarchical_segment_results" in annotations
        assert "nested_ci" in annotations
        assert "segment_heterogeneity_score" in annotations
        assert "overall_hierarchical_ate" in annotations
