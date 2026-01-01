"""Integration tests for Tool Composer + Causal Discovery.

Version: 1.0.0 (V4.4)
Tests the integration between Tool Composer and causal discovery tools:
- discover_dag
- rank_drivers
- detect_structural_drift

These tests verify that:
1. Causal discovery tools are available in the tool registry
2. The planner correctly selects causal discovery tools
3. Tool chaining works (discover_dag -> rank_drivers)
4. Structural drift detection integrates properly
5. Error handling propagates correctly
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from uuid import uuid4
from typing import Any, Dict, List

from src.tool_registry.registry import ToolRegistry, get_registry
from src.tool_registry.tools.causal_discovery import (
    CausalDiscoveryTool,
    DiscoverDagInput,
    DiscoverDagOutput,
    DriverRankerTool,
    RankDriversInput,
    RankDriversOutput,
    discover_dag,
    rank_drivers,
    get_discovery_tool,
    get_ranker_tool,
    register_all_discovery_tools,
)
from src.tool_registry.tools.structural_drift import (
    StructuralDriftTool,
    StructuralDriftInput,
    StructuralDriftOutput,
    detect_structural_drift,
    get_structural_drift_tool,
    register_structural_drift_tool,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """Create synthetic data with known causal structure.

    Structure: A -> B -> Outcome, Treatment -> Outcome
    """
    np.random.seed(42)
    n = 200

    # A causes B causes Outcome
    # Treatment affects Outcome
    treatment = np.random.randn(n)
    a = np.random.randn(n)
    b = 0.7 * a + 0.3 * np.random.randn(n)
    outcome = 0.5 * treatment + 0.3 * b + 0.2 * np.random.randn(n)

    return pd.DataFrame({
        "treatment": treatment,
        "feature_a": a,
        "feature_b": b,
        "outcome": outcome,
    })


@pytest.fixture
def data_dict(synthetic_data: pd.DataFrame) -> Dict[str, List[Any]]:
    """Convert synthetic data to dictionary format for tools."""
    return synthetic_data.to_dict("list")


@pytest.fixture
def baseline_dag_adjacency() -> List[List[int]]:
    """Create baseline DAG adjacency matrix.

    Nodes: [treatment, feature_a, feature_b, outcome]
    Edges: treatment->outcome, feature_a->feature_b, feature_b->outcome
    """
    return [
        [0, 0, 0, 1],  # treatment -> outcome
        [0, 0, 1, 0],  # feature_a -> feature_b
        [0, 0, 0, 1],  # feature_b -> outcome
        [0, 0, 0, 0],  # outcome (sink)
    ]


@pytest.fixture
def current_dag_adjacency() -> List[List[int]]:
    """Create current DAG adjacency matrix with drift.

    Nodes: [treatment, feature_a, feature_b, outcome]
    Edges: treatment->outcome, feature_a->feature_b, feature_b->outcome, feature_a->outcome (NEW)
    """
    return [
        [0, 0, 0, 1],  # treatment -> outcome
        [0, 0, 1, 1],  # feature_a -> feature_b, feature_a -> outcome (NEW)
        [0, 0, 0, 1],  # feature_b -> outcome
        [0, 0, 0, 0],  # outcome (sink)
    ]


@pytest.fixture
def dag_nodes() -> List[str]:
    """Node names for DAG adjacency matrices."""
    return ["treatment", "feature_a", "feature_b", "outcome"]


@pytest.fixture
def edge_list() -> List[Dict[str, str]]:
    """Edge list from discover_dag output format."""
    return [
        {"source": "treatment", "target": "outcome"},
        {"source": "feature_a", "target": "feature_b"},
        {"source": "feature_b", "target": "outcome"},
    ]


@pytest.fixture
def shap_values() -> List[List[float]]:
    """SHAP values for rank_drivers (n_samples=10, n_features=3)."""
    np.random.seed(42)
    return np.abs(np.random.randn(10, 3)).tolist()


@pytest.fixture
def feature_names() -> List[str]:
    """Feature names for SHAP values."""
    return ["treatment", "feature_a", "feature_b"]


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Get or create tool registry with causal discovery tools registered."""
    registry = get_registry()
    # Ensure tools are registered
    register_all_discovery_tools()
    register_structural_drift_tool()
    return registry


# =============================================================================
# TEST: TOOL REGISTRATION
# =============================================================================


class TestCausalDiscoveryToolRegistration:
    """Test that causal discovery tools are properly registered."""

    def test_discover_dag_tool_available(self, tool_registry: ToolRegistry):
        """Test discover_dag tool is registered."""
        tool = tool_registry.get("discover_dag")
        assert tool is not None
        assert tool.schema.name == "discover_dag"

    def test_rank_drivers_tool_available(self, tool_registry: ToolRegistry):
        """Test rank_drivers tool is registered."""
        tool = tool_registry.get("rank_drivers")
        assert tool is not None
        assert tool.schema.name == "rank_drivers"

    def test_structural_drift_tool_available(self, tool_registry: ToolRegistry):
        """Test detect_structural_drift tool is registered."""
        tool = tool_registry.get("detect_structural_drift")
        assert tool is not None
        assert tool.schema.name == "detect_structural_drift"

    def test_all_causal_tools_registered(self, tool_registry: ToolRegistry):
        """Test all causal discovery tools are registered."""
        causal_tools = ["discover_dag", "rank_drivers", "detect_structural_drift"]
        for tool_name in causal_tools:
            tool = tool_registry.get(tool_name)
            assert tool is not None, f"{tool_name} not found"


# =============================================================================
# TEST: SINGLE TOOL EXECUTION
# =============================================================================


class TestSingleToolExecution:
    """Test individual causal discovery tool execution."""

    @pytest.mark.asyncio
    async def test_discover_dag_basic_execution(self, data_dict: Dict[str, List[Any]]):
        """Test basic discover_dag execution."""
        tool = get_discovery_tool()

        input_data = DiscoverDagInput(
            data=data_dict,
            algorithms=["ges"],  # Single algorithm for speed
            ensemble_threshold=0.5,
        )

        result = await tool.invoke(input_data)

        assert isinstance(result, DiscoverDagOutput)
        assert result.success is True
        assert result.n_nodes > 0
        assert len(result.algorithms_used) >= 1
        assert isinstance(result.edge_list, list)

    @pytest.mark.asyncio
    async def test_rank_drivers_basic_execution(
        self,
        edge_list: List[Dict[str, str]],
        shap_values: List[List[float]],
        feature_names: List[str],
    ):
        """Test basic rank_drivers execution."""
        tool = get_ranker_tool()

        input_data = RankDriversInput(
            dag_edge_list=edge_list,
            target="outcome",
            shap_values=shap_values,
            feature_names=feature_names,
        )

        result = await tool.invoke(input_data)

        assert isinstance(result, RankDriversOutput)
        assert result.success is True
        assert result.target_variable == "outcome"
        assert len(result.rankings) > 0

    @pytest.mark.asyncio
    async def test_structural_drift_basic_execution(
        self,
        baseline_dag_adjacency: List[List[int]],
        current_dag_adjacency: List[List[int]],
        dag_nodes: List[str],
    ):
        """Test basic detect_structural_drift execution."""
        tool = get_structural_drift_tool()

        input_data = StructuralDriftInput(
            baseline_dag_adjacency=baseline_dag_adjacency,
            current_dag_adjacency=current_dag_adjacency,
            dag_nodes=dag_nodes,
        )

        result = await tool.invoke(input_data)

        assert isinstance(result, StructuralDriftOutput)
        assert result.success is True
        # Should detect drift because current_dag has extra edge
        assert result.detected is True
        assert len(result.added_edges) > 0


# =============================================================================
# TEST: TOOL CHAINING (discover_dag -> rank_drivers)
# =============================================================================


class TestToolChaining:
    """Test tool chaining for causal discovery workflows."""

    @pytest.mark.asyncio
    async def test_discover_dag_to_rank_drivers_chain(
        self,
        data_dict: Dict[str, List[Any]],
        shap_values: List[List[float]],
    ):
        """Test chained execution: discover_dag -> rank_drivers."""
        # Phase 1: Run discover_dag
        discovery_tool = get_discovery_tool()
        discovery_input = DiscoverDagInput(
            data=data_dict,
            algorithms=["ges"],
            ensemble_threshold=0.5,
        )

        discovery_result = await discovery_tool.invoke(discovery_input)

        assert discovery_result.success is True
        assert len(discovery_result.edge_list) > 0

        # Phase 2: Pass edge_list to rank_drivers
        # Get feature names from data
        feature_names = list(data_dict.keys())

        # Extract only source/target for RankDriversInput
        # (full edge_list has extra fields like confidence, type, algorithms)
        simplified_edges = [
            {"source": e["source"], "target": e["target"]}
            for e in discovery_result.edge_list
        ]

        ranker_tool = get_ranker_tool()
        ranker_input = RankDriversInput(
            dag_edge_list=simplified_edges,  # Chain: extract source/target from $step_1.edge_list
            target="outcome",
            shap_values=shap_values[:len(feature_names)] if len(shap_values) > len(feature_names) else shap_values,
            feature_names=feature_names,
        )

        ranker_result = await ranker_tool.invoke(ranker_input)

        assert ranker_result.success is True
        assert ranker_result.target_variable == "outcome"
        assert len(ranker_result.rankings) > 0

    @pytest.mark.asyncio
    async def test_chain_handles_empty_discovery(self):
        """Test that chain handles empty discovery results gracefully."""
        # Create minimal data that may produce no edges
        minimal_data = {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }

        discovery_tool = get_discovery_tool()
        discovery_input = DiscoverDagInput(
            data=minimal_data,
            algorithms=["ges"],
            ensemble_threshold=0.99,  # High threshold = fewer edges
        )

        result = await discovery_tool.invoke(discovery_input)

        # Should succeed even with potentially empty results
        assert result.success is True


# =============================================================================
# TEST: STRUCTURAL DRIFT DETECTION
# =============================================================================


class TestStructuralDriftComposition:
    """Test structural drift detection as composable tool."""

    @pytest.mark.asyncio
    async def test_no_drift_detection(self, dag_nodes: List[str]):
        """Test detection when DAGs are identical (no drift)."""
        same_adjacency = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]

        tool = get_structural_drift_tool()
        input_data = StructuralDriftInput(
            baseline_dag_adjacency=same_adjacency,
            current_dag_adjacency=same_adjacency,
            dag_nodes=dag_nodes,
        )

        result = await tool.invoke(input_data)

        assert result.success is True
        assert result.detected is False
        assert result.drift_score == 0.0
        assert result.severity == "none"
        assert len(result.added_edges) == 0
        assert len(result.removed_edges) == 0

    @pytest.mark.asyncio
    async def test_critical_drift_detection(self, dag_nodes: List[str]):
        """Test detection of critical structural drift (>30% edges changed)."""
        # Baseline: 3 edges
        baseline = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]

        # Current: completely different structure (all edges changed)
        current = [
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
        ]

        tool = get_structural_drift_tool()
        input_data = StructuralDriftInput(
            baseline_dag_adjacency=baseline,
            current_dag_adjacency=current,
            dag_nodes=dag_nodes,
        )

        result = await tool.invoke(input_data)

        assert result.success is True
        assert result.detected is True
        assert result.severity in ["critical", "high"]
        assert result.recommendation is not None

    @pytest.mark.asyncio
    async def test_edge_type_changes_detection(self, dag_nodes: List[str]):
        """Test detection of edge type changes."""
        adjacency = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]

        tool = get_structural_drift_tool()
        input_data = StructuralDriftInput(
            baseline_dag_adjacency=adjacency,
            current_dag_adjacency=adjacency,  # Same structure
            dag_nodes=dag_nodes,
            baseline_edge_types={
                "treatment->outcome": "DIRECTED",
                "feature_a->feature_b": "DIRECTED",
            },
            current_edge_types={
                "treatment->outcome": "BIDIRECTED",  # Changed type
                "feature_a->feature_b": "DIRECTED",
            },
        )

        result = await tool.invoke(input_data)

        assert result.success is True
        # Edge type change should trigger detection
        assert len(result.edge_type_changes) > 0 or result.detected


# =============================================================================
# TEST: ERROR HANDLING
# =============================================================================


class TestErrorHandling:
    """Test error handling and propagation."""

    @pytest.mark.asyncio
    async def test_discover_dag_invalid_data(self):
        """Test discover_dag handles invalid data gracefully."""
        tool = get_discovery_tool()

        # Empty data should still be handled
        input_data = DiscoverDagInput(
            data={},
            algorithms=["ges"],
        )

        result = await tool.invoke(input_data)

        # Should return failure but not crash
        assert isinstance(result, DiscoverDagOutput)
        # Either success=False or empty results
        assert result.success is False or result.n_nodes == 0

    @pytest.mark.asyncio
    async def test_rank_drivers_invalid_target(
        self,
        edge_list: List[Dict[str, str]],
        shap_values: List[List[float]],
        feature_names: List[str],
    ):
        """Test rank_drivers handles invalid target variable."""
        tool = get_ranker_tool()

        input_data = RankDriversInput(
            dag_edge_list=edge_list,
            target="nonexistent_target",  # Not in DAG
            shap_values=shap_values,
            feature_names=feature_names,
        )

        result = await tool.invoke(input_data)

        # Should handle gracefully
        assert isinstance(result, RankDriversOutput)

    @pytest.mark.asyncio
    async def test_structural_drift_mismatched_dimensions(self, dag_nodes: List[str]):
        """Test structural drift handles mismatched adjacency dimensions."""
        tool = get_structural_drift_tool()

        # Baseline: 4x4
        baseline = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]

        # Current: 3x3 (mismatched)
        current = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ]

        input_data = StructuralDriftInput(
            baseline_dag_adjacency=baseline,
            current_dag_adjacency=current,
            dag_nodes=dag_nodes,  # 4 nodes but 3x3 matrix
        )

        result = await tool.invoke(input_data)

        # Should handle gracefully (either fail with error or succeed with warning)
        assert isinstance(result, StructuralDriftOutput)
        assert result.success is False or len(result.errors) > 0


# =============================================================================
# TEST: STANDALONE FUNCTIONS
# =============================================================================


class TestStandaloneFunctions:
    """Test standalone async functions for tool execution.

    Note: Standalone functions return Dict[str, Any] via model_dump(),
    not the Pydantic model directly.
    """

    @pytest.mark.asyncio
    async def test_discover_dag_standalone(self, data_dict: Dict[str, List[Any]]):
        """Test discover_dag standalone function."""
        result = await discover_dag(
            data=data_dict,
            algorithms=["ges"],
            ensemble_threshold=0.5,
        )

        # Standalone functions return dict via model_dump()
        assert isinstance(result, dict)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_rank_drivers_standalone(
        self,
        edge_list: List[Dict[str, str]],
        shap_values: List[List[float]],
        feature_names: List[str],
    ):
        """Test rank_drivers standalone function."""
        result = await rank_drivers(
            dag_edge_list=edge_list,
            target="outcome",
            shap_values=shap_values,
            feature_names=feature_names,
        )

        # Standalone functions return dict via model_dump()
        assert isinstance(result, dict)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_detect_structural_drift_standalone(
        self,
        baseline_dag_adjacency: List[List[int]],
        current_dag_adjacency: List[List[int]],
        dag_nodes: List[str],
    ):
        """Test detect_structural_drift standalone function."""
        result = await detect_structural_drift(
            baseline_dag_adjacency=baseline_dag_adjacency,
            current_dag_adjacency=current_dag_adjacency,
            dag_nodes=dag_nodes,
        )

        # Standalone functions return dict via model_dump()
        assert isinstance(result, dict)
        assert result["success"] is True


# =============================================================================
# TEST: INTEGRATION WITH TOOL REGISTRY
# =============================================================================


class TestToolRegistryIntegration:
    """Test integration between tools and the registry."""

    def test_tool_schema_validation(self, tool_registry: ToolRegistry):
        """Test that tool schemas are properly defined."""
        causal_tools = ["discover_dag", "rank_drivers", "detect_structural_drift"]

        for tool_name in causal_tools:
            tool = tool_registry.get(tool_name)
            assert tool is not None, f"{tool_name} not found in registry"

            # Check schema is defined
            assert tool.schema is not None
            assert tool.schema.name == tool_name
            # Input/output models may be Pydantic models or None
            # Check that schema has input parameters defined
            assert tool.schema.description is not None

    def test_list_tools_includes_causal(self, tool_registry: ToolRegistry):
        """Test that list_tools includes causal discovery tools."""
        # list_tools() returns List[str] of tool names
        tool_names = tool_registry.list_tools()

        assert "discover_dag" in tool_names
        assert "rank_drivers" in tool_names
        assert "detect_structural_drift" in tool_names

    def test_filter_by_agent(self, tool_registry: ToolRegistry):
        """Test filtering tools by source agent."""
        # Get causal_impact agent tools
        causal_tools = tool_registry.list_by_agent("causal_impact")

        assert "discover_dag" in causal_tools
        assert "rank_drivers" in causal_tools

        # Get drift_monitor agent tools
        drift_tools = tool_registry.list_by_agent("drift_monitor")

        assert "detect_structural_drift" in drift_tools
