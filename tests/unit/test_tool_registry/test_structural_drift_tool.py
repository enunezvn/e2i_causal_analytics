"""Unit tests for StructuralDriftTool.

Tests cover:
- Input/output schema validation
- Drift detection logic
- Severity thresholds
- Edge type change detection
- Registry integration
- Error handling
"""

from datetime import datetime, timezone

import pytest

from src.tool_registry.tools.structural_drift import (
    CRITICAL_THRESHOLD,
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    MEDIUM_THRESHOLD,
    EdgeTypeChange,
    StructuralDriftInput,
    StructuralDriftOutput,
    StructuralDriftTool,
    detect_structural_drift,
    get_structural_drift_tool,
    register_structural_drift_tool,
)

# =============================================================================
# INPUT/OUTPUT SCHEMA TESTS
# =============================================================================


class TestStructuralDriftInput:
    """Tests for StructuralDriftInput schema."""

    def test_minimal_input(self):
        """Test minimal valid input."""
        inp = StructuralDriftInput(
            baseline_dag_adjacency=[[0, 1], [0, 0]],
            current_dag_adjacency=[[0, 1], [0, 0]],
            dag_nodes=["A", "B"],
        )
        assert len(inp.dag_nodes) == 2
        assert inp.baseline_edge_types is None
        assert inp.current_edge_types is None

    def test_full_input(self):
        """Test full input with all fields."""
        inp = StructuralDriftInput(
            baseline_dag_adjacency=[[0, 1, 0], [0, 0, 1], [0, 0, 0]],
            current_dag_adjacency=[[0, 1, 1], [0, 0, 0], [0, 0, 0]],
            dag_nodes=["A", "B", "C"],
            baseline_edge_types={"A->B": "DIRECTED", "B->C": "DIRECTED"},
            current_edge_types={"A->B": "DIRECTED", "A->C": "DIRECTED"},
            trace_context={"trace_id": "abc123"},
        )
        assert len(inp.dag_nodes) == 3
        assert inp.baseline_edge_types["A->B"] == "DIRECTED"
        assert inp.trace_context["trace_id"] == "abc123"


class TestStructuralDriftOutput:
    """Tests for StructuralDriftOutput schema."""

    def test_minimal_output(self):
        """Test minimal valid output."""
        out = StructuralDriftOutput(
            success=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        assert out.success is True
        assert out.detected is False
        assert out.drift_score == 0.0
        assert out.severity == "none"
        assert out.errors == []

    def test_full_output(self):
        """Test output with all fields."""
        out = StructuralDriftOutput(
            success=True,
            detected=True,
            drift_score=0.35,
            severity="critical",
            added_edges=["A->C"],
            removed_edges=["B->C"],
            stable_edges=["A->B"],
            edge_type_changes=[
                EdgeTypeChange(
                    edge="A->B",
                    baseline_type="DIRECTED",
                    current_type="BIDIRECTED",
                )
            ],
            recommendation="CRITICAL: Re-run causal discovery.",
            critical_path_broken=True,
            timestamp="2024-01-01T00:00:00Z",
            trace_id="abc123",
            errors=[],
        )
        assert out.drift_score == 0.35
        assert out.severity == "critical"
        assert len(out.edge_type_changes) == 1
        assert out.edge_type_changes[0].current_type == "BIDIRECTED"


class TestEdgeTypeChange:
    """Tests for EdgeTypeChange schema."""

    def test_edge_type_change(self):
        """Test EdgeTypeChange schema."""
        change = EdgeTypeChange(
            edge="A->B",
            baseline_type="DIRECTED",
            current_type="BIDIRECTED",
        )
        assert change.edge == "A->B"
        assert change.baseline_type == "DIRECTED"
        assert change.current_type == "BIDIRECTED"


# =============================================================================
# TOOL TESTS
# =============================================================================


class TestStructuralDriftTool:
    """Tests for StructuralDriftTool."""

    @pytest.fixture
    def tool(self):
        """Create a fresh tool instance."""
        return StructuralDriftTool(opik_enabled=False)

    @pytest.mark.asyncio
    async def test_no_drift_identical_dags(self, tool):
        """Test detection with identical DAGs (no drift)."""
        # A -> B -> C
        adjacency = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
        nodes = ["A", "B", "C"]

        result = await tool.invoke(
            StructuralDriftInput(
                baseline_dag_adjacency=adjacency,
                current_dag_adjacency=adjacency,
                dag_nodes=nodes,
            )
        )

        assert result.success is True
        assert result.detected is False
        assert result.drift_score == 0.0
        assert result.severity == "none"
        assert len(result.added_edges) == 0
        assert len(result.removed_edges) == 0
        assert len(result.stable_edges) == 2  # A->B, B->C

    @pytest.mark.asyncio
    async def test_low_drift(self, tool):
        """Test detection with low drift (1 edge change out of many)."""
        # Baseline: A -> B -> C -> D (3 edges)
        baseline = [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]
        # Current: A -> B -> C, D removed (2 edges, 1 removed)
        # Drift score = 1/3 = 0.33 (changed/total)
        current = [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        nodes = ["A", "B", "C", "D"]

        result = await tool.invoke(
            StructuralDriftInput(
                baseline_dag_adjacency=baseline,
                current_dag_adjacency=current,
                dag_nodes=nodes,
            )
        )

        assert result.success is True
        assert result.detected is True
        assert len(result.removed_edges) == 1
        assert "C->D" in result.removed_edges

    @pytest.mark.asyncio
    async def test_medium_drift(self, tool):
        """Test detection with medium drift."""
        # Baseline: A -> B, B -> C (2 edges)
        baseline = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
        # Current: A -> C only (1 edge added, 2 removed) -> 3/3 = 1.0 drift
        current = [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
        nodes = ["A", "B", "C"]

        result = await tool.invoke(
            StructuralDriftInput(
                baseline_dag_adjacency=baseline,
                current_dag_adjacency=current,
                dag_nodes=nodes,
            )
        )

        assert result.success is True
        assert result.detected is True
        assert result.severity == "critical"  # 100% edges changed
        assert result.drift_score == 1.0

    @pytest.mark.asyncio
    async def test_critical_drift_threshold(self, tool):
        """Test that critical threshold is applied correctly."""
        # Create a DAG where 30%+ edges change
        # 10 edges baseline, 4 change = 40% drift
        baseline = [
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
        # Remove 3 edges, add 2 new ones
        current = [
            [0, 1, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        nodes = ["A", "B", "C", "D", "E"]

        result = await tool.invoke(
            StructuralDriftInput(
                baseline_dag_adjacency=baseline,
                current_dag_adjacency=current,
                dag_nodes=nodes,
            )
        )

        assert result.success is True
        assert result.detected is True
        assert result.drift_score >= CRITICAL_THRESHOLD
        assert result.severity == "critical"
        assert "CRITICAL" in result.recommendation

    @pytest.mark.asyncio
    async def test_edge_added(self, tool):
        """Test detection of added edges."""
        # Baseline: A -> B
        baseline = [[0, 1], [0, 0]]
        # Current: A -> B, B -> A (added reverse edge)
        current = [[0, 1], [1, 0]]
        nodes = ["A", "B"]

        result = await tool.invoke(
            StructuralDriftInput(
                baseline_dag_adjacency=baseline,
                current_dag_adjacency=current,
                dag_nodes=nodes,
            )
        )

        assert result.success is True
        assert result.detected is True
        assert len(result.added_edges) == 1
        assert "B->A" in result.added_edges
        assert len(result.removed_edges) == 0

    @pytest.mark.asyncio
    async def test_edge_removed(self, tool):
        """Test detection of removed edges."""
        # Baseline: A -> B, B -> A
        baseline = [[0, 1], [1, 0]]
        # Current: A -> B only
        current = [[0, 1], [0, 0]]
        nodes = ["A", "B"]

        result = await tool.invoke(
            StructuralDriftInput(
                baseline_dag_adjacency=baseline,
                current_dag_adjacency=current,
                dag_nodes=nodes,
            )
        )

        assert result.success is True
        assert result.detected is True
        assert len(result.removed_edges) == 1
        assert "B->A" in result.removed_edges
        assert len(result.added_edges) == 0

    @pytest.mark.asyncio
    async def test_edge_type_change_detection(self, tool):
        """Test detection of edge type changes."""
        # Both have A -> B, but type changes from DIRECTED to BIDIRECTED
        adjacency = [[0, 1], [0, 0]]
        nodes = ["A", "B"]

        result = await tool.invoke(
            StructuralDriftInput(
                baseline_dag_adjacency=adjacency,
                current_dag_adjacency=adjacency,
                dag_nodes=nodes,
                baseline_edge_types={"A->B": "DIRECTED"},
                current_edge_types={"A->B": "BIDIRECTED"},
            )
        )

        assert result.success is True
        assert result.detected is True  # Edge type change triggers detection
        assert len(result.edge_type_changes) == 1
        assert result.edge_type_changes[0].baseline_type == "DIRECTED"
        assert result.edge_type_changes[0].current_type == "BIDIRECTED"
        # BIDIRECTED change triggers critical severity
        assert result.severity == "critical"

    @pytest.mark.asyncio
    async def test_empty_dags(self, tool):
        """Test with empty DAGs (no edges)."""
        adjacency = [[0, 0], [0, 0]]
        nodes = ["A", "B"]

        result = await tool.invoke(
            StructuralDriftInput(
                baseline_dag_adjacency=adjacency,
                current_dag_adjacency=adjacency,
                dag_nodes=nodes,
            )
        )

        assert result.success is True
        assert result.detected is False
        assert result.drift_score == 0.0
        assert len(result.stable_edges) == 0

    @pytest.mark.asyncio
    async def test_invalid_adjacency_dimensions(self, tool):
        """Test error handling for mismatched dimensions."""
        result = await tool.invoke(
            StructuralDriftInput(
                baseline_dag_adjacency=[[0, 1], [0, 0]],
                current_dag_adjacency=[[0, 1, 0], [0, 0, 0], [0, 0, 0]],  # Wrong size
                dag_nodes=["A", "B"],
            )
        )

        assert result.success is False
        assert len(result.errors) > 0
        assert "doesn't match" in result.errors[0]

    @pytest.mark.asyncio
    async def test_dict_input(self, tool):
        """Test with dictionary input."""
        result = await tool.invoke(
            {
                "baseline_dag_adjacency": [[0, 1], [0, 0]],
                "current_dag_adjacency": [[0, 1], [0, 0]],
                "dag_nodes": ["A", "B"],
            }
        )

        assert result.success is True
        assert result.detected is False


class TestSeverityThresholds:
    """Tests for severity threshold logic."""

    @pytest.fixture
    def tool(self):
        """Create a fresh tool instance."""
        return StructuralDriftTool(opik_enabled=False)

    def test_threshold_values(self):
        """Verify threshold constants are correctly defined."""
        assert CRITICAL_THRESHOLD == 0.3
        assert HIGH_THRESHOLD == 0.2
        assert MEDIUM_THRESHOLD == 0.1
        assert LOW_THRESHOLD == 0.05

    def test_severity_none(self, tool):
        """Test severity determination for no drift."""
        severity = tool._determine_severity(0.0, [])
        assert severity == "none"

    def test_severity_low(self, tool):
        """Test severity determination for low drift."""
        severity = tool._determine_severity(0.06, [])
        assert severity == "low"

    def test_severity_medium(self, tool):
        """Test severity determination for medium drift."""
        severity = tool._determine_severity(0.15, [])
        assert severity == "medium"

    def test_severity_high(self, tool):
        """Test severity determination for high drift."""
        severity = tool._determine_severity(0.25, [])
        assert severity == "high"

    def test_severity_critical(self, tool):
        """Test severity determination for critical drift."""
        severity = tool._determine_severity(0.35, [])
        assert severity == "critical"

    def test_severity_bidirected_escalates_to_critical(self, tool):
        """Test that BIDIRECTED edge type change escalates to critical."""
        edge_changes = [
            EdgeTypeChange(
                edge="A->B",
                baseline_type="DIRECTED",
                current_type="BIDIRECTED",
            )
        ]
        # Even with low drift score, BIDIRECTED change should trigger critical
        severity = tool._determine_severity(0.01, edge_changes)
        assert severity == "critical"


class TestRecommendations:
    """Tests for recommendation generation."""

    @pytest.fixture
    def tool(self):
        """Create a fresh tool instance."""
        return StructuralDriftTool(opik_enabled=False)

    def test_critical_recommendation(self, tool):
        """Test critical severity recommendation."""
        rec = tool._generate_recommendation(0.35, "critical", ["A->C"], ["B->D"], [])
        assert "CRITICAL" in rec
        assert "Re-run causal discovery" in rec

    def test_high_recommendation(self, tool):
        """Test high severity recommendation."""
        rec = tool._generate_recommendation(0.25, "high", ["A->C"], [], [])
        assert "HIGH" in rec
        assert "model retraining" in rec

    def test_medium_recommendation(self, tool):
        """Test medium severity recommendation."""
        rec = tool._generate_recommendation(0.15, "medium", [], ["B->D"], [])
        assert "MEDIUM" in rec
        assert "Monitor" in rec

    def test_low_recommendation(self, tool):
        """Test low severity recommendation."""
        rec = tool._generate_recommendation(0.06, "low", [], [], [])
        assert "LOW" in rec
        assert "no immediate action" in rec

    def test_edge_type_changes_recommendation(self, tool):
        """Test recommendation includes edge type change info."""
        edge_changes = [
            EdgeTypeChange(
                edge="A->B",
                baseline_type="DIRECTED",
                current_type="BIDIRECTED",
            )
        ]
        rec = tool._generate_recommendation(0.01, "low", [], [], edge_changes)
        assert "Edge type changes" in rec
        assert "IV methods" in rec

    def test_no_recommendation_for_none_severity(self, tool):
        """Test no recommendation for no drift."""
        rec = tool._generate_recommendation(0.0, "none", [], [], [])
        assert rec is None


# =============================================================================
# FUNCTION INTERFACE TESTS
# =============================================================================


class TestDetectStructuralDriftFunction:
    """Tests for the detect_structural_drift function."""

    @pytest.mark.asyncio
    async def test_function_interface(self):
        """Test the function interface returns correct structure."""
        result = await detect_structural_drift(
            baseline_dag_adjacency=[[0, 1], [0, 0]],
            current_dag_adjacency=[[0, 1], [0, 0]],
            dag_nodes=["A", "B"],
        )

        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["detected"] is False
        assert "drift_score" in result
        assert "severity" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_function_with_edge_types(self):
        """Test function with edge type parameters."""
        result = await detect_structural_drift(
            baseline_dag_adjacency=[[0, 1], [0, 0]],
            current_dag_adjacency=[[0, 1], [0, 0]],
            dag_nodes=["A", "B"],
            baseline_edge_types={"A->B": "DIRECTED"},
            current_edge_types={"A->B": "DIRECTED"},
        )

        assert isinstance(result, dict)
        assert result["success"] is True


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_get_structural_drift_tool_returns_singleton(self):
        """Test that singleton pattern works."""
        # Reset singleton for clean test
        import src.tool_registry.tools.structural_drift as module

        module._drift_tool_instance = None

        tool1 = get_structural_drift_tool()
        tool2 = get_structural_drift_tool()

        assert tool1 is tool2

        # Cleanup
        module._drift_tool_instance = None


# =============================================================================
# REGISTRY INTEGRATION TESTS
# =============================================================================


class TestToolRegistration:
    """Tests for tool registry integration."""

    def test_tool_is_registered(self):
        """Test that tool is registered in the registry."""
        from src.tool_registry.registry import get_registry

        # Re-register to ensure it's there
        register_structural_drift_tool()

        registry = get_registry()
        assert registry.validate_tool_exists("detect_structural_drift")

    def test_tool_schema_complete(self):
        """Test that tool schema has all required fields."""
        from src.tool_registry.registry import get_registry

        register_structural_drift_tool()

        registry = get_registry()
        schema = registry.get_schema("detect_structural_drift")

        assert schema is not None
        assert schema.name == "detect_structural_drift"
        assert schema.source_agent == "drift_monitor"
        assert schema.tier == 3
        assert len(schema.input_parameters) >= 3  # Required params

    def test_tool_callable_exists(self):
        """Test that tool callable is registered."""
        from src.tool_registry.registry import get_registry

        register_structural_drift_tool()

        registry = get_registry()
        callable_func = registry.get_callable("detect_structural_drift")

        assert callable_func is not None
        assert callable(callable_func)

    def test_tool_in_agent_list(self):
        """Test that tool appears in agent's tool list."""
        from src.tool_registry.registry import get_registry

        register_structural_drift_tool()

        registry = get_registry()
        agent_tools = registry.list_by_agent("drift_monitor")

        assert "detect_structural_drift" in agent_tools

    def test_tool_in_tier_list(self):
        """Test that tool appears in tier 3 tool list."""
        from src.tool_registry.registry import get_registry

        register_structural_drift_tool()

        registry = get_registry()
        tier3_tools = registry.list_by_tier(3)

        assert "detect_structural_drift" in tier3_tools


# =============================================================================
# GRAPH CONVERSION TESTS
# =============================================================================


class TestGraphConversion:
    """Tests for adjacency matrix to graph conversion."""

    @pytest.fixture
    def tool(self):
        """Create a fresh tool instance."""
        return StructuralDriftTool(opik_enabled=False)

    def test_simple_graph_conversion(self, tool):
        """Test conversion of simple adjacency matrix."""
        adjacency = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
        nodes = ["A", "B", "C"]

        graph = tool._adjacency_to_graph(adjacency, nodes)

        assert len(graph.nodes()) == 3
        assert len(graph.edges()) == 2
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "C")
        assert not graph.has_edge("A", "C")

    def test_empty_graph_conversion(self, tool):
        """Test conversion of empty adjacency matrix."""
        adjacency = [[0, 0], [0, 0]]
        nodes = ["A", "B"]

        graph = tool._adjacency_to_graph(adjacency, nodes)

        assert len(graph.nodes()) == 2
        assert len(graph.edges()) == 0

    def test_fully_connected_graph(self, tool):
        """Test conversion of fully connected graph."""
        adjacency = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        nodes = ["A", "B", "C"]

        graph = tool._adjacency_to_graph(adjacency, nodes)

        assert len(graph.nodes()) == 3
        assert len(graph.edges()) == 6  # All edges except self-loops
