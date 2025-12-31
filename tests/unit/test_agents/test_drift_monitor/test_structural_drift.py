"""Tests for Structural Drift Detection (V4.4).

Tests the StructuralDriftNode that detects changes in causal DAG structure
between baseline and current periods.

Test Categories:
1. Basic drift detection with edge changes
2. Severity determination based on drift score
3. Edge type change detection
4. Edge case handling (empty DAGs, no changes)
5. State update validation
"""

import pytest

from src.agents.drift_monitor.nodes.structural_drift_detector import StructuralDriftNode
from src.agents.drift_monitor.state import DriftMonitorState


@pytest.fixture
def structural_drift_node() -> StructuralDriftNode:
    """Create a StructuralDriftNode instance for testing."""
    return StructuralDriftNode()


@pytest.fixture
def base_state() -> DriftMonitorState:
    """Create a base state for testing."""
    return DriftMonitorState(
        query="test query",
        features_to_monitor=["feature1", "feature2"],
        time_window="7d",
        significance_level=0.05,
        psi_threshold=0.1,
        check_data_drift=True,
        check_model_drift=True,
        check_concept_drift=True,
        check_structural_drift=True,
        errors=[],
        warnings=[],
        status="detecting",
    )


class TestStructuralDriftNodeBasics:
    """Basic structural drift detection tests."""

    @pytest.mark.asyncio
    async def test_no_drift_when_dags_identical(
        self,
        structural_drift_node: StructuralDriftNode,
        base_state: DriftMonitorState,
    ) -> None:
        """Test that identical DAGs show no drift."""
        # Same adjacency matrix for baseline and current
        adjacency = [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
        nodes = ["A", "B", "C"]

        state = {
            **base_state,
            "baseline_dag_adjacency": adjacency,
            "current_dag_adjacency": adjacency,
            "dag_nodes": nodes,
        }

        result = await structural_drift_node.execute(state)

        assert result["structural_drift_results"] == []
        assert result.get("structural_drift_details") is not None
        assert result["structural_drift_details"]["detected"] is False
        assert result["structural_drift_details"]["drift_score"] == 0.0

    @pytest.mark.asyncio
    async def test_drift_detected_when_edges_added(
        self,
        structural_drift_node: StructuralDriftNode,
        base_state: DriftMonitorState,
    ) -> None:
        """Test that added edges are detected as drift."""
        # Baseline: A -> B -> C
        baseline_adjacency = [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
        # Current: A -> B -> C, A -> C (added edge)
        current_adjacency = [
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ]
        nodes = ["A", "B", "C"]

        state = {
            **base_state,
            "baseline_dag_adjacency": baseline_adjacency,
            "current_dag_adjacency": current_adjacency,
            "dag_nodes": nodes,
        }

        result = await structural_drift_node.execute(state)

        details = result["structural_drift_details"]
        assert details["detected"] is True
        assert "A->C" in details["added_edges"]
        assert len(details["removed_edges"]) == 0
        assert details["drift_score"] > 0.0

    @pytest.mark.asyncio
    async def test_drift_detected_when_edges_removed(
        self,
        structural_drift_node: StructuralDriftNode,
        base_state: DriftMonitorState,
    ) -> None:
        """Test that removed edges are detected as drift."""
        # Baseline: A -> B -> C
        baseline_adjacency = [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
        # Current: A -> B (removed B -> C)
        current_adjacency = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        nodes = ["A", "B", "C"]

        state = {
            **base_state,
            "baseline_dag_adjacency": baseline_adjacency,
            "current_dag_adjacency": current_adjacency,
            "dag_nodes": nodes,
        }

        result = await structural_drift_node.execute(state)

        details = result["structural_drift_details"]
        assert details["detected"] is True
        assert "B->C" in details["removed_edges"]
        assert len(details["added_edges"]) == 0


class TestStructuralDriftSeverity:
    """Test severity determination based on drift score."""

    @pytest.mark.asyncio
    async def test_low_severity_for_small_drift(
        self,
        structural_drift_node: StructuralDriftNode,
        base_state: DriftMonitorState,
    ) -> None:
        """Test low severity for drift between 5% and 10%."""
        # Large DAG with many edges so one change is <10%
        # We need >10 edges so that 1 change is <10%
        baseline_adjacency = [
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # N0 -> N1,N2,N3,N4 (4 edges)
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # N1 -> N5,N6 (2 edges)
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # N2 -> N7 (1 edge)
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # N3 -> N8 (1 edge)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # N4 -> N9 (1 edge)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # N5 -> N10 (1 edge)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # N6 -> N11 (1 edge)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # N7 (leaf)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # N8 (leaf)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # N9 (leaf)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # N10 (leaf)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # N11 (leaf)
        ]
        # Total baseline edges: 4+2+1+1+1+1+1 = 11 edges
        # Add one edge: drift = 1/12 = 8.3% -> low severity
        current_adjacency = [row[:] for row in baseline_adjacency]
        current_adjacency[7][8] = 1  # Add one edge N7->N8

        nodes = [f"N{i}" for i in range(12)]

        state = {
            **base_state,
            "baseline_dag_adjacency": baseline_adjacency,
            "current_dag_adjacency": current_adjacency,
            "dag_nodes": nodes,
        }

        result = await structural_drift_node.execute(state)

        details = result["structural_drift_details"]
        assert details["detected"] is True
        # 1/12 = 8.3% is between 5% and 10% -> low severity
        assert details["severity"] == "low"

    @pytest.mark.asyncio
    async def test_critical_severity_for_large_drift(
        self,
        structural_drift_node: StructuralDriftNode,
        base_state: DriftMonitorState,
    ) -> None:
        """Test critical severity for >30% edge changes."""
        # Baseline: Complete different structure
        baseline_adjacency = [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
        # Current: Completely reversed
        current_adjacency = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
        nodes = ["A", "B", "C"]

        state = {
            **base_state,
            "baseline_dag_adjacency": baseline_adjacency,
            "current_dag_adjacency": current_adjacency,
            "dag_nodes": nodes,
        }

        result = await structural_drift_node.execute(state)

        details = result["structural_drift_details"]
        assert details["detected"] is True
        assert details["severity"] == "critical"
        assert details["drift_score"] >= 0.3


class TestEdgeTypeChanges:
    """Test edge type change detection."""

    @pytest.mark.asyncio
    async def test_edge_type_change_detection(
        self,
        structural_drift_node: StructuralDriftNode,
        base_state: DriftMonitorState,
    ) -> None:
        """Test detection of edge type changes (DIRECTED -> BIDIRECTED)."""
        adjacency = [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
        nodes = ["A", "B", "C"]

        # Edge type changes
        baseline_edge_types = {
            "A->B": "DIRECTED",
            "B->C": "DIRECTED",
        }
        current_edge_types = {
            "A->B": "BIDIRECTED",  # Changed to latent confounder
            "B->C": "DIRECTED",
        }

        state = {
            **base_state,
            "baseline_dag_adjacency": adjacency,
            "current_dag_adjacency": adjacency,
            "dag_nodes": nodes,
            "baseline_dag_edge_types": baseline_edge_types,
            "current_dag_edge_types": current_edge_types,
        }

        result = await structural_drift_node.execute(state)

        details = result["structural_drift_details"]
        # Edge type change to BIDIRECTED should trigger critical severity
        assert details["detected"] is True
        assert details["severity"] == "critical"
        assert len(details.get("edge_type_changes", [])) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_skipped_when_disabled(
        self,
        structural_drift_node: StructuralDriftNode,
        base_state: DriftMonitorState,
    ) -> None:
        """Test that detection is skipped when disabled."""
        state = {
            **base_state,
            "check_structural_drift": False,
        }

        result = await structural_drift_node.execute(state)

        assert result["structural_drift_results"] == []
        assert "skipped (disabled)" in str(result.get("warnings", []))

    @pytest.mark.asyncio
    async def test_skipped_when_no_dag_data(
        self,
        structural_drift_node: StructuralDriftNode,
        base_state: DriftMonitorState,
    ) -> None:
        """Test that detection is skipped when no DAG data provided."""
        state = {
            **base_state,
            # No DAG data
        }

        result = await structural_drift_node.execute(state)

        assert result["structural_drift_results"] == []
        assert "no DAG data" in str(result.get("warnings", []))

    @pytest.mark.asyncio
    async def test_skipped_when_status_failed(
        self,
        structural_drift_node: StructuralDriftNode,
        base_state: DriftMonitorState,
    ) -> None:
        """Test that detection is skipped when status is failed."""
        state = {
            **base_state,
            "status": "failed",
            "baseline_dag_adjacency": [[0, 1], [0, 0]],
            "current_dag_adjacency": [[0, 1], [0, 0]],
            "dag_nodes": ["A", "B"],
        }

        result = await structural_drift_node.execute(state)

        assert result["structural_drift_results"] == []

    @pytest.mark.asyncio
    async def test_handles_empty_dags(
        self,
        structural_drift_node: StructuralDriftNode,
        base_state: DriftMonitorState,
    ) -> None:
        """Test handling of empty DAGs (no edges)."""
        empty_adjacency = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        nodes = ["A", "B", "C"]

        state = {
            **base_state,
            "baseline_dag_adjacency": empty_adjacency,
            "current_dag_adjacency": empty_adjacency,
            "dag_nodes": nodes,
        }

        result = await structural_drift_node.execute(state)

        details = result.get("structural_drift_details")
        assert details is not None
        assert details["detected"] is False
        assert details["drift_score"] == 0.0


class TestStateUpdates:
    """Test that state is properly updated."""

    @pytest.mark.asyncio
    async def test_latency_tracking(
        self,
        structural_drift_node: StructuralDriftNode,
        base_state: DriftMonitorState,
    ) -> None:
        """Test that detection latency is tracked."""
        adjacency = [[0, 1], [0, 0]]
        nodes = ["A", "B"]

        state = {
            **base_state,
            "baseline_dag_adjacency": adjacency,
            "current_dag_adjacency": adjacency,
            "dag_nodes": nodes,
            "detection_latency_ms": 100,  # Existing latency
        }

        result = await structural_drift_node.execute(state)

        # Latency should be added to existing
        assert result["detection_latency_ms"] >= 100

    @pytest.mark.asyncio
    async def test_recommendations_generated(
        self,
        structural_drift_node: StructuralDriftNode,
        base_state: DriftMonitorState,
    ) -> None:
        """Test that recommendations are generated for drift."""
        # Large drift scenario
        baseline_adjacency = [
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ]
        current_adjacency = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],
        ]
        nodes = ["A", "B", "C"]

        state = {
            **base_state,
            "baseline_dag_adjacency": baseline_adjacency,
            "current_dag_adjacency": current_adjacency,
            "dag_nodes": nodes,
        }

        result = await structural_drift_node.execute(state)

        details = result["structural_drift_details"]
        assert details["recommendation"] is not None
        assert len(details["recommendation"]) > 0
