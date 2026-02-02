"""Structural Drift Detector Node (V4.4).

This node detects drift in the causal structure (DAG) over time by comparing
the discovered causal graph from baseline vs current period data.

Structural drift occurs when the underlying causal relationships between
variables change, even if individual feature distributions remain stable.
This is critical for causal inference as it indicates that the assumed
causal model may no longer be valid.

Detection Methods:
1. Edge Comparison: Compare edges between baseline and current DAGs
2. Edge Type Comparison: Detect changes in edge types (DIRECTED -> BIDIRECTED)
3. Path Existence: Check if critical causal paths still exist

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md
Contract: .claude/contracts/tier3-contracts.md (V4.4 extension)
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import networkx as nx

from src.agents.drift_monitor.state import (
    DriftMonitorState,
    DriftResult,
    ErrorDetails,
    StructuralDriftResult,
)


class StructuralDriftNode:
    """Detects drift in causal DAG structure over time.

    Structural Drift Detection Strategy:
    1. Compare baseline DAG with current DAG (if both available)
    2. Calculate edge changes (added/removed edges)
    3. Compute drift score based on structural changes
    4. Generate severity assessment and recommendations

    Performance Target: <5s for structural drift check
    """

    # Drift score thresholds
    CRITICAL_THRESHOLD = 0.3  # 30%+ edge changes = critical
    HIGH_THRESHOLD = 0.2  # 20%+ edge changes = high
    MEDIUM_THRESHOLD = 0.1  # 10%+ edge changes = medium
    LOW_THRESHOLD = 0.05  # 5%+ edge changes = low

    def __init__(self) -> None:
        """Initialize structural drift node."""
        pass

    async def execute(self, state: DriftMonitorState) -> DriftMonitorState:
        """Execute structural drift detection.

        Compares baseline_dag vs current_dag to detect structural changes.

        Args:
            state: Current agent state with DAG data

        Returns:
            Updated state with structural_drift_results
        """
        start_time = time.time()

        # Check if structural drift detection is enabled
        if not state.get("check_structural_drift", True):
            state["structural_drift_results"] = []
            state["warnings"] = state.get("warnings", []) + [
                "Structural drift detection skipped (disabled)"
            ]
            return state

        # Skip if status is failed
        if state.get("status") == "failed":
            state["structural_drift_results"] = []
            return state

        try:
            # Get baseline and current DAG data
            baseline_adjacency = state.get("baseline_dag_adjacency")
            current_adjacency = state.get("current_dag_adjacency")
            dag_nodes = state.get("dag_nodes")

            # Check if we have DAG data to compare
            if baseline_adjacency is None or current_adjacency is None:
                state["structural_drift_results"] = []
                state["warnings"] = state.get("warnings", []) + [
                    "Structural drift detection skipped (no DAG data for comparison)"
                ]
                return state

            if dag_nodes is None:
                state["structural_drift_results"] = []
                state["warnings"] = state.get("warnings", []) + [
                    "Structural drift detection skipped (no node names provided)"
                ]
                return state

            # Build NetworkX graphs from adjacency matrices
            baseline_dag = self._adjacency_to_graph(baseline_adjacency, dag_nodes)
            current_dag = self._adjacency_to_graph(current_adjacency, dag_nodes)

            # Compare DAGs and detect drift
            drift_result = self._detect_structural_drift(
                baseline_dag=baseline_dag,
                current_dag=current_dag,
                baseline_edge_types=state.get("baseline_dag_edge_types"),
                current_edge_types=state.get("current_dag_edge_types"),
            )

            # Convert to DriftResult format for consistency
            drift_results: List[DriftResult] = []
            if drift_result.get("detected", False):
                drift_results.append(
                    DriftResult(
                        feature="causal_structure",
                        drift_type="structural",
                        test_statistic=drift_result.get("drift_score", 0.0),
                        p_value=0.0,  # No statistical test, using direct comparison
                        drift_detected=True,
                        severity=drift_result.get("severity", "none"),
                        baseline_period="baseline",
                        current_period="current",
                    )
                )

            # Update state
            state["structural_drift_results"] = drift_results
            state["structural_drift_details"] = drift_result

            if not drift_results:
                state["warnings"] = state.get("warnings", []) + [
                    "Structural drift detection completed - no drift detected"
                ]

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            state["total_latency_ms"] = state.get("total_latency_ms", 0) + latency_ms

        except Exception as e:
            error: ErrorDetails = {
                "node": "structural_drift",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["errors"] = state.get("errors", []) + [error]
            # Don't fail the whole pipeline for structural drift issues
            state["structural_drift_results"] = []
            state["warnings"] = state.get("warnings", []) + [
                f"Structural drift detection encountered an error: {str(e)}"
            ]

        return state

    def _adjacency_to_graph(
        self,
        adjacency: List[List[int]],
        nodes: List[str],
    ) -> nx.DiGraph:
        """Convert adjacency matrix to NetworkX DiGraph.

        Args:
            adjacency: Binary adjacency matrix
            nodes: Node names

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        G.add_nodes_from(nodes)

        n = len(nodes)
        for i in range(n):
            for j in range(n):
                if adjacency[i][j] == 1:
                    G.add_edge(nodes[i], nodes[j])

        return G

    def _detect_structural_drift(
        self,
        baseline_dag: nx.DiGraph,
        current_dag: nx.DiGraph,
        baseline_edge_types: Optional[Dict[str, str]] = None,
        current_edge_types: Optional[Dict[str, str]] = None,
    ) -> StructuralDriftResult:
        """Compare two DAGs to detect structural drift.

        Args:
            baseline_dag: Baseline period causal DAG
            current_dag: Current period causal DAG
            baseline_edge_types: Edge types for baseline (optional)
            current_edge_types: Edge types for current (optional)

        Returns:
            StructuralDriftResult with drift analysis
        """
        # Get edge sets
        baseline_edges = set(baseline_dag.edges())
        current_edges = set(current_dag.edges())

        # Calculate edge changes
        added_edges = current_edges - baseline_edges
        removed_edges = baseline_edges - current_edges
        stable_edges = baseline_edges & current_edges

        # Calculate drift score
        total_edges = len(baseline_edges | current_edges)
        changed_edges = len(added_edges) + len(removed_edges)
        drift_score = changed_edges / total_edges if total_edges > 0 else 0.0

        # Detect edge type changes (if edge types provided)
        edge_type_changes: List[Dict[str, Any]] = []
        if baseline_edge_types and current_edge_types:
            for edge in stable_edges:
                edge_key = f"{edge[0]}->{edge[1]}"
                baseline_type = baseline_edge_types.get(edge_key)
                current_type = current_edge_types.get(edge_key)
                if baseline_type and current_type and baseline_type != current_type:
                    edge_type_changes.append(
                        {
                            "edge": edge,
                            "baseline_type": baseline_type,
                            "current_type": current_type,
                        }
                    )

        # Determine severity
        severity = self._determine_severity(drift_score, edge_type_changes)

        # Check critical paths (if target/treatment available)
        # This is a placeholder for more sophisticated path analysis
        critical_path_broken = False

        # Generate recommendation
        recommendation = self._generate_recommendation(
            drift_score, severity, added_edges, removed_edges, edge_type_changes
        )

        return StructuralDriftResult(
            detected=drift_score > self.LOW_THRESHOLD or len(edge_type_changes) > 0,
            drift_score=drift_score,
            added_edges=[f"{e[0]}->{e[1]}" for e in added_edges],
            removed_edges=[f"{e[0]}->{e[1]}" for e in removed_edges],
            stable_edges=[f"{e[0]}->{e[1]}" for e in stable_edges],
            edge_type_changes=edge_type_changes,
            severity=severity,
            recommendation=recommendation,
            critical_path_broken=critical_path_broken,
        )

    def _determine_severity(
        self,
        drift_score: float,
        edge_type_changes: List[Dict[str, Any]],
    ) -> str:
        """Determine drift severity based on score and edge type changes.

        Args:
            drift_score: Percentage of edges changed
            edge_type_changes: List of edge type changes

        Returns:
            Severity level: "critical", "high", "medium", "low", or "none"
        """
        # Edge type changes (especially DIRECTED -> BIDIRECTED) are serious
        has_serious_type_changes = any(
            change.get("current_type") == "BIDIRECTED" for change in edge_type_changes
        )

        if drift_score >= self.CRITICAL_THRESHOLD or has_serious_type_changes:
            return "critical"
        elif drift_score >= self.HIGH_THRESHOLD:
            return "high"
        elif drift_score >= self.MEDIUM_THRESHOLD:
            return "medium"
        elif drift_score >= self.LOW_THRESHOLD:
            return "low"
        else:
            return "none"

    def _generate_recommendation(
        self,
        drift_score: float,
        severity: str,
        added_edges: set,
        removed_edges: set,
        edge_type_changes: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Generate recommendation based on drift analysis.

        Args:
            drift_score: Percentage of edges changed
            severity: Determined severity level
            added_edges: Set of newly added edges
            removed_edges: Set of removed edges
            edge_type_changes: List of edge type changes

        Returns:
            Recommendation string or None
        """
        recommendations = []

        if severity == "critical":
            recommendations.append(
                "CRITICAL: Causal structure has significantly changed. "
                "Re-run causal discovery and update all downstream models."
            )
        elif severity == "high":
            recommendations.append(
                "HIGH: Substantial structural changes detected. "
                "Review causal assumptions and consider model retraining."
            )
        elif severity == "medium":
            recommendations.append(
                "MEDIUM: Moderate structural changes detected. "
                "Monitor affected causal paths and validate key relationships."
            )
        elif severity == "low":
            recommendations.append(
                "LOW: Minor structural changes detected. "
                "Log for tracking, no immediate action required."
            )

        if removed_edges:
            recommendations.append(
                f"Removed edges ({len(removed_edges)}): {list(removed_edges)[:3]}..."
                if len(removed_edges) > 3
                else f"Removed edges: {list(removed_edges)}"
            )

        if added_edges:
            recommendations.append(
                f"Added edges ({len(added_edges)}): {list(added_edges)[:3]}..."
                if len(added_edges) > 3
                else f"Added edges: {list(added_edges)}"
            )

        if edge_type_changes:
            recommendations.append(
                f"Edge type changes ({len(edge_type_changes)}): "
                "May indicate latent confounders - consider IV methods."
            )

        return " ".join(recommendations) if recommendations else None


async def detect_structural_drift(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node function for structural drift detection.

    Args:
        state: Current state

    Returns:
        Updated state
    """
    node = StructuralDriftNode()
    return await node.execute(state)
