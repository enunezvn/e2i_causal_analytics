"""Structural Drift Detection Tool for Agent Workflows.

This module provides a composable tool for detecting drift in causal
DAG structure over time by comparing baseline vs current period DAGs.

Version: 1.0.0

Tools:
- detect_structural_drift: Compare two DAGs to detect structural changes

Usage:
------
    from src.tool_registry.tools.structural_drift import detect_structural_drift

    # Detect structural drift
    result = await detect_structural_drift(
        baseline_dag_adjacency=baseline_adj,
        current_dag_adjacency=current_adj,
        dag_nodes=node_names,
    )

Author: E2I Causal Analytics Team
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import networkx as nx
from pydantic import BaseModel, Field

from src.tool_registry.registry import ToolParameter, ToolSchema, get_registry

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================


class StructuralDriftInput(BaseModel):
    """Input schema for detect_structural_drift tool."""

    baseline_dag_adjacency: List[List[int]] = Field(
        ...,
        description="Baseline DAG as binary adjacency matrix (n x n)",
    )
    current_dag_adjacency: List[List[int]] = Field(
        ...,
        description="Current DAG as binary adjacency matrix (n x n)",
    )
    dag_nodes: List[str] = Field(
        ...,
        description="Node names in the same order as adjacency matrix rows/columns",
    )
    baseline_edge_types: Optional[Dict[str, str]] = Field(
        default=None,
        description="Edge types for baseline DAG, keys as 'source->target'",
    )
    current_edge_types: Optional[Dict[str, str]] = Field(
        default=None,
        description="Edge types for current DAG, keys as 'source->target'",
    )
    trace_context: Optional[Dict[str, str]] = Field(
        default=None,
        description="Opik trace context for distributed tracing",
    )


class EdgeTypeChange(BaseModel):
    """Details about an edge type change."""

    edge: str = Field(..., description="Edge as 'source->target'")
    baseline_type: str = Field(..., description="Edge type in baseline DAG")
    current_type: str = Field(..., description="Edge type in current DAG")


class StructuralDriftOutput(BaseModel):
    """Output schema for detect_structural_drift tool."""

    success: bool = Field(..., description="Whether detection succeeded")
    detected: bool = Field(default=False, description="Whether structural drift was detected")
    drift_score: float = Field(
        default=0.0,
        description="Proportion of edges changed (0-1)",
    )
    severity: str = Field(
        default="none",
        description="Drift severity: 'critical', 'high', 'medium', 'low', 'none'",
    )
    added_edges: List[str] = Field(
        default_factory=list,
        description="Edges added in current DAG (format: 'source->target')",
    )
    removed_edges: List[str] = Field(
        default_factory=list,
        description="Edges removed from baseline DAG",
    )
    stable_edges: List[str] = Field(
        default_factory=list,
        description="Edges present in both DAGs",
    )
    edge_type_changes: List[EdgeTypeChange] = Field(
        default_factory=list,
        description="Edges with changed types (e.g., DIRECTED->BIDIRECTED)",
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Recommendation based on drift analysis",
    )
    critical_path_broken: bool = Field(
        default=False,
        description="Whether a critical causal path was broken",
    )
    timestamp: str = Field(..., description="Detection timestamp (ISO format)")
    trace_id: Optional[str] = Field(
        default=None,
        description="Opik trace ID for this detection",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during detection",
    )


# =============================================================================
# TOOL IMPLEMENTATION
# =============================================================================

# Drift score thresholds (from StructuralDriftNode)
CRITICAL_THRESHOLD = 0.3  # 30%+ edge changes = critical
HIGH_THRESHOLD = 0.2  # 20%+ edge changes = high
MEDIUM_THRESHOLD = 0.1  # 10%+ edge changes = medium
LOW_THRESHOLD = 0.05  # 5%+ edge changes = low


@dataclass
class StructuralDriftTool:
    """
    Tool for detecting drift in causal DAG structure over time.

    Compares baseline and current DAGs to identify:
    - Added/removed edges
    - Edge type changes
    - Overall structural drift severity

    Attributes:
        opik_enabled: Whether Opik tracing is enabled
    """

    opik_enabled: bool = field(default=True)

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

    def _determine_severity(
        self,
        drift_score: float,
        edge_type_changes: List[EdgeTypeChange],
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
            change.current_type == "BIDIRECTED" for change in edge_type_changes
        )

        if drift_score >= CRITICAL_THRESHOLD or has_serious_type_changes:
            return "critical"
        elif drift_score >= HIGH_THRESHOLD:
            return "high"
        elif drift_score >= MEDIUM_THRESHOLD:
            return "medium"
        elif drift_score >= LOW_THRESHOLD:
            return "low"
        else:
            return "none"

    def _generate_recommendation(
        self,
        drift_score: float,
        severity: str,
        added_edges: List[str],
        removed_edges: List[str],
        edge_type_changes: List[EdgeTypeChange],
    ) -> Optional[str]:
        """Generate recommendation based on drift analysis.

        Args:
            drift_score: Percentage of edges changed
            severity: Determined severity level
            added_edges: List of newly added edges
            removed_edges: List of removed edges
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
            edge_preview = removed_edges[:3]
            if len(removed_edges) > 3:
                recommendations.append(f"Removed edges ({len(removed_edges)}): {edge_preview}...")
            else:
                recommendations.append(f"Removed edges: {removed_edges}")

        if added_edges:
            edge_preview = added_edges[:3]
            if len(added_edges) > 3:
                recommendations.append(f"Added edges ({len(added_edges)}): {edge_preview}...")
            else:
                recommendations.append(f"Added edges: {added_edges}")

        if edge_type_changes:
            recommendations.append(
                f"Edge type changes ({len(edge_type_changes)}): "
                "May indicate latent confounders - consider IV methods."
            )

        return " ".join(recommendations) if recommendations else None

    async def invoke(
        self,
        input_data: Union[Dict[str, Any], StructuralDriftInput],
    ) -> StructuralDriftOutput:
        """
        Detect structural drift between baseline and current DAGs.

        Args:
            input_data: Either a dict or StructuralDriftInput with parameters

        Returns:
            StructuralDriftOutput with drift analysis
        """
        # Parse input
        if isinstance(input_data, dict):
            params = StructuralDriftInput(**input_data)
        else:
            params = input_data

        # Start trace
        trace_id = self._start_trace(params) if self.opik_enabled else None

        errors: List[str] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Validate input dimensions
            n_nodes = len(params.dag_nodes)
            if len(params.baseline_dag_adjacency) != n_nodes:
                raise ValueError(
                    f"Baseline adjacency matrix rows ({len(params.baseline_dag_adjacency)}) "
                    f"doesn't match node count ({n_nodes})"
                )
            if len(params.current_dag_adjacency) != n_nodes:
                raise ValueError(
                    f"Current adjacency matrix rows ({len(params.current_dag_adjacency)}) "
                    f"doesn't match node count ({n_nodes})"
                )

            # Build NetworkX graphs
            baseline_dag = self._adjacency_to_graph(params.baseline_dag_adjacency, params.dag_nodes)
            current_dag = self._adjacency_to_graph(params.current_dag_adjacency, params.dag_nodes)

            # Get edge sets
            baseline_edges = set(baseline_dag.edges())
            current_edges = set(current_dag.edges())

            # Calculate edge changes
            added_edges_set = current_edges - baseline_edges
            removed_edges_set = baseline_edges - current_edges
            stable_edges_set = baseline_edges & current_edges

            # Calculate drift score
            total_edges = len(baseline_edges | current_edges)
            changed_edges = len(added_edges_set) + len(removed_edges_set)
            drift_score = changed_edges / total_edges if total_edges > 0 else 0.0

            # Detect edge type changes (if edge types provided)
            edge_type_changes: List[EdgeTypeChange] = []
            if params.baseline_edge_types and params.current_edge_types:
                for edge in stable_edges_set:
                    edge_key = f"{edge[0]}->{edge[1]}"
                    baseline_type = params.baseline_edge_types.get(edge_key)
                    current_type = params.current_edge_types.get(edge_key)
                    if baseline_type and current_type and baseline_type != current_type:
                        edge_type_changes.append(
                            EdgeTypeChange(
                                edge=edge_key,
                                baseline_type=baseline_type,
                                current_type=current_type,
                            )
                        )

            # Format edge lists as strings
            added_edges = [f"{e[0]}->{e[1]}" for e in added_edges_set]
            removed_edges = [f"{e[0]}->{e[1]}" for e in removed_edges_set]
            stable_edges = [f"{e[0]}->{e[1]}" for e in stable_edges_set]

            # Determine severity
            severity = self._determine_severity(drift_score, edge_type_changes)

            # Generate recommendation
            recommendation = self._generate_recommendation(
                drift_score, severity, added_edges, removed_edges, edge_type_changes
            )

            # Determine if drift was detected
            detected = drift_score > LOW_THRESHOLD or len(edge_type_changes) > 0

            output = StructuralDriftOutput(
                success=True,
                detected=detected,
                drift_score=drift_score,
                severity=severity,
                added_edges=added_edges,
                removed_edges=removed_edges,
                stable_edges=stable_edges,
                edge_type_changes=edge_type_changes,
                recommendation=recommendation,
                critical_path_broken=False,  # Placeholder for future enhancement
                timestamp=timestamp,
                trace_id=trace_id,
                errors=errors,
            )

        except Exception as e:
            logger.error(f"Structural drift detection failed: {e}")
            output = StructuralDriftOutput(
                success=False,
                detected=False,
                drift_score=0.0,
                severity="none",
                added_edges=[],
                removed_edges=[],
                stable_edges=[],
                edge_type_changes=[],
                recommendation=None,
                critical_path_broken=False,
                timestamp=timestamp,
                trace_id=trace_id,
                errors=errors + [str(e)],
            )

        # End trace
        if self.opik_enabled and trace_id:
            self._end_trace(trace_id, output)

        return output

    def _start_trace(self, params: StructuralDriftInput) -> Optional[str]:
        """Start an Opik trace."""
        try:
            import uuid

            import opik

            trace_id = str(uuid.uuid4())
            opik.track(
                name="detect_structural_drift",
                input={
                    "n_nodes": len(params.dag_nodes),
                    "has_edge_types": params.baseline_edge_types is not None,
                },
                metadata={"trace_id": trace_id},
            )
            return trace_id
        except Exception as e:
            logger.debug(f"Opik tracing not available: {e}")
            return None

    def _end_trace(self, trace_id: str, output: StructuralDriftOutput) -> None:
        """End an Opik trace."""
        try:
            import opik

            opik.track(
                name="detect_structural_drift.complete",
                output={
                    "success": output.success,
                    "detected": output.detected,
                    "drift_score": output.drift_score,
                    "severity": output.severity,
                    "added_edges_count": len(output.added_edges),
                    "removed_edges_count": len(output.removed_edges),
                },
                metadata={
                    "trace_id": trace_id,
                    "errors_count": len(output.errors),
                },
            )
        except Exception:
            pass


# =============================================================================
# SINGLETON AND REGISTRATION
# =============================================================================

_drift_tool_instance: Optional[StructuralDriftTool] = None


def get_structural_drift_tool() -> StructuralDriftTool:
    """Get or create the singleton StructuralDriftTool instance."""
    global _drift_tool_instance
    if _drift_tool_instance is None:
        _drift_tool_instance = StructuralDriftTool()
    return _drift_tool_instance


async def detect_structural_drift(
    baseline_dag_adjacency: List[List[int]],
    current_dag_adjacency: List[List[int]],
    dag_nodes: List[str],
    baseline_edge_types: Optional[Dict[str, str]] = None,
    current_edge_types: Optional[Dict[str, str]] = None,
    trace_context: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Detect structural drift between baseline and current DAGs.

    This is the registered tool function that wraps StructuralDriftTool.

    Args:
        baseline_dag_adjacency: Baseline DAG as binary adjacency matrix
        current_dag_adjacency: Current DAG as binary adjacency matrix
        dag_nodes: Node names in adjacency matrix order
        baseline_edge_types: Optional edge types for baseline (e.g., {'A->B': 'DIRECTED'})
        current_edge_types: Optional edge types for current
        trace_context: Opik trace context

    Returns:
        Dictionary with drift analysis results
    """
    tool = get_structural_drift_tool()

    result = await tool.invoke(
        StructuralDriftInput(
            baseline_dag_adjacency=baseline_dag_adjacency,
            current_dag_adjacency=current_dag_adjacency,
            dag_nodes=dag_nodes,
            baseline_edge_types=baseline_edge_types,
            current_edge_types=current_edge_types,
            trace_context=trace_context,
        )
    )

    return result.model_dump()


# =============================================================================
# TOOL REGISTRATION
# =============================================================================


def register_structural_drift_tool() -> None:
    """Register the detect_structural_drift tool in the global registry."""
    schema = ToolSchema(
        name="detect_structural_drift",
        description=(
            "Detect drift in causal DAG structure over time by comparing baseline "
            "and current period DAGs. Returns drift score, severity, added/removed "
            "edges, and recommendations for addressing structural changes."
        ),
        source_agent="drift_monitor",
        tier=3,
        input_parameters=[
            ToolParameter(
                name="baseline_dag_adjacency",
                type="List[List[int]]",
                description="Baseline DAG as binary adjacency matrix",
                required=True,
            ),
            ToolParameter(
                name="current_dag_adjacency",
                type="List[List[int]]",
                description="Current DAG as binary adjacency matrix",
                required=True,
            ),
            ToolParameter(
                name="dag_nodes",
                type="List[str]",
                description="Node names in adjacency matrix order",
                required=True,
            ),
            ToolParameter(
                name="baseline_edge_types",
                type="Dict[str, str]",
                description="Edge types for baseline DAG (optional)",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="current_edge_types",
                type="Dict[str, str]",
                description="Edge types for current DAG (optional)",
                required=False,
                default=None,
            ),
        ],
        output_schema="StructuralDriftOutput",
        avg_execution_ms=2000,
        is_async=True,
        supports_batch=False,
    )

    registry = get_registry()
    registry.register(
        schema=schema,
        callable=detect_structural_drift,
        input_model=StructuralDriftInput,
        output_model=StructuralDriftOutput,
    )

    logger.info("Registered detect_structural_drift tool in ToolRegistry")


# Auto-register on import (can be disabled if needed)
try:
    register_structural_drift_tool()
except Exception as e:
    logger.debug(f"Deferred tool registration: {e}")
