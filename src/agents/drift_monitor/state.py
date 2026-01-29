"""State definitions for Drift Monitor Agent.

This module defines the TypedDict structures for the drift_monitor agent's state management.
Follows contracts defined in .claude/contracts/tier3-contracts.md (lines 349-562).

Tier: 3 (Monitoring)
Agent Type: Standard (Fast Path)
Performance Target: <10s for 50 features

V4.4: Added structural drift detection for causal DAG changes.
"""

from typing import Any, Dict, List, Literal, NotRequired, Optional, TypedDict
from uuid import UUID

# Type aliases
DriftType = Literal["data", "model", "concept", "structural"]  # V4.4: Added structural
DriftSeverity = Literal["none", "low", "medium", "high", "critical"]
AlertSeverity = Literal["warning", "critical"]
AgentStatus = Literal["pending", "detecting", "aggregating", "completed", "failed"]


class DriftResult(TypedDict):
    """Individual drift detection result for a feature.

    Contract: .claude/contracts/tier3-contracts.md lines 401-410
    """

    feature: str
    drift_type: DriftType
    test_statistic: float
    p_value: float
    drift_detected: bool
    severity: DriftSeverity
    baseline_period: str
    current_period: str


class DriftAlert(TypedDict):
    """Drift alert with recommended action.

    Contract: .claude/contracts/tier3-contracts.md lines 412-420
    """

    alert_id: str
    severity: AlertSeverity
    drift_type: DriftType
    affected_features: list[str]
    message: str
    recommended_action: str
    timestamp: str


class StructuralDriftResult(TypedDict):
    """V4.4: Structural drift detection result.

    Tracks changes in the causal DAG structure over time.
    """

    detected: bool
    drift_score: float  # Percentage of edges changed (0.0 to 1.0)
    added_edges: List[str]  # Format: "source->target"
    removed_edges: List[str]  # Format: "source->target"
    stable_edges: NotRequired[List[str]]  # Unchanged edges
    edge_type_changes: NotRequired[List[Dict[str, Any]]]  # Edge type drift
    severity: str  # "none", "low", "medium", "high", "critical"
    recommendation: Optional[str]  # Action recommendation
    critical_path_broken: NotRequired[bool]  # If treatment->outcome path broken


class ErrorDetails(TypedDict):
    """Error information."""

    node: str
    error: str
    timestamp: str


class DriftMonitorState(TypedDict):
    """Complete state for Drift Monitor Agent.

    Contract: .claude/contracts/tier3-contracts.md lines 447-494
    Total Fields: 31 (V4.4: 8 new fields)

    Field Groups:
    - Input (5): query, model_id, features_to_monitor, time_window, brand
    - Configuration (6): significance_level, psi_threshold, check_data_drift,
                         check_model_drift, check_concept_drift, check_structural_drift
    - Detection outputs (5): data_drift_results, model_drift_results, concept_drift_results,
                            structural_drift_results, structural_drift_details
    - Aggregated outputs (3): overall_drift_score, features_with_drift, alerts
    - Summary (2): drift_summary, recommended_actions
    - Execution metadata (4): detection_latency_ms, features_checked,
                              baseline_timestamp, current_timestamp
    - Error handling (3): errors, warnings, status
    - Audit Chain (1): audit_workflow_id
    - V4.4 Discovery (7): baseline_dag_adjacency, baseline_dag_edge_types,
                          current_dag_adjacency, current_dag_edge_types,
                          dag_nodes, discovery_config
    """

    # ===== Input Fields (NotRequired - provided by caller) =====
    query: NotRequired[str]
    model_id: NotRequired[str]
    features_to_monitor: NotRequired[list[str]]
    time_window: NotRequired[str]
    brand: NotRequired[str]
    # Tier0 data passthrough for testing with real synthetic data
    tier0_data: NotRequired[Any]  # pandas DataFrame

    # ===== Configuration (NotRequired - has defaults) =====
    significance_level: NotRequired[float]  # Default: 0.05
    psi_threshold: NotRequired[float]  # Default: 0.1
    check_data_drift: NotRequired[bool]  # Default: True
    check_model_drift: NotRequired[bool]  # Default: True
    check_concept_drift: NotRequired[bool]  # Default: True
    check_structural_drift: NotRequired[bool]  # V4.4: Enable structural drift detection

    # ===== Detection Outputs (Required) =====
    data_drift_results: list[DriftResult]
    model_drift_results: NotRequired[list[DriftResult]]
    concept_drift_results: NotRequired[list[DriftResult]]
    structural_drift_results: NotRequired[list[DriftResult]]  # V4.4: DAG structure drift
    structural_drift_details: NotRequired[StructuralDriftResult]  # V4.4: Detailed drift info

    # ===== Aggregated Outputs (Required) =====
    overall_drift_score: float
    features_with_drift: list[str]
    alerts: NotRequired[list[DriftAlert]]

    # ===== Summary (Required output) =====
    drift_summary: str
    recommended_actions: NotRequired[list[str]]

    # ===== Execution Metadata (NotRequired - populated during execution) =====
    detection_latency_ms: NotRequired[int]
    features_checked: NotRequired[int]
    baseline_timestamp: NotRequired[str]
    current_timestamp: NotRequired[str]

    # ===== Error Handling (Required outputs) =====
    errors: list[ErrorDetails]
    warnings: list[str]
    status: AgentStatus

    # ===== Audit Chain (1) =====
    audit_workflow_id: NotRequired[Optional[UUID]]

    # ========================================================================
    # V4.4: Causal Discovery Integration
    # ========================================================================

    # Baseline DAG (from historical discovery)
    baseline_dag_adjacency: NotRequired[List[List[int]]]  # Binary adjacency matrix
    baseline_dag_edge_types: NotRequired[Dict[str, str]]  # Edge types (DIRECTED, BIDIRECTED)

    # Current DAG (from recent discovery on new data)
    current_dag_adjacency: NotRequired[List[List[int]]]  # Binary adjacency matrix
    current_dag_edge_types: NotRequired[Dict[str, str]]  # Edge types

    # Node names (shared between baseline and current)
    dag_nodes: NotRequired[List[str]]  # Variable names

    # Discovery configuration for re-running on current data
    discovery_config: NotRequired[Dict[str, Any]]
