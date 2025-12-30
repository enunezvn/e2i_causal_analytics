"""State definitions for Drift Monitor Agent.

This module defines the TypedDict structures for the drift_monitor agent's state management.
Follows contracts defined in .claude/contracts/tier3-contracts.md (lines 349-562).

Tier: 3 (Monitoring)
Agent Type: Standard (Fast Path)
Performance Target: <10s for 50 features
"""

from typing import Literal, NotRequired, Optional, TypedDict
from uuid import UUID

# Type aliases
DriftType = Literal["data", "model", "concept"]
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


class ErrorDetails(TypedDict):
    """Error information."""

    node: str
    error: str
    timestamp: str


class DriftMonitorState(TypedDict):
    """Complete state for Drift Monitor Agent.

    Contract: .claude/contracts/tier3-contracts.md lines 447-494
    Total Fields: 23

    Field Groups:
    - Input (5): query, model_id, features_to_monitor, time_window, brand
    - Configuration (5): significance_level, psi_threshold, check_data_drift,
                         check_model_drift, check_concept_drift
    - Detection outputs (3): data_drift_results, model_drift_results, concept_drift_results
    - Aggregated outputs (3): overall_drift_score, features_with_drift, alerts
    - Summary (2): drift_summary, recommended_actions
    - Execution metadata (4): detection_latency_ms, features_checked,
                              baseline_timestamp, current_timestamp
    - Error handling (3): errors, warnings, status
    """

    # ===== Input Fields (5) =====
    query: str
    model_id: NotRequired[Optional[str]]
    features_to_monitor: list[str]
    time_window: str
    brand: NotRequired[Optional[str]]

    # ===== Configuration (5) =====
    significance_level: float
    psi_threshold: float
    check_data_drift: bool
    check_model_drift: bool
    check_concept_drift: bool

    # ===== Detection Outputs (3) =====
    data_drift_results: NotRequired[list[DriftResult]]
    model_drift_results: NotRequired[list[DriftResult]]
    concept_drift_results: NotRequired[list[DriftResult]]

    # ===== Aggregated Outputs (3) =====
    overall_drift_score: NotRequired[float]
    features_with_drift: NotRequired[list[str]]
    alerts: NotRequired[list[DriftAlert]]

    # ===== Summary (2) =====
    drift_summary: NotRequired[str]
    recommended_actions: NotRequired[list[str]]

    # ===== Execution Metadata (4) =====
    detection_latency_ms: NotRequired[int]
    features_checked: NotRequired[int]
    baseline_timestamp: NotRequired[str]
    current_timestamp: NotRequired[str]

    # ===== Error Handling (3) =====
    errors: list[ErrorDetails]
    warnings: list[str]
    status: AgentStatus

    # ===== Audit Chain (1) =====
    audit_workflow_id: NotRequired[Optional[UUID]]
