"""
E2I Health Score Agent - State Definitions
Version: 4.2
Purpose: LangGraph state definitions for health monitoring
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict
from uuid import UUID


class ComponentStatus(TypedDict):
    """Status of a system component"""

    component_name: str
    status: Literal["healthy", "degraded", "unhealthy", "unknown"]
    latency_ms: Optional[int]
    last_check: str
    error_message: Optional[str]


class ModelMetrics(TypedDict):
    """Model performance metrics.

    ``predictions_last_24h``/``error_rate`` are Optional: ``None`` means the metric
    is UNMEASURED (the ml_model_health_dashboard view sources status but not these
    sub-fields), NOT a fabricated 0. Consumers must guard None before comparing or
    formatting (the score composer does).
    """

    model_id: str
    # Human-readable registry name; None when the store doesn't provide one.
    # Consumers fall back to model_id (alerts previously printed bare UUIDs).
    model_name: Optional[str]
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    auc_roc: Optional[float]
    prediction_latency_p50_ms: Optional[int]
    prediction_latency_p99_ms: Optional[int]
    predictions_last_24h: Optional[int]
    error_rate: Optional[float]
    status: Literal["healthy", "degraded", "unhealthy"]


class PipelineStatus(TypedDict):
    """Data pipeline status"""

    pipeline_name: str
    last_run: str
    last_success: str
    rows_processed: int
    freshness_hours: float
    status: Literal["healthy", "stale", "failed"]


class AgentStatus(TypedDict):
    """Agent availability status.

    ``avg_latency_ms``/``success_rate`` are Optional: ``None`` means the agent is
    registered (availability measured) but has NO recent runtime telemetry, so the
    rate/latency are UNMEASURED — never fabricated to 1.0/0. The agent node treats
    a None success_rate as "available => not penalized" (matching the /agents
    endpoint which scores purely on availability), without inventing a measurement.
    """

    agent_name: str
    tier: int
    available: bool
    avg_latency_ms: Optional[int]
    success_rate: Optional[float]
    last_invocation: str


class HealthScoreState(TypedDict):
    """Complete state for Health Score agent"""

    # === INPUT (NotRequired - provided by caller) ===
    query: NotRequired[str]
    check_scope: NotRequired[Literal["full", "quick", "models", "pipelines", "agents"]]

    # === COMPONENT HEALTH ===
    component_statuses: NotRequired[List[ComponentStatus]]
    component_health_score: NotRequired[Optional[float]]
    # True only when a real health backend produced the score (F1 fail-closed).
    # Absent/False => the dimension is UNKNOWN, never fail-open "healthy".
    component_health_measured: NotRequired[bool]

    # === MODEL HEALTH ===
    model_metrics: NotRequired[List[ModelMetrics]]
    model_health_score: NotRequired[Optional[float]]
    model_health_measured: NotRequired[bool]

    # === PIPELINE HEALTH ===
    pipeline_statuses: NotRequired[List[PipelineStatus]]
    pipeline_health_score: NotRequired[Optional[float]]
    pipeline_health_measured: NotRequired[bool]

    # === AGENT HEALTH ===
    agent_statuses: NotRequired[List[AgentStatus]]
    agent_health_score: NotRequired[Optional[float]]
    agent_health_measured: NotRequired[bool]

    # === COMPOSITE SCORE (Required outputs) ===
    overall_health_score: float  # 0-100
    health_grade: Literal["A", "B", "C", "D", "F"]
    # Provenance of the composite score: "measured" (all 4 dims measured),
    # "partial" (1-3 measured), or "unknown" (0 measured). Surfaced so the
    # dashboard/API never presents an unmeasured score as a real measurement.
    data_provenance: NotRequired[Literal["measured", "partial", "unknown"]]

    # === ISSUES ===
    critical_issues: NotRequired[List[str]]
    warnings: NotRequired[List[str]]

    # === RECOMMENDATIONS ===
    recommendations: NotRequired[List[str]]

    # === DIAGNOSTIC REASONING (P2 enhancement) ===
    health_diagnosis: NotRequired[dict]  # Root causes, cascading effects, priority fixes

    # === SUMMARY (Required output) ===
    health_summary: str

    # === EXECUTION METADATA (Contract-required output fields) ===
    total_latency_ms: int  # Contract requires this name (was check_latency_ms)
    timestamp: str  # Contract requires this field

    # === ERROR HANDLING (Required outputs) ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    status: Literal["pending", "checking", "completed", "failed"]

    # === AUDIT CHAIN ===
    audit_workflow_id: NotRequired[UUID]
