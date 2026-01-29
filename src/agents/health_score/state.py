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
    """Model performance metrics"""

    model_id: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    auc_roc: Optional[float]
    prediction_latency_p50_ms: Optional[int]
    prediction_latency_p99_ms: Optional[int]
    predictions_last_24h: int
    error_rate: float
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
    """Agent availability status"""

    agent_name: str
    tier: int
    available: bool
    avg_latency_ms: int
    success_rate: float
    last_invocation: str


class HealthScoreState(TypedDict):
    """Complete state for Health Score agent"""

    # === INPUT (NotRequired - provided by caller) ===
    query: NotRequired[str]
    check_scope: NotRequired[Literal["full", "quick", "models", "pipelines", "agents"]]

    # === COMPONENT HEALTH ===
    component_statuses: NotRequired[List[ComponentStatus]]
    component_health_score: NotRequired[float]

    # === MODEL HEALTH ===
    model_metrics: NotRequired[List[ModelMetrics]]
    model_health_score: NotRequired[float]

    # === PIPELINE HEALTH ===
    pipeline_statuses: NotRequired[List[PipelineStatus]]
    pipeline_health_score: NotRequired[float]

    # === AGENT HEALTH ===
    agent_statuses: NotRequired[List[AgentStatus]]
    agent_health_score: NotRequired[float]

    # === COMPOSITE SCORE (Required outputs) ===
    overall_health_score: float  # 0-100
    health_grade: Literal["A", "B", "C", "D", "F"]

    # === ISSUES ===
    critical_issues: NotRequired[List[str]]
    warnings: NotRequired[List[str]]

    # === SUMMARY (Required output) ===
    health_summary: str

    # === EXECUTION METADATA (NotRequired - populated during execution) ===
    check_latency_ms: NotRequired[int]
    timestamp: NotRequired[str]

    # === ERROR HANDLING (Required outputs) ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    status: Literal["pending", "checking", "completed", "failed"]

    # === AUDIT CHAIN ===
    audit_workflow_id: NotRequired[UUID]
