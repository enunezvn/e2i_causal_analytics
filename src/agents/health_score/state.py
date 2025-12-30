"""
E2I Health Score Agent - State Definitions
Version: 4.2
Purpose: LangGraph state definitions for health monitoring
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
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

    # === INPUT ===
    query: str
    check_scope: Literal["full", "quick", "models", "pipelines", "agents"]

    # === COMPONENT HEALTH ===
    component_statuses: Optional[List[ComponentStatus]]
    component_health_score: Optional[float]

    # === MODEL HEALTH ===
    model_metrics: Optional[List[ModelMetrics]]
    model_health_score: Optional[float]

    # === PIPELINE HEALTH ===
    pipeline_statuses: Optional[List[PipelineStatus]]
    pipeline_health_score: Optional[float]

    # === AGENT HEALTH ===
    agent_statuses: Optional[List[AgentStatus]]
    agent_health_score: Optional[float]

    # === COMPOSITE SCORE ===
    overall_health_score: Optional[float]  # 0-100
    health_grade: Optional[Literal["A", "B", "C", "D", "F"]]

    # === ISSUES ===
    critical_issues: Optional[List[str]]
    warnings: Optional[List[str]]

    # === SUMMARY ===
    health_summary: Optional[str]

    # === EXECUTION METADATA ===
    check_latency_ms: int
    timestamp: str

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    status: Literal["pending", "checking", "completed", "failed"]

    # === AUDIT CHAIN ===
    audit_workflow_id: Optional[UUID]
