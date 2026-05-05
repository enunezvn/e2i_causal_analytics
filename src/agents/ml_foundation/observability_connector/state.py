"""State definition for observability_connector agent.

Migrated from ``TypedDict(total=False)`` to pydantic v2 ``BaseModel``
in Shard C of the migration. Inherits from ``BaseAgentSchema``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import Field

from src.agents.ml_foundation._pydantic_utils import (
    BaseAgentSchema,
    audit_workflow_id_validator,
)


class ObservabilityConnectorState(BaseAgentSchema):
    """State for observability_connector LangGraph workflow.

    This agent operates differently from others - it's primarily used via
    helper methods (span(), track_llm_call()) rather than being invoked
    in the main pipeline. The graph workflow is for collecting metrics.
    """

    # ========================================================================
    # INPUT FIELDS (From ObservabilityEvent contract - for logging)
    # ========================================================================

    # Events to log (batch operation)
    events_to_log: Optional[List[Dict[str, Any]]] = None

    # Time window for metrics collection
    time_window: Optional[str] = None  # "1h", "24h", "7d"

    # Filter criteria for metrics
    agent_name_filter: Optional[str] = None
    trace_id_filter: Optional[str] = None

    # ========================================================================
    # NODE 1 OUTPUT: Span Emission
    # ========================================================================

    # Logging results
    span_ids_logged: Optional[List[str]] = None
    trace_ids_logged: Optional[List[str]] = None
    events_logged: Optional[int] = None
    emission_successful: Optional[bool] = None
    emission_errors: Optional[List[str]] = None

    # Opik metadata
    opik_project: Optional[str] = None
    opik_workspace: Optional[str] = None
    opik_url: Optional[str] = None

    # Database persistence
    db_writes_successful: Optional[bool] = None
    db_write_count: Optional[int] = None

    # ========================================================================
    # NODE 2 OUTPUT: Metrics Aggregation
    # ========================================================================

    # Quality metrics computed
    quality_metrics_computed: Optional[bool] = None

    # Latency metrics (by agent) — Dict[str, Any] preemptively widened
    # to tolerate mixed-type inner dicts at runtime per the Shard B
    # lesson (PR #50 widened model_trainer metric bags after pydantic
    # rejected str + nested-dict values inside declared Dict[str, float]).
    latency_by_agent: Optional[Dict[str, Any]] = None
    # {"scope_definer": {"p50": 2.1, "p95": 4.5, "p99": 8.2, "avg": 3.2}, ...}

    # Latency metrics (by tier)
    latency_by_tier: Optional[Dict[int, Any]] = None
    # {0: {"p50": 5.0, "p95": 12.0, "p99": 20.0, "avg": 7.5}, ...}

    # Error rates (by agent)
    error_rate_by_agent: Optional[Dict[str, float]] = None
    # {"scope_definer": 0.02, "data_preparer": 0.05, ...}

    # Error rates (by tier)
    error_rate_by_tier: Optional[Dict[int, float]] = None

    # Token usage (by agent) - for Hybrid/Deep agents
    token_usage_by_agent: Optional[Dict[str, Any]] = None
    # {"feature_analyzer": {"input": 50000, "output": 12000, "total": 62000}, ...}

    # Overall system metrics
    overall_success_rate: Optional[float] = None  # 1 - (error_count / total_count)
    overall_p95_latency_ms: Optional[float] = None
    overall_p99_latency_ms: Optional[float] = None
    total_spans_analyzed: Optional[int] = None

    # Quality score (derived)
    quality_score: Optional[float] = None  # 0.0-1.0

    # Fallback invocation rate
    fallback_invocation_rate: Optional[float] = None

    # Span count by status
    status_distribution: Optional[Dict[str, int]] = None
    # {"ok": 950, "error": 30, "timeout": 20}

    # ========================================================================
    # CONTEXT MANAGEMENT STATE (helper methods)
    # ========================================================================

    # Current trace context
    current_trace_id: Optional[str] = None
    current_span_id: Optional[str] = None
    current_parent_span_id: Optional[str] = None

    # Request metadata
    request_id: Optional[str] = None
    experiment_id: Optional[str] = None
    user_id: Optional[str] = None

    # Sampling
    sampled: Optional[bool] = None
    sample_rate: Optional[float] = None

    # ========================================================================
    # ERROR HANDLING
    # ========================================================================

    error: Optional[str] = None
    error_type: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # ========================================================================
    # AUDIT CHAIN
    # ========================================================================
    audit_workflow_id: UUID = Field(default_factory=uuid4)

    _validate_audit_id = audit_workflow_id_validator()
