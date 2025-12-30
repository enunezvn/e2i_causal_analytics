"""State definition for observability_connector agent."""

from typing import Any, Dict, List, Optional, TypedDict
from uuid import UUID


class ObservabilityConnectorState(TypedDict, total=False):
    """State for observability_connector LangGraph workflow.

    This agent operates differently from others - it's primarily used via
    helper methods (span(), track_llm_call()) rather than being invoked
    in the main pipeline. The graph workflow is for collecting metrics.
    """

    # ========================================================================
    # INPUT FIELDS (From ObservabilityEvent contract - for logging)
    # ========================================================================

    # Events to log (batch operation)
    events_to_log: List[Dict[str, Any]]

    # Time window for metrics collection
    time_window: str  # "1h", "24h", "7d"

    # Filter criteria for metrics
    agent_name_filter: Optional[str]
    trace_id_filter: Optional[str]

    # ========================================================================
    # NODE 1 OUTPUT: Span Emission
    # ========================================================================

    # Logging results
    span_ids_logged: List[str]
    trace_ids_logged: List[str]
    events_logged: int
    emission_successful: bool
    emission_errors: List[str]

    # Opik metadata
    opik_project: str
    opik_workspace: str
    opik_url: Optional[str]

    # Database persistence
    db_writes_successful: bool
    db_write_count: int

    # ========================================================================
    # NODE 2 OUTPUT: Metrics Aggregation
    # ========================================================================

    # Quality metrics computed
    quality_metrics_computed: bool

    # Latency metrics (by agent)
    latency_by_agent: Dict[str, Dict[str, float]]
    # {"scope_definer": {"p50": 2.1, "p95": 4.5, "p99": 8.2, "avg": 3.2}, ...}

    # Latency metrics (by tier)
    latency_by_tier: Dict[int, Dict[str, float]]
    # {0: {"p50": 5.0, "p95": 12.0, "p99": 20.0, "avg": 7.5}, ...}

    # Error rates (by agent)
    error_rate_by_agent: Dict[str, float]
    # {"scope_definer": 0.02, "data_preparer": 0.05, ...}

    # Error rates (by tier)
    error_rate_by_tier: Dict[int, float]
    # {0: 0.03, 1: 0.01, 2: 0.02, ...}

    # Token usage (by agent) - for Hybrid/Deep agents
    token_usage_by_agent: Dict[str, Dict[str, int]]
    # {"feature_analyzer": {"input": 50000, "output": 12000, "total": 62000}, ...}

    # Overall system metrics
    overall_success_rate: float  # 1 - (error_count / total_count)
    overall_p95_latency_ms: float
    overall_p99_latency_ms: float
    total_spans_analyzed: int

    # Quality score (derived)
    quality_score: Optional[float]  # 0.0-1.0

    # Fallback invocation rate
    fallback_invocation_rate: float

    # Span count by status
    status_distribution: Dict[str, int]
    # {"ok": 950, "error": 30, "timeout": 20}

    # ========================================================================
    # CONTEXT MANAGEMENT STATE (helper methods)
    # ========================================================================

    # Current trace context
    current_trace_id: Optional[str]
    current_span_id: Optional[str]
    current_parent_span_id: Optional[str]

    # Request metadata
    request_id: Optional[str]
    experiment_id: Optional[str]
    user_id: Optional[str]

    # Sampling
    sampled: bool
    sample_rate: float

    # ========================================================================
    # ERROR HANDLING
    # ========================================================================

    error: Optional[str]
    error_type: Optional[str]
    error_details: Optional[Dict[str, Any]]

    # ========================================================================
    # AUDIT CHAIN
    # ========================================================================
    audit_workflow_id: UUID
