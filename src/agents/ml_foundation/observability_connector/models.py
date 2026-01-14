"""
Pydantic models for observability connector.

Defines data models for:
- Observability spans (matching ml_observability_spans table)
- Span events for detailed tracking
- Latency statistics for percentile aggregation
- Quality metrics for system health
- Token usage for LLM cost tracking

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================


class AgentNameEnum(str, Enum):
    """All agent names in the E2I system."""

    # Tier 0: ML Foundation
    SCOPE_DEFINER = "scope_definer"
    DATA_PREPARER = "data_preparer"
    FEATURE_ANALYZER = "feature_analyzer"
    MODEL_SELECTOR = "model_selector"
    MODEL_TRAINER = "model_trainer"
    MODEL_DEPLOYER = "model_deployer"
    OBSERVABILITY_CONNECTOR = "observability_connector"

    # Tier 1: Coordination
    ORCHESTRATOR = "orchestrator"
    TOOL_COMPOSER = "tool_composer"

    # Tier 2: Causal Analytics
    CAUSAL_IMPACT = "causal_impact"
    GAP_ANALYZER = "gap_analyzer"
    HETEROGENEOUS_OPTIMIZER = "heterogeneous_optimizer"

    # Tier 3: Monitoring & Experimentation
    DRIFT_MONITOR = "drift_monitor"
    EXPERIMENT_DESIGNER = "experiment_designer"
    HEALTH_SCORE = "health_score"

    # Tier 4: ML & Predictions
    PREDICTION_SYNTHESIZER = "prediction_synthesizer"
    RESOURCE_OPTIMIZER = "resource_optimizer"

    # Tier 5: Self-Improvement
    EXPLAINER = "explainer"
    FEEDBACK_LEARNER = "feedback_learner"


class AgentTierEnum(str, Enum):
    """Agent tier classifications."""

    ML_FOUNDATION = "ml_foundation"  # Tier 0
    COORDINATION = "coordination"  # Tier 1
    CAUSAL_ANALYTICS = "causal_analytics"  # Tier 2
    MONITORING = "monitoring"  # Tier 3
    ML_PREDICTIONS = "ml_predictions"  # Tier 4
    SELF_IMPROVEMENT = "self_improvement"  # Tier 5


class SpanStatusEnum(str, Enum):
    """Span execution status."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


# =============================================================================
# SPAN EVENT MODEL
# =============================================================================


class SpanEvent(BaseModel):
    """Event that occurred during a span execution.

    Events are immutable markers within a span's timeline,
    such as checkpoints, warnings, or significant state changes.
    """

    name: str = Field(..., description="Event name (e.g., 'checkpoint', 'retry')")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When the event occurred"
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Event-specific attributes"
    )

    model_config = ConfigDict(frozen=True)  # Events are immutable


# =============================================================================
# TOKEN USAGE MODEL
# =============================================================================


class TokenUsage(BaseModel):
    """LLM token usage tracking.

    Tracks input, output, and total tokens for cost attribution
    and usage monitoring.
    """

    input_tokens: int = Field(default=0, ge=0, description="Input/prompt tokens")
    output_tokens: int = Field(default=0, ge=0, description="Output/completion tokens")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens used")

    @field_validator("total_tokens", mode="before")
    @classmethod
    def compute_total_if_zero(cls, v: int, info) -> int:
        """Auto-compute total if not provided."""
        if v == 0 and info.data:
            input_tokens = info.data.get("input_tokens", 0)
            output_tokens = info.data.get("output_tokens", 0)
            return input_tokens + output_tokens
        return v


# =============================================================================
# OBSERVABILITY SPAN MODEL
# =============================================================================


class ObservabilitySpan(BaseModel):
    """Observability span matching ml_observability_spans table.

    Represents a single unit of work tracked in the observability system.
    Can be an agent operation, LLM call, or any traceable action.
    """

    # Primary key (auto-generated if not provided)
    id: UUID = Field(default_factory=uuid4, description="Unique span identifier")

    # Span identification (required for W3C Trace Context)
    trace_id: str = Field(..., min_length=1, max_length=100, description="Trace ID")
    span_id: str = Field(..., min_length=1, max_length=100, description="Span ID")
    parent_span_id: Optional[str] = Field(
        default=None, max_length=100, description="Parent span ID for nested spans"
    )

    # Context
    agent_name: AgentNameEnum = Field(..., description="Name of the agent")
    agent_tier: AgentTierEnum = Field(..., description="Tier of the agent")
    operation_type: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Operation type (inference, training, shap_computation, etc.)",
    )

    # Timing
    started_at: datetime = Field(..., description="Span start time")
    ended_at: Optional[datetime] = Field(default=None, description="Span end time")
    duration_ms: Optional[int] = Field(
        default=None, ge=0, description="Duration in milliseconds"
    )

    # LLM specific (for Hybrid/Deep agents)
    model_name: Optional[str] = Field(
        default=None, max_length=100, description="LLM model name"
    )
    input_tokens: Optional[int] = Field(
        default=None, ge=0, description="Input tokens used"
    )
    output_tokens: Optional[int] = Field(
        default=None, ge=0, description="Output tokens generated"
    )
    total_tokens: Optional[int] = Field(
        default=None, ge=0, description="Total tokens consumed"
    )

    # Status
    status: SpanStatusEnum = Field(
        default=SpanStatusEnum.SUCCESS, description="Execution status"
    )
    error_type: Optional[str] = Field(
        default=None, max_length=100, description="Error type if failed"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )

    # Fallback tracking
    fallback_used: bool = Field(
        default=False, description="Whether fallback was triggered"
    )
    fallback_chain: List[str] = Field(
        default_factory=list, description="Sequence of models tried"
    )

    # Custom attributes
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Additional span attributes"
    )

    # Events within the span
    events: List[SpanEvent] = Field(
        default_factory=list, description="Events during span execution"
    )

    # Links to other entities (optional foreign keys)
    experiment_id: Optional[UUID] = Field(
        default=None, description="Linked experiment ID"
    )
    training_run_id: Optional[UUID] = Field(
        default=None, description="Linked training run ID"
    )
    deployment_id: Optional[UUID] = Field(
        default=None, description="Linked deployment ID"
    )

    # User context
    user_id: Optional[str] = Field(
        default=None, max_length=100, description="User identifier"
    )
    session_id: Optional[str] = Field(
        default=None, max_length=100, description="Session identifier"
    )

    @field_validator("duration_ms", mode="before")
    @classmethod
    def compute_duration_if_needed(cls, v: Optional[int], info) -> Optional[int]:
        """Auto-compute duration if ended_at is provided."""
        if v is None and info.data:
            started_at = info.data.get("started_at")
            ended_at = info.data.get("ended_at")
            if started_at and ended_at:
                delta = ended_at - started_at
                return int(delta.total_seconds() * 1000)
        return v

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to this span.

        Args:
            name: Event name
            attributes: Optional event attributes
        """
        event = SpanEvent(name=name, attributes=attributes or {})
        self.events.append(event)

    def set_error(self, error_type: str, message: str) -> None:
        """Mark span as failed with error details.

        Args:
            error_type: Type of error (e.g., 'ValidationError')
            message: Error message
        """
        self.status = SpanStatusEnum.ERROR
        self.error_type = error_type
        self.error_message = message

    def complete(self, ended_at: Optional[datetime] = None) -> None:
        """Mark span as completed.

        Args:
            ended_at: End time (defaults to now)
        """
        self.ended_at = ended_at or datetime.now(timezone.utc)
        if self.started_at:
            delta = self.ended_at - self.started_at
            self.duration_ms = int(delta.total_seconds() * 1000)

    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion.

        Returns:
            Dictionary with database-compatible field names
        """
        data = self.model_dump(exclude={"events"})
        # Convert UUID to string for Supabase
        data["id"] = str(data["id"])
        if data.get("experiment_id"):
            data["experiment_id"] = str(data["experiment_id"])
        if data.get("training_run_id"):
            data["training_run_id"] = str(data["training_run_id"])
        if data.get("deployment_id"):
            data["deployment_id"] = str(data["deployment_id"])
        # Convert enums to values
        data["agent_name"] = data["agent_name"].value
        data["agent_tier"] = data["agent_tier"].value
        data["status"] = data["status"].value
        # Convert datetime objects to ISO format strings for JSON serialization
        if data.get("started_at") and isinstance(data["started_at"], datetime):
            data["started_at"] = data["started_at"].isoformat()
        if data.get("ended_at") and isinstance(data["ended_at"], datetime):
            data["ended_at"] = data["ended_at"].isoformat()
        return data

    model_config = ConfigDict(from_attributes=True)  # Allow ORM mode


# =============================================================================
# LATENCY STATS MODEL
# =============================================================================


class LatencyStats(BaseModel):
    """Latency statistics with percentiles.

    Used for aggregating latency metrics from v_agent_latency_summary view.
    """

    agent_name: Optional[AgentNameEnum] = Field(
        default=None, description="Agent name (None for tier-level stats)"
    )
    agent_tier: Optional[AgentTierEnum] = Field(
        default=None, description="Agent tier"
    )
    total_spans: int = Field(default=0, ge=0, description="Total spans analyzed")
    avg_duration_ms: float = Field(default=0.0, ge=0, description="Average latency")
    p50_ms: float = Field(default=0.0, ge=0, description="Median latency (50th)")
    p95_ms: float = Field(default=0.0, ge=0, description="95th percentile latency")
    p99_ms: float = Field(default=0.0, ge=0, description="99th percentile latency")
    error_rate: float = Field(
        default=0.0, ge=0, le=1, description="Error rate (0.0-1.0)"
    )
    fallback_rate: float = Field(
        default=0.0, ge=0, le=1, description="Fallback invocation rate (0.0-1.0)"
    )
    total_tokens_used: int = Field(
        default=0, ge=0, description="Total tokens consumed"
    )

    @property
    def is_healthy(self) -> bool:
        """Check if latency stats indicate healthy operation.

        Returns:
            True if p95 < 5000ms and error_rate < 0.05
        """
        return self.p95_ms < 5000 and self.error_rate < 0.05


# =============================================================================
# QUALITY METRICS MODEL
# =============================================================================


class QualityMetrics(BaseModel):
    """Aggregated quality metrics for system health.

    Provides a holistic view of observability system performance.
    """

    # Time window
    time_window: str = Field(
        default="24h", description="Time window for metrics (1h, 24h, 7d)"
    )
    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When metrics were computed"
    )

    # Span counts
    total_spans: int = Field(default=0, ge=0, description="Total spans analyzed")
    success_count: int = Field(default=0, ge=0, description="Successful spans")
    error_count: int = Field(default=0, ge=0, description="Failed spans")
    timeout_count: int = Field(default=0, ge=0, description="Timed out spans")

    # Rates
    success_rate: float = Field(
        default=1.0, ge=0, le=1, description="Success rate (0.0-1.0)"
    )
    error_rate: float = Field(
        default=0.0, ge=0, le=1, description="Error rate (0.0-1.0)"
    )
    fallback_rate: float = Field(
        default=0.0, ge=0, le=1, description="Fallback usage rate (0.0-1.0)"
    )

    # Overall latency
    avg_latency_ms: float = Field(default=0.0, ge=0, description="Average latency")
    p50_latency_ms: float = Field(default=0.0, ge=0, description="Median latency")
    p95_latency_ms: float = Field(default=0.0, ge=0, description="95th percentile")
    p99_latency_ms: float = Field(default=0.0, ge=0, description="99th percentile")

    # Token usage
    total_tokens: int = Field(default=0, ge=0, description="Total tokens used")
    avg_tokens_per_span: float = Field(
        default=0.0, ge=0, description="Average tokens per span"
    )

    # By-agent breakdown
    latency_by_agent: Dict[str, LatencyStats] = Field(
        default_factory=dict, description="Latency stats per agent"
    )
    latency_by_tier: Dict[str, LatencyStats] = Field(
        default_factory=dict, description="Latency stats per tier"
    )

    # Status distribution
    status_distribution: Dict[str, int] = Field(
        default_factory=lambda: {"success": 0, "error": 0, "timeout": 0},
        description="Count by status",
    )

    # Derived quality score (0.0-1.0)
    quality_score: float = Field(
        default=1.0, ge=0, le=1, description="Overall quality score"
    )

    def compute_quality_score(self) -> float:
        """Compute overall quality score from metrics.

        Formula weights:
        - Success rate: 40%
        - Latency health (p95 < 5000ms): 30%
        - Low fallback rate: 20%
        - Low error rate: 10%

        Returns:
            Quality score between 0.0 and 1.0
        """
        # Success rate component (40%)
        success_component = self.success_rate * 0.4

        # Latency health component (30%)
        # Score decreases as p95 approaches 10000ms
        if self.p95_latency_ms <= 1000:
            latency_score = 1.0
        elif self.p95_latency_ms <= 5000:
            latency_score = 0.8
        elif self.p95_latency_ms <= 10000:
            latency_score = 0.5
        else:
            latency_score = 0.2
        latency_component = latency_score * 0.3

        # Low fallback rate component (20%)
        fallback_component = (1.0 - self.fallback_rate) * 0.2

        # Low error rate component (10%)
        error_component = (1.0 - self.error_rate) * 0.1

        self.quality_score = (
            success_component + latency_component + fallback_component + error_component
        )
        return self.quality_score

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_span(
    trace_id: str,
    span_id: str,
    agent_name: AgentNameEnum,
    agent_tier: AgentTierEnum,
    operation_type: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    **kwargs: Any,
) -> ObservabilitySpan:
    """Factory function to create an ObservabilitySpan.

    Args:
        trace_id: Trace identifier
        span_id: Span identifier
        agent_name: Name of the agent
        agent_tier: Tier of the agent
        operation_type: Type of operation
        parent_span_id: Parent span for nested spans
        **kwargs: Additional span attributes

    Returns:
        Configured ObservabilitySpan instance
    """
    return ObservabilitySpan(
        trace_id=trace_id,
        span_id=span_id,
        agent_name=agent_name,
        agent_tier=agent_tier,
        operation_type=operation_type,
        parent_span_id=parent_span_id,
        started_at=kwargs.pop("started_at", datetime.now(timezone.utc)),
        **kwargs,
    )


def create_llm_span(
    trace_id: str,
    span_id: str,
    agent_name: AgentNameEnum,
    agent_tier: AgentTierEnum,
    model_name: str,
    parent_span_id: Optional[str] = None,
    **kwargs: Any,
) -> ObservabilitySpan:
    """Factory function to create an LLM-specific span.

    Args:
        trace_id: Trace identifier
        span_id: Span identifier
        agent_name: Name of the agent
        agent_tier: Tier of the agent
        model_name: LLM model name
        parent_span_id: Parent span for nested spans
        **kwargs: Additional span attributes

    Returns:
        Configured ObservabilitySpan for LLM operations
    """
    return ObservabilitySpan(
        trace_id=trace_id,
        span_id=span_id,
        agent_name=agent_name,
        agent_tier=agent_tier,
        operation_type="llm_call",
        model_name=model_name,
        parent_span_id=parent_span_id,
        started_at=kwargs.pop("started_at", datetime.now(timezone.utc)),
        **kwargs,
    )
