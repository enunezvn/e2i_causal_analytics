"""
E2I Health Score Agent - Opik Distributed Tracing
Version: 4.2
Purpose: Distributed tracing for Health Score agent operations

This module provides Opik integration for the Health Score agent,
enabling distributed tracing across health check operations.

Architecture:
    - HealthScoreOpikTracer: Main tracer class with singleton pattern
    - HealthCheckTraceContext: Async context manager for traces
    - NodeSpanContext: Context for individual node spans

Health Dimensions Traced:
    - component: Database, cache, vector store, API, message queue
    - model: Accuracy, latency, error rates, prediction volume
    - pipeline: Data freshness, processing success, row counts
    - agent: Availability, success rates, latency
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from opik import Opik, Trace
    from opik.api_objects.span import Span

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

HEALTH_CHECK_TYPES = ["full", "quick", "models", "pipelines", "agents"]
HEALTH_NODES = ["component", "model", "pipeline", "agent", "compose"]

# Health score thresholds
GRADE_THRESHOLDS = {
    "A": 90,
    "B": 80,
    "C": 70,
    "D": 60,
    "F": 0,
}


# ============================================================================
# SPAN CONTEXT
# ============================================================================


@dataclass
class NodeSpanContext:
    """Context for individual health check node spans."""

    span: Optional["Span"]
    node_name: str
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the span."""
        self.metadata[key] = value

    def end(self, status: str = "completed") -> None:
        """End the span with final metadata."""
        if self.span:
            try:
                elapsed_ms = int((time.time() - self.start_time) * 1000)
                self.span.end(
                    metadata={
                        **self.metadata,
                        "status": status,
                        "duration_ms": elapsed_ms,
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to end node span: {e}")


# ============================================================================
# TRACE CONTEXT
# ============================================================================


@dataclass
class HealthCheckTraceContext:
    """Async context manager for health check traces."""

    trace: Optional["Trace"]
    tracer: "HealthScoreOpikTracer"
    check_scope: str
    start_time: float = field(default_factory=time.time)
    trace_metadata: Dict[str, Any] = field(default_factory=dict)
    active_spans: Dict[str, NodeSpanContext] = field(default_factory=dict)

    def log_check_started(
        self,
        check_scope: str,
        query: Optional[str] = None,
    ) -> None:
        """Log that a health check has started."""
        if self.trace:
            try:
                self.trace_metadata.update({
                    "check_scope": check_scope,
                    "query": query or "",
                    "check_started_at": datetime.now(timezone.utc).isoformat(),
                })
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log check started: {e}")

    def start_node_span(
        self,
        node_name: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> NodeSpanContext:
        """Start a span for a health check node."""
        span = None
        if self.trace and self.tracer._client:
            try:
                span = self.trace.span(
                    name=f"health_score.{node_name}",
                    input=input_data or {},
                    metadata={
                        "node_name": node_name,
                        "check_scope": self.check_scope,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to create node span: {e}")

        ctx = NodeSpanContext(span=span, node_name=node_name)
        self.active_spans[node_name] = ctx
        return ctx

    def end_node_span(
        self,
        node_name: str,
        output_data: Optional[Dict[str, Any]] = None,
        status: str = "completed",
    ) -> None:
        """End a node span."""
        if node_name in self.active_spans:
            ctx = self.active_spans[node_name]
            if ctx.span:
                try:
                    elapsed_ms = int((time.time() - ctx.start_time) * 1000)
                    ctx.span.end(
                        output=output_data or {},
                        metadata={
                            **ctx.metadata,
                            "status": status,
                            "duration_ms": elapsed_ms,
                        },
                    )
                except Exception as e:
                    logger.debug(f"Failed to end node span {node_name}: {e}")
            del self.active_spans[node_name]

    def log_component_health(
        self,
        score: float,
        statuses: Dict[str, str],
        issues: List[str],
        duration_ms: int,
    ) -> None:
        """Log component health check results."""
        if self.trace:
            try:
                self.trace_metadata.update({
                    "component_health_score": score,
                    "component_statuses": statuses,
                    "component_issues": issues,
                    "component_duration_ms": duration_ms,
                })
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log component health: {e}")

    def log_model_health(
        self,
        score: float,
        model_count: int,
        degraded_models: List[str],
        duration_ms: int,
    ) -> None:
        """Log model health check results."""
        if self.trace:
            try:
                self.trace_metadata.update({
                    "model_health_score": score,
                    "model_count": model_count,
                    "degraded_models": degraded_models,
                    "model_duration_ms": duration_ms,
                })
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log model health: {e}")

    def log_pipeline_health(
        self,
        score: float,
        pipeline_count: int,
        stale_pipelines: List[str],
        duration_ms: int,
    ) -> None:
        """Log pipeline health check results."""
        if self.trace:
            try:
                self.trace_metadata.update({
                    "pipeline_health_score": score,
                    "pipeline_count": pipeline_count,
                    "stale_pipelines": stale_pipelines,
                    "pipeline_duration_ms": duration_ms,
                })
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log pipeline health: {e}")

    def log_agent_health(
        self,
        score: float,
        agent_count: int,
        unavailable_agents: List[str],
        duration_ms: int,
    ) -> None:
        """Log agent health check results."""
        if self.trace:
            try:
                self.trace_metadata.update({
                    "agent_health_score": score,
                    "agent_count": agent_count,
                    "unavailable_agents": unavailable_agents,
                    "agent_duration_ms": duration_ms,
                })
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log agent health: {e}")

    def log_check_complete(
        self,
        status: str,
        success: bool,
        total_duration_ms: int,
        overall_score: float,
        health_grade: str,
        component_score: float,
        model_score: float,
        pipeline_score: float,
        agent_score: float,
        critical_issues: List[str],
        warnings: List[str],
    ) -> None:
        """Log health check completion."""
        if self.trace:
            try:
                elapsed_ms = int((time.time() - self.start_time) * 1000)
                self.trace_metadata.update({
                    "status": status,
                    "success": success,
                    "total_duration_ms": total_duration_ms,
                    "trace_duration_ms": elapsed_ms,
                    "overall_health_score": overall_score,
                    "health_grade": health_grade,
                    "component_health_score": component_score,
                    "model_health_score": model_score,
                    "pipeline_health_score": pipeline_score,
                    "agent_health_score": agent_score,
                    "critical_issues_count": len(critical_issues),
                    "critical_issues": critical_issues[:10],  # Limit for trace size
                    "warnings_count": len(warnings),
                    "warnings": warnings[:10],  # Limit for trace size
                    "check_completed_at": datetime.now(timezone.utc).isoformat(),
                })
                self.trace.update(
                    metadata=self.trace_metadata,
                    output={
                        "overall_score": overall_score,
                        "grade": health_grade,
                        "status": status,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to log check complete: {e}")


# ============================================================================
# MAIN TRACER CLASS
# ============================================================================


class HealthScoreOpikTracer:
    """
    Opik distributed tracer for Health Score agent.

    Provides tracing for:
    - Health check operations (full, quick, scoped)
    - Component health checks
    - Model health checks
    - Pipeline health checks
    - Agent health checks
    - Score composition

    Usage:
        tracer = HealthScoreOpikTracer()
        async with tracer.trace_health_check(scope="full") as ctx:
            ctx.log_check_started(scope)
            # ... perform health checks ...
            ctx.log_check_complete(status="success", ...)

    Thread Safety:
        Uses singleton pattern - safe for concurrent access.
    """

    _instance: Optional["HealthScoreOpikTracer"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "HealthScoreOpikTracer":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        project_name: str = "e2i-health-score",
        sampling_rate: float = 1.0,
        enabled: bool = True,
    ):
        """
        Initialize the Opik tracer.

        Args:
            project_name: Opik project name for traces
            sampling_rate: Fraction of traces to capture (0.0-1.0)
            enabled: Whether tracing is enabled
        """
        if self._initialized:
            return

        self.project_name = project_name
        self.sampling_rate = sampling_rate
        self.enabled = enabled
        self._client: Optional["Opik"] = None
        self._initialized = True

        logger.info(
            f"HealthScoreOpikTracer initialized: project={project_name}, "
            f"sampling_rate={sampling_rate}, enabled={enabled}"
        )

    def _get_client(self) -> Optional["Opik"]:
        """Get or create Opik client (lazy initialization)."""
        if not self.enabled:
            return None

        if self._client is None:
            try:
                from opik import Opik

                self._client = Opik(project_name=self.project_name)
                logger.debug("Opik client initialized for Health Score")
            except ImportError:
                logger.warning("Opik not available - tracing disabled")
                self.enabled = False
                return None
            except Exception as e:
                logger.warning(f"Failed to initialize Opik client: {e}")
                self.enabled = False
                return None

        return self._client

    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled."""
        import random

        return random.random() < self.sampling_rate

    def _generate_trace_id(self) -> str:
        """Generate a UUID v7 compatible trace ID for Opik."""
        # UUID v7 format: timestamp (48 bits) + version (4 bits) + random (12 bits) + variant (2 bits) + random (62 bits)
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # Use uuid4 for randomness, then encode timestamp in first 48 bits
        random_uuid = uuid.uuid4()
        # Extract bytes
        uuid_bytes = random_uuid.bytes
        # Encode timestamp in first 6 bytes
        timestamp_bytes = timestamp_ms.to_bytes(6, byteorder="big")
        # Combine: timestamp (6 bytes) + version nibble + random (1.5 bytes) + variant bits + random (8 bytes)
        new_bytes = (
            timestamp_bytes
            + bytes([0x70 | (uuid_bytes[6] & 0x0F)])  # Version 7
            + uuid_bytes[7:8]
            + bytes([0x80 | (uuid_bytes[8] & 0x3F)])  # Variant 10
            + uuid_bytes[9:16]
        )
        return str(uuid.UUID(bytes=new_bytes))

    @asynccontextmanager
    async def trace_health_check(
        self,
        check_scope: str = "full",
        experiment_name: str = "default",
        check_id: Optional[str] = None,
    ):
        """
        Async context manager for tracing a health check operation.

        Args:
            check_scope: Scope of the health check (full, quick, etc.)
            experiment_name: Name of the experiment
            check_id: Optional unique identifier for this check

        Yields:
            HealthCheckTraceContext for logging trace data
        """
        if not self.enabled or not self._should_sample():
            yield HealthCheckTraceContext(
                trace=None,
                tracer=self,
                check_scope=check_scope,
            )
            return

        client = self._get_client()
        if client is None:
            yield HealthCheckTraceContext(
                trace=None,
                tracer=self,
                check_scope=check_scope,
            )
            return

        trace = None
        try:
            trace_id = self._generate_trace_id()
            trace = client.trace(
                name=f"health_score.{check_scope}",
                id=trace_id,
                input={
                    "check_scope": check_scope,
                    "experiment_name": experiment_name,
                },
                metadata={
                    "agent": "health_score",
                    "tier": 3,
                    "agent_type": "fast_path",
                    "check_scope": check_scope,
                    "check_id": check_id or trace_id,
                    "experiment_name": experiment_name,
                    "trace_started_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.debug(f"Started health check trace: {trace_id}")

            ctx = HealthCheckTraceContext(
                trace=trace,
                tracer=self,
                check_scope=check_scope,
                trace_metadata={
                    "check_scope": check_scope,
                    "experiment_name": experiment_name,
                },
            )
            yield ctx

        except Exception as e:
            logger.warning(f"Failed to create health check trace: {e}")
            yield HealthCheckTraceContext(
                trace=None,
                tracer=self,
                check_scope=check_scope,
            )
        finally:
            if trace:
                try:
                    trace.end()
                except Exception as e:
                    logger.debug(f"Failed to end trace: {e}")

    def flush(self) -> None:
        """Flush any pending traces to Opik backend."""
        if self._client:
            try:
                self._client.flush()
                logger.debug("Flushed Opik traces for Health Score")
            except Exception as e:
                logger.debug(f"Failed to flush Opik traces: {e}")


# ============================================================================
# SINGLETON FACTORY
# ============================================================================

_tracer_instance: Optional[HealthScoreOpikTracer] = None


def get_health_score_tracer(
    project_name: str = "e2i-health-score",
    sampling_rate: float = 1.0,
    enabled: bool = True,
) -> HealthScoreOpikTracer:
    """
    Get or create the singleton Health Score Opik tracer.

    Args:
        project_name: Opik project name
        sampling_rate: Fraction of traces to capture
        enabled: Whether tracing is enabled

    Returns:
        HealthScoreOpikTracer instance
    """
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = HealthScoreOpikTracer(
            project_name=project_name,
            sampling_rate=sampling_rate,
            enabled=enabled,
        )
    return _tracer_instance
