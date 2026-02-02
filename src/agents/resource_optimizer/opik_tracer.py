"""
E2I Resource Optimizer Agent - Opik Distributed Tracing
Version: 4.2
Purpose: Distributed tracing for Resource Optimizer agent operations

This module provides Opik integration for the Resource Optimizer agent,
enabling distributed tracing across optimization operations.

Architecture:
    - ResourceOptimizerOpikTracer: Main tracer class with singleton pattern
    - OptimizationTraceContext: Async context manager for traces
    - NodeSpanContext: Context for individual node spans

Pipeline Nodes Traced:
    - formulate: Problem formulation (constraints, objectives)
    - optimize: Solver execution (linear, milp, nonlinear)
    - scenario: Scenario analysis (what-if comparisons)
    - project: Impact projection (outcome, ROI calculations)
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from opik import Opik, Trace
    from opik.api_objects.span import Span

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

OPTIMIZATION_OBJECTIVES = ["maximize_outcome", "maximize_roi", "minimize_cost", "balance"]
SOLVER_TYPES = ["linear", "milp", "nonlinear"]
PIPELINE_NODES = ["formulate", "optimize", "scenario", "project"]


# ============================================================================
# SPAN CONTEXT
# ============================================================================


@dataclass
class NodeSpanContext:
    """Context for individual optimization pipeline node spans."""

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
class OptimizationTraceContext:
    """Async context manager for resource optimization traces."""

    trace: Optional["Trace"]
    tracer: "ResourceOptimizerOpikTracer"
    resource_type: str
    objective: str
    start_time: float = field(default_factory=time.time)
    trace_metadata: Dict[str, Any] = field(default_factory=dict)
    active_spans: Dict[str, NodeSpanContext] = field(default_factory=dict)

    def log_optimization_started(
        self,
        resource_type: str,
        objective: str,
        solver_type: str,
        target_count: int,
        constraint_count: int,
        run_scenarios: bool,
    ) -> None:
        """Log that an optimization has started."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "resource_type": resource_type,
                        "objective": objective,
                        "solver_type": solver_type,
                        "target_count": target_count,
                        "constraint_count": constraint_count,
                        "run_scenarios": run_scenarios,
                        "optimization_started_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log optimization started: {e}")

    def start_node_span(
        self,
        node_name: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> NodeSpanContext:
        """Start a span for an optimization pipeline node."""
        span = None
        if self.trace and self.tracer._client:
            try:
                span = self.trace.span(
                    name=f"resource_optimizer.{node_name}",
                    input=input_data or {},
                    metadata={
                        "node_name": node_name,
                        "resource_type": self.resource_type,
                        "objective": self.objective,
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

    def log_problem_formulation(
        self,
        target_count: int,
        constraint_count: int,
        variables_count: int,
        formulation_latency_ms: int,
    ) -> None:
        """Log problem formulation results."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "target_count": target_count,
                        "constraint_count": constraint_count,
                        "variables_count": variables_count,
                        "formulation_latency_ms": formulation_latency_ms,
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log problem formulation: {e}")

    def log_solver_execution(
        self,
        solver_type: str,
        solver_status: str,
        solve_time_ms: int,
        objective_value: Optional[float],
        allocations_count: int,
    ) -> None:
        """Log solver execution results."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "solver_type": solver_type,
                        "solver_status": solver_status,
                        "solve_time_ms": solve_time_ms,
                        "objective_value": objective_value,
                        "allocations_count": allocations_count,
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log solver execution: {e}")

    def log_scenario_analysis(
        self,
        scenario_count: int,
        best_scenario: Optional[str],
        scenario_latency_ms: int,
    ) -> None:
        """Log scenario analysis results."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "scenario_count": scenario_count,
                        "best_scenario": best_scenario,
                        "scenario_latency_ms": scenario_latency_ms,
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log scenario analysis: {e}")

    def log_impact_projection(
        self,
        projected_outcome: Optional[float],
        projected_roi: Optional[float],
        segments_count: int,
        projection_latency_ms: int,
    ) -> None:
        """Log impact projection results."""
        if self.trace:
            try:
                self.trace_metadata.update(
                    {
                        "projected_outcome": projected_outcome,
                        "projected_roi": projected_roi,
                        "segments_count": segments_count,
                        "projection_latency_ms": projection_latency_ms,
                    }
                )
                self.trace.update(metadata=self.trace_metadata)
            except Exception as e:
                logger.debug(f"Failed to log impact projection: {e}")

    def log_optimization_complete(
        self,
        status: str,
        success: bool,
        total_duration_ms: int,
        objective_value: Optional[float],
        solver_status: Optional[str],
        projected_outcome: Optional[float],
        projected_roi: Optional[float],
        allocations_count: int,
        increases_count: int,
        decreases_count: int,
        recommendations: List[str],
        errors: List[Dict[str, Any]],
        warnings: List[str],
    ) -> None:
        """Log optimization completion."""
        if self.trace:
            try:
                elapsed_ms = int((time.time() - self.start_time) * 1000)
                self.trace_metadata.update(
                    {
                        "status": status,
                        "success": success,
                        "total_duration_ms": total_duration_ms,
                        "trace_duration_ms": elapsed_ms,
                        "objective_value": objective_value,
                        "solver_status": solver_status,
                        "projected_outcome": projected_outcome,
                        "projected_roi": projected_roi,
                        "allocations_count": allocations_count,
                        "increases_count": increases_count,
                        "decreases_count": decreases_count,
                        "recommendations_count": len(recommendations),
                        "recommendations": recommendations[:5],  # Limit for trace size
                        "errors_count": len(errors),
                        "warnings_count": len(warnings),
                        "warnings": warnings[:5],  # Limit for trace size
                        "optimization_completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                self.trace.update(
                    metadata=self.trace_metadata,
                    output={
                        "objective_value": objective_value,
                        "projected_roi": projected_roi,
                        "status": status,
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to log optimization complete: {e}")


# ============================================================================
# MAIN TRACER CLASS
# ============================================================================


class ResourceOptimizerOpikTracer:
    """
    Opik distributed tracer for Resource Optimizer agent.

    Provides tracing for:
    - Resource optimization operations
    - Problem formulation
    - Solver execution
    - Scenario analysis
    - Impact projection

    Usage:
        tracer = ResourceOptimizerOpikTracer()
        async with tracer.trace_optimization(
            resource_type="budget",
            objective="maximize_outcome"
        ) as ctx:
            ctx.log_optimization_started(...)
            # ... perform optimization ...
            ctx.log_optimization_complete(...)

    Thread Safety:
        Uses singleton pattern - safe for concurrent access.
    """

    _instance: Optional["ResourceOptimizerOpikTracer"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "ResourceOptimizerOpikTracer":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        project_name: str = "e2i-resource-optimizer",
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
            f"ResourceOptimizerOpikTracer initialized: project={project_name}, "
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
                logger.debug("Opik client initialized for Resource Optimizer")
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
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        random_uuid = uuid.uuid4()
        uuid_bytes = random_uuid.bytes
        timestamp_bytes = timestamp_ms.to_bytes(6, byteorder="big")
        new_bytes = (
            timestamp_bytes
            + bytes([0x70 | (uuid_bytes[6] & 0x0F)])  # Version 7
            + uuid_bytes[7:8]
            + bytes([0x80 | (uuid_bytes[8] & 0x3F)])  # Variant 10
            + uuid_bytes[9:16]
        )
        return str(uuid.UUID(bytes=new_bytes))

    @asynccontextmanager
    async def trace_optimization(
        self,
        resource_type: str = "budget",
        objective: str = "maximize_outcome",
        solver_type: str = "linear",
        optimization_id: Optional[str] = None,
        query: Optional[str] = None,
    ):
        """
        Async context manager for tracing an optimization operation.

        Args:
            resource_type: Type of resource being optimized
            objective: Optimization objective
            solver_type: Solver type being used
            optimization_id: Optional unique identifier for this optimization
            query: Optional original query text

        Yields:
            OptimizationTraceContext for logging trace data
        """
        if not self.enabled or not self._should_sample():
            yield OptimizationTraceContext(
                trace=None,
                tracer=self,
                resource_type=resource_type,
                objective=objective,
            )
            return

        client = self._get_client()
        if client is None:
            yield OptimizationTraceContext(
                trace=None,
                tracer=self,
                resource_type=resource_type,
                objective=objective,
            )
            return

        trace = None
        try:
            trace_id = self._generate_trace_id()
            trace = client.trace(
                name=f"resource_optimizer.{resource_type}.{objective}",
                id=trace_id,
                input={
                    "resource_type": resource_type,
                    "objective": objective,
                    "solver_type": solver_type,
                    "query": query or "",
                },
                metadata={
                    "agent": "resource_optimizer",
                    "tier": 4,
                    "agent_type": "ml_predictions",
                    "resource_type": resource_type,
                    "objective": objective,
                    "solver_type": solver_type,
                    "optimization_id": optimization_id or trace_id,
                    "trace_started_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.debug(f"Started optimization trace: {trace_id}")

            ctx = OptimizationTraceContext(
                trace=trace,
                tracer=self,
                resource_type=resource_type,
                objective=objective,
                trace_metadata={
                    "resource_type": resource_type,
                    "objective": objective,
                    "solver_type": solver_type,
                },
            )
            yield ctx

        except Exception as e:
            logger.warning(f"Failed to create optimization trace: {e}")
            yield OptimizationTraceContext(
                trace=None,
                tracer=self,
                resource_type=resource_type,
                objective=objective,
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
                logger.debug("Flushed Opik traces for Resource Optimizer")
            except Exception as e:
                logger.debug(f"Failed to flush Opik traces: {e}")


# ============================================================================
# SINGLETON FACTORY
# ============================================================================

_tracer_instance: Optional[ResourceOptimizerOpikTracer] = None


def get_resource_optimizer_tracer(
    project_name: str = "e2i-resource-optimizer",
    sampling_rate: float = 1.0,
    enabled: bool = True,
) -> ResourceOptimizerOpikTracer:
    """
    Get or create the singleton Resource Optimizer Opik tracer.

    Args:
        project_name: Opik project name
        sampling_rate: Fraction of traces to capture
        enabled: Whether tracing is enabled

    Returns:
        ResourceOptimizerOpikTracer instance
    """
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = ResourceOptimizerOpikTracer(
            project_name=project_name,
            sampling_rate=sampling_rate,
            enabled=enabled,
        )
    return _tracer_instance
