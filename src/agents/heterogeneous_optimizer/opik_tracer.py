"""
Opik Integration for Heterogeneous Optimizer Agent.

This module provides Opik tracing utilities for the Heterogeneous Optimizer
pipeline, enabling observability of:
- Full CATE analysis traces
- Node-level spans (estimate_cate → analyze_segments → learn_policy → generate_profiles)
- Segment heterogeneity metrics
- Policy recommendation tracking

Usage:
    from src.agents.heterogeneous_optimizer.opik_tracer import (
        HeterogeneousOptimizerOpikTracer,
        get_heterogeneous_optimizer_tracer,
    )

    tracer = get_heterogeneous_optimizer_tracer()

    # Trace a full CATE analysis
    async with tracer.trace_analysis(query="...", treatment="rep_visits") as trace:
        async with trace.trace_node("estimate_cate") as node:
            cate_results = await estimate_cate(...)
            node.log_cate_estimation(...)
        # ... other nodes
        trace.log_analysis_complete(result)

Author: E2I Causal Analytics Team
Version: 4.3.0
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar

from uuid_utils import uuid7 as uuid7_func

if TYPE_CHECKING:
    from src.mlops.opik_connector import OpikConnector

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# Agent metadata
AGENT_METADATA = {
    "name": "heterogeneous_optimizer",
    "tier": 2,
    "type": "standard",
    "pipeline": "estimate_cate → analyze_segments → learn_policy → generate_profiles",
}


@dataclass
class NodeSpanContext:
    """Context for a Heterogeneous Optimizer pipeline node span.

    Provides methods to log node-specific events and metrics.

    Attributes:
        trace_id: Parent trace identifier
        span_id: This span's identifier
        node_name: Name of the node
        start_time: When the node started
        end_time: When the node ended (set on exit)
        duration_ms: Node duration in milliseconds
        metadata: Additional node metadata
        _opik_span: Reference to the Opik span context
    """

    trace_id: str
    span_id: str
    node_name: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _opik_span: Optional[Any] = None
    _parent_ctx: Optional["CATEAnalysisTraceContext"] = None

    def log_cate_estimation(
        self,
        segments_count: int,
        overall_ate: float,
        heterogeneity_score: float,
        n_estimators: int = 100,
        estimation_method: str = "CausalForestDML",
        **kwargs: Any,
    ) -> None:
        """Log CATE estimation node metrics.

        Args:
            segments_count: Number of segments estimated
            overall_ate: Overall Average Treatment Effect
            heterogeneity_score: Heterogeneity score (0-1)
            n_estimators: Number of estimators used
            estimation_method: ML method used for CATE
            **kwargs: Additional metrics
        """
        self.metadata.update(
            {
                "segments_count": segments_count,
                "overall_ate": overall_ate,
                "heterogeneity_score": heterogeneity_score,
                "n_estimators": n_estimators,
                "estimation_method": estimation_method,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("segments_count", segments_count)
            self._opik_span.set_attribute("overall_ate", overall_ate)
            self._opik_span.set_attribute("heterogeneity_score", heterogeneity_score)
            self._opik_span.add_event(
                "cate_estimation_complete",
                {
                    "segments_count": segments_count,
                    "overall_ate": overall_ate,
                    "heterogeneity_score": heterogeneity_score,
                },
            )

        logger.debug(
            f"[ESTIMATE_CATE] {segments_count} segments, "
            f"ATE={overall_ate:.4f}, heterogeneity={heterogeneity_score:.3f}"
        )

    def log_segment_analysis(
        self,
        high_responders_count: int,
        low_responders_count: int,
        total_segments_analyzed: int,
        significant_effects_count: int = 0,
        max_cate: float = 0.0,
        min_cate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Log segment analysis node metrics.

        Args:
            high_responders_count: Number of high responder segments
            low_responders_count: Number of low responder segments
            total_segments_analyzed: Total segments analyzed
            significant_effects_count: Segments with statistically significant effects
            max_cate: Maximum CATE value
            min_cate: Minimum CATE value
            **kwargs: Additional metrics
        """
        self.metadata.update(
            {
                "high_responders_count": high_responders_count,
                "low_responders_count": low_responders_count,
                "total_segments_analyzed": total_segments_analyzed,
                "significant_effects_count": significant_effects_count,
                "max_cate": max_cate,
                "min_cate": min_cate,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("high_responders_count", high_responders_count)
            self._opik_span.set_attribute("low_responders_count", low_responders_count)
            self._opik_span.set_attribute("total_segments_analyzed", total_segments_analyzed)
            self._opik_span.add_event(
                "segment_analysis_complete",
                {
                    "high_responders_count": high_responders_count,
                    "low_responders_count": low_responders_count,
                    "significant_effects_count": significant_effects_count,
                },
            )

        logger.debug(
            f"[ANALYZE_SEGMENTS] {total_segments_analyzed} segments: "
            f"{high_responders_count} high, {low_responders_count} low responders"
        )

    def log_policy_learning(
        self,
        recommendations_count: int,
        expected_total_lift: float,
        reallocations_suggested: int = 0,
        budget_impact: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Log policy learner node metrics.

        Args:
            recommendations_count: Number of policy recommendations
            expected_total_lift: Total expected lift from policy
            reallocations_suggested: Number of treatment reallocations
            budget_impact: Estimated budget impact
            **kwargs: Additional metrics
        """
        self.metadata.update(
            {
                "recommendations_count": recommendations_count,
                "expected_total_lift": expected_total_lift,
                "reallocations_suggested": reallocations_suggested,
                "budget_impact": budget_impact,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("recommendations_count", recommendations_count)
            self._opik_span.set_attribute("expected_total_lift", expected_total_lift)
            self._opik_span.set_attribute("reallocations_suggested", reallocations_suggested)
            self._opik_span.add_event(
                "policy_learning_complete",
                {
                    "recommendations_count": recommendations_count,
                    "expected_total_lift": expected_total_lift,
                    "reallocations_suggested": reallocations_suggested,
                },
            )

        logger.debug(
            f"[LEARN_POLICY] {recommendations_count} recommendations, "
            f"expected lift={expected_total_lift:.2f}"
        )

    def log_profile_generation(
        self,
        profiles_generated: int,
        insights_count: int,
        summary_length: int = 0,
        **kwargs: Any,
    ) -> None:
        """Log profile generator node metrics.

        Args:
            profiles_generated: Number of segment profiles created
            insights_count: Number of key insights generated
            summary_length: Length of executive summary
            **kwargs: Additional metrics
        """
        self.metadata.update(
            {
                "profiles_generated": profiles_generated,
                "insights_count": insights_count,
                "summary_length": summary_length,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("profiles_generated", profiles_generated)
            self._opik_span.set_attribute("insights_count", insights_count)
            self._opik_span.add_event(
                "profile_generation_complete",
                {
                    "profiles_generated": profiles_generated,
                    "insights_count": insights_count,
                },
            )

        logger.debug(
            f"[GENERATE_PROFILES] {profiles_generated} profiles, {insights_count} insights"
        )

    def set_output(self, output: Dict[str, Any]) -> None:
        """Set the output data for this node span."""
        if self._opik_span:
            self._opik_span.set_output(output)


@dataclass
class CATEAnalysisTraceContext:
    """Context for a full Heterogeneous Optimizer analysis trace.

    Provides methods to create node spans and log overall metrics.

    Attributes:
        trace_id: Unique trace identifier
        span_id: Root span identifier
        query: Original query being analyzed
        treatment_var: Treatment variable being analyzed
        start_time: When analysis started
        end_time: When analysis ended
        node_spans: Child spans for each pipeline node
        metadata: Additional trace metadata
        _opik_span: Reference to the Opik span
        _tracer: Reference to parent tracer
    """

    trace_id: str
    span_id: str
    query: str
    treatment_var: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    node_spans: Dict[str, NodeSpanContext] = field(default_factory=dict)
    node_durations: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _opik_span: Optional[Any] = None
    _tracer: Optional["HeterogeneousOptimizerOpikTracer"] = None

    @asynccontextmanager
    async def trace_node(
        self,
        node_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing a specific pipeline node.

        Args:
            node_name: Name of the node
            metadata: Additional node metadata

        Yields:
            NodeSpanContext for logging node events
        """
        span_id = str(uuid7_func())
        start_time = datetime.now(timezone.utc)

        node_ctx = NodeSpanContext(
            trace_id=self.trace_id,
            span_id=span_id,
            node_name=node_name,
            start_time=start_time,
            metadata=metadata or {},
            _parent_ctx=self,
        )

        try:
            # Create child span in Opik if parent is traced
            if self._opik_span and self._tracer and self._tracer.enabled:
                try:
                    from src.mlops.opik_connector import get_opik_connector

                    connector = get_opik_connector()
                    if connector.is_enabled:
                        async with connector.trace_agent(
                            agent_name="heterogeneous_optimizer",
                            operation=node_name,
                            trace_id=self.trace_id,
                            parent_span_id=self.span_id,
                            metadata={
                                "node": node_name,
                                "node_index": self._get_node_index(node_name),
                                **(metadata or {}),
                            },
                            tags=["heterogeneous_optimizer", node_name, "pipeline_node"],
                        ) as span:
                            node_ctx._opik_span = span
                            yield node_ctx
                            span.set_output(node_ctx.metadata)
                        return
                except Exception as e:
                    logger.debug(f"Failed to create node span: {e}")

            # Fall through to non-traced version
            yield node_ctx

        finally:
            # Record duration
            end_time = datetime.now(timezone.utc)
            node_ctx.end_time = end_time
            node_ctx.duration_ms = (end_time - start_time).total_seconds() * 1000
            self.node_durations[node_name] = int(node_ctx.duration_ms)

            # Store in parent context
            self.node_spans[node_name] = node_ctx

            logger.debug(f"Node {node_name} completed in {node_ctx.duration_ms:.2f}ms")

    def _get_node_index(self, node_name: str) -> int:
        """Get numeric index for node ordering."""
        node_order = [
            "audit_init",
            "estimate_cate",
            "analyze_segments",
            "hierarchical_analysis",
            "learn_policy",
            "generate_profiles",
        ]
        return node_order.index(node_name) if node_name in node_order else -1

    def log_analysis_complete(
        self,
        status: str,
        success: bool,
        total_duration_ms: int,
        overall_ate: float = 0.0,
        heterogeneity_score: float = 0.0,
        high_responders_count: int = 0,
        low_responders_count: int = 0,
        recommendations_count: int = 0,
        expected_total_lift: float = 0.0,
        confidence: float = 0.0,
        errors: Optional[List[str]] = None,
        suggested_next_agent: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log completion of the full CATE analysis.

        Args:
            status: Final status (success, partial, failed)
            success: Whether analysis succeeded
            total_duration_ms: Total duration in milliseconds
            overall_ate: Overall average treatment effect
            heterogeneity_score: Heterogeneity score
            high_responders_count: Number of high responder segments
            low_responders_count: Number of low responder segments
            recommendations_count: Number of policy recommendations
            expected_total_lift: Expected lift from optimal policy
            confidence: Final confidence score
            errors: Any errors encountered
            suggested_next_agent: Recommended follow-up agent
            **kwargs: Additional metrics
        """
        output_data = {
            "status": status,
            "success": success,
            "total_duration_ms": total_duration_ms,
            "overall_ate": overall_ate,
            "heterogeneity_score": heterogeneity_score,
            "high_responders_count": high_responders_count,
            "low_responders_count": low_responders_count,
            "recommendations_count": recommendations_count,
            "expected_total_lift": expected_total_lift,
            "confidence": confidence,
            "node_durations": self.node_durations,
            "errors": errors or [],
            "suggested_next_agent": suggested_next_agent,
            **kwargs,
        }

        if self._opik_span:
            # Set key attributes for filtering
            self._opik_span.set_attribute("status", status)
            self._opik_span.set_attribute("success", success)
            self._opik_span.set_attribute("total_duration_ms", total_duration_ms)
            self._opik_span.set_attribute("overall_ate", overall_ate)
            self._opik_span.set_attribute("heterogeneity_score", heterogeneity_score)
            self._opik_span.set_attribute("recommendations_count", recommendations_count)
            self._opik_span.set_attribute("confidence", confidence)

            # Set output data
            self._opik_span.set_output(output_data)

        logger.info(
            f"CATE analysis complete: status={status}, "
            f"ATE={overall_ate:.4f}, heterogeneity={heterogeneity_score:.3f}, "
            f"{high_responders_count} high/{low_responders_count} low responders, "
            f"{recommendations_count} recommendations, "
            f"confidence={confidence:.2f}, "
            f"duration={total_duration_ms}ms"
        )


class HeterogeneousOptimizerOpikTracer:
    """Opik tracer for Heterogeneous Optimizer pipeline.

    Provides observability into the CATE analysis pipeline with:
    - Root trace for full analysis
    - Child spans for each pipeline node
    - CATE estimation and segment metrics
    - Policy recommendation tracking

    Uses the shared OpikConnector for circuit breaker protection.

    Example:
        >>> tracer = HeterogeneousOptimizerOpikTracer()
        >>> async with tracer.trace_analysis(query="...", treatment_var="rep_visits") as trace:
        ...     async with trace.trace_node("estimate_cate") as node:
        ...         result = await estimate_cate(...)
        ...         node.log_cate_estimation(...)
        ...     # ... other nodes
        ...     trace.log_analysis_complete(...)
    """

    def __init__(
        self,
        project_name: str = "e2i-heterogeneous-optimizer",
        enabled: bool = True,
        sample_rate: float = 1.0,
    ):
        """Initialize the Heterogeneous Optimizer tracer.

        Args:
            project_name: Opik project name
            enabled: Whether tracing is enabled
            sample_rate: Sample rate (1.0 = trace all, 0.1 = 10%)
        """
        self.project_name = project_name
        self.enabled = enabled
        self.sample_rate = sample_rate
        self._opik_connector: Optional[OpikConnector] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of OpikConnector."""
        if self._initialized:
            return

        try:
            from src.mlops.opik_connector import get_opik_connector

            self._opik_connector = get_opik_connector()
            self._initialized = True
            logger.debug("HeterogeneousOptimizerOpikTracer initialized")
        except ImportError:
            logger.warning("OpikConnector not available, tracing disabled")
            self._opik_connector = None
            self._initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize OpikConnector: {e}")
            self._opik_connector = None
            self._initialized = True

    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled and available."""
        self._ensure_initialized()
        return self.enabled and self._opik_connector is not None and self._opik_connector.is_enabled

    def _should_trace(self) -> bool:
        """Determine if this analysis should be traced."""
        import random

        return random.random() < self.sample_rate

    @asynccontextmanager
    async def trace_analysis(
        self,
        query: str,
        treatment_var: str,
        outcome_var: Optional[str] = None,
        segment_vars: Optional[List[str]] = None,
        brand: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing a full CATE analysis.

        Creates an Opik trace for the analysis and provides a context
        object for creating node spans and logging metrics.

        Args:
            query: The query being analyzed
            treatment_var: Treatment variable being analyzed
            outcome_var: Outcome variable
            segment_vars: Segment variables for analysis
            brand: Brand being analyzed
            session_id: Session identifier
            metadata: Additional trace metadata

        Yields:
            CATEAnalysisTraceContext for node tracing and metric logging
        """
        self._ensure_initialized()

        trace_id = str(uuid7_func())
        span_id = str(uuid7_func())
        start_time = datetime.now(timezone.utc)

        # Build metadata
        trace_metadata = {
            "query_length": len(query),
            "treatment_var": treatment_var,
            "outcome_var": outcome_var,
            "segment_vars_count": len(segment_vars) if segment_vars else 0,
            "brand": brand,
            "session_id": session_id,
            **(metadata or {}),
        }

        # Create trace context
        trace_ctx = CATEAnalysisTraceContext(
            trace_id=trace_id,
            span_id=span_id,
            query=query,
            treatment_var=treatment_var,
            start_time=start_time,
            metadata=trace_metadata,
            _tracer=self,
        )

        try:
            # Create Opik trace if enabled and sampled
            opik_span = None
            if self.is_enabled and self._should_trace():
                try:
                    # Enter the Opik context manager manually to avoid nested yield issues
                    assert self._opik_connector is not None
                    opik_cm = self._opik_connector.trace_agent(
                        agent_name="heterogeneous_optimizer",
                        operation="analyze",
                        trace_id=trace_id,
                        metadata={
                            "pipeline": AGENT_METADATA["pipeline"],
                            "tier": AGENT_METADATA["tier"],
                            "agent_type": AGENT_METADATA["type"],
                            **trace_metadata,
                        },
                        tags=[
                            "heterogeneous_optimizer",
                            "cate_analysis",
                            "tier2",
                            brand or "unknown",
                        ],
                        input_data={
                            "query": query[:500],  # Truncate for Opik
                            "treatment_var": treatment_var,
                            "outcome_var": outcome_var,
                            "segment_vars": (segment_vars or [])[:10],
                        },
                        force_new_trace=True,
                    )
                    opik_span = await opik_cm.__aenter__()
                    trace_ctx._opik_span = opik_span
                except Exception as e:
                    logger.debug(f"Opik tracing failed, continuing without: {e}")
                    opik_span = None
                    opik_cm = None

            # Single yield point - avoids "generator didn't stop after athrow()" errors
            yield trace_ctx

        except Exception as e:
            {"type": type(e).__name__, "message": str(e)}
            raise

        finally:
            # Clean up Opik context manager if it was entered
            if opik_span is not None and opik_cm is not None:
                try:
                    await opik_cm.__aexit__(None, None, None)
                except Exception as e:
                    logger.debug(f"Opik cleanup failed (non-fatal): {e}")

            # Record final timing
            end_time = datetime.now(timezone.utc)
            trace_ctx.end_time = end_time
            trace_ctx.duration_ms = (end_time - start_time).total_seconds() * 1000

            logger.debug(f"CATE analysis trace completed in {trace_ctx.duration_ms:.2f}ms")


def trace_cate_analysis(
    query_param: str = "query",
    treatment_param: str = "treatment_var",
    project_name: str = "e2i-heterogeneous-optimizer",
) -> Callable[[F], F]:
    """Decorator to trace a CATE analysis function.

    Use this decorator on analysis functions for automatic tracing.

    Args:
        query_param: Name of the query parameter in the decorated function
        treatment_param: Name of the treatment parameter
        project_name: Opik project name

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract query and treatment from kwargs or args
            query = kwargs.get(query_param, args[0] if args else "")
            treatment = kwargs.get(treatment_param, args[1] if len(args) > 1 else "unknown")

            tracer = HeterogeneousOptimizerOpikTracer(project_name=project_name)

            async with tracer.trace_analysis(
                query=query,
                treatment_var=treatment,
            ) as trace_ctx:
                kwargs["_trace_ctx"] = trace_ctx
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# Singleton tracer instance
_tracer_instance: Optional[HeterogeneousOptimizerOpikTracer] = None


def get_heterogeneous_optimizer_tracer(
    project_name: str = "e2i-heterogeneous-optimizer",
    enabled: bool = True,
    sample_rate: float = 1.0,
) -> HeterogeneousOptimizerOpikTracer:
    """Get the Heterogeneous Optimizer Opik tracer singleton.

    Args:
        project_name: Opik project name
        enabled: Whether tracing is enabled
        sample_rate: Sample rate for tracing

    Returns:
        HeterogeneousOptimizerOpikTracer instance
    """
    global _tracer_instance

    if _tracer_instance is None:
        _tracer_instance = HeterogeneousOptimizerOpikTracer(
            project_name=project_name,
            enabled=enabled,
            sample_rate=sample_rate,
        )

    return _tracer_instance


def reset_heterogeneous_optimizer_tracer() -> None:
    """Reset the tracer singleton (for testing)."""
    global _tracer_instance
    _tracer_instance = None


__all__ = [
    "NodeSpanContext",
    "CATEAnalysisTraceContext",
    "HeterogeneousOptimizerOpikTracer",
    "trace_cate_analysis",
    "get_heterogeneous_optimizer_tracer",
    "reset_heterogeneous_optimizer_tracer",
    "AGENT_METADATA",
]
