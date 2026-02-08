"""
Opik Integration for Gap Analyzer Agent.

This module provides Opik tracing utilities for the Gap Analyzer 4-node pipeline,
enabling observability of:
- Full gap analysis traces
- Node-level spans (gap_detector → roi_calculator → prioritizer → formatter)
- Opportunity detection metrics
- ROI calculation tracking

Usage:
    from src.agents.gap_analyzer.opik_tracer import (
        GapAnalyzerOpikTracer,
        get_gap_analyzer_tracer,
    )

    tracer = get_gap_analyzer_tracer()

    # Trace a full gap analysis
    async with tracer.trace_analysis(query="...", brand="Kisqali") as trace:
        async with trace.trace_node("gap_detector") as node:
            gaps = await detect_gaps(...)
            node.log_gap_detection(len(gaps), segments_analyzed)
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
    "name": "gap_analyzer",
    "tier": 2,
    "type": "standard",
    "pipeline": "gap_detector → roi_calculator → prioritizer → formatter",
}


@dataclass
class NodeSpanContext:
    """Context for a Gap Analyzer pipeline node span.

    Provides methods to log node-specific events and metrics.

    Attributes:
        trace_id: Parent trace identifier
        span_id: This span's identifier
        node_name: Name of the node (gap_detector, roi_calculator, prioritizer, formatter)
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
    _parent_ctx: Optional["GapAnalysisTraceContext"] = None

    def log_gap_detection(
        self,
        gaps_detected: int,
        segments_analyzed: int,
        gap_types: Optional[List[str]] = None,
        avg_gap_percentage: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Log gap detection node metrics.

        Args:
            gaps_detected: Number of gaps found
            segments_analyzed: Number of segments analyzed
            gap_types: Types of gaps detected
            avg_gap_percentage: Average gap percentage across all gaps
            **kwargs: Additional metrics
        """
        self.metadata.update(
            {
                "gaps_detected": gaps_detected,
                "segments_analyzed": segments_analyzed,
                "gap_types": gap_types or [],
                "avg_gap_percentage": avg_gap_percentage,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("gaps_detected", gaps_detected)
            self._opik_span.set_attribute("segments_analyzed", segments_analyzed)
            self._opik_span.add_event(
                "gap_detection_complete",
                {
                    "gaps_detected": gaps_detected,
                    "segments_analyzed": segments_analyzed,
                    "gap_types": (gap_types or [])[:5],  # Limit for Opik
                },
            )

        logger.debug(
            f"[GAP_DETECTOR] {gaps_detected} gaps detected across {segments_analyzed} segments"
        )

    def log_roi_calculation(
        self,
        opportunities_analyzed: int,
        total_addressable_value: float,
        avg_roi: float = 0.0,
        roi_confidence: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Log ROI calculator node metrics.

        Args:
            opportunities_analyzed: Number of opportunities with ROI calculated
            total_addressable_value: Sum of all addressable values
            avg_roi: Average expected ROI across opportunities
            roi_confidence: Average confidence in ROI estimates
            **kwargs: Additional metrics
        """
        self.metadata.update(
            {
                "opportunities_analyzed": opportunities_analyzed,
                "total_addressable_value": total_addressable_value,
                "avg_roi": avg_roi,
                "roi_confidence": roi_confidence,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("opportunities_analyzed", opportunities_analyzed)
            self._opik_span.set_attribute("total_addressable_value", total_addressable_value)
            self._opik_span.set_attribute("avg_roi", avg_roi)
            self._opik_span.add_event(
                "roi_calculation_complete",
                {
                    "opportunities_analyzed": opportunities_analyzed,
                    "total_addressable_value": total_addressable_value,
                    "avg_roi": avg_roi,
                },
            )

        logger.debug(
            f"[ROI_CALCULATOR] {opportunities_analyzed} opportunities, "
            f"${total_addressable_value:,.0f} addressable value"
        )

    def log_prioritization(
        self,
        total_opportunities: int,
        quick_wins: int,
        strategic_bets: int,
        top_priority_value: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Log prioritizer node metrics.

        Args:
            total_opportunities: Total opportunities prioritized
            quick_wins: Number of quick win opportunities
            strategic_bets: Number of strategic bet opportunities
            top_priority_value: Value of the top priority opportunity
            **kwargs: Additional metrics
        """
        self.metadata.update(
            {
                "total_opportunities": total_opportunities,
                "quick_wins": quick_wins,
                "strategic_bets": strategic_bets,
                "top_priority_value": top_priority_value,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("total_opportunities", total_opportunities)
            self._opik_span.set_attribute("quick_wins", quick_wins)
            self._opik_span.set_attribute("strategic_bets", strategic_bets)
            self._opik_span.add_event(
                "prioritization_complete",
                {
                    "total_opportunities": total_opportunities,
                    "quick_wins": quick_wins,
                    "strategic_bets": strategic_bets,
                },
            )

        logger.debug(
            f"[PRIORITIZER] {total_opportunities} opportunities: "
            f"{quick_wins} quick wins, {strategic_bets} strategic bets"
        )

    def log_formatting(
        self,
        summary_length: int,
        insights_count: int,
        recommendations_count: int = 0,
        **kwargs: Any,
    ) -> None:
        """Log formatter node metrics.

        Args:
            summary_length: Length of executive summary
            insights_count: Number of key insights generated
            recommendations_count: Number of recommendations
            **kwargs: Additional metrics
        """
        self.metadata.update(
            {
                "summary_length": summary_length,
                "insights_count": insights_count,
                "recommendations_count": recommendations_count,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("summary_length", summary_length)
            self._opik_span.set_attribute("insights_count", insights_count)
            self._opik_span.add_event(
                "formatting_complete",
                {
                    "summary_length": summary_length,
                    "insights_count": insights_count,
                    "recommendations_count": recommendations_count,
                },
            )

        logger.debug(f"[FORMATTER] {summary_length} char summary, {insights_count} insights")

    def set_output(self, output: Dict[str, Any]) -> None:
        """Set the output data for this node span."""
        if self._opik_span:
            self._opik_span.set_output(output)


@dataclass
class GapAnalysisTraceContext:
    """Context for a full Gap Analyzer analysis trace.

    Provides methods to create node spans and log overall metrics.

    Attributes:
        trace_id: Unique trace identifier
        span_id: Root span identifier
        query: Original query being analyzed
        brand: Brand being analyzed
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
    brand: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    node_spans: Dict[str, NodeSpanContext] = field(default_factory=dict)
    node_durations: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _opik_span: Optional[Any] = None
    _tracer: Optional["GapAnalyzerOpikTracer"] = None

    @asynccontextmanager
    async def trace_node(
        self,
        node_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing a specific pipeline node.

        Args:
            node_name: Name of the node (gap_detector, roi_calculator, prioritizer, formatter)
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
                        # Use the connector's trace_agent for child span
                        async with connector.trace_agent(
                            agent_name="gap_analyzer",
                            operation=node_name,
                            trace_id=self.trace_id,
                            parent_span_id=self.span_id,
                            metadata={
                                "node": node_name,
                                "node_index": self._get_node_index(node_name),
                                **(metadata or {}),
                            },
                            tags=["gap_analyzer", node_name, "pipeline_node"],
                        ) as span:
                            node_ctx._opik_span = span
                            yield node_ctx
                            # Set output on completion
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
        node_order = ["gap_detector", "roi_calculator", "prioritizer", "formatter"]
        return node_order.index(node_name) if node_name in node_order else -1

    def log_analysis_complete(
        self,
        status: str,
        success: bool,
        total_duration_ms: int,
        gaps_detected: int = 0,
        opportunities_count: int = 0,
        quick_wins_count: int = 0,
        strategic_bets_count: int = 0,
        total_addressable_value: float = 0.0,
        confidence: float = 0.0,
        errors: Optional[List[str]] = None,
        suggested_next_agent: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log completion of the full gap analysis.

        Args:
            status: Final status (success, partial, failed)
            success: Whether analysis succeeded
            total_duration_ms: Total duration in milliseconds
            gaps_detected: Number of gaps detected
            opportunities_count: Total prioritized opportunities
            quick_wins_count: Quick win opportunities
            strategic_bets_count: Strategic bet opportunities
            total_addressable_value: Total addressable value
            confidence: Final confidence score
            errors: Any errors encountered
            suggested_next_agent: Recommended follow-up agent
            **kwargs: Additional metrics
        """
        output_data = {
            "status": status,
            "success": success,
            "total_duration_ms": total_duration_ms,
            "gaps_detected": gaps_detected,
            "opportunities_count": opportunities_count,
            "quick_wins_count": quick_wins_count,
            "strategic_bets_count": strategic_bets_count,
            "total_addressable_value": total_addressable_value,
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
            self._opik_span.set_attribute("gaps_detected", gaps_detected)
            self._opik_span.set_attribute("opportunities_count", opportunities_count)
            self._opik_span.set_attribute("total_addressable_value", total_addressable_value)
            self._opik_span.set_attribute("confidence", confidence)

            # Set output data
            self._opik_span.set_output(output_data)

        logger.info(
            f"Gap analysis complete: status={status}, "
            f"{gaps_detected} gaps, {opportunities_count} opportunities, "
            f"${total_addressable_value:,.0f} addressable, "
            f"confidence={confidence:.2f}, "
            f"duration={total_duration_ms}ms"
        )


class GapAnalyzerOpikTracer:
    """Opik tracer for Gap Analyzer 4-node pipeline.

    Provides observability into the gap analysis pipeline with:
    - Root trace for full analysis
    - Child spans for each pipeline node
    - Gap detection and ROI metrics
    - Opportunity prioritization tracking

    Uses the shared OpikConnector for circuit breaker protection.

    Example:
        >>> tracer = GapAnalyzerOpikTracer()
        >>> async with tracer.trace_analysis(query="...", brand="Kisqali") as trace:
        ...     async with trace.trace_node("gap_detector") as node:
        ...         result = await detect_gaps(...)
        ...         node.log_gap_detection(len(result), ...)
        ...     # ... other nodes
        ...     trace.log_analysis_complete(...)
    """

    def __init__(
        self,
        project_name: str = "e2i-gap-analyzer",
        enabled: bool = True,
        sample_rate: float = 1.0,
    ):
        """Initialize the Gap Analyzer tracer.

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
            logger.debug("GapAnalyzerOpikTracer initialized")
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
        brand: str,
        metrics: Optional[List[str]] = None,
        segments: Optional[List[str]] = None,
        gap_type: str = "vs_potential",
        query_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing a full gap analysis.

        Creates an Opik trace for the analysis and provides a context
        object for creating node spans and logging metrics.

        Args:
            query: The query being analyzed
            brand: Brand being analyzed
            metrics: Metrics being analyzed
            segments: Segments being analyzed
            gap_type: Type of gap analysis
            metadata: Additional trace metadata

        Yields:
            GapAnalysisTraceContext for node tracing and metric logging

        Example:
            async with tracer.trace_analysis(query, brand="Kisqali") as trace:
                async with trace.trace_node("gap_detector") as node:
                    # ... detection logic
                    node.log_gap_detection(...)
        """
        self._ensure_initialized()

        trace_id = str(uuid7_func())
        span_id = str(uuid7_func())
        start_time = datetime.now(timezone.utc)

        # Build metadata
        trace_metadata = {
            "query_length": len(query),
            "brand": brand,
            "gap_type": gap_type,
            "query_id": query_id,
            "metrics_count": len(metrics) if metrics else 0,
            "segments_count": len(segments) if segments else 0,
            **(metadata or {}),
        }

        # Create trace context
        trace_ctx = GapAnalysisTraceContext(
            trace_id=trace_id,
            span_id=span_id,
            query=query,
            brand=brand,
            start_time=start_time,
            metadata=trace_metadata,
            _tracer=self,
        )

        try:
            # Create Opik trace if enabled and sampled
            if self.is_enabled and self._should_trace():
                try:
                    assert self._opik_connector is not None
                    async with (
                        self._opik_connector.trace_agent(
                            agent_name="gap_analyzer",
                            operation="analyze",
                            trace_id=trace_id,
                            metadata={
                                "pipeline": AGENT_METADATA["pipeline"],
                                "tier": AGENT_METADATA["tier"],
                                "agent_type": AGENT_METADATA["type"],
                                **trace_metadata,
                            },
                            tags=["gap_analyzer", "analysis", "tier2", brand],
                            input_data={
                                "query": query[:500],  # Truncate for Opik
                                "brand": brand,
                                "gap_type": gap_type,
                                "metrics": metrics[:10] if metrics else [],
                                "segments": segments[:10] if segments else [],
                            },
                            force_new_trace=True,
                        ) as span
                    ):
                        trace_ctx._opik_span = span
                        yield trace_ctx
                        return
                except Exception as e:
                    logger.debug(f"Opik tracing failed, continuing without: {e}")

            # Fall through to non-traced version
            yield trace_ctx

        except Exception as e:
            {"type": type(e).__name__, "message": str(e)}
            raise

        finally:
            # Record final timing
            end_time = datetime.now(timezone.utc)
            trace_ctx.end_time = end_time
            trace_ctx.duration_ms = (end_time - start_time).total_seconds() * 1000

            logger.debug(f"Gap analysis trace completed in {trace_ctx.duration_ms:.2f}ms")


def trace_gap_analysis(
    query_param: str = "query",
    brand_param: str = "brand",
    project_name: str = "e2i-gap-analyzer",
) -> Callable[[F], F]:
    """Decorator to trace a gap analysis function.

    Use this decorator on analysis functions for automatic tracing.

    Args:
        query_param: Name of the query parameter in the decorated function
        brand_param: Name of the brand parameter
        project_name: Opik project name

    Returns:
        Decorated function

    Example:
        >>> @trace_gap_analysis()
        ... async def analyze(query: str, brand: str, **kwargs):
        ...     # ... analysis logic
        ...     return result
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract query and brand from kwargs or args
            query = kwargs.get(query_param, args[0] if args else "")
            brand = kwargs.get(brand_param, args[1] if len(args) > 1 else "unknown")

            tracer = GapAnalyzerOpikTracer(project_name=project_name)

            async with tracer.trace_analysis(
                query=query,
                brand=brand,
            ) as trace_ctx:
                # Add trace context to kwargs for function to use
                kwargs["_trace_ctx"] = trace_ctx
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# Singleton tracer instance
_tracer_instance: Optional[GapAnalyzerOpikTracer] = None


def get_gap_analyzer_tracer(
    project_name: str = "e2i-gap-analyzer",
    enabled: bool = True,
    sample_rate: float = 1.0,
) -> GapAnalyzerOpikTracer:
    """Get the Gap Analyzer Opik tracer singleton.

    Args:
        project_name: Opik project name
        enabled: Whether tracing is enabled
        sample_rate: Sample rate for tracing

    Returns:
        GapAnalyzerOpikTracer instance
    """
    global _tracer_instance

    if _tracer_instance is None:
        _tracer_instance = GapAnalyzerOpikTracer(
            project_name=project_name,
            enabled=enabled,
            sample_rate=sample_rate,
        )

    return _tracer_instance


def reset_gap_analyzer_tracer() -> None:
    """Reset the tracer singleton (for testing)."""
    global _tracer_instance
    _tracer_instance = None


__all__ = [
    "NodeSpanContext",
    "GapAnalysisTraceContext",
    "GapAnalyzerOpikTracer",
    "trace_gap_analysis",
    "get_gap_analyzer_tracer",
    "reset_gap_analyzer_tracer",
    "AGENT_METADATA",
]
