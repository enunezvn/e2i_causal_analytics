"""
Opik Integration for Causal Impact Agent.

This module provides Opik tracing utilities for the Causal Impact 5-node pipeline,
enabling observability of:
- Full causal analysis traces
- Node-level spans (graph_builder → estimation → refutation → sensitivity → interpretation)
- Graph construction metrics
- Effect estimation tracking
- Refutation test results
- Sensitivity analysis metrics

Usage:
    from src.agents.causal_impact.opik_tracer import (
        CausalImpactOpikTracer,
        get_causal_impact_tracer,
    )

    tracer = get_causal_impact_tracer()

    # Trace a full causal analysis
    async with tracer.trace_analysis(query="...", treatment_var="X", outcome_var="Y") as trace:
        async with trace.trace_node("graph_builder") as node:
            graph = await build_graph(...)
            node.log_graph_construction(graph)
        async with trace.trace_node("estimation") as node:
            result = await estimate_effect(...)
            node.log_estimation(result)
        # ... other nodes
        trace.log_analysis_complete(result)

Author: E2I Causal Analytics Team
Version: 4.8.0
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar, cast

from uuid_utils import uuid7 as uuid7_func

if TYPE_CHECKING:
    from src.mlops.opik_connector import OpikConnector, SpanContext

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# Agent metadata
AGENT_METADATA = {
    "name": "causal_impact",
    "tier": 2,
    "type": "hybrid",
    "pipeline": "graph_builder → estimation → refutation → sensitivity → interpretation",
}


@dataclass
class NodeSpanContext:
    """Context for a Causal Impact pipeline node span.

    Provides methods to log node-specific events and metrics.

    Attributes:
        trace_id: Parent trace identifier
        span_id: This span's identifier
        node_name: Name of the node (graph_builder, estimation, refutation, sensitivity, interpretation)
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
    _opik_span: Optional["SpanContext"] = None

    def log_graph_construction(
        self,
        num_nodes: int,
        num_edges: int,
        confidence: float,
        adjustment_sets: Optional[int] = None,
        discovery_enabled: bool = False,
        discovery_algorithms: Optional[List[str]] = None,
    ) -> None:
        """Log graph construction metrics.

        Args:
            num_nodes: Number of nodes in the DAG
            num_edges: Number of edges in the DAG
            confidence: Graph construction confidence (0-1)
            adjustment_sets: Number of valid adjustment sets
            discovery_enabled: Whether causal discovery was used
            discovery_algorithms: Algorithms used for discovery
        """
        self.metadata.update(
            {
                "graph_nodes": num_nodes,
                "graph_edges": num_edges,
                "graph_confidence": confidence,
                "adjustment_sets": adjustment_sets,
                "discovery_enabled": discovery_enabled,
                "discovery_algorithms": discovery_algorithms or [],
            }
        )
        logger.debug(
            f"Graph construction: {num_nodes} nodes, {num_edges} edges, "
            f"confidence={confidence:.3f}, discovery={discovery_enabled}"
        )

    def log_estimation(
        self,
        ate: float,
        ci_lower: float,
        ci_upper: float,
        method: str,
        sample_size: int,
        energy_score: Optional[float] = None,
        selection_strategy: Optional[str] = None,
        n_estimators_evaluated: Optional[int] = None,
    ) -> None:
        """Log causal effect estimation metrics.

        Args:
            ate: Average Treatment Effect
            ci_lower: 95% CI lower bound
            ci_upper: 95% CI upper bound
            method: Estimation method used
            sample_size: Number of samples
            energy_score: Energy score for estimator selection (0-1)
            selection_strategy: How estimator was selected
            n_estimators_evaluated: Number of estimators tried
        """
        self.metadata.update(
            {
                "ate": ate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_width": ci_upper - ci_lower,
                "method": method,
                "sample_size": sample_size,
                "energy_score": energy_score,
                "selection_strategy": selection_strategy,
                "n_estimators_evaluated": n_estimators_evaluated,
            }
        )
        logger.debug(
            f"Estimation: ATE={ate:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], "
            f"method={method}, energy_score={energy_score}"
        )

    def log_refutation(
        self,
        tests_passed: int,
        tests_total: int,
        refutation_rate: float,
        individual_tests: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Log refutation test results.

        Args:
            tests_passed: Number of tests passed
            tests_total: Total number of tests run
            refutation_rate: Pass rate (0-1)
            individual_tests: Results for each test type
        """
        self.metadata.update(
            {
                "refutation_tests_passed": tests_passed,
                "refutation_tests_total": tests_total,
                "refutation_rate": refutation_rate,
                "individual_tests": individual_tests or {},
            }
        )
        logger.debug(f"Refutation: {tests_passed}/{tests_total} passed ({refutation_rate:.1%})")

    def log_sensitivity(
        self,
        e_value: Optional[float] = None,
        robustness_score: Optional[float] = None,
        sensitivity_passed: bool = True,
        unmeasured_confounding_threshold: Optional[float] = None,
    ) -> None:
        """Log sensitivity analysis results.

        Args:
            e_value: E-value for unmeasured confounding
            robustness_score: Overall robustness score (0-1)
            sensitivity_passed: Whether analysis passed thresholds
            unmeasured_confounding_threshold: Threshold for confounding detection
        """
        self.metadata.update(
            {
                "e_value": e_value,
                "robustness_score": robustness_score,
                "sensitivity_passed": sensitivity_passed,
                "unmeasured_confounding_threshold": unmeasured_confounding_threshold,
            }
        )
        logger.debug(
            f"Sensitivity: e_value={e_value}, robustness={robustness_score}, passed={sensitivity_passed}"
        )

    def log_interpretation(
        self,
        summary_generated: bool = True,
        summary_length: Optional[int] = None,
        confidence_level: Optional[str] = None,
    ) -> None:
        """Log interpretation/summary generation.

        Args:
            summary_generated: Whether NL summary was generated
            summary_length: Length of generated summary
            confidence_level: Confidence level (high/medium/low)
        """
        self.metadata.update(
            {
                "summary_generated": summary_generated,
                "summary_length": summary_length,
                "confidence_level": confidence_level,
            }
        )
        logger.debug(f"Interpretation: generated={summary_generated}, length={summary_length}")

    def set_error(self, error: str, error_type: Optional[str] = None) -> None:
        """Mark this node as having an error.

        Args:
            error: Error message
            error_type: Type of error (e.g., 'estimation_error', 'refutation_error')
        """
        self.metadata["error"] = error
        self.metadata["error_type"] = error_type or "unknown_error"
        self.metadata["status"] = "error"
        logger.warning(f"Node {self.node_name} error: {error}")


@dataclass
class AnalysisTraceContext:
    """Context for a full Causal Impact analysis trace.

    Manages the parent trace and provides methods for creating child node spans.

    Attributes:
        trace_id: Unique trace identifier
        query: The analysis query
        treatment_var: Treatment variable
        outcome_var: Outcome variable
        brand: Brand context
        start_time: When the analysis started
        node_spans: Child node span contexts
        _tracer: Reference to the parent tracer
    """

    trace_id: str
    query: str
    treatment_var: str
    outcome_var: str
    brand: Optional[str] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    node_spans: List[NodeSpanContext] = field(default_factory=list)
    _tracer: Optional["CausalImpactOpikTracer"] = None
    _opik_span: Optional["SpanContext"] = None

    @asynccontextmanager
    async def trace_node(self, node_name: str):
        """Create a span for a pipeline node.

        Args:
            node_name: Name of the node (graph_builder, estimation, refutation, sensitivity, interpretation)

        Yields:
            NodeSpanContext for logging node-specific metrics
        """
        span_id = str(uuid7_func())
        node_ctx = NodeSpanContext(
            trace_id=self.trace_id,
            span_id=span_id,
            node_name=node_name,
        )

        try:
            # If we have an Opik tracer, create a child span
            if self._tracer and self._tracer._opik:
                async with self._tracer._opik.trace_agent(
                    agent_name="causal_impact",
                    operation=node_name,
                    trace_id=self.trace_id,
                    parent_span_id=self._opik_span.span_id if self._opik_span else None,
                    metadata={
                        "node_name": node_name,
                        "pipeline_position": _get_pipeline_position(node_name),
                        "treatment_var": self.treatment_var,
                        "outcome_var": self.outcome_var,
                    },
                    tags=["causal_impact", node_name, "pipeline_node"],
                ) as span:
                    node_ctx._opik_span = span
                    yield node_ctx
            else:
                yield node_ctx

        except Exception as e:
            node_ctx.set_error(str(e), f"{node_name}_error")
            raise
        finally:
            node_ctx.end_time = datetime.now(timezone.utc)
            if node_ctx.start_time:
                node_ctx.duration_ms = (
                    node_ctx.end_time - node_ctx.start_time
                ).total_seconds() * 1000
            self.node_spans.append(node_ctx)

    def log_analysis_complete(
        self,
        success: bool = True,
        ate: Optional[float] = None,
        refutation_rate: Optional[float] = None,
        confidence_score: Optional[float] = None,
        final_status: str = "completed",
    ) -> None:
        """Log completion of the full causal analysis.

        Args:
            success: Whether analysis completed successfully
            ate: Final ATE estimate
            refutation_rate: Overall refutation pass rate
            confidence_score: Overall confidence in results
            final_status: Status string (completed, partial, failed)
        """
        end_time = datetime.now(timezone.utc)
        total_duration_ms = (end_time - self.start_time).total_seconds() * 1000

        completion_data = {
            "success": success,
            "final_status": final_status,
            "ate": ate,
            "refutation_rate": refutation_rate,
            "confidence_score": confidence_score,
            "total_duration_ms": total_duration_ms,
            "nodes_executed": len(self.node_spans),
            "node_durations_ms": {
                span.node_name: span.duration_ms
                for span in self.node_spans
                if span.duration_ms is not None
            },
        }

        logger.info(
            f"Causal analysis complete: success={success}, ATE={ate}, "
            f"refutation_rate={refutation_rate}, duration={total_duration_ms:.0f}ms"
        )

        # Update Opik span if available
        if self._opik_span:
            try:
                self._opik_span.set_output(completion_data)
            except Exception as e:
                logger.debug(f"Could not set Opik output: {e}")


def _get_pipeline_position(node_name: str) -> int:
    """Get the position of a node in the pipeline (1-indexed).

    Args:
        node_name: Name of the node

    Returns:
        Position in pipeline (1-5)
    """
    positions = {
        "graph_builder": 1,
        "estimation": 2,
        "refutation": 3,
        "sensitivity": 4,
        "interpretation": 5,
    }
    return positions.get(node_name, 0)


class CausalImpactOpikTracer:
    """Opik tracer for Causal Impact Agent.

    Provides structured tracing for the 5-node causal analysis pipeline with
    domain-specific metrics for causal inference workflows.

    Attributes:
        project_name: Opik project name
        _opik: Lazy-loaded OpikConnector instance
    """

    def __init__(self, project_name: str = "e2i_causal_analytics"):
        """Initialize the tracer.

        Args:
            project_name: Opik project name for grouping traces
        """
        self.project_name = project_name
        self._opik: Optional["OpikConnector"] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-load Opik connector on first use."""
        if not self._initialized:
            try:
                from src.mlops.opik_connector import get_opik_connector

                self._opik = get_opik_connector()
                self._initialized = True
            except Exception as e:
                logger.debug(f"Opik connector not available: {e}")
                self._opik = None
                self._initialized = True

    @property
    def is_enabled(self) -> bool:
        """Check if Opik tracing is enabled and available."""
        self._ensure_initialized()
        return self._opik is not None and self._opik.is_enabled

    @asynccontextmanager
    async def trace_analysis(
        self,
        query: str,
        treatment_var: str,
        outcome_var: str,
        brand: Optional[str] = None,
        session_id: Optional[str] = None,
        dispatch_id: Optional[str] = None,
    ):
        """Create a trace for a full causal analysis.

        Args:
            query: The analysis query
            treatment_var: Treatment variable name
            outcome_var: Outcome variable name
            brand: Optional brand context
            session_id: Optional session identifier
            dispatch_id: Optional dispatch identifier

        Yields:
            AnalysisTraceContext for managing node spans
        """
        self._ensure_initialized()
        trace_id = str(uuid7_func())

        trace_ctx = AnalysisTraceContext(
            trace_id=trace_id,
            query=query,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            brand=brand,
            _tracer=self,
        )

        try:
            if self._opik and self._opik.is_enabled:
                async with self._opik.trace_agent(
                    agent_name="causal_impact",
                    operation="full_analysis",
                    trace_id=trace_id,
                    metadata={
                        "treatment_var": treatment_var,
                        "outcome_var": outcome_var,
                        "brand": brand,
                        "session_id": session_id,
                        "dispatch_id": dispatch_id,
                        "pipeline": AGENT_METADATA["pipeline"],
                    },
                    tags=["causal_impact", "full_analysis", "tier_2"],
                    input_data={"query": query[:500] if query else None},  # Truncate long queries
                ) as span:
                    trace_ctx._opik_span = span
                    yield trace_ctx
            else:
                yield trace_ctx

        except Exception as e:
            logger.error(f"Causal analysis trace error: {e}")
            raise

    def trace_node_decorator(self, node_name: str) -> Callable[[F], F]:
        """Decorator to add Opik tracing to a workflow node function.

        This provides a simpler alternative to using trace_analysis context manager
        when nodes are defined as standalone functions.

        Args:
            node_name: Name of the node being traced

        Returns:
            Decorated function with Opik tracing
        """

        def decorator(func: F) -> F:
            @wraps(func)
            async def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
                self._ensure_initialized()

                trace_id = state.get("query_id") or state.get("trace_id")
                parent_span_id = state.get("span_id")

                metadata = {
                    "node_name": node_name,
                    "agent_name": "causal_impact",
                    "pipeline_position": _get_pipeline_position(node_name),
                    "treatment_var": state.get("treatment_var"),
                    "outcome_var": state.get("outcome_var"),
                }

                if self._opik and self._opik.is_enabled:
                    async with self._opik.trace_agent(
                        agent_name="causal_impact",
                        operation=node_name,
                        trace_id=trace_id,
                        parent_span_id=parent_span_id,
                        metadata=metadata,
                        tags=["causal_impact", node_name, "workflow_node"],
                        input_data={
                            "query": state.get("query", "")[:200],
                            "current_phase": state.get("current_phase"),
                        },
                    ) as span:
                        result = await func(state)
                        # Set output summary
                        if span:
                            span.set_output(
                                {
                                    "current_phase": result.get("current_phase"),
                                    "status": result.get("status"),
                                    "has_error": bool(result.get(f"{node_name}_error")),
                                }
                            )
                        return cast(Dict[str, Any], result)
                else:
                    return cast(Dict[str, Any], await func(state))

            return wrapper  # type: ignore

        return decorator


# Module-level singleton instance
_tracer_instance: Optional[CausalImpactOpikTracer] = None


def get_causal_impact_tracer() -> CausalImpactOpikTracer:
    """Get the singleton Causal Impact Opik tracer instance.

    Returns:
        CausalImpactOpikTracer instance
    """
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = CausalImpactOpikTracer()
    return _tracer_instance


def reset_tracer() -> None:
    """Reset the singleton tracer instance (for testing)."""
    global _tracer_instance
    _tracer_instance = None


# Export convenience aliases
__all__ = [
    "CausalImpactOpikTracer",
    "AnalysisTraceContext",
    "NodeSpanContext",
    "get_causal_impact_tracer",
    "reset_tracer",
    "AGENT_METADATA",
]
