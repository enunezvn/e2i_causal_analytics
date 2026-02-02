"""GEPA Opik Integration for Optimization Tracing.

This module provides Opik integration for GEPA optimization runs,
enabling observability of the evolutionary prompt optimization process.

Integrates with:
- E2I Opik connector (src/mlops/opik_connector.py)
- GEPA optimizer lifecycle
- Agent optimization experiments

Usage:
    from src.optimization.gepa.integration import GEPAOpikTracer, trace_optimization

    # Use as tracer during optimization
    tracer = GEPAOpikTracer(project_name="gepa_causal_impact")
    async with tracer.trace_run(agent_name="causal_impact", budget="medium") as ctx:
        # optimization runs here
        ctx.log_generation(gen_num, best_score, candidates)
        ctx.log_optimization_complete(best_score, total_gens, total_calls, elapsed)

    # Or use decorator for simple tracing
    @trace_optimization(agent_name="causal_impact")
    async def run_optimization():
        ...

Author: E2I Causal Analytics Team
Version: 4.3.0 (Updated to use OpikConnector trace_agent API)
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

if TYPE_CHECKING:
    from src.mlops.opik_connector import OpikConnector

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class GEPASpanContext:
    """Context for a GEPA optimization span.

    Provides methods to log generation events and optimization results
    within the context of a traced optimization run.

    Attributes:
        trace_id: Unique trace identifier
        span_id: Current span identifier
        parent_span_id: Parent span (if nested)
        agent_name: Name of agent being optimized
        generation: Current generation number (if applicable)
        metadata: Additional span metadata
        _opik_span: Reference to the OpikConnector SpanContext
        _tracer: Reference to the parent GEPAOpikTracer
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    agent_name: Optional[str] = None
    generation: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _opik_span: Optional[Any] = None  # SpanContext from OpikConnector
    _tracer: Optional["GEPAOpikTracer"] = None
    _generation_events: list[dict[str, Any]] = field(default_factory=list)

    def log_generation(
        self,
        generation: int,
        best_score: float,
        pareto_size: int = 0,
        candidate_count: int = 0,
        metric_calls: int = 0,
        elapsed_seconds: float = 0.0,
        candidates: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Log a generation completion event.

        Args:
            generation: Generation number (0-indexed)
            best_score: Best score achieved so far
            pareto_size: Number of candidates on Pareto frontier
            candidate_count: Total candidates evaluated
            metric_calls: Total metric calls so far
            elapsed_seconds: Time elapsed since start
            candidates: Optional list of candidate details
            **kwargs: Additional metrics
        """
        event_data = {
            "generation": generation,
            "best_score": best_score,
            "pareto_size": pareto_size,
            "candidate_count": candidate_count,
            "metric_calls": metric_calls,
            "elapsed_seconds": elapsed_seconds,
            **kwargs,
        }

        # Include top candidates if available
        if candidates and self._tracer and self._tracer.log_candidates:
            event_data["top_candidates"] = candidates[:5]

        self._generation_events.append(event_data)
        self.generation = generation

        # Update the Opik span with generation event
        if self._opik_span:
            self._opik_span.add_event(
                f"generation_{generation}",
                {
                    "best_score": best_score,
                    "metric_calls": metric_calls,
                    "elapsed_seconds": elapsed_seconds,
                },
            )
            self._opik_span.set_attribute(f"gen_{generation}_score", best_score)

        logger.debug(f"Logged GEPA generation {generation}: score={best_score:.4f}")

    def log_candidate_evaluation(
        self,
        generation: int,
        candidate_id: str,
        score: float,
        feedback: str,
        instructions: Optional[str] = None,
        tool_descriptions: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Log an individual candidate evaluation.

        Args:
            generation: Generation number
            candidate_id: Unique candidate identifier
            score: Evaluation score
            feedback: Textual feedback from metric
            instructions: Candidate's instruction text
            tool_descriptions: Candidate's tool descriptions
            **kwargs: Additional metadata
        """
        if self._opik_span:
            event_attrs = {
                "candidate_id": candidate_id[:16],
                "score": score,
                "feedback_preview": feedback[:200] if feedback else "",
            }

            if tool_descriptions:
                event_attrs["tool_count"] = len(tool_descriptions)

            self._opik_span.add_event(f"candidate_{candidate_id[:8]}", event_attrs)

    def log_optimization_complete(
        self,
        best_score: float,
        total_generations: int,
        total_metric_calls: int,
        total_seconds: float,
        optimized_instructions: Optional[str] = None,
        pareto_frontier_size: int = 0,
        **kwargs: Any,
    ) -> None:
        """Log optimization completion metrics.

        Args:
            best_score: Final best score
            total_generations: Total generations run
            total_metric_calls: Total metric evaluations
            total_seconds: Total time in seconds
            optimized_instructions: Final optimized instructions
            pareto_frontier_size: Size of final Pareto frontier
            **kwargs: Additional final metrics
        """
        if self._opik_span:
            # Set final output data
            output_data = {
                "status": "completed",
                "best_score": best_score,
                "total_generations": total_generations,
                "total_metric_calls": total_metric_calls,
                "total_seconds": total_seconds,
                "pareto_frontier_size": pareto_frontier_size,
                "improvement_rate": (
                    best_score / total_metric_calls if total_metric_calls > 0 else 0
                ),
                "generation_history": self._generation_events,
                **kwargs,
            }

            if optimized_instructions and self._tracer and self._tracer.log_instructions:
                output_data["optimized_instructions_preview"] = optimized_instructions[:500]

            self._opik_span.set_output(output_data)

            # Set key attributes for quick filtering
            self._opik_span.set_attribute("final_best_score", best_score)
            self._opik_span.set_attribute("total_generations", total_generations)
            self._opik_span.set_attribute("total_metric_calls", total_metric_calls)

        logger.info(
            f"GEPA optimization complete: score={best_score:.4f}, "
            f"generations={total_generations}, time={total_seconds:.1f}s"
        )


@dataclass
class GEPAOpikTracer:
    """Opik tracer for GEPA optimization runs.

    Provides observability into the GEPA evolutionary optimization process,
    using the OpikConnector's trace_agent() context manager for tracing.

    Example:
        >>> tracer = GEPAOpikTracer(project_name="gepa_optimization")
        >>> async with tracer.trace_run(agent_name="causal_impact", budget="medium") as ctx:
        ...     # Optimization proceeds
        ...     ctx.log_generation(0, best_score=0.7, metric_calls=10)
        ...     ctx.log_generation(1, best_score=0.85, metric_calls=25)
        ...     ctx.log_optimization_complete(best_score=0.85, total_generations=2, ...)
    """

    project_name: str = "gepa_optimization"
    tags: dict[str, str] = field(default_factory=dict)
    log_candidates: bool = True
    log_instructions: bool = True
    sample_rate: float = 1.0  # 1.0 = trace all, 0.1 = trace 10%

    # Internal state
    _opik_connector: Optional["OpikConnector"] = None

    def __post_init__(self) -> None:
        """Initialize Opik connector."""
        try:
            from src.mlops.opik_connector import get_opik_connector

            self._opik_connector = get_opik_connector()
            logger.debug("GEPAOpikTracer initialized with Opik connector")
        except ImportError:
            logger.warning("Opik connector not available")
            self._opik_connector = None

    @property
    def enabled(self) -> bool:
        """Check if Opik tracing is enabled."""
        return self._opik_connector is not None and self._opik_connector.is_enabled

    def _should_trace(self) -> bool:
        """Determine if this run should be traced based on sample rate."""
        import random

        return random.random() < self.sample_rate

    @asynccontextmanager
    async def trace_run(
        self,
        agent_name: str,
        budget: str,
        max_metric_calls: int = 0,
        enable_tool_optimization: bool = False,
        **kwargs: Any,
    ):
        """Context manager for tracing a full GEPA optimization run.

        Uses OpikConnector's trace_agent() for actual tracing.

        Args:
            agent_name: Name of the agent being optimized
            budget: Budget preset (light, medium, heavy)
            max_metric_calls: Maximum metric evaluations
            enable_tool_optimization: Whether tool optimization is enabled
            **kwargs: Additional metadata

        Yields:
            GEPASpanContext for logging optimization events
        """
        if not self.enabled or not self._should_trace():
            # Yield dummy context when tracing disabled
            yield GEPASpanContext(
                trace_id="disabled",
                span_id="disabled",
                agent_name=agent_name,
                _tracer=self,
            )
            return

        try:
            # Use OpikConnector's trace_agent context manager
            async with self._opik_connector.trace_agent(
                agent_name=f"gepa_{agent_name}",
                operation="optimization",
                metadata={
                    "optimizer": "gepa",
                    "budget": budget,
                    "max_metric_calls": max_metric_calls,
                    "enable_tool_optimization": enable_tool_optimization,
                    "project_name": self.project_name,
                    **self.tags,
                    **kwargs,
                },
                tags=["gepa", agent_name, budget],
                input_data={
                    "agent_name": agent_name,
                    "budget": budget,
                    "max_metric_calls": max_metric_calls,
                    "enable_tool_optimization": enable_tool_optimization,
                },
            ) as span:
                # Create context with reference to span
                context = GEPASpanContext(
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                    agent_name=agent_name,
                    metadata={
                        "budget": budget,
                        "max_metric_calls": max_metric_calls,
                    },
                    _opik_span=span,
                    _tracer=self,
                )

                logger.info(f"Started GEPA Opik trace: {context.trace_id}")
                yield context

        except Exception as e:
            logger.error(f"Error in GEPA Opik tracing: {e}")
            # Yield dummy context on error
            yield GEPASpanContext(
                trace_id="error",
                span_id="error",
                agent_name=agent_name,
                _tracer=self,
            )


def trace_optimization(
    agent_name: str,
    budget: str = "medium",
    project_name: str = "gepa_optimization",
    **tracer_kwargs: Any,
) -> Callable[[F], F]:
    """Decorator to trace a GEPA optimization function.

    Use this decorator on optimization functions for automatic tracing.
    The decorated function receives a GEPASpanContext as the first argument
    for logging optimization events.

    Args:
        agent_name: Name of the agent being optimized
        budget: Budget preset
        project_name: Opik project name
        **tracer_kwargs: Additional tracer configuration

    Returns:
        Decorated function

    Example:
        >>> @trace_optimization(agent_name="causal_impact", budget="medium")
        ... async def optimize_causal_impact(ctx, trainset, valset):
        ...     optimizer = create_gepa_optimizer(...)
        ...     result = optimizer.compile(student, trainset)
        ...     ctx.log_optimization_complete(best_score=0.9, ...)
        ...     return result
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = GEPAOpikTracer(
                project_name=project_name,
                **tracer_kwargs,
            )

            async with tracer.trace_run(
                agent_name=agent_name,
                budget=budget,
            ) as ctx:
                # Pass context as first argument
                return await func(ctx, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator


__all__ = [
    "GEPASpanContext",
    "GEPAOpikTracer",
    "trace_optimization",
]
