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
    async with tracer.trace_run(agent_name="causal_impact", budget="medium"):
        # optimization runs here
        tracer.log_generation(gen_num, candidates, scores)

    # Or use decorator for simple tracing
    @trace_optimization(agent_name="causal_impact")
    async def run_optimization():
        ...

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class GEPASpanContext:
    """Context for a GEPA optimization span.

    Attributes:
        trace_id: Unique trace identifier
        span_id: Current span identifier
        parent_span_id: Parent span (if nested)
        agent_name: Name of agent being optimized
        generation: Current generation number (if applicable)
        metadata: Additional span metadata
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    agent_name: Optional[str] = None
    generation: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GEPAOpikTracer:
    """Opik tracer for GEPA optimization runs.

    Provides observability into the GEPA evolutionary optimization process,
    tracing:
    - Full optimization runs
    - Individual generations
    - Candidate evaluations
    - Mutation operations

    Example:
        >>> tracer = GEPAOpikTracer(project_name="gepa_optimization")
        >>> async with tracer.trace_run(agent_name="causal_impact", budget="medium"):
        ...     # Optimization proceeds
        ...     tracer.log_generation(0, candidates, scores)
        ...     tracer.log_generation(1, candidates, scores)
    """

    project_name: str = "gepa_optimization"
    tags: dict[str, str] = field(default_factory=dict)
    log_candidates: bool = True
    log_instructions: bool = True
    sample_rate: float = 1.0  # 1.0 = trace all, 0.1 = trace 10%

    # Internal state
    _opik_connector: Any = None
    _current_trace_id: Optional[str] = None
    _current_span_id: Optional[str] = None
    _generation_spans: dict[int, str] = field(default_factory=dict)

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
        return self._opik_connector is not None and self._opik_connector.enabled

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

        Args:
            agent_name: Name of the agent being optimized
            budget: Budget preset (light, medium, heavy)
            max_metric_calls: Maximum metric evaluations
            enable_tool_optimization: Whether tool optimization is enabled
            **kwargs: Additional metadata

        Yields:
            GEPASpanContext for the optimization run
        """
        if not self.enabled or not self._should_trace():
            # Yield dummy context when tracing disabled
            yield GEPASpanContext(
                trace_id="disabled",
                span_id="disabled",
                agent_name=agent_name,
            )
            return

        try:
            # Generate trace and span IDs
            self._current_trace_id = str(uuid4())
            self._current_span_id = str(uuid4())

            # Start trace with Opik
            async with self._opik_connector.start_trace(
                name=f"gepa_optimization_{agent_name}",
                project_name=self.project_name,
                tags={
                    "agent_name": agent_name,
                    "budget": budget,
                    "tool_optimization": str(enable_tool_optimization),
                    "optimizer": "gepa",
                    **self.tags,
                },
                metadata={
                    "max_metric_calls": max_metric_calls,
                    "enable_tool_optimization": enable_tool_optimization,
                    **kwargs,
                },
            ) as trace:
                context = GEPASpanContext(
                    trace_id=trace.trace_id if hasattr(trace, "trace_id") else self._current_trace_id,
                    span_id=trace.span_id if hasattr(trace, "span_id") else self._current_span_id,
                    agent_name=agent_name,
                    metadata={
                        "budget": budget,
                        "max_metric_calls": max_metric_calls,
                    },
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
            )
        finally:
            self._current_trace_id = None
            self._current_span_id = None
            self._generation_spans.clear()

    async def log_generation(
        self,
        generation: int,
        best_score: float,
        pareto_size: int,
        candidate_count: int,
        metric_calls: int,
        elapsed_seconds: float,
        candidates: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[str]:
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

        Returns:
            Span ID if logged, None otherwise
        """
        if not self.enabled or not self._current_trace_id:
            return None

        try:
            span_id = str(uuid4())
            self._generation_spans[generation] = span_id

            # Build span metadata
            metadata = {
                "generation": generation,
                "best_score": best_score,
                "pareto_size": pareto_size,
                "candidate_count": candidate_count,
                "metric_calls": metric_calls,
                "elapsed_seconds": elapsed_seconds,
                **kwargs,
            }

            # Optionally include candidate details
            if self.log_candidates and candidates:
                # Limit to top candidates to avoid bloat
                metadata["top_candidates"] = candidates[:5]

            # Log span via Opik connector
            await self._opik_connector.log_span(
                trace_id=self._current_trace_id,
                name=f"generation_{generation}",
                span_type="generation",
                metadata=metadata,
            )

            logger.debug(f"Logged GEPA generation {generation}: score={best_score:.4f}")
            return span_id

        except Exception as e:
            logger.warning(f"Failed to log generation: {e}")
            return None

    async def log_candidate_evaluation(
        self,
        generation: int,
        candidate_id: str,
        score: float,
        feedback: str,
        instructions: Optional[str] = None,
        tool_descriptions: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Log an individual candidate evaluation.

        Args:
            generation: Generation number
            candidate_id: Unique candidate identifier
            score: Evaluation score
            feedback: Textual feedback from metric
            instructions: Candidate's instruction text
            tool_descriptions: Candidate's tool descriptions
            **kwargs: Additional metadata

        Returns:
            Span ID if logged, None otherwise
        """
        if not self.enabled or not self._current_trace_id:
            return None

        try:
            parent_span = self._generation_spans.get(generation)

            metadata = {
                "generation": generation,
                "candidate_id": candidate_id,
                "score": score,
                "feedback": feedback,
                **kwargs,
            }

            if self.log_instructions and instructions:
                # Truncate long instructions
                metadata["instructions"] = instructions[:2000] if len(instructions) > 2000 else instructions

            if tool_descriptions:
                metadata["tool_count"] = len(tool_descriptions)

            await self._opik_connector.log_span(
                trace_id=self._current_trace_id,
                parent_span_id=parent_span,
                name=f"candidate_{candidate_id[:8]}",
                span_type="evaluation",
                metadata=metadata,
            )

            return str(uuid4())

        except Exception as e:
            logger.warning(f"Failed to log candidate evaluation: {e}")
            return None

    async def log_optimization_complete(
        self,
        best_score: float,
        total_generations: int,
        total_metric_calls: int,
        total_seconds: float,
        optimized_instructions: Optional[str] = None,
        pareto_frontier_size: int = 0,
        **kwargs: Any,
    ) -> None:
        """Log optimization completion.

        Args:
            best_score: Final best score
            total_generations: Total generations run
            total_metric_calls: Total metric evaluations
            total_seconds: Total time in seconds
            optimized_instructions: Final optimized instructions
            pareto_frontier_size: Size of final Pareto frontier
            **kwargs: Additional final metrics
        """
        if not self.enabled or not self._current_trace_id:
            return

        try:
            metadata = {
                "status": "completed",
                "best_score": best_score,
                "total_generations": total_generations,
                "total_metric_calls": total_metric_calls,
                "total_seconds": total_seconds,
                "pareto_frontier_size": pareto_frontier_size,
                "improvement_rate": best_score / total_metric_calls if total_metric_calls > 0 else 0,
                **kwargs,
            }

            if self.log_instructions and optimized_instructions:
                metadata["optimized_instructions_preview"] = optimized_instructions[:500]

            await self._opik_connector.end_trace(
                trace_id=self._current_trace_id,
                status="success",
                metadata=metadata,
            )

            logger.info(
                f"GEPA optimization complete: score={best_score:.4f}, "
                f"generations={total_generations}, time={total_seconds:.1f}s"
            )

        except Exception as e:
            logger.error(f"Failed to log optimization completion: {e}")


def trace_optimization(
    agent_name: str,
    budget: str = "medium",
    project_name: str = "gepa_optimization",
    **tracer_kwargs: Any,
) -> Callable[[F], F]:
    """Decorator to trace a GEPA optimization function.

    Use this decorator on optimization functions for automatic tracing.

    Args:
        agent_name: Name of the agent being optimized
        budget: Budget preset
        project_name: Opik project name
        **tracer_kwargs: Additional tracer configuration

    Returns:
        Decorated function

    Example:
        >>> @trace_optimization(agent_name="causal_impact", budget="medium")
        ... async def optimize_causal_impact(trainset, valset):
        ...     optimizer = create_gepa_optimizer(...)
        ...     return optimizer.compile(student, trainset)
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
            ):
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


__all__ = [
    "GEPASpanContext",
    "GEPAOpikTracer",
    "trace_optimization",
]
