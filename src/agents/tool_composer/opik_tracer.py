"""
Opik Integration for Tool Composer Agent.

This module provides Opik tracing utilities for the Tool Composer 4-phase pipeline,
enabling observability of:
- Full composition pipeline traces
- Phase-level spans (decompose → plan → execute → synthesize)
- Tool execution metrics
- Parallel execution tracking

Usage:
    from src.agents.tool_composer.opik_tracer import (
        ToolComposerOpikTracer,
        trace_composition,
    )

    tracer = ToolComposerOpikTracer()

    # Trace a full composition
    async with tracer.trace_composition(query="...") as trace:
        async with trace.trace_phase("decompose") as phase:
            decomposition = await decomposer.decompose(query)
            phase.log_decomposition(decomposition)
        # ... other phases
        trace.log_composition_complete(result)

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
    from src.mlops.opik_connector import OpikConnector, SpanContext

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class PhaseSpanContext:
    """Context for a Tool Composer phase span.

    Provides methods to log phase-specific events and metrics.

    Attributes:
        trace_id: Parent trace identifier
        span_id: This span's identifier
        phase_name: Name of the phase (decompose, plan, execute, synthesize)
        start_time: When the phase started
        end_time: When the phase ended (set on exit)
        duration_ms: Phase duration in milliseconds
        metadata: Additional phase metadata
        _opik_span: Reference to the Opik span context
    """

    trace_id: str
    span_id: str
    phase_name: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _opik_span: Optional[Any] = None
    _parent_ctx: Optional["CompositionTraceContext"] = None

    def log_decomposition(
        self,
        sub_question_count: int,
        intents: List[str],
        extracted_entities: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Log decomposition phase metrics.

        Args:
            sub_question_count: Number of sub-questions generated
            intents: List of intent classifications
            extracted_entities: Entities extracted from query
            **kwargs: Additional metrics
        """
        self.metadata.update({
            "sub_question_count": sub_question_count,
            "intents": intents,
            "extracted_entities": extracted_entities or [],
            **kwargs,
        })

        if self._opik_span:
            self._opik_span.set_attribute("sub_question_count", sub_question_count)
            self._opik_span.set_attribute("intent_count", len(intents))
            self._opik_span.add_event(
                "decomposition_complete",
                {
                    "sub_question_count": sub_question_count,
                    "intents": intents[:5],  # Limit for Opik
                },
            )

        logger.debug(
            f"[DECOMPOSE] {sub_question_count} sub-questions, "
            f"intents: {intents}"
        )

    def log_planning(
        self,
        step_count: int,
        tool_mappings: List[str],
        parallel_groups: int = 0,
        avg_confidence: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Log planning phase metrics.

        Args:
            step_count: Number of execution steps planned
            tool_mappings: List of tools selected
            parallel_groups: Number of parallel execution groups
            avg_confidence: Average tool mapping confidence
            **kwargs: Additional metrics
        """
        self.metadata.update({
            "step_count": step_count,
            "tool_mappings": tool_mappings,
            "parallel_groups": parallel_groups,
            "avg_confidence": avg_confidence,
            **kwargs,
        })

        if self._opik_span:
            self._opik_span.set_attribute("step_count", step_count)
            self._opik_span.set_attribute("parallel_groups", parallel_groups)
            self._opik_span.set_attribute("avg_confidence", avg_confidence)
            self._opik_span.add_event(
                "planning_complete",
                {
                    "step_count": step_count,
                    "tools": tool_mappings[:10],  # Limit
                    "parallel_groups": parallel_groups,
                },
            )

        logger.debug(
            f"[PLAN] {step_count} steps, {parallel_groups} parallel groups, "
            f"confidence: {avg_confidence:.2f}"
        )

    def log_execution(
        self,
        tools_executed: int,
        tools_succeeded: int,
        retry_count: int = 0,
        parallel_executions: int = 0,
        step_durations_ms: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        """Log execution phase metrics.

        Args:
            tools_executed: Total tools executed
            tools_succeeded: Tools that succeeded
            retry_count: Total retries across all steps
            parallel_executions: Number of parallel tool calls
            step_durations_ms: Duration of each step
            **kwargs: Additional metrics
        """
        success_rate = tools_succeeded / tools_executed if tools_executed > 0 else 0.0

        self.metadata.update({
            "tools_executed": tools_executed,
            "tools_succeeded": tools_succeeded,
            "success_rate": success_rate,
            "retry_count": retry_count,
            "parallel_executions": parallel_executions,
            "step_durations_ms": step_durations_ms or [],
            **kwargs,
        })

        if self._opik_span:
            self._opik_span.set_attribute("tools_executed", tools_executed)
            self._opik_span.set_attribute("tools_succeeded", tools_succeeded)
            self._opik_span.set_attribute("success_rate", success_rate)
            self._opik_span.set_attribute("parallel_executions", parallel_executions)
            self._opik_span.add_event(
                "execution_complete",
                {
                    "tools_executed": tools_executed,
                    "tools_succeeded": tools_succeeded,
                    "retry_count": retry_count,
                },
            )

        logger.debug(
            f"[EXECUTE] {tools_succeeded}/{tools_executed} succeeded, "
            f"{retry_count} retries, {parallel_executions} parallel"
        )

    def log_synthesis(
        self,
        answer_length: int,
        confidence: float,
        caveat_count: int = 0,
        failed_components: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Log synthesis phase metrics.

        Args:
            answer_length: Length of synthesized answer
            confidence: Response confidence score
            caveat_count: Number of caveats in response
            failed_components: Components that failed
            **kwargs: Additional metrics
        """
        self.metadata.update({
            "answer_length": answer_length,
            "confidence": confidence,
            "caveat_count": caveat_count,
            "failed_components": failed_components or [],
            **kwargs,
        })

        if self._opik_span:
            self._opik_span.set_attribute("answer_length", answer_length)
            self._opik_span.set_attribute("confidence", confidence)
            self._opik_span.set_attribute("caveat_count", caveat_count)
            self._opik_span.add_event(
                "synthesis_complete",
                {
                    "answer_length": answer_length,
                    "confidence": confidence,
                    "has_failures": len(failed_components or []) > 0,
                },
            )

        logger.debug(
            f"[SYNTHESIZE] {answer_length} chars, confidence: {confidence:.2f}, "
            f"{caveat_count} caveats"
        )

    def set_output(self, output: Dict[str, Any]) -> None:
        """Set the output data for this phase span."""
        if self._opik_span:
            self._opik_span.set_output(output)


@dataclass
class CompositionTraceContext:
    """Context for a full Tool Composer composition trace.

    Provides methods to create phase spans and log overall metrics.

    Attributes:
        trace_id: Unique trace identifier
        span_id: Root span identifier
        query: Original query being composed
        start_time: When composition started
        end_time: When composition ended
        phase_spans: Child spans for each phase
        metadata: Additional trace metadata
        _opik_span: Reference to the Opik span
        _tracer: Reference to parent tracer
    """

    trace_id: str
    span_id: str
    query: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    phase_spans: Dict[str, PhaseSpanContext] = field(default_factory=dict)
    phase_durations: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _opik_span: Optional[Any] = None
    _tracer: Optional["ToolComposerOpikTracer"] = None

    @asynccontextmanager
    async def trace_phase(
        self,
        phase_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing a specific phase.

        Args:
            phase_name: Name of the phase (decompose, plan, execute, synthesize)
            metadata: Additional phase metadata

        Yields:
            PhaseSpanContext for logging phase events
        """
        span_id = str(uuid7_func())
        start_time = datetime.now(timezone.utc)

        phase_ctx = PhaseSpanContext(
            trace_id=self.trace_id,
            span_id=span_id,
            phase_name=phase_name,
            start_time=start_time,
            metadata=metadata or {},
            _parent_ctx=self,
        )

        opik_phase_span = None

        try:
            # Create child span in Opik if parent is traced
            if self._opik_span and self._tracer and self._tracer.enabled:
                try:
                    from src.mlops.opik_connector import get_opik_connector

                    connector = get_opik_connector()
                    if connector.is_enabled:
                        # Use the connector's trace_agent for child span
                        async with connector.trace_agent(
                            agent_name="tool_composer",
                            operation=phase_name,
                            trace_id=self.trace_id,
                            parent_span_id=self.span_id,
                            metadata={
                                "phase": phase_name,
                                "phase_index": self._get_phase_index(phase_name),
                                **(metadata or {}),
                            },
                            tags=["tool_composer", phase_name],
                        ) as span:
                            phase_ctx._opik_span = span
                            yield phase_ctx
                            # Set output on completion
                            span.set_output(phase_ctx.metadata)
                        return
                except Exception as e:
                    logger.debug(f"Failed to create phase span: {e}")

            # Fall through to non-traced version
            yield phase_ctx

        finally:
            # Record duration
            end_time = datetime.now(timezone.utc)
            phase_ctx.end_time = end_time
            phase_ctx.duration_ms = (end_time - start_time).total_seconds() * 1000
            self.phase_durations[phase_name] = int(phase_ctx.duration_ms)

            # Store in parent context
            self.phase_spans[phase_name] = phase_ctx

            logger.debug(
                f"Phase {phase_name} completed in {phase_ctx.duration_ms:.2f}ms"
            )

    def _get_phase_index(self, phase_name: str) -> int:
        """Get numeric index for phase ordering."""
        phase_order = ["decompose", "plan", "execute", "synthesize"]
        return phase_order.index(phase_name) if phase_name in phase_order else -1

    def log_composition_complete(
        self,
        status: str,
        success: bool,
        total_duration_ms: int,
        sub_question_count: int = 0,
        tools_executed: int = 0,
        tools_succeeded: int = 0,
        confidence: float = 0.0,
        parallel_executions: int = 0,
        errors: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Log completion of the full composition.

        Args:
            status: Final status (success, partial, failed)
            success: Whether composition succeeded
            total_duration_ms: Total duration in milliseconds
            sub_question_count: Number of sub-questions
            tools_executed: Total tools executed
            tools_succeeded: Tools that succeeded
            confidence: Final response confidence
            parallel_executions: Count of parallel tool calls
            errors: Any errors encountered
            **kwargs: Additional metrics
        """
        output_data = {
            "status": status,
            "success": success,
            "total_duration_ms": total_duration_ms,
            "sub_question_count": sub_question_count,
            "tools_executed": tools_executed,
            "tools_succeeded": tools_succeeded,
            "confidence": confidence,
            "parallel_executions": parallel_executions,
            "phase_durations": self.phase_durations,
            "errors": errors or [],
            **kwargs,
        }

        if self._opik_span:
            # Set key attributes for filtering
            self._opik_span.set_attribute("status", status)
            self._opik_span.set_attribute("success", success)
            self._opik_span.set_attribute("total_duration_ms", total_duration_ms)
            self._opik_span.set_attribute("tools_executed", tools_executed)
            self._opik_span.set_attribute("tools_succeeded", tools_succeeded)
            self._opik_span.set_attribute("confidence", confidence)

            # Set output data
            self._opik_span.set_output(output_data)

        logger.info(
            f"Composition complete: status={status}, "
            f"{tools_succeeded}/{tools_executed} tools, "
            f"confidence={confidence:.2f}, "
            f"duration={total_duration_ms}ms"
        )


class ToolComposerOpikTracer:
    """Opik tracer for Tool Composer 4-phase pipeline.

    Provides observability into the composition pipeline with:
    - Root trace for full composition
    - Child spans for each phase
    - Tool execution metrics
    - Parallel execution tracking

    Uses the shared OpikConnector for circuit breaker protection.

    Example:
        >>> tracer = ToolComposerOpikTracer()
        >>> async with tracer.trace_composition(query="...") as trace:
        ...     async with trace.trace_phase("decompose") as phase:
        ...         result = await decomposer.decompose(query)
        ...         phase.log_decomposition(len(result.sub_questions), ...)
        ...     # ... other phases
        ...     trace.log_composition_complete(...)
    """

    def __init__(
        self,
        project_name: str = "e2i-tool-composer",
        enabled: bool = True,
        sample_rate: float = 1.0,
    ):
        """Initialize the Tool Composer tracer.

        Args:
            project_name: Opik project name
            enabled: Whether tracing is enabled
            sample_rate: Sample rate (1.0 = trace all, 0.1 = 10%)
        """
        self.project_name = project_name
        self.enabled = enabled
        self.sample_rate = sample_rate
        self._opik_connector = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of OpikConnector."""
        if self._initialized:
            return

        try:
            from src.mlops.opik_connector import get_opik_connector

            self._opik_connector = get_opik_connector()
            self._initialized = True
            logger.debug("ToolComposerOpikTracer initialized")
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
        return (
            self.enabled
            and self._opik_connector is not None
            and self._opik_connector.is_enabled
        )

    def _should_trace(self) -> bool:
        """Determine if this composition should be traced."""
        import random

        return random.random() < self.sample_rate

    @asynccontextmanager
    async def trace_composition(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing a full composition.

        Creates an Opik trace for the composition and provides a context
        object for creating phase spans and logging metrics.

        Args:
            query: The query being composed
            context: Composition context (session_id, brand, region)
            metadata: Additional trace metadata

        Yields:
            CompositionTraceContext for phase tracing and metric logging

        Example:
            async with tracer.trace_composition(query) as trace:
                async with trace.trace_phase("decompose") as phase:
                    # ... decomposition logic
                    phase.log_decomposition(...)
        """
        self._ensure_initialized()

        trace_id = str(uuid7_func())
        span_id = str(uuid7_func())
        start_time = datetime.now(timezone.utc)

        # Build metadata
        trace_metadata = {
            "query_length": len(query),
            "has_context": context is not None,
            **(metadata or {}),
        }

        if context:
            trace_metadata.update({
                "session_id": context.get("session_id"),
                "brand": context.get("brand"),
                "region": context.get("region"),
            })

        # Create trace context
        trace_ctx = CompositionTraceContext(
            trace_id=trace_id,
            span_id=span_id,
            query=query,
            start_time=start_time,
            metadata=trace_metadata,
            _tracer=self,
        )

        error_occurred = False
        error_info = None

        try:
            # Create Opik trace if enabled and sampled
            if self.is_enabled and self._should_trace():
                try:
                    async with self._opik_connector.trace_agent(
                        agent_name="tool_composer",
                        operation="compose",
                        trace_id=trace_id,
                        metadata={
                            "pipeline": "decompose→plan→execute→synthesize",
                            "tier": 1,
                            **trace_metadata,
                        },
                        tags=["tool_composer", "composition", "tier1"],
                        input_data={
                            "query": query[:500],  # Truncate for Opik
                            "context_keys": list((context or {}).keys()),
                        },
                    ) as span:
                        trace_ctx._opik_span = span
                        yield trace_ctx
                        return
                except Exception as e:
                    logger.debug(f"Opik tracing failed, continuing without: {e}")

            # Fall through to non-traced version
            yield trace_ctx

        except Exception as e:
            error_occurred = True
            error_info = {"type": type(e).__name__, "message": str(e)}
            raise

        finally:
            # Record final timing
            end_time = datetime.now(timezone.utc)
            trace_ctx.end_time = end_time
            trace_ctx.duration_ms = (end_time - start_time).total_seconds() * 1000

            logger.debug(
                f"Composition trace completed in {trace_ctx.duration_ms:.2f}ms"
            )


def trace_composition(
    query_param: str = "query",
    context_param: str = "context",
    project_name: str = "e2i-tool-composer",
) -> Callable[[F], F]:
    """Decorator to trace a composition function.

    Use this decorator on composition functions for automatic tracing.
    The decorated function receives a CompositionTraceContext as the first argument.

    Args:
        query_param: Name of the query parameter in the decorated function
        context_param: Name of the context parameter
        project_name: Opik project name

    Returns:
        Decorated function

    Example:
        >>> @trace_composition()
        ... async def compose(trace: CompositionTraceContext, query: str, context=None):
        ...     async with trace.trace_phase("decompose") as phase:
        ...         # ... decomposition
        ...     # ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract query from kwargs or args
            query = kwargs.get(query_param, args[0] if args else "")
            context = kwargs.get(context_param)

            tracer = ToolComposerOpikTracer(project_name=project_name)

            async with tracer.trace_composition(
                query=query,
                context=context,
            ) as trace_ctx:
                # Pass context as first argument
                return await func(trace_ctx, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# Singleton tracer instance
_tracer_instance: Optional[ToolComposerOpikTracer] = None


def get_tool_composer_tracer(
    project_name: str = "e2i-tool-composer",
    enabled: bool = True,
    sample_rate: float = 1.0,
) -> ToolComposerOpikTracer:
    """Get the Tool Composer Opik tracer singleton.

    Args:
        project_name: Opik project name
        enabled: Whether tracing is enabled
        sample_rate: Sample rate for tracing

    Returns:
        ToolComposerOpikTracer instance
    """
    global _tracer_instance

    if _tracer_instance is None:
        _tracer_instance = ToolComposerOpikTracer(
            project_name=project_name,
            enabled=enabled,
            sample_rate=sample_rate,
        )

    return _tracer_instance


def reset_tool_composer_tracer() -> None:
    """Reset the tracer singleton (for testing)."""
    global _tracer_instance
    _tracer_instance = None


__all__ = [
    "PhaseSpanContext",
    "CompositionTraceContext",
    "ToolComposerOpikTracer",
    "trace_composition",
    "get_tool_composer_tracer",
    "reset_tool_composer_tracer",
]
