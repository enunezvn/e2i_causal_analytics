"""Observability Connector Agent - STANDARD.

Provides telemetry and monitoring across all 18 agents.

Responsibilities:
- Span emission to Opik
- Quality metrics computation
- Trace context propagation
- Latency and error rate tracking
- Token usage monitoring

Outputs:
- Spans (to Opik and database)
- QualityMetrics (aggregated stats)
- ObservabilityContext (distributed tracing)

Integration:
- Upstream: ALL agents (cross-cutting)
- Downstream: Database (ml_observability_spans)
- Used via: span() context manager, track_llm_call(), get_quality_metrics()
"""

import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from .graph import create_observability_connector_graph
from .state import ObservabilityConnectorState
from .nodes import create_context, extract_context, inject_context


class ObservabilityConnectorAgent:
    """Observability Connector: Telemetry and monitoring for all agents.

    This agent operates differently from others - it wraps operations with
    spans rather than being invoked in the main pipeline.
    """

    tier = 0
    tier_name = "ml_foundation"
    agent_type = "standard"
    sla_seconds = 5  # 5 seconds (cross-cutting agent)

    def __init__(self):
        """Initialize observability_connector agent."""
        self.graph = create_observability_connector_graph()
        # In production, initialize Opik client
        # self.opik = OpikClient()

    # ========================================================================
    # HELPER METHODS (Primary Interface)
    # ========================================================================

    def create_observability_context(
        self,
        request_id: str,
        experiment_id: Optional[str] = None,
        user_id: Optional[str] = None,
        sample_rate: float = 1.0,
    ) -> Dict[str, Any]:
        """Create new observability context for a request.

        Args:
            request_id: Unique request identifier
            experiment_id: Optional experiment ID
            user_id: Optional user ID
            sample_rate: Sampling rate (0.0-1.0), default 1.0 (sample everything)

        Returns:
            Observability context dict
        """
        trace_id = self._generate_trace_id()
        span_id = self._generate_span_id()
        sampled = self._should_sample(sample_rate)

        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": None,
            "request_id": request_id,
            "experiment_id": experiment_id,
            "user_id": user_id,
            "sampled": sampled,
            "sample_rate": sample_rate,
        }

    def create_child_context(
        self, parent_context: Dict[str, Any], operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create child context for nested operations.

        Args:
            parent_context: Parent observability context
            operation_name: Optional name of child operation

        Returns:
            Child observability context
        """
        return {
            "trace_id": parent_context["trace_id"],
            "span_id": self._generate_span_id(),
            "parent_span_id": parent_context["span_id"],
            "request_id": parent_context.get("request_id"),
            "experiment_id": parent_context.get("experiment_id"),
            "user_id": parent_context.get("user_id"),
            "sampled": parent_context.get("sampled", True),
            "sample_rate": parent_context.get("sample_rate", 1.0),
        }

    @asynccontextmanager
    async def span(
        self,
        operation_name: str,
        agent_name: str,
        context: Dict[str, Any],
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for creating observability spans.

        Usage:
            async with observability.span("execute", "scope_definer", ctx) as span:
                # Agent operation
                result = await agent.execute(state)
                span.set_attribute("problem_type", result["problem_type"])

        Args:
            operation_name: Operation name (e.g., "execute", "compute")
            agent_name: Agent name (e.g., "scope_definer")
            context: Observability context dict
            attributes: Optional span attributes

        Yields:
            Span object with set_attribute() and add_event() methods
        """
        span_id = context["span_id"]
        trace_id = context["trace_id"]
        parent_span_id = context.get("parent_span_id")

        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            agent_name=agent_name,
            start_time=datetime.now(timezone.utc),
            status="started",
            attributes=attributes or {},
        )

        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.error_type = type(e).__name__
            span.error_message = str(e)
            raise
        finally:
            span.end_time = datetime.now(timezone.utc)
            # Ensure at least 1ms duration (operations complete in <1ms can show as 0)
            span.duration_ms = max(
                1, int((span.end_time - span.start_time).total_seconds() * 1000)
            )

            # Emit span asynchronously (non-blocking)
            if context.get("sampled", True):
                await self._emit_span_async(span)

    async def track_llm_call(
        self,
        agent_name: str,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        context: Dict[str, Any],
        attributes: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Track LLM usage (for Hybrid/Deep agents).

        Args:
            agent_name: Name of the agent making the LLM call
            operation: Operation name (e.g., "interpret")
            model: LLM model name
            input_tokens: Input token count
            output_tokens: Output token count
            context: Observability context dict
            attributes: Optional additional attributes
            error: Optional error message
            error_type: Optional error type

        Returns:
            LLM call tracking result dict
        """
        span_id = self._generate_span_id()
        trace_id = context.get("trace_id", self._generate_trace_id())
        total_tokens = input_tokens + output_tokens
        status = "error" if error else "ok"

        result = {
            "span_id": span_id,
            "trace_id": trace_id,
            "agent_name": agent_name,
            "operation": operation,
            "model_used": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_used": total_tokens,
            "status": status,
            "attributes": attributes or {},
        }

        if error:
            result["error"] = error
            result["error_type"] = error_type

        # Emit span for tracking
        if context.get("sampled", True):
            event = {
                "span_id": span_id,
                "trace_id": trace_id,
                "agent_name": agent_name,
                "operation": operation,
                "status": status,
                "model_used": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_used": total_tokens,
            }
            await self.graph.ainvoke({"events_to_log": [event]})

        return result

    async def get_quality_metrics(
        self, input_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get aggregated quality metrics.

        Args:
            input_data: Optional input with:
                - time_window: Time window ("1h", "24h", "7d")
                - agent_name_filter: Optional agent name filter
                - trace_id_filter: Optional trace ID filter

        Returns:
            Quality metrics dict
        """
        input_data = input_data or {}
        time_window = input_data.get("time_window", "24h")
        agent_filter = input_data.get("agent_name_filter")
        trace_id_filter = input_data.get("trace_id_filter")

        initial_state: ObservabilityConnectorState = {
            "events_to_log": [],  # No events to log, just compute metrics
            "time_window": time_window,
            "agent_name_filter": agent_filter,
            "trace_id_filter": trace_id_filter,
        }

        # Execute workflow to compute metrics
        final_state = await self.graph.ainvoke(initial_state)

        # Extract metrics
        return {
            "quality_metrics_computed": final_state.get(
                "quality_metrics_computed", True
            ),
            "latency_by_agent": final_state.get("latency_by_agent", {}),
            "latency_by_tier": final_state.get("latency_by_tier", {}),
            "error_rate_by_agent": final_state.get("error_rate_by_agent", {}),
            "error_rate_by_tier": final_state.get("error_rate_by_tier", {}),
            "token_usage_by_agent": final_state.get("token_usage_by_agent", {}),
            "overall_success_rate": final_state.get("overall_success_rate", 1.0),
            "overall_p95_latency_ms": final_state.get("overall_p95_latency_ms", 0.0),
            "overall_p99_latency_ms": final_state.get("overall_p99_latency_ms", 0.0),
            "total_spans_analyzed": final_state.get("total_spans_analyzed", 0),
            "quality_score": final_state.get("quality_score", 1.0),
            "fallback_invocation_rate": final_state.get(
                "fallback_invocation_rate", 0.0
            ),
            "status_distribution": final_state.get(
                "status_distribution", {"ok": 0, "error": 0, "timeout": 0}
            ),
        }

    # ========================================================================
    # MAIN RUN METHOD (For Explicit Invocation)
    # ========================================================================

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute observability workflow (span emission + metrics aggregation).

        This method is typically not called directly. Instead, use:
        - span() context manager for wrapping operations
        - get_quality_metrics() for retrieving metrics

        Args:
            input_data: Input data with optional fields:
                - events_to_log: List[Dict] (ObservabilityEvent dicts)
                - time_window: str (for metrics)
                - agent_name_filter: str (for metrics)
                - trace_id_filter: str (for metrics)

        Returns:
            Output data conforming to ObservabilityConnectorOutput contract
        """
        # Prepare initial state
        initial_state: ObservabilityConnectorState = {
            "events_to_log": input_data.get("events_to_log", []),
            "time_window": input_data.get("time_window", "24h"),
            "agent_name_filter": input_data.get("agent_name_filter"),
            "trace_id_filter": input_data.get("trace_id_filter"),
            "sample_rate": input_data.get("sample_rate", 1.0),
        }

        # Execute LangGraph workflow
        final_state = await self.graph.ainvoke(initial_state)

        # Check for errors
        if final_state.get("error"):
            error_msg = final_state["error"]
            error_type = final_state.get("error_type", "unknown")
            raise RuntimeError(f"{error_type}: {error_msg}")

        # Build output
        output = {
            # Span emission results
            "emission_successful": True,  # No error = success
            "span_ids_logged": final_state.get("span_ids_logged", []),
            "trace_ids_logged": final_state.get("trace_ids_logged", []),
            "events_logged": final_state.get("events_logged", 0),
            # Opik metadata
            "opik_project": final_state.get("opik_project", "e2i-causal-analytics"),
            "opik_workspace": final_state.get("opik_workspace", "default"),
            # Quality metrics
            "quality_metrics_computed": final_state.get(
                "quality_metrics_computed", True
            ),
            "quality_score": final_state.get("quality_score", 1.0),
            "overall_success_rate": final_state.get("overall_success_rate", 1.0),
            "overall_p95_latency_ms": final_state.get("overall_p95_latency_ms", 0.0),
        }

        return output

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    async def _emit_span_async(self, span: "Span"):
        """Emit span asynchronously (non-blocking).

        Args:
            span: Span object
        """
        import asyncio

        # Create task to emit span without blocking
        asyncio.create_task(self._emit_span(span))

    async def _emit_span(self, span: "Span"):
        """Emit span to Opik and database.

        Args:
            span: Span object
        """
        try:
            # Convert span to event dict
            event = {
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "parent_span_id": span.parent_span_id,
                "agent_name": span.agent_name,
                "operation": span.operation_name,
                "started_at": span.start_time.isoformat(),
                "completed_at": span.end_time.isoformat() if span.end_time else None,
                "duration_ms": span.duration_ms,
                "status": span.status,
                "error": span.error_message if hasattr(span, "error_message") else None,
                "metadata": span.attributes,
                "model_used": getattr(span, "llm_model", None),
                "tokens_used": getattr(span, "total_tokens", None),
                "input_tokens": getattr(span, "input_tokens", None),
                "output_tokens": getattr(span, "output_tokens", None),
            }

            # Execute workflow to emit span
            await self.graph.ainvoke({"events_to_log": [event]})

        except Exception as e:
            # Log but don't fail - observability should not break operations
            print(f"Warning: Failed to emit span: {e}")

    def _generate_trace_id(self) -> str:
        """Generate new trace ID."""
        return uuid.uuid4().hex

    def _generate_span_id(self) -> str:
        """Generate new span ID."""
        return uuid.uuid4().hex[:16]

    def _should_sample(self, sample_rate: float) -> bool:
        """Determine if trace should be sampled."""
        import random

        return random.random() < sample_rate


class Span:
    """Observability span for an operation."""

    def __init__(
        self,
        span_id: str,
        trace_id: str,
        parent_span_id: Optional[str],
        operation_name: str,
        agent_name: str,
        start_time: Optional[datetime] = None,
        status: str = "started",
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Initialize span."""
        self.span_id = span_id
        self.trace_id = trace_id
        self.parent_span_id = parent_span_id
        self.operation_name = operation_name
        self.agent_name = agent_name
        self.start_time = start_time or datetime.now(timezone.utc)
        self.end_time: Optional[datetime] = None
        self.duration_ms: Optional[int] = None
        self.status = status
        self.attributes = attributes if attributes is not None else {}
        self.events: List[Dict[str, Any]] = []

    def set_attribute(self, key: str, value: Any):
        """Set span attribute.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span.

        Args:
            name: Event name
            attributes: Event attributes
        """
        self.events.append(
            {
                "name": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attributes": attributes or {},
            }
        )
