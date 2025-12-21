"""
Opik Connector for E2I Causal Analytics.

This module provides a centralized wrapper for the Opik SDK, enabling
LLM and agent observability across all 18 agents in the E2I platform.

Features:
- Singleton pattern for consistent configuration
- Async context managers for tracing agent operations
- LLM call tracking with token usage
- Metric and feedback logging
- Graceful degradation when Opik is unavailable
- Circuit breaker pattern (Phase 3)

Usage:
    from src.mlops.opik_connector import OpikConnector

    opik = OpikConnector()

    # Trace an agent operation
    async with opik.trace_agent("gap_analyzer", "analyze_gaps") as span:
        result = await analyze()
        span.set_attribute("result_count", len(result))

    # Trace an LLM call
    async with opik.trace_llm_call(
        model="claude-sonnet-4-20250514",
        trace_id=trace_id
    ) as llm_span:
        response = await client.messages.create(...)
        llm_span.log_tokens(response.usage.input_tokens, response.usage.output_tokens)

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SpanType(str, Enum):
    """Opik span types."""

    GENERAL = "general"
    TOOL = "tool"
    LLM = "llm"
    GUARDRAIL = "guardrail"


class SpanStatus(str, Enum):
    """Span execution status."""

    STARTED = "started"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class OpikConfig:
    """Configuration for Opik connector."""

    api_key: Optional[str] = None
    workspace: str = "default"
    project_name: str = "e2i-causal-analytics"
    url: Optional[str] = None
    use_local: bool = False
    enabled: bool = True
    sample_rate: float = 1.0  # 1.0 = sample all, 0.1 = sample 10%
    always_sample_errors: bool = True

    @classmethod
    def from_env(cls) -> "OpikConfig":
        """Create config from environment variables."""
        return cls(
            api_key=os.getenv("OPIK_API_KEY"),
            workspace=os.getenv("OPIK_WORKSPACE", "default"),
            project_name=os.getenv("OPIK_PROJECT_NAME", "e2i-causal-analytics"),
            url=os.getenv("OPIK_ENDPOINT"),
            use_local=os.getenv("OPIK_USE_LOCAL", "false").lower() == "true",
            enabled=os.getenv("OPIK_ENABLED", "true").lower() == "true",
            sample_rate=float(os.getenv("OPIK_SAMPLE_RATE", "1.0")),
            always_sample_errors=os.getenv("OPIK_ALWAYS_SAMPLE_ERRORS", "true").lower()
            == "true",
        )


@dataclass
class SpanContext:
    """Context object for an Opik span.

    Provides methods for enriching the span with additional data.
    """

    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str = ""
    agent_name: str = ""
    operation: str = ""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.STARTED
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    _opik_span: Any = None  # Actual Opik Span object

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a custom attribute on the span."""
        self.metadata[key] = value
        if self._opik_span:
            try:
                # Update the Opik span metadata
                current_metadata = getattr(self._opik_span, "_metadata", {}) or {}
                current_metadata[key] = value
                self._opik_span.update(metadata=current_metadata)
            except Exception as e:
                logger.debug(f"Failed to update Opik span attribute: {e}")

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span.

        Events are stored in metadata under 'events' key.
        """
        event = {
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes or {},
        }
        if "events" not in self.metadata:
            self.metadata["events"] = []
        self.metadata["events"].append(event)
        self.set_attribute("events", self.metadata["events"])

    def set_input(self, input_data: Dict[str, Any]) -> None:
        """Set the input data for the span."""
        self.input_data = input_data
        if self._opik_span:
            try:
                self._opik_span.update(input=input_data)
            except Exception as e:
                logger.debug(f"Failed to set Opik span input: {e}")

    def set_output(self, output_data: Dict[str, Any]) -> None:
        """Set the output data for the span."""
        self.output_data = output_data
        if self._opik_span:
            try:
                self._opik_span.update(output=output_data)
            except Exception as e:
                logger.debug(f"Failed to set Opik span output: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert span context to dictionary for database storage."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": f"{self.agent_name}.{self.operation}",
            "agent_name": self.agent_name,
            "started_at": self.start_time.isoformat(),
            "ended_at": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "attributes": self.metadata,
            "input": self.input_data,
            "output": self.output_data,
        }


@dataclass
class LLMSpanContext(SpanContext):
    """Context for LLM call spans with token tracking."""

    model: str = ""
    provider: str = "anthropic"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: Optional[float] = None

    def log_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Log token usage for the LLM call."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens

        # Update Opik span with usage
        if self._opik_span:
            try:
                self._opik_span.update(
                    usage={
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": self.total_tokens,
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to update Opik span tokens: {e}")

    def set_cost(self, cost_usd: float) -> None:
        """Set the cost for the LLM call in USD."""
        self.total_cost = cost_usd
        if self._opik_span:
            try:
                self._opik_span.update(total_cost=cost_usd)
            except Exception as e:
                logger.debug(f"Failed to update Opik span cost: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert LLM span context to dictionary for database storage."""
        base = super().to_dict()
        base.update(
            {
                "model_name": self.model,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            }
        )
        return base


class OpikConnector:
    """Singleton wrapper for Opik SDK operations.

    Provides centralized observability for all E2I agents with:
    - Trace and span creation
    - LLM call tracking
    - Metric and feedback logging
    - Graceful degradation when Opik is unavailable

    Example:
        opik = OpikConnector()

        async with opik.trace_agent("gap_analyzer", "analyze") as span:
            result = await do_analysis()
            span.set_attribute("items_analyzed", len(result))
    """

    _instance: Optional["OpikConnector"] = None
    _initialized: bool = False

    def __new__(cls, config: Optional[OpikConfig] = None) -> "OpikConnector":
        """Singleton pattern - return existing instance if available."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[OpikConfig] = None) -> None:
        """Initialize the Opik connector.

        Args:
            config: Optional configuration. If not provided, reads from environment.
        """
        if self._initialized:
            return

        self.config = config or OpikConfig.from_env()
        self._opik_client = None
        self._active_traces: Dict[str, Any] = {}

        # Try to initialize Opik client
        if self.config.enabled:
            try:
                self._init_opik_client()
                logger.info(
                    f"Opik connector initialized for project: {self.config.project_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Opik client: {e}")
                logger.warning("Opik observability will be disabled for this session")
                self._opik_client = None

        self._initialized = True

    def _init_opik_client(self) -> None:
        """Initialize the Opik client with configuration."""
        try:
            import opik

            # Configure Opik globally if API key is provided
            if self.config.api_key:
                opik.configure(
                    api_key=self.config.api_key,
                    workspace=self.config.workspace,
                    url=self.config.url,
                    use_local=self.config.use_local,
                    force=False,  # Don't overwrite existing config
                )

            # Create Opik client instance
            self._opik_client = opik.Opik(project_name=self.config.project_name)
            logger.debug("Opik client initialized successfully")

        except ImportError:
            logger.error("Opik package not installed. Run: pip install opik")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Opik: {e}")
            raise

    @property
    def is_enabled(self) -> bool:
        """Check if Opik is enabled and client is available."""
        return self.config.enabled and self._opik_client is not None

    def _should_sample(self, is_error: bool = False) -> bool:
        """Determine if this trace should be sampled.

        Args:
            is_error: Whether the operation resulted in an error.

        Returns:
            True if the trace should be recorded.
        """
        if not self.is_enabled:
            return False

        # Always sample errors if configured
        if is_error and self.config.always_sample_errors:
            return True

        # Random sampling based on sample rate
        import random

        return random.random() < self.config.sample_rate

    @asynccontextmanager
    async def trace_agent(
        self,
        agent_name: str,
        operation: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        input_data: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing agent operations.

        Creates a trace (if no trace_id) or span (if trace_id provided) in Opik
        and tracks timing, status, and errors.

        Args:
            agent_name: Name of the agent (e.g., "gap_analyzer")
            operation: Operation being performed (e.g., "analyze_gaps")
            trace_id: Optional trace ID to attach this span to an existing trace
            parent_span_id: Optional parent span ID for nested spans
            metadata: Additional metadata to attach to the span
            tags: Tags for categorizing the span
            input_data: Input data for the operation

        Yields:
            SpanContext: Context object for enriching the span

        Example:
            async with opik.trace_agent("gap_analyzer", "analyze") as span:
                span.set_attribute("brand", "Kisqali")
                result = await analyze()
                span.set_output({"gaps": result})
        """
        span_id = str(uuid.uuid4())
        is_new_trace = trace_id is None
        trace_id = trace_id or str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        # Create span context
        span_ctx = SpanContext(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=f"{agent_name}.{operation}",
            agent_name=agent_name,
            operation=operation,
            start_time=start_time,
            metadata=metadata or {},
            input_data=input_data,
        )

        opik_trace = None
        opik_span = None
        error_occurred = False

        try:
            # Create Opik trace/span if enabled
            if self.is_enabled and self._should_sample():
                try:
                    if is_new_trace:
                        # Create new trace
                        opik_trace = self._opik_client.trace(
                            id=trace_id,
                            name=f"{agent_name}.{operation}",
                            start_time=start_time,
                            input=input_data,
                            metadata={
                                "agent_name": agent_name,
                                "operation": operation,
                                "agent_tier": self._get_agent_tier(agent_name),
                                **(metadata or {}),
                            },
                            tags=tags or [agent_name, operation],
                        )
                        self._active_traces[trace_id] = opik_trace

                        # Create span within trace
                        opik_span = opik_trace.span(
                            id=span_id,
                            name=f"{agent_name}.{operation}",
                            type=SpanType.GENERAL.value,
                            start_time=start_time,
                            input=input_data,
                            metadata=metadata,
                        )
                    else:
                        # Get existing trace or create span directly
                        existing_trace = self._active_traces.get(trace_id)
                        if existing_trace:
                            opik_span = existing_trace.span(
                                id=span_id,
                                parent_span_id=parent_span_id,
                                name=f"{agent_name}.{operation}",
                                type=SpanType.GENERAL.value,
                                start_time=start_time,
                                input=input_data,
                                metadata=metadata,
                            )
                        else:
                            # Create orphan span (trace already ended or not found)
                            logger.debug(
                                f"Trace {trace_id} not found, creating standalone span"
                            )

                    span_ctx._opik_span = opik_span

                except Exception as e:
                    logger.warning(f"Failed to create Opik span: {e}")

            yield span_ctx

            # Mark successful completion
            span_ctx.status = SpanStatus.COMPLETED

        except Exception as e:
            # Mark error
            error_occurred = True
            span_ctx.status = SpanStatus.ERROR
            span_ctx.error_type = type(e).__name__
            span_ctx.error_message = str(e)
            raise

        finally:
            # Calculate duration
            end_time = datetime.now(timezone.utc)
            span_ctx.end_time = end_time
            span_ctx.duration_ms = (end_time - start_time).total_seconds() * 1000

            # End Opik span
            if opik_span:
                try:
                    opik_span.end(
                        end_time=end_time,
                        output=span_ctx.output_data,
                        error_info=(
                            {
                                "exception_type": span_ctx.error_type,
                                "message": span_ctx.error_message,
                            }
                            if error_occurred
                            else None
                        ),
                    )
                except Exception as e:
                    logger.debug(f"Failed to end Opik span: {e}")

            # End trace if we created it
            if is_new_trace and opik_trace:
                try:
                    opik_trace.end(
                        end_time=end_time,
                        output=span_ctx.output_data,
                        error_info=(
                            {
                                "exception_type": span_ctx.error_type,
                                "message": span_ctx.error_message,
                            }
                            if error_occurred
                            else None
                        ),
                    )
                    # Remove from active traces
                    self._active_traces.pop(trace_id, None)
                except Exception as e:
                    logger.debug(f"Failed to end Opik trace: {e}")

            # Log span locally for debugging
            logger.debug(
                f"Span completed: {span_ctx.name} "
                f"[{span_ctx.status.value}] "
                f"{span_ctx.duration_ms:.2f}ms"
            )

    @asynccontextmanager
    async def trace_llm_call(
        self,
        model: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        provider: str = "anthropic",
        prompt_template: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing LLM API calls.

        Creates an LLM-type span with token tracking and cost calculation.

        Args:
            model: LLM model name (e.g., "claude-sonnet-4-20250514")
            trace_id: Trace ID to attach this span to
            parent_span_id: Parent span ID for nesting
            provider: LLM provider (anthropic, openai, etc.)
            prompt_template: Name of prompt template used
            input_data: Input data (prompt, messages, etc.)
            metadata: Additional metadata

        Yields:
            LLMSpanContext: Context for tracking tokens and cost

        Example:
            async with opik.trace_llm_call(
                model="claude-sonnet-4-20250514",
                trace_id=trace_id
            ) as llm_span:
                response = await client.messages.create(...)
                llm_span.log_tokens(
                    response.usage.input_tokens,
                    response.usage.output_tokens
                )
        """
        span_id = str(uuid.uuid4())
        trace_id = trace_id or str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        # Build metadata
        span_metadata = {
            "model": model,
            "provider": provider,
            **(metadata or {}),
        }
        if prompt_template:
            span_metadata["prompt_template"] = prompt_template

        # Create LLM span context
        llm_ctx = LLMSpanContext(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=f"llm.{model}",
            agent_name="llm",
            operation=model,
            model=model,
            provider=provider,
            start_time=start_time,
            metadata=span_metadata,
            input_data=input_data,
        )

        opik_span = None
        error_occurred = False

        try:
            # Create Opik span if enabled
            if self.is_enabled and self._should_sample():
                try:
                    existing_trace = self._active_traces.get(trace_id)
                    if existing_trace:
                        opik_span = existing_trace.span(
                            id=span_id,
                            parent_span_id=parent_span_id,
                            name=f"llm.{model}",
                            type=SpanType.LLM.value,
                            start_time=start_time,
                            model=model,
                            provider=provider,
                            input=input_data,
                            metadata=span_metadata,
                        )
                        llm_ctx._opik_span = opik_span
                except Exception as e:
                    logger.warning(f"Failed to create Opik LLM span: {e}")

            yield llm_ctx

            llm_ctx.status = SpanStatus.COMPLETED

        except Exception as e:
            error_occurred = True
            llm_ctx.status = SpanStatus.ERROR
            llm_ctx.error_type = type(e).__name__
            llm_ctx.error_message = str(e)
            raise

        finally:
            end_time = datetime.now(timezone.utc)
            llm_ctx.end_time = end_time
            llm_ctx.duration_ms = (end_time - start_time).total_seconds() * 1000

            if opik_span:
                try:
                    opik_span.end(
                        end_time=end_time,
                        output=llm_ctx.output_data,
                        usage={
                            "prompt_tokens": llm_ctx.input_tokens,
                            "completion_tokens": llm_ctx.output_tokens,
                            "total_tokens": llm_ctx.total_tokens,
                        },
                        total_cost=llm_ctx.total_cost,
                        error_info=(
                            {
                                "exception_type": llm_ctx.error_type,
                                "message": llm_ctx.error_message,
                            }
                            if error_occurred
                            else None
                        ),
                    )
                except Exception as e:
                    logger.debug(f"Failed to end Opik LLM span: {e}")

    def log_metric(
        self,
        name: str,
        value: float,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a custom metric.

        Args:
            name: Metric name
            value: Metric value
            trace_id: Optional trace to associate metric with
            metadata: Additional metadata
        """
        if not self.is_enabled:
            return

        try:
            # Log as feedback score on trace if trace_id provided
            if trace_id and self._opik_client:
                trace = self._active_traces.get(trace_id)
                if trace:
                    trace.log_feedback_score(
                        name=name,
                        value=value,
                        reason=metadata.get("reason") if metadata else None,
                    )
                    logger.debug(f"Logged metric {name}={value} to trace {trace_id}")
        except Exception as e:
            logger.warning(f"Failed to log metric: {e}")

    def log_feedback(
        self,
        trace_id: str,
        score: float,
        feedback_type: str = "quality",
        reason: Optional[str] = None,
    ) -> None:
        """Log feedback for a trace.

        Used for logging user ratings, quality scores, etc.

        Args:
            trace_id: The trace to log feedback for
            score: Feedback score (0.0 to 1.0)
            feedback_type: Type of feedback (quality, relevance, etc.)
            reason: Optional reason for the score
        """
        if not self.is_enabled:
            return

        try:
            trace = self._active_traces.get(trace_id)
            if trace:
                trace.log_feedback_score(
                    name=feedback_type,
                    value=score,
                    reason=reason,
                )
                logger.debug(
                    f"Logged feedback {feedback_type}={score} to trace {trace_id}"
                )
        except Exception as e:
            logger.warning(f"Failed to log feedback: {e}")

    def flush(self) -> None:
        """Flush any pending data to Opik.

        Call this before process exit to ensure all traces are sent.
        """
        if not self.is_enabled:
            return

        try:
            import opik

            opik.flush_tracker()
            logger.debug("Flushed Opik tracker")
        except Exception as e:
            logger.warning(f"Failed to flush Opik tracker: {e}")

    def _get_agent_tier(self, agent_name: str) -> int:
        """Get the tier number for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Tier number (0-5)
        """
        tier_mapping = {
            # Tier 0 - ML Foundation
            "scope_definer": 0,
            "data_preparer": 0,
            "feature_analyzer": 0,
            "model_selector": 0,
            "model_trainer": 0,
            "model_deployer": 0,
            "observability_connector": 0,
            # Tier 1 - Coordination
            "orchestrator": 1,
            "tool_composer": 1,
            # Tier 2 - Causal Analytics
            "causal_impact": 2,
            "gap_analyzer": 2,
            "heterogeneous_optimizer": 2,
            # Tier 3 - Monitoring
            "drift_monitor": 3,
            "experiment_designer": 3,
            "health_score": 3,
            # Tier 4 - ML Predictions
            "prediction_synthesizer": 4,
            "resource_optimizer": 4,
            # Tier 5 - Self-Improvement
            "explainer": 5,
            "feedback_learner": 5,
        }
        return tier_mapping.get(agent_name, 0)


# Convenience function for getting singleton instance
def get_opik_connector(config: Optional[OpikConfig] = None) -> OpikConnector:
    """Get the OpikConnector singleton instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        OpikConnector instance
    """
    return OpikConnector(config)
