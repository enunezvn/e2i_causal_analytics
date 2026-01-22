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
- Circuit breaker pattern for fault tolerance

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

    # Check circuit breaker status
    if opik.circuit_breaker.is_closed:
        # Opik is healthy, full functionality available
        pass

Author: E2I Causal Analytics Team
Version: 4.3.0 (Phase 3 - Circuit Breaker)
"""

import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager

from uuid_utils import uuid7 as uuid7_func  # For Opik-compatible UUID v7
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


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Circuit tripped, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Consecutive failures before opening
    reset_timeout_seconds: float = 30.0  # Time before trying half-open
    half_open_max_calls: int = 3  # Max test calls in half-open state
    success_threshold: int = 2  # Successes needed to close from half-open


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_state_change_time: Optional[float] = None
    time_in_open_state: float = 0.0
    times_opened: int = 0

    def record_success(self) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.last_success_time = time.time()

    def record_failure(self) -> None:
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.last_failure_time = time.time()

    def record_rejected(self) -> None:
        """Record a rejected call (circuit open)."""
        self.rejected_calls += 1

    def record_state_change(self, new_state: CircuitState) -> None:
        """Record a state change."""
        now = time.time()
        self.state_changes += 1
        if new_state == CircuitState.OPEN:
            self.times_opened += 1
        self.last_state_change_time = now

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "state_changes": self.state_changes,
            "times_opened": self.times_opened,
            "success_rate": (
                self.successful_calls / self.total_calls
                if self.total_calls > 0
                else 1.0
            ),
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
        }


class CircuitBreaker:
    """Circuit breaker for protecting external service calls.

    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service is failing, requests are rejected immediately
    - HALF_OPEN: Testing if service has recovered

    Thread-safe implementation for concurrent access.

    Example:
        cb = CircuitBreaker()

        if cb.allow_request():
            try:
                result = call_external_service()
                cb.record_success()
            except Exception:
                cb.record_failure()
        else:
            # Circuit is open, use fallback
            result = use_fallback()
    """

    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
    ):
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
            on_state_change: Callback when state changes (old_state, new_state)
        """
        self._config = config or CircuitBreakerConfig()
        self._on_state_change = on_state_change

        # State management
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        # Thread safety
        self._lock = threading.RLock()

        # Metrics
        self._metrics = CircuitBreakerMetrics()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self._metrics

    @property
    def consecutive_failures(self) -> int:
        """Get current consecutive failure count."""
        with self._lock:
            return self._consecutive_failures

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if request should proceed, False if circuit is open
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if reset timeout has elapsed
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
                self._metrics.record_rejected()
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                if self._half_open_calls < self._config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self._metrics.record_success()
            self._consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                self._consecutive_successes += 1

    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            self._metrics.record_failure()
            self._last_failure_time = time.time()
            self._consecutive_failures += 1
            self._consecutive_successes = 0

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens circuit
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self._config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._half_open_calls = 0

            if old_state != CircuitState.CLOSED:
                self._metrics.record_state_change(CircuitState.CLOSED)
                logger.info("Circuit breaker manually reset to CLOSED")
                if self._on_state_change:
                    self._on_state_change(old_state, CircuitState.CLOSED)

    def force_open(self) -> None:
        """Manually force the circuit to open state."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                self._transition_to(CircuitState.OPEN)
                logger.warning("Circuit breaker manually forced to OPEN")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker status."""
        with self._lock:
            return {
                "state": self._state.value,
                "is_closed": self._state == CircuitState.CLOSED,
                "is_open": self._state == CircuitState.OPEN,
                "is_half_open": self._state == CircuitState.HALF_OPEN,
                "consecutive_failures": self._consecutive_failures,
                "consecutive_successes": self._consecutive_successes,
                "failure_threshold": self._config.failure_threshold,
                "reset_timeout_seconds": self._config.reset_timeout_seconds,
                "time_until_reset": self._time_until_reset(),
                "metrics": self._metrics.to_dict(),
            }

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self._config.reset_timeout_seconds

    def _time_until_reset(self) -> Optional[float]:
        """Get time remaining until reset attempt (if circuit is open)."""
        if self._state != CircuitState.OPEN:
            return None
        if self._last_failure_time is None:
            return 0.0
        elapsed = time.time() - self._last_failure_time
        remaining = self._config.reset_timeout_seconds - elapsed
        return max(0.0, remaining)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._metrics.record_state_change(new_state)

        # Reset counters on state change
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._consecutive_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._consecutive_failures = 0
            self._half_open_calls = 0

        # Log state change
        logger.info(f"Circuit breaker: {old_state.value} → {new_state.value}")

        # Invoke callback
        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state)
            except Exception as e:
                logger.warning(f"Circuit breaker callback failed: {e}")


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

    @classmethod
    def from_config_file(
        cls, config_path: Optional[str] = None, environment: Optional[str] = None
    ) -> "OpikConfig":
        """Create config from observability.yaml file.

        Args:
            config_path: Path to config file (defaults to config/observability.yaml)
            environment: Environment name for overrides

        Returns:
            OpikConfig instance
        """
        try:
            # Lazy import to avoid circular dependencies
            from src.agents.ml_foundation.observability_connector.config import (
                get_observability_config,
            )

            obs_config = get_observability_config(config_path, environment)
            opik_settings = obs_config.opik
            sampling_settings = obs_config.sampling

            return cls(
                api_key=opik_settings.api_key,
                workspace=opik_settings.workspace,
                project_name=opik_settings.project_name,
                url=opik_settings.endpoint,
                use_local=opik_settings.use_local,
                enabled=opik_settings.enabled,
                sample_rate=sampling_settings.default_rate,
                always_sample_errors=sampling_settings.always_sample_errors,
            )
        except Exception as e:
            logger.warning(f"Failed to load config from file: {e}, using env vars")
            return cls.from_env()


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
    - Circuit breaker for fault tolerance

    The circuit breaker protects against cascading failures when the Opik
    service is unavailable. When the circuit opens, operations fall back
    to database-only logging.

    Example:
        opik = OpikConnector()

        async with opik.trace_agent("gap_analyzer", "analyze") as span:
            result = await do_analysis()
            span.set_attribute("items_analyzed", len(result))

        # Check circuit breaker status
        if opik.circuit_breaker.is_open:
            logger.warning("Opik circuit is open, using fallback")
    """

    _instance: Optional["OpikConnector"] = None
    _initialized: bool = False

    def __new__(cls, config: Optional[OpikConfig] = None) -> "OpikConnector":
        """Singleton pattern - return existing instance if available."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config: Optional[OpikConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        """Initialize the Opik connector.

        Args:
            config: Optional configuration. If not provided, reads from environment.
            circuit_breaker_config: Optional circuit breaker configuration.
        """
        if self._initialized:
            return

        self.config = config or OpikConfig.from_env()
        self._opik_client = None
        self._active_traces: Dict[str, Any] = {}

        # Initialize circuit breaker with callback for state changes
        self._circuit_breaker = CircuitBreaker(
            config=circuit_breaker_config or CircuitBreakerConfig(),
            on_state_change=self._on_circuit_state_change,
        )

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

    def _on_circuit_state_change(
        self, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Handle circuit breaker state changes."""
        if new_state == CircuitState.OPEN:
            logger.warning(
                f"Opik circuit breaker OPENED after {self._circuit_breaker.consecutive_failures} failures. "
                f"Falling back to database-only logging."
            )
        elif new_state == CircuitState.HALF_OPEN:
            logger.info("Opik circuit breaker entering HALF-OPEN state, testing recovery...")
        elif new_state == CircuitState.CLOSED:
            logger.info("Opik circuit breaker CLOSED, resuming normal operation.")

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get the circuit breaker instance."""
        return self._circuit_breaker

    def _init_opik_client(self) -> None:
        """Initialize the Opik client with configuration."""
        try:
            import opik

            # Configure Opik globally if API key is provided or using local instance
            if self.config.api_key or self.config.use_local:
                configure_kwargs = {
                    "workspace": self.config.workspace,
                    "use_local": self.config.use_local,
                    "force": False,  # Don't overwrite existing config
                }
                # Only include api_key if provided (local doesn't need it)
                if self.config.api_key:
                    configure_kwargs["api_key"] = self.config.api_key
                # Only include url if provided
                if self.config.url:
                    configure_kwargs["url"] = self.config.url

                opik.configure(**configure_kwargs)
                logger.info(
                    f"Opik configured: use_local={self.config.use_local}, "
                    f"url={self.config.url}, workspace={self.config.workspace}"
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
        force_new_trace: bool = False,
    ):
        """Context manager for tracing agent operations.

        Creates a trace (if no trace_id or force_new_trace) or span (if trace_id
        provided and not force_new_trace) in Opik and tracks timing, status, and errors.

        Args:
            agent_name: Name of the agent (e.g., "gap_analyzer")
            operation: Operation being performed (e.g., "analyze_gaps")
            trace_id: Optional trace ID to use for the trace/span
            parent_span_id: Optional parent span ID for nested spans
            metadata: Additional metadata to attach to the span
            tags: Tags for categorizing the span
            input_data: Input data for the operation
            force_new_trace: If True, create a new trace even if trace_id is provided.
                            Useful when caller generates their own trace_id for tracking.

        Yields:
            SpanContext: Context object for enriching the span

        Example:
            async with opik.trace_agent("gap_analyzer", "analyze") as span:
                span.set_attribute("brand", "Kisqali")
                result = await analyze()
                span.set_output({"gaps": result})
        """
        span_id = str(uuid7_func())  # Opik requires UUID v7
        is_new_trace = trace_id is None or force_new_trace
        trace_id = trace_id or str(uuid7_func())  # Opik requires UUID v7
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
            should_sample = self._should_sample()
            if self.is_enabled and should_sample:
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
                    print(f"[OPIK_CONNECTOR] Failed to create Opik span: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
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
        span_id = str(uuid7_func())  # Opik requires UUID v7
        trace_id = trace_id or str(uuid7_func())  # Opik requires UUID v7
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

        Uses circuit breaker pattern to protect against Opik failures.
        When circuit is open, metrics are silently dropped (logged to debug).

        Args:
            name: Metric name
            value: Metric value
            trace_id: Optional trace to associate metric with
            metadata: Additional metadata
        """
        if not self.is_enabled:
            return

        # Check circuit breaker
        if not self._circuit_breaker.allow_request():
            logger.debug(
                f"Circuit open, dropping metric {name}={value} "
                f"(will retry in {self._circuit_breaker._time_until_reset():.1f}s)"
            )
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

            # Record success
            self._circuit_breaker.record_success()

        except Exception as e:
            # Record failure
            self._circuit_breaker.record_failure()
            logger.warning(f"Failed to log metric: {e}")

    def log_feedback(
        self,
        trace_id: str,
        score: float,
        feedback_type: str = "quality",
        reason: Optional[str] = None,
    ) -> None:
        """Log feedback for a trace.

        Uses circuit breaker pattern to protect against Opik failures.

        Args:
            trace_id: The trace to log feedback for
            score: Feedback score (0.0 to 1.0)
            feedback_type: Type of feedback (quality, relevance, etc.)
            reason: Optional reason for the score
        """
        if not self.is_enabled:
            return

        # Check circuit breaker
        if not self._circuit_breaker.allow_request():
            logger.debug(f"Circuit open, dropping feedback for trace {trace_id}")
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
            self._circuit_breaker.record_success()
        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.warning(f"Failed to log feedback: {e}")

    async def log_model_prediction(
        self,
        model_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> Optional[str]:
        """Log a model prediction to Opik for audit trail.

        Creates a trace for the prediction with input/output data logged.
        Uses circuit breaker pattern for fault tolerance.

        Args:
            model_name: Name of the model/service making the prediction
            input_data: Input data for the prediction
            output_data: Output/prediction result
            metadata: Additional metadata (latency, service_type, etc.)
            trace_id: Optional trace ID to use (generates UUID v7 if not provided)

        Returns:
            The trace ID if successful, None otherwise

        Example:
            await connector.log_model_prediction(
                model_name="churn_classifier",
                input_data={"features": [...]},
                output_data={"prediction": 0.85, "class": "high_risk"},
                metadata={"latency_ms": 45.2, "service_type": "bentoml"},
            )
        """
        if not self.is_enabled:
            return None

        # Check circuit breaker
        if not self._circuit_breaker.allow_request():
            logger.debug(
                f"Circuit open, dropping prediction log for {model_name} "
                f"(will retry in {self._circuit_breaker._time_until_reset():.1f}s)"
            )
            return None

        trace_id = trace_id or str(uuid7_func())
        start_time = datetime.now(timezone.utc)

        try:
            # Build comprehensive metadata
            prediction_metadata = {
                "model_name": model_name,
                "prediction_type": "model_serving",
                "logged_at": start_time.isoformat(),
                **(metadata or {}),
            }

            # Create trace for the prediction
            opik_trace = self._opik_client.trace(
                id=trace_id,
                name=f"prediction.{model_name}",
                start_time=start_time,
                input=input_data,
                output=output_data,
                metadata=prediction_metadata,
                tags=["prediction", model_name, "model_serving"],
            )

            # End the trace immediately since prediction is complete
            end_time = datetime.now(timezone.utc)
            opik_trace.end(
                end_time=end_time,
                output=output_data,
            )

            # Record success
            self._circuit_breaker.record_success()

            logger.debug(
                f"Logged prediction for {model_name} "
                f"[trace_id={trace_id}]"
            )
            return trace_id

        except Exception as e:
            # Record failure
            self._circuit_breaker.record_failure()
            logger.warning(f"Failed to log prediction to Opik: {e}")
            return None

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

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status including circuit breaker state.

        Returns:
            Dictionary with connector and circuit breaker status
        """
        return {
            "enabled": self.config.enabled,
            "is_enabled": self.is_enabled,
            "project_name": self.config.project_name,
            "workspace": self.config.workspace,
            "sample_rate": self.config.sample_rate,
            "active_traces": len(self._active_traces),
            "circuit_breaker": self._circuit_breaker.get_status(),
        }


# Singleton instance
_opik_connector_instance: Optional[OpikConnector] = None
_opik_connector_lock = threading.Lock()


def get_opik_connector(
    config: Optional[OpikConfig] = None,
    force_new: bool = False,
) -> OpikConnector:
    """Get the OpikConnector singleton instance.

    Args:
        config: Optional configuration (only used on first call or force_new)
        force_new: Force creation of a new instance

    Returns:
        OpikConnector instance
    """
    global _opik_connector_instance

    with _opik_connector_lock:
        if _opik_connector_instance is None or force_new:
            _opik_connector_instance = OpikConnector(config)
        return _opik_connector_instance


def reset_opik_connector() -> None:
    """Reset the OpikConnector singleton instance.

    Useful for testing to ensure clean state between tests.
    """
    global _opik_connector_instance

    with _opik_connector_lock:
        _opik_connector_instance = None
