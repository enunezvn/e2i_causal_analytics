"""Circuit Breaker pattern for protecting external service calls.

Extracted from src.mlops.opik_connector for reuse across the codebase.
Used by health-check functions for Redis, FalkorDB, Supabase, and Opik.

Author: E2I Causal Analytics Team
Version: 4.3.0
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


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
                self.successful_calls / self.total_calls if self.total_calls > 0 else 1.0
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
        logger.info(f"Circuit breaker: {old_state.value} -> {new_state.value}")

        # Invoke callback
        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state)
            except Exception as e:
                logger.warning(f"Circuit breaker callback failed: {e}")
