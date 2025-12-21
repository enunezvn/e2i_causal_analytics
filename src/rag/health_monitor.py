"""
E2I Hybrid RAG - Health Monitor

Implements health monitoring and circuit breaker patterns for RAG backends.
Tracks health status, manages circuit breaker state, and provides
health check endpoints.

Part of Phase 1, Checkpoint 1.5.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional

from src.rag.config import HealthMonitorConfig
from src.rag.exceptions import CircuitBreakerOpenError, HealthCheckError
from src.rag.types import BackendHealth, BackendStatus

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker implementation for backend protection.

    Prevents cascading failures by temporarily blocking requests
    to unhealthy backends.

    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Backend failing, requests rejected immediately
        - HALF_OPEN: Testing if backend recovered, limited requests

    Example:
        ```python
        breaker = CircuitBreaker(
            name="vector",
            failure_threshold=3,
            reset_timeout_seconds=60.0
        )

        async def do_search():
            if not breaker.can_execute():
                raise CircuitBreakerOpenError("vector")
            try:
                result = await backend.search(...)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        ```
    """

    name: str
    failure_threshold: int = 3
    success_threshold: int = 2  # Successes needed to close from half-open
    reset_timeout_seconds: float = 60.0

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED)
    _failure_count: int = field(default=0)
    _success_count: int = field(default=0)
    _last_failure_time: Optional[float] = field(default=None)
    _last_state_change: float = field(default_factory=time.time)

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for automatic transitions."""
        if self._state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def can_execute(self) -> bool:
        """Check if a request should be allowed through."""
        current_state = self.state
        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.HALF_OPEN:
            return True  # Allow test requests
        else:  # OPEN
            return False

    def record_success(self) -> None:
        """Record a successful request."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        else:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Immediately go back to open on failure in half-open
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def force_open(self) -> None:
        """Manually force the circuit to open state."""
        self._last_failure_time = time.time()  # Ensure reset timeout is honored
        self._transition_to(CircuitState.OPEN)

    def force_close(self) -> None:
        """Manually force the circuit to closed state."""
        self._transition_to(CircuitState.CLOSED)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.reset_timeout_seconds

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0

        logger.info(
            f"Circuit breaker '{self.name}' transitioned: "
            f"{old_state.value} -> {new_state.value}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "last_state_change": self._last_state_change,
            "time_in_current_state": time.time() - self._last_state_change,
        }

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name='{self.name}', state={self.state.value}, "
            f"failures={self._failure_count})"
        )


class HealthMonitor:
    """
    Monitors health of RAG backends and manages circuit breakers.

    Provides:
        - Periodic health checks
        - Circuit breaker management
        - Health status aggregation
        - Health endpoints for API

    Example:
        ```python
        monitor = HealthMonitor(
            config=HealthMonitorConfig(),
            backends={
                "vector": vector_backend,
                "fulltext": fulltext_backend,
                "graph": graph_backend
            }
        )

        # Start monitoring (runs in background)
        await monitor.start()

        # Check if a backend is available
        if monitor.is_available("vector"):
            results = await vector_backend.search(...)

        # Get overall health
        health = await monitor.get_health_status()

        # Stop monitoring
        await monitor.stop()
        ```
    """

    def __init__(
        self,
        config: Optional[HealthMonitorConfig] = None,
        backends: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the health monitor.

        Args:
            config: Health monitoring configuration
            backends: Dictionary of backend name -> backend instance
        """
        self.config = config or HealthMonitorConfig()
        self._backends: Dict[str, Any] = backends or {}

        # Circuit breakers for each backend
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Health status for each backend
        self._health_status: Dict[str, BackendHealth] = {}

        # Background task handle
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        # Initialize circuit breakers for configured backends
        for name in self._backends:
            self._circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=self.config.unhealthy_consecutive_failures,
                reset_timeout_seconds=self.config.circuit_breaker_reset_seconds,
            )
            self._health_status[name] = BackendHealth(
                status=BackendStatus.UNKNOWN, latency_ms=0.0, last_check=datetime.now(timezone.utc)
            )

    def register_backend(self, name: str, backend: Any) -> None:
        """
        Register a backend for health monitoring.

        Args:
            name: Backend identifier
            backend: Backend instance (must have health_check method)
        """
        self._backends[name] = backend
        self._circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=self.config.unhealthy_consecutive_failures,
            reset_timeout_seconds=self.config.circuit_breaker_reset_seconds,
        )
        self._health_status[name] = BackendHealth(
            status=BackendStatus.UNKNOWN, latency_ms=0.0, last_check=datetime.now(timezone.utc)
        )
        logger.info(f"Registered backend for health monitoring: {name}")

    def is_available(self, backend_name: str) -> bool:
        """
        Check if a backend is available for requests.

        Args:
            backend_name: Backend to check

        Returns:
            True if backend can accept requests
        """
        if backend_name not in self._circuit_breakers:
            return False

        breaker = self._circuit_breakers[backend_name]
        health = self._health_status.get(backend_name)

        if not breaker.can_execute():
            return False

        if health and not health.is_available():
            return False

        return True

    def get_circuit_breaker(self, backend_name: str) -> Optional[CircuitBreaker]:
        """Get the circuit breaker for a backend."""
        return self._circuit_breakers.get(backend_name)

    async def check_backend_health(self, backend_name: str) -> BackendHealth:
        """
        Perform a health check on a specific backend.

        Args:
            backend_name: Backend to check

        Returns:
            BackendHealth status
        """
        backend = self._backends.get(backend_name)
        if not backend:
            return BackendHealth(
                status=BackendStatus.UNKNOWN,
                latency_ms=0.0,
                last_check=datetime.now(timezone.utc),
                error_message=f"Backend '{backend_name}' not registered",
            )

        try:
            # Call backend's health_check method
            if not hasattr(backend, "health_check"):
                raise HealthCheckError(
                    message=f"Backend '{backend_name}' has no health_check method"
                )

            result = await backend.health_check()

            latency_ms = result.get("latency_ms", 0.0)
            error = result.get("error")

            # Determine status based on result
            if error:
                status = BackendStatus.UNHEALTHY
                consecutive_failures = (
                    self._health_status.get(
                        backend_name,
                        BackendHealth(
                            status=BackendStatus.UNKNOWN,
                            latency_ms=0.0,
                            last_check=datetime.now(timezone.utc),
                        ),
                    ).consecutive_failures
                    + 1
                )
            elif latency_ms > self.config.degraded_latency_ms:
                status = BackendStatus.DEGRADED
                consecutive_failures = 0
            else:
                status = BackendStatus.HEALTHY
                consecutive_failures = 0

            health = BackendHealth(
                status=status,
                latency_ms=latency_ms,
                last_check=datetime.now(timezone.utc),
                consecutive_failures=consecutive_failures,
                error_message=error,
            )

            # Update stored health
            self._health_status[backend_name] = health

            # Update circuit breaker
            breaker = self._circuit_breakers.get(backend_name)
            if breaker:
                if status == BackendStatus.UNHEALTHY:
                    breaker.record_failure()
                else:
                    breaker.record_success()

            return health

        except Exception as e:
            logger.error(f"Health check failed for {backend_name}: {e}")
            health = BackendHealth(
                status=BackendStatus.UNHEALTHY,
                latency_ms=0.0,
                last_check=datetime.now(timezone.utc),
                consecutive_failures=self._health_status.get(
                    backend_name,
                    BackendHealth(
                        status=BackendStatus.UNKNOWN,
                        latency_ms=0.0,
                        last_check=datetime.now(timezone.utc),
                    ),
                ).consecutive_failures
                + 1,
                error_message=str(e),
            )
            self._health_status[backend_name] = health

            breaker = self._circuit_breakers.get(backend_name)
            if breaker:
                breaker.record_failure()

            return health

    async def check_all_backends(self) -> Dict[str, BackendHealth]:
        """
        Perform health checks on all registered backends.

        Returns:
            Dictionary of backend name -> BackendHealth
        """
        results = {}
        for name in self._backends:
            results[name] = await self.check_backend_health(name)
        return results

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get aggregated health status for all backends.

        Returns:
            Dictionary with overall status and per-backend details
        """
        # Determine overall status
        all_healthy = True
        any_available = False
        backends_status = {}

        for name, health in self._health_status.items():
            breaker = self._circuit_breakers.get(name)

            backends_status[name] = {
                "status": health.status.value,
                "latency_ms": health.latency_ms,
                "last_check": health.last_check.isoformat(),
                "consecutive_failures": health.consecutive_failures,
                "error": health.error_message,
                "circuit_breaker": breaker.get_stats() if breaker else None,
                "available": self.is_available(name),
            }

            if health.status != BackendStatus.HEALTHY:
                all_healthy = False
            if self.is_available(name):
                any_available = True

        if all_healthy:
            overall_status = "healthy"
        elif any_available:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        return {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backends": backends_status,
            "monitoring_enabled": self._running,
        }

    async def start(self) -> None:
        """Start background health monitoring."""
        if self._running:
            logger.warning("Health monitor already running")
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started health monitoring")

    async def stop(self) -> None:
        """Stop background health monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Stopped health monitoring")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await self.check_all_backends()

                # Alert on status changes if configured
                for name, health in self._health_status.items():
                    if health.status == BackendStatus.DEGRADED and self.config.alert_on_degraded:
                        logger.warning(f"Backend '{name}' is degraded: {health.error_message}")
                    elif (
                        health.status == BackendStatus.UNHEALTHY and self.config.alert_on_unhealthy
                    ):
                        logger.error(f"Backend '{name}' is unhealthy: {health.error_message}")

                await asyncio.sleep(self.config.check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.config.check_interval_seconds)

    async def wrap_with_circuit_breaker(
        self, backend_name: str, operation: Callable[..., Coroutine], *args, **kwargs
    ) -> Any:
        """
        Execute an operation with circuit breaker protection.

        Args:
            backend_name: Backend to protect
            operation: Async function to execute
            *args, **kwargs: Arguments for the operation

        Returns:
            Result from the operation

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception from the operation
        """
        if not self.config.circuit_breaker_enabled:
            return await operation(*args, **kwargs)

        breaker = self._circuit_breakers.get(backend_name)
        if not breaker:
            return await operation(*args, **kwargs)

        if not breaker.can_execute():
            # Calculate remaining reset time
            remaining = breaker.reset_timeout_seconds
            if breaker._last_failure_time:
                elapsed = time.time() - breaker._last_failure_time
                remaining = max(0, breaker.reset_timeout_seconds - elapsed)
            raise CircuitBreakerOpenError(backend=backend_name, reset_time_seconds=remaining)

        try:
            result = await operation(*args, **kwargs)
            breaker.record_success()
            return result
        except Exception:
            breaker.record_failure()
            raise

    def __repr__(self) -> str:
        backends = list(self._backends.keys())
        return f"HealthMonitor(backends={backends}, running={self._running})"
