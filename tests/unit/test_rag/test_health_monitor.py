"""
Unit tests for E2I RAG Health Monitor.

Tests for CircuitBreaker and HealthMonitor classes.
All external dependencies are mocked.
"""

import asyncio
import pytest
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.rag.health_monitor import CircuitBreaker, CircuitState, HealthMonitor
from src.rag.config import HealthMonitorConfig
from src.rag.types import BackendStatus, BackendHealth
from src.rag.exceptions import CircuitBreakerOpenError


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def circuit_breaker():
    """Create a CircuitBreaker with default settings."""
    return CircuitBreaker(
        name="test_backend",
        failure_threshold=3,
        success_threshold=2,
        reset_timeout_seconds=60.0
    )


@pytest.fixture
def mock_backend():
    """Create a mock backend with health_check method."""
    backend = MagicMock()
    backend.health_check = AsyncMock(return_value={
        "status": "healthy",
        "latency_ms": 50.0,
        "error": None
    })
    return backend


@pytest.fixture
def health_monitor_config():
    """Create a HealthMonitorConfig for testing."""
    return HealthMonitorConfig(
        check_interval_seconds=1.0,
        degraded_latency_ms=1000.0,
        unhealthy_consecutive_failures=3,
        circuit_breaker_enabled=True,
        circuit_breaker_reset_seconds=5.0
    )


@pytest.fixture
def health_monitor(health_monitor_config, mock_backend):
    """Create a HealthMonitor with mock backends."""
    monitor = HealthMonitor(
        config=health_monitor_config,
        backends={"test": mock_backend}
    )
    return monitor


# ============================================================================
# Circuit Breaker Tests
# ============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_is_closed(self, circuit_breaker):
        """Test that circuit breaker starts in closed state."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.can_execute() is True

    def test_successful_requests_keep_circuit_closed(self, circuit_breaker):
        """Test that successful requests keep the circuit closed."""
        for _ in range(10):
            circuit_breaker.record_success()

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failure_count == 0

    def test_failures_open_circuit_after_threshold(self, circuit_breaker):
        """Test that failures open the circuit after reaching threshold."""
        for _ in range(3):  # failure_threshold = 3
            circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.can_execute() is False

    def test_circuit_transitions_to_half_open_after_timeout(self, circuit_breaker):
        """Test that circuit transitions to half-open after reset timeout."""
        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitState.OPEN

        # Manually set last failure time to past
        circuit_breaker._last_failure_time = time.time() - 120  # 2 minutes ago

        # State property should transition to half-open
        assert circuit_breaker.state == CircuitState.HALF_OPEN
        assert circuit_breaker.can_execute() is True

    def test_half_open_success_closes_circuit(self, circuit_breaker):
        """Test that successes in half-open state close the circuit."""
        # Get to half-open state
        for _ in range(3):
            circuit_breaker.record_failure()
        circuit_breaker._last_failure_time = time.time() - 120
        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # Record successful requests
        circuit_breaker.record_success()
        circuit_breaker.record_success()  # success_threshold = 2

        assert circuit_breaker.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self, circuit_breaker):
        """Test that failure in half-open state reopens the circuit."""
        # Get to half-open state
        for _ in range(3):
            circuit_breaker.record_failure()
        circuit_breaker._last_failure_time = time.time() - 120
        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # Record a failure
        circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitState.OPEN

    def test_success_resets_failure_count(self, circuit_breaker):
        """Test that success resets failure count when closed."""
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        assert circuit_breaker._failure_count == 2

        circuit_breaker.record_success()
        assert circuit_breaker._failure_count == 0

    def test_force_open(self, circuit_breaker):
        """Test forcing circuit to open state."""
        assert circuit_breaker.state == CircuitState.CLOSED

        circuit_breaker.force_open()

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.can_execute() is False

    def test_force_close(self, circuit_breaker):
        """Test forcing circuit to closed state."""
        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitState.OPEN

        circuit_breaker.force_close()

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failure_count == 0

    def test_get_stats(self, circuit_breaker):
        """Test getting circuit breaker statistics."""
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()

        stats = circuit_breaker.get_stats()

        assert stats["name"] == "test_backend"
        assert stats["state"] == CircuitState.CLOSED.value
        assert stats["failure_count"] == 2
        assert "last_state_change" in stats
        assert "time_in_current_state" in stats

    def test_repr(self, circuit_breaker):
        """Test string representation."""
        repr_str = repr(circuit_breaker)
        assert "CircuitBreaker" in repr_str
        assert "test_backend" in repr_str
        assert "closed" in repr_str


# ============================================================================
# Health Monitor Tests
# ============================================================================


class TestHealthMonitor:
    """Tests for HealthMonitor class."""

    def test_initialization(self, health_monitor, mock_backend):
        """Test health monitor initialization."""
        assert "test" in health_monitor._backends
        assert "test" in health_monitor._circuit_breakers
        assert "test" in health_monitor._health_status

    def test_register_backend(self, health_monitor):
        """Test registering a new backend."""
        new_backend = MagicMock()
        new_backend.health_check = AsyncMock()

        health_monitor.register_backend("new_backend", new_backend)

        assert "new_backend" in health_monitor._backends
        assert "new_backend" in health_monitor._circuit_breakers
        assert "new_backend" in health_monitor._health_status

    def test_is_available_healthy(self, health_monitor):
        """Test is_available returns True for healthy backend."""
        # Set health status to healthy
        health_monitor._health_status["test"] = BackendHealth(
            status=BackendStatus.HEALTHY,
            latency_ms=50.0,
            last_check=datetime.now(timezone.utc)
        )

        assert health_monitor.is_available("test") is True

    def test_is_available_degraded(self, health_monitor):
        """Test is_available returns True for degraded backend."""
        health_monitor._health_status["test"] = BackendHealth(
            status=BackendStatus.DEGRADED,
            latency_ms=1500.0,
            last_check=datetime.now(timezone.utc)
        )

        assert health_monitor.is_available("test") is True

    def test_is_available_unhealthy(self, health_monitor):
        """Test is_available returns False for unhealthy backend."""
        health_monitor._health_status["test"] = BackendHealth(
            status=BackendStatus.UNHEALTHY,
            latency_ms=0.0,
            last_check=datetime.now(timezone.utc),
            error_message="Connection failed"
        )

        assert health_monitor.is_available("test") is False

    def test_is_available_circuit_open(self, health_monitor):
        """Test is_available returns False when circuit is open."""
        # Force circuit open
        health_monitor._circuit_breakers["test"].force_open()

        assert health_monitor.is_available("test") is False

    def test_is_available_unknown_backend(self, health_monitor):
        """Test is_available returns False for unknown backend."""
        assert health_monitor.is_available("unknown") is False

    @pytest.mark.asyncio
    async def test_check_backend_health_success(self, health_monitor, mock_backend):
        """Test health check for healthy backend."""
        health = await health_monitor.check_backend_health("test")

        assert health.status == BackendStatus.HEALTHY
        assert health.latency_ms == 50.0
        assert health.error_message is None
        mock_backend.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_backend_health_degraded(self, health_monitor, mock_backend):
        """Test health check for degraded backend (high latency)."""
        mock_backend.health_check.return_value = {
            "status": "healthy",
            "latency_ms": 2000.0,  # Above degraded threshold
            "error": None
        }

        health = await health_monitor.check_backend_health("test")

        assert health.status == BackendStatus.DEGRADED
        assert health.latency_ms == 2000.0

    @pytest.mark.asyncio
    async def test_check_backend_health_unhealthy(self, health_monitor, mock_backend):
        """Test health check for unhealthy backend."""
        mock_backend.health_check.return_value = {
            "status": "unhealthy",
            "latency_ms": 0,
            "error": "Connection refused"
        }

        health = await health_monitor.check_backend_health("test")

        assert health.status == BackendStatus.UNHEALTHY
        assert health.error_message == "Connection refused"

    @pytest.mark.asyncio
    async def test_check_backend_health_exception(self, health_monitor, mock_backend):
        """Test health check when exception is raised."""
        mock_backend.health_check.side_effect = Exception("Connection failed")

        health = await health_monitor.check_backend_health("test")

        assert health.status == BackendStatus.UNHEALTHY
        assert "Connection failed" in health.error_message

    @pytest.mark.asyncio
    async def test_check_backend_health_unknown_backend(self, health_monitor):
        """Test health check for unregistered backend."""
        health = await health_monitor.check_backend_health("unknown")

        assert health.status == BackendStatus.UNKNOWN
        assert "not registered" in health.error_message

    @pytest.mark.asyncio
    async def test_check_all_backends(self, health_monitor, mock_backend):
        """Test checking health of all backends."""
        results = await health_monitor.check_all_backends()

        assert "test" in results
        assert results["test"].status == BackendStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_get_health_status(self, health_monitor, mock_backend):
        """Test getting aggregated health status."""
        # First check health
        await health_monitor.check_backend_health("test")

        status = await health_monitor.get_health_status()

        assert status["status"] == "healthy"
        assert "backends" in status
        assert "test" in status["backends"]
        assert status["backends"]["test"]["status"] == "healthy"
        assert "circuit_breaker" in status["backends"]["test"]

    @pytest.mark.asyncio
    async def test_get_health_status_degraded(self, health_monitor, mock_backend):
        """Test health status when one backend is degraded."""
        mock_backend.health_check.return_value = {
            "status": "healthy",
            "latency_ms": 2000.0,
            "error": None
        }
        await health_monitor.check_backend_health("test")

        status = await health_monitor.get_health_status()

        assert status["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_get_health_status_unhealthy(self, health_monitor, mock_backend):
        """Test health status when all backends are unhealthy."""
        mock_backend.health_check.return_value = {
            "status": "unhealthy",
            "latency_ms": 0,
            "error": "Failed"
        }
        await health_monitor.check_backend_health("test")

        status = await health_monitor.get_health_status()

        assert status["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_wrap_with_circuit_breaker_success(self, health_monitor):
        """Test wrapping operation with circuit breaker - success."""
        async def mock_operation():
            return "success"

        result = await health_monitor.wrap_with_circuit_breaker(
            "test",
            mock_operation
        )

        assert result == "success"
        # Circuit should record success
        assert health_monitor._circuit_breakers["test"]._failure_count == 0

    @pytest.mark.asyncio
    async def test_wrap_with_circuit_breaker_failure(self, health_monitor):
        """Test wrapping operation with circuit breaker - failure."""
        async def mock_operation():
            raise ValueError("Operation failed")

        with pytest.raises(ValueError):
            await health_monitor.wrap_with_circuit_breaker(
                "test",
                mock_operation
            )

        # Circuit should record failure
        assert health_monitor._circuit_breakers["test"]._failure_count == 1

    @pytest.mark.asyncio
    async def test_wrap_with_circuit_breaker_open(self, health_monitor):
        """Test wrapping operation when circuit is open."""
        # Force circuit open
        health_monitor._circuit_breakers["test"].force_open()

        async def mock_operation():
            return "success"

        with pytest.raises(CircuitBreakerOpenError):
            await health_monitor.wrap_with_circuit_breaker(
                "test",
                mock_operation
            )

    @pytest.mark.asyncio
    async def test_wrap_with_circuit_breaker_disabled(self, health_monitor_config, mock_backend):
        """Test wrapping when circuit breaker is disabled."""
        health_monitor_config.circuit_breaker_enabled = False
        monitor = HealthMonitor(
            config=health_monitor_config,
            backends={"test": mock_backend}
        )
        monitor._circuit_breakers["test"].force_open()

        async def mock_operation():
            return "success"

        # Should still execute even though circuit is "open"
        result = await monitor.wrap_with_circuit_breaker(
            "test",
            mock_operation
        )

        assert result == "success"

    @pytest.mark.asyncio
    async def test_start_and_stop(self, health_monitor):
        """Test starting and stopping health monitoring."""
        assert health_monitor._running is False

        await health_monitor.start()
        assert health_monitor._running is True
        assert health_monitor._monitor_task is not None

        await health_monitor.stop()
        assert health_monitor._running is False
        assert health_monitor._monitor_task is None

    @pytest.mark.asyncio
    async def test_start_idempotent(self, health_monitor):
        """Test that start is idempotent."""
        await health_monitor.start()
        task1 = health_monitor._monitor_task

        await health_monitor.start()
        task2 = health_monitor._monitor_task

        # Should be same task (not restarted)
        assert task1 is task2

        await health_monitor.stop()

    def test_get_circuit_breaker(self, health_monitor):
        """Test getting circuit breaker for a backend."""
        breaker = health_monitor.get_circuit_breaker("test")
        assert breaker is not None
        assert breaker.name == "test"

        unknown_breaker = health_monitor.get_circuit_breaker("unknown")
        assert unknown_breaker is None

    def test_repr(self, health_monitor):
        """Test string representation."""
        repr_str = repr(health_monitor)
        assert "HealthMonitor" in repr_str
        assert "test" in repr_str
        assert "running=False" in repr_str


# ============================================================================
# Integration Tests
# ============================================================================


class TestHealthMonitorIntegration:
    """Integration tests for HealthMonitor with multiple backends."""

    @pytest.fixture
    def multi_backend_monitor(self, health_monitor_config):
        """Create a monitor with multiple backends."""
        backends = {}
        for name in ["vector", "fulltext", "graph"]:
            backend = MagicMock()
            backend.health_check = AsyncMock(return_value={
                "status": "healthy",
                "latency_ms": 50.0,
                "error": None
            })
            backends[name] = backend

        return HealthMonitor(config=health_monitor_config, backends=backends)

    @pytest.mark.asyncio
    async def test_check_all_backends_multiple(self, multi_backend_monitor):
        """Test checking health of multiple backends."""
        results = await multi_backend_monitor.check_all_backends()

        assert len(results) == 3
        assert all(r.status == BackendStatus.HEALTHY for r in results.values())

    @pytest.mark.asyncio
    async def test_partial_failure(self, multi_backend_monitor):
        """Test behavior when some backends fail."""
        # Make graph backend unhealthy
        multi_backend_monitor._backends["graph"].health_check.return_value = {
            "status": "unhealthy",
            "latency_ms": 0,
            "error": "Connection refused"
        }

        await multi_backend_monitor.check_all_backends()
        status = await multi_backend_monitor.get_health_status()

        assert status["status"] == "degraded"  # Some healthy, some not
        assert multi_backend_monitor.is_available("vector") is True
        assert multi_backend_monitor.is_available("graph") is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, multi_backend_monitor):
        """Test that circuit breaker opens after consecutive failures."""
        # Make fulltext backend fail consistently
        multi_backend_monitor._backends["fulltext"].health_check.return_value = {
            "status": "unhealthy",
            "latency_ms": 0,
            "error": "Timeout"
        }

        # Check health 3 times (threshold)
        for _ in range(3):
            await multi_backend_monitor.check_backend_health("fulltext")

        # Circuit should be open
        assert multi_backend_monitor._circuit_breakers["fulltext"].state == CircuitState.OPEN
        assert multi_backend_monitor.is_available("fulltext") is False
