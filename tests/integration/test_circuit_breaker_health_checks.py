"""Integration tests for circuit breaker health check behavior.

Tests verify circuit breaker behavior with dependency health checks:
- State transitions (closed -> open -> half-open -> closed)
- Multi-service circuit breaker scenarios
- Metrics tracking and accuracy
- State change callbacks
- Recovery workflows
- Concurrent access behavior

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import threading
import time
from unittest.mock import patch

import pytest

from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)

pytestmark = [pytest.mark.integration]


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def circuit_breaker():
    """Create a fresh circuit breaker with default config."""
    return CircuitBreaker()


@pytest.fixture
def fast_circuit_breaker():
    """Create a circuit breaker with fast reset for testing."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=0.5,  # Fast reset for tests
        half_open_max_calls=2,
        success_threshold=2,
    )
    return CircuitBreaker(config=config)


@pytest.fixture
def circuit_breaker_with_callback():
    """Create a circuit breaker with state change callback."""
    state_changes = []

    def on_state_change(old_state: CircuitState, new_state: CircuitState):
        state_changes.append((old_state, new_state, time.time()))

    config = CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=0.5,
        success_threshold=2,
    )
    cb = CircuitBreaker(config=config, on_state_change=on_state_change)
    cb._state_changes = state_changes  # Attach for test access
    return cb


# =============================================================================
# STATE TRANSITION TESTS
# =============================================================================


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_initial_state_is_closed(self, circuit_breaker):
        """Test that circuit breaker starts in closed state."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.is_closed
        assert not circuit_breaker.is_open
        assert not circuit_breaker.is_half_open

    def test_transitions_to_open_after_failures(self, fast_circuit_breaker):
        """Test that circuit opens after reaching failure threshold."""
        cb = fast_circuit_breaker

        # Record failures up to threshold
        for i in range(3):
            assert cb.allow_request()
            cb.record_failure()

        # Circuit should now be open
        assert cb.state == CircuitState.OPEN
        assert cb.is_open

    def test_open_circuit_blocks_requests(self, fast_circuit_breaker):
        """Test that open circuit blocks requests."""
        cb = fast_circuit_breaker

        # Trip the circuit
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        assert cb.is_open

        # Requests should be blocked
        assert not cb.allow_request()
        assert not cb.allow_request()

        # Rejected calls should be tracked
        assert cb.metrics.rejected_calls >= 2

    def test_transitions_to_half_open_after_timeout(self, fast_circuit_breaker):
        """Test that circuit transitions to half-open after timeout."""
        cb = fast_circuit_breaker

        # Trip the circuit
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        assert cb.is_open

        # Wait for reset timeout
        time.sleep(0.6)

        # Next request should be allowed (half-open)
        assert cb.allow_request()
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self, fast_circuit_breaker):
        """Test that half-open circuit closes after successful calls."""
        cb = fast_circuit_breaker

        # Trip the circuit
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        # Wait for reset timeout
        time.sleep(0.6)

        # Enter half-open state
        assert cb.allow_request()
        assert cb.is_half_open

        # Record successes to close circuit
        cb.record_success()
        assert cb.allow_request()
        cb.record_success()

        # Circuit should be closed now
        assert cb.is_closed

    def test_half_open_opens_on_failure(self, fast_circuit_breaker):
        """Test that half-open circuit opens on any failure."""
        cb = fast_circuit_breaker

        # Trip the circuit
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        # Wait for reset timeout
        time.sleep(0.6)

        # Enter half-open state
        assert cb.allow_request()
        assert cb.is_half_open

        # Single failure should open circuit again
        cb.record_failure()
        assert cb.is_open

    def test_success_resets_failure_count(self, fast_circuit_breaker):
        """Test that success resets consecutive failure count."""
        cb = fast_circuit_breaker

        # Record some failures (but not enough to trip)
        cb.allow_request()
        cb.record_failure()
        cb.allow_request()
        cb.record_failure()

        assert cb.consecutive_failures == 2

        # Success should reset
        cb.allow_request()
        cb.record_success()

        assert cb.consecutive_failures == 0
        assert cb.is_closed


# =============================================================================
# HEALTH CHECK INTEGRATION TESTS
# =============================================================================


@pytest.mark.asyncio
class TestRedisHealthCheckCircuitBreaker:
    """Tests for circuit breaker with Redis health check."""

    async def test_redis_health_check_uses_circuit_breaker(self):
        """Test that Redis health check respects circuit breaker state."""
        import src.api.dependencies.redis_client as redis_module

        # Reset circuit breaker
        redis_module._health_circuit_breaker.reset()

        # Force circuit open
        redis_module._health_circuit_breaker.force_open()

        # Health check should return circuit_open status
        result = await redis_module.redis_health_check()
        assert result["status"] == "circuit_open"

        # Reset for other tests
        redis_module._health_circuit_breaker.reset()

    async def test_redis_health_check_failures_trip_circuit(self):
        """Test that repeated health check failures trip the circuit."""
        import src.api.dependencies.redis_client as redis_module

        # Reset circuit breaker
        redis_module._health_circuit_breaker.reset()

        # Mock get_redis to fail
        async def failing_get_redis():
            raise ConnectionError("Simulated Redis failure")

        with patch.object(redis_module, "get_redis", failing_get_redis):
            # Make enough calls to trip circuit (threshold is 3)
            for _ in range(4):
                await redis_module.redis_health_check()

        # Circuit should be open
        assert redis_module._health_circuit_breaker.is_open

        # Restore and reset
        redis_module._health_circuit_breaker.reset()


@pytest.mark.asyncio
class TestFalkorDBHealthCheckCircuitBreaker:
    """Tests for circuit breaker with FalkorDB health check."""

    async def test_falkordb_health_check_uses_circuit_breaker(self):
        """Test that FalkorDB health check respects circuit breaker state."""
        import src.api.dependencies.falkordb_client as falkordb_module

        # Reset circuit breaker
        falkordb_module._health_circuit_breaker.reset()

        # Force circuit open
        falkordb_module._health_circuit_breaker.force_open()

        # Health check should return circuit_open status
        result = await falkordb_module.falkordb_health_check()
        assert result["status"] == "circuit_open"

        # Reset for other tests
        falkordb_module._health_circuit_breaker.reset()

    async def test_falkordb_health_check_failures_trip_circuit(self):
        """Test that repeated health check failures trip the circuit."""
        import src.api.dependencies.falkordb_client as falkordb_module

        # Reset circuit breaker
        falkordb_module._health_circuit_breaker.reset()

        # Mock get_falkordb to fail
        async def failing_get_falkordb():
            raise ConnectionError("Simulated FalkorDB failure")

        with patch.object(falkordb_module, "get_falkordb", failing_get_falkordb):
            # Make enough calls to trip circuit
            for _ in range(4):
                await falkordb_module.falkordb_health_check()

        # Circuit should be open
        assert falkordb_module._health_circuit_breaker.is_open

        # Reset
        falkordb_module._health_circuit_breaker.reset()


@pytest.mark.asyncio
class TestSupabaseHealthCheckCircuitBreaker:
    """Tests for circuit breaker with Supabase health check."""

    async def test_supabase_health_check_uses_circuit_breaker(self):
        """Test that Supabase health check respects circuit breaker state."""
        import src.api.dependencies.supabase_client as supabase_module

        # Reset circuit breaker
        supabase_module._health_circuit_breaker.reset()

        # Force circuit open
        supabase_module._health_circuit_breaker.force_open()

        # Health check should return circuit_open status
        result = await supabase_module.supabase_health_check()
        assert result["status"] == "circuit_open"

        # Reset for other tests
        supabase_module._health_circuit_breaker.reset()


# =============================================================================
# MULTI-SERVICE CIRCUIT BREAKER TESTS
# =============================================================================


@pytest.mark.asyncio
class TestMultiServiceCircuitBreakers:
    """Tests for multiple circuit breakers across services."""

    async def test_independent_circuit_breakers(self):
        """Test that each service has independent circuit breaker."""
        import src.api.dependencies.falkordb_client as falkordb_module
        import src.api.dependencies.redis_client as redis_module
        import src.api.dependencies.supabase_client as supabase_module

        # Reset all circuit breakers
        redis_module._health_circuit_breaker.reset()
        falkordb_module._health_circuit_breaker.reset()
        supabase_module._health_circuit_breaker.reset()

        # Trip only Redis circuit
        redis_module._health_circuit_breaker.force_open()

        # Redis should be open, others closed
        assert redis_module._health_circuit_breaker.is_open
        assert falkordb_module._health_circuit_breaker.is_closed
        assert supabase_module._health_circuit_breaker.is_closed

        # Reset
        redis_module._health_circuit_breaker.reset()

    async def test_multiple_services_failing(self):
        """Test scenario where multiple services fail."""
        import src.api.dependencies.falkordb_client as falkordb_module
        import src.api.dependencies.redis_client as redis_module
        import src.api.dependencies.supabase_client as supabase_module

        # Reset all
        redis_module._health_circuit_breaker.reset()
        falkordb_module._health_circuit_breaker.reset()
        supabase_module._health_circuit_breaker.reset()

        # Trip all circuits
        redis_module._health_circuit_breaker.force_open()
        falkordb_module._health_circuit_breaker.force_open()
        supabase_module._health_circuit_breaker.force_open()

        # All health checks should return circuit_open
        redis_result = await redis_module.redis_health_check()
        falkordb_result = await falkordb_module.falkordb_health_check()
        supabase_result = await supabase_module.supabase_health_check()

        assert redis_result["status"] == "circuit_open"
        assert falkordb_result["status"] == "circuit_open"
        assert supabase_result["status"] == "circuit_open"

        # Reset all
        redis_module._health_circuit_breaker.reset()
        falkordb_module._health_circuit_breaker.reset()
        supabase_module._health_circuit_breaker.reset()

    async def test_partial_service_recovery(self):
        """Test partial recovery where some services recover."""
        import src.api.dependencies.falkordb_client as falkordb_module
        import src.api.dependencies.redis_client as redis_module

        # Reset all
        redis_module._health_circuit_breaker.reset()
        falkordb_module._health_circuit_breaker.reset()

        # Trip both circuits
        redis_module._health_circuit_breaker.force_open()
        falkordb_module._health_circuit_breaker.force_open()

        # "Recover" only Redis
        redis_module._health_circuit_breaker.reset()

        # Redis should be available, FalkorDB still open
        assert redis_module._health_circuit_breaker.is_closed
        assert falkordb_module._health_circuit_breaker.is_open

        # Reset
        falkordb_module._health_circuit_breaker.reset()


# =============================================================================
# METRICS TESTS
# =============================================================================


class TestCircuitBreakerMetrics:
    """Tests for circuit breaker metrics tracking."""

    def test_metrics_track_successful_calls(self, circuit_breaker):
        """Test that metrics track successful calls."""
        cb = circuit_breaker

        for _ in range(5):
            cb.allow_request()
            cb.record_success()

        metrics = cb.metrics
        assert metrics.total_calls == 5
        assert metrics.successful_calls == 5
        assert metrics.failed_calls == 0
        assert metrics.last_success_time is not None

    def test_metrics_track_failed_calls(self, circuit_breaker):
        """Test that metrics track failed calls."""
        cb = circuit_breaker

        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        metrics = cb.metrics
        assert metrics.total_calls == 3
        assert metrics.failed_calls == 3
        assert metrics.last_failure_time is not None

    def test_metrics_track_rejected_calls(self, fast_circuit_breaker):
        """Test that metrics track rejected calls."""
        cb = fast_circuit_breaker

        # Trip circuit
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        # Try more requests (should be rejected)
        for _ in range(5):
            cb.allow_request()

        assert cb.metrics.rejected_calls == 5

    def test_metrics_track_state_changes(self, fast_circuit_breaker):
        """Test that metrics track state changes."""
        cb = fast_circuit_breaker

        initial_changes = cb.metrics.state_changes

        # Trip circuit (closed -> open)
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        assert cb.metrics.state_changes == initial_changes + 1
        assert cb.metrics.times_opened == 1

    def test_metrics_to_dict(self, circuit_breaker):
        """Test that metrics can be serialized to dict."""
        cb = circuit_breaker

        cb.allow_request()
        cb.record_success()
        cb.allow_request()
        cb.record_failure()

        metrics_dict = cb.metrics.to_dict()

        assert "total_calls" in metrics_dict
        assert "successful_calls" in metrics_dict
        assert "failed_calls" in metrics_dict
        assert "success_rate" in metrics_dict
        assert metrics_dict["total_calls"] == 2
        assert metrics_dict["success_rate"] == 0.5

    def test_get_status_includes_all_info(self, fast_circuit_breaker):
        """Test that get_status returns comprehensive info."""
        cb = fast_circuit_breaker

        status = cb.get_status()

        assert "state" in status
        assert "is_closed" in status
        assert "is_open" in status
        assert "is_half_open" in status
        assert "consecutive_failures" in status
        assert "failure_threshold" in status
        assert "reset_timeout_seconds" in status
        assert "metrics" in status


# =============================================================================
# STATE CHANGE CALLBACK TESTS
# =============================================================================


class TestCircuitBreakerCallbacks:
    """Tests for circuit breaker state change callbacks."""

    def test_callback_invoked_on_state_change(self, circuit_breaker_with_callback):
        """Test that callback is invoked when state changes."""
        cb = circuit_breaker_with_callback

        # Trip circuit (should trigger callback)
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        # Check callback was invoked
        assert len(cb._state_changes) >= 1
        old_state, new_state, _ = cb._state_changes[-1]
        assert old_state == CircuitState.CLOSED
        assert new_state == CircuitState.OPEN

    def test_callback_invoked_on_recovery(self, circuit_breaker_with_callback):
        """Test that callback is invoked when circuit recovers."""
        cb = circuit_breaker_with_callback

        # Trip circuit
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        initial_callbacks = len(cb._state_changes)

        # Wait for reset timeout
        time.sleep(0.6)

        # Enter half-open
        cb.allow_request()

        # Record successes to close
        cb.record_success()
        cb.allow_request()
        cb.record_success()

        # Should have more callbacks (open -> half_open -> closed)
        assert len(cb._state_changes) > initial_callbacks

    def test_callback_receives_correct_states(self, circuit_breaker_with_callback):
        """Test that callback receives correct old and new states."""
        cb = circuit_breaker_with_callback

        # Trip circuit
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        # First callback: closed -> open
        assert cb._state_changes[0][0] == CircuitState.CLOSED
        assert cb._state_changes[0][1] == CircuitState.OPEN


# =============================================================================
# MANUAL CONTROL TESTS
# =============================================================================


class TestCircuitBreakerManualControl:
    """Tests for manual circuit breaker control."""

    def test_manual_reset(self, fast_circuit_breaker):
        """Test manual reset of circuit breaker."""
        cb = fast_circuit_breaker

        # Trip circuit
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        assert cb.is_open

        # Manual reset
        cb.reset()

        assert cb.is_closed
        assert cb.consecutive_failures == 0

    def test_force_open(self, circuit_breaker):
        """Test forcing circuit open."""
        cb = circuit_breaker

        assert cb.is_closed

        cb.force_open()

        assert cb.is_open

    def test_reset_from_half_open(self, fast_circuit_breaker):
        """Test reset from half-open state."""
        cb = fast_circuit_breaker

        # Trip circuit
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        # Wait for half-open
        time.sleep(0.6)
        cb.allow_request()

        assert cb.is_half_open

        # Reset
        cb.reset()

        assert cb.is_closed


# =============================================================================
# CONCURRENT ACCESS TESTS
# =============================================================================


class TestCircuitBreakerConcurrency:
    """Tests for circuit breaker thread safety."""

    def test_concurrent_allow_request(self, circuit_breaker):
        """Test concurrent allow_request calls."""
        cb = circuit_breaker
        results = []
        errors = []

        def check_request():
            try:
                for _ in range(100):
                    result = cb.allow_request()
                    results.append(result)
                    if result:
                        cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_request) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0
        # All requests should have been allowed (circuit stayed closed)
        assert all(results)

    def test_concurrent_failures(self):
        """Test concurrent failure recording."""
        config = CircuitBreakerConfig(
            failure_threshold=50,  # High threshold
            reset_timeout_seconds=30.0,
        )
        cb = CircuitBreaker(config=config)
        errors = []

        def record_failures():
            try:
                for _ in range(100):
                    cb.allow_request()
                    cb.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_failures) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0
        # All failures should be recorded
        assert cb.metrics.failed_calls == 500

    def test_concurrent_state_transitions(self):
        """Test concurrent operations during state transitions."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            reset_timeout_seconds=0.1,
            success_threshold=2,
        )
        cb = CircuitBreaker(config=config)
        errors = []

        def mixed_operations():
            try:
                for _ in range(50):
                    if cb.allow_request():
                        if time.time() % 2 < 1:
                            cb.record_success()
                        else:
                            cb.record_failure()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_operations) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0


# =============================================================================
# RECOVERY WORKFLOW TESTS
# =============================================================================


class TestCircuitBreakerRecovery:
    """Tests for circuit breaker recovery workflows."""

    def test_full_recovery_cycle(self, fast_circuit_breaker):
        """Test full recovery cycle: closed -> open -> half-open -> closed."""
        cb = fast_circuit_breaker

        # Start closed
        assert cb.is_closed

        # Trip to open
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        assert cb.is_open

        # Wait for reset timeout
        time.sleep(0.6)

        # Transition to half-open
        assert cb.allow_request()
        assert cb.is_half_open

        # Recover with successes
        cb.record_success()
        cb.allow_request()
        cb.record_success()

        # Back to closed
        assert cb.is_closed

    def test_failed_recovery_attempt(self, fast_circuit_breaker):
        """Test failed recovery: open -> half-open -> open."""
        cb = fast_circuit_breaker

        # Trip to open
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        # Wait for reset timeout
        time.sleep(0.6)

        # Enter half-open
        cb.allow_request()
        assert cb.is_half_open

        # Fail in half-open
        cb.record_failure()

        # Back to open
        assert cb.is_open

    def test_multiple_recovery_attempts(self, fast_circuit_breaker):
        """Test multiple recovery attempts before success."""
        cb = fast_circuit_breaker

        for attempt in range(3):
            # Trip or already open
            if cb.is_closed:
                for _ in range(3):
                    cb.allow_request()
                    cb.record_failure()

            # Wait for reset
            time.sleep(0.6)

            # Try recovery
            cb.allow_request()
            assert cb.is_half_open

            if attempt < 2:
                # Fail recovery
                cb.record_failure()
                assert cb.is_open
            else:
                # Succeed on third attempt
                cb.record_success()
                cb.allow_request()
                cb.record_success()
                assert cb.is_closed


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestCircuitBreakerEdgeCases:
    """Tests for circuit breaker edge cases."""

    def test_immediate_success_after_failure(self, fast_circuit_breaker):
        """Test that immediate success resets failure count."""
        cb = fast_circuit_breaker

        cb.allow_request()
        cb.record_failure()
        cb.allow_request()
        cb.record_failure()

        assert cb.consecutive_failures == 2

        cb.allow_request()
        cb.record_success()

        assert cb.consecutive_failures == 0

    def test_half_open_max_calls_limit(self, fast_circuit_breaker):
        """Test that half-open limits concurrent test calls."""
        cb = fast_circuit_breaker

        # Trip circuit
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        # Wait for reset
        time.sleep(0.6)

        # Half-open allows limited calls (max_calls=2)
        # First call triggers OPEN -> HALF_OPEN transition (doesn't count)
        assert cb.allow_request()  # Transition call
        assert cb.is_half_open
        # Next calls count toward the limit
        assert cb.allow_request()  # 1st counted
        assert cb.allow_request()  # 2nd counted (at limit)
        assert not cb.allow_request()  # Blocked (exceeded limit)

    def test_metrics_persistence_across_transitions(self, fast_circuit_breaker):
        """Test that metrics persist across state transitions."""
        cb = fast_circuit_breaker

        # Record some activity
        for _ in range(2):
            cb.allow_request()
            cb.record_success()

        initial_successes = cb.metrics.successful_calls

        # Trip circuit
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        # Metrics should include both successes and failures
        assert cb.metrics.successful_calls == initial_successes
        assert cb.metrics.failed_calls == 3

        # Reset and check metrics preserved
        cb.reset()
        assert cb.metrics.successful_calls == initial_successes
        assert cb.metrics.failed_calls == 3

    def test_time_until_reset_calculation(self, fast_circuit_breaker):
        """Test time_until_reset calculation."""
        cb = fast_circuit_breaker

        # Trip circuit
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()

        # Check time remaining
        status = cb.get_status()
        time_until = status["time_until_reset"]

        assert time_until is not None
        assert 0 < time_until <= 0.5

        # Wait a bit and check again
        time.sleep(0.2)
        status = cb.get_status()
        assert status["time_until_reset"] < time_until

    def test_zero_failure_threshold(self):
        """Test behavior with zero failure threshold (edge case)."""
        # This is an unusual config but shouldn't crash
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config)

        # Single failure should trip circuit
        cb.allow_request()
        cb.record_failure()

        assert cb.is_open
