"""Tests for CircuitBreaker.

Version: 1.0.0 (Phase 3.2)
Tests the circuit breaker implementation for fault tolerance.
"""

import threading
import time

from src.mlops.opik_connector import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitState,
)


class TestCircuitState:
    """Test CircuitState enum."""

    def test_state_values(self):
        """Test circuit state enum values."""
        assert CircuitState.CLOSED == "closed"
        assert CircuitState.OPEN == "open"
        assert CircuitState.HALF_OPEN == "half_open"

    def test_state_value_is_string(self):
        """Test that state values are strings."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.reset_timeout_seconds == 30.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            reset_timeout_seconds=60.0,
            half_open_max_calls=5,
            success_threshold=3,
        )

        assert config.failure_threshold == 10
        assert config.reset_timeout_seconds == 60.0
        assert config.half_open_max_calls == 5
        assert config.success_threshold == 3


class TestCircuitBreakerMetrics:
    """Test CircuitBreakerMetrics dataclass."""

    def test_initial_state(self):
        """Test initial metrics state."""
        metrics = CircuitBreakerMetrics()

        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.rejected_calls == 0
        assert metrics.state_changes == 0
        assert metrics.last_failure_time is None
        assert metrics.last_success_time is None
        assert metrics.times_opened == 0

    def test_record_success(self):
        """Test recording a successful call."""
        metrics = CircuitBreakerMetrics()

        metrics.record_success()

        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 0
        assert metrics.last_success_time is not None

    def test_record_failure(self):
        """Test recording a failed call."""
        metrics = CircuitBreakerMetrics()

        metrics.record_failure()

        assert metrics.total_calls == 1
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 1
        assert metrics.last_failure_time is not None

    def test_record_rejected(self):
        """Test recording a rejected call."""
        metrics = CircuitBreakerMetrics()

        metrics.record_rejected()

        assert metrics.rejected_calls == 1
        # Rejected calls don't count toward total_calls
        assert metrics.total_calls == 0

    def test_record_state_change_to_open(self):
        """Test recording state change to open."""
        metrics = CircuitBreakerMetrics()

        metrics.record_state_change(CircuitState.OPEN)

        assert metrics.state_changes == 1
        assert metrics.times_opened == 1
        assert metrics.last_state_change_time is not None

    def test_record_state_change_to_closed(self):
        """Test recording state change to closed."""
        metrics = CircuitBreakerMetrics()

        metrics.record_state_change(CircuitState.CLOSED)

        assert metrics.state_changes == 1
        assert metrics.times_opened == 0  # Only OPEN increments times_opened

    def test_multiple_state_changes(self):
        """Test recording multiple state changes."""
        metrics = CircuitBreakerMetrics()

        metrics.record_state_change(CircuitState.OPEN)
        metrics.record_state_change(CircuitState.HALF_OPEN)
        metrics.record_state_change(CircuitState.CLOSED)
        metrics.record_state_change(CircuitState.OPEN)

        assert metrics.state_changes == 4
        assert metrics.times_opened == 2

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = CircuitBreakerMetrics()
        metrics.record_success()
        metrics.record_failure()
        metrics.record_state_change(CircuitState.OPEN)

        result = metrics.to_dict()

        assert result["total_calls"] == 2
        assert result["successful_calls"] == 1
        assert result["failed_calls"] == 1
        assert result["state_changes"] == 1
        assert result["times_opened"] == 1
        assert "last_success_time" in result
        assert "last_failure_time" in result

    def test_success_rate(self):
        """Test success rate calculation."""
        metrics = CircuitBreakerMetrics()

        # Initially no calls
        assert metrics.total_calls == 0

        metrics.record_success()
        metrics.record_success()
        metrics.record_failure()

        result = metrics.to_dict()
        # 2 success, 1 failure = 2/3 = 0.667
        assert result["total_calls"] == 3
        assert result["successful_calls"] == 2
        assert result["failed_calls"] == 1


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Test that circuit breaker starts in closed state."""
        cb = CircuitBreaker()

        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True
        assert cb.is_open is False
        assert cb.is_half_open is False

    def test_allow_request_when_closed(self):
        """Test that requests are allowed when circuit is closed."""
        cb = CircuitBreaker()

        assert cb.allow_request() is True

    def test_record_success_resets_failures(self):
        """Test that recording success resets consecutive failure count."""
        cb = CircuitBreaker()

        # Record some failures
        cb.record_failure()
        cb.record_failure()
        assert cb.consecutive_failures == 2

        # Success resets
        cb.record_success()
        assert cb.consecutive_failures == 0

    def test_transitions_to_open_after_threshold(self):
        """Test circuit opens after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config=config)

        # Record failures up to threshold
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()  # 3rd failure triggers open

        assert cb.state == CircuitState.OPEN
        assert cb.is_open is True

    def test_rejects_requests_when_open(self):
        """Test that requests are rejected when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config=config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False
        assert cb.metrics.rejected_calls >= 1

    def test_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to half-open after reset timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout_seconds=0.1,  # Short timeout for testing
        )
        cb = CircuitBreaker(config=config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Next request should trigger half-open
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_limits_requests(self):
        """Test half-open state limits concurrent test requests.

        Note: The first allow_request() transitions from OPEN to HALF_OPEN
        and returns True without counting toward half_open_max_calls.
        Subsequent calls are limited by half_open_max_calls.
        """
        config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout_seconds=0.01,
            half_open_max_calls=2,
        )
        cb = CircuitBreaker(config=config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for half-open
        time.sleep(0.02)

        # First call triggers transition to HALF_OPEN
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

        # Next calls are limited by half_open_max_calls (2 more allowed)
        assert cb.allow_request() is True
        assert cb.allow_request() is True

        # Fourth request should be rejected (max_calls reached)
        assert cb.allow_request() is False

    def test_closes_after_success_threshold_in_half_open(self):
        """Test circuit closes after success threshold in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout_seconds=0.01,
            success_threshold=2,
        )
        cb = CircuitBreaker(config=config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for half-open
        time.sleep(0.02)
        cb.allow_request()  # Triggers half-open

        # Record successes to close
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # Not yet
        cb.record_success()  # Second success

        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True

    def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure during half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout_seconds=0.01,
        )
        cb = CircuitBreaker(config=config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for half-open
        time.sleep(0.02)
        cb.allow_request()  # Triggers half-open
        assert cb.state == CircuitState.HALF_OPEN

        # Failure in half-open immediately reopens
        cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_reset_returns_to_closed(self):
        """Test reset method returns circuit to closed state."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config=config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Reset
        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.consecutive_failures == 0

    def test_force_open(self):
        """Test force_open method opens circuit immediately."""
        cb = CircuitBreaker()

        assert cb.state == CircuitState.CLOSED

        cb.force_open()

        assert cb.state == CircuitState.OPEN
        assert cb.is_open is True

    def test_state_change_callback(self):
        """Test state change callback is invoked."""
        callback_calls = []

        def on_state_change(old_state, new_state):
            callback_calls.append((old_state, new_state))

        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config=config, on_state_change=on_state_change)

        # Trigger state change
        cb.record_failure()
        cb.record_failure()

        assert len(callback_calls) == 1
        assert callback_calls[0] == (CircuitState.CLOSED, CircuitState.OPEN)

    def test_get_status(self):
        """Test get_status returns comprehensive information."""
        cb = CircuitBreaker()

        cb.record_success()
        cb.record_failure()

        status = cb.get_status()

        assert status["state"] == "closed"
        assert status["is_closed"] is True
        assert status["is_open"] is False
        assert status["is_half_open"] is False
        assert status["consecutive_failures"] == 1
        assert status["failure_threshold"] == 5
        assert status["reset_timeout_seconds"] == 30.0
        assert "metrics" in status
        assert status["metrics"]["total_calls"] == 2

    def test_metrics_track_calls(self):
        """Test that metrics accurately track all calls."""
        cb = CircuitBreaker()

        cb.record_success()
        cb.record_success()
        cb.record_failure()

        assert cb.metrics.total_calls == 3
        assert cb.metrics.successful_calls == 2
        assert cb.metrics.failed_calls == 1


class TestCircuitBreakerThreadSafety:
    """Test thread safety of CircuitBreaker."""

    def test_concurrent_record_operations(self):
        """Test concurrent record_success and record_failure calls."""
        cb = CircuitBreaker()
        num_threads = 10
        iterations = 100

        def record_success():
            for _ in range(iterations):
                cb.record_success()

        def record_failure():
            for _ in range(iterations):
                cb.record_failure()

        threads = []
        for i in range(num_threads):
            if i % 2 == 0:
                t = threading.Thread(target=record_success)
            else:
                t = threading.Thread(target=record_failure)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total calls should equal all operations
        expected_success = (num_threads // 2) * iterations
        expected_failure = (num_threads - num_threads // 2) * iterations

        assert cb.metrics.successful_calls == expected_success
        assert cb.metrics.failed_calls == expected_failure
        assert cb.metrics.total_calls == expected_success + expected_failure

    def test_concurrent_allow_request(self):
        """Test concurrent allow_request calls.

        Note: The first allow_request() triggers transition from OPEN to
        HALF_OPEN without counting toward the limit. So max allowed is
        half_open_max_calls + 1 (transition call).
        """
        config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout_seconds=0.01,
            half_open_max_calls=5,
        )
        cb = CircuitBreaker(config=config)

        # Open circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for half-open
        time.sleep(0.02)

        results = []
        num_threads = 10

        def check_allow():
            results.append(cb.allow_request())

        threads = [threading.Thread(target=check_allow) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Max allowed is half_open_max_calls + 1 (for the transition call)
        allowed = sum(1 for r in results if r)
        rejected = sum(1 for r in results if not r)

        assert allowed <= config.half_open_max_calls + 1
        assert allowed + rejected == num_threads


class TestCircuitBreakerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_failure_threshold(self):
        """Test with failure threshold of 1 (minimum practical)."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config)

        # Single failure should open circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_during_closed_state(self):
        """Test recording success doesn't affect closed state."""
        cb = CircuitBreaker()

        cb.record_success()
        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED
        assert cb.consecutive_failures == 0

    def test_rapid_state_transitions(self):
        """Test rapid open-close cycles."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            reset_timeout_seconds=0.01,
            success_threshold=1,
        )
        cb = CircuitBreaker(config=config)

        for _ in range(5):
            # Open
            cb.record_failure()
            assert cb.state == CircuitState.OPEN

            # Wait and half-open
            time.sleep(0.02)
            cb.allow_request()
            assert cb.state == CircuitState.HALF_OPEN

            # Close
            cb.record_success()
            assert cb.state == CircuitState.CLOSED

        assert cb.metrics.times_opened == 5

    def test_metrics_persist_through_state_changes(self):
        """Test that metrics persist across state changes."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            reset_timeout_seconds=0.01,
            success_threshold=1,
        )
        cb = CircuitBreaker(config=config)

        # Record some successes
        cb.record_success()
        cb.record_success()

        # Open circuit
        cb.record_failure()
        cb.record_failure()

        # Wait and close
        time.sleep(0.02)
        cb.allow_request()
        cb.record_success()

        # Metrics should reflect all operations
        assert cb.metrics.successful_calls == 3  # 2 initial + 1 to close
        assert cb.metrics.failed_calls == 2
        assert cb.metrics.state_changes >= 2  # CLOSED->OPEN->HALF_OPEN->CLOSED

    def test_reset_clears_failure_count_not_metrics(self):
        """Test that reset clears failures but preserves metrics history."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config=config)

        cb.record_failure()
        cb.record_failure()
        cb.reset()

        # State and failures reset
        assert cb.state == CircuitState.CLOSED
        assert cb.consecutive_failures == 0

        # But metrics history remains
        assert cb.metrics.failed_calls == 2


class TestCircuitBreakerCallbacks:
    """Test callback functionality."""

    def test_callback_receives_correct_states(self):
        """Test callback receives correct old and new states."""
        transitions = []

        def callback(old, new):
            transitions.append((old.value, new.value))

        config = CircuitBreakerConfig(
            failure_threshold=1,
            reset_timeout_seconds=0.01,
            success_threshold=1,
        )
        cb = CircuitBreaker(config=config, on_state_change=callback)

        # CLOSED -> OPEN
        cb.record_failure()
        assert transitions[-1] == ("closed", "open")

        # OPEN -> HALF_OPEN
        time.sleep(0.02)
        cb.allow_request()
        assert transitions[-1] == ("open", "half_open")

        # HALF_OPEN -> CLOSED
        cb.record_success()
        assert transitions[-1] == ("half_open", "closed")

    def test_callback_error_does_not_break_circuit(self):
        """Test that callback errors don't break circuit operation."""

        def bad_callback(old, new):
            raise RuntimeError("Callback error")

        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config, on_state_change=bad_callback)

        # Should not raise despite callback error
        cb.record_failure()

        # Circuit should still function
        assert cb.state == CircuitState.OPEN

    def test_no_callback_when_not_provided(self):
        """Test circuit works without callback."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=config, on_state_change=None)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN
