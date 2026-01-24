"""
Unit tests for Feedback Learner Scheduler.
Version: 4.3

Tests async scheduling of feedback learning cycles.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.feedback_learner.scheduler import (
    CycleResult,
    FeedbackLearnerScheduler,
    SchedulerConfig,
    SchedulerMetrics,
    SchedulerState,
    create_scheduler,
)


class TestSchedulerConfig:
    """Tests for SchedulerConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = SchedulerConfig()

        assert config.interval_hours == 6.0
        assert config.initial_delay_seconds == 60.0
        assert config.min_feedback_threshold == 10
        assert config.max_batch_size == 1000
        assert config.cycle_timeout_seconds == 300.0
        assert config.shutdown_timeout_seconds == 60.0
        assert config.max_retries == 3
        assert config.focus_agents is None

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = SchedulerConfig(
            interval_hours=12.0,
            min_feedback_threshold=50,
            focus_agents=["explainer", "causal_impact"],
        )

        assert config.interval_hours == 12.0
        assert config.min_feedback_threshold == 50
        assert config.focus_agents == ["explainer", "causal_impact"]


class TestSchedulerMetrics:
    """Tests for SchedulerMetrics dataclass."""

    def test_default_metrics_values(self):
        """Test default metrics values."""
        metrics = SchedulerMetrics()

        assert metrics.total_cycles == 0
        assert metrics.successful_cycles == 0
        assert metrics.failed_cycles == 0
        assert metrics.skipped_cycles == 0
        assert metrics.total_feedback_processed == 0
        assert metrics.last_cycle_time is None
        assert metrics.last_error is None


class TestCycleResult:
    """Tests for CycleResult dataclass."""

    def test_cycle_result_creation(self):
        """Test creating a cycle result."""
        now = datetime.now(timezone.utc)
        result = CycleResult(
            cycle_id="test_cycle_1",
            started_at=now,
            success=True,
            feedback_count=50,
        )

        assert result.cycle_id == "test_cycle_1"
        assert result.started_at == now
        assert result.success is True
        assert result.feedback_count == 50
        assert result.skipped is False

    def test_cycle_result_skipped(self):
        """Test skipped cycle result."""
        result = CycleResult(
            cycle_id="test_cycle_2",
            started_at=datetime.now(timezone.utc),
            skipped=True,
            skip_reason="Insufficient feedback",
        )

        assert result.skipped is True
        assert result.skip_reason == "Insufficient feedback"
        assert result.success is False


class TestFeedbackLearnerScheduler:
    """Tests for FeedbackLearnerScheduler class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock FeedbackLearnerAgent."""
        agent = MagicMock()
        agent._feedback_store = None

        # Create async learn method
        async def mock_learn(*args, **kwargs):
            output = MagicMock()
            output.status = "completed"
            output.feedback_count = 25
            output.pattern_count = 3
            output.recommendation_count = 2
            output.errors = []
            return output

        agent.learn = mock_learn
        return agent

    @pytest.fixture
    def mock_agent_failing(self):
        """Create a mock agent that fails."""
        agent = MagicMock()
        agent._feedback_store = None

        async def mock_learn(*args, **kwargs):
            output = MagicMock()
            output.status = "failed"
            output.feedback_count = 0
            output.pattern_count = 0
            output.recommendation_count = 0
            output.errors = [{"message": "Test error"}]
            return output

        agent.learn = mock_learn
        return agent

    @pytest.fixture
    def scheduler(self, mock_agent):
        """Create scheduler with short intervals for testing."""
        config = SchedulerConfig(
            interval_hours=0.001,  # Very short for testing
            initial_delay_seconds=0,
            min_feedback_threshold=5,
            cycle_timeout_seconds=5.0,
        )
        return FeedbackLearnerScheduler(mock_agent, config)

    @pytest.fixture
    def scheduler_with_threshold(self, mock_agent):
        """Create scheduler with higher threshold."""
        config = SchedulerConfig(
            interval_hours=0.001,
            initial_delay_seconds=0,
            min_feedback_threshold=100,
        )
        return FeedbackLearnerScheduler(mock_agent, config)

    # Initialization tests
    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initializes correctly."""
        assert scheduler.state == SchedulerState.STOPPED
        assert scheduler.is_running is False
        assert scheduler.metrics.total_cycles == 0

    def test_scheduler_with_custom_config(self, mock_agent):
        """Test scheduler with custom configuration."""
        config = SchedulerConfig(
            interval_hours=12.0,
            min_feedback_threshold=50,
        )
        scheduler = FeedbackLearnerScheduler(mock_agent, config)

        assert scheduler._config.interval_hours == 12.0
        assert scheduler._config.min_feedback_threshold == 50

    # Start/Stop tests
    @pytest.mark.asyncio
    async def test_scheduler_start(self, scheduler):
        """Test scheduler starts correctly."""
        await scheduler.start()

        assert scheduler.state == SchedulerState.RUNNING
        assert scheduler.is_running is True

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_scheduler_stop(self, scheduler):
        """Test scheduler stops correctly."""
        await scheduler.start()
        await scheduler.stop()

        assert scheduler.state == SchedulerState.STOPPED
        assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_scheduler_double_start_ignored(self, scheduler):
        """Test that double start is handled gracefully."""
        await scheduler.start()
        await scheduler.start()  # Should not raise

        assert scheduler.state == SchedulerState.RUNNING

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_scheduler_double_stop_ignored(self, scheduler):
        """Test that double stop is handled gracefully."""
        await scheduler.start()
        await scheduler.stop()
        await scheduler.stop()  # Should not raise

        assert scheduler.state == SchedulerState.STOPPED

    # Pause/Resume tests
    @pytest.mark.asyncio
    async def test_scheduler_pause(self, scheduler):
        """Test scheduler pauses correctly."""
        await scheduler.start()
        await scheduler.pause()

        assert scheduler.state == SchedulerState.PAUSED

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_scheduler_resume(self, scheduler):
        """Test scheduler resumes correctly."""
        await scheduler.start()
        await scheduler.pause()
        await scheduler.resume()

        assert scheduler.state == SchedulerState.RUNNING

        await scheduler.stop()

    # Cycle execution tests
    @pytest.mark.asyncio
    async def test_run_cycle_now(self, scheduler):
        """Test running a cycle immediately."""
        result = await scheduler.run_cycle_now(force=True)

        assert result.success is True
        assert result.feedback_count == 25
        assert result.pattern_count == 3
        assert scheduler.metrics.total_cycles == 1
        assert scheduler.metrics.successful_cycles == 1

    @pytest.mark.asyncio
    async def test_run_cycle_now_with_threshold_check(self, scheduler):
        """Test cycle respects threshold when not forced."""
        # Mock feedback count to be below threshold
        scheduler._feedback_count_fn = lambda: 2  # Below threshold of 5

        result = await scheduler.run_cycle_now(force=False)

        assert result.skipped is True
        assert "Insufficient feedback" in result.skip_reason
        assert scheduler.metrics.skipped_cycles == 1

    @pytest.mark.asyncio
    async def test_run_cycle_now_forced_ignores_threshold(
        self, scheduler_with_threshold
    ):
        """Test forced cycle ignores threshold."""
        scheduler_with_threshold._feedback_count_fn = lambda: 5  # Below 100 threshold

        result = await scheduler_with_threshold.run_cycle_now(force=True)

        assert result.skipped is False
        assert result.success is True

    @pytest.mark.asyncio
    async def test_cycle_failure_recorded(self, mock_agent_failing):
        """Test failed cycle is recorded correctly."""
        config = SchedulerConfig(initial_delay_seconds=0, min_feedback_threshold=0)
        scheduler = FeedbackLearnerScheduler(mock_agent_failing, config)

        result = await scheduler.run_cycle_now(force=True)

        assert result.success is False
        assert scheduler.metrics.failed_cycles == 1

    @pytest.mark.asyncio
    async def test_cycle_timeout(self, mock_agent):
        """Test cycle timeout handling."""
        # Make agent.learn hang
        async def slow_learn(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout

        mock_agent.learn = slow_learn

        config = SchedulerConfig(
            initial_delay_seconds=0,
            min_feedback_threshold=0,
            cycle_timeout_seconds=0.1,  # Very short timeout
        )
        scheduler = FeedbackLearnerScheduler(mock_agent, config)

        result = await scheduler.run_cycle_now(force=True)

        assert result.success is False
        assert "timed out" in result.error.lower()

    # Metrics tracking tests
    @pytest.mark.asyncio
    async def test_metrics_updated_on_success(self, scheduler):
        """Test metrics are updated after successful cycle."""
        await scheduler.run_cycle_now(force=True)

        metrics = scheduler.metrics
        assert metrics.total_cycles == 1
        assert metrics.successful_cycles == 1
        assert metrics.total_feedback_processed == 25
        assert metrics.total_patterns_detected == 3
        assert metrics.total_recommendations == 2
        assert metrics.last_cycle_time is not None

    @pytest.mark.asyncio
    async def test_metrics_updated_on_skip(self, scheduler):
        """Test metrics updated when cycle skipped."""
        scheduler._feedback_count_fn = lambda: 0

        await scheduler.run_cycle_now(force=False)

        assert scheduler.metrics.total_cycles == 1
        assert scheduler.metrics.skipped_cycles == 1
        assert scheduler.metrics.successful_cycles == 0

    # Cycle history tests
    @pytest.mark.asyncio
    async def test_cycle_history_recorded(self, scheduler):
        """Test cycle history is recorded."""
        await scheduler.run_cycle_now(force=True)
        await scheduler.run_cycle_now(force=True)

        history = scheduler.cycle_history
        assert len(history) == 2
        assert all(c.success for c in history)

    @pytest.mark.asyncio
    async def test_cycle_history_bounded(self, scheduler):
        """Test cycle history is bounded to prevent memory issues."""
        # Run many cycles
        for _ in range(150):
            await scheduler.run_cycle_now(force=True)

        # History should be bounded
        assert len(scheduler.cycle_history) <= 100

    # Callback tests
    @pytest.mark.asyncio
    async def test_on_cycle_complete_callback(self, mock_agent):
        """Test callback is called on cycle completion."""
        callback_results = []

        def callback(result):
            callback_results.append(result)

        config = SchedulerConfig(initial_delay_seconds=0, min_feedback_threshold=0)
        scheduler = FeedbackLearnerScheduler(
            mock_agent, config, on_cycle_complete=callback
        )

        await scheduler.run_cycle_now(force=True)

        assert len(callback_results) == 1
        assert callback_results[0].success is True

    @pytest.mark.asyncio
    async def test_callback_error_handled(self, mock_agent):
        """Test callback errors are handled gracefully."""

        def bad_callback(result):
            raise ValueError("Callback error")

        config = SchedulerConfig(initial_delay_seconds=0, min_feedback_threshold=0)
        scheduler = FeedbackLearnerScheduler(
            mock_agent, config, on_cycle_complete=bad_callback
        )

        # Should not raise
        result = await scheduler.run_cycle_now(force=True)
        assert result.success is True

    # Feedback count function tests
    @pytest.mark.asyncio
    async def test_custom_feedback_count_function(self, mock_agent):
        """Test custom feedback count function is used."""
        count_calls = []

        def count_fn():
            count_calls.append(1)
            return 50

        config = SchedulerConfig(
            initial_delay_seconds=0,
            min_feedback_threshold=10,
        )
        scheduler = FeedbackLearnerScheduler(
            mock_agent, config, feedback_count_fn=count_fn
        )

        await scheduler.run_cycle_now(force=False)

        assert len(count_calls) == 1

    @pytest.mark.asyncio
    async def test_async_feedback_count_function(self, mock_agent):
        """Test async feedback count function is supported."""

        async def async_count_fn():
            await asyncio.sleep(0.01)
            return 50

        config = SchedulerConfig(
            initial_delay_seconds=0,
            min_feedback_threshold=10,
        )
        scheduler = FeedbackLearnerScheduler(
            mock_agent, config, feedback_count_fn=async_count_fn
        )

        result = await scheduler.run_cycle_now(force=False)

        assert result.skipped is False  # 50 >= 10

    # Status tests
    def test_get_status(self, scheduler):
        """Test get_status returns correct information."""
        status = scheduler.get_status()

        assert status["state"] == "stopped"
        assert "config" in status
        assert status["config"]["interval_hours"] == 0.001
        assert "metrics" in status
        assert status["metrics"]["total_cycles"] == 0
        assert "recent_cycles" in status

    @pytest.mark.asyncio
    async def test_get_status_after_cycles(self, scheduler):
        """Test get_status after running cycles."""
        await scheduler.run_cycle_now(force=True)
        await scheduler.run_cycle_now(force=True)

        status = scheduler.get_status()

        assert status["metrics"]["total_cycles"] == 2
        assert len(status["recent_cycles"]) == 2

    # Graceful shutdown tests
    @pytest.mark.asyncio
    async def test_graceful_shutdown_waits_for_cycle(self, mock_agent):
        """Test graceful shutdown waits for in-progress cycle."""
        cycle_started = asyncio.Event()
        cycle_can_complete = asyncio.Event()

        async def slow_learn(*args, **kwargs):
            cycle_started.set()
            await cycle_can_complete.wait()
            output = MagicMock()
            output.status = "completed"
            output.feedback_count = 10
            output.pattern_count = 1
            output.recommendation_count = 1
            output.errors = []
            return output

        mock_agent.learn = slow_learn

        config = SchedulerConfig(
            interval_hours=1,  # Long interval
            initial_delay_seconds=0,
            min_feedback_threshold=0,
            shutdown_timeout_seconds=5.0,
        )
        scheduler = FeedbackLearnerScheduler(mock_agent, config)

        await scheduler.start()

        # Trigger a cycle manually
        cycle_task = asyncio.create_task(scheduler.run_cycle_now(force=True))

        # Wait for cycle to start
        await cycle_started.wait()

        # Start shutdown in background
        async def shutdown():
            await scheduler.stop(wait_for_current=True)

        shutdown_task = asyncio.create_task(shutdown())

        # Allow cycle to complete
        await asyncio.sleep(0.1)
        cycle_can_complete.set()

        # Wait for both to complete
        await cycle_task
        await shutdown_task

        assert scheduler.state == SchedulerState.STOPPED


class TestCreateSchedulerFactory:
    """Tests for create_scheduler factory function."""

    def test_create_scheduler_basic(self):
        """Test basic scheduler creation."""
        agent = MagicMock()
        scheduler = create_scheduler(agent)

        assert isinstance(scheduler, FeedbackLearnerScheduler)
        assert scheduler._config.interval_hours == 6.0
        assert scheduler._config.min_feedback_threshold == 10

    def test_create_scheduler_custom_params(self):
        """Test scheduler creation with custom parameters."""
        agent = MagicMock()
        scheduler = create_scheduler(
            agent,
            interval_hours=12.0,
            min_feedback_threshold=50,
            focus_agents=["explainer"],
        )

        assert scheduler._config.interval_hours == 12.0
        assert scheduler._config.min_feedback_threshold == 50
        assert scheduler._config.focus_agents == ["explainer"]


class TestSchedulerIntegration:
    """Integration tests for scheduler with agent."""

    @pytest.mark.asyncio
    async def test_scheduler_runs_multiple_cycles(self):
        """Test scheduler can run multiple cycles in sequence."""
        call_count = 0

        async def mock_learn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            output = MagicMock()
            output.status = "completed"
            output.feedback_count = 10
            output.pattern_count = 1
            output.recommendation_count = 1
            output.errors = []
            return output

        agent = MagicMock()
        agent._feedback_store = None
        agent.learn = mock_learn

        config = SchedulerConfig(
            interval_hours=0.0001,  # Very short
            initial_delay_seconds=0,
            min_feedback_threshold=0,
        )
        scheduler = FeedbackLearnerScheduler(agent, config)

        await scheduler.start()
        await asyncio.sleep(0.1)  # Allow a few cycles
        await scheduler.stop()

        # Should have run at least one cycle
        assert call_count >= 1
        assert scheduler.metrics.successful_cycles >= 1
