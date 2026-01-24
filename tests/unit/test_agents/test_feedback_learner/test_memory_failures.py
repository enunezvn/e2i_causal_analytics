"""
Unit tests for Feedback Learner Memory Failure Scenarios.
Version: 4.3

Tests graceful degradation when memory backends are unavailable.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestFeedbackStoreFailures:
    """Tests for feedback store unavailability."""

    @pytest.fixture
    def failing_feedback_store(self):
        """Create a feedback store that fails."""
        store = MagicMock()
        store.get_feedback = AsyncMock(
            side_effect=ConnectionError("Store connection refused")
        )
        store.count_pending = AsyncMock(
            side_effect=ConnectionError("Store connection refused")
        )
        return store

    @pytest.mark.asyncio
    async def test_agent_handles_store_connection_failure(self, failing_feedback_store):
        """Agent should handle feedback store connection failure."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        agent = FeedbackLearnerAgent(
            feedback_store=failing_feedback_store,
        )

        # Should not raise, should complete with error recorded
        result = await agent.learn(
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        assert result is not None
        # Should have errors or warnings about store failure
        assert result.status in ["failed", "partial", "completed"]

    @pytest.mark.asyncio
    async def test_agent_works_without_any_store(self):
        """Agent should work without any store configured."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        agent = FeedbackLearnerAgent(
            feedback_store=None,
            outcome_store=None,
            knowledge_stores=None,
        )

        result = await agent.learn(
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        assert result is not None
        assert result.feedback_count == 0  # No store, no feedback

    @pytest.mark.asyncio
    async def test_partial_store_failure_handled(self):
        """Agent should handle partial store failures."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        # Working feedback store
        working_store = MagicMock()
        working_store.get_feedback = AsyncMock(
            return_value=[
                {
                    "feedback_id": "fb_1",
                    "feedback_type": "rating",
                    "source_agent": "explainer",
                    "rating": 4,
                    "query": "test",
                    "agent_response": "response",
                    "timestamp": "2025-01-01T12:00:00Z",
                    "metadata": {},
                }
            ]
        )

        # Failing knowledge store
        failing_store = MagicMock()
        failing_store.update = AsyncMock(side_effect=Exception("Update failed"))

        agent = FeedbackLearnerAgent(
            feedback_store=working_store,
            knowledge_stores={"failing": failing_store},
        )

        result = await agent.learn(
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        # Should process feedback even with knowledge store failure
        assert result is not None
        assert result.feedback_count == 1


class TestKnowledgeStoreFailures:
    """Tests for knowledge store unavailability."""

    @pytest.mark.asyncio
    async def test_knowledge_update_failure_recovery(self):
        """Knowledge update failures should be recovered from."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        mock_store = MagicMock()
        mock_store.update = AsyncMock(side_effect=Exception("Update failed"))

        agent = FeedbackLearnerAgent(
            knowledge_stores={"test": mock_store},
        )

        result = await agent.learn(
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        # Should complete even with update failures
        assert result is not None
        # Updates should not be in applied list
        assert len(result.applied_updates) == 0

    @pytest.mark.asyncio
    async def test_multiple_store_failures(self):
        """Multiple store failures should be handled."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        stores = {}
        for name in ["experiments", "baselines", "configs"]:
            store = MagicMock()
            store.update = AsyncMock(side_effect=Exception(f"{name} failed"))
            stores[name] = store

        agent = FeedbackLearnerAgent(knowledge_stores=stores)

        result = await agent.learn(
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        assert result is not None


class TestOutcomeStoreFailures:
    """Tests for outcome store unavailability."""

    @pytest.mark.asyncio
    async def test_outcome_store_failure_handled(self):
        """Outcome store failure should be handled gracefully."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        failing_outcome_store = MagicMock()
        failing_outcome_store.get_outcomes = AsyncMock(
            side_effect=TimeoutError("Outcome store timeout")
        )

        agent = FeedbackLearnerAgent(outcome_store=failing_outcome_store)

        result = await agent.learn(
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        # Should complete without outcomes
        assert result is not None


class TestSchedulerMemoryFailures:
    """Tests for scheduler with memory failures."""

    @pytest.mark.asyncio
    async def test_scheduler_handles_store_failure(self):
        """Scheduler should handle store failures in cycles."""
        from src.agents.feedback_learner import (
            FeedbackLearnerAgent,
            FeedbackLearnerScheduler,
            SchedulerConfig,
        )

        # Agent with failing store
        failing_store = MagicMock()
        failing_store.get_feedback = AsyncMock(
            side_effect=ConnectionError("Store down")
        )

        agent = FeedbackLearnerAgent(feedback_store=failing_store)

        config = SchedulerConfig(
            initial_delay_seconds=0,
            min_feedback_threshold=0,
        )

        scheduler = FeedbackLearnerScheduler(agent, config)

        result = await scheduler.run_cycle_now(force=True)

        # Cycle should complete (possibly with failure status)
        assert result is not None
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_scheduler_continues_after_failure(self):
        """Scheduler should continue running after store failure."""
        from src.agents.feedback_learner import (
            FeedbackLearnerScheduler,
            SchedulerConfig,
        )

        call_count = 0

        async def mock_learn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("First call fails")
            output = MagicMock()
            output.status = "completed"
            output.feedback_count = 5
            output.pattern_count = 1
            output.recommendation_count = 1
            output.errors = []
            return output

        agent = MagicMock()
        agent._feedback_store = None
        agent.learn = mock_learn

        config = SchedulerConfig(
            initial_delay_seconds=0,
            min_feedback_threshold=0,
        )

        scheduler = FeedbackLearnerScheduler(agent, config)

        # Run multiple cycles
        result1 = await scheduler.run_cycle_now(force=True)
        result2 = await scheduler.run_cycle_now(force=True)

        # First should fail, second should succeed
        assert result1.success is False
        assert result2.success is True
        assert call_count == 2


class TestDSPyIntegrationFailures:
    """Tests for DSPy integration failures."""

    @pytest.mark.asyncio
    async def test_agent_works_without_dspy(self):
        """Agent should work when DSPy is not available."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        agent = FeedbackLearnerAgent(use_llm=False)

        result = await agent.learn(
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        assert result is not None
        assert result.dspy_available is not None  # Should indicate availability

    @pytest.mark.asyncio
    async def test_training_signal_failure_handled(self):
        """Training signal failures should be handled."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        agent = FeedbackLearnerAgent()

        result = await agent.learn(
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        assert result is not None
        # Training reward may be None if signal collection failed
        assert result.training_reward is None or isinstance(
            result.training_reward, float
        )


class TestGracefulDegradation:
    """Tests for graceful degradation behavior."""

    @pytest.mark.asyncio
    async def test_degraded_response_structure(self):
        """Degraded responses should maintain proper structure."""
        from src.agents.feedback_learner import (
            FeedbackLearnerAgent,
            FeedbackLearnerOutput,
        )

        agent = FeedbackLearnerAgent()

        result = await agent.learn(
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        assert isinstance(result, FeedbackLearnerOutput)
        assert hasattr(result, "batch_id")
        assert hasattr(result, "detected_patterns")
        assert hasattr(result, "learning_recommendations")
        assert hasattr(result, "status")
        assert hasattr(result, "errors")

    @pytest.mark.asyncio
    async def test_consistent_output_types(self):
        """Output types should be consistent regardless of failures."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        agent = FeedbackLearnerAgent()

        result = await agent.learn(
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        # List fields should be lists (even if empty)
        assert isinstance(result.detected_patterns, list)
        assert isinstance(result.learning_recommendations, list)
        assert isinstance(result.applied_updates, list)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)

        # Numeric fields should be numeric
        assert isinstance(result.feedback_count, int)
        assert isinstance(result.pattern_count, int)
        assert isinstance(result.total_latency_ms, int)

    @pytest.mark.asyncio
    async def test_no_data_loss_on_partial_failure(self):
        """Partial failures should not cause data loss."""
        from src.agents.feedback_learner import FeedbackLearnerAgent

        # Create store that works for reads but fails for writes
        partial_store = MagicMock()
        partial_store.get_feedback = AsyncMock(
            return_value=[
                {
                    "feedback_id": "fb_1",
                    "feedback_type": "rating",
                    "source_agent": "test",
                    "rating": 5,
                    "query": "test",
                    "agent_response": "response",
                    "timestamp": "2025-01-01T12:00:00Z",
                    "metadata": {},
                }
            ]
        )

        agent = FeedbackLearnerAgent(feedback_store=partial_store)

        result = await agent.learn(
            time_range_start="2025-01-01T00:00:00Z",
            time_range_end="2025-01-02T00:00:00Z",
        )

        # Feedback should be counted even if later steps fail
        assert result.feedback_count == 1
