"""
Unit Tests for TrainingSignalReceiver.

Tests for the signal receiver that buffers and processes training signals
from Tier 2 agents in the feedback_learner.

Verifies:
- Signal buffering by agent
- Signal retrieval (all, by agent, by reward)
- FeedbackItem conversion
- Training data extraction
- Statistics tracking

Run: pytest tests/unit/test_agents/test_tier2_signal_routing/test_signal_receiver.py -v
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Dict, Any, List

# Mark all tests in this module for the tier2 xdist group
pytestmark = pytest.mark.xdist_group(name="tier2_signal_routing")


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_signals() -> Dict[str, List[Dict[str, Any]]]:
    """Create sample signals from all Tier 2 agents."""
    return {
        "causal_impact": [
            {
                "signal_id": "ci_001",
                "source_agent": "causal_impact",
                "dspy_type": "sender",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reward": 0.75,
                "input_context": {"query": "Impact of marketing?"},
                "output": {"ate": 0.15},
            },
            {
                "signal_id": "ci_002",
                "source_agent": "causal_impact",
                "dspy_type": "sender",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reward": 0.60,
                "input_context": {"query": "Impact of pricing?"},
                "output": {"ate": 0.22},
            },
        ],
        "gap_analyzer": [
            {
                "signal_id": "ga_001",
                "source_agent": "gap_analyzer",
                "dspy_type": "sender",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reward": 0.82,
                "input_context": {"query": "ROI in northeast?"},
                "output": {"gaps": 5},
            },
        ],
        "heterogeneous_optimizer": [
            {
                "signal_id": "ho_001",
                "source_agent": "heterogeneous_optimizer",
                "dspy_type": "sender",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reward": 0.45,
                "input_context": {"query": "Best segments?"},
                "output": {"segments": 8},
            },
        ],
    }


@pytest.fixture
def signal_receiver():
    """Create a fresh signal receiver instance."""
    from src.agents.feedback_learner.dspy_receiver import TrainingSignalReceiver
    return TrainingSignalReceiver()


@pytest.fixture(autouse=True)
def reset_receiver_singleton():
    """Reset the receiver singleton before and after each test."""
    from src.agents.feedback_learner.dspy_receiver import reset_signal_receiver
    reset_signal_receiver()
    yield
    reset_signal_receiver()


# =============================================================================
# SIGNAL BUFFERING TESTS
# =============================================================================


class TestSignalBuffering:
    """Tests for signal buffering by agent."""

    @pytest.mark.asyncio
    async def test_receive_signals_from_multiple_agents(
        self, signal_receiver, sample_signals
    ):
        """Should buffer signals from multiple agents."""
        received = await signal_receiver.receive_signals(sample_signals)

        assert received == 4  # 2 + 1 + 1
        stats = signal_receiver.get_statistics()
        assert stats["signals_received"] == 4
        assert stats["buffer_sizes"]["causal_impact"] == 2
        assert stats["buffer_sizes"]["gap_analyzer"] == 1
        assert stats["buffer_sizes"]["heterogeneous_optimizer"] == 1

    @pytest.mark.asyncio
    async def test_buffer_respects_max_per_agent(self):
        """Should respect max signals per agent."""
        from src.agents.feedback_learner.dspy_receiver import TrainingSignalReceiver

        receiver = TrainingSignalReceiver(max_signals_per_agent=3)

        # Add more signals than max
        signals = {
            "causal_impact": [
                {"signal_id": f"ci_{i}", "reward": 0.5}
                for i in range(5)
            ]
        }

        await receiver.receive_signals(signals)

        stats = receiver.get_statistics()
        # Buffer is capped at 3
        assert stats["buffer_sizes"]["causal_impact"] == 3

    @pytest.mark.asyncio
    async def test_unknown_agent_signals_skipped(self, signal_receiver):
        """Signals from unknown agents should be skipped."""
        signals = {
            "unknown_agent": [
                {"signal_id": "unk_001", "reward": 0.5}
            ],
            "causal_impact": [
                {"signal_id": "ci_001", "reward": 0.6}
            ],
        }

        received = await signal_receiver.receive_signals(signals)

        # Only causal_impact signal received
        assert received == 1


# =============================================================================
# SIGNAL RETRIEVAL TESTS
# =============================================================================


class TestSignalRetrieval:
    """Tests for signal retrieval."""

    @pytest.mark.asyncio
    async def test_get_pending_signals_all_agents(
        self, signal_receiver, sample_signals
    ):
        """Should get pending signals from all agents."""
        await signal_receiver.receive_signals(sample_signals)

        signals = await signal_receiver.get_pending_signals(limit=10)

        assert len(signals) == 4
        # All should be unprocessed
        assert all(not s.processed for s in signals)

    @pytest.mark.asyncio
    async def test_get_pending_signals_by_agent(
        self, signal_receiver, sample_signals
    ):
        """Should filter signals by agent."""
        await signal_receiver.receive_signals(sample_signals)

        signals = await signal_receiver.get_pending_signals(
            limit=10, agent="causal_impact"
        )

        assert len(signals) == 2
        assert all(s.source_agent == "causal_impact" for s in signals)

    @pytest.mark.asyncio
    async def test_get_pending_signals_by_min_reward(
        self, signal_receiver, sample_signals
    ):
        """Should filter signals by minimum reward."""
        await signal_receiver.receive_signals(sample_signals)

        signals = await signal_receiver.get_pending_signals(
            limit=10, min_reward=0.7
        )

        # Only ci_001 (0.75) and ga_001 (0.82) should pass
        assert len(signals) == 2
        assert all(s.signal_data["reward"] >= 0.7 for s in signals)

    @pytest.mark.asyncio
    async def test_get_pending_signals_respects_limit(
        self, signal_receiver, sample_signals
    ):
        """Should respect limit parameter."""
        await signal_receiver.receive_signals(sample_signals)

        signals = await signal_receiver.get_pending_signals(limit=2)

        assert len(signals) == 2


# =============================================================================
# MARK PROCESSED TESTS
# =============================================================================


class TestMarkProcessed:
    """Tests for marking signals as processed."""

    @pytest.mark.asyncio
    async def test_mark_processed(self, signal_receiver, sample_signals):
        """Should mark signals as processed."""
        await signal_receiver.receive_signals(sample_signals)

        pending = await signal_receiver.get_pending_signals(limit=2)
        count = await signal_receiver.mark_processed(pending)

        assert count == 2

        # Now should only get 2 pending
        remaining = await signal_receiver.get_pending_signals(limit=10)
        assert len(remaining) == 2

    @pytest.mark.asyncio
    async def test_processed_signals_not_returned(
        self, signal_receiver, sample_signals
    ):
        """Processed signals should not be returned by get_pending."""
        await signal_receiver.receive_signals(sample_signals)

        # Get and process all
        all_signals = await signal_receiver.get_pending_signals(limit=10)
        await signal_receiver.mark_processed(all_signals)

        # Should get no pending signals
        pending = await signal_receiver.get_pending_signals(limit=10)
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_clear_processed_removes_signals(
        self, signal_receiver, sample_signals
    ):
        """clear_processed should remove processed signals."""
        await signal_receiver.receive_signals(sample_signals)

        # Process some
        pending = await signal_receiver.get_pending_signals(limit=2)
        await signal_receiver.mark_processed(pending)

        # Clear processed
        removed = signal_receiver.clear_processed()
        assert removed == 2

        # Check buffer sizes
        stats = signal_receiver.get_statistics()
        assert stats["total_buffered"] == 2  # 4 - 2 removed


# =============================================================================
# FEEDBACK ITEM CONVERSION TESTS
# =============================================================================


class TestFeedbackItemConversion:
    """Tests for converting signals to FeedbackItem format."""

    @pytest.mark.asyncio
    async def test_get_signals_as_feedback_items(
        self, signal_receiver, sample_signals
    ):
        """Should convert signals to FeedbackItem format."""
        await signal_receiver.receive_signals(sample_signals)

        feedback_items = signal_receiver.get_signals_as_feedback_items(limit=10)

        assert len(feedback_items) == 4

        for item in feedback_items:
            # Required FeedbackItem fields
            assert "feedback_id" in item
            assert "timestamp" in item
            assert "feedback_type" in item
            assert item["feedback_type"] == "training_signal"
            assert "source_agent" in item
            assert "query" in item
            assert "user_feedback" in item
            assert "metadata" in item

    @pytest.mark.asyncio
    async def test_feedback_item_has_reward(
        self, signal_receiver, sample_signals
    ):
        """FeedbackItem should include reward from signal."""
        await signal_receiver.receive_signals(sample_signals)

        feedback_items = signal_receiver.get_signals_as_feedback_items(limit=10)

        for item in feedback_items:
            assert "reward" in item["user_feedback"]
            assert 0.0 <= item["user_feedback"]["reward"] <= 1.0

    @pytest.mark.asyncio
    async def test_feedback_item_has_raw_signal(
        self, signal_receiver, sample_signals
    ):
        """FeedbackItem metadata should include raw signal."""
        await signal_receiver.receive_signals(sample_signals)

        feedback_items = signal_receiver.get_signals_as_feedback_items(limit=1)

        assert len(feedback_items) == 1
        assert "raw_signal" in feedback_items[0]["metadata"]


# =============================================================================
# TRAINING DATA EXTRACTION TESTS
# =============================================================================


class TestTrainingDataExtraction:
    """Tests for extracting training data for DSPy optimization."""

    @pytest.mark.asyncio
    async def test_get_training_data_filters_by_reward(
        self, signal_receiver, sample_signals
    ):
        """Should filter training data by minimum reward."""
        await signal_receiver.receive_signals(sample_signals)

        # Get training data with min_reward=0.7
        training_data = signal_receiver.get_training_data_for_optimization(
            agent="causal_impact",
            min_reward=0.7,
            limit=10,
        )

        # Only ci_001 (0.75) should pass
        assert len(training_data) == 1
        assert training_data[0]["reward"] >= 0.7

    @pytest.mark.asyncio
    async def test_get_training_data_sorted_by_reward(
        self, signal_receiver, sample_signals
    ):
        """Training data should be sorted by reward (best first)."""
        await signal_receiver.receive_signals(sample_signals)

        training_data = signal_receiver.get_training_data_for_optimization(
            agent="causal_impact",
            min_reward=0.0,
            limit=10,
        )

        assert len(training_data) == 2
        # Should be sorted descending by reward
        assert training_data[0]["reward"] >= training_data[1]["reward"]

    @pytest.mark.asyncio
    async def test_get_training_data_unknown_agent(self, signal_receiver):
        """Unknown agent should return empty list."""
        training_data = signal_receiver.get_training_data_for_optimization(
            agent="unknown_agent",
            min_reward=0.0,
            limit=10,
        )

        assert training_data == []


# =============================================================================
# STATISTICS TESTS
# =============================================================================


class TestStatistics:
    """Tests for statistics tracking."""

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, signal_receiver, sample_signals):
        """Should track receive and process statistics."""
        await signal_receiver.receive_signals(sample_signals)

        pending = await signal_receiver.get_pending_signals(limit=2)
        await signal_receiver.mark_processed(pending)

        stats = signal_receiver.get_statistics()

        assert stats["signals_received"] == 4
        assert stats["signals_processed"] == 2
        assert stats["total_buffered"] == 4
        assert "buffer_sizes" in stats


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_receive_training_signals(self, sample_signals):
        """receive_training_signals should use singleton receiver."""
        from src.agents.feedback_learner.dspy_receiver import (
            receive_training_signals,
            get_signal_receiver,
            reset_signal_receiver,
        )

        reset_signal_receiver()

        received = await receive_training_signals(sample_signals)
        assert received == 4

        receiver = get_signal_receiver()
        stats = receiver.get_statistics()
        assert stats["signals_received"] == 4

        reset_signal_receiver()

    def test_get_pending_training_signals(self, sample_signals):
        """get_pending_training_signals should return signal dicts."""
        from src.agents.feedback_learner.dspy_receiver import (
            get_signal_receiver,
            get_pending_training_signals,
            reset_signal_receiver,
        )

        reset_signal_receiver()
        receiver = get_signal_receiver()

        # Manually add signals to buffers
        for agent, signals in sample_signals.items():
            for signal_data in signals:
                from src.agents.feedback_learner.dspy_receiver import ReceivedSignal
                received_signal = ReceivedSignal(
                    source_agent=agent,
                    signal_data=signal_data,
                )
                receiver._buffers[agent].append(received_signal)

        signals = get_pending_training_signals(limit=10)
        assert len(signals) == 4

        reset_signal_receiver()

    def test_get_feedback_items_from_signals(self, sample_signals):
        """get_feedback_items_from_signals should return FeedbackItem dicts."""
        from src.agents.feedback_learner.dspy_receiver import (
            get_signal_receiver,
            get_feedback_items_from_signals,
            reset_signal_receiver,
        )

        reset_signal_receiver()
        receiver = get_signal_receiver()

        # Manually add signals to buffers
        for agent, signals in sample_signals.items():
            for signal_data in signals:
                from src.agents.feedback_learner.dspy_receiver import ReceivedSignal
                received_signal = ReceivedSignal(
                    source_agent=agent,
                    signal_data=signal_data,
                )
                receiver._buffers[agent].append(received_signal)

        items = get_feedback_items_from_signals(limit=10)
        assert len(items) == 4
        assert all(item["feedback_type"] == "training_signal" for item in items)

        reset_signal_receiver()

    def test_get_training_data(self, sample_signals):
        """get_training_data should return high-quality signals."""
        from src.agents.feedback_learner.dspy_receiver import (
            get_signal_receiver,
            get_training_data,
            reset_signal_receiver,
        )

        reset_signal_receiver()
        receiver = get_signal_receiver()

        # Manually add signals to buffers
        for agent, signals in sample_signals.items():
            for signal_data in signals:
                from src.agents.feedback_learner.dspy_receiver import ReceivedSignal
                received_signal = ReceivedSignal(
                    source_agent=agent,
                    signal_data=signal_data,
                )
                receiver._buffers[agent].append(received_signal)

        data = get_training_data(
            agent="causal_impact",
            min_reward=0.5,
            limit=10,
        )

        assert len(data) == 2  # Both ci_001 and ci_002 pass min_reward=0.5
        assert all(d["reward"] >= 0.5 for d in data)

        reset_signal_receiver()


# =============================================================================
# SINGLETON TESTS
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_signal_receiver_returns_same_instance(self):
        """get_signal_receiver should return same instance."""
        from src.agents.feedback_learner.dspy_receiver import (
            get_signal_receiver,
            reset_signal_receiver,
        )

        reset_signal_receiver()

        receiver1 = get_signal_receiver()
        receiver2 = get_signal_receiver()

        assert receiver1 is receiver2

        reset_signal_receiver()

    def test_reset_signal_receiver_creates_new_instance(self):
        """reset_signal_receiver should create new instance on next get."""
        from src.agents.feedback_learner.dspy_receiver import (
            get_signal_receiver,
            reset_signal_receiver,
        )

        receiver1 = get_signal_receiver()
        reset_signal_receiver()
        receiver2 = get_signal_receiver()

        assert receiver1 is not receiver2

        reset_signal_receiver()
