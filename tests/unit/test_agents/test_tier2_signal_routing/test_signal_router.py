"""
Unit Tests for Tier2SignalRouter.

Tests for the signal router that batches and delivers training signals
from Tier 2 agents to the feedback_learner.

Verifies:
- Signal structure validation
- Queue management (capacity, ordering)
- Batch flushing (auto and manual)
- Metrics tracking
- Graceful fallback when feedback_learner unavailable

Run: pytest tests/unit/test_agents/test_tier2_signal_routing/test_signal_router.py -v
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock

# Mark all tests in this module for the tier2 xdist group
pytestmark = pytest.mark.xdist_group(name="tier2_signal_routing")


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def causal_impact_signal() -> Dict[str, Any]:
    """Create a valid causal_impact training signal."""
    return {
        "signal_id": "ci_test_001",
        "source_agent": "causal_impact",
        "dspy_type": "sender",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reward": 0.75,
        "input_context": {
            "query": "What is the causal impact of marketing?",
            "treatment_var": "marketing_spend",
            "outcome_var": "sales",
        },
        "output": {
            "ate": 0.15,
            "ci_lower": 0.10,
            "ci_upper": 0.20,
        },
    }


@pytest.fixture
def gap_analyzer_signal() -> Dict[str, Any]:
    """Create a valid gap_analyzer training signal."""
    return {
        "signal_id": "ga_test_001",
        "source_agent": "gap_analyzer",
        "dspy_type": "sender",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reward": 0.82,
        "input_context": {
            "query": "Find ROI opportunities in northeast",
            "brand": "TestBrand",
        },
        "output": {
            "gaps_detected": 5,
            "quick_wins": 2,
        },
    }


@pytest.fixture
def heterogeneous_optimizer_signal() -> Dict[str, Any]:
    """Create a valid heterogeneous_optimizer training signal."""
    return {
        "signal_id": "ho_test_001",
        "source_agent": "heterogeneous_optimizer",
        "dspy_type": "sender",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reward": 0.68,
        "input_context": {
            "query": "Which segments respond best to email?",
        },
        "output": {
            "segments_found": 8,
            "high_responders": 3,
        },
    }


@pytest.fixture
def signal_router():
    """Create a fresh signal router instance."""
    from src.agents.tier2_signal_router import Tier2SignalRouter
    return Tier2SignalRouter()


@pytest.fixture(autouse=True)
def reset_router_singleton():
    """Reset the router singleton before and after each test."""
    from src.agents.tier2_signal_router import reset_signal_router
    reset_signal_router()
    yield
    reset_signal_router()


# =============================================================================
# SIGNAL STRUCTURE TESTS
# =============================================================================


class TestSignalStructure:
    """Tests for signal structure validation."""

    def test_causal_impact_signal_has_required_fields(self, causal_impact_signal):
        """Causal impact signal should have all required fields."""
        required_fields = [
            "signal_id",
            "source_agent",
            "dspy_type",
            "timestamp",
            "reward",
            "input_context",
        ]
        for field in required_fields:
            assert field in causal_impact_signal, f"Missing field: {field}"

        assert causal_impact_signal["source_agent"] == "causal_impact"
        assert causal_impact_signal["dspy_type"] == "sender"

    def test_gap_analyzer_signal_has_required_fields(self, gap_analyzer_signal):
        """Gap analyzer signal should have all required fields."""
        required_fields = [
            "signal_id",
            "source_agent",
            "dspy_type",
            "timestamp",
            "reward",
            "input_context",
        ]
        for field in required_fields:
            assert field in gap_analyzer_signal, f"Missing field: {field}"

        assert gap_analyzer_signal["source_agent"] == "gap_analyzer"
        assert gap_analyzer_signal["dspy_type"] == "sender"

    def test_heterogeneous_optimizer_signal_has_required_fields(
        self, heterogeneous_optimizer_signal
    ):
        """Heterogeneous optimizer signal should have all required fields."""
        required_fields = [
            "signal_id",
            "source_agent",
            "dspy_type",
            "timestamp",
            "reward",
            "input_context",
        ]
        for field in required_fields:
            assert field in heterogeneous_optimizer_signal, f"Missing field: {field}"

        assert heterogeneous_optimizer_signal["source_agent"] == "heterogeneous_optimizer"
        assert heterogeneous_optimizer_signal["dspy_type"] == "sender"

    def test_reward_in_valid_range(
        self,
        causal_impact_signal,
        gap_analyzer_signal,
        heterogeneous_optimizer_signal,
    ):
        """All signals should have reward in [0, 1] range."""
        for signal in [
            causal_impact_signal,
            gap_analyzer_signal,
            heterogeneous_optimizer_signal,
        ]:
            reward = signal["reward"]
            assert 0.0 <= reward <= 1.0, f"Invalid reward: {reward}"


# =============================================================================
# QUEUE MANAGEMENT TESTS
# =============================================================================


class TestQueueManagement:
    """Tests for signal queue management."""

    @pytest.mark.asyncio
    async def test_submit_signal_adds_to_queue(
        self, signal_router, causal_impact_signal
    ):
        """Submitting a signal should add it to the queue."""
        # Patch the flush to prevent auto-flush
        with patch.object(signal_router, "_flush_internal", new_callable=AsyncMock):
            result = await signal_router.submit_signal(
                "causal_impact", causal_impact_signal
            )

            assert result is True
            assert signal_router.queue_size == 1

    @pytest.mark.asyncio
    async def test_queue_respects_max_size(self, signal_router):
        """Queue should drop oldest signals when max size reached."""
        # Set small max for testing
        signal_router.MAX_QUEUE_SIZE = 5

        # Patch flush to prevent auto-delivery
        with patch.object(signal_router, "_flush_internal", new_callable=AsyncMock):
            # Add more signals than max
            for i in range(10):
                signal = {"signal_id": f"test_{i}", "reward": 0.5}
                await signal_router.submit_signal("causal_impact", signal)

        # Queue should be at max size
        assert signal_router.queue_size == 5
        # Should have dropped 5 signals
        assert signal_router._metrics["signals_dropped"] == 5

    @pytest.mark.asyncio
    async def test_signals_have_timestamp_on_queue(
        self, signal_router, causal_impact_signal
    ):
        """Signals should have timestamp added when queued."""
        with patch.object(signal_router, "_flush_internal", new_callable=AsyncMock):
            await signal_router.submit_signal("causal_impact", causal_impact_signal)

            # Check queued entry has timestamp
            assert len(signal_router._queue) == 1
            entry = signal_router._queue[0]
            assert "timestamp" in entry
            assert "agent_name" in entry
            assert entry["agent_name"] == "causal_impact"


# =============================================================================
# BATCH FLUSHING TESTS
# =============================================================================


class TestBatchFlushing:
    """Tests for batch flushing behavior."""

    @pytest.mark.asyncio
    async def test_auto_flush_at_batch_size(self, signal_router):
        """Should auto-flush when batch size reached."""
        signal_router.BATCH_SIZE = 3

        flush_count = 0

        async def mock_flush():
            nonlocal flush_count
            flush_count += 1
            signal_router._queue.clear()
            return 3

        with patch.object(
            signal_router, "_flush_internal", side_effect=mock_flush
        ):
            for i in range(3):
                await signal_router.submit_signal(
                    "causal_impact", {"signal_id": f"test_{i}"}
                )

        # Should have auto-flushed once at batch size
        assert flush_count == 1

    @pytest.mark.asyncio
    async def test_manual_flush_clears_queue(self, signal_router):
        """Manual flush should clear the queue."""
        # Add signals without triggering auto-flush
        signal_router.BATCH_SIZE = 100

        with patch.object(
            signal_router,
            "_deliver_to_feedback_learner",
            new_callable=AsyncMock,
            return_value=2,
        ):
            await signal_router.submit_signal(
                "causal_impact", {"signal_id": "test_1"}
            )
            await signal_router.submit_signal(
                "gap_analyzer", {"signal_id": "test_2"}
            )

            assert signal_router.queue_size == 2

            # Manual flush
            delivered = await signal_router.flush()

            assert delivered == 2
            assert signal_router.queue_size == 0

    @pytest.mark.asyncio
    async def test_flush_empty_queue_returns_zero(self, signal_router):
        """Flushing empty queue should return 0."""
        delivered = await signal_router.flush()
        assert delivered == 0


# =============================================================================
# METRICS TRACKING TESTS
# =============================================================================


class TestMetricsTracking:
    """Tests for metrics tracking."""

    @pytest.mark.asyncio
    async def test_signals_received_metric(self, signal_router):
        """Should track signals received."""
        with patch.object(signal_router, "_flush_internal", new_callable=AsyncMock):
            await signal_router.submit_signal("causal_impact", {"signal_id": "1"})
            await signal_router.submit_signal("gap_analyzer", {"signal_id": "2"})

        metrics = signal_router.get_metrics()
        assert metrics["signals_received"] == 2

    @pytest.mark.asyncio
    async def test_signals_delivered_metric(self, signal_router):
        """Should track signals delivered."""
        with patch.object(
            signal_router,
            "_deliver_to_feedback_learner",
            new_callable=AsyncMock,
            return_value=2,
        ):
            await signal_router.submit_signal("causal_impact", {"signal_id": "1"})
            await signal_router.submit_signal("gap_analyzer", {"signal_id": "2"})
            await signal_router.flush()

        metrics = signal_router.get_metrics()
        assert metrics["signals_delivered"] == 2
        assert metrics["batches_sent"] == 1

    @pytest.mark.asyncio
    async def test_delivery_errors_metric(self, signal_router):
        """Should track delivery errors."""
        with patch.object(
            signal_router,
            "_deliver_to_feedback_learner",
            new_callable=AsyncMock,
            side_effect=Exception("Test error"),
        ):
            await signal_router.submit_signal("causal_impact", {"signal_id": "1"})
            await signal_router.flush()

        metrics = signal_router.get_metrics()
        assert metrics["delivery_errors"] == 1
        assert metrics["signals_dropped"] == 1

    def test_reset_metrics(self, signal_router):
        """Should reset all metrics."""
        signal_router._metrics["signals_received"] = 100
        signal_router._metrics["batches_sent"] = 10

        signal_router.reset_metrics()

        metrics = signal_router.get_metrics()
        assert all(v == 0 for v in metrics.values())


# =============================================================================
# DELIVERY TESTS
# =============================================================================


class TestDelivery:
    """Tests for signal delivery to feedback_learner."""

    @pytest.mark.asyncio
    async def test_delivery_organizes_by_agent(self, signal_router):
        """Delivery should organize signals by agent."""
        delivered_signals = None

        async def capture_delivery(signals):
            nonlocal delivered_signals
            delivered_signals = signals
            return len(signals)

        with patch.object(
            signal_router,
            "_deliver_to_feedback_learner",
            side_effect=capture_delivery,
        ):
            await signal_router.submit_signal(
                "causal_impact", {"signal_id": "ci_1"}
            )
            await signal_router.submit_signal(
                "gap_analyzer", {"signal_id": "ga_1"}
            )
            await signal_router.submit_signal(
                "causal_impact", {"signal_id": "ci_2"}
            )
            await signal_router.flush()

        # Should have 3 signals total
        assert len(delivered_signals) == 3

        # Check agent distribution
        agents = [s["agent_name"] for s in delivered_signals]
        assert agents.count("causal_impact") == 2
        assert agents.count("gap_analyzer") == 1

    @pytest.mark.asyncio
    async def test_delivery_to_feedback_learner_receiver(self):
        """Should deliver to feedback_learner receiver when available."""
        from src.agents.tier2_signal_router import (
            get_signal_router,
            reset_signal_router,
        )
        from src.agents.feedback_learner.dspy_receiver import (
            get_signal_receiver,
            reset_signal_receiver,
        )

        reset_signal_router()
        reset_signal_receiver()

        router = get_signal_router()
        receiver = get_signal_receiver()

        # Submit and flush
        await router.submit_signal("causal_impact", {"signal_id": "test_1", "reward": 0.7})
        await router.submit_signal("gap_analyzer", {"signal_id": "test_2", "reward": 0.8})
        await router.flush()

        # Check receiver got signals
        stats = receiver.get_statistics()
        assert stats["signals_received"] == 2

        reset_signal_router()
        reset_signal_receiver()


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience routing functions."""

    @pytest.mark.asyncio
    async def test_route_causal_impact_signal(self, causal_impact_signal):
        """route_causal_impact_signal should route to router."""
        from src.agents.tier2_signal_router import (
            route_causal_impact_signal,
            get_signal_router,
            reset_signal_router,
        )

        reset_signal_router()
        router = get_signal_router()

        with patch.object(router, "_flush_internal", new_callable=AsyncMock):
            result = await route_causal_impact_signal(causal_impact_signal)

            assert result is True
            assert router.queue_size == 1

        reset_signal_router()

    @pytest.mark.asyncio
    async def test_route_gap_analyzer_signal(self, gap_analyzer_signal):
        """route_gap_analyzer_signal should route to router."""
        from src.agents.tier2_signal_router import (
            route_gap_analyzer_signal,
            get_signal_router,
            reset_signal_router,
        )

        reset_signal_router()
        router = get_signal_router()

        with patch.object(router, "_flush_internal", new_callable=AsyncMock):
            result = await route_gap_analyzer_signal(gap_analyzer_signal)

            assert result is True
            assert router.queue_size == 1

        reset_signal_router()

    @pytest.mark.asyncio
    async def test_route_heterogeneous_optimizer_signal(
        self, heterogeneous_optimizer_signal
    ):
        """route_heterogeneous_optimizer_signal should route to router."""
        from src.agents.tier2_signal_router import (
            route_heterogeneous_optimizer_signal,
            get_signal_router,
            reset_signal_router,
        )

        reset_signal_router()
        router = get_signal_router()

        with patch.object(router, "_flush_internal", new_callable=AsyncMock):
            result = await route_heterogeneous_optimizer_signal(
                heterogeneous_optimizer_signal
            )

            assert result is True
            assert router.queue_size == 1

        reset_signal_router()


# =============================================================================
# SINGLETON TESTS
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_signal_router_returns_same_instance(self):
        """get_signal_router should return same instance."""
        from src.agents.tier2_signal_router import get_signal_router, reset_signal_router

        reset_signal_router()

        router1 = get_signal_router()
        router2 = get_signal_router()

        assert router1 is router2

        reset_signal_router()

    def test_reset_signal_router_creates_new_instance(self):
        """reset_signal_router should create new instance on next get."""
        from src.agents.tier2_signal_router import get_signal_router, reset_signal_router

        router1 = get_signal_router()
        reset_signal_router()
        router2 = get_signal_router()

        assert router1 is not router2

        reset_signal_router()
