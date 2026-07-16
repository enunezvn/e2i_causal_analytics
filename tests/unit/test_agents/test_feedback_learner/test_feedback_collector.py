"""
Tests for Feedback Collector node.
"""

from unittest.mock import AsyncMock

import pytest

from src.agents.feedback_learner.nodes.feedback_collector import FeedbackCollectorNode


@pytest.fixture
def raw_store_feedback():
    """Raw feedback data as returned from store (needs transformation)."""
    return [
        {
            "id": "F001",
            "agent": "causal_impact",
            "query": "What caused the TRx drop?",
            "response": "The drop was caused by competitor launch.",
            "rating": 4,
            "timestamp": "2024-01-15T10:00:00Z",
        },
        {
            "id": "F002",
            "agent": "gap_analyzer",
            "query": "Find ROI opportunities",
            "response": "Top opportunity is Region A.",
            "correction": "The opportunity should be Region B, not A",
            "timestamp": "2024-01-15T11:00:00Z",
        },
    ]


class TestFeedbackCollectorNode:
    """Tests for FeedbackCollectorNode."""

    @pytest.mark.asyncio
    async def test_execute_with_feedback_store(self, base_state, raw_store_feedback):
        """Test execution with feedback store."""
        mock_store = AsyncMock()
        mock_store.get_feedback = AsyncMock(return_value=raw_store_feedback)

        node = FeedbackCollectorNode(
            feedback_store=mock_store,
            outcome_store=None,
        )

        result = await node.execute(base_state)

        assert result["status"] == "analyzing"
        assert len(result["feedback_items"]) == 2
        assert result["feedback_summary"] is not None
        assert result["collection_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_without_stores(self, base_state):
        """Test execution without any stores."""
        node = FeedbackCollectorNode(
            feedback_store=None,
            outcome_store=None,
        )

        result = await node.execute(base_state)

        assert result["status"] == "analyzing"
        assert result["feedback_items"] == []
        assert result["feedback_summary"]["total_count"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_focus_agents(self, base_state, raw_store_feedback):
        """Test execution with focus agents filter."""
        mock_store = AsyncMock()
        mock_store.get_feedback = AsyncMock(return_value=raw_store_feedback)

        state = {
            **base_state,
            "focus_agents": ["causal_impact"],
        }
        node = FeedbackCollectorNode(
            feedback_store=mock_store,
            outcome_store=None,
        )

        result = await node.execute(state)

        # Verify feedback was collected
        assert result["status"] == "analyzing"
        assert result["feedback_items"] is not None
        # Verify focus_agents was passed to the store
        mock_store.get_feedback.assert_called_once()
        call_kwargs = mock_store.get_feedback.call_args[1]
        assert call_kwargs["agents"] == ["causal_impact"]

    @pytest.mark.asyncio
    async def test_execute_with_outcome_store(self, base_state, raw_store_feedback):
        """Test execution with both feedback and outcome stores."""
        mock_feedback_store = AsyncMock()
        mock_feedback_store.get_feedback = AsyncMock(return_value=raw_store_feedback)

        mock_outcome_store = AsyncMock()
        mock_outcome_store.get_outcomes = AsyncMock(return_value=[])

        node = FeedbackCollectorNode(
            feedback_store=mock_feedback_store,
            outcome_store=mock_outcome_store,
        )

        result = await node.execute(base_state)

        assert result["status"] == "analyzing"
        assert result["feedback_items"] is not None
        mock_outcome_store.get_outcomes.assert_called_once()

    @pytest.mark.asyncio
    async def test_feedback_summary_generation(self, base_state, raw_store_feedback):
        """Test that feedback summary is correctly generated."""
        mock_store = AsyncMock()
        mock_store.get_feedback = AsyncMock(return_value=raw_store_feedback)

        node = FeedbackCollectorNode(
            feedback_store=mock_store,
            outcome_store=None,
        )

        result = await node.execute(base_state)

        summary = result["feedback_summary"]
        assert summary["total_count"] == 2
        assert "by_type" in summary
        assert "by_agent" in summary
        assert "average_rating" in summary
        # One rating (4) from raw_store_feedback
        assert summary["average_rating"] == 4.0

    @pytest.mark.asyncio
    async def test_average_rating_calculation(self, base_state):
        """Test average rating calculation."""
        raw_feedback = [
            {
                "id": "F1",
                "agent": "agent1",
                "query": "q1",
                "response": "r1",
                "rating": 5,
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "id": "F2",
                "agent": "agent1",
                "query": "q2",
                "response": "r2",
                "rating": 3,
                "timestamp": "2024-01-01T01:00:00Z",
            },
            {
                "id": "F3",
                "agent": "agent1",
                "query": "q3",
                "response": "r3",
                "correction": "correction text",
                "timestamp": "2024-01-01T02:00:00Z",
            },
        ]

        mock_store = AsyncMock()
        mock_store.get_feedback = AsyncMock(return_value=raw_feedback)

        node = FeedbackCollectorNode(
            feedback_store=mock_store,
            outcome_store=None,
        )

        result = await node.execute(base_state)

        # Average of 5 and 3 = 4.0
        assert result["feedback_summary"]["average_rating"] == 4.0

    @pytest.mark.asyncio
    async def test_summary_counts_thumbs_string_ratings(self, base_state):
        """Explicit thumbs arrive as enum STRINGS — the summary must map them
        onto the shared 1-5 scale (up→5, down→1), not silently exclude them
        from average_rating/rating_count (codex round-1 MED: only numeric
        reward-derived ratings were being counted)."""
        raw_feedback = [
            {
                "id": "F1",
                "agent": "copilotkit",
                "query": "q1",
                "response": "r1",
                "rating": "thumbs_up",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "id": "F2",
                "agent": "copilotkit",
                "query": "q2",
                "response": "r2",
                "rating": "thumbs_down",
                "timestamp": "2024-01-01T01:00:00Z",
            },
            {
                "id": "F3",
                "agent": "gap_analyzer",
                "query": "q3",
                "response": "r3",
                "rating": 4.2,
                "timestamp": "2024-01-01T02:00:00Z",
            },
        ]

        mock_store = AsyncMock()
        mock_store.get_feedback = AsyncMock(return_value=raw_feedback)
        node = FeedbackCollectorNode(feedback_store=mock_store, outcome_store=None)

        result = await node.execute(base_state)

        summary = result["feedback_summary"]
        assert summary["rating_count"] == 3
        assert summary["average_rating"] == pytest.approx((5.0 + 1.0 + 4.2) / 3)

    @pytest.mark.asyncio
    async def test_skip_if_already_failed(self, base_state):
        """Test that node skips execution if already failed."""
        state = {**base_state, "status": "failed"}
        node = FeedbackCollectorNode()

        result = await node.execute(state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_error_handling(self, base_state):
        """Test error handling when store throws exception."""
        mock_store = AsyncMock()
        mock_store.get_feedback = AsyncMock(side_effect=Exception("Store error"))

        node = FeedbackCollectorNode(
            feedback_store=mock_store,
            outcome_store=None,
        )

        result = await node.execute(base_state)

        # When user feedback collection fails, it logs warning but continues
        # The exception is caught within _collect_user_feedback
        # Only the outer try/except causes status=failed
        assert result["status"] == "analyzing"
        # It should still have empty results when collection fails gracefully
        assert result["feedback_items"] == []

    @pytest.mark.asyncio
    async def test_latency_tracking(self, base_state, raw_store_feedback):
        """Test that latency is properly tracked."""
        mock_store = AsyncMock()
        mock_store.get_feedback = AsyncMock(return_value=raw_store_feedback)

        node = FeedbackCollectorNode(
            feedback_store=mock_store,
            outcome_store=None,
        )

        result = await node.execute(base_state)

        assert "collection_latency_ms" in result
        assert isinstance(result["collection_latency_ms"], int)
        assert result["collection_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_empty_feedback_list(self, base_state):
        """Test handling of empty feedback list."""
        mock_store = AsyncMock()
        mock_store.get_feedback = AsyncMock(return_value=[])

        node = FeedbackCollectorNode(
            feedback_store=mock_store,
            outcome_store=None,
        )

        result = await node.execute(base_state)

        assert result["status"] == "analyzing"
        assert result["feedback_items"] == []
        assert result["feedback_summary"]["total_count"] == 0
        assert result["feedback_summary"]["average_rating"] is None

    @pytest.mark.asyncio
    async def test_outcome_store_transformation(self, base_state):
        """Test outcome data is properly transformed into feedback items."""
        mock_outcome_store = AsyncMock()
        mock_outcome_store.get_outcomes = AsyncMock(
            return_value=[
                {
                    "id": "O001",
                    "agent": "prediction_synthesizer",
                    "original_query": "Predict TRx",
                    "prediction": 1000,
                    "actual": 950,
                    "timestamp": "2024-01-15T10:00:00Z",
                }
            ]
        )

        node = FeedbackCollectorNode(
            feedback_store=None,
            outcome_store=mock_outcome_store,
        )

        result = await node.execute(base_state)

        assert result["status"] == "analyzing"
        assert len(result["feedback_items"]) == 1
        item = result["feedback_items"][0]
        assert item["feedback_type"] == "outcome"
        assert item["user_feedback"]["predicted"] == 1000
        assert item["user_feedback"]["actual"] == 950
        assert item["user_feedback"]["error"] == -50


class TestSourceAwareSummary:
    """#1251: the summary must expose per-surface rating aggregates so no
    consumer is forced onto the mix-sensitive pooled mean. The pooled
    average_rating is retained (mlflow continuity) but the by-source dicts
    are the source-aware aggregate future top-anchored consumers must read."""

    @pytest.mark.asyncio
    async def test_summary_breaks_ratings_down_by_surface(self, base_state):
        raw_feedback = [
            {
                "id": "F1",
                "agent": "chat_user_agent",
                "query": "q1",
                "response": "r1",
                "rating": "thumbs_up",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "id": "F2",
                "agent": "gap_analyzer",
                "query": "q2",
                "response": "r2",
                "rating": 4.6,
                "timestamp": "2024-01-01T01:00:00Z",
                "metadata": {
                    "source": "learning_signals",
                    "signal_component": "agent",
                    "source_path": None,
                    "reward": 0.9,
                },
            },
            {
                "id": "F3",
                "agent": "copilotkit",
                "query": "q3",
                "response": "r3",
                "rating": 3.4,
                "timestamp": "2024-01-01T02:00:00Z",
                "metadata": {
                    "source": "learning_signals",
                    "signal_component": "agent",
                    "source_path": "copilotkit",
                    "reward": 0.6,
                },
            },
        ]
        mock_store = AsyncMock()
        mock_store.get_feedback = AsyncMock(return_value=raw_feedback)
        node = FeedbackCollectorNode(feedback_store=mock_store, outcome_store=None)

        result = await node.execute(base_state)

        summary = result["feedback_summary"]
        # pooled mean unchanged (backward compat for mlflow/prompt)
        assert summary["average_rating"] == pytest.approx((5.0 + 4.6 + 3.4) / 3)
        assert summary["average_rating_by_source"] == {
            "explicit": pytest.approx(5.0),
            "cognitive": pytest.approx(4.6),
            "copilotkit": pytest.approx(3.4),
        }
        assert summary["rating_count_by_source"] == {
            "explicit": 1,
            "cognitive": 1,
            "copilotkit": 1,
        }

    @pytest.mark.asyncio
    async def test_by_source_fields_always_present_empty_when_no_ratings(self, base_state):
        """Always-present contract ({} when nothing rated) — consumers must
        be able to distinguish 'no ratings' from 'field not produced'."""
        raw_feedback = [
            {
                "id": "F1",
                "agent": "agent1",
                "query": "q1",
                "response": "r1",
                "correction": "wrong number",
                "timestamp": "2024-01-01T00:00:00Z",
            },
        ]
        mock_store = AsyncMock()
        mock_store.get_feedback = AsyncMock(return_value=raw_feedback)
        node = FeedbackCollectorNode(feedback_store=mock_store, outcome_store=None)

        result = await node.execute(base_state)

        summary = result["feedback_summary"]
        assert summary["average_rating"] is None
        assert summary["average_rating_by_source"] == {}
        assert summary["rating_count_by_source"] == {}

    @pytest.mark.asyncio
    async def test_by_source_fields_present_on_empty_run(self, base_state):
        mock_store = AsyncMock()
        mock_store.get_feedback = AsyncMock(return_value=[])
        node = FeedbackCollectorNode(feedback_store=mock_store, outcome_store=None)

        result = await node.execute(base_state)

        summary = result["feedback_summary"]
        assert summary["total_count"] == 0
        assert summary["average_rating_by_source"] == {}
        assert summary["rating_count_by_source"] == {}
