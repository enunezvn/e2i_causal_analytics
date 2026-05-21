"""Tests for F-008 (#428): feedback.py positive_ratio computed from request.items.

The legacy code hardcoded ``positive_ratio=0.7`` regardless of input. These
tests pin the new compute path so the regression cannot return silently.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.api.routes.feedback import (
    FeedbackItem,
    FeedbackType,
    ProcessFeedbackRequest,
    _is_positive_feedback,
    process_feedback,
)


def _make_item(user_feedback, feedback_type=FeedbackType.RATING, idx=0):
    return FeedbackItem(
        feedback_id=f"fbi_{idx}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        feedback_type=feedback_type,
        source_agent="causal_impact",
        query="q",
        agent_response="a",
        user_feedback=user_feedback,
    )


class TestIsPositiveFeedback:
    """Behavior matrix for the new ``_is_positive_feedback`` helper."""

    def test_dict_rating_high(self):
        assert _is_positive_feedback(_make_item({"rating": 5})) is True

    def test_dict_rating_threshold_four(self):
        assert _is_positive_feedback(_make_item({"rating": 4})) is True

    def test_dict_rating_below_threshold(self):
        assert _is_positive_feedback(_make_item({"rating": 3})) is False
        assert _is_positive_feedback(_make_item({"rating": 1})) is False

    def test_dict_sentiment_positive(self):
        assert _is_positive_feedback(_make_item({"sentiment": "positive"})) is True

    def test_dict_sentiment_negative(self):
        assert _is_positive_feedback(_make_item({"sentiment": "negative"})) is False

    def test_dict_helpful_flag(self):
        assert _is_positive_feedback(_make_item({"helpful": True})) is True
        assert _is_positive_feedback(_make_item({"helpful": False})) is False

    def test_numeric_rating_high(self):
        assert _is_positive_feedback(_make_item(5)) is True
        assert _is_positive_feedback(_make_item(4.5)) is True

    def test_numeric_rating_low(self):
        assert _is_positive_feedback(_make_item(3)) is False

    def test_string_label_positive(self):
        assert _is_positive_feedback(_make_item("positive")) is True
        assert _is_positive_feedback(_make_item("thumbs_up")) is True

    def test_string_label_other(self):
        assert _is_positive_feedback(_make_item("neutral")) is False

    def test_bool_payload(self):
        assert _is_positive_feedback(_make_item(True)) is True
        assert _is_positive_feedback(_make_item(False)) is False

    def test_malformed_returns_false(self):
        assert _is_positive_feedback(_make_item({"weird": "shape"})) is False
        assert _is_positive_feedback(_make_item(None)) is False
        assert _is_positive_feedback(_make_item([1, 2, 3])) is False


class TestPositiveRatioComputation:
    """End-to-end: ``positive_ratio`` reflects the actual request payload."""

    @pytest.mark.asyncio
    async def test_three_positive_of_five_yields_0_6(self):
        """3 positive / 5 total → 0.6 (not the legacy 0.7 mock)."""
        items = [
            _make_item({"rating": 5}, idx=0),
            _make_item({"rating": 4}, idx=1),
            _make_item({"rating": 4}, idx=2),
            _make_item({"rating": 2}, idx=3),
            _make_item({"rating": 1}, idx=4),
        ]
        request = ProcessFeedbackRequest(
            items=items,
            detect_patterns=False,
            generate_recommendations=False,
        )
        user = {"user_id": "u", "role": "operator"}

        # Patch out the Opik enabled flag so the fixture from conftest is not required
        with patch("src.api.routes.feedback.OPIK_FEEDBACK_AVAILABLE", False):
            result = await process_feedback(request, user)

        assert result.feedback_summary is not None
        assert result.feedback_summary.positive_ratio == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_all_positive_yields_1_0(self):
        """100% positive items → ratio = 1.0."""
        items = [_make_item({"rating": 5}, idx=i) for i in range(4)]
        request = ProcessFeedbackRequest(
            items=items,
            detect_patterns=False,
            generate_recommendations=False,
        )
        user = {"user_id": "u", "role": "operator"}

        with patch("src.api.routes.feedback.OPIK_FEEDBACK_AVAILABLE", False):
            result = await process_feedback(request, user)

        assert result.feedback_summary.positive_ratio == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_all_negative_yields_0_0(self):
        """100% negative items → ratio = 0.0 (mock floor of 0.7 would fail)."""
        items = [_make_item({"rating": 1}, idx=i) for i in range(4)]
        request = ProcessFeedbackRequest(
            items=items,
            detect_patterns=False,
            generate_recommendations=False,
        )
        user = {"user_id": "u", "role": "operator"}

        with patch("src.api.routes.feedback.OPIK_FEEDBACK_AVAILABLE", False):
            result = await process_feedback(request, user)

        assert result.feedback_summary.positive_ratio == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_legacy_mock_value_not_present(self):
        """No hardcoded 0.7 mock — even a single-item payload reflects content."""
        # One positive item → ratio must be 1.0, not the legacy 0.7.
        items = [_make_item({"rating": 5}, idx=0)]
        request = ProcessFeedbackRequest(
            items=items,
            detect_patterns=False,
            generate_recommendations=False,
        )
        user = {"user_id": "u", "role": "operator"}

        with patch("src.api.routes.feedback.OPIK_FEEDBACK_AVAILABLE", False):
            result = await process_feedback(request, user)

        assert result.feedback_summary.positive_ratio == pytest.approx(1.0)
        # And ensure it is NOT the legacy hard-coded value
        assert result.feedback_summary.positive_ratio != pytest.approx(0.7)
