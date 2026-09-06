"""The feedback-learning strategic insight grounds on the same ACTIVE pattern
window the /feedback-learning page shows (codex iter-1 on the 2026-09-06 stale
"cognitive_investigator" pattern fix): a month-old detection hidden from the
Patterns card must not still steer the LM's interpretation of the loop.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _pattern(pattern_id: str, days_old: float):
    from src.api.routes.feedback import DetectedPattern, PatternSeverity, PatternType

    return DetectedPattern(
        pattern_id=pattern_id,
        pattern_type=PatternType.ACCURACY_ISSUE,
        description="high negative feedback rate",
        frequency=5,
        severity=PatternSeverity.HIGH,
        affected_agents=["cognitive_investigator"],
        example_feedback_ids=[],
        root_cause_hypothesis="h",
        confidence=0.6,
        detected_at=datetime.now(timezone.utc) - timedelta(days=days_old),
    )


@pytest.mark.asyncio
async def test_feedback_learning_insight_grounds_on_recent_patterns_only():
    from src.api.routes import insights_strategic as mod

    repo = MagicMock()
    repo.count_recent_and_last = AsyncMock(return_value=[])
    repo.list_patterns = AsyncMock(return_value=[_pattern("fresh", 2), _pattern("stale", 45)])
    repo.list_updates = AsyncMock(return_value=[])
    thumbs_repo = MagicMock()
    thumbs_repo.get_feedback_summary = AsyncMock(return_value={"total_feedback": 0})
    signal_store = MagicMock()
    signal_store.get_feedback = AsyncMock(return_value=[])

    captured: dict = {}

    def fake_build_grounding(**kwargs):
        captured.update(kwargs)
        return {
            "activity_summary": "a",
            "patterns_summary": "p",
            "updates_summary": "u",
            "signal_quality_summary": "q",
        }

    with (
        patch("src.api.repositories.feedback_repository.FeedbackRepository", return_value=repo),
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            AsyncMock(return_value=object()),
        ),
        patch(
            "src.repositories.chatbot_feedback.get_chatbot_feedback_repository",
            return_value=thumbs_repo,
        ),
        patch(
            "src.repositories.learning_signals_feedback.get_learning_signals_feedback_store",
            return_value=signal_store,
        ),
        patch.object(mod.feedback_learning, "build_grounding", side_effect=fake_build_grounding),
        patch.object(mod, "cache_get", AsyncMock(return_value={"insight": "cached"})),
        patch.object(mod, "_finalize", return_value=MagicMock()),
    ):
        await mod.feedback_learning_insight(mod.FeedbackLearningInsightRequest(days=7), user={})

    assert [p["pattern_id"] for p in captured["patterns"]] == ["fresh"]
