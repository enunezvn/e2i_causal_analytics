"""Unit tests for _CompositeFeedbackStore (agent.py).

Production composes chat thumbs (chatbot_message_feedback) with the cognitive
reward stream (learning_signals). These tests pin the composition contract:
concatenation, per-store failure isolation, kwargs pass-through, None filtering.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.agents.feedback_learner.agent import _CompositeFeedbackStore


class _Store:
    def __init__(self, items: List[Dict[str, Any]] | None = None, err: Exception | None = None):
        self.items = items or []
        self.err = err
        self.calls: List[Dict[str, Any]] = []

    async def get_feedback(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.calls.append(kwargs)
        if self.err:
            raise self.err
        return self.items


@pytest.mark.unit
class TestCompositeFeedbackStore:
    @pytest.mark.asyncio
    async def test_concatenates_all_store_results(self):
        a = _Store([{"id": "a1"}, {"id": "a2"}])
        b = _Store([{"id": "b1"}])
        composite = _CompositeFeedbackStore([a, b])
        items = await composite.get_feedback()
        assert [i["id"] for i in items] == ["a1", "a2", "b1"]

    @pytest.mark.asyncio
    async def test_failing_store_is_isolated(self):
        """A broken source degrades coverage — it must never kill the cycle."""
        broken = _Store(err=RuntimeError("db down"))
        healthy = _Store([{"id": "ok"}])
        composite = _CompositeFeedbackStore([broken, healthy])
        items = await composite.get_feedback()
        assert [i["id"] for i in items] == ["ok"]

    @pytest.mark.asyncio
    async def test_kwargs_passed_to_every_store(self):
        a, b = _Store(), _Store()
        composite = _CompositeFeedbackStore([a, b])
        await composite.get_feedback(
            start_time="2026-07-01T00:00:00+00:00", end_time=None, agents=["gap_analyzer"]
        )
        expected = {
            "start_time": "2026-07-01T00:00:00+00:00",
            "end_time": None,
            "agents": ["gap_analyzer"],
        }
        assert a.calls == [expected]
        assert b.calls == [expected]

    @pytest.mark.asyncio
    async def test_none_stores_filtered(self):
        composite = _CompositeFeedbackStore([None, _Store([{"id": "x"}]), None])
        items = await composite.get_feedback()
        assert [i["id"] for i in items] == ["x"]

    @pytest.mark.asyncio
    async def test_empty_composite_returns_empty(self):
        assert await _CompositeFeedbackStore([]).get_feedback() == []


@pytest.mark.asyncio
async def test_builder_composes_thumbs_and_learning_signals(monkeypatch):
    """build_production_feedback_stores must feed the learner BOTH real sources:
    explicit chat thumbs AND the per-turn cognitive reward stream."""
    from unittest.mock import AsyncMock, MagicMock

    import src.memory.services.factories as factories

    monkeypatch.setattr(
        factories, "get_async_supabase_client", AsyncMock(return_value=MagicMock())
    )

    from src.agents.feedback_learner import agent as agent_mod

    feedback_store, _knowledge_stores, db_client = (
        await agent_mod.build_production_feedback_stores()
    )
    assert isinstance(feedback_store, _CompositeFeedbackStore)
    store_types = {type(s).__name__ for s in feedback_store._stores}
    assert store_types == {"ChatbotFeedbackRepository", "LearningSignalsFeedbackStore"}
    assert db_client is not None
