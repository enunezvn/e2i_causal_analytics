"""Shard 03: get_signals_for_optimization must filter source_agent, not signal_type (F4)."""

from __future__ import annotations

import pytest

from src.rag.memory_adapters import SignalCollectorAdapter


class _RecordingQuery:
    def __init__(self, sink):
        self.sink = sink
        self.sink["eq_calls"] = []

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self.sink["eq_calls"].append((col, val))
        return self

    def gte(self, col, val):
        self.sink["gte"] = (col, val)
        return self

    def order(self, col, desc=False):
        # #1661: newest-first BEFORE the limit, so the beat and the
        # operator-facing gate status take the same slice of the same table.
        self.sink["order"] = (col, desc)
        return self

    def limit(self, n):
        self.sink["limit"] = n
        return self

    def execute(self):
        return type("R", (), {"data": [{"source_agent": "feedback_learner", "reward": 0.9}]})()


class _RecordingClient:
    def __init__(self, sink):
        self.sink = sink

    def table(self, name):
        assert name == "dspy_agent_training_signals"
        return _RecordingQuery(self.sink)


@pytest.mark.asyncio
async def test_filters_source_agent_not_signal_type():
    sink: dict = {}
    adapter = SignalCollectorAdapter(supabase_client=_RecordingClient(sink))
    rows = await adapter.get_signals_for_optimization(
        source_agent="feedback_learner", min_reward=0.5
    )
    # The filter MUST target the real column.
    assert ("source_agent", "feedback_learner") in sink["eq_calls"]
    assert all(col != "signal_type" for col, _ in sink["eq_calls"])
    assert rows and rows[0]["source_agent"] == "feedback_learner"


@pytest.mark.asyncio
async def test_feedback_learner_reader_convenience():
    from src.agents.feedback_learner.signal_store import get_feedback_learner_training_signals

    sink: dict = {}
    rows = await get_feedback_learner_training_signals(
        client=_RecordingClient(sink), min_reward=0.5, limit=50
    )
    assert ("source_agent", "feedback_learner") in sink["eq_calls"]
    assert sink["limit"] == 50
    assert rows and rows[0]["source_agent"] == "feedback_learner"
