"""Shard 02: durable persistence of feedback_learner training signals (F5)."""

from __future__ import annotations

import pytest

from src.agents.feedback_learner.agent import FeedbackLearnerAgent
from src.agents.feedback_learner.dspy_integration import FeedbackLearnerTrainingSignal
from src.agents.feedback_learner.signal_store import build_signal_record


def _signal() -> FeedbackLearnerTrainingSignal:
    return FeedbackLearnerTrainingSignal(
        batch_id="batch_abc",
        feedback_count=20,
        time_range_start="2026-06-07T00:00:00Z",
        time_range_end="2026-06-07T06:00:00Z",
        focus_agents=["causal_impact"],
        patterns_detected=5,
        recommendations_generated=4,
        updates_applied=3,
        recommendation_actionability=0.8,
        update_effectiveness=0.9,
        total_latency_ms=1500.0,
        model_used="deterministic",
        rubric_weighted_score=4.5,
        rubric_decision="approve",
        rubric_pattern_flags=1,
    )


def test_record_maps_to_migration_014_columns():
    rec = build_signal_record(_signal())
    # Columns that MUST exist per database/memory/014_dspy_training_signals.sql
    assert rec["source_agent"] == "feedback_learner"
    assert rec["batch_id"] == "batch_abc"
    assert isinstance(rec["input_context"], dict)
    assert isinstance(rec["output"], dict)
    assert isinstance(rec["quality_metrics"], dict)
    assert isinstance(rec["latency_breakdown"], dict)
    assert rec["total_latency_ms"] == 1500
    assert rec["model_used"] == "deterministic"
    assert rec["has_cognitive_context"] is False
    assert rec["is_training_example"] is True
    # There must be NO signal_type column (that was the F4 reader bug's phantom column)
    assert "signal_type" not in rec


def test_reward_within_check_constraint_bounds():
    rec = build_signal_record(_signal())
    assert 0.0 <= rec["reward"] <= 1.0  # CHECK (reward BETWEEN 0 AND 1)


def test_output_carries_pattern_counts_for_downstream_conversion():
    rec = build_signal_record(_signal())
    assert rec["output"]["patterns_detected"] == 5
    assert rec["output"]["recommendations_generated"] == 4


class _RecordingClient:
    """Real stand-in for a Supabase client; records inserted rows."""

    def __init__(self):
        self.rows: list[dict] = []
        self._pending: list[dict] | None = None

    def table(self, name):
        assert name == "dspy_agent_training_signals"
        return self

    def insert(self, record):
        self._pending = record if isinstance(record, list) else [record]
        return self

    def execute(self):
        assert self._pending is not None
        self.rows.extend(self._pending)
        result = {"data": self._pending}
        self._pending = None
        return result


@pytest.mark.asyncio
async def test_learn_persists_signal_via_injected_client():
    client = _RecordingClient()
    # use_llm=False -> deterministic path, no LLM needed
    agent = FeedbackLearnerAgent(persist_client=client)
    out = await agent.learn(
        time_range_start="2026-06-07T00:00:00Z",
        time_range_end="2026-06-07T06:00:00Z",
        batch_id="batch_persist_test",
    )
    assert out.status in {"completed", "failed"}
    assert len(client.rows) == 1
    assert client.rows[0]["source_agent"] == "feedback_learner"
    assert client.rows[0]["batch_id"] == "batch_persist_test"
