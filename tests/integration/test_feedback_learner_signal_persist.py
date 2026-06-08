"""Shard 02 integration: real round-trip insert+read of a feedback_learner signal."""

from __future__ import annotations

import os
import uuid

import pytest

from src.agents.feedback_learner.dspy_integration import FeedbackLearnerTrainingSignal
from src.agents.feedback_learner.signal_store import persist_training_signal

pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL")
    or not (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")),
    reason="requires live Supabase",
)


@pytest.mark.asyncio
async def test_persist_then_read_back():
    from src.memory.services.factories import get_supabase_client

    client = get_supabase_client()
    assert client is not None
    batch = f"itest_{uuid.uuid4().hex[:8]}"
    sig = FeedbackLearnerTrainingSignal(
        batch_id=batch,
        feedback_count=3,
        time_range_start="t0",
        time_range_end="t1",
        patterns_detected=1,
        recommendations_generated=1,
        updates_applied=0,
        recommendation_actionability=0.2,
        update_effectiveness=0.0,
        total_latency_ms=900.0,
    )
    ok = await persist_training_signal(sig, client=client)
    assert ok is True
    try:
        res = (
            client.table("dspy_agent_training_signals").select("*").eq("batch_id", batch).execute()
        )
        data = getattr(res, "data", res.get("data") if isinstance(res, dict) else None)
        assert data and data[0]["source_agent"] == "feedback_learner"
    finally:
        # Keep the live training corpus clean — remove the integration row.
        client.table("dspy_agent_training_signals").delete().eq("batch_id", batch).execute()
