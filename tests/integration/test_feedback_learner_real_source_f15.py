"""F15 (21-agent audit, MED): the feedback-learner DSPy loop must learn from the
REAL feedback source (``chatbot_message_feedback``), not an unwired empty store
that pinned ``update_effectiveness`` at 0.0.

- ``ChatbotFeedbackRepository.get_feedback(...)`` is the collector-shaped query.
- The prod task constructs ``FeedbackLearnerAgent(feedback_store=...)`` with it.

The real-DB tests are opt-in (the docker supabase-db holds the real feedback
rows); the no-client test runs in CI.
"""

import os

import pytest

from src.repositories.chatbot_feedback import (
    ChatbotFeedbackRepository,
    get_chatbot_feedback_repository,
)

_DB = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)

# Wide window so the real (months-old) feedback rows are captured regardless of age.
_WIDE_START = "2000-01-01T00:00:00+00:00"
_WIDE_END = "2100-01-01T00:00:00+00:00"


@pytest.fixture(autouse=True)
def _fresh_async_supabase_client():
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    yield
    factories._async_supabase_client = None


async def test_get_feedback_fails_closed_without_client():
    """No Supabase client -> honest empty list (never raises)."""
    repo = ChatbotFeedbackRepository(supabase_client=None)
    assert await repo.get_feedback(start_time=_WIDE_START, end_time=_WIDE_END) == []


def test_pattern_analyzer_processes_real_string_ratings():
    """F15 (audit, codex round-1): the deterministic pattern analyzer must
    process the REAL string ratings (``thumbs_down``) the chatbot feedback
    table emits. The prior numeric-only filter dropped every string rating, so
    collected feedback produced zero patterns and ``update_effectiveness``
    stayed pinned at 0.0 through a different path. Negative string ratings must
    now yield a low-rating pattern (the head of the patterns->updates chain)."""
    from src.agents.feedback_learner.nodes.pattern_analyzer import PatternAnalyzerNode

    node = PatternAnalyzerNode(use_llm=False)
    items = [
        {
            "feedback_id": f"f{i}",
            "timestamp": "",
            "feedback_type": "rating",
            "source_agent": "copilotkit",
            "query": "q",
            "agent_response": "r",
            "user_feedback": "thumbs_down",
            "metadata": {},
        }
        for i in range(4)
    ]

    result = node._analyze_deterministic({"feedback_items": items, "feedback_summary": {}})

    patterns = result["patterns"]
    assert any(p["pattern_type"] == "accuracy_issue" for p in patterns), (
        "real string thumbs_down ratings must produce a low-rating pattern "
        "(were silently dropped before F15)"
    )


@_DB
async def test_get_feedback_reads_real_chatbot_feedback():
    """get_feedback returns the REAL chatbot_message_feedback rows mapped into
    the shape the feedback-learner collector consumes."""
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    repo = get_chatbot_feedback_repository(supabase_client=client)

    rows = await repo.get_feedback(start_time=_WIDE_START, end_time=_WIDE_END)

    assert len(rows) > 0, "expected real chatbot_message_feedback rows"
    sample = rows[0]
    # Collector-consumed keys must be present.
    for key in ("id", "timestamp", "agent", "query", "response", "metadata"):
        assert key in sample, f"missing collector key {key!r} in {sample}"


@_DB
async def test_feedback_learner_collects_real_feedback_end_to_end():
    """FeedbackLearnerAgent wired with the REAL feedback_store collects > 0
    feedback items over a wide window — proving the loop is no longer starved by
    an unwired empty store (F15). use_llm=False keeps this offline/deterministic;
    feedback collection does not depend on the LLM."""
    from src.agents.feedback_learner.agent import FeedbackLearnerAgent
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    feedback_store = get_chatbot_feedback_repository(supabase_client=client)

    agent = FeedbackLearnerAgent(
        feedback_store=feedback_store, use_llm=False, persist_signals=False
    )
    output = await agent.learn(time_range_start=_WIDE_START, time_range_end=_WIDE_END)

    assert output.feedback_count > 0, (
        "wired feedback_store must collect real feedback (was 0 with the empty store)"
    )


@_DB
async def test_feedback_learner_without_store_collects_zero_contrast():
    """Contrast (the pre-F15 prod construction): with NO feedback_store the loop
    is starved — feedback_count is 0. This is what F15 changes by wiring the
    real store above."""
    from src.agents.feedback_learner.agent import FeedbackLearnerAgent

    agent = FeedbackLearnerAgent(use_llm=False, persist_signals=False)
    output = await agent.learn(time_range_start=_WIDE_START, time_range_end=_WIDE_END)

    assert output.feedback_count == 0
