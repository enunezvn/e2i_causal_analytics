"""
Tests for finalize-node persistence wiring (task A1).

These tests prove:
A: graph.py finalize node persists via injected client (persist_signals=True)
B: graph.py finalize node skips persistence when persist_signals=False
C: FeedbackLearnerAgent.learn() persists EXACTLY ONCE (not double-persist)
D: API path (_execute_learning_cycle) persists via default factory client

All tests use a counting fake client; no real Supabase is touched.
The directory conftest patches get_supabase_client→None by default.
"""

from __future__ import annotations

import pytest

from src.agents.feedback_learner.graph import build_feedback_learner_graph

# ---------------------------------------------------------------------------
# Minimal graph invocation state
# ---------------------------------------------------------------------------

_MINIMAL_STATE = {
    "batch_id": "t",
    "time_range_start": "2026-01-01T00:00:00+00:00",
    "time_range_end": "2026-01-02T00:00:00+00:00",
    "focus_agents": [],
    "status": "pending",
    "errors": [],
    "warnings": [],
}


# ---------------------------------------------------------------------------
# Fake counting client — sync execute() is fine; signal_store._maybe_await handles it
# ---------------------------------------------------------------------------


class _CountingClient:
    """Sync fake Supabase client; records every insert into dspy_agent_training_signals."""

    def __init__(self):
        self.inserts: list[dict] = []
        self._pending: list[dict] | None = None
        self._table_name: str | None = None

    def table(self, name: str) -> "_CountingClient":
        self._table_name = name
        return self

    def insert(self, record) -> "_CountingClient":
        self._pending = record if isinstance(record, list) else [record]
        return self

    def execute(self):
        if self._table_name == "dspy_agent_training_signals" and self._pending is not None:
            self.inserts.extend(self._pending)
        result = {"data": self._pending}
        self._pending = None
        return result

    @property
    def insert_count(self) -> int:
        return len(self.inserts)


class _ExplodingClient:
    """Fake Supabase client whose .execute() raises — simulates a DB outage."""

    def table(self, name: str) -> "_ExplodingClient":
        return self

    def insert(self, record) -> "_ExplodingClient":
        return self

    def execute(self):
        raise RuntimeError("simulated DB outage during persist")


# ---------------------------------------------------------------------------
# Test A: graph with persist_signals=True persists exactly 1 row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_graph_persist_signals_true_inserts_once():
    """Build graph with persist_signals=True; invoking it inserts exactly 1 row."""
    fake = _CountingClient()
    graph = build_feedback_learner_graph(
        use_llm=False,
        enable_rubric_evaluation=False,
        persist_signals=True,
        persist_client=fake,
    )
    await graph.ainvoke(dict(_MINIMAL_STATE))
    assert fake.insert_count == 1, (
        f"Expected exactly 1 insert into dspy_agent_training_signals, got {fake.insert_count}"
    )
    assert fake.inserts[0]["source_agent"] == "feedback_learner"
    # The persisted row must carry the input batch_id (like Test C asserts)
    assert fake.inserts[0]["batch_id"] == _MINIMAL_STATE["batch_id"]


# ---------------------------------------------------------------------------
# Test B: graph with persist_signals=False skips persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b_graph_persist_signals_false_no_inserts():
    """Build graph with persist_signals=False; invoking it inserts 0 rows."""
    fake = _CountingClient()
    graph = build_feedback_learner_graph(
        use_llm=False,
        enable_rubric_evaluation=False,
        persist_signals=False,
        persist_client=fake,
    )
    await graph.ainvoke(dict(_MINIMAL_STATE))
    assert fake.insert_count == 0, (
        f"Expected 0 inserts when persist_signals=False, got {fake.insert_count}"
    )


# ---------------------------------------------------------------------------
# Test C: FeedbackLearnerAgent.learn() persists EXACTLY ONCE (not double-persist)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_c_agent_learn_persists_exactly_once():
    """Agent.learn() with persist_client must produce exactly 1 DB insert.

    This verifies that after moving persistence into the finalize node,
    the old learn() block was removed so there's no double-persist.
    """
    from src.agents.feedback_learner.agent import FeedbackLearnerAgent

    fake = _CountingClient()
    agent = FeedbackLearnerAgent(
        use_llm=False,
        persist_signals=True,
        persist_client=fake,
    )
    out = await agent.learn(
        time_range_start="2026-01-01T00:00:00+00:00",
        time_range_end="2026-01-02T00:00:00+00:00",
        batch_id="test-single-persist",
    )
    assert out.status in {"completed", "failed"}, f"Unexpected status: {out.status}"
    assert fake.insert_count == 1, (
        f"Expected exactly 1 DB insert (no double-persist), got {fake.insert_count}"
    )
    assert fake.inserts[0]["batch_id"] == "test-single-persist"


# ---------------------------------------------------------------------------
# Test D: API path (_execute_learning_cycle) persists via patched factory client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_d_api_path_persists_via_factory_client(monkeypatch):
    """_execute_learning_cycle must persist the signal via the default factory client.

    The directory conftest patches get_supabase_client→None; here we override
    that to a counting fake so we can assert exactly 1 insert happens.

    signal_store.persist_training_signal does a lazy import of get_supabase_client
    from src.memory.services.factories — patching that module attribute is enough.
    """
    fake = _CountingClient()

    # Override the factory used by signal_store when client=None (lazy import path)
    monkeypatch.setattr(
        "src.memory.services.factories.get_supabase_client",
        lambda: fake,
    )

    from src.api.routes.feedback import RunLearningRequest, _execute_learning_cycle

    request = RunLearningRequest(
        time_range_start="2026-01-01T00:00:00+00:00",
        time_range_end="2026-01-02T00:00:00+00:00",
    )
    response = await _execute_learning_cycle(request)
    assert response is not None
    assert fake.insert_count == 1, (
        f"Expected 1 insert from API path (_execute_learning_cycle), got {fake.insert_count}"
    )


# ---------------------------------------------------------------------------
# Test E: a DB error during persist must NOT fail the node (best-effort)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e_persist_db_error_does_not_fail_node():
    """A DB outage during persist must not fail the run; finalize still completes.

    persist_training_signal swallows the error and the finalize_node closure
    wraps the call in try/except, so the run must complete normally and the
    training_signal must still be present in the result.
    """
    exploding = _ExplodingClient()
    graph = build_feedback_learner_graph(
        use_llm=False,
        enable_rubric_evaluation=False,
        persist_signals=True,
        persist_client=exploding,
    )
    result = await graph.ainvoke(dict(_MINIMAL_STATE))
    # The DB error must NOT have flipped the run to failed.
    assert result.get("status") != "failed", (
        f"DB outage must not fail the node; status={result.get('status')}"
    )
    assert result.get("status") == "completed"
    # The finalized signal must still be present despite the persist failure.
    training_signal = result.get("training_signal")
    assert training_signal is not None
    assert hasattr(training_signal, "compute_reward")
