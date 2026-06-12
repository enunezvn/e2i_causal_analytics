"""#883 §3 faithful integration: the 3 dead-via-chat agents dispatch FOR REAL.

Pre-#883, ``explainer`` / ``health_score`` / ``feedback_learner`` were the only
``AGENT_METHOD_MAP`` entries with ``uses_kwargs=True`` and neither an
``INPUT_RESOLVERS`` entry nor an ``input_model``: the dispatcher splatted the
generic orchestrator payload into the agent method and every chat dispatch died
with a raw TypeError (``... got an unexpected keyword argument 'user_context'``).
These tests drive the PRODUCTION dispatch path (real ``DispatcherNode`` + real
agent instances) against the live substrate and assert each agent now runs
honestly end-to-end:

* health_score — the 'system_health' intent (sole agent, no fallback) completes
  a REAL check; bonus (#881): a default-constructed agent's grade-F fail-closed
  measurement is always significant, so the wired memory hook must land an
  episodic row for the dispatch's session_id (the PR-A/migration-070 enum fixes
  are in this branch's base).
* explainer — the universal fallback agent binds ``analysis_results`` from a
  REAL upstream result seeded in the orchestrator state (the exact substrate
  the fallback path carries), and explains it.
* feedback_learner — the resolver grounds the learn window in the trailing
  default the 6h Celery beat uses, and the agent reports the HONEST outcome
  over the live substrate (zero feedback rows in the window is a legitimate,
  non-fabricated result).

Each test self-cleans the rows it creates. Run with the shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_dispatcher_dead_agents_883_realdb.py'
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB dispatch test; set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]


def _state(agent_name: str, query: str, session_id: str, timeout_ms: int = 60000) -> dict:
    return {
        "query": query,
        "user_context": {"user_id": "itest-883"},
        "session_id": session_id,
        "parsed_query": {"intent": agent_name, "entities": []},
        "dispatch_plan": [
            {
                "agent_name": agent_name,
                "priority": "critical",
                "parameters": {},
                "timeout_ms": timeout_ms,
                "fallback_agent": None,
                "execution_mode": "parallel",
            }
        ],
        "parallel_groups": [[agent_name]],
    }


def _episodic_rows(agent_name: str, session_id: str) -> list:
    from src.memory.episodic_memory import get_supabase_client

    resp = (
        get_supabase_client()
        .table("episodic_memories")
        .select("memory_id, session_id, event_type, outcome_type, agent_name")
        .eq("agent_name", agent_name)
        .eq("session_id", session_id)
        .execute()
    )
    return resp.data or []


def _cleanup_rows(memory_ids: list) -> None:
    from src.memory.episodic_memory import get_supabase_client

    client = get_supabase_client()
    for memory_id in memory_ids:
        client.table("episodic_memories").delete().eq("memory_id", memory_id).execute()


@pytest.mark.asyncio
async def test_health_score_chat_dispatch_runs_and_lands_memory() -> None:
    """'how healthy is the system?' through the REAL dispatcher completes a
    real check (RED pre-#883: TypeError on the generic-payload splat) and the
    #881-wired memory hook lands the episodic row for the dispatch session."""
    from src.agents.health_score import HealthScoreAgent
    from src.agents.orchestrator.nodes.dispatcher import DispatcherNode

    session_id = str(uuid.uuid4())  # episodic_memories.session_id is uuid-typed
    agent = HealthScoreAgent(enable_mlflow=False, enable_opik=False, enable_memory=True)
    node = DispatcherNode(agent_registry={"health_score": agent})

    memory_ids: list = []
    try:
        out = await node.execute(_state("health_score", "how healthy is the system?", session_id))

        res = out["agent_results"][0]
        assert res["success"] is True, res["error"]
        result = res["result"]
        assert result is not None
        assert result.get("status") == "completed"
        assert result.get("health_summary")
        # No backends wired => the F1 fail-closed composer measures 0 dimensions
        # -> grade F + critical issue. That is a REAL (honest) measurement, and
        # per _is_significant_health_event it is ALWAYS significant, so the
        # #881 wiring must have landed the episodic row for THIS session.
        rows = _episodic_rows("health_score", session_id)
        memory_ids = [r["memory_id"] for r in rows]
        assert rows, "the #881-wired memory contribution must land an episodic row"
        assert rows[0]["event_type"] == "health_check_completed"
    finally:
        _cleanup_rows(memory_ids)
        try:
            from src.agents.health_score.memory_hooks import get_health_score_memory_hooks

            await get_health_score_memory_hooks().invalidate_cache("full")
        except Exception:
            pass


@pytest.mark.asyncio
async def test_explainer_dispatch_binds_seeded_real_upstream_result() -> None:
    """A dispatch state carrying a REAL upstream AgentResult (the shape the
    fallback path and a checkpointer-resumed turn carry) must reach
    ``explain()`` as ``analysis_results`` and produce a real explanation."""
    from src.agents.explainer import ExplainerAgent
    from src.agents.health_score import HealthScoreAgent
    from src.agents.orchestrator.nodes.dispatcher import DispatcherNode

    # Seed: run a REAL upstream agent through the same dispatcher first.
    seed_session = str(uuid.uuid4())
    upstream_agent = HealthScoreAgent(enable_mlflow=False, enable_opik=False, enable_memory=False)
    seed_node = DispatcherNode(agent_registry={"health_score": upstream_agent})
    seed_out = await seed_node.execute(
        _state("health_score", "how healthy is the system?", seed_session)
    )
    upstream_result = seed_out["agent_results"][0]
    assert upstream_result["success"] is True, upstream_result["error"]

    session_id = str(uuid.uuid4())
    node = DispatcherNode(
        agent_registry={"explainer": ExplainerAgent(use_llm=False)},
    )
    state = _state("explainer", "explain those results for me", session_id)
    state["agent_results"] = [upstream_result]

    out = await node.execute(state)
    # operator.add semantics in the graph: the node's own results are appended;
    # here (direct node call) the seeded result is replaced by the new list.
    res = next(r for r in out["agent_results"] if r["agent_name"] == "explainer")
    assert res["success"] is True, res["error"]
    result = res["result"]
    assert result is not None
    assert result.get("status") == "completed"
    # The explanation is grounded in the REAL upstream output.
    assert result.get("executive_summary") or result.get("detailed_explanation")


@pytest.mark.asyncio
async def test_feedback_learner_dispatch_binds_real_window_honest_outcome() -> None:
    """The 'feedback' intent runs the REAL agent (production store wiring, the
    same builder the 6h beat uses) over the beat's trailing default window.
    Whatever the live substrate holds, the outcome must be HONEST: a completed
    cycle whose feedback_count matches reality — never fabricated learnings."""
    from src.agents.feedback_learner.agent import (
        FeedbackLearnerAgent,
        build_production_feedback_stores,
    )
    from src.agents.orchestrator.nodes import dispatcher as disp
    from src.agents.orchestrator.nodes.dispatcher import DispatcherNode

    # The resolver's window must mirror the Celery beat default (trailing 24h
    # ending now UTC) when the chat names no period.
    resolved = disp.INPUT_RESOLVERS["feedback_learner"](
        {"query": "what have we learned from feedback?", "parsed_query": {"entities": []}},
        {"agent_name": "feedback_learner", "parameters": {}},
    )
    assert isinstance(resolved, dict)
    start = datetime.fromisoformat(resolved["time_range_start"])
    end = datetime.fromisoformat(resolved["time_range_end"])
    default_hours = float(os.getenv("DSPY_LEARN_WINDOW_HOURS", "24"))
    assert abs((end - start) - timedelta(hours=default_hours)) < timedelta(seconds=10)
    assert abs(end - datetime.now(timezone.utc)) < timedelta(minutes=2)

    feedback_store, knowledge_stores, _db_client = await build_production_feedback_stores()
    agent = FeedbackLearnerAgent(
        feedback_store=feedback_store,
        knowledge_stores=knowledge_stores,
        use_llm=False,  # deterministic analysis; the binding is what's under test
        persist_signals=False,  # no training-signal side effects from a test run
    )
    node = DispatcherNode(agent_registry={"feedback_learner": agent})
    session_id = str(uuid.uuid4())

    out = await node.execute(
        _state("feedback_learner", "what have we learned from feedback?", session_id)
    )
    res = out["agent_results"][0]
    assert res["success"] is True, res["error"]
    result = res["result"]
    assert result is not None
    assert result.get("status") == "completed"
    feedback_count = result.get("feedback_count")
    assert isinstance(feedback_count, int) and feedback_count >= 0
    if feedback_count == 0:
        # The honest no-data outcome: nothing learned, nothing fabricated.
        assert not result.get("applied_updates")
