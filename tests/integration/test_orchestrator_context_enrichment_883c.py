"""#883 read-side deferral: orchestrator second-turn conversation continuity.

PR #886 wired the orchestrator's WRITE side (every completed turn persists the
conversation turn into working memory, among the four per-turn writes) but
deferred the READ side as a latency-on-critical-path product call: nothing ever
consumed ``get_conversation_history`` / ``get_context``, so a second turn in
the same session started from a blank slate even though turn 1's messages sat
in Redis. ``OrchestratorState.conversation_history`` (a contract-documented
input field, CONTRACT_VALIDATION.md §1) was always ``None`` in production —
every prod call site (chatbot_tools/chatbot_graph/cognitive/copilotkit) passes
``session_id`` but never ``conversation_history``.

Fix under test: ``OrchestratorAgent.run`` hydrates ``conversation_history``
from working memory when the caller did not supply it — a single Redis LRANGE
under a HARD latency budget (``MEMORY_READ_BUDGET_SECONDS``, asyncio.wait_for;
fail-open to no-context on timeout/error, never fabricated). The consumer is
the intent classifier's LLM fallback (ambiguous follow-ups get the prior
turns as referent context — "session context for routing" per
CONTRACT_VALIDATION.md §10.3); the budget/fail-open proofs live in
``tests/unit/test_agents/test_orchestrator/test_memory_read_enrichment_883c.py``.

The deliberately NOT-wired reads (decision documented in agent.py): episodic +
semantic ``get_context`` reads would put an embedding API call + FalkorDB
round-trips on the <2s critical path with no node consuming the result, and
``get_routing_decisions``'s documented consumer is batch DSPy routing
optimization — per-request routing is a deterministic intent→agent map that
prior decisions cannot honestly change.

RED on main @ 59b4067a (quoted below): turn 2's graph input carried
``conversation_history=None`` while the probe read turn 1's messages straight
out of Redis.

Run with the shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_orchestrator_context_enrichment_883c.py'
"""

import os
import uuid

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-store continuity test; set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]


def _reset_redis_singletons() -> None:
    """Drop cached Redis singletons so this test binds fresh clients."""
    import src.memory.services.factories as factories
    import src.memory.working_memory as wm

    wm._working_memory = None
    factories._redis_client = None


def _cleanup_episodic_by_session(session_id: str) -> None:
    from src.memory.episodic_memory import get_supabase_client

    get_supabase_client().table("episodic_memories").delete().eq("session_id", session_id).execute()


class _GraphSpy:
    """Record the initial state handed to the real graph, then delegate."""

    def __init__(self, real_graph):
        self._real_graph = real_graph
        self.seen_states: list = []

    async def ainvoke(self, state):
        self.seen_states.append(dict(state))
        return await self._real_graph.ainvoke(state)


@pytest.mark.asyncio
@pytest.mark.timeout(120)  # measured ~12s on the live stores; x3 headroom + embedding variance
async def test_second_turn_receives_first_turns_persisted_context():
    """RED on main: after a fully real turn 1 (whose #886 write side stored the
    user+assistant messages in Redis — re-verified here via a probe), turn 2's
    graph input still carried ``conversation_history=None``. GREEN: turn 2
    hydrates the caller-unsupplied field from working memory and the prior
    turn's content reaches the graph; the turn's status is unaffected.

    Both queries are pattern-strong on purpose: the subject under test is the
    read-side hydration, not the LLM fallback (whose prompt-level consumption
    proof is a deterministic unit test).
    """
    from src.agents.orchestrator.agent import OrchestratorAgent
    from src.agents.orchestrator.memory_hooks import OrchestratorMemoryHooks

    _reset_redis_singletons()
    probe = OrchestratorMemoryHooks()
    if not probe.working_memory:
        pytest.skip("working memory (Redis) not reachable in this environment")
    try:
        await probe.working_memory.get_messages("883c-probe", limit=1)
    except Exception:
        pytest.skip("working memory (Redis) not responding in this environment")

    session_id = str(uuid.uuid4())
    marker1 = f"883c-turn1-{uuid.uuid4().hex[:12]}"
    marker2 = f"883c-turn2-{uuid.uuid4().hex[:12]}"

    agent = OrchestratorAgent(allow_mock=True, enable_opik=False)
    try:
        # ---- turn 1: fully real run; #886 write side persists the turn ----
        result1 = await agent.run(
            {
                "query": f"Why did Remibrutinib TRx drop in the midwest? ({marker1})",
                "session_id": session_id,
            }
        )
        assert result1["status"] in ("completed", "partial_success"), (
            f"turn 1 did not complete: {result1.get('status')}"
        )

        # Independent probe: the messages ARE in working memory (write side
        # proven in #886; re-checked so a turn-2 failure below is attributable
        # to the READ side, not a broken write).
        stored = await probe.get_conversation_history(session_id=session_id, limit=10)
        assert any(
            marker1 in (m.get("content") or "") for m in stored if m.get("role") == "user"
        ), "precondition failed: turn 1's user message is not in working memory"

        # ---- turn 2: same session; spy on the state handed to the graph ----
        spy = _GraphSpy(agent.graph)
        agent.graph = spy  # type: ignore[assignment]

        result2 = await agent.run(
            {
                "query": f"Why did Fabhalta NBRx rise in the northeast? ({marker2})",
                "session_id": session_id,
            }
        )
        assert result2["status"] in ("completed", "partial_success"), (
            f"turn 2 did not complete: {result2.get('status')} — the memory read "
            "must never poison the turn"
        )

        assert spy.seen_states, "graph spy captured no state"
        history = spy.seen_states[0].get("conversation_history")
        assert history, (
            "turn 2's graph input carried conversation_history=None even though "
            "turn 1's messages are sitting in working memory for this session — "
            "the read side is unwired (#883 deferral, PR #886 body)"
        )
        assert any(
            marker1 in (m.get("content") or "") for m in history if m.get("role") == "user"
        ), "hydrated history does not contain turn 1's user message"
        assert any(m.get("role") == "assistant" for m in history), (
            "hydrated history does not contain turn 1's assistant response"
        )
    finally:
        _cleanup_episodic_by_session(session_id)
        # Zero residue in working memory too: messages key + session hash.
        try:
            await probe.working_memory.clear_messages(session_id)
            await probe.working_memory.delete_session(session_id)
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_caller_supplied_history_is_not_overwritten():
    """A caller that supplies ``conversation_history`` keeps full authority —
    the hydration only fills the gap when the field is absent/None (the
    contract documents it as an optional INPUT; the read must not shadow it).
    """
    from src.agents.orchestrator.agent import OrchestratorAgent
    from src.agents.orchestrator.memory_hooks import OrchestratorMemoryHooks

    _reset_redis_singletons()
    probe = OrchestratorMemoryHooks()
    if not probe.working_memory:
        pytest.skip("working memory (Redis) not reachable in this environment")
    try:
        await probe.working_memory.get_messages("883c-probe", limit=1)
    except Exception:
        pytest.skip("working memory (Redis) not responding in this environment")

    session_id = str(uuid.uuid4())
    supplied = [{"role": "user", "content": "caller-supplied prior turn (883c)"}]

    agent = OrchestratorAgent(allow_mock=True, enable_opik=False)
    spy = _GraphSpy(agent.graph)
    agent.graph = spy  # type: ignore[assignment]
    try:
        result = await agent.run(
            {
                "query": "Why did Kisqali TRx drop in the west? (883c-supplied)",
                "session_id": session_id,
                "conversation_history": supplied,
            }
        )
        assert result["status"] in ("completed", "partial_success")
        assert spy.seen_states[0].get("conversation_history") == supplied
    finally:
        _cleanup_episodic_by_session(session_id)
        try:
            await probe.working_memory.clear_messages(session_id)
            await probe.working_memory.delete_session(session_id)
        except Exception:
            pass
