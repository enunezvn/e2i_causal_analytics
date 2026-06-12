"""#883 PR B unit tests: OrchestratorAgent.run must WIRE contribute_to_memory.

PR A (#883) proved the orchestrator memory HOOK persists (migration 071 +
outcome remap, direct-call proof in
tests/integration/test_agent_memory_persistence_883.py). PR B is about the
CALLER — ``OrchestratorAgent.run`` never imported memory_hooks (graph.py's
``MemorySaver`` is an unrelated langgraph checkpointer), so production wrote
no conversation turns / orchestration records / routing decisions despite a
working hook and CONTRACT_VALIDATION.md §10 marking memory integration
BLOCKING. These tests pin the wiring contract (the faithful real-DB proof
lives in ``tests/integration/test_agent_memory_wiring_883b.py``):

* enable_memory=True (default) -> contribute_to_memory called once post-run
  with the built output, the FINAL graph state, and the caller's session_id;
* enable_memory=False          -> no call attempted at all;
* 046-trap posture             -> a RAISING contribution (fabricated failure,
  not a fabricated result) never changes the run's status/response;
* graph failure                -> nothing contributed (no trustworthy turn);
* memory_hooks property        -> lazy, gated, init-failure-swallowing.

The recorder/raiser patches below re-patch the conftest autouse offline guard
(``_no_real_memory_contribution``) — same attribute, test-local behavior.
"""

from typing import Any, Dict, Optional

import pytest

from src.agents.orchestrator.agent import OrchestratorAgent

_AGENT_ATTR = "src.agents.orchestrator.agent.contribute_to_memory"


def _make_agent(**kwargs: Any) -> OrchestratorAgent:
    """Test agent: mock dispatch scaffold on, opik off (no I/O in unit scope)."""
    kwargs.setdefault("allow_mock", True)
    kwargs.setdefault("enable_opik", False)
    return OrchestratorAgent(**kwargs)


@pytest.fixture()
def recorder(monkeypatch):
    """Record the kwargs run() hands to contribute_to_memory."""
    calls = []

    async def _record(
        result: Dict[str, Any],
        state: Dict[str, Any],
        memory_hooks=None,
        session_id: Optional[str] = None,
        brand: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Dict[str, int]:
        calls.append(
            {
                "result": result,
                "state": state,
                "memory_hooks": memory_hooks,
                "session_id": session_id,
            }
        )
        return {
            "episodic_stored": 0,
            "working_cached": 0,
            "conversation_stored": 0,
            "routing_tracked": 0,
        }

    monkeypatch.setattr(_AGENT_ATTR, _record)
    return calls


class TestMemoryWiringEnabled:
    """enable_memory=True (the default): the post-run contribution fires."""

    @pytest.mark.asyncio
    async def test_run_contributes_once_with_output_state_session(self, recorder):
        agent = _make_agent()
        result = await agent.run(
            {
                "query": "what drove the TRx drop?",
                "session_id": "session-883b",
                "user_id": "u-883b",
            }
        )

        assert len(recorder) == 1, "run must contribute to memory exactly once"
        call = recorder[0]
        # The OUTPUT the caller returns is what gets stored.
        assert call["result"]["query_id"] == result["query_id"]
        assert call["result"]["status"] == result["status"]
        assert call["result"]["response_text"] == result["response_text"]
        # The FINAL graph state (not the initial state) rides along.
        assert call["state"].get("query") == "what drove the TRx drop?"
        assert call["state"].get("status") in ("completed", "partial_success")
        # Session plumbing: the caller's session id reaches the hook.
        assert call["session_id"] == "session-883b"

    @pytest.mark.asyncio
    async def test_default_agent_has_memory_enabled(self):
        agent = _make_agent()
        assert agent.enable_memory is True

    @pytest.mark.asyncio
    async def test_no_session_id_passes_none_through(self, recorder):
        """Without a session, None reaches the hook (which falls back to the
        state's session_id, then a generated UUID)."""
        agent = _make_agent()
        await agent.run({"query": "test query"})

        assert len(recorder) == 1
        assert recorder[0]["session_id"] is None


class TestMemoryWiringDisabled:
    """enable_memory=False: no contribution is attempted at all."""

    @pytest.mark.asyncio
    async def test_run_skips_memory_when_disabled(self, recorder):
        agent = _make_agent(enable_memory=False)
        result = await agent.run({"query": "test query", "session_id": "session-883b-off"})

        assert recorder == [], "enable_memory=False must not attempt any contribution"
        assert result["status"] == "completed"

    def test_memory_hooks_property_returns_none_when_disabled(self):
        agent = _make_agent(enable_memory=False)
        assert agent.memory_hooks is None


class TestMemoryFailureNonBlocking:
    """046-trap posture: a memory failure can never poison the turn."""

    @pytest.mark.asyncio
    async def test_raising_contribution_does_not_change_output(self, monkeypatch):
        async def _boom(*args, **kwargs):
            raise RuntimeError("fabricated memory failure (046-trap probe)")

        monkeypatch.setattr(_AGENT_ATTR, _boom)

        agent = _make_agent()
        result = await agent.run({"query": "test query", "session_id": "session-883b-trap"})

        # The turn completed; the swallow happened caller-side.
        assert result["status"] == "completed"
        assert result["response_text"] != ""
        assert "046-trap probe" not in str(result.get("orchestrator_error") or "")

    @pytest.mark.asyncio
    async def test_graph_failure_contributes_nothing(self, recorder):
        """When the GRAPH itself raises there is no trustworthy turn: no
        output is built and nothing is contributed."""

        class _ExplodingGraph:
            async def ainvoke(self, state):
                raise RuntimeError("fabricated graph failure")

        agent = _make_agent()
        agent.graph = _ExplodingGraph()

        with pytest.raises(RuntimeError, match="fabricated graph failure"):
            await agent.run({"query": "test query", "session_id": "session-883b-graphfail"})

        assert recorder == [], "a failed graph must not contribute a fabricated turn"

    @pytest.mark.asyncio
    async def test_baseline_output_identical_with_and_without_memory(self, recorder):
        """The contribution is observability-only: the agent's output is
        equivalent whether memory is on or off (ids/latency aside)."""
        on = await _make_agent().run({"query": "compare trx by region"})
        off = await _make_agent(enable_memory=False).run({"query": "compare trx by region"})

        for field in (
            "status",
            "agents_dispatched",
            "successful_agents",
            "failed_agents",
            "has_partial_failure",
            "intent_classified",
        ):
            assert on[field] == off[field], field


class TestMemoryHooksProperty:
    """Lazy hooks accessor mirrors the #879 convention."""

    def test_lazy_singleton_when_enabled(self):
        from src.agents.orchestrator.memory_hooks import OrchestratorMemoryHooks

        agent = _make_agent()
        hooks = agent.memory_hooks
        assert isinstance(hooks, OrchestratorMemoryHooks)
        assert agent.memory_hooks is hooks

    def test_init_failure_is_swallowed(self, monkeypatch):
        def _boom():
            raise RuntimeError("hooks factory down")

        monkeypatch.setattr("src.agents.orchestrator.agent.get_orchestrator_memory_hooks", _boom)
        agent = _make_agent()
        assert agent.memory_hooks is None
