"""Shared fixtures for orchestrator unit tests."""

import pytest


@pytest.fixture(autouse=True)
def _no_real_memory_contribution(monkeypatch):
    """Isolate unit tests from the live memory systems (#883 PR B).

    Since #883 PR B, ``OrchestratorAgent.run`` contributes every completed
    turn to memory (episodic insert + embedding call, working-memory cache,
    conversation turn, routing-decision signal). Without this guard every
    unit test that drives ``agent.run`` (e.g. ``allow_mock=True`` graph runs)
    from a creds-configured dev box would write REAL rows into the live DB —
    the 883-A lesson: once a previously-always-failing write starts
    SUCCEEDING, formerly de-facto-hermetic suites begin polluting the DB.

    Stub the symbol the agent's caller-side helper resolves
    (``src.agents.orchestrator.agent.contribute_to_memory``) with an honest
    no-op returning the hook's real "nothing stored" shape — NOT a fabricated
    success. The faithful persistence proof lives in
    ``tests/integration/test_agent_memory_wiring_883b.py``; the wiring unit
    tests in ``test_memory_wiring_883b.py`` re-patch this attribute with
    their own recorders (an inner monkeypatch wins over this fixture).

    READ side (#883 read-side deferral): ``agent.run`` now also hydrates
    ``conversation_history`` from working memory pre-graph
    (``_load_conversation_history``). Reads don't pollute, but they would
    open a REAL Redis connection from a creds-configured dev box (and add a
    hang/flake surface where Redis is absent, plus cross-test coupling: a
    real stored session could leak INTO a unit run) — same hermetic posture.
    The factory the agent module resolves is re-pointed at REAL
    ``OrchestratorMemoryHooks`` (construction does no I/O, so the
    lazy-singleton property contract pinned by ``test_memory_wiring_883b``
    stays intact) whose working-memory client is pre-seeded with an offline
    stub returning the honest empty store — the read path then resolves to
    the documented no-context case (None), never a fabricated history.
    Enrichment unit tests (``test_memory_read_enrichment_883c.py``) seed
    ``agent._memory_hooks`` directly (the lazy property short-circuits) or
    re-patch the factory; the faithful continuity proof lives in
    ``tests/integration/test_orchestrator_context_enrichment_883c.py``.
    """

    async def _stub(result, state, memory_hooks=None, session_id=None, brand=None, region=None):
        return {
            "episodic_stored": 0,
            "working_cached": 0,
            "conversation_stored": 0,
            "routing_tracked": 0,
        }

    class _OfflineWorkingMemory:
        """Truthy stand-in: an empty, unreachable-for-writes message store."""

        async def get_messages(self, session_id, limit=None):
            return []

    def _offline_hooks():
        from src.agents.orchestrator.memory_hooks import OrchestratorMemoryHooks

        hooks = OrchestratorMemoryHooks()
        hooks._working_memory = _OfflineWorkingMemory()
        return hooks

    monkeypatch.setattr("src.agents.orchestrator.agent.contribute_to_memory", _stub)
    monkeypatch.setattr(
        "src.agents.orchestrator.agent.get_orchestrator_memory_hooks",
        _offline_hooks,
    )

    # Dispatcher resolver tests in THIS tree drive the REAL ExplainerAgent
    # (test_dispatcher_input_resolver.py registers ExplainerAgent(use_llm=False));
    # its narrative_generator lazily resolves get_explanation_memory_hooks and
    # wrote real 'explanation_generated' episodic rows from a creds-configured
    # box (4 rows observed 2026-06-12 — the same pre-existing leak class #886
    # fixed for tool_composer). Both the node and the agent resolve the factory
    # via a call-time local import from src.agents.explainer.memory_hooks, so
    # patching the source symbol makes every lazy property see "memory hooks
    # unavailable" (None) — the paths' own honest no-op branch.
    monkeypatch.setattr(
        "src.agents.explainer.memory_hooks.get_explanation_memory_hooks",
        lambda: None,
    )
