"""Shared fixtures for observability unit tests."""

import pytest


@pytest.fixture(autouse=True)
def _no_real_memory_io(monkeypatch):
    """Isolate this tree from the live memory systems (the #883-A pattern).

    ``test_partial_failure.py`` drives the REAL ``OrchestratorAgent.run``;
    since #886 every completed turn contributes to memory (episodic insert +
    embedding call, working-memory cache, conversation turn, routing signal),
    and ``contribute_to_memory`` generates a random session_id when none is
    passed — from a creds-configured dev box this file deposited 2 real
    ``orchestration_completed`` rows in one run (observed 2026-06-12 during
    the #883 read-side ripple; this tree simply predates the per-agent
    conftest guards and never got one).

    Mirror tests/unit/test_agents/test_orchestrator/conftest.py: stub the
    write with an honest "nothing stored" no-op and point the hooks factory
    at real hooks whose working-memory client is an offline empty store, so
    the #883 read-side hydration path resolves to the documented no-context
    case without opening a Redis connection.
    """

    async def _stub(result, state, memory_hooks=None, session_id=None, brand=None, region=None):
        return {
            "episodic_stored": 0,
            "working_cached": 0,
            "conversation_stored": 0,
            "routing_tracked": 0,
        }

    class _OfflineWorkingMemory:
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
