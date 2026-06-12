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
    """

    async def _stub(result, state, memory_hooks=None, session_id=None, brand=None, region=None):
        return {
            "episodic_stored": 0,
            "working_cached": 0,
            "conversation_stored": 0,
            "routing_tracked": 0,
        }

    monkeypatch.setattr("src.agents.orchestrator.agent.contribute_to_memory", _stub)
