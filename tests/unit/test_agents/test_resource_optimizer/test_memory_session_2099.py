"""#2099: ResourceOptimizerAgent must not mint a session id before the hooks.

``optimize`` minted ``str(uuid.uuid4())`` whenever the caller passed no session,
so every session-less optimization persisted an ``episodic_memories.session_id``
that belongs to no conversation -- the same invented-identity class #2076 fixed
one level down.

The write side already accepts ``Optional[str]`` (#2094). The read side did not:
``get_context`` was typed ``str`` and keyed working-memory lookups on it. With no
session there is no working-memory conversation and no session-keyed cache to
find, so both reads return empty rather than querying under an invented key.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from src.agents.resource_optimizer.agent import ResourceOptimizerAgent
from src.agents.resource_optimizer.memory_hooks import (
    OptimizationContext,
    ResourceOptimizerMemoryHooks,
)

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_SESSION = "eeba22e7-4d9d-49ea-977b-b9e9d1549c53"


class _FakeHooks:
    """Records the session id ``optimize`` hands to the context read."""

    def __init__(self) -> None:
        self.get_context_calls: List[Optional[str]] = []

    async def get_context(self, session_id: Optional[str], **_kwargs: Any) -> OptimizationContext:
        self.get_context_calls.append(session_id)
        return OptimizationContext(session_id=session_id)


@pytest.fixture
def mock_graph():
    """A graph that returns a completed, empty optimization state."""
    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "optimal_allocations": [],
        "objective_value": 0.0,
        "solver_status": "optimal",
        "optimization_summary": "",
        "recommendations": [],
        "total_latency_ms": 1,
        "timestamp": "2026-09-14T00:00:00+00:00",
        "status": "completed",
        "errors": [],
        "warnings": [],
    }
    return graph


@pytest.fixture
def recorder(monkeypatch):
    """Record what ``optimize`` hands to ``contribute_to_memory``."""
    calls: List[Dict[str, Any]] = []

    async def _record(
        result: Dict[str, Any],
        state: Dict[str, Any],
        memory_hooks: Any = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, int]:
        calls.append({"session_id": session_id})
        return {"episodic_stored": 0, "working_cached": 0, "pattern_learned": 0}

    monkeypatch.setattr("src.agents.resource_optimizer.memory_hooks.contribute_to_memory", _record)
    return calls


def _agent(mock_graph) -> tuple[ResourceOptimizerAgent, _FakeHooks]:
    agent = ResourceOptimizerAgent(enable_opik=False, enable_memory=True)
    agent._simple_graph = mock_graph
    hooks = _FakeHooks()
    agent._memory_hooks = hooks  # type: ignore[assignment]
    return agent, hooks


@pytest.mark.asyncio
async def test_no_session_id_reaches_the_writer_as_none(mock_graph, recorder):
    """No session in -> None out. Never a minted uuid."""
    agent, hooks = _agent(mock_graph)

    await agent.optimize(allocation_targets=[], constraints=[])

    assert len(recorder) == 1
    got = recorder[0]["session_id"]
    assert not (isinstance(got, str) and _UUID_RE.match(got)), f"agent minted a session id: {got!r}"
    assert got is None
    assert hooks.get_context_calls == [None]


@pytest.mark.asyncio
async def test_caller_session_id_is_passed_through_unchanged(mock_graph, recorder):
    """The change is scoped to the session-less path."""
    agent, hooks = _agent(mock_graph)

    await agent.optimize(allocation_targets=[], constraints=[], session_id=_SESSION)

    assert recorder == [{"session_id": _SESSION}]
    assert hooks.get_context_calls == [_SESSION]


@pytest.mark.asyncio
async def test_working_memory_read_is_skipped_without_a_session():
    """A session-less working-memory read would query under an invented key."""
    hooks = ResourceOptimizerMemoryHooks()
    working_memory = AsyncMock()
    hooks._working_memory = working_memory

    assert await hooks._get_working_memory_context(None) == []
    working_memory.get_messages.assert_not_called()


@pytest.mark.asyncio
async def test_session_cache_read_is_skipped_without_a_session():
    """The cache key embeds the session id; without one there is nothing to read."""
    hooks = ResourceOptimizerMemoryHooks()
    working_memory = AsyncMock()
    hooks._working_memory = working_memory

    assert await hooks._get_cached_optimization(None) is None
    working_memory.get_client.assert_not_called()
