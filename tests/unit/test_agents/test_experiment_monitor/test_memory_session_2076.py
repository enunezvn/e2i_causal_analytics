"""#2076: ExperimentMonitorAgent must not mint a session id before the hook.

``contribute_to_memory`` stopped minting a uuid for an absent session id, but
``run_async`` minted one a level higher — ``if session_id is None: session_id =
str(uuid.uuid4())`` — so for this one agent the invented identity survived the
fix and still reached ``episodic_memories.session_id``.

``session_id`` has exactly one consumer in this agent: the
``contribute_to_memory`` call. Nothing keys on it, so passing the caller's
``None`` straight through loses nothing and lets the writer store an honest NULL.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from src.agents.experiment_monitor.agent import (
    ExperimentMonitorAgent,
    ExperimentMonitorInput,
)

_AGENT_ATTR = "src.agents.experiment_monitor.agent.contribute_to_memory"
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


@pytest.fixture
def mock_graph():
    """A graph that returns a completed, alert-free monitoring state."""
    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "experiments": [],
        "alerts": [],
        "experiments_checked": 0,
        "monitor_summary": "All healthy",
        "recommended_actions": [],
        "check_latency_ms": 10,
        "errors": [],
        "status": "completed",
    }
    return graph


@pytest.fixture
def recorder(monkeypatch):
    """Record what run_async hands to contribute_to_memory."""
    calls: List[Dict[str, Any]] = []

    async def _record(
        result: Dict[str, Any],
        state: Dict[str, Any],
        memory_hooks: Any = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, int]:
        calls.append({"result": result, "state": state, "session_id": session_id})
        return {"alerts_stored": 0, "check_stored": 0, "working_cached": 0}

    monkeypatch.setattr(_AGENT_ATTR, _record)
    return calls


def _agent(mock_graph) -> ExperimentMonitorAgent:
    agent = ExperimentMonitorAgent()
    agent.graph = mock_graph
    return agent


@pytest.mark.asyncio
async def test_no_session_id_reaches_the_hook_as_none(mock_graph, recorder):
    """No session in -> None out. Never a minted uuid."""
    await _agent(mock_graph).run_async(ExperimentMonitorInput(query="check"))

    assert len(recorder) == 1
    got = recorder[0]["session_id"]
    assert not (isinstance(got, str) and _UUID_RE.match(got)), f"agent minted a session id: {got!r}"
    assert got is None


@pytest.mark.asyncio
async def test_caller_session_id_is_passed_through_unchanged(mock_graph, recorder):
    """The change is scoped to the session-less path."""
    session_id = "46d40f52-39ac-4b79-b3a4-1f1292059a00~eeba22e7-4d9d-49ea-977b-b9e9d1549c53"
    await _agent(mock_graph).run_async(ExperimentMonitorInput(query="check"), session_id=session_id)

    assert len(recorder) == 1
    assert recorder[0]["session_id"] == session_id
