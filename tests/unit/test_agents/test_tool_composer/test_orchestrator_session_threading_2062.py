"""#2062: a composition recorded from the ORCHESTRATOR dispatch path must carry
the chat session (and the user, when the turn has one).

Context
-------
Real chat traffic reaches the Tool Composer through the orchestrator, never
through ``chatbot_tools.tool_composer_tool`` (``entry_point='chat_tool'`` has
never appeared in ``composer_episodes``). On that path the chain is:

    chatbot_graph.dispatch (session_id, user_id)
      -> OrchestratorAgent.run -> OrchestratorState
      -> DispatcherNode._prepare_agent_input   (builds the agent payload)
      -> ToolComposerAgent.run(input_data)     (builds ``merged_context``)
      -> ToolComposer.compose(context=...)     (``_recording_seed``)

``_recording_seed`` reads ``context["session_id"]`` / ``context["user_id"]``,
but ``ToolComposerAgent.run`` only merged ``context``/``extracted_entities``/
``user_context`` — the dispatcher's pass-through ``session_id`` was dropped on
the floor and ``user_id`` was never threaded at all. Every orchestrator-path
episode therefore landed with ``session_id = NULL``, and
``composition_feedback_tasks`` rejects such an episode before anything else
(``rating["session_id"] != episode["session_id"]``), so it can never be
labelled.

These tests build the agent payload with the REAL dispatcher method rather than
a hand-written dict. The previous coverage (the wiring test now named
``test_composer_recording_wiring.test_direct_invocation_stamps_the_orchestrator_entry_point``)
passed ``{"context": {"session_id": ...}}``, a shape the production caller
never produces, which is exactly how the defect hid. That test now asserts
only the ``entry_point`` stamp; every identity claim lives here.

Scope: the composition-feedback owner gate (``_same_owner``) is a separate
defect and is not touched here.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from src.agents.orchestrator.nodes.dispatcher import DispatcherNode
from src.agents.tool_composer import learning_recorder
from src.agents.tool_composer.agent import ToolComposerAgent
from src.agents.tool_composer.learning_recorder import drain
from tests.unit.test_agents.test_tool_composer.test_composer_recording_wiring import (
    QUERY,
    Capture,
    _composer,
)

SESSION = "1f6b6c9e-6f2a-4a5e-9a2f-2b7c9f0a1d33"
USER = "c0ffee00-dead-4bee-8fed-0123456789ab"


@pytest.fixture(autouse=True)
async def _no_leftover_recording():
    yield
    await drain(timeout=0, cancel_heartbeats=True)
    for task in list(learning_recorder._pending):
        task.cancel()
    await asyncio.sleep(0)


def _state(**overrides: Any) -> Dict[str, Any]:
    """An OrchestratorState-shaped dict as ``OrchestratorAgent.run`` builds it."""
    state: Dict[str, Any] = {
        "query": QUERY,
        "query_id": "q-2062",
        "user_id": USER,
        "session_id": SESSION,
        "user_context": {"brand": "Kisqali", "region": "NE"},
        "agent_results": [],
    }
    state.update(overrides)
    return state


def _dispatch() -> Dict[str, Any]:
    return {
        "agent_name": "tool_composer",
        "priority": "high",
        "parameters": {},
        "timeout_ms": 90000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


def _prepared(**state_overrides: Any) -> Dict[str, Any]:
    """The payload the dispatcher really hands ``ToolComposerAgent.run``.

    ``tool_composer`` is a ``run(dict)`` agent, so ``_dispatch_agent`` merges its
    resolver's output into this payload (``data``/``kpi_outcome``) and calls
    ``run(agent_input)`` — the resolver only ADDS keys, so this is the identity
    of what the agent sees for the fields under test.
    """
    node = DispatcherNode()
    return node._prepare_agent_input(_state(**state_overrides), _dispatch())  # type: ignore[arg-type]


async def _seed_for(payload: Dict[str, Any], mock_llm_client, mock_tool_registry) -> Dict[str, Any]:
    capture = Capture()
    agent = ToolComposerAgent()
    agent._composer = _composer(mock_llm_client, mock_tool_registry, capture)
    output = await agent.run(payload)
    assert await drain(timeout=10) == 0
    assert output.success is True
    assert capture.seeds, "no composition was recorded"
    return capture.seeds[0]


async def test_dispatched_episode_carries_the_chat_session(mock_llm_client, mock_tool_registry):
    """The recorded episode's ``session_id`` is the orchestrator state's session."""
    seed = await _seed_for(_prepared(), mock_llm_client, mock_tool_registry)
    assert seed["entry_point"] == "orchestrator_agent"
    assert seed["session_id"] == SESSION


async def test_dispatched_episode_carries_the_user(mock_llm_client, mock_tool_registry):
    """``user_id`` rides the same dispatch payload when the turn has one."""
    seed = await _seed_for(_prepared(), mock_llm_client, mock_tool_registry)
    assert seed["user_id"] == USER


async def test_absent_identity_stays_absent(mock_llm_client, mock_tool_registry):
    """An anonymous turn records NULL, never a synthesised id.

    ``chatbot_graph`` reads ``state.get("user_id", "")`` — an empty string is
    "no user", and must not be written as one.
    """
    seed = await _seed_for(
        _prepared(session_id=None, user_id=""), mock_llm_client, mock_tool_registry
    )
    assert seed["session_id"] is None
    assert seed["user_id"] is None


async def test_explicit_context_identity_is_not_clobbered(mock_llm_client, mock_tool_registry):
    """A direct caller that puts the identity in ``context`` keeps it when the
    payload carries no top-level ``session_id``/``user_id``.

    The dispatcher never builds a ``context`` key, so this payload is
    hand-written on purpose: it pins the direct-invocation contract, not the
    dispatch path.
    """
    payload = {"query": QUERY, "context": {"session_id": "sess-agent", "user_id": "u-agent"}}
    seed = await _seed_for(payload, mock_llm_client, mock_tool_registry)
    assert (seed["session_id"], seed["user_id"]) == ("sess-agent", "u-agent")


async def test_dispatched_identity_wins_over_a_context_supplied_one(
    mock_llm_client, mock_tool_registry
):
    """When the payload AND the merged context both carry an identity, the
    payload's (the orchestrator state's) is recorded.

    ``_prepare_agent_input`` never builds ``context``/``extracted_entities``; the
    one context channel it passes through is ``state["user_context"]``, which
    ``ToolComposerAgent.run`` spreads into ``merged_context`` before binding the
    top-level ids. A stale id riding in there must not beat the dispatcher's.
    """
    payload = _prepared(
        user_context={"brand": "Kisqali", "session_id": "stale-session", "user_id": "stale-user"}
    )
    assert payload["user_context"]["session_id"] == "stale-session"  # both really present
    seed = await _seed_for(payload, mock_llm_client, mock_tool_registry)
    assert (seed["session_id"], seed["user_id"]) == (SESSION, USER)
