"""#2077: the chat tools record the real conversation and the real user.

Three gaps left open by #2064 / #2088:

1. **The bridge's shadow session reached its tools.** ``run_conversational_bridge``
   binds ``{session}~bridge`` (#1394's device: it keeps the bridged turn's raw
   answer out of the real session's UI history) and puts it in graph state, which
   is where ``SessionBoundToolNode`` now reads the tools' session from. So every
   bridged composition was keyed on an id no rating can ever carry
   (``composition_feedback_tasks`` matches ``session_id`` as an exact string).
   The bridge now names the tools' session explicitly, on the invocation config,
   and the persistence channels keep the shadow id untouched.
2. **``orchestrator_tool`` passed no ``user_id``**, so AG-UI-originated
   ``composer_episodes`` rows stayed NULL-owned. The user is resolved from the
   channels that actually survive to the tools, never fabricated.
3. The model no longer sees a ``session_id`` argument it can only guess.

Only the orchestrator, the composer and the chat model are faked; the graphs,
the bridge and the tool node are the real ones.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk

import src.api.routes.copilotkit as copilotkit_mod
from src.api.routes import chatbot_tools
from src.api.routes.chat_bridge import run_conversational_bridge
from src.api.routes.chatbot_tools import chat_session_id_context

USER = "46d40f52-39ac-4b79-b3a4-1f1292059a00"
REAL_SESSION = f"{USER}~0b7f7d6e-2c1a-4d7e-9a53-3f1f6a0c9e21"
BRIDGE_SESSION = f"{REAL_SESSION}~bridge"


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def run(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {
            "response_text": "ok",
            "response_confidence": 0.9,
            "agents_dispatched": [],
            "status": "completed",
        }


@pytest.fixture
def orchestrator(monkeypatch) -> _FakeOrchestrator:
    fake = _FakeOrchestrator()
    monkeypatch.setattr(chatbot_tools, "get_orchestrator", lambda: fake)
    return fake


class _ScriptedChatModel:
    """The bridge's fast chat leg asks for a tool; the standard synthesis leg answers.

    Each leg records the session bound in its own context, which is the channel
    ``chat_node`` persists on (var > state > config.thread_id).
    """

    def __init__(self, *, calls_tool: bool, seen: list[Optional[str]]) -> None:
        self._calls_tool = calls_tool
        self._seen = seen

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedChatModel":
        return self

    async def astream(self, messages: Any, *args: Any, **kwargs: Any):
        self._seen.append(chat_session_id_context.get())
        if self._calls_tool:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "orchestrator_tool",
                        "args": '{"query": "system health score"}',
                        "id": "call-2077",
                        "index": 0,
                    }
                ],
            )
        else:
            yield AIMessageChunk(content="The health score is 82.")


def _chat_node_boundaries():
    """chat_node's side-effect boundaries, so the real graph can run as a unit."""
    return (
        patch.object(copilotkit_mod, "_ensure_conversation_exists", AsyncMock(return_value=False)),
        patch.object(copilotkit_mod, "_persist_message_sync", MagicMock(return_value=None)),
        patch.object(copilotkit_mod, "_record_analytics_sync", MagicMock(return_value=None)),
        patch.object(copilotkit_mod, "_collect_copilot_learning_signal", AsyncMock()),
        patch.object(copilotkit_mod, "copilotkit_emit_state", AsyncMock()),
        patch.object(copilotkit_mod, "copilotkit_emit_message", AsyncMock()),
    )


# --------------------------------------------------------------- gap 1: the bridge


async def test_a_bridged_turn_gives_its_tools_the_real_session(orchestrator):
    """The real bridge, the real AG-UI graph, the real tool node.

    The tool must see the conversation a rating can be keyed on, while the
    persistence channel keeps the ``~bridge`` shadow id (#1394).
    """
    persistence_sessions: list[Optional[str]] = []

    def _fake_get_chat_llm(**kwargs: Any) -> _ScriptedChatModel:
        return _ScriptedChatModel(
            calls_tool=kwargs.get("model_tier") == "fast", seen=persistence_sessions
        )

    boundaries = _chat_node_boundaries()
    with (
        patch.object(copilotkit_mod, "get_chat_llm", _fake_get_chat_llm),
        boundaries[0],
        boundaries[1],
        boundaries[2],
        boundaries[3],
        boundaries[4],
        boundaries[5],
    ):
        answer = await run_conversational_bridge(query="health score?", session_id=REAL_SESSION)

    assert answer is not None, "the bridge produced no answer"
    assert len(orchestrator.payloads) == 1, "orchestrator_tool never ran"
    assert orchestrator.payloads[0]["session_id"] == REAL_SESSION
    # #1394 is intact: both legs still persist under the shadow session.
    assert persistence_sessions == [BRIDGE_SESSION, BRIDGE_SESSION], persistence_sessions


def _bound_session(state: Any, config: Any) -> Optional[str]:
    """What the tools node binds for this (state, config), then unbinds."""
    from src.api.routes.chat_session_binding import _bind_tool_session
    from src.api.routes.chatbot_tools import reset_chat_session_id

    token = _bind_tool_session(state, config)
    try:
        return chat_session_id_context.get()
    finally:
        if token is not None:
            reset_chat_session_id(token)


def test_the_tools_node_prefers_the_callers_tool_session_over_the_persisted_one():
    bound = _bound_session(
        {"session_id": BRIDGE_SESSION},
        {"configurable": {"thread_id": f"bridge~{REAL_SESSION}", "tool_session_id": REAL_SESSION}},
    )
    assert bound == REAL_SESSION


def test_the_tools_node_still_binds_the_state_session_without_a_tool_session():
    assert _bound_session({"session_id": REAL_SESSION}, {"configurable": {"thread_id": "t"}}) == (
        REAL_SESSION
    )
    assert _bound_session({"session_id": REAL_SESSION}, None) == REAL_SESSION


def test_the_tools_node_invents_nothing_when_neither_channel_carries_a_session():
    assert _bound_session({}, {"configurable": {}}) is None


# ------------------------------------------------- gap 2: who the turn belongs to


def test_a_composite_session_names_its_own_user():
    from src.api.routes.chat_identity import resolve_tool_user_id

    assert resolve_tool_user_id(REAL_SESSION) == USER


def test_a_bridged_session_names_its_own_user():
    """``{user}~{session}~bridge`` splits on the FIRST '~', as computed_user_id does."""
    from src.api.routes.chat_identity import resolve_tool_user_id

    assert resolve_tool_user_id(BRIDGE_SESSION) == USER


def test_a_bare_thread_falls_back_to_the_auth_gates_verified_user():
    """AG-UI thread ids are bare uuids, so the prefix yields nothing there."""
    from src.api.routes.chat_identity import resolve_tool_user_id
    from src.utils.llm_attribution import set_authenticated_user

    set_authenticated_user(USER)
    try:
        assert resolve_tool_user_id("0b7f7d6e-2c1a-4d7e-9a53-3f1f6a0c9e21") == USER
    finally:
        set_authenticated_user(None)


def test_no_identity_resolves_to_none_rather_than_a_fabricated_id():
    """composer_episodes.user_id is uuid-typed and the linker compares it as text."""
    from src.api.routes.chat_identity import resolve_tool_user_id
    from src.utils.llm_attribution import ANONYMOUS_USER_ID, set_authenticated_user

    set_authenticated_user(None)
    assert resolve_tool_user_id(None) is None
    assert resolve_tool_user_id("chatbot-20260914120000") is None
    assert resolve_tool_user_id(f"{ANONYMOUS_USER_ID}~0b7f7d6e-2c1a-4d7e-9a53-3f1f6a0c9e21") is None
    assert resolve_tool_user_id("not-a-uuid~0b7f7d6e-2c1a-4d7e-9a53-3f1f6a0c9e21") is None


async def test_the_auth_gate_channel_survives_the_keepalive_wrapper_and_attribution_does_not():
    """Why the issue's proposed source (``get_attribution()``) cannot be used here.

    ``with_sse_keepalive`` pulls each frame via ``asyncio.ensure_future``, so a
    contextvar set while one frame is produced is gone by the next pull. The auth
    gate sets its value in the REQUEST task, upstream of the wrapper, so it
    survives — measured 2026-09-14 against the real wrapper.
    """
    from src.api.routes.chat_identity import resolve_tool_user_id
    from src.api.utils.sse_keepalive import with_sse_keepalive
    from src.utils.llm_attribution import (
        get_attribution,
        set_authenticated_user,
        set_chat_attribution,
    )

    thread = "cf6b364a-11f1-4b26-a5c0-f6559d06f659"
    seen: dict[str, Any] = {}

    async def body():
        set_chat_attribution(thread, "run-2077")
        yield "frame-1"
        attribution = get_attribution()
        seen["attribution_user"] = None if attribution is None else attribution.user_id
        seen["resolved_user"] = resolve_tool_user_id(thread)
        yield "frame-2"

    set_authenticated_user(USER)
    try:
        async for _ in with_sse_keepalive(body(), interval_seconds=5.0):
            pass
    finally:
        set_authenticated_user(None)

    assert seen["attribution_user"] is None, "get_attribution() unexpectedly survived"
    assert seen["resolved_user"] == USER
