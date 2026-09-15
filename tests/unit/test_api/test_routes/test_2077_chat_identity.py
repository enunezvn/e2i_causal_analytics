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


def test_a_composite_session_names_its_own_user(no_authenticated_user):
    from src.api.routes.chat_identity import resolve_tool_user_id

    assert resolve_tool_user_id(REAL_SESSION) == USER


def test_a_bridged_session_names_its_own_user(no_authenticated_user):
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
    """composer_episodes.user_id is VARCHAR(100), so the column rejects nothing;
    the linker compares it as text, so a wrong value mis-attributes silently."""
    from src.api.routes.chat_identity import resolve_tool_user_id
    from src.utils.llm_attribution import ANONYMOUS_USER_ID, set_authenticated_user

    set_authenticated_user(None)
    assert resolve_tool_user_id(None) is None
    assert resolve_tool_user_id("chatbot-20260914120000") is None
    assert resolve_tool_user_id(f"{ANONYMOUS_USER_ID}~0b7f7d6e-2c1a-4d7e-9a53-3f1f6a0c9e21") is None
    assert resolve_tool_user_id("not-a-uuid~0b7f7d6e-2c1a-4d7e-9a53-3f1f6a0c9e21") is None


async def test_both_identity_channels_survive_the_keepalive_wrapper():
    """Both channels reach a later frame pull — and the verified one still wins.

    When #2077 was written only the auth gate's channel survived: the wrapper
    pulled each frame in a fresh task, so the attribution a body set while
    producing one frame was gone by the next pull. #2100 gave every frame pull
    one shared context, so ``get_attribution()`` survives too — which is why
    chat usage rows, conversation ownership and token counts came back.

    That does NOT demote the verified channel. ``resolve_tool_user_id`` reads it
    first because a ``{user}~`` session prefix is a claim the caller makes, not a
    credential: AG-UI reads ``threadId`` straight from the request body. Both
    channels agreeing here is the point — the surviving attribution derives its
    user from the same verified id.
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

    assert seen["attribution_user"] == USER, "get_attribution() did not survive the frame pull"
    assert seen["resolved_user"] == USER


# ------------------------------------------- gap 2: the resolved user reaches the episode


BARE_THREAD = "cf6b364a-11f1-4b26-a5c0-f6559d06f659"  # the shape CopilotKit mints


@pytest.fixture
def failing_composer(monkeypatch) -> list[dict]:
    """Capture the composer context, then fail so the orchestrator fallback also runs."""
    contexts: list[dict] = []

    async def compose_query(*, query, context):
        contexts.append(context)
        raise RuntimeError("composition failed")

    monkeypatch.setattr(chatbot_tools, "compose_query", compose_query)
    monkeypatch.setattr(chatbot_tools.kpi_resolution, "recognize_kpi", lambda query: None)
    monkeypatch.setattr(chatbot_tools, "_resolve_cohort_frame", lambda *args: None)
    return contexts


@pytest.fixture
def no_authenticated_user():
    from src.utils.llm_attribution import set_authenticated_user

    set_authenticated_user(None)
    yield
    set_authenticated_user(None)


async def test_orchestrator_tool_records_the_user_behind_the_session(
    orchestrator, no_authenticated_user
):
    token = chat_session_id_context.set(REAL_SESSION)
    try:
        await chatbot_tools.orchestrator_tool.ainvoke({"query": "Why is TRx moving?"})
    finally:
        chat_session_id_context.reset(token)

    assert orchestrator.payloads[0]["user_id"] == USER


async def test_an_agui_thread_records_the_auth_gates_user(orchestrator, no_authenticated_user):
    """A bare thread id carries no prefix, so only the verified gate names the user."""
    from src.utils.llm_attribution import set_authenticated_user

    set_authenticated_user(USER)
    token = chat_session_id_context.set(BARE_THREAD)
    try:
        await chatbot_tools.orchestrator_tool.ainvoke({"query": "Why is TRx moving?"})
    finally:
        chat_session_id_context.reset(token)

    assert orchestrator.payloads[0]["user_id"] == USER


async def test_orchestrator_tool_records_null_rather_than_a_fabricated_user(
    orchestrator, no_authenticated_user
):
    await chatbot_tools.orchestrator_tool.ainvoke({"query": "Why is TRx moving?"})

    assert orchestrator.payloads[0]["user_id"] is None
    assert orchestrator.payloads[0]["session_id"] is None


async def test_tool_composer_tool_records_the_user_on_the_composition(
    orchestrator, failing_composer, no_authenticated_user
):
    token = chat_session_id_context.set(REAL_SESSION)
    try:
        await chatbot_tools.tool_composer_tool.ainvoke({"query": "Compare TRx and explain why"})
    finally:
        chat_session_id_context.reset(token)

    assert failing_composer[0]["user_id"] == USER
    # The orchestrator fallback behind a failed composition records it too.
    assert orchestrator.payloads[0]["user_id"] == USER


async def test_tool_composer_tool_records_null_when_no_identity_is_available(
    orchestrator, failing_composer, no_authenticated_user
):
    await chatbot_tools.tool_composer_tool.ainvoke({"query": "Compare TRx and explain why"})

    assert failing_composer[0]["user_id"] is None
    assert orchestrator.payloads[0]["user_id"] is None


# ------------------------------------------- gap 3: the model stops guessing sessions


def test_neither_tool_schema_offers_the_model_a_session_to_guess():
    """Inside chat the argument was already ignored; the schema still advertised it
    (example ``sess_abc123``), spending tokens on a value the model cannot know."""
    for schema in (chatbot_tools.OrchestratorToolInput, chatbot_tools.ToolComposerToolInput):
        assert "session_id" not in schema.model_fields, schema.__name__
        example = schema.model_config["json_schema_extra"]["example"]
        assert "session_id" not in example, schema.__name__


def test_neither_tool_accepts_a_session_argument():
    import inspect

    for fn in (chatbot_tools.orchestrator_tool, chatbot_tools.tool_composer_tool):
        params = inspect.signature(fn.coroutine).parameters
        assert "session_id" not in params, fn.name


async def test_a_session_the_model_invents_is_ignored_entirely(orchestrator):
    """It is dropped by the schema now, not merely outranked by the binding."""
    result = await chatbot_tools.orchestrator_tool.ainvoke(
        {"query": "Why is TRx moving?", "session_id": "sess_abc123"}
    )

    assert orchestrator.payloads[0]["session_id"] is None
    assert result["context"]["session_id"] is None


# ------------------- round 2: a session prefix is a claim, not a credential


OTHER_USER = "8f14e45f-ce0a-4c9b-9f2e-1d3a5b7c9e11"


def test_the_verified_user_wins_over_a_caller_supplied_session_prefix(
    no_authenticated_user, caplog
):
    """AG-UI takes ``threadId`` from the request body (``copilotkit.py:~4706``) and
    copies it into state, so a session prefix is attacker-controlled: user A could
    send ``{B}~anything`` and have B recorded as the owner of A's compositions.
    A uuid-shaped prefix proves syntax, never ownership."""
    import logging

    from src.api.routes.chat_identity import resolve_tool_user_id
    from src.utils.llm_attribution import set_authenticated_user

    set_authenticated_user(USER)
    with caplog.at_level(logging.WARNING, logger="src.utils.llm_attribution"):
        resolved = resolve_tool_user_id(f"{OTHER_USER}~0b7f7d6e-2c1a-4d7e-9a53-3f1f6a0c9e21")

    assert resolved == USER
    warned = [r.getMessage() for r in caplog.records]
    assert any(OTHER_USER in m and USER in m for m in warned), caplog.text


def test_a_matching_prefix_logs_nothing(no_authenticated_user, caplog):
    import logging

    from src.api.routes.chat_identity import resolve_tool_user_id
    from src.utils.llm_attribution import set_authenticated_user

    set_authenticated_user(USER)
    with caplog.at_level(logging.WARNING, logger="src.utils.llm_attribution"):
        assert resolve_tool_user_id(REAL_SESSION) == USER

    assert caplog.records == []


# ------------- round 2: the identity channel on the SDK sub-path fallthrough


def _copilotkit_request(path: str, body: bytes, user: Optional[dict]):
    """A real Starlette Request over a complete ASGI scope, as the 1432 gate tests build.

    ``user`` stands in for what ``JWTAuthMiddleware`` attaches at
    ``auth_middleware.py:~335`` after it has verified the token itself.
    """
    from starlette.requests import Request

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": f"/api/copilotkit/{path}",
        "raw_path": f"/api/copilotkit/{path}".encode(),
        "query_string": b"",
        "headers": [],
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
        "root_path": "",
        "path_params": {"path": path},
        "state": {},
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive)
    if user is not None:
        request.state.user = user
    return request


async def test_a_middleware_authenticated_sub_path_still_names_its_user(
    monkeypatch, no_authenticated_user
):
    """The SDK sub-path skips the gate when the middleware already authenticated.

    ``copilotkit.py:~4862`` runs ``_require_auth_for_copilotkit_execution`` only
    when ``request.state.user`` is None — correct as a gate, but that gate was the
    only thing populating the identity channel. On a bare-uuid thread the resolver
    then had nothing to read and the composition recorded a NULL owner, which
    #2095's owner gate treats as absent-pass.
    """
    from unittest.mock import MagicMock

    from fastapi.responses import JSONResponse

    import src.api.routes.copilotkit as ck
    from src.api.routes.chat_identity import resolve_tool_user_id

    monkeypatch.setattr(ck, "TESTING_MODE", False)

    async def _must_not_run(request):
        raise AssertionError("the gate re-ran although the middleware had authenticated")

    monkeypatch.setattr(ck, "_require_auth_for_copilotkit_execution", _must_not_run)

    seen: dict[str, Any] = {}

    async def _fake_sdk_handler(request, sdk):
        # What the tools would resolve at execution time, on a bare CopilotKit thread.
        seen["resolved"] = resolve_tool_user_id(BARE_THREAD)
        return JSONResponse(content={"reached": "execution"})

    monkeypatch.setattr(ck, "sdk_handler", _fake_sdk_handler)

    request = _copilotkit_request(
        "agent/default", b'{"messages":[]}', user={"id": USER, "email": "u@example.com"}
    )
    response = await ck.copilotkit_custom_handler(request, MagicMock(), path="agent/default")

    assert response.status_code == 200, response.status_code
    assert seen["resolved"] == USER
