"""#2105: the chat tools that reach the conversation and the orchestrator use the
identity the turn bound, not one the model guesses or none at all.

Item 1 — ``conversation_memory_tool`` took a REQUIRED ``session_id`` the model
had to supply. The model cannot know the id of the conversation it is in: the
tools node binds it per turn (``chat_session_binding``) into the channel
``_resolve_session_id`` reads, and this tool never consulted it. #2107 made a
foreign guess harmless (refused as "not found"); it still left the model
guessing. The argument is now optional and defaults to the bound session.

Item 2 — ``run_causal_analysis`` (a CopilotKit action) handed the orchestrator
a query and a user context and nothing about WHO asked: no ``user_id``, no
``session_id``, while every other orchestrator caller passes both. It now
passes the bound session and the identity resolved from it — or None, never a
minted value, when nothing is bound (a direct SDK ``actions/execute`` request).

Only the conversation store, the message store and the orchestrator are
doubled; the tool, its schema and the action are the real ones.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

import src.api.routes.chatbot_tools as tools
from src.api.routes.chatbot_tools import reset_chat_session_id, set_chat_session_id
from src.utils.llm_attribution import set_authenticated_user

CALLER = "8f14e45f-ce0a-4c9b-9f2e-1d3a5b7c9e11"
VICTIM = "46d40f52-39ac-4b79-b3a4-1f1292059a00"
CALLER_THREAD = "7a1f1e2d-0c3b-4a5e-8d9f-6b2c4e1a3d50"
VICTIM_THREAD = "cf6b364a-11f1-4b26-a5c0-f6559d06f659"


class _FakeConversations:
    def __init__(self, rows: Dict[str, str]) -> None:
        self._rows = rows
        self.lookups: list[str] = []

    async def get_by_session_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        self.lookups.append(session_id)
        owner = self._rows.get(session_id)
        return None if owner is None else {"session_id": session_id, "user_id": owner}


@pytest.fixture(autouse=True)
def clean_channels():
    set_authenticated_user(None)
    yield
    set_authenticated_user(None)


@pytest.fixture
def conversations(monkeypatch) -> _FakeConversations:
    """``conversation_memory_tool`` over a doubled conversation and message store."""
    fake = _FakeConversations({CALLER_THREAD: CALLER, VICTIM_THREAD: VICTIM})
    msg_repo = MagicMock()

    async def _recent(session_id: str, count: int = 10) -> list[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": f"a question asked in {session_id}",
                "created_at": "2026-09-01T00:00:00Z",
                "agent_name": None,
                "tool_calls": [],
                "tool_results": [],
            }
        ]

    msg_repo.get_recent_messages = _recent

    async def _client() -> object:
        return object()

    monkeypatch.setattr(tools, "get_async_supabase_client", _client)
    monkeypatch.setattr(tools, "get_chatbot_conversation_repository", lambda client: fake)
    monkeypatch.setattr(tools, "get_chatbot_message_repository", lambda client: msg_repo)
    return fake


@pytest.fixture
def bound_turn():
    """What the tools node binds for a turn in CALLER's own conversation."""
    set_authenticated_user(CALLER)
    token = set_chat_session_id(CALLER_THREAD)
    yield
    reset_chat_session_id(token)


# ----------------------------------------------- item 1: conversation_memory_tool


async def test_the_memory_tool_reads_the_bound_conversation_when_none_is_named(
    conversations, bound_turn
):
    result = await tools.conversation_memory_tool.ainvoke({})

    assert result["success"] is True, result
    assert result["session_id"] == CALLER_THREAD
    assert conversations.lookups == [CALLER_THREAD]
    assert [m["content"] for m in result["messages"]] == [f"a question asked in {CALLER_THREAD}"]


async def test_with_nothing_bound_and_nothing_named_the_tool_invents_no_session(
    conversations, caplog
):
    """Unbound and unnamed is the existing "not found" shape with a NULL session —
    and no lookup at all, so the store is never asked about an invented id.

    The model is told "not found" about a conversation that exists, so the
    operator must be able to see why: an empty session channel inside the graph
    is the #2100 regression class, and it must not pass at INFO."""
    with caplog.at_level(logging.WARNING, logger="src.api.routes.chat_identity"):
        result = await tools.conversation_memory_tool.ainvoke({})

    assert result == {"success": False, "error": "Conversation not found", "session_id": None}
    assert conversations.lookups == []
    unbound = [
        r
        for r in caplog.records
        if r.name == "src.api.routes.chat_identity"
        and r.levelno == logging.WARNING
        and "no conversation named and none bound" in r.getMessage()
    ]
    assert len(unbound) == 1, [r.getMessage() for r in caplog.records]


async def test_a_named_foreign_conversation_is_still_refused_inside_a_bound_turn(
    conversations, bound_turn
):
    """#2107's guard is unchanged: a model that still names another user's
    conversation gets "not found", with the foreign history nowhere in the answer."""
    result = await tools.conversation_memory_tool.ainvoke({"session_id": VICTIM_THREAD})

    assert result["success"] is False
    assert "messages" not in result
    assert VICTIM_THREAD not in json.dumps(result.get("messages", []))
    assert conversations.lookups == [VICTIM_THREAD]


def test_the_model_facing_schema_makes_the_session_optional_and_says_to_omit_it():
    schema = tools.conversation_memory_tool.args_schema.model_json_schema()

    assert "session_id" not in schema.get("required", []), schema.get("required")
    description = schema["properties"]["session_id"]["description"].lower()
    assert "omit" in description and "current conversation" in description, description
    assert tools.ConversationMemoryInput().session_id is None


# ------------------------------------------------------ item 2: run_causal_analysis


class _RecordingOrchestrator:
    def __init__(self) -> None:
        self.inputs: list[Dict[str, Any]] = []

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.inputs.append(payload)
        return {
            "response_text": "Real causal analysis: ATE=0.184",
            "ate": 0.184,
            "ci": [0.142, 0.226],
            "p_value": 0.012,
            "significant": True,
        }


@pytest.fixture
def orchestrator(monkeypatch) -> _RecordingOrchestrator:
    import src.api.routes.copilotkit as ck

    fake = _RecordingOrchestrator()
    monkeypatch.setattr(ck, "_get_orchestrator", lambda: fake)
    return fake


async def _run_action(session: Optional[str]) -> Dict[str, Any]:
    """The real action, with the session channel bound the way its two callers
    leave it: ``execute()`` binds the thread; an SDK ``actions/execute`` binds none."""
    import src.api.routes.copilotkit as ck

    token = ck._session_id_context.set(session)
    try:
        return await ck.run_causal_analysis(
            intervention="HCP Engagement", target_kpi="TRx Volume", brand="Kisqali"
        )
    finally:
        ck._session_id_context.reset(token)


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _uuid_shaped(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


async def test_the_action_passes_the_bound_session_and_the_verified_caller(orchestrator):
    """Inside a turn: ``execute()`` bound both channels, and the bare AG-UI
    thread carries no prefix, so the user can only come from the verified one."""
    set_authenticated_user(CALLER)

    result = await _run_action(CALLER_THREAD)

    assert result.get("data_source") == "orchestrator", result
    [payload] = orchestrator.inputs
    assert payload["session_id"] == CALLER_THREAD
    assert payload["user_id"] == CALLER


async def test_an_sdk_call_with_no_session_still_names_the_verified_caller(orchestrator):
    """A direct ``actions/execute`` request: the auth dependency bound the
    identity, nothing bound a session. The session stays None — never minted."""
    set_authenticated_user(CALLER)

    await _run_action(None)

    [payload] = orchestrator.inputs
    assert payload["session_id"] is None
    assert payload["user_id"] == CALLER


async def test_unbound_the_action_hands_the_orchestrator_nothing_it_did_not_have(orchestrator):
    await _run_action(None)

    [payload] = orchestrator.inputs
    assert payload["session_id"] is None
    assert payload["user_id"] is None
    minted = [s for s in _strings(payload) if _uuid_shaped(s)]
    assert not minted, f"the action invented an id: {minted}"
