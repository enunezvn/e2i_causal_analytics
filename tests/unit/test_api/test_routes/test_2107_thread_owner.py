"""#2107: an EXISTING chat thread is refused to anyone but its stored owner.

#2077 closed the *prefixed* claim (``{victim}~{uuid}`` is 403 on every entry
point). It did not check the stored owner of an EXISTING **bare** uuid thread,
and CopilotKit mints exactly those: token X could send Y's thread id and the
turn would run in Y's conversation — reading Y's last ten messages into the
prompt (``chatbot_graph.py:~1006``) and writing messages that migration 123's
trigger stamps with Y's ``user_id``.

The gate is one helper, ``chat_identity.thread_owner_denied``. It never raises:
the AG-UI call site sits inside the handler's broad ``except``, which falls
through to the *ungated* SDK path, so an exception there would defeat the check
it is meant to enforce. The helper returns a bool and each caller answers 403.

Policy encoded here (design §5, owner-confirmed defaults):

* a lookup that fails is **fail-OPEN** with a WARNING — the write and the read
  that constitute the harm go through the same database, so an outage that
  blinds the check equally disables what it protects;
* the anonymous sentinel owner is **unowned** and passes, at INFO — 375 live
  bare conversations predate #1405's JWT attribution and nobody's identity is
  recorded in them;
* a thread with **no row** passes — that is #1405's arbitrary-thread support and
  it is what keeps every NEW bare thread working.

Only the Supabase client and the conversation repository are doubled; the route
handlers, the claim helpers and the identity module are the real ones.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import src.api.routes.copilotkit as ck
from src.api.routes import chat_identity

CALLER = "8f14e45f-ce0a-4c9b-9f2e-1d3a5b7c9e11"
VICTIM = "46d40f52-39ac-4b79-b3a4-1f1292059a00"
ANON = "00000000-0000-0000-0000-000000000000"

# The shape CopilotKit mints: a bare v4 uuid with no owner prefix.
VICTIM_THREAD = "cf6b364a-11f1-4b26-a5c0-f6559d06f659"
CALLER_THREAD = "7a1f1e2d-0c3b-4a5e-8d9f-6b2c4e1a3d50"
NEW_THREAD = "1d2c3b4a-5e6f-4a7b-8c9d-0e1f2a3b4c5d"


class _FakeConversations:
    """The rows ``chatbot_conversations`` holds, keyed by its primary key.

    ``session_id`` IS the primary key and ``user_id`` is ``uuid NOT NULL``
    (verified against the live schema), so a row either exists with an owner or
    does not exist at all.
    """

    def __init__(self, rows: Dict[str, str], *, raises: bool = False) -> None:
        self._rows = rows
        self._raises = raises
        self.lookups: list[str] = []

    async def get_by_session_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        self.lookups.append(session_id)
        if self._raises:
            raise RuntimeError("supabase is unreachable")
        owner = self._rows.get(session_id)
        return None if owner is None else {"session_id": session_id, "user_id": owner}


@pytest.fixture
def conversations(monkeypatch) -> _FakeConversations:
    """Wire a doubled repository into the helper's two module seams."""
    fake = _FakeConversations(
        {VICTIM_THREAD: VICTIM, CALLER_THREAD: CALLER, "anon-owned-thread": ANON}
    )

    async def _client() -> object:
        return object()

    monkeypatch.setattr(chat_identity, "get_async_supabase_client", _client)
    monkeypatch.setattr(
        chat_identity, "ChatbotConversationRepository", lambda supabase_client=None: fake
    )
    return fake


@pytest.fixture
def dead_lookup(monkeypatch) -> _FakeConversations:
    fake = _FakeConversations({}, raises=True)

    async def _client() -> object:
        return object()

    monkeypatch.setattr(chat_identity, "get_async_supabase_client", _client)
    monkeypatch.setattr(
        chat_identity, "ChatbotConversationRepository", lambda supabase_client=None: fake
    )
    return fake


# ------------------------------------------------------------------ T1: the helper


async def test_an_existing_thread_owned_by_someone_else_is_denied(conversations, caplog):
    with caplog.at_level(logging.WARNING, logger="src.api.routes.chat_identity"):
        denied = await chat_identity.thread_owner_denied(VICTIM_THREAD, CALLER)

    assert denied is True
    # A denial is worth seeing, but neither id may be logged.
    assert caplog.records, "a refused turn logged nothing"
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert VICTIM not in logged and CALLER not in logged and VICTIM_THREAD not in logged


async def test_the_callers_own_existing_thread_is_allowed(conversations):
    assert await chat_identity.thread_owner_denied(CALLER_THREAD, CALLER) is False


async def test_a_thread_with_no_row_is_allowed(conversations):
    """#1405's arbitrary-thread support: a NEW bare uuid must still work."""
    assert await chat_identity.thread_owner_denied(NEW_THREAD, CALLER) is False


async def test_an_anonymous_owned_thread_is_allowed(conversations):
    """The sentinel records the ABSENCE of an identity, so it owns nothing (§5 Q2)."""
    assert await chat_identity.thread_owner_denied("anon-owned-thread", CALLER) is False


async def test_without_a_verified_caller_there_is_nothing_to_compare(conversations):
    assert await chat_identity.thread_owner_denied(VICTIM_THREAD, None) is False
    assert await chat_identity.thread_owner_denied(None, CALLER) is False
    # Neither case may spend a database round trip.
    assert conversations.lookups == []


async def test_a_failing_lookup_fails_open_and_warns(dead_lookup, caplog):
    """§5 Q1: the harm and the check share a database — fail-closed would 403
    every chat user during a Supabase hiccup without protecting anything."""
    with caplog.at_level(logging.WARNING, logger="src.api.routes.chat_identity"):
        denied = await chat_identity.thread_owner_denied(VICTIM_THREAD, CALLER)

    assert denied is False
    assert caplog.records, "a blinded owner check passed silently"


async def test_the_helper_never_raises_even_when_the_client_is_gone(monkeypatch):
    """The AG-UI call site is inside a broad ``except`` that falls through to the
    UNGATED SDK path, so a raise here would be worse than no check at all."""

    async def _boom() -> object:
        raise RuntimeError("no client")

    monkeypatch.setattr(chat_identity, "get_async_supabase_client", _boom)

    assert await chat_identity.thread_owner_denied(VICTIM_THREAD, CALLER) is False


# ------------------------------------------------------------------ helpers for T2-T5


def _scope(path: str, method: str = "POST") -> Dict[str, Any]:
    return {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": f"/api/copilotkit/{path}",
        "raw_path": f"/api/copilotkit/{path}".encode(),
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer tok")],
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
        "root_path": "",
        "path_params": {"path": path},
        "state": {},
    }


def _request(path: str, body: bytes, user: Optional[dict], method: str = "POST") -> Request:
    async def receive() -> Dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(_scope(path, method), receive)
    if user is not None:
        request.state.user = user
    return request


def _sdk() -> MagicMock:
    agent = MagicMock()
    agent.name = "default"
    sdk = MagicMock()
    sdk.agents = [agent]
    return sdk


@pytest.fixture
def agui(monkeypatch):
    """The real root handler, with the auth gate resolving to CALLER."""
    monkeypatch.setattr(ck, "TESTING_MODE", False)

    async def _verify(token: str) -> Dict[str, Any]:
        return {"id": CALLER, "email": "caller@example.com"}

    monkeypatch.setattr(ck, "verify_supabase_token", _verify)

    async def _run(thread_id: Optional[str]):
        body: Dict[str, Any] = {"method": "agent/run", "messages": [{"role": "user"}]}
        if thread_id is not None:
            body["threadId"] = thread_id
        return await ck.copilotkit_custom_handler(
            _request("", json.dumps(body).encode(), None), _sdk(), path=""
        )

    return _run


@pytest.fixture
def sdk_path(monkeypatch):
    """The real fallthrough branch, with the SDK handler doubled so we can see
    whether execution was reached at all."""
    monkeypatch.setattr(ck, "TESTING_MODE", False)
    reached: Dict[str, bool] = {"sdk": False}

    from fastapi.responses import JSONResponse

    async def _fake_sdk_handler(request: Request, sdk: Any) -> JSONResponse:
        reached["sdk"] = True
        return JSONResponse(content={"reached": "execution"})

    monkeypatch.setattr(ck, "sdk_handler", _fake_sdk_handler)

    async def _run(path: str, body: Dict[str, Any], method: str = "POST"):
        request = _request(path, json.dumps(body).encode(), {"id": CALLER}, method)
        response = await ck.copilotkit_custom_handler(request, MagicMock(), path=path)
        return response, reached["sdk"]

    return _run


class _Claims:
    """``ChatRequest``-shaped: the two owner claims a chat body can carry."""

    def __init__(self, session_id: Optional[str], user_id: Optional[str] = None) -> None:
        self.session_id = session_id
        self.user_id = user_id


# ------------------------------------------------------------------ T2: AG-UI agent/run


async def test_agui_refuses_an_existing_thread_owned_by_someone_else(agui, conversations):
    response = await agui(VICTIM_THREAD)

    assert response.status_code == 403
    assert conversations.lookups == [VICTIM_THREAD]


async def test_agui_accepts_the_callers_own_existing_thread(agui, conversations):
    from starlette.responses import StreamingResponse

    response = await agui(CALLER_THREAD)

    assert isinstance(response, StreamingResponse)


async def test_agui_accepts_a_brand_new_bare_thread(agui, conversations):
    from starlette.responses import StreamingResponse

    response = await agui(NEW_THREAD)

    assert isinstance(response, StreamingResponse)


# ------------------------------------------------------------------ T3: the SDK sub-paths


@pytest.mark.parametrize("path", ["agent/default", "agents/execute", "agent/default/state"])
async def test_every_sdk_sub_path_refuses_a_foreign_existing_thread(path, sdk_path, conversations):
    response, sdk_reached = await sdk_path(path, {"threadId": VICTIM_THREAD, "messages": []})

    assert response.status_code == 403
    assert sdk_reached is False, "the SDK executed on someone else's conversation"


@pytest.mark.parametrize("path", ["agent/default", "agents/execute", "agent/default/state"])
async def test_every_sdk_sub_path_accepts_the_callers_own_thread(path, sdk_path, conversations):
    response, sdk_reached = await sdk_path(path, {"threadId": CALLER_THREAD, "messages": []})

    assert response.status_code == 200
    assert sdk_reached is True


async def test_an_sdk_preflight_still_skips_the_owner_check(sdk_path, conversations):
    """OPTIONS carries no turn; the check must not cost it a database round trip."""
    response, _ = await sdk_path("agent/default", {"threadId": VICTIM_THREAD}, method="OPTIONS")

    assert response.status_code == 200
    assert conversations.lookups == []


# ------------------------------------------------------------------ T4: /chat and /chat/stream


async def test_chat_identity_refuses_a_foreign_existing_session(conversations):
    with pytest.raises(HTTPException) as exc:
        await chat_identity.authorize_chat_identity(CALLER, _Claims(VICTIM_THREAD), False)

    assert exc.value.status_code == 403


async def test_chat_identity_accepts_the_callers_own_session(conversations):
    resolved = await chat_identity.authorize_chat_identity(CALLER, _Claims(CALLER_THREAD), False)

    assert resolved == CALLER


async def test_chat_identity_accepts_a_new_session(conversations):
    assert await chat_identity.authorize_chat_identity(CALLER, _Claims(NEW_THREAD), False) == CALLER


async def test_stream_chat_answers_403_before_the_generator_starts(monkeypatch, conversations):
    """A 403 raised inside the SSE body would reach the browser as HTTP 200 with
    an error frame; the resolution is deliberately ahead of the response."""
    monkeypatch.setattr(ck, "TESTING_MODE", False)

    def _must_not_run(*args: Any, **kwargs: Any):
        raise AssertionError("the stream body started on a foreign conversation")

    monkeypatch.setattr(ck, "_stream_chat_response", _must_not_run)

    request_model = ck.ChatRequest(query="what is TRx", user_id=CALLER, session_id=VICTIM_THREAD)
    with pytest.raises(HTTPException) as exc:
        await ck.stream_chat(
            request_model, _request("chat/stream", b"", {"id": CALLER}), {"id": CALLER}
        )

    assert exc.value.status_code == 403


# ------------------------------------------------------------------ T5: the regression net


async def test_testing_mode_is_still_exempt_on_every_seam(monkeypatch, conversations):
    """TESTING_MODE bypasses real auth, as every other claim check allows."""
    monkeypatch.setattr(ck, "TESTING_MODE", True)

    body = {"method": "agent/run", "threadId": VICTIM_THREAD, "messages": []}
    assert (
        await chat_identity.owned_thread_id(body, _request("", b"", {"id": CALLER}), True)
        == VICTIM_THREAD
    )
    assert (
        await chat_identity.sdk_thread_denied(
            json.dumps(body).encode(), _request("", b"", {"id": CALLER}), True, "POST"
        )
        is False
    )
    assert (
        await chat_identity.authorize_chat_identity(CALLER, _Claims(VICTIM_THREAD), True) == CALLER
    )
    assert conversations.lookups == []


async def test_the_prefixed_403_of_2077_is_unchanged(agui, conversations):
    """A foreign PREFIX is still refused on syntax alone, with no lookup."""
    response = await agui(f"{VICTIM}~0b7f7d6e-2c1a-4d7e-9a53-3f1f6a0c9e21")

    assert response.status_code == 403
    assert conversations.lookups == []


async def test_an_unparseable_sdk_body_still_names_no_thread(sdk_path, conversations):
    """Left to the SDK to reject, exactly as before — not 403ed here."""
    monkeypatch_free_body = b"{not json"

    request = _request("agent/default", monkeypatch_free_body, {"id": CALLER})
    response = await ck.copilotkit_custom_handler(request, MagicMock(), path="agent/default")

    assert response.status_code == 200
    assert conversations.lookups == []


# ------------------------------------------------ T5: every ingress still accepts real traffic


@pytest.mark.parametrize("path", ["agent/default", "agents/execute", "agent/default/state"])
async def test_every_sdk_sub_path_accepts_a_brand_new_bare_thread(path, sdk_path, conversations):
    """The shape every real CopilotKit turn sends on its first message."""
    response, sdk_reached = await sdk_path(path, {"threadId": NEW_THREAD, "messages": []})

    assert response.status_code == 200
    assert sdk_reached is True


async def test_the_chat_routes_resolver_refuses_a_foreign_existing_session(
    monkeypatch, conversations
):
    """``_resolve_chat_identity`` is the single seam behind both /chat and
    /chat/stream (``copilotkit.py:~5434,~5510``), so it is where the await
    wiring has to hold."""
    monkeypatch.setattr(ck, "TESTING_MODE", False)

    with pytest.raises(HTTPException) as exc:
        await ck._resolve_chat_identity(
            {"id": CALLER}, ck.ChatRequest(query="q", user_id=CALLER, session_id=VICTIM_THREAD)
        )

    assert exc.value.status_code == 403


@pytest.mark.parametrize("session_id", [CALLER_THREAD, NEW_THREAD, None])
async def test_the_chat_routes_resolver_accepts_real_traffic(
    session_id, monkeypatch, conversations
):
    """Own existing thread, a brand-new one, and no session at all."""
    monkeypatch.setattr(ck, "TESTING_MODE", False)

    resolved = await ck._resolve_chat_identity(
        {"id": CALLER}, ck.ChatRequest(query="q", user_id=CALLER, session_id=session_id)
    )

    assert resolved == CALLER


async def test_a_prefixed_session_the_caller_owns_is_still_accepted(monkeypatch, conversations):
    """/chat/stream mints ``{owner}~{uuid}`` itself; #2107 must not refuse its own ids."""
    monkeypatch.setattr(ck, "TESTING_MODE", False)
    own_prefixed = f"{CALLER}~0b7f7d6e-2c1a-4d7e-9a53-3f1f6a0c9e21"

    resolved = await ck._resolve_chat_identity(
        {"id": CALLER}, ck.ChatRequest(query="q", user_id=CALLER, session_id=own_prefixed)
    )

    assert resolved == CALLER
    assert conversations.lookups == [own_prefixed]


async def test_the_owner_gate_binds_the_identity_channel_only_after_it_passes(
    monkeypatch, conversations
):
    """A refused turn must not leave the victim's conversation bound as the
    caller's attribution channel for whatever runs next in this task."""
    from src.utils.llm_attribution import get_authenticated_user_id, set_authenticated_user

    monkeypatch.setattr(ck, "TESTING_MODE", False)
    set_authenticated_user(None)
    try:
        with pytest.raises(HTTPException):
            await chat_identity.authorize_chat_identity(CALLER, _Claims(VICTIM_THREAD), False)
        assert get_authenticated_user_id() is None
    finally:
        set_authenticated_user(None)


# ------------------- round 2 HIGH: the gate also covers reads from INSIDE a turn


@pytest.fixture
def memory_tool(monkeypatch, conversations):
    """``conversation_memory_tool`` over the doubled conversation store.

    The tool accepts a session id the MODEL may supply (#2105 made it optional,
    defaulting to the bound session), so being inside a conversation the caller
    owns does not make a NAMED target conversation theirs. The owner check is
    what makes the argument safe.
    """
    import src.api.routes.chatbot_tools as tools

    messages = [
        {
            "role": "user",
            "content": "the victim's private question",
            "created_at": "2026-09-01T00:00:00Z",
            "agent_name": None,
            "tool_calls": [],
            "tool_results": [],
        }
    ]
    msg_repo = MagicMock()

    async def _recent(session_id, count=10):
        return messages

    msg_repo.get_recent_messages = _recent

    async def _client() -> object:
        return object()

    monkeypatch.setattr(tools, "get_async_supabase_client", _client)
    monkeypatch.setattr(tools, "get_chatbot_conversation_repository", lambda client: conversations)
    monkeypatch.setattr(tools, "get_chatbot_message_repository", lambda client: msg_repo)
    return tools.conversation_memory_tool


@pytest.fixture
def bound_caller():
    from src.utils.llm_attribution import set_authenticated_user

    set_authenticated_user(CALLER)
    yield CALLER
    set_authenticated_user(None)


async def test_the_memory_tool_will_not_read_another_users_conversation(
    memory_tool, bound_caller, conversations, caplog
):
    """The every-read guarantee has to hold INSIDE an allowed turn too: a caller
    can steer the model into naming Y's session, and the tool used to load it."""
    with caplog.at_level(logging.WARNING, logger="src.api.routes.chat_identity"):
        result = await memory_tool.ainvoke({"session_id": VICTIM_THREAD})

    assert result["success"] is False
    assert "messages" not in result
    assert "the victim's private question" not in json.dumps(result)
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert logged, "a refused history read logged nothing"
    assert VICTIM not in logged and CALLER not in logged and VICTIM_THREAD not in logged


async def test_the_memory_tool_still_reads_the_callers_own_conversation(memory_tool, bound_caller):
    result = await memory_tool.ainvoke({"session_id": CALLER_THREAD})

    assert result["success"] is True
    assert result["message_count"] == 1


async def test_an_unknown_session_keeps_todays_answer(memory_tool, bound_caller):
    """Deny and not-found are the same shape, so the tool is no existence oracle."""
    result = await memory_tool.ainvoke({"session_id": NEW_THREAD})

    assert result["success"] is False
    assert "Conversation not found" in result["error"]


# ------------- round 2 MEDIUM: the plain routes, through the real handlers


@pytest.fixture
def chat_client(monkeypatch, conversations):
    """A TestClient over the real router, authenticated as CALLER.

    The resolver-level tests above cannot see the ``await`` at
    ``copilotkit.py:~5434`` and ``:~5510``: drop either one and
    ``_resolve_chat_identity`` returns an un-awaited coroutine, the gate never
    runs, and the turn proceeds. Only the handler answers that.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.dependencies.auth import require_viewer

    monkeypatch.setattr(ck, "TESTING_MODE", False)

    app = FastAPI()
    app.include_router(ck.router)
    app.dependency_overrides[require_viewer] = lambda: {"id": CALLER}
    return TestClient(app)


@pytest.fixture
def graph_calls(monkeypatch):
    """Doubles for the two graph entry points, recording whether they ran."""
    import src.api.routes.chatbot_graph as graph

    calls: Dict[str, int] = {"run": 0, "stream": 0}

    async def _run_chatbot(**kwargs: Any) -> Dict[str, Any]:
        calls["run"] += 1
        return {"response_text": "ok", "session_id": kwargs.get("session_id", "")}

    async def _stream_chatbot(**kwargs: Any):
        calls["stream"] += 1
        return
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(graph, "run_chatbot", _run_chatbot)
    monkeypatch.setattr(graph, "stream_chatbot", _stream_chatbot)
    return calls


def _body(session_id: Optional[str]) -> Dict[str, Any]:
    return {"query": "What is TRx?", "user_id": CALLER, "session_id": session_id}


async def test_chat_handler_refuses_a_foreign_existing_session(chat_client, graph_calls):
    response = chat_client.post("/copilotkit/chat", json=_body(VICTIM_THREAD))

    assert response.status_code == 403
    assert graph_calls["run"] == 0, "the graph ran on someone else's conversation"


async def test_chat_stream_handler_refuses_a_foreign_existing_session(chat_client, graph_calls):
    """403 as the HTTP status, not an error frame inside a 200 SSE body."""
    response = chat_client.post("/copilotkit/chat/stream", json=_body(VICTIM_THREAD))

    assert response.status_code == 403
    assert "text/event-stream" not in response.headers.get("content-type", "")
    assert graph_calls["stream"] == 0, "the stream body started on a foreign conversation"


@pytest.mark.parametrize("session_id", [CALLER_THREAD, NEW_THREAD])
async def test_chat_handler_accepts_real_traffic(session_id, chat_client, graph_calls):
    response = chat_client.post("/copilotkit/chat", json=_body(session_id))

    assert response.status_code == 200
    assert graph_calls["run"] == 1


@pytest.mark.parametrize("session_id", [CALLER_THREAD, NEW_THREAD])
async def test_chat_stream_handler_accepts_real_traffic(session_id, chat_client, graph_calls):
    response = chat_client.post("/copilotkit/chat/stream", json=_body(session_id))

    assert response.status_code == 200
    assert graph_calls["stream"] == 1


# --------------- round 2 LOW + minors: diagnosability and one lookup per id


async def test_a_missing_client_warns_instead_of_passing_silently(monkeypatch, caplog):
    """Fail-open is policy, but it has to be visible.

    With no client the repository returns None, which is the same answer as
    "no such conversation" — so the gate looked like it had run and allowed,
    when in fact it never ran at all. Those two must not read alike in a log.
    """

    async def _no_client() -> None:
        return None

    monkeypatch.setattr(chat_identity, "get_async_supabase_client", _no_client)

    with caplog.at_level(logging.WARNING, logger="src.api.routes.chat_identity"):
        denied = await chat_identity.thread_owner_denied(VICTIM_THREAD, CALLER)

    assert denied is False
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert logged, "an owner check that never ran passed silently"
    assert VICTIM_THREAD not in logged and CALLER not in logged


async def test_a_failing_lookup_warns_with_the_traceback(dead_lookup, caplog):
    """A fail-open that names no cause cannot be diagnosed after the fact."""
    with caplog.at_level(logging.WARNING, logger="src.api.routes.chat_identity"):
        await chat_identity.thread_owner_denied(VICTIM_THREAD, CALLER)

    assert any(r.exc_info for r in caplog.records), "the fail-open carried no traceback"


async def test_a_body_naming_one_thread_twice_costs_one_lookup(conversations):
    """AG-UI sends threadId nested AND top level; the same id is one claim."""
    body = {"method": "agent/run", "threadId": CALLER_THREAD, "body": {"threadId": CALLER_THREAD}}

    resolved = await chat_identity.owned_thread_id(body, _request("", b"", {"id": CALLER}), False)

    assert resolved == CALLER_THREAD
    assert conversations.lookups == [CALLER_THREAD]


# ------ round 3: an unbound caller allows the read, but never silently


@pytest.fixture
def unbound_caller():
    """No verified identity in the channel the tool guard reads."""
    from src.utils.llm_attribution import set_authenticated_user

    set_authenticated_user(None)
    yield
    set_authenticated_user(None)


async def test_an_unbound_caller_allows_the_read_but_says_so(
    memory_tool, unbound_caller, conversations, caplog
):
    """An empty attribution channel inside the AG-UI graph is the regression
    class #2100 is repairing, and here it silently disables the owner check.

    The allow stays — with no verified caller there is nothing to compare, and
    fail-open is the lane's policy — but the route seams WARN on every one of
    their fail-open paths and this one did not, so an operator could not tell
    an owner check that passed from one that never ran.
    """
    with caplog.at_level(logging.WARNING, logger="src.api.routes.chat_identity"):
        result = await memory_tool.ainvoke({"session_id": VICTIM_THREAD})

    # The allow is real: the foreign history actually came back, not merely a
    # success flag. Anything less would pass even if the read had been refused.
    assert result["success"] is True, "policy changed: the unbound read must still be allowed"
    assert [m["content"] for m in result["messages"]] == ["the victim's private question"]

    # THIS warning, not merely some captured text: an unrelated WARNING from any
    # logger would otherwise satisfy the assertion and hide a silent fail-open.
    unbound = [
        record
        for record in caplog.records
        if record.name == "src.api.routes.chat_identity"
        and record.levelno == logging.WARNING
        and "No verified caller bound" in record.getMessage()
        and "owner check did not run" in record.getMessage()
    ]
    assert len(unbound) == 1, [(r.name, r.levelname, r.getMessage()) for r in caplog.records]
    assert VICTIM not in unbound[0].getMessage()
    assert VICTIM_THREAD not in unbound[0].getMessage()
