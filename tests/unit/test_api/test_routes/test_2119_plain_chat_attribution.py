"""#2119: ``/chat`` and ``/chat/stream`` bind LLM attribution before the graph runs.

``set_chat_attribution`` had exactly one caller, inside the AG-UI
``LangGraphAgent.execute()``. The two plain routes bound the verified identity
(``authorize_chat_identity``) but never the attribution, so every LLM call a
plain-route turn made recorded ``surface='other'`` with a NULL user and session,
and its assistant row drained no token usage.

The binding has to happen in the REQUEST task, before the ``StreamingResponse``
is built or ``run_chatbot`` is awaited: the keepalive wrapper copies the request
context once and every frame pull shares that copy (#2100), so a write made
inside the generator would reach the graph but not the request task that owns
the turn. It also needs the turn's session id to exist already — which it did
not, because ``_stream_chat_response`` and ``create_initial_state`` each minted
it later. The route seam now finalises the ids first and binds on them.

The HTTP cases drive the REAL routes, the real ``TracingMiddleware`` (so the
``X-Request-ID`` header takes the path it takes in production) and the real
``with_sse_keepalive``. Their doubles are the ``require_viewer`` dependency
(overridden to return USER), the two graph entry points, the conversation store
the owner check reads and the Supabase client it is built on. The last test is
not an HTTP case: it awaits the resolver directly, as both routes do.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routes.chatbot_graph as graph
import src.api.routes.copilotkit as ck
from src.api.dependencies.auth import require_viewer
from src.api.middleware.tracing import TracingMiddleware
from src.api.routes import chat_identity
from src.utils.llm_attribution import (
    LLMAttribution,
    clear_attribution,
    get_attribution,
    set_authenticated_user,
)

USER = "8f14e45f-ce0a-4c9b-9f2e-1d3a5b7c9e11"
HEADER_REQUEST_ID = "req-2119-from-header"
BODY_REQUEST_ID = "req-2119-from-body"
FIXED_SESSION = f"{USER}~fixed-2119"


@pytest.fixture(autouse=True)
def clean_channels():
    """Neither channel may leak between tests, nor into them from a sibling."""
    clear_attribution()
    set_authenticated_user(None)
    yield
    clear_attribution()
    set_authenticated_user(None)


@pytest.fixture
def no_stored_conversation(monkeypatch):
    """The owner check's store: every session is new (rule 2 of #2107)."""

    class _Repo:
        async def get_by_session_id(self, session_id: str) -> Optional[Dict[str, Any]]:
            return None

    async def _client() -> object:
        return object()

    monkeypatch.setattr(chat_identity, "get_async_supabase_client", _client)
    monkeypatch.setattr(
        chat_identity, "ChatbotConversationRepository", lambda supabase_client=None: _Repo()
    )


@pytest.fixture(params=[False, True], ids=["production", "testing_mode"])
def chat_client(request, monkeypatch, no_stored_conversation) -> TestClient:
    """The real router behind the real tracing middleware, authenticated as USER.

    Parametrised over ``TESTING_MODE`` because the binding is not a gate: it must
    happen whether or not the claim checks are exempted.
    """
    monkeypatch.setattr(ck, "TESTING_MODE", request.param)
    app = FastAPI()
    app.add_middleware(TracingMiddleware)
    app.include_router(ck.router)
    app.dependency_overrides[require_viewer] = lambda: {"id": USER}
    return TestClient(app)


@pytest.fixture
def stream_seen(monkeypatch) -> Dict[str, Any]:
    """A ``stream_chatbot`` double that records the attribution where the graph
    nodes would read it: on its first pull, on a later pull, and inside a task
    spawned from a pull (a node that fans out work)."""
    seen: Dict[str, Any] = {}

    async def _nested() -> Optional[LLMAttribution]:
        return get_attribution()

    async def _stream_chatbot(**kwargs: Any):
        seen["kwargs"] = kwargs
        seen["first_pull"] = get_attribution()
        seen["nested_task"] = await asyncio.create_task(_nested())
        yield {"generate": {"response_text": "The TRx is up."}}
        seen["later_pull"] = get_attribution()

    monkeypatch.setattr(graph, "stream_chatbot", _stream_chatbot)
    return seen


@pytest.fixture
def run_seen(monkeypatch) -> Dict[str, Any]:
    """A ``run_chatbot`` double that records the attribution and echoes the
    session it was given, the way the real graph returns the state's session."""
    seen: Dict[str, Any] = {}

    async def _run_chatbot(**kwargs: Any) -> Dict[str, Any]:
        seen["kwargs"] = kwargs
        seen["attribution"] = get_attribution()
        return {"response_text": "The TRx is up.", "session_id": kwargs.get("session_id")}

    monkeypatch.setattr(graph, "run_chatbot", _run_chatbot)
    return seen


def _body(**overrides: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {"query": "What is TRx?", "user_id": USER}
    body.update(overrides)
    return body


def _sse_frames(text: str) -> list[Dict[str, Any]]:
    frames = []
    for chunk in text.split("\n\n"):
        if chunk.startswith("data: "):
            frames.append(json.loads(chunk[len("data: ") :]))
    return frames


def _minted_for(user: str, session_id: Optional[str]) -> bool:
    """``{user}~{uuid4}``: the one shape a plain-route session may be minted in."""
    if not session_id or "~" not in session_id:
        return False
    prefix, suffix = session_id.split("~", 1)
    try:
        return prefix == user and uuid.UUID(suffix).version == 4
    except ValueError:
        return False


def _assert_chat_attribution(
    attribution: Optional[LLMAttribution], *, session_id: str, request_id: str
) -> None:
    assert attribution is not None, "no attribution was bound for the turn"
    assert attribution.surface == "chat"
    assert attribution.session_id == session_id
    assert attribution.user_id == USER
    assert attribution.request_id == request_id


# ------------------------------------------------------------- T1: /chat/stream


def test_stream_chat_binds_attribution_every_frame_pull_can_read(chat_client, stream_seen):
    response = chat_client.post(
        "/copilotkit/chat/stream",
        json=_body(session_id=""),
        headers={"X-Request-ID": HEADER_REQUEST_ID},
    )
    frames = _sse_frames(response.text)

    assert response.status_code == 200
    session_frame = next(f for f in frames if f["type"] == "session_id")
    session_id = session_frame["data"]
    assert _minted_for(USER, session_id), session_id
    assert stream_seen["kwargs"]["session_id"] == session_id
    for where in ("first_pull", "nested_task", "later_pull"):
        _assert_chat_attribution(
            stream_seen[where], session_id=session_id, request_id=HEADER_REQUEST_ID
        )


# -------------------------------------------------------------------- T2: /chat


def test_chat_binds_attribution_and_mints_the_session_before_the_graph(chat_client, run_seen):
    response = chat_client.post(
        "/copilotkit/chat",
        json=_body(session_id=""),
        headers={"X-Request-ID": HEADER_REQUEST_ID},
    )

    assert response.status_code == 200
    received = run_seen["kwargs"]["session_id"]
    assert _minted_for(USER, received), received
    assert response.json()["session_id"] == received
    _assert_chat_attribution(
        run_seen["attribution"], session_id=received, request_id=HEADER_REQUEST_ID
    )


# ---------------------------------------------------------- T3: body precedence


def test_a_supplied_session_and_request_id_are_bound_as_given(chat_client, stream_seen):
    """Nothing is minted over a session the body names, and the body's
    request_id outranks the header's, exactly as the routes' own
    ``effective_request_id`` expression has always ranked them."""
    response = chat_client.post(
        "/copilotkit/chat/stream",
        json=_body(session_id=FIXED_SESSION, request_id=BODY_REQUEST_ID),
        headers={"X-Request-ID": HEADER_REQUEST_ID},
    )
    frames = _sse_frames(response.text)

    assert response.status_code == 200
    assert next(f for f in frames if f["type"] == "session_id")["data"] == FIXED_SESSION
    assert stream_seen["kwargs"]["session_id"] == FIXED_SESSION
    _assert_chat_attribution(
        stream_seen["first_pull"], session_id=FIXED_SESSION, request_id=BODY_REQUEST_ID
    )


def test_chat_keeps_a_supplied_session_and_request_id(chat_client, run_seen):
    response = chat_client.post(
        "/copilotkit/chat",
        json=_body(session_id=FIXED_SESSION, request_id=BODY_REQUEST_ID),
        headers={"X-Request-ID": HEADER_REQUEST_ID},
    )

    assert response.status_code == 200
    assert run_seen["kwargs"]["session_id"] == FIXED_SESSION
    assert response.json()["session_id"] == FIXED_SESSION
    _assert_chat_attribution(
        run_seen["attribution"], session_id=FIXED_SESSION, request_id=BODY_REQUEST_ID
    )


# ------------------------------------- the seam itself: bound in the caller's task


async def test_the_resolver_binds_attribution_in_the_task_that_calls_it(
    monkeypatch, no_stored_conversation
):
    """Both routes ``await _resolve_chat_identity`` in the request task, outside
    any generator. A binding made there is what the keepalive copies; one made
    anywhere later would be invisible to this caller."""
    monkeypatch.setattr(ck, "TESTING_MODE", False)
    chat_request = ck.ChatRequest(query="What is TRx?", user_id=USER)

    identity = await ck._resolve_chat_identity({"id": USER}, chat_request)

    assert identity == USER
    assert _minted_for(USER, chat_request.session_id), chat_request.session_id
    assert chat_request.request_id == "unknown", "no header, no body: the routes' own fallback"
    assert chat_request.session_id is not None
    _assert_chat_attribution(
        get_attribution(), session_id=chat_request.session_id, request_id="unknown"
    )
