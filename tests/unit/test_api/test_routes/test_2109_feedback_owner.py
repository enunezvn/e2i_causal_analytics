"""#2109: ``submit_feedback`` refuses to rate a message in someone else's thread.

The route (``POST /api/copilotkit/feedback``) resolves the rated message two
ways: (a) by an explicit ``message_id``, taking the session FROM THE ROW, or
(b) by a caller-supplied ``session_id`` plus a ``message_uuid`` /
``response_preview`` / ``response_text``. Both converge on one write of a
``chatbot_message_feedback`` row keyed by that session. Nothing between the
token and the write compared the conversation's STORED owner with the verified
caller, so token X could attach a rating to Y's conversation, and path (b) read
Y's messages to resolve it. The api writes with the service-role client, so RLS
never saw either.

The gate is ``chat_identity.refuse_foreign_thread`` at the convergence point,
built on the same ``thread_owner_denied`` every other chat ingress uses (#2107),
and it answers with the same 403 detail the AG-UI gate answers with. The route
body sits inside a broad ``except Exception`` that turns anything raised into
a 200 ``{"success": false, "error": ...}`` body — the swallow trap — so the
route re-raises ``HTTPException`` ahead of it. T6 is the teeth of that clause.

Doubles are the ``require_viewer`` dependency, the sync Supabase client the
lookup is built on, the async client and feedback repository the write goes
through, and the conversation store the owner check reads. The route, the
identity helpers and the resolution logic are the real ones.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pytest
import supabase
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import src.api.routes.copilotkit as ck
import src.memory.services.factories as factories
import src.repositories as repositories
from src.api.dependencies.auth import require_viewer
from src.api.routes import chat_identity

CALLER = "8f14e45f-ce0a-4c9b-9f2e-1d3a5b7c9e11"
VICTIM = "46d40f52-39ac-4b79-b3a4-1f1292059a00"

# The shape CopilotKit mints: a bare v4 uuid with no owner prefix.
VICTIM_THREAD = "cf6b364a-11f1-4b26-a5c0-f6559d06f659"
CALLER_THREAD = "7a1f1e2d-0c3b-4a5e-8d9f-6b2c4e1a3d50"
NEW_THREAD = "1d2c3b4a-5e6f-4a7b-8c9d-0e1f2a3b4c5d"

VICTIM_MESSAGE_ID = 123
CALLER_MESSAGE_ID = 456
NEW_MESSAGE_ID = 789

# The detail the AG-UI gate answers with (copilotkit.py, ``"threadId not yours"``).
SHARED_DETAIL = "threadId not yours"


# ------------------------------------------------------------------ the doubles


class _FakeConversations:
    """The rows ``chatbot_conversations`` holds, keyed by its primary key."""

    def __init__(self, rows: Dict[str, str], *, raises: bool = False) -> None:
        self._rows = rows
        self._raises = raises
        self.lookups: List[str] = []

    async def get_by_session_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        self.lookups.append(session_id)
        if self._raises:
            raise RuntimeError("supabase is unreachable")
        owner = self._rows.get(session_id)
        return None if owner is None else {"session_id": session_id, "user_id": owner}


def _wire_conversations(monkeypatch, fake: _FakeConversations) -> _FakeConversations:
    async def _client() -> object:
        return object()

    monkeypatch.setattr(chat_identity, "get_async_supabase_client", _client)
    monkeypatch.setattr(
        chat_identity, "ChatbotConversationRepository", lambda supabase_client=None: fake
    )
    return fake


@pytest.fixture
def conversations(monkeypatch) -> _FakeConversations:
    return _wire_conversations(
        monkeypatch, _FakeConversations({VICTIM_THREAD: VICTIM, CALLER_THREAD: CALLER})
    )


@pytest.fixture
def dead_lookup(monkeypatch) -> _FakeConversations:
    return _wire_conversations(monkeypatch, _FakeConversations({}, raises=True))


class _Result:
    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self.data = data


class _FakeMessages:
    """Chainable stand-in for the sync ``client.table("chatbot_messages")`` reads.

    Honours the two filters the route resolves by — ``id`` and ``session_id`` —
    so path (a) finds the row by id and path (b) only ever sees rows of the
    session the caller named, exactly as the real query would.
    """

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows
        self._filters: List[tuple[str, Any]] = []

    def table(self, name: str) -> "_FakeMessages":
        assert name == "chatbot_messages", name
        return self

    def select(self, *args: Any, **kwargs: Any) -> "_FakeMessages":
        return self

    def eq(self, column: str, value: Any) -> "_FakeMessages":
        self._filters.append((column, value))
        return self

    def order(self, *args: Any, **kwargs: Any) -> "_FakeMessages":
        return self

    def limit(self, *args: Any, **kwargs: Any) -> "_FakeMessages":
        return self

    def execute(self) -> _Result:
        filters, self._filters = self._filters, []
        rows = [
            row
            for row in self._rows
            if all(row.get(col) == val for col, val in filters if col in ("id", "session_id"))
        ]
        return _Result([dict(r) for r in rows])


def _message(message_id: int, session_id: str) -> Dict[str, Any]:
    return {
        "id": message_id,
        "session_id": session_id,
        "role": "assistant",
        "content": "The TRx performance is up 4% quarter over quarter.",
        "agent_name": "orchestrator",
        "metadata": {"frontend_message_id": f"m-{message_id}"},
        "tool_calls": None,
        "tool_results": None,
    }


class _FakeFeedbackRepo:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def add_feedback(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        return {"id": len(self.calls)}


@pytest.fixture
def writer(monkeypatch) -> _FakeFeedbackRepo:
    """The write seam: records every feedback row the route would persist."""
    repo = _FakeFeedbackRepo()

    async def _client() -> object:
        return object()

    monkeypatch.setattr(factories, "get_async_supabase_client", _client)
    monkeypatch.setattr(
        repositories, "get_chatbot_feedback_repository", lambda supabase_client=None: repo
    )
    return repo


@pytest.fixture
def feedback_client(monkeypatch, writer) -> TestClient:
    """The real router, authenticated as CALLER, over a doubled message table."""
    monkeypatch.setenv("SUPABASE_URL", "http://supabase.test")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key-2109")
    messages = _FakeMessages(
        [
            _message(VICTIM_MESSAGE_ID, VICTIM_THREAD),
            _message(CALLER_MESSAGE_ID, CALLER_THREAD),
            _message(NEW_MESSAGE_ID, NEW_THREAD),
        ]
    )
    monkeypatch.setattr(supabase, "create_client", lambda url, key: messages)
    app = FastAPI()
    app.include_router(ck.router)
    app.dependency_overrides[require_viewer] = lambda: {"id": CALLER}
    return TestClient(app)


def _post(client: TestClient, **body: Any):
    body.setdefault("rating", "thumbs_up")
    return client.post("/copilotkit/feedback", json=body)


def _assert_refused(response, writer: _FakeFeedbackRepo) -> None:
    assert response.status_code == 403, response.text
    assert response.json() == {"detail": SHARED_DETAIL}
    assert writer.calls == [], "a refused rating was still written"


# ------------------------------------------------ T1/T2: someone else's thread


def test_path_a_a_message_in_someone_elses_thread_is_refused(
    feedback_client, writer, conversations
):
    """Path (a): the session comes FROM THE ROW, so the gate can only run after
    the lookup — and must run before the write."""
    response = _post(feedback_client, message_id=VICTIM_MESSAGE_ID)

    _assert_refused(response, writer)
    assert conversations.lookups == [VICTIM_THREAD]


def test_path_b_a_caller_supplied_foreign_session_is_refused(
    feedback_client, writer, conversations
):
    """Path (b): the caller names the session; the stored owner, not the body,
    decides."""
    response = _post(
        feedback_client, session_id=VICTIM_THREAD, message_uuid=f"m-{VICTIM_MESSAGE_ID}"
    )

    _assert_refused(response, writer)
    assert conversations.lookups == [VICTIM_THREAD]


# --------------------------------------------------------- T3: the own thread


def test_the_callers_own_thread_is_rated_on_both_paths(feedback_client, writer, conversations):
    by_id = _post(feedback_client, message_id=CALLER_MESSAGE_ID)
    assert by_id.status_code == 200, by_id.text
    assert by_id.json()["success"] is True

    by_session = _post(
        feedback_client, session_id=CALLER_THREAD, message_uuid=f"m-{CALLER_MESSAGE_ID}"
    )
    assert by_session.status_code == 200, by_session.text
    assert by_session.json()["success"] is True

    assert [c["session_id"] for c in writer.calls] == [CALLER_THREAD, CALLER_THREAD]
    assert [c["message_id"] for c in writer.calls] == [CALLER_MESSAGE_ID, CALLER_MESSAGE_ID]


# ------------------------------------------- T4: a session nobody has opened


def test_a_session_with_no_conversation_row_is_still_rated(feedback_client, writer, conversations):
    """Rule 2 of #2107: no row means nobody to protect, and the write proceeds."""
    response = _post(feedback_client, message_id=NEW_MESSAGE_ID)

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    assert [c["session_id"] for c in writer.calls] == [NEW_THREAD]
    assert conversations.lookups == [NEW_THREAD]


# ------------------------------------------------ T5: the lookup itself fails


def test_a_failed_owner_lookup_allows_the_rating_with_a_warning(
    feedback_client, writer, dead_lookup, caplog
):
    """Fail-OPEN, as on every other ingress: the write goes through the same
    database the check reads, so an outage disables both."""
    with caplog.at_level(logging.WARNING, logger="src.api.routes.chat_identity"):
        response = _post(feedback_client, message_id=VICTIM_MESSAGE_ID)

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    assert [c["session_id"] for c in writer.calls] == [VICTIM_THREAD]
    warnings = [
        r
        for r in caplog.records
        if r.name == "src.api.routes.chat_identity"
        and r.levelno == logging.WARNING
        and "fail-open" in r.getMessage()
    ]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]


# ------------------------------------------------ T6: the swallow trap's teeth


def test_a_403_from_the_gate_escapes_the_routes_broad_except(
    feedback_client, writer, conversations, monkeypatch
):
    """The route body is wrapped in ``except Exception`` that answers 200 with
    an error body. A refusal must reach the client as a 403, not as
    ``{"success": false}`` — this is what ``except HTTPException: raise`` buys."""

    async def _refuse(thread_id: Optional[str], token_user_id: Optional[str]) -> None:
        raise HTTPException(status_code=403, detail=SHARED_DETAIL)

    monkeypatch.setattr(chat_identity, "refuse_foreign_thread", _refuse)

    response = _post(feedback_client, message_id=CALLER_MESSAGE_ID)

    _assert_refused(response, writer)


# --------------------------------------------------------- the helper itself


async def test_the_helper_raises_the_shared_403_only_for_a_foreign_owner(conversations):
    with pytest.raises(HTTPException) as refused:
        await chat_identity.refuse_foreign_thread(VICTIM_THREAD, CALLER)
    assert refused.value.status_code == 403
    assert refused.value.detail == SHARED_DETAIL

    assert await chat_identity.refuse_foreign_thread(CALLER_THREAD, CALLER) is None
    assert await chat_identity.refuse_foreign_thread(NEW_THREAD, CALLER) is None
    assert await chat_identity.refuse_foreign_thread(None, CALLER) is None
    assert await chat_identity.refuse_foreign_thread(VICTIM_THREAD, None) is None
