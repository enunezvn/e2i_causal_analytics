"""Unit tests for the shared session-id -> uuid coercion (#1393).

The chat surfaces use a composite ``{user_uuid}~{session_uuid}`` session id (the
DB even derives ``chatbot_messages.computed_user_id`` from
``split_part(session_id, '~', 1)``). Several ``uuid``-typed columns key on the
session (``audit_chain_entries.session_id``, ``episodic_memories.session_id``);
the composite string fails Postgres 22P02 ("invalid input syntax for type
uuid"). Since PR #1394 the conversational bridge also appends a ``~bridge``
marker, producing ``{user}~{session}~bridge`` whose trailing segment is NOT a
uuid.

``coerce_session_uuid`` is the single hardened helper both consumers share.
"""

from __future__ import annotations

import uuid
from uuid import UUID

import pytest

from src.utils.session_ids import coerce_session_uuid


def test_plain_uuid_string_is_preserved() -> None:
    u = uuid.uuid4()
    assert coerce_session_uuid(str(u)) == u


def test_uuid_object_is_passed_through() -> None:
    u = uuid.uuid4()
    assert coerce_session_uuid(u) == u


def test_composite_returns_trailing_session_uuid() -> None:
    """``{user_uuid}~{session_uuid}`` -> the session uuid (trailing segment)."""
    session_uuid = uuid.uuid4()
    composite = f"{uuid.uuid4()}~{session_uuid}"
    assert coerce_session_uuid(composite) == session_uuid


def test_bridge_suffixed_composite_returns_session_uuid() -> None:
    """``{user}~{session}~bridge`` (PR #1394) -> the middle session uuid.

    The trailing ``bridge`` marker is not uuid-parseable; the helper recovers
    the rightmost uuid-shaped segment, which is the session's own uuid.
    """
    session_uuid = uuid.uuid4()
    bridged = f"{uuid.uuid4()}~{session_uuid}~bridge"
    assert coerce_session_uuid(bridged) == session_uuid


def test_bridge_suffixed_plain_uuid_returns_that_uuid() -> None:
    """A bare ``{plain_uuid}~bridge`` (AG-UI threadId + bridge marker) -> the uuid.

    Guards that stripping the marker recovers the session even when there is no
    user segment, so the coercion is not mis-tuned to only the composite shape.
    """
    session_uuid = uuid.uuid4()
    assert coerce_session_uuid(f"{session_uuid}~bridge") == session_uuid


def test_malformed_composite_returns_none_not_user_uuid() -> None:
    """``{user_uuid}~garbage`` -> None, NOT the user uuid (codex #1393 iter-1 LOW).

    A corrupt session segment must yield an honest null rather than silently
    mis-associating the episodic row to the user's uuid.
    """
    user_uuid = uuid.uuid4()
    assert coerce_session_uuid(f"{user_uuid}~garbage") is None


def test_none_is_none() -> None:
    assert coerce_session_uuid(None) is None


@pytest.mark.parametrize("bad", ["", "not-a-uuid", "a~b~c", "user~also-not-uuid"])
def test_non_uuid_returns_none_not_fabricated(bad: str) -> None:
    """No uuid-shaped segment -> honest ``None``, never a fabricated id.

    Mirrors the audit-chain contract (``test_1335_chatstream_persistence``): a
    garbage id yields ``None`` for the nullable column rather than a made-up uuid.
    """
    assert coerce_session_uuid(bad) is None


def test_result_is_uuid_type() -> None:
    """The value is a real ``UUID`` (str() gives a valid uuid-column literal)."""
    session_uuid = uuid.uuid4()
    coerced = coerce_session_uuid(f"{uuid.uuid4()}~{session_uuid}")
    assert isinstance(coerced, UUID)
    assert UUID(str(coerced)) == session_uuid
