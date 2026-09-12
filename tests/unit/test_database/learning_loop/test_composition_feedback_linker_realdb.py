"""The nightly linker labels a composition with the user's own verdict (spec §5, O1).

Opt-in: ``E2I_DB_INTEGRATION=1``. Episodes are written through the real recording RPCs and the
feedback rows through the real table, so the matcher is exercised against the shapes production
actually stores.

What the linker may and may not do:
- a thumbs up sets ``success = true``; a thumbs down sets ``success = false``;
- ``feedback_text`` is never written — the comment is free user text, which §5.5 keeps out of the
  learning tables — only ``feedback_at``;
- it matches within one session and a bounded time window, so an unrelated later composition in
  the same session is not labelled by an older rating;
- it never changes behaviour, only labels;
- re-running it changes nothing (the nightly beat runs it again tomorrow).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pytest

from src.agents.tool_composer.registry_sync import RegistrySync
from tests.unit.test_database.learning_loop import _pg

pytestmark = [
    pytest.mark.skipif(
        not _pg.db_integration_enabled(),
        reason="real-DB integration; set E2I_DB_INTEGRATION=1 on the droplet (docker + supabase-db)",
    ),
    pytest.mark.timeout(600),
]

UPTO = "ml/041_composer_learning_loop_recording.sql"
SESSION = "11111111-1111-1111-1111-111111111111~sess-1"


@pytest.fixture
def synced(clone_db) -> _pg.PgConn:
    db = clone_db("feedback_linker")
    _pg.migrate(db, UPTO)
    asyncio.run(RegistrySync(port=_pg.PsycopgRpcPort(db)).sync_once())
    return db


def _seed(cid: str, session_id: str = SESSION) -> Dict[str, Any]:
    return {
        "composition_id": cid,
        "query_text": "which regions drive TRx?",
        "session_id": session_id,
        "user_id": "user-1",
        "entry_point": "chat_tool",
        "brand": "Kisqali",
        "region": "US",
        "audit_workflow_id": None,
        "is_synthetic": True,
    }


async def _record_episode(
    port: _pg.PsycopgRpcPort, cid: str, *, session_id: str = SESSION, minutes_ago: int = 0
) -> None:
    seed = _seed(cid, session_id)
    await port.call("composer_record_start", {"p_seed": seed})
    await port.call(
        "composer_record_finish",
        {
            "p_seed": seed,
            "p_final": {
                "status": "COMPLETED",
                "outcome": "success",
                "plan_source": "llm",
                "total_latency_ms": 1000,
                "tools_executed": 1,
                "tools_succeeded": 1,
            },
        },
    )
    if minutes_ago:
        stamp = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
        _exec(
            port.conn,
            "UPDATE composer_episodes SET created_at = %s::timestamptz, "
            "last_activity_at = %s::timestamptz WHERE composition_id = %s",
            (stamp, stamp, cid),
        )


def _exec(db: _pg.PgConn, sql: str, params: tuple = ()) -> None:
    with db.connect() as conn:
        conn.execute(sql, params)
        conn.commit()


def _feedback(
    db: _pg.PgConn, rating: str, *, session_id: str = SESSION, minutes_ago: int = 0
) -> None:
    """A real chatbot_message_feedback row, through the real conversation/message chain."""
    stamp = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    # The chat chain is real: a profile owns the conversation, which owns the message the rating
    # hangs off. chatbot_conversations.user_id is NOT NULL and references that profile.
    owner = session_id.split("~", 1)[0]
    _exec(
        db,
        "INSERT INTO chatbot_user_profiles (id, email) VALUES (%s::uuid, %s) "
        "ON CONFLICT DO NOTHING",
        (owner, f"{owner}@example.invalid"),
    )
    _exec(
        db,
        "INSERT INTO chatbot_conversations (session_id, user_id) VALUES (%s, %s::uuid) "
        "ON CONFLICT DO NOTHING",
        (session_id, owner),
    )
    _exec(
        db,
        "INSERT INTO chatbot_messages (session_id, role, content) VALUES (%s, 'assistant', 'a') "
        "ON CONFLICT DO NOTHING",
        (session_id,),
    )
    _exec(
        db,
        "INSERT INTO chatbot_message_feedback (message_id, session_id, rating, comment, "
        "query_text, created_at) SELECT id, %s, %s::chatbot_feedback_rating, %s, %s, "
        "%s::timestamptz FROM chatbot_messages WHERE session_id = %s ORDER BY id DESC LIMIT 1",
        (session_id, rating, "user typed this", "which regions drive TRx?", stamp, session_id),
    )


def _episode(db: _pg.PgConn, cid: str) -> Dict[str, Any]:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT success, feedback_text, feedback_at FROM composer_episodes "
            "WHERE composition_id = %s",
            (cid,),
        ).fetchone()
    return {"success": row[0], "feedback_text": row[1], "feedback_at": row[2]}


def _link(db: _pg.PgConn, **kwargs: Any) -> Dict[str, Any]:
    return _link_with(_client(db), **kwargs)


def _link_with(client: Any, **kwargs: Any) -> Dict[str, Any]:
    from src.tasks.composition_feedback_tasks import link_composition_feedback

    return link_composition_feedback(client=client, **kwargs)


def _client(db: _pg.PgConn) -> Any:
    from tests.unit.test_database.learning_loop.test_admin_tool_composer_realdb import (
        PsycopgSupabase,
    )

    return PsycopgSupabase(db)


async def test_a_thumbs_up_marks_the_composition_successful(synced):
    await _record_episode(_pg.PsycopgRpcPort(synced), "comp_up")
    _feedback(synced, "thumbs_up")

    result = _link(synced)

    assert result["labelled"] == 1 and result["failed"] == 0
    episode = _episode(synced, "comp_up")
    assert episode["success"] is True and episode["feedback_at"] is not None
    # The comment is free user text; §5.5 keeps it out of the learning tables.
    assert episode["feedback_text"] is None


async def test_a_failed_write_is_reported_not_counted_as_nothing_to_do(synced):
    """A run whose writes all fail must not read like a run that found no feedback."""
    await _record_episode(_pg.PsycopgRpcPort(synced), "comp_write_fails")
    _feedback(synced, "thumbs_up")

    class RefusingClient:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def table(self, name: str) -> Any:
            query = self.inner.table(name)
            query.update = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("write refused"))
            return query

    result = _link_with(RefusingClient(_client(synced)))

    assert result["labelled"] == 0 and result["failed"] == 1
    assert _episode(synced, "comp_write_fails")["success"] is None


async def test_a_thumbs_down_marks_it_unsuccessful_without_storing_the_comment(synced):
    await _record_episode(_pg.PsycopgRpcPort(synced), "comp_down")
    _feedback(synced, "thumbs_down")

    _link(synced)

    episode = _episode(synced, "comp_down")
    assert episode["success"] is False and episode["feedback_text"] is None


async def test_an_episode_with_no_feedback_is_left_alone(synced):
    await _record_episode(_pg.PsycopgRpcPort(synced), "comp_none")

    result = _link(synced)

    assert result["labelled"] == 0
    assert _episode(synced, "comp_none") == {
        "success": None,
        "feedback_text": None,
        "feedback_at": None,
    }


async def test_the_rating_labels_the_composition_it_followed_not_an_earlier_one(synced):
    """Two compositions in one session: the rating belongs to the one it came after."""
    port = _pg.PsycopgRpcPort(synced)
    await _record_episode(port, "comp_older", minutes_ago=120)
    await _record_episode(port, "comp_recent", minutes_ago=1)
    _feedback(synced, "thumbs_up", minutes_ago=0)

    _link(synced)

    assert _episode(synced, "comp_recent")["success"] is True
    assert _episode(synced, "comp_older")["success"] is None


async def test_one_rating_labels_one_composition(synced):
    """Two compositions minutes apart, one rating: it belongs to the one it followed.

    The earlier regression put the other composition two hours away, outside the window, so it
    proved nothing about the case that actually happens — a user asking twice, then rating.
    """
    port = _pg.PsycopgRpcPort(synced)
    await _record_episode(port, "comp_first", minutes_ago=11)
    await _record_episode(port, "comp_second", minutes_ago=1)
    _feedback(synced, "thumbs_up", minutes_ago=0)

    result = _link(synced)

    assert result["labelled"] == 1
    assert _episode(synced, "comp_second")["success"] is True
    assert _episode(synced, "comp_first")["success"] is None


async def test_a_rating_never_labels_another_users_composition(synced):
    """Sessions are per user, but a composition carries its own user: match on both."""
    other_session = "22222222-2222-2222-2222-222222222222~sess-2"
    port = _pg.PsycopgRpcPort(synced)
    await _record_episode(port, "comp_mine", session_id=SESSION, minutes_ago=1)
    await _record_episode(port, "comp_theirs", session_id=other_session, minutes_ago=1)
    _feedback(synced, "thumbs_down", session_id=SESSION, minutes_ago=0)

    _link(synced)

    assert _episode(synced, "comp_mine")["success"] is False
    assert _episode(synced, "comp_theirs")["success"] is None


async def test_running_it_again_changes_nothing(synced):
    await _record_episode(_pg.PsycopgRpcPort(synced), "comp_idem")
    _feedback(synced, "thumbs_up")

    first = _link(synced)
    before = _episode(synced, "comp_idem")
    second = _link(synced)

    assert first["labelled"] == 1 and second["labelled"] == 0
    assert _episode(synced, "comp_idem") == before
