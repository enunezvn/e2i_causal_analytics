"""#1405: CopilotKit persistence — real-owner attribution + null-safe owner derivation.

Two failure modes fixed together:
  * `_ensure_conversation_exists` hardcoded the anon sentinel, discarding the
    JWT-verified owner that the auth gate already stashed in the attribution
    contextvar. These tests pin that the conversation is attributed to the real
    owner (falling back to anon only when genuinely unattributed).
  * Migration 123 replaces the `CAST(SPLIT_PART(session_id,'~',1) AS UUID)`
    generated column (which 22P02-crashed on a non-UUID threadId, dropping the
    message + its feedback and starving the human-thumbs signal) with a trigger
    that inherits the parent conversation's user_id. The parity test pins the DDL.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.llm_attribution import ANONYMOUS_USER_ID, LLMAttribution

_MIGRATION = Path("database/migrations/123_chatbot_message_owner_inherit.sql")
_OWNER = "44444444-4444-4444-4444-444444444444"


def _repo_capturing_insert():
    """A conversation repo whose client records the chatbot_conversations insert payload
    and reports 'no existing conversation' so the create branch runs."""
    client = MagicMock()
    tbl = client.table.return_value
    tbl.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    tbl.insert.return_value.execute.return_value.data = [{"session_id": "x"}]
    repo = MagicMock()
    repo.client = client
    return repo, client


@pytest.mark.asyncio
async def test_conversation_attributed_to_jwt_owner_not_anon():
    """A bare-uuid threadId (no user~ prefix) must still attribute the conversation to
    the JWT-verified owner (from get_attribution()), NOT the anon sentinel."""
    from src.api.routes import copilotkit

    repo, client = _repo_capturing_insert()
    with (
        patch.object(copilotkit, "get_attribution", return_value=LLMAttribution(user_id=_OWNER)),
        patch.object(copilotkit, "_get_chatbot_conversation_repository", return_value=repo),
    ):
        ok = await copilotkit._ensure_conversation_exists("bare-thread-uuid")
    assert ok is True
    payload = client.table.return_value.insert.call_args[0][0]
    assert payload["user_id"] == _OWNER


@pytest.mark.asyncio
async def test_conversation_falls_back_to_anon_when_unattributed():
    """With no attribution in scope, the owner is the honest anon sentinel (never fabricated)."""
    from src.api.routes import copilotkit

    repo, client = _repo_capturing_insert()
    with (
        patch.object(copilotkit, "get_attribution", return_value=None),
        patch.object(copilotkit, "_get_chatbot_conversation_repository", return_value=repo),
    ):
        ok = await copilotkit._ensure_conversation_exists("some-thread")
    assert ok is True
    payload = client.table.return_value.insert.call_args[0][0]
    assert payload["user_id"] == ANONYMOUS_USER_ID


def test_migration_123_inherits_owner_and_drops_generated_cast():
    """Migration 123 must stop casting split_part(session_id) on BOTH tables (the 22P02
    source) and inherit the owner from the parent conversation via a trigger."""
    sql = _MIGRATION.read_text()
    assert "chatbot_messages ALTER COLUMN computed_user_id DROP EXPRESSION" in sql
    assert "chatbot_message_feedback ALTER COLUMN computed_user_id DROP EXPRESSION" in sql
    assert "chatbot_inherit_conversation_owner" in sql
    assert re.search(r"SELECT\s+user_id\s+INTO\s+NEW\.computed_user_id", sql)
    assert "trg_chatbot_messages_inherit_owner" in sql
    assert "trg_chatbot_message_feedback_inherit_owner" in sql
