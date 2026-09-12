"""#2068: tool_composer episodic rows must key on the REAL chat session uuid.

``store_composition`` used to pre-hash the session id with a local
``_ensure_uuid`` helper before calling the episodic writer: a composite
``{user_uuid}~{session_uuid}`` became a uuid5 of the whole string, and an empty
id became a fresh random uuid4. Both are already valid uuids, so the writer's
#1404 coercion passed them through unchanged — the stored session matched no
chat or audit-chain row. The hook must hand the RAW id to the writer, which
recovers the trailing session uuid or stores an honest NULL.

These drive the real writer path (``insert_episodic_memory_with_text`` ->
``insert_episodic_memory``); only the Supabase client and the embedding service
are doubled.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.tool_composer.memory_hooks import ToolComposerMemoryHooks

# Captured at import, before this directory's autouse #883 guard swaps the source
# attribute for a raiser; the helper restores it so the real writer runs against
# a doubled Supabase client.
from src.memory.episodic_memory import insert_episodic_memory_with_text as _real_insert_with_text

_USER = "46d40f52-39ac-4b79-b3a4-1f1292059a00"
_SESSION = "eeba22e7-4d9d-49ea-977b-b9e9d1549c53"
_COMPOSITE = f"{_USER}~{_SESSION}"


@pytest.fixture
def mock_supabase():
    client = MagicMock()
    table = MagicMock()
    client.table.return_value = table
    table.insert.return_value = table
    result = MagicMock()
    result.data = [{"memory_id": "m"}]
    table.execute.return_value = result
    return client


async def _stored_session_record(mock_supabase, session_id: str) -> dict:
    embedding_service = AsyncMock()
    embedding_service.embed.return_value = [0.1] * 1536
    with (
        patch(
            "src.memory.episodic_memory.insert_episodic_memory_with_text", _real_insert_with_text
        ),
        patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase),
        patch("src.memory.episodic_memory.get_embedding_service", return_value=embedding_service),
    ):
        memory_id = await ToolComposerMemoryHooks().store_composition(
            session_id=session_id,
            result={
                "composition_id": "comp-2068",
                "query": "Compare TRx uplift by region",
                "status": "success",
                "success": True,
            },
        )
    # The hook swallows writer errors into None; a None here is a setup failure,
    # not the behaviour under test.
    assert memory_id is not None
    return mock_supabase.table.return_value.insert.call_args[0][0]


@pytest.mark.asyncio
async def test_composite_session_id_stored_as_session_uuid(mock_supabase):
    record = await _stored_session_record(mock_supabase, _COMPOSITE)
    assert record["session_id"] == _SESSION


@pytest.mark.asyncio
async def test_plain_session_uuid_stored_unchanged(mock_supabase):
    record = await _stored_session_record(mock_supabase, _SESSION)
    assert record["session_id"] == _SESSION


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", ["", "cert3b-composer-run"])
async def test_empty_or_non_uuid_session_id_stored_as_null(mock_supabase, session_id):
    record = await _stored_session_record(mock_supabase, session_id)
    # None values are filtered from the record, so the nullable uuid column stores
    # NULL — never an invented uuid4/uuid5.
    assert "session_id" not in record
