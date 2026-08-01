"""Writer-boundary session_id coercion for episodic memory (#1404, #1403).

``episodic_memories.session_id`` is a ``uuid`` column. Chat surfaces hand the
writer a composite ``{user_uuid}~{session_uuid}`` id (and, since PR #1394, a
``~bridge``-suffixed variant). Previously the writer passed the value straight
through — a non-uuid failed ``22P02`` at the DB and was swallowed by the agent
hooks' broad excepts (#1393), or a caller minted a *random* uuid that correlates
with nothing (#1403).

This pins the writer-boundary contract via ``coerce_session_uuid``:
- composite / bridge id  -> the real trailing session uuid;
- a plain valid uuid     -> unchanged (idempotent — the common case is untouched);
- any other non-uuid     -> honest NULL (the None-filter drops the key), NEVER a
  fabricated random uuid.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.memory.episodic_memory import (
    EpisodicMemoryInput,
    bulk_insert_episodic_memories,
    insert_episodic_memory,
)

_USER = "46d40f52-39ac-4b79-b3a4-1f1292059a00"
_SESSION = "53f47dba-378e-4c39-96d9-ec3fda26e168"
_COMPOSITE = f"{_USER}~{_SESSION}"
_EMBEDDING = [0.1] * 1536


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


def _memory() -> EpisodicMemoryInput:
    return EpisodicMemoryInput(
        event_type="query_answer",
        description="Answered a causal question",
        agent_name="causal_impact",
    )


def _inserted_record(mock_supabase) -> dict:
    return mock_supabase.table.return_value.insert.call_args[0][0]


def _bulk_inserted_record(mock_supabase) -> dict:
    # bulk_insert_episodic_memories calls insert() with a LIST of records.
    return mock_supabase.table.return_value.insert.call_args[0][0][0]


@pytest.mark.asyncio
async def test_composite_session_id_coerced_to_session_uuid(mock_supabase):
    with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
        await insert_episodic_memory(memory=_memory(), embedding=_EMBEDDING, session_id=_COMPOSITE)
    assert _inserted_record(mock_supabase)["session_id"] == _SESSION


@pytest.mark.asyncio
async def test_bridge_suffixed_session_id_coerced_to_session_uuid(mock_supabase):
    with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
        await insert_episodic_memory(
            memory=_memory(), embedding=_EMBEDDING, session_id=f"{_COMPOSITE}~bridge"
        )
    assert _inserted_record(mock_supabase)["session_id"] == _SESSION


@pytest.mark.asyncio
async def test_non_uuid_session_id_stored_as_null_not_random(mock_supabase):
    with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
        await insert_episodic_memory(memory=_memory(), embedding=_EMBEDDING, session_id="sess_123")
    # None values are filtered from the record, so the uuid column defaults to NULL.
    assert "session_id" not in _inserted_record(mock_supabase)


@pytest.mark.asyncio
async def test_valid_uuid_session_id_unchanged(mock_supabase):
    with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
        await insert_episodic_memory(memory=_memory(), embedding=_EMBEDDING, session_id=_SESSION)
    assert _inserted_record(mock_supabase)["session_id"] == _SESSION


@pytest.mark.asyncio
async def test_bulk_composite_session_id_coerced(mock_supabase):
    with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
        await bulk_insert_episodic_memories(
            memories=[(_memory(), _EMBEDDING)], session_id=_COMPOSITE
        )
    assert _bulk_inserted_record(mock_supabase)["session_id"] == _SESSION


@pytest.mark.asyncio
async def test_bulk_non_uuid_session_id_stored_as_null(mock_supabase):
    with patch("src.memory.episodic_memory.get_supabase_client", return_value=mock_supabase):
        await bulk_insert_episodic_memories(
            memories=[(_memory(), _EMBEDDING)], session_id="sess_123"
        )
    assert "session_id" not in _bulk_inserted_record(mock_supabase)
