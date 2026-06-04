"""Embedding-dimension guard on the memory WRITE paths (review finding M1).

M1: every episodic/procedural write stored the embedding into a ``vector(1536)``
column with NO length check. When the primary embedding service is down, the
``FallbackEmbeddingService`` emits a 384-dim local vector; pgvector then rejects
the insert with a cryptic dimension error — and in agent hooks that error is
swallowed, so the memory write is silently LOST exactly during an outage.

Fix: validate ``len(embedding) == vector_dims`` BEFORE the DB write and fail
fast with a clear, diagnosable error, so a mismatched (e.g. fallback 384-dim)
vector never reaches the database and the failure is explicit, not silent.

The expected dimension is sourced from the memory config
(``get_config().episodic.vector_dims`` / ``.procedural.vector_dims``, both 1536),
which is tied to the actual ``vector(N)`` column width.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.memory.episodic_memory import (
    EpisodicMemoryInput,
    bulk_insert_episodic_memories,
    insert_episodic_memory,
)
from src.memory.procedural_memory import ProceduralMemoryInput, insert_procedural_memory
from src.memory.services.factories import validate_embedding_dimensions

GOOD = [0.0] * 1536
BAD = [0.0] * 384  # what the local fallback emits


def _episode() -> EpisodicMemoryInput:
    return EpisodicMemoryInput(event_type="query", description="why did TRx drop")


def _procedure() -> ProceduralMemoryInput:
    return ProceduralMemoryInput(procedure_name="p", tool_sequence=[{"tool": "x"}])


# ---------------------------------------------------------------------------
# validate_embedding_dimensions helper
# ---------------------------------------------------------------------------


def test_helper_passes_on_matching_dims() -> None:
    # Should not raise.
    validate_embedding_dimensions(GOOD, 1536, context="episodic embedding")


def test_helper_raises_on_mismatch() -> None:
    with pytest.raises(ValueError):
        validate_embedding_dimensions(BAD, 1536, context="episodic embedding")


def test_helper_raises_on_empty() -> None:
    # A present-but-empty list would reach pgvector as an empty vector -> reject.
    with pytest.raises(ValueError):
        validate_embedding_dimensions([], 1536)


def test_helper_allows_none() -> None:
    # None = caller's explicit "store no vector" choice (nullable column);
    # NOT an M1 error. Must not raise.
    validate_embedding_dimensions(None, 1536)


def test_vector_dims_invariant_is_1536() -> None:
    """The guard's expected width must track the actual vector(1536) column
    (database/memory/001_agentic_memory_schema_v1.3.sql). Locks the invariant so
    a config edit can't silently disable the guard (codex LOW-1)."""
    from src.memory.services.config import get_config

    cfg = get_config()
    assert cfg.episodic.vector_dims == 1536
    assert cfg.procedural.vector_dims == 1536


# ---------------------------------------------------------------------------
# insert_episodic_memory
# ---------------------------------------------------------------------------


async def test_insert_episodic_rejects_wrong_dim_before_db() -> None:
    with patch("src.memory.episodic_memory.get_supabase_client") as mock_client:
        with pytest.raises(ValueError):
            await insert_episodic_memory(_episode(), BAD)
    mock_client.assert_not_called()


async def test_insert_episodic_accepts_correct_dim() -> None:
    client = MagicMock()
    with patch("src.memory.episodic_memory.get_supabase_client", return_value=client):
        memory_id = await insert_episodic_memory(_episode(), GOOD)
    assert isinstance(memory_id, str)
    client.table.return_value.insert.return_value.execute.assert_called_once()


async def test_insert_episodic_allows_none_embedding() -> None:
    """Agent hooks pass embedding=None to store a memory with no vector (the
    column is nullable). The M1 guard must NOT break that established fleet
    pattern — only a present, wrong-width vector is rejected (codex HIGH-1)."""
    client = MagicMock()
    with patch("src.memory.episodic_memory.get_supabase_client", return_value=client):
        memory_id = await insert_episodic_memory(_episode(), None)  # type: ignore[arg-type]
    assert isinstance(memory_id, str)
    client.table.return_value.insert.return_value.execute.assert_called_once()


# ---------------------------------------------------------------------------
# bulk_insert_episodic_memories
# ---------------------------------------------------------------------------


async def test_bulk_insert_rejects_wrong_dim_before_db() -> None:
    with patch("src.memory.episodic_memory.get_supabase_client") as mock_client:
        with pytest.raises(ValueError):
            await bulk_insert_episodic_memories([(_episode(), GOOD), (_episode(), BAD)])
    mock_client.assert_not_called()


# ---------------------------------------------------------------------------
# insert_procedural_memory
# ---------------------------------------------------------------------------


async def test_insert_procedural_rejects_wrong_dim_before_db() -> None:
    with patch("src.memory.procedural_memory.get_supabase_client") as mock_client:
        with pytest.raises(ValueError):
            await insert_procedural_memory(_procedure(), BAD)
    mock_client.assert_not_called()
