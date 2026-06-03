"""
Unit tests for src/api/routes/memory.py

Focus areas (security review finding #4 — episodic memory tenant scoping):
- Episodic create/list/get enforce the caller's brand grants (the existing
  tenant boundary in the ``episodic_memories`` table — there is no per-user
  ownership column; brand/region/agent_name are the scope keys).
- PHI/PII references (patient_id, hcp_id) are NOT echoed in responses.
- Error handlers return generic messages (no str(e) disclosure).

Tests call the endpoint coroutines directly and pass an explicit ``user``
dict (the same shape ``require_viewer`` yields), so brand grants come from
``app_metadata.brands``.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

from src.api.routes.memory import (
    EpisodicMemoryInput,
    create_episodic_memory,
    get_episodic_memory_endpoint,
    list_episodic_memories,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def admin_user():
    """Cross-brand admin (sees/writes any brand)."""
    return {
        "id": "admin-001",
        "email": "admin@e2i-analytics.com",
        "app_metadata": {"role": "admin"},
    }


@pytest.fixture
def kisqali_user():
    """Viewer granted the 'Kisqali' brand only."""
    return {
        "id": "user-kisqali",
        "email": "kis@e2i-analytics.com",
        "app_metadata": {"role": "viewer", "brands": ["Kisqali"]},
    }


@pytest.fixture
def fabhalta_user():
    """Viewer granted the 'Fabhalta' brand only (NOT Kisqali)."""
    return {
        "id": "user-fabhalta",
        "email": "fab@e2i-analytics.com",
        "app_metadata": {"role": "viewer", "brands": ["Fabhalta"]},
    }


# =============================================================================
# create_episodic_memory
# =============================================================================


class TestCreateEpisodicMemory:
    @pytest.mark.asyncio
    async def test_create_in_granted_brand_allowed(self, kisqali_user):
        """A user may create a memory in a brand they are granted."""
        payload = EpisodicMemoryInput(
            content="User asked about TRx trends",
            event_type="query",
            brand="Kisqali",
            patient_id="PAT_123",
            hcp_id="HCP_999",
        )
        with patch(
            "src.api.routes.memory.insert_episodic_memory_with_text",
            new=AsyncMock(return_value="mem-1"),
        ):
            resp = await create_episodic_memory(payload, BackgroundTasks(), user=kisqali_user)
            assert resp.id == "mem-1"
            assert resp.brand == "Kisqali"

    @pytest.mark.asyncio
    async def test_create_phi_not_echoed_in_response(self, kisqali_user):
        """FINDING #4: patient_id/hcp_id are accepted but NOT echoed back as
        response fields (PHI/PII minimization)."""
        payload = EpisodicMemoryInput(
            content="note",
            event_type="action",
            brand="Kisqali",
            patient_id="PAT_SECRET",
            hcp_id="HCP_SECRET",
        )
        with patch(
            "src.api.routes.memory.insert_episodic_memory_with_text",
            new=AsyncMock(return_value="mem-2"),
        ):
            resp = await create_episodic_memory(payload, BackgroundTasks(), user=kisqali_user)
            dumped = resp.model_dump()
            assert "patient_id" not in dumped
            assert "hcp_id" not in dumped
            assert "PAT_SECRET" not in str(dumped)
            assert "HCP_SECRET" not in str(dumped)

    @pytest.mark.asyncio
    async def test_create_out_of_grant_brand_blocked(self, fabhalta_user):
        """FINDING #4: a user cannot write a memory into a brand they lack a
        grant for. Must 403 and NOT call the insert."""
        payload = EpisodicMemoryInput(
            content="cross-brand write",
            event_type="query",
            brand="Kisqali",  # fabhalta_user only has Fabhalta
        )
        insert = AsyncMock(return_value="should-not-happen")
        with patch("src.api.routes.memory.insert_episodic_memory_with_text", new=insert):
            with pytest.raises(HTTPException) as exc_info:
                await create_episodic_memory(payload, BackgroundTasks(), user=fabhalta_user)
            assert exc_info.value.status_code == 403
            insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_error_is_generic(self, kisqali_user):
        """FINDING #3: 500 detail must not leak str(e)."""
        payload = EpisodicMemoryInput(content="x", event_type="query", brand="Kisqali")
        with patch(
            "src.api.routes.memory.insert_episodic_memory_with_text",
            new=AsyncMock(side_effect=Exception("db boom secret")),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_episodic_memory(payload, BackgroundTasks(), user=kisqali_user)
            assert exc_info.value.status_code == 500
            assert "db boom secret" not in str(exc_info.value.detail)


# =============================================================================
# list_episodic_memories
# =============================================================================


class TestListEpisodicMemories:
    @pytest.mark.asyncio
    async def test_list_scoped_to_caller_brand(self, kisqali_user):
        """FINDING #4: a non-admin's list is scoped to their granted brand,
        even when no ?brand filter is supplied (no cross-brand enumeration)."""
        recent = AsyncMock(return_value=[])
        with patch("src.api.routes.memory.get_recent_memories", new=recent):
            await list_episodic_memories(user=kisqali_user)
            _, kwargs = recent.call_args
            assert kwargs["brand"] == "Kisqali"

    @pytest.mark.asyncio
    async def test_list_out_of_grant_brand_returns_empty(self, fabhalta_user):
        """FINDING #4: requesting a brand the caller lacks returns an empty
        list (defensive — does not leak existence)."""
        recent = AsyncMock(return_value=[])
        with patch("src.api.routes.memory.get_recent_memories", new=recent):
            result = await list_episodic_memories(brand="Kisqali", user=fabhalta_user)
            assert result == []
            recent.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_admin_any_brand(self, admin_user):
        """Admin may list any brand (or all brands when unfiltered)."""
        recent = AsyncMock(return_value=[])
        with patch("src.api.routes.memory.get_recent_memories", new=recent):
            await list_episodic_memories(brand="Kisqali", user=admin_user)
            _, kwargs = recent.call_args
            assert kwargs["brand"] == "Kisqali"

    @pytest.mark.asyncio
    async def test_list_phi_not_echoed(self, kisqali_user):
        """FINDING #4: even if a stored row carries patient_id/hcp_id, the
        response does not surface them."""
        row = {
            "memory_id": "m1",
            "description": "d",
            "event_type": "query",
            "brand": "Kisqali",
            "patient_id": "PAT_LEAK",
            "hcp_id": "HCP_LEAK",
            "occurred_at": datetime.now(timezone.utc),
        }
        with patch(
            "src.api.routes.memory.get_recent_memories",
            new=AsyncMock(return_value=[row]),
        ):
            result = await list_episodic_memories(user=kisqali_user)
            assert len(result) == 1
            dumped = result[0].model_dump()
            assert "PAT_LEAK" not in str(dumped)
            assert "HCP_LEAK" not in str(dumped)


# =============================================================================
# get_episodic_memory_endpoint
# =============================================================================


class TestGetEpisodicMemory:
    @pytest.mark.asyncio
    async def test_get_in_grant_allowed(self, kisqali_user):
        with patch(
            "src.api.routes.memory.get_memory_by_id",
            new=AsyncMock(
                return_value={
                    "memory_id": "m1",
                    "description": "d",
                    "event_type": "query",
                    "brand": "Kisqali",
                }
            ),
        ):
            resp = await get_episodic_memory_endpoint("m1", user=kisqali_user)
            assert resp.id == "m1"

    @pytest.mark.asyncio
    async def test_get_out_of_grant_404(self, fabhalta_user):
        """FINDING #4: fetching a memory from a brand you lack returns 404
        (existence not disclosed)."""
        with patch(
            "src.api.routes.memory.get_memory_by_id",
            new=AsyncMock(
                return_value={
                    "memory_id": "m1",
                    "description": "d",
                    "event_type": "query",
                    "brand": "Kisqali",
                }
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_episodic_memory_endpoint("m1", user=fabhalta_user)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_not_found(self, admin_user):
        with patch(
            "src.api.routes.memory.get_memory_by_id",
            new=AsyncMock(return_value=None),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_episodic_memory_endpoint("nope", user=admin_user)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_error_is_generic(self, admin_user):
        """FINDING #3: 500 detail must not leak str(e)."""
        with patch(
            "src.api.routes.memory.get_memory_by_id",
            new=AsyncMock(side_effect=Exception("internal path /secret")),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_episodic_memory_endpoint("m1", user=admin_user)
            assert exc_info.value.status_code == 500
            assert "internal path /secret" not in str(exc_info.value.detail)
