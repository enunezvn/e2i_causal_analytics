"""Brand-membership enforcement on sentinels routes.

PR #250's initial implementation admitted in module docstring that the auth
layer didn't surface per-brand grants, so list_sentinels returned whatever
the user could see by JWT role. Operators with Brand-X grants could pass
``?brand=Brand-Y`` and see Brand-Y's sentinels.

Until full RLS lands, the routes now:

* Defensive empty list on out-of-grant ``?brand=X`` (avoids existence leak).
* Restrict no-``?brand`` listings to the caller's allowed brand set.
* 404 (not 403) on out-of-grant ``get_sentinel`` (same info-leak defense).
* Admin role (or brand grant ``'all'``) retains cross-brand access.

These tests use direct function-level invocation of ``list_sentinels`` /
``get_sentinel`` with patched dependencies so the brand-membership branches
are exercised independently of FastAPI routing / JWT decoding.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


def _operator_brand_x() -> Dict[str, Any]:
    return {"sub": "user-operator", "role": "operator", "brands": ["Brand-X"]}


def _admin_user() -> Dict[str, Any]:
    return {"sub": "user-admin", "role": "admin", "brands": []}


def _operator_no_brands() -> Dict[str, Any]:
    return {"sub": "user-zero", "role": "operator", "brands": []}


def _make_db_with_rows(*rows: Dict[str, Any]) -> MagicMock:
    """Build a MagicMock supabase client that returns the given rows.

    Mirrors the chained ``client.table(...).select(...).execute()`` shape
    sentinels.py uses. Doesn't enforce filters — the route's filter logic
    is the unit under test.
    """
    db = MagicMock()
    chain = db.table.return_value
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.in_.return_value = chain
    chain.limit.return_value = chain
    chain.execute.return_value = MagicMock(data=list(rows))
    return db


@pytest.mark.asyncio
async def test_operator_cannot_list_other_brand_sentinels() -> None:
    """Operator with only Brand-X grant must NOT see Brand-Y rows via
    ``?brand=Brand-Y``. Defensive empty list returned BEFORE DB call.
    """
    from src.api.routes.sentinels import list_sentinels

    with patch(
        "src.api.routes.sentinels.get_supabase_client",
        return_value=_make_db_with_rows(
            {
                "sentinel_id": "s-1",
                "name": "kis-sentinel",
                "pattern_type": "freshness",
                "action_type": "notify",
                "brand": "Brand-Y",
                "region": None,
                "enabled": True,
                "last_fired_at": None,
                "fire_count": 0,
            }
        ),
    ):
        result = await list_sentinels(brand="Brand-Y", enabled_only=True, user=_operator_brand_x())

    assert result == [], f"expected empty list, got {result}"


@pytest.mark.asyncio
async def test_admin_can_list_any_brand_sentinels() -> None:
    """Admin role retains cross-brand access regardless of their brands claim."""
    from src.api.routes.sentinels import list_sentinels

    row = {
        "sentinel_id": "s-1",
        "name": "kis-sentinel",
        "pattern_type": "freshness",
        "action_type": "notify",
        "brand": "Brand-Y",
        "region": None,
        "enabled": True,
        "last_fired_at": None,
        "fire_count": 0,
    }
    with patch(
        "src.api.routes.sentinels.get_supabase_client",
        return_value=_make_db_with_rows(row),
    ):
        result = await list_sentinels(brand="Brand-Y", enabled_only=True, user=_admin_user())

    assert len(result) == 1, result
    assert result[0].brand == "Brand-Y"


@pytest.mark.asyncio
async def test_operator_with_brand_can_list_own_brand_sentinels() -> None:
    """Operator with Brand-X grant CAN see Brand-X sentinels."""
    from src.api.routes.sentinels import list_sentinels

    row = {
        "sentinel_id": "s-1",
        "name": "bx-sentinel",
        "pattern_type": "freshness",
        "action_type": "notify",
        "brand": "Brand-X",
        "region": None,
        "enabled": True,
        "last_fired_at": None,
        "fire_count": 0,
    }
    with patch(
        "src.api.routes.sentinels.get_supabase_client",
        return_value=_make_db_with_rows(row),
    ):
        result = await list_sentinels(brand="Brand-X", enabled_only=True, user=_operator_brand_x())

    assert len(result) == 1, result
    assert result[0].brand == "Brand-X"


@pytest.mark.asyncio
async def test_operator_no_brands_no_filter_gets_empty() -> None:
    """Operator with no brand grants AND no ?brand filter sees nothing.

    Defensive: prevents an unprivileged operator from listing every
    sentinel via the no-brand-param path.
    """
    from src.api.routes.sentinels import list_sentinels

    with patch(
        "src.api.routes.sentinels.get_supabase_client",
        return_value=_make_db_with_rows(),
    ):
        result = await list_sentinels(brand=None, enabled_only=True, user=_operator_no_brands())

    assert result == []


@pytest.mark.asyncio
async def test_get_sentinel_returns_404_for_out_of_grant_row() -> None:
    """get_sentinel: out-of-grant row returns 404 (not 403) — info-leak defense."""
    from fastapi import HTTPException

    from src.api.routes.sentinels import get_sentinel

    out_of_grant_row = {
        "sentinel_id": "s-1",
        "name": "kis-sentinel",
        "pattern_type": "freshness",
        "action_type": "notify",
        "brand": "Brand-Y",
        "region": None,
        "enabled": True,
        "last_fired_at": None,
        "fire_count": 0,
    }
    with patch(
        "src.api.routes.sentinels.get_supabase_client",
        return_value=_make_db_with_rows(out_of_grant_row),
    ):
        with pytest.raises(HTTPException) as excinfo:
            await get_sentinel("s-1", user=_operator_brand_x())

    assert excinfo.value.status_code == 404, (
        f"out-of-grant get_sentinel must return 404 (not {excinfo.value.status_code} "
        f"— info-leak defense)"
    )


@pytest.mark.asyncio
async def test_create_sentinel_rejects_out_of_grant_brand() -> None:
    """Operator with Brand-X grant cannot create Brand-Y sentinel.

    Codex iter-0 HIGH-3: pre-iter-1 only ``brand='all'`` was blocked for
    non-admins. Any operator could register a sentinel for any specific
    brand. Now ``create_sentinel`` enforces the same brand-grant check
    as list/get.
    """
    from fastapi import HTTPException

    from src.api.routes.sentinels import SentinelCreateRequest, create_sentinel

    payload = SentinelCreateRequest(
        name="oob-sentinel",
        pattern_type="freshness",
        pattern_config={"table": "triggers", "ts_column": "updated_at", "max_age_hours": 24},
        action_type="notify",
        action_config={},
        brand="Brand-Y",
    )
    with pytest.raises(HTTPException) as excinfo:
        await create_sentinel(payload=payload, user=_operator_brand_x())

    assert excinfo.value.status_code == 403
    assert "brand" in str(excinfo.value.detail).lower()


@pytest.mark.asyncio
async def test_create_sentinel_admin_can_register_any_brand() -> None:
    """Admin retains cross-brand registration capability."""
    from src.api.routes.sentinels import SentinelCreateRequest, create_sentinel

    payload = SentinelCreateRequest(
        name="admin-cross",
        pattern_type="freshness",
        pattern_config={"table": "triggers", "ts_column": "updated_at", "max_age_hours": 24},
        action_type="notify",
        action_config={},
        brand="Brand-Y",
    )
    with patch(
        "src.api.routes.sentinels.register_sentinel",
        return_value="s-admin-1",
    ):
        result = await create_sentinel(payload=payload, user=_admin_user())

    assert result.sentinel_id == "s-admin-1"
    assert result.brand == "Brand-Y"


@pytest.mark.asyncio
async def test_create_sentinel_operator_can_register_own_brand() -> None:
    """Operator with Brand-X grant CAN create Brand-X sentinel."""
    from src.api.routes.sentinels import SentinelCreateRequest, create_sentinel

    payload = SentinelCreateRequest(
        name="in-grant",
        pattern_type="freshness",
        pattern_config={"table": "triggers", "ts_column": "updated_at", "max_age_hours": 24},
        action_type="notify",
        action_config={},
        brand="Brand-X",
    )
    with patch(
        "src.api.routes.sentinels.register_sentinel",
        return_value="s-bx-1",
    ):
        result = await create_sentinel(payload=payload, user=_operator_brand_x())

    assert result.sentinel_id == "s-bx-1"
    assert result.brand == "Brand-X"


@pytest.mark.asyncio
async def test_get_sentinel_admin_can_read_any_brand() -> None:
    """Admin role retains cross-brand read on get_sentinel."""
    from src.api.routes.sentinels import get_sentinel

    row = {
        "sentinel_id": "s-1",
        "name": "kis-sentinel",
        "pattern_type": "freshness",
        "action_type": "notify",
        "brand": "Brand-Y",
        "region": None,
        "enabled": True,
        "last_fired_at": None,
        "fire_count": 0,
    }
    with patch(
        "src.api.routes.sentinels.get_supabase_client",
        return_value=_make_db_with_rows(row),
    ):
        result = await get_sentinel("s-1", user=_admin_user())

    assert result.sentinel_id == "s-1"
    assert result.brand == "Brand-Y"


# =============================================================================
# Finding 3 — cross-brand IDOR on PATCH (update) / DELETE
# =============================================================================
#
# update_sentinel / delete_sentinel used require_operator but never checked
# brand membership (unlike get_sentinel). Worse, update_sentinel ran the
# UPDATE before any read-back, so an Operator with only Brand-X could
# enable/disable or delete a Brand-Y sentinel. The mutation must be GATED by
# a brand-membership check that fetches the row FIRST and 404s when the row's
# brand is not in the caller's grant set — and the mutation must NOT run.


def _mutation_tracking_db(*rows: Dict[str, Any]) -> MagicMock:
    """Supabase mock that records whether update()/delete() were invoked.

    select()/update()/delete() all return a chainable object supporting
    .eq()/.select()/.limit()/.execute(); execute() yields the given rows.
    ``db.update_called`` / ``db.delete_called`` flip True when those verbs run.
    """
    db = MagicMock()
    db.update_called = False
    db.delete_called = False

    chain = MagicMock()
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.limit.return_value = chain
    chain.in_.return_value = chain
    chain.execute.return_value = MagicMock(data=list(rows))

    def _update(*_a: Any, **_k: Any) -> MagicMock:
        db.update_called = True
        return chain

    def _delete(*_a: Any, **_k: Any) -> MagicMock:
        db.delete_called = True
        return chain

    chain.update.side_effect = _update
    chain.delete.side_effect = _delete
    db.table.return_value = chain
    return db


@pytest.mark.asyncio
async def test_update_sentinel_rejects_out_of_grant_brand_before_mutating() -> None:
    """Operator with only Brand-X must NOT be able to disable a Brand-Y
    sentinel. Expect 404 (info-leak defense, same as get) AND no UPDATE run.
    """
    from fastapi import HTTPException

    from src.api.routes.sentinels import SentinelUpdateRequest, update_sentinel

    out_of_grant_row = {
        "sentinel_id": "s-1",
        "name": "by-sentinel",
        "pattern_type": "freshness",
        "action_type": "notify",
        "brand": "Brand-Y",
        "region": None,
        "enabled": True,
        "last_fired_at": None,
        "fire_count": 0,
    }
    db = _mutation_tracking_db(out_of_grant_row)
    with patch("src.api.routes.sentinels.get_supabase_client", return_value=db):
        with pytest.raises(HTTPException) as excinfo:
            await update_sentinel(
                "s-1",
                payload=SentinelUpdateRequest(enabled=False),
                user=_operator_brand_x(),
            )

    assert excinfo.value.status_code == 404, (
        f"out-of-grant update must 404 (got {excinfo.value.status_code})"
    )
    assert db.update_called is False, "UPDATE must NOT run for an out-of-grant sentinel"


@pytest.mark.asyncio
async def test_delete_sentinel_rejects_out_of_grant_brand_before_mutating() -> None:
    """Operator with only Brand-X must NOT delete a Brand-Y sentinel.
    Expect 404 AND no DELETE run.
    """
    from fastapi import HTTPException

    from src.api.routes.sentinels import delete_sentinel

    out_of_grant_row = {
        "sentinel_id": "s-1",
        "name": "by-sentinel",
        "pattern_type": "freshness",
        "action_type": "notify",
        "brand": "Brand-Y",
        "region": None,
        "enabled": True,
        "last_fired_at": None,
        "fire_count": 0,
    }
    db = _mutation_tracking_db(out_of_grant_row)
    with patch("src.api.routes.sentinels.get_supabase_client", return_value=db):
        with pytest.raises(HTTPException) as excinfo:
            await delete_sentinel("s-1", user=_operator_brand_x())

    assert excinfo.value.status_code == 404
    assert db.delete_called is False, "DELETE must NOT run for an out-of-grant sentinel"


@pytest.mark.asyncio
async def test_update_sentinel_missing_row_404_no_mutation() -> None:
    """Updating a non-existent sentinel returns 404 and runs no UPDATE."""
    from fastapi import HTTPException

    from src.api.routes.sentinels import SentinelUpdateRequest, update_sentinel

    db = _mutation_tracking_db()  # no rows
    with patch("src.api.routes.sentinels.get_supabase_client", return_value=db):
        with pytest.raises(HTTPException) as excinfo:
            await update_sentinel(
                "missing",
                payload=SentinelUpdateRequest(enabled=True),
                user=_operator_brand_x(),
            )

    assert excinfo.value.status_code == 404
    assert db.update_called is False


@pytest.mark.asyncio
async def test_delete_sentinel_missing_row_404_no_mutation() -> None:
    """Deleting a non-existent sentinel returns 404 and runs no DELETE."""
    from fastapi import HTTPException

    from src.api.routes.sentinels import delete_sentinel

    db = _mutation_tracking_db()  # no rows
    with patch("src.api.routes.sentinels.get_supabase_client", return_value=db):
        with pytest.raises(HTTPException) as excinfo:
            await delete_sentinel("missing", user=_operator_brand_x())

    assert excinfo.value.status_code == 404
    assert db.delete_called is False


@pytest.mark.asyncio
async def test_update_sentinel_in_grant_operator_can_disable() -> None:
    """Operator with Brand-X grant CAN disable a Brand-X sentinel (UPDATE runs)."""
    from src.api.routes.sentinels import SentinelUpdateRequest, update_sentinel

    in_grant_row = {
        "sentinel_id": "s-1",
        "name": "bx-sentinel",
        "pattern_type": "freshness",
        "action_type": "notify",
        "brand": "Brand-X",
        "region": None,
        "enabled": False,
        "last_fired_at": None,
        "fire_count": 0,
    }
    db = _mutation_tracking_db(in_grant_row)
    with patch("src.api.routes.sentinels.get_supabase_client", return_value=db):
        result = await update_sentinel(
            "s-1",
            payload=SentinelUpdateRequest(enabled=False),
            user=_operator_brand_x(),
        )

    assert db.update_called is True, "UPDATE must run for an in-grant sentinel"
    assert result.sentinel_id == "s-1"
    assert result.brand == "Brand-X"


@pytest.mark.asyncio
async def test_delete_sentinel_in_grant_operator_can_delete() -> None:
    """Operator with Brand-X grant CAN delete a Brand-X sentinel (DELETE runs)."""
    from src.api.routes.sentinels import delete_sentinel

    in_grant_row = {
        "sentinel_id": "s-1",
        "name": "bx-sentinel",
        "pattern_type": "freshness",
        "action_type": "notify",
        "brand": "Brand-X",
        "region": None,
        "enabled": True,
        "last_fired_at": None,
        "fire_count": 0,
    }
    db = _mutation_tracking_db(in_grant_row)
    with patch("src.api.routes.sentinels.get_supabase_client", return_value=db):
        await delete_sentinel("s-1", user=_operator_brand_x())

    assert db.delete_called is True, "DELETE must run for an in-grant sentinel"


@pytest.mark.asyncio
async def test_admin_can_update_any_brand_sentinel() -> None:
    """Admin retains cross-brand mutation on update_sentinel."""
    from src.api.routes.sentinels import SentinelUpdateRequest, update_sentinel

    row = {
        "sentinel_id": "s-1",
        "name": "by-sentinel",
        "pattern_type": "freshness",
        "action_type": "notify",
        "brand": "Brand-Y",
        "region": None,
        "enabled": False,
        "last_fired_at": None,
        "fire_count": 0,
    }
    db = _mutation_tracking_db(row)
    with patch("src.api.routes.sentinels.get_supabase_client", return_value=db):
        result = await update_sentinel(
            "s-1",
            payload=SentinelUpdateRequest(enabled=False),
            user=_admin_user(),
        )

    assert db.update_called is True
    assert result.brand == "Brand-Y"


# =============================================================================
# Finding 4 — sync supabase .execute() inside async handlers blocks the loop
# =============================================================================


@pytest.mark.asyncio
async def test_handlers_offload_blocking_execute_to_thread() -> None:
    """The async sentinel handlers must not call the sync supabase chain's
    blocking ``.execute()`` directly on the event loop; they must offload it
    via ``asyncio.to_thread``. Falsifiability: patch ``asyncio.to_thread`` and
    assert it was used to run the DB work for list/get/update/delete.
    """
    import asyncio

    from src.api.routes.sentinels import (
        SentinelUpdateRequest,
        delete_sentinel,
        get_sentinel,
        list_sentinels,
        update_sentinel,
    )

    row = {
        "sentinel_id": "s-1",
        "name": "bx-sentinel",
        "pattern_type": "freshness",
        "action_type": "notify",
        "brand": "Brand-X",
        "region": None,
        "enabled": True,
        "last_fired_at": None,
        "fire_count": 0,
    }

    real_to_thread = asyncio.to_thread

    async def _counting_to_thread(fn, *a, **k):
        _counting_to_thread.calls += 1
        return await real_to_thread(fn, *a, **k)

    _counting_to_thread.calls = 0

    # list_sentinels
    with (
        patch("src.api.routes.sentinels.get_supabase_client", return_value=_make_db_with_rows(row)),
        patch("src.api.routes.sentinels.asyncio.to_thread", side_effect=_counting_to_thread),
    ):
        await list_sentinels(brand="Brand-X", enabled_only=True, user=_operator_brand_x())
    assert _counting_to_thread.calls >= 1, "list_sentinels must offload .execute via to_thread"

    # get_sentinel
    _counting_to_thread.calls = 0
    with (
        patch("src.api.routes.sentinels.get_supabase_client", return_value=_make_db_with_rows(row)),
        patch("src.api.routes.sentinels.asyncio.to_thread", side_effect=_counting_to_thread),
    ):
        await get_sentinel("s-1", user=_operator_brand_x())
    assert _counting_to_thread.calls >= 1, "get_sentinel must offload .execute via to_thread"

    # update_sentinel (read + update)
    _counting_to_thread.calls = 0
    with (
        patch(
            "src.api.routes.sentinels.get_supabase_client",
            return_value=_mutation_tracking_db(row),
        ),
        patch("src.api.routes.sentinels.asyncio.to_thread", side_effect=_counting_to_thread),
    ):
        await update_sentinel(
            "s-1", payload=SentinelUpdateRequest(enabled=True), user=_operator_brand_x()
        )
    assert _counting_to_thread.calls >= 1, "update_sentinel must offload .execute via to_thread"

    # delete_sentinel (read + delete)
    _counting_to_thread.calls = 0
    with (
        patch(
            "src.api.routes.sentinels.get_supabase_client",
            return_value=_mutation_tracking_db(row),
        ),
        patch("src.api.routes.sentinels.asyncio.to_thread", side_effect=_counting_to_thread),
    ):
        await delete_sentinel("s-1", user=_operator_brand_x())
    assert _counting_to_thread.calls >= 1, "delete_sentinel must offload .execute via to_thread"
