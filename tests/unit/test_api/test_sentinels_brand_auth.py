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
