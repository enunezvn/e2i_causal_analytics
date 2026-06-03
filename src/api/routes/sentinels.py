"""
Sentinels API — register and inspect data-driven watchers.

Endpoints:
    POST   /api/sentinels         OPERATOR — register a new sentinel
    GET    /api/sentinels         AUTH     — list sentinels (brand-filtered by caller)
    GET    /api/sentinels/{id}    AUTH     — get one (brand-checked)
    PATCH  /api/sentinels/{id}    OPERATOR — enable/disable
    DELETE /api/sentinels/{id}    OPERATOR — delete

Brand enforcement
-----------------
- ``brand`` is required on registration.
- Registering ``brand='all'`` requires ADMIN role (cross-brand sentinels
  are dangerous because their actions cascade outside any single brand).
- List/Get filter by the caller's brand permissions (best-effort: today
  the auth layer does not yet expose per-brand grants, so we fall back to
  returning whatever the user can see by JWT role and let the row itself
  carry the brand label).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.api.dependencies.auth import require_auth, require_operator
from src.memory.sentinels.registry import register_sentinel
from src.memory.services.factories import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sentinels", tags=["Sentinels"])


class SentinelCreateRequest(BaseModel):
    """Payload to register a new sentinel."""

    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    pattern_type: str = Field(
        ...,
        description="threshold_breach | freshness | drift_score | new_causal_path",
    )
    pattern_config: Dict[str, Any]
    action_type: str = Field(..., description="invalidate | dispatch_agent | notify")
    action_config: Dict[str, Any] = Field(default_factory=dict)
    brand: str = Field(..., description="Brand scope; 'all' requires admin")
    region: Optional[str] = None

    @field_validator("brand")
    @classmethod
    def _no_empty_brand(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("brand must be a non-empty string")
        return v.strip()


class SentinelResponse(BaseModel):
    sentinel_id: str
    name: str
    pattern_type: str
    action_type: str
    brand: str
    region: Optional[str] = None
    enabled: bool
    last_fired_at: Optional[str] = None
    fire_count: int = 0


@router.post(
    "",
    response_model=SentinelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new sentinel",
)
async def create_sentinel(
    payload: SentinelCreateRequest,
    user: Dict[str, Any] = Depends(require_operator),
) -> SentinelResponse:
    """Register a sentinel. brand='all' requires ADMIN; other brands require
    that the caller has a grant for that brand (or is admin)."""
    from src.api.dependencies.auth import UserRole, get_user_brands, has_role

    is_admin = has_role(user, UserRole.ADMIN)
    if payload.brand == "all":
        # 'all' is cross-brand and stronger than any individual grant.
        if not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="brand='all' sentinels require admin role",
            )
    else:
        # Non-'all' brand registration must match the caller's brand grants.
        # Codex-rescue iter-0 HIGH-3: previously any Operator could register
        # for any brand; list/get enforced membership but create did not.
        allowed_brands = set(get_user_brands(user))
        if not is_admin and "all" not in allowed_brands and payload.brand not in allowed_brands:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"no grant for brand={payload.brand!r}",
            )

    try:
        sentinel_id = await register_sentinel(
            name=payload.name,
            description=payload.description,
            pattern_type=payload.pattern_type,
            pattern_config=payload.pattern_config,
            action_type=payload.action_type,
            action_config=payload.action_config,
            brand=payload.brand,
            region=payload.region,
            created_by_user_id=user.get("sub") or user.get("id"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception("sentinel registration failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return SentinelResponse(
        sentinel_id=str(sentinel_id),
        name=payload.name,
        pattern_type=payload.pattern_type,
        action_type=payload.action_type,
        brand=payload.brand,
        region=payload.region,
        enabled=True,
        fire_count=0,
    )


@router.get("", response_model=List[SentinelResponse], summary="List sentinels")
async def list_sentinels(
    brand: Optional[str] = None,
    enabled_only: bool = True,
    user: Dict[str, Any] = Depends(require_auth),
) -> List[SentinelResponse]:
    """List sentinels visible to the caller.

    Brand enforcement (until full RLS lands):

    * Admin role (or brand grant ``'all'``) can query any brand.
    * Otherwise the caller's ``brands`` claim is the allowed set.
    * ``?brand=X`` with X not in the allowed set returns an empty list
      (defensive — doesn't leak whether brand X exists).
    * Omitting ``?brand`` returns sentinels across the caller's allowed
      brands only.
    """
    from src.api.dependencies.auth import UserRole, get_user_brands, has_role

    allowed_brands = set(get_user_brands(user))
    is_admin = has_role(user, UserRole.ADMIN) or "all" in allowed_brands

    if brand is not None and not is_admin and brand not in allowed_brands:
        # Defensive empty list — avoid leaking existence of other-brand rows.
        return []

    client = get_supabase_client()
    query = client.table("sentinels").select(
        "sentinel_id, name, pattern_type, action_type, brand, region, "
        "enabled, last_fired_at, fire_count"
    )
    if brand:
        query = query.eq("brand", brand)
    elif not is_admin and allowed_brands:
        # No brand param + non-admin: restrict to caller's allowed brands.
        query = query.in_("brand", list(allowed_brands))
    elif not is_admin and not allowed_brands:
        # No brand param + non-admin + no grants: nothing visible.
        return []
    if enabled_only:
        query = query.eq("enabled", True)
    # Finding 4: the supabase client is sync — offload the blocking .execute()
    # to a worker thread so it doesn't stall the event loop.
    result = await asyncio.to_thread(query.execute)
    rows = (result.data) or []
    return [
        SentinelResponse(
            sentinel_id=str(r["sentinel_id"]),
            name=r["name"],
            pattern_type=r["pattern_type"],
            action_type=r["action_type"],
            brand=r["brand"],
            region=r.get("region"),
            enabled=bool(r.get("enabled", True)),
            last_fired_at=r.get("last_fired_at"),
            fire_count=r.get("fire_count") or 0,
        )
        for r in rows
    ]


@router.get("/{sentinel_id}", response_model=SentinelResponse)
async def get_sentinel(
    sentinel_id: str,
    user: Dict[str, Any] = Depends(require_auth),
) -> SentinelResponse:
    client = get_supabase_client()
    # Finding 4: offload the sync supabase .execute() off the event loop.
    query = client.table("sentinels").select("*").eq("sentinel_id", sentinel_id).limit(1)
    result = await asyncio.to_thread(query.execute)
    rows = (result.data) or []
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="sentinel not found")
    r = rows[0]

    # Brand-membership check: return 404 (not 403) for out-of-grant rows so
    # the response doesn't leak existence to unauthorized callers.
    from src.api.dependencies.auth import UserRole, get_user_brands, has_role

    allowed_brands = set(get_user_brands(user))
    is_admin = has_role(user, UserRole.ADMIN) or "all" in allowed_brands
    if not is_admin and r.get("brand") not in allowed_brands:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="sentinel not found")

    return SentinelResponse(
        sentinel_id=str(r["sentinel_id"]),
        name=r["name"],
        pattern_type=r["pattern_type"],
        action_type=r["action_type"],
        brand=r["brand"],
        region=r.get("region"),
        enabled=bool(r.get("enabled", True)),
        last_fired_at=r.get("last_fired_at"),
        fire_count=r.get("fire_count") or 0,
    )


class SentinelUpdateRequest(BaseModel):
    enabled: Optional[bool] = None


@router.patch("/{sentinel_id}", response_model=SentinelResponse)
async def update_sentinel(
    sentinel_id: str,
    payload: SentinelUpdateRequest,
    user: Dict[str, Any] = Depends(require_operator),
) -> SentinelResponse:
    """Enable/disable a sentinel.

    Finding 3 (cross-brand IDOR): brand membership is enforced BEFORE the
    mutation. Previously the UPDATE ran first and only the read-back would
    404, so any Operator could enable/disable another tenant's sentinel.
    ``get_sentinel`` fetches the row and 404s when its brand is not in the
    caller's grant set (admins bypass), so we call it first as the
    authorization gate — and only mutate once it succeeds.
    """
    if payload.enabled is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="no updatable fields")

    # Authorization gate: 404s for missing OR out-of-grant rows (info-leak
    # defense), and runs BEFORE any mutation.
    await get_sentinel(sentinel_id, user=user)

    client = get_supabase_client()
    # Finding 4: offload the sync supabase .execute() off the event loop.
    update_query = (
        client.table("sentinels")
        .update({"enabled": payload.enabled})
        .eq("sentinel_id", sentinel_id)
    )
    await asyncio.to_thread(update_query.execute)
    result: SentinelResponse = await get_sentinel(sentinel_id, user=user)
    return result


@router.delete("/{sentinel_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_sentinel(
    sentinel_id: str,
    user: Dict[str, Any] = Depends(require_operator),
) -> None:
    """Delete a sentinel.

    Finding 3 (cross-brand IDOR): like update, brand membership is enforced
    BEFORE the DELETE. ``get_sentinel`` is the authorization gate — it 404s
    for missing or out-of-grant rows (admins bypass) — and only then do we
    run the delete.
    """
    # Authorization gate: 404s for missing OR out-of-grant rows, BEFORE delete.
    await get_sentinel(sentinel_id, user=user)

    client = get_supabase_client()
    # Finding 4: offload the sync supabase .execute() off the event loop.
    delete_query = client.table("sentinels").delete().eq("sentinel_id", sentinel_id)
    await asyncio.to_thread(delete_query.execute)
