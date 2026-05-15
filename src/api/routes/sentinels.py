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
    """Register a sentinel. brand='all' requires ADMIN."""
    if payload.brand == "all":
        # Re-check role: require_operator passed but 'all' needs ADMIN.
        from src.api.dependencies.auth import UserRole, has_role

        if not has_role(user, UserRole.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="brand='all' sentinels require admin role",
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
    """List sentinels visible to the caller."""
    client = get_supabase_client()
    query = client.table("sentinels").select(
        "sentinel_id, name, pattern_type, action_type, brand, region, "
        "enabled, last_fired_at, fire_count"
    )
    if brand:
        query = query.eq("brand", brand)
    if enabled_only:
        query = query.eq("enabled", True)
    rows = (query.execute().data) or []
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
    rows = (
        client.table("sentinels").select("*").eq("sentinel_id", sentinel_id).limit(1).execute().data
    ) or []
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="sentinel not found")
    r = rows[0]
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
    """Enable/disable a sentinel."""
    if payload.enabled is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="no updatable fields")
    client = get_supabase_client()
    client.table("sentinels").update({"enabled": payload.enabled}).eq(
        "sentinel_id", sentinel_id
    ).execute()
    result: SentinelResponse = await get_sentinel(sentinel_id, user=user)
    return result


@router.delete("/{sentinel_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_sentinel(
    sentinel_id: str,
    user: Dict[str, Any] = Depends(require_operator),
) -> None:
    client = get_supabase_client()
    client.table("sentinels").delete().eq("sentinel_id", sentinel_id).execute()
