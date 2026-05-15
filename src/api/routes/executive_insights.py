"""
Executive insights API — read crystallized cross-agent narratives.

Endpoints:
    GET /api/executive-insights         AUTH — list, brand-filtered
    GET /api/executive-insights/{id}    AUTH — get one (gated by JIT verifier middleware)
    POST /api/executive-insights/crystallize   OPERATOR — manually trigger
                                                          crystallization for a brand

JIT verification
----------------
The ``InsightVerifierMiddleware`` is configured to intercept this prefix
and replace the response with a 410 Gone if any provenance ancestor has
been overturned/invalidated. The route handlers themselves do NOT need
to call the verifier — the middleware does it after the handler returns.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies.auth import require_auth, require_operator
from src.memory.crystallization.crystallizer import crystallize_for_brand
from src.memory.services.factories import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/executive-insights", tags=["Executive Insights"])


class ExecutiveInsightResponse(BaseModel):
    insight_id: str
    title: str
    narrative: str
    brand: str
    region: Optional[str] = None
    kpi: Optional[str] = None
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None
    key_metrics: Dict[str, Any] = Field(default_factory=dict)
    recall: bool = False
    recall_reason: Optional[str] = None
    crystallized_at: datetime
    source_count: int = 0


class CrystallizeRequest(BaseModel):
    brand: str
    region: Optional[str] = None


class CrystallizeResponse(BaseModel):
    examined_groups: int
    insights_created: int
    edges_created: int


@router.get("", response_model=List[ExecutiveInsightResponse])
async def list_executive_insights(
    brand: Optional[str] = None,
    region: Optional[str] = None,
    include_recalled: bool = False,
    limit: int = 50,
    user: Dict[str, Any] = Depends(require_auth),
) -> List[ExecutiveInsightResponse]:
    """List executive insights. Brand filter is strongly recommended."""
    client = get_supabase_client()
    query = (
        client.table("executive_insights")
        .select(
            "insight_id, title, narrative, brand, region, kpi, "
            "time_window_start, time_window_end, key_metrics, "
            "recall, recall_reason, crystallized_at, source_count"
        )
        .order("crystallized_at", desc=True)
        .limit(limit)
    )
    if brand:
        query = query.eq("brand", brand)
    if region:
        query = query.eq("region", region)
    if not include_recalled:
        query = query.eq("recall", False)
    rows = (query.execute().data) or []
    return [_to_response(r) for r in rows]


@router.get("/{insight_id}", response_model=ExecutiveInsightResponse)
async def get_executive_insight(
    insight_id: str,
    user: Dict[str, Any] = Depends(require_auth),
) -> ExecutiveInsightResponse:
    """
    Get one insight. The InsightVerifierMiddleware will turn this into a
    410 Gone if any ancestor is overturned.
    """
    client = get_supabase_client()
    rows = (
        client.table("executive_insights")
        .select("*")
        .eq("insight_id", insight_id)
        .limit(1)
        .execute()
        .data
    ) or []
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="insight not found")
    return _to_response(rows[0])


@router.post(
    "/crystallize",
    response_model=CrystallizeResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_crystallization(
    payload: CrystallizeRequest,
    user: Dict[str, Any] = Depends(require_operator),
) -> CrystallizeResponse:
    """Manually trigger crystallization for a brand/region pair."""
    result = await crystallize_for_brand(
        brand=payload.brand,
        region=payload.region,
        crystallized_by_user_id=user.get("sub") or user.get("id"),
    )
    return CrystallizeResponse(
        examined_groups=result.examined_groups,
        insights_created=result.insights_created,
        edges_created=result.edges_created,
    )


def _to_response(row: Dict[str, Any]) -> ExecutiveInsightResponse:
    return ExecutiveInsightResponse(
        insight_id=str(row["insight_id"]),
        title=row.get("title", ""),
        narrative=row.get("narrative", ""),
        brand=row.get("brand", ""),
        region=row.get("region"),
        kpi=row.get("kpi"),
        time_window_start=row.get("time_window_start"),
        time_window_end=row.get("time_window_end"),
        key_metrics=row.get("key_metrics") or {},
        recall=bool(row.get("recall", False)),
        recall_reason=row.get("recall_reason"),
        crystallized_at=row.get("crystallized_at"),
        source_count=row.get("source_count") or 0,
    )
