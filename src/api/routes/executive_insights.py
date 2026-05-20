"""
Executive insights API — read crystallized cross-agent narratives.

Endpoints:
    GET /api/executive-insights         AUTH — list, brand-filtered
    GET /api/executive-insights/{id}    AUTH — get one (gated by JIT verifier middleware)
    GET /api/executive-insights/portfolio-summary  AUTH — per-brand aggregation
    POST /api/executive-insights/crystallize   OPERATOR — manually trigger
                                                          crystallization for a brand

JIT verification
----------------
The ``InsightVerifierMiddleware`` is configured to intercept this prefix
and replace the response with a 410 Gone if any provenance ancestor has
been overturned/invalidated. The route handlers themselves do NOT need
to call the verifier — the middleware does it after the handler returns.

Schema completion (#376)
------------------------
The 15 analytical/lineage fields below were added in lock-step with
migration ``database/memory/025_crystaldigest_schema_completion.sql``.
Per Decision 3 = KEEP BINARY (adopted 2026-05-19), the
``staleness_score`` field is intentionally omitted; staleness remains
boolean via ``invalidated_at IS NULL``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies.auth import require_auth, require_operator
from src.memory.crystallization.crystallizer import crystallize_for_brand
from src.memory.services.factories import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/executive-insights", tags=["Executive Insights"])


# Columns that participate in the extended response. Pulled from the
# row in select(). Kept as a module constant so additions to the
# ExecutiveInsightResponse stay in sync with the SELECT.
EXTENDED_SELECT_COLUMNS = (
    # original 13
    "insight_id, title, narrative, brand, region, kpi, "
    "time_window_start, time_window_end, key_metrics, "
    "recall, recall_reason, crystallized_at, source_count, "
    # analytical (8)
    "effect_size, effect_ci_lower, effect_ci_upper, effect_direction, "
    "cohort_size, confounders_controlled, sensitivity_checks_passed, "
    "sensitivity_checks_failed, "
    # narrative prose (2)
    "limitations, recommended_next_analysis, "
    # lineage (5)
    "provenance_chain_id, provenance_depth, consolidation_tier, "
    "replication_count, data_version"
)


class ExecutiveInsightResponse(BaseModel):
    """Pydantic response for crystallized executive insights.

    Per issue #376 + plan §"DECISIONS ADOPTED — 2026-05-19"
    Decision 2 = HYBRID (sub-decision 2a numeric effect_size) and
    Decision 3 = KEEP BINARY (no staleness_score field).

    Three categories of new fields:
      * Analytical (8): ``effect_size``, ``effect_ci_lower``,
        ``effect_ci_upper``, ``effect_direction``, ``cohort_size``,
        ``confounders_controlled``, ``sensitivity_checks_passed``,
        ``sensitivity_checks_failed``.
      * Narrative-prose (2): ``limitations``,
        ``recommended_next_analysis``. LLM-generated when the feature
        flag is on; deterministic heuristic otherwise.
      * Lineage (5): ``provenance_chain_id``, ``provenance_depth``,
        ``consolidation_tier``, ``replication_count``, ``data_version``.
    """

    # --- Original 13 fields (PR #250) ---
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

    # --- Analytical (#376 §A.1-8) ---
    effect_size: Optional[float] = None
    effect_ci_lower: Optional[float] = None
    effect_ci_upper: Optional[float] = None
    effect_direction: Optional[str] = None  # 'positive' | 'negative' | 'null'
    cohort_size: Optional[int] = None
    confounders_controlled: List[str] = Field(default_factory=list)
    sensitivity_checks_passed: List[str] = Field(default_factory=list)
    sensitivity_checks_failed: List[str] = Field(default_factory=list)

    # --- Narrative-prose (#376 §A.9-10; Decision 2 LLM path) ---
    limitations: Optional[str] = None
    recommended_next_analysis: Optional[str] = None

    # --- Lineage (#376 §A.11-15) ---
    provenance_chain_id: Optional[str] = None
    provenance_depth: Optional[int] = None
    consolidation_tier: Optional[str] = None  # working|episodic|semantic|procedural
    replication_count: Optional[int] = None
    data_version: Optional[str] = None

    def to_dashboard_payload(self) -> Dict[str, Any]:
        """Serialize for the CopilotKit dashboard payload.

        Datetimes flatten to ISO strings so the frontend can ingest
        without an extra coerce step. The contract intentionally mirrors
        the row schema so a column added here is exposed to the
        dashboard automatically.
        """
        return self.model_dump(mode="json")


class CrystallizeRequest(BaseModel):
    brand: str
    region: Optional[str] = None


class CrystallizeResponse(BaseModel):
    examined_groups: int
    insights_created: int
    edges_created: int


class PortfolioBrandSummary(BaseModel):
    """Per-brand aggregation for ``/portfolio-summary``."""

    brand: str
    insight_count: int = 0
    latest_crystallized_at: Optional[datetime] = None
    # ``average_effect_size`` aggregates across crystals where
    # ``effect_size IS NOT NULL`` (only numeric ATEs contribute; rows
    # with a NULL effect_size — e.g. legacy pre-#376 rows — are
    # excluded from the mean).
    average_effect_size: Optional[float] = None
    # Number of crystals contributing to the mean (denominator).
    effect_size_sample_count: int = 0


class PortfolioSummaryResponse(BaseModel):
    """Aggregated portfolio summary across all brands.

    Aggregation:
      * count of insights per brand
      * latest crystallized_at per brand
      * average effect_size per brand (excluding NULL effect_size rows)
    """

    by_brand: List[PortfolioBrandSummary] = Field(default_factory=list)
    total_brands: int = 0
    total_insights: int = 0


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
        .select(EXTENDED_SELECT_COLUMNS)
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


@router.get("/portfolio-summary", response_model=PortfolioSummaryResponse)
async def get_portfolio_summary(
    user: Dict[str, Any] = Depends(require_auth),
) -> PortfolioSummaryResponse:
    """Per-brand aggregation of crystallized insights (#376 §D).

    Aggregates across all non-recalled, non-invalidated crystals.
    Returns counts, latest timestamps, and an average effect_size per
    brand (over rows where ``effect_size IS NOT NULL``).

    Caller-side filtering by tier or KPI is out of scope here — this
    endpoint is the top-of-funnel portfolio view; drill-down uses the
    list endpoint with brand+kpi query params.

    NOTE: this endpoint MUST be declared BEFORE ``/{insight_id}`` in
    route order, otherwise FastAPI matches it as an insight_id.
    """
    client = get_supabase_client()
    rows = (
        client.table("executive_insights")
        .select("brand, crystallized_at, effect_size")
        .eq("recall", False)
        .is_("invalidated_at", "null")
        .execute()
        .data
    ) or []

    by_brand: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "latest": None,
            "effect_sum": 0.0,
            "effect_n": 0,
        }
    )
    for row in rows:
        brand = row.get("brand")
        if not brand:
            continue
        bucket = by_brand[brand]
        bucket["count"] = int(bucket["count"]) + 1
        ts = row.get("crystallized_at")
        if ts is not None and (bucket["latest"] is None or ts > bucket["latest"]):
            bucket["latest"] = ts
        effect = row.get("effect_size")
        if effect is not None:
            try:
                bucket["effect_sum"] = float(bucket["effect_sum"]) + float(effect)
                bucket["effect_n"] = int(bucket["effect_n"]) + 1
            except (TypeError, ValueError):
                # Defensive: skip rows whose effect_size is non-numeric
                # (should never happen post-#376 migration, but if a
                # legacy row carries a sentinel like 'NaN', don't crash).
                continue

    summaries: List[PortfolioBrandSummary] = []
    for brand_name in sorted(by_brand.keys()):
        b = by_brand[brand_name]
        avg = (float(b["effect_sum"]) / int(b["effect_n"])) if int(b["effect_n"]) > 0 else None
        summaries.append(
            PortfolioBrandSummary(
                brand=brand_name,
                insight_count=int(b["count"]),
                latest_crystallized_at=b["latest"],
                average_effect_size=avg,
                effect_size_sample_count=int(b["effect_n"]),
            )
        )

    return PortfolioSummaryResponse(
        by_brand=summaries,
        total_brands=len(summaries),
        total_insights=sum(s.insight_count for s in summaries),
    )


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


def _coerce_list(value: Any) -> List[str]:
    """Defensive coercion of TEXT[] columns to a Python list of strs.

    Supabase returns lists as-is, but a legacy row that wrote NULL or
    a JSON string would fail Pydantic validation on a typed list. This
    helper normalizes the shape so the route never 500s on an
    unexpected row.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        # Best-effort: treat single string as one-element list
        return [value]
    return []


def _to_response(row: Dict[str, Any]) -> ExecutiveInsightResponse:
    return ExecutiveInsightResponse(
        # --- original 13 ---
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
        crystallized_at=row["crystallized_at"],  # NOT NULL DEFAULT now() in schema
        source_count=row.get("source_count") or 0,
        # --- analytical (8) ---
        effect_size=row.get("effect_size"),
        effect_ci_lower=row.get("effect_ci_lower"),
        effect_ci_upper=row.get("effect_ci_upper"),
        effect_direction=row.get("effect_direction"),
        cohort_size=row.get("cohort_size"),
        confounders_controlled=_coerce_list(row.get("confounders_controlled")),
        sensitivity_checks_passed=_coerce_list(row.get("sensitivity_checks_passed")),
        sensitivity_checks_failed=_coerce_list(row.get("sensitivity_checks_failed")),
        # --- narrative prose (2) ---
        limitations=row.get("limitations"),
        recommended_next_analysis=row.get("recommended_next_analysis"),
        # --- lineage (5) ---
        provenance_chain_id=row.get("provenance_chain_id"),
        provenance_depth=row.get("provenance_depth"),
        consolidation_tier=row.get("consolidation_tier"),
        replication_count=row.get("replication_count"),
        data_version=row.get("data_version"),
    )
