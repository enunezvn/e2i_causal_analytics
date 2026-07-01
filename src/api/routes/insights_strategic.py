"""Per-page strategic-insight endpoints. Each grounds an LLM interpretation in REAL
data with an honest deterministic fallback (no OPENAI_API_KEY -> is_fallback=True)."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.dependencies.auth import require_analyst
from src.insights import (
    causal_discovery,
    knowledge_graph,
    model_performance,
    predictive_cohort,
    resource_optimization,
)
from src.insights.common import cache_get, cache_key, cache_set

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/insights", tags=["Strategic Insights"])


class GroundingChip(BaseModel):
    label: str
    value: str


class StrategicInsightResponse(BaseModel):
    insight: str
    key_takeaways: list[str] = Field(default_factory=list)
    grounding: list[GroundingChip] = Field(default_factory=list)
    is_fallback: bool
    generated_at: str
    provenance: str


def _finalize(payload: dict[str, Any], provenance: str) -> StrategicInsightResponse:
    return StrategicInsightResponse(
        insight=payload["insight"],
        key_takeaways=payload.get("key_takeaways", []),
        grounding=[GroundingChip(**c) for c in payload.get("grounding", [])],
        is_fallback=payload["is_fallback"],
        generated_at=datetime.now(timezone.utc).isoformat(),
        provenance=provenance,
    )


# ---- Request models -----------------------------------------------------------
class KGInsightRequest(BaseModel):
    brand: str = "All"
    curated_only: bool = True


class ModelPerfInsightRequest(BaseModel):
    model_version: str


class CausalEffect(BaseModel):
    treatment: str
    outcome: str
    ate: float
    ate_ci_lower: float | None = None
    ate_ci_upper: float | None = None
    status: str | None = None
    selected_estimator: str | None = None


class CausalInsightRequest(BaseModel):
    brand: str
    grain: str
    effects: list[CausalEffect]


class TargetRow(BaseModel):
    entity_id: str
    probability: float


class DriverRow(BaseModel):
    feature: str
    importance: float


class PredictiveInsightRequest(BaseModel):
    model_version: str
    n_scored: int
    mean_prob: float
    top_targets: list[TargetRow] = Field(default_factory=list)
    top_drivers: list[DriverRow] = Field(default_factory=list)


class ResourceInsightRequest(BaseModel):
    optimization_summary: str = ""
    recommendations: list[str] = Field(default_factory=list)
    projected_lift_pct: float | None = None
    solver_status: str | None = None


# ---- Endpoints ----------------------------------------------------------------
@router.post("/knowledge-graph", response_model=StrategicInsightResponse)
async def knowledge_graph_insight(
    req: KGInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    from src.memory.semantic_memory import get_semantic_memory

    sm = get_semantic_memory()
    brand = None if req.brand == "All" else req.brand

    def _load() -> dict[str, Any]:
        nodes = sm.list_nodes(limit=500, curated_only=req.curated_only)
        rels = sm.list_relationships(limit=500, curated_only=req.curated_only)
        if brand:  # scope edges to the brand when the property is present
            rels = [
                r for r in rels
                if (r.get("properties") or {}).get("brand") in (None, brand)
            ]
        return knowledge_graph.build_grounding(
            req.brand, nodes, rels,
            node_count=sm.count_nodes(curated_only=req.curated_only),
            rel_count=len(rels),
        )

    g = await asyncio.to_thread(_load)
    key = cache_key("knowledge-graph", req.brand,
                    {"n": g["node_summary"], "e": g["edge_summary"]})
    cached = await cache_get(key)
    payload = cached or await asyncio.to_thread(knowledge_graph.generate_insight, g)
    if not cached:
        await cache_set(key, payload)
    return _finalize(payload, provenance="Curated knowledge graph (server-derived)")


@router.post("/model-performance", response_model=StrategicInsightResponse)
async def model_performance_insight(
    req: ModelPerfInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    from src.services.performance_tracking import get_performance_tracker

    tracker = get_performance_tracker()
    trend = await tracker.get_performance_trend(req.model_version, "accuracy")
    confusion = await tracker.get_confusion_matrix(req.model_version)
    roc = await tracker.get_roc_curve(req.model_version)
    alerts = await tracker.check_performance_alerts(req.model_version)
    g = model_performance.build_grounding(
        model_version=req.model_version,
        current_accuracy=float(getattr(trend, "current_value", 0.0) or 0.0),
        baseline_accuracy=float(getattr(trend, "baseline_value", 0.0) or 0.0),
        trend=str(getattr(trend, "trend", "stable")),
        confusion=confusion,
        auc=(float(roc["auc"]) if roc and roc.get("auc") is not None else None),
        alerts=alerts,
    )
    key = cache_key("model-performance", req.model_version,
                    {"a": g["accuracy_summary"], "c": g["confusion_summary"]})
    cached = await cache_get(key)
    payload = cached or await asyncio.to_thread(model_performance.generate_insight, g)
    if not cached:
        await cache_set(key, payload)
    return _finalize(payload, provenance="Live model-performance metrics (server-derived)")


@router.post("/causal-discovery", response_model=StrategicInsightResponse)
async def causal_discovery_insight(
    req: CausalInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    g = causal_discovery.build_grounding(
        req.brand, req.grain, [e.model_dump() for e in req.effects]
    )
    key = cache_key("causal-discovery", req.brand, {"t": g["effects_table"]})
    cached = await cache_get(key)
    payload = cached or await asyncio.to_thread(causal_discovery.generate_insight, g)
    if not cached:
        await cache_set(key, payload)
    return _finalize(payload, provenance="Agent-validated discovered effects")


@router.post("/predictive-cohort", response_model=StrategicInsightResponse)
async def predictive_cohort_insight(
    req: PredictiveInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    g = predictive_cohort.build_grounding(
        req.model_version, req.n_scored, req.mean_prob,
        [t.model_dump() for t in req.top_targets],
        [d.model_dump() for d in req.top_drivers],
    )
    key = cache_key("predictive-cohort", req.model_version,
                    {"d": g["distribution_summary"], "t": g["top_targets_summary"]})
    cached = await cache_get(key)
    payload = cached or await asyncio.to_thread(predictive_cohort.generate_insight, g)
    if not cached:
        await cache_set(key, payload)
    return _finalize(payload, provenance="Out-of-sample scored cohort + SHAP")


@router.post("/resource-optimization", response_model=StrategicInsightResponse)
async def resource_optimization_insight(
    req: ResourceInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    payload = resource_optimization.to_insight(
        req.optimization_summary, req.recommendations,
        req.projected_lift_pct, req.solver_status,
    )
    return _finalize(payload, provenance="Resource optimizer (existing agent output)")
