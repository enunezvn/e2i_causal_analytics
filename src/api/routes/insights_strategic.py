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
    treatment_effect,
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


class AllocationMove(BaseModel):
    entity_id: str
    change_percentage: float | None = None
    change: float | None = None


class ResourceInsightRequest(BaseModel):
    optimization_summary: str = ""
    recommendations: list[str] = Field(default_factory=list)
    projected_lift_pct: float | None = None
    solver_status: str | None = None
    objective: str | None = None
    brand: str | None = None
    resource_type: str | None = None
    entity_count: int | None = None
    total_budget: float | None = None
    top_increases: list[AllocationMove] = Field(default_factory=list)
    top_decreases: list[AllocationMove] = Field(default_factory=list)
    synthetic: bool = True


class TreatmentEffectInsightRequest(BaseModel):
    cohort: str
    brand: str
    treatment_var: str
    outcome_var: str
    confounders: list[str] = Field(default_factory=list)
    ate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    n: int
    estimator: str | None = None


# ---- Endpoints ----------------------------------------------------------------
@router.post("/knowledge-graph", response_model=StrategicInsightResponse)
async def knowledge_graph_insight(
    req: KGInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Strategic interpretation of the curated knowledge graph for a brand (server-derived grounding)."""
    from src.memory.semantic_memory import get_semantic_memory

    brand = None if req.brand == "All" else req.brand

    def _load() -> dict[str, Any]:
        sm = get_semantic_memory()
        nodes = sm.list_nodes(limit=500, curated_only=req.curated_only)
        rels = sm.list_relationships(limit=500, curated_only=req.curated_only)
        if brand:  # scope edges to the brand when the property is present
            rels = [r for r in rels if (r.get("properties") or {}).get("brand") in (None, brand)]
        return knowledge_graph.build_grounding(
            req.brand,
            nodes,
            rels,
            node_count=sm.count_nodes(curated_only=req.curated_only),
            rel_count=len(rels),
        )

    try:
        g = await asyncio.to_thread(_load)
    except Exception as e:  # noqa: BLE001 — degrade honestly, never 500
        logger.warning("KG insight grounding unavailable: %s", e)
        return _finalize(
            {
                "insight": "The knowledge graph is currently unavailable, so no "
                "grounded interpretation can be produced right now.",
                "key_takeaways": [],
                "grounding": [],
                "is_fallback": True,
            },
            provenance="Curated knowledge graph (unavailable)",
        )
    key = cache_key("knowledge-graph", req.brand, {"n": g["node_summary"], "e": g["edge_summary"]})
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(knowledge_graph.generate_insight, g)
        await cache_set(key, payload)
    return _finalize(payload, provenance="Curated knowledge graph (server-derived)")


@router.post("/model-performance", response_model=StrategicInsightResponse)
async def model_performance_insight(
    req: ModelPerfInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Strategic health diagnosis of a deployed model from live performance metrics."""
    from src.services.performance_tracking import get_performance_tracker

    tracker = get_performance_tracker()
    try:
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
    except Exception as e:  # noqa: BLE001 — degrade honestly, never 500
        logger.warning("model-performance insight metrics unavailable: %s", e)
        return _finalize(
            {
                "insight": f"Performance metrics for {req.model_version} are currently "
                "unavailable, so no grounded interpretation can be produced.",
                "key_takeaways": [],
                "grounding": [],
                "is_fallback": True,
            },
            provenance="Live model-performance metrics (unavailable)",
        )
    key = cache_key(
        "model-performance",
        req.model_version,
        {"a": g["accuracy_summary"], "c": g["confusion_summary"]},
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(model_performance.generate_insight, g)
        await cache_set(key, payload)
    return _finalize(payload, provenance="Live model-performance metrics (server-derived)")


@router.post("/causal-discovery", response_model=StrategicInsightResponse)
async def causal_discovery_insight(
    req: CausalInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Portfolio-level interpretation of the discovered-effects leaderboard."""
    g = causal_discovery.build_grounding(
        req.brand, req.grain, [e.model_dump() for e in req.effects]
    )
    key = cache_key("causal-discovery", req.brand, {"t": g["effects_table"]})
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(causal_discovery.generate_insight, g)
        await cache_set(key, payload)
    return _finalize(payload, provenance="Agent-validated discovered effects")


@router.post("/treatment-effect", response_model=StrategicInsightResponse)
async def treatment_effect_insight(
    req: TreatmentEffectInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Interpret a single de-confounded (cohort, brand) treatment-effect estimate."""
    g = treatment_effect.build_grounding(
        req.cohort,
        req.brand,
        req.treatment_var,
        req.outcome_var,
        list(req.confounders),
        req.ate,
        req.ci_lower,
        req.ci_upper,
        req.p_value,
        req.n,
        req.estimator,
    )
    # Key on the derived grounding strings (which encode ate, CI, p, n, estimator,
    # treatment/outcome, confounders) so two estimates that differ only in CI — and
    # thus in the actionability verdict — never collide on {ate, n} alone.
    key = cache_key(
        "treatment-effect",
        f"{req.cohort}/{req.brand}",
        {"e": g["estimate"], "d": g["design"]},
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(treatment_effect.generate_insight, g)
        await cache_set(key, payload)
    return _finalize(
        payload,
        provenance="Interpretation of the DoWhy+EconML treatment-effect estimate",
    )


@router.post("/predictive-cohort", response_model=StrategicInsightResponse)
async def predictive_cohort_insight(
    req: PredictiveInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Targeting interpretation of a scored cohort (distribution, top targets, SHAP drivers)."""
    g = predictive_cohort.build_grounding(
        req.model_version,
        req.n_scored,
        req.mean_prob,
        [t.model_dump() for t in req.top_targets],
        [d.model_dump() for d in req.top_drivers],
    )
    key = cache_key(
        "predictive-cohort",
        req.model_version,
        {"d": g["distribution_summary"], "t": g["top_targets_summary"]},
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(predictive_cohort.generate_insight, g)
        await cache_set(key, payload)
    return _finalize(payload, provenance="Out-of-sample scored cohort + SHAP")


@router.post("/resource-optimization", response_model=StrategicInsightResponse)
async def resource_optimization_insight(
    req: ResourceInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Business interpretation (DSPy) of a resource-optimization run's allocation moves."""
    g = resource_optimization.build_grounding(
        objective=req.objective or "",
        brand=req.brand,
        resource_type=req.resource_type,
        solver_status=req.solver_status,
        entity_count=req.entity_count,
        total_budget=req.total_budget,
        projected_lift_pct=req.projected_lift_pct,
        top_increases=[m.model_dump() for m in req.top_increases],
        top_decreases=[m.model_dump() for m in req.top_decreases],
        synthetic=req.synthetic,
        optimization_summary=req.optimization_summary,
        recommendations=req.recommendations,
    )
    # Key on the derived grounding strings (scope + moves + outcome) so two runs
    # that differ in any move or in the projected lift never collide.
    key = cache_key(
        "resource-optimization",
        f"{req.brand or 'All'}/{req.objective or ''}",
        {"s": g["scope"], "m": g["moves"], "o": g["outcome"]},
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(resource_optimization.generate_insight, g)
        await cache_set(key, payload)
    return _finalize(payload, provenance="Resource optimizer solver result (LLM interpretation)")
