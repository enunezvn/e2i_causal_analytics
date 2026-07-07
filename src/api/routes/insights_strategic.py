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
    executive_brief,
    feedback_learning,
    hte,
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
    # Actual optimized spend (sum of optimized allocations). maximize_roi can
    # intentionally deploy less than the budget; the narrative must not claim
    # the full budget is "under optimization" when it isn't.
    total_spend: float | None = None
    top_increases: list[AllocationMove] = Field(default_factory=list)
    top_decreases: list[AllocationMove] = Field(default_factory=list)
    synthetic: bool = True


class ExecutiveBriefInsightRequest(BaseModel):
    # Brand only: the figures are derived SERVER-SIDE from the latest completed
    # gap analysis (same read path as GET /gaps/opportunities). Accepting
    # caller-posted figures would let any authenticated caller mint a
    # grounded-looking brief from arbitrary numbers under gap-analyzer
    # provenance (codex PR-5 round 3).
    brand: str


class HTEInsightRequest(BaseModel):
    # analysis_id only: the figures are derived SERVER-SIDE from the persisted
    # segment-analysis record. Accepting caller-posted figures would let any
    # authenticated caller mint a grounded-looking insight from arbitrary
    # numbers under segment-analysis provenance (same trust boundary as
    # /insights/executive-brief).
    analysis_id: str


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
    # Registry context (2026-07-07): commercial chains the grain-scope guard
    # keeps OUT of estimation runs — cited as additional modeled coverage,
    # digit-free. fetch fails soft to [] — a registry hiccup never blocks.
    from src.insights.causal_context import fetch_commercial_drivers

    drivers = await fetch_commercial_drivers(
        req.brand, outcomes=("TRx", "NRx", "market share", "ROI")
    )
    g = causal_discovery.build_grounding(
        req.brand, req.grain, [e.model_dump() for e in req.effects], causal_drivers=drivers
    )
    key = cache_key(
        "causal-discovery", req.brand, {"t": g["effects_table"], "r": g["registry_context"]}
    )
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
    # Registry context (2026-07-07): curated chains token-matched to the
    # estimated pair — rendered digit-free and kept SEPARATE from the estimate
    # narrative (contrast, never corroboration). fetch fails soft to [].
    from src.insights.causal_context import fetch_commercial_drivers

    drivers = await fetch_commercial_drivers(
        req.brand, outcomes=(req.outcome_var, req.treatment_var)
    )
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
        causal_drivers=drivers,
    )
    # Key on the derived grounding strings (which encode ate, CI, p, n, estimator,
    # treatment/outcome, confounders) so two estimates that differ only in CI — and
    # thus in the actionability verdict — never collide on {ate, n} alone.
    key = cache_key(
        "treatment-effect",
        f"{req.cohort}/{req.brand}",
        {"e": g["estimate"], "d": g["design"], "r": g["registry_context"]},
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
    # Registry context (2026-07-07): outcome-matched chains derived from the
    # gold-standard model name; unrecognizable names honestly fetch nothing.
    from src.insights.causal_context import fetch_commercial_drivers

    reg_brand, reg_terms = predictive_cohort.outcome_terms_for_model(req.model_version)
    drivers = await fetch_commercial_drivers(reg_brand, outcomes=reg_terms) if reg_terms else []
    g = predictive_cohort.build_grounding(
        req.model_version,
        req.n_scored,
        req.mean_prob,
        [t.model_dump() for t in req.top_targets],
        [d.model_dump() for d in req.top_drivers],
        causal_drivers=drivers,
    )
    key = cache_key(
        "predictive-cohort",
        req.model_version,
        {
            "d": g["distribution_summary"],
            "t": g["top_targets_summary"],
            "r": g["registry_context"],
        },
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(predictive_cohort.generate_insight, g)
        await cache_set(key, payload)
    return _finalize(payload, provenance="Out-of-sample scored cohort + SHAP")


@router.post("/executive-brief", response_model=StrategicInsightResponse)
async def executive_brief_insight(
    req: ExecutiveBriefInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Executive distillation (DSPy) of the brand's latest gap-analysis figures (server-derived)."""
    # Same read path the dashboard's opportunities feed uses — the trust
    # boundary is the API, so the grounding must come from the server's own
    # gap-analysis data, never from caller-posted figures.
    from src.api.routes.gaps import list_opportunities

    try:
        feed = await list_opportunities(brand=req.brand, min_roi=None, difficulty=None, limit=5)
    except Exception as e:  # noqa: BLE001 — degrade honestly, never 500
        logger.warning("executive-brief opportunities feed unavailable: %s", e)
        return _finalize(
            {
                "insight": f"The gap-analysis figures for {req.brand} are currently "
                "unavailable, so no grounded executive brief can be produced — this "
                "is a data-source failure, not an empty portfolio.",
                "key_takeaways": [],
                "grounding": [],
                "is_fallback": True,
            },
            provenance="Gap-analyzer ROI opportunities (unavailable)",
        )
    # Causal levers (commercial grain, 2026-07-07): server-derived through the
    # chatbot's exact registry read path + provenance gate; names-only,
    # digit-free (the brief's placeholder guard rejects any digit the LM
    # emits). fetch fails soft to [] — a registry hiccup never blocks a brief.
    from src.insights.causal_context import fetch_commercial_drivers, format_driver_names

    levers = format_driver_names(
        await fetch_commercial_drivers(req.brand, outcomes=("TRx", "NRx", "market share", "ROI"))
    )
    g = executive_brief.build_grounding(
        brand=req.brand,
        total_addressable_value=feed.total_addressable_value,
        quick_wins_count=feed.quick_wins_count,
        steady_plays_count=feed.steady_plays_count,
        strategic_bets_count=feed.strategic_bets_count,
        suppressed_count=feed.suppressed_count or 0,
        opportunities=[
            {
                "rank": o.rank,
                "recommended_action": o.recommended_action,
                "expected_roi": o.roi_estimate.expected_roi,
                "revenue_impact": o.roi_estimate.estimated_revenue_impact,
                "gap_metric": o.gap.metric,
                "gap_percentage": o.gap.gap_percentage,
                "segment_value": o.gap.segment_value,
                "implementation_difficulty": o.implementation_difficulty.value,
            }
            for o in feed.opportunities
        ],
        causal_drivers=levers,
    )
    # Key on the derived grounding strings (scope + opportunities + caveats +
    # causal levers) so two portfolios that differ in any figure never collide.
    key = cache_key(
        "executive-brief",
        req.brand,
        {
            "s": g["scope"],
            "o": g["opportunities"],
            "c": g["caveats"],
            "cc": g["lm_causal_context"],
        },
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(executive_brief.generate_insight, g)
        # A fallback marks a transient state (LM outage or a rejected sample):
        # cache it briefly so the page self-heals on the next visit instead of
        # pinning the factual summary for the full hour.
        await cache_set(key, payload, ttl_seconds=300 if payload.get("is_fallback") else 3600)
    return _finalize(payload, provenance="Gap-analyzer ROI opportunities (server-derived)")


@router.post("/hte", response_model=StrategicInsightResponse)
async def hte_insight(
    req: HTEInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Strategic interpretation (DSPy) of a persisted segment-level CATE run (server-derived)."""
    # The grounding comes from the server's own persisted analysis record —
    # the caller supplies only the analysis_id (trust boundary is the API).
    from src.api.routes.segments import get_persisted_analysis

    try:
        record = await get_persisted_analysis(req.analysis_id)
    except Exception as e:  # noqa: BLE001 — degrade honestly, never 500
        logger.warning("HTE insight: persisted-analysis read failed: %s", e)
        record = None
    if record is None:
        return _finalize(
            {
                "insight": (
                    "The referenced segment analysis was not found — persisted runs "
                    "are kept for 7 days, so it may have expired. Re-run the "
                    "heterogeneous-treatment-effects analysis to generate an insight."
                ),
                "key_takeaways": [],
                "grounding": [],
                "is_fallback": True,
            },
            provenance="Persisted segment-level CATE analysis (unavailable)",
        )
    if record.status.value != "completed":
        return _finalize(
            {
                "insight": (
                    f"The referenced segment analysis has status "
                    f"'{record.status.value}', so its figures cannot ground a "
                    "strategic interpretation. Re-run the analysis to completion."
                ),
                "key_takeaways": [],
                "grounding": [],
                "is_fallback": True,
            },
            provenance="Persisted segment-level CATE analysis (incomplete run)",
        )
    record_dict = record.model_dump()
    # Commercial levers (2026-07-07): brand-scoped registry chains around the
    # analyzed outcome + volume KPIs — digit-free, so the fail-closed numeric
    # guard is untouched. fetch fails soft to [].
    from src.insights.causal_context import fetch_commercial_drivers

    lever_terms = tuple(t for t in (record_dict.get("outcome_var"), "TRx", "NRx") if t)
    drivers = await fetch_commercial_drivers(record_dict.get("brand"), outcomes=lever_terms)
    g = hte.build_grounding(record_dict, causal_drivers=drivers)
    # Key on the derived grounding strings so two runs differing in any figure
    # never collide (the analysis_id alone would pin a stale re-run).
    key = cache_key(
        "hte",
        req.analysis_id,
        {
            "e": g["effect_summary"],
            "s": g["segments"],
            "t": g["targeting"],
            "cc": g["commercial_context"],
        },
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(hte.generate_insight, g)
        # Same fallback-TTL policy as /executive-brief: a fallback marks a
        # transient state (LM outage or a guard-rejected sample) — cache it
        # briefly so the page self-heals instead of pinning it for the hour.
        await cache_set(key, payload, ttl_seconds=300 if payload.get("is_fallback") else 3600)
    return _finalize(payload, provenance="Segment-level CATE analysis (server-derived)")


@router.post("/resource-optimization", response_model=StrategicInsightResponse)
async def resource_optimization_insight(
    req: ResourceInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Business interpretation (DSPy) of a resource-optimization run's allocation moves."""
    # Causal drivers (commercial grain, 2026-07-07): server-derived through the
    # chatbot's exact registry read path + provenance gate, so the insight can
    # ground the WHY behind the moves. fetch fails soft to [] — a registry
    # hiccup never blocks the insight.
    from src.insights.causal_context import fetch_commercial_drivers

    causal_drivers = await fetch_commercial_drivers(req.brand, outcomes=("TRx", "ROI"))
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
        total_spend=req.total_spend,
        causal_drivers=causal_drivers,
    )
    # Key on the derived grounding strings (scope + moves + outcome + causal
    # context) so two runs that differ in any input never collide.
    key = cache_key(
        "resource-optimization",
        f"{req.brand or 'All'}/{req.objective or ''}",
        {"s": g["scope"], "m": g["moves"], "o": g["outcome"], "cc": g["causal_context"]},
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(resource_optimization.generate_insight, g)
        await cache_set(key, payload)
    return _finalize(payload, provenance="Resource optimizer solver result (LLM interpretation)")


class FeedbackLearningInsightRequest(BaseModel):
    """All grounding is server-derived (persisted cycles/patterns/updates + real
    feedback inflow); the caller only picks the inflow window."""

    days: int = Field(default=7, ge=1, le=30)


@router.post("/feedback-learning", response_model=StrategicInsightResponse)
async def feedback_learning_insight(
    req: FeedbackLearningInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Strategic interpretation of the Tier-5 feedback-learning loop, grounded in
    persisted learning cycles, detected patterns, knowledge updates, and the real
    feedback inflow (chat thumbs + cognitive reward signals)."""
    from datetime import timedelta

    from src.api.repositories.feedback_repository import FeedbackRepository
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.chatbot_feedback import get_chatbot_feedback_repository
    from src.repositories.learning_signals_feedback import (
        get_learning_signals_feedback_store,
    )

    try:
        repo = FeedbackRepository()
        batches = await repo.count_recent_and_last()
        patterns = await repo.list_patterns()
        updates = await repo.list_updates()

        now = datetime.now(timezone.utc)
        cycles_24h = sum(1 for b in batches if (now - b.timestamp).total_seconds() < 86400)
        last_cycle_at = max((b.timestamp for b in batches), default=None)

        client = await get_async_supabase_client()
        thumbs = await get_chatbot_feedback_repository(supabase_client=client).get_feedback_summary(
            days=req.days
        )
        window_start = (now - timedelta(days=req.days)).isoformat()
        signals = await get_learning_signals_feedback_store(supabase_client=client).get_feedback(
            start_time=window_start
        )

        # metadata.reward is the RAW 0..1 workflow reward; the item's "rating"
        # is that reward remapped to the analyzer's 1-5 scale — insight math
        # stays on the raw scale.
        rewards = [float(s["metadata"]["reward"]) for s in signals]
        avg_reward = sum(rewards) / len(rewards) if rewards else None
        per_agent: dict[str, list[float]] = {}
        for s in signals:
            per_agent.setdefault(str(s["agent"]), []).append(float(s["metadata"]["reward"]))
        low_reward_agents = sorted(
            ((agent, sum(v) / len(v)) for agent, v in per_agent.items() if sum(v) / len(v) < 0.5),
            key=lambda t: t[1],
        )

        g = feedback_learning.build_grounding(
            cycles_24h=cycles_24h,
            last_cycle_at=(last_cycle_at.isoformat() if last_cycle_at else None),
            thumbs_7d=int(thumbs.get("total_feedback", 0) or 0),
            signals_7d=len(signals),
            avg_reward_7d=avg_reward,
            patterns=[p.model_dump(mode="json") for p in patterns],
            updates=[u.model_dump(mode="json") for u in updates],
            low_reward_agents=low_reward_agents,
        )
    except Exception as e:  # noqa: BLE001 — degrade honestly, never 500
        logger.warning("feedback-learning insight data unavailable: %s", e)
        return _finalize(
            {
                "insight": "Feedback-learning loop data is currently unavailable, so no "
                "grounded interpretation can be produced.",
                "key_takeaways": [],
                "grounding": [],
                "is_fallback": True,
            },
            provenance="Live feedback-learning loop data (unavailable)",
        )
    key = cache_key(
        "feedback-learning",
        str(req.days),
        {
            "a": g["activity_summary"],
            "p": g["patterns_summary"],
            "u": g["updates_summary"],
            # signal_quality feeds the LM too — omitting it served stale
            # low-reward-agent narratives when only the reward stats changed
            "q": g["signal_quality_summary"],
        },
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(feedback_learning.generate_insight, g)
        await cache_set(key, payload)
    return _finalize(payload, provenance="Live feedback-learning loop data (server-derived)")
