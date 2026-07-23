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
    data_constraint_context,
    digital_twin,
    executive_brief,
    feedback_learning,
    home_kpi,
    hte,
    knowledge_graph,
    model_performance,
    predictive_cohort,
    resource_optimization,
    treatment_effect,
)
from src.insights import (
    experiments as experiments_insight_mod,
)
from src.insights.common import cache_get, cache_key, cache_set

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/insights", tags=["Strategic Insights"])


class GroundingChip(BaseModel):
    label: str
    value: str


class MitigationSourceClass(BaseModel):
    """One proxy source class from the authored claims-lag mitigation playbook
    (domain_vocabulary.yaml data_constraints.mitigation_playbook)."""

    name: str
    latency: str
    coverage: str
    illustrative_vendors: list[str] = Field(default_factory=list)
    # e.g. "already live in this platform (the provisional/nowcast KPI overlay)"
    status: str | None = None


class MitigationPlaybook(BaseModel):
    """The authored claims-lag mitigation playbook, served VERBATIM (never
    LM-generated) so the structural-constraints block is actionable: proxy
    source classes, class-level latency bands, coverage caveats, illustrative
    vendors pending data-strategy validation (frontend review 2026-07-22,
    item 2b)."""

    preamble: str
    vendor_note: str
    source_classes: list[MitigationSourceClass] = Field(default_factory=list)


class StrategicInsightResponse(BaseModel):
    insight: str
    key_takeaways: list[str] = Field(default_factory=list)
    grounding: list[GroundingChip] = Field(default_factory=list)
    is_fallback: bool
    generated_at: str
    provenance: str
    # Channel 2 of the constraint-aware two-channel triage (home-kpis today):
    # structural constraints — escalation/investment considerations for
    # data-strategy/platform owners, rendered by the page as a distinct block
    # so channel-1 recommendations are not diluted. None on surfaces that do
    # not produce it.
    structural_considerations: str | None = None
    # Deterministic companion to the structural block (home-kpis today):
    # authored playbook rendered by the page beneath the LM channel. None on
    # surfaces that do not produce it, and on playbook authoring failures.
    mitigation_playbook: MitigationPlaybook | None = None


def _finalize(payload: dict[str, Any], provenance: str) -> StrategicInsightResponse:
    return StrategicInsightResponse(
        insight=payload["insight"],
        key_takeaways=payload.get("key_takeaways", []),
        grounding=[GroundingChip(**c) for c in payload.get("grounding", [])],
        is_fallback=payload["is_fallback"],
        generated_at=datetime.now(timezone.utc).isoformat(),
        provenance=provenance,
        structural_considerations=payload.get("structural_considerations"),
        mitigation_playbook=payload.get("mitigation_playbook"),
    )


# ---- Request models -----------------------------------------------------------
class KGInsightRequest(BaseModel):
    brand: str = "All"
    curated_only: bool = True


class ModelPerfInsightRequest(BaseModel):
    model_version: str


class HomeKpiInsightRequest(BaseModel):
    """Only the SCOPE is caller-supplied; the KPI figures are recomputed
    server-side (same trust boundary as ExecutiveBriefInsightRequest)."""

    brand: str = "All"
    # Lowercase US-census region ('northeast'/'south'/'midwest'/'west') or
    # None for the All-US portfolio view — same param the KPI batch route takes.
    region: str | None = None


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


class PredictiveWhatIfInsightRequest(BaseModel):
    """One hypothetical what-if row: the entered profile + the model's score."""

    model_version: str
    features: dict[str, Any] = Field(default_factory=dict)
    probability: float
    confidence: float | None = None
    cohort_mean: float | None = None
    n_scored: int | None = None
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


class DigitalTwinInsightRequest(BaseModel):
    # Brand only: the grounding is derived SERVER-SIDE from the twin-model
    # inventory, simulation history and the per-intervention identification map
    # (same read paths as /digital-twin/health, /simulations and
    # /intervention-types). Accepting caller-posted figures would let any
    # authenticated caller mint a grounded-looking insight from arbitrary
    # numbers under digital-twin provenance (same trust boundary as
    # /insights/executive-brief).
    brand: str = "Remibrutinib"
    twin_type: str = "hcp"


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


@router.post("/home-kpis", response_model=StrategicInsightResponse)
async def home_kpi_insight(
    req: HomeKpiInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Strategic interpretation of the home KPI grid for a brand + territory
    (server-derived grounding: registry KPIs recomputed under the same
    brand/region context the dashboard's batch endpoint uses)."""
    from src.api.routes.kpi import get_kpi_calculator

    def _load() -> dict[str, Any]:
        calc = get_kpi_calculator()
        metas = calc.list_kpis()
        # Brand scope (frontend review 2026-07-22): under a selected brand,
        # another brand's hard-bound KPIs leave the batch, the grounding, AND
        # the constraint context together — mirroring the dashboard grid's
        # automatic brand scoping (build_grounding re-applies the same filter
        # as defense in depth).
        if req.brand != "All":
            metas = [m for m in metas if not m.brand or m.brand == req.brand]
        # Same context shape the dashboard's POST /kpis/batch sends; use_cache
        # means this usually re-reads the values the grid just computed.
        context: dict[str, Any] = {}
        if req.brand != "All":
            context["brand"] = req.brand
        if req.region:
            context["region"] = req.region
        batch = calc.calculate_batch(kpi_ids=[m.id for m in metas], use_cache=True, context=context)
        g = home_kpi.build_grounding(req.brand, req.region, metas, batch.results)
        # Constraint-aware two-channel triage (2026-07-20): deterministic
        # measurement-constraint block for the KPIs that actually computed.
        # Empty on failure (loud degradation): the chip surfaces it and the
        # short cache TTL below keeps the constraint-blind generation from
        # pinning for the full hour.
        computed_ids = {r.kpi_id for r in batch.results if r.value is not None and not r.error}
        g["data_constraint_context"] = data_constraint_context.build_constraint_context(
            req.brand, [m for m in metas if m.id in computed_ids]
        )
        if not g["data_constraint_context"]:
            g["grounding"].append({"label": "Constraint context", "value": "unavailable"})
        return g

    try:
        g = await asyncio.to_thread(_load)
    except Exception as e:  # noqa: BLE001 — degrade honestly, never 500
        logger.warning("home KPI insight grounding unavailable: %s", e)
        return _finalize(
            {
                "insight": "The KPI values for this scope are currently unavailable, "
                "so no grounded interpretation can be produced right now — this is "
                "a data-source failure, not an empty dashboard.",
                "key_takeaways": [],
                "grounding": [],
                "is_fallback": True,
            },
            provenance="Registry KPIs for this scope (unavailable)",
        )
    # Key on the derived grounding strings so two scopes that differ in any
    # computed value or status never collide. The constraint context is part
    # of the key ("dcc", mirroring exec-brief's "cc"/"cl"): a caveat/profile
    # edit must produce a new generation, not serve the stale narrative for
    # the residual TTL.
    key = cache_key(
        "home-kpis",
        f"{req.brand}:{req.region or 'all-us'}",
        {
            "t": g["kpi_table"],
            "s": g["status_summary"],
            "c": g["coverage"],
            "dcc": g["data_constraint_context"],
        },
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(home_kpi.generate_insight, g)
        # Degraded states are transient (LM outage, rejected samples, or a
        # constraint-context builder hiccup): cache briefly so the page
        # self-heals instead of pinning the degraded narrative for an hour.
        degraded = payload.get("is_fallback") or not g["data_constraint_context"]
        await cache_set(key, payload, ttl_seconds=300 if degraded else 3600)
    # Authored playbook, attached OUTSIDE the cached LM payload: deterministic
    # from the vocabulary, so a vocab edit shows immediately (the "dcc" cache
    # component already forces a fresh narrative for the same edit). None on
    # authoring failure — the block simply doesn't render.
    payload = {
        **payload,
        "mitigation_playbook": data_constraint_context.build_mitigation_playbook(),
    }
    return _finalize(payload, provenance="Registry KPIs recomputed for this scope (server-derived)")


@router.post("/digital-twin", response_model=StrategicInsightResponse)
async def digital_twin_insight(
    req: DigitalTwinInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Strategic interpretation of a brand's digital-twin simulation program
    (server-derived grounding: twin models, simulation history, effect coverage)."""
    from src.api.routes.digital_twin import _get_twin_repo
    from src.digital_twin.effect.cohort_loader import cohort_treatment_availability
    from src.digital_twin.effect.provider import INTERVENTION_CATALOG
    from src.digital_twin.models.twin_models import TwinType

    try:
        repo = await _get_twin_repo()
        models = await repo.list_active_models(twin_type=TwinType(req.twin_type), brand=req.brand)
        simulations = await repo.simulations.list_simulations(brand=req.brand, limit=100)
        effect_available = await cohort_treatment_availability(repo.client, req.brand)
        g = digital_twin.build_grounding(
            req.brand,
            models or [],
            simulations or [],
            effect_available,
            INTERVENTION_CATALOG,
        )
    except Exception as e:  # noqa: BLE001 — degrade honestly, never 500
        logger.warning("digital-twin insight grounding unavailable: %s", e)
        return _finalize(
            {
                "insight": f"Digital-twin data for {req.brand} is currently unavailable, "
                "so no grounded interpretation can be produced right now.",
                "key_takeaways": [],
                "grounding": [],
                "is_fallback": True,
            },
            provenance="Digital-twin simulation program (unavailable)",
        )
    key = cache_key(
        "digital-twin",
        f"{req.brand}:{req.twin_type}",
        {"s": g["simulation_summary"], "c": g["intervention_coverage"], "m": g["model_summary"]},
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(digital_twin.generate_insight, g)
        await cache_set(key, payload)
    return _finalize(payload, provenance="Digital-twin simulation program (server-derived)")


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
    from src.insights.clinical_context import format_clinical_positioning

    drivers = await fetch_commercial_drivers(
        req.brand, outcomes=("TRx", "NRx", "market share", "ROI")
    )
    # Clinical setting (2026-07-23): the brand's labeled target population + line
    # of therapy GATE the commercial recommendations — a strong modeled effect in
    # a clinically off-target population is not actionable. Curated label facts
    # (no network); an unknown brand yields "" and the interpretation proceeds
    # without a clinical gate.
    positioning = format_clinical_positioning(req.brand)
    g = causal_discovery.build_grounding(
        req.brand,
        req.grain,
        [e.model_dump() for e in req.effects],
        causal_drivers=drivers,
        clinical_positioning=positioning,
    )
    # Fold the positioning into the cache key so an edit to it (or a brand whose
    # positioning changes) never serves a stale, ungated interpretation.
    key = cache_key(
        "causal-discovery",
        req.brand,
        {"t": g["effects_table"], "r": g["registry_context"], "cp": g["clinical_positioning"]},
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
            "dr": g["drivers_summary"],
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


@router.post("/predictive-whatif", response_model=StrategicInsightResponse)
async def predictive_whatif_insight(
    req: PredictiveWhatIfInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Plain-language read of ONE hypothetical what-if prediction (inputs, score,
    SHAP drivers, cohort-mean comparison) + how to use it. Predictive, not causal."""
    from src.insights.causal_context import fetch_commercial_drivers

    reg_brand, reg_terms = predictive_cohort.outcome_terms_for_model(req.model_version)
    drivers = await fetch_commercial_drivers(reg_brand, outcomes=reg_terms) if reg_terms else []
    g = predictive_cohort.build_whatif_grounding(
        req.model_version,
        req.features,
        req.probability,
        req.confidence,
        req.cohort_mean,
        req.n_scored,
        [d.model_dump() for d in req.top_drivers],
        causal_drivers=drivers,
    )
    # Keyed on the profile + result + drivers strings: two what-ifs that differ
    # in any entered input, the score, or the SHAP read never collide.
    key = cache_key(
        "predictive-whatif",
        req.model_version,
        {
            "p": g["profile_summary"],
            "s": g["result_summary"],
            "d": g["drivers_summary"],
            "r": g["registry_context"],
        },
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(predictive_cohort.generate_whatif_insight, g)
        await cache_set(key, payload)
    return _finalize(payload, provenance="What-if prediction + per-row SHAP")


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
    # Clinical setting (2026-07-12): commercial moves are not made in a clinical
    # vacuum. Digit-free by construction; fetch fails open to None -> honest
    # "no clinical context" grounding, never a blocked brief.
    from src.insights.clinical_context import fetch_clinical_payload, format_clinical_context

    clinical_context = format_clinical_context(await fetch_clinical_payload(req.brand, "TRx"))
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
        clinical_context=clinical_context,
    )
    # Key on the derived grounding strings (scope + opportunities + caveats +
    # causal levers + clinical context) so two portfolios that differ in any
    # figure — or in the clinical grounding — never collide.
    key = cache_key(
        "executive-brief",
        req.brand,
        {
            "s": g["scope"],
            "o": g["opportunities"],
            "c": g["caveats"],
            "cc": g["lm_causal_context"],
            "cl": g["lm_clinical_context"],
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
    # Clinical setting (2026-07-12): the brand's clinical context grounds the
    # targeting read qualitatively. Digit-free; fetch fails open to None. Keyed
    # on the analyzed outcome so the mapped endpoint matches the run.
    from src.insights.clinical_context import fetch_clinical_payload, format_clinical_context

    clinical_context = format_clinical_context(
        await fetch_clinical_payload(
            record_dict.get("brand"), record_dict.get("outcome_var") or "TRx"
        )
    )
    g = hte.build_grounding(record_dict, causal_drivers=drivers, clinical_context=clinical_context)
    # Key on the derived grounding strings so two runs differing in any figure
    # — or in the clinical grounding — never collide (the analysis_id alone
    # would pin a stale re-run).
    key = cache_key(
        "hte",
        req.analysis_id,
        {
            "e": g["effect_summary"],
            "s": g["segments"],
            "t": g["targeting"],
            "cc": g["commercial_context"],
            "cl": g["clinical_context"],
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


class ExperimentsInsightRequest(BaseModel):
    """Grounding is server-derived (running experiments × final A/B results,
    grouped by intervention channel); the caller only picks the scope. The
    provenance opt-in mirrors the monitor sweep's (#894): the A/B portfolio IS
    the synthetic-gold substrate, so a real-mode deployment needs the same
    explicit opt-in the page's checkbox sends."""

    brand: str = "All"
    include_synthetic: bool = False


@router.post("/experiments", response_model=StrategicInsightResponse)
async def experiments_portfolio_insight(
    req: ExperimentsInsightRequest, user: dict[str, Any] = Depends(require_analyst)
) -> StrategicInsightResponse:
    """Strategic interpretation of the in-silico A/B experimentation portfolio:
    which engagement interventions show statistically supported lift on the
    brand's primary outcome, and what adopting the winners would be worth
    (2026-07-11 /experiments usefulness review)."""
    from src.api.dependencies.supabase_client import get_supabase
    from src.repositories.provenance import apply_provenance_filter

    def _load() -> dict[str, Any]:
        client = get_supabase()
        if client is None:
            raise RuntimeError("Database unavailable")
        # One embed query: running channel-tagged experiments with their final
        # results. Channel presence (migration 100) defines A/B-portfolio
        # membership — scope_definer scaffolding rows have no channel.
        query = (
            client.table("ml_experiments")
            .select(
                "id, brand, intervention_channel, "
                "ab_experiment_results(effect_estimate, p_value, is_significant)"
            )
            .eq("status", "running")
            .not_.is_("intervention_channel", "null")
        )
        if req.brand and req.brand != "All":
            query = query.eq("brand", req.brand)
        query = apply_provenance_filter(query, req.include_synthetic)
        rows = query.execute().data or []
        return experiments_insight_mod.build_grounding(req.brand, rows)

    try:
        g = await asyncio.to_thread(_load)
    except Exception as e:  # noqa: BLE001 — degrade honestly, never 500
        logger.warning("experiments insight grounding unavailable: %s", e)
        return _finalize(
            {
                "insight": "The A/B portfolio results for this scope are currently "
                "unavailable, so no grounded interpretation can be produced right "
                "now — this is a data-source failure, not an empty portfolio.",
                "key_takeaways": [],
                "grounding": [],
                "is_fallback": True,
            },
            provenance="A/B portfolio results for this scope (unavailable)",
        )
    # Key on the derived grounding strings so any change in computed effects or
    # scope produces a fresh narrative (same discipline as the other routes).
    key = cache_key(
        "experiments",
        f"{req.brand}:{req.include_synthetic}",
        {"s": g["scope"], "c": g["channel_effects"], "h": g["highlights"]},
    )
    cached = await cache_get(key)
    if cached is not None:
        payload = cached
    else:
        payload = await asyncio.to_thread(experiments_insight_mod.generate_insight, g)
        await cache_set(key, payload)
    return _finalize(
        payload, provenance="A/B portfolio results grouped by intervention (server-derived)"
    )
