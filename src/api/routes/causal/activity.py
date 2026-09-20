"""Activity, registry and effect-readout routes for the causal package (#1991 debt 4).

What the dashboard reads back: the estimator registry behind GET /estimators, the
public GET /health with its memoized analysis-activity counters, GET /history,
GET /value-chains, and GET /treatment-effects (the cohort x brand ATE fit).

Import rule: may import ``_common``, ``datasets``, ``loaders`` and ``pipelines``
(the treatment-effects estimator reuses the Surface-C sequential pipeline) and
non-package modules; never the package root, and never a route module that
imports this one.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies.auth import require_viewer
from src.api.dependencies.compute import HeavyComputeSaturated, heavy_compute_slot
from src.api.models.graph import (
    CausalChainResponse,
    EntityType,
    GraphNode,
    GraphPath,
    GraphRelationship,
    RelationshipType,
)
from src.api.schemas.causal import (
    CausalAnalysisHistoryItem,
    CausalAnalysisHistoryResponse,
    CausalHealthResponse,
    CausalLibrary,
    EstimatorInfo,
    EstimatorListResponse,
    TreatmentEffectResponse,
)
from src.causal_engine.estimator_registry import ESTIMATOR_SPECS
from src.causal_engine.pipeline.state import PipelineInput

# #931: the health check's analysis-activity fields and the Analysis History tab
# read REAL completed causal-analysis events from episodic_memories (the
# canonical store written by the causal_impact agent's
# ``causal_analysis_completed`` episodic hook). Reuse the episodic repository
# rather than issuing raw SQL from the route. Imported at module level so the
# read functions are patchable in tests as ``activity.count_memories_by_type`` /
# ``activity.get_recent_memories``.
from src.memory.episodic_memory import count_memories_by_type, get_recent_memories
from src.repositories.provenance import apply_provenance_filter

from ._common import (
    _GENERIC_500_DETAIL,
    _ROBUSTNESS_UNVALIDATED_WARNING,
    CAUSAL_COMPLETED_EVENT_TYPE,
    _as_float,
    _as_optional_float,
    _dowhy_interval,
    _parse_occurred_at,
    _te_pvalue_from_z,
)
from .loaders import _TE_MAX_PAGES, _TE_PAGE_SIZE, _te_paged_select
from .pipelines import _SurfaceCSequentialPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


# #931 (review M1): /causal/health is a PUBLIC, unauthenticated endpoint the
# dashboard polls every ~30s. The activity fields now read episodic_memories,
# so memoize the result for a short window to keep repeated/unauthenticated
# polls from amplifying into two DB reads each. The cache holds the REAL value
# (or the honest fallback) — it never serves a fabricated number.
_ACTIVITY_CACHE_TTL_SECONDS = 30.0
_activity_cache: dict[str, Any] = {"expires_at": 0.0, "value": (0, None)}


# =============================================================================
# ESTIMATOR INFO ENDPOINT
# =============================================================================


# Single source of truth for the supported causal estimators. Both
# ``list_estimators`` (the /estimators endpoint) and ``causal_health_check``
# (``estimators_loaded``) read from this so the health count can never drift
# from the registry (previously ``estimators_loaded`` was a hardcoded ``12``).
_ESTIMATOR_REGISTRY: List[EstimatorInfo] = [
    *[
        EstimatorInfo(
            name=spec.public_name or spec.estimator_type.value,
            library=CausalLibrary(spec.public_library),
            estimator_type=spec.public_estimator_type,
            description=spec.description,
            best_for=list(spec.best_for),
            parameters=list(spec.parameters),
            supports_confidence_intervals=spec.supports_confidence_intervals,
            supports_heterogeneous_effects=spec.produces_cate,
            agent_override=spec.forceable_alias,
            default_enabled=spec.default_priority is not None,
        )
        for spec in ESTIMATOR_SPECS
    ],
    # CausalML
    EstimatorInfo(
        name="uplift_random_forest",
        library=CausalLibrary.CAUSALML,
        estimator_type="Uplift",
        description="Uplift Random Forest for targeting",
        best_for=["Marketing optimization", "Customer targeting"],
        parameters=["n_estimators", "max_depth", "min_samples_treatment"],
        supports_confidence_intervals=False,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="uplift_gradient_boosting",
        library=CausalLibrary.CAUSALML,
        estimator_type="Uplift",
        description="Uplift Gradient Boosting",
        best_for=["High accuracy targeting"],
        parameters=["n_estimators", "learning_rate", "max_depth"],
        supports_confidence_intervals=False,
        supports_heterogeneous_effects=True,
    ),
    # DoWhy
    EstimatorInfo(
        name="propensity_score_matching",
        library=CausalLibrary.DOWHY,
        estimator_type="Identification",
        description="Propensity Score Matching",
        best_for=["Observational studies", "Selection bias"],
        parameters=["caliper", "n_neighbors"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=False,
    ),
    EstimatorInfo(
        name="inverse_propensity_weighting",
        library=CausalLibrary.DOWHY,
        estimator_type="Identification",
        description="Inverse Propensity Score Weighting",
        best_for=["Survey adjustments", "Treatment weighting"],
        parameters=["propensity_model", "stabilized"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=False,
        agent_override="propensity_score_weighting",
    ),
    EstimatorInfo(
        name="instrumental_variable",
        library=CausalLibrary.DOWHY,
        estimator_type="Identification",
        description="Instrumental Variable (2SLS/LIML)",
        best_for=["Endogeneity", "Unmeasured confounders"],
        parameters=["instruments", "method"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=False,
    ),
]


@router.get(
    "/estimators",
    response_model=EstimatorListResponse,
    summary="List available estimators",
    operation_id="list_estimators",
)
async def list_estimators(
    library: Optional[CausalLibrary] = Query(None, description="Filter by library"),
) -> EstimatorListResponse:
    """
    List available causal estimators.

    Args:
        library: Optional filter by library

    Returns:
        EstimatorListResponse with estimator information
    """
    estimators = list(_ESTIMATOR_REGISTRY)

    # Filter by library if specified
    if library:
        estimators = [e for e in estimators if e.library == library]

    # Group by library
    by_library: Dict[str, List[str]] = {}
    for est in estimators:
        lib_name = est.library.value
        if lib_name not in by_library:
            by_library[lib_name] = []
        by_library[lib_name].append(est.name)

    return EstimatorListResponse(
        estimators=estimators,
        total=len(estimators),
        by_library=by_library,
    )


# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================


@router.get(
    "/health",
    response_model=CausalHealthResponse,
    summary="Causal engine health check",
    operation_id="causal_health_check",
)
async def causal_health_check() -> CausalHealthResponse:
    """
    Health check for causal inference engine.

    Returns:
        CausalHealthResponse with component status
    """
    libraries_available = {
        "dowhy": False,
        "econml": False,
        "causalml": False,
        "networkx": False,
    }

    # Check library availability
    try:
        import dowhy  # noqa: F401

        libraries_available["dowhy"] = True
    except ImportError:
        pass

    try:
        import econml  # noqa: F401

        libraries_available["econml"] = True
    except ImportError:
        pass

    try:
        import causalml  # noqa: F401

        libraries_available["causalml"] = True
    except ImportError:
        pass

    try:
        import networkx  # noqa: F401

        libraries_available["networkx"] = True
    except ImportError:
        pass

    # Check engine components
    hierarchical_ready = False
    pipeline_ready = False
    try:
        from src.causal_engine.hierarchical import HierarchicalAnalyzer  # noqa: F401

        hierarchical_ready = True
    except ImportError:
        pass

    try:
        from src.causal_engine.pipeline import PipelineOrchestrator  # noqa: F401

        pipeline_ready = True
    except ImportError:
        pass

    # Determine overall status
    all_libs = all(libraries_available.values())
    status = (
        "healthy" if all_libs else "degraded" if any(libraries_available.values()) else "unhealthy"
    )

    # #931: surface REAL recent causal-analysis activity from episodic_memories
    # (was a hardcoded 0/None stub from the original phase-B scaffold). The
    # count is the number of completed causal analyses in the last 24h; the
    # most-recent event's timestamp is ``last_analysis``. A read failure
    # degrades to an honest 0/None (never a fabricated value) so a transient
    # episodic-store issue can't take the whole health check down.
    analysis_count_24h, last_analysis = await _recent_causal_activity()

    return CausalHealthResponse(
        status=status,
        libraries_available=libraries_available,
        estimators_loaded=len(_ESTIMATOR_REGISTRY),  # real count from the registry
        pipeline_orchestrator_ready=pipeline_ready,
        hierarchical_analyzer_ready=hierarchical_ready,
        last_analysis=last_analysis,
        analysis_count_24h=analysis_count_24h,
        average_latency_ms=None,
        error=None if status == "healthy" else "Some libraries unavailable",
    )


async def _recent_causal_activity() -> tuple[int, Optional[datetime]]:
    """Return ``(count_last_24h, last_analysis_timestamp)`` for completed causal
    analyses, cached for a short window (review M1).

    The cache exists only to keep the public, frequently-polled health endpoint
    from amplifying into repeated DB reads; the cached value is the REAL reading
    (or the honest fallback), never a fabricated number.
    """
    now = time.monotonic()
    if now < _activity_cache["expires_at"]:
        return cast("tuple[int, Optional[datetime]]", _activity_cache["value"])

    value = await _read_causal_activity()
    _activity_cache["value"] = value
    _activity_cache["expires_at"] = now + _ACTIVITY_CACHE_TTL_SECONDS
    return value


async def _read_causal_activity() -> tuple[int, Optional[datetime]]:
    """Read ``(count_last_24h, last_analysis_timestamp)`` for completed causal
    analyses from episodic_memories.

    Both values are REAL (traced to ``causal_analysis_completed`` episodic rows)
    or an honest fallback (``0`` / ``None``) — never fabricated. On any read
    error we log and fall back so the health check stays available.
    """
    try:
        # ``days_back=1`` is the 24h window; provenance filter defaults to
        # excluding synthetic rows so the KPI reflects real activity.
        count = await count_memories_by_type(
            event_type=CAUSAL_COMPLETED_EVENT_TYPE,
            days_back=1,
        )
    except Exception:  # pragma: no cover - defensive
        logger.warning("causal health: 24h analysis count read failed", exc_info=True)
        count = 0

    last_analysis: Optional[datetime] = None
    try:
        recent = await get_recent_memories(
            limit=1,
            event_types=[CAUSAL_COMPLETED_EVENT_TYPE],
        )
        if recent:
            last_analysis = _parse_occurred_at(recent[0].get("occurred_at"))
    except Exception:  # pragma: no cover - defensive
        logger.warning("causal health: last-analysis read failed", exc_info=True)
        last_analysis = None

    return count, last_analysis


@router.get(
    "/history",
    response_model=CausalAnalysisHistoryResponse,
    summary="Recent completed causal analyses",
    operation_id="get_causal_analysis_history",
)
async def get_causal_analysis_history(
    limit: int = Query(20, ge=1, le=100, description="Maximum history items to return"),
    user: Dict[str, Any] = Depends(require_viewer),
) -> CausalAnalysisHistoryResponse:
    """Return recent completed causal analyses for the Analysis History tab.

    #931: feeds the previously-unwired History tab from REAL
    ``causal_analysis_completed`` episodic_memories rows (newest first). ATE,
    confidence and model are read from each row's ``raw_content`` when present;
    when a field is missing it stays ``None`` (never fabricated). An empty store
    yields an honest empty history rather than a synthesized series.
    """
    try:
        rows = await get_recent_memories(
            limit=limit,
            event_types=[CAUSAL_COMPLETED_EVENT_TYPE],
        )
    except Exception as exc:
        logger.error("Failed to read causal analysis history: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=_GENERIC_500_DETAIL) from exc

    items: List[CausalAnalysisHistoryItem] = []
    for row in rows:
        memory_id = row.get("memory_id")
        if not memory_id:
            # memory_id is the PK; a row without one can't be keyed honestly on
            # the client (empty keys collide). Skip rather than emit a blank id.
            logger.warning("causal history: skipping row with missing memory_id")
            continue
        occurred_at = _parse_occurred_at(row.get("occurred_at"))
        if occurred_at is None:
            # A row without a parseable timestamp can't be placed on the history
            # timeline honestly; skip it rather than invent a time.
            continue
        raw_content = row.get("raw_content")
        if not isinstance(raw_content, dict):
            raw_content = {}
        items.append(
            CausalAnalysisHistoryItem(
                memory_id=str(memory_id),
                event_type=str(row.get("event_type", CAUSAL_COMPLETED_EVENT_TYPE)),
                description=row.get("description"),
                occurred_at=occurred_at,
                agent_name=row.get("agent_name"),
                ate_estimate=_as_float(raw_content.get("ate_estimate")),
                confidence=_as_float(raw_content.get("confidence")),
                model_used=raw_content.get("model_used"),
            )
        )

    return CausalAnalysisHistoryResponse(items=items, total=len(items))


# =============================================================================
# CAUSAL VALUE CHAINS (dashboard "Primary Causal Value Chains" — REAL, dynamic)
# =============================================================================

# 'All'/'portfolio' selections from the Home dropdowns mean "no scope filter".
_ALL_BRAND_SENTINELS = {"all", "all brands", "portfolio", "all (combined portfolio)"}
_ALL_REGION_SENTINELS = {"all", "all us", "all regions", "all us regions"}


def _chain_node_sequence(row: Mapping[str, Any]) -> List[str]:
    """Ordered node names for a ``causal_paths`` row.

    Prefer the stored ``causal_chain.nodes`` ordering; fall back to
    ``start_node`` + ``intermediate_nodes`` + ``end_node``.
    """
    chain = row.get("causal_chain")
    if isinstance(chain, dict):
        nodes = chain.get("nodes")
        if isinstance(nodes, list) and len(nodes) >= 2 and all(isinstance(n, str) for n in nodes):
            return list(nodes)
    seq: List[str] = []
    start = row.get("start_node")
    if isinstance(start, str) and start:
        seq.append(start)
    inter = row.get("intermediate_nodes")
    if isinstance(inter, list):
        seq.extend(n for n in inter if isinstance(n, str) and n)
    end = row.get("end_node")
    if isinstance(end, str) and end:
        seq.append(end)
    return seq


def _chain_score(row: Mapping[str, Any]) -> float:
    """Rank key: |effect| x confidence (None-safe). Surfaces the strongest,
    best-supported chains first — never a fabricated magnitude."""
    eff = _as_optional_float(row.get("causal_effect_size"))
    conf = _as_optional_float(row.get("confidence_level"))
    return abs(eff or 0.0) * (conf or 0.0)


def _causal_path_to_graphpath(row: Mapping[str, Any]) -> GraphPath:
    """Map a ``causal_paths`` row to the ``GraphPath`` shape the dashboard renders.

    The chain-level effect (``causal_effect_size``) is placed on the TERMINAL
    edge as ``ate_estimate`` and the method on every edge — the dashboard reads
    the ate from the terminal edge and the method from the first, so the card
    surfaces the REAL effect/method. ``effect_size`` is intentionally NOT set as
    a number (it is a categorical label in this platform).
    """
    names = _chain_node_sequence(row)
    nodes = [
        GraphNode(
            id=f"var:{n}",
            type=EntityType.AGENT,
            name=n,
            properties={"original_type": "Variable"},
            created_at=None,
            updated_at=None,
        )
        for n in names
    ]
    conf_f = _as_optional_float(row.get("confidence_level"))
    eff_f = _as_optional_float(row.get("causal_effect_size"))
    method = row.get("method_used")

    rels: List[GraphRelationship] = []
    n_edges = len(nodes) - 1
    for i in range(n_edges):
        props: Dict[str, Any] = {}
        if method:
            props["method"] = method
        if i == n_edges - 1:  # terminal edge: chain-level effect + lifecycle/temporal
            if eff_f is not None:
                props["ate_estimate"] = eff_f
            # Real lifecycle/temporal signals for the dashboard's status badge —
            # NOT a confidence bucket. The frontend derives the tag from these.
            vstatus = row.get("validation_status")
            if vstatus:
                props["validation_status"] = vstatus
            cc = row.get("confirmation_count")
            if cc is not None:
                props["confirmation_count"] = cc
            ddate = row.get("discovery_date")
            if ddate:
                props["discovery_date"] = ddate
        rels.append(
            GraphRelationship(
                id="",
                type=RelationshipType.CAUSES,
                source_id=nodes[i].id,
                target_id=nodes[i + 1].id,
                properties=props,
                confidence=conf_f,
                created_at=None,
            )
        )

    plen = row.get("path_length")
    try:
        plen_i = int(plen) if plen is not None else n_edges
    except (TypeError, ValueError):
        plen_i = n_edges

    return GraphPath(
        nodes=nodes,
        relationships=rels,
        total_confidence=conf_f,
        path_length=plen_i,
    )


@router.get(
    "/value-chains",
    response_model=CausalChainResponse,
    summary="Top discovered causal value chains (brand/region scoped)",
    operation_id="get_causal_value_chains",
)
async def get_causal_value_chains(
    brand: Optional[str] = Query(
        None, description="Scope to a brand; omit or 'All' for the portfolio view"
    ),
    region: Optional[str] = Query(
        None, description="Scope to a region; omit or 'All US' for all regions"
    ),
    limit: int = Query(
        3, ge=1, le=20, description="Max distinct chains (top by |effect| x confidence)"
    ),
    user: Dict[str, Any] = Depends(require_viewer),
) -> CausalChainResponse:
    """Return the strongest REAL discovered causal value chains from ``causal_paths``.

    These are the live, dataset-derived chains the causal engine has *validated*
    (DoWhy backdoor estimation) — NOT a seeded graph fixture. Scoped by the Home
    dashboard's brand/region selectors, ranked by ``|effect| x confidence``, and
    de-duplicated by full pathway so the top-N are distinct value chains. Honors
    the synthetic-showcase provenance flag (``E2I_INCLUDE_SYNTHETIC``): on a
    synthetic-gold instance the synthetic chains ARE the substrate; on a strict
    real-data instance they are excluded verbatim.
    """
    start = time.time()
    try:
        from src.memory.services.factories import get_async_supabase_client

        client = await get_async_supabase_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Causal store unavailable")

        query = (
            client.table("causal_paths")
            .select(
                "path_id,start_node,end_node,intermediate_nodes,causal_chain,"
                "causal_effect_size,confidence_level,method_used,validation_status,"
                "confirmation_count,discovery_date,brand,region,path_length"
            )
            .eq("validation_status", "validated")
        )
        if brand and brand.strip().lower() not in _ALL_BRAND_SENTINELS:
            query = query.eq("brand", brand)
        if region and region.strip().lower() not in _ALL_REGION_SENTINELS:
            # causal_paths.region is stored lowercase (US-Census regions:
            # northeast/south/midwest/west). The dropdown sends title-case labels
            # ('Northeast'), so normalize to lowercase — an exact .eq against a
            # title-case value would silently return zero chains.
            query = query.eq("region", region.strip().lower())

        # Synthetic-showcase aware (SSOT). Showcase → include synthetic chains;
        # strict real-mode → excluded verbatim.
        query = apply_provenance_filter(query)

        # Pull a generous, effect-ordered slice; dedupe by pathway; rank by
        # |effect| x confidence so the top-N are DISTINCT, strongly-supported chains.
        result = await (
            query.order("causal_effect_size", desc=True).limit(max(limit * 12, 60)).execute()
        )
        rows: List[Dict[str, Any]] = result.data or []

        seen: set = set()
        distinct: List[Dict[str, Any]] = []
        for r in rows:
            seq = _chain_node_sequence(r)
            if len(seq) < 2:
                continue
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            distinct.append(r)

        distinct.sort(key=_chain_score, reverse=True)
        top = distinct[:limit]

        chains = [_causal_path_to_graphpath(r) for r in top]
        strongest = chains[0] if chains else None
        latency_ms = (time.time() - start) * 1000.0

        return CausalChainResponse(
            chains=chains,
            total_chains=len(chains),
            strongest_chain=strongest,
            # Heterogeneous pathways/scales — no honest scalar aggregate; the UI
            # hides the badge when this is None (never renders a fabricated 0.0%).
            aggregate_effect=None,
            query_latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Causal value-chains query failed")
        raise HTTPException(status_code=500, detail="Causal value-chains query failed") from exc


# =============================================================================
# TREATMENT EFFECTS (GET /causal/treatment-effects — cohort x brand ATE)
# =============================================================================

# The four cohorts the Treatment Effects surface supports. Each maps to the
# outcome column that becomes the binary label. The patient cohorts read
# patient_journeys; hcp_adoption reads hcp_brand_adoption JOIN hcp_profiles.
_TE_PATIENT_OUTCOME = {
    "initiation": "treatment_initiated",
    "persistence": "persistent_180d",
    "discontinuation": "discontinued_180d",
}
_TE_COHORTS = set(_TE_PATIENT_OUTCOME) | {"hcp_adoption"}
_TE_BRANDS = {"Remibrutinib", "Fabhalta", "Kisqali"}

# treatment column shared by all four cohorts (binary 0/1 arm).
_TE_TREATMENT_VAR = "treatment_arm"

# Numeric confounders per cohort family. geographic_region is DELIBERATELY
# EXCLUDED for the patient cohorts: it is a categorical string column that breaks
# DoWhy/EconML (they require numeric inputs). The HCP cohort joins hcp_profiles
# for the centrality confounders that drive treatment_arm by construction.
_TE_PATIENT_CONFOUNDERS = ["disease_severity", "academic_hcp"]
_TE_HCP_CONFOUNDERS = ["peer_influence_score", "influence_network_size"]


# Per-request compute budget (seconds) for the DoWhy+EconML fit under the
# heavy-compute slot. A single cohort fit is seconds; this bounds a degenerate
# run so a slow/contended box returns 408 rather than holding the slot forever.
_TE_TIMEOUT_SECONDS = 90.0


async def _resolve_treatment_effect_frame(
    cohort: str,
    brand: str,
) -> Optional["_TEFrameSpec"]:
    """Load a confounded estimation frame for (cohort, brand) from the DB.

    Returns a ``_TEFrameSpec`` (numeric-coerced + dropna'd DataFrame +
    treatment/outcome/confounder names) or ``None`` (caller fail-closes 503) when
    the cohort frame cannot be resolved or is empty after coercion. NEVER
    fabricates rows. The async Supabase client mirrors /value-chains.
    """
    import pandas as pd

    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        return None

    if cohort in _TE_PATIENT_OUTCOME:
        outcome_var = _TE_PATIENT_OUTCOME[cohort]
        confounders = list(_TE_PATIENT_CONFOUNDERS)
        columns = ",".join([_TE_TREATMENT_VAR, outcome_var, *confounders])
        rows = await _te_paged_select(client, "patient_journeys", columns, brand)
        if not rows:
            return None
        df = pd.DataFrame(rows)
    else:
        # hcp_adoption: hcp_brand_adoption (treatment_arm, adopted) JOIN
        # hcp_profiles (peer_influence_score, influence_network_size) on hcp_id.
        # Two reads + a pandas merge (not the cohort_resolution hcp_profiles
        # branch, which uses a DIFFERENT continuous-treatment substrate).
        outcome_var = "adopted"
        confounders = list(_TE_HCP_CONFOUNDERS)
        adoption_rows = await _te_paged_select(
            client,
            "hcp_brand_adoption",
            "hcp_id,treatment_arm,adopted",
            brand,
        )
        if not adoption_rows:
            return None
        # hcp_profiles is NOT brand-partitioned; read its centrality covariates
        # paged across is_synthetic rows and merge by hcp_id (100% coverage
        # verified). brand filter does not apply here, so read without it.
        profile_rows: List[Dict[str, Any]] = []
        for page in range(_TE_MAX_PAGES):
            offset = page * _TE_PAGE_SIZE
            prof_q = (
                client.table("hcp_profiles")
                .select("hcp_id,peer_influence_score,influence_network_size")
                .eq("is_synthetic", True)
                .range(offset, offset + _TE_PAGE_SIZE - 1)
            )
            prof_res = await prof_q.execute()
            prof_batch: List[Dict[str, Any]] = prof_res.data or []
            profile_rows.extend(prof_batch)
            if len(prof_batch) < _TE_PAGE_SIZE:
                break
        if not profile_rows:
            return None
        adoption_df = pd.DataFrame(adoption_rows)
        profile_df = pd.DataFrame(profile_rows).drop_duplicates(subset="hcp_id")
        df = adoption_df.merge(profile_df, on="hcp_id", how="inner")
        if df.empty:
            return None

    # Build the numeric estimation frame: coerce treatment/outcome/confounders to
    # numeric, drop any row with a non-coercible/NA cell. n = surviving rows. An
    # empty frame after coercion -> None (honest 503), never a fabricated fit.
    use_cols = [_TE_TREATMENT_VAR, outcome_var, *confounders]
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        logger.warning(
            "treatment-effects: cohort=%s brand=%s frame missing columns %s",
            cohort,
            brand,
            missing,
        )
        return None
    est = df[use_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if est.empty:
        return None
    return _TEFrameSpec(
        frame=est.reset_index(drop=True),
        treatment_var=_TE_TREATMENT_VAR,
        outcome_var=outcome_var,
        confounders=confounders,
    )


class _TEFrameSpec(NamedTuple):
    """Resolved estimation frame + runnable var-set for one (cohort, brand) cell."""

    frame: Any  # pd.DataFrame
    treatment_var: str
    outcome_var: str
    confounders: List[str]


async def _run_treatment_effect_estimate(
    cohort: str,
    brand: str,
    spec: "_TEFrameSpec",
) -> TreatmentEffectResponse:
    """Run the wired DoWhy+EconML sequential pipeline on the resolved frame.

    Prefers EconML's ate/ci/std (it carries the CI); falls back to DoWhy's
    causal_effect/standard_error (CI from that SE, #2014) when EconML fails. Raises
    HTTPException(503) when NEITHER executor produces a usable estimate. NEVER
    fabricates a number.
    """
    start = time.time()
    n = int(len(spec.frame))

    pipeline_input = PipelineInput(
        query=f"Treatment effect: cohort={cohort}, brand={brand}",
        treatment_var=spec.treatment_var,
        outcome_var=spec.outcome_var,
        confounders=list(spec.confounders),
        effect_modifiers=None,
        data_source=f"{cohort}/{brand}",
        filters={},
        estimation_data=spec.frame,
        mode="sequential",
        # Only DoWhy + EconML: NetworkX/CausalML are not needed for a single ATE
        # cell and would add latency. _get_execution_order filters SEQUENTIAL_ORDER
        # by this set, so DoWhy then EconML run in order.
        libraries_enabled=["dowhy", "econml"],
        cross_validate=None,
        run_refutation=False,
    )

    pipeline = _SurfaceCSequentialPipeline(fail_fast=False)
    await pipeline.execute(pipeline_input)
    state: Mapping[str, Any] = pipeline.last_state or {}

    # ---- Prefer EconML (carries CI) ----
    econml_result = state.get("econml_result")
    econml_payload = (
        econml_result.get("result")
        if isinstance(econml_result, dict) and isinstance(econml_result.get("result"), dict)
        else None
    )
    dowhy_result = state.get("dowhy_result")
    dowhy_payload = (
        dowhy_result.get("result")
        if isinstance(dowhy_result, dict) and isinstance(dowhy_result.get("result"), dict)
        else None
    )

    ate: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    std_error: Optional[float] = None
    estimator: Optional[str] = None

    if econml_payload is not None and econml_payload.get("ate") is not None:
        ate = _as_optional_float(econml_payload.get("ate"))
        ci_lower = _as_optional_float(econml_payload.get("ate_ci_lower"))
        ci_upper = _as_optional_float(econml_payload.get("ate_ci_upper"))
        std_error = _as_optional_float(econml_payload.get("ate_std"))
        est_name = econml_payload.get("estimator")
        estimator = str(est_name) if est_name is not None else None
    elif dowhy_payload is not None and dowhy_payload.get("causal_effect") is not None:
        # DoWhy fallback: the CI is the 95 % normal interval of its SE (#2014; it was
        # left None although the SE and its p-value were reported).
        ate = _as_optional_float(dowhy_payload.get("causal_effect"))
        std_error = _as_optional_float(dowhy_payload.get("standard_error"))
        estimator = dowhy_payload.get("dowhy_method")
        dowhy_interval = _dowhy_interval(dowhy_payload)
        if dowhy_interval is not None:
            ci_lower, ci_upper = dowhy_interval

    if ate is None:
        # Neither executor produced a usable estimate — honest fail-close.
        logger.warning(
            "treatment-effects: no usable estimate (cohort=%s brand=%s n=%d errors=%s)",
            cohort,
            brand,
            n,
            state.get("errors"),
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Causal pipeline produced no usable treatment-effect estimate for "
                f"cohort={cohort!r} brand={brand!r} (both DoWhy and EconML failed)."
            ),
        )

    p_value = _te_pvalue_from_z(ate, std_error)
    latency_ms = int((time.time() - start) * 1000)

    return TreatmentEffectResponse(
        cohort=cohort,
        brand=brand,
        treatment_var=spec.treatment_var,
        outcome_var=spec.outcome_var,
        confounders=list(spec.confounders),
        ate=ate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        std_error=std_error,
        n=n,
        estimator=estimator,
        method="dowhy+econml sequential",
        confidence_level=0.95,
        latency_ms=latency_ms,
        is_synthetic=True,
        warnings=[_ROBUSTNESS_UNVALIDATED_WARNING],
    )


@router.get(
    "/treatment-effects",
    response_model=TreatmentEffectResponse,
    summary="Estimate the treatment effect for a (cohort, brand) cell",
    operation_id="get_treatment_effect",
)
async def get_treatment_effect(
    cohort: str = Query(
        ...,
        description="Cohort: initiation | persistence | discontinuation | hcp_adoption",
    ),
    brand: str = Query(
        ...,
        description="Brand: Remibrutinib | Fabhalta | Kisqali",
    ),
    user: Dict[str, Any] = Depends(require_viewer),
) -> TreatmentEffectResponse:
    """Return a REAL average treatment effect for one (cohort, brand) cell.

    Loads a confounded cohort frame from the DB (patient_journeys for the patient
    cohorts; hcp_brand_adoption JOIN hcp_profiles for hcp_adoption), then runs the
    EXISTING DoWhy+EconML sequential pipeline to recover a de-confounded ATE + CI
    + p_value + n. Honors the synthetic-showcase substrate (is_synthetic=true).

    Fail-closed: 422 on an unknown cohort/brand; 503 when the cohort frame cannot
    be resolved (no rows) or the pipeline yields no usable estimate; 408 on
    timeout; 503 (Retry-After) when the heavy-compute slot is saturated. NEVER
    fabricates an effect.
    """
    cohort_key = cohort.strip().lower()
    if cohort_key not in _TE_COHORTS:
        raise HTTPException(
            status_code=422,
            detail=(f"Unknown cohort {cohort!r}. Expected one of: {sorted(_TE_COHORTS)}."),
        )
    if brand not in _TE_BRANDS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown brand {brand!r}. Expected one of: {sorted(_TE_BRANDS)}.",
        )

    try:
        spec = await _resolve_treatment_effect_frame(cohort_key, brand)
        if spec is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No resolvable cohort data for cohort={cohort_key!r} brand={brand!r}. "
                    "The cohort frame was empty or unavailable; refusing to fabricate an effect."
                ),
            )
        # The DoWhy+EconML fit is the genuinely heavy in-process compute. Bound it
        # to ONE per-worker heavy-compute slot (OOM guard) + a wall-clock timeout
        # so a contended box returns 408 rather than holding the slot forever.
        async with heavy_compute_slot():
            return await asyncio.wait_for(
                _run_treatment_effect_estimate(cohort_key, brand, spec),
                timeout=_TE_TIMEOUT_SECONDS,
            )
    except HTTPException:
        raise
    except asyncio.TimeoutError as e:
        raise HTTPException(
            status_code=408,
            detail=f"Treatment-effect estimation timed out after {_TE_TIMEOUT_SECONDS}s",
        ) from e
    except HeavyComputeSaturated:
        # Reject fast under load — mapped to 503 + Retry-After by the app handler.
        # Must precede the broad handler so it is not swallowed into a 500.
        raise
    except Exception as e:  # noqa: BLE001 - last-resort 500
        logger.error(f"Treatment-effect estimation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_GENERIC_500_DETAIL) from e
