"""Read-only catalog routes for the causal package (#1991 debt 4).

What the causal surfaces can be asked about: which library a free-text question
routes to, which brands a cohort holds, which columns are offerable as
treatment / outcome / covariate, which question pairs are worth proposing, the
clinical context for a brand-outcome pair, and the gold-standard estimation rows
the frontend posts back into the pipeline.

Import rule: may import ``_common``, ``datasets``, ``loaders`` and non-package
modules only; never another route module or the package root.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies.auth import require_analyst, require_viewer
from src.api.schemas.causal import (
    CausalBrandsResponse,
    CausalLibrary,
    CausalVariablesResponse,
    ClinicalContext,
    EstimationDataResponse,
    PipelineMode,
    ProposedQuestion,
    ProposeQuestionsResponse,
    QuestionType,
    RouteQueryRequest,
    RouteQueryResponse,
)
from src.causal_engine.pipeline.router import (
    LibraryRouter,
)
from src.causal_engine.pipeline.router import (
    QuestionType as RouterQuestionType,
)
from src.repositories.provenance import apply_provenance_filter
from src.utils.redaction import redact_query

from .datasets import (
    _ALL_CLINICAL_COVARIATES,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_FILL_ZERO_OUTCOMES,
    _CAUSAL_NUMERIC_COLUMNS,
    _CAUSAL_NUMERIC_DERIVATIONS,
    _CAUSAL_PHYSICAL_TABLE,
    _DEFAULT_CAUSAL_DATASET,
    _JOIN_DATASETS,
    _NBA_JOINED_COVARIATES,
    _brand_scoped_covariates,
    _column_label,
    _is_randomized_treatment,
    _list_dataset_brands,
)
from .loaders import _coerce_estimation_row, _load_agent_estimation_frame

if TYPE_CHECKING:
    from src.services.clinical_context import ClinicalContextService

logger = logging.getLogger(__name__)

router = APIRouter()

# /clinical-context and /estimation-data were registered AFTER the
# discover-effects block in the flat module, so they cannot ride the same router
# as the rest of catalog. Path ORDER is part of the byte-identical contract, not
# just path content: the OpenAPI ``paths`` object preserves insertion order,
# ``openapi-typescript`` emits frontend/src/types/generated/api.ts in that order,
# and CI's verify-types workflow diffs that file byte-for-byte. These two ride a
# second router the aggregator includes at their original position. Pinned by
# test_causal_openapi_path_order_unchanged.
context_router = APIRouter()


# =============================================================================
# LIBRARY ROUTING ENDPOINTS
# =============================================================================


@router.post(
    "/route",
    response_model=RouteQueryResponse,
    summary="Route causal query to library",
    operation_id="route_causal_query",
)
async def route_causal_query(
    request: RouteQueryRequest,
    user: Dict[str, Any] = Depends(require_analyst),
) -> RouteQueryResponse:
    """
    Route a causal query to the appropriate library.

    Uses NLP classification to determine the best causal library:
    - "Does X cause Y?" → DoWhy (causal identification)
    - "How does effect vary?" → EconML (heterogeneous effects)
    - "Who should we target?" → CausalML (uplift modeling)
    - "How does impact flow?" → NetworkX (system dependencies)

    Args:
        request: Query routing request

    Returns:
        RouteQueryResponse with recommended library and estimators
    """
    logger.info(f"Routing query: {redact_query(request.query)}")

    # Delegate to the production LibraryRouter — the same weighted
    # regex/keyword classifier the pipeline orchestrator uses — instead of the
    # former hardcoded keyword stub. The stub fabricated a fixed 0.75/0.9
    # routing_confidence and ignored this router entirely; the real router
    # computes a confidence from pattern-match strength (0.0 when it cannot
    # classify), so the number shown to the user is earned, not invented.
    if request.prefer_library:
        # Explicit override: force the chosen library. question_type is derived
        # from the library so the UI still shows a precise label, and the
        # router returns confidence=1.0 with a "Forced libraries" rationale.
        decision = _library_router.route(
            request.query or "",
            force_libraries=[request.prefer_library.value],
        )
        api_question_type = _library_to_question_type(request.prefer_library)
    else:
        decision = _library_router.route(request.query or "")
        api_question_type = _router_question_type_to_api(decision.question_type)

    # router.CausalLibrary and api.CausalLibrary are distinct enum classes with
    # identical string values — translate by value.
    primary_library = CausalLibrary(decision.primary_library.value)
    secondary_libraries = [CausalLibrary(lib.value) for lib in decision.secondary_libraries]

    return RouteQueryResponse(
        query=request.query,
        question_type=api_question_type,
        primary_library=primary_library,
        secondary_libraries=secondary_libraries,
        recommended_estimators=_RECOMMENDED_ESTIMATORS.get(primary_library, []),
        routing_confidence=decision.confidence,
        routing_rationale=decision.rationale,
        suggested_pipeline=_recommended_mode_to_pipeline(decision.recommended_mode),
    )


# Module-level singleton: the production question-type classifier, shared with
# the pipeline orchestrator. Stateless after construction (compiles its regex
# patterns once); safe to reuse across requests.
_library_router = LibraryRouter()


def _library_to_question_type(library: CausalLibrary) -> QuestionType:
    """Map a forced/preferred library to its natural API question type."""
    mapping = {
        CausalLibrary.DOWHY: QuestionType.CAUSAL_EFFECT,
        CausalLibrary.ECONML: QuestionType.EFFECT_HETEROGENEITY,
        CausalLibrary.CAUSALML: QuestionType.TARGETING,
        CausalLibrary.NETWORKX: QuestionType.SYSTEM_DEPENDENCIES,
    }
    return mapping.get(library, QuestionType.COMPREHENSIVE)


# Recommended estimators per library (informational; the router does not pick
# estimators). NetworkX is a graph/path tool with no point-estimator.
_RECOMMENDED_ESTIMATORS: Dict[CausalLibrary, List[str]] = {
    CausalLibrary.DOWHY: ["propensity_score_matching", "inverse_propensity_weighting"],
    CausalLibrary.ECONML: ["causal_forest", "linear_dml", "dr_learner"],
    CausalLibrary.CAUSALML: ["uplift_random_forest", "uplift_gradient_boosting"],
    CausalLibrary.NETWORKX: [],
}

# RouterQuestionType (causal_engine) -> API QuestionType. The two enums were
# defined independently with different member names; this is the single
# translation point. router.UNKNOWN has no API peer -> COMPREHENSIVE (the
# router already reports confidence 0.0 for unclassifiable queries, so the low
# confidence — not a fabricated label — signals the uncertainty to the UI).
_ROUTER_QT_TO_API: Dict[RouterQuestionType, QuestionType] = {
    RouterQuestionType.CAUSAL_RELATIONSHIP: QuestionType.CAUSAL_EFFECT,
    RouterQuestionType.EFFECT_HETEROGENEITY: QuestionType.EFFECT_HETEROGENEITY,
    RouterQuestionType.TARGETING_OPTIMIZATION: QuestionType.TARGETING,
    RouterQuestionType.IMPACT_FLOW: QuestionType.SYSTEM_DEPENDENCIES,
    RouterQuestionType.COMPREHENSIVE: QuestionType.COMPREHENSIVE,
    RouterQuestionType.UNKNOWN: QuestionType.COMPREHENSIVE,
}


def _router_question_type_to_api(router_type: RouterQuestionType) -> QuestionType:
    """Translate a causal_engine RouterQuestionType to the API QuestionType."""
    return _ROUTER_QT_TO_API.get(router_type, QuestionType.COMPREHENSIVE)


def _recommended_mode_to_pipeline(mode: str) -> Optional[PipelineMode]:
    """Map RoutingDecision.recommended_mode to the API PipelineMode.

    The router emits 'sequential', 'parallel', or 'validation_loop'. The API
    PipelineMode exposes only SEQUENTIAL/PARALLEL; a validation_loop (iterative
    cross-library refutation) is surfaced as PARALLEL since it engages multiple
    libraries. An unrecognized mode yields None (no suggestion).
    """
    if mode in ("parallel", "validation_loop"):
        return PipelineMode.PARALLEL
    if mode == "sequential":
        return PipelineMode.SEQUENTIAL
    return None


# =============================================================================
# GOLD-STANDARD VARIABLE DISCOVERY + ESTIMATION DATA
# =============================================================================
#
# The causal-discovery page used to free-type treatment/outcome/covariate
# column names (defaults rep_visits/trx_count were not real columns) and never
# attached data, so "Run parallel pipeline" fail-closed with 503. These two
# read-only endpoints fix both: /variables drives data-backed dropdowns, and
# /estimation-data loads REAL gold-standard rows server-side that the frontend
# posts into the existing (unchanged) pipeline path.
#


# Column display labels + definitions live in the leaf module
# src/insights/column_labels.py (moved 2026-09-05, #1895, so the insight
# builders can label prose without importing this route). Re-exported above as
# ``_COLUMN_LABELS`` / ``_column_label`` / ``_COLUMN_DEFINITIONS`` — six test
# modules and segments.py import them from here under those names.


@router.get(
    "/brands",
    response_model=CausalBrandsResponse,
    summary="List the brands present in a gold-standard dataset's cohort",
    operation_id="list_causal_brands",
)
async def list_causal_brands(
    dataset: str = Query(
        _DEFAULT_CAUSAL_DATASET,
        description="Gold-standard dataset to enumerate brands for (e.g. patient_journeys)",
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> CausalBrandsResponse:
    """Return the distinct brands present in ``dataset`` for the discovery page's
    brand dropdown. Data-driven: only brands with real rows are offered; selecting
    one scopes the discovery run's cohort to that brand.
    """
    if dataset not in _CAUSAL_DATASET_SPECS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown causal dataset '{dataset}'. "
                f"Known datasets: {sorted(_CAUSAL_DATASET_SPECS)}"
            ),
        )
    brands = await _list_dataset_brands(dataset)
    return CausalBrandsResponse(dataset=dataset, brands=brands)


@router.get(
    "/variables",
    response_model=CausalVariablesResponse,
    summary="List causal variables for a gold-standard dataset",
    operation_id="list_causal_variables",
)
async def list_causal_variables(
    dataset: str = Query(
        _DEFAULT_CAUSAL_DATASET,
        description="Gold-standard dataset to enumerate (e.g. patient_journeys)",
    ),
    brand: Optional[str] = Query(
        None,
        description=(
            "Brand the analysis will be scoped to. Covariate candidates are "
            "brand-scoped: a brand's own clinical biomarkers are offered only "
            "for that brand; omitted (all-brands) offers the universals only — "
            "mirroring what the estimation paths actually adjust for."
        ),
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> CausalVariablesResponse:
    """Return treatment/outcome/covariate candidates for the causal-discovery
    dropdowns.

    Candidates are the curated causally-meaningful columns for ``dataset``,
    intersected with the columns actually present in the live table — so the
    dropdowns are data-driven and never offer a non-existent column.

    Covariate candidates are additionally BRAND-scoped through the same
    ``_brand_scoped_covariates`` gate the estimation paths use. Before this
    (2026-07-13 clinical-faithfulness review) the candidate surface was
    brand-blind: a Fabhalta (PNH) question was offered ``urticaria_severity_uas7``
    — an urticaria activity score whose column is NULL for every Fabhalta row —
    which estimation then silently dropped. Offer and estimation now agree.
    ``columns`` (the raw schema inventory) is deliberately NOT scoped.
    """
    spec = _CAUSAL_DATASET_SPECS.get(dataset)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown causal dataset '{dataset}'. "
                f"Known datasets: {sorted(_CAUSAL_DATASET_SPECS)}"
            ),
        )

    # JOIN datasets have no single physical table to probe; their curated spec
    # lists ARE the candidate variables (derived columns like centrality_z live
    # on no table). Return them directly instead of 500-ing on a missing relation.
    if dataset in _JOIN_DATASETS:
        covariate_candidates = _brand_scoped_covariates(list(spec["covariate"]), brand)
        all_cols = sorted(set(spec["treatment"]) | set(spec["outcome"]) | set(spec["covariate"]))
        _offered = list(spec["treatment"]) + list(spec["outcome"]) + covariate_candidates
        labels = {c: _column_label(c) for c in _offered}
        return CausalVariablesResponse(
            dataset=dataset,
            treatment_candidates=list(spec["treatment"]),
            outcome_candidates=list(spec["outcome"]),
            covariate_candidates=covariate_candidates,
            columns=all_cols,
            labels=labels,
            clinical_biomarkers=sorted(_ALL_CLINICAL_COVARIATES),
        )

    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Causal data store unavailable")

    # Probe one row to learn the columns actually present in the live schema.
    probe = (
        await client.table(_CAUSAL_PHYSICAL_TABLE.get(dataset, dataset))
        .select("*")
        .limit(1)
        .execute()
    )
    rows = probe.data or []
    present = set(rows[0].keys()) if rows else set()

    def _available(role: str) -> List[str]:
        # If the probe returned nothing (empty table), fall back to the curated
        # list so the dropdowns still populate rather than collapsing to empty.
        if not present:
            return list(spec[role])
        return [c for c in spec[role] if c in present]

    # #1872: nba_triggers covariates are JOINED from patient_journeys (like the
    # #1188 baselines below) — the physical-table probe would filter every one
    # of them out, so the curated list bypasses the probe for this dataset.
    if dataset == "nba_triggers":
        covariate_candidates = _brand_scoped_covariates(list(spec["covariate"]), brand)
    else:
        covariate_candidates = _brand_scoped_covariates(_available("covariate"), brand)

    # #1188: baselines are JOINED from patient_journeys, not columns of this
    # dataset's physical table — the probe cannot vet them, so the curated
    # list is returned directly (like the JOIN-dataset branch above). The list
    # is universal patient_journeys columns only, so it needs no brand scoping.
    baseline_candidates = list(spec.get("baseline_covariate", []))

    _offered = (
        _available("treatment") + _available("outcome") + covariate_candidates + baseline_candidates
    )
    labels = {c: _column_label(c) for c in _offered}
    return CausalVariablesResponse(
        dataset=dataset,
        treatment_candidates=_available("treatment"),
        outcome_candidates=_available("outcome"),
        covariate_candidates=covariate_candidates,
        baseline_candidates=baseline_candidates,
        columns=sorted(present),
        labels=labels,
        clinical_biomarkers=sorted(_ALL_CLINICAL_COVARIATES),
    )


def _adjusted_partial_corr(
    df: "pd.DataFrame",  # type: ignore[name-defined] # noqa: F821
    treatment: str,
    outcome: str,
    covariates: List[str],
) -> Optional[float]:
    """Frisch-Waugh-Lovell partial correlation of treatment & outcome adjusting
    for ``covariates`` — a cheap (no-EconML) screening signal for proposing
    questions. Residualize treatment and outcome on the covariates, correlate
    the residuals. Returns None when undefined (zero-variance residuals)."""
    import numpy as np

    t = df[treatment].to_numpy(dtype=float)
    o = df[outcome].to_numpy(dtype=float)
    if t.std() == 0 or o.std() == 0:
        return None
    if covariates:
        cov_mat = df[covariates].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(cov_mat)), cov_mat])
        beta_t, *_ = np.linalg.lstsq(design, t, rcond=None)
        beta_o, *_ = np.linalg.lstsq(design, o, rcond=None)
        rt = t - design @ beta_t
        ro = o - design @ beta_o
    else:
        rt, ro = t - t.mean(), o - o.mean()
    if rt.std() == 0 or ro.std() == 0:
        return None
    return float(np.corrcoef(rt, ro)[0, 1])


@router.get(
    "/propose-questions",
    response_model=ProposeQuestionsResponse,
    summary="Propose data-ranked candidate causal questions for a dataset",
    operation_id="propose_causal_questions",
)
async def propose_causal_questions(
    dataset: str = Query(
        _DEFAULT_CAUSAL_DATASET,
        description="Gold-standard dataset to propose questions for",
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> ProposeQuestionsResponse:
    """Rank candidate treatment->outcome questions by a DATA-DRIVEN screening
    signal, so the agent PROPOSES the question instead of the analyst guessing
    from blind dropdowns.

    For each allowed (treatment, outcome) pair the adjusted partial correlation
    (controlling for the dataset's curated covariates) is computed and ranked by
    magnitude. This is a SCREENING signal — NOT a validated causal effect; the
    user confirms a question and the full agent analysis builds the DAG,
    estimates, and refutes it. Fail-closed: unknown dataset 404, no store 503.
    """
    spec = _CAUSAL_DATASET_SPECS.get(dataset)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown causal dataset '{dataset}'. "
                f"Known datasets: {sorted(_CAUSAL_DATASET_SPECS)}"
            ),
        )

    # Brand-agnostic screening: exclude the indication-specific clinical covariates
    # (each is populated for only one brand, so cross-brand they are ~2/3 NaN and
    # would poison the FWL residualization). Universals are always populated.
    covariates_all = _brand_scoped_covariates(list(spec["covariate"]), None)
    pairs = [(t, o) for t in spec["treatment"] for o in spec["outcome"] if t != o]

    async def _score(t: str, o: str) -> Optional[ProposedQuestion]:
        # #1872 (codex iter-1): mirror the submit endpoint's per-treatment
        # default — a RANDOMIZED pair screens UNADJUSTED on the single-table
        # path (adjusting is unnecessary post-randomization, and riding the
        # patient JOIN would make the RCT proposal join-dependent: droppable
        # on missing patient columns).
        per_treatment = [] if _is_randomized_treatment(dataset, t) else covariates_all
        cov = [c for c in per_treatment if c not in (t, o)]
        try:
            df, select_cols = await _load_agent_estimation_frame(
                dataset=dataset,
                treatment_var=t,
                outcome_var=o,
                covariates=cov,
                limit=1500,
            )
        except HTTPException:
            # A pair with no usable data is simply omitted (never fabricated).
            return None
        # Screen on the loader's EXPANDED columns, not the requested names: a
        # categorical covariate (geographic_region; the seven Optum text
        # baselines) leaves the frame as ``<col>=<level>`` dummies, and an
        # all-NULL covariate is dropped — indexing ``df`` by the raw list
        # KeyError-ed (HTTP 500) for the default dataset and for every dataset
        # with a categorical. Same contract as discovery._prerank_signal (D5).
        cov_expanded = [c for c in select_cols if c not in (t, o)]
        pc = _adjusted_partial_corr(df, t, o, cov_expanded)
        if pc is None:
            return None
        return ProposedQuestion(
            treatment=t,
            outcome=o,
            association_strength=abs(pc),
            direction="positive" if pc > 0 else ("negative" if pc < 0 else "none"),
            n_rows=int(df.shape[0]),
        )

    scored = await asyncio.gather(*[_score(t, o) for t, o in pairs])
    candidates = sorted(
        [c for c in scored if c is not None],
        key=lambda c: c.association_strength,
        reverse=True,
    )
    return ProposeQuestionsResponse(dataset=dataset, candidates=candidates)


# Clinical Context enrichment service (lazy real REST clients inside; see
# src/services/clinical_context). Built on FIRST USE rather than at import
# (#1991 debt 4) so importing this route module does not construct the four HTTP
# clients. The patch seam is this module-level cache, NOT the accessor: a test
# seeds ``catalog._clinical_context_service`` with a stub, and the accessor's
# ``global`` resolves in THIS module's dict no matter whose binding of the
# function was called. That is what makes one patch point serve every reader —
# discovery from-imports the accessor and so holds its own binding of it, which
# a patch on the accessor would miss. Stateless apart from its in-process
# per-brand cache.
_clinical_context_service: Optional["ClinicalContextService"] = None


def _get_clinical_context_service() -> "ClinicalContextService":
    """Built on first use, not at import (#1991 debt 4).

    The constructor builds four HTTP clients, so importing this module must not
    pay for them; a request that actually needs context does.
    """
    global _clinical_context_service
    if _clinical_context_service is None:
        from src.services.clinical_context import ClinicalContextService

        _clinical_context_service = ClinicalContextService()
    return _clinical_context_service


@context_router.get(
    "/clinical-context",
    response_model=ClinicalContext,
    summary="Brand-faithful, sourced clinical context for a discovered effect",
    operation_id="get_causal_clinical_context",
)
async def get_clinical_context(
    brand: str = Query(..., description="Brand to enrich (e.g. Kisqali / Fabhalta / Remibrutinib)"),
    outcome: str = Query(
        ...,
        description=(
            "The synthetic outcome column the effect uses (e.g. persistent_180d); "
            "mapped to the real pivotal endpoint."
        ),
    ),
    treatment: Optional[str] = Query(
        default=None,
        description=(
            "The synthetic treatment column the analysis estimates the effect of "
            "(e.g. treatment_arm / copay_support). Optional: with it the context is "
            "framed for THAT analysis and the literature search follows it; without "
            "it the response is the brand-level view."
        ),
    ),
    user: Dict[str, Any] = Depends(require_viewer),
) -> ClinicalContext:
    """Return the drug + mechanism of action (ChEMBL), the disease's real pivotal
    endpoints (ClinicalTrials.gov), and a real-world-evidence citation (PubMed)
    for ``brand``, mapping our synthetic ``outcome`` to the real endpoint framing.

    With ``treatment``, the response also frames the specific analysis being
    interrogated (treatment -> outcome), the literature search follows that
    analysis instead of the brand alone, and ``causal_evidence`` carries the
    public-knowledge-graph evidence for it: the Open Targets indication edge and
    literature whose abstracts were verified to name both entities. A commercial
    treatment lever (copay, PSP, detailing) returns an explicit
    ``commercial_lever`` state instead of the drug's evidence (#1763).

    Additive narrative ONLY — does not touch the causal estimate or its
    adjustment set. Degrades gracefully (static fallbacks) when an upstream API
    is down; never fabricates a citation. The payload's ``honesty_label`` states
    the synthetic-estimate / real-context boundary.
    """
    available = await _list_dataset_brands(_DEFAULT_CAUSAL_DATASET)
    if available and brand not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown brand '{brand}'. Known brands: {available}",
        )
    try:
        # Offload the synchronous httpx fan-out (ChEMBL + CT.gov + PubMed) to a
        # worker thread so a slow / timing-out / rate-limited upstream cannot block
        # the event loop (the cold-cache call can take tens of seconds worst case).
        payload = await asyncio.to_thread(
            _get_clinical_context_service().get_context,
            brand,
            outcome,
            treatment=treatment,
            # This is the panel the analyst opened — the one place the extra live
            # evidence calls are worth paying for (the leaderboard fan-out is not).
            include_causal_evidence=True,
        )
    except KeyError:
        # The brand_map has no profile for this brand (no enrichment facts).
        raise HTTPException(
            status_code=404,
            detail=f"No clinical-context profile for brand '{brand}'.",
        )
    return ClinicalContext.model_validate(payload)


@context_router.get(
    "/estimation-data",
    response_model=EstimationDataResponse,
    summary="Load real estimation records from a gold-standard dataset",
    operation_id="get_causal_estimation_data",
)
async def get_causal_estimation_data(
    treatment_var: str = Query(..., description="Treatment column to load"),
    outcome_var: str = Query(..., description="Outcome column to load"),
    dataset: str = Query(_DEFAULT_CAUSAL_DATASET, description="Gold-standard dataset"),
    covariates: Optional[str] = Query(
        None, description="Comma-separated covariate columns (confounders)"
    ),
    limit: int = Query(4000, ge=100, le=20000, description="Max rows to load"),
    user: Dict[str, Any] = Depends(require_analyst),
) -> EstimationDataResponse:
    """Load REAL estimation rows for the requested variables, server-side.

    The frontend posts the returned ``estimation_data_records`` into a pipeline
    request's ``filters`` so the (unchanged) parallel/sequential pipeline can
    estimate a real effect. Requested columns are validated against the
    dataset's curated allowlist (an arbitrary column/table cannot be read), and
    rows missing a treatment/outcome value are dropped. Never fabricates data:
    if no usable rows exist the endpoint fails closed with 503.
    """
    spec = _CAUSAL_DATASET_SPECS.get(dataset)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown causal dataset '{dataset}'. "
                f"Known datasets: {sorted(_CAUSAL_DATASET_SPECS)}"
            ),
        )

    if dataset in _JOIN_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dataset '{dataset}' is a JOIN grain not served by this raw "
                "estimation-data endpoint; use POST /causal/discover-effects or "
                "POST /causal/agent-analyze."
            ),
        )

    allowed = set(spec["treatment"]) | set(spec["outcome"]) | set(spec["covariate"])
    covs = [c.strip() for c in (covariates or "").split(",") if c.strip()]
    requested = [treatment_var, outcome_var, *covs]
    not_allowed = [c for c in requested if c not in allowed]
    if not_allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column(s) {not_allowed} are not permitted for dataset "
                f"'{dataset}'. Allowed: {sorted(allowed)}"
            ),
        )

    # #1872: the nba_triggers covariates live on patient_journeys, not on the
    # triggers table this endpoint reads — selecting them here would be a raw
    # PostgREST 42703. Honest 400 pointing at the join-aware agent path.
    # Checked over ALL requested slots (codex iter-2): the union allowlist
    # above is role-insensitive, so a joined column could otherwise ride in as
    # treatment_var/outcome_var.
    if dataset == "nba_triggers":
        joined = [c for c in requested if c in _NBA_JOINED_COVARIATES]
        if joined:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Covariate(s) {joined} are patient-joined on nba_triggers "
                    "(patient_journeys columns, not triggers columns); this raw "
                    "single-table endpoint cannot serve them — use the join-aware "
                    "POST /causal/agent-analyze or POST /causal/discover-effects."
                ),
            )

    # De-duplicate while preserving order (treatment/outcome may also be covariates).
    select_cols = list(dict.fromkeys(requested))

    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Causal data store unavailable")

    query = client.table(_CAUSAL_PHYSICAL_TABLE.get(dataset, dataset)).select(",".join(select_cols))
    # Synthetic-showcase aware: on a synthetic-gold instance the synthetic rows
    # ARE the substrate; on a strict real-data instance they are excluded.
    query = apply_provenance_filter(query)
    result = await query.limit(limit).execute()
    rows = result.data or []

    records: List[Dict[str, Any]] = []
    for row in rows:
        record = _coerce_estimation_row(
            row,
            select_cols=select_cols,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            numeric_cols=_CAUSAL_NUMERIC_COLUMNS.get(dataset, set()),
            derivations=_CAUSAL_NUMERIC_DERIVATIONS.get(dataset),
            fill_zero=frozenset(_CAUSAL_FILL_ZERO_OUTCOMES.get(dataset, set())),
        )
        if record is not None:
            records.append(record)

    if not records:
        raise HTTPException(
            status_code=503,
            detail=(
                "No usable estimation rows for the requested variables "
                f"({treatment_var} -> {outcome_var}) in dataset '{dataset}'."
            ),
        )

    return EstimationDataResponse(
        dataset=dataset,
        columns=select_cols,
        n_rows=len(records),
        estimation_data_records=records,
    )
