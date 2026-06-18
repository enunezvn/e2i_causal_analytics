"""
E2I Causal Inference API
========================

FastAPI endpoints for causal inference capabilities.

Phase B10: Causal API endpoints for:
- Hierarchical analysis (EconML within CausalML segments)
- Library routing (DoWhy, EconML, CausalML, NetworkX)
- Multi-library pipelines (sequential, parallel)
- Cross-validation between libraries

Endpoints:
- /causal/hierarchical/analyze: Run hierarchical CATE analysis
- /causal/hierarchical/{analysis_id}: Get analysis results
- /causal/route: Route query to appropriate library
- /causal/pipeline/sequential: Run sequential multi-library pipeline
- /causal/pipeline/parallel: Run parallel multi-library analysis
- /causal/validate: Run cross-library validation
- /causal/estimators: List available estimators
- /causal/health: Health check for causal engine

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import asyncio
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from src.api.dependencies.auth import require_analyst, require_viewer
from src.api.dependencies.compute import HeavyComputeSaturated, heavy_compute_slot
from src.api.dependencies.durable_job_store import DurableJobStore
from src.api.errors import user_safe_503_detail
from src.api.models.graph import (
    CausalChainResponse,
    EntityType,
    GraphNode,
    GraphPath,
    GraphRelationship,
    RelationshipType,
)
from src.api.schemas.causal import (
    AGENT_FORCEABLE_ESTIMATORS,
    AgentCausalAnalysisRequest,
    AgentCausalAnalysisResponse,
    AggregationMethod,
    AnalysisStatus,
    CausalAnalysisHistoryItem,
    CausalAnalysisHistoryResponse,
    CausalBrandsResponse,
    CausalDAGModel,
    CausalHealthResponse,
    CausalLibrary,
    CausalVariablesResponse,
    CrossValidationRequest,
    CrossValidationResponse,
    DiscoveredEffect,
    DiscoverEffectsResponse,
    EstimationDataResponse,
    EstimatorInfo,
    EstimatorListResponse,
    HierarchicalAnalysisRequest,
    HierarchicalAnalysisResponse,
    NestedCIResult,
    ParallelPipelineRequest,
    ParallelPipelineResponse,
    PipelineMode,
    PipelineStageResult,
    ProposedQuestion,
    ProposeQuestionsResponse,
    QuestionType,
    RefutationSummary,
    RefutationTestDetail,
    RouteQueryRequest,
    RouteQueryResponse,
    SegmentationMethod,
    SegmentCATEResult,
    SequentialPipelineRequest,
    SequentialPipelineResponse,
    TreatmentEffectResponse,
)
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.causal.stats import z_score_for_confidence

# #354 C-8: real-pipeline wiring (replaces 503-default short-circuit in
# non-demo mode). Imported lazily-safely; the LibraryExecutor implementations
# inside ParallelPipeline / SequentialPipeline themselves guard their backend
# dependencies (dowhy/econml/causalml/networkx availability), so importing
# the orchestrator classes is cheap.
from src.causal_engine.pipeline.parallel import ParallelPipeline
from src.causal_engine.pipeline.router import (
    LibraryRouter,
)
from src.causal_engine.pipeline.router import (
    QuestionType as RouterQuestionType,
)
from src.causal_engine.pipeline.sequential import SequentialPipeline
from src.causal_engine.pipeline.state import (
    PipelineInput,
    PipelineOutput,
    PipelineState,
)

# #931: the health check's analysis-activity fields and the Analysis History tab
# read REAL completed causal-analysis events from episodic_memories (the
# canonical store written by the causal_impact agent's
# ``causal_analysis_completed`` episodic hook). Reuse the episodic repository
# rather than issuing raw SQL from the route. Imported at module level so the
# read functions are patchable in tests as ``causal.count_memories_by_type`` /
# ``causal.get_recent_memories``.
from src.memory.episodic_memory import count_memories_by_type, get_recent_memories
from src.repositories.provenance import apply_provenance_filter, deployment_includes_synthetic

logger = logging.getLogger(__name__)

# #931: the episodic event_type the causal_impact agent emits when an analysis
# completes. Used by both the health-check activity fields and the history
# endpoint so the KPI count and the History tab share one source of truth.
CAUSAL_COMPLETED_EVENT_TYPE = "causal_analysis_completed"

# #931 (review M1): /causal/health is a PUBLIC, unauthenticated endpoint the
# dashboard polls every ~30s. The activity fields now read episodic_memories,
# so memoize the result for a short window to keep repeated/unauthenticated
# polls from amplifying into two DB reads each. The cache holds the REAL value
# (or the honest fallback) — it never serves a fabricated number.
_ACTIVITY_CACHE_TTL_SECONDS = 30.0
_activity_cache: dict[str, Any] = {"expires_at": 0.0, "value": (0, None)}

# Agent-run wall-clock budgets (orphan-fix). The async agent task wraps the
# whole graph in ``asyncio.wait_for(..., _AGENT_HARD_TIMEOUT_S)`` — a HARD cap.
# But the heavy refutation suite runs in a worker thread that wait_for CANNOT
# cancel (Python can't force-kill a thread), so hitting the hard cap would
# orphan a still-grinding refutation thread that keeps burning a CPU core and
# accumulates across runs. To prevent that we pass the graph a COOPERATIVE
# deadline (``_REFUTATION_COMPUTE_BUDGET_S`` from task start); the refutation
# node skips refuters that would run past it and fails-closed cleanly, so the
# thread returns and releases the heavy-compute slot BEFORE the hard cap fires.
# The gap between them is headroom for one in-flight refuter's overshoot plus
# the post-refutation sensitivity/interpretation nodes.
_AGENT_HARD_TIMEOUT_S = 900.0
_REFUTATION_COMPUTE_BUDGET_S = 720.0

# Generic 5xx detail. Raw exception text MUST NOT be echoed to clients: it can
# leak stack-internal paths, library/module names, table/column names, and other
# information useful to an attacker. The full exception is logged server-side
# (with exc_info) instead; the client receives only this opaque message.
_GENERIC_500_DETAIL = "Internal server error"

_ROBUSTNESS_UNVALIDATED_WARNING = (
    "robustness_validation_performed=false: this ATE was estimated but NOT "
    "refutation-tested (the sequential/parallel pipeline does not run "
    "refutation/sensitivity checks). Treat the effect as UNVALIDATED for "
    "robustness; do not present it as a validated causal claim."
)

# R6-F1 (#740): caveats for the opt-in refutation path. The refutation runs only
# on the DoWhy estimate (Owner-decision 1: a labeled proxy for the consensus),
# and on REVIEW/BLOCK the pipeline DOWNGRADES (still 200, flag=False) rather than
# 503-blocking the whole multi-library answer (Owner-decision 2).
_ROBUSTNESS_REVIEW_WARNING = (
    "robustness_validation_performed=false (gate=REVIEW): the DoWhy refutation "
    "suite returned a REVIEW band (borderline-robust) for this estimate — it is "
    "usable only with expert review and MUST NOT be presented as validated. "
    "Robustness was validated on the DoWhy estimate only; EconML/CausalML "
    "estimates in the consensus are unrefuted."
)
_ROBUSTNESS_BLOCK_WARNING = (
    "robustness_validation_performed=false (gate=BLOCK): the DoWhy refutation "
    "suite BLOCKED this estimate (a critical refutation test failed or confidence "
    "was below threshold). Treat the effect as NOT robust. Robustness was "
    "validated on the DoWhy estimate only; EconML/CausalML estimates in the "
    "consensus are unrefuted."
)

# M-fo2 (precise): a directed cycle only breaks identification when it lands on the
# (treatment, outcome) ancestral subgraph (``undefined_cyclic``). That caveat is
# un-ignorable — appended to BOTH the warnings list and robustness_warning — and it
# FORCES robustness False, sets requires_review=True, and WITHHOLDS the consensus
# effect (backdoor adjustment is mathematically undefined on such a graph).
_NON_DAG_STRUCTURAL_WARNING = (
    "Discovered causal graph contains a directed cycle ON the treatment-outcome "
    "ancestral subgraph; backdoor adjustment is undefined for this estimand. The "
    "consensus effect is WITHHELD and the result is quarantined for review "
    "(requires_review=true) — do NOT treat any per-library number as a causal claim."
)
# A cycle OFF the ancestral subgraph leaves the estimand identifiable: informational
# only, no penalty, consensus preserved.
_CYCLE_IRRELEVANT_WARNING = (
    "Discovered causal graph contains a cycle OUTSIDE the treatment-outcome "
    "ancestral subgraph; this estimand remains identifiable and no structural "
    "penalty was applied."
)

router = APIRouter(
    prefix="/causal",
    tags=["Causal Inference"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# IN-MEMORY STORAGE (for demo - replace with database in production)
# =============================================================================

_analysis_cache: Dict[str, HierarchicalAnalysisResponse] = {}
_pipeline_cache: Dict[str, Dict[str, Any]] = {}
_validation_cache: Dict[str, CrossValidationResponse] = {}
# causal_impact agent runs (POST /causal/agent-analyze submit -> GET poll). The
# agent's energy-score selection + refutation is too slow for a synchronous
# request (~minutes), so it runs as a background task and the FE polls.
# Cross-worker job store (Redis-backed; in-memory fallback). The API runs
# multiple gunicorn workers, so a module-level dict would 404 on poll when the
# GET lands on a different worker than the POST. See DurableJobStore.
_agent_analysis_store: DurableJobStore["AgentCausalAnalysisResponse"] = DurableJobStore(
    "causal:agent_analyze", AgentCausalAnalysisResponse
)


# =============================================================================
# HIERARCHICAL ANALYSIS ENDPOINTS
# =============================================================================


@router.post(
    "/hierarchical/analyze",
    response_model=HierarchicalAnalysisResponse,
    summary="Run hierarchical CATE analysis",
    operation_id="run_hierarchical_analysis",
)
async def run_hierarchical_analysis(
    request: HierarchicalAnalysisRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(default=False, description="Run asynchronously"),
    demo_mode: bool = Query(
        default=False,
        description=(
            "If true, return pinned-zero placeholder results labeled with "
            "is_demo=true (for UI demonstrations only). Default is false: "
            "the endpoint runs the real analyzer over inline "
            "estimation_data_records or fails with 503."
        ),
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> HierarchicalAnalysisResponse:
    """
    Run hierarchical CATE analysis (EconML within CausalML segments).

    This endpoint performs Pattern 4 from multi-library synergies:
    - Segments data using uplift scores (quantile, k-means, threshold)
    - Estimates CATE within each segment using EconML
    - Aggregates segment CATEs with nested confidence intervals
    - Computes heterogeneity statistics (I², τ²)

    Fail-closed contract (C1): the default path resolves a real DataFrame from
    ``request.filters.estimation_data_records`` and raises 503 when none is
    present — it NEVER fabricates input data. Pass ``demo_mode=true`` for a
    clearly-labeled pinned-zero placeholder.

    Args:
        request: Hierarchical analysis configuration
        background_tasks: FastAPI background tasks
        async_mode: If True, runs analysis asynchronously
        demo_mode: If True, return clearly-labeled placeholder values

    Returns:
        HierarchicalAnalysisResponse with segment-level CATE results
    """
    analysis_id = str(uuid.uuid4())
    time.time()

    logger.info(
        f"Hierarchical analysis requested: {analysis_id}",
        extra={
            "analysis_id": analysis_id,
            "treatment_var": request.treatment_var,
            "outcome_var": request.outcome_var,
            "n_segments": request.n_segments,
            "estimator_type": request.estimator_type.value,
        },
    )

    if async_mode:
        # Preflight the fail-closed contract BEFORE accepting the submission, so
        # an async non-demo request with no real data fails fast with 503/400
        # (C1) instead of being accepted as pending and then cached as a generic
        # FAILED record by the background task. demo_mode skips the preflight.
        if not demo_mode:
            _resolve_hierarchical_dataframe(request)
        # Create pending response and run in background
        pending_response = HierarchicalAnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.PENDING,
            segment_results=[],
            nested_ci=None,
            overall_ate=None,
            overall_ci_lower=None,
            overall_ci_upper=None,
            confidence_level=request.confidence_level,
            segment_heterogeneity=None,
            n_segments_analyzed=0,
            segmentation_method=request.segmentation_method.value,
            estimator_type=request.estimator_type.value,
            latency_ms=0,
            created_at=datetime.now(timezone.utc),
            warnings=[],
            errors=[],
        )
        _analysis_cache[analysis_id] = pending_response

        background_tasks.add_task(_run_hierarchical_analysis_task, analysis_id, request, demo_mode)

        return pending_response

    # Synchronous execution
    try:
        result = await _execute_hierarchical_analysis(analysis_id, request, demo_mode=demo_mode)
        _analysis_cache[analysis_id] = result
        return result

    except HTTPException:
        # Honest fail-close (503 no-real-data) / client errors (400 bad columns)
        # must pass through unchanged — HTTPException is an Exception subclass, so
        # this MUST precede the broad handler or the 503 becomes a 500.
        raise
    except HeavyComputeSaturated:
        # Reject fast under load — surfaced as 503 + Retry-After by the app
        # exception handler (OOM guard). Must precede the broad handler so it
        # is not swallowed into a 500.
        raise
    except Exception as e:
        logger.error(f"Hierarchical analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_GENERIC_500_DETAIL) from e


@router.get(
    "/hierarchical/{analysis_id}",
    response_model=HierarchicalAnalysisResponse,
    summary="Get hierarchical analysis results",
    operation_id="get_hierarchical_analysis",
)
async def get_hierarchical_analysis(
    analysis_id: str,
) -> HierarchicalAnalysisResponse:
    """
    Get results of a hierarchical analysis by ID.

    Args:
        analysis_id: Unique analysis identifier

    Returns:
        HierarchicalAnalysisResponse with analysis results
    """
    if analysis_id not in _analysis_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {analysis_id} not found",
        )

    return _analysis_cache[analysis_id]


async def _run_hierarchical_analysis_task(
    analysis_id: str,
    request: HierarchicalAnalysisRequest,
    demo_mode: bool = False,
) -> None:
    """Background task for hierarchical analysis."""
    try:
        result = await _execute_hierarchical_analysis(analysis_id, request, demo_mode=demo_mode)
        _analysis_cache[analysis_id] = result
    except Exception as e:
        # Log the raw error server-side (with traceback); the cached FAILED
        # record is later returned to clients, so it must carry only a generic
        # message, not raw exception text.
        logger.error(f"Background hierarchical analysis failed: {e}", exc_info=True)
        _analysis_cache[analysis_id] = HierarchicalAnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.FAILED,
            segment_results=[],
            nested_ci=None,
            overall_ate=None,
            overall_ci_lower=None,
            overall_ci_upper=None,
            confidence_level=request.confidence_level,
            segment_heterogeneity=None,
            n_segments_analyzed=0,
            segmentation_method=request.segmentation_method.value,
            estimator_type=request.estimator_type.value,
            latency_ms=0,
            created_at=datetime.now(timezone.utc),
            warnings=[],
            errors=["Analysis failed due to an internal error."],
        )


def _build_hierarchical_demo_response(
    analysis_id: str,
    request: HierarchicalAnalysisRequest,
    start_time: float,
) -> HierarchicalAnalysisResponse:
    """Build a clearly-labeled pinned-zero placeholder for demo_mode=true.

    Never returns RNG values (C1 de-fabrication): every segment is a hard zero
    and the envelope carries ``is_demo=true`` plus a do-not-use warning so a
    consumer cannot mistake the demo for a real analysis.
    """
    segment_results = [
        SegmentCATEResult(
            segment_id=i,
            segment_name=f"demo_segment_{i}",
            n_samples=0,
            uplift_range=[0.0, 0.0],
            cate_mean=0.0,
            cate_std=0.0,
            cate_ci_lower=0.0,
            cate_ci_upper=0.0,
            success=True,
            error_message=None,
        )
        for i in range(request.n_segments)
    ]
    latency_ms = int((time.time() - start_time) * 1000)
    return HierarchicalAnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.COMPLETED,
        segment_results=segment_results,
        nested_ci=None,
        overall_ate=0.0,
        overall_ci_lower=0.0,
        overall_ci_upper=0.0,
        confidence_level=request.confidence_level,
        segment_heterogeneity=0.0,
        n_segments_analyzed=request.n_segments,
        segmentation_method=request.segmentation_method.value,
        estimator_type=request.estimator_type.value,
        latency_ms=latency_ms,
        created_at=datetime.now(timezone.utc),
        warnings=[
            "demo_mode=true: results are pinned-zero placeholders with "
            "is_demo=true; do NOT use for decisions.",
        ],
        errors=[],
        is_demo=True,
    )


async def _execute_hierarchical_analysis(
    analysis_id: str,
    request: HierarchicalAnalysisRequest,
    *,
    demo_mode: bool = False,
) -> HierarchicalAnalysisResponse:
    """Execute hierarchical analysis using the causal engine.

    Fail-closed contract (C1): the non-demo path requires a real estimation
    DataFrame (resolved from ``request.filters.estimation_data_records``); when
    none is present it raises 503 — it NEVER fabricates synthetic input via RNG.
    ``demo_mode=true`` returns a clearly-labeled pinned-zero placeholder.
    """
    start_time = time.time()

    if demo_mode:
        return _build_hierarchical_demo_response(analysis_id, request, start_time)

    try:
        from src.causal_engine.hierarchical import (
            AggregationMethod as EngineAggregationMethod,
        )
        from src.causal_engine.hierarchical import (
            HierarchicalAnalyzer,
            HierarchicalConfig,
            NestedCIConfig,
            NestedConfidenceInterval,
        )
        from src.causal_engine.hierarchical.analyzer import (
            SegmentationMethod as EngineSegmentationMethod,
        )
        from src.causal_engine.hierarchical.nested_ci import SegmentEstimate

        # Map API enums to engine enums
        segmentation_map = {
            SegmentationMethod.QUANTILE: EngineSegmentationMethod.QUANTILE,
            SegmentationMethod.KMEANS: EngineSegmentationMethod.KMEANS,
            SegmentationMethod.THRESHOLD: EngineSegmentationMethod.THRESHOLD,
            SegmentationMethod.TREE: EngineSegmentationMethod.TREE,
        }

        aggregation_map = {
            AggregationMethod.VARIANCE_WEIGHTED: EngineAggregationMethod.VARIANCE_WEIGHTED,
            AggregationMethod.SAMPLE_WEIGHTED: EngineAggregationMethod.SAMPLE_WEIGHTED,
            AggregationMethod.EQUAL: EngineAggregationMethod.EQUAL,
            AggregationMethod.BOOTSTRAP: EngineAggregationMethod.BOOTSTRAP,
        }

        # Resolve a REAL estimation DataFrame from request filters. No real data
        # backend → honest 503 (C1: never fabricate synthetic input via RNG);
        # missing required columns → 400. Mirrors the sequential/parallel
        # sibling endpoints in this file.
        df = _resolve_hierarchical_dataframe(request)

        # Prepare data from the REAL frame (columns are read as data, not names).
        if request.effect_modifiers:
            X = df[request.effect_modifiers]
        else:
            X = df.drop(columns=[request.treatment_var, request.outcome_var])
        treatment = df[request.treatment_var].values
        outcome = df[request.outcome_var].values

        # Create config and run analysis
        config = HierarchicalConfig(
            n_segments=request.n_segments,
            segmentation_method=segmentation_map.get(
                request.segmentation_method, EngineSegmentationMethod.QUANTILE
            ),
            min_segment_size=request.min_segment_size,
            estimator_type=request.estimator_type.value,
            ci_confidence_level=request.confidence_level,
            compute_nested_ci=True,
        )

        analyzer = HierarchicalAnalyzer(config)
        # The EconML-within-segments fit is the genuinely heavy in-process
        # compute here. Hold ONE per-worker heavy-compute slot for the duration
        # so concurrent heavy requests cannot stack and OOM-kill the cgroup
        # (OOM guard, P1b). Both callers of this helper — the sync path and the
        # background task — route through here, so bounding it once covers both.
        # On a saturated worker, heavy_compute_slot() raises
        # HeavyComputeSaturated on enter (mapped to 503 + Retry-After by the app
        # exception handler) — nothing is queued. The slot wraps the await so it
        # is held for the whole compute, including under the wait_for timeout.
        async with heavy_compute_slot():
            result = await asyncio.wait_for(
                analyzer.analyze(X=X, treatment=treatment, outcome=outcome),
                timeout=request.timeout_seconds,
            )

        # Convert to API response format
        segment_results = []
        for seg in result.segment_results:
            segment_results.append(
                SegmentCATEResult(
                    segment_id=seg.segment_id,
                    segment_name=seg.segment_name,
                    n_samples=seg.n_samples,
                    uplift_range=list(seg.uplift_range),
                    cate_mean=seg.cate_mean,
                    cate_std=seg.cate_std,
                    cate_ci_lower=seg.cate_ci_lower,
                    cate_ci_upper=seg.cate_ci_upper,
                    success=seg.success,
                    error_message=seg.error_message,
                )
            )

        # Compute nested CI
        nested_ci_result = None
        if len([s for s in result.segment_results if s.success]) >= 1:
            nested_ci_config = NestedCIConfig(
                confidence_level=request.confidence_level,
                aggregation_method=aggregation_map.get(
                    request.aggregation_method, EngineAggregationMethod.VARIANCE_WEIGHTED
                ),
                min_segment_size=request.min_segment_size,
            )
            nested_ci_calc = NestedConfidenceInterval(nested_ci_config)

            segment_estimates = [
                SegmentEstimate(
                    segment_id=seg.segment_id,
                    segment_name=seg.segment_name,
                    ate=seg.cate_mean,
                    ate_std=(seg.cate_se if seg.cate_se is not None else (seg.cate_std or 0.01)),
                    ci_lower=seg.cate_ci_lower or seg.cate_mean - 0.1,
                    ci_upper=seg.cate_ci_upper or seg.cate_mean + 0.1,
                    sample_size=seg.n_samples,
                    cate=None,
                )
                for seg in result.segment_results
                if seg.success and seg.cate_mean is not None
            ]

            if segment_estimates:
                ci_result = nested_ci_calc.compute(segment_estimates)
                nested_ci_result = NestedCIResult(
                    aggregate_ate=ci_result.aggregate_ate,
                    aggregate_ci_lower=ci_result.aggregate_ci_lower,
                    aggregate_ci_upper=ci_result.aggregate_ci_upper,
                    aggregate_std=ci_result.aggregate_std,
                    confidence_level=ci_result.confidence_level,
                    aggregation_method=ci_result.aggregation_method,
                    segment_contributions=ci_result.segment_contributions,
                    i_squared=ci_result.i_squared,
                    tau_squared=ci_result.tau_squared,
                    n_segments_included=ci_result.n_segments_included,
                    total_sample_size=ci_result.total_sample_size,
                )

        latency_ms = int((time.time() - start_time) * 1000)

        return HierarchicalAnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.COMPLETED,
            segment_results=segment_results,
            nested_ci=nested_ci_result,
            overall_ate=result.overall_ate,
            overall_ci_lower=result.overall_ate_ci_lower,
            overall_ci_upper=result.overall_ate_ci_upper,
            confidence_level=request.confidence_level,
            segment_heterogeneity=result.segment_heterogeneity,
            n_segments_analyzed=result.n_segments,
            segmentation_method=request.segmentation_method.value,
            estimator_type=request.estimator_type.value,
            latency_ms=latency_ms,
            created_at=datetime.now(timezone.utc),
            warnings=result.warnings if hasattr(result, "warnings") else [],
            errors=result.errors if result.errors else [],
            is_demo=False,
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Analysis timed out after {request.timeout_seconds}s",
        )
    except ImportError as e:
        # Log the specific missing module server-side for ops; do NOT echo the
        # internal dependency name to clients.
        logger.error(f"Segment analysis dependency unavailable: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="A required analysis dependency is currently unavailable.",
        ) from e


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
    logger.info(f"Routing query: {(request.query or '')[:50]}...")

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
# patient_journeys is the gold-standard causal frame: a fully-populated,
# patient-level cohort (treatment_arm -> persistent_180d, controlling for
# disease_severity / engagement_score / age_at_diagnosis) — the same cohort the
# gold-standard models use, with a known TRUE_ATE. (business_metrics is sparse:
# its causal columns are mostly NULL, so it is not offered here.)
#
# Covariate candidates are NUMERIC confounders the executors consume directly.
# Categorical confounders (geographic_region, brand) are still excluded — they
# would need server-side encoding the DoWhy/EconML executors don't do here (brand
# is offered instead as a cohort FILTER via the brand dropdown, not a covariate).
# The covariate list was expanded (#1027) with the additional numeric clinical
# markers that are 100%-populated WITH variance in the gold-standard cohort
# (verified against the live table: academic_hcp, egfr, proteinuria_g_day,
# ldh_ratio, urticaria_severity_uas7, ecog_performance_status) so the analyst has
# a richer adjustment set. Columns that LOOK like confounders but are 100% NULL
# (risk_score, adherence_rate, refill_count, gap_days) are deliberately NOT
# offered — they would fail-close every run. treatment/outcome stay the curated
# causal columns (the synthetic gold-standard only wires those relationships).
_CAUSAL_DATASET_SPECS: Dict[str, Dict[str, List[str]]] = {
    "patient_journeys": {
        "treatment": ["treatment_arm", "treatment_initiated"],
        "outcome": ["persistent_180d", "discontinued_180d", "treatment_initiated"],
        "covariate": [
            "disease_severity",
            "engagement_score",
            "age_at_diagnosis",
            "academic_hcp",
            "egfr",
            "proteinuria_g_day",
            "ldh_ratio",
            "urticaria_severity_uas7",
            "ecog_performance_status",
        ],
    },
}
_DEFAULT_CAUSAL_DATASET = "patient_journeys"

# Columns coerced to float before handing the frame to the executors. Every
# curated candidate above is numeric, so all are coerced; a value that cannot
# be coerced becomes None and (for treatment/outcome) drops the row.
_CAUSAL_NUMERIC_COLUMNS: Dict[str, set] = {
    "patient_journeys": {
        "treatment_arm",
        "treatment_initiated",
        "persistent_180d",
        "discontinued_180d",
        "disease_severity",
        "engagement_score",
        "age_at_diagnosis",
        "academic_hcp",
        "egfr",
        "proteinuria_g_day",
        "ldh_ratio",
        "urticaria_severity_uas7",
        "ecog_performance_status",
    },
}


async def _list_dataset_brands(dataset: str) -> List[str]:
    """Distinct, non-null brand values present in ``dataset``'s live table.

    Data-driven (mirrors how /variables intersects with the live schema): the
    dropdown only ever offers a brand that actually has rows. Returns [] if the
    table has no ``brand`` column or the store is unavailable (the FE then shows
    only 'All brands'). Bounded select — the cohort is small and a few thousand
    rows reliably cover every brand.
    """
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        return []
    try:
        query = client.table(dataset).select("brand")
        query = apply_provenance_filter(query)
        result = await query.limit(20000).execute()
    except Exception as e:  # noqa: BLE001 — missing column / store hiccup => no brands
        logger.warning(f"causal brands: could not enumerate brands for '{dataset}': {e}")
        return []
    seen = {
        str(row["brand"])
        for row in (result.data or [])
        if isinstance(row, dict) and row.get("brand")
    }
    return sorted(seen)


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
    user: Dict[str, Any] = Depends(require_analyst),
) -> CausalVariablesResponse:
    """Return treatment/outcome/covariate candidates for the causal-discovery
    dropdowns.

    Candidates are the curated causally-meaningful columns for ``dataset``,
    intersected with the columns actually present in the live table — so the
    dropdowns are data-driven and never offer a non-existent column.
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

    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Causal data store unavailable")

    # Probe one row to learn the columns actually present in the live schema.
    probe = await client.table(dataset).select("*").limit(1).execute()
    rows = probe.data or []
    present = set(rows[0].keys()) if rows else set()

    def _available(role: str) -> List[str]:
        # If the probe returned nothing (empty table), fall back to the curated
        # list so the dropdowns still populate rather than collapsing to empty.
        if not present:
            return list(spec[role])
        return [c for c in spec[role] if c in present]

    return CausalVariablesResponse(
        dataset=dataset,
        treatment_candidates=_available("treatment"),
        outcome_candidates=_available("outcome"),
        covariate_candidates=_available("covariate"),
        columns=sorted(present),
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

    covariates_all = list(spec["covariate"])
    pairs = [(t, o) for t in spec["treatment"] for o in spec["outcome"] if t != o]

    async def _score(t: str, o: str) -> Optional[ProposedQuestion]:
        cov = [c for c in covariates_all if c not in (t, o)]
        try:
            df, _ = await _load_agent_estimation_frame(
                dataset=dataset,
                treatment_var=t,
                outcome_var=o,
                covariates=cov,
                limit=1500,
            )
        except HTTPException:
            # A pair with no usable data is simply omitted (never fabricated).
            return None
        pc = _adjusted_partial_corr(df, t, o, cov)
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


# =============================================================================
# DISCOVER EFFECTS — validated-effects leaderboard (async submit -> poll)
# =============================================================================

# Cross-worker job store (Redis-backed; mirrors _agent_analysis_store). Each job
# runs the agent for a set of candidate questions and ranks the VALIDATED effects.
_discover_effects_store: DurableJobStore["DiscoverEffectsResponse"] = DurableJobStore(
    "causal:discover_effects", DiscoverEffectsResponse
)

# Complementary outcomes are 1 - each other (persistent_180d vs discontinued_180d);
# running both is redundant, so one is skipped to dedupe the leaderboard.
_COMPLEMENT_OUTCOMES_SKIP = {"discontinued_180d"}


def _discover_candidate_pairs(spec: Dict[str, Any]) -> List[tuple]:
    """Deduped (treatment, outcome) questions for the leaderboard: no self-pairs,
    and complementary outcomes collapsed to one representative."""
    pairs: List[tuple] = []
    for t in spec["treatment"]:
        for o in spec["outcome"]:
            if t == o or o in _COMPLEMENT_OUTCOMES_SKIP:
                continue
            pairs.append((t, o))
    return pairs


def _effect_confidence_score(gate_decision: Optional[str], significant: bool) -> float:
    """Map the robustness gate + significance to a 0-1 ranking signal."""
    base = {"proceed": 0.6, "review": 0.35}.get(gate_decision or "", 0.1)
    return min(1.0, base + (0.3 if significant else 0.0))


def _effect_status_from_gate(ate: Optional[float], gate: Optional[str], resp_status: str) -> str:
    """Honest leaderboard status. A run that produced an estimate is reported by
    its robustness verdict (completed/needs_review/blocked) — only a run that
    produced NO estimate is 'failed'. This separates 'the gate blocked it'
    (computed, worth inspecting) from 'it could not run'."""
    if ate is None:
        return "failed" if resp_status not in {"pending", "running"} else resp_status
    if gate == "proceed":
        return "completed"
    if gate == "review":
        return "needs_review"
    if gate == "block":
        return "blocked"
    return resp_status


def _effect_from_agent_response(
    treatment: str, outcome: str, resp: "AgentCausalAnalysisResponse", analysis_id: str
) -> DiscoveredEffect:
    gate = resp.refutation.gate_decision if resp.refutation else None
    return DiscoveredEffect(
        treatment=treatment,
        outcome=outcome,
        status=_effect_status_from_gate(resp.ate, gate, resp.status),
        ate=resp.ate,
        ate_ci_lower=resp.ate_ci_lower,
        ate_ci_upper=resp.ate_ci_upper,
        p_value=resp.p_value,
        statistical_significance=bool(resp.statistical_significance),
        selected_estimator=resp.selected_estimator,
        gate_decision=gate,
        confidence_score=_effect_confidence_score(gate, bool(resp.statistical_significance)),
        impact=abs(resp.ate) if resp.ate is not None else None,
        n_rows=resp.n_rows,
        analysis_id=analysis_id,
    )


def _rank_effects(effects: List[DiscoveredEffect]) -> List[DiscoveredEffect]:
    """Rank by confidence (gate + significance) then impact (|ate|). Not-yet-run
    questions (score 0) sort last."""
    return sorted(
        effects,
        key=lambda e: (e.confidence_score, e.impact if e.impact is not None else -1.0),
        reverse=True,
    )


async def _run_discover_effects_task(
    job_id: str, dataset: str, pairs: List[tuple], data_source: str, brand: Optional[str] = None
) -> None:
    """Background: validate each candidate question with the causal_impact agent
    (serial — each acquires the heavy-compute slot), updating the cached job after
    each so the FE leaderboard fills in progressively, ranked by confidence+impact.

    ``brand`` (optional) scopes every candidate's estimation frame to one brand."""
    spec = _CAUSAL_DATASET_SPECS[dataset]
    covariates_all = list(spec["covariate"])
    # Keyed pending effects we mutate in place across the run.
    effects: Dict[tuple, DiscoveredEffect] = {
        (t, o): DiscoveredEffect(treatment=t, outcome=o, status="pending") for (t, o) in pairs
    }

    async def _publish(status: str, completed: int) -> None:
        await _discover_effects_store.set(
            job_id,
            DiscoverEffectsResponse(
                job_id=job_id,
                status=status,
                dataset=dataset,
                brand=brand,
                total=len(pairs),
                completed=completed,
                effects=_rank_effects(list(effects.values())),
            ),
        )

    completed = 0
    for t, o in pairs:
        effects[(t, o)] = DiscoveredEffect(treatment=t, outcome=o, status="running")
        await _publish("running", completed)
        try:
            cov = [c for c in covariates_all if c not in (t, o)]
            df, _select = await _load_agent_estimation_frame(
                dataset=dataset,
                treatment_var=t,
                outcome_var=o,
                covariates=cov,
                limit=1500,
                brand=brand,
            )
            aid = str(uuid.uuid4())
            req = AgentCausalAnalysisRequest(
                treatment_var=t,
                outcome_var=o,
                dataset=dataset,
                limit=1500,
                auto_discover=True,
                brand=brand,
            )
            await _agent_analysis_store.set(
                aid,
                AgentCausalAnalysisResponse(
                    analysis_id=aid,
                    status="pending",
                    treatment_var=t,
                    outcome_var=o,
                    dataset=dataset,
                    n_rows=int(df.shape[0]),
                    data_source=data_source,
                    dag=CausalDAGModel(),
                    statistical_significance=False,
                    refutation=RefutationSummary(),
                    latency_ms=0,
                ),
            )
            await _run_agent_analysis_task(aid, req, df, cov, data_source)
            resp = await _agent_analysis_store.get(aid)
            if resp is None:
                raise RuntimeError(f"agent analysis {aid} produced no cached result")
            effects[(t, o)] = _effect_from_agent_response(t, o, resp, aid)
        except HTTPException as e:
            # Fail-closed: a question with no usable data is marked failed, not faked.
            logger.warning(f"discover-effects: {t}->{o} failed-closed: {e.detail}")
            effects[(t, o)] = DiscoveredEffect(treatment=t, outcome=o, status="failed")
        except Exception as e:  # noqa: BLE001
            logger.error(f"discover-effects: {t}->{o} errored: {e}", exc_info=True)
            effects[(t, o)] = DiscoveredEffect(treatment=t, outcome=o, status="failed")
        completed += 1
        await _publish("running" if completed < len(pairs) else "completed", completed)


@router.post(
    "/discover-effects",
    response_model=DiscoverEffectsResponse,
    summary="Discover & rank the agent's VALIDATED causal effects (async submit -> poll)",
    operation_id="discover_causal_effects",
)
async def discover_causal_effects(
    background_tasks: BackgroundTasks,
    dataset: str = Query(_DEFAULT_CAUSAL_DATASET, description="Gold-standard dataset"),
    brand: Optional[str] = Query(
        None,
        description=(
            "Optional brand to scope the cohort to (e.g. Kisqali). None = all "
            "brands. The candidate questions are unchanged; only the rows the "
            "agent estimates on are subset to this brand."
        ),
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> DiscoverEffectsResponse:
    """Run the causal_impact agent across the dataset's candidate questions and
    rank the VALIDATED effects (discovered DAG + estimator + refutation gate) by
    confidence then impact. Heavy (minutes per effect) -> async: returns a pending
    job; poll ``GET /causal/discover-effects/{job_id}``. Fail-closed per question.

    ``brand`` (optional) scopes the cohort to one brand — a plain row subset, so
    each candidate is validated on that brand's patients only.
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
    brand = brand or None
    if brand is not None:
        available = await _list_dataset_brands(dataset)
        if available and brand not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown brand '{brand}' for dataset '{dataset}'. Known: {available}",
            )
    pairs = _discover_candidate_pairs(spec)
    job_id = str(uuid.uuid4())
    data_source = "synthetic" if deployment_includes_synthetic() else "database"
    initial = DiscoverEffectsResponse(
        job_id=job_id,
        status="pending",
        dataset=dataset,
        brand=brand,
        total=len(pairs),
        completed=0,
        effects=[DiscoveredEffect(treatment=t, outcome=o, status="pending") for (t, o) in pairs],
    )
    await _discover_effects_store.set(job_id, initial)
    background_tasks.add_task(
        _run_discover_effects_task, job_id, dataset, pairs, data_source, brand
    )
    return initial


@router.get(
    "/discover-effects/{job_id}",
    response_model=DiscoverEffectsResponse,
    summary="Poll a discover-effects job",
    operation_id="get_discover_causal_effects",
)
async def get_discover_causal_effects(
    job_id: str,
    user: Dict[str, Any] = Depends(require_viewer),
) -> DiscoverEffectsResponse:
    job = await _discover_effects_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown discover-effects job '{job_id}'")
    return job


@router.get(
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

    # De-duplicate while preserving order (treatment/outcome may also be covariates).
    select_cols = list(dict.fromkeys(requested))

    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Causal data store unavailable")

    query = client.table(dataset).select(",".join(select_cols))
    # Synthetic-showcase aware: on a synthetic-gold instance the synthetic rows
    # ARE the substrate; on a strict real-data instance they are excluded.
    query = apply_provenance_filter(query)
    result = await query.limit(limit).execute()
    rows = result.data or []

    numeric_cols = _CAUSAL_NUMERIC_COLUMNS.get(dataset, set())
    records: List[Dict[str, Any]] = []
    for row in rows:
        record: Dict[str, Any] = {}
        usable = True
        for col in select_cols:
            value = row.get(col)
            if col in numeric_cols and value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None
            # A missing treatment/outcome value makes the row unusable for
            # estimation — drop it rather than impute.
            if col in (treatment_var, outcome_var) and value is None:
                usable = False
                break
            record[col] = value
        if usable:
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


# =============================================================================
# AGENT ANALYSIS ENDPOINT (causal_impact agent, end-to-end)
# =============================================================================


async def _load_agent_estimation_frame(
    *,
    dataset: str,
    treatment_var: str,
    outcome_var: str,
    covariates: List[str],
    limit: int,
    brand: Optional[str] = None,
) -> tuple["pd.DataFrame", List[str]]:  # type: ignore[name-defined] # noqa: F821
    """Load a REAL estimation DataFrame for the causal_impact agent.

    Mirrors :func:`get_causal_estimation_data` (validates columns against the
    dataset's curated allowlist, provenance-filters, drops rows missing a
    treatment/outcome value) but returns a pandas DataFrame ready for the agent's
    ``data_cache['estimation_data']``. Fail-closed: raises ``HTTPException`` (404
    unknown dataset, 400 disallowed column, 503 no data store / no usable rows) —
    never fabricates rows.
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

    allowed = set(spec["treatment"]) | set(spec["outcome"]) | set(spec["covariate"])
    requested = [treatment_var, outcome_var, *covariates]
    not_allowed = [c for c in requested if c not in allowed]
    if not_allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column(s) {not_allowed} are not permitted for dataset "
                f"'{dataset}'. Allowed: {sorted(allowed)}"
            ),
        )

    select_cols = list(dict.fromkeys(requested))

    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Causal data store unavailable")

    # ``brand`` is a categorical FILTER (row subset), NOT a causal variable — it
    # scopes the cohort to one brand and stays out of the estimation columns
    # (categorical confounders would need encoding the executors don't do here).
    fetch_cols = list(select_cols)
    if brand:
        fetch_cols = list(dict.fromkeys([*select_cols, "brand"]))
    query = client.table(dataset).select(",".join(fetch_cols))
    query = apply_provenance_filter(query)
    if brand:
        query = query.eq("brand", brand)
    result = await query.limit(limit).execute()
    rows = result.data or []

    numeric_cols = _CAUSAL_NUMERIC_COLUMNS.get(dataset, set())
    records: List[Dict[str, Any]] = []
    for row in rows:
        record: Dict[str, Any] = {}
        usable = True
        for col in select_cols:
            value = row.get(col)
            if col in numeric_cols and value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None
            if col in (treatment_var, outcome_var) and value is None:
                usable = False
                break
            record[col] = value
        if usable:
            records.append(record)

    if not records:
        raise HTTPException(
            status_code=503,
            detail=(
                "No usable estimation rows for the requested variables "
                f"({treatment_var} -> {outcome_var}) in dataset '{dataset}'."
            ),
        )

    import pandas as pd

    return pd.DataFrame(records), select_cols


@router.post(
    "/agent-analyze",
    response_model=AgentCausalAnalysisResponse,
    summary="Run the causal_impact agent end-to-end (DAG + effect + refutation)",
    operation_id="run_causal_agent_analysis",
)
async def run_causal_agent_analysis(
    request: AgentCausalAnalysisRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_analyst),
) -> AgentCausalAnalysisResponse:
    """Submit a causal_impact agent run (async) and return a pending handle.

    Leverages the agent: it builds the causal DAG, selects an estimator
    DATA-DRIVENLY via the energy-score router across the registry (or the forced
    one when ``estimator`` is set), estimates the treatment->outcome effect, and
    runs refutation + sensitivity. That work takes MINUTES, so it runs as a
    BackgroundTask and the client polls ``GET /causal/agent-analyze/{id}`` (same
    submit->poll shape as the hierarchical / pipeline endpoints).

    Data is validated + loaded SYNCHRONOUSLY here, so bad columns / no data
    fail-closed immediately with the right HTTP status (400/404/503); only the
    heavy agent run is deferred. Fail-closed throughout — never a fabricated ATE.
    """
    # Validate the optional estimator override BEFORE loading — the agent
    # restricts forced methods to _VALID_EXPLICIT_METHODS; surface an honest 400.
    if request.estimator and request.estimator not in AGENT_FORCEABLE_ESTIMATORS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Estimator '{request.estimator}' cannot be forced. Supported "
                f"overrides: {list(AGENT_FORCEABLE_ESTIMATORS)}. Omit `estimator` "
                "for Auto (the agent's data-driven routing across the registry)."
            ),
        )

    # Covariates default to the dataset's curated confounders (data-driven), and
    # can never include the treatment/outcome themselves.
    spec = _CAUSAL_DATASET_SPECS.get(request.dataset)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown causal dataset '{request.dataset}'. "
                f"Known datasets: {sorted(_CAUSAL_DATASET_SPECS)}"
            ),
        )
    covariates = [
        c
        for c in (request.covariates if request.covariates is not None else spec["covariate"])
        if c not in (request.treatment_var, request.outcome_var)
    ]

    # Load synchronously -> fail-closed early (400 bad column / 404 dataset /
    # 503 no data) before scheduling the heavy run. ``brand`` (optional) scopes
    # the cohort to one brand (a row subset; brand stays out of the estimation
    # columns) so the analyst can analyze a single brand's patients.
    df, _select_cols = await _load_agent_estimation_frame(
        dataset=request.dataset,
        treatment_var=request.treatment_var,
        outcome_var=request.outcome_var,
        covariates=covariates,
        limit=request.limit,
        brand=request.brand,
    )

    analysis_id = str(uuid.uuid4())
    data_source = "synthetic" if deployment_includes_synthetic() else "database"
    pending = AgentCausalAnalysisResponse(
        analysis_id=analysis_id,
        status="pending",
        treatment_var=request.treatment_var,
        outcome_var=request.outcome_var,
        dataset=request.dataset,
        n_rows=int(df.shape[0]),
        data_source=data_source,
        dag=CausalDAGModel(),
        statistical_significance=False,
        refutation=RefutationSummary(),
        warnings=["Analysis submitted; poll GET /causal/agent-analyze/{id} for the result."],
        latency_ms=0,
    )
    await _agent_analysis_store.set(analysis_id, pending)
    background_tasks.add_task(
        _run_agent_analysis_task, analysis_id, request, df, covariates, data_source
    )
    return pending


@router.get(
    "/agent-analyze/{analysis_id}",
    response_model=AgentCausalAnalysisResponse,
    summary="Poll a causal_impact agent run by id",
    operation_id="get_causal_agent_analysis",
)
async def get_causal_agent_analysis(analysis_id: str) -> AgentCausalAnalysisResponse:
    """Poll a submitted agent run. 404 until the submit registered it; then
    pending -> running -> completed / needs_review / failed."""
    job = await _agent_analysis_store.get(analysis_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    return job


async def _run_agent_analysis_task(
    analysis_id: str,
    request: AgentCausalAnalysisRequest,
    df: "pd.DataFrame",  # type: ignore[name-defined] # noqa: F821
    covariates: List[str],
    data_source: str,
) -> None:
    """Background: run the agent on the pre-loaded frame; cache the result.

    The state is built DIRECTLY (not via CausalImpactAgent, whose wrapper would
    short-circuit a "synthetic" data_source to fast OLS) — so "Auto" runs the
    real energy-score selection across the registry. ``data_source`` here is only
    the response provenance label; the data always comes from data_cache. The
    refutation is bounded so the run completes in minutes (not the full
    ~610-re-estimation suite), with a generous wall-clock cap.
    """
    import time as _time

    prev = await _agent_analysis_store.get(analysis_id)
    if prev is not None:
        await _agent_analysis_store.set(analysis_id, prev.model_copy(update={"status": "running"}))

    parameters: Dict[str, Any] = {}
    if request.estimator:
        parameters["method"] = request.estimator
    parameters.setdefault(
        "refutation_config",
        {
            "bootstrap": {"num_bootstraps": 20},
            "placebo_treatment": {"num_simulations": 10},
            "data_subset": {"num_subsets": 5},
            "random_common_cause": {"num_simulations": 10},
        },
    )
    initial_state: Dict[str, Any] = {
        "query": (
            f"What is the causal effect of {request.treatment_var} on {request.outcome_var}?"
        ),
        "query_id": analysis_id,
        "treatment_var": request.treatment_var,
        "outcome_var": request.outcome_var,
        "confounders": covariates,
        "data_source": data_source,
        "data_cache": {"estimation_data": df},
        # Learn the DAG from data via GUIDED discovery (graph_builder anchors the
        # treatment/outcome roles; the data selects the confounders). Falls back
        # to the domain DAG if discovery is skipped or not accepted by the gate.
        "auto_discover": request.auto_discover,
        "discovery_guided": True,
        "parameters": parameters,
        "interpretation_depth": "standard",
        "brand": request.brand,
        # Cooperative compute deadline so the refutation suite self-terminates
        # before the hard wait_for cap below (orphan-fix): timed-out runs return
        # cleanly instead of orphaning an uncancellable to_thread refutation.
        "compute_deadline": time.monotonic() + _REFUTATION_COMPUTE_BUDGET_S,
        "errors": [],
        "warnings": [],
        "fallback_used": False,
        "retry_count": 0,
    }

    start = _time.time()
    try:
        from src.agents.causal_impact.graph import create_causal_impact_graph

        graph = create_causal_impact_graph()
        # Bound concurrency to ONE per-worker heavy-compute slot (OOM guard),
        # mirroring the hierarchical / parallel endpoints.
        async with heavy_compute_slot():
            final_state = await asyncio.wait_for(
                graph.ainvoke(initial_state), timeout=_AGENT_HARD_TIMEOUT_S
            )
        await _agent_analysis_store.set(
            analysis_id,
            _agent_state_to_response(
                analysis_id=analysis_id,
                request=request,
                data_source=data_source,
                n_rows=int(df.shape[0]),
                final_state=final_state,
                latency_ms=int((_time.time() - start) * 1000),
            ),
        )
    except Exception as e:  # noqa: BLE001 — cache a generic FAILED record
        logger.error(f"Background causal agent analysis failed: {e}", exc_info=True)
        await _agent_analysis_store.set(
            analysis_id,
            AgentCausalAnalysisResponse(
                analysis_id=analysis_id,
                status="failed",
                treatment_var=request.treatment_var,
                outcome_var=request.outcome_var,
                dataset=request.dataset,
                n_rows=int(df.shape[0]),
                data_source=data_source,
                dag=CausalDAGModel(),
                statistical_significance=False,
                refutation=RefutationSummary(),
                warnings=["Analysis failed due to an internal error."],
                latency_ms=int((_time.time() - start) * 1000),
            ),
        )


def _opt_float(value: Any) -> Optional[float]:
    """Coerce a refuter field to a float, or None if absent/non-numeric."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _refutation_tests_from_state(refutation: Dict[str, Any]) -> List[RefutationTestDetail]:
    """Map the agent's ``refutation_results['individual_tests']`` dict onto the
    per-test detail list the drill-down table renders.

    ``individual_tests`` is keyed by the CONTRACT test name (placebo_treatment,
    random_common_cause, data_subset, unobserved_common_cause, bootstrap); each
    value carries ``passed``/``original_effect``/``new_effect``/``p_value``/
    ``details``. Returns [] when refutation did not run (the FE then shows the
    honest 'refutation did not run' state rather than a misleading prompt).

    We surface the DICT KEY as ``test_name``, not the inner ``test_name`` field:
    to_legacy_format keys the sensitivity test under ``unobserved_common_cause``
    but sets its inner test_name to the raw enum ``sensitivity_e_value`` — using
    the inner value would make the FE fall back to the wrong label ("Random
    Common Cause") and duplicate that row. The key is the name the FE maps on.
    """
    individual = refutation.get("individual_tests")
    if not isinstance(individual, dict):
        return []
    tests: List[RefutationTestDetail] = []
    for key, t in individual.items():
        if not isinstance(t, dict):
            continue
        # Canonical (contract) name = the dict key; fall back to the inner field
        # only if the key is somehow empty.
        name = str(key) if key else str(t.get("test_name") or "")
        tests.append(
            RefutationTestDetail(
                test_name=name,
                passed=bool(t.get("passed", False)),
                original_effect=_opt_float(t.get("original_effect")),
                new_effect=_opt_float(t.get("new_effect")),
                p_value=_opt_float(t.get("p_value")),
                details=(str(t["details"]) if t.get("details") else None),
            )
        )
    return tests


def _agent_state_to_response(
    *,
    analysis_id: str,
    request: AgentCausalAnalysisRequest,
    data_source: str,
    n_rows: int,
    final_state: Dict[str, Any],
    latency_ms: int,
) -> AgentCausalAnalysisResponse:
    """Map the causal_impact agent's final state onto the API response.

    Fail-closed status mirrors the agent's own gate (CausalImpactAgent
    ._build_output): a run is ``completed`` only with a real ATE, a non-blocked
    refutation gate, and no sensitivity failure; ``review`` band is surfaced as
    ``needs_review``; anything else is ``failed`` with the reason in warnings.
    """
    causal_graph = final_state.get("causal_graph") or {}
    estimation = final_state.get("estimation_result") or {}
    refutation = final_state.get("refutation_results") or {}
    sensitivity = final_state.get("sensitivity_analysis") or {}
    interpretation = final_state.get("interpretation") or {}

    dag = CausalDAGModel(
        nodes=list(causal_graph.get("nodes", []) or []),
        edges=[list(e) for e in (causal_graph.get("edges", []) or []) if len(e) == 2],
        treatment_nodes=list(causal_graph.get("treatment_nodes", []) or []),
        outcome_nodes=list(causal_graph.get("outcome_nodes", []) or []),
        adjustment_sets=[list(s) for s in (causal_graph.get("adjustment_sets", []) or [])],
        dag_dot=causal_graph.get("dag_dot"),
    )

    # How was the DAG built? 'discovered' = learned from data (guided structure
    # discovery accepted by the gate), 'augmented' = domain DAG + discovered
    # edges, 'domain_knowledge' = the agent's curated DAG (discovery skipped or
    # not accepted). discovery_result is present only when discovery ran.
    discovery_ran = final_state.get("discovery_result") is not None
    _gate_dec = causal_graph.get("discovery_gate_decision")
    if discovery_ran and _gate_dec == "accept":
        dag_source = "discovered"
    elif discovery_ran and _gate_dec == "augment":
        dag_source = "augmented"
    else:
        dag_source = "domain_knowledge"
    # Confounders the DATA identified (the backdoor adjustment set) — only
    # surfaced when the structure was actually learned from data.
    _adj_sets = causal_graph.get("adjustment_sets", []) or []
    discovered_confounders = (
        [str(c) for c in _adj_sets[0]]
        if dag_source in ("discovered", "augmented") and _adj_sets
        else []
    )

    ate = estimation.get("ate")
    gate_decision = refutation.get("gate_decision") or final_state.get("gate_decision")
    refutation_error = final_state.get("refutation_error")
    sensitivity_failed = bool(final_state.get("sensitivity_error"))
    refutation_ran = bool(refutation) and not refutation_error
    gate_blocked = gate_decision == "block"
    needs_review = gate_decision == "review"

    if ate is not None and refutation_ran and not gate_blocked and not sensitivity_failed:
        status = "needs_review" if needs_review else "completed"
    else:
        status = "failed"

    refutation_summary = RefutationSummary(
        gate_decision=gate_decision,
        passed=bool(refutation_ran and gate_decision == "proceed"),
        needs_review=needs_review,
        tests_passed=refutation.get("tests_passed"),
        tests_total=refutation.get("total_tests"),
        sensitivity_e_value=sensitivity.get("e_value"),
        # Surface the per-test refutation results so the drill-down renders the
        # full table (placebo / random-common-cause / data-subset / bootstrap),
        # not just the pass/total count. The agent always computes these in the
        # leaderboard path; they were previously dropped here.
        tests=_refutation_tests_from_state(refutation),
    )

    # Surface honest warnings when the run did not yield a usable, validated effect.
    warnings: List[str] = list(final_state.get("warnings", []) or [])
    if ate is None:
        warnings.append("No treatment effect was estimated (the agent fail-closed on the data).")
    if not refutation_ran:
        warnings.append("Refutation did not run — the effect is unvalidated.")
    elif gate_blocked:
        warnings.append("Refutation gate BLOCKED — the estimate did not survive robustness checks.")
    if sensitivity_failed:
        warnings.append("Sensitivity analysis failed — robustness is unvalidated.")

    return AgentCausalAnalysisResponse(
        analysis_id=analysis_id,
        status=status,
        treatment_var=request.treatment_var,
        outcome_var=request.outcome_var,
        dataset=request.dataset,
        n_rows=n_rows,
        data_source=data_source,
        dag=dag,
        dag_source=dag_source,
        discovered_confounders=discovered_confounders,
        ate=ate,
        ate_ci_lower=estimation.get("ate_ci_lower"),
        ate_ci_upper=estimation.get("ate_ci_upper"),
        standard_error=estimation.get("standard_error"),
        p_value=estimation.get("p_value"),
        statistical_significance=bool(estimation.get("statistical_significance", False)),
        selected_estimator=estimation.get("method") or estimation.get("selected_estimator"),
        confidence=final_state.get("overall_confidence"),
        refutation=refutation_summary,
        narrative=interpretation.get("narrative"),
        executive_summary=interpretation.get("executive_summary"),
        recommendations=list(interpretation.get("recommendations", []) or []),
        key_insights=list(interpretation.get("key_findings", []) or []),
        warnings=warnings,
        latency_ms=latency_ms,
    )


# =============================================================================
# PIPELINE ENDPOINTS
# =============================================================================


@router.post(
    "/pipeline/sequential",
    response_model=SequentialPipelineResponse,
    summary="Run sequential multi-library pipeline",
    operation_id="run_sequential_pipeline",
)
async def run_sequential_pipeline(
    request: SequentialPipelineRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(default=False, description="Run asynchronously"),
    demo_mode: bool = Query(
        default=False,
        description=(
            "If true, return pinned-zero placeholder results labeled with "
            "is_demo=true (for UI demonstrations only). Default is false: "
            "the endpoint runs real estimator selection or fails with 503."
        ),
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> SequentialPipelineResponse:
    """
    Run sequential multi-library pipeline.

    Executes causal analysis stages in sequence:
    NetworkX → DoWhy → EconML → CausalML

    Each stage can pass results to the next for validation and refinement.

    Args:
        request: Pipeline configuration
        background_tasks: FastAPI background tasks
        async_mode: If True, runs asynchronously
        demo_mode: If True, return clearly-labeled placeholder values

    Returns:
        SequentialPipelineResponse with stage results and consensus
    """
    pipeline_id = str(uuid.uuid4())
    time.time()

    logger.info(
        f"Sequential pipeline requested: {pipeline_id}",
        extra={
            "pipeline_id": pipeline_id,
            "stages": len(request.stages),
            "libraries": [s.library.value for s in request.stages],
            "demo_mode": demo_mode,
        },
    )

    if async_mode:
        # Return pending response
        pending_response = SequentialPipelineResponse(
            pipeline_id=pipeline_id,
            status=AnalysisStatus.PENDING,
            stages_completed=0,
            stages_total=len(request.stages),
            stage_results=[],
            consensus_effect=None,
            consensus_ci_lower=None,
            consensus_ci_upper=None,
            confidence_level=request.confidence_level,
            library_agreement_score=None,
            effect_estimate_variance=None,
            total_latency_ms=0,
            created_at=datetime.now(timezone.utc),
            warnings=[],
        )
        _pipeline_cache[pipeline_id] = pending_response.model_dump()
        background_tasks.add_task(_run_sequential_pipeline_task, pipeline_id, request, demo_mode)
        return pending_response

    # Synchronous execution
    try:
        result = await _execute_sequential_pipeline(pipeline_id, request, demo_mode=demo_mode)
        _pipeline_cache[pipeline_id] = result.model_dump()
        return result
    except HTTPException:
        raise
    except HeavyComputeSaturated:
        # Reject fast under load — surfaced as 503 + Retry-After by the app
        # exception handler (OOM guard). HeavyComputeSaturated is NOT an
        # HTTPException, so this must precede the broad handler below to avoid
        # being swallowed into a 500.
        raise
    except Exception as e:
        logger.error(f"Sequential pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_GENERIC_500_DETAIL) from e


async def _run_sequential_pipeline_task(
    pipeline_id: str,
    request: SequentialPipelineRequest,
    demo_mode: bool = False,
) -> None:
    """Background task for sequential pipeline."""
    try:
        result = await _execute_sequential_pipeline(pipeline_id, request, demo_mode=demo_mode)
        _pipeline_cache[pipeline_id] = result.model_dump()
    except Exception as e:
        # Log the raw error server-side (with traceback); the cached FAILED
        # record is later returned to clients, so it must carry only a generic
        # message, not raw exception text.
        logger.error(f"Background sequential pipeline failed: {e}", exc_info=True)
        _pipeline_cache[pipeline_id] = SequentialPipelineResponse(
            pipeline_id=pipeline_id,
            status=AnalysisStatus.FAILED,
            stages_completed=0,
            stages_total=len(request.stages),
            stage_results=[],
            consensus_effect=None,
            consensus_ci_lower=None,
            consensus_ci_upper=None,
            confidence_level=request.confidence_level,
            library_agreement_score=None,
            effect_estimate_variance=None,
            total_latency_ms=0,
            created_at=datetime.now(timezone.utc),
            warnings=["Pipeline failed due to an internal error."],
        ).model_dump()


# These are curated, exception-free explanations meant for end users, so they opt
# in to the global 503 handler surfacing them verbatim (the FE Heterogeneous
# Treatment Effects card matches "no real data backend" to render an honest
# "data isn't wired yet" state). Keep the wording in sync with that FE gate.
_NO_REAL_DATA_BACKEND_DETAIL = user_safe_503_detail(
    "Causal pipeline endpoints have no real data backend wired. "
    "There is no production data source returning treatment/outcome columns by name. "
    "Pass demo_mode=true to get a clearly-labeled pinned-zero placeholder for UI demos, "
    "or wire real data and re-issue the request."
)

_NO_RESOLVABLE_DATA_DETAIL = user_safe_503_detail(
    "Sequential/parallel pipeline executed but no library produced a result: "
    "no DataFrame was resolvable from the request filters and there is no "
    "production data backend wired for arbitrary data_source identifiers. "
    "Supply inline data via filters.estimation_data_records (list of dicts with "
    "treatment / outcome / covariate columns), or pass demo_mode=true for the "
    "clearly-labeled pinned-zero placeholder used in UI demos."
)

# Libraries that REQUIRE a DataFrame to produce a real causal estimate.
# NetworkX is excluded because it is a symbolic-input graph executor (see
# C-5 design spike) — it can succeed with only variable names and an
# upstream `state['causal_graph']`. If a request includes any of these
# data-required libraries AND none of them succeed, we fail-close even
# when NetworkX succeeded, because the pipeline did not answer the
# causal-effect question the user asked.
_DATA_REQUIRED_LIBRARIES: frozenset[str] = frozenset({"dowhy", "econml", "causalml"})


# =============================================================================
# #354 C-8: real-pipeline wiring helpers
# =============================================================================


def _resolve_pipeline_dataframe(
    filters: Optional[Dict[str, Any]],
) -> Optional["pd.DataFrame"]:  # type: ignore[name-defined] # noqa: F821
    """Rehydrate an estimation DataFrame from request filters.

    Surface C accepts a DataFrame only via inline JSON-serialized records in
    ``filters.estimation_data_records``. This preserves the existing schema
    (``filters: Optional[Dict[str, Any]]``) without forcing a separate file
    upload surface. Returns ``None`` when no DataFrame can be rehydrated —
    the caller fail-closes with 503.

    Per CLAUDE.md anti-mocking discipline: this helper does NOT manufacture
    synthetic data when no DataFrame is provided. The 503 fail-close path
    is the honest response when the data backend is absent.
    """
    import pandas as pd

    if not isinstance(filters, dict):
        return None
    records = filters.get("estimation_data_records")
    if not isinstance(records, list) or not records:
        return None
    try:
        df = pd.DataFrame.from_records(records)
    except Exception:  # noqa: BLE001 - any rehydration failure → fail-close
        return None
    if df.empty:
        return None
    return df


def _resolve_hierarchical_dataframe(
    request: HierarchicalAnalysisRequest,
) -> "pd.DataFrame":  # type: ignore[name-defined] # noqa: F821
    """Resolve a real estimation DataFrame for hierarchical analysis, or raise.

    Fail-closed (C1): raises ``HTTPException(503)`` when no inline data is
    present and ``HTTPException(400)`` when the required treatment / outcome /
    effect-modifier columns are missing. NEVER fabricates synthetic input.
    Shared by the sync execute path and the async-submission preflight so both
    enforce the identical contract.
    """
    df = _resolve_pipeline_dataframe(request.filters)
    if df is None:
        raise HTTPException(status_code=503, detail=_NO_REAL_DATA_BACKEND_DETAIL)
    required_cols = [
        request.treatment_var,
        request.outcome_var,
        *request.effect_modifiers,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "estimation_data_records is missing required column(s): "
                f"{missing}. Supply treatment / outcome / effect-modifier "
                "columns as record keys."
            ),
        )
    return df


def _build_pipeline_input_sequential(
    request: SequentialPipelineRequest,
    *,
    libraries_enabled: Optional[List[str]] = None,
) -> PipelineInput:
    """Construct a PipelineInput for SequentialPipeline.execute() from request.

    The DataFrame is conveyed via the first-class ``PipelineInput.estimation_data``
    field (#458). The orchestrator copies it into ``state["estimation_data"]``
    and every executor resolves it via ``resolve_estimation_dataframe(state)``.
    ``request.filters`` (which carries inline-record passthrough and any
    DoWhy method override) is forwarded unchanged.
    """
    df = _resolve_pipeline_dataframe(request.filters)
    request_filters: Dict[str, Any] = dict(request.filters or {})

    return PipelineInput(
        query=(
            f"Sequential pipeline: treatment={request.treatment_var}, outcome={request.outcome_var}"
        ),
        treatment_var=request.treatment_var,
        outcome_var=request.outcome_var,
        confounders=list(request.covariates),
        effect_modifiers=None,
        data_source=request.data_source,
        filters=request_filters,
        estimation_data=df,
        mode="sequential",
        libraries_enabled=libraries_enabled,
        cross_validate=None,
        # R6-F1 (#740): opt-in real-refutation flag → state["config"]["run_refutation"].
        run_refutation=request.run_refutation,
    )


def _build_pipeline_input_parallel(
    request: ParallelPipelineRequest,
) -> PipelineInput:
    """Construct a PipelineInput for ParallelPipeline.execute() from request."""
    df = _resolve_pipeline_dataframe(request.filters)
    request_filters: Dict[str, Any] = dict(request.filters or {})

    return PipelineInput(
        query=(
            f"Parallel pipeline: treatment={request.treatment_var}, outcome={request.outcome_var}"
        ),
        treatment_var=request.treatment_var,
        outcome_var=request.outcome_var,
        confounders=list(request.covariates),
        effect_modifiers=None,
        data_source=request.data_source,
        filters=request_filters,
        estimation_data=df,
        mode="parallel",
        libraries_enabled=[lib.value for lib in request.libraries],
        cross_validate=None,
        # R6-F1 (#740): opt-in real-refutation flag → state["config"]["run_refutation"].
        run_refutation=request.run_refutation,
    )


class _SurfaceCSequentialPipeline(SequentialPipeline):
    """SequentialPipeline subclass for Surface C wiring.

    Provides one C-8-specific extension on top of the base orchestrator:

    **Per-library result capture** (``self.last_state``). The base
    ``execute()`` returns ``PipelineOutput`` which only carries the
    primary library's full payload. The C-8 response builder needs every
    executed library's payload (to populate per-library stage results /
    library_results without dropping data) — so we capture the final
    state in ``_create_output`` for the adapter to read.

    DataFrame injection into ``state["data_cache"]`` was a separate concern
    handled by an earlier ``dataframe=`` constructor kwarg + a
    ``_create_initial_state`` override. That mechanism is gone as of #458:
    the DataFrame now travels through ``PipelineInput.estimation_data`` and
    the orchestrator copies it into ``state["estimation_data"]`` itself,
    so this subclass no longer touches initial state.
    """

    def __init__(self, *, fail_fast: bool = False) -> None:
        super().__init__(fail_fast=fail_fast)
        self.last_state: Optional[PipelineState] = None

    def _create_output(self, state: PipelineState) -> PipelineOutput:
        # Capture the final state so the adapter can read per-library results
        # (state["<lib>_result"]) — PipelineOutput.primary_result only carries
        # the primary library's payload, which would drop non-primary library
        # data from the API response.
        self.last_state = state
        return super()._create_output(state)


class _SurfaceCParallelPipeline(ParallelPipeline):
    """ParallelPipeline subclass that mirrors ``_SurfaceCSequentialPipeline``.

    Captures ``self.last_state`` for per-library result extraction; DataFrame
    conveyance is via ``PipelineInput.estimation_data`` (#458), not a
    constructor kwarg.
    """

    def __init__(
        self,
        *,
        max_parallel: int = 4,
        fail_fast: bool = False,
    ) -> None:
        super().__init__(max_parallel=max_parallel, fail_fast=fail_fast)
        self.last_state: Optional[PipelineState] = None

    def _create_output(self, state: PipelineState) -> PipelineOutput:
        self.last_state = state
        return super()._create_output(state)


async def _run_real_sequential_pipeline(
    pipeline_id: str,
    request: SequentialPipelineRequest,
) -> SequentialPipelineResponse:
    """Invoke the wired SequentialPipeline.execute() and adapt to the API response.

    Fail-closed contract:
        - If no library produced a successful result (every executor returned
          ``success=False`` because no DataFrame was resolvable from state),
          raises ``HTTPException(503)`` — the honest signal that the data
          backend is absent for this request.
        - If at least one library executes successfully, returns a real
          ``SequentialPipelineResponse`` constructed from per-library state
          (no hardcoded values).
    """
    pipeline = _SurfaceCSequentialPipeline(
        fail_fast=request.stop_on_failure,
    )
    libraries_enabled = [stage.library.value for stage in request.stages]
    pipeline_input = _build_pipeline_input_sequential(request, libraries_enabled=libraries_enabled)
    output = await pipeline.execute(pipeline_input)
    return _sequential_output_to_response(pipeline_id, request, output, state=pipeline.last_state)


async def _run_real_parallel_pipeline(
    pipeline_id: str,
    request: ParallelPipelineRequest,
) -> ParallelPipelineResponse:
    """Invoke the wired ParallelPipeline.execute() and adapt to the API response."""
    pipeline = _SurfaceCParallelPipeline(
        max_parallel=len(request.libraries),
        fail_fast=False,
    )
    pipeline_input = _build_pipeline_input_parallel(request)
    output = await pipeline.execute(pipeline_input)
    return _parallel_output_to_response(pipeline_id, request, output, state=pipeline.last_state)


def _resolve_graph_quality(
    output: Optional[Mapping[str, Any]],
    state: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve the structural graph-quality dict from state (preferred) or output.

    The real pipeline carries graph_quality on ``state`` (orchestrator
    ``_extract_graph_quality``); ``PipelineOutput`` does NOT copy it. The existing
    M-fo2 surface test injects it via ``output``. Prefer ``state`` when present
    (the real path) and fall back to ``output`` (synthetic/test path). Returns an
    empty dict (not None) so callers can ``.get(...)`` safely.
    """
    for source in (state, output):
        if source is None:
            continue
        gq = source.get("graph_quality")
        if isinstance(gq, dict):
            return gq
    return {}


def _robustness_from_state(
    state: Optional[Mapping[str, Any]],
) -> tuple[bool, Optional[str]]:
    """Gate ``robustness_validation_performed`` on the REAL refutation gate band.

    Mirrors the agent path's ``is_estimate_valid`` (PROCEED is usable-as-robust;
    REVIEW/BLOCK are not) and its caveat semantics (nodes/refutation.py):

    - refutation_results falsy / empty / ``skipped`` / no gate → ``(False,
      _ROBUSTNESS_UNVALIDATED_WARNING)`` (today's default-path behaviour; an
      honest skip is NOT a validation).
    - ``gate_decision == "proceed"`` → ``(True, None)`` (validated, no caveat).
    - ``gate_decision == "review"`` → ``(False, _ROBUSTNESS_REVIEW_WARNING)``.
    - ``gate_decision in {"block", "error"}`` or an ``error`` key → ``(False,
      _ROBUSTNESS_BLOCK_WARNING)`` (fail-closed: an errored/blocked refutation
      must NEVER flip robustness True).

    NOTE: the M-fo2 non-DAG structural override is applied by the response
    builders AFTER this helper (a cyclic graph forces False regardless of band).
    """
    rr = (state or {}).get("refutation_results")
    if not isinstance(rr, dict) or not rr or rr.get("skipped") is True:
        return False, _ROBUSTNESS_UNVALIDATED_WARNING
    if rr.get("error") is not None:
        return False, _ROBUSTNESS_BLOCK_WARNING
    gate = rr.get("gate_decision")
    if gate == "proceed":
        return True, None
    if gate == "review":
        return False, _ROBUSTNESS_REVIEW_WARNING
    if gate == "block":
        return False, _ROBUSTNESS_BLOCK_WARNING
    # Unknown / missing gate on a populated dict → fail-closed unvalidated.
    return False, _ROBUSTNESS_UNVALIDATED_WARNING


def _classify_structural_identification(
    graph_quality: Mapping[str, Any],
) -> tuple[Optional[str], bool]:
    """M-fo2 (precise): return ``(structural_identification, identification_blocked)``.

    Reads the precise fields the orchestrator stamps onto ``graph_quality``:

    - ``is_dag`` None/missing → the structural check did not run → ``(None, False)``.
    - ``is_dag`` True → ``("acyclic", False)``.
    - ``is_dag`` False → use ``structural_identification`` when present; otherwise
      derive from ``cycle_affects_identification`` and FAIL CLOSED (a non-DAG that
      lacks the precise flag is treated as ``undefined_cyclic``).
    """
    is_dag = graph_quality.get("is_dag")
    if is_dag is None:
        return None, False
    if is_dag is True:
        return "acyclic", False
    # is_dag is False — read the precise label, fail-closed on absence.
    label = graph_quality.get("structural_identification")
    if label in ("undefined_cyclic", "cycle_irrelevant"):
        return label, (label == "undefined_cyclic")
    affects = graph_quality.get("cycle_affects_identification")
    if affects is None:
        affects = True  # conservative: a non-DAG without the precise flag
    return ("undefined_cyclic" if affects else "cycle_irrelevant"), bool(affects)


class _StructuralGateOutcome(NamedTuple):
    """Result of the M-fo2 structural-identifiability gate applied by the builders."""

    robustness_performed: bool
    robustness_warning: Optional[str]
    warnings: List[str]
    requires_review: bool
    structural_identification: Optional[str]
    withhold_consensus: bool


def _apply_structural_identification_gate(
    *,
    graph_quality: Mapping[str, Any],
    robustness_performed: bool,
    robustness_warning: Optional[str],
    warnings: List[str],
) -> _StructuralGateOutcome:
    """M-fo2 (precise): quarantine ONLY when a cycle actually breaks identification.

    - ``undefined_cyclic`` (a directed cycle on the (T,Y) ancestral subgraph):
      backdoor adjustment is undefined → FORCE robustness False (overrides any
      PROCEED, a downgrade not a 503, per F1 Owner-decision 2), set
      ``requires_review=True``, WITHHOLD the consensus effect, and append an
      un-ignorable caveat to BOTH the warnings list and robustness_warning.
    - ``cycle_irrelevant`` (a cycle OFF the ancestral subgraph): the estimand is
      still identifiable → no robustness override, consensus preserved, only an
      informational warning.
    - ``acyclic`` / not-run: unchanged.
    """
    structural_identification, blocked = _classify_structural_identification(graph_quality)
    new_warnings = list(warnings)

    if blocked:
        if _NON_DAG_STRUCTURAL_WARNING not in new_warnings:
            new_warnings.append(_NON_DAG_STRUCTURAL_WARNING)
        if robustness_warning and _NON_DAG_STRUCTURAL_WARNING not in robustness_warning:
            combined: Optional[str] = f"{robustness_warning} {_NON_DAG_STRUCTURAL_WARNING}"
        else:
            combined = _NON_DAG_STRUCTURAL_WARNING
        return _StructuralGateOutcome(
            robustness_performed=False,
            robustness_warning=combined,
            warnings=new_warnings,
            requires_review=True,
            structural_identification=structural_identification,
            withhold_consensus=True,
        )

    if structural_identification == "cycle_irrelevant" and _CYCLE_IRRELEVANT_WARNING not in (
        new_warnings
    ):
        new_warnings.append(_CYCLE_IRRELEVANT_WARNING)

    return _StructuralGateOutcome(
        robustness_performed=robustness_performed,
        robustness_warning=robustness_warning,
        warnings=new_warnings,
        requires_review=False,
        structural_identification=structural_identification,
        withhold_consensus=False,
    )


def _sequential_output_to_response(
    pipeline_id: str,
    request: SequentialPipelineRequest,
    output: PipelineOutput,
    *,
    state: Optional[PipelineState] = None,
) -> SequentialPipelineResponse:
    """Adapt PipelineOutput → SequentialPipelineResponse.

    Builds one PipelineStageResult per requested stage, honoring the request's
    stage order. Fails closed with 503 when no library produced a *successful*
    result — see ``_run_real_sequential_pipeline`` for the contract.

    Note on "successful library" derivation:
        The engine appends to ``state["libraries_executed"]`` on every call
        (orchestrator.py:227), regardless of ``success``. The engine's failed
        executors also populate ``state["errors"]`` with one entry per failure.
        So a library is "successful" only if it appears in ``libraries_used``
        AND is NOT named in ``errors``.

    Note on per-library payload extraction:
        ``PipelineOutput.primary_result`` only carries the primary library's
        full payload. For non-primary stages we read from
        ``state["<lib>_result"]["result"]`` (via ``_extract_library_payload``)
        so we never silently drop a successful non-primary library's data.
    """
    libraries_used = list(output.get("libraries_used") or [])
    errors = output.get("errors") or []
    failed_libraries = {
        str(err.get("library")) for err in errors if isinstance(err, dict) and err.get("library")
    }
    successful_libraries = [lib for lib in libraries_used if lib not in failed_libraries]

    requested_libraries = [stage.library.value for stage in request.stages]
    _enforce_data_required_fail_close(requested_libraries, successful_libraries)

    error_by_library: Dict[str, str] = {
        str(err.get("library")): str(err.get("error") or "")
        for err in errors
        if isinstance(err, dict) and err.get("library")
    }

    stage_results: List[PipelineStageResult] = []
    for idx, stage_config in enumerate(request.stages, 1):
        lib_value = stage_config.library.value
        stage_results.append(
            _build_stage_result_from_output(
                stage_number=idx,
                stage_config_library=lib_value,
                stage_config_estimator=stage_config.estimator,
                output=output,
                state=state,
                successful_libraries=successful_libraries,
                error_by_library=error_by_library,
            )
        )

    stages_completed = sum(1 for r in stage_results if r.status == AnalysisStatus.COMPLETED)

    # M-fo2: read structural graph-quality from the (optional) mapping, guarding
    # the non-dict case so mypy narrows the type and a malformed value yields None.
    graph_quality = _resolve_graph_quality(output, state)

    # R6-F1: gate robustness on the REAL refutation suite (PROCEED → validated;
    # REVIEW/BLOCK/error/skipped/empty → False + a band-naming caveat). Only
    # append the caveat to ``warnings`` when one is set (a validated PROCEED has
    # no caveat — drop the "unvalidated" line entirely).
    robustness_performed, robustness_warning = _robustness_from_state(state)
    warnings = list(output.get("warnings") or [])
    if robustness_warning:
        warnings.append(robustness_warning)

    # M-fo2 (precise): quarantine ONLY when a cycle breaks identification of the
    # (T,Y) estimand (cycle on the ancestral subgraph). undefined_cyclic FORCES
    # robustness False + requires_review + WITHHOLDS the consensus; an off-subgraph
    # cycle (cycle_irrelevant) leaves the estimate untouched.
    gate = _apply_structural_identification_gate(
        graph_quality=graph_quality,
        robustness_performed=robustness_performed,
        robustness_warning=robustness_warning,
        warnings=warnings,
    )
    consensus_effect = None if gate.withhold_consensus else output.get("consensus_effect")

    return SequentialPipelineResponse(
        pipeline_id=pipeline_id,
        status=_derive_response_status(stages_completed, len(request.stages)),
        stages_completed=stages_completed,
        stages_total=len(request.stages),
        stage_results=stage_results,
        consensus_effect=consensus_effect,
        consensus_ci_lower=None,  # Not produced by the engine output today
        consensus_ci_upper=None,
        # #27: report the level the consensus CI WOULD use (CI itself None today).
        confidence_level=request.confidence_level,
        # H8: a REAL library-agreement metric (mean pairwise concordance), NOT
        # consensus_confidence (the mean of per-library confidences, which the API
        # previously mislabeled as agreement).
        library_agreement_score=(state.get("library_agreement_score") if state else None),
        effect_estimate_variance=None,
        total_latency_ms=int(output.get("total_latency_ms") or 0),
        created_at=datetime.now(timezone.utc),
        warnings=gate.warnings,
        robustness_validation_performed=gate.robustness_performed,
        robustness_warning=gate.robustness_warning,
        graph_is_dag=graph_quality.get("is_dag"),
        structural_quality=graph_quality.get("structural_quality"),
        requires_review=gate.requires_review,
        structural_identification=gate.structural_identification,
    )


def _parallel_output_to_response(
    pipeline_id: str,
    request: ParallelPipelineRequest,
    output: PipelineOutput,
    *,
    state: Optional[PipelineState] = None,
) -> ParallelPipelineResponse:
    """Adapt PipelineOutput → ParallelPipelineResponse.

    Fails closed with 503 when no library produced a successful result.
    Same "successful library" derivation and per-library payload reading
    as the sequential adapter — see its docstring for the rationale.
    """
    libraries_used = list(output.get("libraries_used") or [])
    errors = output.get("errors") or []
    error_by_library: Dict[str, str] = {
        str(err.get("library")): str(err.get("error") or "")
        for err in errors
        if isinstance(err, dict) and err.get("library")
    }
    successful_libraries = [lib for lib in libraries_used if lib not in error_by_library]

    requested_libraries = [lib.value for lib in request.libraries]
    _enforce_data_required_fail_close(requested_libraries, successful_libraries)

    library_results: Dict[str, Dict[str, Any]] = {}
    succeeded: List[str] = []
    failed: List[str] = []

    for lib in request.libraries:
        lib_value = lib.value
        if lib_value in successful_libraries:
            succeeded.append(lib_value)
            library_results[lib_value] = _extract_library_payload(lib_value, output, state=state)
        elif lib_value in error_by_library:
            failed.append(lib_value)
            library_results[lib_value] = {"error": error_by_library[lib_value]}
        else:
            # Library was requested but neither executed nor errored —
            # validate_input rejected it before run.
            failed.append(lib_value)
            library_results[lib_value] = {"error": "library skipped during execution"}

    # M-fo2: read structural graph-quality from state (real path) or output
    # (synthetic/test path); empty dict when absent so .get(...) is safe.
    graph_quality = _resolve_graph_quality(output, state)

    # R6-F1: gate robustness on the REAL refutation suite (see sequential builder).
    robustness_performed, robustness_warning = _robustness_from_state(state)
    warnings = list(output.get("warnings") or [])
    if robustness_warning:
        warnings.append(robustness_warning)

    # M-fo2 (precise): see the sequential builder. undefined_cyclic forces
    # robustness False + requires_review + withholds the consensus; cycle_irrelevant
    # leaves the estimate untouched.
    gate = _apply_structural_identification_gate(
        graph_quality=graph_quality,
        robustness_performed=robustness_performed,
        robustness_warning=robustness_warning,
        warnings=warnings,
    )
    consensus_effect = None if gate.withhold_consensus else output.get("consensus_effect")

    return ParallelPipelineResponse(
        pipeline_id=pipeline_id,
        status=(
            AnalysisStatus.COMPLETED
            if len(succeeded) == len(request.libraries)
            else AnalysisStatus.FAILED
        ),
        libraries_succeeded=succeeded,
        libraries_failed=failed,
        library_results=library_results,
        consensus_effect=consensus_effect,
        consensus_ci_lower=None,
        consensus_ci_upper=None,
        # #27: report the confidence level the consensus CI WOULD use. The real
        # engine does not emit a consensus CI today (lower/upper stay None), but
        # echoing the requested level keeps the field consistent with the demo
        # path and lets the UI label any future interval truthfully.
        confidence_level=request.confidence_level,
        # H8: real mean-pairwise-concordance agreement, not consensus_confidence.
        library_agreement_score=(state.get("library_agreement_score") if state else None),
        consensus_method=request.consensus_method,
        total_latency_ms=int(output.get("total_latency_ms") or 0),
        created_at=datetime.now(timezone.utc),
        warnings=gate.warnings,
        robustness_validation_performed=gate.robustness_performed,
        robustness_warning=gate.robustness_warning,
        graph_is_dag=graph_quality.get("is_dag"),
        structural_quality=graph_quality.get("structural_quality"),
        requires_review=gate.requires_review,
        structural_identification=gate.structural_identification,
    )


def _derive_response_status(stages_completed: int, stages_total: int) -> AnalysisStatus:
    """Derive API AnalysisStatus from completed/total stage counts."""
    if stages_completed == stages_total:
        return AnalysisStatus.COMPLETED
    return AnalysisStatus.FAILED


def _enforce_data_required_fail_close(
    requested_libraries: List[str],
    successful_libraries: List[str],
) -> None:
    """Fail-close with 503 when no library produced an answer to the question asked.

    Two fail-close conditions, both honest signals of "pipeline did not answer":

    1. **No library succeeded** — every executor returned success=False
       (typically because no DataFrame was resolvable from state).
    2. **Only symbolic-input libraries succeeded** when an effect question
       was asked — i.e. the request named at least one of
       ``_DATA_REQUIRED_LIBRARIES`` (dowhy/econml/causalml — the libraries
       that produce causal effect estimates) AND none of them succeeded.
       NetworkX alone cannot answer "what is the causal effect?" — it
       answers "what is the graph structure?". Returning 200 with only
       NetworkX in this case would be a labeling problem (succeeded =
       True; answered effect question = False).

    All-symbolic requested sets intentionally bypass this fail-close —
    a graph-only question is a valid use case and NetworkX is the
    canonical answer. (Note: today the request schemas enforce
    ``min_length=2`` on ``stages`` / ``libraries`` (see
    ``api/schemas/causal.py``), so a literal NetworkX-only API request
    would be rejected by Pydantic validation before reaching this
    helper. The bypass still matters for any future schema relaxation
    and for the symbolic-only path inside this helper.)

    Raises:
        HTTPException(503): with ``_NO_RESOLVABLE_DATA_DETAIL`` body.
    """
    if not successful_libraries:
        raise HTTPException(status_code=503, detail=_NO_RESOLVABLE_DATA_DETAIL)

    requested_data_required = {
        lib for lib in requested_libraries if lib in _DATA_REQUIRED_LIBRARIES
    }
    if requested_data_required:
        successful_data_required = {
            lib for lib in successful_libraries if lib in _DATA_REQUIRED_LIBRARIES
        }
        if not successful_data_required:
            # User asked for at least one effect-estimating library, none
            # succeeded. NetworkX's symbolic success doesn't answer the
            # effect question — fail-close.
            raise HTTPException(status_code=503, detail=_NO_RESOLVABLE_DATA_DETAIL)


def _build_stage_result_from_output(
    *,
    stage_number: int,
    stage_config_library: str,
    stage_config_estimator: Optional[str],
    output: PipelineOutput,
    state: Optional[PipelineState],
    successful_libraries: List[str],
    error_by_library: Dict[str, str],
) -> PipelineStageResult:
    """Build a PipelineStageResult for one stage.

    Reads real values from per-library result captured in ``state`` (so
    non-primary library payloads are not dropped). Marks the stage FAILED
    with the engine's descriptive error when the library did not succeed.
    """
    if stage_config_library not in successful_libraries:
        return PipelineStageResult(
            stage_number=stage_number,
            library=stage_config_library,
            estimator=stage_config_estimator,
            status=AnalysisStatus.FAILED,
            effect_estimate=None,
            ci_lower=None,
            ci_upper=None,
            p_value=None,
            additional_results={},
            latency_ms=0,
            error=error_by_library.get(
                stage_config_library,
                "library skipped or failed during pipeline execution",
            ),
        )

    payload = _extract_library_payload(stage_config_library, output, state=state)
    effect = payload.get("effect_estimate")
    ci_lower = payload.get("ci_lower")
    ci_upper = payload.get("ci_upper")
    p_value = payload.get("p_value")
    stage_latency = _get_stage_latency_ms(state, stage_config_library, output)

    return PipelineStageResult(
        stage_number=stage_number,
        library=stage_config_library,
        estimator=stage_config_estimator,
        status=AnalysisStatus.COMPLETED,
        effect_estimate=effect if isinstance(effect, (int, float)) else None,
        ci_lower=ci_lower if isinstance(ci_lower, (int, float)) else None,
        ci_upper=ci_upper if isinstance(ci_upper, (int, float)) else None,
        p_value=p_value if isinstance(p_value, (int, float)) else None,
        additional_results={
            k: v
            for k, v in payload.items()
            if k not in {"effect_estimate", "ci_lower", "ci_upper", "p_value"}
        },
        latency_ms=stage_latency,
        error=None,
    )


def _get_stage_latency_ms(
    state: Optional[PipelineState], library: str, output: PipelineOutput
) -> int:
    """Per-stage latency from state.stage_latencies, falling back to total."""
    if state is not None:
        stage_latencies = cast(Dict[str, Any], state).get("stage_latencies") or {}
        if isinstance(stage_latencies, dict):
            v = stage_latencies.get(library)
            if isinstance(v, (int, float)):
                return int(v)
    # Fallback: total latency (still real, just less granular)
    return int(output.get("total_latency_ms") or 0)


def _extract_library_payload(
    library: str,
    output: PipelineOutput,
    *,
    state: Optional[PipelineState] = None,
) -> Dict[str, Any]:
    """Extract a per-library payload for the API response.

    Resolution order:
        1. If ``state`` is provided AND ``state["<lib>_result"]`` is a
           success-flagged ``LibraryExecutionResult``, read its ``result``
           dict (the canonical per-library payload from the executor).
        2. Else if ``library`` is the primary library, fall back to
           ``output["primary_result"]``.
        3. Else return ``{"library": library}`` (the engine surfaced no
           per-library payload — this is a labeling honest minimum).

    Reading state first matters in parallel mode: when EconML (primary)
    fails and DoWhy (secondary) succeeds, ``output.primary_result`` is
    EconML's empty/error payload — reading it for DoWhy would drop the
    real DoWhy data. The per-library state fields preserve every executor's
    result.
    """
    result_payload = _read_library_result_from_state(library, state)
    if result_payload is None:
        # Fall back to primary_result when state is unavailable (defensive
        # path; the C-8 route always captures state).
        primary_lib = _output_primary_library(state, output)
        if primary_lib == library:
            result_payload = dict(output.get("primary_result") or {})
        else:
            result_payload = {}

    payload: Dict[str, Any] = {"library": library}

    if library == "dowhy":
        effect = result_payload.get("causal_effect")
        if isinstance(effect, (int, float)):
            payload["effect_estimate"] = float(effect)
        method = result_payload.get("dowhy_method")
        if isinstance(method, str):
            payload["method"] = method
        estimand = result_payload.get("identified_estimand")
        if isinstance(estimand, str):
            payload["identified_estimand"] = estimand
    elif library == "econml":
        ate = result_payload.get("ate") or result_payload.get("overall_ate")
        if isinstance(ate, (int, float)):
            payload["effect_estimate"] = float(ate)
        ci_lower = result_payload.get("ate_ci_lower") or result_payload.get("ci_lower")
        ci_upper = result_payload.get("ate_ci_upper") or result_payload.get("ci_upper")
        if isinstance(ci_lower, (int, float)):
            payload["ci_lower"] = float(ci_lower)
        if isinstance(ci_upper, (int, float)):
            payload["ci_upper"] = float(ci_upper)
        method = result_payload.get("econml_method") or result_payload.get("estimator")
        if isinstance(method, str):
            payload["method"] = method
    elif library == "causalml":
        ate = result_payload.get("ate")
        if isinstance(ate, (int, float)):
            payload["effect_estimate"] = float(ate)
        auuc = result_payload.get("auuc")
        if isinstance(auuc, (int, float)):
            payload["auuc"] = float(auuc)
        qini = result_payload.get("qini")
        if isinstance(qini, (int, float)):
            payload["qini"] = float(qini)
    elif library == "networkx":
        n_nodes = result_payload.get("n_nodes")
        if isinstance(n_nodes, (int, float)):
            payload["n_nodes"] = int(n_nodes)
        n_edges = result_payload.get("n_edges")
        if isinstance(n_edges, (int, float)):
            payload["n_edges"] = int(n_edges)
        is_dag = result_payload.get("is_dag")
        if isinstance(is_dag, bool):
            payload["is_dag"] = is_dag

    return payload


def _read_library_result_from_state(
    library: str, state: Optional[PipelineState]
) -> Optional[Dict[str, Any]]:
    """Read ``state["<lib>_result"]["result"]`` when present AND success=True.

    Returns ``None`` when state is absent OR the library has no successful
    result. Per-library state keys: ``dowhy_result``, ``econml_result``,
    ``causalml_result``, ``networkx_result``.
    """
    if state is None:
        return None
    state_dict = cast(Dict[str, Any], state)
    key = f"{library}_result"
    lib_result = state_dict.get(key)
    if not isinstance(lib_result, dict):
        return None
    if not lib_result.get("success"):
        return None
    result = lib_result.get("result")
    if isinstance(result, dict):
        return dict(result)
    return None


def _output_primary_library(
    state: Optional[PipelineState], output: PipelineOutput
) -> Optional[str]:
    """Best-effort primary library lookup from state or output."""
    if state is not None:
        config = cast(Dict[str, Any], state).get("config") or {}
        if isinstance(config, dict):
            primary = config.get("primary_library")
            if isinstance(primary, str):
                return primary
    libraries_used = output.get("libraries_used") or []
    if libraries_used:
        return libraries_used[0]
    return None


def _demo_stage_placeholder(
    *,
    stage_number: int,
    library: str,
    estimator: Optional[str],
    latency_ms: int,
) -> PipelineStageResult:
    """Pinned-zero placeholder used for explicit demo_mode=True flows.

    Never returns RNG values; the caller (with demo_mode=True) is responsible
    for labeling the surrounding envelope with ``is_demo=true``.
    """
    return PipelineStageResult(
        stage_number=stage_number,
        library=library,
        estimator=estimator,
        status=AnalysisStatus.COMPLETED,
        effect_estimate=0.0,
        ci_lower=0.0,
        ci_upper=0.0,
        p_value=1.0,
        additional_results={
            "n_samples": 0,
            "method": estimator or "default",
            "is_demo": True,
        },
        latency_ms=latency_ms,
        error=None,
    )


async def _execute_sequential_pipeline(
    pipeline_id: str,
    request: SequentialPipelineRequest,
    demo_mode: bool = False,
) -> SequentialPipelineResponse:
    """Execute sequential pipeline stages.

    Default path (``demo_mode=False``, #354 C-8): delegates to the real
    ``SequentialPipeline.execute()`` wired in C-1..C-6 (all 4 executors —
    DoWhy/EconML/CausalML/NetworkX — and 4-library aggregation). The
    caller MUST supply a DataFrame via ``request.filters['estimation_data_records']``
    (list of dicts). If no DataFrame is resolvable and every wired executor
    therefore returns ``success=False``, the response is an honest
    ``HTTPException(503)`` — there is still no production data backend that
    can resolve arbitrary ``data_source`` identifiers to real columns by name.

    With ``demo_mode=True``: returns pinned-zero placeholder stage results
    clearly labeled with ``is_demo=true``. This UI-demo branch is unchanged
    from the F-005 contract (see v4 §2.3); C-8 preserves it verbatim.

    Pre-C-8 behavior (now superseded): the default path raised 503
    unconditionally. The 503 stays as the honest no-data signal, but it
    now reflects the wired pipeline's actual outcome rather than a
    hardcoded short-circuit. See F-005 audit iter-1 HIGH-1 for the prior
    synthetic-data-fabrication trap that #354 was opened to fix.
    """
    if not demo_mode:
        # #354 C-8: invoke the wired pipeline. The helper raises
        # HTTPException(503) with _NO_RESOLVABLE_DATA_DETAIL when no library
        # produced a result (honest fail-close), or returns a real response
        # built from the engine's PipelineOutput when at least one library
        # succeeded. NO silent fallback to synthetic data, NO hardcoded values.
        #
        # The real 4-library pipeline (DoWhy/EconML/CausalML/NetworkX) is the
        # genuinely heavy in-process compute. Bound it to ONE per-worker
        # heavy-compute slot (OOM guard, P1b) so concurrent real pipelines
        # cannot stack and OOM-kill the cgroup. A saturated worker raises
        # HeavyComputeSaturated on enter (mapped to 503 + Retry-After) — nothing
        # is queued. The demo path below does NO heavy work and is intentionally
        # left unbounded. Both callers (sync + background task) route through
        # here, so bounding once covers both.
        async with heavy_compute_slot():
            return await _run_real_sequential_pipeline(pipeline_id, request)

    start_time = time.time()
    stage_results: List[PipelineStageResult] = []
    effect_estimates: List[float] = []
    warnings: List[str] = [
        "demo_mode=true: results are pinned-zero placeholders with is_demo=true; "
        "do NOT use for decisions.",
    ]

    for i, stage_config in enumerate(request.stages, 1):
        stage_start = time.time()
        stage_result = _demo_stage_placeholder(
            stage_number=i,
            library=stage_config.library.value,
            estimator=stage_config.estimator,
            latency_ms=int((time.time() - stage_start) * 1000),
        )
        effect_estimates.append(0.0)
        stage_results.append(stage_result)

    # Demo consensus is zero by construction (all stages return 0.0).
    consensus_effect = 0.0
    consensus_ci_lower = 0.0
    consensus_ci_upper = 0.0
    agreement_score = 1.0
    variance = 0.0

    total_latency_ms = int((time.time() - start_time) * 1000)
    stages_completed = len([r for r in stage_results if r.status == AnalysisStatus.COMPLETED])

    return SequentialPipelineResponse(
        pipeline_id=pipeline_id,
        status=AnalysisStatus.COMPLETED
        if stages_completed == len(request.stages)
        else AnalysisStatus.FAILED,
        stages_completed=stages_completed,
        stages_total=len(request.stages),
        stage_results=stage_results,
        consensus_effect=consensus_effect,
        consensus_ci_lower=consensus_ci_lower,
        consensus_ci_upper=consensus_ci_upper,
        confidence_level=request.confidence_level,
        library_agreement_score=agreement_score,
        effect_estimate_variance=variance,
        total_latency_ms=total_latency_ms,
        created_at=datetime.now(timezone.utc),
        warnings=warnings,
    )


@router.post(
    "/pipeline/parallel",
    response_model=ParallelPipelineResponse,
    summary="Run parallel multi-library analysis",
    operation_id="run_parallel_pipeline",
)
async def run_parallel_pipeline(
    request: ParallelPipelineRequest,
    demo_mode: bool = Query(
        default=False,
        description=(
            "If true, return pinned-zero placeholder results labeled with "
            "is_demo=true (for UI demonstrations only). Default is false: "
            "the endpoint runs real estimator selection or fails with 503."
        ),
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> ParallelPipelineResponse:
    """
    Run parallel multi-library analysis.

    Executes multiple causal libraries simultaneously and computes
    consensus results weighted by confidence.

    Args:
        request: Parallel pipeline configuration
        demo_mode: If True, return clearly-labeled placeholder values

    Returns:
        ParallelPipelineResponse with library results and consensus
    """
    pipeline_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        f"Parallel pipeline requested: {pipeline_id}",
        extra={
            "pipeline_id": pipeline_id,
            "libraries": [lib.value for lib in request.libraries],
            "demo_mode": demo_mode,
        },
    )

    if not demo_mode:
        # #354 C-8: invoke the wired ParallelPipeline. Helper raises
        # HTTPException(503) when no library produced a result (honest
        # fail-close), or returns a real response from the engine's
        # PipelineOutput. NO silent fallback, NO hardcoded values.
        try:
            # The real multi-library fan-out is the genuinely heavy in-process
            # compute. Bound it to ONE per-worker heavy-compute slot (OOM guard,
            # P1b) so concurrent real pipelines cannot stack and OOM-kill the
            # cgroup. A saturated worker raises HeavyComputeSaturated on enter
            # (mapped to 503 + Retry-After) — nothing is queued. The demo path
            # below does NO heavy work and is intentionally left unbounded.
            async with heavy_compute_slot():
                return await asyncio.wait_for(
                    _run_real_parallel_pipeline(pipeline_id, request),
                    timeout=request.timeout_seconds,
                )
        except HTTPException:
            raise
        except asyncio.TimeoutError as e:
            raise HTTPException(
                status_code=408,
                detail=f"Pipeline timed out after {request.timeout_seconds}s",
            ) from e
        except HeavyComputeSaturated:
            # Reject fast under load — surfaced as 503 + Retry-After by the app
            # exception handler. HeavyComputeSaturated is NOT an HTTPException,
            # so this must precede the broad last-resort handler below to avoid
            # being swallowed into a 500.
            raise
        except Exception as e:  # noqa: BLE001 - last-resort 500
            logger.error(f"Parallel pipeline failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=_GENERIC_500_DETAIL) from e

    try:
        # Run all libraries in parallel
        tasks = [
            _run_library_analysis(lib, request, demo_mode=demo_mode) for lib in request.libraries
        ]

        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=request.timeout_seconds,
        )

        # Process results
        library_results: Dict[str, Dict[str, Any]] = {}
        succeeded: List[str] = []
        failed: List[str] = []
        effect_estimates: List[float] = []

        for lib, result in zip(request.libraries, results, strict=False):
            if isinstance(result, HTTPException):
                # Real-path estimator unavailable for this library — surface
                # the upstream 503 to the client rather than fabricate.
                raise result
            if isinstance(result, Exception):
                library_results[lib.value] = {"error": str(result)}
                failed.append(lib.value)
            else:
                result_dict = cast(Dict[str, Any], result)
                library_results[lib.value] = result_dict
                succeeded.append(lib.value)
                if result_dict.get("effect_estimate") is not None:
                    effect_estimates.append(result_dict["effect_estimate"])

        # Compute consensus
        consensus_effect = None
        consensus_ci_lower = None
        consensus_ci_upper = None
        agreement_score = None

        if effect_estimates:
            import statistics

            # #27: derive the CI z-score from the requested confidence level
            # (default 0.95 => z~1.96, preserving the legacy half-width) instead
            # of a hardcoded magic number, and echo the level in the response so
            # the UI can label the interval truthfully. NB: this is the demo
            # path -- every library returns effect_estimate=0.0, so std==0.0 and
            # the interval is [consensus_effect, consensus_effect] at ANY z.
            z = z_score_for_confidence(request.confidence_level)
            consensus_effect = statistics.mean(effect_estimates)
            if len(effect_estimates) > 1:
                std = statistics.stdev(effect_estimates)
                consensus_ci_lower = consensus_effect - z * std
                consensus_ci_upper = consensus_effect + z * std
                cv = std / abs(consensus_effect) if consensus_effect != 0 else 1
                agreement_score = max(0, 1 - cv)
            else:
                consensus_ci_lower = consensus_effect
                consensus_ci_upper = consensus_effect
                agreement_score = 1.0

        total_latency_ms = int((time.time() - start_time) * 1000)

        warnings: List[str] = []
        if demo_mode:
            warnings.append(
                "demo_mode=true: results are pinned-zero placeholders with is_demo=true; "
                "do NOT use for decisions."
            )

        return ParallelPipelineResponse(
            pipeline_id=pipeline_id,
            status=AnalysisStatus.COMPLETED if succeeded else AnalysisStatus.FAILED,
            libraries_succeeded=succeeded,
            libraries_failed=failed,
            library_results=library_results,
            consensus_effect=consensus_effect,
            consensus_ci_lower=consensus_ci_lower,
            consensus_ci_upper=consensus_ci_upper,
            confidence_level=request.confidence_level,
            library_agreement_score=agreement_score,
            consensus_method=request.consensus_method,
            total_latency_ms=total_latency_ms,
            created_at=datetime.now(timezone.utc),
            warnings=warnings,
        )

    except HTTPException:
        raise
    except asyncio.TimeoutError as e:
        raise HTTPException(
            status_code=408,
            detail=f"Pipeline timed out after {request.timeout_seconds}s",
        ) from e
    except Exception as e:
        logger.error(f"Parallel pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_GENERIC_500_DETAIL) from e


async def _run_library_analysis(
    library: CausalLibrary,
    request: ParallelPipelineRequest,
    demo_mode: bool = False,
) -> Dict[str, Any]:
    """Run analysis for a single library.

    Default path fails closed with HTTPException(503): there is no real data
    backend wired into the parallel pipeline. With ``demo_mode=True`` returns
    a pinned-zero placeholder labeled ``is_demo=true``. Never returns RNG
    values (F-005 fix).

    See F-005 audit iter-1 HIGH-1: synthetic-data + real-estimator in the
    default path is a labeling fabrication, not a functional fix.
    """
    if not demo_mode:
        raise HTTPException(status_code=503, detail=_NO_REAL_DATA_BACKEND_DETAIL)

    return {
        "library": library.value,
        "estimator": request.estimators.get(library.value) if request.estimators else None,
        "effect_estimate": 0.0,
        "ci_lower": 0.0,
        "ci_upper": 0.0,
        "p_value": 1.0,
        "n_samples": 0,
        "is_demo": True,
    }


@router.get(
    "/pipeline/{pipeline_id}", summary="Get pipeline status", operation_id="get_pipeline_status"
)
async def get_pipeline_status(
    pipeline_id: str,
) -> Dict[str, Any]:
    """
    Get status of a pipeline execution.

    Args:
        pipeline_id: Unique pipeline identifier

    Returns:
        Pipeline status and results
    """
    if pipeline_id not in _pipeline_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline {pipeline_id} not found",
        )

    return _pipeline_cache[pipeline_id]


# =============================================================================
# CROSS-VALIDATION ENDPOINT
# =============================================================================


@router.post(
    "/validate",
    response_model=CrossValidationResponse,
    summary="Run cross-library validation",
    operation_id="run_cross_validation",
)
async def run_cross_validation(
    request: CrossValidationRequest,
    demo_mode: bool = Query(
        default=False,
        description=(
            "If true, return pinned-zero placeholder results labeled with "
            "is_demo=true (for UI demonstrations only). Default is false: "
            "the endpoint runs real estimator selection or fails with 503."
        ),
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> CrossValidationResponse:
    """
    Run cross-library validation (DoWhy ↔ CausalML).

    Compares effect estimates between libraries to validate results.

    Args:
        request: Cross-validation configuration
        demo_mode: If True, return clearly-labeled placeholder values

    Returns:
        CrossValidationResponse with agreement metrics
    """
    validation_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        f"Cross-validation requested: {validation_id}",
        extra={
            "validation_id": validation_id,
            "primary_library": request.primary_library.value,
            "validation_library": request.validation_library.value,
            "demo_mode": demo_mode,
        },
    )

    if not demo_mode:
        # Default path has no real data backend; fail-closed (F-005 iter-1 HIGH-1).
        raise HTTPException(status_code=503, detail=_NO_REAL_DATA_BACKEND_DETAIL)

    # demo_mode=True: pinned-zero placeholder. Agreement is trivially perfect
    # because both libraries return the same zero, which we label explicitly.
    primary_effect = 0.0
    validation_effect = 0.0
    primary_ci = (0.0, 0.0)
    validation_ci = (0.0, 0.0)
    effect_difference = 0.0
    relative_difference = 0.0
    ci_overlap_ratio = 1.0
    agreement_score = 1.0
    validation_passed = agreement_score >= request.agreement_threshold

    latency_ms = int((time.time() - start_time) * 1000)

    # Surface the is_demo=true label as the FIRST recommendation so consumers
    # cannot miss it (CrossValidationResponse schema has no is_demo field, so
    # we encode the label in recommendations per F-005 iter-1 HIGH-3).
    recommendations: List[str] = [
        "is_demo=true: results are pinned-zero placeholders; do NOT use for decisions.",
    ]

    response = CrossValidationResponse(
        validation_id=validation_id,
        primary_library=request.primary_library.value,
        validation_library=request.validation_library.value,
        primary_effect=primary_effect,
        primary_ci=list(primary_ci),
        validation_effect=validation_effect,
        validation_ci=list(validation_ci),
        effect_difference=effect_difference,
        relative_difference=relative_difference,
        ci_overlap_ratio=ci_overlap_ratio,
        agreement_score=agreement_score,
        validation_passed=validation_passed,
        agreement_threshold=request.agreement_threshold,
        latency_ms=latency_ms,
        created_at=datetime.now(timezone.utc),
        recommendations=recommendations,
    )

    _validation_cache[validation_id] = response
    return response


# =============================================================================
# ESTIMATOR INFO ENDPOINT
# =============================================================================


# Single source of truth for the supported causal estimators. Both
# ``list_estimators`` (the /estimators endpoint) and ``causal_health_check``
# (``estimators_loaded``) read from this so the health count can never drift
# from the registry (previously ``estimators_loaded`` was a hardcoded ``12``).
_ESTIMATOR_REGISTRY: List[EstimatorInfo] = [
    # EconML
    EstimatorInfo(
        name="causal_forest",
        library=CausalLibrary.ECONML,
        estimator_type="CATE",
        description="Causal Forest for heterogeneous treatment effects",
        best_for=["Effect heterogeneity", "Feature importance"],
        parameters=["n_estimators", "min_samples_leaf", "max_depth"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="linear_dml",
        library=CausalLibrary.ECONML,
        estimator_type="CATE",
        description="Double Machine Learning with linear final stage",
        best_for=["High-dimensional confounders", "Linear effects"],
        parameters=["model_y", "model_t", "cv"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="ortho_forest",
        library=CausalLibrary.ECONML,
        estimator_type="CATE",
        description="Orthogonal Random Forest for CATE",
        best_for=["Non-linear effects", "SHAP integration"],
        parameters=["n_trees", "subsample_ratio", "max_depth"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="dr_learner",
        library=CausalLibrary.ECONML,
        estimator_type="CATE",
        description="Doubly Robust Learner",
        best_for=["Robustness to misspecification"],
        parameters=["model_propensity", "model_regression"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="x_learner",
        library=CausalLibrary.ECONML,
        estimator_type="Meta-Learner",
        description="X-Learner for heterogeneous effects",
        best_for=["Imbalanced treatment groups"],
        parameters=["models", "propensity_model"],
        supports_confidence_intervals=True,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="t_learner",
        library=CausalLibrary.ECONML,
        estimator_type="Meta-Learner",
        description="Two-Model approach",
        best_for=["Simple interpretation"],
        parameters=["models"],
        supports_confidence_intervals=False,
        supports_heterogeneous_effects=True,
    ),
    EstimatorInfo(
        name="s_learner",
        library=CausalLibrary.ECONML,
        estimator_type="Meta-Learner",
        description="Single-Model approach",
        best_for=["Limited data"],
        parameters=["overall_model"],
        supports_confidence_intervals=False,
        supports_heterogeneous_effects=True,
    ),
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


def _parse_occurred_at(value: Any) -> Optional[datetime]:
    """Coerce an episodic ``occurred_at`` (ISO string or datetime) to datetime.

    Returns ``None`` for missing/unparseable values rather than fabricating a
    timestamp.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _as_float(value: Any) -> Optional[float]:
    """Coerce a raw_content numeric field to float, or ``None`` if absent/invalid.

    Returns ``None`` (honest unknown) rather than a fabricated default so a
    missing ATE/confidence never renders as a plausible-looking number. Accepts
    native numbers and numeric strings (a JSONB round-trip or a non-canonical
    writer can encode a float as ``"0.185"``); a non-numeric value is ``None``.
    """
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    return None


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


def _as_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


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

# Paged-read page size. PostgREST returns at most ~1000 rows/request by default;
# patient cohorts have ~8.4k rows and the HCP cohorts 5k — a single unpaged
# .select() would SILENTLY truncate the cohort to a non-representative sample and
# misreport the ATE. We page with .range() until a short page is returned.
_TE_PAGE_SIZE = 1000
# Hard ceiling on pages so a runaway loop cannot exhaust memory; 20 pages * 1000
# = 20k rows comfortably covers the largest cohort (~8.4k) with headroom.
_TE_MAX_PAGES = 20

# Per-request compute budget (seconds) for the DoWhy+EconML fit under the
# heavy-compute slot. A single cohort fit is seconds; this bounds a degenerate
# run so a slow/contended box returns 408 rather than holding the slot forever.
_TE_TIMEOUT_SECONDS = 90.0


async def _te_paged_select(
    client: Any,
    table: str,
    columns: str,
    brand: str,
) -> List[Dict[str, Any]]:
    """Read ALL synthetic rows for ``brand`` from ``table`` via paged .range().

    Mirrors the audit.py .range() paging pattern. Returns the full row list —
    NEVER a silently-truncated sample (the single highest-risk fabrication bug for
    this surface). Raises on PostgREST/transport errors so the caller fail-closes.
    """
    rows: List[Dict[str, Any]] = []
    for page in range(_TE_MAX_PAGES):
        offset = page * _TE_PAGE_SIZE
        query = (
            client.table(table)
            .select(columns)
            .eq("brand", brand)
            .eq("is_synthetic", True)
            .range(offset, offset + _TE_PAGE_SIZE - 1)
        )
        result = await query.execute()
        batch: List[Dict[str, Any]] = result.data or []
        rows.extend(batch)
        if len(batch) < _TE_PAGE_SIZE:
            break
    return rows


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


def _te_pvalue_from_z(ate: float, std_error: Optional[float]) -> Optional[float]:
    """Two-sided model-based z-test p-value, mirroring the agent estimation path.

    ``p = 2*(1 - Phi(|ate|/std_error))``. Returns None when std_error is missing
    or not a usable positive finite value (we never emit p=NaN). This is a
    model-based p-value, NOT a refutation p-value.
    """
    if std_error is None:
        return None
    try:
        se = float(std_error)
    except (TypeError, ValueError):
        return None
    import math as _math

    if not _math.isfinite(se) or se <= 0.0:
        return None
    from scipy import stats as _scipy_stats

    z = abs(float(ate)) / se
    return float(2.0 * (1.0 - _scipy_stats.norm.cdf(z)))


async def _run_treatment_effect_estimate(
    cohort: str,
    brand: str,
    spec: "_TEFrameSpec",
) -> TreatmentEffectResponse:
    """Run the wired DoWhy+EconML sequential pipeline on the resolved frame.

    Prefers EconML's ate/ci/std (it carries the CI); falls back to DoWhy's
    causal_effect/standard_error (no CI) when EconML fails. Raises
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
        # DoWhy fallback: no CI (linear_regression provides only an SE).
        ate = _as_optional_float(dowhy_payload.get("causal_effect"))
        std_error = _as_optional_float(dowhy_payload.get("standard_error"))
        estimator = dowhy_payload.get("dowhy_method")

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
