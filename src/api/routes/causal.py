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
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from src.api.dependencies.auth import require_analyst
from src.api.schemas.causal import (
    AggregationMethod,
    AnalysisStatus,
    CausalHealthResponse,
    CausalLibrary,
    CrossValidationRequest,
    CrossValidationResponse,
    EstimatorInfo,
    EstimatorListResponse,
    HierarchicalAnalysisRequest,
    HierarchicalAnalysisResponse,
    NestedCIResult,
    ParallelPipelineRequest,
    ParallelPipelineResponse,
    PipelineStageResult,
    QuestionType,
    RouteQueryRequest,
    RouteQueryResponse,
    SegmentationMethod,
    SegmentCATEResult,
    SequentialPipelineRequest,
    SequentialPipelineResponse,
)
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse

# #354 C-8: real-pipeline wiring (replaces 503-default short-circuit in
# non-demo mode). Imported lazily-safely; the LibraryExecutor implementations
# inside ParallelPipeline / SequentialPipeline themselves guard their backend
# dependencies (dowhy/econml/causalml/networkx availability), so importing
# the orchestrator classes is cheap.
from src.causal_engine.pipeline.parallel import ParallelPipeline
from src.causal_engine.pipeline.sequential import SequentialPipeline
from src.causal_engine.pipeline.state import PipelineInput, PipelineOutput

logger = logging.getLogger(__name__)

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
    user: Dict[str, Any] = Depends(require_analyst),
) -> HierarchicalAnalysisResponse:
    """
    Run hierarchical CATE analysis (EconML within CausalML segments).

    This endpoint performs Pattern 4 from multi-library synergies:
    - Segments data using uplift scores (quantile, k-means, threshold)
    - Estimates CATE within each segment using EconML
    - Aggregates segment CATEs with nested confidence intervals
    - Computes heterogeneity statistics (I², τ²)

    Args:
        request: Hierarchical analysis configuration
        background_tasks: FastAPI background tasks
        async_mode: If True, runs analysis asynchronously

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
        # Create pending response and run in background
        pending_response = HierarchicalAnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.PENDING,
            segment_results=[],
            nested_ci=None,
            overall_ate=None,
            overall_ci_lower=None,
            overall_ci_upper=None,
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

        background_tasks.add_task(_run_hierarchical_analysis_task, analysis_id, request)

        return pending_response

    # Synchronous execution
    try:
        result = await _execute_hierarchical_analysis(analysis_id, request)
        _analysis_cache[analysis_id] = result
        return result

    except Exception as e:
        logger.error(f"Hierarchical analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
) -> None:
    """Background task for hierarchical analysis."""
    try:
        result = await _execute_hierarchical_analysis(analysis_id, request)
        _analysis_cache[analysis_id] = result
    except Exception as e:
        logger.error(f"Background hierarchical analysis failed: {e}")
        _analysis_cache[analysis_id] = HierarchicalAnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.FAILED,
            segment_results=[],
            nested_ci=None,
            overall_ate=None,
            overall_ci_lower=None,
            overall_ci_upper=None,
            segment_heterogeneity=None,
            n_segments_analyzed=0,
            segmentation_method=request.segmentation_method.value,
            estimator_type=request.estimator_type.value,
            latency_ms=0,
            created_at=datetime.now(timezone.utc),
            warnings=[],
            errors=[str(e)],
        )


async def _execute_hierarchical_analysis(
    analysis_id: str,
    request: HierarchicalAnalysisRequest,
) -> HierarchicalAnalysisResponse:
    """Execute hierarchical analysis using the causal engine."""
    start_time = time.time()

    try:
        import numpy as np
        import pandas as pd

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

        # Generate mock data for demonstration
        np.random.seed(42)
        n = 500
        df = pd.DataFrame(
            {
                request.treatment_var: np.random.binomial(1, 0.5, n),
                request.outcome_var: np.random.normal(100, 20, n),
            }
        )
        for modifier in request.effect_modifiers:
            df[modifier] = np.random.randn(n)

        # Add heterogeneous treatment effect
        if request.effect_modifiers:
            treatment_effect = 5.0 + df[request.effect_modifiers[0]] * 3.0
            df.loc[df[request.treatment_var] == 1, request.outcome_var] += treatment_effect[
                df[request.treatment_var] == 1
            ]

        # Prepare data
        X = df[request.effect_modifiers] if request.effect_modifiers else df.iloc[:, 2:]
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
                    ate_std=seg.cate_std or 0.01,
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
            segment_heterogeneity=result.segment_heterogeneity,
            n_segments_analyzed=result.n_segments,
            segmentation_method=request.segmentation_method.value,
            estimator_type=request.estimator_type.value,
            latency_ms=latency_ms,
            created_at=datetime.now(timezone.utc),
            warnings=result.warnings if hasattr(result, "warnings") else [],
            errors=result.errors if result.errors else [],
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Analysis timed out after {request.timeout_seconds}s",
        )
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Required library not available: {e}",
        )


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
    logger.info(f"Routing query: {request.query[:50]}...")

    # Simple keyword-based routing (replace with NLP classifier in production)
    query_lower = request.query.lower()

    # Override if preference specified
    if request.prefer_library:
        return _create_routing_response(
            request.query,
            _library_to_question_type(request.prefer_library),
            request.prefer_library,
            confidence=0.9,
            rationale=f"User preference: {request.prefer_library.value}",
        )

    # Classify question type
    if any(kw in query_lower for kw in ["cause", "causes", "effect of", "impact of", "does"]):
        question_type = QuestionType.CAUSAL_EFFECT
        primary_library = CausalLibrary.DOWHY
        rationale = "Question asks about causal relationship - DoWhy is best for identification"
    elif any(kw in query_lower for kw in ["vary", "heterogen", "different", "segment", "subgroup"]):
        question_type = QuestionType.EFFECT_HETEROGENEITY
        primary_library = CausalLibrary.ECONML
        rationale = "Question asks about effect heterogeneity - EconML provides CATE estimates"
    elif any(kw in query_lower for kw in ["target", "who should", "which", "optimize", "best"]):
        question_type = QuestionType.TARGETING
        primary_library = CausalLibrary.CAUSALML
        rationale = "Question about targeting - CausalML provides uplift modeling"
    elif any(kw in query_lower for kw in ["flow", "propagate", "system", "network", "dependency"]):
        question_type = QuestionType.SYSTEM_DEPENDENCIES
        primary_library = CausalLibrary.NETWORKX
        rationale = "Question about system dependencies - NetworkX for graph analysis"
    else:
        question_type = QuestionType.COMPREHENSIVE
        primary_library = CausalLibrary.ECONML
        rationale = "Ambiguous question - defaulting to EconML for comprehensive analysis"

    return _create_routing_response(
        request.query,
        question_type,
        primary_library,
        confidence=0.75,
        rationale=rationale,
    )


def _library_to_question_type(library: CausalLibrary) -> QuestionType:
    """Map library to question type."""
    mapping = {
        CausalLibrary.DOWHY: QuestionType.CAUSAL_EFFECT,
        CausalLibrary.ECONML: QuestionType.EFFECT_HETEROGENEITY,
        CausalLibrary.CAUSALML: QuestionType.TARGETING,
        CausalLibrary.NETWORKX: QuestionType.SYSTEM_DEPENDENCIES,
    }
    return mapping.get(library, QuestionType.COMPREHENSIVE)


def _create_routing_response(
    query: str,
    question_type: QuestionType,
    primary_library: CausalLibrary,
    confidence: float,
    rationale: str,
) -> RouteQueryResponse:
    """Create routing response with recommendations."""
    # Recommended estimators by library
    estimator_recommendations = {
        CausalLibrary.DOWHY: ["propensity_score_matching", "inverse_propensity_weighting"],
        CausalLibrary.ECONML: ["causal_forest", "linear_dml", "dr_learner"],
        CausalLibrary.CAUSALML: ["uplift_random_forest", "uplift_gradient_boosting"],
        CausalLibrary.NETWORKX: [],
    }

    # Secondary libraries
    secondary_map = {
        CausalLibrary.DOWHY: [CausalLibrary.ECONML],
        CausalLibrary.ECONML: [CausalLibrary.CAUSALML, CausalLibrary.DOWHY],
        CausalLibrary.CAUSALML: [CausalLibrary.ECONML],
        CausalLibrary.NETWORKX: [CausalLibrary.DOWHY],
    }

    return RouteQueryResponse(
        query=query,
        question_type=question_type,
        primary_library=primary_library,
        secondary_libraries=secondary_map.get(primary_library, []),
        recommended_estimators=estimator_recommendations.get(primary_library, []),
        routing_confidence=confidence,
        routing_rationale=rationale,
        suggested_pipeline=None,
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
    except Exception as e:
        logger.error(f"Sequential pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
        logger.error(f"Background sequential pipeline failed: {e}")
        _pipeline_cache[pipeline_id] = SequentialPipelineResponse(
            pipeline_id=pipeline_id,
            status=AnalysisStatus.FAILED,
            stages_completed=0,
            stages_total=len(request.stages),
            stage_results=[],
            consensus_effect=None,
            consensus_ci_lower=None,
            consensus_ci_upper=None,
            library_agreement_score=None,
            effect_estimate_variance=None,
            total_latency_ms=0,
            created_at=datetime.now(timezone.utc),
            warnings=[str(e)],
        ).model_dump()


_NO_REAL_DATA_BACKEND_DETAIL = (
    "Causal pipeline endpoints have no real data backend wired. "
    "There is no production data source returning treatment/outcome columns by name. "
    "Pass demo_mode=true to get a clearly-labeled pinned-zero placeholder for UI demos, "
    "or wire real data and re-issue the request."
)

_NO_RESOLVABLE_DATA_DETAIL = (
    "Sequential/parallel pipeline executed but no library produced a result: "
    "no DataFrame was resolvable from the request filters and there is no "
    "production data backend wired for arbitrary data_source identifiers. "
    "Supply inline data via filters.estimation_data_records (list of dicts with "
    "treatment / outcome / covariate columns), or pass demo_mode=true for the "
    "clearly-labeled pinned-zero placeholder used in UI demos."
)


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


def _build_pipeline_input_sequential(
    request: SequentialPipelineRequest,
    *,
    libraries_enabled: Optional[List[str]] = None,
) -> PipelineInput:
    """Construct a PipelineInput for SequentialPipeline.execute() from request.

    The DataFrame is conveyed via ``filters["estimation_data"]`` so the
    Wave-1 executors (DoWhy reads ``state["filters"]["estimation_data"]``)
    see it; CausalML reads ``filters["dataframe"]`` so we mirror under both
    keys (the C-6 data_resolver helper picks whichever is present).
    """
    df = _resolve_pipeline_dataframe(request.filters)
    request_filters: Dict[str, Any] = dict(request.filters or {})
    if df is not None:
        # Mirror under all per-executor conventions C-1..C-6 left in place
        # (DoWhy reads `filters.estimation_data`, CausalML reads
        # `filters.dataframe`); EconML reads `data_cache.estimation_data`
        # which we plumb via the dedicated state slot below in the caller.
        request_filters["estimation_data"] = df
        request_filters["dataframe"] = df

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
        mode="sequential",
        libraries_enabled=libraries_enabled,
        cross_validate=None,
    )


def _build_pipeline_input_parallel(
    request: ParallelPipelineRequest,
) -> PipelineInput:
    """Construct a PipelineInput for ParallelPipeline.execute() from request."""
    df = _resolve_pipeline_dataframe(request.filters)
    request_filters: Dict[str, Any] = dict(request.filters or {})
    if df is not None:
        request_filters["estimation_data"] = df
        request_filters["dataframe"] = df

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
        mode="parallel",
        libraries_enabled=[lib.value for lib in request.libraries],
        cross_validate=None,
    )


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
          ``SequentialPipelineResponse`` constructed from
          ``PipelineOutput`` + the per-library state (no hardcoded values).
    """
    pipeline = SequentialPipeline(
        fail_fast=request.stop_on_failure,
    )
    libraries_enabled = [stage.library.value for stage in request.stages]
    pipeline_input = _build_pipeline_input_sequential(request, libraries_enabled=libraries_enabled)

    # Mirror the EconML executor convention (`data_cache.estimation_data`)
    # by stashing the DataFrame in PipelineInput.filters above; the orchestrator
    # passes filters through to state["filters"], and the C-6 data_resolver
    # helper picks the DataFrame from whichever key is populated. EconML reads
    # `state["data_cache"]["estimation_data"]` — we plumb it post-routing by
    # subclassing the pipeline below if needed. For now C-6's helper handles
    # the filters-side resolution, and the EconML executor will succeed when
    # its dedicated key is present OR fall through to its existing error path.
    output = await pipeline.execute(pipeline_input)

    return _sequential_output_to_response(pipeline_id, request, output)


async def _run_real_parallel_pipeline(
    pipeline_id: str,
    request: ParallelPipelineRequest,
) -> ParallelPipelineResponse:
    """Invoke the wired ParallelPipeline.execute() and adapt to the API response."""
    pipeline = ParallelPipeline(
        max_parallel=len(request.libraries),
        fail_fast=False,
    )
    pipeline_input = _build_pipeline_input_parallel(request)
    output = await pipeline.execute(pipeline_input)
    return _parallel_output_to_response(pipeline_id, request, output)


def _sequential_output_to_response(
    pipeline_id: str,
    request: SequentialPipelineRequest,
    output: PipelineOutput,
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
    """
    libraries_used = list(output.get("libraries_used") or [])
    errors = output.get("errors") or []
    failed_libraries = {
        str(err.get("library")) for err in errors if isinstance(err, dict) and err.get("library")
    }
    successful_libraries = [lib for lib in libraries_used if lib not in failed_libraries]

    if not successful_libraries:
        # Pipeline ran but every executor returned success=False (typically
        # because no DataFrame was resolvable from state). Honest fail-close.
        raise HTTPException(status_code=503, detail=_NO_RESOLVABLE_DATA_DETAIL)

    stage_results: List[PipelineStageResult] = []
    for idx, stage_config in enumerate(request.stages, 1):
        lib_value = stage_config.library.value
        stage_results.append(
            _build_stage_result_from_output(
                stage_number=idx,
                stage_config_library=lib_value,
                stage_config_estimator=stage_config.estimator,
                output=output,
                successful_libraries=successful_libraries,
                error_by_library={
                    str(err.get("library")): str(err.get("error") or "")
                    for err in errors
                    if isinstance(err, dict) and err.get("library")
                },
            )
        )

    stages_completed = sum(1 for r in stage_results if r.status == AnalysisStatus.COMPLETED)

    return SequentialPipelineResponse(
        pipeline_id=pipeline_id,
        status=_derive_response_status(stages_completed, len(request.stages)),
        stages_completed=stages_completed,
        stages_total=len(request.stages),
        stage_results=stage_results,
        consensus_effect=output.get("consensus_effect"),
        consensus_ci_lower=None,  # Not produced by the engine output today
        consensus_ci_upper=None,
        library_agreement_score=output.get("consensus_confidence"),
        effect_estimate_variance=None,
        total_latency_ms=int(output.get("total_latency_ms") or 0),
        created_at=datetime.now(timezone.utc),
        warnings=list(output.get("warnings") or []),
    )


def _parallel_output_to_response(
    pipeline_id: str,
    request: ParallelPipelineRequest,
    output: PipelineOutput,
) -> ParallelPipelineResponse:
    """Adapt PipelineOutput → ParallelPipelineResponse.

    Fails closed with 503 when no library produced a successful result.
    Same "successful library" derivation as the sequential adapter — see
    its docstring for the rationale.
    """
    libraries_used = list(output.get("libraries_used") or [])
    errors = output.get("errors") or []
    error_by_library: Dict[str, str] = {
        str(err.get("library")): str(err.get("error") or "")
        for err in errors
        if isinstance(err, dict) and err.get("library")
    }
    successful_libraries = [lib for lib in libraries_used if lib not in error_by_library]

    if not successful_libraries:
        # Honest fail-close — every requested library failed (e.g. no
        # DataFrame resolvable from state). 503 reflects reality.
        raise HTTPException(status_code=503, detail=_NO_RESOLVABLE_DATA_DETAIL)

    library_results: Dict[str, Dict[str, Any]] = {}
    succeeded: List[str] = []
    failed: List[str] = []

    for lib in request.libraries:
        lib_value = lib.value
        if lib_value in successful_libraries:
            succeeded.append(lib_value)
            library_results[lib_value] = _extract_library_payload(lib_value, output)
        elif lib_value in error_by_library:
            failed.append(lib_value)
            library_results[lib_value] = {"error": error_by_library[lib_value]}
        else:
            # Library was requested but neither executed nor errored —
            # validate_input rejected it before run.
            failed.append(lib_value)
            library_results[lib_value] = {"error": "library skipped during execution"}

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
        consensus_effect=output.get("consensus_effect"),
        consensus_ci_lower=None,
        consensus_ci_upper=None,
        library_agreement_score=output.get("consensus_confidence"),
        consensus_method=request.consensus_method,
        total_latency_ms=int(output.get("total_latency_ms") or 0),
        created_at=datetime.now(timezone.utc),
        warnings=list(output.get("warnings") or []),
    )


def _derive_response_status(stages_completed: int, stages_total: int) -> AnalysisStatus:
    """Derive API AnalysisStatus from completed/total stage counts."""
    if stages_completed == stages_total:
        return AnalysisStatus.COMPLETED
    return AnalysisStatus.FAILED


def _build_stage_result_from_output(
    *,
    stage_number: int,
    stage_config_library: str,
    stage_config_estimator: Optional[str],
    output: PipelineOutput,
    successful_libraries: List[str],
    error_by_library: Dict[str, str],
) -> PipelineStageResult:
    """Build a PipelineStageResult for one stage from PipelineOutput.

    Reads real values from ``output.primary_result`` when the stage's
    library matches the primary library and the library was successful.
    Otherwise marks the stage failed with a descriptive error.
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

    payload = _extract_library_payload(stage_config_library, output)
    effect = payload.get("effect_estimate")
    ci_lower = payload.get("ci_lower")
    ci_upper = payload.get("ci_upper")
    p_value = payload.get("p_value")

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
        latency_ms=int(output.get("total_latency_ms") or 0),
        error=None,
    )


def _extract_library_payload(library: str, output: PipelineOutput) -> Dict[str, Any]:
    """Extract a per-library payload from PipelineOutput.

    The engine's PipelineOutput surfaces only the primary library's full
    result in ``primary_result``. For non-primary libraries, return a compact
    summary derived from the consensus fields when available, otherwise an
    empty marker dict. This is intentionally minimal because Surface C
    response models are flat; deeper per-library payloads belong to a
    future enhancement (and would need the engine to surface per-library
    results, currently aggregated into consensus).
    """
    primary_result = output.get("primary_result") or {}
    payload: Dict[str, Any] = {"library": library}

    if library == "dowhy":
        effect = primary_result.get("causal_effect")
        if isinstance(effect, (int, float)):
            payload["effect_estimate"] = float(effect)
        method = primary_result.get("dowhy_method")
        if isinstance(method, str):
            payload["method"] = method
    elif library == "econml":
        ate = primary_result.get("ate") or primary_result.get("overall_ate")
        if isinstance(ate, (int, float)):
            payload["effect_estimate"] = float(ate)
        ci_lower = primary_result.get("ate_ci_lower") or primary_result.get("ci_lower")
        ci_upper = primary_result.get("ate_ci_upper") or primary_result.get("ci_upper")
        if isinstance(ci_lower, (int, float)):
            payload["ci_lower"] = float(ci_lower)
        if isinstance(ci_upper, (int, float)):
            payload["ci_upper"] = float(ci_upper)
    elif library == "causalml":
        ate = primary_result.get("ate")
        if isinstance(ate, (int, float)):
            payload["effect_estimate"] = float(ate)
        auuc = primary_result.get("auuc")
        if isinstance(auuc, (int, float)):
            payload["auuc"] = float(auuc)
        qini = primary_result.get("qini")
        if isinstance(qini, (int, float)):
            payload["qini"] = float(qini)
    elif library == "networkx":
        n_nodes = primary_result.get("n_nodes")
        if isinstance(n_nodes, (int, float)):
            payload["n_nodes"] = int(n_nodes)
        n_edges = primary_result.get("n_edges")
        if isinstance(n_edges, (int, float)):
            payload["n_edges"] = int(n_edges)
        is_dag = primary_result.get("is_dag")
        if isinstance(is_dag, bool):
            payload["is_dag"] = is_dag

    return payload


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
        except Exception as e:  # noqa: BLE001 - last-resort 500
            logger.error(f"Parallel pipeline failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

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

            consensus_effect = statistics.mean(effect_estimates)
            if len(effect_estimates) > 1:
                std = statistics.stdev(effect_estimates)
                consensus_ci_lower = consensus_effect - 1.96 * std
                consensus_ci_upper = consensus_effect + 1.96 * std
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
        raise HTTPException(status_code=500, detail=str(e)) from e


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
    estimators = [
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

    return CausalHealthResponse(
        status=status,
        libraries_available=libraries_available,
        estimators_loaded=12,  # Count from list_estimators
        pipeline_orchestrator_ready=pipeline_ready,
        hierarchical_analyzer_ready=hierarchical_ready,
        last_analysis=None,
        analysis_count_24h=0,
        average_latency_ms=None,
        error=None if status == "healthy" else "Some libraries unavailable",
    )
