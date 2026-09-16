"""Hierarchical (segment-level CATE) analysis routes for the causal package.

Submit, poll and execute the EconML-within-segments fit that produces per-segment
CATEs and the nested confidence interval, plus the demo placeholder and the
fail-closed DataFrame preflight that the sync and async paths share.

Import rule: may import ``_common``, ``datasets``, ``loaders`` and non-package
modules only; never another route module or the package root.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from src.api.dependencies.auth import require_analyst
from src.api.dependencies.compute import HeavyComputeSaturated, heavy_compute_slot
from src.api.schemas.causal import (
    AggregationMethod,
    AnalysisStatus,
    HierarchicalAnalysisRequest,
    HierarchicalAnalysisResponse,
    NestedCIResult,
    SegmentationMethod,
    SegmentCATEResult,
)

from ._common import _GENERIC_500_DETAIL, _NO_REAL_DATA_BACKEND_DETAIL, _resolve_pipeline_dataframe

logger = logging.getLogger(__name__)

router = APIRouter()


# In-memory storage (for demo — replace with a database in production).
_analysis_cache: Dict[str, HierarchicalAnalysisResponse] = {}


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
        from src.causal_engine.hierarchical.nested_ci import (
            NESTED_CI_EXCLUDED_NO_MEASURED_UNCERTAINTY,
            NESTED_CI_EXCLUDED_NO_MEASURED_UNCERTAINTY_DETAIL,
            SegmentEstimate,
            nested_ci_exclusion_warning,
        )

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

        # Compute nested CI — fail closed on unmeasured segments (#2027). A
        # segment enters the aggregate only with a measured SE (cate_se: the
        # true SE of the segment ATE, H6 — never cate_std, a per-unit
        # dispersion) AND both CI bounds; `is not None` so a real 0.0 bound is
        # kept. The rest are listed with a reason and named in `warnings`.
        # NestedConfidenceInterval.compute returns ±inf on zero segments, so
        # "no segment left" is decided here: nested_ci stays None.
        nested_ci_result = None
        segment_estimates: List[SegmentEstimate] = []
        nested_ci_excluded: List[Dict[str, Any]] = []
        for seg in result.segment_results:
            if not (seg.success and seg.cate_mean is not None):
                continue
            if seg.cate_se is None or seg.cate_ci_lower is None or seg.cate_ci_upper is None:
                nested_ci_excluded.append(
                    {
                        "segment_id": seg.segment_id,
                        "segment_name": seg.segment_name,
                        "n": seg.n_samples,
                        "reason": NESTED_CI_EXCLUDED_NO_MEASURED_UNCERTAINTY,
                        "detail": NESTED_CI_EXCLUDED_NO_MEASURED_UNCERTAINTY_DETAIL,
                    }
                )
                continue
            segment_estimates.append(
                SegmentEstimate(
                    segment_id=seg.segment_id,
                    segment_name=seg.segment_name,
                    ate=seg.cate_mean,
                    ate_std=seg.cate_se,
                    ci_lower=seg.cate_ci_lower,
                    ci_upper=seg.cate_ci_upper,
                    sample_size=seg.n_samples,
                    cate=None,
                )
            )
        exclusion_warnings = [nested_ci_exclusion_warning(e) for e in nested_ci_excluded]

        if segment_estimates:
            nested_ci_config = NestedCIConfig(
                confidence_level=request.confidence_level,
                aggregation_method=aggregation_map.get(
                    request.aggregation_method, EngineAggregationMethod.VARIANCE_WEIGHTED
                ),
                min_segment_size=request.min_segment_size,
            )
            nested_ci_calc = NestedConfidenceInterval(nested_ci_config)

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
            nested_ci_excluded_segments=nested_ci_excluded,
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
            warnings=[
                *(result.warnings if hasattr(result, "warnings") else []),
                *exclusion_warnings,
            ],
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
