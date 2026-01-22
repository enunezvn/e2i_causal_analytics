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
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.dependencies.auth import require_analyst, require_auth

from src.api.schemas.causal import (
    AggregationMethod,
    AnalysisStatus,
    CausalHealthResponse,
    CausalLibrary,
    CrossValidationRequest,
    CrossValidationResponse,
    EstimatorInfo,
    EstimatorListResponse,
    EstimatorType,
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/causal", tags=["Causal Inference"])


# =============================================================================
# IN-MEMORY STORAGE (for demo - replace with database in production)
# =============================================================================

_analysis_cache: Dict[str, HierarchicalAnalysisResponse] = {}
_pipeline_cache: Dict[str, Dict[str, Any]] = {}
_validation_cache: Dict[str, CrossValidationResponse] = {}


# =============================================================================
# HIERARCHICAL ANALYSIS ENDPOINTS
# =============================================================================


@router.post("/hierarchical/analyze", response_model=HierarchicalAnalysisResponse)
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
    start_time = time.time()

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

        background_tasks.add_task(
            _run_hierarchical_analysis_task, analysis_id, request
        )

        return pending_response

    # Synchronous execution
    try:
        result = await _execute_hierarchical_analysis(analysis_id, request)
        _analysis_cache[analysis_id] = result
        return result

    except Exception as e:
        logger.error(f"Hierarchical analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hierarchical/{analysis_id}", response_model=HierarchicalAnalysisResponse)
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
        from src.causal_engine.hierarchical import (
            AggregationMethod as EngineAggregationMethod,
            HierarchicalAnalyzer,
            HierarchicalConfig,
            NestedCIConfig,
            NestedConfidenceInterval,
        )
        from src.causal_engine.hierarchical.analyzer import SegmentationMethod as EngineSegmentationMethod
        from src.causal_engine.hierarchical.nested_ci import SegmentEstimate

        import numpy as np
        import pandas as pd

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
        df = pd.DataFrame({
            request.treatment_var: np.random.binomial(1, 0.5, n),
            request.outcome_var: np.random.normal(100, 20, n),
        })
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
                    uplift_range=seg.uplift_range,
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


@router.post("/route", response_model=RouteQueryResponse)
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


@router.post("/pipeline/sequential", response_model=SequentialPipelineResponse)
async def run_sequential_pipeline(
    request: SequentialPipelineRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(default=False, description="Run asynchronously"),
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

    Returns:
        SequentialPipelineResponse with stage results and consensus
    """
    pipeline_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        f"Sequential pipeline requested: {pipeline_id}",
        extra={
            "pipeline_id": pipeline_id,
            "stages": len(request.stages),
            "libraries": [s.library.value for s in request.stages],
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
        background_tasks.add_task(_run_sequential_pipeline_task, pipeline_id, request)
        return pending_response

    # Synchronous execution
    try:
        result = await _execute_sequential_pipeline(pipeline_id, request)
        _pipeline_cache[pipeline_id] = result.model_dump()
        return result
    except Exception as e:
        logger.error(f"Sequential pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _run_sequential_pipeline_task(
    pipeline_id: str,
    request: SequentialPipelineRequest,
) -> None:
    """Background task for sequential pipeline."""
    try:
        result = await _execute_sequential_pipeline(pipeline_id, request)
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


async def _execute_sequential_pipeline(
    pipeline_id: str,
    request: SequentialPipelineRequest,
) -> SequentialPipelineResponse:
    """Execute sequential pipeline stages."""
    start_time = time.time()
    stage_results: List[PipelineStageResult] = []
    effect_estimates: List[float] = []
    warnings: List[str] = []

    for i, stage_config in enumerate(request.stages, 1):
        stage_start = time.time()

        try:
            # Simulate stage execution (replace with actual library calls)
            await asyncio.sleep(0.1)  # Simulate processing

            # Mock effect estimate (varies by library for demo)
            import random
            base_effect = 0.15
            effect = base_effect + random.uniform(-0.05, 0.05)
            ci_half_width = random.uniform(0.03, 0.08)

            stage_result = PipelineStageResult(
                stage_number=i,
                library=stage_config.library.value,
                estimator=stage_config.estimator,
                status=AnalysisStatus.COMPLETED,
                effect_estimate=effect,
                ci_lower=effect - ci_half_width,
                ci_upper=effect + ci_half_width,
                p_value=random.uniform(0.001, 0.05),
                additional_results={
                    "n_samples": 500,
                    "method": stage_config.estimator or "default",
                },
                latency_ms=int((time.time() - stage_start) * 1000),
                error=None,
            )
            effect_estimates.append(effect)

        except Exception as e:
            stage_result = PipelineStageResult(
                stage_number=i,
                library=stage_config.library.value,
                estimator=stage_config.estimator,
                status=AnalysisStatus.FAILED,
                effect_estimate=None,
                ci_lower=None,
                ci_upper=None,
                p_value=None,
                additional_results={},
                latency_ms=int((time.time() - stage_start) * 1000),
                error=str(e),
            )
            if request.stop_on_failure:
                stage_results.append(stage_result)
                break

        stage_results.append(stage_result)

    # Compute consensus
    consensus_effect = None
    consensus_ci_lower = None
    consensus_ci_upper = None
    agreement_score = None
    variance = None

    if effect_estimates:
        import statistics
        consensus_effect = statistics.mean(effect_estimates)
        if len(effect_estimates) > 1:
            variance = statistics.variance(effect_estimates)
            std = statistics.stdev(effect_estimates)
            consensus_ci_lower = consensus_effect - 1.96 * std
            consensus_ci_upper = consensus_effect + 1.96 * std
            # Agreement score based on coefficient of variation
            cv = std / abs(consensus_effect) if consensus_effect != 0 else 1
            agreement_score = max(0, 1 - cv)
        else:
            consensus_ci_lower = consensus_effect - 0.05
            consensus_ci_upper = consensus_effect + 0.05
            agreement_score = 1.0

    total_latency_ms = int((time.time() - start_time) * 1000)
    stages_completed = len([r for r in stage_results if r.status == AnalysisStatus.COMPLETED])

    return SequentialPipelineResponse(
        pipeline_id=pipeline_id,
        status=AnalysisStatus.COMPLETED if stages_completed == len(request.stages) else AnalysisStatus.FAILED,
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


@router.post("/pipeline/parallel", response_model=ParallelPipelineResponse)
async def run_parallel_pipeline(
    request: ParallelPipelineRequest,
    user: Dict[str, Any] = Depends(require_analyst),
) -> ParallelPipelineResponse:
    """
    Run parallel multi-library analysis.

    Executes multiple causal libraries simultaneously and computes
    consensus results weighted by confidence.

    Args:
        request: Parallel pipeline configuration

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
        },
    )

    try:
        # Run all libraries in parallel
        tasks = [
            _run_library_analysis(lib, request)
            for lib in request.libraries
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

        for lib, result in zip(request.libraries, results):
            if isinstance(result, Exception):
                library_results[lib.value] = {"error": str(result)}
                failed.append(lib.value)
            else:
                library_results[lib.value] = result
                succeeded.append(lib.value)
                if result.get("effect_estimate") is not None:
                    effect_estimates.append(result["effect_estimate"])

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
                consensus_ci_lower = consensus_effect - 0.05
                consensus_ci_upper = consensus_effect + 0.05
                agreement_score = 1.0

        total_latency_ms = int((time.time() - start_time) * 1000)

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
            warnings=[],
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Pipeline timed out after {request.timeout_seconds}s",
        )
    except Exception as e:
        logger.error(f"Parallel pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _run_library_analysis(
    library: CausalLibrary,
    request: ParallelPipelineRequest,
) -> Dict[str, Any]:
    """Run analysis for a single library."""
    import random

    # Simulate library-specific analysis
    await asyncio.sleep(random.uniform(0.05, 0.15))

    base_effect = 0.15
    effect = base_effect + random.uniform(-0.05, 0.05)
    ci_half_width = random.uniform(0.03, 0.08)

    return {
        "library": library.value,
        "estimator": request.estimators.get(library.value) if request.estimators else None,
        "effect_estimate": effect,
        "ci_lower": effect - ci_half_width,
        "ci_upper": effect + ci_half_width,
        "p_value": random.uniform(0.001, 0.05),
        "n_samples": 500,
    }


@router.get("/pipeline/{pipeline_id}")
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


@router.post("/validate", response_model=CrossValidationResponse)
async def run_cross_validation(
    request: CrossValidationRequest,
    user: Dict[str, Any] = Depends(require_analyst),
) -> CrossValidationResponse:
    """
    Run cross-library validation (DoWhy ↔ CausalML).

    Compares effect estimates between libraries to validate results.

    Args:
        request: Cross-validation configuration

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
        },
    )

    try:
        import random

        # Simulate library results
        primary_effect = 0.15 + random.uniform(-0.02, 0.02)
        primary_ci_half = random.uniform(0.03, 0.06)
        primary_ci = (primary_effect - primary_ci_half, primary_effect + primary_ci_half)

        validation_effect = 0.15 + random.uniform(-0.03, 0.03)
        validation_ci_half = random.uniform(0.03, 0.06)
        validation_ci = (validation_effect - validation_ci_half, validation_effect + validation_ci_half)

        # Compute agreement metrics
        effect_difference = abs(primary_effect - validation_effect)
        relative_difference = effect_difference / abs(primary_effect) if primary_effect != 0 else 1

        # CI overlap
        overlap_start = max(primary_ci[0], validation_ci[0])
        overlap_end = min(primary_ci[1], validation_ci[1])
        if overlap_start < overlap_end:
            overlap_width = overlap_end - overlap_start
            total_width = max(primary_ci[1], validation_ci[1]) - min(primary_ci[0], validation_ci[0])
            ci_overlap_ratio = overlap_width / total_width
        else:
            ci_overlap_ratio = 0.0

        # Overall agreement score
        agreement_score = (1 - relative_difference) * ci_overlap_ratio
        validation_passed = agreement_score >= request.agreement_threshold

        latency_ms = int((time.time() - start_time) * 1000)

        recommendations = []
        if not validation_passed:
            recommendations.append("Consider investigating sources of disagreement between libraries")
            if ci_overlap_ratio < 0.5:
                recommendations.append("CI overlap is low - check model specifications")
        else:
            recommendations.append("Results validated successfully across libraries")

        response = CrossValidationResponse(
            validation_id=validation_id,
            primary_library=request.primary_library.value,
            validation_library=request.validation_library.value,
            primary_effect=primary_effect,
            primary_ci=primary_ci,
            validation_effect=validation_effect,
            validation_ci=validation_ci,
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

    except Exception as e:
        logger.error(f"Cross-validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ESTIMATOR INFO ENDPOINT
# =============================================================================


@router.get("/estimators", response_model=EstimatorListResponse)
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


@router.get("/health", response_model=CausalHealthResponse)
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
        import dowhy
        libraries_available["dowhy"] = True
    except ImportError:
        pass

    try:
        import econml
        libraries_available["econml"] = True
    except ImportError:
        pass

    try:
        import causalml
        libraries_available["causalml"] = True
    except ImportError:
        pass

    try:
        import networkx
        libraries_available["networkx"] = True
    except ImportError:
        pass

    # Check engine components
    hierarchical_ready = False
    pipeline_ready = False
    try:
        from src.causal_engine.hierarchical import HierarchicalAnalyzer
        hierarchical_ready = True
    except ImportError:
        pass

    # Determine overall status
    all_libs = all(libraries_available.values())
    status = "healthy" if all_libs else "degraded" if any(libraries_available.values()) else "unhealthy"

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
