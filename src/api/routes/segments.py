"""
E2I Segment Analysis & Heterogeneous Optimization API
======================================================

FastAPI endpoints for segment-level CATE analysis and targeting optimization.

Phase: Agent Output Routing

Endpoints:
- POST /segments/analyze: Run segment analysis (CATE estimation)
- GET  /segments/{analysis_id}: Get analysis results
- GET  /segments/policies: Get targeting recommendations
- GET  /segments/health: Service health check

Integration Points:
- Heterogeneous Optimizer Agent (Tier 2)
- EconML for CATE estimation
- CausalML for uplift modeling
- Supabase for persistence

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies.auth import require_analyst

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/segments", tags=["Segment Analysis"])


# =============================================================================
# ENUMS
# =============================================================================


class ResponderType(str, Enum):
    """Types of treatment responders."""

    HIGH = "high"
    LOW = "low"
    AVERAGE = "average"


class SegmentationMethod(str, Enum):
    """Methods for creating segments."""

    QUANTILE = "quantile"
    KMEANS = "kmeans"
    THRESHOLD = "threshold"
    TREE = "tree"


class AnalysisStatus(str, Enum):
    """Status of segment analysis."""

    PENDING = "pending"
    ESTIMATING = "estimating"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestionType(str, Enum):
    """Type of analysis question for library routing."""

    EFFECT_HETEROGENEITY = "effect_heterogeneity"  # EconML primary
    TARGETING = "targeting"  # CausalML primary
    SEGMENT_OPTIMIZATION = "segment_optimization"  # Both libraries
    COMPREHENSIVE = "comprehensive"  # All libraries with DoWhy validation


# =============================================================================
# REQUEST MODELS
# =============================================================================


class RunSegmentAnalysisRequest(BaseModel):
    """Request to run segment analysis."""

    query: str = Field(..., description="Natural language query describing the analysis")
    treatment_var: str = Field(
        ..., description="Treatment variable name (e.g., 'rep_visits', 'email_campaigns')"
    )
    outcome_var: str = Field(
        ..., description="Outcome variable name (e.g., 'trx', 'conversion')"
    )
    segment_vars: List[str] = Field(
        ..., description="Variables to segment by (e.g., ['region', 'specialty'])"
    )
    effect_modifiers: Optional[List[str]] = Field(
        default=None, description="Variables that modify treatment effect"
    )
    data_source: str = Field(
        default="hcp_data", description="Data source identifier"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional filters"
    )

    # Configuration
    n_estimators: int = Field(
        default=100, description="Causal Forest trees", ge=10, le=1000
    )
    min_samples_leaf: int = Field(
        default=10, description="Minimum samples per leaf", ge=1, le=100
    )
    significance_level: float = Field(
        default=0.05, description="For CI calculation", gt=0.0, lt=0.5
    )
    top_segments_count: int = Field(
        default=10, description="Number of top segments to return", ge=1, le=50
    )
    question_type: Optional[QuestionType] = Field(
        default=None, description="Analysis question type for library routing"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Which HCP segments respond best to rep visits?",
                "treatment_var": "rep_visits",
                "outcome_var": "trx",
                "segment_vars": ["region", "specialty"],
                "effect_modifiers": ["practice_size", "years_experience"],
                "data_source": "hcp_data",
                "n_estimators": 100,
                "top_segments_count": 10,
            }
        }
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class CATEResult(BaseModel):
    """CATE estimation result for a segment."""

    segment_name: str = Field(..., description="Segment dimension name")
    segment_value: str = Field(..., description="Segment value")
    cate_estimate: float = Field(..., description="Conditional Average Treatment Effect")
    cate_ci_lower: float = Field(..., description="95% CI lower bound")
    cate_ci_upper: float = Field(..., description="95% CI upper bound")
    sample_size: int = Field(..., description="Number of observations in segment")
    statistical_significance: bool = Field(
        ..., description="Whether effect is statistically significant"
    )


class SegmentProfile(BaseModel):
    """Profile of a high/low responder segment."""

    segment_id: str = Field(..., description="Unique segment identifier")
    responder_type: ResponderType = Field(..., description="Responder classification")
    cate_estimate: float = Field(..., description="CATE for this segment")
    defining_features: List[Dict[str, Any]] = Field(
        ..., description="Features that define this segment"
    )
    size: int = Field(..., description="Segment size (observations)")
    size_percentage: float = Field(..., description="Percentage of total population")
    recommendation: str = Field(..., description="Targeting recommendation")


class PolicyRecommendation(BaseModel):
    """Treatment allocation recommendation."""

    segment: str = Field(..., description="Segment identifier")
    current_treatment_rate: float = Field(
        ..., description="Current treatment rate (0-1)"
    )
    recommended_treatment_rate: float = Field(
        ..., description="Recommended treatment rate (0-1)"
    )
    expected_incremental_outcome: float = Field(
        ..., description="Expected incremental outcome from change"
    )
    confidence: float = Field(..., description="Recommendation confidence (0-1)")


class UpliftMetrics(BaseModel):
    """Uplift modeling metrics."""

    overall_auuc: float = Field(..., description="Area Under Uplift Curve (0-1)")
    overall_qini: float = Field(..., description="Qini coefficient")
    targeting_efficiency: float = Field(
        ..., description="How well model targets responders (0-1)"
    )
    model_type_used: str = Field(
        ..., description="Model type (random_forest, gradient_boosting)"
    )


class SegmentAnalysisResponse(BaseModel):
    """Response from segment analysis."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: AnalysisStatus = Field(..., description="Analysis status")
    question_type: Optional[QuestionType] = Field(
        default=None, description="Question type used for routing"
    )

    # CATE results
    cate_by_segment: Dict[str, List[CATEResult]] = Field(
        default_factory=dict, description="CATE results grouped by segment variable"
    )
    overall_ate: Optional[float] = Field(
        default=None, description="Overall Average Treatment Effect"
    )
    heterogeneity_score: Optional[float] = Field(
        default=None, description="Treatment effect heterogeneity (0-1)"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        default=None, description="Feature importance for CATE"
    )

    # Uplift results
    uplift_metrics: Optional[UpliftMetrics] = Field(
        default=None, description="Uplift modeling metrics"
    )

    # Segment discovery
    high_responders: List[SegmentProfile] = Field(
        default_factory=list, description="High responder segments"
    )
    low_responders: List[SegmentProfile] = Field(
        default_factory=list, description="Low responder segments"
    )

    # Policy recommendations
    policy_recommendations: List[PolicyRecommendation] = Field(
        default_factory=list, description="Targeting recommendations"
    )
    expected_total_lift: Optional[float] = Field(
        default=None, description="Expected lift from optimal allocation"
    )
    optimal_allocation_summary: Optional[str] = Field(
        default=None, description="Summary of optimal allocation"
    )

    # Summary
    executive_summary: Optional[str] = Field(
        default=None, description="Executive-level summary"
    )
    key_insights: List[str] = Field(default_factory=list, description="Key findings")

    # Multi-library support
    libraries_used: Optional[List[str]] = Field(
        default=None, description="Causal libraries used"
    )
    library_agreement_score: Optional[float] = Field(
        default=None, description="Agreement between libraries (0-1)"
    )
    validation_passed: Optional[bool] = Field(
        default=None, description="Whether cross-validation passed"
    )

    # Metadata
    estimation_latency_ms: int = Field(default=0, description="CATE estimation time")
    analysis_latency_ms: int = Field(default=0, description="Segment analysis time")
    total_latency_ms: int = Field(default=0, description="Total workflow time")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp",
    )
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")
    confidence: float = Field(default=0.0, description="Overall analysis confidence")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "analysis_id": "seg_abc123",
                "status": "completed",
                "overall_ate": 12.5,
                "heterogeneity_score": 0.65,
                "total_latency_ms": 4500,
            }
        }
    )


class PolicyListResponse(BaseModel):
    """Response for listing policy recommendations."""

    total_count: int = Field(..., description="Total recommendations")
    recommendations: List[PolicyRecommendation] = Field(
        ..., description="Policy recommendations"
    )
    expected_total_lift: float = Field(
        ..., description="Total expected lift if all policies adopted"
    )


class SegmentHealthResponse(BaseModel):
    """Health check response for segment analysis service."""

    status: str = Field(..., description="Service status")
    agent_available: bool = Field(
        ..., description="Heterogeneous Optimizer agent status"
    )
    econml_available: bool = Field(default=True, description="EconML availability")
    causalml_available: bool = Field(default=True, description="CausalML availability")
    last_analysis: Optional[datetime] = Field(
        default=None, description="Last analysis timestamp"
    )
    analyses_24h: int = Field(default=0, description="Analyses in last 24 hours")


# =============================================================================
# IN-MEMORY STORAGE (replace with Supabase in production)
# =============================================================================

_analyses_store: Dict[str, SegmentAnalysisResponse] = {}


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post(
    "/analyze",
    response_model=SegmentAnalysisResponse,
    summary="Run segment analysis",
    description="Analyze treatment effect heterogeneity across segments using CATE/uplift modeling.",
)
async def run_segment_analysis(
    request: RunSegmentAnalysisRequest,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(
        default=True, description="Run asynchronously (returns immediately with ID)"
    ),
    user: Dict[str, Any] = Depends(require_analyst),
) -> SegmentAnalysisResponse:
    """
    Run segment analysis for treatment effect heterogeneity.

    This endpoint invokes the Heterogeneous Optimizer agent (Tier 2) to:
    1. Estimate CATE using EconML (Causal Forest)
    2. Run uplift modeling using CausalML
    3. Identify high/low responder segments
    4. Generate targeting policy recommendations

    Args:
        request: Segment analysis parameters
        background_tasks: FastAPI background tasks
        async_mode: If True, returns immediately with analysis ID

    Returns:
        Segment analysis results or pending status if async
    """
    analysis_id = f"seg_{uuid4().hex[:12]}"

    # Create initial response
    response = SegmentAnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.PENDING if async_mode else AnalysisStatus.ESTIMATING,
        question_type=request.question_type,
    )

    if async_mode:
        # Store pending analysis
        _analyses_store[analysis_id] = response

        # Schedule background task
        background_tasks.add_task(
            _run_segment_analysis_task,
            analysis_id=analysis_id,
            request=request,
        )

        logger.info(f"Segment analysis {analysis_id} queued for background execution")
        return response

    # Synchronous execution
    try:
        result = await _execute_segment_analysis(request)
        result.analysis_id = analysis_id
        _analyses_store[analysis_id] = result
        return result
    except Exception as e:
        logger.error(f"Segment analysis failed: {e}")
        response.status = AnalysisStatus.FAILED
        response.warnings.append(str(e))
        _analyses_store[analysis_id] = response
        raise HTTPException(status_code=500, detail=f"Segment analysis failed: {e}")


@router.get(
    "/{analysis_id}",
    response_model=SegmentAnalysisResponse,
    summary="Get segment analysis results",
    description="Retrieve results of a segment analysis by ID.",
)
async def get_segment_analysis(analysis_id: str) -> SegmentAnalysisResponse:
    """
    Get segment analysis results by ID.

    Args:
        analysis_id: Unique analysis identifier

    Returns:
        Segment analysis results

    Raises:
        HTTPException: If analysis not found
    """
    if analysis_id not in _analyses_store:
        raise HTTPException(
            status_code=404,
            detail=f"Segment analysis {analysis_id} not found",
        )

    return _analyses_store[analysis_id]


@router.get(
    "/policies",
    response_model=PolicyListResponse,
    summary="List targeting recommendations",
    description="List all targeting policy recommendations.",
)
async def list_policies(
    min_lift: Optional[float] = Query(
        default=None, description="Minimum expected lift threshold"
    ),
    min_confidence: Optional[float] = Query(
        default=None, description="Minimum confidence threshold"
    ),
    limit: int = Query(default=20, description="Maximum results", ge=1, le=100),
) -> PolicyListResponse:
    """
    List targeting policy recommendations.

    Args:
        min_lift: Minimum expected lift threshold
        min_confidence: Minimum confidence threshold
        limit: Maximum number of results

    Returns:
        List of policy recommendations
    """
    all_recommendations: List[PolicyRecommendation] = []
    total_lift = 0.0

    for analysis in _analyses_store.values():
        if analysis.status != AnalysisStatus.COMPLETED:
            continue

        for rec in analysis.policy_recommendations:
            # Apply filters
            if min_lift and rec.expected_incremental_outcome < min_lift:
                continue
            if min_confidence and rec.confidence < min_confidence:
                continue

            all_recommendations.append(rec)
            total_lift += rec.expected_incremental_outcome

    # Sort by expected outcome and limit
    all_recommendations.sort(
        key=lambda x: x.expected_incremental_outcome, reverse=True
    )
    all_recommendations = all_recommendations[:limit]

    return PolicyListResponse(
        total_count=len(all_recommendations),
        recommendations=all_recommendations,
        expected_total_lift=total_lift,
    )


@router.get(
    "/health",
    response_model=SegmentHealthResponse,
    summary="Segment analysis service health",
    description="Check health status of the segment analysis service.",
)
async def get_segment_health() -> SegmentHealthResponse:
    """
    Get health status of segment analysis service.

    Returns:
        Service health information
    """
    # Check agent availability
    agent_available = True
    try:
        from src.agents.heterogeneous_optimizer import HeterogeneousOptimizerAgent

        agent_available = True
    except ImportError:
        agent_available = False

    # Check library availability
    econml_available = True
    causalml_available = True
    try:
        import econml  # noqa: F401
    except ImportError:
        econml_available = False

    try:
        import causalml  # noqa: F401
    except ImportError:
        causalml_available = False

    # Count recent analyses
    now = datetime.now(timezone.utc)
    analyses_24h = sum(
        1
        for a in _analyses_store.values()
        if (now - a.timestamp).total_seconds() < 86400
    )

    # Get last analysis
    last_analysis = None
    if _analyses_store:
        last_analysis = max(a.timestamp for a in _analyses_store.values())

    status = "healthy"
    if not agent_available:
        status = "degraded"
    elif not (econml_available and causalml_available):
        status = "partial"

    return SegmentHealthResponse(
        status=status,
        agent_available=agent_available,
        econml_available=econml_available,
        causalml_available=causalml_available,
        last_analysis=last_analysis,
        analyses_24h=analyses_24h,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _run_segment_analysis_task(
    analysis_id: str,
    request: RunSegmentAnalysisRequest,
) -> None:
    """Background task to run segment analysis."""
    try:
        logger.info(f"Starting segment analysis task {analysis_id}")

        # Update status
        if analysis_id in _analyses_store:
            _analyses_store[analysis_id].status = AnalysisStatus.ESTIMATING

        # Execute analysis
        result = await _execute_segment_analysis(request)
        result.analysis_id = analysis_id

        # Store result
        _analyses_store[analysis_id] = result

        logger.info(f"Segment analysis {analysis_id} completed successfully")

    except Exception as e:
        logger.error(f"Segment analysis {analysis_id} failed: {e}")
        if analysis_id in _analyses_store:
            _analyses_store[analysis_id].status = AnalysisStatus.FAILED
            _analyses_store[analysis_id].warnings.append(str(e))


async def _execute_segment_analysis(
    request: RunSegmentAnalysisRequest,
) -> SegmentAnalysisResponse:
    """
    Execute segment analysis using Heterogeneous Optimizer agent.

    This function orchestrates the Heterogeneous Optimizer agent (Tier 2) to:
    1. Estimate CATE via cate_estimator node
    2. Analyze segments via segment_analyzer node
    3. Learn policies via policy_learner node
    4. Generate profiles via profile_generator node
    """
    import time

    start_time = time.time()

    try:
        # Try to use the actual Heterogeneous Optimizer agent
        from src.agents.heterogeneous_optimizer.graph import (
            create_heterogeneous_optimizer_graph,
        )
        from src.agents.heterogeneous_optimizer.state import HeterogeneousOptimizerState

        # Initialize state
        initial_state: HeterogeneousOptimizerState = {
            "query": request.query,
            "treatment_var": request.treatment_var,
            "outcome_var": request.outcome_var,
            "segment_vars": request.segment_vars,
            "effect_modifiers": request.effect_modifiers or [],
            "data_source": request.data_source,
            "filters": request.filters,
            "n_estimators": request.n_estimators,
            "min_samples_leaf": request.min_samples_leaf,
            "significance_level": request.significance_level,
            "top_segments_count": request.top_segments_count,
            "status": "pending",
            "errors": [],
            "warnings": [],
            "estimation_latency_ms": 0,
            "analysis_latency_ms": 0,
            "total_latency_ms": 0,
        }

        # Create and run graph
        graph = create_heterogeneous_optimizer_graph()
        result = await graph.ainvoke(initial_state)

        # Convert agent output to API response
        total_latency = int((time.time() - start_time) * 1000)

        return SegmentAnalysisResponse(
            analysis_id="",  # Will be set by caller
            status=AnalysisStatus.COMPLETED
            if result.get("status") == "completed"
            else AnalysisStatus.FAILED,
            question_type=request.question_type,
            cate_by_segment=_convert_cate_results(result.get("cate_by_segment", {})),
            overall_ate=result.get("overall_ate"),
            heterogeneity_score=result.get("heterogeneity_score"),
            feature_importance=result.get("feature_importance"),
            uplift_metrics=_convert_uplift_metrics(result),
            high_responders=_convert_segment_profiles(
                result.get("high_responders", [])
            ),
            low_responders=_convert_segment_profiles(result.get("low_responders", [])),
            policy_recommendations=_convert_policies(
                result.get("policy_recommendations", [])
            ),
            expected_total_lift=result.get("expected_total_lift"),
            optimal_allocation_summary=result.get("optimal_allocation_summary"),
            executive_summary=result.get("executive_summary"),
            key_insights=result.get("key_insights", []),
            libraries_used=result.get("libraries_executed"),
            library_agreement_score=result.get("library_agreement_score"),
            validation_passed=result.get("validation_passed"),
            estimation_latency_ms=result.get("estimation_latency_ms", 0),
            analysis_latency_ms=result.get("analysis_latency_ms", 0),
            total_latency_ms=total_latency,
            warnings=result.get("warnings", []),
            confidence=result.get("confidence", 0.0),
        )

    except ImportError as e:
        logger.warning(
            f"Heterogeneous Optimizer agent not available: {e}, using mock data"
        )
        return _generate_mock_response(request, start_time)

    except Exception as e:
        logger.error(f"Segment analysis execution failed: {e}")
        raise


def _convert_cate_results(
    cate_data: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[CATEResult]]:
    """Convert agent CATE output to API response format."""
    result = {}
    for segment_var, cate_list in cate_data.items():
        result[segment_var] = []
        for cate in cate_list:
            try:
                result[segment_var].append(
                    CATEResult(
                        segment_name=cate.get("segment_name", segment_var),
                        segment_value=cate.get("segment_value", ""),
                        cate_estimate=cate.get("cate_estimate", 0.0),
                        cate_ci_lower=cate.get("cate_ci_lower", 0.0),
                        cate_ci_upper=cate.get("cate_ci_upper", 0.0),
                        sample_size=cate.get("sample_size", 0),
                        statistical_significance=cate.get(
                            "statistical_significance", False
                        ),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to convert CATE result: {e}")
    return result


def _convert_uplift_metrics(result: Dict[str, Any]) -> Optional[UpliftMetrics]:
    """Convert agent uplift output to API response format."""
    if not result.get("overall_auuc"):
        return None

    return UpliftMetrics(
        overall_auuc=result.get("overall_auuc", 0.0),
        overall_qini=result.get("overall_qini", 0.0),
        targeting_efficiency=result.get("targeting_efficiency", 0.0),
        model_type_used=result.get("model_type_used", "random_forest"),
    )


def _convert_segment_profiles(
    profiles: List[Dict[str, Any]],
) -> List[SegmentProfile]:
    """Convert agent segment profiles to API response format."""
    result = []
    for profile in profiles:
        try:
            result.append(
                SegmentProfile(
                    segment_id=profile.get("segment_id", ""),
                    responder_type=ResponderType(
                        profile.get("responder_type", "average")
                    ),
                    cate_estimate=profile.get("cate_estimate", 0.0),
                    defining_features=profile.get("defining_features", []),
                    size=profile.get("size", 0),
                    size_percentage=profile.get("size_percentage", 0.0),
                    recommendation=profile.get("recommendation", ""),
                )
            )
        except Exception as e:
            logger.warning(f"Failed to convert segment profile: {e}")
    return result


def _convert_policies(
    policies: List[Dict[str, Any]],
) -> List[PolicyRecommendation]:
    """Convert agent policy recommendations to API response format."""
    result = []
    for policy in policies:
        try:
            result.append(
                PolicyRecommendation(
                    segment=policy.get("segment", ""),
                    current_treatment_rate=policy.get("current_treatment_rate", 0.0),
                    recommended_treatment_rate=policy.get(
                        "recommended_treatment_rate", 0.0
                    ),
                    expected_incremental_outcome=policy.get(
                        "expected_incremental_outcome", 0.0
                    ),
                    confidence=policy.get("confidence", 0.0),
                )
            )
        except Exception as e:
            logger.warning(f"Failed to convert policy: {e}")
    return result


def _generate_mock_response(
    request: RunSegmentAnalysisRequest,
    start_time: float,
) -> SegmentAnalysisResponse:
    """Generate mock response when agent is not available."""
    import time

    # Mock CATE results
    mock_cate = {
        request.segment_vars[0]: [
            CATEResult(
                segment_name=request.segment_vars[0],
                segment_value="Northeast",
                cate_estimate=15.2,
                cate_ci_lower=8.5,
                cate_ci_upper=21.9,
                sample_size=1250,
                statistical_significance=True,
            ),
            CATEResult(
                segment_name=request.segment_vars[0],
                segment_value="Southeast",
                cate_estimate=8.7,
                cate_ci_lower=3.2,
                cate_ci_upper=14.2,
                sample_size=980,
                statistical_significance=True,
            ),
        ]
    }

    # Mock segment profiles
    mock_high_responder = SegmentProfile(
        segment_id=f"{request.segment_vars[0]}_northeast",
        responder_type=ResponderType.HIGH,
        cate_estimate=15.2,
        defining_features=[
            {"feature": request.segment_vars[0], "value": "Northeast"},
            {"feature": "specialty", "value": "Oncology"},
        ],
        size=1250,
        size_percentage=28.5,
        recommendation="Increase treatment intensity for this segment",
    )

    mock_low_responder = SegmentProfile(
        segment_id=f"{request.segment_vars[0]}_southeast",
        responder_type=ResponderType.LOW,
        cate_estimate=3.1,
        defining_features=[
            {"feature": request.segment_vars[0], "value": "Southeast"},
        ],
        size=420,
        size_percentage=9.5,
        recommendation="Consider reducing or reallocating resources",
    )

    # Mock policy recommendation
    mock_policy = PolicyRecommendation(
        segment="Northeast_Oncology",
        current_treatment_rate=0.35,
        recommended_treatment_rate=0.55,
        expected_incremental_outcome=125.5,
        confidence=0.82,
    )

    total_latency = int((time.time() - start_time) * 1000)

    return SegmentAnalysisResponse(
        analysis_id="",
        status=AnalysisStatus.COMPLETED,
        question_type=request.question_type,
        cate_by_segment=mock_cate,
        overall_ate=10.5,
        heterogeneity_score=0.65,
        feature_importance={
            request.segment_vars[0]: 0.42,
            "specialty": 0.28,
            "practice_size": 0.18,
        },
        uplift_metrics=UpliftMetrics(
            overall_auuc=0.72,
            overall_qini=0.58,
            targeting_efficiency=0.68,
            model_type_used="random_forest",
        ),
        high_responders=[mock_high_responder],
        low_responders=[mock_low_responder],
        policy_recommendations=[mock_policy],
        expected_total_lift=125.5,
        optimal_allocation_summary="Reallocate 20% of resources from low-responder to high-responder segments",
        executive_summary=f"Analysis identified significant treatment effect heterogeneity across {request.segment_vars}. Northeast region shows 74% higher response than average.",
        key_insights=[
            "Northeast region shows highest treatment response (CATE: 15.2)",
            "Oncology specialty is key effect modifier",
            "Optimal targeting could increase outcomes by 18%",
        ],
        estimation_latency_ms=200,
        analysis_latency_ms=150,
        total_latency_ms=total_latency,
        warnings=["Using mock data - Heterogeneous Optimizer agent not available"],
        confidence=0.75,
    )
