"""
Causal Inference API Schemas

Pydantic schemas for Causal API request/response validation.

Phase B10: Causal API endpoints for:
- Hierarchical analysis (EconML within CausalML segments)
- Library routing (DoWhy, EconML, CausalML, NetworkX)
- Multi-library pipelines (sequential, parallel)
- Cross-validation between libraries
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# ENUMS
# =============================================================================


class CausalLibrary(str, Enum):
    """Supported causal inference libraries."""

    DOWHY = "dowhy"
    ECONML = "econml"
    CAUSALML = "causalml"
    NETWORKX = "networkx"


class QuestionType(str, Enum):
    """Types of causal questions for routing."""

    CAUSAL_EFFECT = "causal_effect"  # "Does X cause Y?" → DoWhy
    EFFECT_HETEROGENEITY = "effect_heterogeneity"  # "How does effect vary?" → EconML
    TARGETING = "targeting"  # "Who should we target?" → CausalML
    SYSTEM_DEPENDENCIES = "system_dependencies"  # "How does impact flow?" → NetworkX
    COMPREHENSIVE = "comprehensive"  # All libraries


class EstimatorType(str, Enum):
    """Available causal estimators."""

    # EconML
    CAUSAL_FOREST = "causal_forest"
    LINEAR_DML = "linear_dml"
    ORTHO_FOREST = "ortho_forest"
    DR_LEARNER = "dr_learner"
    X_LEARNER = "x_learner"
    T_LEARNER = "t_learner"
    S_LEARNER = "s_learner"
    OLS = "ols"

    # CausalML
    UPLIFT_RANDOM_FOREST = "uplift_random_forest"
    UPLIFT_GRADIENT_BOOSTING = "uplift_gradient_boosting"

    # DoWhy
    PROPENSITY_SCORE_MATCHING = "propensity_score_matching"
    INVERSE_PROPENSITY_WEIGHTING = "inverse_propensity_weighting"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    INSTRUMENTAL_VARIABLE = "instrumental_variable"


class SegmentationMethod(str, Enum):
    """Segmentation methods for hierarchical analysis."""

    QUANTILE = "quantile"
    KMEANS = "kmeans"
    THRESHOLD = "threshold"
    TREE = "tree"


class AggregationMethod(str, Enum):
    """Aggregation methods for nested CI."""

    VARIANCE_WEIGHTED = "variance_weighted"
    SAMPLE_WEIGHTED = "sample_weighted"
    EQUAL = "equal"
    BOOTSTRAP = "bootstrap"


class PipelineMode(str, Enum):
    """Pipeline execution mode."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class AnalysisStatus(str, Enum):
    """Analysis status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# HIERARCHICAL ANALYSIS SCHEMAS
# =============================================================================


class HierarchicalAnalysisRequest(BaseModel):
    """Request for hierarchical CATE analysis."""

    treatment_var: str = Field(..., description="Treatment variable name")
    outcome_var: str = Field(..., description="Outcome variable name")
    effect_modifiers: List[str] = Field(
        default_factory=list,
        description="Variables that modify treatment effect",
    )
    data_source: str = Field(
        default="mock_data",
        description="Data source identifier",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Data filters",
    )

    # Hierarchical configuration
    n_segments: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of uplift segments",
    )
    segmentation_method: SegmentationMethod = Field(
        default=SegmentationMethod.QUANTILE,
        description="Method for creating segments",
    )
    estimator_type: EstimatorType = Field(
        default=EstimatorType.CAUSAL_FOREST,
        description="EconML estimator for segment-level CATE",
    )
    min_segment_size: int = Field(
        default=50,
        ge=10,
        description="Minimum samples per segment",
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.80,
        le=0.99,
        description="Confidence level for CIs",
    )
    aggregation_method: AggregationMethod = Field(
        default=AggregationMethod.VARIANCE_WEIGHTED,
        description="Method for aggregating segment CATEs",
    )
    timeout_seconds: int = Field(
        default=180,
        ge=30,
        le=600,
        description="Maximum execution time",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "effect_modifiers": ["age", "income", "region"],
                "n_segments": 3,
                "segmentation_method": "quantile",
                "estimator_type": "causal_forest",
            }
        }
    )


class SegmentCATEResult(BaseModel):
    """CATE result for a single segment."""

    segment_id: int = Field(..., description="Segment identifier")
    segment_name: str = Field(..., description="Segment name (e.g., 'high_uplift')")
    n_samples: int = Field(..., description="Number of samples in segment")
    uplift_range: List[float] = Field(..., min_length=2, max_length=2, description="Uplift score range [min, max]")
    cate_mean: Optional[float] = Field(None, description="Mean CATE estimate")
    cate_std: Optional[float] = Field(None, description="CATE standard deviation")
    cate_ci_lower: Optional[float] = Field(None, description="CATE CI lower bound")
    cate_ci_upper: Optional[float] = Field(None, description="CATE CI upper bound")
    success: bool = Field(..., description="Whether estimation succeeded")
    error_message: Optional[str] = Field(None, description="Error if failed")


class NestedCIResult(BaseModel):
    """Nested confidence interval aggregation result."""

    aggregate_ate: float = Field(..., description="Aggregate ATE from segments")
    aggregate_ci_lower: float = Field(..., description="Aggregate CI lower bound")
    aggregate_ci_upper: float = Field(..., description="Aggregate CI upper bound")
    aggregate_std: float = Field(..., description="Aggregate standard error")
    confidence_level: float = Field(..., description="Confidence level used")
    aggregation_method: str = Field(..., description="Aggregation method used")
    segment_contributions: Dict[str, float] = Field(
        ..., description="Weight contribution from each segment"
    )
    i_squared: Optional[float] = Field(None, description="I² heterogeneity statistic (0-100)")
    tau_squared: Optional[float] = Field(None, description="τ² between-segment variance")
    n_segments_included: int = Field(..., description="Segments included in aggregate")
    total_sample_size: int = Field(..., description="Total samples across segments")


class HierarchicalAnalysisResponse(BaseModel):
    """Response from hierarchical CATE analysis."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: AnalysisStatus = Field(..., description="Analysis status")
    segment_results: List[SegmentCATEResult] = Field(
        default_factory=list, description="Per-segment CATE results"
    )
    nested_ci: Optional[NestedCIResult] = Field(None, description="Nested CI aggregation")
    overall_ate: Optional[float] = Field(None, description="Overall ATE estimate")
    overall_ci_lower: Optional[float] = Field(None, description="Overall CI lower")
    overall_ci_upper: Optional[float] = Field(None, description="Overall CI upper")
    segment_heterogeneity: Optional[float] = Field(None, description="Heterogeneity score (I²)")
    n_segments_analyzed: int = Field(0, description="Number of segments analyzed")
    segmentation_method: str = Field(..., description="Segmentation method used")
    estimator_type: str = Field(..., description="EconML estimator used")
    latency_ms: int = Field(..., description="Execution time in milliseconds")
    created_at: datetime = Field(..., description="Analysis timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    errors: List[str] = Field(default_factory=list, description="Errors")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "analysis_id": "hier_abc12345",
                "status": "completed",
                "segment_results": [
                    {
                        "segment_id": 0,
                        "segment_name": "low_uplift",
                        "n_samples": 150,
                        "uplift_range": [-0.02, 0.15],
                        "cate_mean": 0.08,
                        "cate_std": 0.03,
                        "cate_ci_lower": 0.02,
                        "cate_ci_upper": 0.14,
                        "success": True,
                    },
                    {
                        "segment_id": 1,
                        "segment_name": "high_uplift",
                        "n_samples": 120,
                        "uplift_range": [0.15, 0.45],
                        "cate_mean": 0.32,
                        "cate_std": 0.05,
                        "cate_ci_lower": 0.22,
                        "cate_ci_upper": 0.42,
                        "success": True,
                    },
                ],
                "overall_ate": 0.18,
                "overall_ci_lower": 0.10,
                "overall_ci_upper": 0.26,
                "segment_heterogeneity": 62.3,
                "n_segments_analyzed": 2,
                "segmentation_method": "quantile",
                "estimator_type": "causal_forest",
                "latency_ms": 4520,
                "created_at": "2026-02-06T12:00:00Z",
            }
        }
    )


# =============================================================================
# LIBRARY ROUTING SCHEMAS
# =============================================================================


class RouteQueryRequest(BaseModel):
    """Request to route a causal query to appropriate library."""

    query: str = Field(
        ...,
        description="Natural language causal question",
        examples=[
            "Does increasing sales rep visits cause higher TRx?",
            "How does treatment effect vary by region?",
            "Who should we target for the promotional campaign?",
        ],
    )
    treatment_var: Optional[str] = Field(None, description="Treatment variable if known")
    outcome_var: Optional[str] = Field(None, description="Outcome variable if known")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for routing")
    prefer_library: Optional[CausalLibrary] = Field(
        None, description="Preferred library (optional override)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Does increasing sales rep visits cause higher TRx?",
                "treatment_var": "rep_visits",
                "outcome_var": "trx_count",
            }
        }
    )


class RouteQueryResponse(BaseModel):
    """Response from query routing."""

    query: str = Field(..., description="Original query")
    question_type: QuestionType = Field(..., description="Classified question type")
    primary_library: CausalLibrary = Field(..., description="Recommended primary library")
    secondary_libraries: List[CausalLibrary] = Field(
        default_factory=list, description="Recommended secondary libraries"
    )
    recommended_estimators: List[str] = Field(
        default_factory=list, description="Recommended estimators"
    )
    routing_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in routing decision"
    )
    routing_rationale: str = Field(..., description="Explanation for routing decision")
    suggested_pipeline: Optional[PipelineMode] = Field(None, description="Suggested pipeline mode")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Does increasing sales rep visits cause higher TRx?",
                "question_type": "causal_effect",
                "primary_library": "dowhy",
                "secondary_libraries": ["econml"],
                "recommended_estimators": [
                    "propensity_score_matching",
                    "causal_forest",
                ],
                "routing_confidence": 0.91,
                "routing_rationale": "Direct causal-effect question best suited for DoWhy identification + EconML estimation.",
                "suggested_pipeline": "sequential",
            }
        }
    )


# =============================================================================
# PIPELINE SCHEMAS
# =============================================================================


class PipelineStageConfig(BaseModel):
    """Configuration for a pipeline stage."""

    library: CausalLibrary = Field(..., description="Library for this stage")
    estimator: Optional[str] = Field(None, description="Specific estimator")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Stage parameters")
    timeout_seconds: int = Field(default=60, ge=10, le=300, description="Stage timeout")


class SequentialPipelineRequest(BaseModel):
    """Request for sequential multi-library pipeline."""

    treatment_var: str = Field(..., description="Treatment variable")
    outcome_var: str = Field(..., description="Outcome variable")
    covariates: List[str] = Field(default_factory=list, description="Covariate variables")
    data_source: str = Field(default="mock_data", description="Data source")
    filters: Optional[Dict[str, Any]] = Field(None, description="Data filters")

    # Pipeline configuration
    stages: List[PipelineStageConfig] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="Pipeline stages in order",
    )
    propagate_state: bool = Field(
        default=True,
        description="Propagate results between stages",
    )
    stop_on_failure: bool = Field(
        default=True,
        description="Stop pipeline on stage failure",
    )
    validation_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Minimum agreement threshold for validation",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "covariates": ["age", "income"],
                "stages": [
                    {"library": "networkx", "parameters": {}},
                    {"library": "dowhy", "estimator": "propensity_score_matching"},
                    {"library": "econml", "estimator": "causal_forest"},
                    {"library": "causalml", "estimator": "uplift_random_forest"},
                ],
            }
        }
    )


class PipelineStageResult(BaseModel):
    """Result from a single pipeline stage."""

    stage_number: int = Field(..., description="Stage position (1-indexed)")
    library: str = Field(..., description="Library used")
    estimator: Optional[str] = Field(None, description="Estimator used")
    status: AnalysisStatus = Field(..., description="Stage status")
    effect_estimate: Optional[float] = Field(None, description="Effect estimate")
    ci_lower: Optional[float] = Field(None, description="CI lower bound")
    ci_upper: Optional[float] = Field(None, description="CI upper bound")
    p_value: Optional[float] = Field(None, description="P-value")
    additional_results: Dict[str, Any] = Field(
        default_factory=dict, description="Library-specific results"
    )
    latency_ms: int = Field(..., description="Stage execution time")
    error: Optional[str] = Field(None, description="Error message if failed")


class SequentialPipelineResponse(BaseModel):
    """Response from sequential pipeline execution."""

    pipeline_id: str = Field(..., description="Unique pipeline identifier")
    status: AnalysisStatus = Field(..., description="Overall pipeline status")
    stages_completed: int = Field(..., description="Number of stages completed")
    stages_total: int = Field(..., description="Total number of stages")
    stage_results: List[PipelineStageResult] = Field(
        default_factory=list, description="Results from each stage"
    )

    # Consensus results
    consensus_effect: Optional[float] = Field(
        None, description="Confidence-weighted consensus effect"
    )
    consensus_ci_lower: Optional[float] = Field(None, description="Consensus CI lower")
    consensus_ci_upper: Optional[float] = Field(None, description="Consensus CI upper")
    library_agreement_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Agreement between libraries"
    )
    effect_estimate_variance: Optional[float] = Field(
        None, description="Variance across library estimates"
    )

    total_latency_ms: int = Field(..., description="Total pipeline execution time")
    created_at: datetime = Field(..., description="Pipeline start timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pipeline_id": "pipe_seq_001",
                "status": "completed",
                "stages_completed": 3,
                "stages_total": 3,
                "consensus_effect": 0.21,
                "consensus_ci_lower": 0.12,
                "consensus_ci_upper": 0.30,
                "library_agreement_score": 0.87,
                "total_latency_ms": 8900,
                "created_at": "2026-02-06T12:00:00Z",
            }
        }
    )


class ParallelPipelineRequest(BaseModel):
    """Request for parallel multi-library analysis."""

    treatment_var: str = Field(..., description="Treatment variable")
    outcome_var: str = Field(..., description="Outcome variable")
    covariates: List[str] = Field(default_factory=list, description="Covariate variables")
    data_source: str = Field(default="mock_data", description="Data source")
    filters: Optional[Dict[str, Any]] = Field(None, description="Data filters")

    # Parallel configuration
    libraries: List[CausalLibrary] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="Libraries to run in parallel",
    )
    estimators: Optional[Dict[str, str]] = Field(
        None,
        description="Specific estimator per library",
    )
    consensus_method: str = Field(
        default="variance_weighted",
        description="Method for consensus computation",
    )
    timeout_seconds: int = Field(
        default=120,
        ge=30,
        le=300,
        description="Overall timeout",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "libraries": ["dowhy", "econml", "causalml"],
                "estimators": {
                    "econml": "causal_forest",
                    "causalml": "uplift_random_forest",
                },
            }
        }
    )


class ParallelPipelineResponse(BaseModel):
    """Response from parallel pipeline execution."""

    pipeline_id: str = Field(..., description="Unique pipeline identifier")
    status: AnalysisStatus = Field(..., description="Overall status")
    libraries_succeeded: List[str] = Field(
        default_factory=list, description="Libraries that succeeded"
    )
    libraries_failed: List[str] = Field(default_factory=list, description="Libraries that failed")
    library_results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Results per library"
    )

    # Consensus
    consensus_effect: Optional[float] = Field(None, description="Consensus effect")
    consensus_ci_lower: Optional[float] = Field(None, description="Consensus CI lower")
    consensus_ci_upper: Optional[float] = Field(None, description="Consensus CI upper")
    library_agreement_score: Optional[float] = Field(None, description="Agreement score")
    consensus_method: str = Field(..., description="Consensus method used")

    total_latency_ms: int = Field(..., description="Total execution time")
    created_at: datetime = Field(..., description="Analysis timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pipeline_id": "pipe_par_001",
                "status": "completed",
                "libraries_succeeded": ["dowhy", "econml", "causalml"],
                "libraries_failed": [],
                "consensus_effect": 0.19,
                "consensus_ci_lower": 0.11,
                "consensus_ci_upper": 0.27,
                "library_agreement_score": 0.92,
                "consensus_method": "variance_weighted",
                "total_latency_ms": 5200,
                "created_at": "2026-02-06T12:00:00Z",
            }
        }
    )


# =============================================================================
# CROSS-VALIDATION SCHEMAS
# =============================================================================


class CrossValidationRequest(BaseModel):
    """Request for cross-library validation."""

    treatment_var: str = Field(..., description="Treatment variable")
    outcome_var: str = Field(..., description="Outcome variable")
    covariates: List[str] = Field(default_factory=list, description="Covariate variables")
    data_source: str = Field(default="mock_data", description="Data source")

    # Validation configuration
    primary_library: CausalLibrary = Field(..., description="Primary library for validation")
    validation_library: CausalLibrary = Field(..., description="Library to validate against")
    agreement_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Minimum agreement threshold",
    )
    bootstrap_iterations: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Bootstrap iterations for CI comparison",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "primary_library": "econml",
                "validation_library": "causalml",
                "agreement_threshold": 0.85,
            }
        }
    )


class CrossValidationResponse(BaseModel):
    """Response from cross-library validation."""

    validation_id: str = Field(..., description="Unique validation identifier")
    primary_library: str = Field(..., description="Primary library")
    validation_library: str = Field(..., description="Validation library")

    # Results
    primary_effect: float = Field(..., description="Effect from primary library")
    primary_ci: List[float] = Field(..., min_length=2, max_length=2, description="Primary confidence interval [lower, upper]")
    validation_effect: float = Field(..., description="Effect from validation library")
    validation_ci: List[float] = Field(..., min_length=2, max_length=2, description="Validation confidence interval [lower, upper]")

    # Agreement metrics
    effect_difference: float = Field(..., description="Absolute difference in effects")
    relative_difference: float = Field(..., description="Relative difference percentage")
    ci_overlap_ratio: float = Field(..., ge=0.0, le=1.0, description="CI overlap ratio")
    agreement_score: float = Field(..., ge=0.0, le=1.0, description="Overall agreement score")
    validation_passed: bool = Field(..., description="Whether validation threshold met")
    agreement_threshold: float = Field(..., description="Threshold used")

    latency_ms: int = Field(..., description="Validation execution time")
    created_at: datetime = Field(..., description="Validation timestamp")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations based on results"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "validation_id": "xval_001",
                "primary_library": "econml",
                "validation_library": "causalml",
                "primary_effect": 0.22,
                "primary_ci": [0.14, 0.30],
                "validation_effect": 0.20,
                "validation_ci": [0.11, 0.29],
                "effect_difference": 0.02,
                "relative_difference": 9.1,
                "ci_overlap_ratio": 0.88,
                "agreement_score": 0.91,
                "validation_passed": True,
                "agreement_threshold": 0.85,
                "latency_ms": 6300,
                "created_at": "2026-02-06T12:00:00Z",
                "recommendations": [
                    "High cross-library agreement supports causal conclusion.",
                ],
            }
        }
    )


# =============================================================================
# ESTIMATOR INFO SCHEMAS
# =============================================================================


class EstimatorInfo(BaseModel):
    """Information about a causal estimator."""

    name: str = Field(..., description="Estimator name")
    library: CausalLibrary = Field(..., description="Source library")
    estimator_type: str = Field(..., description="Type (CATE, uplift, identification, etc.)")
    description: str = Field(..., description="Brief description")
    best_for: List[str] = Field(default_factory=list, description="Best use cases")
    parameters: List[str] = Field(default_factory=list, description="Key parameters")
    supports_confidence_intervals: bool = Field(..., description="Whether CI is supported")
    supports_heterogeneous_effects: bool = Field(..., description="Whether HTE is supported")


class EstimatorListResponse(BaseModel):
    """Response listing available estimators."""

    estimators: List[EstimatorInfo] = Field(
        default_factory=list, description="Available estimators"
    )
    total: int = Field(..., description="Total estimators")
    by_library: Dict[str, List[str]] = Field(
        default_factory=dict, description="Estimators grouped by library"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "estimators": [
                    {
                        "name": "causal_forest",
                        "library": "econml",
                        "estimator_type": "CATE",
                        "description": "Generalized Random Forest for heterogeneous treatment effects",
                        "best_for": ["effect heterogeneity", "targeting"],
                        "parameters": ["n_estimators", "min_samples_leaf"],
                        "supports_confidence_intervals": True,
                        "supports_heterogeneous_effects": True,
                    }
                ],
                "total": 14,
                "by_library": {
                    "econml": ["causal_forest", "linear_dml", "dr_learner"],
                    "causalml": ["uplift_random_forest", "uplift_gradient_boosting"],
                    "dowhy": ["propensity_score_matching", "inverse_propensity_weighting"],
                },
            }
        }
    )


# =============================================================================
# HEALTH CHECK SCHEMAS
# =============================================================================


class CausalHealthResponse(BaseModel):
    """Response for causal engine health check."""

    status: str = Field(
        ...,
        description="Overall health status",
        examples=["healthy", "degraded", "unhealthy"],
    )
    libraries_available: Dict[str, bool] = Field(..., description="Availability of each library")
    estimators_loaded: int = Field(..., description="Number of estimators loaded")
    pipeline_orchestrator_ready: bool = Field(
        ..., description="Whether pipeline orchestrator is ready"
    )
    hierarchical_analyzer_ready: bool = Field(
        ..., description="Whether hierarchical analyzer is ready"
    )
    last_analysis: Optional[datetime] = Field(None, description="Timestamp of last analysis")
    analysis_count_24h: int = Field(0, description="Analyses run in last 24 hours")
    average_latency_ms: Optional[int] = Field(None, description="Average analysis latency")
    error: Optional[str] = Field(None, description="Error message if unhealthy")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "libraries_available": {
                    "dowhy": True,
                    "econml": True,
                    "causalml": True,
                    "networkx": True,
                },
                "estimators_loaded": 14,
                "pipeline_orchestrator_ready": True,
                "hierarchical_analyzer_ready": True,
                "analysis_count_24h": 42,
                "average_latency_ms": 3200,
            }
        }
    )
