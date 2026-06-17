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
from typing import Any, Dict, List, Optional, Tuple

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
        default="default",
        description=(
            "Data source identifier (a passthrough label/tag; does NOT trigger "
            "mock data — actual records come from `filters`/`estimation_data`)"
        ),
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
    uplift_range: List[float] = Field(
        ..., min_length=2, max_length=2, description="Uplift score range [min, max]"
    )
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
    confidence_level: float = Field(
        default=0.95,
        ge=0.80,
        le=0.99,
        description=(
            "Confidence level the CATE/overall CIs (cate_ci_lower/upper, "
            "overall_ci_lower/upper) are computed at, e.g. 0.95 => a 95% CI "
            "(z=1.96). Mirrors the request's confidence_level (alpha = "
            "1 - confidence_level). Exposed so consumers can label the interval "
            "truthfully instead of assuming 95%."
        ),
    )
    segment_heterogeneity: Optional[float] = Field(None, description="Heterogeneity score (I²)")
    n_segments_analyzed: int = Field(0, description="Number of segments analyzed")
    segmentation_method: str = Field(..., description="Segmentation method used")
    estimator_type: str = Field(..., description="EconML estimator used")
    latency_ms: int = Field(..., description="Execution time in milliseconds")
    created_at: datetime = Field(..., description="Analysis timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    errors: List[str] = Field(default_factory=list, description="Errors")
    is_demo: Optional[bool] = Field(
        default=None,
        description=(
            "True when this response is a demo_mode=true pinned-zero placeholder "
            "(NOT a real analysis). False/None for real computed results."
        ),
    )

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


class CausalVariablesResponse(BaseModel):
    """Available causal variables for a gold-standard dataset.

    Drives the causal-discovery page's treatment / outcome / covariate
    dropdowns. The candidate lists are the curated, causally-meaningful columns
    for the dataset intersected with its LIVE schema, so a dropdown never offers
    a column that is not actually present in the data.
    """

    dataset: str = Field(..., description="Gold-standard dataset the variables come from")
    treatment_candidates: List[str] = Field(
        default_factory=list, description="Columns valid as a treatment variable"
    )
    outcome_candidates: List[str] = Field(
        default_factory=list, description="Columns valid as an outcome variable"
    )
    covariate_candidates: List[str] = Field(
        default_factory=list, description="Columns valid as covariates/confounders"
    )
    columns: List[str] = Field(
        default_factory=list, description="All columns present in the dataset sample"
    )


class EstimationDataResponse(BaseModel):
    """Real estimation records loaded server-side from a gold-standard dataset.

    The frontend posts ``estimation_data_records`` into a pipeline request's
    ``filters.estimation_data_records`` so the parallel/sequential pipeline can
    estimate a real effect. Records are loaded from the live table (never
    fabricated); rows missing a treatment/outcome value are dropped.
    """

    dataset: str = Field(..., description="Dataset the records were loaded from")
    columns: List[str] = Field(..., description="Columns included in each record")
    n_rows: int = Field(..., description="Number of usable estimation rows returned")
    estimation_data_records: List[Dict[str, Any]] = Field(
        default_factory=list, description="Row records for the pipeline DataFrame"
    )


# =============================================================================
# AGENT ANALYSIS SCHEMAS (causal_impact LangGraph agent, end-to-end)
# =============================================================================
# The forceable estimator overrides MUST stay in sync with
# ``_VALID_EXPLICIT_METHODS`` in src/agents/causal_impact/nodes/estimation.py.
# Leaving ``estimator`` unset runs the agent's data-driven energy-score routing
# across the full registry (the recommended path); setting it forces one method.
AGENT_FORCEABLE_ESTIMATORS = (
    "CausalForestDML",
    "LinearDML",
    "drlearner",
    "ols",
    "propensity_score_weighting",
)


class AgentCausalAnalysisRequest(BaseModel):
    """Run the causal_impact agent end-to-end on a gold-standard dataset.

    The agent builds the causal DAG, selects an estimator data-drivenly (or the
    forced one), estimates the treatment->outcome effect, and runs refutation /
    sensitivity. The treatment / outcome / covariate columns are validated
    server-side against the dataset's curated allowlist.
    """

    treatment_var: str = Field(..., description="Treatment column (cause)")
    outcome_var: str = Field(..., description="Outcome column (effect)")
    dataset: str = Field("patient_journeys", description="Gold-standard dataset")
    covariates: Optional[List[str]] = Field(
        default=None,
        description="Confounder columns; omit to use the dataset's curated covariates.",
    )
    estimator: Optional[str] = Field(
        default=None,
        description=(
            "Force a specific estimator (one of AGENT_FORCEABLE_ESTIMATORS); omit "
            "for Auto (the agent's energy-score routing over the full registry)."
        ),
    )
    brand: Optional[str] = Field(default=None, description="Optional brand context")
    # 1500 keeps the default (Causal Forest) run tractable async (~4 min); the
    # planted effect is clearly recovered at this size (probe: p~0 at 1200 rows).
    limit: int = Field(1500, ge=100, le=20000, description="Max rows to load")


class CausalDAGModel(BaseModel):
    """The causal DAG the agent's graph_builder constructed for this analysis."""

    nodes: List[str] = Field(default_factory=list, description="Variable names")
    edges: List[Tuple[str, str]] = Field(
        default_factory=list, description="Directed (from, to) edges"
    )
    treatment_nodes: List[str] = Field(default_factory=list)
    outcome_nodes: List[str] = Field(default_factory=list)
    adjustment_sets: List[List[str]] = Field(
        default_factory=list, description="Valid backdoor adjustment sets"
    )
    dag_dot: Optional[str] = Field(default=None, description="Graphviz DOT for rendering")


class RefutationSummary(BaseModel):
    """Robustness gate + refutation/sensitivity summary from the agent."""

    gate_decision: Optional[str] = Field(default=None, description="proceed / review / block")
    passed: bool = Field(default=False, description="True only on a PROCEED gate")
    needs_review: bool = Field(default=False)
    tests_passed: Optional[int] = Field(default=None)
    tests_total: Optional[int] = Field(default=None)
    sensitivity_e_value: Optional[float] = Field(default=None)


class AgentCausalAnalysisResponse(BaseModel):
    """Result of an end-to-end causal_impact agent run.

    Carries the constructed DAG, the treatment->outcome effect, which estimator
    the agent used (data-driven or forced), and the refutation/interpretation —
    everything the page renders. Fail-closed: a run with no estimate surfaces
    status ``failed`` with the reason in ``warnings`` (never a fabricated ATE).
    """

    analysis_id: str
    status: str = Field(..., description="completed / needs_review / failed")
    treatment_var: str
    outcome_var: str
    dataset: str
    n_rows: int = Field(..., description="Usable estimation rows the agent ran on")
    data_source: str = Field(..., description="database / synthetic")
    dag: CausalDAGModel
    ate: Optional[float] = Field(default=None, description="Average treatment effect")
    ate_ci_lower: Optional[float] = Field(default=None)
    ate_ci_upper: Optional[float] = Field(default=None)
    standard_error: Optional[float] = Field(default=None)
    p_value: Optional[float] = Field(default=None)
    statistical_significance: bool = Field(default=False)
    selected_estimator: Optional[str] = Field(
        default=None, description="Estimator the agent actually used"
    )
    confidence: Optional[float] = Field(default=None, description="Overall confidence (0-1)")
    refutation: RefutationSummary = Field(default_factory=RefutationSummary)
    narrative: Optional[str] = Field(default=None, description="Natural-language interpretation")
    executive_summary: Optional[str] = Field(default=None)
    recommendations: List[str] = Field(default_factory=list)
    key_insights: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    latency_ms: int = Field(..., description="Total wall-clock latency")


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
    data_source: str = Field(
        default="default",
        description="Data source label/tag (passthrough; does NOT trigger mock data)",
    )
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
    confidence_level: float = Field(
        default=0.95,
        ge=0.80,
        le=0.99,
        description=(
            "Confidence level for the consensus CI, e.g. 0.95 => a 95% CI "
            "(z=1.96). Echoed back in the response so the UI labels the interval "
            "truthfully. Default 0.95 keeps the legacy +/-1.96*std behavior. "
            "SCOPE: governs the consensus CI only; per-library engine CIs are "
            "computed independently (currently fixed at 95% upstream)."
        ),
    )
    run_refutation: bool = Field(
        default=False,
        description=(
            "Opt-in: run the real DoWhy refutation suite on the DoWhy estimate and "
            "gate the result (PROCEED/REVIEW/BLOCK). Default False keeps the fast path "
            "(no refutation; robustness_validation_performed=False). True adds ~33s for "
            "OLS and up to 35-60 min for forest estimators (#622) — runs synchronously "
            "inside the heavy_compute_slot. NOTE (Owner-decision 5): refutation needs a "
            "real confidence interval, so v1 validates ONLY the linear-regression DoWhy "
            "estimate (the one method with a native standard error); non-linear methods "
            "are honestly skipped, not fabricated. Robustness is validated on the DoWhy "
            "estimate only; EconML/CausalML estimates in the consensus are unrefuted."
        ),
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
    confidence_level: float = Field(
        default=0.95,
        ge=0.80,
        le=0.99,
        description=(
            "Confidence level the consensus CI (consensus_ci_lower/upper) is "
            "computed at, e.g. 0.95 => a 95% CI (z=1.96). Exposed so consumers "
            "can label the interval truthfully instead of assuming 95%. The CI "
            "half-width is z*std where z is derived from this level. SCOPE: this "
            "labels the CONSENSUS CI only -- per-stage CIs in stage_results "
            "carry their own engine-computed bounds (currently fixed at 95% "
            "upstream). NOTE: the real (non-demo) path does not emit a consensus "
            "CI today (consensus_ci_lower/upper are None); this field still "
            "reports the level the CI would use."
        ),
    )
    library_agreement_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Agreement between libraries"
    )
    effect_estimate_variance: Optional[float] = Field(
        None, description="Variance across library estimates"
    )

    total_latency_ms: int = Field(..., description="Total pipeline execution time")
    created_at: datetime = Field(..., description="Pipeline start timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    robustness_validation_performed: bool = Field(
        default=False,
        description=(
            "True only if the DoWhy refutation/sensitivity suite was actually run "
            "for this ATE AND returned PROCEED. Defaults to False on the fast path "
            "(request.run_refutation=False, the default — the DoWhy executor returns "
            "refutation_results={}). When a caller opts in via run_refutation=True, "
            "this becomes True only if the suite PROCEEDed on a DAG; a REVIEW/BLOCK "
            "gate, an errored or skipped (non-linear, no SE) refutation, or a cyclic "
            "(non-DAG) graph keep it False with a caveat in robustness_warning. "
            "Robustness is validated on the DoWhy estimate only (Owner-decision 1)."
        ),
    )
    robustness_warning: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable caveat populated when robustness_validation_performed "
            "is False, so consumers cannot mistake an unrefuted/unvalidated ATE for "
            "a validated one. Names the reason: not opted-in (default fast path), "
            "REVIEW/BLOCK gate band, skipped (non-linear method, no SE), or a "
            "non-DAG structural warning. None when validation PROCEEDed."
        ),
    )
    graph_is_dag: Optional[bool] = Field(
        default=None,
        description="Whether the discovered/symbolic causal graph is acyclic (M-fo2). "
        "None when not computed; False signals a cyclic graph (estimate is structurally suspect).",
    )
    structural_quality: Optional[float] = Field(
        default=None,
        description="NetworkX structural-quality score in [0,1] (M-fo2): 1.0 identifiable "
        "(DAG or off-subgraph cycle) + path, 0.5 identifiable but path missing/<3 nodes, "
        "0.0 identification blocked (cycle on the treatment-outcome ancestral subgraph). "
        "Drives the consensus-confidence haircut.",
    )
    requires_review: bool = Field(
        default=False,
        description=(
            "M-fo2 (precise): True when the estimate is quarantined for human review "
            "because a directed cycle sits on the treatment-outcome ancestral subgraph, "
            "making backdoor identification undefined. When True, consensus_effect is "
            "WITHHELD (None) and robustness_validation_performed is forced False."
        ),
    )
    structural_identification: Optional[str] = Field(
        default=None,
        description=(
            "M-fo2 precise identifiability label: 'acyclic' (a DAG); 'cycle_irrelevant' "
            "(a cycle exists but OFF the treatment-outcome ancestral subgraph, so the "
            "estimand stays identifiable and the consensus is preserved); "
            "'undefined_cyclic' (a cycle ON the ancestral subgraph → consensus withheld, "
            "requires_review). None when the structural check did not run."
        ),
    )

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
    data_source: str = Field(
        default="default",
        description="Data source label/tag (passthrough; does NOT trigger mock data)",
    )
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
    confidence_level: float = Field(
        default=0.95,
        ge=0.80,
        le=0.99,
        description=(
            "Confidence level for the consensus CI, e.g. 0.95 => a 95% CI "
            "(z=1.96). Echoed back in the response so the UI labels the interval "
            "truthfully. Default 0.95 keeps the legacy +/-1.96*std behavior. "
            "SCOPE: governs the consensus CI only; per-library engine CIs are "
            "computed independently (currently fixed at 95% upstream)."
        ),
    )
    timeout_seconds: int = Field(
        default=120,
        ge=30,
        le=300,
        description="Overall timeout",
    )
    run_refutation: bool = Field(
        default=False,
        description=(
            "Opt-in: run the real DoWhy refutation suite on the DoWhy estimate and "
            "gate the result (PROCEED/REVIEW/BLOCK). Default False keeps the fast path "
            "(no refutation; robustness_validation_performed=False). True adds ~33s for "
            "OLS and up to 35-60 min for forest estimators (#622) — runs synchronously "
            "inside the heavy_compute_slot. NOTE (Owner-decision 5): refutation needs a "
            "real confidence interval, so v1 validates ONLY the linear-regression DoWhy "
            "estimate (the one method with a native standard error); non-linear methods "
            "are honestly skipped, not fabricated. Robustness is validated on the DoWhy "
            "estimate only; EconML/CausalML estimates in the consensus are unrefuted."
        ),
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
    confidence_level: float = Field(
        default=0.95,
        ge=0.80,
        le=0.99,
        description=(
            "Confidence level the consensus CI (consensus_ci_lower/upper) is "
            "computed at, e.g. 0.95 => a 95% CI (z=1.96). Exposed so consumers "
            "can label the interval truthfully instead of assuming 95%. The CI "
            "half-width is z*std where z is derived from this level. SCOPE: this "
            "labels the CONSENSUS CI only -- per-library CIs in library_results "
            "carry their own engine-computed bounds (currently fixed at 95% "
            "upstream). NOTE: the real (non-demo) path does not emit a consensus "
            "CI today (consensus_ci_lower/upper are None); this field still "
            "reports the level the CI would use."
        ),
    )
    library_agreement_score: Optional[float] = Field(None, description="Agreement score")
    consensus_method: str = Field(..., description="Consensus method used")

    total_latency_ms: int = Field(..., description="Total execution time")
    created_at: datetime = Field(..., description="Analysis timestamp")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    robustness_validation_performed: bool = Field(
        default=False,
        description=(
            "True only if the DoWhy refutation/sensitivity suite was actually run "
            "for this ATE AND returned PROCEED. Defaults to False on the fast path "
            "(request.run_refutation=False, the default — the DoWhy executor returns "
            "refutation_results={}). When a caller opts in via run_refutation=True, "
            "this becomes True only if the suite PROCEEDed on a DAG; a REVIEW/BLOCK "
            "gate, an errored or skipped (non-linear, no SE) refutation, or a cyclic "
            "(non-DAG) graph keep it False with a caveat in robustness_warning. "
            "Robustness is validated on the DoWhy estimate only (Owner-decision 1)."
        ),
    )
    robustness_warning: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable caveat populated when robustness_validation_performed "
            "is False, so consumers cannot mistake an unrefuted/unvalidated ATE for "
            "a validated one. Names the reason: not opted-in (default fast path), "
            "REVIEW/BLOCK gate band, skipped (non-linear method, no SE), or a "
            "non-DAG structural warning. None when validation PROCEEDed."
        ),
    )
    graph_is_dag: Optional[bool] = Field(
        default=None,
        description="Whether the discovered/symbolic causal graph is acyclic (M-fo2). "
        "None when not computed; False signals a cyclic graph (estimate is structurally suspect).",
    )
    structural_quality: Optional[float] = Field(
        default=None,
        description="NetworkX structural-quality score in [0,1] (M-fo2): 1.0 identifiable "
        "(DAG or off-subgraph cycle) + path, 0.5 identifiable but path missing/<3 nodes, "
        "0.0 identification blocked (cycle on the treatment-outcome ancestral subgraph). "
        "Drives the consensus-confidence haircut.",
    )
    requires_review: bool = Field(
        default=False,
        description=(
            "M-fo2 (precise): True when the estimate is quarantined for human review "
            "because a directed cycle sits on the treatment-outcome ancestral subgraph, "
            "making backdoor identification undefined. When True, consensus_effect is "
            "WITHHELD (None) and robustness_validation_performed is forced False."
        ),
    )
    structural_identification: Optional[str] = Field(
        default=None,
        description=(
            "M-fo2 precise identifiability label: 'acyclic' (a DAG); 'cycle_irrelevant' "
            "(a cycle exists but OFF the treatment-outcome ancestral subgraph); "
            "'undefined_cyclic' (a cycle ON the ancestral subgraph → consensus withheld, "
            "requires_review). None when the structural check did not run."
        ),
    )

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
    data_source: str = Field(
        default="default",
        description="Data source label/tag (passthrough; does NOT trigger mock data)",
    )

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
    primary_ci: List[float] = Field(
        ..., min_length=2, max_length=2, description="Primary confidence interval [lower, upper]"
    )
    validation_effect: float = Field(..., description="Effect from validation library")
    validation_ci: List[float] = Field(
        ..., min_length=2, max_length=2, description="Validation confidence interval [lower, upper]"
    )

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


class CausalAnalysisHistoryItem(BaseModel):
    """A single completed causal-analysis event, sourced from episodic_memories.

    These are REAL recorded analyses (``event_type='causal_analysis_completed'``),
    not a fabricated series. Numeric fields are ``None`` when the source row did
    not carry them — never defaulted to a plausible-looking value.
    """

    memory_id: str = Field(..., description="Episodic memory id of the analysis event")
    event_type: str = Field(..., description="Episodic event type")
    description: Optional[str] = Field(None, description="Human-readable analysis summary")
    occurred_at: datetime = Field(..., description="When the analysis completed")
    agent_name: Optional[str] = Field(None, description="Agent that produced the analysis")
    ate_estimate: Optional[float] = Field(
        None, description="Average treatment effect, if recorded in raw_content"
    )
    confidence: Optional[float] = Field(
        None, description="Confidence in the estimate, if recorded in raw_content"
    )
    model_used: Optional[str] = Field(
        None, description="Estimator/model used, if recorded in raw_content"
    )


class CausalAnalysisHistoryResponse(BaseModel):
    """Recent completed causal analyses for the Analysis History tab."""

    items: List[CausalAnalysisHistoryItem] = Field(
        default_factory=list, description="Recent completed causal analyses (newest first)"
    )
    total: int = Field(0, description="Number of items returned")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 1,
                "items": [
                    {
                        "memory_id": "92c7da7b-7fa8-4b94-a161-30033bc8780f",
                        "event_type": "causal_analysis_completed",
                        "description": "Causal analysis: treatment -> outcome, ATE=0.185",
                        "occurred_at": "2026-06-13T11:35:11.002171+00:00",
                        "agent_name": "causal_impact",
                        "ate_estimate": 0.1849,
                        "confidence": 0.78,
                        "model_used": "linear_regression",
                    }
                ],
            }
        }
    )


# =============================================================================
# TREATMENT EFFECTS (GET /causal/treatment-effects — cohort x brand ATE)
# =============================================================================


class CohortName(str, Enum):
    """The four cohorts the Treatment Effects surface supports.

    Each cohort selects WHICH outcome column is the label:
    - initiation       -> patient_journeys.treatment_initiated
    - persistence      -> patient_journeys.persistent_180d
    - discontinuation  -> patient_journeys.discontinued_180d
    - hcp_adoption     -> hcp_brand_adoption.adopted
    """

    INITIATION = "initiation"
    PERSISTENCE = "persistence"
    DISCONTINUATION = "discontinuation"
    HCP_ADOPTION = "hcp_adoption"


class BrandName(str, Enum):
    """The three brands the Treatment Effects surface supports."""

    REMIBRUTINIB = "Remibrutinib"
    FABHALTA = "Fabhalta"
    KISQALI = "Kisqali"


class TreatmentEffectResponse(BaseModel):
    """A REAL estimated average treatment effect for one (cohort, brand) cell.

    Produced by the live DoWhy+EconML sequential pipeline over a confounded
    cohort frame loaded from the DB. Every numeric field traces to a real
    estimator output — no fabricated/placeholder values. ``ci_lower``/``ci_upper``
    come from EconML's analytic CI; ``p_value`` is a model-based two-sided
    z-test ``2*(1-Phi(|ate|/std_error))`` (NOT a refutation p-value), and
    ``std_error`` is EconML's ``ate_std`` (or DoWhy's ``standard_error`` on the
    DoWhy fallback path, where no CI is available and ci_lower/ci_upper are None).

    The pipeline does NOT run refutation/sensitivity checks, so ``warnings``
    always carries an honest 'robustness not validated' caveat — the UI must
    never present this as a validated causal claim.
    """

    cohort: str = Field(
        ..., description="Cohort name (initiation/persistence/discontinuation/hcp_adoption)"
    )
    brand: str = Field(..., description="Brand (Remibrutinib/Fabhalta/Kisqali)")
    treatment_var: str = Field(..., description="Treatment column used (treatment_arm)")
    outcome_var: str = Field(
        ...,
        description="Outcome column used (treatment_initiated/persistent_180d/discontinued_180d/adopted)",
    )
    confounders: List[str] = Field(
        default_factory=list, description="The numeric confounders adjusted for (backdoor set)"
    )
    ate: float = Field(
        ..., description="Average treatment effect (EconML ate; agrees with DoWhy causal_effect)"
    )
    ci_lower: Optional[float] = Field(
        None, description="Lower bound of the EconML 95% CI (None on DoWhy fallback)"
    )
    ci_upper: Optional[float] = Field(
        None, description="Upper bound of the EconML 95% CI (None on DoWhy fallback)"
    )
    p_value: Optional[float] = Field(
        None,
        description="Model-based two-sided z-test p-value 2*(1-Phi(|ate|/std_error)); None when no usable std_error",
    )
    std_error: Optional[float] = Field(
        None,
        description="Standard error of the ATE (EconML ate_std, or DoWhy standard_error fallback)",
    )
    n: int = Field(..., description="Rows in the estimation frame after numeric-coerce + dropna")
    estimator: Optional[str] = Field(
        None, description="EconML selected estimator (e.g. 'ols'); None on DoWhy fallback"
    )
    method: str = Field("dowhy+econml sequential", description="Estimation method/pipeline")
    confidence_level: float = Field(0.95, description="Confidence level of the reported CI")
    latency_ms: int = Field(..., description="End-to-end compute latency in milliseconds")
    is_synthetic: bool = Field(
        True,
        description="True: this showcase substrate is synthetic-gold (E2I_INCLUDE_SYNTHETIC=true). Warning, not gate.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Honest caveats (always includes the robustness-not-validated note)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cohort": "hcp_adoption",
                "brand": "Fabhalta",
                "treatment_var": "treatment_arm",
                "outcome_var": "adopted",
                "confounders": ["peer_influence_score", "influence_network_size"],
                "ate": 0.1916,
                "ci_lower": 0.1644,
                "ci_upper": 0.2189,
                "p_value": 0.0,
                "std_error": 0.0139,
                "n": 5000,
                "estimator": "ols",
                "method": "dowhy+econml sequential",
                "confidence_level": 0.95,
                "latency_ms": 4200,
                "is_synthetic": True,
                "warnings": [
                    "robustness_validation_performed=false: this ATE was estimated but NOT "
                    "refutation-tested; do not present it as a validated causal claim."
                ],
            }
        }
    )
