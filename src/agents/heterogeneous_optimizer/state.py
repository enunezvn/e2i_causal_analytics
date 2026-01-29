"""State definitions for Heterogeneous Optimizer Agent.

This module defines the LangGraph state structure for segment-level CATE analysis.
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    import pandas as pd


class CATEResult(TypedDict):
    """CATE estimation result for a segment."""

    segment_name: str
    segment_value: str
    cate_estimate: float
    cate_ci_lower: float
    cate_ci_upper: float
    sample_size: int
    statistical_significance: bool


class SegmentProfile(TypedDict):
    """Profile of a high/low responder segment."""

    segment_id: str
    responder_type: Literal["high", "low", "average"]
    cate_estimate: float
    defining_features: List[Dict[str, Any]]
    size: int
    size_percentage: float
    recommendation: str


class PolicyRecommendation(TypedDict):
    """Treatment allocation recommendation."""

    segment: str
    current_treatment_rate: float
    recommended_treatment_rate: float
    expected_incremental_outcome: float
    confidence: float


class HeterogeneousOptimizerState(TypedDict):
    """Complete state for Heterogeneous Optimizer agent.

    This state flows through the 4-node workflow:
    1. cate_estimator: Estimate CATE using EconML
    2. segment_analyzer: Identify high/low responders
    3. policy_learner: Generate optimal allocation policy
    4. profile_generator: Create visualization data
    """

    # === INPUT (8 fields) ===
    query: str
    treatment_var: str
    outcome_var: str
    segment_vars: List[str]  # Variables to segment by
    effect_modifiers: List[str]  # Variables that modify treatment effect
    data_source: str
    filters: Optional[Dict[str, Any]]
    tier0_data: Optional[Any]  # DataFrame passthrough from tier0 testing (use Any to avoid pd import)

    # === CONFIGURATION (4 fields) ===
    n_estimators: int  # Causal Forest trees (default: 100)
    min_samples_leaf: int  # Minimum samples per leaf (default: 10)
    significance_level: float  # For CI calculation (default: 0.05)
    top_segments_count: int  # Number of top segments to return (default: 10)

    # === CATE OUTPUTS (4 fields) ===
    cate_by_segment: Optional[Dict[str, List[CATEResult]]]
    overall_ate: Optional[float]
    heterogeneity_score: Optional[float]  # 0-1, higher = more heterogeneity
    feature_importance: Optional[Dict[str, float]]

    # === UPLIFT OUTPUTS (6 fields) ===
    uplift_by_segment: Optional[Dict[str, List[Dict[str, Any]]]]  # Uplift scores by segment
    overall_auuc: Optional[float]  # Area Under Uplift Curve (0-1)
    overall_qini: Optional[float]  # Qini coefficient
    targeting_efficiency: Optional[float]  # How well model targets responders (0-1)
    model_type_used: Optional[str]  # "random_forest" or "gradient_boosting"
    uplift_latency_ms: Optional[int]  # Uplift analysis latency

    # ========================================================================
    # B9.4: Hierarchical Nesting (EconML within CausalML segments)
    # ========================================================================
    hierarchical_segment_results: Optional[List[Dict[str, Any]]]  # Per-segment CATE results
    nested_ci: Optional[Dict[str, Any]]  # Nested confidence interval aggregation
    segment_heterogeneity_score: Optional[float]  # I² statistic (0-100)
    overall_hierarchical_ate: Optional[float]  # Aggregate ATE from hierarchical analysis
    overall_hierarchical_ci_lower: Optional[float]  # Lower CI bound
    overall_hierarchical_ci_upper: Optional[float]  # Upper CI bound
    n_segments_analyzed: Optional[int]  # Number of segments analyzed
    segmentation_method_used: Optional[str]  # "quantile", "kmeans", "threshold", "tree"
    hierarchical_estimator_type: Optional[str]  # "causal_forest", "linear_dml", etc.
    hierarchical_latency_ms: Optional[int]  # Hierarchical analysis latency

    # === SEGMENT DISCOVERY OUTPUTS (3 fields) ===
    high_responders: Optional[List[SegmentProfile]]
    low_responders: Optional[List[SegmentProfile]]
    segment_comparison: Optional[Dict[str, Any]]

    # === POLICY OUTPUTS (3 fields) ===
    policy_recommendations: Optional[List[PolicyRecommendation]]
    expected_total_lift: Optional[float]
    optimal_allocation_summary: Optional[str]

    # === VISUALIZATION DATA (2 fields) ===
    cate_plot_data: Optional[Dict[str, Any]]
    segment_grid_data: Optional[Dict[str, Any]]

    # === SUMMARY (2 fields) ===
    executive_summary: Optional[str]
    key_insights: Optional[List[str]]

    # === EXECUTION METADATA (3 fields) ===
    estimation_latency_ms: int
    analysis_latency_ms: int
    total_latency_ms: int

    # === ERROR HANDLING (3 fields) ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "estimating", "analyzing", "optimizing", "completed", "failed"]

    # === CONTRACT-REQUIRED FIELDS (3 fields) ===
    # Added per .claude/contracts/tier2-contracts.md
    confidence: Optional[float]  # Overall analysis confidence (0.0-1.0)
    requires_further_analysis: Optional[bool]  # Whether further analysis is recommended
    suggested_next_agent: Optional[str]  # Next agent to invoke if further analysis needed

    # === MEMORY CONTEXT (3 fields) ===
    # Added for tri-memory integration per specialist document
    session_id: Optional[str]  # Session ID for memory operations
    working_memory_context: Optional[Dict[str, Any]]  # Context from working memory
    episodic_context: Optional[List[Dict[str, Any]]]  # Similar past analyses

    # ========================================================================
    # B7.4: Multi-Library Support (EconML + CausalML Cross-Validation)
    # ========================================================================

    # Library execution plan
    library_execution_plan: Optional[List[str]]  # e.g., ["econml", "causalml"]
    library_execution_mode: Optional[Literal["sequential", "parallel"]]
    primary_library: Optional[Literal["econml", "causalml"]]  # Main library for CATE/uplift
    libraries_executed: Optional[List[str]]  # Actually executed libraries
    libraries_skipped: Optional[List[str]]  # Skipped due to validation or errors

    # EconML-specific outputs (CATE from CausalForestDML/OrthoForest)
    econml_cate_result: Optional[Dict[str, Any]]  # Full CATE result from EconML
    econml_model_used: Optional[str]  # "CausalForestDML", "OrthoForest", etc.
    econml_latency_ms: Optional[int]

    # CausalML-specific outputs (Uplift from Random Forest/Gradient Boosting)
    causalml_uplift_result: Optional[Dict[str, Any]]  # Full uplift result from CausalML
    causalml_model_used: Optional[str]  # "random_forest", "gradient_boosting"
    causalml_latency_ms: Optional[int]

    # Cross-library validation (DoWhy ↔ CausalML)
    cross_library_validation: Optional[Dict[str, Any]]  # Validation results between libraries
    econml_causalml_agreement: Optional[float]  # Agreement between EconML CATE and CausalML uplift (0-1)
    dowhy_validation_result: Optional[Dict[str, Any]]  # DoWhy validation of CATE estimates
    validation_passed: Optional[bool]  # Whether cross-validation passed thresholds

    # Multi-library consensus
    library_consensus_effect: Optional[float]  # Confidence-weighted consensus CATE
    library_agreement_score: Optional[float]  # Overall agreement between libraries (0-1)
    effect_estimate_variance: Optional[float]  # Variance across library estimates

    # Multi-library routing metadata
    question_type: Optional[Literal[
        "effect_heterogeneity",  # "How does effect vary?" → EconML primary
        "targeting",  # "Who should we target?" → CausalML primary
        "segment_optimization",  # Both EconML and CausalML
        "comprehensive",  # All libraries including DoWhy validation
    ]]
    routing_confidence: Optional[float]  # Confidence in library routing decision
    routing_rationale: Optional[str]  # Why this library routing was chosen

    # Audit chain (tamper-evident logging)
    audit_workflow_id: Optional[UUID]

    # ========================================================================
    # V4.4: Causal Discovery Integration
    # ========================================================================

    # Discovered DAG from causal discovery module
    discovered_dag_adjacency: Optional[List[List[int]]]  # Adjacency matrix
    discovered_dag_nodes: Optional[List[str]]  # Node names
    discovered_dag_edge_types: Optional[Dict[str, str]]  # Edge types (DIRECTED, BIDIRECTED, UNDIRECTED)

    # Discovery gate decision
    discovery_gate_decision: Optional[Literal["accept", "review", "reject", "augment"]]
    discovery_gate_confidence: Optional[float]  # Gate confidence [0, 1]

    # DAG validation outputs
    dag_validated_segments: Optional[List[str]]  # Segments with valid causal paths
    dag_invalid_segments: Optional[List[str]]  # Segments without causal paths
    latent_confounder_segments: Optional[List[str]]  # Segments with bidirected edges (latent confounders)
    dag_validation_warnings: Optional[List[str]]  # Warnings from DAG validation


class HeterogeneousOptimizerInput(TypedDict):
    """Input contract for Heterogeneous Optimizer agent (from orchestrator).

    Contract: .claude/contracts/tier2-contracts.md lines 401-600
    """

    query: str
    treatment_var: str
    outcome_var: str
    segment_vars: List[str]
    effect_modifiers: Optional[List[str]]
    data_source: str
    filters: Optional[Dict[str, Any]]
    n_estimators: Optional[int]
    min_samples_leaf: Optional[int]
    significance_level: Optional[float]
    top_segments_count: Optional[int]
    # Memory integration (optional)
    session_id: Optional[str]  # For memory context retrieval/storage
    brand: Optional[str]  # Brand context for memory
    region: Optional[str]  # Region context for memory


class HeterogeneousOptimizerOutput(TypedDict):
    """Output contract for Heterogeneous Optimizer agent (to orchestrator).

    Contract: .claude/contracts/tier2-contracts.md lines 401-600
    """

    # Core results
    cate_by_segment: Optional[Dict[str, List[CATEResult]]]
    overall_ate: Optional[float]
    heterogeneity_score: Optional[float]

    # Uplift results (from CausalML integration)
    uplift_by_segment: Optional[Dict[str, List[Dict[str, Any]]]]
    overall_auuc: Optional[float]
    overall_qini: Optional[float]
    targeting_efficiency: Optional[float]

    # B9.4: Hierarchical nesting results (EconML within CausalML segments)
    hierarchical_segment_results: Optional[List[Dict[str, Any]]]
    nested_ci: Optional[Dict[str, Any]]
    segment_heterogeneity_score: Optional[float]  # I² statistic
    overall_hierarchical_ate: Optional[float]
    overall_hierarchical_ci_lower: Optional[float]
    overall_hierarchical_ci_upper: Optional[float]
    n_segments_analyzed: Optional[int]
    segmentation_method_used: Optional[str]
    hierarchical_estimator_type: Optional[str]
    hierarchical_latency_ms: Optional[int]

    # Segment analysis
    high_responders: Optional[List[SegmentProfile]]
    low_responders: Optional[List[SegmentProfile]]

    # Policy recommendations
    policy_recommendations: Optional[List[PolicyRecommendation]]
    expected_total_lift: float  # Contract requires non-optional
    optimal_allocation_summary: Optional[str]

    # Summary
    executive_summary: Optional[str]
    key_insights: Optional[List[str]]

    # Latency
    total_latency_ms: int

    # Contract-required fields
    confidence: float  # Overall analysis confidence (0.0-1.0)
    requires_further_analysis: bool  # Whether further analysis is recommended
    suggested_next_agent: Optional[str]  # Next agent to invoke

    # B7.4: Multi-Library Support Output
    libraries_used: Optional[List[str]]  # Libraries that were executed
    library_execution_mode: Optional[Literal["sequential", "parallel"]]  # How libraries were executed
    primary_library: Optional[str]  # Main library used for the analysis
    library_agreement_score: Optional[float]  # Agreement between libraries (0-1)
    library_consensus_effect: Optional[float]  # Confidence-weighted consensus CATE
    cross_library_validation: Optional[Dict[str, Any]]  # Cross-library validation results
    validation_passed: Optional[bool]  # Whether cross-validation passed thresholds
    question_type: Optional[str]  # Type of question that determined routing

    # Status
    status: Literal["completed", "failed"]
    warnings: List[str]
    errors: List[Dict[str, Any]]
