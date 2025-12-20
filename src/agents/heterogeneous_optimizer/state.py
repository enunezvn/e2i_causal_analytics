"""State definitions for Heterogeneous Optimizer Agent.

This module defines the LangGraph state structure for segment-level CATE analysis.
"""

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator


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

    # === INPUT (7 fields) ===
    query: str
    treatment_var: str
    outcome_var: str
    segment_vars: List[str]  # Variables to segment by
    effect_modifiers: List[str]  # Variables that modify treatment effect
    data_source: str
    filters: Optional[Dict[str, Any]]

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


class HeterogeneousOptimizerOutput(TypedDict):
    """Output contract for Heterogeneous Optimizer agent (to orchestrator).

    Contract: .claude/contracts/tier2-contracts.md lines 401-600
    """

    # Core results
    cate_by_segment: Optional[Dict[str, List[CATEResult]]]
    overall_ate: Optional[float]
    heterogeneity_score: Optional[float]

    # Segment analysis
    high_responders: Optional[List[SegmentProfile]]
    low_responders: Optional[List[SegmentProfile]]

    # Policy recommendations
    policy_recommendations: Optional[List[PolicyRecommendation]]
    expected_total_lift: Optional[float]
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

    # Status
    status: Literal["completed", "failed"]
    warnings: List[str]
    errors: List[Dict[str, Any]]
