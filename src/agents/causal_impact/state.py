"""State definition for Causal Impact Agent.

This module defines the TypedDict state used throughout the causal impact workflow.
Contract: .claude/contracts/tier2-contracts.md
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from typing_extensions import NotRequired


class CausalGraph(TypedDict, total=False):
    """Causal DAG representation."""

    nodes: List[str]  # Variable names
    edges: List[tuple[str, str]]  # (from, to) tuples
    treatment_nodes: List[str]
    outcome_nodes: List[str]
    adjustment_sets: List[List[str]]  # Valid backdoor adjustment sets
    dag_dot: str  # DOT format for visualization
    confidence: float  # Graph construction confidence (0-1)
    dag_version_hash: str  # SHA256 hash for expert review tracking


class EstimationResult(TypedDict, total=False):
    """Causal effect estimation results."""

    method: Literal[
        "CausalForestDML",
        "LinearDML",
        "linear_regression",
        "propensity_score_weighting",
    ]
    ate: float  # Average Treatment Effect
    ate_ci_lower: float  # 95% confidence interval lower bound
    ate_ci_upper: float  # 95% confidence interval upper bound
    standard_error: float  # Standard error of the ATE estimate
    cate_segments: NotRequired[List[Dict[str, Any]]]  # Conditional ATE by segment
    effect_size: str  # "small", "medium", "large"
    statistical_significance: bool
    p_value: float
    sample_size: int
    covariates_adjusted: List[str]
    heterogeneity_detected: bool  # Whether CATE varies significantly


class RefutationTest(TypedDict, total=False):
    """Single refutation test result."""

    test_name: Literal[
        "placebo_treatment",
        "random_common_cause",
        "data_subset_validation",
        "data_subset",  # Alias for data_subset_validation
        "bootstrap",
        "sensitivity_e_value",  # E-value sensitivity analysis
        "unobserved_common_cause",  # Contract key for sensitivity test
    ]
    passed: bool  # Whether effect survived refutation
    new_effect: float  # Effect after refutation
    original_effect: float  # Original ATE for comparison
    p_value: float
    details: str  # Human-readable explanation


class RefutationResults(TypedDict, total=False):
    """Complete refutation analysis.

    Contract: individual_tests must be Dict with test names as keys:
    - placebo_treatment
    - random_common_cause
    - data_subset
    - unobserved_common_cause
    """

    tests_passed: int
    tests_failed: int
    total_tests: int
    overall_robust: bool  # True if majority of tests passed
    # Contract: Dict by test name, NOT List
    individual_tests: Dict[str, RefutationTest]
    confidence_adjustment: float  # Multiplier for final confidence (0-1)
    gate_decision: NotRequired[Literal["proceed", "review", "block"]]  # Validation gate


class SensitivityAnalysis(TypedDict, total=False):
    """Sensitivity to unmeasured confounding."""

    e_value: float  # E-value for point estimate
    e_value_ci: float  # E-value for CI bound
    interpretation: str  # What E-value means in context
    robust_to_confounding: bool  # Whether effect is robust
    unmeasured_confounder_strength: str  # "weak", "moderate", "strong" threshold


class NaturalLanguageInterpretation(TypedDict, total=False):
    """Deep reasoning interpretation output."""

    narrative: str  # Main interpretation (200-800 words)
    key_findings: List[str]  # 3-5 bullet points
    effect_magnitude: str  # Qualitative description
    causal_confidence: str  # "low", "medium", "high"
    assumptions_made: List[str]  # Key assumptions stated
    limitations: List[str]  # Caveats and limitations
    recommendations: List[str]  # Actionable next steps
    depth_level: Literal["none", "minimal", "standard", "deep"]
    user_expertise_adjusted: bool


class CausalImpactState(TypedDict):
    """Complete state for causal impact workflow.

    Follows 5-node pipeline:
    1. graph_builder: Construct causal DAG
    2. estimation: Estimate causal effect
    3. refutation: Robustness tests
    4. sensitivity: Sensitivity analysis
    5. interpretation: Natural language output
    """

    # Input fields (from orchestrator) - Contract REQUIRED fields
    query: str
    query_id: str
    treatment_var: str  # REQUIRED per contract - e.g., "hcp_engagement_level"
    outcome_var: str  # REQUIRED per contract - e.g., "patient_conversion_rate"
    confounders: List[str]  # REQUIRED per contract - adjustment variables
    data_source: str  # REQUIRED per contract - data source identifier

    # Input fields - Optional
    mediators: NotRequired[List[str]]  # Optional mediator variables
    effect_modifiers: NotRequired[List[str]]  # Optional effect modifiers
    instruments: NotRequired[List[str]]  # Optional instrumental variables
    segment_filters: NotRequired[Dict[str, Any]]  # Optional segment restrictions
    interpretation_depth: NotRequired[Literal["none", "minimal", "standard", "deep"]]
    user_context: NotRequired[Dict[str, Any]]  # expertise level, preferences
    parameters: NotRequired[Dict[str, Any]]  # Agent-specific parameters
    time_period: NotRequired[Dict[str, str]]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    brand: NotRequired[str]  # Brand context

    # Graph builder outputs
    causal_graph: NotRequired[CausalGraph]
    dag_version_hash: NotRequired[str]  # SHA256 hash for expert review tracking
    graph_builder_latency_ms: NotRequired[float]
    graph_builder_error: NotRequired[str]

    # Estimation outputs
    estimation_result: NotRequired[EstimationResult]
    estimation_latency_ms: NotRequired[float]
    estimation_error: NotRequired[str]

    # Refutation outputs
    refutation_results: NotRequired[RefutationResults]
    refutation_latency_ms: NotRequired[float]
    refutation_error: NotRequired[str]

    # Sensitivity outputs
    sensitivity_analysis: NotRequired[SensitivityAnalysis]
    sensitivity_latency_ms: NotRequired[float]
    sensitivity_error: NotRequired[str]

    # Interpretation outputs (final agent output)
    interpretation: NotRequired[NaturalLanguageInterpretation]
    interpretation_latency_ms: NotRequired[float]
    interpretation_error: NotRequired[str]

    # Workflow metadata
    current_phase: NotRequired[
        Literal[
            "graph_building",
            "estimating",
            "refuting",
            "analyzing_sensitivity",
            "interpreting",
            "completed",
            "failed",
        ]
    ]
    # Contract: status progression is pending → computing → interpreting → completed/failed
    status: NotRequired[Literal["pending", "computing", "interpreting", "completed", "failed"]]
    total_latency_ms: NotRequired[float]
    timestamp: NotRequired[str]  # ISO 8601
    error_message: NotRequired[str]

    # Data access (injected by dispatcher)
    data_cache: NotRequired[Dict[str, Any]]  # Cached query results
    historical_analyses: NotRequired[List[Dict[str, Any]]]  # Similar past analyses

    # Agent coordination
    handoff_to: NotRequired[str]  # Next agent in multi-agent flow
    confidence_score: NotRequired[float]  # Overall confidence (0-1)
    requires_followup: NotRequired[bool]
    followup_suggestions: NotRequired[List[str]]

    # Error handling (contract: operator.add accumulators for LangGraph)
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    fallback_used: NotRequired[bool]
    retry_count: NotRequired[int]
    refutation_passed: NotRequired[bool]


class CausalImpactInput(TypedDict):
    """Input contract for CausalImpact agent (from orchestrator).

    Contract: .claude/contracts/tier2-contracts.md
    """

    # REQUIRED fields per contract
    query: str
    treatment_var: str  # REQUIRED - treatment variable name
    outcome_var: str  # REQUIRED - outcome variable name
    confounders: List[str]  # REQUIRED - adjustment variables
    data_source: str  # REQUIRED - data source identifier

    # Optional fields
    mediators: NotRequired[List[str]]  # Optional mediator variables
    effect_modifiers: NotRequired[List[str]]  # Optional effect modifiers
    instruments: NotRequired[List[str]]  # Optional instrumental variables
    segment_filters: NotRequired[Dict[str, Any]]
    interpretation_depth: NotRequired[Literal["none", "minimal", "standard", "deep"]]
    user_context: NotRequired[Dict[str, Any]]
    parameters: NotRequired[Dict[str, Any]]
    time_period: NotRequired[Dict[str, str]]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    brand: NotRequired[str]  # Brand context


class CausalImpactOutput(TypedDict):
    """Output contract for CausalImpact agent (to orchestrator).

    Contract: .claude/contracts/tier2-contracts.md
    """

    # Required fields
    query_id: str
    status: Literal["completed", "failed"]

    # Core results - Contract field names
    causal_narrative: str
    ate_estimate: NotRequired[float]
    confidence_interval: NotRequired[tuple[float, float]]
    standard_error: NotRequired[float]
    statistical_significance: bool
    p_value: NotRequired[float]
    effect_type: NotRequired[str]  # "ate", "cate", "itt", etc.
    estimation_method: NotRequired[str]

    # Contract REQUIRED fields (renamed/added per contract)
    confidence: float  # Contract field name (was overall_confidence)
    model_used: str  # Contract REQUIRED - estimation method name
    key_insights: List[str]  # Contract REQUIRED - bullet points
    assumption_warnings: List[str]  # Contract REQUIRED - assumption violations
    actionable_recommendations: List[str]  # Contract field name (was recommendations)
    requires_further_analysis: bool  # Contract REQUIRED
    refutation_passed: bool  # Contract REQUIRED - overall refutation status
    executive_summary: str  # Contract REQUIRED - 2-3 sentence summary

    # Rich metadata
    mechanism_explanation: NotRequired[str]
    causal_graph_summary: NotRequired[str]
    key_assumptions: List[str]
    limitations: List[str]

    # Performance metrics
    computation_latency_ms: float
    interpretation_latency_ms: float
    total_latency_ms: float

    # Robustness indicators
    refutation_tests_passed: NotRequired[int]
    refutation_tests_total: NotRequired[int]
    sensitivity_e_value: NotRequired[float]

    # Visualizations (optional)
    visualizations: NotRequired[List[Dict[str, Any]]]

    # Follow-up support
    follow_up_suggestions: List[str]
    citations: List[str]

    # Error handling
    error_message: NotRequired[str]
    partial_results: NotRequired[bool]
