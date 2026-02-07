"""Experiment Designer Agent State Definitions.

This module defines the TypedDict state structures for the experiment designer agent's
LangGraph workflow.

Contract: .claude/contracts/tier3-contracts.md lines 82-200
Specialist: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md
"""

from typing import Any, Literal, Optional
from uuid import UUID

from typing_extensions import NotRequired, TypedDict

# ===== TYPE ALIASES =====

AgentStatus = Literal[
    "pending",
    "loading_context",
    "simulating_twins",  # Phase 15: Twin simulation step
    "reasoning",
    "calculating",
    "auditing",
    "redesigning",
    "generating",
    "completed",
    "skipped",  # Phase 15: Experiment skipped due to twin recommendation
    "failed",
]
FormalityLevel = Literal["light", "medium", "heavy"]
DesignType = Literal[
    "RCT",
    "cluster_rct",  # Added: cluster randomized controlled trial
    "quasi_experiment",
    "difference_in_differences",
    "regression_discontinuity",
    "instrumental_variable",
    "synthetic_control",
]
RandomizationUnit = Literal["individual", "cluster", "time_period", "geography", "territory"]
ValidityThreatSeverity = Literal["low", "medium", "high", "critical"]
ConfidenceLevel = Literal["low", "medium", "high"]


# ===== NESTED TYPED DICTS =====


class TreatmentDefinition(TypedDict):
    """Definition of a treatment arm in the experiment.

    Contract: .claude/contracts/tier3-contracts.md lines 145-155
    """

    name: str
    description: str
    implementation_details: str
    target_population: str
    dosage_or_intensity: NotRequired[str]
    duration: NotRequired[str]
    delivery_mechanism: NotRequired[str]


class OutcomeDefinition(TypedDict):
    """Definition of an outcome metric to measure.

    Contract: .claude/contracts/tier3-contracts.md lines 157-170
    """

    name: str
    metric_type: Literal["continuous", "binary", "count", "time_to_event"]
    measurement_method: str
    measurement_frequency: str
    baseline_value: NotRequired[float]
    expected_effect_size: NotRequired[float]
    minimum_detectable_effect: NotRequired[float]
    is_primary: bool


class ValidityThreat(TypedDict):
    """Identified threat to experimental validity.

    Contract: .claude/contracts/tier3-contracts.md lines 172-185
    """

    threat_type: Literal["internal", "external", "construct", "statistical_conclusion"]
    threat_name: str
    description: str
    severity: ValidityThreatSeverity
    affected_outcomes: list[str]
    mitigation_possible: bool
    mitigation_strategy: NotRequired[str]


class MitigationRecommendation(TypedDict):
    """Recommended mitigation for a validity threat.

    Contract: .claude/contracts/tier3-contracts.md lines 187-198
    """

    threat_addressed: str
    strategy: str
    implementation_steps: list[str]
    cost_estimate: NotRequired[str]
    effectiveness_rating: Literal["low", "medium", "high"]
    trade_offs: list[str]


class PowerAnalysisResult(TypedDict):
    """Results from statistical power analysis.

    Contract: .claude/contracts/tier3-contracts.md lines 200-215
    """

    required_sample_size: int
    required_sample_size_per_arm: int
    achieved_power: float
    minimum_detectable_effect: float
    alpha: float
    effect_size_type: Literal["cohens_d", "odds_ratio", "rate_ratio", "percentage_change"]
    assumptions: list[str]
    sensitivity_analysis: NotRequired[dict[str, Any]]


class DoWhySpec(TypedDict):
    """DoWhy causal model specification.

    Contract: .claude/contracts/tier3-contracts.md lines 217-230
    """

    treatment_variable: str
    outcome_variable: str
    common_causes: list[str]
    instruments: NotRequired[list[str]]
    effect_modifiers: NotRequired[list[str]]
    graph_dot: str
    identification_strategy: str


class ExperimentTemplate(TypedDict):
    """Generated experiment template for execution.

    Contract: .claude/contracts/tier3-contracts.md lines 232-250
    """

    template_id: str
    template_version: str
    design_summary: str
    treatments: list[TreatmentDefinition]
    outcomes: list[OutcomeDefinition]
    sample_size: int
    duration_days: int
    randomization_unit: RandomizationUnit
    randomization_method: str
    blocking_variables: NotRequired[list[str]]
    stratification_variables: NotRequired[list[str]]
    pre_registration_document: NotRequired[str]
    analysis_code_template: NotRequired[str]
    monitoring_checkpoints: list[dict[str, Any]]


class ErrorDetails(TypedDict):
    """Error information for debugging.

    Contract: .claude/contracts/tier3-contracts.md lines 252-260
    """

    node: str
    error: str
    timestamp: str
    recoverable: NotRequired[bool]
    retry_count: NotRequired[int]


class DesignIteration(TypedDict):
    """Record of a design iteration in the redesign loop.

    Specialist: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md lines 450-470
    """

    iteration_number: int
    design_type: DesignType
    validity_threats_identified: int
    critical_threats: int
    power_achieved: float
    redesign_reason: NotRequired[str]
    timestamp: str


# ===== MAIN STATE =====


class ExperimentDesignState(TypedDict):
    """Complete state for experiment designer agent workflow.

    This state flows through all nodes in the graph:
    context_loader → design_reasoning → power_analysis → validity_audit →
    (conditional redesign) → template_generator

    Contract: .claude/contracts/tier3-contracts.md lines 82-142
    Specialist: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md
    """

    # ===== Input Fields =====
    # Note: Input fields may not be in output state (consumed during processing)
    business_question: NotRequired[str]
    constraints: NotRequired[dict[str, Any]]
    available_data: NotRequired[dict[str, Any]]
    preregistration_formality: NotRequired[FormalityLevel]
    max_redesign_iterations: NotRequired[int]
    enable_validity_audit: NotRequired[bool]

    # ===== Digital Twin Pre-Screening =====
    # Added in Phase 15 for twin simulation integration
    enable_twin_simulation: NotRequired[bool]
    intervention_type: NotRequired[str]
    brand: NotRequired[str]
    treatment_variable: NotRequired[str]
    outcome_variable: NotRequired[str]

    # ===== Organizational Context =====
    historical_experiments: NotRequired[list[dict[str, Any]]]
    domain_knowledge: NotRequired[dict[str, Any]]
    regulatory_requirements: NotRequired[list[str]]
    budget_constraints: NotRequired[dict[str, Any]]
    timeline_constraints: NotRequired[dict[str, Any]]

    # ===== Twin Simulation Outputs =====
    # Phase 15: Digital Twin pre-screening results
    twin_simulation_result: NotRequired[dict[str, Any]]
    twin_recommendation: NotRequired[str]  # "deploy", "skip", "refine"
    twin_simulated_ate: NotRequired[float]
    twin_recommended_sample_size: NotRequired[int]
    twin_top_segments: NotRequired[list[dict[str, Any]]]
    skip_experiment: NotRequired[bool]  # True if twin recommends skip

    # ===== Design Reasoning Outputs =====
    # Note: Required outputs from design reasoning node
    design_type: DesignType
    design_rationale: str
    treatments: NotRequired[list[TreatmentDefinition]]
    outcomes: NotRequired[list[OutcomeDefinition]]
    randomization_unit: NotRequired[RandomizationUnit]
    randomization_method: NotRequired[str]
    blocking_variables: NotRequired[list[str]]
    stratification_variables: NotRequired[list[str]]
    causal_assumptions: NotRequired[list[str]]

    # ===== Power Analysis Outputs =====
    power_analysis: NotRequired[PowerAnalysisResult]
    sample_size_justification: NotRequired[str]
    duration_estimate_days: NotRequired[int]
    interim_analysis_schedule: NotRequired[list[dict[str, Any]]]

    # Top-level exposure for quality gates and easy access (v4.3)
    required_sample_size: NotRequired[int]  # Exposed from power_analysis
    statistical_power: NotRequired[float]  # Exposed from power_analysis

    # ===== Validity Audit Outputs =====
    # Note: Required outputs from validity audit node
    validity_threats: list[ValidityThreat]
    mitigations: NotRequired[list[MitigationRecommendation]]
    overall_validity_score: float
    validity_confidence: NotRequired[ConfidenceLevel]
    redesign_needed: NotRequired[bool]
    redesign_recommendations: NotRequired[list[str]]

    # ===== DoWhy Integration Outputs =====
    dowhy_spec: NotRequired[DoWhySpec]
    causal_graph_dot: NotRequired[str]
    identification_result: NotRequired[dict[str, Any]]
    estimand: NotRequired[str]

    # ===== Template Generation Outputs =====
    experiment_template: NotRequired[ExperimentTemplate]
    analysis_code: NotRequired[str]
    monitoring_dashboard_spec: NotRequired[dict[str, Any]]

    # ===== Execution Metadata =====
    current_iteration: NotRequired[int]
    iteration_history: NotRequired[list[DesignIteration]]
    total_llm_tokens_used: NotRequired[int]
    node_latencies_ms: NotRequired[dict[str, int]]
    preregistration_document: NotRequired[str]
    redesign_iterations: NotRequired[int]

    # ===== Contract-Required Output Fields =====
    total_latency_ms: int  # Contract requires this for all Tier 3 agents
    timestamp: str  # Contract requires this for all Tier 3 agents

    # ===== Error Handling =====
    errors: list[ErrorDetails]
    warnings: list[str]
    status: AgentStatus

    # ===== Audit Chain =====
    audit_workflow_id: NotRequired[Optional[UUID]]

    # ========================================================================
    # V4.4: Causal Discovery Integration
    # ========================================================================

    # Discovered DAG from causal discovery module
    discovered_dag_adjacency: NotRequired[list[list[int]]]  # Adjacency matrix
    discovered_dag_nodes: NotRequired[list[str]]  # Node names
    discovered_dag_edge_types: NotRequired[
        dict[str, str]
    ]  # Edge types (DIRECTED, BIDIRECTED, UNDIRECTED)

    # Discovery gate decision
    discovery_gate_decision: NotRequired[Literal["accept", "review", "reject", "augment"]]
    discovery_gate_confidence: NotRequired[float]  # Gate confidence [0, 1]

    # DAG-aware validity enhancements
    dag_confounders_validated: NotRequired[list[str]]  # Confounders in DAG that need control
    dag_missing_confounders: NotRequired[list[str]]  # Assumed confounders NOT in DAG
    dag_latent_confounders: NotRequired[list[str]]  # Latent confounders from FCI bidirected edges
    dag_instrument_candidates: NotRequired[list[str]]  # Valid IV candidates from DAG
    dag_effect_modifiers: NotRequired[list[str]]  # Effect modifiers identified from DAG
    dag_validation_warnings: NotRequired[list[str]]  # Warnings from DAG validation
