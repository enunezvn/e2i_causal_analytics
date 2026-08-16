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


#: The documented values of ``validity_audit_status`` (#1639). ``unknown`` is
#: the explicit out-of-band member: a status we did not write and cannot map.
#:
#: Defined here rather than beside either consumer because BOTH the template
#: generator and the agent's public output publish this field, and a second
#: copy is a second thing to forget -- which is exactly how the raw
#: passthrough at the output boundary survived the guard added one round
#: earlier.
AUDIT_STATUSES = frozenset({"completed", "skipped", "timed_out", "failed", "not_run", "unknown"})


def normalize_audit_status(value: object) -> str:
    """Coerce a recorded audit status onto :data:`AUDIT_STATUSES`.

    A hydrated checkpoint can carry the previous BAD value ``"was skipped"``
    (human prose in a machine field, fixed in this branch) or a typo like
    ``"timeout"``. Either would land in the documented enum and match nothing
    for a consumer filtering on it.

    Out-of-band values become ``"unknown"`` -- never ``"not_run"``, which would
    ASSERT that the audit did not run when all we know is that we cannot read
    what it said.
    """
    # `str(...)` rather than returning `value`: membership in AUDIT_STATUSES
    # guarantees it IS one of those strings, so this is exact, and the
    # declared `object` parameter otherwise leaks out as the return type.
    return str(value) if value in AUDIT_STATUSES else "unknown"


def infer_audit_status(
    explicit: object = None, *, has_threats: bool = False, score: object = None
) -> str:
    """The ONE rule for deciding what an audit's status was (#1639).

    An explicit status always wins. Otherwise the only honest readings are:

    * threats were found, or a score above zero was recorded -> ``completed``;
      something produced a verdict.
    * nothing at all -> ``unknown``, NOT ``not_run``.

    That last distinction is codex iter-14's point and it is right: ``0.0`` is a
    VALID validity score, so `bool(score)` cannot tell "the audit completed and
    scored zero with no threats" apart from "no audit happened". Reporting
    ``not_run`` there asserts something we do not know. ``unknown`` says only
    what is true -- which is the whole reason that member exists.
    """
    if explicit is not None:
        return normalize_audit_status(explicit)
    numeric = score if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0
    if has_threats or numeric > 0:
        return "completed"
    return "unknown"


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
    #: What ``minimum_detectable_effect`` is measured in (#1639). Distinct from
    #: ``effect_size_type``, which describes the INPUT effect and cannot express
    #: an absolute risk difference — leaving a reader to compare a 0.0015
    #: absolute MDE against a 0.030 RELATIVE effect and see a contradiction.
    minimum_detectable_effect_scale: NotRequired[str]
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
    #: #1639. This template is the EXECUTION artifact: it carries sample_size
    #: and duration_days into checkpoints and timelines. A consumer holding only
    #: the template must be able to see that the design cannot be run.
    feasibility_warnings: NotRequired[list[str]]


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
    # Optional: seeded from ExperimentDesignerInput, which allows None (#705 H8).
    intervention_type: NotRequired[Optional[str]]
    brand: NotRequired[Optional[str]]
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
    #: #1639. Set unconditionally by the power-analysis node, so an empty list
    #: means "checked, feasible" rather than "never checked".
    feasibility_warnings: NotRequired[list[str]]
    interim_analysis_schedule: NotRequired[list[dict[str, Any]]]

    # Top-level exposure for quality gates and easy access (v4.3)
    required_sample_size: NotRequired[int]  # Exposed from power_analysis
    statistical_power: NotRequired[float]  # Exposed from power_analysis

    # ===== Validity Audit Outputs =====
    # Note: Required outputs from validity audit node
    validity_threats: list[ValidityThreat]
    #: #1639. Whether the audit reached a verdict: completed | skipped |
    #: timed_out | failed. Absent means the node never executed. Without this,
    #: an empty ``validity_threats`` is indistinguishable from a clean audit.
    validity_audit_status: NotRequired[str]
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
