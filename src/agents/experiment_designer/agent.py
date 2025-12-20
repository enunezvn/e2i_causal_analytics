"""Experiment Designer Agent.

This agent designs rigorous experiments for causal inference in pharmaceutical
commercial operations.

Tier: 3 (Monitoring & Design)
Agent Type: Hybrid (Deep Reasoning + Computation)
Performance Target: <60s total latency

Key Features:
- Deep reasoning for experiment design (LLM)
- Statistical power analysis
- Adversarial validity audit (LLM)
- DoWhy code generation
- Pre-registration document generation

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md
Contract: .claude/contracts/tier3-contracts.md lines 82-142
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.agents.experiment_designer.state import ExperimentDesignState
from src.agents.experiment_designer.graph import (
    create_experiment_designer_graph,
    experiment_designer_graph,
)


# ===== INPUT/OUTPUT MODELS =====


class ExperimentDesignerInput(BaseModel):
    """Input model for Experiment Designer Agent.

    Contract: .claude/contracts/tier3-contracts.md lines 82-129
    """

    # Required fields
    business_question: str = Field(
        ...,
        min_length=10,
        description="The business question to design an experiment for",
    )

    # Optional fields with defaults
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Experimental constraints (budget, timeline, ethical, operational)",
    )
    available_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Available data sources and variables for the experiment",
    )

    # Configuration
    preregistration_formality: Literal["light", "medium", "heavy"] = Field(
        "medium",
        description="Level of pre-registration document detail",
    )
    max_redesign_iterations: int = Field(
        2,
        ge=0,
        le=5,
        description="Maximum number of design iterations based on validity audit",
    )
    enable_validity_audit: bool = Field(
        True,
        description="Whether to run adversarial validity audit",
    )

    # Optional context
    brand: Optional[str] = Field(
        None,
        description="Brand filter for domain-specific context",
    )

    @field_validator("constraints")
    @classmethod
    def validate_constraints(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate constraint structure."""
        valid_keys = {
            "budget",
            "timeline",
            "ethical",
            "operational",
            "expected_effect_size",
            "alpha",
            "power",
            "weekly_accrual",
            "cluster_size",
            "expected_icc",
            "baseline_rate",
        }
        for key in v.keys():
            if key not in valid_keys:
                # Allow unknown keys but log a warning would be helpful
                pass
        return v


class TreatmentOutput(BaseModel):
    """Treatment definition in output."""

    name: str
    description: str
    implementation_details: str
    target_population: str
    dosage_or_intensity: Optional[str] = None
    duration: Optional[str] = None


class OutcomeOutput(BaseModel):
    """Outcome definition in output."""

    name: str
    metric_type: str
    measurement_method: str
    measurement_frequency: str
    is_primary: bool
    baseline_value: Optional[float] = None
    expected_effect_size: Optional[float] = None


class ValidityThreatOutput(BaseModel):
    """Validity threat in output."""

    threat_type: str
    threat_name: str
    description: str
    severity: str
    mitigation_possible: bool
    mitigation_strategy: Optional[str] = None


class PowerAnalysisOutput(BaseModel):
    """Power analysis results in output."""

    required_sample_size: int
    required_sample_size_per_arm: int
    achieved_power: float
    minimum_detectable_effect: float
    alpha: float
    effect_size_type: str
    assumptions: list[str] = Field(default_factory=list)


class ExperimentDesignerOutput(BaseModel):
    """Output model for Experiment Designer Agent.

    Contract: .claude/contracts/tier3-contracts.md lines 131-200
    """

    # Design outputs
    design_type: str = Field(..., description="Type of experimental design")
    design_rationale: str = Field(..., description="Rationale for chosen design")
    treatments: list[TreatmentOutput] = Field(
        default_factory=list, description="Treatment definitions"
    )
    outcomes: list[OutcomeOutput] = Field(
        default_factory=list, description="Outcome definitions"
    )
    randomization_unit: str = Field(
        "individual", description="Unit of randomization"
    )
    randomization_method: str = Field("simple", description="Randomization method")
    stratification_variables: list[str] = Field(
        default_factory=list, description="Variables used for stratification"
    )
    blocking_variables: list[str] = Field(
        default_factory=list, description="Variables used for blocking"
    )

    # Power analysis outputs
    power_analysis: Optional[PowerAnalysisOutput] = Field(
        None, description="Power analysis results"
    )
    sample_size_justification: str = Field(
        "", description="Justification for sample size"
    )
    duration_estimate_days: int = Field(0, description="Estimated experiment duration")

    # Validity audit outputs
    validity_threats: list[ValidityThreatOutput] = Field(
        default_factory=list, description="Identified validity threats"
    )
    overall_validity_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Overall validity score"
    )
    validity_confidence: str = Field("low", description="Confidence in validity assessment")

    # Generated templates
    causal_graph_dot: str = Field("", description="DOT format causal graph")
    analysis_code: str = Field("", description="Python analysis code template")
    preregistration_document: str = Field(
        "", description="Pre-registration document"
    )

    # Execution metadata
    total_latency_ms: int = Field(0, description="Total execution latency")
    redesign_iterations: int = Field(0, description="Number of redesign iterations")
    warnings: list[str] = Field(default_factory=list, description="Warnings")


# ===== MAIN AGENT =====


class ExperimentDesignerAgent:
    """Experiment Designer Agent - Designs rigorous experiments for causal inference.

    This agent designs experiments to answer business questions using:
    1. Deep reasoning for strategic design choices
    2. Statistical power analysis for sample sizing
    3. Adversarial validity audit for threat identification
    4. DoWhy code generation for analysis templates

    Usage:
        agent = ExperimentDesignerAgent()
        result = agent.run(ExperimentDesignerInput(
            business_question="Does increasing rep visit frequency improve HCP engagement?",
            constraints={
                "expected_effect_size": 0.25,
                "power": 0.80,
                "weekly_accrual": 50
            },
            available_data={
                "variables": ["hcp_id", "territory", "visit_count", "engagement_score"]
            }
        ))

        print(f"Design: {result.design_type}")
        print(f"Sample Size: {result.power_analysis.required_sample_size}")
        print(f"Validity Score: {result.overall_validity_score}")

    Performance:
        - Total latency: <60s
        - Design reasoning: <30s
        - Power analysis: <100ms
        - Validity audit: <30s
        - Template generation: <500ms
    """

    def __init__(self, max_redesign_iterations: int = 2):
        """Initialize experiment designer agent.

        Args:
            max_redesign_iterations: Maximum number of design iterations
        """
        self.max_redesign_iterations = max_redesign_iterations
        self.graph = create_experiment_designer_graph(
            max_redesign_iterations=max_redesign_iterations
        )

    def run(self, input_data: ExperimentDesignerInput) -> ExperimentDesignerOutput:
        """Execute experiment design workflow.

        Args:
            input_data: Validated input parameters

        Returns:
            ExperimentDesignerOutput with design results and templates

        Raises:
            ValueError: If input validation fails
            RuntimeError: If experiment design fails
        """
        # Create initial state from input
        initial_state = self._create_initial_state(input_data)

        # Execute graph
        final_state = self.graph.invoke(initial_state)

        # Convert to output model
        output = self._create_output(final_state)

        # Check for failures
        if final_state.get("status") == "failed":
            error_messages = [e.get("error", "Unknown") for e in final_state.get("errors", [])]
            raise RuntimeError(f"Experiment design failed: {'; '.join(error_messages)}")

        return output

    async def arun(self, input_data: ExperimentDesignerInput) -> ExperimentDesignerOutput:
        """Execute experiment design workflow asynchronously.

        Args:
            input_data: Validated input parameters

        Returns:
            ExperimentDesignerOutput with design results and templates

        Raises:
            ValueError: If input validation fails
            RuntimeError: If experiment design fails
        """
        # Create initial state from input
        initial_state = self._create_initial_state(input_data)

        # Execute graph asynchronously
        final_state = await self.graph.ainvoke(initial_state)

        # Convert to output model
        output = self._create_output(final_state)

        # Check for failures
        if final_state.get("status") == "failed":
            error_messages = [e.get("error", "Unknown") for e in final_state.get("errors", [])]
            raise RuntimeError(f"Experiment design failed: {'; '.join(error_messages)}")

        return output

    def _create_initial_state(
        self, input_data: ExperimentDesignerInput
    ) -> ExperimentDesignState:
        """Create initial state from input.

        Args:
            input_data: Validated input

        Returns:
            Initial ExperimentDesignState
        """
        state: ExperimentDesignState = {
            # Input fields
            "business_question": input_data.business_question,
            "constraints": input_data.constraints,
            "available_data": input_data.available_data,
            "preregistration_formality": input_data.preregistration_formality,
            "max_redesign_iterations": input_data.max_redesign_iterations,
            "enable_validity_audit": input_data.enable_validity_audit,
            # Error handling
            "errors": [],
            "warnings": [],
            "status": "pending",
        }

        return state

    def _create_output(self, state: ExperimentDesignState) -> ExperimentDesignerOutput:
        """Create output from final state.

        Args:
            state: Final state after graph execution

        Returns:
            ExperimentDesignerOutput
        """
        # Parse treatments
        treatments = []
        for t in state.get("treatments", []):
            treatments.append(
                TreatmentOutput(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    implementation_details=t.get("implementation_details", ""),
                    target_population=t.get("target_population", ""),
                    dosage_or_intensity=t.get("dosage_or_intensity"),
                    duration=t.get("duration"),
                )
            )

        # Parse outcomes
        outcomes = []
        for o in state.get("outcomes", []):
            outcomes.append(
                OutcomeOutput(
                    name=o.get("name", ""),
                    metric_type=o.get("metric_type", "continuous"),
                    measurement_method=o.get("measurement_method", ""),
                    measurement_frequency=o.get("measurement_frequency", ""),
                    is_primary=o.get("is_primary", False),
                    baseline_value=o.get("baseline_value"),
                    expected_effect_size=o.get("expected_effect_size"),
                )
            )

        # Parse validity threats
        threats = []
        for t in state.get("validity_threats", []):
            threats.append(
                ValidityThreatOutput(
                    threat_type=t.get("threat_type", "internal"),
                    threat_name=t.get("threat_name", ""),
                    description=t.get("description", ""),
                    severity=t.get("severity", "medium"),
                    mitigation_possible=t.get("mitigation_possible", True),
                    mitigation_strategy=t.get("mitigation_strategy"),
                )
            )

        # Parse power analysis
        power_analysis = None
        pa = state.get("power_analysis")
        if pa:
            power_analysis = PowerAnalysisOutput(
                required_sample_size=pa.get("required_sample_size", 0),
                required_sample_size_per_arm=pa.get("required_sample_size_per_arm", 0),
                achieved_power=pa.get("achieved_power", 0.0),
                minimum_detectable_effect=pa.get("minimum_detectable_effect", 0.0),
                alpha=pa.get("alpha", 0.05),
                effect_size_type=pa.get("effect_size_type", "cohens_d"),
                assumptions=pa.get("assumptions", []),
            )

        # Get total latency
        node_latencies = state.get("node_latencies_ms", {})
        total_latency = node_latencies.get("total", sum(node_latencies.values()))

        # Get pre-registration document from template
        template = state.get("experiment_template", {})
        prereg = template.get("pre_registration_document", "") if template else ""

        return ExperimentDesignerOutput(
            # Design outputs
            design_type=state.get("design_type", "RCT"),
            design_rationale=state.get("design_rationale", ""),
            treatments=treatments,
            outcomes=outcomes,
            randomization_unit=state.get("randomization_unit", "individual"),
            randomization_method=state.get("randomization_method", "simple"),
            stratification_variables=state.get("stratification_variables", []),
            blocking_variables=state.get("blocking_variables", []),
            # Power analysis outputs
            power_analysis=power_analysis,
            sample_size_justification=state.get("sample_size_justification", ""),
            duration_estimate_days=state.get("duration_estimate_days", 0),
            # Validity audit outputs
            validity_threats=threats,
            overall_validity_score=state.get("overall_validity_score", 0.0),
            validity_confidence=state.get("validity_confidence", "low"),
            # Generated templates
            causal_graph_dot=state.get("causal_graph_dot", ""),
            analysis_code=state.get("analysis_code", ""),
            preregistration_document=prereg,
            # Execution metadata
            total_latency_ms=total_latency,
            redesign_iterations=state.get("current_iteration", 0),
            warnings=state.get("warnings", []),
        )
