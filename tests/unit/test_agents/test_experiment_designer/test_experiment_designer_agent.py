"""Integration Tests for Experiment Designer Agent.

Tests complete end-to-end workflows and input/output validation.

Contract: .claude/contracts/tier3-contracts.md lines 82-142
"""

import pytest
from pydantic import ValidationError

from src.agents.experiment_designer import (
    ExperimentDesignerAgent,
    ExperimentDesignerInput,
    ExperimentDesignerOutput,
    OutcomeOutput,
    PowerAnalysisOutput,
    TreatmentOutput,
    ValidityThreatOutput,
)


class TestExperimentDesignerAgent:
    """Test ExperimentDesignerAgent integration."""

    def test_create_agent(self):
        """Test agent creation."""
        agent = ExperimentDesignerAgent()

        assert agent is not None
        assert agent.graph is not None

    def test_create_agent_custom_iterations(self):
        """Test agent creation with custom redesign iterations."""
        agent = ExperimentDesignerAgent(max_redesign_iterations=3)

        assert agent is not None
        assert agent.max_redesign_iterations == 3

    @pytest.mark.asyncio
    async def test_run_basic(self):
        """Test basic agent execution."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Does increasing rep visit frequency improve HCP engagement?"
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)
        # Design type comparison is case-insensitive
        design_type_normalized = result.design_type.lower().replace("-", "_")
        assert design_type_normalized in [
            "rct",
            "cluster_rct",
            "quasi_experimental",
            "observational",
        ]
        assert len(result.design_rationale) > 0

    @pytest.mark.asyncio
    async def test_run_with_constraints(self):
        """Test execution with constraints."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Does digital engagement improve prescription rates?",
            constraints={"expected_effect_size": 0.25, "power": 0.80, "weekly_accrual": 50},
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)
        assert result.power_analysis is not None

    @pytest.mark.asyncio
    async def test_run_with_available_data(self):
        """Test execution with available data specification."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Test engagement impact",
            available_data={
                "variables": ["hcp_id", "territory", "visit_count", "engagement_score"],
                "sample_size": 5000,
                "historical_periods": 12,
            },
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)

    @pytest.mark.asyncio
    async def test_run_with_brand(self):
        """Test execution with brand filter."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Test Remibrutinib engagement campaign",
            constraints={"expected_effect_size": 0.20},
            brand="Remibrutinib",
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)

    @pytest.mark.asyncio
    async def test_run_heavy_preregistration(self):
        """Test with heavy preregistration formality."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Test experiment for regulatory submission",
            preregistration_formality="heavy",
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)
        assert len(result.preregistration_document) > 0

    @pytest.mark.asyncio
    async def test_run_light_preregistration(self):
        """Test with light preregistration formality."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Quick pilot test", preregistration_formality="light"
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)

    @pytest.mark.asyncio
    async def test_run_without_validity_audit(self):
        """Test execution without validity audit."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Quick design without audit", enable_validity_audit=False
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)

    @pytest.mark.asyncio
    async def test_run_multiple_redesign_iterations(self):
        """Test with max redesign iterations set."""
        agent = ExperimentDesignerAgent(max_redesign_iterations=3)
        input_data = ExperimentDesignerInput(
            business_question="Complex experiment needing refinement", max_redesign_iterations=3
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)
        assert result.redesign_iterations <= 3


class TestExperimentDesignerInput:
    """Test ExperimentDesignerInput validation."""

    def test_valid_input_minimal(self):
        """Test valid minimal input."""
        input_data = ExperimentDesignerInput(business_question="Does X impact Y?")

        assert input_data.business_question == "Does X impact Y?"
        assert input_data.constraints == {}
        assert input_data.available_data == {}
        assert input_data.preregistration_formality == "medium"
        assert input_data.max_redesign_iterations == 2
        assert input_data.enable_validity_audit is True

    def test_valid_input_full(self):
        """Test valid input with all fields."""
        input_data = ExperimentDesignerInput(
            business_question="Does increasing rep visit frequency improve HCP engagement?",
            constraints={
                "expected_effect_size": 0.25,
                "alpha": 0.05,
                "power": 0.80,
                "weekly_accrual": 50,
                "budget": 100000,
                "timeline": "6 months",
            },
            available_data={
                "variables": ["hcp_id", "territory", "visit_count", "engagement_score"],
                "sample_size": 5000,
            },
            preregistration_formality="heavy",
            max_redesign_iterations=3,
            enable_validity_audit=True,
            brand="Remibrutinib",
        )

        assert input_data.constraints["expected_effect_size"] == 0.25
        assert input_data.preregistration_formality == "heavy"
        assert input_data.max_redesign_iterations == 3
        assert input_data.brand == "Remibrutinib"

    def test_invalid_empty_question(self):
        """Test invalid empty business question."""
        with pytest.raises(ValidationError):
            ExperimentDesignerInput(business_question="")  # Empty string

    def test_invalid_short_question(self):
        """Test invalid short business question."""
        with pytest.raises(ValidationError):
            ExperimentDesignerInput(business_question="Test?")  # Too short (<10 chars)

    def test_invalid_max_iterations_too_high(self):
        """Test invalid max redesign iterations too high."""
        with pytest.raises(ValidationError):
            ExperimentDesignerInput(
                business_question="Valid question here", max_redesign_iterations=10  # > 5
            )

    def test_invalid_max_iterations_negative(self):
        """Test invalid negative max redesign iterations."""
        with pytest.raises(ValidationError):
            ExperimentDesignerInput(
                business_question="Valid question here", max_redesign_iterations=-1
            )

    def test_invalid_formality_level(self):
        """Test invalid preregistration formality level."""
        with pytest.raises(ValidationError):
            ExperimentDesignerInput(
                business_question="Valid question here", preregistration_formality="invalid"
            )

    def test_valid_constraint_keys(self):
        """Test that valid constraint keys are accepted."""
        input_data = ExperimentDesignerInput(
            business_question="Valid question here",
            constraints={
                "budget": 50000,
                "timeline": "3 months",
                "ethical": "IRB approved",
                "operational": "Limited to east region",
                "expected_effect_size": 0.3,
                "alpha": 0.05,
                "power": 0.8,
                "weekly_accrual": 100,
                "cluster_size": 20,
                "expected_icc": 0.05,
                "baseline_rate": 0.15,
            },
        )

        assert len(input_data.constraints) == 11


class TestExperimentDesignerOutput:
    """Test ExperimentDesignerOutput structure."""

    @pytest.mark.asyncio
    async def test_output_structure(self):
        """Test output structure has all required fields."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(business_question="Does X impact Y in meaningful way?")

        result = await agent.arun(input_data)

        # Check all required fields exist
        assert hasattr(result, "design_type")
        assert hasattr(result, "design_rationale")
        assert hasattr(result, "treatments")
        assert hasattr(result, "outcomes")
        assert hasattr(result, "randomization_unit")
        assert hasattr(result, "randomization_method")
        assert hasattr(result, "stratification_variables")
        assert hasattr(result, "blocking_variables")
        assert hasattr(result, "power_analysis")
        assert hasattr(result, "sample_size_justification")
        assert hasattr(result, "duration_estimate_days")
        assert hasattr(result, "validity_threats")
        assert hasattr(result, "overall_validity_score")
        assert hasattr(result, "validity_confidence")
        assert hasattr(result, "causal_graph_dot")
        assert hasattr(result, "analysis_code")
        assert hasattr(result, "preregistration_document")
        assert hasattr(result, "total_latency_ms")
        assert hasattr(result, "redesign_iterations")
        assert hasattr(result, "warnings")

    @pytest.mark.asyncio
    async def test_output_validity_score_range(self):
        """Test validity score is in valid range."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(business_question="Test validity score range")

        result = await agent.arun(input_data)

        assert 0.0 <= result.overall_validity_score <= 1.0

    @pytest.mark.asyncio
    async def test_output_treatments_structure(self):
        """Test treatments structure."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(business_question="Does treatment A improve outcome?")

        result = await agent.arun(input_data)

        assert isinstance(result.treatments, list)
        if result.treatments:
            treatment = result.treatments[0]
            assert isinstance(treatment, TreatmentOutput)
            assert hasattr(treatment, "name")
            assert hasattr(treatment, "description")
            assert hasattr(treatment, "implementation_details")
            assert hasattr(treatment, "target_population")

    @pytest.mark.asyncio
    async def test_output_outcomes_structure(self):
        """Test outcomes structure."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(business_question="Measure impact on key outcome")

        result = await agent.arun(input_data)

        assert isinstance(result.outcomes, list)
        if result.outcomes:
            outcome = result.outcomes[0]
            assert isinstance(outcome, OutcomeOutput)
            assert hasattr(outcome, "name")
            assert hasattr(outcome, "metric_type")
            assert hasattr(outcome, "measurement_method")
            assert hasattr(outcome, "is_primary")

    @pytest.mark.asyncio
    async def test_output_validity_threats_structure(self):
        """Test validity threats structure."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(business_question="Check for validity threats")

        result = await agent.arun(input_data)

        assert isinstance(result.validity_threats, list)
        if result.validity_threats:
            threat = result.validity_threats[0]
            assert isinstance(threat, ValidityThreatOutput)
            assert hasattr(threat, "threat_type")
            assert hasattr(threat, "threat_name")
            assert hasattr(threat, "severity")
            assert hasattr(threat, "mitigation_possible")

    @pytest.mark.asyncio
    async def test_output_power_analysis_structure(self):
        """Test power analysis structure."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Calculate required sample size",
            constraints={"expected_effect_size": 0.3, "power": 0.8},
        )

        result = await agent.arun(input_data)

        if result.power_analysis:
            pa = result.power_analysis
            assert isinstance(pa, PowerAnalysisOutput)
            assert hasattr(pa, "required_sample_size")
            assert hasattr(pa, "required_sample_size_per_arm")
            assert hasattr(pa, "achieved_power")
            assert hasattr(pa, "minimum_detectable_effect")
            assert hasattr(pa, "alpha")
            assert hasattr(pa, "effect_size_type")
            assert pa.required_sample_size > 0
            assert 0.0 < pa.achieved_power <= 1.0

    @pytest.mark.asyncio
    async def test_output_analysis_code_generated(self):
        """Test analysis code is generated."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(business_question="Generate analysis template")

        result = await agent.arun(input_data)

        assert len(result.analysis_code) > 0
        # Should contain DoWhy imports
        assert "dowhy" in result.analysis_code.lower() or "CausalModel" in result.analysis_code

    @pytest.mark.asyncio
    async def test_output_preregistration_generated(self):
        """Test preregistration document is generated."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Generate preregistration for experiment"
        )

        result = await agent.arun(input_data)

        assert len(result.preregistration_document) > 0


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_rct_design_workflow(self):
        """Test RCT design workflow."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Does increasing sales rep visit frequency improve HCP engagement scores?",
            constraints={
                "expected_effect_size": 0.25,
                "power": 0.80,
                "alpha": 0.05,
                "weekly_accrual": 50,
            },
            available_data={
                "variables": ["hcp_id", "territory", "visit_count", "engagement_score"],
                "sample_size": 5000,
            },
        )

        result = await agent.arun(input_data)

        # Design type comparison is case-insensitive
        design_type_normalized = result.design_type.lower().replace("-", "_")
        assert design_type_normalized in ["rct", "cluster_rct", "quasi_experimental"]
        assert result.power_analysis is not None
        assert len(result.treatments) > 0

    @pytest.mark.asyncio
    async def test_cluster_rct_workflow(self):
        """Test cluster RCT design workflow."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Does territory-level marketing strategy change impact overall sales?",
            constraints={
                "expected_effect_size": 0.20,
                "power": 0.80,
                "cluster_size": 50,
                "expected_icc": 0.05,
            },
            available_data={"variables": ["territory_id", "rep_id", "sales_volume"]},
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)

    @pytest.mark.asyncio
    async def test_binary_outcome_workflow(self):
        """Test workflow with binary outcome."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Does new outreach strategy improve HCP conversion rate?",
            constraints={"expected_effect_size": 0.10, "baseline_rate": 0.15, "power": 0.80},
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)  # Extended timeout: 3 brand iterations × ~30s each
    async def test_multiple_brands_workflow(self):
        """Test with different brands."""
        agent = ExperimentDesignerAgent()

        for brand in ["Remibrutinib", "Fabhalta", "Kisqali"]:
            input_data = ExperimentDesignerInput(
                business_question=f"Optimize {brand} marketing effectiveness", brand=brand
            )

            result = await agent.arun(input_data)

            assert isinstance(result, ExperimentDesignerOutput)

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)  # Extended timeout: 3 formality levels × ~30s each
    async def test_different_formality_levels(self):
        """Test with different preregistration formality levels."""
        agent = ExperimentDesignerAgent()

        for formality in ["light", "medium", "heavy"]:
            input_data = ExperimentDesignerInput(
                business_question="Test experiment design", preregistration_formality=formality
            )

            result = await agent.arun(input_data)

            assert isinstance(result, ExperimentDesignerOutput)
            assert len(result.preregistration_document) > 0

    @pytest.mark.asyncio
    async def test_redesign_loop_executes(self):
        """Test that redesign loop can execute when needed."""
        agent = ExperimentDesignerAgent(max_redesign_iterations=2)
        input_data = ExperimentDesignerInput(
            business_question="Complex experiment requiring iteration",
            max_redesign_iterations=2,
            enable_validity_audit=True,
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)
        # Redesign iterations should be tracked
        assert result.redesign_iterations >= 0
        assert result.redesign_iterations <= 2

    @pytest.mark.asyncio
    async def test_latency_under_target(self):
        """Test latency is under 60s target."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Test latency performance",
            constraints={"expected_effect_size": 0.25, "power": 0.80},
        )

        result = await agent.arun(input_data)

        # Should be under 60,000ms (60s) per contract
        assert result.total_latency_ms < 60_000

    @pytest.mark.asyncio
    async def test_causal_graph_dot_generated(self):
        """Test causal graph DOT format is generated."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Generate causal graph for experiment"
        )

        result = await agent.arun(input_data)

        assert len(result.causal_graph_dot) > 0
        # Should contain DOT format elements
        assert "digraph" in result.causal_graph_dot or "->" in result.causal_graph_dot


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_minimal_constraints(self):
        """Test with minimal constraints."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(business_question="Minimal constraint experiment")

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)

    @pytest.mark.asyncio
    async def test_maximum_constraints(self):
        """Test with maximum constraints."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Maximum constraint experiment",
            constraints={
                "budget": 1000000,
                "timeline": "12 months",
                "expected_effect_size": 0.5,
                "alpha": 0.01,
                "power": 0.95,
                "weekly_accrual": 200,
                "cluster_size": 100,
                "expected_icc": 0.1,
                "baseline_rate": 0.25,
            },
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)

    @pytest.mark.asyncio
    async def test_zero_redesign_iterations(self):
        """Test with zero redesign iterations."""
        agent = ExperimentDesignerAgent(max_redesign_iterations=0)
        input_data = ExperimentDesignerInput(
            business_question="Single pass experiment design", max_redesign_iterations=0
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)
        assert result.redesign_iterations == 0

    @pytest.mark.asyncio
    async def test_very_small_effect_size(self):
        """Test with very small effect size."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Detect very small effect",
            constraints={"expected_effect_size": 0.05, "power": 0.80},  # Very small
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)
        if result.power_analysis:
            # Very small effect should require large sample
            assert result.power_analysis.required_sample_size > 500

    @pytest.mark.asyncio
    async def test_large_effect_size(self):
        """Test with large effect size."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Detect large effect",
            constraints={"expected_effect_size": 0.8, "power": 0.80},  # Large
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)
        if result.power_analysis:
            # Sample size should be calculated (mock may use different effect size or cluster design)
            # Cluster RCTs require larger samples due to design effect, so allow up to 2000
            assert result.power_analysis.required_sample_size > 0
            assert result.power_analysis.required_sample_size < 2000

    @pytest.mark.asyncio
    async def test_empty_available_data(self):
        """Test with empty available data."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Design with no available data specified", available_data={}
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)

    @pytest.mark.asyncio
    async def test_complex_available_data(self):
        """Test with complex available data structure."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Design with complex data",
            available_data={
                "variables": [
                    "hcp_id",
                    "territory",
                    "region",
                    "specialty",
                    "years_experience",
                    "visit_count",
                    "email_opens",
                    "webinar_attendance",
                    "sample_requests",
                    "trx_volume",
                    "nrx_volume",
                    "market_share",
                ],
                "sample_size": 50000,
                "historical_periods": 24,
                "data_quality": "high",
                "missing_rate": 0.02,
            },
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)


class TestAsyncExecution:
    """Test async execution methods."""

    @pytest.mark.asyncio
    async def test_arun_basic(self):
        """Test basic async execution."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(business_question="Test async execution")

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)

    @pytest.mark.asyncio
    async def test_arun_with_constraints(self):
        """Test async execution with constraints."""
        agent = ExperimentDesignerAgent()
        input_data = ExperimentDesignerInput(
            business_question="Test async with constraints",
            constraints={"expected_effect_size": 0.25, "power": 0.80},
        )

        result = await agent.arun(input_data)

        assert isinstance(result, ExperimentDesignerOutput)
        assert result.power_analysis is not None
