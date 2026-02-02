"""
Tests for Resource Optimizer DSPy Integration.

Tests the Recipient role implementation including:
- Optimized prompt templates
- Prompt consumer functionality
- DSPy signature availability
- Singleton pattern for integration
- Prompt formatting and updates
"""

import pytest

# Mark all tests in this module as dspy_integration to group them
pytestmark = pytest.mark.xdist_group(name="dspy_integration")


class TestResourceOptimizationPrompts:
    """Test ResourceOptimizationPrompts dataclass."""

    def test_default_initialization(self):
        """Test prompts initialize with default templates."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizationPrompts,
        )

        prompts = ResourceOptimizationPrompts()

        assert "{resource_type}" in prompts.summary_template
        assert "{entity_id}" in prompts.recommendation_template
        assert "{scenario_names}" in prompts.scenario_comparison_template
        assert "{constraint_type}" in prompts.constraint_warning_template
        assert prompts.version == "1.0"
        assert prompts.last_optimized == ""
        assert prompts.optimization_score == 0.0

    def test_to_dict_structure(self):
        """Test to_dict produces correct structure."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizationPrompts,
        )

        prompts = ResourceOptimizationPrompts(
            version="1.5",
            last_optimized="2025-12-30T10:00:00Z",
            optimization_score=0.85,
        )

        result = prompts.to_dict()

        assert "summary_template" in result
        assert "recommendation_template" in result
        assert "scenario_comparison_template" in result
        assert "constraint_warning_template" in result
        assert result["version"] == "1.5"
        assert result["last_optimized"] == "2025-12-30T10:00:00Z"
        assert result["optimization_score"] == 0.85


class TestResourceOptimizerDSPyIntegration:
    """Test ResourceOptimizerDSPyIntegration class (Recipient pattern)."""

    def test_initialization(self):
        """Test integration initializes correctly."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()

        assert integration.dspy_type == "recipient"
        assert integration.prompts is not None
        assert integration._prompt_versions == {}

    def test_prompts_property(self):
        """Test prompts property returns ResourceOptimizationPrompts."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizationPrompts,
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()

        assert isinstance(integration.prompts, ResourceOptimizationPrompts)

    def test_update_optimized_prompts(self):
        """Test updating prompts with optimized versions."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()

        new_prompts = {
            "summary_template": "New summary: {resource_type} optimized.",
            "recommendation_template": "New recommendation for {entity_id}.",
        }

        integration.update_optimized_prompts(
            prompts=new_prompts,
            optimization_score=0.92,
        )

        assert "New summary" in integration.prompts.summary_template
        assert "New recommendation" in integration.prompts.recommendation_template
        assert integration.prompts.optimization_score == 0.92
        assert integration.prompts.last_optimized != ""
        assert integration.prompts.version == "1.1"

    def test_update_prompts_multiple_times(self):
        """Test version updates and score updates with multiple calls."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()
        # Start fresh - check initial version
        initial_version = integration.prompts.version
        assert initial_version == "1.0"

        # Valid template with all required placeholders
        template = (
            "Version test: {resource_type} {objective} {solver_type} "
            "{objective_value} {projected_roi} {entity_count} "
            "{increase_count} {decrease_count}"
        )

        # First update - version changes from 1.0 to 1.1
        integration.update_optimized_prompts({"summary_template": template}, 0.8)
        first_version = integration.prompts.version
        assert first_version > initial_version  # "1.1" > "1.0"
        assert integration.prompts.optimization_score == 0.8
        first_timestamp = integration.prompts.last_optimized

        # Second update - optimization_score and timestamp update
        integration.update_optimized_prompts(
            {"summary_template": template.replace("Version test", "Version 2")}, 0.85
        )
        assert integration.prompts.optimization_score == 0.85
        second_timestamp = integration.prompts.last_optimized
        assert second_timestamp >= first_timestamp

        # Third update - verify score keeps updating
        integration.update_optimized_prompts(
            {"summary_template": template.replace("Version test", "Version 3")}, 0.9
        )
        assert integration.prompts.optimization_score == 0.9
        assert "Version 3" in integration.prompts.summary_template

    def test_get_summary_prompt(self):
        """Test summary prompt formatting."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()

        prompt = integration.get_summary_prompt(
            resource_type="budget",
            objective="maximize_roi",
            solver_type="linear",
            objective_value=450000.0,
            projected_roi=2.25,
            entity_count=10,
            increase_count=6,
            decrease_count=4,
        )

        assert "budget" in prompt
        assert "maximize_roi" in prompt
        assert "linear" in prompt
        assert "450000" in prompt
        assert "2.25" in prompt
        assert "10" in prompt

    def test_get_recommendation_prompt(self):
        """Test recommendation prompt formatting."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()

        prompt = integration.get_recommendation_prompt(
            entity_id="territory_northeast",
            entity_type="territory",
            current=50000.0,
            optimized=65000.0,
            change_pct=30.0,
            expected_impact=195000.0,
        )

        assert "territory_northeast" in prompt
        assert "territory" in prompt
        assert "50000" in prompt
        assert "65000" in prompt
        assert "30" in prompt

    def test_get_scenario_comparison_prompt(self):
        """Test scenario comparison prompt formatting."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()

        prompt = integration.get_scenario_comparison_prompt(
            scenario_names="baseline, aggressive, conservative",
            best_scenario="aggressive",
            best_roi=2.5,
            violations="budget exceeded in conservative",
        )

        assert "baseline, aggressive, conservative" in prompt
        assert "aggressive" in prompt
        assert "2.5" in prompt

    def test_get_constraint_warning_prompt(self):
        """Test constraint warning prompt formatting."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()

        prompt = integration.get_constraint_warning_prompt(
            constraint_type="budget",
            description="Budget constraint exceeded",
            value=110000.0,
            scope="territory_northeast",
            impact="Allocation reduced by 10%",
        )

        assert "budget" in prompt
        assert "Budget constraint exceeded" in prompt
        assert "110000" in prompt
        assert "territory_northeast" in prompt

    def test_get_prompt_metadata(self):
        """Test getting prompt metadata."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()

        metadata = integration.get_prompt_metadata()

        assert metadata["agent"] == "resource_optimizer"
        assert metadata["dspy_type"] == "recipient"
        assert metadata["prompt_count"] == 4
        assert "prompts" in metadata
        assert "dspy_available" in metadata


class TestSingletonAccess:
    """Test singleton pattern for DSPy integration."""

    def test_get_integration_creates_singleton(self):
        """Test that getter creates singleton."""
        from src.agents.resource_optimizer.dspy_integration import (
            get_resource_optimizer_dspy_integration,
            reset_dspy_integration,
        )

        reset_dspy_integration()

        integration1 = get_resource_optimizer_dspy_integration()
        integration2 = get_resource_optimizer_dspy_integration()

        assert integration1 is integration2

    def test_reset_clears_singleton(self):
        """Test that reset clears singleton."""
        from src.agents.resource_optimizer.dspy_integration import (
            get_resource_optimizer_dspy_integration,
            reset_dspy_integration,
        )

        integration1 = get_resource_optimizer_dspy_integration()
        reset_dspy_integration()
        integration2 = get_resource_optimizer_dspy_integration()

        assert integration1 is not integration2


class TestDSPySignatures:
    """Test DSPy signature availability."""

    def test_dspy_available_flag(self):
        """Test DSPY_AVAILABLE flag."""
        from src.agents.resource_optimizer.dspy_integration import DSPY_AVAILABLE

        assert isinstance(DSPY_AVAILABLE, bool)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_optimization_summary_signature(self):
        """Test OptimizationSummarySignature is valid DSPy signature."""
        from src.agents.resource_optimizer.dspy_integration import (
            DSPY_AVAILABLE,
            OptimizationSummarySignature,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(OptimizationSummarySignature, dspy.Signature)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_allocation_recommendation_signature(self):
        """Test AllocationRecommendationSignature is valid DSPy signature."""
        from src.agents.resource_optimizer.dspy_integration import (
            DSPY_AVAILABLE,
            AllocationRecommendationSignature,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(AllocationRecommendationSignature, dspy.Signature)

    @pytest.mark.skipif(
        "not pytest.importorskip('dspy')",
        reason="DSPy not available",
    )
    def test_scenario_narrative_signature(self):
        """Test ScenarioNarrativeSignature is valid DSPy signature."""
        from src.agents.resource_optimizer.dspy_integration import (
            DSPY_AVAILABLE,
            ScenarioNarrativeSignature,
        )

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        import dspy

        assert issubclass(ScenarioNarrativeSignature, dspy.Signature)


class TestPromptOptimizationWorkflow:
    """Test the prompt optimization workflow for Recipient agents."""

    def test_initial_prompts_work(self):
        """Test that initial prompts can format correctly."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()

        # All prompt methods should work with default templates
        summary = integration.get_summary_prompt(
            resource_type="rep_time",
            objective="maximize_outcome",
            solver_type="milp",
            objective_value=1000000.0,
            projected_roi=1.8,
            entity_count=50,
            increase_count=30,
            decrease_count=20,
        )

        recommendation = integration.get_recommendation_prompt(
            entity_id="hcp_001",
            entity_type="hcp",
            current=10.0,
            optimized=15.0,
            change_pct=50.0,
            expected_impact=25000.0,
        )

        scenario = integration.get_scenario_comparison_prompt(
            scenario_names="A, B, C",
            best_scenario="B",
            best_roi=2.0,
            violations="None",
        )

        warning = integration.get_constraint_warning_prompt(
            constraint_type="capacity",
            description="Rep capacity exceeded",
            value=45.0,
            scope="region_west",
            impact="Reallocation needed",
        )

        # All should be non-empty strings
        assert len(summary) > 0
        assert len(recommendation) > 0
        assert len(scenario) > 0
        assert len(warning) > 0

    def test_optimized_prompts_can_update(self):
        """Test that prompts can be updated and still format correctly."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()

        # Update with new templates (must include all placeholders)
        integration.update_optimized_prompts(
            prompts={
                "summary_template": (
                    "OPTIMIZED: Resource optimization for {resource_type}. "
                    "Objective: {objective}. Solver: {solver_type}. "
                    "Value: {objective_value}. ROI: {projected_roi}. "
                    "Entities: {entity_count}. Up: {increase_count}. Down: {decrease_count}."
                ),
            },
            optimization_score=0.88,
        )

        # Should still format correctly
        summary = integration.get_summary_prompt(
            resource_type="budget",
            objective="balance",
            solver_type="nonlinear",
            objective_value=500000.0,
            projected_roi=2.1,
            entity_count=25,
            increase_count=15,
            decrease_count=10,
        )

        assert "OPTIMIZED:" in summary
        assert "budget" in summary
        assert "balance" in summary

    def test_partial_prompt_update(self):
        """Test that partial updates only affect specified prompts."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()
        original_recommendation = integration.prompts.recommendation_template

        # Only update summary template
        integration.update_optimized_prompts(
            prompts={
                "summary_template": "New summary: {resource_type} {objective} {solver_type} {objective_value} {projected_roi} {entity_count} {increase_count} {decrease_count}",
            },
            optimization_score=0.75,
        )

        # Recommendation template should be unchanged
        assert integration.prompts.recommendation_template == original_recommendation
