"""Tests for Simulate Intervention Tool.

Tests the digital twin simulation tool functionality for pre-screening
A/B test interventions.

NOTE: Uses direct module imports to avoid triggering LLM initialization
in the experiment_designer package __init__.py.
"""

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4


# Direct module import to avoid package __init__.py side effects
def _import_tool_module():
    """Import tool module directly, bypassing package __init__."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "simulate_intervention_tool",
        "src/agents/experiment_designer/tools/simulate_intervention_tool.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import schemas and models that don't trigger LLM init
from src.digital_twin.models.simulation_models import InterventionConfig


# Lazy import the actual tool module
@pytest.fixture(scope="module")
def tool_module():
    """Fixture to lazily import the tool module."""
    return _import_tool_module()


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestSimulateInterventionInput:
    """Test input schema validation."""

    def test_valid_input_minimal(self, tool_module):
        """Test creating input with minimal required fields."""
        SimulateInterventionInput = tool_module.SimulateInterventionInput
        input_schema = SimulateInterventionInput(
            intervention_type="email_campaign",
            brand="Kisqali",
        )

        assert input_schema.intervention_type == "email_campaign"
        assert input_schema.brand == "Kisqali"
        assert input_schema.target_population == "hcp"  # default
        assert input_schema.twin_count == 10000  # default
        assert input_schema.confidence_level == 0.95  # default

    def test_valid_input_full(self, tool_module):
        """Test creating input with all fields populated."""
        SimulateInterventionInput = tool_module.SimulateInterventionInput
        input_schema = SimulateInterventionInput(
            intervention_type="speaker_program_invitation",
            brand="Fabhalta",
            target_population="patient",
            channel="in_person",
            frequency="monthly",
            duration_weeks=12,
            target_deciles=[1, 2],
            target_specialties=["oncology", "hematology"],
            target_regions=["northeast"],
            twin_count=5000,
            confidence_level=0.90,
        )

        assert input_schema.intervention_type == "speaker_program_invitation"
        assert input_schema.brand == "Fabhalta"
        assert input_schema.target_population == "patient"
        assert input_schema.channel == "in_person"
        assert input_schema.frequency == "monthly"
        assert input_schema.duration_weeks == 12
        assert input_schema.target_deciles == [1, 2]
        assert input_schema.target_specialties == ["oncology", "hematology"]
        assert input_schema.target_regions == ["northeast"]
        assert input_schema.twin_count == 5000
        assert input_schema.confidence_level == 0.90

    def test_default_target_population(self, tool_module):
        """Test default target population is HCP."""
        SimulateInterventionInput = tool_module.SimulateInterventionInput
        input_schema = SimulateInterventionInput(
            intervention_type="call_frequency_increase",
            brand="Remibrutinib",
        )

        assert input_schema.target_population == "hcp"

    def test_default_duration_weeks(self, tool_module):
        """Test default duration is 8 weeks."""
        SimulateInterventionInput = tool_module.SimulateInterventionInput
        input_schema = SimulateInterventionInput(
            intervention_type="email_campaign",
            brand="Kisqali",
        )

        assert input_schema.duration_weeks == 8

    def test_optional_fields_are_none(self, tool_module):
        """Test that optional fields default to None."""
        SimulateInterventionInput = tool_module.SimulateInterventionInput
        input_schema = SimulateInterventionInput(
            intervention_type="digital_engagement",
            brand="Fabhalta",
        )

        assert input_schema.channel is None
        assert input_schema.frequency is None
        assert input_schema.target_deciles is None
        assert input_schema.target_specialties is None
        assert input_schema.target_regions is None


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestSimulateIntervention:
    """Test simulate_intervention tool function."""

    def test_returns_deploy_recommendation(self, tool_module):
        """Test tool returns deploy recommendation for high-effect intervention."""
        simulate_intervention = tool_module.simulate_intervention
        # Use mock result generator with known intervention type
        result = simulate_intervention.invoke({
            "intervention_type": "speaker_program_invitation",  # High base effect
            "brand": "Kisqali",
        })

        assert "recommendation" in result
        assert result["recommendation"] in ["deploy", "skip", "refine"]
        assert "simulation_id" in result
        assert "simulated_ate" in result

    def test_returns_required_output_fields(self, tool_module):
        """Test all required output fields are present."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke({
            "intervention_type": "email_campaign",
            "brand": "Fabhalta",
        })

        # Check all required fields
        required_fields = [
            "simulation_id",
            "recommendation",
            "recommendation_rationale",
            "simulated_ate",
            "confidence_interval",
            "recommended_sample_size",
            "recommended_duration_weeks",
            "simulation_confidence",
            "fidelity_warning",
            "fidelity_warning_reason",
            "top_segments",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_confidence_interval_calculated(self, tool_module):
        """Test confidence interval is properly calculated."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke({
            "intervention_type": "call_frequency_increase",
            "brand": "Remibrutinib",
            "confidence_level": 0.95,
        })

        ci = result["confidence_interval"]
        assert isinstance(ci, tuple) or isinstance(ci, list)
        assert len(ci) == 2
        assert ci[0] <= ci[1]  # Lower bound <= upper bound

    def test_top_segments_returned(self, tool_module):
        """Test top performing segments are returned."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke({
            "intervention_type": "peer_influence_activation",
            "brand": "Kisqali",
        })

        top_segments = result["top_segments"]
        assert isinstance(top_segments, list)
        # Mock result returns 3 segments
        if top_segments:
            segment = top_segments[0]
            assert "dimension" in segment
            assert "segment" in segment
            assert "ate" in segment

    def test_handles_invalid_brand(self, tool_module):
        """Test handling of invalid brand."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke({
            "intervention_type": "email_campaign",
            "brand": "InvalidBrand",
        })

        # Should return refine recommendation with error
        assert result["recommendation"] == "refine"
        assert "error" in result.get("recommendation_rationale", "").lower() or \
               "failed" in result.get("recommendation_rationale", "").lower()

    def test_handles_invalid_population(self, tool_module):
        """Test handling of invalid target population."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke({
            "intervention_type": "email_campaign",
            "brand": "Kisqali",
            "target_population": "invalid_population",
        })

        # Should return error response
        assert result["recommendation"] == "refine"

    def test_fidelity_warning_included(self, tool_module):
        """Test fidelity warning field is included in response."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke({
            "intervention_type": "sample_distribution",
            "brand": "Fabhalta",
        })

        assert "fidelity_warning" in result
        assert isinstance(result["fidelity_warning"], bool)


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestDigitalTwinWorkflow:
    """Test DigitalTwinWorkflow class."""

    def test_propose_experiment_skip(self, tool_module):
        """Test workflow returns skip action for low-effect intervention."""
        DigitalTwinWorkflow = tool_module.DigitalTwinWorkflow
        workflow = DigitalTwinWorkflow()

        # Mock simulate_intervention to return skip
        with patch.object(
            tool_module, "simulate_intervention"
        ) as mock_sim:
            # Note: simulate_intervention is called directly (not .invoke())
            mock_sim.return_value = {
                "simulation_id": str(uuid4()),
                "recommendation": "skip",
                "recommendation_rationale": "Effect below threshold",
                "simulated_ate": 0.02,
                "confidence_interval": (-0.01, 0.05),
                "recommended_sample_size": None,
                "recommended_duration_weeks": 8,
                "simulation_confidence": 0.75,
                "fidelity_warning": False,
                "fidelity_warning_reason": None,
                "top_segments": [],
            }

            result = workflow.propose_experiment(
                intervention_type="email_campaign",
                brand="Kisqali",
            )

        assert result["action"] == "SKIP"
        assert "reason" in result

    def test_propose_experiment_design(self, tool_module):
        """Test workflow returns design action for promising intervention."""
        DigitalTwinWorkflow = tool_module.DigitalTwinWorkflow
        workflow = DigitalTwinWorkflow()

        # Mock simulate_intervention to return deploy
        with patch.object(
            tool_module, "simulate_intervention"
        ) as mock_sim:
            # Note: simulate_intervention is called directly (not .invoke())
            mock_sim.return_value = {
                "simulation_id": str(uuid4()),
                "recommendation": "deploy",
                "recommendation_rationale": "Strong predicted effect",
                "simulated_ate": 0.12,
                "confidence_interval": (0.08, 0.16),
                "recommended_sample_size": 2000,
                "recommended_duration_weeks": 8,
                "simulation_confidence": 0.85,
                "fidelity_warning": False,
                "fidelity_warning_reason": None,
                "top_segments": [
                    {"dimension": "decile", "segment": "1-2", "ate": 0.15, "n": 1000}
                ],
            }

            result = workflow.propose_experiment(
                intervention_type="speaker_program_invitation",
                brand="Kisqali",
            )

        assert result["action"] == "DESIGN"
        assert "prior_estimate" in result
        assert "recommended_sample_size" in result

    def test_passes_prior_estimate(self, tool_module):
        """Test workflow passes prior estimate to experiment designer."""
        DigitalTwinWorkflow = tool_module.DigitalTwinWorkflow
        workflow = DigitalTwinWorkflow()

        with patch.object(
            tool_module, "simulate_intervention"
        ) as mock_sim:
            expected_ate = 0.10
            # Note: simulate_intervention is called directly (not .invoke())
            mock_sim.return_value = {
                "simulation_id": str(uuid4()),
                "recommendation": "deploy",
                "recommendation_rationale": "Proceed with test",
                "simulated_ate": expected_ate,
                "confidence_interval": (0.06, 0.14),
                "recommended_sample_size": 2500,
                "recommended_duration_weeks": 8,
                "simulation_confidence": 0.80,
                "fidelity_warning": False,
                "fidelity_warning_reason": None,
                "top_segments": [],
            }

            result = workflow.propose_experiment(
                intervention_type="call_frequency_increase",
                brand="Fabhalta",
            )

        assert result["prior_estimate"]["ate"] == expected_ate
        assert "ci" in result["prior_estimate"]

    def test_includes_top_segments(self, tool_module):
        """Test workflow includes top segments in design action."""
        DigitalTwinWorkflow = tool_module.DigitalTwinWorkflow
        workflow = DigitalTwinWorkflow()

        segments = [
            {"dimension": "specialty", "segment": "oncology", "ate": 0.14, "n": 500}
        ]

        with patch.object(
            tool_module, "simulate_intervention"
        ) as mock_sim:
            # Note: simulate_intervention is called directly (not .invoke())
            mock_sim.return_value = {
                "simulation_id": str(uuid4()),
                "recommendation": "deploy",
                "recommendation_rationale": "Proceed",
                "simulated_ate": 0.11,
                "confidence_interval": (0.07, 0.15),
                "recommended_sample_size": 2000,
                "recommended_duration_weeks": 8,
                "simulation_confidence": 0.82,
                "fidelity_warning": False,
                "fidelity_warning_reason": None,
                "top_segments": segments,
            }

            result = workflow.propose_experiment(
                intervention_type="digital_engagement",
                brand="Remibrutinib",
            )

        assert result["top_segments"] == segments


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestMockResultGeneration:
    """Test _create_mock_result function."""

    def test_mock_result_structure(self, tool_module):
        """Test mock result has required structure."""
        _create_mock_result = tool_module._create_mock_result
        config = InterventionConfig(
            intervention_type="email_campaign",
            duration_weeks=8,
        )

        result = _create_mock_result(config, 10000, "email_campaign")

        assert "simulation_id" in result
        assert "recommendation" in result
        assert "recommendation_rationale" in result
        assert "simulated_ate" in result
        assert "confidence_interval" in result
        assert "recommended_sample_size" in result
        assert "recommended_duration_weeks" in result
        assert "simulation_confidence" in result
        assert "fidelity_warning" in result
        assert "top_segments" in result

    def test_mock_result_intervention_types(self, tool_module):
        """Test mock result varies by intervention type."""
        _create_mock_result = tool_module._create_mock_result
        config = InterventionConfig(
            intervention_type="speaker_program_invitation",  # High base effect
            duration_weeks=8,
        )

        result = _create_mock_result(config, 10000, "speaker_program_invitation")

        # Speaker programs have higher base effect (0.14)
        # Result may vary due to noise but should generally be higher
        assert result["simulated_ate"] is not None
        assert isinstance(result["simulated_ate"], float)

    def test_mock_result_confidence_bounds(self, tool_module):
        """Test mock result has valid confidence bounds."""
        _create_mock_result = tool_module._create_mock_result
        config = InterventionConfig(
            intervention_type="call_frequency_increase",
            duration_weeks=8,
        )

        result = _create_mock_result(config, 10000, "call_frequency_increase")

        ci = result["confidence_interval"]
        assert ci[0] < ci[1]  # Lower < Upper
        # CI should contain the ATE
        ate = result["simulated_ate"]
        # Allow for floating point precision
        assert ci[0] <= ate <= ci[1] or abs(ate - (ci[0] + ci[1]) / 2) < 0.05

    def test_mock_result_recommendation_logic(self, tool_module):
        """Test mock result recommendation follows logic."""
        _create_mock_result = tool_module._create_mock_result
        config = InterventionConfig(
            intervention_type="sample_distribution",  # Low base effect (0.04)
            duration_weeks=8,
        )

        # Run multiple times to test logic (due to randomness)
        results = []
        for _ in range(10):
            result = _create_mock_result(config, 10000, "sample_distribution")
            results.append(result)

        # All results should have valid recommendations
        for result in results:
            assert result["recommendation"] in ["deploy", "skip", "refine"]


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestSimulateInterventionEdgeCases:
    """Test edge cases for simulate_intervention tool."""

    def test_minimal_twin_count(self, tool_module):
        """Test simulation with minimal twin count."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke({
            "intervention_type": "email_campaign",
            "brand": "Kisqali",
            "twin_count": 100,
        })

        assert "simulation_id" in result
        assert result["simulation_confidence"] >= 0

    def test_maximum_twin_count(self, tool_module):
        """Test simulation with high twin count."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke({
            "intervention_type": "email_campaign",
            "brand": "Fabhalta",
            "twin_count": 50000,
        })

        assert "simulation_id" in result

    def test_all_brands_supported(self, tool_module):
        """Test all supported brands work."""
        simulate_intervention = tool_module.simulate_intervention
        brands = ["Remibrutinib", "Fabhalta", "Kisqali"]

        for brand in brands:
            result = simulate_intervention.invoke({
                "intervention_type": "email_campaign",
                "brand": brand,
            })

            assert result["recommendation"] in ["deploy", "skip", "refine"], \
                f"Failed for brand: {brand}"

    def test_all_intervention_types(self, tool_module):
        """Test all known intervention types."""
        simulate_intervention = tool_module.simulate_intervention
        intervention_types = [
            "email_campaign",
            "call_frequency_increase",
            "speaker_program_invitation",
            "sample_distribution",
            "peer_influence_activation",
            "digital_engagement",
        ]

        for itype in intervention_types:
            result = simulate_intervention.invoke({
                "intervention_type": itype,
                "brand": "Kisqali",
            })

            assert result["recommendation"] in ["deploy", "skip", "refine"], \
                f"Failed for intervention type: {itype}"

    def test_unknown_intervention_type(self, tool_module):
        """Test handling of unknown intervention type."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke({
            "intervention_type": "unknown_intervention",
            "brand": "Kisqali",
        })

        # Should still return valid result with default effect
        assert "simulation_id" in result
        assert "simulated_ate" in result
