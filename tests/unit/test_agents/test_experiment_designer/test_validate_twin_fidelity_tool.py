"""Tests for Validate Twin Fidelity Tool.

Tests the digital twin fidelity validation tool for comparing
simulation predictions against actual experiment outcomes.

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
        "validate_twin_fidelity_tool",
        "src/agents/experiment_designer/tools/validate_twin_fidelity_tool.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import models that don't trigger LLM init
from src.digital_twin.models.simulation_models import FidelityGrade, FidelityRecord


# Lazy import the actual tool module
@pytest.fixture(scope="module")
def tool_module():
    """Fixture to lazily import the tool module."""
    return _import_tool_module()


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestValidateFidelityInput:
    """Test input schema validation."""

    def test_valid_input_minimal(self, tool_module):
        """Test creating input with minimal required fields."""
        ValidateFidelityInput = tool_module.ValidateFidelityInput
        input_schema = ValidateFidelityInput(
            simulation_id=str(uuid4()),
            actual_ate=0.085,
        )

        assert input_schema.actual_ate == 0.085
        assert input_schema.actual_ci_lower is None
        assert input_schema.actual_ci_upper is None
        assert input_schema.actual_sample_size is None

    def test_valid_input_full(self, tool_module):
        """Test creating input with all fields populated."""
        ValidateFidelityInput = tool_module.ValidateFidelityInput
        sim_id = str(uuid4())
        exp_id = str(uuid4())

        input_schema = ValidateFidelityInput(
            simulation_id=sim_id,
            actual_ate=0.092,
            actual_ci_lower=0.075,
            actual_ci_upper=0.109,
            actual_sample_size=2500,
            actual_experiment_id=exp_id,
            validation_notes="Experiment completed successfully",
            confounding_factors=["seasonal_variation", "competitor_launch"],
        )

        assert input_schema.simulation_id == sim_id
        assert input_schema.actual_ate == 0.092
        assert input_schema.actual_ci_lower == 0.075
        assert input_schema.actual_ci_upper == 0.109
        assert input_schema.actual_sample_size == 2500
        assert input_schema.actual_experiment_id == exp_id
        assert input_schema.validation_notes == "Experiment completed successfully"
        assert len(input_schema.confounding_factors) == 2

    def test_confounding_factors_list(self, tool_module):
        """Test confounding factors list handling."""
        ValidateFidelityInput = tool_module.ValidateFidelityInput
        input_schema = ValidateFidelityInput(
            simulation_id=str(uuid4()),
            actual_ate=0.08,
            confounding_factors=["covid_impact", "market_shift", "regulatory_change"],
        )

        assert input_schema.confounding_factors == [
            "covid_impact",
            "market_shift",
            "regulatory_change",
        ]

    def test_optional_fields_default_none(self, tool_module):
        """Test that optional fields default to None."""
        ValidateFidelityInput = tool_module.ValidateFidelityInput
        input_schema = ValidateFidelityInput(
            simulation_id=str(uuid4()),
            actual_ate=0.05,
        )

        assert input_schema.actual_ci_lower is None
        assert input_schema.actual_ci_upper is None
        assert input_schema.actual_sample_size is None
        assert input_schema.actual_experiment_id is None
        assert input_schema.validation_notes is None
        assert input_schema.confounding_factors is None


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestValidateTwinFidelity:
    """Test validate_twin_fidelity tool function."""

    def test_returns_required_output_fields(self, tool_module):
        """Test all required output fields are present."""
        validate_twin_fidelity = tool_module.validate_twin_fidelity
        sim_id = str(uuid4())

        result = validate_twin_fidelity.invoke({
            "simulation_id": sim_id,
            "actual_ate": 0.08,
        })

        # Check all required fields
        required_fields = [
            "tracking_id",
            "simulation_id",
            "simulated_ate",
            "actual_ate",
            "prediction_error",
            "absolute_error",
            "ci_coverage",
            "fidelity_grade",
            "assessment_message",
            "model_update_recommended",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_handles_invalid_simulation_id_format(self, tool_module):
        """Test handling of invalid simulation ID format."""
        validate_twin_fidelity = tool_module.validate_twin_fidelity
        result = validate_twin_fidelity.invoke({
            "simulation_id": "not-a-valid-uuid",
            "actual_ate": 0.08,
        })

        # Should return error response
        assert result["tracking_id"] == "error"
        assert "validation failed" in result["assessment_message"].lower() or \
               "invalid" in result["assessment_message"].lower()

    def test_handles_missing_simulation(self, tool_module):
        """Test handling of simulation not found in tracker."""
        validate_twin_fidelity = tool_module.validate_twin_fidelity
        # Use a valid but non-existent UUID
        sim_id = str(uuid4())

        result = validate_twin_fidelity.invoke({
            "simulation_id": sim_id,
            "actual_ate": 0.075,
        })

        # Should create new validation record
        assert "tracking_id" in result
        assert result["actual_ate"] == 0.075
        # When simulation not found, simulated_ate is 0 (unknown)
        assert result["fidelity_grade"] == FidelityGrade.UNVALIDATED.value

    def test_actual_ate_recorded(self, tool_module):
        """Test that actual ATE is properly recorded."""
        validate_twin_fidelity = tool_module.validate_twin_fidelity
        sim_id = str(uuid4())
        expected_ate = 0.092

        result = validate_twin_fidelity.invoke({
            "simulation_id": sim_id,
            "actual_ate": expected_ate,
        })

        assert result["actual_ate"] == expected_ate


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestFidelityGradeAssignment:
    """Test fidelity grade assignment based on prediction error."""

    def test_excellent_prediction(self, tool_module):
        """Test excellent grade for <10% error."""
        _generate_assessment = tool_module._generate_assessment
        # Mock fidelity record with excellent prediction
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.EXCELLENT
        record.simulated_ate = 0.10
        record.actual_ate = 0.095  # ~5% error
        record.prediction_error = 0.053
        record.ci_coverage = True
        record.confounding_factors = []

        assessment = _generate_assessment(record)

        assert "excellent" in assessment["message"].lower()
        assert assessment["update_recommended"] is False

    def test_good_prediction(self, tool_module):
        """Test good grade for 10-20% error."""
        _generate_assessment = tool_module._generate_assessment
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.GOOD
        record.simulated_ate = 0.10
        record.actual_ate = 0.085  # ~15% error
        record.prediction_error = 0.15
        record.ci_coverage = True
        record.confounding_factors = []

        assessment = _generate_assessment(record)

        assert "good" in assessment["message"].lower()
        assert assessment["update_recommended"] is False

    def test_fair_prediction(self, tool_module):
        """Test fair grade for 20-35% error."""
        _generate_assessment = tool_module._generate_assessment
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.FAIR
        record.simulated_ate = 0.10
        record.actual_ate = 0.075  # ~25% error
        record.prediction_error = 0.25
        record.ci_coverage = False
        record.confounding_factors = []

        assessment = _generate_assessment(record)

        assert "fair" in assessment["message"].lower()
        assert assessment["update_recommended"] is True

    def test_poor_prediction(self, tool_module):
        """Test poor grade for >35% error."""
        _generate_assessment = tool_module._generate_assessment
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.POOR
        record.simulated_ate = 0.10
        record.actual_ate = 0.055  # ~45% error
        record.prediction_error = 0.45
        record.ci_coverage = False
        record.confounding_factors = []

        assessment = _generate_assessment(record)

        assert "poor" in assessment["message"].lower()
        assert assessment["update_recommended"] is True
        assert "retraining" in assessment["message"].lower()


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestCICoverageCheck:
    """Test confidence interval coverage checking."""

    def test_ci_coverage_check_within_bounds(self, tool_module):
        """Test CI coverage when actual is within predicted CI."""
        _generate_assessment = tool_module._generate_assessment
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.GOOD
        record.simulated_ate = 0.10
        record.actual_ate = 0.095
        record.prediction_error = 0.05
        record.ci_coverage = True  # Actual within predicted CI
        record.confounding_factors = []

        assessment = _generate_assessment(record)

        assert "fell within predicted" in assessment["message"]
        # Should not recommend update if CI coverage is good
        assert assessment["update_recommended"] is False

    def test_ci_coverage_check_outside_bounds(self, tool_module):
        """Test CI coverage when actual is outside predicted CI."""
        _generate_assessment = tool_module._generate_assessment
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.FAIR
        record.simulated_ate = 0.10
        record.actual_ate = 0.06
        record.prediction_error = 0.40
        record.ci_coverage = False  # Actual outside predicted CI
        record.confounding_factors = []

        assessment = _generate_assessment(record)

        assert "fell outside predicted" in assessment["message"].lower() or \
               "outside" in assessment["message"].lower()
        # Should recommend update when CI coverage fails
        assert assessment["update_recommended"] is True


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestModelUpdateRecommendation:
    """Test model update recommendation logic."""

    def test_model_update_not_recommended_excellent(self, tool_module):
        """Test no update recommended for excellent predictions."""
        _generate_assessment = tool_module._generate_assessment
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.EXCELLENT
        record.simulated_ate = 0.10
        record.actual_ate = 0.098
        record.prediction_error = 0.02
        record.ci_coverage = True
        record.confounding_factors = []

        assessment = _generate_assessment(record)

        assert assessment["update_recommended"] is False

    def test_model_update_recommended_poor(self, tool_module):
        """Test update recommended for poor predictions."""
        _generate_assessment = tool_module._generate_assessment
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.POOR
        record.simulated_ate = 0.10
        record.actual_ate = 0.05
        record.prediction_error = 0.50
        record.ci_coverage = False
        record.confounding_factors = []

        assessment = _generate_assessment(record)

        assert assessment["update_recommended"] is True

    def test_model_update_recommended_ci_failure(self, tool_module):
        """Test update recommended when CI coverage fails."""
        _generate_assessment = tool_module._generate_assessment
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.GOOD  # Grade is good
        record.simulated_ate = 0.10
        record.actual_ate = 0.085
        record.prediction_error = 0.15
        record.ci_coverage = False  # But CI coverage failed
        record.confounding_factors = []

        assessment = _generate_assessment(record)

        # Should recommend update due to CI failure
        assert assessment["update_recommended"] is True


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestGetModelFidelityReport:
    """Test get_model_fidelity_report tool."""

    def test_report_structure(self, tool_module):
        """Test fidelity report has expected structure."""
        get_model_fidelity_report = tool_module.get_model_fidelity_report
        model_id = str(uuid4())

        result = get_model_fidelity_report.invoke({
            "model_id": model_id,
        })

        expected_fields = [
            "model_id",
            "validation_count",
        ]

        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

    def test_lookback_days_parameter(self, tool_module):
        """Test lookback_days parameter is respected."""
        get_model_fidelity_report = tool_module.get_model_fidelity_report
        model_id = str(uuid4())

        result = get_model_fidelity_report.invoke({
            "model_id": model_id,
            "lookback_days": 30,
        })

        assert "model_id" in result
        # Should return result (may have 0 validations)
        assert result["validation_count"] >= 0

    def test_handles_no_validations(self, tool_module):
        """Test handling of model with no validations."""
        get_model_fidelity_report = tool_module.get_model_fidelity_report
        model_id = str(uuid4())

        result = get_model_fidelity_report.invoke({
            "model_id": model_id,
            "lookback_days": 90,
        })

        assert result["validation_count"] == 0


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestAssessmentGeneration:
    """Test _generate_assessment function."""

    def test_assessment_excellent_grade(self, tool_module):
        """Test assessment message for excellent grade."""
        _generate_assessment = tool_module._generate_assessment
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.EXCELLENT
        record.simulated_ate = 0.10
        record.actual_ate = 0.097
        record.prediction_error = 0.03
        record.ci_coverage = None
        record.confounding_factors = []

        assessment = _generate_assessment(record)

        assert "excellent" in assessment["message"].lower()
        assert assessment["update_recommended"] is False

    def test_assessment_good_grade(self, tool_module):
        """Test assessment message for good grade."""
        _generate_assessment = tool_module._generate_assessment
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.GOOD
        record.simulated_ate = 0.10
        record.actual_ate = 0.085
        record.prediction_error = 0.15
        record.ci_coverage = None
        record.confounding_factors = []

        assessment = _generate_assessment(record)

        assert "good" in assessment["message"].lower()
        assert assessment["update_recommended"] is False

    def test_assessment_fair_grade(self, tool_module):
        """Test assessment message for fair grade."""
        _generate_assessment = tool_module._generate_assessment
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.FAIR
        record.simulated_ate = 0.10
        record.actual_ate = 0.072
        record.prediction_error = 0.28
        record.ci_coverage = None
        record.confounding_factors = []

        assessment = _generate_assessment(record)

        assert "fair" in assessment["message"].lower()
        assert assessment["update_recommended"] is True

    def test_assessment_poor_grade(self, tool_module):
        """Test assessment message for poor grade."""
        _generate_assessment = tool_module._generate_assessment
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.POOR
        record.simulated_ate = 0.10
        record.actual_ate = 0.05
        record.prediction_error = 0.50
        record.ci_coverage = None
        record.confounding_factors = []

        assessment = _generate_assessment(record)

        assert "poor" in assessment["message"].lower()
        assert "retraining" in assessment["message"].lower()
        assert assessment["update_recommended"] is True

    def test_assessment_with_confounding_factors(self, tool_module):
        """Test assessment includes confounding factors."""
        _generate_assessment = tool_module._generate_assessment
        record = MagicMock()
        record.fidelity_grade = FidelityGrade.FAIR
        record.simulated_ate = 0.10
        record.actual_ate = 0.07
        record.prediction_error = 0.30
        record.ci_coverage = False
        record.confounding_factors = ["competitor_launch", "seasonal_effect"]

        assessment = _generate_assessment(record)

        assert "confounding" in assessment["message"].lower()
        assert "competitor_launch" in assessment["message"]


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestErrorHandling:
    """Test error handling in validation tools."""

    def test_create_error_response(self, tool_module):
        """Test error response creation."""
        _create_error_response = tool_module._create_error_response
        error_msg = "Database connection failed"

        response = _create_error_response(error_msg)

        assert response["tracking_id"] == "error"
        assert response["simulation_id"] == "unknown"
        assert error_msg in response["assessment_message"]
        assert response["model_update_recommended"] is False
        assert response["fidelity_grade"] == FidelityGrade.UNVALIDATED.value

    def test_create_new_validation_structure(self, tool_module):
        """Test new validation record structure."""
        _create_new_validation = tool_module._create_new_validation
        sim_id = str(uuid4())

        result = _create_new_validation(
            simulation_id=sim_id,
            actual_ate=0.085,
            actual_ci_lower=0.065,
            actual_ci_upper=0.105,
            validation_notes="Manual entry",
        )

        assert result["simulation_id"] == sim_id
        assert result["actual_ate"] == 0.085
        assert result["fidelity_grade"] == FidelityGrade.UNVALIDATED.value
        assert result["simulated_ate"] == 0.0  # Unknown
        assert "tracking_id" in result


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestValidateTwinFidelityEdgeCases:
    """Test edge cases for validate_twin_fidelity tool."""

    def test_negative_actual_ate(self, tool_module):
        """Test handling of negative actual ATE."""
        validate_twin_fidelity = tool_module.validate_twin_fidelity
        sim_id = str(uuid4())

        result = validate_twin_fidelity.invoke({
            "simulation_id": sim_id,
            "actual_ate": -0.05,  # Negative effect
        })

        assert result["actual_ate"] == -0.05

    def test_zero_actual_ate(self, tool_module):
        """Test handling of zero actual ATE."""
        validate_twin_fidelity = tool_module.validate_twin_fidelity
        sim_id = str(uuid4())

        result = validate_twin_fidelity.invoke({
            "simulation_id": sim_id,
            "actual_ate": 0.0,
        })

        assert result["actual_ate"] == 0.0

    def test_large_actual_ate(self, tool_module):
        """Test handling of large actual ATE."""
        validate_twin_fidelity = tool_module.validate_twin_fidelity
        sim_id = str(uuid4())

        result = validate_twin_fidelity.invoke({
            "simulation_id": sim_id,
            "actual_ate": 0.50,  # 50% effect - unusually high
        })

        assert result["actual_ate"] == 0.50

    def test_validation_with_all_optional_fields(self, tool_module):
        """Test validation with all optional fields provided."""
        validate_twin_fidelity = tool_module.validate_twin_fidelity
        sim_id = str(uuid4())
        exp_id = str(uuid4())

        result = validate_twin_fidelity.invoke({
            "simulation_id": sim_id,
            "actual_ate": 0.082,
            "actual_ci_lower": 0.065,
            "actual_ci_upper": 0.099,
            "actual_sample_size": 2500,
            "actual_experiment_id": exp_id,
            "validation_notes": "Full validation test",
            "confounding_factors": ["factor_a", "factor_b"],
        })

        assert "tracking_id" in result
        assert result["actual_ate"] == 0.082
