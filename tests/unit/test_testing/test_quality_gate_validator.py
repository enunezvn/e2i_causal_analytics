"""Unit tests for QualityGateValidator.

Tests the quality gate validation for agent outputs.
"""

import os

import pytest

from src.testing.agent_quality_gates import AGENT_QUALITY_GATES
from src.testing.quality_gate_validator import (
    QualityCheckResult,
    QualityGateResult,
    QualityGateValidator,
)

# Set testing mode
os.environ["E2I_TESTING_MODE"] = "true"


@pytest.mark.unit
class TestQualityCheckResult:
    """Test QualityCheckResult dataclass."""

    def test_basic_creation(self):
        """Test creating a QualityCheckResult."""
        result = QualityCheckResult(
            check_name="test_check",
            field_name="test_field",
            passed=True,
            message="Check passed",
        )
        assert result.check_name == "test_check"
        assert result.passed is True

    def test_with_expected_actual(self):
        """Test QualityCheckResult with expected/actual values."""
        result = QualityCheckResult(
            check_name="type_check",
            field_name="status",
            passed=False,
            message="Type mismatch",
            expected="str",
            actual="int",
        )
        assert result.expected == "str"
        assert result.actual == "int"


@pytest.mark.unit
class TestQualityGateResult:
    """Test QualityGateResult dataclass."""

    def test_summary_pass(self):
        """Test summary for passed validation."""
        result = QualityGateResult(
            agent_name="test_agent",
            passed=True,
            total_checks=5,
            checks_passed=5,
        )
        assert "PASS" in result.summary
        assert "5/5" in result.summary

    def test_summary_pass_with_warnings(self):
        """Test summary for passed validation with warnings."""
        result = QualityGateResult(
            agent_name="test_agent",
            passed=True,
            total_checks=5,
            checks_passed=5,
            warnings=["Extra field found"],
        )
        assert "PASS" in result.summary
        assert "1 warnings" in result.summary

    def test_summary_fail(self):
        """Test summary for failed validation."""
        failed_check = QualityCheckResult(
            check_name="required_field",
            field_name="status",
            passed=False,
            message="Field missing",
        )
        result = QualityGateResult(
            agent_name="test_agent",
            passed=False,
            failed_checks=[failed_check],
        )
        assert "FAIL" in result.summary


@pytest.mark.unit
class TestQualityGateValidatorInit:
    """Test QualityGateValidator initialization."""

    def test_init_default(self):
        """Test initialization with default quality gates."""
        validator = QualityGateValidator()
        assert validator.quality_gates == AGENT_QUALITY_GATES

    def test_init_custom(self):
        """Test initialization with custom quality gates."""
        custom_gates = {
            "custom_agent": {
                "description": "Custom agent",
                "required_output_fields": ["result"],
            }
        }
        validator = QualityGateValidator(quality_gates=custom_gates)
        assert validator.quality_gates == custom_gates


@pytest.mark.unit
class TestRequiredOutputFields:
    """Test required output fields validation."""

    @pytest.fixture
    def validator(self):
        return QualityGateValidator()

    def test_all_required_fields_present(self, validator):
        """Test validation passes when all required fields present."""
        output = {
            "status": "completed",
            "response_text": "Analysis completed successfully",
        }
        result = validator.validate("orchestrator", output)

        assert len(result.required_output_fields_missing) == 0
        assert len(result.required_output_fields_present) > 0

    def test_required_field_missing(self, validator):
        """Test validation fails when required field missing."""
        output = {
            "response_text": "Analysis completed",
            # Missing "status" field
        }
        result = validator.validate("orchestrator", output)

        assert len(result.required_output_fields_missing) > 0
        assert result.passed is False

    def test_required_field_null(self, validator):
        """Test validation fails when required field is null."""
        output = {
            "status": None,
            "response_text": "Analysis completed",
        }
        result = validator.validate("orchestrator", output)

        assert "status" in result.required_output_fields_missing
        assert result.passed is False


@pytest.mark.unit
class TestMinRequiredFieldsPercentage:
    """Test minimum required fields percentage check."""

    @pytest.fixture
    def validator(self):
        return QualityGateValidator()

    def test_meets_minimum_percentage(self, validator):
        """Test validation passes when meeting minimum percentage."""
        output = {"status": "completed", "ate_estimate": 0.15}
        result = validator.validate(
            "causal_impact",
            output,
            contract_required_fields_pct=0.8,
        )

        # Should pass minimum percentage check (0.8 >= 0.4)
        assert any(
            c.check_name == "min_required_fields_pct" and c.passed for c in result.passed_checks
        )

    def test_below_minimum_percentage(self, validator):
        """Test validation fails below minimum percentage."""
        output = {"status": "completed"}
        result = validator.validate(
            "causal_impact",
            output,
            contract_required_fields_pct=0.3,  # Below 0.4 threshold
        )

        # Should fail minimum percentage check
        assert any(
            c.check_name == "min_required_fields_pct" and not c.passed for c in result.failed_checks
        )

    def test_total_false_typeddict_skips_check(self, validator):
        """Test percentage check skipped for total=False TypedDict."""
        output = {"status": "completed"}
        result = validator.validate(
            "causal_impact",
            output,
            contract_required_fields_pct=0.0,
            contract_required_total=0,  # No required fields
        )

        # Should auto-pass when no required fields
        assert any(
            c.check_name == "min_required_fields_pct" and c.passed for c in result.passed_checks
        )


@pytest.mark.unit
class TestDataQualityChecks:
    """Test data quality checks."""

    @pytest.fixture
    def validator(self):
        return QualityGateValidator()

    def test_type_check_pass(self, validator):
        """Test type check passes for correct type."""
        output = {
            "status": "completed",
            "overall_health_score": 85.5,
        }
        result = validator.validate("health_score", output)

        # Should have passing type check for overall_health_score
        assert any(
            c.check_name == "type" and c.field_name == "overall_health_score" and c.passed
            for c in result.passed_checks
        )

    def test_not_null_check_fail(self, validator):
        """Test not_null check fails when value is null."""
        output = {
            "status": None,  # Should not be null
        }
        result = validator.validate("causal_impact", output)

        assert any(c.check_name == "not_null" and not c.passed for c in result.failed_checks)

    def test_must_be_check_pass(self, validator):
        """Test must_be check passes for correct value."""
        output = {
            "success": True,
            "tools_executed": 3,
        }
        result = validator.validate("tool_composer", output)

        # Should pass must_be check for success=True
        assert any(c.check_name == "must_be" and c.passed for c in result.passed_checks)

    def test_must_not_be_check_fail(self, validator):
        """Test must_not_be check fails for forbidden value."""
        output = {
            "status": "error",  # Forbidden status
        }
        result = validator.validate("causal_impact", output)

        # Should fail must_not_be check
        assert any(c.check_name == "must_not_be" and not c.passed for c in result.failed_checks)

    def test_in_set_check_pass(self, validator):
        """Test in_set check passes for allowed value."""
        output = {
            "status": "completed",
        }
        result = validator.validate("feedback_learner", output)

        # Should pass in_set check
        assert any(c.check_name == "in_set" and c.passed for c in result.passed_checks)

    def test_not_contains_check_fail(self, validator):
        """Test not_contains check fails for error indicators."""
        output = {
            "executive_summary": "Error: Analysis failed to complete",
        }
        result = validator.validate("gap_analyzer", output)

        # Should fail not_contains check
        assert any(c.check_name == "not_contains" and not c.passed for c in result.failed_checks)

    def test_min_value_check(self, validator):
        """Test min_value check."""
        output = {
            "overall_health_score": 50.0,  # Below 0.0 would fail, but this passes
        }
        result = validator.validate("health_score", output)

        # Should have min_value check
        assert any(c.check_name == "min_value" for c in result.passed_checks + result.failed_checks)

    def test_max_value_check(self, validator):
        """Test max_value check."""
        output = {
            "overall_health_score": 85.0,  # Should be <= 100.0
        }
        result = validator.validate("health_score", output)

        # Should pass max_value check
        assert any(c.check_name == "max_value" and c.passed for c in result.passed_checks)

    def test_min_length_check(self, validator):
        """Test min_length check."""
        output = {
            "executive_summary": "This is a sufficiently long executive summary for validation",
        }
        result = validator.validate("explainer", output)

        # Should pass min_length check
        assert any(c.check_name == "min_length" and c.passed for c in result.passed_checks)


@pytest.mark.unit
class TestStatusFailure:
    """Test status-based failure detection."""

    @pytest.fixture
    def validator(self):
        return QualityGateValidator()

    def test_failure_status_detected(self, validator):
        """Test detection of failure status."""
        output = {
            "status": "error",
        }
        result = validator.validate("causal_impact", output)

        assert result.status_failure is True
        assert result.status_value == "error"
        assert result.passed is False

    def test_success_boolean_false(self, validator):
        """Test detection of success=False."""
        output = {
            "success": False,
        }
        result = validator.validate("tool_composer", output)

        assert result.status_failure is True
        assert result.passed is False

    def test_no_failure_status(self, validator):
        """Test no failure status detected."""
        output = {
            "status": "completed",
        }
        result = validator.validate("causal_impact", output)

        assert result.status_failure is False


@pytest.mark.unit
class TestSemanticValidation:
    """Test semantic validation integration."""

    @pytest.fixture
    def validator(self):
        return QualityGateValidator()

    def test_semantic_validation_pass(self, validator):
        """Test semantic validation passes for valid output."""
        output = {
            "status": "completed",
            "ate_estimate": 0.15,
            "confidence_interval": [0.10, 0.20],
        }
        result = validator.validate("causal_impact", output)

        # Should have passing semantic validation
        assert any(c.check_name == "semantic_validation" and c.passed for c in result.passed_checks)

    def test_semantic_validation_fail(self, validator):
        """Test semantic validation fails for invalid output."""
        output = {
            "status": "completed",
            "ate_estimate": 0.30,
            "confidence_interval": [0.10, 0.20],  # ATE outside CI
        }
        result = validator.validate("causal_impact", output)

        # Should have failing semantic validation
        assert any(
            c.check_name == "semantic_validation" and not c.passed for c in result.failed_checks
        )


@pytest.mark.unit
class TestValidationSummary:
    """Test validation summary metrics."""

    @pytest.fixture
    def validator(self):
        return QualityGateValidator()

    def test_summary_metrics(self, validator):
        """Test that summary metrics are calculated correctly."""
        output = {
            "status": "completed",
            "response_text": "Analysis completed successfully with detailed insights",
            "agents_dispatched": ["causal_impact"],
        }
        result = validator.validate("orchestrator", output)

        assert result.total_checks == len(result.passed_checks) + len(result.failed_checks)
        assert result.checks_passed == len(result.passed_checks)
        assert result.checks_failed == len(result.failed_checks)

    def test_overall_pass_fail(self, validator):
        """Test overall pass/fail determination."""
        # Passing case
        output = {
            "status": "completed",
            "response_text": "Analysis completed successfully with detailed insights",
            "agents_dispatched": ["causal_impact"],
        }
        result = validator.validate("orchestrator", output)
        assert result.passed == (len(result.failed_checks) == 0)


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def validator(self):
        return QualityGateValidator()

    def test_unknown_agent(self, validator):
        """Test validation for unknown agent."""
        result = validator.validate("unknown_agent", {"result": "success"})

        assert len(result.warnings) > 0
        assert "No quality gate configured" in result.warnings[0]
        assert result.passed is True

    def test_empty_output(self, validator):
        """Test validation with empty output."""
        result = validator.validate("causal_impact", {})

        # Should fail due to missing required fields
        assert result.passed is False
        assert len(result.failed_checks) > 0

    def test_type_check_with_unknown_type(self, validator):
        """Test type check with unknown type string."""
        # Create a custom gate with unknown type
        custom_gates = {
            "test_agent": {
                "description": "Test",
                "required_output_fields": [],
                "data_quality_checks": {
                    "field": {"type": "unknown_type"},
                },
            }
        }
        custom_validator = QualityGateValidator(quality_gates=custom_gates)

        output = {"field": "value"}
        result = custom_validator.validate("test_agent", output)

        # Should skip unknown type check
        assert any(c.check_name == "type" and c.passed for c in result.passed_checks)

    def test_float_accepts_int(self, validator):
        """Test that float type check accepts int values."""
        # Create a custom gate that expects float
        custom_gates = {
            "test_agent": {
                "description": "Test",
                "required_output_fields": [],
                "data_quality_checks": {
                    "score": {"type": "float"},
                },
            }
        }
        custom_validator = QualityGateValidator(quality_gates=custom_gates)

        output = {"score": 85}  # int instead of float
        result = custom_validator.validate("test_agent", output)

        # Should pass (int acceptable for float)
        assert any(
            c.check_name == "type" and c.field_name == "score" and c.passed
            for c in result.passed_checks
        )

    def test_none_value_skips_checks(self, validator):
        """Test that None values skip non-required checks."""
        custom_gates = {
            "test_agent": {
                "description": "Test",
                "required_output_fields": [],
                "data_quality_checks": {
                    "optional_field": {
                        "type": "str",
                        "min_length": 10,
                    },
                },
            }
        }
        custom_validator = QualityGateValidator(quality_gates=custom_gates)

        output = {"optional_field": None}
        result = custom_validator.validate("test_agent", output)

        # Should skip checks for None value
        assert result.passed is True

    def test_semantic_validator_exception(self, validator):
        """Test handling of semantic validator exceptions."""

        def failing_validator(output):
            raise ValueError("Test error")

        custom_gates = {
            "test_agent": {
                "description": "Test",
                "required_output_fields": [],
                "semantic_validator": failing_validator,
            }
        }
        custom_validator = QualityGateValidator(quality_gates=custom_gates)

        output = {"status": "completed"}
        result = custom_validator.validate("test_agent", output)

        # Should fail with semantic validation error
        assert any(
            c.check_name == "semantic_validation" and not c.passed for c in result.failed_checks
        )


@pytest.mark.unit
class TestIntegration:
    """Integration tests with real agent configurations."""

    @pytest.fixture
    def validator(self):
        return QualityGateValidator()

    def test_causal_impact_complete_validation(self, validator):
        """Test complete validation for causal_impact agent."""
        output = {
            "status": "completed",
            "ate_estimate": 0.15,
            "confidence_interval": [0.10, 0.20],
            "p_value": 0.02,
        }
        result = validator.validate("causal_impact", output)

        assert result.passed is True
        assert result.total_checks > 0

    def test_tool_composer_complete_validation(self, validator):
        """Test complete validation for tool_composer agent."""
        output = {
            "success": True,
            "tools_executed": 3,
            "tools_succeeded": 3,
            "confidence": 0.85,
            "response": "Successfully executed causal analysis and gap identification",
        }
        result = validator.validate("tool_composer", output)

        assert result.passed is True

    def test_health_score_complete_validation(self, validator):
        """Test complete validation for health_score agent."""
        output = {
            "overall_health_score": 85.5,
            "component_health_score": 0.9,
            "health_grade": "B",
        }
        result = validator.validate("health_score", output)

        # Should reject perfect 100.0 scores, accept realistic scores
        assert result.passed is True
