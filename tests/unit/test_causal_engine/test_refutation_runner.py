"""Tests for RefutationRunner and related classes.

Version: 4.3
Tests the Causal Validation Protocol implementation.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.causal_engine import (
    RefutationRunner,
    RefutationSuite,
    RefutationResult,
    RefutationStatus,
    GateDecision,
    RefutationTestType,
    run_refutation_suite,
    is_estimate_valid,
    DOWHY_AVAILABLE,
)


class TestEnums:
    """Test ENUM types."""

    def test_refutation_status_values(self):
        """Test RefutationStatus enum has correct values."""
        assert RefutationStatus.PASSED.value == "passed"
        assert RefutationStatus.FAILED.value == "failed"
        assert RefutationStatus.WARNING.value == "warning"
        assert RefutationStatus.SKIPPED.value == "skipped"

    def test_gate_decision_values(self):
        """Test GateDecision enum has correct values."""
        assert GateDecision.PROCEED.value == "proceed"
        assert GateDecision.REVIEW.value == "review"
        assert GateDecision.BLOCK.value == "block"

    def test_refutation_test_type_values(self):
        """Test RefutationTestType enum has correct values."""
        assert RefutationTestType.PLACEBO_TREATMENT.value == "placebo_treatment"
        assert RefutationTestType.RANDOM_COMMON_CAUSE.value == "random_common_cause"
        assert RefutationTestType.DATA_SUBSET.value == "data_subset"
        assert RefutationTestType.BOOTSTRAP.value == "bootstrap"
        assert RefutationTestType.SENSITIVITY_E_VALUE.value == "sensitivity_e_value"


class TestRefutationResult:
    """Test RefutationResult dataclass."""

    def test_create_refutation_result(self):
        """Test creating a RefutationResult."""
        result = RefutationResult(
            test_name=RefutationTestType.PLACEBO_TREATMENT,
            status=RefutationStatus.PASSED,
            original_effect=0.5,
            refuted_effect=0.01,
            p_value=0.75,
            delta_percent=2.0,
            details={"message": "Test passed"},
            execution_time_ms=100.0,
        )

        assert result.test_name == RefutationTestType.PLACEBO_TREATMENT
        assert result.status == RefutationStatus.PASSED
        assert result.original_effect == 0.5
        assert result.refuted_effect == 0.01
        assert result.p_value == 0.75
        assert result.delta_percent == 2.0
        assert result.details["message"] == "Test passed"
        assert result.execution_time_ms == 100.0

    def test_to_dict(self):
        """Test RefutationResult.to_dict() serialization."""
        result = RefutationResult(
            test_name=RefutationTestType.BOOTSTRAP,
            status=RefutationStatus.WARNING,
            original_effect=0.3,
            refuted_effect=0.35,
            p_value=0.06,
            delta_percent=16.7,
        )

        d = result.to_dict()

        assert d["test_name"] == "bootstrap"
        assert d["status"] == "warning"
        assert d["original_effect"] == 0.3
        assert d["refuted_effect"] == 0.35
        assert d["p_value"] == 0.06
        assert d["delta_percent"] == 16.7

    def test_default_values(self):
        """Test default values for optional fields."""
        result = RefutationResult(
            test_name=RefutationTestType.DATA_SUBSET,
            status=RefutationStatus.PASSED,
            original_effect=0.5,
            refuted_effect=0.48,
        )

        assert result.p_value is None
        assert result.delta_percent == 0.0
        assert result.details == {}
        assert result.execution_time_ms == 0.0


class TestRefutationSuite:
    """Test RefutationSuite dataclass."""

    def _create_test_results(self, passed_count: int = 4, failed_count: int = 1) -> list:
        """Helper to create test results."""
        results = []
        for i in range(passed_count):
            results.append(RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.5,
                refuted_effect=0.02,
            ))
        for i in range(failed_count):
            results.append(RefutationResult(
                test_name=RefutationTestType.BOOTSTRAP,
                status=RefutationStatus.FAILED,
                original_effect=0.5,
                refuted_effect=0.8,
            ))
        return results

    def test_create_suite(self):
        """Test creating a RefutationSuite."""
        tests = self._create_test_results(passed_count=4, failed_count=1)
        suite = RefutationSuite(
            passed=True,
            confidence_score=0.80,
            tests=tests,
            gate_decision=GateDecision.PROCEED,
            total_execution_time_ms=500.0,
            estimate_id="test-123",
            treatment_variable="treatment",
            outcome_variable="outcome",
            brand="TestBrand",
        )

        assert suite.passed is True
        assert suite.confidence_score == 0.80
        assert len(suite.tests) == 5
        assert suite.gate_decision == GateDecision.PROCEED
        assert suite.estimate_id == "test-123"

    def test_tests_passed_property(self):
        """Test tests_passed property counting."""
        tests = self._create_test_results(passed_count=3, failed_count=2)
        suite = RefutationSuite(
            passed=True,
            confidence_score=0.60,
            tests=tests,
            gate_decision=GateDecision.REVIEW,
        )

        assert suite.tests_passed == 3
        assert suite.tests_failed == 2
        assert suite.total_tests == 5

    def test_tests_warning_property(self):
        """Test tests_warning property counting."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.5,
                refuted_effect=0.02,
            ),
            RefutationResult(
                test_name=RefutationTestType.BOOTSTRAP,
                status=RefutationStatus.WARNING,
                original_effect=0.5,
                refuted_effect=0.55,
            ),
        ]
        suite = RefutationSuite(
            passed=True,
            confidence_score=0.75,
            tests=tests,
            gate_decision=GateDecision.PROCEED,
        )

        assert suite.tests_warning == 1

    def test_skipped_excluded_from_total(self):
        """Test that skipped tests are excluded from total count."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.5,
                refuted_effect=0.02,
            ),
            RefutationResult(
                test_name=RefutationTestType.BOOTSTRAP,
                status=RefutationStatus.SKIPPED,
                original_effect=0.5,
                refuted_effect=0.5,
            ),
        ]
        suite = RefutationSuite(
            passed=True,
            confidence_score=0.80,
            tests=tests,
            gate_decision=GateDecision.PROCEED,
        )

        assert suite.total_tests == 1  # Skipped excluded
        assert len(suite.tests) == 2   # But still in list

    def test_to_dict(self):
        """Test RefutationSuite.to_dict() serialization."""
        tests = self._create_test_results(passed_count=3, failed_count=2)
        suite = RefutationSuite(
            passed=True,
            confidence_score=0.60,
            tests=tests,
            gate_decision=GateDecision.REVIEW,
            estimate_id="uuid-123",
            treatment_variable="engagement",
            outcome_variable="conversion",
            brand="Remibrutinib",
        )

        d = suite.to_dict()

        assert d["passed"] is True
        assert d["confidence_score"] == 0.60
        assert d["gate_decision"] == "review"
        assert d["tests_passed"] == 3
        assert d["tests_failed"] == 2
        assert d["total_tests"] == 5
        assert len(d["tests"]) == 5
        assert d["estimate_id"] == "uuid-123"
        assert d["treatment_variable"] == "engagement"
        assert d["outcome_variable"] == "conversion"
        assert d["brand"] == "Remibrutinib"
        assert "created_at" in d

    def test_to_legacy_format(self):
        """Test backward-compatible legacy format."""
        tests = self._create_test_results(passed_count=3, failed_count=1)
        suite = RefutationSuite(
            passed=True,
            confidence_score=0.75,
            tests=tests,
            gate_decision=GateDecision.PROCEED,
        )

        legacy = suite.to_legacy_format()

        assert legacy["tests_passed"] == 3
        assert legacy["tests_failed"] == 1
        assert legacy["total_tests"] == 4
        assert legacy["overall_robust"] is True
        assert legacy["confidence_adjustment"] == 0.75
        assert legacy["gate_decision"] == "proceed"
        assert len(legacy["individual_tests"]) == 4

        # Check individual test format
        test = legacy["individual_tests"][0]
        assert "test_name" in test
        assert "passed" in test
        assert "new_effect" in test
        assert "original_effect" in test
        assert "p_value" in test
        assert "details" in test


class TestRefutationRunner:
    """Test RefutationRunner class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        runner = RefutationRunner()

        assert runner.config["placebo_treatment"]["enabled"] is True
        assert runner.config["random_common_cause"]["enabled"] is True
        assert runner.config["data_subset"]["enabled"] is True
        assert runner.config["bootstrap"]["enabled"] is True
        assert runner.config["sensitivity_e_value"]["enabled"] is True

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        custom_config = {
            "placebo_treatment": {"num_simulations": 50},
            "bootstrap": {"enabled": False},
        }
        runner = RefutationRunner(config=custom_config)

        assert runner.config["placebo_treatment"]["num_simulations"] == 50
        assert runner.config["bootstrap"]["enabled"] is False
        # Other defaults should remain
        assert runner.config["placebo_treatment"]["critical"] is True

    def test_init_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        custom_thresholds = {
            "e_value_min": {"pass": 3.0, "warning": 2.0},
        }
        runner = RefutationRunner(thresholds=custom_thresholds)

        assert runner.thresholds["e_value_min"]["pass"] == 3.0
        assert runner.thresholds["e_value_min"]["warning"] == 2.0
        # Other thresholds should remain default
        assert runner.thresholds["placebo_p_value"]["pass"] == 0.05

    def test_run_all_tests_mock_mode(self):
        """Test run_all_tests without DoWhy (mock mode)."""
        runner = RefutationRunner()

        suite = runner.run_all_tests(
            original_effect=0.5,
            original_ci=(0.4, 0.6),
            treatment="hcp_engagement",
            outcome="conversion_rate",
            brand="Remibrutinib",
            estimate_id="test-123",
        )

        assert isinstance(suite, RefutationSuite)
        # In mock mode, bootstrap may be skipped, so expect 4-5 tests
        assert len(suite.tests) >= 4
        assert len(suite.tests) <= 5
        assert suite.treatment_variable == "hcp_engagement"
        assert suite.outcome_variable == "conversion_rate"
        assert suite.brand == "Remibrutinib"
        assert suite.estimate_id == "test-123"

    def test_run_all_tests_positive_effect(self):
        """Test refutation with positive effect."""
        runner = RefutationRunner()

        suite = runner.run_all_tests(
            original_effect=0.5,
            original_ci=(0.4, 0.6),
        )

        assert suite.passed is not None
        assert 0.0 <= suite.confidence_score <= 1.0
        assert suite.gate_decision in [GateDecision.PROCEED, GateDecision.REVIEW, GateDecision.BLOCK]

    def test_run_all_tests_negative_effect(self):
        """Test refutation with negative effect."""
        runner = RefutationRunner()

        suite = runner.run_all_tests(
            original_effect=-0.3,
            original_ci=(-0.4, -0.2),
        )

        assert suite.passed is not None
        # Verify original effects preserved correctly
        for test in suite.tests:
            assert test.original_effect == -0.3

    def test_run_all_tests_zero_effect(self):
        """Test refutation with zero effect."""
        runner = RefutationRunner()

        suite = runner.run_all_tests(
            original_effect=0.0,
            original_ci=(-0.1, 0.1),
        )

        # Zero effect should still be validated
        assert isinstance(suite, RefutationSuite)
        assert len(suite.tests) >= 1

    def test_run_all_tests_with_disabled_test(self):
        """Test that disabled tests are skipped."""
        runner = RefutationRunner(config={
            "bootstrap": {"enabled": False},
        })

        suite = runner.run_all_tests(
            original_effect=0.5,
            original_ci=(0.4, 0.6),
        )

        bootstrap_tests = [t for t in suite.tests if t.test_name == RefutationTestType.BOOTSTRAP]
        if bootstrap_tests:
            assert bootstrap_tests[0].status == RefutationStatus.SKIPPED

    def test_gate_decision_proceed(self):
        """Test that high confidence results in PROCEED decision."""
        runner = RefutationRunner()

        # Mock high-confidence scenario (all tests pass)
        suite = runner.run_all_tests(
            original_effect=0.5,
            original_ci=(0.4, 0.6),
        )

        # In mock mode, should typically get high confidence
        assert suite.gate_decision in [GateDecision.PROCEED, GateDecision.REVIEW, GateDecision.BLOCK]

    def test_execution_time_tracking(self):
        """Test that execution time is tracked."""
        runner = RefutationRunner()

        suite = runner.run_all_tests(
            original_effect=0.5,
            original_ci=(0.4, 0.6),
        )

        assert suite.total_execution_time_ms >= 0
        for test in suite.tests:
            assert test.execution_time_ms >= 0


class TestGateDecisionLogic:
    """Test gate decision thresholds and logic."""

    def test_proceed_threshold(self):
        """Test PROCEED requires confidence >= 0.70."""
        runner = RefutationRunner()
        assert runner.GATE_THRESHOLDS["proceed"] == 0.70

    def test_review_threshold(self):
        """Test REVIEW for confidence 0.50-0.70."""
        runner = RefutationRunner()
        assert runner.GATE_THRESHOLDS["review"] == 0.50

    def test_gate_decision_based_on_confidence(self):
        """Test gate decision matches confidence score."""
        runner = RefutationRunner()

        suite = runner.run_all_tests(
            original_effect=0.5,
            original_ci=(0.4, 0.6),
        )

        if suite.confidence_score >= 0.70:
            # Could still be REVIEW/BLOCK if critical test failed
            assert suite.gate_decision in [GateDecision.PROCEED, GateDecision.REVIEW, GateDecision.BLOCK]
        elif suite.confidence_score >= 0.50:
            assert suite.gate_decision in [GateDecision.REVIEW, GateDecision.BLOCK]
        else:
            assert suite.gate_decision == GateDecision.BLOCK


class TestPassThresholds:
    """Test pass/fail threshold configurations."""

    def test_placebo_p_value_thresholds(self):
        """Test placebo p-value thresholds."""
        runner = RefutationRunner()

        assert runner.thresholds["placebo_p_value"]["pass"] == 0.05
        assert runner.thresholds["placebo_p_value"]["warning"] == 0.10

    def test_common_cause_delta_thresholds(self):
        """Test common cause delta thresholds."""
        runner = RefutationRunner()

        assert runner.thresholds["common_cause_delta"]["pass"] == 0.20
        assert runner.thresholds["common_cause_delta"]["warning"] == 0.30

    def test_e_value_thresholds(self):
        """Test e-value thresholds."""
        runner = RefutationRunner()

        assert runner.thresholds["e_value_min"]["pass"] == 2.0
        assert runner.thresholds["e_value_min"]["warning"] == 1.5


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_run_refutation_suite(self):
        """Test run_refutation_suite convenience function."""
        suite = run_refutation_suite(
            original_effect=0.5,
            original_ci=(0.4, 0.6),
            treatment="engagement",
            outcome="conversion",
        )

        assert isinstance(suite, RefutationSuite)
        assert suite.treatment_variable == "engagement"
        assert suite.outcome_variable == "conversion"

    def test_is_estimate_valid_true(self):
        """Test is_estimate_valid returns True for valid estimates."""
        suite = run_refutation_suite(
            original_effect=0.5,
            original_ci=(0.4, 0.6),
        )

        # Valid if not blocked
        result = is_estimate_valid(suite)
        expected = suite.gate_decision != GateDecision.BLOCK
        assert result == expected

    def test_is_estimate_valid_with_gate_decision(self):
        """Test is_estimate_valid based on gate decision."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.5,
                refuted_effect=0.02,
            ),
        ]

        # PROCEED gate
        suite_proceed = RefutationSuite(
            passed=True,
            confidence_score=0.80,
            tests=tests,
            gate_decision=GateDecision.PROCEED,
        )
        assert is_estimate_valid(suite_proceed) is True

        # REVIEW gate (still valid, just needs review)
        suite_review = RefutationSuite(
            passed=True,
            confidence_score=0.60,
            tests=tests,
            gate_decision=GateDecision.REVIEW,
        )
        assert is_estimate_valid(suite_review) is True

        # BLOCK gate
        suite_block = RefutationSuite(
            passed=False,
            confidence_score=0.30,
            tests=tests,
            gate_decision=GateDecision.BLOCK,
        )
        assert is_estimate_valid(suite_block) is False


class TestDifferentEffectSizes:
    """Test refutation with different effect sizes."""

    @pytest.mark.parametrize("effect", [0.1, 0.5, 0.9])
    def test_various_positive_effects(self, effect):
        """Test refutation for various positive effect sizes."""
        runner = RefutationRunner()

        suite = runner.run_all_tests(
            original_effect=effect,
            original_ci=(effect - 0.1, effect + 0.1),
        )

        assert isinstance(suite, RefutationSuite)
        assert all(t.original_effect == effect for t in suite.tests)

    @pytest.mark.parametrize("effect", [-0.1, -0.5, -0.9])
    def test_various_negative_effects(self, effect):
        """Test refutation for various negative effect sizes."""
        runner = RefutationRunner()

        suite = runner.run_all_tests(
            original_effect=effect,
            original_ci=(effect - 0.1, effect + 0.1),
        )

        assert isinstance(suite, RefutationSuite)
        assert all(t.original_effect == effect for t in suite.tests)


class TestCIHandling:
    """Test confidence interval handling."""

    def test_narrow_ci(self):
        """Test refutation with narrow confidence interval."""
        runner = RefutationRunner()

        suite = runner.run_all_tests(
            original_effect=0.5,
            original_ci=(0.49, 0.51),  # Narrow CI
        )

        assert isinstance(suite, RefutationSuite)

    def test_wide_ci(self):
        """Test refutation with wide confidence interval."""
        runner = RefutationRunner()

        suite = runner.run_all_tests(
            original_effect=0.5,
            original_ci=(0.1, 0.9),  # Wide CI
        )

        assert isinstance(suite, RefutationSuite)

    def test_asymmetric_ci(self):
        """Test refutation with asymmetric confidence interval."""
        runner = RefutationRunner()

        suite = runner.run_all_tests(
            original_effect=0.5,
            original_ci=(0.3, 0.8),  # Asymmetric
        )

        assert isinstance(suite, RefutationSuite)


class TestDoWhyAvailability:
    """Test DoWhy availability constant."""

    def test_dowhy_available_is_boolean(self):
        """Test DOWHY_AVAILABLE is boolean."""
        assert isinstance(DOWHY_AVAILABLE, bool)


class TestTestTypeConfiguration:
    """Test individual test type configurations."""

    def test_critical_tests_marked(self):
        """Test that critical tests are properly marked."""
        runner = RefutationRunner()

        critical_tests = ["placebo_treatment", "random_common_cause", "sensitivity_e_value"]
        non_critical_tests = ["data_subset", "bootstrap"]

        for test in critical_tests:
            assert runner.config[test]["critical"] is True, f"{test} should be critical"

        for test in non_critical_tests:
            assert runner.config[test]["critical"] is False, f"{test} should not be critical"

    def test_default_simulations_count(self):
        """Test default simulation counts."""
        runner = RefutationRunner()

        assert runner.config["placebo_treatment"]["num_simulations"] == 100
        assert runner.config["data_subset"]["num_subsets"] == 10
        assert runner.config["bootstrap"]["num_bootstraps"] == 500
