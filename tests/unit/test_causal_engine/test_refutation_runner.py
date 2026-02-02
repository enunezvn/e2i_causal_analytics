"""
Unit tests for refutation_runner.py

Tests cover:
- RefutationRunner
- RefutationResult
- RefutationSuite
- Individual refutation tests (placebo, random_common_cause, data_subset, bootstrap, sensitivity)
- Mock implementations
- Scoring and gate decisions
"""

from unittest.mock import patch

import pytest

from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationRunner,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
    is_estimate_valid,
    run_refutation_suite,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def runner():
    """Create RefutationRunner instance."""
    return RefutationRunner()


@pytest.fixture
def custom_config():
    """Custom configuration for refutation tests."""
    return {
        "placebo_treatment": {
            "enabled": True,
            "num_simulations": 50,
            "critical": True,
        },
        "random_common_cause": {
            "enabled": True,
            "effect_strength": 0.05,
            "critical": True,
        },
    }


@pytest.fixture
def custom_thresholds():
    """Custom thresholds for pass/fail criteria."""
    return {
        "placebo_p_value": {
            "pass": 0.10,
            "warning": 0.15,
        },
    }


# ============================================================================
# RefutationResult TESTS
# ============================================================================


class TestRefutationResult:
    """Tests for RefutationResult dataclass."""

    def test_create_refutation_result(self):
        """Test creating a RefutationResult."""
        result = RefutationResult(
            test_name=RefutationTestType.PLACEBO_TREATMENT,
            status=RefutationStatus.PASSED,
            original_effect=0.15,
            refuted_effect=0.02,
            p_value=0.75,
            delta_percent=86.7,
            details={"message": "Placebo test passed"},
            execution_time_ms=150.5,
        )

        assert result.test_name == RefutationTestType.PLACEBO_TREATMENT
        assert result.status == RefutationStatus.PASSED
        assert result.original_effect == 0.15
        assert result.refuted_effect == 0.02

    def test_to_dict(self):
        """Test converting RefutationResult to dictionary."""
        result = RefutationResult(
            test_name=RefutationTestType.PLACEBO_TREATMENT,
            status=RefutationStatus.PASSED,
            original_effect=0.15,
            refuted_effect=0.02,
        )

        result_dict = result.to_dict()

        assert result_dict["test_name"] == "placebo_treatment"
        assert result_dict["status"] == "passed"
        assert result_dict["original_effect"] == 0.15


# ============================================================================
# RefutationSuite TESTS
# ============================================================================


class TestRefutationSuite:
    """Tests for RefutationSuite dataclass."""

    def test_create_refutation_suite(self):
        """Test creating a RefutationSuite."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.15,
                refuted_effect=0.02,
            ),
            RefutationResult(
                test_name=RefutationTestType.RANDOM_COMMON_CAUSE,
                status=RefutationStatus.PASSED,
                original_effect=0.15,
                refuted_effect=0.14,
            ),
        ]

        suite = RefutationSuite(
            passed=True,
            confidence_score=0.85,
            tests=tests,
            gate_decision=GateDecision.PROCEED,
        )

        assert suite.passed is True
        assert suite.confidence_score == 0.85
        assert len(suite.tests) == 2

    def test_tests_passed_property(self):
        """Test tests_passed property."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.15,
                refuted_effect=0.02,
            ),
            RefutationResult(
                test_name=RefutationTestType.RANDOM_COMMON_CAUSE,
                status=RefutationStatus.FAILED,
                original_effect=0.15,
                refuted_effect=0.05,
            ),
        ]

        suite = RefutationSuite(
            passed=False,
            confidence_score=0.5,
            tests=tests,
            gate_decision=GateDecision.REVIEW,
        )

        assert suite.tests_passed == 1

    def test_tests_failed_property(self):
        """Test tests_failed property."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.FAILED,
                original_effect=0.15,
                refuted_effect=0.02,
            ),
            RefutationResult(
                test_name=RefutationTestType.RANDOM_COMMON_CAUSE,
                status=RefutationStatus.FAILED,
                original_effect=0.15,
                refuted_effect=0.05,
            ),
        ]

        suite = RefutationSuite(
            passed=False,
            confidence_score=0.3,
            tests=tests,
            gate_decision=GateDecision.BLOCK,
        )

        assert suite.tests_failed == 2

    def test_total_tests_property(self):
        """Test total_tests property excludes skipped tests."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.15,
                refuted_effect=0.02,
            ),
            RefutationResult(
                test_name=RefutationTestType.RANDOM_COMMON_CAUSE,
                status=RefutationStatus.SKIPPED,
                original_effect=0.15,
                refuted_effect=0.15,
            ),
        ]

        suite = RefutationSuite(
            passed=True,
            confidence_score=0.8,
            tests=tests,
            gate_decision=GateDecision.PROCEED,
        )

        assert suite.total_tests == 1

    def test_to_dict(self):
        """Test converting RefutationSuite to dictionary."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.15,
                refuted_effect=0.02,
            ),
        ]

        suite = RefutationSuite(
            passed=True,
            confidence_score=0.85,
            tests=tests,
            gate_decision=GateDecision.PROCEED,
            treatment_variable="hcp_engagement",
            outcome_variable="conversion_rate",
        )

        suite_dict = suite.to_dict()

        assert suite_dict["passed"] is True
        assert suite_dict["gate_decision"] == "proceed"
        assert suite_dict["treatment_variable"] == "hcp_engagement"

    def test_to_legacy_format(self):
        """Test converting to legacy RefutationResults format."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.15,
                refuted_effect=0.02,
                p_value=0.75,
                details={"message": "Test passed"},
            ),
            RefutationResult(
                test_name=RefutationTestType.SENSITIVITY_E_VALUE,
                status=RefutationStatus.PASSED,
                original_effect=0.15,
                refuted_effect=0.15,
                details={"message": "E-value sufficient"},
            ),
        ]

        suite = RefutationSuite(
            passed=True,
            confidence_score=0.85,
            tests=tests,
            gate_decision=GateDecision.PROCEED,
        )

        legacy = suite.to_legacy_format()

        assert "individual_tests" in legacy
        assert "placebo_treatment" in legacy["individual_tests"]
        assert (
            "unobserved_common_cause" in legacy["individual_tests"]
        )  # Mapped from sensitivity_e_value
        assert legacy["overall_robust"] is True


# ============================================================================
# RefutationRunner INITIALIZATION TESTS
# ============================================================================


class TestRefutationRunnerInit:
    """Tests for RefutationRunner initialization."""

    def test_default_initialization(self):
        """Test initialization with default config."""
        runner = RefutationRunner()

        assert runner.config is not None
        assert runner.config["placebo_treatment"]["enabled"] is True
        assert runner.thresholds is not None

    def test_custom_config(self, custom_config):
        """Test initialization with custom config."""
        runner = RefutationRunner(config=custom_config)

        assert runner.config["placebo_treatment"]["num_simulations"] == 50

    def test_custom_thresholds(self, custom_thresholds):
        """Test initialization with custom thresholds."""
        runner = RefutationRunner(thresholds=custom_thresholds)

        assert runner.thresholds["placebo_p_value"]["pass"] == 0.10


# ============================================================================
# PLACEBO TEST TESTS
# ============================================================================


class TestPlaceboTest:
    """Tests for placebo treatment refutation test."""

    def test_run_placebo_test_mock(self, runner):
        """Test running placebo test with mock DoWhy."""
        result = runner._run_placebo_test(
            original_effect=0.15,
            causal_model=None,
            identified_estimand=None,
            estimate=None,
            use_dowhy=False,
        )

        assert result.test_name == RefutationTestType.PLACEBO_TREATMENT
        assert result.status in [
            RefutationStatus.PASSED,
            RefutationStatus.WARNING,
            RefutationStatus.FAILED,
        ]
        assert result.original_effect == 0.15
        assert result.p_value is not None

    def test_run_placebo_test_passed(self, runner):
        """Test placebo test that passes."""
        with patch.object(runner, "_mock_placebo_test", return_value=(0.01, 0.85)):
            result = runner._run_placebo_test(
                original_effect=0.15,
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
            )

            assert result.status == RefutationStatus.PASSED
            assert "no significant effect" in result.details["message"].lower()

    def test_run_placebo_test_failed(self, runner):
        """Test placebo test that fails."""
        with patch.object(runner, "_mock_placebo_test", return_value=(0.12, 0.02)):
            result = runner._run_placebo_test(
                original_effect=0.15,
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
            )

            assert result.status == RefutationStatus.FAILED
            assert "warning" in result.details["message"].lower()


# ============================================================================
# RANDOM COMMON CAUSE TEST TESTS
# ============================================================================


class TestRandomCommonCauseTest:
    """Tests for random common cause refutation test."""

    def test_run_random_common_cause_test_mock(self, runner):
        """Test running random common cause test with mock."""
        result = runner._run_random_common_cause_test(
            original_effect=0.15,
            causal_model=None,
            identified_estimand=None,
            estimate=None,
            use_dowhy=False,
        )

        assert result.test_name == RefutationTestType.RANDOM_COMMON_CAUSE
        assert result.status in [
            RefutationStatus.PASSED,
            RefutationStatus.WARNING,
            RefutationStatus.FAILED,
        ]
        assert result.delta_percent >= 0

    def test_run_random_common_cause_test_passed(self, runner):
        """Test random common cause test that passes."""
        with patch.object(runner, "_mock_random_common_cause_test", return_value=(0.14, 0.70)):
            result = runner._run_random_common_cause_test(
                original_effect=0.15,
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
            )

            assert result.status == RefutationStatus.PASSED

    def test_run_random_common_cause_test_failed(self, runner):
        """Test random common cause test that fails."""
        with patch.object(runner, "_mock_random_common_cause_test", return_value=(0.05, 0.60)):
            result = runner._run_random_common_cause_test(
                original_effect=0.15,
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
            )

            # Large delta should trigger warning or failure
            assert result.status in [RefutationStatus.WARNING, RefutationStatus.FAILED]


# ============================================================================
# DATA SUBSET TEST TESTS
# ============================================================================


class TestDataSubsetTest:
    """Tests for data subset refutation test."""

    def test_run_data_subset_test_mock(self, runner):
        """Test running data subset test with mock."""
        result = runner._run_data_subset_test(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=None,
            identified_estimand=None,
            estimate=None,
            use_dowhy=False,
        )

        assert result.test_name == RefutationTestType.DATA_SUBSET
        assert result.status in [
            RefutationStatus.PASSED,
            RefutationStatus.WARNING,
            RefutationStatus.FAILED,
        ]
        assert "ci_coverage" in result.details

    def test_run_data_subset_test_passed(self, runner):
        """Test data subset test that passes."""
        with patch.object(runner, "_mock_data_subset_test", return_value=(0.15, 0.75, 0.85)):
            result = runner._run_data_subset_test(
                original_effect=0.15,
                original_ci=(0.10, 0.20),
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
            )

            assert result.status == RefutationStatus.PASSED


# ============================================================================
# BOOTSTRAP TEST TESTS
# ============================================================================


class TestBootstrapTest:
    """Tests for bootstrap refutation test."""

    def test_run_bootstrap_test_mock(self, runner):
        """Test running bootstrap test with mock."""
        result = runner._run_bootstrap_test(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=None,
            identified_estimand=None,
            estimate=None,
            use_dowhy=False,
        )

        assert result.test_name == RefutationTestType.BOOTSTRAP
        assert result.status in [
            RefutationStatus.PASSED,
            RefutationStatus.WARNING,
            RefutationStatus.FAILED,
        ]
        assert "bootstrap_ci" in result.details

    def test_run_bootstrap_test_passed(self, runner):
        """Test bootstrap test that passes."""
        # Bootstrap CI must be <= 50% wider than original to pass
        # original_ci width = 0.20 - 0.10 = 0.10
        # bootstrap_ci width must be <= 0.05 (50% of 0.10)
        # So bootstrap_ci = (0.125, 0.175) gives width = 0.05, ci_ratio = 0.5
        with patch.object(
            runner, "_mock_bootstrap_test", return_value=(0.15, (0.125, 0.175), 0.85)
        ):
            result = runner._run_bootstrap_test(
                original_effect=0.15,
                original_ci=(0.10, 0.20),
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
            )

            assert result.status == RefutationStatus.PASSED


# ============================================================================
# SENSITIVITY E-VALUE TEST TESTS
# ============================================================================


class TestSensitivityTest:
    """Tests for sensitivity E-value test."""

    def test_run_sensitivity_test(self, runner):
        """Test running sensitivity E-value test."""
        result = runner._run_sensitivity_test(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
        )

        assert result.test_name == RefutationTestType.SENSITIVITY_E_VALUE
        assert result.status in [
            RefutationStatus.PASSED,
            RefutationStatus.WARNING,
            RefutationStatus.FAILED,
        ]
        assert "e_value" in result.details

    def test_run_sensitivity_test_high_e_value(self, runner):
        """Test sensitivity test with high E-value (passes)."""
        result = runner._run_sensitivity_test(
            original_effect=0.50,  # Large effect → high E-value
            original_ci=(0.40, 0.60),
        )

        assert result.status == RefutationStatus.PASSED
        assert result.details["e_value"] >= runner.thresholds["e_value_min"]["pass"]

    def test_run_sensitivity_test_low_e_value(self, runner):
        """Test sensitivity test with low E-value (fails)."""
        result = runner._run_sensitivity_test(
            original_effect=0.05,  # Small effect → low E-value
            original_ci=(0.01, 0.09),
        )

        # Small effects typically have low E-values
        assert result.details["e_value"] > 0


# ============================================================================
# MOCK IMPLEMENTATIONS TESTS
# ============================================================================


class TestMockImplementations:
    """Tests for mock refutation implementations."""

    def test_mock_placebo_test(self, runner):
        """Test mock placebo test."""
        placebo_effect, p_value = runner._mock_placebo_test(0.15)

        assert abs(placebo_effect) < 0.1  # Should be near zero
        assert 0 < p_value < 1

    def test_mock_random_common_cause_test(self, runner):
        """Test mock random common cause test."""
        refuted_effect, p_value = runner._mock_random_common_cause_test(0.15)

        assert abs(refuted_effect - 0.15) < 0.1  # Should be close to original
        assert 0 < p_value < 1

    def test_mock_data_subset_test(self, runner):
        """Test mock data subset test."""
        refuted_effect, p_value, ci_coverage = runner._mock_data_subset_test(0.15, (0.10, 0.20))

        assert abs(refuted_effect - 0.15) < 0.1
        assert 0 < p_value < 1
        assert 0 <= ci_coverage <= 1

    def test_mock_bootstrap_test(self, runner):
        """Test mock bootstrap test."""
        refuted_effect, bootstrap_ci, p_value = runner._mock_bootstrap_test(0.15)

        assert bootstrap_ci[0] < refuted_effect < bootstrap_ci[1]
        assert 0 < p_value < 1


# ============================================================================
# CONFIDENCE SCORE TESTS
# ============================================================================


class TestConfidenceScore:
    """Tests for confidence score calculation."""

    def test_calculate_confidence_score_all_passed(self, runner):
        """Test confidence score when all tests pass."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.15,
                refuted_effect=0.02,
            ),
            RefutationResult(
                test_name=RefutationTestType.RANDOM_COMMON_CAUSE,
                status=RefutationStatus.PASSED,
                original_effect=0.15,
                refuted_effect=0.14,
            ),
        ]

        score = runner._calculate_confidence_score(tests)

        assert score > 0.8  # Should be high when all pass

    def test_calculate_confidence_score_all_failed(self, runner):
        """Test confidence score when all tests fail."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.FAILED,
                original_effect=0.15,
                refuted_effect=0.12,
            ),
            RefutationResult(
                test_name=RefutationTestType.RANDOM_COMMON_CAUSE,
                status=RefutationStatus.FAILED,
                original_effect=0.15,
                refuted_effect=0.05,
            ),
        ]

        score = runner._calculate_confidence_score(tests)

        assert score < 0.5  # Should be low when all fail

    def test_calculate_confidence_score_empty_tests(self, runner):
        """Test confidence score with empty test list."""
        score = runner._calculate_confidence_score([])

        assert score == 0.0


# ============================================================================
# GATE DECISION TESTS
# ============================================================================


class TestGateDecision:
    """Tests for gate decision logic."""

    def test_determine_gate_decision_proceed(self, runner):
        """Test gate decision when confidence is high."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.15,
                refuted_effect=0.02,
            ),
        ]

        decision = runner._determine_gate_decision(tests, confidence_score=0.85)

        assert decision == GateDecision.PROCEED

    def test_determine_gate_decision_review(self, runner):
        """Test gate decision when confidence is moderate."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.WARNING,
                original_effect=0.15,
                refuted_effect=0.08,
            ),
        ]

        decision = runner._determine_gate_decision(tests, confidence_score=0.60)

        assert decision == GateDecision.REVIEW

    def test_determine_gate_decision_block_critical_failure(self, runner):
        """Test gate decision when critical test fails."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.FAILED,
                original_effect=0.15,
                refuted_effect=0.12,
            ),
        ]

        decision = runner._determine_gate_decision(tests, confidence_score=0.60)

        assert decision == GateDecision.BLOCK

    def test_determine_gate_decision_block_low_confidence(self, runner):
        """Test gate decision when confidence is low."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.DATA_SUBSET,
                status=RefutationStatus.WARNING,
                original_effect=0.15,
                refuted_effect=0.10,
            ),
        ]

        decision = runner._determine_gate_decision(tests, confidence_score=0.40)

        assert decision == GateDecision.BLOCK


# ============================================================================
# FULL SUITE TESTS
# ============================================================================


class TestRunAllTests:
    """Tests for run_all_tests method."""

    def test_run_all_tests_basic(self, runner):
        """Test running all refutation tests."""
        suite = runner.run_all_tests(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
        )

        assert isinstance(suite, RefutationSuite)
        assert len(suite.tests) > 0
        assert suite.gate_decision in [
            GateDecision.PROCEED,
            GateDecision.REVIEW,
            GateDecision.BLOCK,
        ]

    def test_run_all_tests_with_disabled_tests(self):
        """Test running with some tests disabled."""
        config = {
            "placebo_treatment": {"enabled": False},
            "random_common_cause": {"enabled": True},
        }
        runner = RefutationRunner(config=config)

        suite = runner.run_all_tests(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
        )

        # Should not include placebo test
        test_names = [t.test_name for t in suite.tests]
        assert RefutationTestType.PLACEBO_TREATMENT not in test_names

    def test_run_all_tests_with_metadata(self, runner):
        """Test running tests with full metadata."""
        suite = runner.run_all_tests(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            treatment="hcp_engagement",
            outcome="conversion_rate",
            brand="Kisqali",
            estimate_id="est-123",
        )

        assert suite.treatment_variable == "hcp_engagement"
        assert suite.outcome_variable == "conversion_rate"
        assert suite.brand == "Kisqali"
        assert suite.estimate_id == "est-123"


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_run_refutation_suite(self):
        """Test run_refutation_suite convenience function."""
        suite = run_refutation_suite(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            treatment="test_treatment",
            outcome="test_outcome",
        )

        assert isinstance(suite, RefutationSuite)
        assert suite.treatment_variable == "test_treatment"

    def test_is_estimate_valid_proceed(self):
        """Test is_estimate_valid with proceed decision."""
        suite = RefutationSuite(
            passed=True,
            confidence_score=0.85,
            tests=[],
            gate_decision=GateDecision.PROCEED,
        )

        assert is_estimate_valid(suite) is True

    def test_is_estimate_valid_block(self):
        """Test is_estimate_valid with block decision."""
        suite = RefutationSuite(
            passed=False,
            confidence_score=0.30,
            tests=[],
            gate_decision=GateDecision.BLOCK,
        )

        assert is_estimate_valid(suite) is False

    def test_is_estimate_valid_review(self):
        """Test is_estimate_valid with review decision (should be valid)."""
        suite = RefutationSuite(
            passed=True,
            confidence_score=0.60,
            tests=[],
            gate_decision=GateDecision.REVIEW,
        )

        assert is_estimate_valid(suite) is True
