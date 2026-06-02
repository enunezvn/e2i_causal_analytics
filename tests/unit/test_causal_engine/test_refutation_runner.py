"""
Unit tests for refutation_runner.py

Tests cover:
- RefutationRunner
- RefutationResult
- RefutationSuite
- Individual refutation tests (placebo, random_common_cause, data_subset, bootstrap, sensitivity)
- Fail-closed semantics for DoWhy-unavailable / causal_model=None
- Scoring and gate decisions

F-014 fix (#416): The previous ``_mock_*`` mock paths have been deleted.
Tests that previously patched them now (a) provide a stub CausalModel that
returns deterministic refutation results, or (b) assert that calling
``_run_*_test`` with ``causal_model=None`` raises ``RefutationError``.
"""

from types import SimpleNamespace

import pytest

from src.causal_engine.errors import RefutationError
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
# CAUSAL MODEL STUBS (replace deleted _mock_* methods)
# These produce deterministic refutation values to keep the existing pass/fail
# assertions stable, but go through the real `causal_model.refute_estimate`
# API surface — i.e., the code path that production now exercises.
# ============================================================================


def _make_refutation_result(new_effect: float, p_value: float, **extra) -> SimpleNamespace:
    """Construct a stub object shaped like DoWhy's refutation result."""
    rr: dict = {"p_value": p_value}
    rr.update(extra)
    return SimpleNamespace(new_effect=new_effect, refutation_result=rr)


def _make_stub_causal_model(refutation_results_by_method: dict) -> SimpleNamespace:
    """Construct a stub object shaped like DoWhy's CausalModel.

    Only ``refute_estimate(estimand, estimate, method_name=..., **kwargs)`` is
    implemented; it returns the canned result for ``method_name``.
    """

    def refute_estimate(*_args, method_name: str, **_kwargs):  # noqa: ANN001
        if method_name not in refutation_results_by_method:
            raise KeyError(f"stub did not register method_name={method_name!r}")
        return refutation_results_by_method[method_name]

    return SimpleNamespace(refute_estimate=refute_estimate)


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
# #622 PROD-LATENCY DEFAULT_CONFIG TESTS
# ============================================================================


class TestDefaultConfigLatencyBounds:
    """#622: DEFAULT_CONFIG sim counts must be bounded for prod latency.

    The previous defaults (placebo 100, bootstrap 500, and an UNBOUNDED
    random_common_cause that fell through to DoWhy's internal default of 100)
    made the suite ~610 DoWhy re-estimations -> ~33s (OLS) to ~35-60 min
    (causal_forest). MEASURED on the synthetic fixture (#622). These tests pin
    the lowered, defensible defaults so a future bump back to the slow values
    is caught.
    """

    def test_placebo_default_simulations_bounded(self):
        runner = RefutationRunner()
        assert runner.config["placebo_treatment"]["num_simulations"] == 30

    def test_random_common_cause_has_bounded_num_simulations(self):
        """The KEY fix: random_common_cause previously had NO num_simulations
        key, so the runner never passed one and DoWhy used its internal 100
        (~140s, the issue's named dominant cost). It must now be present and
        bounded so it is actually passed to the refuter."""
        runner = RefutationRunner()
        assert "num_simulations" in runner.config["random_common_cause"]
        assert runner.config["random_common_cause"]["num_simulations"] == 20

    def test_bootstrap_default_bootstraps_bounded(self):
        runner = RefutationRunner()
        assert runner.config["bootstrap"]["num_bootstraps"] == 50

    def test_data_subset_default_subsets_bounded(self):
        runner = RefutationRunner()
        assert runner.config["data_subset"]["num_subsets"] == 5

    def test_random_common_cause_passes_num_simulations_to_refuter(self):
        """The bounded num_simulations must actually reach DoWhy's refuter.

        Pre-#622 the runner only forwarded num_simulations when present in
        config; since it was absent from DEFAULT_CONFIG, the kwarg was never
        sent and DoWhy ran its own 100-sim default. With it present, the runner
        must forward it. We capture the kwargs the stub receives."""
        captured: dict = {}

        def refute_estimate(*_args, method_name: str, **kwargs):  # noqa: ANN001
            captured["method_name"] = method_name
            captured["kwargs"] = kwargs
            return _make_refutation_result(new_effect=0.14, p_value=0.5)

        stub_model = SimpleNamespace(refute_estimate=refute_estimate)
        runner = RefutationRunner()
        runner._run_random_common_cause_test(
            original_effect=0.15,
            causal_model=stub_model,
            identified_estimand=object(),
            estimate=object(),
            use_dowhy=True,
        )
        assert captured["method_name"] == "random_common_cause"
        assert captured["kwargs"].get("num_simulations") == 20

    def test_custom_config_still_overrides_lowered_defaults(self):
        """The #606 smoke-harness override path (per-key merge) must still win
        over the new lowered defaults — they are merged on top, not replaced."""
        runner = RefutationRunner(
            config={
                "random_common_cause": {"num_simulations": 5},
                "bootstrap": {"num_bootstraps": 10},
            }
        )
        assert runner.config["random_common_cause"]["num_simulations"] == 5
        assert runner.config["bootstrap"]["num_bootstraps"] == 10
        # Untouched keys retain the new lowered defaults.
        assert runner.config["placebo_treatment"]["num_simulations"] == 30
        # Effect strength preserved across the merge.
        assert runner.config["random_common_cause"]["effect_strength"] == 0.1


# ============================================================================
# PLACEBO TEST TESTS
# ============================================================================


class TestPlaceboTest:
    """Tests for placebo treatment refutation test."""

    def test_run_placebo_test_no_model_fails_closed(self, runner):
        """F-014 (#416): placebo test with causal_model=None must raise RefutationError,
        not silently dispatch to a mock path.
        """
        with pytest.raises(RefutationError) as exc_info:
            runner._run_placebo_test(
                original_effect=0.15,
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
            )
        assert "placebo" in exc_info.value.details.get("test_name", "").lower()
        assert "unavailable" in str(exc_info.value).lower()

    def test_run_placebo_test_passed(self, runner):
        """Test placebo test that passes (via stub CausalModel)."""
        stub_model = _make_stub_causal_model(
            {"placebo_treatment_refuter": _make_refutation_result(new_effect=0.01, p_value=0.85)}
        )
        result = runner._run_placebo_test(
            original_effect=0.15,
            causal_model=stub_model,
            identified_estimand=object(),
            estimate=object(),
            use_dowhy=True,
        )

        assert result.status == RefutationStatus.PASSED
        assert "no significant effect" in result.details["message"].lower()

    def test_run_placebo_test_failed(self, runner):
        """Test placebo test that fails (via stub CausalModel)."""
        stub_model = _make_stub_causal_model(
            {"placebo_treatment_refuter": _make_refutation_result(new_effect=0.12, p_value=0.02)}
        )
        result = runner._run_placebo_test(
            original_effect=0.15,
            causal_model=stub_model,
            identified_estimand=object(),
            estimate=object(),
            use_dowhy=True,
        )

        assert result.status == RefutationStatus.FAILED
        assert "warning" in result.details["message"].lower()


# ============================================================================
# RANDOM COMMON CAUSE TEST TESTS
# ============================================================================


class TestRandomCommonCauseTest:
    """Tests for random common cause refutation test."""

    def test_run_random_common_cause_test_no_model_fails_closed(self, runner):
        """F-014 (#416): random_common_cause test with causal_model=None must
        raise RefutationError, not silently dispatch to a mock path.
        """
        with pytest.raises(RefutationError) as exc_info:
            runner._run_random_common_cause_test(
                original_effect=0.15,
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
            )
        assert exc_info.value.details.get("test_name") == "random_common_cause"

    def test_run_random_common_cause_test_passed(self, runner):
        """Test random common cause test that passes (via stub CausalModel)."""
        stub_model = _make_stub_causal_model(
            {"random_common_cause": _make_refutation_result(new_effect=0.14, p_value=0.70)}
        )
        result = runner._run_random_common_cause_test(
            original_effect=0.15,
            causal_model=stub_model,
            identified_estimand=object(),
            estimate=object(),
            use_dowhy=True,
        )

        assert result.status == RefutationStatus.PASSED

    def test_run_random_common_cause_test_failed(self, runner):
        """Test random common cause test that fails (via stub CausalModel)."""
        stub_model = _make_stub_causal_model(
            {"random_common_cause": _make_refutation_result(new_effect=0.05, p_value=0.60)}
        )
        result = runner._run_random_common_cause_test(
            original_effect=0.15,
            causal_model=stub_model,
            identified_estimand=object(),
            estimate=object(),
            use_dowhy=True,
        )

        # Large delta should trigger warning or failure
        assert result.status in [RefutationStatus.WARNING, RefutationStatus.FAILED]


# ============================================================================
# DATA SUBSET TEST TESTS
# ============================================================================


class TestDataSubsetTest:
    """Tests for data subset refutation test."""

    def test_run_data_subset_test_no_model_fails_closed(self, runner):
        """F-014 (#416): data_subset test with causal_model=None must raise
        RefutationError, not silently dispatch to a mock path.
        """
        with pytest.raises(RefutationError) as exc_info:
            runner._run_data_subset_test(
                original_effect=0.15,
                original_ci=(0.10, 0.20),
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
            )
        assert exc_info.value.details.get("test_name") == "data_subset"

    def test_run_data_subset_test_passed(self, runner):
        """Test data subset test that passes (via stub CausalModel)."""
        # subset_effects span keeps within original_ci so CI coverage is 1.0 (pass).
        stub_model = _make_stub_causal_model(
            {
                "data_subset_refuter": _make_refutation_result(
                    new_effect=0.15,
                    p_value=0.75,
                    subset_effects=[0.13, 0.14, 0.15, 0.16, 0.17],
                )
            }
        )
        result = runner._run_data_subset_test(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=stub_model,
            identified_estimand=object(),
            estimate=object(),
            use_dowhy=True,
        )

        assert result.status == RefutationStatus.PASSED


# ============================================================================
# BOOTSTRAP TEST TESTS
# ============================================================================


class TestBootstrapTest:
    """Tests for bootstrap refutation test."""

    def test_run_bootstrap_test_no_model_fails_closed(self, runner):
        """F-014 (#416): bootstrap test with causal_model=None must raise
        RefutationError, not silently dispatch to a mock path.
        """
        with pytest.raises(RefutationError) as exc_info:
            runner._run_bootstrap_test(
                original_effect=0.15,
                original_ci=(0.10, 0.20),
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
            )
        assert exc_info.value.details.get("test_name") == "bootstrap"

    def test_run_bootstrap_test_passed(self, runner):
        """Test bootstrap test that passes (via stub CausalModel).

        Bootstrap CI must be <= 50% wider than original to pass.
        - original_ci width = 0.20 - 0.10 = 0.10
        - bootstrap_ci width must be <= 0.05 (50% of 0.10)
        - Bootstrap effects centered at 0.15 with width 0.05 → ratio = 0.5.
        """
        # Provide bootstrap_estimates whose mean ≈ 0.15 and whose 2.5/97.5
        # percentiles fall around (0.125, 0.175).
        stub_model = _make_stub_causal_model(
            {
                "bootstrap_refuter": _make_refutation_result(
                    new_effect=0.15,
                    p_value=0.85,
                    bootstrap_estimates=[0.125, 0.13, 0.14, 0.15, 0.16, 0.17, 0.175],
                )
            }
        )
        result = runner._run_bootstrap_test(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=stub_model,
            identified_estimand=object(),
            estimate=object(),
            use_dowhy=True,
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


class TestMockImplementationsDeleted:
    """F-014 (#416): assert the legacy ``_mock_*`` methods are gone.

    These pins make a future re-introduction of any silent-fallback mock
    method a CI failure (per ``CLAUDE.md`` §"CRITICAL — Anti-Mocking &
    Verification Discipline" and memory ``feedback-no-mocking-no-patching``).
    """

    def test_mock_placebo_test_method_deleted(self, runner):
        """``_mock_placebo_test`` must NOT exist on the runner."""
        assert not hasattr(runner, "_mock_placebo_test"), (
            "F-014 regression: _mock_placebo_test re-introduced. "
            "Use real DoWhy CausalModel or fail-closed with RefutationError."
        )

    def test_mock_random_common_cause_test_method_deleted(self, runner):
        """``_mock_random_common_cause_test`` must NOT exist on the runner."""
        assert not hasattr(runner, "_mock_random_common_cause_test"), (
            "F-014 regression: _mock_random_common_cause_test re-introduced. "
            "Use real DoWhy CausalModel or fail-closed with RefutationError."
        )

    def test_mock_data_subset_test_method_deleted(self, runner):
        """``_mock_data_subset_test`` must NOT exist on the runner."""
        assert not hasattr(runner, "_mock_data_subset_test"), (
            "F-014 regression: _mock_data_subset_test re-introduced. "
            "Use real DoWhy CausalModel or fail-closed with RefutationError."
        )

    def test_mock_bootstrap_test_method_deleted(self, runner):
        """``_mock_bootstrap_test`` must NOT exist on the runner."""
        assert not hasattr(runner, "_mock_bootstrap_test"), (
            "F-014 regression: _mock_bootstrap_test re-introduced. "
            "Use real DoWhy CausalModel or fail-closed with RefutationError."
        )


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


def _full_stub_causal_model() -> "SimpleNamespace":  # noqa: UP037
    """Build a stub CausalModel that returns canned results for all 4 refuters."""
    return _make_stub_causal_model(
        {
            "placebo_treatment_refuter": _make_refutation_result(new_effect=0.01, p_value=0.85),
            "random_common_cause": _make_refutation_result(new_effect=0.14, p_value=0.70),
            "data_subset_refuter": _make_refutation_result(
                new_effect=0.15,
                p_value=0.75,
                subset_effects=[0.13, 0.14, 0.15, 0.16, 0.17],
            ),
            "bootstrap_refuter": _make_refutation_result(
                new_effect=0.15,
                p_value=0.85,
                bootstrap_estimates=[0.125, 0.13, 0.14, 0.15, 0.16, 0.17, 0.175],
            ),
        }
    )


class TestRunAllTests:
    """Tests for run_all_tests method.

    F-014 (#416): These tests now provide a stub CausalModel because
    ``run_all_tests`` no longer silently dispatches to mock paths when the
    model is missing. The stub keeps the test assertions stable while
    exercising the real ``causal_model.refute_estimate`` API.
    """

    def test_run_all_tests_basic(self, runner):
        """Test running all refutation tests with stub CausalModel."""
        suite = runner.run_all_tests(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=object(),
        )

        assert isinstance(suite, RefutationSuite)
        assert len(suite.tests) > 0
        assert suite.gate_decision in [
            GateDecision.PROCEED,
            GateDecision.REVIEW,
            GateDecision.BLOCK,
        ]

    def test_run_all_tests_with_disabled_tests(self):
        """Test running with some tests disabled (with stub CausalModel)."""
        config = {
            "placebo_treatment": {"enabled": False},
            "random_common_cause": {"enabled": True},
        }
        runner = RefutationRunner(config=config)

        suite = runner.run_all_tests(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=object(),
        )

        # Should not include placebo test
        test_names = [t.test_name for t in suite.tests]
        assert RefutationTestType.PLACEBO_TREATMENT not in test_names

    def test_run_all_tests_with_metadata(self, runner):
        """Test running tests with full metadata (with stub CausalModel)."""
        suite = runner.run_all_tests(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=object(),
            treatment="hcp_engagement",
            outcome="conversion_rate",
            brand="Kisqali",
            estimate_id="est-123",
        )

        assert suite.treatment_variable == "hcp_engagement"
        assert suite.outcome_variable == "conversion_rate"
        assert suite.brand == "Kisqali"
        assert suite.estimate_id == "est-123"

    def test_run_all_tests_without_causal_model_fails_closed(self, runner):
        """F-014 (#416): run_all_tests without a CausalModel must fail-closed.

        The first enabled test (placebo by default) raises RefutationError;
        no mock fallback exists.
        """
        with pytest.raises(RefutationError):
            runner.run_all_tests(
                original_effect=0.15,
                original_ci=(0.10, 0.20),
            )


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_run_refutation_suite_with_model(self):
        """F-014 iter-2 (#416, codex H5): run_refutation_suite accepts model
        artifacts as keyword-only args, so external callers can use it with
        their own DoWhy model. Previously the signature did not accept
        ``causal_model`` / ``identified_estimand`` / ``estimate`` which made
        every call fail-closed — that codified a broken public API. Now the
        function is functionally usable.
        """
        suite = run_refutation_suite(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=object(),
            treatment="test_treatment",
            outcome="test_outcome",
        )
        assert isinstance(suite, RefutationSuite)
        assert suite.treatment_variable == "test_treatment"

    def test_run_refutation_suite_missing_model_args_refutation_error(self):
        """Iter-4 codex H3 (#416): legacy positional signature is preserved,
        so callers that don't pass model artifacts get a clear
        ``RefutationError`` (NOT ``TypeError``) instead of crashing on a
        keyword-only signature change. This keeps the iter-0 public contract
        usable while still rejecting the silent-mock dispatch.
        """
        with pytest.raises(RefutationError):
            run_refutation_suite(
                original_effect=0.15,
                original_ci=(0.10, 0.20),
                treatment="test_treatment",
                outcome="test_outcome",
            )

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
