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

import json
from types import SimpleNamespace
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.causal_engine import evalue
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


class _StubRefitEstimator:
    """What ``estimate.estimator.get_new_estimator_object`` returns (spec §4.1).

    ``fit`` records the resample and ``estimate_effect`` reports an effect
    computed FROM it, so subset and bootstrap draws vary the way a real re-fit
    would (a constant series would make DoWhy's normal test divide by zero).
    """

    def __init__(self, effect_fn: Callable[[pd.DataFrame], float]) -> None:
        self._effect_fn = effect_fn
        self.fitted_on: Optional[pd.DataFrame] = None

    def fit(self, data: pd.DataFrame, effect_modifier_names=None, **_fit_params) -> None:  # noqa: ANN001
        self.fitted_on = data

    def estimate_effect(  # noqa: ANN001
        self, data: pd.DataFrame, control_value=0, treatment_value=1, target_units="ate"
    ) -> SimpleNamespace:
        return SimpleNamespace(value=float(self._effect_fn(data)))


def _stub_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """The frame a stub CausalModel was 'built on' (DoWhy keeps it as ``_data``)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"t": rng.integers(0, 2, n), "y": rng.random(n), "c": rng.random(n)})


def _stub_estimate(
    value: float = 0.15,
    effect_fn: Optional[Callable[[pd.DataFrame], float]] = None,
) -> SimpleNamespace:
    """A DoWhy-shaped CausalEstimate: ``.value`` plus the four estimator
    attributes ``refutation_runner._refit_effect_on`` reads."""
    fn = effect_fn or (lambda df: value + 0.01 * (float(df["y"].mean()) - 0.5))
    estimator = SimpleNamespace(
        get_new_estimator_object=lambda _estimand: _StubRefitEstimator(fn),
        _effect_modifier_names=[],
        _target_units="ate",
    )
    return SimpleNamespace(value=value, estimator=estimator, control_value=0, treatment_value=1)


def _sequence_estimate(values: List[float], value: float = 0.15) -> SimpleNamespace:
    """An estimate whose successive re-fits report ``values`` in order."""
    it = iter(values)
    return _stub_estimate(value=value, effect_fn=lambda _df: next(it))


def _make_stub_causal_model(
    refutation_results_by_method: dict, data: Optional[pd.DataFrame] = None
) -> SimpleNamespace:
    """Construct a stub object shaped like DoWhy's CausalModel.

    ``refute_estimate(estimand, estimate, method_name=..., **kwargs)`` returns
    the canned result for ``method_name`` (placebo / random_common_cause still
    go through it); ``_data`` is the frame the two resample loops draw from.
    """

    def refute_estimate(*_args, method_name: str, **_kwargs):  # noqa: ANN001
        if method_name not in refutation_results_by_method:
            raise KeyError(f"stub did not register method_name={method_name!r}")
        return refutation_results_by_method[method_name]

    return SimpleNamespace(
        refute_estimate=refute_estimate,
        _data=data if data is not None else _stub_frame(),
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
# LEGACY FORMAT skipped_tests OBSERVABILITY FIELD (#1249)
# ============================================================================


class TestLegacyFormatSkippedTestsField:
    """#1249: ``to_legacy_format()`` exposes skip *reasons* via an opt-in
    ``skipped_tests: {contract_key: reason}`` field.

    Since #1219 SKIPPED tests are omitted from ``individual_tests`` /
    ``total_tests`` (correct: the pass/fail-shaped dict would render them as
    fake FAILED rows). But that omission destroyed the skip reason for every
    downstream consumer (audit chain, executor) — "skipped because X" became
    indistinguishable from "never attempted". The new field carries the
    reason without re-entering the pass/fail surface.
    """

    def _passed(self, *names):
        return [
            RefutationResult(
                test_name=name,
                status=RefutationStatus.PASSED,
                original_effect=0.08,
                refuted_effect=0.01,
            )
            for name in names
        ]

    def _skipped(self, name, message=None):
        details = {"message": message} if message is not None else {}
        return RefutationResult(
            test_name=name,
            status=RefutationStatus.SKIPPED,
            original_effect=0.08,
            refuted_effect=0.08,
            details=details,
        )

    def _suite(self, tests):
        return RefutationSuite(
            passed=True,
            confidence_score=1.0,
            tests=tests,
            gate_decision=GateDecision.PROCEED,
        )

    def test_skipped_tests_carries_reason_for_randomized_evalue(self):
        """The #1219 randomized-design skip surfaces its reason under the
        contract key (sensitivity_e_value → unobserved_common_cause)."""
        reason = (
            "not applicable: randomized design — treatment assignment is exogenous by construction"
        )
        tests = self._passed(
            RefutationTestType.PLACEBO_TREATMENT,
            RefutationTestType.RANDOM_COMMON_CAUSE,
            RefutationTestType.DATA_SUBSET,
            RefutationTestType.BOOTSTRAP,
        ) + [self._skipped(RefutationTestType.SENSITIVITY_E_VALUE, reason)]

        legacy = self._suite(tests).to_legacy_format()

        assert legacy["skipped_tests"] == {"unobserved_common_cause": reason}

    def test_skipped_tests_empty_dict_when_nothing_skipped(self):
        """Always-present key: consumers read {} instead of KeyError."""
        tests = self._passed(
            RefutationTestType.PLACEBO_TREATMENT,
            RefutationTestType.RANDOM_COMMON_CAUSE,
        )

        legacy = self._suite(tests).to_legacy_format()

        assert legacy["skipped_tests"] == {}

    def test_skipped_tests_uses_contract_keys_for_all_skip_sites(self):
        """All three real skip sites map through the same contract key space
        as individual_tests (data_subset, bootstrap, unobserved_common_cause)."""
        tests = self._passed(RefutationTestType.PLACEBO_TREATMENT) + [
            self._skipped(
                RefutationTestType.DATA_SUBSET,
                "DoWhy refutation_result did not expose 'subset_effects'",
            ),
            self._skipped(
                RefutationTestType.BOOTSTRAP,
                "DoWhy refutation_result did not expose 'bootstrap_estimates'",
            ),
            self._skipped(
                RefutationTestType.SENSITIVITY_E_VALUE,
                "not applicable: randomized design",
            ),
        ]

        legacy = self._suite(tests).to_legacy_format()

        assert set(legacy["skipped_tests"]) == {
            "data_subset",
            "bootstrap",
            "unobserved_common_cause",
        }

    def test_skipped_tests_reason_defaults_to_empty_string(self):
        """A SKIPPED result without a details message still records the skip
        (empty reason) rather than being dropped."""
        tests = [self._skipped(RefutationTestType.DATA_SUBSET)]

        legacy = self._suite(tests).to_legacy_format()

        assert legacy["skipped_tests"] == {"data_subset": ""}

    def test_skipped_entries_stay_out_of_individual_tests_and_totals(self):
        """#1219 invariant preserved: the new field must NOT reintroduce
        fake-FAILED rendering — skipped entries stay out of individual_tests
        and total_tests."""
        tests = self._passed(
            RefutationTestType.PLACEBO_TREATMENT,
            RefutationTestType.RANDOM_COMMON_CAUSE,
        ) + [self._skipped(RefutationTestType.SENSITIVITY_E_VALUE, "not applicable")]

        legacy = self._suite(tests).to_legacy_format()

        assert "unobserved_common_cause" not in legacy["individual_tests"]
        assert legacy["total_tests"] == 2
        assert legacy["tests_passed"] == 2
        assert legacy["skipped_tests"] == {"unobserved_common_cause": "not applicable"}


# ============================================================================
# LEGACY FORMAT THREE-STATE status FIELD (#1867)
# ============================================================================


class TestLegacyFormatStatusField:
    """#1867: each ``individual_tests`` entry carries the engine's three-state
    ``status`` alongside the legacy two-state ``passed``.

    The pass/fail-shaped dict collapsed WARNING to ``passed: False``, so the FE
    rendered a soft warning (e.g. E-value 1.51 in the [1.5, 2.0) warning band)
    identically to a hard failure — contradicting the PROCEED gate shown next
    to it. ``passed`` stays for backward compatibility; ``status`` is the
    honest signal.
    """

    def _suite(self, tests):
        return RefutationSuite(
            passed=True,
            confidence_score=0.85,
            tests=tests,
            gate_decision=GateDecision.PROCEED,
        )

    def test_warning_entry_carries_status_warning_and_passed_false(self):
        tests = [
            RefutationResult(
                test_name=RefutationTestType.SENSITIVITY_E_VALUE,
                status=RefutationStatus.WARNING,
                original_effect=0.075,
                refuted_effect=0.075,
                details={"message": "E-value (CI bound) 1.51 suggests moderate sensitivity"},
            )
        ]
        legacy = self._suite(tests).to_legacy_format()
        entry = legacy["individual_tests"]["unobserved_common_cause"]
        assert entry["status"] == "warning"
        assert entry["passed"] is False  # legacy two-state unchanged

    def test_passed_and_failed_entries_carry_matching_status(self):
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.075,
                refuted_effect=0.001,
            ),
            RefutationResult(
                test_name=RefutationTestType.DATA_SUBSET,
                status=RefutationStatus.FAILED,
                original_effect=0.075,
                refuted_effect=0.02,
            ),
        ]
        legacy = self._suite(tests).to_legacy_format()
        assert legacy["individual_tests"]["placebo_treatment"]["status"] == "passed"
        assert legacy["individual_tests"]["placebo_treatment"]["passed"] is True
        assert legacy["individual_tests"]["data_subset"]["status"] == "failed"
        assert legacy["individual_tests"]["data_subset"]["passed"] is False


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
            estimate=_stub_estimate(),
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
            estimate=_stub_estimate(),
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
            estimate=_stub_estimate(),
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
            estimate=_stub_estimate(),
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
            estimate=_stub_estimate(),
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
        """Real evidence (spec §4.1): the test re-fits on subsets of the model's
        frame and scores how many subset effects fall inside original_ci."""
        result = runner._run_data_subset_test(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=_make_stub_causal_model({}),
            identified_estimand=object(),
            estimate=_stub_estimate(),
            use_dowhy=True,
        )

        assert result.status == RefutationStatus.PASSED
        assert len(result.details["subset_effects"]) == 5
        assert result.details["ci_coverage"] == 1.0


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
        """Real evidence (spec §4.1): bootstrap re-fits on row resamples; the
        2.5–97.5 percentile width is compared with original_ci under the
        INTENDED thresholds (pass ≤ 1.5× the original width)."""
        result = runner._run_bootstrap_test(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=_make_stub_causal_model({}),
            identified_estimand=object(),
            estimate=_stub_estimate(),
            use_dowhy=True,
        )

        assert result.status == RefutationStatus.PASSED
        assert (
            len(result.details["bootstrap_effects"]) == runner.config["bootstrap"]["num_bootstraps"]
        )
        assert result.details["ci_ratio"] <= 1.5


# ============================================================================
# SENSITIVITY E-VALUE TEST TESTS
# ============================================================================


class TestSensitivityTest:
    """The sensitivity test is a READING, never a gate (spec §4.4, §4.5)."""

    def test_sensitivity_is_not_critical_and_has_no_threshold(self, runner):
        assert runner.config["sensitivity_e_value"]["critical"] is False
        assert "e_value_threshold" not in runner.config["sensitivity_e_value"]
        assert "e_value_min" not in runner.thresholds

    def test_beyond_reads_passed_with_the_benchmark_in_details(self, runner):
        result = runner._run_sensitivity_test(
            original_effect=0.15,
            original_ci=(0.08, 0.22),
            baseline_risk=0.30,
            naive_effect=0.20,
        )
        assert result.test_name == RefutationTestType.SENSITIVITY_E_VALUE
        assert result.status == RefutationStatus.PASSED
        d = result.details
        assert d["reading"] == "beyond_measured_confounding"
        assert d["benchmark_basis"] == "joint_naive_vs_adjusted"
        assert d["rr_point"] == pytest.approx(1.5)
        assert d["headline"] == "Robust to confounding at measured strength"
        assert "stronger than all measured confounding" in d["message"]
        assert d["e_value"] == pytest.approx(d["e_value_point"])  # legacy key kept
        # the SERVED path must be persistable too, not only the error paths
        json.dumps(d, allow_nan=False)

    def test_within_reads_warning(self, runner):
        result = runner._run_sensitivity_test(
            original_effect=0.03,
            original_ci=(0.01, 0.05),
            baseline_risk=0.30,
            naive_effect=0.10,
        )
        assert result.status == RefutationStatus.WARNING
        assert result.details["reading"] == "within_measured_confounding"

    def test_null_crossing_ci_is_a_null_finding_not_a_failure(self, runner):
        """Replaces M-stat2: a strong point effect with a null-crossing CI is served
        as a null finding (WARNING), never FAILED, and e_value_ci collapses to 1.0."""
        result = runner._run_sensitivity_test(
            original_effect=0.5,
            original_ci=(-0.3, 0.5),
            baseline_risk=0.30,
            naive_effect=0.6,
        )
        assert result.status == RefutationStatus.WARNING
        assert result.details["reading"] == "null_finding"
        assert result.details["e_value_ci"] == 1.0
        assert result.details["e_value"] > 1.0  # point value still surfaced
        assert "includes zero" in result.details["message"]

    def test_smd_path_when_no_baseline_risk(self, runner):
        import numpy as np

        result = runner._run_sensitivity_test(
            original_effect=0.5, original_ci=(0.4, 0.6), outcome_std=1.0, naive_effect=0.55
        )
        rr_ci = np.exp(0.91 * 0.4)
        assert result.details["conversion"] == "standardized_difference"
        assert result.details["e_value_ci"] == pytest.approx(
            rr_ci + np.sqrt(rr_ci * (rr_ci - 1)), rel=1e-6
        )

    @pytest.mark.parametrize("bad_sd", [0.0, -1.0, float("nan"), float("inf")])
    def test_an_unusable_sd_fails_closed_rather_than_standardizing_by_nothing(self, runner, bad_sd):
        """H3: a PRESENT but unusable SD (constant outcome, negative, NaN, inf) is a
        failure, not a missing input. Sanitizing it to None would send the classifier
        down the SMD path on the UNSTANDARDIZED effect — that changes the NUMBER a
        leader reads, not a label — so it fails closed like every other unusable
        sensitivity input (spec §5). ``None`` remains the served missing-input case."""
        with pytest.raises(RefutationError) as ei:
            runner._run_sensitivity_test(
                original_effect=0.5, original_ci=(0.4, 0.6), outcome_std=bad_sd
            )
        assert ei.value.details["reason"] == "sensitivity_outcome_std_unusable"
        assert ei.value.details["outcome_std"] == repr(bad_sd)
        json.dumps(ei.value.details, allow_nan=False)

    def test_a_numpy_sd_is_served_as_a_native_float(self, runner):
        """``np.float64`` subclasses ``float`` but ``np.float32`` does not, so a
        finite positive numpy SD passes validation and then makes the whole served
        details dict unserializable. Normalize once, at the boundary."""
        result = runner._run_sensitivity_test(
            original_effect=0.5, original_ci=(0.4, 0.6), outcome_std=np.float32(1.0)
        )
        assert type(result.details["outcome_std"]) is float
        json.dumps(result.details, allow_nan=False)

    def test_an_absent_sd_is_still_a_served_unstandardized_reading(self, runner):
        """``None`` means no SD was available; the reading is served on the raw
        effect with the flag honestly False."""
        result = runner._run_sensitivity_test(
            original_effect=0.5, original_ci=(0.4, 0.6), outcome_std=None
        )
        assert result.details["standardized"] is False
        assert result.details["outcome_std"] is None

    def test_a_usable_sd_reports_standardized(self, runner):
        result = runner._run_sensitivity_test(
            original_effect=0.5, original_ci=(0.4, 0.6), outcome_std=1.0
        )
        assert result.details["standardized"] is True
        assert result.details["outcome_std"] == 1.0

    def test_unbenchmarked_without_naive_or_covariates(self, runner):
        result = runner._run_sensitivity_test(original_effect=0.15, original_ci=(0.08, 0.22))
        assert result.status == RefutationStatus.WARNING
        assert result.details["reading"] == "unbenchmarked"

    def test_a_ci_that_excludes_the_estimate_raises_refutation_error(self, runner):
        """``evalue.classify`` refuses an out-of-domain input with ValueError. The
        runner must surface it the way every refit test does — as a structured
        RefutationError — not leak a raw ValueError past ``_run_test_with_tracing``
        (spec §5: a failed sensitivity computation fails closed, it is never served
        as a reading)."""
        with pytest.raises(RefutationError) as ei:
            runner._run_sensitivity_test(
                original_effect=0.1, original_ci=(0.8, 0.9), baseline_risk=0.3
            )
        assert ei.value.details["reason"] == "sensitivity_reading_failed"
        assert ei.value.details["original_effect"] == repr(0.1)
        assert ei.value.details["original_ci"] == [repr(0.8), repr(0.9)]
        json.dumps(ei.value.details, allow_nan=False)

    def test_reading_failure_details_stay_json_persistable(self, runner):
        """The inputs ``classify`` rejects are precisely the non-finite ones, so raw
        floats in ``details`` could be NaN/inf — which a JSONB writer rejects, losing
        the very error record that explains the failure. Diagnostics go in as
        strings."""
        with pytest.raises(RefutationError) as ei:
            runner._run_sensitivity_test(
                original_effect=float("nan"), original_ci=(0.1, 0.2), baseline_risk=0.3
            )
        assert ei.value.details["reason"] == "sensitivity_reading_failed"
        json.dumps(ei.value.details, allow_nan=False)

    def test_benchmark_computation_failure_fails_closed_not_unbenchmarked(self):
        """A covariate that perfectly separates treatment AND outcome is a positivity
        violation and ``benchmark_inputs_from_frame`` raises for it. Swallowing that
        into ``unbenchmarked`` would print "no measured confounders exist for this
        design" — a fabricated statement about the data. It must fail closed."""
        r = RefutationRunner(
            config={
                "placebo_treatment": {"enabled": False},
                "random_common_cause": {"enabled": False},
                "data_subset": {"enabled": False},
                "bootstrap": {"enabled": False},
            }
        )
        # treated units all c=0 (so the treated share of high-c is 0) while controls
        # split on c, and among controls every high-c unit has y=1 and every low-c
        # unit y=0: both Ding-VanderWeele limits fire at once, so the bias factor
        # diverges rather than settling.
        frame = pd.DataFrame(
            {
                "t": [1, 1, 1, 1, 0, 0, 0, 0],
                "c": [0, 0, 0, 0, 1, 1, 0, 0],
                "y": [1, 0, 1, 0, 1, 1, 0, 0],
            }
        )
        model = SimpleNamespace(get_common_causes=lambda: ["c"])
        with pytest.raises(RefutationError) as ei:
            r.run_all_tests(
                original_effect=0.15,
                original_ci=(0.08, 0.22),
                data=frame,
                treatment="t",
                outcome="y",
                causal_model=model,
                identified_estimand=object(),
                estimate=_stub_estimate(),
            )
        assert ei.value.details["reason"] == "sensitivity_benchmark_failed"
        json.dumps(ei.value.details, allow_nan=False)
        assert ei.value.details["treatment"] == "t"
        assert ei.value.details["outcome"] == "y"
        assert ei.value.details["covariates"] == ["c"]

    def _sensitivity_only_runner(self):
        return RefutationRunner(
            config={
                "placebo_treatment": {"enabled": False},
                "random_common_cause": {"enabled": False},
                "data_subset": {"enabled": False},
                "bootstrap": {"enabled": False},
            }
        )

    def test_an_unusable_outcome_column_fails_closed_not_unstandardized(self):
        """H3: a non-numeric outcome column makes the SD computation raise. Degrading
        that to ``None`` would send the classifier down the SMD path on the
        UNSTANDARDIZED effect — a scale-dependent, plausible-wrong number served as a
        reading. Same rule as the benchmark block: a real failure fails closed."""
        r = self._sensitivity_only_runner()
        # ``treatment`` is left None so only the SD guard is under test; the
        # benchmark block needs data AND treatment AND outcome to run at all.
        frame = pd.DataFrame({"y": ["low", "high", "low", "high"]})
        with pytest.raises(RefutationError) as ei:
            r.run_all_tests(
                original_effect=0.15,
                original_ci=(0.08, 0.22),
                data=frame,
                outcome="y",
                causal_model=_full_stub_causal_model(),
                identified_estimand=object(),
                estimate=_stub_estimate(),
            )
        assert ei.value.details["reason"] == "sensitivity_outcome_std_failed"
        assert ei.value.details["outcome"] == "y"

    def test_an_absent_outcome_column_is_still_a_served_reading(self):
        """A MISSING column is not a failure: there is simply no SD to compute, so
        the reading is served unstandardized with ``outcome_std`` reported None."""
        r = self._sensitivity_only_runner()
        frame = pd.DataFrame({"something_else": [1.0, 2.0, 3.0, 4.0]})
        suite = r.run_all_tests(
            original_effect=0.15,
            original_ci=(0.08, 0.22),
            data=frame,
            outcome="y",
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=_stub_estimate(),
        )
        sens = next(t for t in suite.tests if t.test_name == RefutationTestType.SENSITIVITY_E_VALUE)
        assert sens.details["outcome_std"] is None
        assert sens.details["standardized"] is False
        assert sens.details["reading"] == "unbenchmarked"

    def test_an_absent_outcome_column_leaves_the_benchmark_unbenchmarked(self):
        """An ABSENT column is a MISSING input, not a computation failure. The
        benchmark block indexes ``data[outcome]``, so without this guard a caller
        that names a column the refutation frame does not carry fails the whole
        suite closed instead of reading ``unbenchmarked`` — the same distinction the
        SD block above draws."""
        r = self._sensitivity_only_runner()
        frame = pd.DataFrame({"t": [0, 1, 0, 1], "y": [0.0, 1.0, 0.0, 1.0]})
        suite = r.run_all_tests(
            original_effect=0.15,
            original_ci=(0.08, 0.22),
            data=frame,
            treatment="t",
            outcome="missing_col",
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=_stub_estimate(),
        )
        sens = next(t for t in suite.tests if t.test_name == RefutationTestType.SENSITIVITY_E_VALUE)
        assert sens.details["reading"] == "unbenchmarked"
        assert sens.details["benchmark"] is None

    def test_an_absent_treatment_column_leaves_the_benchmark_unbenchmarked(self):
        """Mirror of the above on the treatment side, where the outcome column IS
        present and its SD is computed normally."""
        r = self._sensitivity_only_runner()
        frame = pd.DataFrame({"t": [0, 1, 0, 1], "y": [0.0, 1.0, 0.0, 1.0]})
        suite = r.run_all_tests(
            original_effect=0.15,
            original_ci=(0.08, 0.22),
            data=frame,
            treatment="missing_col",
            outcome="y",
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=_stub_estimate(),
        )
        sens = next(t for t in suite.tests if t.test_name == RefutationTestType.SENSITIVITY_E_VALUE)
        assert sens.details["reading"] == "unbenchmarked"
        assert sens.details["benchmark"] is None

    def test_a_missing_frame_is_still_a_legitimate_unbenchmarked_fallback(self):
        """The MISSING-input path is not a failure: with no data/treatment/outcome
        there is nothing to benchmark against and ``unbenchmarked`` is honest."""
        r = RefutationRunner(
            config={
                "placebo_treatment": {"enabled": False},
                "random_common_cause": {"enabled": False},
                "data_subset": {"enabled": False},
                "bootstrap": {"enabled": False},
            }
        )
        suite = r.run_all_tests(
            original_effect=0.15,
            original_ci=(0.08, 0.22),
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=_stub_estimate(),
        )
        sens = next(t for t in suite.tests if t.test_name == RefutationTestType.SENSITIVITY_E_VALUE)
        assert sens.details["reading"] == "unbenchmarked"

    def test_sensitivity_never_fails(self, runner):
        for effect, ci in [(0.001, (0.0005, 0.0015)), (0.5, (-0.3, 0.5)), (0.02, (0.01, 0.03))]:
            result = runner._run_sensitivity_test(
                original_effect=effect, original_ci=ci, baseline_risk=0.3
            )
            assert result.status != RefutationStatus.FAILED

    def test_sensitivity_never_blocks_the_gate(self, runner):
        tests = [
            RefutationResult(
                RefutationTestType.PLACEBO_TREATMENT, RefutationStatus.PASSED, 0.1, 0.1
            ),
            RefutationResult(
                RefutationTestType.RANDOM_COMMON_CAUSE, RefutationStatus.PASSED, 0.1, 0.1
            ),
            RefutationResult(
                RefutationTestType.SENSITIVITY_E_VALUE, RefutationStatus.WARNING, 0.1, 0.1
            ),
            RefutationResult(RefutationTestType.DATA_SUBSET, RefutationStatus.PASSED, 0.1, 0.1),
            RefutationResult(RefutationTestType.BOOTSTRAP, RefutationStatus.PASSED, 0.1, 0.1),
        ]
        conf = runner._calculate_confidence_score(tests)
        assert conf == pytest.approx(0.90)
        assert runner._determine_gate_decision(tests, conf) == GateDecision.PROCEED

    def test_n_rows_reports_the_full_frame_when_the_caller_overrides(self):
        """#1419: ``data`` may be the refutation SUBSAMPLE while the effect and CI
        are the FULL-frame estimate, so ``len(data)`` would name the wrong n in the
        null-finding sentence a leader reads. ``n_rows`` lets the caller say so."""
        r = RefutationRunner(
            config={
                "placebo_treatment": {"enabled": False},
                "random_common_cause": {"enabled": False},
                "data_subset": {"enabled": False},
                "bootstrap": {"enabled": False},
            }
        )
        subsample = pd.DataFrame({"t": [0, 1] * 20, "y": [0.0, 1.0] * 20})
        suite = r.run_all_tests(
            original_effect=0.05,
            original_ci=(-0.02, 0.12),
            data=subsample,
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=_stub_estimate(),
            n_rows=1500,
        )
        sens = next(t for t in suite.tests if t.test_name == RefutationTestType.SENSITIVITY_E_VALUE)
        assert sens.details["reading"] == "null_finding"
        assert sens.details["n_rows"] == 1500
        assert "n = 1500" in sens.details["message"]

    def test_n_rows_falls_back_to_the_passthrough_frame_length(self):
        """Without the override the reading names ``len(data)`` — correct for a
        caller that passed the full frame, and the reason the override exists for
        one that passed a subsample."""
        r = RefutationRunner(
            config={
                "placebo_treatment": {"enabled": False},
                "random_common_cause": {"enabled": False},
                "data_subset": {"enabled": False},
                "bootstrap": {"enabled": False},
            }
        )
        frame = pd.DataFrame({"t": [0, 1] * 20, "y": [0.0, 1.0] * 20})
        suite = r.run_all_tests(
            original_effect=0.05,
            original_ci=(-0.02, 0.12),
            data=frame,
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=_stub_estimate(),
        )
        sens = next(t for t in suite.tests if t.test_name == RefutationTestType.SENSITIVITY_E_VALUE)
        assert sens.details["n_rows"] == 40
        assert "n = 40" in sens.details["message"]

    def test_critical_set_comes_from_config(self):
        r = RefutationRunner(config={"random_common_cause": {"critical": False}})
        tests = [
            RefutationResult(
                RefutationTestType.PLACEBO_TREATMENT, RefutationStatus.PASSED, 0.1, 0.1
            ),
            RefutationResult(
                RefutationTestType.RANDOM_COMMON_CAUSE, RefutationStatus.FAILED, 0.1, 0.1
            ),
            RefutationResult(
                RefutationTestType.SENSITIVITY_E_VALUE, RefutationStatus.PASSED, 0.1, 0.1
            ),
        ]
        conf = r._calculate_confidence_score(tests)
        assert (
            r._determine_gate_decision(tests, conf) == GateDecision.REVIEW
        )  # 0.667, no critical failure


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

    def test_skipped_excluded_from_confidence_average(self, runner):
        """SKIPPED tests must be EXCLUDED from the weighted average, not padded at 0.5.

        Two critical tests PASSED + one critical test SKIPPED: the score must reflect
        only the evidence actually gathered (1.0), not be diluted to ~0.83 by a
        neutral 0.5 pad for the skipped test.
        """
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
            RefutationResult(
                test_name=RefutationTestType.SENSITIVITY_E_VALUE,
                status=RefutationStatus.SKIPPED,
                original_effect=0.15,
                refuted_effect=0.15,
            ),
        ]

        score = runner._calculate_confidence_score(tests)

        # Excluding the SKIPPED test, both remaining tests PASSED -> 1.0.
        assert score == pytest.approx(1.0)

    def test_all_skipped_fails_closed_to_zero(self, runner):
        """A suite where every test is SKIPPED carries zero evidence and must
        fail CLOSED (0.0 -> BLOCK band), not surface a 0.5 REVIEW-band score.
        """
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.SKIPPED,
                original_effect=0.15,
                refuted_effect=0.15,
            ),
            RefutationResult(
                test_name=RefutationTestType.SENSITIVITY_E_VALUE,
                status=RefutationStatus.SKIPPED,
                original_effect=0.15,
                refuted_effect=0.15,
            ),
        ]

        score = runner._calculate_confidence_score(tests)

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

    def test_critical_warning_does_not_block_proceed(self, runner):
        """Contract: only a critical test FAILED blocks. A critical test in WARNING
        with high confidence still yields PROCEED (documents the corrected
        GateDecision.PROCEED comment).
        """
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,  # critical, WARNING
                status=RefutationStatus.WARNING,
                original_effect=0.15,
                refuted_effect=0.08,
            ),
        ]

        decision = runner._determine_gate_decision(tests, confidence_score=0.85)

        assert decision == GateDecision.PROCEED


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
            estimate=_stub_estimate(),
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
            estimate=_stub_estimate(),
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
            estimate=_stub_estimate(),
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

    # ------------------------------------------------------------------ #
    # Orphan-fix: cooperative compute deadline (timed-out runs must NOT
    # orphan to_thread compute past the caller's hard wall-clock cap).
    # ------------------------------------------------------------------ #
    def test_run_all_tests_deadline_in_past_fails_closed(self, runner):
        """A compute deadline already in the past means the worker is out of
        time budget. ``run_all_tests`` must fail-closed BEFORE launching any
        refuter — each refuter re-fits the estimator and cannot be cancelled
        once started in a thread, so a timed-out run would otherwise orphan
        compute. We surface a clean RefutationError instead.
        """
        import time as _t

        with pytest.raises(RefutationError) as ei:
            runner.run_all_tests(
                original_effect=0.15,
                original_ci=(0.10, 0.20),
                causal_model=_full_stub_causal_model(),
                identified_estimand=object(),
                estimate=_stub_estimate(),
                deadline=_t.monotonic() - 1.0,
            )
        assert ei.value.details.get("reason") == "time_budget_exceeded"
        # Nothing ran — we refused to start a refuter we could not finish.
        assert ei.value.details.get("ran") == []

    def test_run_all_tests_generous_deadline_runs_all(self, runner):
        """A deadline far in the future is equivalent to no deadline: the full
        suite runs and nothing is skipped."""
        import time as _t

        suite = runner.run_all_tests(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=_stub_estimate(),
            deadline=_t.monotonic() + 1e9,
        )
        assert isinstance(suite, RefutationSuite)
        assert len(suite.tests) > 0

    def test_run_all_tests_skips_refuters_that_would_exceed_budget(self, monkeypatch):
        """When the remaining budget cannot fit the next refuter's ESTIMATED cost
        (per-refit time observed so far x that refuter's simulation count), the
        refuter is SKIPPED and ``run_all_tests`` fails-closed — rather than
        starting a refuter that would run past the hard cap and orphan compute.
        """
        import time as _t

        config = {
            "placebo_treatment": {"enabled": True, "num_simulations": 10},
            "random_common_cause": {"enabled": True, "num_simulations": 10},
            "data_subset": {"enabled": True, "num_subsets": 10},
            "bootstrap": {"enabled": True, "num_bootstraps": 10},
            "sensitivity_e_value": {"enabled": True},
        }
        runner = RefutationRunner(config=config)

        clock = {"now": 1000.0}
        monkeypatch.setattr(_t, "monotonic", lambda: clock["now"])
        real = runner._run_test_with_tracing

        def fake_run(**kwargs):
            # Each refuter consumes 60s of (fake) wall-clock when it runs.
            clock["now"] += 60.0
            return real(**kwargs)

        monkeypatch.setattr(runner, "_run_test_with_tracing", fake_run)

        with pytest.raises(RefutationError) as ei:
            runner.run_all_tests(
                original_effect=0.15,
                original_ci=(0.10, 0.20),
                causal_model=_full_stub_causal_model(),
                identified_estimand=object(),
                estimate=_stub_estimate(),
                deadline=1000.0 + 100.0,  # 100s budget from t=1000
            )
        # placebo runs first (no estimate yet, 1000 < 1100); after it the
        # per-refit estimate is 60/10 = 6s, so random_common_cause (10 sims ->
        # +60s) would land at 1120 > 1100 and must be skipped.
        assert ei.value.details.get("reason") == "time_budget_exceeded"
        assert "random_common_cause" in ei.value.details.get("skipped", [])

    def test_run_all_tests_per_refit_hint_gates_first_refuter(self):
        """With a ``per_refit_hint`` (e.g. the reconstruction-fit cost) and a
        tight deadline, even the FIRST refuter is gated: if hint x its sim count
        will not fit before the deadline it is skipped and we fail-closed. This
        closes the 'first refuter runs unconditionally and can orphan' gap — the
        first refuter no longer runs uncalibrated when a hint is available.
        """
        import time as _t

        config = {
            "placebo_treatment": {"enabled": True, "num_simulations": 30},
            "random_common_cause": {"enabled": False},
            "data_subset": {"enabled": False},
            "bootstrap": {"enabled": False},
            "sensitivity_e_value": {"enabled": False},
        }
        runner = RefutationRunner(config=config)

        # 30 sims x 40s/refit hint = 1200s >> 10s budget -> first refuter skipped
        # BEFORE it ever starts (no unconditional first-refuter pass).
        with pytest.raises(RefutationError) as ei:
            runner.run_all_tests(
                original_effect=0.15,
                original_ci=(0.10, 0.20),
                causal_model=_full_stub_causal_model(),
                identified_estimand=object(),
                estimate=_stub_estimate(),
                deadline=_t.monotonic() + 10.0,
                per_refit_hint=40.0,
            )
        assert ei.value.details.get("reason") == "time_budget_exceeded"
        assert "placebo_treatment" in ei.value.details.get("skipped", [])
        assert ei.value.details.get("ran") == []


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
            estimate=_stub_estimate(),
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


class TestSensitivityOutcomeStdMasking:
    """The refutation frame carries the NaN rows the estimation node masked before it
    fit. ``np.std`` over the raw column is NaN, which ``classify`` refuses — so the
    whole suite used to fail closed on a perfectly usable estimate."""

    def _sensitivity_only_runner(self):
        return RefutationRunner(
            config={
                "placebo_treatment": {"enabled": False},
                "random_common_cause": {"enabled": False},
                "data_subset": {"enabled": False},
                "bootstrap": {"enabled": False},
            }
        )

    def test_nan_outcome_rows_are_masked_out_of_the_sd(self):
        r = self._sensitivity_only_runner()
        frame = pd.DataFrame(
            {
                "t": [1.0, 0.0, 1.0, 0.0, np.nan, 1.0],
                "y": [1.0, 0.0, 1.0, np.nan, 0.0, 0.0],
            }
        )
        suite = r.run_all_tests(
            original_effect=0.15,
            original_ci=(0.08, 0.22),
            data=frame,
            treatment="t",
            outcome="y",
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=_stub_estimate(),
        )
        sens = next(t for t in suite.tests if t.test_name == RefutationTestType.SENSITIVITY_E_VALUE)
        assert sens.details["outcome_std"] == pytest.approx(float(np.std([1.0, 0.0, 1.0, 0.0])))
        assert np.isfinite(sens.details["e_value"])
        # The benchmark-inputs branch ran on this frame (6 rows, 4 jointly usable
        # after the treatment/outcome NaN mask) — the reading's n_rows must be the
        # SAME 4 rows the SD and benchmark above describe, not the raw 6-row frame
        # length. See ``BenchmarkInputs.n_rows`` (evalue.py) for why zero also
        # counts as computed rather than falling back to ``len(data)``.
        assert sens.details["n_rows"] == 4

    def test_caller_n_rows_wins_over_the_computed_masked_count(self):
        """#1419: the caller's explicit ``n_rows`` is the FULL estimation frame's
        count (the runner's ``data`` may be a refutation subsample), so it must win
        over the masked count this frame's own benchmark-inputs branch computes."""
        r = self._sensitivity_only_runner()
        frame = pd.DataFrame(
            {
                "t": [1.0, 0.0, 1.0, 0.0, np.nan, 1.0],
                "y": [1.0, 0.0, 1.0, np.nan, 0.0, 0.0],
            }
        )
        suite = r.run_all_tests(
            original_effect=0.15,
            original_ci=(0.08, 0.22),
            data=frame,
            treatment="t",
            outcome="y",
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=_stub_estimate(),
            n_rows=1500,
        )
        sens = next(t for t in suite.tests if t.test_name == RefutationTestType.SENSITIVITY_E_VALUE)
        assert sens.details["n_rows"] == 1500

    def test_a_measured_but_unscoreable_covariate_is_named_in_the_unbenchmarked_reading(self):
        """Live re-band 2026-09-10: ``peer_influence_score -> adopted`` declares ONE
        covariate (``centrality_z``) collinear with the continuous treatment
        (r = 0.9995). Its factor is skipped, the run reads ``unbenchmarked`` — and
        the message used to claim "no measured confounders exist for this design".
        The runner's fallback branch must count the covariates it measured and the
        persisted details must carry the count and the truthful headline."""
        r = self._sensitivity_only_runner()
        rng = np.random.default_rng(3)
        n = 600
        t = rng.normal(0.0, 1.0, n)
        frame = pd.DataFrame(
            {
                "t": t,
                "y": (rng.random(n) < 0.30 + 0.10 * (t > 0)).astype(float),
                "x": t + 1e-9 * rng.normal(0.0, 1.0, n),
            }
        )
        model = _full_stub_causal_model()
        model.get_common_causes = lambda: ["x"]
        suite = r.run_all_tests(
            original_effect=0.15,
            original_ci=(0.08, 0.22),
            data=frame,
            treatment="t",
            outcome="y",
            causal_model=model,
            identified_estimand=object(),
            estimate=_stub_estimate(),
        )
        sens = next(t for t in suite.tests if t.test_name == RefutationTestType.SENSITIVITY_E_VALUE)
        assert sens.status == RefutationStatus.WARNING
        assert sens.details["reading"] == "unbenchmarked"
        assert sens.details["covariate_bias_factors"] == {}
        assert sens.details["covariates_measured"] == 1
        assert sens.details["benchmark_basis"] == "measured_unscoreable"
        assert sens.details["headline"] == evalue.HEADLINE_UNBENCHMARKED_MEASURED_UNSCOREABLE
        assert (
            "none of the 1 measured confounder(s) could be scored on this frame"
            in sens.details["message"]
        )
        assert "no measured confounders exist" not in sens.details["message"]

    def test_caller_covariates_measured_wins_over_the_runners_own_count(self):
        """Same precedence as ``n_rows``: the agent node measures on the FULL frame and
        passes its count; the runner's own (subsample) count is only the fallback."""
        r = self._sensitivity_only_runner()
        frame = pd.DataFrame(
            {
                "t": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "y": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            }
        )
        suite = r.run_all_tests(
            original_effect=0.15,
            original_ci=(0.08, 0.22),
            data=frame,
            treatment="t",
            outcome="y",
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=_stub_estimate(),
            baseline_risk=None,
            naive_effect=None,
            covariate_bias_factors={},
            covariates_measured=2,
        )
        sens = next(t for t in suite.tests if t.test_name == RefutationTestType.SENSITIVITY_E_VALUE)
        assert sens.details["covariates_measured"] == 2
        assert sens.details["benchmark_basis"] == "measured_unscoreable"

    @staticmethod
    def _two_unscoreable_covariates_frame(n: int = 600, seed: int = 3) -> pd.DataFrame:
        """A real frame on which the runner's OWN benchmark-inputs branch runs and
        measures TWO covariates, both unscoreable: ``x`` collinear with the continuous
        treatment (the live ``centrality_z`` shape) and ``w`` its mirror image, whose
        median split puts every control in one stratum. Computed count 2, factors {}."""
        rng = np.random.default_rng(seed)
        t = rng.normal(0.0, 1.0, n)
        return pd.DataFrame(
            {
                "t": t,
                "y": (rng.random(n) < 0.30 + 0.10 * (t > 0)).astype(float),
                "x": t + 1e-9 * rng.normal(0.0, 1.0, n),
                "w": -t,
            }
        )

    def _computed_branch_suite(self, **kwargs):
        """No caller-supplied benchmark inputs, so the runner computes them from the
        frame with the model's common causes (the branch at the top of the
        sensitivity block in ``run_all_tests``)."""
        model = _full_stub_causal_model()
        model.get_common_causes = lambda: ["x", "w"]
        return self._sensitivity_only_runner().run_all_tests(
            original_effect=0.15,
            original_ci=(0.08, 0.22),
            data=self._two_unscoreable_covariates_frame(),
            treatment="t",
            outcome="y",
            causal_model=model,
            identified_estimand=object(),
            estimate=_stub_estimate(),
            **kwargs,
        )

    @staticmethod
    def _sensitivity_details(suite) -> dict:
        sens = next(t for t in suite.tests if t.test_name == RefutationTestType.SENSITIVITY_E_VALUE)
        return sens.details

    def test_computed_covariates_measured_is_the_number_of_covariates_present(self):
        """No caller value: the count the runner's own branch measured on the frame."""
        d = self._sensitivity_details(self._computed_branch_suite())
        assert d["covariate_bias_factors"] == {}  # both were measured, neither scoreable
        assert d["covariates_measured"] == 2
        assert d["benchmark_basis"] == "measured_unscoreable"
        assert d["headline"] == evalue.HEADLINE_UNBENCHMARKED_MEASURED_UNSCOREABLE

    def test_caller_covariates_measured_wins_when_the_computed_branch_ran(self):
        """The agent node measures on the FULL frame; the runner's ``data`` may be a
        subsample. A conflicting caller value must win over the branch's own 2."""
        d = self._sensitivity_details(self._computed_branch_suite(covariates_measured=5))
        assert d["covariates_measured"] == 5
        assert d["benchmark_basis"] == "measured_unscoreable"
        assert "none of the 5 measured confounder(s) could be scored" in d["message"]

    def test_caller_explicit_zero_covariates_measured_wins_over_the_computed_count(self):
        """Zero is a VALUE, not an absence (same rule as ``n_rows``): an explicit 0
        from the caller must not be replaced by the branch's computed 2, and the
        reading then says nothing was measured."""
        d = self._sensitivity_details(self._computed_branch_suite(covariates_measured=0))
        assert d["covariates_measured"] == 0
        assert d["benchmark_basis"] == "none_measured"
        assert d["headline"] == evalue.HEADLINES["unbenchmarked"]
        assert "no measured confounders exist for this design" in d["message"]
