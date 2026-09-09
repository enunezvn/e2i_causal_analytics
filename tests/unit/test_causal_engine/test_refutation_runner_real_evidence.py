"""Lane 1 (spec §4.1): data_subset and bootstrap produce REAL distributional evidence.

Before this change both tests called ``causal_model.refute_estimate`` and then
discarded the result because DoWhy 0.14 does not expose per-sample effects on
the refutation object: 96 of 96 live runs recorded them SKIPPED (measured
2026-09-08) at the same compute cost these loops have. The methods now run
their own resample loops with the public estimator calls DoWhy's refuters use,
keep DoWhy's significance test for the p-value, stop at the cooperative
deadline, and reproduce their draws from a seed.
"""

from __future__ import annotations

import time as _t
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from src.causal_engine.errors import RefutationError
from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationRunner,
    RefutationStatus,
    RefutationTestType,
)
from tests.unit.test_causal_engine.test_refutation_runner import (
    _make_stub_causal_model,
    _sequence_estimate,
    _stub_estimate,
)

CI = (0.10, 0.20)


def _subset(runner: RefutationRunner, estimate, **kw):
    return runner._run_data_subset_test(
        original_effect=0.15,
        original_ci=CI,
        causal_model=_make_stub_causal_model({}),
        identified_estimand=object(),
        estimate=estimate,
        use_dowhy=True,
        **kw,
    )


def _bootstrap(runner: RefutationRunner, estimate, **kw):
    return runner._run_bootstrap_test(
        original_effect=0.15,
        original_ci=CI,
        causal_model=_make_stub_causal_model({}),
        identified_estimand=object(),
        estimate=estimate,
        use_dowhy=True,
        **kw,
    )


def _bootstrap_values(width: float, center: float = 0.15, n: int = 40) -> List[float]:
    """40 evenly spaced values whose 2.5th–97.5th percentile span is ``width``
    (np.percentile's linear interpolation gives span = 0.95 × range)."""
    span = width / 0.95
    return [float(v) for v in np.linspace(center - span / 2, center + span / 2, n)]


class TestDataSubsetRealEvidence:
    def test_records_per_subset_effects_and_scores_coverage(self):
        runner = RefutationRunner()
        result = _subset(runner, _stub_estimate())
        assert result.status == RefutationStatus.PASSED
        assert len(result.details["subset_effects"]) == runner.config["data_subset"]["num_subsets"]
        assert result.details["ci_coverage"] == 1.0
        assert result.details["resamples_completed"] == 5
        assert result.details["resamples_requested"] == 5
        assert result.details["stopped_for_budget"] is False
        assert result.p_value is not None and 0.0 <= result.p_value <= 1.0

    def test_constant_refits_are_an_honest_skip(self):
        """Owner decision 2026-09-09: a zero-variance distribution is SKIPPED with
        a reason -- never a fabricated p-value, never a fail-closed halt from a
        NON-critical test (the critical placebo gate catches an estimator that
        ignores its data)."""
        runner = RefutationRunner()
        result = _subset(runner, _sequence_estimate([0.15] * 5))
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["reason"].startswith("degenerate_resample_distribution")
        assert result.details["resamples_completed"] == 5
        assert result.details["resample_effects"] == [0.15] * 5
        assert result.p_value is None

    @pytest.mark.parametrize(
        "inside,expected",
        [(8, RefutationStatus.PASSED), (7, RefutationStatus.WARNING), (6, RefutationStatus.FAILED)],
    )
    def test_coverage_thresholds_at_the_boundaries(self, inside, expected):
        runner = RefutationRunner(config={"data_subset": {"num_subsets": 10}})
        values = [0.15 + 0.001 * i for i in range(inside)] + [
            0.50 + 0.01 * i for i in range(10 - inside)
        ]
        result = _subset(runner, _sequence_estimate(values))
        assert result.details["ci_coverage"] == pytest.approx(inside / 10)
        assert result.status == expected

    def test_p_value_is_dowhys_significance_test(self):
        from dowhy.causal_refuter import test_significance

        runner = RefutationRunner()
        est = _stub_estimate()
        result = _subset(runner, est)
        expected = test_significance(est, np.asarray(result.details["subset_effects"]))["p_value"]
        assert result.p_value == pytest.approx(float(expected))

    def test_seeded_runs_reproduce_their_evidence(self):
        runner = RefutationRunner()
        a = _subset(runner, _stub_estimate(), resample_seed=7)
        b = _subset(runner, _stub_estimate(), resample_seed=7)
        c = _subset(runner, _stub_estimate(), resample_seed=8)
        assert a.details["subset_effects"] == b.details["subset_effects"]
        assert a.details["subset_effects"] != c.details["subset_effects"]

    def test_model_without_frame_fails_closed(self):
        runner = RefutationRunner()
        with pytest.raises(RefutationError) as ei:
            runner._run_data_subset_test(
                original_effect=0.15,
                original_ci=CI,
                causal_model=SimpleNamespace(),
                identified_estimand=object(),
                estimate=_stub_estimate(),
                use_dowhy=True,
            )
        assert ei.value.details["reason"] == "refutation_frame_missing"

    def test_refit_failure_fails_closed(self):
        runner = RefutationRunner()

        def boom(_df):
            raise ValueError("estimator exploded")

        with pytest.raises(RefutationError) as ei:
            _subset(runner, _stub_estimate(effect_fn=boom))
        assert ei.value.details["test_name"] == "data_subset"


class TestBootstrapRealEvidence:
    def test_thresholds_are_the_intended_values(self):
        assert RefutationRunner.PASS_THRESHOLDS["bootstrap_ci_ratio"] == {
            "pass": 1.50,
            "warning": 1.75,
        }

    @pytest.mark.parametrize(
        "width,expected",
        [
            (0.149, RefutationStatus.PASSED),
            (0.160, RefutationStatus.WARNING),
            (0.180, RefutationStatus.FAILED),
        ],
    )
    def test_width_ratio_thresholds_mean_what_the_comment_says(self, width, expected):
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 40}})
        result = _bootstrap(runner, _sequence_estimate(_bootstrap_values(width)))
        assert result.details["ci_ratio"] == pytest.approx(width / 0.10, rel=1e-6)
        assert result.status == expected

    def test_constant_refits_are_an_honest_skip(self):
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 12}})
        result = _bootstrap(runner, _sequence_estimate([0.15] * 12))
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["reason"].startswith("degenerate_resample_distribution")
        assert "bootstrap_ci" not in result.details
        assert result.p_value is None

    def test_records_per_bootstrap_effects(self):
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 12}})
        result = _bootstrap(runner, _stub_estimate())
        assert len(result.details["bootstrap_effects"]) == 12
        assert result.details["bootstrap_ci_available"] is True
        assert result.details["resamples_requested"] == 12
        assert result.details["resamples_completed"] == 12
        assert len(result.details["bootstrap_ci"]) == 2


class TestDeadlineInsideTheLoop:
    def _clocked(self, monkeypatch, cost_s: float):
        clock = {"now": 0.0}
        monkeypatch.setattr(_t, "monotonic", lambda: clock["now"])

        def effect(_df):
            clock["now"] += cost_s
            return 0.15 + 0.001 * clock["now"]

        return effect

    def test_stops_early_and_scores_on_the_completed_resamples(self, monkeypatch):
        effect = self._clocked(monkeypatch, 10.0)
        result = _subset(RefutationRunner(), _stub_estimate(effect_fn=effect), deadline=25.0)
        assert result.status == RefutationStatus.PASSED
        assert result.details["resamples_completed"] == 3
        assert result.details["resamples_requested"] == 5
        assert result.details["stopped_for_budget"] is True

    def test_below_the_minimum_is_an_honest_budget_skip(self, monkeypatch):
        effect = self._clocked(monkeypatch, 10.0)
        result = _subset(RefutationRunner(), _stub_estimate(effect_fn=effect), deadline=15.0)
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["resamples_completed"] == 2
        assert "time_budget" in result.details["reason"]
        assert "message" in result.details

    @pytest.mark.parametrize("deadline,skipped", [(9.5, False), (8.5, True)])
    def test_bootstrap_minimum_is_ten(self, monkeypatch, deadline, skipped):
        effect = self._clocked(monkeypatch, 1.0)
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 20}})
        result = _bootstrap(runner, _stub_estimate(effect_fn=effect), deadline=deadline)
        assert (result.status == RefutationStatus.SKIPPED) is skipped
        assert result.details["resamples_completed"] == (9 if skipped else 10)


class TestReviewBandArithmetic:
    """Why REVIEW was unreachable, pinned as arithmetic (spec §2)."""

    @staticmethod
    def _r(name: RefutationTestType, status: RefutationStatus) -> RefutationResult:
        return RefutationResult(
            test_name=name, status=status, original_effect=0.15, refuted_effect=0.15
        )

    def test_sensitivity_warning_with_both_noncritical_failed_is_review(self):
        runner = RefutationRunner()
        tests = [
            self._r(RefutationTestType.PLACEBO_TREATMENT, RefutationStatus.PASSED),
            self._r(RefutationTestType.RANDOM_COMMON_CAUSE, RefutationStatus.PASSED),
            self._r(RefutationTestType.SENSITIVITY_E_VALUE, RefutationStatus.WARNING),
            self._r(RefutationTestType.DATA_SUBSET, RefutationStatus.FAILED),
            self._r(RefutationTestType.BOOTSTRAP, RefutationStatus.FAILED),
        ]
        conf = runner._calculate_confidence_score(tests)
        assert conf == pytest.approx(0.65)
        assert runner._determine_gate_decision(tests, conf) == GateDecision.REVIEW

    def test_sensitivity_warning_with_both_passed_is_proceed(self):
        runner = RefutationRunner()
        tests = [
            self._r(RefutationTestType.PLACEBO_TREATMENT, RefutationStatus.PASSED),
            self._r(RefutationTestType.RANDOM_COMMON_CAUSE, RefutationStatus.PASSED),
            self._r(RefutationTestType.SENSITIVITY_E_VALUE, RefutationStatus.WARNING),
            self._r(RefutationTestType.DATA_SUBSET, RefutationStatus.PASSED),
            self._r(RefutationTestType.BOOTSTRAP, RefutationStatus.PASSED),
        ]
        conf = runner._calculate_confidence_score(tests)
        assert conf == pytest.approx(0.90)
        assert runner._determine_gate_decision(tests, conf) == GateDecision.PROCEED

    def test_only_criticals_scoring_could_never_reach_review(self):
        """What production did until this lane: both non-critical tests SKIPPED."""
        runner = RefutationRunner()
        tests = [
            self._r(RefutationTestType.PLACEBO_TREATMENT, RefutationStatus.PASSED),
            self._r(RefutationTestType.RANDOM_COMMON_CAUSE, RefutationStatus.PASSED),
            self._r(RefutationTestType.SENSITIVITY_E_VALUE, RefutationStatus.WARNING),
            self._r(RefutationTestType.DATA_SUBSET, RefutationStatus.SKIPPED),
            self._r(RefutationTestType.BOOTSTRAP, RefutationStatus.SKIPPED),
        ]
        conf = runner._calculate_confidence_score(tests)
        assert conf == pytest.approx(0.8667, abs=1e-3)
        assert runner._determine_gate_decision(tests, conf) == GateDecision.PROCEED


class TestNonFiniteRefitFailsClosed:
    """Quality review A (spec §5): a NaN/inf re-fit is an anomaly inside the loop
    and must be fail-closed like an exception -- never scored (a NaN counts as
    "below the estimate" in DoWhy's percentile test and poisons np.percentile)."""

    def test_subset_nan_refit_raises(self):
        runner = RefutationRunner()
        with pytest.raises(RefutationError) as ei:
            _subset(runner, _sequence_estimate([0.15, 0.16, float("nan"), 0.14, 0.15]))
        assert ei.value.details["reason"] == "non_finite_resample_effect"
        assert ei.value.details["test_name"] == "data_subset"
        assert ei.value.details["first_non_finite_index"] == 2
        assert ei.value.details["resamples_completed"] == 5

    def test_bootstrap_inf_refit_raises(self):
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 12}})
        values = [0.15 + 0.001 * i for i in range(12)]
        values[7] = float("inf")
        with pytest.raises(RefutationError) as ei:
            _bootstrap(runner, _sequence_estimate(values))
        assert ei.value.details["reason"] == "non_finite_resample_effect"
        assert ei.value.details["test_name"] == "bootstrap"
        assert ei.value.details["first_non_finite_index"] == 7


class TestBelowMinimumConfigIsNotABudgetSkip:
    """Quality review B: with no deadline and a requested count below the
    minimum the loop COMPLETES; the skip must say so, not blame the budget."""

    def test_subset_config_below_minimum(self):
        runner = RefutationRunner(config={"data_subset": {"num_subsets": 2}})
        result = _subset(runner, _stub_estimate())
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["reason"].startswith("config_below_minimum")
        assert result.details["stopped_for_budget"] is False
        assert result.details["resamples_completed"] == 2
        assert result.details["resamples_requested"] == 2
        assert "message" in result.details
        assert result.execution_time_ms >= 0.0

    def test_bootstrap_config_below_minimum(self):
        runner = RefutationRunner(config={"bootstrap": {"num_bootstraps": 4}})
        result = _bootstrap(runner, _stub_estimate())
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["reason"].startswith("config_below_minimum")
        assert result.details["stopped_for_budget"] is False
        assert result.details["resamples_completed"] == 4


class TestResampleSeedIs31Bit:
    """Quality review C: the docstring promises a 31-bit seed."""

    def test_seed_is_31_bit_stable_and_distinct(self):
        from src.causal_engine.refutation_runner import _resample_seed_for

        for est_id in ("est-1", "est-2", "3f1c2a9e-0000-4000-8000-000000000000", "x" * 64):
            seed = _resample_seed_for(est_id)
            assert seed is not None and 0 <= seed < 2**31
            assert _resample_seed_for(est_id) == seed
        assert _resample_seed_for("est-1") != _resample_seed_for("est-2")
        assert _resample_seed_for(None) is None
        assert _resample_seed_for("") is None

    def test_a_known_id_would_exceed_31_bits_unmasked(self):
        """Positive control: the mask matters (an unmasked 8-hex-digit prefix is
        32-bit; the reviewer measured a max of 4294943764)."""
        import hashlib

        from src.causal_engine.refutation_runner import _resample_seed_for

        hits = 0
        for i in range(64):
            est_id = f"est-{i}"
            raw = int(hashlib.sha256(est_id.encode("utf-8")).hexdigest()[:8], 16)
            if raw >= 2**31:
                hits += 1
                assert _resample_seed_for(est_id) == raw & 0x7FFFFFFF
        assert hits > 0, "no id in the sample exercised the mask"


class TestDegenerateOriginalCiSkipsBeforeCompute:
    """Quality review D: a widthless reported interval cannot score coverage or a
    width ratio; skip honestly BEFORE any re-fit, never blame the estimate."""

    @staticmethod
    def _never_called(_df):
        raise AssertionError("re-fit must not run for a widthless original_ci")

    def test_subset_widthless_ci(self):
        runner = RefutationRunner()
        result = runner._run_data_subset_test(
            original_effect=0.15,
            original_ci=(0.15, 0.15),
            causal_model=_make_stub_causal_model({}),
            identified_estimand=object(),
            estimate=_stub_estimate(effect_fn=self._never_called),
            use_dowhy=True,
        )
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["reason"].startswith("original_ci_degenerate")
        assert result.details["original_ci"] == (0.15, 0.15)
        assert "message" in result.details
        assert result.details["num_subsets"] == runner.config["data_subset"]["num_subsets"]
        assert result.p_value is None
        assert result.execution_time_ms >= 0.0

    def test_bootstrap_inverted_ci(self):
        runner = RefutationRunner()
        result = runner._run_bootstrap_test(
            original_effect=0.15,
            original_ci=(0.20, 0.10),
            causal_model=_make_stub_causal_model({}),
            identified_estimand=object(),
            estimate=_stub_estimate(effect_fn=self._never_called),
            use_dowhy=True,
        )
        assert result.status == RefutationStatus.SKIPPED
        assert result.details["reason"].startswith("original_ci_degenerate")
        assert result.details["num_bootstraps"] == runner.config["bootstrap"]["num_bootstraps"]
        assert result.p_value is None


class TestSkipResultsCarryExecutionTime:
    """Quality review E: every skip result records execution_time_ms."""

    def test_budget_skip_has_execution_time(self, monkeypatch):
        clock = {"now": 0.0}
        monkeypatch.setattr(_t, "monotonic", lambda: clock["now"])

        def effect(_df):
            clock["now"] += 10.0
            return 0.15 + 0.001 * clock["now"]

        result = _subset(RefutationRunner(), _stub_estimate(effect_fn=effect), deadline=15.0)
        assert result.status == RefutationStatus.SKIPPED
        assert "time_budget" in result.details["reason"]
        assert result.execution_time_ms > 0.0

    def test_degenerate_skip_has_execution_time(self):
        result = _subset(RefutationRunner(), _sequence_estimate([0.15] * 5))
        assert result.status == RefutationStatus.SKIPPED
        assert result.execution_time_ms > 0.0
