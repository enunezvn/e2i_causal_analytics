"""Unit tests for src/utils/power_analysis_lib.py.

Covers:
- Forward power calculations (continuous, binary, cluster RCT, time-to-event)
- Reverse MDE calculation
- Sensitivity grid and sensitivity variations
- Error handling (PowerCalculationError on impossible inputs)
- Monotonicity properties (smaller effect → larger n, higher power → larger n, etc.)
"""

from __future__ import annotations

import pytest

from src.utils.power_analysis_lib import (
    PowerCalculationError,
    binary_outcome_power,
    cluster_rct_power,
    continuous_outcome_power,
    mde_for_sample_size,
    sensitivity_grid,
    sensitivity_variations,
    time_to_event_power,
)


class TestContinuousOutcomePower:
    def test_returns_positive_sample_size(self):
        result = continuous_outcome_power(effect_size=0.5, alpha=0.05, power=0.80)
        assert result.sample_size > 0
        assert result.sample_size_per_arm > 0
        assert result.sample_size == 2 * result.sample_size_per_arm

    def test_effect_size_type_is_cohens_d(self):
        result = continuous_outcome_power(effect_size=0.5, alpha=0.05, power=0.80)
        assert result.effect_size_type == "cohens_d"

    def test_smaller_effect_requires_larger_n(self):
        large = continuous_outcome_power(effect_size=0.8, alpha=0.05, power=0.80)
        small = continuous_outcome_power(effect_size=0.2, alpha=0.05, power=0.80)
        assert small.sample_size > large.sample_size

    def test_higher_power_requires_larger_n(self):
        low = continuous_outcome_power(effect_size=0.3, alpha=0.05, power=0.80)
        high = continuous_outcome_power(effect_size=0.3, alpha=0.05, power=0.95)
        assert high.sample_size > low.sample_size

    def test_smaller_alpha_requires_larger_n(self):
        lax = continuous_outcome_power(effect_size=0.3, alpha=0.05, power=0.80)
        strict = continuous_outcome_power(effect_size=0.3, alpha=0.01, power=0.80)
        assert strict.sample_size > lax.sample_size

    def test_zero_effect_size_raises(self):
        with pytest.raises(PowerCalculationError):
            continuous_outcome_power(effect_size=0.0, alpha=0.05, power=0.80)

    def test_invalid_alpha_raises(self):
        with pytest.raises(PowerCalculationError):
            continuous_outcome_power(effect_size=0.3, alpha=1.5, power=0.80)

    def test_invalid_power_raises(self):
        with pytest.raises(PowerCalculationError):
            continuous_outcome_power(effect_size=0.3, alpha=0.05, power=0.0)


class TestBinaryOutcomePower:
    def test_returns_positive_sample_size(self):
        result = binary_outcome_power(
            effect_size=0.20, alpha=0.05, power=0.80, baseline_rate=0.30
        )
        assert result.sample_size > 0

    def test_effect_size_type_is_rate_ratio(self):
        result = binary_outcome_power(
            effect_size=0.20, alpha=0.05, power=0.80, baseline_rate=0.30
        )
        assert result.effect_size_type == "rate_ratio"

    def test_mde_is_absolute_difference(self):
        # effect_size=0.20 with baseline 0.30 means p2=0.36, |p2-p1|=0.06
        result = binary_outcome_power(
            effect_size=0.20, alpha=0.05, power=0.80, baseline_rate=0.30
        )
        assert abs(result.mde - 0.06) < 1e-9

    def test_invalid_baseline_raises(self):
        with pytest.raises(PowerCalculationError):
            binary_outcome_power(0.20, 0.05, 0.80, baseline_rate=1.5)

    def test_treatment_rate_out_of_bounds_raises(self):
        # p1=0.9, effect_size=0.5 → p2=1.35 (out of bounds)
        with pytest.raises(PowerCalculationError):
            binary_outcome_power(0.5, 0.05, 0.80, baseline_rate=0.9)


class TestClusterRCTPower:
    def test_design_effect_inflates_n(self):
        base = continuous_outcome_power(effect_size=0.3, alpha=0.05, power=0.80)
        cluster = cluster_rct_power(0.3, 0.05, 0.80, icc=0.05, cluster_size=20)
        assert cluster.sample_size >= base.sample_size

    def test_higher_icc_increases_n(self):
        low = cluster_rct_power(0.3, 0.05, 0.80, icc=0.01, cluster_size=20)
        high = cluster_rct_power(0.3, 0.05, 0.80, icc=0.10, cluster_size=20)
        assert high.sample_size > low.sample_size

    def test_design_effect_recorded(self):
        result = cluster_rct_power(0.3, 0.05, 0.80, icc=0.05, cluster_size=20)
        assert "design_effect" in result.extra
        # design_effect = 1 + (20 - 1) * 0.05 = 1.95
        assert abs(result.extra["design_effect"] - 1.95) < 1e-9

    def test_invalid_icc_raises(self):
        with pytest.raises(PowerCalculationError):
            cluster_rct_power(0.3, 0.05, 0.80, icc=1.5, cluster_size=20)

    def test_invalid_cluster_size_raises(self):
        with pytest.raises(PowerCalculationError):
            cluster_rct_power(0.3, 0.05, 0.80, icc=0.05, cluster_size=0)


class TestTimeToEventPower:
    def test_returns_positive_sample_size(self):
        result = time_to_event_power(
            hazard_ratio=0.7, alpha=0.05, power=0.80, event_rate=0.5
        )
        assert result.sample_size > 0
        assert result.extra["required_events"] > 0

    def test_smaller_hr_change_requires_larger_n(self):
        # HR=0.95 (small effect) vs HR=0.5 (large effect)
        small = time_to_event_power(0.95, 0.05, 0.80, 0.5)
        large = time_to_event_power(0.50, 0.05, 0.80, 0.5)
        assert small.sample_size > large.sample_size

    def test_hr_too_close_to_one_raises(self):
        with pytest.raises(PowerCalculationError):
            time_to_event_power(1.0001, 0.05, 0.80, 0.5)

    def test_negative_hr_raises(self):
        with pytest.raises(PowerCalculationError):
            time_to_event_power(-0.5, 0.05, 0.80, 0.5)

    def test_invalid_event_rate_raises(self):
        with pytest.raises(PowerCalculationError):
            time_to_event_power(0.7, 0.05, 0.80, event_rate=1.5)


class TestMDEForSampleSize:
    def test_continuous_reverse_inverts_forward(self):
        # Round-trip: forward(d) → n; reverse(n) → d'; d' should approximately equal d
        forward = continuous_outcome_power(effect_size=0.5, alpha=0.05, power=0.80)
        reversed_d = mde_for_sample_size(
            forward.sample_size, alpha=0.05, power=0.80, outcome_type="continuous"
        )
        # Reverse should give an MDE close to or slightly smaller than the original
        # (because forward rounds up sample size, so reverse can detect slightly smaller d)
        assert reversed_d <= 0.5
        assert reversed_d > 0.4  # within 20% of original

    def test_binary_reverse_gives_positive_diff(self):
        diff = mde_for_sample_size(
            n=1000, alpha=0.05, power=0.80, outcome_type="binary", baseline_rate=0.30
        )
        assert 0 < diff < 1

    def test_binary_reverse_requires_baseline(self):
        with pytest.raises(PowerCalculationError):
            mde_for_sample_size(n=1000, alpha=0.05, power=0.80, outcome_type="binary")

    def test_time_to_event_reverse_gives_hr_below_one(self):
        hr = mde_for_sample_size(
            n=1000, alpha=0.05, power=0.80, outcome_type="time_to_event", event_rate=0.5
        )
        assert 0 < hr < 1  # MDE expressed as protective HR

    def test_larger_n_gives_smaller_detectable_mde(self):
        small_n = mde_for_sample_size(100, 0.05, 0.80, "continuous")
        large_n = mde_for_sample_size(10000, 0.05, 0.80, "continuous")
        assert large_n < small_n

    def test_tiny_n_raises(self):
        with pytest.raises(PowerCalculationError):
            mde_for_sample_size(n=1, alpha=0.05, power=0.80, outcome_type="continuous")

    def test_unknown_outcome_type_raises(self):
        with pytest.raises(PowerCalculationError):
            mde_for_sample_size(
                n=100, alpha=0.05, power=0.80, outcome_type="bogus"  # type: ignore[arg-type]
            )


class TestSensitivityGrid:
    def test_grid_includes_all_candidates(self):
        result = sensitivity_grid(
            n=1000,
            alpha=0.05,
            power=0.80,
            outcome_type="continuous",
            candidates=[0.1, 0.3, 0.5],
        )
        assert len(result["grid"]) == 3

    def test_grid_marks_detectable_correctly(self):
        # n=1000 cannot detect d=0.05, easily detects d=0.5
        result = sensitivity_grid(
            n=1000,
            alpha=0.05,
            power=0.80,
            outcome_type="continuous",
            candidates=[0.05, 0.5],
        )
        small = next(g for g in result["grid"] if g["candidate_effect"] == 0.05)
        large = next(g for g in result["grid"] if g["candidate_effect"] == 0.5)
        assert not small["detectable_at_current_n"]
        assert large["detectable_at_current_n"]

    def test_grid_includes_detectable_mde_at_n(self):
        result = sensitivity_grid(
            n=1000,
            alpha=0.05,
            power=0.80,
            outcome_type="continuous",
            candidates=[0.3],
        )
        assert "detectable_mde_at_n" in result
        assert result["detectable_mde_at_n"] > 0

    def test_binary_grid_requires_baseline_rate(self):
        result = sensitivity_grid(
            n=1000,
            alpha=0.05,
            power=0.80,
            outcome_type="binary",
            candidates=[0.1, 0.3],
            baseline_rate=0.30,
        )
        assert len(result["grid"]) == 2
        for entry in result["grid"]:
            assert "required_n" in entry


class TestSensitivityVariations:
    def test_legacy_shape_preserved(self):
        result = sensitivity_variations(
            effect_size=0.3,
            alpha=0.05,
            power=0.80,
            base_n=200,
            outcome_type="continuous",
        )
        assert "effect_size_variations" in result
        assert "power_variations" in result
        # Same multiplier keys as the original PowerAnalysisNode._run_sensitivity_analysis
        assert "0.8x" in result["effect_size_variations"]
        assert "1.2x" in result["effect_size_variations"]
        assert "70%" in result["power_variations"]
        assert "90%" in result["power_variations"]

    def test_variation_records_change_from_base(self):
        result = sensitivity_variations(
            effect_size=0.3, alpha=0.05, power=0.80, base_n=200
        )
        for entry in result["effect_size_variations"].values():
            assert "change_from_base" in entry
