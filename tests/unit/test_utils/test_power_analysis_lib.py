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
        result = binary_outcome_power(effect_size=0.20, alpha=0.05, power=0.80, baseline_rate=0.30)
        assert result.sample_size > 0

    def test_effect_size_type_is_rate_ratio(self):
        result = binary_outcome_power(effect_size=0.20, alpha=0.05, power=0.80, baseline_rate=0.30)
        assert result.effect_size_type == "rate_ratio"

    def test_mde_is_absolute_difference(self):
        # effect_size=0.20 with baseline 0.30 means p2=0.36, |p2-p1|=0.06
        result = binary_outcome_power(effect_size=0.20, alpha=0.05, power=0.80, baseline_rate=0.30)
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
        result = time_to_event_power(hazard_ratio=0.7, alpha=0.05, power=0.80, event_rate=0.5)
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
                n=100,
                alpha=0.05,
                power=0.80,
                outcome_type="bogus",  # type: ignore[arg-type]
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
        result = sensitivity_variations(effect_size=0.3, alpha=0.05, power=0.80, base_n=200)
        for entry in result["effect_size_variations"].values():
            assert "change_from_base" in entry


class TestEqualAllocationDesignsAreRecruitable:
    """#2015: every design is two arms of equal size, and its reported figures must be one
    that can be recruited. The per-arm figure used to be floor(total / 2) and the cluster
    count floor(clusters / 2): at d=0.3, ICC 0.05, 20 per cluster the library reported 683
    in total but 341 per arm and 17 clusters per arm — 680 subjects. Measured over the grid
    below before the fix: 29 of 64 cluster designs and 8 of 32 time-to-event designs could
    not reach their own total.

    The expectations are rebuilt here from the formulas, not read back from the library.
    """

    @staticmethod
    def _z(alpha: float, power: float) -> float:
        from scipy.stats import norm

        return float(norm.ppf(1 - alpha / 2) + norm.ppf(power))

    @pytest.mark.parametrize("d", [0.1, 0.2, 0.3, 0.5])
    @pytest.mark.parametrize("icc", [0.0, 0.01, 0.05, 0.2])
    @pytest.mark.parametrize("m", [1, 5, 20, 37])
    def test_cluster_totals_are_whole_arms_of_whole_clusters(self, d, icc, m):
        import math

        individual_total = 2 * math.ceil(2 * (self._z(0.05, 0.8) / d) ** 2)
        required = math.ceil(individual_total * (1 + (m - 1) * icc))
        result = cluster_rct_power(d, 0.05, 0.8, icc=icc, cluster_size=m)
        per_arm = result.sample_size_per_arm
        clusters_per_arm = result.extra["n_clusters_per_arm"]

        assert result.sample_size == 2 * per_arm >= required
        assert per_arm == math.ceil(required / 2)
        assert clusters_per_arm * 2 * m >= result.sample_size
        assert (clusters_per_arm - 1) * m < per_arm  # no spare cluster
        assert result.extra["n_clusters_total"] == 2 * clusters_per_arm

    def test_cluster_worked_example(self):
        result = cluster_rct_power(0.3, 0.05, 0.80, icc=0.05, cluster_size=20)
        # 2 x ceil(2 x (2.8016 / 0.3)^2) = 350; x 1.95 = 682.5 -> 683 -> 342 per arm.
        assert (result.sample_size, result.sample_size_per_arm) == (684, 342)
        assert (result.extra["n_clusters_per_arm"], result.extra["n_clusters_total"]) == (18, 36)

    @pytest.mark.parametrize("hr", [0.5, 0.7, 0.8, 1.3])
    @pytest.mark.parametrize("event_rate", [0.1, 0.33, 0.5, 0.9])
    @pytest.mark.parametrize("power", [0.8, 0.9])
    def test_time_to_event_totals_are_whole_arms(self, hr, event_rate, power):
        import math

        import numpy as np

        events = math.ceil(4 * (self._z(0.05, power) / float(np.log(hr))) ** 2)
        result = time_to_event_power(hr, 0.05, power, event_rate)
        assert result.extra["required_events"] == events
        assert result.sample_size == 2 * result.sample_size_per_arm
        assert result.sample_size >= events / event_rate
        assert result.sample_size_per_arm == math.ceil(math.ceil(events / event_rate) / 2)

    def test_time_to_event_worked_example(self):
        result = time_to_event_power(0.7, 0.05, 0.80, 0.33)
        # 247 events / 0.33 = 748.5 -> 749 -> 375 per arm (was 374, i.e. 748 < 749).
        assert result.extra["required_events"] == 247
        assert (result.sample_size, result.sample_size_per_arm) == (750, 375)


class TestUnusableDesignsAreRefused:
    """#2015 (codex whole-diff #5): a two-arm test needs at least two subjects per arm to
    estimate a variance, and a calculation that leaves the float range is not a sample size.
    Before: d=4 gave 1 per arm, d=1e200 gave 0, d=1e-200 raised OverflowError (which the
    composer executor retries), HR=1e100 gave a design with 1 event."""

    @pytest.mark.parametrize("d", [4.0, -4.0, 1e200])
    def test_an_effect_too_large_for_two_per_arm_is_refused(self, d):
        with pytest.raises(PowerCalculationError, match="per arm"):
            continuous_outcome_power(d, 0.05, 0.80)

    def test_the_smallest_usable_design_is_two_per_arm(self):
        # 2 x (2.8016 / 2.8)^2 = 2.002 -> 3; at d=3.9, 1.03 -> 2.
        assert continuous_outcome_power(3.9, 0.05, 0.80).sample_size_per_arm == 2

    @pytest.mark.parametrize(
        ("call", "args"),
        [
            (continuous_outcome_power, (1e-200, 0.05, 0.80)),
            (cluster_rct_power, (1e-200, 0.05, 0.80, 0.05, 20)),
        ],
    )
    def test_a_calculation_outside_the_float_range_is_an_overflow(self, call, args):
        # OverflowError, NOT PowerCalculationError: the data-preparer sufficiency gate
        # turns a PowerCalculationError into a floor-only verdict that can PASS, while an
        # uncomputable requirement must reach its blocking INCONCLUSIVE handler.
        with pytest.raises(OverflowError, match="finite") as raised:
            call(*args)
        assert not isinstance(raised.value, PowerCalculationError)

    def test_a_time_to_event_design_needs_two_events(self):
        with pytest.raises(PowerCalculationError, match="events"):
            time_to_event_power(1e100, 0.05, 0.80, 1.0)

    def test_a_time_to_event_design_needs_two_per_arm(self):
        # codex whole-diff #7: two events at event_rate 1 gave 2 in total, 1 per arm.
        with pytest.raises(PowerCalculationError, match="per arm"):
            time_to_event_power(100, 0.05, 0.80, 1.0)

    def test_a_binary_design_needs_two_per_arm_too(self):
        # codex whole-diff #6: at low power the lower bound (z_a + z_b)^2 / 2 is below 2.
        with pytest.raises(PowerCalculationError, match="per arm"):
            binary_outcome_power(98, 0.05, 0.10, 0.01)

    @pytest.mark.parametrize(
        ("call", "args"),
        [
            (cluster_rct_power, (1e-153, 0.05, 0.80, 0.5, 100)),
            (binary_outcome_power, (1, 1e-200, 0.80, 0.1)),
            (continuous_outcome_power, (0.2, 1e-200, 0.80)),
            (time_to_event_power, (0.7, 1e-200, 0.80, 0.5)),
        ],
    )
    def test_every_overflow_is_an_overflow_error(self, call, args):
        with pytest.raises(OverflowError) as raised:
            call(*args)
        assert not isinstance(raised.value, PowerCalculationError)
