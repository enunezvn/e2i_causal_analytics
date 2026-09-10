"""Hand-value tests for src/causal_engine/evalue.py (spec §4.1, §4.4)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.causal_engine import evalue as ev


class TestMath:
    def test_e_value_from_rr_matches_vanderweele_ding(self):
        # RR 2 -> 2 + sqrt(2) = 3.414...
        assert ev.e_value_from_rr(2.0) == pytest.approx(2.0 + math.sqrt(2.0))
        assert ev.e_value_from_rr(1.0) == 1.0

    def test_protective_rr_is_inverted(self):
        assert ev.e_value_from_rr(0.5) == pytest.approx(ev.e_value_from_rr(2.0))

    def test_rr_from_smd_uses_the_chinn_factor(self):
        assert ev.rr_from_smd(1.0) == pytest.approx(math.exp(0.91))
        assert ev.rr_from_smd(-1.0) == pytest.approx(math.exp(0.91))

    def test_rr_from_risk_difference_orients_and_guards_domain(self):
        assert ev.rr_from_risk_difference(0.10, 0.30) == pytest.approx(0.40 / 0.30)
        # a negative RD reverses the exposure coding: RR = p0 / p1
        assert ev.rr_from_risk_difference(-0.10, 0.30) == pytest.approx(0.30 / 0.20)
        assert ev.rr_from_risk_difference(0.10, 0.0) is None
        assert ev.rr_from_risk_difference(0.80, 0.30) is None  # p1 >= 1

    def test_bias_factor(self):
        assert ev.bias_factor(2.0, 2.0) == pytest.approx(4.0 / 3.0)
        assert ev.bias_factor(0.5, 2.0) == pytest.approx(4.0 / 3.0)  # oriented

    def test_joint_benchmark_is_oriented_and_none_without_naive(self):
        # naive RD 0.29 vs adjusted 0.16 at p0 0.30: (0.59/0.30)/(0.46/0.30)
        b = ev.joint_confounding_benchmark(0.29, 0.16, baseline_risk=0.30, outcome_std=None)
        assert b == pytest.approx((0.59 / 0.30) / (0.46 / 0.30))
        assert ev.joint_confounding_benchmark(
            0.10, 0.20, baseline_risk=0.30, outcome_std=None
        ) == pytest.approx((0.50 / 0.30) / (0.40 / 0.30))
        assert (
            ev.joint_confounding_benchmark(None, 0.16, baseline_risk=0.30, outcome_std=None) is None
        )

    def test_joint_benchmark_falls_back_to_smd_without_baseline_risk(self):
        b = ev.joint_confounding_benchmark(0.4, 0.2, baseline_risk=None, outcome_std=1.0)
        assert b == pytest.approx(math.exp(0.91 * 0.4) / math.exp(0.91 * 0.2))

    def test_measured_confounding_benchmark_prefers_joint(self):
        assert ev.measured_confounding_benchmark(1.2, {"a": 1.5}) == (
            1.2,
            "joint_naive_vs_adjusted",
        )
        assert ev.measured_confounding_benchmark(None, {"a": 1.5, "b": 1.1}) == (
            1.5,
            "strongest_covariate",
        )
        assert ev.measured_confounding_benchmark(None, {}) == (None, "none_measured")

    def test_joint_confounding_benchmark_never_mixes_conversions(self):
        # adjusted 0.8 leaves the RD domain at p0 0.3 (p1 = 1.1 >= 1) even though naive
        # 0.2 alone would be RD-valid (p1 = 0.5); the pair must fall back to the SMD
        # path TOGETHER, never RD-for-naive divided by SMD-for-adjusted.
        b = ev.joint_confounding_benchmark(0.2, 0.8, baseline_risk=0.3, outcome_std=1.0)
        assert b == pytest.approx(math.exp(0.91 * 0.6))

    def test_measured_confounding_benchmark_rejects_non_finite_joint(self):
        with pytest.raises(ValueError):
            ev.measured_confounding_benchmark(float("nan"), {})
        with pytest.raises(ValueError):
            ev.measured_confounding_benchmark(float("inf"), {"a": 1.5})

    def test_measured_confounding_benchmark_rejects_bad_covariate_factors(self):
        # a dropped None/non-finite factor can silently turn a benchmarked run into
        # "unbenchmarked" instead of surfacing the bad input.
        with pytest.raises(ValueError):
            ev.measured_confounding_benchmark(None, {"a": float("nan")})
        with pytest.raises(ValueError):
            ev.measured_confounding_benchmark(None, {"a": None})


class TestCovariateBiasFactors:
    def _frame(self, seed: int = 0, n: int = 4000) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        c = rng.normal(size=n)  # continuous confounder
        b = rng.integers(0, 2, size=n)  # binary confounder
        t = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * c + 0.6 * b - 0.5)))).astype(int)
        y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * c + 0.4 * b + 0.3 * t - 0.4)))).astype(int)
        return pd.DataFrame({"t": t, "y": y, "c": c, "b": b, "noise": rng.normal(size=n)})

    def test_factors_exceed_one_for_real_confounders_and_hug_one_for_noise(self):
        f = ev.covariate_bias_factors(self._frame(), "t", "y", ["c", "b", "noise"])
        assert set(f) == {"c", "b", "noise"}
        assert f["c"] > 1.05 and f["b"] > 1.02
        assert f["noise"] < 1.05

    def test_empty_covariates_gives_empty_dict(self):
        assert ev.covariate_bias_factors(self._frame(), "t", "y", []) == {}

    def test_benchmark_inputs_from_frame(self):
        frame = self._frame()
        inp = ev.benchmark_inputs_from_frame(frame, "t", "y", ["c", "b"])
        p0 = frame.loc[frame.t == 0, "y"].mean()
        assert inp.baseline_risk == pytest.approx(p0)
        assert inp.naive_effect == pytest.approx(frame.loc[frame.t == 1, "y"].mean() - p0)
        assert inp.treatment_is_binary and inp.outcome_is_binary
        assert set(inp.covariate_bias_factors) == {"c", "b"}

    def test_continuous_treatment_has_no_naive_contrast(self):
        frame = self._frame().assign(t=lambda d: d.c)  # continuous treatment
        inp = ev.benchmark_inputs_from_frame(frame, "t", "y", ["b"])
        assert inp.naive_effect is None and inp.baseline_risk is None
        assert not inp.treatment_is_binary
        assert "b" in inp.covariate_bias_factors  # median split on T still works

    def test_constant_treatment_is_not_binary(self):
        # a subset test (set(u) <= {0, 1}) would wrongly call an all-ones column
        # binary; spec §4.2 requires BOTH levels present, mirroring
        # `_compute_naive_contrast` -- there is no contrast to form from one level.
        frame = self._frame().assign(t=1)
        inp = ev.benchmark_inputs_from_frame(frame, "t", "y", ["b"])
        assert inp.treatment_is_binary is False
        assert inp.baseline_risk is None
        assert inp.naive_effect is None

    def test_bias_factor_matches_a_hand_computed_tiny_frame(self):
        # treated (t=1): c=1 for 3/6, c=0 for 3/6 -> p_hi_t = 0.5
        # control (t=0): c=1 for 2/6, c=0 for 4/6 -> p_hi_c = 1/3 -> RR_EU = 0.5/(1/3) = 1.5
        # control c=1 (hi):  y = [1, 1]          -> m_hi = 1.0
        # control c=0 (lo):  y = [0, 0, 0, 1]     -> m_lo = 0.25 -> RR_UD = 1.0/0.25 = 4.0
        df = pd.DataFrame(
            {
                "t": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                "c": [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                "y": [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
            }
        )
        f = ev.covariate_bias_factors(df, "t", "y", ["c"])
        assert f["c"] == pytest.approx(ev.bias_factor(1.5, 4.0))

    def test_one_sided_zero_outcome_rate_reads_as_the_oriented_rr_eu_limit(self):
        # RR_EU: treated p_hi_t = 3/4 = 0.75, control p_hi_c = 2/4 = 0.5 -> RR_EU = 1.5
        # control c=1 (hi): y = [1, 1] -> m_hi = 1.0 (some events)
        # control c=0 (lo): y = [0, 0] -> m_lo = 0.0 (zero events: RR_UD -> infinity)
        # Ding-VanderWeele limit as RR_UD -> infinity: B -> RR_EU (oriented).
        df = pd.DataFrame(
            {
                "t": [1, 1, 1, 1, 0, 0, 0, 0],
                "c": [1, 1, 1, 0, 1, 1, 0, 0],
                "y": [1, 0, 1, 0, 1, 1, 0, 0],
            }
        )
        f = ev.covariate_bias_factors(df, "t", "y", ["c"])
        assert f["c"] == pytest.approx(1.5)

    def test_covariate_with_no_variation_among_controls_is_skipped(self):
        # every control row has c=0: RR_UD has no high-covariate control stratum to
        # compare against, so this is genuinely undefined, not a Ding-VanderWeele limit.
        df = pd.DataFrame(
            {
                "t": [1, 1, 1, 1, 0, 0, 0, 0],
                "c": [1, 1, 0, 0, 0, 0, 0, 0],
                "y": [1, 0, 1, 0, 1, 0, 1, 0],
            }
        )
        f = ev.covariate_bias_factors(df, "t", "y", ["c"])
        assert "c" not in f

    def test_continuous_outcome_uses_the_smd_path_on_the_control_arm_difference(self):
        df = pd.DataFrame(
            {
                "t": [1, 1, 1, 1, 0, 0, 0, 0],
                "c": [1, 1, 0, 0, 1, 1, 0, 0],
                "y": [5.0, 6.0, 1.0, 2.0, 10.0, 12.0, 2.0, 4.0],
            }
        )
        f = ev.covariate_bias_factors(df, "t", "y", ["c"])
        y_sd = float(np.std(df["y"].to_numpy()))  # matches the implementation's np.nanstd
        rr_ud = ev.rr_from_smd((11.0 - 3.0) / y_sd)  # control hi mean 11.0, lo mean 3.0
        assert f["c"] == pytest.approx(ev.bias_factor(1.0, rr_ud))  # rr_eu = 0.5 / 0.5 = 1.0


class TestClassify:
    def _c(self, effect, ci, **kw):
        base = {
            "randomized": False,
            "baseline_risk": 0.30,
            "outcome_std": 0.46,
            "naive_effect": None,
            "covariate_factors": {},
            "n_rows": 1500,
        }
        base.update(kw)
        return ev.classify(effect, ci, **base)

    def test_randomized_is_skipped_and_still_carries_numbers(self):
        r = self._c(0.15, (0.08, 0.22), randomized=True, naive_effect=0.20)
        assert r.reading == "not_applicable_randomized" and r.status == "skipped"
        assert r.e_value_point > 1.0 and r.headline.startswith("Not applicable")

    def test_null_finding_when_ci_includes_zero(self):
        r = self._c(0.05, (-0.02, 0.12), naive_effect=0.10)
        assert r.reading == "null_finding" and r.status == "warning"
        assert r.e_value_ci == 1.0 and r.ci_includes_null
        assert "includes zero" in r.message and "n = 1500" in r.message

    def test_beyond_when_point_rr_exceeds_benchmark(self):
        # adjusted 0.15 at p0 0.30 -> RR 1.5; naive 0.20 -> RR 1.667; B_obs 1.11
        r = self._c(0.15, (0.08, 0.22), naive_effect=0.20)
        assert r.reading == "beyond_measured_confounding" and r.status == "passed"
        assert r.rr_point == pytest.approx(1.5) and r.benchmark == pytest.approx(
            (0.50 / 0.30) / 1.5
        )
        assert r.benchmark_basis == "joint_naive_vs_adjusted" and r.conversion == "risk_ratio"
        assert r.headline == "Robust to confounding at measured strength"

    def test_within_when_benchmark_is_at_least_the_point_rr(self):
        # adjusted 0.03 at p0 0.30 -> RR 1.10; naive 0.10 -> RR 1.333; B_obs 1.21 >= 1.10
        r = self._c(0.03, (0.01, 0.05), naive_effect=0.10)
        assert r.reading == "within_measured_confounding" and r.status == "warning"
        assert r.headline == "Sensitive to confounding"
        assert "could account for the whole effect" in r.message

    def test_tie_reads_within(self):
        # p0 = 0.25, effect = 0.25 -> p1 = 0.50 -> RR = 0.50/0.25 = 2.0 EXACTLY (both
        # exactly representable binary fractions, unlike 0.45/0.30's
        # 1.4999999999999998): a `>=` typo in the code would flip this to "beyond",
        # so the exact equality pins the strict `>` boundary precisely.
        r = self._c(
            0.25, (0.15, 0.35), baseline_risk=0.25, naive_effect=None, covariate_factors={"c": 2.0}
        )
        assert r.rr_point == 2.0 and r.benchmark == 2.0
        assert r.reading == "within_measured_confounding"

    def test_unbenchmarked_without_any_measured_confounding(self):
        r = self._c(0.15, (0.08, 0.22), naive_effect=None, covariate_factors={})
        assert r.reading == "unbenchmarked" and r.status == "warning"
        assert r.benchmark is None and r.benchmark_basis == "none_measured"

    def test_smd_path_without_baseline_risk(self):
        r = self._c(0.15, (0.08, 0.22), baseline_risk=None, naive_effect=0.20)
        assert r.conversion == "standardized_difference"
        assert r.rr_point == pytest.approx(math.exp(0.91 * 0.15 / 0.46))

    def test_negative_effect_is_oriented(self):
        r = self._c(-0.15, (-0.22, -0.08), naive_effect=-0.20)
        assert r.reading == "beyond_measured_confounding"
        assert r.rr_point == pytest.approx(0.30 / 0.15)

    def test_details_dict_is_json_plain(self):
        r = self._c(0.15, (0.08, 0.22), naive_effect=0.20, covariate_factors={"c": 1.1})
        d = r.as_details()
        assert d["reading"] == "beyond_measured_confounding"
        assert isinstance(d["covariate_bias_factors"], dict)
        assert all(isinstance(v, (float, int, str, bool, dict, type(None))) for v in d.values())

    def test_non_finite_inputs_raise(self):
        with pytest.raises(ValueError):
            self._c(float("nan"), (0.08, 0.22))
        with pytest.raises(ValueError):
            self._c(0.15, (0.08, float("inf")))

    def test_outcome_std_must_be_finite_and_positive(self):
        # a NaN/zero SD must never silently fall back to an unstandardized RR
        # (spec §5): for a binary outcome at effect 0.15 that would read 1.146
        # (exp(0.91*0.15)) instead of the correctly standardized 1.345.
        with pytest.raises(ValueError):
            self._c(0.15, (0.08, 0.22), outcome_std=float("nan"))
        with pytest.raises(ValueError):
            self._c(0.15, (0.08, 0.22), outcome_std=0.0)
        # None alone still means "no SD available" and works fine.
        r = self._c(0.15, (0.08, 0.22), outcome_std=None, naive_effect=0.20)
        assert r.rr_point > 1.0

    def test_classify_falls_back_to_smd_when_naive_leaves_the_rd_domain(self):
        # effect 0.15 is RD-valid at p0 0.3, but naive 0.8 is not (p1 = 1.1): the
        # WHOLE reading -- point RR included -- must fall back to the SMD path, not
        # just the naive/adjusted pair used for the benchmark.
        r = self._c(0.15, (0.08, 0.22), baseline_risk=0.3, naive_effect=0.8)
        assert r.conversion == "standardized_difference"
        assert r.rr_point == pytest.approx(math.exp(0.91 * 0.15 / 0.46))

    def test_n_rows_is_coerced_to_a_json_plain_int(self):
        r = self._c(0.15, (0.08, 0.22), naive_effect=0.20, n_rows=np.int64(1500))
        d = r.as_details()
        json.dumps(d)  # must not raise TypeError on a numpy scalar
        assert d["n_rows"] == 1500
        assert type(d["n_rows"]) is int
