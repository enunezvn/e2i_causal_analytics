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

    def test_e_value_from_rr_does_not_overflow_for_a_huge_finite_rr(self):
        # RR*(RR-1) overflows to inf around RR ~ 1.3e154 before the sqrt ever runs
        # (1e150 alone does not reach it: 1e150*(1e150-1) ~ 1e300, still finite);
        # sqrt(RR)*sqrt(RR-1) stays finite well past that point.
        e = ev.e_value_from_rr(1e158)
        assert math.isfinite(e)
        assert e == pytest.approx(2e158, rel=1e-12)

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

    def test_measured_confounding_benchmark_validates_factors_even_with_a_joint_present(self):
        # a finite joint must not short-circuit validation of the covariate factors
        # that were supplied alongside it.
        with pytest.raises(ValueError):
            ev.measured_confounding_benchmark(1.2, {"c": None})
        with pytest.raises(ValueError):
            ev.measured_confounding_benchmark(1.2, {"c": float("nan")})


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

    def test_n_rows_tells_a_computed_zero_apart_from_a_missing_count(self):
        """A frame present but unusable (every treatment value NaN) has a COMPUTED
        count of zero; only a caller that never looked at a frame has none. Collapsing
        the two lets the runner substitute the refutation SUBSAMPLE's length into the
        reading's "n = ..." for a frame that yielded no rows at all."""
        frame = self._frame().assign(t=np.nan)
        inp = ev.benchmark_inputs_from_frame(frame, "t", "y", ["c"])
        assert inp.n_rows == 0 and isinstance(inp.n_rows, int)
        assert ev.BenchmarkInputs(baseline_risk=None, naive_effect=None).n_rows is None

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

    @staticmethod
    def _collinear_frame(n: int = 600, seed: int = 3) -> pd.DataFrame:
        """The live ``peer_influence_score -> adopted`` shape (re-band 2026-09-10):
        a continuous treatment, a binary outcome and ONE declared covariate that is
        collinear with the treatment (live r = 0.9995). The covariate's median split
        coincides with the treatment's, so every control sits in the covariate's
        low stratum and the factor is skipped — the confounder was measured and
        adjusted for, yet it cannot benchmark the effect."""
        rng = np.random.default_rng(seed)
        t = rng.normal(0.0, 1.0, n)
        x = t + 1e-9 * rng.normal(0.0, 1.0, n)
        y = (rng.random(n) < 0.30 + 0.10 * (t > 0)).astype(float)
        return pd.DataFrame({"t": t, "y": y, "x": x})

    def test_a_covariate_collinear_with_the_treatment_is_measured_but_unscoreable(self):
        inp = ev.benchmark_inputs_from_frame(self._collinear_frame(), "t", "y", ["x"])
        assert inp.covariate_bias_factors == {}
        assert inp.naive_effect is None  # continuous treatment: no naive contrast
        assert inp.covariates_measured == 1
        r = ev.classify(
            0.15,
            (0.08, 0.22),
            randomized=False,
            baseline_risk=inp.baseline_risk,
            outcome_std=0.46,
            naive_effect=inp.naive_effect,
            covariate_factors=inp.covariate_bias_factors,
            n_rows=inp.n_rows,
            covariates_measured=inp.covariates_measured,
        )
        assert r.reading == "unbenchmarked" and r.status == "warning"
        assert r.benchmark_basis == "measured_unscoreable"
        assert r.headline == ev.HEADLINE_UNBENCHMARKED_MEASURED_UNSCOREABLE
        assert r.covariates_measured == 1

    def test_a_second_scoreable_covariate_keeps_the_strongest_covariate_basis(self):
        frame = self._collinear_frame()
        rng = np.random.default_rng(11)
        frame["z"] = rng.normal(0.0, 1.0, len(frame)) + 0.5 * (frame.y - frame.y.mean())
        inp = ev.benchmark_inputs_from_frame(frame, "t", "y", ["x", "z"])
        assert set(inp.covariate_bias_factors) == {"z"}
        assert inp.covariates_measured == 2
        r = ev.classify(
            0.15,
            (0.08, 0.22),
            randomized=False,
            baseline_risk=inp.baseline_risk,
            outcome_std=0.46,
            naive_effect=inp.naive_effect,
            covariate_factors=inp.covariate_bias_factors,
            n_rows=inp.n_rows,
            covariates_measured=inp.covariates_measured,
        )
        assert r.benchmark_basis == "strongest_covariate"
        assert r.reading in ("beyond_measured_confounding", "within_measured_confounding")

    def test_covariates_measured_counts_only_columns_present_in_the_frame(self):
        inp = ev.benchmark_inputs_from_frame(self._frame(), "t", "y", ["c", "absent"])
        assert inp.covariates_measured == 1
        assert ev.BenchmarkInputs(baseline_risk=None, naive_effect=None).covariates_measured == 0

    def test_bias_factor_matches_a_hand_computed_tiny_frame(self):
        # treated (t=1): c=1 for 3/6, c=0 for 3/6 -> p_hi_t = 0.5
        # control (t=0): c=1 for 2/6, c=0 for 4/6 -> p_hi_c = 1/3 -> RR_EU = 0.5/(1/3) = 1.5
        # control c=1 (hi):  y = [1, 1]          -> m_hi = 1.0
        # control c=0 (lo):  y = [0, 0, 0, 1]     -> m_lo = 0.25 -> RR_UD = 1.0/0.25 = 4.0
        # B = RR_EU*RR_UD / (RR_EU+RR_UD-1) = 1.5*4.0 / (1.5+4.0-1) = 6.0/4.5 = 4/3
        df = pd.DataFrame(
            {
                "t": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                "c": [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                "y": [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
            }
        )
        f = ev.covariate_bias_factors(df, "t", "y", ["c"])
        assert f["c"] == pytest.approx(4 / 3)

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

    def test_perfect_separation_raises_instead_of_dropping_the_confounder(self):
        # treated c = [0, 0] -> p_hi_t = 0; control c = [1, 0] -> p_hi_c = 0.5
        # (RR_EU one-sided-zero limit). control y = [1, 0], matching c -> hi (c=1)
        # y = [1] -> m_hi = 1.0; lo (c=0) y = [0] -> m_lo = 0.0 (RR_UD limit too):
        # both ratios diverge simultaneously -- a positivity violation, not a skip.
        df = pd.DataFrame({"t": [1, 1, 0, 0], "c": [0, 0, 1, 0], "y": [0, 0, 1, 0]})
        with pytest.raises(ValueError, match="'c'"):
            ev.covariate_bias_factors(df, "t", "y", ["c"])

    def test_continuous_outcome_uses_the_smd_path_on_the_control_arm_difference(self):
        # UNEQUAL high-covariate shares (RR_EU != 1) so bias_factor's combination of
        # RR_EU and RR_UD is actually exercised, not just passed through as RR_UD:
        # treated c = [1,1,1,0,0] -> p_hi_t = 3/5 = 0.6
        # control c = [1,0,0,0,0] -> p_hi_c = 1/5 = 0.2 -> rr_eu = 0.6/0.2 = 3.0
        # control y: hi (c=1) = [20.0] -> mean 20.0; lo (c=0) = [2,4,6,8] -> mean 5.0
        all_y = [10.0, 12.0, 14.0, 1.0, 3.0, 20.0, 2.0, 4.0, 6.0, 8.0]
        df = pd.DataFrame(
            {
                "t": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                "c": [1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                "y": all_y,
            }
        )
        f = ev.covariate_bias_factors(df, "t", "y", ["c"])
        sd = float(np.std(all_y))  # population SD, matching the implementation's np.nanstd
        rr_eu = 0.6 / 0.2
        d = (20.0 - 5.0) / sd
        rr_ud = math.exp(0.91 * abs(d))
        expected = rr_eu * rr_ud / (rr_eu + rr_ud - 1)
        assert f["c"] == pytest.approx(expected)


class TestCategoricalCovariateBiasFactors:
    """A categorical confounder IS measured confounding (spec §4.1).

    Dropping it would understate the fallback benchmark and let a run read
    ``beyond_measured_confounding`` too easily -- the false robustness the
    E-value reading exists to prevent. Each level becomes a 0/1 indicator run
    through the SAME per-covariate path as a numeric column, and the covariate's
    factor is the strongest of them.
    """

    def _frame(self, seed: int = 11, n: int = 3000) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        g = rng.choice(["A", "B", "C"], size=n)
        is_c = (g == "C").astype(float)
        t = (rng.random(n) < 1 / (1 + np.exp(-(1.4 * is_c - 0.7)))).astype(int)
        y = (rng.random(n) < 1 / (1 + np.exp(-(1.2 * is_c + 0.3 * t - 0.5)))).astype(int)
        return pd.DataFrame({"t": t, "y": y, "g": g, "c": rng.normal(size=n)})

    def test_factor_is_the_max_over_per_level_indicators(self):
        df = self._frame()
        factors = ev.covariate_bias_factors(df, "t", "y", ["g"])
        per_level = {
            level: ev.covariate_bias_factors(
                df.assign(ind=(df.g == level).astype(float)), "t", "y", ["ind"]
            )["ind"]
            for level in ("A", "B", "C")
        }
        assert factors["g"] == pytest.approx(max(per_level.values()))
        assert factors["g"] > 1.05  # the C level confounds both T and Y

    def test_a_single_level_covariate_is_skipped(self):
        df = self._frame().assign(g="only_one")
        assert "g" not in ev.covariate_bias_factors(df, "t", "y", ["g"])

    def test_a_separating_level_does_not_take_the_whole_covariate_down(self):
        # the "z" indicator is [0, 0, 1, 0] (x5): treated share 0 (RR_EU limit) and
        # the control high/low outcome rates are 1.0 / 0.0 (RR_UD limit) at once. That
        # is one sparse cell, not a declared variable -- "z" is skipped (it has no
        # treated rows, so the MIN_CELL_SIZE rule catches it before the both-limits
        # refusal could) and "x" still scores. The numeric route keeps refusing the
        # same shape (see TestCovariateBiasFactors::test_perfect_separation_raises_...).
        # Rows are replicated x5 so the "x" cell reaches MIN_CELL_SIZE in both arms
        # (10 treated, 5 controls); every share is unchanged by replication.
        df = pd.DataFrame(
            {"t": [1, 1, 0, 0] * 5, "g": ["x", "x", "z", "x"] * 5, "y": [0, 0, 1, 0] * 5}
        )
        factors = ev.covariate_bias_factors(df, "t", "y", ["g"])
        x_only = ev.covariate_bias_factors(
            df.assign(ind=(df.g == "x").astype(float)), "t", "y", ["ind"]
        )["ind"]
        assert factors["g"] == pytest.approx(x_only)

    def test_benchmark_inputs_carries_both_numeric_and_categorical_factors(self):
        inp = ev.benchmark_inputs_from_frame(self._frame(), "t", "y", ["c", "g"])
        assert set(inp.covariate_bias_factors) == {"c", "g"}
        assert inp.covariate_bias_factors["g"] > 1.0

    def _long_tail(self, seed: int, n: int = 200, k: int = 30, rate: float = 0.10):
        """A sparse long-tail categorical: rare levels with a handful of non-event
        controls and no treated rows. Measured before this rule: 295 of 300 seeds
        tripped the both-limits refusal on SPARSITY, not on a real positivity
        violation. The live shape stays clear (0 of 200 raises at k=4 and k=12,
        n=1500), so this was latent -- but a sparse cell must never fail a run."""
        rng = np.random.default_rng(seed)
        w = 1.0 / np.arange(1, k + 1)
        w = w / w.sum()
        g = rng.choice([f"L{i:02d}" for i in range(k)], size=n, p=w)
        return pd.DataFrame(
            {"t": rng.integers(0, 2, n), "y": (rng.random(n) < rate).astype(int), "g": g}
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_a_sparse_long_tail_covariate_scores_instead_of_raising(self, seed: int):
        factors = ev.covariate_bias_factors(self._long_tail(seed), "t", "y", ["g"])
        assert math.isfinite(factors["g"]) and factors["g"] >= 1.0

    def test_an_unscoreable_level_is_skipped_and_the_others_still_score(self):
        # "sep" sits only among controls (RR_EU limit) and its controls are the only
        # ones with events (RR_UD limit) -- unscoreable at this sample size, and with
        # 0 treated rows it is a sparse cell under MIN_CELL_SIZE first. "x" and "z"
        # each reduce to their oriented RR_EU = (9/18)/(6/18) = 1.5. Rows are
        # replicated x3 so "x" and "z" reach MIN_CELL_SIZE in both arms (9 treated,
        # 6 controls); every share, and so the 1.5, is unchanged by replication.
        df = pd.DataFrame(
            {
                "t": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0] * 3,
                "g": ["x", "x", "x", "z", "z", "z", "sep", "sep", "x", "x", "z", "z"] * 3,
                "y": [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0] * 3,
            }
        )
        # the unscoreable level, handed over as a NUMERIC covariate, still refuses:
        # a declared numeric confounder that separates both is a real data problem.
        with pytest.raises(ValueError, match="'sep_ind'"):
            ev.covariate_bias_factors(
                df.assign(sep_ind=(df.g == "sep").astype(float)), "t", "y", ["sep_ind"]
            )
        scorable = {
            level: ev.covariate_bias_factors(
                df.assign(ind=(df.g == level).astype(float)), "t", "y", ["ind"]
            )["ind"]
            for level in ("x", "z")
        }
        factors = ev.covariate_bias_factors(df, "t", "y", ["g"])
        assert factors["g"] == pytest.approx(max(scorable.values()))
        assert factors["g"] == pytest.approx(1.5)

    def test_a_covariate_whose_every_level_is_unscoreable_is_skipped_not_fatal(self):
        # level "A" only above the treatment median, "B" only below: the covariate is
        # perfectly aligned with treatment, so neither level has a control stratum to
        # compare. A numeric covariate aligned the same way is already a skip; this
        # matches it rather than opening a new fail-closed path.
        rng = np.random.default_rng(3)
        t = np.arange(200, dtype=float)
        df = pd.DataFrame(
            {
                "t": t,
                "y": (rng.random(200) < 0.3).astype(int),
                "g": np.where(t > np.median(t), "A", "B"),
            }
        )
        inp = ev.benchmark_inputs_from_frame(df, "t", "y", ["g"])
        assert inp.covariate_bias_factors == {}
        assert ev.measured_confounding_benchmark(None, inp.covariate_bias_factors) == (
            None,
            "none_measured",
        )
        reading = ev.classify(
            0.2,
            (0.1, 0.3),
            randomized=False,
            baseline_risk=None,
            outcome_std=0.46,
            naive_effect=None,
            covariate_factors=inp.covariate_bias_factors,
            n_rows=inp.n_rows,
        )
        assert reading.reading == ev.READING_UNBENCHMARKED

    def test_a_nullable_string_column_scores_like_its_object_twin(self):
        """pandas' nullable ``string`` dtype holds ``pd.NA``, and comparing a raw
        object array containing it against a level raises "boolean value of NA is
        ambiguous" -- a TypeError that would surface as a suite-killing
        RefutationError. Nulls must be masked out BEFORE the comparison."""
        # replicated x5 so every level reaches MIN_CELL_SIZE in both arms (A and B:
        # 5 treated / 10 controls; C: 5 / 5); the masked indicators stay identical.
        levels = ["A", "B", "C", None, "A", "B", "C", "A", None, "B"] * 5
        df = pd.DataFrame(
            {
                "t": [1, 1, 1, 1, 0, 0, 0, 0, 1, 0] * 5,
                "y": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0] * 5,
            }
        )
        df["g_obj"] = pd.Series(levels, dtype=object)
        df["g_str"] = pd.array(levels, dtype="string")
        factors = ev.covariate_bias_factors(df, "t", "y", ["g_obj", "g_str"])
        assert factors["g_str"] == factors["g_obj"]  # exact: same masked indicator

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_a_bool_covariate_scores_exactly_like_its_integer_twin(self, seed: int):
        """A bool column IS binary, and spec §4.1 takes a binary covariate AS-IS —
        so it belongs on the numeric path, not the categorical max-over-levels one.
        Routing it as categorical made it disagree with its own 0/1 twin (measured
        1.091243 vs 1.092459 on seed 1, 1.133207 vs 1.142987 on seed 4)."""
        rng = np.random.default_rng(seed)
        n = 600
        b = rng.integers(0, 2, n)
        t = (rng.random(n) < 1 / (1 + np.exp(-(0.9 * b - 0.4)))).astype(int)
        y = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * b + 0.3 * t - 0.5)))).astype(int)
        df = pd.DataFrame({"t": t, "y": y, "num": b, "flag": b.astype(bool)})
        factors = ev.covariate_bias_factors(df, "t", "y", ["num", "flag"])
        assert factors["flag"] == factors["num"]  # exact, same code path

    def test_sparse_high_cardinality_categorical_does_not_inflate_the_benchmark(self):
        """A 30-level categorical at n=200 with NO relation to T or Y. Measured
        over seeds 0..49 -- BEFORE the minimum cell size: median 5.71, max 12.75
        (a factor on 50/50 seeds); AFTER (``MIN_CELL_SIZE = 5``): median 1.81,
        max 3.00 (a factor on 28/50 seeds; the rest have no level with 5 rows in
        both arms). Tiny cells (a level with 1 treated and 3 control rows, or 0
        treated rows and a handful of controls) produce ratios of the order of the
        arm ratio, and the covariate keeps the max -- a benchmark manufactured from
        sparsity on a column that carries no confounding at all. The residual 1.81
        is a selection artefact of the 30 % arm at this sparsity: a level needs
        5 treated rows out of ~6.7, so the survivors are treated-heavy (the same
        rule at a 50 % arm measures median 1.20); the live shape (4 and 12 levels,
        n=1500) measures 1.016 and 1.091."""
        collected = []
        for seed in range(50):
            rng = np.random.default_rng(seed)
            n = 200
            df = pd.DataFrame(
                {
                    "t": (rng.random(n) < 0.3).astype(int),
                    "y": (rng.random(n) < 0.10).astype(int),
                    "cat": rng.integers(0, 30, n).astype(str),
                }
            )
            factors = ev.covariate_bias_factors(df, "t", "y", ["cat"])
            if "cat" in factors:
                collected.append(factors["cat"])
        assert collected  # the rule must not silence the column on every seed
        assert float(np.median(collected)) < 2.0  # measured 1.81; was 5.71
        assert max(collected) <= 5.0  # measured 3.00; was 12.75

    def test_categorical_level_below_min_cell_size_is_skipped(self):
        # "sparse" has 3 treated rows and 40 controls with an extreme outcome rate;
        # "big" and "other" have 40 treated and 40 controls each. The sparse cell's
        # own indicator would score far above the well-populated levels; the rule
        # skips it and the covariate keeps the max over the populated levels only.
        assert ev.MIN_CELL_SIZE == 5
        rows = []
        rows += [("sparse", 1, 1)] * 3
        rows += [("sparse", 0, 1)] * 30 + [("sparse", 0, 0)] * 10
        rows += [("big", 1, 1)] * 8 + [("big", 1, 0)] * 32
        rows += [("big", 0, 1)] * 4 + [("big", 0, 0)] * 36
        rows += [("other", 1, 1)] * 8 + [("other", 1, 0)] * 32
        rows += [("other", 0, 1)] * 8 + [("other", 0, 0)] * 32
        df = pd.DataFrame(rows, columns=["g", "t", "y"])
        per_level = {
            level: ev.covariate_bias_factors(
                df.assign(ind=(df.g == level).astype(float)), "t", "y", ["ind"]
            )["ind"]
            for level in ("sparse", "big", "other")
        }
        assert per_level["sparse"] > max(per_level["big"], per_level["other"])
        factors = ev.covariate_bias_factors(df, "t", "y", ["g"])
        assert factors["g"] == pytest.approx(max(per_level["big"], per_level["other"]))

    def test_min_cell_size_leaves_live_cardinality_unchanged(self, monkeypatch):
        # live driver columns: single-digit cardinality at n=1500, every cell large,
        # mild real confounding (T depends on the level). The rule must not bite.
        rng = np.random.default_rng(7)
        n = 1500
        g = rng.choice(["A", "B", "C", "D"], size=n)
        lift = {"A": -0.6, "B": -0.2, "C": 0.2, "D": 0.6}
        logit = np.array([lift[v] for v in g])
        t = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * logit + 0.3 * t - 0.8)))).astype(int)
        df = pd.DataFrame({"t": t, "y": y, "g": g})
        with_rule = ev.covariate_bias_factors(df, "t", "y", ["g"])["g"]
        monkeypatch.setattr(ev, "MIN_CELL_SIZE", 1)
        without_rule = ev.covariate_bias_factors(df, "t", "y", ["g"])["g"]
        assert with_rule == without_rule  # exact: the same levels are scored
        assert with_rule > 1.0


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
        # Review round 1 (2026-09-10): the message used to close "Treat the
        # direction as more reliable than the size." For this reading the
        # benchmark B >= the observed risk ratio, so the Ding-VanderWeele bound
        # gives true RR >= RR_obs / B <= 1 -- a confounder of that strength could
        # null the effect out entirely, so the DIRECTION is not established either.
        assert "more reliable than the size" not in r.message
        assert (
            "Do not act on the size of this effect, and treat its direction as "
            "unconfirmed against confounding of that strength." in r.message
        )

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
        assert r.headline == "Robustness not benchmarked: no measured confounders"
        assert r.headline == ev.HEADLINES["unbenchmarked"]
        assert "no measured confounders exist for this design" in r.message
        assert r.covariates_measured == 0

    def test_unbenchmarked_none_measured_is_explicit_at_zero_covariates(self):
        r = self._c(
            0.15, (0.08, 0.22), naive_effect=None, covariate_factors={}, covariates_measured=0
        )
        assert r.reading == "unbenchmarked" and r.benchmark_basis == "none_measured"
        assert r.headline == "Robustness not benchmarked: no measured confounders"

    def test_unbenchmarked_says_measured_but_unscoreable_when_a_covariate_was_measured(self):
        """Live re-band 2026-09-10: 6 runs of ``peer_influence_score -> adopted`` read
        unbenchmarked with ONE declared covariate (``centrality_z``, r = 0.9995 with
        the treatment) that was measured and adjusted for but could not be scored.
        "no measured confounders exist for this design" is false there; the reading
        keeps its name and status, and the words say what actually happened."""
        r = self._c(
            0.15, (0.08, 0.22), naive_effect=None, covariate_factors={}, covariates_measured=1
        )
        assert r.reading == "unbenchmarked" and r.status == "warning"
        assert r.benchmark is None
        assert r.benchmark_basis == "measured_unscoreable"
        assert r.headline == "Robustness not benchmarked: measured confounders could not be scored"
        assert r.headline == ev.HEADLINE_UNBENCHMARKED_MEASURED_UNSCOREABLE
        # the exact clause, not a loose substring: "could be scored" alone would also
        # match a sentence claiming the confounders COULD be scored
        assert "none of the 1 measured confounder(s) could be scored on this frame" in r.message
        assert "collinear with the treatment" in r.message
        assert "no measured confounders exist" not in r.message
        assert f"E-value {r.e_value_point:.2f}" in r.message
        assert r.covariates_measured == 1
        assert ev.BASIS_IN_WORDS["measured_unscoreable"] == (
            "measured confounders that could not be scored"
        )
        assert r.as_details()["covariates_measured"] == 1

    def test_covariates_measured_does_not_touch_a_benchmarked_reading(self):
        r = self._c(0.15, (0.08, 0.22), naive_effect=0.20, covariates_measured=3)
        assert r.reading == "beyond_measured_confounding"
        assert r.benchmark_basis == "joint_naive_vs_adjusted"
        assert r.covariates_measured == 3

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

    def test_ci_must_contain_the_point_estimate(self):
        # effect 0.1 outside ci (0.8, 0.9): the RD path's CI-bound conversion would
        # otherwise leave the risk-difference domain (p1 = 0.3+0.8 = 1.1) and hit an
        # unreachable internal assert instead of a clean domain error.
        with pytest.raises(ValueError):
            self._c(0.1, (0.8, 0.9), baseline_risk=0.3, naive_effect=0.2)
        # a degenerate CI exactly at the estimate is fine (bound == effect).
        r = self._c(0.1, (0.1, 0.1), baseline_risk=0.3, naive_effect=0.2)
        assert r.rr_point > 1.0

    def test_ci_bound_domain_failure_falls_back_to_smd_not_an_unreachable_assert(self):
        # effect is just inside the CI (containment passes within the 1e-12
        # tolerance: p1 = 0.5 + (0.5 - 5e-13) = 0.9999999999995, barely in-domain),
        # but the bound itself (0.5) pushes p1 to exactly 1.0 -- out of the RD
        # domain even though the point is fine. The whole reading must fall back to
        # the SMD path, not hit an assert on the bound's conversion.
        r = self._c(0.5 - 5e-13, (0.5, 0.6), baseline_risk=0.5, naive_effect=None)
        assert r.conversion == "standardized_difference"

    def test_joint_benchmark_follows_classifys_own_conversion(self):
        # same CI-bound domain failure as above, but WITH a naive_effect: effect
        # (p1 = 0.9999999999995) and naive (p1 = 0.4) are each individually RD-valid
        # at baseline_risk 0.5, so a joint benchmark computed on its own would pick
        # RD -- but classify already fell back to SMD for the whole reading (the CI
        # bound is not RD-valid), and the joint benchmark must follow that choice,
        # not re-derive its own without the bound.
        r = self._c(
            0.5 - 5e-13,
            (0.5, 0.6),
            baseline_risk=0.5,
            outcome_std=None,
            naive_effect=-0.1,
            covariate_factors={},
            n_rows=100,
        )
        assert r.conversion == "standardized_difference"
        # exp(0.91*|eff|) / exp(0.91*|naive|), oriented >= 1 (already is: eff > naive)
        expected_benchmark = math.exp(0.91 * (0.5 - 5e-13 - 0.1))
        assert r.benchmark == pytest.approx(expected_benchmark)
        assert r.reading == "beyond_measured_confounding"

    def test_e_value_stays_finite_for_a_huge_effect(self):
        r = self._c(400, (399, 401), baseline_risk=None, outcome_std=None, naive_effect=None)
        assert math.isfinite(r.e_value_point)


class TestOutcomeStdFromFrame:
    """σ_Y for the reading, measured on the rows the estimate came from.

    ``estimation_data`` is the RAW passthrough frame while the estimation node masks
    NaN treatment/outcome rows before it fits, so NaN outcomes are an EXPECTED input.
    ``np.std`` over the raw column returns NaN, which ``classify`` refuses — turning a
    perfectly usable estimate into a failure. One function, three call sites.
    """

    def test_nan_outcomes_are_dropped_before_the_sd(self):
        frame = pd.DataFrame({"y": [1.0, 0.0, np.nan, 1.0, 0.0, np.nan]})
        assert ev.outcome_std_from_frame(frame, "y") == pytest.approx(
            float(np.std([1.0, 0.0, 1.0, 0.0]))
        )

    def test_the_treatment_mask_is_joint_with_the_outcome_mask(self):
        """The same joint mask ``benchmark_inputs_from_frame`` uses, so the SD and
        ``n_rows`` describe the SAME rows."""
        frame = pd.DataFrame({"t": [1.0, 0.0, np.nan, 1.0], "y": [10.0, 0.0, 100.0, 2.0]})
        assert ev.outcome_std_from_frame(frame, "y", treatment="t") == pytest.approx(
            float(np.std([10.0, 0.0, 2.0]))
        )
        # Without the treatment the row survives and moves the SD.
        assert ev.outcome_std_from_frame(frame, "y") == pytest.approx(
            float(np.std([10.0, 0.0, 100.0, 2.0]))
        )

    def test_a_missing_treatment_column_is_ignored_not_fatal(self):
        frame = pd.DataFrame({"y": [1.0, 0.0, np.nan, 1.0]})
        assert ev.outcome_std_from_frame(frame, "y", treatment="not_here") == pytest.approx(
            float(np.std([1.0, 0.0, 1.0]))
        )

    def test_a_string_outcome_column_raises(self):
        """PRESENT but unusable: coercing strings would invent a σ_Y."""
        frame = pd.DataFrame({"y": ["low", "high", "low"]})
        with pytest.raises(ValueError):
            ev.outcome_std_from_frame(frame, "y")

    def test_an_all_nan_outcome_column_raises(self):
        frame = pd.DataFrame({"y": [np.nan, np.nan]})
        with pytest.raises(ValueError, match="no usable rows"):
            ev.outcome_std_from_frame(frame, "y")

    def test_a_constant_outcome_returns_zero_for_classify_to_refuse(self):
        """0.0 is a measurement, not an error; ``classify`` applies the unusable rule."""
        frame = pd.DataFrame({"y": [2.0, 2.0, np.nan, 2.0]})
        assert ev.outcome_std_from_frame(frame, "y") == 0.0
        with pytest.raises(ValueError, match="outcome_std must be positive"):
            ev.classify(
                0.1,
                (0.05, 0.15),
                randomized=False,
                baseline_risk=None,
                outcome_std=0.0,
                naive_effect=None,
                covariate_factors={},
                n_rows=3,
            )
