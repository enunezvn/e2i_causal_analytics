"""Discovery frame pre-flight (Lane D item 1).

Spec: docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md
"Lane D — guided discovery on claims frames", item 1. Measured starting point
(``docs/demos/results/2026-09-22_discovery_real_claims_disproof/``): the real
Optum persistence frame has a singular correlation matrix (Charlson /
Elixhauser flag families and their composite scores are exact linear
combinations), fisherz refuses it, and even rank-pruned a PC fit on 43
covariates takes 230 s — so the DAG-learning frame must be made full-rank AND
capped before PC runs.

Every frame here is hand-built so the expected decision is known exactly:
an exact duplicate, a composite that is the sum of its parts, a constant,
and a screening rule whose ranking is fixed by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.discovery.preflight import (
    PreflightResult,
    preflight_discovery_frame,
)

T = "t"
Y = "y"


def _base_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c1 = rng.normal(size=n)
    c2 = rng.normal(size=n)
    c3 = rng.integers(0, 2, n).astype(float)
    t = (0.9 * c1 + rng.normal(size=n) > 0).astype(float)
    y = (0.8 * t + 0.7 * c2 + rng.normal(size=n) > 0).astype(float)
    return pd.DataFrame({T: t, Y: y, "c1": c1, "c2": c2, "c3": c3})


class TestConstantColumns:
    def test_constant_column_is_dropped_and_named(self) -> None:
        frame = _base_frame()
        frame["always_zero"] = 0.0
        frame["always_seven"] = 7.0
        result = preflight_discovery_frame(
            frame, T, Y, ["c1", "always_zero", "c2", "always_seven", "c3"]
        )
        assert result.constant == ["always_zero", "always_seven"]
        assert result.kept == ["c1", "c2", "c3"]
        assert result.collinear == []
        assert result.capped == []

    def test_constant_is_exact_equality_not_a_tolerance(self) -> None:
        """A column varying by one part in 1e12 is NOT constant: constancy is
        ``min == max`` (Lane A codex r4: "constant to a tolerance" is a proxy)."""
        frame = _base_frame()
        tiny = np.full(len(frame), 1.0)
        tiny[0] = 1.0 + 1e-12
        frame["nearly_constant"] = tiny
        result = preflight_discovery_frame(frame, T, Y, ["c1", "nearly_constant"])
        assert result.constant == []
        assert "nearly_constant" in result.kept


class TestExactCollinearity:
    def test_exact_duplicate_drops_the_later_column(self) -> None:
        frame = _base_frame()
        frame["c1_dup"] = frame["c1"].to_numpy()
        result = preflight_discovery_frame(frame, T, Y, ["c1", "c1_dup", "c2"])
        assert result.collinear == ["c1_dup"]
        assert result.kept == ["c1", "c2"]

    def test_manifest_order_decides_which_duplicate_survives(self) -> None:
        """The same pair offered in the opposite order drops the OTHER name:
        a column is kept iff it raises the rank of what came before it."""
        frame = _base_frame()
        frame["c1_dup"] = frame["c1"].to_numpy()
        result = preflight_discovery_frame(frame, T, Y, ["c1_dup", "c1", "c2"])
        assert result.collinear == ["c1"]
        assert result.kept == ["c1_dup", "c2"]

    def test_composite_equal_to_sum_of_parts_is_dropped(self) -> None:
        """A Charlson-style score that is a weighted sum of its flags raises no
        rank once the flags are in; the flags stay, the composite goes."""
        rng = np.random.default_rng(3)
        n = 400
        frame = _base_frame(n)
        flags = {f"flag_{i}": rng.integers(0, 2, n).astype(float) for i in range(4)}
        for name, values in flags.items():
            frame[name] = values
        frame["score"] = 1.0 * frame["flag_0"] + 2.0 * frame["flag_1"] + 3.0 * frame["flag_2"]
        result = preflight_discovery_frame(frame, T, Y, list(flags) + ["score", "c1"])
        assert result.collinear == ["score"]
        assert result.kept == list(flags) + ["c1"]

    def test_affine_duplicate_is_dropped_translation_and_scale_invariant(self) -> None:
        """``2**40 + 3 * flag`` is the same information as ``flag`` and is
        exactly representable (an integer flag under a power-of-two offset), so
        the dependence is exact in float64 too. Lane A codex r3/r4: remove the
        offset EXACTLY before centering, or the 2**40 offset's rounding in the
        mean makes an exact duplicate look independent."""
        frame = _base_frame()
        frame["c3_affine"] = 2.0**40 + 3.0 * frame["c3"].to_numpy()
        result = preflight_discovery_frame(frame, T, Y, ["c3", "c3_affine", "c1"])
        assert result.collinear == ["c3_affine"]
        assert result.kept == ["c3", "c1"]

    def test_kept_columns_match_the_correlation_matrix_rank(self) -> None:
        """The incremental criterion IS the spec's "raises the correlation-matrix
        rank": the kept set has full correlation rank and adding any dropped
        column back does not raise it."""
        rng = np.random.default_rng(5)
        n = 500
        frame = _base_frame(n)
        for i in range(5):
            frame[f"f{i}"] = rng.integers(0, 2, n).astype(float)
        frame["f_sum"] = frame["f0"] + frame["f1"]
        frame["f2_dup"] = frame["f2"] * 1.0
        frame["f_mix"] = 2.0 * frame["f3"] - frame["f4"] + 5.0
        cols = [f"f{i}" for i in range(5)] + ["f_sum", "f2_dup", "f_mix", "c1"]
        result = preflight_discovery_frame(frame, T, Y, cols)
        assert set(result.collinear) == {"f_sum", "f2_dup", "f_mix"}

        # Semantic cross-check, independent of the module's arithmetic: every
        # dropped column is an exact linear combination (plus intercept) of the
        # kept ones, and no kept column is one of the others.
        def rel_residual(target: str, basis: list[str]) -> float:
            A = np.column_stack(
                [np.ones(len(frame))] + [frame[b].to_numpy(dtype=float) for b in basis]
            )
            b = frame[target].to_numpy(dtype=float)
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            return float(np.linalg.norm(A @ coef - b) / np.linalg.norm(b - b.mean()))

        for dropped in result.collinear:
            assert rel_residual(dropped, result.kept) < 1e-10, dropped
        for kept in result.kept:
            others = [c for c in result.kept if c != kept]
            assert rel_residual(kept, others) > 1e-3, kept

    def test_a_column_independent_at_five_e_minus_nine_is_kept(self) -> None:
        """Lane A codex r4: the tolerance is ``max(n, k) * eps`` (what
        ``numpy.linalg.lstsq(rcond=None)`` calls rank-deficient), not a round
        1e-8 — a genuine 5e-9-relative component must survive."""
        rng = np.random.default_rng(7)
        n = 600
        frame = _base_frame(n)
        base = rng.normal(size=n)
        frame["base"] = base
        frame["base_plus"] = base + 5e-9 * rng.normal(size=n)
        result = preflight_discovery_frame(frame, T, Y, ["base", "base_plus"])
        assert result.collinear == []


class TestCap:
    def _frame_with_known_ranking(self, n: int = 2000, seed: int = 11) -> pd.DataFrame:
        """Eight covariates whose association with T and with Y is fixed by
        construction. ``a_*`` drive T only, ``b_*`` drive Y only, ``ab`` drives
        both, ``noise_*`` drive neither."""
        rng = np.random.default_rng(seed)
        cols = {
            "a_strong": rng.normal(size=n),
            "a_weak": rng.normal(size=n),
            "b_strong": rng.normal(size=n),
            "b_weak": rng.normal(size=n),
            "ab": rng.normal(size=n),
            "noise_1": rng.normal(size=n),
            "noise_2": rng.normal(size=n),
            "noise_3": rng.normal(size=n),
        }
        t = 2.0 * cols["a_strong"] + 0.6 * cols["a_weak"] + 1.5 * cols["ab"] + rng.normal(size=n)
        # No T term in Y: the a_* columns must reach Y only through nothing, so
        # the two rankings are known exactly (a_* -> T only, b_* -> Y only).
        y = 2.0 * cols["b_strong"] + 0.6 * cols["b_weak"] + 1.5 * cols["ab"] + rng.normal(size=n)
        return pd.DataFrame({T: t, Y: y, **cols})

    def test_cap_keeps_the_union_of_top_k_by_t_and_top_k_by_y(self) -> None:
        frame = self._frame_with_known_ranking()
        covs = list(frame.columns[2:])
        result = preflight_discovery_frame(frame, T, Y, covs, max_covariates=4)
        # top-2 by |assoc T| = {a_strong, ab}; top-2 by |assoc Y| = {b_strong, ab}
        # union (k=2) = 3 <= 4; k=3 adds a_weak and b_weak -> 5 > 4, so k=2,
        # and the one free slot is filled from the T list first (a_weak).
        assert set(result.kept) == {"a_strong", "ab", "b_strong", "a_weak"}
        assert result.screening["k"] == 2
        assert result.screening["k_treatment"] == 3
        assert result.screening["k_outcome"] == 2
        assert result.screening["top_by_treatment"] == ["a_strong", "ab", "a_weak"]
        assert result.screening["top_by_outcome"] == ["b_strong", "ab"]
        assert set(result.capped) == set(covs) - set(result.kept)
        assert len(result.kept) == 4

    def test_cap_fills_every_slot_when_candidates_remain(self) -> None:
        frame = self._frame_with_known_ranking()
        covs = list(frame.columns[2:])
        for cap in (1, 2, 3, 5, 6, 7):
            result = preflight_discovery_frame(frame, T, Y, covs, max_covariates=cap)
            assert len(result.kept) == cap, cap
            assert len(result.kept) + len(result.capped) == len(covs)

    def test_cap_preserves_manifest_order_of_the_kept_columns(self) -> None:
        frame = self._frame_with_known_ranking()
        covs = list(frame.columns[2:])
        result = preflight_discovery_frame(frame, T, Y, covs, max_covariates=4)
        assert result.kept == [c for c in covs if c in set(result.kept)]

    def test_screening_never_reads_the_treatment_outcome_relation(self) -> None:
        """Replacing Y by a permutation of itself (destroying every Y
        association, including with T) must leave the T-ranked half of the
        selection untouched — the rule scores T-association and Y-association
        separately and never Y given T."""
        frame = self._frame_with_known_ranking()
        covs = list(frame.columns[2:])
        before = preflight_discovery_frame(frame, T, Y, covs, max_covariates=2)
        shuffled = frame.copy()
        shuffled[Y] = np.random.default_rng(1).permutation(shuffled[Y].to_numpy())
        after = preflight_discovery_frame(shuffled, T, Y, covs, max_covariates=2)
        assert (
            before.screening["association_with_treatment"]
            == after.screening["association_with_treatment"]
        )
        assert (
            before.screening["association_with_outcome"]
            != after.screening["association_with_outcome"]
        )

    def test_ties_break_by_manifest_order(self) -> None:
        """Two covariates with IDENTICAL association to T and to Y (``b`` is
        ``a`` row-reversed, and T, Y are symmetric in the pair) are a tie; the
        earlier manifest name is ranked first, whichever order they arrive in."""
        rng = np.random.default_rng(2)
        n = 400
        a = rng.normal(size=n)
        b = a[::-1].copy()
        t = a + b
        y = a + b + 1.0
        frame = pd.DataFrame({T: t, Y: y, "a_col": a, "b_col": b})
        first = preflight_discovery_frame(frame, T, Y, ["a_col", "b_col"], max_covariates=1)
        second = preflight_discovery_frame(frame, T, Y, ["b_col", "a_col"], max_covariates=1)
        assert first.kept == ["a_col"]
        assert second.kept == ["b_col"]

    def test_no_cap_when_within_budget(self) -> None:
        frame = _base_frame()
        result = preflight_discovery_frame(frame, T, Y, ["c1", "c2", "c3"], max_covariates=20)
        assert result.capped == []
        assert result.kept == ["c1", "c2", "c3"]
        assert result.screening["k"] is None

    def test_cap_is_deterministic_across_calls(self) -> None:
        frame = self._frame_with_known_ranking()
        covs = list(frame.columns[2:])
        first = preflight_discovery_frame(frame, T, Y, covs, max_covariates=3)
        second = preflight_discovery_frame(frame, T, Y, covs, max_covariates=3)
        assert first.to_dict() == second.to_dict()


class TestProtectedCovariates:
    def test_protected_columns_survive_the_cap(self) -> None:
        frame = TestCap()._frame_with_known_ranking()
        covs = list(frame.columns[2:])
        result = preflight_discovery_frame(
            frame, T, Y, covs, max_covariates=3, protected=["noise_3"]
        )
        assert "noise_3" in result.kept
        assert len(result.kept) <= 3
        assert result.protected == ["noise_3"]

    def test_protected_duplicate_wins_over_an_unprotected_earlier_column(self) -> None:
        frame = _base_frame()
        frame["c1_dup"] = frame["c1"].to_numpy()
        result = preflight_discovery_frame(frame, T, Y, ["c1", "c1_dup"], protected=["c1_dup"])
        assert result.collinear == ["c1"]
        assert result.kept == ["c1_dup"]


class TestContract:
    def test_treatment_and_outcome_are_never_touched(self) -> None:
        frame = _base_frame()
        frame[T] = 1.0  # constant treatment: NOT the pre-flight's business
        result = preflight_discovery_frame(frame, T, Y, ["c1"])
        assert T not in result.constant and T not in result.collinear and T not in result.capped
        assert result.kept == ["c1"]

    def test_missing_covariate_raises(self) -> None:
        with pytest.raises(KeyError):
            preflight_discovery_frame(_base_frame(), T, Y, ["c1", "absent"])

    def test_invalid_cap_raises(self) -> None:
        with pytest.raises(ValueError):
            preflight_discovery_frame(_base_frame(), T, Y, ["c1"], max_covariates=0)

    def test_nan_rows_are_excluded_from_the_statistics_and_reported(self) -> None:
        frame = _base_frame()
        frame.loc[0, "c2"] = np.nan
        result = preflight_discovery_frame(frame, T, Y, ["c1", "c2"])
        assert result.n_rows_used == len(frame) - 1
        assert result.kept == ["c1", "c2"]

    def test_all_rows_nan_raises(self) -> None:
        frame = _base_frame()
        frame["c2"] = np.nan
        with pytest.raises(ValueError):
            preflight_discovery_frame(frame, T, Y, ["c1", "c2"])

    def test_to_dict_carries_every_decision(self) -> None:
        frame = _base_frame()
        frame["k"] = 1.0
        frame["c1_dup"] = frame["c1"].to_numpy()
        result = preflight_discovery_frame(
            frame, T, Y, ["c1", "k", "c1_dup", "c2", "c3"], max_covariates=2
        )
        payload = result.to_dict()
        assert payload["constant"] == ["k"]
        assert payload["collinear"] == ["c1_dup"]
        assert len(payload["kept"]) == 2
        assert set(payload["capped"]) | set(payload["kept"]) == {"c1", "c2", "c3"}
        assert payload["max_covariates"] == 2
        assert payload["n_offered"] == 5
        assert isinstance(result, PreflightResult)
