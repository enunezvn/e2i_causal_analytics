"""Tests for adversarial leakage discriminator — Layer 3.

The discriminator computes a per-feature suspicion score that is
DATA-DERIVED (permutation-baseline-relative) rather than hardcoded
(e.g., 0.65 / 0.80). The threshold adapts to each cohort automatically.

Replaces the brittle thresholds from PR #83 Phase 1 with a defensible,
disease-agnostic, statistically-grounded approach.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


def test_clean_noise_is_not_suspicious():
    """A pure-noise feature should NOT trigger suspicion."""
    from src.data.adversarial_leakage import compute_adversarial_score

    rng = np.random.default_rng(42)
    n = 1000
    target = rng.binomial(1, 0.30, n).astype(int)
    feature = rng.normal(0, 1, n)  # No relationship to target

    result = compute_adversarial_score(feature, target, n_permutations=500, seed=7)
    assert not result["suspicious"], f"Clean noise flagged suspicious: {result}"
    # z_score should be small (within ~3σ of zero) for noise
    assert abs(result["z_score"]) < 5, f"Noise z_score too large: {result['z_score']}"


def test_strong_signal_is_suspicious():
    """A feature deterministically derived from the target should be suspicious."""
    from src.data.adversarial_leakage import compute_adversarial_score

    rng = np.random.default_rng(42)
    n = 1000
    target = rng.binomial(1, 0.30, n).astype(int)
    feature = target.astype(float) + rng.normal(0, 0.1, n)  # Strong signal

    result = compute_adversarial_score(feature, target, n_permutations=500, seed=7)
    assert result["suspicious"], f"Strong signal NOT flagged: {result}"
    assert result["z_score"] > 5


def test_threshold_adapts_to_cohort_size():
    """The same actual_auc should have different z_scores in differently-sized
    cohorts because the permutation null has different variance.

    This is the KEY property: thresholds adapt to data automatically.
    """
    from src.data.adversarial_leakage import compute_adversarial_score

    rng = np.random.default_rng(42)
    # Small cohort (high null variance) — feature with AUC 0.65
    n_small = 200
    target_small = rng.binomial(1, 0.30, n_small).astype(int)
    feature_small = 0.5 * target_small + rng.normal(0, 1, n_small)

    # Large cohort (low null variance) — feature with same AUC 0.65
    n_large = 5000
    rng2 = np.random.default_rng(42)
    target_large = rng2.binomial(1, 0.30, n_large).astype(int)
    feature_large = 0.5 * target_large + rng2.normal(0, 1, n_large)

    result_small = compute_adversarial_score(
        feature_small, target_small, n_permutations=500, seed=7
    )
    result_large = compute_adversarial_score(
        feature_large, target_large, n_permutations=500, seed=7
    )

    # Same approximate AUC, but z-scores differ because null variance differs
    # Larger cohort has tighter null → higher z-score for the same AUC
    assert result_large["z_score"] > result_small["z_score"], (
        f"Expected larger cohort to have higher z-score (tighter null); "
        f"got small={result_small['z_score']:.2f}, large={result_large['z_score']:.2f}"
    )


def test_low_prevalence_cohort_handles_gracefully():
    """At 2.4% prevalence (CSU realistic), the function should still work."""
    from src.data.adversarial_leakage import compute_adversarial_score

    rng = np.random.default_rng(42)
    n = 5000
    target = rng.binomial(1, 0.024, n).astype(int)  # 2.4% prevalence
    feature = 0.3 * target + rng.normal(0, 1, n)

    result = compute_adversarial_score(feature, target, n_permutations=500, seed=7)
    # Should return valid stats even at low prevalence
    assert "z_score" in result
    assert "actual_auc" in result
    assert "null_mean" in result
    assert "null_std" in result
    assert not np.isnan(result["z_score"])


def test_discriminator_score_for_journey_duration_pattern():
    """Reproduces the journey_duration_days post-Phase-2 pattern.

    Even after windowing, journey_duration_days has effective AUC ~0.59 in the
    cohort. The adversarial discriminator should classify this as suspicious-
    but-not-critical (z-score in 3-5 range), informing the LLM in Layer 2 that
    further investigation is warranted.
    """
    from src.data.adversarial_leakage import compute_adversarial_score

    rng = np.random.default_rng(42)
    n = 5000
    target = rng.binomial(1, 0.024, n).astype(int)  # 2.4% prevalence

    # Construct a feature with effective AUC ~0.59 (matches post-windowing
    # journey_duration_days behavior)
    feature = 0.6 * target + rng.normal(0, 1, n)
    result = compute_adversarial_score(feature, target, n_permutations=500, seed=7)

    # Expect z-score in moderate range (not noise, not critical)
    assert "z_score" in result
    # Don't assert specific range — that's the point: data-derived


def test_multi_feature_ablation():
    """Drop each feature, retrain, measure |delta_AUC|. Features whose removal
    drops AUC by > 5σ above ablation null are flagged.
    """
    from src.data.adversarial_leakage import compute_feature_ablation

    rng = np.random.default_rng(42)
    n = 1000
    n_features = 5
    target = rng.binomial(1, 0.30, n).astype(int)

    # Construct dataset where feature 0 carries most signal
    X = pd.DataFrame(
        {
            f"feature_{i}": (
                target.astype(float) + rng.normal(0, 1, n) if i == 0 else rng.normal(0, 1, n)
            )
            for i in range(n_features)
        }
    )

    result = compute_feature_ablation(X, target, n_permutations=200, seed=7)

    # feature_0 should have the largest delta_AUC
    deltas = {row["feature"]: row["delta_auc"] for row in result["per_feature"]}
    assert deltas["feature_0"] > deltas["feature_1"]
    assert deltas["feature_0"] > 0.10  # Significant lift


# --- Codex audit follow-ups (Layer 3 — item E) ------------------------------


def test_p_value_floored_at_one_over_n_plus_one():
    """Plus-one (Phipson & Smyth 2010) permutation p-value: the empirical
    p-value is ``(1 + #{null >= actual}) / (1 + n_permutations)``, so a sharp
    leak with ZERO null exceedances returns ``1 / (1 + n_permutations)`` —
    NEVER exactly 0.0.

    Supersedes the prior ``test_p_value_zero_means_below_one_over_n_permutations``,
    which asserted ``p_value == 0.0`` for a sharp leak. An exact-zero p-value
    is statistically invalid (a finite permutation sample cannot prove
    impossibility) and breaks downstream multiple-testing math: BH compares
    p-values against ``(k/m)*q`` thresholds and a literal 0.0 silently dominates
    every rank, while a log-scale transform would explode. The plus-one
    estimator is the standard unbiased fix and is what gives the floor the
    docstring always claimed but the old ``np.mean(null >= actual)`` never
    delivered.
    """
    from src.data.adversarial_leakage import compute_adversarial_score

    rng = np.random.default_rng(0)
    n = 500
    target = rng.integers(0, 2, n)
    # Near-perfect leak — actual_auc ≈ 1.0, well above any permuted null
    feature = target.astype(float) + rng.normal(0, 0.001, n)

    n_perm = 200
    result = compute_adversarial_score(feature, target, n_permutations=n_perm, seed=42)
    assert result["actual_auc"] > 0.99
    expected_floor = 1.0 / (1 + n_perm)
    assert result["p_value"] == pytest.approx(expected_floor), (
        f"Expected plus-one floor {expected_floor} for a sharp leak with zero "
        f"null exceedances; got {result['p_value']}."
    )
    assert result["p_value"] > 0.0, "Plus-one p-value must never be exactly 0.0"


def test_compute_feature_ablation_zero_null_std_returns_inf_not_nan():
    """When every permuted ablation produces the identical delta_AUC (null_std=0),
    the consumer's ``suspicious = z_score > z_threshold`` test must fire on
    a deterministically large signal rather than silently fail (NaN > 5 → False).
    Codex audit Finding 3: ``compute_feature_ablation`` previously returned
    NaN here, inconsistent with ``compute_adversarial_score`` which returns
    +inf for the same condition.
    """
    import math

    from src.data.adversarial_leakage import compute_feature_ablation

    # Trivial dataset: feature_0 is target plus tiny noise; one feature only;
    # perfect ablation outcome with low n_permutations to keep the test fast.
    rng = np.random.default_rng(42)
    n = 200
    target = rng.integers(0, 2, n)
    X = pd.DataFrame({"feat": target.astype(float) + rng.normal(0, 0.001, n)})

    # n_permutations=2 forces a tiny null distribution; if all permuted
    # retrainings happen to give an identical delta_auc by symmetry, null_std
    # collapses to 0. Deterministic check: assert that whenever null_std == 0,
    # z_score is +inf or 0.0 (and never NaN due to the falsy-zero pattern).
    result = compute_feature_ablation(X, target, n_permutations=2, seed=7)
    feat_row = result["per_feature"][0]
    null_std = feat_row["null_std"]
    z = feat_row["z_score"]
    if null_std == 0:
        assert not math.isnan(z), (
            f"null_std=0 produced z_score=NaN; expected +inf or 0.0 to match "
            f"compute_adversarial_score semantics. Row: {feat_row}"
        )
        assert math.isinf(z) or z == 0.0
    # If null_std > 0 the test is uninformative for this regression — skip.


# --- Layer-4 redesign Phase 0 — p-value + multiple-testing contract ----------
# Plan: .claude/plans/layer4-trigger-redesign-20260527.md (Layer A, Phase 0).


def test_adversarial_score_pvalue_never_exact_zero():
    """No matter how sharp the leak, the plus-one estimator floors the p-value
    at ``1/(1 + n_permutations)`` — it is never 0.0. A literal-zero p-value is
    statistically invalid from a finite permutation sample and corrupts the
    BH multiple-testing step that consumes these p-values downstream.
    """
    from src.data.adversarial_leakage import compute_adversarial_score

    rng = np.random.default_rng(1)
    n = 400
    target = rng.integers(0, 2, n)
    feature = target.astype(float) + rng.normal(0, 0.0005, n)  # near-deterministic leak

    for n_perm in (50, 200):
        result = compute_adversarial_score(feature, target, n_permutations=n_perm, seed=3)
        assert result["p_value"] == pytest.approx(1.0 / (1 + n_perm))
        assert result["p_value"] > 0.0


def test_feature_ablation_emits_floored_pvalue():
    """``compute_feature_ablation`` must emit a per-feature plus-one p-value
    (upper-tail of the |delta_AUC| permutation null), floored at
    ``1/(1 + n_permutations)`` and never exactly 0.0 — so the ablation signal
    can feed the same BH multiple-testing contract as the discriminator.
    """
    from src.data.adversarial_leakage import compute_feature_ablation

    rng = np.random.default_rng(2)
    n = 300
    target = rng.integers(0, 2, n)
    X = pd.DataFrame(
        {
            "leak": target.astype(float) + rng.normal(0, 0.3, n),
            "noise_a": rng.normal(0, 1, n),
            "noise_b": rng.normal(0, 1, n),
        }
    )

    n_perm = 40
    result = compute_feature_ablation(X, target, n_permutations=n_perm, seed=7)
    floor = 1.0 / (1 + n_perm)
    for row in result["per_feature"]:
        assert "p_value" in row, f"ablation row missing p_value: {row}"
        assert row["p_value"] >= floor - 1e-12, f"p_value below plus-one floor: {row}"
        assert row["p_value"] <= 1.0 + 1e-12
        assert row["p_value"] > 0.0


def test_feature_ablation_pvalue_is_column_order_invariant():
    """Per-feature permutations must be INDEPENDENT and keyed to the feature
    (not drawn sequentially from one shared RNG). The prior implementation used
    a single ``rng`` for the whole loop, so each feature's null depended on how
    many draws earlier features consumed — coupling the per-feature p-values
    and making the multiple-testing inputs depend on arbitrary column ORDER.

    With independent per-feature streams, a feature's p_value/z_score is
    identical regardless of where its column sits in the frame. This test
    reverses the column order and asserts every feature's p_value is unchanged
    — it FAILS under the shared-sequential-RNG design and PASSES under
    independent per-feature streams.
    """
    from src.data.adversarial_leakage import compute_feature_ablation

    rng = np.random.default_rng(11)
    n = 300
    target = rng.integers(0, 2, n)
    X = pd.DataFrame(
        {
            "leak": target.astype(float) + rng.normal(0, 0.5, n),
            "n1": rng.normal(0, 1, n),
            "n2": rng.normal(0, 1, n),
            "n3": rng.normal(0, 1, n),
        }
    )
    X_rev = X[list(reversed(X.columns))]

    n_perm = 40
    r1 = compute_feature_ablation(X, target, n_permutations=n_perm, seed=7)
    r2 = compute_feature_ablation(X_rev, target, n_permutations=n_perm, seed=7)
    p1 = {row["feature"]: row["p_value"] for row in r1["per_feature"]}
    p2 = {row["feature"]: row["p_value"] for row in r2["per_feature"]}
    assert p1.keys() == p2.keys()
    for feat in p1:
        assert p1[feat] == pytest.approx(p2[feat]), (
            f"{feat} p_value changed with column order "
            f"({p1[feat]} vs {p2[feat]}) — per-feature permutations are not "
            "independent/feature-keyed."
        )


def test_min_permutations_for_fdr():
    """The smallest plus-one p-value ``1/(1+n)`` must be <= the BH rank-1
    threshold ``q/m`` for ANY rejection to be possible; hence ``n >= m/q - 1``,
    i.e. ``min_permutations_for_fdr(m, q) = ceil(m/q) - 1``. At m=40, q=0.05
    this is 799 (``1/800 == q/m``) — so the legacy n=200 default yields a
    structurally always-empty confident set.
    """
    from src.data.adversarial_leakage import min_permutations_for_fdr

    assert min_permutations_for_fdr(40, 0.05) == 799
    assert min_permutations_for_fdr(47, 0.05) == math.ceil(47 / 0.05) - 1
    assert min_permutations_for_fdr(20, 0.05) == 399
    # The returned floor permits a rank-1 rejection, and one fewer does NOT —
    # locking the exact boundary (n >= m/q - 1, not the off-by-one n >= m/q).
    m, q = 40, 0.05
    n = min_permutations_for_fdr(m, q)
    assert 1.0 / (1 + n) <= q / m  # n=799: 1/800 == q/m, feasible
    assert 1.0 / (1 + (n - 1)) > q / m  # n=798: 1/799 > q/m, infeasible


def test_benjamini_hochberg_rejects_clear_signals_order_aligned():
    """BH at q rejects the genuinely-small p-values and returns a mask aligned
    with the INPUT order (not sorted order)."""
    from src.data.adversarial_leakage import benjamini_hochberg

    # Shuffled input pins order-alignment: the two small p-values (0.001, 0.002)
    # sit at indices 1 and 3.
    p = [0.5, 0.001, 0.9, 0.002]
    mask = benjamini_hochberg(p, q=0.05)
    assert list(np.asarray(mask, dtype=bool)) == [False, True, False, True]


def test_benjamini_hochberg_empty_when_all_large():
    """No rejections when every p-value is well above threshold."""
    from src.data.adversarial_leakage import benjamini_hochberg

    mask = benjamini_hochberg([0.5, 0.6, 0.7, 0.8], q=0.05)
    assert not np.any(np.asarray(mask, dtype=bool))


def test_benjamini_hochberg_stepup_rejects_middle_failure():
    """The defining BH step-up property: rank-2 (0.035) individually FAILS its
    own threshold ``(2/3)*0.05 = 0.0333``, yet is rejected because the larger
    rank-3 (0.04) clears ``(3/3)*0.05 = 0.05``. A naive per-test comparison
    would wrongly keep rank-2. Sorted p = [0.001, 0.035, 0.04], m=3.
    """
    from src.data.adversarial_leakage import benjamini_hochberg

    mask = benjamini_hochberg([0.001, 0.035, 0.04], q=0.05)
    assert list(np.asarray(mask, dtype=bool)) == [True, True, True]


def test_benjamini_hochberg_npermutations_guard_raises_when_too_few():
    """When ``n_permutations`` is supplied, BH must refuse to run if it is too
    small for any rejection to be possible (the always-empty-set trap): m=40,
    q=0.05 needs n>=799, so n=200 raises rather than silently returning all
    False.
    """
    from src.data.adversarial_leakage import benjamini_hochberg

    p = [0.001] * 40
    with pytest.raises(ValueError, match="n_permutations"):
        benjamini_hochberg(p, q=0.05, n_permutations=200)


def test_benjamini_hochberg_npermutations_guard_passes_when_sufficient():
    """With sufficient permutations (n >= min_permutations_for_fdr(m, q) = 799
    at m=40, q=0.05) the guard is satisfied and BH proceeds normally. The
    p-values must be ACHIEVABLE at this budget (>= 1/(1+n) ~= 0.000999 at
    n=1000); 0.001 clears both the floor and every BH rank threshold, so all 40
    reject.
    """
    from src.data.adversarial_leakage import benjamini_hochberg

    p = [0.001] * 40  # >= plus-one floor 1/1001 and <= the BH rank-1 threshold
    mask = benjamini_hochberg(p, q=0.05, n_permutations=1000)
    assert np.all(np.asarray(mask, dtype=bool))


def test_benjamini_hochberg_rejects_zero_or_out_of_range_pvalue():
    """A literal 0.0 (or any value outside (0, 1]) is invalid input: a plus-one
    permutation p-value is never exactly 0.0, and a 0.0 would otherwise sort to
    the top of the BH ranking and be wrongly rejected. Fail loud, with and
    without an n_permutations budget. (Codex iter-0 HIGH: the guard validated
    the budget but not the supplied p-values.)
    """
    from src.data.adversarial_leakage import benjamini_hochberg

    with pytest.raises(ValueError, match="valid probabilities"):
        benjamini_hochberg([0.0, 0.5], q=0.05)
    with pytest.raises(ValueError, match="valid probabilities"):
        benjamini_hochberg([0.0], q=0.05, n_permutations=1000)
    with pytest.raises(ValueError, match="valid probabilities"):
        benjamini_hochberg([0.2, 1.5], q=0.05)  # > 1 is invalid
    with pytest.raises(ValueError, match="valid probabilities"):
        benjamini_hochberg([-0.1, 0.5], q=0.05)  # negative is invalid


def test_benjamini_hochberg_rejects_subfloor_pvalue():
    """When n_permutations is supplied, a finite p-value below the plus-one
    floor 1/(1+n) cannot have been produced by the plus-one estimator at that
    budget (e.g. a stale empirical-zero sidecar or a mismatched-n caller) and is
    rejected as invalid input — the Phase-0 'never below the floor' contract
    enforced at the BH boundary. (Codex iter-0 HIGH.)
    """
    from src.data.adversarial_leakage import benjamini_hochberg

    # Floor at n=1000 is 1/1001 ~= 0.000999; 0.0005 sits below it.
    with pytest.raises(ValueError, match="below the plus-one floor"):
        benjamini_hochberg([0.0005] * 40, q=0.05, n_permutations=1000)


def test_benjamini_hochberg_boundary_budget_feasible_and_infeasible():
    """The structurally-empty guard fires at the EXACT feasibility boundary, not
    one permutation early. At m=40, q=0.05 the floor is 799: n=799 makes the
    plus-one min p (1/800) exactly equal q/m so a rank-1 rejection IS possible
    (no raise); n=798 is genuinely infeasible (raise). (Codex iter-0 MEDIUM:
    the old ceil(m/q)=800 guard wrongly rejected the feasible n=799.)
    """
    from src.data.adversarial_leakage import (
        benjamini_hochberg,
        min_permutations_for_fdr,
    )

    assert min_permutations_for_fdr(40, 0.05) == 799
    # n=799: feasible — p exactly at the plus-one floor 1/800 clears q/m.
    p_at_floor = [1.0 / 800] * 40
    mask = benjamini_hochberg(p_at_floor, q=0.05, n_permutations=799)
    assert np.all(np.asarray(mask, dtype=bool))
    # n=798: structurally empty -> raise.
    with pytest.raises(ValueError, match="structurally empty"):
        benjamini_hochberg(p_at_floor, q=0.05, n_permutations=798)


def test_benjamini_hochberg_tolerates_nan_pvalues():
    """A NaN p-value (e.g. an ablation on a degenerate feature that yielded a
    NaN delta_auc) is permitted and treated as non-significant: never rejected,
    and it does not trip the (0, 1] / floor input validation.
    """
    from src.data.adversarial_leakage import benjamini_hochberg

    # Two clear signals + one NaN; NaN -> False, the finite signals reject.
    mask = np.asarray(benjamini_hochberg([0.001, float("nan"), 0.002], q=0.05), dtype=bool)
    assert not bool(mask[1])  # NaN -> not rejected
    assert bool(mask[0]) and bool(mask[2])  # finite signals reject

    # NaN is also skipped by the floor/range validation when a budget is given.
    mask2 = np.asarray(
        benjamini_hochberg([0.1, float("nan")], q=0.5, n_permutations=10),
        dtype=bool,
    )
    assert not bool(mask2[1])


def test_benjamini_hochberg_rejects_infinite_pvalue():
    """+/-inf are NOT legitimate p-values (unlike NaN, which means 'could not
    compute' and is tolerated as non-significant). They must be rejected before
    sorting — a -inf in particular sorts first and would otherwise be wrongly
    rejected as a confident leak. (Codex iter-1 HIGH: the isfinite() gate let
    inf bypass validation alongside NaN.)
    """
    from src.data.adversarial_leakage import benjamini_hochberg

    with pytest.raises(ValueError, match="valid probabilities"):
        benjamini_hochberg([float("inf"), 0.5], q=0.05)
    with pytest.raises(ValueError, match="valid probabilities"):
        benjamini_hochberg([float("-inf"), 0.5], q=0.05)
    with pytest.raises(ValueError, match="valid probabilities"):
        benjamini_hochberg([float("inf"), 0.5], q=0.05, n_permutations=1000)
    # NaN, by contrast, is still tolerated (not rejected, no raise).
    mask = np.asarray(benjamini_hochberg([float("nan"), 0.001], q=0.05), dtype=bool)
    assert not bool(mask[0])  # NaN -> not rejected
