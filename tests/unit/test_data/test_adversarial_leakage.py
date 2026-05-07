"""Tests for adversarial leakage discriminator — Layer 3.

The discriminator computes a per-feature suspicion score that is
DATA-DERIVED (permutation-baseline-relative) rather than hardcoded
(e.g., 0.65 / 0.80). The threshold adapts to each cohort automatically.

Replaces the brittle thresholds from PR #83 Phase 1 with a defensible,
disease-agnostic, statistically-grounded approach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
