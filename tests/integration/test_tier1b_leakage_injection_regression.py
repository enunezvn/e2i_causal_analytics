"""Plan v3 §3 Tier 1B step 3 — Integration-level leakage-injection regression.

Plan §6 Tier 1B Gate B1 acceptance load-bearing test:

    "Leakage-injection regression: synthetic leak inserted into held-out
     cohort is caught at FPR ≤ current Layer 3 (regression test in CI)"

Pins the contract that HBLP variance-inflation does NOT mask injected
leaks. The test runs the REAL `compute_adversarial_score` (the production
Layer 3 scorer) on a synthetic DataFrame with a known leak feature, then
routes the resulting z_score through `hblp_classify` and asserts severity
remains 'high' under the HBLP-effective threshold.

This guards against the HBLP relaxation accidentally becoming a
"mask the leak by inflating the threshold" regression.

Distinct from the unit-level smoke tests in `test_hblp.py` — those use
hand-crafted z_scores. This test uses the actual scorer end-to-end on
realistic synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    hblp_classify,
)
from src.data.adversarial_leakage import compute_adversarial_score


def _build_clean_cohort(n_samples: int = 800, seed: int = 42):
    """Synthetic cohort with weak demographic-like signal — no leak."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 15, n_samples).clip(18, 89)
    icd_severe = rng.binomial(1, 0.3, n_samples)
    insurance_premium = rng.binomial(1, 0.5, n_samples)
    eligibility_days = rng.integers(60, 365, n_samples)
    # Weak target signal from these features.
    logit = (
        np.log(0.05 / 0.95)  # base rate ~5%
        + 0.20 * (age - 50) / 15
        + 0.30 * icd_severe
        + 0.15 * insurance_premium
        + 0.10 * (eligibility_days > 200)
        + rng.normal(0, 0.5, n_samples)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.uniform(size=n_samples) < prob).astype(int)
    return (
        pd.DataFrame(
            {
                "age": age,
                "icd_severe": icd_severe,
                "insurance_premium": insurance_premium,
                "eligibility_days": eligibility_days,
            }
        ),
        target,
    )


def _inject_strong_leak(
    df: pd.DataFrame, target: np.ndarray, leak_strength: float = 0.95
) -> pd.DataFrame:
    """Inject a leak feature that almost-perfectly predicts target.

    leak_strength is the fraction of target values copied; the
    remainder are flipped (controlled noise). At leak_strength=0.95
    the feature reveals 95% of target values — definitely a leak.
    """
    rng = np.random.default_rng(123)
    flip_mask = rng.uniform(size=len(target)) > leak_strength
    leak_feature = np.where(flip_mask, 1 - target, target).astype(float)
    out = df.copy()
    out["INJECTED_LEAK"] = leak_feature
    return out


def _inject_weak_leak(
    df: pd.DataFrame, target: np.ndarray, leak_strength: float = 0.55
) -> pd.DataFrame:
    """Inject a 'feature' barely better than chance — not actually a
    leak, but useful to confirm HBLP doesn't false-positive on
    legitimate weak signal."""
    rng = np.random.default_rng(456)
    flip_mask = rng.uniform(size=len(target)) > leak_strength
    weak_feature = np.where(flip_mask, 1 - target, target).astype(float)
    out = df.copy()
    out["WEAK_SIGNAL"] = weak_feature
    return out


# --------------------------------------------------------------------------- #
# Strong leak — HBLP MUST catch                                               #
# --------------------------------------------------------------------------- #


class TestStrongLeakCaughtUnderHBLP:
    """Plan §6 Tier 1B Gate B1: a near-perfect leak feature MUST be
    classified as severity=high under HBLP-effective thresholds, even
    at low N where HBLP relaxes the bar."""

    def _run_adversarial_scoring(self, leak_strength: float, n_samples: int):
        df, target = _build_clean_cohort(n_samples=n_samples)
        df_with_leak = _inject_strong_leak(df, target, leak_strength=leak_strength)
        score = compute_adversarial_score(
            df_with_leak["INJECTED_LEAK"].values,
            target,
            n_permutations=200,
            seed=42,
        )
        return score

    def test_strong_leak_caught_at_high_n(self) -> None:
        """At n=800 with HBLP-effective = 5σ (no relaxation), a 95%-strength
        leak MUST score z >> 5σ AND classify as severity=high."""
        score = self._run_adversarial_scoring(leak_strength=0.95, n_samples=800)
        assert score["z_score"] > 5.0, (
            f"strong leak z_score={score['z_score']:.2f} below 5σ; "
            "scorer is broken or leak isn't strong enough"
        )
        # n_positives at high n is well above 50 → no HBLP relaxation
        result = hblp_classify(
            score["z_score"],
            n_positives=int(target_positive_count(800)),
            layer_1_declared_safe=False,
        )
        assert result["severity"] == "high"
        assert result["hblp_relaxed"] is False

    def test_strong_leak_caught_at_low_n_no_layer_1_clear(self) -> None:
        """At low N (n=200, ~10 positives), HBLP raises bar to ~11σ but
        a near-perfect leak should still score z > 11σ."""
        score = self._run_adversarial_scoring(leak_strength=0.95, n_samples=200)
        # Verify the leak still scores HIGH z under the small-N null.
        # (At small N permutation-null variance is wider, so z is
        # smaller for the same effect — the leak STILL must clear HBLP.)
        n_positives = int(target_positive_count(200))
        result = hblp_classify(
            score["z_score"],
            n_positives=n_positives,
            layer_1_declared_safe=False,
        )
        assert result["severity"] == "high", (
            f"injected strong leak NOT caught: z={score['z_score']:.2f}, "
            f"effective_high={result['effective_high_threshold']:.2f}, "
            f"n_positives={n_positives}, severity={result['severity']!r}. "
            "Plan §6 Gate B1 invariant: HBLP must not mask injected leaks."
        )

    def test_strong_leak_at_low_n_with_layer_1_clear_documents_known_limit(
        self,
    ) -> None:
        """**Documents a known HBLP failure mode** per plan §7 risk register:

            "HBLP retains a feature whose leakage path is encoded in a
             way Layer 3.5 cannot detect at low N"

        At n=200 with declared_safe=True, HBLP-effective threshold reaches
        ~21σ (5σ * sqrt(50/10) * 1.5). A 95%-correlation leak produces
        z~17σ on this synthetic — BELOW the relaxed threshold → severity
        becomes 'moderate', NOT 'high'.

        This is EXPECTED behavior per the plan §7 mitigation: "lineage +
        leakage-injection regression + AdversarialProbe + conditional MI"
        is the defense-in-depth that catches what HBLP-relaxed alone
        misses. This test asserts the BOUNDARY, not 'high':
          - if severity becomes 'high' → HBLP is even safer than thought (good)
          - if severity becomes 'moderate' → expected; reviewer must check
            the AdversarialProbe / lineage audit results
          - if severity becomes 'info' → CRITICAL bug — leak completely
            masked
        """
        score = self._run_adversarial_scoring(leak_strength=0.95, n_samples=200)
        n_positives = int(target_positive_count(200))
        result = hblp_classify(
            score["z_score"],
            n_positives=n_positives,
            layer_1_declared_safe=True,
        )
        # Critical failure mode: leak completely masked.
        assert result["severity"] != "info", (
            f"CRITICAL: injected leak classified as INFO at HBLP-relaxed "
            f"threshold: z={score['z_score']:.2f}, effective_high="
            f"{result['effective_high_threshold']:.2f}. Plan §7 risk register "
            "flagged this failure mode but INFO would mean defense-in-depth "
            "is broken — HBLP relaxed beyond all reasonable bounds."
        )
        # Document the actual classification for the plan §7 risk register
        # observation: at low N + declared_safe, even strong leaks may
        # reach only 'moderate' under HBLP, requiring AdversarialProbe /
        # conditional MI as the secondary defense.
        assert result["severity"] in ("high", "moderate")


# --------------------------------------------------------------------------- #
# Clean feature — HBLP must NOT false-positive                                #
# --------------------------------------------------------------------------- #


class TestCleanFeatureNotMisclassified:
    """Plan §6 Tier 1B Gate B1: HBLP-relaxed thresholds must NOT label
    legitimate weak demographic signal as 'high'."""

    def test_legitimate_weak_signal_NOT_high_severity(self) -> None:
        """A 55%-correlation 'feature' is barely above chance — not a
        leak. HBLP at n=200 with Layer 1 unsafe should classify it as
        info (or at most moderate), NEVER high."""
        df, target = _build_clean_cohort(n_samples=200)
        df_with_weak = _inject_weak_leak(df, target, leak_strength=0.55)
        score = compute_adversarial_score(
            df_with_weak["WEAK_SIGNAL"].values,
            target,
            n_permutations=200,
            seed=42,
        )
        n_positives = int(target_positive_count(200))
        result = hblp_classify(
            score["z_score"],
            n_positives=n_positives,
            layer_1_declared_safe=False,
        )
        assert result["severity"] != "high", (
            f"weak legitimate signal MIS-flagged as high: z={score['z_score']:.2f}, "
            f"effective_high={result['effective_high_threshold']:.2f}. "
            "HBLP should not over-flag low-correlation features."
        )

    def test_pure_noise_feature_NOT_high_severity(self) -> None:
        """Random noise feature (no correlation with target) must score
        z near 0 → severity=info."""
        rng = np.random.default_rng(789)
        df, target = _build_clean_cohort(n_samples=400)
        noise_feature = rng.standard_normal(400)
        score = compute_adversarial_score(noise_feature, target, n_permutations=200, seed=42)
        # |z| should be small (~0-2) for pure noise.
        assert abs(score["z_score"]) < 5.0, (
            f"pure noise feature z={score['z_score']:.2f} >= 5σ — perm test "
            "is broken (random noise should not exceed 5σ)"
        )
        n_positives = int(target_positive_count(400))
        result = hblp_classify(
            score["z_score"],
            n_positives=n_positives,
            layer_1_declared_safe=False,
        )
        assert result["severity"] in ("info", "moderate")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def target_positive_count(n_samples: int) -> int:
    """Reproduce the target-positive count from `_build_clean_cohort`
    so tests can pass it to `hblp_classify`. Computed from the same
    DGP seed → deterministic."""
    _, target = _build_clean_cohort(n_samples=n_samples)
    return int(target.sum())
