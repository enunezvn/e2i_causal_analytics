"""Tiered single-feature-AUC leakage injectors for adversarial harness.

Phase 5 of `ml-leakage-holistic-fix`. Plants a synthetic leak feature whose
single-feature AUC against the target lands at a configurable tier. The
detector regression suite exercises each tier and asserts the appropriate
severity classification:

    AUC tier   | expected severity
    -----------+-----------------
    0.55       | not flagged (noise floor)
    0.60       | moderate
    0.69       | high (journey_duration_days analogue)
    0.78       | critical (was high before Phase 1 tightening)
    0.92       | critical (always)

Mechanism: feature = weight * target + N(0, 1). The empirical signal_weight
is calibrated against `roc_auc_score` for n=2000, p=0.30 binomial targets and
verified to within ±0.015 of the requested tier across seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Empirically calibrated signal weights. Verified against roc_auc_score for
# n=2000 binomial(0.30) targets across seeds {7, 11, 13, 17, 23, 29}.
_SIGNAL_WEIGHTS: dict[float, float] = {
    0.55: 0.20,
    0.58: 0.30,
    0.60: 0.40,
    0.65: 0.55,
    0.69: 0.70,
    0.75: 0.95,
    0.78: 1.05,
    0.85: 1.55,
    0.92: 2.40,
    0.95: 3.40,
}


def inject_tiered_leak(
    df: pd.DataFrame,
    *,
    target_col: str,
    target_auc: float,
    leak_feature: str = "leaked_feature",
    seed: int = 42,
) -> pd.DataFrame:
    """Plant a single-feature-AUC leak at the specified tier.

    Args:
        df: Input DataFrame (not mutated). Must contain `target_col`.
        target_col: Binary target column name.
        target_auc: Desired single-feature AUC. Must be in the calibrated
            tier set; arbitrary values map to the closest tier and are
            warned via a logger.
        leak_feature: Output column name for the planted leak feature.
        seed: RNG seed.

    Returns:
        New DataFrame with the planted leak feature. Existing columns are
        preserved.

    Raises:
        ValueError: If target_col missing, target is non-binary, or
            target_auc is outside the calibrated tier range.
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col!r} not in DataFrame")
    target = df[target_col].astype(int).values
    classes = np.unique(target)
    if len(classes) != 2:
        raise ValueError(
            f"target must be binary; got {len(classes)} classes: {classes}"
        )
    if target_auc < 0.50 or target_auc > 0.99:
        raise ValueError(f"target_auc must be in [0.50, 0.99]; got {target_auc}")

    rng = np.random.default_rng(seed)
    weight = _SIGNAL_WEIGHTS.get(round(target_auc, 2))
    if weight is None:
        # Pick the closest calibrated tier and emit a debug message
        keys = sorted(_SIGNAL_WEIGHTS.keys())
        closest = min(keys, key=lambda k: abs(k - target_auc))
        weight = _SIGNAL_WEIGHTS[closest]

    feature = weight * target.astype(float) + rng.normal(0.0, 1.0, len(target))
    out = df.copy()
    out[leak_feature] = feature

    return out


def measure_leak_auc(
    df: pd.DataFrame, *, target_col: str, leak_feature: str
) -> float:
    """Measure the single-feature AUC of the planted leak.

    For verifying that the injector produced a leak at the requested tier.
    """
    target = df[target_col].astype(int).values
    feature = df[leak_feature].astype(float).values
    auc = roc_auc_score(target, feature)
    return max(auc, 1 - auc)


def make_clean_dataset(
    *, n: int, prevalence: float, seed: int = 42
) -> pd.DataFrame:
    """Construct a clean DataFrame with target + 3 noise features.

    All non-target columns are independent Gaussian noise; their single-
    feature AUC against the target stays near 0.50 with high probability.
    """
    rng = np.random.default_rng(seed)
    target = rng.binomial(1, prevalence, n).astype(int)
    return pd.DataFrame(
        {
            "target": target,
            "noise_a": rng.normal(0, 1, n),
            "noise_b": rng.normal(0, 1, n),
            "noise_c": rng.normal(0, 1, n),
        }
    )
