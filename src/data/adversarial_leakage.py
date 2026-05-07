"""Layer 3 — Adversarial Leakage Discriminator.

Per-feature suspicion scoring via permutation-baseline-relative z-score.
The threshold for "suspicious" is DATA-DERIVED, not hardcoded — it adapts
to each cohort's null distribution automatically.

Why this replaces the hardcoded 0.65 / 0.80 thresholds:
- A feature with single-feature AUC 0.65 in a 200-patient low-prevalence
  cohort might be 2σ above the permutation null (legitimate weak signal).
- The same AUC 0.65 in a 5000-patient large cohort might be 8σ above the
  null (clear leakage).
- The hardcoded thresholds treated these the same. The z-score doesn't.

Disease-agnostic by construction: permutation tests work on any binary target.
No per-cohort tuning needed.

Reference: .claude/plans/adaptive_temporal_validity_redesign.md (Layer 3).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_adversarial_score(
    feature: np.ndarray | pd.Series,
    target: np.ndarray | pd.Series,
    *,
    n_permutations: int = 1000,
    seed: int = 42,
    z_threshold: float = 5.0,
) -> dict[str, Any]:
    """Score a feature's suspiciousness via permutation baseline.

    Args:
        feature: 1D array of feature values.
        target: 1D array of binary target values.
        n_permutations: Number of label shuffles to build the null distribution.
        seed: RNG seed for reproducibility.
        z_threshold: How many standard deviations above the null is "suspicious".
            Default 5σ — strict but not absolute. The threshold itself is
            documented and adjustable per use-case (governance), unlike the
            previous hardcoded AUC thresholds which had no statistical meaning.

    Returns:
        Dictionary with:
        - actual_auc: feature's effective AUC (max of auc, 1-auc)
        - null_mean: mean AUC under permuted labels
        - null_std: std of permuted AUC distribution
        - z_score: (actual_auc - null_mean) / null_std
        - p_value: fraction of permuted AUCs >= actual_auc
        - suspicious: True if z_score > z_threshold
        - n_permutations: actual number of permutations completed
    """
    feature_arr = np.asarray(feature, dtype=float)
    target_arr = np.asarray(target, dtype=int)

    # Compute the actual single-feature AUC (effective, max of auc and 1-auc)
    try:
        raw_auc = float(roc_auc_score(target_arr, feature_arr))
    except ValueError:
        # Degenerate cases: only one class in target, all-NaN feature, etc.
        return {
            "actual_auc": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "z_score": float("nan"),
            "p_value": float("nan"),
            "suspicious": False,
            "n_permutations": 0,
        }
    actual_auc = max(raw_auc, 1 - raw_auc)

    # Build the null distribution by shuffling target labels
    rng = np.random.default_rng(seed)
    null_aucs: list[float] = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(target_arr)
        try:
            null_raw = float(roc_auc_score(shuffled, feature_arr))
            null_aucs.append(max(null_raw, 1 - null_raw))
        except ValueError:
            continue

    if not null_aucs:
        return {
            "actual_auc": actual_auc,
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "z_score": float("nan"),
            "p_value": float("nan"),
            "suspicious": False,
            "n_permutations": 0,
        }

    null_arr = np.array(null_aucs)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))

    # Z-score against null distribution
    if null_std > 0:
        z_score = (actual_auc - null_mean) / null_std
    else:
        z_score = float("inf") if actual_auc > null_mean else 0.0

    # Two-sided test (permutation AUC is bounded [0.5, 1.0] after effective transform)
    p_value = float(np.mean(null_arr >= actual_auc))

    return {
        "actual_auc": actual_auc,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "p_value": p_value,
        "suspicious": z_score > z_threshold,
        "n_permutations": len(null_aucs),
    }


def compute_feature_ablation(
    X: pd.DataFrame,
    target: np.ndarray | pd.Series,
    *,
    model_factory: Any | None = None,
    n_permutations: int = 200,
    seed: int = 42,
    z_threshold: float = 5.0,
) -> dict[str, Any]:
    """Per-feature ablation: drop each feature, retrain, measure |delta_AUC|.

    A feature whose removal drops the model's AUC significantly is either a
    critical legitimate predictor OR a leak that the multi-feature model
    relies on. The output is descriptive — Layer 4's downstream judgment
    decides whether the dependency is legitimate or suspect.

    Args:
        X: Feature DataFrame (n_samples × n_features).
        target: 1D binary target array.
        model_factory: Callable returning a fresh sklearn-compatible classifier
            with predict_proba. Defaults to a small Logistic Regression for
            speed; production usage should pass the actual model class.
        n_permutations: Number of permutation rounds to build per-feature null
            of |delta_AUC| (smaller than discriminator since each round
            requires retraining).
        seed: RNG seed.
        z_threshold: Same as compute_adversarial_score.

    Returns:
        Dictionary with:
        - full_auc: AUC of model trained on ALL features
        - per_feature: list of dicts, one per feature, with:
            - feature: name
            - delta_auc: full_auc - auc_without_feature (positive = feature helps)
            - z_score: z-score of delta_auc against permutation null
            - suspicious: True if z_score > z_threshold
    """
    if model_factory is None:
        from sklearn.linear_model import LogisticRegression

        def model_factory():  # noqa: E306
            return LogisticRegression(max_iter=200, random_state=seed)

    X_arr = X.copy()
    y_arr = np.asarray(target, dtype=int)
    feature_names = list(X_arr.columns)

    # Train the full model
    full_model = model_factory()
    full_model.fit(X_arr.values, y_arr)
    full_auc = float(roc_auc_score(y_arr, full_model.predict_proba(X_arr.values)[:, 1]))

    rng = np.random.default_rng(seed)
    per_feature_results: list[dict[str, Any]] = []

    for feat_name in feature_names:
        # Train without this feature
        X_minus = X_arr.drop(columns=[feat_name]).values
        try:
            ablated_model = model_factory()
            ablated_model.fit(X_minus, y_arr)
            ablated_auc = float(roc_auc_score(y_arr, ablated_model.predict_proba(X_minus)[:, 1]))
        except (ValueError, RuntimeError):
            ablated_auc = float("nan")
        delta_auc = full_auc - ablated_auc if not np.isnan(ablated_auc) else float("nan")

        # Permutation null for delta_auc: shuffle the FEATURE column, retrain,
        # measure delta. Smaller n_permutations because of training cost.
        null_deltas: list[float] = []
        for _ in range(n_permutations):
            shuffled_feat = rng.permutation(X_arr[feat_name].values)
            X_perm = X_arr.copy()
            X_perm[feat_name] = shuffled_feat
            try:
                perm_model = model_factory()
                perm_model.fit(X_perm.values, y_arr)
                perm_auc = float(
                    roc_auc_score(y_arr, perm_model.predict_proba(X_perm.values)[:, 1])
                )
                null_deltas.append(full_auc - perm_auc)
            except (ValueError, RuntimeError):
                continue

        null_mean = float(np.mean(null_deltas)) if null_deltas else float("nan")
        null_std = float(np.std(null_deltas)) if null_deltas else float("nan")
        z_score = (
            (delta_auc - null_mean) / null_std
            if null_std and null_std > 0 and not np.isnan(delta_auc)
            else float("nan")
        )

        per_feature_results.append(
            {
                "feature": feat_name,
                "full_auc": full_auc,
                "ablated_auc": ablated_auc,
                "delta_auc": delta_auc,
                "null_mean": null_mean,
                "null_std": null_std,
                "z_score": z_score,
                "suspicious": z_score > z_threshold if not np.isnan(z_score) else False,
            }
        )

    return {
        "full_auc": full_auc,
        "n_features": len(feature_names),
        "n_permutations": n_permutations,
        "per_feature": per_feature_results,
    }
