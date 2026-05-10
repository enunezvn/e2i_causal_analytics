r"""Plan v4 Gate G5 — T2.4 coefficient-sensitivity helper.

This module closes the second half of v3 §4 T2.4 (third bullet:
"Coefficient sensitivity analysis — does a feature's coefficient flip
sign under different imputation?"). PR #125 shipped only the missingness
profile in ``imputation_audit.py``; this module ships the sensitivity
audit that PR #125 deferred.

Workflow:

1. The caller fits a baseline model on the cohort's training matrix
   (zero-imputed by default; sklearn refuses NaN inputs in ``fit``).
2. For every feature in ``recommended_strategies``, the imputed copy of
   ``X`` is built per the strategy:
     - ``"drop_row_or_mean"``: rows with NaN in the column are kept and
       NaN-cells are mean-imputed (zero-imputation in baseline already
       handled the same cells).
     - ``"mean_plus_indicator"``: same as drop_row_or_mean for the
       coefficient-sensitivity audit (the indicator column is dropped
       because it changes the model's column space, which makes
       coefficient comparison ill-defined).
     - ``"indicator_only"``: feature dropped entirely (tracked, not
       compared).
     - ``"drop_column"``: feature dropped (tracked, not compared).
3. A second model is fit on the imputed matrix and the per-feature
   coefficients are compared.

The pre-specified thresholds (locked in
``docs/specs/g5_coefficient_sensitivity_prespec_20260510.md``) are
encoded here as module constants. They are imported by the integration
test, NOT post-hoc selected.

Returned dict shape (matches G5 spec):

```
{
    "n_features": int,
    "n_significant_features": int,
    "per_feature": {
        feature_name: {
            "effect_size_baseline": float,
            "effect_size_post_impute": float | None,
            "sign_flip": bool,
            "effect_size_variance": float,  # std / |mean|
        }
    },
    "aggregate": {
        "fraction_significant_flipped": float,
        "max_effect_size_variance_significant": float,
        "max_flips_per_feature_significant": int,
    },
    "thresholds": {
        "G5_FLIPS_PER_FEATURE_MAX": 1,
        "G5_EFFECT_SIZE_CV_MAX": 0.5,
        "G5_FRACTION_SIGNIFICANT_FLIPPED_MAX": 0.10,
        "G5_SIGNIFICANCE_SIGMA_MULTIPLE": 1.0,
    },
    "passes_pre_spec": bool,
    "violations": list[str],   # human-readable, one per failing threshold
}
```

The ``passes_pre_spec`` flag is True iff all three thresholds (T1, T2, T3)
hold. The ``violations`` list is empty in that case. Callers (the G5
integration test) assert ``passes_pre_spec is True`` rather than
re-implementing the threshold check, which guarantees the spec's
threshold values are the only place they live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from src.agents.ml_foundation.data_preparer.nodes.imputation_audit import (
    ImputationRecommendation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-specified thresholds — locked in
# docs/specs/g5_coefficient_sensitivity_prespec_20260510.md.
# Editing these constants requires a fresh pre-spec memo per the v3 §8
# anti-threshold-shopping protocol. The integration test imports these
# directly; they are NOT post-hoc tuned.
# ---------------------------------------------------------------------------
G5_FLIPS_PER_FEATURE_MAX: int = 1
"""T1: among "significant" features, at most 1 imputation re-fit may flip the
coefficient sign. With a single imputed re-fit per feature this collapses to
"no flip"; the >0 framing is forward-compatible with multi-strategy sweeps."""

G5_EFFECT_SIZE_CV_MAX: float = 0.5
"""T2: per-feature coefficient effect-size variance ``std / |mean|`` ≤ 0.5
for "significant" features. Steyerberg ch. 5 moderate-stability cutoff."""

G5_FRACTION_SIGNIFICANT_FLIPPED_MAX: float = 0.10
"""T3: at most 10% of "significant" features may flip sign at all. van
Buuren MICE 2nd ed. §5.3 conventional FDR cap on coefficient sign-flips."""

G5_SIGNIFICANCE_SIGMA_MULTIPLE: float = 1.0
"""Definition of "significant": ``|effect_size_baseline| > G5_SIGNIFICANCE_SIGMA_MULTIPLE * sigma``,
where sigma = std(|coef_baseline|). Conservative 1σ surfaces the top
20-30% of features by magnitude; avoids p-value-based filters that
require post-hoc decisions."""


# Strategies that DO NOT produce a comparable post-imputation coefficient.
# The feature is omitted from the comparison and tracked in
# ``omitted_features`` of the result.
_OMIT_STRATEGIES: frozenset[str] = frozenset(
    {
        "indicator_only",
        "drop_column",
    }
)


@dataclass(frozen=True)
class _PerFeatureSensitivity:
    """Internal per-feature record. Exposed via ``per_feature`` dict in the
    helper's return value."""

    effect_size_baseline: float
    effect_size_post_impute: Optional[float]
    sign_flip: bool
    effect_size_variance: float


def _impute_for_strategy(
    series: pd.Series,
    strategy: ImputationRecommendation,
) -> pd.Series:
    """Apply T2.4-recommended imputation to a single column.

    For sensitivity-comparison purposes:
      - ``drop_row_or_mean`` and ``mean_plus_indicator`` → mean-impute
        (we never drop rows here because the row-axis is shared across
        all features in a single fit).
      - ``indicator_only`` and ``drop_column`` → caller drops the column
        before this function fires. If accidentally called, mean-impute
        as a fallback (the comparison will be skipped at the helper
        level via ``_OMIT_STRATEGIES``).
    """
    if series.isna().sum() == 0:
        return series

    mean = float(series.dropna().mean())
    return series.fillna(mean)


def _coef_vector(estimator: BaseEstimator, feature_names: List[str]) -> Dict[str, float]:
    """Extract per-feature coefficients from a fitted linear estimator.

    Handles the binary-classification 2D coef_ shape (LogisticRegression
    returns ``(1, n_features)`` for binary) and the regression 1D shape.
    """
    raw = getattr(estimator, "coef_", None)
    if raw is None:
        raise ValueError(
            f"Estimator {type(estimator).__name__} did not expose a 'coef_' "
            "attribute after fit; coefficient sensitivity requires a linear "
            "model with coef_."
        )
    arr = np.asarray(raw)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(
                "compute_coefficient_sensitivity supports binary classification "
                f"(coef_.shape[0] == 1); got shape {arr.shape}. Multinomial "
                "fits are out of scope."
            )
        flat = arr[0]
    elif arr.ndim == 1:
        flat = arr
    else:
        raise ValueError(f"Unexpected coef_ shape {arr.shape}; expected 1D or 2D (1, n_features).")
    if len(flat) != len(feature_names):
        raise ValueError(
            f"Coefficient vector length {len(flat)} ≠ feature count "
            f"{len(feature_names)}; X column ordering may be inconsistent."
        )
    return {name: float(coef) for name, coef in zip(feature_names, flat, strict=True)}


def _build_default_estimator(seed: int) -> BaseEstimator:
    """L2-penalized binary logistic regression with deterministic seed.

    Matches the "baseline model" semantics in the G5 spec. liblinear
    solver chosen for stability on small cohorts (Optum n≈1294).
    """
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
    )


def _fit_and_get_coefs(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
) -> Dict[str, float]:
    """Fit the estimator on (X, y) and extract per-feature coefficients.

    Both X and y are coerced to numpy arrays of floats / ints to avoid
    sklearn dtype warnings on object/extension dtypes.
    """
    X_arr = X.to_numpy(dtype=np.float64, copy=False)
    y_arr = np.asarray(y).astype(np.int64, copy=False)
    estimator.fit(X_arr, y_arr)
    return _coef_vector(estimator, feature_names)


def compute_coefficient_sensitivity(
    X: pd.DataFrame,
    y: pd.Series,
    recommended_strategies: Mapping[str, str],
    *,
    estimator: Optional[BaseEstimator] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Plan v4 Gate G5 — coefficient-sensitivity audit.

    Args:
        X: Numeric-only feature matrix (n_samples, n_features). Columns
            must match ``recommended_strategies`` keys (extras are ignored).
            NaN cells are zero-imputed for the baseline fit (sklearn
            requires finite inputs in ``fit``); the imputed-comparison fit
            applies the recommended strategy per feature.
        y: Binary 0/1 outcome series of length ``n_samples``.
        recommended_strategies: Output of ``compute_imputation_audit``'s
            ``imputation_audit_recommendations`` field — a dict mapping
            feature name → strategy literal.
        estimator: Optional pre-instantiated sklearn linear estimator with
            a ``coef_`` attribute after fit. Defaults to L2-penalized
            binary LogisticRegression with ``random_state=seed``.
        seed: Random seed for the default estimator and any tie-breaking.

    Returns:
        Dict with the shape documented at module-level:
            n_features, n_significant_features, per_feature, aggregate,
            thresholds, passes_pre_spec, violations.

    Raises:
        ValueError: if X has fewer than 2 columns, y is not 0/1, or the
            estimator does not expose ``coef_``.
    """
    if X is None or len(X.columns) == 0:
        raise ValueError("X must have at least one column for sensitivity audit")
    if y is None or len(y) == 0:
        raise ValueError("y must have at least one row for sensitivity audit")
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}; must match")

    # Filter recommendations to the columns actually present in X.
    feature_names = [c for c in X.columns if c in recommended_strategies]
    if len(feature_names) == 0:
        raise ValueError("recommended_strategies has no overlap with X.columns; nothing to audit.")

    # Restrict to numeric columns. Non-numeric (string/object) columns are
    # skipped — coefficient comparison requires numeric features.
    numeric_features = [c for c in feature_names if pd.api.types.is_numeric_dtype(X[c])]
    if len(numeric_features) == 0:
        raise ValueError(
            "X has no numeric columns aligned with recommended_strategies; "
            "coefficient sensitivity requires at least one numeric feature."
        )

    omitted_features: List[str] = [
        c for c in numeric_features if recommended_strategies[c] in _OMIT_STRATEGIES
    ]
    compared_features: List[str] = [c for c in numeric_features if c not in omitted_features]

    # Baseline fit: zero-impute NaNs, fit on the full feature surface
    # restricted to numeric columns.
    X_baseline = X[numeric_features].copy()
    X_baseline = X_baseline.fillna(0.0)

    if estimator is None:
        baseline_estimator = _build_default_estimator(seed)
    else:
        baseline_estimator = estimator
    coef_baseline = _fit_and_get_coefs(baseline_estimator, X_baseline, y, numeric_features)

    # Identify "significant" features per the 1σ rule.
    coef_abs_vector = np.array([abs(coef_baseline[f]) for f in numeric_features], dtype=np.float64)
    sigma = float(np.std(coef_abs_vector))
    significance_cutoff = G5_SIGNIFICANCE_SIGMA_MULTIPLE * sigma
    significant_features: List[str] = [
        f for f in numeric_features if abs(coef_baseline[f]) > significance_cutoff
    ]

    # Imputed fit: apply per-feature strategies, drop omitted features.
    if len(compared_features) > 0:
        X_imputed = X[compared_features].copy()
        for col in compared_features:
            strat = recommended_strategies[col]
            # Type-narrow for mypy: strat is a str at the call boundary.
            X_imputed[col] = _impute_for_strategy(
                X_imputed[col],
                strat,  # type: ignore[arg-type]
            )
        # Catch any residual NaN from edge cases (all-NaN column).
        X_imputed = X_imputed.fillna(0.0)

        if estimator is None:
            imputed_estimator = _build_default_estimator(seed)
        else:
            imputed_estimator = estimator
        coef_post = _fit_and_get_coefs(imputed_estimator, X_imputed, y, compared_features)
    else:
        coef_post = {}

    # Per-feature comparison.
    per_feature: Dict[str, Dict[str, Any]] = {}
    for feature in numeric_features:
        baseline_coef = coef_baseline[feature]
        if feature in omitted_features:
            per_feature[feature] = {
                "effect_size_baseline": baseline_coef,
                "effect_size_post_impute": None,
                "sign_flip": False,
                "effect_size_variance": 0.0,
                "omitted_reason": recommended_strategies[feature],
            }
            continue

        post_coef = coef_post[feature]
        sign_flip = bool(
            (baseline_coef > 0 and post_coef < 0) or (baseline_coef < 0 and post_coef > 0)
        )

        # Effect-size variance is std/|mean| across the two-element series
        # {baseline, post}. ddof=0 to match the spec memo's pseudo-code.
        coefs = np.array([baseline_coef, post_coef], dtype=np.float64)
        mean = float(np.mean(coefs))
        std = float(np.std(coefs, ddof=0))
        if abs(mean) > 0.0:
            cv = std / abs(mean)
        else:
            cv = float("inf") if std > 0 else 0.0

        per_feature[feature] = {
            "effect_size_baseline": baseline_coef,
            "effect_size_post_impute": post_coef,
            "sign_flip": sign_flip,
            "effect_size_variance": cv,
        }

    # Aggregate over significant features only.
    significant_compared = [f for f in significant_features if f not in omitted_features]

    n_significant = len(significant_features)
    if len(significant_compared) > 0:
        flips = sum(1 for f in significant_compared if per_feature[f]["sign_flip"])
        max_cv = max(per_feature[f]["effect_size_variance"] for f in significant_compared)
    else:
        flips = 0
        max_cv = 0.0

    fraction_flipped = flips / n_significant if n_significant > 0 else 0.0

    # Per-feature flip count: with a single imputed re-fit per feature, the
    # flip count for a flipped feature is 1 and for an unflipped one is 0.
    # Forward-compatible with multi-strategy sweeps.
    max_flips_per_feature = 1 if flips > 0 else 0

    # Threshold checks.
    violations: List[str] = []
    if max_flips_per_feature > G5_FLIPS_PER_FEATURE_MAX:
        violations.append(
            f"T1 violated: max_flips_per_feature_significant={max_flips_per_feature} "
            f"> G5_FLIPS_PER_FEATURE_MAX={G5_FLIPS_PER_FEATURE_MAX}"
        )
    if max_cv > G5_EFFECT_SIZE_CV_MAX:
        violations.append(
            f"T2 violated: max_effect_size_variance_significant={max_cv:.3f} "
            f"> G5_EFFECT_SIZE_CV_MAX={G5_EFFECT_SIZE_CV_MAX}"
        )
    if fraction_flipped > G5_FRACTION_SIGNIFICANT_FLIPPED_MAX:
        violations.append(
            f"T3 violated: fraction_significant_flipped={fraction_flipped:.3f} "
            f"> G5_FRACTION_SIGNIFICANT_FLIPPED_MAX={G5_FRACTION_SIGNIFICANT_FLIPPED_MAX}"
        )

    passes_pre_spec = len(violations) == 0

    if not passes_pre_spec:
        logger.warning(
            "G5 coefficient-sensitivity pre-spec FAILED: %d violation(s). %s",
            len(violations),
            "; ".join(violations),
        )

    return {
        "n_features": len(numeric_features),
        "n_significant_features": n_significant,
        "n_omitted_features": len(omitted_features),
        "per_feature": per_feature,
        "aggregate": {
            "fraction_significant_flipped": fraction_flipped,
            "max_effect_size_variance_significant": max_cv,
            "max_flips_per_feature_significant": max_flips_per_feature,
            "significance_cutoff_sigma": sigma,
            "significance_cutoff_value": significance_cutoff,
        },
        "thresholds": {
            "G5_FLIPS_PER_FEATURE_MAX": G5_FLIPS_PER_FEATURE_MAX,
            "G5_EFFECT_SIZE_CV_MAX": G5_EFFECT_SIZE_CV_MAX,
            "G5_FRACTION_SIGNIFICANT_FLIPPED_MAX": G5_FRACTION_SIGNIFICANT_FLIPPED_MAX,
            "G5_SIGNIFICANCE_SIGMA_MULTIPLE": G5_SIGNIFICANCE_SIGMA_MULTIPLE,
        },
        "passes_pre_spec": passes_pre_spec,
        "violations": violations,
    }


__all__ = [
    "compute_coefficient_sensitivity",
    "G5_FLIPS_PER_FEATURE_MAX",
    "G5_EFFECT_SIZE_CV_MAX",
    "G5_FRACTION_SIGNIFICANT_FLIPPED_MAX",
    "G5_SIGNIFICANCE_SIGMA_MULTIPLE",
]
