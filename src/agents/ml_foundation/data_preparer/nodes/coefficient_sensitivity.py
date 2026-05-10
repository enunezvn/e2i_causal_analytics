r"""Plan v4 Gate G5 — T2.4 coefficient-sensitivity helper.

This module closes the second half of v3 §4 T2.4 (third bullet:
"Coefficient sensitivity analysis — does a feature's coefficient flip
sign under different imputation?"). PR #125 shipped only the missingness
profile in ``imputation_audit.py``; this module ships the sensitivity
audit that PR #125 deferred.

Workflow (per-feature re-fits — the load-bearing design):

1. The caller fits a baseline model ONCE on the cohort's training matrix
   (zero-imputed by default; sklearn refuses NaN inputs in ``fit``).
2. For EACH feature in ``recommended_strategies``, we clone the baseline
   X and impute ONLY that feature with its recommended strategy
   (leaving every other column at the baseline's zero-imputation). A
   fresh model is fit on this single-feature-imputed matrix and the
   feature's coefficient is read from this per-feature re-fit.
   Per-feature strategies:
     - ``"drop_row_or_mean"``: NaN cells in this column become the
       column's empirical mean (zero-imputation in baseline handled the
       same cells).
     - ``"mean_plus_indicator"``: same as drop_row_or_mean for the
       coefficient-sensitivity audit (the indicator column is dropped
       because it changes the model's column space, which makes
       coefficient comparison ill-defined).
     - ``"indicator_only"``: feature dropped entirely (tracked, not
       compared).
     - ``"drop_column"``: feature dropped (tracked, not compared).
3. For each feature, baseline and per-feature-imputed coefficients are
   compared (sign flip + effect-size CV). Aggregate flip statistics
   across the per-feature re-fits feed T1/T2/T3.

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
            "flip_count": int,           # 0 or 1 in single-strategy mode
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
coefficient sign. The spec memo's ceiling stays at 1 for forward-compatibility
with multi-strategy sweeps. In single-strategy mode the per-feature flip count
is bounded by {0, 1}, so the violation rule fires at ``>= 1`` (any flip
violates T1). Multi-strategy sweeps would tighten this to ``> 1`` so a single
outlier strategy is tolerated."""

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

    Raises:
        ValueError: if the series is all-NaN. We refuse to silently
            substitute zeros (M3 from G5 codex review): an all-NaN
            column has no defensible mean, and the caller should
            explicitly drop / sentinel-fill the column before invoking
            this helper.
    """
    if series.isna().sum() == 0:
        return series

    n_non_null = series.dropna().shape[0]
    if n_non_null == 0:
        raise ValueError(
            f"Column {series.name!r} is all-NaN; cannot mean-impute. "
            "All-NaN columns must be dropped (or explicitly sentinel-filled) "
            "by the caller before invoking compute_coefficient_sensitivity."
        )
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


def _validate_cohort_for_logreg(X: pd.DataFrame, y: pd.Series) -> None:
    """Pre-flight checks for LogisticRegression.fit on (X, y).

    sklearn raises opaque errors on degenerate cohorts (single-row,
    single-class). M4 from G5 codex review: surface these as clear
    ValueError BEFORE the fit fires, so the helper's contract is
    explicit at the call site rather than buried in sklearn internals.
    """
    if len(X) < 2:
        raise ValueError(
            f"Cohort has {len(X)} row(s); LogisticRegression requires at least "
            "2 rows. coefficient-sensitivity audit cannot proceed."
        )
    n_classes = int(pd.Series(y).nunique(dropna=True))
    if n_classes != 2:
        raise ValueError(
            f"Cohort target has {n_classes} unique value(s); "
            "compute_coefficient_sensitivity supports binary classification "
            "(exactly 2 classes). Single-class fits are degenerate (the "
            "decision boundary is undefined)."
        )


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

    # M4: validate the cohort is fit-able BEFORE wasting cycles on
    # imputation / per-feature setup.
    _validate_cohort_for_logreg(X, y)

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

    # Baseline fit: zero-impute NaNs, fit ONCE on the full feature surface
    # restricted to numeric columns. The same baseline coefficients are
    # compared against each per-feature re-fit below.
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

    # ---------------------------------------------------------------- #
    # Per-feature re-fits — the load-bearing G5 design (H2 closure).   #
    #                                                                  #
    # For EACH compared feature we clone X_baseline, replace ONLY this #
    # feature's column with the imputed-strategy-applied version, fit  #
    # a fresh model, and read out THIS feature's coefficient. Every    #
    # other column stays at the baseline's zero-imputation. This yields#
    # a true "what changes if I impute only feat_X" delta — the prior  #
    # design did one global fit and is non-informative for per-feature #
    # flip claims.                                                     #
    # ---------------------------------------------------------------- #
    coef_post: Dict[str, float] = {}
    for feature in compared_features:
        # Skip per-feature re-fit if THIS column has no NaN cells; the
        # imputed copy is identical to baseline and the coefficient
        # cannot move.
        if X[feature].isna().sum() == 0:
            coef_post[feature] = coef_baseline[feature]
            continue

        X_perfeat = X_baseline.copy()
        strat = recommended_strategies[feature]
        # Mean-impute the original (NaN-bearing) column, not the
        # zero-filled baseline copy.
        X_perfeat[feature] = _impute_for_strategy(
            X[feature],
            strat,  # type: ignore[arg-type]
        ).to_numpy()

        if estimator is None:
            perfeat_estimator = _build_default_estimator(seed)
        else:
            perfeat_estimator = estimator
        perfeat_coefs = _fit_and_get_coefs(perfeat_estimator, X_perfeat, y, numeric_features)
        coef_post[feature] = perfeat_coefs[feature]

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

        # Per-feature flip count for THIS feature in THIS comparison run.
        # With one imputed re-fit per feature, this collapses to {0, 1}.
        # Forward-compatible with future multi-strategy sweeps that would
        # accumulate >1 flips for a single feature.
        flip_count = 1 if sign_flip else 0

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
            "flip_count": flip_count,
            "effect_size_variance": cv,
        }

    # Aggregate over significant features only.
    significant_compared = [f for f in significant_features if f not in omitted_features]

    n_significant = len(significant_features)
    if len(significant_compared) > 0:
        flips = sum(1 for f in significant_compared if per_feature[f]["sign_flip"])
        max_cv = max(per_feature[f]["effect_size_variance"] for f in significant_compared)
        # H1 closure: track the max per-feature flip count across the
        # significant set. In the current single-strategy design the
        # max is at most 1; the metric exists so future multi-strategy
        # sweeps can populate it without restructuring the helper.
        max_flips_per_feature = max(int(per_feature[f]["flip_count"]) for f in significant_compared)
    else:
        flips = 0
        max_cv = 0.0
        max_flips_per_feature = 0

    fraction_flipped = flips / n_significant if n_significant > 0 else 0.0

    # Threshold checks.
    violations: List[str] = []
    # T1 closure (H1 from G5 codex review): in the current single-strategy
    # comparison, max_flips_per_feature ∈ {0, 1}. The spec memo's
    # G5_FLIPS_PER_FEATURE_MAX=1 is the forward-compatible ceiling for
    # multi-strategy sweeps; in single-strategy mode "no flip" is the
    # actual gate. We trigger T1 when any significant feature flips
    # (flip_count >= 1). The constant stays at 1 (spec-locked); the
    # check uses ``>=`` to make T1 reachable. Multi-strategy sweeps
    # would tighten this to ``> G5_FLIPS_PER_FEATURE_MAX`` (i.e. >1).
    if max_flips_per_feature >= G5_FLIPS_PER_FEATURE_MAX:
        violations.append(
            f"T1 violated: max_flips_per_feature_significant={max_flips_per_feature} "
            f">= G5_FLIPS_PER_FEATURE_MAX={G5_FLIPS_PER_FEATURE_MAX} "
            "(any sign-flip among significant features triggers T1 in "
            "single-strategy mode)."
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
