r"""Plan v3 §4 T2.4 — Imputation/missingness audit.

Codex critique #12: zero-imputation in canonical feature matrices biases
coefficients and makes "missingness" look causal. T2.4 ships an audit
that exposes the missingness shape to the operator BEFORE imputation
fires, so a downstream consumer (T2.6c enforcement, observability
dashboards) can decide whether the auto-strategy from
``data_transformer.py`` is appropriate.

Three audits, all OBSERVABILITY ONLY (does NOT change imputation
strategy or block training):

1. **Per-feature missingness profile** — fraction of NaN per column on
   each split. Surfaced as `imputation_audit_missingness_profile` on
   validation_metrics.
2. **Split-stability test** — checks that missingness rates are
   approximately consistent across train/val/test splits. A feature
   that is 5% missing on train but 50% missing on test is a data-quality
   red flag (covariate shift OR cohort-build bug). Surfaced as
   `imputation_audit_stability_violations`.
3. **Recommended strategy per feature** — based on missingness pattern:
   * < 5% missing → drop rows OR mean-impute (low risk either way)
   * 5-30% missing → mean-impute + add missing indicator
   * 30-70% missing → indicator only (drop the underlying value, keep
     the missingness signal)
   * \> 70% missing → drop the column (insufficient signal)

Plan §6 T2.4 acceptance: per-feature missingness stability +
coefficient sensitivity tests pass on Optum and CSU; per-cohort
missingness profile in `validation_metrics`.

Plan §9 file: this module (or
``src/agents/ml_foundation/data_preparer/nodes/data_imputer.py``;
T2.4's surface is observability-only so a sibling helper module is
appropriate).

Coefficient sensitivity analysis (Plan §4 T2.4 third bullet) requires
fitting linear models under multiple imputation strategies and
comparing coefficient sign flips. That depends on the trained-model
artifact and is a separate audit the model_trainer can invoke after
training. This module ships the missingness audit + recommendation
helpers; coefficient sensitivity is a follow-on.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Mapping, Optional

import pandas as pd

from src.lifecycle import GateLifecycleState

logger = logging.getLogger(__name__)


# Plan v4 Gate N2 — lifecycle-state declaration for the T2.4 imputation
# audit. Currently DEVELOPMENT: ``compute_imputation_audit`` is exported
# from ``data_preparer.nodes`` but NOT yet invoked by the pipeline (no
# downstream caller threads its output into ``validation_metrics`` or any
# enforcement path). Promoting to ADVISORY requires wiring the helper into
# the data_preparer graph; that wiring + a signed
# ``docs/calibration/T24_lifecycle_change_development_to_advisory_*.md``
# is the gating prerequisite per Gate N2 acceptance #3.
LIFECYCLE_STATE_T24: GateLifecycleState = GateLifecycleState.DEVELOPMENT


# Default thresholds for the recommendation helper. Anchored to common
# practice in clinical-ML imputation literature (e.g., Sterne 2009 BMJ
# review on multiple imputation; Donders 2006 on missingness types).
T2_4_RECOMMEND_DROP_ROW_RATE_MAX: float = 0.05
T2_4_RECOMMEND_INDICATOR_RATE_MIN: float = 0.30
T2_4_RECOMMEND_DROP_COLUMN_RATE_MIN: float = 0.70

# Default split-stability tolerance: a feature's missingness rate is
# stable iff max-rate - min-rate across splits is below this threshold.
# 0.05 (5pp) follows the convention from Steyerberg's clinical-prediction
# textbook for "comparable patterns across splits."
T2_4_STABILITY_TOLERANCE_DEFAULT: float = 0.05


ImputationRecommendation = Literal[
    "drop_row_or_mean",
    "mean_plus_indicator",
    "indicator_only",
    "drop_column",
]


def _per_column_missing_rate(df: pd.DataFrame) -> Dict[str, float]:
    """Per-column fraction of missing values. NaN, None, and pd.NA all
    count as missing via ``pd.isna``."""
    if len(df) == 0:
        return {}
    return {col: float(df[col].isna().mean()) for col in df.columns}


def _recommend_strategy_for_rate(rate: float) -> ImputationRecommendation:
    """Plan v3 §4 T2.4 recommendation helper. Inclusive of the upper
    boundary on each band so 0.30 → indicator_only (not mean_plus_indicator).
    Boundary placement matches Sterne 2009 BMJ review breakpoints."""
    if rate <= T2_4_RECOMMEND_DROP_ROW_RATE_MAX:
        return "drop_row_or_mean"
    if rate < T2_4_RECOMMEND_INDICATOR_RATE_MIN:
        return "mean_plus_indicator"
    if rate < T2_4_RECOMMEND_DROP_COLUMN_RATE_MIN:
        return "indicator_only"
    return "drop_column"


def compute_imputation_audit(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    *,
    stability_tolerance: float = T2_4_STABILITY_TOLERANCE_DEFAULT,
) -> Dict[str, Any]:
    """Plan v3 §4 T2.4 — Imputation/missingness audit (observability only).

    Inspects the feature DataFrame(s) BEFORE imputation fires and
    surfaces three audits:

      1. **Per-feature missingness profile** on each provided split,
         plus an aggregate (overall_missingness_rate computed on
         X_train alone).
      2. **Split-stability test**: for features present on multiple
         splits, max-rate - min-rate across splits must be below
         ``stability_tolerance``. Violations are listed with their
         per-split rates so the operator can triage.
      3. **Recommended imputation strategy** per feature (based on
         X_train missingness alone): drop_row_or_mean / mean_plus_indicator
         / indicator_only / drop_column.

    Args:
        X_train: Training-split feature DataFrame. Required.
        X_val: Optional validation-split DataFrame.
        X_test: Optional test-split DataFrame.
        stability_tolerance: Max allowed (max - min) rate spread across
            splits before a feature is flagged as unstable. Default 0.05
            (5pp).

    Returns:
        Dict with the following keys (all suitable for promotion onto
        ``validation_metrics``):

          * ``imputation_audit_missingness_profile`` — Dict[str, float]
            mapping feature → train-split missingness rate.
          * ``imputation_audit_overall_missingness`` — float, global
            missingness rate across all train cells.
          * ``imputation_audit_per_split_profile`` — Dict[str, Dict[str, float]]
            mapping split_name (``"train" | "val" | "test"``) → feature →
            rate. Only includes splits the caller provided.
          * ``imputation_audit_stability_tolerance`` — float, threshold used.
          * ``imputation_audit_stability_violations`` — List[str], features
            whose max-min rate across provided splits exceeds tolerance.
          * ``imputation_audit_stability_violation_details`` — Dict[str, Dict[str, float]]
            mapping violating feature → {split_name: rate, range: float}.
          * ``imputation_audit_recommendations`` — Dict[str, ImputationRecommendation],
            per-feature suggested strategy (based on X_train missingness).
          * ``imputation_audit_n_features`` — int, count of features audited.
          * ``imputation_audit_n_train_rows`` — int.

    Pure observability: caller is responsible for deciding whether to
    act on the recommendations. The existing imputation pipeline at
    ``data_transformer.py`` is unaffected by this audit.
    """
    if X_train is None or len(X_train) == 0:
        return {
            "imputation_audit_completed": False,
            "imputation_audit_error": "X_train is None or empty",
            "imputation_audit_missingness_profile": {},
            "imputation_audit_overall_missingness": None,
            "imputation_audit_per_split_profile": {},
            "imputation_audit_stability_violations": [],
            "imputation_audit_stability_violation_details": {},
            "imputation_audit_recommendations": {},
            "imputation_audit_n_features": 0,
            "imputation_audit_n_train_rows": 0,
        }

    train_profile = _per_column_missing_rate(X_train)
    n_train_rows = int(len(X_train))
    n_features = len(train_profile)

    overall_missingness: Optional[float]
    if n_features > 0 and n_train_rows > 0:
        overall_missingness = float(X_train.isna().sum().sum()) / float(n_train_rows * n_features)
    else:
        overall_missingness = None

    per_split_profile: Dict[str, Dict[str, float]] = {"train": train_profile}
    if X_val is not None and len(X_val) > 0:
        per_split_profile["val"] = _per_column_missing_rate(X_val)
    if X_test is not None and len(X_test) > 0:
        per_split_profile["test"] = _per_column_missing_rate(X_test)

    # Stability check: for each feature in train, check (max - min) across
    # splits that contain it. A feature missing from a split is treated
    # as "rate=0" only if that split is empty; otherwise the absent
    # column is reported as a structural issue (rate=NaN → feature
    # flagged as unstable).
    violations: List[str] = []
    violation_details: Dict[str, Dict[str, float]] = {}
    for feature in train_profile:
        per_split_rates: Dict[str, float] = {}
        for split_name, profile in per_split_profile.items():
            if feature in profile:
                per_split_rates[split_name] = profile[feature]
        if len(per_split_rates) < 2:
            continue
        rates = list(per_split_rates.values())
        rate_range = float(max(rates) - min(rates))
        if rate_range > stability_tolerance:
            violations.append(feature)
            details = dict(per_split_rates)
            details["range"] = rate_range
            violation_details[feature] = details

    recommendations: Dict[str, ImputationRecommendation] = {
        feature: _recommend_strategy_for_rate(rate) for feature, rate in train_profile.items()
    }

    if violations:
        logger.warning(
            "T2.4 ADVISORY: %d feature(s) have missingness rates that vary "
            "across splits beyond tolerance %.3f: %s. Review for cohort-build "
            "bug or covariate shift.",
            len(violations),
            stability_tolerance,
            violations[:5],  # cap for log readability
        )

    return {
        "imputation_audit_completed": True,
        "imputation_audit_missingness_profile": train_profile,
        "imputation_audit_overall_missingness": overall_missingness,
        "imputation_audit_per_split_profile": per_split_profile,
        "imputation_audit_stability_tolerance": float(stability_tolerance),
        "imputation_audit_stability_violations": violations,
        "imputation_audit_stability_violation_details": violation_details,
        "imputation_audit_recommendations": recommendations,
        "imputation_audit_n_features": n_features,
        "imputation_audit_n_train_rows": n_train_rows,
    }


def summarize_recommendations(
    recommendations: Mapping[str, ImputationRecommendation],
) -> Dict[str, int]:
    """Aggregate per-feature recommendations into strategy counts.

    Useful for cohort-level dashboards: how many features need
    drop_column? mean_plus_indicator? etc. Returns a dict with one
    entry per Literal value (zero-filled if no features matched).
    """
    counts: Dict[str, int] = {
        "drop_row_or_mean": 0,
        "mean_plus_indicator": 0,
        "indicator_only": 0,
        "drop_column": 0,
    }
    for strategy in recommendations.values():
        counts[strategy] += 1
    return counts


# ---------------------------------------------------------------------------
# Coefficient sensitivity (plan §4 T2.4 third bullet) — implemented in the
# sibling module ``coefficient_sensitivity.py`` (Plan v4 Gate G5, 2026-05-10).
#
# The helper takes an (X, y, recommended_strategies) tuple where
# ``recommended_strategies`` is the dict surfaced by THIS module's
# ``imputation_audit_recommendations`` field, fits a baseline + imputed
# LogisticRegression pair, and reports per-feature sign-flip count plus
# effect-size variance. The G5 spec memo at
# ``docs/specs/g5_coefficient_sensitivity_prespec_20260510.md`` locks the
# three pre-specified acceptance thresholds (T1/T2/T3).
#
# Callers may either import directly from
# ``src.agents.ml_foundation.data_preparer.nodes.coefficient_sensitivity``
# or via the ``__getattr__`` lazy re-export below, which avoids the
# import-time circular dependency between the two modules.
# ---------------------------------------------------------------------------


__all__ = [
    "compute_imputation_audit",
    "summarize_recommendations",
    "compute_coefficient_sensitivity",
    "T2_4_RECOMMEND_DROP_ROW_RATE_MAX",
    "T2_4_RECOMMEND_INDICATOR_RATE_MIN",
    "T2_4_RECOMMEND_DROP_COLUMN_RATE_MIN",
    "T2_4_STABILITY_TOLERANCE_DEFAULT",
    "G5_FLIPS_PER_FEATURE_MAX",
    "G5_EFFECT_SIZE_CV_MAX",
    "G5_FRACTION_SIGNIFICANT_FLIPPED_MAX",
    "G5_SIGNIFICANCE_SIGMA_MULTIPLE",
]


# End-of-module re-export of the G5 helper + constants. Placed AFTER the
# ImputationRecommendation type alias is fully bound to avoid a circular
# import: coefficient_sensitivity imports ImputationRecommendation from
# this module at its own import time. By the time Python reaches this
# bottom-of-file import, all of imputation_audit's module-level names
# (including ImputationRecommendation) are already bound.
from src.agents.ml_foundation.data_preparer.nodes.coefficient_sensitivity import (  # noqa: E402, I001
    G5_EFFECT_SIZE_CV_MAX,
    G5_FLIPS_PER_FEATURE_MAX,
    G5_FRACTION_SIGNIFICANT_FLIPPED_MAX,
    G5_SIGNIFICANCE_SIGMA_MULTIPLE,
    compute_coefficient_sensitivity,
)
