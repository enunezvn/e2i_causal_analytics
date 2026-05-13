"""Risk-score model training + inference for prediction_synthesizer (Tier 4).

Closes GitHub issue #171 (PR C Sub-PR-B).

Public API:
    - ``RiskScoreTrainer``: XGBoost/LightGBM training with calibration and MLflow.
    - ``FORBIDDEN_FEATURE_SUBSTRINGS``: anti-leakage contract enforced by tests.
    - ``assert_no_leakage_in_features``: raises ``LeakageError`` if forbidden
      substrings appear in feature column names.

Target column (per issue #171 supervisor decision 2026-05-13):
    ``initiated_biologic_180d`` — broad initiation cohort, NOT
    ``discontinued_180d``.

Calibration acceptance (per issue body §3):
    Brier <= 0.20 AND ECE <= 0.10 on validation split. Fit calibration on
    validation (not train). If real data fails the bar, log it and surface;
    do NOT silently lower the bar.
"""

from __future__ import annotations

from .leakage_guard import (
    FORBIDDEN_FEATURE_SUBSTRINGS,
    LeakageError,
    assert_no_leakage_in_features,
    find_leaked_features,
)
from .risk_score_trainer import (
    CALIBRATION_BRIER_MAX,
    CALIBRATION_ECE_MAX,
    DEFAULT_MIN_AUC_PR,
    RISK_SCORE_HIGH_TIER,
    RISK_SCORE_LOW_TIER,
    RiskScoreTrainer,
    RiskScoreTrainingResult,
    expected_calibration_error,
    risk_score_to_tier,
)

__all__ = [
    "FORBIDDEN_FEATURE_SUBSTRINGS",
    "LeakageError",
    "assert_no_leakage_in_features",
    "find_leaked_features",
    "RiskScoreTrainer",
    "RiskScoreTrainingResult",
    "CALIBRATION_BRIER_MAX",
    "CALIBRATION_ECE_MAX",
    "DEFAULT_MIN_AUC_PR",
    "RISK_SCORE_HIGH_TIER",
    "RISK_SCORE_LOW_TIER",
    "expected_calibration_error",
    "risk_score_to_tier",
]
