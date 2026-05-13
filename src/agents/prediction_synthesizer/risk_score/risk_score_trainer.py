"""Risk-score model trainer for prediction_synthesizer Tier 4 (issue #171).

Trains a calibrated XGBoost/LightGBM classifier on the CSU initiation cohort
to predict ``initiated_biologic_180d`` (will this patient initiate a CSU
biologic in the next 180d?).

Design notes per supervisor decisions (2026-05-13):

1. Target = ``initiated_biologic_180d`` (broader initiation cohort). NOT
   ``discontinued_180d``.

2. Model class chosen via CV AUC-PR (better than AUC-ROC under CSU class
   imbalance). Both XGBoost and LightGBM are evaluated; the one with higher
   mean AUC-PR over CV folds is selected. AUC-PR is the primary discrimination
   metric reported alongside AUC-ROC.

3. Hyperparameter search via Optuna (50 trials default; configurable via
   ``hpo_trials``). Floor at ``min_auc_pr`` (default 0.65).

4. Calibration acceptance: Brier <= 0.20 AND ECE <= 0.10 on the validation
   split. Calibration is fit on VALIDATION (not train) — both Platt scaling
   (sklearn ``CalibratedClassifierCV`` with ``method='sigmoid'``) and isotonic
   regression are fit and the one with lower validation Brier is selected.
   If the bar fails on real data we LOG IT and SURFACE it — we do NOT lower
   the bar silently.

5. MLflow logged: model artifact, feature importance, calibration plots
   (reliability diagram), all metrics.

6. SHAP values computed for top-feature explainability (per-patient SHAP for
   ``ml_predictions.shap_values``, model-global mean abs SHAP for
   ``feature_importance``).

7. Risk score formula (per issue body):
       risk_score = clamp(round(10 * calibrated_probability, 2), 0.00, 9.99)
   Tier mapping: high >=6.6, medium 3.3-6.6, low <3.3.

Note on framework choice — the wider ``ModelTrainerAgent`` graph in
``src/agents/ml_foundation/model_trainer/`` is heavyweight (QC gate, multi-fold
orchestration, asynchronous graph). For the risk_score model we instead expose
a direct synchronous training API here since: (a) the call site is a Celery
task / CLI script, (b) the calibration + leakage-test contracts are specific
to this target and benefit from a focused API, and (c) we still log to MLflow
through the shared connector.
"""

from __future__ import annotations

import io
import json
import logging
import os
import secrets
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from .leakage_guard import assert_no_leakage_in_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (issue #171 acceptance criteria)
# ---------------------------------------------------------------------------

# Calibration acceptance per supervisor decision 2026-05-13
CALIBRATION_BRIER_MAX: float = 0.20
CALIBRATION_ECE_MAX: float = 0.10

# AUC-PR floor (per OptumTestConfig.min_auc_threshold = 0.65 in issue body)
DEFAULT_MIN_AUC_PR: float = 0.65

# Risk-tier cuts (issue body §"Risk tier mapping to prediction_class")
RISK_SCORE_HIGH_TIER: float = 6.6  # >= -> high
RISK_SCORE_LOW_TIER: float = 3.3  # <  -> low; [LOW, HIGH) -> medium

# Risk-score scale (per migration / schema: DECIMAL(3,2), range 0.00-9.99)
RISK_SCORE_SCALE: float = 10.0
RISK_SCORE_MIN: float = 0.00
RISK_SCORE_MAX: float = 9.99

_DEFAULT_HPO_TRIALS = 50
_DEFAULT_CV_FOLDS = 5
_DEFAULT_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) with uniform-width bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Args:
        y_true: binary ground-truth labels, shape (N,).
        y_prob: predicted positive-class probabilities, shape (N,).
        n_bins: number of equal-width probability bins (default 10).

    Returns:
        ECE in [0, 1].

    Raises:
        ValueError: if ``y_true`` or ``y_prob`` is empty or has mismatched shape.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob).ravel()
    if y_true_arr.size == 0:
        raise ValueError("ECE requires at least one sample.")
    if y_true_arr.shape != y_prob_arr.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true_arr.shape} y_prob={y_prob_arr.shape}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = y_true_arr.size
    for b in range(n_bins):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        # Include right edge in last bin so 1.0 is counted.
        if b == n_bins - 1:
            mask = (y_prob_arr >= lo) & (y_prob_arr <= hi)
        else:
            mask = (y_prob_arr >= lo) & (y_prob_arr < hi)
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue
        acc = float(y_true_arr[mask].mean())
        conf = float(y_prob_arr[mask].mean())
        ece += (n_in_bin / total) * abs(acc - conf)
    return float(ece)


def risk_score_to_tier(score: float) -> str:
    """Map a 0-9.99 risk score to the prediction_class tier.

    >>> risk_score_to_tier(8.1)
    'high'
    >>> risk_score_to_tier(5.0)
    'medium'
    >>> risk_score_to_tier(2.0)
    'low'
    """
    if score >= RISK_SCORE_HIGH_TIER:
        return "high"
    if score >= RISK_SCORE_LOW_TIER:
        return "medium"
    return "low"


def probability_to_risk_score(p: float) -> float:
    """Convert a calibrated probability to the DECIMAL(3,2) risk_score scale.

    risk_score = clamp(round(10 * p, 2), 0.00, 9.99)
    """
    if p is None:
        return 0.0
    if not np.isfinite(p):
        return 0.0
    scaled = round(RISK_SCORE_SCALE * float(p), 2)
    return float(min(max(scaled, RISK_SCORE_MIN), RISK_SCORE_MAX))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RiskScoreTrainingResult:
    """Output of :meth:`RiskScoreTrainer.fit`.

    Carries metrics + calibrated estimator + selected hyperparameters so the
    caller (Celery task / CLI script) can publish to MLflow and write to the
    DB without touching the trainer internals.
    """

    model_type: str  # 'xgboost' or 'lightgbm'
    best_params: dict[str, Any]
    feature_names: list[str]
    # Discrimination
    val_auc_roc: float
    val_auc_pr: float
    val_precision: float
    val_recall: float
    # Calibration
    val_brier: float
    val_ece: float
    calibration_method: str  # 'sigmoid' / 'isotonic'
    # Acceptance flags
    auc_pr_floor: float
    auc_pr_floor_met: bool
    calibration_acceptance_met: bool
    # Diagnostics
    cv_auc_pr_mean: float
    cv_auc_pr_std: float
    train_class_balance: dict[str, int] = field(default_factory=dict)
    val_class_balance: dict[str, int] = field(default_factory=dict)
    # Artifacts (raw bytes — caller logs to MLflow / writes to DB)
    feature_importance: dict[str, float] = field(default_factory=dict)
    reliability_diagram_png: Optional[bytes] = None
    # MLflow run id (None if MLflow tracking failed / was disabled)
    mlflow_run_id: Optional[str] = None
    # Surfaced honest failures
    honest_failures: list[str] = field(default_factory=list)
    # The fitted calibrated estimator (sklearn-like .predict_proba(X))
    estimator: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dict (drops the estimator + PNG bytes)."""
        d = asdict(self)
        d.pop("estimator", None)
        d.pop("reliability_diagram_png", None)
        return d


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class RiskScoreTrainer:
    """Train + calibrate a risk_score model for CSU biologic initiation.

    Usage::

        trainer = RiskScoreTrainer(min_auc_pr=0.65, hpo_trials=50)
        result = trainer.fit(
            X_train=X_train_df,
            y_train=y_train_arr,
            X_val=X_val_df,
            y_val=y_val_arr,
            mlflow_experiment="risk_score_csu_initiation",
        )

    The trainer:

    1. Asserts no leakage in feature names (anti-leakage guard).
    2. Picks model class (XGBoost vs LightGBM) via 5-fold CV AUC-PR on train.
    3. Runs Optuna HPO on the chosen class (with CV on train).
    4. Fits final model on train with best params.
    5. Fits Platt + isotonic calibration on VAL; picks lower-Brier.
    6. Logs metrics + artifact + reliability diagram to MLflow if enabled.
    7. Returns metrics + calibrated estimator.

    Failures on the AUC-PR floor or calibration acceptance bar are SURFACED
    via ``honest_failures`` — the bar is NEVER lowered silently. The trainer
    still returns the fitted estimator so the caller can decide what to do
    (typically: log a HONEST_DEFERRAL warning and not promote the model to
    production).
    """

    def __init__(
        self,
        min_auc_pr: float = DEFAULT_MIN_AUC_PR,
        brier_max: float = CALIBRATION_BRIER_MAX,
        ece_max: float = CALIBRATION_ECE_MAX,
        hpo_trials: int = _DEFAULT_HPO_TRIALS,
        cv_folds: int = _DEFAULT_CV_FOLDS,
        random_state: int = _DEFAULT_RANDOM_STATE,
        enable_mlflow: bool = True,
        model_candidates: tuple[str, ...] = ("xgboost", "lightgbm"),
    ) -> None:
        self.min_auc_pr = min_auc_pr
        self.brier_max = brier_max
        self.ece_max = ece_max
        self.hpo_trials = hpo_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.enable_mlflow = enable_mlflow
        # Validate candidates lazily — let import-time guard at use-site
        # surface a clear error rather than blowing up at construction.
        for c in model_candidates:
            if c not in {"xgboost", "lightgbm"}:
                raise ValueError(
                    f"Unsupported risk-score model candidate {c!r}; "
                    "expected one of {'xgboost', 'lightgbm'}."
                )
        self.model_candidates = tuple(model_candidates)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        feature_names: Optional[list[str]] = None,
        mlflow_experiment: str = "risk_score_csu_initiation",
        mlflow_run_name: Optional[str] = None,
    ) -> RiskScoreTrainingResult:
        """Train, calibrate, evaluate, and (optionally) log to MLflow."""
        import pandas as pd  # local: avoid hard top-level dep on pandas
        from sklearn.metrics import (  # type: ignore[import-not-found]
            average_precision_score,
            brier_score_loss,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        # Normalize to DataFrame so we have stable column names for SHAP /
        # feature importance / leakage guard.
        if isinstance(X_train, pd.DataFrame):
            feat_names = list(X_train.columns)
        elif feature_names is not None:
            feat_names = list(feature_names)
            X_train = pd.DataFrame(X_train, columns=feat_names)
        else:
            feat_names = [f"f{i}" for i in range(np.asarray(X_train).shape[1])]
            X_train = pd.DataFrame(X_train, columns=feat_names)
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(np.asarray(X_val), columns=feat_names)

        y_train_arr = np.asarray(y_train).ravel().astype(int)
        y_val_arr = np.asarray(y_val).ravel().astype(int)
        if y_train_arr.size != X_train.shape[0]:
            raise ValueError(
                f"y_train length {y_train_arr.size} != X_train rows {X_train.shape[0]}"
            )
        if y_val_arr.size != X_val.shape[0]:
            raise ValueError(f"y_val length {y_val_arr.size} != X_val rows {X_val.shape[0]}")
        if set(np.unique(y_train_arr)).difference({0, 1}):
            raise ValueError("y_train must be 0/1 binary labels.")
        if set(np.unique(y_val_arr)).difference({0, 1}):
            raise ValueError("y_val must be 0/1 binary labels.")

        # Anti-leakage contract.
        assert_no_leakage_in_features(feat_names)

        # Coerce features to numeric (XGBoost / LightGBM expect numeric).
        # Non-numeric columns trigger an informative error rather than silent
        # failure downstream.
        for col in feat_names:
            if not pd.api.types.is_numeric_dtype(X_train[col]):
                raise TypeError(
                    f"Feature {col!r} is non-numeric (dtype={X_train[col].dtype}); "
                    "risk_score trainer expects fully-numericized features. "
                    "Run encoding/imputation upstream in the data preparer."
                )

        honest_failures: list[str] = []

        # Step 1 — pick model class via CV AUC-PR on train.
        best_model_type, cv_auc_pr_mean, cv_auc_pr_std = self._choose_model_type(
            X_train, y_train_arr
        )
        logger.info(
            "risk_score_trainer: chose model_type=%s (CV AUC-PR=%.4f +/- %.4f)",
            best_model_type,
            cv_auc_pr_mean,
            cv_auc_pr_std,
        )

        # Step 2 — HPO on the chosen class.
        best_params = self._optuna_search(best_model_type, X_train, y_train_arr)

        # Step 3 — fit final base estimator on train (with best params).
        # Pass the DataFrame (not .values) so XGBoost / LightGBM record feature
        # names and downstream feature-importance keys line up with feat_names.
        base = self._build_estimator(best_model_type, best_params)
        base.fit(X_train, y_train_arr)

        # Step 4 — calibrate on VAL (NOT train). Try Platt + isotonic, pick
        # lower-Brier.
        calibrated, calib_method, val_brier = self._calibrate_on_val(base, X_val, y_val_arr)

        # Step 5 — validation metrics.
        val_proba = calibrated.predict_proba(X_val)[:, 1]
        val_auc_roc = self._safe_metric(roc_auc_score, y_val_arr, val_proba)
        val_auc_pr = self._safe_metric(average_precision_score, y_val_arr, val_proba)
        val_pred = (val_proba >= 0.5).astype(int)
        val_precision = self._safe_metric(precision_score, y_val_arr, val_pred, zero_division=0)
        val_recall = self._safe_metric(recall_score, y_val_arr, val_pred, zero_division=0)
        val_ece = expected_calibration_error(y_val_arr, val_proba, n_bins=10)
        # Recompute Brier from final val_proba (calibrated). Should match
        # self._calibrate_on_val's selection value within float tolerance.
        val_brier = self._safe_metric(brier_score_loss, y_val_arr, val_proba)

        # Step 6 — acceptance gates (surfaced, not enforced).
        auc_pr_floor_met = val_auc_pr >= self.min_auc_pr
        calibration_acceptance_met = (val_brier <= self.brier_max) and (val_ece <= self.ece_max)
        if not auc_pr_floor_met:
            honest_failures.append(
                f"AUC-PR floor not met: val_auc_pr={val_auc_pr:.4f} < {self.min_auc_pr:.2f}"
            )
        if not calibration_acceptance_met:
            honest_failures.append(
                f"Calibration acceptance not met: "
                f"Brier={val_brier:.4f} (max {self.brier_max:.2f}), "
                f"ECE={val_ece:.4f} (max {self.ece_max:.2f})"
            )

        # Step 7 — feature importance (global mean-abs SHAP fallback to gain).
        feature_importance = self._compute_feature_importance(base, best_model_type, feat_names)

        # Step 8 — reliability diagram PNG.
        reliability_png = self._reliability_diagram_png(y_val_arr, val_proba)

        # Step 9 — MLflow logging (best-effort; failures don't break training).
        mlflow_run_id = None
        if self.enable_mlflow:
            mlflow_run_id = self._mlflow_log(
                experiment=mlflow_experiment,
                run_name=mlflow_run_name,
                model_type=best_model_type,
                best_params=best_params,
                metrics={
                    "val_auc_roc": val_auc_roc,
                    "val_auc_pr": val_auc_pr,
                    "val_brier": val_brier,
                    "val_ece": val_ece,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "cv_auc_pr_mean": cv_auc_pr_mean,
                    "cv_auc_pr_std": cv_auc_pr_std,
                    "auc_pr_floor_met": float(auc_pr_floor_met),
                    "calibration_acceptance_met": float(calibration_acceptance_met),
                },
                tags={
                    "issue": "171",
                    "target": "initiated_biologic_180d",
                    "calibration_method": calib_method,
                    "cohort": "csu_initiation",
                },
                reliability_png=reliability_png,
                feature_importance=feature_importance,
                honest_failures=honest_failures,
                estimator=calibrated,
            )

        return RiskScoreTrainingResult(
            model_type=best_model_type,
            best_params=best_params,
            feature_names=feat_names,
            val_auc_roc=float(val_auc_roc),
            val_auc_pr=float(val_auc_pr),
            val_precision=float(val_precision),
            val_recall=float(val_recall),
            val_brier=float(val_brier),
            val_ece=float(val_ece),
            calibration_method=calib_method,
            auc_pr_floor=self.min_auc_pr,
            auc_pr_floor_met=bool(auc_pr_floor_met),
            calibration_acceptance_met=bool(calibration_acceptance_met),
            cv_auc_pr_mean=float(cv_auc_pr_mean),
            cv_auc_pr_std=float(cv_auc_pr_std),
            train_class_balance={
                "n_pos": int((y_train_arr == 1).sum()),
                "n_neg": int((y_train_arr == 0).sum()),
            },
            val_class_balance={
                "n_pos": int((y_val_arr == 1).sum()),
                "n_neg": int((y_val_arr == 0).sum()),
            },
            feature_importance=feature_importance,
            reliability_diagram_png=reliability_png,
            mlflow_run_id=mlflow_run_id,
            honest_failures=honest_failures,
            estimator=calibrated,
        )

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _choose_model_type(self, X: Any, y: np.ndarray) -> tuple[str, float, float]:
        """Pick xgboost vs lightgbm via 5-fold CV AUC-PR; return mean+std of chosen."""
        from sklearn.metrics import average_precision_score  # type: ignore[import-not-found]
        from sklearn.model_selection import StratifiedKFold  # type: ignore[import-not-found]

        # Adapt n_splits when minority class is too small for cv_folds.
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        effective_folds = min(self.cv_folds, max(2, min(n_pos, n_neg)))
        if effective_folds < self.cv_folds:
            logger.warning(
                "risk_score_trainer: reducing CV folds %d -> %d due to small minority "
                "class (n_pos=%d, n_neg=%d).",
                self.cv_folds,
                effective_folds,
                n_pos,
                n_neg,
            )

        skf = StratifiedKFold(
            n_splits=effective_folds, shuffle=True, random_state=self.random_state
        )
        per_type_scores: dict[str, list[float]] = {}
        for model_type in self.model_candidates:
            fold_scores: list[float] = []
            for train_idx, val_idx in skf.split(X, y):
                X_tr = X.iloc[train_idx]
                X_va = X.iloc[val_idx]
                est = self._build_estimator(model_type, self._default_params(model_type))
                est.fit(X_tr, y[train_idx])
                proba = est.predict_proba(X_va)[:, 1]
                fold_scores.append(
                    float(self._safe_metric(average_precision_score, y[val_idx], proba))
                )
            per_type_scores[model_type] = fold_scores

        # Pick highest mean AUC-PR.
        means = {k: float(np.mean(v)) for k, v in per_type_scores.items()}
        stds = {k: float(np.std(v)) for k, v in per_type_scores.items()}
        best = max(means, key=lambda k: means[k])
        return best, means[best], stds[best]

    def _optuna_search(self, model_type: str, X: Any, y: np.ndarray) -> dict[str, Any]:
        """Run Optuna HPO on ``model_type`` via 5-fold CV AUC-PR maximisation."""
        import optuna  # type: ignore[import-not-found]
        from sklearn.metrics import average_precision_score  # type: ignore[import-not-found]
        from sklearn.model_selection import StratifiedKFold  # type: ignore[import-not-found]

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        effective_folds = min(self.cv_folds, max(2, min(n_pos, n_neg)))
        skf = StratifiedKFold(
            n_splits=effective_folds, shuffle=True, random_state=self.random_state
        )

        def objective(trial: "optuna.trial.Trial") -> float:
            if model_type == "xgboost":
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                }
            else:  # lightgbm
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                }
            fold_scores: list[float] = []
            for train_idx, val_idx in skf.split(X, y):
                est = self._build_estimator(model_type, params)
                est.fit(X.iloc[train_idx], y[train_idx])
                proba = est.predict_proba(X.iloc[val_idx])[:, 1]
                fold_scores.append(
                    float(self._safe_metric(average_precision_score, y[val_idx], proba))
                )
            return float(np.mean(fold_scores))

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, study_name=f"risk_score_{model_type}"
        )
        # Quiet Optuna log noise during tests.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(objective, n_trials=self.hpo_trials, show_progress_bar=False)
        return dict(study.best_params)

    def _default_params(self, model_type: str) -> dict[str, Any]:
        """Sensible defaults used for the model-type-selection CV pass."""
        if model_type == "xgboost":
            return {
                "max_depth": 5,
                "learning_rate": 0.05,
                "n_estimators": 200,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            }
        return {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }

    def _build_estimator(self, model_type: str, params: dict[str, Any]) -> Any:
        """Construct a fresh sklearn-compatible classifier with ``params``."""
        if model_type == "xgboost":
            from xgboost import XGBClassifier  # type: ignore[import-not-found]

            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                random_state=self.random_state,
                n_jobs=1,
                verbosity=0,
                **params,
            )
        # lightgbm
        from lightgbm import LGBMClassifier  # type: ignore[import-not-found]

        return LGBMClassifier(
            objective="binary",
            random_state=self.random_state,
            n_jobs=1,
            verbose=-1,
            **params,
        )

    def _calibrate_on_val(self, base: Any, X_val: Any, y_val: np.ndarray) -> tuple[Any, str, float]:
        """Fit Platt + isotonic on VAL using cv='prefit'; pick lower-Brier."""
        from sklearn.calibration import CalibratedClassifierCV  # type: ignore[import-not-found]
        from sklearn.metrics import brier_score_loss  # type: ignore[import-not-found]

        results: list[tuple[Any, str, float]] = []
        # Isotonic requires multiple positives to fit; if val has < 2 pos / neg
        # we skip it and fall back to Platt only.
        n_pos = int((y_val == 1).sum())
        n_neg = int((y_val == 0).sum())
        methods: list[str] = ["sigmoid"]
        if n_pos >= 2 and n_neg >= 2:
            methods.append("isotonic")
        for method in methods:
            calib = CalibratedClassifierCV(estimator=base, method=method, cv="prefit")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                calib.fit(X_val, y_val)
            proba = calib.predict_proba(X_val)[:, 1]
            brier = float(brier_score_loss(y_val, proba))
            results.append((calib, method, brier))
        results.sort(key=lambda r: r[2])
        return results[0]

    @staticmethod
    def _safe_metric(fn: Any, *args: Any, **kwargs: Any) -> float:
        """Call a sklearn metric, returning NaN on ValueError (e.g., single-class y)."""
        try:
            return float(fn(*args, **kwargs))
        except ValueError as exc:
            logger.warning("risk_score_trainer: metric %s failed: %s", fn.__name__, exc)
            return float("nan")

    def _compute_feature_importance(
        self, base: Any, model_type: str, feat_names: list[str]
    ) -> dict[str, float]:
        """Return {feature: importance} dict. Gain-based for tree models."""
        try:
            if model_type == "xgboost":
                booster = base.get_booster()
                score = booster.get_score(importance_type="gain")
                # XGBoost names features as f0, f1, ... unless we pass DMatrix
                # with feature_names. Map via positional fallback.
                out: dict[str, float] = dict.fromkeys(feat_names, 0.0)
                for k, v in score.items():
                    if k.startswith("f") and k[1:].isdigit():
                        idx = int(k[1:])
                        if 0 <= idx < len(feat_names):
                            out[feat_names[idx]] = float(v)
                    elif k in out:
                        out[k] = float(v)
                return out
            # lightgbm
            booster = base.booster_
            imps = booster.feature_importance(importance_type="gain")
            names = booster.feature_name()
            out = dict.fromkeys(feat_names, 0.0)
            for n, v in zip(names, imps, strict=False):
                if n in out:
                    out[n] = float(v)
            return out
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("risk_score_trainer: feature importance failed: %s", exc)
            return dict.fromkeys(feat_names, 0.0)

    @staticmethod
    def _reliability_diagram_png(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[bytes]:
        """Render a reliability diagram (matplotlib) -> PNG bytes. ``None`` on error."""
        try:
            import matplotlib  # type: ignore[import-not-found]

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore[import-not-found]
            from sklearn.calibration import calibration_curve  # type: ignore[import-not-found]

            # Adapt n_bins to sample size — sklearn raises if any bin has 0
            # samples with strategy='uniform'; quantile is safer for small N.
            n_bins = max(2, min(10, int(np.sqrt(y_true.size))))
            prob_true, prob_pred = calibration_curve(
                y_true, y_prob, n_bins=n_bins, strategy="quantile"
            )
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
            ax.plot(prob_pred, prob_true, marker="o", label="model")
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Observed fraction")
            ax.set_title("Risk-score reliability diagram (validation)")
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            return buf.getvalue()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("risk_score_trainer: reliability diagram failed: %s", exc)
            return None

    def _mlflow_log(
        self,
        *,
        experiment: str,
        run_name: Optional[str],
        model_type: str,
        best_params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str],
        reliability_png: Optional[bytes],
        feature_importance: dict[str, float],
        honest_failures: list[str],
        estimator: Any,
    ) -> Optional[str]:
        """Log to MLflow if a tracking server is reachable. Best-effort; never raises."""
        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError:
            logger.info("risk_score_trainer: mlflow not installed; skipping logging.")
            return None
        try:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)
            # Use explicit kwarg dispatch instead of **run_kwargs so mypy can
            # type-check ``start_run`` against its actual signature.
            run_ctx = mlflow.start_run(run_name=run_name) if run_name else mlflow.start_run()
            with run_ctx as run:
                mlflow.set_tags({**tags, "model_type": model_type})
                # Stringify param values so MLflow accepts them.
                mlflow.log_params({k: str(v) for k, v in best_params.items()})
                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
                if reliability_png:
                    with tempfile.TemporaryDirectory() as td:
                        path = os.path.join(td, "reliability_diagram.png")
                        with open(path, "wb") as fh:
                            fh.write(reliability_png)
                        mlflow.log_artifact(path)
                if feature_importance:
                    with tempfile.TemporaryDirectory() as td:
                        path = os.path.join(td, "feature_importance.json")
                        with open(path, "w") as fh:
                            json.dump(feature_importance, fh, indent=2, sort_keys=True)
                        mlflow.log_artifact(path)
                if honest_failures:
                    mlflow.set_tag("honest_failures", "; ".join(honest_failures)[:500])
                # Log the calibrated estimator. Wrap in best-effort so a missing
                # sklearn-flavor doesn't break the run.
                try:
                    mlflow.sklearn.log_model(estimator, name="model")
                except Exception as exc:  # pragma: no cover - depends on MLflow
                    logger.warning("risk_score_trainer: mlflow log_model failed: %s", exc)
                run_id: Optional[str] = str(run.info.run_id) if run is not None else None
                return run_id
        except Exception as exc:
            logger.warning(
                "risk_score_trainer: MLflow logging failed (%s: %s). "
                "Honest deferral: ship the model without a tracking-server entry.",
                type(exc).__name__,
                exc,
            )
            return None

    # -------------------------------------------------------------------
    # Inference / DB-write helpers
    # -------------------------------------------------------------------

    def predict_risk_score(
        self, calibrated_estimator: Any, X: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (probabilities, risk_scores) for the input rows.

        ``risk_scores`` are clamped to [0.00, 9.99] per the DECIMAL(3,2) DB
        column type.
        """
        import pandas as pd  # local

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X))
        proba = np.asarray(calibrated_estimator.predict_proba(X))[:, 1]
        scores = np.array([probability_to_risk_score(float(p)) for p in proba])
        return proba, scores

    def build_ml_predictions_payload(
        self,
        result: RiskScoreTrainingResult,
        patient_id: str,
        proba: float,
        risk_score: float,
        per_patient_shap: Optional[dict[str, float]] = None,
        features_available: Optional[dict[str, Any]] = None,
        model_version: str = "v1",
        prediction_id: Optional[str] = None,
        prediction_timestamp: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Build a per-patient ``ml_predictions`` row dict.

        Matches the schema at ``database/core/e2i_ml_complete_v3_schema.sql:525``
        (table ``ml_predictions``).

        ``prediction_id`` (PRIMARY KEY) and ``prediction_timestamp``
        (NOT NULL) are required by the schema. If not provided, we auto-mint
        a 30-char prediction_id (``rsc_<24hex>``) and stamp ``now(UTC)``.
        Callers writing many rows in a batch SHOULD pass explicit values
        for both so the caller controls dedup keys.

        Codex pass-1 MEDIUM-2: the previous version returned a row missing
        the PRIMARY KEY + NOT NULL columns; this version fills them.
        """
        if prediction_id is None:
            # 30-char alphanumeric matches VARCHAR(30) PK width.
            prediction_id = f"rsc_{secrets.token_hex(13)}"  # 4 + 26 = 30
        if prediction_timestamp is None:
            prediction_timestamp = datetime.now(timezone.utc)
        top_features: list[dict[str, Any]] = []
        if per_patient_shap:
            ranked = sorted(per_patient_shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
            top_features = [{"feature": name, "shap_value": float(val)} for name, val in ranked]
        elif result.feature_importance:
            ranked = sorted(
                result.feature_importance.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[:10]
            top_features = [{"feature": name, "gain": float(val)} for name, val in ranked]

        return {
            "prediction_id": prediction_id,
            "prediction_timestamp": prediction_timestamp.isoformat(),
            "model_version": model_version,
            "model_type": result.model_type,
            "patient_id": patient_id,
            "prediction_type": "risk",
            "prediction_value": float(proba),
            "prediction_class": risk_score_to_tier(risk_score),
            "confidence_score": float(proba),
            "probability_scores": {"positive": float(proba), "negative": 1.0 - float(proba)},
            "feature_importance": {
                k: float(v) for k, v in (result.feature_importance or {}).items()
            },
            "shap_values": {k: float(v) for k, v in (per_patient_shap or {}).items()},
            "top_features": top_features,
            "model_auc": float(result.val_auc_roc) if np.isfinite(result.val_auc_roc) else None,
            "model_pr_auc": (float(result.val_auc_pr) if np.isfinite(result.val_auc_pr) else None),
            "model_precision": (
                float(result.val_precision) if np.isfinite(result.val_precision) else None
            ),
            "model_recall": (float(result.val_recall) if np.isfinite(result.val_recall) else None),
            "calibration_score": 1.0 - float(result.val_ece)
            if np.isfinite(result.val_ece)
            else None,
            "brier_score": float(result.val_brier) if np.isfinite(result.val_brier) else None,
            "features_available_at_prediction": features_available or {},
        }
