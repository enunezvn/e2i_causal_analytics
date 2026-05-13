"""Unit tests for the risk_score trainer (issue #171 PR C Sub-PR-B).

Covers:
    - Leakage guard: forbidden substrings detected (case-insensitive).
    - Calibration helpers: ECE, probability -> risk_score, tier mapping.
    - End-to-end smoke fit on a synthetic separable dataset (validates AUC-PR
      floor, calibration bar, MLflow path off, deterministic params).
    - ``build_ml_predictions_payload`` schema mirrors
      ``database/core/e2i_ml_complete_v3_schema.sql`` ml_predictions columns.
    - Honest-failure surfacing when the bar fails on a noisy dataset.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.agents.prediction_synthesizer.risk_score import (
    CALIBRATION_BRIER_MAX,
    CALIBRATION_ECE_MAX,
    DEFAULT_MIN_AUC_PR,
    FORBIDDEN_FEATURE_SUBSTRINGS,
    RISK_SCORE_HIGH_TIER,
    RISK_SCORE_LOW_TIER,
    LeakageError,
    RiskScoreTrainer,
    assert_no_leakage_in_features,
    expected_calibration_error,
    find_leaked_features,
    risk_score_to_tier,
)
from src.agents.prediction_synthesizer.risk_score.risk_score_trainer import (
    probability_to_risk_score,
)

# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------


class TestLeakageGuard:
    def test_clean_features_pass(self) -> None:
        assert_no_leakage_in_features(["age", "ed_visits_total", "comorbidities_count"])

    @pytest.mark.parametrize("token", FORBIDDEN_FEATURE_SUBSTRINGS)
    def test_each_forbidden_token_detected(self, token: str) -> None:
        feature_name = f"some_feature_with_{token}_in_name"
        with pytest.raises(LeakageError) as exc:
            assert_no_leakage_in_features(["age", feature_name, "ed_visits_total"])
        assert feature_name in exc.value.leaked

    def test_case_insensitive(self) -> None:
        with pytest.raises(LeakageError) as exc:
            assert_no_leakage_in_features(["XOLAIR_ever_filled"])
        assert "XOLAIR_ever_filled" in exc.value.leaked

    def test_find_leaked_returns_empty_on_clean(self) -> None:
        assert find_leaked_features(["age", "weight"]) == []

    def test_find_leaked_returns_all_matches(self) -> None:
        leaked = find_leaked_features(["age", "xolair_fills", "weight", "dupilumab_count"])
        assert leaked == ["xolair_fills", "dupilumab_count"]

    def test_custom_forbidden_list(self) -> None:
        with pytest.raises(LeakageError):
            assert_no_leakage_in_features(["my_feature"], forbidden=("my_feat",))

    def test_empty_forbidden_list_passes_everything(self) -> None:
        # If forbidden list is empty, nothing leaks.
        assert find_leaked_features(["xolair_fills"], forbidden=()) == []

    def test_ndc_prefix_50242_detected(self) -> None:
        # Xolair NDC prefix
        with pytest.raises(LeakageError):
            assert_no_leakage_in_features(["drug_50242_04_fills"])

    def test_ndc_prefix_00024_detected(self) -> None:
        # Dupixent NDC prefix (Sanofi/Regeneron)
        with pytest.raises(LeakageError):
            assert_no_leakage_in_features(["product_00024_45_fills"])


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


class TestExpectedCalibrationError:
    def test_perfect_calibration_yields_zero(self) -> None:
        # Probabilities equal to empirical positive rate per bin.
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.5] * 8)
        # In a single bin all-at-0.5: acc=0.5, conf=0.5 -> ECE 0.
        assert expected_calibration_error(y_true, y_prob, n_bins=1) == pytest.approx(0.0)

    def test_perfectly_miscalibrated_yields_max(self) -> None:
        # All predicted 0 but all true 1: |0 - 1| = 1.0
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0])
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        # ECE is bounded in [0, 1]
        assert 0.0 <= ece <= 1.0
        assert ece == pytest.approx(1.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            expected_calibration_error(np.array([0, 1]), np.array([0.5, 0.5, 0.5]))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            expected_calibration_error(np.array([]), np.array([]))

    def test_n_bins_below_one_raises(self) -> None:
        with pytest.raises(ValueError):
            expected_calibration_error(np.array([0, 1]), np.array([0.3, 0.7]), n_bins=0)

    def test_ece_includes_one_in_last_bin(self) -> None:
        # If prob == 1.0 it should still be counted (right-edge inclusion).
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([1.0, 1.0, 1.0, 1.0])
        # Perfectly calibrated: acc==conf==1.0
        assert expected_calibration_error(y_true, y_prob, n_bins=10) == pytest.approx(0.0)


class TestRiskScoreScale:
    def test_probability_to_risk_score_basic(self) -> None:
        assert probability_to_risk_score(0.0) == 0.00
        assert probability_to_risk_score(0.5) == 5.00
        # 1.0 * 10 = 10.0; clamp to 9.99
        assert probability_to_risk_score(1.0) == 9.99
        assert probability_to_risk_score(0.123) == pytest.approx(1.23)

    def test_probability_to_risk_score_handles_nan(self) -> None:
        assert probability_to_risk_score(float("nan")) == 0.0

    def test_probability_to_risk_score_clamps_above_max(self) -> None:
        # >1.0 (shouldn't happen with valid proba, but be defensive)
        assert probability_to_risk_score(1.5) == 9.99

    def test_probability_to_risk_score_clamps_below_min(self) -> None:
        assert probability_to_risk_score(-0.1) == 0.00

    def test_tier_high_at_threshold(self) -> None:
        assert risk_score_to_tier(RISK_SCORE_HIGH_TIER) == "high"

    def test_tier_medium_in_band(self) -> None:
        assert risk_score_to_tier(5.0) == "medium"

    def test_tier_low_below_threshold(self) -> None:
        assert risk_score_to_tier(RISK_SCORE_LOW_TIER - 0.01) == "low"

    def test_tier_low_at_zero(self) -> None:
        assert risk_score_to_tier(0.0) == "low"


# ---------------------------------------------------------------------------
# Trainer construction
# ---------------------------------------------------------------------------


class TestTrainerConstruction:
    def test_invalid_candidate_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            RiskScoreTrainer(model_candidates=("random_forest",))

    def test_defaults_match_supervisor_decisions(self) -> None:
        t = RiskScoreTrainer()
        assert t.min_auc_pr == DEFAULT_MIN_AUC_PR == 0.65
        assert t.brier_max == CALIBRATION_BRIER_MAX == 0.20
        assert t.ece_max == CALIBRATION_ECE_MAX == 0.10
        assert t.model_candidates == ("xgboost", "lightgbm")

    def test_can_pin_single_candidate(self) -> None:
        t = RiskScoreTrainer(model_candidates=("xgboost",))
        assert t.model_candidates == ("xgboost",)


# ---------------------------------------------------------------------------
# End-to-end smoke fit on synthetic data
# ---------------------------------------------------------------------------


def _make_separable_dataset(
    n: int = 600, n_features: int = 8, random_state: int = 7
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Synthetic separable binary-classification cohort.

    Returns ``(X_train, y_train, X_val, y_val)``.

    Uses sklearn's ``make_classification`` with a strong signal so a basic
    tree-based model can clear the 0.65 AUC-PR floor (this validates the
    plumbing, not real-data behavior). Class imbalance is ~30% positive so
    AUC-PR is meaningful (random baseline ~0.30).
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=4,
        n_redundant=2,
        n_classes=2,
        weights=[0.7, 0.3],
        flip_y=0.02,
        class_sep=1.5,
        random_state=random_state,
    )
    feat_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feat_names)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_df, y, test_size=0.25, stratify=y, random_state=random_state
    )
    # Re-index after split so DataFrame ops are clean.
    return X_tr.reset_index(drop=True), y_tr, X_va.reset_index(drop=True), y_va


@pytest.mark.slow
def test_smoke_fit_separable_passes_auc_pr_floor() -> None:
    """Plumbing smoke: separable synthetic cohort clears AUC-PR floor."""
    X_tr, y_tr, X_va, y_va = _make_separable_dataset()
    # Constrain HPO trials for test speed; pin to xgboost so the CV-pick is
    # deterministic and the test isn't sensitive to which library wins.
    trainer = RiskScoreTrainer(
        hpo_trials=5,
        cv_folds=3,
        enable_mlflow=False,
        model_candidates=("xgboost",),
    )
    result = trainer.fit(X_tr, y_tr, X_va, y_va)
    # Plumbing assertions.
    assert result.model_type == "xgboost"
    assert result.feature_names == list(X_tr.columns)
    assert result.train_class_balance["n_pos"] > 0
    assert result.train_class_balance["n_neg"] > 0
    assert result.val_class_balance["n_pos"] > 0
    assert result.val_class_balance["n_neg"] > 0
    # Discrimination should clear the floor on this synthetic dataset.
    assert result.val_auc_pr >= DEFAULT_MIN_AUC_PR
    assert result.auc_pr_floor_met is True
    # Calibration bar should also be met on a well-separable dataset.
    assert result.val_brier <= CALIBRATION_BRIER_MAX
    # Calibration method must be one of the two supported.
    assert result.calibration_method in {"sigmoid", "isotonic"}
    # Estimator is sklearn-like with predict_proba.
    proba = result.estimator.predict_proba(X_va)[:, 1]
    assert proba.shape == (X_va.shape[0],)
    # Honest failures list is empty on a passing run.
    if result.auc_pr_floor_met and result.calibration_acceptance_met:
        assert result.honest_failures == []


@pytest.mark.slow
def test_smoke_fit_lightgbm_also_works() -> None:
    """Same separable cohort, pinned to lightgbm."""
    X_tr, y_tr, X_va, y_va = _make_separable_dataset(random_state=11)
    trainer = RiskScoreTrainer(
        hpo_trials=5,
        cv_folds=3,
        enable_mlflow=False,
        model_candidates=("lightgbm",),
    )
    result = trainer.fit(X_tr, y_tr, X_va, y_va)
    assert result.model_type == "lightgbm"
    assert math.isfinite(result.val_auc_pr)
    assert math.isfinite(result.val_brier)


@pytest.mark.slow
def test_predict_risk_score_clamps_and_rounds() -> None:
    """End-to-end: predict_risk_score yields in-range DECIMAL(3,2) values."""
    X_tr, y_tr, X_va, y_va = _make_separable_dataset()
    trainer = RiskScoreTrainer(
        hpo_trials=3,
        cv_folds=3,
        enable_mlflow=False,
        model_candidates=("xgboost",),
    )
    result = trainer.fit(X_tr, y_tr, X_va, y_va)
    proba, scores = trainer.predict_risk_score(result.estimator, X_va)
    assert proba.shape == (X_va.shape[0],)
    assert scores.shape == (X_va.shape[0],)
    # All risk scores must respect the DECIMAL(3,2) range.
    assert scores.min() >= 0.0
    assert scores.max() <= 9.99


@pytest.mark.slow
def test_build_ml_predictions_payload_schema() -> None:
    """Payload dict matches ml_predictions table columns."""
    X_tr, y_tr, X_va, y_va = _make_separable_dataset(n=200)
    trainer = RiskScoreTrainer(
        hpo_trials=3, cv_folds=3, enable_mlflow=False, model_candidates=("xgboost",)
    )
    result = trainer.fit(X_tr, y_tr, X_va, y_va)
    payload = trainer.build_ml_predictions_payload(
        result,
        patient_id="P0001",
        proba=0.78,
        risk_score=7.80,
        per_patient_shap={"feature_0": 0.5, "feature_1": -0.2},
        features_available={"age": 45, "comorbidities": 2},
    )
    # Schema-required keys (PRIMARY KEY + NOT NULL columns per
    # database/core/e2i_ml_complete_v3_schema.sql:525). codex pass-1 MEDIUM-2
    # — the payload must be insertable; the test catches missing PK / NOT NULL.
    expected_keys = {
        "prediction_id",  # PRIMARY KEY VARCHAR(30)
        "prediction_timestamp",  # TIMESTAMPTZ NOT NULL
        "model_version",
        "model_type",
        "patient_id",  # VARCHAR(20) NOT NULL
        "prediction_type",
        "prediction_value",
        "prediction_class",
        "confidence_score",
        "probability_scores",
        "feature_importance",
        "shap_values",
        "top_features",
        "model_auc",
        "model_pr_auc",
        "model_precision",
        "model_recall",
        "calibration_score",
        "brier_score",
        "features_available_at_prediction",
    }
    assert expected_keys.issubset(payload.keys())
    # Schema constraints.
    assert payload["prediction_id"].startswith("rsc_"), payload["prediction_id"]
    assert len(payload["prediction_id"]) == 30, (
        f"prediction_id width {len(payload['prediction_id'])} != VARCHAR(30)"
    )
    assert payload["prediction_timestamp"] is not None
    assert payload["patient_id"] == "P0001"
    assert payload["prediction_type"] == "risk"
    assert payload["prediction_class"] == "high"  # 7.80 >= 6.6
    assert payload["model_type"] == "xgboost"
    assert payload["features_available_at_prediction"] == {"age": 45, "comorbidities": 2}
    assert len(payload["top_features"]) == 2
    assert payload["top_features"][0]["feature"] == "feature_0"  # |0.5| > |-0.2|


@pytest.mark.slow
def test_build_ml_predictions_payload_accepts_explicit_id_and_timestamp() -> None:
    """Caller-provided ``prediction_id`` / ``prediction_timestamp`` round-trip."""
    from datetime import datetime, timezone

    X_tr, y_tr, X_va, y_va = _make_separable_dataset(n=200)
    trainer = RiskScoreTrainer(
        hpo_trials=3, cv_folds=3, enable_mlflow=False, model_candidates=("xgboost",)
    )
    result = trainer.fit(X_tr, y_tr, X_va, y_va)
    ts = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)
    explicit_id = "rsc_explicit_callerid000000000"  # exactly 30 chars
    payload = trainer.build_ml_predictions_payload(
        result,
        patient_id="P0002",
        proba=0.3,
        risk_score=3.00,
        prediction_id=explicit_id,
        prediction_timestamp=ts,
    )
    assert payload["prediction_id"] == explicit_id
    assert payload["prediction_timestamp"] == ts.isoformat()
    assert payload["prediction_class"] == "low"  # 3.00 < 3.3


# ---------------------------------------------------------------------------
# Honest-failure surfacing
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_honest_failure_surfaced_on_noise() -> None:
    """On pure-noise features the trainer surfaces (does NOT lower) the bar."""
    rng = np.random.default_rng(13)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"noise_{i}" for i in range(6)])
    y = rng.integers(0, 2, size=n)
    trainer = RiskScoreTrainer(
        hpo_trials=3, cv_folds=3, enable_mlflow=False, model_candidates=("xgboost",)
    )
    X_tr, X_va = X.iloc[: n // 2].reset_index(drop=True), X.iloc[n // 2 :].reset_index(drop=True)
    y_tr, y_va = y[: n // 2], y[n // 2 :]
    result = trainer.fit(X_tr, y_tr, X_va, y_va)
    # Bar was not silently lowered.
    assert result.auc_pr_floor == DEFAULT_MIN_AUC_PR
    # If the floor was not met, honest_failures must record it.
    if not result.auc_pr_floor_met:
        assert any("AUC-PR floor not met" in f for f in result.honest_failures)
    # If calibration was not met, honest_failures must record it.
    if not result.calibration_acceptance_met:
        assert any("Calibration acceptance not met" in f for f in result.honest_failures)


@pytest.mark.slow
def test_honest_failure_deterministic_with_impossible_floor() -> None:
    """Codex pass-1 MEDIUM-5 (issue #173): the noise-features test above
    is *probabilistic* (it asserts honest_failures only IF the floor
    happened not to be met). This test pins min_auc_pr=1.0 — physically
    impossible to satisfy with any real classifier — so the
    honest_failures path is *deterministically* exercised. If a future
    refactor ever silently dropped the surfacing logic, this test would
    fail.
    """
    rng = np.random.default_rng(7)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"feat_{i}" for i in range(6)])
    y = rng.integers(0, 2, size=n)
    # Even a perfect classifier cannot beat 1.0 strictly (== is allowed
    # by `auc_pr_floor_met = val_auc_pr >= floor`, but the noise data
    # makes equality unreachable).
    trainer = RiskScoreTrainer(
        hpo_trials=3,
        cv_folds=3,
        enable_mlflow=False,
        model_candidates=("xgboost",),
        min_auc_pr=1.0,
    )
    X_tr, X_va = X.iloc[: n // 2].reset_index(drop=True), X.iloc[n // 2 :].reset_index(drop=True)
    y_tr, y_va = y[: n // 2], y[n // 2 :]
    result = trainer.fit(X_tr, y_tr, X_va, y_va)
    # Bar was not silently lowered.
    assert result.auc_pr_floor == 1.0
    assert result.auc_pr_floor_met is False
    # honest_failures must surface — deterministically.
    assert any("AUC-PR floor not met" in f for f in result.honest_failures), (
        f"honest_failures regressed: {result.honest_failures}"
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestFitInputValidation:
    def _make_simple_inputs(self) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        X = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1], "b": [1, 2, 3, 4, 5, 6]})
        y = np.array([0, 1, 0, 1, 0, 1])
        return X, y, X.copy(), y.copy()

    def test_leakage_in_feature_name_raises(self) -> None:
        X = pd.DataFrame({"xolair_ever_filled": [0, 1, 0, 1], "age": [30, 40, 50, 60]})
        y = np.array([0, 1, 0, 1])
        trainer = RiskScoreTrainer(enable_mlflow=False, hpo_trials=2, cv_folds=2)
        with pytest.raises(LeakageError):
            trainer.fit(X, y, X, y)

    def test_length_mismatch_y_train_raises(self) -> None:
        X_tr, _, X_va, y_va = self._make_simple_inputs()
        trainer = RiskScoreTrainer(enable_mlflow=False, hpo_trials=2, cv_folds=2)
        with pytest.raises(ValueError, match="y_train length"):
            trainer.fit(X_tr, np.array([0, 1]), X_va, y_va)

    def test_length_mismatch_y_val_raises(self) -> None:
        X_tr, y_tr, X_va, _ = self._make_simple_inputs()
        trainer = RiskScoreTrainer(enable_mlflow=False, hpo_trials=2, cv_folds=2)
        with pytest.raises(ValueError, match="y_val length"):
            trainer.fit(X_tr, y_tr, X_va, np.array([0]))

    def test_non_binary_labels_raise(self) -> None:
        X_tr, y_tr, X_va, _ = self._make_simple_inputs()
        trainer = RiskScoreTrainer(enable_mlflow=False, hpo_trials=2, cv_folds=2)
        with pytest.raises(ValueError, match="binary"):
            trainer.fit(X_tr, y_tr, X_va, np.array([0, 1, 2, 0, 1, 0]))

    def test_non_numeric_features_raise(self) -> None:
        X = pd.DataFrame({"a": ["foo", "bar", "baz", "qux"], "b": [1, 2, 3, 4]})
        y = np.array([0, 1, 0, 1])
        trainer = RiskScoreTrainer(enable_mlflow=False, hpo_trials=2, cv_folds=2)
        with pytest.raises(TypeError, match="non-numeric"):
            trainer.fit(X, y, X, y)
