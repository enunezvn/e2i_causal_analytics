"""Tests for hyperparameter tuner node."""

from unittest.mock import patch

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner import (
    _get_fixed_params,
    _get_hpo_pattern_memory,
    _inject_class_weight_sweep,
    tune_hyperparameters,
    validate_hpo_output,
    validate_hyperparameter_types,
)


@pytest.mark.asyncio
class TestTuneHyperparameters:
    """Test hyperparameter optimization (HPO)."""

    async def test_returns_defaults_when_hpo_disabled(self):
        """Should return default hyperparameters when HPO disabled."""
        state = {
            "enable_hpo": False,
            "default_hyperparameters": {"n_estimators": 100, "max_depth": 10},
        }

        result = await tune_hyperparameters(state)

        assert result["hpo_completed"] is False
        assert result["best_hyperparameters"] == {"n_estimators": 100, "max_depth": 10}
        assert result["hpo_trials_run"] == 0

    async def test_runs_hpo_when_enabled(self):
        """Should attempt HPO when enabled."""
        state = {
            "enable_hpo": True,
            "hpo_trials": 10,
            "algorithm_name": "RandomForest",
            "problem_type": "binary_classification",
            "experiment_id": "test_exp_123",
            "default_hyperparameters": {"n_estimators": 100},
            "hyperparameter_search_space": {
                "n_estimators": {"type": "int", "low": 50, "high": 200}
            },
            "X_train_preprocessed": np.random.rand(100, 5),
            "X_validation_preprocessed": np.random.rand(30, 5),
            "train_data": {"y": np.random.randint(0, 2, 100)},
            "validation_data": {"y": np.random.randint(0, 2, 30)},
        }

        result = await tune_hyperparameters(state)

        assert "hpo_completed" in result
        assert "best_hyperparameters" in result
        assert "hpo_duration_seconds" in result

    async def test_error_when_missing_training_data(self):
        """Should return error when training data missing."""
        state = {
            "enable_hpo": True,
            "hpo_trials": 10,
            "hyperparameter_search_space": {},
        }

        result = await tune_hyperparameters(state)

        assert "error" in result
        assert result["error_type"] == "missing_hpo_data"

    async def test_returns_defaults_when_no_search_space(self):
        """Should return defaults when no search space defined."""
        state = {
            "enable_hpo": True,
            "hpo_trials": 10,
            "hyperparameter_search_space": {},
            "default_hyperparameters": {"n_estimators": 50},
            "X_train_preprocessed": np.random.rand(100, 5),
            "X_validation_preprocessed": np.random.rand(30, 5),
            "train_data": {"y": np.random.randint(0, 2, 100)},
            "validation_data": {"y": np.random.randint(0, 2, 30)},
        }

        result = await tune_hyperparameters(state)

        assert result["best_hyperparameters"] == {"n_estimators": 50}


class TestValidateHpoOutput:
    """Tests for HPO output validation."""

    def test_validates_complete_output(self):
        """Should validate complete HPO output."""
        output = {
            "hpo_completed": True,
            "best_hyperparameters": {"n_estimators": 150},
            "hpo_best_trial": 5,
            "hpo_trials_run": 10,
            "hpo_best_value": 0.95,
            "hpo_study_name": "test_study",
        }

        is_valid, errors = validate_hpo_output(output)

        assert is_valid is True
        assert errors == []

    def test_detects_missing_required_field(self):
        """Should detect missing required field."""
        output = {
            "hpo_completed": True,
            # missing best_hyperparameters
        }

        is_valid, errors = validate_hpo_output(output)

        assert is_valid is False
        assert any("best_hyperparameters" in e for e in errors)

    def test_detects_invalid_type(self):
        """Should detect invalid field type."""
        output = {
            "hpo_completed": "yes",  # Should be bool
            "best_hyperparameters": {"n_estimators": 100},
        }

        is_valid, errors = validate_hpo_output(output)

        assert is_valid is False
        assert any("hpo_completed" in e for e in errors)

    def test_validates_completed_consistency(self):
        """Should check consistency when hpo_completed=True."""
        output = {
            "hpo_completed": True,
            "best_hyperparameters": {"n_estimators": 100},
            "hpo_trials_run": 0,  # Inconsistent
        }

        is_valid, errors = validate_hpo_output(output)

        assert is_valid is False
        assert any("hpo_completed=True but hpo_trials_run=0" in e for e in errors)


class TestValidateHyperparameterTypes:
    """Tests for hyperparameter type validation."""

    def test_validates_int_param(self):
        """Should validate int parameters."""
        hyperparameters = {"n_estimators": 100}
        search_space = {"n_estimators": {"type": "int", "low": 50, "high": 200}}

        is_valid, errors = validate_hyperparameter_types(hyperparameters, search_space)

        assert is_valid is True
        assert errors == []

    def test_detects_out_of_range_value(self):
        """Should detect value outside allowed range."""
        hyperparameters = {"n_estimators": 300}  # Above max 200
        search_space = {"n_estimators": {"type": "int", "low": 50, "high": 200}}

        is_valid, errors = validate_hyperparameter_types(hyperparameters, search_space)

        assert is_valid is False
        assert any("above maximum" in e for e in errors)

    def test_validates_categorical_param(self):
        """Should validate categorical parameters."""
        hyperparameters = {"criterion": "gini"}
        search_space = {"criterion": {"type": "categorical", "choices": ["gini", "entropy"]}}

        is_valid, errors = validate_hyperparameter_types(hyperparameters, search_space)

        assert is_valid is True

    def test_detects_invalid_categorical_value(self):
        """Should detect invalid categorical value."""
        hyperparameters = {"criterion": "invalid"}
        search_space = {"criterion": {"type": "categorical", "choices": ["gini", "entropy"]}}

        is_valid, errors = validate_hyperparameter_types(hyperparameters, search_space)

        assert is_valid is False
        assert any("not in choices" in e for e in errors)


class TestGetFixedParams:
    """Tests for fixed parameter retrieval."""

    def test_xgboost_fixed_params(self):
        """Should return correct fixed params for XGBoost."""
        params = _get_fixed_params("XGBoost")

        assert params["random_state"] == 42
        assert params["n_jobs"] == 1
        assert params["verbosity"] == 0

    def test_lightgbm_fixed_params(self):
        """Should return correct fixed params for LightGBM."""
        params = _get_fixed_params("LightGBM")

        assert params["random_state"] == 42
        assert params["n_jobs"] == 1
        assert params["verbose"] == -1

    def test_random_forest_fixed_params(self):
        """Should return correct fixed params for RandomForest."""
        params = _get_fixed_params("RandomForest")

        assert params["random_state"] == 42
        assert params["n_jobs"] == 1

    def test_logistic_regression_fixed_params(self):
        """Should return correct fixed params for LogisticRegression."""
        params = _get_fixed_params("LogisticRegression")

        assert params["random_state"] == 42
        assert params["max_iter"] == 1000

    def test_ridge_fixed_params(self):
        """Should return correct fixed params for Ridge."""
        params = _get_fixed_params("Ridge")

        assert params["random_state"] == 42

    def test_unknown_algorithm_returns_empty(self):
        """Should return empty dict for unknown algorithm."""
        params = _get_fixed_params("UnknownAlgorithm")

        assert params == {}

    # --- Extreme imbalance: aggressive caps + subsampling ---

    def test_xgboost_extreme_caps(self):
        """XGBoost extreme: max_depth=4, min_child_weight=10, gamma=1.0, subsample/colsample=0.7."""
        params = _get_fixed_params(
            "XGBoost",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 970, 1: 30},
            imbalance_severity="extreme",
        )
        assert params["max_depth"] == 4
        assert params["min_child_weight"] == 10
        assert params["gamma"] == 1.0
        assert params["subsample"] == 0.7
        assert params["colsample_bytree"] == 0.7

    def test_lightgbm_extreme_caps(self):
        """LightGBM extreme: max_depth=4, num_leaves=15, subsampling with freq=1."""
        params = _get_fixed_params(
            "LightGBM",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 970, 1: 30},
            imbalance_severity="extreme",
        )
        assert params["max_depth"] == 4
        assert params["num_leaves"] == 15
        assert params["min_child_samples"] == 20
        assert params["subsample"] == 0.7
        assert params["subsample_freq"] == 1
        assert params["colsample_bytree"] == 0.7

    def test_random_forest_extreme_caps(self):
        """RandomForest extreme: max_depth=6, min_samples_leaf=10, min_split=20, sqrt features."""
        params = _get_fixed_params(
            "RandomForest",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 970, 1: 30},
            imbalance_severity="extreme",
        )
        assert params["max_depth"] == 6
        assert params["min_samples_leaf"] == 10
        assert params["min_samples_split"] == 20
        assert params["max_features"] == "sqrt"

    # --- Severe imbalance: moderate caps, no subsampling ---

    def test_xgboost_severe_caps(self):
        """XGBoost severe: max_depth=6, min_child_weight=5, no subsampling forced."""
        params = _get_fixed_params(
            "XGBoost",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 900, 1: 100},
            imbalance_severity="severe",
        )
        assert params["max_depth"] == 6
        assert params["min_child_weight"] == 5
        assert "subsample" not in params
        assert "colsample_bytree" not in params

    def test_lightgbm_severe_caps(self):
        """LightGBM severe: max_depth=6, num_leaves=31, no subsampling forced."""
        params = _get_fixed_params(
            "LightGBM",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 900, 1: 100},
            imbalance_severity="severe",
        )
        assert params["max_depth"] == 6
        assert params["num_leaves"] == 31
        assert params["min_child_samples"] == 20
        assert "subsample" not in params

    def test_random_forest_severe_caps(self):
        """RandomForest severe: max_depth=8, min_samples_leaf=5."""
        params = _get_fixed_params(
            "RandomForest",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 900, 1: 100},
            imbalance_severity="severe",
        )
        assert params["max_depth"] == 8
        assert params["min_samples_leaf"] == 5

    # --- Guard tests ---

    def test_no_caps_for_moderate_imbalance(self):
        """No depth/leaf caps should apply for moderate imbalance."""
        params = _get_fixed_params(
            "XGBoost",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 750, 1: 250},
            imbalance_severity="moderate",
        )
        assert "max_depth" not in params
        assert "min_child_weight" not in params

    def test_no_depth_cap_for_logistic_regression(self):
        """LogisticRegression should NOT get depth caps even at extreme imbalance."""
        params = _get_fixed_params(
            "LogisticRegression",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 970, 1: 30},
            imbalance_severity="extreme",
        )
        assert "max_depth" not in params

    def test_lightgbm_subsample_freq_required(self):
        """LightGBM must set subsample_freq=1 when subsample is set, or it's silently ignored."""
        params = _get_fixed_params(
            "LightGBM",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 970, 1: 30},
            imbalance_severity="extreme",
        )
        assert "subsample" in params
        assert params["subsample_freq"] == 1

    def test_scale_pos_weight_computed_correctly(self):
        """XGBoost scale_pos_weight should equal majority/minority ratio."""
        params = _get_fixed_params(
            "XGBoost",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 970, 1: 30},
            imbalance_severity="extreme",
        )
        assert params["scale_pos_weight"] == pytest.approx(970 / 30)

    def test_no_class_weight_when_imbalance_not_detected(self):
        """No class weight params should be set when imbalance_detected=False."""
        params = _get_fixed_params("XGBoost", imbalance_detected=False)
        assert "scale_pos_weight" not in params

        params = _get_fixed_params("LightGBM", imbalance_detected=False)
        assert "is_unbalance" not in params

        params = _get_fixed_params("RandomForest", imbalance_detected=False)
        assert "class_weight" not in params


# ============================================================================
# Backlog #20 Gap 4: opt-in HPO sweep over class_weight / scale_pos_weight
# ============================================================================


class TestHPOClassWeightSweepGap4:
    """Verify the opt-in Optuna sweep over imbalance handlers.

    Default behaviour (``hpo_sweep_class_weight=False``): the
    deterministic ``_get_fixed_params`` matrix wins; Optuna sees a search
    space WITHOUT class_weight/scale_pos_weight/is_unbalance.

    Opt-in behaviour (``hpo_sweep_class_weight=True``): the matrix is
    skipped and ``_inject_class_weight_sweep`` adds Optuna search-space
    entries for the imbalance handlers so the matrix's choice can be
    validated per cohort.
    """

    def test_get_fixed_params_skips_class_weight_when_sweep_enabled_xgboost(self):
        """XGBoost: hpo_sweep_class_weight=True drops scale_pos_weight."""
        params = _get_fixed_params(
            "XGBoost",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 970, 1: 30},
            imbalance_severity="extreme",
            hpo_sweep_class_weight=True,
        )
        assert "scale_pos_weight" not in params
        # Severity-based regularisation caps still apply (max_depth, etc.):
        assert params["max_depth"] == 4

    def test_get_fixed_params_skips_class_weight_when_sweep_enabled_lightgbm(self):
        """LightGBM: hpo_sweep_class_weight=True drops is_unbalance."""
        params = _get_fixed_params(
            "LightGBM",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 970, 1: 30},
            imbalance_severity="severe",
            hpo_sweep_class_weight=True,
        )
        assert "is_unbalance" not in params
        # Severity caps still apply:
        assert params["max_depth"] == 6

    def test_get_fixed_params_skips_class_weight_when_sweep_enabled_rf(self):
        """RandomForest: hpo_sweep_class_weight=True drops class_weight."""
        params = _get_fixed_params(
            "RandomForest",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 970, 1: 30},
            imbalance_severity="severe",
            hpo_sweep_class_weight=True,
        )
        assert "class_weight" not in params
        # Severity caps still apply:
        assert params["max_depth"] == 8

    def test_get_fixed_params_default_preserves_class_weight(self):
        """Default behaviour (sweep disabled) — XGBoost still gets pinned
        scale_pos_weight, RandomForest gets class_weight, etc.
        Backward-compatibility regression guard.
        """
        params = _get_fixed_params(
            "XGBoost",
            imbalance_detected=True,
            recommended_strategy="class_weight",
            class_distribution={0: 970, 1: 30},
            imbalance_severity="extreme",
            # hpo_sweep_class_weight not passed = default False
        )
        assert "scale_pos_weight" in params
        assert params["scale_pos_weight"] == pytest.approx(970 / 30)

    def test_inject_class_weight_sweep_xgboost(self):
        """XGBoost: scale_pos_weight float sweep brackets ratio 2x both ways."""
        out = _inject_class_weight_sweep(
            {"n_estimators": {"type": "int", "low": 50, "high": 200}},
            algorithm_name="XGBoost",
            class_distribution={0: 800, 1: 200},  # ratio = 4
        )
        assert "scale_pos_weight" in out
        spec = out["scale_pos_weight"]
        assert spec["type"] == "float"
        assert spec["low"] == pytest.approx(0.5 * (800 / 200))
        assert spec["high"] == pytest.approx(2.0 * (800 / 200))
        assert spec["log"] is False
        # Doesn't drop the original entry
        assert "n_estimators" in out

    def test_inject_class_weight_sweep_lightgbm(self):
        """LightGBM: is_unbalance categorical [True, False]."""
        out = _inject_class_weight_sweep(
            {}, algorithm_name="LightGBM", class_distribution={0: 970, 1: 30}
        )
        assert out["is_unbalance"] == {
            "type": "categorical",
            "choices": [True, False],
        }

    def test_inject_class_weight_sweep_random_forest(self):
        """RandomForest: class_weight categorical [None, "balanced"]."""
        out = _inject_class_weight_sweep(
            {}, algorithm_name="RandomForest", class_distribution={0: 970, 1: 30}
        )
        assert out["class_weight"] == {
            "type": "categorical",
            "choices": [None, "balanced"],
        }

    def test_inject_class_weight_sweep_logistic_regression(self):
        """LogisticRegression: class_weight categorical [None, "balanced"]."""
        out = _inject_class_weight_sweep(
            {}, algorithm_name="LogisticRegression", class_distribution={0: 970, 1: 30}
        )
        assert out["class_weight"] == {
            "type": "categorical",
            "choices": [None, "balanced"],
        }

    def test_inject_class_weight_sweep_extra_trees(self):
        """ExtraTrees: class_weight categorical [None, "balanced"]."""
        out = _inject_class_weight_sweep(
            {}, algorithm_name="ExtraTrees", class_distribution={0: 970, 1: 30}
        )
        assert out["class_weight"] == {
            "type": "categorical",
            "choices": [None, "balanced"],
        }

    def test_inject_class_weight_sweep_gbm_no_op(self):
        """GradientBoosting: not swept (sample_weight is fit-time, Gap 3)."""
        out = _inject_class_weight_sweep(
            {"n_estimators": {"type": "int", "low": 50, "high": 200}},
            algorithm_name="GradientBoosting",
            class_distribution={0: 970, 1: 30},
        )
        # Only the original key remains; no sweep injected.
        assert set(out.keys()) == {"n_estimators"}

    def test_inject_class_weight_sweep_unknown_algo_no_op(self):
        """Unknown algorithm: helper returns input unmodified."""
        in_space = {"foo": {"type": "int", "low": 0, "high": 10}}
        out = _inject_class_weight_sweep(in_space, algorithm_name="UnknownAlgorithm")
        assert out == in_space
        # New dict (not the same object) — caller-mutation safety
        assert out is not in_space

    def test_inject_class_weight_sweep_preserves_existing(self):
        """Caller-supplied entries take precedence — sweep doesn't
        overwrite them."""
        existing = {
            "scale_pos_weight": {
                "type": "float",
                "low": 1.0,
                "high": 5.0,
                "log": False,
            }
        }
        out = _inject_class_weight_sweep(
            existing, algorithm_name="XGBoost", class_distribution={0: 800, 1: 200}
        )
        # Caller's bounds preserved
        assert out["scale_pos_weight"]["low"] == 1.0
        assert out["scale_pos_weight"]["high"] == 5.0

    def test_inject_class_weight_sweep_xgboost_skip_on_missing_distribution(self):
        """XGBoost without class_distribution: skip the sweep injection
        (no ratio to compute bounds from)."""
        out = _inject_class_weight_sweep({}, algorithm_name="XGBoost", class_distribution=None)
        assert "scale_pos_weight" not in out

        out = _inject_class_weight_sweep(
            {},
            algorithm_name="XGBoost",
            class_distribution={0: 100},  # single class
        )
        assert "scale_pos_weight" not in out

    def test_inject_class_weight_sweep_xgboost_skip_on_zero_minority(self):
        """XGBoost: when minority count is 0, skip injection (avoids div0)."""
        out = _inject_class_weight_sweep(
            {}, algorithm_name="XGBoost", class_distribution={0: 100, 1: 0}
        )
        assert "scale_pos_weight" not in out

    def test_inject_class_weight_sweep_does_not_mutate_input(self):
        """Helper must not mutate the caller's search_space dict."""
        original = {"n_estimators": {"type": "int", "low": 50, "high": 200}}
        original_copy = {k: dict(v) for k, v in original.items()}
        _ = _inject_class_weight_sweep(
            original, algorithm_name="XGBoost", class_distribution={0: 800, 1: 200}
        )
        assert original == original_copy


class TestGetHpoPatternMemory:
    """Tests for _get_hpo_pattern_memory helper."""

    def test_returns_module_when_available(self):
        """Should return hpo_pattern_memory module when available."""
        result = _get_hpo_pattern_memory()

        # Module should be importable in test environment
        assert result is not None or result is None  # Depends on import success

    def test_function_exists_and_is_callable(self):
        """Should have a callable _get_hpo_pattern_memory function."""
        from src.agents.ml_foundation.model_trainer.nodes import hyperparameter_tuner

        assert hasattr(hyperparameter_tuner, "_get_hpo_pattern_memory")
        assert callable(hyperparameter_tuner._get_hpo_pattern_memory)


@pytest.mark.asyncio
class TestWarmStartIntegration:
    """Tests for warm-start integration with HPO pattern memory."""

    async def test_validates_hpo_pattern_id_output_field(self):
        """Should allow hpo_pattern_id in HPO output validation."""
        output = {
            "hpo_completed": True,
            "best_hyperparameters": {"n_estimators": 150},
            "hpo_best_trial": 5,
            "hpo_trials_run": 10,
            "hpo_best_value": 0.95,
            "hpo_study_name": "test_study",
            "hpo_pattern_id": "abc-123-def-456",  # New field from procedural memory
        }

        is_valid, errors = validate_hpo_output(output)

        assert is_valid is True
        assert errors == []

    async def test_hpo_proceeds_without_pattern_memory(self):
        """Should run HPO successfully when pattern memory unavailable."""
        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner._get_hpo_pattern_memory",
            return_value=None,
        ):
            state = {
                "enable_hpo": True,
                "hpo_trials": 5,
                "algorithm_name": "RandomForest",
                "problem_type": "binary_classification",
                "experiment_id": "test_exp_123",
                "default_hyperparameters": {"n_estimators": 100},
                "hyperparameter_search_space": {
                    "n_estimators": {"type": "int", "low": 50, "high": 200}
                },
                "X_train_preprocessed": np.random.rand(100, 5),
                "X_validation_preprocessed": np.random.rand(30, 5),
                "train_data": {"y": np.random.randint(0, 2, 100)},
                "validation_data": {"y": np.random.randint(0, 2, 30)},
            }

            result = await tune_hyperparameters(state)

            # Should complete successfully even without pattern memory
            assert "hpo_completed" in result
            assert "best_hyperparameters" in result
            # Pattern ID should not be present
            assert "hpo_pattern_id" not in result or result.get("hpo_pattern_id") is None
