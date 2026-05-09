"""Tests for model training node.

Tests the train_model function with various algorithms and configurations.
"""

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.model_trainer_node import (
    _filter_hyperparameters,
    _get_framework,
    _get_model_class_dynamic,
    _prepare_fit_params,
    train_model,
)

# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def binary_classification_state():
    """Create state for binary classification."""
    np.random.seed(42)
    return {
        "algorithm_name": "RandomForest",
        "problem_type": "binary_classification",
        "best_hyperparameters": {"n_estimators": 10, "max_depth": 3},
        "X_train_preprocessed": np.random.rand(100, 5),
        "X_validation_preprocessed": np.random.rand(30, 5),
        "train_data": {"y": np.random.randint(0, 2, 100)},
        "validation_data": {"y": np.random.randint(0, 2, 30)},
        "early_stopping": False,
    }


@pytest.fixture
def regression_state():
    """Create state for regression."""
    np.random.seed(42)
    return {
        "algorithm_name": "RandomForest",
        "problem_type": "regression",
        "best_hyperparameters": {"n_estimators": 10, "max_depth": 3},
        "X_train_preprocessed": np.random.rand(100, 5),
        "train_data": {"y": np.random.rand(100)},
        "early_stopping": False,
    }


# ============================================================================
# Test train_model function
# ============================================================================


@pytest.mark.asyncio
class TestTrainModel:
    """Test core model training."""

    async def test_trains_random_forest_classifier(self, binary_classification_state):
        """Should train RandomForest classifier successfully."""
        result = await train_model(binary_classification_state)

        assert "error" not in result
        assert result["trained_model"] is not None
        assert result["training_status"] == "completed"
        assert result["algorithm_name"] == "RandomForest"
        assert result["framework"] == "sklearn"

    async def test_trains_random_forest_regressor(self, regression_state):
        """Should train RandomForest regressor successfully."""
        result = await train_model(regression_state)

        assert "error" not in result
        assert result["trained_model"] is not None
        assert result["training_status"] == "completed"

    async def test_trains_logistic_regression(self, binary_classification_state):
        """Should train LogisticRegression successfully."""
        binary_classification_state["algorithm_name"] = "LogisticRegression"
        binary_classification_state["best_hyperparameters"] = {"C": 1.0}

        result = await train_model(binary_classification_state)

        assert "error" not in result
        assert result["trained_model"] is not None
        assert result["framework"] == "sklearn"

    async def test_trains_gradient_boosting(self, binary_classification_state):
        """Should train GradientBoosting successfully."""
        binary_classification_state["algorithm_name"] = "GradientBoosting"
        binary_classification_state["best_hyperparameters"] = {
            "n_estimators": 10,
            "max_depth": 3,
        }

        result = await train_model(binary_classification_state)

        assert "error" not in result
        assert result["trained_model"] is not None

    async def test_trains_extra_trees(self, binary_classification_state):
        """Should train ExtraTrees successfully."""
        binary_classification_state["algorithm_name"] = "ExtraTrees"
        binary_classification_state["best_hyperparameters"] = {
            "n_estimators": 10,
            "max_depth": 3,
        }

        result = await train_model(binary_classification_state)

        assert "error" not in result
        assert result["trained_model"] is not None

    async def test_trains_ridge_regressor(self, regression_state):
        """Should train Ridge regressor successfully."""
        regression_state["algorithm_name"] = "Ridge"
        regression_state["best_hyperparameters"] = {"alpha": 1.0}

        result = await train_model(regression_state)

        assert "error" not in result
        assert result["trained_model"] is not None

    async def test_trains_lasso_regressor(self, regression_state):
        """Should train Lasso regressor successfully."""
        regression_state["algorithm_name"] = "Lasso"
        regression_state["best_hyperparameters"] = {"alpha": 0.1}

        result = await train_model(regression_state)

        assert "error" not in result
        assert result["trained_model"] is not None

    async def test_records_training_duration(self, binary_classification_state):
        """Should record training duration."""
        result = await train_model(binary_classification_state)

        assert result["training_duration_seconds"] >= 0
        assert result["training_started_at"] is not None
        assert result["training_completed_at"] is not None

    async def test_error_when_missing_training_data(self):
        """Should return error when training data missing."""
        state = {
            "algorithm_name": "RandomForest",
            "best_hyperparameters": {},
        }

        result = await train_model(state)

        assert "error" in result
        assert result["error_type"] == "missing_training_data"
        assert result["training_status"] == "failed"

    async def test_error_when_missing_algorithm_name(self, binary_classification_state):
        """Should return error when algorithm_name missing."""
        del binary_classification_state["algorithm_name"]

        result = await train_model(binary_classification_state)

        assert "error" in result
        assert result["error_type"] == "missing_algorithm_name"

    async def test_error_for_unsupported_algorithm(self, binary_classification_state):
        """Should return error for unsupported algorithm."""
        binary_classification_state["algorithm_name"] = "UnsupportedAlgorithm"

        result = await train_model(binary_classification_state)

        assert "error" in result
        assert result["error_type"] == "unsupported_algorithm"

    async def test_filters_incompatible_hyperparameters(self, binary_classification_state):
        """Should filter out incompatible hyperparameters."""
        binary_classification_state["best_hyperparameters"] = {
            "n_estimators": 10,
            "max_depth": 3,
            "invalid_param": "should_be_filtered",
            "learning_rate": 0.1,  # Not valid for RandomForest
        }

        result = await train_model(binary_classification_state)

        # Should still train successfully despite invalid params
        assert "error" not in result
        assert result["trained_model"] is not None


# ============================================================================
# Test helper functions
# ============================================================================


class TestGetModelClassDynamic:
    """Test model class lookup."""

    def test_gets_random_forest_classifier(self):
        """Should get RandomForestClassifier for binary classification."""
        model_class = _get_model_class_dynamic("RandomForest", "binary_classification")
        assert model_class is not None
        assert "RandomForest" in model_class.__name__

    def test_gets_random_forest_regressor(self):
        """Should get RandomForestRegressor for regression."""
        model_class = _get_model_class_dynamic("RandomForest", "regression")
        assert model_class is not None
        assert "RandomForest" in model_class.__name__
        assert "Regressor" in model_class.__name__

    def test_gets_logistic_regression(self):
        """Should get LogisticRegression."""
        model_class = _get_model_class_dynamic("LogisticRegression", "binary_classification")
        assert model_class is not None

    def test_gets_ridge(self):
        """Should get Ridge regressor."""
        model_class = _get_model_class_dynamic("Ridge", "regression")
        assert model_class is not None

    def test_gets_gradient_boosting_classifier(self):
        """Should get GradientBoostingClassifier."""
        model_class = _get_model_class_dynamic("GradientBoosting", "binary_classification")
        assert model_class is not None

    def test_returns_none_for_unknown_algorithm(self):
        """Should return None for unknown algorithm."""
        model_class = _get_model_class_dynamic("UnknownAlgorithm", "binary_classification")
        assert model_class is None

    def test_gets_ngboost_binary_wrapper(self):
        """Phase 1 W2 day-1: NGBoost binary classification resolves to wrapper.

        Reference: shard 19 §A.4 mirror in trainer fallback path.
        """
        from src.mlops.wrappers.ngboost_wrapper import NGBoostBinaryClassifier

        model_class = _get_model_class_dynamic("NGBoost", "binary_classification")
        assert model_class is NGBoostBinaryClassifier

    def test_gets_ngboost_regressor(self):
        """Phase 1 W2 day-1: NGBoost regression resolves to NGBRegressor."""
        from ngboost import NGBRegressor

        model_class = _get_model_class_dynamic("NGBoost", "regression")
        assert model_class is NGBRegressor


class TestFilterHyperparameters:
    """Test hyperparameter filtering."""

    def test_filters_random_forest_params(self):
        """Should keep valid RandomForest params and filter invalid."""
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,  # Invalid for RF
            "invalid_param": "value",
        }

        filtered = _filter_hyperparameters("RandomForest", params)

        assert "n_estimators" in filtered
        assert "max_depth" in filtered
        assert "learning_rate" not in filtered
        assert "invalid_param" not in filtered

    def test_adds_common_params(self):
        """Should add common params like random_state."""
        params = {"n_estimators": 100}

        filtered = _filter_hyperparameters("RandomForest", params)

        assert "random_state" in filtered
        assert "n_jobs" in filtered

    def test_filters_logistic_regression_params(self):
        """Should filter LogisticRegression params correctly."""
        params = {
            "C": 1.0,
            "penalty": "l2",
            "n_estimators": 100,  # Invalid for LR
        }

        filtered = _filter_hyperparameters("LogisticRegression", params)

        assert "C" in filtered
        assert "penalty" in filtered
        assert "n_estimators" not in filtered

    def test_filters_ngboost_params(self):
        """Phase 1 W2 day-1: NGBoost-specific params pass through; foreign reject.

        Reference: shard 19 §A.5 — NGBoost allowlist is n_estimators,
        learning_rate, minibatch_frac, col_sample, verbose, random_state,
        base_max_depth, base_min_samples_leaf.
        """
        params = {
            "n_estimators": 200,
            "learning_rate": 0.01,
            "base_max_depth": 4,
            "base_min_samples_leaf": 10,
            "subsample": 0.8,  # XGBoost-specific; should drop
            "num_leaves": 31,  # LightGBM-specific; should drop
        }

        filtered = _filter_hyperparameters("NGBoost", params)

        assert filtered["n_estimators"] == 200
        assert filtered["learning_rate"] == 0.01
        assert filtered["base_max_depth"] == 4
        assert filtered["base_min_samples_leaf"] == 10
        assert "subsample" not in filtered
        assert "num_leaves" not in filtered
        assert "random_state" in filtered  # common param injected

    def test_filters_conformal_params_compose_base_plus_common(self):
        """Phase 1 W2 day-3 (shard 19 §B.5): conformal allowlist =
        {method, cv, alpha, random_state} ∪ allowed_params[base].
        """
        params = {
            # Base (LogisticRegression):
            "C": 1.0,
            "penalty": "l2",
            # Conformal-common:
            "method": "lac",
            "cv": 5,
            "alpha": 0.10,
            # Foreign params should drop:
            "n_estimators": 200,
            "num_leaves": 31,
        }
        filtered = _filter_hyperparameters("LogisticRegression_Conformal", params)
        # Base params kept:
        assert filtered["C"] == 1.0
        assert filtered["penalty"] == "l2"
        # Conformal common kept:
        assert filtered["method"] == "lac"
        assert filtered["cv"] == 5
        assert filtered["alpha"] == 0.10
        # Foreign dropped:
        assert "n_estimators" not in filtered
        assert "num_leaves" not in filtered

    def test_filters_lightgbm_conformal_params(self):
        """LightGBM_Conformal allowlist composes LightGBM allowlist with conformal common."""
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "method": "lac",
            "cv": 5,
            "alpha": 0.10,
            "C": 1.0,  # Foreign (LR-specific); should drop
        }
        filtered = _filter_hyperparameters("LightGBM_Conformal", params)
        assert filtered["n_estimators"] == 300
        assert filtered["max_depth"] == 6
        assert filtered["learning_rate"] == 0.05
        assert filtered["num_leaves"] == 31
        assert filtered["method"] == "lac"
        assert filtered["cv"] == 5
        assert filtered["alpha"] == 0.10
        assert "C" not in filtered

    def test_get_model_class_dynamic_resolves_conformal(self):
        """Phase 1 W2 day-3 mirror (shard 19 §B.4): _get_model_class_dynamic
        resolves *_Conformal names identically to optuna_optimizer.get_model_class.
        """
        from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

        factory = _get_model_class_dynamic("LogisticRegression_Conformal", "binary_classification")
        assert factory is not None
        assert callable(factory)
        instance = factory(C=2.0, penalty="l2")
        assert isinstance(instance, MapieConformalBinaryClassifier)
        assert instance.base_estimator.C == 2.0

    def test_filters_lightgbm_conformal_injects_verbose_minus_one(self):
        """Cycle-9 codex F2: LightGBM_Conformal must inject verbose=-1 like
        bare LightGBM, otherwise day-5 integration smoke emits per-iteration
        log spam.
        """
        params = {"n_estimators": 200, "learning_rate": 0.05}
        filtered = _filter_hyperparameters("LightGBM_Conformal", params)
        assert filtered.get("verbose") == -1, (
            "LightGBM_Conformal allowlist should inject verbose=-1 default like bare LightGBM"
        )

    def test_filters_logistic_regression_conformal_injects_max_iter_1000(self):
        """Cycle-9 codex F3: LogisticRegression_Conformal must inject max_iter=1000
        like bare LogisticRegression, otherwise sklearn default 100 may emit
        ConvergenceWarning at day-5 integration smoke.
        """
        params = {"C": 1.0, "penalty": "l2"}
        filtered = _filter_hyperparameters("LogisticRegression_Conformal", params)
        assert filtered.get("max_iter") == 1000, (
            "LogisticRegression_Conformal allowlist should inject max_iter=1000 default"
        )

    def test_filters_lightgbm_monotone_uses_base_allowlist(self):
        """Phase 1 W2 day-4 (shard 19 §C.5): LightGBM_Monotone reuses LightGBM
        allowlist via the _Monotone-suffix branch. monotone_constraints must
        be in the allowlist (added to LightGBM base in this commit) but is
        NOT supplied via params — it's injected from state at fit time.
        Foreign params drop; verbose=-1 + monotone-related params survive.
        """
        params = {
            "n_estimators": 200,
            "max_depth": 5,
            "num_leaves": 31,
            "monotone_constraints": [1, 0, -1],  # injected at fit time normally
            "C": 1.0,  # Foreign (LR-specific); should drop
            "method": "lac",  # Foreign (conformal-only); should drop
        }
        filtered = _filter_hyperparameters("LightGBM_Monotone", params)
        assert filtered["n_estimators"] == 200
        assert filtered["max_depth"] == 5
        assert filtered["num_leaves"] == 31
        assert filtered["monotone_constraints"] == [1, 0, -1]
        assert filtered.get("verbose") == -1, (
            "LightGBM_Monotone should inject verbose=-1 like bare LightGBM"
        )
        assert "C" not in filtered
        assert "method" not in filtered

    def test_filters_xgboost_monotone_uses_base_allowlist(self):
        """XGBoost_Monotone reuses XGBoost allowlist. monotone_constraints
        is in the allowlist (added to XGBoost base) but the value comes from
        the trainer's fit-time injection, not params normally.
        """
        params = {
            "n_estimators": 200,
            "max_depth": 5,
            "subsample": 0.8,
            "monotone_constraints": "(1, 0, -1)",  # XGBoost string format
        }
        filtered = _filter_hyperparameters("XGBoost_Monotone", params)
        assert filtered["n_estimators"] == 200
        assert filtered["max_depth"] == 5
        assert filtered["subsample"] == 0.8
        assert filtered["monotone_constraints"] == "(1, 0, -1)"
        assert filtered.get("verbosity") == 0
        assert filtered.get("use_label_encoder") is False

    def test_get_model_class_dynamic_resolves_monotone(self):
        """Phase 1 W2 day-4 mirror (shard 19 §C.2): _Monotone variants resolve
        to the same class as the base estimator.
        """
        from lightgbm import LGBMClassifier

        cls = _get_model_class_dynamic("LightGBM_Monotone", "binary_classification")
        assert cls is LGBMClassifier


@pytest.mark.asyncio
class TestTrainModelMonotoneInjection:
    """Phase 1 W2 day-4 (shard 19 §C.4): when model_candidate carries
    `monotone_constraints_required=True`, train_model injects
    monotone_constraints from `state["monotone_vector"]` into filtered_params
    before model instantiation. Soft-fails to unconstrained training if
    monotone_vector is missing.
    """

    async def test_lightgbm_monotone_with_vector_injects_list(self, binary_classification_state):
        """LightGBM_Monotone with monotone_vector → list[int] in get_params()."""
        binary_classification_state["algorithm_name"] = "LightGBM_Monotone"
        binary_classification_state["best_hyperparameters"] = {
            "n_estimators": 5,
            "max_depth": 3,
            "learning_rate": 0.1,
        }
        binary_classification_state["model_candidate"] = {
            "monotone_constraints_required": True,
        }
        # 5 features (N_FEATURES) → 5-element monotone vector
        binary_classification_state["monotone_vector"] = [1, 0, -1, 0, 1]

        result = await train_model(binary_classification_state)

        assert "error" not in result
        model = result["trained_model"]
        params = model.get_params()
        # LightGBM expects a list
        assert params.get("monotone_constraints") == [1, 0, -1, 0, 1]

    async def test_xgboost_monotone_with_vector_injects_string(self, binary_classification_state):
        """XGBoost_Monotone with monotone_vector → tuple-string format."""
        binary_classification_state["algorithm_name"] = "XGBoost_Monotone"
        binary_classification_state["best_hyperparameters"] = {
            "n_estimators": 5,
            "max_depth": 3,
            "learning_rate": 0.1,
        }
        binary_classification_state["model_candidate"] = {
            "monotone_constraints_required": True,
        }
        binary_classification_state["monotone_vector"] = [1, 0, -1, 0, 1]

        result = await train_model(binary_classification_state)

        assert "error" not in result
        model = result["trained_model"]
        params = model.get_params()
        # XGBoost expects the tuple-string format
        assert params.get("monotone_constraints") == "(1, 0, -1, 0, 1)"

    async def test_monotone_required_but_vector_missing_soft_fails(
        self, binary_classification_state, caplog
    ):
        """When required flag is set but monotone_vector absent, the trainer
        logs a warning and trains without constraints (degraded path).
        """
        import logging

        binary_classification_state["algorithm_name"] = "LightGBM_Monotone"
        binary_classification_state["best_hyperparameters"] = {
            "n_estimators": 5,
            "max_depth": 3,
        }
        binary_classification_state["model_candidate"] = {
            "monotone_constraints_required": True,
        }
        # NO monotone_vector in state.

        with caplog.at_level(logging.WARNING):
            result = await train_model(binary_classification_state)

        assert "error" not in result
        # Trained successfully without constraints:
        assert result["trained_model"] is not None
        # Warning was emitted:
        warning_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            "requires monotone_vector" in m and "training without constraints" in m
            for m in warning_msgs
        ), f"Expected soft-fail warning; got: {warning_msgs}"

    async def test_no_monotone_required_flag_skips_injection(self, binary_classification_state):
        """When model_candidate.monotone_constraints_required is False or
        absent, the injection block is a no-op even if state has a vector
        (defensive: legacy candidates unchanged).
        """
        binary_classification_state["algorithm_name"] = "RandomForest"
        binary_classification_state["best_hyperparameters"] = {"n_estimators": 5}
        # No model_candidate or required=False; even with vector present, no injection.
        binary_classification_state["monotone_vector"] = [1, 0, -1, 0, 1]

        result = await train_model(binary_classification_state)

        assert "error" not in result
        # RandomForest doesn't accept monotone_constraints — would crash if injected.
        # No crash → no injection → defensive guard works.
        assert result["trained_model"] is not None

    async def test_monotone_vector_length_mismatch_soft_degrades_with_warning(
        self, binary_classification_state, caplog
    ):
        """Cycle-10 codex IMPORTANT finding fix: per shard 19 §H risk table,
        a vector-length mismatch must soft-degrade to unconstrained training
        with a WARNING — NOT hard-fail at LightGBM instantiation. Without this
        pre-validation, the trainer would raise ValueError mid-fit and route
        to error_type=instantiation_failed; the §H spec is to log + continue.
        """
        import logging

        binary_classification_state["algorithm_name"] = "LightGBM_Monotone"
        binary_classification_state["best_hyperparameters"] = {
            "n_estimators": 5,
            "max_depth": 3,
        }
        binary_classification_state["model_candidate"] = {
            "monotone_constraints_required": True,
        }
        # Vector LENGTH mismatch: 3-element vector but X_train has N_FEATURES=5.
        binary_classification_state["monotone_vector"] = [1, 0, -1]

        with caplog.at_level(logging.WARNING):
            result = await train_model(binary_classification_state)

        assert "error" not in result, "Length mismatch should soft-degrade, not hard-fail"
        assert result["trained_model"] is not None
        warning_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("monotone_vector length" in m or "length mismatch" in m for m in warning_msgs), (
            f"Expected length-mismatch warning; got: {warning_msgs}"
        )
        # Trained model has NO monotone_constraints (the soft-degrade dropped them):
        params = result["trained_model"].get_params()
        assert (
            params.get("monotone_constraints") is None
            or params.get("monotone_constraints") == "None"
        )


class TestGetFramework:
    """Test framework identification."""

    def test_identifies_sklearn_algorithms(self):
        """Should identify sklearn algorithms."""
        assert _get_framework("RandomForest") == "sklearn"
        assert _get_framework("LogisticRegression") == "sklearn"
        assert _get_framework("Ridge") == "sklearn"
        assert _get_framework("GradientBoosting") == "sklearn"

    def test_identifies_xgboost(self):
        """Should identify XGBoost."""
        assert _get_framework("XGBoost") == "xgboost"

    def test_identifies_lightgbm(self):
        """Should identify LightGBM."""
        assert _get_framework("LightGBM") == "lightgbm"

    def test_identifies_econml_algorithms(self):
        """Should identify econml algorithms."""
        assert _get_framework("CausalForest") == "econml"
        assert _get_framework("LinearDML") == "econml"
        assert _get_framework("SLearner") == "econml"
        assert _get_framework("DRLearner") == "econml"
        assert _get_framework("TLearner") == "econml"
        assert _get_framework("XLearner") == "econml"

    def test_returns_unknown_for_unrecognized(self):
        """Should return 'unknown' for unrecognized algorithms."""
        assert _get_framework("UnknownAlgorithm") == "unknown"


# ============================================================================
# Backlog #20 Gap 3: GradientBoosting sample_weight bridge
# ============================================================================


class TestPrepareFitParamsGBMSampleWeight:
    """Verify _prepare_fit_params wires sample_weight for GBM under imbalance.

    sklearn's GradientBoostingClassifier rejects class_weight at construction
    but accepts sample_weight at .fit(). The bridge mirrors the constructor-
    level handling that XGBoost/LightGBM/RandomForest/LogReg/ExtraTrees get
    in hyperparameter_tuner._get_fixed_params, so GBM is no longer the odd
    one out under class imbalance.
    """

    def test_gbm_imbalance_adds_sample_weight(self):
        """GradientBoosting + imbalance_detected=True populates sample_weight."""
        rng = np.random.default_rng(42)
        # 95:5 imbalance, n=100
        y_train = np.concatenate([np.zeros(95), np.ones(5)]).astype(int)
        rng.shuffle(y_train)

        fit_params = _prepare_fit_params(
            algorithm_name="GradientBoosting",
            early_stopping=False,
            early_stopping_patience=10,
            X_validation=None,
            y_validation=None,
            imbalance_detected=True,
            y_train=y_train,
        )

        assert "sample_weight" in fit_params
        sw = fit_params["sample_weight"]
        assert isinstance(sw, np.ndarray)
        assert sw.shape == (100,)
        # "balanced" weights: minority class gets larger weight than majority
        # n_samples / (n_classes * n_majority) for class 0; same with n_minority for class 1
        majority_weight = sw[y_train == 0][0]
        minority_weight = sw[y_train == 1][0]
        assert minority_weight > majority_weight
        # All samples in same class share the same weight
        assert np.allclose(sw[y_train == 0], majority_weight)
        assert np.allclose(sw[y_train == 1], minority_weight)

    def test_gbm_no_imbalance_skips_sample_weight(self):
        """GradientBoosting + imbalance_detected=False omits sample_weight."""
        y_train = np.random.randint(0, 2, 100)

        fit_params = _prepare_fit_params(
            algorithm_name="GradientBoosting",
            early_stopping=False,
            early_stopping_patience=10,
            X_validation=None,
            y_validation=None,
            imbalance_detected=False,
            y_train=y_train,
        )

        assert "sample_weight" not in fit_params

    def test_non_gbm_algorithms_no_sample_weight_via_bridge(self):
        """RandomForest/XGBoost/LightGBM don't get sample_weight from this bridge.

        They handle imbalance via class_weight/scale_pos_weight/is_unbalance
        constructor params (in _get_fixed_params), not fit-time sample_weight.
        """
        y_train = np.concatenate([np.zeros(95), np.ones(5)]).astype(int)

        for algo in ["RandomForest", "XGBoost", "LightGBM", "LogisticRegression", "ExtraTrees"]:
            fit_params = _prepare_fit_params(
                algorithm_name=algo,
                early_stopping=False,
                early_stopping_patience=10,
                X_validation=None,
                y_validation=None,
                imbalance_detected=True,
                y_train=y_train,
            )
            assert "sample_weight" not in fit_params, (
                f"{algo} should not get sample_weight via _prepare_fit_params"
            )

    def test_gbm_single_class_y_skips_sample_weight(self):
        """Degenerate single-class y_train short-circuits without crashing.

        compute_sample_weight raises on single-class y; the helper guards
        regardless so synthesized states stay safe.
        """
        y_train = np.zeros(100, dtype=int)  # all-zero, degenerate

        fit_params = _prepare_fit_params(
            algorithm_name="GradientBoosting",
            early_stopping=False,
            early_stopping_patience=10,
            X_validation=None,
            y_validation=None,
            imbalance_detected=True,
            y_train=y_train,
        )

        assert "sample_weight" not in fit_params

    def test_gbm_imbalance_with_early_stopping_keeps_both(self):
        """Early stopping fit_params and sample_weight coexist for GBM.

        Sklearn GBM uses validation_fraction/n_iter_no_change in the
        constructor for early stopping (not fit_params), so the early-stopping
        branch in _prepare_fit_params is a no-op for GBM. The sample_weight
        bridge runs independently.
        """
        y_train = np.concatenate([np.zeros(95), np.ones(5)]).astype(int)
        X_val = np.random.rand(30, 5)
        y_val = np.random.randint(0, 2, 30)

        fit_params = _prepare_fit_params(
            algorithm_name="GradientBoosting",
            early_stopping=True,
            early_stopping_patience=10,
            X_validation=X_val,
            y_validation=y_val,
            imbalance_detected=True,
            y_train=y_train,
        )

        assert "sample_weight" in fit_params
        # GBM early stopping doesn't add eval_set/callbacks (that's XGB/LGBM)
        assert "eval_set" not in fit_params
        assert "callbacks" not in fit_params

    def test_gbm_y_train_none_skips_sample_weight(self):
        """Missing y_train is a defensive no-op (don't crash callers)."""
        fit_params = _prepare_fit_params(
            algorithm_name="GradientBoosting",
            early_stopping=False,
            early_stopping_patience=10,
            X_validation=None,
            y_validation=None,
            imbalance_detected=True,
            y_train=None,
        )

        assert "sample_weight" not in fit_params

    def test_gbm_pandas_series_y_train_handled(self):
        """y_train as a pandas Series flattens via np.asarray."""
        pd = pytest.importorskip("pandas")
        y_train = pd.Series(np.concatenate([np.zeros(95), np.ones(5)]).astype(int))

        fit_params = _prepare_fit_params(
            algorithm_name="GradientBoosting",
            early_stopping=False,
            early_stopping_patience=10,
            X_validation=None,
            y_validation=None,
            imbalance_detected=True,
            y_train=y_train,
        )

        assert "sample_weight" in fit_params
        assert fit_params["sample_weight"].shape == (100,)


@pytest.mark.asyncio
class TestTrainModelGBMSampleWeightIntegration:
    """End-to-end check that train_model wires sample_weight into model.fit."""

    async def test_gbm_imbalance_detected_changes_predictions(self):
        """Compare GBM trained with vs without imbalance_detected=True.

        Predictions on the minority class should shift when sample_weight
        rebalancing is in effect, confirming the wiring reaches model.fit.

        Codex pass-2 LOW-3 fix: previously this test ran ``rng.shuffle(y)``
        AFTER applying ``X[y == 1, 0] += 1.5``, which decoupled X/y
        alignment so the "informative signal" was effectively destroyed.
        The test still passed because GBM with vs without sample_weight
        produces different gradients on any input, but it wasn't
        measuring the intended class-conditional signal recovery. Apply
        a single permutation to BOTH X and y so alignment is preserved
        and the class-conditional signal survives the shuffle.
        """
        rng = np.random.default_rng(0)
        n = 200
        # 95:5 split with informative signal in feature 0
        y = np.concatenate([np.zeros(190), np.ones(10)]).astype(int)
        X = rng.standard_normal((n, 4))
        X[y == 1, 0] += 1.5  # informative signal for minority
        # Codex LOW-3: same permutation for X and y to preserve alignment.
        order = rng.permutation(n)
        X = X[order]
        y = y[order]

        base_state = {
            "algorithm_name": "GradientBoosting",
            "problem_type": "binary_classification",
            "best_hyperparameters": {"n_estimators": 20, "max_depth": 3, "random_state": 42},
            "X_train_preprocessed": X,
            "X_validation_preprocessed": X[:30],
            "train_data": {"y": y},
            "validation_data": {"y": y[:30]},
            "early_stopping": False,
        }

        # Without imbalance_detected
        state_no_imb = {**base_state, "imbalance_detected": False}
        result_no_imb = await train_model(state_no_imb)
        # With imbalance_detected
        state_imb = {**base_state, "imbalance_detected": True}
        result_imb = await train_model(state_imb)

        assert "error" not in result_no_imb
        assert "error" not in result_imb

        model_no_imb = result_no_imb["trained_model"]
        model_imb = result_imb["trained_model"]

        # The reweighted model should give DIFFERENT minority-class scores
        # than the unweighted one. Equality would mean sample_weight was a
        # no-op, i.e. the wiring didn't reach model.fit.
        proba_no_imb = model_no_imb.predict_proba(X)[:, 1]
        proba_imb = model_imb.predict_proba(X)[:, 1]

        assert not np.allclose(proba_no_imb, proba_imb), (
            "GBM with imbalance_detected=True produced identical scores to "
            "the unweighted version — sample_weight wiring is not reaching "
            "model.fit (backlog #20 Gap 3 regression)."
        )

        # Codex LOW-3: verify the rebalanced model also moves minority-
        # class scores UP relative to the unweighted version, confirming
        # the class-conditional signal survived the shuffle and the
        # weighting is doing what it should. Tightening the assertion
        # past mere inequality.
        minority_proba_no_imb = proba_no_imb[y == 1].mean()
        minority_proba_imb = proba_imb[y == 1].mean()
        assert minority_proba_imb > minority_proba_no_imb, (
            f"Reweighted GBM minority-class mean proba "
            f"({minority_proba_imb:.4f}) should exceed unweighted "
            f"({minority_proba_no_imb:.4f}); reweighting must lift "
            f"the minority signal."
        )
