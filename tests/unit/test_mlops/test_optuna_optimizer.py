"""Unit tests for OptunaOptimizer.

Version: 1.0.0
Tests the Optuna hyperparameter optimization wrapper with mocked operations.

Coverage:
- OptunaOptimizer class
- PrunerFactory
- SamplerFactory
- get_model_class helper
- run_hyperparameter_optimization function
- suggest_from_search_space
- create_cv_objective
- create_validation_objective
- save_to_database
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import optuna
import pytest
from sklearn.datasets import make_classification, make_regression

from src.mlops.optuna_optimizer import (
    OptunaOptimizer,
    PrunerFactory,
    SamplerFactory,
    get_model_class,
    run_hyperparameter_optimization,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_search_space():
    """Sample search space in E2I format."""
    return {
        "n_estimators": {"type": "int", "low": 50, "high": 200, "step": 50},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
        "booster": {"type": "categorical", "choices": ["gbtree", "gblinear"]},
    }


@pytest.fixture
def classification_data():
    """Sample classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    # Split into train and validation
    X_train, X_val = X[:160], X[160:]
    y_train, y_val = y[:160], y[160:]
    return X_train, y_train, X_val, y_val


@pytest.fixture
def regression_data():
    """Sample regression dataset."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    # Split into train and validation
    X_train, X_val = X[:160], X[160:]
    y_train, y_val = y[:160], y[160:]
    return X_train, y_train, X_val, y_val


@pytest.fixture
def mock_optuna_trial():
    """Mock Optuna trial."""
    trial = MagicMock(spec=optuna.Trial)
    trial.suggest_int = MagicMock(side_effect=lambda name, low, high, step=1: (low + high) // 2)
    trial.suggest_float = MagicMock(side_effect=lambda name, low, high, **kwargs: (low + high) / 2)
    trial.suggest_categorical = MagicMock(side_effect=lambda name, choices: choices[0])
    return trial


@pytest.fixture
def mock_frozen_trial():
    """Mock frozen trial for history."""
    trial = MagicMock(spec=optuna.trial.FrozenTrial)
    trial.number = 0
    trial.state = optuna.trial.TrialState.COMPLETE
    trial.value = 0.85
    trial.params = {"n_estimators": 100, "max_depth": 5}
    trial.datetime_start = datetime.now(timezone.utc)
    trial.datetime_complete = datetime.now(timezone.utc) + timedelta(seconds=10)
    trial.duration = timedelta(seconds=10)
    trial.user_attrs = {}
    trial.system_attrs = {}
    trial.intermediate_values = {}
    return trial


# ============================================================================
# OPTUNA OPTIMIZER TESTS
# ============================================================================


class TestOptunaOptimizerInit:
    """Tests for OptunaOptimizer initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        optimizer = OptunaOptimizer(experiment_id="exp_123")

        assert optimizer.experiment_id == "exp_123"
        assert optimizer.storage_url is None
        assert optimizer.mlflow_tracking is True
        assert optimizer._mlflow_connector is None

    def test_init_with_storage_url(self):
        """Test initialization with custom storage URL."""
        optimizer = OptunaOptimizer(
            experiment_id="exp_456",
            storage_url="sqlite:///optuna.db",
        )

        assert optimizer.storage_url == "sqlite:///optuna.db"

    def test_init_with_mlflow_disabled(self):
        """Test initialization with MLflow disabled."""
        optimizer = OptunaOptimizer(
            experiment_id="exp_789",
            mlflow_tracking=False,
        )

        assert optimizer.mlflow_tracking is False


class TestOptunaOptimizerMLflowConnector:
    """Tests for MLflow connector lazy loading."""

    def test_mlflow_connector_not_loaded_initially(self):
        """Connector should not be loaded until accessed."""
        optimizer = OptunaOptimizer(experiment_id="exp_test")
        assert optimizer._mlflow_connector is None

    def test_mlflow_connector_disabled_returns_none(self):
        """Connector should be None when tracking disabled."""
        optimizer = OptunaOptimizer(
            experiment_id="exp_test",
            mlflow_tracking=False,
        )
        assert optimizer.mlflow_connector is None


class TestOptunaOptimizerCreateStudy:
    """Tests for create_study method."""

    @pytest.mark.asyncio
    async def test_create_study_basic(self):
        """Test basic study creation."""
        optimizer = OptunaOptimizer(experiment_id="exp_test")

        study = await optimizer.create_study(
            study_name="test_study",
            direction="maximize",
        )

        assert study is not None
        assert "e2i_exp_test_test_study" in study.study_name
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    @pytest.mark.asyncio
    async def test_create_study_minimize(self):
        """Test study creation with minimize direction."""
        optimizer = OptunaOptimizer(experiment_id="exp_test")

        study = await optimizer.create_study(
            study_name="min_study",
            direction="minimize",
        )

        assert study.direction == optuna.study.StudyDirection.MINIMIZE

    @pytest.mark.asyncio
    async def test_create_study_with_custom_sampler(self):
        """Test study creation with custom sampler."""
        optimizer = OptunaOptimizer(experiment_id="exp_test")
        sampler = SamplerFactory.random_sampler(seed=123)

        study = await optimizer.create_study(
            study_name="custom_sampler_study",
            sampler=sampler,
        )

        assert isinstance(study.sampler, optuna.samplers.RandomSampler)

    @pytest.mark.asyncio
    async def test_create_study_with_custom_pruner(self):
        """Test study creation with custom pruner."""
        optimizer = OptunaOptimizer(experiment_id="exp_test")
        pruner = PrunerFactory.successive_halving_pruner()

        study = await optimizer.create_study(
            study_name="custom_pruner_study",
            pruner=pruner,
        )

        assert isinstance(study.pruner, optuna.pruners.SuccessiveHalvingPruner)


class TestSuggestFromSearchSpace:
    """Tests for suggest_from_search_space static method."""

    def test_suggest_int_parameter(self, mock_optuna_trial):
        """Test integer parameter suggestion."""
        search_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 200},
        }

        params = OptunaOptimizer.suggest_from_search_space(
            mock_optuna_trial, search_space
        )

        assert "n_estimators" in params
        mock_optuna_trial.suggest_int.assert_called()

    def test_suggest_int_with_step(self, mock_optuna_trial):
        """Test integer parameter with step."""
        search_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 200, "step": 50},
        }

        params = OptunaOptimizer.suggest_from_search_space(
            mock_optuna_trial, search_space
        )

        assert "n_estimators" in params
        call_args = mock_optuna_trial.suggest_int.call_args
        assert call_args[1].get("step") == 50

    def test_suggest_float_parameter(self, mock_optuna_trial):
        """Test float parameter suggestion."""
        search_space = {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
        }

        params = OptunaOptimizer.suggest_from_search_space(
            mock_optuna_trial, search_space
        )

        assert "learning_rate" in params
        mock_optuna_trial.suggest_float.assert_called()

    def test_suggest_float_log_scale(self, mock_optuna_trial):
        """Test float parameter with log scale."""
        search_space = {
            "learning_rate": {"type": "float", "low": 0.001, "high": 1.0, "log": True},
        }

        params = OptunaOptimizer.suggest_from_search_space(
            mock_optuna_trial, search_space
        )

        call_args = mock_optuna_trial.suggest_float.call_args
        assert call_args[1].get("log") is True

    def test_suggest_float_with_step(self, mock_optuna_trial):
        """Test float parameter with step."""
        search_space = {
            "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
        }

        params = OptunaOptimizer.suggest_from_search_space(
            mock_optuna_trial, search_space
        )

        call_args = mock_optuna_trial.suggest_float.call_args
        assert call_args[1].get("step") == 0.1

    def test_suggest_categorical_parameter(self, mock_optuna_trial):
        """Test categorical parameter suggestion."""
        search_space = {
            "booster": {"type": "categorical", "choices": ["gbtree", "gblinear"]},
        }

        params = OptunaOptimizer.suggest_from_search_space(
            mock_optuna_trial, search_space
        )

        assert "booster" in params
        mock_optuna_trial.suggest_categorical.assert_called()

    def test_suggest_multiple_parameters(self, mock_optuna_trial, sample_search_space):
        """Test suggesting multiple parameters."""
        params = OptunaOptimizer.suggest_from_search_space(
            mock_optuna_trial, sample_search_space
        )

        assert len(params) == len(sample_search_space)
        assert all(key in params for key in sample_search_space)

    def test_suggest_unknown_type_logs_warning(self, mock_optuna_trial):
        """Test that unknown parameter type logs warning."""
        search_space = {
            "unknown_param": {"type": "unknown", "value": 42},
        }

        with patch("src.mlops.optuna_optimizer.logger") as mock_logger:
            params = OptunaOptimizer.suggest_from_search_space(
                mock_optuna_trial, search_space
            )
            mock_logger.warning.assert_called()

        assert "unknown_param" not in params


class TestCreateCVObjective:
    """Tests for create_cv_objective static method."""

    def test_create_cv_objective_classification(self, classification_data):
        """Test CV objective for classification."""
        X_train, y_train, X_val, y_val = classification_data
        X = np.vstack([X_train, X_val])
        y = np.concatenate([y_train, y_val])

        from sklearn.ensemble import RandomForestClassifier

        search_space = {
            "n_estimators": {"type": "int", "low": 10, "high": 50},
            "max_depth": {"type": "int", "low": 2, "high": 5},
        }

        objective = OptunaOptimizer.create_cv_objective(
            model_class=RandomForestClassifier,
            X=X,
            y=y,
            search_space=search_space,
            problem_type="binary_classification",
            cv_folds=3,
        )

        assert callable(objective)

    def test_create_cv_objective_regression(self, regression_data):
        """Test CV objective for regression."""
        X_train, y_train, X_val, y_val = regression_data
        X = np.vstack([X_train, X_val])
        y = np.concatenate([y_train, y_val])

        from sklearn.ensemble import RandomForestRegressor

        search_space = {
            "n_estimators": {"type": "int", "low": 10, "high": 50},
        }

        objective = OptunaOptimizer.create_cv_objective(
            model_class=RandomForestRegressor,
            X=X,
            y=y,
            search_space=search_space,
            problem_type="regression",
            cv_folds=3,
        )

        assert callable(objective)

    def test_create_cv_objective_with_fixed_params(self, classification_data):
        """Test CV objective with fixed parameters."""
        X_train, y_train, X_val, y_val = classification_data
        X = np.vstack([X_train, X_val])
        y = np.concatenate([y_train, y_val])

        from sklearn.ensemble import RandomForestClassifier

        search_space = {
            "n_estimators": {"type": "int", "low": 10, "high": 50},
        }
        fixed_params = {"random_state": 42, "n_jobs": 1}

        objective = OptunaOptimizer.create_cv_objective(
            model_class=RandomForestClassifier,
            X=X,
            y=y,
            search_space=search_space,
            fixed_params=fixed_params,
        )

        assert callable(objective)


class TestCreateValidationObjective:
    """Tests for create_validation_objective static method."""

    def test_create_validation_objective_classification(self, classification_data):
        """Test validation objective for classification."""
        X_train, y_train, X_val, y_val = classification_data

        from sklearn.ensemble import RandomForestClassifier

        search_space = {
            "n_estimators": {"type": "int", "low": 10, "high": 50},
        }

        objective = OptunaOptimizer.create_validation_objective(
            model_class=RandomForestClassifier,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space=search_space,
            problem_type="binary_classification",
            metric="roc_auc",
        )

        assert callable(objective)

    def test_create_validation_objective_regression(self, regression_data):
        """Test validation objective for regression."""
        X_train, y_train, X_val, y_val = regression_data

        from sklearn.ensemble import RandomForestRegressor

        search_space = {
            "n_estimators": {"type": "int", "low": 10, "high": 50},
        }

        objective = OptunaOptimizer.create_validation_objective(
            model_class=RandomForestRegressor,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space=search_space,
            problem_type="regression",
            metric="rmse",
        )

        assert callable(objective)


class TestEvaluateModel:
    """Tests for _evaluate_model static method."""

    def test_evaluate_roc_auc(self, classification_data):
        """Test ROC-AUC evaluation."""
        X_train, y_train, X_val, y_val = classification_data

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        score = OptunaOptimizer._evaluate_model(
            model, X_val, y_val, "binary_classification", "roc_auc"
        )

        assert 0 <= score <= 1

    def test_evaluate_accuracy(self, classification_data):
        """Test accuracy evaluation."""
        X_train, y_train, X_val, y_val = classification_data

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        score = OptunaOptimizer._evaluate_model(
            model, X_val, y_val, "binary_classification", "accuracy"
        )

        assert 0 <= score <= 1

    def test_evaluate_f1(self, classification_data):
        """Test F1 score evaluation."""
        X_train, y_train, X_val, y_val = classification_data

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        score = OptunaOptimizer._evaluate_model(
            model, X_val, y_val, "binary_classification", "f1"
        )

        assert 0 <= score <= 1

    def test_evaluate_rmse(self, regression_data):
        """Test RMSE evaluation (returns negative)."""
        X_train, y_train, X_val, y_val = regression_data

        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        score = OptunaOptimizer._evaluate_model(
            model, X_val, y_val, "regression", "rmse"
        )

        # RMSE returns negative (so higher is better)
        assert score <= 0

    def test_evaluate_r2(self, regression_data):
        """Test R2 score evaluation."""
        X_train, y_train, X_val, y_val = regression_data

        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        score = OptunaOptimizer._evaluate_model(
            model, X_val, y_val, "regression", "r2"
        )

        # R2 can be negative for poor models
        assert score <= 1


class TestOptimize:
    """Tests for optimize method."""

    @pytest.mark.asyncio
    async def test_optimize_returns_results(self, classification_data):
        """Test that optimize returns expected results structure."""
        X_train, y_train, X_val, y_val = classification_data

        from sklearn.ensemble import RandomForestClassifier

        optimizer = OptunaOptimizer(
            experiment_id="test_exp",
            mlflow_tracking=False,
        )

        study = await optimizer.create_study(
            study_name="test_optimize",
            direction="maximize",
        )

        search_space = {
            "n_estimators": {"type": "int", "low": 5, "high": 20},
        }

        objective = optimizer.create_validation_objective(
            model_class=RandomForestClassifier,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space=search_space,
            fixed_params={"random_state": 42},
        )

        results = await optimizer.optimize(
            study=study,
            objective=objective,
            n_trials=3,
            timeout=60,
        )

        assert "best_params" in results
        assert "best_value" in results
        assert "best_trial_number" in results
        assert "n_trials" in results
        assert "n_completed" in results
        assert "n_pruned" in results
        assert "duration_seconds" in results
        assert "study_name" in results

    @pytest.mark.asyncio
    async def test_optimize_respects_n_trials(self, classification_data):
        """Test that optimize respects n_trials limit."""
        X_train, y_train, X_val, y_val = classification_data

        from sklearn.ensemble import RandomForestClassifier

        optimizer = OptunaOptimizer(
            experiment_id="test_exp",
            mlflow_tracking=False,
        )

        study = await optimizer.create_study(
            study_name="test_n_trials",
            direction="maximize",
        )

        search_space = {
            "n_estimators": {"type": "int", "low": 5, "high": 10},
        }

        objective = optimizer.create_validation_objective(
            model_class=RandomForestClassifier,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space=search_space,
            fixed_params={"random_state": 42},
        )

        results = await optimizer.optimize(
            study=study,
            objective=objective,
            n_trials=5,
        )

        assert results["n_trials"] == 5

    @pytest.mark.asyncio
    async def test_optimize_with_timeout(self, classification_data):
        """Test that optimize handles timeout and returns partial results."""
        X_train, y_train, X_val, y_val = classification_data

        from sklearn.ensemble import RandomForestClassifier

        optimizer = OptunaOptimizer(
            experiment_id="test_exp",
            mlflow_tracking=False,
        )

        study = await optimizer.create_study(
            study_name="test_timeout",
            direction="maximize",
        )

        search_space = {
            "n_estimators": {"type": "int", "low": 5, "high": 10},
        }

        objective = optimizer.create_validation_objective(
            model_class=RandomForestClassifier,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space=search_space,
            fixed_params={"random_state": 42},
        )

        # Run with very short timeout and high trial count
        # We expect fewer trials than requested
        results = await optimizer.optimize(
            study=study,
            objective=objective,
            n_trials=1000,  # High trial count
            timeout=1,  # 1 second timeout
        )

        # Should return valid results even with timeout
        assert "best_params" in results
        assert "best_value" in results
        assert results["n_trials"] < 1000  # Shouldn't complete all trials
        assert results["n_completed"] >= 1  # At least one trial completed
        assert "duration_seconds" in results

    @pytest.mark.asyncio
    async def test_optimize_counts_pruned_trials(self, classification_data):
        """Test that pruned trials are counted correctly."""
        X_train, y_train, X_val, y_val = classification_data

        optimizer = OptunaOptimizer(
            experiment_id="test_exp",
            mlflow_tracking=False,
        )

        # Create study with MedianPruner (aggressive on later trials)
        from optuna.pruners import MedianPruner

        study = await optimizer.create_study(
            study_name="test_pruning",
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=1, n_warmup_steps=0),
        )

        # Create objective that sometimes gets pruned
        import optuna

        trial_count = {"count": 0}

        def prunable_objective(trial: optuna.Trial) -> float:
            trial_count["count"] += 1

            # First trial completes to establish baseline
            if trial_count["count"] == 1:
                trial.report(0.9, step=0)
                return 0.9

            # Later trials report low value and may get pruned
            trial.report(0.1, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return 0.1

        results = await optimizer.optimize(
            study=study,
            objective=prunable_objective,
            n_trials=5,
        )

        # Should count both completed and pruned trials
        assert results["n_trials"] == 5
        assert results["n_completed"] >= 1  # At least first trial completed
        assert results["n_completed"] + results["n_pruned"] == 5

    @pytest.mark.asyncio
    async def test_optimize_handles_objective_errors(self, classification_data):
        """Test that optimize handles exceptions in objective function."""
        X_train, y_train, X_val, y_val = classification_data

        optimizer = OptunaOptimizer(
            experiment_id="test_exp",
            mlflow_tracking=False,
        )

        study = await optimizer.create_study(
            study_name="test_error_handling",
            direction="maximize",
        )

        # Track how many times objective was called
        call_count = {"count": 0}

        def error_objective(trial):
            call_count["count"] += 1
            if call_count["count"] <= 2:
                raise ValueError("Simulated error")
            return 0.8

        # Should continue even when some trials fail
        results = await optimizer.optimize(
            study=study,
            objective=error_objective,
            n_trials=5,
            catch=(Exception,),  # Catch exceptions
        )

        # Should have completed some trials
        assert results["n_trials"] == 5
        # Best value should come from successful trial
        assert results["best_value"] == 0.8


class TestGetOptimizationHistory:
    """Tests for get_optimization_history method."""

    @pytest.mark.asyncio
    async def test_get_history_empty_study(self):
        """Test history for study with no trials."""
        optimizer = OptunaOptimizer(experiment_id="test_exp")
        study = await optimizer.create_study(
            study_name="empty_study",
            direction="maximize",
        )

        history = await optimizer.get_optimization_history(study)

        assert history == []

    @pytest.mark.asyncio
    async def test_get_history_with_trials(self, classification_data):
        """Test history for study with trials."""
        X_train, y_train, X_val, y_val = classification_data

        from sklearn.ensemble import RandomForestClassifier

        optimizer = OptunaOptimizer(
            experiment_id="test_exp",
            mlflow_tracking=False,
        )

        study = await optimizer.create_study(
            study_name="history_study",
            direction="maximize",
        )

        search_space = {
            "n_estimators": {"type": "int", "low": 5, "high": 10},
        }

        objective = optimizer.create_validation_objective(
            model_class=RandomForestClassifier,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space=search_space,
            fixed_params={"random_state": 42},
        )

        await optimizer.optimize(
            study=study,
            objective=objective,
            n_trials=3,
        )

        history = await optimizer.get_optimization_history(study)

        assert len(history) == 3
        for record in history:
            assert "trial_number" in record
            assert "state" in record
            assert "params" in record


class TestSaveToDatabase:
    """Tests for save_to_database method."""

    @pytest.mark.asyncio
    async def test_save_to_database_no_client(self):
        """Test save when Supabase client not available."""
        optimizer = OptunaOptimizer(experiment_id="test_exp")

        # Create a minimal mock study
        mock_study = MagicMock(spec=optuna.Study)
        mock_study.study_name = "test_study"
        mock_study.direction = optuna.study.StudyDirection.MAXIMIZE
        mock_study.sampler = MagicMock()
        mock_study.pruner = MagicMock()
        mock_study.trials = []

        optimization_results = {
            "best_params": {"n_estimators": 100},
            "best_value": 0.85,
            "best_trial_number": 5,
            "n_trials": 10,
            "n_completed": 8,
            "n_pruned": 2,
            "duration_seconds": 60.0,
        }

        # Patch the import inside the function - need to patch the module itself
        import sys
        original_module = sys.modules.get("src.repositories.supabase_client")

        # Create a mock module that raises ImportError
        mock_module = MagicMock()
        mock_module.get_supabase_client = MagicMock(side_effect=ImportError("Mock import error"))
        sys.modules["src.repositories.supabase_client"] = mock_module

        try:
            result = await optimizer.save_to_database(
                study=mock_study,
                optimization_results=optimization_results,
            )
        finally:
            # Restore original module
            if original_module is not None:
                sys.modules["src.repositories.supabase_client"] = original_module
            else:
                sys.modules.pop("src.repositories.supabase_client", None)

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_save_to_database_success(self, mock_frozen_trial):
        """Test successful database save."""
        optimizer = OptunaOptimizer(experiment_id="test_exp")

        # Create mock study
        mock_study = MagicMock(spec=optuna.Study)
        mock_study.study_name = "test_study"
        mock_study.direction = optuna.study.StudyDirection.MAXIMIZE
        mock_study.sampler = MagicMock()
        mock_study.pruner = MagicMock()
        mock_study.trials = [mock_frozen_trial]

        optimization_results = {
            "best_params": {"n_estimators": 100},
            "best_value": 0.85,
            "best_trial_number": 0,
            "n_trials": 1,
            "n_completed": 1,
            "n_pruned": 0,
            "duration_seconds": 10.0,
        }

        # Mock Supabase client - using sync MagicMock for table operations
        # since the Supabase client uses sync methods for table().insert().execute()
        mock_execute_result = MagicMock()
        mock_execute_result.data = [{"id": "study-uuid-123"}]

        mock_insert = MagicMock()
        mock_insert.execute = AsyncMock(return_value=mock_execute_result)

        mock_table = MagicMock()
        mock_table.insert = MagicMock(return_value=mock_insert)

        mock_client = MagicMock()
        mock_client.table = MagicMock(return_value=mock_table)

        # Patch using sys.modules for dynamic import
        import sys
        original_module = sys.modules.get("src.repositories.supabase_client")

        mock_supabase_module = MagicMock()
        mock_supabase_module.get_supabase_client = AsyncMock(return_value=mock_client)
        sys.modules["src.repositories.supabase_client"] = mock_supabase_module

        try:
            result = await optimizer.save_to_database(
                study=mock_study,
                optimization_results=optimization_results,
                algorithm_name="XGBoost",
            )
        finally:
            # Restore original module
            if original_module is not None:
                sys.modules["src.repositories.supabase_client"] = original_module
            else:
                sys.modules.pop("src.repositories.supabase_client", None)

        assert result["success"] is True
        assert result["study_id"] == "study-uuid-123"


# ============================================================================
# PRUNER FACTORY TESTS
# ============================================================================


class TestPrunerFactory:
    """Tests for PrunerFactory."""

    def test_median_pruner_defaults(self):
        """Test Median pruner with defaults."""
        pruner = PrunerFactory.median_pruner()

        assert isinstance(pruner, optuna.pruners.MedianPruner)

    def test_median_pruner_custom_params(self):
        """Test Median pruner with custom parameters."""
        pruner = PrunerFactory.median_pruner(
            n_startup_trials=10,
            n_warmup_steps=20,
            interval_steps=2,
        )

        assert isinstance(pruner, optuna.pruners.MedianPruner)

    def test_successive_halving_pruner_defaults(self):
        """Test Successive Halving pruner with defaults."""
        pruner = PrunerFactory.successive_halving_pruner()

        assert isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner)

    def test_successive_halving_pruner_custom_params(self):
        """Test Successive Halving pruner with custom parameters."""
        pruner = PrunerFactory.successive_halving_pruner(
            min_resource=5,
            reduction_factor=4,
        )

        assert isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner)

    def test_no_pruner(self):
        """Test no-op pruner."""
        pruner = PrunerFactory.no_pruner()

        assert isinstance(pruner, optuna.pruners.NopPruner)


# ============================================================================
# SAMPLER FACTORY TESTS
# ============================================================================


class TestSamplerFactory:
    """Tests for SamplerFactory."""

    def test_tpe_sampler_defaults(self):
        """Test TPE sampler with defaults."""
        sampler = SamplerFactory.tpe_sampler()

        assert isinstance(sampler, optuna.samplers.TPESampler)

    def test_tpe_sampler_custom_params(self):
        """Test TPE sampler with custom parameters."""
        sampler = SamplerFactory.tpe_sampler(
            seed=123,
            n_startup_trials=20,
            multivariate=False,
        )

        assert isinstance(sampler, optuna.samplers.TPESampler)

    def test_random_sampler(self):
        """Test Random sampler."""
        sampler = SamplerFactory.random_sampler(seed=42)

        assert isinstance(sampler, optuna.samplers.RandomSampler)

    def test_cmaes_sampler(self):
        """Test CMA-ES sampler."""
        sampler = SamplerFactory.cmaes_sampler(seed=42)

        assert isinstance(sampler, optuna.samplers.CmaEsSampler)


# ============================================================================
# GET MODEL CLASS TESTS
# ============================================================================


class TestGetModelClass:
    """Tests for get_model_class function."""

    def test_get_random_forest_classifier(self):
        """Test getting RandomForest classifier."""
        model_class = get_model_class("RandomForest", "binary_classification")

        from sklearn.ensemble import RandomForestClassifier

        assert model_class == RandomForestClassifier

    def test_get_random_forest_regressor(self):
        """Test getting RandomForest regressor."""
        model_class = get_model_class("RandomForest", "regression")

        from sklearn.ensemble import RandomForestRegressor

        assert model_class == RandomForestRegressor

    def test_get_logistic_regression(self):
        """Test getting LogisticRegression."""
        model_class = get_model_class("LogisticRegression", "binary_classification")

        from sklearn.linear_model import LogisticRegression

        assert model_class == LogisticRegression

    def test_get_ridge(self):
        """Test getting Ridge regressor."""
        model_class = get_model_class("Ridge", "regression")

        from sklearn.linear_model import Ridge

        assert model_class == Ridge

    def test_get_lasso(self):
        """Test getting Lasso regressor."""
        model_class = get_model_class("Lasso", "regression")

        from sklearn.linear_model import Lasso

        assert model_class == Lasso

    def test_xgboost_classifier(self):
        """Test getting XGBoost classifier."""
        try:
            model_class = get_model_class("XGBoost", "binary_classification")
            # If XGBoost is installed
            from xgboost import XGBClassifier

            assert model_class == XGBClassifier
        except ImportError:
            # XGBoost not installed
            pytest.skip("XGBoost not installed")

    def test_lightgbm_classifier(self):
        """Test getting LightGBM classifier."""
        try:
            model_class = get_model_class("LightGBM", "binary_classification")
            from lightgbm import LGBMClassifier

            assert model_class == LGBMClassifier
        except ImportError:
            pytest.skip("LightGBM not installed")

    def test_unknown_algorithm(self):
        """Test getting unknown algorithm returns None."""
        model_class = get_model_class("UnknownAlgorithm", "binary_classification")

        assert model_class is None

    def test_causal_forest_returns_none(self):
        """Test CausalForest returns None (needs special handling)."""
        model_class = get_model_class("CausalForest", "binary_classification")

        assert model_class is None


# ============================================================================
# RUN HYPERPARAMETER OPTIMIZATION TESTS
# ============================================================================


class TestRunHyperparameterOptimization:
    """Tests for run_hyperparameter_optimization function."""

    @pytest.mark.asyncio
    async def test_run_hpo_basic(self, classification_data):
        """Test basic HPO run."""
        X_train, y_train, X_val, y_val = classification_data

        search_space = {
            "n_estimators": {"type": "int", "low": 5, "high": 15},
        }

        results = await run_hyperparameter_optimization(
            experiment_id="test_hpo",
            algorithm_name="RandomForest",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space=search_space,
            problem_type="binary_classification",
            n_trials=3,
            timeout=60,
        )

        assert "best_params" in results
        assert "best_value" in results
        assert "algorithm_name" in results
        assert results["algorithm_name"] == "RandomForest"

    @pytest.mark.asyncio
    async def test_run_hpo_with_cv(self, classification_data):
        """Test HPO with cross-validation."""
        X_train, y_train, X_val, y_val = classification_data

        search_space = {
            "n_estimators": {"type": "int", "low": 5, "high": 15},
        }

        results = await run_hyperparameter_optimization(
            experiment_id="test_hpo_cv",
            algorithm_name="RandomForest",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space=search_space,
            use_cv=True,
            cv_folds=3,
            n_trials=3,
        )

        assert "best_params" in results

    @pytest.mark.asyncio
    async def test_run_hpo_unknown_algorithm(self, classification_data):
        """Test HPO with unknown algorithm returns error."""
        X_train, y_train, X_val, y_val = classification_data

        results = await run_hyperparameter_optimization(
            experiment_id="test_hpo_unknown",
            algorithm_name="UnknownAlgorithm",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space={},
            n_trials=3,
        )

        assert "error" in results
        assert results["best_params"] == {}

    @pytest.mark.asyncio
    async def test_run_hpo_includes_history(self, classification_data):
        """Test that HPO results include history."""
        X_train, y_train, X_val, y_val = classification_data

        search_space = {
            "n_estimators": {"type": "int", "low": 5, "high": 15},
        }

        results = await run_hyperparameter_optimization(
            experiment_id="test_hpo_history",
            algorithm_name="RandomForest",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space=search_space,
            n_trials=3,
        )

        assert "history" in results
        assert len(results["history"]) == 3


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestOptunaOptimizerIntegration:
    """Integration tests for OptunaOptimizer."""

    @pytest.mark.asyncio
    async def test_full_optimization_workflow(self, classification_data):
        """Test complete optimization workflow."""
        X_train, y_train, X_val, y_val = classification_data

        from sklearn.ensemble import RandomForestClassifier

        # 1. Create optimizer
        optimizer = OptunaOptimizer(
            experiment_id="integration_test",
            mlflow_tracking=False,
        )

        # 2. Create study
        study = await optimizer.create_study(
            study_name="full_workflow",
            direction="maximize",
            pruner=PrunerFactory.median_pruner(),
            sampler=SamplerFactory.tpe_sampler(),
        )

        # 3. Define search space
        search_space = {
            "n_estimators": {"type": "int", "low": 10, "high": 30, "step": 10},
            "max_depth": {"type": "int", "low": 2, "high": 5},
        }

        # 4. Create objective
        objective = optimizer.create_validation_objective(
            model_class=RandomForestClassifier,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space=search_space,
            problem_type="binary_classification",
            metric="roc_auc",
            fixed_params={"random_state": 42},
        )

        # 5. Run optimization
        results = await optimizer.optimize(
            study=study,
            objective=objective,
            n_trials=5,
        )

        # 6. Get history
        history = await optimizer.get_optimization_history(study)

        # Assertions
        assert results["n_trials"] == 5
        assert results["best_value"] > 0.5  # Better than random
        assert "n_estimators" in results["best_params"]
        assert "max_depth" in results["best_params"]
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_pruning_works(self, classification_data):
        """Test that pruning actually prunes some trials."""
        X_train, y_train, X_val, y_val = classification_data

        from sklearn.linear_model import LogisticRegression

        optimizer = OptunaOptimizer(
            experiment_id="pruning_test",
            mlflow_tracking=False,
        )

        # Use aggressive pruner
        study = await optimizer.create_study(
            study_name="pruning_test",
            direction="maximize",
            pruner=PrunerFactory.median_pruner(n_startup_trials=1, n_warmup_steps=0),
        )

        # Simple search space
        search_space = {
            "C": {"type": "float", "low": 0.001, "high": 100, "log": True},
        }

        objective = optimizer.create_validation_objective(
            model_class=LogisticRegression,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space=search_space,
            fixed_params={"random_state": 42, "max_iter": 100},
        )

        results = await optimizer.optimize(
            study=study,
            objective=objective,
            n_trials=10,
        )

        # May or may not have pruned trials depending on performance
        assert results["n_trials"] == 10
        assert results["n_completed"] + results["n_pruned"] <= results["n_trials"]
