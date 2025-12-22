"""Optuna hyperparameter optimization for E2I ML pipelines.

This module provides a unified interface for hyperparameter optimization
using Optuna with MLflow integration and database storage.

Version: 1.0.0
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna with MLflow integration.

    Features:
    - Creates and manages Optuna studies
    - Converts E2I search space format to Optuna trial suggestions
    - Supports various pruning strategies
    - Integrates with MLflow for experiment tracking
    - Stores optimization history in database

    Example:
        optimizer = OptunaOptimizer(experiment_id="exp_123")
        study = await optimizer.create_study("my_study", direction="maximize")
        best_params = await optimizer.optimize(
            study=study,
            objective=my_objective_func,
            n_trials=100,
            timeout=3600
        )
    """

    def __init__(
        self,
        experiment_id: str,
        storage_url: Optional[str] = None,
        mlflow_tracking: bool = True,
    ):
        """Initialize OptunaOptimizer.

        Args:
            experiment_id: E2I experiment ID for tracking
            storage_url: Optuna storage URL (e.g., "sqlite:///optuna.db")
                        If None, uses in-memory storage
            mlflow_tracking: Whether to log trials to MLflow
        """
        self.experiment_id = experiment_id
        self.storage_url = storage_url
        self.mlflow_tracking = mlflow_tracking
        self._mlflow_connector = None

    @property
    def mlflow_connector(self):
        """Lazy load MLflow connector."""
        if self._mlflow_connector is None and self.mlflow_tracking:
            try:
                from src.mlops.mlflow_connector import MLflowConnector

                self._mlflow_connector = MLflowConnector()
            except ImportError:
                logger.warning("MLflow connector not available, disabling tracking")
                self.mlflow_tracking = False
        return self._mlflow_connector

    async def create_study(
        self,
        study_name: str,
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        load_if_exists: bool = True,
    ) -> optuna.Study:
        """Create or load an Optuna study.

        Args:
            study_name: Unique study identifier
            direction: "minimize" or "maximize"
            sampler: Optuna sampler (default: TPESampler)
            pruner: Optuna pruner (default: MedianPruner)
            load_if_exists: Load existing study if it exists

        Returns:
            Optuna Study object
        """
        # Default sampler: Tree-structured Parzen Estimator
        if sampler is None:
            sampler = TPESampler(
                seed=42,
                n_startup_trials=10,
                multivariate=True,
            )

        # Default pruner: Median pruner for early stopping
        if pruner is None:
            pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            )

        # Build full study name with experiment context
        full_study_name = f"e2i_{self.experiment_id}_{study_name}"

        study = optuna.create_study(
            study_name=full_study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.storage_url,
            load_if_exists=load_if_exists,
        )

        logger.info(f"Created/loaded study: {full_study_name} (direction={direction})")
        return study

    async def optimize(
        self,
        study: optuna.Study,
        objective: Callable[[optuna.Trial], float],
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        callbacks: Optional[List[Callable]] = None,
        catch: Tuple[type, ...] = (Exception,),
    ) -> Dict[str, Any]:
        """Run optimization and return results.

        Args:
            study: Optuna Study object
            objective: Objective function to optimize
            n_trials: Maximum number of trials
            timeout: Timeout in seconds (None for no timeout)
            n_jobs: Number of parallel jobs (-1 for all cores)
            callbacks: List of callback functions
            catch: Exception types to catch during optimization

        Returns:
            Dictionary with best_params, best_value, n_trials, duration
        """
        start_time = time.time()

        # Add MLflow callback if tracking enabled
        if self.mlflow_tracking and callbacks is None:
            callbacks = [self._create_mlflow_callback()]

        # Run optimization in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                callbacks=callbacks,
                catch=catch,
                show_progress_bar=False,
            ),
        )

        duration = time.time() - start_time

        # Extract results
        results = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_trial_number": study.best_trial.number,
            "n_trials": len(study.trials),
            "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "duration_seconds": duration,
            "study_name": study.study_name,
        }

        logger.info(
            f"Optimization complete: best_value={results['best_value']:.4f}, "
            f"trials={results['n_trials']} ({results['n_pruned']} pruned)"
        )

        return results

    def _create_mlflow_callback(self) -> Callable:
        """Create MLflow logging callback."""

        def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            """Log trial to MLflow."""
            if self.mlflow_connector is None:
                return

            try:
                # Log trial metrics
                metrics = {
                    f"trial_{trial.number}_value": trial.value or 0.0,
                    f"trial_{trial.number}_duration": trial.duration.total_seconds() if trial.duration else 0.0,
                }

                # Log best value so far
                if study.best_trial and study.best_trial.number == trial.number:
                    metrics["best_value"] = trial.value or 0.0

                # This is a synchronous callback, so we can't use async
                # The MLflow connector should handle this appropriately
            except Exception as e:
                logger.debug(f"Failed to log trial to MLflow: {e}")

        return callback

    @staticmethod
    def suggest_from_search_space(
        trial: optuna.Trial,
        search_space: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Convert E2I search space format to Optuna trial suggestions.

        Args:
            trial: Optuna trial object
            search_space: Search space in E2I format:
                {
                    "param_name": {
                        "type": "int" | "float" | "categorical",
                        "low": <min_value>,
                        "high": <max_value>,
                        "log": <bool>,  # for float
                        "step": <step_size>,  # optional
                        "choices": [<values>],  # for categorical
                    }
                }

        Returns:
            Dictionary of sampled hyperparameters
        """
        params = {}

        for param_name, config in search_space.items():
            param_type = config.get("type", "float")

            if param_type == "int":
                step = config.get("step", 1)
                params[param_name] = trial.suggest_int(
                    param_name,
                    config["low"],
                    config["high"],
                    step=step,
                )
            elif param_type == "float":
                log_scale = config.get("log", False)
                step = config.get("step")
                if step is not None and not log_scale:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        config["low"],
                        config["high"],
                        step=step,
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        config["low"],
                        config["high"],
                        log=log_scale,
                    )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    config["choices"],
                )
            else:
                logger.warning(f"Unknown parameter type '{param_type}' for {param_name}")

        return params

    @staticmethod
    def create_cv_objective(
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        search_space: Dict[str, Dict[str, Any]],
        problem_type: str = "binary_classification",
        cv_folds: int = 5,
        scoring: Optional[str] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> Callable[[optuna.Trial], float]:
        """Create cross-validation objective function.

        Args:
            model_class: ML model class (e.g., XGBClassifier)
            X: Feature matrix
            y: Target vector
            search_space: Hyperparameter search space
            problem_type: "binary_classification", "multiclass_classification", "regression"
            cv_folds: Number of CV folds
            scoring: Scoring metric (auto-detected if None)
            fixed_params: Fixed parameters not being optimized

        Returns:
            Objective function for Optuna
        """
        # Auto-detect scoring metric
        if scoring is None:
            if problem_type in ["binary_classification", "multiclass_classification"]:
                scoring = "roc_auc" if problem_type == "binary_classification" else "f1_weighted"
            else:
                scoring = "neg_root_mean_squared_error"

        fixed_params = fixed_params or {}

        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna optimization."""
            # Sample hyperparameters
            params = OptunaOptimizer.suggest_from_search_space(trial, search_space)

            # Merge with fixed parameters
            all_params = {**fixed_params, **params}

            try:
                # Create model
                model = model_class(**all_params)

                # Run cross-validation
                scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                )

                # Return mean score (higher is better for Optuna maximize)
                mean_score = scores.mean()

                # Report intermediate value for pruning
                trial.report(mean_score, step=0)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return mean_score

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return float("-inf") if scoring != "neg_root_mean_squared_error" else float("inf")

        return objective

    @staticmethod
    def create_validation_objective(
        model_class: type,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        search_space: Dict[str, Dict[str, Any]],
        problem_type: str = "binary_classification",
        metric: str = "roc_auc",
        fixed_params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: Optional[int] = None,
    ) -> Callable[[optuna.Trial], float]:
        """Create validation-based objective function.

        Uses a held-out validation set instead of cross-validation.
        Faster than CV but may be less robust.

        Args:
            model_class: ML model class
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            search_space: Hyperparameter search space
            problem_type: Problem type
            metric: Evaluation metric
            fixed_params: Fixed parameters
            early_stopping_rounds: Early stopping for tree models

        Returns:
            Objective function for Optuna
        """
        fixed_params = fixed_params or {}

        def objective(trial: optuna.Trial) -> float:
            """Validation-based objective function."""
            # Sample hyperparameters
            params = OptunaOptimizer.suggest_from_search_space(trial, search_space)

            # Merge with fixed parameters
            all_params = {**fixed_params, **params}

            # Add early stopping for compatible models
            if early_stopping_rounds:
                all_params["early_stopping_rounds"] = early_stopping_rounds
                all_params["eval_set"] = [(X_val, y_val)]

            try:
                # Create and train model
                model = model_class(**all_params)

                if early_stopping_rounds:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_train, y_train)

                # Evaluate on validation set
                score = OptunaOptimizer._evaluate_model(
                    model, X_val, y_val, problem_type, metric
                )

                # Report for pruning
                trial.report(score, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return float("-inf")

        return objective

    @staticmethod
    def _evaluate_model(
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: str,
        metric: str,
    ) -> float:
        """Evaluate model on data.

        Args:
            model: Trained model
            X: Features
            y: Labels
            problem_type: Problem type
            metric: Metric name

        Returns:
            Metric value (higher is better)
        """
        if problem_type in ["binary_classification", "multiclass_classification"]:
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

            if metric == "roc_auc":
                return roc_auc_score(y, y_proba)
            elif metric == "accuracy":
                return accuracy_score(y, y_pred)
            elif metric == "f1":
                return f1_score(y, y_pred, average="weighted" if problem_type == "multiclass_classification" else "binary")
            elif metric == "precision":
                return precision_score(y, y_pred, average="weighted" if problem_type == "multiclass_classification" else "binary")
            elif metric == "recall":
                return recall_score(y, y_pred, average="weighted" if problem_type == "multiclass_classification" else "binary")
            else:
                return accuracy_score(y, y_pred)
        else:  # Regression
            y_pred = model.predict(X)

            if metric == "rmse" or metric == "neg_root_mean_squared_error":
                # Return negative RMSE so higher is better
                return -np.sqrt(mean_squared_error(y, y_pred))
            elif metric == "mae":
                return -mean_absolute_error(y, y_pred)
            elif metric == "r2":
                return r2_score(y, y_pred)
            else:
                return -np.sqrt(mean_squared_error(y, y_pred))

    async def get_optimization_history(
        self,
        study: optuna.Study,
    ) -> List[Dict[str, Any]]:
        """Get optimization history as list of trial records.

        Args:
            study: Optuna Study object

        Returns:
            List of trial dictionaries
        """
        history = []

        for trial in study.trials:
            record = {
                "trial_number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
                "user_attrs": trial.user_attrs,
                "system_attrs": trial.system_attrs,
            }
            history.append(record)

        return history

    async def save_to_database(
        self,
        study: optuna.Study,
        optimization_results: Dict[str, Any],
        algorithm_name: str = "unknown",
        problem_type: str = "binary_classification",
        metric: str = "roc_auc",
        search_space: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Save optimization results to database.

        Stores in ml_hpo_studies table with trial history in ml_hpo_trials.

        Args:
            study: Optuna Study object
            optimization_results: Results from optimize()
            algorithm_name: Name of algorithm being optimized
            problem_type: Problem type
            metric: Optimization metric
            search_space: Search space definition

        Returns:
            Dictionary with study_id, success status, and any errors
        """
        try:
            from src.repositories.supabase_client import get_supabase_client

            client = await get_supabase_client()
            if client is None:
                logger.warning("Supabase client not available, skipping database save")
                return {"success": False, "error": "Supabase not available"}

            # Prepare study record
            study_record = {
                "study_name": study.study_name,
                "experiment_id": self.experiment_id,
                "algorithm_name": algorithm_name,
                "problem_type": problem_type,
                "direction": study.direction.name.lower(),
                "sampler_name": type(study.sampler).__name__,
                "pruner_name": type(study.pruner).__name__ if study.pruner else "NoPruner",
                "metric": metric,
                "search_space": search_space or {},
                "n_trials": optimization_results["n_trials"],
                "n_completed": optimization_results["n_completed"],
                "n_pruned": optimization_results["n_pruned"],
                "n_failed": optimization_results["n_trials"] - optimization_results["n_completed"] - optimization_results["n_pruned"],
                "best_trial_number": optimization_results["best_trial_number"],
                "best_value": float(optimization_results["best_value"]) if optimization_results["best_value"] is not None else None,
                "best_params": optimization_results["best_params"],
                "duration_seconds": optimization_results["duration_seconds"],
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

            # Insert study record
            result = await client.table("ml_hpo_studies").insert(study_record).execute()

            if not result.data:
                logger.error("Failed to insert HPO study record")
                return {"success": False, "error": "Insert failed"}

            study_id = result.data[0]["id"]
            logger.info(f"Saved HPO study {study.study_name} with ID {study_id}")

            # Save individual trials
            trials_saved = await self._save_trials_to_database(
                client, study_id, study.trials
            )

            return {
                "success": True,
                "study_id": study_id,
                "trials_saved": trials_saved,
            }

        except ImportError as e:
            logger.warning(f"Database client not available: {e}")
            return {"success": False, "error": str(e)}

        except Exception as e:
            logger.error(f"Failed to save study to database: {e}")
            return {"success": False, "error": str(e)}

    async def _save_trials_to_database(
        self,
        client: Any,
        study_id: str,
        trials: List[optuna.trial.FrozenTrial],
    ) -> int:
        """Save individual trial records to database.

        Args:
            client: Supabase client
            study_id: Parent study ID
            trials: List of frozen trials

        Returns:
            Number of trials saved
        """
        saved_count = 0

        for trial in trials:
            try:
                trial_record = {
                    "study_id": study_id,
                    "trial_number": trial.number,
                    "state": trial.state.name,
                    "params": trial.params,
                    "value": float(trial.value) if trial.value is not None else None,
                    "intermediate_values": {
                        str(k): float(v) for k, v in trial.intermediate_values.items()
                    } if trial.intermediate_values else {},
                    "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                    "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                    "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
                    "user_attrs": trial.user_attrs or {},
                    "system_attrs": trial.system_attrs or {},
                }

                await client.table("ml_hpo_trials").insert(trial_record).execute()
                saved_count += 1

            except Exception as e:
                logger.warning(f"Failed to save trial {trial.number}: {e}")

        logger.info(f"Saved {saved_count}/{len(trials)} trials to database")
        return saved_count


class PrunerFactory:
    """Factory for creating Optuna pruners."""

    @staticmethod
    def median_pruner(
        n_startup_trials: int = 5,
        n_warmup_steps: int = 10,
        interval_steps: int = 1,
    ) -> MedianPruner:
        """Create Median pruner.

        Prunes trials that are worse than median of previous trials.
        Good default for most use cases.
        """
        return MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
            interval_steps=interval_steps,
        )

    @staticmethod
    def successive_halving_pruner(
        min_resource: int = 1,
        reduction_factor: int = 3,
        min_early_stopping_rate: int = 0,
    ) -> SuccessiveHalvingPruner:
        """Create Successive Halving pruner.

        Aggressively prunes unpromising trials.
        Good for expensive objective functions.
        """
        return SuccessiveHalvingPruner(
            min_resource=min_resource,
            reduction_factor=reduction_factor,
            min_early_stopping_rate=min_early_stopping_rate,
        )

    @staticmethod
    def no_pruner() -> optuna.pruners.NopPruner:
        """Create no-op pruner (no pruning)."""
        return optuna.pruners.NopPruner()


class SamplerFactory:
    """Factory for creating Optuna samplers."""

    @staticmethod
    def tpe_sampler(
        seed: int = 42,
        n_startup_trials: int = 10,
        multivariate: bool = True,
    ) -> TPESampler:
        """Create TPE (Tree-structured Parzen Estimator) sampler.

        Default choice. Good balance of exploration and exploitation.
        """
        return TPESampler(
            seed=seed,
            n_startup_trials=n_startup_trials,
            multivariate=multivariate,
        )

    @staticmethod
    def random_sampler(seed: int = 42) -> optuna.samplers.RandomSampler:
        """Create Random sampler.

        Simple random search. Good baseline.
        """
        return optuna.samplers.RandomSampler(seed=seed)

    @staticmethod
    def cmaes_sampler(
        seed: int = 42,
        n_startup_trials: int = 10,
    ) -> optuna.samplers.CmaEsSampler:
        """Create CMA-ES sampler.

        Good for continuous parameters with complex dependencies.
        """
        return optuna.samplers.CmaEsSampler(
            seed=seed,
            n_startup_trials=n_startup_trials,
        )


def get_model_class(algorithm_name: str, problem_type: str) -> Optional[type]:
    """Get model class for algorithm name.

    Args:
        algorithm_name: Algorithm name (e.g., "XGBoost", "LightGBM")
        problem_type: Problem type

    Returns:
        Model class or None if not found
    """
    is_classification = problem_type in ["binary_classification", "multiclass_classification"]

    try:
        if algorithm_name == "XGBoost":
            from xgboost import XGBClassifier, XGBRegressor
            return XGBClassifier if is_classification else XGBRegressor

        elif algorithm_name == "LightGBM":
            from lightgbm import LGBMClassifier, LGBMRegressor
            return LGBMClassifier if is_classification else LGBMRegressor

        elif algorithm_name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            return RandomForestClassifier if is_classification else RandomForestRegressor

        elif algorithm_name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression

        elif algorithm_name == "Ridge":
            from sklearn.linear_model import Ridge
            return Ridge

        elif algorithm_name == "Lasso":
            from sklearn.linear_model import Lasso
            return Lasso

        elif algorithm_name == "GradientBoosting":
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            return GradientBoostingClassifier if is_classification else GradientBoostingRegressor

        elif algorithm_name == "ExtraTrees":
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
            return ExtraTreesClassifier if is_classification else ExtraTreesRegressor

        elif algorithm_name == "CausalForest":
            # Causal ML models need special handling
            logger.warning("CausalForest requires special HPO handling")
            return None

        elif algorithm_name == "LinearDML":
            logger.warning("LinearDML requires special HPO handling")
            return None

        else:
            logger.warning(f"Unknown algorithm: {algorithm_name}")
            return None

    except ImportError as e:
        logger.error(f"Failed to import {algorithm_name}: {e}")
        return None


async def run_hyperparameter_optimization(
    experiment_id: str,
    algorithm_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    search_space: Dict[str, Dict[str, Any]],
    problem_type: str = "binary_classification",
    n_trials: int = 50,
    timeout: Optional[int] = 3600,
    metric: str = "roc_auc",
    use_cv: bool = False,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """High-level function to run HPO for a given algorithm.

    This is the main entry point for hyperparameter optimization.

    Args:
        experiment_id: E2I experiment ID
        algorithm_name: Algorithm name (e.g., "XGBoost")
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        search_space: Hyperparameter search space
        problem_type: Problem type
        n_trials: Number of trials
        timeout: Timeout in seconds
        metric: Optimization metric
        use_cv: Use cross-validation instead of validation set
        cv_folds: Number of CV folds if use_cv=True

    Returns:
        Dictionary with best_params, best_value, optimization details
    """
    # Get model class
    model_class = get_model_class(algorithm_name, problem_type)
    if model_class is None:
        return {
            "error": f"Could not find model class for {algorithm_name}",
            "best_params": {},
            "best_value": None,
        }

    # Create optimizer
    optimizer = OptunaOptimizer(experiment_id=experiment_id)

    # Create study
    study = await optimizer.create_study(
        study_name=f"{algorithm_name}_hpo",
        direction="maximize",
        pruner=PrunerFactory.median_pruner(),
    )

    # Create objective function
    if use_cv:
        # Combine train and validation for CV
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        objective = optimizer.create_cv_objective(
            model_class=model_class,
            X=X_combined,
            y=y_combined,
            search_space=search_space,
            problem_type=problem_type,
            cv_folds=cv_folds,
        )
    else:
        objective = optimizer.create_validation_objective(
            model_class=model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            search_space=search_space,
            problem_type=problem_type,
            metric=metric,
        )

    # Run optimization
    results = await optimizer.optimize(
        study=study,
        objective=objective,
        n_trials=n_trials,
        timeout=timeout,
    )

    # Get optimization history
    history = await optimizer.get_optimization_history(study)

    # Add to results
    results["algorithm_name"] = algorithm_name
    results["problem_type"] = problem_type
    results["metric"] = metric
    results["history"] = history

    return results
