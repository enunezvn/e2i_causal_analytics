"""MLflow experiment tracking for model_trainer.

This module logs training runs to MLflow, including:
- Hyperparameters and configuration
- Training and evaluation metrics
- Trained model artifacts
- Model registration

It also persists training runs to the database with HPO linkage
for complete traceability between Optuna studies and training runs.

Version: 1.1.0
"""

import asyncio
import json
import logging
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast
from uuid import UUID

logger = logging.getLogger(__name__)


# Cycle-16 I-6 (Q3-C): MLflow's `mlflow.start_run(nested=True)` consults a
# thread-local active-run stack to attach the new run to the currently-open
# parent. Under `asyncio.gather(n_jobs > 1)` two folds running concurrently
# in the SAME thread share that thread-local — if fold A opens a nested
# child and yields control to fold B before closing it, fold B's
# `start_run(nested=True)` would attach to A's child instead of the outer
# parent, producing wrong topology in the MLflow UI.
#
# This module-level asyncio.Lock serializes the entire nested-run lifecycle
# (open + log + close) for repeated_k10 fold invocations so concurrent
# nested-run opens are impossible. Trade-off: serializes the MLflow-tracking
# portion of fold work; the rest of each fold (training, evaluation) still
# parallelizes via the orchestrator's Semaphore.
#
# Lazy creation avoids binding the lock to a specific event loop at import
# time — `asyncio.Lock()` constructed on first use binds to the running loop.
_nested_run_lock: Optional[asyncio.Lock] = None


def _get_nested_run_lock() -> asyncio.Lock:
    """Return the module-level nested-run lock, creating it on first use."""
    global _nested_run_lock
    if _nested_run_lock is None:
        _nested_run_lock = asyncio.Lock()
    return _nested_run_lock


from contextlib import asynccontextmanager


@asynccontextmanager
async def _maybe_serialize_nested_run(serialize: bool):
    """Acquire the nested-run lock only when ``serialize=True``.

    Single-mode invocations (``serialize=False``) skip locking entirely so
    tests + production single-runs incur no overhead. Repeated-mode fold
    invocations acquire the lock for the full nested-run lifecycle.
    """
    if serialize:
        async with _get_nested_run_lock():
            yield
    else:
        yield


def _get_training_run_repository():
    """Lazy import of MLTrainingRunRepository to avoid circular imports."""
    try:
        from src.repositories.ml_experiment import MLTrainingRunRepository

        return MLTrainingRunRepository()
    except ImportError:
        logger.debug("MLTrainingRunRepository not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get MLTrainingRunRepository: {e}")
        return None


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    """Coerce a metrics value to a plain dict for logging.

    ``test_metrics`` is typed ``MetricsSchema`` on ModelTrainerState (pydantic,
    coerced on assignment), while train/validation metrics are plain dicts;
    both must reduce to a dict for MLflow logging + JSON serialization.
    """
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = cast(Dict[str, Any], model_dump())
            # MetricsSchema's canonical AUC field is `auc_roc` (roc_auc is an
            # INPUT alias), so model_dump() emits `auc_roc`. But _log_split_metrics
            # logs `<split>_<key>` and _get_primary_metric keys off `roc_auc` (the
            # producer/plain-dict shape that validation_metrics uses). Mirror it so
            # split keys stay consistent across splits and primary_metric resolves.
            if "auc_roc" in dumped and "roc_auc" not in dumped:
                dumped = {**dumped, "roc_auc": dumped["auc_roc"]}
            return dumped
        except Exception:
            pass
    items = getattr(value, "items", None)
    if callable(items):
        try:
            return dict(items())
        except Exception:
            pass
    return {}


def _resolve_evaluation_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return the split-metrics bundle (train/validation/test/holdout).

    The evaluator writes these as TOP-LEVEL state keys (``evaluate_model``
    returns ``{**metrics_result, ...}``); the legacy ``evaluation_metrics``
    wrapper is never written in production (grep: only read, never assigned in
    src/). Prefer an explicit wrapper when a caller supplies one (older callers
    / tests), else assemble it from the top-level keys so split metrics, the
    primary metric, and the summary artifact are actually logged (gap G9).
    """
    explicit = state.get("evaluation_metrics")
    if explicit:
        return cast(Dict[str, Any], explicit)
    assembled: Dict[str, Any] = {}
    for key in ("train_metrics", "validation_metrics", "test_metrics", "holdout_metrics"):
        plain = _to_plain_dict(state.get(key))
        if plain:
            assembled[key] = plain
    return assembled


async def log_to_mlflow(state: Dict[str, Any]) -> Dict[str, Any]:
    """Log training run to MLflow.

    This node logs the complete training run including:
    - Parameters: algorithm, hyperparameters, problem type
    - Metrics: all train/validation/test metrics
    - Model: trained model artifact
    - Tags: metadata for filtering runs

    Args:
        state: ModelTrainerState with trained_model, metrics,
               best_hyperparameters, algorithm_name, etc.

    Returns:
        Dictionary with mlflow_run_id, mlflow_experiment_id,
        mlflow_model_uri, mlflow_registered, mlflow_status
    """
    # Check if MLflow logging is enabled
    enable_mlflow = state.get("enable_mlflow", True)

    if not enable_mlflow:
        logger.info("MLflow logging disabled")
        return {
            "mlflow_status": "disabled",
            "mlflow_run_id": None,
            "mlflow_experiment_id": None,
        }

    # Check if we have a trained model
    trained_model = state.get("trained_model")
    if trained_model is None:
        logger.warning("No trained model to log to MLflow")
        return {
            "mlflow_status": "skipped",
            "mlflow_run_id": None,
            "error": "No trained model available",
        }

    # Extract state values
    experiment_id = state.get("experiment_id", "unknown")
    experiment_name = state.get("experiment_name", f"model_trainer_{experiment_id}")
    algorithm_name = state.get("algorithm_name", "unknown")
    problem_type = state.get("problem_type", "binary_classification")
    framework = state.get("framework", _get_framework(algorithm_name))
    best_hyperparameters = state.get("best_hyperparameters", {})

    # Training metadata
    state.get("training_duration_seconds", 0)
    early_stopped = state.get("early_stopped", False)
    state.get("final_epoch")

    # HPO metadata
    hpo_completed = state.get("hpo_completed", False)
    state.get("hpo_best_value")
    hpo_trials_run = state.get("hpo_trials_run", 0)
    hpo_study_name = state.get("hpo_study_name")  # Optuna study name for linkage
    hpo_best_trial = state.get("hpo_best_trial")  # Best trial number
    feast_fallback_used = state.get("feast_fallback_used", False)

    # Evaluation metrics. Production writes these as TOP-LEVEL state keys, not
    # under an `evaluation_metrics` wrapper (gap G9) — resolve both shapes.
    evaluation_metrics = _resolve_evaluation_metrics(state)
    train_metrics = evaluation_metrics.get("train_metrics", {})
    validation_metrics = evaluation_metrics.get("validation_metrics", {})
    test_metrics = evaluation_metrics.get("test_metrics", {})
    holdout_metrics = evaluation_metrics.get("holdout_metrics", {})

    # Model registration config
    register_model = state.get("register_model", False)
    model_name = state.get("model_name", f"{algorithm_name.lower()}_model")
    model_description = state.get("model_description", "")
    model_tags = state.get("model_tags", {})

    # Training data metadata (for database persistence)
    training_samples = state.get("training_samples", 0)
    validation_samples = state.get("validation_samples")
    test_samples = state.get("test_samples")
    feature_names = state.get("feature_names", [])

    try:
        from src.mlops.mlflow_connector import get_mlflow_connector

        mlflow_conn = get_mlflow_connector()

        # Create/get experiment
        mlflow_experiment_id = await mlflow_conn.get_or_create_experiment(
            name=experiment_name,
            tags={
                "problem_type": problem_type,
                "framework": framework,
                "source": "model_trainer_agent",
            },
        )

        logger.info(f"MLflow experiment: {experiment_name} (ID: {mlflow_experiment_id})")

        # Generate run name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Phase 1 W3-lite Day-5 (cycle-15 I-4): when invoked from the
        # `_run_repeated_splits` orchestrator (per-fold recursion sentinel
        # `repeated_mode_fold_invocation=True`), the parent run is already
        # open at the orchestrator level — open this fold's run as a NESTED
        # child of the parent and tag it with `fold_idx` /
        # `evaluation_mode=repeated_k10` / `fold_seed` so the MLflow UI
        # surfaces the parent ↔ child topology.
        evaluation_mode = state.get("evaluation_mode", "single")
        is_repeated_fold = evaluation_mode == "repeated_k10" and bool(
            state.get("repeated_mode_fold_invocation", False)
        )
        if is_repeated_fold:
            fold_idx_value = state.get("fold_idx", 0)
            fold_seed_value = state.get("fold_random_state", 0)
            run_name = f"fold_{int(fold_idx_value):02d}"
            fold_tags = {
                "fold_idx": str(int(fold_idx_value)),
                "evaluation_mode": "repeated_k10",
                "fold_seed": str(int(fold_seed_value)),
            }
        else:
            run_name = f"{algorithm_name}_{timestamp}"
            fold_tags = {}

        run_tags = {
            "algorithm": algorithm_name,
            "problem_type": problem_type,
            "framework": framework,
            "source": "model_trainer_agent",
            "hpo_enabled": str(hpo_completed),
            "feast_fallback": str(feast_fallback_used),
            # Block 5B (#10): emit the validation-set business_utility
            # number as a tag so the MLflow UI can display "what this
            # run was worth" alongside accuracy/AUC. Mirrors the
            # feast_fallback tag pattern. ``"N/A"`` when no
            # cost_matrix was provided (validation_metrics omits
            # the key in that case — see evaluator.py: the
            # ``if cost_matrix is not None`` short-circuit in
            # ``_compute_classification_metrics`` skips
            # business_utility computation, so the metrics dict
            # never picks up the key).
            "business_utility": str(validation_metrics.get("business_utility", "N/A")),
            **fold_tags,
        }

        # Start MLflow run (nested when invoked per-fold inside repeated_k10)
        # Cycle-16 I-6 (Q3-C): wrap the nested-run lifecycle in
        # _maybe_serialize_nested_run when invoked per-fold so concurrent
        # asyncio.gather(n_jobs > 1) folds cannot overlap their nested-run
        # opens against MLflow's thread-local active-run state.
        async with (
            _maybe_serialize_nested_run(serialize=is_repeated_fold),
            mlflow_conn.start_run(
                experiment_id=mlflow_experiment_id,
                run_name=run_name,
                tags=run_tags,
                description=f"Training run for {algorithm_name} on {problem_type}",
                nested=is_repeated_fold,
            ) as run,
        ):
            mlflow_run_id = run.run_id

            # Log hyperparameters
            await _log_hyperparameters(run, best_hyperparameters, algorithm_name)

            # Log training configuration
            await run.log_params(
                {
                    "algorithm_name": algorithm_name,
                    "problem_type": problem_type,
                    "framework": framework,
                    "hpo_enabled": hpo_completed,
                    "hpo_trials": hpo_trials_run,
                    "early_stopping_enabled": early_stopped,
                }
            )

            # Log training metrics
            await _log_training_metrics(run, state)

            # Log evaluation metrics for each split
            await _log_split_metrics(run, "train", train_metrics)
            await _log_split_metrics(run, "validation", validation_metrics)
            await _log_split_metrics(run, "test", test_metrics)
            if holdout_metrics:
                await _log_split_metrics(run, "holdout", holdout_metrics)

            # Log primary metric for easy comparison
            primary_metric = _get_primary_metric(test_metrics or validation_metrics, problem_type)
            if primary_metric:
                await run.log_metrics({"primary_metric": primary_metric})

            # Log model artifact
            logger.info(
                f"Logging model artifact: algorithm={algorithm_name}, "
                f"framework={framework}, model_type={type(trained_model).__name__}"
            )
            model_uri = await _log_model_artifact(run, trained_model, algorithm_name, framework)
            logger.info(f"Model artifact logging result: model_uri={model_uri}")

            # Log additional artifacts
            await _log_additional_artifacts(run, state)

            metric_str = f"{primary_metric:.4f}" if primary_metric else "N/A"
            logger.info(
                f"Logged run to MLflow: run_id={mlflow_run_id}, primary_metric={metric_str}"
            )

        # Register model if requested
        model_version = None
        if register_model and model_uri:
            model_version = await mlflow_conn.register_model(
                run_id=mlflow_run_id,
                model_name=model_name,
                model_path="model",
                description=model_description or f"{algorithm_name} model for {problem_type}",
                tags={
                    **model_tags,
                    "algorithm": algorithm_name,
                    "problem_type": problem_type,
                },
            )
            if model_version:
                logger.info(f"Registered model: {model_name} v{model_version.version}")

        # Persist training run to database with HPO linkage
        db_run_id = await _persist_training_run(
            experiment_id=experiment_id,
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
            algorithm_name=algorithm_name,
            hyperparameters=best_hyperparameters,
            training_samples=training_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            feature_names=feature_names,
            hpo_study_name=hpo_study_name,
            hpo_best_trial=hpo_best_trial,
        )

        return {
            "mlflow_status": "success",
            "mlflow_run_id": mlflow_run_id,
            "mlflow_experiment_id": mlflow_experiment_id,
            "mlflow_model_uri": model_uri,
            "mlflow_registered": model_version is not None,
            "mlflow_model_version": model_version.version if model_version else None,
            "mlflow_model_name": model_name if model_version else None,
            "db_training_run_id": str(db_run_id) if db_run_id else None,
        }

    except ImportError as e:
        logger.warning(f"MLflow not available: {e}")
        return {
            "mlflow_status": "unavailable",
            "error": f"MLflow import failed: {e}",
            "mlflow_run_id": None,
        }

    except Exception as e:
        logger.error(f"MLflow logging failed: {e}")
        return {
            "mlflow_status": "failed",
            "error": f"MLflow logging failed: {e}",
            "mlflow_run_id": None,
        }


async def _log_hyperparameters(
    run: Any,
    hyperparameters: Dict[str, Any],
    algorithm_name: str,
) -> None:
    """Log hyperparameters to MLflow run.

    Args:
        run: MLflowRun object
        hyperparameters: Hyperparameter dictionary
        algorithm_name: Algorithm name for prefixing
    """
    if not hyperparameters:
        return

    # Log with hp_ prefix for clarity
    params = {}
    for key, value in hyperparameters.items():
        # Skip internal parameters
        if key.startswith("_"):
            continue

        # MLflow requires string values
        if isinstance(value, (list, dict)):
            params[f"hp_{key}"] = json.dumps(value)
        else:
            params[f"hp_{key}"] = value

    if params:
        await run.log_params(params)


async def _log_training_metrics(run: Any, state: Dict[str, Any]) -> None:
    """Log training-related metrics.

    Args:
        run: MLflowRun object
        state: Training state dictionary
    """
    metrics = {}

    # Training duration
    if "training_duration_seconds" in state:
        metrics["training_duration_seconds"] = state["training_duration_seconds"]

    # Early stopping info
    if state.get("early_stopped"):
        metrics["early_stopped"] = 1.0
        if state.get("final_epoch"):
            metrics["final_epoch"] = float(state["final_epoch"])

    # HPO metrics
    if state.get("hpo_completed"):
        metrics["hpo_completed"] = 1.0
        if state.get("hpo_best_value"):
            metrics["hpo_best_value"] = state["hpo_best_value"]
        if state.get("hpo_trials_run"):
            metrics["hpo_trials_run"] = float(state["hpo_trials_run"])
        if state.get("hpo_duration_seconds"):
            metrics["hpo_duration_seconds"] = state["hpo_duration_seconds"]

    if metrics:
        await run.log_metrics(metrics)


async def _log_split_metrics(
    run: Any,
    split_name: str,
    metrics: Dict[str, Any],
) -> None:
    """Log metrics for a data split.

    Args:
        run: MLflowRun object
        split_name: Split name (train, validation, test, holdout)
        metrics: Metrics dictionary
    """
    if not metrics:
        return

    prefixed_metrics = {}
    for key, value in metrics.items():
        # Skip non-numeric values
        if not isinstance(value, (int, float)):
            continue

        # Skip None values
        if value is None:
            continue

        prefixed_metrics[f"{split_name}_{key}"] = float(value)

    if prefixed_metrics:
        await run.log_metrics(prefixed_metrics)


async def _log_model_artifact(
    run: Any,
    model: Any,
    algorithm_name: str,
    framework: str,
) -> Optional[str]:
    """Log trained model as MLflow artifact.

    Args:
        run: MLflowRun object
        model: Trained model object
        algorithm_name: Algorithm name
        framework: ML framework

    Returns:
        Model URI if successful
    """
    # Determine MLflow flavor based on framework/algorithm
    flavor = _get_mlflow_flavor(algorithm_name, framework)

    try:
        logger.info(f"Attempting to log model with flavor={flavor}")
        model_uri = await run.log_model(
            model=model,
            name="model",
            flavor=flavor,
        )
        logger.info(f"Successfully logged model: {model_uri}")
        return cast(str, model_uri)
    except Exception as e:
        logger.warning(f"Failed to log model with {flavor} flavor: {e}", exc_info=True)
        # Fallback to sklearn flavor
        try:
            logger.info("Attempting fallback to sklearn flavor")
            model_uri = await run.log_model(
                model=model,
                name="model",
                flavor="sklearn",
            )
            logger.info(f"Successfully logged model with sklearn fallback: {model_uri}")
            return cast(str, model_uri)
        except Exception as e2:
            logger.error(f"Failed to log model with sklearn fallback: {e2}", exc_info=True)
            return None


async def _log_additional_artifacts(run: Any, state: Dict[str, Any]) -> None:
    """Log additional artifacts like feature importance, etc.

    Args:
        run: MLflowRun object
        state: Training state dictionary
    """
    # Log feature importance if available
    feature_importance = state.get("feature_importance")
    if feature_importance:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(feature_importance, f, indent=2)
                f.flush()
                await run.log_artifact(f.name, "feature_importance.json")
        except Exception as e:
            logger.warning(f"Failed to log feature importance: {e}")

    # Log confusion matrix if available
    confusion_matrix = state.get("confusion_matrix")
    if confusion_matrix:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(confusion_matrix, f, indent=2)
                f.flush()
                await run.log_artifact(f.name, "confusion_matrix.json")
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    # Log evaluation summary. Resolve the production top-level-key shape (gap
    # G9) so the summary isn't silently empty (the `evaluation_metrics` wrapper
    # is never written in src/).
    evaluation_metrics = _resolve_evaluation_metrics(state)
    if evaluation_metrics:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(evaluation_metrics, f, indent=2, default=str)
                f.flush()
                await run.log_artifact(f.name, "evaluation_summary.json")
        except Exception as e:
            logger.warning(f"Failed to log evaluation summary: {e}")

    # Log the rich nested evaluation blocks the evaluator computes as TOP-LEVEL
    # state keys (calibration curve + ECE, cross-validation per-fold values,
    # Layer-3 per-feature ablation). These are NOT part of the per-split metrics
    # scalars, so without this they never reach MLflow and dashboards can't
    # audit them without re-running the evaluator (gap G9). Each is dumped as
    # its own JSON artifact via the existing temp-file → log_artifact pattern
    # (the connector run object exposes no log_dict).
    #
    # NB (codex review): net_benefit_grid + decision_curve_data are NOT top-level
    # state keys — the evaluator nests them inside test_metrics
    # (evaluator.py:2028/2086). net_benefit_grid is a declared MetricsSchema field
    # so it survives coercion and is captured in evaluation_summary.json above;
    # decision_curve_data is NOT a declared field, so MetricsSchema coercion
    # (extra="ignore") drops it before this node — surfacing it needs a schema-
    # declaration fix upstream, not a logger change. Hence neither is listed here.
    rich_artifacts = (
        ("calibration_analysis", "calibration_analysis.json"),
        ("cv_results", "cv_results.json"),
        ("model_eval_ablation", "model_eval_ablation.json"),
    )
    for state_key, artifact_name in rich_artifacts:
        block = state.get(state_key)
        if not block:
            continue
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(block, f, indent=2, default=str)
                f.flush()
                await run.log_artifact(f.name, artifact_name)
        except Exception as e:
            logger.warning(f"Failed to log {artifact_name}: {e}")


def _get_framework(algorithm_name: str) -> str:
    """Get framework name for algorithm.

    Args:
        algorithm_name: Algorithm name

    Returns:
        Framework name
    """
    framework_map = {
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "RandomForest": "sklearn",
        "ExtraTrees": "sklearn",
        "LogisticRegression": "sklearn",
        "Ridge": "sklearn",
        "Lasso": "sklearn",
        "GradientBoosting": "sklearn",
        "SVM": "sklearn",
        "CausalForest": "econml",
        "LinearDML": "econml",
        "SLearner": "econml",
    }
    return framework_map.get(algorithm_name, "sklearn")


def _get_mlflow_flavor(algorithm_name: str, framework: str) -> str:
    """Get MLflow flavor for model logging.

    Args:
        algorithm_name: Algorithm name
        framework: Framework name

    Returns:
        MLflow flavor name
    """
    if algorithm_name == "XGBoost" or framework == "xgboost":
        return "xgboost"
    elif algorithm_name == "LightGBM" or framework == "lightgbm":
        return "lightgbm"
    # Phase 1 W2 day-5 follow-up (cycle-11 codex IMPORTANT): explicit
    # branches for the W2 algorithms so the flavor choice is intentional,
    # not a silent fallthrough. NGBoost and MAPIE conformal wrappers have
    # no native MLflow flavor — they serialize via cloudpickle through
    # the sklearn flavor (works because the wrappers expose
    # sklearn-compatible fit/predict/predict_proba). Future work: switch
    # to mlflow.pyfunc with a PythonModel adapter when full registry
    # deployment lands (W3-lite or later when k=10 runs surface 10× MLflow
    # artifacts per algorithm).
    elif framework == "ngboost":
        return "sklearn"  # cloudpickle: NGBoost wrapper, sklearn-compatible
    elif framework.startswith("mapie+"):
        return "sklearn"  # cloudpickle: MapieConformalBinaryClassifier wrapper
    else:
        return "sklearn"


def _get_primary_metric(
    metrics: Dict[str, Any],
    problem_type: str,
) -> Optional[float]:
    """Get primary metric for model comparison.

    Args:
        metrics: Metrics dictionary
        problem_type: Problem type

    Returns:
        Primary metric value or None
    """
    if not metrics:
        return None

    # Primary metric by problem type
    primary_metric_map = {
        "binary_classification": ["roc_auc", "auc", "f1", "accuracy"],
        "multiclass_classification": ["f1_weighted", "f1", "accuracy"],
        "regression": ["r2", "rmse", "mae"],
    }

    primary_candidates = primary_metric_map.get(problem_type, ["roc_auc", "f1", "r2"])

    for metric_name in primary_candidates:
        if metric_name in metrics and metrics[metric_name] is not None:
            return float(metrics[metric_name])

    return None


async def _persist_training_run(
    experiment_id: str,
    run_name: str,
    mlflow_run_id: str,
    algorithm_name: str,
    hyperparameters: Dict[str, Any],
    training_samples: int,
    validation_samples: Optional[int],
    test_samples: Optional[int],
    feature_names: List[str],
    hpo_study_name: Optional[str],
    hpo_best_trial: Optional[int],
) -> Optional[UUID]:
    """Persist training run to database with HPO linkage.

    This creates a record in ml_training_runs table that links the
    training run to its Optuna HPO study for complete traceability.

    Args:
        experiment_id: ML experiment ID (may be string or UUID)
        run_name: Human-readable run name
        mlflow_run_id: MLflow run ID for cross-reference
        algorithm_name: Algorithm used
        hyperparameters: Best hyperparameters used
        training_samples: Number of training samples
        validation_samples: Number of validation samples
        test_samples: Number of test samples
        feature_names: List of feature names
        hpo_study_name: Optuna study name for HPO linkage
        hpo_best_trial: Best trial number from Optuna

    Returns:
        Database run ID if successful, None otherwise
    """
    repo = _get_training_run_repository()
    if not repo:
        logger.debug("Training run repository not available, skipping DB persistence")
        return None

    try:
        # Convert experiment_id to UUID if it's a valid UUID string
        try:
            exp_uuid = UUID(experiment_id) if experiment_id != "unknown" else None
        except (ValueError, TypeError):
            exp_uuid = None

        if not exp_uuid:
            logger.debug("No valid experiment_id for DB persistence")
            return None

        # Create training run with HPO linkage
        run = await repo.create_run_with_hpo(
            experiment_id=exp_uuid,
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
            algorithm=algorithm_name,
            hyperparameters=hyperparameters or {},
            training_samples=training_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            feature_names=feature_names or [],
            optuna_study_name=hpo_study_name,
            optuna_trial_number=hpo_best_trial,
            is_best_trial=hpo_best_trial is not None,
        )

        logger.info(
            f"Persisted training run to database: id={run.id}, hpo_study={hpo_study_name or 'None'}"
        )
        return cast(UUID, run.id)

    except Exception as e:
        # Non-fatal: MLflow logging succeeded, DB persistence is secondary
        logger.warning(f"Failed to persist training run to database: {e}")
        return None
