"""Model training for model_trainer.

This module trains ML models with the best hyperparameters.
Uses get_model_class from optuna_optimizer for dynamic model instantiation.

Version: 2.0.0
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type

import numpy as np

from src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner import (
    _LR_FIXED_PARAMS,
)
from src.agents.ml_foundation.model_trainer.random_state import (
    resolve_fold_random_state,
)
from src.mlops.lr_solver_policy import reconcile_lr_solver

logger = logging.getLogger(__name__)


async def train_model(state: Dict[str, Any]) -> Dict[str, Any]:
    """Train ML model with best hyperparameters.

    CRITICAL TRAINING PRINCIPLES:
    - Train ONLY on training set
    - Validation set used ONLY for early stopping (if enabled)
    - NEVER train on validation, test, or holdout
    - Test set touched ONCE for final evaluation
    - Holdout locked until post-deployment

    Args:
        state: ModelTrainerState with best_hyperparameters, preprocessed data,
               algorithm_name, problem_type, early_stopping config

    Returns:
        Dictionary with trained_model, training_duration_seconds,
        early_stopped, final_epoch, training_started_at, training_status

    Raises:
        No exceptions - returns error in state if training fails
    """
    # Extract training configuration
    algorithm_name = state.get("algorithm_name", "")
    problem_type = state.get("problem_type", "binary_classification")
    best_hyperparameters = state.get("best_hyperparameters", {})
    early_stopping = state.get("early_stopping", False)
    early_stopping_patience = state.get("early_stopping_patience", 10)

    # Check if resampling was applied - use resampled data if available
    resampling_applied = state.get("resampling_applied", False)

    if resampling_applied:
        # Use resampled training data (already preprocessed)
        X_train_preprocessed = state.get("X_train_resampled")
        y_train = state.get("y_train_resampled")
        logger.info(
            f"Using resampled training data: strategy={state.get('resampling_strategy', 'unknown')}, "
            f"original_shape={state.get('original_train_shape')}, "
            f"resampled_shape={state.get('resampled_train_shape')}"
        )
    else:
        # Use preprocessed training data (standard path)
        X_train_preprocessed = state.get("X_train_preprocessed")
        train_data = state.get("train_data", {})
        y_train = train_data.get("y")

    # Extract validation data (never resampled)
    X_validation_preprocessed = state.get("X_validation_preprocessed")
    validation_data = state.get("validation_data", {})
    y_validation = validation_data.get("y")

    # Extract feature columns for setting on model
    feature_columns = state.get("feature_columns")

    # Validate required data
    if X_train_preprocessed is None or y_train is None:
        logger.error("Missing training data for model training")
        return {
            "error": "Missing training data for model training",
            "error_type": "missing_training_data",
            "training_status": "failed",
        }

    if not algorithm_name:
        logger.error("algorithm_name not specified")
        return {
            "error": "algorithm_name not specified",
            "error_type": "missing_algorithm_name",
            "training_status": "failed",
        }

    # Record training start
    training_started_at = datetime.now(tz=timezone.utc).isoformat()
    start_time = time.time()

    logger.info(
        f"Starting model training: algorithm={algorithm_name}, "
        f"problem_type={problem_type}, "
        f"X_train shape={_get_shape(X_train_preprocessed)}"
    )

    # Get model class
    model_class = _get_model_class_dynamic(algorithm_name, problem_type)
    if model_class is None:
        logger.error(f"Could not get model class for {algorithm_name}")
        return {
            "error": f"Unsupported algorithm: {algorithm_name}",
            "error_type": "unsupported_algorithm",
            "training_status": "failed",
        }

    # Prepare hyperparameters - filter out incompatible params
    filtered_params = _filter_hyperparameters(algorithm_name, best_hyperparameters)

    # W3-lite Day 3 (shard 17 W3 row Day 3): when the orchestrator (Day 4-5)
    # sets a per-fold seed on the state, override whatever HPO baked into
    # ``best_hyperparameters['random_state']`` so the trained model uses the
    # fold-specific seed. ``hyperparameter_tuner`` resolves the same seed
    # upstream, so in-graph the values agree; this assignment makes
    # ``train_model`` self-contained for direct callers (replay tooling,
    # tier-0 smoke tests) that bypass HPO and pass ``best_hyperparameters``
    # in pre-baked.
    #
    # Cycle-14 codex IMPORTANT Q3 fix: gate the injection on whether the
    # algorithm's allowlist actually permits ``random_state``. Because
    # ``_filter_hyperparameters`` adds ``random_state`` to its return dict
    # for any allowed algorithm (via the ``common_params`` defaulting at the
    # bottom of that function), the presence of the key in ``filtered_params``
    # is an authoritative proxy for "allowlist permits it." For algorithms
    # whose allowlist excludes ``random_state`` (e.g., a future
    # ``LightGBM_Brier`` registered without an allowlist entry), blind
    # injection would crash ``model_class(**filtered_params)`` with a
    # ``TypeError`` on the unknown kwarg.
    if "fold_random_state" in state and "random_state" in filtered_params:
        filtered_params["random_state"] = resolve_fold_random_state(state)

    # Phase 1 W2 day-4 (shard 19 §C.4): if the candidate requires monotone
    # constraints, inject them from state["monotone_vector"] post-filter.
    # The constraint vector is per-feature and per-disease — it lives in
    # the data path (synthetic_data_generator_v2 emits it alongside
    # feature_columns), NOT the registry. Soft-fails to unconstrained
    # training if monotone_vector is missing OR length-mismatched (cycle-10
    # codex IMPORTANT fix per shard 19 §H risk-table mitigation).
    model_candidate_meta = state.get("model_candidate") or {}
    if model_candidate_meta.get("monotone_constraints_required"):
        monotone_vector = state.get("monotone_vector")
        if monotone_vector is None:
            logger.warning(
                f"{algorithm_name} requires monotone_vector but state is missing it; "
                "training without constraints (degraded to unconstrained variant)."
            )
        else:
            # Cycle-10 codex IMPORTANT fix: validate length BEFORE dispatch.
            # Without this, LightGBM raises a fatal C-side error mid-fit which
            # routes to error_type=training_failed (hard-fail). §H mitigation
            # says soft-degrade with WARNING.
            n_features = (
                X_train_preprocessed.shape[1] if hasattr(X_train_preprocessed, "shape") else None
            )
            if n_features is not None and len(monotone_vector) != n_features:
                logger.warning(
                    f"{algorithm_name} monotone_vector length ({len(monotone_vector)}) "
                    f"does not match X_train n_features ({n_features}); "
                    "training without constraints (degraded to unconstrained variant)."
                )
            elif algorithm_name.startswith("LightGBM"):
                # Cycle-10 codex COSMETIC: explicit int cast for dtype safety
                # (numpy int64 elements survive list() but explicit cast is
                # safer for forward-compat with future LightGBM versions).
                filtered_params["monotone_constraints"] = [int(v) for v in monotone_vector]
            elif algorithm_name.startswith("XGBoost"):
                # XGBoost expects the string format "(1, 0, -1, ...)"
                filtered_params["monotone_constraints"] = (
                    "(" + ", ".join(str(int(v)) for v in monotone_vector) + ")"
                )

    # Instantiate model
    try:
        model = model_class(**filtered_params)
        logger.info(f"Instantiated {algorithm_name} with params: {list(filtered_params.keys())}")
    except Exception as e:
        logger.error(f"Model instantiation failed: {e}")
        return {
            "error": f"Model instantiation failed: {str(e)}",
            "error_type": "instantiation_failed",
            "training_status": "failed",
        }

    # Prepare fit parameters
    fit_params = _prepare_fit_params(
        algorithm_name=algorithm_name,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        X_validation=X_validation_preprocessed,
        y_validation=y_validation,
        imbalance_detected=bool(state.get("imbalance_detected", False)),
        y_train=y_train,
    )

    # Train the model on TRAIN ONLY
    early_stopped = False
    final_epoch = None

    try:
        # Wrap X_train with the preprocessor's output feature names so
        # LightGBM 4.x doesn't auto-fill `feature_names_in_` with default
        # 'Column_0..N' (which then warns on every numpy predict call in
        # the evaluator). Using post-encoding names also makes SHAP's
        # downstream feature_names_in_ read produce the correct labels —
        # the prior code overrode feature_names_in_ with the *pre*-encoding
        # `feature_columns`, which mismatches the actual fitted feature
        # count and was silently passing wrong labels to SHAP.
        X_train_np = _wrap_with_feature_names(X_train_preprocessed, state)
        y_train_np = _ensure_numpy(y_train)

        # Fit model
        model.fit(X_train_np, y_train_np, **fit_params)

        # Check if early stopping occurred (for XGBoost/LightGBM)
        if early_stopping:
            early_stopped, final_epoch = _check_early_stopping(model, algorithm_name)

        logger.info(
            f"Model training completed: early_stopped={early_stopped}, final_epoch={final_epoch}"
        )

        # Fallback: if the wrap couldn't attach names (preprocessor
        # missing feature_names_out_ or shape mismatch), and we have
        # raw input column names, set them so SHAP still has labels.
        # Skip when fit already populated feature_names_in_ correctly.
        if (
            not hasattr(model, "feature_names_in_")
            or model.feature_names_in_ is None
            or (hasattr(model.feature_names_in_, "__len__") and len(model.feature_names_in_) == 0)
        ):
            if feature_columns is not None and len(feature_columns) > 0:
                try:
                    model.feature_names_in_ = np.array(feature_columns)
                    logger.info(
                        f"Set feature_names_in_ fallback with {len(feature_columns)} features"
                    )
                except (AttributeError, TypeError) as e:
                    logger.debug(f"Could not set feature_names_in_: {e}")

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return {
            "error": f"Model training failed: {str(e)}",
            "error_type": "training_failed",
            "training_status": "failed",
        }

    # Record training completion
    training_duration = time.time() - start_time
    training_completed_at = datetime.now(tz=timezone.utc).isoformat()

    logger.info(f"Training completed in {training_duration:.2f}s")

    return {
        "trained_model": model,
        "training_duration_seconds": training_duration,
        "early_stopped": early_stopped,
        "final_epoch": final_epoch,
        "training_started_at": training_started_at,
        "training_completed_at": training_completed_at,
        "training_status": "completed",
        "algorithm_name": algorithm_name,
        "framework": _get_framework(algorithm_name),
    }


def _get_model_class_dynamic(
    algorithm_name: str,
    problem_type: str,
) -> Optional[Type]:
    """Get model class for algorithm and problem type.

    Uses get_model_class from optuna_optimizer if available,
    falls back to direct imports.

    Args:
        algorithm_name: Algorithm name (XGBoost, LightGBM, RandomForest, etc.)
        problem_type: Problem type (binary_classification, regression, etc.)

    Returns:
        Model class or None if not found
    """
    # Try to use optuna_optimizer's get_model_class
    try:
        from src.mlops.optuna_optimizer import get_model_class

        return get_model_class(algorithm_name, problem_type)  # type: ignore[no-any-return]
    except ImportError:
        pass

    # Fallback to direct imports
    is_classification = problem_type in [
        "binary_classification",
        "multiclass_classification",
    ]

    try:
        if algorithm_name == "XGBoost":
            import xgboost as xgb

            return xgb.XGBClassifier if is_classification else xgb.XGBRegressor  # type: ignore[no-any-return]

        elif algorithm_name == "LightGBM":
            import lightgbm as lgb

            return lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor  # type: ignore[no-any-return]

        elif algorithm_name == "RandomForest":
            from sklearn.ensemble import (
                RandomForestClassifier,
                RandomForestRegressor,
            )

            return RandomForestClassifier if is_classification else RandomForestRegressor  # type: ignore[no-any-return]

        elif algorithm_name == "ExtraTrees":
            from sklearn.ensemble import (
                ExtraTreesClassifier,
                ExtraTreesRegressor,
            )

            return ExtraTreesClassifier if is_classification else ExtraTreesRegressor  # type: ignore[no-any-return]

        elif algorithm_name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression  # type: ignore[no-any-return]

        elif algorithm_name == "Ridge":
            from sklearn.linear_model import Ridge

            return Ridge  # type: ignore[no-any-return]

        elif algorithm_name == "Lasso":
            from sklearn.linear_model import Lasso

            return Lasso  # type: ignore[no-any-return]

        elif algorithm_name == "GradientBoosting":
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                GradientBoostingRegressor,
            )

            return GradientBoostingClassifier if is_classification else GradientBoostingRegressor  # type: ignore[no-any-return]

        elif algorithm_name == "SVM":
            from sklearn.svm import SVC, SVR

            return SVC if is_classification else SVR  # type: ignore[no-any-return]

        elif algorithm_name == "CausalForest":
            from econml.dml import CausalForestDML

            return CausalForestDML  # type: ignore[no-any-return]

        elif algorithm_name == "LinearDML":
            from econml.dml import LinearDML

            return LinearDML  # type: ignore[no-any-return]

        elif algorithm_name in ("DRLearner", "SLearner", "TLearner", "XLearner"):
            # Meta-learners share similar interface
            from econml import dr, metalearners

            mapping = {
                "DRLearner": dr.DRLearner,
                "SLearner": metalearners.SLearner,
                "TLearner": metalearners.TLearner,
                "XLearner": metalearners.XLearner,
            }
            return mapping[algorithm_name]  # type: ignore[no-any-return]

        elif algorithm_name == "NGBoost":
            # Phase 1 W2 (shard 19 §A.4 mirror). Mirrors optuna_optimizer.get_model_class
            # so the fallback path resolves NGBoost identically when the primary
            # delegate import fails.
            if is_classification:
                from src.mlops.wrappers.ngboost_wrapper import NGBoostBinaryClassifier

                return NGBoostBinaryClassifier  # type: ignore[no-any-return]
            from ngboost import NGBRegressor

            return NGBRegressor  # type: ignore[no-any-return]

        elif algorithm_name.endswith("_Monotone"):
            # Phase 1 W2 day-4 (shard 19 §C.2 mirror): _Monotone variants
            # reuse LightGBM/XGBoost. Strip suffix and recurse.
            base_name = algorithm_name[: -len("_Monotone")]
            return _get_model_class_dynamic(base_name, problem_type)

        elif algorithm_name.endswith("_Conformal"):
            # Phase 1 W2 day-3 (shard 19 §B.4 mirror). Same pattern as the
            # primary path in optuna_optimizer.get_model_class: strip suffix,
            # recurse to fetch the base class, return a closure factory.
            from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier

            base_name = algorithm_name[: -len("_Conformal")]
            base_cls = _get_model_class_dynamic(base_name, problem_type)
            if base_cls is None:
                return None

            def _conformal_factory(**params: Any) -> MapieConformalBinaryClassifier:
                method = params.pop("method", "lac")
                cv_val = params.pop("cv", 5)
                alpha = params.pop("alpha", 0.10)
                random_state = params.get("random_state", 42)
                base_inst = base_cls(**params)
                return MapieConformalBinaryClassifier(
                    base_estimator=base_inst,
                    method=method,
                    cv=cv_val,
                    alpha=alpha,
                    random_state=random_state,
                )

            return _conformal_factory  # type: ignore[return-value]

        else:
            logger.warning(f"Unknown algorithm: {algorithm_name}")
            return None

    except ImportError as e:
        logger.warning(f"Could not import model for {algorithm_name}: {e}")
        return None


def _filter_hyperparameters(
    algorithm_name: str,
    hyperparameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Filter hyperparameters to remove incompatible ones.

    Different models accept different parameters. This function
    filters out parameters that would cause errors.

    Args:
        algorithm_name: Algorithm name
        hyperparameters: Raw hyperparameters

    Returns:
        Filtered hyperparameters
    """
    # Base parameters that most sklearn models accept
    common_params = {
        "random_state": 42,
        "n_jobs": 1,
    }

    # Algorithm-specific allowed parameters
    allowed_params = {
        "XGBoost": {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
            "gamma",
            "scale_pos_weight",
            "random_state",
            "n_jobs",
            "verbosity",
            "eval_metric",
            "early_stopping_rounds",
            "use_label_encoder",
            # Phase 1 W2 day-4 (shard 19 §C.5): monotone_constraints injected
            # at fit time from state["monotone_vector"] for XGBoost_Monotone.
            "monotone_constraints",
        },
        "LightGBM": {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_samples",
            "reg_alpha",
            "reg_lambda",
            "num_leaves",
            "is_unbalance",
            "random_state",
            "n_jobs",
            "verbose",
            "importance_type",
            "subsample_freq",
            "min_split_gain",
            # Phase 1 W2 day-4 (shard 19 §C.5): monotone_constraints +
            # monotone_constraints_method injected for LightGBM_Monotone.
            "monotone_constraints",
            "monotone_constraints_method",
        },
        "RandomForest": {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "random_state",
            "n_jobs",
            "class_weight",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "oob_score",
        },
        "ExtraTrees": {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "random_state",
            "n_jobs",
            "class_weight",
        },
        "LogisticRegression": {
            "C",
            "penalty",
            "solver",
            "max_iter",
            "random_state",
            "class_weight",
            "l1_ratio",
            "tol",
            "warm_start",
        },
        "Ridge": {
            "alpha",
            "fit_intercept",
            "solver",
            "random_state",
            "tol",
        },
        "Lasso": {
            "alpha",
            "fit_intercept",
            "max_iter",
            "random_state",
            "tol",
            "warm_start",
            "selection",
        },
        "GradientBoosting": {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "random_state",
            "validation_fraction",
            "n_iter_no_change",
            "tol",
        },
        "SVM": {
            "C",
            "kernel",
            "degree",
            "gamma",
            "coef0",
            "shrinking",
            "probability",
            "tol",
            "cache_size",
            "class_weight",
            "random_state",
        },
        "CausalForest": {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
            "max_features",
            "inference",
            "n_jobs",
            "random_state",
            "model_y",
            "model_t",
            "discrete_treatment",
            "cv",
        },
        "LinearDML": {
            "model_y",
            "model_t",
            "discrete_treatment",
            "cv",
            "mc_iters",
            "random_state",
            "linear_first_stages",
        },
        "DRLearner": {
            "model_propensity",
            "model_regression",
            "model_final",
            "cv",
            "mc_iters",
            "random_state",
            "n_jobs",
        },
        "SLearner": {"overall_model", "cv", "random_state"},
        "TLearner": {"models", "cv", "random_state"},
        "XLearner": {"models", "propensity_model", "cate_models", "cv", "random_state"},
        # Phase 1 W2 day-1 (shard 19 §A.5). Mirror of NGBoostBinaryClassifier
        # constructor signature in src/mlops/wrappers/ngboost_wrapper.py.
        "NGBoost": {
            "n_estimators",
            "learning_rate",
            "minibatch_frac",
            "col_sample",
            "verbose",
            "random_state",
            "base_max_depth",
            "base_min_samples_leaf",
        },
    }

    # Phase 1 W2 day-3 (shard 19 §B.5): conformal entries compose their
    # allowlist as `{method, cv, alpha, random_state} ∪ allowed_params[base]`.
    # Routes the conformal-specific kwargs to the wrapper and the base-estimator
    # kwargs to the underlying constructor. The closure factory in
    # get_model_class.pop()s the conformal kwargs before building the base.
    # Phase 1 W2 day-4 (shard 19 §C.5): _Monotone variants reuse the base
    # allowlist directly (monotone_constraints already added to LightGBM/XGBoost
    # base allowlists above; injected from state["monotone_vector"] at fit time).
    _CONFORMAL_COMMON = {"method", "cv", "alpha", "random_state"}
    if algorithm_name.endswith("_Conformal"):
        base_name = algorithm_name[: -len("_Conformal")]
        base_allowed = allowed_params.get(base_name, set())
        allowed = _CONFORMAL_COMMON | base_allowed
    elif algorithm_name.endswith("_Monotone"):
        base_name = algorithm_name[: -len("_Monotone")]
        allowed = allowed_params.get(base_name, set())
    else:
        # Get allowed params for this algorithm
        allowed = allowed_params.get(algorithm_name, set())

    # Filter hyperparameters
    filtered = {}
    for key, value in hyperparameters.items():
        if key in allowed:
            filtered[key] = value

    # Add common params if not already set
    for key, value in common_params.items():
        if key in allowed and key not in filtered:
            filtered[key] = value

    # Algorithm-specific defaults
    # Cycle-9 codex F2+F3 fix: extend defaults to *_Conformal variants so
    # LightGBM_Conformal gets verbose=-1 (no log spam at day-5 smoke) and
    # LogisticRegression_Conformal gets max_iter=1000 (avoid ConvergenceWarning).
    # Phase 1 W2 day-4: also extend to *_Monotone variants (LightGBM_Monotone
    # / XGBoost_Monotone) for the same reason.
    if algorithm_name in {"XGBoost", "XGBoost_Conformal", "XGBoost_Monotone"}:
        if "verbosity" not in filtered:
            filtered["verbosity"] = 0
        if "use_label_encoder" not in filtered:
            filtered["use_label_encoder"] = False
    elif algorithm_name in {"LightGBM", "LightGBM_Conformal", "LightGBM_Monotone"}:
        if "verbose" not in filtered:
            filtered["verbose"] = -1
    elif algorithm_name in {"LogisticRegression", "LogisticRegression_Conformal"}:
        # Issue #232 defense-in-depth: route through the shared
        # ``_LR_FIXED_PARAMS`` helper so the final training constructor cannot
        # silently drop ``solver="saga"`` when a direct caller bypasses HPO
        # (or seeds ``best_hyperparameters={"penalty": "l1"}``) — same
        # ``Solver lbfgs supports only 'l2' or None penalties`` crash mode.
        for _lr_key, _lr_val in _LR_FIXED_PARAMS.items():
            if _lr_key not in filtered:
                filtered[_lr_key] = _lr_val
        # Issue #232 runtime: the penalty is KNOWN here (HPO best params or
        # registry default), so downgrade the saga floor to the faster lbfgs
        # for l2/None — saga is retained only for l1/elasticnet. Identical AUC,
        # ~20 iters vs 1000 (see tier0_optum_mart..._disproof_20260606.md).
        reconcile_lr_solver(filtered)

    return filtered


def _prepare_fit_params(
    algorithm_name: str,
    early_stopping: bool,
    early_stopping_patience: int,
    X_validation: Any,
    y_validation: Any,
    *,
    imbalance_detected: bool = False,
    y_train: Any = None,
) -> Dict[str, Any]:
    """Prepare fit parameters for training.

    Handles early stopping for XGBoost/LightGBM, and the GradientBoosting
    sample_weight bridge that mirrors what XGBoost gets via scale_pos_weight,
    LightGBM via is_unbalance, and RandomForest/LogisticRegression/ExtraTrees
    via class_weight="balanced". sklearn's ``GradientBoostingClassifier``
    rejects class_weight at construction but accepts ``sample_weight`` at
    ``.fit()``, so the imbalance signal flows through fit_params instead of
    fixed_params (backlog #20 Gap 3).

    Args:
        algorithm_name: Algorithm name
        early_stopping: Whether to use early stopping
        early_stopping_patience: Early stopping patience
        X_validation: Validation features
        y_validation: Validation labels
        imbalance_detected: Whether class imbalance was detected upstream
            (``detect_class_imbalance`` node). Only consulted for the
            GradientBoosting branch — other algos already get class-weight
            handling via ``_get_fixed_params`` constructor kwargs.
        y_train: Training labels used to compute the per-sample weights when
            ``algorithm_name == "GradientBoosting"`` and imbalance is
            detected. Accepts any array-like (numpy, pandas Series); the
            helper flattens via ``np.asarray``. Single-class y_train
            short-circuits to no sample_weight (degenerate; no positives or
            no negatives to reweight).

    Returns:
        Dictionary of fit parameters
    """
    fit_params: Dict[str, Any] = {}

    # Block 5 — Backlog #20 Gap 3: GradientBoosting sample_weight bridge.
    # sklearn GBM doesn't accept class_weight in the constructor, so the
    # symmetric handling is to compute "balanced" sample weights at fit
    # time. Note this runs even with early_stopping=False, since the
    # imbalance bridge is independent of early stopping.
    if algorithm_name == "GradientBoosting" and imbalance_detected and y_train is not None:
        from sklearn.utils.class_weight import compute_sample_weight

        y_train_arr = np.asarray(y_train).flatten()
        # compute_sample_weight raises on single-class y; the imbalance
        # detector emits imbalance_detected=False on degenerate splits but
        # we guard regardless so any caller path that synthesizes the flag
        # stays safe.
        if len(np.unique(y_train_arr)) >= 2:
            fit_params["sample_weight"] = compute_sample_weight("balanced", y_train_arr)
            logger.info(
                "Added sample_weight (n=%d) for GradientBoosting with imbalance_detected=True",
                len(fit_params["sample_weight"]),
            )

    if not early_stopping:
        return fit_params

    if X_validation is None or y_validation is None:
        logger.warning("Early stopping enabled but no validation data available")
        return fit_params

    # Convert validation data to numpy
    X_val_np = _ensure_numpy(X_validation)
    y_val_np = _ensure_numpy(y_validation)

    if algorithm_name == "XGBoost":
        fit_params["eval_set"] = [(X_val_np, y_val_np)]
        fit_params["verbose"] = False
        # Note: early_stopping_rounds is set in model params for newer XGBoost

    elif algorithm_name == "LightGBM":
        fit_params["eval_set"] = [(X_val_np, y_val_np)]
        fit_params["callbacks"] = [_get_lgbm_early_stopping_callback(early_stopping_patience)]

    elif algorithm_name == "GradientBoosting":
        # sklearn GradientBoosting uses validation_fraction and n_iter_no_change
        # for early stopping (set in model params, not fit params). The
        # sample_weight branch above handles imbalance defense-in-depth.
        pass

    return fit_params


def _get_lgbm_early_stopping_callback(patience: int):
    """Get LightGBM early stopping callback.

    Args:
        patience: Number of rounds without improvement

    Returns:
        LightGBM callback
    """
    try:
        import lightgbm as lgb

        return lgb.early_stopping(stopping_rounds=patience, verbose=False)
    except (ImportError, AttributeError):
        return None


def _check_early_stopping(model: Any, algorithm_name: str) -> tuple:
    """Check if early stopping occurred.

    Args:
        model: Trained model
        algorithm_name: Algorithm name

    Returns:
        Tuple of (early_stopped, final_epoch)
    """
    early_stopped = False
    final_epoch = None

    if algorithm_name == "XGBoost":
        # XGBoost stores best iteration
        if hasattr(model, "best_iteration"):
            best_iter = model.best_iteration
            if best_iter is not None and best_iter > 0:
                n_estimators = getattr(model, "n_estimators", None)
                if n_estimators and best_iter < n_estimators - 1:
                    early_stopped = True
                    final_epoch = best_iter

    elif algorithm_name == "LightGBM":
        # LightGBM stores best iteration
        if hasattr(model, "best_iteration_"):
            best_iter = model.best_iteration_
            if best_iter is not None and best_iter > 0:
                n_estimators = getattr(model, "n_estimators", None)
                if n_estimators and best_iter < n_estimators - 1:
                    early_stopped = True
                    final_epoch = best_iter

    elif algorithm_name == "GradientBoosting":
        # sklearn stores n_iter_ for early stopping
        if hasattr(model, "n_iter_"):
            final_epoch = model.n_iter_
            n_estimators = getattr(model, "n_estimators", None)
            if n_estimators and final_epoch < n_estimators:
                early_stopped = True

    return early_stopped, final_epoch


def _ensure_numpy(data: Any) -> Optional[np.ndarray]:
    """Convert data to numpy array if needed.

    Args:
        data: Input data

    Returns:
        Numpy array or None
    """
    if data is None:
        return None

    if isinstance(data, np.ndarray):
        return data

    # Try pandas conversion
    try:
        import pandas as pd

        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values  # type: ignore[no-any-return]
    except ImportError:
        pass

    # Try list/tuple conversion
    if isinstance(data, (list, tuple)):
        return np.array(data)

    return data  # type: ignore[no-any-return]


def _wrap_with_feature_names(data: Any, state: Dict[str, Any]) -> Any:
    """Return X as a DataFrame with the preprocessor's output feature names.

    Falls back to the original `data` when the preprocessor or names are
    unavailable or the column count does not match. See the comment on
    the call site in train_model() for why this matters for LightGBM 4.x.
    """
    try:
        import pandas as pd
    except ImportError:
        return data
    if data is None or isinstance(data, pd.DataFrame):
        return data
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        return data
    preprocessor = state.get("preprocessor")
    names = None
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = preprocessor.get_feature_names_out()
        except Exception:
            names = None
    if names is None or len(names) != data.shape[1]:
        return data
    return pd.DataFrame(data, columns=list(names))


def _get_shape(data: Any) -> str:
    """Get shape string for data.

    Args:
        data: Input data

    Returns:
        Shape string
    """
    if data is None:
        return "None"
    if hasattr(data, "shape"):
        return str(data.shape)
    if hasattr(data, "__len__"):
        return f"({len(data)},)"
    return "unknown"


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
        "DRLearner": "econml",
        "TLearner": "econml",
        "XLearner": "econml",
        # Phase 1 W2 (shard 19): NGBoost + monotone + conformal variants.
        # Framework value matches the registry entry's `framework` field.
        "NGBoost": "ngboost",
        "LightGBM_Monotone": "lightgbm",
        "XGBoost_Monotone": "xgboost",
        "NGBoost_Conformal": "mapie+ngboost",
        "LightGBM_Conformal": "mapie+lightgbm",
        "LogisticRegression_Conformal": "mapie+sklearn",
    }
    return framework_map.get(algorithm_name, "unknown")
