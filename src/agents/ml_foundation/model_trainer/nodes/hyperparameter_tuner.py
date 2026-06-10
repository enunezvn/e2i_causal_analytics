"""Hyperparameter tuning for model_trainer.

This module uses Optuna to optimize hyperparameters on the validation set.
Uses the OptunaOptimizer from src.mlops for unified HPO management.

Features:
- Validation-based HPO (no test set leakage)
- Warm-starting from similar successful patterns
- Pattern storage for procedural memory
- Opik instrumentation for observability

Version: 2.3.0
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from src.agents.ml_foundation.model_trainer.random_state import (
    resolve_fold_random_state,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Issue #232: LR + LR_Conformal must pin ``solver="saga"`` so HPO trials with
# ``penalty="l1"`` don't crash with
#     ``Solver lbfgs supports only 'l2' or None penalties, got l1 penalty.``
#
# Module-level helper so the HPO dispatcher (Path 2) and the tier-0 Step 5b
# alt-train builder (Path 1, ``scripts/run_tier0_test.py``) consume the same
# constant — guaranteeing the two paths cannot drift independently.
#
# Per-call kwargs (e.g. ``random_state``) are NOT part of this constant; they
# stay as call-site arguments so each fold/trial can still pin its own seed.
# ---------------------------------------------------------------------------
_LR_FIXED_PARAMS: Dict[str, Any] = {
    "max_iter": 1000,
    "solver": "saga",
}


# ---------------------------------------------------------------------------
# Issue #868: final-fit degeneracy guard thresholds.
#
# Measured defect (synthetic-CSU persistence cohort): Optuna's best
# LogisticRegression_Conformal trial was ``C=0.0019517/l1`` with best value
# 0.6499999999999999 — which is EXACTLY ``0.7*0.5 + 0.3*1.0``: the blended
# roc_auc objective (``OptunaOptimizer._evaluate_model``) REWARDING a
# degenerate constant classifier (val AUC 0.5 + all-positive predictions →
# minority recall 1.0 when positives are the majority). At that C the L1
# penalty zeroes every coefficient (intercept-only model → near-constant
# probabilities → test AUC exactly 0.500), while the same study's runner-up
# trial (healthy C) delivers real discrimination.
#
# Threshold rationale:
# - PROMISE_FLOOR 0.58: the guard only arms when HPO promised genuine
#   discrimination (observed degenerate promise: 0.6499...). Below it,
#   "collapse vs promise" is undefined and weak-but-honest models must
#   pass through unchanged (conservative: deployed healthy cohorts are
#   byte-identical in behavior).
# - PROB_STD_FLOOR 1e-3: an intercept-only logistic model emits literally
#   identical probabilities (std ~1e-16); healthy fits on these cohorts
#   measure std ~0.05-0.4. 1e-3 separates the two by >1 order of magnitude
#   in both directions.
# - VAL_AUC_FLOOR 0.52: chance level plus a small noise margin on the
#   ~500-1500-row validation splits — a model at/below it has no
#   discrimination despite the >=0.58 promise.
# ---------------------------------------------------------------------------
DEGENERACY_GUARD_PROMISE_FLOOR: float = 0.58
DEGENERACY_GUARD_PROB_STD_FLOOR: float = 1e-3
DEGENERACY_GUARD_VAL_AUC_FLOOR: float = 0.52


def _get_opik_connector():
    """Lazy import of OpikConnector to avoid circular imports."""
    try:
        from src.mlops.opik_connector import get_opik_connector

        return get_opik_connector()
    except ImportError:
        logger.debug("OpikConnector not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get OpikConnector: {e}")
        return None


def _get_hpo_pattern_memory():
    """Lazy import of HPO pattern memory module."""
    try:
        from src.mlops import hpo_pattern_memory

        return hpo_pattern_memory
    except ImportError:
        logger.debug("HPO pattern memory not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get HPO pattern memory: {e}")
        return None


# HPO output contract validation
HPO_REQUIRED_FIELDS = {
    "hpo_completed": bool,
    "best_hyperparameters": dict,
}

HPO_OPTIONAL_FIELDS = {
    "hpo_best_trial": (int, type(None)),
    "hpo_best_value": (float, int, type(None)),
    "hpo_trials_run": int,
    "hpo_trials_pruned": int,
    "hpo_duration_seconds": (float, int),
    "hpo_study_name": (str, type(None)),
    "hpo_metric": (str, type(None)),
    "hpo_pattern_id": (str, type(None)),  # Procedural memory pattern ID
    # Issue #868: final-fit degeneracy guard decision metadata.
    "hpo_degeneracy_guard": (dict, type(None)),
}


def validate_hpo_output(output: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate HPO output against contract.

    This ensures the HPO output contains all required fields with correct types.

    Args:
        output: HPO output dictionary

    Returns:
        Tuple of (is_valid, list of validation errors)
    """
    errors = []

    # Check required fields
    for field, expected_type in HPO_REQUIRED_FIELDS.items():
        if field not in output:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(output[field], expected_type):
            errors.append(
                f"Invalid type for {field}: expected {expected_type.__name__}, "
                f"got {type(output[field]).__name__}"
            )

    # Check optional fields if present
    for field, expected_types in HPO_OPTIONAL_FIELDS.items():
        if field in output and output[field] is not None:
            if not isinstance(output[field], expected_types):  # type: ignore[arg-type]
                type_names = (
                    expected_types.__name__  # type: ignore[attr-defined]
                    if not isinstance(expected_types, tuple)
                    else "/".join(t.__name__ for t in expected_types)
                )
                errors.append(
                    f"Invalid type for {field}: expected {type_names}, "
                    f"got {type(output[field]).__name__}"
                )

    # Validate hpo_completed consistency
    if output.get("hpo_completed"):
        # When HPO completed, these fields should be present
        if output.get("hpo_trials_run", 0) == 0:
            errors.append("hpo_completed=True but hpo_trials_run=0")
        if "hpo_best_value" not in output:
            errors.append("hpo_completed=True but hpo_best_value missing")
        if "hpo_study_name" not in output:
            errors.append("hpo_completed=True but hpo_study_name missing")

    return (len(errors) == 0, errors)


def validate_hyperparameter_types(
    hyperparameters: Dict[str, Any],
    search_space: Dict[str, Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Validate hyperparameter types match search space definitions.

    Args:
        hyperparameters: Best hyperparameters from HPO
        search_space: Search space definitions from model_selector

    Returns:
        Tuple of (is_valid, list of validation errors)
    """
    errors = []

    for param_name, param_def in search_space.items():
        if param_name not in hyperparameters:
            # This may be acceptable if using defaults
            continue

        value = hyperparameters[param_name]
        param_type = param_def.get("type", "float")

        # Validate type
        if param_type == "int":
            if not isinstance(value, (int, np.integer)):
                errors.append(f"Parameter {param_name}: expected int, got {type(value).__name__}")
        elif param_type == "float":
            if not isinstance(value, (int, float, np.floating)):
                errors.append(f"Parameter {param_name}: expected float, got {type(value).__name__}")
        elif param_type == "categorical":
            choices = param_def.get("choices", [])
            if choices and value not in choices:
                errors.append(f"Parameter {param_name}: value {value} not in choices {choices}")

        # Validate range for numeric types
        if param_type in ("int", "float") and value is not None:
            low = param_def.get("low")
            high = param_def.get("high")
            if low is not None and value < low:
                errors.append(f"Parameter {param_name}: value {value} below minimum {low}")
            if high is not None and value > high:
                errors.append(f"Parameter {param_name}: value {value} above maximum {high}")

    return (len(errors) == 0, errors)


def _detect_final_fit_collapse(
    model: Any,
    X_val: Any,
    y_val: Any,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Detect whether a fitted binary classifier collapsed (issue #868).

    Collapse criteria (see DEGENERACY_GUARD_* rationale above):
    - near-constant predicted probabilities (std < 1e-3); subsumes the
      constant/intercept-only classifier, OR
    - chance-level validation AUC (<= 0.52) despite probability spread, OR
    - single-class hard predictions, ONLY for models without
      ``predict_proba`` (a healthy imbalanced model can legitimately
      predict one class everywhere at the 0.5 cut while still ranking
      well — hard predictions alone are not evidence of collapse when
      probabilities are available).

    Returns:
        (collapsed, reason, diagnostics)
    """
    diagnostics: Dict[str, Any] = {}

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_val))
        p1 = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
        prob_std = float(np.std(p1))
        diagnostics["val_prob_std"] = prob_std
        if prob_std < DEGENERACY_GUARD_PROB_STD_FLOOR:
            return (
                True,
                f"near-constant predicted probabilities "
                f"(std={prob_std:.2e} < {DEGENERACY_GUARD_PROB_STD_FLOOR})",
                diagnostics,
            )
        y_arr = np.asarray(y_val).ravel()
        if len(np.unique(y_arr)) >= 2:
            from sklearn.metrics import roc_auc_score

            val_auc = float(roc_auc_score(y_arr, p1))
            diagnostics["val_auc"] = val_auc
            if val_auc <= DEGENERACY_GUARD_VAL_AUC_FLOOR:
                return (
                    True,
                    f"chance-level validation AUC "
                    f"({val_auc:.3f} <= {DEGENERACY_GUARD_VAL_AUC_FLOOR})",
                    diagnostics,
                )
        return (False, "healthy", diagnostics)

    preds = np.asarray(model.predict(X_val)).ravel()
    n_pred_classes = int(len(np.unique(preds)))
    diagnostics["val_pred_classes"] = n_pred_classes
    if n_pred_classes < 2:
        return (
            True,
            "single-class predictions (model has no predict_proba)",
            diagnostics,
        )
    return (False, "healthy", diagnostics)


def _apply_degeneracy_guard(
    *,
    study: Any,
    results: Dict[str, Any],
    model_class: Any,
    algorithm_name: str,
    default_hyperparameters: Dict[str, Any],
    fixed_params: Dict[str, Any],
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    problem_type: str,
) -> Optional[Dict[str, Any]]:
    """Issue #868: final-fit degeneracy guard with next-best-trial fallback.

    The HPO objective can promise real discrimination (best_value >= 0.58)
    for params whose FINAL fit collapses (e.g. the L1 coefficient-zeroing
    cliff: the conformal wrapper's internal proper-train refit is smaller
    than the objective's fit, and the blended roc_auc metric itself rewards
    an all-positive constant classifier with 0.7*0.5 + 0.3*1.0 = 0.65).

    This guard re-fits the best params final-fit-style (same
    ``_filter_hyperparameters`` merge that ``train_model`` consumes),
    checks for collapse on the validation split, and on genuine collapse
    walks the study's COMPLETE trials by value descending — skipping param
    sets equal to ones already tried — adopting the first non-degenerate
    fit by mutating ``results['best_params'/'best_trial_number'/
    'best_value']`` in place (so the downstream merge, contract output and
    HPO pattern memory all see the adopted trial). If ALL trials are
    degenerate the original best is kept and flagged loudly — the
    downstream deployment gates will block it honestly (fail-closed,
    never fabricate).

    Conservative by construction: arms only for binary classification
    with a real promise (>= DEGENERACY_GUARD_PROMISE_FLOOR) and available
    validation data; a healthy best fit returns with zero change to
    ``results`` (deployed healthy cohorts stay byte-identical in
    behavior).

    Returns:
        Guard metadata dict (always includes ``checked``/``fired``), or
        None when the guard did not arm.
    """
    if problem_type != "binary_classification":
        return None
    best_value = results.get("best_value")
    if best_value is None or not np.isfinite(best_value):
        return None
    if best_value < DEGENERACY_GUARD_PROMISE_FLOOR:
        return None
    if X_val is None or y_val is None or len(y_val) == 0:
        return None

    def _final_style_fit(trial_params: Dict[str, Any]) -> Any:
        # Lazy import: model_trainer_node imports _LR_FIXED_PARAMS from this
        # module at import time, so a module-level import here would be
        # circular. Using the SAME merge + filter as the final fit keeps the
        # verification faithful to what train_model will actually construct.
        from src.agents.ml_foundation.model_trainer.nodes.model_trainer_node import (
            _filter_hyperparameters,
        )

        merged = {**default_hyperparameters, **trial_params, **fixed_params}
        filtered = _filter_hyperparameters(algorithm_name, merged)
        model = model_class(**filtered)
        model.fit(X_train, y_train)
        return model

    best_params = dict(results.get("best_params") or {})
    best_model = _final_style_fit(best_params)
    collapsed, reason, diagnostics = _detect_final_fit_collapse(best_model, X_val, y_val)

    guard_meta: Dict[str, Any] = {
        "checked": True,
        "fired": False,
        "hpo_promise": float(best_value),
        "original_trial": results.get("best_trial_number"),
        "original_params": best_params,
        "original_diagnostics": diagnostics,
    }
    if not collapsed:
        return guard_meta

    guard_meta["fired"] = True
    guard_meta["original_reason"] = reason
    logger.warning(
        "Issue #868 degeneracy guard FIRED for %s: HPO best trial %s "
        "(params=%s) promised %.4f but the final-style fit collapsed (%s). "
        "Falling back to the next-best distinct trial.",
        algorithm_name,
        results.get("best_trial_number"),
        best_params,
        float(best_value),
        reason,
    )

    rejected: List[Dict[str, Any]] = [
        {
            "trial": results.get("best_trial_number"),
            "value": float(best_value),
            "params": best_params,
            "reason": reason,
        }
    ]
    tried_param_sets: List[Dict[str, Any]] = [best_params]

    import optuna

    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(t.value)
    ]
    for trial in sorted(completed, key=lambda t: (-float(t.value), t.number)):
        trial_params = dict(trial.params)
        if any(trial_params == tried for tried in tried_param_sets):
            continue
        tried_param_sets.append(trial_params)
        try:
            candidate_model = _final_style_fit(trial_params)
        except Exception as fit_error:
            rejected.append(
                {
                    "trial": trial.number,
                    "value": float(trial.value),
                    "params": trial_params,
                    "reason": f"refit failed: {fit_error}",
                }
            )
            continue
        cand_collapsed, cand_reason, cand_diagnostics = _detect_final_fit_collapse(
            candidate_model, X_val, y_val
        )
        if cand_collapsed:
            rejected.append(
                {
                    "trial": trial.number,
                    "value": float(trial.value),
                    "params": trial_params,
                    "reason": cand_reason,
                }
            )
            continue

        # Adopt: mutate the results contract in place so the downstream
        # merge, hpo_output and pattern memory all see the adopted trial.
        results["best_params"] = trial_params
        results["best_trial_number"] = trial.number
        results["best_value"] = float(trial.value)
        guard_meta["fallback_adopted"] = True
        guard_meta["adopted_trial"] = trial.number
        guard_meta["adopted_value"] = float(trial.value)
        guard_meta["adopted_params"] = trial_params
        guard_meta["adopted_diagnostics"] = cand_diagnostics
        guard_meta["rejected"] = rejected
        logger.warning(
            "Issue #868 degeneracy guard ADOPTED fallback for %s: trial %s "
            "(params=%s, value=%.4f, diagnostics=%s). Rejected as degenerate: %s.",
            algorithm_name,
            trial.number,
            trial_params,
            float(trial.value),
            cand_diagnostics,
            [(r["trial"], r["reason"]) for r in rejected],
        )
        return guard_meta

    # All trials degenerate: keep the original best, flag loudly. The
    # downstream deployment gates will block it honestly (fail-closed).
    guard_meta["fallback_adopted"] = False
    guard_meta["all_degenerate"] = True
    guard_meta["rejected"] = rejected
    logger.error(
        "Issue #868 degeneracy guard: ALL %d distinct HPO trial param sets for "
        "%s produce degenerate final fits (%s). Keeping the original best "
        "params %s — downstream deployment gates will judge it honestly.",
        len(tried_param_sets),
        algorithm_name,
        [(r["trial"], r["reason"]) for r in rejected],
        best_params,
    )
    return guard_meta


async def tune_hyperparameters(state: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize hyperparameters using Optuna on validation set.

    CRITICAL: Hyperparameter tuning uses VALIDATION set only
    - NEVER use test set for tuning
    - Test set is reserved for final evaluation
    - Validation set is for hyperparameter selection and early stopping

    Args:
        state: ModelTrainerState with enable_hpo, hpo_trials,
               hyperparameter_search_space, validation data

    Returns:
        Dictionary with hpo_completed, best_hyperparameters,
        hpo_best_trial, hpo_trials_run, hpo_duration_seconds

    Raises:
        No exceptions - returns error in state if HPO fails
    """
    # Check if HPO is enabled
    enable_hpo = state.get("enable_hpo", False)
    default_hyperparameters = state.get("default_hyperparameters", {})

    if not enable_hpo:
        # HPO disabled, return default hyperparameters
        logger.info("HPO disabled, using default hyperparameters")
        return {
            "hpo_completed": False,
            "hpo_best_trial": None,
            "best_hyperparameters": default_hyperparameters,
            "hpo_trials_run": 0,
            "hpo_duration_seconds": 0.0,
        }

    # Extract HPO configuration
    hpo_trials = state.get("hpo_trials", 50)
    hpo_timeout_hours = state.get("hpo_timeout_hours")
    hyperparameter_search_space = state.get("hyperparameter_search_space", {})
    algorithm_name = state.get("algorithm_name", "")
    problem_type = state.get("problem_type", "binary_classification")
    experiment_id = state.get("experiment_id", "unknown")

    # Extract preprocessed data
    X_train_preprocessed = state.get("X_train_preprocessed")
    X_validation_preprocessed = state.get("X_validation_preprocessed")
    train_data = state.get("train_data", {})
    validation_data = state.get("validation_data", {})
    y_train = train_data.get("y")
    y_validation = validation_data.get("y")

    # Validate required data
    if X_train_preprocessed is None or y_train is None:
        logger.error("Missing training data for HPO")
        return {
            "error": "Missing training data for HPO",
            "error_type": "missing_hpo_data",
            "hpo_completed": False,
            "best_hyperparameters": default_hyperparameters,
        }

    if X_validation_preprocessed is None or y_validation is None:
        logger.error("Missing validation data for HPO")
        return {
            "error": "Missing validation data for HPO",
            "error_type": "missing_hpo_data",
            "hpo_completed": False,
            "best_hyperparameters": default_hyperparameters,
        }

    if not hyperparameter_search_space:
        # No search space defined, return defaults
        logger.info("No search space defined, using default hyperparameters")
        return {
            "hpo_completed": False,
            "hpo_best_trial": None,
            "best_hyperparameters": default_hyperparameters,
            "hpo_trials_run": 0,
            "hpo_duration_seconds": 0.0,
        }

    # Wrap X as DataFrame with the preprocessor's output feature names.
    # The model_trainer preprocessor (preprocessor.py:152) returns a
    # numpy array. LightGBM 4.x stores `feature_names_in_` as
    # ['Column_0',...,'Column_N'] when fit on a bare numpy array, then
    # sklearn warns on every subsequent numpy predict — leaking ~85
    # UserWarnings per LightGBM HPO. Wrapping with the engineered
    # feature names keeps fit and predict aligned and surfaces the real
    # post-encoding column identities to downstream tools (SHAP, etc.).
    # y_* still goes through _ensure_numpy because Optuna metric
    # helpers expect plain arrays.
    X_train = _wrap_with_feature_names(X_train_preprocessed, state)
    X_val = _wrap_with_feature_names(X_validation_preprocessed, state)
    y_train_np = _ensure_numpy(y_train)
    y_val_np = _ensure_numpy(y_validation)

    # Determine optimization metric based on problem type and imbalance
    imbalance_severity = state.get("imbalance_severity", "none")
    metric = _get_default_metric(problem_type, imbalance_severity=imbalance_severity)

    # Calculate timeout in seconds
    timeout_seconds = int(hpo_timeout_hours * 3600) if hpo_timeout_hours else 3600

    try:
        # Import OptunaOptimizer
        from src.mlops.optuna_optimizer import (
            OptunaOptimizer,
            PrunerFactory,
            get_model_class,
        )

        logger.info(
            f"Starting HPO for {algorithm_name}: {hpo_trials} trials, {timeout_seconds}s timeout"
        )

        # Get model class
        model_class = get_model_class(algorithm_name, problem_type)
        if model_class is None:
            logger.warning(f"Could not get model class for {algorithm_name}")
            return {
                "error": f"Unsupported algorithm for HPO: {algorithm_name}",
                "error_type": "unsupported_algorithm",
                "hpo_completed": False,
                "best_hyperparameters": default_hyperparameters,
            }

        # Create optimizer with MLflow tracking
        optimizer = OptunaOptimizer(
            experiment_id=experiment_id,
            mlflow_tracking=True,
        )

        # Try to get warm-start hyperparameters from procedural memory
        warmstart_config = None
        hpo_memory = _get_hpo_pattern_memory()
        if hpo_memory:
            try:
                n_samples = len(X_train) if X_train is not None else None
                n_features = (
                    X_train.shape[1] if X_train is not None and len(X_train.shape) > 1 else None
                )
                warmstart_config = await hpo_memory.get_warmstart_hyperparameters(
                    algorithm_name=algorithm_name,
                    problem_type=problem_type,
                    n_samples=n_samples,
                    n_features=n_features,
                    metric=metric,
                    min_similarity=0.6,
                )
                if warmstart_config:
                    logger.info(
                        f"Using warm-start from pattern {warmstart_config.pattern_id[:8]} "
                        f"(similarity={warmstart_config.similarity_score:.2f})"
                    )
            except Exception as e:
                logger.debug(f"Warm-start retrieval failed (non-critical): {e}")

        # Create study with median pruner
        study = await optimizer.create_study(
            study_name=f"{algorithm_name}_hpo",
            direction="maximize",
            pruner=PrunerFactory.median_pruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            ),
        )

        # Enqueue warm-start hyperparameters as initial trial if available
        if warmstart_config:
            try:
                # Filter to only hyperparameters in the current search space
                warmstart_params = {
                    k: v
                    for k, v in warmstart_config.initial_hyperparameters.items()
                    if k in hyperparameter_search_space
                }
                if warmstart_params:
                    study.enqueue_trial(warmstart_params)
                    logger.info(f"Enqueued warm-start trial with {len(warmstart_params)} params")
            except Exception as e:
                logger.debug(f"Failed to enqueue warm-start trial: {e}")

        # Get fixed parameters (random_state, n_jobs, class weights, etc.)
        imbalance_detected = state.get("imbalance_detected", False)
        recommended_strategy = state.get("recommended_strategy", "none")
        class_distribution = state.get("class_distribution", {})
        # Backlog #20 Gap 4: opt-in HPO sweep over class_weight /
        # scale_pos_weight / is_unbalance. Default False preserves
        # the deterministic _get_fixed_params pinning.
        hpo_sweep_class_weight = bool(state.get("hpo_sweep_class_weight", False))

        fixed_params = _get_fixed_params(
            algorithm_name,
            imbalance_detected=imbalance_detected,
            recommended_strategy=recommended_strategy,
            class_distribution=class_distribution,
            imbalance_severity=state.get("imbalance_severity", "none"),
            random_state=resolve_fold_random_state(state),
            hpo_sweep_class_weight=hpo_sweep_class_weight,
        )

        # Backlog #20 Gap 4: when sweeping is opted in AND imbalance is
        # detected, inject the imbalance-handling params into the Optuna
        # search space so the matrix's choice gets validated empirically.
        if hpo_sweep_class_weight and imbalance_detected:
            pre_injection_keys = set(hyperparameter_search_space)
            hyperparameter_search_space = _inject_class_weight_sweep(
                hyperparameter_search_space, algorithm_name, class_distribution
            )
            injected_keys = sorted(set(hyperparameter_search_space) - pre_injection_keys)
            logger.info(
                "Backlog #20 Gap 4: HPO sweep over imbalance handlers enabled "
                "for %s (injected %s).",
                algorithm_name,
                injected_keys,
            )

        # Filter search space to exclude params already pinned in fixed_params —
        # otherwise Optuna wastes trial budget sampling values that get overridden
        effective_search_space = {
            k: v for k, v in hyperparameter_search_space.items() if k not in fixed_params
        }

        # Create validation-based objective function
        objective = optimizer.create_validation_objective(
            model_class=model_class,
            X_train=X_train,
            y_train=y_train_np,
            X_val=X_val,
            y_val=y_val_np,
            search_space=effective_search_space,
            problem_type=problem_type,
            metric=metric,
            fixed_params=fixed_params,
        )

        # Run optimization with Opik tracing
        opik = _get_opik_connector()
        hpo_start_time = time.time()

        if opik and opik.is_enabled:
            async with opik.trace_agent(
                agent_name="model_trainer",
                operation="hyperparameter_optimization",
                metadata={
                    "algorithm_name": algorithm_name,
                    "problem_type": problem_type,
                    "n_trials": hpo_trials,
                    "timeout_seconds": timeout_seconds,
                    "metric": metric,
                    "search_space_params": list(hyperparameter_search_space.keys()),
                },
                tags=["hpo", "optuna", algorithm_name.lower()],
            ) as hpo_span:
                results = await optimizer.optimize(
                    study=study,
                    objective=objective,
                    n_trials=hpo_trials,
                    timeout=timeout_seconds,
                )
                # Log HPO metrics to Opik
                hpo_span.set_attribute("best_value", results["best_value"])
                hpo_span.set_attribute("n_trials_completed", results["n_completed"])
                hpo_span.set_attribute("n_trials_pruned", results["n_pruned"])
                hpo_span.set_attribute("duration_seconds", time.time() - hpo_start_time)
        else:
            # No Opik tracing available
            results = await optimizer.optimize(
                study=study,
                objective=objective,
                n_trials=hpo_trials,
                timeout=timeout_seconds,
            )

        # Issue #868: final-fit degeneracy guard. The blended HPO objective
        # can promote params whose final fit collapses (L1 cliff →
        # intercept-only model, test AUC 0.500 while HPO promised 0.65).
        # Verify the best params with a final-style fit and fall back to the
        # next-best DISTINCT trial on genuine collapse. Mutates ``results``
        # in place when a fallback is adopted; non-fatal on any error.
        degeneracy_guard_meta: Optional[Dict[str, Any]] = None
        try:
            degeneracy_guard_meta = _apply_degeneracy_guard(
                study=study,
                results=results,
                model_class=model_class,
                algorithm_name=algorithm_name,
                default_hyperparameters=default_hyperparameters,
                fixed_params=fixed_params,
                X_train=X_train,
                y_train=y_train_np,
                X_val=X_val,
                y_val=y_val_np,
                problem_type=problem_type,
            )
        except Exception as guard_error:
            logger.warning(
                "Issue #868 degeneracy guard errored (non-fatal, keeping HPO best params): %s",
                guard_error,
            )

        # Merge: defaults < optuna best < fixed params (fixed params have highest priority)
        best_hyperparameters = {
            **default_hyperparameters,
            **results["best_params"],
            **fixed_params,
        }

        logger.info(
            f"HPO completed: best_value={results['best_value']:.4f}, "
            f"trials={results['n_trials']} ({results['n_pruned']} pruned)"
        )

        hpo_output = {
            "hpo_completed": True,
            "hpo_best_trial": results["best_trial_number"],
            "best_hyperparameters": best_hyperparameters,
            "hpo_trials_run": results["n_completed"],
            "hpo_trials_pruned": results["n_pruned"],
            "hpo_duration_seconds": results["duration_seconds"],
            "hpo_best_value": results["best_value"],
            "hpo_study_name": results["study_name"],
            "hpo_metric": metric,
        }

        # Issue #868: surface the degeneracy-guard decision (declared in
        # ModelTrainerState so it survives the LangGraph channel boundary).
        if degeneracy_guard_meta is not None:
            hpo_output["hpo_degeneracy_guard"] = degeneracy_guard_meta

        # Validate output against contract
        is_valid, validation_errors = validate_hpo_output(hpo_output)
        if not is_valid:
            logger.warning(f"HPO output validation warnings: {validation_errors}")
            hpo_output["validation_warnings"] = validation_errors

        # Validate hyperparameter types
        if hyperparameter_search_space:
            hp_valid, hp_errors = validate_hyperparameter_types(
                best_hyperparameters, hyperparameter_search_space
            )
            if not hp_valid:
                logger.warning(f"Hyperparameter type warnings: {hp_errors}")
                hpo_output["hyperparameter_warnings"] = hp_errors

        # Store successful HPO pattern for future warm-starting
        if hpo_memory and results["best_value"] is not None:
            try:
                from src.mlops.hpo_pattern_memory import HPOPatternInput

                pattern = HPOPatternInput(
                    algorithm_name=algorithm_name,
                    problem_type=problem_type,
                    search_space=hyperparameter_search_space,
                    best_hyperparameters=results["best_params"],
                    best_value=results["best_value"],
                    optimization_metric=metric,
                    n_trials=results["n_trials"],
                    n_completed=results["n_completed"],
                    n_pruned=results["n_pruned"],
                    duration_seconds=results["duration_seconds"],
                    study_name=results["study_name"],
                    n_samples=len(X_train) if X_train is not None else None,
                    n_features=X_train.shape[1]
                    if X_train is not None and len(X_train.shape) > 1
                    else None,
                    experiment_id=experiment_id,
                )

                pattern_id = await hpo_memory.store_hpo_pattern(pattern)
                if pattern_id:
                    hpo_output["hpo_pattern_id"] = pattern_id
                    logger.info(f"Stored HPO pattern: {pattern_id[:8]}")

                # Record warm-start outcome if we used one
                if warmstart_config:
                    await hpo_memory.record_warmstart_outcome(
                        pattern_id=warmstart_config.pattern_id,
                        new_best_value=results["best_value"],
                        original_best_value=warmstart_config.original_best_value,
                    )
            except Exception as e:
                logger.debug(f"Pattern storage failed (non-critical): {e}")

        return hpo_output

    except ImportError as e:
        logger.warning(f"Optuna not available, using default hyperparameters: {e}")
        return {
            "error": f"Optuna import failed: {e}",
            "error_type": "optuna_not_available",
            "hpo_completed": False,
            "best_hyperparameters": default_hyperparameters,
            "hpo_trials_run": 0,
            "hpo_duration_seconds": 0.0,
        }

    except Exception as e:
        logger.error(f"HPO failed: {e}")
        return {
            "error": f"HPO failed: {e}",
            "error_type": "hpo_error",
            "hpo_completed": False,
            "best_hyperparameters": default_hyperparameters,
            "hpo_trials_run": 0,
            "hpo_duration_seconds": 0.0,
        }


def _ensure_numpy(data: Any) -> np.ndarray:
    """Convert data to numpy array if needed.

    Args:
        data: Input data (numpy array, pandas DataFrame/Series, list)

    Returns:
        Numpy array
    """
    if data is None:
        return np.array([])  # Return empty array instead of None

    if isinstance(data, np.ndarray):
        return data

    # Try pandas conversion
    try:
        import pandas as pd

        if isinstance(data, (pd.DataFrame, pd.Series)):
            return cast(np.ndarray, data.values)
    except ImportError:
        pass

    # Try list/tuple conversion
    if isinstance(data, (list, tuple)):
        return np.array(data)

    # Convert to array
    return np.asarray(data)


def _wrap_with_feature_names(data: Any, state: Dict[str, Any]) -> Any:
    """Return X as a DataFrame with the preprocessor's output feature names.

    If `data` is already a DataFrame it is returned unchanged. If it is a
    2-D numpy array and the state's preprocessor exposes
    `get_feature_names_out()` whose length matches the column count, wrap
    it in a DataFrame so LightGBM/sklearn see consistent feature names
    across fit and predict. Falls through to the original `data` if any
    of those preconditions fail — the worst case is the warning we are
    already trying to fix, never an error.
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


def _get_default_metric(
    problem_type: str,
    imbalance_severity: str = "none",
) -> str:
    """Get default optimization metric for problem type.

    For extreme/severe class imbalance, uses average_precision (PR-AUC)
    instead of roc_auc because AUC is threshold-invariant and can be
    misleadingly high even when the model predicts all negatives.

    Args:
        problem_type: Problem type string
        imbalance_severity: Class imbalance severity (none, moderate, severe, extreme)

    Returns:
        Metric name for optimization
    """
    if problem_type == "binary_classification":
        if imbalance_severity in ("severe", "extreme"):
            return "average_precision"
        return "roc_auc"
    elif problem_type == "multiclass_classification":
        return "f1"
    elif problem_type == "regression":
        return "rmse"
    else:
        return "roc_auc"


def _inject_class_weight_sweep(
    search_space: Dict[str, Dict[str, Any]],
    algorithm_name: str,
    class_distribution: Optional[Dict[int, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Inject the class_weight / scale_pos_weight / is_unbalance sweep.

    Backlog #20 Gap 4: when ``hpo_sweep_class_weight`` is opted-in, the
    deterministic ``_get_fixed_params`` pinning is skipped, and this
    helper adds the imbalance-related parameters to the Optuna search
    space so the matrix's choice can be empirically validated per
    cohort. Returns a NEW dict; does not mutate the input. Skips
    parameters already declared in the input search space — caller's
    explicit definition wins.

    Sweep ranges per algorithm:

    - ``XGBoost``:        ``scale_pos_weight`` ∈ float [0.5×ratio,
      2.0×ratio] where ratio = majority/minority. The deterministic
      pick is exactly ``ratio``; the sweep brackets it 2x in both
      directions so Optuna can validate (or beat) it.
    - ``LightGBM``:       ``is_unbalance`` ∈ categorical {True, False}.
    - ``RandomForest`` /  ``LogisticRegression`` / ``ExtraTrees``:
      ``class_weight`` ∈ categorical {None, "balanced"}.
    - ``GradientBoosting``: NOT swept here — sklearn GBM's imbalance
      handling lives in ``model_trainer_node._prepare_fit_params``
      (Gap 3), as a fit-time ``sample_weight`` array, not a
      constructor param. Sweeping it would require Optuna to vary a
      fit-time kwarg, which the current OptunaOptimizer search-space
      schema doesn't model.

    Args:
        search_space: E2I-format search space (the same dict consumed
            by ``OptunaOptimizer._sample_params``).
        algorithm_name: One of the supported algorithms above. Unknown
            algorithms return the input unmodified.
        class_distribution: Class-count dict for the XGBoost ratio
            calculation. None or single-class skips XGBoost injection.

    Returns:
        A new search_space dict with the relevant imbalance-handling
        parameter(s) added. Returns an unmodified copy when the
        algorithm has no swept imbalance handler.
    """
    out: Dict[str, Dict[str, Any]] = dict(search_space)

    if algorithm_name == "XGBoost":
        if "scale_pos_weight" not in out and class_distribution and len(class_distribution) >= 2:
            counts = list(class_distribution.values())
            minority = min(counts)
            majority = max(counts)
            if minority > 0:
                ratio = majority / minority
                out["scale_pos_weight"] = {
                    "type": "float",
                    "low": 0.5 * ratio,
                    "high": 2.0 * ratio,
                    "log": False,
                }
    elif algorithm_name == "LightGBM":
        if "is_unbalance" not in out:
            out["is_unbalance"] = {
                "type": "categorical",
                "choices": [True, False],
            }
    elif algorithm_name in (
        "RandomForest",
        "LogisticRegression",
        "LogisticRegression_Conformal",
        "ExtraTrees",
    ):
        # Issue #232: include LogisticRegression_Conformal so the conformal LR
        # variant sees the same class_weight sweep as bare LR.
        if "class_weight" not in out:
            out["class_weight"] = {
                "type": "categorical",
                "choices": [None, "balanced"],
            }
    # GradientBoosting intentionally not handled — sample_weight is fit-
    # time, not a model constructor param. See Gap 3 in
    # model_trainer_node._prepare_fit_params.

    return out


def _get_fixed_params(
    algorithm_name: str,
    imbalance_detected: bool = False,
    recommended_strategy: str = "none",
    class_distribution: Optional[Dict[int, int]] = None,
    imbalance_severity: str = "none",
    *,
    random_state: int = 42,
    hpo_sweep_class_weight: bool = False,
) -> Dict[str, Any]:
    """Get fixed parameters for algorithm that shouldn't be tuned.

    Includes class weight handling for imbalanced datasets.

    Args:
        algorithm_name: Algorithm name
        imbalance_detected: Whether class imbalance was detected
        recommended_strategy: Recommended remediation strategy
        class_distribution: Class distribution dict {class: count}
        imbalance_severity: Class imbalance severity (none, moderate, severe, extreme)
        random_state: Per-fold seed for the W3-lite repeated-splits protocol
            (shard 17 Day 3). Defaults to 42 for backward-compat with legacy
            single-split callers; the Day 4-5 orchestrator passes a per-fold
            value resolved via ``resolve_fold_random_state(state)``.
        hpo_sweep_class_weight: Backlog #20 Gap 4 — when True (opt-in via
            ``state["hpo_sweep_class_weight"]``), the deterministic
            class_weight / scale_pos_weight / is_unbalance pinning below
            is SKIPPED so Optuna can sweep these values empirically per
            cohort. Caller must inject the corresponding search-space
            entries via ``_inject_class_weight_sweep`` before HPO. The
            severity-based regularisation caps (max_depth, min_child_*,
            etc.) still apply independently — they're tree-shape
            constraints, not imbalance handlers. Default False
            preserves the legacy fixed-matrix behaviour.

    Returns:
        Dictionary of fixed parameters
    """
    fixed_params: Dict[str, Any] = {}

    if algorithm_name == "XGBoost":
        fixed_params = {
            "random_state": random_state,
            "n_jobs": 1,
            "verbosity": 0,
        }
    elif algorithm_name == "LightGBM":
        fixed_params = {
            "random_state": random_state,
            "n_jobs": 1,
            "verbose": -1,
        }
    elif algorithm_name == "RandomForest":
        fixed_params = {
            "random_state": random_state,
            "n_jobs": 1,
        }
    elif algorithm_name in ("LogisticRegression", "LogisticRegression_Conformal"):
        # Issue #232: route BOTH LR and LR_Conformal through the shared
        # ``_LR_FIXED_PARAMS`` helper so penalty=l1 HPO trials don't crash
        # with ``Solver lbfgs supports only 'l2' or None penalties``.
        fixed_params = {
            "random_state": random_state,
            **_LR_FIXED_PARAMS,
        }
    elif algorithm_name in ["Ridge", "Lasso"]:
        fixed_params = {
            "random_state": random_state,
        }
    elif algorithm_name == "CausalForest":
        fixed_params = {
            "n_jobs": 1,
            "inference": True,
            "random_state": random_state,
        }
    elif algorithm_name == "LinearDML":
        fixed_params = {
            "cv": 3,
            "mc_iters": 3,
            "random_state": random_state,
        }
    elif algorithm_name == "DRLearner":
        fixed_params = {
            "cv": 3,
            "random_state": random_state,
        }
    elif algorithm_name in ["SLearner", "TLearner", "XLearner"]:
        fixed_params = {
            "cv": 3,
            "random_state": random_state,
        }

    # Add class weight handling for imbalanced datasets
    # CRITICAL: Always apply when imbalance detected, regardless of strategy
    # Even with SMOTE, class_weight provides defense-in-depth.
    # Backlog #20 Gap 4: when hpo_sweep_class_weight=True, SKIP this block
    # so Optuna can sweep the values via the injected search space.
    if imbalance_detected and not hpo_sweep_class_weight:
        if class_distribution and len(class_distribution) >= 2:
            counts = list(class_distribution.values())
            minority_count = min(counts)
            majority_count = max(counts)

            if minority_count > 0:
                if algorithm_name == "XGBoost":
                    # XGBoost uses scale_pos_weight
                    fixed_params["scale_pos_weight"] = majority_count / minority_count
                    logger.info(
                        f"Added scale_pos_weight={fixed_params['scale_pos_weight']:.2f} for XGBoost"
                    )
                elif algorithm_name in (
                    "RandomForest",
                    "LogisticRegression",
                    "LogisticRegression_Conformal",
                    "ExtraTrees",
                ):
                    # sklearn models use class_weight. Issue #232: include
                    # LogisticRegression_Conformal so the conformal LR variant
                    # also gets balanced class_weight under imbalance.
                    fixed_params["class_weight"] = "balanced"
                    logger.info(f"Added class_weight='balanced' for {algorithm_name}")
                elif algorithm_name == "LightGBM":
                    # LightGBM uses is_unbalance
                    fixed_params["is_unbalance"] = True
                    logger.info("Added is_unbalance=True for LightGBM")
                elif algorithm_name == "GradientBoosting":
                    # sklearn GradientBoosting doesn't accept class_weight in the
                    # constructor, but it DOES accept sample_weight in .fit().
                    # Backlog #20 Gap 3: the symmetric class-weight handling for
                    # GBM lives in
                    # ``model_trainer_node._prepare_fit_params`` — when
                    # ``state["imbalance_detected"]`` is True, that helper
                    # populates ``fit_params["sample_weight"]`` via
                    # ``sklearn.utils.class_weight.compute_sample_weight(
                    # "balanced", y_train)``, which is the per-sample
                    # equivalent of ``class_weight="balanced"`` on the other
                    # tree models.
                    pass

    # Regularization caps for imbalanced datasets (mutually exclusive tiers).
    # Severe (5-20% minority) gets moderate depth/leaf caps.
    # Extreme (<5% minority) gets aggressive caps + subsampling.
    if imbalance_severity == "severe":
        if algorithm_name == "XGBoost":
            fixed_params["max_depth"] = min(fixed_params.get("max_depth", 6), 6)
            fixed_params["min_child_weight"] = max(fixed_params.get("min_child_weight", 1), 5)
        elif algorithm_name == "LightGBM":
            fixed_params["max_depth"] = min(fixed_params.get("max_depth", 6), 6)
            fixed_params["num_leaves"] = min(fixed_params.get("num_leaves", 31), 31)
            fixed_params["min_child_samples"] = max(fixed_params.get("min_child_samples", 20), 20)
        elif algorithm_name == "RandomForest":
            fixed_params["max_depth"] = min(fixed_params.get("max_depth", 8), 8)
            fixed_params["min_samples_leaf"] = max(fixed_params.get("min_samples_leaf", 2), 5)

    elif imbalance_severity == "extreme":
        if algorithm_name == "XGBoost":
            fixed_params["max_depth"] = min(fixed_params.get("max_depth", 4), 4)
            fixed_params["min_child_weight"] = max(fixed_params.get("min_child_weight", 1), 10)
            fixed_params["gamma"] = max(fixed_params.get("gamma", 0), 1.0)
            fixed_params["subsample"] = 0.7
            fixed_params["colsample_bytree"] = 0.7
        elif algorithm_name == "LightGBM":
            fixed_params["max_depth"] = min(fixed_params.get("max_depth", 4), 4)
            fixed_params["num_leaves"] = min(fixed_params.get("num_leaves", 15), 15)
            fixed_params["min_child_samples"] = max(fixed_params.get("min_child_samples", 20), 20)
            fixed_params["subsample"] = 0.7
            fixed_params["subsample_freq"] = 1  # Required or LightGBM silently ignores subsample
            fixed_params["colsample_bytree"] = 0.7
        elif algorithm_name == "RandomForest":
            fixed_params["max_depth"] = min(fixed_params.get("max_depth", 6), 6)
            fixed_params["min_samples_leaf"] = max(fixed_params.get("min_samples_leaf", 2), 10)
            fixed_params["min_samples_split"] = max(fixed_params.get("min_samples_split", 2), 20)
            fixed_params["max_features"] = "sqrt"

    return fixed_params
