"""Algorithm registry and filtering logic for model_selector.

This module defines the catalog of supported algorithms and filtering logic.

Issue #273: the LogisticRegression entry's ``default_hyperparameters`` contains
``solver`` and ``max_iter`` values that MUST stay in sync with the SSOT
``_LR_FIXED_PARAMS`` constant in
``src/agents/ml_foundation/model_trainer/nodes/hyperparameter_tuner.py`` (the
three production-consumption sites: HPO dispatcher, tier-0 alt-train builder,
``model_trainer_node._filter_hyperparameters``). A direct module-level import
of ``_LR_FIXED_PARAMS`` here would create a circular import via
``model_trainer/nodes/__init__.py`` → ``quality_remediation`` →
``algorithm_registry`` (back-edge). The contract is therefore enforced by the
runtime tests in
``tests/unit/test_agents/test_ml_foundation/test_model_trainer/test_lr_saga_fixed_params.py::TestIssue273RegistryDefaultsAgreeWithSSOT``
— if either ``solver`` or ``max_iter`` here diverges from SSOT, the tests trip.
"""

from typing import Any, Dict, List

# Algorithm catalog with specifications
ALGORITHM_REGISTRY = {
    # === CAUSAL ML (DoWhy/EconML) ===
    "CausalForest": {
        "family": "causal_ml",
        "framework": "econml",
        "problem_types": ["binary_classification", "regression"],
        "strengths": ["heterogeneous_effects", "interpretability", "causal_inference"],
        "inference_latency_ms": 50,
        "memory_gb": 4.0,
        "interpretability_score": 0.7,
        "scalability_score": 0.8,
        "hyperparameter_space": {
            "n_estimators": {"type": "int", "low": 100, "high": 1000, "step": 100},
            "max_depth": {"type": "int", "low": 5, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 5, "high": 50, "step": 5},
        },
        "default_hyperparameters": {
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_leaf": 10,
        },
    },
    "LinearDML": {
        "family": "causal_ml",
        "framework": "econml",
        "problem_types": ["binary_classification", "regression"],
        "strengths": ["fast", "interpretable", "low_variance", "causal_inference"],
        "inference_latency_ms": 10,
        "memory_gb": 1.0,
        "interpretability_score": 0.9,
        "scalability_score": 0.9,
        # EconML model types: linear, poly, forest, gbf, nnet, automl
        "hyperparameter_space": {
            "model_y": {"type": "categorical", "choices": ["linear", "forest", "gbf"]},
            "model_t": {"type": "categorical", "choices": ["linear", "forest"]},
            "cv": {"type": "int", "low": 3, "high": 5},
        },
        "default_hyperparameters": {
            "model_y": "linear",  # EconML model type string (not sklearn class)
            "model_t": "linear",  # EconML model type string (not sklearn class)
            "cv": 5,
        },
    },
    # === GRADIENT BOOSTING ===
    "XGBoost": {
        "family": "gradient_boosting",
        "framework": "xgboost",
        "problem_types": ["binary_classification", "multiclass_classification", "regression"],
        "strengths": ["accuracy", "feature_importance", "robust"],
        "inference_latency_ms": 20,
        "memory_gb": 2.0,
        "interpretability_score": 0.6,
        "scalability_score": 0.8,
        "hyperparameter_space": {
            "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
            # max_depth ceiling lowered 10 -> 8 to bound tree complexity on small
            # / low-prevalence data.
            "max_depth": {"type": "int", "low": 3, "high": 8},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            # Regularization in the BASE space (see LightGBM note): feature
            # subsampling + min-child-weight + L2 let the FIRST HPO pass avoid the
            # overfit seen on the optum-mart discontinuation cohort. Low bounds let
            # well-separated data still choose minimal regularization.
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "min_child_weight": {"type": "int", "low": 1, "high": 50},
            "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        },
        "default_hyperparameters": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
        },
    },
    "LightGBM": {
        "family": "gradient_boosting",
        "framework": "lightgbm",
        "problem_types": ["binary_classification", "multiclass_classification", "regression"],
        "strengths": ["speed", "large_datasets", "memory_efficient"],
        "inference_latency_ms": 15,
        "memory_gb": 1.5,
        "interpretability_score": 0.6,
        "scalability_score": 0.95,
        "hyperparameter_space": {
            "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            # num_leaves floor lowered 20 -> 8 so the optimizer can pick shallow
            # trees (the primary overfit control for LightGBM); ceiling lowered
            # 150 -> 64 to keep complexity bounded on small / low-prevalence data.
            "num_leaves": {"type": "int", "low": 8, "high": 64},
            # Regularization in the BASE space (not only the reactive
            # REGULARIZATION_SEARCH_SPACE, which only merges after a separate
            # remediation loop fires): min_child_samples + L2 let the FIRST HPO
            # pass avoid the train-val AUC delta ~0.14 overfit seen on the
            # optum-mart discontinuation cohort. Low bounds let well-separated
            # data still choose minimal regularization.
            "min_child_samples": {"type": "int", "low": 20, "high": 300},
            "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        },
        "default_hyperparameters": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_child_samples": 50,
            "reg_lambda": 1.0,
        },
    },
    # === RANDOM FOREST ===
    "RandomForest": {
        "family": "ensemble",
        "framework": "sklearn",
        "problem_types": ["binary_classification", "multiclass_classification", "regression"],
        "strengths": ["robust", "feature_importance", "no_scaling_required"],
        "inference_latency_ms": 30,
        "memory_gb": 3.0,
        "interpretability_score": 0.5,
        "scalability_score": 0.7,
        "hyperparameter_space": {
            "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
            "max_depth": {"type": "int", "low": 5, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        },
        "default_hyperparameters": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
    },
    # === LINEAR MODELS (Baselines) ===
    "LogisticRegression": {
        "family": "linear",
        "framework": "sklearn",
        "problem_types": ["binary_classification", "multiclass_classification"],
        "strengths": ["interpretable", "fast", "baseline", "stable"],
        "inference_latency_ms": 1,
        "memory_gb": 0.1,
        "interpretability_score": 1.0,
        "scalability_score": 1.0,
        "hyperparameter_space": {
            "C": {"type": "float", "low": 0.001, "high": 100, "log": True},
            "penalty": {"type": "categorical", "choices": ["l1", "l2"]},
        },
        "default_hyperparameters": {
            "C": 1.0,
            "penalty": "l2",
            # Issue #273: these two values mirror
            # ``_LR_FIXED_PARAMS`` in
            # ``src/agents/ml_foundation/model_trainer/nodes/hyperparameter_tuner.py``.
            # A direct import would create a circular dependency via
            # ``model_trainer/nodes/__init__.py`` → ``quality_remediation`` →
            # ``algorithm_registry``. The SSOT contract is enforced at test
            # time by
            # ``test_lr_saga_fixed_params.py::TestIssue273RegistryDefaultsAgreeWithSSOT``
            # — diverging from SSOT here makes that test trip.
            # ``solver=saga`` is load-bearing: per PR #261 / #232, HPO trials
            # with ``penalty="l1"`` crash under ``solver=lbfgs`` with
            # ``Solver lbfgs supports only 'l2' or None penalties``.
            "solver": "saga",
            "max_iter": 1000,
        },
    },
    "Ridge": {
        "family": "linear",
        "framework": "sklearn",
        "problem_types": ["regression"],
        "strengths": ["interpretable", "fast", "baseline", "stable"],
        "inference_latency_ms": 1,
        "memory_gb": 0.1,
        "interpretability_score": 1.0,
        "scalability_score": 1.0,
        "hyperparameter_space": {
            "alpha": {"type": "float", "low": 0.001, "high": 100, "log": True},
        },
        "default_hyperparameters": {
            "alpha": 1.0,
        },
    },
    "Lasso": {
        "family": "linear",
        "framework": "sklearn",
        "problem_types": ["regression"],
        "strengths": ["interpretable", "fast", "feature_selection", "baseline"],
        "inference_latency_ms": 1,
        "memory_gb": 0.1,
        "interpretability_score": 1.0,
        "scalability_score": 1.0,
        "hyperparameter_space": {
            "alpha": {"type": "float", "low": 0.001, "high": 10, "log": True},
        },
        "default_hyperparameters": {
            "alpha": 1.0,
        },
    },
    # === CALIBRATION-NATIVE (NGBoost) ===
    # Phase 1 W2 day-1 (adaptive_criteria_v3_followup shard 19 §A.2). The two
    # new flags `distribution_predictor` and `skip_post_hoc_calibration` are
    # consumed by evaluator gating in W2 day-2; absent on legacy entries means
    # legacy behavior is unchanged.
    "NGBoost": {
        "family": "calibration_native",
        "framework": "ngboost",
        "problem_types": ["binary_classification", "regression"],
        "strengths": ["calibration", "uncertainty", "tabular_medical"],
        "inference_latency_ms": 40,
        "memory_gb": 2.0,
        "interpretability_score": 0.5,
        "scalability_score": 0.7,
        "distribution_predictor": True,
        "skip_post_hoc_calibration": True,
        "hyperparameter_space": {
            "n_estimators": {"type": "int", "low": 100, "high": 800, "step": 100},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.1, "log": True},
            "minibatch_frac": {"type": "float", "low": 0.5, "high": 1.0, "step": 0.1},
            "col_sample": {"type": "float", "low": 0.5, "high": 1.0, "step": 0.1},
            "base_max_depth": {"type": "int", "low": 2, "high": 6},
            "base_min_samples_leaf": {"type": "int", "low": 5, "high": 50, "step": 5},
        },
        "default_hyperparameters": {
            "n_estimators": 500,
            "learning_rate": 0.01,
            "minibatch_frac": 1.0,
            "col_sample": 1.0,
            "base_max_depth": 3,
            "base_min_samples_leaf": 5,
        },
    },
    # === MAPIE CONFORMAL VARIANTS ===
    # Phase 1 W2 day-3 (shard 19 §B.3). Each entry pairs a base estimator with
    # MAPIE cross-conformal calibration (cv-folded, alpha=0.10 → 90% marginal
    # coverage). The factory at optuna_optimizer.get_model_class strips the
    # `_Conformal` suffix, recurses to fetch the base class, and returns a
    # closure that wraps a fitted base in MapieConformalBinaryClassifier.
    # Per amendment 3 (see mapie_wrapper.py docstring), MAPIE 0.8.6 does NOT
    # expose predict_proba on MapieClassifier — the wrapper delegates predict_proba
    # to the base estimator and surfaces conformal sets via predict_sets().
    # `skip_post_hoc_calibration=True` retained verbatim per shard 19 §B.3;
    # revisit at W4 multi-disease if non-calibration-native bases (LightGBM,
    # LogisticRegression) underperform on calibration metrics.
    # Cycle-9 codex F1 amendment: `cv` REMOVED from conformal search spaces +
    # defaults. shard 19 §B.4 specified cv=3-5 implying MAPIE internal
    # cross-conformal splits, but the wrapper's amendment 2 hardcodes
    # cv="prefit" (MAPIE 0.8.6 semantics) — the cv int is silently ignored.
    # Tuning a dead knob wastes Optuna trials. To re-enable real cv-conformal
    # in a future commit: split (X, y) inside wrapper.fit() into base-train +
    # MAPIE-calibrate folds and switch wrapper to cv=self.cv (would also cure
    # the train-set-conformal honesty issue flagged by cycle-9 D1).
    "NGBoost_Conformal": {
        "family": "calibration_native_conformal",
        "framework": "mapie+ngboost",
        "problem_types": ["binary_classification"],
        "strengths": ["calibration", "uncertainty", "coverage_guarantee"],
        "inference_latency_ms": 80,
        "memory_gb": 2.5,
        "interpretability_score": 0.4,
        "scalability_score": 0.6,
        "distribution_predictor": True,
        "skip_post_hoc_calibration": True,
        "conformal_wrapper": True,
        "base_estimator": "NGBoost",
        "hyperparameter_space": {
            "n_estimators": {"type": "int", "low": 200, "high": 600, "step": 100},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.05, "log": True},
        },
        "default_hyperparameters": {
            "n_estimators": 400,
            "learning_rate": 0.01,
        },
    },
    "LightGBM_Conformal": {
        "family": "gradient_boosting_conformal",
        "framework": "mapie+lightgbm",
        "problem_types": ["binary_classification"],
        "strengths": ["calibration", "speed", "coverage_guarantee"],
        "inference_latency_ms": 60,
        "memory_gb": 2.0,
        "interpretability_score": 0.5,
        "scalability_score": 0.85,
        "skip_post_hoc_calibration": True,
        "conformal_wrapper": True,
        "base_estimator": "LightGBM",
        "hyperparameter_space": {
            "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 100},
            "max_depth": {"type": "int", "low": 3, "high": 8},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
        },
        "default_hyperparameters": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
        },
    },
    "LogisticRegression_Conformal": {
        "family": "linear_conformal",
        "framework": "mapie+sklearn",
        "problem_types": ["binary_classification"],
        "strengths": ["interpretable", "coverage_guarantee", "stable_baseline"],
        "inference_latency_ms": 5,
        "memory_gb": 0.2,
        "interpretability_score": 0.95,
        "scalability_score": 1.0,
        "skip_post_hoc_calibration": True,
        "conformal_wrapper": True,
        "base_estimator": "LogisticRegression",
        "hyperparameter_space": {
            "C": {"type": "float", "low": 0.001, "high": 100, "log": True},
            "penalty": {"type": "categorical", "choices": ["l1", "l2"]},
        },
        "default_hyperparameters": {
            "C": 1.0,
            "penalty": "l2",
        },
    },
    # === MONOTONE-CONSTRAINED VARIANTS ===
    # Phase 1 W2 day-4 (shard 19 §C.3). NO new model class — these are
    # registry-level variants of LightGBM/XGBoost that opt into the
    # `monotone_constraints_required` flag. The trainer reads
    # state["monotone_vector"] and injects monotone_constraints into
    # filtered_params at fit time (shard 19 §C.4). The base LightGBM /
    # XGBoost entries stay unchanged so the no-monotone path is bit-identical.
    "LightGBM_Monotone": {
        "family": "gradient_boosting_constrained",
        "framework": "lightgbm",
        "problem_types": ["binary_classification", "regression"],
        "strengths": ["calibration", "clinical_knowledge_injection", "speed"],
        "inference_latency_ms": 18,
        "memory_gb": 1.6,
        "interpretability_score": 0.65,
        "scalability_score": 0.95,
        "monotone_constraints_required": True,
        "base_estimator": "LightGBM",
        "hyperparameter_space": {
            "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
            "max_depth": {"type": "int", "low": 3, "high": 8},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
            "num_leaves": {"type": "int", "low": 15, "high": 63},
            # monotone_constraints is NOT searched — it's a configured input.
        },
        "default_hyperparameters": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            # monotone_constraints injected at fit time from state["monotone_vector"].
        },
    },
    "XGBoost_Monotone": {
        "family": "gradient_boosting_constrained",
        "framework": "xgboost",
        "problem_types": ["binary_classification", "regression"],
        "strengths": ["calibration", "clinical_knowledge_injection", "accuracy"],
        "inference_latency_ms": 22,
        "memory_gb": 2.0,
        "interpretability_score": 0.65,
        "scalability_score": 0.85,
        "monotone_constraints_required": True,
        "base_estimator": "XGBoost",
        "hyperparameter_space": {
            "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
            "max_depth": {"type": "int", "low": 3, "high": 8},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
        },
        "default_hyperparameters": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
        },
    },
}

# Regularization-focused search spaces for quality remediation.
# When a model shows overfitting or poor generalization, these params are
# merged into the HPO search space to encourage stronger regularization.
REGULARIZATION_SEARCH_SPACE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "XGBoost": {
        "reg_alpha": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
        "gamma": {"type": "float", "low": 0.0, "high": 5.0},
        "min_child_weight": {"type": "int", "low": 3, "high": 20},
    },
    "LightGBM": {
        "reg_alpha": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
        "min_split_gain": {"type": "float", "low": 0.0, "high": 1.0},
        "min_child_samples": {"type": "int", "low": 10, "high": 100},
    },
    "RandomForest": {
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]},
        "min_samples_leaf": {"type": "int", "low": 5, "high": 50},
    },
    "LogisticRegression": {
        "C": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
    },
}


async def filter_algorithms(state: Dict[str, Any]) -> Dict[str, Any]:
    """Filter algorithms by problem type, constraints, and preferences.

    Applies progressive filtering:
    1. Filter by problem type
    2. Filter by technical constraints (latency, memory)
    3. Filter by user preferences
    4. Filter by interpretability requirement

    Args:
        state: ModelSelectorState with problem_type, technical_constraints,
               algorithm_preferences, excluded_algorithms, interpretability_required

    Returns:
        Dictionary with filtered candidate lists at each stage
    """
    problem_type = state.get("problem_type", "binary_classification")
    technical_constraints = state.get("technical_constraints", [])
    algorithm_preferences = state.get("algorithm_preferences", [])
    excluded_algorithms = state.get("excluded_algorithms", [])
    interpretability_required = state.get("interpretability_required", False)

    # Step 1: Filter by problem type
    filtered_by_type = _filter_by_problem_type(problem_type)

    # Step 2: Filter by technical constraints
    filtered_by_constraints = _filter_by_constraints(filtered_by_type, technical_constraints)

    # Step 3: Filter by user preferences
    filtered_by_preferences = _filter_by_preferences(
        filtered_by_constraints, algorithm_preferences, excluded_algorithms
    )

    # Step 4: Filter by interpretability requirement
    if interpretability_required:
        filtered_by_preferences = [
            algo for algo in filtered_by_preferences if algo.get("interpretability_score", 0) >= 0.7
        ]

    # Validation: Ensure at least one candidate remains
    if not filtered_by_preferences:
        # Fallback to linear models as baseline
        filtered_by_preferences = [
            {**ALGORITHM_REGISTRY[name], "name": name}
            for name in ["LogisticRegression", "Ridge", "Lasso"]
            if problem_type in ALGORITHM_REGISTRY[name]["problem_types"]  # type: ignore[operator]
        ]

    return {
        "filtered_by_problem_type": filtered_by_type,
        "filtered_by_constraints": filtered_by_constraints,
        "filtered_by_preferences": filtered_by_preferences,
        "candidate_algorithms": filtered_by_preferences,
    }


def _filter_by_problem_type(problem_type: str) -> List[Dict[str, Any]]:
    """Filter algorithms that support the problem type."""
    candidates = []
    for name, spec in ALGORITHM_REGISTRY.items():
        if problem_type in spec["problem_types"]:  # type: ignore[operator]
            candidates.append({**spec, "name": name})
    return candidates


def _filter_by_constraints(
    candidates: List[Dict[str, Any]], constraints: List[str]
) -> List[Dict[str, Any]]:
    """Filter algorithms by technical constraints."""
    filtered = candidates

    for constraint in constraints:
        constraint_lower = constraint.lower()

        # Parse latency constraint: "inference_latency_<100ms"
        if "latency" in constraint_lower and "<" in constraint_lower:
            try:
                max_latency = int(constraint_lower.split("<")[1].replace("ms", "").strip())
                filtered = [c for c in filtered if c["inference_latency_ms"] <= max_latency]
            except (ValueError, IndexError):
                pass  # Skip malformed constraint

        # Parse memory constraint: "memory_<8gb"
        if "memory" in constraint_lower and "<" in constraint_lower:
            try:
                max_memory = float(constraint_lower.split("<")[1].replace("gb", "").strip())
                filtered = [c for c in filtered if c["memory_gb"] <= max_memory]
            except (ValueError, IndexError):
                pass  # Skip malformed constraint

    return filtered


def _filter_by_preferences(
    candidates: List[Dict[str, Any]],
    preferences: List[str],
    excluded: List[str],
) -> List[Dict[str, Any]]:
    """Filter algorithms by user preferences and exclusions."""
    # Remove excluded algorithms
    if excluded:
        candidates = [c for c in candidates if c["name"] not in excluded]

    # If preferences specified, prioritize (but don't hard filter)
    # Preferences will be used in ranking instead
    return candidates
