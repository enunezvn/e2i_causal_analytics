"""Quality remediation node for model_trainer agent.

When post-evaluation metrics indicate poor model quality (overfitting,
low precision/recall), this node enhances the HPO search space with
regularization-focused parameters and loops back through HPO/train/eval.

Modeled on data_preparer/nodes/leakage_remediation.py — deterministic
strategy in MVP, LLM-assisted diagnosis is a future enhancement.
"""

from typing import Any, Dict

from src.agents.ml_foundation.model_selector.nodes.algorithm_registry import (
    REGULARIZATION_SEARCH_SPACE,
)


async def diagnose_and_remediate_quality(state: Dict[str, Any]) -> Dict[str, Any]:
    """Diagnose poor model quality and enhance search space for retry.

    Strategy (deterministic MVP):
    1. Guard: bail if max attempts reached
    2. Assess: compute overfitting delta, check model usefulness metrics
    3. If overfitting detected and regularization params available -> merge them
    4. If severe/extreme imbalance and algo supports class_weight -> force it
    5. Return updated search space + incremented attempt counter

    The graph routes back to tune_hyperparameters after this node returns
    status="enhancing", creating the inner remediation loop.
    """
    attempts = state.get("quality_remediation_attempts", 0)
    max_attempts = state.get("quality_remediation_max_attempts", 2)
    history = list(state.get("quality_remediation_history", []))

    # Guard: max attempts reached
    if attempts >= max_attempts:
        return {
            "quality_remediation_status": "max_attempts",
            "quality_remediation_attempts": attempts,
            "quality_remediation_history": history,
        }

    # Assess current model quality
    train_metrics = state.get("train_metrics", {})
    test_metrics = state.get("test_metrics", {})
    val_metrics = state.get("validation_metrics", {})

    train_auc = train_metrics.get("roc_auc", 0)
    val_auc = val_metrics.get("roc_auc", 0)
    test_auc = test_metrics.get("roc_auc", 0)
    test_precision = test_metrics.get("precision", 0)

    overfitting_delta = train_auc - val_auc if train_auc and val_auc else 0.0

    algo_name = state.get("algorithm_name", "")
    current_search_space = dict(state.get("hyperparameter_search_space", {}))
    default_hyperparameters = dict(state.get("default_hyperparameters", {}))

    # Strategy 0: Check if precision problem is threshold-fixable
    # If the model discriminates well (AUC >= 0.65) but precision is terrible (<5%),
    # check if the evaluator's precision-constrained threshold already fixes it.
    # No retraining needed — just use the right threshold.
    precision_constrained = state.get("precision_constrained", {})
    if precision_constrained and test_precision < 0.05 and (test_auc >= 0.65 or val_auc >= 0.65):
        pc_precision = precision_constrained.get("precision_at_threshold", 0)
        pc_recall = precision_constrained.get("recall_at_threshold", 0)
        if pc_precision >= 0.05 and pc_recall >= 0.10:
            # Precision IS fixable by threshold alone — no retraining needed
            print("\n  Quality remediation: THRESHOLD FIXABLE (no retraining needed)")
            print(f"    Precision at default: {test_precision:.4f}")
            print(f"    Precision at constrained threshold: {pc_precision:.4f}")
            print(f"    Recall at constrained threshold: {pc_recall:.4f}")
            return {
                "quality_remediation_status": "threshold_fixable",
                "quality_remediation_attempts": attempts + 1,
                "quality_remediation_history": history + [{
                    "attempt": attempts + 1,
                    "strategy": "threshold_optimization",
                    "reason": "Precision fixable by threshold alone, no retraining needed",
                    "pre_remediation_metrics": {
                        "test_auc": test_auc,
                        "test_precision": test_precision,
                    },
                    "post_threshold_metrics": {
                        "precision": pc_precision,
                        "recall": pc_recall,
                    },
                    "improved": True,
                }],
                "optimal_threshold": precision_constrained.get("precision_constrained_threshold"),
            }

    strategy = "none"
    changes_made = False

    # Strategy 1: Enhance regularization for overfitting
    if overfitting_delta > 0.10 and algo_name in REGULARIZATION_SEARCH_SPACE:
        reg_params = REGULARIZATION_SEARCH_SPACE[algo_name]
        current_search_space.update(reg_params)
        strategy = "enhance_regularization"
        changes_made = True

    # Strategy 2: Force class_weight for imbalanced datasets
    imbalance_severity = state.get("imbalance_severity", "none")
    if imbalance_severity in ("severe", "extreme"):
        if algo_name == "RandomForest":
            default_hyperparameters["class_weight"] = "balanced"
            strategy = f"{strategy}+force_class_weight" if strategy != "none" else "force_class_weight"
            changes_made = True
        elif algo_name == "LogisticRegression":
            default_hyperparameters["class_weight"] = "balanced"
            strategy = f"{strategy}+force_class_weight" if strategy != "none" else "force_class_weight"
            changes_made = True

    # Strategy 3: If no overfitting but poor metrics, still try regularization
    # (the model may be underfitting on noisy features)
    if not changes_made and algo_name in REGULARIZATION_SEARCH_SPACE:
        reg_params = REGULARIZATION_SEARCH_SPACE[algo_name]
        current_search_space.update(reg_params)
        strategy = "regularization_fallback"
        changes_made = True

    # If nothing can be changed, mark as failed
    if not changes_made:
        return {
            "quality_remediation_status": "failed",
            "quality_remediation_attempts": attempts + 1,
            "quality_remediation_history": history + [{
                "attempt": attempts + 1,
                "strategy": "none",
                "reason": f"No regularization params available for {algo_name}",
                "improved": False,
            }],
        }

    # Record this attempt
    history.append({
        "attempt": attempts + 1,
        "strategy": strategy,
        "overfitting_delta": overfitting_delta,
        "pre_remediation_metrics": {
            "train_auc": train_auc,
            "val_auc": val_auc,
            "test_auc": test_auc,
            "test_precision": test_precision,
        },
        "params_added": list(
            set(current_search_space.keys()) - set(state.get("hyperparameter_search_space", {}).keys())
        ),
    })

    print(f"\n  Quality remediation (attempt {attempts + 1}/{max_attempts}): {strategy}")
    print(f"    Overfitting delta: {overfitting_delta:.3f}")
    print(f"    Search space expanded: {len(state.get('hyperparameter_search_space', {}))} -> {len(current_search_space)} params")

    return {
        "quality_remediation_status": "enhancing",
        "quality_remediation_attempts": attempts + 1,
        "quality_remediation_history": history,
        "hyperparameter_search_space": current_search_space,
        "default_hyperparameters": default_hyperparameters,
        "enhanced_search_space": current_search_space,
    }
