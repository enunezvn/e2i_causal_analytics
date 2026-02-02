"""Class imbalance detection for model_trainer.

This module detects class imbalance in training data and uses Claude
to recommend optimal remediation strategies.

Version: 1.0.0
"""

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

# Severity thresholds based on minority class ratio
SEVERITY_THRESHOLDS = {
    "none": 0.40,  # Minority >= 40% - no action needed
    "moderate": 0.20,  # Minority 20-40% - consider weighting
    "severe": 0.05,  # Minority 5-20% - resampling recommended
    "extreme": 0.0,  # Minority < 5% - aggressive resampling + weighting
}

# Valid remediation strategies
VALID_STRATEGIES = [
    "smote",  # Synthetic minority oversampling
    "random_oversample",  # Duplicate minority samples
    "random_undersample",  # Remove majority samples
    "smote_tomek",  # SMOTE + Tomek links cleaning
    "class_weight",  # Use class weights only (no resampling)
    "combined",  # Moderate resampling + class weights
    "none",  # No action needed
]


def _calculate_imbalance_metrics(y: np.ndarray) -> Dict[str, Any]:
    """Calculate class imbalance metrics.

    Args:
        y: Target labels

    Returns:
        Dictionary with imbalance metrics
    """
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique.astype(int).tolist(), counts.tolist(), strict=False))

    total = len(y)
    minority_count = min(counts)
    majority_count = max(counts)
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]

    minority_ratio = minority_count / total
    imbalance_ratio = majority_count / minority_count if minority_count > 0 else float("inf")

    # Determine severity
    if minority_ratio >= SEVERITY_THRESHOLDS["none"]:
        severity = "none"
    elif minority_ratio >= SEVERITY_THRESHOLDS["moderate"]:
        severity = "moderate"
    elif minority_ratio >= SEVERITY_THRESHOLDS["severe"]:
        severity = "severe"
    else:
        severity = "extreme"

    return {
        "class_distribution": class_distribution,
        "minority_count": int(minority_count),
        "majority_count": int(majority_count),
        "minority_class": int(minority_class),
        "majority_class": int(majority_class),
        "minority_ratio": float(minority_ratio),
        "imbalance_ratio": float(imbalance_ratio),
        "severity": severity,
        "total_samples": total,
    }


async def _get_llm_recommendation(
    metrics: Dict[str, Any],
    algorithm_name: str,
    problem_type: str,
) -> tuple[str, str]:
    """Get LLM recommendation for remediation strategy.

    Args:
        metrics: Imbalance metrics
        algorithm_name: Algorithm being trained
        problem_type: Problem type

    Returns:
        Tuple of (strategy, rationale)
    """
    try:
        import anthropic

        client = anthropic.Anthropic()

        prompt = f"""You are an ML expert analyzing class imbalance in a classification dataset.

Dataset Metrics:
- Total samples: {metrics["total_samples"]}
- Minority class ({metrics["minority_class"]}): {metrics["minority_count"]} samples ({metrics["minority_ratio"]:.1%})
- Majority class ({metrics["majority_class"]}): {metrics["majority_count"]} samples ({1 - metrics["minority_ratio"]:.1%})
- Imbalance ratio: {metrics["imbalance_ratio"]:.1f}:1
- Severity: {metrics["severity"]}

Model: {algorithm_name} for {problem_type}

Based on this imbalance, recommend ONE strategy from:
- smote: Synthetic minority oversampling (best for moderate-severe imbalance with enough minority samples)
- random_oversample: Duplicate minority samples (simple, works with small datasets)
- random_undersample: Remove majority samples (when majority class is very large)
- smote_tomek: SMOTE + Tomek links cleaning (removes noisy samples after SMOTE)
- class_weight: Use class weights only (for tree-based models, no resampling)
- combined: Moderate oversampling + class weights (for extreme imbalance)
- none: No action needed (for minimal imbalance)

Consider:
1. Dataset size - SMOTE needs k_neighbors (typically 5) minority samples
2. Algorithm type - tree-based models handle class_weight well
3. Severity level - extreme imbalance may need combined approaches

Respond in exactly this format:
STRATEGY: <strategy_name>
RATIONALE: <one sentence explanation>"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text

        # Parse response
        strategy = "class_weight"  # Default fallback
        rationale = "Default strategy"

        for line in response_text.split("\n"):
            if line.startswith("STRATEGY:"):
                strategy = line.replace("STRATEGY:", "").strip().lower()
            elif line.startswith("RATIONALE:"):
                rationale = line.replace("RATIONALE:", "").strip()

        # Validate strategy
        if strategy not in VALID_STRATEGIES:
            logger.warning(f"LLM returned invalid strategy '{strategy}', using heuristic")
            return _heuristic_strategy(metrics, algorithm_name)

        return strategy, rationale

    except Exception as e:
        logger.warning(f"LLM recommendation failed: {e}, using heuristic fallback")
        return _heuristic_strategy(metrics, algorithm_name)


def _heuristic_strategy(
    metrics: Dict[str, Any],
    algorithm_name: str,
) -> tuple[str, str]:
    """Heuristic fallback for strategy selection.

    Args:
        metrics: Imbalance metrics
        algorithm_name: Algorithm being trained

    Returns:
        Tuple of (strategy, rationale)
    """
    severity = metrics["severity"]
    minority_count = metrics["minority_count"]
    metrics["total_samples"]

    # Tree-based models handle class weights well
    tree_models = ["XGBoost", "LightGBM", "RandomForest", "GradientBoosting", "CausalForest"]
    is_tree_model = algorithm_name in tree_models

    if severity == "none":
        return "none", "Class distribution is balanced (minority >= 40%)"

    if severity == "moderate":
        if is_tree_model:
            return (
                "class_weight",
                "Moderate imbalance - tree models handle class weights efficiently",
            )
        return "random_oversample", "Moderate imbalance - light oversampling recommended"

    if severity == "severe":
        if minority_count >= 10:  # Enough samples for SMOTE
            return "smote", "Severe imbalance with sufficient minority samples for SMOTE"
        return "random_oversample", "Severe imbalance but too few minority samples for SMOTE"

    # Extreme imbalance
    if minority_count >= 10:
        return "combined", "Extreme imbalance - combining SMOTE with class weights"
    elif minority_count >= 5:
        return "random_oversample", "Extreme imbalance with minimal minority samples"
    else:
        return "class_weight", "Extreme imbalance with too few samples for resampling"


async def detect_class_imbalance(state: Dict[str, Any]) -> Dict[str, Any]:
    """Detect class imbalance in training data and recommend remediation.

    This node:
    1. Analyzes class distribution in training data
    2. Classifies severity (none/moderate/severe/extreme)
    3. Uses Claude to recommend optimal remediation strategy
    4. Returns strategy for downstream resampling node

    Args:
        state: ModelTrainerState with train_data

    Returns:
        Dictionary with imbalance_detected, imbalance_ratio, minority_ratio,
        imbalance_severity, class_distribution, recommended_strategy,
        strategy_rationale
    """
    # Extract training labels
    train_data = state.get("train_data", {})
    y_train = train_data.get("y")
    algorithm_name = state.get("algorithm_name", "Unknown")
    problem_type = state.get("problem_type", "binary_classification")

    if y_train is None:
        logger.warning("No training labels available for imbalance detection")
        return {
            "imbalance_detected": False,
            "imbalance_ratio": 1.0,
            "minority_ratio": 0.5,
            "imbalance_severity": "unknown",
            "class_distribution": {},
            "recommended_strategy": "none",
            "strategy_rationale": "No training data available for imbalance detection",
        }

    # Convert to numpy if needed
    if hasattr(y_train, "values"):
        y_train = y_train.values
    y_train = np.asarray(y_train).flatten()

    # Check for regression problem (no class imbalance applicable)
    if problem_type in ["regression", "continuous"]:
        logger.info("Regression problem - class imbalance not applicable")
        return {
            "imbalance_detected": False,
            "imbalance_ratio": 1.0,
            "minority_ratio": 0.5,
            "imbalance_severity": "not_applicable",
            "class_distribution": {},
            "recommended_strategy": "none",
            "strategy_rationale": "Class imbalance detection not applicable for regression",
        }

    # Check for unique classes
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        logger.warning(f"Only {len(unique_classes)} class(es) found in training data")
        return {
            "imbalance_detected": False,
            "imbalance_ratio": float("inf"),
            "minority_ratio": 0.0,
            "imbalance_severity": "degenerate",
            "class_distribution": {int(c): int(np.sum(y_train == c)) for c in unique_classes},
            "recommended_strategy": "none",
            "strategy_rationale": "Insufficient classes for imbalance analysis",
        }

    # Calculate imbalance metrics
    metrics = _calculate_imbalance_metrics(y_train)

    logger.info(
        f"Class imbalance analysis: severity={metrics['severity']}, "
        f"minority_ratio={metrics['minority_ratio']:.2%}, "
        f"imbalance_ratio={metrics['imbalance_ratio']:.1f}:1"
    )

    # Determine if imbalance is detected (anything beyond "none" severity)
    imbalance_detected = metrics["severity"] != "none"

    if not imbalance_detected:
        return {
            "imbalance_detected": False,
            "imbalance_ratio": metrics["imbalance_ratio"],
            "minority_ratio": metrics["minority_ratio"],
            "imbalance_severity": metrics["severity"],
            "class_distribution": metrics["class_distribution"],
            "recommended_strategy": "none",
            "strategy_rationale": "Class distribution is balanced - no remediation needed",
        }

    # Get LLM recommendation for remediation strategy
    strategy, rationale = await _get_llm_recommendation(metrics, algorithm_name, problem_type)

    logger.info(f"Recommended strategy: {strategy} - {rationale}")

    return {
        "imbalance_detected": True,
        "imbalance_ratio": metrics["imbalance_ratio"],
        "minority_ratio": metrics["minority_ratio"],
        "imbalance_severity": metrics["severity"],
        "class_distribution": metrics["class_distribution"],
        "recommended_strategy": strategy,
        "strategy_rationale": rationale,
    }
