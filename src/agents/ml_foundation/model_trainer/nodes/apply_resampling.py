"""Apply resampling strategies for class imbalance.

This module applies the recommended resampling strategy to training data.
CRITICAL: Resampling is ONLY applied to training data. Validation and test
sets remain untouched to prevent data leakage.

Version: 1.0.0
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_numpy(data: Any) -> np.ndarray:
    """Convert data to numpy array if needed.

    Args:
        data: Input data

    Returns:
        Numpy array
    """
    if data is None:
        return None

    if isinstance(data, np.ndarray):
        return data

    try:
        import pandas as pd

        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
    except ImportError:
        pass

    if isinstance(data, (list, tuple)):
        return np.array(data)

    return np.asarray(data)


def _apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    minority_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling.

    Args:
        X: Feature matrix
        y: Target labels
        minority_count: Number of minority samples

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    from imblearn.over_sampling import SMOTE

    # Adaptive k_neighbors based on minority count
    k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    return smote.fit_resample(X, y)


def _apply_random_oversample(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random oversampling.

    Args:
        X: Feature matrix
        y: Target labels

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    from imblearn.over_sampling import RandomOverSampler

    ros = RandomOverSampler(random_state=42)
    return ros.fit_resample(X, y)


def _apply_random_undersample(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random undersampling.

    Args:
        X: Feature matrix
        y: Target labels

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    from imblearn.under_sampling import RandomUnderSampler

    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample(X, y)


def _apply_smote_tomek(
    X: np.ndarray,
    y: np.ndarray,
    minority_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE + Tomek links cleaning.

    Args:
        X: Feature matrix
        y: Target labels
        minority_count: Number of minority samples

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE

    # Adaptive k_neighbors based on minority count
    k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    smote_tomek = SMOTETomek(random_state=42, smote=smote)
    return smote_tomek.fit_resample(X, y)


def _apply_combined(
    X: np.ndarray,
    y: np.ndarray,
    minority_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply combined strategy: moderate SMOTE (50% ratio) for extreme imbalance.

    This avoids creating too many synthetic samples while still improving balance.
    Class weights will be used in addition during training.

    Args:
        X: Feature matrix
        y: Target labels
        minority_count: Number of minority samples

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    from imblearn.over_sampling import SMOTE

    # Adaptive k_neighbors based on minority count
    k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1

    # Target 50% ratio instead of full balance
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=0.5)
    return smote.fit_resample(X, y)


def _apply_strategy(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str,
    minority_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the specified resampling strategy.

    Args:
        X: Feature matrix
        y: Target labels
        strategy: Resampling strategy name
        minority_count: Number of minority samples

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    if strategy == "smote":
        return _apply_smote(X, y, minority_count)
    elif strategy == "random_oversample":
        return _apply_random_oversample(X, y)
    elif strategy == "random_undersample":
        return _apply_random_undersample(X, y)
    elif strategy == "smote_tomek":
        return _apply_smote_tomek(X, y, minority_count)
    elif strategy == "combined":
        return _apply_combined(X, y, minority_count)
    else:
        # No resampling for "none" or "class_weight"
        return X, y


async def apply_resampling(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply resampling strategy to training data.

    CRITICAL: Resampling is ONLY applied to training data.
    Validation, test, and holdout sets remain untouched.

    This node:
    1. Checks if resampling is recommended
    2. Applies the recommended strategy to training data only
    3. Returns resampled training data for downstream training

    Args:
        state: ModelTrainerState with recommended_strategy, preprocessed data

    Returns:
        Dictionary with X_train_resampled, y_train_resampled,
        resampling_applied, original_train_shape, resampled_train_shape,
        original_distribution, resampled_distribution
    """
    # Extract state
    recommended_strategy = state.get("recommended_strategy", "none")
    imbalance_detected = state.get("imbalance_detected", False)
    X_train_preprocessed = state.get("X_train_preprocessed")
    train_data = state.get("train_data", {})
    y_train = train_data.get("y")
    class_distribution = state.get("class_distribution", {})

    # Get minority count for adaptive SMOTE
    minority_count = min(class_distribution.values()) if class_distribution else 5

    # Convert to numpy
    X_train = _ensure_numpy(X_train_preprocessed)
    y_train_np = _ensure_numpy(y_train)

    # Check if we should apply resampling
    should_resample = (
        imbalance_detected
        and recommended_strategy not in ("none", "class_weight")
        and X_train is not None
        and y_train_np is not None
    )

    if not should_resample:
        logger.info(
            f"Resampling not applied: strategy={recommended_strategy}, "
            f"imbalance_detected={imbalance_detected}"
        )
        return {
            "X_train_resampled": X_train,
            "y_train_resampled": y_train_np,
            "resampling_applied": False,
            "resampling_strategy": recommended_strategy,
            "original_train_shape": X_train.shape if X_train is not None else None,
            "resampled_train_shape": X_train.shape if X_train is not None else None,
            "original_distribution": class_distribution,
            "resampled_distribution": class_distribution,
        }

    # Record original shape and distribution
    original_shape = X_train.shape
    original_distribution = class_distribution.copy()

    logger.info(
        f"Applying resampling strategy: {recommended_strategy}, original shape: {original_shape}"
    )

    try:
        # Apply the recommended strategy
        X_resampled, y_resampled = _apply_strategy(
            X_train, y_train_np, recommended_strategy, minority_count
        )

        # Calculate new distribution
        unique, counts = np.unique(y_resampled, return_counts=True)
        resampled_distribution = dict(
            zip(unique.astype(int).tolist(), counts.tolist(), strict=False)
        )

        logger.info(
            f"Resampling complete: {original_shape} -> {X_resampled.shape}, "
            f"distribution: {original_distribution} -> {resampled_distribution}"
        )

        return {
            "X_train_resampled": X_resampled,
            "y_train_resampled": y_resampled,
            "resampling_applied": True,
            "resampling_strategy": recommended_strategy,
            "original_train_shape": original_shape,
            "resampled_train_shape": X_resampled.shape,
            "original_distribution": original_distribution,
            "resampled_distribution": resampled_distribution,
        }

    except ImportError as e:
        logger.error(f"imbalanced-learn not installed: {e}")
        return {
            "error": f"imbalanced-learn required for resampling: {e}",
            "error_type": "missing_dependency",
            "X_train_resampled": X_train,
            "y_train_resampled": y_train_np,
            "resampling_applied": False,
            "resampling_strategy": "none",
            "original_train_shape": original_shape,
            "resampled_train_shape": original_shape,
            "original_distribution": original_distribution,
            "resampled_distribution": original_distribution,
        }

    except Exception as e:
        logger.error(f"Resampling failed: {e}")
        return {
            "error": f"Resampling failed: {e}",
            "error_type": "resampling_error",
            "X_train_resampled": X_train,
            "y_train_resampled": y_train_np,
            "resampling_applied": False,
            "resampling_strategy": "none",
            "original_train_shape": original_shape,
            "resampled_train_shape": original_shape,
            "original_distribution": original_distribution,
            "resampled_distribution": original_distribution,
        }
