"""Baseline computer node for data_preparer agent.

This node computes baseline metrics from the TRAIN split only.
These metrics are used by drift_monitor for distribution drift detection.
"""

from datetime import datetime
from typing import Dict, Any
import logging
import numpy as np

from ..state import DataPreparerState

logger = logging.getLogger(__name__)


async def compute_baseline_metrics(state: DataPreparerState) -> Dict[str, Any]:
    """Compute baseline metrics from training data.

    CRITICAL: Baseline metrics MUST be computed from TRAIN split only.
    Never include validation, test, or holdout data in baseline computation.

    This node computes:
    1. Feature statistics (mean, std, min, max, percentiles)
    2. Target distribution (for classification)
    3. Target rate (for binary classification)
    4. Correlation matrix

    Args:
        state: Current agent state

    Returns:
        Updated state with baseline metrics
    """
    logger.info(f"Computing baseline metrics for experiment {state['experiment_id']}")

    try:
        train_df = state.get("train_df")
        if train_df is None:
            raise ValueError("train_df not found in state")

        scope_spec = state.get("scope_spec", {})
        target_variable = scope_spec.get("prediction_target")
        required_features = scope_spec.get("required_features", [])

        # Compute feature statistics
        feature_stats = {}
        for feature in required_features:
            if feature not in train_df.columns:
                logger.warning(f"Feature {feature} not found in train data")
                continue

            feature_data = train_df[feature]

            # Compute statistics based on dtype
            if np.issubdtype(feature_data.dtype, np.number):
                # Numerical feature
                stats = {
                    "mean": float(feature_data.mean()),
                    "std": float(feature_data.std()),
                    "min": float(feature_data.min()),
                    "max": float(feature_data.max()),
                    "p25": float(feature_data.quantile(0.25)),
                    "p50": float(feature_data.quantile(0.50)),
                    "p75": float(feature_data.quantile(0.75)),
                    "missing_count": int(feature_data.isna().sum()),
                    "missing_rate": float(feature_data.isna().mean()),
                    "dtype": "numerical",
                }
            else:
                # Categorical feature
                value_counts = feature_data.value_counts()
                stats = {
                    "unique_count": int(feature_data.nunique()),
                    "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "most_common_freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "missing_count": int(feature_data.isna().sum()),
                    "missing_rate": float(feature_data.isna().mean()),
                    "dtype": "categorical",
                }

            feature_stats[feature] = stats

        # Compute target statistics
        target_rate = None
        target_distribution = {}

        if target_variable and target_variable in train_df.columns:
            target_data = train_df[target_variable]

            if np.issubdtype(target_data.dtype, np.number) and target_data.nunique() == 2:
                # Binary classification
                target_rate = float(target_data.mean())
                target_distribution = {
                    "type": "binary",
                    "positive_rate": target_rate,
                    "negative_rate": 1 - target_rate,
                    "positive_count": int(target_data.sum()),
                    "negative_count": int(len(target_data) - target_data.sum()),
                }
            elif target_data.nunique() <= 20:
                # Multiclass classification
                value_counts = target_data.value_counts()
                target_distribution = {
                    "type": "multiclass",
                    "class_counts": {str(k): int(v) for k, v in value_counts.items()},
                    "class_proportions": {
                        str(k): float(v / len(target_data)) for k, v in value_counts.items()
                    },
                }
            else:
                # Regression
                target_distribution = {
                    "type": "regression",
                    "mean": float(target_data.mean()),
                    "std": float(target_data.std()),
                    "min": float(target_data.min()),
                    "max": float(target_data.max()),
                }

        # Compute correlation matrix (numerical features only)
        numerical_features = [
            f
            for f in required_features
            if f in train_df.columns and np.issubdtype(train_df[f].dtype, np.number)
        ]

        correlation_matrix = {}
        if len(numerical_features) > 1:
            corr_df = train_df[numerical_features].corr()
            correlation_matrix = {
                col: {row: float(corr_df.loc[row, col]) for row in corr_df.index}
                for col in corr_df.columns
            }

        # Training sample count
        training_samples = len(train_df)

        # Update state
        updates = {
            "feature_stats": feature_stats,
            "target_rate": target_rate,
            "target_distribution": target_distribution,
            "correlation_matrix": correlation_matrix,
            "computed_at": datetime.now().isoformat(),
            "training_samples": training_samples,
        }

        logger.info(
            f"Baseline metrics computed: {len(feature_stats)} features, "
            f"{training_samples} training samples"
        )

        return updates

    except Exception as e:
        logger.error(f"Baseline computation failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_type": "baseline_computation_error",
        }
