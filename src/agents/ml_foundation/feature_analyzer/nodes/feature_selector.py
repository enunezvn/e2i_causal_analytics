"""Feature Selection Node - NO LLM.

Selects the most relevant features using statistical and model-based methods:
- Variance threshold (remove low-variance features)
- Correlation analysis (remove highly correlated features)
- Model-based importance (Random Forest feature importance)
- Multicollinearity detection (VIF)

This is a deterministic computation node with no LLM calls.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# Selection method constants
VARIANCE_SELECTION = "variance"
CORRELATION_SELECTION = "correlation"
MODEL_SELECTION = "model_importance"
VIF_SELECTION = "vif"


async def select_features(state: Dict[str, Any]) -> Dict[str, Any]:
    """Select optimal features using multiple selection strategies.

    This node:
    1. Removes low-variance features
    2. Removes highly correlated features
    3. Ranks features by model-based importance
    4. Optionally removes multicollinear features (VIF)
    5. Returns selected feature set

    Args:
        state: Current agent state with generated features
            Required:
            - X_train_generated (or X_train): Training features DataFrame
            Optional:
            - X_val_generated (or X_val): Validation features DataFrame
            - X_test_generated (or X_test): Test features DataFrame
            - y_train: Training target (for model-based selection)
            - selection_config: Custom selection configuration
            - problem_type: "classification" or "regression"

    Returns:
        State updates with selected features and selection metadata
    """
    start_time = time.time()

    try:
        # Get DataFrames (prefer generated versions if available)
        # Note: Cannot use `df or fallback` because DataFrame truthiness is ambiguous
        X_train = state.get("X_train_generated")
        if X_train is None:
            X_train = state.get("X_train")
        X_val = state.get("X_val_generated")
        if X_val is None:
            X_val = state.get("X_val")
        X_test = state.get("X_test_generated")
        if X_test is None:
            X_test = state.get("X_test")
        y_train = state.get("y_train")
        selection_config = state.get("selection_config", {})
        problem_type = state.get("problem_type", "classification")

        if X_train is None:
            return {
                "error": "Missing training data",
                "error_type": "missing_training_data",
                "status": "failed",
            }

        # Convert to DataFrame if needed
        if isinstance(X_train, np.ndarray):
            n_features = X_train.shape[1]
            columns = [f"feature_{i}" for i in range(n_features)]
            X_train = pd.DataFrame(X_train, columns=columns)
            if X_val is not None and isinstance(X_val, np.ndarray):
                X_val = pd.DataFrame(X_val, columns=columns)
            if X_test is not None and isinstance(X_test, np.ndarray):
                X_test = pd.DataFrame(X_test, columns=columns)

        # Get only numeric columns for selection
        X_train_numeric = X_train.select_dtypes(include=[np.number])
        original_features = list(X_train_numeric.columns)
        original_count = len(original_features)

        # Track selection process
        selection_history: List[Dict[str, Any]] = []
        removed_features: Dict[str, List[str]] = {
            VARIANCE_SELECTION: [],
            CORRELATION_SELECTION: [],
            VIF_SELECTION: [],
        }

        # Current feature set (will be progressively filtered)
        current_features = original_features.copy()

        # 1. Variance Threshold Selection
        if selection_config.get("apply_variance_threshold", True):
            variance_threshold = selection_config.get("variance_threshold", 0.01)
            current_features, var_removed = _apply_variance_selection(
                X_train_numeric[current_features],
                threshold=variance_threshold,
            )
            removed_features[VARIANCE_SELECTION] = var_removed
            selection_history.append(
                {
                    "step": "variance_threshold",
                    "threshold": variance_threshold,
                    "features_before": len(current_features) + len(var_removed),
                    "features_after": len(current_features),
                    "removed": len(var_removed),
                }
            )
            logger.info(f"Variance selection: removed {len(var_removed)} features")

        # 2. Correlation-based Selection
        if selection_config.get("apply_correlation_filter", True):
            correlation_threshold = selection_config.get("correlation_threshold", 0.95)
            current_features, corr_removed = _apply_correlation_selection(
                X_train_numeric[current_features],
                threshold=correlation_threshold,
            )
            removed_features[CORRELATION_SELECTION] = corr_removed
            selection_history.append(
                {
                    "step": "correlation_filter",
                    "threshold": correlation_threshold,
                    "features_before": len(current_features) + len(corr_removed),
                    "features_after": len(current_features),
                    "removed": len(corr_removed),
                }
            )
            logger.info(f"Correlation selection: removed {len(corr_removed)} features")

        # 3. VIF-based Selection (optional, can be slow)
        if selection_config.get("apply_vif_filter", False):
            vif_threshold = selection_config.get("vif_threshold", 10.0)
            current_features, vif_removed = _apply_vif_selection(
                X_train_numeric[current_features],
                threshold=vif_threshold,
            )
            removed_features[VIF_SELECTION] = vif_removed
            selection_history.append(
                {
                    "step": "vif_filter",
                    "threshold": vif_threshold,
                    "features_before": len(current_features) + len(vif_removed),
                    "features_after": len(current_features),
                    "removed": len(vif_removed),
                }
            )
            logger.info(f"VIF selection: removed {len(vif_removed)} features")

        # 4. Model-based Importance Ranking
        feature_importance: Dict[str, float] = {}
        if selection_config.get("compute_importance", True) and y_train is not None:
            max_features_for_importance = selection_config.get("max_features_for_selection", None)
            feature_importance, importance_ranking = _compute_model_importance(
                X_train_numeric[current_features],
                y_train,
                problem_type=problem_type,
            )
            selection_history.append(
                {
                    "step": "model_importance",
                    "model": "RandomForest",
                    "features_ranked": len(importance_ranking),
                }
            )

            # Apply feature count limit if specified
            if max_features_for_importance and len(current_features) > max_features_for_importance:
                current_features = importance_ranking[:max_features_for_importance]
                selection_history.append(
                    {
                        "step": "importance_filter",
                        "max_features": max_features_for_importance,
                        "features_selected": len(current_features),
                    }
                )

        # 5. Select final features from DataFrames
        # Include both numeric and non-numeric columns that weren't filtered
        non_numeric_cols = [c for c in X_train.columns if c not in X_train_numeric.columns]
        final_features = current_features + non_numeric_cols

        X_train_selected = X_train[final_features]
        X_val_selected = X_val[final_features] if X_val is not None else None
        X_test_selected = X_test[final_features] if X_test is not None else None

        # Compute feature statistics
        feature_stats = _compute_feature_statistics(X_train_selected)

        computation_time = time.time() - start_time

        logger.info(
            f"Feature selection complete: {original_count} -> {len(current_features)} "
            f"numeric features ({len(final_features)} total) in {computation_time:.2f}s"
        )

        return {
            "X_train_selected": X_train_selected,
            "X_val_selected": X_val_selected,
            "X_test_selected": X_test_selected,
            "selected_features": current_features,
            "selected_features_all": final_features,
            "feature_importance": feature_importance,
            "feature_importance_ranked": sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            ),
            "removed_features": removed_features,
            "selection_history": selection_history,
            "feature_statistics": feature_stats,
            "original_feature_count": original_count,
            "selected_feature_count": len(current_features),
            "total_selected_count": len(final_features),
            "selection_time_seconds": computation_time,
        }

    except Exception as e:
        logger.exception("Feature selection failed")
        return {
            "error": f"Feature selection failed: {str(e)}",
            "error_type": "feature_selection_error",
            "error_details": {"exception": str(e)},
            "status": "failed",
        }


def _apply_variance_selection(
    df: pd.DataFrame,
    threshold: float = 0.01,
) -> Tuple[List[str], List[str]]:
    """Remove features with variance below threshold.

    Args:
        df: Input DataFrame (numeric columns only)
        threshold: Minimum variance threshold

    Returns:
        Tuple of (selected features, removed features)
    """
    # Handle edge case of empty DataFrame
    if df.empty or len(df.columns) == 0:
        return [], []

    # Fill NaN for variance calculation
    df_filled = df.fillna(0)

    # Apply variance threshold
    selector = VarianceThreshold(threshold=threshold)
    try:
        selector.fit(df_filled)
        selected_mask = selector.get_support()
        selected_features = list(df.columns[selected_mask])
        removed_features = list(df.columns[~selected_mask])
    except Exception:
        # If variance selection fails, return all features
        selected_features = list(df.columns)
        removed_features = []

    return selected_features, removed_features


def _apply_correlation_selection(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[List[str], List[str]]:
    """Remove highly correlated features.

    For each pair of features with correlation > threshold,
    removes the one with lower mean absolute correlation to other features.

    Args:
        df: Input DataFrame (numeric columns only)
        threshold: Correlation threshold for removal

    Returns:
        Tuple of (selected features, removed features)
    """
    if df.empty or len(df.columns) < 2:
        return list(df.columns), []

    # Compute correlation matrix
    corr_matrix = df.corr().abs()

    # Get upper triangle (to avoid double counting)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features to drop
    to_drop = set()
    for column in upper_tri.columns:
        # Find features highly correlated with this column
        high_corr = upper_tri[column][upper_tri[column] > threshold].index.tolist()
        if high_corr:
            # Compare mean correlation with all other features
            col_mean_corr = corr_matrix[column].mean()
            for corr_col in high_corr:
                if corr_col not in to_drop:
                    corr_col_mean = corr_matrix[corr_col].mean()
                    # Drop the one with higher mean correlation
                    if corr_col_mean > col_mean_corr:
                        to_drop.add(corr_col)
                    else:
                        to_drop.add(column)
                        break

    removed_features = list(to_drop)
    selected_features = [c for c in df.columns if c not in to_drop]

    return selected_features, removed_features


def _apply_vif_selection(
    df: pd.DataFrame,
    threshold: float = 10.0,
    max_iterations: int = 20,
) -> Tuple[List[str], List[str]]:
    """Remove multicollinear features using Variance Inflation Factor (VIF).

    Iteratively removes the feature with highest VIF until all VIFs < threshold.

    Args:
        df: Input DataFrame (numeric columns only)
        threshold: VIF threshold for removal
        max_iterations: Maximum removal iterations

    Returns:
        Tuple of (selected features, removed features)
    """
    if df.empty or len(df.columns) < 2:
        return list(df.columns), []

    # Fill NaN for VIF calculation
    df_filled = df.fillna(0)

    # Add constant for VIF calculation
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    removed = []
    remaining_cols = list(df_filled.columns)

    for iteration in range(max_iterations):
        if len(remaining_cols) < 2:
            break

        # Compute VIF for each feature
        X = df_filled[remaining_cols].values
        vif_data = []
        for i, col in enumerate(remaining_cols):
            try:
                vif = variance_inflation_factor(X, i)
                vif_data.append((col, vif))
            except Exception:
                vif_data.append((col, 0))

        # Find max VIF
        max_vif_feature, max_vif = max(vif_data, key=lambda x: x[1])

        if max_vif < threshold or np.isinf(max_vif):
            break

        # Remove feature with highest VIF
        removed.append(max_vif_feature)
        remaining_cols.remove(max_vif_feature)
        logger.debug(f"VIF iteration {iteration}: removed {max_vif_feature} (VIF={max_vif:.2f})")

    return remaining_cols, removed


def _compute_model_importance(
    X: pd.DataFrame,
    y: Any,
    problem_type: str = "classification",
) -> Tuple[Dict[str, float], List[str]]:
    """Compute feature importance using Random Forest.

    Args:
        X: Feature DataFrame
        y: Target variable
        problem_type: "classification" or "regression"

    Returns:
        Tuple of (importance dict, ranked feature list)
    """
    # Handle target variable
    if isinstance(y, pd.Series):
        y_values = y.values
    elif isinstance(y, np.ndarray):
        y_values = y
    else:
        y_values = np.array(y)

    # Encode categorical target if needed
    if problem_type == "classification":
        if y_values.dtype == object or (
            hasattr(y_values, "dtype") and not np.issubdtype(y_values.dtype, np.number)
        ):
            le = LabelEncoder()
            y_values = le.fit_transform(y_values.astype(str))

    # Fill NaN in features
    X_filled = X.fillna(0)

    # Train simple model for importance
    try:
        if problem_type == "classification":
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )

        model.fit(X_filled, y_values)
        importances = model.feature_importances_

        # Create importance dict
        importance_dict = {
            col: float(imp) for col, imp in zip(X.columns, importances, strict=False)
        }

        # Rank features by importance
        ranked = sorted(
            importance_dict.keys(),
            key=lambda x: importance_dict[x],
            reverse=True,
        )

        return importance_dict, ranked

    except Exception as e:
        logger.warning(f"Model-based importance failed: {e}")
        # Return equal importance as fallback
        equal_importance = 1.0 / len(X.columns)
        return dict.fromkeys(X.columns, equal_importance), list(X.columns)


def _compute_feature_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute statistics for selected features.

    Args:
        df: Selected features DataFrame

    Returns:
        Dict mapping feature name to statistics
    """
    stats = {}

    for col in df.columns:
        col_stats: Dict[str, Any] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "null_pct": float(df[col].isna().mean()),
            "unique_count": int(df[col].nunique()),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update(
                {
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "median": float(df[col].median()) if not df[col].isna().all() else None,
                }
            )

        stats[col] = col_stats

    return stats


def get_feature_selection_summary(state: Dict[str, Any]) -> str:
    """Generate a human-readable summary of feature selection.

    Args:
        state: Agent state with selection results

    Returns:
        Summary string
    """
    lines = ["Feature Selection Summary", "=" * 40]

    original = state.get("original_feature_count", 0)
    selected = state.get("selected_feature_count", 0)
    total = state.get("total_selected_count", 0)

    lines.append(f"Original features (numeric): {original}")
    lines.append(f"Selected features (numeric): {selected}")
    lines.append(f"Total selected (all types): {total}")
    lines.append(f"Features removed: {original - selected}")
    lines.append("")

    # Selection history
    history = state.get("selection_history", [])
    if history:
        lines.append("Selection Steps:")
        for step in history:
            step_name = step.get("step", "unknown")
            removed = step.get("removed", 0)
            lines.append(f"  - {step_name}: removed {removed} features")

    # Removed features by type
    removed = state.get("removed_features", {})
    if any(removed.values()):
        lines.append("")
        lines.append("Removed Features by Type:")
        for method, features in removed.items():
            if features:
                lines.append(f"  {method}: {len(features)} features")
                if len(features) <= 5:
                    for f in features:
                        lines.append(f"    - {f}")
                else:
                    for f in features[:3]:
                        lines.append(f"    - {f}")
                    lines.append(f"    ... and {len(features) - 3} more")

    # Top important features
    importance_ranked = state.get("feature_importance_ranked", [])
    if importance_ranked:
        lines.append("")
        lines.append("Top 10 Features by Importance:")
        for i, (feature, importance) in enumerate(importance_ranked[:10], 1):
            lines.append(f"  {i}. {feature}: {importance:.4f}")

    return "\n".join(lines)
