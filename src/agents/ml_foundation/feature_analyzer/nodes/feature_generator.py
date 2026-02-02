"""Feature Generation Node - NO LLM.

Generates engineered features from raw data:
- Temporal features (lag, rolling statistics)
- Interaction features (categorical crosses, polynomial)
- Domain-specific features (pharma KPIs)

This is a deterministic computation node with no LLM calls.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Feature type constants
TEMPORAL_FEATURES = "temporal"
INTERACTION_FEATURES = "interaction"
DOMAIN_FEATURES = "domain"
AGGREGATE_FEATURES = "aggregate"


async def generate_features(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate engineered features from prepared data.

    This node:
    1. Creates temporal features (lags, rolling stats, date parts)
    2. Creates interaction features (categorical crosses)
    3. Creates domain-specific features (pharma KPIs)
    4. Tracks feature metadata for the feature store

    Args:
        state: Current agent state with prepared DataFrames
            Required:
            - X_train: Training features DataFrame
            - X_val: Validation features DataFrame (optional)
            - X_test: Test features DataFrame (optional)
            Optional:
            - temporal_columns: List of columns for temporal features
            - categorical_columns: List of columns for interactions
            - numeric_columns: List of numeric columns
            - feature_config: Custom feature generation config

    Returns:
        State updates with generated features and metadata
    """
    start_time = time.time()

    try:
        # Extract inputs
        X_train = state.get("X_train")
        X_val = state.get("X_val")
        X_test = state.get("X_test")
        feature_config = state.get("feature_config", {})

        if X_train is None:
            return {
                "error": "Missing X_train DataFrame",
                "error_type": "missing_training_data",
                "status": "failed",
            }

        # Convert to DataFrame if numpy array
        if isinstance(X_train, np.ndarray):
            n_features = X_train.shape[1]
            columns = [f"feature_{i}" for i in range(n_features)]
            X_train = pd.DataFrame(X_train, columns=columns)
            if X_val is not None and isinstance(X_val, np.ndarray):
                X_val = pd.DataFrame(X_val, columns=columns)
            if X_test is not None and isinstance(X_test, np.ndarray):
                X_test = pd.DataFrame(X_test, columns=columns)

        # Auto-detect column types if not provided
        temporal_columns = state.get("temporal_columns", _detect_temporal_columns(X_train))
        categorical_columns = state.get("categorical_columns", _detect_categorical_columns(X_train))
        numeric_columns = state.get("numeric_columns", _detect_numeric_columns(X_train))

        # Track generated features
        generated_features: List[Dict[str, Any]] = []
        feature_metadata: Dict[str, Any] = {
            "temporal": [],
            "interaction": [],
            "domain": [],
            "aggregate": [],
        }

        # 1. Generate temporal features
        if feature_config.get("generate_temporal", True) and temporal_columns:
            X_train, temporal_meta = _generate_temporal_features(
                X_train,
                temporal_columns,
                lag_periods=feature_config.get("lag_periods", [1, 7, 30]),
                rolling_windows=feature_config.get("rolling_windows", [7, 30]),
            )
            feature_metadata["temporal"] = temporal_meta
            generated_features.extend(temporal_meta)

            # Apply same transformations to val/test
            if X_val is not None:
                X_val, _ = _generate_temporal_features(
                    X_val,
                    temporal_columns,
                    lag_periods=feature_config.get("lag_periods", [1, 7, 30]),
                    rolling_windows=feature_config.get("rolling_windows", [7, 30]),
                )
            if X_test is not None:
                X_test, _ = _generate_temporal_features(
                    X_test,
                    temporal_columns,
                    lag_periods=feature_config.get("lag_periods", [1, 7, 30]),
                    rolling_windows=feature_config.get("rolling_windows", [7, 30]),
                )

        # 2. Generate interaction features
        if feature_config.get("generate_interactions", True) and categorical_columns:
            max_interactions = feature_config.get("max_interactions", 10)
            X_train, interaction_meta = _generate_interaction_features(
                X_train,
                categorical_columns,
                numeric_columns,
                max_interactions=max_interactions,
            )
            feature_metadata["interaction"] = interaction_meta
            generated_features.extend(interaction_meta)

            # Apply same transformations to val/test
            if X_val is not None:
                X_val, _ = _generate_interaction_features(
                    X_val,
                    categorical_columns,
                    numeric_columns,
                    max_interactions=max_interactions,
                )
            if X_test is not None:
                X_test, _ = _generate_interaction_features(
                    X_test,
                    categorical_columns,
                    numeric_columns,
                    max_interactions=max_interactions,
                )

        # 3. Generate domain-specific features (pharma KPIs)
        if feature_config.get("generate_domain", True):
            X_train, domain_meta = _generate_domain_features(X_train)
            feature_metadata["domain"] = domain_meta
            generated_features.extend(domain_meta)

            if X_val is not None:
                X_val, _ = _generate_domain_features(X_val)
            if X_test is not None:
                X_test, _ = _generate_domain_features(X_test)

        # 4. Generate aggregate features
        if feature_config.get("generate_aggregates", True) and numeric_columns:
            X_train, aggregate_meta = _generate_aggregate_features(X_train, numeric_columns)
            feature_metadata["aggregate"] = aggregate_meta
            generated_features.extend(aggregate_meta)

            if X_val is not None:
                X_val, _ = _generate_aggregate_features(X_val, numeric_columns)
            if X_test is not None:
                X_test, _ = _generate_aggregate_features(X_test, numeric_columns)

        # Handle any NaN values created by lag/rolling operations
        fill_strategy = feature_config.get("nan_fill_strategy", "median")
        X_train = _handle_generated_nans(X_train, strategy=fill_strategy)
        if X_val is not None:
            X_val = _handle_generated_nans(X_val, strategy=fill_strategy)
        if X_test is not None:
            X_test = _handle_generated_nans(X_test, strategy=fill_strategy)

        computation_time = time.time() - start_time

        # Get feature names
        original_features = list(state.get("X_train", pd.DataFrame()).columns)
        all_features = list(X_train.columns)
        new_features = [f for f in all_features if f not in original_features]

        logger.info(f"Generated {len(new_features)} new features in {computation_time:.2f}s")

        return {
            "X_train_generated": X_train,
            "X_val_generated": X_val,
            "X_test_generated": X_test,
            "generated_features": generated_features,
            "feature_metadata": feature_metadata,
            "original_feature_count": len(original_features),
            "total_feature_count": len(all_features),
            "new_feature_count": len(new_features),
            "new_feature_names": new_features,
            "feature_generation_time_seconds": computation_time,
            "temporal_columns_used": temporal_columns,
            "categorical_columns_used": categorical_columns,
            "numeric_columns_used": numeric_columns,
        }

    except Exception as e:
        logger.exception("Feature generation failed")
        return {
            "error": f"Feature generation failed: {str(e)}",
            "error_type": "feature_generation_error",
            "error_details": {"exception": str(e)},
            "status": "failed",
        }


def _detect_temporal_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns suitable for temporal feature generation."""
    temporal_keywords = [
        "date",
        "time",
        "timestamp",
        "day",
        "month",
        "year",
        "week",
        "quarter",
        "period",
        "created",
        "updated",
    ]
    temporal_cols = []

    for col in df.columns:
        col_lower = col.lower()
        # Check if datetime type
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            temporal_cols.append(col)
        # Check if name suggests temporal
        elif any(keyword in col_lower for keyword in temporal_keywords):
            temporal_cols.append(col)

    return temporal_cols


def _detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Detect categorical columns."""
    categorical_cols = []

    for col in df.columns:
        # Check if object/category type
        if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
            categorical_cols.append(col)
        # Check if low cardinality integer (likely categorical)
        elif pd.api.types.is_integer_dtype(df[col]):
            if df[col].nunique() < 20:  # Arbitrary threshold
                categorical_cols.append(col)

    return categorical_cols


def _detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Detect numeric columns suitable for aggregations."""
    return list(df.select_dtypes(include=[np.number]).columns)


def _generate_temporal_features(
    df: pd.DataFrame,
    temporal_columns: List[str],
    lag_periods: List[int] = None,
    rolling_windows: List[int] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Generate temporal features from time-series columns.

    Creates:
    - Lag features (shift by N periods)
    - Rolling statistics (mean, std over window)
    - Date part extraction (if datetime)

    Args:
        df: Input DataFrame
        temporal_columns: Columns to generate temporal features for
        lag_periods: List of lag periods to create
        rolling_windows: List of rolling window sizes

    Returns:
        Tuple of (transformed DataFrame, feature metadata list)
    """
    if rolling_windows is None:
        rolling_windows = [7, 30]
    if lag_periods is None:
        lag_periods = [1, 7, 30]
    df = df.copy()
    metadata = []

    for col in temporal_columns:
        if col not in df.columns:
            continue

        # Handle datetime columns - extract date parts
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            # Day of week (0=Monday, 6=Sunday)
            new_col = f"{col}_dayofweek"
            df[new_col] = df[col].dt.dayofweek
            metadata.append(
                {
                    "name": new_col,
                    "source": col,
                    "type": TEMPORAL_FEATURES,
                    "transformation": "dayofweek",
                }
            )

            # Month
            new_col = f"{col}_month"
            df[new_col] = df[col].dt.month
            metadata.append(
                {
                    "name": new_col,
                    "source": col,
                    "type": TEMPORAL_FEATURES,
                    "transformation": "month",
                }
            )

            # Quarter
            new_col = f"{col}_quarter"
            df[new_col] = df[col].dt.quarter
            metadata.append(
                {
                    "name": new_col,
                    "source": col,
                    "type": TEMPORAL_FEATURES,
                    "transformation": "quarter",
                }
            )

            # Is weekend
            new_col = f"{col}_is_weekend"
            df[new_col] = (df[col].dt.dayofweek >= 5).astype(int)
            metadata.append(
                {
                    "name": new_col,
                    "source": col,
                    "type": TEMPORAL_FEATURES,
                    "transformation": "is_weekend",
                }
            )

        # Handle numeric columns - create lags and rolling stats
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Lag features
            for lag in lag_periods:
                new_col = f"{col}_lag_{lag}"
                df[new_col] = df[col].shift(lag)
                metadata.append(
                    {
                        "name": new_col,
                        "source": col,
                        "type": TEMPORAL_FEATURES,
                        "transformation": f"lag_{lag}",
                        "lag_period": lag,
                    }
                )

            # Rolling statistics
            for window in rolling_windows:
                # Rolling mean
                new_col = f"{col}_rolling_mean_{window}"
                df[new_col] = df[col].rolling(window=window, min_periods=1).mean()
                metadata.append(
                    {
                        "name": new_col,
                        "source": col,
                        "type": TEMPORAL_FEATURES,
                        "transformation": f"rolling_mean_{window}",
                        "window_size": window,
                    }
                )

                # Rolling std
                new_col = f"{col}_rolling_std_{window}"
                df[new_col] = df[col].rolling(window=window, min_periods=1).std()
                metadata.append(
                    {
                        "name": new_col,
                        "source": col,
                        "type": TEMPORAL_FEATURES,
                        "transformation": f"rolling_std_{window}",
                        "window_size": window,
                    }
                )

    return df, metadata


def _generate_interaction_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str],
    max_interactions: int = 10,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Generate interaction features between columns.

    Creates:
    - Categorical x Categorical (cross product encoding)
    - Numeric x Numeric (multiplication, ratio)

    Args:
        df: Input DataFrame
        categorical_columns: Categorical columns for crossing
        numeric_columns: Numeric columns for interactions
        max_interactions: Maximum number of interactions to create

    Returns:
        Tuple of (transformed DataFrame, feature metadata list)
    """
    df = df.copy()
    metadata = []
    interaction_count = 0

    # Categorical x Categorical interactions
    cat_cols = [c for c in categorical_columns if c in df.columns]
    for i, col1 in enumerate(cat_cols):
        if interaction_count >= max_interactions:
            break
        for col2 in cat_cols[i + 1 :]:
            if interaction_count >= max_interactions:
                break

            new_col = f"{col1}_x_{col2}"
            df[new_col] = df[col1].astype(str) + "_" + df[col2].astype(str)
            metadata.append(
                {
                    "name": new_col,
                    "sources": [col1, col2],
                    "type": INTERACTION_FEATURES,
                    "transformation": "categorical_cross",
                }
            )
            interaction_count += 1

    # Numeric x Numeric interactions (top pairs by correlation)
    num_cols = [c for c in numeric_columns if c in df.columns]
    if len(num_cols) >= 2:
        # Create product features for top correlated pairs
        for i, col1 in enumerate(num_cols[:5]):  # Limit to top 5 numeric cols
            if interaction_count >= max_interactions:
                break
            for col2 in num_cols[i + 1 : 6]:  # Limit pairs
                if interaction_count >= max_interactions:
                    break

                # Product interaction
                new_col = f"{col1}_times_{col2}"
                df[new_col] = df[col1] * df[col2]
                metadata.append(
                    {
                        "name": new_col,
                        "sources": [col1, col2],
                        "type": INTERACTION_FEATURES,
                        "transformation": "product",
                    }
                )
                interaction_count += 1

                # Ratio interaction (with zero handling)
                if interaction_count < max_interactions:
                    new_col = f"{col1}_div_{col2}"
                    df[new_col] = df[col1] / (df[col2].replace(0, np.nan))
                    metadata.append(
                        {
                            "name": new_col,
                            "sources": [col1, col2],
                            "type": INTERACTION_FEATURES,
                            "transformation": "ratio",
                        }
                    )
                    interaction_count += 1

    return df, metadata


def _generate_domain_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Generate domain-specific features for pharma analytics.

    Creates features based on known pharma KPI patterns:
    - TRx/NRx ratios
    - Market share changes
    - HCP engagement scores
    - Regional performance indices

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (transformed DataFrame, feature metadata list)
    """
    df = df.copy()
    metadata = []
    columns_lower = {c.lower(): c for c in df.columns}

    # TRx/NRx ratio (if both present)
    trx_col = columns_lower.get("trx") or columns_lower.get("total_rx")
    nrx_col = columns_lower.get("nrx") or columns_lower.get("new_rx")

    if trx_col and nrx_col and trx_col in df.columns and nrx_col in df.columns:
        new_col = "trx_nrx_ratio"
        df[new_col] = df[trx_col] / (df[nrx_col].replace(0, np.nan))
        metadata.append(
            {
                "name": new_col,
                "sources": [trx_col, nrx_col],
                "type": DOMAIN_FEATURES,
                "transformation": "trx_nrx_ratio",
                "domain": "pharma_kpi",
            }
        )

        # Refill rate proxy
        new_col = "refill_rate"
        df[new_col] = (df[trx_col] - df[nrx_col]) / (df[trx_col].replace(0, np.nan))
        metadata.append(
            {
                "name": new_col,
                "sources": [trx_col, nrx_col],
                "type": DOMAIN_FEATURES,
                "transformation": "refill_rate",
                "domain": "pharma_kpi",
            }
        )

    # Market share (if market_share or share column exists)
    share_col = columns_lower.get("market_share") or columns_lower.get("share")
    if share_col and share_col in df.columns:
        # Share momentum (change)
        new_col = f"{share_col}_momentum"
        df[new_col] = df[share_col].diff()
        metadata.append(
            {
                "name": new_col,
                "source": share_col,
                "type": DOMAIN_FEATURES,
                "transformation": "momentum",
                "domain": "market",
            }
        )

    # Conversion rate (if visits and conversions exist)
    visits_col = columns_lower.get("visits") or columns_lower.get("hcp_visits")
    conversions_col = columns_lower.get("conversions") or columns_lower.get("converted")

    if visits_col and conversions_col:
        if visits_col in df.columns and conversions_col in df.columns:
            new_col = "conversion_rate"
            df[new_col] = df[conversions_col] / (df[visits_col].replace(0, np.nan))
            metadata.append(
                {
                    "name": new_col,
                    "sources": [visits_col, conversions_col],
                    "type": DOMAIN_FEATURES,
                    "transformation": "conversion_rate",
                    "domain": "sales",
                }
            )

    # HCP engagement score (if activity columns exist)
    activity_cols = [
        c
        for c in df.columns
        if any(kw in c.lower() for kw in ["call", "email", "sample", "activity"])
    ]
    if len(activity_cols) >= 2:
        new_col = "hcp_engagement_score"
        # Simple sum-based engagement score (normalized)
        engagement_sum = df[activity_cols].sum(axis=1)
        df[new_col] = (engagement_sum - engagement_sum.min()) / (
            engagement_sum.max() - engagement_sum.min() + 1e-10
        )
        metadata.append(
            {
                "name": new_col,
                "sources": activity_cols,
                "type": DOMAIN_FEATURES,
                "transformation": "engagement_score",
                "domain": "hcp",
            }
        )

    return df, metadata


def _generate_aggregate_features(
    df: pd.DataFrame,
    numeric_columns: List[str],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Generate aggregate statistical features.

    Creates:
    - Row-wise statistics across numeric columns
    - Percentile rankings

    Args:
        df: Input DataFrame
        numeric_columns: Numeric columns to aggregate

    Returns:
        Tuple of (transformed DataFrame, feature metadata list)
    """
    df = df.copy()
    metadata = []

    num_cols = [c for c in numeric_columns if c in df.columns]
    if len(num_cols) < 2:
        return df, metadata

    # Row-wise mean
    new_col = "numeric_mean"
    df[new_col] = df[num_cols].mean(axis=1)
    metadata.append(
        {
            "name": new_col,
            "sources": num_cols,
            "type": AGGREGATE_FEATURES,
            "transformation": "row_mean",
        }
    )

    # Row-wise std
    new_col = "numeric_std"
    df[new_col] = df[num_cols].std(axis=1)
    metadata.append(
        {
            "name": new_col,
            "sources": num_cols,
            "type": AGGREGATE_FEATURES,
            "transformation": "row_std",
        }
    )

    # Row-wise max
    new_col = "numeric_max"
    df[new_col] = df[num_cols].max(axis=1)
    metadata.append(
        {
            "name": new_col,
            "sources": num_cols,
            "type": AGGREGATE_FEATURES,
            "transformation": "row_max",
        }
    )

    # Row-wise range
    new_col = "numeric_range"
    df[new_col] = df[num_cols].max(axis=1) - df[num_cols].min(axis=1)
    metadata.append(
        {
            "name": new_col,
            "sources": num_cols,
            "type": AGGREGATE_FEATURES,
            "transformation": "row_range",
        }
    )

    return df, metadata


def _handle_generated_nans(
    df: pd.DataFrame,
    strategy: str = "median",
) -> pd.DataFrame:
    """Handle NaN values created by feature generation.

    Args:
        df: DataFrame with potential NaN values
        strategy: Fill strategy - "median", "mean", "zero", or "drop"

    Returns:
        DataFrame with NaN values handled
    """
    df = df.copy()

    if strategy == "median":
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
    elif strategy == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
    elif strategy == "zero":
        df = df.fillna(0)
    elif strategy == "drop":
        df = df.dropna()

    # Fill remaining NaN with 0 (for any edge cases)
    df = df.fillna(0)

    return df
