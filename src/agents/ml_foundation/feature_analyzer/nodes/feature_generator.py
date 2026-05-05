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

from src.agents.ml_foundation._pydantic_utils import preserve_audit_workflow_id

# 1B-M-4: temporal helpers (constants, detector, generator, split-marker
# round-trip) live in ``_temporal.py``. Re-export here so the existing public
# import path ``feature_generator._SPLIT_MARKER_COL`` etc. keeps working for
# tests and callers. The two split-marker constants are not referenced inside
# this module after the move (only the round-trip helpers consume them) so
# they use the explicit ``X as X`` re-export form to satisfy ruff F401.
from ._temporal import _SPLIT_MARKER_COL as _SPLIT_MARKER_COL  # noqa: F401
from ._temporal import _SPLIT_ROW_ID_COL as _SPLIT_ROW_ID_COL  # noqa: F401
from ._temporal import (
    _concat_with_split_markers,
    _detect_temporal_columns,
    _generate_temporal_features,
    _split_by_markers,
)

logger = logging.getLogger(__name__)


# Feature type constants. ``TEMPORAL_FEATURES`` moved to ``_temporal.py`` in
# 1B-M-4 — it is consumed only by ``_generate_temporal_features``.
INTERACTION_FEATURES = "interaction"
DOMAIN_FEATURES = "domain"
AGGREGATE_FEATURES = "aggregate"


@preserve_audit_workflow_id
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
            - entity_id_column: Column identifying the entity for groupby-aware
              lag/rolling (e.g. ``patient_id``). May also be set on
              ``feature_config["entity_id_column"]``.
            - event_timestamp_column: Column for chronological ordering inside
              each entity. May also live on ``feature_config``.
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

        # Capture original feature names BEFORE _concat_with_split_markers may
        # mutate state["X_train"] in place by adding dunder split markers (1B-M5).
        original_features = list(X_train.columns)

        # Track generated features
        generated_features: List[Dict[str, Any]] = []
        feature_metadata: Dict[str, Any] = {
            "temporal": [],
            "interaction": [],
            "domain": [],
            "aggregate": [],
        }

        # 1. Generate temporal features
        # Lag and rolling windows are entity-grouped and chronologically ordered,
        # so we MUST compute them on the concatenated train+val+test before each
        # split would otherwise truncate the per-entity history. We tag each row
        # with a split marker, run _generate_temporal_features ONCE on the full
        # frame, then split back on the marker. The per-row interaction/domain/
        # aggregate steps below do NOT need this round-trip — they're row-local.
        if feature_config.get("generate_temporal", True) and temporal_columns:
            entity_id_column = feature_config.get("entity_id_column") or state.get(
                "entity_id_column"
            )
            event_timestamp_column = feature_config.get("event_timestamp_column") or state.get(
                "event_timestamp_column"
            )
            # Block 1B contract: temporal feature generation REQUIRES entity
            # and timestamp columns. Silently falling back to naive shift is
            # exactly the regression Block 1B fixes, so we fail fast with a
            # message that points to the missing config rather than no-op.
            if not entity_id_column or not event_timestamp_column:
                missing = []
                if not entity_id_column:
                    missing.append("entity_id_column")
                if not event_timestamp_column:
                    missing.append("event_timestamp_column")
                raise ValueError(
                    "Temporal feature generation requires "
                    f"{', '.join(missing)}. Set on feature_config "
                    "(or directly on state) when temporal_columns is "
                    "non-empty, or disable temporal features via "
                    "feature_config['generate_temporal']=False. Lag/rolling "
                    "computed without per-entity grouping leaks across "
                    "patient histories — see Block 1B (#2)."
                )

            combined_df, split_index = _concat_with_split_markers(X_train, X_val, X_test)
            combined_df, temporal_meta = _generate_temporal_features(
                combined_df,
                temporal_columns,
                entity_id_column=entity_id_column,
                event_timestamp_column=event_timestamp_column,
                lag_periods=feature_config.get("lag_periods", [1, 7, 30]),
                rolling_windows=feature_config.get("rolling_windows", [7, 30]),
            )
            X_train, X_val, X_test = _split_by_markers(combined_df, split_index)
            feature_metadata["temporal"] = temporal_meta
            generated_features.extend(temporal_meta)

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

        # Get feature names (original_features captured pre-temporal block).
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


def _detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Detect categorical columns."""
    categorical_cols = []

    for col in df.columns:
        # Check if object/category type
        if df[col].dtype == "object" or isinstance(df[col].dtype, pd.CategoricalDtype):
            categorical_cols.append(col)
        # Check if low cardinality integer (likely categorical)
        elif pd.api.types.is_integer_dtype(df[col]):
            if df[col].nunique() < 20:  # Arbitrary threshold
                categorical_cols.append(col)

    return categorical_cols


def _detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Detect numeric columns suitable for aggregations."""
    return list(df.select_dtypes(include=[np.number]).columns)


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
    metadata: List[Dict[str, Any]] = []

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
