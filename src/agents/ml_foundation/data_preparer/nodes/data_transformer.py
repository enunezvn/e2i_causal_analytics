"""Data transformer node for data_preparer agent.

This node handles feature encoding, scaling, and missing value imputation.
Applies transformations consistently across train/val/test splits.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from ..state import DataPreparerState

logger = logging.getLogger(__name__)


async def transform_data(state: DataPreparerState) -> Dict[str, Any]:
    """Transform data with encoding, scaling, and imputation.

    This node:
    1. Identifies feature types (numeric, categorical, datetime)
    2. Applies missing value imputation
    3. Encodes categorical variables
    4. Scales numeric features
    5. Extracts datetime features
    6. Applies transformations consistently to all splits

    CRITICAL: Fit transformers on TRAIN only, apply to val/test.

    Args:
        state: Current agent state

    Returns:
        Updated state with transformed data
    """
    start_time = datetime.now()
    experiment_id = state.get("experiment_id", "unknown")
    logger.info(f"Starting data transformation for experiment {experiment_id}")

    try:
        train_df = state.get("train_df")
        validation_df = state.get("validation_df")
        test_df = state.get("test_df")
        holdout_df = state.get("holdout_df")

        if train_df is None:
            raise ValueError("train_df not found in state")

        # Get configuration from scope_spec
        scope_spec = state.get("scope_spec", {})
        target_column = scope_spec.get("target_column")
        exclude_columns = scope_spec.get("exclude_columns", [])
        scaling_method = scope_spec.get("scaling_method", "standard")
        encoding_method = scope_spec.get("encoding_method", "label")
        imputation_strategy = scope_spec.get("imputation_strategy", "mean")
        datetime_features = scope_spec.get("extract_datetime_features", True)

        # Separate target from features if specified
        target_train = None
        target_val = None
        target_test = None
        target_holdout = None

        if target_column and target_column in train_df.columns:
            target_train = train_df[target_column].copy()
            train_df = train_df.drop(columns=[target_column])

            if validation_df is not None and target_column in validation_df.columns:
                target_val = validation_df[target_column].copy()
                validation_df = validation_df.drop(columns=[target_column])

            if test_df is not None and target_column in test_df.columns:
                target_test = test_df[target_column].copy()
                test_df = test_df.drop(columns=[target_column])

            if holdout_df is not None and target_column in holdout_df.columns:
                target_holdout = holdout_df[target_column].copy()
                holdout_df = holdout_df.drop(columns=[target_column])

        # Identify column types
        numeric_cols, categorical_cols, datetime_cols = _identify_column_types(
            train_df, exclude_columns
        )

        # Store transformation metadata
        transformations_applied = []
        encoders = {}
        scalers = {}
        imputers = {}

        # === DATETIME FEATURE EXTRACTION ===
        if datetime_features and datetime_cols:
            train_df, new_features = _extract_datetime_features(train_df, datetime_cols)
            if validation_df is not None:
                validation_df, _ = _extract_datetime_features(validation_df, datetime_cols)
            if test_df is not None:
                test_df, _ = _extract_datetime_features(test_df, datetime_cols)
            if holdout_df is not None:
                holdout_df, _ = _extract_datetime_features(holdout_df, datetime_cols)

            # Update column lists
            numeric_cols.extend(new_features)
            transformations_applied.append({
                "type": "datetime_extraction",
                "columns": datetime_cols,
                "new_features": new_features,
            })

        # === MISSING VALUE IMPUTATION ===
        # Numeric imputation
        if numeric_cols:
            numeric_imputer = SimpleImputer(strategy=imputation_strategy)

            # Fit on train
            train_numeric = train_df[numeric_cols].values
            if np.isnan(train_numeric).any():
                train_df[numeric_cols] = numeric_imputer.fit_transform(train_numeric)
                imputers["numeric"] = numeric_imputer

                # Apply to other splits
                if validation_df is not None:
                    validation_df[numeric_cols] = numeric_imputer.transform(
                        validation_df[numeric_cols].values
                    )
                if test_df is not None:
                    test_df[numeric_cols] = numeric_imputer.transform(
                        test_df[numeric_cols].values
                    )
                if holdout_df is not None:
                    holdout_df[numeric_cols] = numeric_imputer.transform(
                        holdout_df[numeric_cols].values
                    )

                transformations_applied.append({
                    "type": "imputation",
                    "strategy": imputation_strategy,
                    "columns": numeric_cols,
                })

        # Categorical imputation (mode)
        if categorical_cols:
            for col in categorical_cols:
                if train_df[col].isnull().any():
                    mode_value = train_df[col].mode()
                    if len(mode_value) > 0:
                        fill_value = mode_value.iloc[0]
                    else:
                        fill_value = "unknown"

                    train_df[col] = train_df[col].fillna(fill_value)
                    if validation_df is not None:
                        validation_df[col] = validation_df[col].fillna(fill_value)
                    if test_df is not None:
                        test_df[col] = test_df[col].fillna(fill_value)
                    if holdout_df is not None:
                        holdout_df[col] = holdout_df[col].fillna(fill_value)

        # === CATEGORICAL ENCODING ===
        if categorical_cols:
            if encoding_method == "label":
                for col in categorical_cols:
                    encoder = LabelEncoder()
                    # Fit on all unique values across splits to handle unseen values
                    all_values = train_df[col].astype(str).tolist()
                    encoder.fit(all_values)
                    encoders[col] = encoder

                    train_df[col] = encoder.transform(train_df[col].astype(str))

                    if validation_df is not None:
                        validation_df[col] = _safe_label_encode(
                            encoder, validation_df[col].astype(str)
                        )
                    if test_df is not None:
                        test_df[col] = _safe_label_encode(
                            encoder, test_df[col].astype(str)
                        )
                    if holdout_df is not None:
                        holdout_df[col] = _safe_label_encode(
                            encoder, holdout_df[col].astype(str)
                        )

                transformations_applied.append({
                    "type": "encoding",
                    "method": "label",
                    "columns": categorical_cols,
                })

            elif encoding_method == "onehot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoder.fit(train_df[categorical_cols])
                encoders["onehot"] = encoder

                # Get new column names
                new_cols = encoder.get_feature_names_out(categorical_cols)

                # Transform each split
                train_encoded = pd.DataFrame(
                    encoder.transform(train_df[categorical_cols]),
                    columns=new_cols,
                    index=train_df.index,
                )
                train_df = train_df.drop(columns=categorical_cols)
                train_df = pd.concat([train_df, train_encoded], axis=1)

                if validation_df is not None:
                    val_encoded = pd.DataFrame(
                        encoder.transform(validation_df[categorical_cols]),
                        columns=new_cols,
                        index=validation_df.index,
                    )
                    validation_df = validation_df.drop(columns=categorical_cols)
                    validation_df = pd.concat([validation_df, val_encoded], axis=1)

                if test_df is not None:
                    test_encoded = pd.DataFrame(
                        encoder.transform(test_df[categorical_cols]),
                        columns=new_cols,
                        index=test_df.index,
                    )
                    test_df = test_df.drop(columns=categorical_cols)
                    test_df = pd.concat([test_df, test_encoded], axis=1)

                if holdout_df is not None:
                    holdout_encoded = pd.DataFrame(
                        encoder.transform(holdout_df[categorical_cols]),
                        columns=new_cols,
                        index=holdout_df.index,
                    )
                    holdout_df = holdout_df.drop(columns=categorical_cols)
                    holdout_df = pd.concat([holdout_df, holdout_encoded], axis=1)

                transformations_applied.append({
                    "type": "encoding",
                    "method": "onehot",
                    "original_columns": categorical_cols,
                    "new_columns": list(new_cols),
                })

        # === NUMERIC SCALING ===
        # Update numeric_cols after potential one-hot encoding
        current_numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

        if current_numeric_cols:
            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = None

            if scaler is not None:
                # Fit on train
                train_df[current_numeric_cols] = scaler.fit_transform(
                    train_df[current_numeric_cols]
                )
                scalers["main"] = scaler

                # Apply to other splits
                if validation_df is not None:
                    validation_df[current_numeric_cols] = scaler.transform(
                        validation_df[current_numeric_cols]
                    )
                if test_df is not None:
                    test_df[current_numeric_cols] = scaler.transform(
                        test_df[current_numeric_cols]
                    )
                if holdout_df is not None:
                    holdout_df[current_numeric_cols] = scaler.transform(
                        holdout_df[current_numeric_cols]
                    )

                transformations_applied.append({
                    "type": "scaling",
                    "method": scaling_method,
                    "columns": current_numeric_cols,
                })

        # Calculate transformation duration
        transform_duration = (datetime.now() - start_time).total_seconds()

        # Prepare output - rename to X_train, X_val, etc. as per contract
        updates = {
            "X_train": train_df,
            "X_val": validation_df,
            "X_test": test_df,
            "X_holdout": holdout_df,
            "y_train": target_train,
            "y_val": target_val,
            "y_test": target_test,
            "y_holdout": target_holdout,
            "transformations_applied": transformations_applied,
            "encoders": encoders,
            "scalers": scalers,
            "imputers": imputers,
            "feature_columns": list(train_df.columns),
            "transform_duration_seconds": transform_duration,
        }

        logger.info(
            f"Data transformation completed: "
            f"{len(transformations_applied)} transformations, "
            f"{len(train_df.columns)} features, "
            f"duration={transform_duration:.2f}s"
        )

        return updates

    except Exception as e:
        logger.error(f"Data transformation failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_type": "transformation_error",
            "blocking_issues": [f"Data transformation failed: {str(e)}"],
        }


def _identify_column_types(
    df: pd.DataFrame, exclude_columns: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """Identify column types for transformation.

    Args:
        df: DataFrame to analyze
        exclude_columns: Columns to exclude from transformation

    Returns:
        Tuple of (numeric_cols, categorical_cols, datetime_cols)
    """
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []

    for col in df.columns:
        if col in exclude_columns:
            continue

        dtype = df[col].dtype

        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        elif pd.api.types.is_categorical_dtype(dtype) or dtype == object:
            # Check if it looks like a categorical
            n_unique = df[col].nunique()
            n_total = len(df)

            # If high cardinality (>50% unique), might not be categorical
            # Guard against empty dataframes to avoid division by zero
            if n_total == 0 or n_unique / n_total < 0.5 or n_unique < 50:
                categorical_cols.append(col)
            else:
                # Treat high cardinality as text, skip for now
                logger.warning(
                    f"Column {col} has high cardinality ({n_unique} unique), skipping"
                )

    return numeric_cols, categorical_cols, datetime_cols


def _extract_datetime_features(
    df: pd.DataFrame, datetime_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract features from datetime columns.

    Args:
        df: DataFrame to transform
        datetime_cols: Datetime columns to process

    Returns:
        Tuple of (transformed_df, new_feature_names)
    """
    df = df.copy()
    new_features = []

    for col in datetime_cols:
        if col not in df.columns:
            continue

        try:
            dt_col = pd.to_datetime(df[col], errors="coerce")

            # Extract features
            df[f"{col}_year"] = dt_col.dt.year
            df[f"{col}_month"] = dt_col.dt.month
            df[f"{col}_day"] = dt_col.dt.day
            df[f"{col}_dayofweek"] = dt_col.dt.dayofweek
            df[f"{col}_hour"] = dt_col.dt.hour if dt_col.dt.hour.notna().any() else 0
            df[f"{col}_is_weekend"] = (dt_col.dt.dayofweek >= 5).astype(int)

            new_features.extend([
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_dayofweek",
                f"{col}_hour",
                f"{col}_is_weekend",
            ])

            # Drop original datetime column
            df = df.drop(columns=[col])

        except Exception as e:
            logger.warning(f"Could not extract features from {col}: {e}")

    return df, new_features


def _safe_label_encode(encoder: LabelEncoder, values: pd.Series) -> np.ndarray:
    """Safely encode values, handling unseen categories.

    Args:
        encoder: Fitted LabelEncoder
        values: Values to encode

    Returns:
        Encoded values (unseen values get max_label + 1)
    """
    # Get the classes the encoder knows
    known_classes = set(encoder.classes_)
    default_value = len(encoder.classes_)  # Unseen category value

    encoded = np.zeros(len(values), dtype=int)
    for i, val in enumerate(values):
        if val in known_classes:
            encoded[i] = encoder.transform([val])[0]
        else:
            encoded[i] = default_value

    return encoded
