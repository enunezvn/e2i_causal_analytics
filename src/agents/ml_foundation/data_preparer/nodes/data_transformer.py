"""Data transformer node for data_preparer agent.

This node handles feature encoding, scaling, and missing value imputation.
Applies transformations consistently across train/val/test splits.
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler

from ..state import DataPreparerState

logger = logging.getLogger(__name__)


_EXCLUDE_COLUMNS_DEPRECATION_MESSAGE = (
    "scope_spec['exclude_columns'] is deprecated; use 'excluded_features' "
    "instead. Both keys are honored today, but 'exclude_columns' will be "
    "removed in a future tier-0 release."
)


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
        # The canonical scope key is ``prediction_target`` — it is what the
        # harness (run_tier0_test), scope_builder, baseline_computer and
        # sufficiency_check all set/read. The legacy ``target_column`` alias is
        # still honored as a fallback for older callers/fixtures. Reading ONLY
        # ``target_column`` (the prior bug) meant that on every real run — where
        # the scope sets ``prediction_target`` and never ``target_column`` — the
        # target was not separated here, so the binary target column was swept
        # through StandardScaler (mean-centred to ~0) and later misread by
        # baseline_computer/sufficiency_check as a zero-event cohort.
        target_column = scope_spec.get("prediction_target") or scope_spec.get("target_column")
        # `excluded_features` is the canonical scope-declared list; the
        # legacy `exclude_columns` runtime override is still honored for
        # backward compatibility, but emits a DeprecationWarning when
        # populated. Both lists are merged so callers in transition do not
        # silently lose excluded columns.
        legacy_exclude_columns = list(scope_spec.get("exclude_columns", []))
        if legacy_exclude_columns:
            warnings.warn(
                _EXCLUDE_COLUMNS_DEPRECATION_MESSAGE,
                DeprecationWarning,
                stacklevel=2,
            )
        exclude_columns = legacy_exclude_columns + list(scope_spec.get("excluded_features", []))
        scaling_method = scope_spec.get("scaling_method", "standard")
        # Issue #790: nominal categoricals default to ONE-HOT. The prior
        # ``"label"`` default integer-coded nominal categoricals (e.g.
        # ``payer: HMO=0, PPO=1, EPO=2``), imposing a false magnitude order.
        # Once integer-coded the columns look numeric, so the downstream
        # ``ModelTrainerPreprocessor`` (designed to one-hot object-dtype
        # columns) skips them and the LINEAR champion trains on ordinal codes —
        # degrading discrimination (faithful HCP-adoption run: AUC 0.777 ordinal
        # -> 0.803 one-hot, on merit). An explicit ``encoding_method="label"``
        # still forces the legacy integer encoding for all categoricals; the
        # ``ordinal_features`` allow-list keeps genuinely-ordered categoricals
        # (e.g. risk bands) integer-encoded even under the one-hot default.
        # ``or "onehot"`` (not ``.get(..., "onehot")``) so a scope that dumps
        # the ScopeSpec schema with ``encoding_method=None`` also gets one-hot.
        encoding_method = scope_spec.get("encoding_method") or "onehot"
        ordinal_features = list(scope_spec.get("ordinal_features") or [])
        imputation_strategy = scope_spec.get("imputation_strategy", "mean")
        datetime_features = scope_spec.get("extract_datetime_features", True)

        # Defense-in-depth pre-step (Codex pass-2 MEDIUM-2 + pass-3 MED-3
        # + pass-4 MED-4): scan for unhashable container columns BEFORE
        # any transformation so we can thread the cleaned (but
        # otherwise-untransformed) frames back into state. Downstream
        # nodes in the data_preparer graph that consume
        # ``state.get("train_df")`` — ``feast_registrar``,
        # ``compute_baseline_metrics``, ``finalize_output`` — would
        # otherwise read the ORIGINAL state's frame with list cells
        # intact and crash on their own ``nunique()``/``value_counts()``
        # calls. Mirrors loader-side ``_drop_unhashable_columns``.
        #
        # Pass-3 MED-3: scan EVERY split (train/val/test/holdout) and
        # take the UNION of unhashable cols. A column may be scalar in
        # train but list-typed in val/test (split skew on JSON-decoded
        # CSU/Optum), or a list-only column absent from train altogether.
        # The drop set must cover all splits, not just train.
        #
        # Pass-4 MED-4: the unhashable scan must run INDEPENDENTLY of
        # ``exclude_columns``. The transformation-exclusion path
        # (``excluded_features`` / ``exclude_columns``) only suppresses
        # encoding/scaling — it does NOT remove the column from the
        # frame. A list-typed col placed in ``excluded_features`` would
        # therefore bypass the safety drop and survive into X_* /
        # state frames, re-tripping the same ``nunique()`` crash in
        # ``model_trainer/nodes/preprocessor.py::_detect_feature_types``.
        # The safety drop is a hazard mitigation, not a transformation
        # policy — it must apply regardless of caller intent.
        def _unhashable_in(frame: pd.DataFrame | None) -> set[str]:
            if frame is None:
                return set()
            cols: set[str] = set()
            for col in frame.columns:
                if target_column and col == target_column:
                    # Target column is exempt — caller's responsibility.
                    continue
                if _column_has_unhashable_cells(frame[col]):
                    cols.add(col)
            return cols

        unhashable_set: set[str] = (
            _unhashable_in(train_df)
            | _unhashable_in(validation_df)
            | _unhashable_in(test_df)
            | _unhashable_in(holdout_df)
        )
        unhashable_cols = sorted(unhashable_set)

        # Stash pre-transformation cleaned frames for state replay.
        # These hold the original schema MINUS the unhashable cols, with
        # the target column preserved — preserving the canonical state
        # contract that ``train_df`` contains the target.
        state_train_df_cleaned = None
        state_val_df_cleaned = None
        state_test_df_cleaned = None
        state_holdout_df_cleaned = None

        if unhashable_cols:
            logger.warning(
                "Dropping %d non-encodable column(s) with unhashable cells "
                "(list/dict/set/tuple/ndarray): %s. These crash nunique() / "
                "LabelEncoder; mirroring data_loader._drop_unhashable_columns "
                "semantics so downstream data_preparer nodes (feast_registrar, "
                "baseline_computer, finalize_output) and the model_trainer "
                "preprocessor do not re-trip the same crash on X[col].nunique(). "
                "Route ingestion through _load_from_files to strip them upstream "
                "and avoid this warning.",
                len(unhashable_cols),
                unhashable_cols,
            )
            # Pre-transformation snapshot (originals + target, minus list cols)
            # — used by the downstream-state delta below.
            state_train_df_cleaned = train_df.drop(columns=unhashable_cols, errors="ignore")
            if validation_df is not None:
                state_val_df_cleaned = validation_df.drop(columns=unhashable_cols, errors="ignore")
            if test_df is not None:
                state_test_df_cleaned = test_df.drop(columns=unhashable_cols, errors="ignore")
            if holdout_df is not None:
                state_holdout_df_cleaned = holdout_df.drop(columns=unhashable_cols, errors="ignore")
            # Local working copies (used by transformation steps) also
            # need the list cols dropped — otherwise we re-trip nunique()
            # ourselves in the next _identify_column_types call.
            train_df = state_train_df_cleaned.copy()
            if state_val_df_cleaned is not None:
                validation_df = state_val_df_cleaned.copy()
            if state_test_df_cleaned is not None:
                test_df = state_test_df_cleaned.copy()
            if state_holdout_df_cleaned is not None:
                holdout_df = state_holdout_df_cleaned.copy()

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

        # Identify column types. The fourth slot is empty here because
        # we already stripped unhashable cols above (the early scan).
        # The signature still returns 4 values so callers in tests can
        # introspect the type buckets.
        numeric_cols, categorical_cols, datetime_cols, _ = _identify_column_types(
            train_df, exclude_columns
        )

        # Store transformation metadata
        transformations_applied = []
        encoders = {}
        scalers = {}
        imputers = {}
        if unhashable_cols:
            transformations_applied.append(
                {
                    "type": "drop_unhashable_columns",
                    "columns": unhashable_cols,
                    "reason": "object-dtype cells carrying list/dict/set/tuple/ndarray "
                    "values cannot be hashed by LabelEncoder/OneHotEncoder",
                }
            )

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
            transformations_applied.append(
                {
                    "type": "datetime_extraction",
                    "columns": datetime_cols,
                    "new_features": new_features,
                }
            )

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
                    test_df[numeric_cols] = numeric_imputer.transform(test_df[numeric_cols].values)
                if holdout_df is not None:
                    holdout_df[numeric_cols] = numeric_imputer.transform(
                        holdout_df[numeric_cols].values
                    )

                transformations_applied.append(
                    {
                        "type": "imputation",
                        "strategy": imputation_strategy,
                        "columns": numeric_cols,
                    }
                )

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
        # Issue #790: split the categoricals into ordinal (integer-encoded,
        # order preserved) and nominal (one-hot by default). An explicit
        # ``encoding_method="label"`` routes ALL categoricals through the legacy
        # integer encoder (back-compat); otherwise only the declared
        # ``ordinal_features`` stay integer-encoded and the rest one-hot expand.
        if categorical_cols:
            if encoding_method == "label":
                ordinal_cols = list(categorical_cols)
                nominal_cols: List[str] = []
            else:
                ordinal_cols = [c for c in categorical_cols if c in ordinal_features]
                nominal_cols = [c for c in categorical_cols if c not in ordinal_features]

            # --- Ordinal / explicitly-label-encoded columns (integer codes) ---
            # Fit on TRAIN only; ``_safe_label_encode`` absorbs unseen
            # categories at val/test/holdout time onto the sentinel id.
            for col in ordinal_cols:
                encoder = LabelEncoder()
                encoder.fit(train_df[col].astype(str).tolist())
                encoders[col] = encoder

                train_df[col] = encoder.transform(train_df[col].astype(str))
                if validation_df is not None:
                    validation_df[col] = _safe_label_encode(encoder, validation_df[col].astype(str))
                if test_df is not None:
                    test_df[col] = _safe_label_encode(encoder, test_df[col].astype(str))
                if holdout_df is not None:
                    holdout_df[col] = _safe_label_encode(encoder, holdout_df[col].astype(str))

            if ordinal_cols:
                transformations_applied.append(
                    {
                        "type": "encoding",
                        # "label" preserves the legacy metadata for explicit
                        # callers; declared ordinal_features under the one-hot
                        # default are recorded as "ordinal".
                        "method": "label" if encoding_method == "label" else "ordinal",
                        "columns": ordinal_cols,
                    }
                )

            # --- Nominal columns (one-hot) ---
            if nominal_cols:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoder.fit(train_df[nominal_cols])
                encoders["onehot"] = encoder

                # Get new column names
                new_cols = encoder.get_feature_names_out(nominal_cols)

                def _apply_onehot(frame: pd.DataFrame) -> pd.DataFrame:
                    encoded = pd.DataFrame(
                        encoder.transform(frame[nominal_cols]),
                        columns=new_cols,
                        index=frame.index,
                    )
                    return pd.concat([frame.drop(columns=nominal_cols), encoded], axis=1)

                train_df = _apply_onehot(train_df)
                if validation_df is not None:
                    validation_df = _apply_onehot(validation_df)
                if test_df is not None:
                    test_df = _apply_onehot(test_df)
                if holdout_df is not None:
                    holdout_df = _apply_onehot(holdout_df)

                transformations_applied.append(
                    {
                        "type": "encoding",
                        "method": "onehot",
                        "original_columns": nominal_cols,
                        "new_columns": list(new_cols),
                    }
                )

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

            if scaler is not None and len(train_df) > 0:
                # Fit on train (skip if train is empty)
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
                    test_df[current_numeric_cols] = scaler.transform(test_df[current_numeric_cols])
                if holdout_df is not None:
                    holdout_df[current_numeric_cols] = scaler.transform(
                        holdout_df[current_numeric_cols]
                    )

                transformations_applied.append(
                    {
                        "type": "scaling",
                        "method": scaling_method,
                        "columns": current_numeric_cols,
                    }
                )

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

        # Codex pass-2 MEDIUM-2: when we dropped unhashable cols, also
        # thread the cleaned (pre-transformation, target-preserved)
        # frames back into state under the canonical ``train_df``/
        # ``validation_df``/``test_df``/``holdout_df`` keys so downstream
        # nodes (feast_registrar, baseline_computer, finalize_output)
        # consume the post-drop schema. We surface the PRE-transformation
        # frames (the originals minus list cols) rather than the
        # encoded/scaled X_* frames — the canonical state contract is
        # that train_df mirrors raw schema, just with list cols stripped.
        if unhashable_cols:
            if state_train_df_cleaned is not None:
                updates["train_df"] = state_train_df_cleaned
            if state_val_df_cleaned is not None:
                updates["validation_df"] = state_val_df_cleaned
            if state_test_df_cleaned is not None:
                updates["test_df"] = state_test_df_cleaned
            if state_holdout_df_cleaned is not None:
                updates["holdout_df"] = state_holdout_df_cleaned

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


def _column_has_unhashable_cells(series: pd.Series) -> bool:
    """Return ``True`` if any non-null cell in an object-dtype series is an
    unhashable container (list/dict/set/tuple/ndarray).

    Mirrors the loader-side guard ``_drop_unhashable_columns`` in
    ``data_loader.py`` but applied per-column at transformer entry as
    defense-in-depth. Callers that bypass the file-loader path (preassembled
    DataFrames passed directly through ``state['train_df']``, alternate
    ingestion shapes, future ``data_source`` variants) still hit
    ``_identify_column_types`` — so the type-detection step must not crash on
    ``Series.nunique()`` when an object column carries list-typed cells.

    Empty / all-null object columns return ``False`` (benign for downstream).
    """
    if not pd.api.types.is_object_dtype(series.dtype):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    # Same scan strategy as data_loader._drop_unhashable_columns: pandas
    # object-dtype columns are not type-uniform, so sampling only iloc[0]
    # could miss a list/ndarray cell at a later row.
    return bool(
        non_null.map(lambda v: isinstance(v, (list, dict, set, frozenset, tuple, np.ndarray))).any()
    )


def _identify_column_types(
    df: pd.DataFrame, exclude_columns: List[str]
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Identify column types for transformation.

    Args:
        df: DataFrame to analyze
        exclude_columns: Columns to exclude from transformation

    Returns:
        Tuple of (numeric_cols, categorical_cols, datetime_cols,
        unhashable_cols). The fourth element lists object-dtype columns
        whose cells are unhashable containers — these are NOT encodable
        and the caller (``transform_data``) MUST drop them from all
        split frames before returning so downstream nodes (model_trainer
        ``_detect_feature_types`` in particular) do not re-trip the
        ``unhashable type: 'list'`` crash on ``X[col].nunique()``.

    Defense: object-dtype columns whose cells are unhashable containers
    (``list``/``dict``/``set``/``tuple``/``numpy.ndarray``) crash
    ``nunique()`` / ``value_counts()`` / ``LabelEncoder`` with
    ``TypeError: unhashable type: 'list'``. The file-loader path drops
    these columns via ``data_loader._drop_unhashable_columns`` before
    transform_data sees them; the transformer-side detection here is
    defense-in-depth for callers who bypass the loader (preassembled
    DataFrames passed directly via ``state['train_df']``, alternate
    ingestion shapes). Returning the names lets ``transform_data``
    perform the symmetric drop on all splits — mirroring loader
    semantics so downstream consumers (model_trainer preprocessor,
    feast registrar) see a clean encodable feature surface.
    """
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []
    unhashable_cols: list[str] = []

    for col in df.columns:
        if col in exclude_columns:
            continue

        dtype = df[col].dtype

        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_cols.append(col)
        elif pd.api.types.is_bool_dtype(dtype):
            # Route bool BEFORE numeric: mixing bool + int + float in a single
            # DataFrame slice forces pandas to fall back to object dtype on
            # ``.values`` because there is no safe common numpy dtype, which
            # crashes ``np.isnan`` in the imputation step. Bool is semantically
            # binary categorical anyway; the label encoder produces 0/1 — same
            # downstream effect as numeric treatment.
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        elif isinstance(dtype, pd.CategoricalDtype) or dtype == object:
            # Defense-in-depth: object columns carrying unhashable cells
            # (lists/dicts/ndarrays) crash ``nunique()`` below. Record the
            # column name so the caller drops it from all splits — leaving
            # the column in the frame would just shift the crash downstream
            # (model_trainer ``_detect_feature_types`` calls the same
            # ``nunique()`` on object columns).
            if _column_has_unhashable_cells(df[col]):
                unhashable_cols.append(col)
                continue

            # Check if it looks like a categorical
            n_unique = df[col].nunique()
            n_total = len(df)

            # If high cardinality (>50% unique), might not be categorical
            # Guard against empty dataframes to avoid division by zero
            if n_total == 0 or n_unique / n_total < 0.5 or n_unique < 50:
                categorical_cols.append(col)
            else:
                # Treat high cardinality as text, skip for now
                logger.warning(f"Column {col} has high cardinality ({n_unique} unique), skipping")

    return numeric_cols, categorical_cols, datetime_cols, unhashable_cols


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

            new_features.extend(
                [
                    f"{col}_year",
                    f"{col}_month",
                    f"{col}_day",
                    f"{col}_dayofweek",
                    f"{col}_hour",
                    f"{col}_is_weekend",
                ]
            )

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
