"""Temporal feature generation helpers for the feature_analyzer node.

These helpers handle (a) detecting temporal columns, (b) generating temporal
features (lag, rolling stats, date parts) with strict per-entity grouping, and
(c) the split-aware round-trip via marker columns that lets temporal generation
run on the concatenated train+val+test frame without losing per-split row
identity. Extracted from ``feature_generator.py`` in 1B-M-4.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Feature type constant used by ``_generate_temporal_features`` to tag metadata
# entries. Lives here (not in ``feature_generator.py``) because it is only
# consumed by the temporal generator; the sibling INTERACTION/DOMAIN/AGGREGATE
# constants stay in ``feature_generator.py`` next to their generators.
TEMPORAL_FEATURES = "temporal"


# Internal split-membership marker column. Lives only inside generate_features
# while the train/val/test rows are concatenated for entity-grouped lag/rolling
# computation; stripped before each split is returned to the caller.
_SPLIT_MARKER_COL = "__feature_gen_split__"


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


def _generate_temporal_features(
    df: pd.DataFrame,
    temporal_columns: List[str],
    entity_id_column: str,
    event_timestamp_column: str,
    lag_periods: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Generate temporal features from time-series columns.

    Creates:
    - Lag features (shift by N periods, entity-grouped)
    - Rolling statistics (mean, std over window, entity-grouped)
    - Date part extraction (if datetime)

    Lag and rolling windows are computed PER ENTITY. The DataFrame is sorted
    by ``(entity_id, event_timestamp)`` before grouping so the resulting
    shifts honor chronological order within each entity. Without grouping,
    lag at row 0 of entity B would erroneously pull entity A's tail value —
    that is exactly the regression Block 1B fixes.

    Callers should pass the *concatenated* train+val+test frame; otherwise
    the last row of train would never see val's first row even within the
    same entity, defeating the point of grouping.

    Args:
        df: Input DataFrame (typically the concatenated train+val+test).
        temporal_columns: Columns to generate temporal features for.
        entity_id_column: Column identifying the entity (e.g. ``patient_id``).
            REQUIRED. Must exist in ``df``. Datetime date-part extraction
            does not actually need this, but the contract still requires it
            so callers can never accidentally drop the per-entity grouping.
        event_timestamp_column: Column for chronological ordering inside
            each entity. REQUIRED. Must exist in ``df``.
        lag_periods: Lag periods to create (default ``[1, 7, 30]``).
        rolling_windows: Rolling window sizes (default ``[7, 30]``).

    Returns:
        Tuple of (transformed DataFrame, feature metadata list).

    Raises:
        ValueError: If ``entity_id_column`` or ``event_timestamp_column`` is
            empty, or if either column is missing from ``df``. Block 1B
            tightens this contract so a future caller cannot silently
            re-introduce naive cross-entity shift.
    """
    # Strict contract: both grouping keys must be supplied AND present.
    # Empty / None values would silently re-enable cross-entity leakage.
    if not entity_id_column:
        raise ValueError(
            "_generate_temporal_features requires a non-empty "
            "entity_id_column. Lag/rolling without per-entity grouping "
            "leaks across entities — see Block 1B (#2)."
        )
    if not event_timestamp_column:
        raise ValueError(
            "_generate_temporal_features requires a non-empty "
            "event_timestamp_column for chronological ordering."
        )
    if entity_id_column not in df.columns:
        raise ValueError(
            f"entity_id_column={entity_id_column!r} not found in DataFrame; "
            f"available columns: {list(df.columns)}"
        )
    if event_timestamp_column not in df.columns:
        raise ValueError(
            f"event_timestamp_column={event_timestamp_column!r} not found "
            f"in DataFrame; available columns: {list(df.columns)}"
        )

    if rolling_windows is None:
        rolling_windows = [7, 30]
    if lag_periods is None:
        lag_periods = [1, 7, 30]
    df = df.copy()
    metadata: List[Dict[str, Any]] = []

    # Sort by (entity, event_timestamp) once, up front, so all subsequent
    # groupby().shift / rolling() calls see chronologically ordered groups.
    df = df.sort_values([entity_id_column, event_timestamp_column], kind="mergesort").reset_index(
        drop=True
    )

    for col in temporal_columns:
        if col not in df.columns:
            continue

        # Handle datetime columns - extract date parts (row-local, no grouping)
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
            # Lag features (entity-grouped). group_keys=False keeps the
            # original index alignment so the assignment back to df[new_col]
            # doesn't re-introduce the entity id as a level.
            for lag in lag_periods:
                new_col = f"{col}_lag_{lag}"
                df[new_col] = df.groupby(entity_id_column, group_keys=False)[col].shift(lag)
                metadata.append(
                    {
                        "name": new_col,
                        "source": col,
                        "type": TEMPORAL_FEATURES,
                        "transformation": f"lag_{lag}",
                        "lag_period": lag,
                        "entity_id_column": entity_id_column,
                    }
                )

            # Rolling statistics (entity-grouped).
            for window in rolling_windows:
                rolled = df.groupby(entity_id_column, group_keys=False)[col].rolling(
                    window=window, min_periods=1
                )

                # Rolling mean
                new_col = f"{col}_rolling_mean_{window}"
                rolled_mean = rolled.mean()
                # GroupBy.rolling adds the group key as an index level;
                # reset_index(level=0, drop=True) realigns to the row index.
                if isinstance(rolled_mean.index, pd.MultiIndex):
                    rolled_mean = rolled_mean.reset_index(level=0, drop=True)
                df[new_col] = rolled_mean
                metadata.append(
                    {
                        "name": new_col,
                        "source": col,
                        "type": TEMPORAL_FEATURES,
                        "transformation": f"rolling_mean_{window}",
                        "window_size": window,
                        "entity_id_column": entity_id_column,
                    }
                )

                # Rolling std
                new_col = f"{col}_rolling_std_{window}"
                rolled_std = rolled.std()
                if isinstance(rolled_std.index, pd.MultiIndex):
                    rolled_std = rolled_std.reset_index(level=0, drop=True)
                df[new_col] = rolled_std
                metadata.append(
                    {
                        "name": new_col,
                        "source": col,
                        "type": TEMPORAL_FEATURES,
                        "transformation": f"rolling_std_{window}",
                        "window_size": window,
                        "entity_id_column": entity_id_column,
                    }
                )

    return df, metadata


_SPLIT_ROW_ID_COL = "__feature_gen_row_id__"


def _concat_with_split_markers(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame],
    X_test: Optional[pd.DataFrame],
) -> Tuple[
    pd.DataFrame,
    Dict[str, Tuple[pd.Index, np.ndarray]],
]:
    """Concatenate splits, tagging each row with its split-membership marker.

    Each split's original row index is preserved in the returned map so that
    ``_split_by_markers`` can restore the caller's original row ordering even
    after ``_generate_temporal_features`` sorts by ``(entity, timestamp)`` and
    resets the row index.

    The combined frame uses a synthetic monotonic ``_SPLIT_ROW_ID_COL`` to give
    every row a unique identifier independent of split-local pandas indices
    (which can collide across splits — both train and val often start at 0).

    Args:
        X_train: Training split (required).
        X_val: Validation split (optional).
        X_test: Test split (optional).

    Returns:
        Tuple of:
            - combined DataFrame with ``_SPLIT_MARKER_COL`` and
              ``_SPLIT_ROW_ID_COL`` injected.
            - map ``split_name -> (original_index, row_ids)``. ``row_ids`` are
              the synthetic identifiers carried inside the combined frame for
              this split's rows; ``original_index`` is what each row's pandas
              index used to be on the caller's input frame.

    Raises:
        ValueError: If any input frame already contains a column named
            ``_SPLIT_MARKER_COL`` or ``_SPLIT_ROW_ID_COL``. The dunder names
            are deliberately unlikely to collide with caller columns, but a
            collision would silently overwrite caller data and produce
            mis-routed splits in ``_split_by_markers``.

    Notes:
        Block 1B-M5 dropped the per-split defensive ``piece.copy()`` to avoid
        a full materialise of every input split at RWD scale. Two contracts
        are now caller-visible and MUST be respected:

        1. **Inputs are mutated in place.** This function appends
           ``_SPLIT_MARKER_COL`` and ``_SPLIT_ROW_ID_COL`` directly to
           ``X_train`` / ``X_val`` / ``X_test``. Callers who later try to
           reuse these frames as "untouched" will see the dunder columns. The
           reserved-name guard above ensures we never overwrite real caller
           columns, only add the markers.
        2. **The returned combined frame is NOT defensively copied.** Its
           pre-existing columns may share memory with the input splits.
           Callers MUST NOT mutate returned columns in place; they should
           reindex via ``_SPLIT_ROW_ID_COL`` (see ``_split_by_markers``) or
           assign new columns rather than overwriting existing values. The
           split-aware test harness (``TestConcatWithSplitMarkersMemoryContract``
           and round-trip tests in this module) catches violations.
    """
    # Block 1B-M1: refuse to clobber caller columns even if their names
    # happen to match our internal markers. Cheaper to fail loud than to
    # silently scramble the round-trip.
    reserved = {_SPLIT_MARKER_COL, _SPLIT_ROW_ID_COL}
    for split_name, frame in (
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
    ):
        if frame is None:
            continue
        clash = reserved.intersection(frame.columns)
        if clash:
            raise ValueError(
                f"{split_name} contains reserved internal column(s) "
                f"{sorted(clash)}; rename caller columns or remove them "
                "before passing to feature generation."
            )

    pieces: List[pd.DataFrame] = []
    split_meta: Dict[str, Tuple[pd.Index, np.ndarray]] = {}

    next_row_id = 0

    def _tag(piece: pd.DataFrame, split_name: str) -> pd.DataFrame:
        # Block 1B-M5: no defensive copy — see Notes in the docstring. We
        # mutate ``piece`` in place by adding the dunder marker columns. The
        # reserved-name guard above guarantees these names don't already
        # exist on caller frames, so no real data is overwritten.
        nonlocal next_row_id
        n = len(piece)
        row_ids = np.arange(next_row_id, next_row_id + n, dtype=np.int64)
        next_row_id += n
        piece[_SPLIT_MARKER_COL] = split_name
        piece[_SPLIT_ROW_ID_COL] = row_ids
        split_meta[split_name] = (piece.index.copy(), row_ids)
        return piece

    pieces.append(_tag(X_train, "train"))
    if X_val is not None:
        pieces.append(_tag(X_val, "val"))
    if X_test is not None:
        pieces.append(_tag(X_test, "test"))

    # ignore_index=True is fine — we use _SPLIT_ROW_ID_COL for round-tripping.
    # copy=False keeps the memory-sharing contract documented in Notes; pandas
    # may still copy when consolidating non-aligned dtypes, but for the common
    # case of homogeneous numeric splits the combined frame's blocks alias the
    # input blocks.
    combined = pd.concat(pieces, axis=0, ignore_index=True, copy=False)
    return combined, split_meta


def _split_by_markers(
    combined_df: pd.DataFrame,
    split_meta: Dict[str, Tuple[pd.Index, np.ndarray]],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Split a combined DataFrame back into (train, val, test) using markers.

    Restores each split to its original row ordering (the index captured at
    concatenation time) and strips the internal marker columns.

    Args:
        combined_df: DataFrame produced by ``_concat_with_split_markers`` and
            transformed in place by per-row generators.
        split_meta: Map of ``split_name -> (original_index, row_ids)`` produced
            by ``_concat_with_split_markers``.

    Returns:
        Tuple of (X_train, X_val, X_test). val/test may be None when absent.
    """
    feature_cols = [
        c for c in combined_df.columns if c not in (_SPLIT_MARKER_COL, _SPLIT_ROW_ID_COL)
    ]

    def _restore(split_name: str) -> Optional[pd.DataFrame]:
        if split_name not in split_meta:
            return None
        original_index, row_ids = split_meta[split_name]
        # Pull the rows for this split via the synthetic row id, then put
        # them back in their original order via reindex on the same id.
        subset = combined_df.set_index(_SPLIT_ROW_ID_COL).loc[row_ids, feature_cols]
        subset.index = original_index
        return subset

    train_df = _restore("train")
    val_df = _restore("val")
    test_df = _restore("test")
    # train is always present — typing reflects that.
    assert train_df is not None
    return train_df, val_df, test_df
