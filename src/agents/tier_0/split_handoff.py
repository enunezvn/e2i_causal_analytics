"""data_preparer frames -> model_trainer split contract (#2207 follow-up, codex r1 HIGH-1).

``MLFoundationPipeline.run`` handed the trainer ``input_data.get("train_data")`` — the
CALLER's pre-loaded splits — and never the frames the data_preparer stage had just
produced. On the retraining path (``execute_model_retraining`` -> ``run``) nothing
pre-loads splits, so ``ModelTrainerAgent`` received four empty dicts and
``split_loader`` failed on the first missing ``X`` before any training. The tier-0
harness never hit this because it builds ``{X, y, row_count}`` itself
(``scripts/run_tier0_test.py``). The data-prep Feast gate (the blocker in front of
this one) hid it.

What is handed over: the frames the data_preparer agent returns (``train_df`` /
``validation_df`` / ``test_df`` / ``holdout_df`` — the loaded, target-bearing frames
after every data_preparer node, including leakage remediation; the transformer's
``X_*``/``y_*`` do not survive the state schema and would predate remediation).
``y`` is the scope's ``prediction_target`` column; ``X`` is the frame minus:

- the target, the caller's ``drop_columns`` (entity / date columns, the scope's
  ``excluded_features``) and datetime dtypes;
- identifier-like columns: named ``id`` / ``*_id`` / ``*_hash`` / ``*_uuid`` and the
  split bookkeeping columns (``data_split``, ``split_config_id``, ``split_id``);
- every column the trainer's preprocessor would NOT encode and would pass through raw
  (``model_trainer.nodes.preprocessor._detect_feature_types``: numeric kept; object /
  Categorical kept only with ``nunique <= TRAINER_MAX_CATEGORIES`` (50); anything else —
  ``StringDtype``, high-cardinality strings, datetimes — is passthrough and fails the
  estimator or leaks identifiers, codex r2 HIGH-2 / r3 HIGH-3). Object columns whose
  distinct-value ratio exceeds ``IDENTIFIER_CARDINALITY_RATIO`` are dropped as
  identifiers even under the absolute cutoff. The scope's ``required_features`` is NOT
  used as an allowlist: without caller-supplied candidates it is a placeholder list
  (``scope_builder._define_required_features``).

Categorical columns are left in: ``model_trainer.nodes.preprocessor`` one-hot encodes
them. Every split gets a DISJOINT RangeIndex (codex r2 MED-3): the loaders reset each
split to ``0..n-1`` (``data_loader._split_by_column`` / ``data_splitter``), and the
trainer's ``_check_duplicate_indices`` compares index SETS across splits, so the
loader's indices would read as CRITICAL cross-split duplicates on every run.

Known, pre-existing limit NOT closed here (codex r2 HIGH-1, owner decision): the
Supabase-table route of ``data_loader`` is a temporal ``val_days`` / ``test_days``
split with no holdout unless an entity column is set, while ``split_enforcer``
requires a non-empty holdout and 60/20/10/10 ratios — a table-sourced retrain fails
there with that exact reason. File-sourced cohorts (``data_split`` column, the
documented trigger example) carry all four splits.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SPLIT_KEYS = ("train_data", "validation_data", "test_data", "holdout_data")
_FRAME_FOR_SPLIT = {
    "train_data": "train",
    "validation_data": "validation",
    "test_data": "test",
    "holdout_data": "holdout",
}
_SPLIT_BOOKKEEPING = {"data_split", "split_config_id", "split_id"}
_IDENTIFIER_SUFFIXES = ("_id", "_hash", "_uuid")
IDENTIFIER_CARDINALITY_RATIO = 0.5
# model_trainer.nodes.preprocessor._detect_feature_types: object/Categorical columns with
# more distinct values than this are neither encoded nor scaled — they pass through raw.
TRAINER_MAX_CATEGORIES = 50


def _empty_split() -> Dict[str, Any]:
    return {"X": pd.DataFrame(), "y": pd.Series(dtype=float), "row_count": 0}


def _looks_like_identifier(name: str) -> bool:
    lowered = str(name).lower()
    return (
        lowered == "id" or lowered.endswith(_IDENTIFIER_SUFFIXES) or lowered in _SPLIT_BOOKKEEPING
    )


def feature_columns_to_drop(
    train_df: pd.DataFrame,
    target_column: str,
    drop_columns: Iterable[str] = (),
) -> Dict[str, List[str]]:
    """``{reason: [columns]}`` to remove from X, decided on the TRAIN frame so every
    split keeps the same columns."""
    explicit = [c for c in drop_columns if c and c in train_df.columns and c != target_column]
    datetime_cols = [
        c for c in train_df.columns if pd.api.types.is_datetime64_any_dtype(train_df[c])
    ]
    named_ids = [c for c in train_df.columns if _looks_like_identifier(c)]
    n = max(len(train_df), 1)
    high_cardinality: List[str] = []
    unsupported: List[str] = []
    for c in train_df.columns:
        if c == target_column or c in named_ids or c in datetime_cols:
            continue
        s = train_df[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            continue
        is_trainer_categorical = pd.api.types.is_object_dtype(s) or isinstance(
            s.dtype, pd.CategoricalDtype
        )
        n_unique = s.nunique(dropna=True)
        if not is_trainer_categorical:
            unsupported.append(c)  # StringDtype & co: the preprocessor passes them through raw
        elif n_unique > TRAINER_MAX_CATEGORIES or n_unique / n > IDENTIFIER_CARDINALITY_RATIO:
            high_cardinality.append(c)
    return {
        "target": [target_column],
        "explicit": explicit,
        "datetime": datetime_cols,
        "identifier_named": named_ids,
        "identifier_cardinality": high_cardinality,
        "unsupported_dtype": unsupported,
    }


def frames_to_trainer_splits(
    frames: Mapping[str, Optional[pd.DataFrame]],
    target_column: Optional[str],
    *,
    drop_columns: Iterable[str] = (),
) -> Dict[str, Dict[str, Any]]:
    """``{train_data, validation_data, test_data, holdout_data}`` in the trainer's shape.

    ``frames`` is keyed ``train`` / ``validation`` / ``test`` / ``holdout`` (a missing or
    None holdout becomes an empty split — the trainer's enforcer decides what that
    means; train/validation/test must be present). Raises ``ValueError`` when the
    target column is unknown or absent from a frame — a retrain must never train on a
    frame whose label it cannot find.
    """
    if not target_column:
        raise ValueError(
            "cannot build trainer splits: scope_spec.prediction_target is not set "
            "(the pipeline's target_outcome never reached the scope)"
        )
    train_df = frames.get("train")
    if train_df is None or len(train_df) == 0:
        raise ValueError("cannot build trainer splits: data_preparer produced no train frame")
    if target_column not in train_df.columns:
        raise ValueError(
            f"cannot build trainer splits: target column {target_column!r} is not in the "
            f"train frame (columns: {list(train_df.columns)[:12]}"
            f"{'...' if train_df.shape[1] > 12 else ''})"
        )
    drop_plan = feature_columns_to_drop(train_df, target_column, drop_columns)
    to_drop = list(dict.fromkeys(c for cols in drop_plan.values() for c in cols))

    splits: Dict[str, Dict[str, Any]] = {}
    offset = 0
    for split_key, frame_key in _FRAME_FOR_SPLIT.items():
        df = frames.get(frame_key)
        if df is None or len(df) == 0:
            if frame_key == "holdout":
                splits[split_key] = _empty_split()
                continue
            raise ValueError(
                f"cannot build trainer splits: data_preparer produced no {frame_key} frame"
            )
        if target_column not in df.columns:
            raise ValueError(
                f"cannot build trainer splits: target column {target_column!r} is not in the "
                f"{frame_key} frame"
            )
        # Disjoint RangeIndex per split: the loaders reset every split to 0..n-1 and the
        # trainer's duplicate-index leakage check compares index sets across splits.
        index = pd.RangeIndex(start=offset, stop=offset + len(df))
        offset += len(df)
        X = df.drop(columns=[c for c in to_drop if c in df.columns]).set_index(index)
        y = df[target_column].set_axis(index)
        splits[split_key] = {"X": X, "y": y, "row_count": int(len(X))}
    logger.info(
        "Trainer splits from data_preparer frames: %s (target=%s; dropped from X: %s)",
        {k: v["row_count"] for k, v in splits.items()},
        target_column,
        {reason: cols for reason, cols in drop_plan.items() if cols and reason != "target"},
    )
    return splits


def preloaded_splits(input_data: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The caller's pre-loaded splits when ALL four are present, else None."""
    if all(input_data.get(k) for k in SPLIT_KEYS):
        return {k: input_data[k] for k in SPLIT_KEYS}
    return None


__all__ = [
    "IDENTIFIER_CARDINALITY_RATIO",
    "SPLIT_KEYS",
    "TRAINER_MAX_CATEGORIES",
    "feature_columns_to_drop",
    "frames_to_trainer_splits",
    "preloaded_splits",
]
