"""data_preparer frames -> model_trainer split contract (#2207 follow-up, codex r1 HIGH-1).

``MLFoundationPipeline.run`` handed the trainer ``input_data.get("train_data")`` — the
CALLER's pre-loaded splits — and never the frames the data_preparer stage had just
produced. On the retraining path (``execute_model_retraining`` -> ``run``) nothing
pre-loads splits, so ``ModelTrainerAgent`` received four empty dicts and
``split_loader`` failed on the first missing ``X`` before any training. The tier-0
harness never hit this because it builds ``{X, y, row_count}`` itself
(``scripts/run_tier0_test.py``). The data-prep Feast gate (the blocker in front of
this one) hid it.

This module builds the trainer's contract from the frames the data_preparer agent
returns (``train_df`` / ``validation_df`` / ``test_df`` / ``holdout_df`` — the loaded,
target-bearing frames; the transformer's ``X_*``/``y_*`` do not survive the state
schema): ``y`` is the scope's ``prediction_target`` column, ``X`` is everything else
minus the entity / date columns and datetime dtypes. Original row indices are kept —
the trainer's split validator detects leakage by comparing index sets, so resetting
them would produce false positives (same note as the harness). Categorical columns
are left in: ``model_trainer.nodes.preprocessor`` one-hot encodes them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SPLIT_KEYS = ("train_data", "validation_data", "test_data", "holdout_data")
_FRAME_FOR_SPLIT = {
    "train_data": "train",
    "validation_data": "validation",
    "test_data": "test",
    "holdout_data": "holdout",
}


def _empty_split() -> Dict[str, Any]:
    return {"X": pd.DataFrame(), "y": pd.Series(dtype=float), "row_count": 0}


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
    drop = [c for c in drop_columns if c]
    splits: Dict[str, Dict[str, Any]] = {}
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
                f"{frame_key} frame (columns: {list(df.columns)[:12]}{'...' if df.shape[1] > 12 else ''})"
            )
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        to_drop = [target_column] + [c for c in drop if c in df.columns] + datetime_cols
        X = df.drop(columns=list(dict.fromkeys(to_drop)))
        y = df[target_column]
        splits[split_key] = {"X": X, "y": y, "row_count": int(len(X))}
    logger.info(
        "Trainer splits from data_preparer frames: %s (target=%s)",
        {k: v["row_count"] for k, v in splits.items()},
        target_column,
    )
    return splits


def preloaded_splits(input_data: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The caller's pre-loaded splits when ALL four are present, else None."""
    if all(input_data.get(k) for k in SPLIT_KEYS):
        return {k: input_data[k] for k in SPLIT_KEYS}
    return None


__all__ = ["SPLIT_KEYS", "frames_to_trainer_splits", "preloaded_splits"]
