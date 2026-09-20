"""Shared validation for estimator design matrices."""

from __future__ import annotations

import pandas as pd


def numeric_design_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return a float design matrix, rejecting unencoded categorical columns."""
    unsupported = [
        column
        for column in frame.columns
        if not (
            pd.api.types.is_numeric_dtype(frame[column].dtype)
            or pd.api.types.is_bool_dtype(frame[column].dtype)
        )
    ]
    if unsupported:
        raise TypeError(
            f"{label} must be numeric or boolean after encoding; "
            f"unsupported columns={unsupported}"
        )
    return frame.astype(float, copy=False)
