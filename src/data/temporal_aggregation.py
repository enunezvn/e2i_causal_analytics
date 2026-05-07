"""Mandatory temporal aggregation API.

Phase 3 of `ml-leakage-holistic-fix`. Every event-derived feature MUST go
through this function. The pre-existing pattern of optionally windowing
(`if self.lookback_days is None: return df` / `_apply_lookback_window`) is
forbidden for new code: callers are required to specify `window_days` as a
positive integer.

Why: the pre-existing pattern was opt-in, so authors could forget. The result
was features like `journey_duration_days` (CSU 2026-05-07, single-feature
AUC=0.689) that derived end-of-journey indicators without windowing — leaking
post-prediction-time information into the feature set.

Contract:
- `events` is a long-form table of timestamped events.
- `anchors` is a per-entity anchor (typically `index_date`).
- The function returns one row per anchor with each requested aggregation
  computed over events in `(anchor - window_days, anchor]`.
- Missing entities are filled with the agg-specific neutral value (0 for sum,
  count, nunique; NaN for mean/min/max).
- Required parameters are keyword-only to prevent accidental positional drift.
"""

from __future__ import annotations

from typing import Dict, Literal, Mapping

import pandas as pd

AggFunc = Literal["sum", "mean", "max", "min", "count", "nunique"]

_NEUTRAL: Mapping[AggFunc, float | int] = {
    "sum": 0,
    "count": 0,
    "nunique": 0,
    "mean": float("nan"),
    "min": float("nan"),
    "max": float("nan"),
}


def temporal_aggregation(
    events: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    anchor_col: str,
    event_date_col: str,
    group_col: str,
    window_days: int,
    agg: Dict[str, AggFunc],
) -> pd.DataFrame:
    """Aggregate events within `(anchor - window_days, anchor]` per group.

    Args:
        events: Long-form event table (one row per event). Must contain
            `group_col` and `event_date_col`.
        anchors: Per-entity anchor table (one row per group). Must contain
            `group_col` and `anchor_col`.
        anchor_col: Name of the timestamp column in `anchors`.
        event_date_col: Name of the timestamp column in `events`.
        group_col: Name of the grouping key (e.g., patient_id) present in both.
        window_days: Required positive integer. Events within `(anchor -
            window_days, anchor]` are included; events outside are excluded.
        agg: Mapping of column name (in `events`) to aggregation function.
            Each entry yields an output column named `f"{col}_{func}"`.

    Returns:
        DataFrame with one row per anchor. Columns: `group_col` plus one
        column per `agg` entry. Missing groups are filled with the
        agg-specific neutral value.

    Raises:
        ValueError: If `window_days < 1`, required columns missing, or `agg`
            references a column not present in `events`.
        TypeError: If timestamp columns are not datetime-like.
    """
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1; got {window_days}")
    for col in (anchor_col, group_col):
        if col not in anchors.columns:
            raise ValueError(f"anchors missing required column: {col}")
    for col in (event_date_col, group_col):
        if col not in events.columns:
            raise ValueError(f"events missing required column: {col}")
    for col, func in agg.items():
        if col not in events.columns:
            raise ValueError(f"agg references column not in events: {col}")
        if func not in _NEUTRAL:
            raise ValueError(f"unsupported agg function: {func}")

    if not pd.api.types.is_datetime64_any_dtype(anchors[anchor_col]):
        raise TypeError(f"anchors[{anchor_col!r}] must be datetime-like")
    if not pd.api.types.is_datetime64_any_dtype(events[event_date_col]):
        raise TypeError(f"events[{event_date_col!r}] must be datetime-like")

    # The contract says "one row per anchor"; duplicate group_col rows would
    # silently produce duplicated agg rows on the left-join below (line 117),
    # all carrying the same aggregated value (which is computed once per group
    # by the groupby on line 111). That's a silent-correctness bug — not what
    # callers would expect from "one row per anchor". Refuse rather than
    # producing a misleading result.
    duplicates = anchors[group_col].duplicated()
    if duplicates.any():
        n_dup = int(duplicates.sum())
        raise ValueError(
            f"anchors[{group_col!r}] has {n_dup} duplicate value(s); the API "
            f"contract requires one row per group. If you need per-event "
            f"anchoring (e.g. multiple time-windowed snapshots per patient), "
            f"call this function once per snapshot anchor table."
        )

    # Inner-join anchors and events on group_col so each event picks up its
    # entity's anchor; events for entities without anchors are dropped.
    merged = events.merge(anchors[[group_col, anchor_col]], on=group_col, how="inner")
    delta_days = (merged[anchor_col] - merged[event_date_col]).dt.days
    in_window = (delta_days >= 0) & (delta_days < window_days)
    windowed = merged.loc[in_window]

    output_cols = {f"{col}_{func}": (col, func) for col, func in agg.items()}

    if windowed.empty:
        out = anchors[[group_col]].copy()
        for output_col, (_src_col, func) in output_cols.items():
            out[output_col] = _NEUTRAL[func]
        return out

    grouped = windowed.groupby(group_col).agg(agg)
    grouped.columns = [f"{c}_{f}" for c, f in agg.items()]
    grouped = grouped.reset_index()

    # Left-join so every anchor entity gets a row (with neutral fill for
    # entities that had no in-window events).
    out = anchors[[group_col]].merge(grouped, on=group_col, how="left")
    for output_col, (_src_col, func) in output_cols.items():
        out[output_col] = out[output_col].fillna(_NEUTRAL[func])
    return out
