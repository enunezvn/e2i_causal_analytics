"""Date parsing and comparison for ``leakage_detector``'s temporal checks.

Split out of ``leakage_detector.py`` (module-size ratchet, #1991) when #2294
made these helpers fail closed: every way a date column could reach a
comparison without being examined — coercion to ``NaT``, a numeric column
read as nanoseconds, an offset stripped instead of converted — now raises,
and ``check_temporal_leakage`` records the raise as an incomplete audit.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


def _parse_configured_dates(df: Any, col: str, epoch_units: Optional[Dict[str, str]] = None) -> Any:
    """Parse a date column for a temporal comparison, as NAIVE timestamps.

    Refuses every way a column could be turned into "clean" without being
    examined (#2294), raising so the caller records an incomplete audit:

    * ``errors="coerce"`` maps an unparseable value to ``NaT`` and ``NaT`` rows
      drop out of every comparison, so an unparseable value — possibly the
      leaking row — vanished (codex r2, r5). Strings are parsed as ISO 8601
      value by value, and ANY value that exists but does not parse makes the
      audit unverifiable. A column with no values at all has nothing to leak.
    * A bare number is ambiguous: pandas reads it as NANOseconds, so an
      epoch-seconds 2025 date became 1970 and never looked "after split_date"
      (codex r3), and no magnitude rule can recover the unit near 1970 (codex
      r4). A numeric column is parsed ONLY with a unit declared in
      ``scope_spec.epoch_units``; otherwise it is unverifiable.

    Time zones: tz-aware values (including offset strings that span a DST
    change and so parse to an object series) are converted to UTC and made
    naive; naive values stay naive. ``_naive_utc`` applies the same rule to the
    reference date, so both sides of every comparison are normalised alike.
    """
    values = df[col]
    n_values = int(values.notna().sum())
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        unit = (epoch_units or {}).get(col)
        if unit is None:
            if n_values == 0:
                return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
            raise ValueError(
                f"temporal check unverifiable: '{col}' is numeric with no declared epoch unit "
                "(declare it in scope_spec.epoch_units)"
            )
        dates = pd.to_datetime(values, unit=unit, errors="coerce")
    else:
        # ISO 8601 per value, not one format inferred from the first value: a
        # "2024-12-31" first row made "2025-02-01T00:00:00" (the leaking row)
        # coerce to NaT (codex r5).
        dates = pd.to_datetime(values, errors="coerce", format="ISO8601")
        if dates.dtype == object:
            dates = pd.to_datetime(values, errors="coerce", format="ISO8601", utc=True)
    # Any value that exists but did not parse could be the leaking row.
    n_unparsed = int((values.notna() & dates.isna()).sum())
    if n_unparsed:
        raise ValueError(
            f"temporal check unverifiable: '{col}' has {n_unparsed} unparseable values"
        )
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    return dates


def _naive_utc(moment: Any) -> Any:
    """A reference timestamp normalised like ``_parse_configured_dates``."""
    ts = pd.Timestamp(moment)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def _check_date_ordering(
    df: Any, event_col: str, target_col: str, epoch_units: Optional[Dict[str, str]] = None
) -> tuple:
    """Check if event dates occur after target dates.

    Raises rather than returning ``(0, 0.0)`` on failure: a zero count reads as
    "no temporal leakage" (#2294). The caller's ``except`` turns the failure
    into a "Temporal leakage check incomplete" issue, which blocks.
    """
    event_dates = _parse_configured_dates(df, event_col, epoch_units)
    target_dates = _parse_configured_dates(df, target_col, epoch_units)

    valid_mask = event_dates.notna() & target_dates.notna()
    leakage_mask = valid_mask & (event_dates > target_dates)

    leakage_count = leakage_mask.sum()
    leakage_pct = (leakage_count / len(df)) * 100 if len(df) > 0 else 0

    return leakage_count, leakage_pct


def _check_future_dates(
    df: Any, col: str, reference_date: datetime, epoch_units: Optional[Dict[str, str]] = None
) -> tuple:
    """Check for dates after a reference date.

    Raises on failure for the same reason as ``_check_date_ordering``.
    """
    dates = _parse_configured_dates(df, col, epoch_units)
    valid_mask = dates.notna()

    future_mask = valid_mask & (dates > _naive_utc(reference_date))
    future_count = future_mask.sum()
    future_pct = (future_count / len(df)) * 100 if len(df) > 0 else 0

    return future_count, future_pct


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime."""
    try:
        result = pd.to_datetime(date_str).to_pydatetime()
        return result if isinstance(result, datetime) else None
    except Exception:
        return None


def _detect_date_columns(
    df: Any,
    exclude: Optional[List[str]] = None,
    epoch_units: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Auto-detect date columns in DataFrame.

    A numeric column is a date only when its epoch unit is declared: by name
    alone (``time_on_therapy`` matches ``time_``) it is far more likely a
    duration, and ``pd.to_datetime`` "parses" any integer as nanoseconds, so
    guessing would either hide it (1970) or block every such run (#2294).
    """
    exclude = exclude or []
    epoch_units = epoch_units or {}
    date_cols = []

    for col in df.columns:
        if col in exclude:
            continue

        if pd.api.types.is_datetime64_any_dtype(df[col]) or col in epoch_units:
            date_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue

        date_patterns = ["_date", "_time", "_at", "_timestamp", "date_", "time_"]
        if any(pattern in col.lower() for pattern in date_patterns):
            try:
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    # Same ISO 8601 parser as the audit itself: a column it
                    # cannot parse is not auto-detected (it would only be
                    # unverifiable), whereas a CONFIGURED one blocks.
                    parsed = pd.to_datetime(sample, errors="coerce", format="ISO8601", utc=True)
                    if parsed.notna().sum() > len(sample) * 0.5:
                        date_cols.append(col)
            except Exception:
                pass

    return date_cols
