"""Shared utilities for Real-World Data (RWD) converters.

Pure, dataset-agnostic helpers used by the RWD converter scripts:
  - ``scripts/convert_csu_rwd.py`` (CSU Excel → E2I JSON)
  - ``scripts/convert_optum_rwd.py`` (Optum parquet → E2I parquet per cohort)

Scope rule: only code that both converters could reasonably share lives
here. Dataset-shape-specific cohort/feature logic stays in the converter
scripts themselves. This file is designed to have zero side effects on
import.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- #
# Constants used by multiple converters                                         #
# ----------------------------------------------------------------------------- #

SPLIT_RATIOS: dict[str, float] = {
    "train": 0.60,
    "validation": 0.20,
    "test": 0.15,
    "holdout": 0.05,
}
TEMPORAL_GAP_DAYS = 7

AGE_BINS: list[int] = [0, 18, 35, 50, 65, 200]
AGE_LABELS: list[str] = ["<18", "18-34", "35-49", "50-65", "65+"]

INSURANCE_TYPE_MAP: dict[str, str] = {
    "COM": "Commercial",
    "MCR": "Medicare",
    "MCD": "Medicaid",
}

# US Census region mapping (3-digit ZIP prefix ranges). Identical table as the
# CSU converter so region assignments across converters stay consistent.
REGION_RANGES: dict[str, list[tuple[int, int]]] = {
    "northeast": [(10, 69), (100, 149)],
    "south": [
        (150, 196),
        (200, 268),
        (270, 349),
        (370, 385),
        (700, 799),
    ],
    "midwest": [(386, 427), (430, 499), (500, 599), (600, 658)],
    "west": [(660, 699), (800, 999)],
}


# ----------------------------------------------------------------------------- #
# Pure helpers                                                                  #
# ----------------------------------------------------------------------------- #


def generate_luhn_npi(obfuscated: str) -> str:
    """Generate a deterministic Luhn-valid 10-digit NPI from an obfuscated string.

    Algorithm: SHA-256 → first 9 hex digits mod 10 → Luhn check digit → 10-digit string.
    Safe for obfuscated vendor NPIs. Deterministic: same input always yields
    the same NPI.
    """
    digest = hashlib.sha256(obfuscated.encode("utf-8")).hexdigest()
    base_digits = [int(c, 16) % 10 for c in digest[:9]]

    total = 0
    for i, d in enumerate(reversed(base_digits)):
        if i % 2 == 0:
            doubled = d * 2
            total += doubled - 9 if doubled > 9 else doubled
        else:
            total += d
    check = (10 - (total % 10)) % 10
    base_digits.append(check)
    return "".join(str(d) for d in base_digits)


def patient_hash(patid: Any) -> str:
    """20-char hex hash for a patient id (deterministic, non-reversible)."""
    return hashlib.sha256(str(patid).encode("utf-8")).hexdigest()[:20]


def map_zipcode_to_region(zip_code: str | None) -> str | None:
    """Map a 5-digit ZIP to a US Census region via its 3-digit prefix.

    Returns None for missing/invalid input. Handles underscore-joined
    multi-zips (takes the first).
    """
    if not zip_code or not isinstance(zip_code, str):
        return None
    first_zip = zip_code.split("_")[0].strip()
    if not first_zip or len(first_zip) < 3:
        return None
    try:
        prefix = int(first_zip[:3])
    except ValueError:
        return None
    for region, ranges in REGION_RANGES.items():
        for lo, hi in ranges:
            if lo <= prefix <= hi:
                return region
    return None


def age_group(age: float | int | None) -> str | None:
    """Bin a numeric age into one of ``AGE_LABELS``. Negative / NaN → None."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return None
    age_int = int(age)
    if age_int < 0:
        return None
    for i, upper in enumerate(AGE_BINS[1:]):
        if age_int < upper:
            return AGE_LABELS[i]
    return AGE_LABELS[-1]


def insurance_type(bus: str | None) -> str | None:
    """Map the Optum/CSU ``bus`` field to a human label."""
    if bus is None or not isinstance(bus, str):
        return None
    return INSURANCE_TYPE_MAP.get(bus.strip().upper(), "Other")


def format_diagcode(code: str | None) -> str | None:
    """Format an ICD-10 code with dot insertion (e.g. 'L508' → 'L50.8')."""
    if code is None or not isinstance(code, str):
        return None
    code = code.strip().upper()
    if len(code) > 3 and "." not in code:
        return code[:3] + "." + code[3:]
    return code


def safe_date(val: Any) -> str | None:
    """Return ISO (YYYY-MM-DD) or None for various date-like inputs."""
    if val is None:
        return None
    if isinstance(val, pd.Timestamp):
        if pd.isna(val):
            return None
        return val.strftime("%Y-%m-%d")
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, str):
        return val[:10]
    return None


def safe_int(val: Any) -> int | None:
    """Coerce to int, tolerant of NaN / string / None."""
    if val is None:
        return None
    try:
        if isinstance(val, float) and np.isnan(val):
            return None
        return int(val)
    except (ValueError, TypeError):
        return None


def safe_float(val: Any) -> float | None:
    """Coerce to float, tolerant of NaN / string / None."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


# ----------------------------------------------------------------------------- #
# Chronological splitting                                                       #
# ----------------------------------------------------------------------------- #


def apply_chronological_split(
    records: list[dict[str, Any]],
    *,
    date_key: str = "journey_start_date",
    id_key: str = "patient_id",
    split_ratios: Mapping[str, float] = SPLIT_RATIOS,
) -> dict[str, Any]:
    """Assign a chronological ``data_split`` label to each record.

    Mutates each record in ``records`` by setting ``record["data_split"]``
    and returns a summary dict with the per-split boundary dates for use in
    a split registry.

    - Sorts records by ``record[date_key]`` (None dates → "train" fallback).
    - Assigns splits in order ``train → validation → test → holdout`` using
      the ratios given.
    - Patient-level isolation is preserved only if the caller already
      collapsed records to 1 per patient. For multi-event records, pass the
      per-patient primary event and propagate the assignment downstream.

    Returns:
        Dict with keys:
          - ``counts`` — dict[str, int] of records per split.
          - ``split_dates`` — dict with boundary ISO dates, keyed:
            ``data_start``, ``data_end``, ``train_end``, ``validation_end``,
            ``test_end``. Missing keys where applicable.
    """
    dated = [
        (rec, pd.to_datetime(rec.get(date_key), errors="coerce"))
        for rec in records
    ]
    # Split into dated vs undated
    dated_valid = [(rec, d) for rec, d in dated if pd.notna(d)]
    undated = [rec for rec, d in dated if pd.isna(d)]

    dated_valid.sort(key=lambda t: t[1])
    n = len(dated_valid)

    counts: dict[str, int] = dict.fromkeys(split_ratios, 0)
    split_dates: dict[str, Any] = {}

    if n == 0:
        for rec in records:
            rec["data_split"] = "train"
            counts["train"] += 1
        return {"counts": counts, "split_dates": split_dates}

    train_end = int(n * split_ratios["train"])
    val_end = train_end + int(n * split_ratios["validation"])
    test_end = val_end + int(n * split_ratios["test"])

    for i, (rec, _d) in enumerate(dated_valid):
        if i < train_end:
            label = "train"
        elif i < val_end:
            label = "validation"
        elif i < test_end:
            label = "test"
        else:
            label = "holdout"
        rec["data_split"] = label
        counts[label] = counts.get(label, 0) + 1

    for rec in undated:
        rec["data_split"] = "train"
        counts["train"] += 1

    split_dates["data_start"] = dated_valid[0][1]
    split_dates["data_end"] = dated_valid[-1][1]
    if train_end > 0:
        split_dates["train_end"] = dated_valid[train_end - 1][1]
    if val_end > 0:
        split_dates["validation_end"] = dated_valid[min(val_end - 1, n - 1)][1]
    if test_end > 0:
        split_dates["test_end"] = dated_valid[min(test_end - 1, n - 1)][1]

    return {"counts": counts, "split_dates": split_dates}


def build_split_registry(
    *,
    split_config_id: str,
    config_name: str,
    config_version: str,
    split_dates: Mapping[str, Any],
    split_ratios: Mapping[str, float] = SPLIT_RATIOS,
    temporal_gap_days: int = TEMPORAL_GAP_DAYS,
    created_at: str | None = None,
) -> list[dict[str, Any]]:
    """Build the canonical ``e2i_ml_v3_split_registry`` record.

    Emits a one-element list matching the schema used across the synthetic
    and CSU converters.
    """
    def _iso(v: Any) -> str | None:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.strftime("%Y-%m-%d")
        if isinstance(v, str):
            return v[:10]
        return None

    return [
        {
            "split_config_id": split_config_id,
            "config_name": config_name,
            "config_version": config_version,
            "train_ratio": split_ratios.get("train"),
            "validation_ratio": split_ratios.get("validation"),
            "test_ratio": split_ratios.get("test"),
            "holdout_ratio": split_ratios.get("holdout"),
            "data_start_date": _iso(split_dates.get("data_start")),
            "data_end_date": _iso(split_dates.get("data_end")),
            "train_end_date": _iso(split_dates.get("train_end")),
            "validation_end_date": _iso(split_dates.get("validation_end")),
            "test_end_date": _iso(split_dates.get("test_end")),
            "temporal_gap_days": temporal_gap_days,
            "patient_level_isolation": True,
            "split_strategy": "chronological",
            "is_active": True,
            "created_at": created_at or datetime.now().isoformat(),
        }
    ]


# ----------------------------------------------------------------------------- #
# Output writers                                                                #
# ----------------------------------------------------------------------------- #


def write_records(
    output_dir: Path | str,
    name: str,
    records: Sequence[Mapping[str, Any]],
    *,
    fmt: str = "parquet",
    index_cols: Sequence[str] | None = None,
) -> Path:
    """Write a list of record-dicts to ``{output_dir}/{name}.{fmt}``.

    ``fmt`` ∈ {"parquet", "json"}. Parquet format uses pyarrow via pandas;
    records with nested dicts/lists are preserved as-is (parquet encodes them
    as structs / lists, which is what our schema needs for ``icd_codes``,
    ``lab_values``, etc.).

    Returns the written path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        path = output_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(records), f, indent=2, default=str, ensure_ascii=False)
    elif fmt == "parquet":
        path = output_dir / f"{name}.parquet"
        df = pd.DataFrame(list(records))
        if index_cols:
            valid = [c for c in index_cols if c in df.columns]
            if valid:
                df = df.set_index(valid)
        df.to_parquet(path, index=bool(index_cols))
    else:
        raise ValueError(f"Unsupported fmt={fmt!r} (expected 'parquet' or 'json')")

    logger.info("Wrote %s (%d records)", path, len(records))
    return path


def write_attrition_report(
    output_dir: Path | str,
    steps: Iterable[tuple[str, int]],
    *,
    filename: str = "attrition_report.csv",
) -> Path:
    """Write an attrition log CSV: one row per filter step, columns step/count."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df = pd.DataFrame(list(steps), columns=["step", "count"])
    df.to_csv(path, index=False)
    logger.info("Wrote %s (%d steps)", path, len(df))
    return path


def write_data_dictionary(
    output_dir: Path | str,
    entries: Iterable[Mapping[str, Any]],
    *,
    filename: str = "data_dictionary.csv",
) -> Path:
    """Write a data dictionary CSV for the ingested cohort.

    Each entry should have keys: ``feature``, ``type``, ``source_table``,
    ``lookback_window``, ``null_rate``, ``notes``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df = pd.DataFrame(list(entries))
    df.to_csv(path, index=False)
    logger.info("Wrote %s (%d features)", path, len(df))
    return path
