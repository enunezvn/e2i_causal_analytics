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
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

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
    dated = [(rec, pd.to_datetime(rec.get(date_key), errors="coerce")) for rec in records]
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


# ----------------------------------------------------------------------------- #
# NPPES NPI taxonomy lookup (issue #154)                                        #
# ----------------------------------------------------------------------------- #
#
# CMS NPPES (National Plan and Provider Enumeration System) is a free public
# registry of all US healthcare providers, indexed by 10-digit NPI. It is
# distributed as:
#
#   (a) a monthly bulk CSV dump (~10 GB) at
#       https://download.cms.gov/nppes/NPI_Files.html — ingested into the
#       `npi_taxonomy` Postgres table by ``src.tasks.nppes_tasks`` (see
#       migration 034). The bulk dump is the steady-state source.
#
#   (b) a live JSON-over-HTTPS API at
#       https://npiregistry.cms.hhs.gov/api/?number=<NPI>&version=2.1
#       — fallback for NPIs missing from the local cache. CMS imposes a
#       ~200 req/min rate limit per the public API page.
#
# This module provides ``lookup_npi(npi)`` which prefers the local cache and
# falls back to the live API. Production code injects a callable cache loader
# (e.g. a Postgres-backed function) via ``set_npi_cache_loader``; tests can
# inject an in-memory fixture loader. If no loader is registered the helper
# transparently degrades to API-only behavior (with rate-limit awareness) so
# unit tests that just need a single record don't have to stand up Postgres.

NPPES_API_BASE_URL = "https://npiregistry.cms.hhs.gov/api/"
NPPES_API_VERSION = "2.1"
NPPES_API_RATE_LIMIT_PER_MIN = 200  # CMS published cap
NPPES_API_DEFAULT_TIMEOUT_S = 10.0

# NUCC taxonomy codes used by downstream consumers (issue #154 §5, §6). These
# are surfaced as module-level constants so callers don't drift apart on
# string literals.
NUCC_SPECIALTY_PHARMACY = "3336S0011X"
NUCC_MAIL_ORDER_PHARMACY = "3336M0002X"
NUCC_HOME_INFUSION_PHARMACY = "3336H0001X"

PHARMACY_CHANNEL_CODES: tuple[str, ...] = (
    NUCC_SPECIALTY_PHARMACY,
    NUCC_MAIL_ORDER_PHARMACY,
    NUCC_HOME_INFUSION_PHARMACY,
)

# Coarse academic-medical-center / safety-net facility taxonomy prefixes (NUCC
# group "Hospitals" 282N / 281P; specific subclasses surface academic vs FQHC
# vs Critical Access). Used by §6 site-of-care typing.
ACADEMIC_MEDICAL_CENTER_CODES: tuple[str, ...] = ("282N00000X",)  # General Acute Care
CRITICAL_ACCESS_HOSPITAL_CODES: tuple[str, ...] = ("282NC0060X",)
FQHC_CODES: tuple[str, ...] = ("261QF0400X",)
RURAL_HEALTH_CLINIC_CODES: tuple[str, ...] = ("261QR1300X",)
IHS_CODES: tuple[str, ...] = ("282NR1301X",)
VA_MEDICAL_CENTER_CODES: tuple[str, ...] = ("261QM2500X",)


@dataclass(frozen=True)
class NppesTaxonomy:
    """A single NUCC taxonomy entry attached to an NPI.

    NPPES providers can hold multiple taxonomies; ``primary=True`` marks
    the provider's self-declared primary specialty.
    """

    code: str
    desc: str | None = None
    primary: bool = False
    license: str | None = None
    state: str | None = None


@dataclass(frozen=True)
class NppesAddress:
    """Structured practice-location address from NPPES.

    Empty strings are normalized to ``None`` so callers can use truthiness
    without re-validating each field.
    """

    address_1: str | None = None
    address_2: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    country_code: str | None = None


@dataclass(frozen=True)
class NppesRecord:
    """Normalized NPPES record returned by ``lookup_npi``.

    Field semantics intentionally mirror the columns of ``npi_taxonomy``
    (migration 034) so callers can interchangeably consume bulk-dump rows
    or live-API responses.
    """

    npi: str
    entity_type: str | None = None
    enumeration_date: date | None = None
    last_updated_nppes: date | None = None
    taxonomies: tuple[NppesTaxonomy, ...] = field(default_factory=tuple)
    practice_address: NppesAddress | None = None
    parent_organization_legal_name: str | None = None
    organization_legal_name: str | None = None
    sole_proprietor: bool | None = None
    first_name: str | None = None
    last_name: str | None = None
    source: str = "api_fallback"

    @property
    def primary_taxonomy(self) -> NppesTaxonomy | None:
        """Return the NPPES-declared primary taxonomy if set, else the first
        entry, else ``None`` if the record carries no taxonomies."""
        for t in self.taxonomies:
            if t.primary:
                return t
        return self.taxonomies[0] if self.taxonomies else None

    def years_since_enumeration(self, *, today: date | None = None) -> int | None:
        """Whole years between ``enumeration_date`` and ``today`` (default
        ``date.today()``). Used to derive ``years_experience`` for HCP
        profiles. Returns None if the enumeration date is missing or in the
        future."""
        if self.enumeration_date is None:
            return None
        ref = today or date.today()
        delta = ref - self.enumeration_date
        if delta.days < 0:
            return None
        return delta.days // 365


# Module-level cache loader hook. Callers register a function that takes an
# NPI string and returns Optional[NppesRecord]. A None return means cache
# miss; the helper will then fall back to the live API.
_NPI_CACHE_LOADER: Optional[Any] = None
_NPI_API_RATE_LIMITER_LAST_RESET: float = 0.0
_NPI_API_RATE_LIMITER_COUNT: int = 0


def set_npi_cache_loader(loader: Any) -> None:
    """Register the cache loader for ``lookup_npi``.

    ``loader`` is a callable ``(npi: str) -> Optional[NppesRecord]``. The
    production loader (registered in ``src.tasks.nppes_tasks``) reads from
    the ``npi_taxonomy`` Postgres table; tests can register an in-memory
    dict-backed loader for deterministic behavior without standing up the
    database. Pass ``None`` to clear the registration.
    """
    global _NPI_CACHE_LOADER
    _NPI_CACHE_LOADER = loader


def get_npi_cache_loader() -> Any:
    """Return the current cache loader (for tests / introspection)."""
    return _NPI_CACHE_LOADER


def _parse_nppes_date(val: Any) -> date | None:
    """NPPES dates arrive as ``MM/DD/YYYY`` strings from the API and as
    ``YYYY-MM-DD`` from the bulk dump. Tolerate both."""
    if val is None or val == "":
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str):
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(val.strip(), fmt).date()
            except ValueError:
                continue
    return None


def _none_if_empty(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s if s else None
    return val


def parse_nppes_api_result(payload: Mapping[str, Any], npi: str) -> NppesRecord | None:
    """Parse a single result dict from the NPPES live-API ``results[]``
    array into a normalized ``NppesRecord``.

    The API shape is documented at
    https://npiregistry.cms.hhs.gov/api/demo — the parser tolerates missing
    fields and returns None only when the result is structurally unusable
    (no NPI / no basic block).
    """
    if not isinstance(payload, Mapping):
        return None

    basic = payload.get("basic") or {}
    if not isinstance(basic, Mapping):
        basic = {}

    entity_type_raw = payload.get("enumeration_type") or basic.get("enumeration_type")
    # API returns "NPI-1" / "NPI-2"; bulk dump returns "1" / "2".
    entity_type: str | None = None
    if isinstance(entity_type_raw, str):
        if entity_type_raw.endswith("1"):
            entity_type = "1"
        elif entity_type_raw.endswith("2"):
            entity_type = "2"

    taxonomies_in = payload.get("taxonomies") or []
    taxonomies_out: list[NppesTaxonomy] = []
    if isinstance(taxonomies_in, list):
        for t in taxonomies_in:
            if not isinstance(t, Mapping):
                continue
            code_val = _none_if_empty(t.get("code"))
            if not code_val:
                continue
            taxonomies_out.append(
                NppesTaxonomy(
                    code=str(code_val),
                    desc=_none_if_empty(t.get("desc")),
                    primary=bool(t.get("primary", False)),
                    license=_none_if_empty(t.get("license")),
                    state=_none_if_empty(t.get("state")),
                )
            )

    addresses_in = payload.get("addresses") or []
    practice_address: NppesAddress | None = None
    if isinstance(addresses_in, list):
        # Prefer purpose=LOCATION; fall back to first address.
        chosen: Mapping[str, Any] | None = None
        for a in addresses_in:
            if not isinstance(a, Mapping):
                continue
            if str(a.get("address_purpose", "")).upper() == "LOCATION":
                chosen = a
                break
        if chosen is None and addresses_in and isinstance(addresses_in[0], Mapping):
            chosen = addresses_in[0]
        if chosen is not None:
            practice_address = NppesAddress(
                address_1=_none_if_empty(chosen.get("address_1")),
                address_2=_none_if_empty(chosen.get("address_2")),
                city=_none_if_empty(chosen.get("city")),
                state=_none_if_empty(chosen.get("state")),
                postal_code=_none_if_empty(chosen.get("postal_code")),
                country_code=_none_if_empty(chosen.get("country_code")),
            )

    sole_prop_raw = basic.get("sole_proprietor")
    sole_prop: bool | None
    if isinstance(sole_prop_raw, bool):
        sole_prop = sole_prop_raw
    elif isinstance(sole_prop_raw, str):
        normalized = sole_prop_raw.strip().upper()
        if not normalized:
            sole_prop = None  # empty / whitespace → unknown, not False
        elif normalized in {"YES", "Y", "TRUE", "1"}:
            sole_prop = True
        elif normalized in {"NO", "N", "FALSE", "0"}:
            sole_prop = False
        else:
            sole_prop = None  # unrecognized → unknown, not False
    else:
        sole_prop = None

    return NppesRecord(
        npi=str(npi),
        entity_type=entity_type,
        enumeration_date=_parse_nppes_date(basic.get("enumeration_date")),
        last_updated_nppes=_parse_nppes_date(basic.get("last_updated")),
        taxonomies=tuple(taxonomies_out),
        practice_address=practice_address,
        parent_organization_legal_name=_none_if_empty(basic.get("parent_organization_legal_name")),
        organization_legal_name=_none_if_empty(basic.get("organization_name")),
        sole_proprietor=sole_prop,
        first_name=_none_if_empty(basic.get("first_name")),
        last_name=_none_if_empty(basic.get("last_name")),
        source="api_fallback",
    )


def _is_valid_npi(npi: Any) -> bool:
    """Conservative NPI well-formedness check (10 digits)."""
    if npi is None:
        return False
    s = str(npi).strip()
    return len(s) == 10 and s.isdigit()


def _api_rate_limit_check() -> None:
    """Soft, in-process rate limiter — blocks (briefly) only when the
    per-minute cap is reached. Best-effort: process-restart wipes the
    counter. CMS limit is 200/min so this is generous for sync converter
    runs."""
    import time

    global _NPI_API_RATE_LIMITER_LAST_RESET, _NPI_API_RATE_LIMITER_COUNT
    now = time.monotonic()
    if now - _NPI_API_RATE_LIMITER_LAST_RESET >= 60.0:
        _NPI_API_RATE_LIMITER_LAST_RESET = now
        _NPI_API_RATE_LIMITER_COUNT = 0
    if _NPI_API_RATE_LIMITER_COUNT >= NPPES_API_RATE_LIMIT_PER_MIN:
        wait = 60.0 - (now - _NPI_API_RATE_LIMITER_LAST_RESET)
        if wait > 0:
            logger.warning("NPPES rate limit reached; sleeping %.2fs", wait)
            time.sleep(wait)
        _NPI_API_RATE_LIMITER_LAST_RESET = time.monotonic()
        _NPI_API_RATE_LIMITER_COUNT = 0
    _NPI_API_RATE_LIMITER_COUNT += 1


def _fetch_nppes_via_api_sync(
    npi: str,
    *,
    timeout_s: float = NPPES_API_DEFAULT_TIMEOUT_S,
    max_retries: int = 3,
) -> NppesRecord | None:
    """Synchronous live-API fallback. Returns None on 4xx/5xx/timeout.

    Uses ``httpx.Client`` (already a dep). Retries with exponential backoff
    only on 429 + 5xx; 4xx other than 429 short-circuits. Tests should
    instead patch ``lookup_npi``'s cache loader to avoid hitting the
    network — this code path is exercised by an integration test that is
    skipped when ``NPPES_API_ALLOW_LIVE`` is unset.
    """
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a project dep
        logger.warning("httpx unavailable; cannot fall back to NPPES live API")
        return None

    import time

    params = {"number": str(npi), "version": NPPES_API_VERSION}
    backoff_s = 1.0
    for attempt in range(max_retries):
        _api_rate_limit_check()
        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.get(NPPES_API_BASE_URL, params=params)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            logger.warning("NPPES API transport error (attempt %d): %s", attempt + 1, exc)
            time.sleep(backoff_s)
            backoff_s *= 2
            continue

        if resp.status_code in (429,) or 500 <= resp.status_code < 600:
            logger.warning(
                "NPPES API status %d (attempt %d); backing off %.1fs",
                resp.status_code,
                attempt + 1,
                backoff_s,
            )
            time.sleep(backoff_s)
            backoff_s *= 2
            continue
        if resp.status_code != 200:
            logger.warning("NPPES API status %d for NPI=%s; giving up", resp.status_code, npi)
            return None

        try:
            data = resp.json()
        except (ValueError, json.JSONDecodeError):
            logger.warning("NPPES API returned non-JSON for NPI=%s", npi)
            return None

        results = data.get("results") if isinstance(data, Mapping) else None
        if not results:
            return None
        first = results[0]
        return parse_nppes_api_result(first, npi)

    logger.warning("NPPES API exhausted %d retries for NPI=%s", max_retries, npi)
    return None


def lookup_npi(
    npi: str,
    *,
    use_api_fallback: bool | None = None,
) -> NppesRecord | None:
    """Resolve an NPI to an ``NppesRecord`` using the local cache, falling
    back to the live CMS API on cache miss.

    Parameters
    ----------
    npi
        10-digit NPI string. Non-conforming inputs return ``None``.
    use_api_fallback
        Override for the global default. When ``None`` (default), the value
        is read from the ``NPPES_API_FALLBACK`` env var (default: "1").
        Set ``False`` in offline / CI contexts that should not hit the
        public CMS API.

    Returns
    -------
    NppesRecord | None
        Normalized record, or ``None`` if the NPI is invalid, the cache
        has no entry, and the API fallback is disabled or returns no
        result.

    Notes
    -----
    Production wiring registers a Postgres-backed cache loader via
    ``set_npi_cache_loader`` at converter-startup. Without a loader the
    helper still works but always misses the cache, which is fine for
    one-off CLI inspection but not for bulk converter runs (use the
    Celery refresh task to pre-populate first).
    """
    if not _is_valid_npi(npi):
        return None
    npi_str = str(npi).strip()

    if _NPI_CACHE_LOADER is not None:
        try:
            cached = _NPI_CACHE_LOADER(npi_str)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("NPI cache loader raised for NPI=%s: %s", npi_str, exc)
            cached = None
        if cached is not None:
            return cached

    if use_api_fallback is None:
        use_api_fallback = os.environ.get("NPPES_API_FALLBACK", "1") not in {"0", "false", "False"}

    if not use_api_fallback:
        return None

    return _fetch_nppes_via_api_sync(npi_str)


def bulk_lookup_npis(
    npis: Iterable[str],
    *,
    use_api_fallback: bool | None = None,
) -> dict[str, NppesRecord]:
    """Resolve many NPIs at once, deduplicating + skipping invalid inputs.

    Returns a dict keyed by NPI string. NPIs that resolve to ``None``
    (invalid + cache-miss + API-disabled, or API returned nothing) are
    omitted from the result so callers can iterate over only the
    successfully-resolved subset.
    """
    out: dict[str, NppesRecord] = {}
    seen: set[str] = set()
    for raw in npis:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        rec = lookup_npi(s, use_api_fallback=use_api_fallback)
        if rec is not None:
            out[s] = rec
    return out
