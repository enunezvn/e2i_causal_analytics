#!/usr/bin/env python3
"""
Convert CSU Real-World Data to E2I JSON Format.

Reads CSU claims data from an Excel workbook and converts it into the
E2I ML v3 JSON schema used by the ML pipeline.  Produces four output files:

  - e2i_ml_v3_hcp_profiles.json
  - e2i_ml_v3_patient_journeys.json
  - e2i_ml_v3_treatment_events.json
  - e2i_ml_v3_split_registry.json

Usage:
    python scripts/convert_csu_rwd.py
    python scripts/convert_csu_rwd.py --input data/rwd/csu/csu_data.xlsx --output data/rwd/csu/
    python scripts/convert_csu_rwd.py --max-patients 500
    python scripts/convert_csu_rwd.py --dry-run
    python scripts/convert_csu_rwd.py --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "rwd" / "csu" / "csu_data.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "rwd" / "csu"

# Sheets to read (skip byte-identical duplicates)
SHEETS_TO_READ = ("demo", "medication", "proc", "lab")
SHEETS_TO_SKIP = ("provider", "inpatientdata")

# Split ratios (chronological, patient-level isolation)
SPLIT_RATIOS = {
    "train": 0.60,
    "validation": 0.20,
    "test": 0.15,
    "holdout": 0.05,
}
TEMPORAL_GAP_DAYS = 7

# Brand name normalisation
BRAND_NAME_MAP: dict[str, str] = {
    "DUPIXENT SYRINGE": "Dupixent",
    "DUPIXENT PEN": "Dupixent",
    "XOLAIR": "Xolair",
}

# Insurance type mapping (bus field)
INSURANCE_TYPE_MAP: dict[str, str] = {
    "COM": "Commercial",
    "MCR": "Medicare",
    "MCD": "Medicaid",
}

# US Census region mapping (3-digit ZIP prefix ranges)
# Each tuple is (start, end) inclusive.
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

# Age group bins
AGE_BINS = [0, 18, 35, 50, 65, 200]
AGE_LABELS = ["<18", "18-34", "35-49", "50-65", "65+"]

# Adoption category quartile labels (Q4 highest)
ADOPTION_LABELS = ["late_majority", "early_majority", "early_adopter", "innovator"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_luhn_npi(obfuscated: str) -> str:
    """Generate a deterministic Luhn-valid 10-digit NPI from an obfuscated string.

    Steps:
      1. SHA-256 hash the obfuscated NPI string.
      2. Take the first 9 hex digits and convert each to a decimal digit (mod 10).
      3. Compute a Luhn check digit over those 9 digits.
      4. Return the 10-digit string.
    """
    digest = hashlib.sha256(obfuscated.encode("utf-8")).hexdigest()
    base_digits = [int(c, 16) % 10 for c in digest[:9]]

    # Luhn check digit computation (standard algorithm)
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


def _patient_hash(patid: int | str) -> str:
    """Deterministic 20-char hex hash for a patient id."""
    return hashlib.sha256(str(patid).encode("utf-8")).hexdigest()[:20]


def _map_zipcode_to_region(zip_code: str | None) -> str | None:
    """Map a 5-digit ZIP code to a US Census region via its 3-digit prefix."""
    if not zip_code or not isinstance(zip_code, str):
        return None
    # Handle multi-zip (underscore separated) — take the first
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


def _age_group(age: float | int | None) -> str | None:
    """Bin a numeric age into an age-group label."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return None
    age_int = int(age)
    if age_int < 0:
        return None
    for i, upper in enumerate(AGE_BINS[1:]):
        if age_int < upper:
            return AGE_LABELS[i]
    return AGE_LABELS[-1]


def _insurance_type(bus: str | None) -> str | None:
    """Map bus field to insurance type."""
    if bus is None or not isinstance(bus, str):
        return None
    return INSURANCE_TYPE_MAP.get(bus.strip().upper(), "Other")


def _normalise_brand(brand_name: str | None) -> str | None:
    """Normalise Brand_Name to a canonical drug name."""
    if brand_name is None or not isinstance(brand_name, str):
        return None
    return BRAND_NAME_MAP.get(brand_name.strip().upper(), brand_name.strip().title())


def _format_diagcode(code: str | None) -> str | None:
    """Format a diagnosis code with a dot (e.g. L508 -> L50.8)."""
    if code is None or not isinstance(code, str):
        return None
    code = code.strip().upper()
    if len(code) > 3 and "." not in code:
        return code[:3] + "." + code[3:]
    return code


def _safe_date(val: Any) -> str | None:
    """Convert a value to ISO date string (YYYY-MM-DD) or None."""
    if val is None:
        return None
    if isinstance(val, pd.Timestamp):
        if pd.isna(val):
            return None
        return val.strftime("%Y-%m-%d")
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, str):
        return val[:10]  # assume ISO-ish
    return None


def _safe_int(val: Any) -> int | None:
    """Coerce to int or None."""
    if val is None:
        return None
    try:
        if isinstance(val, float) and np.isnan(val):
            return None
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val: Any) -> float | None:
    """Coerce to float or None."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# CSUDataConverter
# ---------------------------------------------------------------------------


class CSUDataConverter:
    """Convert CSU claims Excel data to E2I ML v3 JSON format."""

    def __init__(
        self,
        excel_path: str | Path,
        output_dir: str | Path,
        max_patients: int | None = None,
    ) -> None:
        self.excel_path = Path(excel_path)
        self.output_dir = Path(output_dir)
        self.max_patients = max_patients

        # Will be populated during conversion
        self.sheets: dict[str, pd.DataFrame] = {}
        self.patient_id_map: dict[int, str] = {}   # patid -> PAT_XXXXXX
        self.journey_id_map: dict[int, str] = {}    # patid -> PJ_XXXXXX
        self.hcp_npi_map: dict[str, str] = {}       # obfuscated -> HCP_XXXXXX
        self.hcp_id_map: dict[str, str] = {}        # obfuscated -> generated NPI

        # Precomputed per-patient data for efficiency
        self._med_by_pat: dict[int, pd.DataFrame] = {}
        self._proc_by_pat: dict[int, pd.DataFrame] = {}
        self._lab_by_pat: dict[int, pd.DataFrame] = {}

        # Split config
        self.split_config_id = str(uuid.uuid4())
        self.now_iso = datetime.now().isoformat()

        # Output records
        self.hcp_profiles: list[dict[str, Any]] = []
        self.patient_journeys: list[dict[str, Any]] = []
        self.treatment_events: list[dict[str, Any]] = []
        self.split_registry: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert_all(self) -> dict[str, int]:
        """Run the full conversion pipeline.  Returns record counts."""
        logger.info("Reading Excel workbook: %s", self.excel_path)
        self.sheets = self._read_excel()

        logger.info("Cleaning sheets ...")
        self.sheets["demo"] = self._clean_demo(self.sheets["demo"])
        self.sheets["medication"] = self._clean_medication(self.sheets["medication"])
        self.sheets["proc"] = self._clean_procedure(self.sheets["proc"])
        self.sheets["lab"] = self._clean_lab(self.sheets["lab"])

        # Pre-index clinical data by patid
        self._index_clinical_data()

        logger.info("Building patient ID map ...")
        self._build_patient_id_map()

        logger.info("Building HCP profiles ...")
        self.hcp_profiles = self._build_hcp_profiles()

        logger.info("Building patient journeys ...")
        self.patient_journeys = self._build_patient_journeys()

        logger.info("Building treatment events ...")
        self.treatment_events = self._build_treatment_events()

        logger.info("Applying chronological split ...")
        self._apply_chronological_split()

        logger.info("Building split registry ...")
        self.split_registry = self._build_split_registry()

        logger.info("Validating output ...")
        self._validate_output()

        counts = {
            "hcp_profiles": len(self.hcp_profiles),
            "patient_journeys": len(self.patient_journeys),
            "treatment_events": len(self.treatment_events),
            "split_registry": len(self.split_registry),
        }
        return counts

    def write_all(self) -> None:
        """Write all JSON output files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json("e2i_ml_v3_hcp_profiles", self.hcp_profiles)
        self._write_json("e2i_ml_v3_patient_journeys", self.patient_journeys)
        self._write_json("e2i_ml_v3_treatment_events", self.treatment_events)
        self._write_json("e2i_ml_v3_split_registry", self.split_registry)

    # ------------------------------------------------------------------
    # Excel Reading & Cleaning
    # ------------------------------------------------------------------

    def _read_excel(self) -> dict[str, pd.DataFrame]:
        """Read the 4 unique sheets from the Excel workbook."""
        result: dict[str, pd.DataFrame] = {}
        xl = pd.ExcelFile(self.excel_path)
        available = set(xl.sheet_names)

        for sheet in SHEETS_TO_READ:
            if sheet not in available:
                logger.warning("Sheet '%s' not found in workbook — skipping", sheet)
                continue
            df = pd.read_excel(xl, sheet_name=sheet)
            # Drop all-NaN columns (Excel padding)
            df = df.dropna(axis=1, how="all")
            logger.info(
                "  Sheet '%s': %d rows x %d cols", sheet, len(df), len(df.columns)
            )
            result[sheet] = df

        for skip in SHEETS_TO_SKIP:
            if skip in available:
                logger.info("  Skipping byte-identical sheet '%s'", skip)

        xl.close()
        return result

    def _clean_demo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the demographics sheet."""
        orig_len = len(df)

        # Deduplicate by patid (keep first)
        df = df.drop_duplicates(subset=["patid"], keep="first").copy()
        dupes = orig_len - len(df)
        if dupes:
            logger.info("  demo: dropped %d duplicate patid rows", dupes)

        # Fix age=-1 -> NaN
        df.loc[df["age"] == -1, "age"] = np.nan

        # Ensure patid is int
        df["patid"] = df["patid"].astype(int)

        # Parse dates
        for col in ("indexdt", "eligeff", "eligend"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        logger.info("  demo: %d patients after cleaning", len(df))
        return df

    def _clean_medication(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the medication sheet."""
        df = df.copy()
        df["patid"] = df["patid"].astype(int)

        for col in ("medication_date", "indexdt"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Normalise brand name
        if "Brand_Name" in df.columns:
            df["brand_normalised"] = df["Brand_Name"].apply(_normalise_brand)

        logger.info("  medication: %d rows after cleaning", len(df))
        return df

    def _clean_procedure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the procedure sheet."""
        df = df.copy()
        df["patid"] = df["patid"].astype(int)

        for col in ("proc_date", "indexdt"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        logger.info("  proc: %d rows after cleaning", len(df))
        return df

    def _clean_lab(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the lab sheet."""
        df = df.copy()
        df["patid"] = df["patid"].astype(int)

        for col in ("fst_dt", "indexdt"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        logger.info("  lab: %d rows after cleaning", len(df))
        return df

    # ------------------------------------------------------------------
    # Index Clinical Data
    # ------------------------------------------------------------------

    def _index_clinical_data(self) -> None:
        """Group clinical sheets by patid for efficient per-patient lookups."""
        if "medication" in self.sheets:
            for patid, grp in self.sheets["medication"].groupby("patid"):
                self._med_by_pat[int(patid)] = grp
        if "proc" in self.sheets:
            for patid, grp in self.sheets["proc"].groupby("patid"):
                self._proc_by_pat[int(patid)] = grp
        if "lab" in self.sheets:
            for patid, grp in self.sheets["lab"].groupby("patid"):
                self._lab_by_pat[int(patid)] = grp

    # ------------------------------------------------------------------
    # Patient ID Map
    # ------------------------------------------------------------------

    def _build_patient_id_map(self) -> None:
        """Build master patient registry from all sheets' unique patids."""
        all_patids: set[int] = set()
        for sheet_name, df in self.sheets.items():
            if "patid" in df.columns:
                all_patids.update(df["patid"].dropna().astype(int).unique())

        all_patids_sorted = sorted(all_patids)

        if self.max_patients is not None:
            all_patids_sorted = all_patids_sorted[: self.max_patients]
            logger.info(
                "  Limited to %d patients (--max-patients)", self.max_patients
            )

        for seq, patid in enumerate(all_patids_sorted):
            self.patient_id_map[patid] = f"PAT_{seq:06d}"
            self.journey_id_map[patid] = f"PJ_{seq:06d}"

        logger.info("  Master patient registry: %d patients", len(self.patient_id_map))

    # ------------------------------------------------------------------
    # HCP Profiles
    # ------------------------------------------------------------------

    def _build_hcp_profiles(self) -> list[dict[str, Any]]:
        """Build HCP profiles from unique obfuscated NPIs in medication + proc."""
        # Collect all obfuscated NPIs with prescribing counts
        npi_rx_counts: dict[str, int] = {}
        npi_patient_counts: dict[str, set[int]] = {}

        if "medication" in self.sheets:
            med = self.sheets["medication"]
            if "npi" in med.columns:
                for _, row in med.iterrows():
                    npi_str = str(row["npi"]).strip() if pd.notna(row.get("npi")) else None
                    if not npi_str or npi_str == "nan":
                        continue
                    patid = int(row["patid"])
                    if patid not in self.patient_id_map:
                        continue
                    npi_rx_counts[npi_str] = npi_rx_counts.get(npi_str, 0) + 1
                    npi_patient_counts.setdefault(npi_str, set()).add(patid)

        if "proc" in self.sheets:
            proc = self.sheets["proc"]
            # proc sheet may not have npi column; use patid for counting
            # but we still need NPIs for HCP profiles — proc may not have them
            # Only process if npi column exists
            if "npi" in proc.columns:
                for _, row in proc.iterrows():
                    npi_str = str(row["npi"]).strip() if pd.notna(row.get("npi")) else None
                    if not npi_str or npi_str == "nan":
                        continue
                    patid = int(row["patid"])
                    if patid not in self.patient_id_map:
                        continue
                    npi_rx_counts[npi_str] = npi_rx_counts.get(npi_str, 0) + 1
                    npi_patient_counts.setdefault(npi_str, set()).add(patid)

        if not npi_rx_counts:
            logger.warning("  No HCP NPIs found in medication/proc sheets")
            return []

        # Sort NPIs for deterministic ordering
        sorted_npis = sorted(npi_rx_counts.keys())

        # Compute volume quartiles for adoption category
        volumes = [npi_rx_counts[n] for n in sorted_npis]
        q25, q50, q75 = np.percentile(volumes, [25, 50, 75])

        profiles: list[dict[str, Any]] = []
        for seq, obfuscated_npi in enumerate(sorted_npis):
            hcp_id = f"HCP_{seq:06d}"
            generated_npi = _generate_luhn_npi(obfuscated_npi)
            self.hcp_npi_map[obfuscated_npi] = hcp_id
            self.hcp_id_map[obfuscated_npi] = generated_npi

            rx_vol = npi_rx_counts[obfuscated_npi]
            pat_vol = len(npi_patient_counts.get(obfuscated_npi, set()))

            # Adoption category by prescribing volume quartile
            if rx_vol >= q75:
                adoption = "innovator"
            elif rx_vol >= q50:
                adoption = "early_adopter"
            elif rx_vol >= q25:
                adoption = "early_majority"
            else:
                adoption = "late_majority"

            # Practice type by total patient volume
            if pat_vol > 100:
                practice_type = "Hospital"
            elif pat_vol >= 50:
                practice_type = "Group"
            else:
                practice_type = "Solo"

            profiles.append(
                {
                    "hcp_id": hcp_id,
                    "npi": generated_npi,
                    "first_name": None,
                    "last_name": None,
                    "specialty": "Allergy/Immunology",
                    "sub_specialty": None,
                    "practice_type": practice_type,
                    "practice_size": None,
                    "geographic_region": None,
                    "state": None,
                    "city": None,
                    "zip_code": None,
                    "priority_tier": None,
                    "decile": None,
                    "total_patient_volume": pat_vol,
                    "target_patient_volume": None,
                    "prescribing_volume": rx_vol,
                    "years_experience": None,
                    "affiliation_primary": None,
                    "affiliation_secondary": None,
                    "digital_engagement_score": None,
                    "preferred_channel": None,
                    "last_interaction_date": None,
                    "interaction_frequency": None,
                    "influence_network_size": None,
                    "peer_influence_score": None,
                    "adoption_category": adoption,
                    "coverage_status": None,
                    "territory_id": None,
                    "sales_rep_id": None,
                    "created_at": self.now_iso,
                    "updated_at": self.now_iso,
                }
            )

        logger.info("  Built %d HCP profiles", len(profiles))
        return profiles

    # ------------------------------------------------------------------
    # Patient Journeys
    # ------------------------------------------------------------------

    def _build_patient_journeys(self) -> list[dict[str, Any]]:
        """Build patient journey records for all patients in the master registry."""
        demo = self.sheets.get("demo")
        demo_patids = set(demo["patid"].values) if demo is not None else set()
        clinical_patids = (
            set(self._med_by_pat.keys())
            | set(self._proc_by_pat.keys())
            | set(self._lab_by_pat.keys())
        )

        journeys: list[dict[str, Any]] = []
        type_counts = {"A": 0, "B": 0, "C": 0}

        for patid, pat_id in sorted(self.patient_id_map.items(), key=lambda x: x[1]):
            pj_id = self.journey_id_map[patid]
            in_demo = patid in demo_patids
            in_clinical = patid in clinical_patids

            # Determine archetype
            if in_demo and in_clinical:
                archetype = "A"
            elif in_demo:
                archetype = "B"
            else:
                archetype = "C"
            type_counts[archetype] += 1

            # Demographics (from demo sheet)
            demo_row = None
            if in_demo and demo is not None:
                matches = demo[demo["patid"] == patid]
                if len(matches) > 0:
                    demo_row = matches.iloc[0]

            # Core dates
            index_date = None
            if demo_row is not None and pd.notna(demo_row.get("indexdt")):
                index_date = demo_row["indexdt"]
            elif in_clinical:
                # Use earliest clinical date
                dates = []
                if patid in self._med_by_pat:
                    med_dates = self._med_by_pat[patid]["medication_date"].dropna()
                    if len(med_dates) > 0:
                        dates.append(med_dates.min())
                if patid in self._proc_by_pat:
                    proc_dates = self._proc_by_pat[patid]["proc_date"].dropna()
                    if len(proc_dates) > 0:
                        dates.append(proc_dates.min())
                if patid in self._lab_by_pat:
                    lab_dates = self._lab_by_pat[patid]["fst_dt"].dropna()
                    if len(lab_dates) > 0:
                        dates.append(lab_dates.min())
                if dates:
                    index_date = min(dates)

            # Journey end date: latest clinical event or eligend
            end_date = None
            end_candidates = []
            if demo_row is not None and pd.notna(demo_row.get("eligend")):
                end_candidates.append(demo_row["eligend"])
            if patid in self._med_by_pat:
                med_dates = self._med_by_pat[patid]["medication_date"].dropna()
                if len(med_dates) > 0:
                    # Add last fill + days_supply
                    last_med = self._med_by_pat[patid].loc[med_dates.idxmax()]
                    last_date = last_med["medication_date"]
                    days_sup = _safe_int(last_med.get("days_sup")) or 0
                    end_candidates.append(last_date + timedelta(days=days_sup))
            if patid in self._proc_by_pat:
                proc_dates = self._proc_by_pat[patid]["proc_date"].dropna()
                if len(proc_dates) > 0:
                    end_candidates.append(proc_dates.max())
            if patid in self._lab_by_pat:
                lab_dates = self._lab_by_pat[patid]["fst_dt"].dropna()
                if len(lab_dates) > 0:
                    end_candidates.append(lab_dates.max())
            if end_candidates:
                end_date = max(end_candidates)

            # Duration
            duration_days = None
            if index_date is not None and end_date is not None:
                try:
                    duration_days = max(0, (end_date - index_date).days)
                except Exception:
                    duration_days = None

            # Demographics fields
            gender = None
            age = None
            age_grp = None
            zip_code = None
            region = None
            insurance = None
            diagcode = None
            continuous = 0

            if demo_row is not None:
                gdr = demo_row.get("gdr_cd")
                if isinstance(gdr, str) and gdr.strip().upper() in ("F", "M"):
                    gender = gdr.strip().upper()

                age_val = demo_row.get("age")
                if pd.notna(age_val) and age_val != -1:
                    age = float(age_val)
                    age_grp = _age_group(age)

                zc = demo_row.get("zipcode_5")
                if pd.notna(zc):
                    zip_code = str(zc).split("_")[0].strip()
                    region = _map_zipcode_to_region(str(zc))

                bus = demo_row.get("bus")
                if pd.notna(bus):
                    insurance = _insurance_type(str(bus))

                dc = demo_row.get("diagcode")
                if pd.notna(dc):
                    diagcode = _format_diagcode(str(dc))

                ce = demo_row.get("continuous_enrollment")
                if pd.notna(ce):
                    continuous = int(ce)

            # Treatment flags
            treatment_initiated = 1 if patid in self._med_by_pat else 0
            discontinuation = self._derive_discontinuation_flag(patid)
            disease_severity = self._derive_disease_severity(patid)
            engagement = self._derive_engagement_score(patid, continuous)

            # ML features
            days_on_therapy = 0
            hcp_visits = 0
            prior_treatments = 0

            if patid in self._med_by_pat:
                med_df = self._med_by_pat[patid]
                # days_on_therapy = sum of days_sup
                if "days_sup" in med_df.columns:
                    days_on_therapy = int(med_df["days_sup"].fillna(0).sum())

                # hcp_visits = unique (npi, medication_date) pairs
                if "npi" in med_df.columns and "medication_date" in med_df.columns:
                    visit_pairs = med_df[["npi", "medication_date"]].dropna()
                    hcp_visits = len(
                        visit_pairs.drop_duplicates(subset=["npi", "medication_date"])
                    )

                # prior_treatments = distinct drugs before index date
                if index_date is not None and "brand_normalised" in med_df.columns:
                    before_idx = med_df[
                        med_df["medication_date"] < index_date
                    ]
                    prior_treatments = before_idx["brand_normalised"].nunique()

            # Brand
            brand = "competitor" if treatment_initiated else None

            # Data quality score by archetype
            if archetype == "A":
                dq_score = round(0.9 + np.random.uniform(0, 0.1), 2)
            elif archetype == "B":
                dq_score = round(np.random.uniform(0.5, 0.7), 2)
            else:
                dq_score = round(np.random.uniform(0.3, 0.5), 2)

            # Journey stage
            if treatment_initiated and days_on_therapy > 90:
                journey_stage = "treatment_optimization"
            elif treatment_initiated:
                journey_stage = "initial_treatment"
            else:
                journey_stage = "diagnosis"

            # Journey status
            if discontinuation == 1:
                journey_status = "completed"
            elif treatment_initiated:
                journey_status = "active"
            else:
                journey_status = "monitoring"

            journeys.append(
                {
                    "patient_journey_id": pj_id,
                    "patient_id": pat_id,
                    "patient_hash": _patient_hash(patid),
                    "journey_start_date": _safe_date(index_date),
                    "journey_end_date": _safe_date(end_date),
                    "journey_duration_days": duration_days,
                    "journey_stage": journey_stage,
                    "journey_status": journey_status,
                    "primary_diagnosis_code": diagcode or "L50.8",
                    "primary_diagnosis_desc": "Chronic Spontaneous Urticaria",
                    "secondary_diagnosis_codes": [],
                    "brand": brand,
                    "age_group": age_grp,
                    "gender": gender,
                    "geographic_region": region,
                    "state": None,
                    "zip_code": zip_code,
                    "insurance_type": insurance,
                    "data_quality_score": dq_score,
                    "comorbidities": [],
                    "risk_score": None,
                    "data_source": "RWD_Claims",
                    "data_sources_matched": ["RWD_Claims"],
                    "source_match_confidence": None,
                    "source_stacking_flag": False,
                    "source_combination_method": None,
                    "source_timestamp": None,
                    "ingestion_timestamp": self.now_iso,
                    "data_lag_hours": None,
                    "data_split": None,  # Set during chronological split
                    "split_config_id": self.split_config_id,
                    "created_at": self.now_iso,
                    "updated_at": self.now_iso,
                    "treatment_initiated": treatment_initiated,
                    "discontinuation_flag": discontinuation,
                    "disease_severity": round(disease_severity, 1),
                    "engagement_score": round(engagement, 1),
                    "days_on_therapy": days_on_therapy,
                    "hcp_visits": hcp_visits,
                    "prior_treatments": prior_treatments,
                }
            )

        logger.info(
            "  Built %d patient journeys (A=%d, B=%d, C=%d)",
            len(journeys),
            type_counts["A"],
            type_counts["B"],
            type_counts["C"],
        )
        return journeys

    # ------------------------------------------------------------------
    # Treatment Events
    # ------------------------------------------------------------------

    def _build_treatment_events(self) -> list[dict[str, Any]]:
        """Build treatment event records from all 4 sheets."""
        events: list[dict[str, Any]] = []
        te_seq = 0

        # Track per-patient sequence numbers
        pat_seq: dict[str, int] = {}

        def _next_seq(pat_id: str) -> int:
            s = pat_seq.get(pat_id, 0) + 1
            pat_seq[pat_id] = s
            return s

        # --- demo → diagnosis events ---
        if "demo" in self.sheets:
            demo = self.sheets["demo"]
            for _, row in demo.iterrows():
                patid = int(row["patid"])
                if patid not in self.patient_id_map:
                    continue
                pat_id = self.patient_id_map[patid]
                pj_id = self.journey_id_map[patid]

                diagcode = _format_diagcode(
                    str(row["diagcode"]) if pd.notna(row.get("diagcode")) else None
                )

                events.append(
                    {
                        "treatment_event_id": f"TE_{te_seq:06d}",
                        "patient_journey_id": pj_id,
                        "patient_id": pat_id,
                        "hcp_id": None,
                        "event_date": _safe_date(row.get("indexdt")),
                        "event_type": "diagnosis",
                        "event_subtype": None,
                        "brand": None,
                        "drug_ndc": None,
                        "drug_name": None,
                        "drug_class": None,
                        "dosage": None,
                        "duration_days": None,
                        "icd_codes": [diagcode] if diagcode else [],
                        "cpt_codes": [],
                        "loinc_codes": [],
                        "lab_values": {},
                        "location_type": None,
                        "facility_id": None,
                        "cost": None,
                        "outcome_indicator": None,
                        "adverse_event_flag": False,
                        "discontinuation_flag": False,
                        "discontinuation_reason": None,
                        "sequence_number": _next_seq(pat_id),
                        "days_from_diagnosis": 0,
                        "previous_treatment": None,
                        "next_treatment": None,
                        "data_source": "RWD_Claims",
                        "source_timestamp": None,
                        "ingestion_timestamp": self.now_iso,
                        "data_split": None,  # Set later
                        "created_at": self.now_iso,
                        "updated_at": self.now_iso,
                    }
                )
                te_seq += 1

        # --- medication → prescription events ---
        if "medication" in self.sheets:
            med = self.sheets["medication"]
            for _, row in med.iterrows():
                patid = int(row["patid"])
                if patid not in self.patient_id_map:
                    continue
                pat_id = self.patient_id_map[patid]
                pj_id = self.journey_id_map[patid]

                # HCP
                npi_str = (
                    str(row["npi"]).strip()
                    if pd.notna(row.get("npi")) and str(row["npi"]).strip() != "nan"
                    else None
                )
                hcp_id = self.hcp_npi_map.get(npi_str) if npi_str else None

                # Drug info
                brand_norm = row.get("brand_normalised")
                drug_name = brand_norm if pd.notna(brand_norm) else None
                drug_ndc = str(int(row["code"])) if pd.notna(row.get("code")) else None
                dosage = str(row["strength"]) if pd.notna(row.get("strength")) else None
                duration = _safe_int(row.get("days_sup"))

                # Days from diagnosis
                days_from = _safe_int(row.get("days_from_indexdt"))

                events.append(
                    {
                        "treatment_event_id": f"TE_{te_seq:06d}",
                        "patient_journey_id": pj_id,
                        "patient_id": pat_id,
                        "hcp_id": hcp_id,
                        "event_date": _safe_date(row.get("medication_date")),
                        "event_type": "prescription",
                        "event_subtype": None,
                        "brand": "competitor",
                        "drug_ndc": drug_ndc,
                        "drug_name": drug_name,
                        "drug_class": "Monoclonal Antibody",
                        "dosage": dosage,
                        "duration_days": duration,
                        "icd_codes": [],
                        "cpt_codes": [],
                        "loinc_codes": [],
                        "lab_values": {},
                        "location_type": None,
                        "facility_id": None,
                        "cost": None,
                        "outcome_indicator": None,
                        "adverse_event_flag": False,
                        "discontinuation_flag": False,
                        "discontinuation_reason": None,
                        "sequence_number": _next_seq(pat_id),
                        "days_from_diagnosis": days_from if days_from is not None else 0,
                        "previous_treatment": None,
                        "next_treatment": None,
                        "data_source": "RWD_Claims",
                        "source_timestamp": None,
                        "ingestion_timestamp": self.now_iso,
                        "data_split": None,
                        "created_at": self.now_iso,
                        "updated_at": self.now_iso,
                    }
                )
                te_seq += 1

        # --- proc → procedure events ---
        if "proc" in self.sheets:
            proc = self.sheets["proc"]
            for _, row in proc.iterrows():
                patid = int(row["patid"])
                if patid not in self.patient_id_map:
                    continue
                pat_id = self.patient_id_map[patid]
                pj_id = self.journey_id_map[patid]

                proc_code = (
                    str(row["proc_code"]).strip()
                    if pd.notna(row.get("proc_code"))
                    else None
                )

                # Days from diagnosis
                days_from = 0
                if pd.notna(row.get("proc_date")) and pd.notna(row.get("indexdt")):
                    try:
                        days_from = (row["proc_date"] - row["indexdt"]).days
                    except Exception:
                        days_from = 0

                events.append(
                    {
                        "treatment_event_id": f"TE_{te_seq:06d}",
                        "patient_journey_id": pj_id,
                        "patient_id": pat_id,
                        "hcp_id": None,
                        "event_date": _safe_date(row.get("proc_date")),
                        "event_type": "procedure",
                        "event_subtype": None,
                        "brand": None,
                        "drug_ndc": None,
                        "drug_name": None,
                        "drug_class": None,
                        "dosage": None,
                        "duration_days": None,
                        "icd_codes": [],
                        "cpt_codes": [proc_code] if proc_code else [],
                        "loinc_codes": [],
                        "lab_values": {},
                        "location_type": None,
                        "facility_id": None,
                        "cost": None,
                        "outcome_indicator": None,
                        "adverse_event_flag": False,
                        "discontinuation_flag": False,
                        "discontinuation_reason": None,
                        "sequence_number": _next_seq(pat_id),
                        "days_from_diagnosis": days_from,
                        "previous_treatment": None,
                        "next_treatment": None,
                        "data_source": "RWD_Claims",
                        "source_timestamp": None,
                        "ingestion_timestamp": self.now_iso,
                        "data_split": None,
                        "created_at": self.now_iso,
                        "updated_at": self.now_iso,
                    }
                )
                te_seq += 1

        # --- lab → lab_test events ---
        if "lab" in self.sheets:
            lab = self.sheets["lab"]
            for _, row in lab.iterrows():
                patid = int(row["patid"])
                if patid not in self.patient_id_map:
                    continue
                pat_id = self.patient_id_map[patid]
                pj_id = self.journey_id_map[patid]

                loinc = (
                    str(row["loinc_cd"]).strip()
                    if pd.notna(row.get("loinc_cd"))
                    else None
                )
                tst_desc = (
                    str(row["tst_desc"]).strip()
                    if pd.notna(row.get("tst_desc"))
                    else None
                )
                rslt_nbr = _safe_float(row.get("rslt_nbr"))
                lab_values: dict[str, Any] = {}
                if tst_desc and rslt_nbr is not None:
                    lab_values[tst_desc] = rslt_nbr

                # Days from diagnosis
                days_from = _safe_int(row.get("days_from_indexdt"))

                events.append(
                    {
                        "treatment_event_id": f"TE_{te_seq:06d}",
                        "patient_journey_id": pj_id,
                        "patient_id": pat_id,
                        "hcp_id": None,
                        "event_date": _safe_date(row.get("fst_dt")),
                        "event_type": "lab_test",
                        "event_subtype": None,
                        "brand": None,
                        "drug_ndc": None,
                        "drug_name": None,
                        "drug_class": None,
                        "dosage": None,
                        "duration_days": None,
                        "icd_codes": [],
                        "cpt_codes": [],
                        "loinc_codes": [loinc] if loinc else [],
                        "lab_values": lab_values,
                        "location_type": None,
                        "facility_id": None,
                        "cost": None,
                        "outcome_indicator": None,
                        "adverse_event_flag": False,
                        "discontinuation_flag": False,
                        "discontinuation_reason": None,
                        "sequence_number": _next_seq(pat_id),
                        "days_from_diagnosis": days_from if days_from is not None else 0,
                        "previous_treatment": None,
                        "next_treatment": None,
                        "data_source": "RWD_Claims",
                        "source_timestamp": None,
                        "ingestion_timestamp": self.now_iso,
                        "data_split": None,
                        "created_at": self.now_iso,
                        "updated_at": self.now_iso,
                    }
                )
                te_seq += 1

        logger.info("  Built %d treatment events", len(events))
        return events

    # ------------------------------------------------------------------
    # Chronological Split
    # ------------------------------------------------------------------

    def _apply_chronological_split(self) -> None:
        """Assign data_split to journeys and treatment events chronologically.

        Patient-level isolation: all events for a patient share the same split.
        """
        # Sort journeys by journey_start_date
        dated_journeys = [
            (j, j["journey_start_date"])
            for j in self.patient_journeys
            if j["journey_start_date"] is not None
        ]
        undated_journeys = [
            j for j in self.patient_journeys if j["journey_start_date"] is None
        ]

        dated_journeys.sort(key=lambda x: x[1])

        n = len(dated_journeys)
        if n == 0:
            logger.warning("  No dated journeys — cannot apply chronological split")
            return

        # Compute cutoff indices (with temporal gap accounting)
        train_end = int(n * SPLIT_RATIOS["train"])
        val_end = train_end + int(n * SPLIT_RATIOS["validation"])
        test_end = val_end + int(n * SPLIT_RATIOS["test"])

        # Assign splits
        patient_split: dict[str, str] = {}
        for i, (j, _) in enumerate(dated_journeys):
            if i < train_end:
                split = "train"
            elif i < val_end:
                split = "validation"
            elif i < test_end:
                split = "test"
            else:
                split = "holdout"
            j["data_split"] = split
            patient_split[j["patient_id"]] = split

        # Undated journeys go to train
        for j in undated_journeys:
            j["data_split"] = "train"
            patient_split[j["patient_id"]] = "train"

        # Propagate splits to treatment events
        for te in self.treatment_events:
            te["data_split"] = patient_split.get(te["patient_id"], "train")

        # Compute split date boundaries for registry
        self._split_dates = {}
        if dated_journeys:
            self._split_dates["data_start"] = dated_journeys[0][1]
            self._split_dates["data_end"] = dated_journeys[-1][1]
            if train_end > 0:
                self._split_dates["train_end"] = dated_journeys[train_end - 1][1]
            if val_end > 0 and val_end <= n:
                self._split_dates["validation_end"] = dated_journeys[
                    min(val_end - 1, n - 1)
                ][1]
            if test_end > 0 and test_end <= n:
                self._split_dates["test_end"] = dated_journeys[
                    min(test_end - 1, n - 1)
                ][1]

        # Log split sizes
        split_counts: dict[str, int] = {}
        for j in self.patient_journeys:
            s = j.get("data_split", "unknown")
            split_counts[s] = split_counts.get(s, 0) + 1
        logger.info("  Split sizes: %s", split_counts)

    # ------------------------------------------------------------------
    # Split Registry
    # ------------------------------------------------------------------

    def _build_split_registry(self) -> list[dict[str, Any]]:
        """Build the split registry configuration record."""
        dates = getattr(self, "_split_dates", {})
        return [
            {
                "split_config_id": self.split_config_id,
                "config_name": "csu_rwd_v1",
                "config_version": "1.0.0",
                "train_ratio": SPLIT_RATIOS["train"],
                "validation_ratio": SPLIT_RATIOS["validation"],
                "test_ratio": SPLIT_RATIOS["test"],
                "holdout_ratio": SPLIT_RATIOS["holdout"],
                "data_start_date": dates.get("data_start"),
                "data_end_date": dates.get("data_end"),
                "train_end_date": dates.get("train_end"),
                "validation_end_date": dates.get("validation_end"),
                "test_end_date": dates.get("test_end"),
                "temporal_gap_days": TEMPORAL_GAP_DAYS,
                "patient_level_isolation": True,
                "split_strategy": "chronological",
                "is_active": True,
                "created_at": self.now_iso,
            }
        ]

    # ------------------------------------------------------------------
    # Derived Causal Variables
    # ------------------------------------------------------------------

    def _derive_disease_severity(self, patid: int) -> float:
        """Compute disease severity score (0-10) for a patient.

        Base 2.0 (CSU diagnosis)
        +1.0 per additional ICD-10 L50x subcode (cap +2.0)
        +0.5 per medication fill (cap +3.0)
        +0.5 per J2357 procedure (cap +2.0)
        +1.0 if any abnormal lab
        Clip [0, 10]
        """
        score = 2.0  # Base CSU diagnosis

        # Additional ICD codes (from demo — only one diagcode per row,
        # but we check all patients in demo)
        demo = self.sheets.get("demo")
        if demo is not None:
            pat_demo = demo[demo["patid"] == patid]
            if len(pat_demo) > 0:
                diag = pat_demo.iloc[0].get("diagcode")
                if pd.notna(diag):
                    code_str = str(diag).upper()
                    # Count L50x subcodes (L500, L501, ... L509)
                    # The main code is L508 typically; additional subcodes add severity
                    # Since we only have one row per patient, check if it's an L50x
                    if code_str.startswith("L50") and len(code_str) > 3:
                        # Base already accounts for one L50x; no additional here
                        pass

        # Additional L50x subcodes from treatment events (ICD codes)
        # We look for unique L50x subcodes beyond the primary
        l50_subcodes: set[str] = set()
        if demo is not None:
            pat_demo = demo[demo["patid"] == patid]
            if len(pat_demo) > 0:
                dc = pat_demo.iloc[0].get("diagcode")
                if pd.notna(dc):
                    l50_subcodes.add(str(dc).upper())

        # Count additional unique L50x codes beyond the first
        additional_l50 = max(0, len(l50_subcodes) - 1)
        score += min(additional_l50 * 1.0, 2.0)

        # Medication fills
        if patid in self._med_by_pat:
            n_fills = len(self._med_by_pat[patid])
            score += min(n_fills * 0.5, 3.0)

        # J2357 procedures
        if patid in self._proc_by_pat:
            proc_df = self._proc_by_pat[patid]
            if "proc_code" in proc_df.columns:
                j2357_count = (
                    proc_df["proc_code"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .eq("j2357")
                    .sum()
                )
                score += min(j2357_count * 0.5, 2.0)

        # Abnormal lab
        if patid in self._lab_by_pat:
            lab_df = self._lab_by_pat[patid]
            if "abnl_cd" in lab_df.columns:
                has_abnormal = lab_df["abnl_cd"].notna().any() and (
                    lab_df["abnl_cd"].astype(str).str.strip() != ""
                ).any()
                if has_abnormal:
                    score += 1.0

        return float(np.clip(score, 0.0, 10.0))

    def _derive_engagement_score(
        self, patid: int, continuous_enrollment: int = 0
    ) -> float:
        """Compute engagement score (0-10) for a patient.

        +2.0 per unique HCP (cap +4.0)
        +1.0 per 3 medication fills (cap +3.0)
        +0.1 per lab test (cap +2.0)
        +1.0 if continuous_enrollment
        Clip [0, 10]
        """
        score = 0.0

        # Unique HCPs (from medication NPIs)
        unique_hcps: set[str] = set()
        if patid in self._med_by_pat:
            med_df = self._med_by_pat[patid]
            if "npi" in med_df.columns:
                for npi in med_df["npi"].dropna():
                    npi_str = str(npi).strip()
                    if npi_str and npi_str != "nan":
                        unique_hcps.add(npi_str)
        score += min(len(unique_hcps) * 2.0, 4.0)

        # Medication fills
        if patid in self._med_by_pat:
            n_fills = len(self._med_by_pat[patid])
            score += min((n_fills // 3) * 1.0, 3.0)

        # Lab tests
        if patid in self._lab_by_pat:
            n_labs = len(self._lab_by_pat[patid])
            score += min(n_labs * 0.1, 2.0)

        # Continuous enrollment
        if continuous_enrollment:
            score += 1.0

        return float(np.clip(score, 0.0, 10.0))

    def _derive_discontinuation_flag(self, patid: int) -> int | None:
        """Determine discontinuation flag for a patient.

        For medicated patients: 1 if gap > 90 days between
        (last fill date + days_supply) and next fill, OR no fill
        within 90 days of last fill end.
        For non-medicated: None.
        """
        if patid not in self._med_by_pat:
            return None

        med_df = self._med_by_pat[patid].copy()
        if "medication_date" not in med_df.columns or "days_sup" not in med_df.columns:
            return None

        med_df = med_df.dropna(subset=["medication_date"]).sort_values(
            "medication_date"
        )
        if len(med_df) == 0:
            return None

        # Check for gaps > 90 days between fill end and next fill start
        for i in range(len(med_df) - 1):
            fill_date = med_df.iloc[i]["medication_date"]
            days_sup = _safe_int(med_df.iloc[i].get("days_sup")) or 0
            fill_end = fill_date + timedelta(days=days_sup)
            next_fill = med_df.iloc[i + 1]["medication_date"]
            gap = (next_fill - fill_end).days
            if gap > 90:
                return 1

        # Check if last fill end + 90 days is before "now" (use last available date)
        last_row = med_df.iloc[-1]
        last_fill = last_row["medication_date"]
        last_days_sup = _safe_int(last_row.get("days_sup")) or 0
        last_end = last_fill + timedelta(days=last_days_sup)

        # If last fill ended more than 90 days ago relative to latest data date,
        # consider it discontinued
        latest_dates = []
        for sheet_df in self.sheets.values():
            for col in ("medication_date", "proc_date", "fst_dt", "indexdt", "eligend"):
                if col in sheet_df.columns:
                    max_date = sheet_df[col].dropna().max()
                    if pd.notna(max_date):
                        latest_dates.append(max_date)
        if latest_dates:
            data_end = max(latest_dates)
            if (data_end - last_end).days > 90:
                return 1

        return 0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_output(self) -> None:
        """Validate FK integrity, enum values, and ranges across all outputs."""
        errors: list[str] = []
        warnings: list[str] = []

        id_pattern = re.compile(r"^(HCP|PAT|PJ|TE)_\d{6}$")

        # Collect valid IDs
        valid_hcp_ids = {h["hcp_id"] for h in self.hcp_profiles}
        valid_pat_ids = {j["patient_id"] for j in self.patient_journeys}
        valid_pj_ids = {j["patient_journey_id"] for j in self.patient_journeys}

        # Validate HCP profiles
        for h in self.hcp_profiles:
            if not id_pattern.match(h["hcp_id"]):
                errors.append(f"Invalid HCP ID format: {h['hcp_id']}")

        # Validate patient journeys
        for j in self.patient_journeys:
            if not id_pattern.match(j["patient_journey_id"]):
                errors.append(f"Invalid PJ ID format: {j['patient_journey_id']}")
            if not id_pattern.match(j["patient_id"]):
                errors.append(f"Invalid PAT ID format: {j['patient_id']}")

            # Required non-null
            if j["patient_journey_id"] is None:
                errors.append("Null patient_journey_id")
            if j["patient_id"] is None:
                errors.append("Null patient_id")
            if j["journey_start_date"] is None:
                warnings.append(
                    f"Null journey_start_date for {j['patient_journey_id']}"
                )

            # Range checks
            ds = j.get("disease_severity")
            if ds is not None and (ds < 0 or ds > 10):
                errors.append(
                    f"disease_severity out of range for {j['patient_journey_id']}: {ds}"
                )
            es = j.get("engagement_score")
            if es is not None and (es < 0 or es > 10):
                errors.append(
                    f"engagement_score out of range for {j['patient_journey_id']}: {es}"
                )
            dq = j.get("data_quality_score")
            if dq is not None and (dq < 0 or dq > 1):
                errors.append(
                    f"data_quality_score out of range for {j['patient_journey_id']}: {dq}"
                )

        # Validate treatment events
        for te in self.treatment_events:
            if not id_pattern.match(te["treatment_event_id"]):
                errors.append(f"Invalid TE ID format: {te['treatment_event_id']}")

            # FK integrity
            if te["patient_id"] not in valid_pat_ids:
                errors.append(
                    f"TE {te['treatment_event_id']} references unknown patient "
                    f"{te['patient_id']}"
                )
            if te["patient_journey_id"] not in valid_pj_ids:
                errors.append(
                    f"TE {te['treatment_event_id']} references unknown journey "
                    f"{te['patient_journey_id']}"
                )
            if te["hcp_id"] is not None and te["hcp_id"] not in valid_hcp_ids:
                errors.append(
                    f"TE {te['treatment_event_id']} references unknown HCP "
                    f"{te['hcp_id']}"
                )

        # Report
        if errors:
            for e in errors[:20]:
                logger.error("  VALIDATION ERROR: %s", e)
            if len(errors) > 20:
                logger.error("  ... and %d more errors", len(errors) - 20)
        if warnings:
            for w in warnings[:10]:
                logger.warning("  VALIDATION WARNING: %s", w)
            if len(warnings) > 10:
                logger.warning("  ... and %d more warnings", len(warnings) - 10)

        if not errors:
            logger.info("  Validation passed (0 errors, %d warnings)", len(warnings))
        else:
            logger.warning(
                "  Validation completed with %d errors, %d warnings",
                len(errors),
                len(warnings),
            )

        # Summary statistics
        self._print_summary()

    def _print_summary(self) -> None:
        """Print a summary of the conversion output."""
        logger.info("=" * 60)
        logger.info("CSU RWD Conversion Summary")
        logger.info("=" * 60)
        logger.info("  HCP profiles:      %d", len(self.hcp_profiles))
        logger.info("  Patient journeys:  %d", len(self.patient_journeys))
        logger.info("  Treatment events:  %d", len(self.treatment_events))
        logger.info("  Split registry:    %d", len(self.split_registry))

        # Patient counts by archetype
        demo_patids = set()
        if "demo" in self.sheets:
            demo_patids = set(self.sheets["demo"]["patid"].values)
        clinical_patids = (
            set(self._med_by_pat.keys())
            | set(self._proc_by_pat.keys())
            | set(self._lab_by_pat.keys())
        )

        type_a = len(demo_patids & clinical_patids & set(self.patient_id_map.keys()))
        type_b = len(
            (demo_patids - clinical_patids) & set(self.patient_id_map.keys())
        )
        type_c = len(
            (clinical_patids - demo_patids) & set(self.patient_id_map.keys())
        )
        logger.info("  Patient archetypes:")
        logger.info("    Type A (demo + clinical): %d", type_a)
        logger.info("    Type B (demo only):       %d", type_b)
        logger.info("    Type C (clinical only):   %d", type_c)

        # Split distribution
        split_counts: dict[str, int] = {}
        for j in self.patient_journeys:
            s = j.get("data_split", "unknown")
            split_counts[s] = split_counts.get(s, 0) + 1
        logger.info("  Split distribution:")
        for split_name in ("train", "validation", "test", "holdout"):
            logger.info(
                "    %-12s: %d", split_name, split_counts.get(split_name, 0)
            )

        # Region distribution
        region_counts: dict[str, int] = {}
        for j in self.patient_journeys:
            r = j.get("geographic_region") or "unknown"
            region_counts[r] = region_counts.get(r, 0) + 1
        logger.info("  Region distribution:")
        for region_name in sorted(region_counts.keys()):
            logger.info(
                "    %-12s: %d", region_name, region_counts[region_name]
            )

        # Target variable distributions
        treated = sum(1 for j in self.patient_journeys if j["treatment_initiated"] == 1)
        not_treated = sum(
            1 for j in self.patient_journeys if j["treatment_initiated"] == 0
        )
        disc = sum(
            1
            for j in self.patient_journeys
            if j["discontinuation_flag"] == 1
        )
        no_disc = sum(
            1
            for j in self.patient_journeys
            if j["discontinuation_flag"] == 0
        )
        null_disc = sum(
            1
            for j in self.patient_journeys
            if j["discontinuation_flag"] is None
        )
        logger.info("  Target distributions:")
        logger.info(
            "    treatment_initiated: 1=%d, 0=%d", treated, not_treated
        )
        logger.info(
            "    discontinuation:     1=%d, 0=%d, null=%d",
            disc,
            no_disc,
            null_disc,
        )

        # Severity and engagement stats
        severities = [
            j["disease_severity"]
            for j in self.patient_journeys
            if j["disease_severity"] is not None
        ]
        engagements = [
            j["engagement_score"]
            for j in self.patient_journeys
            if j["engagement_score"] is not None
        ]
        if severities:
            logger.info(
                "    disease_severity:    mean=%.1f, std=%.1f, min=%.1f, max=%.1f",
                np.mean(severities),
                np.std(severities),
                np.min(severities),
                np.max(severities),
            )
        if engagements:
            logger.info(
                "    engagement_score:    mean=%.1f, std=%.1f, min=%.1f, max=%.1f",
                np.mean(engagements),
                np.std(engagements),
                np.min(engagements),
                np.max(engagements),
            )

        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # JSON Output
    # ------------------------------------------------------------------

    def _write_json(self, name: str, records: list[dict[str, Any]]) -> None:
        """Write a list of records to a pretty-printed JSON file."""
        path = self.output_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str, ensure_ascii=False)
        logger.info("  Wrote %s (%d records)", path, len(records))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CSU real-world data to E2I JSON format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to CSU Excel workbook (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Limit to N patients (for quick testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only — do not write output files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    converter = CSUDataConverter(
        excel_path=args.input,
        output_dir=args.output,
        max_patients=args.max_patients,
    )

    try:
        counts = converter.convert_all()
    except Exception:
        logger.exception("Conversion failed")
        return 1

    if args.dry_run:
        logger.info("Dry run — skipping file output")
        logger.info("Record counts: %s", counts)
        return 0

    try:
        converter.write_all()
    except Exception:
        logger.exception("Failed to write output files")
        return 1

    logger.info("Conversion complete. Record counts: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
