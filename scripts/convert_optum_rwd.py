#!/usr/bin/env python3
"""Convert Optum Real-World Data (parquet) to E2I canonical format per cohort.

Implements the leakage-safe shaping from ``.claude/plans/csu-rwd-analyst-spec.md``:
  - §3 qualifying-dx rule for a claim-anchored index date (not vendor ``indexdt``)
  - §4 temporal architecture (180d lookback, 180d prediction window)
  - §5 inclusion/exclusion criteria
  - §6 code lists (CSU dx, biologics, exclusions, comorbidities, labs, drugs)
  - §7 lookback-only feature catalogue
  - §8 target derivations

Produces three separable cohorts with their own disjoint patient populations:
  A (initiation): ``initiated_biologic_180d`` — treatment-naive, dx-anchored
  B (discontinuation): ``discontinued_180d`` — re-anchored to first biologic fill
  C (persistence): ``persistent_at_180d`` — re-anchored to first biologic fill

For each cohort, writes:
  data/rwd/optum/<cohort>/
    e2i_ml_v3_patient_journeys.parquet
    e2i_ml_v3_treatment_events.parquet
    e2i_ml_v3_hcp_profiles.parquet
    e2i_ml_v3_split_registry.json
    data_dictionary.csv

Plus a top-level ``attrition_report.csv`` documenting filter drop counts.

Usage:
    python scripts/convert_optum_rwd.py
    python scripts/convert_optum_rwd.py --cohort initiation
    python scripts/convert_optum_rwd.py --max-patients 500 --pilot-audit
    python scripts/convert_optum_rwd.py --dry-run --verbose
"""

from __future__ import annotations

import argparse
import calendar
import logging
import re
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import rwd_common as rwdc  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants — spec §3-§8                                                      #
# --------------------------------------------------------------------------- #

DEFAULT_INPUT = PROJECT_ROOT / "data" / "rwd" / "Optum_Parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "rwd" / "optum"

LOOKBACK_DAYS = 180
PREDICTION_DAYS = 180

# Enrollment-window regime constants (Tier 1A bifurcation, plan v3 §3).
# Default = production (360/180). Research regime (180/90) trades stricter
# enrollment-feasibility for a larger eligible cohort and is gated behind the
# `--enrollment-regime research` CLI flag (or `enrollment_regime="research"`
# kwarg) per plan §3 Tier 1A "MAYBE" branch — domain-expert sign-off required
# before research-regime artifacts are used downstream of pure feasibility
# analysis. The empirical anchor (`docs/results/optum_initiation_revalidation_20260510.md`)
# showed research-regime n=1697 crosses the perm p<0.05 GENUINE threshold at
# n_train_positives=~34 vs production-regime n=1294 at ~22.
ENROLLMENT_REGIMES: dict[str, dict[str, int]] = {
    "production": {"pre_days": 360, "post_days": 180},
    "research": {"pre_days": 180, "post_days": 90},
}
DEFAULT_ENROLLMENT_REGIME = "production"
# Module-level aliases for the production regime, retained for any external
# caller that imports the constants directly. Per-converter values live on the
# OptumDataConverter instance attributes `enrollment_pre_days` /
# `enrollment_post_days`, which the cohort-build path uses.
ENROLLMENT_PRE_DAYS = ENROLLMENT_REGIMES[DEFAULT_ENROLLMENT_REGIME]["pre_days"]
ENROLLMENT_POST_DAYS = ENROLLMENT_REGIMES[DEFAULT_ENROLLMENT_REGIME]["post_days"]
WASHOUT_DAYS = 30
BIOLOGIC_DISCONT_GAP_DAYS = 90
BIOLOGIC_PERSISTENCE_GAP_DAYS = 60

# Qualifying CSU diagnosis codes. Optum codes are stored without the dot
# (``L509``), so we match on the de-dotted prefix set.
CSU_DX_PREFIXES = ("L501", "L508", "L509")
EXCLUSION_DX_PREFIXES = {
    "secondary_urticaria": ("T78.40", "T78.1", "L506", "L504", "L502", "L505", "L563"),
    "mastocytosis": ("Q822", "D4702"),
    # Ranges (prefix-based) — ICD-10 chapter C is all cancer, etc.
    "pregnancy_range": ("O",),
    "cancer_range": ("C",),
    "immunosuppression_single": ("B20",),
    "immunosuppression_range": ("D8",),
}

# CSU biologics: Xolair (omalizumab, J2357, NDC prefix 50242-04) and
# Dupixent (dupilumab, J0517 misspelling → J0517 is actually eculizumab;
# but the analyst spec lists it. Dupixent NDC prefix is 0024-59.)
CSU_BIOLOGIC_HCPCS = {"J2357", "J0517"}
CSU_BIOLOGIC_NDC_PREFIXES = ("50242", "00024", "0024")
CSU_BIOLOGIC_GENERICS = ("omalizumab", "dupilumab")
CSU_BIOLOGIC_BRANDS = ("XOLAIR", "DUPIXENT")

COMORBIDITY_CODES: dict[str, tuple[str, ...]] = {
    "atopic_dermatitis": ("L20",),
    "asthma": ("J45",),
    "allergic_rhinitis": ("J30",),
    "anxiety": ("F40", "F41"),
    "depression": ("F32", "F33"),
    "thyroid_autoimmune": ("E063", "E050"),
    "nsaid_hypersensitivity": ("Z886", "T39"),
    "angioedema": ("T783",),
}

CSU_LABS_LOINC: dict[str, tuple[str, ...]] = {
    "ige_total": ("19113-0", "2683-2"),
    "eosinophil": ("6206-7",),
    "crp": ("1988-5",),
    "tpo_ab": ("3051-0", "3053-6"),
    "free_t4": ("3024-7",),
    "tsh": ("3016-3",),
    "ana": ("14741-9",),
    "cbc": ("26453-1",),
}

NON_TARGET_DRUG_CLASSES: dict[str, tuple[str, ...]] = {
    "h1_1g": ("diphenhydramine", "hydroxyzine"),
    "h1_2g": (
        "cetirizine",
        "loratadine",
        "fexofenadine",
        "desloratadine",
        "levocetirizine",
    ),
    "h2": ("famotidine", "ranitidine", "cimetidine"),
    "ltra": ("montelukast", "zafirlukast"),
    "sys_steroid": ("prednisone", "methylprednisolone", "dexamethasone"),
    "top_steroid": ("triamcinolone", "hydrocortisone", "clobetasol"),
    "immunosupp": (
        "cyclosporine",
        "methotrexate",
        "azathioprine",
        "mycophenolate",
    ),
}

# Minimal zip3 → urban_rural crosswalk (approximation — documented in data
# dictionary). Major metropolitan zip3s map to "urban"; everything else
# defaults to "suburban". A full RUCA crosswalk would replace this.
URBAN_ZIP3_PREFIXES: frozenset[str] = frozenset(
    {
        # NYC, LA, Chicago, Houston, Philadelphia, Phoenix, SF, Seattle,
        # Boston, DC, Atlanta, Miami
        "100",
        "101",
        "102",
        "103",
        "104",
        "112",
        "900",
        "902",
        "906",
        "907",
        "606",
        "607",
        "608",
        "770",
        "772",
        "190",
        "191",
        "850",
        "852",
        "940",
        "941",
        "981",
        "020",
        "021",
        "022",
        "200",
        "300",
        "330",
    }
)

ALLOWED_COHORTS = ("initiation", "discontinuation", "persistence", "all")


# --------------------------------------------------------------------------- #
# OptumDataConverter                                                          #
# --------------------------------------------------------------------------- #


class OptumDataConverter:
    """Convert Optum parquet RWD into cohort-specific canonical parquet."""

    def __init__(
        self,
        parquet_dir: Path,
        output_dir: Path,
        cohorts: tuple[str, ...] = ("initiation", "discontinuation", "persistence"),
        max_patients: int | None = None,
        pilot_audit: bool = False,
        enrollment_regime: str = DEFAULT_ENROLLMENT_REGIME,
        extract_ym: str | None = None,
    ) -> None:
        if enrollment_regime not in ENROLLMENT_REGIMES:
            allowed = sorted(ENROLLMENT_REGIMES.keys())
            raise ValueError(
                f"enrollment_regime={enrollment_regime!r} not in {allowed}; "
                f"plan v3 §3 Tier 1A defines exactly two regimes."
            )
        self.parquet_dir = Path(parquet_dir)
        self.output_dir = Path(output_dir)
        self.cohorts = cohorts
        self.max_patients = max_patients
        self.pilot_audit = pilot_audit
        self.enrollment_regime = enrollment_regime
        self.enrollment_pre_days = ENROLLMENT_REGIMES[enrollment_regime]["pre_days"]
        self.enrollment_post_days = ENROLLMENT_REGIMES[enrollment_regime]["post_days"]
        self.now_iso = datetime.now().isoformat()

        # Issue #155 §3: source_timestamp / ingestion_timestamp / data_lag_hours.
        # `extract_ym` (YYYYMM) is the Optum vendor's drop month — month
        # granularity only, so we use the LAST_DAY at 23:59:59 UTC as the
        # WORST-CASE (most conservative) source-timestamp estimate. This
        # never UNDERSTATES lag.
        #
        # If `extract_ym` is not passed, attempt to infer from a YYYYMM
        # substring in `parquet_dir.name` (e.g. "Optum_202604"). When
        # neither input nor inference yields a YYYYMM, the source/lag
        # fields remain None and downstream KPI views document the gap.
        resolved_ym = extract_ym or self._infer_extract_ym(self.parquet_dir)
        self.extract_ym: str | None = resolved_ym
        self.source_timestamp_iso: str | None = None
        self.ingestion_timestamp_iso: str | None = None
        self.data_lag_hours: int | None = None
        if resolved_ym is not None:
            self._compute_drop_timestamps(resolved_ym)

        # Loaded DataFrames, indexed per patient for speed
        self.demo: pd.DataFrame = pd.DataFrame()
        self.med: pd.DataFrame = pd.DataFrame()
        self.proc: pd.DataFrame = pd.DataFrame()
        self.lab: pd.DataFrame = pd.DataFrame()
        self.inpatient: pd.DataFrame = pd.DataFrame()
        self.provider: pd.DataFrame = pd.DataFrame()

        self._med_by_pat: dict[int, pd.DataFrame] = {}
        self._proc_by_pat: dict[int, pd.DataFrame] = {}
        self._lab_by_pat: dict[int, pd.DataFrame] = {}
        self._inpatient_by_pat: dict[int, pd.DataFrame] = {}
        self._provider_by_npi: dict[str, str] = {}  # obfuscated npi → specialty

        # ID maps (regenerated per cohort build so the output is self-contained)
        self._attrition: list[tuple[str, int]] = []

    # ------------------------------------------------------------------ #
    # Issue #155 §3: source_timestamp from Optum extract_ym               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _infer_extract_ym(parquet_dir: Path) -> str | None:
        """Walk path components RIGHT-TO-LEFT, returning the first YYYYMM hit.

        Heuristic fallback for callers that did NOT pass ``--extract-ym``.
        Right-to-left so the deepest (most-specific) directory wins when the
        input layout is e.g. ``/vendor/202604/optum`` — the basename "optum"
        contains no date but the parent "202604" does.

        Returns the first match or None — caller treats None as "do not
        populate source_timestamp; leave None and document in
        data_dictionary.csv".
        """
        pattern = re.compile(r"(19[9]\d|20\d\d)(0[1-9]|1[0-2])")
        for part in reversed(parquet_dir.parts):
            match = pattern.search(part)
            if match is not None:
                return match.group(0)
        return None

    def _compute_drop_timestamps(self, extract_ym: str) -> None:
        """Populate source_timestamp / ingestion_timestamp / data_lag_hours.

        Issue #155 §3 derivation:
          - extract_ym (YYYYMM) → LAST_DAY 23:59:59 UTC. Worst-case (most
            conservative) source-timestamp estimate — never understates lag.
          - ingestion_timestamp = mtime of the first parquet found in
            ``parquet_dir`` (any of {demographics, medication, procedure,
            lab, inpatientdata, provider}).parquet. When NO parquet exists,
            fall back to ``datetime.now()`` so the field is still populated.
          - data_lag_hours = floor((ingestion - source).total_seconds() / 3600).
            CAN BE NEGATIVE if the parquet predates the nominal extract month
            (rare but possible for back-dated drops); leave the negative value
            in place so downstream consumers detect the anomaly.
        """
        if len(extract_ym) != 6 or not extract_ym.isdigit():
            logger.warning(
                "extract_ym=%r is not YYYYMM — skipping source_timestamp population.",
                extract_ym,
            )
            return
        year = int(extract_ym[:4])
        month = int(extract_ym[4:6])
        if not (1 <= month <= 12):
            logger.warning(
                "extract_ym=%r has invalid month — skipping source_timestamp.",
                extract_ym,
            )
            return
        last_day = calendar.monthrange(year, month)[1]
        source_ts = datetime(year, month, last_day, 23, 59, 59, tzinfo=UTC)

        # Pick the first parquet that exists for ingestion_timestamp.
        ingest_ts: datetime | None = None
        for name in ("demographics", "medication", "procedure", "lab", "inpatientdata", "provider"):
            p = self.parquet_dir / f"{name}.parquet"
            if p.exists():
                try:
                    ingest_ts = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
                    break
                except OSError:
                    continue
        if ingest_ts is None:
            ingest_ts = datetime.now(tz=UTC)
            logger.info(
                "No parquet files in %s — using current UTC time as ingestion_timestamp fallback.",
                self.parquet_dir,
            )

        self.source_timestamp_iso = source_ts.isoformat()
        self.ingestion_timestamp_iso = ingest_ts.isoformat()
        self.data_lag_hours = int((ingest_ts - source_ts).total_seconds() // 3600)

    # ------------------------------------------------------------------ #
    # Entry point                                                         #
    # ------------------------------------------------------------------ #

    def convert_all(self) -> dict[str, dict[str, int]]:
        """Run the pipeline. Returns per-cohort record counts."""
        logger.info(
            "Enrollment regime: %s (pre=%dd, post=%dd)",
            self.enrollment_regime,
            self.enrollment_pre_days,
            self.enrollment_post_days,
        )
        logger.info("Reading Optum parquet from %s", self.parquet_dir)
        self._read_parquets()
        self._clean()
        self._index_by_patient()

        cohort_counts: dict[str, dict[str, int]] = {}
        for cohort in self.cohorts:
            logger.info("=" * 60)
            logger.info("Building cohort: %s", cohort)
            logger.info("=" * 60)
            self._attrition = []
            counts = self._build_and_write_cohort(cohort)
            cohort_counts[cohort] = counts

            # Per-cohort attrition report
            rwdc.write_attrition_report(self.output_dir / cohort, self._attrition)

        return cohort_counts

    # ------------------------------------------------------------------ #
    # Parquet reading + cleaning                                          #
    # ------------------------------------------------------------------ #

    def _read_parquets(self) -> None:
        def _read(name: str) -> pd.DataFrame:
            p = self.parquet_dir / f"{name}.parquet"
            if not p.exists():
                raise FileNotFoundError(p)
            df = pd.read_parquet(p)
            logger.info("  %s: %d rows", name, len(df))
            return df

        self.demo = _read("demographics")
        self.med = _read("medication")
        self.proc = _read("procedure")
        self.lab = _read("lab")
        self.inpatient = _read("inpatientdata")
        self.provider = _read("provider")

    def _clean(self) -> None:
        # demographics
        self.demo = self.demo.drop_duplicates(subset=["patid"]).copy()
        for c in ("indexdt", "eligeff", "eligend"):
            if c in self.demo.columns:
                self.demo[c] = pd.to_datetime(self.demo[c], errors="coerce")
        # Normalise diagcode with dot insertion for easier matching downstream
        self.demo["diagcode_raw"] = self.demo["diagcode"].astype(str)

        if self.max_patients is not None:
            keep = self.demo.sort_values("patid").head(self.max_patients)["patid"]
            self.demo = self.demo[self.demo["patid"].isin(keep)]
            self.med = self.med[self.med["patid"].isin(keep)]
            self.proc = self.proc[self.proc["patid"].isin(keep)]
            self.lab = self.lab[self.lab["patid"].isin(keep)]
            self.inpatient = self.inpatient[self.inpatient["patid"].isin(keep)]
            logger.info("  --max-patients: limited to %d", self.max_patients)

        for df, dcols in [
            (self.med, ["medication_date"]),
            (self.proc, ["proc_date"]),
            (self.lab, ["fst_dt"]),
            (self.inpatient, ["admit_date", "disch_date"]),
        ]:
            for c in dcols:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")

        # Provider: npi → taxonomy/specialty (keep first row per npi)
        if "npi" in self.provider.columns and "taxonomy1" in self.provider.columns:
            tmp = self.provider.dropna(subset=["npi"]).drop_duplicates(subset=["npi"])
            self._provider_by_npi = dict(
                zip(tmp["npi"].astype(str), tmp["taxonomy1"].fillna("").astype(str), strict=False)
            )

    def _index_by_patient(self) -> None:
        for src, tgt in (
            (self.med, self._med_by_pat),
            (self.proc, self._proc_by_pat),
            (self.lab, self._lab_by_pat),
            (self.inpatient, self._inpatient_by_pat),
        ):
            if "patid" not in src.columns:
                continue
            for pid, grp in src.groupby("patid"):
                tgt[int(pid)] = grp

    # ------------------------------------------------------------------ #
    # Cohort build orchestration                                          #
    # ------------------------------------------------------------------ #

    def _build_and_write_cohort(self, cohort: str) -> dict[str, int]:
        journeys, events, hcps, split_registry = self._build_cohort(cohort)

        if not journeys:
            logger.warning("  Cohort %s is empty — writing headers only", cohort)

        cohort_dir = self.output_dir / cohort

        # Normalise fields that parquet can't infer (empty dicts / lists).
        _normalise_events_for_parquet(events)
        _normalise_journeys_for_parquet(journeys)
        _normalise_hcps_for_parquet(hcps)

        # Item C of the engineering-actionable arc: gate forbidden columns
        # at the cohort-builder boundary so post-index leakage cannot reach
        # the data_preparer state. Targets (treatment_initiated,
        # initiated_biologic_180d, etc.) are explicitly preserved — see
        # OPTUM_TARGETS in optum_feature_manifest.py. The gate filters the
        # journeys list in-place before parquet serialisation.
        from src.data.manifests.optum_feature_manifest import (
            OPTUM_FORBIDDEN_NON_TARGET,
        )

        gated_journeys = _drop_forbidden_columns(journeys, OPTUM_FORBIDDEN_NON_TARGET)
        rwdc.write_records(cohort_dir, "e2i_ml_v3_patient_journeys", gated_journeys, fmt="parquet")
        rwdc.write_records(cohort_dir, "e2i_ml_v3_treatment_events", events, fmt="parquet")
        rwdc.write_records(cohort_dir, "e2i_ml_v3_hcp_profiles", hcps, fmt="parquet")
        rwdc.write_records(cohort_dir, "e2i_ml_v3_split_registry", split_registry, fmt="json")

        # Data dictionary
        rwdc.write_data_dictionary(cohort_dir, self._build_data_dictionary(cohort))

        counts = {
            "patient_journeys": len(journeys),
            "treatment_events": len(events),
            "hcp_profiles": len(hcps),
            "split_registry": len(split_registry),
        }

        if self.pilot_audit and journeys:
            self._run_pilot_audit(cohort, journeys)

        return counts

    def _build_cohort(
        self, cohort: str
    ) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
    ]:
        """Build a single cohort's records and split registry.

        Returns (journeys, events, hcps, split_registry).
        """
        all_patids = sorted(self.demo["patid"].unique().tolist())
        # Pin enrollment-regime label as the first row of the attrition log so
        # downstream consumers (Tier 1B/1C runs) can audit which regime
        # produced the cohort artifact (plan v3 §3 Tier 1A).
        self._attrition.append(
            (
                f"{cohort}: enrollment_regime={self.enrollment_regime} "
                f"(pre={self.enrollment_pre_days}d, post={self.enrollment_post_days}d)",
                len(all_patids),
            )
        )
        self._attrition.append((f"{cohort}: start", len(all_patids)))

        # 1. Age gate
        demo = self.demo[(self.demo["age"] >= 18) & (self.demo["age"] <= 89)]
        pids = set(demo["patid"])
        self._attrition.append((f"{cohort}: age 18-89", len(pids)))

        # 2. Continuous enrollment flag
        demo = demo[demo["continuous_enrollment"] == 1]
        pids = set(demo["patid"])
        self._attrition.append((f"{cohort}: continuous_enrollment=1", len(pids)))

        # 3. L50.x diagcode on demographics (necessary for all cohorts)
        demo = demo[demo["diagcode_raw"].str.upper().str.startswith(CSU_DX_PREFIXES)]
        pids = set(demo["patid"])
        self._attrition.append((f"{cohort}: L50.x diagcode present", len(pids)))

        # 4. Per-patient: derive index date + apply exclusions + temporal window
        records_pass: list[tuple[int, pd.Timestamp, dict[str, Any]]] = []
        n_smart_index_fallback = 0
        for patid in sorted(pids):
            demo_row = demo[demo["patid"] == patid].iloc[0]
            index_date = self._derive_index_date(patid, cohort, demo_row)

            # Backlog #19 smart-index fallback (cohort A only): the default
            # `_derive_index_date` picks the earliest clinical anchor without
            # considering enrollment-window feasibility. When a patient's
            # earliest anchor predates the [eligeff + self.enrollment_pre_days,
            # eligend - self.enrollment_post_days] feasibility band but a later
            # anchor in the same record fits the band, retry with the
            # feasibility-aware derivation. Cohorts B/C re-anchor on first
            # biologic fill — re-anchoring there changes cohort semantics
            # (disc/pers measure outcomes 180d post-FIRST-fill) so the
            # fallback intentionally does not apply.
            used_smart_index = False
            if cohort == "initiation":
                pass1_failed = index_date is None or not self._check_enrollment_window(
                    demo_row, index_date
                )
                if pass1_failed:
                    smart_idx = self._derive_index_date_feasibility_aware(patid, demo_row)
                    if smart_idx is not None:
                        index_date = smart_idx
                        used_smart_index = True

            if index_date is None:
                continue

            if not self._check_enrollment_window(demo_row, index_date):
                continue

            if self._has_exclusion_condition(patid, index_date, demo_row):
                continue

            # Cohort-specific exclusions
            if cohort == "initiation":
                if self._had_biologic_pre_index(patid, index_date):
                    continue  # treatment-naïveté violation (30d washout)

            # Counter only increments for patients who actually survive all
            # filters via the fallback. Codex MEDIUM-1: incrementing earlier
            # would overstate net cohort addition when a fallback-rescued
            # index date later trips the exclusion or washout gates.
            if used_smart_index:
                n_smart_index_fallback += 1

            record = {
                "patid": patid,
                "index_date": index_date,
                "demo_row": demo_row,
            }
            records_pass.append((patid, index_date, record))

        if cohort == "initiation":
            self._attrition.append((f"{cohort}: smart-index fallback hits", n_smart_index_fallback))
        self._attrition.append(
            (f"{cohort}: after index + enrollment + exclusions", len(records_pass))
        )

        if not records_pass:
            logger.warning("  Cohort %s has 0 eligible patients", cohort)
            return [], [], [], []

        # 5. For B/C: require biologic initiation event
        if cohort in ("discontinuation", "persistence"):
            records_pass = [
                (p, idx, rec)
                for p, idx, rec in records_pass
                if self._first_biologic_fill(p) is not None
            ]
            self._attrition.append((f"{cohort}: with biologic initiation", len(records_pass)))

        # 6. Build journey dicts + compute features + target
        journeys: list[dict[str, Any]] = []
        for patid, index_date, _ in records_pass:
            journey = self._build_journey_record(patid, index_date, cohort)
            if journey is not None:
                journeys.append(journey)

        self._attrition.append((f"{cohort}: journeys constructed", len(journeys)))

        # 7. Chronological split
        split_result = rwdc.apply_chronological_split(
            journeys,
            date_key="journey_start_date",
            id_key="patient_id",
        )

        split_config_id = str(uuid.uuid4())
        split_registry = rwdc.build_split_registry(
            split_config_id=split_config_id,
            config_name=f"optum_{cohort}",
            config_version="1.0.0",
            split_dates=split_result["split_dates"],
            created_at=self.now_iso,
        )
        for j in journeys:
            j["split_config_id"] = split_config_id

        # 8. Treatment events + HCP profiles from the kept patients
        kept_patids = {j["_patid"] for j in journeys}
        events = self._build_treatment_events(kept_patids, journeys)
        hcps = self._build_hcp_profiles(kept_patids)

        # 9. Strip internal fields before return
        for j in journeys:
            j.pop("_patid", None)

        logger.info(
            "  Cohort %s: %d journeys, %d events, %d hcps, splits=%s",
            cohort,
            len(journeys),
            len(events),
            len(hcps),
            split_result["counts"],
        )
        return journeys, events, hcps, split_registry

    # ------------------------------------------------------------------ #
    # Index-date derivation (§3.2)                                        #
    # ------------------------------------------------------------------ #

    def _derive_index_date(
        self, patid: int, cohort: str, demo_row: pd.Series
    ) -> pd.Timestamp | None:
        """Derive a claim-dated index date, never using vendor ``indexdt``.

        Cohort A (initiation): qualifying-dx rule §3.2.
          Priority 1: ≥2 distinct L50.x claim dates (inpatient diag1..5) → 2nd.
          Priority 2: single L50.x inpatient claim → admit_date.
          Priority 3 (pragmatic, documented): earliest claim-dated event
            (med/proc/lab) occurring within enrollment window [eligeff,
            eligend]. Used only when primaries are unavailable; avoids the
            vendor ``indexdt`` leakage by deriving the anchor from an
            observed claim rather than a vendor-assigned field.

        Cohort B/C: re-anchor to first biologic fill date (medication.parquet
        filtered to CSU biologic NDC/HCPCS/brand).
        """
        if cohort in ("discontinuation", "persistence"):
            return self._first_biologic_fill(patid)

        # Cohort A
        ip_dates = self._inpatient_l50_dates(patid)
        if len(ip_dates) >= 2:
            return ip_dates[1]
        if len(ip_dates) == 1:
            return ip_dates[0]

        # Pragmatic fallback: earliest claim-dated event within enrollment
        eligeff = demo_row.get("eligeff")
        if pd.isna(eligeff):
            return None
        candidates: list[pd.Timestamp] = []
        for src, col in (
            (self._med_by_pat, "medication_date"),
            (self._proc_by_pat, "proc_date"),
            (self._lab_by_pat, "fst_dt"),
        ):
            grp = src.get(patid)
            if grp is None or col not in grp.columns:
                continue
            dates = grp[col].dropna()
            dates = dates[dates >= eligeff]
            if len(dates):
                candidates.append(dates.min())
        if not candidates:
            return None
        return min(candidates)

    def _derive_index_date_feasibility_aware(
        self, patid: int, demo_row: pd.Series
    ) -> pd.Timestamp | None:
        """Cohort-A smart-index fallback. Pick the earliest clinical anchor
        that lies inside the enrollment-feasibility band
        ``[eligeff + self.enrollment_pre_days, eligend - self.enrollment_post_days]``.

        Used only when ``_derive_index_date`` has already been tried and the
        resulting date either was ``None`` or failed
        ``_check_enrollment_window``. Mirrors the priority order of
        ``_derive_index_date``: 2nd-or-only inpatient L50.x admit date,
        then earliest med/proc/lab event, all restricted to the feasibility
        band so a downstream enrollment-window re-check is guaranteed to
        succeed when this returns non-None. Returns ``None`` when eligeff /
        eligend are missing, the feasibility band is empty, or no anchor
        exists inside the band.

        Note on "2nd": the priority-1 rule selects the 2nd chronologically
        ``in-band`` inpatient L50.x admit date, not the 2nd of all inpatient
        dates on file. When a patient has 3 total admits with 2 inside the
        feasibility band, ``_derive_index_date`` would pick the 2nd of all 3;
        this method picks the 2nd of the 2 in-band. The two can differ — by
        construction, since the fallback only fires when the original choice
        already failed enrollment.
        """
        eligeff = demo_row.get("eligeff")
        eligend = demo_row.get("eligend")
        if pd.isna(eligeff) or pd.isna(eligend):
            return None

        feasible_start = eligeff + timedelta(days=self.enrollment_pre_days)
        feasible_end = eligend - timedelta(days=self.enrollment_post_days)
        if feasible_start > feasible_end:
            return None

        ip_dates = self._inpatient_l50_dates(patid)
        ip_feasible = [d for d in ip_dates if feasible_start <= d <= feasible_end]
        if len(ip_feasible) >= 2:
            return ip_feasible[1]
        if len(ip_feasible) == 1:
            return ip_feasible[0]

        candidates: list[pd.Timestamp] = []
        for src, col in (
            (self._med_by_pat, "medication_date"),
            (self._proc_by_pat, "proc_date"),
            (self._lab_by_pat, "fst_dt"),
        ):
            grp = src.get(patid)
            if grp is None or col not in grp.columns:
                continue
            dates = grp[col].dropna()
            dates = dates[(dates >= feasible_start) & (dates <= feasible_end)]
            if len(dates):
                candidates.append(dates.min())
        if not candidates:
            return None
        return min(candidates)

    def _inpatient_l50_dates(self, patid: int) -> list[pd.Timestamp]:
        grp = self._inpatient_by_pat.get(patid)
        if grp is None:
            return []
        mask = pd.Series(False, index=grp.index)
        for c in ("diag1", "diag2", "diag3", "diag4", "diag5"):
            if c in grp.columns:
                mask = mask | grp[c].astype(str).str.upper().str.startswith(CSU_DX_PREFIXES)
        hits = grp.loc[mask, "admit_date"].dropna().sort_values().tolist()
        # Collapse to unique dates
        seen: set[pd.Timestamp] = set()
        out: list[pd.Timestamp] = []
        for d in hits:
            if d not in seen:
                seen.add(d)
                out.append(d)
        return out

    def _first_biologic_fill(self, patid: int) -> pd.Timestamp | None:
        grp = self._med_by_pat.get(patid)
        if grp is None or "medication_date" not in grp.columns:
            return None
        mask = self._csu_biologic_mask(grp)
        dates = grp.loc[mask, "medication_date"].dropna().sort_values()
        if len(dates) == 0:
            return None
        return dates.iloc[0]

    def _csu_biologic_mask(self, med_df: pd.DataFrame) -> pd.Series:
        """Boolean mask for rows whose NDC/HCPCS/brand/generic matches a CSU biologic."""
        m = pd.Series(False, index=med_df.index)
        if "code" in med_df.columns:
            code_s = med_df["code"].astype(str).str.upper()
            # HCPCS direct match
            m = m | code_s.isin(CSU_BIOLOGIC_HCPCS)
            # NDC prefix match (codes are often 11-digit NDCs without dashes)
            for pref in CSU_BIOLOGIC_NDC_PREFIXES:
                m = m | code_s.str.startswith(pref)
        if "Brand_Name" in med_df.columns:
            b = med_df["Brand_Name"].astype(str).str.upper()
            for brand in CSU_BIOLOGIC_BRANDS:
                m = m | b.str.contains(brand, na=False)
        if "Generic_Name" in med_df.columns:
            g = med_df["Generic_Name"].astype(str).str.lower()
            for gen in CSU_BIOLOGIC_GENERICS:
                m = m | g.str.contains(gen, na=False)
        return m

    # ------------------------------------------------------------------ #
    # Eligibility checks (§5)                                             #
    # ------------------------------------------------------------------ #

    def _check_enrollment_window(self, demo_row: pd.Series, index_date: pd.Timestamp) -> bool:
        eligeff = demo_row.get("eligeff")
        eligend = demo_row.get("eligend")
        if pd.isna(eligeff) or pd.isna(eligend):
            return False
        need_start = index_date - timedelta(days=self.enrollment_pre_days)
        need_end = index_date + timedelta(days=self.enrollment_post_days)
        return bool(eligeff <= need_start and eligend >= need_end)

    def _has_exclusion_condition(
        self, patid: int, index_date: pd.Timestamp, demo_row: pd.Series
    ) -> bool:
        """Return True if any §5 exclusion condition applies in the lookback window."""
        lookback_start = index_date - timedelta(days=LOOKBACK_DAYS)

        demo_code = str(demo_row.get("diagcode_raw") or "").upper()
        for prefix in EXCLUSION_DX_PREFIXES["secondary_urticaria"]:
            if demo_code.startswith(prefix.replace(".", "")):
                return True
        for prefix in EXCLUSION_DX_PREFIXES["mastocytosis"]:
            if demo_code.startswith(prefix):
                return True
        # Pregnancy O00-O9A, cancer C00-C97, immunosuppression D8x/B20
        if demo_code.startswith(EXCLUSION_DX_PREFIXES["pregnancy_range"]):
            return True
        if demo_code.startswith(EXCLUSION_DX_PREFIXES["cancer_range"]):
            return True
        if demo_code.startswith(EXCLUSION_DX_PREFIXES["immunosuppression_single"]):
            return True
        if demo_code.startswith(EXCLUSION_DX_PREFIXES["immunosuppression_range"]):
            return True

        ip = self._inpatient_by_pat.get(patid)
        if ip is not None:
            mask_window = (ip["admit_date"] >= lookback_start) & (ip["admit_date"] < index_date)
            ip_w = ip.loc[mask_window]
            for c in ("diag1", "diag2", "diag3", "diag4", "diag5"):
                if c not in ip_w.columns:
                    continue
                codes = ip_w[c].dropna().astype(str).str.upper()
                for prefix in EXCLUSION_DX_PREFIXES["secondary_urticaria"]:
                    if codes.str.startswith(prefix.replace(".", "")).any():
                        return True
                if codes.str.startswith(EXCLUSION_DX_PREFIXES["cancer_range"]).any():
                    return True
                if codes.str.startswith(EXCLUSION_DX_PREFIXES["pregnancy_range"]).any():
                    return True
        return False

    def _had_biologic_pre_index(self, patid: int, index_date: pd.Timestamp) -> bool:
        """Cohort A washout: any CSU biologic fill within 30 days before index."""
        grp = self._med_by_pat.get(patid)
        if grp is None:
            return False
        mask = self._csu_biologic_mask(grp)
        bio = grp.loc[mask]
        if bio.empty or "medication_date" not in bio.columns:
            return False
        window_start = index_date - timedelta(days=WASHOUT_DAYS)
        return bool(
            ((bio["medication_date"] >= window_start) & (bio["medication_date"] < index_date)).any()
        )

    # ------------------------------------------------------------------ #
    # Feature computation (§7)                                            #
    # ------------------------------------------------------------------ #

    def _compute_features(
        self, patid: int, index_date: pd.Timestamp, demo_row: pd.Series
    ) -> dict[str, Any]:
        lb_start = index_date - timedelta(days=LOOKBACK_DAYS)
        lb_end = index_date - timedelta(days=1)

        feats: dict[str, Any] = {}

        # 7.1 Demographics
        age = rwdc.safe_float(demo_row.get("age"))
        feats["age_at_index"] = age
        feats["age_group"] = rwdc.age_group(age)
        gdr = demo_row.get("gdr_cd")
        feats["gender"] = str(gdr).strip().upper() if pd.notna(gdr) and str(gdr).strip() else "U"
        zip5 = demo_row.get("zipcode_5")
        zip_str = str(zip5).split("_")[0].strip() if pd.notna(zip5) else None
        feats["zip5"] = zip_str
        feats["zip3"] = zip_str[:3] if zip_str and len(zip_str) >= 3 else None
        feats["geographic_region"] = rwdc.map_zipcode_to_region(zip_str)
        feats["insurance_product"] = rwdc.insurance_type(demo_row.get("bus"))
        plan = demo_row.get("product")
        feats["plan_type"] = str(plan) if pd.notna(plan) else None
        feats["urban_rural_code"] = "urban" if feats["zip3"] in URBAN_ZIP3_PREFIXES else "suburban"

        # 7.2 Disease characteristics (lookback)
        l50_counts: dict[str, int] = {"L501": 0, "L508": 0, "L509": 0, "total": 0}
        ang_count = 0
        ip = self._inpatient_by_pat.get(patid)
        if ip is not None:
            ip_w = ip[(ip["admit_date"] >= lb_start) & (ip["admit_date"] <= lb_end)]
            for c in ("diag1", "diag2", "diag3", "diag4", "diag5"):
                if c in ip_w.columns:
                    codes = ip_w[c].dropna().astype(str).str.upper()
                    for pref in ("L501", "L508", "L509"):
                        l50_counts[pref] += int(codes.str.startswith(pref).sum())
                    ang_count += int(codes.str.startswith("T783").sum())
        # demographics-level diagcode counts 1 toward whichever prefix it matches
        demo_code = str(demo_row.get("diagcode_raw") or "").upper()
        for pref in ("L501", "L508", "L509"):
            if demo_code.startswith(pref):
                l50_counts[pref] += 1
                break
        l50_counts["total"] = sum(l50_counts[p] for p in ("L501", "L508", "L509"))
        feats["dx_l50_1_count"] = l50_counts["L501"]
        feats["dx_l50_8_count"] = l50_counts["L508"]
        feats["dx_l50_9_count"] = l50_counts["L509"]
        feats["dx_total_csu"] = l50_counts["total"]
        feats["dx_angioedema_count"] = ang_count
        feats["months_since_first_dx"] = LOOKBACK_DAYS // 30  # approx (no date-of-first-dx)
        feats["csu_chronicity"] = "chronic"  # all patients qualify for CSU by definition

        # 7.3 Comorbidity burden
        atopy_count = 0
        for name, prefixes in COMORBIDITY_CODES.items():
            has_cond, n_claims = self._comorbidity_counts(patid, lb_start, lb_end, prefixes)
            feats[f"has_{name}"] = int(has_cond)
            feats[f"{name}_claim_count"] = n_claims
            if name in ("atopic_dermatitis", "asthma", "allergic_rhinitis"):
                atopy_count += int(has_cond)
        feats["atopy_score"] = atopy_count
        feats["mental_health_flag"] = int(
            feats.get("has_anxiety", 0) or feats.get("has_depression", 0)
        )
        feats["elixhauser_score"] = self._elixhauser_approx(patid, lb_start, lb_end)
        feats["charlson_score"] = self._charlson_approx(patid, lb_start, lb_end)

        # 7.4 Healthcare utilization (lookback)
        office_total, office_allergist, office_derm, office_pcp = (0, 0, 0, 0)
        ed_total, ed_urticaria = (0, 0)
        hosp_total = 0
        unique_providers: set[str] = set()

        proc = self._proc_by_pat.get(patid)
        if proc is not None:
            proc_w = proc[(proc["proc_date"] >= lb_start) & (proc["proc_date"] <= lb_end)]
            if "proc_code" in proc_w.columns:
                pc = proc_w["proc_code"].astype(str).str.upper()
                # E&M codes 99201-99215 ≈ office visits
                em_mask = pc.str.match(r"^992\d{2}$", na=False)
                office_total = int(em_mask.sum())
            if "npi" in proc_w.columns:
                for n in proc_w["npi"].dropna().astype(str):
                    unique_providers.add(n)
                    tax = self._provider_by_npi.get(n, "")
                    # Issue #154 §7.7: replace 4-char prefix matching with
                    # exact full-taxonomy-code matching so subspecialty codes
                    # are classified deliberately (via rwd_common constants),
                    # not by accidental string-prefix collision.
                    if rwdc.taxonomy_in(tax, rwdc.NUCC_ALLERGY_IMMUNOLOGY_CODES):
                        office_allergist += 1
                    elif rwdc.taxonomy_in(tax, rwdc.NUCC_DERMATOLOGY_CODES):
                        office_derm += 1
                    elif rwdc.taxonomy_in(tax, rwdc.NUCC_PCP_CODES):
                        office_pcp += 1

        if ip is not None:
            ip_w = ip[(ip["admit_date"] >= lb_start) & (ip["admit_date"] <= lb_end)]
            hosp_total = len(ip_w)
            ed_total = int(
                ip_w.get("tos_cd", pd.Series(dtype=object))
                .astype(str)
                .str.contains("ED", case=False, na=False)
                .sum()
            )
            if "diag1" in ip_w.columns:
                ed_urticaria = int(
                    (ip_w["diag1"].astype(str).str.upper().str.startswith(CSU_DX_PREFIXES)).sum()
                )

        feats["office_visits_total"] = office_total
        feats["office_visits_allergist"] = office_allergist
        feats["office_visits_dermatology"] = office_derm
        feats["office_visits_pcp"] = office_pcp
        feats["ed_visits_total"] = ed_total
        feats["ed_visits_urticaria_angio"] = ed_urticaria
        feats["hospitalizations_total"] = hosp_total
        feats["unique_providers"] = len(unique_providers)

        # 7.5 Non-target medication exposure (lookback)
        med = self._med_by_pat.get(patid)
        if med is not None:
            med_w = med[(med["medication_date"] >= lb_start) & (med["medication_date"] <= lb_end)]
            # Exclude biologic rows from non-target drug class features to prevent
            # target leakage (§7.5: "NON-TARGET drugs only")
            bio_mask = self._csu_biologic_mask(med_w)
            med_w = med_w.loc[~bio_mask]

            for cls_name, generics in NON_TARGET_DRUG_CLASSES.items():
                ever, n_fills, ds_total, days_since_last = self._drug_class_features(
                    med_w, generics, index_date
                )
                feats[f"{cls_name}_ever_filled"] = ever
                feats[f"{cls_name}_fill_count"] = n_fills
                feats[f"{cls_name}_days_supply_total"] = ds_total
                feats[f"{cls_name}_days_since_last_fill"] = days_since_last
        else:
            for cls_name in NON_TARGET_DRUG_CLASSES:
                feats[f"{cls_name}_ever_filled"] = 0
                feats[f"{cls_name}_fill_count"] = 0
                feats[f"{cls_name}_days_supply_total"] = 0
                feats[f"{cls_name}_days_since_last_fill"] = None

        # 7.6 Lab features (lookback)
        lab = self._lab_by_pat.get(patid)
        if lab is not None:
            lab_w = lab[(lab["fst_dt"] >= lb_start) & (lab["fst_dt"] <= lb_end)]
            for lab_name, codes in CSU_LABS_LOINC.items():
                tested, last_result, abnormal = self._lab_features(lab_w, codes)
                feats[f"{lab_name}_tested"] = int(tested)
                feats[f"{lab_name}_result_last"] = last_result
                feats[f"{lab_name}_abnormal_flag"] = abnormal
        else:
            for lab_name in CSU_LABS_LOINC:
                feats[f"{lab_name}_tested"] = 0
                feats[f"{lab_name}_result_last"] = None
                feats[f"{lab_name}_abnormal_flag"] = None

        # 7.7 Provider mix (lookback)
        primary_tax = None
        if proc is not None and "npi" in proc.columns:
            proc_w = proc[(proc["proc_date"] >= lb_start) & (proc["proc_date"] <= lb_end)]
            tax_series = (
                proc_w["npi"].dropna().astype(str).map(lambda n: self._provider_by_npi.get(n, ""))
            )
            tax_series = tax_series[tax_series != ""]
            if len(tax_series):
                primary_tax = tax_series.mode().iat[0]
                # HHI concentration
                shares = tax_series.value_counts(normalize=True).to_numpy()
                feats["specialist_concentration"] = float((shares**2).sum())
            else:
                feats["specialist_concentration"] = None
        else:
            feats["specialist_concentration"] = None
        feats["primary_specialist_type"] = primary_tax
        # Issue #154 §7.7: full-taxonomy-code matching against the NUCC
        # specialty groupings declared in rwd_common. The legacy 4-char
        # prefix matching ("207K", "207N") collapsed unrelated subspecialty
        # codes that share a prefix; exact matching against the full code
        # list is auditable and self-documenting.
        feats["saw_allergist_flag"] = int(
            rwdc.taxonomy_in(primary_tax, rwdc.NUCC_ALLERGY_IMMUNOLOGY_CODES)
        )
        feats["saw_dermatologist_flag"] = int(
            rwdc.taxonomy_in(primary_tax, rwdc.NUCC_DERMATOLOGY_CODES)
        )

        return feats

    def _comorbidity_counts(
        self,
        patid: int,
        lb_start: pd.Timestamp,
        lb_end: pd.Timestamp,
        prefixes: tuple[str, ...],
    ) -> tuple[bool, int]:
        has_cond = False
        n_claims = 0
        prefixes_nodots = tuple(p.replace(".", "") for p in prefixes)
        # demographics single diagcode (no claim date — counts as 1 if in lookback
        # proxy: always in lookback since we don't have dx claim-date for demo)
        demo_row = self.demo[self.demo["patid"] == patid]
        if len(demo_row):
            demo_code = str(demo_row.iloc[0].get("diagcode_raw") or "").upper()
            if demo_code.startswith(prefixes_nodots):
                has_cond = True
                # Don't double-count demo-only dx in claim counts
        ip = self._inpatient_by_pat.get(patid)
        if ip is not None:
            ip_w = ip[(ip["admit_date"] >= lb_start) & (ip["admit_date"] <= lb_end)]
            for c in ("diag1", "diag2", "diag3", "diag4", "diag5"):
                if c in ip_w.columns:
                    codes = ip_w[c].dropna().astype(str).str.upper()
                    n_claims += int(codes.str.startswith(prefixes_nodots).sum())
            if n_claims > 0:
                has_cond = True
        return has_cond, n_claims

    def _elixhauser_approx(self, patid: int, lb_start: pd.Timestamp, lb_end: pd.Timestamp) -> int:
        """Minimal Elixhauser proxy: count of distinct ICD-10 chapters in lookback."""
        ip = self._inpatient_by_pat.get(patid)
        if ip is None:
            return 0
        ip_w = ip[(ip["admit_date"] >= lb_start) & (ip["admit_date"] <= lb_end)]
        chapters: set[str] = set()
        for c in ("diag1", "diag2", "diag3", "diag4", "diag5"):
            if c in ip_w.columns:
                codes = ip_w[c].dropna().astype(str).str.upper()
                for code in codes:
                    if len(code) >= 1:
                        chapters.add(code[0])
        return len(chapters)

    def _charlson_approx(self, patid: int, lb_start: pd.Timestamp, lb_end: pd.Timestamp) -> int:
        """Minimal Charlson proxy: distinct high-severity categories present."""
        ip = self._inpatient_by_pat.get(patid)
        if ip is None:
            return 0
        cats = {
            "mi": ("I21", "I22", "I252"),
            "chf": ("I099", "I110", "I130", "I132", "I255", "I420", "I425"),
            "cancer": ("C",),
            "diabetes": ("E10", "E11", "E12", "E13", "E14"),
            "renal": ("N18", "N19"),
        }
        ip_w = ip[(ip["admit_date"] >= lb_start) & (ip["admit_date"] <= lb_end)]
        present: set[str] = set()
        for c in ("diag1", "diag2", "diag3", "diag4", "diag5"):
            if c in ip_w.columns:
                codes = ip_w[c].dropna().astype(str).str.upper()
                for cat_name, prefixes in cats.items():
                    if codes.str.startswith(prefixes).any():
                        present.add(cat_name)
        return len(present)

    def _drug_class_features(
        self,
        med_w: pd.DataFrame,
        generics: tuple[str, ...],
        index_date: pd.Timestamp,
    ) -> tuple[int, int, int, int | None]:
        if med_w.empty or "Generic_Name" not in med_w.columns:
            return 0, 0, 0, None
        gen_s = med_w["Generic_Name"].astype(str).str.lower()
        mask = gen_s.isin([g.lower() for g in generics])
        cls = med_w.loc[mask]
        if cls.empty:
            return 0, 0, 0, None
        n_fills = len(cls)
        ds_total = int(cls["days_sup"].fillna(0).sum()) if "days_sup" in cls.columns else 0
        last_date = cls["medication_date"].max()
        days_since_last = int((index_date - last_date).days) if pd.notna(last_date) else None
        return 1, n_fills, ds_total, days_since_last

    def _lab_features(
        self, lab_w: pd.DataFrame, codes: tuple[str, ...]
    ) -> tuple[bool, float | None, int | None]:
        if lab_w.empty or "loinc_cd" not in lab_w.columns:
            return False, None, None
        mask = lab_w["loinc_cd"].astype(str).isin(codes)
        sel = lab_w.loc[mask].sort_values("fst_dt")
        if sel.empty:
            return False, None, None
        last = sel.iloc[-1]
        last_result = rwdc.safe_float(last.get("rslt_nbr"))
        abnl = last.get("abnl_cd")
        abnormal: int | None
        if pd.notna(abnl) and str(abnl).strip():
            abnormal = 1
        else:
            abnormal = 0
        return True, last_result, abnormal

    # ------------------------------------------------------------------ #
    # Target derivations (§8)                                             #
    # ------------------------------------------------------------------ #

    def _target_initiated_biologic_180d(self, patid: int, index_date: pd.Timestamp) -> int:
        end = index_date + timedelta(days=PREDICTION_DAYS)
        grp = self._med_by_pat.get(patid)
        if grp is None:
            return 0
        mask = self._csu_biologic_mask(grp)
        bio = grp.loc[mask]
        if bio.empty:
            return 0
        in_window = (bio["medication_date"] >= index_date) & (bio["medication_date"] <= end)
        return int(in_window.any())

    def _target_discontinued_180d(self, patid: int, init_date: pd.Timestamp) -> int:
        """Gap > 90 days between (fill_end) and next fill within 180 days of init."""
        end = init_date + timedelta(days=PREDICTION_DAYS)
        grp = self._med_by_pat.get(patid)
        if grp is None:
            return 1  # no further fills = discontinued
        mask = self._csu_biologic_mask(grp)
        bio = grp.loc[mask].sort_values("medication_date")
        bio = bio[(bio["medication_date"] >= init_date) & (bio["medication_date"] <= end)]
        if bio.empty:
            return 1
        for i in range(len(bio) - 1):
            fill_date = bio.iloc[i]["medication_date"]
            ds = rwdc.safe_int(bio.iloc[i].get("days_sup")) or 0
            fill_end = fill_date + timedelta(days=ds)
            next_fill = bio.iloc[i + 1]["medication_date"]
            if (next_fill - fill_end).days > BIOLOGIC_DISCONT_GAP_DAYS:
                return 1
        # Last fill end extends past prediction end? → persistent
        last = bio.iloc[-1]
        ds = rwdc.safe_int(last.get("days_sup")) or 0
        last_end = last["medication_date"] + timedelta(days=ds)
        return int(last_end < end - timedelta(days=BIOLOGIC_DISCONT_GAP_DAYS))

    def _target_persistent_at_180d(self, patid: int, init_date: pd.Timestamp) -> int:
        """Any fill active at day 180 (days_supply-based, no gap > 60d)."""
        target_day = init_date + timedelta(days=PREDICTION_DAYS)
        grp = self._med_by_pat.get(patid)
        if grp is None:
            return 0
        mask = self._csu_biologic_mask(grp)
        bio = grp.loc[mask].sort_values("medication_date")
        if bio.empty:
            return 0
        for _, row in bio.iterrows():
            fd = row["medication_date"]
            ds = rwdc.safe_int(row.get("days_sup")) or 0
            if fd <= target_day <= fd + timedelta(days=ds):
                return 1
        # Check gap criterion: fills spaced < 60d apart up to target_day
        bio = bio[bio["medication_date"] <= target_day]
        if bio.empty:
            return 0
        bio = bio.reset_index(drop=True)
        for i in range(len(bio) - 1):
            gap = (bio.iloc[i + 1]["medication_date"] - bio.iloc[i]["medication_date"]).days
            if gap > BIOLOGIC_PERSISTENCE_GAP_DAYS:
                return 0
        return 1

    # ------------------------------------------------------------------ #
    # Journey record assembly                                             #
    # ------------------------------------------------------------------ #

    def _derive_journey_stage(
        self,
        *,
        cohort: str,
        init_t: int,
        disc_t: int | None,
        pers_t: int | None,
        saw_specialist: bool,
    ) -> str:
        """Map cohort + targets + signals → 7-stage engagement-funnel value.

        Issue #155 §2 / PR #152 row 2 derivation rules (Optum-cohort proxies):

          aware         dx anchored cohort entry, no specialist visit pre-index,
                        no biologic fill in prediction window
          considering   has specialist visit pre-index, no biologic fill in
                        prediction window
          first_fill    biologic fill in prediction window (initiation event)
          adherent      cohort=persistence, persistent_at_180d=1
          discontinued  cohort=discontinuation, discontinued_180d=1
          maintained    cohort=persistence, persistent_at_180d=1 over the
                        full 180-day window (proxy for adherent >= 6mo;
                        180d == ~6mo for CSU biologics)

        Optum claims data is dispensed-only (no Rx-written stream), so
        the `prescribed` value is NOT emitted from this converter — it
        is reserved for cohorts with EHR Rx-write signals. Code paths
        that fall through return `initial_treatment` (legacy value) as
        a safe default so downstream consumers never receive an
        un-derivable empty string.
        """
        # Cohort B / C (already initiated): derive from persistence / disc flags.
        if cohort == "discontinuation":
            if disc_t == 1:
                return "discontinued"
            return "first_fill"  # initiated but not yet discontinued in window
        if cohort == "persistence":
            if pers_t == 1:
                # 180-day persistence in the CSU biologics window ≈ 6mo
                # adherent → maintained per PR #152 derivation.
                return "maintained"
            return "adherent" if init_t == 1 else "first_fill"

        # Cohort A (initiation): derive from init_t + pre-index specialist signal.
        if init_t == 1:
            return "first_fill"
        if saw_specialist:
            return "considering"
        return "aware"

    def _build_journey_record(
        self, patid: int, index_date: pd.Timestamp, cohort: str
    ) -> dict[str, Any] | None:
        demo_row = self.demo[self.demo["patid"] == patid].iloc[0]

        feats = self._compute_features(patid, index_date, demo_row)

        pat_id_str = f"PAT_{patid:012d}"
        pj_id = f"PJ_{patid:012d}"

        # Targets
        init_t = self._target_initiated_biologic_180d(patid, index_date)
        if cohort == "initiation":
            init_date = None
        else:
            init_date = self._first_biologic_fill(patid)

        disc_t = (
            self._target_discontinued_180d(patid, init_date)
            if (cohort == "discontinuation" and init_date is not None)
            else None
        )
        pers_t = (
            self._target_persistent_at_180d(patid, init_date)
            if (cohort == "persistence" and init_date is not None)
            else None
        )

        # Data quality score: fraction of §7 features non-null
        feat_vals = [v for k, v in feats.items() if not k.startswith("_")]
        non_null = sum(1 for v in feat_vals if v is not None and v != "")
        dq_score = round(non_null / max(len(feat_vals), 1), 3)

        # Issue #155 §2: granular 7-stage engagement-funnel value.
        saw_specialist = bool(
            feats.get("saw_allergist_flag") or feats.get("saw_dermatologist_flag")
        )
        granular_stage = self._derive_journey_stage(
            cohort=cohort,
            init_t=init_t,
            disc_t=disc_t,
            pers_t=pers_t,
            saw_specialist=saw_specialist,
        )

        record: dict[str, Any] = {
            "patient_journey_id": pj_id,
            "patient_id": pat_id_str,
            "patient_hash": rwdc.patient_hash(patid),
            "_patid": int(patid),  # internal — stripped before output
            "index_date": rwdc.safe_date(index_date),
            "lookback_start_date": rwdc.safe_date(index_date - timedelta(days=LOOKBACK_DAYS)),
            "prediction_end_date": rwdc.safe_date(index_date + timedelta(days=PREDICTION_DAYS)),
            "journey_start_date": rwdc.safe_date(index_date),
            "journey_end_date": rwdc.safe_date(index_date + timedelta(days=PREDICTION_DAYS)),
            "journey_duration_days": PREDICTION_DAYS + LOOKBACK_DAYS,
            "journey_stage": granular_stage,
            "journey_status": "active",
            "primary_diagnosis_code": rwdc.format_diagcode(str(demo_row.get("diagcode_raw") or "")),
            "primary_diagnosis_desc": "Chronic Spontaneous Urticaria",
            "secondary_diagnosis_codes": [],
            "brand": "competitor",
            "state": None,
            "zip_code": feats.get("zip5"),
            "comorbidities": [],
            "risk_score": None,
            "data_source": "RWD_Claims",
            "data_sources_matched": ["RWD_Claims"],
            "source_match_confidence": None,
            "source_stacking_flag": False,
            "source_combination_method": None,
            "source_timestamp": self.source_timestamp_iso,
            "ingestion_timestamp": self.ingestion_timestamp_iso or self.now_iso,
            "data_lag_hours": self.data_lag_hours,
            "data_split": None,  # set by chronological splitter
            "created_at": self.now_iso,
            "updated_at": self.now_iso,
            "data_quality_score": dq_score,
            # Targets
            "initiated_biologic_180d": init_t,
            "discontinued_180d": disc_t,
            "persistent_at_180d": pers_t,
            "treatment_initiated": init_t,  # backward-compat for tier-0 test runner
            "discontinuation_flag": disc_t,
        }
        # Spread features into flat columns
        record.update(feats)
        return record

    # ------------------------------------------------------------------ #
    # Treatment events + HCP profiles                                     #
    # ------------------------------------------------------------------ #

    def _build_treatment_events(
        self,
        kept_patids: set[int],
        journeys: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Emit canonical treatment_event records for included patients only.

        Events originate from med/proc/lab filtered to [lookback_start, index_date]
        so the downstream ML pipeline observes pre-index events only. The target
        is already encoded on the journey; events are for feature provenance /
        narrative context.
        """
        idx_by_patid = {
            int(j["_patid"]): (
                pd.Timestamp(j["index_date"]),
                pd.Timestamp(j["lookback_start_date"]),
                j["patient_journey_id"],
                j["patient_id"],
            )
            for j in journeys
        }

        events: list[dict[str, Any]] = []
        seq = 0

        def _emit(
            te_seq: int,
            *,
            patid: int,
            event_date: pd.Timestamp | None,
            event_type: str,
            drug_name: str | None = None,
            drug_ndc: str | None = None,
            dosage: str | None = None,
            duration: int | None = None,
            icd: list[str] | None = None,
            cpt: list[str] | None = None,
            loinc: list[str] | None = None,
            lab_values: dict[str, Any] | None = None,
            hcp_id: str | None = None,
        ) -> dict[str, Any]:
            idx, lb, pj, pat = idx_by_patid[patid]
            return {
                "treatment_event_id": f"TE_{te_seq:09d}",
                "patient_journey_id": pj,
                "patient_id": pat,
                "hcp_id": hcp_id,
                "event_date": rwdc.safe_date(event_date),
                "event_type": event_type,
                "event_subtype": None,
                "brand": None,
                "drug_ndc": drug_ndc,
                "drug_name": drug_name,
                "drug_class": None,
                "dosage": dosage,
                "duration_days": duration,
                "icd_codes": icd or [],
                "cpt_codes": cpt or [],
                "loinc_codes": loinc or [],
                "lab_values": lab_values or {},
                "location_type": None,
                "facility_id": None,
                "cost": None,
                "outcome_indicator": None,
                "adverse_event_flag": False,
                "discontinuation_flag": False,
                "discontinuation_reason": None,
                "sequence_number": te_seq,
                "days_from_diagnosis": (event_date - idx).days if event_date else 0,
                "previous_treatment": None,
                "next_treatment": None,
                "data_source": "RWD_Claims",
                "source_timestamp": self.source_timestamp_iso,
                "ingestion_timestamp": self.ingestion_timestamp_iso or self.now_iso,
                "data_split": None,
                "created_at": self.now_iso,
                "updated_at": self.now_iso,
            }

        for patid in sorted(kept_patids):
            idx, lb, _, _ = idx_by_patid[patid]
            # Medication events in lookback window
            grp = self._med_by_pat.get(patid)
            if grp is not None:
                win = grp[(grp["medication_date"] >= lb) & (grp["medication_date"] < idx)]
                for _, row in win.iterrows():
                    events.append(
                        _emit(
                            seq,
                            patid=patid,
                            event_date=row.get("medication_date"),
                            event_type="prescription",
                            drug_name=(
                                str(row["Brand_Name"]).title()
                                if pd.notna(row.get("Brand_Name"))
                                else None
                            ),
                            drug_ndc=(str(row["code"]) if pd.notna(row.get("code")) else None),
                            dosage=(
                                str(row["strength"]) if pd.notna(row.get("strength")) else None
                            ),
                            duration=rwdc.safe_int(row.get("days_sup")),
                        )
                    )
                    seq += 1
            # Procedure events
            grp = self._proc_by_pat.get(patid)
            if grp is not None:
                win = grp[(grp["proc_date"] >= lb) & (grp["proc_date"] < idx)]
                for _, row in win.iterrows():
                    events.append(
                        _emit(
                            seq,
                            patid=patid,
                            event_date=row.get("proc_date"),
                            event_type="procedure",
                            cpt=(
                                [str(row["proc_code"]).upper()]
                                if pd.notna(row.get("proc_code"))
                                else []
                            ),
                        )
                    )
                    seq += 1
            # Lab events (keep top 20 per patient for size; labs can be huge)
            grp = self._lab_by_pat.get(patid)
            if grp is not None:
                win = grp[(grp["fst_dt"] >= lb) & (grp["fst_dt"] < idx)].head(20)
                for _, row in win.iterrows():
                    lab_val = {}
                    if pd.notna(row.get("tst_desc")) and pd.notna(row.get("rslt_nbr")):
                        lab_val[str(row["tst_desc"])] = float(row["rslt_nbr"])
                    events.append(
                        _emit(
                            seq,
                            patid=patid,
                            event_date=row.get("fst_dt"),
                            event_type="lab_test",
                            loinc=([str(row["loinc_cd"])] if pd.notna(row.get("loinc_cd")) else []),
                            lab_values=lab_val,
                        )
                    )
                    seq += 1
            # Inpatient events
            grp = self._inpatient_by_pat.get(patid)
            if grp is not None:
                win = grp[(grp["admit_date"] >= lb) & (grp["admit_date"] < idx)]
                for _, row in win.iterrows():
                    dx_codes = []
                    for c in ("diag1", "diag2", "diag3", "diag4", "diag5"):
                        val = row.get(c)
                        if pd.notna(val):
                            dx_codes.append(rwdc.format_diagcode(str(val)))
                    events.append(
                        _emit(
                            seq,
                            patid=patid,
                            event_date=row.get("admit_date"),
                            event_type="hospitalization",
                            icd=dx_codes,
                        )
                    )
                    seq += 1
        return events

    def _compute_npi_first_fill(
        self,
        kept_patids: set[int],
        npi_rx: dict[str, int],
        brand_launch: pd.Timestamp,
    ) -> tuple[dict[str, int | None], dict[str, bool]]:
        """Per-NPI: days-to-first on-label brand fill (vs brand_launch_date).

        Issue #155 §1 — Rogers diffusion anchor. Returns:

          - ``days_to_first_fill``: dict NPI → int days, or None if no on-label
            fill (HCP becomes ``non_adopter`` in classify_rogers_adoption).
          - ``dupixent_offlabel``: dict NPI → bool (True if HCP has any
            Dupixent fill in scope; Dupixent has NO CSU approval as of
            2026-05-12 so its CSU-cohort fills are off-label).

        On-label = CSU biologic mask MINUS Dupixent (matched on brand-name /
        generic-name / NDC prefix). HCPs whose only fills are off-label
        Dupixent get ``days_to_first_fill=None`` (→ non_adopter), with the
        off-label flag preserved separately so downstream consumers can
        carve them out for cross-indication adoption analysis.
        """
        days_out: dict[str, int | None] = dict.fromkeys(npi_rx)
        offlabel: dict[str, bool] = dict.fromkeys(npi_rx, False)

        med = self.med
        if (
            med is None
            or med.empty
            or "patid" not in med.columns
            or "npi" not in med.columns
            or "medication_date" not in med.columns
        ):
            return days_out, offlabel

        sub = med[med["patid"].isin(kept_patids)].copy()
        if sub.empty:
            return days_out, offlabel

        bio_mask = self._csu_biologic_mask(sub)
        sub = sub.loc[bio_mask]
        if sub.empty:
            return days_out, offlabel

        # Dupixent off-label tag: rows matching dupixent brand OR generic OR
        # NDC prefix. Use parallel boolean test instead of re-running the
        # full biologic mask so we surface ONLY Dupixent (Xolair has on-label
        # CSU approval; flagging Xolair as off-label would be wrong).
        dupixent_mask = pd.Series(False, index=sub.index)
        if "Brand_Name" in sub.columns:
            b = sub["Brand_Name"].astype(str).str.upper()
            dupixent_mask = dupixent_mask | b.str.contains("DUPIXENT", na=False)
        if "Generic_Name" in sub.columns:
            g = sub["Generic_Name"].astype(str).str.lower()
            dupixent_mask = dupixent_mask | g.str.contains("dupilumab", na=False)
        if "code" in sub.columns:
            c = sub["code"].astype(str).str.upper()
            # Dupixent NDC prefix per CSU_BIOLOGIC_NDC_PREFIXES is "00024" / "0024"
            for pref in ("00024", "0024"):
                dupixent_mask = dupixent_mask | c.str.startswith(pref)
            # Dupixent HCPCS code (J0517 per CSU_BIOLOGIC_HCPCS — spec lists it
            # as Dupixent even though J0517 is canonically eculizumab). A
            # code-only row with c == "J0517" matches _csu_biologic_mask but
            # without this clause would NOT be flagged off-label, so the HCP
            # would land in onlabel and receive a Rogers category while
            # dupixent_offlabel stays False.
            dupixent_mask = dupixent_mask | (c == "J0517")

        # First on-label fill per NPI (Xolair-equivalent only).
        onlabel = sub.loc[~dupixent_mask].copy()
        onlabel["medication_date"] = pd.to_datetime(onlabel["medication_date"], errors="coerce")
        onlabel = onlabel.dropna(subset=["medication_date"])
        if not onlabel.empty:
            onlabel["npi"] = onlabel["npi"].astype(str).str.strip()
            first_by_npi = onlabel.groupby("npi")["medication_date"].min()
            for npi_val, first_dt in first_by_npi.items():
                npi_key = str(npi_val)
                if npi_key not in days_out:
                    continue
                delta_days = int((first_dt - brand_launch).days)
                # Negative deltas (fill BEFORE launch — data error or
                # off-label-prior-approval) clamp to 0 so they still get
                # an `innovator` rank rather than skewing the curve.
                days_out[npi_key] = max(delta_days, 0)

        # Dupixent-offlabel flag — any Dupixent fill flags the HCP.
        if dupixent_mask.any():
            dup_npis = sub.loc[dupixent_mask, "npi"].astype(str).str.strip().unique()
            for npi_val in dup_npis:
                if npi_val in offlabel:
                    offlabel[npi_val] = True

        return days_out, offlabel

    def _build_hcp_profiles(self, kept_patids: set[int]) -> list[dict[str, Any]]:
        # Collect obfuscated NPIs from med + proc for kept patients.
        npi_rx: dict[str, int] = {}
        npi_pat: dict[str, set[int]] = {}

        def _accumulate(df: pd.DataFrame, date_col: str) -> None:
            if "npi" not in df.columns:
                return
            sub = df[df["patid"].isin(kept_patids)]
            for _, r in sub.iterrows():
                npi_val = r.get("npi")
                if pd.isna(npi_val):
                    continue
                npi_str = str(npi_val).strip()
                if not npi_str or npi_str == "nan":
                    continue
                npi_rx[npi_str] = npi_rx.get(npi_str, 0) + 1
                npi_pat.setdefault(npi_str, set()).add(int(r["patid"]))

        _accumulate(self.med, "medication_date")
        _accumulate(self.proc, "proc_date")

        profiles: list[dict[str, Any]] = []
        if not npi_rx:
            return profiles

        # Issue #155 §1: Rogers Diffusion of Innovations time-to-adoption.
        # Replaces the legacy volume-quartile classification (which conflated
        # prescribing volume with adoption timing — a high-volume HCP who
        # started late is `late_majority`, not `innovator`).
        #
        # For CSU, the on-label biologic is Xolair (launched 2014-03-21).
        # Dupixent has NO CSU approval as of 2026-05-12 — its CSU fills are
        # OFF-LABEL and excluded from the diffusion curve (flagged separately
        # via `dupixent_offlabel=True`).
        xolair_launch = pd.Timestamp(rwdc.BRAND_LAUNCH_DATES["xolair"]["csu"])
        hcp_days_to_first_fill, hcp_dupixent_offlabel = self._compute_npi_first_fill(
            kept_patids, npi_rx, xolair_launch
        )
        adoption_by_npi = rwdc.classify_rogers_adoption(hcp_days_to_first_fill)

        for seq, obf in enumerate(sorted(npi_rx.keys())):
            rx = npi_rx[obf]
            pv = len(npi_pat[obf])
            adoption = adoption_by_npi.get(obf, rwdc.ROGERS_NON_ADOPTER)
            dupixent_offlabel = hcp_dupixent_offlabel.get(obf, False)
            practice = "Hospital" if pv > 100 else "Group" if pv >= 50 else "Solo"

            taxonomy = self._provider_by_npi.get(obf, "")
            # Issue #154 §7.7 / §3: full-taxonomy-code matching for specialty
            # bucketing (replaces the legacy 4-char prefix). Subspecialty
            # detail is carried separately in `sub_specialty` once NPPES
            # enrichment fires (real-NPI cohorts only).
            if rwdc.taxonomy_in(taxonomy, rwdc.NUCC_ALLERGY_IMMUNOLOGY_CODES):
                specialty = "Allergy/Immunology"
            elif rwdc.taxonomy_in(taxonomy, rwdc.NUCC_DERMATOLOGY_CODES):
                specialty = "Dermatology"
            elif taxonomy:
                specialty = "Other"
            else:
                specialty = "Other"

            # Issue #154 §3: optional NPPES enrichment. Only fires when the
            # cohort carries real (non-obfuscated) NPIs AND a cache loader
            # is registered; for the synthetic / obfuscated Optum extract
            # this is a no-op (lookup_npi returns None) and the eight
            # currently-None fields stay None — that's correct behavior
            # because there is no real provider to look up. When a
            # real-NPI cohort lands, register the loader before this
            # converter runs and these fields auto-populate.
            generated_npi = rwdc.generate_luhn_npi(obf)
            nppes_rec = rwdc.lookup_npi(generated_npi, use_api_fallback=False)
            sub_specialty: str | None = None
            practice_type_resolved = practice
            practice_size_resolved: str | None = None
            geographic_region: str | None = None
            state_val: str | None = None
            city_val: str | None = None
            zip_code_val: str | None = None
            years_experience: int | None = None
            affiliation_primary: str | None = None
            first_name: str | None = None
            last_name: str | None = None
            if nppes_rec is not None:
                primary = nppes_rec.primary_taxonomy
                if primary is not None and primary.desc:
                    sub_specialty = primary.desc
                if nppes_rec.practice_address is not None:
                    addr = nppes_rec.practice_address
                    state_val = addr.state
                    city_val = addr.city
                    zip_code_val = addr.postal_code
                    geographic_region = rwdc.map_zipcode_to_region(zip_code_val)
                years_experience = nppes_rec.years_since_enumeration()
                affiliation_primary = nppes_rec.parent_organization_legal_name
                first_name = nppes_rec.first_name
                last_name = nppes_rec.last_name
                # Org-level providers (entity_type=2) → "Group" / "Hospital"
                # already covered by `practice` heuristic; the `sole_proprietor`
                # flag refines individual providers down to "Solo".
                if nppes_rec.sole_proprietor is True and practice == "Group":
                    practice_type_resolved = "Solo"
                # practice_size: bucket via sole-proprietor + entity flag
                if nppes_rec.sole_proprietor is True:
                    practice_size_resolved = "Solo"
                elif nppes_rec.entity_type == "2":
                    practice_size_resolved = "Group"

            profiles.append(
                {
                    "hcp_id": f"HCP_{seq:06d}",
                    "npi": generated_npi,
                    "first_name": first_name,
                    "last_name": last_name,
                    "specialty": specialty,
                    "sub_specialty": sub_specialty,
                    "practice_type": practice_type_resolved,
                    "practice_size": practice_size_resolved,
                    "geographic_region": geographic_region,
                    "state": state_val,
                    "city": city_val,
                    "zip_code": zip_code_val,
                    "priority_tier": None,
                    "decile": None,
                    "total_patient_volume": pv,
                    "target_patient_volume": None,
                    "prescribing_volume": rx,
                    "years_experience": years_experience,
                    "affiliation_primary": affiliation_primary,
                    "affiliation_secondary": None,
                    "digital_engagement_score": None,
                    "preferred_channel": None,
                    "last_interaction_date": None,
                    "interaction_frequency": None,
                    "influence_network_size": None,
                    "peer_influence_score": None,
                    "adoption_category": adoption,
                    "dupixent_offlabel": dupixent_offlabel,
                    "coverage_status": None,
                    "territory_id": None,
                    "sales_rep_id": None,
                    "created_at": self.now_iso,
                    "updated_at": self.now_iso,
                }
            )
        return profiles

    # ------------------------------------------------------------------ #
    # Data dictionary                                                     #
    # ------------------------------------------------------------------ #

    def _build_data_dictionary(self, cohort: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = [
            {
                "feature": "age_at_index",
                "type": "float",
                "source_table": "demographics",
                "lookback_window": "at index",
                "notes": "Spec §7.1; Optum provides integer age (no DOB).",
            },
            {
                "feature": "gender",
                "type": "enum{M,F,U}",
                "source_table": "demographics",
                "lookback_window": "at index",
                "notes": "§7.1",
            },
            {
                "feature": "geographic_region",
                "type": "enum{NE,S,MW,W}",
                "source_table": "demographics.zipcode_5",
                "lookback_window": "at index",
                "notes": "§7.1 — 3-digit ZIP→Census region",
            },
            {
                "feature": "urban_rural_code",
                "type": "enum{urban,suburban,rural}",
                "source_table": "demographics.zipcode_5",
                "lookback_window": "at index",
                "notes": "§7.1 — minimal zip3 crosswalk (approximation).",
            },
            {
                "feature": "dx_l50_*_count",
                "type": "int",
                "source_table": "inpatientdata.diag1-5 + demographics.diagcode",
                "lookback_window": "[index-180, index-1]",
                "notes": "§7.2",
            },
            {
                "feature": "has_<comorbidity>",
                "type": "int{0,1}",
                "source_table": "inpatientdata.diag1-5 + demographics.diagcode",
                "lookback_window": "[index-180, index-1]",
                "notes": "§7.3 — per comorbidity in §6.3",
            },
            {
                "feature": "office_visits_*",
                "type": "int",
                "source_table": "procedure (E&M codes 99201-99215)",
                "lookback_window": "[index-180, index-1]",
                "notes": "§7.4",
            },
            {
                "feature": "<drug_class>_ever_filled",
                "type": "int{0,1}",
                "source_table": "medication.Generic_Name",
                "lookback_window": "[index-180, index-1]",
                "notes": "§7.5 — NON-TARGET drugs only; biologic fills EXCLUDED to prevent target leakage",
            },
            {
                "feature": "<lab>_tested",
                "type": "int{0,1}",
                "source_table": "lab.loinc_cd",
                "lookback_window": "[index-180, index-1]",
                "notes": "§7.6",
            },
            {
                "feature": "primary_specialist_type",
                "type": "str (taxonomy1)",
                "source_table": "provider.taxonomy1 via procedure.npi",
                "lookback_window": "[index-180, index-1]",
                "notes": "§7.7 — full NUCC taxonomy code (issue #154 sharpens 4-char prefix → exact match)",
            },
            {
                "feature": "saw_allergist_flag",
                "type": "int{0,1}",
                "source_table": "provider.taxonomy1 via procedure.npi",
                "lookback_window": "[index-180, index-1]",
                "notes": "§7.7 — exact-match against NUCC_ALLERGY_IMMUNOLOGY_CODES (issue #154)",
            },
            {
                "feature": "saw_dermatologist_flag",
                "type": "int{0,1}",
                "source_table": "provider.taxonomy1 via procedure.npi",
                "lookback_window": "[index-180, index-1]",
                "notes": "§7.7 — exact-match against NUCC_DERMATOLOGY_CODES (issue #154)",
            },
            {
                "feature": "specialist_concentration",
                "type": "float (HHI)",
                "source_table": "provider.taxonomy1 via procedure.npi",
                "lookback_window": "[index-180, index-1]",
                "notes": "§7.7 — Herfindahl over full taxonomy codes (issue #154)",
            },
            # Issue #155 §1 / §2 / §3
            {
                "feature": "adoption_category",
                "type": "enum{innovator,early_adopter,early_majority,late_majority,laggard,non_adopter}",
                "source_table": "medication (Xolair on-label fills) via NPI",
                "lookback_window": "all CSU biologic fills in scope",
                "notes": (
                    "Issue #155 §1 — Rogers Diffusion of Innovations TIME-to-"
                    "first-fill (anchor: Xolair-CSU launch 2014-03-21). "
                    "non_adopter for HCPs with no on-label fill. Dupixent "
                    "fills excluded from curve (off-label for CSU; see "
                    "dupixent_offlabel flag). Replaces legacy volume "
                    "quartile classification."
                ),
            },
            {
                "feature": "dupixent_offlabel",
                "type": "bool",
                "source_table": "medication (Dupixent fills) via NPI",
                "lookback_window": "all CSU biologic fills in scope",
                "notes": (
                    "Issue #155 §1 — TRUE if HCP has any Dupixent fill in "
                    "the CSU cohort. Dupixent is NOT FDA-approved for CSU "
                    "as of 2026-05-12; flagged for downstream cross-"
                    "indication adoption analysis."
                ),
            },
            {
                "feature": "journey_stage",
                "type": (
                    "enum{aware,considering,first_fill,adherent,"
                    "discontinued,maintained} (+ legacy diagnosis/"
                    "initial_treatment/treatment_optimization/maintenance/"
                    "treatment_switch)"
                ),
                "source_table": "derived from cohort + targets + saw_specialist",
                "lookback_window": "post-index (knowable_at=post_index in manifest)",
                "notes": (
                    "Issue #155 §2 — granular PR #152 engagement-funnel "
                    "value. `prescribed` NOT emitted (Optum is dispensed-"
                    "only). See migration 035."
                ),
            },
            {
                "feature": "source_timestamp",
                "type": "ISO 8601 UTC timestamp",
                "source_table": "extract_ym (vendor drop month)",
                "lookback_window": "n/a (drop-level metadata)",
                "notes": (
                    "Issue #155 §3 — LAST_DAY of extract_ym at 23:59:59 "
                    "UTC. Worst-case lag estimate (never understates). "
                    "Off by up to 30 days vs the true claim-emission "
                    "timestamp. NULL if --extract-ym is omitted and not "
                    "inferable from --input dir name."
                ),
            },
            {
                "feature": "data_lag_hours",
                "type": "int (may be negative)",
                "source_table": "derived from extract_ym + parquet mtime",
                "lookback_window": "n/a",
                "notes": (
                    "Issue #155 §3 — floor((ingestion_timestamp - "
                    "source_timestamp) / 3600). Negative for rare back-"
                    "dated drops; downstream consumers should surface "
                    "the anomaly."
                ),
            },
        ]

        # Target
        if cohort == "initiation":
            entries.append(
                {
                    "feature": "initiated_biologic_180d",
                    "type": "int{0,1}",
                    "source_table": "medication (Xolair/Dupixent NDC+HCPCS+brand)",
                    "lookback_window": "[index, index+180]",
                    "notes": "§8.1 — TARGET; computed from CSU biologic fills in prediction window",
                }
            )
        elif cohort == "discontinuation":
            entries.append(
                {
                    "feature": "discontinued_180d",
                    "type": "int{0,1}",
                    "source_table": "medication biologic fills",
                    "lookback_window": "[init_date, init_date+180]",
                    "notes": "§8.2 — TARGET; gap > 90d between fill_end and next fill",
                }
            )
        elif cohort == "persistence":
            entries.append(
                {
                    "feature": "persistent_at_180d",
                    "type": "int{0,1}",
                    "source_table": "medication biologic fills",
                    "lookback_window": "[init_date, init_date+180]",
                    "notes": "§8.3 — TARGET; active fill (days_supply-based) at day 180",
                }
            )
        return entries

    # ------------------------------------------------------------------ #
    # Pilot audit (§11)                                                   #
    # ------------------------------------------------------------------ #

    def _run_pilot_audit(self, cohort: str, journeys: list[dict[str, Any]]) -> None:
        """Run a fast leakage audit on the converter output.

        Per spec §11: zero CRITICAL findings, <3 HIGH before running the full
        tier-0 pipeline. This is a pre-flight sanity check only — the
        authoritative detection runs in data_preparer.leakage_detector during
        Tier-0. Uses the synchronous pure-function helpers exported from
        leakage_detector (taking a plain DataFrame + target column), not the
        async agent-node ``detect_leakage(state)``.
        """
        logger.info("  Running pilot audit on cohort %s (%d journeys)", cohort, len(journeys))
        try:
            from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import (
                check_perfect_class_separation,
                check_single_feature_auc,
            )
        except Exception as exc:
            logger.warning("  Pilot audit skipped — could not import leakage checks: %s", exc)
            return

        df = pd.DataFrame(journeys)
        target_col = {
            "initiation": "initiated_biologic_180d",
            "discontinuation": "discontinued_180d",
            "persistence": "persistent_at_180d",
        }[cohort]
        if target_col not in df.columns:
            logger.warning("  Pilot audit skipped — target column %s missing", target_col)
            return

        numeric_features = [
            c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])
        ]

        try:
            findings: list[Any] = []
            findings.extend(check_single_feature_auc(df, target_col, numeric_features))
            findings.extend(check_perfect_class_separation(df, target_col, numeric_features))
        except Exception as exc:
            logger.warning("  Pilot audit run failed: %s", exc)
            return

        # Count severity tiers (LeakageFinding.severity is an Enum)
        sev_counts: dict[str, int] = {}
        for f in findings:
            sev = getattr(getattr(f, "severity", None), "value", "") or ""
            sev_counts[sev] = sev_counts.get(sev, 0) + 1
        logger.info("  Pilot audit findings by severity: %s", sev_counts)

        # §11 gate: zero CRITICAL, fewer than 3 HIGH
        if sev_counts.get("critical", 0) > 0 or sev_counts.get("high", 0) >= 3:
            logger.warning(
                "  Pilot audit GATE FAILED for cohort %s: %s — run data_preparer.leakage_detector for details",
                cohort,
                sev_counts,
            )


# --------------------------------------------------------------------------- #
# Parquet-safe normalisation                                                  #
# --------------------------------------------------------------------------- #


def _drop_forbidden_columns(
    records: list[dict[str, Any]], forbidden: list[str]
) -> list[dict[str, Any]]:
    """Drop ``forbidden`` keys from each record before writing to disk.

    Item C of the engineering-actionable arc (2026-05-08). Mirrors the
    same-named helper in ``scripts/convert_csu_rwd.py`` so both
    converters share the boundary-filter contract. Returns a NEW list
    with NEW dicts; the input is not mutated. Targets (e.g.
    ``treatment_initiated``, ``initiated_biologic_180d``) are NOT in
    ``forbidden`` because they are the supervised signal — see
    ``OPTUM_FORBIDDEN_NON_TARGET`` in
    ``src/data/manifests/optum_feature_manifest.py``.
    """
    forbidden_set = set(forbidden)
    return [{k: v for k, v in r.items() if k not in forbidden_set} for r in records]


def _normalise_events_for_parquet(events: list[dict[str, Any]]) -> None:
    """JSON-encode nested fields that pyarrow can't type-infer from empty dicts."""
    import json as _json

    for e in events:
        lv = e.get("lab_values")
        e["lab_values"] = _json.dumps(lv) if lv else "{}"
        for k in ("icd_codes", "cpt_codes", "loinc_codes"):
            v = e.get(k)
            if not isinstance(v, list):
                e[k] = []


def _normalise_journeys_for_parquet(journeys: list[dict[str, Any]]) -> None:
    for j in journeys:
        for k in ("secondary_diagnosis_codes", "data_sources_matched", "comorbidities"):
            v = j.get(k)
            if not isinstance(v, list):
                j[k] = []


def _normalise_hcps_for_parquet(hcps: list[dict[str, Any]]) -> None:
    # HCP dicts are flat — nothing to normalise, but keep a hook for symmetry.
    return


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Optum parquet RWD to E2I canonical cohort outputs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohort", choices=ALLOWED_COHORTS, default="all")
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Limit to first N demographics.patid (for pilot/testing)",
    )
    parser.add_argument(
        "--pilot-audit",
        action="store_true",
        help="After conversion, run leakage_detector on output (spec §11 gate)",
    )
    parser.add_argument(
        "--enrollment-regime",
        choices=sorted(ENROLLMENT_REGIMES.keys()),
        default=DEFAULT_ENROLLMENT_REGIME,
        help=(
            "Enrollment-window regime (plan v3 §3 Tier 1A). "
            "production=360/180 (default, current behavior); "
            "research=180/90 (larger eligible cohort, requires domain-expert "
            "sign-off before downstream use)."
        ),
    )
    parser.add_argument(
        "--extract-ym",
        type=str,
        default=None,
        help=(
            "Optum vendor drop month as YYYYMM (e.g. 202604 for April 2026). "
            "Drives patient_journeys.source_timestamp (LAST_DAY of the month "
            "at 23:59:59 UTC — worst-case lag estimate; never understates). "
            "If omitted, inferred from a YYYYMM substring in --input dir name."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.input.exists():
        logger.error("Input directory not found: %s", args.input)
        return 1

    cohorts: tuple[str, ...]
    if args.cohort == "all":
        cohorts = ("initiation", "discontinuation", "persistence")
    else:
        cohorts = (args.cohort,)

    converter = OptumDataConverter(
        parquet_dir=args.input,
        output_dir=args.output,
        cohorts=cohorts,
        max_patients=args.max_patients,
        pilot_audit=args.pilot_audit,
        enrollment_regime=args.enrollment_regime,
        extract_ym=args.extract_ym,
    )

    if args.dry_run:
        converter._read_parquets()
        converter._clean()
        logger.info("Dry run — exiting after cleaning step")
        return 0

    counts = converter.convert_all()
    logger.info("=" * 60)
    logger.info("Optum conversion complete")
    for c, v in counts.items():
        logger.info("  %s: %s", c, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
