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
from collections import defaultdict
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

# Issue #156 item 1: priority_tier rolling-12-month window for HCP TRx
# aggregation per ZIP3 decile. 12 months = 365 days. Aligns with the
# pharma-industry standard "trailing twelve-month TRx" used for decile
# segmentation in commercial analytics.
PRIORITY_TIER_TRX_WINDOW_DAYS = 365

# Issue #156 item 1: decile → priority_tier mapping (1 = highest priority).
# Per issue body:
#   decile 10        → tier 1
#   decile 8-9       → tier 2
#   decile 4-7       → tier 3
#   decile 2-3       → tier 4
#   decile 1         → tier 5
# HCPs with TRx=0 in the window also map to tier 5 (kept in the
# scoreable pool, not excluded).
PRIORITY_TIER_DECILE_MAP: dict[int, int] = {
    10: 1,
    9: 2,
    8: 2,
    7: 3,
    6: 3,
    5: 3,
    4: 3,
    3: 4,
    2: 4,
    1: 5,
}
PRIORITY_TIER_DEFAULT = 5

# Issue #156 item 2: peer_influence_score is eigenvector_centrality scaled
# from the natural [0, 1] range to fit the DECIMAL(3,2) DB column, which
# admits values in [0.00, 9.99]. We scale by `PEER_INFLUENCE_SCALE` so that
# a top-influencer (centrality ≈ 1.0) lands at 9.99 (then clamped). The
# scale factor is exposed for downstream re-derivation.
PEER_INFLUENCE_SCALE = 9.99

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

# --------------------------------------------------------------------------- #
# Issue #157 PR C — treatment_response proxy rule constants                   #
# --------------------------------------------------------------------------- #
# CSU has no validated lab biomarker for clinical control (UAS7/UCT/CU-Q2oL
# are patient-reported and absent from Optum claims). We derive a 5-value
# response proxy from claim-pattern signals per the issue #157 spec.
#
# Pre-conditions for non-NULL response (else treatment_response = NULL):
#   * Treatment initiated: >=1 fill of a CSU biologic (Xolair or Dupixent)
#   * Persistence: >=TREATMENT_RESPONSE_MIN_COVERAGE_DAYS of biologic coverage
#     by days_sup (matches BIOLOGIC_PERSISTENCE_GAP_DAYS = 60d).
#   * Follow-up: >=TREATMENT_RESPONSE_MIN_FOLLOWUP_DAYS post-initiation
#     observation window.
#
# Classification order (first match wins):
#   1. discontinued — Gap > BIOLOGIC_DISCONT_GAP_DAYS between fill_end and
#      next fill, within TREATMENT_RESPONSE_WINDOW_DAYS of initiation.
#   2. refractory   — Switch to the OTHER biologic (different NDC prefix)
#      within TREATMENT_RESPONSE_WINDOW_DAYS, OR addition of an
#      immunosuppressant (NON_TARGET_DRUG_CLASSES["immunosupp"]) during
#      the post-init coverage window.
#   3. inadequate   — Persistence met but >=1 rescue oral-steroid burst
#      (prednisone/methylprednisolone, >=5 days, post-init coverage window)
#      OR >=1 urticaria/angioedema ED visit (POS=23, dx in L50.x or T78.3).
#   4. controlled   — Persistence met, no rescue events, no ED visit.
#
# The fifth allowed value, `uncontrolled`, is reserved for non-Optum cohorts
# (synthetic generator + EHR-anchored cohorts where UAS7 is available);
# the Optum proxy emits `inadequate` for that semantic position because the
# distinction between "uncontrolled" and "inadequate response to biologic"
# cannot be made from claims alone.
TREATMENT_RESPONSE_MIN_COVERAGE_DAYS = BIOLOGIC_PERSISTENCE_GAP_DAYS  # 60d
TREATMENT_RESPONSE_MIN_FOLLOWUP_DAYS = 90
TREATMENT_RESPONSE_WINDOW_DAYS = 180

# Rescue oral-steroid burst (CDC/MMWR + Maurer 2018 CSU literature):
#   * Drug: prednisone OR methylprednisolone (only — dexamethasone is
#     typically inhaled/IV for CSU and out of scope for the burst signal).
#   * Daily dose: not directly parsed from Optum (NDC strength is per-tablet,
#     not per-day) — we approximate via days_sup >= 5 and rely on the
#     short-course pattern.
#   * Duration: days_sup >= 5.
RESCUE_STEROID_GENERICS: tuple[str, ...] = ("prednisone", "methylprednisolone")
RESCUE_STEROID_MIN_DAYS_SUP = 5

# Urticaria/angioedema ED visit signal.
#   * POS 23 = Emergency Room — Hospital (CMS POS code set).
#   * Diagnosis: any of L50.x (CSU urticaria) or T78.3 (angioedema).
ED_POS_CODE = "23"
ED_CSU_DX_PREFIXES: tuple[str, ...] = ("L50", "T783")

# Treatment-response vocabulary (mirrors the CHECK constraint in
# migration 037_treatment_response_column.sql).
TREATMENT_RESPONSE_VOCAB: frozenset[str] = frozenset(
    {"controlled", "inadequate", "uncontrolled", "refractory", "discontinued"}
)

# outcome_indicator mapping per issue #157 spec.
#   controlled                            → improved
#   inadequate, refractory                → worsened
#   discontinued                          → worsened (no subsequent fill)
#                                        OR stable (subsequent fill exists,
#                                                   handled per-row at emit
#                                                   time)
TREATMENT_RESPONSE_TO_OUTCOME: dict[str, str] = {
    "controlled": "improved",
    "inadequate": "worsened",
    "uncontrolled": "worsened",
    "refractory": "worsened",
    "discontinued": "worsened",  # overridden to 'stable' if subsequent fill
}

# Drug-class-aware gap thresholds for discontinuation/persistence detection.
# Keys correspond to drug class labels. CSU biologics (Xolair / Dupixent) use
# the "biologic" entry, which preserves the historical 90/60 day defaults
# bit-for-bit (see backward-compat tests in test_convert_optum_rwd.py).
#
# Resolution: `_resolve_gap_thresholds(drug_class)` maps a class label to its
# (discontinuation, persistence) days. Class labels for non-target therapies
# come from NON_TARGET_DRUG_CLASSES below; biologics are tagged "biologic" by
# `_csu_biologic_mask`. Unknown classes fall back to "default".
#
# Documented in docs/OPTUM_CONVERSION.md.
GAP_THRESHOLDS: dict[str, dict[str, int]] = {
    "biologic": {
        "discontinuation": BIOLOGIC_DISCONT_GAP_DAYS,
        "persistence": BIOLOGIC_PERSISTENCE_GAP_DAYS,
    },
    "oral_chronic": {"discontinuation": 60, "persistence": 30},
    "specialty_injectable": {"discontinuation": 90, "persistence": 60},
    "default": {"discontinuation": 60, "persistence": 30},
}


def _resolve_gap_thresholds(drug_class: str) -> tuple[int, int]:
    """Resolve (discontinuation_gap_days, persistence_gap_days) for a class.

    Returns the "default" entry for any unknown class label.
    """
    entry = GAP_THRESHOLDS.get(drug_class, GAP_THRESHOLDS["default"])
    return entry["discontinuation"], entry["persistence"]


# --------------------------------------------------------------------------- #
# Data Quality Score weights — issue #156 item 4                              #
# --------------------------------------------------------------------------- #
# Weighted 4-component DQS replaces the legacy uniform feature-completeness
# fraction. Per-claim DQS is computed across medical/medication/inpatient
# claims in the lookback window; the patient DQS is the mean over all claims.
# Weights sum to 1.0.
DQS_WEIGHT_DX = 0.40
DQS_WEIGHT_PROC = 0.25
DQS_WEIGHT_COST = 0.20
DQS_WEIGHT_ENROLL = 0.15

# Cost fields loaded from the Optum parquet extracts (issue #156 item 4).
# std_cost is the PRIMARY, most-reliable standardized cost; the remainder
# are patient/payer breakouts.
DQS_COST_FIELDS_PRIMARY = ("std_cost",)
DQS_COST_FIELDS_FALLBACK = ("charge", "copay", "coins", "deduct")

# Soft data-quality filter for downstream model-training (issue #156 item 5).
# Patients with a DQS below this threshold are flagged in attrition_report
# under "soft-filtered (low DQS)" rather than hard-dropped from the cohort.
DEFAULT_MIN_DATA_QUALITY_SCORE = 0.50


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

# Analyte -> LOINC codes. Codes were corrected on 2026-06-03 after a forensic
# trace of the Optum lab extract (`tst_desc` cross-check) found three analytes
# pointing at the WRONG test. Each entry now lists the canonical LOINC plus the
# variant codes actually observed in the Optum drop, so the feature both reads
# correctly and populates on real data. See
# docs/results/tier0_cohort_comparison_optum_vs_synthetic_20260603.md
# (Root-cause forensics) and TestCsuLabsLoincMapping.
CSU_LABS_LOINC: dict[str, tuple[str, ...]] = {
    "ige_total": ("19113-0", "2683-2"),  # 'IMMUNOGLOBULIN E, TOTAL' (verified correct)
    # 711-2 / 26444-0 = absolute eosinophil count.
    # (was 6206-7 = Peanut IgE 'F013-IGE PEANUT' — wrong analyte.)
    "eosinophil": ("711-2", "26444-0"),
    "crp": ("1988-5",),  # 'C-REACTIVE PROTEIN' (verified correct)
    # 8099-8 canonical thyroid-peroxidase Ab; 8099-4 / 56477-3 = Optum-extract variants.
    # (was 3051-0 / 3053-6 = Free / Total T3 — wrong analyte.)
    "tpo_ab": ("8099-8", "8099-4", "56477-3"),
    "free_t4": ("3024-7",),  # 'T4, FREE' (verified correct)
    "tsh": ("3016-3",),  # 'TSH' (verified correct)
    # 42254-3 'ANA SCREEN, IFA' / 5048-4 'ANA TITER' / 8061-4 'ANA DIRECT'.
    # (was 14741-9 — zero rows in the extract and not an antinuclear-antibody code.)
    "ana": ("42254-3", "5048-4", "8061-4"),
    # 58410-2 = CBC panel; 57021-8 = 'CBC WITH DIFF'.
    # (was 26453-1 = RBC — wrong analyte.)
    "cbc": ("58410-2", "57021-8"),
}

# --------------------------------------------------------------------------- #
# Comorbidity scoring — Quan (2005) ICD-10 mappings for Charlson + Elixhauser  #
# Reference: Quan H et al. Med Care 2005;43(11):1130-1139.                     #
# Charlson weights: original Charlson 1, 2, 3, 6 (no Quan recalibration —      #
# Quan 2011 weights are an alternative; we expose the classical weights for    #
# parity with the most-cited literature).                                      #
# Elixhauser weights: van Walraven et al. Med Care 2009;47(6):626-633.         #
# Hierarchies (e.g. metastatic supersedes any-malignancy; severe liver         #
# supersedes mild liver) are applied at scoring time by _charlson_quan.        #
#                                                                              #
# Codes are stored UPPER-CASE and de-dotted (Optum convention). All prefixes   #
# here are de-dotted ICD-10-CM.                                                #
# --------------------------------------------------------------------------- #

# Default comorbidity method. "quan" enables _charlson_quan/_elixhauser_quan;
# "approx" preserves the legacy approximation used pre-issue-#156 for a clean
# parity test in CI. Override via OptumDataConverter(comorbidity_method=...).
COMORBIDITY_METHOD_DEFAULT = "quan"
COMORBIDITY_METHODS_ALLOWED = ("quan", "approx")

QUAN_CHARLSON: dict[str, tuple[str, ...]] = {
    "myocardial_infarction": ("I21", "I22", "I252"),
    "congestive_heart_failure": (
        "I43",
        "I50",
        "I099",
        "I110",
        "I130",
        "I132",
        "I255",
        "I420",
        "I425",
        "I426",
        "I427",
        "I428",
        "I429",
        "P290",
    ),
    "peripheral_vascular_disease": (
        "I70",
        "I71",
        "I731",
        "I738",
        "I739",
        "I771",
        "I790",
        "I792",
        "K551",
        "K558",
        "K559",
        "Z958",
        "Z959",
    ),
    "cerebrovascular_disease": (
        "G45",
        "G46",
        "I60",
        "I61",
        "I62",
        "I63",
        "I64",
        "I65",
        "I66",
        "I67",
        "I68",
        "I69",
        "H340",
    ),
    "dementia": ("F00", "F01", "F02", "F03", "G30", "F051", "G311"),
    "chronic_pulmonary_disease": (
        "J40",
        "J41",
        "J42",
        "J43",
        "J44",
        "J45",
        "J46",
        "J47",
        "J60",
        "J61",
        "J62",
        "J63",
        "J64",
        "J65",
        "J66",
        "J67",
        "J684",
        "J701",
        "J703",
        "I278",
        "I279",
    ),
    "rheumatic_disease": (
        "M05",
        "M06",
        "M32",
        "M33",
        "M34",
        "M315",
        "M351",
        "M353",
        "M360",
    ),
    "peptic_ulcer_disease": ("K25", "K26", "K27", "K28"),
    "mild_liver_disease": (
        "B18",
        "K73",
        "K74",
        "K700",
        "K701",
        "K702",
        "K703",
        "K709",
        "K717",
        "K713",
        "K714",
        "K715",
        "K760",
        "K762",
        "K763",
        "K764",
        "K768",
        "K769",
        "Z944",
    ),
    "severe_liver_disease": (
        "I850",
        "I859",
        "I864",
        "I982",
        "K704",
        "K711",
        "K721",
        "K729",
        "K765",
        "K766",
        "K767",
    ),
    "diabetes_no_complications": (
        "E100",
        "E101",
        "E106",
        "E108",
        "E109",
        "E110",
        "E111",
        "E116",
        "E118",
        "E119",
        "E120",
        "E121",
        "E126",
        "E128",
        "E129",
        "E130",
        "E131",
        "E136",
        "E138",
        "E139",
        "E140",
        "E141",
        "E146",
        "E148",
        "E149",
    ),
    "diabetes_complications": (
        "E102",
        "E103",
        "E104",
        "E105",
        "E107",
        "E112",
        "E113",
        "E114",
        "E115",
        "E117",
        "E122",
        "E123",
        "E124",
        "E125",
        "E127",
        "E132",
        "E133",
        "E134",
        "E135",
        "E137",
        "E142",
        "E143",
        "E144",
        "E145",
        "E147",
    ),
    "hemiplegia_paraplegia": (
        "G81",
        "G82",
        "G041",
        "G114",
        "G801",
        "G802",
        "G830",
        "G831",
        "G832",
        "G833",
        "G834",
        "G839",
    ),
    "renal_disease": (
        "N18",
        "N19",
        "N052",
        "N053",
        "N054",
        "N055",
        "N056",
        "N057",
        "N250",
        "I120",
        "I131",
        "N032",
        "N033",
        "N034",
        "N035",
        "N036",
        "N037",
        "Z490",
        "Z491",
        "Z492",
        "Z940",
        "Z992",
    ),
    "any_malignancy": (
        "C00",
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "C06",
        "C07",
        "C08",
        "C09",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
        "C30",
        "C31",
        "C32",
        "C33",
        "C34",
        "C37",
        "C38",
        "C39",
        "C40",
        "C41",
        "C43",
        "C45",
        "C46",
        "C47",
        "C48",
        "C49",
        "C50",
        "C51",
        "C52",
        "C53",
        "C54",
        "C55",
        "C56",
        "C57",
        "C58",
        "C60",
        "C61",
        "C62",
        "C63",
        "C64",
        "C65",
        "C66",
        "C67",
        "C68",
        "C69",
        "C70",
        "C71",
        "C72",
        "C73",
        "C74",
        "C75",
        "C76",
        "C81",
        "C82",
        "C83",
        "C84",
        "C85",
        "C88",
        "C90",
        "C91",
        "C92",
        "C93",
        "C94",
        "C95",
        "C96",
        "C97",
    ),
    "metastatic_solid_tumor": ("C77", "C78", "C79", "C80"),
    "aids_hiv": ("B20", "B21", "B22", "B24"),
}

QUAN_ELIXHAUSER: dict[str, tuple[str, ...]] = {
    "congestive_heart_failure": (
        "I43",
        "I50",
        "I099",
        "I110",
        "I130",
        "I132",
        "I255",
        "I420",
        "I425",
        "I426",
        "I427",
        "I428",
        "I429",
        "P290",
    ),
    "cardiac_arrhythmias": (
        "I441",
        "I442",
        "I443",
        "I456",
        "I459",
        "I47",
        "I48",
        "I49",
        "R000",
        "R001",
        "R008",
        "T821",
        "Z450",
        "Z950",
    ),
    "valvular_disease": (
        "A520",
        "I05",
        "I06",
        "I07",
        "I08",
        "I091",
        "I098",
        "I34",
        "I35",
        "I36",
        "I37",
        "I38",
        "I39",
        "Q230",
        "Q231",
        "Q232",
        "Q233",
        "Z952",
        "Z953",
        "Z954",
    ),
    # Quan 2005 Table 2: I27.8 and I27.9 are reclassified into
    # chronic_pulmonary_disease, so pulmonary_circulation_disorders enumerates
    # the I27 sub-codes explicitly (I270..I277) plus I26 and I28 leaves.
    "pulmonary_circulation_disorders": (
        "I26",
        "I270",
        "I271",
        "I272",
        "I273",
        "I274",
        "I275",
        "I276",
        "I277",
        "I280",
        "I288",
        "I289",
    ),
    "peripheral_vascular_disorders": (
        "I70",
        "I71",
        "I731",
        "I738",
        "I739",
        "I771",
        "I790",
        "I792",
        "K551",
        "K558",
        "K559",
        "Z958",
        "Z959",
    ),
    "hypertension_uncomplicated": ("I10",),
    "hypertension_complicated": ("I11", "I12", "I13", "I15"),
    "paralysis": (
        "G041",
        "G114",
        "G801",
        "G802",
        "G81",
        "G82",
        "G830",
        "G831",
        "G832",
        "G833",
        "G834",
        "G839",
    ),
    "other_neurological_disorders": (
        "G10",
        "G11",
        "G12",
        "G13",
        "G20",
        "G21",
        "G22",
        "G254",
        "G255",
        "G312",
        "G318",
        "G319",
        "G32",
        "G35",
        "G36",
        "G37",
        "G40",
        "G41",
        "G931",
        "G934",
        "R470",
        "R56",
    ),
    "chronic_pulmonary_disease": (
        "J40",
        "J41",
        "J42",
        "J43",
        "J44",
        "J45",
        "J46",
        "J47",
        "J60",
        "J61",
        "J62",
        "J63",
        "J64",
        "J65",
        "J66",
        "J67",
        "J684",
        "J701",
        "J703",
        "I278",
        "I279",
    ),
    # Quan 2005 Table 2 includes E10/E11/E12/E13/E14 prefixes split between
    # uncomplicated (terminal digits 0, 1, 9) and complicated (2-8) for the
    # Elixhauser diabetes categories. van Walraven weights for both diabetes
    # categories are 0, but the mapping is kept faithful to the published list.
    "diabetes_uncomplicated": (
        "E100",
        "E101",
        "E109",
        "E110",
        "E111",
        "E119",
        "E120",
        "E121",
        "E129",
        "E130",
        "E131",
        "E139",
        "E140",
        "E141",
        "E149",
    ),
    "diabetes_complicated": (
        "E102",
        "E103",
        "E104",
        "E105",
        "E106",
        "E107",
        "E108",
        "E112",
        "E113",
        "E114",
        "E115",
        "E116",
        "E117",
        "E118",
        "E122",
        "E123",
        "E124",
        "E125",
        "E126",
        "E127",
        "E128",
        "E132",
        "E133",
        "E134",
        "E135",
        "E136",
        "E137",
        "E138",
        "E142",
        "E143",
        "E144",
        "E145",
        "E146",
        "E147",
        "E148",
    ),
    "hypothyroidism": ("E00", "E01", "E02", "E03", "E890"),
    "renal_failure": (
        "I120",
        "I131",
        "N18",
        "N19",
        "N250",
        "Z490",
        "Z491",
        "Z492",
        "Z940",
        "Z992",
    ),
    "liver_disease": (
        "B18",
        "I85",
        "I864",
        "I982",
        "K70",
        "K711",
        "K713",
        "K714",
        "K715",
        "K717",
        "K72",
        "K73",
        "K74",
        "K760",
        "K762",
        "K763",
        "K764",
        "K765",
        "K766",
        "K767",
        "K768",
        "K769",
        "Z944",
    ),
    "peptic_ulcer_disease_excluding_bleeding": (
        "K257",
        "K259",
        "K267",
        "K269",
        "K277",
        "K279",
        "K287",
        "K289",
    ),
    "aids_hiv": ("B20", "B21", "B22", "B24"),
    "lymphoma": ("C81", "C82", "C83", "C84", "C85", "C88", "C96", "C900", "C902"),
    "metastatic_cancer": ("C77", "C78", "C79", "C80"),
    "solid_tumor_without_metastasis": (
        "C00",
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "C06",
        "C07",
        "C08",
        "C09",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
        "C30",
        "C31",
        "C32",
        "C33",
        "C34",
        "C37",
        "C38",
        "C39",
        "C40",
        "C41",
        "C43",
        "C45",
        "C46",
        "C47",
        "C48",
        "C49",
        "C50",
        "C51",
        "C52",
        "C53",
        "C54",
        "C55",
        "C56",
        "C57",
        "C58",
        "C60",
        "C61",
        "C62",
        "C63",
        "C64",
        "C65",
        "C66",
        "C67",
        "C68",
        "C69",
        "C70",
        "C71",
        "C72",
        "C73",
        "C74",
        "C75",
        "C76",
        "C97",
    ),
    "rheumatoid_arthritis_collagen_vascular": (
        "L940",
        "L941",
        "L943",
        "M05",
        "M06",
        "M08",
        "M120",
        "M123",
        "M30",
        "M310",
        "M311",
        "M312",
        "M313",
        "M32",
        "M33",
        "M34",
        "M35",
        "M45",
        "M461",
        "M468",
        "M469",
    ),
    "coagulopathy": (
        "D65",
        "D66",
        "D67",
        "D68",
        "D691",
        "D693",
        "D694",
        "D695",
        "D696",
    ),
    "obesity": ("E66",),
    "weight_loss": ("E40", "E41", "E42", "E43", "E44", "E45", "E46", "R634", "R64"),
    "fluid_electrolyte_disorders": ("E222", "E86", "E87"),
    "blood_loss_anemia": ("D500",),
    "deficiency_anemia": ("D508", "D509", "D51", "D52", "D53"),
    "alcohol_abuse": (
        "F10",
        "E52",
        "G621",
        "I426",
        "K292",
        "K700",
        "K703",
        "K709",
        "T51",
        "Z502",
        "Z714",
        "Z721",
    ),
    "drug_abuse": ("F11", "F12", "F13", "F14", "F15", "F16", "F18", "F19", "Z715", "Z722"),
    "psychoses": ("F20", "F22", "F23", "F24", "F25", "F28", "F29", "F302", "F312", "F315"),
    "depression": ("F204", "F313", "F314", "F315", "F32", "F33", "F341", "F412", "F432"),
}

# van Walraven et al. (2009) integer point system for Elixhauser categories.
# Negative weights are protective in the published model.
VAN_WALRAVEN_WEIGHTS: dict[str, int] = {
    "congestive_heart_failure": 7,
    "cardiac_arrhythmias": 5,
    "valvular_disease": -1,
    "pulmonary_circulation_disorders": 4,
    "peripheral_vascular_disorders": 2,
    "hypertension_uncomplicated": 0,
    "hypertension_complicated": 0,
    "paralysis": 7,
    "other_neurological_disorders": 6,
    "chronic_pulmonary_disease": 3,
    "diabetes_uncomplicated": 0,
    "diabetes_complicated": 0,
    "hypothyroidism": 0,
    "renal_failure": 5,
    "liver_disease": 11,
    "peptic_ulcer_disease_excluding_bleeding": 0,
    "aids_hiv": 0,
    "lymphoma": 9,
    "metastatic_cancer": 12,
    "solid_tumor_without_metastasis": 4,
    "rheumatoid_arthritis_collagen_vascular": 0,
    "coagulopathy": 3,
    "obesity": -4,
    "weight_loss": 6,
    "fluid_electrolyte_disorders": 5,
    "blood_loss_anemia": -2,
    "deficiency_anemia": -2,
    "alcohol_abuse": 0,
    "drug_abuse": -7,
    "psychoses": 0,
    "depression": -3,
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
# Issue #169: reusable HCP-influence graph helpers (factored out of the      #
# converter so that the FalkorDB persistence script can build the EXACT     #
# same shared-patient clique graph PR #168 builds for in-memory scoring).   #
# --------------------------------------------------------------------------- #


def build_hcp_influence_graph(
    kept_patids: set[int],
    med: pd.DataFrame,
    proc: pd.DataFrame,
    idx_by_patid: dict[int, pd.Timestamp] | None = None,
    lookback_days: int = LOOKBACK_DAYS,
) -> Any:
    """Issue #169: build the per-cohort HCP-HCP influence graph.

    Pure helper factored out of ``OptumDataConverter._compute_influence_network``
    so that both the converter (in-memory scoring) and the FalkorDB
    persistence script (Cypher ingest) consume the SAME shared-patient
    clique graph. PR #168 fixed the leakage-safe temporal gate; this
    helper preserves that contract verbatim.

    For each patient in ``kept_patids``, collect the set of treating
    HCPs from ``med.npi ∪ proc.npi`` WITHIN THE PER-PATIENT LOOKBACK
    WINDOW ``(patient_index - lookback_days, patient_index]``. Each
    per-patient set forms a clique; the weighted undirected HCP-HCP
    graph aggregates edge counts over patients (edge weight = number of
    distinct patients seen by BOTH endpoints in the window).

    Args:
        kept_patids: cohort-membership patid filter.
        med: medication.parquet-shaped DataFrame; needs ``npi``, ``patid``,
            ``medication_date`` columns. Rows lacking ``npi`` are skipped.
        proc: procedure.parquet-shaped DataFrame; needs ``npi``, ``patid``,
            ``proc_date`` columns.
        idx_by_patid: per-patient index date. When ``None`` (test
            invocation only), the lookback gate is skipped and ALL
            med/proc rows for kept patients contribute — production
            callers always thread this map.
        lookback_days: days before index_date that bound the inclusion
            window. Defaults to ``LOOKBACK_DAYS`` (180d).

    Returns:
        A ``networkx.Graph`` with string-valued NPI nodes and integer
        ``weight`` edge property, or ``None`` if networkx isn't
        importable (the converter logs and falls back to empty
        dicts in that case). An empty graph (no edges) is still
        returned as a valid ``nx.Graph`` so callers can branch on
        ``graph is None`` for "networkx missing" vs ``len(graph) == 0``
        for "no data".
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning(
            "networkx not available — influence_network fields will be None. "
            "Add networkx>=3.0 to dependencies."
        )
        return None

    idx_map: dict[int, pd.Timestamp] = {}
    if idx_by_patid:
        for k, v in idx_by_patid.items():
            if v is not None:
                idx_map[int(k)] = pd.Timestamp(v)

    hcps_by_patient: dict[int, set[str]] = defaultdict(set)

    def _collect(df: pd.DataFrame, date_col: str) -> None:
        if "npi" not in df.columns or "patid" not in df.columns:
            return
        sub = df[df["patid"].isin(kept_patids)]
        has_date = date_col in sub.columns
        for _, r in sub.iterrows():
            pid = r.get("patid")
            if pd.isna(pid):
                continue
            pid_int = int(pid)
            if idx_map and has_date:
                fill_date = r.get(date_col)
                if pd.isna(fill_date):
                    continue
                idx_dt = idx_map.get(pid_int)
                if idx_dt is None:
                    continue
                if not (
                    (fill_date > idx_dt - timedelta(days=lookback_days)) and (fill_date <= idx_dt)
                ):
                    continue
            nv = r.get("npi")
            if pd.isna(nv):
                continue
            ns = str(nv).strip()
            if not ns or ns == "nan":
                continue
            hcps_by_patient[pid_int].add(ns)

    _collect(med, "medication_date")
    _collect(proc, "proc_date")

    edge_weight: dict[tuple[str, str], int] = defaultdict(int)
    node_set: set[str] = set()
    for hcp_set in hcps_by_patient.values():
        members = sorted(hcp_set)
        node_set.update(members)
        n = len(members)
        for i in range(n):
            for j in range(i + 1, n):
                edge_weight[(members[i], members[j])] += 1

    graph: Any = nx.Graph()
    graph.add_nodes_from(node_set)
    for (a, b), w in edge_weight.items():
        graph.add_edge(a, b, weight=int(w))
    return graph


def score_hcp_influence_graph(
    graph: Any,
    scale: float = PEER_INFLUENCE_SCALE,
) -> tuple[dict[str, int], dict[str, float]]:
    """Issue #169: derive degree + scaled eigenvector centrality from a graph.

    Pure helper factored out of the converter for re-use by the
    FalkorDB round-trip parity test. Matches PR #168's behaviour
    exactly:

    - ``influence_network_size`` = ``graph.degree(n)`` per node.
    - ``peer_influence_score`` = ``eigenvector_centrality`` per
      connected component (to avoid ``PowerIterationFailedConvergence``
      on disconnected graphs), then ``round(min(v, 1.0) * scale, 2)``
      clamped to ``[0.00, 9.99]`` to fit ``DECIMAL(3,2)``.

    Args:
        graph: a ``networkx.Graph`` as returned by
            :func:`build_hcp_influence_graph`. An empty graph yields
            two empty dicts.
        scale: ``PEER_INFLUENCE_SCALE`` (9.99). Exposed for
            unit-test ergonomics; production callers use the default.

    Returns:
        ``(degree_by_npi, centrality_by_npi)`` — same shape as the
        converter's pre-issue-169 internal API.
    """
    try:
        import networkx as nx
    except ImportError:
        return {}, {}

    if graph is None or graph.number_of_nodes() == 0:
        return {}, {}

    degree_by_npi: dict[str, int] = {n: int(graph.degree(n)) for n in graph.nodes()}

    centrality_raw: dict[str, float] = dict.fromkeys(graph.nodes(), 0.0)
    for component in nx.connected_components(graph):
        sub_g = graph.subgraph(component)
        if sub_g.number_of_edges() == 0:
            continue
        try:
            c = nx.eigenvector_centrality(sub_g, max_iter=1000, tol=1e-6, weight="weight")
        except nx.PowerIterationFailedConvergence:
            logger.warning(
                "eigenvector_centrality failed on component size %d — falling back to 0.0",
                sub_g.number_of_nodes(),
            )
            continue
        for k, v in c.items():
            centrality_raw[k] = float(v)

    centrality_by_npi: dict[str, float] = {}
    for n, v in centrality_raw.items():
        scaled = round(min(v, 1.0) * scale, 2)
        centrality_by_npi[n] = max(0.0, min(9.99, scaled))

    return degree_by_npi, centrality_by_npi


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
        comorbidity_method: str = COMORBIDITY_METHOD_DEFAULT,
        soft_enrollment_filter: bool = False,
        min_data_quality_score: float | None = None,
    ) -> None:
        if enrollment_regime not in ENROLLMENT_REGIMES:
            allowed = sorted(ENROLLMENT_REGIMES.keys())
            raise ValueError(
                f"enrollment_regime={enrollment_regime!r} not in {allowed}; "
                f"plan v3 §3 Tier 1A defines exactly two regimes."
            )
        if comorbidity_method not in COMORBIDITY_METHODS_ALLOWED:
            raise ValueError(
                f"comorbidity_method={comorbidity_method!r} not in "
                f"{COMORBIDITY_METHODS_ALLOWED}; issue #156 item 3."
            )
        if min_data_quality_score is not None:
            if not 0.0 <= min_data_quality_score <= 1.0:
                raise ValueError(
                    f"min_data_quality_score={min_data_quality_score!r} not in [0.0, 1.0]; "
                    f"issue #156 item 5."
                )
        self.parquet_dir = Path(parquet_dir)
        self.output_dir = Path(output_dir)
        self.cohorts = cohorts
        self.max_patients = max_patients
        self.pilot_audit = pilot_audit
        self.enrollment_regime = enrollment_regime
        self.comorbidity_method = comorbidity_method
        # Issue #156 item 5: soft enrollment filter (opt-in). When False
        # (default), the historical hard `continuous_enrollment == 1` gate
        # is preserved bit-for-bit; CSU cohort behavior unchanged.
        self.soft_enrollment_filter = soft_enrollment_filter
        self.min_data_quality_score = (
            min_data_quality_score
            if min_data_quality_score is not None
            else DEFAULT_MIN_DATA_QUALITY_SCORE
        )
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
        # Issue #156 item 5: when soft_enrollment_filter is True, partially-
        # enrolled patients are kept in the cohort and their `enroll_complete < 1.0`
        # propagates into data_quality_score. When False (default, preserves
        # historical CSU cohort behavior bit-for-bit), the legacy hard filter
        # applies.
        if self.soft_enrollment_filter:
            partial = demo[demo["continuous_enrollment"] != 1]
            self._attrition.append(
                (
                    f"{cohort}: soft-filtered partial-enrollment kept (DQS gates downstream)",
                    len(partial),
                )
            )
        else:
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

        # 6b. Issue #156 item 5: soft DQS filter. When soft enrollment is on,
        # apply `min_data_quality_score` as a soft filter rather than a hard
        # exclusion — patients below threshold are LOGGED in attrition under
        # "soft-filtered (low DQS)" but remain in the cohort. Analysts choose
        # whether to drop them at model-training time. When soft mode is off
        # the legacy behavior is preserved (no DQS-based filtering at ETL).
        if self.soft_enrollment_filter:
            low_dqs = [
                j
                for j in journeys
                if (j.get("data_quality_score") or 0.0) < self.min_data_quality_score
            ]
            self._attrition.append(
                (
                    f"{cohort}: soft-filtered (low DQS, DQS<{self.min_data_quality_score:.2f})",
                    len(low_dqs),
                )
            )

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

        # 8. Treatment events + HCP profiles from the kept patients.
        # Issue #157 PR C (Sub-PR-A): pass cohort + per-patient init_date
        # so the discontinuation cohort can emit biologic-fill rows
        # within the post-init treatment window and tag them with the
        # claim-pattern `treatment_response` proxy.
        kept_patids = {j["_patid"] for j in journeys}
        init_date_by_patid: dict[int, pd.Timestamp | None] = {}
        if cohort == "discontinuation":
            for j in journeys:
                # Discontinuation cohort's index_date IS the first biologic
                # fill (see `_derive_index_date`), so init == index_date.
                init_date_by_patid[int(j["_patid"])] = pd.Timestamp(j["index_date"])
        events = self._build_treatment_events(
            kept_patids,
            journeys,
            cohort=cohort,
            init_date_by_patid=init_date_by_patid,
        )
        # Issue #156 items 1 + 2: pass patid → index_date so both
        # priority_tier (rolling 12-month TRx) and influence_network
        # (pre-index 180d lookback) can apply PER-PATIENT temporal
        # gating. Updated per codex PR-2 pass-1 MEDIUM-1 + MEDIUM-2:
        # the pre-fix code used a cohort-wide endpoint that admitted
        # post-index leakage for early-index patients. Per-patient
        # gating eliminates that leakage path.
        idx_by_patid_for_hcp = {int(j["_patid"]): pd.Timestamp(j["index_date"]) for j in journeys}
        hcps = self._build_hcp_profiles(kept_patids, idx_by_patid_for_hcp)

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
        """Check whether the patient meets the enrollment-window requirement.

        Strict mode (default, `soft_enrollment_filter=False`): require eligeff
        and eligend non-null AND eligeff ≤ (index - pre_days) AND
        eligend ≥ (index + post_days). This is the historical hard gate.

        Soft mode (`soft_enrollment_filter=True`, issue #156 item 5): keep
        partial-enrollment patients in the cohort even if their eligibility
        window doesn't fully cover pre/post days, as long as eligeff/eligend
        are non-null (so we at least know SOME enrollment span exists). The
        downstream data_quality_score will reflect the partial enrollment via
        `enroll_complete < 1.0`. Patients with null eligibility dates are
        still dropped — we have no signal at all to score against.
        """
        eligeff = demo_row.get("eligeff")
        eligend = demo_row.get("eligend")
        if pd.isna(eligeff) or pd.isna(eligend):
            return False
        if self.soft_enrollment_filter:
            # In soft mode, accept any non-null eligibility span.
            return True
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
    # Data quality score (§7 + issue #156 item 4)                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dx_complete(row: pd.Series) -> float:
        """1 if at least one non-null/non-UNK dx code on the row, else 0.

        Checks inpatient diag1..5 and demographics diagcode. The "UNK"
        placeholder is treated as missing per the issue body.
        """
        candidates = ("diag1", "diag2", "diag3", "diag4", "diag5", "diagcode")
        for col in candidates:
            v = row.get(col) if hasattr(row, "get") else None
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except (TypeError, ValueError):
                pass
            s = str(v).strip().upper()
            if s and s != "UNK":
                return 1.0
        return 0.0

    @staticmethod
    def _proc_complete(row: pd.Series) -> float:
        """1 if `proc_code` non-null/non-empty, else 0."""
        v = row.get("proc_code") if hasattr(row, "get") else None
        if v is None:
            return 0.0
        try:
            if pd.isna(v):
                return 0.0
        except (TypeError, ValueError):
            pass
        return 1.0 if str(v).strip() else 0.0

    @staticmethod
    def _cost_complete(row: pd.Series, is_pharmacy: bool) -> float:
        """Cost completeness per issue #156 item 4 component rules.

        - 1.0 if std_cost present.
        - 0.5 if std_cost absent but any of (charge, copay, coins, deduct) present.
        - 0.0 otherwise.

        is_pharmacy is reserved for future use (dispfee/avgwhlsl only apply to
        pharmacy claims and must NOT penalize medical claims when absent).
        """
        for field in DQS_COST_FIELDS_PRIMARY:
            v = row.get(field) if hasattr(row, "get") else None
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except (TypeError, ValueError):
                pass
            return 1.0
        for field in DQS_COST_FIELDS_FALLBACK:
            v = row.get(field) if hasattr(row, "get") else None
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except (TypeError, ValueError):
                pass
            return 0.5
        return 0.0

    @staticmethod
    def _enroll_complete(demo_row: pd.Series) -> float:
        """Enrollment completeness from demographics row.

        - 1.0 if eligeff + eligend non-null AND continuous_enrollment == 1.
        - 0.5 if eligeff + eligend non-null but continuous_enrollment != 1.
        - 0.0 if any date null.
        """
        eligeff = demo_row.get("eligeff")
        eligend = demo_row.get("eligend")
        ce = demo_row.get("continuous_enrollment")
        dates_present = pd.notna(eligeff) and pd.notna(eligend)
        if not dates_present:
            return 0.0
        try:
            ce_int = int(ce) if ce is not None and not pd.isna(ce) else 0
        except (TypeError, ValueError):
            ce_int = 0
        if ce_int == 1:
            return 1.0
        return 0.5

    def _compute_data_quality_score(
        self,
        *,
        patid: int,
        lb_start: pd.Timestamp,
        lb_end: pd.Timestamp,
        demo_row: pd.Series,
        feats: dict[str, Any],
    ) -> float:
        """Per-patient weighted data quality score (issue #156 item 4).

        Averages a per-claim 4-component DQS over all claims in the lookback
        window:
            claim_dqs = 0.40 dx + 0.25 proc + 0.20 cost + 0.15 enroll
        When the patient has zero claims in lookback, falls back to the
        legacy feature-completeness fraction so empty-window patients still
        get a non-null DQS (their cohort eligibility is gated elsewhere).
        """
        enroll_score = self._enroll_complete(demo_row)
        claim_scores: list[float] = []

        # Inpatient claims — dx + cost (no proc_code column expected)
        ip = self._inpatient_by_pat.get(patid)
        if ip is not None:
            ip_w = ip[(ip["admit_date"] >= lb_start) & (ip["admit_date"] <= lb_end)]
            for _, row in ip_w.iterrows():
                dx = self._dx_complete(row)
                cost = self._cost_complete(row, is_pharmacy=False)
                claim_scores.append(
                    DQS_WEIGHT_DX * dx
                    + DQS_WEIGHT_PROC * 0.0
                    + DQS_WEIGHT_COST * cost
                    + DQS_WEIGHT_ENROLL * enroll_score
                )

        # Procedure claims — proc + cost
        proc = self._proc_by_pat.get(patid)
        if proc is not None:
            proc_w = proc[(proc["proc_date"] >= lb_start) & (proc["proc_date"] <= lb_end)]
            for _, row in proc_w.iterrows():
                pc = self._proc_complete(row)
                cost = self._cost_complete(row, is_pharmacy=False)
                claim_scores.append(
                    DQS_WEIGHT_DX * 0.0
                    + DQS_WEIGHT_PROC * pc
                    + DQS_WEIGHT_COST * cost
                    + DQS_WEIGHT_ENROLL * enroll_score
                )

        # Medication claims — proc (HCPCS in proc_code if present) + cost (pharmacy)
        med = self._med_by_pat.get(patid)
        if med is not None:
            med_w = med[(med["medication_date"] >= lb_start) & (med["medication_date"] <= lb_end)]
            for _, row in med_w.iterrows():
                pc = self._proc_complete(row)
                cost = self._cost_complete(row, is_pharmacy=True)
                claim_scores.append(
                    DQS_WEIGHT_DX * 0.0
                    + DQS_WEIGHT_PROC * pc
                    + DQS_WEIGHT_COST * cost
                    + DQS_WEIGHT_ENROLL * enroll_score
                )

        if claim_scores:
            return round(sum(claim_scores) / len(claim_scores), 3)

        # Empty window — fall back to legacy feature-completeness fraction.
        # Codex pass-1 LOW: exclude the new payer audit-trail fields
        # (payer_bus_raw / payer_product_raw / payer_health_exch_raw /
        # payer_lis_dual_raw) so an empty-window DQS does not shift solely
        # because this PR added audit-trail columns to `feats`. The derived
        # `payer_category` IS included because it's a true downstream
        # feature, not an audit field. Keys starting with `_` were already
        # excluded.
        excluded_audit_keys = {
            "payer_bus_raw",
            "payer_product_raw",
            "payer_health_exch_raw",
            "payer_lis_dual_raw",
        }
        feat_vals = [
            v for k, v in feats.items() if not k.startswith("_") and k not in excluded_audit_keys
        ]
        non_null = sum(1 for v in feat_vals if v is not None and v != "")
        return round(non_null / max(len(feat_vals), 1), 3)

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

        # Issue #156 item 6: payer_category derivation from demographics.
        # Persist BOTH the derived 8-vocabulary value AND the raw source
        # fields (bus / product / health_exch / lis_dual) on the journey
        # record to enable re-derivation without re-ETL and to preserve the
        # audit trail. The legacy `insurance_type` field is kept for back-compat.
        bus_raw = demo_row.get("bus")
        product_raw = demo_row.get("product")
        health_exch_raw = demo_row.get("health_exch")
        lis_dual_raw = demo_row.get("lis_dual")
        feats["payer_category"] = rwdc.derive_payer_category(
            bus=bus_raw,
            product=product_raw,
            health_exch=health_exch_raw,
            lis_dual=lis_dual_raw,
        )
        feats["payer_bus_raw"] = str(bus_raw).strip().upper() if pd.notna(bus_raw) else None
        feats["payer_product_raw"] = (
            str(product_raw).strip().upper() if pd.notna(product_raw) else None
        )
        feats["payer_health_exch_raw"] = (
            rwdc.is_truthy_flag(health_exch_raw)
            if pd.notna(health_exch_raw) and health_exch_raw is not None
            else None
        )
        feats["payer_lis_dual_raw"] = (
            rwdc.is_truthy_flag(lis_dual_raw)
            if pd.notna(lis_dual_raw) and lis_dual_raw is not None
            else None
        )
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
        if self.comorbidity_method == "quan":
            feats["elixhauser_score"] = self._elixhauser_quan(patid, lb_start, lb_end)
            feats["charlson_score"] = self._charlson_quan(patid, lb_start, lb_end)
        else:
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

    def _collect_icd_codes_in_window(
        self, patid: int, lb_start: pd.Timestamp, lb_end: pd.Timestamp
    ) -> list[str]:
        """Collect upper-cased ICD-10 codes from inpatient claims in the window.

        Empty when no inpatient table or no in-window claims. Returns a flat
        list of code strings (de-dotted at source by the Optum extracts).
        """
        ip = self._inpatient_by_pat.get(patid)
        if ip is None:
            return []
        ip_w = ip[(ip["admit_date"] >= lb_start) & (ip["admit_date"] <= lb_end)]
        codes: list[str] = []
        for c in ("diag1", "diag2", "diag3", "diag4", "diag5"):
            if c in ip_w.columns:
                codes.extend(ip_w[c].dropna().astype(str).str.upper().tolist())
        return codes

    def _elixhauser_approx(self, patid: int, lb_start: pd.Timestamp, lb_end: pd.Timestamp) -> int:
        """Minimal Elixhauser proxy: count of distinct ICD-10 chapters in lookback.

        Retained for backwards compatibility under COMORBIDITY_METHOD == "approx".
        The "quan" path is the validated Elixhauser/van Walraven score via
        `_elixhauser_quan`. See issue #156 item 3.
        """
        codes = self._collect_icd_codes_in_window(patid, lb_start, lb_end)
        chapters: set[str] = {code[0] for code in codes if code}
        return len(chapters)

    def _charlson_approx(self, patid: int, lb_start: pd.Timestamp, lb_end: pd.Timestamp) -> int:
        """Minimal Charlson proxy: distinct high-severity categories present.

        Retained for backwards compatibility under COMORBIDITY_METHOD == "approx".
        The "quan" path is the validated Quan-Charlson weighted index via
        `_charlson_quan`. See issue #156 item 3.
        """
        cats = {
            "mi": ("I21", "I22", "I252"),
            "chf": ("I099", "I110", "I130", "I132", "I255", "I420", "I425"),
            "cancer": ("C",),
            "diabetes": ("E10", "E11", "E12", "E13", "E14"),
            "renal": ("N18", "N19"),
        }
        codes = self._collect_icd_codes_in_window(patid, lb_start, lb_end)
        if not codes:
            return 0
        codes_upper = pd.Series(codes, dtype=object)
        present: set[str] = set()
        for cat_name, prefixes in cats.items():
            if codes_upper.str.startswith(prefixes).any():
                present.add(cat_name)
        return len(present)

    def _charlson_quan(self, patid: int, lb_start: pd.Timestamp, lb_end: pd.Timestamp) -> int:
        """Quan et al. (2005) Charlson Comorbidity Index over inpatient claims.

        Implements the 17-category ICD-10 mapping from Quan 2005 Med Care
        43(11):1130-1139 with the original Charlson weights (1, 2, 3, 6).
        Hierarchies are applied: severe liver disease supersedes mild liver
        disease; metastatic solid tumor supersedes any malignancy; diabetes
        with complications supersedes diabetes without complications. The
        returned score is the sum of weights of distinct categories present.
        """
        codes = self._collect_icd_codes_in_window(patid, lb_start, lb_end)
        if not codes:
            return 0
        codes_s = pd.Series(codes, dtype=object)

        def _any(prefixes: tuple[str, ...]) -> bool:
            return bool(codes_s.str.startswith(prefixes).any())

        present: dict[str, int] = {}
        # Weight 1
        if _any(QUAN_CHARLSON["myocardial_infarction"]):
            present["myocardial_infarction"] = 1
        if _any(QUAN_CHARLSON["congestive_heart_failure"]):
            present["congestive_heart_failure"] = 1
        if _any(QUAN_CHARLSON["peripheral_vascular_disease"]):
            present["peripheral_vascular_disease"] = 1
        if _any(QUAN_CHARLSON["cerebrovascular_disease"]):
            present["cerebrovascular_disease"] = 1
        if _any(QUAN_CHARLSON["dementia"]):
            present["dementia"] = 1
        if _any(QUAN_CHARLSON["chronic_pulmonary_disease"]):
            present["chronic_pulmonary_disease"] = 1
        if _any(QUAN_CHARLSON["rheumatic_disease"]):
            present["rheumatic_disease"] = 1
        if _any(QUAN_CHARLSON["peptic_ulcer_disease"]):
            present["peptic_ulcer_disease"] = 1
        # Liver hierarchy (mild vs severe)
        severe_liver = _any(QUAN_CHARLSON["severe_liver_disease"])
        if severe_liver:
            present["severe_liver_disease"] = 3
        elif _any(QUAN_CHARLSON["mild_liver_disease"]):
            present["mild_liver_disease"] = 1
        # Diabetes hierarchy (with-complications supersedes without)
        diab_compl = _any(QUAN_CHARLSON["diabetes_complications"])
        if diab_compl:
            present["diabetes_complications"] = 2
        elif _any(QUAN_CHARLSON["diabetes_no_complications"]):
            present["diabetes_no_complications"] = 1
        if _any(QUAN_CHARLSON["hemiplegia_paraplegia"]):
            present["hemiplegia_paraplegia"] = 2
        if _any(QUAN_CHARLSON["renal_disease"]):
            present["renal_disease"] = 2
        # Cancer hierarchy (metastatic supersedes any-malignancy)
        metastatic = _any(QUAN_CHARLSON["metastatic_solid_tumor"])
        if metastatic:
            present["metastatic_solid_tumor"] = 6
        elif _any(QUAN_CHARLSON["any_malignancy"]):
            present["any_malignancy"] = 2
        if _any(QUAN_CHARLSON["aids_hiv"]):
            present["aids_hiv"] = 6

        return int(sum(present.values()))

    def _elixhauser_quan(self, patid: int, lb_start: pd.Timestamp, lb_end: pd.Timestamp) -> int:
        """Elixhauser index via Quan (2005) ICD-10 + van Walraven (2009) weights.

        Returns the van Walraven weighted summary score (can be negative for
        protective categories such as alcohol abuse in some derivations; here
        we use the published positive/negative weights as-is).
        """
        codes = self._collect_icd_codes_in_window(patid, lb_start, lb_end)
        if not codes:
            return 0
        codes_s = pd.Series(codes, dtype=object)
        score = 0
        for category, prefixes in QUAN_ELIXHAUSER.items():
            if bool(codes_s.str.startswith(prefixes).any()):
                score += VAN_WALRAVEN_WEIGHTS.get(category, 0)
        return int(score)

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
        """Gap > class-specific threshold between (fill_end) and next fill within 180 days.

        For CSU biologics (Xolair/Dupixent) this is the historical 90-day threshold
        via GAP_THRESHOLDS["biologic"]; behavior is bit-for-bit unchanged.
        """
        discont_gap, _ = _resolve_gap_thresholds("biologic")
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
            if (next_fill - fill_end).days > discont_gap:
                return 1
        # Last fill end extends past prediction end? → persistent
        last = bio.iloc[-1]
        ds = rwdc.safe_int(last.get("days_sup")) or 0
        last_end = last["medication_date"] + timedelta(days=ds)
        return int(last_end < end - timedelta(days=discont_gap))

    def _target_persistent_at_180d(self, patid: int, init_date: pd.Timestamp) -> int:
        """Any fill active at day 180 (days_supply-based, no gap > class threshold).

        For CSU biologics (Xolair/Dupixent) this is the historical 60-day threshold
        via GAP_THRESHOLDS["biologic"]; behavior is bit-for-bit unchanged.
        """
        _, pers_gap = _resolve_gap_thresholds("biologic")
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
        # Check gap criterion: fills spaced < persistence threshold apart up to target_day
        bio = bio[bio["medication_date"] <= target_day]
        if bio.empty:
            return 0
        bio = bio.reset_index(drop=True)
        for i in range(len(bio) - 1):
            gap = (bio.iloc[i + 1]["medication_date"] - bio.iloc[i]["medication_date"]).days
            if gap > pers_gap:
                return 0
        # No oversized inter-fill gap — but the fills must also extend to near
        # day 180. Mirror ``_target_discontinued_180d``: require the last
        # in-window fill's days_supply coverage to reach within pers_gap of
        # target_day. (A single early fill never enters the gap loop and would
        # otherwise fall through to a spurious persistent=1.)
        last = bio.iloc[-1]
        ds = rwdc.safe_int(last.get("days_sup")) or 0
        last_end = last["medication_date"] + timedelta(days=ds)
        return int(last_end >= target_day - timedelta(days=pers_gap))

    # ------------------------------------------------------------------ #
    # Issue #157 PR C — treatment_response proxy derivation               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classify_biologic_brand(row: pd.Series) -> str | None:
        """Return 'xolair' / 'dupixent' / None for a medication row.

        Used to detect biologic switch (different NDC prefix) for the
        `refractory` rule. Reuses the same signals as `_csu_biologic_mask`
        but distinguishes between the two brands.
        """
        # Brand_Name takes priority where present.
        bn = row.get("Brand_Name")
        if pd.notna(bn):
            b = str(bn).strip().upper()
            if "XOLAIR" in b:
                return "xolair"
            if "DUPIXENT" in b:
                return "dupixent"
        gn = row.get("Generic_Name")
        if pd.notna(gn):
            g = str(gn).strip().lower()
            if "omalizumab" in g:
                return "xolair"
            if "dupilumab" in g:
                return "dupixent"
        code = row.get("code")
        if pd.notna(code):
            c = str(code).strip().upper()
            # Xolair = NDC prefix 50242, HCPCS J2357.
            if c.startswith("50242") or c == "J2357":
                return "xolair"
            # Dupixent = NDC prefix 0024/00024, HCPCS J0517 (per spec).
            if c.startswith("00024") or c.startswith("0024") or c == "J0517":
                return "dupixent"
        return None

    def _coverage_days(self, bio_fills: pd.DataFrame) -> int:
        """Total covered days across (non-overlapping union of) biologic fills.

        Each fill contributes [fill_date, fill_date + days_sup). Overlaps are
        union'd so back-to-back overlapping fills do not double-count.
        """
        if bio_fills.empty:
            return 0
        intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for _, row in bio_fills.iterrows():
            fd = row.get("medication_date")
            if pd.isna(fd):
                continue
            ds = rwdc.safe_int(row.get("days_sup")) or 0
            if ds <= 0:
                continue
            intervals.append((fd, fd + timedelta(days=ds)))
        if not intervals:
            return 0
        intervals.sort()
        total = 0
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                total += (cur_end - cur_start).days
                cur_start, cur_end = start, end
        total += (cur_end - cur_start).days
        return total

    def _has_rescue_steroid_burst(
        self, patid: int, init_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> bool:
        """True if patient has >=1 oral-steroid burst within [init, end]."""
        grp = self._med_by_pat.get(patid)
        if grp is None or grp.empty:
            return False
        win = grp[(grp["medication_date"] >= init_date) & (grp["medication_date"] <= end_date)]
        if win.empty:
            return False
        gen_series = win.get("Generic_Name")
        if gen_series is None:
            return False
        gen_lower = gen_series.astype(str).str.lower()
        steroid_mask = pd.Series(False, index=win.index)
        for s in RESCUE_STEROID_GENERICS:
            steroid_mask = steroid_mask | gen_lower.str.contains(s, na=False)
        if not steroid_mask.any():
            return False
        cand = win.loc[steroid_mask]
        for _, row in cand.iterrows():
            ds = rwdc.safe_int(row.get("days_sup")) or 0
            if ds >= RESCUE_STEROID_MIN_DAYS_SUP:
                return True
        return False

    def _observable_followup_days(
        self, patid: int, init_date: pd.Timestamp, end_window: pd.Timestamp
    ) -> int:
        """Return observable follow-up days for response pre-condition gate.

        Computed as ``min(eligend, end_window) - init_date``. Defends
        against soft-enrollment-filter / research-mode short-window cases
        where a patient is enrolled in the cohort but has <90d real
        observability post-initiation.

        Falls back to ``self.enrollment_post_days`` when eligend is
        unavailable; under research-mode that fallback can return 90d,
        which equals the pre-condition threshold and still gates
        correctly. Under strict mode the fallback returns 180d and the
        check is satisfied trivially.
        """
        try:
            eligend_series = self.demo.loc[self.demo["patid"] == patid, "eligend"]
        except (KeyError, AttributeError):
            return int((end_window - init_date).days)
        if eligend_series.empty:
            return int(getattr(self, "enrollment_post_days", 180))
        eligend = eligend_series.iloc[0]
        if pd.isna(eligend):
            return int(getattr(self, "enrollment_post_days", 180))
        eligend_ts = pd.Timestamp(eligend)
        cap = min(eligend_ts, end_window)
        return int(max((cap - init_date).days, 0))

    def _has_urticaria_ed_visit(
        self, patid: int, init_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> bool:
        """True if patient has >=1 urticaria/angioedema ED visit in window.

        ED = inpatient claim with pos == '23' (CMS POS code for "Emergency
        Room — Hospital") AND any of diag1..5 prefixes with L50 or T783.
        Procedure-data POS is not populated in Optum (see issue #156 PR B
        item 7), so we use inpatient claims only.
        """
        grp = self._inpatient_by_pat.get(patid)
        if grp is None or grp.empty:
            return False
        win = grp[(grp["admit_date"] >= init_date) & (grp["admit_date"] <= end_date)]
        if win.empty:
            return False
        pos_series = win.get("pos")
        if pos_series is None:
            return False
        pos_str = pos_series.astype(str).str.strip()
        ed_mask = pos_str == ED_POS_CODE
        if not ed_mask.any():
            return False
        ed_rows = win.loc[ed_mask]
        for _, row in ed_rows.iterrows():
            for col in ("diag1", "diag2", "diag3", "diag4", "diag5"):
                val = row.get(col)
                if pd.isna(val):
                    continue
                code = str(val).upper().replace(".", "")
                for prefix in ED_CSU_DX_PREFIXES:
                    if code.startswith(prefix):
                        return True
        return False

    def _derive_treatment_response(
        self, patid: int, init_date: pd.Timestamp
    ) -> tuple[str | None, str | None]:
        """Derive (treatment_response, outcome_indicator) per issue #157 spec.

        Returns ``(None, None)`` when the biologic-fill pre-conditions
        (>=60d coverage, >=90d follow-up) are unmet. Otherwise returns the
        classification per the first-match-wins rule order and the mapped
        outcome_indicator.

        Called by ``_build_treatment_events`` for biologic-fill rows on
        the discontinuation cohort (where ``init_date`` is the first
        biologic fill). Index_date is exposed as ``init_date`` to make
        the contract explicit at the call site.

        Follow-up enforcement: under strict cohort gating
        (`enrollment_post_days=180`) the 180d follow-up is guaranteed by
        construction, but under `soft_enrollment_filter=True` or
        research-mode (`enrollment_post_days=90`) a patient can land in
        the cohort with <90d observable follow-up. We enforce the
        pre-condition against actual `eligend` from demographics; when
        eligend is missing, we fall back to the converter's
        `enrollment_post_days` as a documented over-estimate.
        """
        grp = self._med_by_pat.get(patid)
        if grp is None or grp.empty:
            return None, None

        end_window = init_date + timedelta(days=TREATMENT_RESPONSE_WINDOW_DAYS)

        # Biologic fills within the response window.
        mask = self._csu_biologic_mask(grp)
        bio = grp.loc[mask]
        bio = bio[(bio["medication_date"] >= init_date) & (bio["medication_date"] <= end_window)]
        bio = bio.sort_values("medication_date")
        if bio.empty:
            return None, None

        # Pre-condition: follow-up >= 90d.
        # Observable follow-up = min(eligend, end_window) - init_date.
        # If `self.demo` does not have eligend for this patid, we fall back
        # to `self.enrollment_post_days` (the cohort-gating post-window).
        observable_followup_days = self._observable_followup_days(patid, init_date, end_window)
        if observable_followup_days < TREATMENT_RESPONSE_MIN_FOLLOWUP_DAYS:
            return None, None

        # Pre-condition: coverage >= 60d.
        coverage = self._coverage_days(bio)

        # Rule 1: discontinued — gap > BIOLOGIC_DISCONT_GAP_DAYS within window.
        # Reuses the same logic as `_target_discontinued_180d` but does NOT
        # short-circuit to "discontinued" on a single fill (we require the
        # >=60d coverage pre-condition first; a lone fill with days_sup < 60
        # fails the pre-condition and returns NULL).
        if coverage < TREATMENT_RESPONSE_MIN_COVERAGE_DAYS:
            return None, None

        # Discontinuation gap check. Per issue #157 spec, the rule fires
        # ONLY on the inter-fill gap (> BIOLOGIC_DISCONT_GAP_DAYS between
        # last fill end-date and next fill). The trailing-edge variant
        # (last fill ends > 90d before window close) is part of
        # `_target_discontinued_180d` but NOT part of the response
        # classifier — a patient with one biologic fill that extends
        # 60-90d post-init and no further fills cannot be safely labeled
        # `discontinued` from claims alone (could be physician-directed
        # hold, refill not yet due, etc.). Such cases fall through to
        # the next rule.
        bio_list = bio.reset_index(drop=True)
        is_discontinued = False
        for i in range(len(bio_list) - 1):
            fill_date = bio_list.iloc[i]["medication_date"]
            ds = rwdc.safe_int(bio_list.iloc[i].get("days_sup")) or 0
            fill_end = fill_date + timedelta(days=ds)
            next_fill = bio_list.iloc[i + 1]["medication_date"]
            if (next_fill - fill_end).days > BIOLOGIC_DISCONT_GAP_DAYS:
                is_discontinued = True
                break
        if is_discontinued:
            # Spec: "discontinued -> worsened if no subsequent biologic,
            # else stable". The gap-based rule fires when fill_i+1 exists
            # within the window AND the gap to fill_i+1's end exceeds
            # BIOLOGIC_DISCONT_GAP_DAYS — that subsequent fill (whether
            # in-window or outside) is exactly the "subsequent biologic"
            # the spec refers to. We check ALL fills after init_date
            # (excluding init_date itself) for any later fill — captures
            # both in-window and post-window re-engagement.
            all_bio = grp.loc[mask]
            later = all_bio[all_bio["medication_date"] > init_date]
            outcome = "stable" if not later.empty else "worsened"
            return "discontinued", outcome

        # Rule 2: refractory — switch to the OTHER biologic OR addition of
        # immunosuppressant within window.
        index_brand = self._classify_biologic_brand(bio_list.iloc[0])
        switched = False
        for i in range(1, len(bio_list)):
            other = self._classify_biologic_brand(bio_list.iloc[i])
            if other is not None and index_brand is not None and other != index_brand:
                switched = True
                break
        if not switched:
            # Immunosuppressant addition.
            immuno_generics = NON_TARGET_DRUG_CLASSES.get("immunosupp", ())
            win = grp[
                (grp["medication_date"] >= init_date) & (grp["medication_date"] <= end_window)
            ]
            gen_series = win.get("Generic_Name") if not win.empty else None
            if gen_series is not None:
                gen_lower = gen_series.astype(str).str.lower()
                immuno_mask = pd.Series(False, index=win.index)
                for g in immuno_generics:
                    immuno_mask = immuno_mask | gen_lower.str.contains(g, na=False)
                if immuno_mask.any():
                    switched = True
        if switched:
            return "refractory", TREATMENT_RESPONSE_TO_OUTCOME["refractory"]

        # Rule 3: inadequate — rescue steroid burst OR urticaria/angioedema ED.
        if self._has_rescue_steroid_burst(patid, init_date, end_window):
            return "inadequate", TREATMENT_RESPONSE_TO_OUTCOME["inadequate"]
        if self._has_urticaria_ed_visit(patid, init_date, end_window):
            return "inadequate", TREATMENT_RESPONSE_TO_OUTCOME["inadequate"]

        # Rule 4: controlled — persistence met, no rescue events.
        return "controlled", TREATMENT_RESPONSE_TO_OUTCOME["controlled"]

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

        # Data quality score — issue #156 item 4. Weighted 4-component
        # claim-level DQS averaged per-patient over the lookback window:
        #   claim_dqs = 0.40 dx + 0.25 proc + 0.20 cost + 0.15 enroll
        # When the patient has zero claims in lookback the score falls back
        # to the legacy feature-completeness metric so empty-window patients
        # still get a non-null DQS (their cohort eligibility is gated elsewhere).
        lb_start_dqs = index_date - timedelta(days=LOOKBACK_DAYS)
        lb_end_dqs = index_date - timedelta(days=1)
        dq_score = self._compute_data_quality_score(
            patid=patid,
            lb_start=lb_start_dqs,
            lb_end=lb_end_dqs,
            demo_row=demo_row,
            feats=feats,
        )

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
        *,
        cohort: str = "initiation",
        init_date_by_patid: dict[int, pd.Timestamp | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Emit canonical treatment_event records for included patients only.

        Events originate from med/proc/lab filtered to [lookback_start, index_date]
        so the downstream ML pipeline observes pre-index events only. The target
        is already encoded on the journey; events are for feature provenance /
        narrative context.

        Issue #157 PR C (Sub-PR-A): for the discontinuation cohort, ALSO emit
        biologic-fill events in [init_date, init_date + 180d] with the
        derived `treatment_response` + `outcome_indicator`. These post-index
        events are NEVER used as ML features (the converter writes them only
        to `treatment_events`, never to the journey feature matrix), so the
        anti-leakage discipline for the risk model is preserved.
        """
        init_dates = init_date_by_patid or {}
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
            brand: str | None = None,
            treatment_response: str | None = None,
            outcome_indicator: str | None = None,
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
                "brand": brand,
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
                "outcome_indicator": outcome_indicator,
                "treatment_response": treatment_response,
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

            # Issue #157 PR C (Sub-PR-A): emit biologic-fill events at/after
            # init_date for the discontinuation cohort, with the derived
            # treatment_response + outcome_indicator. These post-index
            # events are NEVER consumed as ML features — they appear only
            # in `treatment_events` for KPI calculation
            # (brand_specific.py BR-001 / BR-002).
            init_dt = init_dates.get(patid) if cohort == "discontinuation" else None
            if init_dt is not None:
                tr, oc = self._derive_treatment_response(patid, init_dt)
                med_grp = self._med_by_pat.get(patid)
                if med_grp is not None and not med_grp.empty:
                    bio_mask = self._csu_biologic_mask(med_grp)
                    end_win = init_dt + timedelta(days=TREATMENT_RESPONSE_WINDOW_DAYS)
                    bio_win = med_grp.loc[bio_mask]
                    bio_win = bio_win[
                        (bio_win["medication_date"] >= init_dt)
                        & (bio_win["medication_date"] <= end_win)
                    ].sort_values("medication_date")
                    if not bio_win.empty:
                        # Only the FIRST biologic fill within the window
                        # carries the treatment_response label — that is
                        # the index biologic-fill episode per issue #157
                        # ("apply per (patient, biologic_fill_episode)").
                        # Later fills in the window are still emitted for
                        # provenance but with NULL response.
                        first_row = bio_win.iloc[0]
                        first_brand = self._classify_biologic_brand(first_row)
                        # `brand` enum on treatment_events is brand_type
                        # (defined in core schema). Schema constraints
                        # accept 'competitor' / 'innovator' / brand-name
                        # values depending on cohort. CSU biologics map
                        # to 'competitor' for non-Pluvicto cohorts per
                        # current converter convention; we mirror that
                        # via the patient_journeys.brand assignment.
                        events.append(
                            _emit(
                                seq,
                                patid=patid,
                                event_date=first_row.get("medication_date"),
                                event_type="prescription",
                                drug_name=(
                                    str(first_row["Brand_Name"]).title()
                                    if pd.notna(first_row.get("Brand_Name"))
                                    else None
                                ),
                                drug_ndc=(
                                    str(first_row["code"])
                                    if pd.notna(first_row.get("code"))
                                    else None
                                ),
                                dosage=(
                                    str(first_row["strength"])
                                    if pd.notna(first_row.get("strength"))
                                    else None
                                ),
                                duration=rwdc.safe_int(first_row.get("days_sup")),
                                brand="competitor",
                                treatment_response=tr,
                                outcome_indicator=oc,
                            )
                        )
                        seq += 1
                        # Trailing fills (provenance only — NULL response).
                        for i in range(1, len(bio_win)):
                            row = bio_win.iloc[i]
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
                                    drug_ndc=(
                                        str(row["code"]) if pd.notna(row.get("code")) else None
                                    ),
                                    dosage=(
                                        str(row["strength"])
                                        if pd.notna(row.get("strength"))
                                        else None
                                    ),
                                    duration=rwdc.safe_int(row.get("days_sup")),
                                    brand="competitor",
                                )
                            )
                            seq += 1
                        # Silence unused-variable lint if downstream code adds
                        # consumers later — `first_brand` is reserved for a
                        # follow-up emission (per-brand audit JSONL).
                        _ = first_brand
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
            Dupixent fill BEFORE the CSU approval date 2025-04-18; fills on
            or after are on-label and counted in the unified CSU curve).

        On-label CSU = (any Xolair fill on or after 2014-03-21) OR (any
        Dupixent fill on or after 2025-04-18). Pre-2025-04-18 Dupixent fills
        flag the HCP as off-label and are EXCLUDED from the on-label adoption
        calculation (but the on-label flag is still set if the same HCP later
        had an on-label fill of either drug).

        HCPs whose only fills are pre-approval (off-label) Dupixent get
        ``days_to_first_fill=None`` → non_adopter, with the off-label flag
        preserved separately so downstream consumers can carve them out for
        cross-indication adoption analysis.
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

        # Date-aware Dupixent eligibility: fills on or after the CSU approval
        # date (2025-04-18 per rwdc.BRAND_LAUNCH_DATES["dupixent"]["csu"]) are
        # on-label; fills before are off-label and excluded from on-label
        # adoption (but still flag the HCP).
        dupixent_csu_launch = pd.Timestamp(rwdc.BRAND_LAUNCH_DATES["dupixent"]["csu"])
        sub = sub.copy()
        sub["medication_date"] = pd.to_datetime(sub["medication_date"], errors="coerce")

        # Pre-approval Dupixent = off-label; flags the HCP regardless of whether
        # they also had a post-approval on-label fill.
        offlabel_mask = dupixent_mask & (sub["medication_date"] < dupixent_csu_launch)

        # On-label CSU fill = Xolair on/after 2014-03-21 OR Dupixent on/after
        # 2025-04-18. Pre-launch fills of either drug are data errors (the
        # drug literally didn't exist for CSU yet) and would otherwise clamp
        # to `innovator` via max(delta, 0), giving the HCP an artificially
        # early adoption rank. Apply the same date-aware gate to BOTH drugs.
        # NaT medication_date drops out via these comparisons.
        onlabel_xolair_mask = (~dupixent_mask) & (sub["medication_date"] >= brand_launch)
        onlabel_dupixent_mask = dupixent_mask & (sub["medication_date"] >= dupixent_csu_launch)
        onlabel_mask = onlabel_xolair_mask | onlabel_dupixent_mask

        onlabel = sub.loc[onlabel_mask].dropna(subset=["medication_date"])
        if not onlabel.empty:
            onlabel = onlabel.copy()
            onlabel["npi"] = onlabel["npi"].astype(str).str.strip()
            first_by_npi = onlabel.groupby("npi")["medication_date"].min()
            for npi_val, first_dt in first_by_npi.items():
                npi_key = str(npi_val)
                if npi_key not in days_out:
                    continue
                # By construction first_dt >= brand_launch (or
                # dupixent_csu_launch >= brand_launch), so delta_days >= 0
                # and no clamp is needed.
                delta_days = int((first_dt - brand_launch).days)
                days_out[npi_key] = max(delta_days, 0)

        # Off-label flag — only pre-approval Dupixent fills flag the HCP.
        if offlabel_mask.any():
            dup_npis = sub.loc[offlabel_mask, "npi"].astype(str).str.strip().unique()
            for npi_val in dup_npis:
                if npi_val in offlabel:
                    offlabel[npi_val] = True

        return days_out, offlabel

    # ------------------------------------------------------------------ #
    # Issue #156 item 1: priority_tier via rolling 12-mo TRx ZIP3 decile #
    # ------------------------------------------------------------------ #

    def _hcp_zip3_modal(self, npi: str, patient_sets: dict[str, set[int]]) -> str | None:
        """Return the modal ZIP3 across the HCP's treated patients.

        ZIP3 is sourced from ``demographics.zipcode_5`` (first 3 chars) per
        the project-wide convention in ``_compute_features``. ZIP3 (not ZIP5)
        is chosen per issue #156 item 1 because ZIP5 has too few HCPs per bin
        for stable decile assignment; ZIP3 gives ~900 bins nationally.

        Ties are broken alphabetically for determinism so two HCPs with
        identical TRx + tied ZIP3 modes always produce identical tiers.
        Returns ``None`` only when no patient in the set has a usable ZIP3
        (rare — demographics ZIP is nearly always populated).
        """
        pats = patient_sets.get(npi)
        if not pats or self.demo.empty or "zipcode_5" not in self.demo.columns:
            return None
        sub = self.demo[self.demo["patid"].isin(pats)]
        if sub.empty:
            return None
        zip_strs = sub["zipcode_5"].dropna().astype(str).str.split("_").str[0].str.strip()
        zip3s = zip_strs[zip_strs.str.len() >= 3].str[:3]
        if zip3s.empty:
            return None
        counts = zip3s.value_counts()
        top = counts.max()
        tied = sorted(counts[counts == top].index.tolist())
        return tied[0]

    def _compute_priority_tiers(
        self,
        kept_patids: set[int],
        npi_pat: dict[str, set[int]],
        idx_by_patid: dict[int, pd.Timestamp] | None,
    ) -> tuple[
        dict[str, int],
        dict[str, str | None],
        dict[str, int],
        dict[str, int | None],
    ]:
        """Issue #156 item 1: rolling 12-month CSU-biologic TRx → ZIP3 decile → tier.

        Returns four dicts keyed by obfuscated NPI:
          - npi → priority_tier (1=high, 5=low; 1-5 always populated, no None)
          - npi → modal ZIP3 (or None when no demographics ZIP is resolvable)
          - npi → biologic TRx count in the per-patient lookback window
          - npi → decile within ZIP3 (1-10; None for tier-5 defaults so the
                  data dictionary can disambiguate "below decile 1" from
                  "TRx=0 / no ZIP3")

        Algorithm (per issue body):
          1. Filter both ``self.med`` (NDC/HCPCS/generic biologic codes)
             AND ``self.proc`` (HCPCS J2357 / J0517 administered as
             buy-and-bill in office settings) for in-scope CSU biologics.
             Codex PR-2 pass-1 HIGH-1: ignoring procedure-side HCPCS
             undercounts office-administered Xolair/Dupixent TRx and
             defaults affected HCPs to tier 5.
          2. PER-PATIENT temporal gating: for each (patient, npi, fill)
             triple, count the fill ONLY IF
                 ``patient_index - 365d < fill_date <= patient_index``.
             This avoids the leakage risk where a cohort-wide endpoint
             (max-index) would let post-index fills for early-index
             patients sneak in. Codex PR-2 pass-1 MEDIUM-1.
          3. For each NPI in ``npi_pat``, aggregate fills across all
             treated patients into a single biologic TRx count.
          4. Group HCPs by modal ZIP3 (across their treated patients).
          5. Within each ZIP3, rank HCPs by TRx (ties broken by
             NDC/HCPCS-distinct count, then alphabetical NPI — per issue
             body — for determinism). Compute 10-bin equal-frequency
             deciles.
          6. Map decile → tier via ``PRIORITY_TIER_DECILE_MAP``.
          7. HCPs with TRx=0 (or no resolvable ZIP3) → tier 5, decile None.
        """
        zip3_by_npi: dict[str, str | None] = {}
        for npi in npi_pat:
            zip3_by_npi[npi] = self._hcp_zip3_modal(npi, npi_pat)

        # Per-patient index_date lookup (fallback: max date in med if a
        # patient has no index_date — used only by tests / standalone
        # invocation. In production every kept patient HAS index_date.)
        idx_map: dict[int, pd.Timestamp] = {}
        if idx_by_patid:
            for k, v in idx_by_patid.items():
                if v is not None:
                    idx_map[int(k)] = pd.Timestamp(v)
        # Fallback endpoint for tests where idx_by_patid is None.
        fallback_end: pd.Timestamp | None = None
        if not idx_map:
            if (
                not self.med.empty
                and "medication_date" in self.med.columns
                and self.med["medication_date"].notna().any()
            ):
                fallback_end = pd.Timestamp(self.med["medication_date"].max())
            else:
                fallback_end = pd.Timestamp(datetime.now(tz=UTC).date())

        def _patient_index(pid: int) -> pd.Timestamp | None:
            return idx_map.get(int(pid)) if idx_map else fallback_end

        # Compute biologic TRx count + NDC-distinct count per NPI.
        # Scans BOTH self.med (NDC + HCPCS + brand/generic) AND
        # self.proc (HCPCS-only — buy-and-bill office admin).
        trx_by_npi: dict[str, int] = dict.fromkeys(npi_pat, 0)
        ndc_distinct_by_npi: dict[str, set[str]] = {npi: set() for npi in npi_pat}

        def _count_fills(df: pd.DataFrame, date_col: str, code_col: str) -> None:
            if (
                df.empty
                or "npi" not in df.columns
                or date_col not in df.columns
                or "patid" not in df.columns
            ):
                return
            sub = df[df["patid"].isin(kept_patids)].copy()
            if sub.empty:
                return
            # For self.med use the full _csu_biologic_mask (NDC + HCPCS +
            # Brand_Name + Generic_Name). For self.proc only HCPCS is
            # available, so test J2357/J0517 directly. Codex PR-2 pass-1
            # HIGH-1: procedure-side biologic admins must contribute to TRx.
            if code_col == "code":
                bio_mask = self._csu_biologic_mask(sub)
            else:
                if code_col not in sub.columns:
                    return
                code_s = sub[code_col].astype(str).str.upper()
                bio_mask = code_s.isin(CSU_BIOLOGIC_HCPCS)
            sub = sub[bio_mask]
            if sub.empty:
                return
            for _, r in sub.iterrows():
                pid = r.get("patid")
                if pd.isna(pid):
                    continue
                fill_date = r.get(date_col)
                if pd.isna(fill_date):
                    continue
                idx_dt = _patient_index(int(pid))
                if idx_dt is None:
                    continue
                # Per-patient lookback: (index - 365, index].
                if not (
                    (fill_date > idx_dt - timedelta(days=PRIORITY_TIER_TRX_WINDOW_DAYS))
                    and (fill_date <= idx_dt)
                ):
                    continue
                nv = r.get("npi")
                if pd.isna(nv):
                    continue
                ns = str(nv).strip()
                if not ns or ns == "nan":
                    continue
                if ns not in trx_by_npi:
                    continue
                trx_by_npi[ns] = trx_by_npi.get(ns, 0) + 1
                code_val = r.get(code_col)
                if code_val is not None and not pd.isna(code_val):
                    ndc_distinct_by_npi[ns].add(str(code_val).strip().upper())

        _count_fills(self.med, "medication_date", "code")
        _count_fills(self.proc, "proc_date", "proc_code")

        # Group by ZIP3 and assign deciles within each group.
        tier_by_npi: dict[str, int] = {}
        decile_by_npi: dict[str, int | None] = {}
        zip3_groups: dict[str, list[str]] = defaultdict(list)
        for npi in npi_pat:
            z = zip3_by_npi.get(npi)
            if z is None or trx_by_npi.get(npi, 0) <= 0:
                # No ZIP3 OR zero TRx → tier 5 (kept in pool), decile None.
                tier_by_npi[npi] = PRIORITY_TIER_DEFAULT
                decile_by_npi[npi] = None
                continue
            zip3_groups[z].append(npi)

        for _zip3, members in zip3_groups.items():
            # Tie-break: descending TRx, then descending NDC-distinct,
            # then ASCENDING alphabetical NPI for full determinism.
            ordered = sorted(
                members,
                key=lambda n: (
                    -trx_by_npi[n],
                    -len(ndc_distinct_by_npi[n]),
                    n,
                ),
            )
            n_members = len(ordered)
            if n_members == 0:
                continue
            # Equal-frequency 10-bin decile. Top-ranked HCP (rank 0) is in
            # decile 10. We compute decile as
            #     decile = 10 - floor(rank * 10 / n_members), clamped to [1, 10].
            # With n_members < 10 each HCP lands in a distinct top-decile;
            # ZIP3s with very few HCPs collapse into the top deciles which
            # is the desired behavior (small market = each HCP is high-priority).
            for rank, npi in enumerate(ordered):
                decile = 10 - (rank * 10 // n_members)
                decile = max(1, min(10, decile))
                decile_by_npi[npi] = decile
                tier_by_npi[npi] = PRIORITY_TIER_DECILE_MAP.get(decile, PRIORITY_TIER_DEFAULT)

        return tier_by_npi, zip3_by_npi, trx_by_npi, decile_by_npi

    # ------------------------------------------------------------------ #
    # Issue #156 item 2: influence_network via shared-patient clique     #
    # ------------------------------------------------------------------ #

    def _compute_influence_network(
        self,
        kept_patids: set[int],
        idx_by_patid: dict[int, pd.Timestamp] | None = None,
    ) -> tuple[dict[str, int], dict[str, float]]:
        """Issue #156 item 2: shared-patient clique → degree + eigenvector_centrality.

        Delegates to the module-level :func:`build_hcp_influence_graph` and
        :func:`score_hcp_influence_graph` helpers so that the issue #169
        FalkorDB persistence script can reuse the exact same graph
        construction without depending on the converter class. The
        behaviour described below is the contract of those helpers.

        For each patient in ``kept_patids``, collect the set of treating
        HCPs from ``medication.npi ∪ procedure.npi`` WITHIN THE PER-PATIENT
        LOOKBACK WINDOW ``(patient_index - LOOKBACK_DAYS, patient_index]``.
        Each per-patient set forms a clique; the weighted HCP-HCP graph
        aggregates over patients.

        Codex PR-2 pass-1 MEDIUM-2: post-index and prediction-window
        HCP contacts would otherwise contribute edges and inflate
        centrality with future information. The per-patient lookback
        gate is symmetric with the priority_tier change (HIGH-1 /
        MEDIUM-1) and matches the project-wide convention that HCP
        features observed at index never include post-index data.

        When ``idx_by_patid`` is None (test invocation), the gate is
        skipped and the full med/proc rows are used. Production callers
        always thread ``idx_by_patid``.

        Returns:
          - degree_by_npi: ``influence_network_size`` per HCP
          - centrality_by_npi: ``peer_influence_score`` per HCP, scaled
            from ``eigenvector_centrality`` [0,1] → [0.00, 9.99] to fit
            DECIMAL(3,2). Power-iteration failures (rare; disconnected
            singletons) fall back to 0.0.

        IMPORTANT: this is a CLAIMS-DERIVED PROXY for KOL influence, NOT
        canonical KOL data. Canonical KOL data requires external
        commercial sources (Definitive Healthcare, HCS Spectrum) or
        PubMed co-authorship — explicitly out of scope per issue #156.
        Documented in the data dictionary.
        """
        graph = build_hcp_influence_graph(
            kept_patids=kept_patids,
            med=self.med,
            proc=self.proc,
            idx_by_patid=idx_by_patid,
            lookback_days=LOOKBACK_DAYS,
        )
        if graph is None:
            return {}, {}
        return score_hcp_influence_graph(graph, scale=PEER_INFLUENCE_SCALE)

    def _build_hcp_profiles(
        self,
        kept_patids: set[int],
        idx_by_patid: dict[int, pd.Timestamp] | None = None,
    ) -> list[dict[str, Any]]:
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
        # For CSU, the unified on-label adoption curve is anchored at Xolair
        # launch (2014-03-21, the class-of-modality anchor). Dupixent's CSU
        # approval came later (2025-04-18); pre-approval Dupixent CSU fills
        # are OFF-LABEL and excluded from the curve (flagged separately via
        # `dupixent_offlabel=True`); post-approval Dupixent CSU fills are
        # on-label and counted in the unified curve via the same Xolair
        # anchor (so a 2025-05 Dupixent first-fill is a late_majority, not
        # an innovator — the modality has been around since 2014).
        xolair_launch = pd.Timestamp(rwdc.BRAND_LAUNCH_DATES["xolair"]["csu"])
        hcp_days_to_first_fill, hcp_dupixent_offlabel = self._compute_npi_first_fill(
            kept_patids, npi_rx, xolair_launch
        )
        adoption_by_npi = rwdc.classify_rogers_adoption(hcp_days_to_first_fill)

        # Issue #156 item 1 + item 2: priority_tier + influence_network.
        (
            priority_tier_by_npi,
            zip3_by_npi,
            _trx_by_npi,
            decile_by_npi,
        ) = self._compute_priority_tiers(kept_patids, npi_pat, idx_by_patid)
        influence_size_by_npi, peer_score_by_npi = self._compute_influence_network(
            kept_patids, idx_by_patid
        )

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

            # Issue #154 §3: optional NPPES enrichment. Lookup uses the RAW
            # input NPI when it's already a valid 10-digit Luhn NPI (a real
            # cohort) so the cache is queried with the same key the loader
            # was populated under. For obfuscated cohorts (Optum / CSU as
            # shipped) the raw value is NOT a valid NPI; we deterministically
            # hash it via generate_luhn_npi() to produce a Luhn-valid output
            # NPI, but the cache lookup correctly misses (no real CMS record
            # to find). Pre-PR-1 codex post-merge MEDIUM-2: lookup was always
            # against generated_npi which would silently miss on real-NPI
            # cohorts even after a loader was registered.
            obf_str = str(obf).strip()
            # Use the STRICT CMS-NPI Luhn check (80840-prefix variant) so a
            # coincidentally-10-digit obfuscated key cannot skip the hashing
            # path. `generate_luhn_npi` uses plain Luhn without the 80840
            # prefix → its output reliably FAILS this check, so the two
            # branches partition cleanly. Codex PR #165 pass-1 MEDIUM.
            if rwdc.is_real_cms_npi(obf_str):
                generated_npi = obf_str
                lookup_key = obf_str
            else:
                generated_npi = rwdc.generate_luhn_npi(obf)
                lookup_key = generated_npi
            nppes_rec = rwdc.lookup_npi(lookup_key, use_api_fallback=False)
            sub_specialty: str | None = None
            practice_type_resolved = practice
            practice_size_resolved: str | None = None
            geographic_region: str | None = None
            state_val: str | None = None
            city_val: str | None = None
            zip_code_val: str | None = None
            years_experience: int | None = None
            affiliation_primary: str | None = None
            # Issue #249 / PR B-prime BP3: academic_hcp confounder field.
            # Derivation uses NPPES org taxonomy (academic medical center /
            # general acute care hospital codes from
            # `rwd_common.ACADEMIC_MEDICAL_CENTER_CODES`). Three-valued:
            #   - None  → cache miss (obfuscated cohort / unknown NPI):
            #             cannot derive academic status. Distinct from False.
            #   - True  → any taxonomy on the record matches the academic
            #             code set.
            #   - False → cache hit but no academic taxonomy match (incl.
            #             empty taxonomy list — the lookup established
            #             "we saw this provider; no academic affiliation").
            #
            # The downstream causal-engine confounder set reads this key
            # directly; obfuscated-NPI cohorts MUST yield None so consumers
            # distinguish "not derivable" from "not academic". See issue #249.
            academic_hcp: bool | None = None
            # PII fields (first_name / last_name) are intentionally NOT
            # populated from NppesRecord. Codex PR #162 post-merge MEDIUM-3:
            # the documented 8-field enrichment contract does not include
            # named provider PII, so keep them None at the cohort output
            # boundary even when the NPPES cache has them. A future PR that
            # explicitly opts into named-provider export must update the data
            # dictionary + downstream consumer contracts first.
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
                # Org-level providers (entity_type=NPPES_ENTITY_TYPE_ORGANIZATION)
                # → "Group" / "Hospital" already covered by the `practice`
                # heuristic; the `sole_proprietor` flag refines individual
                # providers down to "Solo".
                if nppes_rec.sole_proprietor is True and practice == "Group":
                    practice_type_resolved = "Solo"
                # practice_size: bucket via sole-proprietor + entity flag
                if nppes_rec.sole_proprietor is True:
                    practice_size_resolved = "Solo"
                elif nppes_rec.entity_type == rwdc.NPPES_ENTITY_TYPE_ORGANIZATION:
                    practice_size_resolved = "Group"
                # Issue #249: academic_hcp — cache HIT establishes derivability,
                # so the answer is True/False, never None. Match against
                # rwd_common.ACADEMIC_MEDICAL_CENTER_CODES (single source of
                # truth) across ALL taxonomies on the record, not just the
                # primary — a provider whose primary specialty is non-academic
                # but who lists an academic hospital taxonomy is still academic.
                academic_hcp = any(
                    rwdc.taxonomy_in(t.code, rwdc.ACADEMIC_MEDICAL_CENTER_CODES)
                    for t in nppes_rec.taxonomies
                )

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
                    "priority_tier": priority_tier_by_npi.get(obf, PRIORITY_TIER_DEFAULT),
                    "decile": decile_by_npi.get(obf),
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
                    "influence_network_size": influence_size_by_npi.get(obf),
                    "peer_influence_score": peer_score_by_npi.get(obf),
                    "adoption_category": adoption,
                    "academic_hcp": academic_hcp,
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
                "source_table": "medication (CSU on-label fills) via NPI",
                "lookback_window": "all CSU biologic fills in scope",
                "notes": (
                    "Issue #155 §1 — Rogers Diffusion of Innovations TIME-to-"
                    "first-fill (anchor: Xolair-CSU launch 2014-03-21). "
                    "non_adopter for HCPs with no on-label fill. On-label = "
                    "Xolair on/after 2014-03-21 OR Dupixent on/after CSU "
                    "approval 2025-04-18; pre-approval Dupixent fills are "
                    "excluded (see dupixent_offlabel). Replaces legacy "
                    "volume quartile classification."
                ),
            },
            {
                "feature": "dupixent_offlabel",
                "type": "bool",
                "source_table": "medication (Dupixent fills) via NPI",
                "lookback_window": "all CSU biologic fills in scope",
                "notes": (
                    "Issue #155 §1 — TRUE if HCP has any pre-approval "
                    "Dupixent fill in the CSU cohort (FDA approved Dupixent "
                    "for CSU on 2025-04-18; pre-2025-04-18 fills are "
                    "off-label, post-approval fills are on-label and counted "
                    "in the on-label adoption curve). Flagged for downstream "
                    "cross-indication adoption analysis."
                ),
            },
            {
                "feature": "academic_hcp",
                "type": "Optional[bool]",
                "source_table": "NPPES taxonomies via lookup_npi",
                "lookback_window": "n/a (provider attribute)",
                "notes": (
                    "Issue #249 / PR B-prime BP3 — causal-model confounder. "
                    "TRUE iff any NPPES taxonomy on the provider matches "
                    "rwd_common.ACADEMIC_MEDICAL_CENTER_CODES (single source "
                    "of truth). FALSE iff the cache hit but no academic-code "
                    "match (incl. empty taxonomy list). NULL iff the cache "
                    "missed (obfuscated-NPI cohort or unknown real NPI) — "
                    "downstream consumers must distinguish 'not derivable' "
                    "from 'not academic'."
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
            # Issue #156 item 1: priority_tier
            {
                "feature": "priority_tier",
                "type": "int (1=high, 5=low)",
                "source_table": (
                    "medication (NDC + HCPCS + brand/generic) + procedure "
                    "(HCPCS J2357/J0517 for buy-and-bill admin) via NPI + "
                    "demographics.zipcode_5"
                ),
                "lookback_window": "per-patient (index - 365d, index]",
                "notes": (
                    "Issue #156 item 1 — rolling 12-month CSU biologic "
                    "TRx aggregated per (NPI, ZIP3) → equal-frequency "
                    "decile within each ZIP3 → mapped to 5-tier scale "
                    "(decile 10=tier 1, 8-9=tier 2, 4-7=tier 3, 2-3=tier "
                    "4, 1=tier 5). HCPs with TRx=0 in window or no "
                    "resolvable ZIP3 default to tier 5 (kept in scoreable "
                    "pool, not excluded). Tie-break: descending NDC/HCPCS-"
                    "distinct count, then ascending alphabetical NPI. "
                    "ZIP3 (not ZIP5) chosen because ZIP5 has too few "
                    "HCPs per bin for stable decile assignment. NPI ZIP3 "
                    "is the modal ZIP3 across the HCP's treated patients "
                    "in the cohort. Per-patient temporal gating (codex "
                    "PR-2 pass-1 MEDIUM-1) ensures no post-index fills "
                    "leak in. Procedure-side HCPCS admins (codex PR-2 "
                    "pass-1 HIGH-1) are counted so office buy-and-bill "
                    "is not undercounted."
                ),
            },
            {
                "feature": "decile",
                "type": "int [1, 10] or None (tier-5 defaults)",
                "source_table": "derived from priority_tier ranking",
                "lookback_window": "per-patient (index - 365d, index]",
                "notes": (
                    "Issue #156 item 1 / codex PR-2 pass-1 LOW-1 — the "
                    "underlying within-ZIP3 decile that maps to "
                    "priority_tier via PRIORITY_TIER_DECILE_MAP. Exposed "
                    "so the tier derivation is auditable from the "
                    "artifact. None when the HCP has TRx=0 or no "
                    "resolvable ZIP3 (those default to tier 5)."
                ),
            },
            # Issue #156 item 2: influence_network_size + peer_influence_score
            {
                "feature": "influence_network_size",
                "type": "int (degree)",
                "source_table": "medication.npi ∪ procedure.npi (CLAIMS-DERIVED PROXY)",
                "lookback_window": "per-patient (index - 180d, index]",
                "notes": (
                    "Issue #156 item 2 — degree (neighbor count) in the "
                    "shared-patient HCP-HCP graph. For each kept patient, "
                    "the set of treating HCPs (across medication ∪ "
                    "procedure) within the pre-index 180d lookback forms "
                    "a clique; edge weight = number of shared patients "
                    "across two NPIs. Per-patient temporal gating (codex "
                    "PR-2 pass-1 MEDIUM-2) ensures no post-index HCP "
                    "contacts leak into the influence proxy. Graph "
                    "computed via networkx.Graph. CLAIMS-DERIVED PROXY "
                    "for KOL influence — canonical KOL data requires "
                    "external commercial sources (Definitive Healthcare, "
                    "HCS Spectrum) or PubMed co-authorship, which is "
                    "explicitly out of scope (issue #156)."
                ),
            },
            {
                "feature": "peer_influence_score",
                "type": "float DECIMAL(3,2) ∈ [0.00, 9.99]",
                "source_table": "medication.npi ∪ procedure.npi (CLAIMS-DERIVED PROXY)",
                "lookback_window": "per-patient (index - 180d, index]",
                "notes": (
                    "Issue #156 item 2 — weighted eigenvector_centrality "
                    "computed on the shared-patient graph (see "
                    "influence_network_size). Raw centrality ∈ [0, 1] is "
                    "scaled by PEER_INFLUENCE_SCALE=9.99 then clamped to "
                    "fit DECIMAL(3,2). Computed per-component to avoid "
                    "PowerIterationFailedConvergence on disconnected "
                    "subgraphs; singletons (no edges) get 0.0. CLAIMS-"
                    "DERIVED PROXY for KOL influence — see "
                    "influence_network_size notes for canonical-data "
                    "scope deferral."
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

        # Issue #157 PR C (Sub-PR-A) — treatment_response + outcome_indicator
        # are emitted on the biologic-fill row in treatment_events for the
        # discontinuation cohort only. Document the proxy here so downstream
        # consumers can find the derivation rules.
        if cohort == "discontinuation":
            entries.append(
                {
                    "feature": "treatment_response",
                    "type": "enum{controlled,inadequate,uncontrolled,refractory,discontinued}",
                    "source_table": "treatment_events (biologic-fill row)",
                    "lookback_window": "[init_date, init_date+180]",
                    "notes": (
                        "Issue #157 PR C / Sub-PR-A — CSU claim-pattern "
                        "response proxy. NULL outside biologic-fill universe "
                        "or when >=60d coverage / >=90d follow-up "
                        "pre-conditions are unmet. First-match-wins rule "
                        "order: discontinued > refractory > inadequate > "
                        "controlled. `uncontrolled` reserved for non-Optum "
                        "UAS7-anchored cohorts."
                    ),
                }
            )
            entries.append(
                {
                    "feature": "outcome_indicator",
                    "type": "enum{improved,stable,worsened}",
                    "source_table": "treatment_events (biologic-fill row)",
                    "lookback_window": "[init_date, init_date+180]",
                    "notes": (
                        "Issue #157 PR C / Sub-PR-A — mapped from "
                        "treatment_response: controlled→improved, "
                        "{inadequate,refractory}→worsened, "
                        "discontinued→worsened if no subsequent biologic "
                        "fill outside window else stable."
                    ),
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
    # Issue #156 item 3
    parser.add_argument(
        "--comorbidity-method",
        choices=list(COMORBIDITY_METHODS_ALLOWED),
        default=COMORBIDITY_METHOD_DEFAULT,
        help=(
            "Comorbidity scoring algorithm. 'quan' (default) uses Quan (2005) "
            "ICD-10 mappings with classical Charlson weights + van Walraven "
            "(2009) Elixhauser weights. 'approx' uses the legacy chapter-count "
            "Elixhauser proxy + 5-category Charlson proxy (retained for parity)."
        ),
    )
    # Issue #156 item 5 — soft enrollment filter (opt-in).
    parser.add_argument(
        "--soft-enrollment-filter",
        action="store_true",
        help=(
            "Keep partial-enrollment patients in the cohort (DQS gates "
            "downstream). When OMITTED (default), the historical hard filter "
            "`continuous_enrollment == 1` + strict pre/post-day enrollment "
            "window is preserved — CSU cohort behavior unchanged."
        ),
    )
    parser.add_argument(
        "--min-data-quality-score",
        type=float,
        default=None,
        help=(
            "Soft data-quality threshold for downstream model-training "
            "filtering. Patients below this DQS are LOGGED in attrition "
            "(not dropped). Only meaningful when --soft-enrollment-filter "
            f"is set. Default: {DEFAULT_MIN_DATA_QUALITY_SCORE}."
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
        comorbidity_method=args.comorbidity_method,
        soft_enrollment_filter=args.soft_enrollment_filter,
        min_data_quality_score=args.min_data_quality_score,
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
