#!/usr/bin/env python3
"""Leakage-safe HCP-feature enrichment of converter cohorts using Gap features.

Adds *provider-level* commercial covariates — targeting decile/priority and KOL
influence — from the Gap-features HCP tables onto the converter's leakage-safe
``e2i_ml_v3_patient_journeys``. Each patient is linked to their treating
prescriber(s) via the *raw* Optum ``medication.npi`` (the de-identified provider
IDs — these match the Gap HCP tables 112/112; the converter's own
``hcp_profiles.npi`` is synthetic and does NOT match, so we link off raw claims).

Why ONLY HCP features (see .claude/plans/tier0-tier15-testing-readiness-… and the
investigation notes):
  - The patient-level Gap clinical tables (``patient_risk_scores`` etc.) are
    anchored at a DIFFERENT index_date (median +109d later than the converter
    index). Transplanting their comorbidity/risk features would inject
    POST-INDEX information = target leakage, and they duplicate the converter's
    own at-index comorbidities. They are DELIBERATELY EXCLUDED.
  - ``Patient_journey`` / ``Treatment_response`` are post-index outcome/target
    data (``discontinued_flag`` etc.) — excluded as features.

Leakage discipline for the HCP join:
  - Only prescribers with ``medication_date <= the patient's converter
    index_date`` are used, so the provider relationship predates the outcome
    window (filter is per-patient, against the *converter* index — not the
    vendor ``indexdt``).
  - The attached attributes are provider COMMERCIAL scores (targeting decile,
    KOL score), not patient outcomes. TEMPORAL-CURRENCY CAVEAT: the Gap HCP
    scores are "current" rolling-window scores, so a provider's score may
    reflect activity after the patient's index. This is acceptable for Tier0
    *harness testing* (exercising the pipeline on enriched real data), and is
    documented here so the green isn't over-read as a deployable clinical model.

Coverage is data-dependent: the treatment-naive ``initiation`` cohort has ~1-3%
HCP linkage (few biologic prescribers) so its HCP columns are ~all-null (Tier0
QC will skip them); ``discontinuation``/``persistence`` (biologic initiators)
reach ~47% under the leakage-safe filter.

Usage:
    python scripts/enrich_cohort_with_hcp_features.py --all
    python scripts/enrich_cohort_with_hcp_features.py \
        --cohort-dir data/rwd/optum/discontinuation \
        --out-dir data/rwd/optum_gap_enriched/discontinuation
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPTUM_DIR = PROJECT_ROOT / "data" / "rwd" / "Optum_Parquet"
DEFAULT_GAP_DIR = PROJECT_ROOT / "data" / "rwd" / "Gap features in parquet format"
DEFAULT_COHORT_ROOT = PROJECT_ROOT / "data" / "rwd" / "optum"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "rwd" / "optum_gap_enriched"
COHORTS = ("initiation", "discontinuation", "persistence")

PATIENT_JOURNEYS_FILE = "e2i_ml_v3_patient_journeys.parquet"

# Output columns added by the enrichment (provider-level, leakage-safe).
HCP_FEATURE_COLUMNS = [
    "treating_hcp_match_count",
    "treating_hcp_targeting_decile_max",
    "treating_hcp_priority_tier_best",
    "treating_hcp_is_specialist_any",
    "treating_hcp_kol_score_max",
    "treating_hcp_kol_score_100pt_max",
    "treating_hcp_influence_network_size_max",
    "treating_hcp_kol_category_top",
]


def _patid_from_patient_id(s: pd.Series) -> pd.Series:
    """Converter ``patient_id`` is ``PAT_<patid>``; recover the int patid."""
    return s.astype(str).str.removeprefix("PAT_").astype("int64")


def _build_hcp_attr_table(targeting: pd.DataFrame, kol: pd.DataFrame) -> pd.DataFrame:
    """One row per provider ``npi`` with the attributes we attach.

    Both source tables can carry duplicate npis; we keep the strongest signal
    (max decile / specialist-ever; the max-``kol_score`` row for KOL fields).
    """
    tgt = targeting.copy()
    tgt["npi"] = tgt["npi"].astype(str)
    tgt_agg = (
        tgt.groupby("npi")
        .agg(
            decile=("decile", "max"),
            priority_tier=("priority_tier", "min"),  # tier 1 = highest priority
            is_specialist=("is_specialist_hcp", "max"),
        )
        .reset_index()
    )

    k = kol.copy()
    k["npi"] = k["npi"].astype(str)
    # Keep the row with the max kol_score per npi (so kol_category matches it).
    k = k.sort_values("kol_score", ascending=False).drop_duplicates("npi", keep="first")
    kol_agg = k[
        ["npi", "kol_score", "kol_score_100pt", "influence_network_size", "kol_category"]
    ].reset_index(drop=True)

    return tgt_agg.merge(kol_agg, on="npi", how="outer")


def build_patient_hcp_features(
    pj: pd.DataFrame,
    medication: pd.DataFrame,
    targeting: pd.DataFrame,
    kol: pd.DataFrame,
) -> pd.DataFrame:
    """Return ``pj`` with HCP_FEATURE_COLUMNS appended (rows + order preserved).

    Leakage-safe: a prescriber contributes only if
    ``medication_date <= the patient's index_date``.
    """
    out = pj.copy()
    out["_patid"] = _patid_from_patient_id(out["patient_id"])
    out["_idx"] = pd.to_datetime(out["index_date"], errors="coerce")

    hcp_attr = _build_hcp_attr_table(targeting, kol)

    med = medication[["patid", "npi", "medication_date"]].copy()
    med["npi"] = med["npi"].astype(str)
    med["medication_date"] = pd.to_datetime(med["medication_date"], errors="coerce")

    # Attach each patient's index, then keep only PRE-INDEX prescriptions.
    linked = med.merge(
        out[["_patid", "_idx"]].rename(columns={"_patid": "patid"}),
        on="patid",
        how="inner",
    )
    linked = linked[linked["medication_date"] <= linked["_idx"]]
    # Join provider attributes (inner → only providers present in the HCP tables).
    linked = linked.merge(hcp_attr, on="npi", how="inner")

    if linked.empty:
        agg = pd.DataFrame(columns=["patid", *HCP_FEATURE_COLUMNS])
    else:
        # kol_category of the patient's max-kol provider.
        top_cat = (
            linked.sort_values("kol_score", ascending=False)
            .drop_duplicates("patid", keep="first")[["patid", "kol_category"]]
            .rename(columns={"kol_category": "treating_hcp_kol_category_top"})
        )
        grouped = linked.groupby("patid")
        agg = pd.DataFrame(
            {
                "treating_hcp_match_count": grouped["npi"].nunique(),
                "treating_hcp_targeting_decile_max": grouped["decile"].max(),
                "treating_hcp_priority_tier_best": grouped["priority_tier"].min(),
                "treating_hcp_is_specialist_any": grouped["is_specialist"].max(),
                "treating_hcp_kol_score_max": grouped["kol_score"].max(),
                "treating_hcp_kol_score_100pt_max": grouped["kol_score_100pt"].max(),
                "treating_hcp_influence_network_size_max": grouped["influence_network_size"].max(),
            }
        ).reset_index()
        agg = agg.merge(top_cat, on="patid", how="left")

    merged = out.merge(agg, left_on="_patid", right_on="patid", how="left")
    # Patients with no pre-index matched provider → explicit 0 match count.
    merged["treating_hcp_match_count"] = (
        merged["treating_hcp_match_count"].fillna(0).astype("int64")
    )
    merged = merged.drop(columns=["_patid", "_idx", "patid"], errors="ignore")
    # Preserve original row order/count.
    assert len(merged) == len(pj), "row count changed during enrichment"
    return merged[[*pj.columns, *HCP_FEATURE_COLUMNS]]


def enrich_cohort(cohort_dir: Path, optum_dir: Path, gap_dir: Path, out_dir: Path) -> dict:
    """Enrich one cohort dir; copy the dir and overwrite patient_journeys."""
    pj_path = cohort_dir / PATIENT_JOURNEYS_FILE
    if not pj_path.exists():
        raise FileNotFoundError(f"no {PATIENT_JOURNEYS_FILE} in {cohort_dir}")

    pj = pd.read_parquet(pj_path)
    medication = pd.read_parquet(optum_dir / "medication.parquet")
    targeting = pd.read_parquet(gap_dir / "hcp_targeting_tier.parquet")
    # Pre-filter the huge KOL table to the cohort's prescribers before loading all.
    cohort_npis = set(medication["npi"].dropna().astype(str))
    kol = pd.read_parquet(gap_dir / "KOL_influence.parquet")
    kol = kol[kol["npi"].astype(str).isin(cohort_npis)]

    enriched = build_patient_hcp_features(pj, medication, targeting, kol)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy sibling cohort files (treatment_events, hcp_profiles, split_registry,
    # data_dictionary, attrition) so the enriched dir is a complete --data-dir.
    for f in cohort_dir.iterdir():
        if f.is_file() and f.name != PATIENT_JOURNEYS_FILE:
            shutil.copy2(f, out_dir / f.name)
    enriched.to_parquet(out_dir / PATIENT_JOURNEYS_FILE, index=False)

    linked = int((enriched["treating_hcp_match_count"] > 0).sum())
    n = len(enriched)
    stats = {
        "cohort_dir": str(cohort_dir),
        "out_dir": str(out_dir),
        "patients": n,
        "hcp_linked": linked,
        "hcp_linked_pct": round(100 * linked / n, 1) if n else 0.0,
        "added_columns": HCP_FEATURE_COLUMNS,
    }
    logger.info(
        "Enriched %s → %s: %d/%d patients HCP-linked (%.1f%%)",
        cohort_dir.name,
        out_dir,
        linked,
        n,
        stats["hcp_linked_pct"],
    )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="process all 3 cohorts under --cohort-root")
    ap.add_argument("--cohort-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--optum-dir", type=Path, default=DEFAULT_OPTUM_DIR)
    ap.add_argument("--gap-dir", type=Path, default=DEFAULT_GAP_DIR)
    args = ap.parse_args()

    jobs: list[tuple[Path, Path]] = []
    if args.all:
        jobs = [(args.cohort_root / c, args.out_root / c) for c in COHORTS]
    elif args.cohort_dir and args.out_dir:
        jobs = [(args.cohort_dir, args.out_dir)]
    else:
        ap.error("provide --all, or both --cohort-dir and --out-dir")

    results = []
    for cohort_dir, out_dir in jobs:
        results.append(enrich_cohort(cohort_dir, args.optum_dir, args.gap_dir, out_dir))
    logger.info("HCP enrichment complete: %s", {r["out_dir"]: r["hcp_linked_pct"] for r in results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
