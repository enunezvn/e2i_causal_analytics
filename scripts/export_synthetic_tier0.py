#!/usr/bin/env python
"""Export the synthetic parquet snapshot into per-cohort tier0-contract inputs.

Writes ``<src>/tier0/<cohort>/e2i_ml_v3_patient_journeys.parquet`` +
``e2i_ml_v3_split_registry.json`` for all four cohorts (initiation,
discontinuation, persistence on the patient grain; hcp_adoption on the HCP
grain), the shape ``scripts/run_tier0_test.py::load_rwd_data`` consumes via
``--data-dir``. Tier0 proves the synthetic labels are modelable (val_AUC gate)
and its trained artifacts feed the tier1-5 agent stages.

Contract decisions (mirrors the Optum mart cohort dirs):
- Brand-filtered (default Remibrutinib — the CSU set); all-null off-brand
  indication covariates are dropped after the filter.
- GROUND-TRUTH / LEAK columns are excluded from every cohort frame:
  ``propensity_score`` and ``treatment_effect_estimate`` are the DGP answer key;
  ``days_to_treatment`` is populated only for initiators (outcome-derived);
  the *other* cohort's outcome is dropped per cell (``persistent_180d`` is the
  exact complement of ``discontinued_180d`` — a perfect label leak).
- ``data_split`` is REASSIGNED (stratified random 60/20/10/10, seeded): the
  generator's chronological split is scrambled by the --anchor-to-now date
  remap (observed: holdout 61% / train 25%), unusable for tier0 training.
- ``is_synthetic`` is kept on every row (provenance is non-negotiable).

Usage:
    python scripts/export_synthetic_tier0.py --src data/rwd/synthetic_CSU
    python scripts/export_synthetic_tier0.py --src data/rwd/synthetic_CSU --brand Kisqali
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COHORT_TARGETS: dict[str, str] = {
    "initiation": "treatment_initiated",
    "discontinuation": "discontinued_180d",
    "persistence": "persistent_180d",
    "hcp_adoption": "adopted_target_brand",  # mirrors the Optum mart target name
}

# DGP answer-key / outcome-derived columns — never tier0 features.
_LEAK_COLS = ["propensity_score", "treatment_effect_estimate", "days_to_treatment"]

# Brand-eligibility covariates (Shard 04 M5, patient_generator.py:225-235) are
# populated for EVERY patient regardless of brand; keep only the exported
# brand's indication panel so the tier0 frame stays clinically coherent.
_INDICATION_COVARIATES: dict[str, list[str]] = {
    "Remibrutinib": ["urticaria_severity_uas7", "prior_antihistamine_therapy"],
    "Kisqali": ["hr_status", "her2_status", "disease_stage", "ecog_performance_status"],
    "Fabhalta": ["ldh_ratio", "complement_inhibitor_status", "proteinuria_g_day", "egfr"],
}

# All patient outcome columns; each patient cell keeps ONLY its own target.
_PATIENT_OUTCOMES = ["treatment_initiated", "discontinued_180d", "persistent_180d"]

# #44 (2026-07-21): mirrors the seed quota in generators/base.py::_assign_splits
# (test 0.15→0.10, holdout 0.05→0.10 — goldstd holdout enlargement).
_SPLIT_RATIOS = {"train": 0.60, "validation": 0.20, "test": 0.10, "holdout": 0.10}


def _assign_splits(df: pd.DataFrame, target: str, seed: int) -> pd.Series:
    """Stratified random 60/20/10/10 split assignment (one row per unit)."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index)
    for _, idx in df.groupby(df[target]).groups.items():
        idx = rng.permutation(np.asarray(idx))
        bounds = np.cumsum([_SPLIT_RATIOS[s] for s in ("train", "validation", "test")])
        cuts = (bounds * len(idx)).astype(int)
        split.loc[idx[cuts[0] : cuts[1]]] = "validation"
        split.loc[idx[cuts[1] : cuts[2]]] = "test"
        split.loc[idx[cuts[2] :]] = "holdout"
    return split


def _ensure_contract_cols(df: pd.DataFrame) -> pd.DataFrame:
    """v3-contract columns the data_preparer's GE suite auto-detect requires.

    ge_validator.py:92-100 routes to the patient-level ``ml_patients`` suite only
    when BOTH ``patient_journey_id`` and ``discontinuation_flag`` exist (and no
    ``event_type``); otherwise it validates against the event-level
    ``patient_journeys`` suite and the QC gate blocks on missing event columns.
    Mirror the mart converters: ``discontinuation_flag`` is a constant-0
    placeholder (mart initiation ships null_rate 0.0, values {0}) — zero
    variance, so it can never leak; the HCP grain aliases ``hcp_id`` into the
    patient id columns exactly as ``convert_optum_hcp_adoption.py`` does.
    """
    if "patient_journey_id" not in df.columns:
        df["patient_journey_id"] = df["hcp_id"]
    if "patient_id" not in df.columns:
        df["patient_id"] = df["hcp_id"]
    df["discontinuation_flag"] = 0
    # The runner derives step-3 inclusion criteria FROM data_quality_score;
    # absent, CohortConstructor gets no criteria and fails closed (CC_001
    # INVALID_CONFIG -> 0 eligible). Synthetic rows are complete by
    # construction, so the score is uniformly 1.0 (mart frames carry the
    # converter-computed equivalent).
    df["data_quality_score"] = 1.0
    return df


def _write_cell(out_dir: Path, cohort: str, df: pd.DataFrame, brand: str, seed: int) -> None:
    target = COHORT_TARGETS[cohort]
    if df[target].nunique(dropna=True) < 2:
        raise SystemExit(f"{cohort}: target {target!r} is single-class — not modelable")
    df = _ensure_contract_cols(df.copy())
    df["data_split"] = _assign_splits(df, target, seed)
    cell = out_dir / cohort
    cell.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cell / "e2i_ml_v3_patient_journeys.parquet", index=False)
    registry = [
        {
            "split_config_id": f"synthetic_{brand.lower()}_{cohort}_v1",
            "config_name": f"synthetic_{brand.lower()}_{cohort}",
            "config_version": "v1",
            **{f"{k}_ratio": v for k, v in _SPLIT_RATIOS.items()},
            "split_strategy": "stratified_random",
            "split_seed": seed,
            "patient_level_isolation": True,  # one row per unit by construction
            "is_active": True,
            "created_at": datetime.now().isoformat(),
        }
    ]
    (cell / "e2i_ml_v3_split_registry.json").write_text(json.dumps(registry, indent=2))
    pos = int(df[target].sum())
    logger.info(
        "  tier0/%s: %d rows, %d positive (%.1f%%), target=%s",
        cohort,
        len(df),
        pos,
        100 * pos / len(df),
        target,
    )


def export_tier0(src, brand: str = "Remibrutinib", seed: int = 42) -> list[str]:
    """Write tier0/<cohort>/ contract dirs for all 4 cohorts. Returns cohorts written."""
    src = Path(src)
    out = src / "tier0"
    pj = pd.read_parquet(src / "patient_journeys.parquet")
    pj = pj[pj["brand"] == brand].copy()
    pj = pj.dropna(axis="columns", how="all")
    off_brand = [
        c
        for b, cols in _INDICATION_COVARIATES.items()
        if b != brand
        for c in cols
        if c in pj.columns and c not in _INDICATION_COVARIATES.get(brand, [])
    ]
    pj = pj.drop(columns=[c for c in _LEAK_COLS + off_brand if c in pj.columns])

    written: list[str] = []
    for cohort in ("initiation", "discontinuation", "persistence"):
        target = COHORT_TARGETS[cohort]
        sub = pj
        if cohort != "initiation":
            sub = pj[pj["treatment_initiated"] == 1]
        drop = [c for c in _PATIENT_OUTCOMES if c != target and c in sub.columns]
        sub = sub.drop(columns=drop).dropna(subset=[target])
        sub[target] = sub[target].astype(int)
        _write_cell(out, cohort, sub, brand, seed)
        written.append(cohort)

    hcp = pd.read_parquet(src / "hcp_profiles.parquet")
    hcp = hcp[hcp["brand"] == brand].dropna(subset=["adoption_category"]).copy()
    hcp["adopted_target_brand"] = (hcp["adoption_category"] == "ADOPTER").astype(int)
    hcp = hcp.drop(columns=["adoption_category"])  # string twin of the target
    _write_cell(out, "hcp_adoption", hcp, brand, seed)
    written.append("hcp_adoption")

    logger.info("Wrote %d tier0 cohort dirs to %s", len(written), out)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/rwd/synthetic_CSU", help="snapshot dir")
    parser.add_argument("--brand", default="Remibrutinib")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    export_tier0(args.src, brand=args.brand, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
