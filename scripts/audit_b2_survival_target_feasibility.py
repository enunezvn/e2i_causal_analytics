"""v5 Gate B2 — survival target feasibility audit.

Reproducibility anchor for the cohort audit table in
``docs/specs/v5_b2_survival_modeling_prespec_2026-05-12.md`` §2.

For each cohort, reports:
- n_patients
- n_positives (treatment_initiated==1)
- n_post_index_rx_events (event_type=='prescription' AND days_from_diagnosis>=0)
- n_unique_patients_with_post_index_rx
- time-to-event derivability verdict: REAL | DATA-FIDELITY-BOUND
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _load_optum_initiation(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    pj_path = repo_root / "data" / "rwd" / "optum" / "initiation" / "e2i_ml_v3_patient_journeys.parquet"
    ev_path = repo_root / "data" / "rwd" / "optum" / "initiation" / "e2i_ml_v3_treatment_events.parquet"
    if not pj_path.exists() or not ev_path.exists():
        return None
    return pd.read_parquet(pj_path), pd.read_parquet(ev_path)


def _load_csu(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    pj_path = repo_root / "data" / "rwd" / "csu" / "e2i_ml_v3_patient_journeys.json"
    ev_path = repo_root / "data" / "rwd" / "csu" / "e2i_ml_v3_treatment_events.json"
    if not pj_path.exists() or not ev_path.exists():
        return None
    with open(pj_path) as f:
        pj = pd.DataFrame(json.load(f))
    with open(ev_path) as f:
        ev = pd.DataFrame(json.load(f))
    return pj, ev


def _audit_cohort(label: str, pj: pd.DataFrame, ev: pd.DataFrame) -> Dict[str, Any]:
    n = len(pj)
    n_pos = int(pj["treatment_initiated"].sum())
    rx = ev[ev["event_type"] == "prescription"].copy()
    # Coerce days_from_diagnosis to numeric (some loaders store as object/string).
    rx["days_from_diagnosis"] = pd.to_numeric(rx["days_from_diagnosis"], errors="coerce")
    post_rx = rx[rx["days_from_diagnosis"] >= 0]
    n_post_rx = int(len(post_rx))
    n_unique_pids = int(post_rx["patient_id"].nunique()) if "patient_id" in post_rx.columns else 0

    # M5 codex pass-1: 10% threshold rationale. With <10% coverage,
    # the survival time imputation for unmatched positives (fallback
    # to journey_duration_days censoring time) dominates the
    # event-time signal, since the survival regression loss is
    # ranking-based on event-time order. At >=10%, at least ~175 of
    # CSU's 1743 positives have usable rx event-time data; at <10%
    # (e.g., Optum's 0%), the survival framing reduces to binary at
    # the administrative censoring horizon and is documented as
    # DATA-FIDELITY-BOUND in pre-spec §2.
    pids_pos = set(pj.loc[pj["treatment_initiated"] == 1, "patient_id"].unique())
    pids_with_post_rx = set(post_rx["patient_id"].unique())
    n_pos_with_rx = len(pids_pos & pids_with_post_rx)
    coverage = (n_pos_with_rx / n_pos) if n_pos > 0 else 0.0
    coverage_threshold = 0.10
    verdict = "REAL" if coverage >= coverage_threshold else "DATA-FIDELITY-BOUND"

    return {
        "cohort": label,
        "n_patients": n,
        "n_positives": n_pos,
        "n_post_index_rx_events": n_post_rx,
        "n_unique_patients_with_post_index_rx": n_unique_pids,
        "n_positives_with_post_index_rx": n_pos_with_rx,
        "post_index_rx_coverage_of_positives": coverage,
        "coverage_threshold": coverage_threshold,
        "verdict": verdict,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    report: Dict[str, Any] = {}

    optum = _load_optum_initiation(repo_root)
    if optum is None:
        report["optum_initiation"] = {"skipped": True, "reason": "data files missing"}
    else:
        pj, ev = optum
        report["optum_initiation"] = _audit_cohort("optum_initiation", pj, ev)

    csu = _load_csu(repo_root)
    if csu is None:
        report["csu"] = {"skipped": True, "reason": "data files missing"}
    else:
        pj, ev = csu
        report["csu"] = _audit_cohort("csu", pj, ev)

    output_path = repo_root / "docs" / "calibration" / "b2_survival_target_audit_20260512.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Audit written to {output_path}")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
