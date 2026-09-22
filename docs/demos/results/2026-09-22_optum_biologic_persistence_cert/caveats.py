#!/usr/bin/env python3
"""Lane A cert caveats (plan Task 12 Step 4), recorded as DATA a reviewer must weigh:

1. `treatment_response` is brand-coupled (counts by arm) — it is NOT used as an outcome.
2. `max_consecutive_biologic_coverage_days` quartiles by arm — the persistence definition's
   60-day gap threshold is dosing-interval sensitive (14-day Dupixent fills vs 28–45-day
   Xolair fills), which is why the shipped `persistent_at_180d` is a days-supply artefact.

Both fields live on the RAW drop (`Optum_enriched.parquet`), not on the causal export; the
raw drop is restricted to the exported cohort's 15,209 patients (`patient_id` = "PAT_" + patid).
Writes caveats.json next to this script.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("/home/enunez/Projects/e2i_causal_analytics")
RAW = ROOT / "data/rwd/Optum_Parquet/Optum_enriched.parquet"
COHORT = ROOT / "data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet"
OUT = Path(__file__).resolve().parent / "caveats.json"


def main() -> None:
    cohort = pd.read_parquet(COHORT, columns=["patient_id", "index_biologic_brand", "treatment_dupixent"])
    raw = pd.read_parquet(RAW, columns=["patid", "index_biologic_brand", "treatment_response",
                                        "max_consecutive_biologic_coverage_days"])
    raw["patient_id"] = "PAT_" + raw["patid"].astype(str)
    joined = cohort.merge(raw.drop(columns=["index_biologic_brand"]), on="patient_id", how="left", validate="1:1")
    assert len(joined) == len(cohort) == 15209, len(joined)
    unmatched = int(joined["treatment_response"].isna().sum())

    resp = (joined.groupby("index_biologic_brand")["treatment_response"]
            .value_counts(dropna=False).unstack(fill_value=0))
    resp_counts = {arm: {str(k): int(v) for k, v in row.items()} for arm, row in resp.iterrows()}

    q = (joined.groupby("index_biologic_brand")["max_consecutive_biologic_coverage_days"]
         .quantile([0.25, 0.5, 0.75]).unstack())
    quartiles = {arm: {"q25": float(r[0.25]), "q50": float(r[0.5]), "q75": float(r[0.75])} for arm, r in q.iterrows()}
    n_arm = joined["index_biologic_brand"].value_counts().to_dict()

    out = {
        "source": {"raw_drop": str(RAW), "cohort_export": str(COHORT), "join_key": "patient_id = 'PAT_' + patid"},
        "n_cohort": int(len(joined)), "n_by_arm": {k: int(v) for k, v in n_arm.items()},
        "raw_rows_unmatched_to_cohort": unmatched,
        "treatment_response_counts_by_arm": resp_counts,
        "treatment_response_note": "brand-coupled; NOT used as an outcome or covariate in the cert runs",
        "max_consecutive_biologic_coverage_days_quartiles_by_arm": quartiles,
        "coverage_note": "the shipped persistent_at_180d (60-day gap) is days-supply sensitive; "
                         "read it with the grace sweep (persistence_definition_disproof), not as an effect",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
