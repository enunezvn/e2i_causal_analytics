"""Cheapest disproof (owner ask, 2026-09-22): is the mart's ``persistent_at_180d``
definition dosing-interval sensitive per brand?  Replicates
``scripts/convert_optum_mart.py::select_persistence_cohort`` / ``select_discontinuation_cohort``
from the raw enriched drop and sweeps the gap threshold and a coverage-end grace.
Reads the raw drop only; writes nothing.  Run: E2I_DATA_ROOT=<checkout-with-data> python - < run_sweep.py
"""
import os
import numpy as np
import pandas as pd

R = os.environ.get("E2I_DATA_ROOT", ".")
cols = ["patid","index_biologic_brand","treatment_start_date","last_observed_date","last_coverage_end",
        "max_internal_gap_days","terminal_gap_days","claim_record_count","covered_days"]
e = pd.read_parquet(R + "/data/rwd/Optum_Parquet/Optum_enriched.parquet", columns=cols)
ts = pd.to_datetime(e["treatment_start_date"], errors="coerce")
d = e[(e["index_biologic_brand"] != "no_treatment") & ts.notna()].copy()
d = d[d["claim_record_count"].fillna(0) >= 2]
ts = pd.to_datetime(d["treatment_start_date"]); lo = pd.to_datetime(d["last_observed_date"]); lce = pd.to_datetime(d["last_coverage_end"], errors="coerce")
d = d[((lo - ts).dt.days >= 180) & lce.notna()].copy()
ts = pd.to_datetime(d["treatment_start_date"]); lce = pd.to_datetime(d["last_coverage_end"])
d["cov_to_end"] = (lce - ts).dt.days; d["gap"] = d["max_internal_gap_days"].fillna(0); d["term"] = d["terminal_gap_days"].fillna(0)
b = d["index_biologic_brand"]
print("replicated cohort n=", len(d), b.value_counts().to_dict())
def rate(mask):
    r = mask.groupby(b).mean().round(3).to_dict(); return {**r, "gap_pp": round(100 * (r.get("XOLAIR", 0) - r.get("DUPIXENT", 0)), 1)}
print("\n-- persistence = cov_to_end>=180 & max_internal_gap<=G --")
for G in (30, 45, 60, 90, 120, 10**6): print(f"G={G:>7}:", rate((d.cov_to_end >= 180) & (d.gap <= G)))
print("\n-- decomposition at G=60 --")
print("covered through day 180 (cov_to_end>=180):", rate(d.cov_to_end >= 180))
print("no internal gap >60:", rate(d.gap <= 60))
print("\n-- persistence with GRACE on coverage end: cov_to_end>=180-grace & gap<=60 --")
for grace in (0, 14, 28, 45, 60): print(f"grace={grace:>2}:", rate((d.cov_to_end >= 180 - grace) & (d.gap <= 60)))
print("\n-- discontinuation = cov_to_end<180 & (gap>=D | term>=D) --")
for D in (45, 60, 90, 120): print(f"D={D:>3}:", rate((d.cov_to_end < 180) & ((d.gap >= D) | (d.term >= D))))
print("\n-- cov_to_end distribution by brand (days) --")
print(d.groupby(b)["cov_to_end"].quantile([.1, .25, .5, .75, .9]).unstack().round(0).to_dict(orient="index"))
print("share with cov_to_end in [150,180):", rate((d.cov_to_end >= 150) & (d.cov_to_end < 180)))
