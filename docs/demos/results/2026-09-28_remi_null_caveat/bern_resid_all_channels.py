"""Bernoulli-residual contrast (adopted - sigmoid(logit)) for every channel tbin x brand, adjusted
for the estimator's X+W. The residual is the final coin flip only; by construction it is
independent of every covariate, so these z's should look N(0,1). Reads the decompose cache."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("REMI_NO_IMPORT", "1")
from remi_null_probe import CACHE, BRANDS, xw_matrix, ols_coef
from src.data.per_hcp_cohort_collapse import CHANNEL_COLUMNS
rows = []
for b in BRANDS:
    f = pd.read_parquet(f"{CACHE}/analysis_{b}.parquet")
    base = xw_matrix(f)
    for c in CHANNEL_COLUMNS:
        t = f[f"tb_{c}"].to_numpy()
        for term in ("bern_resid", "p_true", "adopted"):
            est, se = ols_coef(f[term].to_numpy(float), t, base)
            rows.append({"brand": b, "channel": c, "term": term, "adj_diff": est, "z": est / se})
d = pd.DataFrame(rows)
d.to_csv("bern_resid_all_channels.csv", index=False)
p = d.pivot_table(index=["brand", "channel"], columns="term", values=["adj_diff", "z"]).round(4)
print(p.to_string())
z = d[d.term == "bern_resid"]["z"]
print(f"\nbern_resid z over 24 cells: mean {z.mean():+.3f} sd {z.std():.3f} max|z| {z.abs().max():.2f}")
