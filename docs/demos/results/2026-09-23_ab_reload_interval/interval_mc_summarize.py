"""Summarise interval_mc.csv: per (brand, channel) the empirical SD of each point estimate
across DGP seeds vs each method's mean reported SE (ratio ~1 = calibrated), 95 % coverage of
the planted RD, and the share of seeds whose interval excludes 0 (power; for the null channel
this is the false-positive rate)."""
import sys

import numpy as np
import pandas as pd

Z = 1.959964
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "interval_mc.csv")
rows = []
for (brand, ch), g in df.groupby(["brand", "channel"], sort=False):
    planted = g["planted"].iloc[0]
    rec = {"brand": brand, "channel": ch, "planted": planted, "seeds": len(g)}
    # forest point (mean CATE) + shipped ate_interval
    rec["forest_bias"] = g["point_forest"].mean() - planted
    rec["forest_empSD"] = g["point_forest"].std(ddof=1)
    rec["ai_SE"] = g["ai_se"].mean()
    rec["ai_cov"] = ((g["ai_lo"] <= planted) & (planted <= g["ai_hi"])).mean()
    rec["ai_excl0"] = ((g["ai_lo"] > 0) | (g["ai_hi"] < 0)).mean()
    # forest DR ATE + its SE
    rec["dr_bias"] = g["dr_ate"].mean() - planted
    rec["dr_empSD"] = g["dr_ate"].std(ddof=1)
    rec["dr_SE"] = g["dr_se"].mean()
    lo, hi = g["dr_ate"] - Z * g["dr_se"], g["dr_ate"] + Z * g["dr_se"]
    rec["dr_cov"] = ((lo <= planted) & (planted <= hi)).mean()
    rec["dr_excl0"] = ((lo > 0) | (hi < 0)).mean()
    # LinearDML
    rec["ldml_bias"] = g["ldml_ate"].mean() - planted
    rec["ldml_empSD"] = g["ldml_ate"].std(ddof=1)
    rec["ldml_SE"] = g["ldml_se"].mean()
    rec["ldml_cov"] = ((g["ldml_lo"] <= planted) & (planted <= g["ldml_hi"])).mean()
    rec["ldml_excl0"] = ((g["ldml_lo"] > 0) | (g["ldml_hi"] < 0)).mean()
    rows.append(rec)
out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
cols = ["brand", "channel", "planted", "seeds"]
print("== forest mean-CATE point with the SHIPPED ate_interval (RMS of pointwise SEs)")
print(out[cols + ["forest_bias", "forest_empSD", "ai_SE", "ai_cov", "ai_excl0"]].to_string(index=False))
print("\n== forest doubly-robust ATE (cf.ate_) with cf.ate_stderr_ (free, same fit)")
print(out[cols + ["dr_bias", "dr_empSD", "dr_SE", "dr_cov", "dr_excl0"]].to_string(index=False))
print("\n== LinearDML ate_inference (exact SE of the mean; second fit)")
print(out[cols + ["ldml_bias", "ldml_empSD", "ldml_SE", "ldml_cov", "ldml_excl0"]].to_string(index=False))
print("\n== SE / empirical-SD ratio (1.0 = calibrated; >1 = conservative)")
out["ai_ratio"] = out["ai_SE"] / out["forest_empSD"]
out["dr_ratio"] = out["dr_SE"] / out["dr_empSD"]
out["ldml_ratio"] = out["ldml_SE"] / out["ldml_empSD"]
print(out[cols + ["ai_ratio", "dr_ratio", "ldml_ratio"]].to_string(index=False))
print(f"\nmean fit seconds: forest {df['t_forest'].mean():.1f}s, lineardml {df['t_ldml'].mean():.1f}s")
