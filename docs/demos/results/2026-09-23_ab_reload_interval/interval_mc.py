"""Interval calibration Monte-Carlo for lane T2 ("confirm interval").

Question: is the narrower population-ATE interval CALIBRATED under the DGP's sampling
distribution, or merely narrower?  For each DGP seed we re-derive `adopted` on the CACHED live
inputs (same centrality, same planted rollups), fit the twin's EXACT CausalForestDML (same
params, same X/W construction via the estimator's own helpers) and record three candidate
intervals from the SAME data:

  forest_ai   = cf.ate_interval(x)             (shipped; RMS of pointwise CATE SEs, econml's
                                                 documented conservative upper bound)
  forest_dr   = cf.ate_ +/- 1.96 * cf.ate_stderr_  (doubly-robust ATE on the training rows,
                                                 free: computed inside fit when drate=True)
  lineardml   = LinearDML.ate_inference(X).conf_int_mean  (exact SE of the mean; second fit)

Calibration = reported SE vs the EMPIRICAL SD of the point estimate across seeds, and 95 %
coverage of the planted RD.  Read-only: cached parquet inputs, no DB access.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import date

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = "/home/enunez/Projects/e2i_causal_analytics"
sys.path.insert(0, ROOT)
assert os.path.realpath(ROOT) in os.path.realpath(__import__("src").__file__), "src must resolve to main"
S = os.path.dirname(os.path.abspath(__file__))

import logging  # noqa: E402

logging.disable(logging.WARNING)

from econml.dml import CausalForestDML, LinearDML  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402

from scripts.backfill_hcp_treatment_arm import derive  # noqa: E402
from src.digital_twin.effect import cohort_causal_estimator as cce  # noqa: E402
from src.data.per_hcp_cohort_columns import (  # noqa: E402
    ADOPTION_CHANNEL_PLANTED_RD,
    INTERVENTION_TREATMENT_MAP,
)

PLANTED = {INTERVENTION_TREATMENT_MAP[k]: float(v) for k, v in ADOPTION_CHANNEL_PLANTED_RD.items()}
CHANNELS = ["engagement_score", "peer_influence_score", "rep_training_score"]  # big, mid, NULL
PLAN = {"Remibrutinib": 30, "Fabhalta": 12, "Kisqali": 12}  # seeds per brand
BASE_SEEDS = [427] + list(range(1001, 1001 + 40))
FIT_SEED = 42  # the twin's estimator seed
OUT = f"{S}/interval_mc.csv"

centrality = pd.read_parquet(f"{S}/mc_centrality.parquet")
rollups = pd.read_parquet(f"{S}/mc_rollups.parquet")
run_date = max(date.today(), rollups["metric_date"].max().date() + pd.Timedelta(days=1).to_pytimedelta())


def fit_cell(sub: pd.DataFrame, col: str) -> dict:
    work = cce._usable_rows(
        sub, col, outcome_col="adopted", region_col="region", confounders=cce.DEFAULT_CONFOUNDERS
    )
    t_thr = float(work["t_raw"].median())
    t = (work["t_raw"] > t_thr).astype(int).to_numpy()
    y = work["y"].to_numpy(dtype=float)
    x = cce._effect_modifier_matrix(work)
    cols = []
    for c in cce.DEFAULT_CONFOUNDERS:
        v = work[c].to_numpy(dtype=float)
        cols.append(np.log1p(np.clip(v, 0.0, None)) if c in cce._LOG_CONFOUNDERS else v)
    w = np.column_stack(cols)

    t0 = time.time()
    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=FIT_SEED),
        model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=FIT_SEED),
        discrete_treatment=True,
        n_estimators=200,
        subforest_size=4,
        min_samples_leaf=10,
        random_state=FIT_SEED,
    )
    cf.fit(y, t, X=x, W=w)
    eff = np.asarray(cf.effect(x), dtype=float).ravel()
    lo, hi = cf.ate_interval(x, alpha=0.05)
    ai_lo, ai_hi = float(np.ravel(lo)[0]), float(np.ravel(hi)[0])
    dr_ate = float(np.ravel(cf.ate_)[0])
    dr_se = float(np.ravel(cf.ate_stderr_)[0])
    t_forest = time.time() - t0

    t0 = time.time()
    xl = x[:, 1:] if x.shape[1] > 1 else x  # drop one dummy: LinearDML fits its own intercept
    ldml = LinearDML(
        model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=FIT_SEED),
        model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=FIT_SEED),
        discrete_treatment=True,
        random_state=FIT_SEED,
    )
    ldml.fit(y, t, X=xl, W=w)
    inf = ldml.ate_inference(xl)
    l_lo, l_hi = inf.conf_int_mean(alpha=0.05)
    t_ldml = time.time() - t0

    return {
        "n": int(len(work)),
        "point_forest": float(np.mean(eff)),
        "ai_lo": ai_lo,
        "ai_hi": ai_hi,
        "ai_se": (ai_hi - ai_lo) / (2 * 1.959964),
        "dr_ate": dr_ate,
        "dr_se": dr_se,
        "ldml_ate": float(np.ravel(inf.mean_point)[0]),
        "ldml_lo": float(np.ravel(l_lo)[0]),
        "ldml_hi": float(np.ravel(l_hi)[0]),
        "ldml_se": float(np.ravel(inf.stderr_mean)[0]),
        "t_forest": round(t_forest, 1),
        "t_ldml": round(t_ldml, 1),
    }


rows = []
done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    rows = prev.to_dict("records")
    done = set(zip(prev["brand"], prev["seed"], prev["channel"]))

for brand, n_seeds in PLAN.items():
    for seed in BASE_SEEDS[:n_seeds]:
        need = [c for c in CHANNELS if (brand, seed, c) not in done]
        if not need:
            continue
        d = derive(centrality, seed=seed, channel_rollups=rollups, run_date=run_date)
        d = d[(d["brand"] == brand) & d["joined"].astype(bool)].reset_index(drop=True)
        for col in need:
            rec = {"brand": brand, "seed": seed, "channel": col, "planted": PLANTED[col]}
            rec.update(fit_cell(d, col))
            rows.append(rec)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(
                f"{brand:13s} seed={seed:5d} {col:22s} n={rec['n']} planted={rec['planted']:.3f} "
                f"forest={rec['point_forest']:+.3f}(ai_se {rec['ai_se']:.3f}) dr={rec['dr_ate']:+.3f}"
                f"(se {rec['dr_se']:.3f}) ldml={rec['ldml_ate']:+.3f}(se {rec['ldml_se']:.3f}) "
                f"[{rec['t_forest']}s+{rec['t_ldml']}s]",
                flush=True,
            )
print("DONE", len(rows))
