#!/usr/bin/env python
"""Guide SYNTHETIC-CAUSAL-DATA-GUIDE.md §5.0 lean checks 1-5 (offline), saved as JSON.

Run from repo root:
    LOKY_MAX_CPU_COUNT=1 .venv/bin/python docs/reports/synthetic_csu_e2e_validation_20260610/lean_checks/run_lean_checks.py
"""

import glob
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")
BASE = "data/rwd/synthetic_CSU"
OUT = Path(__file__).parent / "lean_check_results.json"
results = {"generated": datetime.now().isoformat(), "dataset": BASE, "checks": {}}

f = pd.read_parquet(f"{BASE}/cohort_frames/initiation__Remibrutinib.parquet")
y, t, e = f["outcome"].values, f["treatment_arm"].values, f["propensity_score"].values
gt = json.load(open(sorted(glob.glob(f"{BASE}/ground_truth_*.json"))[-1]))
remi = next(g for g in gt if g["brand"] == "Remibrutinib" and g["dgp_type"] == "confounded")
true_ate = remi["true_ate"]

# 1 ATE recovery (IPW with stored designed propensity)
naive = float(y[t == 1].mean() - y[t == 0].mean())
ipw = float(
    np.average(y[t == 1], weights=1 / e[t == 1])
    - np.average(y[t == 0], weights=1 / (1 - e[t == 0]))
)
results["checks"]["1_ate_recovery"] = {
    "true_ate": true_ate,
    "ipw_estimate": round(ipw, 4),
    "abs_err": round(abs(ipw - true_ate), 4),
    "tolerance": 0.10,
    "pass": abs(ipw - true_ate) < 0.10,
}
# 2 confounding contrast
results["checks"]["2_confounding_contrast"] = {
    "naive_estimate": round(naive, 4),
    "naive_abs_err": round(abs(naive - true_ate), 4),
    "ipw_abs_err": round(abs(ipw - true_ate), 4),
    "pass": abs(ipw - true_ate) < abs(naive - true_ate),
}
# 3 CATE segment ordering
seg = f.groupby("segment_assignment")["treatment_effect_estimate"].mean()
order_ok = bool(seg["high_severity"] > seg["medium_severity"] > seg["low_severity"])
results["checks"]["3_cate_ordering"] = {
    "realized": {k: round(v, 4) for k, v in seg.items()},
    "sidecar": remi.get("cate_by_segment"),
    "pass": order_ok,
}
# 4 propensity recoverability + overlap
pj = pd.read_parquet(
    f"{BASE}/patient_journeys.parquet",
    columns=["brand", "treatment_arm", "disease_severity", "academic_hcp"],
)
r = pj[pj["brand"] == "Remibrutinib"]
Xp = r[["disease_severity", "academic_hcp"]].astype(float).values
auc_p = float(
    roc_auc_score(
        r["treatment_arm"],
        LogisticRegression(max_iter=1000).fit(Xp, r["treatment_arm"]).predict_proba(Xp)[:, 1],
    )
)
results["checks"]["4_propensity_overlap"] = {
    "auc": round(auc_p, 3),
    "e_min": round(float(e.min()), 3),
    "e_max": round(float(e.max()), 3),
    "pass": 0.55 < auc_p < 0.95 and e.min() >= 0.01 and e.max() <= 0.99,
}
# 5 tier0 modelability per cohort (on the tier0 contract frames)
cv = StratifiedKFold(3, shuffle=True, random_state=0)
targets = {
    "initiation": "treatment_initiated",
    "discontinuation": "discontinued_180d",
    "persistence": "persistent_180d",
    "hcp_adoption": "adopted_target_brand",
}
mod = {}
for cohort, target in targets.items():
    df = pd.read_parquet(f"{BASE}/tier0/{cohort}/e2i_ml_v3_patient_journeys.parquet")
    num = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
    num = num.loc[:, num.nunique() > 1]
    aucs = {}
    for name, mdl in [
        ("lr", LogisticRegression(max_iter=1000)),
        ("gbc", GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=0)),
    ]:
        aucs[name] = round(
            float(
                cross_val_score(
                    mdl, num.fillna(0), df[target], cv=cv, scoring="roc_auc", n_jobs=1
                ).mean()
            ),
            3,
        )
    mod[cohort] = {
        "rows": int(len(df)),
        "prevalence": round(float(df[target].mean()), 3),
        "cv_auc": aucs,
        "pass": max(aucs.values()) > 0.55 and max(aucs.values()) < 0.99,
    }
results["checks"]["5_tier0_modelability"] = {
    "cohorts": mod,
    "pass": all(c["pass"] for c in mod.values()),
}

results["all_pass"] = all(
    c["pass"] if "pass" in c else True for c in results["checks"].values()
)
def _np(o):
    return o.item() if hasattr(o, "item") else str(o)


OUT.write_text(json.dumps(results, indent=2, default=_np))
print(json.dumps(results, indent=2, default=_np))
