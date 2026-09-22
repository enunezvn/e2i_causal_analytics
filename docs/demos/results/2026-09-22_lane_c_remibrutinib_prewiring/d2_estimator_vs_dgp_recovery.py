#!/usr/bin/env python3
"""Lane C D2 companion: is the +0.04..+0.05 gap between the graph's LinearDML
estimate and the planted truth a GENERATOR defect or an ESTIMATOR property?
Cheapest disproof: adjust for the planted confounders with the DGP's own
structure (probit outcome, T x age-segment CATE) and see whether the truth is
recovered. Light (statsmodels, seconds). Run from the lane worktree root:
  $PY docs/demos/results/2026-09-22_lane_c_remibrutinib_prewiring/d2_estimator_vs_dgp_recovery.py
"""

from __future__ import annotations

import os
import sys

for _v in ("SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
    os.environ[_v] = ""
sys.path.insert(0, os.getcwd())
import src  # noqa: E402

assert ".worktrees/lane-c-remibrutinib-prewiring" in src.__file__, src.__file__

import numpy as np  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from src.ml.synthetic.generators.csu_escalation_causal import (  # noqa: E402
    PRIMARY_OUTCOME,
    TREATMENT,
    generate_csu_escalation_cohort,
)

f, truths = generate_csu_escalation_cohort()  # the E2E test's frame (n=3000, seed 20260922)
t = truths[PRIMARY_OUTCOME]
print("truth", t.true_ate, "naive", t.naive_diff, "cate", t.cate_by_segment, "arms", t.arm_split)
T = f[TREATMENT].values.astype(float)
y = f[PRIMARY_OUTCOME].values.astype(float)
age = f.age_at_index.values.astype(float)
ch = f.charlson_score.values.astype(float)
com = (f.payer_category == "commercial").astype(float).values
dep = f.elx_depression.values.astype(float)
seg_m = ((age > 40) & (age <= 60)).astype(float)
seg_h = (age > 60).astype(float)


def design(tv: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.ones_like(tv), tv, tv * seg_m, tv * seg_h, age, ch, com, dep, seg_m, seg_h]
    )


ones, zeros = np.ones_like(T), np.zeros_like(T)
m = sm.Probit(y, design(T)).fit(disp=0)
mu1, mu0 = m.predict(design(ones)), m.predict(design(zeros))
print("g-comp probit (DGP-structured) ATE", round(float((mu1 - mu0).mean()), 4))
ols = sm.OLS(y, np.column_stack([ones, T, age, ch, com, dep])).fit()
print("OLS constant-effect ATE", round(float(ols.params[1]), 4))
ps = np.clip(sm.Logit(T, np.column_stack([ones, age, ch, com])).fit(disp=0).predict(), 0.01, 0.99)
print("IPW ATE", round(float((T * y / ps).mean() - ((1 - T) * y / (1 - ps)).mean()), 4))
aipw = (mu1 - mu0 + T * (y - mu1) / ps - (1 - T) * (y - mu0) / (1 - ps)).mean()
print("AIPW ATE", round(float(aipw), 4))
print(
    "graph LinearDML on the same DGP shape (d2_probe_narrow_3000.json): ate 0.4072, ci [0.3726, 0.4418], true 0.3643"
)
