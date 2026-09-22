import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "docs/demos/results/2026-09-22_optum_biologic_persistence_cert")
import numpy as np
import src

assert ".worktrees/real-data-causal" in src.__file__
from preflight_agent import TREATMENT, build_frame

frame, cov = build_frame("persistent_at_180d_g28")
T = frame[TREATMENT].astype(int).to_numpy()
Y = frame["persistent_at_180d_g28"].astype(float).to_numpy()
X = frame[cov]
Xv = X.to_numpy(dtype=float)
out = {"n": len(frame), "k": len(cov)}
# collinearity of the design
out["rank_X"] = int(np.linalg.matrix_rank(Xv - Xv.mean(axis=0)))
out["rank_X_with_intercept"] = int(np.linalg.matrix_rank(np.column_stack([np.ones(len(Xv)), Xv])))
print("rank", out["rank_X"], "of", len(cov), flush=True)
from econml.dml import LinearDML
from src.causal_engine.nuisance_config import linear_dml_model_t, linear_dml_model_y

t0 = time.perf_counter()
model = LinearDML(
    model_y=linear_dml_model_y(),
    model_t=linear_dml_model_t(),
    discrete_treatment=True,
    random_state=42,
)
model.fit(Y, T, X=Xv, W=Xv)
out["fit_s"] = round(time.perf_counter() - t0, 1)
print("fit", out["fit_s"], flush=True)
from src.causal_engine.energy_score.estimator_selector import _honest_ate_ci

t0 = time.perf_counter()
ci = _honest_ate_ci(model, Xv)
out["honest_ate_ci_s"] = round(time.perf_counter() - t0, 1)
out["honest_ci"] = [float(v) for v in ci] if ci else None
print("honest_ci", out["honest_ate_ci_s"], out["honest_ci"], flush=True)
from sklearn.linear_model import LogisticRegressionCV

t0 = time.perf_counter()
ps = LogisticRegressionCV(cv=3, max_iter=500).fit(Xv, T)
out["logregcv_unscaled_s"] = round(time.perf_counter() - t0, 1)
print("logregcv unscaled", out["logregcv_unscaled_s"], flush=True)
from sklearn.preprocessing import StandardScaler

t0 = time.perf_counter()
Xs = StandardScaler().fit_transform(Xv)
ps2 = LogisticRegressionCV(cv=3, max_iter=500).fit(Xs, T)
out["logregcv_scaled_s"] = round(time.perf_counter() - t0, 1)
print("logregcv scaled", out["logregcv_scaled_s"], flush=True)
out["max_rss_mb"] = round(__import__("resource").getrusage(1).ru_maxrss / 1024, 1)
Path(
    "docs/demos/results/2026-09-22_optum_biologic_persistence_cert/timing_probe2_persistent_at_180d_g28.json"
).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
