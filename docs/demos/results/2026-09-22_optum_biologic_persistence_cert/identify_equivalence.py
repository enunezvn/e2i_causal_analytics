import sys, time, json, logging, warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import src

assert ".worktrees/real-data-causal" in src.__file__
from dowhy import CausalModel
from src.causal_engine import nuisance_config as nc
from src.agents.causal_impact.nodes import _dowhy_order


def frame(k, n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    X[:, 0] *= 1e3
    t = (rng.random(n) < 1 / (1 + np.exp(-(X[:, 1] + 0.3 * X[:, 2])))).astype(int)
    y = 0.5 * t + X[:, 1] + 0.2 * X[:, 2] + rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(k)])
    df["t"] = t
    df["y"] = y
    return df, [f"c{i}" for i in range(k)]


def identify(df, cov, optimize):
    m = CausalModel(data=df, treatment="t", outcome="y", common_causes=cov, effect_modifiers=cov)
    t0 = time.perf_counter()
    est = m.identify_effect(proceed_when_unidentifiable=True, optimize_backdoor=optimize)
    dt = time.perf_counter() - t0
    _dowhy_order.pin_adjustment_order(est, cov)
    return m, est, dt


def estimate(m, est, cov):
    ip = {"random_state": 42, "discrete_treatment": True}
    ip.update(nc.linear_dml_init_params(True))
    e = m.estimate_effect(
        est,
        method_name="backdoor.econml.dml.LinearDML",
        method_params={"init_params": ip, "fit_params": {}},
        test_significance=False,
        effect_modifiers=cov,
    )
    return float(e.value), list(e.estimator._observed_common_causes_names)


out = {}
for k in (8, 12):
    df, cov = frame(k)
    m0, e0, d0 = identify(df, cov, False)
    m1, e1, d1 = identify(df, cov, True)
    v0, c0 = estimate(m0, e0, cov)
    v1, c1 = estimate(m1, e1, cov)
    out[k] = {
        "default_s": round(d0, 2),
        "opt_s": round(d1, 2),
        "default_backdoor": {kk: sorted(v) for kk, v in (e0.backdoor_variables or {}).items()},
        "default_general": {
            kk: len(v) for kk, v in (e0.general_adjustment_variables or {}).items()
        },
        "opt_backdoor": {kk: len(v) for kk, v in (e1.backdoor_variables or {}).items()},
        "default_bd_id": e0.default_backdoor_id,
        "opt_bd_id": e1.default_backdoor_id,
        "default_identifier_method": e0.identifier_method,
        "opt_identifier_method": e1.identifier_method,
        "ate_default": v0,
        "ate_opt": v1,
        "ate_identical": v0 == v1,
        "cols_identical": c0 == c1,
        "cols_default": c0,
    }
    print(k, json.dumps(out[k]), flush=True)
# cliff: default path at k=17 (2^17 > 100k iterations cap)
df, cov = frame(17)
m0, e0, d0 = identify(df, cov, False)
m1, e1, d1 = identify(df, cov, True)
out[17] = {
    "default_s": round(d0, 1),
    "opt_s": round(d1, 2),
    "default_backdoor_n": {kk: len(v) for kk, v in (e0.backdoor_variables or {}).items()},
    "default_general_n": {kk: len(v) for kk, v in (e0.general_adjustment_variables or {}).items()},
    "opt_backdoor_n": {kk: len(v) for kk, v in (e1.backdoor_variables or {}).items()},
}
print(17, json.dumps(out[17]), flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
