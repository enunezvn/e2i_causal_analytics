"""Lane H (#2008) scratch spike: DoubleML IRM omitted-variable-bias sensitivity vs LinearDML.

Runs inside the capped scratch container (image f4940c97a, /trial mounted). Appends one JSON
line per pair to /trial/spike.jsonl (resumable). No src/ changes, no DB access.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
import warnings

warnings.filterwarnings("ignore")

OUT = "/trial/spike.jsonl"
DRY = "--dry" in sys.argv


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cgroup_peak_mib():
    for p in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        if os.path.exists(p):
            try:
                return round(int(open(p).read().strip()) / 2**20, 1), p
            except Exception as exc:  # noqa: BLE001
                return None, f"{p}: {exc!r}"
    return None, "no cgroup peak file"


def emit(rec: dict) -> None:
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


# ---- baseline imports the production process already carries -----------------------------
t0 = time.perf_counter()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from econml.dml import LinearDML  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402

from src.ml.synthetic.config import Brand, DGPType  # noqa: E402
from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY  # noqa: E402
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator  # noqa: E402

base_s = time.perf_counter() - t0
rss_base = rss_mib()

# ---- incremental doubleml import (informational; the cold number is in import_probe.py) -----
t0 = time.perf_counter()
import doubleml  # noqa: E402
from doubleml import DoubleMLData, DoubleMLIRM  # noqa: E402

dml_import_s = time.perf_counter() - t0
rss_after_dml = rss_mib()

versions = {
    "doubleml": doubleml.__version__,
    "numpy": np.__version__,
    "sklearn": sklearn.__version__,
    "pandas": pd.__version__,
    "python": sys.version.split()[0],
    "image": os.environ.get("IMAGE_TAG", "ghcr.io/enunezvn/e2i-api:f4940c97a43dad22bf556b89bcce459afdc7c246"),
}

# ---- frame + pairs (mirrors tests/unit/test_causal_engine/test_sensitivity_calibration.py) ---
N_ROWS = 1500
NULL_PAIRS = [
    ("copay_support", "treatment_initiated"),
    ("psp_enrolled", "treatment_initiated"),
    ("rep_detailing_high", "adherent_180d"),
    ("trigger_accepted", "adherent_180d"),
    ("rep_detailing_high", "low_gap_180d"),
    ("trigger_accepted", "low_gap_180d"),
    ("rep_detailing_high", "persistent_180d"),
    ("sample_dropped", "persistent_180d"),
    ("trigger_accepted", "persistent_180d"),
]

t0 = time.perf_counter()
cfg = GeneratorConfig(seed=21, n_records=N_ROWS, brand=Brand.REMIBRUTINIB, dgp_type=DGPType.HETEROGENEOUS)
frame = PatientGenerator(cfg).generate()
frame_s = time.perf_counter() - t0


def _planted_pairs(frame):
    truth = frame.attrs["true_ate_by_arm"]
    return [
        (arm, outcome, list(ARM_REGISTRY[arm].confounders))
        for arm, outs in truth.items()
        if arm in ARM_REGISTRY and arm in frame.columns
        for outcome in outs
        if outcome in frame.columns
    ]


truths = _planted_pairs(frame)
nulls = [(a, o, list(ARM_REGISTRY[a].confounders)) for a, o in NULL_PAIRS]
truth_vals = frame.attrs["true_ate_by_arm"]

header = {
    "kind": "header",
    "versions": versions,
    "baseline_import_s": round(base_s, 2),
    "rss_after_baseline_imports_mib": round(rss_base, 1),
    "doubleml_incremental_import_s": round(dml_import_s, 3),
    "doubleml_incremental_rss_delta_mib": round(rss_after_dml - rss_base, 1),
    "frame_gen_s": round(frame_s, 2),
    "n_rows": int(len(frame)),
    "n_truths": len(truths),
    "n_nulls": len(nulls),
    "truth_pairs": [(a, o) for a, o, _ in truths],
    "nan_counts": {
        c: int(frame[c].isna().sum())
        for c in sorted({x for _, o, cs in truths + nulls for x in [o, *cs]} | {a for a, _, _ in truths + nulls})
        if frame[c].isna().any()
    },
}
if DRY:
    print(json.dumps(header, indent=1))
    for a, o, cs in truths:
        print("TRUTH", a, o, truth_vals.get(a, {}).get(o), cs, frame[o].dtype, frame[a].dtype)
    for a, o, cs in nulls:
        print("NULL ", a, o, cs)
    sys.exit(0)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if r.get("kind") == "pair":
            done.add((r["treatment"], r["outcome"]))
if not done:
    emit(header)


# ---- the LinearDML reference (copied verbatim from the test module's ``_fit``) ---------------
def _fit(df, treatment, outcome, covariates):
    Y = df[outcome].to_numpy(dtype=float)
    T = df[treatment].to_numpy(dtype=int)
    X = df[covariates].to_numpy(dtype=float)
    m = LinearDML(
        model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=42),
        model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42),
        discrete_treatment=True,
        random_state=42,
    )
    m.fit(Y, T, X=X, W=None)
    inf = m.ate_inference(X)
    lo, hi = (float(v) for v in inf.conf_int_mean())
    return float(inf.mean_point), (lo, hi)


def _irm(df, treatment, outcome, covariates):
    """DoubleMLIRM with the production RF nuisance config (n_folds=2 == econml cv=2)."""
    d = df[[outcome, treatment, *covariates]].copy()
    d[outcome] = d[outcome].astype(float)
    d[treatment] = d[treatment].astype(int)
    for c in covariates:
        d[c] = d[c].astype(float)
    data = DoubleMLData(d, y_col=outcome, d_cols=treatment, x_cols=list(covariates))
    # ml_g: the outcome is binary -> RandomForestClassifier (IRM uses predict_proba as E[Y|X,D]);
    # ml_m: propensity -> RandomForestClassifier. Same hyper-parameters as production.
    ml_g = RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42)
    ml_m = RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42)
    # doubleml 0.11.4 draws its folds with ``KFold(shuffle=True)`` and no random_state
    # (utils/resampling.py:88) -> the global numpy RNG. Seed it so the record reproduces.
    np.random.seed(42)
    m = DoubleMLIRM(data, ml_g=ml_g, ml_m=ml_m, n_folds=2, n_rep=1, score="ATE")
    m.fit()
    return m


def run_pair(kind, treatment, outcome, covariates):
    rec = {"kind": "pair", "role": kind, "treatment": treatment, "outcome": outcome, "covariates": covariates}
    if kind == "truth":
        rec["planted_true_ate"] = truth_vals.get(treatment, {}).get(outcome)
    cols = [treatment, outcome, *covariates]
    df = frame[cols].dropna()
    rec["n_rows_used"] = int(len(df))

    # LinearDML reference
    t0 = time.perf_counter()
    ate_l, ci_l = _fit(df, treatment, outcome, covariates)
    rec["lineardml"] = {"ate": ate_l, "ci": list(ci_l), "fit_s": round(time.perf_counter() - t0, 2)}

    # IRM fit (criterion 2 numbers)
    rss0 = rss_mib()
    t0 = time.perf_counter()
    m = _irm(df, treatment, outcome, covariates)
    fit_s = time.perf_counter() - t0
    rss1 = rss_mib()
    ci = m.confint(level=0.95)
    rec["irm"] = {
        "ate": float(m.coef[0]),
        "se": float(m.se[0]),
        "ci": [float(ci.iloc[0, 0]), float(ci.iloc[0, 1])],
        "fit_s": round(fit_s, 2),
        "ru_maxrss_before_mib": round(rss0, 1),
        "ru_maxrss_after_mib": round(rss1, 1),
        "ru_maxrss_delta_mib": round(rss1 - rss0, 1),
    }

    # sensitivity analysis (RV / RVa at default cf_y=cf_d=0.03, rho=1)
    t0 = time.perf_counter()
    m.sensitivity_analysis()
    sp = m.sensitivity_params
    sens_s = time.perf_counter() - t0
    rec["sensitivity"] = {
        "rv": float(sp["rv"][0]),
        "rva": float(sp["rva"][0]),
        "theta_lower": float(sp["theta"]["lower"][0]),
        "theta_upper": float(sp["theta"]["upper"][0]),
        "ci_lower": float(sp["ci"]["lower"][0]),
        "ci_upper": float(sp["ci"]["upper"][0]),
        "analysis_s": round(sens_s, 3),
    }

    # benchmark each declared covariate singly; strongest = largest cf_d
    t0 = time.perf_counter()
    bench = {}
    for c in covariates:
        np.random.seed(42)  # the short-model refit draws new folds too
        b = m.sensitivity_benchmark(benchmarking_set=[c])
        bench[c] = {
            "cf_y": float(b["cf_y"].iloc[0]),
            "cf_d": float(b["cf_d"].iloc[0]),
            "rho": float(b["rho"].iloc[0]),
            "delta_theta": float(b["delta_theta"].iloc[0]),
        }
    bench_s = time.perf_counter() - t0
    strongest = max(bench, key=lambda c: bench[c]["cf_d"])
    rec["benchmark"] = {"per_covariate": bench, "strongest_by_cf_d": strongest, "benchmark_s": round(bench_s, 2)}
    sb = bench[strongest]
    rv = rec["sensitivity"]["rv"]
    if kind == "truth":
        rec["rule"] = {"rv_gt_cf_y": rv > sb["cf_y"], "rv_gt_cf_d": rv > sb["cf_d"], "pass": rv > sb["cf_y"] and rv > sb["cf_d"]}
    else:
        rec["rule"] = {"rv_le_0_02": rv <= 0.02, "ci_includes_0": rec["irm"]["ci"][0] <= 0 <= rec["irm"]["ci"][1], "pass": rv <= 0.02}
    peak, src = cgroup_peak_mib()
    rec["cgroup_peak_mib"] = peak
    rec["cgroup_peak_src"] = src
    rec["ru_maxrss_now_mib"] = round(rss_mib(), 1)
    rec["pair_total_s"] = round(rec["lineardml"]["fit_s"] + fit_s + sens_s + bench_s, 2)
    return rec


for kind, plist in (("truth", truths), ("null", nulls)):
    for a, o, cs in plist:
        if (a, o) in done:
            continue
        try:
            emit(run_pair(kind, a, o, cs))
        except Exception as exc:  # noqa: BLE001
            emit({"kind": "pair", "role": kind, "treatment": a, "outcome": o, "covariates": cs, "error": repr(exc)})

peak, src = cgroup_peak_mib()
emit({"kind": "footer", "cgroup_peak_mib": peak, "cgroup_peak_src": src, "ru_maxrss_final_mib": round(rss_mib(), 1)})
