"""Measure, on the REAL frame, the Gram-Schmidt residual ratios of the SHIPPED prune
(``_prune_numerically_collinear``, imported from the tree under test) against an explicit
inline re-implementation of the Round-3 criterion (raw-norm comparison, 1e-12 constant test,
fixed 1e-8 tolerance) -- the historical baseline codex r4 found wanting -- and run the codex
r4/r5/r6 synthetic probes on both. Re-runnable: the JSON records the commit it was measured at."""
import sys, json, logging, math
from pathlib import Path
sys.path.insert(0, str(Path.cwd())); sys.path.insert(0, "docs/demos/results/2026-09-22_optum_biologic_persistence_cert")
logging.basicConfig(level=logging.ERROR)
import numpy as np, pandas as pd
import src; assert ".worktrees/real-data-causal" in src.__file__
from src.api.routes.causal.loaders import _prune_numerically_collinear as shipped

OUT_DEFAULT = "docs/demos/results/2026-09-22_optum_biologic_persistence_cert/prune_tolerance_probe.json"
EPS = np.finfo(float).eps

def r3_baseline(frame, columns):
    """The Round-3 criterion, verbatim in spirit: residual vs the RAW column norm, constant
    when centered/raw <= 1e-12, fixed rel tol 1e-8, skipped at n < k+1."""
    if not columns or len(frame) < len(columns) + 1:
        return list(columns), [], {}
    X = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        return list(columns), [], {}
    n = X.shape[0]
    intercept = np.full(n, 1.0 / np.sqrt(n)); basis = [intercept]
    kept, dropped, ratios = [], [], {}
    for j, name in enumerate(columns):
        x = X[:, j]
        x_norm = float(np.linalg.norm(x))
        centered = x - intercept * float(intercept @ x)
        c_norm = float(np.linalg.norm(centered))
        if x_norm == 0.0 or c_norm <= 1e-12 * x_norm:
            dropped.append(name); ratios[name] = 0.0; continue
        resid = centered.copy()
        for _ in range(2):
            for q in basis[1:]:
                resid = resid - q * float(q @ resid)
        r_norm = float(np.linalg.norm(resid))
        ratios[name] = r_norm / c_norm
        if r_norm <= 1e-8 * c_norm:
            dropped.append(name); continue
        basis.append(resid / r_norm); kept.append(name)
    return kept, dropped, ratios


def shipped_with_ratios(frame, columns):
    """Shipped decision plus the residual ratios re-derived with the shipped preprocessing."""
    import math
    kept, dropped = shipped(frame, columns)
    if not columns or len(frame) < len(columns) + 1:
        return kept, dropped, {}
    X = frame[columns].to_numpy(dtype=float)
    n = X.shape[0]
    intercept = np.full(n, 1.0 / np.sqrt(n)); basis = [intercept]; ratios = {}
    for j, name in enumerate(columns):
        x = X[:, j]
        if x.min() == x.max():
            ratios[name] = 0.0; continue
        x = np.ldexp(x, -math.frexp(float(np.abs(x).max()))[1]); x = x - x[0]
        centered = x - intercept * float(intercept @ x); c_norm = float(np.linalg.norm(centered))
        resid = centered.copy()
        for _ in range(2):
            for q in basis[1:]:
                resid = resid - q * float(q @ resid)
        r_norm = float(np.linalg.norm(resid)); ratios[name] = r_norm / c_norm
        if name in kept:
            basis.append(resid / r_norm)
    return kept, dropped, ratios


out = {}
from src.api.routes.causal import loaders as L
L._prune_numerically_collinear = lambda f, c: (list(c), [])   # recover the pre-prune 77-column design
from preflight_agent import build_frame
frame, cov = build_frame("persistent_at_180d_g28")
print("pre-prune resolved cov:", len(cov), "n", len(frame))
import subprocess
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
kc, dc, _ = r3_baseline(frame, cov)
kn, dn, ratios = shipped_with_ratios(frame, cov)
print("r3_baseline: kept", len(kc), "dropped", len(dc))
print("shipped    : kept", len(kn), "dropped", len(dn), "tol", max(len(frame), len(cov)+1)*EPS)
print("same dropped set:", set(dc) == set(dn), "same order:", dc == dn)
dr = sorted((ratios[c], c) for c in dn); kr = sorted((ratios[c], c) for c in kn)
print("dropped ratios max:", dr[-1]); print("kept ratios min 5:", kr[:5])
out["measured_at_commit"] = commit
out["real"] = {"n": len(frame), "k": len(cov), "r3_baseline_dropped": dc, "shipped_dropped": dn,
               "tol": max(len(frame), len(cov)+1)*EPS, "dropped_ratio_max": dr[-1][0], "kept_ratio_min": kr[0][0]}
# codex r4 probes
rng = np.random.default_rng(0)
a = rng.normal(size=100); noise = rng.normal(size=100)
probes = {
  "offset_1e14": pd.DataFrame({"x": 1e14 + np.arange(100.0)}),
  "noise_5e-9": pd.DataFrame({"a": a, "almost": a + 5e-9 * noise}),
  "scale_1e-200": pd.DataFrame({"a": a * 1e-200, "b": noise * 1e-200}),
  "exact_dup_offset_1e14": pd.DataFrame({"a": 1e14 + np.arange(100.0), "b": 2.0 * (1e14 + np.arange(100.0)) + 3.0}),
  "exact_dup": pd.DataFrame({"a": a, "b": 2 * a + 3}),
  "constant_float": pd.DataFrame({"a": np.full(100, 0.1)}),
  "float_max": pd.DataFrame({"a": [-1e308, 1e308, 5e307, 0.0, 1.0], "b": [1.0, 2.0, 4.0, 8.0, 16.0]}),
  "one_ulp": pd.DataFrame({"a": np.arange(8, dtype=float), "b": np.r_[np.arange(7, dtype=float), np.nextafter(7.0, np.inf)]}),
}
for name, df in probes.items():
    base = r3_baseline(df, list(df.columns))[1]; ship = shipped(df, list(df.columns))[1]
    print(f"{name:24s} r3_baseline dropped={base}  shipped dropped={ship}")
    out[name] = {"r3_baseline_dropped": base, "shipped_dropped": ship}
Path(sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT).write_text(json.dumps(out, indent=1))
