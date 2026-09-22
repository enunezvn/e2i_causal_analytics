"""Measure, on the REAL frame, the Gram-Schmidt residual ratios of the current prune and of a
candidate criterion (reference-subtracted, power-of-two scaled, exact-equality constant test,
tolerance max(n,k)*eps), and run codex r4's three synthetic probes on both."""
import sys, json, logging, math
from pathlib import Path
sys.path.insert(0, str(Path.cwd())); sys.path.insert(0, "docs/demos/results/2026-09-22_optum_biologic_persistence_cert")
logging.basicConfig(level=logging.ERROR)
import numpy as np, pandas as pd
import src; assert ".worktrees/real-data-causal" in src.__file__
from src.api.routes.causal.loaders import _prune_exactly_collinear as current

EPS = np.finfo(float).eps

def candidate(frame, columns, tol=None):
    if not columns or len(frame) < len(columns) + 1:
        return list(columns), [], {}
    X = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        return list(columns), [], {}
    n, k = X.shape
    rel_tol = tol if tol is not None else max(n, k + 1) * EPS
    intercept = np.full(n, 1.0 / np.sqrt(n)); basis = [intercept]
    kept, dropped, ratios = [], [], {}
    for j, name in enumerate(columns):
        x = X[:, j]
        if x.min() == x.max():
            dropped.append(name); ratios[name] = 0.0; continue
        x = x - x[0]                       # exact for close values; kills the offset
        scale = 2.0 ** math.floor(math.log2(np.abs(x).max()))
        x = x / scale                      # exact (power of two)
        centered = x - intercept * float(intercept @ x)
        c_norm = float(np.linalg.norm(centered))
        resid = centered.copy()
        for _ in range(2):
            for q in basis[1:]:
                resid = resid - q * float(q @ resid)
        r_norm = float(np.linalg.norm(resid))
        ratios[name] = r_norm / c_norm
        if r_norm <= rel_tol * c_norm:
            dropped.append(name); continue
        basis.append(resid / r_norm); kept.append(name)
    return kept, dropped, ratios

out = {}
from src.api.routes.causal import loaders as L
L._prune_exactly_collinear = lambda f, c: (list(c), [])   # recover the pre-prune 77-column design
from preflight_agent import build_frame
frame, cov = build_frame("persistent_at_180d_g28")
print("pre-prune resolved cov:", len(cov), "n", len(frame))
kc, dc = current(frame, cov)
kn, dn, ratios = candidate(frame, cov)
print("current : kept", len(kc), "dropped", len(dc))
print("candidate: kept", len(kn), "dropped", len(dn), "tol", max(len(frame), len(cov)+1)*EPS)
print("same dropped set:", set(dc) == set(dn), "same order:", dc == dn)
dr = sorted((ratios[c], c) for c in dn); kr = sorted((ratios[c], c) for c in kn)
print("dropped ratios max:", dr[-1]); print("kept ratios min 5:", kr[:5])
out["real"] = {"n": len(frame), "k": len(cov), "current_dropped": dc, "candidate_dropped": dn,
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
}
for name, df in probes.items():
    cur = current(df, list(df.columns))[1]; cand = candidate(df, list(df.columns))[1]
    print(f"{name:24s} current dropped={cur}  candidate dropped={cand}")
    out[name] = {"current_dropped": cur, "candidate_dropped": cand}
Path(sys.argv[1] if len(sys.argv) > 1 else "prune_tol_probe.json").write_text(json.dumps(out, indent=1))
