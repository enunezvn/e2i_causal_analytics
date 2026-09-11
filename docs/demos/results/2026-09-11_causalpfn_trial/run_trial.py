"""Container-side #1990 CausalPFN trial runner (spec §11 items 2-3). Reads the
frames prep_frames.py wrote; appends one JSON line per frame to results/<tag>.jsonl.
"""
from __future__ import annotations
import argparse, json, os, resource, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd, torch

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="/trial/frames")
ap.add_argument("--out", default="/trial/results")
ap.add_argument("--tag", required=True)
ap.add_argument("--calibrate", type=int, default=1)
ap.add_argument("--n-samples", type=int, default=10_000)
ap.add_argument("--glob", default="*.json")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--dml", type=int, default=1)
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(a.threads)
from causalpfn import CATEEstimator  # noqa: E402

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def cur_rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 2**20

def dml_reference(X, t, y):
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    m = LinearDML(model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=42),
                  model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42),
                  discrete_treatment=True, random_state=42)
    m.fit(y, t, X=X, W=None)
    inf = m.ate_inference(X)
    lo, hi = inf.conf_int_mean()
    return float(m.ate(X)), float(np.ravel(lo)[0]), float(np.ravel(hi)[0])

out = Path(a.out) / f"{a.tag}.jsonl"
done = set()
if out.exists():
    for line in out.read_text().splitlines():
        try: done.add(json.loads(line)["name"])
        except Exception: pass
metas = sorted(Path(a.frames).glob(a.glob))
if a.limit: metas = metas[: a.limit]
print(f"{len(metas)} frames, {len(done)} already done, rss_start={cur_rss_mb():.0f}MB", flush=True)
for mp in metas:
    name = mp.stem
    if name in done: continue
    meta = json.loads(mp.read_text())
    df = pd.read_parquet(mp.with_suffix(".parquet"))
    rec = {"name": name, **{k: meta[k] for k in ("kind", "brand", "n", "treatment", "outcome", "covariates")},
           "calibrate": a.calibrate, "n_samples": a.n_samples, "threads": a.threads}
    try:
        X = df[meta["covariates"]].to_numpy(dtype=float)
        t = df[meta["treatment"]].to_numpy(dtype=float)
        y = df[meta["outcome"]].to_numpy(dtype=float)
        ok = ~(np.isnan(X).any(axis=1) | np.isnan(t) | np.isnan(y))
        rec["n_dropped_nan"] = int((~ok).sum()); X, t, y = X[ok], t[ok], y[ok]
        rec["n_treated"] = int(t.sum()); rec["n_used"] = int(len(t))
        seg = df.loc[ok, meta["segment_col"]].to_numpy() if meta.get("segment_col") else None
        rss0 = cur_rss_mb(); t0 = time.time()
        est = CATEEstimator(device="cpu", calibrate=bool(a.calibrate), verbose=False, cache_dir="/trial/hf")
        est.fit(X, t, y); t_fit = time.time() - t0
        rec["temperature"] = float(est.temperature)
        t1 = time.time(); cate = est.estimate_cate(X); t_cate = time.time() - t1
        # causalpfn 0.1.4 defect: CATEEstimator.estimate_ate_CI reads output["ate"], a key
        # _estimate_ate_cate_CI never returns (KeyError). Call the helper directly.
        t2 = time.time(); ci = est._estimate_ate_cate_CI(X, alpha=0.05, n_samples=a.n_samples); t_ci = time.time() - t2
        rec.update({"pfn_ate": float(np.mean(cate)),
                    "pfn_ci_lo": float(np.ravel(ci["ate_lower_bound"])[0]),
                    "pfn_ci_hi": float(np.ravel(ci["ate_upper_bound"])[0]),
                    "pfn_cate_ci_median_width": float(np.median(ci["cate_upper_bound"] - ci["cate_lower_bound"])),
                    "t_fit_s": round(t_fit, 2), "t_cate_s": round(t_cate, 2), "t_ci_s": round(t_ci, 2),
                    "t_total_s": round(time.time() - t0, 2),
                    "rss_before_mb": round(rss0), "rss_after_mb": round(cur_rss_mb()), "ru_maxrss_mb": round(rss_mb())})
        if seg is not None:
            rec["pfn_cate_by_segment"] = {s: float(np.mean(cate[seg == s])) for s in np.unique(seg)}
        if "true_ate" in meta:
            rec["true_ate"] = meta["true_ate"]; rec["cate_map"] = meta["cate_by_segment"]
            rec["abs_err"] = abs(rec["pfn_ate"] - meta["true_ate"])
            rec["covered"] = bool(rec["pfn_ci_lo"] <= meta["true_ate"] <= rec["pfn_ci_hi"])
            c = rec["pfn_cate_by_segment"]
            rec["order_ok"] = bool(c.get("high_severity", -9) > c.get("medium_severity", -9) > c.get("low_severity", -9))
        if meta.get("reported_ci_spec_s2"):
            rec["reported_ci_spec_s2"] = meta["reported_ci_spec_s2"]
        if a.dml:
            t3 = time.time(); rec["dml_ate"], rec["dml_ci_lo"], rec["dml_ci_hi"] = dml_reference(X, t, y); rec["t_dml_s"] = round(time.time() - t3, 2)
            if "true_ate" in meta:
                rec["dml_abs_err"] = abs(rec["dml_ate"] - meta["true_ate"]); rec["dml_covered"] = bool(rec["dml_ci_lo"] <= meta["true_ate"] <= rec["dml_ci_hi"])
        del est
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"; rec["trace"] = traceback.format_exc()[-1500:]
    with out.open("a") as f: f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("trace", "cate_map")}), flush=True)
print("done", flush=True)
