#!/usr/bin/env python3
"""#2031 cheapest disproof: does a larger RF leaf make the production LinearDML's small-ATE
estimates seed-stable with honest CI coverage on the planted-truth DGP?

Mirrors the PRODUCTION wrapper (src/causal_engine/energy_score/estimator_selector.py:649-666):
RandomForest{Regressor,Classifier}(n_estimators=50, min_samples_leaf=LEAF, min_impurity_decrease=1e-7,
random_state=SEED), LinearDML(discrete_treatment=True, random_state=SEED), X = W = covariates,
CI = ate_inference(X).conf_int_mean() (the #1188 honest interval). NOTE: the sensitivity
calibration test's _fit uses W=None and no min_impurity_decrease — production is mirrored here.

Usage: sweep_leaf_seed.py <out_dir> [n_rows] [leaves csv] [seeds csv]
Writes rows.jsonl (one per leaf×seed×pair) and summary.md.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
LEAVES = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "5,20,50,100").split(",")]
SEEDS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "42,7,123,2024,99,314,1,77").split(",")]

frame = PatientGenerator(GeneratorConfig(seed=21, n_records=N, brand=Brand.REMIBRUTINIB,
                                         dgp_type=DGPType.HETEROGENEOUS)).generate()
truth = frame.attrs["true_ate_by_arm"]
pairs = [(arm, o, list(ARM_REGISTRY[arm].confounders), float(t))
         for arm, outs in truth.items() if arm in ARM_REGISTRY and arm in frame.columns
         for o, t in ((o, v["ate"]) for o, v in outs.items()) if o in frame.columns]
assert len(pairs) == 11, [p[:2] for p in pairs]
print(f"n={N} pairs={len(pairs)} leaves={LEAVES} seeds={SEEDS}", flush=True)

def fit(df, treatment, outcome, covs, leaf, seed):
    Y = df[outcome].to_numpy(dtype=float); T = df[treatment].to_numpy(dtype=int)
    X = df[covs].to_numpy(dtype=float)
    m = LinearDML(
        model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=leaf, min_impurity_decrease=1e-7, random_state=seed),
        model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=leaf, min_impurity_decrease=1e-7, random_state=seed),
        discrete_treatment=True, random_state=seed)
    m.fit(Y, T, X=X, W=X)
    ate = float(np.mean(m.effect(X)))
    inf = m.ate_inference(X)
    lo, hi = (float(np.squeeze(v)) for v in inf.conf_int_mean())
    return ate, lo, hi, float(np.squeeze(inf.stderr_mean))

rows_path = out / f"rows_n{N}.jsonl"
with rows_path.open("w") as fh:
    t0 = time.time(); k = 0
    for leaf in LEAVES:
        for arm, o, covs, t in pairs:
            for seed in SEEDS:
                ate, lo, hi, se = fit(frame, arm, o, covs, leaf, seed)
                fh.write(json.dumps({"n": N, "leaf": leaf, "seed": seed, "treatment": arm, "outcome": o,
                                     "truth": t, "ate": ate, "lo": lo, "hi": hi, "se": se,
                                     "covers": bool(lo <= t <= hi)}) + "\n"); fh.flush()
                k += 1
            print(f"  leaf={leaf} {arm}->{o} done ({k} fits, {int(time.time()-t0)}s)", flush=True)

rows = [json.loads(l) for l in rows_path.open()]
lines = [f"# #2031 leaf × seed sweep — n={N}, {len(SEEDS)} seeds, {len(pairs)} planted truths, production-mirrored LinearDML",
         "", "## Per leaf (aggregated over 11 pairs × seeds)", "",
         "| leaf | coverage of planted truth | mean \\|bias\\| | median seed-SD / mean-SE | mean CI half-width | max \\|bias\\| pair |",
         "|---|---|---|---|---|---|"]
for leaf in LEAVES:
    R = [r for r in rows if r["leaf"] == leaf]
    cov = np.mean([r["covers"] for r in R]); ratios = []; biases = []; hw = []
    worst = ("", 0.0)
    for arm, o, covs, t in pairs:
        P = [r for r in R if r["treatment"] == arm and r["outcome"] == o]
        ates = np.array([r["ate"] for r in P]); ses = np.array([r["se"] for r in P])
        sd = float(ates.std(ddof=1)); ratios.append(sd / float(ses.mean())); b = float(abs(ates.mean() - t)); biases.append(b)
        hw.append(float(np.mean([(r["hi"] - r["lo"]) / 2 for r in P])))
        if b > worst[1]: worst = (f"{arm}->{o}", b)
    lines.append(f"| {leaf} | {cov:.2f} | {np.mean(biases):.4f} | {np.median(ratios):.2f} | {np.mean(hw):.4f} | {worst[0]} {worst[1]:.3f} |")
lines += ["", "## Per pair × leaf: mean ATE (seed SD) / mean SE / coverage", "",
          "| pair | truth | " + " | ".join(f"leaf {l}" for l in LEAVES) + " |", "|---|---|" + "---|" * len(LEAVES)]
for arm, o, covs, t in pairs:
    cells = []
    for leaf in LEAVES:
        P = [r for r in rows if r["leaf"] == leaf and r["treatment"] == arm and r["outcome"] == o]
        ates = np.array([r["ate"] for r in P])
        cells.append(f"{ates.mean():.3f} ({ates.std(ddof=1):.3f}) / {np.mean([r['se'] for r in P]):.3f} / {np.mean([r['covers'] for r in P]):.2f}")
    lines.append(f"| {arm}->{o} | {t:.3f} | " + " | ".join(cells) + " |")
(out / f"summary_n{N}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
print("DONE", flush=True)
