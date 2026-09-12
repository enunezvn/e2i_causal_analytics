#!/usr/bin/env python3
"""#2031 faithful disproof: the SAME leaf × seed sweep on the LIVE Remibrutinib discovery frames
(patient_journeys, brand Remibrutinib, n = _DISCOVERY_ROW_CAP = 5000, the SSOT adjustment set per
question, loaded through the route's own loader — read-only vs the DB). No planted truth here; the
measured quantities are the ones behind the live finding: seed-to-seed SD of the ATE vs the reported SE,
and whether the seed-42 CI contains the other seeds' estimates.

Usage: sweep_live_frame.py <out_dir> [leaves csv] [seeds csv]
"""
import asyncio, json, os, sys, time
from pathlib import Path
import numpy as np
REPO = "/home/enunez/Projects/e2i_causal_analytics"
sys.path.insert(0, REPO); os.chdir(REPO)
import src  # noqa: E402
assert src.__file__.startswith(REPO + "/"), src.__file__
from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO + "/.env")
from econml.dml import LinearDML  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
LEAVES = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "5,50,100").split(",")]
SEEDS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "42,7,123,2024,99,314").split(",")]
DATASET, BRAND = "patient_journeys", "Remibrutinib"

def fit(df, t, o, covs, leaf, seed):
    Y = df[o].to_numpy(dtype=float); T = df[t].to_numpy(dtype=int); X = df[covs].to_numpy(dtype=float)
    m = LinearDML(
        model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=leaf, min_impurity_decrease=1e-7, random_state=seed),
        model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=leaf, min_impurity_decrease=1e-7, random_state=seed),
        discrete_treatment=True, random_state=seed)
    m.fit(Y, T, X=X, W=X)
    ate = float(np.mean(m.effect(X))); inf = m.ate_inference(X)
    lo, hi = (float(np.squeeze(v)) for v in inf.conf_int_mean())
    return ate, lo, hi, float(np.squeeze(inf.stderr_mean))

async def main():
    from src.api.routes.causal import _DISCOVERY_ROW_CAP, _load_agent_estimation_frame, _resolve_discovery_scope
    _, questions = await _resolve_discovery_scope(DATASET, BRAND)
    print(f"questions={len(questions)} leaves={LEAVES} seeds={SEEDS} cap={_DISCOVERY_ROW_CAP}", flush=True)
    rows_path = out / "rows_live.jsonl"; t0 = time.time(); k = 0
    with rows_path.open("w") as fh:
        for q in questions:
            t, o = q.treatment, q.outcome
            df, select_cols = await _load_agent_estimation_frame(
                dataset=DATASET, treatment_var=t, outcome_var=o, covariates=q.adjustment_set,
                limit=_DISCOVERY_ROW_CAP, brand=q.brand or BRAND)
            covs = [c for c in select_cols if c not in (t, o)]
            for leaf in LEAVES:
                for seed in SEEDS:
                    ate, lo, hi, se = fit(df, t, o, covs, leaf, seed)
                    fh.write(json.dumps({"n": int(len(df)), "leaf": leaf, "seed": seed, "treatment": t, "outcome": o,
                                         "covs": covs, "ate": ate, "lo": lo, "hi": hi, "se": se}) + "\n"); fh.flush(); k += 1
                print(f"  {t}->{o} leaf={leaf} done ({k} fits, {int(time.time()-t0)}s, n={len(df)})", flush=True)
    rows = [json.loads(l) for l in rows_path.open()]
    pairs = sorted({(r["treatment"], r["outcome"]) for r in rows})
    L = ["# #2031 leaf × seed sweep on the LIVE Remibrutinib discovery frames (n=5000, SSOT adjustment sets)", "",
         "## Per leaf (over 11 pairs × seeds)", "",
         "| leaf | median seed-SD / mean-SE | max seed-SD / SE (pair) | pairs where seed-42 CI excludes ≥1 other seed's ATE | mean SE |", "|---|---|---|---|---|"]
    for leaf in LEAVES:
        ratios = []; excl = 0; ses = []; worst = ("", 0.0)
        for t, o in pairs:
            P = [r for r in rows if r["leaf"] == leaf and r["treatment"] == t and r["outcome"] == o]
            ates = np.array([r["ate"] for r in P]); se = float(np.mean([r["se"] for r in P])); ses.append(se)
            ratio = float(ates.std(ddof=1)) / se; ratios.append(ratio)
            if ratio > worst[1]: worst = (f"{t}->{o}", ratio)
            s42 = [r for r in P if r["seed"] == 42][0]
            if any(not (s42["lo"] <= r["ate"] <= s42["hi"]) for r in P if r["seed"] != 42): excl += 1
        L.append(f"| {leaf} | {np.median(ratios):.2f} | {worst[1]:.2f} ({worst[0]}) | {excl}/{len(pairs)} | {np.mean(ses):.4f} |")
    L += ["", "## Per pair × leaf: mean ATE (seed SD) / mean SE / seed-42 ATE [CI]", "",
          "| pair | " + " | ".join(f"leaf {l}" for l in LEAVES) + " |", "|---|" + "---|" * len(LEAVES)]
    for t, o in pairs:
        cells = []
        for leaf in LEAVES:
            P = [r for r in rows if r["leaf"] == leaf and r["treatment"] == t and r["outcome"] == o]
            ates = np.array([r["ate"] for r in P]); s42 = [r for r in P if r["seed"] == 42][0]
            cells.append(f"{ates.mean():.3f} ({ates.std(ddof=1):.3f}) / {np.mean([r['se'] for r in P]):.3f} / {s42['ate']:.3f} [{s42['lo']:.3f}, {s42['hi']:.3f}]")
        L.append(f"| {t}->{o} | " + " | ".join(cells) + " |")
    (out / "summary_live.md").write_text("\n".join(L) + "\n"); print("\n".join(L), flush=True); print("DONE", flush=True)

asyncio.run(main())
