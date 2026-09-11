"""Lane G (#2007) cheapest disproof: does OMITTING an arm's declared confounders move its
negative-control (structurally null) outcome away from zero at the live row cap?

For each NULL pair (arm, nc_outcome) from tests/unit/test_causal_engine/test_sensitivity_calibration.py
fit the production LinearDML (RF nuisances, n_estimators=50, min_samples_leaf=5, seed 42) twice on the
seed-21 HETEROGENEOUS Remibrutinib frame at n = 1500: adjusted on ARM_REGISTRY[arm].confounders, and
with them OMITTED (X = a seeded standard-normal noise column, so the estimator has no confounder to
condition on). The naive difference in outcome rate with a Wald 95 % CI is recorded beside it.
A pair RESPONDS when the adjusted CI includes 0 and the omitted CI excludes 0. The 11 planted truths
are fitted the same two ways as a positive control that the frame and fits match the calibration file.
Nothing here touches src/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import src  # noqa: F401

assert "lane-g-2007" in src.__file__, src.__file__

from src.ml.synthetic.config import Brand, DGPType  # noqa: E402
from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY  # noqa: E402
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator  # noqa: E402

sys.path.insert(0, str(Path(src.__file__).resolve().parent.parent))
from tests.unit.test_causal_engine.test_sensitivity_calibration import (  # noqa: E402
    N_ROWS,
    NULL_PAIRS,
    _fit,
    _planted_pairs,
)

OUT = Path(__file__).resolve().parent
Z = 1.959964


def fit_omitted(df, treatment, outcome, seed=42):
    rng = np.random.default_rng(seed)
    Y = df[outcome].to_numpy(dtype=float)
    T = df[treatment].to_numpy(dtype=int)
    X = rng.standard_normal((len(df), 1))
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


def naive(df, treatment, outcome):
    t = df[treatment].to_numpy(dtype=int)
    y = df[outcome].to_numpy(dtype=float)
    y1, y0 = y[t == 1], y[t == 0]
    d = float(y1.mean() - y0.mean())
    se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    return d, (float(d - Z * se), float(d + Z * se)), int(len(y1)), int(len(y0))


def excludes_zero(ci):
    return bool(ci[0] > 0 or ci[1] < 0)


def main() -> int:
    cfg = GeneratorConfig(
        seed=21, n_records=N_ROWS, brand=Brand.REMIBRUTINIB, dgp_type=DGPType.HETEROGENEOUS
    )
    df = PatientGenerator(cfg).generate()
    rows = []
    t0 = time.time()
    for kind, pairs in (
        ("null", [(a, o, list(ARM_REGISTRY[a].confounders)) for a, o in NULL_PAIRS]),
        ("truth", _planted_pairs(df)),
    ):
        for arm, outcome, covs in pairs:
            ta = time.time()
            adj, adj_ci = _fit(df, arm, outcome, covs)
            om, om_ci = fit_omitted(df, arm, outcome)
            nv, nv_ci, n1, n0 = naive(df, arm, outcome)
            truth = None
            if kind == "truth":
                truth = float(df.attrs["true_ate_by_arm"][arm][outcome]["ate"])
            row = {
                "kind": kind,
                "arm": arm,
                "outcome": outcome,
                "confounders": covs,
                "n": int(len(df)),
                "n_treated": n1,
                "n_control": n0,
                "truth": truth,
                "adjusted": {
                    "ate": adj,
                    "ci": list(adj_ci),
                    "excludes_zero": excludes_zero(adj_ci),
                },
                "omitted": {"ate": om, "ci": list(om_ci), "excludes_zero": excludes_zero(om_ci)},
                "naive": {"ate": nv, "ci": list(nv_ci), "excludes_zero": excludes_zero(nv_ci)},
                "shift_omitted_minus_adjusted": om - adj,
                "responds": (not excludes_zero(adj_ci)) and excludes_zero(om_ci),
                "seconds": round(time.time() - ta, 1),
            }
            rows.append(row)
            print(
                f"{kind:5} {arm:>18} -> {outcome:<20} adj {adj:+.4f} [{adj_ci[0]:+.4f},{adj_ci[1]:+.4f}]"
                f"  omit {om:+.4f} [{om_ci[0]:+.4f},{om_ci[1]:+.4f}]  naive {nv:+.4f}"
                f"  {'RESPONDS' if row['responds'] else ('truth' if kind == 'truth' else 'no')}  {row['seconds']}s",
                flush=True,
            )
    (OUT / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    nulls = [r for r in rows if r["kind"] == "null"]
    responders = [r for r in nulls if r["responds"]]
    adj_false_pos = [r for r in nulls if r["adjusted"]["excludes_zero"]]
    truths = [r for r in rows if r["kind"] == "truth"]
    truth_adj_ok = [r for r in truths if r["adjusted"]["excludes_zero"]]
    md = [
        "# Lane G (#2007) disproof: omitted-confounder fits on the 9 structural nulls (seed 21, n = 1500)",
        "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by `run_disproof.py` in {time.time() - t0:.0f} s. "
        "Estimator = production LinearDML (RF nuisances 50 trees / leaf 5 / seed 42). "
        "`omitted` conditions on a seeded noise column only.",
        "",
        f"**Nulls that RESPOND (adjusted CI ∋ 0, omitted CI ∌ 0): {len(responders)}/{len(nulls)}.** "
        f"Adjusted false positives on nulls: {len(adj_false_pos)}/{len(nulls)}. "
        f"Positive control: planted truths with adjusted CI ∌ 0: {len(truth_adj_ok)}/{len(truths)}.",
        "",
        "| kind | arm → outcome | confounders | adjusted ATE [95 % CI] | omitted ATE [95 % CI] | naive Δ [95 % CI] | shift | responds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        a, o, nv = r["adjusted"], r["omitted"], r["naive"]
        md.append(
            f"| {r['kind']} | {r['arm']} → {r['outcome']}"
            + (f" (truth {r['truth']:+.3f})" if r["truth"] is not None else "")
            + f" | {', '.join(r['confounders'])} | {a['ate']:+.4f} [{a['ci'][0]:+.4f}, {a['ci'][1]:+.4f}]"
            f" | {o['ate']:+.4f} [{o['ci'][0]:+.4f}, {o['ci'][1]:+.4f}]"
            f" | {nv['ate']:+.4f} [{nv['ci'][0]:+.4f}, {nv['ci'][1]:+.4f}]"
            f" | {r['shift_omitted_minus_adjusted']:+.4f} | {'**yes**' if r['responds'] else 'no'} |"
        )
    (OUT / "disproof.md").write_text("\n".join(md) + "\n")
    print(
        f"\nresponders {len(responders)}/{len(nulls)}; adjusted null false positives {len(adj_false_pos)}; "
        f"truths detected adjusted {len(truth_adj_ok)}/{len(truths)}; {time.time() - t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
