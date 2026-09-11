"""Lane H (#2008) SUPPLEMENTARY diagnostics (not part of the fixed criteria).

(a) IRM out-of-fold propensity extremes per pair (overlap under the RF propensity model).
(b) DoubleMLPLR (partially linear = LinearDML's structural twin) with the same RF nuisances,
    same sensitivity_analysis + single-covariate benchmarks, same pass rules, for comparison.
Writes /trial/diag.jsonl.
"""
from __future__ import annotations
import json, os, sys, time, warnings, resource
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from doubleml import DoubleMLData, DoubleMLIRM, DoubleMLPLR
from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

OUT = "/trial/diag.jsonl"
NULL_PAIRS = [("copay_support","treatment_initiated"),("psp_enrolled","treatment_initiated"),("rep_detailing_high","adherent_180d"),("trigger_accepted","adherent_180d"),("rep_detailing_high","low_gap_180d"),("trigger_accepted","low_gap_180d"),("rep_detailing_high","persistent_180d"),("sample_dropped","persistent_180d"),("trigger_accepted","persistent_180d")]
frame = PatientGenerator(GeneratorConfig(seed=21, n_records=1500, brand=Brand.REMIBRUTINIB, dgp_type=DGPType.HETEROGENEOUS)).generate()
truth = frame.attrs["true_ate_by_arm"]
truths = [(a, o, list(ARM_REGISTRY[a].confounders)) for a, outs in truth.items() if a in ARM_REGISTRY and a in frame.columns for o in outs if o in frame.columns]
nulls = [(a, o, list(ARM_REGISTRY[a].confounders)) for a, o in NULL_PAIRS]

def rf_c(): return RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42)
def rf_r(): return RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=42)

def data(treatment, outcome, covs):
    d = frame[[outcome, treatment, *covs]].copy()
    d[outcome] = d[outcome].astype(float); d[treatment] = d[treatment].astype(int)
    for c in covs: d[c] = d[c].astype(float)
    return DoubleMLData(d, y_col=outcome, d_cols=treatment, x_cols=list(covs))

def sens_block_from(m, covs, role):
    sp = m.sensitivity_params
    ci = m.confint(level=0.95)
    bench = {}
    for c in covs:
        np.random.seed(42); b = m.sensitivity_benchmark(benchmarking_set=[c])
        bench[c] = {k: float(b[k].iloc[0]) for k in ("cf_y", "cf_d", "rho", "delta_theta")}
    strongest = max(bench, key=lambda c: bench[c]["cf_d"]); sb = bench[strongest]
    rv, rva = float(sp["rv"][0]), float(sp["rva"][0])
    rule = ({"rv_gt_cf_y": rv > sb["cf_y"], "rv_gt_cf_d": rv > sb["cf_d"], "pass": rv > sb["cf_y"] and rv > sb["cf_d"]} if role == "truth"
            else {"rv_le_0_02": rv <= 0.02, "ci_includes_0": float(ci.iloc[0,0]) <= 0 <= float(ci.iloc[0,1]), "pass": rv <= 0.02})
    return {"ate": float(m.coef[0]), "se": float(m.se[0]), "ci": [float(ci.iloc[0,0]), float(ci.iloc[0,1])], "rv": rv, "rva": rva,
            "strongest_by_cf_d": strongest, "strongest": sb, "per_covariate": bench, "rule": rule}

SEEDS = [42, 1, 2, 3, 4]
with open(OUT, "w") as fh:
    for role, plist in (("truth", truths), ("null", nulls)):
        for a, o, covs in plist:
            rec = {"role": role, "treatment": a, "outcome": o, "covariates": covs}
            dd = data(a, o, covs)
            # (a) fold-seed instability: IRM and PLR ATE / RV over SEEDS (fit + sensitivity_analysis only)
            for name, ctor in (("irm", lambda: DoubleMLIRM(dd, ml_g=rf_c(), ml_m=rf_c(), n_folds=2, n_rep=1, score="ATE")),
                               ("plr", lambda: DoubleMLPLR(dd, ml_l=rf_r(), ml_m=rf_c(), n_folds=2, n_rep=1))):
                rows = []
                for sd in SEEDS:
                    np.random.seed(sd); m = ctor().fit(); m.sensitivity_analysis(); sp = m.sensitivity_params; ci = m.confint()
                    rows.append({"seed": sd, "ate": float(m.coef[0]), "se": float(m.se[0]), "ci": [float(ci.iloc[0,0]), float(ci.iloc[0,1])], "rv": float(sp["rv"][0]), "rva": float(sp["rva"][0])})
                    if name == "irm" and sd == 42:
                        m_hat = np.asarray(m.predictions["ml_m"])[:, 0, 0]
                        rec["irm_propensity_seed42"] = {"min": float(m_hat.min()), "max": float(m_hat.max()),
                            "share_lt_0_01": float((m_hat < 0.01).mean()), "share_gt_0_99": float((m_hat > 0.99).mean()),
                            "share_lt_0_05": float((m_hat < 0.05).mean()), "share_gt_0_95": float((m_hat > 0.95).mean()),
                            "treated_share": float(frame[a].mean()), "n_unique_m_hat": int(len(np.unique(np.round(m_hat, 6))))}
                    if name == "plr" and sd == 42:
                        t0 = time.perf_counter(); rec["plr_seed42"] = sens_block_from(m, covs, role); rec["plr_seed42"]["sens_bench_s"] = round(time.perf_counter() - t0, 2)
                ates = np.array([r["ate"] for r in rows]); rvs = np.array([r["rv"] for r in rows])
                rec[f"{name}_seeds"] = {"rows": rows, "ate_mean": float(ates.mean()), "ate_sd": float(ates.std(ddof=1)), "ate_range": float(ates.max() - ates.min()),
                                       "rv_mean": float(rvs.mean()), "rv_sd": float(rvs.std(ddof=1)), "rv_range": float(rvs.max() - rvs.min()), "se_mean": float(np.mean([r["se"] for r in rows]))}
            if role == "truth": rec["planted_true_ate"] = truth[a][o]["ate"]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(role, a, o, "IRM ate range", round(rec["irm_seeds"]["ate_range"],3), "sd", round(rec["irm_seeds"]["ate_sd"],3), "| PLR ate range", round(rec["plr_seeds"]["ate_range"],3), "sd", round(rec["plr_seeds"]["ate_sd"],3), "| prop", round(rec["irm_propensity_seed42"]["min"],3), round(rec["irm_propensity_seed42"]["max"],3), "| PLR42 rv", round(rec["plr_seed42"]["rv"],3), rec["plr_seed42"]["strongest_by_cf_d"], {k: round(v,3) for k,v in rec["plr_seed42"]["strongest"].items()}, "pass", rec["plr_seed42"]["rule"]["pass"], flush=True)
print("ru_maxrss_mib", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
