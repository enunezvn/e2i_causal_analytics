import json, statistics as st
S="/tmp/claude-1000/-home-enunez-Projects-e2i-causal-analytics/4a2f9df9-a443-4a8b-a5cc-6c3494ce10ae/scratchpad/h2008"
recs=[json.loads(l) for l in open(f"{S}/spike.jsonl")]
h=[r for r in recs if r["kind"]=="header"][0]; f=[r for r in recs if r["kind"]=="footer"][0]
P=[r for r in recs if r["kind"]=="pair"]; TR=[r for r in P if r["role"]=="truth"]; NU=[r for r in P if r["role"]=="null"]
D={ (r["treatment"],r["outcome"]): r for r in (json.loads(l) for l in open(f"{S}/diag.jsonl"))}
imp=json.loads(open(f"{S}/import_probe_final.json").read())
cg_run2=256.9; cg_seeded=round(369446912/2**20,1); cg_after_diag=round(513540096/2**20,1)
max_fit=max(r["irm"]["fit_s"] for r in P); max_delta=max(r["irm"]["ru_maxrss_delta_mib"] for r in P); max_abs=max(r["ru_maxrss_now_mib"] for r in P)
t_pass=sum(r["rule"]["pass"] for r in TR); t_y=sum(r["rule"]["rv_gt_cf_y"] for r in TR); t_d=sum(r["rule"]["rv_gt_cf_d"] for r in TR)
n_pass=sum(r["rule"]["pass"] for r in NU); n_ci=sum(r["rule"]["ci_includes_0"] for r in NU); n_rva=sum(r["sensitivity"]["rva"]<=0.02 for r in NU)
sum_fs=round(sum(r["irm"]["fit_s"]+r["sensitivity"]["analysis_s"] for r in P),1); sum_b=round(sum(r["benchmark"]["benchmark_s"] for r in P),1); sum_l=round(sum(r["lineardml"]["fit_s"] for r in P),1); sum_all=round(sum(r["pair_total_s"] for r in P),1)
cfd1=sum(r["benchmark"]["per_covariate"][r["benchmark"]["strongest_by_cf_d"]]["cf_d"]>=0.999 for r in P)
# PLR supplement
plr_t=[D[(r["treatment"],r["outcome"])] for r in TR]; plr_n=[D[(r["treatment"],r["outcome"])] for r in NU]
p_pass=sum(d["plr_seed42"]["rule"]["pass"] for d in plr_t); p_y=sum(d["plr_seed42"]["rule"]["rv_gt_cf_y"] for d in plr_t); p_d=sum(d["plr_seed42"]["rule"]["rv_gt_cf_d"] for d in plr_t)
pn_pass=sum(d["plr_seed42"]["rule"]["pass"] for d in plr_n); pn_ci=sum(d["plr_seed42"]["rule"]["ci_includes_0"] for d in plr_n); pn_rva=sum(d["plr_seed42"]["rva"]<=0.02 for d in plr_n)
irm_sd_med=st.median(D[k]["irm_seeds"]["ate_sd"] for k in D); plr_sd_med=st.median(D[k]["plr_seeds"]["ate_sd"] for k in D)
irm_rng_max=max(D[k]["irm_seeds"]["ate_range"] for k in D); plr_rng_max=max(D[k]["plr_seeds"]["ate_range"] for k in D)
prop_floor=sum(D[k]["irm_propensity_seed42"]["min"]<=0.0100001 for k in D)
ci=lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
L=[]
L.append("# Lane H (#2008) scratch spike: DoubleML omitted-variable-bias sensitivity, 2026-09-11\n")
L.append(f"**Overall: FAIL** on the four fixed criteria (criteria 1, 2, 4 pass; criterion 3, the RV-separation rule on DoubleMLIRM, fails: {t_pass}/11 truths, {n_pass}/9 nulls). Supplementary, NOT part of the fixed criteria: the partially-linear `DoubleMLPLR` twin passes {p_pass}/11 truths and is ~10x more fold-stable; the IRM failure is an overlap/split-instability property of the interacted score under the production RF propensity on this DGP, not evidence about OVB sensitivity as such.\n")
L.append("Environment: capped scratch container (`--cpus=2 --memory=2560m`) from the deployed image "
         f"`{h['versions']['image']}`, uid 1000, `/app` read-only, packages `pip install --no-deps --target /trial/site`. "
         f"Versions: doubleml {h['versions']['doubleml']}, numpy {h['versions']['numpy']}, scikit-learn {h['versions']['sklearn']}, pandas {h['versions']['pandas']}, python {h['versions']['python']}, plotly {imp['plotly']} (required, see criterion 1). "
         "Frame: `PatientGenerator(GeneratorConfig(seed=21, n_records=1500, brand=REMIBRUTINIB, dgp_type=HETEROGENEOUS))`, 11 planted truths from `_planted_pairs`, 9 `NULL_PAIRS`, 0 NaN in any used column (no dropna effect). "
         "Nuisances = production config: RandomForest `n_estimators=50, min_samples_leaf=5, random_state=42`; IRM `ml_g` = RandomForestClassifier (binary outcome, IRM uses predict_proba), `ml_m` = RandomForestClassifier, `n_folds=2` (== econml `cv=2`). "
         "Reproducibility: doubleml 0.11.4 draws folds with `KFold(shuffle=True)` and no `random_state` (`utils/resampling.py:88`) -> the runner seeds `np.random.seed(42)` before every fit and benchmark refit; the seeded run reproduces bit-for-bit across two executions (two earlier UNSEEDED runs are kept as `spike_run1.jsonl` / `spike_run2_unseeded.jsonl`).\n")
L.append("## Criteria\n")
L.append("| # | Criterion | Result | Numbers |\n|---|---|---|---|")
L.append(f"| 1 | `import doubleml` cold (informational) | PASS (recorded) | without plotly: `ModuleNotFoundError: plotly` after {imp['no_plotly']['import_s']} s / +{imp['no_plotly']['rss_delta_mib']} MiB; with plotly {imp['plotly']} (`--no-deps`): {imp['with_plotly']['import_s']} s, RSS {imp['with_plotly']['rss_before_mib']} -> {imp['with_plotly']['rss_after_mib']} MiB (+{imp['with_plotly']['rss_delta_mib']} MiB, includes numpy/scipy/sklearn/statsmodels that the API process already carries); incremental after those baseline imports: {h['doubleml_incremental_import_s']} s, +{h['doubleml_incremental_rss_delta_mib']} MiB |")
L.append(f"| 2 | IRM fit < 15 s and peak RSS < 400 MiB at n=1500, 2 CPUs | PASS | max fit wall-clock {max_fit} s (20 fits); `ru_maxrss` delta per fit <= {max_delta} MiB, process `ru_maxrss` after all imports+frame {h['rss_after_baseline_imports_mib']} MiB, final {f['ru_maxrss_final_mib']} MiB; cgroup `memory.peak` (fresh container, whole run incl. interpreter) {cg_run2} MiB unseeded run, {cg_seeded} MiB after the two seeded runs, {cg_after_diag} MiB after the diagnostics |")
L.append(f"| 3 | RV separation (IRM): every truth RV > cf_y AND > cf_d of its strongest declared covariate; every null RV <= 0.02 | **FAIL** | truths {t_pass}/11 pass (RV > cf_y on {t_y}/11, RV > cf_d on {t_d}/11); nulls {n_pass}/9 pass on RV <= 0.02 (CI includes 0 on {n_ci}/9, RVa <= 0.02 on {n_rva}/9). The strongest covariate's cf_d is CLAMPED at 1.000 on {cfd1}/20 pairs |")
L.append(f"| 4 | Total wall-clock, 20 sensitivity runs (informational) | PASS (recorded) | IRM fit + `sensitivity_analysis()` {sum_fs} s total ({round(sum_fs/20,2)} s/pair); single-covariate `sensitivity_benchmark()` refits {sum_b} s total (2-3 refits/pair); LinearDML reference {sum_l} s; everything {sum_all} s for 20 pairs, vs the ~110 s/run refutation budget |")
L.append("\n## Per-pair table (canonical seeded IRM run, `spike.jsonl`)\n")
L.append("Strongest covariate = largest benchmark `cf_d` among the arm's declared confounders. RV / RVa from `sensitivity_params` (rho = 1, null = 0). Rule: truth passes if RV > cf_y and RV > cf_d; null passes if RV <= 0.02.\n")
L.append("| role | treatment -> outcome | planted ATE | IRM ATE [95% CI] | LinearDML ATE [95% CI] | RV | RVa | strongest cov | cf_y | cf_d | rho | delta_theta | rule |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in P:
    s=r["sensitivity"]; b=r["benchmark"]; sb=b["per_covariate"][b["strongest_by_cf_d"]]; i=r["irm"]; l=r["lineardml"]
    t=f"{r['planted_true_ate']['ate']:.3f}" if r["role"]=="truth" else "0 (null)"
    L.append(f"| {r['role']} | {r['treatment']} -> {r['outcome']} | {t} | {i['ate']:.3f} {ci(i['ci'])} | {l['ate']:.3f} {ci(l['ci'])} | {s['rv']:.3f} | {s['rva']:.3f} | {b['strongest_by_cf_d']} | {sb['cf_y']:.3f} | {sb['cf_d']:.3f} | {sb['rho']:.2f} | {sb['delta_theta']:.4f} | {'PASS' if r['rule']['pass'] else 'FAIL'} |")
L.append("\n## Why IRM fails here (diagnostics, `diag.jsonl`)\n")
L.append(f"- **Fold instability.** With the fold RNG seeded 5 ways (42, 1, 2, 3, 4) per pair: IRM ATE range across seeds up to {irm_rng_max:.3f} (median per-pair SD {irm_sd_med:.3f}); PLR ATE range up to {plr_rng_max:.3f} (median SD {plr_sd_med:.3f}). The two unseeded IRM runs of the same script differed by > 0.2 on 5/20 pairs; LinearDML was bit-identical. RV inherits this: e.g. `treatment_arm -> treatment_initiated` RV 0.043 (run 1) vs 0.011 (run 2).")
L.append(f"- **Overlap.** IRM's out-of-fold RF propensity hits the 0.01 trimming floor on {prop_floor}/20 pairs (all `treatment_arm` and `copay_support` pairs); max propensity 0.69-0.98. The interacted (AIPW) score divides by m(x)(1-m(x)), so the Riesz-representer variance explodes -> intervals 3-5x wider than LinearDML on the same rows, and the benchmark's `cf_d` gain statistic clamps at 1.000 (`disease_severity` / `engagement_score` are the arm-assignment drivers in the generator, so dropping them makes the propensity model collapse). A `cf_d` of 1.0 is unbeatable by any RV, so the truth rule cannot pass on those pairs by construction.")
L.append("- **The partially-linear twin does not have this problem.** `DoubleMLPLR` (`ml_l` = RF regressor, `ml_m` = RF classifier, same hyper-parameters, `n_folds=2`) is the structural twin of production's `LinearDML` (residual-on-residual), and its benchmark gains stay in (0, 0.12).\n")
L.append("## Supplement (NOT a fixed criterion): the same rules on `DoubleMLPLR`, seed 42\n")
L.append(f"Truths {p_pass}/11 pass (RV > cf_y on {p_y}/11, RV > cf_d on {p_d}/11). Nulls {pn_pass}/9 pass on RV <= 0.02 (CI includes 0 on {pn_ci}/9, RVa <= 0.02 on {pn_rva}/9).\n")
L.append("| role | treatment -> outcome | PLR ATE [95% CI] | RV | RVa | strongest cov | cf_y | cf_d | rho | rule | IRM ATE SD over 5 seeds | PLR ATE SD over 5 seeds | IRM propensity min / max |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in P:
    d=D[(r["treatment"],r["outcome"])]; p=d["plr_seed42"]; sb=p["strongest"]; pr=d["irm_propensity_seed42"]
    L.append(f"| {r['role']} | {r['treatment']} -> {r['outcome']} | {p['ate']:.3f} {ci(p['ci'])} | {p['rv']:.3f} | {p['rva']:.3f} | {p['strongest_by_cf_d']} | {sb['cf_y']:.3f} | {sb['cf_d']:.3f} | {sb['rho']:.2f} | {'PASS' if p['rule']['pass'] else 'FAIL'} | {d['irm_seeds']['ate_sd']:.3f} | {d['plr_seeds']['ate_sd']:.3f} | {pr['min']:.3f} / {pr['max']:.3f} |")
L.append("\n## Reading\n")
L.append("- The fixed pass criteria fail, so per the issue the lane should NOT open a worktree on the IRM design. Keep the E-value reading.")
L.append("- The failure is specific to `DoubleMLIRM` + RF propensity at n=1500 with 2 folds (overlap + split variance), not to OVB sensitivity. If the owner wants the OVB reading reconsidered, the disproof to run is the SAME criteria on `DoubleMLPLR`, which this supplement measured: 10/11 truths pass and the one miss (`trigger_accepted -> treatment_initiated`, planted 0.066, RV 0.052 vs cf_d 0.120) is the smallest-effect pair next to the strongest engagement driver. The null side is the same detection-limit story as the E-value: 5-fold-seed PLR RVs on nulls are 0.006-0.041, so `RV <= 0.02` splits them 3/9 while RVa = 0 and the CI includes 0 on 9/9. Any PLR proposal would need (a) `n_rep > 1` or a fixed seed (the fold RNG is global numpy), (b) a null rule on RVa / CI, not RV, and (c) a re-spike with the production frame, not only the DGP.")
L.append("- The 10.5 MiB / 0.25 s incremental import (plotly 7.0.0 is a hard dependency of `import doubleml`; `--no-deps` was enough for both) is not a memory concern for the capped container, but plotly would be a new production dependency.")
L.append("\nFiles: `run_spike.py` (canonical, seeded), `spike.jsonl` (canonical), `spike_run1.jsonl` / `spike_run2_unseeded.jsonl` (unseeded, kept as the instability evidence), `diag.py` / `diag.jsonl` (supplement), `import_probe.py`.")
open(f"{S}/spike.md","w").write("\n".join(L)+"\n"); print("md written", len("\n".join(L)))
