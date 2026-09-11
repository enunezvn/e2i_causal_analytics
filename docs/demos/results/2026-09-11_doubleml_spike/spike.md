# Lane H (#2008) scratch spike: DoubleML omitted-variable-bias sensitivity, 2026-09-11

**Overall: FAIL** on the four fixed criteria (criteria 1, 2, 4 pass; criterion 3, the RV-separation rule on DoubleMLIRM, fails: 2/11 truths, 4/9 nulls). Supplementary, NOT part of the fixed criteria: the partially-linear `DoubleMLPLR` twin passes 10/11 truths and is ~10x more fold-stable; the IRM failure is an overlap/split-instability property of the interacted score under the production RF propensity on this DGP, not evidence about OVB sensitivity as such.

Environment: capped scratch container (`--cpus=2 --memory=2560m`) from the deployed image `ghcr.io/enunezvn/e2i-api:f4940c97a43dad22bf556b89bcce459afdc7c246`, uid 1000, `/app` read-only, packages `pip install --no-deps --target /trial/site`. Versions: doubleml 0.11.4, numpy 2.3.5, scikit-learn 1.6.1, pandas 2.3.3, python 3.12.14, plotly 7.0.0 (required, see criterion 1). Frame: `PatientGenerator(GeneratorConfig(seed=21, n_records=1500, brand=REMIBRUTINIB, dgp_type=HETEROGENEOUS))`, 11 planted truths from `_planted_pairs`, 9 `NULL_PAIRS`, 0 NaN in any used column (no dropna effect). Nuisances = production config: RandomForest `n_estimators=50, min_samples_leaf=5, random_state=42`; IRM `ml_g` = RandomForestClassifier (binary outcome, IRM uses predict_proba), `ml_m` = RandomForestClassifier, `n_folds=2` (== econml `cv=2`). Reproducibility: doubleml 0.11.4 draws folds with `KFold(shuffle=True)` and no `random_state` (`utils/resampling.py:88`) -> the runner seeds `np.random.seed(42)` before every fit and benchmark refit; the seeded run reproduces bit-for-bit across two executions (two earlier UNSEEDED runs are kept as `spike_run1.jsonl` / `spike_run2_unseeded.jsonl`).

## Criteria

| # | Criterion | Result | Numbers |
|---|---|---|---|
| 1 | `import doubleml` cold (informational) | PASS (recorded) | without plotly: `ModuleNotFoundError: plotly` after 4.597 s / +202.1 MiB; with plotly 7.0.0 (`--no-deps`): 6.182 s, RSS 14.0 -> 247.5 MiB (+233.5 MiB, includes numpy/scipy/sklearn/statsmodels that the API process already carries); incremental after those baseline imports: 0.474 s, +10.5 MiB |
| 2 | IRM fit < 15 s and peak RSS < 400 MiB at n=1500, 2 CPUs | PASS | max fit wall-clock 1.54 s (20 fits); `ru_maxrss` delta per fit <= 0.6 MiB, process `ru_maxrss` after all imports+frame 365.5 MiB, final 388.4 MiB; cgroup `memory.peak` (fresh container, whole run incl. interpreter) 256.9 MiB unseeded run, 352.3 MiB after the two seeded runs, 489.8 MiB after the diagnostics |
| 3 | RV separation (IRM): every truth RV > cf_y AND > cf_d of its strongest declared covariate; every null RV <= 0.02 | **FAIL** | truths 2/11 pass (RV > cf_y on 10/11, RV > cf_d on 3/11); nulls 4/9 pass on RV <= 0.02 (CI includes 0 on 9/9, RVa <= 0.02 on 9/9). The strongest covariate's cf_d is CLAMPED at 1.000 on 13/20 pairs |
| 4 | Total wall-clock, 20 sensitivity runs (informational) | PASS (recorded) | IRM fit + `sensitivity_analysis()` 24.2 s total (1.21 s/pair); single-covariate `sensitivity_benchmark()` refits 49.7 s total (2-3 refits/pair); LinearDML reference 18.0 s; everything 91.9 s for 20 pairs, vs the ~110 s/run refutation budget |

## Per-pair table (canonical seeded IRM run, `spike.jsonl`)

Strongest covariate = largest benchmark `cf_d` among the arm's declared confounders. RV / RVa from `sensitivity_params` (rho = 1, null = 0). Rule: truth passes if RV > cf_y and RV > cf_d; null passes if RV <= 0.02.

| role | treatment -> outcome | planted ATE | IRM ATE [95% CI] | LinearDML ATE [95% CI] | RV | RVa | strongest cov | cf_y | cf_d | rho | delta_theta | rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| truth | treatment_arm -> treatment_initiated | 0.161 | 0.377 [-0.046, 0.800] | 0.153 [0.081, 0.225] | 0.046 | 0.003 | disease_severity | 0.000 | 1.000 | -1.00 | -0.1045 | FAIL |
| truth | treatment_arm -> adherent_180d | 0.243 | 0.475 [0.018, 0.931] | 0.259 [0.188, 0.330] | 0.055 | 0.011 | disease_severity | 0.000 | 1.000 | -1.00 | -0.1877 | FAIL |
| truth | treatment_arm -> low_gap_180d | 0.234 | 0.586 [0.133, 1.039] | 0.243 [0.173, 0.313] | 0.071 | 0.025 | disease_severity | 0.000 | 1.000 | -1.00 | -0.3135 | FAIL |
| truth | copay_support -> adherent_180d | 0.116 | 0.150 [-0.076, 0.375] | 0.121 [0.068, 0.173] | 0.035 | 0.000 | disease_severity | 0.000 | 1.000 | -1.00 | -0.0266 | FAIL |
| truth | copay_support -> low_gap_180d | 0.109 | 0.240 [0.003, 0.478] | 0.099 [0.050, 0.148] | 0.058 | 0.009 | disease_severity | 0.000 | 1.000 | -1.00 | -0.1403 | FAIL |
| truth | copay_support -> persistent_180d | 0.089 | 0.110 [-0.129, 0.348] | 0.115 [0.063, 0.167] | 0.025 | 0.000 | disease_severity | 0.000 | 1.000 | -1.00 | -0.0270 | FAIL |
| truth | psp_enrolled -> adherent_180d | 0.094 | 0.159 [0.084, 0.234] | 0.099 [0.048, 0.150] | 0.198 | 0.131 | disease_severity | 0.035 | 0.000 | 1.00 | 0.0219 | PASS |
| truth | psp_enrolled -> persistent_180d | 0.076 | 0.139 [0.068, 0.209] | 0.088 [0.035, 0.140] | 0.170 | 0.108 | disease_severity | 0.089 | 0.000 | -1.00 | -0.0942 | PASS |
| truth | rep_detailing_high -> treatment_initiated | 0.061 | 0.038 [-0.067, 0.144] | 0.052 [0.004, 0.100] | 0.019 | 0.000 | engagement_score | 0.000 | 1.000 | 1.00 | 0.0347 | FAIL |
| truth | sample_dropped -> treatment_initiated | 0.040 | -0.054 [-0.210, 0.101] | 0.055 [0.007, 0.102] | 0.019 | 0.000 | engagement_score | 0.000 | 1.000 | 1.00 | 0.1153 | FAIL |
| truth | trigger_accepted -> treatment_initiated | 0.066 | 0.019 [-0.041, 0.079] | 0.057 [0.010, 0.104] | 0.025 | 0.000 | disease_severity | 0.232 | 0.000 | 1.00 | 0.0540 | FAIL |
| null | copay_support -> treatment_initiated | 0 (null) | 0.093 [-0.096, 0.283] | 0.017 [-0.031, 0.065] | 0.024 | 0.000 | disease_severity | 0.094 | 1.000 | -0.01 | -0.0156 | FAIL |
| null | psp_enrolled -> treatment_initiated | 0 (null) | 0.023 [-0.041, 0.087] | 0.005 [-0.043, 0.052] | 0.033 | 0.000 | disease_severity | 0.208 | 0.000 | -1.00 | -0.0482 | FAIL |
| null | rep_detailing_high -> adherent_180d | 0 (null) | -0.007 [-0.108, 0.093] | -0.011 [-0.060, 0.039] | 0.004 | 0.000 | engagement_score | 0.000 | 1.000 | 1.00 | 0.0034 | PASS |
| null | trigger_accepted -> adherent_180d | 0 (null) | -0.006 [-0.069, 0.058] | -0.002 [-0.052, 0.049] | 0.007 | 0.000 | disease_severity | 0.047 | 0.000 | -1.00 | -0.0415 | PASS |
| null | rep_detailing_high -> low_gap_180d | 0 (null) | 0.023 [-0.074, 0.120] | -0.005 [-0.052, 0.042] | 0.011 | 0.000 | engagement_score | 0.000 | 1.000 | -1.00 | -0.0118 | PASS |
| null | trigger_accepted -> low_gap_180d | 0 (null) | 0.016 [-0.044, 0.076] | 0.003 [-0.046, 0.052] | 0.020 | 0.000 | disease_severity | 0.071 | 0.000 | -1.00 | -0.0514 | FAIL |
| null | rep_detailing_high -> persistent_180d | 0 (null) | 0.028 [-0.098, 0.155] | 0.031 [-0.020, 0.081] | 0.013 | 0.000 | engagement_score | 0.000 | 1.000 | -1.00 | -0.0195 | PASS |
| null | sample_dropped -> persistent_180d | 0 (null) | 0.121 [-0.053, 0.294] | -0.028 [-0.078, 0.021] | 0.041 | 0.000 | engagement_score | 0.000 | 1.000 | -1.00 | -0.1600 | FAIL |
| null | trigger_accepted -> persistent_180d | 0 (null) | 0.025 [-0.041, 0.092] | -0.009 [-0.060, 0.042] | 0.029 | 0.000 | disease_severity | 0.093 | 0.000 | 1.00 | 0.0232 | FAIL |

## Why IRM fails here (diagnostics, `diag.jsonl`)

- **Fold instability.** With the fold RNG seeded 5 ways (42, 1, 2, 3, 4) per pair: IRM ATE range across seeds up to 0.474 (median per-pair SD 0.056); PLR ATE range up to 0.073 (median SD 0.015). The two unseeded IRM runs of the same script differed by > 0.1 on 3/20 pairs (> 0.2 on 2, max 0.272); LinearDML was bit-identical. RV inherits this: e.g. `treatment_arm -> treatment_initiated` RV 0.043 (run 1) vs 0.011 (run 2).
- **Overlap.** IRM's out-of-fold RF propensity hits the 0.01 trimming floor on 7/20 pairs (all `treatment_arm` and `copay_support` pairs); max propensity 0.69-0.98. The interacted (AIPW) score divides by m(x)(1-m(x)), so the Riesz-representer variance explodes -> intervals 3-5x wider than LinearDML on the same rows, and the benchmark's `cf_d` gain statistic clamps at 1.000 (`disease_severity` / `engagement_score` are the arm-assignment drivers in the generator, so dropping them makes the propensity model collapse). A `cf_d` of 1.0 is unbeatable by any RV, so the truth rule cannot pass on those pairs by construction.
- **The partially-linear twin does not have this problem.** `DoubleMLPLR` (`ml_l` = RF regressor, `ml_m` = RF classifier, same hyper-parameters, `n_folds=2`) is the structural twin of production's `LinearDML` (residual-on-residual), and its benchmark gains stay in (0, 0.12).

## Supplement (NOT a fixed criterion): the same rules on `DoubleMLPLR`, seed 42

Truths 10/11 pass (RV > cf_y on 10/11, RV > cf_d on 10/11). Nulls 3/9 pass on RV <= 0.02 (CI includes 0 on 9/9, RVa <= 0.02 on 9/9).

| role | treatment -> outcome | PLR ATE [95% CI] | RV | RVa | strongest cov | cf_y | cf_d | rho | rule | IRM ATE SD over 5 seeds | PLR ATE SD over 5 seeds | IRM propensity min / max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| truth | treatment_arm -> treatment_initiated | 0.199 [0.135, 0.263] | 0.152 | 0.113 | academic_hcp | 0.040 | 0.035 | 0.23 | PASS | 0.171 | 0.017 | 0.010 / 0.688 |
| truth | treatment_arm -> adherent_180d | 0.289 [0.221, 0.356] | 0.203 | 0.167 | academic_hcp | 0.011 | 0.035 | -0.71 | PASS | 0.156 | 0.023 | 0.010 / 0.688 |
| truth | treatment_arm -> low_gap_180d | 0.256 [0.189, 0.323] | 0.190 | 0.152 | academic_hcp | 0.021 | 0.035 | -0.17 | PASS | 0.186 | 0.019 | 0.010 / 0.688 |
| truth | copay_support -> adherent_180d | 0.114 [0.062, 0.166] | 0.106 | 0.067 | insurance_access_score | 0.006 | 0.018 | -1.00 | PASS | 0.102 | 0.007 | 0.010 / 0.955 |
| truth | copay_support -> low_gap_180d | 0.098 [0.049, 0.148] | 0.096 | 0.057 | insurance_access_score | 0.010 | 0.018 | -0.73 | PASS | 0.097 | 0.013 | 0.010 / 0.955 |
| truth | copay_support -> persistent_180d | 0.082 [0.030, 0.134] | 0.076 | 0.036 | insurance_access_score | 0.041 | 0.018 | -1.00 | PASS | 0.103 | 0.016 | 0.010 / 0.955 |
| truth | psp_enrolled -> adherent_180d | 0.120 [0.069, 0.171] | 0.112 | 0.074 | engagement_score | 0.041 | 0.084 | -0.18 | PASS | 0.035 | 0.019 | 0.031 / 0.907 |
| truth | psp_enrolled -> persistent_180d | 0.096 [0.043, 0.149] | 0.088 | 0.049 | engagement_score | 0.039 | 0.084 | 0.11 | PASS | 0.020 | 0.012 | 0.031 / 0.907 |
| truth | rep_detailing_high -> treatment_initiated | 0.029 [-0.019, 0.076] | 0.030 | 0.000 | academic_hcp | 0.000 | 0.029 | -1.00 | PASS | 0.047 | 0.028 | 0.039 / 0.982 |
| truth | sample_dropped -> treatment_initiated | 0.092 [0.043, 0.141] | 0.092 | 0.052 | academic_hcp | 0.000 | 0.001 | -1.00 | PASS | 0.072 | 0.020 | 0.012 / 0.873 |
| truth | trigger_accepted -> treatment_initiated | 0.049 [0.003, 0.095] | 0.052 | 0.011 | engagement_score | 0.081 | 0.120 | 0.12 | FAIL | 0.016 | 0.013 | 0.048 / 0.930 |
| null | copay_support -> treatment_initiated | 0.021 [-0.027, 0.069] | 0.022 | 0.000 | insurance_access_score | 0.052 | 0.018 | 0.36 | FAIL | 0.109 | 0.014 | 0.010 / 0.955 |
| null | psp_enrolled -> treatment_initiated | 0.011 [-0.038, 0.061] | 0.012 | 0.000 | engagement_score | 0.042 | 0.084 | -0.10 | PASS | 0.024 | 0.013 | 0.031 / 0.907 |
| null | rep_detailing_high -> adherent_180d | -0.007 [-0.055, 0.042] | 0.007 | 0.000 | academic_hcp | 0.021 | 0.029 | -0.17 | PASS | 0.019 | 0.019 | 0.039 / 0.982 |
| null | trigger_accepted -> adherent_180d | -0.026 [-0.076, 0.024] | 0.026 | 0.000 | engagement_score | 0.027 | 0.120 | 0.16 | FAIL | 0.016 | 0.009 | 0.048 / 0.930 |
| null | rep_detailing_high -> low_gap_180d | 0.006 [-0.041, 0.053] | 0.006 | 0.000 | academic_hcp | 0.018 | 0.029 | -0.37 | PASS | 0.075 | 0.017 | 0.039 / 0.982 |
| null | trigger_accepted -> low_gap_180d | -0.021 [-0.069, 0.027] | 0.022 | 0.000 | engagement_score | 0.026 | 0.120 | 0.17 | FAIL | 0.020 | 0.008 | 0.048 / 0.930 |
| null | rep_detailing_high -> persistent_180d | 0.041 [-0.009, 0.091] | 0.041 | 0.000 | academic_hcp | 0.038 | 0.029 | 0.65 | FAIL | 0.045 | 0.013 | 0.039 / 0.982 |
| null | sample_dropped -> persistent_180d | -0.030 [-0.081, 0.021] | 0.029 | 0.000 | academic_hcp | 0.041 | 0.001 | 1.00 | FAIL | 0.064 | 0.012 | 0.012 / 0.873 |
| null | trigger_accepted -> persistent_180d | 0.021 [-0.030, 0.072] | 0.021 | 0.000 | engagement_score | 0.037 | 0.120 | 0.04 | FAIL | 0.017 | 0.018 | 0.048 / 0.930 |

## Reading

- The fixed pass criteria fail, so per the issue the lane should NOT open a worktree on the IRM design. Keep the E-value reading.
- The failure is specific to `DoubleMLIRM` + RF propensity at n=1500 with 2 folds (overlap + split variance), not to OVB sensitivity. If the owner wants the OVB reading reconsidered, the disproof to run is the SAME criteria on `DoubleMLPLR`, which this supplement measured: 10/11 truths pass and the one miss (`trigger_accepted -> treatment_initiated`, planted 0.066, RV 0.052 vs cf_d 0.120) is the smallest-effect pair next to the strongest engagement driver. One of the 10 PLR passes is marginal: `rep_detailing_high -> treatment_initiated` RV 0.030 vs cf_d 0.029 with a CI that includes 0 (RVa = 0), so an RVa-based truth rule would score it 9/11. The null side is the same detection-limit story as the E-value: 5-fold-seed PLR RVs on nulls are 0.006-0.041, so `RV <= 0.02` splits them 3/9 while RVa = 0 and the CI includes 0 on 9/9. Any PLR proposal would need (a) `n_rep > 1` or a fixed seed (the fold RNG is global numpy), (b) a null rule on RVa / CI, not RV, and (c) a re-spike with the production frame, not only the DGP.
- The 10.5 MiB / 0.25 s incremental import (plotly 7.0.0 is a hard dependency of `import doubleml`; `--no-deps` was enough for both) is not a memory concern for the capped container, but plotly would be a new production dependency.

Files: `run_spike.py` (canonical, seeded), `spike.jsonl` (canonical), `spike_run1.jsonl` / `spike_run2_unseeded.jsonl` (unseeded, kept as the instability evidence), `diag.py` / `diag.jsonl` (supplement), `import_probe.py`.
