# Tier-0 Optum-mart INITIATION — events-vs-performance disproof (2026-06-06)

## Question
Does using **~6× more events** (the 50K smoke's 703 → a 300K sample's 4,219) lift model
performance, or is performance **feature-bound** (a ceiling set by the available
pre-index features, not by sample size)?

## Method
- **Cohort:** full-population conversion of the Optum mart → `data/rwd/mart/initiation`
  (`convert_optum_mart.py`, no `--sample-n`): **814,587 patients → 811,286 naive-at-index
  → 787,781 quality-filtered → 11,079 target positives** (1.41% prevalence).
- **Run sample:** stratified-by-target subsample of the full cohort to **300,000 rows /
  4,219 positives** (1.41% prevalence preserved; `random_state=42`) — reused from the
  saved full-pop cohort (no source re-read).
- **Pipeline:** `run_optum_tier0_test.py --cohort initiation_mart --single-model
  --no-bentoml --auc-significance-gate --feature-manifest-source optum_mart`
  (champion = LogisticRegression; `optum_mart` leakage manifest active; HPO trials = 10).

## Run outcome — terminated early (deliberate)
The run was **killed at ~3h15m, during HPO trial 8/10**, on a memory-saturated prod box.
Timeline: Step 2 Data Preparer **1,852 s (~31 min)**; HPO trials 0–7 **~7 min each**; then
**trial 8 took 97 min** (16:57→18:34) as swap pinned at 8.0/8.0 GiB and the run began to
thrash. Root cause of the slowness: **the HPO LR pins `solver="saga"` (issue #232, to allow
L1 trials) and `saga` does not converge here even on standardized data** — it runs all 1,000
epochs every fit (confirmed in the ablation below: scaled `saga` still `n_iter=1000,
converged=False`, while scaled `lbfgs` converges in **20** iters). So every refit (HPO ×10,
plus the final bootstrap) burns the full epoch budget, slower still under swap pressure.
(NB: an earlier note here speculated "features are not standardized" — that is **wrong**;
`fit_preprocessing` *does* apply `StandardScaler` in the tier0 path. See the scaling
ablation below.) No consolidated e2e JSON was produced. **The decision to stop was made
jointly** — the disproof was already conclusively answered by the HPO data (below), so hours
more on a thrashing box had no marginal value.

## Result — feature-bound ceiling CONFIRMED
The HPO objective (the minority-class metric at 1.4% prevalence ≈ precision / PR-AUC) was
**flat across all 9 completed trials**, spanning **5 orders of magnitude of `C`** and both
L1/L2 penalties:

| Trial | C | penalty | objective |
|---|---|---|---|
| 0 | 1.1e-4 | l2 | 0.0316 |
| 1 | 0.075 | l1 | 0.0296 |
| 2 | 0.98 | l1 | 0.0304 |
| 3 | 2.0e-3 | l1 | 0.0312 |
| 4 | 3.47 | l2 | 0.0309 |
| 5 | 14.5 | l1 | **0.0320 (best)** |
| 6 | 8.3e-3 | l2 | 0.0317 |
| 7 | 0.14 | l2 | 0.0300 |
| 8 | 1.1e-4 | l2 | 0.0316 |

**Objective range: 0.0296–0.0320 regardless of hyperparameters.** Tuning does not move it.

### Comparison to the 50K smoke (703 events)
| | 50K smoke (703 events) | 300K (4,219 events) |
|---|---|---|
| Prevalence | 1.41% | 1.41% |
| EPV (events ÷ 64 features) | ~11 | ~66 |
| Best minority objective | ~0.034 (precision@default) | ~0.032 (HPO best) |
| AUC-ROC | 0.637 [95% CI 0.590–0.681] | (not finalized; flat HPO ⇒ no lift expected) |

At **6× the events** (and EPV jumping ~11 → ~66, well clear of the ≥10 floor), the
minority objective is **statistically indistinguishable** from the smoke. **More events did
not raise the ceiling.** The benefit of more events is a *tighter confidence interval*
(more certainty the model is weak), **not a better model** — exactly the EPV prediction: at
703 events the binding constraint was already feature signal, not event count (top feature
`age_at_index` importance 0.006).

## Is it the sampling, or the scaling? (assessed on the FULL population — no, to both)
A full-population scaling ablation was run on the **real split registry** (train 472,668 /
validation 157,556 / test 118,167 / holdout 39,390; prevalence 1.41%; 59 numeric features)
to settle two hypotheses cheaply and faithfully — *not* on a subsample:

| config | converged | val AUC-ROC | val PR-AUC |
|---|---|---|---|
| raw, `saga`, C=1 | ✗ (n_iter=1000) | 0.6713 | 0.0299 |
| scaled, `saga`, C=1 | ✗ (n_iter=1000) | 0.6759 | 0.0294 |
| scaled, `lbfgs`, C=1 | ✓ (n_iter=20) | 0.6760 | 0.0294 |
| scaled, `lbfgs`, C=0.001 → 10 | ✓ | 0.6760–0.6761 | 0.0294–0.0295 |

**Hypothesis #1 — feature scaling / LR non-convergence — DISPROVEN as a performance lever.**
- `fit_preprocessing` (`model_trainer/nodes/preprocessor.py`) **already applies
  `StandardScaler`** in the tier0 path: `tier_0/` has no scaler of its own, and
  `_is_already_preprocessed()` returns **False** on this cohort (only 2% of features fall in
  the "looks pre-scaled" band — the mart mixes huge-σ continuous like
  `enrollment_duration_days` σ≈654 with sparse binary flags σ≈0.1), so `scaling_method`
  stays `"standard"`. Scaling is **on**, not missing.
- Scaling moves the ceiling by **+0.005 AUC** (0.671→0.676) and **0.000 PR-AUC** — negligible.
- The `ConvergenceWarning` is a **runtime/cleanliness wart, not a signal problem**: `saga`
  (pinned for L1 support, issue #232) fails to converge in 1,000 epochs *even when scaled*,
  whereas `lbfgs` on scaled data converges in **20** iters at the **same AUC**. This is the
  real cause of the ~7-min HPO trials → see Recommendation (runtime lever, separable from
  the performance ceiling).

**Hypothesis #2 — sampling masking a signal — DISPROVEN.** This ablation used the **entire
787K cohort's real split** (472K train / 157K val), not the 300K subsample, and lands at the
**same weak ceiling**: AUC ~0.676, PR-AUC ~0.029 (≈2× the 1.41% prevalence baseline). AUC is
**flat across 5 orders of magnitude of `C`** (0.676 throughout), ruling out
regularization/overfitting. Full data → same answer as the subsample → sampling was not
hiding anything.

**What remains (the actual ceiling): the feature SET.** AUC ~0.68 (not 0.5) means there *is*
weak signal — just a low ceiling. The baseline pre-index features (demographics +
comorbidity) plausibly do not capture the true drivers of biologic INITIATION (disease
severity, prior-therapy failures, specialist access, payer authorization) — much of which is
post-index or lives in unlinked raw claims / HCP data.

## Recommendation
- **Do not chase more rows.** The lever is **richer pre-index features**, not sample size,
  scaling, or regularization (all three measured null above).
- **Performance lever (the only one that matters):** engineer additional pre-index signal
  from the **raw Optum claims** (utilization trajectories, prior-Rx sequences, lab/dx
  recency) and the unlinked HCP/market entities — the baseline 64 demographic+comorbidity
  features cap out at AUC ~0.68.
- **Runtime lever (separable, not a performance gain):** the ~7-min HPO trials are caused by
  `saga` running all 1,000 epochs without converging — *even on scaled data*. `lbfgs`
  converges in ~20 iters at identical AUC but cannot do L1. Options: route L2 trials through
  `lbfgs` and reserve `saga` for L1 only; or lower `saga`'s `tol`; or drop L1 from the LR
  search space for this cohort. Any of these makes a full-population HPO tractable on this
  box (the 787K run was swap-bound *because of* the wasted epochs, not the row count per se).
- A fast clean re-run for a finalized AUR/CI: `--hpo-trials 1` (or the lbfgs route above).
- **Operational:** the deployer correctly fail-closes a weak model; this is the honest gate
  refusing a feature-bound model, not a code defect.

## Artifacts
- Full-pop cohort (canonical): `data/rwd/mart/initiation/` (787,781 / 11,079) + backup
  `data/rwd/mart/initiation_fullpop/`; 50K smoke backup `data/rwd/mart/initiation_smoke50k_backup/`.
- Run log: `/tmp/mart_300k.log` (HPO trials 0–8). 50K smoke log: `/tmp/mart_smoke4.log`.
- Cohort: stratified subsample seeded from the full-pop (no e2e JSON — run stopped pre-eval).
- Full-population scaling ablation: `/tmp/scaling_ablation.py` (real split registry; raw vs
  StandardScaler; `saga`/`lbfgs`; C sweep) — the source of the §"Is it the sampling, or the
  scaling?" table.

*Generated 2026-06-06. Branch `feat/optum-mart-multicohort` (multi-cohort build in progress).*
