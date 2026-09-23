# 2026-09-23 owner actions: A/B `--refresh-ab` reload, live feed probe, and the lane-T2 interval decision

Session 1169286c (resumed from `.claude/handoffs/owner-fix-lanes-20260923.md`). Main checkout `63de1f029` (= origin/main, = prod image `e2i-api:63de1f029…`, container StartedAt 19:15:26Z). All commands from `/home/enunez/Projects/e2i_causal_analytics` with `PYTHONPATH=$PWD .venv/bin/dotenv -f .env run --`.

## 1. Adoption re-plant — dry-run DONE, `--execute` BLOCKED (owner)

`scripts/backfill_hcp_treatment_arm.py` (dry-run, `backfill_dryrun.log`) reproduced the handoff's expectations exactly:

| check | expected | measured |
|---|---|---|
| rows derived | 15,000 | 15,000 (5,000 HCPs × 3 brands) |
| treatment_arm match vs live | 1.0000 | 1.0 / 1.0 / 1.0 |
| joined rows Rem/Fab/Kis | 3354/3404/3378 | 3354/3404/3378 |
| clamped to run date | ≈164/185/160 | 164/185/160 |
| future consideration_dates | 0 | 0 |
| prevalence live → new | in band 0.38–0.45 | 0.394→0.389 / 0.418→0.407 / 0.405→0.394 |
| labels flipped | ~9.5 % | 484/479/453 |

Backup of the live labels: `data/backups/hcp_brand_adoption_label_backup_20260923T192739.tsv`.

`--execute` was **denied by the auto-mode classifier** ("Modify Shared Resources") in this agent session. `hcp_brand_adoption` is untouched (`live_state_after_reload.txt`: 15,000 rows, 0 rows dated after 2026-06-01, max updated_at 2026-06-14). The recovery probe (`verify_adoption_channel_recovery.py --live`) and the gold-standard retrain must follow the write, so they were not run.

## 2. A/B substrate reload — DONE and live-verified

`scripts/load_synthetic_data.py --refresh-ab` was allowed and ran 19:29:46–19:32:41Z (`refresh_ab.log`): purge (0 unit_outcomes / 185,532 enrollments / 360 results / 185,532 assignments) → load 360 experiments, 185,532 assignments, 185,532 enrollments, 360 results, 185,532 unit outcomes, **0 failed**.

Live (`live_state_after_reload.txt`):

| table | measured | handoff expectation |
|---|---|---|
| ab_experiment_assignments | 185,532 rows, 5,000 distinct units, `scvhcp_00000..scvhcp_04999`, all in hcp_profiles | 185,532 / ~5,000 / scvhcp_ range / t |
| ab_experiment_unit_outcomes | 185,532 rows, 3 metric_names, 360 experiments | 185,532 / 3 |
| arm × id-parity crosstab | control 46,548/46,167, treatment 46,341/46,476 (flat) | flat (the #2253 confound gone) |

`live_feed_probe.py` (read-only; the REAL `ExperimentOutcomeRepository.load_arrays` for all 360 experiments, `prediction_target` as metric, brand passed, `include_synthetic=True`): `last_outcome_source == 'unit_outcomes'` **360/360**; `mean(t) − mean(c)` equals the stored `effect_estimate` within 1e-9 **360/360**; array sizes equal stored control_n/treatment_n **360/360** → VERDICT PASS (`live_feed_probe.out`). This is the live verification of #2272 (migration 156) and #2253.

## 3. Lane T2 "confirm interval" — ascertained

**Question.** T2 proposed replacing `CausalForestDML.ate_interval` (±0.17 on `adopted`) with a population-ATE interval (±0.035). Narrower is only better if it is *calibrated*. The single assumption to disprove: "the narrower interval covers the planted effect at its nominal rate under the DGP's sampling distribution".

**Source facts (econml 0.16.0).** `ate_interval` → `PopulationSummaryResults.conf_int_mean` with `stderr_mean = sqrt(mean(pred_stderr**2))` — the RMS of the *pointwise* CATE standard errors, documented in econml as "a conservative upper bound", used whenever `mean_pred_stderr is None` (every non-linear final model). `CausalForestDML(discrete_treatment=True)` with the default `drate=True` already computes a doubly-robust ATE and SE **inside `fit`**: `cf.ate_`, `cf.ate_stderr_` = nanmean / nanstd÷√n of per-row DR scores, which remain in `cf.rlearner_model_final_._oob_preds` (verified: nanmean equals `ate_`, 0 NaN rows at n = 3,404).

**Experiment (`interval_mc.py`, read-only, cached live inputs).** Re-derive `adopted` under 30 (Remibrutinib) / 12 / 12 DGP seeds with the planted channel term, fit the twin's *exact* forest (same params, same `_usable_rows` / `_effect_modifier_matrix` / W construction, seed 42) per seed and channel (engagement 0.138, peer 0.093, rep_training NULL), and record three intervals from the same data. Calibration = mean reported SE ÷ empirical SD of the point estimate across seeds; coverage of the planted RD; share of seeds excluding 0. Full table: `interval_mc_summary.txt`, raw: `interval_mc.csv`.

| interval | SE ÷ empirical SD (9 cells) | 95 % coverage | power, planted 0.138 / 0.093 | null excludes 0 |
|---|---|---|---|---|
| shipped `ate_interval` | 3.5 – 6.6 | 1.00 everywhere | 0.43/0.08/0.00 · 0/0/0 | 0/0/0 |
| forest DR `ate_` ± 1.96·`ate_stderr_` | 1.06 – 1.52 (never < 1) | 0.83 – 1.00 | 1.0 in all 6 cells | 0.033 / 0 / 0 |
| LinearDML `ate_inference` (2nd fit) | 0.79 – 1.53 (< 1 in 3/9) | 0.83 – 1.00 | 1.0 in all 6 cells | 0.033 / 0.083 / 0 |

The empirical SD of the point estimate is 0.012–0.022 in every cell, matching the analytic binomial SE (~0.017 at n ≈ 3,400). The shipped interval reports ~0.08 — it is 5× too wide, and it is what makes `SimulationResult.is_significant()` (CI excludes 0 → API `is_significant`) false for every planted channel. The 0.83 coverage cells (Remibrutinib engagement, Kisqali peer) are a **point** bias of +0.02 shared by all three estimators (forest mean-CATE, DR, LinearDML), inside the harness's 0.06 tolerance — not an interval defect.

**Verdict: CONFIRM the interval, using the forest's own doubly-robust `ate_stderr_`.** Report `ate = cf.ate_` with `cf.ate_ ± z·cf.ate_stderr_` (points agree with `mean(effect(x))` within 0.005), keep the CATE-by-region/specialty from `effect(x)`, and for `target_regions` take nanmean/nanstd÷√n of the per-row DR scores over the masked rows (midwest n = 743: SE 0.044 vs the shipped subset half-width 0.157). Reasons over LinearDML: zero extra fit (LinearDML adds a second nuisance fit, +1.2 s per channel), and LinearDML's ratio dips below 1 in 3/9 cells (anti-conservative) while DR never does. Caveats for the lane: `_oob_preds` is an econml private attribute — pin a test that `nanmean(_oob_preds) == ate_` so a version bump is caught; the current `cohort_conversion_outcome` endpoint's intervals narrow the same way, so its `is_significant` flags will flip for existing consumers.

What would reverse this: a DR ratio < 0.9 or null false-positive > 0.10 on the *live* frame after the owner's re-plant (the harness's `--live` run through the twin's loader is T2's acceptance).

## Files
`refresh_ab.log`, `refresh_ab_started_at.txt`, `live_state_after_reload.txt`, `live_feed_probe.{py,out}`, `backfill_dryrun.log`, `interval_mc.py`, `interval_mc_summarize.py`, `interval_mc.csv`, `interval_mc_summary.txt`.
