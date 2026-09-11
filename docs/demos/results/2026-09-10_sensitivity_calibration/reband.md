# Live re-band under the 2026-09-10 sensitivity reading (2026-09-11)

Estimates: 124 (`estimate_source = causal_impact_query`).

## Readings

| reading | runs |
|---|---|
| beyond_measured_confounding | 104 |
| not_applicable_randomized | 6 |
| unbenchmarked | 6 |
| null_finding | 5 |
| unmapped | 3 |

| benchmark basis | runs |
|---|---|
| joint_naive_vs_adjusted | 108 |
| measured_unscoreable | 7 |

| reading / basis | runs |
|---|---|
| beyond_measured_confounding / joint_naive_vs_adjusted | 104 |
| unbenchmarked / measured_unscoreable | 6 |
| null_finding / joint_naive_vs_adjusted | 4 |
| null_finding / measured_unscoreable | 1 |

## Gate moves (today → new)

| move | runs |
|---|---|
| block → block | 6 |
| block → proceed | 56 |
| proceed → proceed | 59 |

## Frame perturbation check

| comparison | runs |
|---|---|
| attempted (mapped, non-randomized) | 115 |
| succeeded — full pull complete | 103 |
| succeeded — full pull CAPPED at 20000 (second bounded subset) | 12 |
| failed — full pull raised | 0 |
| failed — capped pull raised | 0 |
| skipped — randomized | 6 |
| skipped — unmapped | 3 |

Runs whose reading differs between the capped pull and the full brand table: **0**.
The row-order caveat is retired only for the 103 runs compared on a COMPLETE full table with no flip; it stands for the 12 runs whose full pull was capped (<all> treatment_arm→persistent_180d, <all> treatment_initiated→persistent_180d) and for the 0 runs not compared.

## Per pair

| brand | treatment → outcome | runs | today | new | readings | frame-sensitive | capped n / full n | median rr_point | median benchmark |
|---|---|---|---|---|---|---|---|---|---|
| <all> | treatment_arm → persistent_180d | 6 | {'block': 6} | {'proceed': 6} | {'beyond_measured_confounding': 6} | 0 | 1500 / 20000 (capped) | 1.15 | 1.06 |
| <all> | treatment_initiated → persistent_180d | 6 | {'block': 6} | {'proceed': 6} | {'beyond_measured_confounding': 6} | 0 | 1500 / 20000 (capped) | 1.14 | 1.14 |
| Fabhalta | acceptance_status → conversion_flag | 1 | {'block': 1} | {'proceed': 1} | {'beyond_measured_confounding': 1} | 0 | 5000 / 12717 (complete) | 1.40 | 1.02 |
| Fabhalta | control_group_flag → action_taken | 1 | {'proceed': 1} | {'proceed': 1} | {'not_applicable_randomized': 1} | 0 | - | - | - |
| Fabhalta | peer_influence_score → adopted | 1 | {'proceed': 1} | {'proceed': 1} | {'unbenchmarked': 1} | 0 | 5000 / 5000 (complete) | 2.18 | - |
| Fabhalta | treatment_arm → adopted | 1 | {'proceed': 1} | {'proceed': 1} | {'beyond_measured_confounding': 1} | 0 | 5000 / 5000 (complete) | 1.70 | 1.29 |
| Kisqali | acceptance_status → conversion_flag | 4 | {'proceed': 2, 'block': 2} | {'proceed': 4} | {'beyond_measured_confounding': 3, 'null_finding': 1} | 0 | 1500 / 12523 (complete); 5000 / 12523 (complete) | 1.40 | 1.04 |
| Kisqali | accepted → converted | 3 | {'block': 3} | {'?': 3} | {'unmapped': 3} | 0 | - | - | - |
| Kisqali | control_group_flag → action_taken | 3 | {'proceed': 3} | {'proceed': 3} | {'not_applicable_randomized': 3} | 0 | - | - | - |
| Kisqali | copay_support → adherent_180d | 2 | {'proceed': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 5000 / 8730 (complete) | 1.33 | 1.04 |
| Kisqali | copay_support → low_gap_180d | 2 | {'block': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 5000 / 8730 (complete) | 1.32 | 1.06 |
| Kisqali | copay_support → persistent_180d | 2 | {'block': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 5000 / 8730 (complete) | 1.13 | 1.13 |
| Kisqali | disease_stage → persistent_180d | 2 | {'proceed': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 5000 / 8730 (complete) | 1.23 | 1.01 |
| Kisqali | peer_influence_score → adopted | 3 | {'proceed': 2, 'block': 1} | {'proceed': 2, 'block': 1} | {'unbenchmarked': 2, 'null_finding': 1} | 0 | 1500 / 5000 (complete); 5000 / 5000 (complete) | 1.90 | - |
| Kisqali | psp_enrolled → adherent_180d | 2 | {'block': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 5000 / 8730 (complete) | 1.20 | 1.05 |
| Kisqali | psp_enrolled → persistent_180d | 2 | {'block': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 5000 / 8730 (complete) | 1.12 | 1.03 |
| Kisqali | rep_detailing_high → treatment_initiated | 2 | {'block': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 5000 / 8730 (complete) | 1.18 | 1.07 |
| Kisqali | sample_dropped → treatment_initiated | 2 | {'block': 2} | {'proceed': 1, 'block': 1} | {'beyond_measured_confounding': 1, 'null_finding': 1} | 0 | 5000 / 8730 (complete) | 1.09 | 1.06 |
| Kisqali | treatment_arm → adopted | 3 | {'proceed': 3} | {'proceed': 3} | {'beyond_measured_confounding': 3} | 0 | 1500 / 5000 (complete); 5000 / 5000 (complete) | 1.50 | 1.23 |
| Kisqali | treatment_arm → persistent_180d | 2 | {'proceed': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 5000 / 8730 (complete) | 1.28 | 1.01 |
| Kisqali | treatment_arm → treatment_initiated | 2 | {'proceed': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 5000 / 8730 (complete) | 1.54 | 1.26 |
| Kisqali | trigger_accepted → treatment_initiated | 2 | {'block': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 5000 / 8730 (complete) | 1.27 | 1.23 |
| Remibrutinib | acceptance_status → conversion_flag | 2 | {'proceed': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 5000 / 12741 (complete) | 1.47 | 1.02 |
| Remibrutinib | control_group_flag → action_taken | 2 | {'proceed': 2} | {'proceed': 2} | {'not_applicable_randomized': 2} | 0 | - | - | - |
| Remibrutinib | copay_support → adherent_180d | 5 | {'proceed': 4, 'block': 1} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 5000 / 8818 (complete) | 1.32 | 1.05 |
| Remibrutinib | copay_support → low_gap_180d | 5 | {'proceed': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 5000 / 8818 (complete) | 1.39 | 1.04 |
| Remibrutinib | copay_support → persistent_180d | 5 | {'block': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 5000 / 8818 (complete) | 1.14 | 1.12 |
| Remibrutinib | peer_influence_score → adopted | 3 | {'proceed': 3} | {'proceed': 3} | {'unbenchmarked': 3} | 0 | 5000 / 5000 (complete) | 1.98 | - |
| Remibrutinib | psp_enrolled → adherent_180d | 5 | {'block': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 5000 / 8818 (complete) | 1.28 | 1.04 |
| Remibrutinib | psp_enrolled → persistent_180d | 5 | {'block': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 5000 / 8818 (complete) | 1.13 | 1.03 |
| Remibrutinib | rep_detailing_high → treatment_initiated | 5 | {'block': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 5000 / 8818 (complete) | 1.24 | 1.05 |
| Remibrutinib | sample_dropped → treatment_initiated | 5 | {'block': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 5000 / 8818 (complete) | 1.13 | 1.03 |
| Remibrutinib | treatment_arm → adopted | 3 | {'proceed': 3} | {'proceed': 3} | {'beyond_measured_confounding': 3} | 0 | 5000 / 5000 (complete) | 1.57 | 1.31 |
| Remibrutinib | treatment_arm → persistent_180d | 6 | {'block': 6} | {'block': 4, 'proceed': 2} | {'null_finding': 2, 'beyond_measured_confounding': 4} | 0 | 1500 / 8818 (complete); 5000 / 8818 (complete) | 1.09 | 1.04 |
| Remibrutinib | treatment_arm → treatment_initiated | 9 | {'proceed': 9} | {'proceed': 9} | {'beyond_measured_confounding': 9} | 0 | 5000 / 8818 (complete) | 1.61 | 1.24 |
| Remibrutinib | trigger_accepted → treatment_initiated | 5 | {'proceed': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 5000 / 8818 (complete) | 1.37 | 1.19 |
| Remibrutinib | urticaria_severity_uas7 → persistent_180d | 5 | {'proceed': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 5000 / 8818 (complete) | 1.32 | 1.00 |

## Reconciliation with the 2026-09-10 preview (`acceptance_status → conversion_flag`)

Per run, on the route's own frame (the NBA patient JOIN with the brand-scoped curated covariates, designed-NULL `conversion_flag` filled to 0):

| brand | ate | recovered CI | p0 (baseline risk) | naive | rr_point | benchmark | basis | conversion | reading |
|---|---|---|---|---|---|---|---|---|---|
| Fabhalta | 0.0611 | [0.0396, 0.0826] | 0.151 | 0.065 | 1.404 | 1.016 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Kisqali | 0.0350 | [-0.0000, 0.0350] | 0.177 | 0.047 | 1.197 | 1.056 | joint_naive_vs_adjusted | risk_ratio | null_finding |
| Kisqali | 0.0587 | [0.0370, 0.0804] | 0.167 | 0.068 | 1.351 | 1.041 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Kisqali | 0.0743 | [0.0524, 0.0963] | 0.167 | 0.068 | 1.445 | 1.027 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Kisqali | 0.0754 | [0.0533, 0.0976] | 0.167 | 0.068 | 1.452 | 1.032 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Remibrutinib | 0.0726 | [0.0509, 0.0944] | 0.157 | 0.078 | 1.463 | 1.024 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Remibrutinib | 0.0764 | [0.0547, 0.0981] | 0.157 | 0.078 | 1.487 | 1.008 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |

The preview's pull, re-run read-only with its own SQL (`(lower(acceptance_status::text)='accepted')::int`, `conversion_flag::int`, brand filter, `limit n`, no covariates), scored two ways — NULL outcomes DROPPED (the preview's `dropna`) and NULL outcomes FILLED TO 0 (the route's loader) — next to the route frame, with the preview's own reading rule applied to each set of inputs:

| brand | n | ate | preview pull: rows / NULL outcome | p0 / naive (NULL dropped) | preview rule on those | p0 / naive (NULL → 0) | preview rule on those | p0 / naive (route frame) | preview rule on route inputs | current reading |
|---|---|---|---|---|---|---|---|---|---|---|
| Fabhalta | 5000 | 0.0611 | 5000 / 3022 | 0.420 / 0.152 | within | 0.167 / 0.058 | beyond | 0.151 / 0.065 | beyond | beyond_measured_confounding |
| Kisqali | 1500 | 0.0350 | 1500 / 874 | 0.414 / 0.143 | within | 0.171 / 0.063 | beyond | 0.177 / 0.047 | beyond | null_finding |
| Kisqali | 5000 | 0.0587 | 5000 / 2982 | 0.417 / 0.165 | within | 0.168 / 0.067 | beyond | 0.167 / 0.068 | beyond | beyond_measured_confounding |
| Kisqali | 5000 | 0.0743 | 5000 / 2982 | 0.417 / 0.165 | within | 0.168 / 0.067 | beyond | 0.167 / 0.068 | beyond | beyond_measured_confounding |
| Kisqali | 5000 | 0.0754 | 5000 / 2982 | 0.417 / 0.165 | within | 0.168 / 0.067 | beyond | 0.167 / 0.068 | beyond | beyond_measured_confounding |
| Remibrutinib | 5000 | 0.0726 | 5000 / 2974 | 0.368 / 0.192 | within | 0.149 / 0.078 | beyond | 0.157 / 0.078 | beyond | beyond_measured_confounding |
| Remibrutinib | 5000 | 0.0764 | 5000 / 2974 | 0.368 / 0.192 | within | 0.149 / 0.078 | beyond | 0.157 / 0.078 | beyond | beyond_measured_confounding |

Reading of the two tables above: the preview and this script agree on the stored effect and on the conversion; they differ on the INPUTS. The preview's `dropna` removed every trigger whose `conversion_flag` is NULL (a designed NULL — the DB stored-generated `outcome_value > 0` is NULL when no outcome was recorded), so its control-arm rate and naive contrast describe only the triggers with a recorded outcome. The route's loader fills that designed NULL to 0 (`_CAUSAL_FILL_ZERO_OUTCOMES['nba_triggers']`) before the estimator ever sees the frame, which is what produced the stored ATE; filling the raw pull the same way reproduces the route's p0 and naive to the third decimal. The route's pull is the faithful one because it is the frame production estimated on. Whether the preview's own rule reads `within` or `beyond` on each set of inputs is printed per run, so the flip is attributed by measurement, not by argument.

## Caveats

- The CI bound is recovered from the stored `e_value_ci` by the exact algebraic inverse of the old runner's formula. On the standardized branch (a stored, valid `outcome_std`) the inverse uses that SD and the bound is exact. On the unstandardized branch (no valid stored SD) the inverse is taken without an SD — as the old runner computed it — and the SD the new classification needs is measured on the re-pulled frame. 0 runs lacked a stored SD.
- Baseline risk, naive contrast and covariate factors come from a re-pulled frame (not persisted). A `limit N` pull has no guaranteed row order, so each run was classified on the capped pull AND on the full brand table; the frame-sensitive column counts runs whose reading differs, and the perturbation section counts how many runs were actually compared, and on what.
- The full-table pull is bounded by the route's own whole-table ceiling (20000 rows). Where the population exceeds it (the all-brands patient table, 26467 rows) the second frame is another bounded subset, not the population, and the row-order caveat is NOT retired for those pairs; single-brand tables fit under the ceiling and their full pull is complete.
- Pairs listed as `unmapped` have no current dataset mapping and were not guessed.
- Runs whose sensitivity row was SKIPPED (randomized design) keep SKIPPED.
- The per-run covariate set is not persisted; the re-pull uses the brand-scoped curated default the submit route applies. Runs submitted through the discovery path used the SSOT adjustment set, which may differ; the perturbation check covers row order only.
