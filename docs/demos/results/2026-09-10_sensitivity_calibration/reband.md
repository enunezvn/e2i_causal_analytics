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
The row-order caveat is retired only for the 103 runs compared on a COMPLETE full table with the SAME reading on both frames; it stands for the 0 runs whose reading flipped (none), for the 12 runs whose full pull was capped (<all> treatment_arm→persistent_180d, <all> treatment_initiated→persistent_180d) and for the 0 runs not compared.

## Per pair

| brand | treatment → outcome | runs | today | new | readings | frame-sensitive | capped n / full n | median rr_point | median benchmark |
|---|---|---|---|---|---|---|---|---|---|
| <all> | treatment_arm → persistent_180d | 6 | {'block': 6} | {'proceed': 6} | {'beyond_measured_confounding': 6} | 0 | 1500 / 20000 (capped) | 1.15 | 1.06 |
| <all> | treatment_initiated → persistent_180d | 6 | {'block': 6} | {'proceed': 6} | {'beyond_measured_confounding': 6} | 0 | 1500 / 20000 (capped) | 1.14 | 1.14 |
| Fabhalta | acceptance_status → conversion_flag | 1 | {'block': 1} | {'proceed': 1} | {'beyond_measured_confounding': 1} | 0 | 5000 / 12717 (complete) | 1.39 | 1.02 |
| Fabhalta | control_group_flag → action_taken | 1 | {'proceed': 1} | {'proceed': 1} | {'not_applicable_randomized': 1} | 0 | - | - | - |
| Fabhalta | peer_influence_score → adopted | 1 | {'proceed': 1} | {'proceed': 1} | {'unbenchmarked': 1} | 0 | 5000 / 5000 (complete) | 2.18 | - |
| Fabhalta | treatment_arm → adopted | 1 | {'proceed': 1} | {'proceed': 1} | {'beyond_measured_confounding': 1} | 0 | 5000 / 5000 (complete) | 1.70 | 1.29 |
| Kisqali | acceptance_status → conversion_flag | 4 | {'proceed': 2, 'block': 2} | {'proceed': 4} | {'beyond_measured_confounding': 3, 'null_finding': 1} | 0 | 1500 / 12523 (complete); 5000 / 12523 (complete) | 1.42 | 1.06 |
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
| Remibrutinib | acceptance_status → conversion_flag | 2 | {'proceed': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 5000 / 12741 (complete) | 1.50 | 1.07 |
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
| Fabhalta | 0.0611 | [0.0396, 0.0826] | 0.157 | 0.065 | 1.389 | 1.016 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Kisqali | 0.0350 | [-0.0000, 0.0350] | 0.173 | 0.050 | 1.202 | 1.073 | joint_naive_vs_adjusted | risk_ratio | null_finding |
| Kisqali | 0.0587 | [0.0370, 0.0804] | 0.159 | 0.086 | 1.370 | 1.127 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Kisqali | 0.0743 | [0.0524, 0.0963] | 0.159 | 0.086 | 1.469 | 1.052 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Kisqali | 0.0754 | [0.0533, 0.0976] | 0.159 | 0.086 | 1.476 | 1.047 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Remibrutinib | 0.0726 | [0.0509, 0.0944] | 0.149 | 0.091 | 1.487 | 1.084 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Remibrutinib | 0.0764 | [0.0547, 0.0981] | 0.149 | 0.091 | 1.512 | 1.066 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |

The preview's pull, re-run read-only with its own SQL (`(lower(acceptance_status::text)='accepted')::int`, `conversion_flag::int`, brand filter, `limit n`, no covariates), NULL outcomes DROPPED (the preview's `dropna`), read under the CURRENT rule (no covariates, so the benchmark is the joint naive-vs-adjusted one) and under the preview's own rule:

| brand | n | ate | preview pull: rows / NULL outcome / rows scored | recovered CI | p0 (baseline risk) | naive | rr_point | benchmark | basis | conversion | reading (current rule) | preview rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fabhalta | 5000 | 0.0611 | 5000 / 3044 / 1956 | [0.0396, 0.0826] | 0.422 | 0.152 | 1.145 | 1.189 | joint_naive_vs_adjusted | risk_ratio | within_measured_confounding | within |
| Kisqali | 1500 | 0.0350 | 1500 / 890 / 610 | [-0.0000, 0.0350] | 0.424 | 0.113 | 1.083 | 1.169 | joint_naive_vs_adjusted | risk_ratio | null_finding | within |
| Kisqali | 5000 | 0.0587 | 5000 / 2977 / 2023 | [0.0370, 0.0804] | 0.403 | 0.184 | 1.145 | 1.271 | joint_naive_vs_adjusted | risk_ratio | within_measured_confounding | within |
| Kisqali | 5000 | 0.0743 | 5000 / 2977 / 2023 | [0.0524, 0.0963] | 0.403 | 0.184 | 1.184 | 1.229 | joint_naive_vs_adjusted | risk_ratio | within_measured_confounding | within |
| Kisqali | 5000 | 0.0754 | 5000 / 2977 / 2023 | [0.0533, 0.0976] | 0.403 | 0.184 | 1.187 | 1.227 | joint_naive_vs_adjusted | risk_ratio | within_measured_confounding | within |
| Remibrutinib | 5000 | 0.0726 | 5000 / 2965 / 2035 | [0.0509, 0.0944] | 0.376 | 0.192 | 1.193 | 1.267 | joint_naive_vs_adjusted | risk_ratio | within_measured_confounding | within |
| Remibrutinib | 5000 | 0.0764 | 5000 / 2965 / 2035 | [0.0547, 0.0981] | 0.376 | 0.192 | 1.203 | 1.256 | joint_naive_vs_adjusted | risk_ratio | within_measured_confounding | within |

The same pull with NULL outcomes FILLED TO 0 (what the route's loader does for the designed-NULL `conversion_flag`), read under both rules:

| brand | n | ate | preview pull: rows / NULL outcome / rows scored | recovered CI | p0 (baseline risk) | naive | rr_point | benchmark | basis | conversion | reading (current rule) | preview rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fabhalta | 5000 | 0.0611 | 5000 / 3044 / 5000 | [0.0396, 0.0826] | 0.166 | 0.058 | 1.369 | 1.013 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding | beyond |
| Kisqali | 1500 | 0.0350 | 1500 / 890 / 1500 | [-0.0000, 0.0350] | 0.173 | 0.045 | 1.202 | 1.046 | joint_naive_vs_adjusted | risk_ratio | null_finding | beyond |
| Kisqali | 5000 | 0.0587 | 5000 / 2977 / 5000 | [0.0370, 0.0804] | 0.164 | 0.071 | 1.357 | 1.057 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding | beyond |
| Kisqali | 5000 | 0.0743 | 5000 / 2977 / 5000 | [0.0524, 0.0963] | 0.164 | 0.071 | 1.452 | 1.013 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding | beyond |
| Kisqali | 5000 | 0.0754 | 5000 / 2977 / 5000 | [0.0533, 0.0976] | 0.164 | 0.071 | 1.459 | 1.017 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding | beyond |
| Remibrutinib | 5000 | 0.0726 | 5000 / 2965 / 5000 | [0.0509, 0.0944] | 0.151 | 0.082 | 1.480 | 1.043 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding | beyond |
| Remibrutinib | 5000 | 0.0764 | 5000 / 2965 / 5000 | [0.0547, 0.0981] | 0.151 | 0.082 | 1.505 | 1.026 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding | beyond |

Residual between the NULL → 0 preview pull and the route frame, measured on the row sets. The route's NBA join pull is replayed with the route's own helpers (whole-brand paged trigger read, patient join, same coercion and drop rules, request cap applied post-join) carrying `trigger_id`; the preview's SQL is run twice; the replay is run twice:

| brand | n | route frame p0 / naive | replay 1 p0 / naive | replay 2 p0 / naive | preview 1 p0 / naive (NULL → 0) | preview 2 p0 / naive | triggers read / joined / orphans / dropped | preview 1 ∩ preview 2 | replay 1 ∩ replay 2 | preview 1 ∩ replay 1 | values agree on shared rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fabhalta | 5000 | 0.157 / 0.065 | 0.156 / 0.068 | 0.141 / 0.080 | 0.166 / 0.058 | 0.155 / 0.067 | 12717 / 12717 / 0 / 0 | 1149 | 1472 | 1207 | yes |
| Kisqali | 1500 | 0.173 / 0.050 | 0.137 / 0.078 | 0.155 / 0.053 | 0.173 / 0.045 | 0.148 / 0.086 | 12523 / 12523 / 0 / 0 | 0 | 0 | 0 | yes |
| Kisqali | 5000 | 0.159 / 0.086 | 0.167 / 0.055 | 0.163 / 0.061 | 0.164 / 0.071 | 0.167 / 0.069 | 12523 / 12523 / 0 / 0 | 4495 | 1228 | 1511 | yes |
| Remibrutinib | 5000 | 0.149 / 0.091 | 0.158 / 0.085 | 0.164 / 0.068 | 0.151 / 0.082 | 0.161 / 0.074 | 12741 / 12741 / 0 / 0 | 293 | 1602 | 1209 | yes |

Loader facts, read from the route code and the live server: both pulls filter on `brand_id`; neither orders (the route pages the whole brand table with `range()` and no `order()`, the preview uses `limit n` with no `ORDER BY`); the provenance filter is a no-op (`E2I_INCLUDE_SYNTHETIC` is set) on the route path and absent from the preview SQL; `synchronize_seqscans` = on on the server.
Measured cause of the residual: it is row SELECTION, not row VALUES or loader logic. The join adds and drops nothing (every trigger has a patient row and non-NULL covariates; orphans and drops are 0 in every replay); on every trigger shared by the two pulls the treatment and outcome values agree; but the two pulls are different subsets of the same table — at most 1511 of n rows are shared between the preview pull and the route replay, and the SAME pull repeated shares as few as 0 of n rows with itself through psql and at most 1602 through the route's paged read. With `synchronize_seqscans` = on, an unordered `limit n` / `range()` read of this table starts wherever the previous sequential scan left off, so each pull is a different n-row subset and p0 / naive move by sampling variation between subsets (the differences seen here are of the same size as the run-to-run differences of the same pull). The route frame the stored ATE was estimated on was itself one such subset; the reconciliation therefore rests on the NULL fill (preview rule on NULL-dropped inputs: {'within': 7}; on NULL → 0 inputs: {'beyond': 7}) and on what the current rule reads on the NULL → 0 subsets measured ({'beyond_measured_confounding': 6, 'null_finding': 1}), not on any two pulls returning the same rows.

What the tables establish without an equivalence claim: the preview and this script agree on the stored effect and on the conversion; they differ on the INPUTS. The preview's `dropna` removed every trigger whose `conversion_flag` is NULL (a designed NULL — the DB stored-generated `outcome_value > 0` is NULL when no outcome was recorded), so its control-arm rate and naive contrast describe only the triggers with a recorded outcome; on those inputs its own rule reads {'within': 7} and the current rule reads {'within_measured_confounding': 6, 'null_finding': 1}. The route's loader fills that designed NULL to 0 (`_CAUSAL_FILL_ZERO_OUTCOMES['nba_triggers']`) before the estimator ever sees the frame; on NULL → 0 inputs the preview's own rule reads {'beyond': 7} and the current rule reads {'beyond_measured_confounding': 6, 'null_finding': 1} (the current rule reads the interval first, so a run whose CI includes zero is a null finding under it regardless of the benchmark). The route's frame is the faithful one because it is the frame production estimated the stored ATE on.

## Caveats

- The CI bound is recovered from the stored `e_value_ci` by the exact algebraic inverse of the old runner's formula. On the standardized branch (a stored, valid `outcome_std`) the inverse uses that SD and the bound is exact. On the unstandardized branch (no valid stored SD) the inverse is taken without an SD — as the old runner computed it — and the SD the new classification needs is measured on the re-pulled frame. 0 runs lacked a stored SD.
- Baseline risk, naive contrast and covariate factors come from a re-pulled frame (not persisted). A `limit N` pull has no guaranteed row order, so each run was classified on the capped pull AND on the full brand table; the frame-sensitive column counts runs whose reading differs, and the perturbation section counts how many runs were actually compared, and on what.
- The full-table pull is bounded by the route's own whole-table ceiling (20000 rows). Where the population exceeds it (the all-brands patient table, 26467 rows) the second frame is another bounded subset, not the population, and the row-order caveat is NOT retired for those pairs; single-brand tables fit under the ceiling and their full pull is complete.
- Pairs listed as `unmapped` have no current dataset mapping and were not guessed.
- Runs whose sensitivity row was SKIPPED (randomized design) keep SKIPPED.
- The per-run covariate set is not persisted; the re-pull uses the brand-scoped curated default the submit route applies. Runs submitted through the discovery path used the SSOT adjustment set, which may differ; the perturbation check covers row order only.
- Run-to-run movement of this table: between the round-1 table (commit a779170db) and the round-2 table only the two `acceptance_status → conversion_flag` pairs (Fabhalta, Remibrutinib) changed their median rr_point / benchmark; every patient and HCP pair was identical. Cause established as the re-pull, not Task 9b: rr_point depends only on the stored ATE and the frame's p0 under the risk-ratio conversion, Task 9b (50547c566) changed no benchmark arithmetic (it added the `measured_unscoreable` sub-case and its words), and the residual measurement above shows the NBA join pull returns a different row subset on every call, so p0 and the joint benchmark move with each regeneration for that dataset.
