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
| Fabhalta | acceptance_status → conversion_flag | 1 | {'block': 1} | {'proceed': 1} | {'beyond_measured_confounding': 1} | 0 | 5000 / 12717 (complete) | 1.36 | 1.04 |
| Fabhalta | control_group_flag → action_taken | 1 | {'proceed': 1} | {'proceed': 1} | {'not_applicable_randomized': 1} | 0 | - | - | - |
| Fabhalta | peer_influence_score → adopted | 1 | {'proceed': 1} | {'proceed': 1} | {'unbenchmarked': 1} | 0 | 5000 / 5000 (complete) | 2.18 | - |
| Fabhalta | treatment_arm → adopted | 1 | {'proceed': 1} | {'proceed': 1} | {'beyond_measured_confounding': 1} | 0 | 5000 / 5000 (complete) | 1.70 | 1.29 |
| Kisqali | acceptance_status → conversion_flag | 4 | {'proceed': 2, 'block': 2} | {'proceed': 4} | {'beyond_measured_confounding': 3, 'null_finding': 1} | 0 | 1500 / 12523 (complete); 5000 / 12523 (complete) | 1.39 | 1.07 |
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
| Remibrutinib | acceptance_status → conversion_flag | 2 | {'proceed': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 5000 / 12741 (complete) | 1.48 | 1.04 |
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
| Fabhalta | 0.0611 | [0.0396, 0.0826] | 0.168 | 0.071 | 1.364 | 1.042 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Kisqali | 0.0350 | [-0.0000, 0.0350] | 0.172 | 0.073 | 1.204 | 1.186 | joint_naive_vs_adjusted | risk_ratio | null_finding |
| Kisqali | 0.0587 | [0.0370, 0.0804] | 0.168 | 0.060 | 1.348 | 1.005 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Kisqali | 0.0743 | [0.0524, 0.0963] | 0.168 | 0.060 | 1.441 | 1.063 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Kisqali | 0.0754 | [0.0533, 0.0976] | 0.168 | 0.060 | 1.448 | 1.068 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Remibrutinib | 0.0726 | [0.0509, 0.0944] | 0.156 | 0.083 | 1.467 | 1.044 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |
| Remibrutinib | 0.0764 | [0.0547, 0.0981] | 0.156 | 0.083 | 1.492 | 1.027 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding |

The preview's pull, re-run read-only with its own SQL (`(lower(acceptance_status::text)='accepted')::int`, `conversion_flag::int`, brand filter, `limit n`, no covariates), NULL outcomes DROPPED (the preview's `dropna`), read under the CURRENT rule (no covariates, so the benchmark is the joint naive-vs-adjusted one) and under the preview's own rule:

| brand | n | ate | preview pull: rows / NULL outcome / rows scored | recovered CI | p0 (baseline risk) | naive | rr_point | benchmark | basis | conversion | reading (current rule) | preview rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fabhalta | 5000 | 0.0611 | 5000 / 3027 / 1973 | [0.0396, 0.0826] | 0.406 | 0.164 | 1.150 | 1.221 | joint_naive_vs_adjusted | risk_ratio | within_measured_confounding | within |
| Kisqali | 1500 | 0.0350 | 1500 / 890 / 610 | [-0.0000, 0.0350] | 0.392 | 0.173 | 1.089 | 1.323 | joint_naive_vs_adjusted | risk_ratio | null_finding | within |
| Kisqali | 5000 | 0.0587 | 5000 / 2977 / 2023 | [0.0370, 0.0804] | 0.404 | 0.189 | 1.145 | 1.282 | joint_naive_vs_adjusted | risk_ratio | within_measured_confounding | within |
| Kisqali | 5000 | 0.0743 | 5000 / 2977 / 2023 | [0.0524, 0.0963] | 0.404 | 0.189 | 1.184 | 1.240 | joint_naive_vs_adjusted | risk_ratio | within_measured_confounding | within |
| Kisqali | 5000 | 0.0754 | 5000 / 2977 / 2023 | [0.0533, 0.0976] | 0.404 | 0.189 | 1.187 | 1.237 | joint_naive_vs_adjusted | risk_ratio | within_measured_confounding | within |
| Remibrutinib | 5000 | 0.0726 | 5000 / 2975 / 2025 | [0.0509, 0.0944] | 0.376 | 0.191 | 1.193 | 1.265 | joint_naive_vs_adjusted | risk_ratio | within_measured_confounding | within |
| Remibrutinib | 5000 | 0.0764 | 5000 / 2975 / 2025 | [0.0547, 0.0981] | 0.376 | 0.191 | 1.203 | 1.254 | joint_naive_vs_adjusted | risk_ratio | within_measured_confounding | within |

The same pull with NULL outcomes FILLED TO 0 (what the route's loader does for the designed-NULL `conversion_flag`), read under both rules:

| brand | n | ate | preview pull: rows / NULL outcome / rows scored | recovered CI | p0 (baseline risk) | naive | rr_point | benchmark | basis | conversion | reading (current rule) | preview rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fabhalta | 5000 | 0.0611 | 5000 / 3027 / 5000 | [0.0396, 0.0826] | 0.160 | 0.066 | 1.382 | 1.022 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding | beyond |
| Kisqali | 1500 | 0.0350 | 1500 / 890 / 1500 | [-0.0000, 0.0350] | 0.155 | 0.082 | 1.226 | 1.248 | joint_naive_vs_adjusted | risk_ratio | null_finding | within |
| Kisqali | 5000 | 0.0587 | 5000 / 2977 / 5000 | [0.0370, 0.0804] | 0.166 | 0.070 | 1.353 | 1.049 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding | beyond |
| Kisqali | 5000 | 0.0743 | 5000 / 2977 / 5000 | [0.0524, 0.0963] | 0.166 | 0.070 | 1.447 | 1.019 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding | beyond |
| Kisqali | 5000 | 0.0754 | 5000 / 2977 / 5000 | [0.0533, 0.0976] | 0.166 | 0.070 | 1.454 | 1.024 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding | beyond |
| Remibrutinib | 5000 | 0.0726 | 5000 / 2975 / 5000 | [0.0509, 0.0944] | 0.150 | 0.083 | 1.484 | 1.047 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding | beyond |
| Remibrutinib | 5000 | 0.0764 | 5000 / 2975 / 5000 | [0.0547, 0.0981] | 0.150 | 0.083 | 1.510 | 1.029 | joint_naive_vs_adjusted | risk_ratio | beyond_measured_confounding | beyond |

Residual between the NULL → 0 preview pull and the route frame, measured on the row sets. The route's NBA join pull is replayed with the route's own helpers (whole-brand paged trigger read, patient join, same coercion and drop rules, request cap applied post-join) carrying `trigger_id`; the preview's SQL is run twice; the replay is run twice:

| brand | n | route frame p0 / naive | replay 1 p0 / naive | replay 2 p0 / naive | preview 1 p0 / naive (NULL → 0) | preview 2 p0 / naive | triggers read / joined / orphans / dropped | preview 1 ∩ preview 2 | replay 1 ∩ replay 2 | preview 1 ∩ replay 1 | values agree on shared rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fabhalta | 5000 | 0.168 / 0.071 | 0.155 / 0.081 | 0.153 / 0.071 | 0.160 / 0.066 | 0.158 / 0.062 | 12717 / 12717 / 0 (both replays) / 0 (both replays) | 1076 | 1531 | 1251 | yes |
| Kisqali | 1500 | 0.172 / 0.073 | 0.150 / 0.066 | 0.154 / 0.060 | 0.155 / 0.082 | 0.158 / 0.087 | 12523 / 12523 / 0 (both replays) / 0 (both replays) | 0 | 0 | 0 | no shared rows to compare |
| Kisqali | 5000 | 0.168 / 0.060 | 0.161 / 0.067 | 0.160 / 0.058 | 0.166 / 0.070 | 0.165 / 0.075 | 12523 / 12523 / 0 (both replays) / 0 (both replays) | 4593 | 1264 | 1414 | yes |
| Remibrutinib | 5000 | 0.156 / 0.083 | 0.154 / 0.092 | 0.166 / 0.063 | 0.150 / 0.083 | 0.158 / 0.074 | 12741 / 12741 / 0 (both replays) / 0 (both replays) | 308 | 1622 | 1187 | yes |

Loader facts, read from the route code and the live server: both pulls filter on `brand_id`; neither orders (the route pages the whole brand table with `range()` and no `order()`, the preview uses `limit n` with no `ORDER BY`); the provenance filter is a no-op (`E2I_INCLUDE_SYNTHETIC` is set) on the route path and absent from the preview SQL; `synchronize_seqscans` = on on the server.
Measured cause of the residual: it is row SELECTION, not row VALUES or loader logic. In the 1500 / 5000 rows inspected per pull the join added and dropped nothing (each inspected trigger had a patient row and non-NULL covariates; orphans and drops are 0 in both replays of every pair); on every trigger shared by two pulls the treatment and outcome values agree (Kisqali n=1500 had no shared rows to compare and are excluded from that statement); but the pulls are different subsets of the same table — at most 1414 of n rows are shared between the preview pull and the route replay, and the SAME pull repeated shares as few as 0 of n rows with itself through psql and at most 1622 through the route's paged read. With `synchronize_seqscans` = on, an unordered `limit n` / `range()` read of this table starts wherever the previous sequential scan left off, so each pull is a different n-row subset and p0 / naive move by sampling variation between subsets (the differences seen here are of the same size as the run-to-run differences of the same pull). The route frame the stored ATE was estimated on was itself one such subset; the reconciliation therefore rests on the NULL fill (preview rule on NULL-dropped inputs: {'within': 7}; on NULL → 0 inputs: {'beyond': 6, 'within': 1}) and on what the current rule reads on the NULL → 0 subsets measured ({'beyond_measured_confounding': 6, 'null_finding': 1}), not on any two pulls returning the same rows.

What the tables establish without an equivalence claim: the preview and this script agree on the stored effect and on the conversion; they differ on the INPUTS. The preview's `dropna` removed every trigger whose `conversion_flag` is NULL (a designed NULL — the DB stored-generated `outcome_value > 0` is NULL when no outcome was recorded), so its control-arm rate and naive contrast describe only the triggers with a recorded outcome; on those inputs its own rule reads {'within': 7} and the current rule reads {'within_measured_confounding': 6, 'null_finding': 1}. The route's loader fills that designed NULL to 0 (`_CAUSAL_FILL_ZERO_OUTCOMES['nba_triggers']`) before the estimator ever sees the frame; on NULL → 0 inputs the preview's own rule reads {'beyond': 6, 'within': 1} and the current rule reads {'beyond_measured_confounding': 6, 'null_finding': 1} (the current rule reads the interval first, so a run whose CI includes zero is a null finding under it regardless of the benchmark). The route's frame is the faithful one because it is the frame production estimated the stored ATE on.

## Caveats

- The CI bound is recovered from the stored `e_value_ci` by the exact algebraic inverse of the old runner's formula. On the standardized branch (a stored, valid `outcome_std`) the inverse uses that SD and the bound is exact. On the unstandardized branch (no valid stored SD) the inverse is taken without an SD — as the old runner computed it — and the SD the new classification needs is measured on the re-pulled frame. 0 runs lacked a stored SD.
- Baseline risk, naive contrast and covariate factors come from a re-pulled frame (not persisted). A `limit N` pull has no guaranteed row order, so each run was classified on the capped pull AND on the full brand table; the frame-sensitive column counts runs whose reading differs, and the perturbation section counts how many runs were actually compared, and on what.
- The full-table pull is bounded by the route's own whole-table ceiling (20000 rows). Where the population exceeds it (the all-brands patient table, 26467 rows) the second frame is another bounded subset, not the population, and the row-order caveat is NOT retired for those pairs; single-brand tables fit under the ceiling and their full pull is complete.
- Pairs listed as `unmapped` have no current dataset mapping and were not guessed.
- Runs whose sensitivity row was SKIPPED (randomized design) keep SKIPPED.
- The per-run covariate set is not persisted; the re-pull uses the brand-scoped curated default the submit route applies. Runs submitted through the discovery path used the SSOT adjustment set, which may differ; the perturbation check covers row order only.
- Run-to-run movement of this table: between the round-1 table (commit a779170db) and the round-2 table only the two `acceptance_status → conversion_flag` pairs (Fabhalta, Remibrutinib) changed their median rr_point / benchmark; every patient and HCP pair was identical. Cause established as the re-pull, not Task 9b: rr_point depends only on the stored ATE and the frame's p0 under the risk-ratio conversion, Task 9b (50547c566) changed no benchmark arithmetic (it added the `measured_unscoreable` sub-case and its words), and the residual measurement above shows the NBA join pull returns a different row subset on every call, so p0 and the joint benchmark move with each regeneration for that dataset.
