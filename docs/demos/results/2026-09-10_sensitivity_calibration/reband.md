# Live re-band under the 2026-09-10 sensitivity reading (2026-09-10)

Estimates: 124 (`estimate_source = causal_impact_query`).

## Readings

| reading | runs |
|---|---|
| beyond_measured_confounding | 104 |
| not_applicable_randomized | 6 |
| unbenchmarked | 6 |
| null_finding | 5 |
| unmapped | 3 |

## Gate moves (today → new)

| move | runs |
|---|---|
| block → block | 6 |
| block → proceed | 56 |
| proceed → proceed | 59 |

## Frame perturbation check

Runs whose reading differs between the capped pull and the full brand table: **0** — the row-order caveat is retired by measurement.

## Per pair

| brand | treatment → outcome | runs | today | new | readings | frame-sensitive | median rr_point | median benchmark |
|---|---|---|---|---|---|---|---|---|
| <all> | treatment_arm → persistent_180d | 6 | {'block': 6} | {'proceed': 6} | {'beyond_measured_confounding': 6} | 0 | 1.15 | 1.06 |
| <all> | treatment_initiated → persistent_180d | 6 | {'block': 6} | {'proceed': 6} | {'beyond_measured_confounding': 6} | 0 | 1.14 | 1.14 |
| Fabhalta | acceptance_status → conversion_flag | 1 | {'block': 1} | {'proceed': 1} | {'beyond_measured_confounding': 1} | 0 | 1.39 | 1.09 |
| Fabhalta | control_group_flag → action_taken | 1 | {'proceed': 1} | {'proceed': 1} | {'not_applicable_randomized': 1} | 0 | - | - |
| Fabhalta | peer_influence_score → adopted | 1 | {'proceed': 1} | {'proceed': 1} | {'unbenchmarked': 1} | 0 | 2.18 | - |
| Fabhalta | treatment_arm → adopted | 1 | {'proceed': 1} | {'proceed': 1} | {'beyond_measured_confounding': 1} | 0 | 1.70 | 1.29 |
| Kisqali | acceptance_status → conversion_flag | 4 | {'proceed': 2, 'block': 2} | {'proceed': 4} | {'beyond_measured_confounding': 3, 'null_finding': 1} | 0 | 1.40 | 1.04 |
| Kisqali | accepted → converted | 3 | {'block': 3} | {'?': 3} | {'unmapped': 3} | 0 | - | - |
| Kisqali | control_group_flag → action_taken | 3 | {'proceed': 3} | {'proceed': 3} | {'not_applicable_randomized': 3} | 0 | - | - |
| Kisqali | copay_support → adherent_180d | 2 | {'proceed': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 1.33 | 1.04 |
| Kisqali | copay_support → low_gap_180d | 2 | {'block': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 1.32 | 1.06 |
| Kisqali | copay_support → persistent_180d | 2 | {'block': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 1.13 | 1.13 |
| Kisqali | disease_stage → persistent_180d | 2 | {'proceed': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 1.23 | 1.01 |
| Kisqali | peer_influence_score → adopted | 3 | {'proceed': 2, 'block': 1} | {'proceed': 2, 'block': 1} | {'unbenchmarked': 2, 'null_finding': 1} | 0 | 1.90 | - |
| Kisqali | psp_enrolled → adherent_180d | 2 | {'block': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 1.20 | 1.05 |
| Kisqali | psp_enrolled → persistent_180d | 2 | {'block': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 1.12 | 1.03 |
| Kisqali | rep_detailing_high → treatment_initiated | 2 | {'block': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 1.18 | 1.07 |
| Kisqali | sample_dropped → treatment_initiated | 2 | {'block': 2} | {'proceed': 1, 'block': 1} | {'beyond_measured_confounding': 1, 'null_finding': 1} | 0 | 1.09 | 1.06 |
| Kisqali | treatment_arm → adopted | 3 | {'proceed': 3} | {'proceed': 3} | {'beyond_measured_confounding': 3} | 0 | 1.50 | 1.23 |
| Kisqali | treatment_arm → persistent_180d | 2 | {'proceed': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 1.28 | 1.01 |
| Kisqali | treatment_arm → treatment_initiated | 2 | {'proceed': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 1.54 | 1.26 |
| Kisqali | trigger_accepted → treatment_initiated | 2 | {'block': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 1.27 | 1.23 |
| Remibrutinib | acceptance_status → conversion_flag | 2 | {'proceed': 2} | {'proceed': 2} | {'beyond_measured_confounding': 2} | 0 | 1.49 | 1.08 |
| Remibrutinib | control_group_flag → action_taken | 2 | {'proceed': 2} | {'proceed': 2} | {'not_applicable_randomized': 2} | 0 | - | - |
| Remibrutinib | copay_support → adherent_180d | 5 | {'proceed': 4, 'block': 1} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 1.32 | 1.05 |
| Remibrutinib | copay_support → low_gap_180d | 5 | {'proceed': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 1.39 | 1.04 |
| Remibrutinib | copay_support → persistent_180d | 5 | {'block': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 1.14 | 1.12 |
| Remibrutinib | peer_influence_score → adopted | 3 | {'proceed': 3} | {'proceed': 3} | {'unbenchmarked': 3} | 0 | 1.98 | - |
| Remibrutinib | psp_enrolled → adherent_180d | 5 | {'block': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 1.28 | 1.04 |
| Remibrutinib | psp_enrolled → persistent_180d | 5 | {'block': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 1.13 | 1.03 |
| Remibrutinib | rep_detailing_high → treatment_initiated | 5 | {'block': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 1.24 | 1.05 |
| Remibrutinib | sample_dropped → treatment_initiated | 5 | {'block': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 1.13 | 1.03 |
| Remibrutinib | treatment_arm → adopted | 3 | {'proceed': 3} | {'proceed': 3} | {'beyond_measured_confounding': 3} | 0 | 1.57 | 1.31 |
| Remibrutinib | treatment_arm → persistent_180d | 6 | {'block': 6} | {'block': 4, 'proceed': 2} | {'null_finding': 2, 'beyond_measured_confounding': 4} | 0 | 1.09 | 1.04 |
| Remibrutinib | treatment_arm → treatment_initiated | 9 | {'proceed': 9} | {'proceed': 9} | {'beyond_measured_confounding': 9} | 0 | 1.61 | 1.24 |
| Remibrutinib | trigger_accepted → treatment_initiated | 5 | {'proceed': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 1.37 | 1.19 |
| Remibrutinib | urticaria_severity_uas7 → persistent_180d | 5 | {'proceed': 5} | {'proceed': 5} | {'beyond_measured_confounding': 5} | 0 | 1.32 | 1.00 |

## Caveats

- The CI bound is recovered EXACTLY from the stored `e_value_ci` and the stored `outcome_std` (algebraic inverse of the old formula); the reading needs only that bound and whether the CI includes zero.
- Baseline risk, naive contrast and covariate factors come from a re-pulled frame (not persisted). A `limit N` pull has no guaranteed row order, so each run was classified on the capped pull AND on the full brand table; the frame-sensitive column counts runs whose reading differs.
- The full-table pull is capped at the route's own whole-table ceiling (20000 rows); every single-brand table fits under it, the all-brands patient table does not.
- Pairs listed as `unmapped` have no current dataset mapping and were not guessed.
- Runs whose sensitivity row was SKIPPED (randomized design) keep SKIPPED.
