# Structural author measurement

- lm: `fake` (fake source: replay)
- resolver: `offline`; cohort: `all`; briefs: 91
- measured_at: 2026-09-22T18:21:48.027481+00:00
- tree: commit edb96fbd24ddb63ca6b2e3bb93fa619c86df9af1 (dirty src/scripts/tests: False)

## Score

    FAIL: gate missed_leaks == 0 — missed leaks 28 (rate 0.667 over 42 scored golden-leak features; 91 scored, 0 routed to review, n=91)
    exact role agreement 39/91 (0.429); leak-decision agreement 63/91 (0.692); conservative errors 0
      ancestor    support=13  predicted=5   tp=5   precision=1.00 recall=0.38
      collider    support=14  predicted=8   tp=5   precision=0.62 recall=0.36
      confounder  support=17  predicted=66  tp=17  precision=0.26 recall=1.00
      descendant  support=13  predicted=4   tp=4   precision=1.00 recall=0.31
      instrument  support=19  predicted=6   tp=6   precision=1.00 recall=0.32
      mediator    support=15  predicted=2   tp=2   precision=1.00 recall=0.13
      cohort BC_kisqali: n=30 scored=30 exact=5 missed_leaks=14 review=0
      cohort CSU_remibrutinib: n=31 scored=31 exact=28 missed_leaks=0 review=0
      cohort PNH_fabhalta: n=30 scored=30 exact=6 missed_leaks=14 review=0

## Missed leaks

- BC_kisqali/alive_at_24m_flag: golden collider, derived confounder
- BC_kisqali/continuous_enrollment_24m_flag: golden collider, derived confounder
- BC_kisqali/ctdna_clearance_90d_flag_given_baseline_positive: golden collider, derived confounder
- BC_kisqali/dose_reduced_after_grade3_neutropenia_flag: golden collider, derived confounder
- BC_kisqali/on_treatment_at_12m_flag: golden collider, derived confounder
- BC_kisqali/gcsf_dispense_within_180d_post_index_flag: golden descendant, derived confounder
- BC_kisqali/hepatotoxicity_grade3_post_index_flag: golden descendant, derived confounder
- BC_kisqali/qtc_prolongation_grade2_post_index_flag: golden descendant, derived confounder
- BC_kisqali/switch_ai_to_fulvestrant_within_365d_flag: golden descendant, derived confounder
- BC_kisqali/best_recist_response_180d: golden mediator, derived confounder
- BC_kisqali/ca15_3_pct_change_baseline_to_90d: golden mediator, derived confounder
- BC_kisqali/ctdna_esr1_emergence_flag_90d: golden mediator, derived confounder
- BC_kisqali/post_index_neutropenia_max_grade_90d: golden mediator, derived confounder
- BC_kisqali/ribociclib_relative_dose_intensity_180d: golden mediator, derived confounder
- PNH_fabhalta/alive_at_180d_postindex_flag: golden collider, derived confounder
- PNH_fabhalta/any_rbc_transfusion_during_followup_flag: golden collider, derived confounder
- PNH_fabhalta/facit_fatigue_response_180d_postindex_flag: golden collider, derived confounder
- PNH_fabhalta/iptacopan_persistence_at_180d_flag: golden collider, derived confounder
- PNH_fabhalta/egfr_recovery_delta_365d_postindex: golden descendant, derived confounder
- PNH_fabhalta/hemoglobin_normalization_postindex_flag: golden descendant, derived confounder
- PNH_fabhalta/pnh_related_hospitalizations_365d_postindex_count: golden descendant, derived confounder
- PNH_fabhalta/post_treatment_meningococcal_infection_365d_flag: golden descendant, derived confounder
- PNH_fabhalta/post_treatment_thrombotic_event_365d_flag: golden descendant, derived confounder
- PNH_fabhalta/bilirubin_delta_d90_postindex: golden mediator, derived confounder
- PNH_fabhalta/breakthrough_hemolysis_event_180d_postindex: golden mediator, derived confounder
- PNH_fabhalta/c3_deposition_pnh_rbc_pct_d90_postindex: golden mediator, derived confounder
- PNH_fabhalta/ldh_x_uln_d90_postindex: golden mediator, derived confounder
- PNH_fabhalta/reticulocyte_count_delta_d90_postindex: golden mediator, derived confounder

## Routed to review

- none

## Cost estimate for the real run

- prompt tokens 531375, output tokens 81900 → USD 1.72 at ASSUMED 2.0/8.0 per Mtok

NOTE: a fake-LM run measures the PIPELINE (parse → extract_role → grade → score), not the author. The replayed CSU edges reproduce the committed validation record; the other cohorts' stand-in fragments are not authored claims.
