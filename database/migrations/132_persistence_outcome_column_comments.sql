-- ============================================================================
-- Migration 132: correct the column comments on patient_journeys.persistent_180d
-- and patient_journeys.discontinued_180d so they describe the shipped data.
-- ============================================================================
-- Migration 064 (M2) documented both columns as "Filtered treatment_initiated=1".
-- That is the RWD semantic only: scripts/convert_optum_rwd.py populates them
-- solely for patients with an initiation date and leaves the rest NULL. The
-- synthetic DGP (src/ml/synthetic/generators/cohort_outcomes.py,
-- generate_discontinuation_outcomes) takes NO treatment_initiated input and
-- draws an outcome for EVERY unit as a function of treatment_arm; the loaders
-- and the platform read paths (cohort_resolution._PJ_COHORTS, the segment
-- loader) apply no initiator filter either. Measured on prod 2026-09-04, where
-- every row is synthetic: 17,186 / 17,186 treatment_initiated=0 rows carry
-- persistent_180d and discontinued_180d, 0 complement violations. The stale
-- comment produced a wrong user-facing definition once (PR #1893, caught in
-- review). Comment-only, idempotent (COMMENT ON replaces), no DDL.
-- ----------------------------------------------------------------------------

COMMENT ON COLUMN patient_journeys.discontinued_180d IS
    'Discontinuation outcome: 1 = stopped therapy within the 180-day window, 0 = still on therapy. Exactly 1 - persistent_180d. NOT restricted to initiators: the synthetic DGP (cohort_outcomes.generate_discontinuation_outcomes) draws it for every row as a function of treatment_arm, regardless of treatment_initiated (prod 2026-09-04: all 17,186 treatment_initiated=0 rows populated). Only RWD-converted rows (convert_optum_rwd.py) carry it for initiators alone and NULL otherwise. Column added by migration 064; comment corrected by migration 132.';

COMMENT ON COLUMN patient_journeys.persistent_180d IS
    'Persistence outcome: 1 = still on therapy at day 180 of the window, 0 = stopped. Exactly 1 - discontinued_180d. NOT restricted to initiators: the synthetic DGP (cohort_outcomes.generate_discontinuation_outcomes) draws it for every row as a function of treatment_arm, regardless of treatment_initiated (prod 2026-09-04: all 17,186 treatment_initiated=0 rows populated). Only RWD-converted rows (convert_optum_rwd.py) carry it for initiators alone and NULL otherwise. Column added by migration 064; comment corrected by migration 132.';

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)
