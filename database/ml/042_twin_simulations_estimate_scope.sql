-- ============================================================================
-- Migration: 042_twin_simulations_estimate_scope.sql
-- Purpose: Persist what a twin simulation's effect was estimated ON (#2053)
-- Dependencies: 012_digital_twin_tables.sql (twin_simulations)
-- ============================================================================
--
-- #2023 scoped a region-filtered simulation's ATE, CI, SE, recommendation and sample size to
-- the filtered regions, and reports the cohort-wide estimate alongside. Neither the scope nor
-- that comparator had a column, so a stored row read back scope-less: on every history read a
-- region-scoped ATE was indistinguishable from a cohort-wide one.
--
-- effect_scope_regions is a tri-state, and the reads depend on all three values:
--   NULL       scope NOT RECORDED: every row written before this migration. Unknown, never
--              cohort-wide. Before #2023 a regions filter sat over a cohort-wide ATE; after it
--              the same filter scopes the ATE; the row alone cannot say which code wrote it.
--   '{}'       estimated on the whole cohort.
--   non-empty  estimated on these regions; cohort_ate / cohort_ci_lower / cohort_ci_upper hold
--              the cohort-wide estimate it was narrowed from.
--
-- No default and no backfill, on purpose: either would stamp legacy rows with a scope nobody
-- recorded. Scope is never derived from population_filters (the request, not the result).
--
-- Additive + idempotent. scripts/run_migrations.sh applies it inside --single-transaction with
-- its ledger row, BEFORE the deploy flips the app services, so the code that writes these
-- columns never runs against a table without them. Rollback: rollback_042.sql, by hand, after
-- the code revert.
-- ============================================================================

ALTER TABLE twin_simulations
    ADD COLUMN IF NOT EXISTS effect_scope_regions text[],
    ADD COLUMN IF NOT EXISTS cohort_ate double precision,
    ADD COLUMN IF NOT EXISTS cohort_ci_lower double precision,
    ADD COLUMN IF NOT EXISTS cohort_ci_upper double precision;

-- A cohort-wide comparator exists only for an estimate that was narrowed to regions; on a
-- cohort-wide or unrecorded row it would be a second headline nobody can place.
ALTER TABLE twin_simulations
    DROP CONSTRAINT IF EXISTS twin_simulations_cohort_comparator_needs_region_scope;
ALTER TABLE twin_simulations
    ADD CONSTRAINT twin_simulations_cohort_comparator_needs_region_scope CHECK (
        (cohort_ate IS NULL AND cohort_ci_lower IS NULL AND cohort_ci_upper IS NULL)
        OR COALESCE(cardinality(effect_scope_regions), 0) > 0
    );

ALTER TABLE twin_simulations
    DROP CONSTRAINT IF EXISTS twin_simulations_cohort_ci_ordered;
ALTER TABLE twin_simulations
    ADD CONSTRAINT twin_simulations_cohort_ci_ordered CHECK (
        cohort_ci_lower IS NULL OR cohort_ci_upper IS NULL OR cohort_ci_lower <= cohort_ci_upper
    );

COMMENT ON COLUMN twin_simulations.effect_scope_regions IS
    'Regions simulated_ate and its interval were estimated on. Empty array = the whole cohort; '
    'NULL = scope not recorded (written before migration 042), which is unknown, not cohort-wide. '
    'Added by migration 042 (#2053).';
COMMENT ON COLUMN twin_simulations.cohort_ate IS
    'Cohort-wide ATE a region-scoped simulated_ate was narrowed from; NULL unless '
    'effect_scope_regions is non-empty. Added by migration 042 (#2053).';
COMMENT ON COLUMN twin_simulations.cohort_ci_lower IS
    'Lower 95% bound of cohort_ate. Added by migration 042 (#2053).';
COMMENT ON COLUMN twin_simulations.cohort_ci_upper IS
    'Upper 95% bound of cohort_ate. Added by migration 042 (#2053).';

-- The pgrst_ddl_watch event trigger already reloads PostgREST's schema cache on DDL; this makes
-- the reload explicit so the API can write the new columns as soon as the transaction commits.
NOTIFY pgrst, 'reload schema';
