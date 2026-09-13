-- ============================================================================
-- E2I Causal Analytics - ROLLBACK for ml/042_twin_simulations_estimate_scope.sql
-- NOT a forward migration: scripts/run_migrations.sh skips rollback_*.sql. Apply by hand, AFTER
-- the code revert (the #2053 code writes these columns, so without them every simulation insert
-- fails):
--
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--       --single-transaction < database/ml/rollback_042.sql
--
-- Drops the scope columns and their constraints. The scope recorded on rows written since
-- ml/042 is lost; if ml/042 is re-applied, those rows read back as scope unknown.
--
-- Also deletes the ml/042 ledger row, so re-deploying the #2053 code re-applies it.
-- Idempotent: every drop is IF EXISTS, so a second run changes nothing.
-- ============================================================================

ALTER TABLE twin_simulations
    DROP CONSTRAINT IF EXISTS twin_simulations_cohort_comparator_needs_region_scope;
ALTER TABLE twin_simulations
    DROP CONSTRAINT IF EXISTS twin_simulations_cohort_ci_ordered;

ALTER TABLE twin_simulations
    DROP COLUMN IF EXISTS effect_scope_regions,
    DROP COLUMN IF EXISTS cohort_ate,
    DROP COLUMN IF EXISTS cohort_ci_lower,
    DROP COLUMN IF EXISTS cohort_ci_upper;

DELETE FROM public.schema_migrations WHERE filename = 'ml/042_twin_simulations_estimate_scope.sql';

NOTIFY pgrst, 'reload schema';
