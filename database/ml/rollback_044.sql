-- ============================================================================
-- E2I Causal Analytics - ROLLBACK for ml/044_composer_episodes_feedback_id.sql
-- NOT a forward migration: scripts/run_migrations.sh skips rollback_*.sql. Apply by hand, AFTER
-- the code revert (the #2035 linker writes this column, so without it every label write fails):
--
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--       --single-transaction < database/ml/rollback_044.sql
--
-- Drops the claim column. Every recorded rating-to-composition attribution is lost, and it
-- cannot be rebuilt: reconstructing it from the surviving ratings is precisely the defect
-- ml/044 was added to fix. success and feedback_at are untouched, so the episodes stay
-- labelled; they simply stop saying WHICH rating labelled them, and the reverted linker
-- returns to re-deriving that on every run.
--
-- Also deletes the ml/044 ledger row, so re-deploying the #2035 code re-applies it.
-- Idempotent: the drop is IF EXISTS, so a second run changes nothing.
-- ============================================================================

ALTER TABLE public.composer_episodes
    DROP COLUMN IF EXISTS feedback_id;

DELETE FROM public.schema_migrations WHERE filename = 'ml/044_composer_episodes_feedback_id.sql';

NOTIFY pgrst, 'reload schema';
