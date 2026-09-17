-- ============================================================================
-- Migration 145 (CONTRACT): retire the legacy per_hcp_rollup count columns
-- ============================================================================
-- This is the second half of the expand/contract pair begun by
-- database/migrations/144_per_hcp_trigger_count_columns.sql (canonical TRx lane,
-- codex iter1 HIGH-1, owner-approved 2026-09-18).
--
-- 144 ADDED business_metrics.triggers_{delivered,accepted,total}_count beside the
-- legacy trx_count / nrx_count / total_rx_count, backfilled them, and installed
-- business_metrics_sync_legacy_trigger_counts_trg so either name may be read or
-- written by either code version. This file removes the legacy three and the sync
-- machinery that kept them true.
--
-- ---------------------------------------------------------------------------
-- WHY THIS FILE IS NOT IN database/migrations/ -- DO NOT MOVE IT THERE
-- ---------------------------------------------------------------------------
-- scripts/run_migrations.sh applies EVERY pending forward *.sql in each of its
-- MIGRATION_DIRS in a single pass. A 145_*.sql committed beside 144 would run
-- seconds after it, the legacy columns would be gone before one container had
-- been replaced, and the deploy would be exactly as unsafe as the in-place name
-- swap that HIGH-1 rejected. Expand and contract are only expand/contract if they
-- land in two different deploys.
--
-- database/deferred/ appears in no MIGRATION_DIRS entry, so the runner cannot see
-- this file at all -- a structural separation, not a naming convention that a
-- typo or a new skip-pattern could arm. It is pinned by
-- tests/unit/test_database/test_mig145_contract_legacy_per_hcp_columns.py, which
-- parses MIGRATION_DIRS out of the runner rather than restating it.
--
-- ---------------------------------------------------------------------------
-- APPLY THIS BY HAND, AND ONLY WHEN ALL OF THE FOLLOWING ARE TRUE
-- ---------------------------------------------------------------------------
--   1. the deploy carrying 144 completed and its health checks passed, so no
--      app-only rollback (deploy.yml:1104) can put pre-lane containers back;
--   2. no image that reads or writes the legacy names is still deployable as a
--      rollback target -- on origin/main at the time of the lane there were 52
--      such references, among them src/etl/business_metrics_per_hcp_etl.py's
--      `ON CONFLICT DO UPDATE SET trx_count = EXCLUDED.trx_count`;
--   3. the per-HCP rollup ETL has run at least once on the new code, so the
--      canonical columns are the ones being written.
--
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--     --single-transaction < database/deferred/145_drop_legacy_per_hcp_count_columns.sql
--
-- Verify first, in the same shape, with a trailing ROLLBACK instead of a commit.
-- Afterwards, record it: this file is outside the runner, so nothing writes a
-- public.schema_migrations row for it.
--
--   docker exec -i supabase-db psql -U postgres -d postgres -c \
--     "INSERT INTO public.schema_migrations(filename) VALUES ('deferred/145_drop_legacy_per_hcp_count_columns.sql') ON CONFLICT DO NOTHING;"
--
-- ---------------------------------------------------------------------------
-- THE VIEWS
-- ---------------------------------------------------------------------------
-- The four v_{train,test,validation,holdout}_business_metrics views are
-- `SELECT *` over business_metrics, and PostgreSQL froze that star into an
-- explicit column list when they were created -- a list that still names the
-- legacy columns, so removing a column while a view depends on it fails. Each
-- view is therefore removed explicitly and rebuilt below; never left to a
-- dependency-following removal, which would take the views away and put nothing
-- back.
--
-- Rebuilding re-expands the star against today's table, so each view goes from 29
-- columns to 36: the three canonical counts, plus the seven columns
-- (is_synthetic, email_campaign_count, speaker_program_count, sample_volume,
-- peer_influence_score, patient_support_enrollment, rep_training_score) added to
-- business_metrics AFTER the views were created and never reflected in them.
-- That is a deliberate convergence on the committed schema: a box built from
-- database/core/e2i_ml_complete_v3_schema.sql already gets all 36. Measured
-- 2026-09-18: these views have ZERO consumers outside database/ (src/, tests/,
-- scripts/, feature_repo/, frontend/src), so widening them breaks no query.
--
-- Idempotent: every statement carries IF EXISTS / OR REPLACE, so a second
-- application changes nothing and raises nothing.
-- ----------------------------------------------------------------------------

DROP VIEW IF EXISTS public.v_train_business_metrics;
DROP VIEW IF EXISTS public.v_test_business_metrics;
DROP VIEW IF EXISTS public.v_validation_business_metrics;
DROP VIEW IF EXISTS public.v_holdout_business_metrics;

-- The sync trigger goes before the columns it syncs; its function goes with it.
DROP TRIGGER IF EXISTS business_metrics_sync_legacy_trigger_counts_trg ON public.business_metrics;
DROP FUNCTION IF EXISTS public.business_metrics_sync_legacy_trigger_counts();

ALTER TABLE public.business_metrics DROP COLUMN IF EXISTS trx_count;
ALTER TABLE public.business_metrics DROP COLUMN IF EXISTS nrx_count;
ALTER TABLE public.business_metrics DROP COLUMN IF EXISTS total_rx_count;

CREATE OR REPLACE VIEW public.v_train_business_metrics AS
    SELECT * FROM public.business_metrics WHERE data_split = 'train';
CREATE OR REPLACE VIEW public.v_test_business_metrics AS
    SELECT * FROM public.business_metrics WHERE data_split = 'test';
CREATE OR REPLACE VIEW public.v_validation_business_metrics AS
    SELECT * FROM public.business_metrics WHERE data_split = 'validation';
CREATE OR REPLACE VIEW public.v_holdout_business_metrics AS
    SELECT * FROM public.business_metrics WHERE data_split = 'holdout';

NOTIFY pgrst, 'reload schema';
