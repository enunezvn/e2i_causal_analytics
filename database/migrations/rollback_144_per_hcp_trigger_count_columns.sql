-- ROLLBACK for migration 144 (canonical TRx lane). NOT a forward migration:
-- scripts/run_migrations.sh skips rollback_*.sql in apply_dir(), which is what
-- makes it safe to ship this file beside 144 -- were it applied in sequence, it
-- would remove the columns 144 had just added and 144 would silently do nothing.
--
-- ---------------------------------------------------------------------------
-- READ THIS BEFORE REACHING FOR IT: IT IS NO LONGER A DEPLOY-RECOVERY STEP
-- ---------------------------------------------------------------------------
-- The previous draft of 144 swapped the three column names in place, so a deploy
-- whose container replacement failed left pre-lane code facing a schema without
-- the names it reads, and THIS file was the way back. 144 is now an EXPAND: it
-- only adds columns, and the legacy trx_count / nrx_count / total_rx_count stay
-- present and correct throughout, kept in sync by
-- business_metrics_sync_legacy_trigger_counts_trg. A failed container
-- replacement therefore needs no schema recovery at all -- the pre-lane
-- containers keep working against the expanded schema, which is the entire point
-- of the expand/contract split (codex iter1 HIGH-1).
--
-- What remains of this file's job is narrow: returning the schema to its exact
-- pre-144 shape, e.g. to re-rehearse the migration or to abandon the lane. It
-- removes ONLY what 144 added -- the trigger, its function, and the three
-- canonical columns. It never touches the legacy columns, which hold the data.
--
-- Because the canonical columns are pure copies while the trigger is live, no
-- information is lost by removing them; re-applying 144 re-derives every value
-- from the legacy columns via its backfill.
--
-- Apply by hand:
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--     --single-transaction < database/migrations/rollback_144_per_hcp_trigger_count_columns.sql
--
-- Guarded the same way as 144 (IF EXISTS on every statement), so it is equally
-- safe to re-run. The trigger is removed BEFORE its function, which depends on it.

DROP TRIGGER IF EXISTS business_metrics_sync_legacy_trigger_counts_trg ON public.business_metrics;
DROP FUNCTION IF EXISTS public.business_metrics_sync_legacy_trigger_counts();

ALTER TABLE public.business_metrics DROP COLUMN IF EXISTS triggers_delivered_count;
ALTER TABLE public.business_metrics DROP COLUMN IF EXISTS triggers_accepted_count;
ALTER TABLE public.business_metrics DROP COLUMN IF EXISTS triggers_total_count;

-- The legacy column comments 144 rewrote are restored to a plain description, so
-- a rolled-back schema does not advertise a deprecation that no longer applies.
COMMENT ON COLUMN public.business_metrics.trx_count IS
    'per_hcp_rollup: triggers delivered or viewed for this HCP x brand x day (misnamed by migration 033; never prescriptions)';
COMMENT ON COLUMN public.business_metrics.nrx_count IS
    'per_hcp_rollup: triggers accepted or responded (misnamed by migration 033)';
COMMENT ON COLUMN public.business_metrics.total_rx_count IS
    'per_hcp_rollup: all triggers generated (misnamed by migration 033)';

-- Retire the ledger row in the SAME transaction as the schema change it records
-- (codex iter3 HIGH-2). The recovery runbook used to do this as a second `psql`
-- invocation after the rollback had already committed: if that second call failed,
-- the schema was reverted while the runner still believed the migration was
-- applied, so the next deploy would skip re-applying it. `&&` supplies ordering,
-- not atomicity. Applied with `psql --single-transaction`, this line commits with
-- the rollback or not at all.
DELETE FROM public.schema_migrations WHERE filename = '144_per_hcp_trigger_count_columns.sql';

NOTIFY pgrst, 'reload schema';
