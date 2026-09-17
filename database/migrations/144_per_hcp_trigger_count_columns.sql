-- ============================================================================
-- Migration 144 (EXPAND): honest names for the per_hcp_rollup trigger funnel
-- ============================================================================
-- Canonical TRx lane (2026-09-15). business_metrics rows with
-- metric_type = 'per_hcp_rollup' carry three INTEGER columns that migration 033
-- named trx_count / nrx_count / total_rx_count, but the per-HCP ETL
-- (src/etl/business_metrics_per_hcp_etl.py) fills them with TRIGGER counts:
--   trx_count      = COUNT(*) FILTER (delivery_status IN ('delivered','viewed'))
--   nrx_count      = COUNT(*) FILTER (acceptance_status IN ('accepted','responded'))
--   total_rx_count = COUNT(*) of triggers
-- With TRx now a canonical KPI, a column called trx_count holding trigger
-- deliveries is a live mislabel (experiment_outcome mapped primary_metric "trx"
-- to it). Give each count an honest name.
--
-- ---------------------------------------------------------------------------
-- WHY THIS IS AN EXPAND AND NOT A ONE-STEP SWAP (codex iter1 HIGH-1,
-- owner-approved 2026-09-18)
-- ---------------------------------------------------------------------------
-- An earlier draft of this file swapped the three names in place. The deploy
-- cannot survive that:
--   .github/workflows/deploy.yml:956   applies migrations -- OLD containers live
--   .github/workflows/deploy.yml:1044  replaces Feast
--   .github/workflows/deploy.yml:1085  replaces the app services
--   .github/workflows/deploy.yml:1104  on a health failure, restores ONLY the
--                                      app services -- the schema stays migrated
-- Pre-lane code both READS and WRITES the legacy names (52 references on
-- origin/main, among them src/etl/business_metrics_per_hcp_etl.py's
-- `ON CONFLICT DO UPDATE SET trx_count = EXCLUDED.trx_count`). A one-step swap
-- therefore breaks every one of them for the whole replacement window, and
-- PERMANENTLY if the health check fails at :1104, with no automated way back.
--
-- So this half is purely ADDITIVE. It adds the canonical columns beside the
-- legacy ones, backfills them, and installs a BIDIRECTIONAL row trigger, so
-- either name may be read or written by either code version for as long as both
-- versions can run. Nothing is retired here -- not a column, not a view.
--
-- The legacy three are retired by the CONTRACT half,
-- database/deferred/145_drop_legacy_per_hcp_count_columns.sql, applied BY HAND
-- once this deploy is verified. It lives outside every directory listed in
-- scripts/run_migrations.sh's MIGRATION_DIRS precisely so that the runner cannot
-- apply both halves in one pass; tests/unit/test_database/
-- test_mig145_contract_legacy_per_hcp_columns.py pins that separation.
--
-- ---------------------------------------------------------------------------
-- Live census 2026-09-18 (read-only), which is what these statements are aimed at
-- ---------------------------------------------------------------------------
--   * business_metrics holds 22,043 rows; 12,143 are metric_type='per_hcp_rollup'
--     and each legacy column is non-NULL on exactly those 12,143 -- the columns
--     are populated on per-HCP rollup rows and nowhere else, so the backfill
--     below touches 12,143 rows.
--   * all three are nullable INTEGER with no default; no index, constraint, RLS
--     policy or kpi_query_registry statement references any of them, and the
--     table carries no other user trigger.
--   * one function body matched a word-boundary grep,
--     assign_truth_script_conversion; reading it showed the match is its OWN CTE
--     alias (COUNT(*) as nrx_count_window over treatment_events, surfaced as
--     nrx_count), not a reference to business_metrics. A grep hit is a proxy, the
--     body is the capability.
--   * the four v_{train,test,validation,holdout}_business_metrics views are
--     `SELECT *` snapshots, already SEVEN columns behind the table, with ZERO
--     consumers outside database/ (measured across src/, tests/, scripts/,
--     feature_repo/, frontend/src). They keep exposing the legacy names through
--     this deploy -- which is what the pre-lane containers expect -- and the
--     contract half rebuilds them.
--
-- Independent of migration 143: 143 only inserts kpi_query_registry rows, and no
-- registry statement mentions these columns (measured 0). 144 therefore applies
-- correctly on a box where 143 has not run, which today is every box.
--
-- Idempotent throughout: ADD COLUMN IF NOT EXISTS, a backfill whose WHERE excludes
-- rows that already agree, and CREATE OR REPLACE for both the function and the
-- trigger (PostgreSQL 14+; the droplet runs 15.8, confirmed 2026-09-18). A second
-- application changes nothing and raises nothing.
-- ----------------------------------------------------------------------------

-- 1. The canonical columns, as PLAIN STATIC statements on purpose.
-- The hermetic Feast column guard
-- (tests/unit/test_feature_repo/test_data_sources_columns_exist.py) models the
-- canonical schema by text-parsing the files under database/. A dynamic
-- `EXECUTE format('... ADD COLUMN %I ...', ...)` shows that reader nothing but a
-- placeholder, so it would not know business_metrics had gained these columns and
-- would report the renamed Feast source columns as "absent". Keep committed
-- schema additions statically declarable.
ALTER TABLE public.business_metrics ADD COLUMN IF NOT EXISTS triggers_delivered_count INTEGER;
ALTER TABLE public.business_metrics ADD COLUMN IF NOT EXISTS triggers_accepted_count INTEGER;
ALTER TABLE public.business_metrics ADD COLUMN IF NOT EXISTS triggers_total_count INTEGER;

-- 2. Backfill. A freshly added column is NULL on all 12,143 existing rollup rows;
-- without this the new code reads NULL where a real count exists, which is a
-- silently-wrong value -- worse than the mislabel being fixed. The IS DISTINCT
-- FROM clause makes a re-run touch zero rows rather than rewrite the table.
UPDATE public.business_metrics
   SET triggers_delivered_count = trx_count
 WHERE trx_count IS NOT NULL
   AND triggers_delivered_count IS DISTINCT FROM trx_count;

UPDATE public.business_metrics
   SET triggers_accepted_count = nrx_count
 WHERE nrx_count IS NOT NULL
   AND triggers_accepted_count IS DISTINCT FROM nrx_count;

UPDATE public.business_metrics
   SET triggers_total_count = total_rx_count
 WHERE total_rx_count IS NOT NULL
   AND triggers_total_count IS DISTINCT FROM total_rx_count;

-- 3. The bidirectional sync trigger: what makes both names true at once.
-- Old containers write the legacy names, new containers write the canonical ones,
-- and BOTH may be live against this schema -- during the replacement window, and
-- permanently after an app-only rollback at deploy.yml:1104. Whichever side a
-- writer touches, the other follows, so neither code version ever reads a stale
-- value written by the other.
--
-- On INSERT the rule is "fill in whichever side the writer left empty". On UPDATE
-- it is "follow the side that actually changed", which is what distinguishes an
-- old-code write (legacy moved) from a new-code write (canonical moved); when a
-- statement moves both -- no writer in the tree does -- the legacy side wins, so
-- the resolution is deterministic rather than order-dependent.
CREATE OR REPLACE FUNCTION public.business_metrics_sync_legacy_trigger_counts()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $sync$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.triggers_delivered_count IS NULL THEN
            NEW.triggers_delivered_count := NEW.trx_count;
        ELSIF NEW.trx_count IS NULL THEN
            NEW.trx_count := NEW.triggers_delivered_count;
        END IF;

        IF NEW.triggers_accepted_count IS NULL THEN
            NEW.triggers_accepted_count := NEW.nrx_count;
        ELSIF NEW.nrx_count IS NULL THEN
            NEW.nrx_count := NEW.triggers_accepted_count;
        END IF;

        IF NEW.triggers_total_count IS NULL THEN
            NEW.triggers_total_count := NEW.total_rx_count;
        ELSIF NEW.total_rx_count IS NULL THEN
            NEW.total_rx_count := NEW.triggers_total_count;
        END IF;
    ELSE
        IF NEW.trx_count IS DISTINCT FROM OLD.trx_count THEN
            NEW.triggers_delivered_count := NEW.trx_count;
        ELSIF NEW.triggers_delivered_count IS DISTINCT FROM OLD.triggers_delivered_count THEN
            NEW.trx_count := NEW.triggers_delivered_count;
        END IF;

        IF NEW.nrx_count IS DISTINCT FROM OLD.nrx_count THEN
            NEW.triggers_accepted_count := NEW.nrx_count;
        ELSIF NEW.triggers_accepted_count IS DISTINCT FROM OLD.triggers_accepted_count THEN
            NEW.nrx_count := NEW.triggers_accepted_count;
        END IF;

        IF NEW.total_rx_count IS DISTINCT FROM OLD.total_rx_count THEN
            NEW.triggers_total_count := NEW.total_rx_count;
        ELSIF NEW.triggers_total_count IS DISTINCT FROM OLD.triggers_total_count THEN
            NEW.total_rx_count := NEW.triggers_total_count;
        END IF;
    END IF;

    RETURN NEW;
END
$sync$;

CREATE OR REPLACE TRIGGER business_metrics_sync_legacy_trigger_counts_trg
    BEFORE INSERT OR UPDATE ON public.business_metrics
    FOR EACH ROW
    EXECUTE FUNCTION public.business_metrics_sync_legacy_trigger_counts();

-- 4. Say which name is real, so the next reader of \d business_metrics is not
-- left guessing which of the two sides to write.
COMMENT ON COLUMN public.business_metrics.triggers_delivered_count IS
    'per_hcp_rollup: triggers delivered or viewed for this HCP x brand x day (canonical name, migration 144; never prescriptions)';
COMMENT ON COLUMN public.business_metrics.triggers_accepted_count IS
    'per_hcp_rollup: triggers accepted or responded (canonical name, migration 144)';
COMMENT ON COLUMN public.business_metrics.triggers_total_count IS
    'per_hcp_rollup: all triggers generated (canonical name, migration 144)';
COMMENT ON COLUMN public.business_metrics.trx_count IS
    'DEPRECATED alias of triggers_delivered_count. Held in sync by trigger business_metrics_sync_legacy_trigger_counts_trg for pre-lane readers and writers; retired by database/deferred/145. Never prescriptions.';
COMMENT ON COLUMN public.business_metrics.nrx_count IS
    'DEPRECATED alias of triggers_accepted_count. Held in sync by trigger business_metrics_sync_legacy_trigger_counts_trg; retired by database/deferred/145.';
COMMENT ON COLUMN public.business_metrics.total_rx_count IS
    'DEPRECATED alias of triggers_total_count. Held in sync by trigger business_metrics_sync_legacy_trigger_counts_trg; retired by database/deferred/145.';

-- PostgREST caches the schema; reload so the new columns are visible.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
