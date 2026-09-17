-- ============================================================================
-- Migration 144: honest names for the per_hcp_rollup trigger funnel counts
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
-- to it). Rename each column to what it counts.
--
-- Dependents, re-censused read-only on the live DB 2026-09-17 (a 2026-09-15
-- census is a snapshot, so it was re-run rather than quoted):
--   * business_metrics + v_{train,test,validation,holdout}_business_metrics hold
--     all three columns and nothing else in the schema does -- 15 relation-columns.
--   * no index, constraint, column default, RLS policy, or kpi_query_registry
--     statement references any of the three.
--   * one function body matched a word-boundary grep,
--     assign_truth_script_conversion; reading it showed the match is its OWN CTE
--     alias (COUNT(*) as nrx_count_window over treatment_events, surfaced as
--     nrx_count), not a reference to business_metrics. Function bodies are stored
--     as text and a rename does not follow them, so this one was read rather than
--     counted -- a grep hit is a proxy, the body is the capability.
--
-- A table column rename keeps the views VALID (they bind by attnum) but their
-- OUTPUT columns keep the old names, so each view column is renamed as well
-- (ALTER VIEW ... RENAME COLUMN, PostgreSQL >= 13; the droplet runs 15.8,
-- confirmed 2026-09-17). Rehearsed in BEGIN/ROLLBACK first.
--
-- Independent of migration 143: 143 only inserts kpi_query_registry rows, and no
-- registry statement mentions these columns (measured 0). 144 therefore applies
-- correctly on a box where 143 has not run, which today is every box.
--
-- Idempotent: every rename is guarded on the old name still existing, so a second
-- application renames nothing and raises nothing.
-- ----------------------------------------------------------------------------

-- The three TABLE renames are written as PLAIN, STATIC statements on purpose.
-- They were previously an `EXECUTE format('... RENAME COLUMN %I TO %I', ...)`
-- inside the loop below, which is invisible to every STATIC reader of database/ —
-- the column names appear only as %I placeholders. The hermetic Feast guard
-- (tests/unit/test_feature_repo/test_data_sources_columns_exist.py) models the
-- canonical schema by text-parsing these files, so a dynamic rename left it
-- believing business_metrics still had trx_count/nrx_count/total_rx_count and
-- reported the renamed Feast source columns as "absent". Keep committed renames
-- statically declarable; PostgreSQL has no RENAME COLUMN IF EXISTS, so each one
-- carries its own existence guard to stay idempotent.
-- The VIEW renames below stay dynamic: they loop over four views, and no static
-- reader models view columns.
DO $tbl$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'business_metrics'
           AND column_name = 'trx_count'
    ) THEN
        ALTER TABLE public.business_metrics RENAME COLUMN trx_count TO triggers_delivered_count;
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'business_metrics'
           AND column_name = 'nrx_count'
    ) THEN
        ALTER TABLE public.business_metrics RENAME COLUMN nrx_count TO triggers_accepted_count;
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'business_metrics'
           AND column_name = 'total_rx_count'
    ) THEN
        ALTER TABLE public.business_metrics RENAME COLUMN total_rx_count TO triggers_total_count;
    END IF;
END $tbl$;

DO $mig$
DECLARE
    pair text[];
    split_view text;
BEGIN
    FOREACH pair SLICE 1 IN ARRAY ARRAY[
        ['trx_count', 'triggers_delivered_count'],
        ['nrx_count', 'triggers_accepted_count'],
        ['total_rx_count', 'triggers_total_count']
    ]::text[] LOOP
        FOREACH split_view IN ARRAY ARRAY[
            'v_train_business_metrics', 'v_test_business_metrics',
            'v_validation_business_metrics', 'v_holdout_business_metrics'
        ] LOOP
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                 WHERE table_schema = 'public' AND table_name = split_view
                   AND column_name = pair[1]
            ) THEN
                EXECUTE format('ALTER VIEW public.%I RENAME COLUMN %I TO %I', split_view, pair[1], pair[2]);
            END IF;
        END LOOP;
    END LOOP;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'business_metrics'
           AND column_name = 'triggers_delivered_count'
    ) THEN
        COMMENT ON COLUMN public.business_metrics.triggers_delivered_count IS
            'per_hcp_rollup: triggers delivered or viewed for this HCP x brand x day (renamed from trx_count by migration 144; never prescriptions)';
        COMMENT ON COLUMN public.business_metrics.triggers_accepted_count IS
            'per_hcp_rollup: triggers accepted or responded (renamed from nrx_count by migration 144)';
        COMMENT ON COLUMN public.business_metrics.triggers_total_count IS
            'per_hcp_rollup: all triggers generated (renamed from total_rx_count by migration 144)';
    END IF;
END
$mig$;

-- PostgREST caches the schema; reload so the renamed columns are visible.
NOTIFY pgrst, 'reload schema';

-- (No COMMIT; psql --single-transaction owns the outer txn.)
