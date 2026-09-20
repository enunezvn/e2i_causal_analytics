-- #2167 post-deploy confirmation probe. READ-ONLY: no BEGIN/ROLLBACK is needed
-- because nothing here writes. Run against the live production database after the
-- deploy of merge 5d0d3c19a applied 146 for the second time.
\echo '-- 1. ledger: both applications of 146, hand then runner'
SELECT filename, applied_at FROM public.schema_migrations
 WHERE filename LIKE '%146_drop_legacy_per_hcp_count_columns.sql' ORDER BY applied_at;
\echo '-- 2. the legacy three are gone from EVERY table/view in every schema'
SELECT count(*) AS legacy_cols_anywhere FROM information_schema.columns
 WHERE column_name IN ('trx_count','nrx_count','total_rx_count');
\echo '-- 3. canonical columns, row count and the three sums'
SELECT count(*) AS rows,
       sum(triggers_delivered_count) AS delivered,
       sum(triggers_accepted_count)  AS accepted,
       sum(triggers_total_count)     AS total
  FROM public.business_metrics;
\echo '-- 4. the sync machinery 146 removed is still absent'
SELECT (SELECT count(*) FROM pg_trigger WHERE tgname='business_metrics_sync_legacy_trigger_counts_trg') AS sync_trigger,
       (SELECT count(*) FROM pg_proc    WHERE proname='business_metrics_sync_legacy_trigger_counts')    AS sync_function;
\echo '-- 5. the four split views, at their contracted width'
SELECT table_name, count(*) AS cols FROM information_schema.columns
 WHERE table_schema='public' AND table_name IN
   ('v_train_business_metrics','v_test_business_metrics','v_validation_business_metrics','v_holdout_business_metrics')
 GROUP BY 1 ORDER BY 1;
SELECT count(*) AS v_train_business_metrics_rows FROM public.v_train_business_metrics;
