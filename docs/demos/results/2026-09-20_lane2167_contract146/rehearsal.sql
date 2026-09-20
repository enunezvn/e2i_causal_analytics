-- Lane #2167 rehearsal wrapper, run ~2026-09-20T01:48Z against the live supabase-db.
--
-- The migration itself is NOT duplicated here -- it is
-- database/migrations/146_drop_legacy_per_hcp_count_columns.sql, and the sha256 of
-- the exact bytes rehearsed (its pre-move copy) is
-- cc9d952f783ccf041d18a9bdd02606f1c781a24c852fc912a8ef6b8ab9114fb4. To re-run:
-- concatenate <PRE>, that file, <POST>, and pipe the result into
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 --single-transaction
-- The trailing ROLLBACK is what makes this a rehearsal; verify afterwards that the
-- live legacy columns / trigger / view widths are UNCHANGED (the negative control in
-- README.md) before trusting the run.

-- <PRE> -----------------------------------------------------------------------
BEGIN;
CREATE TEMP TABLE _before AS
  SELECT count(*) AS n,
         sum(trx_count) AS s_trx, sum(nrx_count) AS s_nrx, sum(total_rx_count) AS s_total,
         sum(triggers_delivered_count) AS c_del, sum(triggers_accepted_count) AS c_acc,
         sum(triggers_total_count) AS c_tot
    FROM public.business_metrics;
SELECT 'BEFORE view cols' AS probe, table_name, count(*)
  FROM information_schema.columns
 WHERE table_schema='public' AND table_name LIKE 'v_%_business_metrics'
 GROUP BY table_name ORDER BY table_name;

-- <the migration> --------------------------------------------------------------
\i database/migrations/146_drop_legacy_per_hcp_count_columns.sql

-- <POST> ----------------------------------------------------------------------
SELECT 'AFTER legacy cols remaining' AS probe, count(*) AS n
  FROM information_schema.columns
 WHERE table_schema='public' AND table_name='business_metrics'
   AND column_name IN ('trx_count','nrx_count','total_rx_count');
SELECT 'AFTER trigger present' AS probe, count(*) AS n
  FROM pg_trigger WHERE tgname='business_metrics_sync_legacy_trigger_counts_trg' AND NOT tgisinternal;
SELECT 'AFTER function present' AS probe, count(*) AS n
  FROM pg_proc p JOIN pg_namespace ns ON ns.oid=p.pronamespace
 WHERE ns.nspname='public' AND p.proname='business_metrics_sync_legacy_trigger_counts';
SELECT 'AFTER view cols' AS probe, table_name, count(*)
  FROM information_schema.columns
 WHERE table_schema='public' AND table_name LIKE 'v_%_business_metrics'
 GROUP BY table_name ORDER BY table_name;
SELECT 'VALUES preserved' AS probe,
       b.n, b.s_trx, a_del, b.s_nrx, a_acc, b.s_total, a_tot,
       (b.s_trx IS NOT DISTINCT FROM a_del AND b.s_nrx IS NOT DISTINCT FROM a_acc
        AND b.s_total IS NOT DISTINCT FROM a_tot) AS all_match
  FROM _before b,
       LATERAL (SELECT sum(triggers_delivered_count) a_del, sum(triggers_accepted_count) a_acc,
                       sum(triggers_total_count) a_tot FROM public.business_metrics) x;
SELECT 'LEDGER row' AS probe, filename FROM public.schema_migrations WHERE filename LIKE '%146%';
SELECT 'train view rows' AS probe, count(*) FROM public.v_train_business_metrics;
ROLLBACK;
