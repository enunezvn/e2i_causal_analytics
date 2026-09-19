-- ============================================================================
-- Migration 146 (CONTRACT): retire the legacy per_hcp_rollup count columns
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
-- MIGRATION_DIRS in a single pass. A 146_*.sql committed beside 144 would run
-- seconds after it, the legacy columns would be gone before one container had
-- been replaced, and the deploy would be exactly as unsafe as the in-place name
-- swap that HIGH-1 rejected. Expand and contract are only expand/contract if they
-- land in two different deploys.
--
-- database/deferred/ appears in no MIGRATION_DIRS entry, so the runner cannot see
-- this file at all -- a structural separation, not a naming convention that a
-- typo or a new skip-pattern could arm. It is pinned by
-- tests/unit/test_database/test_mig146_contract_legacy_per_hcp_columns.py, which
-- parses MIGRATION_DIRS out of the runner rather than restating it.
--
-- ---------------------------------------------------------------------------
-- WHO OWNS APPLYING THIS: ISSUE #2167
-- ---------------------------------------------------------------------------
-- Nothing in the repository causes this file to run, which is the point and also
-- the risk: without a tracked owner the lane PR merges, #2114 closes, and the
-- legacy columns live forever (codex iter2/iter3 MED-1). Issue #2167 is that
-- owner. It carries the preconditions below, the apply command, and the later PR
-- that moves this file into database/migrations/. If you apply this file, close
-- #2167; if you decide NOT to, say so there rather than letting it lapse.
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
--     --single-transaction < database/deferred/146_drop_legacy_per_hcp_count_columns.sql
--
-- Verify first, in the same shape, with a trailing ROLLBACK instead of a commit.
--
-- The ledger row is written AFTER every schema statement in this file -- only the
-- trailing NOTIFY follows it -- inside the same --single-transaction as the schema
-- change (codex iter2 MED-1; the "last statement" wording was corrected in iter4,
-- because NOTIFY does follow it and a claim in a permanent artifact has to be true
-- as written). It used to be a
-- second, separate command in this header, which could be forgotten or could fail
-- on its own, leaving the ledger disagreeing with the schema. Nothing else writes
-- it: this file is outside every MIGRATION_DIRS entry, so the runner never sees it.
--
-- IF THIS FILE IS EVER MOVED INTO database/migrations/ (the right end state once
-- the lane's SHA, not origin/main, is the automated rollback target — codex iter2
-- MED-1), the runner will key it as '146_drop_legacy_per_hcp_count_columns.sql'
-- with no 'deferred/' prefix, will not match the row written below, and will apply
-- it once more. What that second application DOES, stated exactly (codex iter10):
-- the precondition finds the legacy three already gone and lets it through; the
-- column/trigger/function drops are IF EXISTS no-ops; the four split views are
-- dropped and recreated with the same definitions (new OIDs; nothing else depends
-- on them -- 0 dependents, and a recreated view gets the same privileges as the
-- live ones, postgres + service_role, both measured 2026-09-19); NOTIFY reloads
-- the PostgREST schema cache; and the runner records the new ledger key. No
-- column, row or privilege changes. Delete the stale 'deferred/...' ledger row
-- afterwards if you want the ledger tidy.
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
-- The widening is safe, and the reason is a measurement rather than an argument
-- about the base schema: these views have ZERO consumers outside database/
-- (measured 2026-09-18 across src/, tests/, scripts/, feature_repo/ and
-- frontend/src; re-checked 2026-09-19 across src/, scripts/, feature_repo/ and
-- docker/), so no RUNTIME query selects a column from them. The only queries that
-- do are the lane plan's own rehearsal probes, inside BEGIN ... ROLLBACK, and they
-- select the canonical columns the rebuild adds (codex iter13 LOW). The "only other
-- mention anywhere is docs/data/02-CORE-DATA-DICTIONARY.md" this header used to add
-- was false, and falsified by this lane's own files (ultracode iter7 LOW). The
-- complete list of files naming these views is: database/core/
-- e2i_ml_complete_v3_schema.sql (which creates them), this file (which rebuilds
-- them), docs/data/02-CORE-DATA-DICTIONARY.md (a name list), the lane plan, and
-- tests/unit/test_database/test_mig146_contract_legacy_per_hcp_columns.py (the guard
-- over this file). None of them READS a column from a view, which is the claim that
-- matters -- and stating it as a file count invited exactly this kind of drift. It also restores the AUTHORED intent: e2i_ml_complete_v3_schema.sql writes
-- these views as `SELECT * FROM business_metrics WHERE data_split = ...`, and
-- PostgreSQL froze that star into an explicit list at creation time.
--
-- CORRECTION 2026-09-18 (codex iter2 HIGH-1). An earlier draft of this header
-- justified the widening by claiming a box built from
-- database/core/e2i_ml_complete_v3_schema.sql "already gets all 36" columns. That
-- was asserted, not measured, and it is FALSE: that file's CREATE TABLE
-- business_metrics declares 19 columns, and the other 17 arrive from later
-- migrations (033 and after). Nothing depended on the claim -- but it was wrong,
-- so it is corrected here rather than quietly removed.
--
-- Re-runnable. Each of the 16 statements, by how it survives a second run: the
-- precondition DO block lets an already-contracted schema through; the 4 view,
-- 1 trigger, 1 function and 3 column DROPs carry IF EXISTS; the 4 view creates
-- are CREATE OR REPLACE; the ledger INSERT is ON CONFLICT DO NOTHING; NOTIFY is
-- a cache reload. So a second application raises nothing and changes no column,
-- row or privilege; it does recreate the four views -- see "IF THIS FILE IS EVER
-- MOVED" above for exactly what that does.
-- ----------------------------------------------------------------------------

-- ----------------------------------------------------------------------------
-- PRECONDITION: refuse unless the expand really ran and its values really landed.
-- ----------------------------------------------------------------------------
-- ultracode iter8 HIGH. The DROPs and view creates below carry IF EXISTS / OR REPLACE so
-- that a second application raises nothing -- and the consequence nobody had drawn is that the
-- same property makes this file run happily to completion on a schema where 144 was
-- NEVER applied, or was rolled back. It drops trx_count / nrx_count / total_rx_count
-- with no canonical column holding the values, and reports exit 0.
--
-- Measured 2026-09-18 inside BEGIN/ROLLBACK against production, which is a FAITHFUL
-- environment for it because production genuinely has not had 144 applied: 3 legacy
-- columns and 12,143 rows carrying counts before; 0 legacy columns, 0 canonical
-- columns, 33 columns and psql exit 0 after. Nothing raised. Only two NOTICEs, about
-- skipping a trigger and a function that were never there.
--
-- This file is the destructive half, applied BY HAND, possibly weeks after 144 and
-- possibly by someone who was not here for it. Until now the only thing standing
-- between that operator and the loss above was the prose in this header -- and prose
-- is not a precondition, it is a hope about who is reading.
--
-- The two conditions are asked in this ORDER for a reason. The catalog check must
-- raise BEFORE the row-level query is reached, because that query names the canonical
-- columns and would otherwise fail with a bare `column "triggers_delivered_count" does
-- not exist` -- a refusal, technically, but one an operator cannot act on, which this
-- lane has already classified as a defect in its own right (the prover's `probe()`
-- had exactly this shape). plpgsql plans a statement on first execution rather than at
-- block entry, so an early RAISE means the second query is never planned. That is load
-- bearing, and it is verified live rather than assumed.
--
-- ON_ERROR_STOP=1 in the documented apply command is part of this guard, and the
-- difference is measured, not argued: the same refusal exits **3** with the flag and
-- **0** without it (2026-09-18). Without it psql prints the exception and carries on
-- into the first DROP, and an operator's `&&` chain sees success.
-- tests/unit/test_database/test_mig146_contract_legacy_per_hcp_columns.py pins the
-- flag in the COMMAND, not in this prose, because prose is what failed here already.
--
-- Rehearsed live 2026-09-18, three ways, each inside BEGIN/ROLLBACK:
--   144 never applied            -> REFUSED, 3 of 3 canonical columns absent, 0 DROPs ran
--   144 applied first            -> PROCEEDS, legacy 3 -> 0, canonical 3, 12,143 rows kept
--   144 applied, 1 row disagrees -> REFUSED, names the count (plant verified first)
-- and again 2026-09-19 after the legacy-presence gate (codex iter9 MED), four ways:
--   144, 146, 146 again          -> both PROCEED; legacy 0, canonical 3, 4 views
--   144 never applied            -> REFUSED, 3 of 3 canonical columns absent
--   144 applied, 1 row disagrees -> REFUSED, 1 row (plant verified landed first)
--   one legacy column missing    -> REFUSED, "only 2 of the 3 legacy columns remain"
-- The middle case is why the block asks the catalog before it asks the rows, and the
-- last is what proves the row query is reached and correct when the columns do exist.
DO $precondition$
DECLARE
    missing         integer;
    legacy_present  integer;
    disagreeing     bigint;
BEGIN
    SELECT count(*) INTO missing
      FROM (VALUES ('triggers_delivered_count'),
                   ('triggers_accepted_count'),
                   ('triggers_total_count')) AS want(column_name)
     WHERE NOT EXISTS (SELECT 1
                         FROM information_schema.columns c
                        WHERE c.table_schema = 'public'
                          AND c.table_name   = 'business_metrics'
                          AND c.column_name  = want.column_name);
    IF missing > 0 THEN
        RAISE EXCEPTION
            'contract 146 REFUSED: % of the 3 canonical columns are absent from '
            'public.business_metrics, so dropping the legacy three would destroy the '
            'per-HCP trigger counts outright. Apply '
            'database/migrations/144_per_hcp_trigger_count_columns.sql first, verify '
            'the deploy, then re-run this file.', missing;
    END IF;

    -- A SECOND application (the documented move into database/migrations/ re-keys
    -- this file and the runner applies it once more) finds the legacy three already
    -- gone. The row query below names them, so it may only run while they exist:
    -- none left means already contracted and nothing to lose; some but not all is a
    -- schema nobody designed, and refuses (codex iter9 MED, reproduced live).
    SELECT count(*) INTO legacy_present
      FROM (VALUES ('trx_count'),
                   ('nrx_count'),
                   ('total_rx_count')) AS legacy(column_name)
     WHERE EXISTS (SELECT 1
                     FROM information_schema.columns c
                    WHERE c.table_schema = 'public'
                      AND c.table_name   = 'business_metrics'
                      AND c.column_name  = legacy.column_name);
    IF legacy_present BETWEEN 1 AND 2 THEN
        RAISE EXCEPTION
            'contract 146 REFUSED: only % of the 3 legacy columns remain on '
            'public.business_metrics -- a partially contracted schema. Inspect it by '
            'hand before re-running this file.', legacy_present;
    END IF;

    IF legacy_present = 3 THEN
        SELECT count(*) INTO disagreeing
          FROM public.business_metrics
         WHERE (trx_count      IS NOT NULL AND triggers_delivered_count IS DISTINCT FROM trx_count)
            OR (nrx_count      IS NOT NULL AND triggers_accepted_count  IS DISTINCT FROM nrx_count)
            OR (total_rx_count IS NOT NULL AND triggers_total_count     IS DISTINCT FROM total_rx_count);
        IF disagreeing > 0 THEN
            RAISE EXCEPTION
                'contract 146 REFUSED: % rows hold a legacy count that its canonical column '
                'does not carry, so dropping the legacy three would lose those values. '
                'Re-run 144''s backfill (it is guarded by IS DISTINCT FROM, so it touches '
                'only the rows that disagree) and confirm this count reaches 0.', disagreeing;
        END IF;
    END IF;
END
$precondition$;

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

-- Record the application in the same transaction as the change it records.
INSERT INTO public.schema_migrations(filename)
VALUES ('deferred/146_drop_legacy_per_hcp_count_columns.sql')
ON CONFLICT DO NOTHING;

NOTIFY pgrst, 'reload schema';
