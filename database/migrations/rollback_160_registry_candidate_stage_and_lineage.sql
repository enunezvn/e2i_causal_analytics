-- ROLLBACK for migration 160 (#2310, #2311, #2308). NOT a forward migration:
-- scripts/run_migrations.sh skips rollback_*.sql.
--
-- Restores the pre-160 shape. Rows are returned to what the pre-160 writer produced, which is
-- also exactly what 160 changed them from on prod (measured 2026-09-28):
--   * a retrain row (retrain_of_id set) at 'candidate' whose job completed -> 'staging';
--     at 'archived' whose job failed, or at 'candidate' whose job did not complete ->
--     'development' (only rows a retraining-history row links, so an operator's own
--     archive of an unlinked row is untouched);
--   * a 'registered' deployment of such a row -> 'active' (the pre-#2308 status);
--   * ml_retraining_history.deployment_id pointing at such a deployment -> NULL (never
--     written before 160).
-- Retrain rows written by the post-160 code are returned the same way, which is what the
-- pre-160 code would have written for them. Then the trigger, constraints, index and columns
-- are dropped (the lineage and the exact MLflow versions are lost; re-applying 160 re-derives
-- the backfilled ones, not those of later retrains), and 160's ledger row is removed so a
-- later deploy re-applies it. A second run changes nothing.
--
-- Apply by hand (roll the code back first, or the new writer fails on the missing columns):
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--     --single-transaction < database/migrations/rollback_160_registry_candidate_stage_and_lineage.sql

DO $$
DECLARE
    v_n INTEGER;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'ml_model_registry'
                      AND column_name = 'retrain_of_id') THEN
        RAISE NOTICE 'rollback 160: retrain_of_id absent, nothing to restore';
        RETURN;
    END IF;

    CREATE TEMP TABLE _r160_linked ON COMMIT DROP AS
    SELECT r.id AS row_id, h.id AS history_id, h.status AS job_status
      FROM ml_model_registry r
      JOIN ml_model_registry p ON p.id = r.retrain_of_id
      JOIN ml_retraining_history h
        ON h.model_id = p.id AND h.new_model_version = r.model_version;

    UPDATE ml_retraining_history h
       SET deployment_id = NULL
      FROM _r160_linked l
      JOIN ml_deployments d ON d.model_registry_id = l.row_id
     WHERE h.id = l.history_id
       AND h.deployment_id = d.id;
    GET DIAGNOSTICS v_n = ROW_COUNT;
    RAISE NOTICE 'rollback 160: history.deployment_id cleared on % rows', v_n;

    UPDATE ml_deployments d
       SET status = 'active'
      FROM _r160_linked l
     WHERE d.model_registry_id = l.row_id
       AND d.status::text = 'registered';
    GET DIAGNOSTICS v_n = ROW_COUNT;
    RAISE NOTICE 'rollback 160: deployments registered -> active on % rows', v_n;

    UPDATE ml_model_registry r
       SET stage = 'staging'
      FROM _r160_linked l
     WHERE r.id = l.row_id AND l.job_status = 'completed' AND r.stage::text = 'candidate';
    GET DIAGNOSTICS v_n = ROW_COUNT;
    RAISE NOTICE 'rollback 160: candidate -> staging on % rows', v_n;

    UPDATE ml_model_registry r
       SET stage = 'development'
      FROM _r160_linked l
     WHERE r.id = l.row_id
       AND l.job_status IS DISTINCT FROM 'completed'
       AND r.stage::text IN ('candidate', 'archived')
       AND (r.stage::text = 'candidate' OR l.job_status = 'failed');
    GET DIAGNOSTICS v_n = ROW_COUNT;
    RAISE NOTICE 'rollback 160: failed/unfinished-job rows -> development on % rows', v_n;
END
$$;

DROP TRIGGER IF EXISTS tr_ml_model_registry_retrain_of_immutable ON ml_model_registry;
DROP FUNCTION IF EXISTS ml_model_registry_retrain_of_immutable();
ALTER TABLE ml_model_registry DROP CONSTRAINT IF EXISTS ml_model_registry_retrain_of_not_self;
ALTER TABLE ml_model_registry DROP CONSTRAINT IF EXISTS ml_model_registry_mlflow_model_version_positive;
DROP INDEX IF EXISTS idx_ml_model_registry_retrain_of;
ALTER TABLE ml_model_registry DROP COLUMN IF EXISTS retrain_of_id;
ALTER TABLE ml_model_registry DROP COLUMN IF EXISTS mlflow_model_version;

DELETE FROM public.schema_migrations
 WHERE filename = '160_registry_candidate_stage_and_lineage.sql';
