-- ROLLBACK for migration 160 (#2310, #2311, #2308). NOT a forward migration:
-- scripts/run_migrations.sh skips rollback_*.sql.
--
-- Restores the pre-160 shape for EVERY row that carries lineage (codex r1: not only rows a
-- retraining-history row still links). Rows are returned to what the pre-160 writer
-- produced, which is also exactly what 160 changed them from on prod (measured 2026-09-28):
--   * a lineage row at 'candidate' -> 'staging' when its job completed or no job row links
--     it (a standalone deploy: the old writer promoted it to staging), else 'development'
--     (a failed / unfinished job: the old writer never promoted it);
--   * a lineage row at 'archived' whose linked job FAILED -> 'development' (160 archived it);
--     any other archived row is an operator's decision and is left alone;
--   * every 'registered' deployment -> 'active' (the pre-#2308 status the old code wrote);
--   * ml_retraining_history.deployment_id pointing at a lineage row's deployment -> NULL
--     (never written before 160).
-- It then REFUSES (raises, nothing committed) if any row is still at 'candidate' or any
-- deployment still 'registered', so no row is left in a state the pre-160 code cannot read.
-- Then the trigger, constraints, index and columns are dropped (the lineage and the exact
-- MLflow versions are lost; re-applying 160 re-derives the three backfilled rows, not later
-- retrains), and 160's ledger row is removed so a later deploy re-applies it. A second run
-- changes nothing.
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
    SELECT r.id AS row_id,
           bool_or(h.status = 'completed') AS completed,
           bool_or(h.status = 'failed') AS failed,
           count(h.id) AS jobs
      FROM ml_model_registry r
      LEFT JOIN ml_model_registry p ON p.id = r.retrain_of_id
      LEFT JOIN ml_retraining_history h
        ON h.model_id = p.id AND h.new_model_version = r.model_version
     WHERE r.retrain_of_id IS NOT NULL
     GROUP BY r.id;

    UPDATE ml_retraining_history h
       SET deployment_id = NULL
      FROM ml_deployments d
      JOIN _r160_linked l ON l.row_id = d.model_registry_id
     WHERE h.deployment_id = d.id;
    GET DIAGNOSTICS v_n = ROW_COUNT;
    RAISE NOTICE 'rollback 160: history.deployment_id cleared on % rows', v_n;

    UPDATE ml_deployments d
       SET status = 'active'
     WHERE d.status::text = 'registered';
    GET DIAGNOSTICS v_n = ROW_COUNT;
    RAISE NOTICE 'rollback 160: deployments registered -> active on % rows', v_n;

    UPDATE ml_model_registry r
       SET stage = 'staging'
      FROM _r160_linked l
     WHERE r.id = l.row_id AND r.stage::text = 'candidate'
       AND (l.completed OR l.jobs = 0);
    GET DIAGNOSTICS v_n = ROW_COUNT;
    RAISE NOTICE 'rollback 160: candidate -> staging on % rows', v_n;

    UPDATE ml_model_registry r
       SET stage = 'development'
      FROM _r160_linked l
     WHERE r.id = l.row_id
       AND ((r.stage::text = 'candidate' AND NOT l.completed AND l.jobs > 0)
            OR (r.stage::text = 'archived' AND l.failed AND NOT l.completed));
    GET DIAGNOSTICS v_n = ROW_COUNT;
    RAISE NOTICE 'rollback 160: failed/unfinished-job rows -> development on % rows', v_n;

    IF EXISTS (SELECT 1 FROM ml_model_registry WHERE stage::text = 'candidate')
       OR EXISTS (SELECT 1 FROM ml_deployments WHERE status::text = 'registered') THEN
        RAISE EXCEPTION 'rollback 160 refused: rows remain at candidate/registered';
    END IF;
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
