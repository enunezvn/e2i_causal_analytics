-- ROLLBACK for migration 161 (#2310). NOT a forward migration: scripts/run_migrations.sh skips
-- rollback_*.sql.
--
-- Restores get_latest_model to its database/ml/mlops_tables.sql body (newest row of the
-- experiment, any stage) and removes 161's ledger row so a later deploy re-applies it. A second
-- run changes nothing.

CREATE OR REPLACE FUNCTION get_latest_model(p_experiment_name VARCHAR)
RETURNS TABLE (
    model_id UUID,
    model_name VARCHAR,
    model_version VARCHAR,
    auc DECIMAL,
    stage model_stage_enum
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.model_name,
        m.model_version,
        m.auc,
        m.stage
    FROM ml_model_registry m
    JOIN ml_experiments e ON m.experiment_id = e.id
    WHERE e.experiment_name = p_experiment_name
    ORDER BY m.registered_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

DELETE FROM public.schema_migrations
 WHERE filename = '161_get_latest_model_canonical.sql';
