-- Migration 161: get_latest_model returns the canonical row, never a retrain candidate (#2310).
--
-- WHY: get_latest_model(p_experiment_name) (database/ml/mlops_tables.sql) returns the newest
-- ml_model_registry row of an experiment with no stage predicate. A retrain registers its row
-- in the experiment of the model it retrains at stage 'candidate' (migration 159/160), so the
-- function would now answer with an unreviewed retrain. Owner decision R3 (2026-09-28): the
-- canonical row is stage NOT IN ('candidate', 'archived', 'deprecated'); lineage
-- (retrain_of_id) is never used to exclude a row. No code in src/, scripts/ or frontend/ calls
-- the function (checked 2026-09-28), but it is a public function PostgREST exposes as an RPC,
-- so it is corrected rather than left to answer wrongly.
--
-- Also deterministic now: registered_at DESC NULLS LAST, then id DESC (it was registered_at DESC
-- alone, where a NULL timestamp sorts first and ties are unordered). Same signature and return
-- type, so CREATE OR REPLACE keeps its grants.
--
-- NEEDS 159: the 'candidate' label must exist in model_stage_enum before this body can be
-- planned (plpgsql resolves the literal at first execution; 159 is applied earlier either way).
--
-- NOTE: no transaction control here -- the migration runner wraps each file.

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
      AND (m.stage IS NULL OR m.stage NOT IN ('candidate', 'archived', 'deprecated'))
    ORDER BY m.registered_at DESC NULLS LAST, m.id DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;
