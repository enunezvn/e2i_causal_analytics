-- ============================================================================
-- E2I Causal Analytics - ROLLBACK for ml/041_composer_learning_loop_recording.sql
-- NOT a forward migration: scripts/run_migrations.sh skips rollback_*.sql. Apply by hand,
-- AFTER the code revert, and BEFORE rollback_040.sql:
--
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--       --single-transaction < database/ml/rollback_041.sql
--
-- Returns composer_episodes / composition_steps / tool_performance and the three views to
-- their ml/013 definitions (the function, trigger, index and view bodies below are copied
-- verbatim from ml/013, so the restored objects are identical to the pre-041 ones), and drops
-- every object ml/041 created. Data recorded in the dropped columns is lost.
--
-- Refuses (and, in one transaction, changes nothing) while any episode lacks
-- total_latency_ms: ml/013 declares it NOT NULL, and unfinished or abandoned compositions
-- recorded by the learning loop have none. Delete or complete those rows, then re-run.
--
-- Idempotent: every drop is IF EXISTS and every recreation replaces or checks first, so a
-- second run on an already rolled-back database changes nothing.
-- ============================================================================

DO $guard$
DECLARE
    v_open bigint;
BEGIN
    IF EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid = 'composer_episodes'::regclass
                   AND attname = 'total_latency_ms' AND NOT attnotnull) THEN
        SELECT count(*) INTO v_open FROM composer_episodes WHERE total_latency_ms IS NULL;
        IF v_open > 0 THEN
            RAISE EXCEPTION 'rollback_041: % episode(s) have no total_latency_ms (unfinished or abandoned compositions); ml/013 requires it. Delete or complete them, then re-run.', v_open;
        END IF;
    END IF;
END
$guard$;

-- 1. Views and functions created by ml/041 (views first: they read the functions and columns).
DROP VIEW IF EXISTS v_tool_reliability;
DROP VIEW IF EXISTS v_composition_success_rate;
DROP VIEW IF EXISTS v_active_compositions;
DROP FUNCTION IF EXISTS composer_record_start(jsonb);
DROP FUNCTION IF EXISTS composer_record_phase(jsonb, text, jsonb);
DROP FUNCTION IF EXISTS composer_record_steps(jsonb, jsonb);
DROP FUNCTION IF EXISTS composer_record_heartbeat(jsonb);
DROP FUNCTION IF EXISTS composer_record_finish(jsonb, jsonb);
DROP FUNCTION IF EXISTS composer_steps_for(text[]);
DROP FUNCTION IF EXISTS composer_seed_episode(jsonb);
DROP FUNCTION IF EXISTS composer_episode_patch(jsonb, text[]);
DROP FUNCTION IF EXISTS composer_structure_plan(jsonb, text[]);
DROP FUNCTION IF EXISTS composer_structure_output(jsonb, text[]);
DROP FUNCTION IF EXISTS composer_structure_sub_questions(jsonb);
DROP FUNCTION IF EXISTS composer_structure_groups(jsonb);
DROP FUNCTION IF EXISTS composer_structure_numbers(jsonb);
DROP FUNCTION IF EXISTS composer_structure_params(jsonb, text[], text[], text[]);
DROP FUNCTION IF EXISTS composer_structure_value(jsonb, text[], text[]);
DROP FUNCTION IF EXISTS composer_registered_output_fields();
DROP FUNCTION IF EXISTS composer_schema_names(jsonb);
DROP FUNCTION IF EXISTS composer_public_column_names();
DROP FUNCTION IF EXISTS get_tool_reliability(integer, boolean);

-- 2. Columns, constraints and the index ml/041 added.
DROP INDEX IF EXISTS uq_tool_performance_step;
ALTER TABLE tool_performance DROP CONSTRAINT IF EXISTS tool_performance_outcome_class_check;
ALTER TABLE tool_performance
    DROP COLUMN IF EXISTS outcome_class,
    DROP COLUMN IF EXISTS attempts,
    DROP COLUMN IF EXISTS is_synthetic,
    DROP COLUMN IF EXISTS tool_version;

ALTER TABLE composition_steps DROP CONSTRAINT IF EXISTS composition_steps_outcome_class_check;
ALTER TABLE composition_steps
    DROP COLUMN IF EXISTS outcome_class,
    DROP COLUMN IF EXISTS attempts,
    DROP COLUMN IF EXISTS cache_hit,
    DROP COLUMN IF EXISTS error_type;

ALTER TABLE composer_episodes DROP CONSTRAINT IF EXISTS composer_episodes_outcome_check;
ALTER TABLE composer_episodes DROP CONSTRAINT IF EXISTS composer_episodes_plan_source_check;
ALTER TABLE composer_episodes DROP CONSTRAINT IF EXISTS composer_episodes_failed_phase_check;
ALTER TABLE composer_episodes
    DROP COLUMN IF EXISTS audit_workflow_id,
    DROP COLUMN IF EXISTS outcome,
    DROP COLUMN IF EXISTS failed_phase,
    DROP COLUMN IF EXISTS error_type,
    DROP COLUMN IF EXISTS plan_source,
    DROP COLUMN IF EXISTS entry_point,
    DROP COLUMN IF EXISTS brand,
    DROP COLUMN IF EXISTS region,
    DROP COLUMN IF EXISTS is_synthetic,
    DROP COLUMN IF EXISTS tools_executed,
    DROP COLUMN IF EXISTS tools_succeeded,
    DROP COLUMN IF EXISTS last_activity_at;
ALTER TABLE composer_episodes ALTER COLUMN total_latency_ms SET NOT NULL;

-- 3. The ml/013 objects ml/041 dropped (owner decision O3), verbatim from ml/013.
ALTER TABLE composer_episodes ADD COLUMN IF NOT EXISTS query_embedding vector(1536);
COMMENT ON COLUMN composer_episodes.query_embedding IS 'Vector embedding for similarity search (1536-dim)';

CREATE INDEX IF NOT EXISTS idx_composer_episodes_embedding ON composer_episodes 
    USING ivfflat (query_embedding vector_cosine_ops) 
    WITH (lists = 100);

CREATE OR REPLACE FUNCTION find_similar_compositions(
    p_query_embedding vector(1536),
    p_limit INTEGER DEFAULT 5,
    p_min_similarity FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    episode_id UUID,
    composition_id VARCHAR(100),
    query_text TEXT,
    tool_plan JSONB,
    success BOOLEAN,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ce.episode_id,
        ce.composition_id,
        ce.query_text,
        ce.tool_plan,
        ce.success,
        1 - (ce.query_embedding <=> p_query_embedding) AS similarity
    FROM composer_episodes ce
    WHERE ce.query_embedding IS NOT NULL
      AND ce.status = 'COMPLETED'
      AND ce.success = true
      AND 1 - (ce.query_embedding <=> p_query_embedding) >= p_min_similarity
    ORDER BY ce.query_embedding <=> p_query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION find_similar_compositions IS 'Finds similar successful compositions for plan optimization using vector similarity';

CREATE OR REPLACE FUNCTION trigger_log_step_performance()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'COMPLETED' AND OLD.status != 'COMPLETED' THEN
        INSERT INTO tool_performance (
            tool_id,
            tool_name,
            latency_ms,
            success,
            composition_id,
            step_id,
            called_by
        )
        SELECT 
            NEW.tool_id,
            NEW.tool_name,
            NEW.latency_ms,
            true,
            ce.composition_id,
            NEW.step_id,
            'composer'
        FROM composer_episodes ce
        WHERE ce.episode_id = NEW.episode_id;
    ELSIF NEW.status = 'FAILED' AND OLD.status != 'FAILED' THEN
        INSERT INTO tool_performance (
            tool_id,
            tool_name,
            latency_ms,
            success,
            error_type,
            composition_id,
            step_id,
            called_by
        )
        SELECT 
            NEW.tool_id,
            NEW.tool_name,
            COALESCE(NEW.latency_ms, 0),
            false,
            SUBSTRING(NEW.error_message FROM 1 FOR 100),
            ce.composition_id,
            NEW.step_id,
            'composer'
        FROM composer_episodes ce
        WHERE ce.episode_id = NEW.episode_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_log_step_performance ON composition_steps;
CREATE TRIGGER trg_log_step_performance
    AFTER UPDATE ON composition_steps
    FOR EACH ROW
    EXECUTE FUNCTION trigger_log_step_performance();

-- 4. The ml/013 views, verbatim.
CREATE OR REPLACE VIEW v_composition_success_rate AS
WITH tools_extracted AS (
    SELECT
        episode_id,
        created_at,
        status,
        success,
        total_latency_ms,
        tool_plan,
        jsonb_array_elements_text(
            CASE WHEN tool_plan ? 'steps'
                 THEN (tool_plan->'steps')
                 ELSE '[]'::jsonb
            END
        ) AS tool_name
    FROM composer_episodes
    WHERE created_at > NOW() - INTERVAL '30 days'
      AND tool_plan IS NOT NULL
)
SELECT
    DATE_TRUNC('day', ce.created_at) AS day,
    ce.status,
    COUNT(*) AS total_compositions,
    COUNT(*) FILTER (WHERE ce.success = true) AS successful,
    COUNT(*) FILTER (WHERE ce.success = false) AS failed,
    COUNT(*) FILTER (WHERE ce.success IS NULL) AS pending_feedback,
    ROUND(
        COUNT(*) FILTER (WHERE ce.success = true)::NUMERIC /
        NULLIF(COUNT(*) FILTER (WHERE ce.success IS NOT NULL), 0) * 100,
        2
    ) AS success_rate_pct,
    ROUND(AVG(ce.total_latency_ms)::numeric, 2) AS avg_latency_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ce.total_latency_ms)::numeric, 2) AS p95_latency_ms,
    jsonb_agg(DISTINCT te.tool_name) FILTER (WHERE te.tool_name IS NOT NULL) AS tools_used
FROM composer_episodes ce
LEFT JOIN tools_extracted te ON ce.episode_id = te.episode_id
WHERE ce.created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', ce.created_at), ce.status
ORDER BY day DESC, ce.status;

COMMENT ON VIEW v_composition_success_rate IS 'Daily composition success metrics for monitoring';

CREATE OR REPLACE VIEW v_tool_reliability AS
SELECT 
    tr.tool_id,
    tr.name AS tool_name,
    tr.category,
    tr.source_agent,
    COUNT(tp.performance_id) AS total_executions,
    COUNT(*) FILTER (WHERE tp.success = true) AS successful_executions,
    ROUND(
        COUNT(*) FILTER (WHERE tp.success = true)::NUMERIC / 
        NULLIF(COUNT(*), 0) * 100, 
        2
    ) AS success_rate_pct,
    ROUND(AVG(tp.latency_ms)::numeric, 2) AS avg_latency_ms,
    ROUND(STDDEV(tp.latency_ms)::numeric, 2) AS stddev_latency_ms,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY tp.latency_ms)::numeric, 2) AS p50_latency_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY tp.latency_ms)::numeric, 2) AS p95_latency_ms,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY tp.latency_ms)::numeric, 2) AS p99_latency_ms,
    MAX(tp.executed_at) AS last_executed_at,
    MODE() WITHIN GROUP (ORDER BY tp.error_type) FILTER (WHERE tp.error_type IS NOT NULL) AS most_common_error
FROM tool_registry tr
LEFT JOIN tool_performance tp ON tr.tool_id = tp.tool_id
    AND tp.executed_at > NOW() - INTERVAL '7 days'
WHERE tr.composable = true
GROUP BY tr.tool_id, tr.name, tr.category, tr.source_agent
ORDER BY total_executions DESC;

COMMENT ON VIEW v_tool_reliability IS 'Tool reliability metrics from last 7 days for planner decisions';

CREATE OR REPLACE VIEW v_active_compositions AS
SELECT 
    ce.composition_id,
    ce.status,
    ce.query_text,
    ce.created_at,
    EXTRACT(EPOCH FROM (NOW() - ce.created_at)) * 1000 AS elapsed_ms,
    ce.session_id,
    jsonb_array_length(ce.sub_questions) AS sub_question_count,
    COUNT(cs.step_id) AS total_steps,
    COUNT(cs.step_id) FILTER (WHERE cs.status = 'COMPLETED') AS completed_steps,
    COUNT(cs.step_id) FILTER (WHERE cs.status = 'EXECUTING') AS running_steps,
    COUNT(cs.step_id) FILTER (WHERE cs.status = 'FAILED') AS failed_steps
FROM composer_episodes ce
LEFT JOIN composition_steps cs ON ce.episode_id = cs.episode_id
WHERE ce.status IN ('PENDING', 'DECOMPOSING', 'PLANNING', 'EXECUTING', 'SYNTHESIZING')
GROUP BY ce.episode_id, ce.composition_id, ce.status, ce.query_text, 
         ce.created_at, ce.session_id, ce.sub_questions
ORDER BY ce.created_at DESC;

COMMENT ON VIEW v_active_compositions IS 'Currently active compositions with progress metrics';
