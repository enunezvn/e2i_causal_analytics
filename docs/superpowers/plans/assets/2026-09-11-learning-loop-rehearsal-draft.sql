-- REHEARSAL DRAFT (2026-09-11), NOT A MIGRATION. Rehearsed inside BEGIN … ROLLBACK on the droplet DB.
-- Superseded where the spec (docs/superpowers/specs/2026-09-11-tool-composer-learning-loop-design.md)
-- and plan Tasks 2–3 state deltas: split into ml/039 (enum) / ml/040 (sync) / ml/041 (recording);
-- pre-write sync guards + p_max_deprecations; seeded RPCs incl. composer_record_steps/heartbeat,
-- composer_public_column_names, composer_steps_for; last_activity_at; is_synthetic/tool_version on
-- tool_performance; outcome 'cancelled'; full (non-partial) unique index on tool_performance(step_id).

-- DRAFT (rehearsal only) ml/039_tool_composer_learning_loop.sql
-- 1. registry: category enum + source_agent CHECK + declared-vs-measured columns
ALTER TYPE tool_category ADD VALUE IF NOT EXISTS 'COHORT';

ALTER TABLE tool_registry DROP CONSTRAINT IF EXISTS valid_agent;
ALTER TABLE tool_registry ADD CONSTRAINT valid_agent
    CHECK (source_agent = ANY (enum_range(NULL::e2i_agent_name)::text[]));

ALTER TABLE tool_registry DROP COLUMN IF EXISTS success_rate;
COMMENT ON COLUMN tool_registry.avg_latency_ms IS
    'DECLARED latency baseline from the registered tool (@composable_tool avg_execution_ms), written by sync_tool_registry(). Measured latency lives in get_tool_reliability().';

-- 2. recording columns
ALTER TABLE composer_episodes
    ADD COLUMN IF NOT EXISTS outcome text,
    ADD COLUMN IF NOT EXISTS failed_phase text,
    ADD COLUMN IF NOT EXISTS entry_point text,
    ADD COLUMN IF NOT EXISTS brand text,
    ADD COLUMN IF NOT EXISTS region text,
    ADD COLUMN IF NOT EXISTS is_synthetic boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS tools_executed integer,
    ADD COLUMN IF NOT EXISTS tools_succeeded integer,
    ADD COLUMN IF NOT EXISTS last_phase_at timestamptz NOT NULL DEFAULT now();
ALTER TABLE composer_episodes DROP CONSTRAINT IF EXISTS composer_episodes_outcome_check;
ALTER TABLE composer_episodes ADD CONSTRAINT composer_episodes_outcome_check
    CHECK (outcome IS NULL OR outcome IN ('success','partial','failed','timeout'));
ALTER TABLE composer_episodes ALTER COLUMN total_latency_ms DROP NOT NULL;

DROP FUNCTION IF EXISTS find_similar_compositions(vector, integer, double precision);
DROP INDEX IF EXISTS idx_composer_episodes_embedding;
ALTER TABLE composer_episodes DROP COLUMN IF EXISTS query_embedding;

ALTER TABLE composition_steps
    ADD COLUMN IF NOT EXISTS outcome_class text,
    ADD COLUMN IF NOT EXISTS attempts integer,
    ADD COLUMN IF NOT EXISTS cache_hit boolean NOT NULL DEFAULT false;
ALTER TABLE composition_steps ALTER COLUMN serves_sub_question TYPE varchar(100);
ALTER TABLE composition_steps DROP CONSTRAINT IF EXISTS composition_steps_outcome_class_check;
ALTER TABLE composition_steps ADD CONSTRAINT composition_steps_outcome_class_check
    CHECK (outcome_class IS NULL OR outcome_class IN (
        'succeeded','cache_hit','refused','input_rejected','timeout','error',
        'plan_defect','dependency_unmet','circuit_open','not_registered'));

ALTER TABLE tool_performance ADD COLUMN IF NOT EXISTS outcome_class text;
ALTER TABLE tool_performance ADD COLUMN IF NOT EXISTS attempts integer;
ALTER TABLE tool_performance DROP CONSTRAINT IF EXISTS tool_performance_outcome_class_check;
ALTER TABLE tool_performance ADD CONSTRAINT tool_performance_outcome_class_check
    CHECK (outcome_class IS NULL OR outcome_class IN ('succeeded','refused','input_rejected','timeout','error'));
CREATE UNIQUE INDEX IF NOT EXISTS uq_tool_performance_step ON tool_performance(step_id) WHERE step_id IS NOT NULL;

-- 3. the design-era trigger (UPDATE-only, double-counts COMPLETED->FAILED) and the
--    unguarded registry metrics writer are replaced by explicit writes/reads below
DROP TRIGGER IF EXISTS trg_log_step_performance ON composition_steps;
DROP FUNCTION IF EXISTS trigger_log_step_performance();
DROP FUNCTION IF EXISTS update_tool_registry_metrics();

-- 4. sync RPC
CREATE OR REPLACE FUNCTION sync_tool_registry(p_tools jsonb, p_dependencies jsonb)
RETURNS jsonb LANGUAGE plpgsql SECURITY INVOKER SET search_path = public AS $fn$
DECLARE
    v_inserted int; v_updated int; v_deprecated int; v_dep_upserted int; v_dep_deleted int;
BEGIN
    IF p_tools IS NULL OR jsonb_typeof(p_tools) <> 'array' OR jsonb_array_length(p_tools) = 0 THEN
        RAISE EXCEPTION 'sync_tool_registry: empty tool payload refused';
    END IF;
    PERFORM pg_advisory_xact_lock(hashtext('sync_tool_registry'));

    WITH payload AS (
        SELECT * FROM jsonb_to_recordset(p_tools) AS t(
            name text, description text, category text, source_agent text,
            input_schema jsonb, output_schema jsonb, avg_latency_ms double precision, version text)
    ), up AS (
        INSERT INTO tool_registry AS tr (name, description, category, source_agent, input_schema,
                                         output_schema, composable, avg_latency_ms, version, deprecated_at)
        SELECT name, description, category::tool_category, source_agent, input_schema, output_schema,
               true, avg_latency_ms, version, NULL
        FROM payload
        ON CONFLICT (name) DO UPDATE SET
            description = EXCLUDED.description, category = EXCLUDED.category,
            source_agent = EXCLUDED.source_agent, input_schema = EXCLUDED.input_schema,
            output_schema = EXCLUDED.output_schema, composable = true,
            avg_latency_ms = EXCLUDED.avg_latency_ms, version = EXCLUDED.version, deprecated_at = NULL
        WHERE (tr.description, tr.category, tr.source_agent, tr.input_schema, tr.output_schema,
               tr.composable, tr.avg_latency_ms, tr.version, tr.deprecated_at)
          IS DISTINCT FROM
              (EXCLUDED.description, EXCLUDED.category, EXCLUDED.source_agent, EXCLUDED.input_schema,
               EXCLUDED.output_schema, true, EXCLUDED.avg_latency_ms, EXCLUDED.version, NULL::timestamptz)
        RETURNING (xmax = 0) AS inserted
    )
    SELECT count(*) FILTER (WHERE inserted), count(*) FILTER (WHERE NOT inserted)
      INTO v_inserted, v_updated FROM up;

    UPDATE tool_registry SET deprecated_at = now(), composable = false
     WHERE deprecated_at IS NULL
       AND name NOT IN (SELECT t->>'name' FROM jsonb_array_elements(p_tools) t);
    GET DIAGNOSTICS v_deprecated = ROW_COUNT;

    WITH m AS (
        SELECT c.tool_id AS consumer_tool_id, p.tool_id AS producer_tool_id, d.output_field, d.input_field
        FROM jsonb_to_recordset(COALESCE(p_dependencies, '[]'::jsonb))
             AS d(consumer text, producer text, output_field text, input_field text)
        JOIN tool_registry c ON c.name = d.consumer
        JOIN tool_registry p ON p.name = d.producer
    ), del AS (
        DELETE FROM tool_dependencies td
        WHERE NOT EXISTS (SELECT 1 FROM m WHERE m.consumer_tool_id = td.consumer_tool_id
                                           AND m.producer_tool_id = td.producer_tool_id)
        RETURNING 1
    ), ins AS (
        INSERT INTO tool_dependencies AS td (consumer_tool_id, producer_tool_id, output_field, input_field)
        SELECT consumer_tool_id, producer_tool_id, output_field, input_field FROM m
        ON CONFLICT (consumer_tool_id, producer_tool_id) DO UPDATE
            SET output_field = EXCLUDED.output_field, input_field = EXCLUDED.input_field
        WHERE (td.output_field, td.input_field) IS DISTINCT FROM (EXCLUDED.output_field, EXCLUDED.input_field)
        RETURNING 1
    )
    SELECT (SELECT count(*) FROM ins), (SELECT count(*) FROM del) INTO v_dep_upserted, v_dep_deleted;

    RETURN jsonb_build_object('inserted', v_inserted, 'updated', v_updated, 'deprecated', v_deprecated,
                              'dependencies_upserted', v_dep_upserted, 'dependencies_deleted', v_dep_deleted);
END $fn$;
REVOKE ALL ON FUNCTION sync_tool_registry(jsonb, jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION sync_tool_registry(jsonb, jsonb) TO service_role;

-- 5. reliability read (days-parameterised) + 30-day view wrapper
CREATE OR REPLACE FUNCTION get_tool_reliability(p_days integer DEFAULT 30)
RETURNS TABLE (tool_name varchar, category tool_category, source_agent varchar, declared_latency_ms double precision,
               n_invoked bigint, n_succeeded bigint, n_refused bigint, n_health_failures bigint,
               p50_latency_ms numeric, p95_latency_ms numeric, last_executed_at timestamptz,
               most_common_health_error varchar)
LANGUAGE sql STABLE SECURITY INVOKER SET search_path = public AS $fn$
    SELECT tr.name, tr.category, tr.source_agent, tr.avg_latency_ms,
           count(tp.performance_id),
           count(*) FILTER (WHERE tp.outcome_class = 'succeeded'),
           count(*) FILTER (WHERE tp.outcome_class IN ('refused','input_rejected')),
           count(*) FILTER (WHERE tp.outcome_class IN ('timeout','error')),
           round((percentile_cont(0.5) WITHIN GROUP (ORDER BY tp.latency_ms)
                  FILTER (WHERE tp.outcome_class = 'succeeded'))::numeric, 1),
           round((percentile_cont(0.95) WITHIN GROUP (ORDER BY tp.latency_ms)
                  FILTER (WHERE tp.outcome_class = 'succeeded'))::numeric, 1),
           max(tp.executed_at),
           mode() WITHIN GROUP (ORDER BY tp.error_type) FILTER (WHERE tp.outcome_class IN ('timeout','error'))
    FROM tool_registry tr
    LEFT JOIN tool_performance tp ON tp.tool_id = tr.tool_id
         AND tp.executed_at > now() - make_interval(days => p_days)
    WHERE tr.deprecated_at IS NULL
    GROUP BY tr.tool_id, tr.name, tr.category, tr.source_agent, tr.avg_latency_ms
    ORDER BY tr.name;
$fn$;
REVOKE ALL ON FUNCTION get_tool_reliability(integer) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION get_tool_reliability(integer) TO service_role;

DROP VIEW IF EXISTS v_tool_reliability;
CREATE VIEW v_tool_reliability AS SELECT * FROM get_tool_reliability(30);
REVOKE ALL ON v_tool_reliability FROM PUBLIC, anon, authenticated;
GRANT SELECT ON v_tool_reliability TO service_role;

-- 6. recording RPCs
CREATE OR REPLACE FUNCTION composer_record_start(p_episode jsonb)
RETURNS uuid LANGUAGE plpgsql SECURITY INVOKER SET search_path = public AS $fn$
DECLARE v_id uuid;
BEGIN
    INSERT INTO composer_episodes (composition_id, query_text, status, session_id, user_id, entry_point,
                                   brand, region, is_synthetic, last_phase_at)
    VALUES (p_episode->>'composition_id', p_episode->>'query_text', 'DECOMPOSING',
            p_episode->>'session_id', p_episode->>'user_id', p_episode->>'entry_point',
            p_episode->>'brand', p_episode->>'region', COALESCE((p_episode->>'is_synthetic')::boolean, false), now())
    ON CONFLICT (composition_id) DO NOTHING
    RETURNING episode_id INTO v_id;
    RETURN v_id;
END $fn$;

CREATE OR REPLACE FUNCTION composer_record_phase(p_composition_id text, p_status text, p_patch jsonb)
RETURNS boolean LANGUAGE plpgsql SECURITY INVOKER SET search_path = public AS $fn$
BEGIN
    UPDATE composer_episodes SET
        status = p_status::composition_status,
        decompose_latency_ms = COALESCE((p_patch->>'decompose_latency_ms')::float, decompose_latency_ms),
        plan_latency_ms = COALESCE((p_patch->>'plan_latency_ms')::float, plan_latency_ms),
        execute_latency_ms = COALESCE((p_patch->>'execute_latency_ms')::float, execute_latency_ms),
        sub_questions = COALESCE(p_patch->'sub_questions', sub_questions),
        tool_plan = COALESCE(p_patch->'tool_plan', tool_plan),
        parallelizable_groups = COALESCE(p_patch->'parallelizable_groups', parallelizable_groups),
        last_phase_at = now()
    WHERE composition_id = p_composition_id
      AND status IN ('PENDING','DECOMPOSING','PLANNING','EXECUTING','SYNTHESIZING');
    RETURN FOUND;
END $fn$;

CREATE OR REPLACE FUNCTION composer_record_finish(p_composition_id text, p_episode jsonb, p_steps jsonb)
RETURNS jsonb LANGUAGE plpgsql SECURITY INVOKER SET search_path = public AS $fn$
DECLARE v_episode uuid; v_steps int := 0; v_perf int := 0; v_unknown text[];
BEGIN
    UPDATE composer_episodes SET
        status = (p_episode->>'status')::composition_status,
        outcome = p_episode->>'outcome',
        failed_phase = p_episode->>'failed_phase',
        error_message = left(p_episode->>'error_message', 2000),
        total_latency_ms = (p_episode->>'total_latency_ms')::float,
        decompose_latency_ms = COALESCE((p_episode->>'decompose_latency_ms')::float, decompose_latency_ms),
        plan_latency_ms = COALESCE((p_episode->>'plan_latency_ms')::float, plan_latency_ms),
        execute_latency_ms = COALESCE((p_episode->>'execute_latency_ms')::float, execute_latency_ms),
        synthesize_latency_ms = (p_episode->>'synthesize_latency_ms')::float,
        tools_executed = (p_episode->>'tools_executed')::int,
        tools_succeeded = (p_episode->>'tools_succeeded')::int,
        completed_at = now(), last_phase_at = now()
    WHERE composition_id = p_composition_id
      AND status IN ('PENDING','DECOMPOSING','PLANNING','EXECUTING','SYNTHESIZING')
    RETURNING episode_id INTO v_episode;
    IF v_episode IS NULL THEN
        RETURN jsonb_build_object('recorded', false, 'reason', 'no open episode');
    END IF;

    SELECT array_agg(DISTINCT s->>'tool_name') INTO v_unknown
    FROM jsonb_array_elements(COALESCE(p_steps,'[]'::jsonb)) s
    WHERE NOT EXISTS (SELECT 1 FROM tool_registry tr WHERE tr.name = s->>'tool_name');

    WITH s AS (
        SELECT * FROM jsonb_to_recordset(COALESCE(p_steps,'[]'::jsonb)) AS x(
            step_number int, step_name text, tool_name text, input_params jsonb, output_keys jsonb,
            depends_on_steps int[], serves_sub_question text, started_at timestamptz, completed_at timestamptz,
            latency_ms float, status text, outcome_class text, error_message text, attempts int, cache_hit boolean)
    ), ins AS (
        INSERT INTO composition_steps (episode_id, step_number, step_name, tool_id, tool_name, input_params,
            output_result, depends_on_steps, serves_sub_question, started_at, completed_at, latency_ms, status,
            error_message, retry_count, outcome_class, attempts, cache_hit)
        SELECT v_episode, s.step_number, left(s.step_name,100), tr.tool_id, s.tool_name, COALESCE(s.input_params,'{}'),
               s.output_keys, COALESCE(s.depends_on_steps,'{}'), left(s.serves_sub_question,100), s.started_at,
               s.completed_at, s.latency_ms, s.status::composition_status, left(s.error_message,2000),
               GREATEST(COALESCE(s.attempts,1)-1,0), s.outcome_class, s.attempts, COALESCE(s.cache_hit,false)
        FROM s JOIN tool_registry tr ON tr.name = s.tool_name
        RETURNING step_id, tool_id, tool_name, latency_ms, outcome_class, attempts, error_message, completed_at
    ), perf AS (
        INSERT INTO tool_performance (tool_id, tool_name, latency_ms, success, error_type, composition_id,
                                      step_id, called_by, outcome_class, attempts, executed_at)
        SELECT tool_id, tool_name, COALESCE(latency_ms,0), outcome_class = 'succeeded',
               CASE WHEN outcome_class <> 'succeeded' THEN left(outcome_class || ': ' || COALESCE(error_message,''),100) END,
               p_composition_id, step_id, 'composer', outcome_class, attempts, COALESCE(completed_at, now())
        FROM ins WHERE outcome_class IN ('succeeded','refused','input_rejected','timeout','error')
        RETURNING 1
    )
    SELECT (SELECT count(*) FROM ins), (SELECT count(*) FROM perf) INTO v_steps, v_perf;

    RETURN jsonb_build_object('recorded', true, 'steps', v_steps, 'tool_performance', v_perf,
                              'unknown_tools', COALESCE(to_jsonb(v_unknown), '[]'::jsonb));
END $fn$;
REVOKE ALL ON FUNCTION composer_record_start(jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_record_phase(text, text, jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_record_finish(text, jsonb, jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION composer_record_start(jsonb), composer_record_phase(text, text, jsonb),
      composer_record_finish(text, jsonb, jsonb) TO service_role;

-- 7. composition views rewritten (no fan-out; outcome-based; orphan detection)
DROP VIEW IF EXISTS v_composition_success_rate;
CREATE VIEW v_composition_success_rate AS
SELECT date_trunc('day', created_at) AS day,
       count(*) AS total_compositions,
       count(*) FILTER (WHERE outcome = 'success') AS succeeded,
       count(*) FILTER (WHERE outcome = 'partial') AS partial,
       count(*) FILTER (WHERE outcome = 'failed') AS failed,
       count(*) FILTER (WHERE outcome = 'timeout') AS timed_out,
       count(*) FILTER (WHERE outcome IS NULL) AS unfinished,
       round((percentile_cont(0.5) WITHIN GROUP (ORDER BY total_latency_ms))::numeric, 0) AS p50_total_latency_ms,
       round((percentile_cont(0.95) WITHIN GROUP (ORDER BY total_latency_ms))::numeric, 0) AS p95_total_latency_ms
FROM composer_episodes
WHERE created_at > now() - interval '30 days'
GROUP BY 1 ORDER BY 1 DESC;
REVOKE ALL ON v_composition_success_rate FROM PUBLIC, anon, authenticated;
GRANT SELECT ON v_composition_success_rate TO service_role;

DROP VIEW IF EXISTS v_active_compositions;
CREATE VIEW v_active_compositions AS
SELECT composition_id, status, entry_point, session_id, created_at, last_phase_at,
       round(extract(epoch FROM now() - created_at) * 1000) AS elapsed_ms,
       (last_phase_at < now() - interval '10 minutes') AS abandoned
FROM composer_episodes
WHERE status IN ('PENDING','DECOMPOSING','PLANNING','EXECUTING','SYNTHESIZING')
ORDER BY created_at DESC;
REVOKE ALL ON v_active_compositions FROM PUBLIC, anon, authenticated;
GRANT SELECT ON v_active_compositions TO service_role;
