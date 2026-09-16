-- ============================================================================
-- ml/043 - refusal reason codes on the recording path (#2021, #2050)
--
-- WHY. A refusal's reason was not persisted. ml/041's composer_record_steps recorded the
-- outcome class and exception type, but nothing a reader could aggregate "why tools fail" by.
-- 043 records the CLOSED reason code (src/agents/tool_composer/reason_codes.py) and its
-- numeric details, and mirrors the code into tool_performance so get_tool_reliability can
-- report the most common refusal reason per tool.
--
-- WHAT IS STORED - AND WHAT IS NOT. Structure only, as ml/041 established. reason_code must
-- match the snake_case format; reason_details keeps at most 8 keys following the Python detail-key
-- convention (n_/is_/has_/share_ prefixes) whose values are JSON numbers or booleans, and drops
-- everything else. NO TEXT: the error_message slot of the
-- step INSERT stays NULL - that NULL is ml/041's guard against storing caller text, and three
-- tests pin it. The human sentence for a code is rendered at READ time from the Python
-- catalogue (owner decision D1', 2026-09-12), so adding a code never needs a migration.
--
-- Applied by scripts/run_migrations.sh inside one transaction with its ledger row
-- (key: ml/043_composer_refusal_reason_codes.sql). Re-applying it changes nothing.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. Columns
-- ---------------------------------------------------------------------------
ALTER TABLE composition_steps
    ADD COLUMN IF NOT EXISTS reason_code text,
    ADD COLUMN IF NOT EXISTS reason_details jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE tool_performance
    ADD COLUMN IF NOT EXISTS reason_code text;

-- A format guard, like ml/041's error_type guard. The closed member list lives in Python; a
-- copy here would drift the moment a member is added (test_lane_migration_files checks every
-- member passes this format).
ALTER TABLE composition_steps DROP CONSTRAINT IF EXISTS composition_steps_reason_code_format;
ALTER TABLE composition_steps ADD CONSTRAINT composition_steps_reason_code_format
    CHECK (reason_code IS NULL OR reason_code ~ '^[a-z][a-z0-9_]{0,63}$');
ALTER TABLE composition_steps DROP CONSTRAINT IF EXISTS composition_steps_reason_details_object;
ALTER TABLE composition_steps ADD CONSTRAINT composition_steps_reason_details_object
    CHECK (jsonb_typeof(reason_details) = 'object');
ALTER TABLE tool_performance DROP CONSTRAINT IF EXISTS tool_performance_reason_code_format;
ALTER TABLE tool_performance ADD CONSTRAINT tool_performance_reason_code_format
    CHECK (reason_code IS NULL OR reason_code ~ '^[a-z][a-z0-9_]{0,63}$');

COMMENT ON COLUMN composition_steps.reason_code IS
    'Closed-set code for why the step produced no result (#2021). Rendered to a sentence at read time; no message is stored.';
COMMENT ON COLUMN composition_steps.reason_details IS
    'Numbers and booleans only, at most 8 keys following the Python detail-key convention (n_/is_/has_/share_ prefixes), reduced by composer_structure_reason_details.';
COMMENT ON COLUMN tool_performance.reason_code IS
    'Mirror of composition_steps.reason_code, for get_tool_reliability.most_common_refusal_reason.';

-- ---------------------------------------------------------------------------
-- 2. The details reducer (same shape as ml/041's composer_structure_numbers)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION composer_structure_reason_details(p_value jsonb)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
SECURITY INVOKER
SET search_path = public
AS $fn$
    SELECT COALESCE(
        (SELECT jsonb_object_agg(k.key, k.value)
         FROM (SELECT e.key, e.value
               FROM jsonb_each(CASE WHEN jsonb_typeof(p_value) = 'object'
                                    THEN p_value ELSE '{}'::jsonb END) AS e
               WHERE e.key ~ '^(n|is|has|share)_[a-z0-9_]{1,58}$'
                 AND jsonb_typeof(e.value) IN ('number', 'boolean')
               ORDER BY e.key COLLATE "C"
               LIMIT 8) AS k),
        '{}'::jsonb);
$fn$;

-- ---------------------------------------------------------------------------
-- 3. composer_record_steps: carry the code and the details. The error_message slot stays NULL.
--    Copied from ml/041 with four edits, each marked "-- 043". CREATE OR REPLACE with an
--    unchanged signature keeps the function's grants; section 5 re-asserts them anyway.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION composer_record_steps(p_seed jsonb, p_steps jsonb)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $fn$
DECLARE
    v_id uuid;
    v_synthetic boolean;
    v_columns text[];
    v_ref_fields text[];
    v_known integer;
    v_recorded integer;
    v_unknown text[];
    v_mismatch text[];
BEGIN
    IF jsonb_typeof(p_steps) IS DISTINCT FROM 'array' THEN
        RAISE EXCEPTION 'composer_record_steps: steps must be an array';
    END IF;
    IF EXISTS (
        SELECT 1 FROM jsonb_array_elements(p_steps) AS s
        WHERE jsonb_typeof(s) <> 'object'
           OR jsonb_typeof(s->'step_number') IS DISTINCT FROM 'number'
           OR jsonb_typeof(s->'tool_name') IS DISTINCT FROM 'string'
    ) THEN
        RAISE EXCEPTION 'composer_record_steps: every step needs a step_number and a tool_name';
    END IF;

    v_id := composer_seed_episode(p_seed);
    SELECT is_synthetic INTO v_synthetic FROM composer_episodes WHERE episode_id = v_id;
    v_columns := composer_public_column_names();
    v_ref_fields := composer_registered_output_fields();

    SELECT COALESCE(array_agg(DISTINCT s->>'tool_name' ORDER BY s->>'tool_name'), '{}'::text[])
    INTO v_unknown
    FROM jsonb_array_elements(p_steps) AS s
    WHERE NOT EXISTS (SELECT 1 FROM tool_registry tr WHERE tr.name = s->>'tool_name');

    -- A registered tool whose registry schema does not declare a name the serializer sent (the
    -- serializer names only what the running code declares): the registry is stale, e.g. the
    -- startup sync has not landed. The step is still recorded, with those names reduced (the
    -- outcome is never deferred; privacy fails closed), and the tool is reported so the
    -- recorder runs the lazy sync and later compositions keep their names.
    SELECT COALESCE(array_agg(DISTINCT tr.name::text ORDER BY tr.name::text), '{}'::text[])
    INTO v_mismatch
    FROM jsonb_array_elements(p_steps) AS s
    JOIN tool_registry tr ON tr.name = s->>'tool_name'
    WHERE EXISTS (
              SELECT 1
              FROM jsonb_each(CASE WHEN jsonb_typeof(s->'input_params') = 'object'
                                   THEN s->'input_params' ELSE '{}'::jsonb END) AS e
              WHERE (e.key <> 'undeclared_params'
                     AND NOT (e.key = ANY (composer_schema_names(tr.input_schema))))
                 OR (jsonb_typeof(e.value) = 'object' AND e.value->>'type' = 'ref'
                     AND jsonb_typeof(e.value->'field') = 'string'
                     AND NOT ((e.value->>'field') = ANY (v_ref_fields))))
       OR EXISTS (
              SELECT 1
              FROM jsonb_array_elements(CASE WHEN jsonb_typeof(s->'output_keys'->'keys') = 'array'
                                             THEN s->'output_keys'->'keys' ELSE '[]'::jsonb END) AS k
              WHERE NOT ((k #>> '{}') = ANY (composer_schema_names(tr.output_schema))));

    SELECT count(*) INTO v_known
    FROM jsonb_array_elements(p_steps) AS s
    JOIN tool_registry tr ON tr.name = s->>'tool_name';

    WITH ins AS (
        INSERT INTO composition_steps (
            episode_id, step_number, step_name, tool_id, tool_name, input_params, output_result,
            depends_on_steps, serves_sub_question, started_at, completed_at, latency_ms, status,
            error_message, retry_count, outcome_class, attempts, cache_hit, error_type, reason_code, reason_details  -- 043
        )
        SELECT
            v_id,
            (s->>'step_number')::numeric::integer,
            'step_' || ((s->>'step_number')::numeric::integer),
            tr.tool_id,
            tr.name,
            composer_structure_params(s->'input_params', v_columns,
                                      composer_schema_names(tr.input_schema), v_ref_fields),
            composer_structure_output(s->'output_keys', composer_schema_names(tr.output_schema)),
            ARRAY(SELECT (e #>> '{}')::numeric::integer
                  FROM jsonb_array_elements(composer_structure_numbers(s->'depends_on_steps')) AS e),
            CASE WHEN (s->>'serves_sub_question') ~ '^[0-9]{1,6}$' THEN s->>'serves_sub_question' END,
            (s->>'started_at')::timestamptz,
            (s->>'completed_at')::timestamptz,
            CASE WHEN jsonb_typeof(s->'latency_ms') = 'number'
                 THEN (s->>'latency_ms')::double precision END,
            (CASE WHEN s->>'outcome_class' IN ('succeeded', 'cache_hit')
                  THEN 'COMPLETED' ELSE 'FAILED' END)::composition_status,
            -- error_message: NULL by design (ml/041's guard; D1' sends no text)  -- 043
            NULL,
            GREATEST(COALESCE((s->>'attempts')::numeric::integer, 0) - 1, 0),
            s->>'outcome_class',
            (s->>'attempts')::numeric::integer,
            COALESCE((s->>'cache_hit')::boolean, false),
            CASE WHEN (s->>'error_type') ~ '^[A-Za-z_][A-Za-z0-9_.]{0,99}$' THEN s->>'error_type' END,
            CASE WHEN (s->>'reason_code') ~ '^[a-z][a-z0-9_]{0,63}$' THEN s->>'reason_code' END,  -- 043
            composer_structure_reason_details(s->'reason_details')  -- 043
        FROM jsonb_array_elements(p_steps) AS s
        JOIN tool_registry tr ON tr.name = s->>'tool_name'
        ON CONFLICT (episode_id, step_number) DO NOTHING
        RETURNING 1
    )
    SELECT count(*) INTO v_recorded FROM ins;

    -- A performance row for every step of this payload whose tool was invoked, whether the
    -- step was inserted now or by an earlier (re-sent) call.
    INSERT INTO tool_performance (
        tool_id, tool_name, latency_ms, success, error_type, composition_id, step_id, called_by,
        outcome_class, attempts, is_synthetic, tool_version, executed_at, reason_code  -- 043
    )
    SELECT cs.tool_id, cs.tool_name, COALESCE(cs.latency_ms, 0), cs.outcome_class = 'succeeded',
           left(cs.error_type, 100), p_seed->>'composition_id', cs.step_id, 'composer',
           cs.outcome_class, cs.attempts, v_synthetic, tr.version, COALESCE(cs.completed_at, now()), cs.reason_code  -- 043
    FROM composition_steps cs
    JOIN tool_registry tr ON tr.tool_id = cs.tool_id
    WHERE cs.episode_id = v_id
      AND cs.step_number IN (SELECT (s->>'step_number')::numeric::integer
                             FROM jsonb_array_elements(p_steps) AS s)
      AND cs.outcome_class IN ('succeeded', 'refused', 'input_rejected', 'timeout', 'error')
    ON CONFLICT (step_id) DO NOTHING;

    UPDATE composer_episodes SET last_activity_at = now()
    WHERE episode_id = v_id
      AND status IN ('PENDING', 'DECOMPOSING', 'PLANNING', 'EXECUTING', 'SYNTHESIZING');

    RETURN jsonb_build_object(
        'recorded', v_recorded,
        'already_present', v_known - v_recorded,
        'unknown_tools', to_jsonb(v_unknown),
        'schema_mismatch_tools', to_jsonb(v_mismatch)
    );
END
$fn$;

-- ---------------------------------------------------------------------------
-- 4. get_tool_reliability: add most_common_refusal_reason, n_refused_coded and
--    n_most_common_refusal_reason. The return type changes, so the view that selects from it and
--    the function itself are dropped and recreated, and the grants and view comment that DROP
--    discards are re-applied in section 5.
--    The most common code and its count come from ONE ranking (refusal_codes, rn = 1), so they
--    cannot disagree. The count exists because the code alone cannot tell a 1-1-1 tie from a
--    3-of-3 majority, and would present a minority code as representative. Ties resolve to the
--    highest count, then the code ascending in "C" collation, so the winner does not depend on
--    the database locale. Both count exactly n_refused_coded's rows: coded refusals only.
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_tool_reliability;
DROP FUNCTION IF EXISTS get_tool_reliability(integer, boolean);

CREATE OR REPLACE FUNCTION get_tool_reliability(p_days integer DEFAULT 30, p_include_synthetic boolean DEFAULT true)
RETURNS TABLE (
    tool_name text,
    category text,
    source_agent text,
    version text,
    declared_latency_ms double precision,
    n_invoked bigint,
    n_succeeded bigint,
    n_refused bigint,
    n_health_failures bigint,
    n_health bigint,
    n_retried bigint,
    n_synthetic bigint,
    p50_latency_ms double precision,
    p95_latency_ms double precision,
    last_executed_at timestamptz,
    most_common_health_error text,
    most_common_refusal_reason text,  -- 043
    n_refused_coded bigint,  -- 043
    n_most_common_refusal_reason bigint  -- 043
)
LANGUAGE plpgsql
STABLE
SECURITY INVOKER
SET search_path = public
AS $fn$
#variable_conflict use_column
BEGIN
    IF p_days IS NULL OR p_days < 1 THEN
        RAISE EXCEPTION 'get_tool_reliability: p_days must be a positive number of days';
    END IF;
    IF p_include_synthetic IS NULL THEN
        RAISE EXCEPTION 'get_tool_reliability: p_include_synthetic must be true or false';
    END IF;

    RETURN QUERY
    WITH perf AS (
        SELECT tp.tool_id, tp.outcome_class, tp.attempts, tp.latency_ms, tp.executed_at,
               tp.error_type, tp.reason_code, tp.is_synthetic,  -- 043
               (tp.outcome_class IS NOT NULL AND (p_include_synthetic OR NOT tp.is_synthetic)) AS counted
        FROM tool_performance tp
        WHERE tp.executed_at > now() - make_interval(days => p_days)
    ),
    refusal_codes AS (  -- 043
        SELECT p.tool_id, p.reason_code, count(*) AS n,
               row_number() OVER (PARTITION BY p.tool_id ORDER BY count(*) DESC, p.reason_code COLLATE "C") AS rn
        FROM perf p
        WHERE p.counted AND p.outcome_class IN ('refused', 'input_rejected') AND p.reason_code IS NOT NULL
        GROUP BY p.tool_id, p.reason_code
    )
    SELECT
        tr.name::text,
        tr.category::text,
        tr.source_agent::text,
        tr.version::text,
        tr.avg_latency_ms,
        count(*) FILTER (WHERE p.counted),
        count(*) FILTER (WHERE p.counted AND p.outcome_class = 'succeeded'),
        count(*) FILTER (WHERE p.counted AND p.outcome_class IN ('refused', 'input_rejected')),
        count(*) FILTER (WHERE p.counted AND p.outcome_class IN ('timeout', 'error')),
        count(*) FILTER (WHERE p.counted AND p.outcome_class IN ('succeeded', 'timeout', 'error')),
        count(*) FILTER (WHERE p.counted AND p.outcome_class = 'succeeded' AND p.attempts > 1),
        count(*) FILTER (WHERE p.is_synthetic),
        percentile_cont(0.5) WITHIN GROUP (ORDER BY p.latency_ms)
            FILTER (WHERE p.counted AND p.outcome_class = 'succeeded'),
        percentile_cont(0.95) WITHIN GROUP (ORDER BY p.latency_ms)
            FILTER (WHERE p.counted AND p.outcome_class = 'succeeded'),
        max(p.executed_at) FILTER (WHERE p.counted),
        (mode() WITHIN GROUP (ORDER BY p.error_type)
            FILTER (WHERE p.counted AND p.outcome_class IN ('timeout', 'error')))::text
        ,rc.reason_code::text  -- 043
        ,count(*) FILTER (WHERE p.counted AND p.outcome_class IN ('refused', 'input_rejected') AND p.reason_code IS NOT NULL)  -- 043
        ,rc.n  -- 043
    FROM tool_registry tr
    LEFT JOIN perf p ON p.tool_id = tr.tool_id
    -- 043: at most one row per tool (rn = 1), so no aggregate above is multiplied.
    LEFT JOIN refusal_codes rc ON rc.tool_id = tr.tool_id AND rc.rn = 1
    WHERE tr.deprecated_at IS NULL
    GROUP BY tr.tool_id, tr.name, tr.category, tr.source_agent, tr.version, tr.avg_latency_ms,
             rc.reason_code, rc.n  -- 043
    ORDER BY tr.name;
END
$fn$;

CREATE VIEW v_tool_reliability AS
SELECT * FROM get_tool_reliability(30, true);

COMMENT ON VIEW v_tool_reliability IS
    'Per-tool measured reliability over 30 days, synthetic rows included (get_tool_reliability(30, true)).';

-- ---------------------------------------------------------------------------
-- 5. Access: service_role only, as ml/041 section 6
-- ---------------------------------------------------------------------------
REVOKE ALL ON FUNCTION composer_structure_reason_details(jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_record_steps(jsonb, jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION get_tool_reliability(integer, boolean) FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION
    composer_structure_reason_details(jsonb),
    composer_record_steps(jsonb, jsonb),
    get_tool_reliability(integer, boolean)
TO service_role;

REVOKE ALL ON v_tool_reliability FROM PUBLIC, anon, authenticated;
GRANT SELECT ON v_tool_reliability TO service_role;
