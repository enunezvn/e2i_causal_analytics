-- ============================================================================
-- E2I Causal Analytics - Migration ml/041_composer_learning_loop_recording.sql
-- Tool-composer learning loop, part 3 of 3 (spec
-- docs/superpowers/specs/2026-09-11-tool-composer-learning-loop-design.md §5, §6)
-- ============================================================================
--
-- ml/013 designed composer_episodes / composition_steps / tool_performance "for learning
-- from experience", but nothing ever wrote them. This migration makes them writable by the
-- composer's recorder and readable as measured reliability:
--
-- 1. Recording columns (§5.2): outcome, failed phase, error class, plan source, entry point,
--    provenance, audit identity, liveness; per-step outcome class / attempts / cache hit;
--    per-invocation outcome class, provenance and tool version. total_latency_ms becomes
--    nullable (the row is created before the total is known).
-- 2. Owner decision O3 (2026-09-11), recreated by rollback_041.sql:
--      - composer_episodes.query_embedding + idx_composer_episodes_embedding +
--        find_similar_compositions(): never written or called; similar-composition reuse is
--        served by episodic_memories (memory_hooks.find_similar_compositions).
--      - trg_log_step_performance + trigger_log_step_performance(): fires on UPDATE only (an
--        inserted terminal step is lost) and double-counts COMPLETED->FAILED. Its intent, one
--        performance row per finished step, moves into composer_record_steps().
-- 3. Structure-only helpers (§5.5). The Python serializer is the primary guard; these repeat
--    it inside the database: a {"type":"column"} name survives only if it is a column of a
--    public relation, anything that is not a known structure node is reduced to its type and
--    length, identifiers are positional, intents are normalized, and no error text is stored.
-- 4. Five seeded, idempotent recording RPCs (§5.4). Each begins by creating the episode from
--    the seed if it does not exist, so whichever write lands first records the composition,
--    and each can be re-sent after a lost response.
-- 5. get_tool_reliability(p_days, p_include_synthetic) (§6) and the three views, recreated
--    without the fan-out and with heartbeat-derived abandonment.
--
-- Every function is SECURITY INVOKER with a pinned search_path and is executable by
-- service_role only; the views are readable by service_role only (the default ACLs of role
-- postgres would otherwise grant anon/authenticated).
--
-- Applied by scripts/run_migrations.sh inside --single-transaction with its ledger row.
-- Re-applying it is a no-op for data.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 0. Views are recreated below; drop them first so the functions they read can be replaced.
-- ---------------------------------------------------------------------------
DROP VIEW IF EXISTS v_tool_reliability;
DROP VIEW IF EXISTS v_composition_success_rate;
DROP VIEW IF EXISTS v_active_compositions;

-- ---------------------------------------------------------------------------
-- 1. Recording columns
-- ---------------------------------------------------------------------------
ALTER TABLE composer_episodes
    ADD COLUMN IF NOT EXISTS audit_workflow_id uuid,
    ADD COLUMN IF NOT EXISTS outcome text,
    ADD COLUMN IF NOT EXISTS failed_phase text,
    ADD COLUMN IF NOT EXISTS error_type text,
    ADD COLUMN IF NOT EXISTS plan_source text,
    ADD COLUMN IF NOT EXISTS entry_point text,
    ADD COLUMN IF NOT EXISTS brand text,
    ADD COLUMN IF NOT EXISTS region text,
    ADD COLUMN IF NOT EXISTS is_synthetic boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS tools_executed integer,
    ADD COLUMN IF NOT EXISTS tools_succeeded integer,
    ADD COLUMN IF NOT EXISTS last_activity_at timestamptz NOT NULL DEFAULT now();
ALTER TABLE composer_episodes ALTER COLUMN total_latency_ms DROP NOT NULL;

ALTER TABLE composer_episodes DROP CONSTRAINT IF EXISTS composer_episodes_outcome_check;
ALTER TABLE composer_episodes ADD CONSTRAINT composer_episodes_outcome_check
    CHECK (outcome IS NULL OR outcome IN ('success', 'partial', 'failed', 'cancelled'));
ALTER TABLE composer_episodes DROP CONSTRAINT IF EXISTS composer_episodes_plan_source_check;
ALTER TABLE composer_episodes ADD CONSTRAINT composer_episodes_plan_source_check
    CHECK (plan_source IS NULL OR plan_source IN ('llm', 'plan_cache', 'kpi_deterministic'));
ALTER TABLE composer_episodes DROP CONSTRAINT IF EXISTS composer_episodes_failed_phase_check;
ALTER TABLE composer_episodes ADD CONSTRAINT composer_episodes_failed_phase_check
    CHECK (failed_phase IS NULL OR failed_phase IN ('decompose', 'plan', 'execute', 'synthesize'));

COMMENT ON COLUMN composer_episodes.outcome IS
    'success | partial | failed | cancelled. status keeps the terminal enum value (a cancel is FAILED + cancelled).';
COMMENT ON COLUMN composer_episodes.audit_workflow_id IS
    'audit_chain_entries.workflow_id of this run, from the composer''s local audit start; NULL when no audit row was written.';
COMMENT ON COLUMN composer_episodes.last_activity_at IS
    'Bumped by every recording RPC and the recorder heartbeat; v_active_compositions derives abandoned from it.';

ALTER TABLE composition_steps
    ADD COLUMN IF NOT EXISTS outcome_class text,
    ADD COLUMN IF NOT EXISTS attempts integer,
    ADD COLUMN IF NOT EXISTS cache_hit boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS error_type text;
ALTER TABLE composition_steps DROP CONSTRAINT IF EXISTS composition_steps_outcome_class_check;
ALTER TABLE composition_steps ADD CONSTRAINT composition_steps_outcome_class_check
    CHECK (outcome_class IS NULL OR outcome_class IN (
        'succeeded', 'cache_hit', 'refused', 'input_rejected', 'timeout', 'error',
        'plan_defect', 'dependency_unmet', 'circuit_open', 'not_registered'));

ALTER TABLE tool_performance
    ADD COLUMN IF NOT EXISTS outcome_class text,
    ADD COLUMN IF NOT EXISTS attempts integer,
    ADD COLUMN IF NOT EXISTS is_synthetic boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS tool_version text;
ALTER TABLE tool_performance DROP CONSTRAINT IF EXISTS tool_performance_outcome_class_check;
ALTER TABLE tool_performance ADD CONSTRAINT tool_performance_outcome_class_check
    CHECK (outcome_class IS NULL OR outcome_class IN (
        'succeeded', 'refused', 'input_rejected', 'timeout', 'error'));
-- One performance row per step. A full (not partial) index, so ON CONFLICT (step_id) infers
-- it; NULL step_ids stay distinct, so rows from non-composer callers are unaffected.
CREATE UNIQUE INDEX IF NOT EXISTS uq_tool_performance_step ON tool_performance (step_id);

-- ---------------------------------------------------------------------------
-- 2. O3 drops
-- ---------------------------------------------------------------------------
DROP FUNCTION IF EXISTS find_similar_compositions(vector, integer, double precision);
DROP INDEX IF EXISTS idx_composer_episodes_embedding;
ALTER TABLE composer_episodes DROP COLUMN IF EXISTS query_embedding;
DROP TRIGGER IF EXISTS trg_log_step_performance ON composition_steps;
DROP FUNCTION IF EXISTS trigger_log_step_performance();

-- ---------------------------------------------------------------------------
-- 3. Structure-only helpers
-- ---------------------------------------------------------------------------

-- Column names of public relations: the catalog allowlist for {"type":"column"} entries.
CREATE OR REPLACE FUNCTION composer_public_column_names()
RETURNS text[]
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = public
AS $fn$
    SELECT COALESCE(array_agg(d.name ORDER BY d.name COLLATE "C"), '{}'::text[])
    FROM (
        SELECT DISTINCT a.attname::text AS name
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relnamespace = 'public'::regnamespace
          AND c.relkind IN ('r', 'v', 'm', 'p', 'f')
          AND a.attnum > 0
          AND NOT a.attisdropped
    ) AS d;
$fn$;

-- One parameter value reduced to structure.
CREATE OR REPLACE FUNCTION composer_structure_value(p_value jsonb, p_columns text[])
RETURNS jsonb
LANGUAGE plpgsql
IMMUTABLE
SECURITY INVOKER
SET search_path = public
AS $fn$
DECLARE
    v_type text;
BEGIN
    CASE jsonb_typeof(p_value)
        WHEN 'number', 'boolean', 'null' THEN
            RETURN p_value;
        WHEN 'string' THEN
            RETURN jsonb_build_object('type', 'str', 'len', length(p_value #>> '{}'));
        WHEN 'array' THEN
            RETURN jsonb_build_object('type', 'list', 'len', jsonb_array_length(p_value));
        ELSE
            NULL;
    END CASE;
    IF p_value IS NULL THEN
        RETURN 'null'::jsonb;
    END IF;

    IF jsonb_typeof(p_value->'type') = 'string' THEN
        v_type := p_value->>'type';
        IF v_type = 'column' THEN
            IF jsonb_typeof(p_value->'name') = 'string' AND (p_value->>'name') = ANY (p_columns) THEN
                RETURN jsonb_build_object('type', 'column', 'name', p_value->>'name');
            END IF;
            RETURN jsonb_build_object(
                'type', 'str',
                'len', CASE WHEN jsonb_typeof(p_value->'name') = 'string'
                            THEN to_jsonb(length(p_value->>'name')) ELSE 'null'::jsonb END);
        ELSIF v_type IN ('str', 'list', 'dict') THEN
            RETURN jsonb_build_object(
                'type', v_type,
                'len', CASE WHEN jsonb_typeof(p_value->'len') = 'number'
                            THEN p_value->'len' ELSE 'null'::jsonb END);
        ELSIF v_type = 'frame' THEN
            RETURN jsonb_build_object(
                'type', 'frame',
                'rows', CASE WHEN jsonb_typeof(p_value->'rows') = 'number'
                             THEN p_value->'rows' ELSE 'null'::jsonb END,
                'columns', CASE WHEN jsonb_typeof(p_value->'columns') = 'number'
                                THEN p_value->'columns' ELSE 'null'::jsonb END);
        ELSIF v_type = 'ref' THEN
            RETURN jsonb_build_object(
                'type', 'ref',
                'step', CASE WHEN jsonb_typeof(p_value->'step') = 'number'
                             THEN p_value->'step' ELSE 'null'::jsonb END,
                'field', CASE WHEN jsonb_typeof(p_value->'field') = 'string'
                               AND (p_value->>'field') ~ '^[A-Za-z_][A-Za-z0-9_]{0,99}$'
                              THEN p_value->'field' ELSE 'null'::jsonb END);
        END IF;
    END IF;
    RETURN jsonb_build_object('type', 'dict', 'len', (SELECT count(*) FROM jsonb_object_keys(p_value)));
END
$fn$;

-- An input map: identifier keys (the tool's declared parameter names) to structure values.
CREATE OR REPLACE FUNCTION composer_structure_params(p_params jsonb, p_columns text[])
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
SECURITY INVOKER
SET search_path = public
AS $fn$
    SELECT CASE WHEN jsonb_typeof(p_params) = 'object' THEN
        COALESCE(
            (SELECT jsonb_object_agg(e.key, composer_structure_value(e.value, p_columns))
             FROM jsonb_each(p_params) AS e
             WHERE e.key ~ '^[A-Za-z_][A-Za-z0-9_]{0,99}$'),
            '{}'::jsonb)
    ELSE '{}'::jsonb END;
$fn$;

-- A JSON array reduced to its number elements, in order.
CREATE OR REPLACE FUNCTION composer_structure_numbers(p_value jsonb)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
SECURITY INVOKER
SET search_path = public
AS $fn$
    SELECT COALESCE(
        (SELECT jsonb_agg(x.e ORDER BY x.o)
         FROM jsonb_array_elements(CASE WHEN jsonb_typeof(p_value) = 'array'
                                        THEN p_value ELSE '[]'::jsonb END)
              WITH ORDINALITY AS x(e, o)
         WHERE jsonb_typeof(x.e) = 'number'),
        '[]'::jsonb);
$fn$;

-- Parallel groups: a list of lists of step numbers, or [] if it is anything else.
CREATE OR REPLACE FUNCTION composer_structure_groups(p_value jsonb)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
SECURITY INVOKER
SET search_path = public
AS $fn$
    SELECT CASE
        WHEN jsonb_typeof(p_value) IS DISTINCT FROM 'array' THEN '[]'::jsonb
        WHEN EXISTS (
            SELECT 1 FROM jsonb_array_elements(p_value) AS g
            WHERE CASE WHEN jsonb_typeof(g) <> 'array' THEN true
                       ELSE EXISTS (SELECT 1 FROM jsonb_array_elements(g) AS e
                                    WHERE jsonb_typeof(e) <> 'number')
                  END
        ) THEN '[]'::jsonb
        ELSE p_value
    END;
$fn$;

-- Sub-questions: positional index and an intent from the decomposer's vocabulary only.
CREATE OR REPLACE FUNCTION composer_structure_sub_questions(p_value jsonb)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
SECURITY INVOKER
SET search_path = public
AS $fn$
    SELECT COALESCE(
        (SELECT jsonb_agg(
                    jsonb_build_object(
                        'index', to_jsonb(x.o - 1),
                        'intent', CASE WHEN upper(x.q->>'intent') IN
                                            ('CAUSAL', 'COMPARATIVE', 'PREDICTIVE', 'DESCRIPTIVE', 'EXPERIMENTAL')
                                       THEN to_jsonb(upper(x.q->>'intent'))
                                       ELSE '"OTHER"'::jsonb END)
                    ORDER BY x.o)
         FROM jsonb_array_elements(CASE WHEN jsonb_typeof(p_value) = 'array'
                                        THEN p_value ELSE '[]'::jsonb END)
              WITH ORDINALITY AS x(q, o)),
        '[]'::jsonb);
$fn$;

-- A step's output: the registered output-model field names and a count of other keys.
CREATE OR REPLACE FUNCTION composer_structure_output(p_value jsonb)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
SECURITY INVOKER
SET search_path = public
AS $fn$
    SELECT jsonb_build_object(
        'keys', COALESCE(
            (SELECT jsonb_agg(x.k ORDER BY x.o)
             FROM jsonb_array_elements(CASE WHEN jsonb_typeof(p_value->'keys') = 'array'
                                            THEN p_value->'keys' ELSE '[]'::jsonb END)
                  WITH ORDINALITY AS x(k, o)
             WHERE jsonb_typeof(x.k) = 'string'
               AND (x.k #>> '{}') ~ '^[A-Za-z_][A-Za-z0-9_]{0,99}$'),
            '[]'::jsonb),
        'other_keys', CASE WHEN jsonb_typeof(p_value->'other_keys') = 'number'
                           THEN p_value->'other_keys' ELSE 'null'::jsonb END);
$fn$;

-- A plan: per step its number, registered tool name, dependency step numbers and input map.
CREATE OR REPLACE FUNCTION composer_structure_plan(p_plan jsonb, p_columns text[])
RETURNS jsonb
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = public
AS $fn$
    SELECT CASE WHEN jsonb_typeof(p_plan) IS DISTINCT FROM 'object' THEN '{}'::jsonb ELSE
        jsonb_build_object(
            'steps', COALESCE(
                (SELECT jsonb_agg(
                            jsonb_build_object(
                                'step_number', CASE WHEN jsonb_typeof(x.s->'step_number') = 'number'
                                                    THEN x.s->'step_number' ELSE 'null'::jsonb END,
                                'tool_name', CASE WHEN EXISTS (SELECT 1 FROM tool_registry tr
                                                               WHERE tr.name = x.s->>'tool_name')
                                                  THEN x.s->'tool_name' ELSE 'null'::jsonb END,
                                'depends_on_steps', composer_structure_numbers(x.s->'depends_on_steps'),
                                'input_params', composer_structure_params(x.s->'input_params', p_columns))
                            ORDER BY x.o)
                 FROM jsonb_array_elements(CASE WHEN jsonb_typeof(p_plan->'steps') = 'array'
                                                THEN p_plan->'steps' ELSE '[]'::jsonb END)
                      WITH ORDINALITY AS x(s, o)),
                '[]'::jsonb),
            'execution_order_repaired',
                CASE WHEN jsonb_typeof(p_plan->'execution_order_repaired') = 'string'
                      AND (p_plan->>'execution_order_repaired') ~ '^[a-z_]{1,50}$'
                     THEN p_plan->'execution_order_repaired' ELSE 'null'::jsonb END)
    END;
$fn$;

-- The episode fields a phase patch or finish snapshot may carry, each reduced to structure.
-- Only keys present in the input appear in the output (a phase write leaves absent fields).
CREATE OR REPLACE FUNCTION composer_episode_patch(p_patch jsonb, p_columns text[])
RETURNS jsonb
LANGUAGE plpgsql
STABLE
SECURITY INVOKER
SET search_path = public
AS $fn$
DECLARE
    v_out jsonb := '{}'::jsonb;
    v_key text;
BEGIN
    IF jsonb_typeof(p_patch) IS DISTINCT FROM 'object' THEN
        RETURN v_out;
    END IF;
    FOREACH v_key IN ARRAY ARRAY['decompose_latency_ms', 'plan_latency_ms', 'execute_latency_ms',
                                 'synthesize_latency_ms', 'total_latency_ms', 'tools_executed',
                                 'tools_succeeded'] LOOP
        IF p_patch ? v_key THEN
            v_out := v_out || jsonb_build_object(
                v_key, CASE WHEN jsonb_typeof(p_patch->v_key) = 'number'
                            THEN p_patch->v_key ELSE 'null'::jsonb END);
        END IF;
    END LOOP;
    -- Constrained by CHECKs / the status enum: an unknown value raises rather than being stored.
    FOREACH v_key IN ARRAY ARRAY['status', 'outcome', 'failed_phase', 'plan_source'] LOOP
        IF p_patch ? v_key THEN
            v_out := v_out || jsonb_build_object(
                v_key, CASE WHEN jsonb_typeof(p_patch->v_key) = 'string'
                            THEN p_patch->v_key ELSE 'null'::jsonb END);
        END IF;
    END LOOP;
    IF p_patch ? 'error_type' THEN
        v_out := v_out || jsonb_build_object(
            'error_type', CASE WHEN jsonb_typeof(p_patch->'error_type') = 'string'
                                AND (p_patch->>'error_type') ~ '^[A-Za-z_][A-Za-z0-9_.]{0,99}$'
                               THEN p_patch->'error_type' ELSE 'null'::jsonb END);
    END IF;
    IF p_patch ? 'sub_questions' THEN
        v_out := v_out || jsonb_build_object(
            'sub_questions', composer_structure_sub_questions(p_patch->'sub_questions'));
    END IF;
    IF p_patch ? 'tool_plan' THEN
        v_out := v_out || jsonb_build_object(
            'tool_plan', composer_structure_plan(p_patch->'tool_plan', p_columns));
    END IF;
    IF p_patch ? 'parallelizable_groups' THEN
        v_out := v_out || jsonb_build_object(
            'parallelizable_groups', composer_structure_groups(p_patch->'parallelizable_groups'));
    END IF;
    RETURN v_out;
END
$fn$;

-- ---------------------------------------------------------------------------
-- 4. Seeded, idempotent recording RPCs
-- ---------------------------------------------------------------------------

-- Creates the episode from the seed when it does not exist; returns its id either way.
CREATE OR REPLACE FUNCTION composer_seed_episode(p_seed jsonb)
RETURNS uuid
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $fn$
DECLARE
    v_id uuid;
BEGIN
    IF jsonb_typeof(p_seed) IS DISTINCT FROM 'object'
       OR jsonb_typeof(p_seed->'composition_id') IS DISTINCT FROM 'string'
       OR (p_seed->>'composition_id') = ''
       OR length(p_seed->>'composition_id') > 100 THEN
        RAISE EXCEPTION 'composer recording: a seed with a composition_id is required';
    END IF;

    INSERT INTO composer_episodes (
        composition_id, query_text, status, session_id, user_id, entry_point, brand, region,
        audit_workflow_id, is_synthetic, last_activity_at
    )
    VALUES (
        p_seed->>'composition_id',
        left(COALESCE(p_seed->>'query_text', ''), 1000),
        'DECOMPOSING',
        left(p_seed->>'session_id', 100),
        left(p_seed->>'user_id', 100),
        left(p_seed->>'entry_point', 50),
        left(p_seed->>'brand', 100),
        left(p_seed->>'region', 100),
        (p_seed->>'audit_workflow_id')::uuid,
        COALESCE((p_seed->>'is_synthetic')::boolean, false),
        now()
    )
    ON CONFLICT (composition_id) DO NOTHING;

    SELECT episode_id INTO v_id FROM composer_episodes WHERE composition_id = p_seed->>'composition_id';
    RETURN v_id;
END
$fn$;

CREATE OR REPLACE FUNCTION composer_record_start(p_seed jsonb)
RETURNS uuid
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $fn$
DECLARE
    v_id uuid;
BEGIN
    v_id := composer_seed_episode(p_seed);
    UPDATE composer_episodes SET last_activity_at = now()
    WHERE episode_id = v_id
      AND status IN ('PENDING', 'DECOMPOSING', 'PLANNING', 'EXECUTING', 'SYNTHESIZING');
    RETURN v_id;
END
$fn$;

CREATE OR REPLACE FUNCTION composer_record_phase(p_seed jsonb, p_status text, p_patch jsonb)
RETURNS boolean
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $fn$
DECLARE
    v_id uuid;
    v jsonb;
BEGIN
    IF p_status IS NULL OR p_status NOT IN ('DECOMPOSING', 'PLANNING', 'EXECUTING', 'SYNTHESIZING') THEN
        RAISE EXCEPTION 'composer_record_phase: phase status must be DECOMPOSING, PLANNING, EXECUTING or SYNTHESIZING';
    END IF;
    v_id := composer_seed_episode(p_seed);
    v := composer_episode_patch(p_patch, composer_public_column_names()) - 'status' - 'outcome'
         - 'failed_phase' - 'error_type' - 'total_latency_ms' - 'tools_executed' - 'tools_succeeded';

    UPDATE composer_episodes SET
        status = p_status::composition_status,
        decompose_latency_ms = CASE WHEN v ? 'decompose_latency_ms'
                                    THEN (v->>'decompose_latency_ms')::double precision
                                    ELSE decompose_latency_ms END,
        plan_latency_ms = CASE WHEN v ? 'plan_latency_ms'
                               THEN (v->>'plan_latency_ms')::double precision
                               ELSE plan_latency_ms END,
        execute_latency_ms = CASE WHEN v ? 'execute_latency_ms'
                                  THEN (v->>'execute_latency_ms')::double precision
                                  ELSE execute_latency_ms END,
        synthesize_latency_ms = CASE WHEN v ? 'synthesize_latency_ms'
                                     THEN (v->>'synthesize_latency_ms')::double precision
                                     ELSE synthesize_latency_ms END,
        sub_questions = CASE WHEN v ? 'sub_questions' THEN v->'sub_questions' ELSE sub_questions END,
        tool_plan = CASE WHEN v ? 'tool_plan' THEN v->'tool_plan' ELSE tool_plan END,
        parallelizable_groups = CASE WHEN v ? 'parallelizable_groups'
                                     THEN v->'parallelizable_groups' ELSE parallelizable_groups END,
        plan_source = CASE WHEN v ? 'plan_source' THEN v->>'plan_source' ELSE plan_source END,
        last_activity_at = now()
    WHERE episode_id = v_id
      AND status IN ('PENDING', 'DECOMPOSING', 'PLANNING', 'EXECUTING', 'SYNTHESIZING');
    RETURN FOUND;
END
$fn$;

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
    v_known integer;
    v_recorded integer;
    v_unknown text[];
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

    SELECT COALESCE(array_agg(DISTINCT s->>'tool_name' ORDER BY s->>'tool_name'), '{}'::text[])
    INTO v_unknown
    FROM jsonb_array_elements(p_steps) AS s
    WHERE NOT EXISTS (SELECT 1 FROM tool_registry tr WHERE tr.name = s->>'tool_name');

    SELECT count(*) INTO v_known
    FROM jsonb_array_elements(p_steps) AS s
    JOIN tool_registry tr ON tr.name = s->>'tool_name';

    WITH ins AS (
        INSERT INTO composition_steps (
            episode_id, step_number, step_name, tool_id, tool_name, input_params, output_result,
            depends_on_steps, serves_sub_question, started_at, completed_at, latency_ms, status,
            error_message, retry_count, outcome_class, attempts, cache_hit, error_type
        )
        SELECT
            v_id,
            (s->>'step_number')::numeric::integer,
            'step_' || ((s->>'step_number')::numeric::integer),
            tr.tool_id,
            tr.name,
            composer_structure_params(s->'input_params', v_columns),
            composer_structure_output(s->'output_keys'),
            ARRAY(SELECT (e #>> '{}')::numeric::integer
                  FROM jsonb_array_elements(composer_structure_numbers(s->'depends_on_steps')) AS e),
            CASE WHEN (s->>'serves_sub_question') ~ '^[0-9]{1,6}$' THEN s->>'serves_sub_question' END,
            (s->>'started_at')::timestamptz,
            (s->>'completed_at')::timestamptz,
            CASE WHEN jsonb_typeof(s->'latency_ms') = 'number'
                 THEN (s->>'latency_ms')::double precision END,
            (CASE WHEN s->>'outcome_class' IN ('succeeded', 'cache_hit')
                  THEN 'COMPLETED' ELSE 'FAILED' END)::composition_status,
            NULL,
            GREATEST(COALESCE((s->>'attempts')::numeric::integer, 0) - 1, 0),
            s->>'outcome_class',
            (s->>'attempts')::numeric::integer,
            COALESCE((s->>'cache_hit')::boolean, false),
            CASE WHEN (s->>'error_type') ~ '^[A-Za-z_][A-Za-z0-9_.]{0,99}$' THEN s->>'error_type' END
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
        outcome_class, attempts, is_synthetic, tool_version, executed_at
    )
    SELECT cs.tool_id, cs.tool_name, COALESCE(cs.latency_ms, 0), cs.outcome_class = 'succeeded',
           left(cs.error_type, 100), p_seed->>'composition_id', cs.step_id, 'composer',
           cs.outcome_class, cs.attempts, v_synthetic, tr.version, COALESCE(cs.completed_at, now())
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
        'unknown_tools', to_jsonb(v_unknown)
    );
END
$fn$;

CREATE OR REPLACE FUNCTION composer_record_heartbeat(p_seed jsonb)
RETURNS boolean
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $fn$
DECLARE
    v_id uuid;
BEGIN
    v_id := composer_seed_episode(p_seed);
    UPDATE composer_episodes SET last_activity_at = now()
    WHERE episode_id = v_id
      AND status IN ('PENDING', 'DECOMPOSING', 'PLANNING', 'EXECUTING', 'SYNTHESIZING');
    RETURN FOUND;
END
$fn$;

CREATE OR REPLACE FUNCTION composer_record_finish(p_seed jsonb, p_final jsonb)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $fn$
DECLARE
    v_id uuid;
    v jsonb;
BEGIN
    IF jsonb_typeof(p_final) IS DISTINCT FROM 'object'
       OR jsonb_typeof(p_final->'status') IS DISTINCT FROM 'string'
       OR (p_final->>'status') NOT IN ('COMPLETED', 'FAILED', 'TIMEOUT') THEN
        RAISE EXCEPTION 'composer_record_finish: a terminal status (COMPLETED, FAILED or TIMEOUT) is required';
    END IF;
    IF jsonb_typeof(p_final->'outcome') IS DISTINCT FROM 'string' THEN
        RAISE EXCEPTION 'composer_record_finish: an outcome is required';
    END IF;

    v_id := composer_seed_episode(p_seed);
    v := composer_episode_patch(p_final, composer_public_column_names());

    -- The complete snapshot: every phase field a lost phase write carried is restored here.
    UPDATE composer_episodes SET
        status = (v->>'status')::composition_status,
        outcome = v->>'outcome',
        failed_phase = v->>'failed_phase',
        error_type = v->>'error_type',
        total_latency_ms = (v->>'total_latency_ms')::double precision,
        decompose_latency_ms = COALESCE((v->>'decompose_latency_ms')::double precision, decompose_latency_ms),
        plan_latency_ms = COALESCE((v->>'plan_latency_ms')::double precision, plan_latency_ms),
        execute_latency_ms = COALESCE((v->>'execute_latency_ms')::double precision, execute_latency_ms),
        synthesize_latency_ms = COALESCE((v->>'synthesize_latency_ms')::double precision, synthesize_latency_ms),
        sub_questions = COALESCE(v->'sub_questions', sub_questions),
        tool_plan = COALESCE(v->'tool_plan', tool_plan),
        parallelizable_groups = COALESCE(v->'parallelizable_groups', parallelizable_groups),
        plan_source = COALESCE(v->>'plan_source', plan_source),
        tools_executed = (v->>'tools_executed')::numeric::integer,
        tools_succeeded = (v->>'tools_succeeded')::numeric::integer,
        completed_at = now(),
        last_activity_at = now()
    WHERE episode_id = v_id
      AND status IN ('PENDING', 'DECOMPOSING', 'PLANNING', 'EXECUTING', 'SYNTHESIZING');
    IF NOT FOUND THEN
        RETURN jsonb_build_object('recorded', false, 'already_terminal', true);
    END IF;
    RETURN jsonb_build_object('recorded', true);
END
$fn$;

-- Recorded steps of the given compositions, keyed by composition_id, in step order (episodic
-- references say which tools worked).
CREATE OR REPLACE FUNCTION composer_steps_for(p_composition_ids text[])
RETURNS jsonb
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = public
AS $fn$
    SELECT COALESCE(jsonb_object_agg(e.composition_id, COALESCE(s.steps, '[]'::jsonb)), '{}'::jsonb)
    FROM composer_episodes e
    LEFT JOIN LATERAL (
        SELECT jsonb_agg(
                   jsonb_build_object('step_number', cs.step_number, 'tool_name', cs.tool_name,
                                      'outcome_class', cs.outcome_class)
                   ORDER BY cs.step_number) AS steps
        FROM composition_steps cs
        WHERE cs.episode_id = e.episode_id
    ) AS s ON true
    WHERE e.composition_id = ANY (p_composition_ids);
$fn$;

-- ---------------------------------------------------------------------------
-- 5. Reliability and views
-- ---------------------------------------------------------------------------

-- One row per active tool. n_health = succeeded + health failures (timeout, error) is the
-- denominator of the reliability rule; refusals (refused, input_rejected) never enter it.
-- p_include_synthetic = false excludes rows recorded under a deployment that includes
-- synthetic substrate (n_synthetic still shows them). Latency is of succeeded invocations only.
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
    most_common_health_error text
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
               tp.error_type, tp.is_synthetic,
               (tp.outcome_class IS NOT NULL AND (p_include_synthetic OR NOT tp.is_synthetic)) AS counted
        FROM tool_performance tp
        WHERE tp.executed_at > now() - make_interval(days => p_days)
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
    FROM tool_registry tr
    LEFT JOIN perf p ON p.tool_id = tr.tool_id
    WHERE tr.deprecated_at IS NULL
    GROUP BY tr.tool_id, tr.name, tr.category, tr.source_agent, tr.version, tr.avg_latency_ms
    ORDER BY tr.name;
END
$fn$;

CREATE VIEW v_tool_reliability AS
SELECT * FROM get_tool_reliability(30, true);

COMMENT ON VIEW v_tool_reliability IS
    'Per-tool measured reliability over 30 days, synthetic rows included (get_tool_reliability(30, true)).';

CREATE VIEW v_composition_success_rate AS
SELECT
    date_trunc('day', created_at) AS day,
    count(*) AS total_compositions,
    count(*) FILTER (WHERE outcome = 'success') AS succeeded,
    count(*) FILTER (WHERE outcome = 'partial') AS partial,
    count(*) FILTER (WHERE outcome = 'failed') AS failed,
    count(*) FILTER (WHERE outcome = 'cancelled') AS cancelled,
    count(*) FILTER (WHERE outcome IS NULL) AS unfinished,
    count(*) FILTER (WHERE plan_source = 'llm') AS plan_llm,
    count(*) FILTER (WHERE plan_source = 'plan_cache') AS plan_cache,
    count(*) FILTER (WHERE plan_source = 'kpi_deterministic') AS plan_kpi_deterministic,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY total_latency_ms) AS p50_total_latency_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY total_latency_ms) AS p95_total_latency_ms
FROM composer_episodes
WHERE created_at > now() - interval '30 days'
GROUP BY date_trunc('day', created_at)
ORDER BY day DESC;

COMMENT ON VIEW v_composition_success_rate IS
    'Daily composition counts by outcome and plan source over 30 days; one row per episode, no per-tool fan-out.';

CREATE VIEW v_active_compositions AS
SELECT
    ce.composition_id,
    ce.status,
    ce.entry_point,
    ce.session_id,
    ce.created_at,
    ce.last_activity_at,
    round(extract(epoch FROM now() - ce.created_at) * 1000) AS elapsed_ms,
    (ce.last_activity_at < now() - interval '5 minutes') AS abandoned,
    (SELECT count(*) FROM composition_steps cs WHERE cs.episode_id = ce.episode_id) AS steps_recorded
FROM composer_episodes ce
WHERE ce.status IN ('PENDING', 'DECOMPOSING', 'PLANNING', 'EXECUTING', 'SYNTHESIZING')
ORDER BY ce.created_at DESC;

COMMENT ON VIEW v_active_compositions IS
    'Compositions not yet terminal; abandoned = no recording write or heartbeat for 5 minutes (five missed heartbeats).';

-- ---------------------------------------------------------------------------
-- 6. Access: service_role only
-- ---------------------------------------------------------------------------
REVOKE ALL ON FUNCTION composer_public_column_names() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_structure_value(jsonb, text[]) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_structure_params(jsonb, text[]) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_structure_numbers(jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_structure_groups(jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_structure_sub_questions(jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_structure_output(jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_structure_plan(jsonb, text[]) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_episode_patch(jsonb, text[]) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_seed_episode(jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_record_start(jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_record_phase(jsonb, text, jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_record_steps(jsonb, jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_record_heartbeat(jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_record_finish(jsonb, jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION composer_steps_for(text[]) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION get_tool_reliability(integer, boolean) FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION
    composer_public_column_names(),
    composer_structure_value(jsonb, text[]),
    composer_structure_params(jsonb, text[]),
    composer_structure_numbers(jsonb),
    composer_structure_groups(jsonb),
    composer_structure_sub_questions(jsonb),
    composer_structure_output(jsonb),
    composer_structure_plan(jsonb, text[]),
    composer_episode_patch(jsonb, text[]),
    composer_seed_episode(jsonb),
    composer_record_start(jsonb),
    composer_record_phase(jsonb, text, jsonb),
    composer_record_steps(jsonb, jsonb),
    composer_record_heartbeat(jsonb),
    composer_record_finish(jsonb, jsonb),
    composer_steps_for(text[]),
    get_tool_reliability(integer, boolean)
TO service_role;

REVOKE ALL ON v_tool_reliability FROM PUBLIC, anon, authenticated;
REVOKE ALL ON v_composition_success_rate FROM PUBLIC, anon, authenticated;
REVOKE ALL ON v_active_compositions FROM PUBLIC, anon, authenticated;
GRANT SELECT ON v_tool_reliability, v_composition_success_rate, v_active_compositions TO service_role;
