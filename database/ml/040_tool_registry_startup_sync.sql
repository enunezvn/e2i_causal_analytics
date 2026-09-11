-- ============================================================================
-- E2I Causal Analytics - Migration ml/040_tool_registry_startup_sync.sql
-- Tool-composer learning loop, part 2 of 3 (spec
-- docs/superpowers/specs/2026-09-11-tool-composer-learning-loop-design.md §4, §6)
-- ============================================================================
--
-- Replaces the migration-per-schema-change regime (#2003 / ml/037) with a runtime sync:
-- the API calls sync_tool_registry() at startup with the tools the running code
-- registers, so tool_registry / tool_dependencies describe the code that is running.
--
-- 1. valid_agent follows the maintained agent taxonomy (e2i_agent_name) instead of a
--    hard-coded list of six agents, so a new agent never needs an ml migration.
-- 2. Owner decision O3 (2026-09-11), dropped with their dependents, recreated by
--    rollback_040.sql:
--      - update_tool_registry_metrics() + tool_registry.success_rate. The function wrote
--        success_rate from any single run (no minimum n: one failed run wrote 0) and never
--        had a caller; the column holds seed defaults that look measured (0.95) and nothing
--        reads it. Measured reliability lives in get_tool_reliability() (ml/041).
--      - get_tool_execution_order(text[]). It orders tool NAMES by tool-level dependencies,
--        returns {NULL} for root tools, and never had a caller; plans are ordered at step
--        level in-process (ExecutionPlan.get_execution_order()).
-- 3. sync_tool_registry(p_tools, p_dependencies, p_max_deprecations): validates, then
--    upserts tools, deprecates tools the code no longer registers (never deletes: steps
--    reference them), and makes dependencies exactly the payload set. Atomic, idempotent
--    (unchanged rows are not rewritten), serialised by an advisory lock, service_role only.
--
-- Applied by scripts/run_migrations.sh inside --single-transaction with its ledger row.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. Agent CHECK follows e2i_agent_name
-- ---------------------------------------------------------------------------
ALTER TABLE tool_registry DROP CONSTRAINT IF EXISTS valid_agent;
ALTER TABLE tool_registry ADD CONSTRAINT valid_agent
    CHECK (source_agent::text = ANY (enum_range(NULL::e2i_agent_name)::text[]));

COMMENT ON COLUMN tool_registry.avg_latency_ms IS
    'DECLARED latency baseline of the registered tool (avg_execution_ms), written by sync_tool_registry() at API startup. Measured latency is read from get_tool_reliability().';

-- ---------------------------------------------------------------------------
-- 2. O3 drops (function before the column it reads)
-- ---------------------------------------------------------------------------
DROP FUNCTION IF EXISTS update_tool_registry_metrics();
ALTER TABLE tool_registry DROP COLUMN IF EXISTS success_rate;
DROP FUNCTION IF EXISTS get_tool_execution_order(text[]);

-- ---------------------------------------------------------------------------
-- 3. sync_tool_registry
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION sync_tool_registry(
    p_tools jsonb,
    p_dependencies jsonb,
    p_max_deprecations integer DEFAULT 3
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public
AS $fn$
DECLARE
    v_names text[];
    v_bad text;
    v_would_deprecate integer;
    v_inserted integer;
    v_updated integer;
    v_deprecated integer;
    v_dep_upserted integer;
    v_dep_deleted integer;
BEGIN
    -- Validation reads only the arguments; nothing below it runs on a bad payload.
    IF p_tools IS NULL OR jsonb_typeof(p_tools) <> 'array' THEN
        RAISE EXCEPTION 'sync_tool_registry: empty tool payload refused (not an array)';
    END IF;
    IF jsonb_array_length(p_tools) = 0 THEN
        RAISE EXCEPTION 'sync_tool_registry: empty tool payload refused';
    END IF;
    IF p_dependencies IS NULL OR jsonb_typeof(p_dependencies) <> 'array' THEN
        RAISE EXCEPTION 'sync_tool_registry: dependency payload must be an array';
    END IF;
    IF p_max_deprecations IS NULL OR p_max_deprecations < 0 THEN
        RAISE EXCEPTION 'sync_tool_registry: p_max_deprecations must be a non-negative integer';
    END IF;

    IF EXISTS (
        SELECT 1 FROM jsonb_array_elements(p_tools) AS t
        WHERE jsonb_typeof(t) <> 'object'
           OR jsonb_typeof(t->'name') IS DISTINCT FROM 'string'
           OR t->>'name' = ''
    ) THEN
        RAISE EXCEPTION 'sync_tool_registry: tool entry without a name refused';
    END IF;

    SELECT string_agg(d.name, ', ' ORDER BY d.name) INTO v_bad
    FROM (
        SELECT t->>'name' AS name FROM jsonb_array_elements(p_tools) AS t
        GROUP BY 1 HAVING count(*) > 1
    ) AS d;
    IF v_bad IS NOT NULL THEN
        RAISE EXCEPTION 'sync_tool_registry: duplicate tool name(s) refused: %', v_bad;
    END IF;

    v_names := ARRAY(SELECT t->>'name' FROM jsonb_array_elements(p_tools) AS t);

    IF EXISTS (
        SELECT 1 FROM jsonb_array_elements(p_dependencies) AS d
        WHERE jsonb_typeof(d) <> 'object'
           OR jsonb_typeof(d->'consumer') IS DISTINCT FROM 'string'
           OR jsonb_typeof(d->'producer') IS DISTINCT FROM 'string'
    ) THEN
        RAISE EXCEPTION 'sync_tool_registry: dependency entry without a consumer and producer name refused';
    END IF;

    SELECT string_agg(DISTINCT e.endpoint, ', ' ORDER BY e.endpoint) INTO v_bad
    FROM jsonb_array_elements(p_dependencies) AS d
    CROSS JOIN LATERAL (VALUES (d->>'consumer'), (d->>'producer')) AS e(endpoint)
    WHERE NOT (e.endpoint = ANY (v_names));
    IF v_bad IS NOT NULL THEN
        RAISE EXCEPTION 'sync_tool_registry: dependency endpoint(s) not in the tool payload: %', v_bad;
    END IF;

    SELECT string_agg(format('%s<-%s', p.consumer, p.producer), ', ' ORDER BY p.consumer, p.producer)
    INTO v_bad
    FROM (
        SELECT d->>'consumer' AS consumer, d->>'producer' AS producer
        FROM jsonb_array_elements(p_dependencies) AS d
        GROUP BY 1, 2 HAVING count(*) > 1
    ) AS p;
    IF v_bad IS NOT NULL THEN
        RAISE EXCEPTION 'sync_tool_registry: duplicate dependency pair(s) refused: %', v_bad;
    END IF;

    -- One sync at a time (two API workers boot together; a lazy sync may overlap). Taken
    -- before the deprecation count so the guard sees the state the writes will change.
    PERFORM pg_advisory_xact_lock(hashtext('sync_tool_registry'));

    SELECT count(*) INTO v_would_deprecate
    FROM tool_registry
    WHERE deprecated_at IS NULL AND NOT (name::text = ANY (v_names));
    IF v_would_deprecate > p_max_deprecations THEN
        RAISE EXCEPTION 'sync_tool_registry: payload would deprecate % active tools, limit %; refused before any write',
            v_would_deprecate, p_max_deprecations;
    END IF;

    WITH payload AS (
        SELECT *
        FROM jsonb_to_recordset(p_tools) AS t(
            name text, description text, category text, source_agent text,
            input_schema jsonb, output_schema jsonb, avg_latency_ms double precision, version text
        )
    ), up AS (
        INSERT INTO tool_registry AS tr (
            name, description, category, source_agent, input_schema, output_schema,
            composable, avg_latency_ms, version, deprecated_at
        )
        SELECT name, description, category::tool_category, source_agent, input_schema,
               output_schema, true, avg_latency_ms, version, NULL
        FROM payload
        ON CONFLICT (name) DO UPDATE SET
            description = EXCLUDED.description,
            category = EXCLUDED.category,
            source_agent = EXCLUDED.source_agent,
            input_schema = EXCLUDED.input_schema,
            output_schema = EXCLUDED.output_schema,
            composable = true,
            avg_latency_ms = EXCLUDED.avg_latency_ms,
            version = EXCLUDED.version,
            deprecated_at = NULL
        WHERE (tr.description, tr.category, tr.source_agent, tr.input_schema, tr.output_schema,
               tr.composable, tr.avg_latency_ms, tr.version, tr.deprecated_at)
              IS DISTINCT FROM
              (EXCLUDED.description, EXCLUDED.category, EXCLUDED.source_agent,
               EXCLUDED.input_schema, EXCLUDED.output_schema, true, EXCLUDED.avg_latency_ms,
               EXCLUDED.version, NULL::timestamptz)
        RETURNING (xmax = 0) AS inserted
    )
    SELECT count(*) FILTER (WHERE inserted), count(*) FILTER (WHERE NOT inserted)
    INTO v_inserted, v_updated
    FROM up;

    UPDATE tool_registry
    SET deprecated_at = now(), composable = false
    WHERE deprecated_at IS NULL AND NOT (name::text = ANY (v_names));
    GET DIAGNOSTICS v_deprecated = ROW_COUNT;

    WITH m AS (
        SELECT c.tool_id AS consumer_tool_id, p.tool_id AS producer_tool_id,
               d.output_field, d.input_field
        FROM jsonb_to_recordset(p_dependencies)
             AS d(consumer text, producer text, output_field text, input_field text)
        JOIN tool_registry c ON c.name = d.consumer
        JOIN tool_registry p ON p.name = d.producer
    ), del AS (
        DELETE FROM tool_dependencies td
        WHERE NOT EXISTS (
            SELECT 1 FROM m
            WHERE m.consumer_tool_id = td.consumer_tool_id
              AND m.producer_tool_id = td.producer_tool_id
        )
        RETURNING 1
    ), ins AS (
        INSERT INTO tool_dependencies AS td (consumer_tool_id, producer_tool_id, output_field, input_field)
        SELECT consumer_tool_id, producer_tool_id, output_field, input_field FROM m
        ON CONFLICT (consumer_tool_id, producer_tool_id) DO UPDATE SET
            output_field = EXCLUDED.output_field,
            input_field = EXCLUDED.input_field
        WHERE (td.output_field, td.input_field)
              IS DISTINCT FROM (EXCLUDED.output_field, EXCLUDED.input_field)
        RETURNING 1
    )
    SELECT (SELECT count(*) FROM ins), (SELECT count(*) FROM del)
    INTO v_dep_upserted, v_dep_deleted;

    RETURN jsonb_build_object(
        'inserted', v_inserted,
        'updated', v_updated,
        'deprecated', v_deprecated,
        'dependencies_upserted', v_dep_upserted,
        'dependencies_deleted', v_dep_deleted
    );
END
$fn$;

COMMENT ON FUNCTION sync_tool_registry(jsonb, jsonb, integer) IS
    'Makes tool_registry / tool_dependencies equal to the running code''s registered tools (called at API startup). Validates before any write; refuses a payload that would deprecate more than p_max_deprecations active tools; idempotent; serialised by an advisory lock. Returns {inserted, updated, deprecated, dependencies_upserted, dependencies_deleted}.';

REVOKE ALL ON FUNCTION sync_tool_registry(jsonb, jsonb, integer) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION sync_tool_registry(jsonb, jsonb, integer) TO service_role;
