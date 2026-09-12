-- ============================================================================
-- E2I Causal Analytics - ROLLBACK for ml/040_tool_registry_startup_sync.sql
-- NOT a forward migration: scripts/run_migrations.sh skips rollback_*.sql. Apply by hand,
-- AFTER the code revert and AFTER rollback_041.sql:
--
--   docker exec -i supabase-db psql -U postgres -d postgres -v ON_ERROR_STOP=1 \
--       --single-transaction < database/ml/rollback_040.sql
--
-- Drops sync_tool_registry(), restores the ml/013 six-agent valid_agent CHECK,
-- tool_registry.success_rate (with the seed values of ml/013 / ml/027) and the two functions
-- ml/040 dropped (owner decision O3), verbatim from ml/013.
--
-- Refuses (and, in one transaction, changes nothing) while any tool_registry row names an
-- agent outside the six: the runtime sync writes cohort_constructor / cohort_profiler tools.
-- Deprecate or delete those rows (composition_steps may reference them), then re-run.
--
-- Also deletes the ml/040 ledger row, so re-deploying the learning-loop code re-applies it.
-- ml/039 has no rollback: a value cannot be removed from an enum, and COHORT is harmless.
-- Always apply with --single-transaction and ON_ERROR_STOP (a partial rollback otherwise).
-- Idempotent: a second run on an already rolled-back database changes nothing.
-- ============================================================================

DO $guard$
DECLARE
    v_bad text;
BEGIN
    SELECT string_agg(name || ' (' || source_agent || ')', ', ' ORDER BY name) INTO v_bad
    FROM tool_registry
    WHERE source_agent NOT IN ('causal_impact', 'heterogeneous_optimizer', 'gap_analyzer',
                               'experiment_designer', 'prediction_synthesizer', 'drift_monitor');
    IF v_bad IS NOT NULL THEN
        RAISE EXCEPTION 'rollback_040: tool_registry rows violate the ml/013 six-agent valid_agent CHECK: %. Deprecate or delete them, then re-run.', v_bad;
    END IF;
END
$guard$;

-- The ledger row, so a later deploy of the learning-loop code re-applies ml/040. ml/039 stays
-- recorded: it has no rollback.
DELETE FROM public.schema_migrations WHERE filename = 'ml/040_tool_registry_startup_sync.sql';

DROP FUNCTION IF EXISTS sync_tool_registry(jsonb, jsonb, integer);

ALTER TABLE tool_registry DROP CONSTRAINT IF EXISTS valid_agent;
-- Written in the form prod holds (pg_get_constraintdef of the restored ml/013 CHECK), so the
-- restored constraint is identical to the pre-040 one, not only equivalent.
ALTER TABLE tool_registry ADD CONSTRAINT valid_agent CHECK (((source_agent)::text = ANY (ARRAY[
    ('causal_impact'::character varying)::text, ('heterogeneous_optimizer'::character varying)::text,
    ('gap_analyzer'::character varying)::text, ('experiment_designer'::character varying)::text,
    ('prediction_synthesizer'::character varying)::text, ('drift_monitor'::character varying)::text])));

COMMENT ON COLUMN tool_registry.avg_latency_ms IS NULL;

DO $restore$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_attribute WHERE attrelid = 'tool_registry'::regclass
                       AND attname = 'success_rate' AND NOT attisdropped) THEN
        ALTER TABLE tool_registry ADD COLUMN success_rate FLOAT DEFAULT 0.95;
        ALTER TABLE tool_registry ADD CONSTRAINT tool_registry_success_rate_check
            CHECK (success_rate >= 0 AND success_rate <= 1);
        -- Seed values of ml/027 (every other seeded row carries the 0.95 default). The
        -- updated_at trigger is held off so restoring a dropped column does not stamp the rows.
        ALTER TABLE tool_registry DISABLE TRIGGER trg_tool_registry_updated;
        UPDATE tool_registry SET success_rate = 0.92 WHERE name = 'discover_dag';
        UPDATE tool_registry SET success_rate = 0.98 WHERE name = 'detect_structural_drift';
        ALTER TABLE tool_registry ENABLE TRIGGER trg_tool_registry_updated;
    END IF;
END
$restore$;

CREATE OR REPLACE FUNCTION update_tool_registry_metrics()
RETURNS void AS $$
BEGIN
    UPDATE tool_registry tr
    SET 
        avg_latency_ms = subq.avg_latency,
        success_rate = subq.success_rate,
        updated_at = NOW()
    FROM (
        SELECT 
            tool_id,
            AVG(latency_ms) AS avg_latency,
            COUNT(*) FILTER (WHERE success = true)::FLOAT / 
                NULLIF(COUNT(*), 0) AS success_rate
        FROM tool_performance
        WHERE executed_at > NOW() - INTERVAL '7 days'
        GROUP BY tool_id
    ) subq
    WHERE tr.tool_id = subq.tool_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_tool_registry_metrics IS 'Updates tool registry with aggregated performance metrics from last 7 days';

CREATE OR REPLACE FUNCTION public.get_tool_execution_order(p_tool_names text[])
 RETURNS TABLE(execution_order integer, tool_name character varying, depends_on text[])
 LANGUAGE sql
AS $function$
WITH RECURSIVE tool_graph AS (
    -- Base case: tools with no dependencies
    SELECT 
        tr.name::VARCHAR(100) AS tool_name,
        0 AS depth,
        tr.name::VARCHAR(100) AS path_tool
    FROM tool_registry tr
    WHERE tr.name = ANY(p_tool_names)
      AND NOT EXISTS (
          SELECT 1 FROM tool_dependencies td
          JOIN tool_registry producer ON td.producer_tool_id = producer.tool_id
          WHERE td.consumer_tool_id = tr.tool_id
            AND producer.name = ANY(p_tool_names)
      )
    
    UNION ALL
    
    -- Recursive case: tools depending on already-processed tools
    SELECT 
        tr.name::VARCHAR(100),
        tg.depth + 1,
        tr.name::VARCHAR(100)
    FROM tool_registry tr
    JOIN tool_dependencies dep ON tr.tool_id = dep.consumer_tool_id
    JOIN tool_registry producer ON dep.producer_tool_id = producer.tool_id
    JOIN tool_graph tg ON producer.name::VARCHAR(100) = tg.tool_name
    WHERE tr.name = ANY(p_tool_names)
      AND producer.name = ANY(p_tool_names)
      AND tr.name::VARCHAR(100) != tg.path_tool  -- Prevent cycles
),
tool_depths AS (
    SELECT 
        tool_name,
        MAX(depth) AS max_depth
    FROM tool_graph
    GROUP BY tool_name
),
tool_deps_agg AS (
    SELECT 
        td.tool_name,
        ARRAY_AGG(DISTINCT producer.name ORDER BY producer.name)::TEXT[] AS depends_on
    FROM tool_depths td
    LEFT JOIN tool_registry tr ON tr.name = td.tool_name
    LEFT JOIN tool_dependencies dep ON tr.tool_id = dep.consumer_tool_id
    LEFT JOIN tool_registry producer ON dep.producer_tool_id = producer.tool_id
        AND producer.name = ANY(p_tool_names)
    GROUP BY td.tool_name
)
SELECT 
    ROW_NUMBER() OVER (ORDER BY td.max_depth, td.tool_name)::INTEGER AS execution_order,
    td.tool_name,
    COALESCE(tda.depends_on, ARRAY[]::TEXT[]) AS depends_on
FROM tool_depths td
LEFT JOIN tool_deps_agg tda ON td.tool_name = tda.tool_name
ORDER BY td.max_depth, td.tool_name;
$function$;

COMMENT ON FUNCTION get_tool_execution_order IS 'Returns topologically sorted execution order for given tools';
