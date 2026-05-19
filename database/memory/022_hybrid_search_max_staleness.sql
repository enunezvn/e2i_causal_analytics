-- ============================================================================
-- E2I CAUSAL ANALYTICS - max_staleness SUPPORT IN HYBRID SEARCH RPC
-- ============================================================================
-- Migration: 022_hybrid_search_max_staleness.sql
-- Purpose: Extend hybrid_fulltext_search to honor a max_staleness filter on
--          tables that carry invalidated_at, and surface invalidated_at in
--          row metadata so downstream Python callers can render or post-filter.
-- Dependencies: 011_hybrid_search_functions_fixed.sql, 021_insight_lifecycle.sql
-- Phase 2 finishing (issue #373); plan
--   .claude/plans/e2i_memory_subsystems_implementation_plan.md
--   §"DECISIONS ADOPTED — 2026-05-19" (Decision 3 = KEEP BINARY).
-- ----------------------------------------------------------------------------
-- Semantics of the new filter key ``filters->>'max_staleness'`` (text decoded
-- to float):
--   NULL                 → no filter (backward-compatible default)
--   ::float >= 1.0       → include all rows (no-op, but explicit)
--   ::float <  1.0       → exclude rows with invalidated_at IS NOT NULL on
--                          tables that carry the column
-- Tables touched (currently only `triggers` of the three sources):
--   - causal_paths        → NO invalidated_at column (no-op filter)
--   - agent_activities    → NO invalidated_at column (no-op filter)
--   - triggers            → HAS invalidated_at (migration 021); filter applies
-- Vector RPC (`hybrid_vector_search`) is left structurally unchanged because
-- neither `episodic_memories` nor `procedural_memories` carries an
-- invalidated_at column today (021_insight_lifecycle.sql:316 documents that
-- explicitly). Python post-filter in `src/rag/memory_connector.py` is the
-- belt-and-suspenders for any future row whose metadata leaks invalidated_at
-- via JSONB.
-- ============================================================================

CREATE OR REPLACE FUNCTION hybrid_fulltext_search(
    search_query text,
    match_count int DEFAULT 20,
    filters jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id text,
    content text,
    rank double precision,
    metadata jsonb,
    source_table text
)
LANGUAGE plpgsql
AS $$
DECLARE
    tsquery_val tsquery;
    v_max_staleness float;
    v_apply_staleness boolean;
BEGIN
    -- Parse search query with prefix matching for partial words
    tsquery_val := websearch_to_tsquery('english', search_query);

    -- Decode max_staleness once. Under Decision 3 = KEEP BINARY, the only
    -- behavioral threshold is "< 1.0 means exclude invalidated rows".
    -- Catch malformed input (non-numeric strings) and degrade to no-op rather
    -- than crashing the RPC, per codex iter-0 MED on input safety.
    BEGIN
        v_max_staleness := NULLIF(filters->>'max_staleness', '')::float;
    EXCEPTION
        WHEN invalid_text_representation OR numeric_value_out_of_range THEN
            -- Codex iter-1 MED: also catch oversized exponents (e.g., '1e500')
            -- in addition to non-numeric strings. We do NOT use `WHEN OTHERS`
            -- so the no-op contract stays narrow and intentional.
            v_max_staleness := NULL;
    END;
    -- Note: NaN < 1.0 evaluates to False in PostgreSQL, so NaN does not
    -- activate the filter; matches the Python helper's NaN semantics in
    -- src/rag/memory_connector.py::_is_invalidated_under_max_staleness.
    v_apply_staleness := (v_max_staleness IS NOT NULL AND v_max_staleness < 1.0);

    RETURN QUERY

    -- Search causal_paths for causal relationships and effects
    -- (no invalidated_at column → no staleness filter applies)
    SELECT
        cp.path_id::text as id,
        COALESCE(cp.start_node, '') || ' → ' || COALESCE(cp.end_node, '') || ': ' ||
        COALESCE(cp.method_used, '') as content,
        ts_rank_cd(cp.search_vector, tsquery_val)::double precision as rank,
        jsonb_build_object(
            'start_node', cp.start_node,
            'end_node', cp.end_node,
            'causal_effect_size', cp.causal_effect_size,
            'confidence_level', cp.confidence_level,
            'method_used', cp.method_used,
            'created_at', cp.created_at
        ) as metadata,
        'causal_paths'::text as source_table
    FROM causal_paths cp
    WHERE
        cp.search_vector @@ tsquery_val

    UNION ALL

    -- Search agent_activities for agent analyses and outputs
    -- (no invalidated_at column → no staleness filter applies)
    SELECT
        aa.activity_id::text as id,
        aa.agent_name || ' (' || aa.activity_type || ')' as content,
        ts_rank_cd(aa.search_vector, tsquery_val)::double precision as rank,
        jsonb_build_object(
            'agent_name', aa.agent_name,
            'agent_tier', aa.agent_tier,
            'activity_type', aa.activity_type,
            'status', aa.status,
            'created_at', aa.created_at,
            'workstream', aa.workstream
        ) as metadata,
        'agent_activities'::text as source_table
    FROM agent_activities aa
    WHERE
        aa.search_vector @@ tsquery_val
        AND (filters->>'agent_name' IS NULL OR aa.agent_name = filters->>'agent_name')
        AND (filters->>'status' IS NULL OR aa.status = filters->>'status')

    UNION ALL

    -- Search triggers with reason and context
    -- triggers HAS invalidated_at (migration 021) → max_staleness filter applies
    -- and invalidated_at is exposed in metadata so Python belt-and-suspenders
    -- post-filter has the data to inspect.
    SELECT
        t.trigger_id::text as id,
        t.trigger_reason as content,
        ts_rank_cd(t.search_vector, tsquery_val)::double precision as rank,
        jsonb_build_object(
            'trigger_type', t.trigger_type,
            'priority', t.priority,
            'confidence_score', t.confidence_score,
            'created_at', t.created_at,
            'recommended_action', t.recommended_action,
            'invalidated_at', t.invalidated_at
        ) as metadata,
        'triggers'::text as source_table
    FROM triggers t
    WHERE
        t.search_vector @@ tsquery_val
        AND (filters->>'priority' IS NULL OR t.priority::text = filters->>'priority')
        -- max_staleness filter: when applied (v_max_staleness < 1.0), exclude
        -- rows with invalidated_at set.
        AND (NOT v_apply_staleness OR t.invalidated_at IS NULL)

    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION hybrid_fulltext_search IS
'Full-text search across causal_paths, agent_activities, and triggers using PostgreSQL tsvector and ts_rank. Supports phrases, boolean operators, partial matching, and optional max_staleness filter (Decision 3 = KEEP BINARY, plan §DECISIONS ADOPTED 2026-05-19): when filters->>''max_staleness''::float < 1.0, triggers rows with invalidated_at IS NOT NULL are excluded.';

GRANT EXECUTE ON FUNCTION hybrid_fulltext_search TO authenticated;

-- ============================================================================
-- ROLLBACK
-- ============================================================================
-- Restore the previous (no-max_staleness) version by re-applying
-- 011_hybrid_search_functions_fixed.sql.
