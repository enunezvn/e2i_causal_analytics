-- ============================================================================
-- E2I CAUSAL ANALYTICS — harden hybrid-search synthetic exclusion to NULL-safe
-- ============================================================================
-- Migration: 045_hybrid_search_coalesce_synthetic.sql  (ledger label MEM2)
-- Purpose: Shard 07 (Provenance Read-Path Enforcement) Gate 9 (RAG leg, R13)
--          hardening. Migration 044 added `AND (... OR em.is_synthetic = false)`
--          to the episodic branch of both hybrid search functions. After
--          migration 063 the column is BOOLEAN NOT NULL DEFAULT false, so a real
--          NULL row cannot exist going forward — but a legacy episodic_memories
--          row written BEFORE 063 ran (or any row that somehow carries NULL) would
--          be SILENTLY DROPPED from real-mode search by the bare `= false`
--          predicate (SQL tri-valued logic: `NULL = false` → NULL → excluded).
--          That is a real-data availability bug: a legitimate (real) memory would
--          vanish from the chatbot. Harden the predicate to COALESCE-default so a
--          NULL provenance reads as "real" (false) and is RETAINED.
--
--   Both functions are re-created VERBATIM from
--   044_hybrid_search_exclude_synthetic.sql; the ONLY change is the episodic (em)
--   provenance predicate in each function's WHERE clause:
--       BEFORE (044):  AND (COALESCE(filters->>'include_synthetic','false') = 'true'
--                           OR em.is_synthetic = false)
--       AFTER  (045):  AND (COALESCE(filters->>'include_synthetic','false') = 'true'
--                           OR COALESCE(em.is_synthetic, false) = false)
--   This keeps the default-exclude semantics for synthetic rows (is_synthetic=true
--   is still excluded) while NEVER dropping a real/NULL row.
--
-- Dependencies: 044_hybrid_search_exclude_synthetic.sql (the bodies copied here),
--               migration 063 (episodic_memories.is_synthetic column).
-- Idempotent: CREATE OR REPLACE (no signature change); safe to re-apply.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- PART 1 — dense RPC: NULL-safe default-exclude on the episodic branch (R13)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION hybrid_vector_search(
    query_embedding vector(1536),
    match_count int DEFAULT 20,
    filters jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id text,
    content text,
    similarity double precision,
    metadata jsonb,
    source_table text
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY

    -- Search episodic_memories (conversation history, user queries, agent actions,
    -- and the operational KPI corpus indexed by corpus_ingestion)
    SELECT
        em.memory_id::text as id,
        em.description as content,
        1 - (em.embedding <=> query_embedding) as similarity,
        jsonb_build_object(
            'event_type', em.event_type,
            'agent_name', em.agent_name,
            'occurred_at', em.occurred_at,
            'brand', em.brand,
            'region', em.region,
            'patient_id', em.patient_id,
            'hcp_id', em.hcp_id,
            'importance_score', em.importance_score
        ) as metadata,
        'episodic_memories'::text as source_table
    FROM episodic_memories em
    WHERE
        -- F3a: case-insensitive brand/region so a title-case chatbot filter
        -- ("Kisqali") matches the lowercase-stored corpus ("kisqali").
        (filters->>'brand' IS NULL OR lower(em.brand) = lower(filters->>'brand'))
        AND (filters->>'region' IS NULL OR lower(em.region) = lower(filters->>'region'))
        AND (filters->>'agent_name' IS NULL OR em.agent_name::text = filters->>'agent_name')
        AND (filters->>'date_from' IS NULL OR em.occurred_at >= (filters->>'date_from')::timestamp)
        AND (filters->>'date_to' IS NULL OR em.occurred_at <= (filters->>'date_to')::timestamp)
        -- Provenance (migration 044 / Shard 07 R13; hardened NULL-safe in 045):
        -- default-exclude synthetic; opt in via filters->>'include_synthetic'='true'.
        -- COALESCE keeps a real/NULL row visible (NULL reads as false = real).
        AND (COALESCE(filters->>'include_synthetic','false') = 'true' OR COALESCE(em.is_synthetic, false) = false)
        -- Only return results with reasonable similarity
        AND (1 - (em.embedding <=> query_embedding)) > 0.5

    UNION ALL

    -- Search procedural_memories (successful patterns, tool sequences)
    SELECT
        pm.procedure_id::text as id,
        pm.procedure_name || ': ' || COALESCE(pm.trigger_pattern, '') as content,
        1 - (pm.trigger_embedding <=> query_embedding) as similarity,
        jsonb_build_object(
            'procedure_type', pm.procedure_type,
            'success_rate', pm.success_rate,
            'usage_count', pm.usage_count,
            'applicable_brands', pm.applicable_brands,
            'applicable_regions', pm.applicable_regions,
            'detected_intent', pm.detected_intent
        ) as metadata,
        'procedural_memories'::text as source_table
    FROM procedural_memories pm
    WHERE
        pm.is_active = true
        AND pm.success_count > 0
        AND (1 - (pm.trigger_embedding <=> query_embedding)) > 0.5

    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION hybrid_vector_search IS
'Semantic search across episodic_memories (incl. the operational KPI corpus) and procedural_memories using pgvector cosine similarity. brand/region filters are case-insensitive (migration 043, audit F3a). Default-excludes synthetic episodic rows (NULL-safe via COALESCE, migration 045); opt in via filters->>''include_synthetic''=''true'' (migration 044, Shard 07 R13). Returns top matches with metadata.';

GRANT EXECUTE ON FUNCTION hybrid_vector_search TO authenticated;

-- ----------------------------------------------------------------------------
-- PART 2 — sparse RPC: NULL-safe default-exclude on the episodic branch (R13),
--          preserving the 043 corpus reachability + 022 max_staleness contract.
-- ----------------------------------------------------------------------------
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
    v_or_query tsquery;
    v_max_staleness float;
    v_apply_staleness boolean;
BEGIN
    tsquery_val := websearch_to_tsquery('english', search_query);

    -- F2: the operational corpus is TERSE structured prose ("TRx for Kisqali in
    -- the northeast on <date>: value ..."). A real user/rewrite query is verbose
    -- ("TRx trend for Kisqali in the Northeast this quarter") and websearch's
    -- default AND-semantics (trx & trend & kisqali & northeast & quarter) matches
    -- ZERO corpus rows because 'trend'/'quarter' are absent. BM25-like retrieval
    -- wants documents sharing ANY query term, ranked by overlap (ts_rank_cd). We
    -- OR-combine the AND tsquery (& -> |) for the episodic branch ONLY; the
    -- causal_paths/agent_activities/triggers branches keep their tested
    -- AND-semantics (out of F2 scope). A naive blanket replace would also turn a
    -- negation (a & !b) into (a | !b), inverting the exclusion into a match-most
    -- probe (codex MED). Guard it: if the parsed query carries a negation, keep
    -- the original AND-semantics for safety; otherwise OR-combine the positives.
    IF position('!' in tsquery_val::text) > 0 THEN
        v_or_query := tsquery_val;
    ELSE
        v_or_query := COALESCE(
            NULLIF(replace(tsquery_val::text, '&', '|'), '')::tsquery,
            tsquery_val
        );
    END IF;

    BEGIN
        v_max_staleness := NULLIF(filters->>'max_staleness', '')::float;
    EXCEPTION
        WHEN invalid_text_representation OR numeric_value_out_of_range THEN
            v_max_staleness := NULL;
    END;
    v_apply_staleness := (v_max_staleness IS NOT NULL AND v_max_staleness < 1.0);

    RETURN QUERY

    -- F2: Search episodic_memories (incl. the operational KPI corpus) via the
    -- already-populated, GIN-indexed search_text tsvector. content = description
    -- (NOT the tsvector) so it is byte-identical to the dense episodic content
    -- and RRF dedup_key collapses the two legs (F4). episodic_memories has no
    -- invalidated_at column → max_staleness is a no-op here (like causal_paths).
    SELECT
        em.memory_id::text as id,
        em.description as content,
        ts_rank_cd(em.search_text, v_or_query)::double precision as rank,
        jsonb_build_object(
            'event_type', em.event_type,
            'agent_name', em.agent_name,
            'occurred_at', em.occurred_at,
            'brand', em.brand,
            'region', em.region,
            'importance_score', em.importance_score
        ) as metadata,
        'episodic_memories'::text as source_table
    FROM episodic_memories em
    WHERE
        em.search_text @@ v_or_query
        AND (filters->>'brand' IS NULL OR lower(em.brand) = lower(filters->>'brand'))
        AND (filters->>'region' IS NULL OR lower(em.region) = lower(filters->>'region'))
        AND (filters->>'agent_name' IS NULL OR em.agent_name::text = filters->>'agent_name')
        -- Provenance (migration 044 / Shard 07 R13; hardened NULL-safe in 045):
        -- default-exclude synthetic; opt in via filters->>'include_synthetic'='true'.
        -- COALESCE keeps a real/NULL row visible (NULL reads as false = real).
        AND (COALESCE(filters->>'include_synthetic','false') = 'true' OR COALESCE(em.is_synthetic, false) = false)

    UNION ALL

    -- Search causal_paths for causal relationships and effects
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

    -- Search triggers with reason and context (HAS invalidated_at → staleness)
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
        AND (NOT v_apply_staleness OR t.invalidated_at IS NULL)

    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION hybrid_fulltext_search IS
'Full-text search across episodic_memories (incl. operational KPI corpus; migration 043 / audit F2), causal_paths, agent_activities, and triggers using PostgreSQL tsvector and ts_rank. brand/region filters are case-insensitive. Default-excludes synthetic episodic rows (NULL-safe via COALESCE, migration 045); opt in via filters->>''include_synthetic''=''true'' (migration 044, Shard 07 R13). Supports phrases, boolean operators, partial matching, and optional max_staleness filter (Decision 3 = KEEP BINARY): when filters->>''max_staleness''::float < 1.0, triggers rows with invalidated_at IS NOT NULL are excluded (episodic_memories carries no invalidated_at → no-op).';

GRANT EXECUTE ON FUNCTION hybrid_fulltext_search TO authenticated;

-- ============================================================================
-- ROLLBACK
-- ============================================================================
-- Re-apply 044_hybrid_search_exclude_synthetic.sql to restore the previous
-- (bare `em.is_synthetic = false`) episodic branches.
-- ============================================================================
