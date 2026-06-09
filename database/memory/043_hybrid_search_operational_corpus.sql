-- ============================================================================
-- E2I CAUSAL ANALYTICS — make the hybrid search legs serve the operational corpus
-- ============================================================================
-- Migration: 043_hybrid_search_operational_corpus.sql
-- Purpose: Close audit findings F3a / F2 / F4 (rag-hybrid-search-audit-20260608)
--          AFTER the operational corpus was populated into episodic_memories by
--          src/rag/corpus_ingestion.py (agent_name='corpus_ingestion').
--
--   F3a (dense brand-filter case): the live chatbot forwards a TITLE-CASE brand
--        filter ("Kisqali") but corpus rows store brand LOWERCASE ("kisqali")
--        (corpus_ingestion lowercases brand/region for episodic_memories.brand/
--        .region). hybrid_vector_search did an exact `em.brand = filters->>'brand'`
--        match, so every corpus row was silently excluded and only NULL-brand
--        procedural ([PROC]) junk survived. Faithful A/B/C probe (2026-06-09):
--        filter "Kisqali" -> only [PROC] junk; "kisqali"/no-filter -> real TRx
--        corpus rows. Fix = case-insensitive brand/region matching.
--
--   F2 (sparse leg blind to the corpus): hybrid_fulltext_search UNIONed only
--        causal_paths / agent_activities / triggers; episodic_memories (where
--        the corpus lives) was never queried, so the sparse leg returned 0 for
--        every commercial query. episodic_memories.search_text is already a
--        GENERATED tsvector over description (weight A) + event_subtype (B) with
--        a GIN index (idx_episodic_search) and is populated for corpus rows.
--        Fix = add an episodic_memories branch over em.search_text.
--
--   F4 (hybrid degenerates to single-leg dense): with the SAME corpus row now
--        reachable on BOTH dense and sparse, it appears in two ranked lists and
--        _reciprocal_rank_fusion (content-aware src/rag/fusion_utils.dedup_key,
--        keyed on content) reinforces it. The sparse branch returns
--        content = em.description (NOT the tsvector) — byte-identical to the
--        dense episodic content — so the two legs collapse to one fused key.
--
-- Dependencies: 011_hybrid_search_functions_fixed.sql (base defs),
--               022_hybrid_search_max_staleness.sql (fulltext max_staleness),
--               041_add_corpus_ingestion_agent.sql (corpus attribution).
-- Scope note: search_episodic_memory() (035) carries the same exact-case
--   brand/region match but is a DIFFERENT subsystem (EpisodicSearchFilters in
--   src/memory/episodic_memory.py), NOT the cognitive-RAG hybrid path these
--   findings concern; left unchanged here and surfaced in the PR body.
-- Idempotent: CREATE OR REPLACE (no signature change); safe to re-apply.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- PART 1 — dense RPC: case-insensitive brand/region on the episodic branch (F3a)
-- ----------------------------------------------------------------------------
-- Reproduces the live definition verbatim except the two filter predicates,
-- which become lower()-normalized. procedural_memories branch is unchanged.
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
'Semantic search across episodic_memories (incl. the operational KPI corpus) and procedural_memories using pgvector cosine similarity. brand/region filters are case-insensitive (migration 043, audit F3a). Returns top matches with metadata.';

GRANT EXECUTE ON FUNCTION hybrid_vector_search TO authenticated;

-- ----------------------------------------------------------------------------
-- PART 2 — sparse RPC: add an episodic_memories branch (F2) + case-insensitive
--          brand/region, preserving the existing 022 max_staleness contract.
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
    -- AND-semantics (out of F2 scope). Caveat: an explicit negation (-term ->
    -- !term) would also flip under & -> |; these analytics queries do not negate.
    v_or_query := COALESCE(
        NULLIF(replace(tsquery_val::text, '&', '|'), '')::tsquery,
        tsquery_val
    );

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
'Full-text search across episodic_memories (incl. operational KPI corpus; migration 043 / audit F2), causal_paths, agent_activities, and triggers using PostgreSQL tsvector and ts_rank. brand/region filters are case-insensitive. Supports phrases, boolean operators, partial matching, and optional max_staleness filter (Decision 3 = KEEP BINARY): when filters->>''max_staleness''::float < 1.0, triggers rows with invalidated_at IS NOT NULL are excluded (episodic_memories carries no invalidated_at → no-op).';

GRANT EXECUTE ON FUNCTION hybrid_fulltext_search TO authenticated;

-- ============================================================================
-- ROLLBACK
-- ============================================================================
-- Re-apply 011_hybrid_search_functions_fixed.sql then
-- 022_hybrid_search_max_staleness.sql to restore the previous (corpus-blind,
-- case-sensitive) behavior.
-- ============================================================================
