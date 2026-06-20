-- ============================================================================
-- E2I CAUSAL ANALYTICS — close issue #973: rag_document_chunks provenance
-- ============================================================================
-- Migration: rag/005_chunks_synthetic_provenance.sql
-- Purpose: rag/004 added the synthetic default-exclude predicate to the
--          episodic_memories branch of rag_vector_search / rag_fulltext_search
--          but left rag_document_chunks and procedural_memories EXEMPT — for one
--          stated reason: "NO is_synthetic column" (rag/004 header, lines 54-64).
--          That exemption is purely structural. #973 (the 2026-06-15 e2e lineage
--          audit, Findings #896 follow-up) tracked it as a latent gap: if either
--          table ever gains a synthetic writer, synthetic rows would reach
--          dispatched agents through the RAG path with no opt-in.
--
--          This migration removes the structural exemption for rag_document_chunks
--          by ADDING the is_synthetic column (PART 1) and extending the SAME
--          platform default-exclude predicate to every branch of the rag_* RPCs
--          that reads it (PARTS 2-3). procedural_memories — the OTHER exempt
--          source, searched by rag_vector_search here AND by hybrid_vector_search
--          / find_relevant_procedures in the memory/ tree — is handled in the
--          companion migration memory/047_procedural_synthetic_provenance.sql
--          (its column lives in the memory tree; run_migrations.sh applies
--          memory/ BEFORE rag/, so pm.is_synthetic already exists when the
--          rag_vector_search body below executes its pm-branch predicate).
--
-- Predicate shape — IDENTICAL to rag/004 + memory/044+045 (the platform pattern):
--     AND (COALESCE(filters->>'include_synthetic','false') = 'true'
--          OR COALESCE(<alias>.is_synthetic, false) = false)
--   * filters-jsonb opt-in, NOT a new SQL parameter: CREATE OR REPLACE cannot
--     extend a signature without spawning a second overload that breaks PostgREST
--     RPC resolution on the shared name (rag/004 header, lines 15-21).
--   * NULL-safe COALESCE on the column side: a legacy/NULL row reads as real
--     (false) and is RETAINED — no real-data availability regression.
--
-- Behaviour today: NO-OP. is_synthetic defaults to false (NOT NULL), so every
-- existing chunk reads as real and the default-exclude predicate passes it
-- through unchanged. The guard only bites once a synthetic writer sets true.
-- (Audit holds: grep finds no INSERT into rag_document_chunks with is_synthetic
-- = true today; the column closes the latent gap pre-emptively.)
--
-- Branch coverage in this file:
--   * rag_vector_search   — rag_document_chunks branch (dc) + procedural_memories
--                           branch (pm) gain the predicate; episodic branch (em)
--                           unchanged (already hardened by rag/004).
--   * rag_fulltext_search — rag_document_chunks branch (dc) gains the predicate;
--                           causal_paths / agent_activities / triggers branches
--                           unchanged (different tables, no provenance column,
--                           out of #973 scope — mirrors rag/004 + memory/044
--                           blast-radius scoping).
--   Auto-covered by delegation (no edit needed): find_similar_documents and
--   rag_hybrid_search both call rag_vector_search / rag_fulltext_search and
--   inherit the fix. get_search_stats is a counts-only surface (returns no row
--   content to agents) and is intentionally left unchanged.
--
-- Dependencies: rag/001 (base defs), rag/004 (episodic provenance + the bodies
--               reproduced here verbatim except the new dc/pm predicates),
--               memory/047 (procedural_memories.is_synthetic column).
-- Idempotent: ADD COLUMN IF NOT EXISTS + CREATE OR REPLACE (no signature change).
-- ============================================================================

-- ----------------------------------------------------------------------------
-- PART 1 — provenance column on rag_document_chunks (NOT NULL DEFAULT false so
--          existing real rows are unchanged; NULL can never arise going forward).
-- ----------------------------------------------------------------------------
ALTER TABLE rag_document_chunks
    ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;

COMMENT ON COLUMN rag_document_chunks.is_synthetic IS
'Provenance flag (issue #973): true = synthetic-corpus row. Default-excluded from rag_vector_search / rag_fulltext_search unless filters->>''include_synthetic''=''true''. Default false = real (NOT NULL).';

-- ----------------------------------------------------------------------------
-- PART 2 — rag_vector_search: extend the default-exclude predicate to the
--          rag_document_chunks (dc) AND procedural_memories (pm) branches. Body
--          reproduced from rag/004 verbatim except the two new predicates.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION rag_vector_search(
    query_embedding vector(1536),
    match_count int DEFAULT 20,
    filters jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id text,
    content text,
    similarity float,
    metadata jsonb,
    source_table text
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_brands  text[] := rag_filter_values(filters, 'brand', 'brands');
    v_regions text[] := rag_filter_values(filters, 'region', 'regions');
BEGIN
    RETURN QUERY

    -- Search rag_document_chunks
    SELECT
        dc.chunk_id::text as id,
        dc.content as content,
        1 - (dc.embedding <=> query_embedding) as similarity,
        jsonb_build_object(
            'document_id', dc.document_id,
            'document_type', dc.document_type,
            'chunk_index', dc.chunk_index,
            'brand', dc.brand,
            'region', dc.region,
            'agent_name', dc.agent_name,
            'kpi_name', dc.kpi_name,
            'created_at', dc.created_at
        ) || dc.metadata as metadata,
        'rag_document_chunks'::text as source_table
    FROM rag_document_chunks dc
    WHERE
        dc.embedding IS NOT NULL
        AND (v_brands IS NULL OR lower(dc.brand) = ANY(v_brands))
        AND (v_regions IS NULL OR lower(dc.region) = ANY(v_regions))
        AND (filters->>'document_type' IS NULL OR dc.document_type = filters->>'document_type')
        AND (filters->>'agent_name' IS NULL OR dc.agent_name = filters->>'agent_name')
        -- #973 provenance: default-exclude synthetic chunks (NULL-safe COALESCE),
        -- opt in via filters->>'include_synthetic'='true' (mirrors rag/004 episodic).
        AND (COALESCE(filters->>'include_synthetic','false') = 'true' OR COALESCE(dc.is_synthetic, false) = false)
        AND (1 - (dc.embedding <=> query_embedding)) > 0.3

    UNION ALL

    -- Search episodic_memories
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
            'importance_score', em.importance_score
        ) as metadata,
        'episodic_memories'::text as source_table
    FROM episodic_memories em
    WHERE
        em.embedding IS NOT NULL
        AND (v_brands IS NULL OR lower(em.brand) = ANY(v_brands))
        AND (v_regions IS NULL OR lower(em.region) = ANY(v_regions))
        AND (filters->>'agent_name' IS NULL OR em.agent_name::text = filters->>'agent_name')
        -- Provenance (rag/004, mirroring memory/044 + NULL-safe 045): unchanged.
        AND (COALESCE(filters->>'include_synthetic','false') = 'true' OR COALESCE(em.is_synthetic, false) = false)
        AND (1 - (em.embedding <=> query_embedding)) > 0.3

    UNION ALL

    -- Search procedural_memories (pm.is_synthetic added by memory/047, which
    -- run_migrations.sh applies before this rag/ migration)
    SELECT
        pm.procedure_id::text as id,
        pm.procedure_name || ': ' || COALESCE(pm.trigger_pattern, '') as content,
        1 - (pm.trigger_embedding <=> query_embedding) as similarity,
        jsonb_build_object(
            'procedure_type', pm.procedure_type,
            'success_rate', pm.success_rate,
            'usage_count', pm.usage_count,
            'applicable_brands', pm.applicable_brands,
            'applicable_regions', pm.applicable_regions
        ) as metadata,
        'procedural_memories'::text as source_table
    FROM procedural_memories pm
    WHERE
        pm.trigger_embedding IS NOT NULL
        AND pm.is_active = true
        -- #973 provenance: default-exclude synthetic procedures (NULL-safe),
        -- opt in via filters->>'include_synthetic'='true'.
        AND (COALESCE(filters->>'include_synthetic','false') = 'true' OR COALESCE(pm.is_synthetic, false) = false)
        AND (1 - (pm.trigger_embedding <=> query_embedding)) > 0.3

    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION rag_vector_search IS
'Extended vector search for RAG system. Searches rag_document_chunks, episodic_memories, and procedural_memories. brand/region filters are case-insensitive and accept singular text (brand) or plural list (brands) keys (rag/004, #896). Default-excludes synthetic rows on ALL three branches (NULL-safe COALESCE); opt in via filters->>''include_synthetic''=''true'' (episodic: rag/004; document chunks + procedures: rag/005 + memory/047, #973).';

GRANT EXECUTE ON FUNCTION rag_vector_search TO authenticated;

-- ----------------------------------------------------------------------------
-- PART 3 — rag_fulltext_search: extend the default-exclude predicate to the
--          rag_document_chunks (dc) branch. Body reproduced from rag/004
--          verbatim except the new dc predicate; causal_paths / agent_activities
--          / triggers branches unchanged (no provenance column, out of scope).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION rag_fulltext_search(
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
    v_brands  text[] := rag_filter_values(filters, 'brand', 'brands');
    v_regions text[] := rag_filter_values(filters, 'region', 'regions');
BEGIN
    tsquery_val := websearch_to_tsquery('english', search_query);

    RETURN QUERY

    -- Search rag_document_chunks
    SELECT
        dc.chunk_id::text as id,
        dc.content as content,
        ts_rank_cd(dc.search_vector, tsquery_val)::double precision as rank,
        jsonb_build_object(
            'document_id', dc.document_id,
            'document_type', dc.document_type,
            'brand', dc.brand,
            'region', dc.region
        ) || dc.metadata as metadata,
        'rag_document_chunks'::text as source_table
    FROM rag_document_chunks dc
    WHERE
        dc.search_vector @@ tsquery_val
        AND (v_brands IS NULL OR lower(dc.brand) = ANY(v_brands))
        AND (v_regions IS NULL OR lower(dc.region) = ANY(v_regions))
        AND (filters->>'document_type' IS NULL OR dc.document_type = filters->>'document_type')
        -- #973 provenance: default-exclude synthetic chunks (NULL-safe COALESCE),
        -- opt in via filters->>'include_synthetic'='true'.
        AND (COALESCE(filters->>'include_synthetic','false') = 'true' OR COALESCE(dc.is_synthetic, false) = false)

    UNION ALL

    -- Search causal_paths
    SELECT
        cp.path_id::text as id,
        COALESCE(cp.start_node, '') || ' → ' || COALESCE(cp.end_node, '') || ': ' ||
        COALESCE(cp.method_used, '') as content,
        ts_rank_cd(cp.search_vector, tsquery_val)::double precision as rank,
        jsonb_build_object(
            'start_node', cp.start_node,
            'end_node', cp.end_node,
            'causal_effect_size', cp.causal_effect_size,
            'confidence_level', cp.confidence_level
        ) as metadata,
        'causal_paths'::text as source_table
    FROM causal_paths cp
    WHERE
        cp.search_vector @@ tsquery_val

    UNION ALL

    -- Search agent_activities
    SELECT
        aa.activity_id::text as id,
        aa.agent_name || ' (' || aa.activity_type || ')' as content,
        ts_rank_cd(aa.search_vector, tsquery_val)::double precision as rank,
        jsonb_build_object(
            'agent_name', aa.agent_name,
            'agent_tier', aa.agent_tier,
            'activity_type', aa.activity_type,
            'status', aa.status
        ) as metadata,
        'agent_activities'::text as source_table
    FROM agent_activities aa
    WHERE
        aa.search_vector @@ tsquery_val
        AND (filters->>'agent_name' IS NULL OR aa.agent_name = filters->>'agent_name')

    UNION ALL

    -- Search triggers
    SELECT
        t.trigger_id::text as id,
        t.trigger_reason as content,
        ts_rank_cd(t.search_vector, tsquery_val)::double precision as rank,
        jsonb_build_object(
            'trigger_type', t.trigger_type,
            'priority', t.priority,
            'confidence_score', t.confidence_score
        ) as metadata,
        'triggers'::text as source_table
    FROM triggers t
    WHERE
        t.search_vector @@ tsquery_val

    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION rag_fulltext_search IS
'Extended fulltext search for RAG system. Searches rag_document_chunks, causal_paths, agent_activities, and triggers. brand/region filters on the rag_document_chunks branch are case-insensitive and accept singular text (brand) or plural list (brands) keys (rag/004, #896). Default-excludes synthetic rag_document_chunks rows (NULL-safe COALESCE); opt in via filters->>''include_synthetic''=''true'' (rag/005, #973).';

GRANT EXECUTE ON FUNCTION rag_fulltext_search TO authenticated;

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)

-- ============================================================================
-- ROLLBACK
-- ============================================================================
-- Re-apply rag/004_rag_vector_search_provenance.sql (restores the dc/pm-exempt
-- function bodies) and:
--   ALTER TABLE rag_document_chunks DROP COLUMN IF EXISTS is_synthetic;
-- (Companion: roll back memory/047 to drop procedural_memories.is_synthetic.)
-- ============================================================================
