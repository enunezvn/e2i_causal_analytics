-- ============================================================================
-- E2I CAUSAL ANALYTICS — close issue #973: procedural_memories provenance
-- ============================================================================
-- Migration: memory/047_procedural_synthetic_provenance.sql
-- Purpose: Companion to rag/005. rag/004 left procedural_memories EXEMPT from the
--          synthetic default-exclude predicate for one stated reason — "NO
--          is_synthetic column" (rag/004 header, lines 59-64). #973 (e2e lineage
--          audit, 2026-06-15) tracked that as a latent gap: procedural_memories
--          is read by three retrieval RPCs (rag_vector_search, hybrid_vector_search,
--          find_relevant_procedures), so a future synthetic procedure writer would
--          surface synthetic rows to dispatched agents / the chatbot with no opt-in.
--
--          This migration removes the structural exemption by ADDING the column
--          (PART 1) and extending the platform default-exclude predicate to the two
--          procedural-reading RPCs that live in the memory/ tree:
--            * hybrid_vector_search   (PART 2) — chatbot dense path; pm branch.
--            * find_relevant_procedures (PART 3) — procedural matcher.
--          rag_vector_search (the rag/ tree) gets the pm-branch predicate in the
--          companion rag/005; run_migrations.sh applies memory/ BEFORE rag/, so the
--          column added here already exists when rag/005's body executes.
--
-- Predicate shape — IDENTICAL to memory/044+045 / rag/004 where a filters arg
-- exists (filters-jsonb opt-in, NULL-safe COALESCE):
--     AND (COALESCE(filters->>'include_synthetic','false') = 'true'
--          OR COALESCE(pm.is_synthetic, false) = false)
--   find_relevant_procedures has a TYPED signature (no filters jsonb) and no
--   caller passes an opt-in, so it carries the bare safe default-exclude
--   `AND COALESCE(pm.is_synthetic, false) = false` — synthetic procedures must
--   never surface; adding an opt-in parameter would spawn a second overload and
--   change the call contract (deferred until a real caller needs it).
--
-- Behaviour today: NO-OP. is_synthetic defaults false (NOT NULL) -> every existing
-- procedure reads as real and passes the predicate unchanged. No synthetic writer
-- to procedural_memories exists today (procedural_memory.py / hpo_pattern_memory.py
-- insert real records only); the column closes the latent gap pre-emptively.
--
-- hybrid_fulltext_search is intentionally NOT touched: it unions episodic_memories,
-- causal_paths, agent_activities, triggers — NOT procedural_memories (verified
-- against memory/045) — so it has no #973 exposure.
--
-- Dependencies: memory/001 (procedural_memories + find_relevant_procedures base),
--               memory/045 (hybrid_vector_search body reproduced here verbatim
--               except the new pm predicate), migration 063-style provenance
--               precedent.
-- Idempotent: ADD COLUMN IF NOT EXISTS + CREATE OR REPLACE (no signature change).
-- ============================================================================

-- ----------------------------------------------------------------------------
-- PART 1 — provenance column on procedural_memories (NOT NULL DEFAULT false).
-- ----------------------------------------------------------------------------
ALTER TABLE procedural_memories
    ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT false;

COMMENT ON COLUMN procedural_memories.is_synthetic IS
'Provenance flag (issue #973): true = synthetic procedure. Default-excluded from rag_vector_search / hybrid_vector_search (opt in via filters->>''include_synthetic''=''true'') and from find_relevant_procedures (bare default-exclude, no opt-in). Default false = real (NOT NULL).';

-- ----------------------------------------------------------------------------
-- PART 2 — hybrid_vector_search: default-exclude on the procedural_memories (pm)
--          branch. Body reproduced from memory/045 verbatim except the new pm
--          predicate; episodic branch (em) unchanged (already hardened by 044/045).
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
        (filters->>'brand' IS NULL OR lower(em.brand) = lower(filters->>'brand'))
        AND (filters->>'region' IS NULL OR lower(em.region) = lower(filters->>'region'))
        AND (filters->>'agent_name' IS NULL OR em.agent_name::text = filters->>'agent_name')
        AND (filters->>'date_from' IS NULL OR em.occurred_at >= (filters->>'date_from')::timestamp)
        AND (filters->>'date_to' IS NULL OR em.occurred_at <= (filters->>'date_to')::timestamp)
        -- Provenance (migration 044 / Shard 07 R13; hardened NULL-safe in 045): unchanged.
        AND (COALESCE(filters->>'include_synthetic','false') = 'true' OR COALESCE(em.is_synthetic, false) = false)
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
        -- #973 provenance: default-exclude synthetic procedures (NULL-safe COALESCE),
        -- opt in via filters->>'include_synthetic'='true' (mirrors the episodic branch).
        AND (COALESCE(filters->>'include_synthetic','false') = 'true' OR COALESCE(pm.is_synthetic, false) = false)
        AND (1 - (pm.trigger_embedding <=> query_embedding)) > 0.5

    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION hybrid_vector_search IS
'Semantic search across episodic_memories (incl. the operational KPI corpus) and procedural_memories using pgvector cosine similarity. brand/region filters are case-insensitive (migration 043, audit F3a). Default-excludes synthetic rows on BOTH branches (NULL-safe COALESCE); opt in via filters->>''include_synthetic''=''true'' (episodic: migration 044/045; procedures: memory/047, #973). Returns top matches with metadata.';

GRANT EXECUTE ON FUNCTION hybrid_vector_search TO authenticated;

-- ----------------------------------------------------------------------------
-- PART 3 — find_relevant_procedures: bare default-exclude on procedural_memories.
--          Body reproduced from the live memory/001 definition verbatim except
--          the new predicate. TYPED signature (no filters jsonb) -> no opt-in
--          path; default-exclude is the safe, correct behaviour for the matcher.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION find_relevant_procedures(
    query_embedding vector(1536),
    match_threshold double precision DEFAULT 0.6,
    match_count integer DEFAULT 5,
    filter_type procedure_type DEFAULT NULL::procedure_type,
    filter_intent character varying DEFAULT NULL::character varying,
    filter_brand character varying DEFAULT NULL::character varying
)
RETURNS TABLE(
    procedure_id uuid,
    procedure_name character varying,
    procedure_type procedure_type,
    tool_sequence jsonb,
    trigger_pattern text,
    usage_count integer,
    success_count integer,
    success_rate double precision,
    similarity double precision
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        pm.procedure_id,
        pm.procedure_name,
        pm.procedure_type,
        pm.tool_sequence,
        pm.trigger_pattern,
        pm.usage_count,
        pm.success_count,
        pm.success_rate,
        1 - (pm.trigger_embedding <=> query_embedding) AS similarity
    FROM procedural_memories pm
    WHERE
        pm.is_active = TRUE
        AND 1 - (pm.trigger_embedding <=> query_embedding) > match_threshold
        AND (filter_type IS NULL OR pm.procedure_type = filter_type)
        AND (filter_intent IS NULL OR pm.detected_intent = filter_intent)
        AND (filter_brand IS NULL OR filter_brand = ANY(pm.applicable_brands) OR 'all' = ANY(pm.applicable_brands))
        -- #973 provenance: bare default-exclude (no filters arg -> no opt-in path;
        -- synthetic procedures must never surface to the procedural matcher).
        AND COALESCE(pm.is_synthetic, false) = false
    ORDER BY
        similarity * (0.5 + 0.5 * pm.success_rate) DESC
    LIMIT match_count;
END;
$$;

GRANT EXECUTE ON FUNCTION find_relevant_procedures TO authenticated;

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)

-- ============================================================================
-- ROLLBACK
-- ============================================================================
-- Re-apply memory/045_hybrid_search_coalesce_synthetic.sql (restores the
-- pm-exempt hybrid_vector_search) and memory/001's find_relevant_procedures,
-- then:
--   ALTER TABLE procedural_memories DROP COLUMN IF EXISTS is_synthetic;
-- ============================================================================
