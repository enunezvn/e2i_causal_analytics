-- ============================================================================
-- E2I CAUSAL ANALYTICS — rag_vector_search provenance + filter-contract fix
-- ============================================================================
-- Migration: rag/004_rag_vector_search_provenance.sql
-- Purpose: Close issue #896 (e2e lineage audit 2026-06-12, Deep Dive A): the
--          orchestrator/dispatched-agent RAG path's rag_vector_search RPC
--          (defined in rag/001:151-241) kept pre-043 semantics after the
--          chatbot-path RPCs (hybrid_vector_search / hybrid_fulltext_search)
--          were hardened by memory/043+044+045. Three defects, one REPLACE:
--
--   (1) PROVENANCE — the episodic_memories branch had NO is_synthetic
--       predicate: any synthetic episodic row was retrievable by dispatched
--       Tier-2 agents with no opt-in (latent today only because
--       corpus_ingestion filters at source). Fix mirrors memory/044+045
--       EXACTLY: default-exclude with the NULL-safe COALESCE predicate, opt-in
--       via filters->>'include_synthetic' = 'true'. The opt-in stays a FILTERS
--       KEY (not a new SQL parameter) deliberately: CREATE OR REPLACE cannot
--       extend a signature — a 4-arg variant would be a SECOND overload living
--       beside the vulnerable 3-arg one, and PostgREST RPC dispatch on the
--       shared name would become ambiguous. The filters-key shape is the
--       platform pattern the hardened siblings already use.
--
--   (2) CASE — brand/region matching was case-SENSITIVE (em.brand =
--       filters->>'brand') while corpus rows store lowercased brand/region
--       (corpus_ingestion.py:277-278) and callers pass canonical case
--       ("Kisqali"). Same bug memory/043 fixed on the chatbot path (audit
--       F3a). Fix = lower()-normalized matching on BOTH filterable branches
--       (episodic_memories AND rag_document_chunks — dc has the same defect).
--
--   (3) FILTER-KEY CONTRACT — _build_filters_from_entities
--       (src/agents/orchestrator/nodes/rag_context.py) emits PLURAL list keys
--       (brands/regions); the RPC read only the SINGULAR filters->>'brand' →
--       NULL → entity-derived filters were a silent NO-OP. Fix = the
--       rag_filter_values normalizer below: the RPC honors BOTH shapes
--       (singular text and plural array, under either key), so existing
--       singular callers (free-form request.filters via /rag/search,
--       VectorBackend docstring usage) keep working unchanged.
--
--   (3b, codex iter-1 MED) rag_fulltext_search — the OTHER leg of the SAME
--       dispatched-agent HybridRetriever call — receives the SAME filters
--       dict in parallel and carried the same case-sensitive, singular-only
--       brand predicate on its rag_document_chunks branch (and no region
--       predicate at all). Fixing only the vector leg would leave the FUSED
--       results polluted by off-brand fulltext chunks. PART 3 gives that
--       branch the same normalized treatment. rag_fulltext_search has NO
--       episodic branch, and its causal_paths / agent_activities / triggers
--       branches stay provenance-unfiltered to mirror memory/044's explicit
--       scoping ("not part of the synthetic corpus blast radius" — the same
--       branches are unfiltered in the hardened hybrid_fulltext_search).
--
-- Branch-by-branch provenance scope (verified against migrations/063, 069 and
-- the live schema):
--   * episodic_memories      — HAS is_synthetic (063) → full treatment.
--   * rag_document_chunks    — NO is_synthetic column; structurally exempt.
--       Rows are written by real-only ingestion paths (corpus_ingestion skips
--       synthetic business_metrics at source); if this table ever receives a
--       synthetic writer it must first gain the column (follow-up migration),
--       at which point the 045-style predicate applies here too.
--   * procedural_memories    — NO is_synthetic column; structurally exempt
--       (consistent with memory/044's explicit scoping: "not part of the
--       synthetic corpus blast radius"). Also carries no scalar brand/region
--       columns (only applicable_brands/applicable_regions arrays), and the
--       hardened hybrid_vector_search precedent leaves its procedural branch
--       unfiltered — mirrored here: branch unchanged.
--
-- Tree/number rationale: rag_vector_search is DEFINED in database/rag/ and
-- scripts/run_migrations.sh applies database/memory/ BEFORE database/rag/ on a
-- fresh rebuild — a fix filed under memory/ would be clobbered by rag/001's
-- CREATE OR REPLACE. So the amendment lives here, as rag/004 (next number;
-- rag/002 is a validation_queries file the runner excludes).
--
-- Dependencies: rag/001 (base definition), migrations/063
--               (episodic_memories.is_synthetic), memory/043/044/045 (the
--               semantics mirrored here).
-- Idempotent: CREATE OR REPLACE, no signature change; safe to re-apply.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- PART 1 — filter-value normalizer (singular/plural, string/array, lowercased)
-- ----------------------------------------------------------------------------
-- Returns the lowercased set of filter values for a dimension, accepting every
-- shape callers actually send (#896 bug 3):
--   {"brand": "Kisqali"}              -> {kisqali}
--   {"brands": ["Kisqali","Fabhalta"]} -> {kisqali,fabhalta}
--   {"brand": ["Kisqali"]} / {"brands": "Kisqali"} (defensive) -> normalized
-- Returns NULL when the dimension is entirely unfiltered (absent keys, empty
-- arrays, blank strings) so callers can use the standard
-- `v IS NULL OR lower(col) = ANY(v)` no-filter idiom.
CREATE OR REPLACE FUNCTION rag_filter_values(
    filters jsonb,
    singular_key text,
    plural_key text
)
RETURNS text[]
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT NULLIF(ARRAY(
        SELECT DISTINCT lower(val)
        FROM (
            -- singular scalar value ({"brand": "Kisqali"})
            SELECT filters->>singular_key AS val
            WHERE jsonb_typeof(filters->singular_key) = 'string'
            UNION ALL
            -- plural key holding a scalar (defensive: {"brands": "Kisqali"})
            SELECT filters->>plural_key
            WHERE jsonb_typeof(filters->plural_key) = 'string'
            UNION ALL
            -- singular key holding an array (defensive: {"brand": ["Kisqali"]})
            SELECT s.elem FROM jsonb_array_elements_text(
                CASE WHEN jsonb_typeof(filters->singular_key) = 'array'
                     THEN filters->singular_key ELSE '[]'::jsonb END) AS s(elem)
            UNION ALL
            -- plural array ({"brands": ["Kisqali", "Fabhalta"]})
            SELECT p.elem FROM jsonb_array_elements_text(
                CASE WHEN jsonb_typeof(filters->plural_key) = 'array'
                     THEN filters->plural_key ELSE '[]'::jsonb END) AS p(elem)
        ) vals
        WHERE val IS NOT NULL AND btrim(val) <> ''
    ), '{}'::text[])
$$;

COMMENT ON FUNCTION rag_filter_values IS
'Normalizes a rag_* search filter dimension to a lowercased text[] accepting both the singular text key (brand) and the plural list key (brands) in either shape; NULL = dimension unfiltered. Added by rag/004 (#896 bug 3: plural entity-derived filter keys were a silent no-op against the singular-only predicate).';

GRANT EXECUTE ON FUNCTION rag_filter_values TO authenticated;

-- ----------------------------------------------------------------------------
-- PART 2 — rag_vector_search: provenance + case-insensitive, plural-aware
--          filtering. Body reproduced from rag/001 verbatim except the
--          documented predicates; procedural_memories branch unchanged.
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
    -- #896 bug 3: honor singular text AND plural array filter keys,
    -- lowercased once per call (see rag_filter_values).
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
        -- #896 bug 2: case-insensitive (rows store lowercase, callers pass
        -- canonical case), plural-aware brand/region matching.
        -- NOTE: no is_synthetic predicate here — rag_document_chunks carries
        -- no provenance column (structurally exempt; see header).
        AND (v_brands IS NULL OR lower(dc.brand) = ANY(v_brands))
        AND (v_regions IS NULL OR lower(dc.region) = ANY(v_regions))
        AND (filters->>'document_type' IS NULL OR dc.document_type = filters->>'document_type')
        AND (filters->>'agent_name' IS NULL OR dc.agent_name = filters->>'agent_name')
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
        -- #896 bug 2/3: case-insensitive, plural-aware brand/region (mirrors
        -- memory/043 F3a semantics on the chatbot-path RPCs).
        AND (v_brands IS NULL OR lower(em.brand) = ANY(v_brands))
        AND (v_regions IS NULL OR lower(em.region) = ANY(v_regions))
        AND (filters->>'agent_name' IS NULL OR em.agent_name::text = filters->>'agent_name')
        -- #896 bug 1 (provenance, mirrors memory/044 + NULL-safe 045):
        -- default-exclude synthetic episodic rows; opt in via
        -- filters->>'include_synthetic'='true'. COALESCE keeps a legacy
        -- real/NULL row visible (NULL reads as false = real).
        AND (COALESCE(filters->>'include_synthetic','false') = 'true' OR COALESCE(em.is_synthetic, false) = false)
        AND (1 - (em.embedding <=> query_embedding)) > 0.3

    UNION ALL

    -- Search procedural_memories (unchanged from rag/001: no is_synthetic
    -- column and no scalar brand/region columns — structurally exempt, same
    -- scoping as the hardened hybrid_vector_search precedent; see header)
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
        AND (1 - (pm.trigger_embedding <=> query_embedding)) > 0.3

    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION rag_vector_search IS
'Extended vector search for RAG system. Searches rag_document_chunks, episodic_memories, and procedural_memories. brand/region filters are case-insensitive and accept singular text (brand) or plural list (brands) keys (rag/004, #896). Default-excludes synthetic episodic rows (NULL-safe COALESCE); opt in via filters->>''include_synthetic''=''true'' (rag/004, mirroring memory/044+045).';

GRANT EXECUTE ON FUNCTION rag_vector_search TO authenticated;

-- ----------------------------------------------------------------------------
-- PART 3 — rag_fulltext_search: case-insensitive, plural-aware brand/region on
--          the rag_document_chunks branch (codex iter-1 MED; #896 bugs 2+3 on
--          the parallel fulltext leg). Body reproduced from rag/001 verbatim
--          except that branch's predicates; causal_paths / agent_activities /
--          triggers branches unchanged (no brand/region predicates existed,
--          and provenance stays out per the memory/044 blast-radius scoping).
--          The region predicate is NEW on this branch (rag/001 had none): the
--          search-filters contract filters brand+region, and the vector leg's
--          rag_document_chunks branch already enforced region — entity-derived
--          regions were silently unfiltered here.
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
    -- #896 bug 3: honor singular text AND plural array filter keys (see
    -- rag_filter_values).
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
        -- #896 bug 2/3: case-insensitive, plural-aware brand/region matching
        -- (no is_synthetic predicate — the table carries no provenance
        -- column; structurally exempt, see header).
        AND (v_brands IS NULL OR lower(dc.brand) = ANY(v_brands))
        AND (v_regions IS NULL OR lower(dc.region) = ANY(v_regions))
        AND (filters->>'document_type' IS NULL OR dc.document_type = filters->>'document_type')

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
'Extended fulltext search for RAG system. Searches rag_document_chunks, causal_paths, agent_activities, and triggers. brand/region filters on the rag_document_chunks branch are case-insensitive and accept singular text (brand) or plural list (brands) keys (rag/004, #896).';

GRANT EXECUTE ON FUNCTION rag_fulltext_search TO authenticated;

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)

-- ============================================================================
-- ROLLBACK
-- ============================================================================
-- Re-apply the rag_vector_search and rag_fulltext_search definitions from
-- rag/001_rag_schema.sql (restores the pre-043-semantics functions) and
--   DROP FUNCTION IF EXISTS rag_filter_values(jsonb, text, text);
-- ============================================================================
