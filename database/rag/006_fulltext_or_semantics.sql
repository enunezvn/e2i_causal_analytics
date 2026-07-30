-- ============================================================================
-- E2I CAUSAL ANALYTICS — issue #1376: rag_fulltext_search OR-semantics
-- ============================================================================
-- Migration: rag/006_fulltext_or_semantics.sql
-- Purpose: rag_fulltext_search matched with websearch_to_tsquery's AND
--          semantics: every plain word in the query had to be present in a
--          row's search_vector. A natural-language chat question ("What
--          caused the TRx decline for Kisqali in the West region?" ->
--          caus & trx & declin & kisqali & west & region) therefore matched
--          nothing (live f:0), while keyword probes ('Kisqali west trx')
--          matched fine. A BM25-style keyword leg should match ANY query
--          term and let ranking reward multi-term coverage.
--
--          This migration keeps websearch_to_tsquery for parsing (phrase
--          quoting and negation still work) but derives an OR-form of the
--          parsed tsquery — the top-level ' & ' conjunctions become ' | ' —
--          and both MATCHES and RANKS with the OR form. ts_rank_cd scores
--          rows matching more distinct lexemes higher, so rows satisfying
--          the full AND still rank first; the change is strictly
--          recall-additive with rank-preserving precision. ORDER BY rank
--          DESC + LIMIT match_count are unchanged.
--
--          The OR-form derivation is wrapped in an exception handler: if the
--          textual rewrite ever produces an uncastable tsquery (e.g. an
--          operator sequence from exotic websearch input), the function
--          falls back to the original AND-form rather than erroring.
--
-- Function body is otherwise IDENTICAL to the deployed revision from
-- rag/005 (chunks provenance predicate included).
-- ============================================================================

CREATE OR REPLACE FUNCTION public.rag_fulltext_search(search_query text, match_count integer DEFAULT 20, filters jsonb DEFAULT '{}'::jsonb)
 RETURNS TABLE(id text, content text, rank double precision, metadata jsonb, source_table text)
 LANGUAGE plpgsql
AS $function$
DECLARE
    tsquery_val tsquery;
    tsquery_any tsquery;
    v_brands  text[] := rag_filter_values(filters, 'brand', 'brands');
    v_regions text[] := rag_filter_values(filters, 'region', 'regions');
BEGIN
    tsquery_val := websearch_to_tsquery('english', search_query);

    -- #1376: OR-form of the same lexemes. Match-any with ts_rank_cd ordering
    -- is the BM25-style contract; the AND form made NL questions match 0 rows.
    -- Negated queries (websearch '-term' -> '!term') keep strict AND
    -- semantics: OR-ifying '!foo & bar' into '!foo | bar' would admit rows
    -- CONTAINING foo whenever bar matches, inverting the user's exclusion
    -- (codex iter-1 MEDIUM).
    IF position('!' in tsquery_val::text) > 0 THEN
        tsquery_any := tsquery_val;
    ELSE
        BEGIN
            tsquery_any := replace(tsquery_val::text, ' & ', ' | ')::tsquery;
        EXCEPTION WHEN OTHERS THEN
            tsquery_any := tsquery_val;
        END;
    END IF;

    RETURN QUERY

    -- Search rag_document_chunks
    SELECT
        dc.chunk_id::text as id,
        dc.content as content,
        ts_rank_cd(dc.search_vector, tsquery_any)::double precision as rank,
        jsonb_build_object(
            'document_id', dc.document_id,
            'document_type', dc.document_type,
            'brand', dc.brand,
            'region', dc.region
        ) || dc.metadata as metadata,
        'rag_document_chunks'::text as source_table
    FROM rag_document_chunks dc
    WHERE
        dc.search_vector @@ tsquery_any
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
        ts_rank_cd(cp.search_vector, tsquery_any)::double precision as rank,
        jsonb_build_object(
            'start_node', cp.start_node,
            'end_node', cp.end_node,
            'causal_effect_size', cp.causal_effect_size,
            'confidence_level', cp.confidence_level
        ) as metadata,
        'causal_paths'::text as source_table
    FROM causal_paths cp
    WHERE
        cp.search_vector @@ tsquery_any

    UNION ALL

    -- Search agent_activities
    SELECT
        aa.activity_id::text as id,
        aa.agent_name || ' (' || aa.activity_type || ')' as content,
        ts_rank_cd(aa.search_vector, tsquery_any)::double precision as rank,
        jsonb_build_object(
            'agent_name', aa.agent_name,
            'agent_tier', aa.agent_tier,
            'activity_type', aa.activity_type,
            'status', aa.status
        ) as metadata,
        'agent_activities'::text as source_table
    FROM agent_activities aa
    WHERE
        aa.search_vector @@ tsquery_any
        AND (filters->>'agent_name' IS NULL OR aa.agent_name = filters->>'agent_name')

    UNION ALL

    -- Search triggers
    SELECT
        t.trigger_id::text as id,
        t.trigger_reason as content,
        ts_rank_cd(t.search_vector, tsquery_any)::double precision as rank,
        jsonb_build_object(
            'trigger_type', t.trigger_type,
            'priority', t.priority,
            'confidence_score', t.confidence_score
        ) as metadata,
        'triggers'::text as source_table
    FROM triggers t
    WHERE
        t.search_vector @@ tsquery_any

    ORDER BY rank DESC
    LIMIT match_count;
END;
$function$;
