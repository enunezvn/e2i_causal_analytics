-- ============================================================================
-- E2I CAUSAL ANALYTICS - HYBRID SEARCH FUNCTIONS
-- ============================================================================
-- Migration: 003_hybrid_search_functions.sql
-- Purpose: RRF-based hybrid search combining vector and fulltext
-- Dependencies: 001_rag_schema.sql
-- Created: 2025-12-20
-- Phase: Phase 2, Checkpoint 2.2
-- ============================================================================

-- ============================================================================
-- PART 1: RECIPROCAL RANK FUSION (RRF) FUNCTION
-- ============================================================================
-- Combines results from vector and fulltext search using RRF algorithm
-- RRF Score = SUM(1 / (k + rank)) for each result across all sources
-- Default k=60 is standard for RRF

CREATE OR REPLACE FUNCTION rag_hybrid_search(
    query_embedding vector(1536),
    search_query text,
    match_count int DEFAULT 20,
    filters jsonb DEFAULT '{}'::jsonb,
    rrf_k int DEFAULT 60,
    vector_weight float DEFAULT 0.5,
    fulltext_weight float DEFAULT 0.5
)
RETURNS TABLE (
    id text,
    content text,
    combined_score float,
    vector_score float,
    fulltext_score float,
    metadata jsonb,
    source_table text
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH
    -- Get vector search results with rank
    vector_results AS (
        SELECT
            v.id,
            v.content,
            v.similarity as score,
            v.metadata,
            v.source_table,
            ROW_NUMBER() OVER (ORDER BY v.similarity DESC) as rank
        FROM rag_vector_search(query_embedding, match_count * 2, filters) v
    ),
    -- Get fulltext search results with rank
    fulltext_results AS (
        SELECT
            f.id,
            f.content,
            f.rank::float as score,
            f.metadata,
            f.source_table,
            ROW_NUMBER() OVER (ORDER BY f.rank DESC) as rank
        FROM rag_fulltext_search(search_query, match_count * 2, filters) f
    ),
    -- Combine using RRF
    combined AS (
        SELECT
            COALESCE(v.id, f.id) as id,
            COALESCE(v.content, f.content) as content,
            -- RRF formula with weights
            (
                COALESCE(vector_weight * (1.0 / (rrf_k + COALESCE(v.rank, 1000))), 0) +
                COALESCE(fulltext_weight * (1.0 / (rrf_k + COALESCE(f.rank, 1000))), 0)
            ) as combined_score,
            COALESCE(v.score, 0)::float as vector_score,
            COALESCE(f.score, 0)::float as fulltext_score,
            COALESCE(v.metadata, f.metadata) as metadata,
            COALESCE(v.source_table, f.source_table) as source_table
        FROM vector_results v
        FULL OUTER JOIN fulltext_results f ON v.id = f.id
    )
    SELECT
        c.id,
        c.content,
        c.combined_score,
        c.vector_score,
        c.fulltext_score,
        c.metadata,
        c.source_table
    FROM combined c
    WHERE c.combined_score > 0
    ORDER BY c.combined_score DESC
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION rag_hybrid_search IS
'Hybrid search combining vector similarity and fulltext using Reciprocal Rank Fusion (RRF).
 Parameters:
   - query_embedding: 1536-dim vector from embedding model
   - search_query: Natural language query for fulltext
   - match_count: Number of results to return
   - filters: JSONB with brand, region, document_type filters
   - rrf_k: RRF constant (default 60)
   - vector_weight: Weight for vector results (default 0.5)
   - fulltext_weight: Weight for fulltext results (default 0.5)';


-- ============================================================================
-- PART 2: SEARCH STATISTICS FUNCTION
-- ============================================================================
-- Get search statistics for the /stats endpoint

CREATE OR REPLACE FUNCTION get_rag_search_stats(
    hours_lookback int DEFAULT 24
)
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
    result jsonb;
    cutoff_time timestamptz;
BEGIN
    cutoff_time := NOW() - (hours_lookback || ' hours')::interval;

    SELECT jsonb_build_object(
        'period_hours', hours_lookback,
        'total_searches', COALESCE(COUNT(*), 0),
        'avg_latency_ms', COALESCE(ROUND(AVG(total_latency_ms)::numeric, 2), 0),
        'p95_latency_ms', COALESCE(
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_latency_ms)::numeric, 2),
            0
        ),
        'avg_results', COALESCE(ROUND(AVG(fused_count)::numeric, 2), 0),
        'error_rate', COALESCE(
            ROUND(
                (SUM(CASE WHEN errors != '[]' THEN 1 ELSE 0 END)::float /
                 NULLIF(COUNT(*), 0) * 100)::numeric,
                2
            ),
            0
        ),
        'backend_usage', jsonb_build_object(
            'vector', COALESCE(SUM(vector_count), 0),
            'fulltext', COALESCE(SUM(fulltext_count), 0),
            'graph', COALESCE(SUM(graph_count), 0)
        ),
        'top_queries', COALESCE(
            (SELECT jsonb_agg(q.query_info)
             FROM (
                 SELECT jsonb_build_object(
                     'query', query,
                     'count', COUNT(*),
                     'avg_latency_ms', ROUND(AVG(total_latency_ms)::numeric, 2)
                 ) as query_info
                 FROM rag_search_logs
                 WHERE created_at >= cutoff_time
                 GROUP BY query
                 ORDER BY COUNT(*) DESC
                 LIMIT 10
             ) q),
            '[]'::jsonb
        )
    ) INTO result
    FROM rag_search_logs
    WHERE created_at >= cutoff_time;

    RETURN result;
END;
$$;

COMMENT ON FUNCTION get_rag_search_stats IS
'Get aggregated search statistics for the specified time period.
 Returns: JSONB with total_searches, avg_latency_ms, p95_latency_ms, error_rate, backend_usage, top_queries';


-- ============================================================================
-- PART 3: DOCUMENT CHUNK UPSERT FUNCTION
-- ============================================================================
-- Upsert documents with automatic deduplication

CREATE OR REPLACE FUNCTION upsert_rag_chunk(
    p_document_id text,
    p_content text,
    p_embedding vector(1536),
    p_document_type text DEFAULT 'general',
    p_chunk_index int DEFAULT 0,
    p_brand text DEFAULT NULL,
    p_region text DEFAULT NULL,
    p_agent_name text DEFAULT NULL,
    p_kpi_name text DEFAULT NULL,
    p_metadata jsonb DEFAULT '{}'::jsonb
)
RETURNS uuid
LANGUAGE plpgsql
AS $$
DECLARE
    v_chunk_id uuid;
    v_content_hash text;
BEGIN
    -- Compute content hash for deduplication
    v_content_hash := encode(sha256(p_content::bytea), 'hex');

    -- Upsert the chunk
    INSERT INTO rag_document_chunks (
        document_id,
        document_type,
        chunk_index,
        content,
        content_hash,
        embedding,
        brand,
        region,
        agent_name,
        kpi_name,
        metadata,
        token_count
    ) VALUES (
        p_document_id,
        p_document_type,
        p_chunk_index,
        p_content,
        v_content_hash,
        p_embedding,
        p_brand,
        p_region,
        p_agent_name,
        p_kpi_name,
        p_metadata,
        array_length(regexp_split_to_array(p_content, '\s+'), 1)
    )
    ON CONFLICT (document_id, chunk_index)
    DO UPDATE SET
        content = EXCLUDED.content,
        content_hash = EXCLUDED.content_hash,
        embedding = EXCLUDED.embedding,
        brand = EXCLUDED.brand,
        region = EXCLUDED.region,
        agent_name = EXCLUDED.agent_name,
        kpi_name = EXCLUDED.kpi_name,
        metadata = EXCLUDED.metadata,
        token_count = EXCLUDED.token_count,
        updated_at = NOW()
    RETURNING chunk_id INTO v_chunk_id;

    RETURN v_chunk_id;
END;
$$;

COMMENT ON FUNCTION upsert_rag_chunk IS
'Upsert a document chunk with automatic content hashing and deduplication.
 Returns the chunk_id (UUID) of the upserted chunk.';


-- ============================================================================
-- PART 4: BULK DOCUMENT INDEXING FUNCTION
-- ============================================================================
-- Index multiple documents in a single transaction

CREATE OR REPLACE FUNCTION bulk_index_rag_chunks(
    p_chunks jsonb
)
RETURNS TABLE (
    document_id text,
    chunk_index int,
    chunk_id uuid,
    success boolean,
    error_message text
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_chunk jsonb;
    v_chunk_id uuid;
    v_error text;
BEGIN
    FOR v_chunk IN SELECT * FROM jsonb_array_elements(p_chunks)
    LOOP
        BEGIN
            SELECT upsert_rag_chunk(
                v_chunk->>'document_id',
                v_chunk->>'content',
                (v_chunk->>'embedding')::vector(1536),
                COALESCE(v_chunk->>'document_type', 'general'),
                COALESCE((v_chunk->>'chunk_index')::int, 0),
                v_chunk->>'brand',
                v_chunk->>'region',
                v_chunk->>'agent_name',
                v_chunk->>'kpi_name',
                COALESCE(v_chunk->'metadata', '{}'::jsonb)
            ) INTO v_chunk_id;

            document_id := v_chunk->>'document_id';
            chunk_index := COALESCE((v_chunk->>'chunk_index')::int, 0);
            chunk_id := v_chunk_id;
            success := true;
            error_message := NULL;
            RETURN NEXT;

        EXCEPTION WHEN OTHERS THEN
            document_id := v_chunk->>'document_id';
            chunk_index := COALESCE((v_chunk->>'chunk_index')::int, 0);
            chunk_id := NULL;
            success := false;
            error_message := SQLERRM;
            RETURN NEXT;
        END;
    END LOOP;
END;
$$;

COMMENT ON FUNCTION bulk_index_rag_chunks IS
'Bulk index multiple document chunks. Accepts JSONB array of chunks.
 Each chunk should have: document_id, content, embedding, and optional metadata fields.
 Returns status for each chunk including any errors.';


-- ============================================================================
-- PART 5: FIND SIMILAR DOCUMENTS FUNCTION
-- ============================================================================
-- Find documents similar to a given document (for related content suggestions)

CREATE OR REPLACE FUNCTION find_similar_documents(
    p_document_id text,
    match_count int DEFAULT 5
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
    v_embedding vector(1536);
BEGIN
    -- Get the embedding for the source document
    SELECT embedding INTO v_embedding
    FROM rag_document_chunks
    WHERE document_id = p_document_id
    LIMIT 1;

    IF v_embedding IS NULL THEN
        RAISE EXCEPTION 'Document not found: %', p_document_id;
    END IF;

    -- Find similar documents (excluding the source)
    RETURN QUERY
    SELECT
        r.id,
        r.content,
        r.similarity,
        r.metadata,
        r.source_table
    FROM rag_vector_search(v_embedding, match_count + 1, '{}'::jsonb) r
    WHERE r.metadata->>'document_id' != p_document_id
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION find_similar_documents IS
'Find documents similar to a given document ID.
 Useful for "related content" suggestions.';


-- ============================================================================
-- PART 6: DELETE OLD SEARCH LOGS FUNCTION
-- ============================================================================
-- Cleanup function for old search logs

CREATE OR REPLACE FUNCTION cleanup_old_search_logs(
    days_to_keep int DEFAULT 30
)
RETURNS int
LANGUAGE plpgsql
AS $$
DECLARE
    v_deleted_count int;
BEGIN
    DELETE FROM rag_search_logs
    WHERE created_at < NOW() - (days_to_keep || ' days')::interval;

    GET DIAGNOSTICS v_deleted_count = ROW_COUNT;
    RETURN v_deleted_count;
END;
$$;

COMMENT ON FUNCTION cleanup_old_search_logs IS
'Delete search logs older than the specified number of days.
 Returns the number of deleted records.';


-- ============================================================================
-- PERMISSIONS
-- ============================================================================

GRANT EXECUTE ON FUNCTION rag_hybrid_search TO authenticated;
GRANT EXECUTE ON FUNCTION get_rag_search_stats TO authenticated;
GRANT EXECUTE ON FUNCTION upsert_rag_chunk TO authenticated;
GRANT EXECUTE ON FUNCTION bulk_index_rag_chunks TO authenticated;
GRANT EXECUTE ON FUNCTION find_similar_documents TO authenticated;
GRANT EXECUTE ON FUNCTION cleanup_old_search_logs TO service_role;


-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Test the hybrid search (requires populated data):
-- SELECT * FROM rag_hybrid_search(
--     '[0.1, 0.2, ...]'::vector(1536),  -- actual embedding
--     'Kisqali TRx conversion',
--     10
-- );

-- Test statistics:
-- SELECT * FROM get_rag_search_stats(24);

-- ============================================================================
-- ROLLBACK (if needed)
-- ============================================================================

/*
DROP FUNCTION IF EXISTS cleanup_old_search_logs;
DROP FUNCTION IF EXISTS find_similar_documents;
DROP FUNCTION IF EXISTS bulk_index_rag_chunks;
DROP FUNCTION IF EXISTS upsert_rag_chunk;
DROP FUNCTION IF EXISTS get_rag_search_stats;
DROP FUNCTION IF EXISTS rag_hybrid_search;
*/
