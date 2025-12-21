-- ============================================================================
-- E2I CAUSAL ANALYTICS - RAG SYSTEM SCHEMA
-- ============================================================================
-- Migration: 001_rag_schema.sql
-- Purpose: Tables and functions for the Hybrid RAG system
-- Dependencies: pgvector extension, memory schema
-- Created: 2025-12-20
-- Phase: Phase 1, Checkpoint 1.4
-- ============================================================================

-- Ensure pgvector is enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- PART 1: RAG DOCUMENT CHUNKS TABLE
-- ============================================================================
-- Stores chunked documents for vector search
-- This supplements the existing episodic_memories and procedural_memories tables

CREATE TABLE IF NOT EXISTS rag_document_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Document identification
    document_id VARCHAR(100) NOT NULL,
    document_type VARCHAR(50) NOT NULL,  -- 'causal_insight', 'agent_output', 'trigger_explanation', etc.
    chunk_index INTEGER NOT NULL DEFAULT 0,

    -- Content
    content TEXT NOT NULL,
    content_hash VARCHAR(64),  -- SHA-256 hash for deduplication

    -- E2I context (nullable, for filtering)
    brand VARCHAR(50),
    region VARCHAR(50),
    agent_name VARCHAR(50),
    kpi_name VARCHAR(100),

    -- Metadata
    metadata JSONB DEFAULT '{}',

    -- Vector embedding (OpenAI text-embedding-3-small = 1536 dims)
    embedding vector(1536),

    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(content, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(document_type, '')), 'B')
    ) STORED,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Quality metrics
    embedding_model VARCHAR(100) DEFAULT 'text-embedding-3-small',
    token_count INTEGER,

    -- Uniqueness constraint
    CONSTRAINT unique_document_chunk UNIQUE (document_id, chunk_index)
);

-- Indexes for rag_document_chunks
CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id
    ON rag_document_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_type
    ON rag_document_chunks(document_type);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_brand
    ON rag_document_chunks(brand);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_search_vector
    ON rag_document_chunks USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_hnsw
    ON rag_document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_metadata
    ON rag_document_chunks USING GIN(metadata);

-- ============================================================================
-- PART 2: SEARCH AUDIT LOG TABLE
-- ============================================================================
-- Stores SearchStats for debugging and performance monitoring

CREATE TABLE IF NOT EXISTS rag_search_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Query information
    query TEXT NOT NULL,
    query_hash VARCHAR(64),  -- For grouping identical queries

    -- User/session context
    session_id UUID,
    user_id VARCHAR(100),

    -- Result counts
    vector_count INTEGER DEFAULT 0,
    fulltext_count INTEGER DEFAULT 0,
    graph_count INTEGER DEFAULT 0,
    fused_count INTEGER DEFAULT 0,

    -- Latency breakdown (milliseconds)
    total_latency_ms FLOAT NOT NULL,
    vector_latency_ms FLOAT,
    fulltext_latency_ms FLOAT,
    graph_latency_ms FLOAT,
    fusion_latency_ms FLOAT,

    -- Sources used
    sources_used JSONB DEFAULT '{}',
    -- Example: {"vector": true, "fulltext": true, "graph": false}

    -- Errors
    errors JSONB DEFAULT '[]',
    -- Example: ["Graph backend timeout", "..."]

    -- Search configuration used
    config JSONB DEFAULT '{}',
    -- Example: {"vector_top_k": 20, "rrf_k": 60, ...}

    -- Query metadata
    extracted_entities JSONB DEFAULT '{}',
    -- Example: {"brands": ["Remibrutinib"], "kpis": ["TRx"]}

    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for search logs
CREATE INDEX IF NOT EXISTS idx_rag_search_logs_created_at
    ON rag_search_logs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_rag_search_logs_query_hash
    ON rag_search_logs(query_hash);

CREATE INDEX IF NOT EXISTS idx_rag_search_logs_session
    ON rag_search_logs(session_id);

CREATE INDEX IF NOT EXISTS idx_rag_search_logs_slow_queries
    ON rag_search_logs(total_latency_ms DESC)
    WHERE total_latency_ms > 1000;

-- ============================================================================
-- PART 3: EXTENDED VECTOR SEARCH FUNCTION
-- ============================================================================
-- Adds rag_document_chunks to the vector search

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
        AND (filters->>'brand' IS NULL OR dc.brand = filters->>'brand')
        AND (filters->>'region' IS NULL OR dc.region = filters->>'region')
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
        AND (filters->>'brand' IS NULL OR em.brand = filters->>'brand')
        AND (filters->>'region' IS NULL OR em.region = filters->>'region')
        AND (filters->>'agent_name' IS NULL OR em.agent_name::text = filters->>'agent_name')
        AND (1 - (em.embedding <=> query_embedding)) > 0.3

    UNION ALL

    -- Search procedural_memories
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
'Extended vector search for RAG system. Searches rag_document_chunks, episodic_memories, and procedural_memories.';

-- ============================================================================
-- PART 4: EXTENDED FULLTEXT SEARCH FUNCTION
-- ============================================================================
-- Adds rag_document_chunks to the fulltext search

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
        AND (filters->>'brand' IS NULL OR dc.brand = filters->>'brand')
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
'Extended fulltext search for RAG system. Searches rag_document_chunks, causal_paths, agent_activities, and triggers.';

-- ============================================================================
-- PART 5: SEARCH LOG HELPER FUNCTION
-- ============================================================================
-- Function to insert search logs from Python

CREATE OR REPLACE FUNCTION log_rag_search(
    p_query text,
    p_session_id uuid DEFAULT NULL,
    p_user_id text DEFAULT NULL,
    p_vector_count int DEFAULT 0,
    p_fulltext_count int DEFAULT 0,
    p_graph_count int DEFAULT 0,
    p_fused_count int DEFAULT 0,
    p_total_latency_ms float DEFAULT 0,
    p_vector_latency_ms float DEFAULT NULL,
    p_fulltext_latency_ms float DEFAULT NULL,
    p_graph_latency_ms float DEFAULT NULL,
    p_fusion_latency_ms float DEFAULT NULL,
    p_sources_used jsonb DEFAULT '{}',
    p_errors jsonb DEFAULT '[]',
    p_config jsonb DEFAULT '{}',
    p_extracted_entities jsonb DEFAULT '{}'
)
RETURNS uuid
LANGUAGE plpgsql
AS $$
DECLARE
    v_log_id uuid;
BEGIN
    INSERT INTO rag_search_logs (
        query,
        query_hash,
        session_id,
        user_id,
        vector_count,
        fulltext_count,
        graph_count,
        fused_count,
        total_latency_ms,
        vector_latency_ms,
        fulltext_latency_ms,
        graph_latency_ms,
        fusion_latency_ms,
        sources_used,
        errors,
        config,
        extracted_entities
    ) VALUES (
        p_query,
        encode(sha256(p_query::bytea), 'hex'),
        p_session_id,
        p_user_id,
        p_vector_count,
        p_fulltext_count,
        p_graph_count,
        p_fused_count,
        p_total_latency_ms,
        p_vector_latency_ms,
        p_fulltext_latency_ms,
        p_graph_latency_ms,
        p_fusion_latency_ms,
        p_sources_used,
        p_errors,
        p_config,
        p_extracted_entities
    )
    RETURNING log_id INTO v_log_id;

    RETURN v_log_id;
END;
$$;

COMMENT ON FUNCTION log_rag_search IS
'Helper function to log RAG search queries for auditing and debugging.';

-- ============================================================================
-- PART 6: ANALYTICS VIEWS
-- ============================================================================

-- View for slow queries (queries taking > 1 second)
CREATE OR REPLACE VIEW rag_slow_queries AS
SELECT
    log_id,
    query,
    total_latency_ms,
    vector_latency_ms,
    fulltext_latency_ms,
    graph_latency_ms,
    fused_count,
    errors,
    created_at
FROM rag_search_logs
WHERE total_latency_ms > 1000
ORDER BY total_latency_ms DESC;

-- View for search performance summary
CREATE OR REPLACE VIEW rag_search_stats AS
SELECT
    date_trunc('hour', created_at) as hour,
    COUNT(*) as query_count,
    AVG(total_latency_ms) as avg_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_latency_ms) as p95_latency_ms,
    AVG(fused_count) as avg_results,
    SUM(CASE WHEN errors != '[]' THEN 1 ELSE 0 END) as error_count
FROM rag_search_logs
GROUP BY date_trunc('hour', created_at)
ORDER BY hour DESC;

-- ============================================================================
-- PART 7: PERMISSIONS
-- ============================================================================

-- Grant execute permissions to authenticated users
GRANT SELECT, INSERT ON rag_document_chunks TO authenticated;
GRANT SELECT, INSERT ON rag_search_logs TO authenticated;
GRANT EXECUTE ON FUNCTION rag_vector_search TO authenticated;
GRANT EXECUTE ON FUNCTION rag_fulltext_search TO authenticated;
GRANT EXECUTE ON FUNCTION log_rag_search TO authenticated;

-- Grant service role full access
GRANT ALL ON rag_document_chunks TO service_role;
GRANT ALL ON rag_search_logs TO service_role;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Verify installation:
-- SELECT * FROM information_schema.tables WHERE table_name LIKE 'rag_%';
-- SELECT * FROM information_schema.routines WHERE routine_name LIKE 'rag_%';

-- Test functions:
-- SELECT * FROM rag_fulltext_search('conversion rate', 5);
-- SELECT * FROM log_rag_search('test query', null, null, 5, 3, 0, 8, 150.0);

-- ============================================================================
-- ROLLBACK (if needed)
-- ============================================================================

/*
DROP VIEW IF EXISTS rag_search_stats;
DROP VIEW IF EXISTS rag_slow_queries;
DROP FUNCTION IF EXISTS log_rag_search;
DROP FUNCTION IF EXISTS rag_fulltext_search;
DROP FUNCTION IF EXISTS rag_vector_search;
DROP TABLE IF EXISTS rag_search_logs;
DROP TABLE IF EXISTS rag_document_chunks;
*/
