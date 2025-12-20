-- ============================================================================
-- E2I CAUSAL ANALYTICS - HYBRID SEARCH FUNCTIONS (FIXED VERSION)
-- ============================================================================
-- Migration: 011_hybrid_search_functions_fixed.sql
-- Purpose: Enable hybrid RAG search across vector, fulltext, and graph sources
-- Dependencies: pgvector extension, existing memory tables
-- Created: 2025-12-15
-- Fixed: Multiple corrections applied
--   1. Column references to match actual schema (causal_paths uses causal_chain, not description)
--   2. Ambiguous column names (table_name → tbl_name in get_search_stats)
--   3. ID type mismatches (changed from uuid to text to support varchar IDs)
--   4. Type casting for ts_rank_cd (real → double precision)
-- ============================================================================

-- Verify pgvector extension is enabled
-- Run this manually if not already enabled:
-- CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- PART 1: ADD SEARCH VECTOR COLUMNS TO EXISTING TABLES
-- ============================================================================

-- Add full-text search vectors to causal_paths table
-- Note: causal_paths has NO 'description' column, using causal_chain JSONB instead
ALTER TABLE causal_paths ADD COLUMN IF NOT EXISTS
    search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(start_node, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(end_node, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(method_used, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(causal_chain::text, '')), 'C')
    ) STORED;

-- Add full-text search vectors to agent_activities table
ALTER TABLE agent_activities ADD COLUMN IF NOT EXISTS
    search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(agent_name, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(activity_type, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(analysis_results::text, '')), 'C')
    ) STORED;

-- Add full-text search vectors to triggers table
ALTER TABLE triggers ADD COLUMN IF NOT EXISTS
    search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(trigger_reason, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(trigger_type::text, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(recommended_action, '')), 'C')
    ) STORED;

-- ============================================================================
-- PART 2: CREATE INDEXES FOR FAST SEARCH
-- ============================================================================

-- GIN indexes for fast full-text search
-- These indexes enable efficient ts_rank queries
CREATE INDEX IF NOT EXISTS idx_causal_paths_search
    ON causal_paths USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS idx_agent_activities_search
    ON agent_activities USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS idx_triggers_search
    ON triggers USING GIN(search_vector);

-- HNSW indexes for fast vector search on memory tables
-- These indexes enable approximate nearest neighbor search
-- Adjust m and ef_construction for performance/accuracy tradeoff
CREATE INDEX IF NOT EXISTS idx_episodic_memories_vector_hnsw
    ON episodic_memories
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_procedural_memories_vector_hnsw
    ON procedural_memories
    USING hnsw (trigger_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- PART 3: VECTOR SEARCH FUNCTION
-- ============================================================================

CREATE OR REPLACE FUNCTION hybrid_vector_search(
    query_embedding vector(1536),  -- OpenAI text-embedding-3-small dimension
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

    -- Search episodic_memories (conversation history, user queries, agent actions)
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
        -- Apply optional filters
        (filters->>'brand' IS NULL OR em.brand = filters->>'brand')
        AND (filters->>'region' IS NULL OR em.region = filters->>'region')
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

-- Add comment for documentation
COMMENT ON FUNCTION hybrid_vector_search IS
'Semantic search across episodic_memories and procedural_memories using pgvector cosine similarity. Returns top matches with metadata.';

-- ============================================================================
-- PART 4: FULL-TEXT SEARCH FUNCTION
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
BEGIN
    -- Parse search query with prefix matching for partial words
    -- websearch_to_tsquery handles phrases and boolean operators naturally
    tsquery_val := websearch_to_tsquery('english', search_query);

    RETURN QUERY

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

    -- Search triggers with reason and context
    SELECT
        t.trigger_id::text as id,
        t.trigger_reason as content,
        ts_rank_cd(t.search_vector, tsquery_val)::double precision as rank,
        jsonb_build_object(
            'trigger_type', t.trigger_type,
            'priority', t.priority,
            'confidence_score', t.confidence_score,
            'created_at', t.created_at,
            'recommended_action', t.recommended_action
        ) as metadata,
        'triggers'::text as source_table
    FROM triggers t
    WHERE
        t.search_vector @@ tsquery_val
        AND (filters->>'priority' IS NULL OR t.priority::text = filters->>'priority')

    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- Add comment for documentation
COMMENT ON FUNCTION hybrid_fulltext_search IS
'Full-text search across causal_paths, agent_activities, and triggers using PostgreSQL tsvector and ts_rank. Supports phrases, boolean operators, and partial matching.';

-- ============================================================================
-- PART 5: HELPER FUNCTIONS FOR DEBUGGING
-- ============================================================================

-- Function to test vector search with sample query
CREATE OR REPLACE FUNCTION test_vector_search(
    sample_query text,
    top_k int DEFAULT 5
)
RETURNS TABLE (
    id text,
    content text,
    similarity float,
    source_table text
)
LANGUAGE plpgsql
AS $$
DECLARE
    test_embedding vector(1536);
BEGIN
    -- Generate a zero vector for testing (replace with actual embedding in production)
    test_embedding := array_fill(0.0, ARRAY[1536])::vector(1536);

    RETURN QUERY
    SELECT
        vs.id,
        vs.content,
        vs.similarity,
        vs.source_table
    FROM hybrid_vector_search(test_embedding, top_k) vs;
END;
$$;

-- Function to test fulltext search with sample query
CREATE OR REPLACE FUNCTION test_fulltext_search(
    sample_query text,
    top_k int DEFAULT 5
)
RETURNS TABLE (
    id text,
    content text,
    rank double precision,
    source_table text
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        fs.id,
        fs.content,
        fs.rank,
        fs.source_table
    FROM hybrid_fulltext_search(sample_query, top_k) fs;
END;
$$;

-- Function to get search statistics
CREATE OR REPLACE FUNCTION get_search_stats()
RETURNS TABLE (
    tbl_name text,
    total_rows bigint,
    has_search_vector boolean,
    has_embedding_index boolean
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        'episodic_memories'::text,
        COUNT(*)::bigint,
        EXISTS(SELECT 1 FROM information_schema.columns c
               WHERE c.table_name = 'episodic_memories' AND c.column_name = 'search_text') as has_search_vector,
        EXISTS(SELECT 1 FROM pg_indexes idx
               WHERE idx.tablename = 'episodic_memories' AND idx.indexname LIKE '%vector%')
    FROM episodic_memories

    UNION ALL

    SELECT
        'procedural_memories'::text,
        COUNT(*)::bigint,
        false,
        EXISTS(SELECT 1 FROM pg_indexes idx
               WHERE idx.tablename = 'procedural_memories' AND idx.indexname LIKE '%vector%')
    FROM procedural_memories

    UNION ALL

    SELECT
        'causal_paths'::text,
        COUNT(*)::bigint,
        EXISTS(SELECT 1 FROM pg_indexes idx
               WHERE idx.tablename = 'causal_paths' AND idx.indexname LIKE '%search%'),
        false
    FROM causal_paths

    UNION ALL

    SELECT
        'agent_activities'::text,
        COUNT(*)::bigint,
        EXISTS(SELECT 1 FROM pg_indexes idx
               WHERE idx.tablename = 'agent_activities' AND idx.indexname LIKE '%search%'),
        false
    FROM agent_activities

    UNION ALL

    SELECT
        'triggers'::text,
        COUNT(*)::bigint,
        EXISTS(SELECT 1 FROM pg_indexes idx
               WHERE idx.tablename = 'triggers' AND idx.indexname LIKE '%search%'),
        false
    FROM triggers;
END;
$$;

-- ============================================================================
-- PART 6: GRANT PERMISSIONS (adjust as needed for your security setup)
-- ============================================================================

-- Grant execute permissions to authenticated users
-- Adjust role names based on your Supabase setup
GRANT EXECUTE ON FUNCTION hybrid_vector_search TO authenticated;
GRANT EXECUTE ON FUNCTION hybrid_fulltext_search TO authenticated;
GRANT EXECUTE ON FUNCTION test_vector_search TO authenticated;
GRANT EXECUTE ON FUNCTION test_fulltext_search TO authenticated;
GRANT EXECUTE ON FUNCTION get_search_stats TO authenticated;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Verify installation by running:
-- SELECT * FROM get_search_stats();

-- Test vector search (requires actual embeddings):
-- SELECT * FROM hybrid_vector_search(
--     array_fill(0.1, ARRAY[1536])::vector(1536),
--     10,
--     '{}'::jsonb
-- );

-- Test fulltext search:
-- SELECT * FROM hybrid_fulltext_search(
--     'causal effect conversion',
--     10,
--     '{}'::jsonb
-- );

-- ============================================================================
-- ROLLBACK (if needed)
-- ============================================================================

-- To rollback this migration, run:
/*
-- Note: If functions were already dropped and recreated, you may need to drop them first
DROP FUNCTION IF EXISTS get_search_stats();
DROP FUNCTION IF EXISTS hybrid_vector_search(vector(1536), int, jsonb);
DROP FUNCTION IF EXISTS hybrid_fulltext_search(text, int, jsonb);
DROP FUNCTION IF EXISTS test_vector_search(text, int);
DROP FUNCTION IF EXISTS test_fulltext_search(text, int);

DROP INDEX IF EXISTS idx_causal_paths_search;
DROP INDEX IF EXISTS idx_agent_activities_search;
DROP INDEX IF EXISTS idx_triggers_search;
DROP INDEX IF EXISTS idx_episodic_memories_vector_hnsw;
DROP INDEX IF EXISTS idx_procedural_memories_vector_hnsw;

ALTER TABLE causal_paths DROP COLUMN IF EXISTS search_vector;
ALTER TABLE agent_activities DROP COLUMN IF EXISTS search_vector;
ALTER TABLE triggers DROP COLUMN IF EXISTS search_vector;
*/
