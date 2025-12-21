-- =============================================================================
-- Conversation Similarity Search RPC Function
-- Migration: 016_conversation_similarity_search.sql
-- Date: 2025-12-20
-- Purpose: Enable vector similarity search on cognitive_cycles for RAG context
-- Reference: https://supabase.com/docs/guides/ai/vector-columns
-- =============================================================================

-- -----------------------------------------------------------------------------
-- search_similar_conversations
--
-- Finds similar past conversations/queries using pgvector cosine similarity.
-- Uses cognitive_cycles table which stores user queries with embeddings.
--
-- Usage from Supabase client:
--   result = await client.rpc('search_similar_conversations', {
--       'query_embedding': [0.1, 0.2, ...],  -- 1536-dim vector
--       'match_count': 5,
--       'min_similarity': 0.5
--   }).execute()
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION search_similar_conversations(
    query_embedding vector(1536),
    match_count int DEFAULT 5,
    min_similarity float DEFAULT 0.5,
    filter_user_id text DEFAULT NULL,
    filter_session_id uuid DEFAULT NULL
)
RETURNS TABLE (
    cycle_id uuid,
    session_id uuid,
    user_id varchar(100),
    user_query text,
    detected_intent varchar(50),
    detected_entities jsonb,
    agent_response text,
    response_type varchar(50),
    feedback_type varchar(20),
    feedback_text text,
    similarity float,
    created_at timestamptz
)
LANGUAGE plpgsql
SECURITY INVOKER  -- Respects RLS policies
AS $$
BEGIN
    RETURN QUERY
    SELECT
        cc.cycle_id,
        cc.session_id,
        cc.user_id,
        cc.user_query,
        cc.detected_intent,
        cc.detected_entities,
        cc.agent_response,
        cc.response_type,
        cc.feedback_type,
        cc.feedback_text,
        1 - (cc.query_embedding <=> query_embedding) AS similarity,
        cc.created_at
    FROM cognitive_cycles cc
    WHERE
        cc.query_embedding IS NOT NULL
        AND (1 - (cc.query_embedding <=> query_embedding)) >= min_similarity
        AND (filter_user_id IS NULL OR cc.user_id = filter_user_id)
        AND (filter_session_id IS NULL OR cc.session_id = filter_session_id)
    ORDER BY cc.query_embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Grant execute permission to authenticated users
GRANT EXECUTE ON FUNCTION search_similar_conversations TO authenticated;
GRANT EXECUTE ON FUNCTION search_similar_conversations TO service_role;

-- -----------------------------------------------------------------------------
-- get_conversations_with_feedback
--
-- Retrieves conversations that have user feedback for self-improvement.
-- Supports filtering by feedback type (positive/negative/neutral).
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION get_conversations_with_feedback(
    p_feedback_type text DEFAULT NULL,
    p_limit int DEFAULT 100,
    p_offset int DEFAULT 0,
    p_user_id text DEFAULT NULL,
    p_since timestamptz DEFAULT NULL
)
RETURNS TABLE (
    cycle_id uuid,
    session_id uuid,
    user_id varchar(100),
    user_query text,
    detected_intent varchar(50),
    agent_response text,
    response_type varchar(50),
    feedback_type varchar(20),
    feedback_text text,
    feedback_score int,
    created_at timestamptz,
    feedback_at timestamptz
)
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    RETURN QUERY
    SELECT
        cc.cycle_id,
        cc.session_id,
        cc.user_id,
        cc.user_query,
        cc.detected_intent,
        cc.agent_response,
        cc.response_type,
        cc.feedback_type,
        cc.feedback_text,
        cc.feedback_score,
        cc.created_at,
        cc.feedback_at
    FROM cognitive_cycles cc
    WHERE
        cc.feedback_type IS NOT NULL
        AND (p_feedback_type IS NULL OR cc.feedback_type = p_feedback_type)
        AND (p_user_id IS NULL OR cc.user_id = p_user_id)
        AND (p_since IS NULL OR cc.created_at >= p_since)
    ORDER BY cc.feedback_at DESC NULLS LAST, cc.created_at DESC
    LIMIT p_limit
    OFFSET p_offset;
END;
$$;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION get_conversations_with_feedback TO authenticated;
GRANT EXECUTE ON FUNCTION get_conversations_with_feedback TO service_role;

-- -----------------------------------------------------------------------------
-- Comment documentation
-- -----------------------------------------------------------------------------

COMMENT ON FUNCTION search_similar_conversations IS
'Vector similarity search on cognitive_cycles using pgvector cosine distance.
Returns past conversations similar to the query embedding for RAG context retrieval.
Uses IVFFlat index on query_embedding for efficient approximate nearest neighbor search.';

COMMENT ON FUNCTION get_conversations_with_feedback IS
'Retrieves conversations with user feedback for self-improvement learning.
Supports filtering by feedback type and time range.';
