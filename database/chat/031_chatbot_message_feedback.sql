-- =============================================================================
-- E2I Causal Analytics - Chatbot Message Feedback
-- =============================================================================
-- Version: 1.0.0
-- Created: 2026-01-13
-- Description: User feedback (thumbs up/down) for chatbot messages
--
-- Features:
--   - Tracks user ratings on assistant responses
--   - Supports optional text feedback
--   - Links to messages for response quality analysis
--   - Enables feedback-driven prompt optimization
-- =============================================================================

-- =============================================================================
-- ENUMS
-- =============================================================================

-- Feedback rating type
DO $$ BEGIN
    CREATE TYPE public.chatbot_feedback_rating AS ENUM (
        'thumbs_up',    -- Positive feedback
        'thumbs_down'   -- Negative feedback
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- TABLES
-- =============================================================================

-- Chatbot message feedback
CREATE TABLE IF NOT EXISTS public.chatbot_message_feedback (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,

    -- Message reference
    message_id BIGINT NOT NULL REFERENCES public.chatbot_messages(id) ON DELETE CASCADE,

    -- Session reference for RLS (denormalized for performance)
    session_id VARCHAR NOT NULL REFERENCES public.chatbot_conversations(session_id) ON DELETE CASCADE,

    -- Computed user_id from session_id for RLS
    computed_user_id UUID GENERATED ALWAYS AS (
        CAST(SPLIT_PART(session_id, '~', 1) AS UUID)
    ) STORED,

    -- Feedback data
    rating public.chatbot_feedback_rating NOT NULL,
    comment TEXT,  -- Optional user comment explaining the rating

    -- Context at time of feedback
    query_text TEXT,      -- The user query that led to this response
    response_preview TEXT, -- First 500 chars of the response

    -- Analytics
    agent_name TEXT,      -- Which agent generated the response
    tools_used TEXT[],    -- Tools used in generating the response

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata for extensibility
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Prevent duplicate feedback on same message from same session
    CONSTRAINT unique_message_feedback UNIQUE (message_id, session_id)
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Fast lookup by message
CREATE INDEX IF NOT EXISTS idx_chatbot_feedback_message
    ON public.chatbot_message_feedback(message_id);

-- Fast lookup by session
CREATE INDEX IF NOT EXISTS idx_chatbot_feedback_session
    ON public.chatbot_message_feedback(session_id);

-- User-based queries for RLS
CREATE INDEX IF NOT EXISTS idx_chatbot_feedback_user
    ON public.chatbot_message_feedback(computed_user_id);

-- Rating analysis
CREATE INDEX IF NOT EXISTS idx_chatbot_feedback_rating
    ON public.chatbot_message_feedback(rating);

-- Agent performance analysis
CREATE INDEX IF NOT EXISTS idx_chatbot_feedback_agent
    ON public.chatbot_message_feedback(agent_name)
    WHERE agent_name IS NOT NULL;

-- Time-based analytics
CREATE INDEX IF NOT EXISTS idx_chatbot_feedback_created
    ON public.chatbot_message_feedback(created_at DESC);

-- Combined rating + agent for performance queries
CREATE INDEX IF NOT EXISTS idx_chatbot_feedback_agent_rating
    ON public.chatbot_message_feedback(agent_name, rating);

-- =============================================================================
-- ROW LEVEL SECURITY
-- =============================================================================

-- Enable RLS
ALTER TABLE public.chatbot_message_feedback ENABLE ROW LEVEL SECURITY;

-- Users can only see their own feedback
CREATE POLICY chatbot_feedback_select_own ON public.chatbot_message_feedback
    FOR SELECT
    USING (computed_user_id = auth.uid());

-- Users can only insert feedback for their own sessions
CREATE POLICY chatbot_feedback_insert_own ON public.chatbot_message_feedback
    FOR INSERT
    WITH CHECK (computed_user_id = auth.uid());

-- Users can update their own feedback
CREATE POLICY chatbot_feedback_update_own ON public.chatbot_message_feedback
    FOR UPDATE
    USING (computed_user_id = auth.uid());

-- Users can delete their own feedback
CREATE POLICY chatbot_feedback_delete_own ON public.chatbot_message_feedback
    FOR DELETE
    USING (computed_user_id = auth.uid());

-- Service role has full access (for analytics)
CREATE POLICY chatbot_feedback_service_all ON public.chatbot_message_feedback
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to get feedback statistics for an agent
CREATE OR REPLACE FUNCTION public.get_agent_feedback_stats(
    p_agent_name TEXT DEFAULT NULL,
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    agent_name TEXT,
    total_feedback BIGINT,
    thumbs_up_count BIGINT,
    thumbs_down_count BIGINT,
    approval_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.agent_name,
        COUNT(*) as total_feedback,
        COUNT(*) FILTER (WHERE f.rating = 'thumbs_up') as thumbs_up_count,
        COUNT(*) FILTER (WHERE f.rating = 'thumbs_down') as thumbs_down_count,
        ROUND(
            COUNT(*) FILTER (WHERE f.rating = 'thumbs_up')::NUMERIC /
            NULLIF(COUNT(*), 0) * 100,
            2
        ) as approval_rate
    FROM public.chatbot_message_feedback f
    WHERE f.created_at >= NOW() - (p_days || ' days')::INTERVAL
      AND (p_agent_name IS NULL OR f.agent_name = p_agent_name)
    GROUP BY f.agent_name
    ORDER BY total_feedback DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get recent negative feedback for review
CREATE OR REPLACE FUNCTION public.get_negative_feedback(
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id BIGINT,
    message_id BIGINT,
    agent_name TEXT,
    query_text TEXT,
    response_preview TEXT,
    comment TEXT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.id,
        f.message_id,
        f.agent_name,
        f.query_text,
        f.response_preview,
        f.comment,
        f.created_at
    FROM public.chatbot_message_feedback f
    WHERE f.rating = 'thumbs_down'
    ORDER BY f.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE public.chatbot_message_feedback IS
    'User feedback on chatbot assistant responses for quality improvement';

COMMENT ON COLUMN public.chatbot_message_feedback.rating IS
    'User rating: thumbs_up (positive) or thumbs_down (negative)';

COMMENT ON COLUMN public.chatbot_message_feedback.comment IS
    'Optional user comment explaining their rating';

COMMENT ON COLUMN public.chatbot_message_feedback.query_text IS
    'The user query that led to the rated response (for context)';

COMMENT ON COLUMN public.chatbot_message_feedback.response_preview IS
    'First 500 characters of the response (for quick review)';

COMMENT ON FUNCTION public.get_agent_feedback_stats IS
    'Get approval rates and feedback counts per agent for analytics';

COMMENT ON FUNCTION public.get_negative_feedback IS
    'Get recent negative feedback for review and prompt improvement';
