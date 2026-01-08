-- =============================================================================
-- E2I Causal Analytics - Chatbot Conversations and Messages
-- =============================================================================
-- Version: 1.0.0
-- Created: 2026-01-08
-- Description: Conversation history and messages for E2I chatbot
--
-- Features:
--   - Conversations with E2I context (brand, region, query type)
--   - Messages with agent attribution and tool call tracking
--   - Session-based organization (session_id format: user_id~uuid)
--   - Support for conversation archiving
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- ENUMS
-- =============================================================================

-- Query type classification for analytics
DO $$ BEGIN
    CREATE TYPE public.chatbot_query_type AS ENUM (
        'kpi_inquiry',          -- Questions about KPIs
        'causal_analysis',      -- Causal chain/impact questions
        'agent_status',         -- Agent system queries
        'recommendation',       -- Asking for recommendations
        'experiment',           -- A/B test and experiment queries
        'prediction',           -- ML prediction queries
        'drift_alert',          -- Drift monitoring queries
        'general',              -- General questions
        'multi_faceted'         -- Complex queries spanning multiple types
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Message role types
DO $$ BEGIN
    CREATE TYPE public.chatbot_message_role AS ENUM (
        'user',         -- User input
        'assistant',    -- AI response
        'system',       -- System messages
        'tool'          -- Tool call results
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- TABLES
-- =============================================================================

-- Chatbot conversations with E2I context
CREATE TABLE IF NOT EXISTS public.chatbot_conversations (
    session_id VARCHAR PRIMARY KEY NOT NULL,
    user_id UUID NOT NULL REFERENCES public.chatbot_user_profiles(id) ON DELETE CASCADE,

    -- Conversation metadata
    title VARCHAR,  -- Auto-generated from first message
    summary TEXT,   -- AI-generated summary

    -- E2I Context
    brand_context TEXT,   -- Brand filter for this conversation
    region_context TEXT,  -- Region filter for this conversation
    query_type public.chatbot_query_type DEFAULT 'general',

    -- State
    is_archived BOOLEAN DEFAULT FALSE,
    is_pinned BOOLEAN DEFAULT FALSE,

    -- Message counts
    message_count INTEGER DEFAULT 0,
    user_message_count INTEGER DEFAULT 0,
    assistant_message_count INTEGER DEFAULT 0,

    -- Tools and agents used
    tools_used TEXT[] DEFAULT '{}',
    agents_invoked TEXT[] DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_message_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata for extensibility
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Chatbot messages with agent attribution
CREATE TABLE IF NOT EXISTS public.chatbot_messages (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,

    -- Session reference (format: user_id~uuid)
    session_id VARCHAR NOT NULL REFERENCES public.chatbot_conversations(session_id) ON DELETE CASCADE,

    -- Computed user_id from session_id for RLS
    computed_user_id UUID GENERATED ALWAYS AS (
        CAST(SPLIT_PART(session_id, '~', 1) AS UUID)
    ) STORED,

    -- Message content
    role public.chatbot_message_role NOT NULL,
    content TEXT NOT NULL,

    -- Agent attribution (which agent generated this response)
    agent_name TEXT,  -- e.g., 'orchestrator', 'causal_impact', 'explainer'
    agent_tier INTEGER,  -- Tier 0-5

    -- Tool tracking
    tool_calls JSONB DEFAULT '[]'::jsonb,  -- Array of {tool_name, input, output}
    tool_results JSONB DEFAULT '[]'::jsonb,

    -- RAG context
    rag_context JSONB DEFAULT '[]'::jsonb,  -- Retrieved documents used
    rag_sources TEXT[] DEFAULT '{}',

    -- Response metadata
    model_used TEXT,  -- e.g., 'claude-3-opus'
    tokens_used INTEGER,
    latency_ms INTEGER,
    confidence_score FLOAT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Extensibility
    metadata JSONB DEFAULT '{}'::jsonb
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Conversations indexes
CREATE INDEX IF NOT EXISTS idx_chatbot_conversations_user
    ON public.chatbot_conversations(user_id);

CREATE INDEX IF NOT EXISTS idx_chatbot_conversations_created
    ON public.chatbot_conversations(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_chatbot_conversations_last_message
    ON public.chatbot_conversations(last_message_at DESC);

CREATE INDEX IF NOT EXISTS idx_chatbot_conversations_brand
    ON public.chatbot_conversations(brand_context)
    WHERE brand_context IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chatbot_conversations_query_type
    ON public.chatbot_conversations(query_type);

CREATE INDEX IF NOT EXISTS idx_chatbot_conversations_archived
    ON public.chatbot_conversations(is_archived)
    WHERE is_archived = FALSE;

CREATE INDEX IF NOT EXISTS idx_chatbot_conversations_tools_used
    ON public.chatbot_conversations USING GIN(tools_used);

-- Messages indexes
CREATE INDEX IF NOT EXISTS idx_chatbot_messages_session
    ON public.chatbot_messages(session_id);

CREATE INDEX IF NOT EXISTS idx_chatbot_messages_computed_user
    ON public.chatbot_messages(computed_user_id);

CREATE INDEX IF NOT EXISTS idx_chatbot_messages_created
    ON public.chatbot_messages(created_at);

CREATE INDEX IF NOT EXISTS idx_chatbot_messages_role
    ON public.chatbot_messages(role);

CREATE INDEX IF NOT EXISTS idx_chatbot_messages_agent
    ON public.chatbot_messages(agent_name)
    WHERE agent_name IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chatbot_messages_session_created
    ON public.chatbot_messages(session_id, created_at);

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- Auto-update conversation updated_at
CREATE OR REPLACE FUNCTION public.update_chatbot_conversations_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_chatbot_conversations_timestamp
    ON public.chatbot_conversations;
CREATE TRIGGER trigger_update_chatbot_conversations_timestamp
    BEFORE UPDATE ON public.chatbot_conversations
    FOR EACH ROW
    EXECUTE FUNCTION public.update_chatbot_conversations_timestamp();

-- Auto-update conversation stats on new message
CREATE OR REPLACE FUNCTION public.update_conversation_on_new_message()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE public.chatbot_conversations
    SET
        message_count = message_count + 1,
        user_message_count = user_message_count + CASE WHEN NEW.role = 'user' THEN 1 ELSE 0 END,
        assistant_message_count = assistant_message_count + CASE WHEN NEW.role = 'assistant' THEN 1 ELSE 0 END,
        last_message_at = NEW.created_at,
        updated_at = NOW(),
        -- Accumulate agents used
        agents_invoked = CASE
            WHEN NEW.agent_name IS NOT NULL AND NOT NEW.agent_name = ANY(agents_invoked)
            THEN array_append(agents_invoked, NEW.agent_name)
            ELSE agents_invoked
        END
    WHERE session_id = NEW.session_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_conversation_on_new_message
    ON public.chatbot_messages;
CREATE TRIGGER trigger_update_conversation_on_new_message
    AFTER INSERT ON public.chatbot_messages
    FOR EACH ROW
    EXECUTE FUNCTION public.update_conversation_on_new_message();

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Generate session_id from user_id
CREATE OR REPLACE FUNCTION public.generate_chatbot_session_id(p_user_id UUID)
RETURNS VARCHAR AS $$
BEGIN
    RETURN p_user_id::TEXT || '~' || gen_random_uuid()::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Get conversation history for a session
CREATE OR REPLACE FUNCTION public.get_chatbot_conversation_history(
    p_session_id VARCHAR,
    p_limit INTEGER DEFAULT 50,
    p_offset INTEGER DEFAULT 0
)
RETURNS TABLE (
    id BIGINT,
    role public.chatbot_message_role,
    content TEXT,
    agent_name TEXT,
    tool_calls JSONB,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.role,
        m.content,
        m.agent_name,
        m.tool_calls,
        m.created_at
    FROM public.chatbot_messages m
    WHERE m.session_id = p_session_id
    ORDER BY m.created_at ASC
    LIMIT p_limit
    OFFSET p_offset;
END;
$$ LANGUAGE plpgsql;

-- Get user's recent conversations
CREATE OR REPLACE FUNCTION public.get_user_chatbot_conversations(
    p_user_id UUID,
    p_limit INTEGER DEFAULT 20,
    p_include_archived BOOLEAN DEFAULT FALSE
)
RETURNS TABLE (
    session_id VARCHAR,
    title VARCHAR,
    query_type public.chatbot_query_type,
    brand_context TEXT,
    message_count INTEGER,
    last_message_at TIMESTAMPTZ,
    is_archived BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.session_id,
        c.title,
        c.query_type,
        c.brand_context,
        c.message_count,
        c.last_message_at,
        c.is_archived
    FROM public.chatbot_conversations c
    WHERE c.user_id = p_user_id
      AND (p_include_archived OR c.is_archived = FALSE)
    ORDER BY c.last_message_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Auto-generate conversation title from first user message
CREATE OR REPLACE FUNCTION public.set_chatbot_conversation_title(
    p_session_id VARCHAR,
    p_title VARCHAR
)
RETURNS VOID AS $$
BEGIN
    UPDATE public.chatbot_conversations
    SET title = p_title, updated_at = NOW()
    WHERE session_id = p_session_id AND title IS NULL;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- VIEWS
-- =============================================================================

-- View: Recent conversations with summary
CREATE OR REPLACE VIEW public.v_chatbot_recent_conversations AS
SELECT
    c.session_id,
    c.user_id,
    c.title,
    c.query_type,
    c.brand_context,
    c.message_count,
    c.agents_invoked,
    c.last_message_at,
    c.created_at,
    up.email as user_email,
    up.expertise_level
FROM public.chatbot_conversations c
JOIN public.chatbot_user_profiles up ON up.id = c.user_id
WHERE c.is_archived = FALSE
ORDER BY c.last_message_at DESC;

-- View: Agent usage statistics
CREATE OR REPLACE VIEW public.v_chatbot_agent_usage AS
SELECT
    agent_name,
    COUNT(*) as message_count,
    COUNT(DISTINCT session_id) as conversation_count,
    AVG(latency_ms) as avg_latency_ms,
    AVG(tokens_used) as avg_tokens
FROM public.chatbot_messages
WHERE agent_name IS NOT NULL
GROUP BY agent_name
ORDER BY message_count DESC;

-- =============================================================================
-- GRANTS
-- =============================================================================

GRANT SELECT, INSERT, UPDATE ON public.chatbot_conversations TO authenticated;
GRANT SELECT, INSERT ON public.chatbot_messages TO authenticated;
GRANT SELECT ON public.v_chatbot_recent_conversations TO authenticated;
GRANT SELECT ON public.v_chatbot_agent_usage TO authenticated;

GRANT EXECUTE ON FUNCTION public.generate_chatbot_session_id(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION public.get_chatbot_conversation_history(VARCHAR, INTEGER, INTEGER) TO authenticated;
GRANT EXECUTE ON FUNCTION public.get_user_chatbot_conversations(UUID, INTEGER, BOOLEAN) TO authenticated;
GRANT EXECUTE ON FUNCTION public.set_chatbot_conversation_title(VARCHAR, VARCHAR) TO authenticated;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE public.chatbot_conversations IS
    'Chatbot conversation sessions with E2I context (brand, region, query type)';
COMMENT ON TABLE public.chatbot_messages IS
    'Individual messages in chatbot conversations with agent attribution';
COMMENT ON COLUMN public.chatbot_conversations.session_id IS
    'Format: user_id~uuid (e.g., abc123-...~def456-...)';
COMMENT ON COLUMN public.chatbot_messages.computed_user_id IS
    'Auto-extracted user_id from session_id for RLS enforcement';
COMMENT ON COLUMN public.chatbot_messages.agent_name IS
    'E2I agent that generated this response (orchestrator, causal_impact, etc.)';
COMMENT ON COLUMN public.chatbot_messages.tool_calls IS
    'Array of tool calls made: [{tool_name, input, output}]';
