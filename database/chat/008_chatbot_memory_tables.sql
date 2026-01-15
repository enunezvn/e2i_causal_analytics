-- ============================================================================
-- E2I Causal Analytics - Chatbot Memory Tables
-- Migration: 008_chatbot_memory_tables.sql
-- 
-- Purpose: Extend existing tri-memory architecture for chatbot interface
-- Dependencies: 001_agentic_memory_schema.sql (episodic_memories, etc.)
-- 
-- NEW TABLES:
--   1. chat_threads        - Conversation threads (maps to LangGraph thread_id)
--   2. chat_messages       - Message history (supplements Redis working memory)
--   3. user_preferences    - User preferences (simple key-value for chat context)
--
-- MODIFIED TABLES:
--   - user_sessions: Add preferences JSONB column
--
-- NOTE: Does NOT replace Redis working memory or FalkorDB semantic memory
--       These tables provide persistence & search for chat-specific needs
-- ============================================================================

-- Enable required extensions (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ============================================================================
-- TABLE 1: chat_threads
-- Purpose: Track conversation threads (maps to LangGraph checkpointer thread_id)
-- ============================================================================

CREATE TABLE IF NOT EXISTS chat_threads (
    thread_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- User & Session Context
    user_id VARCHAR(100) NOT NULL,
    session_id UUID REFERENCES user_sessions(session_id) ON DELETE SET NULL,
    
    -- Thread Metadata
    title VARCHAR(255),                          -- Auto-generated or user-provided
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_message_at TIMESTAMPTZ,
    
    -- Thread State
    status VARCHAR(20) DEFAULT 'active' 
        CHECK (status IN ('active', 'archived', 'deleted')),
    message_count INTEGER DEFAULT 0,
    
    -- Context Snapshot (filter state at thread creation)
    initial_context JSONB DEFAULT '{}'::jsonb,
    -- Example: {"brand": "Remibrutinib", "region": "south", "tab": "causal"}
    
    -- Agent Activity Summary
    agents_used TEXT[] DEFAULT ARRAY[]::TEXT[],  -- e.g., ['causal-impact', 'gap-analyzer']
    primary_agent VARCHAR(50),                    -- Most active agent in thread
    
    -- Validation Summary (last validation in thread)
    last_validation_id UUID,
    last_gate_decision VARCHAR(20),
    
    -- For search across threads
    topic_embedding vector(1536),                 -- Embedding of thread summary
    
    -- Indexes will be created separately
    CONSTRAINT valid_status CHECK (status IN ('active', 'archived', 'deleted'))
);

-- Indexes for chat_threads
CREATE INDEX IF NOT EXISTS idx_chat_threads_user_id ON chat_threads(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_threads_updated_at ON chat_threads(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_threads_status ON chat_threads(status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_chat_threads_agents ON chat_threads USING GIN(agents_used);

-- Vector index for semantic thread search
CREATE INDEX IF NOT EXISTS idx_chat_threads_embedding ON chat_threads 
    USING ivfflat (topic_embedding vector_cosine_ops) WITH (lists = 50);

-- ============================================================================
-- TABLE 2: chat_messages
-- Purpose: Persist messages for long-term search (Redis handles short-term)
-- ============================================================================

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id UUID NOT NULL REFERENCES chat_threads(thread_id) ON DELETE CASCADE,
    
    -- Message Content
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Sequence in Thread
    sequence_num INTEGER NOT NULL,
    
    -- Agent Attribution (for assistant messages)
    agent_ids TEXT[] DEFAULT ARRAY[]::TEXT[],    -- Agents that contributed
    primary_agent VARCHAR(50),                    -- Main agent for this response
    agent_tier INTEGER,                           -- 0-5 tier of primary agent
    
    -- Tool Calls (for tool messages)
    tool_name VARCHAR(100),
    tool_input JSONB,
    tool_output JSONB,
    
    -- Validation (if message includes causal analysis)
    validation_id UUID,
    gate_decision VARCHAR(20),
    confidence_score FLOAT,
    
    -- Context at Message Time
    filter_context JSONB,                         -- Dashboard filters when sent
    -- Example: {"brand": "Kisqali", "region": "northeast"}
    
    -- For semantic search across messages
    content_embedding vector(1536),
    
    -- Feedback
    feedback_rating INTEGER CHECK (feedback_rating BETWEEN 1 AND 5),
    feedback_text TEXT,
    feedback_at TIMESTAMPTZ,
    
    -- Metadata
    tokens_used INTEGER,
    latency_ms INTEGER,
    
    CONSTRAINT valid_role CHECK (role IN ('user', 'assistant', 'system', 'tool'))
);

-- Indexes for chat_messages
CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id ON chat_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_role ON chat_messages(role);
CREATE INDEX IF NOT EXISTS idx_chat_messages_sequence ON chat_messages(thread_id, sequence_num);
CREATE INDEX IF NOT EXISTS idx_chat_messages_agents ON chat_messages USING GIN(agent_ids);
CREATE INDEX IF NOT EXISTS idx_chat_messages_validation ON chat_messages(validation_id) 
    WHERE validation_id IS NOT NULL;

-- Vector index for semantic message search
CREATE INDEX IF NOT EXISTS idx_chat_messages_embedding ON chat_messages 
    USING ivfflat (content_embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_chat_messages_content_fts ON chat_messages 
    USING GIN(to_tsvector('english', content));

-- ============================================================================
-- TABLE 3: user_preferences
-- Purpose: Simple key-value store for user chat preferences
-- ============================================================================

CREATE TABLE IF NOT EXISTS user_preferences (
    preference_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100) NOT NULL,
    
    -- Preference Data
    key VARCHAR(100) NOT NULL,
    value JSONB NOT NULL,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source VARCHAR(50) DEFAULT 'user',           -- 'user', 'agent', 'system'
    
    -- Unique constraint per user+key
    CONSTRAINT unique_user_preference UNIQUE (user_id, key)
);

-- Common preference keys (documented, not enforced):
-- 'detail_level': 'executive' | 'analyst' | 'data_scientist'
-- 'default_brand': 'Remibrutinib' | 'Fabhalta' | 'Kisqali'
-- 'default_region': 'northeast' | 'south' | 'midwest' | 'west'
-- 'show_validation_badges': true | false
-- 'preferred_chart_types': ['bar', 'line', 'scatter']
-- 'notification_settings': { email: true, in_app: true }

CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_user_preferences_key ON user_preferences(key);

-- ============================================================================
-- MODIFICATION: Add preferences to existing user_sessions table
-- ============================================================================

-- Add preferences JSONB column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'user_sessions' AND column_name = 'preferences'
    ) THEN
        ALTER TABLE user_sessions 
        ADD COLUMN preferences JSONB DEFAULT '{}'::jsonb;
        
        COMMENT ON COLUMN user_sessions.preferences IS 
            'User preferences for chatbot: detail_level, defaults, UI settings';
    END IF;
END $$;

-- ============================================================================
-- FUNCTIONS: Helper functions for chat memory operations
-- ============================================================================

-- Function: Search chat messages semantically
CREATE OR REPLACE FUNCTION search_chat_messages(
    p_user_id VARCHAR(100),
    p_query_embedding vector(1536),
    p_limit INTEGER DEFAULT 10,
    p_thread_id UUID DEFAULT NULL
)
RETURNS TABLE (
    message_id UUID,
    thread_id UUID,
    role VARCHAR(20),
    content TEXT,
    created_at TIMESTAMPTZ,
    agent_ids TEXT[],
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.message_id,
        m.thread_id,
        m.role,
        m.content,
        m.created_at,
        m.agent_ids,
        1 - (m.content_embedding <=> p_query_embedding) AS similarity
    FROM chat_messages m
    JOIN chat_threads t ON m.thread_id = t.thread_id
    WHERE t.user_id = p_user_id
      AND t.status = 'active'
      AND m.content_embedding IS NOT NULL
      AND (p_thread_id IS NULL OR m.thread_id = p_thread_id)
    ORDER BY m.content_embedding <=> p_query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function: Get user's recent threads with message preview
CREATE OR REPLACE FUNCTION get_recent_threads(
    p_user_id VARCHAR(100),
    p_limit INTEGER DEFAULT 20
)
RETURNS TABLE (
    thread_id UUID,
    title VARCHAR(255),
    last_message_at TIMESTAMPTZ,
    message_count INTEGER,
    agents_used TEXT[],
    last_user_message TEXT,
    last_assistant_message TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.thread_id,
        t.title,
        t.last_message_at,
        t.message_count,
        t.agents_used,
        (
            SELECT content FROM chat_messages 
            WHERE chat_messages.thread_id = t.thread_id AND role = 'user'
            ORDER BY sequence_num DESC LIMIT 1
        ) AS last_user_message,
        (
            SELECT content FROM chat_messages 
            WHERE chat_messages.thread_id = t.thread_id AND role = 'assistant'
            ORDER BY sequence_num DESC LIMIT 1
        ) AS last_assistant_message
    FROM chat_threads t
    WHERE t.user_id = p_user_id
      AND t.status = 'active'
    ORDER BY t.last_message_at DESC NULLS LAST
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function: Update thread metadata after new message
CREATE OR REPLACE FUNCTION update_thread_on_message()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE chat_threads
    SET 
        updated_at = NOW(),
        last_message_at = NEW.created_at,
        message_count = message_count + 1,
        agents_used = CASE 
            WHEN NEW.agent_ids IS NOT NULL AND array_length(NEW.agent_ids, 1) > 0
            THEN array_cat(agents_used, NEW.agent_ids)
            ELSE agents_used
        END
    WHERE thread_id = NEW.thread_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Auto-update thread on new message
DROP TRIGGER IF EXISTS trg_update_thread_on_message ON chat_messages;
CREATE TRIGGER trg_update_thread_on_message
    AFTER INSERT ON chat_messages
    FOR EACH ROW
    EXECUTE FUNCTION update_thread_on_message();

-- Function: Get or create user preference
CREATE OR REPLACE FUNCTION upsert_user_preference(
    p_user_id VARCHAR(100),
    p_key VARCHAR(100),
    p_value JSONB,
    p_source VARCHAR(50) DEFAULT 'user'
)
RETURNS user_preferences AS $$
DECLARE
    result user_preferences;
BEGIN
    INSERT INTO user_preferences (user_id, key, value, source, updated_at)
    VALUES (p_user_id, p_key, p_value, p_source, NOW())
    ON CONFLICT (user_id, key) 
    DO UPDATE SET 
        value = p_value,
        updated_at = NOW(),
        source = p_source
    RETURNING * INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Function: Get all preferences for a user as JSONB object
CREATE OR REPLACE FUNCTION get_user_preferences(p_user_id VARCHAR(100))
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_object_agg(key, value)
    INTO result
    FROM user_preferences
    WHERE user_id = p_user_id;
    
    RETURN COALESCE(result, '{}'::jsonb);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- LINKING: Connect chat to existing episodic memory
-- ============================================================================

-- Function: Save chat interaction to episodic memory (for long-term learning)
CREATE OR REPLACE FUNCTION save_chat_to_episodic(
    p_message_id UUID
)
RETURNS UUID AS $$
DECLARE
    v_memory_id UUID;
    v_message RECORD;
BEGIN
    -- Get the message
    SELECT * INTO v_message FROM chat_messages WHERE message_id = p_message_id;
    
    IF NOT FOUND THEN
        RETURN NULL;
    END IF;
    
    -- Only save assistant messages with validation
    IF v_message.role != 'assistant' OR v_message.validation_id IS NULL THEN
        RETURN NULL;
    END IF;
    
    -- Insert into episodic_memories (existing table)
    INSERT INTO episodic_memories (
        event_type,
        description,
        entities,
        outcome_type,
        embedding,
        metadata
    ) VALUES (
        'chat_interaction',
        LEFT(v_message.content, 500),  -- Truncate for description
        jsonb_build_object(
            'thread_id', v_message.thread_id,
            'message_id', v_message.message_id,
            'agents', v_message.agent_ids,
            'filters', v_message.filter_context
        ),
        v_message.gate_decision,
        v_message.content_embedding,
        jsonb_build_object(
            'confidence', v_message.confidence_score,
            'source', 'chatbot'
        )
    )
    RETURNING memory_id INTO v_memory_id;
    
    RETURN v_memory_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS on chat tables
ALTER TABLE chat_threads ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own threads
CREATE POLICY chat_threads_user_policy ON chat_threads
    FOR ALL
    USING (user_id = current_setting('app.current_user_id', true))
    WITH CHECK (user_id = current_setting('app.current_user_id', true));

-- Policy: Users can only see messages in their threads
CREATE POLICY chat_messages_user_policy ON chat_messages
    FOR ALL
    USING (
        thread_id IN (
            SELECT thread_id FROM chat_threads 
            WHERE user_id = current_setting('app.current_user_id', true)
        )
    );

-- Policy: Users can only see their own preferences
CREATE POLICY user_preferences_user_policy ON user_preferences
    FOR ALL
    USING (user_id = current_setting('app.current_user_id', true))
    WITH CHECK (user_id = current_setting('app.current_user_id', true));

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE chat_threads IS 
    'Conversation threads for E2I chatbot. Maps to LangGraph thread_id. Provides search & history beyond Redis TTL.';

COMMENT ON TABLE chat_messages IS 
    'Persistent chat messages. Redis handles working memory; this enables semantic search & long-term history.';

COMMENT ON TABLE user_preferences IS 
    'User preferences for chat interface. Exposed via useCopilotReadable for agent context awareness.';

COMMENT ON FUNCTION search_chat_messages IS 
    'Semantic search across user chat history using pgvector. Used by recallPastDiscussion action.';

COMMENT ON FUNCTION save_chat_to_episodic IS 
    'Promotes significant chat interactions to episodic memory for cross-session learning.';

-- ============================================================================
-- SUMMARY
-- ============================================================================
-- 
-- This migration adds 3 tables for chatbot memory:
--
-- 1. chat_threads (conversation containers)
--    - Maps to LangGraph checkpointer thread_id
--    - Tracks agents used, validation status, topic embedding
--
-- 2. chat_messages (message persistence)
--    - Supplements Redis short-term memory
--    - Enables semantic search with pgvector
--    - Links to validation results
--
-- 3. user_preferences (key-value preferences)
--    - Simple storage for detail_level, defaults, etc.
--    - Exposed to agents via useCopilotReadable
--
-- Integration with existing tables:
--    - user_sessions: Added preferences JSONB column
--    - episodic_memories: Function to promote chat to long-term memory
--
-- Does NOT replace:
--    - Redis: Still primary working memory for active sessions
--    - FalkorDB: Still semantic memory for causal knowledge graph
--    - LangGraph checkpointer: Still manages conversation state
--
-- ============================================================================
