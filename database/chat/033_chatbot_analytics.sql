-- =============================================================================
-- E2I Causal Analytics - Chatbot Usage Analytics
-- =============================================================================
-- Version: 1.0.0
-- Created: 2026-01-13
-- Description: Track chatbot usage patterns, response times, and tool usage
--
-- Features:
--   - Tracks query types and classification distribution
--   - Measures response times for performance monitoring
--   - Records tool usage for agent optimization
--   - Enables usage analytics dashboards and reporting
-- =============================================================================

-- =============================================================================
-- TABLES
-- =============================================================================

-- Chatbot usage analytics
CREATE TABLE IF NOT EXISTS public.chatbot_analytics (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,

    -- Session reference
    session_id VARCHAR NOT NULL REFERENCES public.chatbot_conversations(session_id) ON DELETE CASCADE,

    -- Message reference (optional - not all analytics entries have messages)
    message_id BIGINT REFERENCES public.chatbot_messages(id) ON DELETE SET NULL,

    -- Query classification
    query_type TEXT NOT NULL,          -- kpi_inquiry, causal_analysis, agent_status, etc.
    query_complexity TEXT DEFAULT 'simple', -- simple, moderate, complex, multi_faceted

    -- Performance metrics
    response_time_ms INTEGER,           -- Time from query to final response
    first_token_time_ms INTEGER,        -- Time to first streaming token
    total_tokens INTEGER,               -- Approximate token count in response

    -- Tool usage tracking
    tools_invoked TEXT[],               -- Array of tool names used
    tools_succeeded TEXT[],             -- Tools that completed successfully
    tools_failed TEXT[],                -- Tools that failed

    -- Agent routing
    primary_agent TEXT,                 -- Main agent that handled the query
    agents_consulted TEXT[],            -- All agents consulted during execution
    orchestrator_used BOOLEAN DEFAULT FALSE,
    tool_composer_used BOOLEAN DEFAULT FALSE,

    -- Context usage
    rag_queries INTEGER DEFAULT 0,      -- Number of RAG retrievals
    rag_documents_retrieved INTEGER DEFAULT 0,
    memory_context_loaded BOOLEAN DEFAULT FALSE,
    episodic_memory_saved BOOLEAN DEFAULT FALSE,

    -- Error tracking
    error_occurred BOOLEAN DEFAULT FALSE,
    error_type TEXT,                    -- Timeout, validation, agent_error, etc.
    error_message TEXT,

    -- User engagement (set after interaction)
    user_satisfied BOOLEAN,             -- Derived from feedback if provided
    feedback_received BOOLEAN DEFAULT FALSE,

    -- Timestamps
    query_received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    response_completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata for extensibility
    metadata JSONB DEFAULT '{}'::jsonb
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Session-based queries
CREATE INDEX IF NOT EXISTS idx_chatbot_analytics_session
    ON public.chatbot_analytics(session_id);

-- Time-based analytics (most recent first)
CREATE INDEX IF NOT EXISTS idx_chatbot_analytics_created
    ON public.chatbot_analytics(created_at DESC);

-- Query type distribution analysis
CREATE INDEX IF NOT EXISTS idx_chatbot_analytics_query_type
    ON public.chatbot_analytics(query_type);

-- Performance analysis (slow queries)
CREATE INDEX IF NOT EXISTS idx_chatbot_analytics_response_time
    ON public.chatbot_analytics(response_time_ms DESC)
    WHERE response_time_ms IS NOT NULL;

-- Agent performance tracking
CREATE INDEX IF NOT EXISTS idx_chatbot_analytics_primary_agent
    ON public.chatbot_analytics(primary_agent)
    WHERE primary_agent IS NOT NULL;

-- Error analysis
CREATE INDEX IF NOT EXISTS idx_chatbot_analytics_errors
    ON public.chatbot_analytics(error_type)
    WHERE error_occurred = TRUE;

-- Tool usage analysis (GIN for array)
CREATE INDEX IF NOT EXISTS idx_chatbot_analytics_tools
    ON public.chatbot_analytics USING GIN(tools_invoked);

-- Combined index for dashboard queries
CREATE INDEX IF NOT EXISTS idx_chatbot_analytics_dashboard
    ON public.chatbot_analytics(created_at DESC, query_type, error_occurred);

-- =============================================================================
-- ROW LEVEL SECURITY
-- =============================================================================

-- Enable RLS
ALTER TABLE public.chatbot_analytics ENABLE ROW LEVEL SECURITY;

-- Service role has full access (for analytics aggregation)
CREATE POLICY chatbot_analytics_service_all ON public.chatbot_analytics
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Anonymous/authenticated can insert their own analytics
CREATE POLICY chatbot_analytics_insert_own ON public.chatbot_analytics
    FOR INSERT
    WITH CHECK (true);

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to get usage summary by time period
CREATE OR REPLACE FUNCTION public.get_chatbot_usage_summary(
    p_start_date TIMESTAMPTZ DEFAULT NOW() - INTERVAL '7 days',
    p_end_date TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE (
    total_queries BIGINT,
    unique_sessions BIGINT,
    avg_response_time_ms NUMERIC,
    p95_response_time_ms NUMERIC,
    error_rate NUMERIC,
    query_type_distribution JSONB,
    top_tools_used JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH stats AS (
        SELECT
            COUNT(*) as total,
            COUNT(DISTINCT session_id) as sessions,
            AVG(response_time_ms) as avg_time,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_time,
            AVG(CASE WHEN error_occurred THEN 1 ELSE 0 END) * 100 as err_rate
        FROM public.chatbot_analytics
        WHERE created_at BETWEEN p_start_date AND p_end_date
    ),
    query_types AS (
        SELECT jsonb_object_agg(query_type, cnt) as dist
        FROM (
            SELECT query_type, COUNT(*) as cnt
            FROM public.chatbot_analytics
            WHERE created_at BETWEEN p_start_date AND p_end_date
            GROUP BY query_type
        ) sub
    ),
    tools AS (
        SELECT jsonb_object_agg(tool_name, usage_count) as top_tools
        FROM (
            SELECT unnest(tools_invoked) as tool_name, COUNT(*) as usage_count
            FROM public.chatbot_analytics
            WHERE created_at BETWEEN p_start_date AND p_end_date
              AND tools_invoked IS NOT NULL
            GROUP BY unnest(tools_invoked)
            ORDER BY usage_count DESC
            LIMIT 10
        ) sub
    )
    SELECT
        s.total,
        s.sessions,
        ROUND(s.avg_time, 2),
        ROUND(s.p95_time, 2),
        ROUND(s.err_rate, 2),
        COALESCE(q.dist, '{}'::jsonb),
        COALESCE(t.top_tools, '{}'::jsonb)
    FROM stats s
    CROSS JOIN query_types q
    CROSS JOIN tools t;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get agent performance metrics
CREATE OR REPLACE FUNCTION public.get_agent_performance_metrics(
    p_agent_name TEXT DEFAULT NULL,
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    agent_name TEXT,
    total_queries BIGINT,
    avg_response_time_ms NUMERIC,
    p95_response_time_ms NUMERIC,
    error_rate NUMERIC,
    success_rate NUMERIC,
    avg_tools_per_query NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        a.primary_agent as agent_name,
        COUNT(*) as total_queries,
        ROUND(AVG(a.response_time_ms), 2) as avg_response_time_ms,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY a.response_time_ms), 2) as p95_response_time_ms,
        ROUND(AVG(CASE WHEN a.error_occurred THEN 1 ELSE 0 END) * 100, 2) as error_rate,
        ROUND(AVG(CASE WHEN NOT a.error_occurred THEN 1 ELSE 0 END) * 100, 2) as success_rate,
        ROUND(AVG(COALESCE(array_length(a.tools_invoked, 1), 0)), 2) as avg_tools_per_query
    FROM public.chatbot_analytics a
    WHERE a.created_at >= NOW() - (p_days || ' days')::INTERVAL
      AND a.primary_agent IS NOT NULL
      AND (p_agent_name IS NULL OR a.primary_agent = p_agent_name)
    GROUP BY a.primary_agent
    ORDER BY total_queries DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get hourly usage pattern
CREATE OR REPLACE FUNCTION public.get_hourly_usage_pattern(
    p_days INTEGER DEFAULT 7
)
RETURNS TABLE (
    hour_of_day INTEGER,
    query_count BIGINT,
    avg_response_time_ms NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        EXTRACT(HOUR FROM a.created_at)::INTEGER as hour_of_day,
        COUNT(*) as query_count,
        ROUND(AVG(a.response_time_ms), 2) as avg_response_time_ms
    FROM public.chatbot_analytics a
    WHERE a.created_at >= NOW() - (p_days || ' days')::INTERVAL
    GROUP BY EXTRACT(HOUR FROM a.created_at)
    ORDER BY hour_of_day;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to record analytics entry (helper for Python backend)
CREATE OR REPLACE FUNCTION public.record_chatbot_analytics(
    p_session_id VARCHAR,
    p_query_type TEXT,
    p_response_time_ms INTEGER DEFAULT NULL,
    p_tools_invoked TEXT[] DEFAULT NULL,
    p_primary_agent TEXT DEFAULT NULL,
    p_error_occurred BOOLEAN DEFAULT FALSE,
    p_error_type TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS BIGINT AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO public.chatbot_analytics (
        session_id,
        query_type,
        response_time_ms,
        tools_invoked,
        primary_agent,
        error_occurred,
        error_type,
        metadata,
        response_completed_at
    ) VALUES (
        p_session_id,
        p_query_type,
        p_response_time_ms,
        p_tools_invoked,
        p_primary_agent,
        p_error_occurred,
        p_error_type,
        p_metadata,
        CASE WHEN p_response_time_ms IS NOT NULL THEN NOW() ELSE NULL END
    )
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE public.chatbot_analytics IS
    'Usage analytics for the E2I chatbot - tracks queries, performance, and tool usage';

COMMENT ON COLUMN public.chatbot_analytics.query_type IS
    'Classification of the query: kpi_inquiry, causal_analysis, agent_status, etc.';

COMMENT ON COLUMN public.chatbot_analytics.response_time_ms IS
    'Total time from query received to response completion in milliseconds';

COMMENT ON COLUMN public.chatbot_analytics.tools_invoked IS
    'Array of tool names that were invoked during query processing';

COMMENT ON COLUMN public.chatbot_analytics.primary_agent IS
    'The main agent responsible for handling the query';

COMMENT ON FUNCTION public.get_chatbot_usage_summary IS
    'Get aggregated usage statistics for a date range';

COMMENT ON FUNCTION public.get_agent_performance_metrics IS
    'Get performance metrics per agent for optimization';

COMMENT ON FUNCTION public.get_hourly_usage_pattern IS
    'Get usage distribution by hour of day for capacity planning';

COMMENT ON FUNCTION public.record_chatbot_analytics IS
    'Helper function to record analytics entries from the Python backend';
