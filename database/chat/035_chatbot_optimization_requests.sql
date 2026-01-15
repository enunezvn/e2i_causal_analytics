-- =============================================================================
-- E2I Causal Analytics - Chatbot Optimization Requests
-- =============================================================================
-- Version: 1.0.0
-- Created: 2026-01-14
-- Description: Store optimization requests queued by the ChatbotOptimizer for
--              the feedback_learner agent to process.
--
-- Features:
--   - Queue optimization requests for DSPy modules
--   - Track request status and processing results
--   - Support priority-based processing
--   - Link to GEPA optimization runs when executed
-- =============================================================================

-- =============================================================================
-- TABLES
-- =============================================================================

-- Chatbot module optimization requests queue
CREATE TABLE IF NOT EXISTS public.chatbot_optimization_requests (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,

    -- Request identification
    request_id VARCHAR(64) NOT NULL UNIQUE,

    -- Module to optimize
    module_name VARCHAR(50) NOT NULL,  -- 'intent_classifier', 'agent_router', 'query_rewriter', 'synthesizer'

    -- Configuration
    signal_count INTEGER NOT NULL DEFAULT 0,
    min_reward FLOAT NOT NULL DEFAULT 0.5,
    budget VARCHAR(20) NOT NULL DEFAULT 'light',  -- 'light', 'medium', 'heavy'
    priority INTEGER NOT NULL DEFAULT 1,  -- 1=low, 2=medium, 3=high

    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- 'pending', 'processing', 'completed', 'failed', 'cancelled'

    -- Processing results
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    baseline_score FLOAT,
    optimized_score FLOAT,
    improvement_percent FLOAT,
    error_message TEXT,

    -- Link to GEPA optimization run (if executed)
    optimization_run_id UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata for extensibility
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Constraints
    CONSTRAINT valid_module_name CHECK (
        module_name IN ('intent_classifier', 'agent_router', 'query_rewriter', 'synthesizer')
    ),
    CONSTRAINT valid_budget CHECK (
        budget IN ('light', 'medium', 'heavy')
    ),
    CONSTRAINT valid_status CHECK (
        status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')
    ),
    CONSTRAINT valid_priority CHECK (
        priority >= 1 AND priority <= 3
    )
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Primary lookups
CREATE INDEX IF NOT EXISTS idx_chatbot_opt_requests_request_id
    ON public.chatbot_optimization_requests(request_id);

-- Status-based queries (most common)
CREATE INDEX IF NOT EXISTS idx_chatbot_opt_requests_status
    ON public.chatbot_optimization_requests(status);

-- Pending requests ordered by priority and creation time
CREATE INDEX IF NOT EXISTS idx_chatbot_opt_requests_pending
    ON public.chatbot_optimization_requests(priority DESC, created_at ASC)
    WHERE status = 'pending';

-- Module-specific queries
CREATE INDEX IF NOT EXISTS idx_chatbot_opt_requests_module
    ON public.chatbot_optimization_requests(module_name, status);

-- Time-based queries
CREATE INDEX IF NOT EXISTS idx_chatbot_opt_requests_created
    ON public.chatbot_optimization_requests(created_at DESC);

-- =============================================================================
-- ROW LEVEL SECURITY
-- =============================================================================

-- Enable RLS
ALTER TABLE public.chatbot_optimization_requests ENABLE ROW LEVEL SECURITY;

-- Service role has full access
CREATE POLICY chatbot_opt_requests_service_all ON public.chatbot_optimization_requests
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Authenticated users can view optimization requests
CREATE POLICY chatbot_opt_requests_select ON public.chatbot_optimization_requests
    FOR SELECT
    TO authenticated
    USING (true);

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to insert an optimization request
CREATE OR REPLACE FUNCTION public.insert_optimization_request(
    p_request_id VARCHAR,
    p_module_name VARCHAR,
    p_signal_count INTEGER DEFAULT 0,
    p_min_reward FLOAT DEFAULT 0.5,
    p_budget VARCHAR DEFAULT 'light',
    p_priority INTEGER DEFAULT 1,
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS BIGINT AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO public.chatbot_optimization_requests (
        request_id,
        module_name,
        signal_count,
        min_reward,
        budget,
        priority,
        status,
        metadata
    ) VALUES (
        p_request_id,
        p_module_name,
        p_signal_count,
        p_min_reward,
        p_budget,
        p_priority,
        'pending',
        p_metadata
    )
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get next pending request (priority-based)
CREATE OR REPLACE FUNCTION public.get_next_optimization_request(
    p_module_name VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    request_id VARCHAR,
    module_name VARCHAR,
    signal_count INTEGER,
    min_reward FLOAT,
    budget VARCHAR,
    priority INTEGER,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.id,
        r.request_id,
        r.module_name,
        r.signal_count,
        r.min_reward,
        r.budget,
        r.priority,
        r.created_at
    FROM public.chatbot_optimization_requests r
    WHERE r.status = 'pending'
      AND (p_module_name IS NULL OR r.module_name = p_module_name)
    ORDER BY r.priority DESC, r.created_at ASC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to update request status
CREATE OR REPLACE FUNCTION public.update_optimization_request_status(
    p_request_id VARCHAR,
    p_status VARCHAR,
    p_baseline_score FLOAT DEFAULT NULL,
    p_optimized_score FLOAT DEFAULT NULL,
    p_error_message TEXT DEFAULT NULL,
    p_optimization_run_id UUID DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    v_count INTEGER;
    v_improvement FLOAT;
BEGIN
    -- Calculate improvement if both scores provided
    IF p_baseline_score IS NOT NULL AND p_optimized_score IS NOT NULL AND p_baseline_score > 0 THEN
        v_improvement := ((p_optimized_score - p_baseline_score) / p_baseline_score) * 100;
    END IF;

    UPDATE public.chatbot_optimization_requests
    SET
        status = p_status,
        started_at = CASE
            WHEN p_status = 'processing' AND started_at IS NULL THEN NOW()
            ELSE started_at
        END,
        completed_at = CASE
            WHEN p_status IN ('completed', 'failed', 'cancelled') THEN NOW()
            ELSE completed_at
        END,
        baseline_score = COALESCE(p_baseline_score, baseline_score),
        optimized_score = COALESCE(p_optimized_score, optimized_score),
        improvement_percent = COALESCE(v_improvement, improvement_percent),
        error_message = COALESCE(p_error_message, error_message),
        optimization_run_id = COALESCE(p_optimization_run_id, optimization_run_id),
        updated_at = NOW()
    WHERE request_id = p_request_id;

    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count > 0;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get optimization request stats
CREATE OR REPLACE FUNCTION public.get_optimization_request_stats(
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    total_requests BIGINT,
    pending_requests BIGINT,
    completed_requests BIGINT,
    failed_requests BIGINT,
    avg_improvement FLOAT,
    requests_by_module JSONB,
    requests_by_status JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH stats AS (
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE r.status = 'pending') as pending,
            COUNT(*) FILTER (WHERE r.status = 'completed') as completed,
            COUNT(*) FILTER (WHERE r.status = 'failed') as failed,
            AVG(r.improvement_percent) FILTER (WHERE r.status = 'completed') as avg_improve
        FROM public.chatbot_optimization_requests r
        WHERE r.created_at >= NOW() - (p_days || ' days')::INTERVAL
    ),
    by_module AS (
        SELECT jsonb_object_agg(module_name, cnt) as dist
        FROM (
            SELECT module_name, COUNT(*) as cnt
            FROM public.chatbot_optimization_requests
            WHERE created_at >= NOW() - (p_days || ' days')::INTERVAL
            GROUP BY module_name
        ) sub
    ),
    by_status AS (
        SELECT jsonb_object_agg(status, cnt) as dist
        FROM (
            SELECT status, COUNT(*) as cnt
            FROM public.chatbot_optimization_requests
            WHERE created_at >= NOW() - (p_days || ' days')::INTERVAL
            GROUP BY status
        ) sub
    )
    SELECT
        s.total,
        s.pending,
        s.completed,
        s.failed,
        ROUND(s.avg_improve::NUMERIC, 2)::FLOAT,
        COALESCE(m.dist, '{}'::jsonb),
        COALESCE(st.dist, '{}'::jsonb)
    FROM stats s
    CROSS JOIN by_module m
    CROSS JOIN by_status st;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to cancel stale pending requests
CREATE OR REPLACE FUNCTION public.cancel_stale_optimization_requests(
    p_max_age_hours INTEGER DEFAULT 24
)
RETURNS INTEGER AS $$
DECLARE
    v_count INTEGER;
BEGIN
    UPDATE public.chatbot_optimization_requests
    SET
        status = 'cancelled',
        error_message = 'Cancelled: request exceeded maximum pending age',
        completed_at = NOW(),
        updated_at = NOW()
    WHERE status = 'pending'
      AND created_at < NOW() - (p_max_age_hours || ' hours')::INTERVAL;

    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- TRIGGER FOR UPDATED_AT
-- =============================================================================

CREATE OR REPLACE FUNCTION public.trigger_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_updated_at ON public.chatbot_optimization_requests;
CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON public.chatbot_optimization_requests
    FOR EACH ROW
    EXECUTE FUNCTION public.trigger_set_updated_at();

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE public.chatbot_optimization_requests IS
    'Queue of optimization requests for chatbot DSPy modules';

COMMENT ON COLUMN public.chatbot_optimization_requests.module_name IS
    'DSPy module to optimize: intent_classifier, agent_router, query_rewriter, synthesizer';

COMMENT ON COLUMN public.chatbot_optimization_requests.budget IS
    'GEPA optimization budget: light (~500 calls), medium (~2000 calls), heavy (~4000+ calls)';

COMMENT ON COLUMN public.chatbot_optimization_requests.priority IS
    'Processing priority: 1=low, 2=medium, 3=high';

COMMENT ON COLUMN public.chatbot_optimization_requests.optimization_run_id IS
    'Link to prompt_optimization_runs table if this request was executed';

COMMENT ON FUNCTION public.insert_optimization_request IS
    'Insert a new optimization request into the queue';

COMMENT ON FUNCTION public.get_next_optimization_request IS
    'Get the next pending request to process (priority-based)';

COMMENT ON FUNCTION public.update_optimization_request_status IS
    'Update request status and results';

COMMENT ON FUNCTION public.get_optimization_request_stats IS
    'Get statistics on optimization requests';

COMMENT ON FUNCTION public.cancel_stale_optimization_requests IS
    'Cancel pending requests older than specified hours';
