-- =============================================================================
-- E2I Causal Analytics - Chatbot Training Signals
-- =============================================================================
-- Version: 1.0.0
-- Created: 2026-01-14
-- Description: Store unified training signals from chatbot sessions for
--              DSPy prompt optimization via the feedback_learner agent.
--
-- Features:
--   - Captures signals from all DSPy phases (intent, routing, RAG, synthesis)
--   - Stores computed reward scores for optimization
--   - Enables the feedback_learner to train DSPy modules
--   - Supports GEPA (Generative Evolutionary Prompting with AI) optimization
-- =============================================================================

-- =============================================================================
-- TABLES
-- =============================================================================

-- Unified training signals for the chatbot
CREATE TABLE IF NOT EXISTS public.chatbot_training_signals (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,

    -- Session identification
    session_id VARCHAR NOT NULL,
    thread_id VARCHAR NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,

    -- Query context
    query TEXT NOT NULL,
    brand_context TEXT DEFAULT '',
    region_context TEXT DEFAULT '',

    -- Phase 3: Intent Classification
    predicted_intent TEXT DEFAULT '',
    intent_confidence FLOAT DEFAULT 0.0,
    intent_method TEXT DEFAULT '',              -- 'dspy' or 'hardcoded'
    intent_reasoning TEXT DEFAULT '',

    -- Phase 4: Agent Routing
    predicted_agent TEXT DEFAULT '',
    secondary_agents TEXT[] DEFAULT '{}',
    routing_confidence FLOAT DEFAULT 0.0,
    routing_method TEXT DEFAULT '',             -- 'dspy' or 'hardcoded'
    routing_rationale TEXT DEFAULT '',

    -- Phase 5: Cognitive RAG
    rewritten_query TEXT DEFAULT '',
    search_keywords TEXT[] DEFAULT '{}',
    graph_entities TEXT[] DEFAULT '{}',
    evidence_count INTEGER DEFAULT 0,
    hop_count INTEGER DEFAULT 0,
    avg_relevance_score FLOAT DEFAULT 0.0,
    rag_method TEXT DEFAULT '',                 -- 'cognitive' or 'basic'

    -- Phase 6: Evidence Synthesis
    response_length INTEGER DEFAULT 0,
    synthesis_confidence TEXT DEFAULT '',       -- 'high', 'moderate', 'low'
    citations_count INTEGER DEFAULT 0,
    synthesis_method TEXT DEFAULT '',           -- 'dspy' or 'hardcoded'
    follow_up_count INTEGER DEFAULT 0,

    -- User feedback (populated asynchronously)
    user_rating FLOAT,                          -- 1.0 to 5.0
    was_helpful BOOLEAN,
    user_followed_up BOOLEAN,
    had_hallucination BOOLEAN,

    -- Timing metrics
    total_duration_ms FLOAT,
    intent_duration_ms FLOAT,
    routing_duration_ms FLOAT,
    rag_duration_ms FLOAT,
    synthesis_duration_ms FLOAT,

    -- Computed reward scores (0.0 to 1.0)
    reward_accuracy FLOAT DEFAULT 0.0,
    reward_efficiency FLOAT DEFAULT 0.0,
    reward_satisfaction FLOAT DEFAULT 0.0,
    reward_overall FLOAT DEFAULT 0.0,

    -- Training status
    used_for_training BOOLEAN DEFAULT FALSE,
    training_phase TEXT,                        -- Which phase used this signal
    training_batch_id VARCHAR,                  -- Batch ID when used

    -- Timestamps
    session_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata for extensibility
    metadata JSONB DEFAULT '{}'::jsonb
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Primary lookups
CREATE INDEX IF NOT EXISTS idx_chatbot_training_signals_session
    ON public.chatbot_training_signals(session_id);

CREATE INDEX IF NOT EXISTS idx_chatbot_training_signals_user
    ON public.chatbot_training_signals(user_id)
    WHERE user_id IS NOT NULL;

-- Time-based queries (most recent first)
CREATE INDEX IF NOT EXISTS idx_chatbot_training_signals_created
    ON public.chatbot_training_signals(created_at DESC);

-- High-quality signals for training
CREATE INDEX IF NOT EXISTS idx_chatbot_training_signals_quality
    ON public.chatbot_training_signals(reward_overall DESC)
    WHERE reward_overall >= 0.6;

-- Unused signals for training batches
CREATE INDEX IF NOT EXISTS idx_chatbot_training_signals_unused
    ON public.chatbot_training_signals(created_at DESC)
    WHERE used_for_training = FALSE;

-- DSPy method tracking
CREATE INDEX IF NOT EXISTS idx_chatbot_training_signals_dspy_intent
    ON public.chatbot_training_signals(intent_method)
    WHERE intent_method IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chatbot_training_signals_dspy_synthesis
    ON public.chatbot_training_signals(synthesis_method)
    WHERE synthesis_method IS NOT NULL;

-- Intent classification analysis
CREATE INDEX IF NOT EXISTS idx_chatbot_training_signals_intent
    ON public.chatbot_training_signals(predicted_intent);

-- Agent routing analysis
CREATE INDEX IF NOT EXISTS idx_chatbot_training_signals_agent
    ON public.chatbot_training_signals(predicted_agent)
    WHERE predicted_agent IS NOT NULL;

-- Combined index for training queries
CREATE INDEX IF NOT EXISTS idx_chatbot_training_signals_training
    ON public.chatbot_training_signals(used_for_training, reward_overall DESC, created_at DESC);

-- =============================================================================
-- ROW LEVEL SECURITY
-- =============================================================================

-- Enable RLS
ALTER TABLE public.chatbot_training_signals ENABLE ROW LEVEL SECURITY;

-- Service role has full access (for training and analytics)
CREATE POLICY chatbot_training_signals_service_all ON public.chatbot_training_signals
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Anonymous/authenticated can insert signals
CREATE POLICY chatbot_training_signals_insert ON public.chatbot_training_signals
    FOR INSERT
    WITH CHECK (true);

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to insert a training signal
CREATE OR REPLACE FUNCTION public.insert_training_signal(
    p_session_id VARCHAR,
    p_thread_id VARCHAR,
    p_user_id UUID DEFAULT NULL,
    p_query TEXT DEFAULT '',
    p_brand_context TEXT DEFAULT '',
    p_region_context TEXT DEFAULT '',
    p_predicted_intent TEXT DEFAULT '',
    p_intent_confidence FLOAT DEFAULT 0.0,
    p_intent_method TEXT DEFAULT '',
    p_intent_reasoning TEXT DEFAULT '',
    p_predicted_agent TEXT DEFAULT '',
    p_secondary_agents TEXT[] DEFAULT '{}',
    p_routing_confidence FLOAT DEFAULT 0.0,
    p_routing_method TEXT DEFAULT '',
    p_routing_rationale TEXT DEFAULT '',
    p_rewritten_query TEXT DEFAULT '',
    p_search_keywords TEXT[] DEFAULT '{}',
    p_graph_entities TEXT[] DEFAULT '{}',
    p_evidence_count INTEGER DEFAULT 0,
    p_hop_count INTEGER DEFAULT 0,
    p_avg_relevance_score FLOAT DEFAULT 0.0,
    p_rag_method TEXT DEFAULT '',
    p_response_length INTEGER DEFAULT 0,
    p_synthesis_confidence TEXT DEFAULT '',
    p_citations_count INTEGER DEFAULT 0,
    p_synthesis_method TEXT DEFAULT '',
    p_follow_up_count INTEGER DEFAULT 0,
    p_total_duration_ms FLOAT DEFAULT NULL,
    p_reward_accuracy FLOAT DEFAULT 0.0,
    p_reward_efficiency FLOAT DEFAULT 0.0,
    p_reward_satisfaction FLOAT DEFAULT 0.0,
    p_reward_overall FLOAT DEFAULT 0.0,
    p_session_timestamp TIMESTAMPTZ DEFAULT NOW(),
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS BIGINT AS $$
DECLARE
    v_id BIGINT;
BEGIN
    INSERT INTO public.chatbot_training_signals (
        session_id,
        thread_id,
        user_id,
        query,
        brand_context,
        region_context,
        predicted_intent,
        intent_confidence,
        intent_method,
        intent_reasoning,
        predicted_agent,
        secondary_agents,
        routing_confidence,
        routing_method,
        routing_rationale,
        rewritten_query,
        search_keywords,
        graph_entities,
        evidence_count,
        hop_count,
        avg_relevance_score,
        rag_method,
        response_length,
        synthesis_confidence,
        citations_count,
        synthesis_method,
        follow_up_count,
        total_duration_ms,
        reward_accuracy,
        reward_efficiency,
        reward_satisfaction,
        reward_overall,
        session_timestamp,
        metadata
    ) VALUES (
        p_session_id,
        p_thread_id,
        p_user_id,
        p_query,
        p_brand_context,
        p_region_context,
        p_predicted_intent,
        p_intent_confidence,
        p_intent_method,
        p_intent_reasoning,
        p_predicted_agent,
        p_secondary_agents,
        p_routing_confidence,
        p_routing_method,
        p_routing_rationale,
        p_rewritten_query,
        p_search_keywords,
        p_graph_entities,
        p_evidence_count,
        p_hop_count,
        p_avg_relevance_score,
        p_rag_method,
        p_response_length,
        p_synthesis_confidence,
        p_citations_count,
        p_synthesis_method,
        p_follow_up_count,
        p_total_duration_ms,
        p_reward_accuracy,
        p_reward_efficiency,
        p_reward_satisfaction,
        p_reward_overall,
        p_session_timestamp,
        p_metadata
    )
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get high-quality signals for training
CREATE OR REPLACE FUNCTION public.get_training_signals(
    p_phase TEXT DEFAULT NULL,
    p_min_reward FLOAT DEFAULT 0.5,
    p_limit INTEGER DEFAULT 100,
    p_exclude_used BOOLEAN DEFAULT TRUE
)
RETURNS TABLE (
    id BIGINT,
    session_id VARCHAR,
    query TEXT,
    brand_context TEXT,
    region_context TEXT,
    predicted_intent TEXT,
    intent_confidence FLOAT,
    intent_method TEXT,
    predicted_agent TEXT,
    routing_confidence FLOAT,
    routing_method TEXT,
    rewritten_query TEXT,
    rag_method TEXT,
    synthesis_confidence TEXT,
    synthesis_method TEXT,
    reward_accuracy FLOAT,
    reward_efficiency FLOAT,
    reward_satisfaction FLOAT,
    reward_overall FLOAT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.id,
        s.session_id,
        s.query,
        s.brand_context,
        s.region_context,
        s.predicted_intent,
        s.intent_confidence,
        s.intent_method,
        s.predicted_agent,
        s.routing_confidence,
        s.routing_method,
        s.rewritten_query,
        s.rag_method,
        s.synthesis_confidence,
        s.synthesis_method,
        s.reward_accuracy,
        s.reward_efficiency,
        s.reward_satisfaction,
        s.reward_overall,
        s.created_at
    FROM public.chatbot_training_signals s
    WHERE s.reward_overall >= p_min_reward
      AND (NOT p_exclude_used OR s.used_for_training = FALSE)
      AND (
          p_phase IS NULL
          OR (p_phase = 'intent' AND s.intent_method IS NOT NULL AND s.intent_method != '')
          OR (p_phase = 'routing' AND s.routing_method IS NOT NULL AND s.routing_method != '')
          OR (p_phase = 'rag' AND s.rag_method IS NOT NULL AND s.rag_method != '')
          OR (p_phase = 'synthesis' AND s.synthesis_method IS NOT NULL AND s.synthesis_method != '')
      )
    ORDER BY s.reward_overall DESC, s.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to mark signals as used for training
CREATE OR REPLACE FUNCTION public.mark_signals_used(
    p_signal_ids BIGINT[],
    p_training_phase TEXT,
    p_batch_id VARCHAR DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    v_count INTEGER;
BEGIN
    UPDATE public.chatbot_training_signals
    SET
        used_for_training = TRUE,
        training_phase = p_training_phase,
        training_batch_id = COALESCE(p_batch_id, 'batch_' || to_char(NOW(), 'YYYYMMDD_HH24MISS')),
        updated_at = NOW()
    WHERE id = ANY(p_signal_ids);

    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get training signal statistics
CREATE OR REPLACE FUNCTION public.get_training_signal_stats(
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    total_signals BIGINT,
    high_quality_signals BIGINT,
    used_for_training BIGINT,
    avg_reward_overall FLOAT,
    dspy_intent_rate FLOAT,
    dspy_routing_rate FLOAT,
    dspy_rag_rate FLOAT,
    dspy_synthesis_rate FLOAT,
    intent_distribution JSONB,
    agent_distribution JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH stats AS (
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE s.reward_overall >= 0.6) as high_quality,
            COUNT(*) FILTER (WHERE s.used_for_training) as used,
            AVG(s.reward_overall) as avg_reward,
            AVG(CASE WHEN s.intent_method = 'dspy' THEN 1 ELSE 0 END) * 100 as dspy_intent,
            AVG(CASE WHEN s.routing_method = 'dspy' THEN 1 ELSE 0 END) * 100 as dspy_routing,
            AVG(CASE WHEN s.rag_method = 'cognitive' THEN 1 ELSE 0 END) * 100 as dspy_rag,
            AVG(CASE WHEN s.synthesis_method = 'dspy' THEN 1 ELSE 0 END) * 100 as dspy_synth
        FROM public.chatbot_training_signals s
        WHERE s.created_at >= NOW() - (p_days || ' days')::INTERVAL
    ),
    intents AS (
        SELECT jsonb_object_agg(predicted_intent, cnt) as dist
        FROM (
            SELECT predicted_intent, COUNT(*) as cnt
            FROM public.chatbot_training_signals
            WHERE created_at >= NOW() - (p_days || ' days')::INTERVAL
              AND predicted_intent IS NOT NULL AND predicted_intent != ''
            GROUP BY predicted_intent
        ) sub
    ),
    agents AS (
        SELECT jsonb_object_agg(predicted_agent, cnt) as dist
        FROM (
            SELECT predicted_agent, COUNT(*) as cnt
            FROM public.chatbot_training_signals
            WHERE created_at >= NOW() - (p_days || ' days')::INTERVAL
              AND predicted_agent IS NOT NULL AND predicted_agent != ''
            GROUP BY predicted_agent
        ) sub
    )
    SELECT
        s.total,
        s.high_quality,
        s.used,
        ROUND(s.avg_reward::NUMERIC, 4)::FLOAT,
        ROUND(s.dspy_intent::NUMERIC, 2)::FLOAT,
        ROUND(s.dspy_routing::NUMERIC, 2)::FLOAT,
        ROUND(s.dspy_rag::NUMERIC, 2)::FLOAT,
        ROUND(s.dspy_synth::NUMERIC, 2)::FLOAT,
        COALESCE(i.dist, '{}'::jsonb),
        COALESCE(a.dist, '{}'::jsonb)
    FROM stats s
    CROSS JOIN intents i
    CROSS JOIN agents a;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to update feedback for a signal
CREATE OR REPLACE FUNCTION public.update_signal_feedback(
    p_session_id VARCHAR,
    p_user_rating FLOAT DEFAULT NULL,
    p_was_helpful BOOLEAN DEFAULT NULL,
    p_user_followed_up BOOLEAN DEFAULT NULL,
    p_had_hallucination BOOLEAN DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    v_count INTEGER;
    v_new_satisfaction FLOAT;
BEGIN
    -- Calculate new satisfaction reward based on feedback
    v_new_satisfaction := 0.3; -- Base score

    IF p_user_rating IS NOT NULL THEN
        v_new_satisfaction := v_new_satisfaction + (p_user_rating / 5.0) * 0.4;
    END IF;

    IF p_was_helpful = TRUE THEN
        v_new_satisfaction := v_new_satisfaction + 0.2;
    ELSIF p_was_helpful = FALSE THEN
        v_new_satisfaction := v_new_satisfaction - 0.1;
    END IF;

    IF p_had_hallucination = TRUE THEN
        v_new_satisfaction := v_new_satisfaction - 0.3;
    END IF;

    -- Clamp to 0.0 - 1.0
    v_new_satisfaction := GREATEST(0.0, LEAST(1.0, v_new_satisfaction));

    UPDATE public.chatbot_training_signals
    SET
        user_rating = COALESCE(p_user_rating, user_rating),
        was_helpful = COALESCE(p_was_helpful, was_helpful),
        user_followed_up = COALESCE(p_user_followed_up, user_followed_up),
        had_hallucination = COALESCE(p_had_hallucination, had_hallucination),
        reward_satisfaction = v_new_satisfaction,
        reward_overall = (reward_accuracy * 0.4 + reward_efficiency * 0.15 + v_new_satisfaction * 0.45),
        updated_at = NOW()
    WHERE session_id = p_session_id
      AND used_for_training = FALSE;  -- Don't update already-used signals

    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count > 0;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE public.chatbot_training_signals IS
    'Unified training signals from chatbot sessions for DSPy prompt optimization';

COMMENT ON COLUMN public.chatbot_training_signals.predicted_intent IS
    'Intent classification: kpi_query, causal_analysis, search, etc.';

COMMENT ON COLUMN public.chatbot_training_signals.intent_method IS
    'Method used for classification: dspy (ML) or hardcoded (fallback)';

COMMENT ON COLUMN public.chatbot_training_signals.reward_overall IS
    'Combined reward score (0.0-1.0) for training signal quality';

COMMENT ON COLUMN public.chatbot_training_signals.used_for_training IS
    'Whether this signal has been consumed by the feedback_learner';

COMMENT ON FUNCTION public.insert_training_signal IS
    'Insert a new training signal from a chatbot session';

COMMENT ON FUNCTION public.get_training_signals IS
    'Retrieve high-quality signals for DSPy training by phase';

COMMENT ON FUNCTION public.mark_signals_used IS
    'Mark signals as consumed by the feedback_learner training process';

COMMENT ON FUNCTION public.get_training_signal_stats IS
    'Get statistics on training signal collection and usage';

COMMENT ON FUNCTION public.update_signal_feedback IS
    'Update a signal with user feedback and recalculate rewards';
