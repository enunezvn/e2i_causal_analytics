-- ============================================================================
-- E2I DSPy Training Signals Enhancement
-- Migration: 014_dspy_training_signals
-- Version: 4.2
-- Purpose: Comprehensive DSPy integration for MIPROv2 optimization
--
-- This migration adds:
-- 1. Agent-specific training signals with full quality metrics
-- 2. Cognitive context history for replay and analysis
-- 3. MIPROv2 optimization run tracking
-- 4. Enhanced learning signals with DSPy-specific fields
-- ============================================================================

-- ============================================================================
-- ENUM TYPES
-- ============================================================================

-- Agent signal source types (extends e2i_agent_name if not complete)
DO $$ BEGIN
    CREATE TYPE dspy_optimization_phase AS ENUM (
        'pattern_detection',
        'recommendation_generation',
        'knowledge_update',
        'learning_summary',
        'causal_impact',
        'gap_analysis',
        'experiment_design',
        'prediction_synthesis'
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- Optimization status
DO $$ BEGIN
    CREATE TYPE optimization_status AS ENUM (
        'pending',
        'running',
        'completed',
        'failed',
        'cancelled'
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;


-- ============================================================================
-- TABLE 1: AGENT TRAINING SIGNALS
-- Comprehensive training signal storage for all E2I agents
-- ============================================================================

CREATE TABLE IF NOT EXISTS dspy_agent_training_signals (
    -- Primary key
    signal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- === Signal Identity ===
    source_agent VARCHAR(50) NOT NULL,  -- feedback_learner, causal_impact, etc.
    batch_id VARCHAR(100),              -- Batch/job identifier

    -- === Input Context ===
    input_context JSONB NOT NULL DEFAULT '{}',
    -- Contains: query, entities, time_range, cognitive_context_id, etc.

    -- === Processing Outputs ===
    output JSONB NOT NULL DEFAULT '{}',
    -- Agent-specific: patterns, recommendations, predictions, etc.

    -- === Quality Metrics ===
    quality_metrics JSONB NOT NULL DEFAULT '{}',
    -- Agent-specific: accuracy, actionability, effectiveness, etc.

    -- Computed reward for MIPROv2 (cached for quick queries)
    reward FLOAT CHECK (reward BETWEEN 0 AND 1),

    -- === Latency Breakdown ===
    latency_breakdown JSONB NOT NULL DEFAULT '{}',
    -- Contains: collection_ms, analysis_ms, synthesis_ms, etc.
    total_latency_ms INTEGER,

    -- === LLM Usage ===
    model_used VARCHAR(100),
    llm_calls INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,

    -- === Cognitive Context Reference ===
    cognitive_context_id UUID,  -- FK to dspy_cognitive_context_history
    has_cognitive_context BOOLEAN DEFAULT FALSE,

    -- === Delayed Outcomes (Updated Later) ===
    metric_improvement_7d FLOAT,
    metric_improvement_30d FLOAT,
    user_satisfaction_delta FLOAT,
    downstream_impact JSONB,  -- How this affected other agents

    -- === Validation ===
    human_validated BOOLEAN DEFAULT FALSE,
    validation_timestamp TIMESTAMPTZ,
    validated_by VARCHAR(100),
    validation_notes TEXT,

    -- === DSPy Integration ===
    is_training_example BOOLEAN DEFAULT TRUE,
    excluded_from_training BOOLEAN DEFAULT FALSE,
    exclusion_reason TEXT,
    used_in_optimization_runs INTEGER[] DEFAULT ARRAY[]::INTEGER[],

    -- === Timestamps ===
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for agent training signals
CREATE INDEX IF NOT EXISTS idx_dspy_signals_agent ON dspy_agent_training_signals(source_agent);
CREATE INDEX IF NOT EXISTS idx_dspy_signals_batch ON dspy_agent_training_signals(batch_id);
CREATE INDEX IF NOT EXISTS idx_dspy_signals_reward ON dspy_agent_training_signals(reward DESC) WHERE reward IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_dspy_signals_created ON dspy_agent_training_signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dspy_signals_training ON dspy_agent_training_signals(is_training_example, excluded_from_training)
    WHERE is_training_example = TRUE AND excluded_from_training = FALSE;
CREATE INDEX IF NOT EXISTS idx_dspy_signals_validated ON dspy_agent_training_signals(human_validated) WHERE human_validated = TRUE;
CREATE INDEX IF NOT EXISTS idx_dspy_signals_context ON dspy_agent_training_signals(cognitive_context_id) WHERE cognitive_context_id IS NOT NULL;


-- ============================================================================
-- TABLE 2: COGNITIVE CONTEXT HISTORY
-- Stores enriched context from CognitiveRAG 4-phase cycle
-- ============================================================================

CREATE TABLE IF NOT EXISTS dspy_cognitive_context_history (
    -- Primary key
    context_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- === Source Cycle ===
    cycle_id UUID REFERENCES cognitive_cycles(cycle_id) ON DELETE SET NULL,
    session_id UUID,

    -- === Query Context ===
    original_query TEXT,
    query_embedding vector(1536),
    detected_intent VARCHAR(50),
    detected_entities JSONB DEFAULT '{}',

    -- === Synthesized Summary (Phase 1: Summarizer) ===
    synthesized_summary TEXT,
    compression_ratio FLOAT,

    -- === Retrieved Evidence (Phase 2: Investigator) ===
    historical_patterns JSONB DEFAULT '[]',
    optimization_examples JSONB DEFAULT '[]',
    agent_baselines JSONB DEFAULT '{}',
    prior_learnings JSONB DEFAULT '[]',
    correlation_insights JSONB DEFAULT '[]',

    -- === Evidence Quality ===
    evidence_confidence FLOAT CHECK (evidence_confidence BETWEEN 0 AND 1),
    total_hops INTEGER DEFAULT 0,
    retrieval_latency_ms INTEGER,

    -- === Memory Sources Used ===
    episodic_hits INTEGER DEFAULT 0,
    semantic_hits INTEGER DEFAULT 0,
    procedural_hits INTEGER DEFAULT 0,

    -- === Agents Using This Context ===
    consuming_agents TEXT[] DEFAULT ARRAY[]::TEXT[],

    -- === Timestamps ===
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for cognitive context history
CREATE INDEX IF NOT EXISTS idx_cognitive_context_cycle ON dspy_cognitive_context_history(cycle_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_context_session ON dspy_cognitive_context_history(session_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_context_created ON dspy_cognitive_context_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cognitive_context_embedding ON dspy_cognitive_context_history
    USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX IF NOT EXISTS idx_cognitive_context_confidence ON dspy_cognitive_context_history(evidence_confidence DESC);


-- ============================================================================
-- TABLE 3: MIPROv2 OPTIMIZATION RUNS
-- Tracks prompt optimization experiments
-- ============================================================================

CREATE TABLE IF NOT EXISTS dspy_optimization_runs (
    -- Primary key
    run_id SERIAL PRIMARY KEY,
    run_uuid UUID DEFAULT gen_random_uuid() UNIQUE,

    -- === Optimization Target ===
    target_agent VARCHAR(50) NOT NULL,
    optimization_phase dspy_optimization_phase NOT NULL,
    signature_name VARCHAR(100) NOT NULL,

    -- === Configuration ===
    config JSONB NOT NULL DEFAULT '{}',
    -- Contains: num_candidates, max_bootstrapped_demos, num_threads, budget

    -- === Training Data ===
    training_signal_ids UUID[] DEFAULT ARRAY[]::UUID[],
    training_examples_count INTEGER DEFAULT 0,
    validation_examples_count INTEGER DEFAULT 0,

    -- === Results ===
    status optimization_status DEFAULT 'pending',

    -- Before/after metrics
    baseline_metric FLOAT,
    optimized_metric FLOAT,
    improvement_pct FLOAT,

    -- Best prompt found
    best_prompt_template TEXT,
    best_few_shot_examples JSONB,

    -- === Execution Details ===
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    trials_completed INTEGER DEFAULT 0,

    -- Error handling
    error_message TEXT,
    error_traceback TEXT,

    -- === Deployment ===
    deployed BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMPTZ,
    rollback_available BOOLEAN DEFAULT FALSE,
    previous_prompt_template TEXT,

    -- === Impact Tracking ===
    post_deployment_metrics JSONB,
    -- Tracks: reward_avg, latency_avg, user_satisfaction over time

    -- === Timestamps ===
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for optimization runs
CREATE INDEX IF NOT EXISTS idx_optim_runs_agent ON dspy_optimization_runs(target_agent);
CREATE INDEX IF NOT EXISTS idx_optim_runs_phase ON dspy_optimization_runs(optimization_phase);
CREATE INDEX IF NOT EXISTS idx_optim_runs_status ON dspy_optimization_runs(status);
CREATE INDEX IF NOT EXISTS idx_optim_runs_created ON dspy_optimization_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_optim_runs_deployed ON dspy_optimization_runs(deployed, deployed_at DESC) WHERE deployed = TRUE;


-- ============================================================================
-- TABLE 4: DSPy PROMPT VERSIONS
-- Version control for optimized prompts
-- ============================================================================

CREATE TABLE IF NOT EXISTS dspy_prompt_versions (
    -- Primary key
    version_id SERIAL PRIMARY KEY,
    version_uuid UUID DEFAULT gen_random_uuid() UNIQUE,

    -- === Prompt Identity ===
    agent_name VARCHAR(50) NOT NULL,
    signature_name VARCHAR(100) NOT NULL,

    -- === Version Info ===
    version_number INTEGER NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,

    -- === Prompt Content ===
    prompt_template TEXT NOT NULL,
    few_shot_examples JSONB DEFAULT '[]',
    system_prompt TEXT,

    -- === Performance Metrics ===
    avg_reward FLOAT,
    sample_size INTEGER DEFAULT 0,

    -- === Source ===
    optimization_run_id INTEGER REFERENCES dspy_optimization_runs(run_id),
    created_by VARCHAR(100),  -- 'miprov2', 'manual', 'baseline'

    -- === Timestamps ===
    created_at TIMESTAMPTZ DEFAULT NOW(),
    activated_at TIMESTAMPTZ,
    deactivated_at TIMESTAMPTZ,

    UNIQUE(agent_name, signature_name, version_number)
);

-- Indexes for prompt versions
CREATE INDEX IF NOT EXISTS idx_prompt_versions_agent ON dspy_prompt_versions(agent_name, signature_name);
CREATE INDEX IF NOT EXISTS idx_prompt_versions_active ON dspy_prompt_versions(agent_name, signature_name, is_active) WHERE is_active = TRUE;


-- ============================================================================
-- ENHANCE EXISTING learning_signals TABLE
-- Add missing columns for comprehensive DSPy support
-- ============================================================================

-- Add columns if they don't exist
DO $$
BEGIN
    -- Reward score
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'reward') THEN
        ALTER TABLE learning_signals ADD COLUMN reward FLOAT CHECK (reward BETWEEN 0 AND 1);
    END IF;

    -- Latency breakdown
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'latency_breakdown') THEN
        ALTER TABLE learning_signals ADD COLUMN latency_breakdown JSONB DEFAULT '{}';
    END IF;

    -- LLM usage details
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'llm_calls') THEN
        ALTER TABLE learning_signals ADD COLUMN llm_calls INTEGER DEFAULT 0;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'total_tokens') THEN
        ALTER TABLE learning_signals ADD COLUMN total_tokens INTEGER DEFAULT 0;
    END IF;

    -- Cognitive context reference
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'cognitive_context_id') THEN
        ALTER TABLE learning_signals ADD COLUMN cognitive_context_id UUID;
    END IF;

    -- Optimization run reference
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'optimization_run_id') THEN
        ALTER TABLE learning_signals ADD COLUMN optimization_run_id INTEGER;
    END IF;
END $$;


-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get training examples for an agent
CREATE OR REPLACE FUNCTION get_dspy_training_examples(
    p_agent VARCHAR,
    p_min_reward FLOAT DEFAULT 0.5,
    p_limit INT DEFAULT 100
)
RETURNS TABLE (
    signal_id UUID,
    input_context JSONB,
    output JSONB,
    quality_metrics JSONB,
    reward FLOAT,
    created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.signal_id,
        s.input_context,
        s.output,
        s.quality_metrics,
        s.reward,
        s.created_at
    FROM dspy_agent_training_signals s
    WHERE
        s.source_agent = p_agent
        AND s.is_training_example = TRUE
        AND s.excluded_from_training = FALSE
        AND s.reward >= p_min_reward
    ORDER BY s.reward DESC, s.created_at DESC
    LIMIT p_limit;
END;
$$;


-- Function to compute aggregate metrics for an agent
CREATE OR REPLACE FUNCTION get_dspy_agent_metrics(
    p_agent VARCHAR,
    p_days INT DEFAULT 30
)
RETURNS TABLE (
    signal_count INTEGER,
    avg_reward FLOAT,
    avg_latency_ms FLOAT,
    validated_count INTEGER,
    improvement_trend FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER,
        AVG(s.reward),
        AVG(s.total_latency_ms)::FLOAT,
        COUNT(*) FILTER (WHERE s.human_validated = TRUE)::INTEGER,
        -- Trend: compare recent week to previous period
        COALESCE(
            AVG(s.reward) FILTER (WHERE s.created_at > NOW() - INTERVAL '7 days') -
            AVG(s.reward) FILTER (WHERE s.created_at <= NOW() - INTERVAL '7 days'),
            0
        )
    FROM dspy_agent_training_signals s
    WHERE
        s.source_agent = p_agent
        AND s.created_at > NOW() - (p_days || ' days')::INTERVAL;
END;
$$;


-- Function to get the active prompt for an agent/signature
CREATE OR REPLACE FUNCTION get_active_prompt(
    p_agent VARCHAR,
    p_signature VARCHAR
)
RETURNS TABLE (
    version_id INTEGER,
    version_number INTEGER,
    prompt_template TEXT,
    few_shot_examples JSONB,
    avg_reward FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        v.version_id,
        v.version_number,
        v.prompt_template,
        v.few_shot_examples,
        v.avg_reward
    FROM dspy_prompt_versions v
    WHERE
        v.agent_name = p_agent
        AND v.signature_name = p_signature
        AND v.is_active = TRUE
    LIMIT 1;
END;
$$;


-- ============================================================================
-- GRANTS
-- ============================================================================

GRANT SELECT, INSERT, UPDATE ON dspy_agent_training_signals TO authenticated;
GRANT SELECT, INSERT, UPDATE ON dspy_cognitive_context_history TO authenticated;
GRANT SELECT, INSERT, UPDATE ON dspy_optimization_runs TO authenticated;
GRANT SELECT, INSERT, UPDATE ON dspy_prompt_versions TO authenticated;
GRANT USAGE ON SEQUENCE dspy_optimization_runs_run_id_seq TO authenticated;
GRANT USAGE ON SEQUENCE dspy_prompt_versions_version_id_seq TO authenticated;


-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE dspy_agent_training_signals IS 'Comprehensive DSPy training signals for MIPROv2 optimization of all E2I agents';
COMMENT ON TABLE dspy_cognitive_context_history IS 'Historical cognitive contexts from CognitiveRAG 4-phase cycle for replay and analysis';
COMMENT ON TABLE dspy_optimization_runs IS 'Tracking table for MIPROv2 prompt optimization experiments';
COMMENT ON TABLE dspy_prompt_versions IS 'Version control for DSPy-optimized prompts with rollback support';


-- ============================================================================
-- SUCCESS MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '========================================================';
    RAISE NOTICE 'E2I DSPy Training Signals Enhancement installed!';
    RAISE NOTICE '';
    RAISE NOTICE 'New tables created:';
    RAISE NOTICE '  - dspy_agent_training_signals';
    RAISE NOTICE '  - dspy_cognitive_context_history';
    RAISE NOTICE '  - dspy_optimization_runs';
    RAISE NOTICE '  - dspy_prompt_versions';
    RAISE NOTICE '';
    RAISE NOTICE 'Enhanced tables:';
    RAISE NOTICE '  - learning_signals (new DSPy columns)';
    RAISE NOTICE '';
    RAISE NOTICE 'Helper functions:';
    RAISE NOTICE '  - get_dspy_training_examples()';
    RAISE NOTICE '  - get_dspy_agent_metrics()';
    RAISE NOTICE '  - get_active_prompt()';
    RAISE NOTICE '========================================================';
END $$;
