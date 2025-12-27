-- ============================================================================
-- E2I Causal Analytics - Migration 023: GEPA Optimization Infrastructure
-- ============================================================================
-- Version: 4.2
-- Date: December 2025
-- Description: Adds tables for GEPA prompt optimization tracking
-- 
-- IMPORTANT: This migration ADDS to existing schema. NO tables are deleted.
-- All V4.1 tables (28 total) remain unchanged and fully operational.
-- 
-- New Tables: 4
-- New ENUMs: 4
-- Total Tables After Migration: 32
-- ============================================================================

-- ============================================================================
-- SECTION 1: NEW ENUMS FOR GEPA OPTIMIZATION
-- ============================================================================

-- Optimizer type enum (extends beyond just GEPA for future-proofing)
CREATE TYPE optimizer_type AS ENUM (
    'miprov2',              -- Previous DSPy optimizer (baseline)
    'gepa',                 -- GEPA: Generative Evolutionary Prompting with AI
    'bootstrap_fewshot',    -- DSPy BootstrapFewShot
    'copro',                -- DSPy COPRO
    'simba',                -- DSPy SIMBA
    'manual'                -- Manual prompt engineering
);

-- GEPA budget presets
CREATE TYPE gepa_budget_preset AS ENUM (
    'light',    -- Quick experimentation, ~500 metric calls
    'medium',   -- Balanced optimization, ~2000 metric calls
    'heavy',    -- Thorough optimization, ~4000+ metric calls
    'custom'    -- User-defined max_metric_calls
);

-- Agent optimization status
CREATE TYPE optimization_status AS ENUM (
    'pending',      -- Scheduled but not started
    'running',      -- Currently optimizing
    'completed',    -- Successfully completed
    'failed',       -- Failed with error
    'cancelled',    -- Manually cancelled
    'rolled_back'   -- Rolled back to previous version
);

-- A/B test variant type
CREATE TYPE ab_test_variant AS ENUM (
    'baseline',     -- MIPROv2 or manual baseline
    'gepa',         -- GEPA optimized version
    'gepa_v2',      -- Second GEPA iteration
    'control'       -- Unoptimized control
);

-- ============================================================================
-- SECTION 2: PROMPT OPTIMIZATION RUNS TABLE
-- ============================================================================
-- Tracks each GEPA optimization session (analogous to ml_experiments but for prompts)

CREATE TABLE prompt_optimization_runs (
    -- Primary Key
    run_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Run Identification
    run_name            VARCHAR(255) NOT NULL,
    agent_name          VARCHAR(100) NOT NULL,  -- e.g., 'causal_impact', 'experiment_designer'
    agent_tier          SMALLINT NOT NULL CHECK (agent_tier >= 0 AND agent_tier <= 5),
    agent_type          VARCHAR(20) NOT NULL,   -- 'standard', 'hybrid', 'deep'
    
    -- Optimizer Configuration
    optimizer_type      optimizer_type NOT NULL DEFAULT 'gepa',
    budget_preset       gepa_budget_preset NOT NULL DEFAULT 'medium',
    max_metric_calls    INTEGER,
    reflection_model    VARCHAR(100),           -- e.g., 'claude-sonnet-4-20250514'
    reflection_minibatch_size INTEGER DEFAULT 3,
    enable_tool_optimization BOOLEAN DEFAULT FALSE,
    candidate_selection_strategy VARCHAR(20) DEFAULT 'pareto',
    
    -- Dataset Information
    trainset_size       INTEGER NOT NULL,
    valset_size         INTEGER,
    trainset_hash       VARCHAR(64),            -- SHA256 of trainset for reproducibility
    valset_hash         VARCHAR(64),
    
    -- Status & Timing
    status              optimization_status NOT NULL DEFAULT 'pending',
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    duration_seconds    INTEGER,
    
    -- Results Summary
    baseline_score      DECIMAL(5,4),
    optimized_score     DECIMAL(5,4),
    improvement_percent DECIMAL(5,2),
    total_metric_calls  INTEGER,
    num_candidates_explored INTEGER,
    pareto_frontier_size INTEGER,
    
    -- Artifacts
    log_dir             VARCHAR(500),
    mlflow_run_id       VARCHAR(100),
    best_candidate_idx  INTEGER,
    
    -- Error Handling
    error_message       TEXT,
    error_traceback     TEXT,
    
    -- Metadata
    config_json         JSONB,                  -- Full GEPA config for reproducibility
    seed                INTEGER DEFAULT 42,
    created_by          VARCHAR(100),
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now(),
    
    -- Constraints
    CONSTRAINT valid_improvement 
        CHECK (improvement_percent IS NULL OR improvement_percent >= -100)
);

-- Indexes for common queries
CREATE INDEX idx_prompt_opt_runs_agent ON prompt_optimization_runs(agent_name);
CREATE INDEX idx_prompt_opt_runs_status ON prompt_optimization_runs(status);
CREATE INDEX idx_prompt_opt_runs_created ON prompt_optimization_runs(created_at DESC);
CREATE INDEX idx_prompt_opt_runs_optimizer ON prompt_optimization_runs(optimizer_type);

-- ============================================================================
-- SECTION 3: OPTIMIZED INSTRUCTIONS TABLE
-- ============================================================================
-- Stores versioned agent instructions/prompts produced by GEPA

CREATE TABLE optimized_instructions (
    -- Primary Key
    instruction_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Relationship to Optimization Run
    run_id              UUID NOT NULL REFERENCES prompt_optimization_runs(run_id),
    
    -- Agent Identification
    agent_name          VARCHAR(100) NOT NULL,
    predictor_name      VARCHAR(100) NOT NULL,  -- DSPy predictor name within agent
    
    -- Versioning
    version             VARCHAR(50) NOT NULL,   -- e.g., 'gepa_v1_20251224_143052'
    is_active           BOOLEAN DEFAULT FALSE,  -- Currently deployed version
    is_baseline         BOOLEAN DEFAULT FALSE,  -- MIPROv2 baseline for comparison
    
    -- Instruction Content
    instruction_text    TEXT NOT NULL,          -- The optimized instruction/prompt
    instruction_hash    VARCHAR(64) NOT NULL,   -- SHA256 for deduplication
    
    -- Performance Metrics
    val_score           DECIMAL(5,4),
    val_score_components JSONB,                 -- Breakdown by metric component
    
    -- Pareto Information
    candidate_idx       INTEGER,                -- Index in GEPA candidate list
    parent_indices      INTEGER[],              -- Lineage from GEPA evolution
    discovery_eval_count INTEGER,               -- Budget consumed at discovery
    
    -- Metadata
    created_at          TIMESTAMPTZ DEFAULT now(),
    activated_at        TIMESTAMPTZ,
    deactivated_at      TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT unique_active_instruction 
        UNIQUE NULLS NOT DISTINCT (agent_name, predictor_name, is_active) 
        -- Only one active instruction per agent/predictor
);

-- Indexes
CREATE INDEX idx_opt_instructions_agent ON optimized_instructions(agent_name);
CREATE INDEX idx_opt_instructions_active ON optimized_instructions(agent_name, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_opt_instructions_run ON optimized_instructions(run_id);
CREATE UNIQUE INDEX idx_opt_instructions_hash ON optimized_instructions(instruction_hash);

-- ============================================================================
-- SECTION 4: OPTIMIZED TOOL DESCRIPTIONS TABLE
-- ============================================================================
-- Stores optimized tool descriptions when enable_tool_optimization=True

CREATE TABLE optimized_tool_descriptions (
    -- Primary Key
    tool_description_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Relationship to Optimization Run
    run_id              UUID NOT NULL REFERENCES prompt_optimization_runs(run_id),
    
    -- Tool Identification
    agent_name          VARCHAR(100) NOT NULL,
    tool_name           VARCHAR(100) NOT NULL,  -- e.g., 'causal_forest', 'linear_dml'
    
    -- Versioning
    version             VARCHAR(50) NOT NULL,
    is_active           BOOLEAN DEFAULT FALSE,
    
    -- Content
    description_text    TEXT NOT NULL,          -- Optimized tool description
    argument_descriptions JSONB,                -- Optimized argument descriptions
    description_hash    VARCHAR(64) NOT NULL,
    
    -- Original for Comparison
    original_description TEXT,
    original_arguments  JSONB,
    
    -- Performance Impact
    tool_selection_accuracy DECIMAL(5,4),       -- How often correct tool selected
    
    -- Metadata
    created_at          TIMESTAMPTZ DEFAULT now(),
    
    -- Constraints
    CONSTRAINT unique_active_tool_desc 
        UNIQUE NULLS NOT DISTINCT (agent_name, tool_name, is_active)
);

-- Indexes
CREATE INDEX idx_tool_desc_agent ON optimized_tool_descriptions(agent_name);
CREATE INDEX idx_tool_desc_active ON optimized_tool_descriptions(agent_name, is_active) WHERE is_active = TRUE;

-- ============================================================================
-- SECTION 5: PROMPT A/B TESTS TABLE
-- ============================================================================
-- Tracks A/B tests comparing GEPA vs baseline in production

CREATE TABLE prompt_ab_tests (
    -- Primary Key
    test_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Test Configuration
    test_name           VARCHAR(255) NOT NULL,
    agent_name          VARCHAR(100) NOT NULL,
    
    -- Variants
    baseline_instruction_id UUID REFERENCES optimized_instructions(instruction_id),
    treatment_instruction_id UUID REFERENCES optimized_instructions(instruction_id),
    
    -- Traffic Split
    traffic_split       DECIMAL(3,2) NOT NULL DEFAULT 0.10,  -- % to treatment
    
    -- Status & Timing
    status              VARCHAR(20) DEFAULT 'draft',  -- draft, running, completed, stopped
    started_at          TIMESTAMPTZ,
    ended_at            TIMESTAMPTZ,
    target_sample_size  INTEGER,
    
    -- Results
    baseline_requests   INTEGER DEFAULT 0,
    treatment_requests  INTEGER DEFAULT 0,
    baseline_score_avg  DECIMAL(5,4),
    treatment_score_avg DECIMAL(5,4),
    baseline_latency_p50 INTEGER,               -- milliseconds
    treatment_latency_p50 INTEGER,
    
    -- Statistical Significance
    p_value             DECIMAL(5,4),
    confidence_interval_lower DECIMAL(5,4),
    confidence_interval_upper DECIMAL(5,4),
    is_significant      BOOLEAN,
    
    -- Decision
    winner              ab_test_variant,
    decision_reason     TEXT,
    rolled_out          BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    created_by          VARCHAR(100),
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX idx_ab_tests_agent ON prompt_ab_tests(agent_name);
CREATE INDEX idx_ab_tests_status ON prompt_ab_tests(status);

-- ============================================================================
-- SECTION 6: A/B TEST OBSERVATIONS TABLE
-- ============================================================================
-- Individual observations for A/B test analysis

CREATE TABLE prompt_ab_test_observations (
    -- Primary Key
    observation_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Relationship
    test_id             UUID NOT NULL REFERENCES prompt_ab_tests(test_id),
    
    -- Request Info
    request_id          VARCHAR(100) NOT NULL,
    variant             ab_test_variant NOT NULL,
    
    -- Outcome
    score               DECIMAL(5,4),
    latency_ms          INTEGER,
    success             BOOLEAN,
    error_type          VARCHAR(100),
    
    -- Context
    user_id             VARCHAR(100),
    session_id          VARCHAR(100),
    
    -- Metadata
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX idx_ab_observations_test ON prompt_ab_test_observations(test_id);
CREATE INDEX idx_ab_observations_variant ON prompt_ab_test_observations(test_id, variant);
CREATE INDEX idx_ab_observations_created ON prompt_ab_test_observations(created_at DESC);

-- ============================================================================
-- SECTION 7: HELPER VIEWS
-- ============================================================================

-- View: Current active instructions per agent
CREATE OR REPLACE VIEW v_active_instructions AS
SELECT 
    oi.agent_name,
    oi.predictor_name,
    oi.version,
    oi.instruction_text,
    oi.val_score,
    por.optimizer_type,
    por.run_name,
    por.improvement_percent,
    oi.activated_at
FROM optimized_instructions oi
JOIN prompt_optimization_runs por ON oi.run_id = por.run_id
WHERE oi.is_active = TRUE
ORDER BY oi.agent_name, oi.predictor_name;

-- View: Optimization run summary with improvement metrics
CREATE OR REPLACE VIEW v_optimization_summary AS
SELECT 
    por.agent_name,
    por.agent_tier,
    por.agent_type,
    por.optimizer_type,
    COUNT(*) as total_runs,
    COUNT(*) FILTER (WHERE por.status = 'completed') as completed_runs,
    AVG(por.improvement_percent) FILTER (WHERE por.status = 'completed') as avg_improvement,
    MAX(por.improvement_percent) FILTER (WHERE por.status = 'completed') as max_improvement,
    AVG(por.total_metric_calls) as avg_metric_calls,
    MAX(por.completed_at) as last_optimized
FROM prompt_optimization_runs por
GROUP BY por.agent_name, por.agent_tier, por.agent_type, por.optimizer_type
ORDER BY por.agent_tier, por.agent_name;

-- View: A/B test results summary
CREATE OR REPLACE VIEW v_ab_test_results AS
SELECT 
    pat.test_name,
    pat.agent_name,
    pat.status,
    pat.baseline_requests,
    pat.treatment_requests,
    pat.baseline_score_avg,
    pat.treatment_score_avg,
    (pat.treatment_score_avg - pat.baseline_score_avg) as score_delta,
    pat.p_value,
    pat.is_significant,
    pat.winner,
    pat.rolled_out
FROM prompt_ab_tests pat
ORDER BY pat.created_at DESC;

-- View: GEPA vs MIPROv2 comparison
CREATE OR REPLACE VIEW v_optimizer_comparison AS
SELECT 
    agent_name,
    agent_tier,
    optimizer_type,
    COUNT(*) as run_count,
    AVG(baseline_score) as avg_baseline,
    AVG(optimized_score) as avg_optimized,
    AVG(improvement_percent) as avg_improvement,
    AVG(total_metric_calls) as avg_metric_calls,
    AVG(duration_seconds) as avg_duration_seconds
FROM prompt_optimization_runs
WHERE status = 'completed'
GROUP BY agent_name, agent_tier, optimizer_type
ORDER BY agent_name, optimizer_type;

-- ============================================================================
-- SECTION 8: TRIGGERS FOR UPDATED_AT
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_prompt_optimization_runs_updated_at
    BEFORE UPDATE ON prompt_optimization_runs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_prompt_ab_tests_updated_at
    BEFORE UPDATE ON prompt_ab_tests
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SECTION 9: COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE prompt_optimization_runs IS 
'Tracks GEPA prompt optimization sessions. Each run optimizes one agent.';

COMMENT ON TABLE optimized_instructions IS 
'Stores versioned agent instructions produced by GEPA with Pareto lineage.';

COMMENT ON TABLE optimized_tool_descriptions IS 
'Stores optimized DoWhy/EconML tool descriptions when enable_tool_optimization=True.';

COMMENT ON TABLE prompt_ab_tests IS 
'Tracks A/B tests comparing GEPA-optimized vs baseline agent versions.';

COMMENT ON TABLE prompt_ab_test_observations IS 
'Individual request observations for A/B test statistical analysis.';

-- ============================================================================
-- SECTION 10: GRANTS (Adjust roles as needed)
-- ============================================================================

-- Grant permissions to application role
-- GRANT SELECT, INSERT, UPDATE ON prompt_optimization_runs TO e2i_app;
-- GRANT SELECT, INSERT, UPDATE ON optimized_instructions TO e2i_app;
-- GRANT SELECT, INSERT, UPDATE ON optimized_tool_descriptions TO e2i_app;
-- GRANT SELECT, INSERT, UPDATE ON prompt_ab_tests TO e2i_app;
-- GRANT SELECT, INSERT ON prompt_ab_test_observations TO e2i_app;

-- Grant read-only to analytics role
-- GRANT SELECT ON v_active_instructions TO e2i_analytics;
-- GRANT SELECT ON v_optimization_summary TO e2i_analytics;
-- GRANT SELECT ON v_ab_test_results TO e2i_analytics;
-- GRANT SELECT ON v_optimizer_comparison TO e2i_analytics;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- 
-- Summary:
-- - 4 new ENUMs added: optimizer_type, gepa_budget_preset, optimization_status, ab_test_variant
-- - 5 new tables added: prompt_optimization_runs, optimized_instructions, 
--   optimized_tool_descriptions, prompt_ab_tests, prompt_ab_test_observations
-- - 4 new views added: v_active_instructions, v_optimization_summary,
--   v_ab_test_results, v_optimizer_comparison
-- - 2 triggers added for updated_at maintenance
--
-- Total Tables: 28 (existing) + 5 (new) = 33 tables
-- 
-- Rollback: DROP tables in reverse order, then DROP ENUMs
-- ============================================================================
