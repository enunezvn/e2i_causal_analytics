-- ============================================================================
-- E2I Causal Analytics - Migration 011: Tool Composer & Orchestrator Classifier
-- ============================================================================
-- 
-- Version: 4.2.0
-- Date: 2024-12-17
-- Description: Adds tables for Tool Composer pattern and Orchestrator classifier
--              to support multi-faceted query handling with dependency-aware
--              tool composition.
--
-- New Tables (6):
--   - composer_episodes: Episodic memory for tool compositions
--   - tool_performance: Tool execution metrics for learning
--   - classification_logs: Query classification decisions audit trail
--   - tool_registry: Composable tools metadata
--   - composition_steps: Individual step execution records
--   - tool_dependencies: Tool dependency relationships
--
-- New ENUMs (4):
--   - routing_pattern: SINGLE_AGENT, PARALLEL_DELEGATION, TOOL_COMPOSER, CLARIFICATION_NEEDED
--   - dependency_type: REFERENCE_CHAIN, CONDITIONAL, LOGICAL_SEQUENCE, ENTITY_TRANSFORMATION
--   - tool_category: CAUSAL, SEGMENTATION, GAP, EXPERIMENT, PREDICTION, MONITORING
--   - composition_status: PENDING, RUNNING, COMPLETED, FAILED, TIMEOUT
--
-- New Views (4):
--   - v_composition_success_rate: Success rates by query pattern
--   - v_tool_reliability: Tool reliability metrics
--   - v_classification_accuracy: Classifier performance metrics
--   - v_active_compositions: Currently running compositions
--
-- Dependencies: Requires previous migrations (001-010)
-- ============================================================================

-- ============================================================================
-- SECTION 1: ENUM TYPES
-- ============================================================================

-- Routing patterns for query classification
CREATE TYPE routing_pattern AS ENUM (
    'SINGLE_AGENT',           -- Route to single primary agent
    'PARALLEL_DELEGATION',    -- Route to multiple independent agents
    'TOOL_COMPOSER',          -- Use Tool Composer for dependent multi-domain
    'CLARIFICATION_NEEDED'    -- Query too ambiguous, request clarification
);

COMMENT ON TYPE routing_pattern IS 'Query routing patterns determined by Orchestrator classifier';

-- Dependency types between sub-questions
CREATE TYPE dependency_type AS ENUM (
    'REFERENCE_CHAIN',        -- Pronoun/phrase references earlier result ("that", "those")
    'CONDITIONAL',            -- Conditional logic ("if X then Y")
    'LOGICAL_SEQUENCE',       -- Natural ordering required (cause → effect → intervention)
    'ENTITY_TRANSFORMATION'   -- Entity filtered/transformed by earlier step
);

COMMENT ON TYPE dependency_type IS 'Types of data dependencies between decomposed sub-questions';

-- Tool capability categories (maps to agent domains)
CREATE TYPE tool_category AS ENUM (
    'CAUSAL',                 -- Causal Impact Agent tools
    'SEGMENTATION',           -- Heterogeneous Optimizer tools
    'GAP',                    -- Gap Analyzer tools
    'EXPERIMENT',             -- Experiment Designer tools
    'PREDICTION',             -- Prediction Synthesizer tools
    'MONITORING'              -- Drift Monitor tools
);

COMMENT ON TYPE tool_category IS 'Tool categories mapping to agent capability domains';

-- Composition execution status
CREATE TYPE composition_status AS ENUM (
    'PENDING',                -- Composition queued
    'DECOMPOSING',            -- Phase 1: Breaking down query
    'PLANNING',               -- Phase 2: Creating execution plan
    'EXECUTING',              -- Phase 3: Running tools
    'SYNTHESIZING',           -- Phase 4: Combining results
    'COMPLETED',              -- Successfully finished
    'FAILED',                 -- Error during execution
    'TIMEOUT'                 -- Exceeded time limit
);

COMMENT ON TYPE composition_status IS 'Status of a tool composition execution';


-- ============================================================================
-- SECTION 2: CORE TABLES
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Table: tool_registry
-- Purpose: Central registry of tools exposed by agents for composition
-- ----------------------------------------------------------------------------
CREATE TABLE tool_registry (
    tool_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Tool identification
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    category tool_category NOT NULL,
    source_agent VARCHAR(50) NOT NULL,
    
    -- Schemas (JSON Schema format)
    input_schema JSONB NOT NULL DEFAULT '{}',
    output_schema JSONB NOT NULL DEFAULT '{}',
    
    -- Composition flags
    composable BOOLEAN NOT NULL DEFAULT true,
    
    -- Performance baselines (updated by tool_performance aggregation)
    avg_latency_ms FLOAT DEFAULT 500.0,
    success_rate FLOAT DEFAULT 0.95 CHECK (success_rate >= 0 AND success_rate <= 1),
    
    -- Metadata
    version VARCHAR(20) DEFAULT '1.0.0',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deprecated_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT valid_agent CHECK (source_agent IN (
        'causal_impact', 'heterogeneous_optimizer', 'gap_analyzer',
        'experiment_designer', 'prediction_synthesizer', 'drift_monitor'
    ))
);

CREATE INDEX idx_tool_registry_category ON tool_registry(category);
CREATE INDEX idx_tool_registry_agent ON tool_registry(source_agent);
CREATE INDEX idx_tool_registry_composable ON tool_registry(composable) WHERE composable = true;

COMMENT ON TABLE tool_registry IS 'Central registry of composable tools exposed by agents';
COMMENT ON COLUMN tool_registry.input_schema IS 'JSON Schema defining tool input parameters';
COMMENT ON COLUMN tool_registry.output_schema IS 'JSON Schema defining tool output structure';
COMMENT ON COLUMN tool_registry.composable IS 'Whether tool can be used by Tool Composer';


-- ----------------------------------------------------------------------------
-- Table: tool_dependencies
-- Purpose: Define which tools can consume output from other tools
-- ----------------------------------------------------------------------------
CREATE TABLE tool_dependencies (
    dependency_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Relationship
    consumer_tool_id UUID NOT NULL REFERENCES tool_registry(tool_id) ON DELETE CASCADE,
    producer_tool_id UUID NOT NULL REFERENCES tool_registry(tool_id) ON DELETE CASCADE,
    
    -- Mapping configuration
    output_field VARCHAR(100),  -- Specific field from producer (NULL = entire output)
    input_field VARCHAR(100),   -- Specific field in consumer (NULL = auto-map)
    transform_expression TEXT,  -- Optional jq/JSONPath expression for transformation
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT no_self_dependency CHECK (consumer_tool_id != producer_tool_id),
    CONSTRAINT unique_dependency UNIQUE (consumer_tool_id, producer_tool_id)
);

CREATE INDEX idx_tool_deps_consumer ON tool_dependencies(consumer_tool_id);
CREATE INDEX idx_tool_deps_producer ON tool_dependencies(producer_tool_id);

COMMENT ON TABLE tool_dependencies IS 'Defines tool input/output compatibility for composition';
COMMENT ON COLUMN tool_dependencies.transform_expression IS 'Optional transformation for output→input mapping';


-- ----------------------------------------------------------------------------
-- Table: classification_logs
-- Purpose: Audit trail of query classification decisions
-- ----------------------------------------------------------------------------
CREATE TABLE classification_logs (
    classification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Query information
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64) NOT NULL,  -- SHA-256 for deduplication analysis
    
    -- Classification results
    routing_pattern routing_pattern NOT NULL,
    target_agents TEXT[] NOT NULL DEFAULT '{}',
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Detailed analysis (Stage outputs)
    features_extracted JSONB NOT NULL DEFAULT '{}',
    domain_mapping JSONB NOT NULL DEFAULT '{}',
    dependency_analysis JSONB NOT NULL DEFAULT '{}',
    
    -- Sub-questions if multi-part
    sub_questions JSONB DEFAULT '[]',
    dependencies JSONB DEFAULT '[]',
    
    -- Classification metadata
    used_llm_layer BOOLEAN NOT NULL DEFAULT false,
    classification_latency_ms FLOAT NOT NULL,
    
    -- Context
    session_id VARCHAR(100),
    user_id VARCHAR(100),
    is_followup BOOLEAN DEFAULT false,
    
    -- Feedback (populated later by user/system)
    was_correct BOOLEAN,  -- NULL until feedback received
    correct_pattern routing_pattern,  -- If was_correct=false, what should it have been
    feedback_notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_classification_logs_pattern ON classification_logs(routing_pattern);
CREATE INDEX idx_classification_logs_created ON classification_logs(created_at DESC);
CREATE INDEX idx_classification_logs_session ON classification_logs(session_id);
CREATE INDEX idx_classification_logs_hash ON classification_logs(query_hash);
CREATE INDEX idx_classification_logs_feedback ON classification_logs(was_correct) WHERE was_correct IS NOT NULL;

-- GIN index for JSONB searches
CREATE INDEX idx_classification_logs_domains ON classification_logs USING GIN (domain_mapping);

COMMENT ON TABLE classification_logs IS 'Audit trail of Orchestrator query classification decisions';
COMMENT ON COLUMN classification_logs.query_hash IS 'SHA-256 hash for finding similar queries';
COMMENT ON COLUMN classification_logs.was_correct IS 'Feedback: NULL=pending, true=correct, false=incorrect';


-- ----------------------------------------------------------------------------
-- Table: composer_episodes
-- Purpose: Episodic memory for tool compositions (learning from experience)
-- ----------------------------------------------------------------------------
CREATE TABLE composer_episodes (
    episode_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    composition_id VARCHAR(100) NOT NULL UNIQUE,
    
    -- Query information
    query_text TEXT NOT NULL,
    query_embedding vector(1536),  -- For similarity search (pgvector)
    
    -- Decomposition (Phase 1 output)
    sub_questions JSONB NOT NULL DEFAULT '[]',
    
    -- Execution plan (Phase 2 output)
    tool_plan JSONB NOT NULL DEFAULT '{}',
    parallelizable_groups JSONB DEFAULT '[]',
    
    -- Execution results (Phase 3 output)
    tool_outputs JSONB DEFAULT '{}',
    
    -- Synthesis (Phase 4 output)
    synthesized_response TEXT,
    
    -- Performance metrics
    total_latency_ms FLOAT NOT NULL,
    decompose_latency_ms FLOAT,
    plan_latency_ms FLOAT,
    execute_latency_ms FLOAT,
    synthesize_latency_ms FLOAT,
    
    -- Status
    status composition_status NOT NULL DEFAULT 'PENDING',
    error_message TEXT,
    
    -- Feedback (for learning)
    success BOOLEAN,  -- NULL until feedback
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
    feedback_text TEXT,
    
    -- Context
    session_id VARCHAR(100),
    user_id VARCHAR(100),
    classification_id UUID REFERENCES classification_logs(classification_id),
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    feedback_at TIMESTAMPTZ
);

-- Standard indexes
CREATE INDEX idx_composer_episodes_status ON composer_episodes(status);
CREATE INDEX idx_composer_episodes_created ON composer_episodes(created_at DESC);
CREATE INDEX idx_composer_episodes_session ON composer_episodes(session_id);
CREATE INDEX idx_composer_episodes_success ON composer_episodes(success) WHERE success IS NOT NULL;

-- Vector similarity index for finding similar past compositions
CREATE INDEX idx_composer_episodes_embedding ON composer_episodes 
    USING ivfflat (query_embedding vector_cosine_ops) 
    WITH (lists = 100);

-- GIN indexes for JSONB queries
CREATE INDEX idx_composer_episodes_plan ON composer_episodes USING GIN (tool_plan);
CREATE INDEX idx_composer_episodes_subq ON composer_episodes USING GIN (sub_questions);

COMMENT ON TABLE composer_episodes IS 'Episodic memory for tool compositions - enables learning from experience';
COMMENT ON COLUMN composer_episodes.query_embedding IS 'Vector embedding for similarity search (1536-dim)';
COMMENT ON COLUMN composer_episodes.parallelizable_groups IS 'Groups of tools that can execute in parallel';


-- ----------------------------------------------------------------------------
-- Table: composition_steps
-- Purpose: Individual step execution records within a composition
-- ----------------------------------------------------------------------------
CREATE TABLE composition_steps (
    step_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Parent composition
    episode_id UUID NOT NULL REFERENCES composer_episodes(episode_id) ON DELETE CASCADE,
    
    -- Step identification
    step_number INTEGER NOT NULL,
    step_name VARCHAR(100) NOT NULL,
    
    -- Tool execution
    tool_id UUID NOT NULL REFERENCES tool_registry(tool_id),
    tool_name VARCHAR(100) NOT NULL,
    
    -- Input/Output
    input_params JSONB NOT NULL DEFAULT '{}',
    output_result JSONB,
    
    -- Dependencies
    depends_on_steps INTEGER[] DEFAULT '{}',
    serves_sub_question VARCHAR(20),  -- e.g., "Q1", "Q2"
    
    -- Execution metrics
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    latency_ms FLOAT,
    
    -- Status
    status composition_status NOT NULL DEFAULT 'PENDING',
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT unique_step_in_episode UNIQUE (episode_id, step_number)
);

CREATE INDEX idx_composition_steps_episode ON composition_steps(episode_id);
CREATE INDEX idx_composition_steps_tool ON composition_steps(tool_id);
CREATE INDEX idx_composition_steps_status ON composition_steps(status);

COMMENT ON TABLE composition_steps IS 'Individual tool execution steps within a composition';
COMMENT ON COLUMN composition_steps.depends_on_steps IS 'Step numbers this step depends on';
COMMENT ON COLUMN composition_steps.serves_sub_question IS 'Which sub-question this step answers';


-- ----------------------------------------------------------------------------
-- Table: tool_performance
-- Purpose: Track tool execution performance for optimization
-- ----------------------------------------------------------------------------
CREATE TABLE tool_performance (
    performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Tool reference
    tool_id UUID NOT NULL REFERENCES tool_registry(tool_id) ON DELETE CASCADE,
    tool_name VARCHAR(100) NOT NULL,
    
    -- Execution context
    context_type VARCHAR(50),  -- e.g., "regional_comparison", "segment_analysis"
    input_complexity INTEGER,  -- Rough measure of input size/complexity
    
    -- Metrics
    latency_ms FLOAT NOT NULL,
    success BOOLEAN NOT NULL,
    error_type VARCHAR(100),
    
    -- Caller context
    composition_id VARCHAR(100),
    step_id UUID REFERENCES composition_steps(step_id),
    called_by VARCHAR(50),  -- 'composer', 'agent', 'direct'
    
    -- Timestamps
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tool_performance_tool ON tool_performance(tool_id);
CREATE INDEX idx_tool_performance_executed ON tool_performance(executed_at DESC);
CREATE INDEX idx_tool_performance_context ON tool_performance(context_type);
CREATE INDEX idx_tool_performance_success ON tool_performance(success);

-- Partitioning by month for performance (if table grows large)
-- Note: Uncomment and adjust for production use
-- CREATE TABLE tool_performance_y2024m12 PARTITION OF tool_performance
--     FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

COMMENT ON TABLE tool_performance IS 'Tool execution metrics for performance optimization';
COMMENT ON COLUMN tool_performance.context_type IS 'Category of usage context for segment-specific analysis';


-- ============================================================================
-- SECTION 3: HELPER VIEWS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- View: v_composition_success_rate
-- Purpose: Success rates by query pattern and complexity
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_composition_success_rate AS
SELECT 
    DATE_TRUNC('day', created_at) AS day,
    status,
    COUNT(*) AS total_compositions,
    COUNT(*) FILTER (WHERE success = true) AS successful,
    COUNT(*) FILTER (WHERE success = false) AS failed,
    COUNT(*) FILTER (WHERE success IS NULL) AS pending_feedback,
    ROUND(
        COUNT(*) FILTER (WHERE success = true)::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE success IS NOT NULL), 0) * 100, 
        2
    ) AS success_rate_pct,
    ROUND(AVG(total_latency_ms), 2) AS avg_latency_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_latency_ms), 2) AS p95_latency_ms,
    jsonb_agg(DISTINCT jsonb_array_elements_text(
        CASE WHEN tool_plan ? 'steps' 
             THEN (tool_plan->'steps') 
             ELSE '[]'::jsonb 
        END
    )) FILTER (WHERE tool_plan IS NOT NULL) AS tools_used
FROM composer_episodes
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at), status
ORDER BY day DESC, status;

COMMENT ON VIEW v_composition_success_rate IS 'Daily composition success metrics for monitoring';


-- ----------------------------------------------------------------------------
-- View: v_tool_reliability
-- Purpose: Tool reliability metrics for planner optimization
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_tool_reliability AS
SELECT 
    tr.tool_id,
    tr.name AS tool_name,
    tr.category,
    tr.source_agent,
    COUNT(tp.performance_id) AS total_executions,
    COUNT(*) FILTER (WHERE tp.success = true) AS successful_executions,
    ROUND(
        COUNT(*) FILTER (WHERE tp.success = true)::NUMERIC / 
        NULLIF(COUNT(*), 0) * 100, 
        2
    ) AS success_rate_pct,
    ROUND(AVG(tp.latency_ms), 2) AS avg_latency_ms,
    ROUND(STDDEV(tp.latency_ms), 2) AS stddev_latency_ms,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY tp.latency_ms), 2) AS p50_latency_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY tp.latency_ms), 2) AS p95_latency_ms,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY tp.latency_ms), 2) AS p99_latency_ms,
    MAX(tp.executed_at) AS last_executed_at,
    MODE() WITHIN GROUP (ORDER BY tp.error_type) FILTER (WHERE tp.error_type IS NOT NULL) AS most_common_error
FROM tool_registry tr
LEFT JOIN tool_performance tp ON tr.tool_id = tp.tool_id
    AND tp.executed_at > NOW() - INTERVAL '7 days'
WHERE tr.composable = true
GROUP BY tr.tool_id, tr.name, tr.category, tr.source_agent
ORDER BY total_executions DESC;

COMMENT ON VIEW v_tool_reliability IS 'Tool reliability metrics from last 7 days for planner decisions';


-- ----------------------------------------------------------------------------
-- View: v_classification_accuracy
-- Purpose: Classifier performance for improvement tracking
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_classification_accuracy AS
SELECT 
    DATE_TRUNC('day', created_at) AS day,
    routing_pattern,
    COUNT(*) AS total_classifications,
    COUNT(*) FILTER (WHERE was_correct = true) AS correct,
    COUNT(*) FILTER (WHERE was_correct = false) AS incorrect,
    COUNT(*) FILTER (WHERE was_correct IS NULL) AS awaiting_feedback,
    ROUND(
        COUNT(*) FILTER (WHERE was_correct = true)::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE was_correct IS NOT NULL), 0) * 100, 
        2
    ) AS accuracy_pct,
    ROUND(AVG(confidence), 3) AS avg_confidence,
    ROUND(AVG(classification_latency_ms), 2) AS avg_latency_ms,
    COUNT(*) FILTER (WHERE used_llm_layer = true) AS used_llm_count
FROM classification_logs
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at), routing_pattern
ORDER BY day DESC, routing_pattern;

COMMENT ON VIEW v_classification_accuracy IS 'Daily classifier accuracy metrics by routing pattern';


-- ----------------------------------------------------------------------------
-- View: v_active_compositions
-- Purpose: Currently running compositions for monitoring
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_active_compositions AS
SELECT 
    ce.composition_id,
    ce.status,
    ce.query_text,
    ce.created_at,
    EXTRACT(EPOCH FROM (NOW() - ce.created_at)) * 1000 AS elapsed_ms,
    ce.session_id,
    jsonb_array_length(ce.sub_questions) AS sub_question_count,
    COUNT(cs.step_id) AS total_steps,
    COUNT(cs.step_id) FILTER (WHERE cs.status = 'COMPLETED') AS completed_steps,
    COUNT(cs.step_id) FILTER (WHERE cs.status = 'EXECUTING') AS running_steps,
    COUNT(cs.step_id) FILTER (WHERE cs.status = 'FAILED') AS failed_steps
FROM composer_episodes ce
LEFT JOIN composition_steps cs ON ce.episode_id = cs.episode_id
WHERE ce.status IN ('PENDING', 'DECOMPOSING', 'PLANNING', 'EXECUTING', 'SYNTHESIZING')
GROUP BY ce.episode_id, ce.composition_id, ce.status, ce.query_text, 
         ce.created_at, ce.session_id, ce.sub_questions
ORDER BY ce.created_at DESC;

COMMENT ON VIEW v_active_compositions IS 'Currently active compositions with progress metrics';


-- ============================================================================
-- SECTION 4: SEED DATA - DEFAULT TOOLS
-- ============================================================================

-- Insert default composable tools from each agent
-- Note: tool_id will be auto-generated, fn references handled at runtime

INSERT INTO tool_registry (name, description, category, source_agent, input_schema, output_schema, composable) VALUES

-- Causal Impact Agent Tools
(
    'causal_effect_estimator',
    'Estimates Average Treatment Effect (ATE) with confidence intervals using DoWhy/EconML',
    'CAUSAL',
    'causal_impact',
    '{
        "type": "object",
        "properties": {
            "treatment_col": {"type": "string", "description": "Column name for treatment variable"},
            "outcome_col": {"type": "string", "description": "Column name for outcome variable"},
            "confounders": {"type": "array", "items": {"type": "string"}, "description": "Confounder column names"}
        },
        "required": ["treatment_col", "outcome_col"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "ate": {"type": "number", "description": "Average Treatment Effect"},
            "ci_lower": {"type": "number", "description": "95% CI lower bound"},
            "ci_upper": {"type": "number", "description": "95% CI upper bound"},
            "p_value": {"type": "number", "description": "Statistical significance"},
            "method": {"type": "string", "description": "Estimation method used"}
        }
    }'::jsonb,
    true
),
(
    'refutation_runner',
    'Runs DoWhy refutation tests (placebo, common cause, subset, bootstrap, sensitivity) on causal estimates',
    'CAUSAL',
    'causal_impact',
    '{
        "type": "object",
        "properties": {
            "causal_result": {"type": "object", "description": "Output from causal_effect_estimator"},
            "tests": {"type": "array", "items": {"type": "string"}, "description": "Which tests to run"}
        },
        "required": ["causal_result"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "passed": {"type": "boolean", "description": "Overall pass/fail"},
            "confidence_score": {"type": "number", "description": "Aggregate confidence 0-1"},
            "test_results": {"type": "array", "description": "Individual test outcomes"},
            "gate_decision": {"type": "string", "description": "proceed/review/block"}
        }
    }'::jsonb,
    true
),
(
    'sensitivity_analyzer',
    'Performs sensitivity analysis on causal estimates to assess robustness to unmeasured confounding',
    'CAUSAL',
    'causal_impact',
    '{
        "type": "object",
        "properties": {
            "causal_result": {"type": "object"},
            "gamma_range": {"type": "array", "items": {"type": "number"}}
        },
        "required": ["causal_result"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "e_value": {"type": "number"},
            "robustness_value": {"type": "number"},
            "sensitivity_plot_data": {"type": "array"}
        }
    }'::jsonb,
    true
),

-- Heterogeneous Optimizer Tools
(
    'cate_analyzer',
    'Computes Conditional Average Treatment Effects (CATE) by segment using CausalForest/CausalML',
    'SEGMENTATION',
    'heterogeneous_optimizer',
    '{
        "type": "object",
        "properties": {
            "causal_result": {"type": "object", "description": "Base ATE from causal_effect_estimator"},
            "segment_cols": {"type": "array", "items": {"type": "string"}, "description": "Columns to segment by"}
        },
        "required": ["segment_cols"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "segments": {"type": "array", "description": "Identified segments"},
            "effect_by_segment": {"type": "object", "description": "CATE for each segment"},
            "high_responders": {"type": "array", "description": "Top responding segments"},
            "feature_importance": {"type": "object"}
        }
    }'::jsonb,
    true
),
(
    'segment_ranker',
    'Ranks segments by treatment effect, ROI, or other criteria for targeting optimization',
    'SEGMENTATION',
    'heterogeneous_optimizer',
    '{
        "type": "object",
        "properties": {
            "cate_result": {"type": "object", "description": "Output from cate_analyzer"},
            "rank_by": {"type": "string", "enum": ["effect", "roi", "volume", "uplift"]},
            "top_n": {"type": "integer", "default": 10}
        },
        "required": ["cate_result"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "ranked_segments": {"type": "array"},
            "targeting_recommendations": {"type": "array"}
        }
    }'::jsonb,
    true
),

-- Gap Analyzer Tools
(
    'gap_calculator',
    'Calculates performance gaps between entities (territories, HCPs, segments)',
    'GAP',
    'gap_analyzer',
    '{
        "type": "object",
        "properties": {
            "entity_type": {"type": "string", "enum": ["territory", "hcp", "segment", "region"]},
            "metric": {"type": "string", "description": "Metric to analyze"},
            "group_by": {"type": "string", "description": "Grouping dimension"},
            "benchmark": {"type": "string", "enum": ["mean", "median", "top_decile", "target"]}
        },
        "required": ["entity_type", "metric"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "gaps": {"type": "array", "description": "Gap by entity"},
            "top_performers": {"type": "array"},
            "bottom_performers": {"type": "array"},
            "total_opportunity": {"type": "number"}
        }
    }'::jsonb,
    true
),
(
    'roi_estimator',
    'Estimates ROI of closing identified performance gaps',
    'GAP',
    'gap_analyzer',
    '{
        "type": "object",
        "properties": {
            "gap_result": {"type": "object", "description": "Output from gap_calculator"},
            "intervention_cost": {"type": "number"},
            "time_horizon_months": {"type": "integer", "default": 12}
        },
        "required": ["gap_result"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "estimated_roi": {"type": "number"},
            "breakeven_months": {"type": "number"},
            "npv": {"type": "number"},
            "confidence_range": {"type": "array"}
        }
    }'::jsonb,
    true
),

-- Experiment Designer Tools
(
    'power_calculator',
    'Calculates statistical power and required sample size for experiment design',
    'EXPERIMENT',
    'experiment_designer',
    '{
        "type": "object",
        "properties": {
            "effect_size": {"type": "number", "description": "Expected effect size (from prior causal analysis)"},
            "baseline_rate": {"type": "number"},
            "sample_size": {"type": "integer"},
            "alpha": {"type": "number", "default": 0.05},
            "power_target": {"type": "number", "default": 0.8}
        },
        "required": ["effect_size"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "power": {"type": "number"},
            "required_sample_size": {"type": "integer"},
            "minimum_detectable_effect": {"type": "number"},
            "duration_estimate_weeks": {"type": "integer"}
        }
    }'::jsonb,
    true
),
(
    'counterfactual_simulator',
    'Simulates counterfactual outcomes for what-if scenarios and intervention planning',
    'EXPERIMENT',
    'experiment_designer',
    '{
        "type": "object",
        "properties": {
            "baseline": {"type": "object", "description": "Current state metrics"},
            "intervention": {"type": "object", "description": "Proposed intervention details"},
            "target_population": {"type": "object", "description": "Population to apply intervention to"},
            "causal_model": {"type": "object", "description": "Optional: specific causal model to use"}
        },
        "required": ["baseline", "intervention"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "predicted_outcome": {"type": "number"},
            "confidence_interval": {"type": "array"},
            "lift_percentage": {"type": "number"},
            "assumptions": {"type": "array"},
            "caveats": {"type": "array"}
        }
    }'::jsonb,
    true
),

-- Prediction Synthesizer Tools
(
    'risk_scorer',
    'Scores entities by predicted risk (discontinuation, churn, adverse events)',
    'PREDICTION',
    'prediction_synthesizer',
    '{
        "type": "object",
        "properties": {
            "entity_type": {"type": "string"},
            "risk_type": {"type": "string", "enum": ["discontinuation", "churn", "adverse_event", "non_adherence"]},
            "time_horizon_days": {"type": "integer", "default": 90}
        },
        "required": ["entity_type", "risk_type"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "scores": {"type": "array"},
            "high_risk_entities": {"type": "array"},
            "risk_factors": {"type": "object"},
            "model_performance": {"type": "object"}
        }
    }'::jsonb,
    true
),
(
    'propensity_estimator',
    'Estimates propensity scores for treatment assignment analysis',
    'PREDICTION',
    'prediction_synthesizer',
    '{
        "type": "object",
        "properties": {
            "treatment_col": {"type": "string"},
            "covariates": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["treatment_col", "covariates"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "propensity_scores": {"type": "array"},
            "balance_statistics": {"type": "object"},
            "overlap_assessment": {"type": "object"}
        }
    }'::jsonb,
    true
),

-- Drift Monitor Tools
(
    'psi_calculator',
    'Calculates Population Stability Index (PSI) for distribution drift detection',
    'MONITORING',
    'drift_monitor',
    '{
        "type": "object",
        "properties": {
            "feature_name": {"type": "string"},
            "reference_period": {"type": "string"},
            "comparison_period": {"type": "string"}
        },
        "required": ["feature_name"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "psi_value": {"type": "number"},
            "drift_severity": {"type": "string", "enum": ["none", "minor", "moderate", "severe"]},
            "bin_contributions": {"type": "array"}
        }
    }'::jsonb,
    true
),
(
    'distribution_comparator',
    'Compares distributions between periods or segments for data quality monitoring',
    'MONITORING',
    'drift_monitor',
    '{
        "type": "object",
        "properties": {
            "columns": {"type": "array", "items": {"type": "string"}},
            "group_a": {"type": "object"},
            "group_b": {"type": "object"},
            "tests": {"type": "array", "items": {"type": "string"}, "default": ["ks", "chi2"]}
        },
        "required": ["columns"]
    }'::jsonb,
    '{
        "type": "object",
        "properties": {
            "test_results": {"type": "array"},
            "significant_differences": {"type": "array"},
            "visualization_data": {"type": "object"}
        }
    }'::jsonb,
    true
);


-- ============================================================================
-- SECTION 5: SEED DATA - TOOL DEPENDENCIES
-- ============================================================================

-- Define which tools can consume output from other tools
-- These enable the Planner to build valid execution DAGs

-- Get tool IDs for dependency setup
DO $$
DECLARE
    v_causal_effect_id UUID;
    v_refutation_id UUID;
    v_sensitivity_id UUID;
    v_cate_id UUID;
    v_segment_ranker_id UUID;
    v_gap_calc_id UUID;
    v_roi_est_id UUID;
    v_power_calc_id UUID;
    v_counterfactual_id UUID;
    v_risk_scorer_id UUID;
    v_propensity_id UUID;
BEGIN
    -- Get tool IDs
    SELECT tool_id INTO v_causal_effect_id FROM tool_registry WHERE name = 'causal_effect_estimator';
    SELECT tool_id INTO v_refutation_id FROM tool_registry WHERE name = 'refutation_runner';
    SELECT tool_id INTO v_sensitivity_id FROM tool_registry WHERE name = 'sensitivity_analyzer';
    SELECT tool_id INTO v_cate_id FROM tool_registry WHERE name = 'cate_analyzer';
    SELECT tool_id INTO v_segment_ranker_id FROM tool_registry WHERE name = 'segment_ranker';
    SELECT tool_id INTO v_gap_calc_id FROM tool_registry WHERE name = 'gap_calculator';
    SELECT tool_id INTO v_roi_est_id FROM tool_registry WHERE name = 'roi_estimator';
    SELECT tool_id INTO v_power_calc_id FROM tool_registry WHERE name = 'power_calculator';
    SELECT tool_id INTO v_counterfactual_id FROM tool_registry WHERE name = 'counterfactual_simulator';
    SELECT tool_id INTO v_risk_scorer_id FROM tool_registry WHERE name = 'risk_scorer';
    SELECT tool_id INTO v_propensity_id FROM tool_registry WHERE name = 'propensity_estimator';

    -- Insert dependencies
    -- refutation_runner consumes from causal_effect_estimator
    INSERT INTO tool_dependencies (consumer_tool_id, producer_tool_id, output_field, input_field)
    VALUES (v_refutation_id, v_causal_effect_id, NULL, 'causal_result');

    -- sensitivity_analyzer consumes from causal_effect_estimator
    INSERT INTO tool_dependencies (consumer_tool_id, producer_tool_id, output_field, input_field)
    VALUES (v_sensitivity_id, v_causal_effect_id, NULL, 'causal_result');

    -- cate_analyzer can consume from causal_effect_estimator (optional)
    INSERT INTO tool_dependencies (consumer_tool_id, producer_tool_id, output_field, input_field)
    VALUES (v_cate_id, v_causal_effect_id, NULL, 'causal_result');

    -- segment_ranker consumes from cate_analyzer
    INSERT INTO tool_dependencies (consumer_tool_id, producer_tool_id, output_field, input_field)
    VALUES (v_segment_ranker_id, v_cate_id, NULL, 'cate_result');

    -- roi_estimator consumes from gap_calculator
    INSERT INTO tool_dependencies (consumer_tool_id, producer_tool_id, output_field, input_field)
    VALUES (v_roi_est_id, v_gap_calc_id, NULL, 'gap_result');

    -- power_calculator can consume from causal_effect_estimator
    INSERT INTO tool_dependencies (consumer_tool_id, producer_tool_id, output_field, input_field)
    VALUES (v_power_calc_id, v_causal_effect_id, 'ate', 'effect_size');

    -- power_calculator can also consume from cate_analyzer
    INSERT INTO tool_dependencies (consumer_tool_id, producer_tool_id, output_field, input_field)
    VALUES (v_power_calc_id, v_cate_id, 'effect_by_segment', 'effect_size');

    -- counterfactual_simulator can consume from multiple sources
    INSERT INTO tool_dependencies (consumer_tool_id, producer_tool_id, output_field, input_field)
    VALUES (v_counterfactual_id, v_causal_effect_id, NULL, 'causal_model');

    INSERT INTO tool_dependencies (consumer_tool_id, producer_tool_id, output_field, input_field)
    VALUES (v_counterfactual_id, v_cate_id, 'high_responders', 'target_population');

    INSERT INTO tool_dependencies (consumer_tool_id, producer_tool_id, output_field, input_field)
    VALUES (v_counterfactual_id, v_gap_calc_id, 'bottom_performers', 'target_population');

END $$;


-- ============================================================================
-- SECTION 6: FUNCTIONS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Function: update_tool_registry_metrics
-- Purpose: Aggregate tool_performance into tool_registry baselines
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_tool_registry_metrics()
RETURNS void AS $$
BEGIN
    UPDATE tool_registry tr
    SET 
        avg_latency_ms = subq.avg_latency,
        success_rate = subq.success_rate,
        updated_at = NOW()
    FROM (
        SELECT 
            tool_id,
            AVG(latency_ms) AS avg_latency,
            COUNT(*) FILTER (WHERE success = true)::FLOAT / 
                NULLIF(COUNT(*), 0) AS success_rate
        FROM tool_performance
        WHERE executed_at > NOW() - INTERVAL '7 days'
        GROUP BY tool_id
    ) subq
    WHERE tr.tool_id = subq.tool_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_tool_registry_metrics IS 'Updates tool registry with aggregated performance metrics from last 7 days';


-- ----------------------------------------------------------------------------
-- Function: find_similar_compositions
-- Purpose: Find similar past compositions for plan optimization
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION find_similar_compositions(
    p_query_embedding vector(1536),
    p_limit INTEGER DEFAULT 5,
    p_min_similarity FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    episode_id UUID,
    composition_id VARCHAR(100),
    query_text TEXT,
    tool_plan JSONB,
    success BOOLEAN,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ce.episode_id,
        ce.composition_id,
        ce.query_text,
        ce.tool_plan,
        ce.success,
        1 - (ce.query_embedding <=> p_query_embedding) AS similarity
    FROM composer_episodes ce
    WHERE ce.query_embedding IS NOT NULL
      AND ce.status = 'COMPLETED'
      AND ce.success = true
      AND 1 - (ce.query_embedding <=> p_query_embedding) >= p_min_similarity
    ORDER BY ce.query_embedding <=> p_query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION find_similar_compositions IS 'Finds similar successful compositions for plan optimization using vector similarity';


-- ----------------------------------------------------------------------------
-- Function: get_tool_execution_order
-- Purpose: Topologically sort tools based on dependencies
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_tool_execution_order(
    p_tool_names TEXT[]
)
RETURNS TABLE (
    execution_order INTEGER,
    tool_name VARCHAR(100),
    depends_on TEXT[]
) AS $$
WITH RECURSIVE tool_deps AS (
    -- Base case: tools with no dependencies
    SELECT 
        tr.name::VARCHAR(100) AS tool_name,
        0 AS depth,
        ARRAY[]::TEXT[] AS depends_on
    FROM tool_registry tr
    WHERE tr.name = ANY(p_tool_names)
      AND NOT EXISTS (
          SELECT 1 FROM tool_dependencies td
          JOIN tool_registry producer ON td.producer_tool_id = producer.tool_id
          WHERE td.consumer_tool_id = tr.tool_id
            AND producer.name = ANY(p_tool_names)
      )
    
    UNION ALL
    
    -- Recursive case: tools depending on already-processed tools
    SELECT 
        tr.name::VARCHAR(100),
        td.depth + 1,
        array_agg(DISTINCT producer.name)::TEXT[]
    FROM tool_registry tr
    JOIN tool_dependencies dep ON tr.tool_id = dep.consumer_tool_id
    JOIN tool_registry producer ON dep.producer_tool_id = producer.tool_id
    JOIN tool_deps td ON producer.name::VARCHAR(100) = td.tool_name
    WHERE tr.name = ANY(p_tool_names)
      AND producer.name = ANY(p_tool_names)
    GROUP BY tr.name, td.depth
)
SELECT 
    ROW_NUMBER() OVER (ORDER BY depth, tool_name)::INTEGER AS execution_order,
    tool_name,
    depends_on
FROM tool_deps
ORDER BY depth, tool_name;
$$ LANGUAGE sql;

COMMENT ON FUNCTION get_tool_execution_order IS 'Returns topologically sorted execution order for given tools';


-- ============================================================================
-- SECTION 7: TRIGGERS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Trigger: Update tool_registry.updated_at on modification
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION trigger_update_tool_registry_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_tool_registry_updated
    BEFORE UPDATE ON tool_registry
    FOR EACH ROW
    EXECUTE FUNCTION trigger_update_tool_registry_timestamp();


-- ----------------------------------------------------------------------------
-- Trigger: Log tool performance when composition step completes
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION trigger_log_step_performance()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'COMPLETED' AND OLD.status != 'COMPLETED' THEN
        INSERT INTO tool_performance (
            tool_id,
            tool_name,
            latency_ms,
            success,
            composition_id,
            step_id,
            called_by
        )
        SELECT 
            NEW.tool_id,
            NEW.tool_name,
            NEW.latency_ms,
            true,
            ce.composition_id,
            NEW.step_id,
            'composer'
        FROM composer_episodes ce
        WHERE ce.episode_id = NEW.episode_id;
    ELSIF NEW.status = 'FAILED' AND OLD.status != 'FAILED' THEN
        INSERT INTO tool_performance (
            tool_id,
            tool_name,
            latency_ms,
            success,
            error_type,
            composition_id,
            step_id,
            called_by
        )
        SELECT 
            NEW.tool_id,
            NEW.tool_name,
            COALESCE(NEW.latency_ms, 0),
            false,
            SUBSTRING(NEW.error_message FROM 1 FOR 100),
            ce.composition_id,
            NEW.step_id,
            'composer'
        FROM composer_episodes ce
        WHERE ce.episode_id = NEW.episode_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_log_step_performance
    AFTER UPDATE ON composition_steps
    FOR EACH ROW
    EXECUTE FUNCTION trigger_log_step_performance();


-- ============================================================================
-- SECTION 8: GRANTS (adjust for your roles)
-- ============================================================================

-- Application role grants
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO e2i_app;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO e2i_app;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO e2i_app;

-- Read-only role for dashboards
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO e2i_readonly;


-- ============================================================================
-- SECTION 9: COMMENTS SUMMARY
-- ============================================================================

COMMENT ON SCHEMA public IS 'E2I Causal Analytics V4.2 - Tool Composer & Orchestrator Classifier';

-- Add table count to schema comment for tracking
-- Current count: 28 (V4.1) + 6 (V4.2) = 34 tables


-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- 
-- Post-migration steps:
-- 1. Run: SELECT update_tool_registry_metrics(); -- Initialize metrics
-- 2. Verify: SELECT * FROM v_tool_reliability; -- Check tool setup
-- 3. Test: SELECT * FROM get_tool_execution_order(ARRAY['causal_effect_estimator', 'cate_analyzer', 'counterfactual_simulator']);
--
-- ============================================================================
