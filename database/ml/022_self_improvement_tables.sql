-- ============================================================================
-- SELF-IMPROVEMENT SCHEMA (DELTA MIGRATION)
-- ============================================================================
-- Migration: 022_self_improvement_tables.sql
-- Version: 1.0.0
-- Date: 2025-12-26
-- Purpose: Add self-improvement infrastructure for RAGAS-Opik integration
--
-- This is a DELTA migration - only adds NEW tables/columns not in existing schema.
--
-- NEW ENUMs (2):
--   - improvement_type
--   - improvement_priority
--
-- NEW COLUMNS on existing tables:
--   - learning_signals: ragas_scores, rubric_scores, combined_score, etc.
--
-- NEW TABLES (5):
--   - evaluation_results
--   - retrieval_configurations
--   - prompt_configurations
--   - improvement_actions
--   - experiment_knowledge_store
--
-- Prerequisites:
--   - learning_signals table exists (001_agentic_memory_schema_v1.3.sql)
--   - cognitive_cycles table exists (001_agentic_memory_schema_v1.3.sql)
-- ============================================================================

-- ============================================================================
-- SECTION 1: NEW ENUMS
-- ============================================================================

-- Improvement action types
DO $$ BEGIN
    CREATE TYPE improvement_type AS ENUM (
        'retrieval',      -- Tune k, chunks, RRF weights
        'prompt',         -- Improve system prompts, few-shot
        'workflow',       -- Agent routing, tier priority
        'none'            -- No improvement needed
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Improvement priority levels
DO $$ BEGIN
    CREATE TYPE improvement_priority AS ENUM (
        'critical',       -- Combined score < 0.40
        'high',           -- Combined score 0.40-0.54
        'medium',         -- Combined score 0.55-0.69
        'low',            -- Combined score 0.70-0.84
        'none'            -- Combined score >= 0.85
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

COMMENT ON TYPE improvement_type IS 'Types of self-improvement actions';
COMMENT ON TYPE improvement_priority IS 'Priority levels for improvement actions';


-- ============================================================================
-- SECTION 2: ALTER EXISTING TABLES
-- ============================================================================

-- Add new columns to learning_signals for RAGAS/Rubric integration
-- Using DO block to handle existing columns gracefully
DO $$
BEGIN
    -- RAGAS scores JSON column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'ragas_scores') THEN
        ALTER TABLE learning_signals ADD COLUMN ragas_scores JSONB DEFAULT '{}'::jsonb;
        COMMENT ON COLUMN learning_signals.ragas_scores IS
            'RAGAS metric scores: faithfulness, answer_relevancy, context_precision, context_recall';
    END IF;

    -- RAGAS weighted aggregate
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'ragas_weighted') THEN
        ALTER TABLE learning_signals ADD COLUMN ragas_weighted FLOAT
            CHECK (ragas_weighted >= 0 AND ragas_weighted <= 1);
    END IF;

    -- Rubric scores JSON column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'rubric_scores') THEN
        ALTER TABLE learning_signals ADD COLUMN rubric_scores JSONB DEFAULT '{}'::jsonb;
        COMMENT ON COLUMN learning_signals.rubric_scores IS
            'Domain rubric scores: causal_validity, actionability, evidence_chain, etc.';
    END IF;

    -- Rubric total (1-5 scale)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'rubric_total') THEN
        ALTER TABLE learning_signals ADD COLUMN rubric_total FLOAT
            CHECK (rubric_total >= 0 AND rubric_total <= 5);
    END IF;

    -- Rubric weighted
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'rubric_weighted') THEN
        ALTER TABLE learning_signals ADD COLUMN rubric_weighted FLOAT
            CHECK (rubric_weighted >= 0 AND rubric_weighted <= 5);
    END IF;

    -- Combined score (0-1 scale)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'combined_score') THEN
        ALTER TABLE learning_signals ADD COLUMN combined_score FLOAT
            CHECK (combined_score >= 0 AND combined_score <= 1);
        COMMENT ON COLUMN learning_signals.combined_score IS
            'Combined RAGAS + Rubric score: (ragas * 0.4) + (rubric_normalized * 0.6)';
    END IF;

    -- Improvement type
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'improvement_type') THEN
        ALTER TABLE learning_signals ADD COLUMN improvement_type improvement_type DEFAULT 'none';
    END IF;

    -- Improvement priority
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'improvement_priority') THEN
        ALTER TABLE learning_signals ADD COLUMN improvement_priority improvement_priority DEFAULT 'none';
    END IF;

    -- Improvement applied flag
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'improvement_applied') THEN
        ALTER TABLE learning_signals ADD COLUMN improvement_applied BOOLEAN DEFAULT FALSE;
    END IF;

    -- Improvement details JSON
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'improvement_details') THEN
        ALTER TABLE learning_signals ADD COLUMN improvement_details JSONB DEFAULT '{}'::jsonb;
    END IF;

    -- Retrieved chunks for RAGAS evaluation
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'retrieved_chunks') THEN
        ALTER TABLE learning_signals ADD COLUMN retrieved_chunks JSONB DEFAULT '[]'::jsonb;
    END IF;

    -- Retrieval scores
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'retrieval_scores') THEN
        ALTER TABLE learning_signals ADD COLUMN retrieval_scores JSONB DEFAULT '[]'::jsonb;
    END IF;

    -- Processed timestamp
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'learning_signals' AND column_name = 'processed_at') THEN
        ALTER TABLE learning_signals ADD COLUMN processed_at TIMESTAMPTZ;
    END IF;
END $$;

-- Add indexes for new columns
CREATE INDEX IF NOT EXISTS idx_learning_combined_score
    ON learning_signals(combined_score);
CREATE INDEX IF NOT EXISTS idx_learning_improvement_type
    ON learning_signals(improvement_type) WHERE improvement_type != 'none';
CREATE INDEX IF NOT EXISTS idx_learning_pending
    ON learning_signals(improvement_applied) WHERE improvement_applied = FALSE AND improvement_type != 'none';
CREATE INDEX IF NOT EXISTS idx_learning_ragas
    ON learning_signals USING GIN (ragas_scores);
CREATE INDEX IF NOT EXISTS idx_learning_rubric
    ON learning_signals USING GIN (rubric_scores);


-- ============================================================================
-- SECTION 3: NEW TABLES
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 3.1 Evaluation Results (Detailed RAGAS + Rubric Evaluations)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evaluation_results (
    evaluation_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Links
    learning_signal_id  UUID REFERENCES learning_signals(signal_id) ON DELETE CASCADE,
    cognitive_cycle_id  UUID REFERENCES cognitive_cycles(cycle_id),

    -- Query and response
    query               TEXT NOT NULL,
    response            TEXT NOT NULL,
    ground_truth        TEXT,  -- Optional ground truth for answer_correctness

    -- Retrieved context
    retrieved_contexts  JSONB NOT NULL DEFAULT '[]'::jsonb,
    context_count       INTEGER,

    -- Individual RAGAS metrics (0-1 scale)
    faithfulness        FLOAT CHECK (faithfulness >= 0 AND faithfulness <= 1),
    answer_relevancy    FLOAT CHECK (answer_relevancy >= 0 AND answer_relevancy <= 1),
    context_precision   FLOAT CHECK (context_precision >= 0 AND context_precision <= 1),
    context_recall      FLOAT CHECK (context_recall >= 0 AND context_recall <= 1),
    answer_correctness  FLOAT CHECK (answer_correctness >= 0 AND answer_correctness <= 1),

    -- RAGAS weighted aggregate
    ragas_aggregate     FLOAT CHECK (ragas_aggregate >= 0 AND ragas_aggregate <= 1),

    -- Individual Rubric metrics (1-5 scale)
    causal_validity     FLOAT CHECK (causal_validity >= 1 AND causal_validity <= 5),
    actionability       FLOAT CHECK (actionability >= 1 AND actionability <= 5),
    evidence_chain      FLOAT CHECK (evidence_chain >= 1 AND evidence_chain <= 5),
    regulatory_awareness FLOAT CHECK (regulatory_awareness >= 1 AND regulatory_awareness <= 5),
    uncertainty_communication FLOAT CHECK (uncertainty_communication >= 1 AND uncertainty_communication <= 5),

    -- Rubric weighted aggregate
    rubric_aggregate    FLOAT CHECK (rubric_aggregate >= 1 AND rubric_aggregate <= 5),

    -- Evaluation metadata
    evaluation_model    VARCHAR(100),
    evaluation_duration_ms INTEGER,

    -- Timestamps
    created_at          TIMESTAMPTZ DEFAULT now()
);

COMMENT ON TABLE evaluation_results IS 'Detailed RAGAS and rubric evaluation results per query-response pair';

CREATE INDEX IF NOT EXISTS idx_eval_learning_signal ON evaluation_results(learning_signal_id);
CREATE INDEX IF NOT EXISTS idx_eval_cognitive_cycle ON evaluation_results(cognitive_cycle_id);
CREATE INDEX IF NOT EXISTS idx_eval_created ON evaluation_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_ragas ON evaluation_results(ragas_aggregate);
CREATE INDEX IF NOT EXISTS idx_eval_rubric ON evaluation_results(rubric_aggregate);
CREATE INDEX IF NOT EXISTS idx_eval_faithfulness_low ON evaluation_results(faithfulness) WHERE faithfulness < 0.7;


-- ----------------------------------------------------------------------------
-- 3.2 Retrieval Configurations (Tuning History)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS retrieval_configurations (
    config_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Configuration name and version
    config_name         VARCHAR(100) NOT NULL,
    config_version      INTEGER DEFAULT 1,

    -- Retrieval parameters
    k_value             INTEGER NOT NULL DEFAULT 10,
    chunk_size          INTEGER NOT NULL DEFAULT 512,
    chunk_overlap       INTEGER DEFAULT 50,

    -- Hybrid RAG weights (must sum to 1.0)
    vector_weight       FLOAT NOT NULL DEFAULT 0.4,
    fulltext_weight     FLOAT NOT NULL DEFAULT 0.3,
    graph_weight        FLOAT NOT NULL DEFAULT 0.3,

    -- Reranking configuration
    rerank_enabled      BOOLEAN DEFAULT TRUE,
    rerank_model        VARCHAR(100) DEFAULT 'cross-encoder/ms-marco-MiniLM-L-12-v2',
    rerank_top_k        INTEGER DEFAULT 5,

    -- Performance metrics
    avg_context_precision FLOAT,
    avg_context_recall    FLOAT,
    avg_faithfulness      FLOAT,
    evaluation_count      INTEGER DEFAULT 0,

    -- Status
    is_active           BOOLEAN DEFAULT FALSE,
    is_baseline         BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at          TIMESTAMPTZ DEFAULT now(),
    activated_at        TIMESTAMPTZ,
    deactivated_at      TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT chk_weights_sum CHECK (
        ABS(vector_weight + fulltext_weight + graph_weight - 1.0) < 0.001
    )
);

COMMENT ON TABLE retrieval_configurations IS 'Retrieval parameter configurations with performance tracking';

CREATE INDEX IF NOT EXISTS idx_retrieval_active ON retrieval_configurations(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_retrieval_performance ON retrieval_configurations(avg_context_precision DESC);


-- ----------------------------------------------------------------------------
-- 3.3 Prompt Configurations (Prompt Tuning History)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prompt_configurations (
    prompt_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identification
    prompt_name         VARCHAR(100) NOT NULL,
    prompt_version      INTEGER DEFAULT 1,
    agent_name          VARCHAR(100),

    -- Prompt content
    system_prompt       TEXT NOT NULL,
    few_shot_examples   JSONB DEFAULT '[]'::jsonb,

    -- Performance metrics
    avg_answer_relevancy  FLOAT,
    avg_rubric_score      FLOAT,
    avg_combined_score    FLOAT,
    evaluation_count      INTEGER DEFAULT 0,

    -- Status
    is_active           BOOLEAN DEFAULT FALSE,
    is_baseline         BOOLEAN DEFAULT FALSE,

    -- Change tracking
    parent_prompt_id    UUID REFERENCES prompt_configurations(prompt_id),
    change_reason       TEXT,

    -- Timestamps
    created_at          TIMESTAMPTZ DEFAULT now(),
    activated_at        TIMESTAMPTZ,
    deactivated_at      TIMESTAMPTZ
);

COMMENT ON TABLE prompt_configurations IS 'Prompt versions with performance tracking for systematic optimization';

CREATE INDEX IF NOT EXISTS idx_prompt_active ON prompt_configurations(agent_name, is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_prompt_agent ON prompt_configurations(agent_name);
CREATE INDEX IF NOT EXISTS idx_prompt_performance ON prompt_configurations(avg_combined_score DESC);


-- ----------------------------------------------------------------------------
-- 3.4 Improvement Actions (Applied Improvements Log)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS improvement_actions (
    action_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Trigger
    learning_signal_id  UUID REFERENCES learning_signals(signal_id),
    trigger_score       FLOAT,

    -- Action details
    improvement_type    improvement_type NOT NULL,
    action_description  TEXT NOT NULL,

    -- Before/After state
    before_config       JSONB NOT NULL,
    after_config        JSONB NOT NULL,

    -- Related configuration changes
    retrieval_config_id UUID REFERENCES retrieval_configurations(config_id),
    prompt_config_id    UUID REFERENCES prompt_configurations(prompt_id),

    -- Outcome tracking
    outcome_measured    BOOLEAN DEFAULT FALSE,
    outcome_improved    BOOLEAN,
    outcome_delta       FLOAT,

    -- Status
    status              VARCHAR(20) DEFAULT 'applied',
    rolled_back_at      TIMESTAMPTZ,
    rollback_reason     TEXT,

    -- Timestamps
    created_at          TIMESTAMPTZ DEFAULT now(),
    measured_at         TIMESTAMPTZ
);

COMMENT ON TABLE improvement_actions IS 'Audit log of all self-improvement actions with outcome tracking';

CREATE INDEX IF NOT EXISTS idx_improvement_signal ON improvement_actions(learning_signal_id);
CREATE INDEX IF NOT EXISTS idx_improvement_type ON improvement_actions(improvement_type);
CREATE INDEX IF NOT EXISTS idx_improvement_status ON improvement_actions(status);
CREATE INDEX IF NOT EXISTS idx_improvement_created ON improvement_actions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_improvement_pending ON improvement_actions(outcome_measured) WHERE outcome_measured = FALSE;


-- ----------------------------------------------------------------------------
-- 3.5 Experiment Knowledge Store (Organizational Learning)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiment_knowledge_store (
    knowledge_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source
    source_type         VARCHAR(50) NOT NULL,
    source_id           UUID,

    -- Knowledge content
    lesson_type         VARCHAR(50) NOT NULL,
    title               VARCHAR(200) NOT NULL,
    description         TEXT NOT NULL,

    -- Structured knowledge
    knowledge_json      JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Applicability
    applicable_brands   TEXT[],
    applicable_contexts TEXT[],

    -- Embedding for similarity search
    embedding           vector(1536),

    -- Confidence and validation
    confidence_level    FLOAT CHECK (confidence_level >= 0 AND confidence_level <= 1),
    validated           BOOLEAN DEFAULT FALSE,
    validation_date     DATE,

    -- Usage tracking
    times_applied       INTEGER DEFAULT 0,
    success_rate        FLOAT,

    -- Timestamps
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now(),

    -- Soft delete
    is_active           BOOLEAN DEFAULT TRUE
);

COMMENT ON TABLE experiment_knowledge_store IS 'Organizational knowledge from experiments for continuous improvement';

CREATE INDEX IF NOT EXISTS idx_knowledge_embedding ON experiment_knowledge_store
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX IF NOT EXISTS idx_knowledge_type ON experiment_knowledge_store(lesson_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_source ON experiment_knowledge_store(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_active ON experiment_knowledge_store(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_knowledge_brands ON experiment_knowledge_store USING GIN (applicable_brands);


-- ============================================================================
-- SECTION 4: VIEWS
-- ============================================================================

-- RAGAS Performance Trends View
CREATE OR REPLACE VIEW v_ragas_performance_trends AS
SELECT
    DATE_TRUNC('day', created_at) AS evaluation_date,
    COUNT(*) AS evaluation_count,
    AVG(faithfulness) AS avg_faithfulness,
    AVG(answer_relevancy) AS avg_answer_relevancy,
    AVG(context_precision) AS avg_context_precision,
    AVG(context_recall) AS avg_context_recall,
    AVG(answer_correctness) AS avg_answer_correctness,
    AVG(ragas_aggregate) AS avg_ragas_score,
    AVG(rubric_aggregate) AS avg_rubric_score,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ragas_aggregate) AS median_ragas,
    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY ragas_aggregate) AS p10_ragas
FROM evaluation_results
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY evaluation_date DESC;

COMMENT ON VIEW v_ragas_performance_trends IS 'Daily RAGAS metric trends for monitoring';


-- Improvement Actions Summary View
CREATE OR REPLACE VIEW v_improvement_summary AS
SELECT
    improvement_type,
    DATE_TRUNC('week', created_at) AS week,
    COUNT(*) AS action_count,
    COUNT(*) FILTER (WHERE outcome_improved = TRUE) AS improved_count,
    COUNT(*) FILTER (WHERE outcome_improved = FALSE) AS regressed_count,
    AVG(outcome_delta) FILTER (WHERE outcome_measured = TRUE) AS avg_delta,
    COUNT(*) FILTER (WHERE status = 'rolled_back') AS rollback_count
FROM improvement_actions
WHERE created_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY improvement_type, DATE_TRUNC('week', created_at)
ORDER BY week DESC, improvement_type;

COMMENT ON VIEW v_improvement_summary IS 'Weekly summary of self-improvement actions and outcomes';


-- Learning Signal Quality Distribution View
CREATE OR REPLACE VIEW v_learning_signal_distribution AS
SELECT
    agent_name,
    improvement_type,
    improvement_priority,
    COUNT(*) AS signal_count,
    AVG(combined_score) AS avg_combined_score,
    AVG((ragas_scores->>'faithfulness')::float) AS avg_faithfulness,
    AVG(rubric_total) AS avg_rubric_total,
    COUNT(*) FILTER (WHERE improvement_applied = TRUE) AS applied_count,
    COUNT(*) FILTER (WHERE improvement_applied = FALSE AND improvement_type != 'none') AS pending_count
FROM learning_signals
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY agent_name, improvement_type, improvement_priority
ORDER BY signal_count DESC;

COMMENT ON VIEW v_learning_signal_distribution IS 'Distribution of learning signals by agent and improvement type';


-- ============================================================================
-- SECTION 5: HELPER FUNCTIONS
-- ============================================================================

-- Calculate Combined Score Function
CREATE OR REPLACE FUNCTION calculate_combined_score(
    p_ragas_scores JSONB,
    p_rubric_total FLOAT,
    p_ragas_weight FLOAT DEFAULT 0.4,
    p_rubric_weight FLOAT DEFAULT 0.6
) RETURNS FLOAT AS $$
DECLARE
    v_ragas_weighted FLOAT;
    v_rubric_normalized FLOAT;
    v_combined FLOAT;
BEGIN
    -- Calculate weighted RAGAS score (assumes 0-1 scale scores)
    v_ragas_weighted := (
        COALESCE((p_ragas_scores->>'faithfulness')::float, 0) * 0.25 +
        COALESCE((p_ragas_scores->>'answer_relevancy')::float, 0) * 0.20 +
        COALESCE((p_ragas_scores->>'context_precision')::float, 0) * 0.20 +
        COALESCE((p_ragas_scores->>'context_recall')::float, 0) * 0.20 +
        COALESCE((p_ragas_scores->>'answer_correctness')::float, 0) * 0.15
    );

    -- Normalize rubric to 0-1 scale (from 1-5)
    v_rubric_normalized := COALESCE((p_rubric_total - 1) / 4.0, 0);

    -- Calculate combined score
    v_combined := (v_ragas_weighted * p_ragas_weight) + (v_rubric_normalized * p_rubric_weight);

    RETURN ROUND(v_combined::numeric, 4);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION calculate_combined_score IS 'Calculate combined RAGAS + Rubric score with configurable weights';


-- Determine Improvement Type Function
CREATE OR REPLACE FUNCTION determine_improvement_type(
    p_ragas_weighted FLOAT,
    p_rubric_normalized FLOAT,
    p_ragas_threshold FLOAT DEFAULT 0.7,
    p_rubric_threshold FLOAT DEFAULT 0.7
) RETURNS improvement_type AS $$
BEGIN
    IF p_ragas_weighted >= p_ragas_threshold AND p_rubric_normalized >= p_rubric_threshold THEN
        RETURN 'none';
    ELSIF p_ragas_weighted < p_ragas_threshold AND p_rubric_normalized >= p_rubric_threshold THEN
        RETURN 'retrieval';
    ELSIF p_ragas_weighted >= p_ragas_threshold AND p_rubric_normalized < p_rubric_threshold THEN
        RETURN 'prompt';
    ELSE
        RETURN 'workflow';
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION determine_improvement_type IS 'Route to appropriate improvement action based on RAGAS vs Rubric scores';


-- Determine Improvement Priority Function
CREATE OR REPLACE FUNCTION determine_improvement_priority(
    p_combined_score FLOAT
) RETURNS improvement_priority AS $$
BEGIN
    IF p_combined_score >= 0.85 THEN
        RETURN 'none';
    ELSIF p_combined_score >= 0.70 THEN
        RETURN 'low';
    ELSIF p_combined_score >= 0.55 THEN
        RETURN 'medium';
    ELSIF p_combined_score >= 0.40 THEN
        RETURN 'high';
    ELSE
        RETURN 'critical';
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION determine_improvement_priority IS 'Determine improvement priority based on combined score thresholds';


-- Update Learning Signal with Evaluation Results
CREATE OR REPLACE FUNCTION update_learning_signal_evaluation(
    p_signal_id UUID,
    p_ragas_scores JSONB,
    p_rubric_scores JSONB,
    p_rubric_total FLOAT
) RETURNS VOID AS $$
DECLARE
    v_ragas_weighted FLOAT;
    v_rubric_normalized FLOAT;
    v_combined_score FLOAT;
    v_improvement_type improvement_type;
    v_improvement_priority improvement_priority;
BEGIN
    -- Calculate scores
    v_combined_score := calculate_combined_score(p_ragas_scores, p_rubric_total);
    v_ragas_weighted := (
        COALESCE((p_ragas_scores->>'faithfulness')::float, 0) * 0.25 +
        COALESCE((p_ragas_scores->>'answer_relevancy')::float, 0) * 0.20 +
        COALESCE((p_ragas_scores->>'context_precision')::float, 0) * 0.20 +
        COALESCE((p_ragas_scores->>'context_recall')::float, 0) * 0.20 +
        COALESCE((p_ragas_scores->>'answer_correctness')::float, 0) * 0.15
    );
    v_rubric_normalized := (p_rubric_total - 1) / 4.0;

    -- Determine improvement routing
    v_improvement_type := determine_improvement_type(v_ragas_weighted, v_rubric_normalized);
    v_improvement_priority := determine_improvement_priority(v_combined_score);

    -- Update the learning signal
    UPDATE learning_signals SET
        ragas_scores = p_ragas_scores,
        ragas_weighted = v_ragas_weighted,
        rubric_scores = p_rubric_scores,
        rubric_total = p_rubric_total,
        rubric_weighted = p_rubric_total,
        combined_score = v_combined_score,
        improvement_type = v_improvement_type,
        improvement_priority = v_improvement_priority,
        processed_at = now()
    WHERE signal_id = p_signal_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_learning_signal_evaluation IS 'Update learning signal with evaluation results and determine improvement routing';


-- Get Active Retrieval Configuration Function
CREATE OR REPLACE FUNCTION get_active_retrieval_config()
RETURNS TABLE (
    config_id UUID,
    k_value INTEGER,
    chunk_size INTEGER,
    vector_weight FLOAT,
    fulltext_weight FLOAT,
    graph_weight FLOAT,
    rerank_enabled BOOLEAN,
    rerank_model VARCHAR(100),
    rerank_top_k INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        rc.config_id,
        rc.k_value,
        rc.chunk_size,
        rc.vector_weight,
        rc.fulltext_weight,
        rc.graph_weight,
        rc.rerank_enabled,
        rc.rerank_model,
        rc.rerank_top_k
    FROM retrieval_configurations rc
    WHERE rc.is_active = TRUE
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_active_retrieval_config IS 'Get the currently active retrieval configuration';


-- ============================================================================
-- SECTION 6: TRIGGERS
-- ============================================================================

-- Auto-update updated_at for experiment_knowledge_store
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tr_experiment_knowledge_updated ON experiment_knowledge_store;
CREATE TRIGGER tr_experiment_knowledge_updated
    BEFORE UPDATE ON experiment_knowledge_store
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- ============================================================================
-- SECTION 7: INITIAL DATA (BASELINE CONFIGURATION)
-- ============================================================================

-- Insert baseline retrieval configuration (only if not exists)
INSERT INTO retrieval_configurations (
    config_name,
    config_version,
    k_value,
    chunk_size,
    chunk_overlap,
    vector_weight,
    fulltext_weight,
    graph_weight,
    rerank_enabled,
    rerank_model,
    rerank_top_k,
    is_active,
    is_baseline
)
SELECT
    'baseline_hybrid_rag',
    1,
    10,
    512,
    50,
    0.4,
    0.3,
    0.3,
    TRUE,
    'cross-encoder/ms-marco-MiniLM-L-12-v2',
    5,
    TRUE,
    TRUE
WHERE NOT EXISTS (
    SELECT 1 FROM retrieval_configurations WHERE config_name = 'baseline_hybrid_rag'
);


-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
--
-- Summary of created objects:
--
-- NEW ENUMs (2):
--   improvement_type, improvement_priority
--
-- ALTERED TABLES (1):
--   learning_signals - Added 12 new columns for RAGAS/Rubric integration
--
-- NEW TABLES (5):
--   1. evaluation_results        - Detailed RAGAS/Rubric evaluations
--   2. retrieval_configurations  - Retrieval parameter tuning
--   3. prompt_configurations     - Prompt version tracking
--   4. improvement_actions       - Applied improvements log
--   5. experiment_knowledge_store - Organizational learning
--
-- NEW VIEWS (3):
--   v_ragas_performance_trends, v_improvement_summary,
--   v_learning_signal_distribution
--
-- NEW FUNCTIONS (5):
--   calculate_combined_score, determine_improvement_type,
--   determine_improvement_priority, update_learning_signal_evaluation,
--   get_active_retrieval_config
-- ============================================================================
