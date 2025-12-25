-- ============================================================================
-- HPO Pattern Memory Schema
-- Version: 1.0.0
-- Created: 2025-12-25
--
-- Purpose: Store successful hyperparameter optimization patterns for
-- warm-starting future HPO runs. Integrates with procedural_memories table.
--
-- Usage:
--   - Store patterns after successful HPO runs
--   - Retrieve similar patterns to warm-start new HPO studies
--   - Track pattern effectiveness over time
-- ============================================================================

-- Add hpo_pattern to procedure_type enum if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum
        WHERE enumlabel = 'hpo_pattern'
        AND enumtypid = 'procedure_type'::regtype
    ) THEN
        ALTER TYPE procedure_type ADD VALUE 'hpo_pattern';
    END IF;
END $$;

-- ============================================================================
-- HPO PATTERNS TABLE (extends procedural_memories with HPO-specific fields)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_hpo_patterns (
    -- Primary key (references procedural_memories)
    pattern_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    procedure_id UUID REFERENCES procedural_memories(procedure_id) ON DELETE CASCADE,

    -- HPO Context
    algorithm_name VARCHAR(100) NOT NULL,
    problem_type VARCHAR(50) NOT NULL,  -- binary_classification, multiclass, regression

    -- Search Space Definition
    search_space JSONB NOT NULL,  -- The hyperparameter search space used

    -- Best Hyperparameters Found
    best_hyperparameters JSONB NOT NULL,
    best_value FLOAT NOT NULL,  -- Best objective value achieved
    optimization_metric VARCHAR(50) NOT NULL,  -- roc_auc, f1, rmse, etc.

    -- Dataset Characteristics (for similarity matching)
    n_samples INTEGER,
    n_features INTEGER,
    n_classes INTEGER,  -- For classification
    class_balance FLOAT,  -- For binary classification (minority class ratio)
    feature_types JSONB,  -- {numeric: 10, categorical: 5, ...}

    -- HPO Run Details
    n_trials INTEGER NOT NULL,
    n_completed INTEGER NOT NULL,
    n_pruned INTEGER DEFAULT 0,
    duration_seconds FLOAT,
    study_name VARCHAR(200),

    -- Effectiveness Tracking
    times_used_as_warmstart INTEGER DEFAULT 0,
    warmstart_improvement_avg FLOAT,  -- Average improvement when used

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_problem_type CHECK (
        problem_type IN ('binary_classification', 'multiclass_classification', 'regression')
    )
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_hpo_patterns_algorithm
    ON ml_hpo_patterns(algorithm_name);
CREATE INDEX IF NOT EXISTS idx_hpo_patterns_problem_type
    ON ml_hpo_patterns(problem_type);
CREATE INDEX IF NOT EXISTS idx_hpo_patterns_metric
    ON ml_hpo_patterns(optimization_metric);
CREATE INDEX IF NOT EXISTS idx_hpo_patterns_created
    ON ml_hpo_patterns(created_at DESC);

-- Composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_hpo_patterns_algo_problem
    ON ml_hpo_patterns(algorithm_name, problem_type);

-- ============================================================================
-- FUNCTIONS FOR HPO PATTERN MANAGEMENT
-- ============================================================================

-- Function to find similar HPO patterns for warm-starting
CREATE OR REPLACE FUNCTION find_similar_hpo_patterns(
    p_algorithm_name VARCHAR(100),
    p_problem_type VARCHAR(50),
    p_n_samples INTEGER DEFAULT NULL,
    p_n_features INTEGER DEFAULT NULL,
    p_metric VARCHAR(50) DEFAULT NULL,
    p_limit INTEGER DEFAULT 5
)
RETURNS TABLE (
    pattern_id UUID,
    algorithm_name VARCHAR(100),
    problem_type VARCHAR(50),
    best_hyperparameters JSONB,
    best_value FLOAT,
    optimization_metric VARCHAR(50),
    n_samples INTEGER,
    n_features INTEGER,
    similarity_score FLOAT,
    times_used INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        hp.pattern_id,
        hp.algorithm_name,
        hp.problem_type,
        hp.best_hyperparameters,
        hp.best_value,
        hp.optimization_metric,
        hp.n_samples,
        hp.n_features,
        -- Calculate similarity score based on dataset characteristics
        (
            CASE WHEN hp.algorithm_name = p_algorithm_name THEN 0.4 ELSE 0 END +
            CASE WHEN hp.problem_type = p_problem_type THEN 0.3 ELSE 0 END +
            CASE WHEN p_metric IS NULL OR hp.optimization_metric = p_metric THEN 0.1 ELSE 0 END +
            -- Sample size similarity (0.1 max)
            CASE
                WHEN p_n_samples IS NULL THEN 0.05
                WHEN hp.n_samples IS NULL THEN 0.02
                ELSE 0.1 * (1.0 - LEAST(1.0, ABS(LOG10(hp.n_samples::FLOAT + 1) - LOG10(p_n_samples::FLOAT + 1)) / 3.0))
            END +
            -- Feature count similarity (0.1 max)
            CASE
                WHEN p_n_features IS NULL THEN 0.05
                WHEN hp.n_features IS NULL THEN 0.02
                ELSE 0.1 * (1.0 - LEAST(1.0, ABS(hp.n_features - p_n_features)::FLOAT / GREATEST(hp.n_features, p_n_features, 1)::FLOAT))
            END
        )::FLOAT AS similarity_score,
        hp.times_used_as_warmstart
    FROM ml_hpo_patterns hp
    WHERE hp.algorithm_name = p_algorithm_name
      AND hp.problem_type = p_problem_type
    ORDER BY similarity_score DESC, hp.best_value DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to record warmstart usage
CREATE OR REPLACE FUNCTION record_hpo_warmstart_usage(
    p_pattern_id UUID,
    p_improvement FLOAT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    UPDATE ml_hpo_patterns
    SET
        times_used_as_warmstart = times_used_as_warmstart + 1,
        warmstart_improvement_avg = CASE
            WHEN warmstart_improvement_avg IS NULL THEN p_improvement
            WHEN p_improvement IS NULL THEN warmstart_improvement_avg
            ELSE (warmstart_improvement_avg * (times_used_as_warmstart - 1) + p_improvement) / times_used_as_warmstart
        END,
        updated_at = NOW()
    WHERE pattern_id = p_pattern_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE ml_hpo_patterns IS
    'Stores successful HPO patterns for warm-starting future optimization runs';

COMMENT ON COLUMN ml_hpo_patterns.search_space IS
    'The hyperparameter search space definition used for optimization';

COMMENT ON COLUMN ml_hpo_patterns.best_hyperparameters IS
    'The best hyperparameters found during optimization';

COMMENT ON COLUMN ml_hpo_patterns.warmstart_improvement_avg IS
    'Average improvement in objective value when this pattern is used for warm-starting';

COMMENT ON FUNCTION find_similar_hpo_patterns IS
    'Find similar HPO patterns based on algorithm, problem type, and dataset characteristics';
