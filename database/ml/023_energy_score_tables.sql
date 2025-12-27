-- Energy Score Enhancement Tables
-- V4.2: Estimator selection based on energy score quality metrics
-- Created: 2025-12-26

-- =============================================================================
-- 1. ENUMS
-- =============================================================================

-- Estimator types
DO $$ BEGIN
    CREATE TYPE estimator_type AS ENUM (
        'causal_forest',
        'linear_dml',
        'drlearner',
        'ols'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Selection strategies
DO $$ BEGIN
    CREATE TYPE selection_strategy AS ENUM (
        'first_success',
        'best_energy',
        'ensemble'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Quality tiers
DO $$ BEGIN
    CREATE TYPE quality_tier AS ENUM (
        'excellent',
        'good',
        'acceptable',
        'poor',
        'unreliable'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- 2. MAIN TABLE: estimator_evaluations
-- =============================================================================

CREATE TABLE IF NOT EXISTS estimator_evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Context
    query_id TEXT NOT NULL,
    session_id TEXT,
    treatment_var TEXT NOT NULL,
    outcome_var TEXT NOT NULL,

    -- Estimator info
    estimator_type estimator_type NOT NULL,
    selection_strategy selection_strategy NOT NULL DEFAULT 'best_energy',
    was_selected BOOLEAN NOT NULL DEFAULT FALSE,

    -- Energy score components (0-1, lower is better)
    energy_score FLOAT NOT NULL CHECK (energy_score >= 0 AND energy_score <= 1),
    treatment_balance_score FLOAT CHECK (treatment_balance_score >= 0 AND treatment_balance_score <= 1),
    outcome_fit_score FLOAT CHECK (outcome_fit_score >= 0 AND outcome_fit_score <= 1),
    propensity_calibration FLOAT CHECK (propensity_calibration >= 0 AND propensity_calibration <= 1),

    -- Quality assessment
    quality_tier quality_tier NOT NULL,
    energy_score_ci_lower FLOAT,
    energy_score_ci_upper FLOAT,
    bootstrap_std FLOAT,

    -- Estimation results (if successful)
    ate_estimate FLOAT,
    ate_ci_lower FLOAT,
    ate_ci_upper FLOAT,
    standard_error FLOAT,
    p_value FLOAT,

    -- Execution metadata
    succeeded BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    computation_time_ms FLOAT NOT NULL DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Indexes hint
    CONSTRAINT valid_ci CHECK (
        (energy_score_ci_lower IS NULL AND energy_score_ci_upper IS NULL) OR
        (energy_score_ci_lower <= energy_score AND energy_score <= energy_score_ci_upper)
    )
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_estimator_evaluations_query_id
    ON estimator_evaluations(query_id);
CREATE INDEX IF NOT EXISTS idx_estimator_evaluations_session
    ON estimator_evaluations(session_id);
CREATE INDEX IF NOT EXISTS idx_estimator_evaluations_estimator_type
    ON estimator_evaluations(estimator_type);
CREATE INDEX IF NOT EXISTS idx_estimator_evaluations_was_selected
    ON estimator_evaluations(was_selected);
CREATE INDEX IF NOT EXISTS idx_estimator_evaluations_created_at
    ON estimator_evaluations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_estimator_evaluations_quality
    ON estimator_evaluations(quality_tier, energy_score);

-- =============================================================================
-- 3. VIEWS
-- =============================================================================

-- View: Estimator performance summary
CREATE OR REPLACE VIEW v_estimator_performance AS
SELECT
    estimator_type,
    COUNT(*) as total_evaluations,
    SUM(CASE WHEN succeeded THEN 1 ELSE 0 END) as successful_evaluations,
    SUM(CASE WHEN was_selected THEN 1 ELSE 0 END) as times_selected,
    ROUND(AVG(energy_score)::numeric, 4) as avg_energy_score,
    ROUND(STDDEV(energy_score)::numeric, 4) as stddev_energy_score,
    ROUND(MIN(energy_score)::numeric, 4) as min_energy_score,
    ROUND(MAX(energy_score)::numeric, 4) as max_energy_score,
    ROUND(AVG(computation_time_ms)::numeric, 2) as avg_computation_ms,
    -- Quality distribution
    SUM(CASE WHEN quality_tier = 'excellent' THEN 1 ELSE 0 END) as excellent_count,
    SUM(CASE WHEN quality_tier = 'good' THEN 1 ELSE 0 END) as good_count,
    SUM(CASE WHEN quality_tier = 'acceptable' THEN 1 ELSE 0 END) as acceptable_count,
    SUM(CASE WHEN quality_tier = 'poor' THEN 1 ELSE 0 END) as poor_count,
    SUM(CASE WHEN quality_tier = 'unreliable' THEN 1 ELSE 0 END) as unreliable_count
FROM estimator_evaluations
WHERE succeeded = TRUE
GROUP BY estimator_type
ORDER BY avg_energy_score ASC;

-- View: Energy score trends over time
CREATE OR REPLACE VIEW v_energy_score_trends AS
SELECT
    DATE_TRUNC('day', created_at) as date,
    estimator_type,
    COUNT(*) as evaluation_count,
    ROUND(AVG(energy_score)::numeric, 4) as avg_energy_score,
    ROUND(AVG(treatment_balance_score)::numeric, 4) as avg_treatment_balance,
    ROUND(AVG(outcome_fit_score)::numeric, 4) as avg_outcome_fit,
    ROUND(AVG(propensity_calibration)::numeric, 4) as avg_propensity_calibration,
    SUM(CASE WHEN was_selected THEN 1 ELSE 0 END) as selections
FROM estimator_evaluations
WHERE succeeded = TRUE
GROUP BY DATE_TRUNC('day', created_at), estimator_type
ORDER BY date DESC, estimator_type;

-- View: Selection comparison (best vs runner-up)
CREATE OR REPLACE VIEW v_selection_comparison AS
WITH ranked AS (
    SELECT
        query_id,
        estimator_type,
        energy_score,
        was_selected,
        ROW_NUMBER() OVER (PARTITION BY query_id ORDER BY energy_score ASC) as rank
    FROM estimator_evaluations
    WHERE succeeded = TRUE
)
SELECT
    r1.query_id,
    r1.estimator_type as selected_estimator,
    r1.energy_score as selected_score,
    r2.estimator_type as runner_up_estimator,
    r2.energy_score as runner_up_score,
    ROUND((r2.energy_score - r1.energy_score)::numeric, 4) as energy_gap
FROM ranked r1
LEFT JOIN ranked r2 ON r1.query_id = r2.query_id AND r2.rank = 2
WHERE r1.rank = 1
ORDER BY r1.query_id;

-- =============================================================================
-- 4. FUNCTIONS
-- =============================================================================

-- Function: Log estimator evaluation
CREATE OR REPLACE FUNCTION log_estimator_evaluation(
    p_query_id TEXT,
    p_session_id TEXT,
    p_treatment_var TEXT,
    p_outcome_var TEXT,
    p_estimator_type TEXT,
    p_selection_strategy TEXT,
    p_was_selected BOOLEAN,
    p_energy_score FLOAT,
    p_treatment_balance_score FLOAT DEFAULT NULL,
    p_outcome_fit_score FLOAT DEFAULT NULL,
    p_propensity_calibration FLOAT DEFAULT NULL,
    p_quality_tier TEXT DEFAULT 'acceptable',
    p_ate_estimate FLOAT DEFAULT NULL,
    p_ate_ci_lower FLOAT DEFAULT NULL,
    p_ate_ci_upper FLOAT DEFAULT NULL,
    p_standard_error FLOAT DEFAULT NULL,
    p_p_value FLOAT DEFAULT NULL,
    p_succeeded BOOLEAN DEFAULT TRUE,
    p_error_message TEXT DEFAULT NULL,
    p_computation_time_ms FLOAT DEFAULT 0
) RETURNS UUID AS $$
DECLARE
    v_id UUID;
BEGIN
    INSERT INTO estimator_evaluations (
        query_id,
        session_id,
        treatment_var,
        outcome_var,
        estimator_type,
        selection_strategy,
        was_selected,
        energy_score,
        treatment_balance_score,
        outcome_fit_score,
        propensity_calibration,
        quality_tier,
        ate_estimate,
        ate_ci_lower,
        ate_ci_upper,
        standard_error,
        p_value,
        succeeded,
        error_message,
        computation_time_ms
    ) VALUES (
        p_query_id,
        p_session_id,
        p_treatment_var,
        p_outcome_var,
        p_estimator_type::estimator_type,
        p_selection_strategy::selection_strategy,
        p_was_selected,
        p_energy_score,
        p_treatment_balance_score,
        p_outcome_fit_score,
        p_propensity_calibration,
        p_quality_tier::quality_tier,
        p_ate_estimate,
        p_ate_ci_lower,
        p_ate_ci_upper,
        p_standard_error,
        p_p_value,
        p_succeeded,
        p_error_message,
        p_computation_time_ms
    ) RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 5. ROW LEVEL SECURITY (optional - uncomment if needed)
-- =============================================================================

-- ALTER TABLE estimator_evaluations ENABLE ROW LEVEL SECURITY;
--
-- CREATE POLICY estimator_evaluations_select_policy ON estimator_evaluations
--     FOR SELECT USING (true);
--
-- CREATE POLICY estimator_evaluations_insert_policy ON estimator_evaluations
--     FOR INSERT WITH CHECK (true);

COMMENT ON TABLE estimator_evaluations IS 'V4.2 Energy Score Enhancement: Tracks estimator evaluations and energy scores for quality-based selection';
COMMENT ON VIEW v_estimator_performance IS 'Aggregated estimator performance metrics by type';
COMMENT ON VIEW v_energy_score_trends IS 'Daily energy score trends by estimator type';
COMMENT ON VIEW v_selection_comparison IS 'Compares selected estimator with runner-up for each query';
COMMENT ON FUNCTION log_estimator_evaluation IS 'Convenience function to log an estimator evaluation';
