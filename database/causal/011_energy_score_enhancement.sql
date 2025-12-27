-- Migration: 011_energy_score_enhancement
-- Description: Add energy score tracking for causal estimator selection
-- Version: V4.2
-- Date: 2025-12-24
-- 
-- This migration adds columns to track energy score-based estimator selection,
-- enabling analysis of which estimators perform best across different contexts.

-- ============================================================================
-- PART 1: Extend ml_experiments with energy score metadata
-- ============================================================================

-- Add energy score columns to ml_experiments
ALTER TABLE ml_experiments 
ADD COLUMN IF NOT EXISTS energy_score_enabled BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS selection_strategy VARCHAR(50) DEFAULT 'first_success';

COMMENT ON COLUMN ml_experiments.energy_score_enabled IS 'Whether energy score selection was used for this experiment';
COMMENT ON COLUMN ml_experiments.selection_strategy IS 'Estimator selection strategy: first_success, best_energy, ensemble';

-- ============================================================================
-- PART 2: New table for estimator evaluation results
-- ============================================================================

CREATE TABLE IF NOT EXISTS estimator_evaluations (
    evaluation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Link to experiment
    experiment_id UUID NOT NULL REFERENCES ml_experiments(id) ON DELETE CASCADE,
    
    -- Estimator identification
    estimator_type VARCHAR(50) NOT NULL,
    estimator_priority INTEGER NOT NULL DEFAULT 1,
    
    -- Execution status
    success BOOLEAN NOT NULL,
    error_message TEXT,
    error_type VARCHAR(100),
    
    -- Effect estimates
    ate DOUBLE PRECISION,
    ate_std DOUBLE PRECISION,
    ate_ci_lower DOUBLE PRECISION,
    ate_ci_upper DOUBLE PRECISION,
    
    -- Energy score components
    energy_score DOUBLE PRECISION,
    treatment_balance_score DOUBLE PRECISION,
    outcome_fit_score DOUBLE PRECISION,
    propensity_calibration DOUBLE PRECISION,
    
    -- Energy score confidence interval (from bootstrap)
    energy_ci_lower DOUBLE PRECISION,
    energy_ci_upper DOUBLE PRECISION,
    energy_bootstrap_std DOUBLE PRECISION,
    
    -- Sample info
    n_samples INTEGER,
    n_treated INTEGER,
    n_control INTEGER,
    
    -- Timing
    estimation_time_ms DOUBLE PRECISION,
    energy_computation_time_ms DOUBLE PRECISION,
    
    -- Was this estimator selected?
    was_selected BOOLEAN DEFAULT FALSE,
    selection_reason TEXT,
    
    -- Detailed config and results (JSON)
    estimator_params JSONB DEFAULT '{}',
    energy_details JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_estimator_type CHECK (estimator_type IN (
        'causal_forest', 'linear_dml', 'dml_learner', 'drlearner',
        'ortho_forest', 's_learner', 't_learner', 'x_learner', 'ols'
    )),
    CONSTRAINT valid_energy_score CHECK (energy_score IS NULL OR energy_score >= 0)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_estimator_evals_experiment 
    ON estimator_evaluations(experiment_id);

CREATE INDEX IF NOT EXISTS idx_estimator_evals_type 
    ON estimator_evaluations(estimator_type);

CREATE INDEX IF NOT EXISTS idx_estimator_evals_selected 
    ON estimator_evaluations(was_selected) WHERE was_selected = TRUE;

CREATE INDEX IF NOT EXISTS idx_estimator_evals_energy 
    ON estimator_evaluations(energy_score) WHERE energy_score IS NOT NULL;

COMMENT ON TABLE estimator_evaluations IS 
    'Stores evaluation results for each estimator in the selection chain, including energy scores';

-- ============================================================================
-- PART 3: Aggregate view for estimator performance analysis
-- ============================================================================

CREATE OR REPLACE VIEW v_estimator_performance AS
SELECT 
    estimator_type,
    COUNT(*) AS total_evaluations,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) AS successful_runs,
    ROUND(100.0 * SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*), 2) AS success_rate_pct,
    SUM(CASE WHEN was_selected THEN 1 ELSE 0 END) AS times_selected,
    ROUND(100.0 * SUM(CASE WHEN was_selected THEN 1 ELSE 0 END) / 
          NULLIF(SUM(CASE WHEN success THEN 1 ELSE 0 END), 0), 2) AS selection_rate_pct,
    ROUND(AVG(energy_score)::numeric, 4) AS avg_energy_score,
    ROUND(STDDEV(energy_score)::numeric, 4) AS std_energy_score,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY energy_score)::numeric, 4) AS median_energy_score,
    ROUND(AVG(ate)::numeric, 4) AS avg_ate,
    ROUND(AVG(estimation_time_ms)::numeric, 2) AS avg_estimation_time_ms
FROM estimator_evaluations
WHERE created_at > NOW() - INTERVAL '90 days'
GROUP BY estimator_type
ORDER BY times_selected DESC;

COMMENT ON VIEW v_estimator_performance IS 
    'Aggregated performance metrics for each estimator type over the last 90 days';

-- ============================================================================
-- PART 4: View for energy score trends over time
-- ============================================================================

CREATE OR REPLACE VIEW v_energy_score_trends AS
SELECT 
    DATE_TRUNC('week', ee.created_at) AS week,
    ee.estimator_type,
    COUNT(*) AS n_evaluations,
    ROUND(AVG(ee.energy_score)::numeric, 4) AS avg_energy_score,
    ROUND(MIN(ee.energy_score)::numeric, 4) AS min_energy_score,
    ROUND(MAX(ee.energy_score)::numeric, 4) AS max_energy_score,
    SUM(CASE WHEN ee.was_selected THEN 1 ELSE 0 END) AS times_selected
FROM estimator_evaluations ee
WHERE ee.success = TRUE
  AND ee.energy_score IS NOT NULL
  AND ee.created_at > NOW() - INTERVAL '6 months'
GROUP BY DATE_TRUNC('week', ee.created_at), ee.estimator_type
ORDER BY week DESC, estimator_type;

COMMENT ON VIEW v_energy_score_trends IS 
    'Weekly trends in energy scores by estimator type for the last 6 months';

-- ============================================================================
-- PART 5: View for selection comparison (energy vs legacy)
-- ============================================================================

CREATE OR REPLACE VIEW v_selection_comparison AS
WITH ranked_evals AS (
    SELECT 
        experiment_id,
        estimator_type,
        energy_score,
        success,
        was_selected,
        ROW_NUMBER() OVER (PARTITION BY experiment_id ORDER BY estimator_priority) AS priority_rank,
        ROW_NUMBER() OVER (PARTITION BY experiment_id ORDER BY energy_score NULLS LAST) AS energy_rank
    FROM estimator_evaluations
    WHERE success = TRUE
),
comparisons AS (
    SELECT 
        experiment_id,
        -- What legacy (first_success) would have selected
        MAX(CASE WHEN priority_rank = 1 THEN estimator_type END) AS legacy_selection,
        MAX(CASE WHEN priority_rank = 1 THEN energy_score END) AS legacy_energy_score,
        -- What energy score selection chose
        MAX(CASE WHEN energy_rank = 1 THEN estimator_type END) AS energy_selection,
        MAX(CASE WHEN energy_rank = 1 THEN energy_score END) AS best_energy_score,
        -- Actual selection
        MAX(CASE WHEN was_selected THEN estimator_type END) AS actual_selection
    FROM ranked_evals
    GROUP BY experiment_id
)
SELECT 
    COUNT(*) AS total_experiments,
    SUM(CASE WHEN legacy_selection = energy_selection THEN 1 ELSE 0 END) AS same_selection,
    SUM(CASE WHEN legacy_selection != energy_selection THEN 1 ELSE 0 END) AS different_selection,
    ROUND(100.0 * SUM(CASE WHEN legacy_selection != energy_selection THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_improved,
    ROUND(AVG(legacy_energy_score - best_energy_score)::numeric, 4) AS avg_energy_improvement
FROM comparisons
WHERE legacy_selection IS NOT NULL AND energy_selection IS NOT NULL;

COMMENT ON VIEW v_selection_comparison IS 
    'Compares legacy first-success selection vs energy score selection to measure improvement';

-- ============================================================================
-- PART 6: Function to log estimator evaluation
-- ============================================================================

CREATE OR REPLACE FUNCTION log_estimator_evaluation(
    p_experiment_id UUID,
    p_estimator_type VARCHAR(50),
    p_priority INTEGER,
    p_success BOOLEAN,
    p_ate DOUBLE PRECISION DEFAULT NULL,
    p_ate_std DOUBLE PRECISION DEFAULT NULL,
    p_energy_score DOUBLE PRECISION DEFAULT NULL,
    p_treatment_balance DOUBLE PRECISION DEFAULT NULL,
    p_outcome_fit DOUBLE PRECISION DEFAULT NULL,
    p_propensity_cal DOUBLE PRECISION DEFAULT NULL,
    p_was_selected BOOLEAN DEFAULT FALSE,
    p_selection_reason TEXT DEFAULT NULL,
    p_estimation_time_ms DOUBLE PRECISION DEFAULT NULL,
    p_n_samples INTEGER DEFAULT NULL,
    p_n_treated INTEGER DEFAULT NULL,
    p_n_control INTEGER DEFAULT NULL,
    p_error_message TEXT DEFAULT NULL,
    p_estimator_params JSONB DEFAULT '{}'
) RETURNS UUID AS $$
DECLARE
    v_evaluation_id UUID;
BEGIN
    INSERT INTO estimator_evaluations (
        experiment_id,
        estimator_type,
        estimator_priority,
        success,
        ate,
        ate_std,
        energy_score,
        treatment_balance_score,
        outcome_fit_score,
        propensity_calibration,
        was_selected,
        selection_reason,
        estimation_time_ms,
        n_samples,
        n_treated,
        n_control,
        error_message,
        estimator_params
    ) VALUES (
        p_experiment_id,
        p_estimator_type,
        p_priority,
        p_success,
        p_ate,
        p_ate_std,
        p_energy_score,
        p_treatment_balance,
        p_outcome_fit,
        p_propensity_cal,
        p_was_selected,
        p_selection_reason,
        p_estimation_time_ms,
        p_n_samples,
        p_n_treated,
        p_n_control,
        p_error_message,
        p_estimator_params
    ) RETURNING evaluation_id INTO v_evaluation_id;
    
    RETURN v_evaluation_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION log_estimator_evaluation IS 
    'Helper function to log individual estimator evaluation results';

-- ============================================================================
-- PART 7: Update domain vocabulary with new ENUMs
-- ============================================================================

-- Note: This should be added to domain_vocabulary.yaml v3.2.0
-- 
-- estimator_types:
--   - causal_forest
--   - linear_dml
--   - dml_learner
--   - drlearner
--   - ortho_forest
--   - s_learner
--   - t_learner
--   - x_learner
--   - ols
--
-- selection_strategies:
--   - first_success
--   - best_energy
--   - ensemble
--
-- energy_score_variants:
--   - standard
--   - weighted
--   - doubly_robust

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
