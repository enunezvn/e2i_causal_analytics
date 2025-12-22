-- ============================================================================
-- Migration 016: Hyperparameter Optimization (HPO) Studies Tables
--
-- Adds tables for tracking Optuna hyperparameter optimization studies
-- and their trial history for the E2I ML pipeline.
--
-- Created: 2025-12-22
-- ============================================================================

-- ============================================================================
-- TABLE: ml_hpo_studies
-- Tracks Optuna study metadata and summary results
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_hpo_studies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Study identification
    study_name VARCHAR(255) NOT NULL UNIQUE,
    experiment_id UUID REFERENCES ml_experiments(id),

    -- Algorithm being optimized
    algorithm_name VARCHAR(100) NOT NULL,
    problem_type VARCHAR(50) NOT NULL,

    -- Optimization configuration
    direction VARCHAR(20) DEFAULT 'maximize',  -- 'maximize' or 'minimize'
    sampler_name VARCHAR(50) DEFAULT 'TPESampler',
    pruner_name VARCHAR(50) DEFAULT 'MedianPruner',
    metric VARCHAR(50) NOT NULL,  -- 'roc_auc', 'rmse', etc.

    -- Search space definition
    search_space JSONB NOT NULL DEFAULT '{}',
    fixed_params JSONB DEFAULT '{}',

    -- Trial statistics
    n_trials INTEGER DEFAULT 0,
    n_completed INTEGER DEFAULT 0,
    n_pruned INTEGER DEFAULT 0,
    n_failed INTEGER DEFAULT 0,

    -- Best trial results
    best_trial_number INTEGER,
    best_value DECIMAL(10, 6),
    best_params JSONB DEFAULT '{}',

    -- Timing
    duration_seconds DECIMAL(10, 2),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Status
    status VARCHAR(50) DEFAULT 'running',  -- 'running', 'completed', 'failed', 'pruned'
    error_message TEXT,

    -- Metadata
    created_by VARCHAR(100) DEFAULT 'optuna_optimizer',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_hpo_studies_experiment ON ml_hpo_studies(experiment_id);
CREATE INDEX idx_hpo_studies_algorithm ON ml_hpo_studies(algorithm_name);
CREATE INDEX idx_hpo_studies_status ON ml_hpo_studies(status);
CREATE INDEX idx_hpo_studies_study_name ON ml_hpo_studies(study_name);

-- ============================================================================
-- TABLE: ml_hpo_trials
-- Tracks individual Optuna trials within a study
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_hpo_trials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_id UUID REFERENCES ml_hpo_studies(id) ON DELETE CASCADE,

    -- Trial identification
    trial_number INTEGER NOT NULL,

    -- Trial state
    state VARCHAR(50) NOT NULL,  -- 'COMPLETE', 'PRUNED', 'FAIL', 'WAITING', 'RUNNING'

    -- Hyperparameters sampled for this trial
    params JSONB NOT NULL DEFAULT '{}',

    -- Trial result
    value DECIMAL(10, 6),  -- Objective function value

    -- Intermediate values (for pruning)
    intermediate_values JSONB DEFAULT '{}',  -- {step: value}

    -- Timing
    datetime_start TIMESTAMP WITH TIME ZONE,
    datetime_complete TIMESTAMP WITH TIME ZONE,
    duration_seconds DECIMAL(10, 3),

    -- User and system attributes
    user_attrs JSONB DEFAULT '{}',
    system_attrs JSONB DEFAULT '{}',

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT unique_trial_in_study UNIQUE(study_id, trial_number)
);

CREATE INDEX idx_hpo_trials_study ON ml_hpo_trials(study_id);
CREATE INDEX idx_hpo_trials_state ON ml_hpo_trials(state);
CREATE INDEX idx_hpo_trials_value ON ml_hpo_trials(value);

-- ============================================================================
-- Function: Update updated_at timestamp
-- ============================================================================
CREATE OR REPLACE FUNCTION update_hpo_studies_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_hpo_studies_updated_at
    BEFORE UPDATE ON ml_hpo_studies
    FOR EACH ROW
    EXECUTE FUNCTION update_hpo_studies_updated_at();

-- ============================================================================
-- Comments
-- ============================================================================
COMMENT ON TABLE ml_hpo_studies IS
'Tracks Optuna hyperparameter optimization studies including configuration, search space, and best results';

COMMENT ON TABLE ml_hpo_trials IS
'Individual trial records within an HPO study, including parameters and objective values';

COMMENT ON COLUMN ml_hpo_studies.search_space IS
'E2I format search space definition: {"param": {"type": "int|float|categorical", "low": X, "high": Y, "log": bool, "choices": []}}';

COMMENT ON COLUMN ml_hpo_studies.direction IS
'Optimization direction: "maximize" for metrics like roc_auc, "minimize" for metrics like rmse';

COMMENT ON COLUMN ml_hpo_trials.intermediate_values IS
'Intermediate objective values reported during trial for pruning decisions';
