-- Migration 007: MLOps Tables for E2I ML Foundation
-- Extends V3 schema with 8 new tables for ML lifecycle management
-- Created: December 4, 2025

-- ============================================================================
-- ENUM UPDATES
-- ============================================================================

-- Add new ML Foundation agents to agent_name_enum
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'scope_definer';
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'data_preparer';
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'model_selector';
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'model_trainer';
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'feature_analyzer';
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'model_deployer';
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'observability_connector';

-- Create agent tier enum
DO $$ BEGIN
    CREATE TYPE agent_tier_enum AS ENUM (
        'ml_foundation',  -- Tier 0: New ML lifecycle agents
        'coordination',   -- Tier 1: Orchestrator
        'causal_analytics', -- Tier 2: Causal Impact, Gap Analyzer, Het Optimizer
        'monitoring',     -- Tier 3: Drift Monitor, Experiment Designer, Health Score
        'ml_predictions', -- Tier 4: Prediction Synthesizer, Resource Optimizer
        'self_improvement' -- Tier 5: Explainer, Feedback Learner
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create model stage enum for registry
DO $$ BEGIN
    CREATE TYPE model_stage_enum AS ENUM (
        'development',
        'staging',
        'shadow',
        'production',
        'archived',
        'deprecated'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create data quality status enum
DO $$ BEGIN
    CREATE TYPE dq_status_enum AS ENUM (
        'passed',
        'failed',
        'warning',
        'skipped'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create deployment status enum
DO $$ BEGIN
    CREATE TYPE deployment_status_enum AS ENUM (
        'pending',
        'deploying',
        'active',
        'draining',
        'rolled_back',
        'failed'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- ============================================================================
-- TABLE 1: ml_experiments
-- MLflow experiment metadata, links to model versions
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_name VARCHAR(255) NOT NULL,
    mlflow_experiment_id VARCHAR(100) UNIQUE,
    description TEXT,
    
    -- Scope definition
    prediction_target VARCHAR(100) NOT NULL,
    target_population TEXT,
    observation_window_days INTEGER,
    prediction_horizon_days INTEGER,
    
    -- Success criteria
    minimum_auc DECIMAL(4,3),
    minimum_precision_at_k DECIMAL(4,3),
    maximum_fpr DECIMAL(4,3),
    
    -- Metadata
    brand brand_type,
    region region_type,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- ML split tracking
    data_split data_split_type DEFAULT 'unassigned',
    
    CONSTRAINT valid_auc CHECK (minimum_auc BETWEEN 0.5 AND 1.0),
    CONSTRAINT valid_precision CHECK (minimum_precision_at_k BETWEEN 0 AND 1.0)
);

CREATE INDEX idx_ml_experiments_name ON ml_experiments(experiment_name);
CREATE INDEX idx_ml_experiments_mlflow_id ON ml_experiments(mlflow_experiment_id);
CREATE INDEX idx_ml_experiments_brand ON ml_experiments(brand);

-- ============================================================================
-- TABLE 2: ml_model_registry
-- Model versions, stages, artifacts, performance metrics
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_model_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES ml_experiments(id),
    
    -- Model identification
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    mlflow_run_id VARCHAR(100),
    mlflow_model_uri TEXT,
    
    -- Model metadata
    algorithm VARCHAR(100) NOT NULL,
    hyperparameters JSONB DEFAULT '{}',
    feature_count INTEGER,
    training_samples INTEGER,
    
    -- Performance metrics
    auc DECIMAL(5,4),
    pr_auc DECIMAL(5,4),
    brier_score DECIMAL(5,4),
    calibration_slope DECIMAL(5,4),
    
    -- Fairness metrics
    fairness_metrics JSONB DEFAULT '{}',
    
    -- Registry status
    stage model_stage_enum DEFAULT 'development',
    is_champion BOOLEAN DEFAULT FALSE,
    
    -- Artifacts
    artifact_path TEXT,
    preprocessing_pipeline_path TEXT,
    
    -- Timestamps
    trained_at TIMESTAMP WITH TIME ZONE,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    promoted_at TIMESTAMP WITH TIME ZONE,
    
    -- ML split tracking
    data_split data_split_type DEFAULT 'train',
    
    CONSTRAINT unique_model_version UNIQUE(model_name, model_version),
    CONSTRAINT valid_model_auc CHECK (auc BETWEEN 0 AND 1.0)
);

CREATE INDEX idx_ml_registry_experiment ON ml_model_registry(experiment_id);
CREATE INDEX idx_ml_registry_name ON ml_model_registry(model_name);
CREATE INDEX idx_ml_registry_stage ON ml_model_registry(stage);
CREATE INDEX idx_ml_registry_champion ON ml_model_registry(is_champion) WHERE is_champion = TRUE;

-- ============================================================================
-- TABLE 3: ml_training_runs
-- Training job records with hyperparameters, metrics, duration
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES ml_experiments(id),
    model_registry_id UUID REFERENCES ml_model_registry(id),
    
    -- Run identification
    run_name VARCHAR(255),
    mlflow_run_id VARCHAR(100) UNIQUE,
    
    -- Configuration
    algorithm VARCHAR(100) NOT NULL,
    hyperparameters JSONB DEFAULT '{}',
    
    -- Data info
    training_samples INTEGER NOT NULL,
    validation_samples INTEGER,
    test_samples INTEGER,
    feature_names JSONB DEFAULT '[]',
    
    -- Metrics per split
    train_metrics JSONB DEFAULT '{}',
    validation_metrics JSONB DEFAULT '{}',
    test_metrics JSONB DEFAULT '{}',
    
    -- Run status
    status VARCHAR(50) DEFAULT 'running',
    error_message TEXT,
    
    -- Timing
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    
    -- Resources
    compute_type VARCHAR(50),
    gpu_used BOOLEAN DEFAULT FALSE,
    
    -- ML split tracking (which split was used for training)
    data_split data_split_type DEFAULT 'train',
    
    -- Optuna specific
    optuna_study_name VARCHAR(255),
    optuna_trial_number INTEGER,
    is_best_trial BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_training_runs_experiment ON ml_training_runs(experiment_id);
CREATE INDEX idx_training_runs_mlflow ON ml_training_runs(mlflow_run_id);
CREATE INDEX idx_training_runs_status ON ml_training_runs(status);
CREATE INDEX idx_training_runs_best ON ml_training_runs(is_best_trial) WHERE is_best_trial = TRUE;

-- ============================================================================
-- TABLE 4: ml_feature_store
-- Feature definitions, versioning, point-in-time values
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_feature_store (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Feature identification
    feature_name VARCHAR(255) NOT NULL,
    feature_version VARCHAR(50) DEFAULT '1.0',
    feature_group VARCHAR(100),
    
    -- Feature metadata
    description TEXT,
    data_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50), -- 'patient', 'hcp', 'treatment'
    
    -- Computation
    computation_sql TEXT,
    computation_python TEXT,
    source_tables JSONB DEFAULT '[]',
    
    -- Statistics (computed on train split only)
    train_statistics JSONB DEFAULT '{}',
    -- {mean, std, min, max, null_rate, unique_count, histogram}
    
    -- Feature importance (from SHAP)
    global_importance DECIMAL(6,5),
    importance_rank INTEGER,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- ML split tracking
    data_split data_split_type DEFAULT 'train',
    
    CONSTRAINT unique_feature_version UNIQUE(feature_name, feature_version)
);

CREATE INDEX idx_feature_store_name ON ml_feature_store(feature_name);
CREATE INDEX idx_feature_store_group ON ml_feature_store(feature_group);
CREATE INDEX idx_feature_store_entity ON ml_feature_store(entity_type);
CREATE INDEX idx_feature_store_importance ON ml_feature_store(importance_rank);

-- ============================================================================
-- TABLE 5: ml_data_quality_reports
-- Great Expectations results, validation timestamps
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_data_quality_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Report identification
    report_name VARCHAR(255) NOT NULL,
    expectation_suite_name VARCHAR(255),
    
    -- Scope
    table_name VARCHAR(100),
    brand brand_type,
    region region_type,
    
    -- Results
    overall_status dq_status_enum NOT NULL,
    expectations_evaluated INTEGER NOT NULL,
    expectations_passed INTEGER NOT NULL,
    expectations_failed INTEGER NOT NULL,
    success_rate DECIMAL(5,4),
    
    -- Detailed results
    failed_expectations JSONB DEFAULT '[]',
    -- [{expectation_type, column, kwargs, observed_value, success}]
    
    -- Quality dimensions (from E2I framework)
    completeness_score DECIMAL(5,4),
    validity_score DECIMAL(5,4),
    uniqueness_score DECIMAL(5,4),
    consistency_score DECIMAL(5,4),
    timeliness_score DECIMAL(5,4),
    accuracy_score DECIMAL(5,4),
    
    -- Leakage detection
    leakage_detected BOOLEAN DEFAULT FALSE,
    leakage_details JSONB DEFAULT '{}',
    
    -- Metadata
    run_by VARCHAR(100),
    run_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    duration_seconds INTEGER,
    
    -- ML split tracking
    data_split data_split_type NOT NULL,
    
    -- Link to training run if applicable
    training_run_id UUID REFERENCES ml_training_runs(id)
);

CREATE INDEX idx_dq_reports_suite ON ml_data_quality_reports(expectation_suite_name);
CREATE INDEX idx_dq_reports_table ON ml_data_quality_reports(table_name);
CREATE INDEX idx_dq_reports_status ON ml_data_quality_reports(overall_status);
CREATE INDEX idx_dq_reports_split ON ml_data_quality_reports(data_split);
CREATE INDEX idx_dq_reports_run ON ml_data_quality_reports(training_run_id);

-- ============================================================================
-- TABLE 6: ml_shap_analyses
-- Global/local SHAP values, feature importance, segment analysis
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_shap_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_registry_id UUID REFERENCES ml_model_registry(id) NOT NULL,
    
    -- Analysis scope
    analysis_type VARCHAR(50) NOT NULL, -- 'global', 'local', 'segment'
    segment_name VARCHAR(100), -- For segment analysis: 'midwest', 'high_volume', etc.
    segment_filter JSONB, -- Filter criteria used
    
    -- Global SHAP values
    global_importance JSONB DEFAULT '{}',
    -- {feature_name: mean_abs_shap_value}
    
    -- Feature interactions
    top_interactions JSONB DEFAULT '[]',
    -- [{feature_1, feature_2, interaction_strength}]
    
    -- For local explanations (patient-level)
    entity_type VARCHAR(50), -- 'patient', 'hcp'
    entity_id VARCHAR(100),
    local_shap_values JSONB DEFAULT '{}',
    -- {feature_name: shap_value}
    prediction_value DECIMAL(6,5),
    base_value DECIMAL(6,5),
    
    -- Natural language explanation (from Hybrid LLM node)
    natural_language_explanation TEXT,
    key_drivers TEXT[], -- Top 5 features as bullets
    actionable_features TEXT[], -- Features that can be influenced
    
    -- Metadata
    sample_size INTEGER,
    computation_method VARCHAR(50), -- 'TreeExplainer', 'KernelExplainer'
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    computation_duration_seconds INTEGER,
    
    -- ML split tracking
    data_split data_split_type DEFAULT 'test'
);

CREATE INDEX idx_shap_model ON ml_shap_analyses(model_registry_id);
CREATE INDEX idx_shap_type ON ml_shap_analyses(analysis_type);
CREATE INDEX idx_shap_segment ON ml_shap_analyses(segment_name);
CREATE INDEX idx_shap_entity ON ml_shap_analyses(entity_type, entity_id);

-- ============================================================================
-- TABLE 7: ml_deployments
-- Deployment history, endpoints, rollback records
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_deployments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_registry_id UUID REFERENCES ml_model_registry(id) NOT NULL,
    
    -- Deployment identification
    deployment_name VARCHAR(255) NOT NULL,
    environment VARCHAR(50) NOT NULL, -- 'development', 'staging', 'production'
    
    -- Endpoint info
    endpoint_name VARCHAR(255),
    endpoint_url TEXT,
    
    -- Status
    status deployment_status_enum DEFAULT 'pending',
    
    -- Deployment metadata
    deployed_by VARCHAR(100),
    deployment_config JSONB DEFAULT '{}',
    
    -- Performance during deployment
    shadow_metrics JSONB DEFAULT '{}', -- Metrics from shadow mode
    production_metrics JSONB DEFAULT '{}', -- Metrics after promotion
    
    -- Rollback info
    previous_deployment_id UUID REFERENCES ml_deployments(id),
    rollback_reason TEXT,
    rolled_back_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deployed_at TIMESTAMP WITH TIME ZONE,
    deactivated_at TIMESTAMP WITH TIME ZONE,
    
    -- SLA tracking
    latency_p50_ms INTEGER,
    latency_p95_ms INTEGER,
    latency_p99_ms INTEGER,
    error_rate DECIMAL(5,4)
);

CREATE INDEX idx_deployments_model ON ml_deployments(model_registry_id);
CREATE INDEX idx_deployments_env ON ml_deployments(environment);
CREATE INDEX idx_deployments_status ON ml_deployments(status);
CREATE INDEX idx_deployments_active ON ml_deployments(status) WHERE status = 'active';

-- ============================================================================
-- TABLE 8: ml_observability_spans
-- Opik span summaries, latency metrics, error rates
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_observability_spans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Span identification
    trace_id VARCHAR(100) NOT NULL,
    span_id VARCHAR(100) NOT NULL,
    parent_span_id VARCHAR(100),
    
    -- Context
    agent_name agent_name_enum NOT NULL,
    agent_tier agent_tier_enum NOT NULL,
    operation_type VARCHAR(100), -- 'inference', 'training', 'shap_computation', etc.
    
    -- Timing
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ended_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    
    -- LLM specific (for Hybrid/Deep agents)
    model_name VARCHAR(100),
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    
    -- Status
    status VARCHAR(50) DEFAULT 'success', -- 'success', 'error', 'timeout'
    error_type VARCHAR(100),
    error_message TEXT,
    
    -- Fallback tracking
    fallback_used BOOLEAN DEFAULT FALSE,
    fallback_chain JSONB DEFAULT '[]', -- Sequence of models tried
    
    -- Custom attributes
    attributes JSONB DEFAULT '{}',
    
    -- Links to other entities
    experiment_id UUID REFERENCES ml_experiments(id),
    training_run_id UUID REFERENCES ml_training_runs(id),
    deployment_id UUID REFERENCES ml_deployments(id),
    
    -- User context
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    
    CONSTRAINT unique_span UNIQUE(trace_id, span_id)
);

CREATE INDEX idx_obs_spans_trace ON ml_observability_spans(trace_id);
CREATE INDEX idx_obs_spans_agent ON ml_observability_spans(agent_name);
CREATE INDEX idx_obs_spans_tier ON ml_observability_spans(agent_tier);
CREATE INDEX idx_obs_spans_time ON ml_observability_spans(started_at);
CREATE INDEX idx_obs_spans_status ON ml_observability_spans(status);
CREATE INDEX idx_obs_spans_fallback ON ml_observability_spans(fallback_used) WHERE fallback_used = TRUE;

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Active champion models by experiment
CREATE OR REPLACE VIEW v_champion_models AS
SELECT 
    e.experiment_name,
    e.prediction_target,
    e.brand,
    e.region,
    m.model_name,
    m.model_version,
    m.algorithm,
    m.auc,
    m.pr_auc,
    m.stage,
    m.trained_at
FROM ml_model_registry m
JOIN ml_experiments e ON m.experiment_id = e.id
WHERE m.is_champion = TRUE
AND m.stage IN ('production', 'staging');

-- View: Latest data quality by table
CREATE OR REPLACE VIEW v_latest_data_quality AS
SELECT DISTINCT ON (table_name, data_split)
    table_name,
    data_split,
    overall_status,
    success_rate,
    completeness_score,
    validity_score,
    leakage_detected,
    run_at
FROM ml_data_quality_reports
ORDER BY table_name, data_split, run_at DESC;

-- View: Agent latency summary
CREATE OR REPLACE VIEW v_agent_latency_summary AS
SELECT 
    agent_name,
    agent_tier,
    COUNT(*) as total_spans,
    AVG(duration_ms) as avg_duration_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) as p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99_ms,
    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as error_rate,
    SUM(CASE WHEN fallback_used THEN 1 ELSE 0 END)::DECIMAL / COUNT(*) as fallback_rate,
    SUM(total_tokens) as total_tokens_used
FROM ml_observability_spans
WHERE started_at > NOW() - INTERVAL '24 hours'
GROUP BY agent_name, agent_tier;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Get latest model for experiment
CREATE OR REPLACE FUNCTION get_latest_model(p_experiment_name VARCHAR)
RETURNS TABLE (
    model_id UUID,
    model_name VARCHAR,
    model_version VARCHAR,
    auc DECIMAL,
    stage model_stage_enum
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        m.id,
        m.model_name,
        m.model_version,
        m.auc,
        m.stage
    FROM ml_model_registry m
    JOIN ml_experiments e ON m.experiment_id = e.id
    WHERE e.experiment_name = p_experiment_name
    ORDER BY m.registered_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function: Check if data quality passed for training
CREATE OR REPLACE FUNCTION check_data_quality_for_training(p_table_name VARCHAR)
RETURNS BOOLEAN AS $$
DECLARE
    v_status dq_status_enum;
    v_leakage BOOLEAN;
BEGIN
    SELECT overall_status, leakage_detected
    INTO v_status, v_leakage
    FROM ml_data_quality_reports
    WHERE table_name = p_table_name
    AND data_split = 'train'
    ORDER BY run_at DESC
    LIMIT 1;
    
    IF v_status IS NULL THEN
        RETURN FALSE; -- No QC report found
    END IF;
    
    RETURN v_status = 'passed' AND NOT v_leakage;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger: Update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_ml_experiments_updated
    BEFORE UPDATE ON ml_experiments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER tr_ml_feature_store_updated
    BEFORE UPDATE ON ml_feature_store
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Trigger: Ensure only one champion per experiment
CREATE OR REPLACE FUNCTION ensure_single_champion()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_champion = TRUE THEN
        UPDATE ml_model_registry
        SET is_champion = FALSE
        WHERE experiment_id = NEW.experiment_id
        AND id != NEW.id
        AND is_champion = TRUE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_single_champion
    BEFORE INSERT OR UPDATE ON ml_model_registry
    FOR EACH ROW EXECUTE FUNCTION ensure_single_champion();

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE ml_experiments IS 'MLflow experiments with scope definitions and success criteria';
COMMENT ON TABLE ml_model_registry IS 'Model versions with performance metrics and deployment stages';
COMMENT ON TABLE ml_training_runs IS 'Individual training runs with hyperparameters and metrics';
COMMENT ON TABLE ml_feature_store IS 'Feature definitions with statistics computed on train split only';
COMMENT ON TABLE ml_data_quality_reports IS 'Great Expectations validation results with E2I quality dimensions';
COMMENT ON TABLE ml_shap_analyses IS 'SHAP values for global, local, and segment-level explainability';
COMMENT ON TABLE ml_deployments IS 'Deployment history with shadow mode and rollback tracking';
COMMENT ON TABLE ml_observability_spans IS 'Opik span data for agent latency and error monitoring';

-- ============================================================================
-- GRANTS (adjust based on your roles)
-- ============================================================================
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO e2i_app;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO e2i_app;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO e2i_app;

COMMIT;
