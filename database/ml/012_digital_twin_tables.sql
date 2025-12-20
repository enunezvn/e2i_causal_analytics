-- ============================================
-- Migration: 011_digital_twin_tables.sql
-- Purpose: Digital Twin infrastructure for A/B test pre-screening
-- Version: E2I V4.2.0
-- Dependencies: 010_causal_validation_tables.sql
-- ============================================

-- ============================================
-- SECTION 1: ENUM TYPES
-- ============================================

-- Twin entity types
CREATE TYPE twin_type AS ENUM (
    'hcp',           -- Healthcare Professional twins
    'patient',       -- Patient journey twins
    'territory'      -- Geographic territory twins
);

-- Simulation execution status
CREATE TYPE simulation_status AS ENUM (
    'pending',       -- Queued for execution
    'running',       -- Currently simulating
    'completed',     -- Successfully finished
    'failed'         -- Execution error
);

-- Simulation recommendation
CREATE TYPE simulation_recommendation AS ENUM (
    'deploy',        -- Proceed to real A/B test
    'skip',          -- Do not run experiment
    'refine'         -- Refine intervention and re-simulate
);

-- Fidelity assessment grade
CREATE TYPE fidelity_grade AS ENUM (
    'excellent',     -- Prediction error < 10%
    'good',          -- Prediction error 10-20%
    'fair',          -- Prediction error 20-35%
    'poor',          -- Prediction error > 35%
    'unvalidated'    -- No real-world comparison yet
);

-- ============================================
-- SECTION 2: CORE TABLES
-- ============================================

-- Table: digital_twin_models
-- Stores trained twin generator models with MLflow integration
CREATE TABLE digital_twin_models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Model identification
    model_name VARCHAR(200) NOT NULL,
    model_description TEXT,
    twin_type twin_type NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    
    -- MLflow integration
    mlflow_run_id VARCHAR(100),
    mlflow_model_uri VARCHAR(500),
    
    -- Training configuration
    training_config JSONB NOT NULL DEFAULT '{}',
    /*
    {
        "algorithm": "random_forest",
        "n_estimators": 100,
        "max_depth": 10,
        "training_samples": 50000,
        "validation_samples": 10000,
        "cv_folds": 5
    }
    */
    
    -- Feature specification
    feature_columns TEXT[] NOT NULL,
    target_columns TEXT[] NOT NULL,
    
    -- Performance metrics
    performance_metrics JSONB NOT NULL DEFAULT '{}',
    /*
    {
        "r2_score": 0.85,
        "rmse": 0.12,
        "mae": 0.08,
        "cv_scores": [0.83, 0.86, 0.84, 0.87, 0.85]
    }
    */
    
    -- Fidelity tracking
    fidelity_score FLOAT CHECK (fidelity_score >= 0 AND fidelity_score <= 1),
    fidelity_sample_count INTEGER DEFAULT 0,
    last_fidelity_update TIMESTAMP WITH TIME ZONE,
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    activation_date TIMESTAMP WITH TIME ZONE,
    deactivation_date TIMESTAMP WITH TIME ZONE,
    deactivation_reason TEXT,
    
    -- Scope
    brand VARCHAR(100) NOT NULL,
    geographic_scope VARCHAR(100) DEFAULT 'national',
    
    -- Audit
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_fidelity CHECK (fidelity_score IS NULL OR (fidelity_score >= 0 AND fidelity_score <= 1)),
    CONSTRAINT valid_brand CHECK (brand IN ('Remibrutinib', 'Fabhalta', 'Kisqali', 'All'))
);

-- Table: twin_simulations
-- Stores individual simulation runs with results
CREATE TABLE twin_simulations (
    simulation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Model reference
    model_id UUID NOT NULL REFERENCES digital_twin_models(model_id) ON DELETE RESTRICT,
    
    -- Optional experiment design reference
    experiment_design_id UUID,  -- FK to ml_experiments when linked
    
    -- Intervention specification
    intervention_type VARCHAR(100) NOT NULL,
    /*
    Common types:
    - email_campaign
    - call_frequency_increase
    - sample_distribution
    - speaker_program_invitation
    - digital_engagement
    - peer_influence_activation
    */
    intervention_config JSONB NOT NULL DEFAULT '{}',
    /*
    {
        "channel": "email",
        "frequency": "weekly",
        "content_type": "clinical_data",
        "duration_weeks": 8,
        "target_segment": "high_potential_decile_1_2"
    }
    */
    
    -- Population
    target_population twin_type NOT NULL,
    population_filters JSONB DEFAULT '{}',
    /*
    {
        "specialty": ["oncology", "hematology"],
        "decile": [1, 2, 3],
        "region": ["northeast", "midwest"],
        "adoption_stage": ["early_adopter"]
    }
    */
    twin_count INTEGER NOT NULL CHECK (twin_count > 0),
    
    -- Simulation results
    simulated_ate FLOAT,  -- Average Treatment Effect
    simulated_ci_lower FLOAT,  -- 95% CI lower bound
    simulated_ci_upper FLOAT,  -- 95% CI upper bound
    simulated_std_error FLOAT,
    effect_heterogeneity JSONB DEFAULT '{}',
    /*
    {
        "by_specialty": {
            "oncology": {"ate": 0.12, "n": 3000},
            "hematology": {"ate": 0.08, "n": 2000}
        },
        "by_decile": {
            "1-2": {"ate": 0.15, "n": 2000},
            "3-5": {"ate": 0.07, "n": 3000}
        }
    }
    */
    
    -- Outcome
    simulation_status simulation_status NOT NULL DEFAULT 'pending',
    recommendation simulation_recommendation,
    recommendation_rationale TEXT,
    recommended_sample_size INTEGER,
    recommended_duration_weeks INTEGER,
    
    -- Confidence and warnings
    simulation_confidence FLOAT CHECK (simulation_confidence >= 0 AND simulation_confidence <= 1),
    fidelity_warning BOOLEAN DEFAULT false,
    fidelity_warning_reason TEXT,
    
    -- Performance
    execution_time_ms INTEGER,
    memory_usage_mb INTEGER,
    
    -- Scope
    brand VARCHAR(100) NOT NULL,
    
    -- Error tracking
    error_message TEXT,
    error_traceback TEXT,
    
    -- Audit
    requested_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT valid_simulation_confidence CHECK (
        simulation_confidence IS NULL OR 
        (simulation_confidence >= 0 AND simulation_confidence <= 1)
    ),
    CONSTRAINT valid_brand CHECK (brand IN ('Remibrutinib', 'Fabhalta', 'Kisqali'))
);

-- Table: twin_fidelity_tracking
-- Tracks validation of twin predictions vs. real-world outcomes
CREATE TABLE twin_fidelity_tracking (
    tracking_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- References
    simulation_id UUID NOT NULL REFERENCES twin_simulations(simulation_id) ON DELETE CASCADE,
    actual_experiment_id UUID,  -- FK to ml_experiments when available
    
    -- Predicted vs. Actual
    simulated_ate FLOAT NOT NULL,
    simulated_ci_lower FLOAT,
    simulated_ci_upper FLOAT,
    
    actual_ate FLOAT,
    actual_ci_lower FLOAT,
    actual_ci_upper FLOAT,
    actual_sample_size INTEGER,
    
    -- Fidelity metrics
    prediction_error FLOAT,  -- (simulated - actual) / actual
    absolute_error FLOAT,    -- |simulated - actual|
    ci_coverage BOOLEAN,     -- Did actual fall within simulated CI?
    
    -- Assessment
    fidelity_grade fidelity_grade NOT NULL DEFAULT 'unvalidated',
    
    -- Context
    validation_notes TEXT,
    confounding_factors JSONB DEFAULT '[]',
    /*
    [
        "Market conditions changed during experiment",
        "Competitor launched similar product",
        "COVID impact on HCP engagement"
    ]
    */
    
    -- Audit
    validated_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    validated_at TIMESTAMP WITH TIME ZONE
);

-- ============================================
-- SECTION 3: INDEXES
-- ============================================

-- digital_twin_models indexes
CREATE INDEX idx_twin_models_type ON digital_twin_models(twin_type);
CREATE INDEX idx_twin_models_brand ON digital_twin_models(brand);
CREATE INDEX idx_twin_models_active ON digital_twin_models(is_active) WHERE is_active = true;
CREATE INDEX idx_twin_models_fidelity ON digital_twin_models(fidelity_score) WHERE fidelity_score IS NOT NULL;
CREATE INDEX idx_twin_models_mlflow ON digital_twin_models(mlflow_run_id) WHERE mlflow_run_id IS NOT NULL;

-- twin_simulations indexes
CREATE INDEX idx_simulations_model ON twin_simulations(model_id);
CREATE INDEX idx_simulations_status ON twin_simulations(simulation_status);
CREATE INDEX idx_simulations_brand ON twin_simulations(brand);
CREATE INDEX idx_simulations_intervention ON twin_simulations(intervention_type);
CREATE INDEX idx_simulations_recommendation ON twin_simulations(recommendation);
CREATE INDEX idx_simulations_created ON twin_simulations(created_at DESC);
CREATE INDEX idx_simulations_experiment ON twin_simulations(experiment_design_id) 
    WHERE experiment_design_id IS NOT NULL;

-- twin_fidelity_tracking indexes
CREATE INDEX idx_fidelity_simulation ON twin_fidelity_tracking(simulation_id);
CREATE INDEX idx_fidelity_grade ON twin_fidelity_tracking(fidelity_grade);
CREATE INDEX idx_fidelity_validated ON twin_fidelity_tracking(validated_at) 
    WHERE validated_at IS NOT NULL;

-- ============================================
-- SECTION 4: HELPER VIEWS
-- ============================================

-- View: v_active_twin_models
-- Lists all active twin models with latest fidelity status
CREATE OR REPLACE VIEW v_active_twin_models AS
SELECT 
    m.model_id,
    m.model_name,
    m.twin_type,
    m.model_version,
    m.brand,
    m.fidelity_score,
    m.fidelity_sample_count,
    m.last_fidelity_update,
    m.performance_metrics->>'r2_score' AS r2_score,
    m.created_at,
    COUNT(DISTINCT s.simulation_id) AS total_simulations,
    COUNT(DISTINCT CASE WHEN s.recommendation = 'deploy' THEN s.simulation_id END) AS deploy_recommendations,
    COUNT(DISTINCT CASE WHEN s.recommendation = 'skip' THEN s.simulation_id END) AS skip_recommendations
FROM digital_twin_models m
LEFT JOIN twin_simulations s ON m.model_id = s.model_id
WHERE m.is_active = true
GROUP BY m.model_id
ORDER BY m.fidelity_score DESC NULLS LAST, m.created_at DESC;

-- View: v_simulation_summary
-- Aggregates simulation results with fidelity tracking status
CREATE OR REPLACE VIEW v_simulation_summary AS
SELECT 
    s.simulation_id,
    s.intervention_type,
    s.brand,
    s.target_population,
    s.twin_count,
    s.simulated_ate,
    s.simulated_ci_lower,
    s.simulated_ci_upper,
    s.recommendation,
    s.recommended_sample_size,
    s.simulation_status,
    s.fidelity_warning,
    s.created_at,
    s.completed_at,
    m.model_name,
    m.fidelity_score AS model_fidelity,
    f.fidelity_grade,
    f.actual_ate,
    f.prediction_error,
    f.validated_at
FROM twin_simulations s
JOIN digital_twin_models m ON s.model_id = m.model_id
LEFT JOIN twin_fidelity_tracking f ON s.simulation_id = f.simulation_id
ORDER BY s.created_at DESC;

-- View: v_model_fidelity_history
-- Tracks fidelity over time for each model
CREATE OR REPLACE VIEW v_model_fidelity_history AS
SELECT 
    m.model_id,
    m.model_name,
    m.twin_type,
    m.brand,
    f.tracking_id,
    f.simulated_ate,
    f.actual_ate,
    f.prediction_error,
    f.absolute_error,
    f.ci_coverage,
    f.fidelity_grade,
    f.validated_at,
    f.validation_notes
FROM digital_twin_models m
JOIN twin_simulations s ON m.model_id = s.model_id
JOIN twin_fidelity_tracking f ON s.simulation_id = f.simulation_id
WHERE f.validated_at IS NOT NULL
ORDER BY m.model_id, f.validated_at DESC;

-- View: v_fidelity_degradation_alerts
-- Identifies models with declining fidelity (recent validations worse than historical)
CREATE OR REPLACE VIEW v_fidelity_degradation_alerts AS
WITH recent_fidelity AS (
    SELECT 
        m.model_id,
        AVG(f.absolute_error) AS recent_avg_error,
        COUNT(*) AS recent_count
    FROM digital_twin_models m
    JOIN twin_simulations s ON m.model_id = s.model_id
    JOIN twin_fidelity_tracking f ON s.simulation_id = f.simulation_id
    WHERE f.validated_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
    GROUP BY m.model_id
),
historical_fidelity AS (
    SELECT 
        m.model_id,
        AVG(f.absolute_error) AS historical_avg_error,
        COUNT(*) AS historical_count
    FROM digital_twin_models m
    JOIN twin_simulations s ON m.model_id = s.model_id
    JOIN twin_fidelity_tracking f ON s.simulation_id = f.simulation_id
    WHERE f.validated_at < CURRENT_TIMESTAMP - INTERVAL '30 days'
      AND f.validated_at >= CURRENT_TIMESTAMP - INTERVAL '180 days'
    GROUP BY m.model_id
)
SELECT 
    m.model_id,
    m.model_name,
    m.twin_type,
    m.brand,
    r.recent_avg_error,
    r.recent_count AS recent_validations,
    h.historical_avg_error,
    h.historical_count AS historical_validations,
    r.recent_avg_error - h.historical_avg_error AS error_increase,
    CASE 
        WHEN r.recent_avg_error > h.historical_avg_error * 1.5 THEN 'CRITICAL'
        WHEN r.recent_avg_error > h.historical_avg_error * 1.25 THEN 'WARNING'
        ELSE 'OK'
    END AS degradation_status
FROM digital_twin_models m
JOIN recent_fidelity r ON m.model_id = r.model_id
JOIN historical_fidelity h ON m.model_id = h.model_id
WHERE r.recent_avg_error > h.historical_avg_error * 1.1
ORDER BY error_increase DESC;

-- ============================================
-- SECTION 5: FUNCTIONS
-- ============================================

-- Function: calculate_fidelity_grade
-- Determines grade based on prediction error
CREATE OR REPLACE FUNCTION calculate_fidelity_grade(prediction_error FLOAT)
RETURNS fidelity_grade AS $$
BEGIN
    IF prediction_error IS NULL THEN
        RETURN 'unvalidated';
    ELSIF ABS(prediction_error) < 0.10 THEN
        RETURN 'excellent';
    ELSIF ABS(prediction_error) < 0.20 THEN
        RETURN 'good';
    ELSIF ABS(prediction_error) < 0.35 THEN
        RETURN 'fair';
    ELSE
        RETURN 'poor';
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function: update_model_fidelity
-- Recalculates model-level fidelity from tracking records
CREATE OR REPLACE FUNCTION update_model_fidelity(p_model_id UUID)
RETURNS void AS $$
DECLARE
    v_avg_error FLOAT;
    v_sample_count INTEGER;
BEGIN
    SELECT 
        AVG(ABS(f.prediction_error)),
        COUNT(*)
    INTO v_avg_error, v_sample_count
    FROM twin_simulations s
    JOIN twin_fidelity_tracking f ON s.simulation_id = f.simulation_id
    WHERE s.model_id = p_model_id
      AND f.prediction_error IS NOT NULL
      AND f.validated_at >= CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    UPDATE digital_twin_models
    SET 
        fidelity_score = CASE 
            WHEN v_sample_count >= 5 THEN 1.0 - LEAST(v_avg_error, 1.0)
            ELSE fidelity_score  -- Keep existing if insufficient samples
        END,
        fidelity_sample_count = v_sample_count,
        last_fidelity_update = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE model_id = p_model_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- SECTION 6: TRIGGERS
-- ============================================

-- Trigger: auto_grade_fidelity
-- Automatically calculates fidelity grade when validation is recorded
CREATE OR REPLACE FUNCTION trigger_auto_grade_fidelity()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.actual_ate IS NOT NULL AND NEW.simulated_ate IS NOT NULL THEN
        NEW.prediction_error := (NEW.simulated_ate - NEW.actual_ate) / NULLIF(NEW.actual_ate, 0);
        NEW.absolute_error := ABS(NEW.simulated_ate - NEW.actual_ate);
        NEW.ci_coverage := (
            NEW.actual_ate >= COALESCE(NEW.simulated_ci_lower, NEW.actual_ate) AND
            NEW.actual_ate <= COALESCE(NEW.simulated_ci_upper, NEW.actual_ate)
        );
        NEW.fidelity_grade := calculate_fidelity_grade(NEW.prediction_error);
        NEW.validated_at := COALESCE(NEW.validated_at, CURRENT_TIMESTAMP);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_auto_grade_fidelity
    BEFORE INSERT OR UPDATE OF actual_ate ON twin_fidelity_tracking
    FOR EACH ROW
    EXECUTE FUNCTION trigger_auto_grade_fidelity();

-- Trigger: update_model_fidelity_on_validation
-- Updates model-level fidelity when new validation is added
CREATE OR REPLACE FUNCTION trigger_update_model_fidelity()
RETURNS TRIGGER AS $$
DECLARE
    v_model_id UUID;
BEGIN
    SELECT model_id INTO v_model_id
    FROM twin_simulations
    WHERE simulation_id = NEW.simulation_id;
    
    IF v_model_id IS NOT NULL AND NEW.actual_ate IS NOT NULL THEN
        PERFORM update_model_fidelity(v_model_id);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_model_fidelity
    AFTER INSERT OR UPDATE OF actual_ate ON twin_fidelity_tracking
    FOR EACH ROW
    EXECUTE FUNCTION trigger_update_model_fidelity();

-- Trigger: updated_at timestamp for digital_twin_models
CREATE OR REPLACE FUNCTION trigger_update_twin_model_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_twin_models_updated_at
    BEFORE UPDATE ON digital_twin_models
    FOR EACH ROW
    EXECUTE FUNCTION trigger_update_twin_model_timestamp();

-- ============================================
-- SECTION 7: COMMENTS
-- ============================================

COMMENT ON TABLE digital_twin_models IS 
    'Stores trained ML models that generate digital twins for simulation';

COMMENT ON TABLE twin_simulations IS 
    'Records simulation runs with predicted intervention effects';

COMMENT ON TABLE twin_fidelity_tracking IS 
    'Tracks validation of twin predictions against real experiment outcomes';

COMMENT ON VIEW v_active_twin_models IS 
    'Lists all active twin models with aggregated simulation statistics';

COMMENT ON VIEW v_simulation_summary IS 
    'Comprehensive view of simulations with model and fidelity information';

COMMENT ON VIEW v_fidelity_degradation_alerts IS 
    'Identifies models showing declining prediction accuracy';

-- ============================================
-- SECTION 8: GRANTS (adjust for your roles)
-- ============================================

-- Grant SELECT on views to read-only users
-- GRANT SELECT ON v_active_twin_models TO e2i_readonly;
-- GRANT SELECT ON v_simulation_summary TO e2i_readonly;
-- GRANT SELECT ON v_model_fidelity_history TO e2i_readonly;
-- GRANT SELECT ON v_fidelity_degradation_alerts TO e2i_readonly;

-- Grant CRUD on tables to application user
-- GRANT SELECT, INSERT, UPDATE, DELETE ON digital_twin_models TO e2i_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON twin_simulations TO e2i_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON twin_fidelity_tracking TO e2i_app;

-- ============================================
-- END OF MIGRATION
-- ============================================
