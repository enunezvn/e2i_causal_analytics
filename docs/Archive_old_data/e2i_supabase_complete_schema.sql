-- ============================================================================
-- E2I CAUSAL ANALYTICS DASHBOARD - COMPLETE SUPABASE SCHEMA
-- ============================================================================
-- Version: 2.0.0
-- Created: November 2025
-- Description: Complete database schema for E2I Causal Analytics Dashboard
--              with built-in ML train/validation/test split support
-- ============================================================================

-- ============================================================================
-- PART 1: EXTENSIONS AND ENUM TYPES
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Data split types for ML pipeline
CREATE TYPE data_split_type AS ENUM (
    'train', 
    'validation', 
    'test', 
    'holdout', 
    'unassigned'
);

-- Split strategy types
CREATE TYPE split_strategy_type AS ENUM (
    'chronological',
    'patient_stratified',
    'rolling_window',
    'causal_holdout'
);

-- Brand types
CREATE TYPE brand_type AS ENUM (
    'Remibrutinib',
    'Fabhalta',
    'Kisqali',
    'competitor',
    'other'
);

-- Region types
CREATE TYPE region_type AS ENUM (
    'northeast',
    'south',
    'midwest',
    'west'
);

-- Trigger priority types
CREATE TYPE priority_type AS ENUM (
    'critical',
    'high',
    'medium',
    'low'
);

-- Journey stage types
CREATE TYPE journey_stage_type AS ENUM (
    'diagnosis',
    'initial_treatment',
    'treatment_optimization',
    'maintenance',
    'treatment_switch'
);

-- Journey status types
CREATE TYPE journey_status_type AS ENUM (
    'active',
    'stable',
    'transitioning',
    'completed'
);

-- Event types
CREATE TYPE event_type AS ENUM (
    'diagnosis',
    'prescription',
    'lab_test',
    'procedure',
    'consultation',
    'hospitalization'
);

-- Prediction types
CREATE TYPE prediction_type AS ENUM (
    'trigger',
    'propensity',
    'risk',
    'churn',
    'next_best_action'
);

-- Agent names
CREATE TYPE agent_name_type AS ENUM (
    'causal_chain_analyzer',
    'multiplier_discoverer',
    'data_drift_monitor',
    'prediction_synthesizer',
    'heterogeneous_optimizer',
    'competitive_landscape',
    'explainer_agent',
    'feedback_learner'
);

-- Workstream types
CREATE TYPE workstream_type AS ENUM (
    'WS1',
    'WS2',
    'WS3'
);

-- ============================================================================
-- PART 2: ML SPLIT MANAGEMENT TABLES
-- ============================================================================

-- Central registry for ML split configurations
CREATE TABLE ml_split_registry (
    split_config_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_name VARCHAR(100) NOT NULL UNIQUE,
    config_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    
    -- Split ratios (must sum to 1.0)
    train_ratio DECIMAL(3,2) NOT NULL DEFAULT 0.60,
    validation_ratio DECIMAL(3,2) NOT NULL DEFAULT 0.20,
    test_ratio DECIMAL(3,2) NOT NULL DEFAULT 0.15,
    holdout_ratio DECIMAL(3,2) NOT NULL DEFAULT 0.05,
    
    -- Temporal boundaries
    data_start_date DATE NOT NULL,
    data_end_date DATE NOT NULL,
    train_end_date DATE NOT NULL,
    validation_end_date DATE NOT NULL,
    test_end_date DATE NOT NULL,
    
    -- Configuration settings
    temporal_gap_days INTEGER NOT NULL DEFAULT 7,
    patient_level_isolation BOOLEAN NOT NULL DEFAULT TRUE,
    split_strategy split_strategy_type NOT NULL DEFAULT 'chronological',
    random_seed INTEGER,
    
    -- Metadata
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100),
    notes TEXT,
    
    -- Constraints
    CONSTRAINT valid_ratios CHECK (
        ABS(train_ratio + validation_ratio + test_ratio + holdout_ratio - 1.0) < 0.001
    ),
    CONSTRAINT valid_date_order CHECK (
        data_start_date < train_end_date AND
        train_end_date < validation_end_date AND
        validation_end_date < test_end_date AND
        test_end_date <= data_end_date
    )
);

-- Patient-to-split assignment tracking
CREATE TABLE ml_patient_split_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    split_config_id UUID NOT NULL REFERENCES ml_split_registry(split_config_id) ON DELETE CASCADE,
    patient_id VARCHAR(20) NOT NULL,
    assigned_split data_split_type NOT NULL,
    assignment_reason VARCHAR(100),
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Each patient can only be in one split per config
    UNIQUE(split_config_id, patient_id)
);

-- Preprocessing metadata (MUST be computed on training data only)
CREATE TABLE ml_preprocessing_metadata (
    metadata_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    split_config_id UUID NOT NULL REFERENCES ml_split_registry(split_config_id) ON DELETE CASCADE,
    
    -- Source verification - CRITICAL: Must always be 'train'
    computed_on_split data_split_type NOT NULL DEFAULT 'train',
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Numerical feature statistics
    feature_means JSONB NOT NULL DEFAULT '{}',
    feature_stds JSONB NOT NULL DEFAULT '{}',
    feature_mins JSONB NOT NULL DEFAULT '{}',
    feature_maxs JSONB NOT NULL DEFAULT '{}',
    
    -- Categorical encodings
    categorical_encodings JSONB NOT NULL DEFAULT '{}',
    
    -- Additional metadata
    num_training_samples INTEGER,
    feature_list TEXT[],
    preprocessing_pipeline_version VARCHAR(20),
    
    -- CRITICAL: Ensure preprocessing only computed on training data
    CONSTRAINT train_only CHECK (computed_on_split = 'train'),
    
    -- One metadata record per config
    UNIQUE(split_config_id)
);

-- Leakage audit trail
CREATE TABLE ml_leakage_audit (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    split_config_id UUID REFERENCES ml_split_registry(split_config_id) ON DELETE SET NULL,
    audit_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    check_type VARCHAR(50) NOT NULL,
    passed BOOLEAN NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'warning', 'info')),
    details TEXT,
    affected_records INTEGER,
    remediation_action TEXT,
    audited_by VARCHAR(100)
);

-- ============================================================================
-- PART 3: CORE DATA TABLES
-- ============================================================================

-- HCP (Healthcare Provider) Profiles
CREATE TABLE hcp_profiles (
    hcp_id VARCHAR(20) PRIMARY KEY,
    npi VARCHAR(20) UNIQUE,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    specialty VARCHAR(100),
    sub_specialty VARCHAR(100),
    practice_type VARCHAR(50),
    practice_size VARCHAR(20),
    geographic_region region_type,
    state VARCHAR(2),
    city VARCHAR(100),
    zip_code VARCHAR(10),
    priority_tier INTEGER CHECK (priority_tier BETWEEN 1 AND 5),
    decile INTEGER CHECK (decile BETWEEN 1 AND 10),
    total_patient_volume INTEGER,
    target_patient_volume INTEGER,
    prescribing_volume INTEGER,
    years_experience INTEGER,
    affiliation_primary VARCHAR(200),
    affiliation_secondary TEXT[],
    digital_engagement_score DECIMAL(3,2),
    preferred_channel VARCHAR(20),
    last_interaction_date DATE,
    interaction_frequency DECIMAL(3,1),
    influence_network_size INTEGER,
    peer_influence_score DECIMAL(3,2),
    adoption_category VARCHAR(20),
    coverage_status BOOLEAN DEFAULT TRUE,
    territory_id VARCHAR(20),
    sales_rep_id VARCHAR(20),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Patient Journeys
CREATE TABLE patient_journeys (
    patient_journey_id VARCHAR(20) PRIMARY KEY,
    patient_id VARCHAR(20) NOT NULL,
    patient_hash VARCHAR(50),
    journey_start_date DATE NOT NULL,
    journey_end_date DATE,
    journey_duration_days INTEGER,
    journey_stage journey_stage_type,
    journey_status journey_status_type,
    primary_diagnosis_code VARCHAR(20),
    primary_diagnosis_desc TEXT,
    secondary_diagnosis_codes TEXT[],
    brand brand_type,
    age_group VARCHAR(10),
    gender VARCHAR(1),
    geographic_region region_type,
    state VARCHAR(2),
    zip_code VARCHAR(10),
    insurance_type VARCHAR(20),
    data_quality_score DECIMAL(3,2),
    comorbidities TEXT[],
    risk_score DECIMAL(3,2),
    
    -- ML Split tracking
    data_split data_split_type NOT NULL DEFAULT 'unassigned',
    split_config_id UUID REFERENCES ml_split_registry(split_config_id),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Treatment Events
CREATE TABLE treatment_events (
    treatment_event_id VARCHAR(30) PRIMARY KEY,
    patient_journey_id VARCHAR(20) REFERENCES patient_journeys(patient_journey_id) ON DELETE CASCADE,
    patient_id VARCHAR(20) NOT NULL,
    hcp_id VARCHAR(20) REFERENCES hcp_profiles(hcp_id),
    event_date DATE NOT NULL,
    event_type event_type,
    event_subtype VARCHAR(50),
    brand brand_type,
    drug_ndc VARCHAR(20),
    drug_name VARCHAR(100),
    drug_class VARCHAR(50),
    dosage VARCHAR(50),
    duration_days INTEGER,
    icd_codes TEXT[],
    cpt_codes TEXT[],
    loinc_codes TEXT[],
    lab_values JSONB DEFAULT '{}',
    location_type VARCHAR(50),
    facility_id VARCHAR(20),
    cost DECIMAL(10,2),
    outcome_indicator VARCHAR(20),
    adverse_event_flag BOOLEAN DEFAULT FALSE,
    discontinuation_flag BOOLEAN DEFAULT FALSE,
    discontinuation_reason VARCHAR(100),
    sequence_number INTEGER,
    days_from_diagnosis INTEGER,
    previous_treatment VARCHAR(100),
    next_treatment VARCHAR(100),
    
    -- ML Split tracking
    data_split data_split_type NOT NULL DEFAULT 'unassigned',
    split_config_id UUID REFERENCES ml_split_registry(split_config_id),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ML Model Predictions
CREATE TABLE ml_predictions (
    prediction_id VARCHAR(30) PRIMARY KEY,
    model_version VARCHAR(20),
    model_type VARCHAR(30),
    prediction_timestamp TIMESTAMPTZ NOT NULL,
    patient_id VARCHAR(20) NOT NULL,
    hcp_id VARCHAR(20) REFERENCES hcp_profiles(hcp_id),
    prediction_type prediction_type,
    prediction_value DECIMAL(5,4),
    prediction_class VARCHAR(20),
    confidence_score DECIMAL(5,4),
    probability_scores JSONB DEFAULT '{}',
    feature_importance JSONB DEFAULT '{}',
    shap_values JSONB DEFAULT '{}',
    top_features JSONB DEFAULT '[]',
    model_auc DECIMAL(4,3),
    model_precision DECIMAL(4,3),
    model_recall DECIMAL(4,3),
    calibration_score DECIMAL(4,3),
    fairness_metrics JSONB DEFAULT '{}',
    explanation_text TEXT,
    treatment_effect_estimate DECIMAL(4,3),
    heterogeneous_effect DECIMAL(4,3),
    segment_assignment VARCHAR(30),
    causal_confidence DECIMAL(4,3),
    counterfactual_outcome DECIMAL(4,3),
    
    -- CRITICAL: Track what features were available at prediction time
    features_available_at_prediction JSONB DEFAULT '{}',
    
    -- ML Split tracking
    data_split data_split_type NOT NULL DEFAULT 'unassigned',
    split_config_id UUID REFERENCES ml_split_registry(split_config_id),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Triggers
CREATE TABLE triggers (
    trigger_id VARCHAR(30) PRIMARY KEY,
    patient_id VARCHAR(20) NOT NULL,
    hcp_id VARCHAR(20) REFERENCES hcp_profiles(hcp_id),
    trigger_timestamp TIMESTAMPTZ NOT NULL,
    trigger_type VARCHAR(50),
    priority priority_type,
    confidence_score DECIMAL(4,3),
    lead_time_days INTEGER,
    expiration_date DATE,
    delivery_channel VARCHAR(20),
    delivery_status VARCHAR(20),
    delivery_timestamp TIMESTAMPTZ,
    view_timestamp TIMESTAMPTZ,
    acceptance_status VARCHAR(20),
    acceptance_timestamp TIMESTAMPTZ,
    action_taken TEXT,
    action_timestamp TIMESTAMPTZ,
    false_positive_flag BOOLEAN DEFAULT FALSE,
    trigger_reason TEXT,
    causal_chain JSONB DEFAULT '{}',
    supporting_evidence JSONB DEFAULT '{}',
    recommended_action TEXT,
    outcome_tracked BOOLEAN DEFAULT FALSE,
    outcome_value DECIMAL(4,3),
    
    -- ML Split tracking
    data_split data_split_type NOT NULL DEFAULT 'unassigned',
    split_config_id UUID REFERENCES ml_split_registry(split_config_id),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Agent Activities
CREATE TABLE agent_activities (
    activity_id VARCHAR(30) PRIMARY KEY,
    agent_name VARCHAR(50),
    activity_timestamp TIMESTAMPTZ NOT NULL,
    activity_type VARCHAR(30),
    workstream workstream_type,
    processing_duration_ms INTEGER,
    input_data JSONB DEFAULT '{}',
    records_processed INTEGER,
    time_window VARCHAR(20),
    analysis_results JSONB DEFAULT '{}',
    causal_paths_analyzed INTEGER,
    confidence_level DECIMAL(4,3),
    recommendations JSONB DEFAULT '[]',
    actions_initiated JSONB DEFAULT '[]',
    impact_estimate DECIMAL(15,2),
    roi_estimate DECIMAL(5,2),
    status VARCHAR(20),
    error_message TEXT,
    resource_usage JSONB DEFAULT '{}',
    
    -- ML Split tracking
    data_split data_split_type NOT NULL DEFAULT 'unassigned',
    split_config_id UUID REFERENCES ml_split_registry(split_config_id),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Business Metrics
CREATE TABLE business_metrics (
    metric_id VARCHAR(50) PRIMARY KEY,
    metric_date DATE NOT NULL,
    metric_type VARCHAR(30),
    metric_name VARCHAR(100),
    brand brand_type,
    region region_type,
    value DECIMAL(15,2),
    target DECIMAL(15,2),
    achievement_rate DECIMAL(5,3),
    year_over_year_change DECIMAL(5,3),
    month_over_month_change DECIMAL(5,3),
    roi DECIMAL(5,2),
    statistical_significance DECIMAL(4,3),
    confidence_interval_lower DECIMAL(15,2),
    confidence_interval_upper DECIMAL(15,2),
    sample_size INTEGER,
    
    -- ML Split tracking
    data_split data_split_type NOT NULL DEFAULT 'unassigned',
    split_config_id UUID REFERENCES ml_split_registry(split_config_id),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Causal Paths (discovered causal relationships)
CREATE TABLE causal_paths (
    path_id VARCHAR(20) PRIMARY KEY,
    discovery_date DATE NOT NULL,
    causal_chain JSONB NOT NULL,
    start_node VARCHAR(100),
    end_node VARCHAR(100),
    intermediate_nodes TEXT[],
    path_length INTEGER,
    causal_effect_size DECIMAL(5,3),
    confidence_level DECIMAL(4,3),
    method_used VARCHAR(50),
    confounders_controlled TEXT[],
    mediators_identified TEXT[],
    interaction_effects JSONB DEFAULT '{}',
    time_lag_days INTEGER,
    validation_status VARCHAR(20),
    business_impact_estimate DECIMAL(15,2),
    
    -- ML Split tracking
    data_split data_split_type NOT NULL DEFAULT 'unassigned',
    split_config_id UUID REFERENCES ml_split_registry(split_config_id),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- PART 4: INDEXES FOR PERFORMANCE
-- ============================================================================

-- Indexes for split-based queries (most common pattern)
CREATE INDEX idx_patient_journeys_split ON patient_journeys(data_split, split_config_id);
CREATE INDEX idx_treatment_events_split ON treatment_events(data_split, split_config_id);
CREATE INDEX idx_ml_predictions_split ON ml_predictions(data_split, split_config_id);
CREATE INDEX idx_triggers_split ON triggers(data_split, split_config_id);
CREATE INDEX idx_agent_activities_split ON agent_activities(data_split, split_config_id);
CREATE INDEX idx_business_metrics_split ON business_metrics(data_split, split_config_id);
CREATE INDEX idx_causal_paths_split ON causal_paths(data_split, split_config_id);

-- Indexes for common query patterns
CREATE INDEX idx_patient_journeys_patient ON patient_journeys(patient_id);
CREATE INDEX idx_patient_journeys_brand ON patient_journeys(brand);
CREATE INDEX idx_patient_journeys_region ON patient_journeys(geographic_region);
CREATE INDEX idx_patient_journeys_dates ON patient_journeys(journey_start_date, journey_end_date);

CREATE INDEX idx_treatment_events_patient ON treatment_events(patient_id);
CREATE INDEX idx_treatment_events_date ON treatment_events(event_date);
CREATE INDEX idx_treatment_events_journey ON treatment_events(patient_journey_id);

CREATE INDEX idx_ml_predictions_patient ON ml_predictions(patient_id);
CREATE INDEX idx_ml_predictions_timestamp ON ml_predictions(prediction_timestamp);
CREATE INDEX idx_ml_predictions_type ON ml_predictions(prediction_type);

CREATE INDEX idx_triggers_patient ON triggers(patient_id);
CREATE INDEX idx_triggers_hcp ON triggers(hcp_id);
CREATE INDEX idx_triggers_timestamp ON triggers(trigger_timestamp);
CREATE INDEX idx_triggers_status ON triggers(delivery_status, acceptance_status);

CREATE INDEX idx_agent_activities_agent ON agent_activities(agent_name);
CREATE INDEX idx_agent_activities_workstream ON agent_activities(workstream);
CREATE INDEX idx_agent_activities_timestamp ON agent_activities(activity_timestamp);

CREATE INDEX idx_business_metrics_date ON business_metrics(metric_date);
CREATE INDEX idx_business_metrics_brand_region ON business_metrics(brand, region);

CREATE INDEX idx_patient_split_lookup ON ml_patient_split_assignments(patient_id, split_config_id);
CREATE INDEX idx_leakage_audit_config ON ml_leakage_audit(split_config_id, audit_timestamp DESC);

-- ============================================================================
-- PART 5: VIEWS FOR EASY SPLIT ACCESS
-- ============================================================================

-- Training data views
CREATE OR REPLACE VIEW v_train_patient_journeys AS
SELECT * FROM patient_journeys WHERE data_split = 'train';

CREATE OR REPLACE VIEW v_train_treatment_events AS
SELECT * FROM treatment_events WHERE data_split = 'train';

CREATE OR REPLACE VIEW v_train_ml_predictions AS
SELECT * FROM ml_predictions WHERE data_split = 'train';

CREATE OR REPLACE VIEW v_train_triggers AS
SELECT * FROM triggers WHERE data_split = 'train';

CREATE OR REPLACE VIEW v_train_agent_activities AS
SELECT * FROM agent_activities WHERE data_split = 'train';

CREATE OR REPLACE VIEW v_train_business_metrics AS
SELECT * FROM business_metrics WHERE data_split = 'train';

-- Validation data views
CREATE OR REPLACE VIEW v_validation_patient_journeys AS
SELECT * FROM patient_journeys WHERE data_split = 'validation';

CREATE OR REPLACE VIEW v_validation_treatment_events AS
SELECT * FROM treatment_events WHERE data_split = 'validation';

CREATE OR REPLACE VIEW v_validation_ml_predictions AS
SELECT * FROM ml_predictions WHERE data_split = 'validation';

CREATE OR REPLACE VIEW v_validation_triggers AS
SELECT * FROM triggers WHERE data_split = 'validation';

-- Test data views
CREATE OR REPLACE VIEW v_test_patient_journeys AS
SELECT * FROM patient_journeys WHERE data_split = 'test';

CREATE OR REPLACE VIEW v_test_treatment_events AS
SELECT * FROM treatment_events WHERE data_split = 'test';

CREATE OR REPLACE VIEW v_test_ml_predictions AS
SELECT * FROM ml_predictions WHERE data_split = 'test';

CREATE OR REPLACE VIEW v_test_triggers AS
SELECT * FROM triggers WHERE data_split = 'test';

-- Holdout data views (restricted - for final evaluation only)
CREATE OR REPLACE VIEW v_holdout_patient_journeys AS
SELECT * FROM patient_journeys WHERE data_split = 'holdout';

-- Split statistics summary view
CREATE OR REPLACE VIEW v_split_statistics AS
SELECT 
    sr.config_name,
    sr.split_strategy::TEXT,
    pj.data_split::TEXT,
    COUNT(DISTINCT pj.patient_id) as patient_count,
    COUNT(DISTINCT te.treatment_event_id) as event_count,
    COUNT(DISTINCT mp.prediction_id) as prediction_count,
    COUNT(DISTINCT t.trigger_id) as trigger_count,
    MIN(pj.journey_start_date) as earliest_date,
    MAX(pj.journey_end_date) as latest_date
FROM ml_split_registry sr
LEFT JOIN patient_journeys pj ON pj.split_config_id = sr.split_config_id
LEFT JOIN treatment_events te ON te.patient_journey_id = pj.patient_journey_id
LEFT JOIN ml_predictions mp ON mp.patient_id = pj.patient_id AND mp.split_config_id = sr.split_config_id
LEFT JOIN triggers t ON t.patient_id = pj.patient_id AND t.split_config_id = sr.split_config_id
WHERE sr.is_active = TRUE
GROUP BY sr.config_name, sr.split_strategy, pj.data_split
ORDER BY sr.config_name, 
    CASE pj.data_split 
        WHEN 'train' THEN 1 
        WHEN 'validation' THEN 2 
        WHEN 'test' THEN 3 
        WHEN 'holdout' THEN 4 
        ELSE 5 
    END;

-- ============================================================================
-- PART 6: HELPER FUNCTIONS
-- ============================================================================

-- Function to assign patient to split based on journey date
CREATE OR REPLACE FUNCTION assign_patient_split(
    p_patient_id VARCHAR(20),
    p_journey_start_date DATE,
    p_split_config_id UUID
) RETURNS data_split_type AS $$
DECLARE
    v_config ml_split_registry%ROWTYPE;
    v_assigned_split data_split_type;
    v_gap INTEGER;
BEGIN
    -- Get split configuration
    SELECT * INTO v_config 
    FROM ml_split_registry 
    WHERE split_config_id = p_split_config_id;
    
    IF v_config IS NULL THEN
        RETURN 'unassigned';
    END IF;
    
    v_gap := v_config.temporal_gap_days;
    
    -- Check if patient already assigned
    SELECT assigned_split INTO v_assigned_split
    FROM ml_patient_split_assignments
    WHERE patient_id = p_patient_id 
    AND split_config_id = p_split_config_id;
    
    IF v_assigned_split IS NOT NULL THEN
        RETURN v_assigned_split;
    END IF;
    
    -- Assign based on date (respecting gaps)
    IF p_journey_start_date < v_config.train_end_date THEN
        v_assigned_split := 'train';
    ELSIF p_journey_start_date >= (v_config.train_end_date + v_gap) 
          AND p_journey_start_date < v_config.validation_end_date THEN
        v_assigned_split := 'validation';
    ELSIF p_journey_start_date >= (v_config.validation_end_date + v_gap)
          AND p_journey_start_date < v_config.test_end_date THEN
        v_assigned_split := 'test';
    ELSIF p_journey_start_date >= (v_config.test_end_date + v_gap) THEN
        v_assigned_split := 'holdout';
    ELSE
        -- In gap period - assign to earlier split to avoid leakage
        IF p_journey_start_date < (v_config.train_end_date + v_gap) THEN
            v_assigned_split := 'train';
        ELSIF p_journey_start_date < (v_config.validation_end_date + v_gap) THEN
            v_assigned_split := 'validation';
        ELSE
            v_assigned_split := 'test';
        END IF;
    END IF;
    
    -- Record assignment
    INSERT INTO ml_patient_split_assignments (
        split_config_id, patient_id, assigned_split, assignment_reason
    ) VALUES (
        p_split_config_id, p_patient_id, v_assigned_split, 'chronological'
    ) ON CONFLICT (split_config_id, patient_id) DO NOTHING;
    
    RETURN v_assigned_split;
END;
$$ LANGUAGE plpgsql;

-- Function to get preprocessing stats
CREATE OR REPLACE FUNCTION get_preprocessing_stats(
    p_split_config_id UUID
) RETURNS TABLE(
    feature_name TEXT,
    mean_value NUMERIC,
    std_value NUMERIC,
    min_value NUMERIC,
    max_value NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        key as feature_name,
        (pm.feature_means->>key)::NUMERIC as mean_value,
        (pm.feature_stds->>key)::NUMERIC as std_value,
        (pm.feature_mins->>key)::NUMERIC as min_value,
        (pm.feature_maxs->>key)::NUMERIC as max_value
    FROM ml_preprocessing_metadata pm,
    LATERAL jsonb_object_keys(pm.feature_means) as key
    WHERE pm.split_config_id = p_split_config_id
    AND pm.computed_on_split = 'train';
END;
$$ LANGUAGE plpgsql;

-- Function to run leakage audit
CREATE OR REPLACE FUNCTION run_leakage_audit(
    p_split_config_id UUID
) RETURNS TABLE(
    check_type VARCHAR, 
    passed BOOLEAN, 
    severity VARCHAR,
    details TEXT
) AS $$
DECLARE
    v_duplicate_patients INTEGER;
    v_preprocessing_valid BOOLEAN;
BEGIN
    -- Check 1: Patient split isolation
    SELECT COUNT(*) INTO v_duplicate_patients
    FROM (
        SELECT patient_id, COUNT(DISTINCT data_split) as split_count
        FROM patient_journeys
        WHERE split_config_id = p_split_config_id
        GROUP BY patient_id
        HAVING COUNT(DISTINCT data_split) > 1
    ) duplicates;
    
    INSERT INTO ml_leakage_audit (
        split_config_id, check_type, passed, severity, details, affected_records
    ) VALUES (
        p_split_config_id, 
        'patient_split_isolation',
        v_duplicate_patients = 0,
        CASE WHEN v_duplicate_patients = 0 THEN 'info' ELSE 'critical' END,
        CASE WHEN v_duplicate_patients = 0 
            THEN 'All patients correctly isolated to single split'
            ELSE format('%s patients found in multiple splits', v_duplicate_patients)
        END,
        v_duplicate_patients
    );
    
    RETURN QUERY SELECT 
        'patient_split_isolation'::VARCHAR,
        v_duplicate_patients = 0,
        CASE WHEN v_duplicate_patients = 0 THEN 'info' ELSE 'critical' END::VARCHAR,
        CASE WHEN v_duplicate_patients = 0 
            THEN 'All patients correctly isolated'
            ELSE format('%s patients in multiple splits', v_duplicate_patients)
        END::TEXT;
    
    -- Check 2: Preprocessing metadata source
    SELECT EXISTS(
        SELECT 1 FROM ml_preprocessing_metadata 
        WHERE split_config_id = p_split_config_id 
        AND computed_on_split = 'train'
    ) INTO v_preprocessing_valid;
    
    INSERT INTO ml_leakage_audit (
        split_config_id, check_type, passed, severity, details
    ) VALUES (
        p_split_config_id,
        'preprocessing_source',
        v_preprocessing_valid,
        CASE WHEN v_preprocessing_valid THEN 'info' ELSE 'critical' END,
        CASE WHEN v_preprocessing_valid 
            THEN 'Preprocessing metadata computed on training data only'
            ELSE 'WARNING: Preprocessing metadata missing or not from training data'
        END
    );
    
    RETURN QUERY SELECT 
        'preprocessing_source'::VARCHAR,
        v_preprocessing_valid,
        CASE WHEN v_preprocessing_valid THEN 'info' ELSE 'critical' END::VARCHAR,
        CASE WHEN v_preprocessing_valid 
            THEN 'Preprocessing metadata computed on training data only'
            ELSE 'WARNING: Preprocessing metadata missing or not from training data'
        END::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Function to get split date boundaries
CREATE OR REPLACE FUNCTION get_split_boundaries(
    p_split_config_id UUID
) RETURNS TABLE(
    split_name TEXT,
    start_date DATE,
    end_date DATE,
    patient_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    WITH config AS (
        SELECT * FROM ml_split_registry WHERE split_config_id = p_split_config_id
    ),
    patient_counts AS (
        SELECT data_split, COUNT(DISTINCT patient_id) as cnt
        FROM patient_journeys
        WHERE split_config_id = p_split_config_id
        GROUP BY data_split
    )
    SELECT 
        'train'::TEXT,
        c.data_start_date,
        c.train_end_date,
        COALESCE(pc.cnt, 0)
    FROM config c
    LEFT JOIN patient_counts pc ON pc.data_split = 'train'
    UNION ALL
    SELECT 
        'validation'::TEXT,
        c.train_end_date + c.temporal_gap_days,
        c.validation_end_date,
        COALESCE(pc.cnt, 0)
    FROM config c
    LEFT JOIN patient_counts pc ON pc.data_split = 'validation'
    UNION ALL
    SELECT 
        'test'::TEXT,
        c.validation_end_date + c.temporal_gap_days,
        c.test_end_date,
        COALESCE(pc.cnt, 0)
    FROM config c
    LEFT JOIN patient_counts pc ON pc.data_split = 'test'
    UNION ALL
    SELECT 
        'holdout'::TEXT,
        c.test_end_date + c.temporal_gap_days,
        c.data_end_date,
        COALESCE(pc.cnt, 0)
    FROM config c
    LEFT JOIN patient_counts pc ON pc.data_split = 'holdout';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PART 7: TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to tables with updated_at
CREATE TRIGGER update_hcp_profiles_timestamp
    BEFORE UPDATE ON hcp_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_patient_journeys_timestamp
    BEFORE UPDATE ON patient_journeys
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_triggers_timestamp
    BEFORE UPDATE ON triggers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_split_registry_timestamp
    BEFORE UPDATE ON ml_split_registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- PART 8: ROW-LEVEL SECURITY (Optional - enable as needed)
-- ============================================================================

-- Uncomment these lines to enable RLS
-- ALTER TABLE patient_journeys ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ml_predictions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE triggers ENABLE ROW LEVEL SECURITY;

-- Example policies (customize based on your auth setup)
-- CREATE POLICY analyst_train_val_only ON patient_journeys
--     FOR SELECT
--     USING (
--         data_split IN ('train', 'validation')
--         OR current_setting('app.role', true) = 'data_scientist'
--     );

-- ============================================================================
-- PART 9: COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE ml_split_registry IS 
'Central registry for ML data split configurations. Ensures consistent train/validation/test/holdout splits.';

COMMENT ON TABLE ml_patient_split_assignments IS 
'Tracks patient-to-split assignments. Ensures patient-level isolation to prevent data leakage.';

COMMENT ON TABLE ml_preprocessing_metadata IS 
'Stores preprocessing statistics computed ONLY on training data. Critical for preventing preprocessing leakage.';

COMMENT ON TABLE ml_leakage_audit IS 
'Audit trail for data leakage checks. Important for pharmaceutical compliance.';

COMMENT ON TABLE patient_journeys IS 
'Patient journey records with ML split tracking. Each patient exists in exactly one split.';

COMMENT ON TABLE treatment_events IS 
'Treatment events linked to patient journeys. Inherits split from parent journey.';

COMMENT ON TABLE ml_predictions IS 
'ML model predictions with features_available_at_prediction to prevent outcome leakage.';

COMMENT ON COLUMN ml_predictions.features_available_at_prediction IS 
'CRITICAL: Tracks what features were available at prediction time to prevent outcome leakage.';

COMMENT ON FUNCTION assign_patient_split IS 
'Assigns a patient to a data split based on journey start date. Ensures chronological ordering.';

COMMENT ON FUNCTION run_leakage_audit IS 
'Runs comprehensive data leakage audit. Returns pass/fail for each check type.';

-- ============================================================================
-- SCHEMA COMPLETE
-- ============================================================================

-- Print summary
DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'E2I Causal Analytics Schema Created Successfully!';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'Tables created:';
    RAISE NOTICE '  - ml_split_registry (split configuration)';
    RAISE NOTICE '  - ml_patient_split_assignments (patient isolation)';
    RAISE NOTICE '  - ml_preprocessing_metadata (train-only stats)';
    RAISE NOTICE '  - ml_leakage_audit (compliance tracking)';
    RAISE NOTICE '  - hcp_profiles (healthcare providers)';
    RAISE NOTICE '  - patient_journeys (with data_split column)';
    RAISE NOTICE '  - treatment_events (with data_split column)';
    RAISE NOTICE '  - ml_predictions (with data_split column)';
    RAISE NOTICE '  - triggers (with data_split column)';
    RAISE NOTICE '  - agent_activities (with data_split column)';
    RAISE NOTICE '  - business_metrics (with data_split column)';
    RAISE NOTICE '  - causal_paths (with data_split column)';
    RAISE NOTICE '';
    RAISE NOTICE 'Views created: v_train_*, v_validation_*, v_test_*, v_holdout_*';
    RAISE NOTICE 'Functions: assign_patient_split, get_preprocessing_stats, run_leakage_audit';
    RAISE NOTICE '============================================================';
END $$;
