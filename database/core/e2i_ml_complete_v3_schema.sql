-- ============================================================================
-- E2I CAUSAL ANALYTICS DASHBOARD - COMPLETE SCHEMA V3.0
-- ============================================================================
-- Version: 3.0.0
-- Created: November 2025
-- Description: Complete schema addressing ALL KPI calculability gaps
--              Includes 11-agent tiered architecture
-- 
-- CHANGE LOG from V2:
--   - Added user_sessions table (WS3 Active Users gap)
--   - Added data_source_tracking table (WS1 Cross-source Match, Stacking Lift)
--   - Added ingestion_timestamps to patient_journeys (WS1 Data Lag)
--   - Added ml_annotations table (WS1 Label Quality/IAA)
--   - Added etl_pipeline_metrics table (WS1 Time-to-Release)
--   - Added model_pr_auc, rank_metrics_jsonb, brier_score to ml_predictions
--   - Added change_tracking fields to triggers (WS2 Change-Fail Rate)
--   - Added hcp_intent_surveys table (Brand-Specific Intent-to-Prescribe)
--   - Integrated 11-agent tiered architecture (from migration 006)
--   - Added reference_universe table for coverage calculations
-- ============================================================================

-- ============================================================================
-- PART 0: CLEAN DROP (Run this section ONLY if doing fresh install)
-- ============================================================================
-- Uncomment the following DROP statements for clean reinstall:

/*
DROP VIEW IF EXISTS v_holdout_causal_paths CASCADE;
DROP VIEW IF EXISTS v_test_causal_paths CASCADE;
DROP VIEW IF EXISTS v_validation_causal_paths CASCADE;
DROP VIEW IF EXISTS v_train_causal_paths CASCADE;
DROP VIEW IF EXISTS v_holdout_business_metrics CASCADE;
DROP VIEW IF EXISTS v_test_business_metrics CASCADE;
DROP VIEW IF EXISTS v_validation_business_metrics CASCADE;
DROP VIEW IF EXISTS v_train_business_metrics CASCADE;
DROP VIEW IF EXISTS v_holdout_agent_activities CASCADE;
DROP VIEW IF EXISTS v_test_agent_activities CASCADE;
DROP VIEW IF EXISTS v_validation_agent_activities CASCADE;
DROP VIEW IF EXISTS v_train_agent_activities CASCADE;
DROP VIEW IF EXISTS v_holdout_triggers CASCADE;
DROP VIEW IF EXISTS v_test_triggers CASCADE;
DROP VIEW IF EXISTS v_validation_triggers CASCADE;
DROP VIEW IF EXISTS v_train_triggers CASCADE;
DROP VIEW IF EXISTS v_holdout_ml_predictions CASCADE;
DROP VIEW IF EXISTS v_test_ml_predictions CASCADE;
DROP VIEW IF EXISTS v_validation_ml_predictions CASCADE;
DROP VIEW IF EXISTS v_train_ml_predictions CASCADE;
DROP VIEW IF EXISTS v_holdout_treatment_events CASCADE;
DROP VIEW IF EXISTS v_test_treatment_events CASCADE;
DROP VIEW IF EXISTS v_validation_treatment_events CASCADE;
DROP VIEW IF EXISTS v_train_treatment_events CASCADE;
DROP VIEW IF EXISTS v_holdout_patient_journeys CASCADE;
DROP VIEW IF EXISTS v_test_patient_journeys CASCADE;
DROP VIEW IF EXISTS v_validation_patient_journeys CASCADE;
DROP VIEW IF EXISTS v_train_patient_journeys CASCADE;
DROP VIEW IF EXISTS v_agent_routing CASCADE;

DROP TABLE IF EXISTS hcp_intent_surveys CASCADE;
DROP TABLE IF EXISTS etl_pipeline_metrics CASCADE;
DROP TABLE IF EXISTS ml_annotations CASCADE;
DROP TABLE IF EXISTS data_source_tracking CASCADE;
DROP TABLE IF EXISTS user_sessions CASCADE;
DROP TABLE IF EXISTS reference_universe CASCADE;
DROP TABLE IF EXISTS agent_registry CASCADE;
DROP TABLE IF EXISTS _agent_name_migration CASCADE;
DROP TABLE IF EXISTS causal_paths CASCADE;
DROP TABLE IF EXISTS business_metrics CASCADE;
DROP TABLE IF EXISTS agent_activities CASCADE;
DROP TABLE IF EXISTS triggers CASCADE;
DROP TABLE IF EXISTS ml_predictions CASCADE;
DROP TABLE IF EXISTS treatment_events CASCADE;
DROP TABLE IF EXISTS patient_journeys CASCADE;
DROP TABLE IF EXISTS hcp_profiles CASCADE;
DROP TABLE IF EXISTS ml_leakage_audit CASCADE;
DROP TABLE IF EXISTS ml_preprocessing_metadata CASCADE;
DROP TABLE IF EXISTS ml_patient_split_assignments CASCADE;
DROP TABLE IF EXISTS ml_split_registry CASCADE;

DROP FUNCTION IF EXISTS route_intent_to_agent CASCADE;
DROP FUNCTION IF EXISTS get_split_boundaries CASCADE;
DROP FUNCTION IF EXISTS run_leakage_audit CASCADE;
DROP FUNCTION IF EXISTS get_preprocessing_stats CASCADE;
DROP FUNCTION IF EXISTS assign_patient_split CASCADE;
DROP FUNCTION IF EXISTS update_updated_at CASCADE;

DROP TYPE IF EXISTS agent_name_type_v2 CASCADE;
DROP TYPE IF EXISTS agent_tier_type CASCADE;
DROP TYPE IF EXISTS workstream_type CASCADE;
DROP TYPE IF EXISTS agent_name_type CASCADE;
DROP TYPE IF EXISTS prediction_type CASCADE;
DROP TYPE IF EXISTS event_type CASCADE;
DROP TYPE IF EXISTS journey_status_type CASCADE;
DROP TYPE IF EXISTS journey_stage_type CASCADE;
DROP TYPE IF EXISTS priority_type CASCADE;
DROP TYPE IF EXISTS region_type CASCADE;
DROP TYPE IF EXISTS brand_type CASCADE;
DROP TYPE IF EXISTS split_strategy_type CASCADE;
DROP TYPE IF EXISTS data_split_type CASCADE;
*/

-- ============================================================================
-- PART 1: EXTENSIONS AND ENUM TYPES
-- ============================================================================

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

-- Agent tier types (NEW - 11-agent architecture)
CREATE TYPE agent_tier_type AS ENUM (
    'coordination',       -- Tier 1: orchestrator
    'causal_analytics',   -- Tier 2: causal_impact, gap_analyzer, heterogeneous_optimizer
    'monitoring',         -- Tier 3: drift_monitor, experiment_designer, health_score
    'ml_predictions',     -- Tier 4: prediction_synthesizer, resource_optimizer
    'self_improvement'    -- Tier 5: explainer, feedback_learner
);

-- Agent names (V2 - 11-agent integrated architecture)
CREATE TYPE agent_name_type_v2 AS ENUM (
    -- Tier 1: Coordination
    'orchestrator',
    -- Tier 2: Causal Analytics
    'causal_impact',
    'gap_analyzer',
    'heterogeneous_optimizer',
    -- Tier 3: Monitoring & Experimentation
    'drift_monitor',
    'experiment_designer',
    'health_score',
    -- Tier 4: ML & Predictions
    'prediction_synthesizer',
    'resource_optimizer',
    -- Tier 5: Self-Improvement
    'explainer',
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

CREATE TABLE ml_patient_split_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    split_config_id UUID NOT NULL REFERENCES ml_split_registry(split_config_id) ON DELETE CASCADE,
    patient_id VARCHAR(20) NOT NULL,
    assigned_split data_split_type NOT NULL,
    assignment_reason VARCHAR(100),
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    UNIQUE(split_config_id, patient_id)
);

CREATE TABLE ml_preprocessing_metadata (
    metadata_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    split_config_id UUID NOT NULL REFERENCES ml_split_registry(split_config_id) ON DELETE CASCADE,
    
    computed_on_split data_split_type NOT NULL DEFAULT 'train',
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Numerical feature statistics
    feature_means JSONB NOT NULL DEFAULT '{}',
    feature_stds JSONB NOT NULL DEFAULT '{}',
    feature_mins JSONB NOT NULL DEFAULT '{}',
    feature_maxs JSONB NOT NULL DEFAULT '{}',
    
    -- Categorical encodings
    categorical_encodings JSONB NOT NULL DEFAULT '{}',
    
    -- Feature distribution tracking (for drift detection)
    feature_distributions JSONB NOT NULL DEFAULT '{}',
    
    -- Additional metadata
    num_training_samples INTEGER,
    feature_list TEXT[],
    preprocessing_pipeline_version VARCHAR(20),
    
    CONSTRAINT train_only CHECK (computed_on_split = 'train'),
    UNIQUE(split_config_id)
);

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

-- -----------------------------------------------------------------------------
-- 3.1 Reference Universe (NEW - for coverage calculations)
-- -----------------------------------------------------------------------------
CREATE TABLE reference_universe (
    universe_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    universe_type VARCHAR(50) NOT NULL,  -- 'patient', 'hcp', 'territory'
    brand brand_type,
    region region_type,
    specialty VARCHAR(100),
    
    -- Counts
    total_count INTEGER NOT NULL,
    target_count INTEGER,
    
    -- Date validity
    effective_date DATE NOT NULL,
    expiration_date DATE,
    
    -- Source
    data_source VARCHAR(50),
    methodology_notes TEXT,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add unique constraint for lookups
CREATE UNIQUE INDEX idx_reference_universe_lookup 
ON reference_universe(universe_type, brand, region, specialty, effective_date);

-- -----------------------------------------------------------------------------
-- 3.2 HCP Profiles
-- -----------------------------------------------------------------------------
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

-- -----------------------------------------------------------------------------
-- 3.3 Patient Journeys (with source tracking for WS1 gaps)
-- -----------------------------------------------------------------------------
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
    
    -- NEW: Source tracking fields (WS1 Cross-source Match, Stacking Lift)
    data_source VARCHAR(50),  -- Primary source
    data_sources_matched TEXT[],  -- Array of all sources that matched this patient
    source_match_confidence DECIMAL(3,2),  -- Cross-source match confidence
    source_stacking_flag BOOLEAN DEFAULT FALSE,  -- True if multiple sources combined
    source_combination_method VARCHAR(50),  -- How sources were combined
    
    -- NEW: Timestamp tracking (WS1 Data Lag)
    source_timestamp TIMESTAMPTZ,  -- When data was generated at source
    ingestion_timestamp TIMESTAMPTZ,  -- When we received it
    data_lag_hours INTEGER,  -- Calculated lag in hours
    
    -- ML Split tracking
    data_split data_split_type NOT NULL DEFAULT 'unassigned',
    split_config_id UUID REFERENCES ml_split_registry(split_config_id),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- 3.4 Treatment Events
-- -----------------------------------------------------------------------------
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
    
    -- NEW: Source tracking
    data_source VARCHAR(50),
    source_timestamp TIMESTAMPTZ,
    
    -- ML Split tracking
    data_split data_split_type NOT NULL DEFAULT 'unassigned',
    split_config_id UUID REFERENCES ml_split_registry(split_config_id),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- 3.5 ML Predictions (with gaps filled)
-- -----------------------------------------------------------------------------
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
    
    -- Standard metrics
    model_auc DECIMAL(4,3),
    model_precision DECIMAL(4,3),
    model_recall DECIMAL(4,3),
    calibration_score DECIMAL(4,3),
    
    -- NEW: Additional metrics (WS1 Model Performance gaps)
    model_pr_auc DECIMAL(4,3),  -- PR-AUC gap
    rank_metrics JSONB DEFAULT '{}',  -- Recall@Top-K gap: {"recall_at_5": 0.85, "recall_at_10": 0.92, ...}
    brier_score DECIMAL(5,4),  -- Brier Score gap
    
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

-- -----------------------------------------------------------------------------
-- 3.6 Triggers (with change tracking for WS2 CFR gap)
-- -----------------------------------------------------------------------------
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
    
    -- NEW: Change tracking fields (WS2 Change-Fail Rate gap)
    previous_trigger_id VARCHAR(30),  -- Link to prior version if changed
    change_type VARCHAR(30),  -- 'new', 'update', 'escalation', 'downgrade'
    change_reason TEXT,
    change_timestamp TIMESTAMPTZ,
    change_failed BOOLEAN DEFAULT FALSE,  -- Did the change improve outcomes?
    change_outcome_delta DECIMAL(4,3),  -- Outcome difference vs previous
    
    -- ML Split tracking
    data_split data_split_type NOT NULL DEFAULT 'unassigned',
    split_config_id UUID REFERENCES ml_split_registry(split_config_id),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- 3.7 Agent Activities (with tier support)
-- -----------------------------------------------------------------------------
CREATE TABLE agent_activities (
    activity_id VARCHAR(30) PRIMARY KEY,
    agent_name VARCHAR(50),
    agent_tier agent_tier_type,  -- NEW: Tier classification
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

-- -----------------------------------------------------------------------------
-- 3.8 Business Metrics
-- -----------------------------------------------------------------------------
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

-- -----------------------------------------------------------------------------
-- 3.9 Causal Paths
-- -----------------------------------------------------------------------------
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
-- PART 4: NEW TABLES FOR KPI GAPS
-- ============================================================================

-- -----------------------------------------------------------------------------
-- 4.1 User Sessions (WS3 Active Users/MAU/WAU gap)
-- -----------------------------------------------------------------------------
CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(50) NOT NULL,
    user_email VARCHAR(255),
    user_role VARCHAR(50),
    user_region region_type,
    user_territory_id VARCHAR(20),
    
    -- Session timing
    session_start TIMESTAMPTZ NOT NULL,
    session_end TIMESTAMPTZ,
    session_duration_seconds INTEGER,
    
    -- Activity tracking
    page_views INTEGER DEFAULT 0,
    queries_executed INTEGER DEFAULT 0,
    triggers_viewed INTEGER DEFAULT 0,
    actions_taken INTEGER DEFAULT 0,
    exports_downloaded INTEGER DEFAULT 0,
    
    -- Context
    device_type VARCHAR(20),
    browser VARCHAR(50),
    ip_hash VARCHAR(64),
    
    -- Engagement scoring
    engagement_score DECIMAL(3,2),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_user_sessions_user ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_start ON user_sessions(session_start);
CREATE INDEX idx_user_sessions_date ON user_sessions(DATE(session_start));

-- -----------------------------------------------------------------------------
-- 4.2 Data Source Tracking (WS1 Cross-source Match, Stacking Lift)
-- -----------------------------------------------------------------------------
CREATE TABLE data_source_tracking (
    tracking_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tracking_date DATE NOT NULL,
    
    -- Source identification
    source_name VARCHAR(50) NOT NULL,  -- IQVIA_APLD, IQVIA_LAAD, HealthVerity, Komodo, Veeva
    source_type VARCHAR(30),  -- claims, lab, emr, crm
    
    -- Volume metrics
    records_received INTEGER,
    records_matched INTEGER,
    records_unique INTEGER,
    
    -- Match rates (for cross-source calculations)
    match_rate_vs_iqvia DECIMAL(4,3),
    match_rate_vs_healthverity DECIMAL(4,3),
    match_rate_vs_komodo DECIMAL(4,3),
    match_rate_vs_veeva DECIMAL(4,3),
    
    -- Stacking metrics
    stacking_eligible_records INTEGER,
    stacking_applied_records INTEGER,
    stacking_lift_percentage DECIMAL(5,2),
    
    -- Multi-source flag combinations
    source_combination_flags JSONB DEFAULT '{}',  -- {"IQVIA+HV": 1234, "IQVIA+Komodo": 567, ...}
    
    -- Quality
    data_quality_score DECIMAL(3,2),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_data_source_tracking_date ON data_source_tracking(tracking_date);
CREATE INDEX idx_data_source_tracking_source ON data_source_tracking(source_name);

-- -----------------------------------------------------------------------------
-- 4.3 ML Annotations (WS1 Label Quality/IAA gap)
-- -----------------------------------------------------------------------------
CREATE TABLE ml_annotations (
    annotation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- What was annotated
    entity_type VARCHAR(30) NOT NULL,  -- 'patient_journey', 'treatment_event', 'trigger'
    entity_id VARCHAR(30) NOT NULL,
    annotation_type VARCHAR(50) NOT NULL,  -- 'diagnosis_validation', 'outcome_label', etc.
    
    -- Annotation details
    annotator_id VARCHAR(50) NOT NULL,
    annotator_role VARCHAR(30),  -- 'physician', 'data_scientist', 'domain_expert'
    annotation_value JSONB NOT NULL,
    annotation_confidence DECIMAL(3,2),
    annotation_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Time spent (for quality metrics)
    annotation_duration_seconds INTEGER,
    
    -- Adjudication tracking
    is_adjudicated BOOLEAN DEFAULT FALSE,
    adjudication_result JSONB,
    adjudicated_by VARCHAR(50),
    adjudicated_at TIMESTAMPTZ,
    
    -- IAA (Inter-Annotator Agreement) fields
    iaa_group_id UUID,  -- Groups annotations of same item by different annotators
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ml_annotations_entity ON ml_annotations(entity_type, entity_id);
CREATE INDEX idx_ml_annotations_type ON ml_annotations(annotation_type);
CREATE INDEX idx_ml_annotations_iaa ON ml_annotations(iaa_group_id) WHERE iaa_group_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 4.4 ETL Pipeline Metrics (WS1 Time-to-Release/TTR gap)
-- -----------------------------------------------------------------------------
CREATE TABLE etl_pipeline_metrics (
    pipeline_run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_name VARCHAR(100) NOT NULL,
    pipeline_version VARCHAR(20),
    
    -- Timing
    run_start TIMESTAMPTZ NOT NULL,
    run_end TIMESTAMPTZ,
    duration_seconds INTEGER,
    
    -- Source data timing
    source_data_date DATE,  -- Date of source data
    source_data_timestamp TIMESTAMPTZ,  -- When source data was available
    
    -- TTR calculation
    time_to_release_hours DECIMAL(6,2),  -- Hours from source availability to dashboard
    
    -- Pipeline stages
    stage_timings JSONB DEFAULT '{}',  -- {"extract": 120, "transform": 300, "load": 60}
    
    -- Volume
    records_processed INTEGER,
    records_failed INTEGER,
    
    -- Status
    status VARCHAR(20),  -- 'success', 'partial', 'failed'
    error_details TEXT,
    
    -- Quality gates
    quality_checks_passed INTEGER,
    quality_checks_failed INTEGER,
    quality_check_details JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_etl_pipeline_metrics_date ON etl_pipeline_metrics(DATE(run_start));
CREATE INDEX idx_etl_pipeline_metrics_name ON etl_pipeline_metrics(pipeline_name);

-- -----------------------------------------------------------------------------
-- 4.5 HCP Intent Surveys (Brand-Specific Intent-to-Prescribe Δ gap)
-- -----------------------------------------------------------------------------
CREATE TABLE hcp_intent_surveys (
    survey_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hcp_id VARCHAR(20) REFERENCES hcp_profiles(hcp_id),
    
    -- Survey details
    survey_date DATE NOT NULL,
    survey_type VARCHAR(50),  -- 'market_research', 'detail_followup', 'conference'
    brand brand_type NOT NULL,
    
    -- Intent metrics (1-7 scale typically)
    intent_to_prescribe_score INTEGER CHECK (intent_to_prescribe_score BETWEEN 1 AND 7),
    intent_to_prescribe_change INTEGER,  -- Delta from previous survey
    awareness_score INTEGER CHECK (awareness_score BETWEEN 1 AND 7),
    favorability_score INTEGER CHECK (favorability_score BETWEEN 1 AND 7),
    
    -- Additional context
    previous_survey_id UUID,
    days_since_last_survey INTEGER,
    
    -- Intervention tracking
    interventions_since_last JSONB DEFAULT '[]',  -- What happened between surveys
    
    -- Validation
    survey_source VARCHAR(50),
    response_quality_flag BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_hcp_intent_surveys_hcp ON hcp_intent_surveys(hcp_id);
CREATE INDEX idx_hcp_intent_surveys_brand ON hcp_intent_surveys(brand);
CREATE INDEX idx_hcp_intent_surveys_date ON hcp_intent_surveys(survey_date);

-- ============================================================================
-- PART 5: AGENT REGISTRY (11-agent architecture)
-- ============================================================================

CREATE TABLE agent_registry (
    agent_name VARCHAR(50) PRIMARY KEY,
    agent_tier agent_tier_type NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    capabilities JSONB DEFAULT '[]',
    routes_from_intents JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    priority_order INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Populate agent registry
INSERT INTO agent_registry (agent_name, agent_tier, display_name, description, capabilities, routes_from_intents, priority_order) VALUES
    -- Tier 1: Coordination
    ('orchestrator', 'coordination', 'Orchestrator Agent', 
     'Query routing, multi-agent coordination, response synthesis',
     '["intent_classification", "query_decomposition", "response_synthesis", "context_management"]'::jsonb,
     '["ROUTE", "COORDINATE", "SYNTHESIZE"]'::jsonb, 1),
    
    -- Tier 2: Causal Analytics
    ('causal_impact', 'causal_analytics', 'Causal Impact Agent',
     'Traces causal chains, estimates treatment effects, validates attribution',
     '["ate_estimation", "cate_calculation", "dag_construction", "mediation_analysis"]'::jsonb,
     '["IMPACT", "CAUSAL", "EFFECT", "WHY"]'::jsonb, 10),
    
    ('gap_analyzer', 'causal_analytics', 'Gap Analyzer Agent',
     'Discovers ROI multipliers through gap analysis and opportunity identification',
     '["gap_identification", "roi_calculation", "opportunity_ranking", "benchmark_comparison"]'::jsonb,
     '["GAP", "OPPORTUNITY", "BENCHMARK", "ROI"]'::jsonb, 11),
    
    ('heterogeneous_optimizer', 'causal_analytics', 'Heterogeneous Optimizer Agent',
     'Optimizes treatment effects across patient segments',
     '["segment_discovery", "cate_optimization", "policy_learning", "uplift_modeling"]'::jsonb,
     '["SEGMENT", "PERSONALIZE", "OPTIMIZE", "UPLIFT"]'::jsonb, 12),
    
    -- Tier 3: Monitoring & Experimentation
    ('drift_monitor', 'monitoring', 'Drift Monitor Agent',
     'Detects data drift, concept drift, and model degradation',
     '["psi_calculation", "drift_detection", "alert_generation", "causal_attribution"]'::jsonb,
     '["DRIFT", "CHANGE", "DEGRADE", "ALERT"]'::jsonb, 20),
    
    ('experiment_designer', 'monitoring', 'Experiment Designer Agent',
     'Designs A/B tests with causal rigor',
     '["test_design", "power_analysis", "sample_sizing", "causal_validation"]'::jsonb,
     '["EXPERIMENT", "TEST", "AB_TEST", "VALIDATE"]'::jsonb, 21),
    
    ('health_score', 'monitoring', 'Health Score Agent',
     'Computes system health metrics',
     '["composite_scoring", "pareto_optimization", "health_alerting"]'::jsonb,
     '["HEALTH", "STATUS", "SCORE", "MONITOR"]'::jsonb, 22),
    
    -- Tier 4: ML & Predictions
    ('prediction_synthesizer', 'ml_predictions', 'Prediction Synthesizer Agent',
     'Coordinates ML model predictions and ensembles',
     '["ensemble_coordination", "model_selection", "confidence_calibration"]'::jsonb,
     '["PREDICT", "FORECAST", "MODEL", "ENSEMBLE"]'::jsonb, 30),
    
    ('resource_optimizer', 'ml_predictions', 'Resource Optimizer Agent',
     'Optimizes resource allocation by ROI',
     '["budget_optimization", "effort_routing", "roi_allocation"]'::jsonb,
     '["RESOURCE", "ALLOCATE", "BUDGET", "OPTIMIZE"]'::jsonb, 31),
    
    -- Tier 5: Self-Improvement
    ('explainer', 'self_improvement', 'Explainer Agent',
     'Generates natural language explanations and narratives',
     '["narrative_generation", "viz_explanation", "summary_generation"]'::jsonb,
     '["EXPLAIN", "DESCRIBE", "SUMMARIZE", "NARRATIVE"]'::jsonb, 40),
    
    ('feedback_learner', 'self_improvement', 'Feedback Learner Agent',
     'Learns from feedback to improve system performance',
     '["dspy_optimization", "prompt_refinement", "pattern_learning"]'::jsonb,
     '["LEARN", "IMPROVE", "FEEDBACK", "OPTIMIZE_PROMPT"]'::jsonb, 41)
ON CONFLICT (agent_name) DO UPDATE SET
    agent_tier = EXCLUDED.agent_tier,
    display_name = EXCLUDED.display_name,
    description = EXCLUDED.description,
    capabilities = EXCLUDED.capabilities,
    routes_from_intents = EXCLUDED.routes_from_intents,
    priority_order = EXCLUDED.priority_order,
    updated_at = NOW();

-- ============================================================================
-- PART 6: INDEXES
-- ============================================================================

-- Split-based indexes
CREATE INDEX idx_patient_journeys_split ON patient_journeys(data_split, split_config_id);
CREATE INDEX idx_treatment_events_split ON treatment_events(data_split, split_config_id);
CREATE INDEX idx_ml_predictions_split ON ml_predictions(data_split, split_config_id);
CREATE INDEX idx_triggers_split ON triggers(data_split, split_config_id);
CREATE INDEX idx_agent_activities_split ON agent_activities(data_split, split_config_id);
CREATE INDEX idx_business_metrics_split ON business_metrics(data_split, split_config_id);
CREATE INDEX idx_causal_paths_split ON causal_paths(data_split, split_config_id);

-- Common query indexes
CREATE INDEX idx_patient_journeys_patient ON patient_journeys(patient_id);
CREATE INDEX idx_patient_journeys_brand ON patient_journeys(brand);
CREATE INDEX idx_patient_journeys_region ON patient_journeys(geographic_region);
CREATE INDEX idx_patient_journeys_dates ON patient_journeys(journey_start_date, journey_end_date);
CREATE INDEX idx_patient_journeys_source ON patient_journeys(data_source);

CREATE INDEX idx_treatment_events_patient ON treatment_events(patient_id);
CREATE INDEX idx_treatment_events_date ON treatment_events(event_date);
CREATE INDEX idx_treatment_events_journey ON treatment_events(patient_journey_id);
CREATE INDEX idx_treatment_events_type ON treatment_events(event_type);

CREATE INDEX idx_ml_predictions_patient ON ml_predictions(patient_id);
CREATE INDEX idx_ml_predictions_timestamp ON ml_predictions(prediction_timestamp);
CREATE INDEX idx_ml_predictions_type ON ml_predictions(prediction_type);

CREATE INDEX idx_triggers_patient ON triggers(patient_id);
CREATE INDEX idx_triggers_hcp ON triggers(hcp_id);
CREATE INDEX idx_triggers_timestamp ON triggers(trigger_timestamp);
CREATE INDEX idx_triggers_status ON triggers(delivery_status, acceptance_status);
CREATE INDEX idx_triggers_change ON triggers(change_type) WHERE change_type IS NOT NULL;

CREATE INDEX idx_agent_activities_agent ON agent_activities(agent_name);
CREATE INDEX idx_agent_activities_tier ON agent_activities(agent_tier);
CREATE INDEX idx_agent_activities_workstream ON agent_activities(workstream);
CREATE INDEX idx_agent_activities_timestamp ON agent_activities(activity_timestamp);

CREATE INDEX idx_business_metrics_date ON business_metrics(metric_date);
CREATE INDEX idx_business_metrics_brand_region ON business_metrics(brand, region);

CREATE INDEX idx_patient_split_lookup ON ml_patient_split_assignments(patient_id, split_config_id);
CREATE INDEX idx_leakage_audit_config ON ml_leakage_audit(split_config_id, audit_timestamp DESC);

CREATE INDEX idx_agent_registry_tier ON agent_registry(agent_tier);
CREATE INDEX idx_agent_registry_active ON agent_registry(is_active) WHERE is_active = TRUE;

-- ============================================================================
-- PART 7: VIEWS
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

CREATE OR REPLACE VIEW v_train_causal_paths AS
SELECT * FROM causal_paths WHERE data_split = 'train';

-- Validation data views
CREATE OR REPLACE VIEW v_validation_patient_journeys AS
SELECT * FROM patient_journeys WHERE data_split = 'validation';

CREATE OR REPLACE VIEW v_validation_treatment_events AS
SELECT * FROM treatment_events WHERE data_split = 'validation';

CREATE OR REPLACE VIEW v_validation_ml_predictions AS
SELECT * FROM ml_predictions WHERE data_split = 'validation';

CREATE OR REPLACE VIEW v_validation_triggers AS
SELECT * FROM triggers WHERE data_split = 'validation';

CREATE OR REPLACE VIEW v_validation_agent_activities AS
SELECT * FROM agent_activities WHERE data_split = 'validation';

CREATE OR REPLACE VIEW v_validation_business_metrics AS
SELECT * FROM business_metrics WHERE data_split = 'validation';

CREATE OR REPLACE VIEW v_validation_causal_paths AS
SELECT * FROM causal_paths WHERE data_split = 'validation';

-- Test data views
CREATE OR REPLACE VIEW v_test_patient_journeys AS
SELECT * FROM patient_journeys WHERE data_split = 'test';

CREATE OR REPLACE VIEW v_test_treatment_events AS
SELECT * FROM treatment_events WHERE data_split = 'test';

CREATE OR REPLACE VIEW v_test_ml_predictions AS
SELECT * FROM ml_predictions WHERE data_split = 'test';

CREATE OR REPLACE VIEW v_test_triggers AS
SELECT * FROM triggers WHERE data_split = 'test';

CREATE OR REPLACE VIEW v_test_agent_activities AS
SELECT * FROM agent_activities WHERE data_split = 'test';

CREATE OR REPLACE VIEW v_test_business_metrics AS
SELECT * FROM business_metrics WHERE data_split = 'test';

CREATE OR REPLACE VIEW v_test_causal_paths AS
SELECT * FROM causal_paths WHERE data_split = 'test';

-- Holdout data views
CREATE OR REPLACE VIEW v_holdout_patient_journeys AS
SELECT * FROM patient_journeys WHERE data_split = 'holdout';

CREATE OR REPLACE VIEW v_holdout_treatment_events AS
SELECT * FROM treatment_events WHERE data_split = 'holdout';

CREATE OR REPLACE VIEW v_holdout_ml_predictions AS
SELECT * FROM ml_predictions WHERE data_split = 'holdout';

CREATE OR REPLACE VIEW v_holdout_triggers AS
SELECT * FROM triggers WHERE data_split = 'holdout';

CREATE OR REPLACE VIEW v_holdout_agent_activities AS
SELECT * FROM agent_activities WHERE data_split = 'holdout';

CREATE OR REPLACE VIEW v_holdout_business_metrics AS
SELECT * FROM business_metrics WHERE data_split = 'holdout';

CREATE OR REPLACE VIEW v_holdout_causal_paths AS
SELECT * FROM causal_paths WHERE data_split = 'holdout';

-- Agent routing view
CREATE OR REPLACE VIEW v_agent_routing AS
SELECT 
    ar.agent_name,
    ar.agent_tier,
    ar.display_name,
    ar.description,
    ar.priority_order,
    intent.value::text AS routes_from_intent
FROM agent_registry ar
CROSS JOIN LATERAL jsonb_array_elements(ar.routes_from_intents) AS intent
WHERE ar.is_active = TRUE
ORDER BY ar.priority_order, intent.value;

-- ============================================================================
-- PART 8: FUNCTIONS
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

CREATE TRIGGER update_agent_registry_timestamp
    BEFORE UPDATE ON agent_registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Function to assign patient to split based on journey date
CREATE OR REPLACE FUNCTION assign_patient_split(
    p_journey_start_date DATE,
    p_split_config_id UUID
) RETURNS data_split_type AS $$
DECLARE
    v_config ml_split_registry%ROWTYPE;
BEGIN
    SELECT * INTO v_config FROM ml_split_registry WHERE split_config_id = p_split_config_id;
    
    IF NOT FOUND THEN
        RETURN 'unassigned';
    END IF;
    
    IF p_journey_start_date <= v_config.train_end_date THEN
        RETURN 'train';
    ELSIF p_journey_start_date <= v_config.validation_end_date THEN
        RETURN 'validation';
    ELSIF p_journey_start_date <= v_config.test_end_date THEN
        RETURN 'test';
    ELSE
        RETURN 'holdout';
    END IF;
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
        key AS feature_name,
        (pm.feature_means->>key)::NUMERIC AS mean_value,
        (pm.feature_stds->>key)::NUMERIC AS std_value,
        (pm.feature_mins->>key)::NUMERIC AS min_value,
        (pm.feature_maxs->>key)::NUMERIC AS max_value
    FROM ml_preprocessing_metadata pm,
         LATERAL jsonb_object_keys(pm.feature_means) AS key
    WHERE pm.split_config_id = p_split_config_id;
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
    v_temporal_violations INTEGER;
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

    -- Check 3: Temporal boundary violations
    SELECT COUNT(*) INTO v_temporal_violations
    FROM patient_journeys pj
    JOIN ml_split_registry sr ON pj.split_config_id = sr.split_config_id
    WHERE pj.split_config_id = p_split_config_id
    AND (
        (pj.data_split = 'train' AND pj.journey_start_date > sr.train_end_date) OR
        (pj.data_split = 'validation' AND (pj.journey_start_date <= sr.train_end_date OR pj.journey_start_date > sr.validation_end_date)) OR
        (pj.data_split = 'test' AND (pj.journey_start_date <= sr.validation_end_date OR pj.journey_start_date > sr.test_end_date))
    );

    INSERT INTO ml_leakage_audit (
        split_config_id, check_type, passed, severity, details, affected_records
    ) VALUES (
        p_split_config_id,
        'temporal_boundaries',
        v_temporal_violations = 0,
        CASE WHEN v_temporal_violations = 0 THEN 'info' ELSE 'critical' END,
        CASE WHEN v_temporal_violations = 0 
            THEN 'All records respect temporal split boundaries'
            ELSE format('%s records violate temporal boundaries', v_temporal_violations)
        END,
        v_temporal_violations
    );

    RETURN QUERY SELECT 
        'temporal_boundaries'::VARCHAR,
        v_temporal_violations = 0,
        CASE WHEN v_temporal_violations = 0 THEN 'info' ELSE 'critical' END::VARCHAR,
        CASE WHEN v_temporal_violations = 0 
            THEN 'All records respect temporal split boundaries'
            ELSE format('%s records violate temporal boundaries', v_temporal_violations)
        END::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Function to get split boundaries
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

-- Function to route intent to agent
CREATE OR REPLACE FUNCTION route_intent_to_agent(p_intent TEXT)
RETURNS TABLE(agent_name VARCHAR, agent_tier agent_tier_type, priority INTEGER) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ar.agent_name,
        ar.agent_tier,
        ar.priority_order
    FROM agent_registry ar
    WHERE ar.is_active = TRUE
      AND ar.routes_from_intents ? UPPER(p_intent)
    ORDER BY ar.priority_order
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PART 9: KPI CALCULATION HELPER VIEWS
-- ============================================================================

-- WS1: Cross-source Match Rate view
CREATE OR REPLACE VIEW v_kpi_cross_source_match AS
SELECT 
    tracking_date,
    source_name,
    records_received,
    records_matched,
    CASE WHEN records_received > 0 
         THEN records_matched::DECIMAL / records_received 
         ELSE 0 END AS match_rate
FROM data_source_tracking;

-- WS1: Stacking Lift view
CREATE OR REPLACE VIEW v_kpi_stacking_lift AS
SELECT 
    tracking_date,
    SUM(stacking_eligible_records) AS total_eligible,
    SUM(stacking_applied_records) AS total_stacked,
    AVG(stacking_lift_percentage) AS avg_lift_pct
FROM data_source_tracking
GROUP BY tracking_date;

-- WS1: Data Lag view
CREATE OR REPLACE VIEW v_kpi_data_lag AS
SELECT 
    DATE(created_at) AS report_date,
    data_source,
    AVG(data_lag_hours) AS avg_lag_hours,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY data_lag_hours) AS median_lag_hours,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY data_lag_hours) AS p95_lag_hours
FROM patient_journeys
WHERE data_lag_hours IS NOT NULL
GROUP BY DATE(created_at), data_source;

-- WS1: Label Quality (IAA) view
CREATE OR REPLACE VIEW v_kpi_label_quality AS
SELECT 
    annotation_type,
    COUNT(DISTINCT iaa_group_id) AS annotation_groups,
    COUNT(*) AS total_annotations,
    AVG(annotation_confidence) AS avg_confidence,
    SUM(CASE WHEN is_adjudicated THEN 1 ELSE 0 END)::DECIMAL / 
        NULLIF(COUNT(*), 0) AS adjudication_rate
FROM ml_annotations
GROUP BY annotation_type;

-- WS1: Time-to-Release view
CREATE OR REPLACE VIEW v_kpi_time_to_release AS
SELECT 
    DATE(run_start) AS run_date,
    pipeline_name,
    AVG(time_to_release_hours) AS avg_ttr_hours,
    MIN(time_to_release_hours) AS min_ttr_hours,
    MAX(time_to_release_hours) AS max_ttr_hours
FROM etl_pipeline_metrics
WHERE status = 'success'
GROUP BY DATE(run_start), pipeline_name;

-- WS2: Change-Fail Rate view
CREATE OR REPLACE VIEW v_kpi_change_fail_rate AS
SELECT 
    DATE(change_timestamp) AS change_date,
    change_type,
    COUNT(*) AS total_changes,
    SUM(CASE WHEN change_failed THEN 1 ELSE 0 END) AS failed_changes,
    SUM(CASE WHEN change_failed THEN 1 ELSE 0 END)::DECIMAL / 
        NULLIF(COUNT(*), 0) AS fail_rate
FROM triggers
WHERE change_type IS NOT NULL
GROUP BY DATE(change_timestamp), change_type;

-- WS3: Active Users (MAU/WAU) view
CREATE OR REPLACE VIEW v_kpi_active_users AS
SELECT 
    DATE_TRUNC('month', session_start) AS month,
    COUNT(DISTINCT user_id) AS monthly_active_users,
    COUNT(DISTINCT CASE 
        WHEN session_start >= DATE_TRUNC('week', NOW()) 
        THEN user_id END) AS weekly_active_users,
    COUNT(DISTINCT CASE 
        WHEN session_start >= DATE_TRUNC('day', NOW()) 
        THEN user_id END) AS daily_active_users
FROM user_sessions
GROUP BY DATE_TRUNC('month', session_start);

-- Brand-Specific: Intent-to-Prescribe Δ view
CREATE OR REPLACE VIEW v_kpi_intent_to_prescribe AS
SELECT 
    brand,
    DATE_TRUNC('month', survey_date) AS survey_month,
    AVG(intent_to_prescribe_score) AS avg_intent_score,
    AVG(intent_to_prescribe_change) AS avg_intent_change,
    COUNT(*) AS survey_count
FROM hcp_intent_surveys
WHERE response_quality_flag = TRUE
GROUP BY brand, DATE_TRUNC('month', survey_date);

-- ============================================================================
-- PART 10: COMMENTS
-- ============================================================================

COMMENT ON TABLE user_sessions IS 
'Tracks user sessions for MAU/WAU/DAU calculations (WS3 Active Users KPI)';

COMMENT ON TABLE data_source_tracking IS 
'Daily tracking of data source volumes and match rates (WS1 Cross-source Match, Stacking Lift KPIs)';

COMMENT ON TABLE ml_annotations IS 
'Human annotations for ML training with IAA tracking (WS1 Label Quality KPI)';

COMMENT ON TABLE etl_pipeline_metrics IS 
'ETL pipeline run metrics including TTR (WS1 Time-to-Release KPI)';

COMMENT ON TABLE hcp_intent_surveys IS 
'HCP intent survey data for prescribing intent tracking (Brand-Specific Intent-to-Prescribe Δ KPI)';

COMMENT ON TABLE agent_registry IS 
'Runtime configuration for 11-agent tiered architecture. Used for intent routing and capability discovery.';

COMMENT ON COLUMN patient_journeys.data_sources_matched IS 
'Array of all data sources that matched this patient record (for cross-source match calculation)';

COMMENT ON COLUMN patient_journeys.source_stacking_flag IS 
'True if patient record combines data from multiple sources (stacking)';

COMMENT ON COLUMN patient_journeys.data_lag_hours IS 
'Hours between source_timestamp and ingestion_timestamp (for data lag KPI)';

COMMENT ON COLUMN ml_predictions.model_pr_auc IS 
'Precision-Recall AUC for imbalanced class performance (WS1 Model Performance KPI)';

COMMENT ON COLUMN ml_predictions.rank_metrics IS 
'JSONB containing Recall@Top-K metrics: {"recall_at_5": 0.85, "recall_at_10": 0.92}';

COMMENT ON COLUMN ml_predictions.brier_score IS 
'Brier score for probability calibration quality (WS1 Model Performance KPI)';

COMMENT ON COLUMN triggers.change_failed IS 
'Whether a trigger change resulted in worse outcomes than previous version (WS2 Change-Fail Rate KPI)';

-- ============================================================================
-- SCHEMA COMPLETE - V3.0
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'E2I Causal Analytics Schema V3.0 Created Successfully!';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '';
    RAISE NOTICE 'CORE TABLES:';
    RAISE NOTICE '  - ml_split_registry, ml_patient_split_assignments';
    RAISE NOTICE '  - ml_preprocessing_metadata, ml_leakage_audit';
    RAISE NOTICE '  - reference_universe, hcp_profiles';
    RAISE NOTICE '  - patient_journeys (with source tracking)';
    RAISE NOTICE '  - treatment_events, ml_predictions (with PR-AUC, Brier)';
    RAISE NOTICE '  - triggers (with change tracking)';
    RAISE NOTICE '  - agent_activities (with tier), business_metrics, causal_paths';
    RAISE NOTICE '';
    RAISE NOTICE 'NEW TABLES FOR KPI GAPS:';
    RAISE NOTICE '  - user_sessions (WS3 Active Users)';
    RAISE NOTICE '  - data_source_tracking (WS1 Cross-source, Stacking)';
    RAISE NOTICE '  - ml_annotations (WS1 Label Quality/IAA)';
    RAISE NOTICE '  - etl_pipeline_metrics (WS1 Time-to-Release)';
    RAISE NOTICE '  - hcp_intent_surveys (Brand Intent-to-Prescribe)';
    RAISE NOTICE '  - agent_registry (11-agent architecture)';
    RAISE NOTICE '';
    RAISE NOTICE 'KPI HELPER VIEWS:';
    RAISE NOTICE '  - v_kpi_cross_source_match, v_kpi_stacking_lift';
    RAISE NOTICE '  - v_kpi_data_lag, v_kpi_label_quality';
    RAISE NOTICE '  - v_kpi_time_to_release, v_kpi_change_fail_rate';
    RAISE NOTICE '  - v_kpi_active_users, v_kpi_intent_to_prescribe';
    RAISE NOTICE '';
    RAISE NOTICE 'SPLIT VIEWS: v_train_*, v_validation_*, v_test_*, v_holdout_*';
    RAISE NOTICE 'FUNCTIONS: assign_patient_split, run_leakage_audit, route_intent_to_agent';
    RAISE NOTICE '============================================================';
END $$;
