-- ============================================================================
-- E2I Causal Analytics - Supabase Schema for ML-Compliant Data Splits
-- ============================================================================
-- This migration adds proper train/validation/test split tracking to prevent
-- data leakage in machine learning pipelines.
--
-- Key Features:
-- 1. data_split column on core tables for efficient querying
-- 2. Central split_registry for configuration management
-- 3. Preprocessing metadata storage
-- 4. Audit trail for compliance
-- 5. Row-level security policies by split
-- ============================================================================

-- ============================================================================
-- PART 1: ENUM TYPES AND CORE INFRASTRUCTURE
-- ============================================================================

-- Create enum for data splits
CREATE TYPE data_split_type AS ENUM ('train', 'validation', 'test', 'holdout', 'unassigned');

-- Create enum for split strategies
CREATE TYPE split_strategy_type AS ENUM (
    'chronological',        -- Time-based split
    'patient_stratified',   -- Patient-level isolation
    'rolling_window',       -- Time-series CV
    'causal_holdout'        -- Pre/post intervention
);

-- ============================================================================
-- PART 2: SPLIT REGISTRY AND CONFIGURATION
-- ============================================================================

-- Central registry for split configurations
CREATE TABLE IF NOT EXISTS ml_split_registry (
    split_config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_name VARCHAR(100) NOT NULL UNIQUE,
    config_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    
    -- Split ratios
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
    
    -- Configuration
    temporal_gap_days INTEGER NOT NULL DEFAULT 7,
    patient_level_isolation BOOLEAN NOT NULL DEFAULT TRUE,
    split_strategy split_strategy_type NOT NULL DEFAULT 'chronological',
    random_seed INTEGER,
    
    -- Metadata
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100),
    notes TEXT,
    
    -- Validation
    CONSTRAINT valid_ratios CHECK (
        train_ratio + validation_ratio + test_ratio + holdout_ratio = 1.0
    ),
    CONSTRAINT valid_dates CHECK (
        data_start_date < train_end_date AND
        train_end_date < validation_end_date AND
        validation_end_date < test_end_date AND
        test_end_date <= data_end_date
    )
);

-- Patient-to-split assignment table (for flexible re-splitting)
CREATE TABLE IF NOT EXISTS ml_patient_split_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    split_config_id UUID NOT NULL REFERENCES ml_split_registry(split_config_id),
    patient_id VARCHAR(20) NOT NULL,
    assigned_split data_split_type NOT NULL,
    assignment_reason VARCHAR(100), -- e.g., 'chronological', 'stratified'
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Ensure each patient has ONE assignment per config
    UNIQUE(split_config_id, patient_id)
);

-- Index for fast lookups
CREATE INDEX idx_patient_split_lookup 
ON ml_patient_split_assignments(patient_id, split_config_id);

-- ============================================================================
-- PART 3: PREPROCESSING METADATA STORAGE
-- ============================================================================

-- Store preprocessing statistics (computed on training data only)
CREATE TABLE IF NOT EXISTS ml_preprocessing_metadata (
    metadata_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    split_config_id UUID NOT NULL REFERENCES ml_split_registry(split_config_id),
    
    -- Source verification
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
    
    -- Validation
    CONSTRAINT train_only CHECK (computed_on_split = 'train')
);

-- ============================================================================
-- PART 4: ADD SPLIT COLUMNS TO EXISTING TABLES
-- ============================================================================

-- Add data_split to patient_journeys
ALTER TABLE patient_journeys 
ADD COLUMN IF NOT EXISTS data_split data_split_type DEFAULT 'unassigned',
ADD COLUMN IF NOT EXISTS split_config_id UUID REFERENCES ml_split_registry(split_config_id);

-- Add data_split to treatment_events
ALTER TABLE treatment_events 
ADD COLUMN IF NOT EXISTS data_split data_split_type DEFAULT 'unassigned',
ADD COLUMN IF NOT EXISTS split_config_id UUID REFERENCES ml_split_registry(split_config_id);

-- Add data_split to ml_predictions
ALTER TABLE ml_predictions 
ADD COLUMN IF NOT EXISTS data_split data_split_type DEFAULT 'unassigned',
ADD COLUMN IF NOT EXISTS split_config_id UUID REFERENCES ml_split_registry(split_config_id),
ADD COLUMN IF NOT EXISTS features_available_at_prediction JSONB DEFAULT '{}';

-- Add data_split to triggers
ALTER TABLE triggers 
ADD COLUMN IF NOT EXISTS data_split data_split_type DEFAULT 'unassigned',
ADD COLUMN IF NOT EXISTS split_config_id UUID REFERENCES ml_split_registry(split_config_id);

-- Add data_split to agent_activities
ALTER TABLE agent_activities 
ADD COLUMN IF NOT EXISTS data_split data_split_type DEFAULT 'unassigned',
ADD COLUMN IF NOT EXISTS split_config_id UUID REFERENCES ml_split_registry(split_config_id);

-- Add data_split to business_metrics
ALTER TABLE business_metrics 
ADD COLUMN IF NOT EXISTS data_split data_split_type DEFAULT 'unassigned',
ADD COLUMN IF NOT EXISTS split_config_id UUID REFERENCES ml_split_registry(split_config_id);

-- ============================================================================
-- PART 5: INDEXES FOR SPLIT-BASED QUERIES
-- ============================================================================

-- Create indexes for efficient split-based queries
CREATE INDEX IF NOT EXISTS idx_patient_journeys_split 
ON patient_journeys(data_split, split_config_id);

CREATE INDEX IF NOT EXISTS idx_treatment_events_split 
ON treatment_events(data_split, split_config_id);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_split 
ON ml_predictions(data_split, split_config_id);

CREATE INDEX IF NOT EXISTS idx_triggers_split 
ON triggers(data_split, split_config_id);

CREATE INDEX IF NOT EXISTS idx_agent_activities_split 
ON agent_activities(data_split, split_config_id);

CREATE INDEX IF NOT EXISTS idx_business_metrics_split 
ON business_metrics(data_split, split_config_id);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_patient_journeys_split_brand 
ON patient_journeys(data_split, brand);

CREATE INDEX IF NOT EXISTS idx_treatment_events_split_date 
ON treatment_events(data_split, event_date);

-- ============================================================================
-- PART 6: VIEWS FOR EASY SPLIT ACCESS
-- ============================================================================

-- Training data view
CREATE OR REPLACE VIEW v_train_patient_journeys AS
SELECT * FROM patient_journeys WHERE data_split = 'train';

CREATE OR REPLACE VIEW v_train_treatment_events AS
SELECT * FROM treatment_events WHERE data_split = 'train';

CREATE OR REPLACE VIEW v_train_ml_predictions AS
SELECT * FROM ml_predictions WHERE data_split = 'train';

CREATE OR REPLACE VIEW v_train_triggers AS
SELECT * FROM triggers WHERE data_split = 'train';

-- Validation data view
CREATE OR REPLACE VIEW v_validation_patient_journeys AS
SELECT * FROM patient_journeys WHERE data_split = 'validation';

CREATE OR REPLACE VIEW v_validation_treatment_events AS
SELECT * FROM treatment_events WHERE data_split = 'validation';

CREATE OR REPLACE VIEW v_validation_ml_predictions AS
SELECT * FROM ml_predictions WHERE data_split = 'validation';

CREATE OR REPLACE VIEW v_validation_triggers AS
SELECT * FROM triggers WHERE data_split = 'validation';

-- Test data view
CREATE OR REPLACE VIEW v_test_patient_journeys AS
SELECT * FROM patient_journeys WHERE data_split = 'test';

CREATE OR REPLACE VIEW v_test_treatment_events AS
SELECT * FROM treatment_events WHERE data_split = 'test';

CREATE OR REPLACE VIEW v_test_ml_predictions AS
SELECT * FROM ml_predictions WHERE data_split = 'test';

CREATE OR REPLACE VIEW v_test_triggers AS
SELECT * FROM triggers WHERE data_split = 'test';

-- Holdout data view (restricted access recommended)
CREATE OR REPLACE VIEW v_holdout_patient_journeys AS
SELECT * FROM patient_journeys WHERE data_split = 'holdout';

-- ============================================================================
-- PART 7: LEAKAGE AUDIT TRAIL
-- ============================================================================

-- Audit table for tracking potential leakage issues
CREATE TABLE IF NOT EXISTS ml_leakage_audit (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    split_config_id UUID REFERENCES ml_split_registry(split_config_id),
    audit_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    check_type VARCHAR(50) NOT NULL,
    passed BOOLEAN NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'warning', 'info')),
    details TEXT,
    affected_records INTEGER,
    remediation_action TEXT,
    audited_by VARCHAR(100)
);

-- Index for audit queries
CREATE INDEX idx_leakage_audit_config 
ON ml_leakage_audit(split_config_id, audit_timestamp DESC);

-- ============================================================================
-- PART 8: HELPER FUNCTIONS
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
BEGIN
    -- Get split configuration
    SELECT * INTO v_config 
    FROM ml_split_registry 
    WHERE split_config_id = p_split_config_id;
    
    -- Check if patient already assigned
    SELECT assigned_split INTO v_assigned_split
    FROM ml_patient_split_assignments
    WHERE patient_id = p_patient_id 
    AND split_config_id = p_split_config_id;
    
    IF v_assigned_split IS NOT NULL THEN
        RETURN v_assigned_split;
    END IF;
    
    -- Assign based on date (with gaps)
    IF p_journey_start_date < v_config.train_end_date THEN
        v_assigned_split := 'train';
    ELSIF p_journey_start_date >= (v_config.train_end_date + v_config.temporal_gap_days) 
          AND p_journey_start_date < v_config.validation_end_date THEN
        v_assigned_split := 'validation';
    ELSIF p_journey_start_date >= (v_config.validation_end_date + v_config.temporal_gap_days)
          AND p_journey_start_date < v_config.test_end_date THEN
        v_assigned_split := 'test';
    ELSIF p_journey_start_date >= (v_config.test_end_date + v_config.temporal_gap_days) THEN
        v_assigned_split := 'holdout';
    ELSE
        -- In gap period - assign to earlier split
        IF p_journey_start_date < (v_config.train_end_date + v_config.temporal_gap_days) THEN
            v_assigned_split := 'train';
        ELSIF p_journey_start_date < (v_config.validation_end_date + v_config.temporal_gap_days) THEN
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
    );
    
    RETURN v_assigned_split;
END;
$$ LANGUAGE plpgsql;

-- Function to run leakage audit
CREATE OR REPLACE FUNCTION run_leakage_audit(
    p_split_config_id UUID
) RETURNS TABLE(check_type VARCHAR, passed BOOLEAN, details TEXT) AS $$
DECLARE
    v_duplicate_patients INTEGER;
    v_boundary_violations INTEGER;
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
        split_config_id, check_type, passed, severity, details
    ) VALUES (
        p_split_config_id, 
        'patient_split_isolation',
        v_duplicate_patients = 0,
        CASE WHEN v_duplicate_patients = 0 THEN 'info' ELSE 'critical' END,
        CASE WHEN v_duplicate_patients = 0 
            THEN 'All patients correctly isolated to single split'
            ELSE format('%s patients found in multiple splits', v_duplicate_patients)
        END
    );
    
    RETURN QUERY SELECT 
        'patient_split_isolation'::VARCHAR,
        v_duplicate_patients = 0,
        CASE WHEN v_duplicate_patients = 0 
            THEN 'All patients correctly isolated'
            ELSE format('%s patients in multiple splits', v_duplicate_patients)
        END;
    
    -- Check 2: Preprocessing metadata source
    RETURN QUERY SELECT 
        'preprocessing_source'::VARCHAR,
        EXISTS(
            SELECT 1 FROM ml_preprocessing_metadata 
            WHERE split_config_id = p_split_config_id 
            AND computed_on_split = 'train'
        ),
        'Preprocessing metadata computed on training data only'::TEXT;
    
END;
$$ LANGUAGE plpgsql;

-- Function to get preprocessing stats for a split config
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
        (feature_means->>key)::NUMERIC as mean_value,
        (feature_stds->>key)::NUMERIC as std_value,
        (feature_mins->>key)::NUMERIC as min_value,
        (feature_maxs->>key)::NUMERIC as max_value
    FROM ml_preprocessing_metadata,
    LATERAL jsonb_object_keys(feature_means) as key
    WHERE split_config_id = p_split_config_id
    AND computed_on_split = 'train';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PART 9: ROW-LEVEL SECURITY POLICIES (Optional but recommended)
-- ============================================================================

-- Enable RLS on sensitive tables
ALTER TABLE patient_journeys ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_predictions ENABLE ROW LEVEL SECURITY;

-- Policy: Analysts can only see train/validation during development
CREATE POLICY analyst_train_val_only ON patient_journeys
    FOR SELECT
    USING (
        data_split IN ('train', 'validation')
        OR current_setting('app.role', true) = 'data_scientist'
    );

-- Policy: Only data scientists can access test/holdout
CREATE POLICY ds_full_access ON patient_journeys
    FOR SELECT
    USING (current_setting('app.role', true) = 'data_scientist');

-- ============================================================================
-- PART 10: SAMPLE DATA INSERTION
-- ============================================================================

-- Insert default split configuration
INSERT INTO ml_split_registry (
    config_name,
    config_version,
    train_ratio,
    validation_ratio,
    test_ratio,
    holdout_ratio,
    data_start_date,
    data_end_date,
    train_end_date,
    validation_end_date,
    test_end_date,
    temporal_gap_days,
    patient_level_isolation,
    split_strategy,
    random_seed,
    created_by,
    notes
) VALUES (
    'e2i_pilot_v1',
    '1.0.0',
    0.60,
    0.20,
    0.15,
    0.05,
    '2024-01-01',
    '2025-09-28',
    '2025-01-16',
    '2025-05-30',
    '2025-09-09',
    7,
    TRUE,
    'chronological',
    42,
    'system',
    'Initial ML-compliant split configuration for E2I pilot'
) ON CONFLICT (config_name) DO NOTHING;

-- ============================================================================
-- PART 11: UTILITY QUERIES FOR PIPELINE
-- ============================================================================

-- View: Split statistics summary
CREATE OR REPLACE VIEW v_split_statistics AS
SELECT 
    sr.config_name,
    sr.split_strategy,
    pj.data_split,
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
-- MIGRATION COMPLETE
-- ============================================================================

COMMENT ON TABLE ml_split_registry IS 
'Central registry for ML data split configurations. Ensures consistent train/validation/test/holdout splits across the platform.';

COMMENT ON TABLE ml_patient_split_assignments IS 
'Tracks which patients are assigned to which split. Ensures patient-level isolation to prevent data leakage.';

COMMENT ON TABLE ml_preprocessing_metadata IS 
'Stores preprocessing statistics computed ONLY on training data. Critical for preventing preprocessing leakage.';

COMMENT ON TABLE ml_leakage_audit IS 
'Audit trail for data leakage checks. Important for pharmaceutical compliance and model validation.';

COMMENT ON FUNCTION assign_patient_split IS 
'Assigns a patient to a data split based on their journey start date. Ensures chronological ordering and patient isolation.';

COMMENT ON FUNCTION run_leakage_audit IS 
'Runs comprehensive data leakage audit checks. Returns pass/fail status for each check type.';
