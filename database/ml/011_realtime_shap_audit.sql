-- =============================================================================
-- Migration: 011_realtime_shap_audit.sql
-- Description: Extend ml_shap_analyses for real-time explanation audit trail
-- Version: 4.1.0
-- Author: E2I Causal Analytics Team
-- =============================================================================

-- Add analysis_type enum value for realtime explanations
ALTER TYPE shap_analysis_type ADD VALUE IF NOT EXISTS 'local_realtime';

-- Add columns for real-time explanation tracking
ALTER TABLE ml_shap_analyses 
ADD COLUMN IF NOT EXISTS explanation_id VARCHAR(50) UNIQUE,
ADD COLUMN IF NOT EXISTS patient_id VARCHAR(50),
ADD COLUMN IF NOT EXISTS hcp_id VARCHAR(50),
ADD COLUMN IF NOT EXISTS request_timestamp TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS response_time_ms FLOAT,
ADD COLUMN IF NOT EXISTS prediction_class VARCHAR(100),
ADD COLUMN IF NOT EXISTS prediction_probability FLOAT,
ADD COLUMN IF NOT EXISTS top_k_requested INTEGER DEFAULT 5,
ADD COLUMN IF NOT EXISTS format_requested VARCHAR(50) DEFAULT 'top_k',
ADD COLUMN IF NOT EXISTS narrative_generated BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS api_version VARCHAR(20) DEFAULT '4.1.0';

-- Create index for patient-level queries (for audit trail lookups)
CREATE INDEX IF NOT EXISTS idx_shap_patient_lookup 
ON ml_shap_analyses(patient_id, request_timestamp DESC)
WHERE analysis_type = 'local_realtime';

-- Create index for HCP-level queries  
CREATE INDEX IF NOT EXISTS idx_shap_hcp_lookup
ON ml_shap_analyses(hcp_id, request_timestamp DESC)
WHERE analysis_type = 'local_realtime' AND hcp_id IS NOT NULL;

-- Create index for model version analysis
CREATE INDEX IF NOT EXISTS idx_shap_model_analysis
ON ml_shap_analyses(model_version_id, request_timestamp DESC);

-- Create index for performance analysis
CREATE INDEX IF NOT EXISTS idx_shap_performance
ON ml_shap_analyses(response_time_ms)
WHERE analysis_type = 'local_realtime';

-- =============================================================================
-- View: Real-time explanation summary per patient
-- =============================================================================
CREATE OR REPLACE VIEW v_patient_explanation_history AS
SELECT 
    patient_id,
    COUNT(*) as total_explanations,
    COUNT(DISTINCT model_version_id) as models_used,
    AVG(response_time_ms) as avg_response_time_ms,
    MIN(request_timestamp) as first_explanation,
    MAX(request_timestamp) as last_explanation,
    -- Most common prediction
    MODE() WITHIN GROUP (ORDER BY prediction_class) as most_common_prediction,
    -- Average probability
    AVG(prediction_probability) as avg_prediction_probability
FROM ml_shap_analyses
WHERE analysis_type = 'local_realtime'
  AND patient_id IS NOT NULL
GROUP BY patient_id;

-- =============================================================================
-- View: Real-time SHAP API performance metrics
-- =============================================================================
CREATE OR REPLACE VIEW v_shap_api_performance AS
SELECT 
    DATE_TRUNC('hour', request_timestamp) as hour,
    model_type,
    COUNT(*) as request_count,
    AVG(response_time_ms) as avg_latency_ms,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY response_time_ms) as p50_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_latency_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) as p99_latency_ms,
    SUM(CASE WHEN response_time_ms > 500 THEN 1 ELSE 0 END) as slow_requests,
    COUNT(DISTINCT patient_id) as unique_patients
FROM ml_shap_analyses
WHERE analysis_type = 'local_realtime'
  AND request_timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', request_timestamp), model_type
ORDER BY hour DESC, model_type;

-- =============================================================================
-- View: Feature importance trends (aggregate SHAP across explanations)
-- =============================================================================
CREATE OR REPLACE VIEW v_feature_importance_trends AS
SELECT 
    model_version_id,
    DATE_TRUNC('day', request_timestamp) as day,
    -- Extract top feature from JSONB
    (shap_values->>'top_feature_1')::TEXT as top_feature,
    COUNT(*) as appearance_count,
    AVG((shap_values->>'top_feature_1_value')::FLOAT) as avg_shap_value
FROM ml_shap_analyses
WHERE analysis_type = 'local_realtime'
  AND request_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY model_version_id, DATE_TRUNC('day', request_timestamp), shap_values->>'top_feature_1'
ORDER BY day DESC, appearance_count DESC;

-- =============================================================================
-- Function: Get recent explanations for a patient
-- =============================================================================
CREATE OR REPLACE FUNCTION get_patient_explanations(
    p_patient_id VARCHAR(50),
    p_limit INTEGER DEFAULT 10,
    p_model_type VARCHAR(50) DEFAULT NULL
)
RETURNS TABLE (
    explanation_id VARCHAR(50),
    model_type VARCHAR(50),
    model_version_id VARCHAR(100),
    prediction_class VARCHAR(100),
    prediction_probability FLOAT,
    top_features JSONB,
    request_timestamp TIMESTAMPTZ,
    response_time_ms FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        sa.explanation_id,
        sa.model_type,
        sa.model_version_id,
        sa.prediction_class,
        sa.prediction_probability,
        sa.shap_values as top_features,
        sa.request_timestamp,
        sa.response_time_ms
    FROM ml_shap_analyses sa
    WHERE sa.patient_id = p_patient_id
      AND sa.analysis_type = 'local_realtime'
      AND (p_model_type IS NULL OR sa.model_type = p_model_type)
    ORDER BY sa.request_timestamp DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Function: Get explanation by ID (for audit)
-- =============================================================================
CREATE OR REPLACE FUNCTION get_explanation_audit(
    p_explanation_id VARCHAR(50)
)
RETURNS TABLE (
    explanation_id VARCHAR(50),
    patient_id VARCHAR(50),
    hcp_id VARCHAR(50),
    model_type VARCHAR(50),
    model_version_id VARCHAR(100),
    input_features JSONB,
    shap_values JSONB,
    prediction_class VARCHAR(100),
    prediction_probability FLOAT,
    base_value FLOAT,
    request_timestamp TIMESTAMPTZ,
    response_time_ms FLOAT,
    narrative_generated BOOLEAN,
    api_version VARCHAR(20)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        sa.explanation_id,
        sa.patient_id,
        sa.hcp_id,
        sa.model_type,
        sa.model_version_id,
        sa.input_features,
        sa.shap_values,
        sa.prediction_class,
        sa.prediction_probability,
        sa.base_value,
        sa.request_timestamp,
        sa.response_time_ms,
        sa.narrative_generated,
        sa.api_version
    FROM ml_shap_analyses sa
    WHERE sa.explanation_id = p_explanation_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- RLS Policy: Restrict explanation access by user role
-- =============================================================================
-- Enable RLS on ml_shap_analyses
ALTER TABLE ml_shap_analyses ENABLE ROW LEVEL SECURITY;

-- Policy: Allow full access for admins and data scientists
CREATE POLICY shap_admin_access ON ml_shap_analyses
    FOR ALL
    TO admin, data_scientist
    USING (TRUE)
    WITH CHECK (TRUE);

-- Policy: Field reps can only see explanations for their assigned HCPs
CREATE POLICY shap_field_rep_access ON ml_shap_analyses
    FOR SELECT
    TO field_rep
    USING (
        hcp_id IN (
            SELECT hcp_id FROM user_hcp_assignments 
            WHERE user_id = current_user_id()
        )
    );

-- Policy: HCPs can only see their own explanations
CREATE POLICY shap_hcp_access ON ml_shap_analyses
    FOR SELECT
    TO hcp_user
    USING (hcp_id = current_user_hcp_id());

-- =============================================================================
-- Trigger: Auto-populate audit metadata
-- =============================================================================
CREATE OR REPLACE FUNCTION set_shap_audit_metadata()
RETURNS TRIGGER AS $$
BEGIN
    -- Set explanation_id if not provided
    IF NEW.explanation_id IS NULL THEN
        NEW.explanation_id := 'EXPL-' || TO_CHAR(NOW(), 'YYYYMMDD') || '-' || SUBSTR(MD5(RANDOM()::TEXT), 1, 8);
    END IF;
    
    -- Set request_timestamp if not provided
    IF NEW.request_timestamp IS NULL THEN
        NEW.request_timestamp := NOW();
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_shap_audit_metadata
    BEFORE INSERT ON ml_shap_analyses
    FOR EACH ROW
    WHEN (NEW.analysis_type = 'local_realtime')
    EXECUTE FUNCTION set_shap_audit_metadata();

-- =============================================================================
-- Comments
-- =============================================================================
COMMENT ON COLUMN ml_shap_analyses.explanation_id IS 'Unique identifier for real-time explanations, used for audit trail';
COMMENT ON COLUMN ml_shap_analyses.patient_id IS 'Patient for whom the explanation was generated';
COMMENT ON COLUMN ml_shap_analyses.hcp_id IS 'HCP context (if applicable) for the explanation';
COMMENT ON COLUMN ml_shap_analyses.request_timestamp IS 'When the explanation request was received';
COMMENT ON COLUMN ml_shap_analyses.response_time_ms IS 'Total time to generate explanation in milliseconds';
COMMENT ON COLUMN ml_shap_analyses.narrative_generated IS 'Whether a natural language explanation was generated via Claude';

COMMENT ON VIEW v_patient_explanation_history IS 'Summary of all explanations generated for each patient';
COMMENT ON VIEW v_shap_api_performance IS 'Performance metrics for the real-time SHAP API';
COMMENT ON FUNCTION get_patient_explanations IS 'Retrieve recent explanations for a specific patient';
COMMENT ON FUNCTION get_explanation_audit IS 'Retrieve full audit record for a specific explanation';
