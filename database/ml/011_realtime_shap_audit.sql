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
ON ml_shap_analyses(model_registry_id, request_timestamp DESC);

-- Create index for performance analysis
CREATE INDEX IF NOT EXISTS idx_shap_performance
ON ml_shap_analyses(response_time_ms)
WHERE analysis_type = 'local_realtime';

-- =============================================================================
-- View: Real-time explanation summary per patient
-- =============================================================================
CREATE OR REPLACE VIEW v_patient_explanation_history AS
 SELECT ml_shap_analyses.patient_id,
    count(*) AS total_explanations,
    count(DISTINCT ml_shap_analyses.model_registry_id) AS models_used,
    avg(ml_shap_analyses.response_time_ms) AS avg_response_time_ms,
    min(ml_shap_analyses.request_timestamp) AS first_explanation,
    max(ml_shap_analyses.request_timestamp) AS last_explanation,
    mode() WITHIN GROUP (ORDER BY ml_shap_analyses.prediction_class) AS most_common_prediction,
    avg(ml_shap_analyses.prediction_probability) AS avg_prediction_probability
   FROM ml_shap_analyses
  WHERE ml_shap_analyses.analysis_type::text = 'local_realtime'::text AND ml_shap_analyses.patient_id IS NOT NULL
  GROUP BY ml_shap_analyses.patient_id;

-- =============================================================================
-- View: Real-time SHAP API performance metrics
-- =============================================================================
CREATE OR REPLACE VIEW v_shap_api_performance AS
 SELECT date_trunc('hour'::text, ml_shap_analyses.request_timestamp) AS hour,
    ml_shap_analyses.analysis_type,
    count(*) AS request_count,
    avg(ml_shap_analyses.response_time_ms) AS avg_latency_ms,
    percentile_cont(0.50::double precision) WITHIN GROUP (ORDER BY ml_shap_analyses.response_time_ms) AS p50_latency_ms,
    percentile_cont(0.95::double precision) WITHIN GROUP (ORDER BY ml_shap_analyses.response_time_ms) AS p95_latency_ms,
    percentile_cont(0.99::double precision) WITHIN GROUP (ORDER BY ml_shap_analyses.response_time_ms) AS p99_latency_ms,
    sum(
        CASE
            WHEN ml_shap_analyses.response_time_ms > 500::double precision THEN 1
            ELSE 0
        END) AS slow_requests,
    count(DISTINCT ml_shap_analyses.patient_id) AS unique_patients
   FROM ml_shap_analyses
  WHERE ml_shap_analyses.analysis_type::text = 'local_realtime'::text AND ml_shap_analyses.request_timestamp >= (now() - '24:00:00'::interval)
  GROUP BY (date_trunc('hour'::text, ml_shap_analyses.request_timestamp)), ml_shap_analyses.analysis_type
  ORDER BY (date_trunc('hour'::text, ml_shap_analyses.request_timestamp)) DESC, ml_shap_analyses.analysis_type;

-- =============================================================================
-- View: Feature importance trends (aggregate SHAP across explanations)
-- =============================================================================
CREATE OR REPLACE VIEW v_feature_importance_trends AS
 SELECT ml_shap_analyses.model_registry_id,
    date_trunc('day'::text, ml_shap_analyses.request_timestamp) AS day,
    ml_shap_analyses.prediction_class,
    count(*) AS explanation_count,
    avg(ml_shap_analyses.response_time_ms) AS avg_response_time_ms,
    avg(ml_shap_analyses.prediction_probability) AS avg_prediction_probability,
    count(DISTINCT ml_shap_analyses.patient_id) AS unique_patients
   FROM ml_shap_analyses
  WHERE ml_shap_analyses.analysis_type::text = 'local_realtime'::text AND ml_shap_analyses.request_timestamp >= (now() - '30 days'::interval)
  GROUP BY ml_shap_analyses.model_registry_id, (date_trunc('day'::text, ml_shap_analyses.request_timestamp)), ml_shap_analyses.prediction_class
  ORDER BY (date_trunc('day'::text, ml_shap_analyses.request_timestamp)) DESC, (count(*)) DESC;

-- =============================================================================
-- Function: Get recent explanations for a patient
-- =============================================================================
CREATE OR REPLACE FUNCTION public.get_patient_explanations(p_patient_id character varying, p_limit integer DEFAULT 10, p_analysis_type character varying DEFAULT NULL::character varying)
 RETURNS TABLE(explanation_id character varying, analysis_type character varying, model_registry_id uuid, prediction_class character varying, prediction_probability double precision, top_features jsonb, request_timestamp timestamp with time zone, response_time_ms double precision)
 LANGUAGE plpgsql
AS $function$
BEGIN
    RETURN QUERY
    SELECT 
        sa.explanation_id,
        sa.analysis_type,
        sa.model_registry_id,
        sa.prediction_class,
        sa.prediction_probability,
        sa.local_shap_values as top_features,
        sa.request_timestamp,
        sa.response_time_ms
    FROM ml_shap_analyses sa
    WHERE sa.patient_id = p_patient_id
      AND sa.analysis_type = 'local_realtime'
      AND (p_analysis_type IS NULL OR sa.analysis_type = p_analysis_type)
    ORDER BY sa.request_timestamp DESC
    LIMIT p_limit;
END;
$function$;

-- =============================================================================
-- Function: Get explanation by ID (for audit)
-- =============================================================================
CREATE OR REPLACE FUNCTION public.get_explanation_audit(p_explanation_id character varying)
 RETURNS TABLE(explanation_id character varying, patient_id character varying, hcp_id character varying, analysis_type character varying, model_registry_id uuid, local_shap_values jsonb, prediction_class character varying, prediction_probability double precision, base_value numeric, request_timestamp timestamp with time zone, response_time_ms double precision, narrative_generated boolean, api_version character varying)
 LANGUAGE plpgsql
AS $function$
BEGIN
    RETURN QUERY
    SELECT 
        sa.explanation_id,
        sa.patient_id,
        sa.hcp_id,
        sa.analysis_type,
        sa.model_registry_id,
        sa.local_shap_values,
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
$function$;

-- =============================================================================
-- RLS Policy: Restrict explanation access by user role
-- =============================================================================
-- Enable RLS on ml_shap_analyses
ALTER TABLE ml_shap_analyses ENABLE ROW LEVEL SECURITY;

-- Policy: anonymous role full access
DROP POLICY IF EXISTS shap_anon_access ON ml_shap_analyses;
CREATE POLICY shap_anon_access ON ml_shap_analyses FOR ALL TO anon USING (true) WITH CHECK (true);

-- Policy: authenticated role full access
DROP POLICY IF EXISTS shap_authenticated_access ON ml_shap_analyses;
CREATE POLICY shap_authenticated_access ON ml_shap_analyses FOR ALL TO authenticated USING (true) WITH CHECK (true);

-- Policy: service role full access
DROP POLICY IF EXISTS shap_service_access ON ml_shap_analyses;
CREATE POLICY shap_service_access ON ml_shap_analyses FOR ALL TO authenticated USING (true) WITH CHECK (true);

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

DROP TRIGGER IF EXISTS trg_shap_audit_metadata ON ml_shap_analyses;
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
-- Argument list required: get_patient_explanations is overloaded on prod
-- (a legacy 2-arg variant also exists); a bare reference would be ambiguous.
COMMENT ON FUNCTION get_patient_explanations(VARCHAR, INTEGER, VARCHAR) IS 'Retrieve recent explanations for a specific patient';
COMMENT ON FUNCTION get_explanation_audit(VARCHAR) IS 'Retrieve full audit record for a specific explanation';
