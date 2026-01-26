-- ==============================================================================
-- E2I Causal Analytics - Add Patient Journey Causal Columns
-- ==============================================================================
-- Migration: 020_add_patient_causal_columns.sql
-- Created: 2026-01-25
-- Description: Adds causal variable columns to patient_journeys table
--              Required by: Gap Analyzer, Drift Monitor, and all causal agents
-- ==============================================================================

-- Add HCP reference column
ALTER TABLE patient_journeys
ADD COLUMN IF NOT EXISTS hcp_id VARCHAR(20);

-- Add index on hcp_id for join performance
CREATE INDEX IF NOT EXISTS idx_patient_journeys_hcp_id ON patient_journeys(hcp_id);

-- Add causal confounder: disease severity (0-10 scale)
ALTER TABLE patient_journeys
ADD COLUMN IF NOT EXISTS disease_severity DECIMAL(4,2);

-- Add causal confounder: academic HCP indicator (binary)
ALTER TABLE patient_journeys
ADD COLUMN IF NOT EXISTS academic_hcp INTEGER;

-- Add treatment variable: engagement score (0-10 scale)
ALTER TABLE patient_journeys
ADD COLUMN IF NOT EXISTS engagement_score DECIMAL(4,2);

-- Add outcome variable: treatment initiated (binary)
ALTER TABLE patient_journeys
ADD COLUMN IF NOT EXISTS treatment_initiated INTEGER;

-- Add outcome timing: days to treatment (NULL if not initiated)
ALTER TABLE patient_journeys
ADD COLUMN IF NOT EXISTS days_to_treatment INTEGER;

-- Add demographic: age at diagnosis (years)
ALTER TABLE patient_journeys
ADD COLUMN IF NOT EXISTS age_at_diagnosis INTEGER;

-- Add check constraints for data quality
ALTER TABLE patient_journeys
ADD CONSTRAINT chk_disease_severity CHECK (disease_severity IS NULL OR (disease_severity >= 0 AND disease_severity <= 10));

ALTER TABLE patient_journeys
ADD CONSTRAINT chk_academic_hcp CHECK (academic_hcp IS NULL OR academic_hcp IN (0, 1));

ALTER TABLE patient_journeys
ADD CONSTRAINT chk_engagement_score CHECK (engagement_score IS NULL OR (engagement_score >= 0 AND engagement_score <= 10));

ALTER TABLE patient_journeys
ADD CONSTRAINT chk_treatment_initiated CHECK (treatment_initiated IS NULL OR treatment_initiated IN (0, 1));

ALTER TABLE patient_journeys
ADD CONSTRAINT chk_age_at_diagnosis CHECK (age_at_diagnosis IS NULL OR (age_at_diagnosis >= 0 AND age_at_diagnosis <= 120));

-- Create composite index for causal queries
CREATE INDEX IF NOT EXISTS idx_patient_journeys_causal_vars
ON patient_journeys(brand, data_split, treatment_initiated)
WHERE treatment_initiated IS NOT NULL;

-- Add comment explaining the causal structure
COMMENT ON COLUMN patient_journeys.disease_severity IS 'Confounder: Disease severity score 0-10 (affects both treatment and outcome)';
COMMENT ON COLUMN patient_journeys.academic_hcp IS 'Confounder: 1 if HCP is academic, 0 otherwise (affects treatment propensity)';
COMMENT ON COLUMN patient_journeys.engagement_score IS 'Treatment variable: Engagement level 0-10 (TRUE CAUSAL EFFECT on outcome)';
COMMENT ON COLUMN patient_journeys.treatment_initiated IS 'Outcome variable: 1 if treatment was initiated, 0 otherwise';
COMMENT ON COLUMN patient_journeys.days_to_treatment IS 'Outcome timing: Days from journey start to treatment (NULL if not initiated)';
COMMENT ON COLUMN patient_journeys.age_at_diagnosis IS 'Demographic: Patient age at diagnosis in years';
COMMENT ON COLUMN patient_journeys.hcp_id IS 'Foreign key reference to hcp_profiles for causal analysis';

-- ==============================================================================
-- End of Migration
-- ==============================================================================
