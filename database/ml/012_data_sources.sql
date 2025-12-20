-- ═══════════════════════════════════════════════════════════════════
-- E2I Causal Analytics - Data Sources Reference Table
-- Migration: 012_data_sources.sql
-- Version: V4.1
-- Purpose: Formalize data source tracking with quality metrics
-- Gap: Gap 7 - Data Sources Schema
-- ═══════════════════════════════════════════════════════════════════

-- Create ENUM for source types
CREATE TYPE source_type_enum AS ENUM (
    'claims',      -- Medical claims data
    'lab',         -- Laboratory test results
    'emr',         -- Electronic medical records
    'crm',         -- Customer relationship management
    'specialty',   -- Specialty pharmacy data
    'registry'     -- Patient registries
);

-- ═══════════════════════════════════════════════════════════════════
-- DATA SOURCES TABLE
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE data_sources (
    -- Primary identification
    source_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL UNIQUE,
    source_type source_type_enum NOT NULL,
    vendor TEXT,

    -- Quality metrics
    coverage_percent DECIMAL(5,2) CHECK (coverage_percent >= 0 AND coverage_percent <= 100),
    completeness_score DECIMAL(5,4) CHECK (completeness_score >= 0 AND completeness_score <= 1),
    freshness_days INTEGER CHECK (freshness_days >= 0),
    match_rate DECIMAL(5,4) CHECK (match_rate >= 0 AND match_rate <= 1),

    -- Data characteristics
    patient_count INTEGER,
    hcp_count INTEGER,
    record_count BIGINT,
    date_range_start DATE,
    date_range_end DATE,

    -- Integration metadata
    integration_method TEXT,  -- 'api', 'batch', 'stream', 'manual'
    refresh_frequency TEXT,   -- 'daily', 'weekly', 'monthly', 'quarterly'
    last_refresh_date TIMESTAMPTZ,
    next_scheduled_refresh TIMESTAMPTZ,

    -- Status and control
    is_active BOOLEAN DEFAULT true,
    is_primary BOOLEAN DEFAULT false,  -- Primary source for data type
    priority INTEGER DEFAULT 10,       -- Lower = higher priority

    -- Lineage and governance
    data_owner TEXT,
    compliance_status TEXT,  -- 'approved', 'pending', 'deprecated'
    phi_flag BOOLEAN DEFAULT false,
    pii_flag BOOLEAN DEFAULT false,

    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT,
    updated_by TEXT,

    -- Documentation
    description TEXT,
    documentation_url TEXT,
    contact_email TEXT
);

-- ═══════════════════════════════════════════════════════════════════
-- SEED DATA - Major Healthcare Data Sources
-- ═══════════════════════════════════════════════════════════════════

INSERT INTO data_sources (
    source_id,
    source_name,
    source_type,
    vendor,
    coverage_percent,
    completeness_score,
    freshness_days,
    match_rate,
    patient_count,
    hcp_count,
    integration_method,
    refresh_frequency,
    is_active,
    is_primary,
    priority,
    description
) VALUES
-- IQVIA APLD (Anonymized Patient-Level Data)
(
    'iqvia_apld',
    'IQVIA APLD',
    'claims',
    'IQVIA',
    92.5,
    0.9450,
    7,       -- Weekly refresh
    0.8900,
    1500000, -- 1.5M patients
    45000,   -- 45K HCPs
    'batch',
    'weekly',
    true,
    true,    -- Primary claims source
    1,
    'IQVIA Anonymized Patient-Level Data - comprehensive claims and prescription data covering 92.5% of US retail pharmacies'
),

-- IQVIA LAAD (Longitudinal Access and Adjudication Data)
(
    'iqvia_laad',
    'IQVIA LAAD',
    'lab',
    'IQVIA',
    85.0,
    0.9200,
    14,      -- Biweekly refresh
    0.8500,
    1200000, -- 1.2M patients
    38000,   -- 38K HCPs
    'batch',
    'biweekly',
    true,
    true,    -- Primary lab source
    2,
    'IQVIA Longitudinal Access and Adjudication Data - lab test results and diagnostic data'
),

-- HealthVerity
(
    'healthverity',
    'HealthVerity',
    'claims',
    'HealthVerity',
    88.0,
    0.9350,
    10,      -- ~Weekly refresh
    0.8700,
    1800000, -- 1.8M patients
    52000,   -- 52K HCPs
    'api',
    'weekly',
    true,
    false,   -- Secondary claims source
    3,
    'HealthVerity privacy-safe data exchange platform - de-identified claims, EMR, and lab data'
),

-- Komodo Health
(
    'komodo',
    'Komodo Health',
    'emr',
    'Komodo',
    78.0,
    0.8900,
    21,      -- ~Monthly refresh
    0.8200,
    800000,  -- 800K patients
    28000,   -- 28K HCPs
    'api',
    'monthly',
    true,
    true,    -- Primary EMR source
    4,
    'Komodo Health Healthcare Map - real-world patient journey data from EMR and claims'
),

-- Veeva OCE (Omnichannel Customer Engagement)
(
    'veeva_oce',
    'Veeva OCE',
    'crm',
    'Veeva',
    95.0,
    0.9800,
    1,       -- Daily refresh
    0.9500,
    NULL,    -- Not patient-centric
    65000,   -- 65K HCPs
    'api',
    'daily',
    true,
    true,    -- Primary CRM source
    5,
    'Veeva CRM data - sales rep activities, HCP interactions, sample distribution, call notes'
),

-- Specialty Pharmacy Network
(
    'specialty_pharmacy',
    'Specialty Pharmacy Network',
    'specialty',
    'Multiple',
    72.0,
    0.9100,
    7,       -- Weekly refresh
    0.7800,
    450000,  -- 450K patients
    15000,   -- 15K HCPs
    'batch',
    'weekly',
    true,
    true,    -- Primary specialty source
    6,
    'Aggregated specialty pharmacy data - high-cost medications, patient support programs, adherence data'
),

-- Patient Registry (Internal)
(
    'internal_registry',
    'Internal Patient Registry',
    'registry',
    'Internal',
    100.0,   -- Full coverage of enrolled patients
    0.9950,
    1,       -- Daily updates
    1.0000,  -- Perfect match (it''s our data)
    125000,  -- 125K enrolled patients
    18000,   -- 18K participating HCPs
    'stream',
    'realtime',
    true,
    false,   -- Supplementary source
    10,
    'Internal patient enrollment and program participation registry'
);

-- ═══════════════════════════════════════════════════════════════════
-- UPDATE EXISTING TABLES WITH FOREIGN KEY REFERENCE
-- ═══════════════════════════════════════════════════════════════════

-- Add source_id to data_source_tracking table (if not already present)
-- This links tracking entries to the reference table
DO $$
BEGIN
    -- Check if column doesn''t exist before adding
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'data_source_tracking'
          AND column_name = 'source_id'
    ) THEN
        ALTER TABLE data_source_tracking
        ADD COLUMN source_id TEXT REFERENCES data_sources(source_id);

        -- Create index for foreign key
        CREATE INDEX idx_data_source_tracking_source_id
        ON data_source_tracking(source_id);
    END IF;
END
$$;

-- ═══════════════════════════════════════════════════════════════════
-- INDEXES
-- ═══════════════════════════════════════════════════════════════════

-- Performance indexes
CREATE INDEX idx_data_sources_type ON data_sources(source_type);
CREATE INDEX idx_data_sources_vendor ON data_sources(vendor);
CREATE INDEX idx_data_sources_active ON data_sources(is_active);
CREATE INDEX idx_data_sources_primary ON data_sources(is_primary);
CREATE INDEX idx_data_sources_priority ON data_sources(priority);

-- Quality metric indexes for filtering
CREATE INDEX idx_data_sources_coverage ON data_sources(coverage_percent) WHERE is_active = true;
CREATE INDEX idx_data_sources_completeness ON data_sources(completeness_score) WHERE is_active = true;
CREATE INDEX idx_data_sources_freshness ON data_sources(freshness_days) WHERE is_active = true;

-- ═══════════════════════════════════════════════════════════════════
-- VIEWS
-- ═══════════════════════════════════════════════════════════════════

-- View: Active data sources summary
CREATE OR REPLACE VIEW v_active_data_sources AS
SELECT
    source_id,
    source_name,
    source_type,
    vendor,
    coverage_percent,
    completeness_score,
    freshness_days,
    match_rate,
    is_primary,
    priority,
    refresh_frequency,
    last_refresh_date,
    CASE
        WHEN last_refresh_date IS NULL THEN 'Never refreshed'
        WHEN EXTRACT(EPOCH FROM (NOW() - last_refresh_date)) / 86400 <= freshness_days THEN 'Fresh'
        WHEN EXTRACT(EPOCH FROM (NOW() - last_refresh_date)) / 86400 <= freshness_days * 2 THEN 'Aging'
        ELSE 'Stale'
    END AS freshness_status
FROM data_sources
WHERE is_active = true
ORDER BY priority, source_type;

-- View: Data source quality scorecard
CREATE OR REPLACE VIEW v_data_source_quality AS
SELECT
    source_id,
    source_name,
    source_type,
    -- Overall quality score (weighted average)
    ROUND(
        (coverage_percent * 0.3 +
         completeness_score * 100 * 0.3 +
         (100 - LEAST(freshness_days, 30)) * 0.2 +
         match_rate * 100 * 0.2)
    , 2) AS quality_score,
    coverage_percent,
    completeness_score,
    freshness_days,
    match_rate,
    CASE
        WHEN coverage_percent >= 90 AND completeness_score >= 0.95 THEN 'Excellent'
        WHEN coverage_percent >= 80 AND completeness_score >= 0.90 THEN 'Good'
        WHEN coverage_percent >= 70 AND completeness_score >= 0.85 THEN 'Fair'
        ELSE 'Poor'
    END AS quality_tier
FROM data_sources
WHERE is_active = true
ORDER BY quality_score DESC;

-- View: Primary data sources by type
CREATE OR REPLACE VIEW v_primary_data_sources AS
SELECT
    source_type,
    source_id,
    source_name,
    vendor,
    coverage_percent,
    completeness_score,
    freshness_days,
    patient_count,
    hcp_count
FROM data_sources
WHERE is_active = true AND is_primary = true
ORDER BY source_type, priority;

-- ═══════════════════════════════════════════════════════════════════
-- FUNCTIONS
-- ═══════════════════════════════════════════════════════════════════

-- Function: Update data source last refresh timestamp
CREATE OR REPLACE FUNCTION update_data_source_refresh(
    p_source_id TEXT
)
RETURNS VOID AS $$
BEGIN
    UPDATE data_sources
    SET
        last_refresh_date = NOW(),
        next_scheduled_refresh = CASE refresh_frequency
            WHEN 'realtime' THEN NOW()
            WHEN 'daily' THEN NOW() + INTERVAL '1 day'
            WHEN 'weekly' THEN NOW() + INTERVAL '1 week'
            WHEN 'biweekly' THEN NOW() + INTERVAL '2 weeks'
            WHEN 'monthly' THEN NOW() + INTERVAL '1 month'
            WHEN 'quarterly' THEN NOW() + INTERVAL '3 months'
            ELSE NOW() + INTERVAL '1 week'
        END,
        updated_at = NOW()
    WHERE source_id = p_source_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Calculate composite data quality score
CREATE OR REPLACE FUNCTION calculate_data_quality_score(
    p_source_id TEXT
)
RETURNS DECIMAL(5,2) AS $$
DECLARE
    v_score DECIMAL(5,2);
BEGIN
    SELECT
        ROUND(
            (coverage_percent * 0.3 +
             completeness_score * 100 * 0.3 +
             (100 - LEAST(freshness_days, 30)) * 0.2 +
             match_rate * 100 * 0.2)
        , 2)
    INTO v_score
    FROM data_sources
    WHERE source_id = p_source_id;

    RETURN COALESCE(v_score, 0);
END;
$$ LANGUAGE plpgsql;

-- ═══════════════════════════════════════════════════════════════════
-- TRIGGERS
-- ═══════════════════════════════════════════════════════════════════

-- Trigger: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_data_sources_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_data_sources_updated_at
BEFORE UPDATE ON data_sources
FOR EACH ROW
EXECUTE FUNCTION update_data_sources_timestamp();

-- ═══════════════════════════════════════════════════════════════════
-- GRANTS (Adjust based on your security model)
-- ═══════════════════════════════════════════════════════════════════

-- Grant read access to all authenticated users
-- GRANT SELECT ON data_sources TO authenticated;
-- GRANT SELECT ON v_active_data_sources TO authenticated;
-- GRANT SELECT ON v_data_source_quality TO authenticated;
-- GRANT SELECT ON v_primary_data_sources TO authenticated;

-- Grant write access to service role only
-- GRANT INSERT, UPDATE, DELETE ON data_sources TO service_role;

-- ═══════════════════════════════════════════════════════════════════
-- COMMENTS
-- ═══════════════════════════════════════════════════════════════════

COMMENT ON TABLE data_sources IS 'Reference table for external data sources with quality metrics and governance metadata';
COMMENT ON COLUMN data_sources.source_id IS 'Unique identifier for data source (e.g., iqvia_apld)';
COMMENT ON COLUMN data_sources.coverage_percent IS 'Percentage of target population covered by this source (0-100)';
COMMENT ON COLUMN data_sources.completeness_score IS 'Proportion of required fields populated (0-1)';
COMMENT ON COLUMN data_sources.freshness_days IS 'Number of days since last data refresh';
COMMENT ON COLUMN data_sources.match_rate IS 'Patient/HCP match rate across sources (0-1)';
COMMENT ON COLUMN data_sources.is_primary IS 'Indicates if this is the primary source for its type';
COMMENT ON COLUMN data_sources.priority IS 'Source priority for conflict resolution (lower = higher priority)';

-- ═══════════════════════════════════════════════════════════════════
-- VALIDATION QUERIES
-- ═══════════════════════════════════════════════════════════════════

-- Test query 1: List all active sources with quality scores
-- SELECT * FROM v_data_source_quality;

-- Test query 2: Find stale data sources
-- SELECT source_id, source_name, last_refresh_date, freshness_days
-- FROM data_sources
-- WHERE is_active = true
--   AND EXTRACT(EPOCH FROM (NOW() - last_refresh_date)) / 86400 > freshness_days * 2;

-- Test query 3: Primary sources by type
-- SELECT * FROM v_primary_data_sources;

-- Test query 4: Calculate quality score for a specific source
-- SELECT source_name, calculate_data_quality_score('iqvia_apld') AS quality_score
-- FROM data_sources WHERE source_id = 'iqvia_apld';

-- ═══════════════════════════════════════════════════════════════════
-- MIGRATION COMPLETE
-- ═══════════════════════════════════════════════════════════════════

-- Table count: 28 → 29 (added data_sources)
-- Views: 3 (v_active_data_sources, v_data_source_quality, v_primary_data_sources)
-- Functions: 2 (update_data_source_refresh, calculate_data_quality_score)
-- Triggers: 1 (trg_data_sources_updated_at)
