-- ============================================
-- E2I Cohort Constructor - Database Migration
-- Migration: 028_cohort_constructor_tables.sql
-- ============================================
--
-- Purpose: Store cohort definitions, eligibility logs, and metadata
-- Integration: Tier 0 ML Foundation (CohortConstructor Agent)
-- Usage: Track cohort construction across brands/indications
--
-- Tables Created:
--   1. ml_cohort_definitions - Versioned cohort configurations
--   2. ml_cohort_executions - Execution tracking
--   3. ml_cohort_eligibility_log - Step-by-step exclusions
--   4. ml_patient_cohort_assignments - Patient eligibility records
--   5. ml_cohort_comparisons - Cohort comparison results
--

-- ============================================
-- 1. COHORT DEFINITIONS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS ml_cohort_definitions (
    -- Primary Key
    cohort_id TEXT PRIMARY KEY,

    -- Brand and Indication
    brand TEXT NOT NULL,
    indication TEXT NOT NULL,
    cohort_name TEXT NOT NULL,

    -- Version Control
    version TEXT DEFAULT '1.0.0',
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,

    -- Criteria (stored as JSON)
    inclusion_criteria JSONB NOT NULL,
    exclusion_criteria JSONB NOT NULL,

    -- Temporal Requirements
    lookback_days INTEGER DEFAULT 180,
    followup_days INTEGER DEFAULT 90,
    index_date_field TEXT DEFAULT 'diagnosis_date',

    -- Required Fields
    required_fields JSONB,

    -- Population Statistics
    initial_population INTEGER,
    eligible_population INTEGER,
    excluded_population INTEGER,
    exclusion_rate DECIMAL(5,4),

    -- Status
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'draft', 'archived', 'locked')),

    -- Clinical Rationale
    clinical_rationale TEXT,
    regulatory_justification TEXT,

    -- Metadata
    config_json JSONB,
    config_hash TEXT,  -- SHA256 hash for version tracking

    -- Constraints
    CONSTRAINT unique_brand_indication_version UNIQUE (brand, indication, version)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_cohort_brand ON ml_cohort_definitions(brand);
CREATE INDEX IF NOT EXISTS idx_cohort_indication ON ml_cohort_definitions(indication);
CREATE INDEX IF NOT EXISTS idx_cohort_status ON ml_cohort_definitions(status);
CREATE INDEX IF NOT EXISTS idx_cohort_created_date ON ml_cohort_definitions(created_date);

-- ============================================
-- 2. COHORT EXECUTIONS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS ml_cohort_executions (
    -- Primary Key
    execution_id TEXT PRIMARY KEY,

    -- Foreign Key
    cohort_id TEXT NOT NULL REFERENCES ml_cohort_definitions(cohort_id) ON DELETE CASCADE,

    -- Execution Details
    execution_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    executed_by TEXT,
    environment TEXT DEFAULT 'production' CHECK (environment IN ('development', 'staging', 'production')),

    -- Input
    input_dataset TEXT,
    input_row_count INTEGER,

    -- Output
    eligible_row_count INTEGER,
    excluded_row_count INTEGER,
    output_dataset TEXT,

    -- Performance (SLA target: <120 seconds for 100K patients)
    execution_time_ms INTEGER,  -- Changed to ms for precision
    execution_time_seconds DECIMAL(10,3) GENERATED ALWAYS AS (execution_time_ms / 1000.0) STORED,

    -- Node Latencies
    validate_config_ms INTEGER,
    apply_criteria_ms INTEGER,
    validate_temporal_ms INTEGER,
    generate_metadata_ms INTEGER,

    -- Status
    status TEXT DEFAULT 'success' CHECK (status IN ('success', 'failed', 'partial')),
    error_message TEXT,
    error_code TEXT,  -- CC_001 through CC_007

    -- Metadata
    execution_metadata JSONB
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_execution_cohort_id ON ml_cohort_executions(cohort_id);
CREATE INDEX IF NOT EXISTS idx_execution_timestamp ON ml_cohort_executions(execution_timestamp);
CREATE INDEX IF NOT EXISTS idx_execution_status ON ml_cohort_executions(status);

-- ============================================
-- 3. COHORT ELIGIBILITY LOG TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS ml_cohort_eligibility_log (
    -- Primary Key
    log_id SERIAL PRIMARY KEY,

    -- Foreign Keys
    cohort_id TEXT NOT NULL REFERENCES ml_cohort_definitions(cohort_id) ON DELETE CASCADE,
    execution_id TEXT NOT NULL REFERENCES ml_cohort_executions(execution_id) ON DELETE CASCADE,

    -- Criterion Order (for audit trail)
    criterion_order INTEGER NOT NULL,

    -- Criterion Details
    criterion_name TEXT NOT NULL,
    criterion_type TEXT NOT NULL CHECK (criterion_type IN ('inclusion', 'exclusion', 'temporal')),
    operator TEXT NOT NULL,
    criterion_value JSONB,

    -- Impact
    removed_count INTEGER NOT NULL,
    remaining_count INTEGER NOT NULL,

    -- Context
    description TEXT,
    clinical_rationale TEXT,

    -- Timestamp
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_eligibility_cohort_id ON ml_cohort_eligibility_log(cohort_id);
CREATE INDEX IF NOT EXISTS idx_eligibility_execution_id ON ml_cohort_eligibility_log(execution_id);
CREATE INDEX IF NOT EXISTS idx_eligibility_criterion_type ON ml_cohort_eligibility_log(criterion_type);
CREATE INDEX IF NOT EXISTS idx_eligibility_order ON ml_cohort_eligibility_log(execution_id, criterion_order);

-- ============================================
-- 4. PATIENT COHORT ASSIGNMENTS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS ml_patient_cohort_assignments (
    -- Composite Primary Key
    patient_journey_id TEXT NOT NULL,
    cohort_id TEXT NOT NULL,
    execution_id TEXT NOT NULL,

    -- Assignment Details
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_eligible BOOLEAN NOT NULL,

    -- Failure Reasons (if not eligible)
    failed_criteria JSONB,

    -- Temporal Validation
    lookback_complete BOOLEAN,
    followup_complete BOOLEAN,
    index_date DATE,
    journey_start_date DATE,
    journey_end_date DATE,

    -- Foreign Keys
    CONSTRAINT fk_cohort_assign FOREIGN KEY (cohort_id) REFERENCES ml_cohort_definitions(cohort_id) ON DELETE CASCADE,
    CONSTRAINT fk_execution_assign FOREIGN KEY (execution_id) REFERENCES ml_cohort_executions(execution_id) ON DELETE CASCADE,

    -- Primary Key
    PRIMARY KEY (patient_journey_id, cohort_id, execution_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_patient_cohort ON ml_patient_cohort_assignments(patient_journey_id, cohort_id);
CREATE INDEX IF NOT EXISTS idx_patient_eligible ON ml_patient_cohort_assignments(is_eligible);
CREATE INDEX IF NOT EXISTS idx_patient_execution ON ml_patient_cohort_assignments(execution_id);

-- ============================================
-- 5. COHORT COMPARISONS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS ml_cohort_comparisons (
    -- Primary Key
    comparison_id TEXT PRIMARY KEY,

    -- Comparison Details
    comparison_name TEXT NOT NULL,
    comparison_description TEXT,
    compared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    compared_by TEXT,

    -- Cohorts Being Compared
    cohort_ids JSONB NOT NULL,  -- Array of cohort_id values

    -- Input Dataset
    input_dataset TEXT,
    input_row_count INTEGER,

    -- Results (stored as JSON for flexibility)
    comparison_results JSONB,

    -- Visualization
    visualization_config JSONB
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_comparison_timestamp ON ml_cohort_comparisons(compared_at);

-- ============================================
-- 6. VIEWS FOR COMMON QUERIES
-- ============================================

-- View: Latest cohort definition per brand/indication
CREATE OR REPLACE VIEW vw_latest_cohort_definitions AS
SELECT DISTINCT ON (brand, indication)
    cohort_id,
    brand,
    indication,
    cohort_name,
    version,
    created_date,
    status,
    eligible_population,
    exclusion_rate
FROM ml_cohort_definitions
WHERE status IN ('active', 'locked')
ORDER BY brand, indication, created_date DESC;

-- View: Cohort execution summary
CREATE OR REPLACE VIEW vw_cohort_execution_summary AS
SELECT
    ce.cohort_id,
    cd.brand,
    cd.indication,
    cd.cohort_name,
    COUNT(*) as total_executions,
    SUM(CASE WHEN ce.status = 'success' THEN 1 ELSE 0 END) as successful_executions,
    AVG(ce.execution_time_seconds) as avg_execution_time,
    MAX(ce.execution_timestamp) as last_execution,
    AVG(CAST(ce.eligible_row_count AS FLOAT) / NULLIF(ce.input_row_count, 0)) as avg_eligibility_rate
FROM ml_cohort_executions ce
JOIN ml_cohort_definitions cd ON ce.cohort_id = cd.cohort_id
GROUP BY ce.cohort_id, cd.brand, cd.indication, cd.cohort_name;

-- View: Eligibility funnel
CREATE OR REPLACE VIEW vw_eligibility_funnel AS
SELECT
    el.cohort_id,
    el.execution_id,
    el.criterion_order,
    el.criterion_name,
    el.criterion_type,
    el.removed_count,
    el.remaining_count,
    CAST(el.removed_count AS FLOAT) / NULLIF(
        LAG(el.remaining_count) OVER (
            PARTITION BY el.cohort_id, el.execution_id
            ORDER BY el.criterion_order
        ) + el.removed_count,
        0
    ) as exclusion_rate_at_step
FROM ml_cohort_eligibility_log el
ORDER BY el.cohort_id, el.execution_id, el.criterion_order;

-- ============================================
-- 7. FUNCTIONS FOR COMMON OPERATIONS
-- ============================================

-- Function: Get latest cohort configuration
CREATE OR REPLACE FUNCTION get_latest_cohort_config(
    p_brand TEXT,
    p_indication TEXT
)
RETURNS JSONB AS $$
DECLARE
    v_config JSONB;
BEGIN
    SELECT config_json
    INTO v_config
    FROM ml_cohort_definitions
    WHERE brand = p_brand
      AND indication = p_indication
      AND status IN ('active', 'locked')
    ORDER BY created_date DESC
    LIMIT 1;

    RETURN v_config;
END;
$$ LANGUAGE plpgsql;

-- Function: Calculate cohort overlap
CREATE OR REPLACE FUNCTION calculate_cohort_overlap(
    p_cohort_id_1 TEXT,
    p_cohort_id_2 TEXT,
    p_execution_id_1 TEXT,
    p_execution_id_2 TEXT
)
RETURNS TABLE(
    overlap_count INTEGER,
    cohort_1_only INTEGER,
    cohort_2_only INTEGER,
    total_unique INTEGER,
    overlap_percentage DECIMAL(5,2)
) AS $$
BEGIN
    RETURN QUERY
    WITH cohort_1 AS (
        SELECT patient_journey_id
        FROM ml_patient_cohort_assignments
        WHERE cohort_id = p_cohort_id_1
          AND execution_id = p_execution_id_1
          AND is_eligible = TRUE
    ),
    cohort_2 AS (
        SELECT patient_journey_id
        FROM ml_patient_cohort_assignments
        WHERE cohort_id = p_cohort_id_2
          AND execution_id = p_execution_id_2
          AND is_eligible = TRUE
    )
    SELECT
        COUNT(DISTINCT c1.patient_journey_id)::INTEGER as overlap_count,
        ((SELECT COUNT(*) FROM cohort_1) - COUNT(DISTINCT c1.patient_journey_id))::INTEGER as cohort_1_only,
        ((SELECT COUNT(*) FROM cohort_2) - COUNT(DISTINCT c1.patient_journey_id))::INTEGER as cohort_2_only,
        ((SELECT COUNT(*) FROM cohort_1) + (SELECT COUNT(*) FROM cohort_2) - COUNT(DISTINCT c1.patient_journey_id))::INTEGER as total_unique,
        (COUNT(DISTINCT c1.patient_journey_id)::DECIMAL / NULLIF((SELECT COUNT(*) FROM cohort_1), 0) * 100)::DECIMAL(5,2) as overlap_percentage
    FROM cohort_1 c1
    INNER JOIN cohort_2 c2 ON c1.patient_journey_id = c2.patient_journey_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 8. TRIGGERS FOR AUDIT TRAIL
-- ============================================

-- Update timestamp on cohort definition changes
CREATE OR REPLACE FUNCTION update_cohort_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_date = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tr_update_cohort_timestamp ON ml_cohort_definitions;
CREATE TRIGGER tr_update_cohort_timestamp
BEFORE UPDATE ON ml_cohort_definitions
FOR EACH ROW
EXECUTE FUNCTION update_cohort_timestamp();

-- ============================================
-- 9. COMMENTS FOR DOCUMENTATION
-- ============================================

COMMENT ON TABLE ml_cohort_definitions IS 'Stores cohort eligibility criteria definitions for each brand/indication combination';
COMMENT ON TABLE ml_cohort_eligibility_log IS 'Audit log of which criteria were applied and their impact on population size';
COMMENT ON TABLE ml_cohort_executions IS 'Tracks each execution of cohort construction with performance metrics';
COMMENT ON TABLE ml_patient_cohort_assignments IS 'Records which patients were deemed eligible for each cohort execution';
COMMENT ON TABLE ml_cohort_comparisons IS 'Stores results of comparing multiple cohort definitions';

COMMENT ON COLUMN ml_cohort_definitions.inclusion_criteria IS 'JSON array of inclusion criteria with field, operator, value, description';
COMMENT ON COLUMN ml_cohort_definitions.exclusion_criteria IS 'JSON array of exclusion criteria with field, operator, value, description';
COMMENT ON COLUMN ml_cohort_definitions.exclusion_rate IS 'Percentage of initial population excluded (0-1 scale)';
COMMENT ON COLUMN ml_cohort_definitions.config_hash IS 'SHA256 hash of configuration for version tracking';

COMMENT ON COLUMN ml_cohort_eligibility_log.criterion_type IS 'Type of criterion: inclusion, exclusion, or temporal';
COMMENT ON COLUMN ml_cohort_eligibility_log.removed_count IS 'Number of patients excluded by this criterion';
COMMENT ON COLUMN ml_cohort_eligibility_log.remaining_count IS 'Number of patients remaining after this criterion';
COMMENT ON COLUMN ml_cohort_eligibility_log.criterion_order IS 'Order in which criterion was applied (for audit trail)';

COMMENT ON COLUMN ml_cohort_executions.execution_time_ms IS 'Execution time in milliseconds (SLA: <120000ms for 100K patients)';
COMMENT ON COLUMN ml_cohort_executions.error_code IS 'Error codes: CC_001 (invalid_config) through CC_007 (timeout)';

-- ============================================
-- VERIFICATION QUERIES (Run to validate)
-- ============================================
--
-- Check table creation:
-- SELECT table_name
-- FROM information_schema.tables
-- WHERE table_schema = 'public'
--   AND table_name LIKE 'ml_cohort%';
--
-- Check indexes:
-- SELECT indexname, indexdef
-- FROM pg_indexes
-- WHERE tablename LIKE 'ml_cohort%';
--
-- Check views:
-- SELECT viewname
-- FROM pg_views
-- WHERE schemaname = 'public'
--   AND viewname LIKE 'vw_%cohort%';
