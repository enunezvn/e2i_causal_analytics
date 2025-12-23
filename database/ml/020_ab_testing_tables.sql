-- ============================================
-- Migration: 020_ab_testing_tables.sql
-- Purpose: A/B Testing Execution Infrastructure
-- Version: E2I V4.2.0
-- Dependencies: mlops_tables.sql (ml_experiments)
-- Phase: 15 - A/B Testing Infrastructure
-- ============================================

-- ============================================
-- SECTION 1: ENUM TYPES
-- ============================================

-- Unit types for experiment assignment
DO $$ BEGIN
    CREATE TYPE ab_unit_type AS ENUM (
        'hcp',           -- Healthcare Professional
        'patient',       -- Patient
        'territory',     -- Geographic territory
        'account'        -- Account/organization
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Randomization methods
DO $$ BEGIN
    CREATE TYPE randomization_method AS ENUM (
        'simple',        -- Simple random assignment
        'stratified',    -- Stratified by covariates
        'block',         -- Block randomization
        'cluster',       -- Cluster randomization
        'adaptive'       -- Response-adaptive randomization
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Enrollment status
DO $$ BEGIN
    CREATE TYPE enrollment_status AS ENUM (
        'active',        -- Currently enrolled
        'withdrawn',     -- Voluntarily withdrawn
        'excluded',      -- Excluded (protocol violation)
        'completed',     -- Completed experiment duration
        'lost_to_followup'  -- Lost to follow-up
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Interim analysis decision
DO $$ BEGIN
    CREATE TYPE interim_decision AS ENUM (
        'continue',        -- Continue experiment
        'stop_efficacy',   -- Stop early for efficacy
        'stop_futility',   -- Stop early for futility
        'stop_safety',     -- Stop for safety concerns
        'modify_sample'    -- Modify sample size
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- ============================================
-- SECTION 2: CORE TABLES
-- ============================================

-- Table: ab_experiment_assignments
-- Stores treatment/control assignments for units
CREATE TABLE IF NOT EXISTS ab_experiment_assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Experiment reference
    experiment_id UUID NOT NULL REFERENCES ml_experiments(id) ON DELETE RESTRICT,

    -- Unit identification
    unit_id VARCHAR(255) NOT NULL,
    unit_type ab_unit_type NOT NULL,

    -- Assignment details
    variant VARCHAR(50) NOT NULL,  -- 'control', 'treatment_a', 'treatment_b', etc.
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Randomization metadata
    randomization_method randomization_method NOT NULL DEFAULT 'simple',
    stratification_key JSONB DEFAULT '{}',
    /*
    Example stratification_key:
    {
        "region": "northeast",
        "specialty": "oncology",
        "decile": 1
    }
    */
    block_id VARCHAR(100),  -- For block randomization

    -- Hash for reproducibility
    assignment_hash VARCHAR(64),  -- SHA-256 of (experiment_id, unit_id, salt)

    -- Audit
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_unit_per_experiment UNIQUE(experiment_id, unit_id),
    CONSTRAINT valid_variant CHECK (variant ~ '^[a-z0-9_]+$')  -- lowercase alphanumeric with underscores
);

-- Indexes for assignments
CREATE INDEX IF NOT EXISTS idx_ab_assignments_experiment ON ab_experiment_assignments(experiment_id);
CREATE INDEX IF NOT EXISTS idx_ab_assignments_unit ON ab_experiment_assignments(unit_id);
CREATE INDEX IF NOT EXISTS idx_ab_assignments_variant ON ab_experiment_assignments(experiment_id, variant);
CREATE INDEX IF NOT EXISTS idx_ab_assignments_unit_type ON ab_experiment_assignments(unit_type);
CREATE INDEX IF NOT EXISTS idx_ab_assignments_block ON ab_experiment_assignments(experiment_id, block_id) WHERE block_id IS NOT NULL;

-- Table: ab_experiment_enrollments
-- Tracks enrollment lifecycle for assigned units
CREATE TABLE IF NOT EXISTS ab_experiment_enrollments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Assignment reference
    assignment_id UUID NOT NULL REFERENCES ab_experiment_assignments(id) ON DELETE CASCADE,

    -- Enrollment lifecycle
    enrolled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    enrollment_status enrollment_status NOT NULL DEFAULT 'active',

    -- Eligibility tracking
    eligibility_criteria_met JSONB DEFAULT '{}',
    /*
    Example:
    {
        "min_rx_history": true,
        "active_patient_panel": true,
        "not_in_concurrent_study": true,
        "consent_obtained": true
    }
    */
    eligibility_check_timestamp TIMESTAMP WITH TIME ZONE,

    -- Consent tracking
    consent_timestamp TIMESTAMP WITH TIME ZONE,
    consent_method VARCHAR(50),  -- 'email', 'phone', 'in_person', 'implied'
    consent_version VARCHAR(20),

    -- Withdrawal tracking
    withdrawal_timestamp TIMESTAMP WITH TIME ZONE,
    withdrawal_reason TEXT,
    withdrawal_initiated_by VARCHAR(50),  -- 'subject', 'investigator', 'sponsor', 'system'

    -- Protocol adherence
    protocol_deviations JSONB DEFAULT '[]',
    /*
    Example:
    [
        {"date": "2025-01-15", "type": "missed_touchpoint", "severity": "minor"},
        {"date": "2025-01-22", "type": "late_data_collection", "severity": "minor"}
    ]
    */

    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_withdrawal CHECK (
        (enrollment_status != 'withdrawn' AND withdrawal_timestamp IS NULL) OR
        (enrollment_status = 'withdrawn' AND withdrawal_timestamp IS NOT NULL)
    )
);

-- Indexes for enrollments
CREATE INDEX IF NOT EXISTS idx_ab_enrollments_assignment ON ab_experiment_enrollments(assignment_id);
CREATE INDEX IF NOT EXISTS idx_ab_enrollments_status ON ab_experiment_enrollments(enrollment_status);
CREATE INDEX IF NOT EXISTS idx_ab_enrollments_active ON ab_experiment_enrollments(assignment_id)
    WHERE enrollment_status = 'active';

-- Table: ab_interim_analyses
-- Tracks interim analysis results with stopping rules
CREATE TABLE IF NOT EXISTS ab_interim_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Experiment reference
    experiment_id UUID NOT NULL REFERENCES ml_experiments(id) ON DELETE RESTRICT,

    -- Analysis identification
    analysis_number INTEGER NOT NULL,  -- 1, 2, 3, ...
    performed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    analysis_type VARCHAR(50) NOT NULL DEFAULT 'scheduled',  -- 'scheduled', 'ad_hoc', 'triggered'

    -- Information fraction (proportion of total data collected)
    information_fraction FLOAT NOT NULL CHECK (information_fraction > 0 AND information_fraction <= 1),
    sample_size_at_analysis INTEGER NOT NULL,
    target_sample_size INTEGER,

    -- Alpha spending (O'Brien-Fleming or other)
    spending_function VARCHAR(50) NOT NULL DEFAULT 'obrien_fleming',
    alpha_spent FLOAT NOT NULL CHECK (alpha_spent >= 0 AND alpha_spent <= 1),
    cumulative_alpha_spent FLOAT NOT NULL CHECK (cumulative_alpha_spent >= 0 AND cumulative_alpha_spent <= 1),
    adjusted_alpha FLOAT NOT NULL CHECK (adjusted_alpha >= 0 AND adjusted_alpha <= 1),

    -- Test statistics
    test_statistic FLOAT,
    standard_error FLOAT,
    p_value FLOAT CHECK (p_value >= 0 AND p_value <= 1),
    effect_estimate FLOAT,
    effect_ci_lower FLOAT,
    effect_ci_upper FLOAT,

    -- Conditional power (futility assessment)
    conditional_power FLOAT CHECK (conditional_power >= 0 AND conditional_power <= 1),
    predictive_probability FLOAT CHECK (predictive_probability >= 0 AND predictive_probability <= 1),

    -- Decision
    decision interim_decision NOT NULL DEFAULT 'continue',
    decision_rationale TEXT,

    -- Metrics snapshot (all metrics at this point)
    metrics_snapshot JSONB NOT NULL DEFAULT '{}',
    /*
    Example:
    {
        "primary_metric": {
            "name": "conversion_rate",
            "control_mean": 0.12,
            "treatment_mean": 0.15,
            "relative_lift": 0.25
        },
        "secondary_metrics": [
            {"name": "engagement_score", "control_mean": 7.2, "treatment_mean": 7.8}
        ],
        "sample_sizes": {"control": 500, "treatment": 498}
    }
    */

    -- Audit
    performed_by VARCHAR(100),
    approved_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_analysis_number UNIQUE(experiment_id, analysis_number),
    CONSTRAINT valid_effect_ci CHECK (effect_ci_lower IS NULL OR effect_ci_upper IS NULL OR effect_ci_lower <= effect_ci_upper)
);

-- Indexes for interim analyses
CREATE INDEX IF NOT EXISTS idx_ab_interim_experiment ON ab_interim_analyses(experiment_id);
CREATE INDEX IF NOT EXISTS idx_ab_interim_decision ON ab_interim_analyses(experiment_id, decision);
CREATE INDEX IF NOT EXISTS idx_ab_interim_performed ON ab_interim_analyses(performed_at);

-- ============================================
-- SECTION 3: SUPPORTING FUNCTIONS
-- ============================================

-- Function to get current enrollment count by variant
CREATE OR REPLACE FUNCTION get_enrollment_counts(p_experiment_id UUID)
RETURNS TABLE(variant VARCHAR(50), enrolled_count BIGINT, active_count BIGINT) AS $$
BEGIN
    RETURN QUERY
    SELECT
        a.variant,
        COUNT(e.id) as enrolled_count,
        COUNT(e.id) FILTER (WHERE e.enrollment_status = 'active') as active_count
    FROM ab_experiment_assignments a
    LEFT JOIN ab_experiment_enrollments e ON a.id = e.assignment_id
    WHERE a.experiment_id = p_experiment_id
    GROUP BY a.variant
    ORDER BY a.variant;
END;
$$ LANGUAGE plpgsql;

-- Function to check for sample ratio mismatch
CREATE OR REPLACE FUNCTION check_sample_ratio(
    p_experiment_id UUID,
    p_expected_ratio JSONB DEFAULT '{"control": 0.5, "treatment": 0.5}'::JSONB
)
RETURNS TABLE(
    variant VARCHAR(50),
    expected_proportion FLOAT,
    actual_proportion FLOAT,
    actual_count BIGINT,
    deviation FLOAT
) AS $$
DECLARE
    v_total_count BIGINT;
BEGIN
    -- Get total enrolled count
    SELECT COUNT(*) INTO v_total_count
    FROM ab_experiment_assignments a
    JOIN ab_experiment_enrollments e ON a.id = e.assignment_id
    WHERE a.experiment_id = p_experiment_id
    AND e.enrollment_status = 'active';

    IF v_total_count = 0 THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT
        a.variant,
        COALESCE((p_expected_ratio->>(a.variant))::FLOAT, 0.5) as expected_proportion,
        COUNT(e.id)::FLOAT / v_total_count as actual_proportion,
        COUNT(e.id) as actual_count,
        ABS(COUNT(e.id)::FLOAT / v_total_count - COALESCE((p_expected_ratio->>(a.variant))::FLOAT, 0.5)) as deviation
    FROM ab_experiment_assignments a
    JOIN ab_experiment_enrollments e ON a.id = e.assignment_id
    WHERE a.experiment_id = p_experiment_id
    AND e.enrollment_status = 'active'
    GROUP BY a.variant
    ORDER BY a.variant;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- SECTION 4: TRIGGERS
-- ============================================

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_enrollment_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_enrollment_updated
    BEFORE UPDATE ON ab_experiment_enrollments
    FOR EACH ROW
    EXECUTE FUNCTION update_enrollment_timestamp();

-- ============================================
-- SECTION 5: COMMENTS
-- ============================================

COMMENT ON TABLE ab_experiment_assignments IS 'Treatment/control assignments for A/B test units';
COMMENT ON TABLE ab_experiment_enrollments IS 'Enrollment lifecycle tracking for assigned units';
COMMENT ON TABLE ab_interim_analyses IS 'Interim analysis results with stopping rules (O''Brien-Fleming)';

COMMENT ON COLUMN ab_experiment_assignments.stratification_key IS 'JSONB containing stratification variables used for balanced randomization';
COMMENT ON COLUMN ab_experiment_assignments.assignment_hash IS 'Deterministic hash for reproducible assignment verification';
COMMENT ON COLUMN ab_interim_analyses.information_fraction IS 'Proportion of planned sample size collected at analysis (0-1)';
COMMENT ON COLUMN ab_interim_analyses.conditional_power IS 'Probability of achieving significance given current data (futility assessment)';
