-- ============================================
-- Migration: 021_ab_results_tables.sql
-- Purpose: A/B Testing Results and Analysis Tables
-- Version: E2I V4.2.0
-- Dependencies: 020_ab_testing_tables.sql, 012_digital_twin_tables.sql
-- Phase: 15 - A/B Testing Infrastructure
-- ============================================

-- ============================================
-- SECTION 1: ENUM TYPES
-- ============================================

-- Analysis type for results
DO $$ BEGIN
    CREATE TYPE ab_analysis_type AS ENUM (
        'interim',       -- Interim analysis during experiment
        'final',         -- Final analysis at experiment end
        'post_hoc'       -- Post-hoc exploratory analysis
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Analysis method
DO $$ BEGIN
    CREATE TYPE ab_analysis_method AS ENUM (
        'itt',               -- Intent-to-treat
        'per_protocol',      -- Per-protocol analysis
        'as_treated',        -- As-treated analysis
        'cace'               -- Complier Average Causal Effect
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- SRM severity
DO $$ BEGIN
    CREATE TYPE srm_severity AS ENUM (
        'none',          -- No SRM detected
        'minor',         -- p > 0.01
        'moderate',      -- 0.001 < p <= 0.01
        'severe'         -- p <= 0.001
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- ============================================
-- SECTION 2: CORE TABLES
-- ============================================

-- Table: ab_experiment_results
-- Stores computed experiment results
CREATE TABLE IF NOT EXISTS ab_experiment_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Experiment reference
    experiment_id UUID NOT NULL REFERENCES ml_experiments(id) ON DELETE RESTRICT,

    -- Analysis metadata
    analysis_type ab_analysis_type NOT NULL,
    analysis_method ab_analysis_method NOT NULL DEFAULT 'itt',
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Primary metric results
    primary_metric VARCHAR(100) NOT NULL,
    primary_metric_type VARCHAR(50) DEFAULT 'continuous',  -- 'continuous', 'binary', 'count', 'time_to_event'

    -- Control arm statistics
    control_mean FLOAT,
    control_std FLOAT,
    control_median FLOAT,
    control_n INTEGER NOT NULL,

    -- Treatment arm statistics
    treatment_mean FLOAT,
    treatment_std FLOAT,
    treatment_median FLOAT,
    treatment_n INTEGER NOT NULL,

    -- Effect estimates
    effect_estimate FLOAT NOT NULL,
    effect_type VARCHAR(50) DEFAULT 'absolute',  -- 'absolute', 'relative', 'odds_ratio', 'risk_ratio', 'hazard_ratio'
    effect_ci_lower FLOAT,
    effect_ci_upper FLOAT,
    confidence_level FLOAT DEFAULT 0.95,

    -- Statistical inference
    test_statistic FLOAT,
    test_type VARCHAR(50),  -- 't_test', 'z_test', 'chi_squared', 'mann_whitney', 'log_rank'
    p_value FLOAT CHECK (p_value >= 0 AND p_value <= 1),
    is_significant BOOLEAN,
    adjusted_p_value FLOAT,  -- Bonferroni or other adjustment

    -- Statistical power
    observed_power FLOAT CHECK (observed_power >= 0 AND observed_power <= 1),
    minimum_detectable_effect FLOAT,

    -- Secondary metrics (all at once)
    secondary_metrics JSONB DEFAULT '[]',
    /*
    Example:
    [
        {
            "name": "engagement_score",
            "control_mean": 7.2,
            "treatment_mean": 7.8,
            "effect_estimate": 0.6,
            "p_value": 0.023,
            "is_significant": true
        },
        {
            "name": "retention_rate",
            "control_mean": 0.82,
            "treatment_mean": 0.85,
            "effect_estimate": 0.03,
            "p_value": 0.15,
            "is_significant": false
        }
    ]
    */

    -- Heterogeneous treatment effects by segment
    segment_results JSONB DEFAULT '{}',
    /*
    Example:
    {
        "by_specialty": {
            "oncology": {"effect": 0.15, "ci": [0.08, 0.22], "n": 234},
            "hematology": {"effect": 0.12, "ci": [0.04, 0.20], "n": 156}
        },
        "by_decile": {
            "1": {"effect": 0.18, "ci": [0.10, 0.26], "n": 120},
            "2": {"effect": 0.14, "ci": [0.06, 0.22], "n": 118}
        }
    }
    */

    -- Covariate-adjusted results
    covariates_adjusted JSONB DEFAULT '[]',  -- List of covariates used
    adjusted_effect_estimate FLOAT,
    adjusted_ci_lower FLOAT,
    adjusted_ci_upper FLOAT,

    -- Data quality indicators
    missing_rate_control FLOAT,
    missing_rate_treatment FLOAT,
    outliers_removed INTEGER DEFAULT 0,

    -- Audit
    computed_by VARCHAR(100),
    approved_by VARCHAR(100),
    approval_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_ci CHECK (effect_ci_lower IS NULL OR effect_ci_upper IS NULL OR effect_ci_lower <= effect_ci_upper),
    CONSTRAINT valid_sample_sizes CHECK (control_n > 0 AND treatment_n > 0)
);

-- Indexes for results
CREATE INDEX IF NOT EXISTS idx_ab_results_experiment ON ab_experiment_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_ab_results_type ON ab_experiment_results(experiment_id, analysis_type);
CREATE INDEX IF NOT EXISTS idx_ab_results_computed ON ab_experiment_results(computed_at);
CREATE INDEX IF NOT EXISTS idx_ab_results_significant ON ab_experiment_results(experiment_id, is_significant);

-- Table: ab_srm_checks
-- Sample Ratio Mismatch detection history
CREATE TABLE IF NOT EXISTS ab_srm_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Experiment reference
    experiment_id UUID NOT NULL REFERENCES ml_experiments(id) ON DELETE RESTRICT,

    -- Check metadata
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    check_type VARCHAR(50) DEFAULT 'scheduled',  -- 'scheduled', 'triggered', 'manual'

    -- Expected allocation
    expected_ratio JSONB NOT NULL,
    /*
    Example:
    {
        "control": 0.5,
        "treatment": 0.5
    }
    */

    -- Actual counts
    actual_counts JSONB NOT NULL,
    /*
    Example:
    {
        "control": 4823,
        "treatment": 5177
    }
    */

    -- Statistical test
    chi_squared_statistic FLOAT,
    degrees_of_freedom INTEGER DEFAULT 1,
    p_value FLOAT CHECK (p_value >= 0 AND p_value <= 1),

    -- Detection
    is_srm_detected BOOLEAN NOT NULL DEFAULT false,
    severity srm_severity NOT NULL DEFAULT 'none',
    deviation_percent FLOAT,  -- Absolute deviation from expected

    -- Investigation
    investigation_notes TEXT,
    root_cause VARCHAR(100),  -- 'bot_traffic', 'tracking_bug', 'assignment_bug', 'unknown', etc.
    resolution_action TEXT,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(100),

    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_deviation CHECK (deviation_percent IS NULL OR deviation_percent >= 0)
);

-- Indexes for SRM checks
CREATE INDEX IF NOT EXISTS idx_ab_srm_experiment ON ab_srm_checks(experiment_id);
CREATE INDEX IF NOT EXISTS idx_ab_srm_detected ON ab_srm_checks(experiment_id, is_srm_detected);
CREATE INDEX IF NOT EXISTS idx_ab_srm_severity ON ab_srm_checks(severity) WHERE severity != 'none';
CREATE INDEX IF NOT EXISTS idx_ab_srm_checked ON ab_srm_checks(checked_at);

-- Table: ab_fidelity_comparisons
-- Compare Digital Twin predictions with actual results
CREATE TABLE IF NOT EXISTS ab_fidelity_comparisons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Experiment reference
    experiment_id UUID NOT NULL REFERENCES ml_experiments(id) ON DELETE RESTRICT,

    -- Digital Twin simulation reference
    twin_simulation_id UUID NOT NULL REFERENCES twin_simulations(simulation_id) ON DELETE RESTRICT,

    -- Results reference (for final comparisons)
    results_id UUID REFERENCES ab_experiment_results(id) ON DELETE SET NULL,

    -- Comparison metadata
    comparison_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    comparison_type VARCHAR(50) DEFAULT 'final',  -- 'interim', 'final'

    -- Predicted values (from Digital Twin)
    predicted_effect FLOAT NOT NULL,
    predicted_ci_lower FLOAT,
    predicted_ci_upper FLOAT,
    predicted_sample_size INTEGER,

    -- Actual values (from experiment)
    actual_effect FLOAT,
    actual_ci_lower FLOAT,
    actual_ci_upper FLOAT,
    actual_sample_size INTEGER,

    -- Fidelity metrics
    prediction_error FLOAT,  -- actual - predicted
    absolute_prediction_error FLOAT,
    relative_prediction_error FLOAT,  -- (actual - predicted) / abs(predicted)
    confidence_interval_coverage BOOLEAN,  -- Did actual fall within predicted CI?
    direction_match BOOLEAN,  -- Did signs of effect match?

    -- Fidelity score (0-1, higher is better)
    fidelity_score FLOAT CHECK (fidelity_score >= 0 AND fidelity_score <= 1),
    fidelity_grade VARCHAR(20),  -- 'excellent', 'good', 'fair', 'poor'

    -- Calibration adjustment (for improving future predictions)
    calibration_adjustment JSONB DEFAULT '{}',
    /*
    Example:
    {
        "bias_correction": 0.02,
        "variance_scaling": 1.1,
        "segment_adjustments": {
            "high_potential": -0.01,
            "low_potential": 0.03
        }
    }
    */

    -- Model feedback
    feedback_applied_to_model BOOLEAN DEFAULT false,
    feedback_applied_at TIMESTAMP WITH TIME ZONE,

    -- Audit
    compared_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_comparison UNIQUE(experiment_id, twin_simulation_id, comparison_type),
    CONSTRAINT valid_fidelity_grade CHECK (fidelity_grade IS NULL OR fidelity_grade IN ('excellent', 'good', 'fair', 'poor'))
);

-- Indexes for fidelity comparisons
CREATE INDEX IF NOT EXISTS idx_ab_fidelity_experiment ON ab_fidelity_comparisons(experiment_id);
CREATE INDEX IF NOT EXISTS idx_ab_fidelity_simulation ON ab_fidelity_comparisons(twin_simulation_id);
CREATE INDEX IF NOT EXISTS idx_ab_fidelity_score ON ab_fidelity_comparisons(fidelity_score);
CREATE INDEX IF NOT EXISTS idx_ab_fidelity_grade ON ab_fidelity_comparisons(fidelity_grade);

-- ============================================
-- SECTION 3: SUPPORTING FUNCTIONS
-- ============================================

-- Function to calculate fidelity grade from score
CREATE OR REPLACE FUNCTION calculate_fidelity_grade(p_fidelity_score FLOAT)
RETURNS VARCHAR(20) AS $$
BEGIN
    IF p_fidelity_score IS NULL THEN
        RETURN 'poor';
    ELSIF p_fidelity_score >= 0.9 THEN
        RETURN 'excellent';
    ELSIF p_fidelity_score >= 0.75 THEN
        RETURN 'good';
    ELSIF p_fidelity_score >= 0.5 THEN
        RETURN 'fair';
    ELSE
        RETURN 'poor';
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to compute fidelity score from prediction error
CREATE OR REPLACE FUNCTION compute_fidelity_score(
    p_predicted_effect FLOAT,
    p_actual_effect FLOAT,
    p_predicted_ci_lower FLOAT DEFAULT NULL,
    p_predicted_ci_upper FLOAT DEFAULT NULL
)
RETURNS FLOAT AS $$
DECLARE
    v_abs_error FLOAT;
    v_ci_width FLOAT;
    v_score FLOAT;
BEGIN
    IF p_predicted_effect IS NULL OR p_actual_effect IS NULL THEN
        RETURN NULL;
    END IF;

    v_abs_error := ABS(p_actual_effect - p_predicted_effect);

    -- Base score from prediction error (scaled)
    IF ABS(p_predicted_effect) > 0.001 THEN
        v_score := GREATEST(0, 1 - (v_abs_error / ABS(p_predicted_effect)));
    ELSE
        v_score := CASE WHEN v_abs_error < 0.01 THEN 1.0 ELSE 0.5 END;
    END IF;

    -- Bonus for CI coverage
    IF p_predicted_ci_lower IS NOT NULL AND p_predicted_ci_upper IS NOT NULL THEN
        IF p_actual_effect >= p_predicted_ci_lower AND p_actual_effect <= p_predicted_ci_upper THEN
            v_score := LEAST(1.0, v_score + 0.1);
        END IF;
    END IF;

    RETURN ROUND(v_score::numeric, 4);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to get experiment results summary
CREATE OR REPLACE FUNCTION get_experiment_results_summary(p_experiment_id UUID)
RETURNS TABLE(
    primary_metric VARCHAR(100),
    analysis_type ab_analysis_type,
    control_n INTEGER,
    treatment_n INTEGER,
    effect_estimate FLOAT,
    p_value FLOAT,
    is_significant BOOLEAN,
    computed_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.primary_metric,
        r.analysis_type,
        r.control_n,
        r.treatment_n,
        r.effect_estimate,
        r.p_value,
        r.is_significant,
        r.computed_at
    FROM ab_experiment_results r
    WHERE r.experiment_id = p_experiment_id
    ORDER BY r.computed_at DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- SECTION 4: TRIGGERS
-- ============================================

-- Trigger to auto-calculate fidelity grade
CREATE OR REPLACE FUNCTION set_fidelity_grade()
RETURNS TRIGGER AS $$
BEGIN
    NEW.fidelity_grade := calculate_fidelity_grade(NEW.fidelity_score);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_fidelity_grade
    BEFORE INSERT OR UPDATE OF fidelity_score ON ab_fidelity_comparisons
    FOR EACH ROW
    EXECUTE FUNCTION set_fidelity_grade();

-- Trigger to auto-calculate prediction errors
CREATE OR REPLACE FUNCTION set_prediction_errors()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.predicted_effect IS NOT NULL AND NEW.actual_effect IS NOT NULL THEN
        NEW.prediction_error := NEW.actual_effect - NEW.predicted_effect;
        NEW.absolute_prediction_error := ABS(NEW.prediction_error);

        IF ABS(NEW.predicted_effect) > 0.001 THEN
            NEW.relative_prediction_error := NEW.prediction_error / ABS(NEW.predicted_effect);
        END IF;

        NEW.direction_match := (
            (NEW.predicted_effect > 0 AND NEW.actual_effect > 0) OR
            (NEW.predicted_effect < 0 AND NEW.actual_effect < 0) OR
            (ABS(NEW.predicted_effect) < 0.001 AND ABS(NEW.actual_effect) < 0.001)
        );

        IF NEW.predicted_ci_lower IS NOT NULL AND NEW.predicted_ci_upper IS NOT NULL THEN
            NEW.confidence_interval_coverage := (
                NEW.actual_effect >= NEW.predicted_ci_lower AND
                NEW.actual_effect <= NEW.predicted_ci_upper
            );
        END IF;

        -- Auto-calculate fidelity score if not provided
        IF NEW.fidelity_score IS NULL THEN
            NEW.fidelity_score := compute_fidelity_score(
                NEW.predicted_effect,
                NEW.actual_effect,
                NEW.predicted_ci_lower,
                NEW.predicted_ci_upper
            );
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_prediction_errors
    BEFORE INSERT OR UPDATE OF predicted_effect, actual_effect ON ab_fidelity_comparisons
    FOR EACH ROW
    EXECUTE FUNCTION set_prediction_errors();

-- ============================================
-- SECTION 5: VIEWS
-- ============================================

-- View: Active experiments with latest results
CREATE OR REPLACE VIEW vw_experiment_results_summary AS
SELECT
    e.id AS experiment_id,
    e.experiment_name,
    e.prediction_target,
    r.analysis_type,
    r.primary_metric,
    r.effect_estimate,
    r.p_value,
    r.is_significant,
    r.control_n + r.treatment_n AS total_n,
    r.computed_at
FROM ml_experiments e
LEFT JOIN LATERAL (
    SELECT *
    FROM ab_experiment_results ar
    WHERE ar.experiment_id = e.id
    ORDER BY ar.computed_at DESC
    LIMIT 1
) r ON true;

-- View: Fidelity tracking summary
CREATE OR REPLACE VIEW vw_fidelity_summary AS
SELECT
    fc.experiment_id,
    e.experiment_name,
    COUNT(fc.id) AS comparison_count,
    AVG(fc.fidelity_score) AS avg_fidelity_score,
    AVG(fc.absolute_prediction_error) AS avg_abs_error,
    SUM(CASE WHEN fc.direction_match THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS direction_accuracy,
    SUM(CASE WHEN fc.confidence_interval_coverage THEN 1 ELSE 0 END)::FLOAT / COUNT(*)::FLOAT AS ci_coverage_rate
FROM ab_fidelity_comparisons fc
JOIN ml_experiments e ON fc.experiment_id = e.id
GROUP BY fc.experiment_id, e.experiment_name;

-- ============================================
-- SECTION 6: COMMENTS
-- ============================================

COMMENT ON TABLE ab_experiment_results IS 'Computed A/B test results with effect estimates and statistical inference';
COMMENT ON TABLE ab_srm_checks IS 'Sample Ratio Mismatch detection history for experiment health monitoring';
COMMENT ON TABLE ab_fidelity_comparisons IS 'Digital Twin prediction accuracy tracking vs actual experiment results';

COMMENT ON COLUMN ab_experiment_results.segment_results IS 'Heterogeneous treatment effects by segment (specialty, decile, etc.)';
COMMENT ON COLUMN ab_srm_checks.severity IS 'SRM severity based on p-value: none (no SRM), minor (p>0.01), moderate (0.001<p<=0.01), severe (p<=0.001)';
COMMENT ON COLUMN ab_fidelity_comparisons.fidelity_score IS 'Overall prediction accuracy score (0-1), auto-computed from prediction errors';
