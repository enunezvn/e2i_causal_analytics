-- =============================================================================
-- ROI Calculations Table
-- Migration: 014_roi_calculations.sql
-- Date: 2025-12-20
-- Purpose: Persistence for ROI calculations from ROICalculationService
-- Reference: docs/roi_methodology.md
-- =============================================================================

-- -----------------------------------------------------------------------------
-- ENUMs for ROI Calculations
-- -----------------------------------------------------------------------------

-- Value driver types
CREATE TYPE value_driver_type AS ENUM (
    'trx_lift',                 -- $850/TRx
    'patient_identification',   -- $1,200/patient
    'action_rate',              -- $45/pp/1000 triggers
    'intent_to_prescribe',      -- $320/HCP/pp
    'data_quality',             -- $200/FP, $650/FN
    'drift_prevention'          -- 2x multiplier
);

-- Attribution levels
CREATE TYPE attribution_level_type AS ENUM (
    'full',       -- 100% - RCT validated, sole driver
    'partial',    -- 50-80% - Primary driver, some confounding
    'shared',     -- 20-50% - Multiple initiatives contribute
    'minimal'     -- <20% - Minor contributor, correlation only
);

-- Risk levels
CREATE TYPE risk_level_type AS ENUM (
    'low',
    'medium',
    'high'
);

-- Initiative types for cost estimation
CREATE TYPE initiative_type AS ENUM (
    'data_source_integration',
    'new_ml_model',
    'algorithm_optimization',
    'dashboard_enhancement',
    'trigger_redesign',
    'ab_test_implementation',
    'other'
);

-- -----------------------------------------------------------------------------
-- Main ROI Calculations Table
-- -----------------------------------------------------------------------------

CREATE TABLE roi_calculations (
    -- Primary Key
    calculation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Initiative Identity
    initiative_name VARCHAR(200) NOT NULL,
    initiative_type initiative_type NOT NULL DEFAULT 'other',
    initiative_description TEXT,
    brand brand_type,
    workstream workstream_type,

    -- Core ROI Metrics
    incremental_value DECIMAL(15,2) NOT NULL,
    attributed_value DECIMAL(15,2) NOT NULL,
    implementation_cost DECIMAL(15,2) NOT NULL,
    base_roi DECIMAL(8,2) NOT NULL,              -- Before adjustments
    risk_adjusted_roi DECIMAL(8,2) NOT NULL,     -- After risk adjustment

    -- Confidence Interval (95% CI from bootstrap)
    ci_lower DECIMAL(8,2) NOT NULL,              -- 2.5th percentile
    ci_median DECIMAL(8,2) NOT NULL,             -- 50th percentile (median)
    ci_upper DECIMAL(8,2) NOT NULL,              -- 97.5th percentile
    probability_positive DECIMAL(5,4),           -- P(ROI > 1x)
    probability_target DECIMAL(5,4),             -- P(ROI > target)
    target_roi DECIMAL(8,2) DEFAULT 5.0,         -- Target ROI threshold
    simulation_count INTEGER DEFAULT 1000,       -- Bootstrap simulations

    -- Attribution
    attribution_level attribution_level_type NOT NULL DEFAULT 'partial',
    attribution_rate DECIMAL(5,4) NOT NULL,      -- 0.0-1.0

    -- Risk Assessment (individual factors)
    risk_technical_complexity risk_level_type DEFAULT 'low',
    risk_organizational_change risk_level_type DEFAULT 'low',
    risk_data_dependencies risk_level_type DEFAULT 'low',
    risk_timeline_uncertainty risk_level_type DEFAULT 'low',
    total_risk_adjustment DECIMAL(5,4),          -- Combined risk adjustment

    -- Value Breakdown (JSONB for flexibility)
    value_by_driver JSONB NOT NULL DEFAULT '{}',
    -- Example: {"trx_lift": 850000, "patient_identification": 360000}

    -- Cost Breakdown (JSONB for flexibility)
    cost_breakdown JSONB NOT NULL DEFAULT '{}',
    -- Example: {"engineering": 62500, "data_acquisition": 150000, "change_management": 80000}

    -- Sensitivity Analysis (JSONB array)
    sensitivity_results JSONB,
    -- Example: [{"parameter": "trx_lift", "impact_range": 3.2, ...}]

    -- NPV Metrics (for multi-year initiatives)
    npv_value DECIMAL(15,2),
    npv_roi DECIMAL(8,2),
    npv_years INTEGER,
    discount_rate DECIMAL(5,4) DEFAULT 0.10,

    -- Metadata
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    calculated_by VARCHAR(100),                  -- User or agent that created
    methodology_version VARCHAR(20) DEFAULT '1.0',

    -- Linked Entities
    gap_id UUID,                                 -- If linked to a performance gap
    causal_path_id UUID,                         -- If linked to a causal analysis
    experiment_id UUID,                          -- If linked to an A/B test design

    -- Audit Fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    superseded_by UUID REFERENCES roi_calculations(calculation_id),
    notes TEXT
);

-- -----------------------------------------------------------------------------
-- Value Driver Details Table (for itemized tracking)
-- -----------------------------------------------------------------------------

CREATE TABLE roi_value_driver_details (
    detail_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL REFERENCES roi_calculations(calculation_id) ON DELETE CASCADE,

    -- Driver Info
    driver_type value_driver_type NOT NULL,
    quantity DECIMAL(12,2) NOT NULL,             -- Number of units
    unit_value DECIMAL(10,2) NOT NULL,           -- Dollar value per unit
    total_value DECIMAL(15,2) NOT NULL,          -- quantity × unit_value

    -- Optional Parameters (based on driver type)
    hcp_count INTEGER,                           -- For ITP calculations
    trigger_count INTEGER,                       -- For action rate
    fp_reduction INTEGER,                        -- For data quality
    fn_reduction INTEGER,                        -- For data quality
    auc_drop_prevented DECIMAL(5,4),             -- For drift prevention
    baseline_model_value DECIMAL(15,2),          -- For drift prevention

    -- Uncertainty
    uncertainty_std DECIMAL(5,4) DEFAULT 0.15,   -- Std dev as fraction of mean

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- Cost Category Details Table
-- -----------------------------------------------------------------------------

CREATE TABLE roi_cost_details (
    detail_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL REFERENCES roi_calculations(calculation_id) ON DELETE CASCADE,

    -- Cost Category
    category VARCHAR(50) NOT NULL,               -- engineering, data_acquisition, etc.
    subcategory VARCHAR(100),                    -- More specific (e.g., IQVIA APLD)

    -- Cost Details
    quantity DECIMAL(10,2),                      -- e.g., engineering days
    unit_cost DECIMAL(10,2),                     -- e.g., $2,500/day
    total_cost DECIMAL(15,2) NOT NULL,

    -- Flags
    is_one_time BOOLEAN DEFAULT TRUE,            -- vs recurring
    recurrence_months INTEGER,                   -- If recurring, for how long

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- Indexes
-- -----------------------------------------------------------------------------

CREATE INDEX idx_roi_calculations_initiative ON roi_calculations(initiative_name);
CREATE INDEX idx_roi_calculations_brand ON roi_calculations(brand);
CREATE INDEX idx_roi_calculations_workstream ON roi_calculations(workstream);
CREATE INDEX idx_roi_calculations_calculated_at ON roi_calculations(calculated_at DESC);
CREATE INDEX idx_roi_calculations_gap_id ON roi_calculations(gap_id) WHERE gap_id IS NOT NULL;
CREATE INDEX idx_roi_calculations_active ON roi_calculations(is_active) WHERE is_active = TRUE;

CREATE INDEX idx_roi_value_details_calculation ON roi_value_driver_details(calculation_id);
CREATE INDEX idx_roi_value_details_driver ON roi_value_driver_details(driver_type);

CREATE INDEX idx_roi_cost_details_calculation ON roi_cost_details(calculation_id);
CREATE INDEX idx_roi_cost_details_category ON roi_cost_details(category);

-- JSONB indexes for value/cost breakdown queries
CREATE INDEX idx_roi_calculations_value_drivers ON roi_calculations USING gin(value_by_driver);
CREATE INDEX idx_roi_calculations_cost_breakdown ON roi_calculations USING gin(cost_breakdown);

-- -----------------------------------------------------------------------------
-- Triggers
-- -----------------------------------------------------------------------------

-- Update updated_at on change
CREATE TRIGGER update_roi_calculations_timestamp
    BEFORE UPDATE ON roi_calculations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- -----------------------------------------------------------------------------
-- Helper Views
-- -----------------------------------------------------------------------------

-- View for active ROI calculations with key metrics
CREATE OR REPLACE VIEW v_roi_calculations_summary AS
SELECT
    calculation_id,
    initiative_name,
    initiative_type,
    brand,
    workstream,
    incremental_value,
    implementation_cost,
    base_roi,
    risk_adjusted_roi,
    ci_lower,
    ci_median,
    ci_upper,
    probability_positive,
    attribution_level,
    calculated_at
FROM roi_calculations
WHERE is_active = TRUE
ORDER BY calculated_at DESC;

-- View for ROI by brand
CREATE OR REPLACE VIEW v_roi_by_brand AS
SELECT
    brand,
    COUNT(*) AS calculation_count,
    AVG(risk_adjusted_roi) AS avg_roi,
    SUM(attributed_value) AS total_attributed_value,
    SUM(implementation_cost) AS total_cost,
    AVG(probability_positive) AS avg_probability_positive
FROM roi_calculations
WHERE is_active = TRUE
  AND brand IS NOT NULL
GROUP BY brand;

-- View for ROI by workstream
CREATE OR REPLACE VIEW v_roi_by_workstream AS
SELECT
    workstream,
    COUNT(*) AS calculation_count,
    AVG(risk_adjusted_roi) AS avg_roi,
    SUM(attributed_value) AS total_attributed_value,
    SUM(implementation_cost) AS total_cost,
    AVG(probability_positive) AS avg_probability_positive
FROM roi_calculations
WHERE is_active = TRUE
  AND workstream IS NOT NULL
GROUP BY workstream;

-- View for value driver contribution analysis
CREATE OR REPLACE VIEW v_roi_value_driver_contribution AS
SELECT
    rc.calculation_id,
    rc.initiative_name,
    rc.brand,
    vd.driver_type,
    vd.total_value,
    vd.total_value / NULLIF(rc.incremental_value, 0) AS contribution_pct
FROM roi_calculations rc
JOIN roi_value_driver_details vd ON rc.calculation_id = vd.calculation_id
WHERE rc.is_active = TRUE
ORDER BY rc.calculated_at DESC, vd.total_value DESC;

-- -----------------------------------------------------------------------------
-- RLS Policies (if using Supabase RLS)
-- -----------------------------------------------------------------------------

-- Enable RLS
ALTER TABLE roi_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE roi_value_driver_details ENABLE ROW LEVEL SECURITY;
ALTER TABLE roi_cost_details ENABLE ROW LEVEL SECURITY;

-- Allow all authenticated users to read
CREATE POLICY "Allow authenticated read roi_calculations" ON roi_calculations
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "Allow authenticated read roi_value_driver_details" ON roi_value_driver_details
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "Allow authenticated read roi_cost_details" ON roi_cost_details
    FOR SELECT TO authenticated USING (true);

-- Allow service role full access
CREATE POLICY "Allow service role full access roi_calculations" ON roi_calculations
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Allow service role full access roi_value_driver_details" ON roi_value_driver_details
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Allow service role full access roi_cost_details" ON roi_cost_details
    FOR ALL TO service_role USING (true) WITH CHECK (true);

-- -----------------------------------------------------------------------------
-- Comments
-- -----------------------------------------------------------------------------

COMMENT ON TABLE roi_calculations IS
    'Stores ROI calculations from ROICalculationService. Each record represents a complete ROI analysis for an initiative.';

COMMENT ON COLUMN roi_calculations.value_by_driver IS
    'JSONB breakdown of incremental value by value driver type. Example: {"trx_lift": 850000, "patient_identification": 360000}';

COMMENT ON COLUMN roi_calculations.cost_breakdown IS
    'JSONB breakdown of implementation cost by category. Example: {"engineering": 62500, "data_acquisition": 150000}';

COMMENT ON COLUMN roi_calculations.sensitivity_results IS
    'JSONB array of sensitivity analysis results for tornado diagram. Each entry has parameter, impact_range, roi_at_low/base/high.';

COMMENT ON TABLE roi_value_driver_details IS
    'Itemized value driver details for each ROI calculation. Supports detailed audit trail and analysis.';

COMMENT ON TABLE roi_cost_details IS
    'Itemized cost details for each ROI calculation. Tracks one-time vs recurring costs.';

-- -----------------------------------------------------------------------------
-- Verification
-- -----------------------------------------------------------------------------

DO $$
BEGIN
    -- Verify tables created
    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'roi_calculations') THEN
        RAISE NOTICE 'SUCCESS: roi_calculations table created';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'roi_value_driver_details') THEN
        RAISE NOTICE 'SUCCESS: roi_value_driver_details table created';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'roi_cost_details') THEN
        RAISE NOTICE 'SUCCESS: roi_cost_details table created';
    END IF;

    -- Verify views created
    IF EXISTS (SELECT 1 FROM pg_views WHERE viewname = 'v_roi_calculations_summary') THEN
        RAISE NOTICE 'SUCCESS: v_roi_calculations_summary view created';
    END IF;

    RAISE NOTICE '';
    RAISE NOTICE '=== ROI Calculations Schema Complete ===';
    RAISE NOTICE 'Tables: roi_calculations, roi_value_driver_details, roi_cost_details';
    RAISE NOTICE 'Views: v_roi_calculations_summary, v_roi_by_brand, v_roi_by_workstream, v_roi_value_driver_contribution';
    RAISE NOTICE 'ENUMs: value_driver_type, attribution_level_type, risk_level_type, initiative_type';
END $$;
