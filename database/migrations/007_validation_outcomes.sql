-- ============================================================================
-- Migration 007: Validation Outcomes Persistence
-- ============================================================================
-- Purpose: Store validation outcomes for ExperimentKnowledgeStore
-- Part of: Feedback Loop Architecture for Concept Drift Detection
-- Reference: .claude/plans/feedback-loop-concept-drift-audit.md
-- ============================================================================

-- Table to store validation outcomes from causal validation protocol
CREATE TABLE IF NOT EXISTS validation_outcomes (
    -- Primary Key
    outcome_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Validation reference
    estimate_id VARCHAR(100),

    -- Outcome classification
    outcome_type VARCHAR(30) NOT NULL CHECK (
        outcome_type IN (
            'passed',
            'failed_refutation',
            'failed_sensitivity',
            'failed_placebo',
            'partial_pass',
            'inconclusive'
        )
    ),

    -- Causal relationship identifiers
    treatment_variable VARCHAR(200),
    outcome_variable VARCHAR(200),
    brand VARCHAR(50),

    -- Analysis details
    sample_size INTEGER,
    effect_size DECIMAL(10,6),
    confidence_interval JSONB DEFAULT '[]',

    -- Failure patterns (array of pattern objects)
    failure_patterns JSONB DEFAULT '[]',

    -- Additional metadata
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for querying by outcome type
CREATE INDEX IF NOT EXISTS idx_validation_outcomes_type
    ON validation_outcomes(outcome_type);

-- Index for querying by treatment/outcome variables
CREATE INDEX IF NOT EXISTS idx_validation_outcomes_variables
    ON validation_outcomes(treatment_variable, outcome_variable);

-- Index for querying by brand
CREATE INDEX IF NOT EXISTS idx_validation_outcomes_brand
    ON validation_outcomes(brand);

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS idx_validation_outcomes_timestamp
    ON validation_outcomes(timestamp DESC);

-- GIN index for JSONB failure_patterns queries
CREATE INDEX IF NOT EXISTS idx_validation_outcomes_patterns
    ON validation_outcomes USING GIN (failure_patterns);

-- ============================================================================
-- Trigger for updated_at
-- ============================================================================

CREATE OR REPLACE FUNCTION update_validation_outcomes_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_validation_outcomes_updated ON validation_outcomes;
CREATE TRIGGER trg_validation_outcomes_updated
    BEFORE UPDATE ON validation_outcomes
    FOR EACH ROW
    EXECUTE FUNCTION update_validation_outcomes_updated_at();

-- ============================================================================
-- View: Aggregated Failure Patterns
-- ============================================================================

CREATE OR REPLACE VIEW v_validation_failure_patterns AS
SELECT
    pattern->>'category' AS category,
    pattern->>'test_name' AS test_name,
    COUNT(*) AS failure_count,
    AVG((pattern->>'delta_percent')::DECIMAL) AS avg_delta_percent,
    array_agg(DISTINCT pattern->>'recommendation') AS recommendations,
    MAX(timestamp) AS last_seen
FROM
    validation_outcomes vo,
    jsonb_array_elements(vo.failure_patterns) AS pattern
WHERE
    vo.outcome_type != 'passed'
GROUP BY
    pattern->>'category',
    pattern->>'test_name'
ORDER BY
    failure_count DESC;

-- ============================================================================
-- View: Recent Failures Summary
-- ============================================================================

CREATE OR REPLACE VIEW v_validation_recent_failures AS
SELECT
    outcome_id,
    outcome_type,
    treatment_variable,
    outcome_variable,
    brand,
    sample_size,
    effect_size,
    jsonb_array_length(failure_patterns) AS pattern_count,
    timestamp,
    -- Extract first failure category
    (failure_patterns->0->>'category') AS primary_failure_category
FROM
    validation_outcomes
WHERE
    outcome_type != 'passed'
    AND timestamp >= NOW() - INTERVAL '30 days'
ORDER BY
    timestamp DESC;

-- ============================================================================
-- Function: Query Similar Failures
-- ============================================================================

CREATE OR REPLACE FUNCTION query_similar_validation_failures(
    p_treatment_variable VARCHAR(200),
    p_outcome_variable VARCHAR(200),
    p_limit INTEGER DEFAULT 5
)
RETURNS TABLE (
    outcome_id UUID,
    outcome_type VARCHAR(30),
    treatment_variable VARCHAR(200),
    outcome_variable VARCHAR(200),
    effect_size DECIMAL(10,6),
    failure_patterns JSONB,
    similarity_score INTEGER,
    timestamp TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        vo.outcome_id,
        vo.outcome_type,
        vo.treatment_variable,
        vo.outcome_variable,
        vo.effect_size,
        vo.failure_patterns,
        -- Simple similarity scoring
        CASE
            WHEN vo.treatment_variable = p_treatment_variable AND vo.outcome_variable = p_outcome_variable THEN 4
            WHEN vo.treatment_variable = p_treatment_variable THEN 2
            WHEN vo.outcome_variable = p_outcome_variable THEN 2
            WHEN vo.treatment_variable ILIKE '%' || p_treatment_variable || '%' THEN 1
            WHEN vo.outcome_variable ILIKE '%' || p_outcome_variable || '%' THEN 1
            ELSE 0
        END AS similarity_score,
        vo.timestamp
    FROM validation_outcomes vo
    WHERE vo.outcome_type != 'passed'
    ORDER BY similarity_score DESC, vo.timestamp DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- RLS Policies (if enabled)
-- ============================================================================

-- Enable RLS on the table
ALTER TABLE validation_outcomes ENABLE ROW LEVEL SECURITY;

-- Policy for authenticated users (read access)
CREATE POLICY validation_outcomes_read_policy ON validation_outcomes
    FOR SELECT
    USING (auth.role() = 'authenticated' OR auth.role() = 'service_role');

-- Policy for service role (full access)
CREATE POLICY validation_outcomes_write_policy ON validation_outcomes
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE validation_outcomes IS
    'Stores validation outcomes from causal validation protocol for learning';

COMMENT ON COLUMN validation_outcomes.outcome_type IS
    'Type of validation outcome: passed, failed_refutation, failed_sensitivity, etc.';

COMMENT ON COLUMN validation_outcomes.failure_patterns IS
    'Array of failure pattern objects with category, test_name, delta_percent, recommendation';

COMMENT ON VIEW v_validation_failure_patterns IS
    'Aggregated view of failure patterns across all validation outcomes';

COMMENT ON VIEW v_validation_recent_failures IS
    'Recent validation failures from the last 30 days';

COMMENT ON FUNCTION query_similar_validation_failures IS
    'Query validation failures similar to a given treatment/outcome pair';
