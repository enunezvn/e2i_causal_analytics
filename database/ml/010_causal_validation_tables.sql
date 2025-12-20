-- ============================================================
-- Migration: 010_causal_validation_tables.sql
-- Version: V4.3 (Corrected)
-- Purpose: Add causal validation infrastructure for DoWhy refutation tests
-- 
-- Creates:
--    - ENUMs: refutation_test_type, validation_status, gate_decision, expert_review_type
--    - Tables: causal_validations, expert_reviews
--    - Views: v_validation_summary, v_active_expert_approvals, v_blocked_estimates
--    - Functions: is_dag_approved(), get_validation_gate()
--
-- Dependencies:
--    - agent_activities table (from 003_core_tables.sql)
--    - causal_paths table (from 003_core_tables.sql)
-- ============================================================

-- ============================================================
-- 1. CREATE ENUM TYPES
-- ============================================================

-- Refutation test types (DoWhy refuters)
CREATE TYPE refutation_test_type AS ENUM (
    'placebo_treatment',      -- Replace treatment with random noise → effect should disappear
    'random_common_cause',    -- Add fake confounder → effect should remain stable
    'data_subset',            -- Test on random subsamples → effect should be consistent
    'bootstrap',              -- Bootstrap resampling → estimate variance
    'sensitivity_e_value'     -- E-value sensitivity → how strong unmeasured confounding must be
);

COMMENT ON TYPE refutation_test_type IS 'DoWhy refutation test types for causal estimate validation';

-- Validation status (per-test result)
CREATE TYPE validation_status AS ENUM (
    'passed',    -- Test passed all criteria
    'failed',    -- Test failed criteria
    'warning',   -- Borderline result, recommend review
    'skipped'    -- Test not applicable or disabled
);

COMMENT ON TYPE validation_status IS 'Result status of individual refutation tests';

-- Gate decision (aggregate suite result)
CREATE TYPE gate_decision AS ENUM (
    'proceed',   -- All required tests passed, confidence ≥ 0.7, safe for production
    'review',    -- Partial pass, requires expert review before use
    'block'      -- Failed required tests, do not use estimate
);

COMMENT ON TYPE gate_decision IS 'Aggregate decision from RefutationSuite determining if estimate can be used';

-- Expert review types
CREATE TYPE expert_review_type AS ENUM (
    'dag_approval',         -- New DAG structure requires expert sign-off
    'methodology_review',   -- Causal method validation
    'quarterly_audit',      -- Scheduled periodic review
    'ad_hoc_validation'     -- On-demand expert review
);

COMMENT ON TYPE expert_review_type IS 'Types of domain expert validation for causal analysis';


-- ============================================================
-- 2. CREATE causal_validations TABLE
-- ============================================================

CREATE TABLE causal_validations (
    -- Primary key
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Foreign keys (polymorphic - can reference causal_paths or ml_experiments)
    estimate_id UUID NOT NULL,
    estimate_source VARCHAR(50) DEFAULT 'causal_paths',  -- 'causal_paths' or 'ml_experiments'
    
    -- Test identification
    test_type refutation_test_type NOT NULL,
    
    -- Results
    status validation_status NOT NULL,
    original_effect DECIMAL(12,6),          -- Original causal effect estimate
    refuted_effect DECIMAL(12,6),           -- Effect after refutation manipulation
    p_value DECIMAL(6,5),                   -- Statistical significance (if applicable)
    delta_percent DECIMAL(8,4),             -- Percentage change from original
    
    -- Aggregate scoring
    confidence_score DECIMAL(4,3) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    gate_decision gate_decision NOT NULL DEFAULT 'review',
    
    -- Configuration and details
    test_config JSONB,                      -- Test parameters used (num_simulations, thresholds, etc.)
    details_json JSONB,                     -- Full DoWhy refutation output
    
    -- Traceability
    agent_activity_id VARCHAR(100),         -- FK to agent_activities
    
    brand VARCHAR(50),                      -- Brand context (Remibrutinib, Fabhalta, Kisqali)
    analysis_context TEXT,                  -- Description of what was being analyzed
    treatment_variable VARCHAR(100),        -- What treatment is being analyzed
    outcome_variable VARCHAR(100),          -- What outcome is being measured
    
    -- ML split tracking (for compliance)
    data_split VARCHAR(20),                 -- train, validation, test, holdout
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    
    -- Foreign key constraints
    CONSTRAINT fk_validation_agent_activity 
        FOREIGN KEY (agent_activity_id) 
        REFERENCES agent_activities(activity_id)
        ON DELETE SET NULL
);

-- Table comment
COMMENT ON TABLE causal_validations IS 'Stores DoWhy refutation test results for validating causal estimates. Integrates with Causal Impact agent Node 3 (Refutation).';

-- Column comments
COMMENT ON COLUMN causal_validations.estimate_id IS 'ID of the causal estimate being validated (from causal_paths or ml_experiments)';
COMMENT ON COLUMN causal_validations.estimate_source IS 'Source table: causal_paths or ml_experiments';
COMMENT ON COLUMN causal_validations.test_type IS 'Which DoWhy refutation test was run';
COMMENT ON COLUMN causal_validations.status IS 'Pass/fail result of this specific test';
COMMENT ON COLUMN causal_validations.original_effect IS 'The causal effect estimate before refutation';
COMMENT ON COLUMN causal_validations.refuted_effect IS 'The effect after applying refutation manipulation';
COMMENT ON COLUMN causal_validations.confidence_score IS 'Weighted confidence from all tests (0-1)';
COMMENT ON COLUMN causal_validations.gate_decision IS 'Whether to proceed, review, or block based on test results';
COMMENT ON COLUMN causal_validations.treatment_variable IS 'The treatment/intervention being analyzed';
COMMENT ON COLUMN causal_validations.outcome_variable IS 'The outcome metric being measured';


-- ============================================================
-- 3. CREATE expert_reviews TABLE
-- ============================================================

CREATE TABLE expert_reviews (
    -- Primary key
    review_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Review identification
    review_type expert_review_type NOT NULL,
    dag_version_hash VARCHAR(64),           -- SHA256 hash of DAG structure for versioning
    
    -- Reviewer information
    reviewer_id VARCHAR(100) NOT NULL,      -- User ID of the reviewer
    reviewer_name VARCHAR(200),             -- Display name
    reviewer_role VARCHAR(100),             -- 'commercial_ops', 'medical_affairs', 'data_science', etc.
    reviewer_email VARCHAR(200),
    
    -- Approval status
    approval_status VARCHAR(30) NOT NULL DEFAULT 'pending',  -- pending, approved, rejected, expired
    
    -- Review content
    checklist_json JSONB,                   -- Completed checklist items with responses
    comments_json JSONB,                    -- Reviewer notes and feedback
    concerns_raised TEXT[],                 -- Array of specific concerns
    conditions TEXT,                        -- Any conditions on approval
    
    -- Context
    brand VARCHAR(50),                      -- Brand context
    analysis_context TEXT,                  -- What analysis this review covers
    treatment_variable VARCHAR(100),        -- What treatment is being analyzed
    outcome_variable VARCHAR(100),          -- What outcome is being measured
    
    -- Validity period
    valid_from DATE DEFAULT CURRENT_DATE,
    valid_until DATE,                       -- Expiration for quarterly reviews
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    approved_at TIMESTAMPTZ,                -- When approval was granted
    
    -- Links
    related_validation_ids UUID[],          -- Array of related causal_validations
    supersedes_review_id UUID               -- If this review replaces a previous one
);

-- Table comment
COMMENT ON TABLE expert_reviews IS 'Tracks domain expert validation of causal DAGs and methodology. Required for new DAG construction and quarterly audits.';

-- Column comments
COMMENT ON COLUMN expert_reviews.dag_version_hash IS 'SHA256 hash of DAG structure to track which version was approved';
COMMENT ON COLUMN expert_reviews.checklist_json IS 'Completed DAG review checklist (confounder completeness, edge direction, SUTVA, etc.)';
COMMENT ON COLUMN expert_reviews.valid_until IS 'Expiration date - quarterly reviews require renewal';
COMMENT ON COLUMN expert_reviews.supersedes_review_id IS 'If this review replaces a previous one, reference here';


-- ============================================================
-- 4. CREATE INDEXES
-- ============================================================

-- causal_validations indexes
CREATE INDEX idx_cv_estimate_id ON causal_validations(estimate_id);
CREATE INDEX idx_cv_status ON causal_validations(status);
CREATE INDEX idx_cv_gate_decision ON causal_validations(gate_decision);
CREATE INDEX idx_cv_test_type ON causal_validations(test_type);
CREATE INDEX idx_cv_created_at ON causal_validations(created_at DESC);
CREATE INDEX idx_cv_agent_activity ON causal_validations(agent_activity_id);
CREATE INDEX idx_cv_brand ON causal_validations(brand);
CREATE INDEX idx_cv_treatment ON causal_validations(treatment_variable);

-- Composite index for common query pattern
CREATE INDEX idx_cv_estimate_gate ON causal_validations(estimate_id, gate_decision);
CREATE INDEX idx_cv_estimate_source ON causal_validations(estimate_id, estimate_source);

-- expert_reviews indexes
CREATE INDEX idx_er_dag_hash ON expert_reviews(dag_version_hash);
CREATE INDEX idx_er_approval_status ON expert_reviews(approval_status);
CREATE INDEX idx_er_reviewer ON expert_reviews(reviewer_id);
CREATE INDEX idx_er_review_type ON expert_reviews(review_type);
CREATE INDEX idx_er_brand ON expert_reviews(brand);
CREATE INDEX idx_er_valid_until ON expert_reviews(valid_until);
CREATE INDEX idx_er_treatment ON expert_reviews(treatment_variable);

-- Index for finding active approvals
CREATE INDEX idx_er_active_approvals ON expert_reviews(dag_version_hash, valid_until) 
    WHERE approval_status = 'approved';


-- ============================================================
-- 5. CREATE VIEWS
-- ============================================================

-- Validation summary view (aggregates all tests for an estimate)
CREATE OR REPLACE VIEW v_validation_summary AS
SELECT 
    estimate_id,
    estimate_source,
    brand,
    treatment_variable,
    outcome_variable,
    COUNT(*) AS total_tests,
    COUNT(*) FILTER (WHERE status = 'passed') AS passed_count,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed_count,
    COUNT(*) FILTER (WHERE status = 'warning') AS warning_count,
    COUNT(*) FILTER (WHERE status = 'skipped') AS skipped_count,
    ROUND(AVG(confidence_score), 3) AS avg_confidence,
    MIN(confidence_score) AS min_confidence,
    MAX(confidence_score) AS max_confidence,
    -- Gate decision priority: block > review > proceed
    CASE 
        WHEN COUNT(*) FILTER (WHERE gate_decision = 'block') > 0 THEN 'block'
        WHEN COUNT(*) FILTER (WHERE gate_decision = 'review') > 0 THEN 'review'
        ELSE 'proceed'
    END AS final_gate,
    MAX(created_at) AS last_validation_at,
    MIN(created_at) AS first_validation_at,
    -- List of failed tests
    ARRAY_AGG(DISTINCT test_type) FILTER (WHERE status = 'failed') AS failed_tests
FROM causal_validations
GROUP BY estimate_id, estimate_source, brand, treatment_variable, outcome_variable;

COMMENT ON VIEW v_validation_summary IS 'Aggregated validation results per causal estimate with final gate decision';


-- Active expert approvals view
CREATE OR REPLACE VIEW v_active_expert_approvals AS
SELECT 
    review_id,
    review_type,
    dag_version_hash,
    reviewer_id,
    reviewer_name,
    reviewer_role,
    brand,
    treatment_variable,
    outcome_variable,
    analysis_context,
    approved_at,
    valid_until,
    CASE 
        WHEN valid_until IS NULL THEN 'permanent'
        WHEN valid_until >= CURRENT_DATE THEN 'active'
        ELSE 'expired'
    END AS validity_status,
    CASE 
        WHEN valid_until IS NULL THEN NULL
        -- *** FIXED: Removed EXTRACT() because Date-Date returns Integer already ***
        ELSE (valid_until - CURRENT_DATE)
    END AS days_until_expiry,
    checklist_json,
    conditions
FROM expert_reviews
WHERE approval_status = 'approved';

COMMENT ON VIEW v_active_expert_approvals IS 'Currently active expert approvals with expiration tracking';


-- Blocked estimates view (for quick identification)
CREATE OR REPLACE VIEW v_blocked_estimates AS
SELECT DISTINCT
    cv.estimate_id,
    cv.estimate_source,
    cv.brand,
    cv.treatment_variable,
    cv.outcome_variable,
    cv.analysis_context,
    STRING_AGG(DISTINCT cv.test_type::text, ', ') AS failed_tests,
    MIN(cv.confidence_score) AS min_confidence,
    MAX(cv.created_at) AS blocked_at,
    COUNT(*) AS failed_test_count
FROM causal_validations cv
WHERE cv.gate_decision = 'block'
GROUP BY cv.estimate_id, cv.estimate_source, cv.brand, 
         cv.treatment_variable, cv.outcome_variable, cv.analysis_context;

COMMENT ON VIEW v_blocked_estimates IS 'Causal estimates that failed validation and should not be used';


-- Pending expert reviews view
CREATE OR REPLACE VIEW v_pending_expert_reviews AS
SELECT 
    review_id,
    review_type,
    dag_version_hash,
    brand,
    treatment_variable,
    outcome_variable,
    analysis_context,
    created_at,
    EXTRACT(DAY FROM (now() - created_at)) AS days_pending
FROM expert_reviews
WHERE approval_status = 'pending'
ORDER BY created_at ASC;

COMMENT ON VIEW v_pending_expert_reviews IS 'Expert reviews awaiting approval, ordered by age';


-- ============================================================
-- 6. CREATE FUNCTIONS
-- ============================================================

-- Function to check if a DAG is approved
CREATE OR REPLACE FUNCTION is_dag_approved(
    p_dag_hash VARCHAR(64), 
    p_brand VARCHAR(50) DEFAULT NULL
)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 
        FROM expert_reviews 
        WHERE dag_version_hash = p_dag_hash
          AND approval_status = 'approved'
          AND (valid_until IS NULL OR valid_until >= CURRENT_DATE)
          AND (p_brand IS NULL OR brand = p_brand)
    );
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION is_dag_approved IS 'Check if a DAG (by hash) has an active expert approval';


-- Function to get validation gate for an estimate
CREATE OR REPLACE FUNCTION get_validation_gate(p_estimate_id UUID)
RETURNS gate_decision AS $$
DECLARE
    v_gate gate_decision;
BEGIN
    SELECT 
        CASE 
            WHEN COUNT(*) FILTER (WHERE gate_decision = 'block') > 0 THEN 'block'::gate_decision
            WHEN COUNT(*) FILTER (WHERE gate_decision = 'review') > 0 THEN 'review'::gate_decision
            ELSE 'proceed'::gate_decision
        END INTO v_gate
    FROM causal_validations
    WHERE estimate_id = p_estimate_id;
    
    -- Default to 'review' if no validations exist
    RETURN COALESCE(v_gate, 'review'::gate_decision);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_validation_gate IS 'Get the aggregate gate decision for a causal estimate';


-- Function to get validation confidence for an estimate
CREATE OR REPLACE FUNCTION get_validation_confidence(p_estimate_id UUID)
RETURNS DECIMAL(4,3) AS $$
DECLARE
    v_confidence DECIMAL(4,3);
BEGIN
    SELECT AVG(confidence_score) INTO v_confidence
    FROM causal_validations
    WHERE estimate_id = p_estimate_id;
    
    RETURN COALESCE(v_confidence, 0.0);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_validation_confidence IS 'Get the average validation confidence for a causal estimate';


-- Function to check if estimate can be used (combines gate + expert approval)
CREATE OR REPLACE FUNCTION can_use_estimate(
    p_estimate_id UUID,
    p_dag_hash VARCHAR(64) DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    v_gate gate_decision;
    v_dag_approved BOOLEAN;
BEGIN
    -- Get validation gate
    v_gate := get_validation_gate(p_estimate_id);
    
    -- If blocked, always return false
    IF v_gate = 'block' THEN
        RETURN FALSE;
    END IF;
    
    -- If proceed, return true
    IF v_gate = 'proceed' THEN
        RETURN TRUE;
    END IF;
    
    -- If review, check if DAG is approved (if hash provided)
    IF p_dag_hash IS NOT NULL THEN
        v_dag_approved := is_dag_approved(p_dag_hash);
        RETURN v_dag_approved;
    END IF;
    
    -- Default: review needed, can't use without expert approval
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION can_use_estimate IS 'Check if a causal estimate can be used (combines validation gate and expert approval)';


-- ============================================================
-- 7. CREATE TRIGGERS
-- ============================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Only create trigger if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_cv_updated_at'
    ) THEN
        CREATE TRIGGER trg_cv_updated_at
            BEFORE UPDATE ON causal_validations
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_er_updated_at'
    ) THEN
        CREATE TRIGGER trg_er_updated_at
            BEFORE UPDATE ON expert_reviews
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;
END
$$;


-- ============================================================
-- 8. INSERT DEFAULT CHECKLIST TEMPLATE (Optional)
-- ============================================================

-- This creates a template checklist that can be copied for new reviews
-- Uncomment to use

/*
INSERT INTO expert_reviews (
    review_type,
    reviewer_id,
    reviewer_name,
    approval_status,
    checklist_json,
    analysis_context
) VALUES (
    'dag_approval',
    'SYSTEM_TEMPLATE',
    'System Template',
    'pending',
    '{
        "checklist_items": [
            {
                "id": "conf_complete", 
                "question": "Are all known confounders included?", 
                "required": true,
                "category": "Confounder Identification"
            },
            {
                "id": "edge_plausible", 
                "question": "Do causal arrows reflect domain knowledge?", 
                "required": true,
                "category": "Edge Direction"
            },
            {
                "id": "no_forbidden", 
                "question": "Are there no forbidden edges (future→past)?", 
                "required": true,
                "category": "Temporal Consistency"
            },
            {
                "id": "mediators_correct", 
                "question": "Are intermediate variables correctly positioned?", 
                "required": true,
                "category": "Mediation"
            },
            {
                "id": "iv_exogenous", 
                "question": "Are proposed instrumental variables genuinely exogenous?", 
                "required": false,
                "category": "Instruments"
            },
            {
                "id": "selection_bias", 
                "question": "Is selection bias considered?", 
                "required": false,
                "category": "Selection"
            },
            {
                "id": "sutva_plausible", 
                "question": "Is the no-interference assumption reasonable?", 
                "required": true,
                "category": "SUTVA"
            },
            {
                "id": "positivity", 
                "question": "Is there sufficient overlap in treatment groups?", 
                "required": true,
                "category": "Positivity"
            }
        ],
        "version": "1.0",
        "created_at": "2025-12-13"
    }'::jsonb,
    'Template checklist for DAG expert reviews - copy for new reviews'
);
*/

-- ============================================================
-- MIGRATION COMPLETE
-- ============================================================