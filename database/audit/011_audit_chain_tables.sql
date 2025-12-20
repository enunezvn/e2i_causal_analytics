-- ============================================================================
-- E2I Causal Analytics - Audit Chain Tables
-- Migration: 011_audit_chain_tables.sql
-- Purpose: Tamper-evident agent action logging with hash-linked chains
-- Version: 4.1
-- Date: December 2025
-- ============================================================================

-- Enable pgcrypto for SHA-256 hashing
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================================
-- Table: audit_chain_entries
-- Core hash-linked audit trail for agent actions
-- ============================================================================
CREATE TABLE audit_chain_entries (
    -- Primary identification
    entry_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id         UUID NOT NULL,            -- Groups entries by workflow execution
    sequence_number     INTEGER NOT NULL,         -- Order within workflow (1, 2, 3...)
    
    -- Agent identification (matches domain_vocabulary.yaml)
    agent_name          VARCHAR(50) NOT NULL,     -- e.g., 'causal_impact', 'orchestrator'
    agent_tier          INTEGER NOT NULL,         -- 0-5 tier classification
    action_type         VARCHAR(50) NOT NULL,     -- e.g., 'estimation', 'refutation', 'routing'
    
    -- Timing
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    duration_ms         INTEGER,                  -- Action execution time
    
    -- Action payload (hashed for chain integrity)
    input_hash          VARCHAR(64),              -- SHA-256 of input data
    output_hash         VARCHAR(64),              -- SHA-256 of output data
    
    -- Validation results (for causal agents - Tier 2)
    validation_passed   BOOLEAN,
    confidence_score    NUMERIC(5,4),             -- 0.0000 to 1.0000
    refutation_results  JSONB,                    -- DoWhy refutation test outcomes
    
    -- Hash chain fields (THE KEY INNOVATION)
    previous_entry_id   UUID,                     -- Links to prior entry (NULL for genesis)
    previous_hash       VARCHAR(64),              -- Hash of previous entry (NULL for genesis)
    entry_hash          VARCHAR(64) NOT NULL,     -- This entry's hash (computed on insert)
    
    -- Metadata for regulatory queries
    user_id             VARCHAR(100),             -- Who triggered the workflow
    session_id          UUID,                     -- User session reference
    query_text          TEXT,                     -- Original user query (if applicable)
    brand               VARCHAR(50),              -- Remibrutinib, Fabhalta, Kisqali
    
    -- Constraints
    CONSTRAINT unique_workflow_sequence UNIQUE (workflow_id, sequence_number),
    CONSTRAINT fk_previous_entry FOREIGN KEY (previous_entry_id) 
        REFERENCES audit_chain_entries(entry_id),
    CONSTRAINT valid_agent_tier CHECK (agent_tier >= 0 AND agent_tier <= 5),
    CONSTRAINT valid_confidence CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1))
);

-- ============================================================================
-- Indexes for common query patterns
-- ============================================================================
CREATE INDEX idx_audit_chain_workflow ON audit_chain_entries(workflow_id, sequence_number);
CREATE INDEX idx_audit_chain_agent ON audit_chain_entries(agent_name, created_at);
CREATE INDEX idx_audit_chain_created ON audit_chain_entries(created_at);
CREATE INDEX idx_audit_chain_brand ON audit_chain_entries(brand, created_at);
CREATE INDEX idx_audit_chain_validation ON audit_chain_entries(validation_passed, created_at);
CREATE INDEX idx_audit_chain_tier ON audit_chain_entries(agent_tier);

-- ============================================================================
-- Table: audit_chain_verification_log
-- Records when chain integrity was verified (for auditors)
-- ============================================================================
CREATE TABLE audit_chain_verification_log (
    verification_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verified_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    workflow_id         UUID,                     -- NULL = full chain verification
    entries_verified    INTEGER NOT NULL,
    chain_valid         BOOLEAN NOT NULL,
    first_broken_entry  UUID,                     -- If invalid, where did it break?
    verified_by         VARCHAR(100),             -- User/system that ran verification
    verification_method VARCHAR(50) DEFAULT 'sha256',
    verification_notes  TEXT
);

CREATE INDEX idx_verification_log_workflow ON audit_chain_verification_log(workflow_id);
CREATE INDEX idx_verification_log_date ON audit_chain_verification_log(verified_at);
CREATE INDEX idx_verification_log_valid ON audit_chain_verification_log(chain_valid);

-- ============================================================================
-- Function: compute_entry_hash
-- Generates SHA-256 hash of entry contents for chain linking
-- ============================================================================
CREATE OR REPLACE FUNCTION compute_entry_hash(
    p_entry_id UUID,
    p_workflow_id UUID,
    p_sequence_number INTEGER,
    p_agent_name VARCHAR,
    p_action_type VARCHAR,
    p_created_at TIMESTAMPTZ,
    p_input_hash VARCHAR,
    p_output_hash VARCHAR,
    p_previous_hash VARCHAR
) RETURNS VARCHAR(64) AS $$
BEGIN
    RETURN encode(
        digest(
            concat(
                p_entry_id::text,
                p_workflow_id::text,
                p_sequence_number::text,
                p_agent_name,
                p_action_type,
                p_created_at::text,
                COALESCE(p_input_hash, ''),
                COALESCE(p_output_hash, ''),
                COALESCE(p_previous_hash, 'GENESIS')
            ),
            'sha256'
        ),
        'hex'
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION compute_entry_hash IS 
'Computes SHA-256 hash for audit chain entry. Used for chain linking and verification.';

-- ============================================================================
-- Function: verify_chain_integrity
-- Validates that all hashes in a workflow chain are correct
-- ============================================================================
CREATE OR REPLACE FUNCTION verify_chain_integrity(p_workflow_id UUID)
RETURNS TABLE (
    is_valid BOOLEAN,
    entries_checked INTEGER,
    first_invalid_entry UUID,
    error_message TEXT
) AS $$
DECLARE
    v_entry RECORD;
    v_expected_hash VARCHAR(64);
    v_count INTEGER := 0;
    v_prev_hash VARCHAR(64) := NULL;
BEGIN
    FOR v_entry IN
        SELECT * FROM audit_chain_entries
        WHERE workflow_id = p_workflow_id
        ORDER BY sequence_number
    LOOP
        v_count := v_count + 1;
        
        -- Verify previous_hash matches actual previous entry's hash
        IF v_entry.sequence_number > 1 AND v_entry.previous_hash != v_prev_hash THEN
            is_valid := FALSE;
            entries_checked := v_count;
            first_invalid_entry := v_entry.entry_id;
            error_message := 'Previous hash mismatch at sequence ' || v_entry.sequence_number;
            RETURN NEXT;
            RETURN;
        END IF;
        
        -- Recompute the hash
        v_expected_hash := compute_entry_hash(
            v_entry.entry_id,
            v_entry.workflow_id,
            v_entry.sequence_number,
            v_entry.agent_name,
            v_entry.action_type,
            v_entry.created_at,
            v_entry.input_hash,
            v_entry.output_hash,
            v_entry.previous_hash
        );
        
        -- Compare with stored hash
        IF v_entry.entry_hash != v_expected_hash THEN
            is_valid := FALSE;
            entries_checked := v_count;
            first_invalid_entry := v_entry.entry_id;
            error_message := 'Entry hash mismatch at sequence ' || v_entry.sequence_number;
            RETURN NEXT;
            RETURN;
        END IF;
        
        -- Save this entry's hash for next iteration
        v_prev_hash := v_entry.entry_hash;
    END LOOP;
    
    -- All entries valid
    is_valid := TRUE;
    entries_checked := v_count;
    first_invalid_entry := NULL;
    error_message := NULL;
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION verify_chain_integrity IS 
'Verifies the cryptographic integrity of a workflow audit chain. Returns validation result.';

-- ============================================================================
-- Function: verify_all_chains
-- Bulk verification of all chains in a date range
-- ============================================================================
CREATE OR REPLACE FUNCTION verify_all_chains(
    p_start_date TIMESTAMPTZ DEFAULT NULL,
    p_end_date TIMESTAMPTZ DEFAULT NULL
)
RETURNS TABLE (
    workflow_id UUID,
    is_valid BOOLEAN,
    entries_checked INTEGER,
    workflow_start TIMESTAMPTZ,
    brand VARCHAR
) AS $$
DECLARE
    v_workflow RECORD;
    v_result RECORD;
BEGIN
    FOR v_workflow IN
        SELECT DISTINCT 
            ace.workflow_id,
            MIN(ace.created_at) as workflow_start,
            MAX(ace.brand) as brand
        FROM audit_chain_entries ace
        WHERE (p_start_date IS NULL OR ace.created_at >= p_start_date)
          AND (p_end_date IS NULL OR ace.created_at <= p_end_date)
        GROUP BY ace.workflow_id
    LOOP
        SELECT * INTO v_result FROM verify_chain_integrity(v_workflow.workflow_id);
        
        workflow_id := v_workflow.workflow_id;
        is_valid := v_result.is_valid;
        entries_checked := v_result.entries_checked;
        workflow_start := v_workflow.workflow_start;
        brand := v_workflow.brand;
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- View: v_audit_chain_summary
-- Quick overview of workflows with chain status
-- ============================================================================
CREATE VIEW v_audit_chain_summary AS
SELECT
    workflow_id,
    MIN(created_at) AS workflow_start,
    MAX(created_at) AS workflow_end,
    COUNT(*) AS total_entries,
    COUNT(DISTINCT agent_name) AS agents_involved,
    array_agg(DISTINCT agent_name ORDER BY agent_name) AS agent_list,
    BOOL_AND(validation_passed) AS all_validations_passed,
    AVG(confidence_score)::NUMERIC(5,4) AS avg_confidence,
    SUM(duration_ms) AS total_duration_ms,
    MAX(brand) AS brand,
    MAX(user_id) AS user_id,
    MAX(query_text) AS query_text
FROM audit_chain_entries
GROUP BY workflow_id;

COMMENT ON VIEW v_audit_chain_summary IS 
'Aggregated view of workflow audit chains for quick status overview.';

-- ============================================================================
-- View: v_causal_validation_chain
-- Specialized view for Tier 2 causal agent validation chains
-- ============================================================================
CREATE VIEW v_causal_validation_chain AS
SELECT
    ace.workflow_id,
    ace.sequence_number,
    ace.agent_name,
    ace.action_type,
    ace.validation_passed,
    ace.confidence_score,
    ace.refutation_results->>'placebo_treatment' AS placebo_passed,
    ace.refutation_results->>'random_common_cause' AS random_cause_passed,
    ace.refutation_results->>'data_subset' AS subset_passed,
    ace.refutation_results->>'unobserved_confound' AS confound_passed,
    ace.entry_hash,
    ace.previous_hash,
    ace.created_at,
    ace.duration_ms,
    ace.brand
FROM audit_chain_entries ace
WHERE ace.agent_tier = 2  -- Tier 2: Causal Analytics
ORDER BY ace.workflow_id, ace.sequence_number;

COMMENT ON VIEW v_causal_validation_chain IS 
'Detailed view of Tier 2 causal agent validation results with refutation test outcomes.';

-- ============================================================================
-- View: v_audit_chain_daily_stats
-- Daily statistics for monitoring and dashboards
-- ============================================================================
CREATE VIEW v_audit_chain_daily_stats AS
SELECT
    DATE(created_at) AS date,
    COUNT(DISTINCT workflow_id) AS workflows_count,
    COUNT(*) AS total_entries,
    SUM(CASE WHEN validation_passed = true THEN 1 ELSE 0 END) AS validations_passed,
    SUM(CASE WHEN validation_passed = false THEN 1 ELSE 0 END) AS validations_failed,
    AVG(confidence_score)::NUMERIC(5,4) AS avg_confidence,
    AVG(duration_ms)::INTEGER AS avg_duration_ms,
    COUNT(DISTINCT agent_name) AS unique_agents,
    COUNT(DISTINCT brand) AS brands_analyzed
FROM audit_chain_entries
GROUP BY DATE(created_at)
ORDER BY DATE(created_at) DESC;

-- ============================================================================
-- Permissions
-- ============================================================================

-- [ADDED FIX]: Create roles if they do not exist
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'e2i_readonly') THEN
    CREATE ROLE e2i_readonly;
  END IF;
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'e2i_service') THEN
    CREATE ROLE e2i_service;
  END IF;
END
$$;

-- Read-only access for auditors and reporting
GRANT SELECT ON audit_chain_entries TO e2i_readonly;
GRANT SELECT ON audit_chain_verification_log TO e2i_readonly;
GRANT SELECT ON v_audit_chain_summary TO e2i_readonly;
GRANT SELECT ON v_causal_validation_chain TO e2i_readonly;
GRANT SELECT ON v_audit_chain_daily_stats TO e2i_readonly;

-- Service account for writing entries
GRANT SELECT, INSERT ON audit_chain_entries TO e2i_service;
GRANT SELECT, INSERT ON audit_chain_verification_log TO e2i_service;

-- Function execution
GRANT EXECUTE ON FUNCTION compute_entry_hash TO e2i_readonly, e2i_service;
GRANT EXECUTE ON FUNCTION verify_chain_integrity TO e2i_readonly, e2i_service;
GRANT EXECUTE ON FUNCTION verify_all_chains TO e2i_readonly, e2i_service;

-- ============================================================================
-- Migration metadata
-- ============================================================================
COMMENT ON TABLE audit_chain_entries IS 
'Hash-linked audit trail for agent actions. Each entry references the hash of the previous entry, creating tamper-evident chains. Part of E2I V4.1 regulatory compliance infrastructure.';

COMMENT ON TABLE audit_chain_verification_log IS 
'Log of chain integrity verifications for audit purposes.';