-- ============================================================================
-- Migration 021: Add agent_context Column to validation_outcomes
-- ============================================================================
-- Purpose: Add agent_context JSONB column used by
--          src/causal_engine/validation_outcome_store.py:374
-- Reference: run_tier1_5_test.py error output
-- ============================================================================

ALTER TABLE validation_outcomes
ADD COLUMN IF NOT EXISTS agent_context JSONB DEFAULT '{}';

COMMENT ON COLUMN validation_outcomes.agent_context IS
    'Agent execution context at the time of validation (agent name, query, parameters)';
