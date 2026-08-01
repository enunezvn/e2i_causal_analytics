-- ============================================================================
-- Migration 121: Reconcile validation_outcomes with the ValidationOutcome contract
-- ============================================================================
-- Purpose: The Supabase insert path
--          (SupabaseValidationOutcomeStore._outcome_to_row) serializes a
--          ValidationOutcome as a flat row and inserts it directly. The live
--          table (created by migration 007) lacks EIGHT of those columns, and
--          its outcome_type CHECK constraint rejects the values the
--          ValidationOutcomeType enum actually produces. Every chat-path
--          refutation insert therefore failed at PostgREST with PGRST204
--          ("could not find the 'agent_context' column") and the store degraded
--          to the ephemeral in-memory fallback — silently dropping the
--          Feedback-Learner signal on every refutation run (#1423).
--
--          Note: migration 021 declared agent_context, but the migration ledger
--          recorded 021 as applied while the column never reached the live
--          table (schema/ledger drift). This migration re-adds it idempotently
--          (ADD COLUMN IF NOT EXISTS) alongside the other seven so the live
--          schema is repaired regardless of the ledger state.
--
-- Reference: #1423 — first surfaced live on 2026-08-01 when the #1419 refutation
--            path became the first real producer of validation outcomes.
-- Safety:    validation_outcomes is empty (0 rows) at authoring time, so
--            replacing the outcome_type CHECK constraint cannot invalidate any
--            existing data.
-- ============================================================================

-- 1. Add the columns the serializer writes but the 007 schema never had.
ALTER TABLE validation_outcomes
    ADD COLUMN IF NOT EXISTS gate_decision    TEXT,
    ADD COLUMN IF NOT EXISTS confidence_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS tests_passed     INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS tests_failed     INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS tests_total      INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS raw_suite        JSONB DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS agent_context    JSONB DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS dag_hash         TEXT;

COMMENT ON COLUMN validation_outcomes.gate_decision IS
    'Gate decision from the refutation suite (e.g. proceed / block / review).';
COMMENT ON COLUMN validation_outcomes.confidence_score IS
    'Overall confidence score for the validation outcome (0.0-1.0).';
COMMENT ON COLUMN validation_outcomes.tests_passed IS
    'Count of refutation tests that passed.';
COMMENT ON COLUMN validation_outcomes.tests_failed IS
    'Count of refutation tests that failed.';
COMMENT ON COLUMN validation_outcomes.tests_total IS
    'Total count of refutation tests run.';
COMMENT ON COLUMN validation_outcomes.raw_suite IS
    'Raw RefutationSuite output payload for provenance/debugging.';
COMMENT ON COLUMN validation_outcomes.agent_context IS
    'Agent execution context at the time of validation (agent name, query, parameters).';
COMMENT ON COLUMN validation_outcomes.dag_hash IS
    'Hash of the causal DAG the validation was run against.';

-- 2. Reconcile the outcome_type CHECK constraint with the ValidationOutcomeType
--    enum. The application only ever produces these five values
--    (src/causal_engine/validation_outcome.py); the original 007 set
--    (failed_refutation / failed_sensitivity / failed_placebo / partial_pass /
--    inconclusive) is produced nowhere in the code and blocked every non-passed
--    outcome (e.g. the E-value BLOCKED verdict from the first live suite).
ALTER TABLE validation_outcomes
    DROP CONSTRAINT IF EXISTS validation_outcomes_outcome_type_check;

ALTER TABLE validation_outcomes
    ADD CONSTRAINT validation_outcomes_outcome_type_check
    CHECK (
        outcome_type IN (
            'passed',
            'failed_critical',
            'failed_multiple',
            'needs_review',
            'blocked'
        )
    );
