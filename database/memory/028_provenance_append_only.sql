-- ============================================================================
-- E2I Provenance Append-Only Enforcement + LLM Prompt Audit Sink
-- Migration: 028_provenance_append_only.sql
-- Purpose: (#391 security box 1) Block UPDATE / DELETE on provenance records
--          to satisfy the tamper-evidence contract.
--          (#391 security box 4) Add ``crystal_narrative_audits`` table so
--          the offline PHI scanner can audit LLM INPUT prompts (not just
--          outputs) for PHI/PII leaks.
--
-- Background — Box 1
-- ------------------
-- Two surfaces carry provenance:
--   (a) ``audit_chain_entries`` (migration 011) — hash-linked audit trail.
--       The hash chain ALREADY makes any post-hoc mutation detectable, but
--       a DB-level trigger that REJECTS the write is stricter: it stops the
--       attempt at the boundary instead of detecting it after the fact.
--   (b) ``executive_insights.invalidated_at`` (migration 021) — set-once
--       lifecycle field driven by the cascade-invalidator. Once
--       ``invalidated_at IS NOT NULL`` the row is "frozen": any UPDATE
--       (including clearing ``invalidated_at`` back to NULL) MUST fail so
--       no admin script can resurrect an invalidated insight without a
--       fresh row (which is the documented re-crystallize path; the
--       partial-unique-index at 021:219-226 enforces uniqueness for
--       active rows only, so a fresh row can be inserted in its place).
--
-- Background — Box 4
-- ------------------
-- The crystallizer's ``LLMCrystalNarrativeAudit`` dataclass captures the
-- LLM OUTPUT (key_finding / limitations / recommended_next) + telemetry.
-- For the PHI audit harness to verify NO PHI leaks via the LLM path, we
-- also need the INPUT prompt persisted (otherwise the auditor can only
-- see what the LLM said, not what was asked). This migration adds the
-- ``crystal_narrative_audits`` table — one row per executive insight,
-- foreign-keyed via ``insight_id``. The PHI scanner script reads this
-- table's ``input_prompt`` column.
--
-- Scope
-- -----
-- This migration installs TWO triggers + ONE table:
--
--   1. ``trg_audit_chain_entries_append_only`` on ``audit_chain_entries``:
--      BEFORE UPDATE OR DELETE — UNCONDITIONALLY rejects.
--      Audit-chain semantics demand strict append-only.
--
--   2. ``trg_executive_insights_invalidation_set_once`` on
--      ``executive_insights``:
--      BEFORE UPDATE OR DELETE — rejects iff OLD.invalidated_at IS NOT NULL.
--      Rows that are still active (``invalidated_at IS NULL``) remain
--      mutable (recall/recall_reason updates flow through the normal
--      lifecycle write path). Only invalidated rows are frozen.
--
--   3. ``crystal_narrative_audits`` TABLE — persists the
--      :class:`LLMCrystalNarrativeAudit` payload, including the
--      ``input_prompt`` column the PHI scanner audits. One row per
--      ``insight_id`` (FK to executive_insights).
--
-- Idempotency
-- -----------
-- DROP TRIGGER IF EXISTS + CREATE TRIGGER + CREATE OR REPLACE FUNCTION +
-- CREATE TABLE IF NOT EXISTS make this safe to re-apply. Idempotency is
-- verified by ``tests/integration/test_028_provenance_append_only_migration.py``.
--
-- Reversibility
-- -------------
-- ``DROP TRIGGER … ON …; DROP FUNCTION …; DROP TABLE crystal_narrative_audits;``
-- reverses the migration. The triggers DO NOT touch row data — only block
-- writes — so removing them restores the prior write surface without data
-- loss. Dropping the audit table loses captured prompts (acceptable —
-- audit retention is an operator concern, not a forward-compat one).
-- ============================================================================

BEGIN;

-- ----------------------------------------------------------------------------
-- Trigger function: prevent_update_delete_audit_chain
-- ----------------------------------------------------------------------------
-- Unconditional rejection. Audit-chain rows are append-only by spec.
-- ----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION prevent_update_delete_audit_chain()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION
        'audit_chain_entries is append-only: % is not permitted (entry_id=%)',
        TG_OP, COALESCE(OLD.entry_id::TEXT, '<unknown>')
        USING ERRCODE = 'check_violation';
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION prevent_update_delete_audit_chain IS
'Append-only enforcement for audit_chain_entries (#391). Rejects UPDATE/DELETE unconditionally to preserve the hash-chain''s tamper-evidence property at the DB boundary.';

-- ----------------------------------------------------------------------------
-- Trigger: trg_audit_chain_entries_append_only
-- ----------------------------------------------------------------------------

DROP TRIGGER IF EXISTS trg_audit_chain_entries_append_only ON audit_chain_entries;

CREATE TRIGGER trg_audit_chain_entries_append_only
    BEFORE UPDATE OR DELETE ON audit_chain_entries
    FOR EACH ROW
    EXECUTE FUNCTION prevent_update_delete_audit_chain();

-- ----------------------------------------------------------------------------
-- Trigger function: prevent_change_invalidated_executive_insight
-- ----------------------------------------------------------------------------
-- Conditional: rejects iff the OLD row already had ``invalidated_at`` set.
-- Active rows (NULL invalidated_at) remain mutable through the normal
-- lifecycle path (recall, recall_reason, etc.).
-- ----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION prevent_change_invalidated_executive_insight()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.invalidated_at IS NOT NULL THEN
        RAISE EXCEPTION
            'executive_insights row is invalidated and frozen: % is not permitted (insight_id=%, invalidated_at=%)',
            TG_OP, OLD.insight_id, OLD.invalidated_at
            USING ERRCODE = 'check_violation';
    END IF;
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION prevent_change_invalidated_executive_insight IS
'Append-only enforcement for executive_insights rows that have already been invalidated (#391). Active rows (invalidated_at IS NULL) remain mutable; once invalidated, the row is frozen so admin scripts cannot resurrect an overturned crystal without a fresh re-crystallization.';

-- ----------------------------------------------------------------------------
-- Trigger: trg_executive_insights_invalidation_set_once
-- ----------------------------------------------------------------------------

DROP TRIGGER IF EXISTS trg_executive_insights_invalidation_set_once ON executive_insights;

CREATE TRIGGER trg_executive_insights_invalidation_set_once
    BEFORE UPDATE OR DELETE ON executive_insights
    FOR EACH ROW
    EXECUTE FUNCTION prevent_change_invalidated_executive_insight();

-- ----------------------------------------------------------------------------
-- Table: crystal_narrative_audits (#391 security box 4)
-- ----------------------------------------------------------------------------
-- Persists the LLMCrystalNarrativeAudit dataclass for offline PHI auditing.
-- One row per crystal (insight_id FK to executive_insights). The
-- ``input_prompt`` column is the FULL prompt sent to the LLM narrator —
-- the PHI scanner audit harness reads it via psycopg.
--
-- Cascade-delete: when the parent executive_insights row is removed (only
-- possible for ACTIVE rows per the trigger above — invalidated rows are
-- frozen), the audit row is removed too. This keeps the audit table from
-- accumulating orphan rows in dev / test workflows. Production retention
-- is operator-controlled via a separate scheduled job (not part of #391).
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS crystal_narrative_audits (
    audit_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    insight_id        UUID NOT NULL UNIQUE
                          REFERENCES executive_insights(insight_id)
                          ON DELETE CASCADE,
    narrator_model    VARCHAR(100) NOT NULL,
    -- LLM-generated output prose (audit copy of executive_insights cols).
    key_finding       TEXT NOT NULL DEFAULT '',
    limitations       TEXT NOT NULL DEFAULT '',
    recommended_next  TEXT NOT NULL DEFAULT '',
    -- The full prompt text sent to the LLM (#391 box 4). NOT NULL with
    -- empty default so flag-off / exception-path audits still persist
    -- with input_prompt='' (the empty string itself is a meaningful
    -- audit signal: "we did NOT send anything").
    input_prompt      TEXT NOT NULL DEFAULT '',
    -- Telemetry (mirrors LLMCrystalNarrativeAudit telemetry fields).
    latency_ms        DOUBLE PRECISION,
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    cost_usd          DOUBLE PRECISION,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_crystal_narrative_audits_insight_id
    ON crystal_narrative_audits(insight_id);

COMMENT ON TABLE crystal_narrative_audits IS
'Per-crystal LLM narrator audit payload. Captures the full input_prompt (#391 security box 4) so the offline PHI scanner harness can audit LLM inputs as well as outputs for PHI/PII leaks.';

COMMIT;
