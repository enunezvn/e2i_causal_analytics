-- ============================================================================
-- Migration 042: audit-evaluator shadow-mode columns on
--                adaptive_validity_verdicts (Stage 1 of Issue #240)
-- ============================================================================
-- Design reference: ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3
-- Stage 1 — Shadow mode (instrument; no behavior change).
--
-- Numbering note: the design doc names this migration "041", written
-- before ``041_role_attributions.sql`` (Issue #237 Phase 1) landed on
-- main. We bump to 042; the column names and intent are unchanged.
--
-- Background: Issue #240 promotes the Layer-4 Haiku evaluator (today
-- audit-only sidecar on ``LLMVerdict.evaluator_audit``) toward an
-- optional severity-modulating gate. Stage 1 is the data-collection
-- phase: every Layer-4 invocation that produces an audit also computes
-- what the future soft-gate WOULD do per each of three rules (R1/R2/R3
-- — see ``src/data/evaluator_promotion_rules.py`` and design §4), and
-- persists that decision into a per-row shadow column. The voter is
-- NOT modified at Stage 1; these columns are written for analytics
-- only.
--
-- Three columns, one per rule, are kept separate so each rule's firing
-- is independently observable (rather than packing into a single
-- composite key that hides per-rule firing rates).
--
-- ----------------------------------------------------------------------------
-- Column semantics (mirror the spec at design §3 Stage 1 + §4 R1/R2/R3).
-- ----------------------------------------------------------------------------
--
--   would_promote_severity  TEXT — set by R1 to the proposed severity
--       (currently always literal ``'high'`` when R1 fires; the column
--       is TEXT not an enum to leave room for future rules without a
--       constraint migration). NULL when R1 did not fire.
--
--   would_flag_for_review  BOOLEAN — set by R2 to ``TRUE`` when its
--       trigger fires (worker dissatisfied + ≥2 missed considerations).
--       NULL when R2 did not fire. R2 is intentionally not promoted
--       to Stage 3 in the design; it accelerates curation review only.
--
--   rationale_incomplete_flag  BOOLEAN — set by R3 to ``TRUE`` when
--       the evaluator's ``rationale_complete`` is ``False``. NULL
--       when R3 did not fire. R3 is deliberately never promoted; it
--       is a documentation-quality signal.
--
-- All three are nullable / no DEFAULT. NULL means "rule did not fire
-- for this row" OR "the row pre-existed migration 042 and was not
-- backfilled". The mirror script's ``ON CONFLICT DO UPDATE`` is
-- extended to write these columns; old rows stay NULL.
--
-- ----------------------------------------------------------------------------
-- Idempotency.
-- ----------------------------------------------------------------------------
-- ``ADD COLUMN IF NOT EXISTS`` is the standard idempotent pattern used
-- by migrations 040 and 041 in this directory. Safe to re-apply.
--
-- ----------------------------------------------------------------------------
-- Transaction control (matches 040 / 041).
-- ----------------------------------------------------------------------------
-- No script-level ``BEGIN;`` / ``COMMIT;``. ``scripts/run_migrations.sh``
-- invokes psql with ``--single-transaction`` and appends an
-- ``INSERT INTO schema_migrations`` after the file; an inner COMMIT
-- would prematurely commit before the bookkeeping insert.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Columns (additive, nullable, no default).
-- ----------------------------------------------------------------------------
ALTER TABLE adaptive_validity_verdicts
    ADD COLUMN IF NOT EXISTS would_promote_severity     TEXT,
    ADD COLUMN IF NOT EXISTS would_flag_for_review      BOOLEAN,
    ADD COLUMN IF NOT EXISTS rationale_incomplete_flag  BOOLEAN;

COMMENT ON COLUMN adaptive_validity_verdicts.would_promote_severity IS
'Stage 1 of Issue #240 audit-evaluator gate promotion. Set by promotion rule R1 (info→moderate escalation when evaluator dissatisfied AND ≥1 missed considerations; reframed 2026-05-25 from moderate→high per docs/plans/240-r1-reachability-investigation.md) to the proposed severity (currently always ''moderate''). NULL when R1 did not fire. Stage 1 is SHADOW MODE — the voter does NOT read this column. Populated by scripts/mirror_audit_sidecar_to_supabase.py from the sidecar ``would_promote_severity`` key (additive at sidecar schema 1.2+; absent on pre-#240 sidecars → column NULL).';

COMMENT ON COLUMN adaptive_validity_verdicts.would_flag_for_review IS
'Stage 1 of Issue #240 audit-evaluator gate promotion. Set by promotion rule R2 (≥2 missed considerations) to TRUE when its trigger fires. NULL when R2 did not fire. R2 accelerates curation review and is NOT promoted to Stage 3 by design.';

COMMENT ON COLUMN adaptive_validity_verdicts.rationale_incomplete_flag IS
'Stage 1 of Issue #240 audit-evaluator gate promotion. Set by promotion rule R3 (evaluator.rationale_complete=False) to TRUE when its trigger fires. NULL when R3 did not fire. R3 is a documentation-quality signal — deliberately never promoted to a runtime gate.';

-- ----------------------------------------------------------------------------
-- 2. CHECK constraint: would_promote_severity must be one of the
--    known severity labels when set. Mirrors EnsembleSeverity Literal
--    in src/data/kg/types.py. Guarded by DO $$ for idempotency.
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_adaptive_validity_verdicts_would_promote_severity'
          AND conrelid = 'adaptive_validity_verdicts'::regclass
    ) THEN
        ALTER TABLE adaptive_validity_verdicts
            ADD CONSTRAINT chk_adaptive_validity_verdicts_would_promote_severity
            CHECK (
                would_promote_severity IS NULL
                OR would_promote_severity IN ('high', 'moderate', 'info', 'abstain')
            );
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- 3. Rollback (operator runbook).
-- ----------------------------------------------------------------------------
-- The columns are additive and nullable; the safest rollback is to leave
-- them in place (any future Stage 2/3 work depends on the column shape).
-- If the columns must be physically removed:
--
--   ALTER TABLE adaptive_validity_verdicts
--       DROP CONSTRAINT IF EXISTS chk_adaptive_validity_verdicts_would_promote_severity,
--       DROP COLUMN IF EXISTS rationale_incomplete_flag,
--       DROP COLUMN IF EXISTS would_flag_for_review,
--       DROP COLUMN IF EXISTS would_promote_severity;
--
-- (Not executed as part of this migration. Rollback is operator-driven.)

-- (No COMMIT; psql --single-transaction owns the outer txn. See header.)
