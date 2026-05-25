-- ============================================================================
-- Migration 043: audit-evaluator soft-gate columns on
--                adaptive_validity_verdicts (Stage 3 of Issue #240)
-- ============================================================================
-- Design reference: ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3
-- Stage 3 — Soft-gate severity modulation (limited, reversible, fail-open)
-- and §5 R-4 (audit-loop-coupling mitigation).
--
-- Numbering note: the design doc names this migration "042" (it was written
-- before ``041_role_attributions.sql`` and ``042_audit_evaluator_shadow_columns.sql``
-- landed on main). The latest on-disk migration is 042; we bump to 043. The
-- column names and intent are unchanged from the design.
--
-- Background: Issue #240 Stage 3 ships the MECHANISM for an env-gated
-- (``ADAPTIVE_VALIDITY_EVALUATOR_GATE_ENABLED``, default OFF), fail-open,
-- reversible severity gate. When (and only when) an operator enables the
-- flag AND promotion rule R1 fires on a moderate candidate, the voter
-- substitutes ``severity=high`` (remediation follows deterministically).
-- Two audit columns record what happened:
--
--   gate_rule_fired  TEXT — the promotion rule that actually modulated the
--       verdict's severity inside the voter (only literal ``'R1'`` today;
--       TEXT not an enum to leave room for future rules without a
--       constraint migration). NULL when the gate was disabled (the
--       default) OR enabled but did not flip the decision (fail-open:
--       candidate severity != moderate, evaluator audit absent, or
--       evaluator errored).
--
--   worker_severity_pre_gate  TEXT — the un-mutated worker severity
--       recorded BEFORE the voter substituted (design §5 R-4 audit-loop-
--       coupling mitigation). Compile-set curation reads THIS column so it
--       never trains the worker on a gate-escalated label. For R1 this is
--       always ``'info'`` (R1's only transition is info→moderate, reframed
--       2026-05-25 — see docs/plans/240-r1-reachability-investigation.md);
--       NULL when no gate fired (then ``verdict.severity`` already IS the
--       worker severity).
--
-- Both are nullable / no DEFAULT. NULL means "gate did not fire for this
-- row" OR "the row pre-existed migration 043 and was not backfilled". The
-- mirror script's ``ON CONFLICT DO UPDATE`` is extended to write these
-- columns; old rows stay NULL.
--
-- Stage 3 is the MECHANISM only: this migration does NOT enable the gate.
-- ACTIVATION (flag=1 in production) remains HARD-gated on #242 (multi-vendor
-- evaluator) + Stage-1/2 empirical data + stakeholder sign-off (design AC3.5).
--
-- AC3.4 rollback query (operator runbook):
--   SELECT feature, severity, gate_rule_fired, worker_severity_pre_gate
--   FROM adaptive_validity_verdicts
--   WHERE gate_rule_fired IS NOT NULL
--   ORDER BY written_at DESC LIMIT 100;
--
-- ----------------------------------------------------------------------------
-- Idempotency.
-- ----------------------------------------------------------------------------
-- ``ADD COLUMN IF NOT EXISTS`` is the standard idempotent pattern used by
-- migrations 040 / 041 / 042 in this directory. Safe to re-apply.
--
-- ----------------------------------------------------------------------------
-- Transaction control (matches 040 / 041 / 042).
-- ----------------------------------------------------------------------------
-- No script-level ``BEGIN;`` / ``COMMIT;``. ``scripts/run_migrations.sh``
-- invokes psql with ``--single-transaction`` and appends an
-- ``INSERT INTO schema_migrations`` after the file; an inner COMMIT would
-- prematurely commit before the bookkeeping insert.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Columns (additive, nullable, no default).
-- ----------------------------------------------------------------------------
ALTER TABLE adaptive_validity_verdicts
    ADD COLUMN IF NOT EXISTS gate_rule_fired           TEXT,
    ADD COLUMN IF NOT EXISTS worker_severity_pre_gate  TEXT;

COMMENT ON COLUMN adaptive_validity_verdicts.gate_rule_fired IS
'Stage 3 of Issue #240 audit-evaluator gate promotion. Names the promotion rule that actually modulated this verdict''s severity inside EnsembleVoter.vote (only ''R1'' today). NULL when the env-gated soft-gate (ADAPTIVE_VALIDITY_EVALUATOR_GATE_ENABLED, default OFF) was disabled OR enabled but did not flip the decision (fail-open). Populated by scripts/mirror_audit_sidecar_to_supabase.py from the sidecar ''gate_rule_fired'' key (additive at sidecar schema 1.3+; absent on pre-Stage-3 sidecars → column NULL).';

COMMENT ON COLUMN adaptive_validity_verdicts.worker_severity_pre_gate IS
'Stage 3 of Issue #240 (design §5 R-4 audit-loop-coupling mitigation). The un-mutated worker severity recorded BEFORE the voter substituted (always ''moderate'' for R1; NULL when no gate fired). Compile-set curation reads THIS column so it never trains the worker on a gate-escalated label.';

-- ----------------------------------------------------------------------------
-- 2. CHECK constraints: both columns hold a known label when set.
--    gate_rule_fired ∈ {R1}; worker_severity_pre_gate mirrors the
--    EnsembleSeverity Literal in src/data/kg/types.py. Guarded by DO $$
--    for idempotency (matches 042).
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_adaptive_validity_verdicts_gate_rule_fired'
          AND conrelid = 'adaptive_validity_verdicts'::regclass
    ) THEN
        ALTER TABLE adaptive_validity_verdicts
            ADD CONSTRAINT chk_adaptive_validity_verdicts_gate_rule_fired
            CHECK (
                gate_rule_fired IS NULL
                OR gate_rule_fired IN ('R1')
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_adaptive_validity_verdicts_worker_severity_pre_gate'
          AND conrelid = 'adaptive_validity_verdicts'::regclass
    ) THEN
        ALTER TABLE adaptive_validity_verdicts
            ADD CONSTRAINT chk_adaptive_validity_verdicts_worker_severity_pre_gate
            CHECK (
                worker_severity_pre_gate IS NULL
                OR worker_severity_pre_gate IN ('high', 'moderate', 'info', 'abstain')
            );
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- 3. Rollback (operator runbook).
-- ----------------------------------------------------------------------------
-- The columns are additive and nullable; the safest rollback is to leave
-- them in place (the AC3.4 rollback query depends on them to identify
-- gate-flipped rows). If the columns must be physically removed:
--
--   ALTER TABLE adaptive_validity_verdicts
--       DROP CONSTRAINT IF EXISTS chk_adaptive_validity_verdicts_worker_severity_pre_gate,
--       DROP CONSTRAINT IF EXISTS chk_adaptive_validity_verdicts_gate_rule_fired,
--       DROP COLUMN IF EXISTS worker_severity_pre_gate,
--       DROP COLUMN IF EXISTS gate_rule_fired;
--
-- (Not executed as part of this migration. Rollback is operator-driven.)

-- (No COMMIT; psql --single-transaction owns the outer txn. See header.)
