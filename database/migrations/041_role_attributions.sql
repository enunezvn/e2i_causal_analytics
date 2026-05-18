-- ============================================================================
-- Migration 041: role_attributions columns on adaptive_validity_verdicts
--                (Phase 1 of Issue #237 causal-role propagation reframe)
-- ============================================================================
-- Plan reference: ``.claude/plans/causal_role_propagation_FINAL.md`` §1.7.
--
-- Background: Phase 1 of the causal-role propagation contract makes the
-- Layer-4 LLM ``causal_role`` (already persisted into adaptive_verdicts
-- via the sidecar) ALSO flow into a typed ``RoleAttribution`` list. Each
-- attribution carries a trust-source label so downstream Tier-2 agents
-- (causal_impact, heterogeneous_optimizer) can gate on verified
-- attributions only.
--
-- The producer (``src.data.role_attribution.derive_role_attributions``)
-- emits ``source ∈ {"manifest", "llm"}`` in Phase 1; KG attributions
-- arrive in Phase 6 via a separate enrichment node. The Supabase mirror
-- (``scripts/mirror_audit_sidecar_to_supabase.py``) persists per-row
-- attribution provenance to two new nullable columns on the existing
-- ``adaptive_validity_verdicts`` table:
--
--   - ``causal_role_final``: the resolved causal role
--     (ancestor | confounder | mediator | collider | descendant |
--     instrument). NULL for verdicts whose feature had no role
--     attribution (no manifest entry AND no LLM ``llm_role``).
--
--   - ``causal_role_source``: the trust label
--     (manifest | llm | kg). NULL when ``causal_role_final`` is NULL.
--
-- Idempotent design (ADD COLUMN IF NOT EXISTS + table-scoped DO $$ guard
-- on the CHECK constraint) mirrors the precedent set by
-- ``database/memory/021_insight_lifecycle.sql`` per the PR #250 close
-- memory entry. The migration is safe to apply repeatedly.
--
-- Non-natural-key: both columns are mutated by the mirror's
-- ``ON CONFLICT DO UPDATE`` set and are NOT part of the natural-key
-- uniqueness contract.
-- ----------------------------------------------------------------------------
-- Transaction control (matches 040):
-- ----------------------------------------------------------------------------
-- No script-level ``BEGIN;`` / ``COMMIT;``. ``scripts/run_migrations.sh``
-- invokes psql with ``--single-transaction`` and appends an INSERT INTO
-- schema_migrations after the file; an inner COMMIT would prematurely
-- commit before the bookkeeping insert.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Columns (additive, nullable, no default — Phase 1 producer writes
--    NULL when no attribution row exists for the verdict's feature).
-- ----------------------------------------------------------------------------
ALTER TABLE adaptive_validity_verdicts
    ADD COLUMN IF NOT EXISTS causal_role_final  TEXT,
    ADD COLUMN IF NOT EXISTS causal_role_source TEXT;

COMMENT ON COLUMN adaptive_validity_verdicts.causal_role_final IS
'Phase 1 of Issue #237 causal-role propagation. The resolved causal role for this verdict''s feature, one of {ancestor, confounder, mediator, collider, descendant, instrument}. NULL when the feature has no attribution row in role_attributions (no manifest entry and no Layer-4 LLM verdict). Populated by scripts/mirror_audit_sidecar_to_supabase.py from sidecar ``role_attributions`` (schema 1.1+).';

COMMENT ON COLUMN adaptive_validity_verdicts.causal_role_source IS
'Phase 1 of Issue #237 causal-role propagation. Trust label for causal_role_final: manifest (verification-grade, bypasses LLM-evaluator gate per C1), llm (gated on evaluator_audit.satisfied), or kg (Phase 6 enrichment). NULL when causal_role_final is NULL.';

-- ----------------------------------------------------------------------------
-- 2. CHECK constraint: causal_role_source must be one of the three
--    known trust labels when set. Guarded by DO $$ so a re-run does not
--    fail with duplicate_object.
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_adaptive_validity_verdicts_role_source'
          AND conrelid = 'adaptive_validity_verdicts'::regclass
    ) THEN
        ALTER TABLE adaptive_validity_verdicts
            ADD CONSTRAINT chk_adaptive_validity_verdicts_role_source
            CHECK (
                causal_role_source IS NULL
                OR causal_role_source IN ('manifest', 'llm', 'kg')
            );
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- 3. CHECK constraint: causal_role_final must be one of the six known
--    causal-role labels when set. Mirrors the
--    ``src.data.causal_role_classifier.CausalRole`` Literal.
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_adaptive_validity_verdicts_role_final'
          AND conrelid = 'adaptive_validity_verdicts'::regclass
    ) THEN
        ALTER TABLE adaptive_validity_verdicts
            ADD CONSTRAINT chk_adaptive_validity_verdicts_role_final
            CHECK (
                causal_role_final IS NULL
                OR causal_role_final IN (
                    'ancestor',
                    'confounder',
                    'mediator',
                    'collider',
                    'descendant',
                    'instrument'
                )
            );
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- 4. Co-presence integrity: source and final are populated together.
--    A row with one NULL and one set indicates a partial mirror write —
--    surface as a constraint violation rather than silently storing
--    ambiguous state.
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_adaptive_validity_verdicts_role_copresence'
          AND conrelid = 'adaptive_validity_verdicts'::regclass
    ) THEN
        ALTER TABLE adaptive_validity_verdicts
            ADD CONSTRAINT chk_adaptive_validity_verdicts_role_copresence
            CHECK (
                (causal_role_final IS NULL AND causal_role_source IS NULL)
                OR (causal_role_final IS NOT NULL AND causal_role_source IS NOT NULL)
            );
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- 5. Read-path index: cross-experiment queries by causal role
--    (e.g. "list all confounders with manifest provenance in the
--    last 30 days"). Partial index on the non-NULL subset keeps the
--    index small — most adaptive_verdicts rows are descendants/leakage
--    signals without an attribution.
-- ----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_adaptive_validity_verdicts_role_lookup
    ON adaptive_validity_verdicts (causal_role_final, causal_role_source, written_at DESC)
    WHERE causal_role_final IS NOT NULL;

-- (No COMMIT; psql --single-transaction owns the outer txn. See header.)
