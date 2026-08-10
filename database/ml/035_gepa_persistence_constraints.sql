-- ============================================================================
-- E2I Causal Analytics - Migration 035: GEPA Persistence Constraint Fixes
-- ============================================================================
-- Date: August 2026
-- Description: Constraint fixes required to WIRE the migration-023 GEPA
-- tables (src/repositories/prompt_optimization.py). All three defects were
-- verified against the live schema before this migration was written, and
-- each is pinned by a test in
-- tests/integration/test_gepa_persistence_realdb.py.
--
-- Idempotent: safe to re-run.
-- ============================================================================

-- 1) unique_active_instruction was UNIQUE NULLS NOT DISTINCT
--    (agent_name, predictor_name, is_active). A boolean column has three
--    distinct states, so this caps history at ONE inactive row per
--    predictor — versioned instruction history (the table's stated purpose)
--    is impossible under it. Replace with a partial unique index enforcing
--    what the 023 comment actually intended: one ACTIVE row per predictor.
ALTER TABLE optimized_instructions DROP CONSTRAINT IF EXISTS unique_active_instruction;
CREATE UNIQUE INDEX IF NOT EXISTS uq_opt_instructions_one_active
    ON optimized_instructions (agent_name, predictor_name)
    WHERE is_active;

-- 2) idx_opt_instructions_hash was GLOBALLY unique on instruction_hash.
--    Two agents routinely produce identical instruction text (dspy's default
--    signature instruction is shared boilerplate; unoptimized modules can
--    carry empty instructions), so the global scope makes the second agent's
--    insert fail. Scope the dedup to the predictor it belongs to.
DROP INDEX IF EXISTS idx_opt_instructions_hash;
CREATE UNIQUE INDEX IF NOT EXISTS uq_opt_instructions_predictor_hash
    ON optimized_instructions (agent_name, predictor_name, instruction_hash);

-- 3) version VARCHAR(50) is smaller than real version ids produced by
--    src/optimization/gepa/versioning.py:
--    'gepa_v1_feedback_learner_recommendation_20260810_133055' is 55 chars.
--    v_active_instructions (023 Section 7) depends on the column, so it is
--    dropped and recreated (definition unchanged from 023) around the ALTER.
DROP VIEW IF EXISTS v_active_instructions;
ALTER TABLE optimized_instructions ALTER COLUMN version TYPE VARCHAR(100);
ALTER TABLE optimized_tool_descriptions ALTER COLUMN version TYPE VARCHAR(100);

CREATE OR REPLACE VIEW v_active_instructions AS
SELECT
    oi.agent_name,
    oi.predictor_name,
    oi.version,
    oi.instruction_text,
    oi.val_score,
    por.optimizer_type,
    por.run_name,
    por.improvement_percent,
    oi.activated_at
FROM optimized_instructions oi
JOIN prompt_optimization_runs por ON oi.run_id = por.run_id
WHERE oi.is_active = TRUE
ORDER BY oi.agent_name, oi.predictor_name;

-- 4) Same one-active defect on tool descriptions as (1).
ALTER TABLE optimized_tool_descriptions DROP CONSTRAINT IF EXISTS unique_active_tool_desc;
CREATE UNIQUE INDEX IF NOT EXISTS uq_tool_desc_one_active
    ON optimized_tool_descriptions (agent_name, tool_name)
    WHERE is_active;

-- ============================================================================
-- MIGRATION COMPLETE
-- Rollback: recreate the 023 constraints/index and re-narrow version to
-- VARCHAR(50) (only safe while the tables hold no >50-char versions).
-- ============================================================================
