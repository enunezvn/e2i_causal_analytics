-- ============================================================================
-- MIGRATION 055: Add cohort_constructor + experiment_monitor to agent_name_enum
-- ============================================================================
-- Date: 2026-06-01
-- Issue: #607 (follow-up to #601)
--
-- Purpose: Bring the observability agent_name_enum (ml_observability_spans) into
--   parity with the Pydantic AgentNameEnum in
--   src/agents/ml_foundation/observability_connector/models.py, which now lists
--   all 21 agents. Migration 023 added the original Tier 1-5 set but predates
--   cohort_constructor (Tier 0) and experiment_monitor (Tier 3); without these,
--   an observability span tagged with either name raises 22P02 (today masked by
--   a Pydantic-layer coerce-to-ORCHESTRATOR, i.e. silent misattribution).
--
-- ALTER TYPE ... ADD VALUE IF NOT EXISTS is idempotent and safe to re-run.
-- In Postgres < 12 these cannot run inside a transaction; the migration runner
-- uses autocommit (see scripts/run_migration.py), consistent with migration 023.
-- ============================================================================

-- Tier 0: ML Foundation
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'cohort_constructor';

-- Tier 3: Monitoring & Experimentation
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'experiment_monitor';

-- ============================================================================
-- VERIFICATION (run after migration):
--   SELECT enumlabel FROM pg_enum e
--     JOIN pg_type t ON e.enumtypid = t.oid
--    WHERE t.typname = 'agent_name_enum'
--    ORDER BY e.enumsortorder;
-- ============================================================================
