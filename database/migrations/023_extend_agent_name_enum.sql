-- ============================================================================
-- MIGRATION 023: Extend agent_name_enum with Tier 1-5 agent names
-- ============================================================================
-- Date: 2026-04-25
-- Purpose: Resolve runtime "invalid input value for enum agent_name_enum"
--          errors when ml_observability_spans is written with non-Tier-0
--          agent names (orchestrator, causal_impact, etc.).
--
-- Background:
-- database/ml/mlops_tables.sql:9-16 only ADDed Tier-0 agent names to
-- agent_name_enum. The base enum was created by an earlier (now-replaced)
-- version of that file with only legacy values, so Tier 1-5 names raise
-- 22P02 when inserted. This migration brings agent_name_enum into parity
-- with the Pydantic AgentNameEnum in
-- src/agents/ml_foundation/observability_connector/models.py.
--
-- ALTER TYPE … ADD VALUE IF NOT EXISTS is idempotent and safe to re-run.
-- Note: in Postgres < 12 these statements cannot run inside a transaction;
-- scripts/run_migration.py uses autocommit which handles this correctly.
-- ============================================================================

-- Tier 1: Coordination
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'orchestrator';
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'tool_composer';

-- Tier 2: Causal Analytics
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'causal_impact';
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'gap_analyzer';
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'heterogeneous_optimizer';

-- Tier 3: Monitoring & Experimentation
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'drift_monitor';
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'experiment_designer';
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'health_score';

-- Tier 4: ML & Predictions
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'prediction_synthesizer';
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'resource_optimizer';

-- Tier 5: Self-Improvement
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'explainer';
ALTER TYPE agent_name_enum ADD VALUE IF NOT EXISTS 'feedback_learner';

-- ============================================================================
-- VERIFICATION (run after migration):
--   SELECT enumlabel FROM pg_enum
--   WHERE enumtypid = 'agent_name_enum'::regtype
--   ORDER BY enumsortorder;
--
-- Expected: 19 values (7 Tier-0 + 12 added here).
-- ============================================================================
