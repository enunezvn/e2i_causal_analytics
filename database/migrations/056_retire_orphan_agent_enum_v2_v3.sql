-- ============================================================================
-- MIGRATION 056: Retire orphan agent enum types v2/v3 + dead shadow columns/functions
-- ============================================================================
-- Date: 2026-06-01
-- Issue: #607 (full agent-taxonomy reconciliation)
--
-- Background: migration 029 created agent_name_type_v3 + agent_tier_type_v2 + two
-- conversion functions (map_agent_v2_to_v3, map_tier_v1_to_v2) + shadow columns
-- (agent_registry.name_v3 / tier_v2) as a planned "rename to canonical" path that
-- NEVER executed. e2i_ml_complete_v3_schema.sql also created agent_name_type_v2.
-- VERIFIED (grep across src/ + tests/): none of these types/functions/columns are
-- read by any code -- the only reference is a docstring comment. The live agent_name
-- columns (agent_registries / agent_activities) are plain VARCHAR(50), so these enum
-- TYPES are pure orphans with no column or constraint depending on them.
--
-- The production agent roster source-of-truth is the code registry
-- (src/agents/factory.py AGENT_REGISTRY_CONFIG = 21 agents). We RETIRE the dead
-- v2/v3 enum taxonomy (model_evaluator/model_monitor/data_quality_monitor/
-- risk_assessor) rather than promote it, since no column ever adopted it.
--
-- Forward-only, idempotent. No script-level BEGIN/COMMIT: scripts/run_migrations.sh
-- runs psql with --single-transaction, which owns the outer transaction (issue #186).
--
-- NOTE: deploy skips migrations -> APPLY MANUALLY to the deployed Supabase, e.g.:
--   docker exec -i <supabase-db-container> psql -U postgres -d postgres \
--     < database/migrations/056_retire_orphan_agent_enum_v2_v3.sql
-- ============================================================================

-- 1-2: drop the unread shadow columns on agent_registry (created by mig 029).
ALTER TABLE IF EXISTS agent_registry DROP COLUMN IF EXISTS name_v3;
ALTER TABLE IF EXISTS agent_registry DROP COLUMN IF EXISTS tier_v2;

-- 3-4: drop the dead conversion functions (created by mig 029).
DROP FUNCTION IF EXISTS map_agent_v2_to_v3(TEXT);
DROP FUNCTION IF EXISTS map_tier_v1_to_v2(TEXT);

-- 5-7: drop the orphan enum types. Plain DROP TYPE IF EXISTS (NOT CASCADE): if a
-- future column unexpectedly adopts one of these, this FAILS LOUD instead of
-- silently dropping that column.
DROP TYPE IF EXISTS agent_name_type_v3;
DROP TYPE IF EXISTS agent_tier_type_v2;
DROP TYPE IF EXISTS agent_name_type_v2;

-- ============================================================================
-- VERIFICATION (run after migration):
--   SELECT typname FROM pg_type
--    WHERE typname IN ('agent_name_type_v2','agent_name_type_v3','agent_tier_type_v2');
--   -- expect 0 rows
--   SELECT proname FROM pg_proc WHERE proname IN ('map_agent_v2_to_v3','map_tier_v1_to_v2');
--   -- expect 0 rows
--   SELECT column_name FROM information_schema.columns
--    WHERE table_name='agent_registry' AND column_name IN ('name_v3','tier_v2');
--   -- expect 0 rows
-- ============================================================================
