-- ============================================================================
-- MIGRATION 029 (memory): Add experiment_monitor + cohort_constructor to e2i_agent_name
-- ============================================================================
-- Date: 2026-06-01
-- Issue: #607 (follow-up to #601)
--
-- Problem (LIVE silent data loss):
--   experiment_monitor (Tier 3) and cohort_constructor (Tier 0) are real, dispatched
--   agents in AGENT_METHOD_MAP, but were never added to the e2i_agent_name enum.
--   Migration 018 added the other 7 Tier-0 agents + tool_composer but OMITTED both.
--   experiment_monitor's memory hooks (src/agents/experiment_monitor/memory_hooks.py
--   :427/433, :508/514) write agent_name='experiment_monitor' into
--   episodic_memories.agent_name (typed e2i_agent_name) -> Postgres 22P02
--   ('invalid input value for enum') -> caught by the broad except -> logger.warning
--   -> SILENT loss of experiment_monitor alert + monitoring-check episodic memory.
--
-- Fix: forward-only, idempotent enum extension (same pattern as migration 018).
--   cohort_constructor is included for completeness/safety (idempotent if already present).
--
-- NOTE: deploy skips migrations -> APPLY MANUALLY to the deployed Supabase, e.g.:
--   docker exec -i <supabase-db-container> psql -U postgres -d postgres \
--     < database/memory/029_add_experiment_monitor_cohort_constructor_to_e2i_enum.sql
-- ============================================================================

-- Tier 0: Cohort Constructor (ML Foundation; placed before orchestrator like the
-- other Tier-0 agents added in migration 018).
DO $$ BEGIN
    ALTER TYPE e2i_agent_name ADD VALUE IF NOT EXISTS 'cohort_constructor' BEFORE 'orchestrator';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Tier 3: Experiment Monitor (A/B experiment health: SRM, interim, enrollment).
DO $$ BEGIN
    ALTER TYPE e2i_agent_name ADD VALUE IF NOT EXISTS 'experiment_monitor' AFTER 'experiment_designer';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================================
-- VERIFICATION (run after migration):
--   SELECT enumlabel FROM pg_enum e
--     JOIN pg_type t ON e.enumtypid = t.oid
--    WHERE t.typname = 'e2i_agent_name'
--    ORDER BY e.enumsortorder;
--   -- Expect 'cohort_constructor' and 'experiment_monitor' present.
-- ============================================================================
