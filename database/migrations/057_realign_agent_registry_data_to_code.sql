-- ============================================================================
-- MIGRATION 057: Realign agent_registry + agent_tier_mapping DATA to the code roster
-- ============================================================================
-- Date: 2026-06-01
-- Issue: #607 (full agent-taxonomy reconciliation)
--
-- ⚠️  UNVERIFIED LOCALLY — this dev environment has no direct DB connection
--     (REST creds only; prod Supabase is a self-hosted droplet Docker). REVIEW
--     for foreign-key references and APPLY + VERIFY on the droplet before trust.
--     The faithful arbiter is tests/integration/test_migration/test_phase3_data.py
--     (gated — needs a reachable Postgres). deploy.yml skips migrations -> MANUAL apply:
--       docker exec -i <supabase-db> psql -U postgres -d postgres \
--         < database/migrations/057_realign_agent_registry_data_to_code.sql
--
-- Source of truth = src/agents/factory.py AGENT_REGISTRY_CONFIG (21 agents).
--
-- Two tables carry stale roster DATA (the enum TYPES are retired separately in 056):
--   * agent_registry.agent_tier is agent_tier_type (Tier 1-5 labels only, by design)
--     -> it is missing tool_composer (coordination) + experiment_monitor (monitoring).
--   * agent_tier_mapping.tier is plain TEXT (carries all 6 tiers) -> it still lists the
--     dead non-agents model_evaluator/model_monitor/data_quality_monitor/risk_assessor,
--     omits cohort_constructor/feature_analyzer/observability_connector/experiment_monitor,
--     and misclassifies experiment_designer as tier_2 (code: tier_3).
--
-- Idempotent (ON CONFLICT DO NOTHING / IF EXISTS). DELETEs use plain DELETE so an
-- unexpected FK reference FAILS LOUD rather than cascading silently.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- agent_registry: add the 2 missing Tier 1-5 agents (existing agent_tier labels).
-- ---------------------------------------------------------------------------
INSERT INTO agent_registry (agent_name, agent_tier, display_name, description, capabilities, routes_from_intents, priority_order) VALUES
    ('tool_composer', 'coordination', 'Tool Composer Agent',
     'Multi-faceted query decomposition and tool orchestration',
     '["query_decomposition", "tool_orchestration", "multi_faceted_synthesis"]'::jsonb,
     '["COMPOSE", "MULTI_FACETED"]'::jsonb, 2),
    ('experiment_monitor', 'monitoring', 'Experiment Monitor Agent',
     'Monitors running A/B experiments: SRM detection, interim analysis, enrollment & fidelity health',
     '["srm_detection", "interim_analysis", "enrollment_health", "fidelity_check"]'::jsonb,
     '["MONITOR_EXPERIMENT", "EXPERIMENT_STATUS"]'::jsonb, 32)
ON CONFLICT (agent_name) DO NOTHING;

-- ---------------------------------------------------------------------------
-- agent_tier_mapping: retire dead non-agents, add missing agents, fix tiers.
-- ---------------------------------------------------------------------------
DELETE FROM agent_tier_mapping
 WHERE agent_name IN ('model_evaluator', 'model_monitor', 'data_quality_monitor', 'risk_assessor');

UPDATE agent_tier_mapping SET tier = 'tier_3_monitoring'
 WHERE agent_name = 'experiment_designer';

INSERT INTO agent_tier_mapping (agent_name, tier, agent_type, sla_seconds, description) VALUES
    ('cohort_constructor',      'tier_0_ml_foundation', 'standard', 120, 'Patient cohort construction & eligibility'),
    ('feature_analyzer',        'tier_0_ml_foundation', 'hybrid',   120, 'Feature analysis & SHAP interpretation'),
    ('observability_connector', 'tier_0_ml_foundation', 'standard', 100, 'Opik spans, cross-tier telemetry'),
    ('experiment_monitor',      'tier_3_monitoring',    'standard', 20,  'A/B experiment health: SRM, interim, enrollment')
ON CONFLICT (agent_name) DO NOTHING;

-- ============================================================================
-- VERIFICATION (run after migration):
--   SELECT agent_name FROM agent_registry ORDER BY agent_name;  -- expect 13 Tier 1-5 incl tool_composer, experiment_monitor
--   SELECT agent_name, tier FROM agent_tier_mapping ORDER BY tier, agent_name;  -- expect 21 code agents, none of the 4 dead names
-- ============================================================================
