-- ============================================================================
-- E2I CAUSAL ANALYTICS - AGENT ENUM MIGRATION
-- ============================================================================
-- Migration: 006_update_agent_enum.sql
-- Version: 2.1.0
-- Date: 2025-11-28
-- Description: Updates agent_name_type ENUM from 8 legacy agents to 
--              integrated 11-agent tiered architecture
-- ============================================================================

-- ============================================================================
-- MIGRATION SUMMARY
-- ============================================================================
-- 
-- OLD ENUM (8 agents):                    NEW ENUM (11 agents):
-- ─────────────────────────────────────   ─────────────────────────────────────
-- causal_chain_analyzer          ──────►  causal_impact
-- multiplier_discoverer          ──────►  gap_analyzer  
-- data_drift_monitor             ──────►  drift_monitor
-- prediction_synthesizer         ──────►  prediction_synthesizer (unchanged)
-- heterogeneous_optimizer        ──────►  heterogeneous_optimizer (unchanged)
-- competitive_landscape          ──────►  (REMOVED - optional, can add later)
-- explainer_agent                ──────►  explainer
-- feedback_learner               ──────►  feedback_learner (unchanged)
-- (missing)                      ──────►  orchestrator (NEW)
-- (missing)                      ──────►  experiment_designer (NEW)
-- (missing)                      ──────►  health_score (NEW)
-- (missing)                      ──────►  resource_optimizer (NEW)
--
-- ============================================================================

BEGIN;

-- ============================================================================
-- STEP 1: Create new agent tier enum (for hierarchical organization)
-- ============================================================================

DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'agent_tier_type') THEN
        CREATE TYPE agent_tier_type AS ENUM (
            'coordination',       -- Tier 1: orchestrator
            'causal_analytics',   -- Tier 2: causal_impact, gap_analyzer, heterogeneous_optimizer
            'monitoring',         -- Tier 3: drift_monitor, experiment_designer, health_score
            'ml_predictions',     -- Tier 4: prediction_synthesizer, resource_optimizer
            'self_improvement'    -- Tier 5: explainer, feedback_learner
        );
        RAISE NOTICE 'Created agent_tier_type ENUM';
    ELSE
        RAISE NOTICE 'agent_tier_type ENUM already exists';
    END IF;
END $$;

-- ============================================================================
-- STEP 2: Create new agent name enum with integrated 11-agent architecture
-- ============================================================================

-- Create new enum type (can't directly ALTER ENUM to remove values in PostgreSQL)
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'agent_name_type_v2') THEN
        CREATE TYPE agent_name_type_v2 AS ENUM (
            -- Tier 1: Coordination
            'orchestrator',
            
            -- Tier 2: Causal Analytics (Core E2I Mission)
            'causal_impact',
            'gap_analyzer',
            'heterogeneous_optimizer',
            
            -- Tier 3: Monitoring & Experimentation
            'drift_monitor',
            'experiment_designer',
            'health_score',
            
            -- Tier 4: ML & Predictions
            'prediction_synthesizer',
            'resource_optimizer',
            
            -- Tier 5: Self-Improvement (Critical for NLV + RAG)
            'explainer',
            'feedback_learner'
        );
        RAISE NOTICE 'Created agent_name_type_v2 ENUM';
    ELSE
        RAISE NOTICE 'agent_name_type_v2 ENUM already exists';
    END IF;
END $$;

-- ============================================================================
-- STEP 3: Create mapping table for legacy to new agent names
-- ============================================================================

CREATE TABLE IF NOT EXISTS _agent_name_migration (
    old_name VARCHAR(50) PRIMARY KEY,
    new_name VARCHAR(50) NOT NULL,
    migration_notes TEXT
);

-- Insert mapping (idempotent)
INSERT INTO _agent_name_migration (old_name, new_name, migration_notes) VALUES
    ('causal_chain_analyzer', 'causal_impact', 'Renamed for clarity - traces causal chains and estimates impact'),
    ('multiplier_discoverer', 'gap_analyzer', 'Renamed - discovers ROI multipliers through gap analysis'),
    ('data_drift_monitor', 'drift_monitor', 'Simplified name'),
    ('prediction_synthesizer', 'prediction_synthesizer', 'Unchanged'),
    ('heterogeneous_optimizer', 'heterogeneous_optimizer', 'Unchanged'),
    ('competitive_landscape', NULL, 'REMOVED - optional agent, can be added later if needed'),
    ('explainer_agent', 'explainer', 'Simplified name'),
    ('feedback_learner', 'feedback_learner', 'Unchanged')
ON CONFLICT (old_name) DO UPDATE SET
    new_name = EXCLUDED.new_name,
    migration_notes = EXCLUDED.migration_notes;

-- ============================================================================
-- STEP 4: Update agent_activities table (uses VARCHAR, not ENUM)
-- ============================================================================

-- Update existing agent names in agent_activities table
UPDATE agent_activities 
SET agent_name = m.new_name
FROM _agent_name_migration m
WHERE agent_activities.agent_name = m.old_name
  AND m.new_name IS NOT NULL;

-- Log any unmigrated records (competitive_landscape)
DO $$
DECLARE
    unmigrated_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO unmigrated_count
    FROM agent_activities aa
    JOIN _agent_name_migration m ON aa.agent_name = m.old_name
    WHERE m.new_name IS NULL;
    
    IF unmigrated_count > 0 THEN
        RAISE WARNING 'Found % agent_activities records with deprecated agent names (competitive_landscape). Consider archiving or reassigning.', unmigrated_count;
    END IF;
END $$;

-- ============================================================================
-- STEP 5: Add agent_tier column to agent_activities (optional enhancement)
-- ============================================================================

-- Add tier column if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'agent_activities' AND column_name = 'agent_tier'
    ) THEN
        ALTER TABLE agent_activities ADD COLUMN agent_tier agent_tier_type;
        RAISE NOTICE 'Added agent_tier column to agent_activities';
    END IF;
END $$;

-- Populate agent_tier based on agent_name
UPDATE agent_activities SET agent_tier = 
    CASE agent_name
        WHEN 'orchestrator' THEN 'coordination'::agent_tier_type
        WHEN 'causal_impact' THEN 'causal_analytics'::agent_tier_type
        WHEN 'gap_analyzer' THEN 'causal_analytics'::agent_tier_type
        WHEN 'heterogeneous_optimizer' THEN 'causal_analytics'::agent_tier_type
        WHEN 'drift_monitor' THEN 'monitoring'::agent_tier_type
        WHEN 'experiment_designer' THEN 'monitoring'::agent_tier_type
        WHEN 'health_score' THEN 'monitoring'::agent_tier_type
        WHEN 'prediction_synthesizer' THEN 'ml_predictions'::agent_tier_type
        WHEN 'resource_optimizer' THEN 'ml_predictions'::agent_tier_type
        WHEN 'explainer' THEN 'self_improvement'::agent_tier_type
        WHEN 'feedback_learner' THEN 'self_improvement'::agent_tier_type
        ELSE NULL
    END
WHERE agent_tier IS NULL;

-- ============================================================================
-- STEP 6: Create agent registry table for runtime configuration
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_registry (
    agent_name VARCHAR(50) PRIMARY KEY,
    agent_tier agent_tier_type NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    capabilities JSONB DEFAULT '[]',
    routes_from_intents JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    priority_order INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Populate agent registry
INSERT INTO agent_registry (agent_name, agent_tier, display_name, description, capabilities, routes_from_intents, priority_order) VALUES
    -- Tier 1: Coordination
    ('orchestrator', 'coordination', 'Orchestrator Agent', 
     'Query routing, multi-agent coordination, response synthesis',
     '["query_routing", "agent_coordination", "response_synthesis", "state_management"]'::jsonb,
     '["ROUTE", "COORDINATE", "SYNTHESIZE"]'::jsonb, 1),
    
    -- Tier 2: Causal Analytics
    ('causal_impact', 'causal_analytics', 'Causal Impact Agent',
     'Traces causal chains, estimates treatment effects, path analysis',
     '["chain_tracing", "effect_estimation", "counterfactual", "path_analysis"]'::jsonb,
     '["CAUSAL", "WHY", "EFFECT", "CHAIN"]'::jsonb, 10),
    
    ('gap_analyzer', 'causal_analytics', 'Gap Analyzer Agent',
     'Identifies performance gaps and ROI opportunities',
     '["gap_detection", "roi_calculation", "multiplier_discovery", "prioritization"]'::jsonb,
     '["GAP", "OPPORTUNITY", "ROI", "PRIORITIZE"]'::jsonb, 11),
    
    ('heterogeneous_optimizer', 'causal_analytics', 'Heterogeneous Optimizer Agent',
     'Analyzes segment-level treatment effects (CATE)',
     '["cate_estimation", "segment_analysis", "personalization"]'::jsonb,
     '["SEGMENT", "PERSONALIZE", "CATE", "SUBGROUP"]'::jsonb, 12),
    
    -- Tier 3: Monitoring & Experimentation
    ('drift_monitor', 'monitoring', 'Drift Monitor Agent',
     'Detects data drift and model degradation',
     '["psi_calculation", "drift_detection", "alert_generation", "causal_attribution"]'::jsonb,
     '["DRIFT", "CHANGE", "DEGRADE", "ALERT"]'::jsonb, 20),
    
    ('experiment_designer', 'monitoring', 'Experiment Designer Agent',
     'Designs A/B tests with causal rigor',
     '["test_design", "power_analysis", "sample_sizing", "causal_validation"]'::jsonb,
     '["EXPERIMENT", "TEST", "AB_TEST", "VALIDATE"]'::jsonb, 21),
    
    ('health_score', 'monitoring', 'Health Score Agent',
     'Computes system health metrics',
     '["composite_scoring", "pareto_optimization", "health_alerting"]'::jsonb,
     '["HEALTH", "STATUS", "SCORE", "MONITOR"]'::jsonb, 22),
    
    -- Tier 4: ML & Predictions
    ('prediction_synthesizer', 'ml_predictions', 'Prediction Synthesizer Agent',
     'Coordinates ML model predictions and ensembles',
     '["ensemble_coordination", "model_selection", "confidence_calibration"]'::jsonb,
     '["PREDICT", "FORECAST", "MODEL", "ENSEMBLE"]'::jsonb, 30),
    
    ('resource_optimizer', 'ml_predictions', 'Resource Optimizer Agent',
     'Optimizes resource allocation by ROI',
     '["budget_optimization", "effort_routing", "roi_allocation"]'::jsonb,
     '["RESOURCE", "ALLOCATE", "BUDGET", "OPTIMIZE"]'::jsonb, 31),
    
    -- Tier 5: Self-Improvement
    ('explainer', 'self_improvement', 'Explainer Agent',
     'Generates natural language explanations and narratives',
     '["narrative_generation", "viz_explanation", "summary_generation"]'::jsonb,
     '["EXPLAIN", "DESCRIBE", "SUMMARIZE", "NARRATIVE"]'::jsonb, 40),
    
    ('feedback_learner', 'self_improvement', 'Feedback Learner Agent',
     'Learns from feedback to improve system performance',
     '["dspy_optimization", "prompt_refinement", "pattern_learning"]'::jsonb,
     '["LEARN", "IMPROVE", "FEEDBACK", "OPTIMIZE_PROMPT"]'::jsonb, 41)
ON CONFLICT (agent_name) DO UPDATE SET
    agent_tier = EXCLUDED.agent_tier,
    display_name = EXCLUDED.display_name,
    description = EXCLUDED.description,
    capabilities = EXCLUDED.capabilities,
    routes_from_intents = EXCLUDED.routes_from_intents,
    priority_order = EXCLUDED.priority_order,
    updated_at = NOW();

-- ============================================================================
-- STEP 7: Create index for agent tier queries
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_agent_activities_tier 
    ON agent_activities(agent_tier);

CREATE INDEX IF NOT EXISTS idx_agent_registry_tier 
    ON agent_registry(agent_tier);

CREATE INDEX IF NOT EXISTS idx_agent_registry_active 
    ON agent_registry(is_active) WHERE is_active = TRUE;

-- ============================================================================
-- STEP 8: Create view for agent routing
-- ============================================================================

CREATE OR REPLACE VIEW v_agent_routing AS
SELECT 
    ar.agent_name,
    ar.agent_tier,
    ar.display_name,
    ar.description,
    ar.priority_order,
    intent.value::text AS routes_from_intent
FROM agent_registry ar
CROSS JOIN LATERAL jsonb_array_elements(ar.routes_from_intents) AS intent
WHERE ar.is_active = TRUE
ORDER BY ar.priority_order, intent.value;

-- ============================================================================
-- STEP 9: Create function to route intent to agent
-- ============================================================================

CREATE OR REPLACE FUNCTION route_intent_to_agent(p_intent TEXT)
RETURNS TABLE(agent_name VARCHAR, agent_tier agent_tier_type, priority INTEGER) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ar.agent_name,
        ar.agent_tier,
        ar.priority_order
    FROM agent_registry ar
    WHERE ar.is_active = TRUE
      AND ar.routes_from_intents ? UPPER(p_intent)
    ORDER BY ar.priority_order
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- STEP 10: Update comments
-- ============================================================================

COMMENT ON TYPE agent_tier_type IS 
'Agent tier classification for hierarchical organization (v2.1.0)';

COMMENT ON TABLE agent_registry IS 
'Runtime configuration for 11-agent tiered architecture. Used for intent routing and capability discovery.';

COMMENT ON FUNCTION route_intent_to_agent IS 
'Routes a query intent to the appropriate agent based on agent_registry configuration.';

-- ============================================================================
-- STEP 11: Cleanup (optional - run manually after validation)
-- ============================================================================

-- Uncomment these after validating the migration:

-- Drop migration helper table
-- DROP TABLE IF EXISTS _agent_name_migration;

-- Drop old enum type (only if no columns reference it)
-- DROP TYPE IF EXISTS agent_name_type;

-- Rename new enum to standard name
-- ALTER TYPE agent_name_type_v2 RENAME TO agent_name_type;

-- ============================================================================
-- MIGRATION VALIDATION
-- ============================================================================

DO $$
DECLARE
    v_agent_count INTEGER;
    v_tier_count INTEGER;
    v_activity_count INTEGER;
BEGIN
    -- Count agents in registry
    SELECT COUNT(*) INTO v_agent_count FROM agent_registry WHERE is_active = TRUE;
    
    -- Count tiers
    SELECT COUNT(DISTINCT agent_tier) INTO v_tier_count FROM agent_registry;
    
    -- Count migrated activities
    SELECT COUNT(*) INTO v_activity_count 
    FROM agent_activities 
    WHERE agent_tier IS NOT NULL;
    
    RAISE NOTICE '============================================';
    RAISE NOTICE 'MIGRATION VALIDATION SUMMARY';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Active agents in registry: %', v_agent_count;
    RAISE NOTICE 'Agent tiers defined: %', v_tier_count;
    RAISE NOTICE 'Agent activities with tier assigned: %', v_activity_count;
    RAISE NOTICE '============================================';
    
    IF v_agent_count = 11 THEN
        RAISE NOTICE '✓ All 11 agents registered successfully';
    ELSE
        RAISE WARNING '✗ Expected 11 agents, found %', v_agent_count;
    END IF;
    
    IF v_tier_count = 5 THEN
        RAISE NOTICE '✓ All 5 tiers present';
    ELSE
        RAISE WARNING '✗ Expected 5 tiers, found %', v_tier_count;
    END IF;
END $$;

COMMIT;

-- ============================================================================
-- ROLLBACK SCRIPT (save separately)
-- ============================================================================
/*
BEGIN;

-- Remove new columns
ALTER TABLE agent_activities DROP COLUMN IF EXISTS agent_tier;

-- Revert agent names
UPDATE agent_activities SET agent_name = 
    CASE agent_name
        WHEN 'causal_impact' THEN 'causal_chain_analyzer'
        WHEN 'gap_analyzer' THEN 'multiplier_discoverer'
        WHEN 'drift_monitor' THEN 'data_drift_monitor'
        WHEN 'explainer' THEN 'explainer_agent'
        ELSE agent_name
    END;

-- Drop new objects
DROP VIEW IF EXISTS v_agent_routing;
DROP FUNCTION IF EXISTS route_intent_to_agent;
DROP TABLE IF EXISTS agent_registry;
DROP TABLE IF EXISTS _agent_name_migration;
DROP TYPE IF EXISTS agent_name_type_v2;
DROP TYPE IF EXISTS agent_tier_type;

COMMIT;
*/
