-- ============================================================================
-- MIGRATION 030: Fix Agent Tier Classification for health_score & resource_optimizer
-- ============================================================================
-- Date: 2026-01-19
-- Purpose: Correct tier classification for health_score and resource_optimizer
--
-- Background:
-- These agents were incorrectly placed in the 'legacy' section of domain_vocabulary.yaml
-- and migration 029 (agent_name_type_v3), but per the official E2I MLOps Implementation
-- Plan v1.1:
--   - health_score belongs in Tier 3 (Monitoring), NOT legacy
--   - resource_optimizer belongs in Tier 4 (ML & Predictions), NOT legacy
--
-- This migration:
-- 1. Creates an agent_tier_mapping table as authoritative agent→tier mapping
-- 2. Updates any existing agent_registry records with correct tier assignments
-- 3. Provides verification queries
-- ============================================================================

BEGIN;

-- ============================================================================
-- PART 1: CREATE agent_tier_mapping TABLE (if not exists)
-- ============================================================================
-- This serves as the authoritative mapping between agents and their correct tiers

CREATE TABLE IF NOT EXISTS agent_tier_mapping (
    id SERIAL PRIMARY KEY,
    agent_name TEXT NOT NULL UNIQUE,
    tier TEXT NOT NULL,
    agent_type TEXT NOT NULL DEFAULT 'standard',  -- standard, hybrid, deep
    sla_seconds INTEGER,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for fast lookups
CREATE INDEX IF NOT EXISTS idx_agent_tier_mapping_agent_name ON agent_tier_mapping(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_tier_mapping_tier ON agent_tier_mapping(tier);

-- ============================================================================
-- PART 2: UPSERT AUTHORITATIVE AGENT-TIER MAPPINGS
-- ============================================================================
-- Based on E2I MLOps Implementation Plan v1.1 and domain_vocabulary.yaml v5.0

INSERT INTO agent_tier_mapping (agent_name, tier, agent_type, sla_seconds, description)
VALUES
    -- Tier 0: ML Foundation (7 agents)
    ('scope_definer', 'tier_0_ml_foundation', 'standard', 60, 'Define ML problem scope'),
    ('data_preparer', 'tier_0_ml_foundation', 'standard', 120, 'Data preparation & validation'),
    ('model_selector', 'tier_0_ml_foundation', 'standard', 60, 'Model selection & benchmarking'),
    ('model_trainer', 'tier_0_ml_foundation', 'hybrid', 300, 'Model training & hyperparameter tuning'),
    ('model_evaluator', 'tier_0_ml_foundation', 'standard', 60, 'Model evaluation & metrics'),
    ('model_deployer', 'tier_0_ml_foundation', 'standard', 120, 'Model deployment & versioning'),
    ('model_monitor', 'tier_0_ml_foundation', 'standard', 60, 'Model monitoring in production'),

    -- Tier 1: Coordination (2 agents)
    ('orchestrator', 'tier_1_coordination', 'standard', 10, 'Coordinates all agents, routes queries'),
    ('tool_composer', 'tier_1_coordination', 'standard', 30, 'Multi-faceted query decomposition & tool orchestration'),

    -- Tier 2: Causal Analytics (4 agents)
    ('causal_impact', 'tier_2_causal', 'hybrid', 90, 'Traces causal chains, effect estimation'),
    ('heterogeneous_optimizer', 'tier_2_causal', 'hybrid', 120, 'Segment-level CATE analysis'),
    ('gap_analyzer', 'tier_2_causal', 'standard', 60, 'ROI opportunity detection'),
    ('experiment_designer', 'tier_2_causal', 'hybrid', 120, 'A/B test design with Digital Twin pre-screening'),

    -- Tier 3: Monitoring (3 agents) - CORRECTED: includes health_score
    ('drift_monitor', 'tier_3_monitoring', 'standard', 60, 'Data/model drift detection'),
    ('data_quality_monitor', 'tier_3_monitoring', 'standard', 60, 'Data quality monitoring'),
    ('health_score', 'tier_3_monitoring', 'standard', 60, 'System health metrics'),

    -- Tier 4: Prediction (3 agents) - CORRECTED: includes resource_optimizer
    ('prediction_synthesizer', 'tier_4_prediction', 'hybrid', 30, 'ML prediction aggregation'),
    ('risk_assessor', 'tier_4_prediction', 'standard', 30, 'Risk assessment scoring'),
    ('resource_optimizer', 'tier_4_prediction', 'standard', 20, 'Resource allocation optimization'),

    -- Tier 5: Self-Improvement (2 agents)
    ('explainer', 'tier_5_self_improvement', 'deep', 180, 'Natural language explanations'),
    ('feedback_learner', 'tier_5_self_improvement', 'deep', 120, 'Self-improvement from feedback')
ON CONFLICT (agent_name) DO UPDATE SET
    tier = EXCLUDED.tier,
    agent_type = EXCLUDED.agent_type,
    sla_seconds = EXCLUDED.sla_seconds,
    description = EXCLUDED.description,
    updated_at = NOW();

-- ============================================================================
-- PART 3: UPDATE agent_registry TABLE (if exists)
-- ============================================================================
-- Update any existing records with incorrect tier assignments

DO $$
DECLARE
    has_agent_registry BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT FROM information_schema.tables
        WHERE table_name = 'agent_registry'
    ) INTO has_agent_registry;

    IF has_agent_registry THEN
        -- Update health_score to Tier 3 (if tier_v2 column exists)
        IF EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_name = 'agent_registry' AND column_name = 'tier_v2'
        ) THEN
            UPDATE agent_registry
            SET tier_v2 = 'tier_3_monitoring'::agent_tier_type_v2,
                updated_at = NOW()
            WHERE (agent_name::TEXT = 'health_score' OR name_v3::TEXT = 'health_score');

            UPDATE agent_registry
            SET tier_v2 = 'tier_4_prediction'::agent_tier_type_v2,
                updated_at = NOW()
            WHERE (agent_name::TEXT = 'resource_optimizer' OR name_v3::TEXT = 'resource_optimizer');
        END IF;

        -- Update using agent_tier column (original column name)
        IF EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_name = 'agent_registry' AND column_name = 'agent_tier'
        ) THEN
            -- For text-based tier column
            UPDATE agent_registry
            SET agent_tier = 'tier_3_monitoring',
                updated_at = NOW()
            WHERE agent_name::TEXT = 'health_score';

            UPDATE agent_registry
            SET agent_tier = 'tier_4_prediction',
                updated_at = NOW()
            WHERE agent_name::TEXT = 'resource_optimizer';
        END IF;
    END IF;
END $$;

-- ============================================================================
-- PART 4: CREATE HELPER FUNCTION FOR TIER LOOKUP
-- ============================================================================

CREATE OR REPLACE FUNCTION get_agent_tier(p_agent_name TEXT)
RETURNS TEXT AS $$
DECLARE
    v_tier TEXT;
BEGIN
    SELECT tier INTO v_tier
    FROM agent_tier_mapping
    WHERE agent_name = p_agent_name;

    RETURN COALESCE(v_tier, 'tier_1_coordination');  -- Default fallback
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- PART 5: CREATE VIEW FOR EASY TIER VERIFICATION
-- ============================================================================

CREATE OR REPLACE VIEW v_agent_tier_summary AS
SELECT
    tier,
    COUNT(*) as agent_count,
    STRING_AGG(agent_name, ', ' ORDER BY agent_name) as agents
FROM agent_tier_mapping
GROUP BY tier
ORDER BY tier;

COMMIT;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================
-- Run these after migration to verify:

-- 1. Verify health_score is in Tier 3:
-- SELECT agent_name, tier FROM agent_tier_mapping WHERE agent_name = 'health_score';
-- Expected: tier = 'tier_3_monitoring'

-- 2. Verify resource_optimizer is in Tier 4:
-- SELECT agent_name, tier FROM agent_tier_mapping WHERE agent_name = 'resource_optimizer';
-- Expected: tier = 'tier_4_prediction'

-- 3. View tier summary:
-- SELECT * FROM v_agent_tier_summary;
-- Expected:
-- | tier                    | agent_count | agents                                              |
-- |-------------------------|-------------|-----------------------------------------------------|
-- | tier_0_ml_foundation    | 7           | data_preparer, model_deployer, model_evaluator, ... |
-- | tier_1_coordination     | 2           | orchestrator, tool_composer                         |
-- | tier_2_causal           | 4           | causal_impact, experiment_designer, gap_analyzer,...|
-- | tier_3_monitoring       | 3           | data_quality_monitor, drift_monitor, health_score   |
-- | tier_4_prediction       | 3           | prediction_synthesizer, resource_optimizer, risk_...|
-- | tier_5_self_improvement | 2           | explainer, feedback_learner                         |

-- 4. Verify no agents in legacy:
-- SELECT agent_name FROM agent_tier_mapping WHERE tier LIKE '%legacy%';
-- Expected: 0 rows
-- ============================================================================
