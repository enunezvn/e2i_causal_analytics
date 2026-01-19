-- ============================================================================
-- MIGRATION 029: Update Agent ENUMs to v4 (18-Agent Architecture)
-- ============================================================================
-- Date: 2026-01-19
-- Purpose: Align core schema ENUMs with domain_vocabulary.yaml v5.0.0
--
-- Changes:
-- 1. agent_tier_type: Replace old naming with tier_X_* format
-- 2. agent_name_type_v2: Expand from 11 to 18 agents (add Tier 0)
--
-- This migration addresses vocabulary/database ENUM synchronization issues.
-- ============================================================================

-- ============================================================================
-- PART 1: CREATE NEW agent_tier_type_v2 ENUM
-- ============================================================================
-- We can't modify ENUM values directly in PostgreSQL, so we create a new type

DO $$ BEGIN
    CREATE TYPE agent_tier_type_v2 AS ENUM (
        'tier_0_ml_foundation',    -- 7 agents - ML lifecycle
        'tier_1_coordination',     -- 2 agents - Orchestrator + Tool Composer
        'tier_2_causal',           -- 3 agents - Causal analytics
        'tier_3_monitoring',       -- 3 agents - Drift & experimentation
        'tier_4_prediction',       -- 2 agents - ML predictions
        'tier_5_self_improvement'  -- 2 agents - Learning & feedback
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================================
-- PART 2: CREATE NEW agent_name_type_v3 ENUM (18 Agents)
-- ============================================================================

DO $$ BEGIN
    CREATE TYPE agent_name_type_v3 AS ENUM (
        -- Tier 0: ML Foundation (7 agents)
        'scope_definer',
        'data_preparer',
        'model_selector',
        'model_trainer',
        'model_evaluator',
        'model_deployer',
        'model_monitor',
        -- Tier 1: Coordination (2 agents)
        'orchestrator',
        'tool_composer',
        -- Tier 2: Causal Analytics (3 agents)
        'causal_impact',
        'heterogeneous_optimizer',
        'gap_analyzer',
        -- Tier 3: Monitoring (3 agents)
        'drift_monitor',
        'experiment_designer',
        'data_quality_monitor',
        -- Tier 4: Prediction (2 agents)
        'prediction_synthesizer',
        'risk_assessor',
        -- Tier 5: Self-Improvement (2 agents)
        'explainer',
        'feedback_learner',
        -- Legacy agents (for backwards compatibility)
        'health_score',
        'resource_optimizer'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================================
-- PART 3: MIGRATION HELPER - Map old tier names to new
-- ============================================================================

CREATE OR REPLACE FUNCTION map_tier_v1_to_v2(old_tier TEXT)
RETURNS agent_tier_type_v2 AS $$
BEGIN
    RETURN CASE old_tier
        WHEN 'coordination' THEN 'tier_1_coordination'::agent_tier_type_v2
        WHEN 'causal_analytics' THEN 'tier_2_causal'::agent_tier_type_v2
        WHEN 'monitoring' THEN 'tier_3_monitoring'::agent_tier_type_v2
        WHEN 'ml_predictions' THEN 'tier_4_prediction'::agent_tier_type_v2
        WHEN 'self_improvement' THEN 'tier_5_self_improvement'::agent_tier_type_v2
        ELSE 'tier_1_coordination'::agent_tier_type_v2  -- Default
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- PART 4: MIGRATION HELPER - Map old agent names to v3
-- ============================================================================

CREATE OR REPLACE FUNCTION map_agent_v2_to_v3(old_name TEXT)
RETURNS agent_name_type_v3 AS $$
BEGIN
    -- Direct mappings (agents that exist in both versions)
    IF old_name IN (
        'orchestrator', 'causal_impact', 'gap_analyzer', 'heterogeneous_optimizer',
        'drift_monitor', 'experiment_designer', 'prediction_synthesizer',
        'explainer', 'feedback_learner'
    ) THEN
        RETURN old_name::agent_name_type_v3;
    END IF;

    -- Renamed/legacy mappings
    RETURN CASE old_name
        WHEN 'health_score' THEN 'health_score'::agent_name_type_v3
        WHEN 'resource_optimizer' THEN 'resource_optimizer'::agent_name_type_v3
        ELSE 'orchestrator'::agent_name_type_v3  -- Default fallback
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- PART 5: UPDATE agent_registry TABLE (if exists)
-- ============================================================================

DO $$
DECLARE
    has_agent_registry BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT FROM information_schema.tables
        WHERE table_name = 'agent_registry'
    ) INTO has_agent_registry;

    IF has_agent_registry THEN
        -- Add new columns with v2/v3 types
        IF NOT EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_name = 'agent_registry' AND column_name = 'tier_v2'
        ) THEN
            ALTER TABLE agent_registry
            ADD COLUMN tier_v2 agent_tier_type_v2;
        END IF;

        IF NOT EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_name = 'agent_registry' AND column_name = 'name_v3'
        ) THEN
            ALTER TABLE agent_registry
            ADD COLUMN name_v3 agent_name_type_v3;
        END IF;

        -- Migrate existing data (if old columns exist)
        UPDATE agent_registry
        SET tier_v2 = map_tier_v1_to_v2(tier::TEXT)
        WHERE tier IS NOT NULL AND tier_v2 IS NULL;

        UPDATE agent_registry
        SET name_v3 = map_agent_v2_to_v3(name::TEXT)
        WHERE name IS NOT NULL AND name_v3 IS NULL;
    END IF;
END $$;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================
-- Run these after migration to verify:
--
-- Check new tier enum values:
-- SELECT enumlabel FROM pg_enum
-- WHERE enumtypid = 'agent_tier_type_v2'::regtype
-- ORDER BY enumsortorder;
--
-- Check new agent name enum values:
-- SELECT enumlabel FROM pg_enum
-- WHERE enumtypid = 'agent_name_type_v3'::regtype
-- ORDER BY enumsortorder;
--
-- Expected tier values (6):
-- tier_0_ml_foundation, tier_1_coordination, tier_2_causal,
-- tier_3_monitoring, tier_4_prediction, tier_5_self_improvement
--
-- Expected agent values (20):
-- scope_definer, data_preparer, model_selector, model_trainer, model_evaluator,
-- model_deployer, model_monitor, orchestrator, tool_composer, causal_impact,
-- heterogeneous_optimizer, gap_analyzer, drift_monitor, experiment_designer,
-- data_quality_monitor, prediction_synthesizer, risk_assessor, explainer,
-- feedback_learner, health_score, resource_optimizer
-- ============================================================================

-- ============================================================================
-- NOTES FOR FUTURE WORK
-- ============================================================================
-- 1. Once all tables are migrated to v2/v3 columns, old columns can be dropped
-- 2. After verification, old ENUM types can be dropped:
--    DROP TYPE IF EXISTS agent_tier_type CASCADE;
--    DROP TYPE IF EXISTS agent_name_type_v2 CASCADE;
-- 3. Rename new types to replace old ones:
--    ALTER TYPE agent_tier_type_v2 RENAME TO agent_tier_type;
--    ALTER TYPE agent_name_type_v3 RENAME TO agent_name_type;
-- ============================================================================
