-- ============================================================================
-- MIGRATION 018: Add Tier 0 Agents and Tool Composer to e2i_agent_name Enum
-- ============================================================================
-- Date: 2025-12-26
-- Purpose: Align e2i_agent_name enum with documented 18-agent architecture
--
-- Current enum (12 agents):
--   orchestrator, causal_impact, gap_analyzer, drift_monitor, heterogeneous_optimizer,
--   fairness_guardian, health_score, experiment_designer, prediction_synthesizer,
--   feedback_learner, explainer, resource_optimizer
--
-- Missing (8 agents):
--   Tier 0 (7): scope_definer, data_preparer, feature_analyzer, model_selector,
--               model_trainer, model_deployer, observability_connector
--   Tier 1 (1): tool_composer
--
-- After migration: 20 agents (18 documented + fairness_guardian + resource_optimizer)
-- Note: fairness_guardian kept for backwards compatibility with existing data
-- ============================================================================

-- Add Tier 0 agents (ML Foundation)
-- Using BEFORE to maintain logical ordering

-- Tier 0: Scope Definer (first in pipeline)
DO $$ BEGIN
    ALTER TYPE e2i_agent_name ADD VALUE IF NOT EXISTS 'scope_definer' BEFORE 'orchestrator';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Tier 0: Data Preparer
DO $$ BEGIN
    ALTER TYPE e2i_agent_name ADD VALUE IF NOT EXISTS 'data_preparer' BEFORE 'orchestrator';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Tier 0: Feature Analyzer
DO $$ BEGIN
    ALTER TYPE e2i_agent_name ADD VALUE IF NOT EXISTS 'feature_analyzer' BEFORE 'orchestrator';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Tier 0: Model Selector
DO $$ BEGIN
    ALTER TYPE e2i_agent_name ADD VALUE IF NOT EXISTS 'model_selector' BEFORE 'orchestrator';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Tier 0: Model Trainer
DO $$ BEGIN
    ALTER TYPE e2i_agent_name ADD VALUE IF NOT EXISTS 'model_trainer' BEFORE 'orchestrator';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Tier 0: Model Deployer
DO $$ BEGIN
    ALTER TYPE e2i_agent_name ADD VALUE IF NOT EXISTS 'model_deployer' BEFORE 'orchestrator';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Tier 0: Observability Connector
DO $$ BEGIN
    ALTER TYPE e2i_agent_name ADD VALUE IF NOT EXISTS 'observability_connector' BEFORE 'orchestrator';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Tier 1: Tool Composer (after orchestrator)
DO $$ BEGIN
    ALTER TYPE e2i_agent_name ADD VALUE IF NOT EXISTS 'tool_composer' AFTER 'orchestrator';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================================
-- VERIFICATION QUERY (run after migration)
-- ============================================================================
-- SELECT enumlabel
-- FROM pg_enum
-- WHERE enumtypid = 'e2i_agent_name'::regtype
-- ORDER BY enumsortorder;

-- Expected output (20 agents):
-- scope_definer, data_preparer, feature_analyzer, model_selector, model_trainer,
-- model_deployer, observability_connector, orchestrator, tool_composer,
-- causal_impact, gap_analyzer, drift_monitor, heterogeneous_optimizer,
-- fairness_guardian, health_score, experiment_designer, prediction_synthesizer,
-- feedback_learner, explainer, resource_optimizer

-- ============================================================================
-- AGENT TIER REFERENCE (18-agent documented architecture)
-- ============================================================================
-- Tier 0 (ML Foundation - 7 agents):
--   scope_definer, data_preparer, feature_analyzer, model_selector,
--   model_trainer, model_deployer, observability_connector
--
-- Tier 1 (Coordination - 2 agents):
--   orchestrator, tool_composer
--
-- Tier 2 (Causal Analytics - 3 agents):
--   causal_impact, gap_analyzer, heterogeneous_optimizer
--
-- Tier 3 (Monitoring - 3 agents):
--   drift_monitor, experiment_designer, health_score
--
-- Tier 4 (ML Predictions - 2 agents):
--   prediction_synthesizer, resource_optimizer
--
-- Tier 5 (Self-Improvement - 2 agents):
--   explainer, feedback_learner
-- ============================================================================
