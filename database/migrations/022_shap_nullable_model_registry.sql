-- ============================================================================
-- Migration 022: Make model_registry_id nullable in ml_shap_analyses
-- Version: 1.0.0
-- Created: 2026-04-12
--
-- Purpose: SHAP analysis runs at Step 6 (feature_analyzer), before model
-- registration happens at Step 7 (model_deployer). The NOT NULL constraint
-- prevents storing SHAP results when model_registry_id is not yet available.
--
-- The FK relationship is preserved — when a model_registry_id IS provided,
-- it must still reference a valid ml_model_registry row.
-- ============================================================================

ALTER TABLE ml_shap_analyses
    ALTER COLUMN model_registry_id DROP NOT NULL;

COMMENT ON COLUMN ml_shap_analyses.model_registry_id IS
    'FK to ml_model_registry. Nullable: SHAP analysis may run before model registration.';
