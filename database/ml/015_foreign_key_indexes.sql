-- =============================================================================
-- Foreign Key Indexes
-- Migration: 015_foreign_key_indexes.sql
-- Date: 2025-12-20
-- Purpose: Add missing indexes on foreign key columns for JOIN performance
-- Reference: Supabase Performance Advisor - 32 unindexed foreign keys
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Split Config Foreign Keys (8 tables)
-- These are frequently used for train/test data splits
-- -----------------------------------------------------------------------------

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_activities_split_config_id
    ON agent_activities(split_config_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_business_metrics_split_config_id
    ON business_metrics(split_config_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_paths_split_config_id
    ON causal_paths(split_config_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_predictions_split_config_id
    ON ml_predictions(split_config_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_patient_journeys_split_config_id
    ON patient_journeys(split_config_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_treatment_events_split_config_id
    ON treatment_events(split_config_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_triggers_split_config_id
    ON triggers(split_config_id);

-- -----------------------------------------------------------------------------
-- HCP Foreign Keys (frequently joined for HCP analytics)
-- -----------------------------------------------------------------------------

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_predictions_hcp_id
    ON ml_predictions(hcp_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_treatment_events_hcp_id
    ON treatment_events(hcp_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_learning_signals_related_hcp_id
    ON learning_signals(related_hcp_id);

-- -----------------------------------------------------------------------------
-- Patient Foreign Keys
-- -----------------------------------------------------------------------------

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_learning_signals_related_patient_id
    ON learning_signals(related_patient_id);

-- -----------------------------------------------------------------------------
-- Trigger Foreign Keys
-- -----------------------------------------------------------------------------

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_learning_signals_related_trigger_id
    ON learning_signals(related_trigger_id);

-- -----------------------------------------------------------------------------
-- Memory Tables Foreign Keys (episodic, semantic, procedural)
-- -----------------------------------------------------------------------------

-- Episodic memories
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_episodic_memories_agent_activity_id
    ON episodic_memories(agent_activity_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_episodic_memories_prediction_id
    ON episodic_memories(prediction_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_episodic_memories_treatment_event_id
    ON episodic_memories(treatment_event_id);

-- Semantic memory cache
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_semantic_memory_cache_object_causal_id
    ON semantic_memory_cache(object_causal_path_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_semantic_memory_cache_object_hcp_id
    ON semantic_memory_cache(object_hcp_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_semantic_memory_cache_object_patient_id
    ON semantic_memory_cache(object_patient_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_semantic_memory_cache_object_trigger_id
    ON semantic_memory_cache(object_trigger_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_semantic_memory_cache_subject_causal_id
    ON semantic_memory_cache(subject_causal_path_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_semantic_memory_cache_subject_trigger_id
    ON semantic_memory_cache(subject_trigger_id);

-- Procedural memories
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_procedural_memories_parent_id
    ON procedural_memories(parent_procedure_id);

-- -----------------------------------------------------------------------------
-- ML/MLOps Tables Foreign Keys
-- -----------------------------------------------------------------------------

-- ML Deployments
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_deployments_previous_deployment_id
    ON ml_deployments(previous_deployment_id);

-- ML Observability Spans
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_observability_spans_deployment_id
    ON ml_observability_spans(deployment_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_observability_spans_experiment_id
    ON ml_observability_spans(experiment_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_observability_spans_training_run_id
    ON ml_observability_spans(training_run_id);

-- ML Training Runs
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_training_runs_model_registry_id
    ON ml_training_runs(model_registry_id);

-- -----------------------------------------------------------------------------
-- Other Tables Foreign Keys
-- -----------------------------------------------------------------------------

-- Audit chain (linked list of entries)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_chain_entries_previous_entry_id
    ON audit_chain_entries(previous_entry_id);

-- Composer episodes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_composer_episodes_classification_id
    ON composer_episodes(classification_id);

-- DSPy prompt versions
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dspy_prompt_versions_optimization_run_id
    ON dspy_prompt_versions(optimization_run_id);

-- Tool performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tool_performance_step_id
    ON tool_performance(step_id);

-- ROI calculations (superseded chain)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_roi_calculations_superseded_by
    ON roi_calculations(superseded_by);
