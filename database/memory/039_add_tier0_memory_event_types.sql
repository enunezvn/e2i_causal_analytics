-- ============================================================================
-- Migration 039: Add Tier-0 ml_foundation memory event types
-- ============================================================================
-- Purpose: The Tier-0 ml_foundation agents' episodic memory hooks emit
--          event_types that were never added to the ``memory_event_type`` enum,
--          so every Tier-0 episodic write failed with
--          ``invalid input value for enum memory_event_type`` (one of the four
--          drift layers that left the Tier-0 episodic path non-functional, #749).
--          Mirrors migration 020, which added the Tier-1 (tool_composer /
--          resource_optimizer / explainer) event types the same way.
-- Reference: #749 — make a Tier-0 run produce ``episodic_memories > 0``.
-- Safety: additive only (ADD VALUE IF NOT EXISTS) — cannot break existing rows.
-- ============================================================================

ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'scope_definition_completed';
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'qc_report_completed';
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'model_selection_completed';
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'model_training_completed';
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'feature_analysis_completed';
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'model_deployment_completed';
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'observability_metrics_collected';

-- cohort_constructor (Tier-1) emits this via the SAME broken legacy-hook pattern; the
-- compat shim now routes its calls to the DB, so the enum must accept it too
-- (codex-rescue MED-1 — otherwise the row is silently rejected at the enum check).
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'cohort_construction_completed';
