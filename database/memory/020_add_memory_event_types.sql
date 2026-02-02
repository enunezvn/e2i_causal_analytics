-- ============================================================================
-- Migration 020: Add Missing Memory Event Types
-- ============================================================================
-- Purpose: Add event types used by tool_composer, resource_optimizer, and
--          explainer agents that are missing from the memory_event_type enum
-- Reference: run_tier1_5_test.py error output
-- ============================================================================

ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'composition_completed';
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'optimization_completed';
ALTER TYPE memory_event_type ADD VALUE IF NOT EXISTS 'explanation_generated';
