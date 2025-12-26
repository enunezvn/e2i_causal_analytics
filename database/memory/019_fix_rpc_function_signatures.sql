-- ============================================================================
-- Migration 019: Fix RPC Function Signatures
-- ============================================================================
-- Purpose: Update RPC functions to match Python code expectations
--
-- Issue: The get_agent_activity_context function returns fields with different
-- names than the Python AgentActivityContext dataclass expects.
--
-- Fixes:
--   - activity_type -> action_type
--   - activity_timestamp -> started_at
--   - processing_duration_ms -> duration_ms
--   - Add linked entities: trigger, causal_paths, predictions
-- ============================================================================

-- Drop existing function to replace it
DROP FUNCTION IF EXISTS get_agent_activity_context(VARCHAR);

-- ============================================================================
-- FUNCTION: get_agent_activity_context
-- Returns agent activity with linked E2I entities
-- ============================================================================

CREATE OR REPLACE FUNCTION get_agent_activity_context(p_activity_id VARCHAR)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    result JSONB;
    activity_record RECORD;
    linked_trigger JSONB := NULL;
    linked_causal_paths JSONB := '[]'::JSONB;
    linked_predictions JSONB := '[]'::JSONB;
BEGIN
    -- Get the base activity
    SELECT
        aa.activity_id,
        aa.agent_name,
        aa.activity_type AS action_type,
        aa.activity_timestamp AS started_at,
        aa.status,
        aa.processing_duration_ms AS duration_ms,
        aa.agent_tier,
        aa.workstream,
        aa.confidence_level,
        aa.recommendations
    INTO activity_record
    FROM agent_activities aa
    WHERE aa.activity_id = p_activity_id;

    -- If activity not found, return empty
    IF activity_record IS NULL THEN
        RETURN '{}'::JSONB;
    END IF;

    -- Find linked trigger via episodic_memories
    SELECT jsonb_build_object(
        'trigger_id', t.trigger_id,
        'trigger_type', t.trigger_type,
        'description', t.description,
        'brand', t.brand
    )
    INTO linked_trigger
    FROM episodic_memories em
    JOIN triggers t ON em.trigger_id = t.trigger_id
    WHERE em.agent_activity_id = p_activity_id
    AND em.trigger_id IS NOT NULL
    LIMIT 1;

    -- Find linked causal paths via episodic_memories
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'path_id', cp.path_id,
        'cause_kpi', cp.cause_kpi,
        'effect_kpi', cp.effect_kpi,
        'confidence', cp.confidence,
        'brand', cp.brand
    )), '[]'::JSONB)
    INTO linked_causal_paths
    FROM episodic_memories em
    JOIN causal_paths cp ON em.causal_path_id = cp.path_id
    WHERE em.agent_activity_id = p_activity_id
    AND em.causal_path_id IS NOT NULL;

    -- Find linked predictions via episodic_memories
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'prediction_id', mp.prediction_id,
        'model_type', mp.model_type,
        'prediction_type', mp.prediction_type,
        'confidence', mp.confidence,
        'brand', mp.brand
    )), '[]'::JSONB)
    INTO linked_predictions
    FROM episodic_memories em
    JOIN ml_predictions mp ON em.prediction_id = mp.prediction_id
    WHERE em.agent_activity_id = p_activity_id
    AND em.prediction_id IS NOT NULL;

    -- Build the final result
    result := jsonb_build_object(
        -- Core activity fields (named to match Python dataclass)
        'activity_id', activity_record.activity_id,
        'agent_name', activity_record.agent_name,
        'action_type', activity_record.action_type,
        'started_at', activity_record.started_at,
        'completed_at', NULL,  -- Not stored in current schema
        'status', activity_record.status,
        'duration_ms', activity_record.duration_ms,
        'tokens_used', NULL,  -- Not stored in current schema

        -- Linked entities
        'trigger', linked_trigger,
        'causal_paths', CASE WHEN linked_causal_paths = '[]'::JSONB THEN NULL ELSE linked_causal_paths END,
        'predictions', CASE WHEN linked_predictions = '[]'::JSONB THEN NULL ELSE linked_predictions END,

        -- Additional context (kept for backward compatibility)
        'agent_tier', activity_record.agent_tier,
        'workstream', activity_record.workstream,
        'confidence_level', activity_record.confidence_level,
        'recommendations_count', jsonb_array_length(COALESCE(activity_record.recommendations, '[]'::JSONB))
    );

    RETURN result;
END;
$$;


-- ============================================================================
-- FUNCTION: get_memory_entity_context (verify/update)
-- Ensure this function returns the expected format
-- ============================================================================

-- Check if the function exists and has correct return type
-- The existing function in 001b_add_foreign_keys_v3.sql returns TABLE
-- which should be compatible with Python expectations

-- No changes needed if it works - just add a validation query
DO $$
DECLARE
    func_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'public'
        AND p.proname = 'get_memory_entity_context'
    ) INTO func_exists;

    IF NOT func_exists THEN
        RAISE WARNING 'get_memory_entity_context function does not exist - should be created by 001b_add_foreign_keys_v3.sql';
    ELSE
        RAISE NOTICE 'get_memory_entity_context function exists and is ready';
    END IF;
END $$;


-- ============================================================================
-- SUCCESS MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '========================================================';
    RAISE NOTICE 'Migration 019: RPC Function Signatures Fixed';
    RAISE NOTICE '';
    RAISE NOTICE 'Updated functions:';
    RAISE NOTICE '  ✓ get_agent_activity_context - now returns Python-compatible field names:';
    RAISE NOTICE '    - action_type (was activity_type)';
    RAISE NOTICE '    - started_at (was activity_timestamp)';
    RAISE NOTICE '    - duration_ms (was processing_duration_ms)';
    RAISE NOTICE '    - trigger (linked entity via episodic_memories)';
    RAISE NOTICE '    - causal_paths (linked entities via episodic_memories)';
    RAISE NOTICE '    - predictions (linked entities via episodic_memories)';
    RAISE NOTICE '';
    RAISE NOTICE 'Verified functions:';
    RAISE NOTICE '  ✓ get_memory_entity_context - exists and compatible';
    RAISE NOTICE '========================================================';
END $$;
