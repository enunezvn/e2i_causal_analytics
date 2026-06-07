-- ============================================================================
-- Migration 038: sync_hcp_patient_relationships_to_cache — accumulate both
--                INSERT counts (audit 2026-06-05, L22 / #694)
-- ============================================================================
--
-- WHAT: the function captured `synced_count` via GET DIAGNOSTICS only after the
--       FIRST insert (Patient-HCP) and never added the SECOND insert's ROW_COUNT
--       (HCP-Brand), so the returned count under-reported (logging only, but
--       misleading). Accumulate both. Body is otherwise identical to 001b.
--
-- Idempotent (CREATE OR REPLACE preserves existing grants).
-- ============================================================================

CREATE OR REPLACE FUNCTION sync_hcp_patient_relationships_to_cache()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    synced_count INTEGER := 0;
    v_count INTEGER := 0;
BEGIN
    -- Sync HCP-Patient treatment relationships from treatment_events
    INSERT INTO semantic_memory_cache (
        subject_type, subject_id, subject_patient_id,
        predicate,
        object_type, object_id, object_hcp_id,
        confidence, source
    )
    SELECT DISTINCT
        'Patient', te.patient_id, te.patient_journey_id,
        'TREATED_BY',
        'HCP', te.hcp_id, te.hcp_id,
        1.0,
        'data_layer_sync'
    FROM treatment_events te
    WHERE te.patient_journey_id IS NOT NULL
      AND te.hcp_id IS NOT NULL
    ON CONFLICT (subject_type, subject_id, predicate, object_type, object_id)
    DO UPDATE SET updated_at = NOW();

    GET DIAGNOSTICS v_count = ROW_COUNT;
    synced_count := synced_count + v_count;

    -- Sync HCP-Brand prescribing relationships
    INSERT INTO semantic_memory_cache (
        subject_type, subject_id, subject_hcp_id,
        predicate,
        object_type, object_id,
        confidence, source
    )
    SELECT DISTINCT
        'HCP', te.hcp_id, te.hcp_id,
        'PRESCRIBES',
        'Brand', te.brand::TEXT,
        1.0,
        'data_layer_sync'
    FROM treatment_events te
    WHERE te.hcp_id IS NOT NULL
      AND te.brand IS NOT NULL
      AND te.event_type = 'prescription'
    ON CONFLICT (subject_type, subject_id, predicate, object_type, object_id)
    DO UPDATE SET updated_at = NOW();

    -- L22 (#694): accumulate the second insert's ROW_COUNT (was dropped before).
    GET DIAGNOSTICS v_count = ROW_COUNT;
    synced_count := synced_count + v_count;

    RETURN synced_count;
END;
$$;
