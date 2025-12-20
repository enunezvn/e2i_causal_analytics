-- ============================================================================
-- E2I AGENTIC MEMORY - ADD FOREIGN KEYS (V3 Schema Compatible)
-- 
-- Matches e2i_ml_complete_v3_schema.sql exactly:
--   - patient_journeys (PK: patient_journey_id VARCHAR(20))
--   - hcp_profiles (PK: hcp_id VARCHAR(20))
--   - treatment_events (PK: treatment_event_id VARCHAR(30))
--   - triggers (PK: trigger_id VARCHAR(30))
--   - ml_predictions (PK: prediction_id VARCHAR(30))
--   - causal_paths (PK: path_id VARCHAR(20))
--   - agent_activities (PK: activity_id VARCHAR(30))
--   - business_metrics (PK: metric_id VARCHAR(50))
-- ============================================================================


-- ============================================================================
-- EPISODIC MEMORIES FOREIGN KEYS
-- ============================================================================

-- FK: episodic_memories -> patient_journeys
ALTER TABLE episodic_memories
ADD CONSTRAINT fk_episodic_patient_journey
FOREIGN KEY (patient_journey_id) 
REFERENCES patient_journeys(patient_journey_id) 
ON DELETE SET NULL;

-- FK: episodic_memories -> hcp_profiles
ALTER TABLE episodic_memories
ADD CONSTRAINT fk_episodic_hcp
FOREIGN KEY (hcp_id) 
REFERENCES hcp_profiles(hcp_id) 
ON DELETE SET NULL;

-- FK: episodic_memories -> treatment_events
ALTER TABLE episodic_memories
ADD CONSTRAINT fk_episodic_treatment
FOREIGN KEY (treatment_event_id) 
REFERENCES treatment_events(treatment_event_id) 
ON DELETE SET NULL;

-- FK: episodic_memories -> triggers
ALTER TABLE episodic_memories
ADD CONSTRAINT fk_episodic_trigger
FOREIGN KEY (trigger_id) 
REFERENCES triggers(trigger_id) 
ON DELETE SET NULL;

-- FK: episodic_memories -> ml_predictions
ALTER TABLE episodic_memories
ADD CONSTRAINT fk_episodic_prediction
FOREIGN KEY (prediction_id) 
REFERENCES ml_predictions(prediction_id) 
ON DELETE SET NULL;

-- FK: episodic_memories -> causal_paths
ALTER TABLE episodic_memories
ADD CONSTRAINT fk_episodic_causal_path
FOREIGN KEY (causal_path_id) 
REFERENCES causal_paths(path_id) 
ON DELETE SET NULL;

-- FK: episodic_memories -> agent_activities
ALTER TABLE episodic_memories
ADD CONSTRAINT fk_episodic_agent_activity
FOREIGN KEY (agent_activity_id) 
REFERENCES agent_activities(activity_id) 
ON DELETE SET NULL;

-- NOTE: No experiments table in V3 schema - experiment_id column exists but no FK


-- ============================================================================
-- SEMANTIC MEMORY CACHE FOREIGN KEYS
-- ============================================================================

-- FK: semantic_memory_cache -> patient_journeys (subject)
ALTER TABLE semantic_memory_cache
ADD CONSTRAINT fk_semantic_subject_patient
FOREIGN KEY (subject_patient_id) 
REFERENCES patient_journeys(patient_journey_id) 
ON DELETE CASCADE;

-- FK: semantic_memory_cache -> patient_journeys (object)
ALTER TABLE semantic_memory_cache
ADD CONSTRAINT fk_semantic_object_patient
FOREIGN KEY (object_patient_id) 
REFERENCES patient_journeys(patient_journey_id) 
ON DELETE CASCADE;

-- FK: semantic_memory_cache -> hcp_profiles (subject)
ALTER TABLE semantic_memory_cache
ADD CONSTRAINT fk_semantic_subject_hcp
FOREIGN KEY (subject_hcp_id) 
REFERENCES hcp_profiles(hcp_id) 
ON DELETE CASCADE;

-- FK: semantic_memory_cache -> hcp_profiles (object)
ALTER TABLE semantic_memory_cache
ADD CONSTRAINT fk_semantic_object_hcp
FOREIGN KEY (object_hcp_id) 
REFERENCES hcp_profiles(hcp_id) 
ON DELETE CASCADE;

-- FK: semantic_memory_cache -> triggers (subject)
ALTER TABLE semantic_memory_cache
ADD CONSTRAINT fk_semantic_subject_trigger
FOREIGN KEY (subject_trigger_id) 
REFERENCES triggers(trigger_id) 
ON DELETE CASCADE;

-- FK: semantic_memory_cache -> triggers (object)
ALTER TABLE semantic_memory_cache
ADD CONSTRAINT fk_semantic_object_trigger
FOREIGN KEY (object_trigger_id) 
REFERENCES triggers(trigger_id) 
ON DELETE CASCADE;

-- FK: semantic_memory_cache -> causal_paths (subject)
ALTER TABLE semantic_memory_cache
ADD CONSTRAINT fk_semantic_subject_causal
FOREIGN KEY (subject_causal_path_id) 
REFERENCES causal_paths(path_id) 
ON DELETE CASCADE;

-- FK: semantic_memory_cache -> causal_paths (object)
ALTER TABLE semantic_memory_cache
ADD CONSTRAINT fk_semantic_object_causal
FOREIGN KEY (object_causal_path_id) 
REFERENCES causal_paths(path_id) 
ON DELETE CASCADE;


-- ============================================================================
-- LEARNING SIGNALS FOREIGN KEYS
-- ============================================================================

-- FK: learning_signals -> patient_journeys
ALTER TABLE learning_signals
ADD CONSTRAINT fk_signals_patient
FOREIGN KEY (related_patient_id) 
REFERENCES patient_journeys(patient_journey_id) 
ON DELETE SET NULL;

-- FK: learning_signals -> hcp_profiles
ALTER TABLE learning_signals
ADD CONSTRAINT fk_signals_hcp
FOREIGN KEY (related_hcp_id) 
REFERENCES hcp_profiles(hcp_id) 
ON DELETE SET NULL;

-- FK: learning_signals -> triggers
ALTER TABLE learning_signals
ADD CONSTRAINT fk_signals_trigger
FOREIGN KEY (related_trigger_id) 
REFERENCES triggers(trigger_id) 
ON DELETE SET NULL;


-- ============================================================================
-- HELPER FUNCTIONS (using exact V3 column names)
-- ============================================================================

-- Function to get full E2I entity context for a memory
CREATE OR REPLACE FUNCTION get_memory_entity_context(p_memory_id UUID)
RETURNS TABLE (
    entity_type VARCHAR,
    entity_id VARCHAR,
    entity_name VARCHAR,
    entity_details JSONB
)
LANGUAGE plpgsql
AS $$
DECLARE
    mem RECORD;
BEGIN
    SELECT * INTO mem FROM episodic_memories WHERE memory_id = p_memory_id;
    
    -- Return patient info if linked
    IF mem.patient_journey_id IS NOT NULL THEN
        RETURN QUERY
        SELECT 
            'patient'::VARCHAR,
            pj.patient_journey_id::VARCHAR,
            CONCAT('Patient ', pj.patient_id)::VARCHAR,
            jsonb_build_object(
                'journey_stage', pj.journey_stage,
                'primary_diagnosis', pj.primary_diagnosis_desc,
                'region', pj.geographic_region,
                'risk_score', pj.risk_score,
                'brand', pj.brand
            )
        FROM patient_journeys pj
        WHERE pj.patient_journey_id = mem.patient_journey_id;
    END IF;
    
    -- Return HCP info if linked
    IF mem.hcp_id IS NOT NULL THEN
        RETURN QUERY
        SELECT 
            'hcp'::VARCHAR,
            hp.hcp_id::VARCHAR,
            CONCAT(hp.first_name, ' ', hp.last_name)::VARCHAR,
            jsonb_build_object(
                'specialty', hp.specialty,
                'priority_tier', hp.priority_tier,
                'region', hp.geographic_region,
                'adoption_category', hp.adoption_category,
                'practice_type', hp.practice_type
            )
        FROM hcp_profiles hp
        WHERE hp.hcp_id = mem.hcp_id;
    END IF;
    
    -- Return trigger info if linked
    IF mem.trigger_id IS NOT NULL THEN
        RETURN QUERY
        SELECT 
            'trigger'::VARCHAR,
            tr.trigger_id::VARCHAR,
            tr.trigger_type::VARCHAR,
            jsonb_build_object(
                'priority', tr.priority,
                'confidence_score', tr.confidence_score,
                'delivery_status', tr.delivery_status,
                'acceptance_status', tr.acceptance_status,
                'outcome_value', tr.outcome_value
            )
        FROM triggers tr
        WHERE tr.trigger_id = mem.trigger_id;
    END IF;
    
    -- Return causal path info if linked
    IF mem.causal_path_id IS NOT NULL THEN
        RETURN QUERY
        SELECT 
            'causal_path'::VARCHAR,
            cp.path_id::VARCHAR,
            CONCAT(cp.start_node, ' → ', cp.end_node)::VARCHAR,
            jsonb_build_object(
                'path_length', cp.path_length,
                'effect_size', cp.causal_effect_size,
                'confidence', cp.confidence_level,
                'method', cp.method_used,
                'business_impact', cp.business_impact_estimate
            )
        FROM causal_paths cp
        WHERE cp.path_id = mem.causal_path_id;
    END IF;
    
    -- Return ml_prediction info if linked
    IF mem.prediction_id IS NOT NULL THEN
        RETURN QUERY
        SELECT 
            'prediction'::VARCHAR,
            mp.prediction_id::VARCHAR,
            CONCAT(mp.prediction_type::TEXT, ' prediction')::VARCHAR,
            jsonb_build_object(
                'model_version', mp.model_version,
                'prediction_value', mp.prediction_value,
                'confidence_score', mp.confidence_score,
                'model_auc', mp.model_auc
            )
        FROM ml_predictions mp
        WHERE mp.prediction_id = mem.prediction_id;
    END IF;
    
    RETURN;
END;
$$;


-- Function to sync HCP-Patient relationships to semantic cache
CREATE OR REPLACE FUNCTION sync_hcp_patient_relationships_to_cache()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    synced_count INTEGER := 0;
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
    
    GET DIAGNOSTICS synced_count = ROW_COUNT;
    
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
    
    RETURN synced_count;
END;
$$;


-- Function to get agent activity context
CREATE OR REPLACE FUNCTION get_agent_activity_context(p_activity_id VARCHAR)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'activity_id', aa.activity_id,
        'agent_name', aa.agent_name,
        'agent_tier', aa.agent_tier,
        'activity_type', aa.activity_type,
        'workstream', aa.workstream,
        'confidence_level', aa.confidence_level,
        'processing_duration_ms', aa.processing_duration_ms,
        'status', aa.status,
        'recommendations_count', jsonb_array_length(aa.recommendations)
    )
    INTO result
    FROM agent_activities aa
    WHERE aa.activity_id = p_activity_id;
    
    RETURN COALESCE(result, '{}'::JSONB);
END;
$$;


-- ============================================================================
-- SUCCESS MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '========================================================';
    RAISE NOTICE 'Foreign keys added successfully!';
    RAISE NOTICE '';
    RAISE NOTICE 'Connected to E2I V3 tables:';
    RAISE NOTICE '  ✓ patient_journeys (patient_journey_id)';
    RAISE NOTICE '  ✓ hcp_profiles (hcp_id)';
    RAISE NOTICE '  ✓ treatment_events (treatment_event_id)';
    RAISE NOTICE '  ✓ triggers (trigger_id)';
    RAISE NOTICE '  ✓ ml_predictions (prediction_id)';
    RAISE NOTICE '  ✓ causal_paths (path_id)';
    RAISE NOTICE '  ✓ agent_activities (activity_id)';
    RAISE NOTICE '';
    RAISE NOTICE 'Functions created:';
    RAISE NOTICE '  - get_memory_entity_context(memory_id)';
    RAISE NOTICE '  - sync_hcp_patient_relationships_to_cache()';
    RAISE NOTICE '  - get_agent_activity_context(activity_id)';
    RAISE NOTICE '========================================================';
END $$;
