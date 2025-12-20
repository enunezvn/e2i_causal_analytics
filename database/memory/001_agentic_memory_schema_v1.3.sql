-- ============================================================================
-- E2I AGENTIC MEMORY SCHEMA v1.3
-- STANDALONE VERSION - No Foreign Key Dependencies
-- 
-- This version can be run BEFORE the E2I data layer tables exist.
-- Foreign keys are added separately via 001b_add_foreign_keys.sql
-- ============================================================================

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- ENUM TYPES
-- ============================================================================

-- Event types for episodic memory
DO $$ BEGIN
    CREATE TYPE memory_event_type AS ENUM (
        'user_query',
        'agent_action', 
        'system_event',
        'feedback',
        'error',
        'causal_discovery',
        'trigger_generated',
        'experiment_completed'
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- Outcome types
DO $$ BEGIN
    CREATE TYPE memory_outcome_type AS ENUM (
        'success',
        'partial_success',
        'failure',
        'pending',
        'escalated'
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- Procedure types
DO $$ BEGIN
    CREATE TYPE procedure_type AS ENUM (
        'tool_sequence',
        'query_pattern',
        'error_recovery',
        'optimization',
        'causal_chain_traversal'
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- Cognitive phase types
DO $$ BEGIN
    CREATE TYPE cognitive_phase AS ENUM (
        'summarizer',
        'investigator',
        'agent',
        'reflector'
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- Agent enum (11-agent architecture)
DO $$ BEGIN
    CREATE TYPE e2i_agent_name AS ENUM (
        -- Tier 1: Coordination
        'orchestrator',
        -- Tier 2: Causal Analytics
        'causal_impact',
        'gap_analyzer', 
        'drift_monitor',
        'heterogeneous_optimizer',
        -- Tier 3: Monitoring
        'fairness_guardian',
        'health_score',
        -- Tier 4: ML Predictions
        'experiment_designer',
        'prediction_synthesizer',
        -- Tier 5: Self-Improvement
        'feedback_learner',
        'explainer',
        -- Supporting
        'resource_optimizer'
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- Learning signal types
DO $$ BEGIN
    CREATE TYPE learning_signal_type AS ENUM (
        'thumbs_up',
        'thumbs_down',
        'correction',
        'rating',
        'implicit_positive',
        'implicit_negative'
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;


-- ============================================================================
-- TABLE 1: EPISODIC MEMORIES
-- Stores experiences: "What did I do?" "What happened yesterday?"
-- E2I entity columns are VARCHAR without FK constraints (added later)
-- ============================================================================

CREATE TABLE IF NOT EXISTS episodic_memories (
    -- Primary key
    memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Session linkage (working memory)
    session_id UUID,
    cycle_id UUID,  -- References cognitive_cycles (internal)
    
    -- Event classification
    event_type memory_event_type NOT NULL,
    event_subtype VARCHAR(100),
    
    -- Content
    description TEXT NOT NULL,
    raw_content JSONB DEFAULT '{}',
    
    -- =========================================
    -- E2I DATA LAYER REFERENCES (NO FK YET)
    -- These columns store IDs that can be linked
    -- to E2I data layer tables via FKs later
    -- =========================================
    
    -- Patient context (nullable)
    patient_journey_id VARCHAR(50),
    patient_id VARCHAR(50),  -- Denormalized for quick filtering
    
    -- HCP context (nullable)
    hcp_id VARCHAR(50),
    
    -- Treatment event context (nullable)
    treatment_event_id VARCHAR(50),
    
    -- Trigger context (nullable)
    trigger_id VARCHAR(50),
    
    -- ML prediction context (nullable)
    prediction_id VARCHAR(50),
    
    -- Causal path context (nullable)
    causal_path_id VARCHAR(50),
    
    -- Experiment context (nullable)
    experiment_id VARCHAR(50),
    
    -- Agent activity linkage (nullable)
    agent_activity_id VARCHAR(50),
    
    -- =========================================
    -- FLEXIBLE ENTITY REFERENCES (JSONB)
    -- For entities not covered by columns above
    -- =========================================
    entities JSONB DEFAULT '{}',
    -- Example: {"brands": ["kisqali"], "regions": ["northeast"], "kpis": ["trigger_precision"]}
    
    -- =========================================
    -- MEMORY METADATA
    -- =========================================
    
    -- Outcome tracking
    outcome_type memory_outcome_type,
    outcome_details JSONB DEFAULT '{}',
    
    -- User satisfaction (1-5 scale, null if not rated)
    user_satisfaction_score SMALLINT CHECK (user_satisfaction_score BETWEEN 1 AND 5),
    
    -- Which agent created this memory
    agent_name e2i_agent_name,
    
    -- Brand and region context (denormalized for filtering)
    brand VARCHAR(50),  -- Remibrutinib, Fabhalta, Kisqali
    region VARCHAR(20), -- northeast, south, midwest, west
    
    -- Vector embedding for semantic search (OpenAI ada-002 = 1536 dims)
    embedding vector(1536),
    
    -- Full-text search
    search_text TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(description, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(event_subtype, '')), 'B')
    ) STORED,
    
    -- Timestamps
    occurred_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Importance score (calculated by reflector)
    importance_score FLOAT DEFAULT 0.5 CHECK (importance_score BETWEEN 0 AND 1)
);

-- Indexes for episodic memories
CREATE INDEX IF NOT EXISTS idx_episodic_embedding ON episodic_memories 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_episodic_session ON episodic_memories(session_id);
CREATE INDEX IF NOT EXISTS idx_episodic_event_type ON episodic_memories(event_type);
CREATE INDEX IF NOT EXISTS idx_episodic_occurred ON episodic_memories(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_agent ON episodic_memories(agent_name);
CREATE INDEX IF NOT EXISTS idx_episodic_search ON episodic_memories USING GIN(search_text);
CREATE INDEX IF NOT EXISTS idx_episodic_entities ON episodic_memories USING GIN(entities);

-- E2I entity indexes for efficient filtering (even without FKs)
CREATE INDEX IF NOT EXISTS idx_episodic_patient ON episodic_memories(patient_journey_id) WHERE patient_journey_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_episodic_patient_id ON episodic_memories(patient_id) WHERE patient_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_episodic_hcp ON episodic_memories(hcp_id) WHERE hcp_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_episodic_trigger ON episodic_memories(trigger_id) WHERE trigger_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_episodic_causal_path ON episodic_memories(causal_path_id) WHERE causal_path_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_episodic_brand_region ON episodic_memories(brand, region);


-- ============================================================================
-- TABLE 2: PROCEDURAL MEMORIES
-- Stores skills: "How did I solve this error last time?"
-- ============================================================================

CREATE TABLE IF NOT EXISTS procedural_memories (
    -- Primary key
    procedure_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Procedure identification
    procedure_name VARCHAR(200) NOT NULL,
    procedure_type procedure_type NOT NULL DEFAULT 'tool_sequence',
    
    -- The actual procedure (ordered tool calls)
    tool_sequence JSONB NOT NULL,
    
    -- Trigger pattern (what query pattern suggests this procedure)
    trigger_pattern TEXT,
    trigger_embedding vector(1536),
    
    -- Intent association
    intent_keywords TEXT[],
    detected_intent VARCHAR(50),  -- causal, trend, comparison, optimization, experiment
    
    -- =========================================
    -- E2I CONTEXT (no FKs needed)
    -- =========================================
    
    -- Which brands/regions this procedure works well for
    applicable_brands TEXT[] DEFAULT ARRAY['all'],
    applicable_regions TEXT[] DEFAULT ARRAY['all'],
    
    -- Which agent types typically use this procedure
    applicable_agents e2i_agent_name[],
    
    -- =========================================
    -- PERFORMANCE TRACKING
    -- =========================================
    
    usage_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    
    success_rate FLOAT GENERATED ALWAYS AS (
        CASE WHEN usage_count > 0 THEN success_count::FLOAT / usage_count ELSE 0 END
    ) STORED,
    
    avg_execution_time_ms INTEGER,
    avg_quality_score FLOAT,
    
    -- Version control
    version INTEGER DEFAULT 1,
    parent_procedure_id UUID REFERENCES procedural_memories(procedure_id),
    
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

-- Indexes for procedural memories
CREATE INDEX IF NOT EXISTS idx_procedural_trigger_embedding ON procedural_memories 
    USING ivfflat (trigger_embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX IF NOT EXISTS idx_procedural_type ON procedural_memories(procedure_type);
CREATE INDEX IF NOT EXISTS idx_procedural_intent ON procedural_memories(detected_intent);
CREATE INDEX IF NOT EXISTS idx_procedural_active ON procedural_memories(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_procedural_success ON procedural_memories(success_rate DESC) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_procedural_keywords ON procedural_memories USING GIN(intent_keywords);
CREATE INDEX IF NOT EXISTS idx_procedural_brands ON procedural_memories USING GIN(applicable_brands);


-- ============================================================================
-- TABLE 3: SEMANTIC MEMORY CACHE
-- Hot cache of FalkorDB graph triplets
-- NO foreign keys - stores IDs as strings for flexibility
-- ============================================================================

CREATE TABLE IF NOT EXISTS semantic_memory_cache (
    -- Primary key
    cache_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Triplet structure (Subject -[Predicate]-> Object)
    subject_type VARCHAR(50) NOT NULL,
    subject_id VARCHAR(100) NOT NULL,
    
    predicate VARCHAR(50) NOT NULL,
    
    object_type VARCHAR(50) NOT NULL,
    object_id VARCHAR(100) NOT NULL,
    
    -- =========================================
    -- E2I ENTITY ID REFERENCES (NO FKs)
    -- Store IDs for optional joining to data layer
    -- =========================================
    
    subject_patient_id VARCHAR(50),
    subject_hcp_id VARCHAR(50),
    subject_trigger_id VARCHAR(50),
    subject_causal_path_id VARCHAR(50),
    
    object_patient_id VARCHAR(50),
    object_hcp_id VARCHAR(50),
    object_trigger_id VARCHAR(50),
    object_causal_path_id VARCHAR(50),
    
    -- =========================================
    -- TRIPLET METADATA
    -- =========================================
    
    confidence FLOAT DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
    source VARCHAR(50),  -- 'graphity_extraction', 'user_stated', 'causal_discovery', 'data_layer_sync'
    properties JSONB DEFAULT '{}',
    
    -- FalkorDB sync metadata
    falkordb_synced BOOLEAN DEFAULT FALSE,
    falkordb_sync_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Unique constraint on triplet
    UNIQUE(subject_type, subject_id, predicate, object_type, object_id)
);

-- Indexes for semantic cache
CREATE INDEX IF NOT EXISTS idx_semantic_subject ON semantic_memory_cache(subject_type, subject_id);
CREATE INDEX IF NOT EXISTS idx_semantic_object ON semantic_memory_cache(object_type, object_id);
CREATE INDEX IF NOT EXISTS idx_semantic_predicate ON semantic_memory_cache(predicate);
CREATE INDEX IF NOT EXISTS idx_semantic_confidence ON semantic_memory_cache(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_subject_patient ON semantic_memory_cache(subject_patient_id) WHERE subject_patient_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_semantic_subject_hcp ON semantic_memory_cache(subject_hcp_id) WHERE subject_hcp_id IS NOT NULL;


-- ============================================================================
-- TABLE 4: COGNITIVE CYCLES
-- Tracks execution of 4-phase cognitive workflow
-- ============================================================================

CREATE TABLE IF NOT EXISTS cognitive_cycles (
    -- Primary key
    cycle_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    
    -- User context
    user_id VARCHAR(100),
    
    -- Input
    user_query TEXT NOT NULL,
    query_embedding vector(1536),
    detected_intent VARCHAR(50),
    detected_entities JSONB DEFAULT '{}',
    
    -- =========================================
    -- E2I ENTITIES INVOLVED (as arrays, no FKs)
    -- =========================================
    
    involved_patient_ids TEXT[],
    involved_hcp_ids TEXT[],
    involved_trigger_ids TEXT[],
    involved_causal_path_ids TEXT[],
    brands_discussed TEXT[],
    regions_discussed TEXT[],
    
    -- =========================================
    -- PHASE TRACKING
    -- =========================================
    
    current_phase cognitive_phase DEFAULT 'summarizer',
    
    -- Phase 1: Summarizer
    phase1_started_at TIMESTAMPTZ,
    phase1_completed_at TIMESTAMPTZ,
    context_compressed BOOLEAN DEFAULT FALSE,
    compression_ratio FLOAT,
    
    -- Phase 2: Investigator
    phase2_started_at TIMESTAMPTZ,
    phase2_completed_at TIMESTAMPTZ,
    hops_executed INTEGER DEFAULT 0,
    evidence_collected INTEGER DEFAULT 0,
    investigation_decision TEXT,
    
    -- Phase 3: Agent
    phase3_started_at TIMESTAMPTZ,
    phase3_completed_at TIMESTAMPTZ,
    agents_invoked e2i_agent_name[],
    agent_outputs JSONB DEFAULT '{}',
    
    -- Phase 4: Reflector
    phase4_started_at TIMESTAMPTZ,
    phase4_completed_at TIMESTAMPTZ,
    facts_extracted INTEGER DEFAULT 0,
    procedures_learned INTEGER DEFAULT 0,
    
    -- =========================================
    -- OUTPUT
    -- =========================================
    
    synthesized_response TEXT,
    confidence_score FLOAT CHECK (confidence_score BETWEEN 0 AND 1),
    visualization_config JSONB,
    
    -- Overall status
    status VARCHAR(20) DEFAULT 'running',
    error_message TEXT,
    
    -- Timestamps
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    total_duration_ms INTEGER
);

-- Indexes for cognitive cycles
CREATE INDEX IF NOT EXISTS idx_cycles_session ON cognitive_cycles(session_id);
CREATE INDEX IF NOT EXISTS idx_cycles_user ON cognitive_cycles(user_id);
CREATE INDEX IF NOT EXISTS idx_cycles_status ON cognitive_cycles(status);
CREATE INDEX IF NOT EXISTS idx_cycles_started ON cognitive_cycles(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_cycles_intent ON cognitive_cycles(detected_intent);
CREATE INDEX IF NOT EXISTS idx_cycles_embedding ON cognitive_cycles 
    USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX IF NOT EXISTS idx_cycles_patients ON cognitive_cycles USING GIN(involved_patient_ids);
CREATE INDEX IF NOT EXISTS idx_cycles_hcps ON cognitive_cycles USING GIN(involved_hcp_ids);
CREATE INDEX IF NOT EXISTS idx_cycles_brands ON cognitive_cycles USING GIN(brands_discussed);


-- ============================================================================
-- TABLE 5: INVESTIGATION HOPS
-- Detailed tracking of each hop in the investigator phase
-- ============================================================================

CREATE TABLE IF NOT EXISTS investigation_hops (
    -- Primary key
    hop_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cycle_id UUID NOT NULL REFERENCES cognitive_cycles(cycle_id) ON DELETE CASCADE,
    
    -- Hop sequence
    hop_number INTEGER NOT NULL,
    
    -- Query details
    memory_type VARCHAR(20) NOT NULL,
    query_type VARCHAR(50),
    query_details JSONB,
    
    -- Results
    results_count INTEGER DEFAULT 0,
    results_summary JSONB,
    top_result_ids TEXT[],
    
    -- E2I entities found (no FKs)
    found_patient_ids TEXT[],
    found_hcp_ids TEXT[],
    found_trigger_ids TEXT[],
    found_causal_path_ids TEXT[],
    
    -- Relevance assessment
    relevance_score FLOAT CHECK (relevance_score BETWEEN 0 AND 1),
    contributes_to_answer BOOLEAN,
    
    -- Performance
    execution_time_ms INTEGER,
    
    -- Timestamps
    executed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for investigation hops
CREATE INDEX IF NOT EXISTS idx_hops_cycle ON investigation_hops(cycle_id);
CREATE INDEX IF NOT EXISTS idx_hops_memory_type ON investigation_hops(memory_type);
CREATE INDEX IF NOT EXISTS idx_hops_relevance ON investigation_hops(relevance_score DESC);


-- ============================================================================
-- TABLE 6: LEARNING SIGNALS
-- Captures feedback for DSPy optimization
-- ============================================================================

CREATE TABLE IF NOT EXISTS learning_signals (
    -- Primary key
    signal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Context
    cycle_id UUID REFERENCES cognitive_cycles(cycle_id) ON DELETE CASCADE,
    session_id UUID,
    
    -- Signal details
    signal_type learning_signal_type NOT NULL,
    signal_value FLOAT,
    signal_details JSONB DEFAULT '{}',
    
    -- What the signal applies to
    applies_to_type VARCHAR(50),
    applies_to_id VARCHAR(100),
    
    -- =========================================
    -- E2I CONTEXT (NO FKs - store IDs only)
    -- =========================================
    
    related_patient_id VARCHAR(50),
    related_hcp_id VARCHAR(50),
    related_trigger_id VARCHAR(50),
    
    brand VARCHAR(50),
    region VARCHAR(20),
    
    rated_agent e2i_agent_name,
    
    -- =========================================
    -- DSPY TRAINING
    -- =========================================
    
    is_training_example BOOLEAN DEFAULT FALSE,
    dspy_metric_name VARCHAR(100),
    dspy_metric_value FLOAT,
    training_input TEXT,
    training_output TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for learning signals
CREATE INDEX IF NOT EXISTS idx_signals_cycle ON learning_signals(cycle_id);
CREATE INDEX IF NOT EXISTS idx_signals_type ON learning_signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_signals_training ON learning_signals(is_training_example) WHERE is_training_example = TRUE;
CREATE INDEX IF NOT EXISTS idx_signals_agent ON learning_signals(rated_agent);
CREATE INDEX IF NOT EXISTS idx_signals_brand_region ON learning_signals(brand, region);
CREATE INDEX IF NOT EXISTS idx_signals_dspy_metric ON learning_signals(dspy_metric_name) WHERE dspy_metric_name IS NOT NULL;


-- ============================================================================
-- TABLE 7: MEMORY STATISTICS
-- Aggregated metrics for monitoring memory system health
-- ============================================================================

CREATE TABLE IF NOT EXISTS memory_statistics (
    -- Primary key
    stat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Time window
    stat_date DATE NOT NULL,
    stat_hour INTEGER,
    
    -- Memory counts
    episodic_count INTEGER DEFAULT 0,
    procedural_count INTEGER DEFAULT 0,
    semantic_cache_count INTEGER DEFAULT 0,
    
    -- Usage metrics
    cycles_completed INTEGER DEFAULT 0,
    avg_cycle_duration_ms INTEGER,
    avg_investigation_hops FLOAT,
    
    -- Performance metrics
    cache_hit_rate FLOAT,
    avg_episodic_search_time_ms INTEGER,
    avg_semantic_query_time_ms INTEGER,
    
    -- Quality metrics
    avg_confidence_score FLOAT,
    avg_user_satisfaction FLOAT,
    positive_feedback_count INTEGER DEFAULT 0,
    negative_feedback_count INTEGER DEFAULT 0,
    
    -- E2I-specific metrics
    unique_patients_referenced INTEGER,
    unique_hcps_referenced INTEGER,
    unique_triggers_referenced INTEGER,
    brand_distribution JSONB,
    region_distribution JSONB,
    agent_invocation_counts JSONB,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(stat_date, stat_hour)
);

CREATE INDEX IF NOT EXISTS idx_stats_date ON memory_statistics(stat_date DESC);


-- ============================================================================
-- HELPER FUNCTIONS (STANDALONE - no data layer dependencies)
-- ============================================================================

-- Function to search episodic memory with E2I entity filters
CREATE OR REPLACE FUNCTION search_episodic_memory(
    query_embedding vector(1536),
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 10,
    filter_event_type memory_event_type DEFAULT NULL,
    filter_agent e2i_agent_name DEFAULT NULL,
    filter_brand VARCHAR DEFAULT NULL,
    filter_region VARCHAR DEFAULT NULL,
    filter_patient_id VARCHAR DEFAULT NULL,
    filter_hcp_id VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    memory_id UUID,
    event_type memory_event_type,
    description TEXT,
    entities JSONB,
    patient_journey_id VARCHAR,
    patient_id VARCHAR,
    hcp_id VARCHAR,
    trigger_id VARCHAR,
    agent_name e2i_agent_name,
    brand VARCHAR,
    region VARCHAR,
    occurred_at TIMESTAMPTZ,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        em.memory_id,
        em.event_type,
        em.description,
        em.entities,
        em.patient_journey_id,
        em.patient_id,
        em.hcp_id,
        em.trigger_id,
        em.agent_name,
        em.brand,
        em.region,
        em.occurred_at,
        1 - (em.embedding <=> query_embedding) AS similarity
    FROM episodic_memories em
    WHERE 
        1 - (em.embedding <=> query_embedding) > match_threshold
        AND (filter_event_type IS NULL OR em.event_type = filter_event_type)
        AND (filter_agent IS NULL OR em.agent_name = filter_agent)
        AND (filter_brand IS NULL OR em.brand = filter_brand)
        AND (filter_region IS NULL OR em.region = filter_region)
        AND (filter_patient_id IS NULL OR em.patient_id = filter_patient_id)
        AND (filter_hcp_id IS NULL OR em.hcp_id = filter_hcp_id)
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;


-- Function to find relevant procedures
CREATE OR REPLACE FUNCTION find_relevant_procedures(
    query_embedding vector(1536),
    match_threshold FLOAT DEFAULT 0.6,
    match_count INT DEFAULT 5,
    filter_type procedure_type DEFAULT NULL,
    filter_intent VARCHAR DEFAULT NULL,
    filter_brand VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    procedure_id UUID,
    procedure_name VARCHAR,
    procedure_type procedure_type,
    tool_sequence JSONB,
    trigger_pattern TEXT,
    usage_count INTEGER,
    success_count INTEGER,
    success_rate FLOAT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pm.procedure_id,
        pm.procedure_name,
        pm.procedure_type,
        pm.tool_sequence,
        pm.trigger_pattern,
        pm.usage_count,
        pm.success_count,
        pm.success_rate,
        1 - (pm.trigger_embedding <=> query_embedding) AS similarity
    FROM procedural_memories pm
    WHERE 
        pm.is_active = TRUE
        AND 1 - (pm.trigger_embedding <=> query_embedding) > match_threshold
        AND (filter_type IS NULL OR pm.procedure_type = filter_type)
        AND (filter_intent IS NULL OR pm.detected_intent = filter_intent)
        AND (filter_brand IS NULL OR filter_brand = ANY(pm.applicable_brands) OR 'all' = ANY(pm.applicable_brands))
    ORDER BY 
        similarity * (0.5 + 0.5 * pm.success_rate) DESC
    LIMIT match_count;
END;
$$;


-- Function to get E2I entity IDs for a memory (STANDALONE - just returns IDs, no joins)
CREATE OR REPLACE FUNCTION get_memory_e2i_ids(p_memory_id UUID)
RETURNS TABLE (
    entity_type VARCHAR,
    entity_id VARCHAR
)
LANGUAGE plpgsql
AS $$
DECLARE
    mem RECORD;
BEGIN
    SELECT * INTO mem FROM episodic_memories WHERE memory_id = p_memory_id;
    
    IF mem.patient_journey_id IS NOT NULL THEN
        RETURN QUERY SELECT 'patient_journey'::VARCHAR, mem.patient_journey_id;
    END IF;
    
    IF mem.patient_id IS NOT NULL THEN
        RETURN QUERY SELECT 'patient'::VARCHAR, mem.patient_id;
    END IF;
    
    IF mem.hcp_id IS NOT NULL THEN
        RETURN QUERY SELECT 'hcp'::VARCHAR, mem.hcp_id;
    END IF;
    
    IF mem.trigger_id IS NOT NULL THEN
        RETURN QUERY SELECT 'trigger'::VARCHAR, mem.trigger_id;
    END IF;
    
    IF mem.causal_path_id IS NOT NULL THEN
        RETURN QUERY SELECT 'causal_path'::VARCHAR, mem.causal_path_id;
    END IF;
    
    IF mem.prediction_id IS NOT NULL THEN
        RETURN QUERY SELECT 'prediction'::VARCHAR, mem.prediction_id;
    END IF;
    
    IF mem.experiment_id IS NOT NULL THEN
        RETURN QUERY SELECT 'experiment'::VARCHAR, mem.experiment_id;
    END IF;
    
    RETURN;
END;
$$;


-- ============================================================================
-- GRANTS
-- ============================================================================

GRANT SELECT, INSERT, UPDATE ON episodic_memories TO authenticated;
GRANT SELECT, INSERT, UPDATE ON procedural_memories TO authenticated;
GRANT SELECT, INSERT, UPDATE ON semantic_memory_cache TO authenticated;
GRANT SELECT, INSERT, UPDATE ON cognitive_cycles TO authenticated;
GRANT SELECT, INSERT ON investigation_hops TO authenticated;
GRANT SELECT, INSERT ON learning_signals TO authenticated;
GRANT SELECT, INSERT, UPDATE ON memory_statistics TO authenticated;


-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE episodic_memories IS 'Long-term episodic memory. E2I entity columns store IDs without FK constraints (add via 001b script).';
COMMENT ON TABLE procedural_memories IS 'Long-term procedural memory storing successful tool call sequences.';
COMMENT ON TABLE semantic_memory_cache IS 'Hot cache of FalkorDB semantic graph triplets.';
COMMENT ON TABLE cognitive_cycles IS 'Tracks 4-phase cognitive workflow execution.';
COMMENT ON TABLE investigation_hops IS 'Detailed hop-by-hop tracking for investigation phase.';
COMMENT ON TABLE learning_signals IS 'Feedback signals for DSPy optimization.';
COMMENT ON TABLE memory_statistics IS 'Aggregated memory system metrics.';


-- ============================================================================
-- SUCCESS MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '========================================================';
    RAISE NOTICE 'E2I Agentic Memory Schema v1.3 installed successfully!';
    RAISE NOTICE '';
    RAISE NOTICE 'Tables created:';
    RAISE NOTICE '  - episodic_memories';
    RAISE NOTICE '  - procedural_memories';
    RAISE NOTICE '  - semantic_memory_cache';
    RAISE NOTICE '  - cognitive_cycles';
    RAISE NOTICE '  - investigation_hops';
    RAISE NOTICE '  - learning_signals';
    RAISE NOTICE '  - memory_statistics';
    RAISE NOTICE '';
    RAISE NOTICE 'NEXT STEP: Once your E2I data layer tables exist,';
    RAISE NOTICE 'run 001b_add_foreign_keys.sql to add FK constraints.';
    RAISE NOTICE '========================================================';
END $$;
