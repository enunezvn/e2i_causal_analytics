-- ============================================================================
-- E2I AGENTIC MEMORY SCHEMA
-- Tri-Memory Architecture for Self-Improving NLV + RAG
-- Version: 1.0
-- Compatible with: Supabase (PostgreSQL 15+) with pgvector
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For fuzzy text search

-- ============================================================================
-- MEMORY TYPE ENUMS
-- ============================================================================

CREATE TYPE memory_type AS ENUM (
    'episodic',      -- What happened (events, interactions)
    'procedural',    -- How to do things (tool sequences, patterns)
    'semantic',      -- Facts and relationships (cached from graph)
    'working'        -- Active session context
);

CREATE TYPE evidence_source AS ENUM (
    'supabase_query',
    'graph_traversal', 
    'vector_search',
    'agent_output',
    'user_feedback',
    'causal_analysis'
);

CREATE TYPE cognitive_phase AS ENUM (
    'summarizer',    -- Phase 1: Context pruning
    'investigator',  -- Phase 2: Multi-hop retrieval
    'agent',         -- Phase 3: Synthesis & action
    'reflector'      -- Phase 4: Learning & self-update
);

-- ============================================================================
-- 1. EPISODIC MEMORY
-- Stores: What happened, when, with what outcome
-- Backend: Supabase/PostgreSQL + pgvector
-- ============================================================================

CREATE TABLE episodic_memories (
    memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Temporal context
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id UUID,  -- Links to working memory session
    
    -- Event classification
    event_type VARCHAR(50) NOT NULL,  -- 'user_query', 'agent_action', 'system_event', 'feedback'
    event_subtype VARCHAR(100),
    
    -- Core content
    description TEXT NOT NULL,  -- Natural language summary
    raw_content JSONB,  -- Original structured data
    
    -- Entity references (links to E2I core schema)
    entities JSONB DEFAULT '{}',  -- {"patient_ids": [], "hcp_ids": [], "brands": [], "regions": []}
    
    -- Outcome tracking
    outcome_type VARCHAR(50),  -- 'success', 'partial', 'failure', 'pending'
    outcome_details JSONB,
    user_satisfaction_score FLOAT,  -- 1-5 from feedback, NULL if no feedback
    
    -- Vector embedding for semantic search
    embedding vector(1536),  -- OpenAI ada-002 or similar
    
    -- Metadata
    agent_name VARCHAR(50),  -- Which agent created/processed this
    data_split VARCHAR(10) DEFAULT 'train',  -- ML split tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(description, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(event_type, '')), 'B')
    ) STORED
);

-- Indexes for episodic memory
CREATE INDEX idx_episodic_occurred_at ON episodic_memories(occurred_at DESC);
CREATE INDEX idx_episodic_event_type ON episodic_memories(event_type);
CREATE INDEX idx_episodic_session ON episodic_memories(session_id);
CREATE INDEX idx_episodic_agent ON episodic_memories(agent_name);
CREATE INDEX idx_episodic_embedding ON episodic_memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_episodic_search ON episodic_memories USING gin(search_vector);
CREATE INDEX idx_episodic_entities ON episodic_memories USING gin(entities);

-- ============================================================================
-- 2. PROCEDURAL MEMORY
-- Stores: How to do things (successful tool call sequences)
-- Backend: Supabase/PostgreSQL + pgvector
-- ============================================================================

CREATE TABLE procedural_memories (
    procedure_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Classification
    procedure_name VARCHAR(200) NOT NULL,  -- Human-readable name
    procedure_type VARCHAR(50) NOT NULL,  -- 'tool_sequence', 'query_pattern', 'causal_analysis'
    
    -- The procedure itself
    tool_sequence JSONB NOT NULL,  -- Ordered list of tool calls with parameters
    -- Example: [{"tool": "causal_impact_agent", "action": "trace_chain", "params": {...}}, ...]
    
    -- Trigger conditions (when to use this procedure)
    trigger_pattern TEXT,  -- Regex or semantic pattern that triggers this
    trigger_embedding vector(1536),  -- For semantic matching of similar queries
    intent_keywords TEXT[],  -- Keywords that suggest this procedure
    
    -- Performance tracking
    usage_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    avg_execution_time_ms FLOAT,
    avg_quality_score FLOAT,  -- From LLM-as-Judge or user feedback
    
    -- Versioning
    version INTEGER DEFAULT 1,
    parent_procedure_id UUID REFERENCES procedural_memories(procedure_id),
    is_active BOOLEAN DEFAULT true,
    
    -- Context requirements
    required_context JSONB,  -- What must be in working memory to use this
    produces_context JSONB,  -- What this adds to working memory
    
    -- Metadata
    discovered_from_session UUID,  -- Which session led to learning this
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    data_split VARCHAR(10) DEFAULT 'train'
);

-- Indexes for procedural memory
CREATE INDEX idx_procedural_type ON procedural_memories(procedure_type);
CREATE INDEX idx_procedural_active ON procedural_memories(is_active) WHERE is_active = true;
CREATE INDEX idx_procedural_trigger_embed ON procedural_memories USING ivfflat (trigger_embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX idx_procedural_keywords ON procedural_memories USING gin(intent_keywords);
CREATE INDEX idx_procedural_success_rate ON procedural_memories((success_count::float / NULLIF(usage_count, 0))) WHERE usage_count > 5;

-- ============================================================================
-- 3. SEMANTIC MEMORY CACHE
-- Stores: Facts and relationships (synced from FalkorDB graph)
-- Note: Primary semantic store is FalkorDB; this is a hot cache for fast retrieval
-- ============================================================================

CREATE TABLE semantic_memory_cache (
    cache_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Triple representation
    subject_type VARCHAR(50) NOT NULL,  -- 'patient', 'hcp', 'brand', 'region', 'kpi'
    subject_id VARCHAR(100) NOT NULL,
    predicate VARCHAR(100) NOT NULL,  -- Relationship type
    object_type VARCHAR(50) NOT NULL,
    object_id VARCHAR(100) NOT NULL,
    
    -- Denormalized for query performance
    subject_label TEXT,  -- Human-readable subject name
    object_label TEXT,   -- Human-readable object name
    
    -- Relationship metadata
    relationship_weight FLOAT DEFAULT 1.0,
    confidence FLOAT DEFAULT 1.0,
    evidence_count INTEGER DEFAULT 1,
    
    -- Source tracking
    source_graph_id VARCHAR(100),  -- ID in FalkorDB
    last_synced_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Embedding for semantic search across relationships
    embedding vector(1536),
    
    -- Validity
    valid_from TIMESTAMPTZ DEFAULT NOW(),
    valid_until TIMESTAMPTZ,  -- NULL = still valid
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for semantic cache
CREATE INDEX idx_semantic_subject ON semantic_memory_cache(subject_type, subject_id);
CREATE INDEX idx_semantic_object ON semantic_memory_cache(object_type, object_id);
CREATE INDEX idx_semantic_predicate ON semantic_memory_cache(predicate);
CREATE INDEX idx_semantic_embedding ON semantic_memory_cache USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
CREATE UNIQUE INDEX idx_semantic_triple ON semantic_memory_cache(subject_type, subject_id, predicate, object_type, object_id) 
    WHERE valid_until IS NULL;

-- ============================================================================
-- 4. WORKING MEMORY (Session State)
-- Stores: Active context, evidence board, conversation state
-- Backend: Redis in production, PostgreSQL for local pilot
-- ============================================================================

CREATE TABLE working_memory_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Session lifecycle
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    
    -- User context
    user_id VARCHAR(100),
    user_preferences JSONB DEFAULT '{}',
    
    -- Conversation state
    conversation_summary TEXT,  -- Compressed summary of older messages
    message_count INTEGER DEFAULT 0,
    
    -- Current reasoning context
    current_goal TEXT,
    current_phase cognitive_phase DEFAULT 'summarizer',
    
    -- Evidence board (accumulated during investigation)
    evidence_trail JSONB DEFAULT '[]',  -- Array of evidence items
    -- Example: [{"hop": 1, "source": "episodic", "content": "...", "relevance": 0.9}, ...]
    
    -- Active entities in focus
    active_entities JSONB DEFAULT '{}',  -- Currently relevant patient_ids, hcp_ids, etc.
    
    -- Filters and constraints
    active_filters JSONB DEFAULT '{}',  -- brand, region, date_range
    
    -- Performance tracking
    total_tool_calls INTEGER DEFAULT 0,
    total_hops INTEGER DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Session messages (lightweight conversation history)
CREATE TABLE working_memory_messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES working_memory_sessions(session_id) ON DELETE CASCADE,
    
    role VARCHAR(20) NOT NULL,  -- 'user', 'assistant', 'system', 'tool'
    content TEXT NOT NULL,
    
    -- For tool messages
    tool_name VARCHAR(50),
    tool_input JSONB,
    tool_output JSONB,
    
    -- Metadata
    token_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Position for ordering
    sequence_number INTEGER NOT NULL
);

CREATE INDEX idx_wm_messages_session ON working_memory_messages(session_id, sequence_number);

-- ============================================================================
-- 5. COGNITIVE CYCLE TRACKING
-- Tracks the 4-phase processing of each user interaction
-- ============================================================================

CREATE TABLE cognitive_cycles (
    cycle_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES working_memory_sessions(session_id),
    
    -- Trigger
    user_message_id UUID REFERENCES working_memory_messages(message_id),
    user_query TEXT NOT NULL,
    
    -- Phase 1: Summarizer
    phase1_started_at TIMESTAMPTZ,
    phase1_completed_at TIMESTAMPTZ,
    context_before_compression JSONB,
    context_after_compression JSONB,
    messages_compressed INTEGER DEFAULT 0,
    
    -- Phase 2: Investigator
    phase2_started_at TIMESTAMPTZ,
    phase2_completed_at TIMESTAMPTZ,
    investigation_goal TEXT,
    hops_executed INTEGER DEFAULT 0,
    max_hops_allowed INTEGER DEFAULT 4,
    evidence_collected JSONB DEFAULT '[]',
    investigation_decision TEXT,  -- 'sufficient_evidence', 'max_hops_reached', 'no_relevant_data'
    
    -- Phase 3: Agent (Synthesis)
    phase3_started_at TIMESTAMPTZ,
    phase3_completed_at TIMESTAMPTZ,
    agents_invoked TEXT[],  -- Which E2I agents were called
    synthesis_reasoning TEXT,
    response_generated TEXT,
    confidence_score FLOAT,
    
    -- Phase 4: Reflector
    phase4_started_at TIMESTAMPTZ,
    phase4_completed_at TIMESTAMPTZ,
    worth_remembering BOOLEAN,
    new_facts_extracted JSONB DEFAULT '[]',  -- Triplets for semantic memory
    new_procedures_learned JSONB DEFAULT '[]',  -- Tool sequences for procedural memory
    episodic_memory_created UUID REFERENCES episodic_memories(memory_id),
    
    -- Overall metrics
    total_duration_ms INTEGER,
    success BOOLEAN,
    error_message TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cycles_session ON cognitive_cycles(session_id);
CREATE INDEX idx_cycles_created ON cognitive_cycles(created_at DESC);

-- ============================================================================
-- 6. INVESTIGATION HOPS
-- Detailed tracking of multi-hop investigation in Phase 2
-- ============================================================================

CREATE TABLE investigation_hops (
    hop_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cycle_id UUID REFERENCES cognitive_cycles(cycle_id) ON DELETE CASCADE,
    
    hop_number INTEGER NOT NULL,
    
    -- What was queried
    query_type VARCHAR(50) NOT NULL,  -- 'vector_search', 'graph_traverse', 'sql_query', 'agent_call'
    query_target VARCHAR(100),  -- Which memory/source was queried
    query_content JSONB NOT NULL,  -- The actual query/parameters
    
    -- What was found
    results_count INTEGER,
    results_summary TEXT,
    results_raw JSONB,
    
    -- Relevance assessment
    relevance_score FLOAT,
    selected_for_evidence BOOLEAN DEFAULT false,
    
    -- Performance
    execution_time_ms INTEGER,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_hops_cycle ON investigation_hops(cycle_id, hop_number);

-- ============================================================================
-- 7. LEARNING SIGNALS
-- Captures signals that drive self-improvement
-- ============================================================================

CREATE TABLE learning_signals (
    signal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Source
    cycle_id UUID REFERENCES cognitive_cycles(cycle_id),
    session_id UUID REFERENCES working_memory_sessions(session_id),
    
    -- Signal type
    signal_type VARCHAR(50) NOT NULL,
    -- Types: 'user_feedback', 'llm_judge', 'outcome_success', 'outcome_failure', 
    --        'correction', 'preference', 'new_pattern'
    
    -- The signal
    signal_value FLOAT,  -- Numeric signal (-1 to 1, or score)
    signal_details JSONB,
    
    -- What it applies to
    applies_to_type VARCHAR(50),  -- 'procedure', 'response', 'agent', 'query_pattern'
    applies_to_id UUID,
    
    -- For DSPy optimization
    is_training_example BOOLEAN DEFAULT false,
    dspy_metric_name VARCHAR(100),
    dspy_metric_value FLOAT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_signals_cycle ON learning_signals(cycle_id);
CREATE INDEX idx_signals_type ON learning_signals(signal_type);
CREATE INDEX idx_signals_training ON learning_signals(is_training_example) WHERE is_training_example = true;

-- ============================================================================
-- 8. MEMORY STATISTICS (for monitoring & optimization)
-- ============================================================================

CREATE TABLE memory_statistics (
    stat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measured_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Counts
    episodic_count BIGINT,
    procedural_count BIGINT,
    semantic_cache_count BIGINT,
    active_sessions_count INTEGER,
    
    -- Performance
    avg_investigation_hops FLOAT,
    avg_cycle_duration_ms FLOAT,
    cache_hit_rate FLOAT,
    
    -- Quality
    avg_user_satisfaction FLOAT,
    successful_cycles_pct FLOAT,
    procedures_learned_24h INTEGER,
    facts_learned_24h INTEGER,
    
    -- Storage
    episodic_storage_mb FLOAT,
    procedural_storage_mb FLOAT,
    vector_index_size_mb FLOAT
);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to search episodic memory by embedding similarity
CREATE OR REPLACE FUNCTION search_episodic_memory(
    query_embedding vector(1536),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INTEGER DEFAULT 10,
    filter_event_type VARCHAR DEFAULT NULL,
    filter_agent VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    memory_id UUID,
    description TEXT,
    event_type VARCHAR,
    occurred_at TIMESTAMPTZ,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        em.memory_id,
        em.description,
        em.event_type,
        em.occurred_at,
        1 - (em.embedding <=> query_embedding) AS similarity
    FROM episodic_memories em
    WHERE 
        (filter_event_type IS NULL OR em.event_type = filter_event_type)
        AND (filter_agent IS NULL OR em.agent_name = filter_agent)
        AND (1 - (em.embedding <=> query_embedding)) >= match_threshold
    ORDER BY em.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function to find relevant procedures by query similarity
CREATE OR REPLACE FUNCTION find_relevant_procedures(
    query_embedding vector(1536),
    match_threshold FLOAT DEFAULT 0.6,
    match_count INTEGER DEFAULT 5
)
RETURNS TABLE (
    procedure_id UUID,
    procedure_name VARCHAR,
    tool_sequence JSONB,
    success_rate FLOAT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pm.procedure_id,
        pm.procedure_name,
        pm.tool_sequence,
        CASE WHEN pm.usage_count > 0 
             THEN pm.success_count::float / pm.usage_count 
             ELSE 0 END AS success_rate,
        1 - (pm.trigger_embedding <=> query_embedding) AS similarity
    FROM procedural_memories pm
    WHERE 
        pm.is_active = true
        AND pm.trigger_embedding IS NOT NULL
        AND (1 - (pm.trigger_embedding <=> query_embedding)) >= match_threshold
    ORDER BY 
        similarity DESC,
        success_rate DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function to prune old working memory sessions
CREATE OR REPLACE FUNCTION cleanup_old_sessions(
    max_age_hours INTEGER DEFAULT 24
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    WITH deleted AS (
        DELETE FROM working_memory_sessions
        WHERE 
            ended_at IS NOT NULL 
            AND ended_at < NOW() - (max_age_hours || ' hours')::INTERVAL
        RETURNING session_id
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- ROW-LEVEL SECURITY (for multi-tenant scenarios)
-- ============================================================================

-- Enable RLS on sensitive tables
ALTER TABLE working_memory_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE working_memory_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE episodic_memories ENABLE ROW LEVEL SECURITY;

-- Policies can be added based on user_id from session context
-- Example: CREATE POLICY user_sessions ON working_memory_sessions
--          FOR ALL USING (user_id = current_setting('app.current_user_id'));

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE episodic_memories IS 'Stores what happened - user interactions, agent actions, system events. Enables "what did I do yesterday?" queries.';
COMMENT ON TABLE procedural_memories IS 'Stores how to do things - successful tool call sequences. Enables few-shot learning for similar queries.';
COMMENT ON TABLE semantic_memory_cache IS 'Hot cache of FalkorDB graph data. Stores entity relationships for fast traversal.';
COMMENT ON TABLE working_memory_sessions IS 'Active session state. Holds context window, evidence board, and current reasoning state.';
COMMENT ON TABLE cognitive_cycles IS 'Tracks each user interaction through the 4-phase cognitive workflow.';
COMMENT ON TABLE investigation_hops IS 'Detailed log of multi-hop investigation in Phase 2 (investigator node).';
COMMENT ON TABLE learning_signals IS 'Captures feedback and outcome signals for DSPy prompt optimization.';
