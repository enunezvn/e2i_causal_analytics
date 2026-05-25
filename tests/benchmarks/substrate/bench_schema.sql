-- tests/benchmarks/substrate/bench_schema.sql
-- Minimal schema for the HybridRetriever latency substrate (#414).
-- Load order: this file -> 011_hybrid_search_functions_fixed.sql -> 022_...sql -> seed.
CREATE EXTENSION IF NOT EXISTS vector;

-- 011/022 GRANT EXECUTE ... TO authenticated; create the role so they don't error.
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        CREATE ROLE authenticated;
    END IF;
END $$;

-- Vector-search source tables (hybrid_vector_search) --------------------------
CREATE TABLE IF NOT EXISTS episodic_memories (
    memory_id        text PRIMARY KEY,
    description      text,
    embedding        vector(1536),
    event_type       text,
    agent_name       text,
    occurred_at      timestamptz,
    brand            text,
    region           text,
    patient_id       text,
    hcp_id           text,
    importance_score double precision
);

CREATE TABLE IF NOT EXISTS procedural_memories (
    procedure_id       text PRIMARY KEY,
    procedure_name     text,
    trigger_pattern    text,
    trigger_embedding  vector(1536),
    is_active          boolean DEFAULT true,
    success_count      integer DEFAULT 0,
    procedure_type     text,
    success_rate       double precision,
    usage_count        integer,
    applicable_brands  text[],
    applicable_regions text[],
    detected_intent    text
);

-- Full-text source tables (hybrid_fulltext_search) ----------------------------
-- search_vector (GENERATED) + GIN indexes are added by 011's ALTER statements.
CREATE TABLE IF NOT EXISTS causal_paths (
    path_id            text PRIMARY KEY,
    start_node         text,
    end_node           text,
    method_used        text,
    causal_chain       jsonb,
    causal_effect_size double precision,
    confidence_level   double precision,
    created_at         timestamptz
);

CREATE TABLE IF NOT EXISTS agent_activities (
    activity_id      text PRIMARY KEY,
    agent_name       text,
    activity_type    text,
    analysis_results jsonb,
    agent_tier       text,
    status           text,
    created_at       timestamptz,
    workstream       text
);

CREATE TABLE IF NOT EXISTS triggers (
    trigger_id         text PRIMARY KEY,
    trigger_reason     text,
    trigger_type       text,
    recommended_action text,
    priority           text,
    confidence_score   double precision,
    created_at         timestamptz,
    invalidated_at     timestamptz  -- referenced by 022's max_staleness filter
);
