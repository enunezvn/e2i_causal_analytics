-- =============================================================================
-- Restore the cognitive_cycles + investigation_hops trio (REVERSES migration 032)
-- Migration: 042_restore_cognitive_cycles_trio.sql
-- Date: 2026-06-09
-- =============================================================================
--
-- OWNER DECISION 2026-06-09 (REASON-BEFORE-RULES): migration 032 dropped
-- `cognitive_cycles` + `investigation_hops` on the audit-F1 rationale "no writer
-- in src/ -> superseded -> drop". That classification was WRONG. `cognitive_cycles`
-- is the purpose-built parent ledger of the live 4-phase cognitive cycle
-- (Summarizer -> Investigator -> Agent -> Reflector): the live workflow
-- `src/memory/cognitive_integration.py::CognitiveService.process_query` GENERATES a
-- `cycle_id` for every query and threads it onto working-memory messages,
-- `episodic_memories.cycle_id`, and `learning_signals.cycle_id` -- but the PRODUCER
-- that writes the parent row was never wired. The table was therefore a
-- *scaffolded placeholder for requested functionality*, NOT a superseded design;
-- after 032 those `cycle_id` references dangle (their parent table is gone) and
-- `evaluation_results.cognitive_cycle_id` (database/ml/022) points at nothing.
--
-- This migration recreates the two tables EXACTLY as defined in
-- `001_agentic_memory_schema_v1.3.sql` (idempotent, IF NOT EXISTS), so the live
-- 4-phase producer (wired in the same change) can persist a real cognitive_cycles
-- row per cycle and the dangling cycle_id references become valid again.
--
-- It also resolves the fresh-rebuild ordering hazard the 032 drop introduced:
-- `run_migrations.sh` applies database/memory/ BEFORE database/ml/, so on a fresh
-- full replay 032 dropped cognitive_cycles before ml/022's
-- `evaluation_results.cognitive_cycle_id UUID REFERENCES cognitive_cycles(cycle_id)`
-- ran -> FK-target-missing error. Recreating the table here (still in memory/,
-- after 032) makes that FK target exist again before ml/ runs.
--
-- GRANTS: service_role only (the backend authenticates as service-role and
-- bypasses RLS). The original 001 `GRANT ... TO authenticated` is intentionally
-- NOT reissued -- migration 058 REVOKEd the anon/authenticated over-grant (M9/#703)
-- and the live memory tables carry service_role grants only.
--
-- Depends on enums created in 001 (cognitive_phase, e2i_agent_name) + the pgvector
-- extension -- all present (never dropped by 032). Idempotent; safe to re-apply.
-- =============================================================================

-- Parent first (investigation_hops FKs cognitive_cycles).
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

    -- E2I entities involved (arrays, no FKs)
    involved_patient_ids TEXT[],
    involved_hcp_ids TEXT[],
    involved_trigger_ids TEXT[],
    involved_causal_path_ids TEXT[],
    brands_discussed TEXT[],
    regions_discussed TEXT[],

    -- Phase tracking
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

    -- Output
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

-- FK child.
CREATE TABLE IF NOT EXISTS investigation_hops (
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

CREATE INDEX IF NOT EXISTS idx_hops_cycle ON investigation_hops(cycle_id);
CREATE INDEX IF NOT EXISTS idx_hops_memory_type ON investigation_hops(memory_type);
CREATE INDEX IF NOT EXISTS idx_hops_relevance ON investigation_hops(relevance_score DESC);

-- Service-role only (see header; do NOT re-grant authenticated — reversed by 058).
GRANT SELECT, INSERT, UPDATE, DELETE ON cognitive_cycles TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON investigation_hops TO service_role;

COMMENT ON TABLE cognitive_cycles IS 'Parent ledger of the 4-phase cognitive workflow (restored mig 042, reverses 032). Producer: src/memory/cognitive_integration.py::CognitiveService.';
COMMENT ON TABLE investigation_hops IS 'Per-hop detail of the Investigator phase; FK child of cognitive_cycles (restored mig 042).';
