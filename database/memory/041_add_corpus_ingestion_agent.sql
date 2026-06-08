-- ============================================================================
-- MIGRATION 041 (memory): Add corpus_ingestion to e2i_agent_name
-- ============================================================================
-- Date: 2026-06-08
-- Audit: RAG hybrid-search remediation (F3), Phase 5 (corpus population)
--
-- Purpose:
--   The durable RAG corpus ingestion path (src/rag/corpus_ingestion.py) writes
--   rendered operational-analytics rows (from the business_metrics KPI fact
--   table) into episodic_memories so the chatbot's hybrid_vector_search can
--   retrieve them. Those rows need a distinguishable, valid agent_name for
--   clean attribution + selective removal/re-sync.
--
--   episodic_memories.agent_name is typed e2i_agent_name (an ENUM). The Phase-0
--   spike proved (faithfully, via a live insert) that an arbitrary agent_name
--   raises Postgres 22P02 ('invalid input value for enum') -> caught by the
--   broad except -> logger.warning -> SILENT loss of the corpus row. The spike
--   reused the valid 'observability_connector'; this migration adds a dedicated
--   'corpus_ingestion' value so corpus rows are cleanly attributable.
--
-- Fix: forward-only, idempotent enum extension (same pattern as migrations
--   018/029/039).
--
-- NOTE: deploy skips migrations -> APPLY MANUALLY to the deployed Supabase:
--   docker exec -i <supabase-db-container> psql -U postgres -d postgres \
--     < database/memory/041_add_corpus_ingestion_agent.sql
-- ============================================================================

-- RAG corpus ingestion (operational-analytics corpus -> episodic_memories).
-- Placed before 'orchestrator' like the other non-core agents added in 018/029.
DO $$ BEGIN
    ALTER TYPE e2i_agent_name ADD VALUE IF NOT EXISTS 'corpus_ingestion' BEFORE 'orchestrator';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Verification (run separately, NOT inside the ADD VALUE transaction):
--   SELECT e.enumlabel FROM pg_type t JOIN pg_enum e ON t.oid = e.enumtypid
--   WHERE t.typname = 'e2i_agent_name' AND e.enumlabel = 'corpus_ingestion';
