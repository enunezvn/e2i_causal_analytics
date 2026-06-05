-- =============================================================================
-- Drop the orphaned cognitive_cycles / investigation_hops tables (audit F1 / D1b)
-- Migration: 032_drop_cognitive_cycles_trio.sql
-- Date: 2026-06-05
-- =============================================================================
--
-- `cognitive_cycles` has NO writer anywhere in src/ (verified:
-- `grep -E 'table\("cognitive_cycles"\)\.(insert|upsert|update)' src/` → none).
-- It was a superseded conversation/query-history store; the live 4-phase
-- cognitive workflow persists to `episodic_memories` + `learning_signals` +
-- FalkorDB, and live conversation history lives in `chatbot_conversations`.
-- `investigation_hops` is an FK child of `cognitive_cycles` with no writer/reader
-- (audit F3). The only consumer of either was the retired 016 RPCs +
-- ConversationRepository (migration 031).
--
-- Drop FK child (investigation_hops) before the parent (cognitive_cycles).
-- Idempotent (IF EXISTS); CASCADE clears dependent indexes/policies.
-- =============================================================================

DROP TABLE IF EXISTS investigation_hops CASCADE;
DROP TABLE IF EXISTS cognitive_cycles CASCADE;
