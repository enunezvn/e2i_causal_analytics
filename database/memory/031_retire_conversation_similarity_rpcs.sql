-- =============================================================================
-- Retire the superseded conversation-similarity RPCs (audit 2026-06-05, F1 / D1)
-- Migration: 031_retire_conversation_similarity_rpcs.sql
-- Date: 2026-06-05
-- =============================================================================
--
-- The `016` RPCs (`search_similar_conversations`, `get_conversations_with_feedback`)
-- operate on `cognitive_cycles` and reference 7 columns that do not exist in its
-- DDL (agent_response, response_type, feedback_type, feedback_text,
-- feedback_score, feedback_at, created_at) — they would raise `UndefinedColumn`
-- if ever executed. Their sole caller, `ConversationRepository`, is never
-- instantiated (export-only) and is retired in the same change.
--
-- Confirmed safe to retire (D1, evidence-based): the live capability — vector
-- similarity over conversation/query history for RAG — is provided by
-- `episodic_memories` via the `hybrid_vector_search` / `hybrid_fulltext_search`
-- RPCs (HybridRetriever → MemoryConnector). `chatbot_conversations` is the live
-- conversation store (CRUD); feedback recording lives in
-- `chatbot_message_feedback` (ChatbotFeedbackRepository). Nothing reachable calls
-- the 016 RPCs.
--
-- Idempotent: drops every overload of each function by name via pg_proc, so it
-- is a no-op on re-apply and robust to signature drift.
-- =============================================================================

DO $$
DECLARE
    r record;
BEGIN
    FOR r IN
        SELECT 'DROP FUNCTION IF EXISTS ' || p.oid::regprocedure || ' CASCADE;' AS stmt
        FROM pg_proc p
        WHERE p.proname IN (
            'search_similar_conversations',
            'get_conversations_with_feedback'
        )
    LOOP
        EXECUTE r.stmt;
    END LOOP;
END
$$;
