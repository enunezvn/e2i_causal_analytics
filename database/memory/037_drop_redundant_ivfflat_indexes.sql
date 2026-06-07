-- ============================================================================
-- Migration 037: drop redundant IVFFlat vector indexes (audit 2026-06-05, L20 / #694)
-- ============================================================================
--
-- WHAT: 011_hybrid_search_functions_fixed.sql added HNSW indexes on
--       episodic_memories.embedding (idx_episodic_memories_vector_hnsw) and
--       procedural_memories.trigger_embedding (idx_procedural_memories_vector_hnsw).
--       The original IVFFlat indexes from 001 coexist on the SAME columns →
--       double write amplification (every insert maintains BOTH). HNSW is the
--       chosen, superior index; the IVFFlat ones are leftover.
--
-- WHY SAFE (verified on the droplet 2026-06-07): both HNSW indexes are present
--       and valid; episodic_memories has 0 rows, procedural_memories ~1.26k —
--       so reads continue to use HNSW with no regression, and dropping the
--       IVFFlat indexes only removes redundant write work.
--       (The 3rd 001 IVFFlat index, idx_cycles_embedding on cognitive_cycles,
--       was already removed when migration 032 dropped that table.)
--
-- LOCKING: uses DROP INDEX CONCURRENTLY so the drop never takes an
-- AccessExclusiveLock that blocks reads/writes on the (live) tables. CONCURRENTLY
-- CANNOT run inside a transaction block — this migration must NOT be wrapped in
-- BEGIN/COMMIT (it is not). Idempotent (DROP INDEX IF EXISTS).
-- ============================================================================

DROP INDEX CONCURRENTLY IF EXISTS idx_episodic_embedding;
DROP INDEX CONCURRENTLY IF EXISTS idx_procedural_trigger_embedding;
