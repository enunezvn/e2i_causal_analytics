-- ============================================================================
-- E2I Episodic Memory Deduplication — issue #388 (Phase 3 §3.4)
-- Migration: 026_episodic_dedup.sql
--
-- Adds the deduplication surface to ``episodic_memories``:
--
--   * ``dedup_signature TEXT``      — deterministic hash over the dedup
--                                     key fields, computed by
--                                     src/memory/lifecycle/consolidator.py
--                                     ::_compute_dedup_signature.
--   * ``dedup_counter INT DEFAULT 1`` — count of underlying events
--                                     represented by the canonical row
--                                     after the consolidator's
--                                     ``deduplicate_episodic`` pass.
--   * partial-unique-index on (brand, dedup_signature) WHERE
--     dedup_signature IS NOT NULL — DB-level race-condition safety
--     for concurrent inserts that would otherwise produce duplicates.
--     Mirrors the pattern at 021_insight_lifecycle.sql:219-226
--     (uix_executive_insights_active_causal_path).
--
-- Key shape (justified in src/memory/lifecycle/consolidator.py
-- ::_compute_dedup_signature):
--   * PRIMARY: (brand, event_type, event_subtype, causal_path_id)
--   * FALLBACK (when causal_path_id IS NULL):
--       (brand, event_type, event_subtype, agent_name, sha256(description))
--   * Brand is ALWAYS included — defense in depth alongside the
--     (brand, dedup_signature) index (out-of-scope item: cross-brand
--     dedup is forbidden by spec).
--
-- Out of scope (issue #388 §Out of scope):
--   * Semantic-embedding-based dedup. Would require a vector-similarity
--     index + cosine-threshold parameter; tracked as separate decision.
--   * Fuzzy-key dedup. Same.
--   * Cross-brand dedup. Forbidden — brand is the tenant boundary.
--
-- Idempotency:
--   * ADD COLUMN IF NOT EXISTS — re-running on a migrated DB is a no-op.
--   * CREATE UNIQUE INDEX IF NOT EXISTS — same.
--   * No ENUM additions, so no DO/EXCEPTION-duplicate_object blocks
--     needed.
--
-- Naming/migration number rationale:
--   * 025 = crystaldigest_schema_completion (#376)
--   * 026 = this migration. Next free slot in database/memory/.
--
-- Forward-link to the Python surface:
--   * src/memory/lifecycle/consolidator.py::deduplicate_episodic — runs
--     before semantic promotion in ``Consolidator.run()``.
--   * src/memory/lifecycle/consolidator.py::_compute_dedup_signature
--     — pure helper computing the signature value the DB index enforces.
-- ============================================================================

BEGIN;

-- ----------------------------------------------------------------------------
-- 1. Schema additions
-- ----------------------------------------------------------------------------
-- dedup_signature: deterministic hash over the dedup key fields. NULL
-- means "not yet examined by the deduplicator" — first pass stamps it.
-- TEXT is forward-compatible with signature-version bumps
-- (_compute_dedup_signature prefixes with "v1:"; bumping to v2 yields
-- distinct signature space without index collisions on rollover).
ALTER TABLE episodic_memories ADD COLUMN IF NOT EXISTS dedup_signature TEXT;

-- dedup_counter: number of underlying events represented by this
-- canonical row after dedup. DEFAULT 1 preserves the semantic that an
-- un-deduped row represents exactly one event; the consolidator
-- increments this on collapse.
ALTER TABLE episodic_memories
    ADD COLUMN IF NOT EXISTS dedup_counter INTEGER NOT NULL DEFAULT 1;

-- ----------------------------------------------------------------------------
-- 2. Constraints
-- ----------------------------------------------------------------------------
-- dedup_counter must be >= 1 (a canonical row always represents at least
-- itself). Wrapped in a DO block for idempotency on re-apply.
DO $$ BEGIN
    ALTER TABLE episodic_memories
        ADD CONSTRAINT episodic_memories_dedup_counter_min_check
        CHECK (dedup_counter >= 1);
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- ----------------------------------------------------------------------------
-- 3. Partial-unique-index for race-condition safety
-- ----------------------------------------------------------------------------
-- Two simultaneous inserts with the same (brand, dedup_signature) would
-- otherwise produce duplicate canonical rows that the next consolidator
-- pass would have to merge. The partial-unique-index enforces uniqueness
-- at the DB level so the second writer raises unique_violation. The
-- application path catches this on UPSERT and increments
-- dedup_counter on the existing canonical row instead.
--
-- COALESCE on brand: brand is VARCHAR(50) NULLABLE on episodic_memories
-- (008_agentic_memory_schema.sql:180). Postgres UNIQUE treats each NULL
-- as distinct, so without COALESCE two NULL-brand rows with the same
-- signature would BOTH succeed and produce duplicates. Coercing NULL ->
-- '' makes the constraint behave as expected for the brand-namespaced
-- dedup contract.
--
-- WHERE dedup_signature IS NOT NULL: rows that have not yet been
-- examined by the deduplicator (signature NULL) are not yet subject to
-- the constraint. Once stamped, they are.
CREATE UNIQUE INDEX IF NOT EXISTS uix_episodic_memories_dedup_signature
    ON episodic_memories (COALESCE(brand, ''), dedup_signature)
    WHERE dedup_signature IS NOT NULL;

-- ----------------------------------------------------------------------------
-- 4. Supporting index for the dedup query
-- ----------------------------------------------------------------------------
-- The consolidator's deduplicate_episodic SELECTs rows where
-- dedup_signature IS NULL (un-examined). This partial index keeps that
-- query fast even when the table is dominated by already-deduped rows.
CREATE INDEX IF NOT EXISTS idx_episodic_memories_dedup_pending
    ON episodic_memories (brand, region)
    WHERE dedup_signature IS NULL;

COMMIT;
