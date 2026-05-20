-- ============================================================================
-- E2I Procedural-Template Extraction — issue #389 (Phase 3 §3.4)
-- Migration: 027_procedural_templates_schema.sql
--
-- Adds the persistence surface for procedural-template extraction:
--
--   * ``procedural_templates`` — new table holding the (brand,
--     template_signature) → template_body mappings produced by
--     ``src/memory/lifecycle/consolidator.py
--     ::Consolidator.extract_procedural_templates``.
--   * Partial-unique-index on ``(COALESCE(brand,''), template_signature)
--     WHERE template_signature IS NOT NULL`` — DB-level race-condition
--     safety for concurrent inserts. Mirrors the pattern from migration
--     026_episodic_dedup.sql:104 (``uix_episodic_memories_dedup_signature``)
--     and 021_insight_lifecycle.sql:219-226
--     (``uix_executive_insights_active_causal_path``).
--   * CHECK constraints on ``extraction_confidence`` (0..1) and
--     ``extraction_method`` ('symbolic'/'llm_with_fallback') — wrapped
--     in a DO block per migration 025 precedent (Postgres lacks
--     ``ADD CONSTRAINT IF NOT EXISTS``).
--
-- Design (justified in the Python surface at
-- src/memory/lifecycle/consolidator.py
-- ::Consolidator.extract_procedural_templates):
--
--   * Clustering basis: exact-match key-tuples ``(brand, event_type,
--     event_subtype, sorted(action_keys))``. Embedding-similarity is
--     V2 follow-up — out of scope here.
--   * Template body: Pydantic schema serialized to JSONB. NOT Jinja2
--     (cross-language fragility) and NOT free-form ``{var}`` text.
--   * Confidence: mean pairwise Jaccard cohesion over per-row
--     ``action_keys`` sets (deterministic, in [0..1]); when
--     ``PROCEDURAL_LLM_EXTRACTION_ENABLED=true``, multiplied by an
--     LLM-rated coherence in [0..1].
--   * Extraction method: 'symbolic' (always-on path) or
--     'llm_with_fallback' (LLM-augmented confidence; falls back to
--     'symbolic' on SDK exception).
--   * Brand boundary preserved: brand is included in every signature
--     AND is the leading column of the partial-unique-index.
--   * No revision/versioning in V1: extract once per cluster; revision
--     is V2 follow-up.
--
-- Idempotency:
--   * CREATE TABLE IF NOT EXISTS — re-running on a migrated DB is a no-op.
--   * CREATE UNIQUE INDEX IF NOT EXISTS — same.
--   * CHECK constraints wrapped in DO/EXCEPTION-duplicate_object blocks
--     per migration 025 precedent.
--
-- Naming/migration number rationale:
--   * 025 = crystaldigest_schema_completion (#376)
--   * 026 = episodic_dedup (#388)
--   * 027 = this migration. Next free slot in database/memory/.
--
-- Forward-link to the Python surface:
--   * src/memory/lifecycle/consolidator.py::extract_procedural_templates —
--     primary writer of ``procedural_templates`` rows.
--   * src/memory/lifecycle/consolidator.py::ProceduralTemplate — Pydantic
--     model whose serialized form populates ``template_body``.
--   * src/memory/lifecycle/consolidator.py::_compute_template_signature —
--     pure helper computing the signature value the DB index enforces.
-- ============================================================================

BEGIN;

-- ----------------------------------------------------------------------------
-- 1. Table
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS procedural_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    brand TEXT NOT NULL,
    template_signature TEXT NOT NULL,
    -- JSONB carries the serialized ``ProceduralTemplate`` body (Pydantic
    -- model). Shape: {"event_type": str, "event_subtype": str,
    -- "shared_action_keys": [str], "variables": [str]}.
    template_body JSONB NOT NULL,
    -- Provenance: the episodic_memories.memory_id values that the
    -- template was extracted from. UUID[] preserves type safety against
    -- the parent table; the consolidator passes these in as the
    -- canonical observation set for auditability.
    derived_from_episodic_ids UUID[] NOT NULL,
    -- Cluster cohesion score (mean pairwise Jaccard over action_keys
    -- sets within the cluster) — optionally multiplied by an
    -- LLM-rated coherence when the flag is on. [0..1] bounded by the
    -- CHECK constraint below.
    extraction_confidence FLOAT NOT NULL,
    -- Which extraction path produced this row.
    -- 'symbolic' — always-on Jaccard cohesion path (default).
    -- 'llm_with_fallback' — LLM-augmented confidence; falls back to
    --    'symbolic' if the SDK raised one of the narrow-catch
    --    anthropic.* error classes.
    extraction_method TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- 2. Constraints (wrapped in DO blocks for idempotency on re-apply)
-- ----------------------------------------------------------------------------
-- extraction_confidence must be in [0..1] — Jaccard cohesion is
-- bounded by definition and the LLM multiplier is rated in [0..1].
DO $$ BEGIN
    ALTER TABLE procedural_templates
        ADD CONSTRAINT procedural_templates_confidence_range_check
        CHECK (extraction_confidence >= 0 AND extraction_confidence <= 1);
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- extraction_method is constrained to the two literal values declared
-- by the ``ProceduralTemplate`` Pydantic model.
DO $$ BEGIN
    ALTER TABLE procedural_templates
        ADD CONSTRAINT procedural_templates_method_enum_check
        CHECK (extraction_method IN ('symbolic', 'llm_with_fallback'));
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- ----------------------------------------------------------------------------
-- 3. Partial-unique-index for race-condition safety
-- ----------------------------------------------------------------------------
-- Two simultaneous inserts with the same (brand, template_signature)
-- would otherwise produce duplicate template rows that the next
-- consolidator pass would have to merge. The partial-unique-index
-- enforces uniqueness at the DB level so the second writer raises
-- unique_violation. The application path catches this on INSERT and
-- swallows it for idempotency (re-extraction on the same cluster is a
-- no-op).
--
-- COALESCE on brand: brand is NOT NULL on procedural_templates so the
-- COALESCE is defensive only — kept for parity with migration 026's
-- index shape which DOES allow NULL brand on episodic_memories.
--
-- WHERE template_signature IS NOT NULL: defensive — template_signature
-- is NOT NULL but a partial-unique-index makes the index narrow even
-- if a future migration relaxes that constraint.
CREATE UNIQUE INDEX IF NOT EXISTS uix_procedural_templates_signature
    ON procedural_templates (COALESCE(brand, ''), template_signature)
    WHERE template_signature IS NOT NULL;

-- ----------------------------------------------------------------------------
-- 4. Supporting indexes
-- ----------------------------------------------------------------------------
-- Brand lookup: the consolidator extracts per-brand, so a leading
-- brand index keeps the typical SELECT pattern fast.
CREATE INDEX IF NOT EXISTS idx_procedural_templates_brand
    ON procedural_templates (brand);

-- Recency lookup: dashboards / observability may want the N most
-- recent templates per brand.
CREATE INDEX IF NOT EXISTS idx_procedural_templates_created_at
    ON procedural_templates (created_at DESC);

COMMIT;
