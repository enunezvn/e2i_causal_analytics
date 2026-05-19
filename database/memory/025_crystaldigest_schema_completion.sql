-- ============================================================================
-- E2I CrystalDigest Schema Completion — issue #376 (Phase 4)
-- Migration: 025_crystaldigest_schema_completion.sql
--
-- Adds the 15 missing CrystalDigest analytical/lineage columns to
-- ``executive_insights`` (shipped in 021_insight_lifecycle.sql with 13 fields).
--
-- Source-of-truth: GitHub issue #376 + plan
--   .claude/plans/e2i_memory_subsystems_implementation_plan.md
-- §"DECISIONS ADOPTED — 2026-05-19" (Decisions 2 + 3).
--
-- Per Decision 2 = HYBRID (sub-decision 2a):
--   * effect_size + ci bounds are NUMERIC (float), not categorical strings.
--   * 13 of these 15 fields are deterministically derivable from estimator
--     state / insight_edges graph / tier-table fields / episodic-memory
--     key_metrics. The 2 prose fields (``limitations`` and
--     ``recommended_next_analysis``) are LLM-narrative; see
--     src/data/kg/types.py::LLMCrystalNarrativeAudit.
--
-- Per Decision 3 = KEEP BINARY:
--   * ``staleness_score`` is OMITTED. Staleness remains boolean
--     (``invalidated_at IS NULL``). If a future workflow surfaces a graded
--     need, the plan §"DECISIONS ADOPTED" reinstatement checklist enumerates
--     the reversal cost (~800-1,200 LoC).
--
-- Naming/migration number rationale:
--   * 023 = sentinel_cooldown (#375)
--   * 024 = sentinel_invalidation_count_pattern (#381)
--   * 025 = this migration (issue #376 DoD says "023" but that slot is
--           already taken; the plan §Recommended sequencing item 3 line
--           reads "the other 15 fields are independent" and does not pin
--           a specific migration number, so 025 is the next free slot).
--
-- Idempotency:
--   * All ADD COLUMN statements use IF NOT EXISTS so re-running on a
--     deployment that's already migrated is a no-op.
--   * The CHECK constraint on effect_direction is wrapped in a DO block
--     with EXCEPTION duplicate_object handling so re-running does not
--     fail on the constraint already existing.
--
-- Forward-link to the Pydantic surface:
--   * src/api/routes/executive_insights.py::ExecutiveInsightResponse —
--     extended in lock-step with this migration (#376 DoD §C).
-- ============================================================================

BEGIN;

-- ----------------------------------------------------------------------------
-- 1. Analytical fields (#376 §A items 1-8)
-- ----------------------------------------------------------------------------
-- effect_size: numeric ATE (point estimate). Replaces the categorical
-- "small"/"medium"/"large" carried in src/agents/causal_impact/state.py
-- per sub-decision 2a.
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS effect_size FLOAT;

-- effect_ci_lower / effect_ci_upper: 95% confidence bounds. The estimator
-- already produces these (EstimationResult.ate_ci_lower / ate_ci_upper).
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS effect_ci_lower FLOAT;
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS effect_ci_upper FLOAT;

-- effect_direction: deterministic from the sign of effect_size + the
-- CI bounds. Three-valued: 'positive' / 'negative' / 'null' (the literal
-- string, not SQL NULL).
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS effect_direction TEXT;

-- cohort_size: integer count of subjects in the cohort underlying the
-- finding. Derived from the estimator's sample_size field.
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS cohort_size INTEGER;

-- confounders_controlled: array of covariate names. Derived from
-- EstimationResult.covariates_adjusted (List[str]).
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS confounders_controlled TEXT[];

-- sensitivity_checks_passed / sensitivity_checks_failed: paired arrays
-- of refutation-test names. Derived from RefutationResults.individual_tests
-- dict (passed=True → passed array; passed=False → failed array).
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS sensitivity_checks_passed TEXT[];
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS sensitivity_checks_failed TEXT[];

-- ----------------------------------------------------------------------------
-- 2. Narrative-prose fields (#376 §A items 9-10; Decision 2 LLM path)
-- ----------------------------------------------------------------------------
-- limitations / recommended_next_analysis: LLM-generated text. The
-- narrator is Haiku-backed and feature-flagged (env var
-- E2I_CRYSTAL_LLM_NARRATIVES_ENABLED). When the flag is off, the
-- deterministic _compose_narrative falls back and these columns are
-- populated from a heuristic.
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS limitations TEXT;
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS recommended_next_analysis TEXT;

-- ----------------------------------------------------------------------------
-- 3. Lineage fields (#376 §A items 11-15)
-- ----------------------------------------------------------------------------
-- provenance_chain_id: FK to insight_edges (LOGICAL, not enforced via
-- foreign key because insight_edges is an edge table not a node table —
-- the "chain" is a deterministic hash over the BFS-resolved ancestor set).
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS provenance_chain_id TEXT;

-- provenance_depth: integer hop-count from this crystal to the deepest
-- ancestor in insight_edges. Derived by the crystallizer via a single-pass
-- BFS at write time.
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS provenance_depth INTEGER;

-- consolidation_tier: 'working' | 'episodic' | 'semantic' | 'procedural'.
-- Matches the enum at 021_insight_lifecycle.sql line 42-44 but is stored
-- as TEXT here to keep this migration enum-creation-free.
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS consolidation_tier TEXT;

-- replication_count: number of independent confirmations of the finding.
-- Equivalent to causal_paths.confirmation_count for the source path,
-- captured at crystallization time so the crystal is self-contained.
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS replication_count INTEGER;

-- data_version: opaque tag identifying the dataset snapshot used. E2I
-- uses date-stamped snapshots (e.g. '2026-05-19-snapshot'). Populated
-- from the cohort manifest.
ALTER TABLE executive_insights ADD COLUMN IF NOT EXISTS data_version TEXT;

-- ----------------------------------------------------------------------------
-- 4. CHECK constraints
-- ----------------------------------------------------------------------------
-- effect_direction is enumerated to three values. We use DO + EXCEPTION
-- duplicate_object to keep the migration idempotent (re-running does not
-- fail on the constraint already existing).
DO $$ BEGIN
    ALTER TABLE executive_insights
        ADD CONSTRAINT exec_insights_effect_direction_check
        CHECK (
            effect_direction IS NULL
            OR effect_direction IN ('positive', 'negative', 'null')
        );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- consolidation_tier matches the insight_consolidation_tier enum semantics
-- (working/episodic/semantic/procedural). We do not retype the column
-- because TEXT is more forward-compatible with future tier names; we
-- check it at row insert in src/memory/crystallization/crystallizer.py.
DO $$ BEGIN
    ALTER TABLE executive_insights
        ADD CONSTRAINT exec_insights_consolidation_tier_check
        CHECK (
            consolidation_tier IS NULL
            OR consolidation_tier IN ('working', 'episodic', 'semantic', 'procedural')
        );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- ----------------------------------------------------------------------------
-- 5. Indexes for common dashboard queries
-- ----------------------------------------------------------------------------
-- portfolio-summary endpoint groups by brand + filters by effect_size IS
-- NOT NULL (the "has numeric effect" subset).
CREATE INDEX IF NOT EXISTS idx_executive_insights_brand_effect
    ON executive_insights(brand, effect_size)
    WHERE effect_size IS NOT NULL;

-- consolidation_tier filtering for tiered dashboards.
CREATE INDEX IF NOT EXISTS idx_executive_insights_consolidation_tier
    ON executive_insights(consolidation_tier)
    WHERE consolidation_tier IS NOT NULL;

COMMIT;
