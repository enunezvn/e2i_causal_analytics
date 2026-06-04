-- ============================================================================
-- E2I Insight Lifecycle Schema v1.0
-- Migration: 021_insight_lifecycle.sql
-- Purpose: Adapt agentmemory patterns (consolidation, cascading invalidation,
--          JIT provenance verification, sentinels, executive crystallization)
--          to E2I causal analytics.
--
-- Adds:
--   - insight_edges            : DAG of provenance links between artifacts
--   - sentinels                : data-driven watcher registry
--   - executive_insights       : crystallized cross-agent narratives for leadership
--   - causal_paths.brand                : brand scoping for cascade
--   - causal_paths.confirmation_count   : repeated-rediscovery counter
--   - causal_paths.last_confirmed_at    : timestamp of last confirmation
--   - causal_paths.consolidated_at      : when promoted Episodic -> Semantic
--   - episodic_memories.consolidation_tier  : working/episodic/semantic/procedural
--   - triggers.invalidated_at / invalidation_reason
--   - ml_predictions.invalidated_at / invalidation_reason
--   - RPC verify_insight_chain : JIT walk of provenance ancestors
--
-- Tenancy model: brand is the de facto tenant boundary (see plan, §"Tenancy
-- Model"). Every cascade hop filters by brand. Cross-brand edges must be
-- authored explicitly with brand='all' (admin-only at the application layer).
-- ============================================================================

BEGIN;

-- ----------------------------------------------------------------------------
-- 0. EXTENSIONS
-- ----------------------------------------------------------------------------
-- pgcrypto provides gen_random_uuid(). Supabase enables this by default but
-- older / self-hosted Postgres projects may not, so declare explicitly so the
-- migration is portable. Must come BEFORE the first gen_random_uuid() call.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ----------------------------------------------------------------------------
-- 1. ENUMS
-- ----------------------------------------------------------------------------

DO $$ BEGIN
    CREATE TYPE insight_consolidation_tier AS ENUM (
        'working', 'episodic', 'semantic', 'procedural'
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE insight_edge_type AS ENUM (
        'derived_from',   -- trigger -> causal_path; ml_prediction -> causal_path
        'cites',          -- ml_prediction -> causal_path (model used path as feature)
        'summarizes',     -- executive_insight -> episodic_memory / causal_path
        'consolidated_from'  -- semantic causal_path -> episodic_memory ancestors
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE sentinel_pattern_type AS ENUM (
        'threshold_breach',  -- numeric metric crossed a configured threshold
        'freshness',         -- a row hasn't been updated in N hours
        'drift_score',       -- drift_monitor reported score above limit
        'new_causal_path'    -- new validated causal_path appeared
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE sentinel_action_type AS ENUM (
        'invalidate',     -- call cascade_invalidate on a target
        'dispatch_agent', -- enqueue an orchestrator task
        'notify'          -- emit Slack/email/webhook
    );
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- ----------------------------------------------------------------------------
-- 2. causal_paths additions
-- ----------------------------------------------------------------------------
-- Brand is added as nullable to avoid breaking existing rows; new writes
-- through the lifecycle subsystem must populate it. A backfill task is left
-- to the operator (see plan §Verification).

ALTER TABLE causal_paths ADD COLUMN IF NOT EXISTS brand VARCHAR(50);
ALTER TABLE causal_paths ADD COLUMN IF NOT EXISTS region VARCHAR(20);
ALTER TABLE causal_paths ADD COLUMN IF NOT EXISTS confirmation_count INTEGER NOT NULL DEFAULT 1;
ALTER TABLE causal_paths ADD COLUMN IF NOT EXISTS last_confirmed_at TIMESTAMPTZ;
ALTER TABLE causal_paths ADD COLUMN IF NOT EXISTS consolidated_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_causal_paths_brand ON causal_paths(brand, validation_status);
CREATE INDEX IF NOT EXISTS idx_causal_paths_consolidated ON causal_paths(consolidated_at)
    WHERE consolidated_at IS NOT NULL;

-- ----------------------------------------------------------------------------
-- 3. episodic_memories.consolidation_tier
-- ----------------------------------------------------------------------------

ALTER TABLE episodic_memories
    ADD COLUMN IF NOT EXISTS consolidation_tier insight_consolidation_tier NOT NULL DEFAULT 'episodic';

CREATE INDEX IF NOT EXISTS idx_episodic_memories_tier
    ON episodic_memories(consolidation_tier, brand);

-- ----------------------------------------------------------------------------
-- 4. triggers / ml_predictions invalidation columns
-- ----------------------------------------------------------------------------

ALTER TABLE triggers ADD COLUMN IF NOT EXISTS invalidated_at TIMESTAMPTZ;
ALTER TABLE triggers ADD COLUMN IF NOT EXISTS invalidation_reason TEXT;
CREATE INDEX IF NOT EXISTS idx_triggers_invalidated ON triggers(invalidated_at)
    WHERE invalidated_at IS NOT NULL;

-- ml_predictions: column may or may not exist depending on prior migrations.
-- Use defensive ADD COLUMN IF NOT EXISTS to be idempotent.
ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS invalidated_at TIMESTAMPTZ;
ALTER TABLE ml_predictions ADD COLUMN IF NOT EXISTS invalidation_reason TEXT;
CREATE INDEX IF NOT EXISTS idx_ml_predictions_invalidated ON ml_predictions(invalidated_at)
    WHERE invalidated_at IS NOT NULL;

-- ----------------------------------------------------------------------------
-- 5. insight_edges
-- ----------------------------------------------------------------------------
-- IDs are TEXT to accommodate all upstream schemes (UUID, VARCHAR(20),
-- VARCHAR(30)). Brand is the cascade scope key.

CREATE TABLE IF NOT EXISTS insight_edges (
    edge_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type       VARCHAR(40) NOT NULL,  -- 'causal_path', 'episodic_memory', ...
    source_id         TEXT        NOT NULL,
    target_type       VARCHAR(40) NOT NULL,  -- 'trigger', 'ml_prediction', 'executive_insight', ...
    target_id         TEXT        NOT NULL,
    edge_type         insight_edge_type NOT NULL,
    brand             VARCHAR(50) NOT NULL,  -- tenant boundary; 'all' = explicit cross-brand
    region            VARCHAR(20),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by_user_id VARCHAR(100),
    metadata          JSONB DEFAULT '{}',
    CONSTRAINT insight_edges_unique UNIQUE (source_type, source_id, target_type, target_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_insight_edges_source ON insight_edges(source_type, source_id, brand);
CREATE INDEX IF NOT EXISTS idx_insight_edges_target ON insight_edges(target_type, target_id, brand);
CREATE INDEX IF NOT EXISTS idx_insight_edges_brand  ON insight_edges(brand, created_at);

COMMENT ON TABLE insight_edges IS
'Provenance DAG between E2I artifacts. cascade_invalidate(BFS) and verify_insight_chain(walk up) both traverse this table. brand=''all'' is the only cross-brand authoring.';

-- ----------------------------------------------------------------------------
-- 6. sentinels
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sentinels (
    sentinel_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name               VARCHAR(200) NOT NULL,
    description        TEXT,
    pattern_type       sentinel_pattern_type NOT NULL,
    pattern_config     JSONB NOT NULL,   -- e.g. {"table":"causal_paths","column":"causal_effect_size","op":"<","value":0.05}
    action_type        sentinel_action_type NOT NULL,
    action_config      JSONB NOT NULL DEFAULT '{}',
    brand              VARCHAR(50) NOT NULL,  -- 'all' admin-only at API layer
    region             VARCHAR(20),
    created_by_user_id VARCHAR(100),
    enabled            BOOLEAN NOT NULL DEFAULT TRUE,
    last_fired_at      TIMESTAMPTZ,
    fire_count         INTEGER NOT NULL DEFAULT 0,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sentinels_enabled ON sentinels(enabled, pattern_type);
CREATE INDEX IF NOT EXISTS idx_sentinels_brand   ON sentinels(brand);

COMMENT ON TABLE sentinels IS
'Data-driven watchers. Dispatcher (Celery beat sentinel_dispatcher) evaluates each enabled sentinel every 5 min, fires action when pattern matches. Brand is the tenant boundary for evaluation AND for downstream action scoping.';

-- ----------------------------------------------------------------------------
-- 7. executive_insights
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS executive_insights (
    insight_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title                   VARCHAR(500) NOT NULL,
    narrative               TEXT NOT NULL,
    brand                   VARCHAR(50) NOT NULL,  -- NEVER cross-brand
    region                  VARCHAR(20),
    kpi                     VARCHAR(100),
    time_window_start       TIMESTAMPTZ,
    time_window_end         TIMESTAMPTZ,
    key_metrics             JSONB NOT NULL DEFAULT '{}',
    recall                  BOOLEAN NOT NULL DEFAULT FALSE,
    recall_reason           TEXT,
    recall_at               TIMESTAMPTZ,
    crystallized_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    crystallized_by_cycle_id UUID,
    crystallized_by_user_id VARCHAR(100),
    invalidated_at          TIMESTAMPTZ,         -- mirrors triggers/predictions for cascade uniformity
    invalidation_reason     TEXT,
    source_count            INTEGER NOT NULL DEFAULT 0  -- count of insight_edges where target=this row
);

CREATE INDEX IF NOT EXISTS idx_executive_insights_brand_recall
    ON executive_insights(brand, recall, crystallized_at DESC);
CREATE INDEX IF NOT EXISTS idx_executive_insights_invalidated
    ON executive_insights(invalidated_at) WHERE invalidated_at IS NOT NULL;

-- Partial-unique-index prevents duplicate active crystallizations for the
-- same (brand, region, kpi, causal_path) group. Concurrent crystallizer
-- runs (Celery beat + operator-triggered POST /crystallize) collide here;
-- crystallizer.py catches the unique-violation and skips. Recall +
-- recrystallize is allowed because the index is partial on invalidated_at:
-- once invalidated_at IS NOT NULL, the row is no longer subject to this
-- constraint and a new active row can be inserted in its place.
--
-- COALESCE on region/kpi: both columns are NULLABLE in the table; Postgres
-- UNIQUE permits multiple NULLs (each NULL is distinct), so without
-- COALESCE two crystallizations with region=NULL or kpi=NULL would BOTH
-- succeed and produce duplicates. Coercing NULL → '' makes the constraint
-- behave as expected for the dedup contract. See codex-rescue iter-0 HIGH-2.
CREATE UNIQUE INDEX IF NOT EXISTS uix_executive_insights_active_causal_path
    ON executive_insights (
        brand,
        COALESCE(region, ''),
        COALESCE(kpi, ''),
        COALESCE((key_metrics ->> 'causal_path_id'), '')
    )
    WHERE invalidated_at IS NULL;

COMMENT ON TABLE executive_insights IS
'Crystallized cross-agent narratives. Crystallizer aggregates strictly within (brand, region, kpi, time_window). NEVER produced cross-brand. recall=TRUE means at least one provenance ancestor was overturned (set by JIT verifier).';

-- ----------------------------------------------------------------------------
-- 8. verify_insight_chain RPC
-- ----------------------------------------------------------------------------
-- Walks insight_edges upward (target -> source). Returns (is_valid, broken_at,
-- reason). is_valid=false if ANY ancestor is invalidated_at IS NOT NULL OR
-- (for causal_paths) validation_status = 'overturned'. This STABLE function
-- returns the verdict only and writes nothing; the caller
-- (/api/middleware/insight_verifier.py) logs each check to
-- audit_chain_verification_log with verification_method='jit_provenance'.
--
-- Max recursion depth defaults to 16 to keep walks bounded; in practice
-- provenance chains in E2I are 2-4 hops.

CREATE OR REPLACE FUNCTION verify_insight_chain(
    p_insight_id   TEXT,
    p_insight_type TEXT,
    p_max_depth    INTEGER DEFAULT 16
) RETURNS TABLE (
    is_valid     BOOLEAN,
    broken_at_type TEXT,
    broken_at_id TEXT,
    reason       TEXT,
    depth_walked INTEGER
) AS $$
DECLARE
    v_current_type TEXT;
    v_current_id   TEXT;
    v_depth        INTEGER := 0;
    v_validation   TEXT;
    v_invalidated  TIMESTAMPTZ;
BEGIN
    -- Check the insight itself first.
    -- Only a few target tables carry invalidated_at; check the ones we know.
    IF p_insight_type = 'executive_insight' THEN
        SELECT invalidated_at INTO v_invalidated
            FROM executive_insights WHERE insight_id::TEXT = p_insight_id;
        IF v_invalidated IS NOT NULL THEN
            RETURN QUERY SELECT FALSE, p_insight_type, p_insight_id,
                'executive_insight already invalidated'::TEXT, 0;
            RETURN;
        END IF;
    ELSIF p_insight_type = 'trigger' THEN
        SELECT invalidated_at INTO v_invalidated
            FROM triggers WHERE trigger_id = p_insight_id;
        IF v_invalidated IS NOT NULL THEN
            RETURN QUERY SELECT FALSE, p_insight_type, p_insight_id,
                'trigger already invalidated'::TEXT, 0;
            RETURN;
        END IF;
    ELSIF p_insight_type = 'causal_path' THEN
        SELECT validation_status INTO v_validation
            FROM causal_paths WHERE path_id = p_insight_id;
        IF v_validation = 'overturned' THEN
            RETURN QUERY SELECT FALSE, p_insight_type, p_insight_id,
                'causal_path overturned'::TEXT, 0;
            RETURN;
        END IF;
    END IF;

    -- Walk ancestors via insight_edges.
    -- Iterative BFS via a CTE; we stop on first broken ancestor.
    FOR v_current_type, v_current_id IN
        WITH RECURSIVE ancestors AS (
            SELECT ie.source_type, ie.source_id, 1 AS depth
              FROM insight_edges ie
             WHERE ie.target_type = p_insight_type AND ie.target_id = p_insight_id
            UNION
            SELECT ie2.source_type, ie2.source_id, a.depth + 1
              FROM insight_edges ie2
              JOIN ancestors a
                ON ie2.target_type = a.source_type AND ie2.target_id = a.source_id
             WHERE a.depth < p_max_depth
        )
        SELECT source_type, source_id FROM ancestors
    LOOP
        v_depth := v_depth + 1;

        -- Per-type validity probe. Each branch is a simple lookup.
        IF v_current_type = 'causal_path' THEN
            SELECT validation_status INTO v_validation
              FROM causal_paths WHERE path_id = v_current_id;
            IF v_validation = 'overturned' THEN
                RETURN QUERY SELECT FALSE, v_current_type, v_current_id,
                    ('ancestor causal_path overturned: ' || v_current_id)::TEXT, v_depth;
                RETURN;
            END IF;
        ELSIF v_current_type = 'episodic_memory' THEN
            -- episodic_memories has no invalidated_at currently; nothing to check
            CONTINUE;
        ELSIF v_current_type = 'trigger' THEN
            SELECT invalidated_at INTO v_invalidated
              FROM triggers WHERE trigger_id = v_current_id;
            IF v_invalidated IS NOT NULL THEN
                RETURN QUERY SELECT FALSE, v_current_type, v_current_id,
                    ('ancestor trigger invalidated: ' || v_current_id)::TEXT, v_depth;
                RETURN;
            END IF;
        END IF;
    END LOOP;

    RETURN QUERY SELECT TRUE, NULL::TEXT, NULL::TEXT, NULL::TEXT, v_depth;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION verify_insight_chain IS
'JIT provenance walk. Returns is_valid=FALSE on first overturned/invalidated ancestor. Bounded by p_max_depth (default 16). Used by /api/middleware/insight_verifier.py on read.';

-- ----------------------------------------------------------------------------
-- 9. Permissions
-- ----------------------------------------------------------------------------

GRANT SELECT ON insight_edges, sentinels, executive_insights TO e2i_readonly;
GRANT SELECT, INSERT, UPDATE ON insight_edges, sentinels, executive_insights TO e2i_service;
GRANT EXECUTE ON FUNCTION verify_insight_chain TO e2i_readonly, e2i_service;

COMMIT;
