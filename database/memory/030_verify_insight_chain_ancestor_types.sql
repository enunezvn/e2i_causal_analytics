-- ============================================================================
-- Migration 030: verify_insight_chain — add ml_prediction + executive_insight
--                invalidation branches (L21 / issue #702)
-- ============================================================================
--
-- WHAT: CREATE OR REPLACE verify_insight_chain (defined in 021_insight_lifecycle.sql)
--       to recognise two artifact types it currently ignores:
--         * ml_prediction      — invalidated_at IS NOT NULL (column added in 021)
--         * executive_insight  — invalidated_at IS NOT NULL (ancestor walk only;
--                                 the self-check already handled this type)
--
-- WHY (L21): the JIT provenance walk returns is_valid=FALSE on the first
--       overturned/invalidated ancestor. 021 wired branches for causal_path,
--       trigger, and (self-check only) executive_insight — but NOT ml_prediction,
--       even though that same migration added ml_predictions.invalidated_at and
--       cascade_invalidate / INVALIDATABLE_TABLES treat ml_prediction and
--       executive_insight as first-class invalidatable artifacts. The result is
--       a FALSE-VALID verdict: an insight derived from an invalidated
--       ml_prediction (or an invalidated executive_insight ancestor) is reported
--       valid, so the read-path verifier (insight_verifier.py) serves stale data
--       instead of returning 410 Gone. Reproduced faithfully on the droplet: a
--       seeded ml_prediction(invalidated) -> executive_insight chain returned
--       is_valid=t (depth_walked=1) under the old function.
--
-- SCOPE: function body only. No table/column/data changes. The added branches
--        mirror the existing trigger branch (invalidated_at IS NOT NULL) and the
--        existing executive_insight self-check (insight_id::TEXT = id). Existing
--        verdicts are unchanged for every already-handled type; this strictly
--        flips previously-missed invalidated ancestors from valid -> invalid.
--
-- IDEMPOTENT: CREATE OR REPLACE FUNCTION is re-runnable. Safe to re-apply.
-- ============================================================================

BEGIN;

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
    ELSIF p_insight_type = 'ml_prediction' THEN
        -- L21: ml_predictions.invalidated_at added in 021 but never checked here.
        SELECT invalidated_at INTO v_invalidated
            FROM ml_predictions WHERE prediction_id = p_insight_id;
        IF v_invalidated IS NOT NULL THEN
            RETURN QUERY SELECT FALSE, p_insight_type, p_insight_id,
                'ml_prediction already invalidated'::TEXT, 0;
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
        ELSIF v_current_type = 'ml_prediction' THEN
            -- L21: an invalidated upstream prediction taints everything derived
            -- from it. Mirrors the trigger branch (invalidated_at IS NOT NULL).
            SELECT invalidated_at INTO v_invalidated
              FROM ml_predictions WHERE prediction_id = v_current_id;
            IF v_invalidated IS NOT NULL THEN
                RETURN QUERY SELECT FALSE, v_current_type, v_current_id,
                    ('ancestor ml_prediction invalidated: ' || v_current_id)::TEXT, v_depth;
                RETURN;
            END IF;
        ELSIF v_current_type = 'executive_insight' THEN
            -- L21: an executive_insight can be an ancestor of another insight
            -- (e.g. consolidated_from). The self-check handled this type but the
            -- ancestor walk did not. Mirrors the self-check (insight_id::TEXT).
            SELECT invalidated_at INTO v_invalidated
              FROM executive_insights WHERE insight_id::TEXT = v_current_id;
            IF v_invalidated IS NOT NULL THEN
                RETURN QUERY SELECT FALSE, v_current_type, v_current_id,
                    ('ancestor executive_insight invalidated: ' || v_current_id)::TEXT, v_depth;
                RETURN;
            END IF;
        END IF;
    END LOOP;

    RETURN QUERY SELECT TRUE, NULL::TEXT, NULL::TEXT, NULL::TEXT, v_depth;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION verify_insight_chain IS
'JIT provenance walk. Returns is_valid=FALSE on first overturned/invalidated ancestor. Handles causal_path (overturned) + trigger/ml_prediction/executive_insight (invalidated_at) in BOTH the self-check and the ancestor walk (L21/#702 added ml_prediction + ancestor-walk executive_insight). Bounded by p_max_depth (default 16). Used by /api/middleware/insight_verifier.py on read.';

GRANT EXECUTE ON FUNCTION verify_insight_chain TO e2i_readonly, e2i_service;

COMMIT;
