-- ============================================================================
-- Migration 039: Drop triggers-table joins from migration-006 feedback-loop
--                truth-assignment functions (issue #182)
-- ============================================================================
-- Issue #182 — Migration 006 (and the brand-stripped reissue in migration 038)
-- left two truth-assignment functions joining against `triggers` on columns
-- that do not exist on that table:
--
--     LEFT JOIN triggers t ON t.prediction_id = p.prediction_id
--     ...
--     t.status as trigger_status
--
-- The real `triggers` schema (database/core/e2i_ml_complete_v3_schema.sql
-- §3.6, lines 579-619) has neither `prediction_id` nor `status` —
-- `delivery_status` and `acceptance_status` are the actual status columns,
-- and there is no FK from `triggers` back to `ml_predictions`. PL/pgSQL
-- uses deferred name resolution, so `CREATE OR REPLACE FUNCTION` succeeds
-- at definition time but EVERY execution of the function raises
-- "column t.prediction_id does not exist" at plan time.
--
-- The two affected functions are reachable in production via the Celery
-- beat schedule:
--
--   - `run_feedback_loop_short_window` (every 4h) fans out to
--     `run_feedback_loop('trigger')` and `run_feedback_loop('next_best_action')`
--     (src/tasks/feedback_loop_tasks.py:40, 287).
--   - The orchestrator in 006 §6 dispatches those types to
--     `assign_truth_script_conversion` and `assign_truth_next_best_action`
--     respectively (lines 704-714).
--
-- Today there are zero `prediction_type IN ('trigger', 'next_best_action')`
-- rows produced by `src/tasks/risk_score_prediction_tasks.py` (the only
-- live writer of `ml_predictions`), so the broken join never executes
-- against real data — but the FUNCTION CALL itself raises whenever any
-- 'trigger' / 'next_best_action' PENDING row would be evaluated. The
-- live droplet has been hand-patched to drop these joins; this migration
-- codifies that patched state so fresh-DB replays produce the same
-- function bodies.
--
-- Decision (matches issue #182 "Fix options" drift 1, conservative path):
-- STRIP the `LEFT JOIN triggers t` and all `t.*` references rather than
-- adding `prediction_id` + `status` columns to `triggers`. Rationale:
--
--   (1) No live consumer reads `truth_source = 'triggers_treatment_events'`
--       (the original NBA literal). A grep across src/, scripts/, tests/
--       returns zero results. Stripping is safe.
--
--   (2) The live droplet runs the patched form already
--       (`SELECT pg_get_functiondef('assign_truth_script_conversion'::regproc)`
--       confirmed on 2026-05-13). 039 brings the fresh-replay state into
--       parity with production.
--
--   (3) Adding `prediction_id` to `triggers` is non-trivial: there is no
--       producer code that knows which prediction generated a trigger,
--       and back-filling the column is undefined absent a runtime link.
--       The brand-strip rationale from migration 038 applies here too —
--       no current consumer means no benefit to adding the column.
--
-- POST-039 SEMANTICS:
--
--   - `assign_truth_script_conversion` now labels POSITIVE iff the HCP
--     has ≥1 prescription event in the observation window post-prediction,
--     ELSE NEGATIVE. The original EXCLUDED branch (gated on
--     `pc.trigger_status = 'not_delivered'`) is removed; no row is ever
--     EXCLUDED any more (`v_excluded` stays 0). This matches the live
--     droplet behaviour.
--
--   - `assign_truth_next_best_action` now labels POSITIVE iff the HCP
--     has any downstream activity (any treatment_event regardless of
--     event_type) in the window, ELSE NEGATIVE. The original branches
--     gated on `pc.trigger_status = 'accepted'` / `IS NULL` are removed.
--     `truth_confidence` was bifurcated 0.90/0.70 on accepted-vs-other;
--     post-039 we collapse to a single 0.70 (matches live droplet —
--     `pg_get_functiondef('assign_truth_next_best_action'::regproc)`).
--     `truth_source` is changed from 'triggers_treatment_events' to
--     'treatment_events' to reflect that triggers no longer contribute.
--
-- Re-application is idempotent because we use `CREATE OR REPLACE FUNCTION`
-- everywhere. No downgrade path is provided — restoring the triggers join
-- would re-introduce the original defect (every execution raises), so
-- there is no defensible forward-backward symmetric pair.
--
-- Out of scope for this migration (drifts 2 + 3 in issue #182):
--   - `business_metrics.measurement_date` (real column: `metric_date`).
--   - `business_metrics` per-HCP over-match on (region, month) joins.
-- These were addressed by migration 038's EXCLUDED-only stub for
-- `assign_truth_market_share` (database/migrations/038, §5). See
-- memory/issue_176_close_20260513.md for the rationale and the deferred
-- "implement brand-scoped market_share truth assignment" follow-on.
-- ============================================================================

BEGIN;

-- ----------------------------------------------------------------------------
-- 1. Replace assign_truth_script_conversion
--
-- Drops `LEFT JOIN triggers t ON t.prediction_id = p.prediction_id` plus
-- the `t.trigger_id` SELECT and the `pc.trigger_status = 'not_delivered'`
-- EXCLUDED branch. Otherwise identical to the 038-replaced body.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION assign_truth_script_conversion(
    p_observation_window_days INTEGER DEFAULT 21,
    p_batch_size INTEGER DEFAULT 1000
)
RETURNS TABLE (
    predictions_evaluated INTEGER,
    predictions_labeled INTEGER,
    predictions_excluded INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_evaluated INTEGER := 0;
    v_labeled INTEGER := 0;
    v_excluded INTEGER := 0;
BEGIN
    CREATE TEMP TABLE temp_conversion_candidates AS
    WITH prediction_context AS (
        SELECT
            p.prediction_id,
            p.hcp_id,
            p.prediction_timestamp
        FROM ml_predictions p
        WHERE p.prediction_type = 'trigger'
          AND p.outcome_label = 'PENDING'
          AND p.prediction_timestamp < NOW() - (p_observation_window_days || ' days')::INTERVAL
        LIMIT p_batch_size
    ),
    conversion_activity AS (
        SELECT
            pc.prediction_id,
            COUNT(*) as nrx_count_window,
            COUNT(DISTINCT te.patient_id) as unique_patients
        FROM prediction_context pc
        JOIN treatment_events te
            ON te.hcp_id = pc.hcp_id
            AND te.event_type = 'prescription'
            AND te.event_date BETWEEN
                pc.prediction_timestamp::DATE
                AND (pc.prediction_timestamp + (p_observation_window_days || ' days')::INTERVAL)::DATE
        GROUP BY pc.prediction_id
    )
    SELECT
        pc.prediction_id,
        COALESCE(ca.nrx_count_window, 0) as nrx_count,
        CASE
            WHEN COALESCE(ca.nrx_count_window, 0) >= 1 THEN 'POSITIVE'
            ELSE 'NEGATIVE'
        END as outcome_label,
        0.90 as truth_confidence
    FROM prediction_context pc
    LEFT JOIN conversion_activity ca ON ca.prediction_id = pc.prediction_id;

    SELECT COUNT(*) INTO v_evaluated FROM temp_conversion_candidates;
    SELECT COUNT(*) INTO v_labeled FROM temp_conversion_candidates WHERE outcome_label IN ('POSITIVE', 'NEGATIVE');
    SELECT COUNT(*) INTO v_excluded FROM temp_conversion_candidates WHERE outcome_label = 'EXCLUDED';

    UPDATE ml_predictions p
    SET
        actual_outcome = CASE
            WHEN tc.outcome_label = 'POSITIVE' THEN 1.0
            WHEN tc.outcome_label = 'NEGATIVE' THEN 0.0
            ELSE NULL
        END,
        outcome_recorded_at = NOW(),
        truth_source = 'treatment_events',
        truth_confidence = tc.truth_confidence,
        outcome_label = tc.outcome_label
    FROM temp_conversion_candidates tc
    WHERE p.prediction_id = tc.prediction_id;

    DROP TABLE temp_conversion_candidates;

    RETURN QUERY SELECT v_evaluated, v_labeled, v_excluded;
END;
$$;

-- ----------------------------------------------------------------------------
-- 2. Replace assign_truth_next_best_action
--
-- Drops `LEFT JOIN triggers t ON t.prediction_id = p.prediction_id`, the
-- `t.trigger_id` / `t.status` SELECT, and the `trigger_status`-gated
-- POSITIVE / EXCLUDED branches. Now: POSITIVE iff any downstream
-- treatment_event in the window, ELSE NEGATIVE. `truth_source` changes
-- from 'triggers_treatment_events' to 'treatment_events' since triggers
-- no longer contribute. Matches live-droplet patched body.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION assign_truth_next_best_action(
    p_observation_window_days INTEGER DEFAULT 30,
    p_batch_size INTEGER DEFAULT 1000
)
RETURNS TABLE (
    predictions_evaluated INTEGER,
    predictions_labeled INTEGER,
    predictions_excluded INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_evaluated INTEGER := 0;
    v_labeled INTEGER := 0;
    v_excluded INTEGER := 0;
BEGIN
    CREATE TEMP TABLE temp_nba_candidates AS
    WITH prediction_context AS (
        SELECT
            p.prediction_id,
            p.hcp_id,
            p.prediction_timestamp
        FROM ml_predictions p
        WHERE p.prediction_type = 'next_best_action'
          AND p.outcome_label = 'PENDING'
          AND p.prediction_timestamp < NOW() - (p_observation_window_days || ' days')::INTERVAL
        LIMIT p_batch_size
    ),
    downstream_activity AS (
        SELECT DISTINCT pc.prediction_id, true as has_activity
        FROM prediction_context pc
        JOIN treatment_events te
            ON te.hcp_id = pc.hcp_id
            AND te.event_date BETWEEN
                pc.prediction_timestamp::DATE
                AND (pc.prediction_timestamp + (p_observation_window_days || ' days')::INTERVAL)::DATE
    )
    SELECT
        pc.prediction_id,
        COALESCE(da.has_activity, false) as has_downstream_activity,
        CASE
            WHEN COALESCE(da.has_activity, false) THEN 'POSITIVE'
            ELSE 'NEGATIVE'
        END as outcome_label,
        0.70 as truth_confidence
    FROM prediction_context pc
    LEFT JOIN downstream_activity da ON da.prediction_id = pc.prediction_id;

    SELECT COUNT(*) INTO v_evaluated FROM temp_nba_candidates;
    SELECT COUNT(*) INTO v_labeled FROM temp_nba_candidates WHERE outcome_label IN ('POSITIVE', 'NEGATIVE');
    SELECT COUNT(*) INTO v_excluded FROM temp_nba_candidates WHERE outcome_label = 'EXCLUDED';

    UPDATE ml_predictions p
    SET
        actual_outcome = CASE
            WHEN tc.outcome_label = 'POSITIVE' THEN 1.0
            WHEN tc.outcome_label = 'NEGATIVE' THEN 0.0
            ELSE NULL
        END,
        outcome_recorded_at = NOW(),
        truth_source = 'treatment_events',
        truth_confidence = tc.truth_confidence,
        outcome_label = tc.outcome_label
    FROM temp_nba_candidates tc
    WHERE p.prediction_id = tc.prediction_id;

    DROP TABLE temp_nba_candidates;

    RETURN QUERY SELECT v_evaluated, v_labeled, v_excluded;
END;
$$;

COMMIT;
