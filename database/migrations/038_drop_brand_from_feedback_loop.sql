-- ============================================================================
-- Migration 038: Drop brand references from migration-006 feedback-loop
--                infrastructure
-- ============================================================================
-- Issue #176 — Migration 006 SQL functions and indexes reference a `brand`
-- column on `ml_predictions` that is never added by any migration. A fresh
-- `alembic upgrade head` (or equivalent psql replay) against an empty DB
-- would fail at function-creation time, OR succeed with functions that are
-- dead-on-arrival when called against `ml_predictions` (selecting a
-- nonexistent `p.brand`).
--
-- Option A (add the column) was rejected: no code in `src/` populates
-- `brand` on `ml_predictions` (the PR #175 writer at
-- `src/tasks/risk_score_prediction_tasks.py` does not include it in the
-- insert column list); `src/agents/drift_monitor/connectors/supabase_connector.py`
-- references brand only in docstrings, not in any actual query; and the
-- live droplet Postgres has already been patched without the brand-join
-- (matches reality). Adding the column would force a new write-time lookup
-- from patient -> treatment_events to populate it, adding failure modes
-- with no current consumer.
--
-- Option B (this migration) — strip brand from migration 006:
--   * DROP the two indexes that include the nonexistent column.
--   * CREATE OR REPLACE FUNCTION for the 5 truth-assignment functions
--     plus the master orchestrator, omitting `p.brand` from SELECT and
--     dropping `AND te.brand::text = pc.brand::text` from the joins.
--
-- Forward-only. Re-applying this migration is idempotent because we use
-- `DROP INDEX IF EXISTS` and `CREATE OR REPLACE FUNCTION`. A downgrade
-- path is not provided — restoring the brand-join would re-introduce the
-- original defect (broken function references), so there is no defensible
-- forward + backward symmetric pair.
--
-- After applying this migration, `tests/integration/test_risk_score_feedback_loop.py`
-- still passes both branches of `_ml_predictions_has_brand_column` because
-- the seeding code already tolerates the column being absent.
-- ============================================================================

BEGIN;

-- ----------------------------------------------------------------------------
-- 1. Drop the two indexes that reference the nonexistent `brand` column.
-- ----------------------------------------------------------------------------
-- These are no-ops if the table never had `brand` (the fresh-DB case), or
-- if the indexes were already dropped by a runtime patch on the droplet.
DROP INDEX IF EXISTS idx_predictions_hcp_brand;
DROP INDEX IF EXISTS idx_predictions_patient_brand;

-- ----------------------------------------------------------------------------
-- 2. Replace assign_truth_hcp_churn (was: 006 §5.1)
--    Removes p.brand from SELECT and te.brand::text = pc.brand::text from
--    both treatment_events joins. Logic is otherwise identical.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION assign_truth_hcp_churn(
    p_observation_window_days INTEGER DEFAULT 90,
    p_decline_threshold DECIMAL DEFAULT 0.30,
    p_min_prior_scripts INTEGER DEFAULT 3,
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
    CREATE TEMP TABLE temp_churn_candidates AS
    WITH prediction_context AS (
        SELECT
            p.prediction_id,
            p.hcp_id,
            p.prediction_timestamp,
            p.prediction_value as predicted_churn_prob
        FROM ml_predictions p
        WHERE p.prediction_type = 'churn'
          AND p.outcome_label = 'PENDING'
          AND p.prediction_timestamp < NOW() - (p_observation_window_days || ' days')::INTERVAL
        LIMIT p_batch_size
    ),
    prior_activity AS (
        SELECT
            pc.prediction_id,
            COUNT(*) as trx_count_prior
        FROM prediction_context pc
        JOIN treatment_events te
            ON te.hcp_id = pc.hcp_id
            AND te.event_type = 'prescription'
            AND te.event_date BETWEEN
                (pc.prediction_timestamp - INTERVAL '90 days')::DATE
                AND pc.prediction_timestamp::DATE
        GROUP BY pc.prediction_id
    ),
    window_activity AS (
        SELECT
            pc.prediction_id,
            COUNT(*) as trx_count_window
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
        COALESCE(pa.trx_count_prior, 0) as trx_prior,
        COALESCE(wa.trx_count_window, 0) as trx_window,
        CASE
            WHEN COALESCE(pa.trx_count_prior, 0) < 1 THEN 'INDETERMINATE'
            WHEN COALESCE(wa.trx_count_window, 0) = 0
                 AND COALESCE(pa.trx_count_prior, 0) >= 1 THEN 'POSITIVE'
            WHEN COALESCE(pa.trx_count_prior, 0) >= p_min_prior_scripts
                 AND COALESCE(wa.trx_count_window, 0)::DECIMAL / pa.trx_count_prior < p_decline_threshold THEN 'POSITIVE'
            ELSE 'NEGATIVE'
        END as outcome_label,
        CASE
            WHEN COALESCE(pa.trx_count_prior, 0) >= 5 THEN 0.95
            WHEN COALESCE(pa.trx_count_prior, 0) >= 3 THEN 0.85
            WHEN COALESCE(pa.trx_count_prior, 0) >= 1 THEN 0.70
            ELSE 0.50
        END as truth_confidence
    FROM prediction_context pc
    LEFT JOIN prior_activity pa ON pa.prediction_id = pc.prediction_id
    LEFT JOIN window_activity wa ON wa.prediction_id = pc.prediction_id;

    SELECT COUNT(*) INTO v_evaluated FROM temp_churn_candidates;
    SELECT COUNT(*) INTO v_labeled FROM temp_churn_candidates WHERE outcome_label IN ('POSITIVE', 'NEGATIVE');
    SELECT COUNT(*) INTO v_excluded FROM temp_churn_candidates WHERE outcome_label = 'INDETERMINATE';

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
        outcome_label = tc.outcome_label,
        exclusion_reason = CASE
            WHEN tc.outcome_label = 'INDETERMINATE' THEN 'Insufficient prior activity (<1 TRx)'
            ELSE NULL
        END
    FROM temp_churn_candidates tc
    WHERE p.prediction_id = tc.prediction_id;

    DROP TABLE temp_churn_candidates;

    RETURN QUERY SELECT v_evaluated, v_labeled, v_excluded;
END;
$$;

-- ----------------------------------------------------------------------------
-- 3. Replace assign_truth_script_conversion (was: 006 §5.2)
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
            p.prediction_timestamp,
            t.trigger_id,
            t.status as trigger_status
        FROM ml_predictions p
        LEFT JOIN triggers t ON t.prediction_id = p.prediction_id
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
        pc.trigger_id,
        pc.trigger_status,
        COALESCE(ca.nrx_count_window, 0) as nrx_count,
        CASE
            WHEN pc.trigger_id IS NOT NULL AND pc.trigger_status = 'not_delivered' THEN 'EXCLUDED'
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
        outcome_label = tc.outcome_label,
        exclusion_reason = CASE
            WHEN tc.outcome_label = 'EXCLUDED' THEN 'Trigger not delivered'
            ELSE NULL
        END
    FROM temp_conversion_candidates tc
    WHERE p.prediction_id = tc.prediction_id;

    DROP TABLE temp_conversion_candidates;

    RETURN QUERY SELECT v_evaluated, v_labeled, v_excluded;
END;
$$;

-- ----------------------------------------------------------------------------
-- 4. Replace assign_truth_treatment_response (was: 006 §5.3 — risk path)
--    This is the function exercised by tests/integration/test_risk_score_feedback_loop.py
--    via run_feedback_loop('risk'). Matches the live droplet patched version
--    (join on patient_id + event_type + event_date; no brand match).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION assign_truth_treatment_response(
    p_observation_window_days INTEGER DEFAULT 180,
    p_pdc_threshold DECIMAL DEFAULT 0.80,
    p_max_gap_days INTEGER DEFAULT 60,
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
    CREATE TEMP TABLE temp_response_candidates AS
    WITH prediction_context AS (
        SELECT
            p.prediction_id,
            p.patient_id,
            p.prediction_timestamp
        FROM ml_predictions p
        WHERE p.prediction_type = 'risk'
          AND p.outcome_label = 'PENDING'
          AND p.prediction_timestamp < NOW() - (p_observation_window_days || ' days')::INTERVAL
        LIMIT p_batch_size
    ),
    fill_events AS (
        SELECT
            pc.prediction_id,
            te.event_date,
            COALESCE(te.duration_days, 30) as duration_days,
            ROW_NUMBER() OVER (PARTITION BY pc.prediction_id ORDER BY te.event_date) as fill_num
        FROM prediction_context pc
        JOIN treatment_events te
            ON te.patient_id = pc.patient_id
            AND te.event_type = 'prescription'
            AND te.event_date BETWEEN
                pc.prediction_timestamp::DATE
                AND (pc.prediction_timestamp + (p_observation_window_days || ' days')::INTERVAL)::DATE
    ),
    fill_pattern AS (
        SELECT
            prediction_id,
            COUNT(*) as fill_count,
            SUM(duration_days) as days_covered,
            MAX(next_date - event_date) as max_gap_days
        FROM (
            SELECT
                fe.*,
                LEAD(fe.event_date) OVER (PARTITION BY fe.prediction_id ORDER BY fe.event_date) as next_date
            FROM fill_events fe
        ) gaps
        GROUP BY prediction_id
    )
    SELECT
        pc.prediction_id,
        COALESCE(fp.fill_count, 0) as fill_count,
        COALESCE(fp.days_covered, 0) as days_covered,
        fp.max_gap_days,
        CASE
            WHEN fp.days_covered IS NOT NULL
            THEN LEAST(fp.days_covered::DECIMAL / p_observation_window_days, 1.0)
            ELSE 0
        END as pdc,
        CASE
            WHEN COALESCE(fp.fill_count, 0) = 0 THEN 'NEGATIVE'
            WHEN fp.days_covered::DECIMAL / p_observation_window_days >= p_pdc_threshold THEN 'POSITIVE'
            WHEN COALESCE(fp.max_gap_days, 999) <= p_max_gap_days THEN 'POSITIVE'
            ELSE 'NEGATIVE'
        END as outcome_label,
        CASE
            WHEN COALESCE(fp.fill_count, 0) >= 3 THEN 0.90
            WHEN COALESCE(fp.fill_count, 0) >= 1 THEN 0.75
            ELSE 0.60
        END as truth_confidence
    FROM prediction_context pc
    LEFT JOIN fill_pattern fp ON fp.prediction_id = pc.prediction_id;

    SELECT COUNT(*) INTO v_evaluated FROM temp_response_candidates;
    SELECT COUNT(*) INTO v_labeled FROM temp_response_candidates WHERE outcome_label IN ('POSITIVE', 'NEGATIVE');
    SELECT COUNT(*) INTO v_excluded FROM temp_response_candidates WHERE outcome_label NOT IN ('POSITIVE', 'NEGATIVE');

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
    FROM temp_response_candidates tc
    WHERE p.prediction_id = tc.prediction_id;

    DROP TABLE temp_response_candidates;

    RETURN QUERY SELECT v_evaluated, v_labeled, v_excluded;
END;
$$;

-- ----------------------------------------------------------------------------
-- 5. Replace assign_truth_next_best_action (was: 006 §5.4)
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
            p.prediction_timestamp,
            p.metadata->>'action_type' as action_type,
            t.trigger_id,
            t.status as trigger_status
        FROM ml_predictions p
        LEFT JOIN triggers t ON t.prediction_id = p.prediction_id
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
        pc.trigger_status,
        COALESCE(da.has_activity, false) as has_downstream_activity,
        CASE
            WHEN pc.trigger_status = 'accepted' AND COALESCE(da.has_activity, false) THEN 'POSITIVE'
            WHEN pc.trigger_status IS NULL THEN 'EXCLUDED'
            ELSE 'NEGATIVE'
        END as outcome_label,
        CASE
            WHEN pc.trigger_status = 'accepted' THEN 0.90
            ELSE 0.70
        END as truth_confidence
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
        truth_source = 'triggers_treatment_events',
        truth_confidence = tc.truth_confidence,
        outcome_label = tc.outcome_label,
        exclusion_reason = CASE
            WHEN tc.outcome_label = 'EXCLUDED' THEN 'No trigger generated'
            ELSE NULL
        END
    FROM temp_nba_candidates tc
    WHERE p.prediction_id = tc.prediction_id;

    DROP TABLE temp_nba_candidates;

    RETURN QUERY SELECT v_evaluated, v_labeled, v_excluded;
END;
$$;

-- ----------------------------------------------------------------------------
-- 6. Replace assign_truth_market_share (was: 006 §5.5)
--    business_metrics still has a real `brand` column (see
--    database/core/e2i_ml_complete_v3_schema.sql:879), but the join
--    `bm.brand::text = pc.brand::text` references `pc.brand` which comes
--    from `ml_predictions p.brand` — that's what's missing. We replace
--    the brand-keyed join with `region + measurement_date` only (already
--    the discriminating keys for the market-share lookup), keeping the
--    function consistent with the live droplet's stripped behaviour. If
--    a future migration adds a real brand-scoping column to
--    ml_predictions, market_share can be re-tightened in a follow-up.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION assign_truth_market_share(
    p_observation_window_days INTEGER DEFAULT 90,
    p_accuracy_threshold DECIMAL DEFAULT 0.02,
    p_batch_size INTEGER DEFAULT 500
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
    CREATE TEMP TABLE temp_ms_candidates AS
    WITH prediction_context AS (
        SELECT
            p.prediction_id,
            p.metadata->>'region' as region,
            p.prediction_timestamp,
            p.prediction_value as predicted_delta
        FROM ml_predictions p
        WHERE p.prediction_type = 'market_share_impact'
          AND p.outcome_label = 'PENDING'
          AND p.prediction_timestamp < NOW() - (p_observation_window_days || ' days')::INTERVAL
        LIMIT p_batch_size
    ),
    baseline_share AS (
        SELECT
            pc.prediction_id,
            bm.market_share as baseline_ms
        FROM prediction_context pc
        JOIN business_metrics bm
            ON bm.region = pc.region
            AND bm.measurement_date = DATE_TRUNC('month', pc.prediction_timestamp)
        WHERE bm.metric_type = 'market_share'
    ),
    outcome_share AS (
        SELECT
            pc.prediction_id,
            bm.market_share as outcome_ms
        FROM prediction_context pc
        JOIN business_metrics bm
            ON bm.region = pc.region
            AND bm.measurement_date = DATE_TRUNC('month',
                pc.prediction_timestamp + (p_observation_window_days || ' days')::INTERVAL)
        WHERE bm.metric_type = 'market_share'
    )
    SELECT
        pc.prediction_id,
        pc.predicted_delta,
        bs.baseline_ms,
        os.outcome_ms,
        CASE
            WHEN bs.baseline_ms IS NOT NULL AND os.outcome_ms IS NOT NULL
            THEN os.outcome_ms - bs.baseline_ms
            ELSE NULL
        END as actual_delta,
        CASE
            WHEN bs.baseline_ms IS NULL OR os.outcome_ms IS NULL THEN 'EXCLUDED'
            ELSE 'POSITIVE'
        END as outcome_label,
        0.95 as truth_confidence
    FROM prediction_context pc
    LEFT JOIN baseline_share bs ON bs.prediction_id = pc.prediction_id
    LEFT JOIN outcome_share os ON os.prediction_id = pc.prediction_id;

    SELECT COUNT(*) INTO v_evaluated FROM temp_ms_candidates;
    SELECT COUNT(*) INTO v_labeled FROM temp_ms_candidates WHERE outcome_label = 'POSITIVE';
    SELECT COUNT(*) INTO v_excluded FROM temp_ms_candidates WHERE outcome_label = 'EXCLUDED';

    UPDATE ml_predictions p
    SET
        actual_outcome = tc.actual_delta,
        outcome_recorded_at = NOW(),
        truth_source = 'business_metrics',
        truth_confidence = tc.truth_confidence,
        outcome_label = tc.outcome_label,
        exclusion_reason = CASE
            WHEN tc.outcome_label = 'EXCLUDED' THEN 'Missing baseline or outcome market share data'
            ELSE NULL
        END
    FROM temp_ms_candidates tc
    WHERE p.prediction_id = tc.prediction_id;

    DROP TABLE temp_ms_candidates;

    RETURN QUERY SELECT v_evaluated, v_labeled, v_excluded;
END;
$$;

COMMIT;
