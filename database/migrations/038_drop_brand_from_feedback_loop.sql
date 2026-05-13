-- ============================================================================
-- Migration 038: Drop brand references from migration-006 feedback-loop
--                infrastructure
-- ============================================================================
-- Issue #176 — Migration 006 SQL functions and indexes reference a `brand`
-- column on `ml_predictions` that is never added by any migration. A fresh
-- `psql -v ON_ERROR_STOP=1 --single-transaction` replay against an empty DB
-- (the runner mode at `scripts/run_migrations.sh:100`) would fail at the
-- two CREATE INDEX statements in 006 §2.4, aborting the transaction before
-- this migration could run. THE FIX therefore has TWO PARTS:
--
--   PART 1 (in `006_feedback_loop_infrastructure.sql`): drop `brand` from
--   the two index keys in place. This is the ONLY part of 006 that
--   immediate-DDL-fails on missing-column; the function bodies use
--   deferred name resolution and CREATE OR REPLACE FUNCTION succeeds at
--   definition time regardless. See codex pass-1 HIGH-1 (2026-05-13).
--
--   PART 2 (this migration, 038): `CREATE OR REPLACE FUNCTION` for the
--   five truth-assignment functions, stripping `p.brand` from SELECT and
--   `te.brand::text = pc.brand::text` from the joins. Without 038, the
--   functions create successfully but are dead-on-arrival when invoked
--   by `run_feedback_loop('risk')` (the only path exercised today via
--   `tests/integration/test_risk_score_feedback_loop.py`).
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
-- IMPORTANT SEMANTIC NOTE (codex pass-1 MEDIUM-1): stripping the
-- `te.brand::text = pc.brand::text` join changes the outcome semantics —
-- a same-HCP or same-patient prescription of ANY brand now counts toward
-- the truth label (formerly only the predicted brand counted). This
-- matches the droplet's hand-patched live function, so post-deploy
-- behaviour is consistent. It is NOT faithful to migration 006's
-- brand-scoped intent. A future migration that adds a real brand key to
-- `ml_predictions` and re-narrows these joins is recommended.
--
-- Re-applying this migration is idempotent because we use
-- `CREATE OR REPLACE FUNCTION` everywhere. A downgrade path is not
-- provided — restoring the brand-join would re-introduce the original
-- defect (broken function references), so there is no defensible forward
-- + backward symmetric pair.
--
-- After applying this migration,
-- `tests/integration/test_risk_score_feedback_loop.py` still passes both
-- branches of `_ml_predictions_has_brand_column` (the seeding code
-- already tolerates the column being absent), and the new explicit
-- regression pins
-- `test_assign_truth_*_has_no_brand_reference` lock the post-038 state
-- for all five functions.
-- ============================================================================

BEGIN;

-- ----------------------------------------------------------------------------
-- 0. Remediate brand-keyed indexes on EXISTING DBs (codex pass-2 MEDIUM-1).
--
-- Migration 006 §2.4 was edited in place so fresh-DB replays now create
-- brand-free indexes. But any DB that ALREADY recorded migration 006
-- before the §2.4 fix would have failed at index-creation and never
-- recorded 006 anyway (so this DROP is a no-op there). The droplet
-- production DB is a special case: 006 had been hand-patched there
-- after the original failure, so the indexes may or may not exist with
-- either definition. DROP IF EXISTS + recreate makes the post-038 state
-- deterministic regardless of starting point.
-- ----------------------------------------------------------------------------
DROP INDEX IF EXISTS idx_predictions_hcp_brand;
DROP INDEX IF EXISTS idx_predictions_patient_brand;

CREATE INDEX IF NOT EXISTS idx_predictions_hcp_brand
ON ml_predictions (hcp_id, prediction_timestamp)
WHERE prediction_type IN ('churn', 'trigger', 'next_best_action');

CREATE INDEX IF NOT EXISTS idx_predictions_patient_brand
ON ml_predictions (patient_id, prediction_timestamp)
WHERE prediction_type IN ('risk', 'propensity');

-- ----------------------------------------------------------------------------
-- 1. Replace assign_truth_hcp_churn (was: 006 §5.1)
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
-- 2. Replace assign_truth_script_conversion (was: 006 §5.2)
--    Out of scope for issue #176: this function ALSO references
--    `t.prediction_id` and `t.status` on the `triggers` table, which has
--    neither column (see database/core/e2i_ml_complete_v3_schema.sql:579 —
--    the real columns are `delivery_status` / `acceptance_status`, and
--    triggers has no FK back to ml_predictions). Calling
--    run_feedback_loop('trigger') will fail at execution time on this
--    schema drift. We strip brand here for consistency with the rest of
--    this migration but leave the t.* drift as a separate latent bug
--    (file a follow-on issue for "triggers schema drift in migration 006").
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
-- 3. Replace assign_truth_treatment_response (was: 006 §5.3 — risk path)
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
-- 4. Replace assign_truth_next_best_action (was: 006 §5.4)
--    Out of scope for issue #176: also depends on `t.prediction_id` and
--    `t.status` (see §2 note above). Strip brand only; t.* drift is a
--    separate latent bug.
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
-- 5. Replace assign_truth_market_share (was: 006 §5.5)
--
-- The 006-as-written function was double-broken:
--   1. SELECT p.brand — column doesn't exist on ml_predictions.
--   2. bm.measurement_date — `business_metrics` actually has `metric_date`
--      (database/core/e2i_ml_complete_v3_schema.sql:658).
--   Note: `bm.market_share` IS provided by migration 033 (033 line 279
--   `ALTER TABLE business_metrics ADD COLUMN IF NOT EXISTS market_share`),
--   so that column reference is fine. Codex pass-2 LOW-1 (2026-05-13).
--
--   3. Even if 1-2 were patched, joining baseline/outcome rows on
--      (region, month) alone over-matches: business_metrics has both
--      per-HCP and per-aggregate rows for the same brand/region/month
--      (codex pass-1 HIGH-3; see src/etl/business_metrics_per_hcp_etl.py),
--      so `UPDATE ... FROM` would pick an arbitrary source row and
--      multiply baseline × outcome matches per prediction.
--
-- Issue #176 is scoped to fixing the *fresh-replay blocker* and the
-- *dead-on-arrival call paths*, not to designing a correct market-share
-- truth pipeline. We replace this function with a stub that always
-- marks `market_share_impact` predictions EXCLUDED with a forensic
-- reason, so the function is safely callable end-to-end and the broken
-- 006 logic is decisively retired. A future PR (file separate issue:
-- "implement brand-scoped market_share truth assignment") can replace
-- this stub with a real implementation once `ml_predictions` has a real
-- brand key AND `business_metrics` rows are unambiguously aggregable
-- per (brand, region, month).
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
    -- Mark every eligible PENDING market_share_impact prediction as
    -- EXCLUDED. The original 006-as-written truth-assignment SQL
    -- referenced a missing brand-scope column on ml_predictions AND
    -- a missing measurement_date column on business_metrics; even with
    -- both patched, the (region, month) join over-matches per-HCP rows
    -- on business_metrics. Computing market_share truth correctly
    -- requires brand-scoping that is not currently representable on
    -- ml_predictions. See migration 038 header (issue #176) and the
    -- follow-on issue for the rebuild.
    -- Codex pass-2 MEDIUM-3: avoid the literal tokens p[dot]brand /
    -- bm[dot]market_share / bm[dot]measurement_date in this body so
    -- the parametrized pg_get_functiondef regression pin doesn't
    -- false-positive on this stub.
    CREATE TEMP TABLE temp_ms_candidates AS
    SELECT p.prediction_id
    FROM ml_predictions p
    WHERE p.prediction_type = 'market_share_impact'
      AND p.outcome_label = 'PENDING'
      AND p.prediction_timestamp < NOW() - (p_observation_window_days || ' days')::INTERVAL
    LIMIT p_batch_size;

    SELECT COUNT(*) INTO v_evaluated FROM temp_ms_candidates;

    UPDATE ml_predictions p
    SET
        outcome_recorded_at = NOW(),
        truth_source = 'business_metrics',
        truth_confidence = 0.0,
        outcome_label = 'EXCLUDED',
        exclusion_reason = 'market_share truth assignment disabled post-issue-176: requires brand-scoping on ml_predictions'
    FROM temp_ms_candidates tc
    WHERE p.prediction_id = tc.prediction_id;

    v_labeled := 0;
    v_excluded := v_evaluated;

    DROP TABLE temp_ms_candidates;

    RETURN QUERY SELECT v_evaluated, v_labeled, v_excluded;
END;
$$;

COMMIT;
