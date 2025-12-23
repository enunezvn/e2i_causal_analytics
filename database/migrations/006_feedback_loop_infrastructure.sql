-- ============================================================================
-- E2I Feedback Loop Migration Script
-- Purpose: Enable concept drift detection via ground truth labeling
-- Version: 1.0.0
-- Date: 2025-12-23
-- ============================================================================

-- ============================================================================
-- PART 1: SCHEMA EXTENSIONS
-- ============================================================================

-- 1.1 Add ground truth columns to ml_predictions
ALTER TABLE ml_predictions 
ADD COLUMN IF NOT EXISTS actual_outcome DECIMAL(5,4);

ALTER TABLE ml_predictions 
ADD COLUMN IF NOT EXISTS outcome_recorded_at TIMESTAMPTZ;

ALTER TABLE ml_predictions 
ADD COLUMN IF NOT EXISTS truth_source VARCHAR(50);

ALTER TABLE ml_predictions 
ADD COLUMN IF NOT EXISTS truth_confidence DECIMAL(3,2);

ALTER TABLE ml_predictions 
ADD COLUMN IF NOT EXISTS outcome_label VARCHAR(30);  -- POSITIVE, NEGATIVE, INDETERMINATE, EXCLUDED

ALTER TABLE ml_predictions 
ADD COLUMN IF NOT EXISTS exclusion_reason VARCHAR(100);  -- Why excluded from drift calc

-- 1.2 Add constraint for outcome_label
ALTER TABLE ml_predictions 
ADD CONSTRAINT chk_outcome_label 
CHECK (outcome_label IN ('POSITIVE', 'NEGATIVE', 'INDETERMINATE', 'EXCLUDED', 'PENDING'));

-- 1.3 Add constraint for truth_confidence range
ALTER TABLE ml_predictions 
ADD CONSTRAINT chk_truth_confidence 
CHECK (truth_confidence IS NULL OR (truth_confidence >= 0 AND truth_confidence <= 1));

-- 1.4 Set default for new predictions
ALTER TABLE ml_predictions 
ALTER COLUMN outcome_label SET DEFAULT 'PENDING';

-- ============================================================================
-- PART 2: INDEXES FOR FEEDBACK LOOP PERFORMANCE
-- ============================================================================

-- 2.1 Index for finding unlabeled predictions ready for truth assignment
CREATE INDEX IF NOT EXISTS idx_predictions_pending_truth 
ON ml_predictions (prediction_type, prediction_timestamp)
WHERE actual_outcome IS NULL AND outcome_label = 'PENDING';

-- 2.2 Index for drift analysis queries (labeled predictions)
CREATE INDEX IF NOT EXISTS idx_predictions_labeled 
ON ml_predictions (prediction_type, outcome_recorded_at, actual_outcome)
WHERE actual_outcome IS NOT NULL;

-- 2.3 Index for model performance tracking
CREATE INDEX IF NOT EXISTS idx_predictions_performance 
ON ml_predictions (model_version, prediction_type, prediction_timestamp)
INCLUDE (prediction_value, actual_outcome, truth_confidence);

-- 2.4 Composite index for truth queries joining to treatment_events
CREATE INDEX IF NOT EXISTS idx_predictions_hcp_brand 
ON ml_predictions (hcp_id, brand, prediction_timestamp)
WHERE prediction_type IN ('churn', 'trigger', 'next_best_action');

CREATE INDEX IF NOT EXISTS idx_predictions_patient_brand 
ON ml_predictions (patient_id, brand, prediction_timestamp)
WHERE prediction_type IN ('risk', 'propensity');

-- ============================================================================
-- PART 3: FEEDBACK LOOP CONFIGURATION TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_feedback_loop_config (
    config_id SERIAL PRIMARY KEY,
    prediction_type VARCHAR(50) NOT NULL UNIQUE,
    observation_window_days INTEGER NOT NULL,
    min_observation_days INTEGER NOT NULL,
    data_source_lag_days INTEGER NOT NULL DEFAULT 14,
    truth_query_template TEXT,
    schedule_cron VARCHAR(50),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert configuration for each model type
INSERT INTO ml_feedback_loop_config 
(prediction_type, observation_window_days, min_observation_days, data_source_lag_days, schedule_cron)
VALUES 
    ('churn', 90, 60, 14, '0 2 * * *'),           -- Daily at 2 AM
    ('trigger', 21, 14, 12, '0 */4 * * *'),       -- Every 4 hours
    ('next_best_action', 30, 14, 7, '0 */4 * * *'), -- Every 4 hours
    ('market_share_impact', 90, 60, 14, '0 3 * * 0'), -- Weekly Sunday 3 AM
    ('risk', 180, 90, 12, '0 3 * * 0')            -- Weekly Sunday 3 AM (treatment response)
ON CONFLICT (prediction_type) DO UPDATE SET
    observation_window_days = EXCLUDED.observation_window_days,
    min_observation_days = EXCLUDED.min_observation_days,
    updated_at = NOW();

-- ============================================================================
-- PART 4: FEEDBACK LOOP EXECUTION LOG
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_feedback_loop_runs (
    run_id SERIAL PRIMARY KEY,
    prediction_type VARCHAR(50) NOT NULL,
    run_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_completed_at TIMESTAMPTZ,
    predictions_evaluated INTEGER DEFAULT 0,
    predictions_labeled INTEGER DEFAULT 0,
    predictions_excluded INTEGER DEFAULT 0,
    predictions_indeterminate INTEGER DEFAULT 0,
    avg_truth_confidence DECIMAL(3,2),
    error_message TEXT,
    run_status VARCHAR(20) DEFAULT 'RUNNING',
    
    CONSTRAINT chk_run_status CHECK (run_status IN ('RUNNING', 'COMPLETED', 'FAILED', 'PARTIAL'))
);

CREATE INDEX IF NOT EXISTS idx_feedback_runs_type_time 
ON ml_feedback_loop_runs (prediction_type, run_started_at DESC);

-- ============================================================================
-- PART 5: TRUTH ASSIGNMENT FUNCTIONS
-- ============================================================================

-- 5.1 HCP Churn Truth Assignment
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
    -- Create temp table with predictions ready for labeling
    CREATE TEMP TABLE temp_churn_candidates AS
    WITH prediction_context AS (
        SELECT 
            p.prediction_id,
            p.hcp_id,
            p.brand,
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
            AND te.brand::text = pc.brand::text
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
            AND te.brand::text = pc.brand::text
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
        -- Determine outcome
        CASE 
            -- Insufficient baseline - indeterminate
            WHEN COALESCE(pa.trx_count_prior, 0) < 1 THEN 'INDETERMINATE'
            -- Zero activity with prior activity = churn
            WHEN COALESCE(wa.trx_count_window, 0) = 0 
                 AND COALESCE(pa.trx_count_prior, 0) >= 1 THEN 'POSITIVE'
            -- Significant decline = churn
            WHEN COALESCE(pa.trx_count_prior, 0) >= p_min_prior_scripts 
                 AND COALESCE(wa.trx_count_window, 0)::DECIMAL / pa.trx_count_prior < p_decline_threshold THEN 'POSITIVE'
            -- Otherwise = retained
            ELSE 'NEGATIVE'
        END as outcome_label,
        -- Calculate confidence
        CASE 
            WHEN COALESCE(pa.trx_count_prior, 0) >= 5 THEN 0.95
            WHEN COALESCE(pa.trx_count_prior, 0) >= 3 THEN 0.85
            WHEN COALESCE(pa.trx_count_prior, 0) >= 1 THEN 0.70
            ELSE 0.50
        END as truth_confidence
    FROM prediction_context pc
    LEFT JOIN prior_activity pa ON pa.prediction_id = pc.prediction_id
    LEFT JOIN window_activity wa ON wa.prediction_id = pc.prediction_id;

    -- Get counts
    SELECT COUNT(*) INTO v_evaluated FROM temp_churn_candidates;
    SELECT COUNT(*) INTO v_labeled FROM temp_churn_candidates WHERE outcome_label IN ('POSITIVE', 'NEGATIVE');
    SELECT COUNT(*) INTO v_excluded FROM temp_churn_candidates WHERE outcome_label = 'INDETERMINATE';

    -- Update ml_predictions
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

-- 5.2 Script Conversion Truth Assignment
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
            p.brand,
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
            AND te.brand::text = pc.brand::text
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
        -- Outcome determination
        CASE 
            -- Trigger not delivered - exclude
            WHEN pc.trigger_id IS NOT NULL AND pc.trigger_status = 'not_delivered' THEN 'EXCLUDED'
            -- Any conversion = positive
            WHEN COALESCE(ca.nrx_count_window, 0) >= 1 THEN 'POSITIVE'
            -- No conversion = negative
            ELSE 'NEGATIVE'
        END as outcome_label,
        -- Confidence is high for prescription events
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

-- 5.3 Treatment Response (Risk) Truth Assignment
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
            p.brand,
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
            AND te.brand::text = pc.brand::text
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
        -- Calculate PDC
        CASE 
            WHEN fp.days_covered IS NOT NULL 
            THEN LEAST(fp.days_covered::DECIMAL / p_observation_window_days, 1.0)
            ELSE 0
        END as pdc,
        -- Outcome determination
        CASE 
            -- No fills at all - check if therapy switch
            WHEN COALESCE(fp.fill_count, 0) = 0 THEN 'NEGATIVE'
            -- PDC threshold met
            WHEN fp.days_covered::DECIMAL / p_observation_window_days >= p_pdc_threshold THEN 'POSITIVE'
            -- Refill continuity maintained
            WHEN COALESCE(fp.max_gap_days, 999) <= p_max_gap_days THEN 'POSITIVE'
            -- Non-persistent
            ELSE 'NEGATIVE'
        END as outcome_label,
        -- Confidence based on fill count
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

-- 5.4 Next Best Action Truth Assignment
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
            p.brand,
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
            AND te.brand::text = pc.brand::text
            AND te.event_date BETWEEN 
                pc.prediction_timestamp::DATE 
                AND (pc.prediction_timestamp + (p_observation_window_days || ' days')::INTERVAL)::DATE
    )
    SELECT 
        pc.prediction_id,
        pc.trigger_status,
        COALESCE(da.has_activity, false) as has_downstream_activity,
        -- Outcome: accepted AND downstream activity
        CASE 
            WHEN pc.trigger_status = 'accepted' AND COALESCE(da.has_activity, false) THEN 'POSITIVE'
            WHEN pc.trigger_status IS NULL THEN 'EXCLUDED'  -- No trigger generated
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

-- 5.5 Market Share Impact Truth Assignment
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
            p.brand,
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
            ON bm.brand::text = pc.brand::text
            AND bm.region = pc.region
            AND bm.measurement_date = DATE_TRUNC('month', pc.prediction_timestamp)
        WHERE bm.metric_type = 'market_share'
    ),
    outcome_share AS (
        SELECT 
            pc.prediction_id,
            bm.market_share as outcome_ms
        FROM prediction_context pc
        JOIN business_metrics bm 
            ON bm.brand::text = pc.brand::text
            AND bm.region = pc.region
            AND bm.measurement_date = DATE_TRUNC('month', 
                pc.prediction_timestamp + (p_observation_window_days || ' days')::INTERVAL)
        WHERE bm.metric_type = 'market_share'
    )
    SELECT 
        pc.prediction_id,
        pc.predicted_delta,
        bs.baseline_ms,
        os.outcome_ms,
        -- Actual delta (continuous outcome)
        CASE 
            WHEN bs.baseline_ms IS NOT NULL AND os.outcome_ms IS NOT NULL 
            THEN os.outcome_ms - bs.baseline_ms
            ELSE NULL
        END as actual_delta,
        -- Label: did we have data to calculate?
        CASE 
            WHEN bs.baseline_ms IS NULL OR os.outcome_ms IS NULL THEN 'EXCLUDED'
            ELSE 'POSITIVE'  -- For continuous, we just store the actual value
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
        actual_outcome = tc.actual_delta,  -- Continuous value
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

-- ============================================================================
-- PART 6: MASTER FEEDBACK LOOP ORCHESTRATOR
-- ============================================================================

CREATE OR REPLACE FUNCTION run_feedback_loop(
    p_prediction_type VARCHAR DEFAULT NULL  -- NULL = run all
)
RETURNS TABLE (
    prediction_type VARCHAR,
    run_status VARCHAR,
    predictions_evaluated INTEGER,
    predictions_labeled INTEGER,
    predictions_excluded INTEGER,
    run_duration_seconds NUMERIC
) 
LANGUAGE plpgsql
AS $$
DECLARE
    v_run_id INTEGER;
    v_start_time TIMESTAMPTZ;
    v_evaluated INTEGER;
    v_labeled INTEGER;
    v_excluded INTEGER;
    v_config RECORD;
BEGIN
    -- Iterate through active configurations
    FOR v_config IN 
        SELECT * FROM ml_feedback_loop_config 
        WHERE is_active = true 
          AND (p_prediction_type IS NULL OR prediction_type = p_prediction_type)
    LOOP
        v_start_time := NOW();
        
        -- Log run start
        INSERT INTO ml_feedback_loop_runs (prediction_type, run_started_at)
        VALUES (v_config.prediction_type, v_start_time)
        RETURNING run_id INTO v_run_id;
        
        BEGIN
            -- Execute appropriate truth assignment function
            CASE v_config.prediction_type
                WHEN 'churn' THEN
                    SELECT * INTO v_evaluated, v_labeled, v_excluded 
                    FROM assign_truth_hcp_churn(v_config.observation_window_days);
                    
                WHEN 'trigger' THEN
                    SELECT * INTO v_evaluated, v_labeled, v_excluded 
                    FROM assign_truth_script_conversion(v_config.observation_window_days);
                    
                WHEN 'risk' THEN
                    SELECT * INTO v_evaluated, v_labeled, v_excluded 
                    FROM assign_truth_treatment_response(v_config.observation_window_days);
                    
                WHEN 'next_best_action' THEN
                    SELECT * INTO v_evaluated, v_labeled, v_excluded 
                    FROM assign_truth_next_best_action(v_config.observation_window_days);
                    
                WHEN 'market_share_impact' THEN
                    SELECT * INTO v_evaluated, v_labeled, v_excluded 
                    FROM assign_truth_market_share(v_config.observation_window_days);
                    
                ELSE
                    v_evaluated := 0;
                    v_labeled := 0;
                    v_excluded := 0;
            END CASE;
            
            -- Log run completion
            UPDATE ml_feedback_loop_runs
            SET 
                run_completed_at = NOW(),
                predictions_evaluated = v_evaluated,
                predictions_labeled = v_labeled,
                predictions_excluded = v_excluded,
                run_status = 'COMPLETED'
            WHERE run_id = v_run_id;
            
            RETURN QUERY SELECT 
                v_config.prediction_type::VARCHAR,
                'COMPLETED'::VARCHAR,
                v_evaluated,
                v_labeled,
                v_excluded,
                EXTRACT(EPOCH FROM (NOW() - v_start_time))::NUMERIC;
                
        EXCEPTION WHEN OTHERS THEN
            -- Log failure
            UPDATE ml_feedback_loop_runs
            SET 
                run_completed_at = NOW(),
                run_status = 'FAILED',
                error_message = SQLERRM
            WHERE run_id = v_run_id;
            
            RETURN QUERY SELECT 
                v_config.prediction_type::VARCHAR,
                'FAILED'::VARCHAR,
                0,
                0,
                0,
                EXTRACT(EPOCH FROM (NOW() - v_start_time))::NUMERIC;
        END;
    END LOOP;
END;
$$;

-- ============================================================================
-- PART 7: DRIFT DETECTION VIEWS
-- ============================================================================

-- 7.1 Concept Drift Metrics View
CREATE OR REPLACE VIEW v_concept_drift_metrics AS
WITH labeled_predictions AS (
    SELECT 
        prediction_type,
        model_version,
        DATE_TRUNC('week', prediction_timestamp) as prediction_week,
        prediction_value,
        actual_outcome,
        truth_confidence,
        outcome_label
    FROM ml_predictions
    WHERE actual_outcome IS NOT NULL
      AND outcome_label IN ('POSITIVE', 'NEGATIVE')
),
weekly_metrics AS (
    SELECT 
        prediction_type,
        model_version,
        prediction_week,
        COUNT(*) as sample_size,
        AVG(actual_outcome) as actual_positive_rate,
        AVG(prediction_value) as avg_predicted_prob,
        -- Accuracy (for binary: TP + TN / Total)
        AVG(CASE 
            WHEN (prediction_value >= 0.5 AND actual_outcome = 1) 
              OR (prediction_value < 0.5 AND actual_outcome = 0) 
            THEN 1.0 ELSE 0.0 
        END) as accuracy,
        -- Calibration error (predicted prob - actual rate)
        ABS(AVG(prediction_value) - AVG(actual_outcome)) as calibration_error,
        -- Brier score
        AVG(POWER(prediction_value - actual_outcome, 2)) as brier_score,
        AVG(truth_confidence) as avg_truth_confidence
    FROM labeled_predictions
    GROUP BY prediction_type, model_version, prediction_week
)
SELECT 
    wm.*,
    -- Week-over-week changes
    wm.accuracy - LAG(wm.accuracy) OVER (
        PARTITION BY wm.prediction_type, wm.model_version 
        ORDER BY wm.prediction_week
    ) as accuracy_delta,
    wm.calibration_error - LAG(wm.calibration_error) OVER (
        PARTITION BY wm.prediction_type, wm.model_version 
        ORDER BY wm.prediction_week
    ) as calibration_delta
FROM weekly_metrics wm;

-- 7.2 Model Performance Over Time View
CREATE OR REPLACE VIEW v_model_performance_tracking AS
SELECT 
    prediction_type,
    model_version,
    DATE_TRUNC('month', outcome_recorded_at) as label_month,
    COUNT(*) as total_labeled,
    SUM(CASE WHEN outcome_label = 'POSITIVE' THEN 1 ELSE 0 END) as true_positives,
    SUM(CASE WHEN outcome_label = 'NEGATIVE' THEN 1 ELSE 0 END) as true_negatives,
    SUM(CASE WHEN outcome_label = 'INDETERMINATE' THEN 1 ELSE 0 END) as indeterminate,
    SUM(CASE WHEN outcome_label = 'EXCLUDED' THEN 1 ELSE 0 END) as excluded,
    AVG(truth_confidence) as avg_confidence,
    -- Time to truth
    AVG(EXTRACT(EPOCH FROM (outcome_recorded_at - prediction_timestamp)) / 86400) as avg_days_to_truth
FROM ml_predictions
WHERE outcome_recorded_at IS NOT NULL
GROUP BY prediction_type, model_version, DATE_TRUNC('month', outcome_recorded_at);

-- 7.3 Drift Alert Thresholds View
CREATE OR REPLACE VIEW v_drift_alerts AS
WITH recent_metrics AS (
    SELECT * FROM v_concept_drift_metrics
    WHERE prediction_week >= NOW() - INTERVAL '30 days'
),
baseline_metrics AS (
    SELECT 
        prediction_type,
        model_version,
        AVG(accuracy) as baseline_accuracy,
        AVG(calibration_error) as baseline_calibration,
        AVG(actual_positive_rate) as baseline_class_rate
    FROM v_concept_drift_metrics
    WHERE prediction_week BETWEEN NOW() - INTERVAL '120 days' AND NOW() - INTERVAL '30 days'
    GROUP BY prediction_type, model_version
)
SELECT 
    rm.prediction_type,
    rm.model_version,
    rm.prediction_week,
    rm.accuracy as current_accuracy,
    bm.baseline_accuracy,
    (bm.baseline_accuracy - rm.accuracy) as accuracy_drop,
    CASE WHEN (bm.baseline_accuracy - rm.accuracy) > 0.05 THEN 'ALERT' ELSE 'OK' END as accuracy_status,
    rm.calibration_error as current_calibration,
    bm.baseline_calibration,
    CASE WHEN rm.calibration_error > bm.baseline_calibration + 0.10 THEN 'ALERT' ELSE 'OK' END as calibration_status,
    rm.actual_positive_rate as current_class_rate,
    bm.baseline_class_rate,
    ABS(rm.actual_positive_rate - bm.baseline_class_rate) as class_shift,
    CASE WHEN ABS(rm.actual_positive_rate - bm.baseline_class_rate) > 0.15 THEN 'ALERT' ELSE 'OK' END as class_shift_status
FROM recent_metrics rm
JOIN baseline_metrics bm 
    ON rm.prediction_type = bm.prediction_type 
    AND rm.model_version = bm.model_version;

-- ============================================================================
-- PART 8: UTILITY FUNCTIONS
-- ============================================================================

-- 8.1 Get pending predictions count by type
CREATE OR REPLACE FUNCTION get_pending_predictions_summary()
RETURNS TABLE (
    prediction_type VARCHAR,
    pending_count BIGINT,
    oldest_pending TIMESTAMPTZ,
    ready_for_labeling BIGINT
) 
LANGUAGE SQL
AS $$
    SELECT 
        p.prediction_type::VARCHAR,
        COUNT(*) as pending_count,
        MIN(p.prediction_timestamp) as oldest_pending,
        COUNT(*) FILTER (
            WHERE p.prediction_timestamp < NOW() - (c.observation_window_days || ' days')::INTERVAL
        ) as ready_for_labeling
    FROM ml_predictions p
    JOIN ml_feedback_loop_config c ON c.prediction_type = p.prediction_type
    WHERE p.outcome_label = 'PENDING'
    GROUP BY p.prediction_type;
$$;

-- 8.2 Clean up old run logs
CREATE OR REPLACE FUNCTION cleanup_feedback_loop_logs(
    p_retention_days INTEGER DEFAULT 90
)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_deleted INTEGER;
BEGIN
    DELETE FROM ml_feedback_loop_runs
    WHERE run_started_at < NOW() - (p_retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS v_deleted = ROW_COUNT;
    RETURN v_deleted;
END;
$$;

-- ============================================================================
-- PART 9: GRANTS (Adjust roles as needed)
-- ============================================================================

-- Grant execute on functions to application role
-- GRANT EXECUTE ON FUNCTION assign_truth_hcp_churn TO e2i_app_role;
-- GRANT EXECUTE ON FUNCTION assign_truth_script_conversion TO e2i_app_role;
-- GRANT EXECUTE ON FUNCTION assign_truth_treatment_response TO e2i_app_role;
-- GRANT EXECUTE ON FUNCTION assign_truth_next_best_action TO e2i_app_role;
-- GRANT EXECUTE ON FUNCTION assign_truth_market_share TO e2i_app_role;
-- GRANT EXECUTE ON FUNCTION run_feedback_loop TO e2i_app_role;

-- Grant select on views
-- GRANT SELECT ON v_concept_drift_metrics TO e2i_app_role;
-- GRANT SELECT ON v_model_performance_tracking TO e2i_app_role;
-- GRANT SELECT ON v_drift_alerts TO e2i_app_role;

-- ============================================================================
-- USAGE EXAMPLES
-- ============================================================================

/*
-- Run feedback loop for all model types:
SELECT * FROM run_feedback_loop();

-- Run feedback loop for specific model type:
SELECT * FROM run_feedback_loop('churn');

-- Check pending predictions:
SELECT * FROM get_pending_predictions_summary();

-- View drift alerts:
SELECT * FROM v_drift_alerts WHERE accuracy_status = 'ALERT' OR calibration_status = 'ALERT';

-- View concept drift metrics over time:
SELECT * FROM v_concept_drift_metrics 
WHERE prediction_type = 'churn' 
ORDER BY prediction_week DESC;

-- Check recent run history:
SELECT * FROM ml_feedback_loop_runs 
ORDER BY run_started_at DESC 
LIMIT 20;
*/

-- ============================================================================
-- END OF MIGRATION
-- ============================================================================
