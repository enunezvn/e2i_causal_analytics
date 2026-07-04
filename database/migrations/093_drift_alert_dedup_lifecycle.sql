-- ============================================================================
-- 093_drift_alert_dedup_lifecycle.sql
-- Drift-alert storm fix: dedup-on-write in the trigger writer, honest PSI
-- messaging, retention-safe FKs, and a one-time purge of planted-model
-- monitoring noise (2026-07-04 investigation).
--
-- WHY: on 2026-07-04 the Home page showed "50 active alerts" — the API page
-- cap. The table actually held 10,080 active alerts, ALL created in one
-- 11-minute window by the 6-hourly drift sweep. Compounding causes, each
-- fixed in the paired app PR (fix/alert-storm-drift-sweep):
--   (a) the sweep enumerated 360 planted `synth_*_exp_*` showcase models
--       (E2I_INCLUDE_SYNTHETIC=true turned the shared provenance filter into
--       a no-op, defeating #894's documented sweep guard);
--   (b) the sweep was beat-scheduled TWICE ("monitor-drift" +
--       "drift-detection-sweep") so everything double-fired;
--   (c) this trigger inserts one alert per drifted feature per run with no
--       dedup, and nothing ever resolves alerts — monotonic accumulation.
--
-- WHAT THIS MIGRATION DOES:
--   1. Recreates create_drift_alert() with a NOT-EXISTS dedup guard (an
--      active alert with the same model + title already flags the condition)
--      and an honest message: PSI is a threshold statistic (warning >= 0.1,
--      critical >= 0.25), not a hypothesis test — the stored p_value is the
--      COMPANION KS test's. The old message rendered "Drift detected with
--      psi test ... p-value: 0.998", reading like a wildly non-significant
--      test had fired the alert.
--      The title templates are LOAD-BEARING: _justified_alert_titles() in
--      src/tasks/drift_monitoring_tasks.py mirrors them to auto-resolve
--      cleared alerts. Change them in both places or auto-resolve breaks.
--   2. Makes retention deletes safe: ml_monitoring_alerts.drift_history_id
--      and ml_retraining_history.alert_id become ON DELETE SET NULL so the
--      daily cleanup task can prune old drift history / resolved alerts
--      without FK violations.
--   3. One-time purge of monitoring rows tied to planted (is_synthetic)
--      registry models: alerts, drift history, and runs. These are sweep
--      artifacts against showcase models — noise, not monitoring history.
--      Idempotent: re-running deletes 0 rows once clean.
--
-- NOTE: no BEGIN/COMMIT here — scripts/run_migrations.sh applies migrations
-- with psql --single-transaction; script-level transaction control would
-- break that atomicity (guarded by test_migrations_no_inner_txn.py). Manual
-- applies must pass -1/--single-transaction.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Trigger writer: dedup + honest PSI message
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION create_drift_alert()
RETURNS TRIGGER AS $$
DECLARE
    v_title   varchar(500);
    v_message text;
BEGIN
    -- Only create alert for significant drift
    IF NEW.drift_detected AND NEW.severity IN ('medium', 'high', 'critical') THEN
        v_title := CASE NEW.drift_type
            WHEN 'data' THEN 'Data Drift Detected: ' || COALESCE(NEW.feature_name, 'Multiple Features')
            WHEN 'model' THEN 'Model Prediction Drift Detected'
            WHEN 'concept' THEN 'Concept Drift Detected: Feature-Target Relationship Changed'
        END;

        -- Honest message: PSI fires on the statistic's threshold, not on a
        -- p-value; the p_value column holds the companion KS test's result.
        IF NEW.test_type::text = 'psi' THEN
            v_message := 'Drift detected: PSI '
                || COALESCE(NEW.test_statistic::text, 'N/A')
                || ' (warning >= 0.1, critical >= 0.25)'
                || COALESCE('; companion KS-test p-value: ' || NEW.p_value::text, '');
        ELSE
            v_message := 'Drift detected with ' || NEW.test_type::text || ' test. '
                || 'Test statistic: ' || COALESCE(NEW.test_statistic::text, 'N/A')
                || ', p-value: ' || COALESCE(NEW.p_value::text, 'N/A');
        END IF;

        -- Dedup: an ACTIVE alert with the same (model, title) already flags
        -- this condition; re-inserting per sweep makes the alert list an
        -- append-only event log. Resolved/acknowledged alerts do not block a
        -- new occurrence.
        IF NOT EXISTS (
            SELECT 1
            FROM ml_monitoring_alerts a
            WHERE a.status = 'active'
              AND a.model_id IS NOT DISTINCT FROM NEW.model_id
              AND a.title = v_title
        ) THEN
            INSERT INTO ml_monitoring_alerts (
                alert_type,
                title,
                severity,
                model_id,
                experiment_id,
                deployment_id,
                drift_history_id,
                message,
                affected_features,
                drift_type,
                composite_drift_score,
                recommended_action,
                recommended_priority
            ) VALUES (
                'drift',
                v_title,
                NEW.severity,
                NEW.model_id,
                NEW.experiment_id,
                NEW.deployment_id,
                NEW.id,
                v_message,
                CASE WHEN NEW.feature_name IS NOT NULL THEN ARRAY[NEW.feature_name] ELSE NULL END,
                NEW.drift_type,
                NEW.drift_score,
                CASE NEW.severity
                    WHEN 'critical' THEN 'Immediate model retraining recommended'
                    WHEN 'high' THEN 'Schedule model retraining within 24 hours'
                    WHEN 'medium' THEN 'Monitor closely and consider retraining if drift persists'
                END,
                CASE NEW.severity
                    WHEN 'critical' THEN 'immediate'
                    WHEN 'high' THEN 'high'
                    WHEN 'medium' THEN 'medium'
                END
            );
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ----------------------------------------------------------------------------
-- 2. Retention-safe FKs (idempotent: drop-if-exists then re-add)
-- ----------------------------------------------------------------------------
ALTER TABLE ml_monitoring_alerts
    DROP CONSTRAINT IF EXISTS ml_monitoring_alerts_drift_history_id_fkey;
ALTER TABLE ml_monitoring_alerts
    ADD CONSTRAINT ml_monitoring_alerts_drift_history_id_fkey
    FOREIGN KEY (drift_history_id) REFERENCES ml_drift_history(id) ON DELETE SET NULL;

ALTER TABLE ml_retraining_history
    DROP CONSTRAINT IF EXISTS ml_retraining_history_alert_id_fkey;
ALTER TABLE ml_retraining_history
    ADD CONSTRAINT ml_retraining_history_alert_id_fkey
    FOREIGN KEY (alert_id) REFERENCES ml_monitoring_alerts(id) ON DELETE SET NULL;

-- ----------------------------------------------------------------------------
-- 3. One-time purge of planted-model monitoring noise (idempotent)
-- ----------------------------------------------------------------------------
DELETE FROM ml_monitoring_alerts a
USING ml_model_registry m
WHERE a.model_id = m.id
  AND m.is_synthetic;

DELETE FROM ml_drift_history d
USING ml_model_registry m
WHERE d.model_id = m.id
  AND m.is_synthetic;

DELETE FROM ml_monitoring_runs r
WHERE r.model_ids IS NOT NULL
  AND EXISTS (
      SELECT 1
      FROM ml_model_registry m
      WHERE m.id = ANY (r.model_ids)
        AND m.is_synthetic
  );
