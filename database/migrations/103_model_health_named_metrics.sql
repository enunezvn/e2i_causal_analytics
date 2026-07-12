-- Migration 103: expose latest named eval metrics on ml_model_health_dashboard
--
-- Problem: the view's single "latest metric" LATERAL returns whichever
-- ml_performance_metrics row for a model carries the newest measured_at —
-- frequently a confusion_matrix / roc_curve summary row written alongside the
-- scalar metrics. The /system-health Model Health card maps only
-- primary_metric IN ('accuracy','acc') or LIKE '%auc%' to a displayable value,
-- so 8 of the 12 gold-standard models rendered NO accuracy even though every
-- model carries a full scalar metric set (accuracy / auc_roc / f1 / ...).
--
-- Fix: append per-metric LATERALs exposing the latest accuracy, auc_roc and
-- f1 for each model. Existing columns keep their exact order and semantics
-- (CREATE OR REPLACE VIEW appends new columns at the end), so current
-- consumers of latest_metric_value / primary_metric are unaffected.
--
-- NOTE: no BEGIN/COMMIT here — the migration runner wraps files itself.

CREATE OR REPLACE VIEW ml_model_health_dashboard AS
SELECT m.id AS model_id,
    m.model_name,
    m.stage AS model_stage,
    COALESCE(d.has_drift, false) AS has_active_drift,
    d.max_severity::text AS max_drift_severity,
    d.drift_check_count,
    COALESCE(a.total_active, 0::bigint) AS active_alerts,
    COALESCE(a.critical_count, 0::bigint) AS critical_alerts,
    p.latest_metric_value,
    p.metric_name AS primary_metric,
    p.is_degraded AS performance_degraded,
        CASE
            WHEN a.critical_count > 0 THEN 'critical'::text
            WHEN a.high_count > 0 OR d.max_severity >= 'critical'::drift_severity_enum THEN 'warning'::text
            WHEN d.has_drift AND d.max_severity >= 'medium'::drift_severity_enum OR p.is_degraded THEN 'attention'::text
            ELSE 'healthy'::text
        END AS health_status,
    m.is_synthetic,
    acc.latest_accuracy,
    auc.latest_auc_roc,
    f1.latest_f1
   FROM ml_model_registry m
     LEFT JOIN ( SELECT ml_drift_history.model_id,
            bool_or(ml_drift_history.drift_detected) AS has_drift,
            max(ml_drift_history.severity) AS max_severity,
            count(*) AS drift_check_count
           FROM ml_drift_history
          WHERE ml_drift_history.created_at >= (now() - '24:00:00'::interval)
          GROUP BY ml_drift_history.model_id) d ON m.id = d.model_id
     LEFT JOIN ml_active_alerts_summary a ON m.id = a.model_id
     LEFT JOIN LATERAL ( SELECT ml_performance_metrics.metric_name,
            ml_performance_metrics.metric_value AS latest_metric_value,
            ml_performance_metrics.is_degraded
           FROM ml_performance_metrics
          WHERE ml_performance_metrics.model_id = m.id
          ORDER BY ml_performance_metrics.measured_at DESC
         LIMIT 1) p ON true
     LEFT JOIN LATERAL ( SELECT pm.metric_value AS latest_accuracy
           FROM ml_performance_metrics pm
          WHERE pm.model_id = m.id AND pm.metric_name = 'accuracy'
          ORDER BY pm.measured_at DESC
         LIMIT 1) acc ON true
     LEFT JOIN LATERAL ( SELECT pm.metric_value AS latest_auc_roc
           FROM ml_performance_metrics pm
          WHERE pm.model_id = m.id AND pm.metric_name = 'auc_roc'
          ORDER BY pm.measured_at DESC
         LIMIT 1) auc ON true
     LEFT JOIN LATERAL ( SELECT pm.metric_value AS latest_f1
           FROM ml_performance_metrics pm
          WHERE pm.model_id = m.id AND pm.metric_name = 'f1'
          ORDER BY pm.measured_at DESC
         LIMIT 1) f1 ON true
  WHERE m.stage = ANY (ARRAY['production'::model_stage_enum, 'staging'::model_stage_enum]);
