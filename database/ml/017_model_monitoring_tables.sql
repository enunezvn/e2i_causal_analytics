-- ============================================================================
-- Migration 017: Model Monitoring & Drift Detection Tables
--
-- Adds tables for tracking drift detection results, performance metrics over
-- time, and monitoring alerts for the E2I ML pipeline.
--
-- Phase 14: Model Monitoring & Drift Detection
-- Created: 2025-12-22
-- ============================================================================

-- ============================================================================
-- ENUM TYPES
-- ============================================================================

-- Drift type enum for categorizing drift detection results
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'drift_type_enum') THEN
        CREATE TYPE drift_type_enum AS ENUM (
            'data',      -- Feature distribution drift
            'model',     -- Prediction/score distribution drift
            'concept'    -- Feature-target relationship drift
        );
    END IF;
END$$;

-- Drift severity enum for categorizing drift magnitude
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'drift_severity_enum') THEN
        CREATE TYPE drift_severity_enum AS ENUM (
            'none',      -- No drift detected
            'low',       -- Minor drift, within acceptable range
            'medium',    -- Moderate drift, should monitor
            'high',      -- Significant drift, action recommended
            'critical'   -- Severe drift, immediate action required
        );
    END IF;
END$$;

-- Alert status enum for tracking alert lifecycle
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'alert_status_enum') THEN
        CREATE TYPE alert_status_enum AS ENUM (
            'active',      -- Alert is active and unacknowledged
            'acknowledged', -- Alert has been seen by a user
            'investigating', -- Alert is being investigated
            'resolved',    -- Alert has been resolved
            'dismissed'    -- Alert was dismissed as non-issue
        );
    END IF;
END$$;

-- Statistical test type enum
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'statistical_test_enum') THEN
        CREATE TYPE statistical_test_enum AS ENUM (
            'psi',           -- Population Stability Index
            'ks',            -- Kolmogorov-Smirnov test
            'chi_square',    -- Chi-square test
            'wasserstein',   -- Wasserstein/Earth Mover's Distance
            'js_divergence', -- Jensen-Shannon divergence
            'importance_correlation' -- Feature importance correlation
        );
    END IF;
END$$;

-- ============================================================================
-- TABLE: ml_drift_history
-- Stores drift detection results over time for trend analysis
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_drift_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Model and experiment identification
    model_id UUID REFERENCES ml_model_registry(id),
    experiment_id UUID REFERENCES ml_experiments(id),
    deployment_id UUID REFERENCES ml_deployments(id),

    -- Drift classification
    drift_type drift_type_enum NOT NULL,
    feature_name VARCHAR(255),  -- NULL for model/concept drift (overall)

    -- Statistical test results
    test_type statistical_test_enum NOT NULL,
    test_statistic DECIMAL(12, 6),
    p_value DECIMAL(12, 10),
    threshold DECIMAL(8, 4) DEFAULT 0.05,

    -- Drift assessment
    drift_detected BOOLEAN NOT NULL DEFAULT FALSE,
    severity drift_severity_enum NOT NULL DEFAULT 'none',

    -- Period comparison
    baseline_start TIMESTAMP WITH TIME ZONE NOT NULL,
    baseline_end TIMESTAMP WITH TIME ZONE NOT NULL,
    current_start TIMESTAMP WITH TIME ZONE NOT NULL,
    current_end TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Baseline statistics
    baseline_mean DECIMAL(15, 6),
    baseline_std DECIMAL(15, 6),
    baseline_min DECIMAL(15, 6),
    baseline_max DECIMAL(15, 6),
    baseline_count INTEGER,

    -- Current period statistics
    current_mean DECIMAL(15, 6),
    current_std DECIMAL(15, 6),
    current_min DECIMAL(15, 6),
    current_max DECIMAL(15, 6),
    current_count INTEGER,

    -- Additional metrics
    drift_score DECIMAL(8, 4),  -- Normalized 0-1 drift score
    contribution_to_overall DECIMAL(8, 4),  -- Feature contribution %

    -- Raw data reference (for debugging)
    raw_results JSONB DEFAULT '{}',

    -- Metadata
    detected_by VARCHAR(100) DEFAULT 'drift_monitor_agent',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes will be created separately
    CONSTRAINT valid_period CHECK (baseline_end <= current_start)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_drift_history_model ON ml_drift_history(model_id);
CREATE INDEX IF NOT EXISTS idx_drift_history_experiment ON ml_drift_history(experiment_id);
CREATE INDEX IF NOT EXISTS idx_drift_history_deployment ON ml_drift_history(deployment_id);
CREATE INDEX IF NOT EXISTS idx_drift_history_type ON ml_drift_history(drift_type);
CREATE INDEX IF NOT EXISTS idx_drift_history_severity ON ml_drift_history(severity);
CREATE INDEX IF NOT EXISTS idx_drift_history_feature ON ml_drift_history(feature_name);
CREATE INDEX IF NOT EXISTS idx_drift_history_detected ON ml_drift_history(drift_detected);
CREATE INDEX IF NOT EXISTS idx_drift_history_created ON ml_drift_history(created_at DESC);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_drift_history_model_type_time
    ON ml_drift_history(model_id, drift_type, created_at DESC);

-- ============================================================================
-- TABLE: ml_performance_metrics
-- Time-series storage for model performance tracking
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Model identification
    model_id UUID REFERENCES ml_model_registry(id),
    experiment_id UUID REFERENCES ml_experiments(id),
    deployment_id UUID REFERENCES ml_deployments(id),

    -- Metric details
    metric_name VARCHAR(100) NOT NULL,  -- 'roc_auc', 'precision', 'rmse', etc.
    metric_value DECIMAL(12, 6) NOT NULL,

    -- Context
    data_split VARCHAR(50) DEFAULT 'production',  -- 'train', 'val', 'test', 'production'
    segment VARCHAR(255),  -- Optional segmentation (brand, geography, etc.)

    -- Sample info
    sample_size INTEGER,
    positive_rate DECIMAL(8, 6),  -- For classification

    -- Confidence interval (if available)
    ci_lower DECIMAL(12, 6),
    ci_upper DECIMAL(12, 6),
    ci_level DECIMAL(4, 3) DEFAULT 0.95,

    -- Comparison to baseline
    baseline_value DECIMAL(12, 6),
    delta DECIMAL(12, 6),
    delta_pct DECIMAL(8, 4),
    is_degraded BOOLEAN DEFAULT FALSE,

    -- Timing
    measured_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    measurement_window_start TIMESTAMP WITH TIME ZONE,
    measurement_window_end TIMESTAMP WITH TIME ZONE,

    -- Source
    source VARCHAR(100) DEFAULT 'mlflow',  -- 'mlflow', 'bentoml', 'manual'

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for time-series queries
CREATE INDEX IF NOT EXISTS idx_perf_metrics_model ON ml_performance_metrics(model_id);
CREATE INDEX IF NOT EXISTS idx_perf_metrics_experiment ON ml_performance_metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_perf_metrics_deployment ON ml_performance_metrics(deployment_id);
CREATE INDEX IF NOT EXISTS idx_perf_metrics_name ON ml_performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_perf_metrics_measured ON ml_performance_metrics(measured_at DESC);
CREATE INDEX IF NOT EXISTS idx_perf_metrics_segment ON ml_performance_metrics(segment);
CREATE INDEX IF NOT EXISTS idx_perf_metrics_degraded ON ml_performance_metrics(is_degraded) WHERE is_degraded = TRUE;

-- Composite index for model performance over time
CREATE INDEX IF NOT EXISTS idx_perf_metrics_model_metric_time
    ON ml_performance_metrics(model_id, metric_name, measured_at DESC);

-- ============================================================================
-- TABLE: ml_monitoring_alerts
-- Alert history and lifecycle tracking
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_monitoring_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Alert identification
    alert_type VARCHAR(100) NOT NULL,  -- 'drift', 'performance', 'staleness', 'error'
    title VARCHAR(500) NOT NULL,

    -- Severity and status
    severity drift_severity_enum NOT NULL,
    status alert_status_enum NOT NULL DEFAULT 'active',

    -- Context
    model_id UUID REFERENCES ml_model_registry(id),
    experiment_id UUID REFERENCES ml_experiments(id),
    deployment_id UUID REFERENCES ml_deployments(id),
    drift_history_id UUID REFERENCES ml_drift_history(id),

    -- Alert details
    message TEXT NOT NULL,
    affected_features TEXT[],  -- Array of affected feature names
    drift_type drift_type_enum,
    composite_drift_score DECIMAL(8, 4),

    -- Recommendations
    recommended_action TEXT,
    recommended_priority VARCHAR(50),  -- 'immediate', 'high', 'medium', 'low'
    auto_action_taken BOOLEAN DEFAULT FALSE,
    auto_action_details JSONB,

    -- Notification tracking
    notified_channels TEXT[],  -- ['slack', 'email', 'pagerduty']
    notification_sent_at TIMESTAMP WITH TIME ZONE,
    notification_error TEXT,

    -- Acknowledgement
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(255),
    acknowledgement_notes TEXT,

    -- Resolution
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(255),
    resolution_notes TEXT,
    resolution_action VARCHAR(100),  -- 'retrained', 'rolled_back', 'dismissed', 'fixed_data'

    -- Retraining trigger
    triggered_retraining BOOLEAN DEFAULT FALSE,
    retraining_job_id UUID,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for alert management
CREATE INDEX IF NOT EXISTS idx_alerts_status ON ml_monitoring_alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON ml_monitoring_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON ml_monitoring_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_model ON ml_monitoring_alerts(model_id);
CREATE INDEX IF NOT EXISTS idx_alerts_deployment ON ml_monitoring_alerts(deployment_id);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON ml_monitoring_alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_active ON ml_monitoring_alerts(status, severity)
    WHERE status = 'active';

-- Composite index for dashboard queries
CREATE INDEX IF NOT EXISTS idx_alerts_model_status_time
    ON ml_monitoring_alerts(model_id, status, created_at DESC);

-- ============================================================================
-- TABLE: ml_monitoring_runs
-- Track monitoring job executions
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_monitoring_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Run identification
    run_type VARCHAR(100) NOT NULL,  -- 'data_drift', 'model_drift', 'concept_drift', 'performance', 'full'
    trigger_type VARCHAR(50) NOT NULL DEFAULT 'scheduled',  -- 'scheduled', 'manual', 'event'

    -- Scope
    model_ids UUID[],  -- Models checked in this run
    deployment_ids UUID[],

    -- Timing
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds DECIMAL(10, 2),

    -- Results summary
    total_checks INTEGER DEFAULT 0,
    drift_detected_count INTEGER DEFAULT 0,
    alerts_generated INTEGER DEFAULT 0,
    critical_count INTEGER DEFAULT 0,
    high_count INTEGER DEFAULT 0,
    medium_count INTEGER DEFAULT 0,
    low_count INTEGER DEFAULT 0,

    -- Overall assessment
    overall_health_score DECIMAL(5, 2),  -- 0-100
    requires_attention BOOLEAN DEFAULT FALSE,

    -- Status
    status VARCHAR(50) DEFAULT 'running',  -- 'running', 'completed', 'failed'
    error_message TEXT,

    -- Metadata
    config JSONB DEFAULT '{}',  -- Configuration used for this run
    summary JSONB DEFAULT '{}',  -- Detailed summary results
    created_by VARCHAR(100) DEFAULT 'celery_beat',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_monitoring_runs_type ON ml_monitoring_runs(run_type);
CREATE INDEX IF NOT EXISTS idx_monitoring_runs_status ON ml_monitoring_runs(status);
CREATE INDEX IF NOT EXISTS idx_monitoring_runs_started ON ml_monitoring_runs(started_at DESC);

-- ============================================================================
-- TABLE: ml_retraining_history
-- Track automated retraining triggered by monitoring
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_retraining_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- What triggered retraining
    trigger_type VARCHAR(100) NOT NULL,  -- 'drift', 'performance', 'scheduled', 'manual'
    alert_id UUID REFERENCES ml_monitoring_alerts(id),
    monitoring_run_id UUID REFERENCES ml_monitoring_runs(id),

    -- Model being retrained
    model_id UUID REFERENCES ml_model_registry(id),
    old_model_version VARCHAR(100),
    new_model_version VARCHAR(100),

    -- Reason
    trigger_reason TEXT NOT NULL,
    drift_severity drift_severity_enum,
    performance_delta DECIMAL(8, 4),

    -- Training job
    training_run_id UUID REFERENCES ml_training_runs(id),

    -- Results
    status VARCHAR(50) DEFAULT 'pending',  -- 'pending', 'training', 'validating', 'deployed', 'failed', 'rejected'
    old_metric_value DECIMAL(12, 6),
    new_metric_value DECIMAL(12, 6),
    improvement DECIMAL(8, 4),

    -- Deployment
    auto_deployed BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMP WITH TIME ZONE,
    deployment_id UUID REFERENCES ml_deployments(id),

    -- Timing
    triggered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds DECIMAL(10, 2),

    -- Metadata
    config JSONB DEFAULT '{}',
    notes TEXT,
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_retraining_trigger ON ml_retraining_history(trigger_type);
CREATE INDEX IF NOT EXISTS idx_retraining_model ON ml_retraining_history(model_id);
CREATE INDEX IF NOT EXISTS idx_retraining_status ON ml_retraining_history(status);
CREATE INDEX IF NOT EXISTS idx_retraining_triggered ON ml_retraining_history(triggered_at DESC);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View: Latest drift status per model
CREATE OR REPLACE VIEW ml_drift_status_latest AS
SELECT DISTINCT ON (model_id, drift_type)
    model_id,
    drift_type,
    feature_name,
    drift_detected,
    severity,
    test_type,
    test_statistic,
    p_value,
    drift_score,
    created_at as last_checked
FROM ml_drift_history
ORDER BY model_id, drift_type, created_at DESC;

-- View: Drift trend summary (last 7 days)
CREATE OR REPLACE VIEW ml_drift_trend_7d AS
SELECT
    model_id,
    drift_type,
    DATE(created_at) as check_date,
    COUNT(*) as check_count,
    SUM(CASE WHEN drift_detected THEN 1 ELSE 0 END) as drift_count,
    AVG(drift_score) as avg_drift_score,
    MAX(severity::text) as max_severity
FROM ml_drift_history
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY model_id, drift_type, DATE(created_at)
ORDER BY model_id, drift_type, check_date DESC;

-- View: Active alerts summary
CREATE OR REPLACE VIEW ml_active_alerts_summary AS
SELECT
    model_id,
    COUNT(*) as total_active,
    SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_count,
    SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_count,
    SUM(CASE WHEN severity = 'medium' THEN 1 ELSE 0 END) as medium_count,
    MIN(created_at) as oldest_alert,
    MAX(created_at) as newest_alert
FROM ml_monitoring_alerts
WHERE status = 'active'
GROUP BY model_id;

-- View: Model health dashboard
CREATE OR REPLACE VIEW ml_model_health_dashboard AS
SELECT
    m.id as model_id,
    m.name as model_name,
    m.stage as model_stage,
    -- Latest drift status
    COALESCE(d.has_drift, FALSE) as has_active_drift,
    d.max_drift_severity,
    d.drift_check_count,
    -- Active alerts
    COALESCE(a.total_active, 0) as active_alerts,
    COALESCE(a.critical_count, 0) as critical_alerts,
    -- Latest performance
    p.latest_metric_value,
    p.metric_name as primary_metric,
    p.is_degraded as performance_degraded,
    -- Overall health
    CASE
        WHEN a.critical_count > 0 THEN 'critical'
        WHEN a.high_count > 0 OR d.max_drift_severity = 'critical' THEN 'warning'
        WHEN d.has_drift OR p.is_degraded THEN 'attention'
        ELSE 'healthy'
    END as health_status
FROM ml_model_registry m
LEFT JOIN (
    SELECT
        model_id,
        bool_or(drift_detected) as has_drift,
        MAX(severity::text) as max_drift_severity,
        COUNT(*) as drift_check_count
    FROM ml_drift_history
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    GROUP BY model_id
) d ON m.id = d.model_id
LEFT JOIN ml_active_alerts_summary a ON m.id = a.model_id
LEFT JOIN LATERAL (
    SELECT metric_name, metric_value as latest_metric_value, is_degraded
    FROM ml_performance_metrics
    WHERE model_id = m.id
    ORDER BY measured_at DESC
    LIMIT 1
) p ON true
WHERE m.stage IN ('production', 'staging');

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Update updated_at timestamp for alerts
CREATE OR REPLACE FUNCTION update_monitoring_alerts_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_monitoring_alerts_updated_at
    BEFORE UPDATE ON ml_monitoring_alerts
    FOR EACH ROW
    EXECUTE FUNCTION update_monitoring_alerts_updated_at();

-- Function: Auto-generate alert from drift detection
CREATE OR REPLACE FUNCTION create_drift_alert()
RETURNS TRIGGER AS $$
BEGIN
    -- Only create alert for significant drift
    IF NEW.drift_detected AND NEW.severity IN ('medium', 'high', 'critical') THEN
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
            CASE NEW.drift_type
                WHEN 'data' THEN 'Data Drift Detected: ' || COALESCE(NEW.feature_name, 'Multiple Features')
                WHEN 'model' THEN 'Model Prediction Drift Detected'
                WHEN 'concept' THEN 'Concept Drift Detected: Feature-Target Relationship Changed'
            END,
            NEW.severity,
            NEW.model_id,
            NEW.experiment_id,
            NEW.deployment_id,
            NEW.id,
            'Drift detected with ' || NEW.test_type::text || ' test. '
                || 'Test statistic: ' || COALESCE(NEW.test_statistic::text, 'N/A')
                || ', p-value: ' || COALESCE(NEW.p_value::text, 'N/A'),
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
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_create_drift_alert
    AFTER INSERT ON ml_drift_history
    FOR EACH ROW
    EXECUTE FUNCTION create_drift_alert();

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE ml_drift_history IS
'Time-series storage of drift detection results for data, model, and concept drift';

COMMENT ON TABLE ml_performance_metrics IS
'Time-series storage of model performance metrics for trend analysis and degradation detection';

COMMENT ON TABLE ml_monitoring_alerts IS
'Alert history with full lifecycle tracking from creation to resolution';

COMMENT ON TABLE ml_monitoring_runs IS
'Monitoring job execution history for auditing and debugging';

COMMENT ON TABLE ml_retraining_history IS
'Automated retraining events triggered by monitoring alerts';

COMMENT ON COLUMN ml_drift_history.drift_score IS
'Normalized drift score from 0 (no drift) to 1 (maximum drift)';

COMMENT ON COLUMN ml_drift_history.test_type IS
'Statistical test used: psi (Population Stability Index), ks (Kolmogorov-Smirnov), chi_square, etc.';

COMMENT ON COLUMN ml_monitoring_alerts.auto_action_taken IS
'Whether an automated action (like retraining) was triggered by this alert';

COMMENT ON VIEW ml_model_health_dashboard IS
'Dashboard view combining drift status, alerts, and performance for production models';
