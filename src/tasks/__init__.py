"""Celery Tasks for E2I Causal Analytics.

This module contains task definitions for background job processing.
Tasks are organized by domain and routed to appropriate worker queues.

Task Categories:
- Feature Store: Feast materialization and freshness checks
- Drift Monitoring: Drift detection, alerts, and cleanup (Phase 14)
- Performance Tracking: Model performance metrics and alerts (Phase 14)
- A/B Testing: Experiment execution, monitoring, and analysis (Phase 15)
"""

# Import tasks for auto-discovery
from src.tasks.feast_tasks import (
    materialize_features,
    materialize_incremental_features,
    check_feature_freshness,
)

# Drift Monitoring Tasks (Phase 14)
from src.tasks.drift_monitoring_tasks import (
    run_drift_detection,
    check_all_production_models,
    cleanup_old_drift_history,
    send_drift_alert_notifications,
    track_model_performance,
    check_model_performance_alerts,
    evaluate_retraining_need,
    execute_model_retraining,
    check_retraining_for_all_models,
)

# A/B Testing Tasks (Phase 15)
from src.tasks.ab_testing_tasks import (
    scheduled_interim_analysis,
    enrollment_health_check,
    srm_detection_sweep,
    compute_experiment_results,
    fidelity_tracking_update,
    check_all_active_experiments,
    cleanup_old_ab_results,
)

__all__ = [
    # Feature Store
    "materialize_features",
    "materialize_incremental_features",
    "check_feature_freshness",
    # Drift Monitoring
    "run_drift_detection",
    "check_all_production_models",
    "cleanup_old_drift_history",
    "send_drift_alert_notifications",
    # Performance Tracking
    "track_model_performance",
    "check_model_performance_alerts",
    # Retraining Triggers
    "evaluate_retraining_need",
    "execute_model_retraining",
    "check_retraining_for_all_models",
    # A/B Testing
    "scheduled_interim_analysis",
    "enrollment_health_check",
    "srm_detection_sweep",
    "compute_experiment_results",
    "fidelity_tracking_update",
    "check_all_active_experiments",
    "cleanup_old_ab_results",
]
