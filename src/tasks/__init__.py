"""Celery Tasks for E2I Causal Analytics.

This module contains task definitions for background job processing.
Tasks are organized by domain and routed to appropriate worker queues.

Task Categories:
- Feature Store: Feast materialization and freshness checks
- Drift Monitoring: Drift detection, alerts, and cleanup (Phase 14)
- Performance Tracking: Model performance metrics and alerts (Phase 14)
- A/B Testing: Experiment execution, monitoring, and analysis (Phase 15)
"""

# Import tasks for auto-discovery.
#
# Sentinel action handlers (#375 iter-1 H2): importing the module is enough
# for the @celery_app.task decorators to fire and register the four
# plan-specced action handlers (``rerun_all_active_cohorts``,
# ``notify_and_queue_reanalysis``, ``flag_for_review``,
# ``run_full_consolidation``). Without this line, Celery worker boot — which
# imports ``src.tasks`` for task discovery — would NOT see the tasks, and
# ``send_task`` calls from the sentinel dispatcher's
# ``dispatch_agent → Celery`` bridge would dead-letter. Tests directly
# importing ``src.tasks.sentinel_actions`` mask the bug — always-on
# production import is the load-bearing line.
from src.tasks import sentinel_actions  # noqa: F401

# A/B Testing Tasks (Phase 15)
from src.tasks.ab_testing_tasks import (
    check_all_active_experiments,
    cleanup_old_ab_results,
    compute_experiment_results,
    enrollment_health_check,
    fidelity_tracking_update,
    scheduled_interim_analysis,
    srm_detection_sweep,
)

# Crystallization subsystem (#376 Phase 4): importing the module fires the
# @celery_app.task decorator so the portfolio task is discoverable by the
# Celery worker. Without this line, the worker boot would NOT see
# ``crystallize_portfolio`` and beat-driven dispatches would dead-letter.
# Memory: [[feat-375-phase3-hardening-close-20260519]].
from src.tasks.crystallization_tasks import crystallize_portfolio

# Drift Monitoring Tasks (Phase 14)
from src.tasks.drift_monitoring_tasks import (
    check_all_production_models,
    check_model_performance_alerts,
    check_retraining_for_all_models,
    cleanup_old_drift_history,
    evaluate_retraining_need,
    execute_model_retraining,
    run_drift_detection,
    send_drift_alert_notifications,
    track_model_performance,
)

# DSPy prompt self-improvement loop (audit F1 keystone): importing the module
# fires the @celery_app.task decorator so the scheduled optimization trigger is
# registered for the beat + worker.
from src.tasks.dspy_optimization_tasks import run_dspy_prompt_optimization
from src.tasks.feast_tasks import (
    check_feature_freshness,
    materialize_features,
    materialize_incremental_features,
)

# Feedback Loop Tasks (Concept Drift Detection)
from src.tasks.feedback_loop_tasks import (
    analyze_concept_drift_from_truth,
    run_feedback_loop_long_window,
    run_feedback_loop_medium_window,
    run_feedback_loop_short_window,
    run_full_feedback_loop,
)

# P2 heavy-compute offload (DARK by default): SHAP + twin-population simulation.
# Importing the module fires the @celery_app.task decorators so worker_heavy
# discovers ``src.tasks.compute_shap_values`` / ``src.tasks.simulate_population``
# (whose names match the pre-existing task_routes entries). The API only enqueues
# these when HEAVY_OFFLOAD_ENABLED is set; the import itself is inert.
from src.tasks.heavy_offload_tasks import (
    compute_shap_values,
    simulate_population,
)

# Insight lifecycle (consolidator + sentinel dispatcher + #378 reanalysis)
from src.tasks.insight_lifecycle_tasks import (
    consolidate_insights,
    reanalyze_finding,
    sentinel_dispatcher,
)

# NPPES NPI taxonomy cache (issue #154)
from src.tasks.nppes_tasks import refresh_npi_taxonomy_cache

# Risk-score prediction DB write task (issue #173)
from src.tasks.risk_score_prediction_tasks import write_risk_score_predictions

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
    # Feedback Loop (Concept Drift Detection)
    "run_feedback_loop_short_window",
    "run_feedback_loop_medium_window",
    "run_feedback_loop_long_window",
    "analyze_concept_drift_from_truth",
    "run_full_feedback_loop",
    # DSPy prompt self-improvement loop (audit F1 keystone)
    "run_dspy_prompt_optimization",
    # NPPES NPI taxonomy cache (issue #154)
    "refresh_npi_taxonomy_cache",
    # Risk-score prediction DB writes (issue #173)
    "write_risk_score_predictions",
    # Insight lifecycle subsystem
    "consolidate_insights",
    "sentinel_dispatcher",
    "reanalyze_finding",
    # Crystallization subsystem (#376 Phase 4)
    "crystallize_portfolio",
    # P2 heavy-compute offload (DARK by default)
    "compute_shap_values",
    "simulate_population",
]
