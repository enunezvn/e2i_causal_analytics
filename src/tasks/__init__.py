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

# Chatbot DSPy optimization queue drainer (#1515): importing the module fires
# the @celery_app.task decorator so the "chatbot-optimization-drain" beat
# entry is discoverable by the Celery worker + beat. Without this line the
# beat entry would dead-letter. Fail-closed: the task itself is a logged no-op
# unless CHATBOT_OPT_DRAIN_ENABLED is set.
from src.tasks.chatbot_optimization_tasks import drain_chatbot_optimization_queue

# Operational KPI corpus sync (audit F3b): importing the module registers the
# Celery task so the worker discovers it for the beat schedule. sync_chunk_corpus
# (#1373) syncs the chat-RAG chunk substrate (text-embedding-3-small).
from src.tasks.corpus_ingestion_tasks import sync_chunk_corpus, sync_operational_corpus

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
# run_feedback_learning_cycle GENERATES the signals; run_dspy_prompt_optimization
# CONSUMES them — both are imported here so the Celery worker discovers both.
from src.tasks.dspy_optimization_tasks import (
    run_dspy_prompt_optimization,
    run_feedback_learning_cycle,
)
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

# Knowledge-graph emptiness sentinel (#1761): importing the module fires the
# @celery_app.task decorator so the "graph-emptiness-sentinel" beat entry is
# discoverable by the Celery worker + beat. Without this line the beat entry
# would dead-letter (guarded by test_beat_schedule_registration.py). The module
# imports nothing heavy at top level -- falkordb/redis load lazily inside the
# task, and the seed scripts run as subprocesses, never imports.
from src.tasks.graph_reseed_tasks import graph_emptiness_sentinel

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

# Routing-label loop (#1341 Phase 1): importing the module fires the
# @celery_app.task decorator so the nightly labeler is discoverable by the
# Celery worker + beat. Without this line the "routing-label-nightly" beat
# entry would dead-letter.
from src.tasks.routing_label_tasks import run_routing_label_cycle

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
    # run_feedback_learning_cycle GENERATES signals; run_dspy_prompt_optimization CONSUMES them.
    "run_feedback_learning_cycle",
    "run_dspy_prompt_optimization",
    # Routing-label loop (#1341 Phase 1)
    "run_routing_label_cycle",
    # Chatbot DSPy optimization queue drainer (#1515)
    "drain_chatbot_optimization_queue",
    # NPPES NPI taxonomy cache (issue #154)
    "refresh_npi_taxonomy_cache",
    # Risk-score prediction DB writes (issue #173)
    "write_risk_score_predictions",
    # Operational KPI corpus sync (audit F3b)
    "sync_operational_corpus",
    # Chat-RAG chunk corpus sync (#1373)
    "sync_chunk_corpus",
    # Knowledge-graph emptiness sentinel + self-heal reseed (#1761)
    "graph_emptiness_sentinel",
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
