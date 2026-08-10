"""
Celery Application Configuration
=================================

Multi-tier worker architecture with auto-scaling support.

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

import os

from celery import Celery
from celery.schedules import crontab
from kombu import Exchange, Queue

# Initialize Celery app
celery_app = Celery("e2i_causal_analytics")

# Redis connection from environment
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6382/1")
REDIS_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6382/2")

# =============================================================================
# CELERY CONFIGURATION
# =============================================================================

celery_app.conf.update(
    # Broker settings
    broker_url=REDIS_URL,
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    # Result backend
    result_backend=REDIS_BACKEND,
    result_expires=86400,  # 24 hours
    result_extended=True,
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Performance settings
    worker_prefetch_multiplier=1,  # Prefetch 1 task to avoid blocking
    task_acks_late=True,  # Acknowledge after completion
    task_reject_on_worker_lost=True,  # Requeue if worker crashes
    # Time limits
    task_time_limit=7200,  # 2 hours hard limit
    task_soft_time_limit=6600,  # 1h 50m soft limit
    # Retry settings
    task_autoretry_for=(Exception,),
    task_retry_kwargs={"max_retries": 3},
    task_retry_backoff=True,
    task_retry_backoff_max=600,  # 10 minutes max backoff
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# =============================================================================
# QUEUE DEFINITIONS
# =============================================================================

# Default exchange
default_exchange = Exchange("default", type="direct")

celery_app.conf.task_queues = (
    # Light worker queues
    Queue("default", exchange=default_exchange, routing_key="default"),
    Queue("quick", exchange=default_exchange, routing_key="quick"),
    Queue("api", exchange=default_exchange, routing_key="api"),
    # Medium worker queues
    Queue("analytics", exchange=default_exchange, routing_key="analytics"),
    Queue("reports", exchange=default_exchange, routing_key="reports"),
    Queue("aggregations", exchange=default_exchange, routing_key="aggregations"),
    # Heavy worker queues
    Queue("shap", exchange=default_exchange, routing_key="shap"),
    Queue("causal", exchange=default_exchange, routing_key="causal"),
    Queue("ml", exchange=default_exchange, routing_key="ml"),
    Queue("twins", exchange=default_exchange, routing_key="twins"),
    # Dead letter queue for failed tasks
    Queue("dead_letter", exchange=default_exchange, routing_key="dead_letter"),
)

# Default queue
celery_app.conf.task_default_queue = "default"
celery_app.conf.task_default_exchange = "default"
celery_app.conf.task_default_routing_key = "default"

# =============================================================================
# TASK ROUTING
# =============================================================================

celery_app.conf.task_routes = {
    # -------------------------------------------------------------------------
    # Light Worker Tasks (2 CPUs, 2GB RAM)
    # -------------------------------------------------------------------------
    # API-related tasks
    "src.tasks.api.*": {"queue": "api"},
    "src.tasks.fetch_*": {"queue": "api"},
    "src.tasks.get_*": {"queue": "api"},
    # Cache operations
    "src.tasks.cache.*": {"queue": "quick"},
    "src.tasks.invalidate_cache": {"queue": "quick"},
    "src.tasks.warm_cache": {"queue": "quick"},
    # Notifications
    "src.tasks.notify.*": {"queue": "quick"},
    "src.tasks.send_email": {"queue": "quick"},
    "src.tasks.send_alert": {"queue": "quick"},
    # Quick data operations
    "src.tasks.save_*": {"queue": "quick"},
    "src.tasks.update_*": {"queue": "quick"},
    "src.tasks.delete_*": {"queue": "quick"},
    # -------------------------------------------------------------------------
    # Medium Worker Tasks (4 CPUs, 8GB RAM)
    # -------------------------------------------------------------------------
    # Analytics and aggregations
    "src.tasks.calculate_metrics": {"queue": "analytics"},
    "src.tasks.aggregate_*": {"queue": "aggregations"},
    "src.tasks.compute_statistics": {"queue": "analytics"},
    # Report generation
    "src.tasks.generate_report": {"queue": "reports"},
    "src.tasks.export_report": {"queue": "reports"},
    "src.tasks.create_dashboard": {"queue": "reports"},
    # Data processing
    "src.tasks.process_batch": {"queue": "analytics"},
    "src.tasks.transform_data": {"queue": "analytics"},
    # -------------------------------------------------------------------------
    # Heavy Worker Tasks (16 CPUs, 32GB RAM)
    # -------------------------------------------------------------------------
    # SHAP explanations
    "src.tasks.shap_explain": {"queue": "shap"},
    "src.tasks.shap_explainer.*": {"queue": "shap"},
    "src.tasks.compute_shap_values": {"queue": "shap"},
    "src.tasks.shap_summary": {"queue": "shap"},
    # Causal inference
    "src.tasks.causal_refutation": {"queue": "causal"},
    "src.tasks.causal_sensitivity": {"queue": "causal"},
    "src.tasks.estimate_effect": {"queue": "causal"},
    "src.tasks.refutation.*": {"queue": "causal"},
    "src.tasks.sensitivity_analysis": {"queue": "causal"},
    "src.tasks.bootstrap_*": {"queue": "causal"},
    # ML training and cross-validation
    "src.tasks.train_model": {"queue": "ml"},
    "src.tasks.cross_validate_model": {"queue": "ml"},
    "src.tasks.hyperparameter_tune": {"queue": "ml"},
    "src.tasks.train_*": {"queue": "ml"},
    "src.tasks.fit_*": {"queue": "ml"},
    # Live retraining runs a full MLFoundationPipeline (scope→prep→train→deploy),
    # so it belongs on worker_heavy's `ml` queue, not the default queue. The
    # name doesn't match the train_*/fit_* globs, so route it explicitly.
    "src.tasks.execute_model_retraining": {"queue": "ml"},
    # Digital twin generation
    # (src.tasks.generate_twins removed — H15: dead route stub with no task body and
    # no producer; real population work is simulate_population / twin.* / train_twin_model.)
    "src.tasks.twin.*": {"queue": "twins"},
    "src.tasks.train_twin_model": {"queue": "ml"},
    "src.tasks.simulate_population": {"queue": "twins"},
    # Live twin retraining (#548) runs a full TwinGenerator.train (sklearn
    # ensemble), so it belongs on worker_heavy's `ml` queue. The name doesn't
    # match the train_*/twin.* globs, so route it explicitly.
    "src.tasks.execute_twin_retraining": {"queue": "ml"},
    # -------------------------------------------------------------------------
    # A/B Testing Tasks (Phase 15)
    # -------------------------------------------------------------------------
    # Interim analysis (medium compute)
    "src.tasks.scheduled_interim_analysis": {"queue": "analytics"},
    "src.tasks.compute_experiment_results": {"queue": "analytics"},
    # Health checks (quick)
    "src.tasks.enrollment_health_check": {"queue": "quick"},
    "src.tasks.srm_detection_sweep": {"queue": "quick"},
    "src.tasks.check_all_active_experiments": {"queue": "quick"},
    # Fidelity tracking (involves Digital Twin comparison)
    "src.tasks.fidelity_tracking_update": {"queue": "twins"},
    # Cleanup
    "src.tasks.cleanup_old_ab_results": {"queue": "quick"},
    # -------------------------------------------------------------------------
    # Feedback Loop Tasks (Concept Drift Detection)
    # -------------------------------------------------------------------------
    "src.tasks.run_feedback_loop_*": {"queue": "analytics"},
    "src.tasks.analyze_concept_drift_*": {"queue": "analytics"},
    "src.tasks.run_full_feedback_loop": {"queue": "analytics"},
    # DSPy self-improvement loop: signal-generation beat (every 6h) +
    # prompt-optimization beat (daily). Generator runs first; optimizer reads
    # the persisted signals and gates on GEPAOptimizationTrigger.
    "src.tasks.run_feedback_learning_cycle": {"queue": "analytics"},
    "src.tasks.run_dspy_prompt_optimization": {"queue": "analytics"},
    # Chatbot optimization queue drainer (#1515). The beat entry already pins
    # the queue via options.queue; this route covers the documented manual
    # trigger (`celery call src.tasks.drain_chatbot_optimization_queue`,
    # force=True) which would otherwise land on the default queue and run the
    # LLM-expensive GEPA executor on worker_light (codex #1515 iter-2 LOW).
    "src.tasks.drain_chatbot_optimization_queue": {"queue": "analytics"},
    # -------------------------------------------------------------------------
    # ETL Tasks (Block 6B-infra-2*: per-HCP business_metrics, per-patient
    # adherence, territory rollup). Beat schedules already pin these to
    # 'analytics' via options.queue; the wildcard here also routes any
    # CLI-dispatched (`celery call src.etl.*`) call to the same queue so
    # worker_medium picks them up. Without this entry the CLI path lands
    # the task in the default queue, which worker_medium does not consume.
    # -------------------------------------------------------------------------
    "src.etl.*": {"queue": "analytics"},
}

# =============================================================================
# BEAT SCHEDULE (for scheduler service)
# =============================================================================

celery_app.conf.beat_schedule = {
    # Drift monitoring every 6 hours. Targets check_all_production_models (the
    # real per-model drift sweep in src/tasks/drift_monitoring_tasks.py); the
    # prior "src.tasks.monitor_model_drift" was a dangling ref to a task that
    # never existed and would crash the scheduler when this entry fired.
    # This is the ONLY schedule entry for the sweep. A second entry
    # ("drift-detection-sweep", added via on_after_finalize in
    # drift_monitoring_tasks.py) used to double-fire the whole sweep every
    # cycle; that hook is gone — keep sweep scheduling here only.
    "monitor-drift": {
        "task": "src.tasks.check_all_production_models",
        "schedule": 21600.0,  # 6 hours
        "options": {"queue": "analytics"},
    },
    # Daily drift-monitoring retention: prunes old ml_drift_history rows,
    # resolved ml_monitoring_alerts, and old ml_monitoring_runs. Moved here
    # from drift_monitoring_tasks.py's removed on_after_finalize hook so all
    # beat scheduling lives in this one dict (see guard test
    # test_beat_schedule_registration.py).
    "drift-history-cleanup": {
        "task": "src.tasks.cleanup_old_drift_history",
        "schedule": 86400.0,  # 24 hours
        "options": {"queue": "quick"},
    },
    # NOTE (#897): the scaffolded "health-check" -> src.tasks.health_check and
    # "cache-cleanup" -> src.tasks.cleanup_old_cache entries were removed.
    # Neither task was ever defined in any commit; each tick enqueued a message
    # the worker rejected with KeyError ("Received unregistered task ...").
    # Their intent is already covered by live machinery: container healthchecks
    # (celery inspect ping), the API /health probe, the "queue-metrics" entry
    # below (5-min worker/queue stats), Redis result_expires +
    # celery.backend_cleanup, and the dedicated cleanup_old_ab_results /
    # cleanup_old_drift_history schedules.
    # tests/unit/test_workers/test_beat_schedule_registration.py guards the
    # whole class: every beat entry must reference a registered task.
    # Queue metrics every 5 minutes (for autoscaler)
    "queue-metrics": {
        "task": "src.tasks.collect_queue_metrics",
        "schedule": 300.0,  # 5 minutes
        "options": {"queue": "quick"},
    },
    # -------------------------------------------------------------------------
    # Feast Feature Store Tasks
    # -------------------------------------------------------------------------
    # Incremental feature materialization every 6 hours
    "feast-materialize-incremental": {
        "task": "src.tasks.materialize_incremental_features",
        "schedule": 21600.0,  # 6 hours
        "options": {"queue": "analytics"},
    },
    # Feature freshness check every 4 hours
    "feast-check-freshness": {
        "task": "src.tasks.check_feature_freshness",
        "schedule": 14400.0,  # 4 hours
        "kwargs": {"alert_on_stale": True},
        "options": {"queue": "analytics"},
    },
    # Full materialization weekly (Sunday at midnight UTC)
    "feast-materialize-full-weekly": {
        "task": "src.tasks.materialize_features",
        "schedule": 604800.0,  # 7 days
        "kwargs": {"feature_views": None},  # All feature views
        "options": {"queue": "ml"},
    },
    # -------------------------------------------------------------------------
    # ETL Tasks (block 6B-infra-2*)
    # -------------------------------------------------------------------------
    # Per-HCP business_metrics rollup every 24 hours (block 6B-infra-2a). Routed
    # to `analytics`, which `task_routes` consumes via worker_medium.
    "business-metrics-per-hcp-rollup": {
        "task": "src.etl.business_metrics_per_hcp_etl.run_per_hcp_rollup",
        "schedule": 86400.0,  # 24 hours
        "options": {"queue": "analytics"},
    },
    # Per-patient adherence/refill/gap derivation every 24 hours (block
    # 6B-infra-2b). Routed to `analytics` (worker_medium). Updates
    # patient_journeys.adherence_rate and gap_days; refill_count is left
    # NULL until a refill source lands -- see module docstring.
    "patient-adherence-rollup": {
        "task": "src.etl.patient_adherence_etl.run_patient_adherence_rollup",
        "schedule": 86400.0,  # 24 hours
        "options": {"queue": "analytics"},
    },
    # Territory_metrics rollup every 24 hours (block 6B-infra-2c). Routed
    # to `analytics` (worker_medium). Aggregates per-HCP business_metrics
    # rows produced by 6B-infra-2a -- in production the per-HCP rollup must
    # run first; see module docstring for the order dependency note.
    # market_potential / resource_allocation_score remain NULL until a real
    # Reltio/Veeva source lands.
    "territory-metrics-rollup": {
        "task": "src.etl.territory_metrics_etl.run_territory_rollup",
        "schedule": 86400.0,  # 24 hours
        "options": {"queue": "analytics"},
    },
    # Operational KPI corpus sync every 24 hours (audit F3b). Re-indexes the
    # latest snapshot of every (brand, metric_name, region) combo from
    # business_metrics into the RAG substrate (episodic_memories). Scheduled
    # AFTER the business_metrics ETL rollups above so it picks up fresh facts;
    # idempotent prose dedup means only new/changed snapshots are embedded.
    "sync-operational-corpus": {
        "task": "src.tasks.sync_operational_corpus",
        "schedule": 86400.0,  # 24 hours
        "options": {"queue": "analytics"},
    },
    # Chat-RAG chunk corpus sync every 24 hours (#1373). Re-indexes the latest
    # snapshot of every (brand, metric_name, region) combo from business_metrics
    # into rag_document_chunks -- the OTHER RAG substrate, embedded in the
    # text-embedding-3-small space the chat HybridRetriever queries in (the
    # episodic corpus above is in the memory-path ada-002 space, invisible to
    # chat). Scheduled AFTER the business_metrics ETL rollups; idempotent
    # content-hash dedup means only new/changed snapshots are embedded.
    "sync-chunk-corpus": {
        "task": "src.tasks.sync_chunk_corpus",
        "schedule": 86400.0,  # 24 hours
        "options": {"queue": "analytics"},
    },
    # -------------------------------------------------------------------------
    # A/B Testing Tasks (Phase 15)
    # -------------------------------------------------------------------------
    # Daily interim analysis check at 2 AM
    "ab-interim-analysis-check": {
        "task": "src.tasks.check_all_active_experiments",
        "schedule": 86400.0,  # 24 hours
        "options": {"queue": "quick"},
    },
    # Enrollment health check every 12 hours
    "ab-enrollment-health-check": {
        "task": "src.tasks.enrollment_health_check",
        "schedule": 43200.0,  # 12 hours
        "options": {"queue": "quick"},
    },
    # SRM detection every 6 hours
    "ab-srm-detection-sweep": {
        "task": "src.tasks.srm_detection_sweep",
        "schedule": 21600.0,  # 6 hours
        "options": {"queue": "quick"},
    },
    # Weekly A/B results cleanup (Sundays)
    "ab-results-cleanup": {
        "task": "src.tasks.cleanup_old_ab_results",
        "schedule": 604800.0,  # 7 days
        "options": {"queue": "quick"},
    },
    # -------------------------------------------------------------------------
    # Feedback Loop Tasks (Concept Drift Detection)
    # -------------------------------------------------------------------------
    # Short-window feedback loop every 4 hours (trigger, next_best_action)
    "feedback-loop-short-window": {
        "task": "src.tasks.run_feedback_loop_short_window",
        "schedule": 14400.0,  # 4 hours
        "options": {"queue": "analytics"},
    },
    # Medium-window feedback loop daily at 2 AM (churn)
    "feedback-loop-medium-window": {
        "task": "src.tasks.run_feedback_loop_medium_window",
        "schedule": 86400.0,  # 24 hours
        "options": {"queue": "analytics"},
    },
    # Long-window feedback loop weekly on Sundays (market_share_impact, risk)
    "feedback-loop-long-window": {
        "task": "src.tasks.run_feedback_loop_long_window",
        "schedule": 604800.0,  # 7 days
        "options": {"queue": "analytics"},
    },
    # Concept drift analysis after feedback loop (daily at 3 AM)
    "feedback-loop-drift-analysis": {
        "task": "src.tasks.analyze_concept_drift_from_truth",
        "schedule": 86400.0,  # 24 hours
        "options": {"queue": "analytics"},
    },
    # DSPy signal-generation beat — every 6h. Runs FeedbackLearnerAgent.learn()
    # to process user feedback and persist a training signal to
    # dspy_agent_training_signals (finalize-node persistence). The daily
    # optimize beat below READS those signals and gates on GEPAOptimizationTrigger
    # before running GEPA (MIPROv2 fallback). This entry must run BEFORE the optimize beat so
    # there are fresh signals to evaluate; the 6h cadence satisfies that
    # regardless of beat restart timing.
    "feedback-learning-cycle": {
        "task": "src.tasks.run_feedback_learning_cycle",
        "schedule": 21600.0,  # 6 hours
        "options": {"queue": "analytics"},
    },
    # DSPy prompt self-improvement loop (audit F1) — daily. Reads persisted
    # feedback_learner signals, gates on GEPAOptimizationTrigger, optimizes the
    # analysis prompts, and installs optimized recipient bundles.
    "dspy-prompt-optimization-daily": {
        "task": "src.tasks.run_dspy_prompt_optimization",
        "schedule": 86400.0,  # 24 hours
        "options": {"queue": "analytics"},
    },
    # Routing-label loop (#1341 Phase 1) — nightly labeler for
    # classification_logs. Populates was_correct/correct_pattern from
    # live-traffic outcome signals (explicit feedback > implicit outcome >
    # capped LLM judge) so v_classification_accuracy aggregates real data.
    # Fixed wall-clock slot (crontab, 04:30 UTC) rather than a relative
    # interval: the droplet also runs a 02:00 backup and a Mon-03:00 reseed,
    # and this task must stay off those windows. Gated on
    # ROUTING_LABEL_MIN_NEW_ROWS unlabeled rows; judge calls capped per run
    # via ROUTING_LABEL_JUDGE_CAP (token spend + droplet capacity).
    "routing-label-nightly": {
        "task": "src.tasks.run_routing_label_cycle",
        "schedule": crontab(hour=4, minute=30),
        "options": {"queue": "analytics"},
    },
    # Chatbot DSPy optimization queue drainer (#1515) — nightly. Polls the 035
    # chatbot_optimization_requests table (get_next_optimization_request),
    # claims via compare-and-set, executes ChatbotOptimizer.optimize_module
    # (GEPA — the path #1507 fixed) and closes out via
    # update_optimization_request_status. Fail-closed cost gate: the task is a
    # logged no-op unless CHATBOT_OPT_DRAIN_ENABLED is set (#1513 precedent),
    # and even then executes at most CHATBOT_OPT_DRAIN_MAX_PER_CYCLE (default
    # 1) GEPA runs per tick. Fixed wall-clock slot for the same reason as
    # routing-label-nightly: stay off the 02:00 backup, Mon-03:00 reseed and
    # 04:30 labeler windows.
    "chatbot-optimization-drain": {
        "task": "src.tasks.drain_chatbot_optimization_queue",
        "schedule": crontab(hour=5, minute=30),
        "options": {"queue": "analytics"},
    },
    # -------------------------------------------------------------------------
    # NPPES NPI taxonomy cache refresh (issue #154)
    # -------------------------------------------------------------------------
    # Monthly refresh of the npi_taxonomy table from the CMS NPPES bulk dump.
    # The task is a no-op stub when NPPES_BULK_DUMP_PATH is unset (e.g. in CI),
    # so the schedule is always wired but only fires real work in production.
    "nppes-refresh-monthly": {
        "task": "src.tasks.refresh_npi_taxonomy_cache",
        "schedule": 2592000.0,  # ~30 days
        "options": {"queue": "analytics"},
    },
    # -------------------------------------------------------------------------
    # Insight Lifecycle subsystem (consolidation + sentinels)
    # -------------------------------------------------------------------------
    # Daily consolidator: promotes confirmed causal_paths to semantic tier
    # and high-success procedural_memories to procedural tier.
    "insight-lifecycle-consolidate": {
        "task": "src.tasks.consolidate_insights",
        "schedule": 86400.0,  # 24 hours
        "options": {"queue": "analytics"},
    },
    # Sentinel dispatcher: evaluates data-driven watchers (threshold, freshness, etc.)
    # and fires actions (invalidate cascade, agent dispatch, notify). Runs every 5
    # minutes for near-real-time response to data changes.
    "insight-lifecycle-sentinels": {
        "task": "src.tasks.sentinel_dispatcher",
        "schedule": 300.0,  # 5 minutes
        "options": {"queue": "quick"},
    },
    # Crystallization (#376 Phase 4): aggregate cross-agent findings into
    # executive_insights every 6 hours on the analytics queue.
    #
    # Schedule semantics (codex iter-1 M3 honest-doc update):
    # This entry uses Celery beat's relative-interval form
    # (``schedule: 21600.0``) which runs every 6 hours measured from
    # the BEAT SCHEDULER start, NOT from a fixed wall-clock time. The
    # implementation does NOT enforce any offset relative to the daily
    # ``insight-lifecycle-consolidate`` entry; the two tasks run
    # independently and their phase relationship depends on beat
    # restart timing.
    #
    # The plan §Phase 4 line 141 specced a "30 min after
    # consolidation" offset; that framing is NOT a load-bearing
    # operational contract. If a fixed wall-clock offset becomes a
    # real ops requirement (e.g. CI runs prove a race condition
    # against the consolidator), replace this entry with a
    # ``celery.schedules.crontab`` form — but absent a demonstrated
    # need, the relative-interval form is simpler and idempotent.
    "crystallization-portfolio": {
        "task": "src.tasks.crystallization_tasks.crystallize_portfolio",
        "schedule": 21600.0,  # 6 hours (no strict offset to consolidator)
        "options": {"queue": "analytics"},
    },
}

# =============================================================================
# AUTO-DISCOVERY
# =============================================================================

# Auto-discover tasks in these modules
celery_app.autodiscover_tasks(
    [
        "src.tasks",
        "src.etl",
        "src.mlops",
        "src.causal",
        "src.digital_twin",
        "src.agents",
    ]
)

# =============================================================================
# MONITORING TASKS (G12)
# =============================================================================

# Register monitoring tasks for queue depth and worker metrics
try:
    from .monitoring import register_monitoring_tasks

    register_monitoring_tasks(celery_app)
except ImportError:
    pass  # Monitoring module not available

# =============================================================================
# CUSTOM CONFIGURATION
# =============================================================================


@celery_app.task(bind=True, name="src.tasks.debug_task")
def debug_task(self):
    """Debug task for testing worker connectivity."""
    return f"Request: {self.request!r}"


# Worker tier information
def get_worker_info():
    """Get current worker tier information."""
    worker_type = os.getenv("WORKER_TYPE", "unknown")
    return {
        "type": worker_type,
        "queues": {
            "light": ["default", "quick", "api"],
            "medium": ["analytics", "reports", "aggregations"],
            "heavy": ["shap", "causal", "ml", "twins"],
        }.get(worker_type, []),
    }


# =============================================================================
# DEAD LETTER QUEUE
# =============================================================================

import logging

from celery.exceptions import MaxRetriesExceededError
from celery.signals import task_failure

_dlq_logger = logging.getLogger("e2i.celery.dlq")


@task_failure.connect
def handle_task_failure(
    sender=None,
    task_id=None,
    exception=None,
    args=None,
    kwargs=None,
    traceback=None,
    einfo=None,
    **kw,
):
    """Route permanently failed tasks to the dead letter queue."""
    if isinstance(exception, MaxRetriesExceededError):
        _dlq_logger.warning(
            "Task %s (%s) exceeded max retries — routing to dead_letter queue",
            task_id,
            sender.name if sender else "unknown",
        )
        celery_app.send_task(
            "src.tasks.dead_letter_entry",
            queue="dead_letter",
            kwargs={
                "original_task": sender.name if sender else "unknown",
                "original_task_id": task_id,
                "original_args": str(args),
                "original_kwargs": str(kwargs),
                "exception": str(exception),
            },
        )


@celery_app.task(bind=True, name="src.tasks.dead_letter_entry")
def dead_letter_entry(self, **kwargs):
    """Placeholder task that sits in the dead_letter queue for inspection."""
    _dlq_logger.info("Dead letter entry: %s", kwargs)
    return kwargs


@celery_app.task(bind=True, name="src.tasks.monitor_dead_letter_queue")
def monitor_dead_letter_queue(self):
    """Monitor dead letter queue depth and log warnings."""
    try:
        with celery_app.connection_or_acquire() as conn:
            channel = conn.default_channel
            _, queue_depth, _ = channel.queue_declare(queue="dead_letter", passive=True)
            if queue_depth > 10:
                _dlq_logger.warning(
                    "Dead letter queue depth is %d — review failed tasks", queue_depth
                )
            return {"dead_letter_depth": queue_depth}
    except Exception as e:
        _dlq_logger.debug("Could not check DLQ depth: %s", e)
        return {"error": str(e)}


# Add DLQ monitor to beat schedule
celery_app.conf.beat_schedule["monitor-dead-letter-queue"] = {
    "task": "src.tasks.monitor_dead_letter_queue",
    "schedule": 1800.0,  # 30 minutes
    "options": {"queue": "quick"},
}
