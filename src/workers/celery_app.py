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
#
# WALL-CLOCK SLOT MAP (all times UTC — conf.timezone above is "UTC")
#
# #1645: every daily entry here used to be a bare ``86400.0`` interval. Celery's
# interval form is measured from ``last_run_at``, which PersistentScheduler keeps
# in its state file — and that file lived in the scheduler container's *ephemeral*
# /tmp, so every deploy reset it. On a box that deploys several times a day a
# 24-hour interval can therefore NEVER become due (measured in the issue: over a
# 5-hour container life only the <=4h entries ever fired, and the two 4h entries
# fired exactly once each). The observable casualty was #1649:
# ``sync_operational_corpus`` is scheduled, implemented and its queue consumed,
# and it had never run.
#
# The fix has two halves and needs both:
#   1. docker/docker-compose.yml moves the beat state file onto the named volume
#      ``celerybeat_state`` (--schedule=/app/data/celerybeat/celerybeat-schedule),
#      so ``last_run_at`` survives a container recreate. A crontab slot missed
#      while the scheduler was down then fires immediately on the next start
#      (verified against celery 5.6.0: crontab.is_due() with a >24h-old
#      last_run_at returns is_due=True).
#   2. this map — daily work lands at a fixed hour instead of "deploy time + 24h",
#      and dependency order between entries becomes real rather than incidental
#      (with intervals every daily entry fired on the same tick, in dict order,
#      which guarantees nothing about completion order).
#
# Half (2) is also load management, not cosmetics. Every entry in the live state
# file carried last_run_at=02:03:53.7333xx — identical to the sub-millisecond,
# because beat stamps them all at startup. Persistence alone (half 1) would leave
# the eleven daily entries in that lockstep forever: all eleven come due at the
# same instant every day, on a worker_medium running --concurrency=2. The slots
# below are what breaks the lockstep.
#
# First-fire behaviour, measured against celery 5.6.0's PersistentScheduler:
#   * FIRST deploy after this change -> the volume is empty, beat stamps every
#     entry last_run_at=now, and 0 of 11 are immediately due. No catch-up burst;
#     each task first runs at its own slot later that day.
#   * routine redeploy (20 min / 2 h gap with no slot inside it) -> 0 of 11.
#   * gap that spans a slot -> exactly the entries whose slot fell inside it.
#   * multi-day outage -> 11 of 11 catch up, each exactly ONCE (crontab fires the
#     missed slot, not one fire per missed day).
#
# Slots already spoken for — this map must stay clear of them (measured 2026-08-16):
#   02:00-02:02  host cron, daily:   scripts/backup_cron.sh (pg_dump + RDB dumps)
#   03:00-03:02  host cron, MONDAYS: scripts/reseed_synthetic.sh (frontier append)
#   04:30        beat:               routing-label-nightly    (analytics, LLM judge)
#   05:30        beat:               chatbot-optimization-drain (analytics, GEPA)
#
# Daily entries, in firing order:
#   00:45  drift-history-cleanup            quick      prune before the 02:00 backup
#   01:15  ab-interim-analysis-check        quick      quiet hours, clear of the backup
#   02:10  feedback-loop-medium-window      analytics  "2 AM daily" per config, post-backup
#   02:40  feedback-loop-drift-analysis     analytics  after medium-window
#   03:15  business-metrics-per-hcp-rollup  analytics  after the Monday reseed
#   03:30  patient-adherence-rollup         analytics
#   03:45  territory-metrics-rollup         analytics  after per-HCP (hard dependency)
#   04:00  sync-operational-corpus          analytics  after the rollups
#   04:15  sync-chunk-corpus                analytics  after the rollups
#   06:00  dspy-prompt-optimization-daily   analytics  after the 05:30 GEPA drain
#   06:30  insight-lifecycle-consolidate    analytics
#
# The ETL -> corpus chain sits at 03:15-04:15 (not in the 00:xx quiet band) on
# purpose: the Monday 03:00 reseed appends a fresh data frontier, and a chain that
# ran before it would leave the corpus a full day behind every Monday.
# worker_medium runs --concurrency=2 on the analytics queue, so the 15-minute
# spacing buys determinism and ordering, not queue capacity.
#
# Guard: tests/unit/test_workers/test_beat_daily_wallclock_1645.py (no daily entry
# may regress to a bare interval; slots must stay distinct and off the reserved
# windows) and tests/unit/test_docker/test_compose_beat_state_volume_1645.py (the
# state file must live on a named volume, never tmpfs).
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
    # Slot 00:45 UTC (#1645): a pure retention prune with no upstream, placed
    # ahead of the 02:00 host backup so the dump captures an already-pruned DB.
    "drift-history-cleanup": {
        "task": "src.tasks.cleanup_old_drift_history",
        "schedule": crontab(hour=0, minute=45),
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
    # Per-HCP business_metrics rollup, daily at 03:15 UTC (block 6B-infra-2a).
    # Routed to `analytics`, which `task_routes` consumes via worker_medium.
    # Slot (#1645): head of the ETL -> corpus chain, and deliberately AFTER the
    # Monday 03:00 host reseed (scripts/reseed_synthetic.sh, ~60s) so Monday's
    # freshly appended frontier is rolled up the same morning.
    "business-metrics-per-hcp-rollup": {
        "task": "src.etl.business_metrics_per_hcp_etl.run_per_hcp_rollup",
        "schedule": crontab(hour=3, minute=15),
        "options": {"queue": "analytics"},
    },
    # Per-patient adherence/refill/gap derivation, daily at 03:30 UTC (block
    # 6B-infra-2b). Routed to `analytics` (worker_medium). Updates
    # patient_journeys.adherence_rate and gap_days; refill_count is left
    # NULL until a refill source lands -- see module docstring.
    # Slot (#1645): no ordering dependency on the per-HCP rollup, but it shares
    # the analytics queue, so it is offset 15 min rather than stacked.
    "patient-adherence-rollup": {
        "task": "src.etl.patient_adherence_etl.run_patient_adherence_rollup",
        "schedule": crontab(hour=3, minute=30),
        "options": {"queue": "analytics"},
    },
    # Territory_metrics rollup, daily at 03:45 UTC (block 6B-infra-2c). Routed
    # to `analytics` (worker_medium). Aggregates per-HCP business_metrics
    # rows produced by 6B-infra-2a -- in production the per-HCP rollup must
    # run first; see module docstring for the order dependency note.
    # market_potential / resource_allocation_score remain NULL until a real
    # Reltio/Veeva source lands.
    # Slot (#1645): the "per-HCP must run first" note is only ENFORCEABLE with a
    # wall-clock schedule. Under the previous 86400.0 intervals both entries
    # became due on the same beat tick and their order was whatever the queue
    # happened to do; 30 min after the per-HCP slot makes the dependency real.
    "territory-metrics-rollup": {
        "task": "src.etl.territory_metrics_etl.run_territory_rollup",
        "schedule": crontab(hour=3, minute=45),
        "options": {"queue": "analytics"},
    },
    # Operational KPI corpus sync, daily at 04:00 UTC (audit F3b). Re-indexes the
    # latest snapshot of every (brand, metric_name, region) combo from
    # business_metrics into the RAG substrate (episodic_memories). Scheduled
    # AFTER the business_metrics ETL rollups above so it picks up fresh facts;
    # idempotent prose dedup means only new/changed snapshots are embedded.
    # Slot (#1645/#1649): this entry is the reason the issue was filed -- it was
    # correctly implemented with a consumed queue and had still never run, because
    # a reset 86400.0 interval never comes due. "AFTER the rollups" was previously
    # only dict ordering; 04:00 makes it a real 45-min lag behind the 03:15 head
    # of the chain.
    "sync-operational-corpus": {
        "task": "src.tasks.sync_operational_corpus",
        "schedule": crontab(hour=4, minute=0),
        "options": {"queue": "analytics"},
    },
    # Chat-RAG chunk corpus sync, daily at 04:15 UTC (#1373). Re-indexes the latest
    # snapshot of every (brand, metric_name, region) combo from business_metrics
    # into rag_document_chunks -- the OTHER RAG substrate, embedded in the
    # text-embedding-3-small space the chat HybridRetriever queries in (the
    # episodic corpus above is in the memory-path ada-002 space, invisible to
    # chat). Scheduled AFTER the business_metrics ETL rollups; idempotent
    # content-hash dedup means only new/changed snapshots are embedded.
    # Slot (#1645): 04:15 UTC -- same post-rollup position as the episodic corpus
    # sync above, offset 15 min so the two embedding passes do not contend, and
    # clear of the 04:30 routing labeler.
    "sync-chunk-corpus": {
        "task": "src.tasks.sync_chunk_corpus",
        "schedule": crontab(hour=4, minute=15),
        "options": {"queue": "analytics"},
    },
    # -------------------------------------------------------------------------
    # A/B Testing Tasks (Phase 15)
    # -------------------------------------------------------------------------
    # Daily interim analysis check at 01:15 UTC. Slot (#1645): the comment here
    # always said "2 AM" but the schedule was a bare 86400.0 interval, so it in
    # fact fired at deploy-time + 24h -- i.e. never. Sequential-testing alpha
    # spending assumes a regular cadence, so a fixed wall clock matters more here
    # than for most entries. 01:15 keeps the intended quiet-hours placement while
    # staying clear of the 02:00 host backup.
    "ab-interim-analysis-check": {
        "task": "src.tasks.check_all_active_experiments",
        "schedule": crontab(hour=1, minute=15),
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
    # Medium-window feedback loop daily at 02:10 UTC (churn). Slot (#1645):
    # config/outcome_truth_rules.yaml declares the intent as
    # `medium_window_cron: "0 2 * * *"` (2 AM daily) -- that key has no code
    # consumer, so this beat entry is the only thing that can honour it, and as a
    # bare 86400.0 interval it never did. Offset 10 min past the 02:00 host backup
    # (measured 02:00:01 -> 02:01:26) so the churn scan does not read under pg_dump.
    "feedback-loop-medium-window": {
        "task": "src.tasks.run_feedback_loop_medium_window",
        "schedule": crontab(hour=2, minute=10),
        "options": {"queue": "analytics"},
    },
    # Long-window feedback loop weekly on Sundays (market_share_impact, risk)
    "feedback-loop-long-window": {
        "task": "src.tasks.run_feedback_loop_long_window",
        "schedule": 604800.0,  # 7 days
        "options": {"queue": "analytics"},
    },
    # Concept drift analysis after the feedback loop, daily at 02:40 UTC.
    # Slot (#1645): the "after feedback loop" ordering is the point of this
    # entry, so it sits 30 min behind the medium-window slot above. The original
    # comment said 3 AM; that hour now belongs to the Monday host reseed
    # (scripts/reseed_synthetic.sh), so the slot moved earlier rather than later.
    "feedback-loop-drift-analysis": {
        "task": "src.tasks.analyze_concept_drift_from_truth",
        "schedule": crontab(hour=2, minute=40),
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
    # Slot (#1645): 06:00 UTC. The upstream freshness requirement is satisfied by
    # the 6h feedback-learning-cycle regardless of hour (see its comment above),
    # so the binding constraint here is cost/contention, not ordering: this is a
    # GEPA run, and 06:00 puts it clear of the 05:30 chatbot-optimization drain
    # (also GEPA) rather than racing it for the analytics queue.
    "dspy-prompt-optimization-daily": {
        "task": "src.tasks.run_dspy_prompt_optimization",
        "schedule": crontab(hour=6, minute=0),
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
    # Knowledge-graph emptiness sentinel (#1761, #1758 follow-up)
    # -------------------------------------------------------------------------
    # Probes the CURATED core of the FalkorDB graph
    # (MATCH (n) WHERE n.agent IS NULL -- the same predicate /knowledge-graph
    # sends as curated_only) and, when it is empty, reseeds it under a Redis
    # lock from scripts/seed_falkordb.py + scripts/sync_causal_paths_to_falkordb.py.
    #
    # Interval, not crontab, and deliberately so: #1645 moved DAILY entries to a
    # wall clock because an interval longer than container uptime never comes
    # due. A 30-minute interval has the opposite property -- it is due within
    # half an hour of every boot, which is exactly what a recovery sentinel
    # wants. The 30 min is the outage window this bounds: #1758 ran four days.
    #
    # Queue `quick` is consumed by worker_light (--queues=default,quick,api),
    # which is also where the memory budget for the reseed subprocess lives --
    # see #1761's src.rag import shed in scripts/seed_falkordb.py.
    "graph-emptiness-sentinel": {
        "task": "src.tasks.graph_emptiness_sentinel",
        "schedule": 1800.0,  # 30 minutes
        "options": {"queue": "quick"},
    },
    # -------------------------------------------------------------------------
    # Insight Lifecycle subsystem (consolidation + sentinels)
    # -------------------------------------------------------------------------
    # Daily consolidator at 06:30 UTC: promotes confirmed causal_paths to
    # semantic tier and high-success procedural_memories to procedural tier.
    # Slot (#1645): no hard upstream — it reads accumulated memory tiers, not a
    # same-morning ETL product — so it takes the tail of the daily band, after
    # the 06:00 DSPy optimization.
    "insight-lifecycle-consolidate": {
        "task": "src.tasks.consolidate_insights",
        "schedule": crontab(hour=6, minute=30),
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
