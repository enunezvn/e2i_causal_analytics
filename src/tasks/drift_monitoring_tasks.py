"""Drift Monitoring Celery Tasks.

Scheduled tasks for drift detection and model monitoring.
These tasks run the drift monitor agent on production models.

Tasks:
- run_drift_detection: Full drift detection for a model
- check_all_models: Check drift for all production models
- cleanup_old_results: Archive old drift history records

Scheduling:
- Drift detection runs every 6 hours for production models
- Full model sweep runs daily at midnight
- Cleanup runs weekly on Sundays

Configuration:
- config/drift_monitoring.yaml for thresholds and schedules
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Configuration path
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "drift_monitoring.yaml"

# Default configuration
DEFAULT_CONFIG = {
    "detection": {
        "time_window": "7d",
        "significance_level": 0.05,
        "psi_threshold": 0.1,
        "features_to_monitor": [],
    },
    "schedule": {
        "detection_interval_hours": 6,
        "full_sweep_hour": 0,  # Midnight
        "cleanup_day": 6,  # Sunday
        "retention_days": 90,
    },
    "alerts": {
        "critical_threshold": 0.8,
        "warning_threshold": 0.4,
        "email_recipients": [],
        "slack_webhook": None,
    },
}


def load_config() -> Dict[str, Any]:
    """Load drift monitoring configuration."""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f) or {}
                # Merge with defaults
                return {**DEFAULT_CONFIG, **config}
    except Exception as e:
        logger.warning(f"Failed to load drift config, using defaults: {e}")
    return DEFAULT_CONFIG


def run_async(coro):
    """Helper to run async coroutine in sync context.

    Compatible with pytest-asyncio auto mode and pytest-xdist workers.
    """
    try:
        # Check if we're in an existing event loop (pytest-asyncio)
        loop = asyncio.get_running_loop()
        # We're in a running loop - use nest_asyncio
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop - get existing or create new event loop
        # Using get_event_loop() with fallback ensures thread-local loop reuse
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop at all - create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


@celery_app.task(bind=True, name="src.tasks.run_drift_detection")
def run_drift_detection(
    self,
    model_id: str,
    time_window: Optional[str] = None,
    features: Optional[List[str]] = None,
    check_data_drift: bool = True,
    check_model_drift: bool = True,
    check_concept_drift: bool = True,
    brand: Optional[str] = None,
) -> Dict[str, Any]:
    """Run drift detection for a specific model.

    Args:
        model_id: Model version/ID to monitor
        time_window: Time window for comparison (e.g., "7d", "30d")
        features: Specific features to check (None = all available)
        check_data_drift: Enable data drift detection
        check_model_drift: Enable model drift detection
        check_concept_drift: Enable concept drift detection
        brand: Optional brand filter

    Returns:
        Drift detection results with alerts
    """
    from src.agents.drift_monitor.connectors import get_connector
    from src.agents.drift_monitor.nodes.alert_aggregator import AlertAggregatorNode
    from src.agents.drift_monitor.nodes.concept_drift import ConceptDriftNode
    from src.agents.drift_monitor.nodes.data_drift import DataDriftNode
    from src.agents.drift_monitor.nodes.model_drift import ModelDriftNode
    from src.agents.drift_monitor.state import DriftMonitorState
    from src.repositories.drift_monitoring import (
        DriftHistoryRepository,
        MonitoringAlertRepository,
        MonitoringRunRepository,
        get_drift_monitoring_client,
    )

    logger.info(f"Starting drift detection for model {model_id}: task {self.request.id}")

    config = load_config()
    start_time = time.time()

    # Build initial state
    detection_config = config.get("detection", {})
    state: DriftMonitorState = {
        "model_id": model_id,
        "time_window": time_window or detection_config.get("time_window", "7d"),
        "features_to_monitor": features or detection_config.get("features_to_monitor", []),
        "significance_level": detection_config.get("significance_level", 0.05),
        "psi_threshold": detection_config.get("psi_threshold", 0.1),
        "check_data_drift": check_data_drift,
        "check_model_drift": check_model_drift,
        "check_concept_drift": check_concept_drift,
        "brand": brand or "",
        "status": "pending",
        "errors": [],
        "warnings": [],
        "data_drift_results": [],
        "model_drift_results": [],
        "concept_drift_results": [],
        "overall_drift_score": 0.0,
        "features_with_drift": [],
        "alerts": [],
        "drift_summary": "",
        "recommended_actions": [],
        "total_latency_ms": 0,
        "timestamp": "",
        "features_checked": 0,
    }

    async def execute_detection():
        # Resolve the Supabase client up front. Fail-closed (#845): if Supabase
        # is unconfigured this raises before any monitoring work, so the task
        # fails loudly instead of running a "successful" sweep that silently
        # persisted nothing.
        client = await get_drift_monitoring_client()

        # Initialize connector
        connector = get_connector()

        # Start monitoring run
        run_repo = MonitoringRunRepository(client)
        run_record = await run_repo.start_run(
            model_version=model_id,
            run_type="scheduled",
            config={
                "time_window": state["time_window"],
                "features": state["features_to_monitor"],
                "checks": {
                    "data_drift": check_data_drift,
                    "model_drift": check_model_drift,
                    "concept_drift": check_concept_drift,
                },
            },
        )

        try:
            # Get available features if not specified
            if not state["features_to_monitor"]:
                available_features = await connector.get_available_features()
                state["features_to_monitor"] = available_features[:50]  # Limit for performance

            # Run drift detection nodes
            if check_data_drift:
                data_drift_node = DataDriftNode(connector=connector)
                state.update(await data_drift_node.execute(state))

            if check_model_drift:
                model_drift_node = ModelDriftNode(connector=connector)
                state.update(await model_drift_node.execute(state))

            if check_concept_drift:
                concept_drift_node = ConceptDriftNode(connector=connector)
                state.update(await concept_drift_node.execute(state))

            # Aggregate alerts
            alert_node = AlertAggregatorNode()
            state.update(await alert_node.execute(state))

            # Persist results
            drift_repo = DriftHistoryRepository(client)
            alert_repo = MonitoringAlertRepository(client)

            # Collect all results
            all_results = (
                state.get("data_drift_results", [])
                + state.get("model_drift_results", [])
                + state.get("concept_drift_results", [])
            )

            # Parse time windows for persistence
            days = int(state["time_window"].replace("d", ""))
            now = datetime.now(timezone.utc)
            baseline_window = {
                "start": now - timedelta(days=days * 2),
                "end": now - timedelta(days=days),
            }
            current_window = {
                "start": now - timedelta(days=days),
                "end": now,
            }

            # Record drift history
            if all_results:
                await drift_repo.record_drift_results(
                    model_version=model_id,
                    drift_results=all_results,
                    baseline_window=baseline_window,
                    current_window=current_window,
                )

            # Create alerts
            alerts = await alert_repo.create_alerts_from_drift(
                model_version=model_id,
                drift_results=all_results,
            )

            # Route alerts to notification channels
            if alerts:
                from src.services.alert_routing import route_drift_alerts

                await route_drift_alerts(
                    model_version=model_id,
                    drift_results=all_results,
                    overall_score=state.get("overall_drift_score", 0.0),
                    summary=state.get("drift_summary", ""),
                    recommended_actions=state.get("recommended_actions", []),
                )

            # Complete monitoring run
            duration_ms = int((time.time() - start_time) * 1000)
            await run_repo.complete_run(
                run_id=run_record.id,
                features_checked=len(state.get("features_to_monitor", [])),
                drift_detected_count=len(state.get("features_with_drift", [])),
                alerts_generated=len(alerts),
                duration_ms=duration_ms,
            )

            return {
                "run_id": run_record.id,
                "model_id": model_id,
                "status": state.get("status", "completed"),
                "overall_drift_score": state.get("overall_drift_score", 0.0),
                "features_checked": len(state.get("features_to_monitor", [])),
                "features_with_drift": state.get("features_with_drift", []),
                "alerts_generated": len(alerts),
                "drift_summary": state.get("drift_summary", ""),
                "recommended_actions": state.get("recommended_actions", []),
                "detection_latency_ms": duration_ms,
                "errors": state.get("errors", []),
                "warnings": state.get("warnings", []),
            }

        except Exception as e:
            # Record failed run
            duration_ms = int((time.time() - start_time) * 1000)
            await run_repo.complete_run(
                run_id=run_record.id,
                features_checked=0,
                drift_detected_count=0,
                alerts_generated=0,
                duration_ms=duration_ms,
                error_message=str(e),
            )
            raise

    return run_async(execute_detection())  # type: ignore[no-any-return]


@celery_app.task(bind=True, name="src.tasks.check_all_production_models")
def check_all_production_models(
    self,
    time_window: Optional[str] = None,
) -> Dict[str, Any]:
    """Check drift for all production models.

    Queries the model registry for production models and runs
    drift detection on each one.

    Args:
        time_window: Time window for comparison

    Returns:
        Summary of all drift checks
    """
    from src.agents.drift_monitor.connectors import get_connector

    logger.info(f"Starting production model sweep: task {self.request.id}")

    config = load_config()
    detection_config = config.get("detection", {})
    effective_window = time_window or detection_config.get("time_window", "7d")

    async def run_sweep():
        connector = get_connector()

        # Get all production models
        models = await connector.get_available_models(stage="production")

        if not models:
            logger.warning("No production models found to monitor")
            return {
                "status": "completed",
                "models_checked": 0,
                "message": "No production models found",
            }

        results = []
        errors = []

        for model in models:
            model_id = model.get("id") or model.get("name")
            try:
                # Trigger individual drift detection
                result = run_drift_detection.delay(
                    model_id=model_id,
                    time_window=effective_window,
                )
                results.append(
                    {
                        "model_id": model_id,
                        "task_id": result.id,
                        "status": "queued",
                    }
                )
            except Exception as e:
                errors.append(
                    {
                        "model_id": model_id,
                        "error": str(e),
                    }
                )

        return {
            "status": "completed",
            "models_checked": len(models),
            "tasks_queued": len(results),
            "errors": errors,
            "results": results,
        }

    return run_async(run_sweep())  # type: ignore[no-any-return]


@celery_app.task(bind=True, name="src.tasks.cleanup_old_drift_history")
def cleanup_old_drift_history(
    self,
    retention_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Clean up old drift history records.

    Archives or deletes drift history records older than retention period.

    Args:
        retention_days: Number of days to retain (default: 90)

    Returns:
        Cleanup summary
    """
    logger.info(f"Starting drift history cleanup: task {self.request.id}")

    config = load_config()
    schedule_config = config.get("schedule", {})
    effective_retention = retention_days or schedule_config.get("retention_days", 90)

    async def run_cleanup():
        from src.memory.services.factories import get_async_supabase_client

        try:
            client = await get_async_supabase_client()
            if not client:
                return {"status": "skipped", "reason": "No database client available"}

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=effective_retention)
            cutoff_iso = cutoff_date.isoformat()

            # Delete old drift history records. The real ml_drift_history
            # timestamp column is created_at (there is no detected_at; #825).
            drift_result = await (
                client.table("ml_drift_history").delete().lt("created_at", cutoff_iso).execute()
            )
            drift_deleted = len(drift_result.data) if drift_result.data else 0

            # Delete old resolved alerts (keep active ones). The real
            # ml_monitoring_alerts schema has no triggered_at column; created_at
            # is when the alert fired (#825). The status=resolved filter ensures
            # active alerts are retained regardless of age.
            alert_result = await (
                client.table("ml_monitoring_alerts")
                .delete()
                .lt("created_at", cutoff_iso)
                .eq("status", "resolved")
                .execute()
            )
            alerts_deleted = len(alert_result.data) if alert_result.data else 0

            # Delete old monitoring runs
            run_result = await (
                client.table("ml_monitoring_runs").delete().lt("started_at", cutoff_iso).execute()
            )
            runs_deleted = len(run_result.data) if run_result.data else 0

            return {
                "status": "completed",
                "retention_days": effective_retention,
                "cutoff_date": cutoff_iso,
                "drift_records_deleted": drift_deleted,
                "alerts_deleted": alerts_deleted,
                "runs_deleted": runs_deleted,
            }

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    return run_async(run_cleanup())  # type: ignore[no-any-return]


@celery_app.task(bind=True, name="src.tasks.track_model_performance")
def track_model_performance(
    self,
    model_id: str,
    predictions: List[int],
    actuals: List[int],
    prediction_scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Track model performance metrics.

    Records model performance metrics and checks for alerts.

    Args:
        model_id: Model version/ID
        predictions: Predicted labels
        actuals: Actual labels
        prediction_scores: Optional probability scores

    Returns:
        Performance tracking results
    """
    logger.info(f"Tracking performance for model {model_id}: task {self.request.id}")

    async def execute_tracking():
        from src.services.performance_tracking import (
            get_performance_tracker,
            record_model_performance,
        )

        try:
            # Record performance metrics
            result = await record_model_performance(
                model_version=model_id,
                predictions=predictions,
                actuals=actuals,
                prediction_scores=prediction_scores,
            )

            # Check for performance alerts
            tracker = get_performance_tracker()
            alerts = await tracker.check_performance_alerts(model_id)

            # Route alerts if any
            if alerts:
                from src.services.alert_routing import AlertPayload, get_alert_router

                router = get_alert_router()
                for alert in alerts:
                    payload = AlertPayload(
                        alert_type="performance_degradation",
                        severity=alert.get("severity", "medium"),
                        model_version=model_id,
                        message=alert.get("message", "Performance degradation detected"),
                        details=alert,
                    )
                    await router.route_alert(payload)

            return {
                "status": "completed",
                "model_id": model_id,
                "metrics": result.get("metrics", {}),
                "sample_size": result.get("sample_size", 0),
                "alerts_generated": len(alerts),
                "alerts": alerts,
            }

        except Exception as e:
            logger.error(f"Performance tracking failed: {e}")
            return {
                "status": "failed",
                "model_id": model_id,
                "error": str(e),
            }

    return run_async(execute_tracking())  # type: ignore[no-any-return]


@celery_app.task(bind=True, name="src.tasks.check_model_performance_alerts")
def check_model_performance_alerts(
    self,
    model_id: str,
) -> Dict[str, Any]:
    """Check performance alerts for a model.

    Analyzes performance trends and generates alerts if needed.

    Args:
        model_id: Model version/ID

    Returns:
        Alert check results
    """
    logger.info(f"Checking performance alerts for model {model_id}")

    async def execute_check():
        from src.services.alert_routing import AlertPayload, get_alert_router
        from src.services.performance_tracking import get_performance_tracker

        try:
            tracker = get_performance_tracker()
            alerts = await tracker.check_performance_alerts(model_id)

            # Route alerts if any
            if alerts:
                router = get_alert_router()
                for alert in alerts:
                    payload = AlertPayload(
                        alert_type="performance_degradation",
                        severity=alert.get("severity", "medium"),
                        model_version=model_id,
                        message=alert.get("message", "Performance degradation detected"),
                        details=alert,
                    )
                    await router.route_alert(payload)

            return {
                "status": "completed",
                "model_id": model_id,
                "alerts_generated": len(alerts),
                "alerts": alerts,
            }

        except Exception as e:
            logger.error(f"Performance alert check failed: {e}")
            return {
                "status": "failed",
                "model_id": model_id,
                "error": str(e),
            }

    return run_async(execute_check())  # type: ignore[no-any-return]


@celery_app.task(bind=True, name="src.tasks.send_drift_alert_notifications")
def send_drift_alert_notifications(
    self,
    alert_ids: List[str],
) -> Dict[str, Any]:
    """Send notifications for drift alerts.

    Sends email and/or Slack notifications for active alerts.

    Args:
        alert_ids: List of alert UUIDs to notify about

    Returns:
        Notification status
    """
    logger.info(f"Sending drift alert notifications: {len(alert_ids)} alerts")

    config = load_config()
    alert_config = config.get("alerts", {})

    async def send_notifications():
        from src.repositories.drift_monitoring import (
            MonitoringAlertRepository,
            get_drift_monitoring_client,
        )

        alert_repo = MonitoringAlertRepository(await get_drift_monitoring_client())
        notifications_sent = []
        errors = []

        for alert_id in alert_ids:
            try:
                # Get alert details
                alert = await alert_repo.get_by_id(alert_id)
                if not alert:
                    continue

                # Send email if configured
                email_recipients = alert_config.get("email_recipients", [])
                if email_recipients:
                    # Placeholder for email sending
                    logger.info(f"Would send email for alert {alert_id} to {email_recipients}")
                    notifications_sent.append(
                        {
                            "alert_id": alert_id,
                            "channel": "email",
                            "status": "simulated",
                        }
                    )

                # Send Slack if configured
                slack_webhook = alert_config.get("slack_webhook")
                if slack_webhook:
                    # Placeholder for Slack webhook
                    logger.info(f"Would send Slack notification for alert {alert_id}")
                    notifications_sent.append(
                        {
                            "alert_id": alert_id,
                            "channel": "slack",
                            "status": "simulated",
                        }
                    )

            except Exception as e:
                errors.append(
                    {
                        "alert_id": alert_id,
                        "error": str(e),
                    }
                )

        return {
            "status": "completed",
            "notifications_sent": len(notifications_sent),
            "errors": errors,
            "details": notifications_sent,
        }

    return run_async(send_notifications())  # type: ignore[no-any-return]


@celery_app.task(bind=True, name="src.tasks.evaluate_retraining_need")
def evaluate_retraining_need(
    self,
    model_id: str,
    auto_approve: bool = False,
) -> Dict[str, Any]:
    """Evaluate and optionally trigger model retraining.

    Checks drift scores and performance metrics to determine if
    retraining is needed, then optionally triggers it.

    Args:
        model_id: Model version/ID to evaluate
        auto_approve: Skip approval requirement if True

    Returns:
        Evaluation results and trigger status
    """
    logger.info(f"Evaluating retraining need for model {model_id}: task {self.request.id}")

    async def execute_evaluation():
        from src.services.retraining_trigger import evaluate_and_trigger_retraining

        try:
            result = await evaluate_and_trigger_retraining(
                model_version=model_id,
                auto_approve=auto_approve,
            )
            return result

        except Exception as e:
            logger.error(f"Retraining evaluation failed: {e}")
            return {
                "status": "failed",
                "model_version": model_id,
                "error": str(e),
            }

    return run_async(execute_evaluation())  # type: ignore[no-any-return]


def _cohort_input_from_training_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ``MLFoundationPipeline.run`` input_data from a retraining
    contract.

    A live retrain must name *which* committed cohort batch/table to retrain on
    (``data_source``) and *what* to predict (``target_outcome``). Both are
    required — fail loud rather than let a retrain silently run against the
    wrong (or no) cohort. ``feature_manifest_source`` (resolved upstream from
    the cohort identity) opts the run into Layer-5 manifest contracts.
    """
    data_source = training_config.get("data_source")
    if not data_source:
        raise ValueError(
            "training_config is missing 'data_source' — a live retrain must name "
            "the committed cohort batch/table to retrain on (no on-demand cohort "
            "assembly exists). Pass it via the trigger's cohort contract."
        )
    target_outcome = training_config.get("target_outcome")
    if not target_outcome:
        raise ValueError(
            "training_config is missing 'target_outcome' — a live retrain must "
            "name the prediction target column."
        )

    input_data: Dict[str, Any] = {
        "problem_description": (
            training_config.get("problem_description") or f"Retrain model on cohort {data_source}"
        ),
        "business_objective": (
            training_config.get("business_objective") or "Triggered model refresh (drift/manual)"
        ),
        "target_outcome": target_outcome,
        "data_source": data_source,
    }
    # Optional pass-through: brand context, the Layer-5 manifest opt-in, and the
    # deployment target. feature_manifest_source flows into scope_spec via the
    # pipeline's scope stage (Phase B), engaging Layer-1 contracts + #544.
    for opt in ("brand", "feature_manifest_source", "target_environment"):
        if training_config.get(opt) is not None:
            input_data[opt] = training_config[opt]
    return input_data


def _extract_validation_auc(result: Any) -> Optional[float]:
    """Return the real validation AUC the pipeline produced, or ``None``.

    Reads ``result.training_result['validation_metrics']`` (roc_auc / auc_roc /
    auc). ``None`` means the pipeline produced no certifiable metric — the
    caller MUST fail closed rather than synthesize a placeholder.
    """
    training_result = getattr(result, "training_result", None) or {}
    metrics = training_result.get("validation_metrics") or {}
    for key in ("roc_auc", "auc_roc", "auc", "val_auc"):
        val = metrics.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    return None


async def _execute_real_retraining(
    retraining_id: str,
    model_version: str,
    new_version: str,
    training_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the real ``MLFoundationPipeline`` for a retraining job and persist the
    REAL outcome.

    Fails closed: any path that does not yield a certifiable validation metric
    (missing cohort identity, pipeline exception, QC-gate halt, or a completed
    run with no metric) marks the job ``failed`` and writes NO performance value.
    This replaces the Phase-14 ``performance_after = 0.85  # Simulated`` stub —
    a fake metric must never reach ``ml_retraining_history`` (or the monitoring
    API that surfaces it). The deploy/gate decision (incl. the Layer-4 leakage /
    AUC gates) is enforced inside the pipeline's model_deployer stage.
    """
    from src.repositories.drift_monitoring import (
        RetrainingHistoryRepository,
        get_drift_monitoring_client,
    )
    from src.services.retraining_trigger import get_retraining_trigger_service

    repo = RetrainingHistoryRepository(await get_drift_monitoring_client())
    service = get_retraining_trigger_service()

    async def _mark_failed(reason: str) -> Dict[str, Any]:
        logger.error(f"Retraining {retraining_id} failed closed: {reason}")
        try:
            # ml_retraining_history has no error_message column; the failure
            # reason is recorded in `notes` via mark_failed (#842).
            await repo.mark_failed(retraining_id, reason)
        except Exception as e:  # noqa: BLE001 — never mask the original failure
            logger.error(f"Also failed to mark retraining {retraining_id} failed: {e}")
        return {
            "status": "failed",
            "retraining_id": retraining_id,
            "old_version": model_version,
            "new_version": new_version,
            "error": reason,
        }

    # Resolve the cohort to retrain on BEFORE touching state — fail loud if the
    # contract lacks the identity (the pipeline is never invoked in that case).
    try:
        pipeline_input = _cohort_input_from_training_config(training_config)
    except ValueError as e:
        return await _mark_failed(str(e))

    try:
        await repo.update(retraining_id, {"status": "training"})
        logger.info(
            f"Retraining {retraining_id}: running MLFoundationPipeline on "
            f"data_source={pipeline_input['data_source']!r} "
            f"(manifest={pipeline_input.get('feature_manifest_source')!r})"
        )
        from src.agents.tier_0.pipeline import MLFoundationPipeline

        pipeline = MLFoundationPipeline()
        result = await pipeline.run(pipeline_input)
    except Exception as e:  # noqa: BLE001 — convert any failure into fail-closed
        return await _mark_failed(f"pipeline raised {type(e).__name__}: {e}")

    status = getattr(result, "status", "failed")
    performance_after = _extract_validation_auc(result)
    # The pipeline reports status="completed" even when it SKIPS deployment
    # because success criteria were not met (pipeline.py:553-560). A
    # trained-but-not-promotable run is NOT a successful retrain — require the
    # success-criteria gate too, else fail closed.
    training_result = getattr(result, "training_result", None) or {}
    success_criteria_met = bool(training_result.get("success_criteria_met"))
    if status != "completed" or performance_after is None or not success_criteria_met:
        return await _mark_failed(
            f"pipeline status={status!r}, validation_auc={performance_after!r}, "
            f"success_criteria_met={success_criteria_met} — not a promotable result; "
            "job marked failed (no metric written)"
        )

    # Real metric + success criteria met — record completion. The pipeline also
    # gated deployment on the regulatory AUC/leakage gate.
    await service.complete_retraining(
        job_id=retraining_id,
        performance_after=performance_after,
        success=True,
    )
    deployment = getattr(result, "deployment_result", None) or {}
    return {
        "status": "completed",
        "retraining_id": retraining_id,
        "old_version": model_version,
        "new_version": new_version,
        "performance_after": performance_after,
        "mlflow_model_version": deployment.get("model_version"),
        "deployed": bool(deployment.get("deployment_successful", deployment.get("model_version"))),
        "message": f"Model {new_version} retrained; validation AUC={performance_after:.4f}",
    }


@celery_app.task(bind=True, name="src.tasks.execute_model_retraining")
def execute_model_retraining(
    self,
    retraining_id: str,
    model_version: str,
    new_version: str,
    training_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a model retraining job by running the REAL MLFoundationPipeline.

    Loads the committed cohort named in ``training_config`` (data_source +
    target_outcome + optional feature_manifest_source), runs scope → data-prep
    (QC gate) → train → deploy (gated), and records the real validation metric.
    Fails closed — never writes a simulated metric. Routed to the ``ml`` queue
    (worker_heavy) since it runs a full training pipeline.

    Args:
        retraining_id: Retraining history record ID
        model_version: Original model version
        new_version: New model version to create
        training_config: Training contract incl. the cohort identity

    Returns:
        Retraining results (real metric on success; failure reason otherwise)
    """
    logger.info(
        f"Starting model retraining: {model_version} -> {new_version}, task {self.request.id}"
    )
    return run_async(  # type: ignore[no-any-return]
        _execute_real_retraining(retraining_id, model_version, new_version, training_config)
    )


@celery_app.task(bind=True, name="src.tasks.check_retraining_for_all_models")
def check_retraining_for_all_models(
    self,
    auto_approve: bool = False,
) -> Dict[str, Any]:
    """Check retraining needs for all production models.

    Evaluates each production model and triggers retraining if needed.

    Args:
        auto_approve: Skip approval requirement for all models

    Returns:
        Summary of evaluations and triggered retraining jobs
    """
    from src.agents.drift_monitor.connectors import get_connector

    logger.info(f"Checking retraining for all production models: task {self.request.id}")

    async def run_check():
        connector = get_connector()

        # Get all production models
        models = await connector.get_available_models(stage="production")

        if not models:
            logger.warning("No production models found")
            return {
                "status": "completed",
                "models_checked": 0,
                "message": "No production models found",
            }

        results = []
        errors = []

        for model in models:
            model_id = model.get("id") or model.get("name")
            try:
                # Queue evaluation task
                task = evaluate_retraining_need.delay(
                    model_id=model_id,
                    auto_approve=auto_approve,
                )
                results.append(
                    {
                        "model_id": model_id,
                        "task_id": task.id,
                        "status": "queued",
                    }
                )
            except Exception as e:
                errors.append(
                    {
                        "model_id": model_id,
                        "error": str(e),
                    }
                )

        return {
            "status": "completed",
            "models_checked": len(models),
            "tasks_queued": len(results),
            "errors": errors,
            "results": results,
        }

    return run_async(run_check())  # type: ignore[no-any-return]


# Celery Beat schedule configuration
@celery_app.on_after_finalize.connect
def setup_periodic_tasks(sender, **kwargs):
    """Set up periodic drift monitoring tasks."""
    config = load_config()
    schedule_config = config.get("schedule", {})

    # Drift detection every 6 hours
    detection_interval = schedule_config.get("detection_interval_hours", 6)
    sender.add_periodic_task(
        detection_interval * 3600,
        check_all_production_models.s(),
        name="drift-detection-sweep",
    )

    # Daily cleanup at 2 AM
    sender.add_periodic_task(
        86400,  # 24 hours
        cleanup_old_drift_history.s(),
        name="drift-history-cleanup",
    )
