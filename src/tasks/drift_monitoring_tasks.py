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
    """Helper to run async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


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
        "brand": brand,
        "status": "initializing",
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
        "detection_latency_ms": 0,
        "features_checked": 0,
    }

    async def execute_detection():
        # Initialize connector
        connector = get_connector()

        # Start monitoring run
        run_repo = MonitoringRunRepository()
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
            drift_repo = DriftHistoryRepository()
            alert_repo = MonitoringAlertRepository()

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

    return run_async(execute_detection())


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
                results.append({
                    "model_id": model_id,
                    "task_id": result.id,
                    "status": "queued",
                })
            except Exception as e:
                errors.append({
                    "model_id": model_id,
                    "error": str(e),
                })

        return {
            "status": "completed",
            "models_checked": len(models),
            "tasks_queued": len(results),
            "errors": errors,
            "results": results,
        }

    return run_async(run_sweep())


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
        from src.memory.services.factories import get_supabase_client

        try:
            client = await get_supabase_client()
            if not client:
                return {"status": "skipped", "reason": "No database client available"}

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=effective_retention)
            cutoff_iso = cutoff_date.isoformat()

            # Delete old drift history records
            drift_result = await (
                client.table("ml_drift_history")
                .delete()
                .lt("detected_at", cutoff_iso)
                .execute()
            )
            drift_deleted = len(drift_result.data) if drift_result.data else 0

            # Delete old resolved alerts (keep active ones)
            alert_result = await (
                client.table("ml_monitoring_alerts")
                .delete()
                .lt("triggered_at", cutoff_iso)
                .eq("status", "resolved")
                .execute()
            )
            alerts_deleted = len(alert_result.data) if alert_result.data else 0

            # Delete old monitoring runs
            run_result = await (
                client.table("ml_monitoring_runs")
                .delete()
                .lt("started_at", cutoff_iso)
                .execute()
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

    return run_async(run_cleanup())


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
                from src.services.alert_routing import get_alert_router, AlertPayload

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

    return run_async(execute_tracking())


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
        from src.services.performance_tracking import get_performance_tracker
        from src.services.alert_routing import get_alert_router, AlertPayload

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

    return run_async(execute_check())


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
        from src.repositories.drift_monitoring import MonitoringAlertRepository

        alert_repo = MonitoringAlertRepository()
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
                    notifications_sent.append({
                        "alert_id": alert_id,
                        "channel": "email",
                        "status": "simulated",
                    })

                # Send Slack if configured
                slack_webhook = alert_config.get("slack_webhook")
                if slack_webhook:
                    # Placeholder for Slack webhook
                    logger.info(f"Would send Slack notification for alert {alert_id}")
                    notifications_sent.append({
                        "alert_id": alert_id,
                        "channel": "slack",
                        "status": "simulated",
                    })

            except Exception as e:
                errors.append({
                    "alert_id": alert_id,
                    "error": str(e),
                })

        return {
            "status": "completed",
            "notifications_sent": len(notifications_sent),
            "errors": errors,
            "details": notifications_sent,
        }

    return run_async(send_notifications())


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

    return run_async(execute_evaluation())


@celery_app.task(bind=True, name="src.tasks.execute_model_retraining")
def execute_model_retraining(
    self,
    retraining_id: str,
    model_version: str,
    new_version: str,
    training_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute model retraining job.

    This is a placeholder that integrates with the Model Trainer agent.
    In production, this would:
    1. Prepare training data
    2. Run hyperparameter tuning
    3. Train the model
    4. Validate performance
    5. Register new model version

    Args:
        retraining_id: Retraining history record ID
        model_version: Original model version
        new_version: New model version to create
        training_config: Training configuration

    Returns:
        Retraining results
    """
    logger.info(
        f"Starting model retraining: {model_version} -> {new_version}, "
        f"task {self.request.id}"
    )

    async def execute_retraining():
        from src.repositories.drift_monitoring import RetrainingHistoryRepository
        from src.services.retraining_trigger import get_retraining_trigger_service

        repo = RetrainingHistoryRepository()
        service = get_retraining_trigger_service()

        try:
            # Update status to training
            await repo.update(retraining_id, {"status": "training"})

            # NOTE: This is a placeholder for actual model training.
            # In production, this would:
            # 1. Load training data using the Model Trainer agent
            # 2. Prepare features via Feature Analyzer
            # 3. Train model with hyperparameter tuning
            # 4. Evaluate on holdout set
            # 5. Register with MLflow

            logger.info(f"Training config: {training_config}")

            # Simulate training (placeholder)
            import asyncio
            await asyncio.sleep(1)  # Would be actual training time

            # For now, simulate a successful retraining with slight improvement
            performance_after = 0.85  # Simulated performance

            # Complete the retraining
            await service.complete_retraining(
                job_id=retraining_id,
                performance_after=performance_after,
                success=True,
            )

            return {
                "status": "completed",
                "retraining_id": retraining_id,
                "old_version": model_version,
                "new_version": new_version,
                "performance_after": performance_after,
                "message": f"Model {new_version} trained successfully",
            }

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

            # Mark as failed
            await repo.update(
                retraining_id,
                {
                    "status": "failed",
                    "error_message": str(e),
                },
            )

            return {
                "status": "failed",
                "retraining_id": retraining_id,
                "error": str(e),
            }

    return run_async(execute_retraining())


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
        triggered = []
        errors = []

        for model in models:
            model_id = model.get("id") or model.get("name")
            try:
                # Queue evaluation task
                task = evaluate_retraining_need.delay(
                    model_id=model_id,
                    auto_approve=auto_approve,
                )
                results.append({
                    "model_id": model_id,
                    "task_id": task.id,
                    "status": "queued",
                })
            except Exception as e:
                errors.append({
                    "model_id": model_id,
                    "error": str(e),
                })

        return {
            "status": "completed",
            "models_checked": len(models),
            "tasks_queued": len(results),
            "errors": errors,
            "results": results,
        }

    return run_async(run_check())


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
