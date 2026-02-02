"""Feedback Loop Celery Tasks.

Scheduled tasks for feedback loop execution and concept drift analysis.
These tasks assign ground truth to predictions and detect concept drift.

Tasks:
- run_feedback_loop_short_window: Process trigger, next_best_action (4h)
- run_feedback_loop_medium_window: Process hcp_churn (daily 2AM)
- run_feedback_loop_long_window: Process market_share_impact, risk (weekly Sun 3AM)
- analyze_concept_drift_from_truth: Post-labeling drift analysis

Configuration:
- config/outcome_truth_rules.yaml for prediction type definitions
- database/migrations/006_feedback_loop_infrastructure.sql for PL/pgSQL functions

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Configuration path
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "outcome_truth_rules.yaml"

# Default configuration
DEFAULT_CONFIG = {
    "feedback_loop": {
        "schedule": {
            "short_window_types": ["trigger", "next_best_action"],
            "medium_window_types": ["churn"],
            "long_window_types": ["market_share_impact", "risk"],
        },
        "processing": {
            "batch_size": 1000,
            "max_retries": 3,
            "retry_delay_minutes": 15,
            "min_confidence_threshold": 0.60,
        },
        "alerts": {
            "accuracy_degradation_threshold": 0.10,
            "indeterminate_rate_threshold": 0.20,
        },
    },
    "drift_integration": {
        "concept_drift": {
            "enabled": True,
            "comparison_windows": {
                "baseline_days": 90,
                "current_days": 30,
            },
            "alert_thresholds": {
                "accuracy_drop": 0.05,
                "calibration_error": 0.10,
                "class_shift": 0.15,
            },
        },
    },
}


def load_config() -> Dict[str, Any]:
    """Load feedback loop configuration."""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f) or {}
                return {**DEFAULT_CONFIG, **config}
    except Exception as e:
        logger.warning(f"Failed to load feedback loop config, using defaults: {e}")
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


async def _execute_feedback_loop(
    prediction_types: List[str],
    task_id: str,
    window_name: str,
) -> Dict[str, Any]:
    """Execute feedback loop for specified prediction types.

    Calls the PL/pgSQL run_feedback_loop() function via Supabase RPC.

    Args:
        prediction_types: List of prediction types to process
        task_id: Celery task ID for logging
        window_name: Window identifier (short/medium/long)

    Returns:
        Aggregated results from all prediction types
    """
    from src.memory.services.factories import get_supabase_client

    config = load_config()
    processing_config = config.get("feedback_loop", {}).get("processing", {})
    batch_size = processing_config.get("batch_size", 1000)

    start_time = time.time()
    results = []
    errors = []
    total_labeled = 0
    total_skipped = 0

    try:
        client = await get_supabase_client()
        if not client:
            return {
                "status": "skipped",
                "reason": "No database client available",
                "task_id": task_id,
            }

        for prediction_type in prediction_types:
            try:
                logger.info(
                    f"Running feedback loop for {prediction_type}: "
                    f"task {task_id}, window {window_name}"
                )

                # Call the PL/pgSQL function via RPC
                rpc_result = await client.rpc(
                    "run_feedback_loop",
                    {
                        "p_prediction_type": prediction_type,
                        "p_batch_size": batch_size,
                    },
                ).execute()

                result_data = rpc_result.data if rpc_result.data else {}

                # Handle different response formats
                if isinstance(result_data, list) and len(result_data) > 0:
                    result_data = result_data[0]
                elif isinstance(result_data, dict):
                    pass
                else:
                    result_data = {
                        "predictions_labeled": 0,
                        "predictions_skipped": 0,
                    }

                labeled_count = result_data.get("predictions_labeled", 0)
                skipped_count = result_data.get("predictions_skipped", 0)

                total_labeled += labeled_count
                total_skipped += skipped_count

                results.append(
                    {
                        "prediction_type": prediction_type,
                        "predictions_labeled": labeled_count,
                        "predictions_skipped": skipped_count,
                        "status": "completed",
                        "run_id": result_data.get("run_id"),
                    }
                )

                logger.info(
                    f"Feedback loop completed for {prediction_type}: "
                    f"labeled={labeled_count}, skipped={skipped_count}"
                )

            except Exception as e:
                logger.error(f"Feedback loop failed for {prediction_type}: {e}")
                errors.append(
                    {
                        "prediction_type": prediction_type,
                        "error": str(e),
                    }
                )

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "completed" if not errors else "partial",
            "task_id": task_id,
            "window": window_name,
            "prediction_types": prediction_types,
            "total_labeled": total_labeled,
            "total_skipped": total_skipped,
            "duration_ms": duration_ms,
            "results": results,
            "errors": errors,
        }

    except Exception as e:
        logger.error(f"Feedback loop execution failed: {e}")
        return {
            "status": "failed",
            "task_id": task_id,
            "window": window_name,
            "error": str(e),
        }


@celery_app.task(bind=True, name="src.tasks.run_feedback_loop_short_window")
def run_feedback_loop_short_window(
    self,
    prediction_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run feedback loop for short observation window predictions.

    Processes: trigger, next_best_action
    Schedule: Every 4 hours

    These prediction types have 14-30 day observation windows and
    can be labeled more frequently.

    Args:
        prediction_types: Override default types (optional)

    Returns:
        Feedback loop execution results
    """
    logger.info(f"Starting short-window feedback loop: task {self.request.id}")

    config = load_config()
    schedule_config = config.get("feedback_loop", {}).get("schedule", {})

    types_to_process = prediction_types or schedule_config.get(
        "short_window_types", ["trigger", "next_best_action"]
    )

    return run_async(
        _execute_feedback_loop(
            prediction_types=types_to_process,
            task_id=self.request.id,
            window_name="short",
        )
    )


@celery_app.task(bind=True, name="src.tasks.run_feedback_loop_medium_window")
def run_feedback_loop_medium_window(
    self,
    prediction_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run feedback loop for medium observation window predictions.

    Processes: churn
    Schedule: Daily at 2 AM

    HCP churn predictions require 60-90 day observation windows.

    Args:
        prediction_types: Override default types (optional)

    Returns:
        Feedback loop execution results
    """
    logger.info(f"Starting medium-window feedback loop: task {self.request.id}")

    config = load_config()
    schedule_config = config.get("feedback_loop", {}).get("schedule", {})

    types_to_process = prediction_types or schedule_config.get("medium_window_types", ["churn"])

    return run_async(
        _execute_feedback_loop(
            prediction_types=types_to_process,
            task_id=self.request.id,
            window_name="medium",
        )
    )


@celery_app.task(bind=True, name="src.tasks.run_feedback_loop_long_window")
def run_feedback_loop_long_window(
    self,
    prediction_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run feedback loop for long observation window predictions.

    Processes: market_share_impact, risk (treatment response)
    Schedule: Weekly on Sundays at 3 AM

    These predictions require 90-180 day observation windows.

    Args:
        prediction_types: Override default types (optional)

    Returns:
        Feedback loop execution results
    """
    logger.info(f"Starting long-window feedback loop: task {self.request.id}")

    config = load_config()
    schedule_config = config.get("feedback_loop", {}).get("schedule", {})

    types_to_process = prediction_types or schedule_config.get(
        "long_window_types", ["market_share_impact", "risk"]
    )

    return run_async(
        _execute_feedback_loop(
            prediction_types=types_to_process,
            task_id=self.request.id,
            window_name="long",
        )
    )


@celery_app.task(bind=True, name="src.tasks.analyze_concept_drift_from_truth")
def analyze_concept_drift_from_truth(
    self,
    prediction_type: Optional[str] = None,
    baseline_days: Optional[int] = None,
    current_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze concept drift using ground truth labels.

    Queries v_drift_alerts view after truth assignment to detect:
    - Accuracy degradation (predicted vs actual outcomes)
    - Calibration drift (predicted probabilities vs actual rates)
    - Class ratio shift (change in positive/negative class distribution)

    Args:
        prediction_type: Specific type to analyze (None = all types)
        baseline_days: Baseline window in days (default: 90)
        current_days: Current window in days (default: 30)

    Returns:
        Concept drift analysis results with alert triggers
    """
    logger.info(f"Starting concept drift analysis: task {self.request.id}")

    config = load_config()
    drift_config = config.get("drift_integration", {}).get("concept_drift", {})
    comparison_windows = drift_config.get("comparison_windows", {})
    alert_thresholds = drift_config.get("alert_thresholds", {})

    effective_baseline = baseline_days or comparison_windows.get("baseline_days", 90)
    effective_current = current_days or comparison_windows.get("current_days", 30)

    async def execute_analysis():
        from src.memory.services.factories import get_supabase_client

        start_time = time.time()
        alerts_triggered = []
        drift_results = []

        try:
            client = await get_supabase_client()
            if not client:
                return {
                    "status": "skipped",
                    "reason": "No database client available",
                }

            # Query drift alerts view
            query = client.table("v_drift_alerts").select("*")

            if prediction_type:
                query = query.eq("prediction_type", prediction_type)

            result = await query.execute()
            drift_alerts = result.data if result.data else []

            # Process each drift alert
            for alert in drift_alerts:
                drift_type = alert.get("prediction_type")
                accuracy_status = alert.get("accuracy_status")
                calibration_status = alert.get("calibration_status")
                accuracy_drop = alert.get("accuracy_drop", 0)
                calibration_error = alert.get("calibration_error", 0)

                drift_result = {
                    "prediction_type": drift_type,
                    "accuracy_drop": accuracy_drop,
                    "calibration_error": calibration_error,
                    "accuracy_status": accuracy_status,
                    "calibration_status": calibration_status,
                    "baseline_accuracy": alert.get("baseline_accuracy"),
                    "current_accuracy": alert.get("current_accuracy"),
                    "predictions_count": alert.get("predictions_count"),
                }
                drift_results.append(drift_result)

                # Check alert conditions
                threshold_accuracy = alert_thresholds.get("accuracy_drop", 0.05)
                threshold_calibration = alert_thresholds.get("calibration_error", 0.10)

                if accuracy_status == "ALERT" or accuracy_drop >= threshold_accuracy:
                    alerts_triggered.append(
                        {
                            "type": "accuracy_degradation",
                            "prediction_type": drift_type,
                            "severity": "critical" if accuracy_drop >= 0.10 else "high",
                            "metric": "accuracy_drop",
                            "value": accuracy_drop,
                            "threshold": threshold_accuracy,
                            "message": (
                                f"Accuracy dropped by {accuracy_drop:.1%} for {drift_type} "
                                f"predictions (threshold: {threshold_accuracy:.1%})"
                            ),
                        }
                    )

                if calibration_status == "ALERT" or calibration_error >= threshold_calibration:
                    alerts_triggered.append(
                        {
                            "type": "calibration_drift",
                            "prediction_type": drift_type,
                            "severity": "high" if calibration_error >= 0.15 else "medium",
                            "metric": "calibration_error",
                            "value": calibration_error,
                            "threshold": threshold_calibration,
                            "message": (
                                f"Calibration error of {calibration_error:.1%} for {drift_type} "
                                f"predictions (threshold: {threshold_calibration:.1%})"
                            ),
                        }
                    )

            # Route alerts if any were triggered
            if alerts_triggered:
                try:
                    from src.services.alert_routing import route_concept_drift_alerts

                    await route_concept_drift_alerts(
                        drift_results=drift_results,
                        alerts=alerts_triggered,
                        baseline_days=effective_baseline,
                        current_days=effective_current,
                    )
                except ImportError:
                    # route_concept_drift_alerts not yet implemented (Phase 4)
                    logger.warning(
                        "route_concept_drift_alerts not available, "
                        f"skipping alert routing for {len(alerts_triggered)} alerts"
                    )
                except Exception as e:
                    logger.error(f"Failed to route concept drift alerts: {e}")

            # Query concept drift metrics for summary
            metrics_result = await client.table("v_concept_drift_metrics").select("*").execute()
            metrics = metrics_result.data if metrics_result.data else []

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "status": "completed",
                "task_id": self.request.id,
                "prediction_type": prediction_type or "all",
                "baseline_days": effective_baseline,
                "current_days": effective_current,
                "drift_results": drift_results,
                "alerts_triggered": len(alerts_triggered),
                "alerts": alerts_triggered,
                "concept_drift_metrics": metrics,
                "duration_ms": duration_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Concept drift analysis failed: {e}")
            return {
                "status": "failed",
                "task_id": self.request.id,
                "error": str(e),
            }

    return run_async(execute_analysis())


@celery_app.task(bind=True, name="src.tasks.run_full_feedback_loop")
def run_full_feedback_loop(
    self,
    include_drift_analysis: bool = True,
) -> Dict[str, Any]:
    """Run complete feedback loop for all prediction types.

    Convenience task that runs all feedback loop windows and optionally
    triggers drift analysis afterward.

    Args:
        include_drift_analysis: Run drift analysis after labeling

    Returns:
        Combined results from all feedback loops
    """
    logger.info(f"Starting full feedback loop: task {self.request.id}")

    config = load_config()
    schedule_config = config.get("feedback_loop", {}).get("schedule", {})

    all_types = (
        schedule_config.get("short_window_types", [])
        + schedule_config.get("medium_window_types", [])
        + schedule_config.get("long_window_types", [])
    )

    async def execute_full_loop():
        start_time = time.time()

        # Run feedback loop for all types
        loop_result = await _execute_feedback_loop(
            prediction_types=all_types,
            task_id=self.request.id,
            window_name="full",
        )

        drift_result = None
        if include_drift_analysis and loop_result.get("status") != "failed":
            # Trigger drift analysis
            drift_task = analyze_concept_drift_from_truth.delay()
            drift_result = {
                "drift_analysis_task_id": drift_task.id,
                "status": "queued",
            }

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            **loop_result,
            "drift_analysis": drift_result,
            "total_duration_ms": duration_ms,
        }

    return run_async(execute_full_loop())


# Celery Beat schedule configuration
@celery_app.on_after_finalize.connect
def setup_feedback_loop_periodic_tasks(sender, **kwargs):
    """Set up periodic feedback loop tasks.

    Note: This is in addition to the static beat_schedule in celery_app.py.
    Static schedule is preferred for production stability.
    """
    config = load_config()

    if not config.get("feedback_loop", {}).get("schedule"):
        logger.info("Feedback loop schedule not configured, skipping periodic setup")
        return

    logger.info("Feedback loop periodic tasks configured via beat_schedule")
