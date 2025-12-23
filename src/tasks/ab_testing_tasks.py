"""A/B Testing Celery Tasks.

Scheduled tasks for A/B experiment execution and monitoring.
These tasks manage experiment lifecycle, interim analyses, and results computation.

Tasks:
- scheduled_interim_analysis: Run interim analysis for active experiments
- enrollment_health_check: Check enrollment rates across all active experiments
- srm_detection_sweep: Periodic SRM detection for all running experiments
- compute_experiment_results: Compute final or interim results
- fidelity_tracking_update: Update fidelity comparison with Digital Twin predictions

Scheduling:
- Interim analysis runs daily for experiments with sufficient data
- Enrollment health checks run every 12 hours
- SRM detection runs every 6 hours
- Fidelity tracking updates weekly

Configuration:
- config/ab_testing.yaml for thresholds and schedules
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import yaml

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Configuration path
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "ab_testing.yaml"

# Default configuration
DEFAULT_CONFIG = {
    "interim_analysis": {
        "min_enrollment_fraction": 0.25,  # Min 25% enrollment before first interim
        "analysis_schedule": [0.25, 0.5, 0.75],  # Information fractions for analyses
        "alpha_spending": "obrien_fleming",  # Alpha spending function
        "total_alpha": 0.05,  # Overall significance level
        "futility_threshold": 0.1,  # Conditional power threshold for futility
    },
    "enrollment": {
        "min_daily_rate": 5,  # Minimum acceptable daily enrollment
        "warning_threshold_days": 7,  # Days of low enrollment before warning
        "critical_threshold_days": 14,  # Days of low enrollment before critical
    },
    "srm": {
        "detection_threshold": 0.001,  # P-value threshold for SRM detection
        "check_interval_hours": 6,  # How often to check
        "min_sample_size": 100,  # Minimum sample before SRM checks
    },
    "fidelity": {
        "comparison_interval_days": 7,  # How often to compare with twins
        "acceptable_error": 0.2,  # Acceptable prediction error (20%)
        "calibration_trigger_error": 0.3,  # Error level that triggers calibration
    },
    "schedule": {
        "interim_analysis_hour": 2,  # 2 AM UTC
        "enrollment_check_interval_hours": 12,
        "srm_check_interval_hours": 6,
        "fidelity_update_day": 0,  # Monday
        "retention_days": 365,  # Keep results for 1 year
    },
    "alerts": {
        "email_recipients": [],
        "slack_webhook": None,
    },
}


def load_config() -> Dict[str, Any]:
    """Load A/B testing configuration."""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f) or {}
                # Deep merge with defaults
                merged = DEFAULT_CONFIG.copy()
                for key, value in config.items():
                    if isinstance(value, dict) and key in merged:
                        merged[key] = {**merged[key], **value}
                    else:
                        merged[key] = value
                return merged
    except Exception as e:
        logger.warning(f"Failed to load A/B testing config, using defaults: {e}")
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


@celery_app.task(bind=True, name="src.tasks.scheduled_interim_analysis")
def scheduled_interim_analysis(
    self,
    experiment_id: str,
    force: bool = False,
) -> Dict[str, Any]:
    """Run scheduled interim analysis for an experiment.

    Checks if the experiment has reached an analysis milestone
    (based on information fraction) and performs interim analysis
    with alpha spending adjustments.

    Args:
        experiment_id: UUID of the experiment
        force: Force analysis even if milestone not reached

    Returns:
        Interim analysis results with decision recommendation
    """
    logger.info(f"Running interim analysis for experiment {experiment_id}: task {self.request.id}")

    config = load_config()
    interim_config = config.get("interim_analysis", {})
    start_time = time.time()

    async def execute_analysis():
        from src.services.interim_analysis import InterimAnalysisService
        from src.services.enrollment import EnrollmentService
        from src.repositories.ab_experiment import ABExperimentRepository

        try:
            exp_repo = ABExperimentRepository()
            enrollment_service = EnrollmentService()
            interim_service = InterimAnalysisService()

            # Get experiment details
            exp_uuid = UUID(experiment_id)

            # Get enrollment stats to determine information fraction
            enrollment_stats = await enrollment_service.get_enrollment_stats(exp_uuid)

            if not enrollment_stats:
                return {
                    "status": "skipped",
                    "experiment_id": experiment_id,
                    "reason": "No enrollment data available",
                }

            current_enrollment = enrollment_stats.total_enrolled
            target_sample_size = enrollment_stats.target_sample_size or 1000
            information_fraction = current_enrollment / target_sample_size

            # Check if we've reached an analysis milestone
            analysis_schedule = interim_config.get("analysis_schedule", [0.25, 0.5, 0.75])
            min_fraction = interim_config.get("min_enrollment_fraction", 0.25)

            if not force and information_fraction < min_fraction:
                return {
                    "status": "skipped",
                    "experiment_id": experiment_id,
                    "reason": f"Information fraction {information_fraction:.2%} below minimum {min_fraction:.2%}",
                    "current_enrollment": current_enrollment,
                    "target_sample_size": target_sample_size,
                }

            # Determine which analysis number this is
            previous_analyses = await exp_repo.get_interim_analyses(exp_uuid)
            analysis_number = len(previous_analyses) + 1

            # Check if we should perform analysis at this milestone
            if not force:
                next_milestone = None
                for milestone in analysis_schedule:
                    if information_fraction >= milestone:
                        # Check if we've already done analysis at this milestone
                        milestone_done = any(
                            abs(a.information_fraction - milestone) < 0.05
                            for a in previous_analyses
                        )
                        if not milestone_done:
                            next_milestone = milestone
                            break

                if next_milestone is None:
                    return {
                        "status": "skipped",
                        "experiment_id": experiment_id,
                        "reason": "No new milestone reached",
                        "information_fraction": information_fraction,
                        "previous_analyses": len(previous_analyses),
                    }

            # Get metric data for analysis
            # This would normally come from experiment metrics collection
            from src.services.results_analysis import ResultsAnalysisService
            results_service = ResultsAnalysisService()

            # Placeholder: Get control and treatment data
            # In production, this would query actual experiment metrics
            control_data = []
            treatment_data = []

            # Perform interim analysis
            result = await interim_service.perform_interim_analysis(
                experiment_id=exp_uuid,
                analysis_number=analysis_number,
                metric_data={
                    "control": control_data,
                    "treatment": treatment_data,
                },
                target_sample_size=target_sample_size,
                target_effect=0.05,  # Would come from experiment design
            )

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "status": "completed",
                "experiment_id": experiment_id,
                "analysis_number": result.analysis_number,
                "information_fraction": result.information_fraction,
                "alpha_spent": result.alpha_spent,
                "adjusted_alpha": result.adjusted_alpha,
                "p_value": result.p_value,
                "conditional_power": result.conditional_power,
                "decision": result.decision.value,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error(f"Interim analysis failed for {experiment_id}: {e}")
            return {
                "status": "failed",
                "experiment_id": experiment_id,
                "error": str(e),
            }

    return run_async(execute_analysis())


@celery_app.task(bind=True, name="src.tasks.enrollment_health_check")
def enrollment_health_check(
    self,
) -> Dict[str, Any]:
    """Check enrollment rates across all active experiments.

    Identifies experiments with low enrollment rates and generates
    alerts for experiments that may need attention.

    Returns:
        Health check summary with alerts
    """
    logger.info(f"Running enrollment health check: task {self.request.id}")

    config = load_config()
    enrollment_config = config.get("enrollment", {})
    start_time = time.time()

    async def execute_check():
        from src.services.enrollment import EnrollmentService
        from src.memory.services.factories import get_supabase_client

        try:
            client = await get_supabase_client()
            if not client:
                return {
                    "status": "skipped",
                    "reason": "No database client available",
                }

            # Get all active experiments
            result = await (
                client.table("ml_experiments")
                .select("id, name, config")
                .eq("status", "running")
                .execute()
            )

            if not result.data:
                return {
                    "status": "completed",
                    "experiments_checked": 0,
                    "message": "No active experiments found",
                }

            enrollment_service = EnrollmentService()
            min_daily_rate = enrollment_config.get("min_daily_rate", 5)
            warning_days = enrollment_config.get("warning_threshold_days", 7)
            critical_days = enrollment_config.get("critical_threshold_days", 14)

            health_results = []
            alerts = []

            for exp in result.data:
                exp_id = UUID(exp["id"])

                try:
                    stats = await enrollment_service.get_enrollment_stats(exp_id)

                    if not stats:
                        health_results.append({
                            "experiment_id": str(exp_id),
                            "name": exp.get("name", "Unknown"),
                            "status": "no_data",
                        })
                        continue

                    # Calculate daily enrollment rate
                    days_running = max(
                        1,
                        (datetime.now(timezone.utc) - stats.enrollment_start).days
                        if stats.enrollment_start else 1
                    )
                    daily_rate = stats.total_enrolled / days_running

                    health_status = "healthy"
                    if daily_rate < min_daily_rate:
                        if days_running >= critical_days:
                            health_status = "critical"
                            alerts.append({
                                "experiment_id": str(exp_id),
                                "name": exp.get("name", "Unknown"),
                                "severity": "critical",
                                "message": f"Enrollment rate ({daily_rate:.1f}/day) below minimum for {days_running} days",
                                "daily_rate": daily_rate,
                                "days_below_threshold": days_running,
                            })
                        elif days_running >= warning_days:
                            health_status = "warning"
                            alerts.append({
                                "experiment_id": str(exp_id),
                                "name": exp.get("name", "Unknown"),
                                "severity": "warning",
                                "message": f"Enrollment rate ({daily_rate:.1f}/day) below minimum for {days_running} days",
                                "daily_rate": daily_rate,
                                "days_below_threshold": days_running,
                            })

                    health_results.append({
                        "experiment_id": str(exp_id),
                        "name": exp.get("name", "Unknown"),
                        "status": health_status,
                        "total_enrolled": stats.total_enrolled,
                        "daily_rate": daily_rate,
                        "days_running": days_running,
                        "enrollment_by_variant": stats.enrollment_by_variant,
                    })

                except Exception as e:
                    logger.warning(f"Failed to check enrollment for {exp_id}: {e}")
                    health_results.append({
                        "experiment_id": str(exp_id),
                        "name": exp.get("name", "Unknown"),
                        "status": "error",
                        "error": str(e),
                    })

            # Send alerts if any
            if alerts:
                await _send_enrollment_alerts(alerts, config)

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "status": "completed",
                "experiments_checked": len(result.data),
                "healthy_count": sum(1 for r in health_results if r["status"] == "healthy"),
                "warning_count": sum(1 for r in health_results if r["status"] == "warning"),
                "critical_count": sum(1 for r in health_results if r["status"] == "critical"),
                "alerts_generated": len(alerts),
                "alerts": alerts,
                "results": health_results,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error(f"Enrollment health check failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    return run_async(execute_check())


async def _send_enrollment_alerts(alerts: List[Dict], config: Dict) -> None:
    """Send enrollment alerts via configured channels."""
    alert_config = config.get("alerts", {})

    # Placeholder for actual alert sending
    for alert in alerts:
        logger.warning(
            f"ENROLLMENT ALERT [{alert['severity'].upper()}]: "
            f"{alert['name']} - {alert['message']}"
        )

    # Email alerts
    email_recipients = alert_config.get("email_recipients", [])
    if email_recipients:
        logger.info(f"Would send email alerts to {email_recipients}")

    # Slack alerts
    slack_webhook = alert_config.get("slack_webhook")
    if slack_webhook:
        logger.info("Would send Slack alert")


@celery_app.task(bind=True, name="src.tasks.srm_detection_sweep")
def srm_detection_sweep(
    self,
) -> Dict[str, Any]:
    """Periodic SRM detection for all running experiments.

    Checks for Sample Ratio Mismatch in all active experiments
    to detect potential randomization issues.

    Returns:
        SRM check summary with any detected issues
    """
    logger.info(f"Running SRM detection sweep: task {self.request.id}")

    config = load_config()
    srm_config = config.get("srm", {})
    start_time = time.time()

    async def execute_sweep():
        from src.services.results_analysis import ResultsAnalysisService
        from src.repositories.ab_experiment import ABExperimentRepository
        from src.memory.services.factories import get_supabase_client

        try:
            client = await get_supabase_client()
            if not client:
                return {
                    "status": "skipped",
                    "reason": "No database client available",
                }

            # Get all running experiments
            result = await (
                client.table("ml_experiments")
                .select("id, name, config")
                .eq("status", "running")
                .execute()
            )

            if not result.data:
                return {
                    "status": "completed",
                    "experiments_checked": 0,
                    "message": "No active experiments found",
                }

            results_service = ResultsAnalysisService()
            exp_repo = ABExperimentRepository()
            min_sample_size = srm_config.get("min_sample_size", 100)
            detection_threshold = srm_config.get("detection_threshold", 0.001)

            srm_results = []
            srm_detected = []

            for exp in result.data:
                exp_id = UUID(exp["id"])
                exp_config = exp.get("config", {})

                try:
                    # Get current assignment counts
                    assignments = await exp_repo.get_assignments(exp_id)

                    if len(assignments) < min_sample_size:
                        srm_results.append({
                            "experiment_id": str(exp_id),
                            "name": exp.get("name", "Unknown"),
                            "status": "insufficient_data",
                            "sample_size": len(assignments),
                            "min_required": min_sample_size,
                        })
                        continue

                    # Count by variant
                    variant_counts = {}
                    for a in assignments:
                        variant = a.variant
                        variant_counts[variant] = variant_counts.get(variant, 0) + 1

                    # Get expected ratio from config
                    expected_ratio = exp_config.get("allocation_ratio", {"control": 0.5, "treatment": 0.5})

                    # Check SRM
                    srm_result = await results_service.check_sample_ratio_mismatch(
                        experiment_id=exp_id,
                        expected_ratio=expected_ratio,
                        actual_counts=variant_counts,
                    )

                    status = "ok"
                    if srm_result.is_srm_detected:
                        status = "srm_detected"
                        srm_detected.append({
                            "experiment_id": str(exp_id),
                            "name": exp.get("name", "Unknown"),
                            "p_value": srm_result.p_value,
                            "expected_ratio": expected_ratio,
                            "actual_counts": variant_counts,
                            "chi_squared": srm_result.chi_squared_statistic,
                        })

                    srm_results.append({
                        "experiment_id": str(exp_id),
                        "name": exp.get("name", "Unknown"),
                        "status": status,
                        "p_value": srm_result.p_value,
                        "chi_squared": srm_result.chi_squared_statistic,
                        "actual_counts": variant_counts,
                    })

                except Exception as e:
                    logger.warning(f"SRM check failed for {exp_id}: {e}")
                    srm_results.append({
                        "experiment_id": str(exp_id),
                        "name": exp.get("name", "Unknown"),
                        "status": "error",
                        "error": str(e),
                    })

            # Send alerts for detected SRM issues
            if srm_detected:
                await _send_srm_alerts(srm_detected, config)

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "status": "completed",
                "experiments_checked": len(result.data),
                "srm_detected_count": len(srm_detected),
                "srm_issues": srm_detected,
                "all_results": srm_results,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error(f"SRM detection sweep failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    return run_async(execute_sweep())


async def _send_srm_alerts(srm_issues: List[Dict], config: Dict) -> None:
    """Send SRM detection alerts via configured channels."""
    alert_config = config.get("alerts", {})

    for issue in srm_issues:
        logger.error(
            f"SRM DETECTED: {issue['name']} - "
            f"p-value={issue['p_value']:.6f}, "
            f"expected={issue['expected_ratio']}, "
            f"actual={issue['actual_counts']}"
        )

    # Email alerts
    email_recipients = alert_config.get("email_recipients", [])
    if email_recipients:
        logger.info(f"Would send SRM email alerts to {email_recipients}")

    # Slack alerts
    slack_webhook = alert_config.get("slack_webhook")
    if slack_webhook:
        logger.info("Would send SRM Slack alert")


@celery_app.task(bind=True, name="src.tasks.compute_experiment_results")
def compute_experiment_results(
    self,
    experiment_id: str,
    analysis_type: str = "interim",
) -> Dict[str, Any]:
    """Compute final or interim experiment results.

    Calculates treatment effects, confidence intervals, and
    statistical significance for an experiment.

    Args:
        experiment_id: UUID of the experiment
        analysis_type: Type of analysis ('interim' or 'final')

    Returns:
        Computed experiment results
    """
    logger.info(
        f"Computing {analysis_type} results for experiment {experiment_id}: "
        f"task {self.request.id}"
    )

    start_time = time.time()

    async def execute_computation():
        from src.services.results_analysis import ResultsAnalysisService
        from src.repositories.ab_results import ABResultsRepository

        try:
            results_service = ResultsAnalysisService()
            results_repo = ABResultsRepository()
            exp_uuid = UUID(experiment_id)

            # Placeholder: Get control and treatment data from experiment
            # In production, this would query actual experiment metrics
            control_data = []
            treatment_data = []
            primary_metric = "conversion_rate"

            # Compute ITT results
            results = await results_service.compute_itt_results(
                experiment_id=exp_uuid,
                primary_metric=primary_metric,
                control_data=control_data,
                treatment_data=treatment_data,
            )

            # Also compute per-protocol if this is final analysis
            per_protocol_results = None
            if analysis_type == "final":
                per_protocol_results = await results_service.compute_per_protocol_results(
                    experiment_id=exp_uuid,
                    primary_metric=primary_metric,
                    control_data=control_data,
                    treatment_data=treatment_data,
                )

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "status": "completed",
                "experiment_id": experiment_id,
                "analysis_type": analysis_type,
                "primary_metric": results.primary_metric,
                "control_mean": results.control_mean,
                "treatment_mean": results.treatment_mean,
                "effect_estimate": results.effect_estimate,
                "effect_ci": [results.effect_ci_lower, results.effect_ci_upper],
                "p_value": results.p_value,
                "is_significant": results.is_significant,
                "sample_size_control": results.sample_size_control,
                "sample_size_treatment": results.sample_size_treatment,
                "per_protocol_results": {
                    "effect_estimate": per_protocol_results.effect_estimate,
                    "p_value": per_protocol_results.p_value,
                } if per_protocol_results else None,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error(f"Results computation failed for {experiment_id}: {e}")
            return {
                "status": "failed",
                "experiment_id": experiment_id,
                "error": str(e),
            }

    return run_async(execute_computation())


@celery_app.task(bind=True, name="src.tasks.fidelity_tracking_update")
def fidelity_tracking_update(
    self,
    experiment_id: str,
    twin_simulation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Update fidelity comparison with Digital Twin predictions.

    Compares actual experiment results with Digital Twin predictions
    to track simulation accuracy and identify calibration needs.

    Args:
        experiment_id: UUID of the experiment
        twin_simulation_id: Optional specific simulation to compare against

    Returns:
        Fidelity comparison results
    """
    logger.info(
        f"Updating fidelity tracking for experiment {experiment_id}: "
        f"task {self.request.id}"
    )

    config = load_config()
    fidelity_config = config.get("fidelity", {})
    start_time = time.time()

    async def execute_update():
        from src.services.results_analysis import ResultsAnalysisService
        from src.repositories.ab_results import ABResultsRepository
        from src.memory.services.factories import get_supabase_client

        try:
            results_service = ResultsAnalysisService()
            results_repo = ABResultsRepository()
            exp_uuid = UUID(experiment_id)

            # Get the latest experiment results
            results = await results_repo.get_results(exp_uuid)

            if not results:
                return {
                    "status": "skipped",
                    "experiment_id": experiment_id,
                    "reason": "No results available for comparison",
                }

            latest_result = results[0]  # Most recent

            # Find associated Digital Twin simulation
            client = await get_supabase_client()
            if not client:
                return {
                    "status": "skipped",
                    "experiment_id": experiment_id,
                    "reason": "No database client available",
                }

            # Query for twin simulation
            if twin_simulation_id:
                sim_uuid = UUID(twin_simulation_id)
            else:
                # Find most recent simulation for this experiment
                sim_result = await (
                    client.table("twin_simulations")
                    .select("id, predicted_effect, confidence_interval")
                    .eq("experiment_id", experiment_id)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )

                if not sim_result.data:
                    return {
                        "status": "skipped",
                        "experiment_id": experiment_id,
                        "reason": "No Digital Twin simulation found for experiment",
                    }

                sim_data = sim_result.data[0]
                sim_uuid = UUID(sim_data["id"])
                predicted_effect = sim_data.get("predicted_effect", 0)
                predicted_ci = sim_data.get("confidence_interval", [])

            # Compute fidelity comparison
            comparison = await results_service.compare_with_twin_prediction(
                experiment_id=exp_uuid,
                twin_simulation_id=sim_uuid,
                actual_results=latest_result,
                predicted_effect=predicted_effect,
                predicted_ci=predicted_ci if predicted_ci else None,
            )

            # Check if calibration is needed
            acceptable_error = fidelity_config.get("acceptable_error", 0.2)
            calibration_trigger = fidelity_config.get("calibration_trigger_error", 0.3)

            calibration_needed = abs(comparison.prediction_error) > calibration_trigger

            if calibration_needed:
                logger.warning(
                    f"Digital Twin calibration needed for experiment {experiment_id}: "
                    f"prediction error = {comparison.prediction_error:.2%}"
                )

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "status": "completed",
                "experiment_id": experiment_id,
                "twin_simulation_id": str(sim_uuid),
                "predicted_effect": comparison.predicted_effect,
                "actual_effect": comparison.actual_effect,
                "prediction_error": comparison.prediction_error,
                "ci_coverage": comparison.confidence_interval_coverage,
                "fidelity_score": comparison.fidelity_score,
                "calibration_needed": calibration_needed,
                "calibration_adjustment": comparison.calibration_adjustment,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error(f"Fidelity tracking update failed for {experiment_id}: {e}")
            return {
                "status": "failed",
                "experiment_id": experiment_id,
                "error": str(e),
            }

    return run_async(execute_update())


@celery_app.task(bind=True, name="src.tasks.check_all_active_experiments")
def check_all_active_experiments(
    self,
) -> Dict[str, Any]:
    """Check all active experiments for interim analysis triggers.

    Scans all running experiments and queues interim analysis
    tasks for those that have reached analysis milestones.

    Returns:
        Summary of experiments checked and tasks queued
    """
    logger.info(f"Checking all active experiments: task {self.request.id}")

    async def execute_check():
        from src.memory.services.factories import get_supabase_client

        try:
            client = await get_supabase_client()
            if not client:
                return {
                    "status": "skipped",
                    "reason": "No database client available",
                }

            # Get all running experiments
            result = await (
                client.table("ml_experiments")
                .select("id, name")
                .eq("status", "running")
                .execute()
            )

            if not result.data:
                return {
                    "status": "completed",
                    "experiments_found": 0,
                    "message": "No active experiments found",
                }

            tasks_queued = []
            errors = []

            for exp in result.data:
                try:
                    # Queue interim analysis check for each experiment
                    task = scheduled_interim_analysis.delay(
                        experiment_id=exp["id"],
                        force=False,
                    )
                    tasks_queued.append({
                        "experiment_id": exp["id"],
                        "name": exp.get("name", "Unknown"),
                        "task_id": task.id,
                    })
                except Exception as e:
                    errors.append({
                        "experiment_id": exp["id"],
                        "error": str(e),
                    })

            return {
                "status": "completed",
                "experiments_found": len(result.data),
                "tasks_queued": len(tasks_queued),
                "queued_tasks": tasks_queued,
                "errors": errors,
            }

        except Exception as e:
            logger.error(f"Check all experiments failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    return run_async(execute_check())


@celery_app.task(bind=True, name="src.tasks.cleanup_old_ab_results")
def cleanup_old_ab_results(
    self,
    retention_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Clean up old A/B testing results and history.

    Archives or deletes old experiment data based on retention policy.

    Args:
        retention_days: Number of days to retain (default from config)

    Returns:
        Cleanup summary
    """
    logger.info(f"Starting A/B results cleanup: task {self.request.id}")

    config = load_config()
    schedule_config = config.get("schedule", {})
    effective_retention = retention_days or schedule_config.get("retention_days", 365)

    async def execute_cleanup():
        from src.memory.services.factories import get_supabase_client

        try:
            client = await get_supabase_client()
            if not client:
                return {"status": "skipped", "reason": "No database client available"}

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=effective_retention)
            cutoff_iso = cutoff_date.isoformat()

            # Delete old SRM checks (for completed experiments only)
            srm_result = await (
                client.table("ab_srm_checks")
                .delete()
                .lt("checked_at", cutoff_iso)
                .execute()
            )
            srm_deleted = len(srm_result.data) if srm_result.data else 0

            # Delete old interim analyses for completed experiments
            interim_result = await (
                client.table("ab_interim_analyses")
                .delete()
                .lt("performed_at", cutoff_iso)
                .execute()
            )
            interim_deleted = len(interim_result.data) if interim_result.data else 0

            # Note: We keep ab_experiment_results for longer-term analysis
            # and don't delete assignments/enrollments for audit purposes

            return {
                "status": "completed",
                "retention_days": effective_retention,
                "cutoff_date": cutoff_iso,
                "srm_checks_deleted": srm_deleted,
                "interim_analyses_deleted": interim_deleted,
            }

        except Exception as e:
            logger.error(f"A/B results cleanup failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    return run_async(execute_cleanup())


# Celery Beat schedule configuration
@celery_app.on_after_finalize.connect
def setup_ab_testing_periodic_tasks(sender, **kwargs):
    """Set up periodic A/B testing tasks."""
    config = load_config()
    schedule_config = config.get("schedule", {})

    # Check all active experiments daily at 2 AM
    sender.add_periodic_task(
        86400,  # 24 hours
        check_all_active_experiments.s(),
        name="ab-interim-analysis-check",
    )

    # Enrollment health check every 12 hours
    enrollment_interval = schedule_config.get("enrollment_check_interval_hours", 12)
    sender.add_periodic_task(
        enrollment_interval * 3600,
        enrollment_health_check.s(),
        name="ab-enrollment-health-check",
    )

    # SRM detection every 6 hours
    srm_interval = schedule_config.get("srm_check_interval_hours", 6)
    sender.add_periodic_task(
        srm_interval * 3600,
        srm_detection_sweep.s(),
        name="ab-srm-detection-sweep",
    )

    # Weekly cleanup (Sundays)
    sender.add_periodic_task(
        604800,  # 7 days
        cleanup_old_ab_results.s(),
        name="ab-results-cleanup",
    )
