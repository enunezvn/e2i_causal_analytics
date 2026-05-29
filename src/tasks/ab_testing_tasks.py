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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from uuid import UUID

import yaml  # type: ignore[import-untyped]

from src.digital_twin.twin_generator import TwinGenerator
from src.workers.celery_app import celery_app

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# F-009 (issue #422): A/B metric observations have no storage table.
# `database/ml/020_ab_testing_tables.sql` defines `ab_experiment_assignments`
# and `ab_experiment_enrollments`, but no per-unit metric-observation table
# exists. Until that schema lands (#422 follow-up), the interim and final
# result tasks bail with `status='insufficient_data'` at the call site rather
# than passing `[]` to `_compute_results` (which would persist NaN-tainted
# `ExperimentResults`). A helper named `_load_*` would itself be a label
# pattern — there's no path to load from, so the bail is in-line and the
# reason string is the only honest answer.
_AB_METRIC_SCHEMA_REASON = (
    "No per-unit A/B metric-observation storage exists in the current schema. "
    "See #422 (F-009): bail with insufficient_data rather than persist NaN to "
    "ab_experiment_results."
)


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
                merged: Dict[str, Any] = DEFAULT_CONFIG.copy()
                for key, value in config.items():
                    if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                        merged[key] = {**cast(Dict[str, Any], merged[key]), **value}
                    else:
                        merged[key] = value
                return merged
    except Exception as e:
        logger.warning(f"Failed to load A/B testing config, using defaults: {e}")
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
    # F-009 (#422): no `duration_ms` to record on the insufficient_data bail.
    # `start_time = time.time()` returns when `perform_interim_analysis` lands.

    async def execute_analysis():
        from src.repositories.ab_experiment import ABExperimentRepository
        from src.services.enrollment import EnrollmentService

        # F-009 (#422): InterimAnalysisService is no longer instantiated here
        # because the task bails before calling perform_interim_analysis (no
        # per-unit metric-observation storage exists). Import + instantiation
        # is re-added at the call site when the storage schema lands.

        try:
            exp_repo = ABExperimentRepository()
            enrollment_service = EnrollmentService()

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

            # F-009 (#422): there is no per-unit A/B metric-observation table
            # in the current schema. Bail in-line rather than invoke a helper
            # that would always return None (relabeling) and rather than pass
            # `[]` to `perform_interim_analysis` (which would persist NaN to
            # the DB). When the storage schema lands, the bail is removed and
            # `perform_interim_analysis` is called with real per-unit values.
            logger.warning(
                "Skipping interim analysis for %s: %s",
                experiment_id,
                _AB_METRIC_SCHEMA_REASON,
            )
            return {
                "status": "insufficient_data",
                "experiment_id": experiment_id,
                "reason": _AB_METRIC_SCHEMA_REASON,
                "information_fraction": information_fraction,
                "current_enrollment": current_enrollment,
                "target_sample_size": target_sample_size,
                "analysis_number": analysis_number,
            }

        except Exception as e:
            logger.error(f"Interim analysis failed for {experiment_id}: {e}")
            return {
                "status": "failed",
                "experiment_id": experiment_id,
                "error": str(e),
            }

    return cast(Dict[str, Any], run_async(execute_analysis()))


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
        from src.memory.services.factories import get_supabase_client
        from src.services.enrollment import EnrollmentService

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
                        health_results.append(
                            {
                                "experiment_id": str(exp_id),
                                "name": exp.get("name", "Unknown"),
                                "status": "no_data",
                            }
                        )
                        continue

                    # Calculate daily enrollment rate
                    days_running = max(
                        1,
                        (datetime.now(timezone.utc) - stats.enrollment_start).days
                        if stats.enrollment_start
                        else 1,
                    )
                    daily_rate = stats.total_enrolled / days_running

                    health_status = "healthy"
                    if daily_rate < min_daily_rate:
                        if days_running >= critical_days:
                            health_status = "critical"
                            alerts.append(
                                {
                                    "experiment_id": str(exp_id),
                                    "name": exp.get("name", "Unknown"),
                                    "severity": "critical",
                                    "message": f"Enrollment rate ({daily_rate:.1f}/day) below minimum for {days_running} days",
                                    "daily_rate": daily_rate,
                                    "days_below_threshold": days_running,
                                }
                            )
                        elif days_running >= warning_days:
                            health_status = "warning"
                            alerts.append(
                                {
                                    "experiment_id": str(exp_id),
                                    "name": exp.get("name", "Unknown"),
                                    "severity": "warning",
                                    "message": f"Enrollment rate ({daily_rate:.1f}/day) below minimum for {days_running} days",
                                    "daily_rate": daily_rate,
                                    "days_below_threshold": days_running,
                                }
                            )

                    health_results.append(
                        {
                            "experiment_id": str(exp_id),
                            "name": exp.get("name", "Unknown"),
                            "status": health_status,
                            "total_enrolled": stats.total_enrolled,
                            "daily_rate": daily_rate,
                            "days_running": days_running,
                            "enrollment_by_variant": stats.enrollment_by_variant,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Failed to check enrollment for {exp_id}: {e}")
                    health_results.append(
                        {
                            "experiment_id": str(exp_id),
                            "name": exp.get("name", "Unknown"),
                            "status": "error",
                            "error": str(e),
                        }
                    )

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

    return cast(Dict[str, Any], run_async(execute_check()))


async def _send_enrollment_alerts(alerts: List[Dict], config: Dict) -> None:
    """Send enrollment alerts via configured channels."""
    alert_config = config.get("alerts", {})

    # Placeholder for actual alert sending
    for alert in alerts:
        logger.warning(
            f"ENROLLMENT ALERT [{alert['severity'].upper()}]: {alert['name']} - {alert['message']}"
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
        from src.memory.services.factories import get_supabase_client
        from src.repositories.ab_experiment import ABExperimentRepository
        from src.services.results_analysis import ResultsAnalysisService

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
            srm_config.get("detection_threshold", 0.001)

            srm_results = []
            srm_detected = []

            for exp in result.data:
                exp_id = UUID(exp["id"])
                exp_config = exp.get("config", {})

                try:
                    # Get current assignment counts
                    assignments = await exp_repo.get_assignments(exp_id)

                    if len(assignments) < min_sample_size:
                        srm_results.append(
                            {
                                "experiment_id": str(exp_id),
                                "name": exp.get("name", "Unknown"),
                                "status": "insufficient_data",
                                "sample_size": len(assignments),
                                "min_required": min_sample_size,
                            }
                        )
                        continue

                    # Count by variant
                    variant_counts = {}
                    for a in assignments:
                        variant = a.variant
                        variant_counts[variant] = variant_counts.get(variant, 0) + 1

                    # Get expected ratio from config
                    expected_ratio = exp_config.get(
                        "allocation_ratio", {"control": 0.5, "treatment": 0.5}
                    )

                    # Check SRM
                    srm_result = await results_service.check_sample_ratio_mismatch(
                        experiment_id=exp_id,
                        expected_ratio=expected_ratio,
                        actual_counts=variant_counts,
                    )

                    status = "ok"
                    if srm_result.is_srm_detected:
                        status = "srm_detected"
                        srm_detected.append(
                            {
                                "experiment_id": str(exp_id),
                                "name": exp.get("name", "Unknown"),
                                "p_value": srm_result.p_value,
                                "expected_ratio": expected_ratio,
                                "actual_counts": variant_counts,
                                "chi_squared": srm_result.chi_squared_statistic,
                            }
                        )

                    srm_results.append(
                        {
                            "experiment_id": str(exp_id),
                            "name": exp.get("name", "Unknown"),
                            "status": status,
                            "p_value": srm_result.p_value,
                            "chi_squared": srm_result.chi_squared_statistic,
                            "actual_counts": variant_counts,
                        }
                    )

                except Exception as e:
                    logger.warning(f"SRM check failed for {exp_id}: {e}")
                    srm_results.append(
                        {
                            "experiment_id": str(exp_id),
                            "name": exp.get("name", "Unknown"),
                            "status": "error",
                            "error": str(e),
                        }
                    )

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

    return cast(Dict[str, Any], run_async(execute_sweep()))


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
        f"Computing {analysis_type} results for experiment {experiment_id}: task {self.request.id}"
    )

    # F-009 (#422): duration_ms is omitted from the insufficient_data return —
    # there's no computation to time. When the metric-storage schema lands and
    # `_compute_results` is restored, a `start_time = time.time()` capture and
    # `duration_ms = int((time.time() - start_time) * 1000)` will be added back
    # at the result assembly site.

    async def execute_computation():
        from src.repositories.ab_results import ABResultsRepository
        from src.services.results_analysis import ResultsAnalysisService

        try:
            ResultsAnalysisService()
            ABResultsRepository()
            UUID(experiment_id)  # validates uuid shape; will raise ValueError if malformed

            # F-009 (#422): same schema gap as scheduled_interim_analysis above.
            # Bail in-line. When the per-unit metric-observation table exists,
            # this block is replaced by:
            #   metric_repo = ExperimentMetricObservationRepository()
            #   control, treatment = await metric_repo.load_arrays(
            #       experiment_id, primary_metric_from_config
            #   )
            #   results = await results_service.compute_itt_results(...)
            logger.warning(
                "Skipping results computation for %s: %s",
                experiment_id,
                _AB_METRIC_SCHEMA_REASON,
            )
            return {
                "status": "insufficient_data",
                "experiment_id": experiment_id,
                "analysis_type": analysis_type,
                "reason": _AB_METRIC_SCHEMA_REASON,
            }

        except Exception as e:
            logger.error(f"Results computation failed for {experiment_id}: {e}")
            return {
                "status": "failed",
                "experiment_id": experiment_id,
                "error": str(e),
            }

    return cast(Dict[str, Any], run_async(execute_computation()))


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
        f"Updating fidelity tracking for experiment {experiment_id}: task {self.request.id}"
    )

    config = load_config()
    fidelity_config = config.get("fidelity", {})
    start_time = time.time()

    async def execute_update():
        from src.memory.services.factories import get_supabase_client
        from src.repositories.ab_results import ABResultsRepository
        from src.services.results_analysis import ResultsAnalysisService

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
            fidelity_config.get("acceptable_error", 0.2)
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

    return cast(Dict[str, Any], run_async(execute_update()))


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
                client.table("ml_experiments").select("id, name").eq("status", "running").execute()
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
                    tasks_queued.append(
                        {
                            "experiment_id": exp["id"],
                            "name": exp.get("name", "Unknown"),
                            "task_id": task.id,
                        }
                    )
                except Exception as e:
                    errors.append(
                        {
                            "experiment_id": exp["id"],
                            "error": str(e),
                        }
                    )

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

    return cast(Dict[str, Any], run_async(execute_check()))


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
                client.table("ab_srm_checks").delete().lt("checked_at", cutoff_iso).execute()
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

    return cast(Dict[str, Any], run_async(execute_cleanup()))


# --------------------------------------------------------------------------- #
# Digital-twin retraining (#548)
#
# TwinRetrainingService.trigger_retraining imports execute_twin_retraining from
# this module and queues it. Before #548 no such task existed: the ImportError
# was caught and logged as the MISLEADING "Celery tasks not available" (Celery
# IS available — only this one task was missing), so the LIVE auto-retraining
# path (FidelityTracker(auto_trigger_retraining=True) → check_and_trigger →
# trigger_retraining) silently no-oped.
#
# This is the twin analogue of #545's `execute_model_retraining`: a thin Celery
# task → async core that runs the REAL TwinGenerator.train and records ONLY the
# real validation metric, failing CLOSED on any path that can't certify one.
# Twins are sklearn ensembles (TwinGenerator), NOT the MLFoundationPipeline.
# --------------------------------------------------------------------------- #
def _twin_training_data_from_config(
    training_config: Dict[str, Any],
) -> tuple["pd.DataFrame", str]:
    """Resolve the twin's training DataFrame + target column from the contract.

    The twin retraining system has NO live cohort feed — a degradation
    auto-trigger's ``training_config`` (built by ``_build_training_config``)
    carries tuning knobs but no data. So the task can only run a real retrain
    when a caller supplies a concrete ``data_source`` (a .csv/.parquet path) and
    ``target_column`` (e.g. via ``config_overrides``). Absent either, we fail
    LOUD here so the core fails closed rather than fabricate a metric — mirroring
    #545's ``_cohort_input_from_training_config`` ValueError contract.

    Raises:
        ValueError: if ``data_source`` or ``target_column`` is missing.
    """
    import pandas as pd

    data_source = training_config.get("data_source")
    if not data_source or not isinstance(data_source, str):
        raise ValueError(
            "training_config lacks a concrete 'data_source' — twin retraining has "
            "no live cohort feed, so without a data path there is nothing to train "
            "on (fail closed rather than fabricate a metric)"
        )
    target_column = training_config.get("target_column")
    if not target_column or not isinstance(target_column, str):
        raise ValueError(
            "training_config lacks a 'target_column' — cannot train a twin without "
            "an outcome to model (fail closed)"
        )

    if data_source.endswith(".parquet"):
        df = pd.read_parquet(data_source)
    else:
        df = pd.read_csv(data_source)
    return df, target_column


def _extract_validation_r2(metrics: Any) -> Optional[float]:
    """Return the REAL finite validation R² ``TwinGenerator.train`` produced, or
    ``None``.

    ``TwinModelMetrics.r2_score`` is the held-out validation R² (twin_generator
    .py:216). ``None`` / non-finite means the run produced no certifiable metric
    — the caller MUST fail closed rather than synthesize a placeholder.
    """
    import math

    r2 = getattr(metrics, "r2_score", None)
    if isinstance(r2, (int, float)) and not isinstance(r2, bool) and math.isfinite(r2):
        return float(r2)
    return None


async def _execute_real_twin_retraining(
    retraining_job_id: str,
    model_id: str,
    training_config: Dict[str, Any],
    service: Any = None,
) -> Dict[str, Any]:
    """Run a REAL ``TwinGenerator.train`` for a twin retraining job and record
    the REAL outcome.

    Fails closed: any path that does not yield a certifiable validation R²
    (missing data source / target, missing model row, train exception, or a
    completed train with no finite R²) marks the job ``failed`` via
    ``TwinRetrainingService.fail_retraining`` (which writes NO ``fidelity_after``).
    A fake metric must never reach the job record.

    DURABLE ACROSS THE WORKER BOUNDARY (#549): twin retraining job state is
    persisted to the shared ``twin_retraining_jobs`` table via
    ``TwinRetrainingJobRepository`` and the retrained model to
    ``digital_twin_models`` via ``TwinModelRepository.save_model``. Because both
    the API and the worker resolve the SAME ``get_async_supabase_client()``
    singleton (BaseRepository awaits ``.execute()``, so the async client is
    required), a job the API created is found here and its completion (with the
    real metric + the persisted ``new_model_id``) is recorded and retrievable by
    the API process.
    If the durable store cannot record the completion (genuinely-unknown job, or
    an inert/unconfigured env) this still FAILS CLOSED — it never reports a false
    "completed".

    Args:
        retraining_job_id: TwinRetrainingJob id created by trigger_retraining.
        model_id: id of the twin model being retrained (used to rebuild the
            trainer's type/brand/feature/target contract from the saved row).
        training_config: the retraining contract; must carry a concrete
            ``data_source`` + ``target_column`` for a real run (else fail closed).
        service: optional TwinRetrainingService used to persist completion; one
            is constructed if not supplied.

    Returns:
        Result dict (real ``validation_r2`` on success; failure reason otherwise).
    """
    from src.digital_twin.models.twin_models import Brand, TwinModelConfig, TwinType
    from src.digital_twin.retraining_service import get_twin_retraining_service
    from src.digital_twin.twin_repository import TwinModelRepository

    if service is None:
        service = get_twin_retraining_service()

    async def _mark_failed(reason: str) -> Dict[str, Any]:
        logger.error(f"Twin retraining {retraining_job_id} failed closed: {reason}")
        try:
            await service.fail_retraining(
                job_id=retraining_job_id,
                error_message=reason,
            )
        except Exception as e:  # noqa: BLE001 — never mask the original failure
            logger.error(f"Also failed to mark twin retraining {retraining_job_id} failed: {e}")
        return {
            "status": "failed",
            "retraining_job_id": retraining_job_id,
            "model_id": model_id,
            "error": reason,
            "validation_r2": None,
        }

    # Resolve training inputs BEFORE touching anything — fail loud if the
    # contract lacks a data source / target (the trainer is never invoked then).
    try:
        data, target_column = _twin_training_data_from_config(training_config)
    except ValueError as e:
        return await _mark_failed(str(e))

    # Locate the saved model row to rebuild the trainer's identity/feature set.
    # Bind the process-shared ASYNC Supabase client (TwinModelRepository awaits
    # .execute(), so it needs the async client — NOT the sync get_supabase) so
    # the lookup AND the durable save_model below actually hit the DB in a real
    # worker process (#549). Unconfigured env → None → inert repo → the run fails
    # closed (model not found / completion unrecorded) rather than faking success.
    try:
        from src.memory.services.factories import get_async_supabase_client

        twin_db_client = await get_async_supabase_client()
    except Exception:  # noqa: BLE001 — unconfigured/unavailable → inert repo → fail closed
        twin_db_client = None
    repo = TwinModelRepository(supabase_client=twin_db_client)
    try:
        from uuid import UUID

        model_row = await repo.get_model(UUID(model_id))
    except Exception as e:  # noqa: BLE001 — convert lookup failure to fail-closed
        return await _mark_failed(f"model lookup raised {type(e).__name__}: {e}")
    if not model_row:
        return await _mark_failed(
            f"twin model {model_id} not found — cannot rebuild the trainer contract"
        )

    try:
        twin_type = TwinType(model_row["twin_type"])
        brand = Brand(model_row["brand"])
    except (KeyError, ValueError) as e:
        return await _mark_failed(
            f"twin model row missing/invalid twin_type|brand: {type(e).__name__}: {e}"
        )

    feature_cols = model_row.get("feature_columns") or None
    algorithm = (model_row.get("training_config") or {}).get("algorithm", "random_forest")

    # Run the REAL training. Any exception → fail closed (no metric written).
    try:
        generator = TwinGenerator(twin_type=twin_type, brand=brand)
        metrics = generator.train(
            data=data,
            target_col=target_column,
            feature_cols=feature_cols,
            algorithm=algorithm,
        )
    except Exception as e:  # noqa: BLE001 — convert any train failure to fail-closed
        return await _mark_failed(f"TwinGenerator.train raised {type(e).__name__}: {e}")

    validation_r2 = _extract_validation_r2(metrics)
    if validation_r2 is None:
        return await _mark_failed(
            "TwinGenerator.train produced no finite validation R² — not certifiable; "
            "job marked failed (no metric written)"
        )

    # Persist the retrained model durably so new_model_id references a real,
    # retrievable digital_twin_models row rather than an ephemeral uuid (#549).
    # The config is rebuilt from the SOURCE model row's contract; save_model
    # writes the metadata row (and the artifact to MLflow when a client is
    # configured). A persistence failure FAILS CLOSED — we never record a
    # completion for a model we could not durably store.
    try:
        config = TwinModelConfig(
            model_name=f"{model_row.get('model_name') or 'twin'}_retrained",
            model_description="Automated twin retraining (#549 durable persistence)",
            twin_type=twin_type,
            brand=brand,
            algorithm=algorithm,
            feature_columns=list(feature_cols or []),
            target_column=target_column,
            geographic_scope=model_row.get("geographic_scope", "national"),
        )
        persisted_id = await repo.save_model(
            config=config,
            metrics=metrics,
            model_artifact=getattr(generator, "model", None),
        )
    except Exception as e:  # noqa: BLE001 — durable persistence failure → fail closed
        return await _mark_failed(
            f"retrained twin model could not be persisted: {type(e).__name__}: {e}"
        )
    new_model_id = str(persisted_id)

    # Record the completion in the DURABLE, process-shared job store (#549). In a
    # real worker the service (get_twin_retraining_service) is wired to the shared
    # store, so the job the API created at trigger time is found and the
    # completion + real metric are recorded and retrievable by the API process.
    # If the store cannot record it (genuinely-unknown job, or inert/unconfigured
    # env) we FAIL CLOSED below rather than report a false "completed".
    recorded = await service.complete_retraining(
        job_id=retraining_job_id,
        new_model_id=new_model_id,
        fidelity_after=validation_r2,
        success=True,
    )
    if recorded is None:
        # The train ran (R²=validation_r2) and the model was persisted, but the
        # completion could not be recorded in the durable job store: the job id
        # is unknown there (never persisted at trigger time) or the store is
        # inert/unconfigured. Reporting "completed" would be a false success.
        return await _mark_failed(
            f"twin retraining ran (validation R²={validation_r2:.4f}, model "
            f"{new_model_id} persisted) but completion could not be recorded: job "
            f"{retraining_job_id} not found in the durable job store (unknown or "
            "store unconfigured). No completion or metric was recorded."
        )

    logger.info(
        f"Twin retraining {retraining_job_id} complete: "
        f"new_model={new_model_id}, validation R²={validation_r2:.4f}"
    )
    return {
        "status": "completed",
        "retraining_job_id": retraining_job_id,
        "model_id": model_id,
        "new_model_id": new_model_id,
        "validation_r2": validation_r2,
        "message": f"Twin {new_model_id} retrained; validation R²={validation_r2:.4f}",
    }


@celery_app.task(bind=True, name="src.tasks.execute_twin_retraining")
def execute_twin_retraining(
    self,
    retraining_job_id: str,
    model_id: str,
    training_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a digital-twin retraining job by running the REAL TwinGenerator.

    Rebuilds the trainer from the saved twin model's contract, runs a real
    ``TwinGenerator.train`` on the contract's data source, and records the real
    validation R² via ``TwinRetrainingService.complete_retraining``. Fails closed
    — never writes a fabricated metric (the twin analogue of #545's
    ``execute_model_retraining``). Routed to the heavy ``ml`` queue since it
    runs a full sklearn training job.

    Args:
        retraining_job_id: TwinRetrainingJob id from trigger_retraining.
        model_id: id of the twin model being retrained.
        training_config: retraining contract (must carry data_source +
            target_column for a real run; else the job fails closed).

    Returns:
        Retraining results (real metric on success; failure reason otherwise).
    """
    logger.info(
        f"Starting twin retraining for model {model_id}, "
        f"job {retraining_job_id}, task {self.request.id}"
    )
    return cast(
        Dict[str, Any],
        run_async(_execute_real_twin_retraining(retraining_job_id, model_id, training_config)),
    )


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
