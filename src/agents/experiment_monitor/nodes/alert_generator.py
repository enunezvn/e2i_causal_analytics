"""Alert Generator Node.

This node aggregates all monitoring results and generates alerts
with severity levels and recommended actions.

Performance Target: <500ms
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from src.agents.experiment_monitor.dspy_integration import (
    get_experiment_monitor_dspy_integration,
)
from src.agents.experiment_monitor.state import (
    ErrorDetails,
    ExperimentMonitorState,
    MonitorAlert,
)

logger = logging.getLogger(__name__)


class AlertGeneratorNode:
    """Generates alerts from monitoring results.

    Alert Generation Strategy:
    1. Aggregate SRM issues into SRM alerts
    2. Aggregate enrollment issues into enrollment alerts
    3. Generate interim analysis trigger notifications
    4. Create summary and recommended actions

    Performance Target: <500ms
    """

    def __init__(self, use_dspy_prompts: bool = True):
        """Initialize alert generator node.

        Args:
            use_dspy_prompts: Whether to use DSPy-optimized prompts for messages
        """
        self.use_dspy_prompts = use_dspy_prompts
        self._dspy_integration = None

    @property
    def dspy_integration(self):
        """Lazy-load DSPy integration."""
        if self._dspy_integration is None and self.use_dspy_prompts:
            try:
                self._dspy_integration = get_experiment_monitor_dspy_integration()
                logger.debug("DSPy integration loaded for alert generation")
            except Exception as e:
                logger.warning(f"DSPy integration unavailable: {e}")
        return self._dspy_integration

    async def execute(self, state: ExperimentMonitorState) -> ExperimentMonitorState:
        """Execute alert generation.

        Args:
            state: Current agent state with monitoring results

        Returns:
            Updated state with alerts and summary
        """
        start_time = time.time()

        try:
            state["status"] = "alerting"

            alerts: List[MonitorAlert] = []

            # Generate SRM alerts
            srm_alerts = self._generate_srm_alerts(state)
            alerts.extend(srm_alerts)

            # Generate enrollment alerts
            enrollment_alerts = self._generate_enrollment_alerts(state)
            alerts.extend(enrollment_alerts)

            # Generate stale data alerts
            stale_data_alerts = self._generate_stale_data_alerts(state)
            alerts.extend(stale_data_alerts)

            # Generate interim trigger alerts
            interim_alerts = self._generate_interim_alerts(state)
            alerts.extend(interim_alerts)

            # Generate fidelity alerts (if any)
            fidelity_alerts = self._generate_fidelity_alerts(state)
            alerts.extend(fidelity_alerts)

            # Create summary
            summary = self._create_summary(state, alerts)

            # Generate recommendations
            recommendations = self._generate_recommendations(state, alerts)

            # Update state
            state["alerts"] = alerts
            state["monitor_summary"] = summary
            state["recommended_actions"] = recommendations
            state["status"] = "completed"

            # Update latency
            latency_ms = int((time.time() - start_time) * 1000)
            state["check_latency_ms"] = state.get("check_latency_ms", 0) + latency_ms

        except Exception as e:
            error: ErrorDetails = {
                "node": "alert_generator",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["errors"] = state.get("errors", []) + [error]
            state["status"] = "failed"
            state["alerts"] = []
            state["monitor_summary"] = "Alert generation failed"
            state["recommended_actions"] = []

        return state

    def _get_srm_message(
        self,
        exp_name: str,
        chi_squared: float,
        p_value: float,
        expected_ratio: str,
        actual_counts: str,
    ) -> str:
        """Get SRM alert message, using DSPy prompt if available.

        Args:
            exp_name: Experiment name
            chi_squared: Chi-squared test statistic
            p_value: P-value from test
            expected_ratio: Expected ratio string
            actual_counts: Actual counts string

        Returns:
            Formatted message string
        """
        if self.dspy_integration:
            try:
                return self.dspy_integration.get_srm_prompt(
                    experiment_name=exp_name,
                    chi_squared=chi_squared,
                    p_value=p_value,
                    expected_ratio=expected_ratio,
                    actual_counts=actual_counts,
                )
            except Exception as e:
                logger.debug(f"DSPy prompt failed, using fallback: {e}")

        # Fallback to hardcoded message
        return f"Sample Ratio Mismatch detected in '{exp_name}' (p={p_value:.6f})"

    def _generate_srm_alerts(
        self, state: ExperimentMonitorState
    ) -> List[MonitorAlert]:
        """Generate alerts for SRM issues.

        Args:
            state: Current agent state

        Returns:
            List of SRM alerts
        """
        alerts: List[MonitorAlert] = []
        srm_issues = state.get("srm_issues", [])

        # Get experiment names for alerts
        experiments = {e["experiment_id"]: e["name"] for e in state.get("experiments", [])}

        for issue in srm_issues:
            if not issue.get("detected"):
                continue

            exp_id = issue["experiment_id"]
            exp_name = experiments.get(exp_id, "Unknown Experiment")

            # Get message (uses DSPy prompt if available)
            message = self._get_srm_message(
                exp_name=exp_name,
                chi_squared=issue["chi_squared"],
                p_value=issue["p_value"],
                expected_ratio=str(issue["expected_ratio"]),
                actual_counts=str(issue["actual_counts"]),
            )

            alert = MonitorAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="srm",
                severity=issue.get("severity", "warning"),
                experiment_id=exp_id,
                experiment_name=exp_name,
                message=message,
                details={
                    "p_value": issue["p_value"],
                    "chi_squared": issue["chi_squared"],
                    "expected_ratio": issue["expected_ratio"],
                    "actual_counts": issue["actual_counts"],
                },
                recommended_action="Investigate randomization process and data collection. "
                "SRM may indicate bugs in the experiment setup or data pipeline.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            alerts.append(alert)

        return alerts

    def _get_enrollment_message(
        self,
        exp_name: str,
        current_rate: float,
        expected_rate: float,
        days_below_threshold: int,
    ) -> str:
        """Get enrollment alert message, using DSPy prompt if available.

        Args:
            exp_name: Experiment name
            current_rate: Current enrollment rate
            expected_rate: Expected enrollment rate
            days_below_threshold: Days below threshold

        Returns:
            Formatted message string
        """
        if self.dspy_integration:
            try:
                return self.dspy_integration.get_enrollment_prompt(
                    experiment_name=exp_name,
                    current_rate=current_rate,
                    expected_rate=expected_rate,
                    days_below_threshold=days_below_threshold,
                )
            except Exception as e:
                logger.debug(f"DSPy prompt failed, using fallback: {e}")

        # Fallback to hardcoded message
        return (
            f"Low enrollment rate in '{exp_name}': "
            f"{current_rate:.1f}/day (expected: {expected_rate:.1f}/day)"
        )

    def _generate_enrollment_alerts(
        self, state: ExperimentMonitorState
    ) -> List[MonitorAlert]:
        """Generate alerts for enrollment issues.

        Args:
            state: Current agent state

        Returns:
            List of enrollment alerts
        """
        alerts: List[MonitorAlert] = []
        enrollment_issues = state.get("enrollment_issues", [])

        # Get experiment names
        experiments = {e["experiment_id"]: e["name"] for e in state.get("experiments", [])}

        for issue in enrollment_issues:
            exp_id = issue["experiment_id"]
            exp_name = experiments.get(exp_id, "Unknown Experiment")

            # Get message (uses DSPy prompt if available)
            message = self._get_enrollment_message(
                exp_name=exp_name,
                current_rate=issue["current_rate"],
                expected_rate=issue["expected_rate"],
                days_below_threshold=issue["days_below_threshold"],
            )

            alert = MonitorAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="enrollment",
                severity=issue.get("severity", "warning"),
                experiment_id=exp_id,
                experiment_name=exp_name,
                message=message,
                details={
                    "current_rate": issue["current_rate"],
                    "expected_rate": issue["expected_rate"],
                    "days_below_threshold": issue["days_below_threshold"],
                },
                recommended_action="Review experiment eligibility criteria and targeting. "
                "Consider expanding the target population or adjusting experiment timeline.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            alerts.append(alert)

        return alerts

    def _generate_stale_data_alerts(
        self, state: ExperimentMonitorState
    ) -> List[MonitorAlert]:
        """Generate alerts for stale data issues.

        Args:
            state: Current agent state

        Returns:
            List of stale data alerts
        """
        alerts: List[MonitorAlert] = []
        stale_data_issues = state.get("stale_data_issues", [])

        # Get experiment names
        experiments = {e["experiment_id"]: e["name"] for e in state.get("experiments", [])}

        for issue in stale_data_issues:
            exp_id = issue["experiment_id"]
            exp_name = experiments.get(exp_id, "Unknown Experiment")

            hours_since = issue["hours_since_update"]
            if hours_since >= 72:
                time_desc = f"{hours_since / 24:.1f} days"
            else:
                time_desc = f"{hours_since:.1f} hours"

            alert = MonitorAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="stale_data",
                severity=issue.get("severity", "warning"),
                experiment_id=exp_id,
                experiment_name=exp_name,
                message=f"Data staleness detected in '{exp_name}': "
                f"no new data for {time_desc} (threshold: {issue['threshold_hours']}h)",
                details={
                    "last_data_timestamp": issue["last_data_timestamp"],
                    "hours_since_update": issue["hours_since_update"],
                    "threshold_hours": issue["threshold_hours"],
                },
                recommended_action="Check data pipeline and enrollment sources. "
                "Verify experiment is still receiving traffic.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            alerts.append(alert)

        return alerts

    def _generate_interim_alerts(
        self, state: ExperimentMonitorState
    ) -> List[MonitorAlert]:
        """Generate alerts for interim analysis triggers.

        Args:
            state: Current agent state

        Returns:
            List of interim analysis alerts
        """
        alerts: List[MonitorAlert] = []
        interim_triggers = state.get("interim_triggers", [])

        # Get experiment names
        experiments = {e["experiment_id"]: e["name"] for e in state.get("experiments", [])}

        for trigger in interim_triggers:
            if not trigger.get("triggered"):
                continue

            exp_id = trigger["experiment_id"]
            exp_name = experiments.get(exp_id, "Unknown Experiment")

            alert = MonitorAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="interim_trigger",
                severity="info",
                experiment_id=exp_id,
                experiment_name=exp_name,
                message=f"Interim analysis #{trigger['analysis_number']} triggered for '{exp_name}' "
                f"at {trigger['milestone_reached']} enrollment",
                details={
                    "analysis_number": trigger["analysis_number"],
                    "information_fraction": trigger["information_fraction"],
                    "milestone": trigger["milestone_reached"],
                },
                recommended_action="Review interim analysis results and make stopping decision "
                "based on O'Brien-Fleming boundaries.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            alerts.append(alert)

        return alerts

    def _get_fidelity_message(
        self,
        exp_name: str,
        predicted_effect: float,
        actual_effect: float,
        prediction_error: float,
        calibration_needed: bool,
    ) -> str:
        """Get fidelity alert message, using DSPy prompt if available.

        Args:
            exp_name: Experiment name
            predicted_effect: Digital Twin predicted effect
            actual_effect: Actual observed effect
            prediction_error: Prediction error as decimal
            calibration_needed: Whether calibration is needed

        Returns:
            Formatted message string
        """
        if self.dspy_integration:
            try:
                return self.dspy_integration.get_fidelity_prompt(
                    experiment_name=exp_name,
                    predicted_effect=predicted_effect,
                    actual_effect=actual_effect,
                    prediction_error=prediction_error,
                    calibration_needed=calibration_needed,
                )
            except Exception as e:
                logger.debug(f"DSPy prompt failed, using fallback: {e}")

        # Fallback to hardcoded message
        if calibration_needed:
            return (
                f"Digital Twin calibration needed for '{exp_name}': "
                f"prediction error = {prediction_error:.2%}"
            )
        else:
            return (
                f"Digital Twin fidelity check for '{exp_name}': "
                f"prediction error = {prediction_error:.2%}"
            )

    def _generate_fidelity_alerts(
        self, state: ExperimentMonitorState
    ) -> List[MonitorAlert]:
        """Generate alerts for fidelity issues.

        Args:
            state: Current agent state

        Returns:
            List of fidelity alerts
        """
        alerts: List[MonitorAlert] = []
        fidelity_issues = state.get("fidelity_issues", [])

        # Get experiment names
        experiments = {e["experiment_id"]: e["name"] for e in state.get("experiments", [])}

        for issue in fidelity_issues:
            exp_id = issue["experiment_id"]
            exp_name = experiments.get(exp_id, "Unknown Experiment")

            calibration_needed = issue.get("calibration_needed", False)
            severity = "warning" if calibration_needed else "info"
            action = (
                "Recalibrate Digital Twin model using actual experiment data."
                if calibration_needed
                else "No action required - prediction within acceptable range."
            )

            # Get message (uses DSPy prompt if available)
            message = self._get_fidelity_message(
                exp_name=exp_name,
                predicted_effect=issue["predicted_effect"],
                actual_effect=issue["actual_effect"],
                prediction_error=issue["prediction_error"],
                calibration_needed=calibration_needed,
            )

            alert = MonitorAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="fidelity",
                severity=severity,
                experiment_id=exp_id,
                experiment_name=exp_name,
                message=message,
                details={
                    "predicted_effect": issue["predicted_effect"],
                    "actual_effect": issue["actual_effect"],
                    "prediction_error": issue["prediction_error"],
                    "calibration_needed": issue["calibration_needed"],
                },
                recommended_action=action,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            alerts.append(alert)

        return alerts

    def _create_summary(
        self, state: ExperimentMonitorState, alerts: List[MonitorAlert]
    ) -> str:
        """Create monitoring summary.

        Args:
            state: Current agent state
            alerts: Generated alerts

        Returns:
            Summary string
        """
        experiments_checked = state.get("experiments_checked", 0)
        experiments = state.get("experiments", [])

        # Count by health status
        health_counts = {"healthy": 0, "warning": 0, "critical": 0, "unknown": 0}
        for exp in experiments:
            status = exp.get("health_status", "unknown")
            health_counts[status] = health_counts.get(status, 0) + 1

        # Count alerts by severity
        alert_counts = {"critical": 0, "warning": 0, "info": 0}
        for alert in alerts:
            severity = alert.get("severity", "info")
            alert_counts[severity] = alert_counts.get(severity, 0) + 1

        # Build summary
        parts = [
            f"Experiment Monitor Summary",
            f"Experiments checked: {experiments_checked}",
            f"Health status: {health_counts['healthy']} healthy, "
            f"{health_counts['warning']} warning, {health_counts['critical']} critical",
            f"Alerts: {alert_counts['critical']} critical, "
            f"{alert_counts['warning']} warning, {alert_counts['info']} info",
        ]

        # Add SRM summary
        srm_issues = state.get("srm_issues", [])
        if srm_issues:
            parts.append(f"SRM issues detected: {len(srm_issues)}")

        # Add enrollment summary
        enrollment_issues = state.get("enrollment_issues", [])
        if enrollment_issues:
            parts.append(f"Enrollment issues: {len(enrollment_issues)}")

        # Add stale data summary
        stale_data_issues = state.get("stale_data_issues", [])
        if stale_data_issues:
            parts.append(f"Stale data issues: {len(stale_data_issues)}")

        # Add interim triggers
        interim_triggers = state.get("interim_triggers", [])
        if interim_triggers:
            parts.append(f"Interim analyses triggered: {len(interim_triggers)}")

        return "\n".join(parts)

    def _generate_recommendations(
        self, state: ExperimentMonitorState, alerts: List[MonitorAlert]
    ) -> List[str]:
        """Generate recommended actions.

        Args:
            state: Current agent state
            alerts: Generated alerts

        Returns:
            List of recommended actions
        """
        recommendations: List[str] = []

        # Critical SRM issues
        critical_srm = [a for a in alerts if a["alert_type"] == "srm" and a["severity"] == "critical"]
        if critical_srm:
            recommendations.append(
                f"URGENT: {len(critical_srm)} experiments have critical SRM issues - "
                "investigate immediately and consider pausing affected experiments"
            )

        # Critical enrollment issues
        critical_enrollment = [
            a for a in alerts if a["alert_type"] == "enrollment" and a["severity"] == "critical"
        ]
        if critical_enrollment:
            recommendations.append(
                f"{len(critical_enrollment)} experiments have critically low enrollment - "
                "review experiment design and targeting"
            )

        # Stale data issues
        stale_data_alerts = [a for a in alerts if a["alert_type"] == "stale_data"]
        critical_stale = [a for a in stale_data_alerts if a["severity"] == "critical"]
        if critical_stale:
            recommendations.append(
                f"URGENT: {len(critical_stale)} experiments have critically stale data - "
                "check data pipelines immediately"
            )
        elif stale_data_alerts:
            recommendations.append(
                f"{len(stale_data_alerts)} experiments have stale data - "
                "verify data pipelines are operational"
            )

        # Interim analyses to review
        interim_alerts = [a for a in alerts if a["alert_type"] == "interim_trigger"]
        if interim_alerts:
            recommendations.append(
                f"{len(interim_alerts)} experiments ready for interim analysis - "
                "review results and make stopping decisions"
            )

        # Fidelity calibration needed
        fidelity_warnings = [
            a for a in alerts if a["alert_type"] == "fidelity" and a["severity"] == "warning"
        ]
        if fidelity_warnings:
            recommendations.append(
                f"Digital Twin calibration recommended for {len(fidelity_warnings)} experiments"
            )

        # General health check
        experiments = state.get("experiments", [])
        healthy_count = sum(1 for e in experiments if e.get("health_status") == "healthy")
        if healthy_count == len(experiments) and not alerts:
            recommendations.append("All experiments are running healthily - no action required")

        return recommendations
