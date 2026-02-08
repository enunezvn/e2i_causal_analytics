"""Health Checker Node.

This node checks the health status of active experiments including:
1. Enrollment rates and trends
2. Data quality and freshness
3. Overall experiment health status

Performance Target: <2s per experiment
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from src.agents.experiment_monitor.state import (
    EnrollmentIssue,
    ErrorDetails,
    ExperimentMonitorState,
    ExperimentSummary,
    StaleDataIssue,
)


class HealthCheckerNode:
    """Checks experiment health and enrollment rates.

    Health Check Strategy:
    1. Query active experiments from database
    2. Calculate enrollment rates and trends
    3. Identify experiments with health issues
    4. Update state with experiment summaries and issues

    Performance Target: <2s per experiment
    """

    def __init__(self):
        """Initialize health checker node."""
        self._client = None

    async def _get_client(self):
        """Lazy load Supabase client."""
        if self._client is None:
            from src.memory.services.factories import get_supabase_client

            self._client = await get_supabase_client()
        return self._client

    async def execute(self, state: ExperimentMonitorState) -> ExperimentMonitorState:
        """Execute health check on experiments.

        Args:
            state: Current agent state

        Returns:
            Updated state with experiment health information
        """
        start_time = time.time()

        try:
            state["status"] = "checking"

            # Get client
            client = await self._get_client()
            if not client:
                state["warnings"] = state.get("warnings", []) + [
                    "No database client available, using mock data"
                ]
                # Use mock data for testing
                experiments = await self._get_mock_experiments(state)
            else:
                experiments = await self._get_experiments(client, state)

            # Process each experiment
            experiment_summaries: List[ExperimentSummary] = []
            enrollment_issues: List[EnrollmentIssue] = []
            stale_data_issues: List[StaleDataIssue] = []

            for exp in experiments:
                summary = await self._check_experiment_health(exp, client)
                experiment_summaries.append(summary)

                # Check for enrollment issues
                issue = self._check_enrollment_rate(exp, summary, state)
                if issue:
                    enrollment_issues.append(issue)

                # Check for stale data
                stale_issue = await self._check_stale_data(exp, client, state)
                if stale_issue:
                    stale_data_issues.append(stale_issue)

            # Update state
            state["experiments"] = experiment_summaries
            state["enrollment_issues"] = enrollment_issues
            state["stale_data_issues"] = stale_data_issues
            state["experiments_checked"] = len(experiment_summaries)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            state["check_latency_ms"] = latency_ms

        except Exception as e:
            error: ErrorDetails = {
                "node": "health_checker",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["errors"] = state.get("errors", []) + [error]
            state["experiments"] = []
            state["enrollment_issues"] = []
            state["stale_data_issues"] = []

        return state

    async def _get_experiments(self, client: Any, state: ExperimentMonitorState) -> List[Dict]:
        """Get experiments to check from database.

        Args:
            client: Supabase client
            state: Current state with filter criteria

        Returns:
            List of experiment dictionaries
        """
        try:
            if state.get("check_all_active"):
                # Get all active experiments
                result = await (
                    client.table("ml_experiments")
                    .select("id, name, status, config, created_at")
                    .eq("status", "running")
                    .execute()
                )
            elif state.get("experiment_ids"):
                # Get specific experiments
                result = await (
                    client.table("ml_experiments")
                    .select("id, name, status, config, created_at")
                    .in_("id", state["experiment_ids"])
                    .execute()
                )
            else:
                return []

            return result.data if result.data else []

        except Exception:
            return []

    async def _get_mock_experiments(self, state: ExperimentMonitorState) -> List[Dict]:
        """Get mock experiments for testing.

        Args:
            state: Current state

        Returns:
            List of mock experiment dictionaries
        """
        now = datetime.now(timezone.utc)
        return [
            {
                "id": "exp-001",
                "name": "Test Experiment 1",
                "status": "running",
                "config": {
                    "target_sample_size": 1000,
                    "allocation_ratio": {"control": 0.5, "treatment": 0.5},
                },
                "created_at": (now - timedelta(days=14)).isoformat(),
            },
            {
                "id": "exp-002",
                "name": "Test Experiment 2",
                "status": "running",
                "config": {
                    "target_sample_size": 500,
                    "allocation_ratio": {"control": 0.5, "treatment": 0.5},
                },
                "created_at": (now - timedelta(days=7)).isoformat(),
            },
        ]

    async def _check_experiment_health(
        self, experiment: Dict, client: Optional[Any]
    ) -> ExperimentSummary:
        """Check health of a single experiment.

        Args:
            experiment: Experiment dictionary
            client: Optional Supabase client

        Returns:
            ExperimentSummary with health status
        """
        exp_id = experiment["id"]
        config = experiment.get("config", {})
        created_at = experiment.get("created_at", datetime.now(timezone.utc).isoformat())

        # Calculate days running
        if isinstance(created_at, str):
            start_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        else:
            start_date = created_at
        days_running = max(1, (datetime.now(timezone.utc) - start_date).days)

        # Get enrollment data
        total_enrolled = 0
        if client:
            try:
                result = await (
                    client.table("ab_experiment_assignments")
                    .select("id", count="exact")
                    .eq("experiment_id", exp_id)
                    .execute()
                )
                total_enrolled = result.count or 0
            except Exception:
                pass

        # Calculate metrics
        enrollment_rate = total_enrolled / days_running if days_running > 0 else 0
        target_sample_size = config.get("target_sample_size", 1000)
        information_fraction = total_enrolled / target_sample_size if target_sample_size > 0 else 0

        # Determine health status
        health_status = self._determine_health_status(
            enrollment_rate, information_fraction, days_running
        )

        return ExperimentSummary(
            experiment_id=exp_id,
            name=experiment.get("name", "Unknown"),
            status=experiment.get("status", "unknown"),
            health_status=health_status,
            days_running=days_running,
            total_enrolled=total_enrolled,
            enrollment_rate=round(enrollment_rate, 2),
            current_information_fraction=round(information_fraction, 4),
        )

    def _determine_health_status(
        self, enrollment_rate: float, information_fraction: float, days_running: int
    ) -> Literal["healthy", "warning", "critical", "unknown"]:
        """Determine overall health status.

        Args:
            enrollment_rate: Daily enrollment rate
            information_fraction: Fraction of target sample enrolled
            days_running: Days since experiment start

        Returns:
            Health status string
        """
        # Critical: Very low enrollment after significant time
        if days_running >= 14 and enrollment_rate < 2:
            return "critical"

        # Warning: Below expected enrollment
        if days_running >= 7 and enrollment_rate < 5:
            return "warning"

        # Warning: Behind schedule
        expected_fraction = days_running / 30  # Assuming 30-day experiments
        if information_fraction < expected_fraction * 0.5:
            return "warning"

        return "healthy"

    def _check_enrollment_rate(
        self,
        experiment: Dict,
        summary: ExperimentSummary,
        state: ExperimentMonitorState,
    ) -> Optional[EnrollmentIssue]:
        """Check if enrollment rate is below threshold.

        Args:
            experiment: Experiment dictionary
            summary: Experiment summary
            state: Current state with thresholds

        Returns:
            EnrollmentIssue if rate is below threshold, None otherwise
        """
        threshold = state.get("enrollment_threshold", 5.0)

        if summary["enrollment_rate"] < threshold:
            # Calculate severity based on how long below threshold
            days = summary["days_running"]
            if days >= 14:
                severity = "critical"
            elif days >= 7:
                severity = "warning"
            else:
                severity = "info"

            return EnrollmentIssue(
                experiment_id=summary["experiment_id"],
                current_rate=summary["enrollment_rate"],
                expected_rate=threshold,
                days_below_threshold=days,
                severity=severity,  # type: ignore
            )

        return None

    async def _check_stale_data(
        self,
        experiment: Dict,
        client: Optional[Any],
        state: ExperimentMonitorState,
    ) -> Optional[StaleDataIssue]:
        """Check if experiment data is stale.

        Args:
            experiment: Experiment dictionary
            client: Optional Supabase client
            state: Current state with thresholds

        Returns:
            StaleDataIssue if data is stale, None otherwise
        """
        threshold_hours = state.get("stale_data_threshold_hours", 24.0)
        exp_id = experiment["id"]

        if not client:
            return None

        try:
            # Get the most recent assignment timestamp for this experiment
            result = await (
                client.table("ab_experiment_assignments")
                .select("assigned_at")
                .eq("experiment_id", exp_id)
                .order("assigned_at", desc=True)
                .limit(1)
                .execute()
            )

            if not result.data:
                # No assignments yet - might be stale or just new
                # Check experiment created_at to determine
                created_at = experiment.get("created_at")
                if created_at:
                    if isinstance(created_at, str):
                        created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    else:
                        created_time = created_at

                    hours_since_creation = (
                        datetime.now(timezone.utc) - created_time
                    ).total_seconds() / 3600

                    # If experiment is older than threshold and no data, it's stale
                    if hours_since_creation > threshold_hours:
                        return StaleDataIssue(
                            experiment_id=exp_id,
                            last_data_timestamp="N/A - No assignments",
                            hours_since_update=hours_since_creation,
                            threshold_hours=threshold_hours,
                            severity="warning" if hours_since_creation < 48 else "critical",
                        )
                return None

            # Get the last assignment timestamp
            last_timestamp_str = result.data[0]["assigned_at"]
            if isinstance(last_timestamp_str, str):
                last_timestamp = datetime.fromisoformat(last_timestamp_str.replace("Z", "+00:00"))
            else:
                last_timestamp = last_timestamp_str

            # Calculate hours since last update
            hours_since_update = (
                datetime.now(timezone.utc) - last_timestamp
            ).total_seconds() / 3600

            if hours_since_update > threshold_hours:
                # Determine severity based on staleness
                if hours_since_update > 72:  # 3 days
                    severity = "critical"
                elif hours_since_update > 48:  # 2 days
                    severity = "warning"
                else:
                    severity = "info"

                return StaleDataIssue(
                    experiment_id=exp_id,
                    last_data_timestamp=last_timestamp.isoformat(),
                    hours_since_update=round(hours_since_update, 2),
                    threshold_hours=threshold_hours,
                    severity=severity,  # type: ignore
                )

        except Exception:
            # Don't fail the whole check if stale data detection fails
            pass

        return None
