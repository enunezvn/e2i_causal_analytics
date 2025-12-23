"""Fidelity Checker Node.

This node checks Digital Twin simulation fidelity by comparing
predicted effects with actual experiment results.

Performance Target: <1s per experiment
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.agents.experiment_monitor.state import (
    ErrorDetails,
    ExperimentMonitorState,
    FidelityIssue,
)


class FidelityCheckerNode:
    """Checks Digital Twin fidelity against actual experiment results.

    Fidelity Check Strategy:
    1. Query twin_fidelity_tracking for completed simulations
    2. Compare predicted effects with actual effects
    3. Identify experiments where prediction error exceeds threshold
    4. Flag experiments needing recalibration

    Performance Target: <1s per experiment
    """

    def __init__(self):
        """Initialize fidelity checker node."""
        self._client = None

    async def _get_client(self):
        """Lazy load Supabase client."""
        if self._client is None:
            from src.memory.services.factories import get_supabase_client

            self._client = await get_supabase_client()
        return self._client

    async def execute(self, state: ExperimentMonitorState) -> ExperimentMonitorState:
        """Execute fidelity checks on experiments.

        Args:
            state: Current agent state

        Returns:
            Updated state with fidelity issues
        """
        start_time = time.time()

        try:
            # Get client
            client = await self._get_client()
            if not client:
                state["warnings"] = state.get("warnings", []) + [
                    "No database client available for fidelity checks"
                ]
                state["fidelity_issues"] = []
                return state

            # Get experiments from state
            experiments = state.get("experiments", [])
            fidelity_threshold = state.get("fidelity_threshold", 0.2)

            fidelity_issues: List[FidelityIssue] = []

            for exp in experiments:
                exp_id = exp["experiment_id"]
                issue = await self._check_fidelity(
                    exp_id, client, fidelity_threshold
                )
                if issue:
                    fidelity_issues.append(issue)

            # Update state
            state["fidelity_issues"] = fidelity_issues

            # Update latency
            latency_ms = int((time.time() - start_time) * 1000)
            state["check_latency_ms"] = state.get("check_latency_ms", 0) + latency_ms

        except Exception as e:
            error: ErrorDetails = {
                "node": "fidelity_checker",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["errors"] = state.get("errors", []) + [error]
            state["fidelity_issues"] = []

        return state

    async def _check_fidelity(
        self,
        experiment_id: str,
        client: Any,
        threshold: float,
    ) -> Optional[FidelityIssue]:
        """Check fidelity for a single experiment.

        Args:
            experiment_id: The experiment ID to check
            client: Supabase client
            threshold: Maximum acceptable prediction error

        Returns:
            FidelityIssue if fidelity exceeds threshold, None otherwise
        """
        try:
            # Query twin_fidelity_tracking for this experiment
            result = await (
                client.table("twin_fidelity_tracking")
                .select(
                    "simulation_id, simulated_ate, actual_ate, "
                    "prediction_error, fidelity_grade"
                )
                .eq("actual_experiment_id", experiment_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )

            if not result.data:
                # No fidelity tracking for this experiment
                return None

            tracking = result.data[0]
            prediction_error = tracking.get("prediction_error", 0)

            # Check if prediction error exceeds threshold
            if abs(prediction_error) > threshold:
                # Determine severity based on error magnitude
                if abs(prediction_error) > threshold * 2:
                    severity = "warning"
                    calibration_needed = True
                else:
                    severity = "info"
                    calibration_needed = False

                return FidelityIssue(
                    experiment_id=experiment_id,
                    twin_simulation_id=str(tracking.get("simulation_id", "")),
                    predicted_effect=float(tracking.get("simulated_ate", 0)),
                    actual_effect=float(tracking.get("actual_ate", 0)),
                    prediction_error=float(prediction_error),
                    calibration_needed=calibration_needed,
                    severity=severity,  # type: ignore
                )

        except Exception:
            # Don't fail the whole check if fidelity check fails
            pass

        return None

    async def _get_fidelity_from_simulation_summary(
        self,
        experiment_id: str,
        client: Any,
        threshold: float,
    ) -> Optional[FidelityIssue]:
        """Alternative: Get fidelity from v_simulation_summary view.

        This is a fallback if twin_fidelity_tracking doesn't have data.

        Args:
            experiment_id: The experiment ID to check
            client: Supabase client
            threshold: Maximum acceptable prediction error

        Returns:
            FidelityIssue if fidelity exceeds threshold, None otherwise
        """
        try:
            # Query v_simulation_summary view
            result = await (
                client.table("v_simulation_summary")
                .select(
                    "simulation_id, experiment_design_id, simulated_ate, "
                    "prediction_error, fidelity_grade"
                )
                .eq("experiment_design_id", experiment_id)
                .order("simulation_end", desc=True)
                .limit(1)
                .execute()
            )

            if not result.data:
                return None

            summary = result.data[0]
            prediction_error = summary.get("prediction_error")

            if prediction_error is None:
                return None

            if abs(prediction_error) > threshold:
                return FidelityIssue(
                    experiment_id=experiment_id,
                    twin_simulation_id=str(summary.get("simulation_id", "")),
                    predicted_effect=float(summary.get("simulated_ate", 0)),
                    actual_effect=0.0,  # Not available in summary
                    prediction_error=float(prediction_error),
                    calibration_needed=abs(prediction_error) > threshold * 2,
                    severity="warning" if abs(prediction_error) > threshold * 2 else "info",
                )

        except Exception:
            pass

        return None
