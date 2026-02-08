"""SRM Detector Node.

This node detects Sample Ratio Mismatch (SRM) in experiments by comparing
the actual allocation ratio to the expected ratio using chi-squared tests.

SRM indicates potential issues with:
- Randomization bugs
- Data collection issues
- Sample pollution

Performance Target: <1s per experiment
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from scipy import stats

from src.agents.experiment_monitor.state import (
    ErrorDetails,
    ExperimentMonitorState,
    ExperimentSummary,
    SRMIssue,
)


class SRMDetectorNode:
    """Detects Sample Ratio Mismatch in experiments.

    Detection Strategy:
    1. For each experiment, get actual variant counts
    2. Compare with expected allocation ratio
    3. Perform chi-squared test
    4. Flag experiments with significant mismatch (p < threshold)

    Performance Target: <1s per experiment
    """

    def __init__(self):
        """Initialize SRM detector node."""
        self._client = None

    async def _get_client(self):
        """Lazy load Supabase client."""
        if self._client is None:
            from src.memory.services.factories import get_supabase_client

            self._client = await get_supabase_client()
        return self._client

    async def execute(self, state: ExperimentMonitorState) -> ExperimentMonitorState:
        """Execute SRM detection on experiments.

        Args:
            state: Current agent state with experiments to check

        Returns:
            Updated state with SRM detection results
        """
        start_time = time.time()

        try:
            # Get experiments to check
            experiments = state.get("experiments", [])

            if not experiments:
                state["srm_issues"] = []
                return state

            # Get client
            client = await self._get_client()

            # Check each experiment for SRM
            srm_issues: List[SRMIssue] = []

            for exp in experiments:
                issue = await self._check_srm(exp, client, state)
                if issue:
                    srm_issues.append(issue)

            # Update state
            state["srm_issues"] = srm_issues

            # Update latency
            latency_ms = int((time.time() - start_time) * 1000)
            state["check_latency_ms"] = state.get("check_latency_ms", 0) + latency_ms

        except Exception as e:
            error: ErrorDetails = {
                "node": "srm_detector",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["errors"] = state.get("errors", []) + [error]
            state["srm_issues"] = []

        return state

    async def _check_srm(
        self,
        experiment: ExperimentSummary,
        client: Optional[Any],
        state: ExperimentMonitorState,
    ) -> Optional[SRMIssue]:
        """Check a single experiment for SRM.

        Args:
            experiment: Experiment summary
            client: Optional Supabase client
            state: Current state with thresholds

        Returns:
            SRMIssue if mismatch detected, None otherwise
        """
        exp_id = experiment["experiment_id"]
        total_enrolled = experiment.get("total_enrolled", 0)

        # Need minimum sample size for reliable SRM detection
        min_sample = 100
        if total_enrolled < min_sample:
            return None

        # Get expected ratio (default 50/50 split)
        expected_ratio = {"control": 0.5, "treatment": 0.5}

        # Get actual counts
        if client:
            actual_counts = await self._get_variant_counts(client, exp_id)
        else:
            # Mock data for testing
            actual_counts = {"control": 48, "treatment": 52}

        if not actual_counts or sum(actual_counts.values()) == 0:
            return None

        # Perform chi-squared test
        chi_squared, p_value = self._chi_squared_test(expected_ratio, actual_counts)

        # Check against threshold
        threshold = state.get("srm_threshold", 0.001)
        detected = p_value < threshold

        # Determine severity
        if detected:
            if p_value < 0.0001:
                severity = "critical"
            elif p_value < 0.001:
                severity = "warning"
            else:
                severity = "info"

            return SRMIssue(
                experiment_id=exp_id,
                detected=True,
                p_value=round(p_value, 6),
                chi_squared=round(chi_squared, 4),
                expected_ratio=expected_ratio,
                actual_counts=actual_counts,
                severity=severity,  # type: ignore
            )

        return None

    async def _get_variant_counts(self, client: Any, experiment_id: str) -> Dict[str, int]:
        """Get counts by variant for an experiment.

        Args:
            client: Supabase client
            experiment_id: Experiment ID

        Returns:
            Dictionary mapping variant names to counts
        """
        try:
            result = await (
                client.table("ab_experiment_assignments")
                .select("variant")
                .eq("experiment_id", experiment_id)
                .execute()
            )

            if not result.data:
                return {}

            # Count by variant
            counts: Dict[str, int] = {}
            for row in result.data:
                variant = row.get("variant", "unknown")
                counts[variant] = counts.get(variant, 0) + 1

            return counts

        except Exception:
            return {}

    def _chi_squared_test(
        self, expected_ratio: Dict[str, float], actual_counts: Dict[str, int]
    ) -> tuple:
        """Perform chi-squared test for SRM.

        Args:
            expected_ratio: Expected allocation ratio
            actual_counts: Actual counts by variant

        Returns:
            Tuple of (chi_squared_statistic, p_value)
        """
        total = sum(actual_counts.values())
        if total == 0:
            return 0.0, 1.0

        # Calculate expected counts
        variants = list(actual_counts.keys())
        observed = [actual_counts.get(v, 0) for v in variants]
        expected = [expected_ratio.get(v, 1 / len(variants)) * total for v in variants]

        # Chi-squared test
        try:
            chi2, p_value = stats.chisquare(observed, f_exp=expected)
            return float(chi2), float(p_value)
        except Exception:
            return 0.0, 1.0
