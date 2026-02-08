"""Interim Analyzer Node.

This node checks if experiments have reached analysis milestones
and triggers interim analyses when appropriate.

Milestones are typically at 25%, 50%, and 75% of target sample size.

Performance Target: <1s per experiment
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.agents.experiment_monitor.state import (
    ErrorDetails,
    ExperimentMonitorState,
    ExperimentSummary,
    InterimTrigger,
)


class InterimAnalyzerNode:
    """Checks for interim analysis triggers.

    Analysis Strategy:
    1. For each experiment, calculate information fraction
    2. Check against milestone schedule (25%, 50%, 75%)
    3. Verify milestone hasn't already been analyzed
    4. Trigger interim analysis if new milestone reached

    Performance Target: <1s per experiment
    """

    # Default milestones (information fractions)
    DEFAULT_MILESTONES = [0.25, 0.50, 0.75]

    def __init__(self):
        """Initialize interim analyzer node."""
        self._client = None

    async def _get_client(self):
        """Lazy load Supabase client."""
        if self._client is None:
            from src.memory.services.factories import get_supabase_client

            self._client = await get_supabase_client()
        return self._client

    async def execute(self, state: ExperimentMonitorState) -> ExperimentMonitorState:
        """Execute interim analysis trigger check.

        Args:
            state: Current agent state with experiments to check

        Returns:
            Updated state with interim triggers
        """
        start_time = time.time()

        # Skip if not configured to check interim
        if not state.get("check_interim", True):
            state["interim_triggers"] = []
            return state

        try:
            # Get experiments to check
            experiments = state.get("experiments", [])

            if not experiments:
                state["interim_triggers"] = []
                return state

            # Get client
            client = await self._get_client()

            # Check each experiment for interim triggers
            interim_triggers: List[InterimTrigger] = []

            for exp in experiments:
                trigger = await self._check_interim_trigger(exp, client)
                if trigger and trigger.get("triggered"):
                    interim_triggers.append(trigger)

            # Update state
            state["interim_triggers"] = interim_triggers

            # Update latency
            latency_ms = int((time.time() - start_time) * 1000)
            state["check_latency_ms"] = state.get("check_latency_ms", 0) + latency_ms

        except Exception as e:
            error: ErrorDetails = {
                "node": "interim_analyzer",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["errors"] = state.get("errors", []) + [error]
            state["interim_triggers"] = []

        return state

    async def _check_interim_trigger(
        self, experiment: ExperimentSummary, client: Optional[Any]
    ) -> Optional[InterimTrigger]:
        """Check if an experiment has reached an interim analysis milestone.

        Args:
            experiment: Experiment summary
            client: Optional Supabase client

        Returns:
            InterimTrigger if milestone reached, None otherwise
        """
        exp_id = experiment["experiment_id"]
        info_fraction = experiment.get("current_information_fraction", 0)

        # Find the highest milestone reached
        current_milestone = None
        for milestone in self.DEFAULT_MILESTONES:
            if info_fraction >= milestone:
                current_milestone = milestone

        if current_milestone is None:
            return None

        # Check if this milestone has already been analyzed
        if client:
            already_analyzed = await self._check_milestone_analyzed(
                client, exp_id, current_milestone
            )
        else:
            # Mock: assume not analyzed
            already_analyzed = False

        if already_analyzed:
            return None

        # Calculate analysis number
        analysis_number = self.DEFAULT_MILESTONES.index(current_milestone) + 1

        return InterimTrigger(
            experiment_id=exp_id,
            analysis_number=analysis_number,
            information_fraction=round(info_fraction, 4),
            milestone_reached=f"{int(current_milestone * 100)}%",
            triggered=True,
        )

    async def _check_milestone_analyzed(
        self, client: Any, experiment_id: str, milestone: float
    ) -> bool:
        """Check if a milestone has already been analyzed.

        Args:
            client: Supabase client
            experiment_id: Experiment ID
            milestone: Milestone fraction to check

        Returns:
            True if milestone already analyzed
        """
        try:
            # Query interim analyses for this experiment
            result = await (
                client.table("ab_interim_analyses")
                .select("information_fraction")
                .eq("experiment_id", experiment_id)
                .execute()
            )

            if not result.data:
                return False

            # Check if any analysis is close to this milestone (within 5%)
            tolerance = 0.05
            for analysis in result.data:
                if abs(analysis.get("information_fraction", 0) - milestone) < tolerance:
                    return True

            return False

        except Exception:
            return False
