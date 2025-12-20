"""
E2I Health Score Agent - Score Composer Node
Version: 4.2
Purpose: Compose overall health score from component scores
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from ..state import HealthScoreState
from ..metrics import DEFAULT_WEIGHTS, DEFAULT_GRADES, GradeThresholds, ScoreWeights

logger = logging.getLogger(__name__)


class ScoreComposerNode:
    """
    Compose overall health score from component scores.
    Pure computation - no LLM calls.
    """

    def __init__(
        self,
        weights: Optional[ScoreWeights] = None,
        grades: Optional[GradeThresholds] = None,
    ):
        """
        Initialize score composer.

        Args:
            weights: Custom weights for score components
            grades: Custom grade thresholds
        """
        self.weights = weights or DEFAULT_WEIGHTS
        self.grades = grades or DEFAULT_GRADES

    async def execute(self, state: HealthScoreState) -> HealthScoreState:
        """Compose overall health score."""
        start_time = time.time()

        try:
            # Collect scores (default to 1.0 if not present)
            scores = {
                "component": state.get("component_health_score", 1.0),
                "model": state.get("model_health_score", 1.0),
                "pipeline": state.get("pipeline_health_score", 1.0),
                "agent": state.get("agent_health_score", 1.0),
            }

            # Calculate weighted average
            weights_dict = self.weights.to_dict()
            overall_score = sum(
                scores[dim] * weight for dim, weight in weights_dict.items()
            )

            # Convert to 0-100 scale
            overall_score_100 = overall_score * 100

            # Determine grade
            grade = self.grades.get_grade(overall_score)

            # Identify issues
            critical_issues, warnings = self._identify_issues(state)

            # Generate summary
            summary = self._generate_summary(overall_score_100, grade, critical_issues)

            check_time = state.get("check_latency_ms", 0) + int(
                (time.time() - start_time) * 1000
            )

            logger.info(
                f"Score composition complete: score={overall_score_100:.1f}, "
                f"grade={grade}, issues={len(critical_issues)}, warnings={len(warnings)}"
            )

            return {
                **state,
                "overall_health_score": overall_score_100,
                "health_grade": grade,
                "critical_issues": critical_issues,
                "warnings": warnings,
                "health_summary": summary,
                "check_latency_ms": check_time,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Score composition failed: {e}")
            return {
                **state,
                "errors": [{"node": "score_composer", "error": str(e)}],
                "overall_health_score": 0.0,
                "health_grade": "F",
                "critical_issues": [f"Score composition failed: {e}"],
                "warnings": [],
                "health_summary": "Unable to compute health score due to an error.",
                "status": "failed",
            }

    def _identify_issues(
        self, state: HealthScoreState
    ) -> Tuple[List[str], List[str]]:
        """Identify critical issues and warnings."""
        critical = []
        warnings = []

        # Check components
        for comp in state.get("component_statuses") or []:
            if comp["status"] == "unhealthy":
                critical.append(f"Component '{comp['component_name']}' is unhealthy")
            elif comp["status"] == "degraded":
                warnings.append(f"Component '{comp['component_name']}' is degraded")
            elif comp["status"] == "unknown":
                warnings.append(
                    f"Component '{comp['component_name']}' status is unknown"
                )

        # Check models
        for model in state.get("model_metrics") or []:
            if model["status"] == "unhealthy":
                critical.append(f"Model '{model['model_id']}' is unhealthy")
            elif model["status"] == "degraded":
                warnings.append(f"Model '{model['model_id']}' is degraded")

        # Check pipelines
        for pipeline in state.get("pipeline_statuses") or []:
            if pipeline["status"] == "failed":
                critical.append(
                    f"Pipeline '{pipeline['pipeline_name']}' has failed"
                )
            elif pipeline["status"] == "stale":
                warnings.append(
                    f"Pipeline '{pipeline['pipeline_name']}' data is stale"
                )

        # Check agents
        for agent in state.get("agent_statuses") or []:
            if not agent["available"]:
                critical.append(f"Agent '{agent['agent_name']}' is unavailable")
            elif agent["success_rate"] < 0.9:
                warnings.append(
                    f"Agent '{agent['agent_name']}' has low success rate "
                    f"({agent['success_rate']:.1%})"
                )

        # Check accumulated errors
        for error in state.get("errors") or []:
            node = error.get("node", "unknown")
            msg = error.get("error", "Unknown error")
            warnings.append(f"Error in {node}: {msg}")

        return critical, warnings

    def _generate_summary(
        self, score: float, grade: str, issues: List[str]
    ) -> str:
        """Generate health summary."""
        status_map = {
            "A": "excellent",
            "B": "good",
            "C": "fair",
            "D": "poor",
            "F": "critical",
        }
        status = status_map.get(grade, "unknown")

        summary = f"System health is {status} (Grade: {grade}, Score: {score:.1f}/100)."

        if issues:
            summary += f" {len(issues)} critical issue(s) detected."
        else:
            summary += " All systems operational."

        return summary
