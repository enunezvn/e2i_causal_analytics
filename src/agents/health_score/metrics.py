"""
E2I Health Score Agent - Metrics and Thresholds
Version: 4.2
Purpose: Define metric thresholds and health criteria
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class HealthThresholds:
    """Thresholds for determining health status"""

    # Model health thresholds
    min_accuracy: float = 0.7
    min_auc: float = 0.65
    max_error_rate: float = 0.05
    max_latency_p99_ms: int = 1000
    min_predictions_24h: int = 100

    # Pipeline health thresholds
    max_freshness_hours: float = 24.0
    stale_threshold_hours: float = 12.0
    min_rows_processed: int = 0

    # Agent health thresholds
    min_success_rate: float = 0.9
    max_avg_latency_ms: int = 5000

    # Component health thresholds
    health_check_timeout_ms: int = 2000
    max_component_latency_ms: int = 500


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for calculating composite health score"""

    component: float = 0.30
    model: float = 0.30
    pipeline: float = 0.25
    agent: float = 0.15

    def to_dict(self) -> Dict[str, float]:
        return {
            "component": self.component,
            "model": self.model,
            "pipeline": self.pipeline,
            "agent": self.agent,
        }


@dataclass(frozen=True)
class GradeThresholds:
    """Thresholds for letter grades"""

    A: float = 0.9
    B: float = 0.8
    C: float = 0.7
    D: float = 0.6
    F: float = 0.0

    def get_grade(self, score: float) -> str:
        """Get letter grade for a score (0-1 scale)"""
        if score >= self.A:
            return "A"
        elif score >= self.B:
            return "B"
        elif score >= self.C:
            return "C"
        elif score >= self.D:
            return "D"
        return "F"


# Default instances
DEFAULT_THRESHOLDS = HealthThresholds()
DEFAULT_WEIGHTS = ScoreWeights()
DEFAULT_GRADES = GradeThresholds()
