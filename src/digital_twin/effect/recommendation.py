"""CI-based three-way pre-screen decision: DEPLOY / REFINE / SKIP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from src.digital_twin.effect.estimate import EffectEstimate


class Recommendation(str, Enum):
    DEPLOY = "deploy"
    REFINE = "refine"
    SKIP = "skip"


@dataclass
class PolicyThresholds:
    min_effect: float = 0.05  # calibrated (Task 9), not the old fake 0.05
    power: float = 0.80
    alpha: float = 0.05


class RecommendationPolicy:
    def __init__(self, thresholds: PolicyThresholds) -> None:
        self.t = thresholds

    def decide(
        self, estimate: EffectEstimate, baseline_rate: float
    ) -> tuple[Recommendation, str, int]:
        lo, hi, m = estimate.ate_ci_lower, estimate.ate_ci_upper, self.t.min_effect
        n = self._recommended_sample_size(estimate.ate, baseline_rate)
        if lo > m:
            return (
                Recommendation.DEPLOY,
                f"CI lower bound {lo:.3f} exceeds min effect {m:.3f}.",
                n,
            )
        if hi < m:
            return (
                Recommendation.SKIP,
                f"CI upper bound {hi:.3f} is below min effect {m:.3f}.",
                n,
            )
        return (
            Recommendation.REFINE,
            f"CI [{lo:.3f}, {hi:.3f}] straddles min effect {m:.3f}; refine or gather more data.",
            n,
        )

    def _recommended_sample_size(self, effect: float, baseline_rate: float) -> int:
        """Two-proportion sample size per arm at the configured power/alpha."""
        from scipy.stats import norm

        effect = abs(effect)
        if effect < 1e-6:
            return 0
        p1 = min(max(baseline_rate, 1e-3), 1 - 1e-3)
        p2 = min(max(p1 + effect, 1e-3), 1 - 1e-3)
        pbar = (p1 + p2) / 2.0
        z_a = norm.ppf(1 - self.t.alpha / 2.0)
        z_b = norm.ppf(self.t.power)
        num = (
            z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ) ** 2
        return int(math.ceil(num / (effect**2)))
