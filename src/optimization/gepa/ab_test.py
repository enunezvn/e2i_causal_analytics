"""GEPA A/B Testing Infrastructure for E2I Agents.

This module provides A/B testing capabilities for comparing GEPA-optimized
agent versions against baselines in production.

Integrates with:
- prompt_ab_tests table (database/ml/023_gepa_optimization_tables.sql)
- prompt_ab_test_observations table
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from scipy import stats


@dataclass
class ABTestVariant:
    """Represents a variant in an A/B test."""

    variant_id: str
    name: str
    instruction_id: Optional[str] = None
    is_baseline: bool = False


@dataclass
class ABTestObservation:
    """Single observation in an A/B test."""

    observation_id: str
    test_id: str
    variant: str
    request_id: str
    score: Optional[float] = None
    latency_ms: Optional[int] = None
    success: bool = True
    error_type: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ABTestResults:
    """Results from an A/B test analysis."""

    test_id: str
    baseline_requests: int
    treatment_requests: int
    baseline_score_avg: Optional[float]
    treatment_score_avg: Optional[float]
    score_delta: Optional[float]
    baseline_latency_p50: Optional[int]
    treatment_latency_p50: Optional[int]
    p_value: Optional[float]
    confidence_interval: Optional[tuple[float, float]]
    is_significant: bool
    winner: Optional[str]
    recommendation: str


class GEPAABTest:
    """A/B testing for GEPA-optimized vs baseline agent versions.

    Provides:
    - Traffic splitting
    - Score and latency tracking
    - Statistical significance analysis
    - Winner determination

    Example:
        >>> ab_test = GEPAABTest(
        ...     test_name="causal_impact_gepa_v1",
        ...     agent_name="causal_impact",
        ...     traffic_split=0.10,
        ... )
        >>> variant = ab_test.assign_variant(user_id="user123")
        >>> ab_test.record_observation(
        ...     request_id="req456",
        ...     variant=variant,
        ...     score=0.85,
        ...     latency_ms=150,
        ... )
        >>> results = ab_test.analyze()
    """

    def __init__(
        self,
        test_name: str,
        agent_name: str,
        traffic_split: float = 0.10,
        baseline_instruction_id: Optional[str] = None,
        treatment_instruction_id: Optional[str] = None,
        target_sample_size: int = 1000,
        significance_level: float = 0.05,
    ):
        """Initialize A/B test.

        Args:
            test_name: Human-readable test name
            agent_name: Name of the agent being tested
            traffic_split: Fraction of traffic to treatment (0-1)
            baseline_instruction_id: ID of baseline instruction
            treatment_instruction_id: ID of GEPA-optimized instruction
            target_sample_size: Minimum samples before significance test
            significance_level: P-value threshold for significance (default 0.05)
        """
        self.test_id = str(uuid4())
        self.test_name = test_name
        self.agent_name = agent_name
        self.traffic_split = traffic_split
        self.baseline_instruction_id = baseline_instruction_id
        self.treatment_instruction_id = treatment_instruction_id
        self.target_sample_size = target_sample_size
        self.significance_level = significance_level

        self.status = "draft"
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None

        self.observations: list[ABTestObservation] = []

        self.variants = {
            "baseline": ABTestVariant(
                variant_id=str(uuid4()),
                name="baseline",
                instruction_id=baseline_instruction_id,
                is_baseline=True,
            ),
            "gepa": ABTestVariant(
                variant_id=str(uuid4()),
                name="gepa",
                instruction_id=treatment_instruction_id,
                is_baseline=False,
            ),
        }

    def start(self) -> None:
        """Start the A/B test."""
        self.status = "running"
        self.started_at = datetime.now()

    def stop(self) -> None:
        """Stop the A/B test."""
        self.status = "stopped"
        self.ended_at = datetime.now()

    def assign_variant(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Assign a variant for a request.

        Uses consistent hashing for user-level stickiness.

        Args:
            user_id: Optional user ID for sticky assignment
            session_id: Optional session ID for sticky assignment

        Returns:
            Variant name ("baseline" or "gepa")
        """
        if self.status != "running":
            return "baseline"

        # Use consistent hashing if user_id provided
        if user_id:
            hash_val = hash(f"{self.test_id}:{user_id}") % 100
            return "gepa" if hash_val < (self.traffic_split * 100) else "baseline"

        # Random assignment
        return "gepa" if random.random() < self.traffic_split else "baseline"

    def record_observation(
        self,
        request_id: str,
        variant: str,
        score: Optional[float] = None,
        latency_ms: Optional[int] = None,
        success: bool = True,
        error_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ABTestObservation:
        """Record an observation for the A/B test.

        Args:
            request_id: Unique request identifier
            variant: Variant name ("baseline" or "gepa")
            score: Metric score (0-1)
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
            error_type: Error type if failed
            user_id: Optional user ID
            session_id: Optional session ID

        Returns:
            Created observation
        """
        observation = ABTestObservation(
            observation_id=str(uuid4()),
            test_id=self.test_id,
            variant=variant,
            request_id=request_id,
            score=score,
            latency_ms=latency_ms,
            success=success,
            error_type=error_type,
            user_id=user_id,
            session_id=session_id,
        )
        self.observations.append(observation)
        return observation

    def analyze(self) -> ABTestResults:
        """Analyze A/B test results.

        Performs:
        - Score comparison with t-test
        - Latency comparison
        - Significance determination
        - Winner recommendation

        Returns:
            ABTestResults with analysis
        """
        # Split observations by variant
        baseline_obs = [o for o in self.observations if o.variant == "baseline"]
        treatment_obs = [o for o in self.observations if o.variant == "gepa"]

        # Score analysis
        baseline_scores = [o.score for o in baseline_obs if o.score is not None]
        treatment_scores = [o.score for o in treatment_obs if o.score is not None]

        baseline_score_avg = (
            sum(baseline_scores) / len(baseline_scores) if baseline_scores else None
        )
        treatment_score_avg = (
            sum(treatment_scores) / len(treatment_scores) if treatment_scores else None
        )

        score_delta = None
        if baseline_score_avg is not None and treatment_score_avg is not None:
            score_delta = treatment_score_avg - baseline_score_avg

        # Latency analysis (P50)
        baseline_latencies = sorted([o.latency_ms for o in baseline_obs if o.latency_ms])
        treatment_latencies = sorted([o.latency_ms for o in treatment_obs if o.latency_ms])

        baseline_latency_p50 = (
            baseline_latencies[len(baseline_latencies) // 2] if baseline_latencies else None
        )
        treatment_latency_p50 = (
            treatment_latencies[len(treatment_latencies) // 2] if treatment_latencies else None
        )

        # Statistical significance
        p_value = None
        confidence_interval = None
        is_significant = False

        if len(baseline_scores) >= 30 and len(treatment_scores) >= 30:
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(treatment_scores, baseline_scores)

            # 95% confidence interval for difference
            mean_diff = treatment_score_avg - baseline_score_avg
            pooled_se = (stats.sem(baseline_scores) ** 2 + stats.sem(treatment_scores) ** 2) ** 0.5
            ci_margin = 1.96 * pooled_se
            confidence_interval = (mean_diff - ci_margin, mean_diff + ci_margin)

            is_significant = p_value < self.significance_level

        # Winner determination
        winner = None
        recommendation = "Insufficient data for recommendation"

        total_samples = len(baseline_obs) + len(treatment_obs)
        if total_samples >= self.target_sample_size and is_significant:
            if score_delta > 0:
                winner = "gepa"
                recommendation = (
                    f"GEPA variant shows +{score_delta:.1%} improvement. Recommend rolling out."
                )
            else:
                winner = "baseline"
                recommendation = f"Baseline performs better by {-score_delta:.1%}. Keep baseline."
        elif total_samples >= self.target_sample_size:
            recommendation = (
                "No significant difference detected. Consider extending test or keeping baseline."
            )
        else:
            remaining = self.target_sample_size - total_samples
            recommendation = f"Need {remaining} more samples before analysis."

        return ABTestResults(
            test_id=self.test_id,
            baseline_requests=len(baseline_obs),
            treatment_requests=len(treatment_obs),
            baseline_score_avg=baseline_score_avg,
            treatment_score_avg=treatment_score_avg,
            score_delta=score_delta,
            baseline_latency_p50=baseline_latency_p50,
            treatment_latency_p50=treatment_latency_p50,
            p_value=p_value,
            confidence_interval=confidence_interval,
            is_significant=is_significant,
            winner=winner,
            recommendation=recommendation,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize A/B test to dictionary for persistence."""
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "agent_name": self.agent_name,
            "traffic_split": self.traffic_split,
            "baseline_instruction_id": self.baseline_instruction_id,
            "treatment_instruction_id": self.treatment_instruction_id,
            "target_sample_size": self.target_sample_size,
            "significance_level": self.significance_level,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "observation_count": len(self.observations),
        }


__all__ = [
    "GEPAABTest",
    "ABTestVariant",
    "ABTestObservation",
    "ABTestResults",
]
