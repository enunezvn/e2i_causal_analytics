"""
Interim Analysis Service for A/B Testing.

Phase 15: A/B Testing Infrastructure

Implements interim analysis with:
- O'Brien-Fleming alpha spending
- Conditional power calculation
- Early stopping rules (efficacy/futility)
- Multiple comparison adjustments
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class SpendingFunction(str, Enum):
    """Alpha spending function types."""

    OBRIEN_FLEMING = "obrien_fleming"
    POCOCK = "pocock"
    HAYBITTLE_PETO = "haybittle_peto"
    CUSTOM = "custom"


class StoppingDecision(str, Enum):
    """Interim analysis stopping decisions."""

    CONTINUE = "continue"
    STOP_EFFICACY = "stop_efficacy"
    STOP_FUTILITY = "stop_futility"
    STOP_SAFETY = "stop_safety"
    MODIFY_SAMPLE = "modify_sample"


@dataclass
class InterimAnalysisConfig:
    """Configuration for interim analysis."""

    # Alpha spending
    total_alpha: float = 0.05
    spending_function: SpendingFunction = SpendingFunction.OBRIEN_FLEMING

    # Number of planned analyses
    num_planned_analyses: int = 3

    # Futility boundary
    futility_threshold: float = 0.2  # Conditional power below this triggers futility stop
    enable_futility_stopping: bool = True

    # Custom spending (if using CUSTOM function)
    custom_alpha_schedule: Optional[List[float]] = None

    # Effect size assumptions for conditional power
    assumed_effect_size: Optional[float] = None


@dataclass
class InterimAnalysisResult:
    """Result of an interim analysis."""

    experiment_id: UUID
    analysis_number: int
    performed_at: datetime

    # Data summary
    information_fraction: float
    sample_size_control: int
    sample_size_treatment: int
    total_sample_size: int
    target_sample_size: Optional[int] = None

    # Primary metric results
    control_mean: float = 0.0
    treatment_mean: float = 0.0
    effect_estimate: float = 0.0
    standard_error: float = 0.0
    test_statistic: float = 0.0
    p_value: float = 1.0
    effect_ci_lower: float = 0.0
    effect_ci_upper: float = 0.0

    # Alpha spending
    alpha_boundary: float = 0.05
    alpha_spent: float = 0.0
    cumulative_alpha_spent: float = 0.0

    # Conditional power
    conditional_power: Optional[float] = None
    predictive_probability: Optional[float] = None

    # Decision
    decision: StoppingDecision = StoppingDecision.CONTINUE
    decision_rationale: str = ""

    # All metrics snapshot
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricData:
    """Data for a single metric."""

    name: str
    control_values: np.ndarray
    treatment_values: np.ndarray


class InterimAnalysisService:
    """
    Service for performing interim analyses in A/B experiments.

    Implements group sequential designs with proper alpha spending
    to control Type I error while allowing early stopping.
    """

    def __init__(self, config: Optional[InterimAnalysisConfig] = None):
        """
        Initialize interim analysis service.

        Args:
            config: Analysis configuration
        """
        self.config = config or InterimAnalysisConfig()

    def calculate_obrien_fleming_boundary(
        self,
        information_fraction: float,
        total_alpha: float = 0.05,
        num_analyses: int = 3,
    ) -> float:
        """
        Calculate O'Brien-Fleming boundary for interim analysis.

        The O'Brien-Fleming boundary is very conservative early on,
        spending little alpha at early looks and more at the end.

        Args:
            information_fraction: Proportion of total information (0-1)
            total_alpha: Overall Type I error rate
            num_analyses: Total planned analyses

        Returns:
            Adjusted alpha (significance boundary) for this analysis
        """
        if information_fraction <= 0 or information_fraction > 1:
            raise ValueError(f"Information fraction must be in (0, 1], got {information_fraction}")

        # O'Brien-Fleming uses alpha * (2 * (1 - Phi(z_alpha/sqrt(t))))
        # where t is information fraction
        z_alpha = stats.norm.ppf(1 - total_alpha / 2)
        z_boundary = z_alpha / math.sqrt(information_fraction)
        alpha_boundary = 2 * (1 - stats.norm.cdf(z_boundary))

        return alpha_boundary

    def calculate_pocock_boundary(
        self,
        information_fraction: float,
        total_alpha: float = 0.05,
        num_analyses: int = 3,
    ) -> float:
        """
        Calculate Pocock boundary for interim analysis.

        The Pocock boundary uses the same significance level at each analysis,
        resulting in more aggressive early stopping compared to O'Brien-Fleming.

        Args:
            information_fraction: Proportion of total information (0-1)
            total_alpha: Overall Type I error rate
            num_analyses: Total planned analyses

        Returns:
            Adjusted alpha for this analysis
        """
        # Pocock uses constant boundary at each analysis
        # Approximation for the critical value
        # The exact value requires numerical integration

        # Simple approximation: divide alpha by sqrt of number of analyses
        # This is conservative but commonly used
        alpha_boundary = total_alpha / (1 + 0.5 * (num_analyses - 1))

        return alpha_boundary

    def calculate_haybittle_peto_boundary(
        self,
        information_fraction: float,
        is_final_analysis: bool = False,
    ) -> float:
        """
        Calculate Haybittle-Peto boundary.

        Uses a very stringent threshold (p < 0.001) for all interim analyses,
        preserving almost all alpha for the final analysis.

        Args:
            information_fraction: Not used, included for interface consistency
            is_final_analysis: Whether this is the final analysis

        Returns:
            Adjusted alpha for this analysis
        """
        if is_final_analysis:
            return 0.05  # Nearly all alpha preserved for final
        return 0.001  # Very stringent for interim

    def get_alpha_boundary(
        self,
        information_fraction: float,
        analysis_number: int,
        is_final: bool = False,
    ) -> float:
        """
        Get alpha boundary based on configured spending function.

        Args:
            information_fraction: Proportion of data collected
            analysis_number: Which analysis this is (1-indexed)
            is_final: Whether this is the final analysis

        Returns:
            Alpha boundary for this analysis
        """
        if self.config.spending_function == SpendingFunction.OBRIEN_FLEMING:
            return self.calculate_obrien_fleming_boundary(
                information_fraction,
                self.config.total_alpha,
                self.config.num_planned_analyses,
            )
        elif self.config.spending_function == SpendingFunction.POCOCK:
            return self.calculate_pocock_boundary(
                information_fraction,
                self.config.total_alpha,
                self.config.num_planned_analyses,
            )
        elif self.config.spending_function == SpendingFunction.HAYBITTLE_PETO:
            return self.calculate_haybittle_peto_boundary(
                information_fraction,
                is_final,
            )
        elif self.config.spending_function == SpendingFunction.CUSTOM:
            if self.config.custom_alpha_schedule and analysis_number <= len(
                self.config.custom_alpha_schedule
            ):
                return self.config.custom_alpha_schedule[analysis_number - 1]
            return self.config.total_alpha

        return self.config.total_alpha

    def calculate_conditional_power(
        self,
        current_effect: float,
        current_variance: float,
        target_effect: float,
        current_n: int,
        target_n: int,
        alpha: float = 0.05,
    ) -> float:
        """
        Calculate conditional power given current data.

        Conditional power is the probability of achieving statistical
        significance at the final analysis, given the current observed effect.

        Args:
            current_effect: Observed effect size
            current_variance: Observed variance
            target_effect: Expected/assumed true effect size
            current_n: Current sample size
            target_n: Target sample size
            alpha: Significance level

        Returns:
            Conditional power (0-1)
        """
        if current_n >= target_n or current_variance <= 0:
            return 0.0

        target_n - current_n
        z_alpha = stats.norm.ppf(1 - alpha / 2)

        # Information proportion
        info_current = 1 / current_variance if current_variance > 0 else 0
        info_current * (target_n / current_n)

        # Under the current trend (assumes effect remains same)
        # Z_final = sqrt(n_total/n_current) * Z_current + drift
        if info_current > 0:
            z_current = current_effect / math.sqrt(current_variance)
            # Predicted final Z under current trend
            z_final_trend = z_current * math.sqrt(target_n / current_n)

            # Account for remaining data
            # Conditional power under observed trend
            conditional_power = stats.norm.cdf(z_final_trend - z_alpha) + stats.norm.cdf(
                -z_final_trend - z_alpha
            )
        else:
            conditional_power = 0.0

        return max(0.0, min(1.0, conditional_power))

    def calculate_predictive_probability(
        self,
        current_effect: float,
        current_se: float,
        target_effect: float,
        current_n: int,
        target_n: int,
        alpha: float = 0.05,
    ) -> float:
        """
        Calculate predictive probability of success.

        Unlike conditional power, predictive probability averages over
        uncertainty in the true effect size.

        Args:
            current_effect: Observed effect
            current_se: Standard error of current estimate
            target_effect: Prior mean for effect
            current_n: Current sample size
            target_n: Target sample size
            alpha: Significance level

        Returns:
            Predictive probability (0-1)
        """
        if current_n >= target_n or current_se <= 0:
            return 0.0

        z_alpha = stats.norm.ppf(1 - alpha / 2)

        # Remaining information
        1 - (current_n / target_n)

        # Bayesian predictive probability
        # Posterior variance combining prior and data
        prior_variance = (target_effect / 2) ** 2 if target_effect != 0 else 1.0
        data_variance = current_se**2

        posterior_variance = 1 / (1 / prior_variance + 1 / data_variance)
        posterior_mean = posterior_variance * (
            target_effect / prior_variance + current_effect / data_variance
        )

        # Predictive variance for final estimate
        final_se = current_se * math.sqrt(current_n / target_n)
        predictive_variance = posterior_variance + final_se**2

        # Probability that final estimate exceeds threshold
        threshold = z_alpha * final_se
        predictive_prob = 1 - stats.norm.cdf(
            threshold, loc=posterior_mean, scale=math.sqrt(predictive_variance)
        )
        predictive_prob += stats.norm.cdf(
            -threshold, loc=posterior_mean, scale=math.sqrt(predictive_variance)
        )

        return max(0.0, min(1.0, predictive_prob))

    async def perform_interim_analysis(
        self,
        experiment_id: UUID,
        analysis_number: int,
        metric_data: MetricData,
        target_sample_size: Optional[int] = None,
        target_effect: Optional[float] = None,
    ) -> InterimAnalysisResult:
        """
        Perform a complete interim analysis.

        Args:
            experiment_id: Experiment UUID
            analysis_number: Analysis sequence number
            metric_data: Control and treatment metric values
            target_sample_size: Planned final sample size
            target_effect: Expected/minimum detectable effect

        Returns:
            Complete interim analysis result
        """
        now = datetime.now(timezone.utc)

        # Sample sizes
        n_control = len(metric_data.control_values)
        n_treatment = len(metric_data.treatment_values)
        total_n = n_control + n_treatment

        # Calculate information fraction
        if target_sample_size:
            info_fraction = total_n / target_sample_size
        else:
            # Assume we're at the planned analysis point
            info_fraction = analysis_number / self.config.num_planned_analyses

        info_fraction = min(1.0, info_fraction)

        # Calculate statistics
        control_mean = float(np.mean(metric_data.control_values))
        treatment_mean = float(np.mean(metric_data.treatment_values))
        effect_estimate = treatment_mean - control_mean

        # Pooled variance and standard error
        pooled_var = (
            np.var(metric_data.control_values, ddof=1) * (n_control - 1)
            + np.var(metric_data.treatment_values, ddof=1) * (n_treatment - 1)
        ) / (n_control + n_treatment - 2)
        standard_error = float(np.sqrt(pooled_var * (1 / n_control + 1 / n_treatment)))

        # Test statistic and p-value (two-sided t-test)
        if standard_error > 0:
            test_statistic = effect_estimate / standard_error
            df = n_control + n_treatment - 2
            p_value = 2 * (1 - stats.t.cdf(abs(test_statistic), df))
        else:
            test_statistic = 0.0
            p_value = 1.0

        # Confidence interval (95%)
        t_crit = stats.t.ppf(0.975, df=n_control + n_treatment - 2)
        ci_lower = effect_estimate - t_crit * standard_error
        ci_upper = effect_estimate + t_crit * standard_error

        # Alpha spending
        is_final = analysis_number >= self.config.num_planned_analyses
        alpha_boundary = self.get_alpha_boundary(info_fraction, analysis_number, is_final)

        # Calculate cumulative alpha spent
        cumulative_alpha = self._calculate_cumulative_alpha(info_fraction)

        # Conditional power
        target_effect = target_effect or self.config.assumed_effect_size or effect_estimate
        conditional_power = None
        predictive_prob = None

        if target_sample_size and total_n < target_sample_size:
            conditional_power = self.calculate_conditional_power(
                current_effect=effect_estimate,
                current_variance=standard_error**2,
                target_effect=target_effect,
                current_n=total_n,
                target_n=target_sample_size,
                alpha=self.config.total_alpha,
            )
            predictive_prob = self.calculate_predictive_probability(
                current_effect=effect_estimate,
                current_se=standard_error,
                target_effect=target_effect,
                current_n=total_n,
                target_n=target_sample_size,
                alpha=self.config.total_alpha,
            )

        # Make stopping decision
        decision, rationale = self._make_stopping_decision(
            p_value=p_value,
            alpha_boundary=alpha_boundary,
            conditional_power=conditional_power,
            effect_estimate=effect_estimate,
            is_final=is_final,
        )

        # Create result
        result = InterimAnalysisResult(
            experiment_id=experiment_id,
            analysis_number=analysis_number,
            performed_at=now,
            information_fraction=info_fraction,
            sample_size_control=n_control,
            sample_size_treatment=n_treatment,
            total_sample_size=total_n,
            target_sample_size=target_sample_size,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            effect_estimate=effect_estimate,
            standard_error=standard_error,
            test_statistic=test_statistic,
            p_value=p_value,
            effect_ci_lower=float(ci_lower),
            effect_ci_upper=float(ci_upper),
            alpha_boundary=alpha_boundary,
            alpha_spent=alpha_boundary,  # Simplified
            cumulative_alpha_spent=cumulative_alpha,
            conditional_power=conditional_power,
            predictive_probability=predictive_prob,
            decision=decision,
            decision_rationale=rationale,
            metrics_snapshot={
                "primary_metric": {
                    "name": metric_data.name,
                    "control_mean": control_mean,
                    "treatment_mean": treatment_mean,
                    "effect": effect_estimate,
                    "relative_lift": (effect_estimate / control_mean * 100)
                    if control_mean != 0
                    else 0,
                },
                "sample_sizes": {
                    "control": n_control,
                    "treatment": n_treatment,
                },
            },
        )

        # Persist to database
        await self._persist_analysis(result)

        logger.info(
            f"Interim analysis {analysis_number} for experiment {experiment_id}: "
            f"p={p_value:.4f}, boundary={alpha_boundary:.4f}, decision={decision.value}"
        )

        return result

    def _calculate_cumulative_alpha(self, information_fraction: float) -> float:
        """Calculate cumulative alpha spent up to current analysis."""
        if self.config.spending_function == SpendingFunction.OBRIEN_FLEMING:
            # O'Brien-Fleming cumulative spending
            z_alpha = stats.norm.ppf(1 - self.config.total_alpha / 2)
            z_boundary = z_alpha / math.sqrt(information_fraction)
            return 2 * (1 - stats.norm.cdf(z_boundary))
        elif self.config.spending_function == SpendingFunction.POCOCK:
            # Pocock spends alpha proportionally
            return self.config.total_alpha * information_fraction
        else:
            return self.config.total_alpha * information_fraction

    def _make_stopping_decision(
        self,
        p_value: float,
        alpha_boundary: float,
        conditional_power: Optional[float],
        effect_estimate: float,
        is_final: bool,
    ) -> Tuple[StoppingDecision, str]:
        """
        Make stopping decision based on analysis results.

        Args:
            p_value: Observed p-value
            alpha_boundary: Significance threshold
            conditional_power: Conditional power (if available)
            effect_estimate: Observed effect
            is_final: Whether this is the final analysis

        Returns:
            Tuple of (decision, rationale)
        """
        # Efficacy stopping
        if p_value < alpha_boundary:
            if effect_estimate > 0:
                return (
                    StoppingDecision.STOP_EFFICACY,
                    f"Treatment effect significant (p={p_value:.4f} < {alpha_boundary:.4f}) with positive effect ({effect_estimate:.4f})",
                )
            else:
                return (
                    StoppingDecision.STOP_EFFICACY,
                    f"Significant result (p={p_value:.4f} < {alpha_boundary:.4f}) but effect is negative ({effect_estimate:.4f}) - treatment appears harmful",
                )

        # Futility stopping (only if not final and enabled)
        if self.config.enable_futility_stopping and conditional_power is not None and not is_final:
            if conditional_power < self.config.futility_threshold:
                return (
                    StoppingDecision.STOP_FUTILITY,
                    f"Low conditional power ({conditional_power:.2%} < {self.config.futility_threshold:.0%}) suggests futility of continuing",
                )

        # Continue if final and not significant
        if is_final:
            return (
                StoppingDecision.CONTINUE,
                f"Final analysis: not significant (p={p_value:.4f} >= {alpha_boundary:.4f})",
            )

        # Continue for more data
        return (
            StoppingDecision.CONTINUE,
            f"Continue experiment: p={p_value:.4f} >= {alpha_boundary:.4f}, need more data",
        )

    async def _persist_analysis(self, result: InterimAnalysisResult) -> None:
        """Persist interim analysis to database."""
        from src.repositories.ab_experiment import ABExperimentRepository

        repo = ABExperimentRepository()
        await repo.record_interim_analysis(
            experiment_id=result.experiment_id,
            analysis_number=result.analysis_number,
            information_fraction=result.information_fraction,
            sample_size_at_analysis=result.total_sample_size,
            target_sample_size=result.target_sample_size,
            spending_function=self.config.spending_function.value,
            alpha_spent=result.alpha_spent,
            cumulative_alpha_spent=result.cumulative_alpha_spent,
            adjusted_alpha=result.alpha_boundary,
            test_statistic=result.test_statistic,
            standard_error=result.standard_error,
            p_value=result.p_value,
            effect_estimate=result.effect_estimate,
            effect_ci_lower=result.effect_ci_lower,
            effect_ci_upper=result.effect_ci_upper,
            conditional_power=result.conditional_power,
            predictive_probability=result.predictive_probability,
            decision=result.decision.value,
            decision_rationale=result.decision_rationale,
            metrics_snapshot=result.metrics_snapshot,
        )

    async def get_analysis_history(
        self,
        experiment_id: UUID,
    ) -> List[InterimAnalysisResult]:
        """
        Get all interim analyses for an experiment.

        Args:
            experiment_id: Experiment UUID

        Returns:
            List of interim analysis results
        """
        from src.repositories.ab_experiment import ABExperimentRepository

        repo = ABExperimentRepository()
        analyses = await repo.get_interim_analyses(experiment_id)

        return [
            InterimAnalysisResult(
                experiment_id=a.experiment_id,
                analysis_number=a.analysis_number,
                performed_at=a.performed_at,
                information_fraction=a.information_fraction,
                sample_size_control=0,  # Not stored separately
                sample_size_treatment=0,
                total_sample_size=a.sample_size_at_analysis,
                target_sample_size=a.target_sample_size,
                effect_estimate=a.effect_estimate or 0.0,
                standard_error=a.standard_error or 0.0,
                test_statistic=a.test_statistic or 0.0,
                p_value=a.p_value or 1.0,
                effect_ci_lower=a.effect_ci_lower or 0.0,
                effect_ci_upper=a.effect_ci_upper or 0.0,
                alpha_boundary=a.adjusted_alpha,
                alpha_spent=a.alpha_spent,
                cumulative_alpha_spent=a.cumulative_alpha_spent,
                conditional_power=a.conditional_power,
                predictive_probability=a.predictive_probability,
                decision=StoppingDecision(a.decision),
                decision_rationale=a.decision_rationale or "",
                metrics_snapshot=a.metrics_snapshot,
            )
            for a in analyses
        ]

    def recommend_next_analysis_timing(
        self,
        current_n: int,
        target_n: int,
        current_analysis_number: int,
    ) -> Dict[str, Any]:
        """
        Recommend when to perform the next interim analysis.

        Args:
            current_n: Current sample size
            target_n: Target sample size
            current_analysis_number: Last completed analysis number

        Returns:
            Recommendation with target sample size for next analysis
        """
        remaining_analyses = self.config.num_planned_analyses - current_analysis_number
        if remaining_analyses <= 0:
            return {
                "recommendation": "no_more_analyses",
                "reason": "All planned interim analyses complete",
            }

        # Calculate information fraction for next analysis
        next_analysis_num = current_analysis_number + 1
        next_info_fraction = next_analysis_num / self.config.num_planned_analyses

        # Target sample size for next analysis
        next_target_n = int(target_n * next_info_fraction)

        # Samples needed
        samples_needed = next_target_n - current_n

        return {
            "next_analysis_number": next_analysis_num,
            "target_information_fraction": next_info_fraction,
            "target_sample_size": next_target_n,
            "samples_needed": max(0, samples_needed),
            "current_progress": current_n / target_n if target_n > 0 else 0,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def get_interim_analysis_service(
    config: Optional[InterimAnalysisConfig] = None,
) -> InterimAnalysisService:
    """Get interim analysis service instance."""
    return InterimAnalysisService(config)
