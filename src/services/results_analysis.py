"""
Results Analysis Service for A/B Testing.

Phase 15: A/B Testing Infrastructure

Implements experiment results analysis:
- Intent-to-treat (ITT) analysis
- Per-protocol analysis
- Heterogeneous treatment effects (HTE)
- Sample Ratio Mismatch (SRM) detection
- Digital Twin fidelity tracking
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class AnalysisMethod(str, Enum):
    """Analysis method types."""

    ITT = "itt"  # Intent-to-treat
    PER_PROTOCOL = "per_protocol"
    AS_TREATED = "as_treated"


class AnalysisType(str, Enum):
    """Analysis timing types."""

    INTERIM = "interim"
    FINAL = "final"
    AD_HOC = "ad_hoc"


class SRMSeverity(str, Enum):
    """Sample Ratio Mismatch severity levels."""

    NONE = "none"
    WARNING = "warning"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class ExperimentResults:
    """Complete experiment results."""

    experiment_id: UUID
    analysis_type: AnalysisType
    analysis_method: AnalysisMethod
    computed_at: datetime

    # Primary metric
    primary_metric: str
    control_mean: float
    treatment_mean: float
    effect_estimate: float
    effect_ci_lower: float
    effect_ci_upper: float
    relative_lift: float  # Percentage change
    relative_lift_ci_lower: float
    relative_lift_ci_upper: float
    p_value: float
    is_significant: bool

    # Sample sizes
    sample_size_control: int
    sample_size_treatment: int
    statistical_power: float

    # Secondary metrics
    secondary_metrics: List[Dict[str, Any]] = field(default_factory=list)

    # Segment results (HTE)
    segment_results: Optional[Dict[str, Dict[str, Any]]] = None


@dataclass
class SRMCheckResult:
    """Sample Ratio Mismatch check result."""

    experiment_id: UUID
    checked_at: datetime
    expected_ratio: Dict[str, float]
    actual_counts: Dict[str, int]
    actual_ratio: Dict[str, float]
    chi_squared_statistic: float
    p_value: float
    is_srm_detected: bool
    severity: SRMSeverity
    investigation_notes: Optional[str] = None


@dataclass
class FidelityComparison:
    """Comparison between actual results and Digital Twin prediction."""

    experiment_id: UUID
    twin_simulation_id: UUID
    comparison_timestamp: datetime

    # Predicted vs actual
    predicted_effect: float
    actual_effect: float
    prediction_error: float
    prediction_error_percent: float

    # Confidence interval coverage
    predicted_ci_lower: float
    predicted_ci_upper: float
    ci_coverage: bool  # Did actual fall within predicted CI?

    # Fidelity scoring
    fidelity_score: float  # 0-1 score
    fidelity_grade: str  # A, B, C, D, F

    # Calibration recommendations
    calibration_adjustment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultsAnalysisConfig:
    """Configuration for results analysis."""

    # Significance
    alpha: float = 0.05
    power_threshold: float = 0.8

    # SRM detection
    srm_p_threshold: float = 0.001
    srm_warning_threshold: float = 0.01

    # Fidelity scoring
    fidelity_excellent_threshold: float = 0.1  # <10% error
    fidelity_good_threshold: float = 0.2  # <20% error
    fidelity_acceptable_threshold: float = 0.3  # <30% error


class ResultsAnalysisService:
    """
    Service for analyzing A/B experiment results.

    Provides comprehensive analysis including statistical testing,
    heterogeneous effects, and Digital Twin validation.
    """

    def __init__(self, config: Optional[ResultsAnalysisConfig] = None):
        """
        Initialize results analysis service.

        Args:
            config: Analysis configuration
        """
        self.config = config or ResultsAnalysisConfig()

    async def compute_itt_results(
        self,
        experiment_id: UUID,
        primary_metric: str,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        secondary_metrics: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        analysis_type: AnalysisType = AnalysisType.FINAL,
    ) -> ExperimentResults:
        """
        Compute Intent-to-Treat analysis.

        ITT includes all randomized units regardless of protocol adherence.

        Args:
            experiment_id: Experiment UUID
            primary_metric: Name of primary metric
            control_data: Control group metric values
            treatment_data: Treatment group metric values
            secondary_metrics: Optional dict of metric name -> (control, treatment) data
            analysis_type: Type of analysis

        Returns:
            Complete experiment results
        """
        return await self._compute_results(
            experiment_id=experiment_id,
            primary_metric=primary_metric,
            control_data=control_data,
            treatment_data=treatment_data,
            secondary_metrics=secondary_metrics,
            analysis_type=analysis_type,
            analysis_method=AnalysisMethod.ITT,
        )

    async def compute_per_protocol_results(
        self,
        experiment_id: UUID,
        primary_metric: str,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        control_compliant_mask: np.ndarray,
        treatment_compliant_mask: np.ndarray,
        secondary_metrics: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        analysis_type: AnalysisType = AnalysisType.FINAL,
    ) -> ExperimentResults:
        """
        Compute Per-Protocol analysis.

        PP only includes units that adhered to the protocol.

        Args:
            experiment_id: Experiment UUID
            primary_metric: Name of primary metric
            control_data: Control group metric values
            treatment_data: Treatment group metric values
            control_compliant_mask: Boolean mask for compliant control units
            treatment_compliant_mask: Boolean mask for compliant treatment units
            secondary_metrics: Optional secondary metrics
            analysis_type: Type of analysis

        Returns:
            Complete experiment results
        """
        # Filter to compliant units
        compliant_control = control_data[control_compliant_mask]
        compliant_treatment = treatment_data[treatment_compliant_mask]

        # Filter secondary metrics too
        filtered_secondary = None
        if secondary_metrics:
            filtered_secondary = {}
            for name, (ctrl, treat) in secondary_metrics.items():
                filtered_secondary[name] = (
                    ctrl[control_compliant_mask],
                    treat[treatment_compliant_mask],
                )

        return await self._compute_results(
            experiment_id=experiment_id,
            primary_metric=primary_metric,
            control_data=compliant_control,
            treatment_data=compliant_treatment,
            secondary_metrics=filtered_secondary,
            analysis_type=analysis_type,
            analysis_method=AnalysisMethod.PER_PROTOCOL,
        )

    async def _compute_results(
        self,
        experiment_id: UUID,
        primary_metric: str,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        secondary_metrics: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
        analysis_type: AnalysisType,
        analysis_method: AnalysisMethod,
    ) -> ExperimentResults:
        """Internal method to compute results."""
        now = datetime.now(timezone.utc)

        # Primary metric analysis
        n_control = len(control_data)
        n_treatment = len(treatment_data)

        control_mean = float(np.mean(control_data))
        treatment_mean = float(np.mean(treatment_data))
        effect_estimate = treatment_mean - control_mean

        # Standard error and test statistic
        pooled_var = (
            np.var(control_data, ddof=1) * (n_control - 1)
            + np.var(treatment_data, ddof=1) * (n_treatment - 1)
        ) / (n_control + n_treatment - 2)
        se = float(np.sqrt(pooled_var * (1 / n_control + 1 / n_treatment)))

        # t-test
        df = n_control + n_treatment - 2
        if se > 0:
            t_stat = effect_estimate / se
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            p_value = 1.0

        # Confidence intervals
        t_crit = stats.t.ppf(1 - self.config.alpha / 2, df)
        ci_lower = effect_estimate - t_crit * se
        ci_upper = effect_estimate + t_crit * se

        # Relative lift
        if control_mean != 0:
            relative_lift = (effect_estimate / control_mean) * 100
            rel_se = se / abs(control_mean) * 100
            relative_ci_lower = relative_lift - t_crit * rel_se
            relative_ci_upper = relative_lift + t_crit * rel_se
        else:
            relative_lift = 0.0
            relative_ci_lower = 0.0
            relative_ci_upper = 0.0

        # Statistical power (post-hoc)
        power = self._calculate_power(
            effect=effect_estimate,
            se=se,
            n_control=n_control,
            n_treatment=n_treatment,
        )

        is_significant = p_value < self.config.alpha

        # Secondary metrics
        secondary_results = []
        if secondary_metrics:
            for metric_name, (ctrl, treat) in secondary_metrics.items():
                secondary_results.append(self._analyze_metric(metric_name, ctrl, treat))

        result = ExperimentResults(
            experiment_id=experiment_id,
            analysis_type=analysis_type,
            analysis_method=analysis_method,
            computed_at=now,
            primary_metric=primary_metric,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            effect_estimate=effect_estimate,
            effect_ci_lower=float(ci_lower),
            effect_ci_upper=float(ci_upper),
            relative_lift=relative_lift,
            relative_lift_ci_lower=relative_ci_lower,
            relative_lift_ci_upper=relative_ci_upper,
            p_value=p_value,
            is_significant=is_significant,
            sample_size_control=n_control,
            sample_size_treatment=n_treatment,
            statistical_power=power,
            secondary_metrics=secondary_results,
        )

        # Persist results
        await self._persist_results(result)

        logger.info(
            f"Computed {analysis_method.value} results for {experiment_id}: "
            f"effect={effect_estimate:.4f}, p={p_value:.4f}, significant={is_significant}"
        )

        return result

    def _analyze_metric(
        self,
        metric_name: str,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze a single metric."""
        n_control = len(control_data)
        n_treatment = len(treatment_data)

        control_mean = float(np.mean(control_data))
        treatment_mean = float(np.mean(treatment_data))
        effect = treatment_mean - control_mean

        # Welch's t-test (doesn't assume equal variances)
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)

        return {
            "name": metric_name,
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "effect": effect,
            "relative_lift": (effect / control_mean * 100) if control_mean != 0 else 0,
            "p_value": float(p_value),
            "is_significant": p_value < self.config.alpha,
            "sample_size_control": n_control,
            "sample_size_treatment": n_treatment,
        }

    def _calculate_power(
        self,
        effect: float,
        se: float,
        n_control: int,
        n_treatment: int,
    ) -> float:
        """Calculate observed statistical power."""
        if se <= 0:
            return 0.0

        z_alpha = stats.norm.ppf(1 - self.config.alpha / 2)
        z_effect = abs(effect) / se

        # Power = P(reject H0 | H1 true)
        power = stats.norm.cdf(z_effect - z_alpha) + stats.norm.cdf(-z_effect - z_alpha)

        return float(max(0.0, min(1.0, power)))

    async def compute_heterogeneous_effects(
        self,
        experiment_id: UUID,
        primary_metric: str,
        segment_data: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute heterogeneous treatment effects by segment.

        Args:
            experiment_id: Experiment UUID
            primary_metric: Name of primary metric
            segment_data: Dict of segment_name -> {"control": array, "treatment": array}

        Returns:
            Dict of segment_name -> results
        """
        results = {}

        for segment_name, data in segment_data.items():
            control = data.get("control", np.array([]))
            treatment = data.get("treatment", np.array([]))

            if len(control) < 10 or len(treatment) < 10:
                logger.warning(
                    f"Skipping segment {segment_name}: insufficient data "
                    f"(control={len(control)}, treatment={len(treatment)})"
                )
                continue

            segment_result = self._analyze_metric(primary_metric, control, treatment)
            segment_result["segment"] = segment_name
            segment_result["n_total"] = len(control) + len(treatment)

            results[segment_name] = segment_result

        logger.info(f"Computed HTE for {experiment_id}: {len(results)} segments analyzed")

        return results

    async def check_sample_ratio_mismatch(
        self,
        experiment_id: UUID,
        expected_ratio: Dict[str, float],
        actual_counts: Dict[str, int],
    ) -> SRMCheckResult:
        """
        Check for Sample Ratio Mismatch (SRM).

        SRM indicates potential issues with randomization or data collection.

        Args:
            experiment_id: Experiment UUID
            expected_ratio: Expected allocation ratio
            actual_counts: Actual enrollment counts per variant

        Returns:
            SRM check result
        """
        now = datetime.now(timezone.utc)

        # Calculate total and expected counts
        total = sum(actual_counts.values())
        expected_counts = {variant: ratio * total for variant, ratio in expected_ratio.items()}

        # Chi-squared test
        observed = [actual_counts.get(v, 0) for v in expected_ratio.keys()]
        expected = [expected_counts.get(v, 0) for v in expected_ratio.keys()]

        if sum(expected) > 0:
            chi2, p_value = stats.chisquare(observed, f_exp=expected)
        else:
            chi2 = 0.0
            p_value = 1.0

        # Determine severity
        if p_value < self.config.srm_p_threshold:
            severity = SRMSeverity.CRITICAL
            is_srm = True
        elif p_value < self.config.srm_warning_threshold:
            severity = SRMSeverity.WARNING
            is_srm = True
        else:
            severity = SRMSeverity.NONE
            is_srm = False

        # Calculate actual ratio
        actual_ratio = {
            variant: count / total if total > 0 else 0 for variant, count in actual_counts.items()
        }

        result = SRMCheckResult(
            experiment_id=experiment_id,
            checked_at=now,
            expected_ratio=expected_ratio,
            actual_counts=actual_counts,
            actual_ratio=actual_ratio,
            chi_squared_statistic=float(chi2),
            p_value=float(p_value),
            is_srm_detected=is_srm,
            severity=severity,
        )

        # Persist SRM check
        await self._persist_srm_check(result)

        if is_srm:
            logger.warning(
                f"SRM detected for {experiment_id}: chi2={chi2:.2f}, p={p_value:.6f}, "
                f"severity={severity.value}"
            )
        else:
            logger.info(f"No SRM detected for {experiment_id}: chi2={chi2:.2f}, p={p_value:.4f}")

        return result

    async def compare_with_twin_prediction(
        self,
        experiment_id: UUID,
        twin_simulation_id: UUID,
        actual_results: ExperimentResults,
        predicted_effect: float,
        predicted_ci_lower: float,
        predicted_ci_upper: float,
    ) -> FidelityComparison:
        """
        Compare actual results with Digital Twin prediction.

        Args:
            experiment_id: Experiment UUID
            twin_simulation_id: Digital Twin simulation UUID
            actual_results: Computed experiment results
            predicted_effect: Twin's predicted effect
            predicted_ci_lower: Twin's CI lower bound
            predicted_ci_upper: Twin's CI upper bound

        Returns:
            Fidelity comparison result
        """
        now = datetime.now(timezone.utc)

        actual_effect = actual_results.effect_estimate
        prediction_error = actual_effect - predicted_effect
        prediction_error_percent = (
            abs(prediction_error / predicted_effect * 100) if predicted_effect != 0 else 0.0
        )

        # Check CI coverage
        ci_coverage = predicted_ci_lower <= actual_effect <= predicted_ci_upper

        # Calculate fidelity score
        fidelity_score = self._calculate_fidelity_score(
            prediction_error_percent,
            ci_coverage,
        )

        # Assign grade
        fidelity_grade = self._assign_fidelity_grade(fidelity_score)

        # Generate calibration recommendations
        calibration = self._generate_calibration_recommendations(
            prediction_error=prediction_error,
            prediction_error_percent=prediction_error_percent,
            ci_coverage=ci_coverage,
        )

        result = FidelityComparison(
            experiment_id=experiment_id,
            twin_simulation_id=twin_simulation_id,
            comparison_timestamp=now,
            predicted_effect=predicted_effect,
            actual_effect=actual_effect,
            prediction_error=prediction_error,
            prediction_error_percent=prediction_error_percent,
            predicted_ci_lower=predicted_ci_lower,
            predicted_ci_upper=predicted_ci_upper,
            ci_coverage=ci_coverage,
            fidelity_score=fidelity_score,
            fidelity_grade=fidelity_grade,
            calibration_adjustment=calibration,
        )

        # Persist comparison
        await self._persist_fidelity_comparison(result)

        logger.info(
            f"Twin fidelity for {experiment_id}: "
            f"predicted={predicted_effect:.4f}, actual={actual_effect:.4f}, "
            f"error={prediction_error_percent:.1f}%, grade={fidelity_grade}"
        )

        return result

    def _calculate_fidelity_score(
        self,
        error_percent: float,
        ci_coverage: bool,
    ) -> float:
        """Calculate fidelity score (0-1)."""
        # Base score from prediction error
        if error_percent < self.config.fidelity_excellent_threshold * 100:
            base_score = 1.0
        elif error_percent < self.config.fidelity_good_threshold * 100:
            base_score = 0.8
        elif error_percent < self.config.fidelity_acceptable_threshold * 100:
            base_score = 0.6
        else:
            base_score = max(0.0, 0.4 - (error_percent - 30) / 100)

        # Bonus for CI coverage
        if ci_coverage:
            base_score = min(1.0, base_score + 0.1)

        return base_score

    def _assign_fidelity_grade(self, score: float) -> str:
        """Assign letter grade based on fidelity score."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"

    def _generate_calibration_recommendations(
        self,
        prediction_error: float,
        prediction_error_percent: float,
        ci_coverage: bool,
    ) -> Dict[str, Any]:
        """Generate calibration recommendations based on fidelity."""
        direction = "overestimated" if prediction_error < 0 else "underestimated"
        suggestions: List[str] = []

        if prediction_error_percent > 30:
            suggestions.append("Consider updating twin model parameters")
        if not ci_coverage:
            suggestions.append("Twin uncertainty estimates may need widening")
        if abs(prediction_error) > 0.1:
            suggestions.append(
                f"Twin {direction} effect by {prediction_error_percent:.0f}%"
            )

        recommendations: Dict[str, Any] = {
            "needs_calibration": prediction_error_percent > 20,
            "direction": direction,
            "magnitude_adjustment": -prediction_error,
            "suggestions": suggestions,
        }

        return recommendations

    async def _persist_results(self, result: ExperimentResults) -> None:
        """Persist experiment results to database."""
        from src.repositories.ab_results import ABResultsRepository

        repo = ABResultsRepository()
        await repo.save_results(result)

    async def _persist_srm_check(self, result: SRMCheckResult) -> None:
        """Persist SRM check to database."""
        from src.repositories.ab_results import ABResultsRepository

        repo = ABResultsRepository()
        await repo.save_srm_check(result)

    async def _persist_fidelity_comparison(self, result: FidelityComparison) -> None:
        """Persist fidelity comparison to database."""
        from src.repositories.ab_results import ABResultsRepository

        repo = ABResultsRepository()
        await repo.save_fidelity_comparison(result)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def get_results_analysis_service(
    config: Optional[ResultsAnalysisConfig] = None,
) -> ResultsAnalysisService:
    """Get results analysis service instance."""
    return ResultsAnalysisService(config)
