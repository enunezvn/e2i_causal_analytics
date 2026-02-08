"""Nested Confidence Interval computation for hierarchical analysis.

B9.3: Combines segment-level confidence intervals into aggregate estimates.

This module implements various aggregation methods for combining
confidence intervals from multiple segments into a single nested CI.

Methods:
- SAMPLE_WEIGHTED: Weight by segment sample size
- VARIANCE_WEIGHTED: Inverse-variance weighting (most efficient)
- EQUAL: Simple average across segments
- BOOTSTRAP: Bootstrap aggregation for robust estimates
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


class AggregationMethod(str, Enum):
    """Methods for aggregating segment-level estimates."""

    SAMPLE_WEIGHTED = "sample_weighted"  # Weight by segment sample size
    VARIANCE_WEIGHTED = "variance_weighted"  # Inverse-variance weighting
    EQUAL = "equal"  # Simple average
    BOOTSTRAP = "bootstrap"  # Bootstrap aggregation


@dataclass
class NestedCIConfig:
    """Configuration for nested confidence interval computation.

    Attributes:
        confidence_level: Confidence level (0-1), default 0.95 for 95% CI.
        aggregation_method: Method for combining segment estimates.
        min_segment_size: Minimum segment size to include in aggregation.
        bootstrap_iterations: Number of bootstrap iterations (if using bootstrap).
        bootstrap_random_state: Random state for reproducibility.
    """

    confidence_level: float = 0.95
    aggregation_method: AggregationMethod = AggregationMethod.VARIANCE_WEIGHTED
    min_segment_size: int = 30
    bootstrap_iterations: int = 1000
    bootstrap_random_state: Optional[int] = 42


@dataclass
class SegmentEstimate:
    """Individual segment estimate for aggregation.

    Attributes:
        segment_id: Unique identifier for the segment.
        segment_name: Human-readable segment name.
        ate: Average treatment effect estimate for segment.
        ate_std: Standard error of ATE estimate.
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        sample_size: Number of observations in segment.
        cate: Individual-level treatment effects (optional).
    """

    segment_id: int
    segment_name: str
    ate: float
    ate_std: float
    ci_lower: float
    ci_upper: float
    sample_size: int
    cate: Optional[NDArray[np.float64]] = None


@dataclass
class NestedCIResult:
    """Result of nested confidence interval computation.

    Attributes:
        aggregate_ate: Population-level average treatment effect.
        aggregate_ci_lower: Lower bound of nested CI.
        aggregate_ci_upper: Upper bound of nested CI.
        aggregate_std: Standard error of aggregate estimate.
        confidence_level: Confidence level used.
        aggregation_method: Method used for aggregation.
        segment_contributions: Weight/contribution of each segment.
        heterogeneity_measure: Q-statistic for between-segment heterogeneity.
        i_squared: I² statistic (% variation due to heterogeneity).
        tau_squared: Between-study variance (random effects).
        n_segments_included: Number of segments included in aggregation.
        total_sample_size: Total observations across all segments.
        warnings: Any warnings from computation.
    """

    aggregate_ate: float
    aggregate_ci_lower: float
    aggregate_ci_upper: float
    aggregate_std: float
    confidence_level: float
    aggregation_method: str
    segment_contributions: dict = field(default_factory=dict)
    heterogeneity_measure: float = 0.0
    i_squared: float = 0.0
    tau_squared: float = 0.0
    n_segments_included: int = 0
    total_sample_size: int = 0
    warnings: List[str] = field(default_factory=list)


class NestedConfidenceInterval:
    """Computes nested confidence intervals from segment-level estimates.

    This class implements meta-analytic methods for combining treatment
    effect estimates from multiple segments into a single aggregate
    estimate with appropriate confidence intervals.

    The variance-weighted (inverse-variance) method is recommended for
    most applications as it is statistically efficient and handles
    varying segment sizes appropriately.

    Example:
        >>> ci_calculator = NestedConfidenceInterval(
        ...     NestedCIConfig(confidence_level=0.95)
        ... )
        >>> segment_estimates = [
        ...     SegmentEstimate(0, "low", 0.10, 0.02, 0.06, 0.14, 500),
        ...     SegmentEstimate(1, "medium", 0.15, 0.03, 0.09, 0.21, 300),
        ...     SegmentEstimate(2, "high", 0.25, 0.04, 0.17, 0.33, 200),
        ... ]
        >>> result = ci_calculator.compute(segment_estimates)
        >>> print(f"Aggregate ATE: {result.aggregate_ate:.3f}")
        >>> print(f"95% CI: [{result.aggregate_ci_lower:.3f}, {result.aggregate_ci_upper:.3f}]")
    """

    def __init__(self, config: Optional[NestedCIConfig] = None) -> None:
        """Initialize the nested CI calculator.

        Args:
            config: Configuration for CI computation. Uses defaults if None.
        """
        self.config = config or NestedCIConfig()
        self._rng = np.random.default_rng(self.config.bootstrap_random_state)

    def compute(
        self,
        segment_estimates: List[SegmentEstimate],
    ) -> NestedCIResult:
        """Compute nested confidence interval from segment estimates.

        Args:
            segment_estimates: List of segment-level treatment effect estimates.

        Returns:
            NestedCIResult with aggregate estimate and confidence interval.
        """
        warnings: List[str] = []

        # Filter segments by minimum size
        valid_segments = [
            s for s in segment_estimates if s.sample_size >= self.config.min_segment_size
        ]

        n_excluded = len(segment_estimates) - len(valid_segments)
        if n_excluded > 0:
            warnings.append(
                f"Excluded {n_excluded} segments with < {self.config.min_segment_size} samples"
            )

        if len(valid_segments) == 0:
            return NestedCIResult(
                aggregate_ate=0.0,
                aggregate_ci_lower=float("-inf"),
                aggregate_ci_upper=float("inf"),
                aggregate_std=float("inf"),
                confidence_level=self.config.confidence_level,
                aggregation_method=self.config.aggregation_method.value,
                n_segments_included=0,
                total_sample_size=0,
                warnings=["No valid segments for aggregation"],
            )

        if len(valid_segments) == 1:
            seg = valid_segments[0]
            warnings.append("Single segment - no aggregation performed")
            return NestedCIResult(
                aggregate_ate=seg.ate,
                aggregate_ci_lower=seg.ci_lower,
                aggregate_ci_upper=seg.ci_upper,
                aggregate_std=seg.ate_std,
                confidence_level=self.config.confidence_level,
                aggregation_method=self.config.aggregation_method.value,
                segment_contributions={seg.segment_name: 1.0},
                n_segments_included=1,
                total_sample_size=seg.sample_size,
                warnings=warnings,
            )

        # Dispatch to appropriate aggregation method
        method = self.config.aggregation_method
        if method == AggregationMethod.SAMPLE_WEIGHTED:
            result = self._compute_sample_weighted(valid_segments)
        elif method == AggregationMethod.VARIANCE_WEIGHTED:
            result = self._compute_variance_weighted(valid_segments)
        elif method == AggregationMethod.EQUAL:
            result = self._compute_equal_weighted(valid_segments)
        elif method == AggregationMethod.BOOTSTRAP:
            result = self._compute_bootstrap(valid_segments)
        else:
            # Default to variance-weighted
            result = self._compute_variance_weighted(valid_segments)

        # Add heterogeneity measures
        heterogeneity = self._compute_heterogeneity(valid_segments)
        result.heterogeneity_measure = heterogeneity["q_statistic"]
        result.i_squared = heterogeneity["i_squared"]
        result.tau_squared = heterogeneity["tau_squared"]

        # Add metadata
        result.confidence_level = self.config.confidence_level
        result.aggregation_method = self.config.aggregation_method.value
        result.n_segments_included = len(valid_segments)
        result.total_sample_size = sum(s.sample_size for s in valid_segments)
        result.warnings.extend(warnings)

        # Warn if high heterogeneity
        if result.i_squared > 75:
            result.warnings.append(
                f"High heterogeneity (I²={result.i_squared:.1f}%) - "
                "consider segment-specific analysis"
            )
        elif result.i_squared > 50:
            result.warnings.append(f"Moderate heterogeneity (I²={result.i_squared:.1f}%)")

        return result

    def _compute_sample_weighted(
        self,
        segments: List[SegmentEstimate],
    ) -> NestedCIResult:
        """Compute sample-size weighted aggregate.

        Weights each segment by its sample size. Simple but may not
        account for varying precision across segments.
        """
        total_n = sum(s.sample_size for s in segments)
        weights = {s.segment_name: s.sample_size / total_n for s in segments}

        # Weighted mean
        ate = sum(s.ate * s.sample_size for s in segments) / total_n

        # Pooled variance with weights
        # Var(weighted mean) = sum(w_i^2 * var_i)
        weighted_var = sum((s.sample_size / total_n) ** 2 * s.ate_std**2 for s in segments)
        std = np.sqrt(weighted_var)

        # Confidence interval
        z = self._get_z_score()
        ci_lower = ate - z * std
        ci_upper = ate + z * std

        return NestedCIResult(
            aggregate_ate=ate,
            aggregate_ci_lower=ci_lower,
            aggregate_ci_upper=ci_upper,
            aggregate_std=std,
            confidence_level=self.config.confidence_level,
            aggregation_method=AggregationMethod.SAMPLE_WEIGHTED.value,
            segment_contributions=weights,
        )

    def _compute_variance_weighted(
        self,
        segments: List[SegmentEstimate],
    ) -> NestedCIResult:
        """Compute inverse-variance weighted aggregate (fixed effects).

        This is the most efficient estimator when segments are estimating
        the same underlying effect. Uses inverse-variance weighting.
        """
        # Inverse-variance weights
        variances = np.array([s.ate_std**2 for s in segments])

        # Handle zero variance (exact estimates)
        variances = np.where(variances < 1e-10, 1e-10, variances)

        inv_var = 1.0 / variances
        total_inv_var = np.sum(inv_var)
        weights_array = inv_var / total_inv_var

        weights = {s.segment_name: float(w) for s, w in zip(segments, weights_array, strict=False)}

        # Weighted mean
        effects = np.array([s.ate for s in segments])
        ate = float(np.sum(effects * weights_array))

        # Variance of weighted mean
        var = 1.0 / total_inv_var
        std = np.sqrt(var)

        # Confidence interval
        z = self._get_z_score()
        ci_lower = ate - z * std
        ci_upper = ate + z * std

        return NestedCIResult(
            aggregate_ate=ate,
            aggregate_ci_lower=ci_lower,
            aggregate_ci_upper=ci_upper,
            aggregate_std=std,
            confidence_level=self.config.confidence_level,
            aggregation_method=AggregationMethod.VARIANCE_WEIGHTED.value,
            segment_contributions=weights,
        )

    def _compute_equal_weighted(
        self,
        segments: List[SegmentEstimate],
    ) -> NestedCIResult:
        """Compute equal-weighted (simple average) aggregate.

        Treats each segment equally regardless of size or precision.
        May be useful when segments represent distinct populations.
        """
        n = len(segments)
        weights = {s.segment_name: 1.0 / n for s in segments}

        # Simple mean
        effects = np.array([s.ate for s in segments])
        ate = float(np.mean(effects))

        # Standard error of mean
        variances = np.array([s.ate_std**2 for s in segments])
        var = np.sum(variances) / (n**2)
        std = np.sqrt(var)

        # Confidence interval
        z = self._get_z_score()
        ci_lower = ate - z * std
        ci_upper = ate + z * std

        return NestedCIResult(
            aggregate_ate=ate,
            aggregate_ci_lower=ci_lower,
            aggregate_ci_upper=ci_upper,
            aggregate_std=std,
            confidence_level=self.config.confidence_level,
            aggregation_method=AggregationMethod.EQUAL.value,
            segment_contributions=weights,
        )

    def _compute_bootstrap(
        self,
        segments: List[SegmentEstimate],
    ) -> NestedCIResult:
        """Compute bootstrap-aggregated estimate.

        Uses bootstrap resampling of individual-level CATEs when available,
        otherwise falls back to parametric bootstrap of segment estimates.
        """
        n_iter = self.config.bootstrap_iterations

        # Check if we have individual-level CATEs
        has_cate = all(s.cate is not None for s in segments)

        if has_cate:
            # Bootstrap from individual CATEs
            bootstrap_ates = self._bootstrap_from_cates(segments, n_iter)
        else:
            # Parametric bootstrap from segment estimates
            bootstrap_ates = self._parametric_bootstrap(segments, n_iter)

        # Compute statistics from bootstrap distribution
        ate = float(np.mean(bootstrap_ates))
        std = float(np.std(bootstrap_ates, ddof=1))

        # Percentile confidence interval
        alpha = 1 - self.config.confidence_level
        ci_lower = float(np.percentile(bootstrap_ates, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_ates, 100 * (1 - alpha / 2)))

        # Weights are implicit in bootstrap
        n = len(segments)
        weights = {s.segment_name: 1.0 / n for s in segments}

        return NestedCIResult(
            aggregate_ate=ate,
            aggregate_ci_lower=ci_lower,
            aggregate_ci_upper=ci_upper,
            aggregate_std=std,
            confidence_level=self.config.confidence_level,
            aggregation_method=AggregationMethod.BOOTSTRAP.value,
            segment_contributions=weights,
        )

    def _bootstrap_from_cates(
        self,
        segments: List[SegmentEstimate],
        n_iter: int,
    ) -> NDArray[np.float64]:
        """Bootstrap from individual-level CATEs."""
        # Pool all CATEs with segment weights by sample size
        all_cates = []
        segment_ids = []

        for seg in segments:
            if seg.cate is not None:
                all_cates.extend(seg.cate.tolist())
                segment_ids.extend([seg.segment_id] * len(seg.cate))

        all_cates_arr = np.array(all_cates)
        n_total = len(all_cates_arr)

        bootstrap_ates = np.zeros(n_iter)
        for i in range(n_iter):
            # Resample with replacement
            indices = self._rng.choice(n_total, size=n_total, replace=True)
            bootstrap_ates[i] = np.mean(all_cates_arr[indices])

        return bootstrap_ates

    def _parametric_bootstrap(
        self,
        segments: List[SegmentEstimate],
        n_iter: int,
    ) -> NDArray[np.float64]:
        """Parametric bootstrap from segment estimates."""
        bootstrap_ates = np.zeros(n_iter)
        len(segments)

        for i in range(n_iter):
            # Sample from each segment's distribution
            segment_samples = [self._rng.normal(s.ate, s.ate_std) for s in segments]
            # Sample-weighted average
            total_n = sum(s.sample_size for s in segments)
            weights = [s.sample_size / total_n for s in segments]
            bootstrap_ates[i] = sum(
                w * sample for w, sample in zip(weights, segment_samples, strict=False)
            )

        return bootstrap_ates

    def _compute_heterogeneity(
        self,
        segments: List[SegmentEstimate],
    ) -> dict:
        """Compute heterogeneity statistics (Q, I², τ²).

        Returns:
            Dictionary with:
            - q_statistic: Cochran's Q test for heterogeneity
            - i_squared: Percentage of variation due to heterogeneity
            - tau_squared: Between-segment variance (DerSimonian-Laird)
        """
        k = len(segments)
        if k < 2:
            return {"q_statistic": 0.0, "i_squared": 0.0, "tau_squared": 0.0}

        # Inverse-variance weights
        variances = np.array([s.ate_std**2 for s in segments])
        variances = np.where(variances < 1e-10, 1e-10, variances)
        inv_var = 1.0 / variances
        effects = np.array([s.ate for s in segments])

        # Fixed-effect pooled estimate
        theta_fe = np.sum(effects * inv_var) / np.sum(inv_var)

        # Cochran's Q
        q = float(np.sum(inv_var * (effects - theta_fe) ** 2))

        # I² = (Q - df) / Q * 100%
        df = k - 1
        i_squared = max(0.0, (q - df) / q * 100) if q > 0 else 0.0

        # τ² (DerSimonian-Laird estimator)
        c = np.sum(inv_var) - np.sum(inv_var**2) / np.sum(inv_var)
        tau_squared = max(0.0, (q - df) / c) if c > 0 else 0.0

        return {
            "q_statistic": q,
            "i_squared": i_squared,
            "tau_squared": tau_squared,
        }

    def _get_z_score(self) -> float:
        """Get z-score for configured confidence level."""
        from scipy import stats

        alpha = 1 - self.config.confidence_level
        return float(stats.norm.ppf(1 - alpha / 2))

    def compute_random_effects(
        self,
        segments: List[SegmentEstimate],
    ) -> NestedCIResult:
        """Compute random-effects meta-analysis estimate.

        Uses DerSimonian-Laird method to account for between-segment
        heterogeneity. Recommended when I² > 50%.

        Args:
            segments: List of segment-level estimates.

        Returns:
            NestedCIResult with random-effects aggregate.
        """
        warnings: List[str] = []

        # Filter segments
        valid_segments = [s for s in segments if s.sample_size >= self.config.min_segment_size]

        if len(valid_segments) < 2:
            return self.compute(segments)

        # Compute heterogeneity
        het = self._compute_heterogeneity(valid_segments)
        tau_squared = het["tau_squared"]

        # Random-effects weights: 1 / (var_i + tau^2)
        variances = np.array([s.ate_std**2 for s in valid_segments])
        re_var = variances + tau_squared
        re_var = np.where(re_var < 1e-10, 1e-10, re_var)
        inv_var_re = 1.0 / re_var
        total_inv_var = np.sum(inv_var_re)

        weights_array = inv_var_re / total_inv_var
        weights = {
            s.segment_name: float(w) for s, w in zip(valid_segments, weights_array, strict=False)
        }

        # Random-effects pooled estimate
        effects = np.array([s.ate for s in valid_segments])
        ate = float(np.sum(effects * weights_array))

        # Variance of RE estimate
        var = 1.0 / total_inv_var
        std = np.sqrt(var)

        # Confidence interval
        z = self._get_z_score()
        ci_lower = ate - z * std
        ci_upper = ate + z * std

        return NestedCIResult(
            aggregate_ate=ate,
            aggregate_ci_lower=ci_lower,
            aggregate_ci_upper=ci_upper,
            aggregate_std=std,
            confidence_level=self.config.confidence_level,
            aggregation_method="random_effects",
            segment_contributions=weights,
            heterogeneity_measure=het["q_statistic"],
            i_squared=het["i_squared"],
            tau_squared=het["tau_squared"],
            n_segments_included=len(valid_segments),
            total_sample_size=sum(s.sample_size for s in valid_segments),
            warnings=warnings,
        )
