"""
Hierarchical Analyzer - B9.1

Orchestrates nested analysis combining CausalML uplift segmentation
with EconML CATE estimation within each segment.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    HierarchicalAnalyzer                          │
    │  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐│
    │  │  CausalML   │     │  Segment    │     │  EconML CATE       ││
    │  │  Uplift     │────▶│  Creation   │────▶│  per Segment       ││
    │  └─────────────┘     └─────────────┘     └─────────────────────┘│
    │                              │                     │            │
    │                              ▼                     ▼            │
    │                     ┌─────────────────────────────────┐         │
    │                     │   Nested Confidence Intervals   │         │
    │                     └─────────────────────────────────┘         │
    └─────────────────────────────────────────────────────────────────┘

Author: E2I Causal Analytics Team
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SegmentationMethod(str, Enum):
    """Methods for creating segments from uplift scores."""

    QUANTILE = "quantile"  # Split by uplift score quantiles
    THRESHOLD = "threshold"  # Fixed thresholds
    KMEANS = "kmeans"  # K-means clustering on uplift scores
    TREE = "tree"  # Decision tree-based segmentation


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical analysis.

    Attributes:
        n_segments: Number of segments to create (for quantile/kmeans)
        segmentation_method: How to create segments from uplift scores
        quantile_thresholds: Custom quantile thresholds (if quantile method)
        min_segment_size: Minimum samples per segment for CATE estimation
        estimator_type: EconML estimator to use within segments
        parallel_segments: Run segment CATE in parallel
        compute_nested_ci: Whether to compute nested confidence intervals
        ci_confidence_level: Confidence level for intervals (default 0.95)
    """

    n_segments: int = 3
    segmentation_method: SegmentationMethod = SegmentationMethod.QUANTILE
    quantile_thresholds: Optional[List[float]] = None
    min_segment_size: int = 50
    estimator_type: str = "causal_forest"  # or linear_dml, drlearner, etc.
    parallel_segments: bool = False
    compute_nested_ci: bool = True
    ci_confidence_level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_segments": self.n_segments,
            "segmentation_method": self.segmentation_method.value,
            "quantile_thresholds": self.quantile_thresholds,
            "min_segment_size": self.min_segment_size,
            "estimator_type": self.estimator_type,
            "parallel_segments": self.parallel_segments,
            "compute_nested_ci": self.compute_nested_ci,
            "ci_confidence_level": self.ci_confidence_level,
        }


@dataclass
class SegmentResult:
    """Result for a single segment's CATE analysis.

    Attributes:
        segment_id: Identifier for the segment
        segment_name: Human-readable segment name (e.g., "high_uplift")
        n_samples: Number of samples in segment
        uplift_range: (min, max) uplift scores in segment
        cate_mean: Mean CATE within segment (from EconML)
        cate_std: Standard deviation of CATE within segment
        cate_ci_lower: Lower bound of segment CATE CI
        cate_ci_upper: Upper bound of segment CATE CI
        cate_values: Individual CATE estimates (optional)
        estimator_used: EconML estimator that produced the estimate
        success: Whether CATE estimation succeeded
        error_message: Error message if failed
        estimation_time_ms: Time taken for segment estimation
        metadata: Additional segment metadata
    """

    segment_id: int
    segment_name: str
    n_samples: int
    uplift_range: Tuple[float, float]
    cate_mean: Optional[float] = None
    cate_std: Optional[float] = None
    cate_ci_lower: Optional[float] = None
    cate_ci_upper: Optional[float] = None
    cate_values: Optional[NDArray[np.float64]] = None
    estimator_used: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    estimation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "segment_id": self.segment_id,
            "segment_name": self.segment_name,
            "n_samples": self.n_samples,
            "uplift_range": self.uplift_range,
            "cate_mean": self.cate_mean,
            "cate_std": self.cate_std,
            "cate_ci_lower": self.cate_ci_lower,
            "cate_ci_upper": self.cate_ci_upper,
            "estimator_used": self.estimator_used,
            "success": self.success,
            "error_message": self.error_message,
            "estimation_time_ms": self.estimation_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class HierarchicalResult:
    """Result from hierarchical analysis.

    Attributes:
        success: Whether the overall analysis succeeded
        treatment_var: Treatment variable name
        outcome_var: Outcome variable name
        n_total_samples: Total samples analyzed
        n_segments: Number of segments created
        segment_results: Results for each segment
        overall_ate: Aggregate ATE across segments
        overall_ate_ci_lower: Lower bound of aggregate CI
        overall_ate_ci_upper: Upper bound of aggregate CI
        segment_heterogeneity: Measure of between-segment variance
        uplift_model_used: CausalML model used for segmentation
        segmentation_method: Method used to create segments
        total_latency_ms: Total time for hierarchical analysis
        errors: List of errors encountered
        warnings: List of warnings
        metadata: Additional metadata
    """

    success: bool
    treatment_var: str
    outcome_var: str
    n_total_samples: int
    n_segments: int
    segment_results: List[SegmentResult]
    overall_ate: Optional[float] = None
    overall_ate_ci_lower: Optional[float] = None
    overall_ate_ci_upper: Optional[float] = None
    segment_heterogeneity: Optional[float] = None
    uplift_model_used: Optional[str] = None
    segmentation_method: Optional[str] = None
    total_latency_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "treatment_var": self.treatment_var,
            "outcome_var": self.outcome_var,
            "n_total_samples": self.n_total_samples,
            "n_segments": self.n_segments,
            "segment_results": [s.to_dict() for s in self.segment_results],
            "overall_ate": self.overall_ate,
            "overall_ate_ci_lower": self.overall_ate_ci_lower,
            "overall_ate_ci_upper": self.overall_ate_ci_upper,
            "segment_heterogeneity": self.segment_heterogeneity,
            "uplift_model_used": self.uplift_model_used,
            "segmentation_method": self.segmentation_method,
            "total_latency_ms": self.total_latency_ms,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def get_segment_by_name(self, name: str) -> Optional[SegmentResult]:
        """Get segment result by name."""
        for seg in self.segment_results:
            if seg.segment_name == name:
                return seg
        return None

    def get_high_uplift_segment(self) -> Optional[SegmentResult]:
        """Get the high uplift segment."""
        return self.get_segment_by_name("high_uplift")

    def get_segment_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all segments."""
        data = []
        for seg in self.segment_results:
            data.append(
                {
                    "segment": seg.segment_name,
                    "n_samples": seg.n_samples,
                    "uplift_min": seg.uplift_range[0],
                    "uplift_max": seg.uplift_range[1],
                    "cate_mean": seg.cate_mean,
                    "cate_ci_lower": seg.cate_ci_lower,
                    "cate_ci_upper": seg.cate_ci_upper,
                    "success": seg.success,
                }
            )
        return pd.DataFrame(data)


class HierarchicalAnalyzer:
    """Orchestrates hierarchical analysis combining CausalML and EconML.

    This analyzer implements the hierarchical nesting pattern:
    1. Uses CausalML uplift model to identify segments (high/med/low uplift)
    2. Runs EconML CATE estimation within each segment
    3. Combines results with nested confidence intervals

    Example:
        analyzer = HierarchicalAnalyzer()
        result = await analyzer.analyze(
            X=features,
            treatment=treatment_assignment,
            outcome=outcomes,
            treatment_var="marketing_spend",
            outcome_var="conversion_rate",
        )

        # Get segment-level insights
        for seg in result.segment_results:
            print(f"{seg.segment_name}: CATE = {seg.cate_mean:.4f}")
    """

    def __init__(self, config: Optional[HierarchicalConfig] = None):
        """Initialize analyzer.

        Args:
            config: Configuration for hierarchical analysis
        """
        self.config = config or HierarchicalConfig()
        self._uplift_model = None
        self._segment_cate_calculator = None

    async def analyze(
        self,
        X: Union[pd.DataFrame, NDArray[np.float64]],
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        treatment_var: str = "treatment",
        outcome_var: str = "outcome",
        uplift_scores: Optional[NDArray[np.float64]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> HierarchicalResult:
        """Run hierarchical analysis.

        Args:
            X: Feature matrix (n_samples, n_features)
            treatment: Treatment assignment array
            outcome: Outcome array
            treatment_var: Name of treatment variable
            outcome_var: Name of outcome variable
            uplift_scores: Pre-computed uplift scores (optional)
            feature_names: Feature names (inferred from DataFrame if available)

        Returns:
            HierarchicalResult with segment-level CATE estimates
        """
        start_time = time.perf_counter()
        errors: List[str] = []
        warnings: List[str] = []

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            if feature_names is not None:
                X_df = pd.DataFrame(X, columns=feature_names)
            else:
                X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            X_df = X

        n_samples = len(X_df)

        try:
            # Step 1: Get uplift scores (from CausalML or pre-computed)
            if uplift_scores is None:
                logger.info("Computing uplift scores using CausalML...")
                uplift_scores, uplift_model = await self._compute_uplift_scores(
                    X_df, treatment, outcome
                )
                uplift_model_name = uplift_model
            else:
                logger.info("Using pre-computed uplift scores")
                uplift_model_name = "pre_computed"

            # Step 2: Create segments from uplift scores
            logger.info(f"Creating {self.config.n_segments} segments...")
            segment_indices, segment_names = self._create_segments(uplift_scores)

            # Step 3: Run EconML CATE within each segment
            logger.info("Computing CATE within each segment...")
            segment_results = await self._compute_segment_cate(
                X_df, treatment, outcome, uplift_scores, segment_indices, segment_names
            )

            # Step 4: Compute aggregate statistics
            overall_ate, overall_ci_lower, overall_ci_upper = self._aggregate_results(
                segment_results, n_samples
            )

            # Step 5: Compute segment heterogeneity
            heterogeneity = self._compute_heterogeneity(segment_results)

            # Check for segment failures
            failed_segments = [s for s in segment_results if not s.success]
            if failed_segments:
                warnings.append(f"{len(failed_segments)} segment(s) failed CATE estimation")

            elapsed = (time.perf_counter() - start_time) * 1000

            return HierarchicalResult(
                success=True,
                treatment_var=treatment_var,
                outcome_var=outcome_var,
                n_total_samples=n_samples,
                n_segments=len(segment_results),
                segment_results=segment_results,
                overall_ate=overall_ate,
                overall_ate_ci_lower=overall_ci_lower,
                overall_ate_ci_upper=overall_ci_upper,
                segment_heterogeneity=heterogeneity,
                uplift_model_used=uplift_model_name,
                segmentation_method=self.config.segmentation_method.value,
                total_latency_ms=elapsed,
                errors=errors,
                warnings=warnings,
                metadata={
                    "config": self.config.to_dict(),
                    "n_features": X_df.shape[1],
                    "feature_names": list(X_df.columns),
                },
            )

        except Exception as e:
            logger.error(f"Hierarchical analysis failed: {e}")
            elapsed = (time.perf_counter() - start_time) * 1000
            return HierarchicalResult(
                success=False,
                treatment_var=treatment_var,
                outcome_var=outcome_var,
                n_total_samples=n_samples,
                n_segments=0,
                segment_results=[],
                total_latency_ms=elapsed,
                errors=[str(e)],
                warnings=warnings,
            )

    async def _compute_uplift_scores(
        self,
        X: pd.DataFrame,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], str]:
        """Compute uplift scores using CausalML.

        Returns:
            Tuple of (uplift_scores, model_name)
        """
        from ..uplift import UpliftConfig, UpliftRandomForest

        config = UpliftConfig(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=50,
            min_samples_treatment=10,
            n_reg=100,
            control_name="0",
            random_state=42,
        )

        model = UpliftRandomForest(config)
        result = model.estimate(X, treatment, outcome)

        if not result.success:
            raise RuntimeError(f"Uplift estimation failed: {result.error_message}")

        # Handle multi-dimensional uplift scores
        assert result.uplift_scores is not None
        scores = result.uplift_scores
        if len(scores.shape) > 1:
            scores = scores[:, 0]

        return scores, result.model_type.value

    def _create_segments(
        self, uplift_scores: NDArray[np.float64]
    ) -> Tuple[NDArray[np.int_], List[str]]:
        """Create segments from uplift scores.

        Returns:
            Tuple of (segment_indices, segment_names)
        """
        method = self.config.segmentation_method

        if method == SegmentationMethod.QUANTILE:
            return self._segment_by_quantile(uplift_scores)
        elif method == SegmentationMethod.THRESHOLD:
            return self._segment_by_threshold(uplift_scores)
        elif method == SegmentationMethod.KMEANS:
            return self._segment_by_kmeans(uplift_scores)
        else:
            # Default to quantile
            return self._segment_by_quantile(uplift_scores)

    def _segment_by_quantile(
        self, uplift_scores: NDArray[np.float64]
    ) -> Tuple[NDArray[np.int_], List[str]]:
        """Segment by uplift score quantiles."""
        n_segments = self.config.n_segments

        if self.config.quantile_thresholds is not None:
            quantiles = self.config.quantile_thresholds
        else:
            # Equal-sized segments
            quantiles = np.linspace(0, 1, n_segments + 1)[1:-1].tolist()

        # Compute quantile thresholds
        thresholds = np.quantile(uplift_scores, quantiles)

        # Assign segments
        segment_indices = np.digitize(uplift_scores, thresholds)

        # Create segment names
        segment_names = []
        if n_segments == 3:
            segment_names = ["low_uplift", "medium_uplift", "high_uplift"]
        elif n_segments == 4:
            segment_names = ["very_low_uplift", "low_uplift", "high_uplift", "very_high_uplift"]
        else:
            segment_names = [f"segment_{i}" for i in range(n_segments)]

        return segment_indices, segment_names

    def _segment_by_threshold(
        self, uplift_scores: NDArray[np.float64]
    ) -> Tuple[NDArray[np.int_], List[str]]:
        """Segment by fixed thresholds."""
        # Default thresholds if not specified
        if self.config.quantile_thresholds is not None:
            thresholds = self.config.quantile_thresholds
        else:
            # Use median as single threshold
            thresholds = [float(np.median(uplift_scores))]

        segment_indices = np.digitize(uplift_scores, thresholds)
        segment_names = [f"segment_{i}" for i in range(len(thresholds) + 1)]

        return segment_indices, segment_names

    def _segment_by_kmeans(
        self, uplift_scores: NDArray[np.float64]
    ) -> Tuple[NDArray[np.int_], List[str]]:
        """Segment by K-means clustering on uplift scores."""
        from sklearn.cluster import KMeans

        n_segments = self.config.n_segments

        # Reshape for sklearn
        X = uplift_scores.reshape(-1, 1)

        kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        segment_indices = kmeans.fit_predict(X)

        # Order segments by cluster center (low to high uplift)
        centers = kmeans.cluster_centers_.flatten()
        order = np.argsort(centers)
        mapping = {old: new for new, old in enumerate(order)}
        segment_indices = np.array([mapping[i] for i in segment_indices])

        # Generate names based on cluster centers
        if n_segments == 3:
            segment_names = ["low_uplift", "medium_uplift", "high_uplift"]
        else:
            segment_names = [f"segment_{i}" for i in range(n_segments)]

        return segment_indices, segment_names

    async def _compute_segment_cate(
        self,
        X: pd.DataFrame,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        uplift_scores: NDArray[np.float64],
        segment_indices: NDArray[np.int_],
        segment_names: List[str],
    ) -> List[SegmentResult]:
        """Compute EconML CATE within each segment."""
        from .segment_cate import SegmentCATECalculator, SegmentCATEConfig

        calculator_config = SegmentCATEConfig(
            estimator_type=self.config.estimator_type,
            min_samples=self.config.min_segment_size,
            compute_ci=self.config.compute_nested_ci,
            ci_confidence_level=self.config.ci_confidence_level,
        )
        calculator = SegmentCATECalculator(calculator_config)

        segment_results: List[SegmentResult] = []
        unique_segments = sorted(np.unique(segment_indices))

        for seg_id in unique_segments:
            seg_mask = segment_indices == seg_id
            seg_name = segment_names[seg_id] if seg_id < len(segment_names) else f"segment_{seg_id}"

            n_samples = int(np.sum(seg_mask))
            uplift_range = (
                float(np.min(uplift_scores[seg_mask])),
                float(np.max(uplift_scores[seg_mask])),
            )

            # Check minimum segment size
            if n_samples < self.config.min_segment_size:
                logger.warning(
                    f"Segment {seg_name} has {n_samples} samples, "
                    f"below minimum {self.config.min_segment_size}"
                )
                segment_results.append(
                    SegmentResult(
                        segment_id=seg_id,
                        segment_name=seg_name,
                        n_samples=n_samples,
                        uplift_range=uplift_range,
                        success=False,
                        error_message=f"Insufficient samples ({n_samples} < {self.config.min_segment_size})",
                    )
                )
                continue

            # Extract segment data
            X_seg = X.iloc[seg_mask]
            treatment_seg = treatment[seg_mask]
            outcome_seg = outcome[seg_mask]

            # Compute CATE within segment
            cate_result = await calculator.compute(
                X=X_seg,
                treatment=treatment_seg,
                outcome=outcome_seg,
                segment_id=seg_id,
                segment_name=seg_name,
            )

            segment_results.append(
                SegmentResult(
                    segment_id=seg_id,
                    segment_name=seg_name,
                    n_samples=n_samples,
                    uplift_range=uplift_range,
                    cate_mean=cate_result.cate_mean,
                    cate_std=cate_result.cate_std,
                    cate_ci_lower=cate_result.ci_lower,
                    cate_ci_upper=cate_result.ci_upper,
                    cate_values=cate_result.cate_values,
                    estimator_used=cate_result.estimator_used,
                    success=cate_result.success,
                    error_message=cate_result.error_message,
                    estimation_time_ms=cate_result.estimation_time_ms,
                    metadata=cate_result.metadata,
                )
            )

        return segment_results

    def _aggregate_results(
        self,
        segment_results: List[SegmentResult],
        n_total: int,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Aggregate segment-level CATE to overall ATE.

        Uses sample-size weighted average of segment CATEs.
        """
        successful = [s for s in segment_results if s.success and s.cate_mean is not None]

        if not successful:
            return None, None, None

        # Weighted average by sample size
        total_samples = sum(s.n_samples for s in successful)
        if total_samples == 0:
            return None, None, None

        # cate_mean is guaranteed non-None by the filter above
        overall_ate: float = sum(  # type: ignore[assignment]
            (s.cate_mean or 0.0) * s.n_samples / total_samples for s in successful
        )

        # Aggregate confidence intervals using nested CI calculation
        if self.config.compute_nested_ci:
            from .nested_ci import (
                NestedCIConfig,
                NestedConfidenceInterval,
                SegmentEstimate,
            )

            # Convert SegmentResult to SegmentEstimate for nested CI
            segment_estimates = [
                SegmentEstimate(
                    segment_id=s.segment_id,
                    segment_name=s.segment_name,
                    ate=s.cate_mean or 0.0,
                    ate_std=s.cate_std or 0.0,
                    ci_lower=s.cate_ci_lower or (s.cate_mean or 0.0),
                    ci_upper=s.cate_ci_upper or (s.cate_mean or 0.0),
                    sample_size=s.n_samples,
                    cate=s.cate_values,
                )
                for s in successful
            ]

            ci_calculator = NestedConfidenceInterval(
                NestedCIConfig(confidence_level=self.config.ci_confidence_level)
            )

            nested_result = ci_calculator.compute(segment_estimates)
            overall_ci_lower = nested_result.aggregate_ci_lower
            overall_ci_upper = nested_result.aggregate_ci_upper
        else:
            # Simple pooled CI (conservative)
            ci_lower_weighted = sum(
                float(s.cate_ci_lower or s.cate_mean or 0.0) * s.n_samples / total_samples
                for s in successful
            )
            ci_upper_weighted = sum(
                float(s.cate_ci_upper or s.cate_mean or 0.0) * s.n_samples / total_samples
                for s in successful
            )
            overall_ci_lower = ci_lower_weighted
            overall_ci_upper = ci_upper_weighted

        return overall_ate, overall_ci_lower, overall_ci_upper

    def _compute_heterogeneity(self, segment_results: List[SegmentResult]) -> Optional[float]:
        """Compute between-segment heterogeneity.

        Uses coefficient of variation of segment CATEs.
        """
        successful = [s for s in segment_results if s.success and s.cate_mean is not None]

        if len(successful) < 2:
            return None

        cate_values = [float(s.cate_mean) for s in successful if s.cate_mean is not None]
        mean_cate = np.mean(cate_values)
        std_cate = np.std(cate_values)

        if mean_cate == 0:
            return None

        # Coefficient of variation (higher = more heterogeneity)
        return float(std_cate / abs(mean_cate))
