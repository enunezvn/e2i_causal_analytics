"""Hierarchical Analyzer Node for Heterogeneous Optimizer Agent.

This node uses hierarchical nesting (EconML within CausalML segments) to compute
segment-level CATE estimates with nested confidence intervals.

B9.4: Integration with heterogeneous_optimizer agent.

Integration with Uplift Analyzer:
- Uplift provides individual-level targeting scores and segments
- Hierarchical provides segment-level CATE with proper nested CIs
- Combined: Better uncertainty quantification and segment-specific effects
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd

from ..state import HeterogeneousOptimizerState

logger = logging.getLogger(__name__)


class HierarchicalCATEResult(TypedDict):
    """CATE result for a single segment from hierarchical analysis."""

    segment_id: int
    segment_name: str
    n_samples: int
    uplift_range: tuple[float, float]
    cate_mean: Optional[float]
    cate_std: Optional[float]
    cate_ci_lower: Optional[float]
    cate_ci_upper: Optional[float]
    success: bool
    error_message: Optional[str]


class NestedCIResult(TypedDict):
    """Nested confidence interval aggregation result."""

    aggregate_ate: float
    aggregate_ci_lower: float
    aggregate_ci_upper: float
    aggregate_std: float
    confidence_level: float
    aggregation_method: str
    segment_contributions: Dict[str, float]
    i_squared: Optional[float]  # Heterogeneity statistic
    tau_squared: Optional[float]  # Between-segment variance
    n_segments_included: int
    total_sample_size: int


class HierarchicalAnalyzerOutput(TypedDict):
    """Output from hierarchical analyzer node."""

    hierarchical_segment_results: List[HierarchicalCATEResult]
    nested_ci: Optional[NestedCIResult]
    segment_heterogeneity: Optional[float]
    overall_hierarchical_ate: Optional[float]
    overall_hierarchical_ci_lower: Optional[float]
    overall_hierarchical_ci_upper: Optional[float]
    n_segments_analyzed: int
    segmentation_method: str
    estimator_type: str
    hierarchical_latency_ms: int


class HierarchicalAnalyzerNode:
    """Analyze treatment effect heterogeneity using hierarchical nesting.

    This node implements Pattern 4 from the multi-library synergies architecture:
    EconML CATE estimation within CausalML uplift segments.

    Key Features:
    1. Segments data using CausalML uplift scores (quantile, k-means, threshold)
    2. Estimates CATE within each segment using EconML (CausalForest, LinearDML, etc.)
    3. Aggregates segment CATEs with proper nested confidence intervals
    4. Computes heterogeneity statistics (I², τ², Cochran's Q)

    Supports:
    - Multiple segmentation methods: quantile, kmeans, threshold, tree
    - Multiple EconML estimators: causal_forest, linear_dml, dr_learner, ols
    - Multiple aggregation methods: variance_weighted, sample_weighted, bootstrap
    """

    def __init__(
        self,
        n_segments: int = 3,
        segmentation_method: str = "quantile",
        estimator_type: str = "causal_forest",
        min_segment_size: int = 50,
        confidence_level: float = 0.95,
        aggregation_method: str = "variance_weighted",
        timeout_seconds: int = 180,
    ):
        """Initialize Hierarchical Analyzer node.

        Args:
            n_segments: Number of segments to create (default: 3)
            segmentation_method: How to segment: "quantile", "kmeans", "threshold", "tree"
            estimator_type: EconML estimator: "causal_forest", "linear_dml", "dr_learner", "ols"
            min_segment_size: Minimum samples per segment (default: 50)
            confidence_level: CI confidence level (default: 0.95)
            aggregation_method: How to aggregate CATEs: "variance_weighted", "sample_weighted", "bootstrap"
            timeout_seconds: Maximum execution time (default: 180)
        """
        self.n_segments = n_segments
        self.segmentation_method = segmentation_method
        self.estimator_type = estimator_type
        self.min_segment_size = min_segment_size
        self.confidence_level = confidence_level
        self.aggregation_method = aggregation_method
        self.timeout_seconds = timeout_seconds

    async def execute(self, state: HeterogeneousOptimizerState) -> Dict[str, Any]:
        """Execute hierarchical analysis.

        Uses the HierarchicalAnalyzer to compute segment-level CATE estimates
        with nested confidence intervals.

        Args:
            state: Current heterogeneous optimizer state (after uplift analysis)

        Returns:
            Updated state with hierarchical analysis results
        """
        start_time = time.time()

        logger.info(
            "Starting hierarchical analysis",
            extra={
                "node": "hierarchical_analyzer",
                "n_segments": self.n_segments,
                "segmentation_method": self.segmentation_method,
                "estimator_type": self.estimator_type,
                "treatment_var": state.get("treatment_var"),
                "outcome_var": state.get("outcome_var"),
            },
        )

        try:
            # Check prerequisites
            if state.get("status") == "failed":
                logger.warning("Skipping hierarchical analysis - previous step failed")
                return {
                    "warnings": ["Hierarchical analysis skipped due to previous failure"],
                }

            # Get data
            df = await self._get_data(state)
            if df is None or len(df) < self.min_segment_size * 2:
                return {
                    "warnings": [
                        f"Insufficient data for hierarchical analysis "
                        f"(need >= {self.min_segment_size * 2} rows)"
                    ],
                }

            # Prepare features and labels
            X, treatment, y = self._prepare_data(df, state)

            # Get uplift scores from state or compute fresh
            uplift_scores = self._get_uplift_scores(state, len(df))

            # Run hierarchical analysis
            result = await self._run_hierarchical_analysis(X, treatment, y, uplift_scores)

            hierarchical_latency_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "Hierarchical analysis complete",
                extra={
                    "node": "hierarchical_analyzer",
                    "n_segments": result.get("n_segments_analyzed", 0),
                    "overall_ate": result.get("overall_hierarchical_ate"),
                    "heterogeneity": result.get("segment_heterogeneity"),
                    "latency_ms": hierarchical_latency_ms,
                },
            )

            return {
                **result,
                "hierarchical_latency_ms": hierarchical_latency_ms,
                "segmentation_method": self.segmentation_method,
                "estimator_type": self.estimator_type,
            }

        except asyncio.TimeoutError:
            logger.error(
                "Hierarchical analysis timed out",
                extra={"node": "hierarchical_analyzer", "timeout_seconds": self.timeout_seconds},
            )
            return {
                "warnings": [f"Hierarchical analysis timed out after {self.timeout_seconds}s"],
            }
        except ImportError as e:
            logger.warning(
                f"Required library not available for hierarchical analysis: {e}",
                extra={"node": "hierarchical_analyzer"},
            )
            return {
                "warnings": [f"Hierarchical analysis skipped - missing dependency: {e}"],
            }
        except Exception as e:
            logger.error(
                "Hierarchical analysis failed",
                extra={"node": "hierarchical_analyzer", "error": str(e)},
                exc_info=True,
            )
            return {
                "errors": [
                    {
                        "node": "hierarchical_analyzer",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                ],
                "warnings": ["Hierarchical analysis failed - using overall CATE only"],
            }

    async def _get_data(self, state: HeterogeneousOptimizerState) -> Optional[pd.DataFrame]:
        """Get data for hierarchical analysis.

        Uses mock data if data connector not available.
        """
        if hasattr(self, "data_connector") and self.data_connector:
            columns = (
                [state["treatment_var"], state["outcome_var"]]
                + state.get("effect_modifiers", [])
                + state.get("segment_vars", [])
            )
            return await self.data_connector.query(
                source=state["data_source"],
                columns=list(set(columns)),
                filters=state.get("filters"),
            )

        # Generate mock data for testing
        return self._generate_mock_data(state)

    def _generate_mock_data(self, state: HeterogeneousOptimizerState) -> pd.DataFrame:
        """Generate mock data for testing."""
        np.random.seed(42)
        n = 500

        data = {
            state["treatment_var"]: np.random.binomial(1, 0.5, n),
            state["outcome_var"]: np.random.normal(100, 20, n),
        }

        # Add effect modifiers
        for modifier in state.get("effect_modifiers", []):
            data[modifier] = np.random.randn(n)

        # Add segment variables
        for seg_var in state.get("segment_vars", []):
            data[seg_var] = np.random.choice(["A", "B", "C"], n)

        # Add heterogeneous treatment effect
        df = pd.DataFrame(data)
        if state.get("effect_modifiers"):
            treatment_effect = 5.0 + df[state["effect_modifiers"][0]] * 3.0
            df.loc[df[state["treatment_var"]] == 1, state["outcome_var"]] += treatment_effect[
                df[state["treatment_var"]] == 1
            ]

        return df

    def _prepare_data(
        self,
        df: pd.DataFrame,
        state: HeterogeneousOptimizerState,
    ) -> tuple:
        """Prepare data for hierarchical analysis.

        Args:
            df: DataFrame with all columns
            state: Agent state with variable names

        Returns:
            Tuple of (X features DataFrame, treatment array, outcome array)
        """
        treatment = df[state["treatment_var"]].values
        y = df[state["outcome_var"]].values

        # Prepare features (effect modifiers or all numeric columns)
        effect_modifiers = state.get("effect_modifiers", [])
        if effect_modifiers:
            X = df[effect_modifiers].copy()
        else:
            # Use all numeric columns except treatment/outcome
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude = [state["treatment_var"], state["outcome_var"]]
            X = df[[c for c in numeric_cols if c not in exclude]].copy()

        # Encode categorical columns
        for col in X.columns:
            if X[col].dtype == "object" or str(X[col].dtype) == "category":
                categories = X[col].unique()
                cat_to_int = {cat: i for i, cat in enumerate(categories)}
                X[col] = X[col].map(cat_to_int).astype(float)

        return X, treatment, y

    def _get_uplift_scores(
        self,
        state: HeterogeneousOptimizerState,
        n_samples: int,
    ) -> Optional[np.ndarray]:
        """Extract or generate uplift scores for segmentation.

        Args:
            state: Agent state (may contain uplift results)
            n_samples: Number of samples needed

        Returns:
            Uplift scores array or None (will use internal estimation)
        """
        # Check if uplift_by_segment has scores
        uplift_by_segment = state.get("uplift_by_segment")
        if uplift_by_segment:
            # If we have segment-level scores, we can't recover individual scores
            # Let the hierarchical analyzer compute its own
            return None

        # Check for CATE results that could be used as uplift proxy
        cate_by_segment = state.get("cate_by_segment")
        if cate_by_segment:
            # We don't have individual-level scores
            return None

        # No pre-computed scores - let hierarchical analyzer estimate internally
        return None

    async def _run_hierarchical_analysis(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
        uplift_scores: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Run the full hierarchical analysis pipeline.

        Args:
            X: Feature matrix
            treatment: Treatment assignment
            y: Outcome variable
            uplift_scores: Pre-computed uplift scores (optional)

        Returns:
            Dictionary with hierarchical analysis results
        """
        from src.causal_engine.hierarchical import (
            AggregationMethod,
            HierarchicalAnalyzer,
            HierarchicalConfig,
            NestedCIConfig,
            NestedConfidenceInterval,
        )
        from src.causal_engine.hierarchical.analyzer import SegmentationMethod
        from src.causal_engine.hierarchical.nested_ci import SegmentEstimate

        # Map string methods to enums
        segmentation_method_map = {
            "quantile": SegmentationMethod.QUANTILE,
            "kmeans": SegmentationMethod.KMEANS,
            "threshold": SegmentationMethod.THRESHOLD,
            "tree": SegmentationMethod.TREE,
        }

        aggregation_method_map = {
            "variance_weighted": AggregationMethod.VARIANCE_WEIGHTED,
            "sample_weighted": AggregationMethod.SAMPLE_WEIGHTED,
            "equal": AggregationMethod.EQUAL,
            "bootstrap": AggregationMethod.BOOTSTRAP,
        }

        # Create hierarchical config
        hierarchical_config = HierarchicalConfig(
            n_segments=self.n_segments,
            segmentation_method=segmentation_method_map.get(
                self.segmentation_method, SegmentationMethod.QUANTILE
            ),
            min_segment_size=self.min_segment_size,
            estimator_type=self.estimator_type,
            ci_confidence_level=self.confidence_level,
            compute_nested_ci=True,
        )

        # Create analyzer and run with timeout
        analyzer = HierarchicalAnalyzer(hierarchical_config)

        result = await asyncio.wait_for(
            analyzer.analyze(
                X=X,
                treatment=treatment,
                outcome=y,
                uplift_scores=uplift_scores,
            ),
            timeout=self.timeout_seconds,
        )

        if not result.success:
            return {
                "hierarchical_segment_results": [],
                "nested_ci": None,
                "segment_heterogeneity": None,
                "overall_hierarchical_ate": None,
                "overall_hierarchical_ci_lower": None,
                "overall_hierarchical_ci_upper": None,
                "n_segments_analyzed": 0,
                "warnings": result.errors if result.errors else ["Hierarchical analysis failed"],
            }

        # Convert segment results to output format
        segment_results: List[HierarchicalCATEResult] = []
        for seg in result.segment_results:
            segment_results.append(
                HierarchicalCATEResult(
                    segment_id=seg.segment_id,
                    segment_name=seg.segment_name,
                    n_samples=seg.n_samples,
                    uplift_range=seg.uplift_range,
                    cate_mean=seg.cate_mean,
                    cate_std=seg.cate_std,
                    cate_ci_lower=seg.cate_ci_lower,
                    cate_ci_upper=seg.cate_ci_upper,
                    success=seg.success,
                    error_message=seg.error_message,
                )
            )

        # Compute nested CI for aggregate estimate
        nested_ci_result: Optional[NestedCIResult] = None
        if len([s for s in result.segment_results if s.success]) >= 1:
            nested_ci_config = NestedCIConfig(
                confidence_level=self.confidence_level,
                aggregation_method=aggregation_method_map.get(
                    self.aggregation_method, AggregationMethod.VARIANCE_WEIGHTED
                ),
                min_segment_size=self.min_segment_size,
            )
            nested_ci_calc = NestedConfidenceInterval(nested_ci_config)

            # Convert to SegmentEstimate format
            segment_estimates = []
            for seg in result.segment_results:
                if seg.success and seg.cate_mean is not None:
                    segment_estimates.append(
                        SegmentEstimate(
                            segment_id=seg.segment_id,
                            segment_name=seg.segment_name,
                            ate=seg.cate_mean,
                            ate_std=seg.cate_std or 0.01,
                            ci_lower=seg.cate_ci_lower or seg.cate_mean - 0.1,
                            ci_upper=seg.cate_ci_upper or seg.cate_mean + 0.1,
                            sample_size=seg.n_samples,
                            cate=seg.cate_values if hasattr(seg, "cate_values") else None,
                        )
                    )

            if segment_estimates:
                ci_result = nested_ci_calc.compute(segment_estimates)
                nested_ci_result = NestedCIResult(
                    aggregate_ate=ci_result.aggregate_ate,
                    aggregate_ci_lower=ci_result.aggregate_ci_lower,
                    aggregate_ci_upper=ci_result.aggregate_ci_upper,
                    aggregate_std=ci_result.aggregate_std,
                    confidence_level=ci_result.confidence_level,
                    aggregation_method=ci_result.aggregation_method,
                    segment_contributions=ci_result.segment_contributions,
                    i_squared=ci_result.i_squared,
                    tau_squared=ci_result.tau_squared,
                    n_segments_included=ci_result.n_segments_included,
                    total_sample_size=ci_result.total_sample_size,
                )

        return {
            "hierarchical_segment_results": segment_results,
            "nested_ci": nested_ci_result,
            "segment_heterogeneity": result.segment_heterogeneity,
            "overall_hierarchical_ate": result.overall_ate,
            "overall_hierarchical_ci_lower": result.overall_ate_ci_lower,
            "overall_hierarchical_ci_upper": result.overall_ate_ci_upper,
            "n_segments_analyzed": result.n_segments,
        }
