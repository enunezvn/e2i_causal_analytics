"""Uplift Analyzer Node for Heterogeneous Optimizer Agent.

This node uses CausalML uplift models to calculate individual-level uplift scores
and metrics (AUUC, Qini) that complement the EconML CATE estimates.

Integration with CATE Estimator:
- CATE provides population-level average treatment effects by segment
- Uplift provides individual-level targeting scores
- Combined: Better segment prioritization and targeting precision
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, TypedDict, cast

import numpy as np
import pandas as pd

from ..cross_validation import compute_cross_library_validation, describe_ordering
from ..design import binarize_treatment, sanitize_effect_modifiers
from ..state import HeterogeneousOptimizerState

logger = logging.getLogger(__name__)


class UpliftScoreResult(TypedDict):
    """Individual uplift scoring result for a segment."""

    segment_name: str
    segment_value: str
    mean_uplift_score: float
    uplift_score_std: float
    auuc: Optional[float]
    qini_coefficient: Optional[float]
    top_10_pct_lift: float  # Lift in top 10% vs random
    sample_size: int


class UpliftAnalyzerOutput(TypedDict):
    """Output from uplift analyzer node."""

    uplift_by_segment: Dict[str, List[UpliftScoreResult]]
    overall_auuc: Optional[float]
    overall_qini: Optional[float]
    targeting_efficiency: float  # 0-1, how well model targets responders
    model_type_used: str
    uplift_latency_ms: int


def _control_label(treatment: np.ndarray) -> str:
    """Stringified control-group label for CausalML's ``control_name``.

    ``BaseUpliftModel.fit`` stringifies treatment values with ``str(t)`` and derives
    treatment groups as ``[str(g) for g in sorted(unique(treatment)) if str(g) != control_name]``.
    For CausalML to recognize the control group, ``control_name`` must equal one of
    those strings. Our substrate is binary {0,1} with 0=control, so the control is the
    smallest distinct value; return ``str(...)`` of it (same encoding base.fit uses).
    """
    vals = np.unique(np.asarray(treatment))
    return str(vals[0]) if len(vals) else "0"


class UpliftAnalyzerNode:
    """Analyze treatment effect heterogeneity using CausalML uplift models.

    This node complements the CATE estimator by:
    1. Computing individual-level uplift scores
    2. Evaluating model quality with AUUC and Qini metrics
    3. Identifying high-value targeting segments

    Supports:
    - UpliftRandomForest (CausalML)
    - UpliftGradientBoosting (CausalML with meta-learners)
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 50,
        max_depth: int = 5,
        use_propensity: bool = True,
        timeout_seconds: int = 120,
        data_connector: Any = None,
    ):
        """Initialize Uplift Analyzer node.

        Args:
            model_type: "random_forest" or "gradient_boosting"
            n_estimators: Number of trees/estimators
            max_depth: Maximum tree depth
            use_propensity: Whether to use propensity scores
            timeout_seconds: Maximum execution time
            data_connector: Shared data connector. The graph factory passes the
                SAME instance used by cate_estimator/hierarchical_analyzer so all
                data nodes read one substrate. May be None when the caller
                supplies ``tier0_data`` instead (the page's server-side-loaded
                gold-standard frame). Previously absent — the node was unwired and
                a no-connector construction fell straight to the fail-closed
                RuntimeError (codex HIGH-1).
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.use_propensity = use_propensity
        self.timeout_seconds = timeout_seconds
        self.data_connector = data_connector

    async def execute(self, state: HeterogeneousOptimizerState) -> Dict[str, Any]:
        """Execute uplift analysis.

        Uses the CausalML uplift module to compute individual-level
        uplift scores and targeting metrics.

        Args:
            state: Current heterogeneous optimizer state (after CATE estimation)

        Returns:
            Updated state with uplift analysis results
        """
        start_time = time.time()

        logger.info(
            "Starting uplift analysis",
            extra={
                "node": "uplift_analyzer",
                "model_type": self.model_type,
                "treatment_var": state.get("treatment_var"),
                "outcome_var": state.get("outcome_var"),
            },
        )

        try:
            # Check if we have data from CATE estimator
            if state.get("status") == "failed":
                logger.warning("Skipping uplift analysis - CATE estimation failed")
                return {
                    "warnings": ["Uplift analysis skipped due to CATE failure"],
                }

            # Get data for uplift modeling
            df = await self._get_data(state)
            if df is None or len(df) < 100:
                return {
                    "warnings": ["Insufficient data for uplift modeling (need >= 100 rows)"],
                }

            # Prepare features and labels
            X, treatment, y = self._prepare_data(df, state)

            # Fit uplift model and get scores
            uplift_scores, model_info = await self._fit_uplift_model(X, treatment, y)

            # CausalML uplift models return per-treatment-group columns
            # (n_samples, n_groups). Reduce to the single treatment-vs-control
            # column so every downstream consumer (metrics, segments, targeting)
            # aligns 1:1 with treatment/y (length n). Without this the metric
            # functions flatten() inflates the array to n*n_groups and indexes the
            # n-length treatment array out of bounds — failing uplift on every run
            # (see test_uplift_binary_treatment_produces_metrics_not_out_of_bounds).
            uplift_scores = np.asarray(uplift_scores)
            if uplift_scores.ndim > 1:
                uplift_scores = uplift_scores[:, -1]

            # Calculate metrics
            metrics = self._calculate_metrics(uplift_scores, treatment, y)

            # Segment-level analysis
            uplift_by_segment = await self._analyze_segments(
                df, uplift_scores, state["segment_vars"]
            )

            # Calculate targeting efficiency
            targeting_efficiency = self._calculate_targeting_efficiency(uplift_scores, treatment, y)

            # Cross-library validation: does this (CausalML) uplift model agree
            # with the EconML CATE estimates on segment direction and ranking?
            # Guarded so a validation bug can never cost the uplift outputs.
            cross_validation = self._cross_library_update(
                state, uplift_by_segment, model_info["model_type"]
            )

            # Honest library provenance: EconML ran upstream iff CATE results
            # exist; CausalML only when a real CausalML model fitted (the
            # ImportError fallback is a plain diff-in-means heuristic).
            libraries = ["econml"] if state.get("cate_by_segment") else []
            if model_info["model_type"] in ("random_forest", "gradient_boosting"):
                libraries.append("causalml")

            uplift_latency_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "Uplift analysis complete",
                extra={
                    "node": "uplift_analyzer",
                    "overall_auuc": metrics.get("auuc"),
                    "overall_qini": metrics.get("qini"),
                    "targeting_efficiency": targeting_efficiency,
                    "latency_ms": uplift_latency_ms,
                },
            )

            return {
                "uplift_by_segment": uplift_by_segment,
                "overall_auuc": metrics.get("auuc"),
                "overall_qini": metrics.get("qini"),
                "targeting_efficiency": targeting_efficiency,
                "model_type_used": model_info["model_type"],
                "uplift_latency_ms": uplift_latency_ms,
                "libraries_executed": libraries,
                **cross_validation,
            }

        except asyncio.TimeoutError:
            logger.error(
                "Uplift analysis timed out",
                extra={"node": "uplift_analyzer", "timeout_seconds": self.timeout_seconds},
            )
            return {
                "warnings": [f"Uplift analysis timed out after {self.timeout_seconds}s"],
            }
        except ImportError as e:
            logger.warning(
                f"CausalML not available for uplift analysis: {e}",
                extra={"node": "uplift_analyzer"},
            )
            return {
                "warnings": ["CausalML not installed - uplift analysis skipped"],
            }
        except RuntimeError:
            # F-013 (codex iter-3 H1): the synthetic-data fail-closed
            # signal from _get_data() must not be downgraded to a soft
            # warning. Re-raise so the graph status reflects the explicit
            # configuration error rather than continuing with CATE-only
            # output.
            raise
        except Exception as e:
            logger.error(
                "Uplift analysis failed",
                extra={"node": "uplift_analyzer", "error": str(e)},
                exc_info=True,
            )
            return {
                "errors": [
                    {
                        "node": "uplift_analyzer",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                ],
                "warnings": ["Uplift analysis failed - continuing with CATE results only"],
            }

    def _cross_library_update(
        self,
        state: HeterogeneousOptimizerState,
        uplift_by_segment: Dict[str, List[UpliftScoreResult]],
        model_type: str,
    ) -> Dict[str, Any]:
        """Compute the EconML↔CausalML agreement state update, never raising.

        On failure or a FAILED verdict the update carries a warning (append
        reducer channel) so the run surfaces the disagreement honestly.
        """
        try:
            update = compute_cross_library_validation(
                state.get("cate_by_segment"),
                cast(Dict[str, List[Dict[str, Any]]], uplift_by_segment),
                uplift_model_type=model_type,
            )
        except Exception as e:  # noqa: BLE001 - validation must not cost uplift outputs
            logger.warning("Cross-library validation errored (non-fatal): %s", e)
            return {
                "cross_library_validation": {
                    "computed": False,
                    "reason": f"validation error: {e}",
                }
            }
        if update.get("validation_passed") is False:
            score = update.get("library_agreement_score") or 0.0
            detail = update.get("cross_library_validation") or {}
            sign = detail.get("sign_agreement")
            direction = f"direction {float(sign):.0%}" if sign is not None else "direction n/a"
            # "ordering agreement 0% on 3 ..." -> "ordering 0% on 3 ..." so the
            # warning reads as components of the one score.
            ordering = describe_ordering(detail).replace("ordering agreement ", "ordering ")
            threshold = float(detail.get("threshold") or 0.0)
            # "only" when the composite fell short; when it met the threshold
            # the failure is the ordering chance floor, which the ordering
            # phrase already states -- "only 70% ... threshold 70%" would read
            # as a contradiction.
            qualifier = "only " if score < threshold else ""
            update["warnings"] = [
                f"Cross-library validation FAILED: EconML and CausalML agree {qualifier}"
                f"{score:.0%} ({direction}; {ordering}); threshold "
                f"{threshold:.0%}; treat targeting conclusions "
                f"with caution."
            ]
        return update

    async def _get_data(self, state: HeterogeneousOptimizerState) -> Optional[pd.DataFrame]:
        """Get data for uplift modeling.

        Uses mock data if data connector not available AND mock fallback is
        explicitly enabled (see ``_mock_connector_allowed``).
        """
        # Priority 1: tier0 passthrough — the route's server-side gold-standard
        # frame (loaded include_synthetic=True + brand filter + continuous banding).
        # When present and carrying treatment+outcome, use it directly: no connector
        # fetch (so we don't double-read the substrate, codex MED-1), no fabrication,
        # and uplift sees the SAME banded frame the CATE/hierarchical nodes consume.
        # #1734: resolved via the frame-registry handle (state carries only
        # tier0_frame_ref); a direct in-dict frame is honored for non-graph callers.
        from src.utils.frame_registry import resolve_state_frame

        tier0 = resolve_state_frame(state)
        if isinstance(tier0, pd.DataFrame) and not tier0.empty:
            needed = {state["treatment_var"], state["outcome_var"]}
            if needed.issubset(set(tier0.columns)):
                logger.info(
                    "Uplift using tier0_data passthrough (%d rows)",
                    len(tier0),
                    extra={"node": "uplift_analyzer", "data_source": "tier0_passthrough"},
                )
                return tier0

        # Priority 2: shared data connector (real substrate).
        if self.data_connector:
            columns = (
                [state["treatment_var"], state["outcome_var"]]
                + state["effect_modifiers"]
                + state["segment_vars"]
            )
            return await self.data_connector.query(
                source=state["data_source"],
                columns=list(set(columns)),
                filters=state.get("filters"),
            )

        # F-013 (codex iter-2 H1): mirror the cate_estimator policy here so
        # this exported node cannot silently fabricate data if it's wired
        # into the graph later or used directly. Import lazily to avoid a
        # dependency cycle.
        from src.agents.heterogeneous_optimizer.nodes.cate_estimator import (
            _mock_connector_allowed,
        )

        if not _mock_connector_allowed():
            raise RuntimeError(
                "UpliftAnalyzerNode has no data_connector and synthetic-data "
                "fallback is disabled (E2I_ALLOW_MOCK_CONNECTOR=1 or "
                "ENVIRONMENT in {development,dev,test,testing,local} required "
                "for mock). Attach a real data_connector or set the env-gate "
                "explicitly for offline development."
            )

        logger.warning(
            "Using synthetic mock data in UpliftAnalyzerNode "
            "(E2I_ALLOW_MOCK_CONNECTOR/ENVIRONMENT dev opt-in active) — "
            "DO NOT use this output for any production decision.",
            extra={"node": "uplift_analyzer", "data_source": "mock"},
        )
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

        # Add treatment effect heterogeneity
        df = pd.DataFrame(data)
        treatment_effect = 5.0 + df[state["effect_modifiers"][0]] * 2.0
        df.loc[df[state["treatment_var"]] == 1, state["outcome_var"]] += treatment_effect[
            df[state["treatment_var"]] == 1
        ]

        return df

    def _prepare_data(
        self,
        df: pd.DataFrame,
        state: HeterogeneousOptimizerState,
    ) -> tuple:
        """Prepare data for uplift modeling.

        Args:
            df: DataFrame with all columns
            state: Agent state with variable names

        Returns:
            Tuple of (X features, treatment, outcome)
        """
        # Get treatment and outcome
        treatment = df[state["treatment_var"]].values
        y = df[state["outcome_var"]].values

        # Binarize a continuous treatment at the median — the SAME rule the CATE
        # and hierarchical nodes apply (design.binarize_treatment). Before wave 53
        # this node alone handed CausalML the raw score: live seg_05f29d1b3295 fit
        # 27 "treatment groups" with control "16.0", and the retained column was
        # the uplift of score 42 vs 16 — a different estimand from EconML's, so
        # the cross-library check compared apples to oranges (9% "agreement").
        treatment, binarized = binarize_treatment(treatment)
        if binarized is not None:
            logger.info(
                f"Binarized continuous treatment at median={binarized['median_threshold']:.2f}",
                extra={
                    "node": "uplift_analyzer",
                    "treatment_var": state["treatment_var"],
                    **binarized,
                },
            )

        # Prepare features (effect modifiers) — never the treatment, the outcome
        # or a provenance column (wave 53 / Shard 07 C2).
        effect_modifiers, dropped_modifiers = sanitize_effect_modifiers(state)
        if dropped_modifiers:
            logger.warning(
                "Dropped effect modifiers that must not enter the uplift design matrix",
                extra={
                    "node": "uplift_analyzer",
                    "dropped": dropped_modifiers,
                    "treatment_var": state["treatment_var"],
                    "outcome_var": state["outcome_var"],
                },
            )
        X = df[effect_modifiers].copy()

        # Encode categorical columns
        for col in X.columns:
            if X[col].dtype == "object" or str(X[col].dtype) == "category":
                categories = X[col].unique()
                cat_to_int = {cat: i for i, cat in enumerate(categories)}
                X[col] = X[col].map(cat_to_int).astype(float)

        return X, treatment, y

    async def _fit_uplift_model(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> tuple:
        """Fit uplift model and return scores.

        Args:
            X: Feature matrix
            treatment: Treatment assignment
            y: Outcome variable

        Returns:
            Tuple of (uplift_scores, model_info)
        """
        model_info = {"model_type": self.model_type, "fitted": False}

        try:
            if self.model_type == "gradient_boosting":
                uplift_scores = await self._fit_gradient_boosting(X, treatment, y)
                model_info["model_type"] = "gradient_boosting"
            else:
                uplift_scores = await self._fit_random_forest(X, treatment, y)
                model_info["model_type"] = "random_forest"

            model_info["fitted"] = True
            return uplift_scores, model_info

        except ImportError:
            # Fall back to simple difference in means estimator
            logger.warning("CausalML not available, using simple uplift estimator")
            uplift_scores = self._simple_uplift_estimate(X, treatment, y)
            model_info["model_type"] = "simple_diff"
            model_info["fitted"] = True
            return uplift_scores, model_info

    async def _fit_random_forest(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Fit UpliftRandomForest from our CausalML module.

        Args:
            X: Features
            treatment: Treatment assignment
            y: Outcome

        Returns:
            Uplift scores array
        """
        from src.causal_engine.uplift import UpliftConfig, UpliftRandomForest

        config = UpliftConfig(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=max(10, len(X) // 50),
            random_state=42,
            # CausalML needs the control group's (stringified) label to match an
            # actual treatment value. Our binary treatment is {0,1} with 0=control;
            # leaving the default "control" matched NEITHER, so CausalML treated both
            # 0 and 1 as treatments and returned a degenerate (n, 2) prediction.
            control_name=_control_label(treatment),
        )

        model = UpliftRandomForest(config)

        # Fit with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(
                model.estimate,
                X,
                treatment,
                y.astype(float),
            ),
            timeout=self.timeout_seconds,
        )

        if result.success and result.uplift_scores is not None:
            return result.uplift_scores
        else:
            raise RuntimeError(f"Uplift estimation failed: {result.error_message}")

    async def _fit_gradient_boosting(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Fit UpliftGradientBoosting from our CausalML module.

        Args:
            X: Features
            treatment: Treatment assignment
            y: Outcome

        Returns:
            Uplift scores array
        """
        from src.causal_engine.uplift import (
            GradientBoostingMetaLearner,
            GradientBoostingUpliftConfig,
            UpliftGradientBoosting,
        )

        config = GradientBoostingUpliftConfig(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            meta_learner=GradientBoostingMetaLearner.T_LEARNER,
            use_xgboost=False,  # Use sklearn for compatibility
            random_state=42,
            # See _fit_random_forest: the control label must match an actual
            # treatment value (binary {0,1}, 0=control), not the default "control".
            control_name=_control_label(treatment),
        )

        model = UpliftGradientBoosting(config)

        # Fit with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(
                model.estimate,
                X,
                treatment,
                y.astype(float),
            ),
            timeout=self.timeout_seconds,
        )

        if result.success and result.uplift_scores is not None:
            return result.uplift_scores
        else:
            raise RuntimeError(f"Uplift estimation failed: {result.error_message}")

    def _simple_uplift_estimate(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Simple uplift estimator (difference in means by feature bins).

        Fallback when CausalML is not available.
        """
        # Use first feature for simple binning
        feature = X.iloc[:, 0].values
        bins = np.percentile(feature, [0, 25, 50, 75, 100])
        bin_indices = np.digitize(feature, bins[1:-1])

        uplift_scores = np.zeros(len(X))

        for bin_idx in range(4):
            mask = bin_indices == bin_idx
            if mask.sum() > 0:
                treated_mask = mask & (treatment == 1)
                control_mask = mask & (treatment == 0)

                if treated_mask.sum() > 0 and control_mask.sum() > 0:
                    uplift = y[treated_mask].mean() - y[control_mask].mean()
                    uplift_scores[mask] = uplift

        return uplift_scores

    def _calculate_metrics(
        self,
        uplift_scores: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Optional[float]]:
        """Calculate uplift model metrics (AUUC, Qini).

        Args:
            uplift_scores: Predicted uplift scores
            treatment: Treatment assignment
            y: Actual outcomes

        Returns:
            Dictionary with auuc and qini metrics
        """
        try:
            from src.causal_engine.uplift.metrics import (
                auuc as calculate_auuc,
            )
            from src.causal_engine.uplift.metrics import (
                qini_coefficient,
            )

            auuc = calculate_auuc(uplift_scores, treatment, y)
            qini = qini_coefficient(uplift_scores, treatment, y)

            return {"auuc": auuc, "qini": qini}

        except Exception as e:
            logger.warning(f"Could not calculate uplift metrics: {e}")
            return {"auuc": None, "qini": None}

    async def _analyze_segments(
        self,
        df: pd.DataFrame,
        uplift_scores: np.ndarray,
        segment_vars: List[str],
    ) -> Dict[str, List[UpliftScoreResult]]:
        """Analyze uplift scores by segment.

        Args:
            df: Original DataFrame
            uplift_scores: Individual uplift scores
            segment_vars: Segment variable names

        Returns:
            Uplift results grouped by segment
        """
        df = df.copy()
        df["_uplift_score"] = uplift_scores

        results: Dict[str, List[UpliftScoreResult]] = {}

        for seg_var in segment_vars:
            segment_results: List[UpliftScoreResult] = []

            for seg_value in df[seg_var].unique():
                mask = df[seg_var] == seg_value
                segment_scores = df.loc[mask, "_uplift_score"]

                if len(segment_scores) < 10:
                    continue

                # Calculate top 10% lift
                top_10_threshold = np.percentile(segment_scores, 90)
                top_10_mean = segment_scores[segment_scores >= top_10_threshold].mean()
                overall_mean = segment_scores.mean()
                top_10_lift = (
                    (top_10_mean - overall_mean) / abs(overall_mean) if overall_mean != 0 else 0.0
                )

                segment_results.append(
                    UpliftScoreResult(
                        segment_name=seg_var,
                        segment_value=str(seg_value),
                        mean_uplift_score=float(segment_scores.mean()),
                        uplift_score_std=float(segment_scores.std()),
                        auuc=None,  # Per-segment AUUC requires outcome data
                        qini_coefficient=None,
                        top_10_pct_lift=float(top_10_lift),
                        sample_size=len(segment_scores),
                    )
                )

            # Sort by mean uplift score (highest first)
            segment_results.sort(key=lambda x: x["mean_uplift_score"], reverse=True)
            results[seg_var] = segment_results

        return results

    def _calculate_targeting_efficiency(
        self,
        uplift_scores: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Calculate targeting efficiency metric.

        Measures how well the uplift model identifies true responders.
        Returns 0-1 where 1 = perfect targeting.

        Args:
            uplift_scores: Predicted uplift scores
            treatment: Treatment assignment
            y: Actual outcomes

        Returns:
            Targeting efficiency (0-1)
        """
        # Sort by uplift score
        sorted_indices = np.argsort(uplift_scores)[::-1]

        # Calculate cumulative treated outcomes in top 20%
        top_20_pct_idx = int(len(uplift_scores) * 0.2)
        top_20_indices = sorted_indices[:top_20_pct_idx]

        # Among top 20%, calculate average outcome for treated
        treated_in_top_20 = treatment[top_20_indices] == 1
        if treated_in_top_20.sum() == 0:
            return 0.5  # No treated in top 20%

        top_20_treated_outcome = y[top_20_indices][treated_in_top_20].mean()

        # Compare to overall treated outcome
        overall_treated_outcome = y[treatment == 1].mean()

        if overall_treated_outcome == 0:
            return 0.5

        # Efficiency: how much better is top 20% vs overall
        efficiency = top_20_treated_outcome / overall_treated_outcome

        # Normalize to 0-1 (cap at 2x as "perfect")
        return cast(float, min(efficiency / 2.0, 1.0))
