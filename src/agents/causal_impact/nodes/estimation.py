"""Estimation Node - Causal effect estimation using DoWhy/EconML.

Estimates Average Treatment Effect (ATE) and Conditional ATE (CATE).

V4.2 Enhancement: Energy Score-based Estimator Selection
- Replaces single-method estimation with multi-estimator evaluation
- Selects estimator with lowest energy score (best quality)
- Backward compatible: explicit method parameter uses legacy path

F-006 fix (#417): Legacy ``_estimate_*`` methods that previously returned
``np.corrcoef``-based mock ATE values have been REPLACED with thin
delegators to the energy-score selector. On energy-score failure (or any
estimator failure), this module now FAIL-CLOSES by raising
``EstimationError`` instead of silently returning a fake correlation-based
result.

TODO(#354 follow-up): ``_get_data()`` still contains a synthetic-data
trapdoor (``np.random.seed(42)`` + hardcoded HCP/conversion-rate fixtures
at lines ~390-425). Documented in memory ``issue_354_stub_vs_real_estimators_20260521.md``
as one of the 4 silent-fallback trapdoors. Deferred to a future PR per
F-006 brief; we kept it in-place because rewiring data loading is outside
the F-006/F-014 scope.
"""

import logging
import time
from typing import Any, Dict, List, Literal, Optional, cast

import numpy as np
import pandas as pd

from src.agents.causal_impact.state import CausalImpactState, EstimationResult

# V4.2: Energy Score imports
from src.causal_engine.energy_score import (
    EstimatorConfig,
    EstimatorSelector,
    EstimatorSelectorConfig,
    EstimatorType,
    SelectionResult,
    SelectionStrategy,
)

# F-006 fix (#417): structured fail-closed error for legacy method dispatch
from src.causal_engine.errors import EstimationError

logger = logging.getLogger(__name__)


class EstimationNode:
    """Estimates causal effects using DoWhy/EconML.

    Performance target: <30s
    Type: Standard (computation-heavy)

    V4.2 Enhancement: Energy Score-based Estimator Selection
    - Default: Evaluate all estimators, select best by energy score
    - Legacy: Explicit method parameter uses single-estimator path
    - Strategies: first_success, best_energy, ensemble
    """

    # Quality tier thresholds for energy score
    QUALITY_TIERS = {
        "excellent": 0.25,
        "good": 0.45,
        "acceptable": 0.65,
        "poor": 0.80,
        "unreliable": 1.0,
    }

    def __init__(self):
        """Initialize estimation node."""
        self._estimator_selector: Optional[EstimatorSelector] = None

    def _get_quality_tier(self, energy_score: float) -> str:
        """Map energy score to quality tier.

        Args:
            energy_score: Energy score (0-1, lower is better)

        Returns:
            Quality tier: excellent, good, acceptable, poor, unreliable
        """
        for tier, threshold in self.QUALITY_TIERS.items():
            if energy_score <= threshold:
                return tier
        return "unreliable"

    def _get_estimator_selector(
        self,
        strategy: SelectionStrategy = SelectionStrategy.BEST_ENERGY_SCORE,
        restrict_to: Optional[EstimatorType] = None,
    ) -> EstimatorSelector:
        """Get or create EstimatorSelector.

        Args:
            strategy: Selection strategy to use
            restrict_to: If set, the selector evaluates ONLY this estimator
                (legacy ``parameters.method`` compatibility for F-006 #417).
                Otherwise the default chain (CausalForest, LinearDML,
                DRLearner, OLS) is evaluated.

        Returns:
            Configured EstimatorSelector instance
        """
        if restrict_to is not None:
            config = EstimatorSelectorConfig(
                strategy=strategy,
                estimators=[EstimatorConfig(restrict_to, priority=1)],
            )
        else:
            config = EstimatorSelectorConfig(strategy=strategy)
        return EstimatorSelector(config)

    def _select_estimator_with_energy_score(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: List[str],
        strategy: str = "best_energy",
        explicit_method: Optional[str] = None,
    ) -> tuple[EstimationResult, Dict[str, Any], float]:
        """Select best estimator using energy score.

        V4.2 Enhancement: Multi-estimator evaluation and selection.

        F-006 fix (#417): when ``explicit_method`` is set, the selector is
        constrained to that estimator only (matching the legacy single-method
        contract). Otherwise, all configured estimators are evaluated.

        Args:
            data: DataFrame with treatment, outcome, and covariates
            treatment: Treatment variable name
            outcome: Outcome variable name
            adjustment_set: List of adjustment variables
            strategy: Selection strategy (first_success, best_energy, ensemble)
            explicit_method: If set, run only this estimator (legacy compat).

        Returns:
            Tuple of (EstimationResult, selection_result_dict, latency_ms)
        """
        start_time = time.time()

        # Map strategy string to enum
        strategy_map = {
            "first_success": SelectionStrategy.FIRST_SUCCESS,
            "best_energy": SelectionStrategy.BEST_ENERGY_SCORE,
            "ensemble": SelectionStrategy.ENSEMBLE,
        }
        selection_strategy = strategy_map.get(strategy, SelectionStrategy.BEST_ENERGY_SCORE)

        # Get treatment/outcome arrays
        treatment_col = data.get(treatment, data.iloc[:, 0]).values
        outcome_col = data.get(outcome, data.iloc[:, 1]).values

        # Get covariates (use adjustment set if available, else all other columns).
        # PROVENANCE_DROP_COLS keeps the is_synthetic tag out of the design matrix: the
        # all-other-columns branch would otherwise capture it as a constant covariate on
        # an include_synthetic=True validation run (Shard 07 C1).
        from src.repositories.provenance import PROVENANCE_DROP_COLS

        _excluded = [treatment, outcome, *PROVENANCE_DROP_COLS]
        covariate_cols = (
            adjustment_set
            if adjustment_set
            else [c for c in data.columns if c not in _excluded]
        )
        covariates = (
            data[covariate_cols]
            if covariate_cols
            else data.drop(columns=_excluded, errors="ignore")
        )

        # Convert treatment to binary if continuous
        if not np.array_equal(treatment_col, treatment_col.astype(int)):
            # Continuous treatment - binarize at median
            treatment_binary = (treatment_col > np.median(treatment_col)).astype(int)
        else:
            treatment_binary = treatment_col.astype(int)

        # F-006 (#417): when the caller specifies an explicit legacy method,
        # restrict the selector to that single estimator so the result
        # actually reflects the requested method (not whichever the
        # energy-score chain prefers).
        method_to_type = {
            "CausalForestDML": EstimatorType.CAUSAL_FOREST,
            "causal_forest": EstimatorType.CAUSAL_FOREST,
            "LinearDML": EstimatorType.LINEAR_DML,
            "linear_dml": EstimatorType.LINEAR_DML,
            "linear_regression": EstimatorType.OLS,
            "ols": EstimatorType.OLS,
            "drlearner": EstimatorType.DRLEARNER,
            "propensity_score_weighting": EstimatorType.DRLEARNER,
        }
        restrict_to = method_to_type.get(explicit_method) if explicit_method else None
        selector = self._get_estimator_selector(selection_strategy, restrict_to=restrict_to)

        try:
            selection_result: SelectionResult = selector.select(
                treatment=treatment_binary,
                outcome=outcome_col,
                covariates=covariates,
            )
        except Exception as e:
            logger.warning(f"Energy score selection failed: {e}, falling back to legacy")
            # Return fallback - will be handled by caller
            raise

        latency_ms = (time.time() - start_time) * 1000

        # Convert SelectionResult to EstimationResult
        selected = selection_result.selected

        # F-006 fix iter-2 (#417, codex H1): EstimatorSelector returns a
        # success=False EstimatorResult when ALL configured estimators fail
        # (see ``EstimatorSelector._select_best_energy`` in
        # ``src/causal_engine/energy_score/estimator_selector.py``). Without
        # this check we would silently emit ``ate=0.0``, ``ate_ci=(0.0, 0.0)``,
        # ``ate_se=0.0``, ``energy_score=0.0`` — a NEW silent-wrong path that
        # replaces the deleted corrcoef mocks. Fail-closed instead.
        if not selected.success or selected.ate is None:
            failed_estimators = [
                {
                    "estimator": r.estimator_type.value,
                    "error": r.error_message,
                    "error_type": r.error_type,
                }
                for r in selection_result.all_results
                if not r.success
            ]
            raise EstimationError(
                "All configured estimators failed; refusing to report ate=0.0 silent-wrong. "
                f"Selected estimator '{selected.estimator_type.value}' returned "
                f"success={selected.success}, ate={selected.ate}.",
                details={
                    "selected_estimator": selected.estimator_type.value,
                    "selected_success": selected.success,
                    "selected_ate": selected.ate,
                    "n_estimators_attempted": len(selection_result.all_results),
                    "n_succeeded": sum(1 for r in selection_result.all_results if r.success),
                    "failed_estimators": failed_estimators,
                    "explicit_method": explicit_method,
                    "treatment": treatment,
                    "outcome": outcome,
                },
            )

        energy_score = selected.energy_score
        # M-est3: the selector flags requires_review when the best estimator's
        # energy score breaches max_acceptable_energy_score. A breach means the
        # ATE is NOT a clean valid result: force the 'unreliable' tier and mark
        # the result for review rather than reporting it as reliable.
        requires_review = bool(getattr(selection_result, "requires_review", False))
        if requires_review:
            quality_tier = "unreliable"
            logger.warning(
                "Energy-score review gate: selected estimator %s has energy_score=%.4f "
                "exceeding max_acceptable; marking result requires_review (unreliable).",
                selected.estimator_type.value,
                energy_score,
            )
        else:
            quality_tier = self._get_quality_tier(energy_score)

        # Map estimator type to method name
        estimator_to_method = {
            "causal_forest": "CausalForestDML",
            "linear_dml": "LinearDML",
            "drlearner": "linear_regression",  # Map to existing
            "ols": "linear_regression",
        }

        # Cast method to the expected Literal type
        MethodType = Literal[
            "CausalForestDML",
            "LinearDML",
            "linear_regression",
            "propensity_score_weighting",
            "causal_forest",
            "linear_dml",
            "drlearner",
            "ols",
        ]
        method_name = cast(
            MethodType, estimator_to_method.get(selected.estimator_type.value, "CausalForestDML")
        )

        # CausalForestDML produces real CATE estimates per data point → emits
        # heterogeneity-aware segments. Other estimators (LinearDML, DRLearner,
        # OLS) produce a single ATE without per-segment CATE. Map this via
        # ``selected.cate`` non-None + ``estimator_type == CAUSAL_FOREST``.
        is_causal_forest = selected.estimator_type.value in ("causal_forest", "CausalForestDML")
        heterogeneity_detected = bool(is_causal_forest and selected.cate is not None)

        # Build CATE segments from real CATE estimates when available.
        cate_segments: List[Dict[str, Any]] = []
        if heterogeneity_detected and selected.cate is not None:
            cate_arr = np.asarray(selected.cate, dtype=float)
            # Split into high/low halves by CATE magnitude to mirror the
            # legacy two-segment shape; uses REAL CATE means per half (not
            # the deleted hardcoded ate * 1.2 / 0.8 mock multipliers).
            if cate_arr.size >= 2:
                threshold = float(np.median(cate_arr))
                high_mask = cate_arr >= threshold
                low_mask = ~high_mask
                if high_mask.any():
                    cate_segments.append(
                        {
                            "segment": "High CATE",
                            "cate": float(np.mean(cate_arr[high_mask])),
                            "size": int(high_mask.sum()),
                            "description": "Records with CATE at or above median",
                        }
                    )
                if low_mask.any():
                    cate_segments.append(
                        {
                            "segment": "Low CATE",
                            "cate": float(np.mean(cate_arr[low_mask])),
                            "size": int(low_mask.sum()),
                            "description": "Records with CATE below median",
                        }
                    )

        # Iter-4 codex H-iter3-1 + iter-5 codex H-iter4-1 (#417): CI bounds +
        # standard_error must come from the estimator AND be usable. If they
        # are None/missing/non-finite/non-positive/zero-width/inconsistent
        # on a successful EstimatorResult, fail-closed rather than emitting
        # a degenerate result that propagates as silent-wrong evidence into
        # refutation's data_subset / bootstrap scoring.
        import math as _math

        if selected.ate_ci_lower is None or selected.ate_ci_upper is None:
            raise EstimationError(
                "Selected estimator produced an ATE without confidence interval bounds; "
                "refusing to materialize ate_ci=(0.0, 0.0) which would propagate as "
                "silent-wrong evidence into refutation scoring.",
                details={
                    "reason": "missing_ci_bounds_on_success",
                    "selected_estimator": selected.estimator_type.value,
                    "has_ate_ci_lower": selected.ate_ci_lower is not None,
                    "has_ate_ci_upper": selected.ate_ci_upper is not None,
                    "ate": selected.ate,
                    "ate_std": selected.ate_std,
                },
            )
        ate_ci_lower_f = float(selected.ate_ci_lower)
        ate_ci_upper_f = float(selected.ate_ci_upper)
        if not (_math.isfinite(ate_ci_lower_f) and _math.isfinite(ate_ci_upper_f)):
            raise EstimationError(
                "Selected estimator produced non-finite CI bounds; "
                f"ate_ci_lower={ate_ci_lower_f}, ate_ci_upper={ate_ci_upper_f}.",
                details={
                    "reason": "non_finite_ci_bounds_on_success",
                    "selected_estimator": selected.estimator_type.value,
                    "ate_ci_lower": ate_ci_lower_f,
                    "ate_ci_upper": ate_ci_upper_f,
                },
            )
        if ate_ci_lower_f >= ate_ci_upper_f:
            raise EstimationError(
                "Selected estimator produced degenerate / out-of-order CI bounds; "
                f"ate_ci_lower={ate_ci_lower_f} >= ate_ci_upper={ate_ci_upper_f}. "
                "Zero-width or inverted CI propagates as silent-wrong evidence "
                "into refutation scoring (data_subset / bootstrap).",
                details={
                    "reason": "degenerate_ci_bounds_on_success",
                    "selected_estimator": selected.estimator_type.value,
                    "ate_ci_lower": ate_ci_lower_f,
                    "ate_ci_upper": ate_ci_upper_f,
                },
            )
        # Sanity: the reported ATE must lie within the reported CI. If not,
        # the estimator's outputs are internally inconsistent.
        ate_f = float(selected.ate)
        if not (ate_ci_lower_f <= ate_f <= ate_ci_upper_f):
            raise EstimationError(
                "Selected estimator produced an ATE outside its own confidence interval; "
                f"ate={ate_f}, ci=[{ate_ci_lower_f}, {ate_ci_upper_f}].",
                details={
                    "reason": "ate_outside_ci_on_success",
                    "selected_estimator": selected.estimator_type.value,
                    "ate": ate_f,
                    "ate_ci_lower": ate_ci_lower_f,
                    "ate_ci_upper": ate_ci_upper_f,
                },
            )

        if selected.ate_std is None:
            raise EstimationError(
                "Selected estimator produced an ATE without a standard error; "
                "refusing to materialize ate_se=0.0 which would skip uncertainty "
                "downstream.",
                details={
                    "reason": "missing_standard_error_on_success",
                    "selected_estimator": selected.estimator_type.value,
                    "ate": selected.ate,
                },
            )
        ate_std_f = float(selected.ate_std)
        if not _math.isfinite(ate_std_f) or ate_std_f <= 0.0:
            raise EstimationError(
                f"Selected estimator returned unusable standard error: ate_std={ate_std_f}. "
                "Refusing to emit p_value=NaN with statistical_significance=False which "
                "would be a classification without usable uncertainty.",
                details={
                    "reason": "non_positive_or_non_finite_standard_error",
                    "selected_estimator": selected.estimator_type.value,
                    "ate_std": ate_std_f,
                    "ate": ate_f,
                },
            )

        # Iter-3 codex H2 (#417) + iter-5 H-iter4-1: compute the p-value from
        # the estimator's actual uncertainty model (two-sided z-test on
        # ate / ate_std). By this point ate_std has been guarded above to be
        # > 0 and finite, so we can compute unconditionally instead of
        # emitting a NaN-with-False placeholder.
        from scipy import stats as _scipy_stats

        z_score = abs(ate_f) / ate_std_f
        # Two-sided p-value from standard normal: 2 * (1 - Phi(|z|))
        p_value_real = float(2.0 * (1.0 - _scipy_stats.norm.cdf(z_score)))
        statistical_significance_real = p_value_real < 0.05

        result: EstimationResult = {
            "method": method_name,
            "ate": ate_f,
            "ate_ci_lower": ate_ci_lower_f,
            "ate_ci_upper": ate_ci_upper_f,
            "standard_error": ate_std_f,
            "effect_size": self._classify_effect_size(ate_f),
            "statistical_significance": statistical_significance_real,
            "p_value": p_value_real,
            "sample_size": len(data),
            "covariates_adjusted": covariate_cols,
            "heterogeneity_detected": heterogeneity_detected,
            "cate_segments": cate_segments,
            # V4.2: Energy score fields
            "selection_strategy": cast(
                Literal["first_success", "best_energy", "ensemble"], strategy
            ),
            "selected_estimator": selected.estimator_type.value,
            "energy_score": float(energy_score),
            "energy_score_data": {
                "score": float(energy_score),
                "treatment_balance_score": float(
                    selected.energy_score_result.treatment_balance_score
                )
                if selected.energy_score_result
                else 0.0,
                "outcome_fit_score": float(selected.energy_score_result.outcome_fit_score)
                if selected.energy_score_result
                else 0.0,
                "propensity_calibration": float(selected.energy_score_result.propensity_calibration)
                if selected.energy_score_result
                else 0.0,
                "computation_time_ms": float(selected.energy_score_result.computation_time_ms)
                if selected.energy_score_result
                else 0.0,
                "quality_tier": quality_tier,
            },
            "selection_reason": selection_result.selection_reason,
            "energy_score_gap": float(selection_result.energy_score_gap),
            "n_estimators_evaluated": len(selection_result.all_results),
            "n_estimators_succeeded": sum(1 for r in selection_result.all_results if r.success),
            # M-est3: surface the review gate so downstream nodes don't treat a
            # breached (unreliable) ATE as clean valid evidence.
            "requires_review": requires_review,
        }

        # Include all estimator results for logging
        all_results = []
        for r in selection_result.all_results:
            all_results.append(
                {
                    "estimator": r.estimator_type.value,
                    "success": r.success,
                    "energy_score": float(r.energy_score) if r.success else None,
                    "ate": float(r.ate) if r.ate is not None else None,
                    "error": r.error_message if not r.success else None,
                }
            )
        result["all_estimators_evaluated"] = all_results

        # Selection result dict for state
        selection_dict = {
            "selected_estimator": selected.estimator_type.value,
            "energy_score": float(energy_score),
            "quality_tier": quality_tier,
            "strategy": strategy,
            "selection_reason": selection_result.selection_reason,
            "n_evaluated": len(selection_result.all_results),
            "n_succeeded": sum(1 for r in selection_result.all_results if r.success),
            "energy_scores": {k: float(v) for k, v in selection_result.energy_scores.items()},
            "requires_review": requires_review,
        }

        return result, selection_dict, latency_ms

    async def execute(self, state: CausalImpactState) -> Dict:
        """Estimate causal effect.

        V4.2: Energy Score-based Selection (default enabled)
        - If parameters.use_energy_score=False OR parameters.method is set → legacy path
        - Otherwise → multi-estimator evaluation with energy score selection

        Args:
            state: Current workflow state with causal_graph

        Returns:
            Updated state with estimation_result
        """
        start_time = time.time()

        try:
            # Get graph and variables
            causal_graph = state.get("causal_graph")
            if not causal_graph:
                raise ValueError("Causal graph not found in state")

            treatment = causal_graph["treatment_nodes"][0]
            outcome = causal_graph["outcome_nodes"][0]
            adjustment_set = (
                causal_graph["adjustment_sets"][0] if causal_graph["adjustment_sets"] else []
            )

            # Get or generate data
            data = self._get_data(state)

            # V4.2: Check if energy score selection should be used
            parameters = state.get("parameters", {})
            use_energy_score = parameters.get("use_energy_score", True)
            explicit_method = parameters.get("method")
            selection_strategy = parameters.get("selection_strategy", "best_energy")

            # Determine which path to use
            # Energy score path: enabled by default, disabled if explicit method or use_energy_score=False
            use_energy_score_path = use_energy_score and not explicit_method

            # F-006 fix (#417): single REAL estimation path. Whether the
            # caller asked for energy-score selection OR for an explicit
            # legacy method, both paths funnel through
            # ``_select_estimator_with_energy_score``. On failure we raise
            # ``EstimationError`` (caught below) — NEVER silently fall back
            # to the deleted ``np.corrcoef``-based ``_estimate_*`` mocks.

            # Validate explicit method name (preserves the legacy contract
            # that ``parameters.method`` must be a known estimator label).
            _VALID_EXPLICIT_METHODS = {
                "CausalForestDML",
                "LinearDML",
                "linear_regression",
                "propensity_score_weighting",
                "causal_forest",
                "linear_dml",
                "drlearner",
                "ols",
            }
            if explicit_method and explicit_method not in _VALID_EXPLICIT_METHODS:
                raise ValueError(f"Unknown estimation method: {explicit_method}")

            logger.info(
                "Using energy score selection (path=%s, explicit_method=%s, strategy=%s)",
                "energy_score" if use_energy_score_path else "legacy_delegated",
                explicit_method,
                selection_strategy,
            )
            try:
                result, selection_dict, energy_latency_ms = (
                    self._select_estimator_with_energy_score(
                        data,
                        treatment,
                        outcome,
                        adjustment_set,
                        selection_strategy,
                        explicit_method=explicit_method,
                    )
                )
            except Exception as e:
                # F-006 fail-closed: re-raise as EstimationError so the caller
                # sees a structured failure, not a silently-wrong corrcoef ATE.
                # Caught by the outer ``except`` below and surfaced as
                # status=failed + error_message.
                raise EstimationError(
                    f"Energy-score estimator selection failed for method="
                    f"{explicit_method or selection_strategy!r}; refusing silent fallback.",
                    details={
                        "explicit_method": explicit_method,
                        "selection_strategy": selection_strategy,
                        "treatment": treatment,
                        "outcome": outcome,
                        "adjustment_set": adjustment_set,
                    },
                    original_error=e,
                ) from e

            # When the caller specified an explicit legacy method, override
            # the ``method`` field in the result so downstream code (chat UI,
            # database persistence) still sees the requested method name.
            if explicit_method:
                # Preserve the explicit method label while keeping the real
                # underlying ATE / CI / SE from the energy-score selector.
                result["method"] = cast(
                    Literal[
                        "CausalForestDML",
                        "LinearDML",
                        "linear_regression",
                        "propensity_score_weighting",
                        "causal_forest",
                        "linear_dml",
                        "drlearner",
                        "ols",
                    ],
                    explicit_method,
                )

            latency_ms = (time.time() - start_time) * 1000

            return {
                **state,
                "estimation_result": result,
                "estimation_latency_ms": latency_ms,
                "current_phase": "refuting",
                "status": "computing",
                # V4.2: Energy score state fields
                "energy_score_enabled": use_energy_score_path,
                "selection_strategy": selection_strategy,
                "estimator_selection_result": selection_dict,
                "energy_score_latency_ms": energy_latency_ms,
                "best_energy_score": result.get("energy_score"),
                "energy_score_quality_tier": result.get("energy_score_data", {}).get(
                    "quality_tier", "unreliable"
                ),
                # Passthrough data for refutation node
                "estimation_data": data,
            }

        except EstimationError as ee:
            # F-006 fail-closed: structured estimation error → chat UI surfaces
            # as "Service unavailable, retry" rather than silent-wrong corrcoef.
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Estimation fail-closed (no mock fallback): {ee.message}",
                extra={"details": ee.details},
            )
            errors = [{"phase": "estimation", "message": ee.message, "details": ee.details}]
            return {
                **state,
                "estimation_error": ee.message,
                "estimation_error_details": ee.details,
                "estimation_latency_ms": latency_ms,
                "status": "failed",
                "error_message": f"Estimation failed: {ee.message}",
                "errors": errors,
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            # Contract: accumulate errors using operator.add
            errors = [{"phase": "estimation", "message": str(e)}]
            return {
                **state,
                "estimation_error": str(e),
                "estimation_latency_ms": latency_ms,
                "status": "failed",
                "error_message": f"Estimation failed: {e}",
                "errors": errors,  # Contract error accumulator
            }

    def _get_data(self, state: CausalImpactState) -> pd.DataFrame:
        """Get data for estimation.

        F-006 fix iter-2 (#417, codex H2): the previous unconditional
        synthetic-data fallback was a silent-wrong path — real estimators
        would produce polished causal answers over fabricated HCP/conversion
        data, then refutation tests ran against the same fake frame.

        Resolution order:
        1. ``state['data_cache']['estimation_data']`` (real data passthrough
           from upstream nodes / repositories).
        2. ``state['data_source'] == 'synthetic'`` → seeded synthetic data
           for tests + developer fixtures. The synthetic path is now
           OPT-IN via explicit ``data_source='synthetic'`` rather than the
           default; production callers that omit ``data_source`` fail-closed.
        3. Otherwise, raise ``EstimationError``.

        Args:
            state: Workflow state with potential data_cache + data_source

        Returns:
            DataFrame with treatment, outcome, and covariates

        Raises:
            EstimationError: when no real data and ``data_source`` is not
                explicitly set to ``"synthetic"`` (fail-closed; no silent
                fabrication for production calls).
        """
        # Check cache first (real data passthrough)
        data_cache = state.get("data_cache", {})
        if "estimation_data" in data_cache:
            return data_cache["estimation_data"]

        # Synthetic path is opt-in via explicit data_source flag.
        # This preserves the testing/dev workflow that relies on
        # ``data_source='synthetic'`` while making the production default
        # fail-closed (no silent fabrication).
        data_source = state.get("data_source")
        if data_source != "synthetic":
            raise EstimationError(
                "Estimation requires data; no estimation_data in data_cache and "
                "data_source != 'synthetic'. Provide real data via "
                "state['data_cache']['estimation_data'] or explicitly set "
                "state['data_source'] = 'synthetic' for testing fixtures.",
                details={
                    "reason": "no_real_data_no_synthetic_optin",
                    "data_source": data_source,
                    "has_data_cache_key": "estimation_data" in data_cache,
                },
            )

        # Generate synthetic data for testing (data_source='synthetic' opt-in).
        # TODO(#354 follow-up): replace this branch with a real repository
        # query when the data-loading surface is built out. See memory
        # ``issue_354_stub_vs_real_estimators_20260521.md`` for the broader
        # silent-fallback trapdoor inventory.
        np.random.seed(42)
        n = 1000

        # Covariates (confounders)
        geographic_region = np.random.choice(["Northeast", "South", "West"], n)
        hcp_specialty = np.random.choice(["Oncology", "Cardiology", "Endocrinology"], n)

        # Convert to numeric for estimation
        region_numeric = (geographic_region == "South").astype(int)
        specialty_numeric = (hcp_specialty == "Oncology").astype(int)

        # Treatment (influenced by confounders)
        hcp_engagement_level = (
            0.3 * region_numeric + 0.2 * specialty_numeric + np.random.normal(0, 0.5, n)
        )

        # Outcome (influenced by treatment and confounders)
        patient_conversion_rate = (
            0.5 * hcp_engagement_level
            + 0.2 * region_numeric
            + 0.1 * specialty_numeric
            + np.random.normal(0, 0.3, n)
        )

        data = pd.DataFrame(
            {
                "hcp_engagement_level": hcp_engagement_level,
                "patient_conversion_rate": patient_conversion_rate,
                "geographic_region": region_numeric,
                "hcp_specialty": specialty_numeric,
            }
        )

        return data

    # ========================================================================
    # F-006 (#417): The legacy ``_estimate_causal_forest``,
    # ``_estimate_linear_dml``, ``_estimate_linear_regression``, and
    # ``_estimate_propensity_weighting`` methods that returned
    # ``np.corrcoef(treatment, outcome) * np.std(outcome)`` plus a hardcoded
    # ``ate_se`` value (0.05/0.06/0.07/0.08) have been DELETED.
    #
    # All four methods funneled into ``execute`` via the
    # ``if method == "CausalForestDML": ...`` dispatch which was reached
    # whenever ``parameters.use_energy_score=False`` OR
    # ``parameters.method`` was explicitly set. ``execute`` now routes BOTH
    # cases through ``_select_estimator_with_energy_score`` (real econml /
    # EconML CausalForestDML / LinearDML / DRLearner / OLS) and raises
    # ``EstimationError`` on selection failure — eliminating the
    # silent-fallback path that returned plausible-range corrcoef ATEs.
    #
    # Per ``CLAUDE.md`` §"CRITICAL — Anti-Mocking & Verification Discipline":
    # mock surfaces with zero non-test production consumers must be DELETED.
    # Consumer grep at commit time confirmed only ``execute`` (internal
    # dispatch, now removed) consumed these methods.
    # ========================================================================

    def _classify_effect_size(self, ate: float) -> str:
        """Classify effect size as small/medium/large.

        Args:
            ate: Average treatment effect

        Returns:
            "small", "medium", or "large"
        """
        abs_ate = abs(ate)

        if abs_ate < 0.2:
            return "small"
        elif abs_ate < 0.5:
            return "medium"
        else:
            return "large"


# Standalone function for LangGraph integration
async def estimate_causal_effect(state: CausalImpactState) -> Dict:
    """Estimate causal effect (standalone function).

    Args:
        state: Current workflow state

    Returns:
        Updated state with estimation_result
    """
    node = EstimationNode()
    return await node.execute(state)
