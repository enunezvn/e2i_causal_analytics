"""Estimation Node - Causal effect estimation using DoWhy/EconML.

Estimates Average Treatment Effect (ATE) and Conditional ATE (CATE).

V4.2 Enhancement: Energy Score-based Estimator Selection
- Replaces single-method estimation with multi-estimator evaluation
- Selects estimator with lowest energy score (best quality)
- Backward compatible: explicit method parameter uses legacy path
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.agents.causal_impact.state import CausalImpactState, EstimationResult

# V4.2: Energy Score imports
from src.causal_engine.energy_score import (
    EstimatorSelector,
    EstimatorSelectorConfig,
    SelectionResult,
    SelectionStrategy,
)

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
    ) -> EstimatorSelector:
        """Get or create EstimatorSelector.

        Args:
            strategy: Selection strategy to use

        Returns:
            Configured EstimatorSelector instance
        """
        config = EstimatorSelectorConfig(strategy=strategy)
        return EstimatorSelector(config)

    def _select_estimator_with_energy_score(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: List[str],
        strategy: str = "best_energy",
    ) -> tuple[EstimationResult, Dict[str, Any], float]:
        """Select best estimator using energy score.

        V4.2 Enhancement: Multi-estimator evaluation and selection.

        Args:
            data: DataFrame with treatment, outcome, and covariates
            treatment: Treatment variable name
            outcome: Outcome variable name
            adjustment_set: List of adjustment variables
            strategy: Selection strategy (first_success, best_energy, ensemble)

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

        # Get covariates (use adjustment set if available, else all other columns)
        covariate_cols = adjustment_set if adjustment_set else [
            c for c in data.columns if c not in [treatment, outcome]
        ]
        covariates = data[covariate_cols] if covariate_cols else data.drop(
            columns=[treatment, outcome], errors='ignore'
        )

        # Convert treatment to binary if continuous
        if not np.array_equal(treatment_col, treatment_col.astype(int)):
            # Continuous treatment - binarize at median
            treatment_binary = (treatment_col > np.median(treatment_col)).astype(int)
        else:
            treatment_binary = treatment_col.astype(int)

        # Create selector and run selection
        selector = self._get_estimator_selector(selection_strategy)

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
        energy_score = selected.energy_score
        quality_tier = self._get_quality_tier(energy_score)

        # Map estimator type to method name
        estimator_to_method = {
            "causal_forest": "CausalForestDML",
            "linear_dml": "LinearDML",
            "drlearner": "linear_regression",  # Map to existing
            "ols": "linear_regression",
        }

        result: EstimationResult = {
            "method": estimator_to_method.get(
                selected.estimator_type.value, "CausalForestDML"
            ),
            "ate": float(selected.ate) if selected.ate is not None else 0.0,
            "ate_ci_lower": float(selected.ate_ci_lower) if selected.ate_ci_lower else 0.0,
            "ate_ci_upper": float(selected.ate_ci_upper) if selected.ate_ci_upper else 0.0,
            "standard_error": float(selected.ate_std) if selected.ate_std else 0.0,
            "effect_size": self._classify_effect_size(selected.ate or 0.0),
            "statistical_significance": bool(
                selected.ate and selected.ate_std and
                abs(selected.ate) > 1.96 * selected.ate_std
            ),
            "p_value": 0.001 if (
                selected.ate and selected.ate_std and
                abs(selected.ate) > 1.96 * selected.ate_std
            ) else 0.15,
            "sample_size": len(data),
            "covariates_adjusted": covariate_cols,
            "heterogeneity_detected": False,
            # V4.2: Energy score fields
            "selection_strategy": strategy,
            "selected_estimator": selected.estimator_type.value,
            "energy_score": float(energy_score),
            "energy_score_data": {
                "score": float(energy_score),
                "treatment_balance_score": float(
                    selected.energy_score_result.treatment_balance_score
                ) if selected.energy_score_result else 0.0,
                "outcome_fit_score": float(
                    selected.energy_score_result.outcome_fit_score
                ) if selected.energy_score_result else 0.0,
                "propensity_calibration": float(
                    selected.energy_score_result.propensity_calibration
                ) if selected.energy_score_result else 0.0,
                "computation_time_ms": float(
                    selected.energy_score_result.computation_time_ms
                ) if selected.energy_score_result else 0.0,
                "quality_tier": quality_tier,
            },
            "selection_reason": selection_result.selection_reason,
            "energy_score_gap": float(selection_result.energy_score_gap),
            "n_estimators_evaluated": len(selection_result.all_results),
            "n_estimators_succeeded": sum(
                1 for r in selection_result.all_results if r.success
            ),
        }

        # Include all estimator results for logging
        all_results = []
        for r in selection_result.all_results:
            all_results.append({
                "estimator": r.estimator_type.value,
                "success": r.success,
                "energy_score": float(r.energy_score) if r.success else None,
                "ate": float(r.ate) if r.ate is not None else None,
                "error": r.error_message if not r.success else None,
            })
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
            "energy_scores": {
                k: float(v) for k, v in selection_result.energy_scores.items()
            },
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

            if use_energy_score_path:
                # V4.2: Energy score-based selection
                logger.info(f"Using energy score selection with strategy: {selection_strategy}")
                try:
                    result, selection_dict, energy_latency_ms = self._select_estimator_with_energy_score(
                        data, treatment, outcome, adjustment_set, selection_strategy
                    )
                    latency_ms = (time.time() - start_time) * 1000

                    return {
                        **state,
                        "estimation_result": result,
                        "estimation_latency_ms": latency_ms,
                        "current_phase": "refuting",
                        "status": "computing",
                        # V4.2: Energy score state fields
                        "energy_score_enabled": True,
                        "selection_strategy": selection_strategy,
                        "estimator_selection_result": selection_dict,
                        "energy_score_latency_ms": energy_latency_ms,
                        "best_energy_score": result.get("energy_score"),
                        "energy_score_quality_tier": result.get("energy_score_data", {}).get(
                            "quality_tier", "unreliable"
                        ),
                    }
                except Exception as e:
                    # Fall back to legacy if energy score fails
                    logger.warning(f"Energy score selection failed: {e}, using legacy path")
                    use_energy_score_path = False

            # Legacy path: single method estimation
            if not use_energy_score_path:
                method = explicit_method or "CausalForestDML"
                logger.info(f"Using legacy estimation with method: {method}")

                if method == "CausalForestDML":
                    result = self._estimate_causal_forest(data, treatment, outcome, adjustment_set)
                elif method == "LinearDML":
                    result = self._estimate_linear_dml(data, treatment, outcome, adjustment_set)
                elif method == "linear_regression":
                    result = self._estimate_linear_regression(data, treatment, outcome, adjustment_set)
                elif method == "propensity_score_weighting":
                    result = self._estimate_propensity_weighting(
                        data, treatment, outcome, adjustment_set
                    )
                else:
                    raise ValueError(f"Unknown estimation method: {method}")

                latency_ms = (time.time() - start_time) * 1000

                return {
                    **state,
                    "estimation_result": result,
                    "estimation_latency_ms": latency_ms,
                    "current_phase": "refuting",
                    "status": "computing",
                    "energy_score_enabled": False,
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

        For now, generates synthetic data. In production, would query
        from repositories.

        Args:
            state: Workflow state with potential data_cache

        Returns:
            DataFrame with treatment, outcome, and covariates
        """
        # Check cache first
        data_cache = state.get("data_cache", {})
        if "estimation_data" in data_cache:
            return data_cache["estimation_data"]

        # Generate synthetic data for testing
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

    def _estimate_causal_forest(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: List[str],
    ) -> EstimationResult:
        """Estimate using Causal Forest DML (heterogeneous effects).

        Mock implementation - in production would use econml.CausalForestDML
        """
        # Mock ATE calculation
        treatment_col = data.get(treatment, data.iloc[:, 0])
        outcome_col = data.get(outcome, data.iloc[:, 1])

        # Simple correlation-based mock
        ate = np.corrcoef(treatment_col, outcome_col)[0, 1] * np.std(outcome_col)
        ate_se = 0.05  # Mock standard error

        # Mock CATE by segments
        cate_segments = [
            {
                "segment": "High Engagement",
                "cate": ate * 1.2,
                "size": 300,
                "description": "HCPs with high engagement",
            },
            {
                "segment": "Low Engagement",
                "cate": ate * 0.8,
                "size": 700,
                "description": "HCPs with low engagement",
            },
        ]

        result: EstimationResult = {
            "method": "CausalForestDML",
            "ate": float(ate),
            "ate_ci_lower": float(ate - 1.96 * ate_se),
            "ate_ci_upper": float(ate + 1.96 * ate_se),
            "standard_error": float(ate_se),
            "cate_segments": cate_segments,
            "effect_size": self._classify_effect_size(ate),
            "statistical_significance": bool(abs(ate) > 1.96 * ate_se),
            "p_value": 0.001 if abs(ate) > 1.96 * ate_se else 0.15,
            "sample_size": len(data),
            "covariates_adjusted": adjustment_set,
            "heterogeneity_detected": True,
        }

        return result

    def _estimate_linear_dml(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: List[str],
    ) -> EstimationResult:
        """Estimate using Linear DML.

        Mock implementation - in production would use econml.LinearDML
        """
        treatment_col = data.get(treatment, data.iloc[:, 0])
        outcome_col = data.get(outcome, data.iloc[:, 1])

        ate = np.corrcoef(treatment_col, outcome_col)[0, 1] * np.std(outcome_col)
        ate_se = 0.06

        result: EstimationResult = {
            "method": "LinearDML",
            "ate": float(ate),
            "ate_ci_lower": float(ate - 1.96 * ate_se),
            "ate_ci_upper": float(ate + 1.96 * ate_se),
            "standard_error": float(ate_se),
            "effect_size": self._classify_effect_size(ate),
            "statistical_significance": bool(abs(ate) > 1.96 * ate_se),
            "p_value": 0.002 if abs(ate) > 1.96 * ate_se else 0.20,
            "sample_size": len(data),
            "covariates_adjusted": adjustment_set,
            "heterogeneity_detected": False,
        }

        return result

    def _estimate_linear_regression(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: List[str],
    ) -> EstimationResult:
        """Estimate using simple linear regression.

        Mock implementation - in production would use statsmodels or sklearn
        """
        treatment_col = data.get(treatment, data.iloc[:, 0])
        outcome_col = data.get(outcome, data.iloc[:, 1])

        ate = np.corrcoef(treatment_col, outcome_col)[0, 1] * np.std(outcome_col)
        ate_se = 0.07

        result: EstimationResult = {
            "method": "linear_regression",
            "ate": float(ate),
            "ate_ci_lower": float(ate - 1.96 * ate_se),
            "ate_ci_upper": float(ate + 1.96 * ate_se),
            "standard_error": float(ate_se),
            "effect_size": self._classify_effect_size(ate),
            "statistical_significance": bool(abs(ate) > 1.96 * ate_se),
            "p_value": 0.005 if abs(ate) > 1.96 * ate_se else 0.25,
            "sample_size": len(data),
            "covariates_adjusted": adjustment_set,
            "heterogeneity_detected": False,
        }

        return result

    def _estimate_propensity_weighting(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: List[str],
    ) -> EstimationResult:
        """Estimate using propensity score weighting.

        Mock implementation - in production would use dowhy or custom implementation
        """
        treatment_col = data.get(treatment, data.iloc[:, 0])
        outcome_col = data.get(outcome, data.iloc[:, 1])

        ate = np.corrcoef(treatment_col, outcome_col)[0, 1] * np.std(outcome_col)
        ate_se = 0.08

        result: EstimationResult = {
            "method": "propensity_score_weighting",
            "ate": float(ate),
            "ate_ci_lower": float(ate - 1.96 * ate_se),
            "ate_ci_upper": float(ate + 1.96 * ate_se),
            "standard_error": float(ate_se),
            "effect_size": self._classify_effect_size(ate),
            "statistical_significance": bool(abs(ate) > 1.96 * ate_se),
            "p_value": 0.01 if abs(ate) > 1.96 * ate_se else 0.30,
            "sample_size": len(data),
            "covariates_adjusted": adjustment_set,
            "heterogeneity_detected": False,
        }

        return result

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
