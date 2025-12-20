"""Estimation Node - Causal effect estimation using DoWhy/EconML.

Estimates Average Treatment Effect (ATE) and Conditional ATE (CATE).
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.agents.causal_impact.state import CausalImpactState, EstimationResult


class EstimationNode:
    """Estimates causal effects using DoWhy/EconML.

    Performance target: <30s
    Type: Standard (computation-heavy)
    """

    def __init__(self):
        """Initialize estimation node."""
        pass

    async def execute(self, state: CausalImpactState) -> Dict:
        """Estimate causal effect.

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
                causal_graph["adjustment_sets"][0]
                if causal_graph["adjustment_sets"]
                else []
            )

            # Get or generate data
            data = self._get_data(state)

            # Choose estimation method
            method = state.get("parameters", {}).get("method", "CausalForestDML")

            # Estimate effect
            if method == "CausalForestDML":
                result = self._estimate_causal_forest(
                    data, treatment, outcome, adjustment_set
                )
            elif method == "LinearDML":
                result = self._estimate_linear_dml(
                    data, treatment, outcome, adjustment_set
                )
            elif method == "linear_regression":
                result = self._estimate_linear_regression(
                    data, treatment, outcome, adjustment_set
                )
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
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                **state,
                "estimation_error": str(e),
                "estimation_latency_ms": latency_ms,
                "status": "failed",
                "error_message": f"Estimation failed: {e}",
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
            0.3 * region_numeric
            + 0.2 * specialty_numeric
            + np.random.normal(0, 0.5, n)
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
