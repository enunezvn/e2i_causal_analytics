"""CATE Estimator Node for Heterogeneous Optimizer Agent.

This node estimates Conditional Average Treatment Effects using EconML's CausalForestDML.
Core computational node with minimal LLM usage.
"""

import asyncio
import time
import traceback
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from ..state import HeterogeneousOptimizerState, CATEResult


class MockDataConnector:
    """Mock data connector for testing.

    TODO: Replace with SupabaseDataConnector in integration phase.
    """

    async def query(
        self,
        source: str,
        columns: List[str],
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Generate mock pharma CATE data."""
        np.random.seed(42)
        n_samples = 1000

        # Generate heterogeneous treatment effects
        data = {
            # Segment variables
            "hcp_specialty": np.random.choice(
                ["Oncology", "Cardiology", "Primary Care", "Rheumatology"],
                n_samples
            ),
            "patient_volume_decile": np.random.choice(
                ["1-2", "3-4", "5-6", "7-8", "9-10"],
                n_samples
            ),
            "region": np.random.choice(
                ["Northeast", "Southeast", "Midwest", "West"],
                n_samples
            ),
            # Effect modifiers
            "hcp_tenure": np.random.uniform(1, 30, n_samples),
            "competitive_pressure": np.random.uniform(0, 1, n_samples),
            "formulary_status": np.random.choice([0, 1], n_samples),
            # Treatment (binary: 0 or 1)
            "hcp_engagement_frequency": np.random.choice([0, 1], n_samples),
        }

        # Generate outcome with heterogeneous treatment effects
        # Different segments have different treatment responses
        outcome = np.zeros(n_samples)
        for i in range(n_samples):
            # Base outcome
            base = 100 + np.random.normal(0, 20)

            # Heterogeneous treatment effect based on specialty
            treatment_effect = 0
            if data["hcp_engagement_frequency"][i] == 1:
                if data["hcp_specialty"][i] == "Oncology":
                    treatment_effect = 50 + np.random.normal(0, 10)  # High responder
                elif data["hcp_specialty"][i] == "Cardiology":
                    treatment_effect = 30 + np.random.normal(0, 10)  # Medium responder
                elif data["hcp_specialty"][i] == "Primary Care":
                    treatment_effect = 10 + np.random.normal(0, 10)  # Low responder
                else:  # Rheumatology
                    treatment_effect = 25 + np.random.normal(0, 10)

                # Modify by tenure
                treatment_effect *= (1 + data["hcp_tenure"][i] / 100)

                # Modify by competitive pressure (negative)
                treatment_effect *= (1 - data["competitive_pressure"][i] * 0.3)

            outcome[i] = base + treatment_effect

        data["trx_total"] = outcome

        df = pd.DataFrame(data)

        # Filter columns
        if columns:
            available_cols = [col for col in columns if col in df.columns]
            df = df[available_cols]

        return df


class CATEEstimatorNode:
    """Estimate Conditional Average Treatment Effects using EconML.

    This node uses CausalForestDML to estimate treatment effect heterogeneity
    across segments.
    """

    def __init__(self, data_connector=None):
        self.data_connector = data_connector or MockDataConnector()
        self.timeout_seconds = 180

    async def execute(
        self, state: HeterogeneousOptimizerState
    ) -> HeterogeneousOptimizerState:
        """Execute CATE estimation."""
        start_time = time.time()

        try:
            from econml.dml import CausalForestDML
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

            # Fetch data
            df = await self._fetch_data(state)

            if df is None or len(df) < 100:
                return {
                    **state,
                    "errors": [{
                        "node": "cate_estimator",
                        "error": "Insufficient data (need >= 100 rows)"
                    }],
                    "status": "failed"
                }

            # Prepare data
            Y = df[state["outcome_var"]].values
            T = df[state["treatment_var"]].values

            # Encode effect modifiers (handle categorical)
            X_df = df[state["effect_modifiers"]].copy()
            X = self._encode_features(X_df)

            # Encode segment variables (handle categorical)
            if state["segment_vars"]:
                W_df = df[state["segment_vars"]].copy()
                W = self._encode_features(W_df)
            else:
                W = None

            # Fit Causal Forest
            is_binary_treatment = self._is_binary(T)
            cf = CausalForestDML(
                model_y=RandomForestRegressor(n_estimators=50, random_state=42),
                model_t=RandomForestClassifier(n_estimators=50, random_state=42)
                if is_binary_treatment
                else RandomForestRegressor(n_estimators=50, random_state=42),
                discrete_treatment=is_binary_treatment,
                n_estimators=state.get("n_estimators", 100),
                min_samples_leaf=state.get("min_samples_leaf", 10),
                random_state=42,
            )

            # Fit with timeout
            await asyncio.wait_for(
                asyncio.to_thread(cf.fit, Y, T, X=X, W=W),
                timeout=self.timeout_seconds,
            )

            # Get overall ATE
            ate = cf.ate(X)

            # Get individual treatment effects
            cate_individual = cf.effect(X)

            # Calculate heterogeneity score
            heterogeneity = self._calculate_heterogeneity(cate_individual, ate)

            # Get feature importance
            feature_importance = dict(
                zip(
                    state["effect_modifiers"],
                    cf.feature_importances_.tolist()
                    if hasattr(cf, "feature_importances_")
                    else [0] * len(state["effect_modifiers"]),
                )
            )

            # Calculate CATE by segment
            cate_by_segment = await self._calculate_cate_by_segment(
                df,
                cf,
                state["segment_vars"],
                state["effect_modifiers"],
                state.get("significance_level", 0.05),
            )

            estimation_time = int((time.time() - start_time) * 1000)

            return {
                **state,
                "overall_ate": float(ate),
                "heterogeneity_score": heterogeneity,
                "feature_importance": feature_importance,
                "cate_by_segment": cate_by_segment,
                "estimation_latency_ms": estimation_time,
                "status": "analyzing",
            }

        except asyncio.TimeoutError:
            return {
                **state,
                "errors": [{
                    "node": "cate_estimator",
                    "error": f"Timed out after {self.timeout_seconds}s"
                }],
                "status": "failed",
            }
        except Exception as e:
            return {
                **state,
                "errors": [{
                    "node": "cate_estimator",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }],
                "status": "failed",
            }

    async def _fetch_data(
        self, state: HeterogeneousOptimizerState
    ) -> pd.DataFrame:
        """Fetch data for CATE estimation."""
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

    def _is_binary(self, T: np.ndarray) -> bool:
        """Check if treatment is binary."""
        unique_vals = np.unique(T)
        return len(unique_vals) == 2

    def _encode_features(self, df: pd.DataFrame) -> np.ndarray:
        """Encode features, handling categorical columns.

        Uses label encoding for categorical columns.

        Args:
            df: DataFrame with features

        Returns:
            Numpy array with encoded features
        """
        result = df.copy()

        for col in result.columns:
            if result[col].dtype == 'object' or str(result[col].dtype) == 'category':
                # Label encode categorical columns
                categories = result[col].unique()
                cat_to_int = {cat: i for i, cat in enumerate(categories)}
                result[col] = result[col].map(cat_to_int).astype(float)

        return result.values

    def _calculate_heterogeneity(
        self, cate_individual: np.ndarray, ate: float
    ) -> float:
        """Calculate heterogeneity score (coefficient of variation).

        Returns 0-1 score where higher = more heterogeneity.
        """
        std = np.std(cate_individual)
        if ate == 0:
            return 0.0
        cv = std / abs(ate)
        # Normalize to 0-1 scale (CV/2, capped at 1.0)
        return min(cv / 2, 1.0)

    async def _calculate_cate_by_segment(
        self,
        df: pd.DataFrame,
        cf,
        segment_vars: List[str],
        effect_modifiers: List[str],
        alpha: float,
    ) -> Dict[str, List[CATEResult]]:
        """Calculate CATE for each segment value."""

        cate_by_segment = {}

        for segment_var in segment_vars:
            segment_results = []

            for segment_value in df[segment_var].unique():
                mask = df[segment_var] == segment_value
                segment_df = df[mask]

                if len(segment_df) < 10:
                    continue

                X_segment = segment_df[effect_modifiers].values

                # Get CATE for segment
                cate = cf.effect(X_segment)
                cate_mean = float(np.mean(cate))

                # Get confidence interval
                try:
                    cate_interval = cf.effect_interval(X_segment, alpha=alpha)
                    ci_lower = float(np.mean(cate_interval[0]))
                    ci_upper = float(np.mean(cate_interval[1]))
                except Exception:
                    ci_lower = cate_mean - 1.96 * float(np.std(cate))
                    ci_upper = cate_mean + 1.96 * float(np.std(cate))

                # Determine statistical significance
                significant = (ci_lower > 0) or (ci_upper < 0)

                segment_results.append(
                    CATEResult(
                        segment_name=segment_var,
                        segment_value=str(segment_value),
                        cate_estimate=cate_mean,
                        cate_ci_lower=ci_lower,
                        cate_ci_upper=ci_upper,
                        sample_size=len(segment_df),
                        statistical_significance=significant,
                    )
                )

            # Sort by CATE estimate
            segment_results.sort(key=lambda x: x["cate_estimate"], reverse=True)
            cate_by_segment[segment_var] = segment_results

        return cate_by_segment
