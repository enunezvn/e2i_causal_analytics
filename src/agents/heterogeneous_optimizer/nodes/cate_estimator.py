"""CATE Estimator Node for Heterogeneous Optimizer Agent.

This node estimates Conditional Average Treatment Effects using EconML's CausalForestDML.
Core computational node with minimal LLM usage.
"""

import asyncio
import logging
import os
import time
import traceback
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd

from ..state import CATEResult, HeterogeneousOptimizerState

logger = logging.getLogger(__name__)


def _get_default_data_connector():
    """Get the default data connector based on environment.

    Uses HeterogeneousOptimizerDataConnector if Supabase credentials are available,
    otherwise falls back to MockDataConnector for development/testing.
    """
    if os.getenv("SUPABASE_URL") and (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    ):
        try:
            from ..connectors import HeterogeneousOptimizerDataConnector

            logger.info("Using HeterogeneousOptimizerDataConnector (Supabase)")
            return HeterogeneousOptimizerDataConnector()
        except Exception as e:
            logger.warning(f"Failed to initialize Supabase connector: {e}")

    # Fallback to mock connector
    from ..connectors import MockDataConnector

    logger.info("Using MockDataConnector (development/testing mode)")
    return MockDataConnector()


class CATEEstimatorNode:
    """Estimate Conditional Average Treatment Effects using EconML.

    This node uses CausalForestDML to estimate treatment effect heterogeneity
    across segments.
    """

    def __init__(self, data_connector=None, require_real_data: bool = False):
        """Initialize CATE estimator node.

        Args:
            data_connector: Data connector for fetching analysis data.
                           If None, uses default based on environment.
            require_real_data: If True, raises ValueError if only mock data
                              is available. Used in testing to ensure real
                              Supabase data is used.
        """
        self.require_real_data = require_real_data
        self.data_connector = data_connector or _get_default_data_connector()
        self.timeout_seconds = 180

        # Validate real data requirement
        if self.require_real_data:
            connector_type = type(self.data_connector).__name__
            if "Mock" in connector_type:
                raise ValueError(
                    f"require_real_data=True but data connector is {connector_type}. "
                    "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables "
                    "to use real Supabase data."
                )

    async def execute(self, state: HeterogeneousOptimizerState) -> HeterogeneousOptimizerState:
        """Execute CATE estimation."""
        start_time = time.time()
        logger.info(
            "Starting CATE estimation",
            extra={
                "node": "cate_estimator",
                "treatment_var": state.get("treatment_var"),
                "outcome_var": state.get("outcome_var"),
                "effect_modifiers": state.get("effect_modifiers", []),
                "n_estimators": state.get("n_estimators", 100),
            },
        )

        try:
            from econml.dml import CausalForestDML
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # Fetch data
            df = await self._fetch_data(state)

            if df is None or len(df) < 100:
                return {
                    **state,
                    "errors": [
                        {"node": "cate_estimator", "error": "Insufficient data (need >= 100 rows)"}
                    ],
                    "status": "failed",
                }

            # Prepare data
            Y = df[state["outcome_var"]].values
            T_raw = df[state["treatment_var"]].values

            # Binarize continuous treatment at median (consistent with causal_impact agent)
            # This ensures comparable results between agents and better CATE estimation
            if len(np.unique(T_raw)) > 2:
                median_val = np.median(T_raw)
                T = (T_raw > median_val).astype(int)
                logger.info(
                    f"Binarized continuous treatment at median={median_val:.2f}",
                    extra={
                        "node": "cate_estimator",
                        "treatment_var": state["treatment_var"],
                        "original_unique_values": int(len(np.unique(T_raw))),
                        "median_threshold": float(median_val),
                        "treated_count": int(np.sum(T)),
                        "control_count": int(np.sum(1 - T)),
                    },
                )
            else:
                T = T_raw

            # Diagnostic logging for debugging ATE=0 issue
            logger.info(
                "CATE data prepared",
                extra={
                    "node": "cate_estimator",
                    "n_rows": len(df),
                    "treatment_var": state["treatment_var"],
                    "outcome_var": state["outcome_var"],
                    "T_mean": float(np.mean(T)),
                    "T_std": float(np.std(T)),
                    "T_min": float(np.min(T)),
                    "T_max": float(np.max(T)),
                    "T_unique": int(len(np.unique(T))),
                    "Y_mean": float(np.mean(Y)),
                    "Y_std": float(np.std(Y)),
                    "Y_unique": int(len(np.unique(Y))),
                    "correlation_T_Y": float(np.corrcoef(T, Y)[0, 1])
                    if len(np.unique(T)) > 1
                    else 0.0,
                },
            )

            # Encode effect modifiers (handle categorical)
            X_df = df[state["effect_modifiers"]].copy()
            X = self._encode_features(X_df)

            # W (confounders for nuisance model) is set to None unconditionally.
            # segment_vars are for post-hoc CATE-by-segment analysis in
            # _calculate_cate_by_segment(), NOT for CausalForestDML's W parameter.
            # Using segment_vars as W conflates segmentation with confounding and
            # can produce ATE=0 when segment categories absorb treatment variation.
            W = None

            # Fit Causal Forest
            is_binary_treatment = self._is_binary(T)

            # EconML's CausalForestDML requires n_estimators to be divisible by subforest_size
            # Default subforest_size is 4, so adjust n_estimators to be divisible by 4
            subforest_size = 4
            raw_n_estimators = state.get("n_estimators", 100)
            # Round up to nearest multiple of subforest_size
            n_estimators = (
                (raw_n_estimators + subforest_size - 1) // subforest_size
            ) * subforest_size

            if n_estimators != raw_n_estimators:
                logger.info(
                    f"Adjusted n_estimators from {raw_n_estimators} to {n_estimators} "
                    f"(must be divisible by subforest_size={subforest_size})",
                    extra={"node": "cate_estimator"},
                )

            cf = CausalForestDML(
                model_y=RandomForestRegressor(
                    n_estimators=50, min_samples_leaf=5,
                    min_impurity_decrease=1e-7, random_state=42,
                ),
                model_t=(
                    RandomForestClassifier(
                        n_estimators=50, min_samples_leaf=5,
                        min_impurity_decrease=1e-7, random_state=42,
                    )
                    if is_binary_treatment
                    else RandomForestRegressor(
                        n_estimators=50, min_samples_leaf=5,
                        min_impurity_decrease=1e-7, random_state=42,
                    )
                ),
                discrete_treatment=is_binary_treatment,
                n_estimators=n_estimators,
                subforest_size=subforest_size,
                min_samples_leaf=state.get("min_samples_leaf", 10),
                min_impurity_decrease=1e-7,
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

            # Diagnostic logging for debugging ATE=0 issue
            logger.info(
                "CausalForestDML results",
                extra={
                    "node": "cate_estimator",
                    "ate_raw": ate,
                    "ate_type": type(ate).__name__,
                    "cate_mean": float(np.mean(cate_individual)),
                    "cate_std": float(np.std(cate_individual)),
                    "cate_min": float(np.min(cate_individual)),
                    "cate_max": float(np.max(cate_individual)),
                    "is_binary_treatment": is_binary_treatment,
                },
            )

            # Calculate heterogeneity score
            heterogeneity = self._calculate_heterogeneity(cate_individual, ate)

            # Get feature importance
            feature_importance = dict(
                zip(
                    state["effect_modifiers"],
                    (
                        cf.feature_importances_.tolist()
                        if hasattr(cf, "feature_importances_")
                        else [0] * len(state["effect_modifiers"])
                    ),
                    strict=False,
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

            logger.info(
                "CATE estimation complete",
                extra={
                    "node": "cate_estimator",
                    "overall_ate": float(ate),
                    "heterogeneity_score": heterogeneity,
                    "segment_count": len(cate_by_segment),
                    "latency_ms": estimation_time,
                },
            )

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
            logger.error(
                "CATE estimation timed out",
                extra={"node": "cate_estimator", "timeout_seconds": self.timeout_seconds},
            )
            return {
                **state,
                "errors": [
                    {"node": "cate_estimator", "error": f"Timed out after {self.timeout_seconds}s"}
                ],
                "status": "failed",
            }
        except Exception as e:
            logger.error(
                "CATE estimation failed",
                extra={"node": "cate_estimator", "error": str(e)},
                exc_info=True,
            )
            return {
                **state,
                "errors": [
                    {"node": "cate_estimator", "error": str(e), "traceback": traceback.format_exc()}
                ],
                "status": "failed",
            }

    async def _fetch_data(self, state: HeterogeneousOptimizerState) -> pd.DataFrame:
        """Fetch data for CATE estimation.

        Data source priority:
        1. tier0_data passthrough (from tier0 testing framework)
        2. Primary data connector (Supabase)
        3. Raises ValueError if insufficient data (NO mock fallback)

        Args:
            state: HeterogeneousOptimizerState with data configuration

        Returns:
            DataFrame with required columns for CATE estimation

        Raises:
            ValueError: If insufficient data available
        """
        required_columns = (
            [state["treatment_var"], state["outcome_var"]]
            + state["effect_modifiers"]
            + state["segment_vars"]
        )

        # Priority 1: Use tier0 passthrough data if available
        tier0_data = state.get("tier0_data")
        if tier0_data is not None and len(tier0_data) >= 100:
            # Validate required columns exist in tier0 data
            missing_cols = [c for c in required_columns if c not in tier0_data.columns]
            if not missing_cols:
                logger.info(
                    f"Using tier0 passthrough data ({len(tier0_data)} rows)",
                    extra={
                        "node": "cate_estimator",
                        "data_source": "tier0_passthrough",
                        "row_count": len(tier0_data),
                    },
                )
                return tier0_data
            else:
                logger.warning(
                    f"Tier0 data missing columns {missing_cols}, trying primary connector",
                    extra={"node": "cate_estimator", "missing_columns": missing_cols},
                )

        # Priority 2: Fetch from primary data connector (Supabase)
        df = await self.data_connector.query(
            source=state["data_source"],
            columns=list(set(required_columns)),
            filters=state.get("filters"),
        )

        # Validate we have sufficient data
        if df is None or len(df) < 100:
            row_count = len(df) if df is not None else 0
            raise ValueError(
                f"Insufficient data for CATE estimation ({row_count} rows, need >= 100). "
                f"Either pass tier0_data with required columns or configure Supabase. "
                f"Required columns: {required_columns}"
            )

        logger.info(
            f"Using primary connector data ({len(df)} rows)",
            extra={
                "node": "cate_estimator",
                "data_source": "primary_connector",
                "row_count": len(df),
            },
        )
        return df

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
            if result[col].dtype == "object" or str(result[col].dtype) == "category":
                # Label encode categorical columns
                categories = result[col].unique()
                cat_to_int = {cat: i for i, cat in enumerate(categories)}
                result[col] = result[col].map(cat_to_int).astype(float)

        return cast("np.ndarray[Any, Any]", result.values)

    def _calculate_heterogeneity(self, cate_individual: np.ndarray, ate: float) -> float:
        """Calculate heterogeneity score (coefficient of variation).

        Returns 0-1 score where higher = more heterogeneity.
        """
        std = np.std(cate_individual)
        if ate == 0:
            return 0.0
        cv = std / abs(ate)
        # Normalize to 0-1 scale (CV/2, capped at 1.0)
        return cast(float, min(cv / 2, 1.0))

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
