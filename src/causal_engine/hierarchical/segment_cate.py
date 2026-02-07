"""
Segment CATE Calculator - B9.2

Computes EconML CATE estimates within CausalML-defined segments.

This module provides fine-grained treatment effect estimation within
pre-defined segments, enabling:
- Segment-specific CATE with proper uncertainty quantification
- Comparison of treatment effects across segments
- Identification of segments with highest/lowest treatment response

Author: E2I Causal Analytics Team
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SegmentEstimatorType(str, Enum):
    """EconML estimators available for segment CATE."""

    CAUSAL_FOREST = "causal_forest"
    LINEAR_DML = "linear_dml"
    DRLEARNER = "drlearner"
    S_LEARNER = "s_learner"
    T_LEARNER = "t_learner"
    X_LEARNER = "x_learner"
    OLS = "ols"


@dataclass
class SegmentCATEConfig:
    """Configuration for segment CATE calculation.

    Attributes:
        estimator_type: EconML estimator to use
        min_samples: Minimum samples required for estimation
        compute_ci: Whether to compute confidence intervals
        ci_confidence_level: Confidence level for intervals
        n_bootstrap: Bootstrap iterations for CI (if needed)
        random_state: Random seed for reproducibility
        estimator_params: Additional parameters for the estimator
    """

    estimator_type: str = "causal_forest"
    min_samples: int = 50
    compute_ci: bool = True
    ci_confidence_level: float = 0.95
    n_bootstrap: int = 100
    random_state: int = 42
    estimator_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "estimator_type": self.estimator_type,
            "min_samples": self.min_samples,
            "compute_ci": self.compute_ci,
            "ci_confidence_level": self.ci_confidence_level,
            "n_bootstrap": self.n_bootstrap,
            "random_state": self.random_state,
            "estimator_params": self.estimator_params,
        }


@dataclass
class SegmentCATEResult:
    """Result from segment CATE calculation.

    Attributes:
        segment_id: Segment identifier
        segment_name: Human-readable segment name
        success: Whether estimation succeeded
        cate_mean: Mean CATE within segment
        cate_std: Standard deviation of CATE
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        cate_values: Individual CATE values (optional)
        n_samples: Number of samples in segment
        n_treated: Number of treated samples
        n_control: Number of control samples
        estimator_used: Name of estimator used
        estimation_time_ms: Time taken for estimation
        error_message: Error message if failed
        metadata: Additional result metadata
    """

    segment_id: int
    segment_name: str
    success: bool
    cate_mean: Optional[float] = None
    cate_std: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    cate_values: Optional[NDArray[np.float64]] = None
    n_samples: int = 0
    n_treated: int = 0
    n_control: int = 0
    estimator_used: Optional[str] = None
    estimation_time_ms: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "segment_id": self.segment_id,
            "segment_name": self.segment_name,
            "success": self.success,
            "cate_mean": self.cate_mean,
            "cate_std": self.cate_std,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_samples": self.n_samples,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "estimator_used": self.estimator_used,
            "estimation_time_ms": self.estimation_time_ms,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class SegmentCATECalculator:
    """Calculates EconML CATE within pre-defined segments.

    This calculator runs EconML estimators on subsets of data corresponding
    to different uplift segments from CausalML.

    Example:
        calculator = SegmentCATECalculator()
        result = await calculator.compute(
            X=segment_features,
            treatment=segment_treatment,
            outcome=segment_outcome,
            segment_id=0,
            segment_name="high_uplift",
        )
        print(f"Segment CATE: {result.cate_mean:.4f}")
    """

    def __init__(self, config: Optional[SegmentCATEConfig] = None):
        """Initialize calculator.

        Args:
            config: Configuration for CATE calculation
        """
        self.config = config or SegmentCATEConfig()

    async def compute(
        self,
        X: Union[pd.DataFrame, NDArray[np.float64]],
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        segment_id: int,
        segment_name: str,
    ) -> SegmentCATEResult:
        """Compute CATE within a segment.

        Args:
            X: Feature matrix for segment
            treatment: Treatment assignment for segment
            outcome: Outcomes for segment
            segment_id: Numeric segment identifier
            segment_name: Human-readable segment name

        Returns:
            SegmentCATEResult with CATE estimates
        """
        start_time = time.perf_counter()

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            X_df = X

        n_samples = len(X_df)
        n_treated = int(np.sum(treatment == 1))
        n_control = int(np.sum(treatment == 0))

        # Validate minimum samples
        if n_samples < self.config.min_samples:
            elapsed = (time.perf_counter() - start_time) * 1000
            return SegmentCATEResult(
                segment_id=segment_id,
                segment_name=segment_name,
                success=False,
                n_samples=n_samples,
                n_treated=n_treated,
                n_control=n_control,
                estimation_time_ms=elapsed,
                error_message=f"Insufficient samples: {n_samples} < {self.config.min_samples}",
            )

        # Validate treatment/control balance
        if n_treated < 10 or n_control < 10:
            elapsed = (time.perf_counter() - start_time) * 1000
            return SegmentCATEResult(
                segment_id=segment_id,
                segment_name=segment_name,
                success=False,
                n_samples=n_samples,
                n_treated=n_treated,
                n_control=n_control,
                estimation_time_ms=elapsed,
                error_message=f"Insufficient treatment/control: {n_treated}/{n_control}",
            )

        try:
            # Run EconML estimator
            (
                cate_values,
                cate_mean,
                cate_std,
                ci_lower,
                ci_upper,
                estimator_name,
            ) = await self._run_econml_estimator(X_df, treatment, outcome)

            elapsed = (time.perf_counter() - start_time) * 1000

            return SegmentCATEResult(
                segment_id=segment_id,
                segment_name=segment_name,
                success=True,
                cate_mean=cate_mean,
                cate_std=cate_std,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                cate_values=cate_values,
                n_samples=n_samples,
                n_treated=n_treated,
                n_control=n_control,
                estimator_used=estimator_name,
                estimation_time_ms=elapsed,
                metadata={
                    "config": self.config.to_dict(),
                    "n_features": X_df.shape[1],
                },
            )

        except Exception as e:
            logger.warning(f"Segment CATE failed for {segment_name}: {e}")
            elapsed = (time.perf_counter() - start_time) * 1000
            return SegmentCATEResult(
                segment_id=segment_id,
                segment_name=segment_name,
                success=False,
                n_samples=n_samples,
                n_treated=n_treated,
                n_control=n_control,
                estimation_time_ms=elapsed,
                error_message=str(e),
            )

    async def _run_econml_estimator(
        self,
        X: pd.DataFrame,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
    ) -> tuple:
        """Run the configured EconML estimator.

        Returns:
            Tuple of (cate_values, cate_mean, cate_std, ci_lower, ci_upper, estimator_name)
        """
        estimator_type = self.config.estimator_type.lower()
        X_arr = X.values

        if estimator_type == "causal_forest":
            return self._run_causal_forest(X_arr, treatment, outcome)
        elif estimator_type == "linear_dml":
            return self._run_linear_dml(X_arr, treatment, outcome)
        elif estimator_type == "drlearner":
            return self._run_drlearner(X_arr, treatment, outcome)
        elif estimator_type == "s_learner":
            return self._run_s_learner(X_arr, treatment, outcome)
        elif estimator_type == "t_learner":
            return self._run_t_learner(X_arr, treatment, outcome)
        elif estimator_type == "x_learner":
            return self._run_x_learner(X_arr, treatment, outcome)
        elif estimator_type == "ols":
            return self._run_ols(X_arr, treatment, outcome)
        else:
            # Default to causal forest
            return self._run_causal_forest(X_arr, treatment, outcome)

    def _run_causal_forest(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
    ) -> tuple:
        """Run CausalForestDML estimator."""
        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_selection import VarianceThreshold

        # Remove near-constant features to prevent numerical instability in tree nodes
        selector = VarianceThreshold(threshold=0.01)
        X_clean = selector.fit_transform(X)
        if X_clean.shape[1] == 0:
            X_clean = X  # Fall back if all features were filtered

        is_binary = len(np.unique(treatment)) == 2

        model = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=50, min_samples_leaf=5,
                min_impurity_decrease=1e-7, random_state=self.config.random_state,
            ),
            model_t=(
                RandomForestClassifier(
                    n_estimators=50, min_samples_leaf=5,
                    min_impurity_decrease=1e-7, random_state=self.config.random_state,
                )
                if is_binary
                else RandomForestRegressor(
                    n_estimators=50, min_samples_leaf=5,
                    min_impurity_decrease=1e-7, random_state=self.config.random_state,
                )
            ),
            discrete_treatment=is_binary,
            n_estimators=self.config.estimator_params.get("n_estimators", 100),
            min_samples_leaf=self.config.estimator_params.get("min_samples_leaf", 10),
            min_impurity_decrease=1e-7,
            random_state=self.config.random_state,
        )
        model.fit(outcome, treatment, X=X_clean, W=X_clean)

        cate_values = model.effect(X_clean)
        cate_mean = float(np.mean(cate_values))
        cate_std = float(np.std(cate_values))

        # Confidence intervals
        ci_lower, ci_upper = self._compute_ci(model, X_clean, cate_values, cate_mean, cate_std)

        return cate_values, cate_mean, cate_std, ci_lower, ci_upper, "causal_forest"

    def _run_linear_dml(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
    ) -> tuple:
        """Run LinearDML estimator."""
        from econml.dml import LinearDML
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        model = LinearDML(
            model_y=RandomForestRegressor(
                n_estimators=50, min_samples_leaf=5,
                min_impurity_decrease=1e-7, random_state=self.config.random_state,
            ),
            model_t=RandomForestClassifier(
                n_estimators=50, min_samples_leaf=5,
                min_impurity_decrease=1e-7, random_state=self.config.random_state,
            ),
            discrete_treatment=True,  # Required for binary treatment
            random_state=self.config.random_state,
        )
        model.fit(outcome, treatment, X=X, W=X)

        cate_values = model.effect(X)
        cate_mean = float(np.mean(cate_values))
        cate_std = float(np.std(cate_values))

        ci_lower, ci_upper = self._compute_ci(model, X, cate_values, cate_mean, cate_std)

        return cate_values, cate_mean, cate_std, ci_lower, ci_upper, "linear_dml"

    def _run_drlearner(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
    ) -> tuple:
        """Run DRLearner (Doubly Robust) estimator."""
        from econml.dr import DRLearner
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        model = DRLearner(
            model_regression=GradientBoostingRegressor(
                n_estimators=50, random_state=self.config.random_state
            ),
            model_propensity=GradientBoostingClassifier(
                n_estimators=50, random_state=self.config.random_state
            ),
            model_final=GradientBoostingRegressor(
                n_estimators=50, random_state=self.config.random_state
            ),
            random_state=self.config.random_state,
        )
        model.fit(outcome, treatment, X=X, W=X)

        cate_values = model.effect(X)
        cate_mean = float(np.mean(cate_values))
        cate_std = float(np.std(cate_values))

        ci_lower, ci_upper = self._compute_ci_bootstrap(X, treatment, outcome, cate_mean)

        return cate_values, cate_mean, cate_std, ci_lower, ci_upper, "drlearner"

    def _run_s_learner(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
    ) -> tuple:
        """Run S-Learner (Single Model)."""
        from sklearn.ensemble import GradientBoostingRegressor

        # Include treatment as feature
        X_with_t = np.column_stack([treatment, X])

        model = GradientBoostingRegressor(
            n_estimators=self.config.estimator_params.get("n_estimators", 100),
            max_depth=self.config.estimator_params.get("max_depth", 5),
            random_state=self.config.random_state,
        )
        model.fit(X_with_t, outcome)

        # CATE = E[Y|X, T=1] - E[Y|X, T=0]
        X_treat_1 = np.column_stack([np.ones(len(X)), X])
        X_treat_0 = np.column_stack([np.zeros(len(X)), X])
        cate_values = model.predict(X_treat_1) - model.predict(X_treat_0)

        cate_mean = float(np.mean(cate_values))
        cate_std = float(np.std(cate_values))

        ci_lower, ci_upper = self._compute_ci_bootstrap(X, treatment, outcome, cate_mean)

        return cate_values, cate_mean, cate_std, ci_lower, ci_upper, "s_learner"

    def _run_t_learner(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
    ) -> tuple:
        """Run T-Learner (Two Models)."""
        from sklearn.ensemble import GradientBoostingRegressor

        X_1 = X[treatment == 1]
        X_0 = X[treatment == 0]
        Y_1 = outcome[treatment == 1]
        Y_0 = outcome[treatment == 0]

        params = {
            "n_estimators": self.config.estimator_params.get("n_estimators", 100),
            "max_depth": self.config.estimator_params.get("max_depth", 5),
            "random_state": self.config.random_state,
        }

        model_1 = GradientBoostingRegressor(**params)
        model_0 = GradientBoostingRegressor(**params)
        model_1.fit(X_1, Y_1)
        model_0.fit(X_0, Y_0)

        cate_values = model_1.predict(X) - model_0.predict(X)

        cate_mean = float(np.mean(cate_values))
        cate_std = float(np.std(cate_values))

        ci_lower, ci_upper = self._compute_ci_bootstrap(X, treatment, outcome, cate_mean)

        return cate_values, cate_mean, cate_std, ci_lower, ci_upper, "t_learner"

    def _run_x_learner(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
    ) -> tuple:
        """Run X-Learner."""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegressionCV

        X_1 = X[treatment == 1]
        X_0 = X[treatment == 0]
        Y_1 = outcome[treatment == 1]
        Y_0 = outcome[treatment == 0]

        params = {
            "n_estimators": self.config.estimator_params.get("n_estimators", 100),
            "max_depth": self.config.estimator_params.get("max_depth", 5),
            "random_state": self.config.random_state,
        }

        # Stage 1: Fit response models
        model_1 = GradientBoostingRegressor(**params)
        model_0 = GradientBoostingRegressor(**params)
        model_1.fit(X_1, Y_1)
        model_0.fit(X_0, Y_0)

        # Stage 2: Imputed treatment effects
        tau_1 = Y_1 - model_0.predict(X_1)
        tau_0 = model_1.predict(X_0) - Y_0

        # Second-stage models
        model_tau_1 = GradientBoostingRegressor(**params)
        model_tau_0 = GradientBoostingRegressor(**params)
        model_tau_1.fit(X_1, tau_1)
        model_tau_0.fit(X_0, tau_0)

        # Propensity scores
        ps_model = LogisticRegressionCV(cv=3, max_iter=500)
        ps_model.fit(X, treatment)
        propensity = ps_model.predict_proba(X)[:, 1]

        # Combine with propensity weighting
        tau_hat_1 = model_tau_1.predict(X)
        tau_hat_0 = model_tau_0.predict(X)
        cate_values = propensity * tau_hat_0 + (1 - propensity) * tau_hat_1

        cate_mean = float(np.mean(cate_values))
        cate_std = float(np.std(cate_values))

        ci_lower, ci_upper = self._compute_ci_bootstrap(X, treatment, outcome, cate_mean)

        return cate_values, cate_mean, cate_std, ci_lower, ci_upper, "x_learner"

    def _run_ols(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
    ) -> tuple:
        """Run simple OLS estimator."""
        from sklearn.linear_model import LinearRegression

        X_with_t = np.column_stack([treatment, X])
        model = LinearRegression()
        model.fit(X_with_t, outcome)

        ate = float(model.coef_[0])
        cate_values = np.full(len(X), ate)

        cate_mean = ate
        cate_std = 0.0  # OLS gives constant CATE

        ci_lower, ci_upper = self._compute_ci_bootstrap(X, treatment, outcome, cate_mean)

        return cate_values, cate_mean, cate_std, ci_lower, ci_upper, "ols"

    def _compute_ci(
        self,
        model: Any,
        X: NDArray[np.float64],
        cate_values: NDArray[np.float64],
        cate_mean: float,
        cate_std: float,
    ) -> tuple:
        """Compute confidence interval from model inference if available."""
        if not self.config.compute_ci:
            return None, None

        try:
            # Try to get CI from model's inference method
            if hasattr(model, "effect_inference"):
                inf = model.effect_inference(X)
                if hasattr(inf, "conf_int_mean"):
                    ci = inf.conf_int_mean(alpha=1 - self.config.ci_confidence_level)
                    return float(ci[0]), float(ci[1])
        except Exception:
            pass

        # Fallback to normal approximation
        z = 1.96 if self.config.ci_confidence_level == 0.95 else 2.576
        n = len(cate_values)
        se = cate_std / np.sqrt(n) if n > 0 else cate_std

        ci_lower = cate_mean - z * se
        ci_upper = cate_mean + z * se

        return ci_lower, ci_upper

    def _compute_ci_bootstrap(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        cate_mean: float,
    ) -> tuple:
        """Compute confidence interval via bootstrap."""
        if not self.config.compute_ci:
            return None, None

        # Simple normal approximation for segment CI
        # (bootstrap would be too slow for production use)
        n = len(X)
        n_t = np.sum(treatment == 1)
        n_c = np.sum(treatment == 0)

        if n_t > 0 and n_c > 0:
            # Variance estimate
            y_t = outcome[treatment == 1]
            y_c = outcome[treatment == 0]
            var_t = np.var(y_t) / n_t if n_t > 1 else 0
            var_c = np.var(y_c) / n_c if n_c > 1 else 0
            se = np.sqrt(var_t + var_c)
        else:
            se = np.std(outcome) / np.sqrt(n) if n > 0 else 0

        z = 1.96 if self.config.ci_confidence_level == 0.95 else 2.576
        ci_lower = cate_mean - z * se
        ci_upper = cate_mean + z * se

        return float(ci_lower), float(ci_upper)
