"""
Enhanced Estimator Selector with Energy Score-Based Selection

Replaces the "first success" fallback strategy with "best score" selection.
Each estimator in the chain is evaluated, and the one with the lowest
energy score is selected as the final estimate.

Integration:
    - Called by Causal Impact agent in the Estimation node
    - Results logged to ml_experiments with energy_score metadata
    - Feeds into Refutation node with selected estimator info

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    EstimatorSelector                             │
    │  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐ │
    │  │ Causal    │   │ Linear    │   │ DML       │   │ OLS       │ │
    │  │ Forest    │──▶│ DML       │──▶│ Learner   │──▶│ Fallback  │ │
    │  └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘ │
    │        │               │               │               │        │
    │        ▼               ▼               ▼               ▼        │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │              Energy Score Calculator                        ││
    │  │  Score each estimator → Select minimum → Return best       ││
    │  └─────────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .score_calculator import (
    EnergyScoreCalculator,
    EnergyScoreConfig,
    EnergyScoreResult,
)

logger = logging.getLogger(__name__)


class EstimatorType(str, Enum):
    """Supported causal estimator types."""

    CAUSAL_FOREST = "causal_forest"
    LINEAR_DML = "linear_dml"
    DML_LEARNER = "dml_learner"
    DRLEARNER = "drlearner"
    ORTHO_FOREST = "ortho_forest"
    S_LEARNER = "s_learner"
    T_LEARNER = "t_learner"
    X_LEARNER = "x_learner"
    OLS = "ols"


class SelectionStrategy(str, Enum):
    """Strategy for selecting among estimators."""

    FIRST_SUCCESS = "first_success"  # Legacy: use first that doesn't fail
    BEST_ENERGY_SCORE = "best_energy"  # New: use lowest energy score
    ENSEMBLE = "ensemble"  # Future: combine multiple estimators


@dataclass
class EstimatorResult:
    """Result from a single estimator run."""

    estimator_type: EstimatorType
    success: bool

    # Effect estimates
    ate: Optional[float] = None
    cate: Optional[NDArray[np.float64]] = None

    # Uncertainty
    ate_std: Optional[float] = None
    ate_ci_lower: Optional[float] = None
    ate_ci_upper: Optional[float] = None

    # Energy score (computed post-estimation)
    energy_score_result: Optional[EnergyScoreResult] = None

    # Propensity scores (for energy score computation)
    propensity_scores: Optional[NDArray[np.float64]] = None

    # Error info if failed
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    # Timing
    estimation_time_ms: float = 0.0

    # Raw estimator object (for refutation)
    raw_estimate: Optional[Any] = None

    @property
    def energy_score(self) -> float:
        """Get energy score value, or infinity if not computed."""
        if self.energy_score_result is None:
            return float("inf")
        return self.energy_score_result.energy_score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "estimator_type": self.estimator_type.value,
            "success": self.success,
            "ate": self.ate,
            "ate_std": self.ate_std,
            "ate_ci_lower": self.ate_ci_lower,
            "ate_ci_upper": self.ate_ci_upper,
            "energy_score": self.energy_score if self.success else None,
            "error_message": self.error_message,
            "estimation_time_ms": self.estimation_time_ms,
        }


@dataclass
class SelectionResult:
    """Result of estimator selection process."""

    # Selected estimator
    selected: EstimatorResult
    selection_strategy: SelectionStrategy

    # All evaluated estimators (for logging/analysis)
    all_results: list[EstimatorResult] = field(default_factory=list)

    # Selection metadata
    selection_reason: str = ""
    total_time_ms: float = 0.0

    # Energy score comparison
    energy_scores: dict[str, float] = field(default_factory=dict)
    energy_score_gap: float = 0.0  # Gap between best and second-best

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "selected_estimator": self.selected.estimator_type.value,
            "selection_strategy": self.selection_strategy.value,
            "selection_reason": self.selection_reason,
            "ate": self.selected.ate,
            "energy_score": self.selected.energy_score,
            "energy_scores": self.energy_scores,
            "energy_score_gap": self.energy_score_gap,
            "total_time_ms": self.total_time_ms,
            "n_estimators_evaluated": len(self.all_results),
            "n_estimators_succeeded": sum(1 for r in self.all_results if r.success),
        }


@dataclass
class EstimatorConfig:
    """Configuration for a single estimator."""

    estimator_type: EstimatorType
    enabled: bool = True
    priority: int = 1  # Lower = higher priority in fallback chain

    # Estimator-specific parameters
    params: dict[str, Any] = field(default_factory=dict)

    # Timeout
    timeout_seconds: float = 30.0


@dataclass
class EstimatorSelectorConfig:
    """Configuration for the estimator selector."""

    strategy: SelectionStrategy = SelectionStrategy.BEST_ENERGY_SCORE

    # Estimator chain (ordered by priority)
    estimators: list[EstimatorConfig] = field(
        default_factory=lambda: [
            EstimatorConfig(EstimatorType.CAUSAL_FOREST, priority=1),
            EstimatorConfig(EstimatorType.LINEAR_DML, priority=2),
            EstimatorConfig(EstimatorType.DRLEARNER, priority=3),
            EstimatorConfig(EstimatorType.OLS, priority=4),
        ]
    )

    # Energy score configuration
    energy_score_config: EnergyScoreConfig = field(default_factory=EnergyScoreConfig)

    # Selection thresholds
    min_energy_score_gap: float = 0.05  # Minimum gap to prefer one over another
    max_acceptable_energy_score: float = 0.8  # Warn if best score is above this

    # Fallback behavior
    fallback_on_all_fail: bool = True
    fallback_estimator: EstimatorType = EstimatorType.OLS

    # Parallelization (future)
    parallel_evaluation: bool = False
    max_workers: int = 4


class BaseEstimatorWrapper(ABC):
    """Abstract base class for estimator wrappers."""

    @abstractmethod
    def fit(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        **kwargs,
    ) -> EstimatorResult:
        """Fit the estimator and return results."""
        pass

    @property
    @abstractmethod
    def estimator_type(self) -> EstimatorType:
        """Return the estimator type."""
        pass


class CausalForestWrapper(BaseEstimatorWrapper):
    """Wrapper for EconML CausalForest."""

    def __init__(self, config: EstimatorConfig):
        self.config = config

    @property
    def estimator_type(self) -> EstimatorType:
        return EstimatorType.CAUSAL_FOREST

    def fit(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        **kwargs,
    ) -> EstimatorResult:
        import time
        import warnings

        start = time.perf_counter()

        try:
            from econml.dml import CausalForestDML
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # Extract parameters
            base_min_leaf = self.config.params.get("min_samples_leaf", 10)

            # Fix 5A: Adaptive min_samples_leaf based on control group size
            n_control = int((1 - treatment.mean()) * len(treatment))
            adaptive_min_leaf = max(5, min(base_min_leaf, n_control // 10))

            is_binary = len(np.unique(treatment)) == 2
            rs = self.config.params.get("random_state", 42)

            params = {
                "model_y": RandomForestRegressor(
                    n_estimators=50, min_samples_leaf=5,
                    min_impurity_decrease=1e-7, random_state=rs,
                ),
                "model_t": (
                    RandomForestClassifier(
                        n_estimators=50, min_samples_leaf=5,
                        min_impurity_decrease=1e-7, random_state=rs,
                    )
                    if is_binary
                    else RandomForestRegressor(
                        n_estimators=50, min_samples_leaf=5,
                        min_impurity_decrease=1e-7, random_state=rs,
                    )
                ),
                "discrete_treatment": is_binary,
                "n_estimators": self.config.params.get("n_estimators", 100),
                "min_samples_leaf": adaptive_min_leaf,
                "min_impurity_decrease": 1e-7,
                "max_depth": self.config.params.get("max_depth", None),
                "random_state": rs,
            }

            # Fit model with warning suppression for small control groups
            model = CausalForestDML(**params)
            X = covariates.values
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Too few control units")
                model.fit(outcome, treatment, X=X, W=X)

            # Get estimates
            cate = model.effect(X)
            ate = float(np.mean(cate))
            ate_std = float(np.std(cate) / np.sqrt(len(cate)))

            # Get confidence intervals
            try:
                ate_inf = model.effect_inference(X)
                ci = ate_inf.conf_int_mean()
                ate_ci_lower, ate_ci_upper = float(ci[0]), float(ci[1])
            except Exception:
                ate_ci_lower = ate - 1.96 * ate_std
                ate_ci_upper = ate + 1.96 * ate_std

            # Estimate propensity scores for energy score
            from sklearn.linear_model import LogisticRegressionCV

            ps_model = LogisticRegressionCV(cv=3, max_iter=500)
            ps_model.fit(X, treatment)
            propensity_scores = ps_model.predict_proba(X)[:, 1]

            elapsed = (time.perf_counter() - start) * 1000

            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=True,
                ate=ate,
                cate=cate,
                ate_std=ate_std,
                ate_ci_lower=ate_ci_lower,
                ate_ci_upper=ate_ci_upper,
                propensity_scores=propensity_scores,
                estimation_time_ms=elapsed,
                raw_estimate=model,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(f"CausalForest failed: {e}")
            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                estimation_time_ms=elapsed,
            )


class LinearDMLWrapper(BaseEstimatorWrapper):
    """Wrapper for EconML LinearDML."""

    def __init__(self, config: EstimatorConfig):
        self.config = config

    @property
    def estimator_type(self) -> EstimatorType:
        return EstimatorType.LINEAR_DML

    def fit(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        **kwargs,
    ) -> EstimatorResult:
        import time

        start = time.perf_counter()

        try:
            from econml.dml import LinearDML
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # Fit model
            model = LinearDML(
                model_y=RandomForestRegressor(
                    n_estimators=50, min_samples_leaf=5,
                    min_impurity_decrease=1e-7, random_state=42,
                ),
                model_t=RandomForestClassifier(
                    n_estimators=50, min_samples_leaf=5,
                    min_impurity_decrease=1e-7, random_state=42,
                ),
                discrete_treatment=True,
                random_state=42,
            )
            X = covariates.values
            model.fit(outcome, treatment, X=X, W=X)

            # Get estimates
            cate = model.effect(X)
            ate = float(np.mean(cate))
            ate_std = float(np.std(cate) / np.sqrt(len(cate)))

            # Confidence intervals
            try:
                ate_inf = model.effect_inference(X)
                ci = ate_inf.conf_int_mean()
                ate_ci_lower, ate_ci_upper = float(ci[0]), float(ci[1])
            except Exception:
                ate_ci_lower = ate - 1.96 * ate_std
                ate_ci_upper = ate + 1.96 * ate_std

            # Propensity scores
            from sklearn.linear_model import LogisticRegressionCV

            ps_model = LogisticRegressionCV(cv=3, max_iter=500)
            ps_model.fit(X, treatment)
            propensity_scores = ps_model.predict_proba(X)[:, 1]

            elapsed = (time.perf_counter() - start) * 1000

            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=True,
                ate=ate,
                cate=cate,
                ate_std=ate_std,
                ate_ci_lower=ate_ci_lower,
                ate_ci_upper=ate_ci_upper,
                propensity_scores=propensity_scores,
                estimation_time_ms=elapsed,
                raw_estimate=model,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(f"LinearDML failed: {e}")
            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                estimation_time_ms=elapsed,
            )


class DRLearnerWrapper(BaseEstimatorWrapper):
    """Wrapper for EconML DRLearner (Doubly Robust)."""

    def __init__(self, config: EstimatorConfig):
        self.config = config

    @property
    def estimator_type(self) -> EstimatorType:
        return EstimatorType.DRLEARNER

    def fit(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        **kwargs,
    ) -> EstimatorResult:
        import time

        start = time.perf_counter()

        try:
            from econml.dr import DRLearner
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

            model = DRLearner(
                model_regression=GradientBoostingRegressor(n_estimators=50, random_state=42),
                model_propensity=GradientBoostingClassifier(n_estimators=50, random_state=42),
                model_final=GradientBoostingRegressor(n_estimators=50, random_state=42),
                random_state=42,
            )
            X = covariates.values
            model.fit(outcome, treatment, X=X, W=X)

            cate = model.effect(X)
            ate = float(np.mean(cate))
            ate_std = float(np.std(cate) / np.sqrt(len(cate)))
            ate_ci_lower = ate - 1.96 * ate_std
            ate_ci_upper = ate + 1.96 * ate_std

            # Propensity scores
            from sklearn.linear_model import LogisticRegressionCV

            ps_model = LogisticRegressionCV(cv=3, max_iter=500)
            ps_model.fit(X, treatment)
            propensity_scores = ps_model.predict_proba(X)[:, 1]

            elapsed = (time.perf_counter() - start) * 1000

            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=True,
                ate=ate,
                cate=cate,
                ate_std=ate_std,
                ate_ci_lower=ate_ci_lower,
                ate_ci_upper=ate_ci_upper,
                propensity_scores=propensity_scores,
                estimation_time_ms=elapsed,
                raw_estimate=model,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(f"DRLearner failed: {e}")
            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                estimation_time_ms=elapsed,
            )


class OLSWrapper(BaseEstimatorWrapper):
    """Simple OLS fallback estimator."""

    def __init__(self, config: EstimatorConfig):
        self.config = config

    @property
    def estimator_type(self) -> EstimatorType:
        return EstimatorType.OLS

    def fit(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        **kwargs,
    ) -> EstimatorResult:
        import time

        start = time.perf_counter()

        try:
            from sklearn.linear_model import LinearRegression

            X = covariates.values
            X_with_treatment = np.column_stack([treatment, X])

            model = LinearRegression()
            model.fit(X_with_treatment, outcome)

            ate = float(model.coef_[0])  # Treatment coefficient

            # Bootstrap standard error
            n_boot = 100
            boot_ates = []
            for _ in range(n_boot):
                idx = np.random.choice(len(treatment), len(treatment), replace=True)
                m = LinearRegression()
                m.fit(X_with_treatment[idx], outcome[idx])
                boot_ates.append(m.coef_[0])

            ate_std = float(np.std(boot_ates))
            ate_ci_lower = ate - 1.96 * ate_std
            ate_ci_upper = ate + 1.96 * ate_std

            # Constant CATE (OLS gives ATE only)
            cate = np.full(len(treatment), ate)

            # Propensity scores
            from sklearn.linear_model import LogisticRegressionCV

            ps_model = LogisticRegressionCV(cv=3, max_iter=500)
            ps_model.fit(X, treatment)
            propensity_scores = ps_model.predict_proba(X)[:, 1]

            elapsed = (time.perf_counter() - start) * 1000

            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=True,
                ate=ate,
                cate=cate,
                ate_std=ate_std,
                ate_ci_lower=ate_ci_lower,
                ate_ci_upper=ate_ci_upper,
                propensity_scores=propensity_scores,
                estimation_time_ms=elapsed,
                raw_estimate=model,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"OLS fallback failed: {e}")
            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                estimation_time_ms=elapsed,
            )


class SLearnerWrapper(BaseEstimatorWrapper):
    """
    S-Learner (Single-model Learner) for heterogeneous treatment effects.

    Trains a single model on both treatment and control groups,
    including treatment as a feature. CATE is estimated as the
    difference in predictions when treatment is set to 1 vs 0.

    Pros: Simple, works with any base learner
    Cons: May underestimate treatment effect heterogeneity
    """

    def __init__(self, config: EstimatorConfig):
        self.config = config

    @property
    def estimator_type(self) -> EstimatorType:
        return EstimatorType.S_LEARNER

    def fit(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        **kwargs,
    ) -> EstimatorResult:
        import time

        start = time.perf_counter()

        try:
            from sklearn.ensemble import GradientBoostingRegressor

            X = covariates.values
            # Include treatment as feature
            X_with_treatment = np.column_stack([treatment, X])

            # Train single model
            base_learner = self.config.params.get("base_learner", None)
            if base_learner is None:
                base_learner = GradientBoostingRegressor(
                    n_estimators=self.config.params.get("n_estimators", 100),
                    max_depth=self.config.params.get("max_depth", 5),
                    random_state=self.config.params.get("random_state", 42),
                )

            base_learner.fit(X_with_treatment, outcome)

            # Estimate CATE: E[Y|X, T=1] - E[Y|X, T=0]
            X_treat_1 = np.column_stack([np.ones(len(X)), X])
            X_treat_0 = np.column_stack([np.zeros(len(X)), X])
            cate = base_learner.predict(X_treat_1) - base_learner.predict(X_treat_0)

            ate = float(np.mean(cate))
            ate_std = float(np.std(cate) / np.sqrt(len(cate)))
            ate_ci_lower = ate - 1.96 * ate_std
            ate_ci_upper = ate + 1.96 * ate_std

            # Propensity scores
            from sklearn.linear_model import LogisticRegressionCV

            ps_model = LogisticRegressionCV(cv=3, max_iter=500)
            ps_model.fit(X, treatment)
            propensity_scores = ps_model.predict_proba(X)[:, 1]

            elapsed = (time.perf_counter() - start) * 1000

            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=True,
                ate=ate,
                cate=cate,
                ate_std=ate_std,
                ate_ci_lower=ate_ci_lower,
                ate_ci_upper=ate_ci_upper,
                propensity_scores=propensity_scores,
                estimation_time_ms=elapsed,
                raw_estimate=base_learner,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(f"S-Learner failed: {e}")
            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                estimation_time_ms=elapsed,
            )


class TLearnerWrapper(BaseEstimatorWrapper):
    """
    T-Learner (Two-model Learner) for heterogeneous treatment effects.

    Trains separate models for treatment and control groups.
    CATE is estimated as the difference in predictions.

    Pros: Captures heterogeneity well when treatment effects vary
    Cons: May have high variance with small sample sizes
    """

    def __init__(self, config: EstimatorConfig):
        self.config = config

    @property
    def estimator_type(self) -> EstimatorType:
        return EstimatorType.T_LEARNER

    def fit(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        **kwargs,
    ) -> EstimatorResult:
        import time

        start = time.perf_counter()

        try:
            from sklearn.ensemble import GradientBoostingRegressor

            X = covariates.values

            # Split by treatment
            X_1 = X[treatment == 1]
            X_0 = X[treatment == 0]
            Y_1 = outcome[treatment == 1]
            Y_0 = outcome[treatment == 0]

            # Base learner configuration
            base_params = {
                "n_estimators": self.config.params.get("n_estimators", 100),
                "max_depth": self.config.params.get("max_depth", 5),
                "random_state": self.config.params.get("random_state", 42),
            }

            # Train separate models
            model_1 = GradientBoostingRegressor(**base_params)
            model_0 = GradientBoostingRegressor(**base_params)
            model_1.fit(X_1, Y_1)
            model_0.fit(X_0, Y_0)

            # Estimate CATE: μ1(X) - μ0(X)
            cate = model_1.predict(X) - model_0.predict(X)

            ate = float(np.mean(cate))
            ate_std = float(np.std(cate) / np.sqrt(len(cate)))
            ate_ci_lower = ate - 1.96 * ate_std
            ate_ci_upper = ate + 1.96 * ate_std

            # Propensity scores
            from sklearn.linear_model import LogisticRegressionCV

            ps_model = LogisticRegressionCV(cv=3, max_iter=500)
            ps_model.fit(X, treatment)
            propensity_scores = ps_model.predict_proba(X)[:, 1]

            elapsed = (time.perf_counter() - start) * 1000

            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=True,
                ate=ate,
                cate=cate,
                ate_std=ate_std,
                ate_ci_lower=ate_ci_lower,
                ate_ci_upper=ate_ci_upper,
                propensity_scores=propensity_scores,
                estimation_time_ms=elapsed,
                raw_estimate={"model_1": model_1, "model_0": model_0},
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(f"T-Learner failed: {e}")
            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                estimation_time_ms=elapsed,
            )


class XLearnerWrapper(BaseEstimatorWrapper):
    """
    X-Learner for heterogeneous treatment effects.

    A two-stage meta-learner that:
    1. Fits T-learner models to get initial CATE estimates
    2. Uses imputed treatment effects to train second-stage models
    3. Combines using propensity-weighted average

    Pros: Performs well with unbalanced treatment groups
    Cons: More complex, requires propensity score estimation

    Reference: Künzel et al. (2019) "Metalearners for Estimating
    Heterogeneous Treatment Effects using Machine Learning"
    """

    def __init__(self, config: EstimatorConfig):
        self.config = config

    @property
    def estimator_type(self) -> EstimatorType:
        return EstimatorType.X_LEARNER

    def fit(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        **kwargs,
    ) -> EstimatorResult:
        import time

        start = time.perf_counter()

        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.linear_model import LogisticRegressionCV

            X = covariates.values

            # Split by treatment
            X_1 = X[treatment == 1]
            X_0 = X[treatment == 0]
            Y_1 = outcome[treatment == 1]
            Y_0 = outcome[treatment == 0]

            # Base learner configuration
            base_params = {
                "n_estimators": self.config.params.get("n_estimators", 100),
                "max_depth": self.config.params.get("max_depth", 5),
                "random_state": self.config.params.get("random_state", 42),
            }

            # Stage 1: Fit response models (like T-learner)
            model_1 = GradientBoostingRegressor(**base_params)
            model_0 = GradientBoostingRegressor(**base_params)
            model_1.fit(X_1, Y_1)
            model_0.fit(X_0, Y_0)

            # Stage 2: Compute imputed treatment effects
            # For treated: τ̃1 = Y1 - μ0(X1)
            tau_1 = Y_1 - model_0.predict(X_1)
            # For control: τ̃0 = μ1(X0) - Y0
            tau_0 = model_1.predict(X_0) - Y_0

            # Fit second-stage models on imputed effects
            model_tau_1 = GradientBoostingRegressor(**base_params)
            model_tau_0 = GradientBoostingRegressor(**base_params)
            model_tau_1.fit(X_1, tau_1)
            model_tau_0.fit(X_0, tau_0)

            # Propensity scores for weighting
            ps_model = LogisticRegressionCV(cv=3, max_iter=500)
            ps_model.fit(X, treatment)
            propensity_scores = ps_model.predict_proba(X)[:, 1]

            # Combine using propensity-weighted average:
            # τ̂(x) = e(x) * τ̂0(x) + (1 - e(x)) * τ̂1(x)
            tau_hat_1 = model_tau_1.predict(X)
            tau_hat_0 = model_tau_0.predict(X)
            cate = propensity_scores * tau_hat_0 + (1 - propensity_scores) * tau_hat_1

            ate = float(np.mean(cate))
            ate_std = float(np.std(cate) / np.sqrt(len(cate)))
            ate_ci_lower = ate - 1.96 * ate_std
            ate_ci_upper = ate + 1.96 * ate_std

            elapsed = (time.perf_counter() - start) * 1000

            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=True,
                ate=ate,
                cate=cate,
                ate_std=ate_std,
                ate_ci_lower=ate_ci_lower,
                ate_ci_upper=ate_ci_upper,
                propensity_scores=propensity_scores,
                estimation_time_ms=elapsed,
                raw_estimate={
                    "model_1": model_1,
                    "model_0": model_0,
                    "model_tau_1": model_tau_1,
                    "model_tau_0": model_tau_0,
                    "ps_model": ps_model,
                },
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(f"X-Learner failed: {e}")
            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                estimation_time_ms=elapsed,
            )


class OrthoForestWrapper(BaseEstimatorWrapper):
    """
    Orthogonal Random Forest (OrthoForest) for high-dimensional CATE.

    Uses double machine learning with random forest splitting.
    Provides valid confidence intervals even in high dimensions.

    Reference: Oprescu et al. (2019) "Orthogonal Random Forest
    for Causal Inference"
    """

    def __init__(self, config: EstimatorConfig):
        self.config = config

    @property
    def estimator_type(self) -> EstimatorType:
        return EstimatorType.ORTHO_FOREST

    def fit(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        **kwargs,
    ) -> EstimatorResult:
        import time

        start = time.perf_counter()

        try:
            from econml.orf import DMLOrthoForest

            X = covariates.values

            # Configure OrthoForest
            params = {
                "n_trees": self.config.params.get("n_trees", 100),
                "min_leaf_size": self.config.params.get("min_leaf_size", 10),
                "max_depth": self.config.params.get("max_depth", None),
                "random_state": self.config.params.get("random_state", 42),
            }

            # Fit model
            model = DMLOrthoForest(**params)
            model.fit(outcome, treatment, X=X, W=X)

            # Get estimates
            cate = model.effect(X)
            ate = float(np.mean(cate))
            ate_std = float(np.std(cate) / np.sqrt(len(cate)))

            # Confidence intervals from OrthoForest
            try:
                cate_inf = model.effect_inference(X)
                ci = cate_inf.conf_int_mean()
                ate_ci_lower, ate_ci_upper = float(ci[0]), float(ci[1])
            except Exception:
                ate_ci_lower = ate - 1.96 * ate_std
                ate_ci_upper = ate + 1.96 * ate_std

            # Propensity scores
            from sklearn.linear_model import LogisticRegressionCV

            ps_model = LogisticRegressionCV(cv=3, max_iter=500)
            ps_model.fit(X, treatment)
            propensity_scores = ps_model.predict_proba(X)[:, 1]

            elapsed = (time.perf_counter() - start) * 1000

            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=True,
                ate=ate,
                cate=cate,
                ate_std=ate_std,
                ate_ci_lower=ate_ci_lower,
                ate_ci_upper=ate_ci_upper,
                propensity_scores=propensity_scores,
                estimation_time_ms=elapsed,
                raw_estimate=model,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(f"OrthoForest failed: {e}")
            return EstimatorResult(
                estimator_type=self.estimator_type,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                estimation_time_ms=elapsed,
            )


# Estimator factory
ESTIMATOR_WRAPPERS: dict[EstimatorType, type[BaseEstimatorWrapper]] = {
    EstimatorType.CAUSAL_FOREST: CausalForestWrapper,
    EstimatorType.LINEAR_DML: LinearDMLWrapper,
    EstimatorType.DRLEARNER: DRLearnerWrapper,
    EstimatorType.S_LEARNER: SLearnerWrapper,
    EstimatorType.T_LEARNER: TLearnerWrapper,
    EstimatorType.X_LEARNER: XLearnerWrapper,
    EstimatorType.ORTHO_FOREST: OrthoForestWrapper,
    EstimatorType.OLS: OLSWrapper,
}


class EstimatorSelector:
    """
    Selects the best causal estimator using energy score.

    Instead of using the first successful estimator (legacy approach),
    this selector evaluates all estimators and picks the one with
    the lowest energy score.

    Usage:
        selector = EstimatorSelector()
        result = selector.select(
            treatment=df['treatment'].values,
            outcome=df['outcome'].values,
            covariates=df[['x1', 'x2', 'x3']]
        )
        print(f"Selected: {result.selected.estimator_type}")
        print(f"ATE: {result.selected.ate:.4f}")
        print(f"Energy Score: {result.selected.energy_score:.4f}")
    """

    def __init__(self, config: Optional[EstimatorSelectorConfig] = None):
        """Initialize selector with configuration."""
        self.config = config or EstimatorSelectorConfig()
        self.energy_calculator = EnergyScoreCalculator(self.config.energy_score_config)

        # Build estimator chain
        self.estimators: list[BaseEstimatorWrapper] = []
        for est_config in sorted(self.config.estimators, key=lambda x: x.priority):
            if est_config.enabled and est_config.estimator_type in ESTIMATOR_WRAPPERS:
                wrapper_class = ESTIMATOR_WRAPPERS[est_config.estimator_type]
                self.estimators.append(wrapper_class(est_config))

    def select(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        **kwargs,
    ) -> SelectionResult:
        """
        Evaluate all estimators and select the best one.

        Args:
            treatment: Binary treatment indicator
            outcome: Observed outcomes
            covariates: Covariate DataFrame
            **kwargs: Additional arguments passed to estimators

        Returns:
            SelectionResult with selected estimator and comparison data
        """
        import time

        total_start = time.perf_counter()

        results: list[EstimatorResult] = []

        # Evaluate each estimator
        for wrapper in self.estimators:
            logger.info(f"Evaluating {wrapper.estimator_type.value}...")

            result = wrapper.fit(treatment, outcome, covariates, **kwargs)

            # Compute energy score for successful estimations
            if result.success and result.cate is not None:
                # Fix 5B: Recursion guard for energy score computation
                import sys

                old_limit = sys.getrecursionlimit()
                sys.setrecursionlimit(max(old_limit, 5000))
                try:
                    energy_result = self.energy_calculator.compute(
                        treatment=treatment,
                        outcome=outcome,
                        covariates=covariates,
                        estimated_effects=result.cate,
                        propensity_scores=result.propensity_scores,
                        estimator_name=wrapper.estimator_type.value,
                    )
                    result.energy_score_result = energy_result
                    logger.info(
                        f"  {wrapper.estimator_type.value}: "
                        f"ATE={result.ate:.4f}, Energy={energy_result.energy_score:.4f}"
                    )
                except RecursionError:
                    logger.warning(
                        f"Energy score selection hit recursion limit for "
                        f"{wrapper.estimator_type.value}, using legacy path"
                    )
                    energy_result = None
                    result.energy_score_result = None
                finally:
                    sys.setrecursionlimit(old_limit)

            results.append(result)

        # Select based on strategy
        if self.config.strategy == SelectionStrategy.BEST_ENERGY_SCORE:
            selection = self._select_best_energy(results)
        elif self.config.strategy == SelectionStrategy.FIRST_SUCCESS:
            selection = self._select_first_success(results)
        else:
            selection = self._select_best_energy(results)  # Default

        # Build energy score comparison
        energy_scores = {r.estimator_type.value: r.energy_score for r in results if r.success}

        # Compute gap between best and second best
        sorted_scores = sorted([s for s in energy_scores.values() if np.isfinite(s)])
        energy_score_gap = 0.0
        if len(sorted_scores) >= 2:
            energy_score_gap = sorted_scores[1] - sorted_scores[0]

        total_time = (time.perf_counter() - total_start) * 1000

        return SelectionResult(
            selected=selection,
            selection_strategy=self.config.strategy,
            all_results=results,
            selection_reason=self._get_selection_reason(selection, results),
            total_time_ms=total_time,
            energy_scores=energy_scores,
            energy_score_gap=energy_score_gap,
        )

    def _select_best_energy(self, results: list[EstimatorResult]) -> EstimatorResult:
        """Select estimator with lowest energy score."""
        successful = [r for r in results if r.success]

        if not successful:
            logger.warning("All estimators failed, using fallback")
            # Return the last failure or create a dummy result
            return (
                results[-1]
                if results
                else EstimatorResult(
                    estimator_type=EstimatorType.OLS,
                    success=False,
                    error_message="All estimators failed",
                )
            )

        # Sort by energy score (lower is better)
        sorted_results = sorted(successful, key=lambda r: r.energy_score)

        best = sorted_results[0]

        # Log warning if energy score is high
        if best.energy_score > self.config.max_acceptable_energy_score:
            logger.warning(
                f"Best energy score ({best.energy_score:.4f}) exceeds threshold "
                f"({self.config.max_acceptable_energy_score}). Results may be unreliable."
            )

        return best

    def _select_first_success(self, results: list[EstimatorResult]) -> EstimatorResult:
        """Legacy: select first successful estimator."""
        for result in results:
            if result.success:
                return result

        # All failed
        return (
            results[-1]
            if results
            else EstimatorResult(
                estimator_type=EstimatorType.OLS,
                success=False,
                error_message="All estimators failed",
            )
        )

    def _get_selection_reason(
        self, selected: EstimatorResult, all_results: list[EstimatorResult]
    ) -> str:
        """Generate human-readable selection reason."""
        if not selected.success:
            return "All estimators failed; returning last attempt"

        successful = [r for r in all_results if r.success]

        if len(successful) == 1:
            return f"Only {selected.estimator_type.value} succeeded"

        if self.config.strategy == SelectionStrategy.BEST_ENERGY_SCORE:
            scores = [(r.estimator_type.value, r.energy_score) for r in successful]
            scores_str = ", ".join(f"{name}={score:.4f}" for name, score in scores)
            return f"Lowest energy score among: {scores_str}"

        return f"Selected by {self.config.strategy.value} strategy"


def select_best_estimator(
    treatment: NDArray[np.int_],
    outcome: NDArray[np.float64],
    covariates: pd.DataFrame,
    config: Optional[EstimatorSelectorConfig] = None,
    **kwargs,
) -> SelectionResult:
    """
    Convenience function for estimator selection.

    Example:
        result = select_best_estimator(
            treatment=df['T'].values,
            outcome=df['Y'].values,
            covariates=df[['X1', 'X2', 'X3']]
        )
        print(f"Best estimator: {result.selected.estimator_type}")
        print(f"ATE: {result.selected.ate}")
    """
    selector = EstimatorSelector(config)
    return selector.select(treatment, outcome, covariates, **kwargs)
