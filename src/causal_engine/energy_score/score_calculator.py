"""
Energy Score Implementation for Causal Estimator Evaluation

The energy score measures the quality of treatment effect estimates by evaluating
how well the estimated propensity scores and outcomes align with observed data.
Lower energy scores indicate better causal estimates.

This implementation follows the methodology from:
- "AutoML for Causal Inference: Selecting Treatment Effect Estimators" (CausalTune paper)
- Székely, G. J., & Rizzo, M. L. (2013). Energy statistics

Integration:
    Used by EstimatorSelector to choose the best estimator from the fallback chain.
    Replaces "first success" with "best score" selection strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class EnergyScoreVariant(str, Enum):
    """Variants of energy score computation."""

    STANDARD = "standard"  # Classic energy distance
    WEIGHTED = "weighted"  # IPW-weighted energy distance
    DOUBLY_ROBUST = "doubly_robust"  # DR-adjusted energy distance


@dataclass
class EnergyScoreResult:
    """Result of energy score computation for a single estimator."""

    estimator_name: str
    energy_score: float

    # Component scores for interpretability
    treatment_balance_score: float
    outcome_fit_score: float
    propensity_calibration: float

    # Metadata
    n_samples: int
    n_treated: int
    n_control: int
    computation_time_ms: float

    # Optional: bootstrap confidence interval
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    bootstrap_std: Optional[float] = None

    # Raw components for debugging
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if the score is valid (not NaN or Inf)."""
        return np.isfinite(self.energy_score)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "estimator_name": self.estimator_name,
            "energy_score": float(self.energy_score) if np.isfinite(self.energy_score) else None,
            "treatment_balance_score": float(self.treatment_balance_score),
            "outcome_fit_score": float(self.outcome_fit_score),
            "propensity_calibration": float(self.propensity_calibration),
            "n_samples": self.n_samples,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "computation_time_ms": self.computation_time_ms,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "bootstrap_std": self.bootstrap_std,
            "details": self.details,
        }


@dataclass
class EnergyScoreConfig:
    """Configuration for energy score computation."""

    variant: EnergyScoreVariant = EnergyScoreVariant.DOUBLY_ROBUST

    # Component weights (must sum to 1.0)
    weight_treatment_balance: float = 0.35
    weight_outcome_fit: float = 0.45
    weight_propensity_calibration: float = 0.20

    # Bootstrap settings
    enable_bootstrap: bool = True
    n_bootstrap: int = 100
    bootstrap_confidence: float = 0.95

    # Propensity score bounds (for numerical stability)
    propensity_clip_min: float = 0.01
    propensity_clip_max: float = 0.99

    # Sample size limits
    min_samples_per_group: int = 30
    max_samples_for_exact: int = 5000  # Use sampling above this

    def __post_init__(self):
        """Validate configuration."""
        total_weight = (
            self.weight_treatment_balance
            + self.weight_outcome_fit
            + self.weight_propensity_calibration
        )
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(f"Component weights must sum to 1.0, got {total_weight}")


class EnergyScoreCalculator:
    """
    Calculates energy score for causal effect estimators.

    The energy score evaluates how well an estimator's assumptions hold by
    measuring the discrepancy between treated and control groups after
    adjustment for confounders.

    Usage:
        calculator = EnergyScoreCalculator()
        result = calculator.compute(
            treatment=treatment_array,
            outcome=outcome_array,
            covariates=covariate_df,
            estimated_effects=cate_array,
            propensity_scores=ps_array,
            estimator_name="CausalForest"
        )
    """

    def __init__(self, config: Optional[EnergyScoreConfig] = None):
        """Initialize calculator with configuration."""
        self.config = config or EnergyScoreConfig()

    def compute(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        estimated_effects: NDArray[np.float64],
        propensity_scores: Optional[NDArray[np.float64]] = None,
        estimator_name: str = "unknown",
        _skip_bootstrap: bool = False,
    ) -> EnergyScoreResult:
        """
        Compute energy score for an estimator's output.

        Args:
            treatment: Binary treatment indicator (0/1)
            outcome: Observed outcomes
            covariates: Covariate matrix used for estimation
            estimated_effects: Estimated CATE or constant ATE for each unit
            propensity_scores: Optional propensity scores (will estimate if not provided)
            estimator_name: Name of the estimator for logging

        Returns:
            EnergyScoreResult with composite score and components
        """
        import time

        start_time = time.perf_counter()

        # Validate inputs
        n = len(treatment)
        if not all(len(arr) == n for arr in [outcome, estimated_effects]):
            raise ValueError("All arrays must have same length")

        if len(covariates) != n:
            raise ValueError("Covariates must have same number of rows as treatment")

        # Identify treatment groups
        treated_mask = treatment == 1
        control_mask = treatment == 0
        n_treated = treated_mask.sum()
        n_control = control_mask.sum()

        # Check minimum sample sizes
        if n_treated < self.config.min_samples_per_group:
            logger.warning(f"Too few treated units: {n_treated}")
        if n_control < self.config.min_samples_per_group:
            logger.warning(f"Too few control units: {n_control}")

        # Estimate propensity scores if not provided
        if propensity_scores is None:
            propensity_scores = self._estimate_propensity(covariates, treatment)

        # Clip propensity scores for stability
        propensity_scores = np.clip(
            propensity_scores, self.config.propensity_clip_min, self.config.propensity_clip_max
        )

        # Compute component scores
        treatment_balance = self._compute_treatment_balance(
            covariates, treated_mask, control_mask, propensity_scores
        )

        outcome_fit = self._compute_outcome_fit(
            outcome, treatment, estimated_effects, propensity_scores
        )

        propensity_cal = self._compute_propensity_calibration(treatment, propensity_scores)

        # Compute weighted composite score
        energy_score = (
            self.config.weight_treatment_balance * treatment_balance
            + self.config.weight_outcome_fit * outcome_fit
            + self.config.weight_propensity_calibration * propensity_cal
        )

        # Bootstrap confidence interval if enabled
        ci_lower, ci_upper, bootstrap_std = None, None, None
        if (
            self.config.enable_bootstrap
            and not _skip_bootstrap
            and n <= self.config.max_samples_for_exact
        ):
            adaptive_n_bootstrap = min(self.config.n_bootstrap, max(20, n // 30))
            ci_lower, ci_upper, bootstrap_std = self._bootstrap_ci(
                treatment,
                outcome,
                covariates,
                estimated_effects,
                propensity_scores,
                n_bootstrap=adaptive_n_bootstrap,
            )

        computation_time = (time.perf_counter() - start_time) * 1000

        return EnergyScoreResult(
            estimator_name=estimator_name,
            energy_score=energy_score,
            treatment_balance_score=treatment_balance,
            outcome_fit_score=outcome_fit,
            propensity_calibration=propensity_cal,
            n_samples=n,
            n_treated=n_treated,
            n_control=n_control,
            computation_time_ms=computation_time,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            bootstrap_std=bootstrap_std,
            details={
                "variant": self.config.variant.value,
                "weights": {
                    "treatment_balance": self.config.weight_treatment_balance,
                    "outcome_fit": self.config.weight_outcome_fit,
                    "propensity_calibration": self.config.weight_propensity_calibration,
                },
            },
        )

    def _estimate_propensity(
        self, covariates: pd.DataFrame, treatment: NDArray[np.int_]
    ) -> NDArray[np.float64]:
        """Estimate propensity scores using logistic regression."""
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler

        # Standardize covariates
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(covariates)

        # Fit logistic regression with cross-validation
        lr = LogisticRegressionCV(
            cv=5, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42
        )
        lr.fit(X_scaled, treatment)

        return lr.predict_proba(X_scaled)[:, 1]

    def _compute_treatment_balance(
        self,
        covariates: pd.DataFrame,
        treated_mask: NDArray[np.bool_],
        control_mask: NDArray[np.bool_],
        propensity_scores: NDArray[np.float64],
    ) -> float:
        """
        Compute treatment balance score using energy distance.

        Measures how well the weighted covariate distributions match
        between treatment groups after IPW adjustment.
        """
        # Convert to numpy
        X = covariates.values

        # Compute IPW weights
        weights_treated = 1.0 / propensity_scores[treated_mask]
        weights_control = 1.0 / (1.0 - propensity_scores[control_mask])

        # Normalize weights
        weights_treated = weights_treated / weights_treated.sum()
        weights_control = weights_control / weights_control.sum()

        # Extract group data
        X_treated = X[treated_mask]
        X_control = X[control_mask]

        # Compute weighted energy distance
        energy_dist = self._weighted_energy_distance(
            X_treated, X_control, weights_treated, weights_control
        )

        # Normalize to [0, 1] range (higher is worse)
        # Using empirical scaling factor
        normalized = np.tanh(energy_dist / 2.0)

        return float(normalized)

    def _weighted_energy_distance(
        self,
        X1: NDArray[np.float64],
        X2: NDArray[np.float64],
        w1: NDArray[np.float64],
        w2: NDArray[np.float64],
    ) -> float:
        """
        Compute weighted energy distance between two samples.

        E(X, Y) = 2*E||X-Y|| - E||X-X'|| - E||Y-Y'||
        """
        # Sample if too large
        max_n = self.config.max_samples_for_exact
        if len(X1) > max_n or len(X2) > max_n:
            idx1 = np.random.choice(len(X1), min(len(X1), max_n), replace=False, p=w1)
            idx2 = np.random.choice(len(X2), min(len(X2), max_n), replace=False, p=w2)
            X1, X2 = X1[idx1], X2[idx2]
            w1 = np.ones(len(X1)) / len(X1)
            w2 = np.ones(len(X2)) / len(X2)

        # Cross-group distances
        D12 = cdist(X1, X2, "euclidean")
        cross_term = 2.0 * float(w1 @ D12 @ w2)

        # Within-group distances
        D11 = cdist(X1, X1, "euclidean")
        within1 = float(w1 @ D11 @ w1)

        D22 = cdist(X2, X2, "euclidean")
        within2 = float(w2 @ D22 @ w2)

        return float(cross_term - within1 - within2)

    def _compute_outcome_fit(
        self,
        outcome: NDArray[np.float64],
        treatment: NDArray[np.int_],
        estimated_effects: NDArray[np.float64],
        propensity_scores: NDArray[np.float64],
    ) -> float:
        """
        Compute outcome fit score using doubly-robust residuals.

        Measures how well the estimated treatment effects explain
        the observed outcome variation.
        """
        # Compute pseudo-outcomes (DR transformation)
        ps = propensity_scores
        T = treatment
        Y = outcome
        tau = estimated_effects

        # Estimate outcome models (simplified: use linear regression)

        # We need mu_0 and mu_1 estimates - approximate with simple model
        # In practice, these would come from the estimator
        treated_mask = T == 1
        control_mask = T == 0

        mu_1_estimate = np.mean(Y[treated_mask]) if treated_mask.any() else 0
        mu_0_estimate = np.mean(Y[control_mask]) if control_mask.any() else 0

        # DR pseudo-outcome
        pseudo_outcome = (
            (T * (Y - mu_1_estimate)) / ps
            - ((1 - T) * (Y - mu_0_estimate)) / (1 - ps)
            + (mu_1_estimate - mu_0_estimate)
        )

        # Residual: how far is our CATE estimate from pseudo-outcome
        residuals = pseudo_outcome - tau

        # Normalized RMSE
        rmse = np.sqrt(np.mean(residuals**2))
        outcome_std = np.std(Y) + 1e-8
        normalized_rmse = rmse / outcome_std

        # Cap at 1.0
        return float(min(normalized_rmse, 1.0))

    def _compute_propensity_calibration(
        self, treatment: NDArray[np.int_], propensity_scores: NDArray[np.float64]
    ) -> float:
        """
        Compute propensity score calibration.

        Measures how well the propensity scores match observed
        treatment rates across deciles.
        """
        # Create decile bins
        n_bins = 10
        bins = np.percentile(propensity_scores, np.linspace(0, 100, n_bins + 1))
        bins[0] = 0
        bins[-1] = 1

        calibration_errors = []

        for i in range(n_bins):
            mask = (propensity_scores >= bins[i]) & (propensity_scores < bins[i + 1])
            if mask.sum() > 0:
                predicted_rate = propensity_scores[mask].mean()
                actual_rate = treatment[mask].mean()
                calibration_errors.append(abs(predicted_rate - actual_rate))

        if not calibration_errors:
            return 1.0  # Worst case if no valid bins

        # Mean absolute calibration error
        mace = np.mean(calibration_errors)

        return float(mace)

    def _bootstrap_ci(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        estimated_effects: NDArray[np.float64],
        propensity_scores: NDArray[np.float64],
        n_bootstrap: int | None = None,
        time_budget_s: float = 5.0,
    ) -> tuple[float, float, float]:
        """Compute bootstrap confidence interval for energy score."""
        import time as _time

        start = _time.perf_counter()
        n = len(treatment)
        n_bootstrap = n_bootstrap or self.config.n_bootstrap
        bootstrap_scores = []

        for i in range(n_bootstrap):
            if _time.perf_counter() - start > time_budget_s:
                logger.info(
                    "Bootstrap stopped at %d/%d iterations (time budget %.1fs)",
                    i,
                    n_bootstrap,
                    time_budget_s,
                )
                break
            # Resample with replacement
            idx = np.random.choice(n, n, replace=True)

            result = self.compute(
                treatment=treatment[idx],
                outcome=outcome[idx],
                covariates=covariates.iloc[idx].reset_index(drop=True),
                estimated_effects=estimated_effects[idx],
                propensity_scores=propensity_scores[idx],
                estimator_name="bootstrap",
                _skip_bootstrap=True,
            )

            if result.is_valid:
                bootstrap_scores.append(result.energy_score)

        if len(bootstrap_scores) < 10:
            return None, None, None

        alpha = 1 - self.config.bootstrap_confidence
        ci_lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))
        bootstrap_std = float(np.std(bootstrap_scores))

        return ci_lower, ci_upper, bootstrap_std


def compute_energy_score(
    treatment: NDArray[np.int_],
    outcome: NDArray[np.float64],
    covariates: pd.DataFrame,
    estimated_effects: NDArray[np.float64],
    propensity_scores: Optional[NDArray[np.float64]] = None,
    estimator_name: str = "unknown",
    config: Optional[EnergyScoreConfig] = None,
) -> EnergyScoreResult:
    """
    Convenience function for computing energy score.

    Example:
        result = compute_energy_score(
            treatment=df['treatment'].values,
            outcome=df['outcome'].values,
            covariates=df[['x1', 'x2', 'x3']],
            estimated_effects=cate_estimates,
            estimator_name="CausalForest"
        )
        print(f"Energy Score: {result.energy_score:.4f}")
    """
    calculator = EnergyScoreCalculator(config)
    return calculator.compute(
        treatment=treatment,
        outcome=outcome,
        covariates=covariates,
        estimated_effects=estimated_effects,
        propensity_scores=propensity_scores,
        estimator_name=estimator_name,
    )
