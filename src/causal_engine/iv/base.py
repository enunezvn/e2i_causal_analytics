"""
Instrumental Variable (IV) Estimation - Base Classes

Provides foundational types and abstract base classes for IV estimation:
- IVResult: Result container for IV estimates
- IVConfig: Configuration for IV estimators
- BaseIVEstimator: Abstract base class for IV implementations

Use Cases:
- Endogeneity correction when treatment is correlated with unobservables
- Natural experiments with valid instruments
- Regression discontinuity designs

References:
    - Angrist & Pischke (2009) "Mostly Harmless Econometrics"
    - Imbens & Angrist (1994) "Identification and Estimation of LATE"
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

logger = logging.getLogger(__name__)


class IVEstimatorType(str, Enum):
    """Supported IV estimator types."""

    TWO_STAGE_LS = "2sls"
    LIML = "liml"
    GMM = "gmm"
    FULLER = "fuller"
    JIVE = "jive"  # Jackknife IV


class InstrumentStrength(str, Enum):
    """Classification of instrument strength."""

    STRONG = "strong"  # F-stat > 10
    MODERATE = "moderate"  # F-stat 5-10
    WEAK = "weak"  # F-stat < 5
    VERY_WEAK = "very_weak"  # F-stat < 2


@dataclass
class IVDiagnostics:
    """Diagnostic statistics for IV estimation."""

    # First-stage F-statistic (Staiger-Stock rule: F > 10)
    first_stage_f_stat: float = 0.0
    first_stage_f_pvalue: float = 1.0

    # Instrument strength classification
    instrument_strength: InstrumentStrength = InstrumentStrength.WEAK

    # Weak instrument robust inference
    anderson_rubin_stat: float = 0.0
    anderson_rubin_pvalue: float = 1.0

    # Overidentification test (if k > 1 instruments)
    sargan_stat: Optional[float] = None
    sargan_pvalue: Optional[float] = None
    hansen_j_stat: Optional[float] = None
    hansen_j_pvalue: Optional[float] = None

    # Endogeneity test (Hausman)
    hausman_stat: Optional[float] = None
    hausman_pvalue: Optional[float] = None

    # Partial R-squared of instruments
    partial_r_squared: float = 0.0

    # Stock-Yogo critical values for weak instruments
    stock_yogo_10pct: float = 16.38  # 10% maximal IV size
    stock_yogo_15pct: float = 8.96
    stock_yogo_20pct: float = 6.66
    stock_yogo_25pct: float = 5.53

    def is_weak_instrument(self) -> bool:
        """Check if instruments are weak using Staiger-Stock rule."""
        return self.first_stage_f_stat < 10.0

    def passes_overid_test(self, alpha: float = 0.05) -> bool:
        """Check if overidentification test passes."""
        if self.sargan_pvalue is not None:
            return self.sargan_pvalue > alpha
        if self.hansen_j_pvalue is not None:
            return self.hansen_j_pvalue > alpha
        return True  # Not overidentified (exactly identified)

    def is_endogenous(self, alpha: float = 0.05) -> bool:
        """Check if treatment is likely endogenous (Hausman test)."""
        if self.hausman_pvalue is not None:
            return self.hausman_pvalue < alpha
        return True  # Assume endogenous if not tested

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "first_stage_f_stat": self.first_stage_f_stat,
            "first_stage_f_pvalue": self.first_stage_f_pvalue,
            "instrument_strength": self.instrument_strength.value,
            "partial_r_squared": self.partial_r_squared,
            "anderson_rubin_stat": self.anderson_rubin_stat,
            "anderson_rubin_pvalue": self.anderson_rubin_pvalue,
            "sargan_stat": self.sargan_stat,
            "sargan_pvalue": self.sargan_pvalue,
            "hausman_stat": self.hausman_stat,
            "hausman_pvalue": self.hausman_pvalue,
            "is_weak_instrument": self.is_weak_instrument(),
            "passes_overid_test": self.passes_overid_test(),
        }


@dataclass
class IVResult:
    """Result from IV estimation."""

    estimator_type: IVEstimatorType
    success: bool

    # Effect estimates
    coefficient: Optional[float] = None  # IV coefficient (LATE)
    std_error: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    t_stat: Optional[float] = None
    p_value: Optional[float] = None

    # First-stage results
    first_stage_coef: Optional[NDArray[np.float64]] = None
    first_stage_std_error: Optional[NDArray[np.float64]] = None
    first_stage_r_squared: float = 0.0

    # Diagnostics
    diagnostics: IVDiagnostics = field(default_factory=IVDiagnostics)

    # Sample info
    n_observations: int = 0
    n_instruments: int = 0
    n_covariates: int = 0

    # Error info
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    # Timing
    estimation_time_ms: float = 0.0

    # Raw estimator for further analysis
    raw_estimate: Optional[Any] = None

    @property
    def late(self) -> Optional[float]:
        """Alias for coefficient - Local Average Treatment Effect."""
        return self.coefficient

    def is_valid(self) -> bool:
        """Check if IV estimate is valid based on diagnostics."""
        if not self.success:
            return False
        if self.diagnostics.is_weak_instrument():
            return False
        if not self.diagnostics.passes_overid_test():
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "estimator_type": self.estimator_type.value,
            "success": self.success,
            "coefficient": self.coefficient,
            "std_error": self.std_error,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "first_stage_r_squared": self.first_stage_r_squared,
            "n_observations": self.n_observations,
            "n_instruments": self.n_instruments,
            "diagnostics": self.diagnostics.to_dict(),
            "is_valid": self.is_valid(),
            "error_message": self.error_message,
            "estimation_time_ms": self.estimation_time_ms,
        }


@dataclass
class IVConfig:
    """Configuration for IV estimators."""

    # Estimation settings
    confidence_level: float = 0.95
    robust_std_errors: bool = True  # Heteroskedasticity-robust SE
    cluster_var: Optional[str] = None  # Cluster-robust SE

    # Weak instrument handling
    weak_iv_robust: bool = True  # Use Anderson-Rubin for weak IV
    weak_iv_threshold: float = 10.0  # F-stat threshold

    # LIML-specific
    fuller_k: Optional[float] = None  # Fuller modification parameter

    # Diagnostic options
    run_diagnostics: bool = True
    run_overid_test: bool = True
    run_hausman_test: bool = True

    # Bootstrap for inference (if needed)
    bootstrap_iterations: int = 0  # 0 = no bootstrap
    bootstrap_seed: int = 42


class BaseIVEstimator(ABC):
    """Abstract base class for IV estimators."""

    def __init__(self, config: Optional[IVConfig] = None):
        """Initialize with configuration."""
        self.config = config or IVConfig()

    @abstractmethod
    def fit(
        self,
        outcome: NDArray[np.float64],
        treatment: NDArray[np.float64],
        instruments: NDArray[np.float64],
        covariates: Optional[NDArray[np.float64]] = None,
        **kwargs
    ) -> IVResult:
        """
        Fit the IV estimator.

        Args:
            outcome: Y - observed outcomes (n,)
            treatment: D - endogenous treatment (n,)
            instruments: Z - instruments (n, k) where k >= 1
            covariates: X - exogenous controls (n, p) or None
            **kwargs: Additional estimator-specific arguments

        Returns:
            IVResult with coefficient estimates and diagnostics
        """
        pass

    @property
    @abstractmethod
    def estimator_type(self) -> IVEstimatorType:
        """Return the estimator type."""
        pass

    def _validate_inputs(
        self,
        outcome: NDArray[np.float64],
        treatment: NDArray[np.float64],
        instruments: NDArray[np.float64],
        covariates: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Validate input arrays."""
        n = len(outcome)

        if len(treatment) != n:
            raise ValueError(f"Treatment length {len(treatment)} != outcome length {n}")

        if instruments.ndim == 1:
            instruments = instruments.reshape(-1, 1)

        if len(instruments) != n:
            raise ValueError(f"Instruments length {len(instruments)} != outcome length {n}")

        if covariates is not None and len(covariates) != n:
            raise ValueError(f"Covariates length {len(covariates)} != outcome length {n}")

        # Check for sufficient instruments (order condition)
        k = instruments.shape[1] if instruments.ndim > 1 else 1
        if k < 1:
            raise ValueError("At least one instrument required")

    def _classify_instrument_strength(self, f_stat: float) -> InstrumentStrength:
        """Classify instrument strength based on F-statistic."""
        if f_stat >= 10:
            return InstrumentStrength.STRONG
        elif f_stat >= 5:
            return InstrumentStrength.MODERATE
        elif f_stat >= 2:
            return InstrumentStrength.WEAK
        else:
            return InstrumentStrength.VERY_WEAK

    def _compute_confidence_interval(
        self,
        coef: float,
        std_error: float,
        df: int,
    ) -> tuple[float, float]:
        """Compute confidence interval."""
        from scipy import stats

        alpha = 1 - self.config.confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        ci_lower = coef - t_crit * std_error
        ci_upper = coef + t_crit * std_error

        return ci_lower, ci_upper
