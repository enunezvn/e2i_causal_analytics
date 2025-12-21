"""Synthetic Data Generators for Causal Validation Benchmarks.

Version: 4.3
Purpose: Generate datasets with known causal effects for CI/CD regression testing

Data Generating Processes (DGPs):
    1. simple_linear: T → Y with known effect, no confounding
    2. confounded_moderate: C → T, C → Y with known adjustment effect
    3. heterogeneous_cate: Segment-specific treatment effects

Reference: docs/E2I_Causal_Validation_Protocol.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class SyntheticDataset:
    """Container for synthetic causal dataset with ground truth.

    Attributes:
        data: DataFrame with treatment, outcome, and covariates
        true_ate: True average treatment effect
        true_cate: True conditional average treatment effects (if heterogeneous)
        tolerance: Acceptable estimation error for CI/CD tests
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        confounder_cols: List of confounder column names
        dgp_name: Name of the data generating process
        n_samples: Number of samples
        seed: Random seed used for reproducibility
    """

    data: pd.DataFrame
    true_ate: float
    true_cate: Optional[Dict[str, float]] = None
    tolerance: float = 0.05
    treatment_col: str = "T"
    outcome_col: str = "Y"
    confounder_cols: List[str] = None
    dgp_name: str = "unknown"
    n_samples: int = 0
    seed: int = 42

    def __post_init__(self):
        if self.confounder_cols is None:
            self.confounder_cols = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "true_ate": self.true_ate,
            "true_cate": self.true_cate,
            "tolerance": self.tolerance,
            "treatment_col": self.treatment_col,
            "outcome_col": self.outcome_col,
            "confounder_cols": self.confounder_cols,
            "dgp_name": self.dgp_name,
            "n_samples": self.n_samples,
            "seed": self.seed,
        }


# ============================================================================
# DATA GENERATING PROCESSES
# ============================================================================


def generate_simple_linear(
    n: int = 10000,
    true_ate: float = 0.50,
    noise_std: float = 0.1,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate simple linear DGP with no confounding.

    Data Generating Process:
        T ~ Bernoulli(0.5)
        Y = true_ate * T + epsilon
        epsilon ~ N(0, noise_std^2)

    This is a baseline sanity check - any valid causal estimator
    should recover the true effect with high accuracy.

    Args:
        n: Number of samples
        true_ate: True average treatment effect
        noise_std: Standard deviation of noise term
        seed: Random seed for reproducibility

    Returns:
        SyntheticDataset with known ground truth
    """
    np.random.seed(seed)

    # Treatment: Random assignment (no confounding)
    T = np.random.binomial(1, 0.5, n)

    # Outcome: Simple linear effect
    epsilon = np.random.normal(0, noise_std, n)
    Y = true_ate * T + epsilon

    data = pd.DataFrame(
        {
            "T": T,
            "Y": Y,
        }
    )

    return SyntheticDataset(
        data=data,
        true_ate=true_ate,
        tolerance=0.05,  # Strict tolerance for simple case
        treatment_col="T",
        outcome_col="Y",
        confounder_cols=[],
        dgp_name="simple_linear",
        n_samples=n,
        seed=seed,
    )


def generate_confounded_moderate(
    n: int = 10000,
    true_ate: float = 0.30,
    confounder_strength: float = 0.40,
    noise_std: float = 0.1,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate DGP with moderate observed confounding.

    Data Generating Process:
        C ~ N(0, 1)                           # Confounder
        T ~ Bernoulli(sigmoid(confounder_strength * C))  # Treatment affected by C
        Y = true_ate * T + confounder_strength * C + epsilon
        epsilon ~ N(0, noise_std^2)

    The naive difference in means will be biased. Correct adjustment
    for C should recover the true_ate.

    Args:
        n: Number of samples
        true_ate: True average treatment effect
        confounder_strength: Strength of confounder on both T and Y
        noise_std: Standard deviation of noise term
        seed: Random seed for reproducibility

    Returns:
        SyntheticDataset with known ground truth
    """
    np.random.seed(seed)

    # Confounder
    C = np.random.normal(0, 1, n)

    # Treatment: Affected by confounder
    prob_T = 1 / (1 + np.exp(-confounder_strength * C))  # Sigmoid
    T = np.random.binomial(1, prob_T)

    # Outcome: Affected by both treatment and confounder
    epsilon = np.random.normal(0, noise_std, n)
    Y = true_ate * T + confounder_strength * C + epsilon

    data = pd.DataFrame(
        {
            "T": T,
            "Y": Y,
            "C": C,
        }
    )

    return SyntheticDataset(
        data=data,
        true_ate=true_ate,
        tolerance=0.05,
        treatment_col="T",
        outcome_col="Y",
        confounder_cols=["C"],
        dgp_name="confounded_moderate",
        n_samples=n,
        seed=seed,
    )


def generate_heterogeneous_cate(
    n: int = 50000,
    base_effect: float = 0.20,
    segment_effects: Optional[Dict[str, float]] = None,
    noise_std: float = 0.1,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate DGP with heterogeneous treatment effects by segment.

    Data Generating Process:
        C1 ~ N(0, 1)                # Confounder 1
        C2 ~ N(0, 1)                # Confounder 2
        Segment = categorize(C1)    # Low, Medium, High based on C1
        T ~ Bernoulli(sigmoid(0.3*C1 + 0.2*C2))  # Treatment

        CATE(segment) = base_effect + segment_modifier
        Y = CATE(segment) * T + 0.3*C1 + 0.2*C2 + epsilon

    Args:
        n: Number of samples
        base_effect: Base treatment effect
        segment_effects: Dict mapping segment name to effect modifier
        noise_std: Standard deviation of noise term
        seed: Random seed for reproducibility

    Returns:
        SyntheticDataset with segment-specific ground truth CATEs
    """
    np.random.seed(seed)

    if segment_effects is None:
        segment_effects = {
            "low": -0.10,  # CATE = 0.10
            "medium": 0.00,  # CATE = 0.20
            "high": 0.20,  # CATE = 0.40
        }

    # Confounders
    C1 = np.random.normal(0, 1, n)
    C2 = np.random.normal(0, 1, n)

    # Segment based on C1 (terciles)
    segment = pd.cut(C1, bins=[-np.inf, -0.67, 0.67, np.inf], labels=["low", "medium", "high"])

    # Treatment: Affected by confounders
    prob_T = 1 / (1 + np.exp(-(0.3 * C1 + 0.2 * C2)))
    T = np.random.binomial(1, prob_T)

    # CATE by segment
    cate = np.array([base_effect + segment_effects[str(s)] for s in segment])

    # Outcome with heterogeneous effect
    epsilon = np.random.normal(0, noise_std, n)
    Y = cate * T + 0.3 * C1 + 0.2 * C2 + epsilon

    data = pd.DataFrame(
        {
            "T": T,
            "Y": Y,
            "C1": C1,
            "C2": C2,
            "segment": segment,
        }
    )

    # Compute true CATEs for each segment
    true_cate = {seg: base_effect + mod for seg, mod in segment_effects.items()}

    # ATE is weighted average of CATEs
    segment_counts = data["segment"].value_counts(normalize=True)
    true_ate = sum(true_cate[seg] * segment_counts.get(seg, 0) for seg in true_cate)

    return SyntheticDataset(
        data=data,
        true_ate=true_ate,
        true_cate=true_cate,
        tolerance=0.08,  # Slightly looser for heterogeneous case
        treatment_col="T",
        outcome_col="Y",
        confounder_cols=["C1", "C2"],
        dgp_name="heterogeneous_cate",
        n_samples=n,
        seed=seed,
    )


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture
def simple_linear_dataset() -> SyntheticDataset:
    """Generate simple linear dataset for baseline testing.

    True ATE: 0.50
    Confounding: None
    N: 10,000
    Purpose: Baseline sanity check
    """
    return generate_simple_linear(n=10000, true_ate=0.50, seed=42)


@pytest.fixture
def confounded_moderate_dataset() -> SyntheticDataset:
    """Generate dataset with moderate observed confounding.

    True ATE: 0.30
    Confounding: 1 observed confounder (strength=0.4)
    N: 10,000
    Purpose: Adjustment recovery test
    """
    return generate_confounded_moderate(
        n=10000,
        true_ate=0.30,
        confounder_strength=0.40,
        seed=42,
    )


@pytest.fixture
def heterogeneous_cate_dataset() -> SyntheticDataset:
    """Generate dataset with heterogeneous treatment effects.

    True ATE: ~0.23 (weighted average)
    True CATEs: low=0.10, medium=0.20, high=0.40
    Confounding: 2 observed confounders
    N: 50,000
    Purpose: CATE estimation accuracy
    """
    return generate_heterogeneous_cate(n=50000, seed=42)


@pytest.fixture
def small_simple_dataset() -> SyntheticDataset:
    """Small simple dataset for quick unit tests."""
    return generate_simple_linear(n=1000, true_ate=0.50, seed=42)


@pytest.fixture
def small_confounded_dataset() -> SyntheticDataset:
    """Small confounded dataset for quick unit tests."""
    return generate_confounded_moderate(n=1000, true_ate=0.30, seed=42)


# ============================================================================
# ESTIMATION UTILITIES
# ============================================================================


def estimate_ate_naive(
    data: pd.DataFrame,
    treatment_col: str = "T",
    outcome_col: str = "Y",
) -> float:
    """Estimate ATE using naive difference in means.

    This is biased in the presence of confounding.

    Args:
        data: DataFrame with treatment and outcome
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column

    Returns:
        Naive ATE estimate (difference in means)
    """
    treated = data[data[treatment_col] == 1][outcome_col].mean()
    control = data[data[treatment_col] == 0][outcome_col].mean()
    return treated - control


def estimate_ate_adjusted(
    data: pd.DataFrame,
    treatment_col: str = "T",
    outcome_col: str = "Y",
    confounder_cols: List[str] = None,
) -> float:
    """Estimate ATE using linear regression adjustment.

    Args:
        data: DataFrame with treatment, outcome, and confounders
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        confounder_cols: List of confounder column names

    Returns:
        Adjusted ATE estimate
    """
    from sklearn.linear_model import LinearRegression

    if confounder_cols is None:
        confounder_cols = []

    # Build feature matrix
    feature_cols = [treatment_col] + confounder_cols
    X = data[feature_cols].values
    y = data[outcome_col].values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Treatment effect is coefficient of treatment variable
    return model.coef_[0]


def estimate_cate_by_segment(
    data: pd.DataFrame,
    segment_col: str,
    treatment_col: str = "T",
    outcome_col: str = "Y",
    confounder_cols: List[str] = None,
) -> Dict[str, float]:
    """Estimate CATE for each segment using regression adjustment.

    Args:
        data: DataFrame with treatment, outcome, segment, and confounders
        segment_col: Name of segment column
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        confounder_cols: List of confounder column names

    Returns:
        Dictionary mapping segment names to CATE estimates
    """
    cates = {}

    for segment in data[segment_col].unique():
        segment_data = data[data[segment_col] == segment]
        cates[str(segment)] = estimate_ate_adjusted(
            segment_data,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            confounder_cols=confounder_cols,
        )

    return cates
