"""
E2I Causal Analytics - Uplift Metrics
=====================================

Evaluation metrics for uplift models including:
- AUUC (Area Under Uplift Curve)
- Qini Coefficient and Qini Curve
- Cumulative Gain
- Treatment Effect Calibration

Author: E2I Causal Analytics Team
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class UpliftMetrics:
    """Container for uplift evaluation metrics.

    Attributes:
        auuc: Area Under Uplift Curve
        qini_coefficient: Qini coefficient (normalized)
        qini_auc: Area under Qini curve
        cumulative_gain_auc: Area under cumulative gain curve
        uplift_at_k: Uplift at various percentiles
        calibration_error: Mean calibration error
        treatment_balance: Treatment/control balance ratio
    """

    auuc: float
    qini_coefficient: float
    qini_auc: float
    cumulative_gain_auc: float
    uplift_at_k: Dict[str, float] = field(default_factory=dict)
    calibration_error: Optional[float] = None
    treatment_balance: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "auuc": self.auuc,
            "qini_coefficient": self.qini_coefficient,
            "qini_auc": self.qini_auc,
            "cumulative_gain_auc": self.cumulative_gain_auc,
            "uplift_at_k": self.uplift_at_k,
            "calibration_error": self.calibration_error,
            "treatment_balance": self.treatment_balance,
            "metadata": self.metadata,
        }


def calculate_uplift_curve(
    uplift_scores: NDArray[np.float64],
    treatment: NDArray[np.int_],
    outcome: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate uplift curve coordinates.

    The uplift curve shows the incremental gain from treatment as a
    function of the proportion of population targeted.

    Args:
        uplift_scores: Predicted uplift scores (higher = more likely to respond)
        treatment: Binary treatment indicator (0=control, 1=treated)
        outcome: Binary or continuous outcome

    Returns:
        Tuple of (x_values, uplift_values) for plotting
    """
    # Ensure arrays
    uplift_scores = np.asarray(uplift_scores).flatten()
    treatment = np.asarray(treatment).flatten()
    outcome = np.asarray(outcome).flatten()

    n = len(uplift_scores)
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    # Sort by uplift score descending
    sorted_idx = np.argsort(-uplift_scores)
    treatment_sorted = treatment[sorted_idx]
    outcome_sorted = outcome[sorted_idx]

    # Calculate cumulative metrics
    n_treated_cum = np.cumsum(treatment_sorted)
    n_control_cum = np.cumsum(1 - treatment_sorted)
    outcome_treated_cum = np.cumsum(treatment_sorted * outcome_sorted)
    outcome_control_cum = np.cumsum((1 - treatment_sorted) * outcome_sorted)

    # Avoid division by zero
    n_treated_cum = np.maximum(n_treated_cum, 1)
    n_control_cum = np.maximum(n_control_cum, 1)

    # Calculate uplift at each point
    response_rate_treated = outcome_treated_cum / n_treated_cum
    response_rate_control = outcome_control_cum / n_control_cum

    uplift_values = response_rate_treated - response_rate_control

    # X values are proportion of population
    x_values = np.arange(1, n + 1) / n

    # Prepend origin
    x_values = np.concatenate([[0], x_values])
    uplift_values = np.concatenate([[0], uplift_values])

    return x_values, uplift_values


def calculate_qini_curve(
    uplift_scores: NDArray[np.float64],
    treatment: NDArray[np.int_],
    outcome: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate Qini curve coordinates.

    The Qini curve shows the cumulative number of incremental positive
    outcomes attributable to treatment.

    Args:
        uplift_scores: Predicted uplift scores
        treatment: Binary treatment indicator
        outcome: Binary or continuous outcome

    Returns:
        Tuple of (x_values, qini_values) for plotting
    """
    # Ensure arrays
    uplift_scores = np.asarray(uplift_scores).flatten()
    treatment = np.asarray(treatment).flatten()
    outcome = np.asarray(outcome).flatten()

    n = len(uplift_scores)
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    # Sort by uplift score descending
    sorted_idx = np.argsort(-uplift_scores)
    treatment_sorted = treatment[sorted_idx]
    outcome_sorted = outcome[sorted_idx]

    # Calculate cumulative counts
    n_treated_cum = np.cumsum(treatment_sorted)
    n_control_cum = np.cumsum(1 - treatment_sorted)
    outcome_treated_cum = np.cumsum(treatment_sorted * outcome_sorted)
    outcome_control_cum = np.cumsum((1 - treatment_sorted) * outcome_sorted)

    # Total counts
    n_treated_total = np.sum(treatment)
    n_control_total = n - n_treated_total

    # Qini values: incremental outcomes normalized by treatment/control ratio
    if n_control_total > 0:
        qini_values = outcome_treated_cum - outcome_control_cum * (
            n_treated_total / n_control_total
        )
    else:
        qini_values = outcome_treated_cum

    # X values are proportion of population
    x_values = np.arange(1, n + 1) / n

    # Prepend origin
    x_values = np.concatenate([[0], x_values])
    qini_values = np.concatenate([[0], qini_values])

    return x_values, qini_values


def calculate_cumulative_gain(
    uplift_scores: NDArray[np.float64],
    treatment: NDArray[np.int_],
    outcome: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate cumulative gain curve.

    Cumulative gain shows the total uplift achieved by targeting
    the top k% of the population.

    Args:
        uplift_scores: Predicted uplift scores
        treatment: Binary treatment indicator
        outcome: Binary or continuous outcome

    Returns:
        Tuple of (x_values, gain_values) for plotting
    """
    # Ensure arrays
    uplift_scores = np.asarray(uplift_scores).flatten()
    treatment = np.asarray(treatment).flatten()
    outcome = np.asarray(outcome).flatten()

    n = len(uplift_scores)
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    # Sort by uplift score descending
    sorted_idx = np.argsort(-uplift_scores)
    treatment_sorted = treatment[sorted_idx]
    outcome_sorted = outcome[sorted_idx]

    # Calculate response rates in treatment and control
    n_treated_cum = np.cumsum(treatment_sorted)
    n_control_cum = np.cumsum(1 - treatment_sorted)
    outcome_treated_cum = np.cumsum(treatment_sorted * outcome_sorted)
    outcome_control_cum = np.cumsum((1 - treatment_sorted) * outcome_sorted)

    # Avoid division by zero
    n_treated_cum_safe = np.maximum(n_treated_cum, 1)
    n_control_cum_safe = np.maximum(n_control_cum, 1)

    # Cumulative gain is the difference in cumulative response rates
    # multiplied by the number of people targeted
    response_rate_treated = outcome_treated_cum / n_treated_cum_safe
    response_rate_control = outcome_control_cum / n_control_cum_safe

    # Gain at each point
    k = np.arange(1, n + 1)
    gain_values = (response_rate_treated - response_rate_control) * k

    # X values are proportion of population
    x_values = k / n

    # Prepend origin
    x_values = np.concatenate([[0], x_values])
    gain_values = np.concatenate([[0], gain_values])

    return x_values, gain_values


def auuc(
    uplift_scores: NDArray[np.float64],
    treatment: NDArray[np.int_],
    outcome: NDArray[np.float64],
    normalize: bool = True,
) -> float:
    """Calculate Area Under Uplift Curve (AUUC).

    AUUC measures overall model performance by computing the area under
    the uplift curve. Higher values indicate better targeting ability.

    Args:
        uplift_scores: Predicted uplift scores
        treatment: Binary treatment indicator
        outcome: Binary or continuous outcome
        normalize: Whether to normalize by random baseline

    Returns:
        AUUC score (0-1 if normalized, unbounded otherwise)
    """
    x_values, uplift_values = calculate_uplift_curve(
        uplift_scores, treatment, outcome
    )

    # Calculate area using trapezoidal rule
    area = np.trapz(uplift_values, x_values)

    if normalize:
        # Normalize by perfect model area
        # Perfect model would have constant uplift
        treatment = np.asarray(treatment).flatten()
        outcome = np.asarray(outcome).flatten()

        n_treated = np.sum(treatment)
        n_control = len(treatment) - n_treated

        if n_treated > 0 and n_control > 0:
            overall_uplift = (
                np.sum(treatment * outcome) / n_treated
                - np.sum((1 - treatment) * outcome) / n_control
            )
            random_area = overall_uplift * 0.5  # Random model area
            if abs(random_area) > 1e-10:
                area = area / abs(random_area)

    return float(area)


def qini_coefficient(
    uplift_scores: NDArray[np.float64],
    treatment: NDArray[np.int_],
    outcome: NDArray[np.float64],
) -> float:
    """Calculate Qini coefficient.

    The Qini coefficient is the normalized area between the Qini curve
    and the random selection diagonal. Higher values indicate better
    model performance.

    Args:
        uplift_scores: Predicted uplift scores
        treatment: Binary treatment indicator
        outcome: Binary or continuous outcome

    Returns:
        Qini coefficient (typically -1 to 1)
    """
    x_values, qini_values = calculate_qini_curve(
        uplift_scores, treatment, outcome
    )

    # Area under Qini curve
    qini_area = np.trapz(qini_values, x_values)

    # Random model area (diagonal)
    random_area = qini_values[-1] * 0.5

    # Qini coefficient: (actual - random) / (perfect - random)
    # Perfect model area is qini_values[-1]
    perfect_area = qini_values[-1]

    if abs(perfect_area - random_area) > 1e-10:
        coefficient = (qini_area - random_area) / (perfect_area - random_area)
    else:
        coefficient = 0.0

    return float(coefficient)


def qini_auc(
    uplift_scores: NDArray[np.float64],
    treatment: NDArray[np.int_],
    outcome: NDArray[np.float64],
) -> float:
    """Calculate Area Under Qini Curve.

    Args:
        uplift_scores: Predicted uplift scores
        treatment: Binary treatment indicator
        outcome: Binary or continuous outcome

    Returns:
        Area under Qini curve
    """
    x_values, qini_values = calculate_qini_curve(
        uplift_scores, treatment, outcome
    )
    return float(np.trapz(qini_values, x_values))


def cumulative_gain_auc(
    uplift_scores: NDArray[np.float64],
    treatment: NDArray[np.int_],
    outcome: NDArray[np.float64],
) -> float:
    """Calculate Area Under Cumulative Gain Curve.

    Args:
        uplift_scores: Predicted uplift scores
        treatment: Binary treatment indicator
        outcome: Binary or continuous outcome

    Returns:
        Area under cumulative gain curve
    """
    x_values, gain_values = calculate_cumulative_gain(
        uplift_scores, treatment, outcome
    )
    return float(np.trapz(gain_values, x_values))


def uplift_at_k(
    uplift_scores: NDArray[np.float64],
    treatment: NDArray[np.int_],
    outcome: NDArray[np.float64],
    k_percentiles: List[float] = [10, 20, 30, 40, 50],
) -> Dict[str, float]:
    """Calculate uplift at various percentiles.

    Returns the observed uplift when targeting the top k% of
    the population (sorted by predicted uplift).

    Args:
        uplift_scores: Predicted uplift scores
        treatment: Binary treatment indicator
        outcome: Binary or continuous outcome
        k_percentiles: List of percentiles to evaluate

    Returns:
        Dictionary mapping percentile to observed uplift
    """
    # Ensure arrays
    uplift_scores = np.asarray(uplift_scores).flatten()
    treatment = np.asarray(treatment).flatten()
    outcome = np.asarray(outcome).flatten()

    n = len(uplift_scores)
    if n == 0:
        return {f"uplift_at_{k}": 0.0 for k in k_percentiles}

    # Sort by uplift score descending
    sorted_idx = np.argsort(-uplift_scores)
    treatment_sorted = treatment[sorted_idx]
    outcome_sorted = outcome[sorted_idx]

    results = {}
    for k in k_percentiles:
        # Number of individuals at top k%
        n_k = max(1, int(n * k / 100))

        # Get top k% subset
        treatment_k = treatment_sorted[:n_k]
        outcome_k = outcome_sorted[:n_k]

        # Calculate uplift in subset
        n_treated = np.sum(treatment_k)
        n_control = n_k - n_treated

        if n_treated > 0 and n_control > 0:
            uplift_k = (
                np.sum(treatment_k * outcome_k) / n_treated
                - np.sum((1 - treatment_k) * outcome_k) / n_control
            )
        elif n_treated > 0:
            uplift_k = np.sum(treatment_k * outcome_k) / n_treated
        else:
            uplift_k = 0.0

        results[f"uplift_at_{k}"] = float(uplift_k)

    return results


def treatment_effect_calibration(
    predicted_uplift: NDArray[np.float64],
    treatment: NDArray[np.int_],
    outcome: NDArray[np.float64],
    n_bins: int = 10,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Calculate treatment effect calibration.

    Compares predicted uplift to observed uplift in bins,
    similar to reliability diagrams for probability calibration.

    Args:
        predicted_uplift: Predicted uplift scores
        treatment: Binary treatment indicator
        outcome: Binary or continuous outcome
        n_bins: Number of bins for calibration

    Returns:
        Tuple of (predicted_means, observed_means, calibration_error)
    """
    # Ensure arrays
    predicted_uplift = np.asarray(predicted_uplift).flatten()
    treatment = np.asarray(treatment).flatten()
    outcome = np.asarray(outcome).flatten()

    n = len(predicted_uplift)
    if n == 0:
        return np.array([]), np.array([]), 0.0

    # Bin by predicted uplift
    bin_edges = np.percentile(predicted_uplift, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-10  # Ensure max value is included

    predicted_means = []
    observed_means = []
    bin_sizes = []

    for i in range(n_bins):
        mask = (predicted_uplift >= bin_edges[i]) & (predicted_uplift < bin_edges[i + 1])
        if np.sum(mask) < 2:
            continue

        # Predicted mean in bin
        pred_mean = np.mean(predicted_uplift[mask])

        # Observed uplift in bin
        treatment_bin = treatment[mask]
        outcome_bin = outcome[mask]

        n_treated = np.sum(treatment_bin)
        n_control = len(treatment_bin) - n_treated

        if n_treated > 0 and n_control > 0:
            obs_uplift = (
                np.sum(treatment_bin * outcome_bin) / n_treated
                - np.sum((1 - treatment_bin) * outcome_bin) / n_control
            )
        else:
            obs_uplift = pred_mean  # No calibration info

        predicted_means.append(pred_mean)
        observed_means.append(obs_uplift)
        bin_sizes.append(np.sum(mask))

    predicted_means = np.array(predicted_means)
    observed_means = np.array(observed_means)
    bin_sizes = np.array(bin_sizes)

    # Weighted mean calibration error
    if len(bin_sizes) > 0 and np.sum(bin_sizes) > 0:
        calibration_error = np.sum(
            bin_sizes * np.abs(predicted_means - observed_means)
        ) / np.sum(bin_sizes)
    else:
        calibration_error = 0.0

    return predicted_means, observed_means, float(calibration_error)


def evaluate_uplift_model(
    uplift_scores: NDArray[np.float64],
    treatment: NDArray[np.int_],
    outcome: NDArray[np.float64],
    k_percentiles: List[float] = [10, 20, 30, 40, 50],
) -> UpliftMetrics:
    """Comprehensive evaluation of uplift model.

    Calculates all standard uplift metrics in a single function.

    Args:
        uplift_scores: Predicted uplift scores
        treatment: Binary treatment indicator
        outcome: Binary or continuous outcome
        k_percentiles: Percentiles for uplift@k calculation

    Returns:
        UpliftMetrics containing all evaluation metrics
    """
    # Handle multi-dimensional scores
    if len(uplift_scores.shape) > 1:
        scores = uplift_scores[:, 0]
    else:
        scores = uplift_scores

    treatment = np.asarray(treatment).flatten()
    outcome = np.asarray(outcome).flatten()

    # Calculate all metrics
    auuc_score = auuc(scores, treatment, outcome, normalize=True)
    qini_coef = qini_coefficient(scores, treatment, outcome)
    qini_auc_score = qini_auc(scores, treatment, outcome)
    cg_auc = cumulative_gain_auc(scores, treatment, outcome)
    uplift_k = uplift_at_k(scores, treatment, outcome, k_percentiles)

    # Calibration
    _, _, cal_error = treatment_effect_calibration(scores, treatment, outcome)

    # Treatment balance
    n_treated = np.sum(treatment)
    n_control = len(treatment) - n_treated
    if n_control > 0:
        balance = n_treated / n_control
    else:
        balance = None

    return UpliftMetrics(
        auuc=auuc_score,
        qini_coefficient=qini_coef,
        qini_auc=qini_auc_score,
        cumulative_gain_auc=cg_auc,
        uplift_at_k=uplift_k,
        calibration_error=cal_error,
        treatment_balance=balance,
        metadata={
            "n_samples": len(scores),
            "n_treated": int(n_treated),
            "n_control": int(n_control),
            "k_percentiles": k_percentiles,
        },
    )
