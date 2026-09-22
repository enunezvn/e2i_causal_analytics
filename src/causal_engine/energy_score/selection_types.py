"""Selection types of the energy-score estimator selector.

``SelectionStrategy``, ``EstimatorResult``, ``SelectionResult``, ``EstimatorConfig``
and ``EstimatorSelectorConfig``, extracted from ``estimator_selector.py`` (module-
size ratchet) and re-exported there, so every existing import keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from src.causal_engine.estimator_registry import DEFAULT_ESTIMATOR_SPECS, EstimatorType

from .score_calculator import EnergyScoreConfig, EnergyScoreResult

# #1392: row cap for the energy-score TOURNAMENT. Energy-score selection is a
# RANKING, not the final estimate — yet the 4-way tournament fitted every
# estimator on the full frame, which on the live 37,371-row conversion
# substrate cost 116s warm / 144s cold and consumed the entire chat-turn
# compute budget, so the mandatory refutation gate failed closed and every
# chat-path causal turn failed honestly. Above this cap the tournament runs on
# a DETERMINISTIC stratified subsample (treatment × outcome-bin strata, seed
# derived from the frame content); the WINNER is then refit on the FULL frame
# and only that full-frame fit is reported — the refutation node reconstructs
# the estimator from the full estimation_data passthrough and enforces a
# reconstructed-vs-reported ATE tolerance, so the reported ATE/CI must come
# from a full-frame fit. 5,000 aligns with
# ``EnergyScoreConfig.max_samples_for_exact``: beyond that the energy-distance
# term itself falls back to internal random subsampling, so scoring more rows
# adds sampling noise, not ranking signal (it is also the bottom of the
# owner-approved 5-10k range — the largest latency win). Frames at or below
# the cap keep today's full-frame selection unchanged.
SELECTION_MAX_ROWS_DEFAULT = 5_000


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

    # NOT-APPLICABLE (skipped, not failed): the estimator was deliberately not
    # run because it cannot apply to this design — e.g. a covariate-requiring
    # DML / forest / meta-learner on a ZERO-covariate (randomized / empty-backdoor)
    # question, where the correct estimator is the unadjusted contrast (OLS).
    # ``skipped`` distinguishes this from a genuine ``.fit()`` failure so the UI
    # renders "not applicable" instead of a cryptic sklearn traceback.
    skipped: bool = False
    # Served-refit status (codex r5). None: no separate served refit happened
    # (an unsubsampled selection serves the tournament fit itself, or this
    # candidate lost the tournament). True: this candidate's served full-frame
    # refit succeeded. False: it was refused -- ``energy_score_result`` then
    # still carries the TOURNAMENT score (the ranking is immutable metadata).
    served_refit: Optional[bool] = None

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
            "skipped": self.skipped,
            "ate": self.ate,
            "ate_std": self.ate_std,
            "ate_ci_lower": self.ate_ci_lower,
            "ate_ci_upper": self.ate_ci_upper,
            "energy_score": self.energy_score if self.success else None,
            # explicit, not success-gated: the tournament ranking score whenever
            # this candidate was scored, and whether its served refit succeeded
            "tournament_energy_score": self.energy_score
            if np.isfinite(self.energy_score)
            else None,
            "served_refit": self.served_refit,
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

    # M-est3: reliability gate. ``exceeded_max_energy_score`` is True when the
    # selected (best) estimator's energy score is above
    # ``EstimatorSelectorConfig.max_acceptable_energy_score``. ``requires_review``
    # is the consumer-facing signal that the selected ATE is NOT a clean valid
    # result and must be surfaced for review rather than reported as reliable.
    exceeded_max_energy_score: bool = False
    requires_review: bool = False

    # #1188: what the covariates MEAN for this run. "confounding" = a non-empty
    # backdoor was adjusted (observational de-biasing); "efficiency" = a
    # randomized/empty-backdoor design where curated pre-treatment baselines
    # entered as variance-reduction controls (ANCOVA-style precision — the
    # point estimate is unbiased either way); "none" = unadjusted contrast.
    adjustment_type: str = "none"

    # #1392: subsampled-tournament disclosure. When the frame exceeded
    # ``EstimatorSelectorConfig.selection_max_rows`` the tournament RANKED the
    # estimators on a deterministic stratified subsample of
    # ``selection_n_rows`` rows (out of ``selection_n_rows_total``); the
    # reported ``selected`` result is the winner REFIT on the full frame.
    # Downstream honesty surfaces must disclose this — the per-estimator
    # energy scores are ranking artifacts computed on the subsample.
    selection_subsampled: bool = False
    selection_n_rows: int = 0
    selection_n_rows_total: int = 0

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
            "exceeded_max_energy_score": self.exceeded_max_energy_score,
            "requires_review": self.requires_review,
            "adjustment_type": self.adjustment_type,
            "selection_subsampled": self.selection_subsampled,
            "selection_n_rows": self.selection_n_rows,
            "selection_n_rows_total": self.selection_n_rows_total,
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
            EstimatorConfig(spec.estimator_type, priority=int(spec.default_priority or 0))
            for spec in DEFAULT_ESTIMATOR_SPECS
        ]
    )

    # Energy score configuration
    energy_score_config: EnergyScoreConfig = field(default_factory=EnergyScoreConfig)

    # Selection thresholds
    min_energy_score_gap: float = 0.05  # Minimum gap to prefer one over another
    max_acceptable_energy_score: float = 0.8  # Warn if best score is above this

    # #1392: tournament row cap (see SELECTION_MAX_ROWS_DEFAULT for the full
    # rationale). Frames larger than this run the multi-estimator tournament on
    # a deterministic stratified subsample; the winner is refit on the full
    # frame and only that full-frame fit is reported. Frames at or below the
    # cap keep full-frame selection unchanged. Must be >= 1 (validated in
    # ``__post_init__`` — codex iter-2 LOW).
    selection_max_rows: int = SELECTION_MAX_ROWS_DEFAULT

    # Fallback behavior
    fallback_on_all_fail: bool = True
    fallback_estimator: EstimatorType = EstimatorType.OLS

    # Parallelization (future)
    parallel_evaluation: bool = False
    max_workers: int = 4

    def __post_init__(self) -> None:
        """Validate configuration (codex iter-2 LOW, #1392)."""
        if self.selection_max_rows < 1:
            raise ValueError(f"selection_max_rows must be >= 1, got {self.selection_max_rows}")
