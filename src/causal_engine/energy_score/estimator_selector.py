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

import hashlib
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


# Relative estimation+refutation cost rank for each estimator (lower = faster).
# Used ONLY to break ties when two estimators have statistically
# indistinguishable energy scores (#622). The dominant downstream cost is the
# refutation suite, which re-fits the SELECTED estimator dozens of times: the
# refutation node reconstructs a DoWhy model using the same estimator that
# produced the reported ATE, and MEASURED that costs ~0.05s/re-estimation for
# the linear OLS refit vs ~3.1s for CausalForestDML (DoWhy 0.14 / EconML 0.16).
# So when energy scores tie, picking the linear estimator turns a ~35-60 min
# refutation suite into a ~30s one with no measured loss of estimate quality
# (the energy scores were equal by definition). Ranks are coarse buckets:
# closed-form / single-fit linear models are cheapest; meta-learners that fit a
# handful of boosted models are mid; ensemble/forest DML estimators that fit
# many trees with cross-fitting are the most expensive.
_ESTIMATOR_SPEED_RANK: dict[EstimatorType, int] = {
    EstimatorType.OLS: 0,
    EstimatorType.S_LEARNER: 1,
    EstimatorType.T_LEARNER: 1,
    EstimatorType.X_LEARNER: 2,
    EstimatorType.LINEAR_DML: 2,
    EstimatorType.DML_LEARNER: 2,
    EstimatorType.DRLEARNER: 3,
    EstimatorType.ORTHO_FOREST: 4,
    EstimatorType.CAUSAL_FOREST: 4,
}


# Estimators that do NOT orthogonalize / cross-fit, and so are more exposed to
# confounding bias. The energy score measures goodness-of-fit on the outcome,
# NOT causal validity — so a naive OLS can fit the outcome marginal as well as a
# DML/forest estimator (an energy-score "tie") yet remain biased under
# confounding and FAIL the downstream refutation gate. MEASURED on the
# patient_journeys gold standard: at energy scores within ~0.0005 of each other,
# OLS gated BLOCK (ate=0.089, CI 0.020-0.158) while the DML/forest family gated
# PROCEED (causal_forest ate=0.120, CI 0.117-0.123). So when energy scores tie,
# a confounding-robust estimator must be preferred over these — otherwise "Auto"
# silently selects the fast-but-biased estimator. (#622 had broken ties by raw
# speed, which always picked OLS; this restricts the speed tiebreak to the
# confounding-robust subset.)
_CONFOUNDING_BLIND_ESTIMATORS: frozenset[EstimatorType] = frozenset({EstimatorType.OLS})


# Estimators that can produce an estimate from a ZERO-covariate design matrix (an
# empty backdoor — the correct adjustment set for a randomized / exogenous
# treatment, e.g. the nba_triggers RCT). Only OLS has an empty-backdoor path (its
# treatment coefficient is the unadjusted difference-in-means); every DML / forest
# / meta-learner orthogonalises against covariates and its nuisance/propensity fit
# raises sklearn's "Found array with 0 feature(s)" on an empty X. So on an empty
# backdoor the covariate-requiring estimators are NOT applicable (not "failed"):
# they are skipped with an honest reason instead of surfacing a raw traceback.
_EMPTY_BACKDOOR_CAPABLE: frozenset[EstimatorType] = frozenset({EstimatorType.OLS})

_EMPTY_BACKDOOR_SKIP_REASON = (
    "not applicable: no covariates to adjust for (randomized / empty-backdoor "
    "design). Covariate-based estimators (DML, causal forest, meta-learners) have "
    "nothing to orthogonalize against here; the unadjusted contrast (OLS) is the "
    "correct estimator."
)


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

# Outcome stratification (#1392): a low-cardinality outcome (e.g. the live
# substrate's rare-binary ``converted``) is stratified on its distinct values
# so per-arm outcome prevalence is preserved; a continuous outcome falls back
# to quartile bins.
_OUTCOME_STRATA_MAX_DISTINCT = 10
_OUTCOME_QUANTILE_EDGES = (0.25, 0.5, 0.75)


def _stratified_subsample_indices(
    treatment: NDArray[Any],
    outcome: NDArray[Any],
    max_rows: int,
) -> NDArray[np.intp]:
    """Deterministic stratified subsample of ``max_rows`` row indices (#1392).

    Strata are the cross of treatment value × outcome bin (distinct outcome
    values when ≤ ``_OUTCOME_STRATA_MAX_DISTINCT`` uniques, else quartile
    bins; non-finite outcomes get their own bin), with proportional
    largest-remainder allocation — so arm shares AND per-arm outcome
    prevalence are preserved within ±1 row per stratum.

    Determinism: the RNG seed is derived from the CONTENT of the treatment and
    outcome arrays (sha256) plus ``max_rows`` — never wall-clock or global RNG
    state — so the same frame/spec always draws the same indices and chat
    turns stay reproducible.
    """
    if max_rows < 1:
        # codex iter-2 LOW (#1392): fail LOUD on a nonsensical cap instead of
        # crashing later with a cryptic np.concatenate ValueError.
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")
    t = np.asarray(treatment)
    y = np.asarray(outcome, dtype=np.float64)
    n = int(len(t))
    if n <= max_rows:
        return np.arange(n, dtype=np.intp)

    # --- outcome bins ---
    finite = np.isfinite(y)
    finite_vals = np.unique(y[finite])
    if finite_vals.size <= _OUTCOME_STRATA_MAX_DISTINCT:
        y_bin = np.full(n, finite_vals.size, dtype=np.int64)  # non-finite: own bin
        y_bin[finite] = np.searchsorted(finite_vals, y[finite])
    else:
        edges = np.quantile(y[finite], _OUTCOME_QUANTILE_EDGES)
        y_bin = np.full(n, len(edges) + 1, dtype=np.int64)  # non-finite: own bin
        y_bin[finite] = np.digitize(y[finite], edges)

    # --- treatment × outcome-bin strata ---
    _, t_codes = np.unique(t, return_inverse=True)
    combo = t_codes.astype(np.int64) * (int(y_bin.max()) + 1) + y_bin
    strata, strata_inv, strata_counts = np.unique(combo, return_inverse=True, return_counts=True)

    # --- content-derived deterministic seed ---
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(t, dtype=np.float64).tobytes())
    hasher.update(np.ascontiguousarray(y).tobytes())
    hasher.update(str(int(max_rows)).encode("ascii"))
    rng = np.random.default_rng(int.from_bytes(hasher.digest()[:8], "little"))

    # --- proportional allocation, largest remainder, ≥1 per non-empty stratum ---
    raw = strata_counts * (max_rows / n)
    take = np.floor(raw).astype(np.int64)
    if strata.size <= max_rows:
        take = np.maximum(take, 1)
    take = np.minimum(take, strata_counts)
    remainders = raw - np.floor(raw)
    deficit = max_rows - int(take.sum())
    if deficit > 0:
        # Fill remaining slots by largest fractional remainder, cycling while
        # capacity remains (total capacity is n > max_rows, so this terminates).
        order = np.argsort(-remainders, kind="stable")
        while deficit > 0:
            progressed = False
            for i in order:
                if deficit == 0:
                    break
                if take[i] < strata_counts[i]:
                    take[i] += 1
                    deficit -= 1
                    progressed = True
            if not progressed:  # pragma: no cover — capacity math prevents this
                break
    elif deficit < 0:
        # The ≥1 bumps overshot: trim from the smallest remainders, never below 1.
        order = np.argsort(remainders, kind="stable")
        while deficit < 0:
            progressed = False
            for i in order:
                if deficit == 0:
                    break
                if take[i] > 1:
                    take[i] -= 1
                    deficit += 1
                    progressed = True
            if not progressed:  # pragma: no cover — strata.size ≤ max_rows guard
                break

    # --- deterministic per-stratum draw (strata are sorted by np.unique) ---
    chosen: list[NDArray[np.intp]] = []
    for s_i in range(strata.size):
        members = np.flatnonzero(strata_inv == s_i)
        k = int(take[s_i])
        if k <= 0:
            continue
        if k >= members.size:
            chosen.append(members)
        else:
            chosen.append(rng.choice(members, size=k, replace=False))
    return np.sort(np.concatenate(chosen)).astype(np.intp)


def _honest_ate_ci(
    model: Any, X: Optional[NDArray[np.float64]]
) -> Optional[tuple[float, float, float]]:
    """Population ATE SAMPLING interval from econml's ``ate_inference``.

    Returns ``(ci_lower, ci_upper, stderr_mean)`` or None when the estimator
    exposes no inference (the caller must then surface an honest absent CI,
    NEVER a fabricated one).

    Why this exists (#1188): the previous ``effect_inference(X).conf_int_mean()``
    call raises AttributeError on econml 0.16 (``effect_inference`` returns
    ``NormalInferenceResults``; ``conf_int_mean`` lives on
    ``PopulationSummaryResults``), so every wrapper silently fell into an
    ``ate ± 1.96·std(cate)/sqrt(n)`` fallback. That quantity is the spread of
    the HETEROGENEOUS effect distribution divided by sqrt(n) — not a sampling
    interval for the ATE — and measured ~50x too narrow (0.0008 vs an MC-truth
    ~0.043 on a planted DGP). ``ate_inference(X).conf_int_mean()`` is the
    calibrated population-ATE interval (validated against a seed-varied Monte
    Carlo; CausalForestDML's is a conservative upper bound by construction).
    """
    try:
        pop = model.ate_inference(X) if X is not None else model.ate_inference()
        lo, hi = pop.conf_int_mean()
        stderr = float(np.squeeze(pop.stderr_mean))
        return float(np.squeeze(lo)), float(np.squeeze(hi)), stderr
    except Exception as e:  # noqa: BLE001 — absence of inference is a valid state
        logger.warning(f"ATE inference unavailable ({type(e).__name__}: {e}); CI omitted.")
        return None


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
                    n_estimators=50,
                    min_samples_leaf=5,
                    min_impurity_decrease=1e-7,
                    random_state=rs,
                ),
                "model_t": (
                    RandomForestClassifier(
                        n_estimators=50,
                        min_samples_leaf=5,
                        min_impurity_decrease=1e-7,
                        random_state=rs,
                    )
                    if is_binary
                    else RandomForestRegressor(
                        n_estimators=50,
                        min_samples_leaf=5,
                        min_impurity_decrease=1e-7,
                        random_state=rs,
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

            # Population ATE SAMPLING interval (honest; #1188). For the forest
            # this is a conservative upper bound (RMS of pointwise stderrs) —
            # wide is honest, the old fake-narrow fallback was not.
            inference = _honest_ate_ci(model, X)
            if inference is not None:
                ate_ci_lower, ate_ci_upper, ate_std = inference
            else:
                ate_ci_lower = ate_ci_upper = ate_std = None  # type: ignore[assignment]

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
                    n_estimators=50,
                    min_samples_leaf=5,
                    min_impurity_decrease=1e-7,
                    random_state=42,
                ),
                model_t=RandomForestClassifier(
                    n_estimators=50,
                    min_samples_leaf=5,
                    min_impurity_decrease=1e-7,
                    random_state=42,
                ),
                discrete_treatment=True,
                random_state=42,
            )
            X = covariates.values
            model.fit(outcome, treatment, X=X, W=X)

            # Get estimates
            cate = model.effect(X)
            ate = float(np.mean(cate))

            # Population ATE SAMPLING interval (honest; #1188).
            inference = _honest_ate_ci(model, X)
            if inference is not None:
                ate_ci_lower, ate_ci_upper, ate_std = inference
            else:
                ate_ci_lower = ate_ci_upper = ate_std = None  # type: ignore[assignment]

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
            from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

            # model_final is the LINEAR statsmodels regression (not GBR): it is
            # the only final stage exposing prediction stderr, i.e. the only way
            # DRLearner yields an honest population-ATE sampling interval
            # (#1188). Nuisances (outcome regression + propensity) stay
            # gradient-boosted; only the CATE(X) surface becomes linear-in-X.
            model = DRLearner(
                model_regression=GradientBoostingRegressor(n_estimators=50, random_state=42),
                model_propensity=GradientBoostingClassifier(n_estimators=50, random_state=42),
                model_final=StatsModelsLinearRegression(),
                random_state=42,
            )
            X = covariates.values
            model.fit(outcome, treatment, X=X, W=X)

            cate = model.effect(X)
            ate = float(np.mean(cate))

            # Population ATE SAMPLING interval (honest; #1188). The previous
            # ate ± 1.96·std(cate)/sqrt(n) was a heterogeneity spread, not a CI.
            inference = _honest_ate_ci(model, X)
            if inference is not None:
                ate_ci_lower, ate_ci_upper, ate_std = inference
            else:
                ate_ci_lower = ate_ci_upper = ate_std = None  # type: ignore[assignment]

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
            # An EMPTY backdoor (0-feature X) is the CORRECT adjustment set for a
            # randomized / exogenous treatment. ``column_stack`` then reduces to
            # the treatment column alone, so OLS yields the UNADJUSTED ATE (the
            # treatment coefficient == difference-in-means for a binary T).
            empty_backdoor = X.shape[1] == 0
            if empty_backdoor:
                # An unadjusted contrast needs EXACTLY TWO arms (both present). Use
                # the DISTINCT values (not count_nonzero, which would mis-handle a
                # non-0/1 encoding such as 1/2 or -1/1): a one-arm or multi-level
                # treatment has no well-defined difference-in-means, so fail-closed
                # rather than report a fake number. Normalize the two arms to 0/1 so
                # the OLS coefficient IS the difference-in-means (the same category
                # contrast the discrete-treatment CATE estimators report), regardless
                # of the raw encoding.
                arms = np.unique(treatment)
                if arms.size != 2:
                    raise ValueError(
                        "Unadjusted empty-backdoor estimation requires exactly two "
                        f"treatment arms; got {arms.size} distinct value(s): "
                        f"{arms.tolist()}."
                    )
                treatment = (treatment == arms[1]).astype(int)
            X_with_treatment = np.column_stack([treatment, X])

            model = LinearRegression()
            model.fit(X_with_treatment, outcome)

            ate = float(model.coef_[0])  # Treatment coefficient

            if empty_backdoor:
                # #1188 (codex iter-1): the unadjusted anchor uses the ANALYTIC
                # Welch (per-arm variance) standard error — deterministic and
                # apples-to-apples with the covariate estimators' analytic
                # ``ate_inference`` intervals, unlike the jittery 100-draw
                # bootstrap it replaces.
                y1 = outcome[treatment == 1]
                y0 = outcome[treatment == 0]
                ate_std = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
            else:
                # Bootstrap standard error — SEEDED so two identical fits give
                # identical CIs (the anchor comparison must not jitter).
                rng = np.random.default_rng(42)
                n_boot = 100
                boot_ates = []
                for _ in range(n_boot):
                    idx = rng.choice(len(treatment), len(treatment), replace=True)
                    m = LinearRegression()
                    m.fit(X_with_treatment[idx], outcome[idx])
                    boot_ates.append(m.coef_[0])
                ate_std = float(np.std(boot_ates))
            ate_ci_lower = ate - 1.96 * ate_std
            ate_ci_upper = ate + 1.96 * ate_std

            # Constant CATE (OLS gives ATE only)
            cate = np.full(len(treatment), ate)

            # Propensity scores. An EMPTY backdoor (0-feature X) is the CORRECT
            # adjustment set for a randomized / exogenous treatment: there is
            # nothing to model, so P(T|X) collapses to the constant marginal
            # P(T)=mean(T). Fitting LogisticRegressionCV on a 0-feature X raises
            # ("0 feature(s)"), so use the constant propensity instead — this is
            # what lets the unadjusted OLS estimate be PRODUCED rather than
            # fail-closing the whole (RCT / exogenous-treatment) question. With
            # >=1 covariate we fit the propensity model as before.
            if empty_backdoor:
                propensity_scores = np.full(len(treatment), float(np.mean(treatment)))
            else:
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
                self.estimators.append(wrapper_class(est_config))  # type: ignore[call-arg]

    def select(
        self,
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        *,
        efficiency_controls: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> SelectionResult:
        """
        Evaluate all estimators and select the best one.

        Args:
            treatment: Binary treatment indicator
            outcome: Observed outcomes
            covariates: DE-CONFOUNDING covariate DataFrame (the backdoor
                adjustment set; zero-width on a randomized design)
            efficiency_controls: #1188 — curated PRE-TREATMENT baselines for a
                randomized (empty-backdoor) design. Ignored unless ``covariates``
                is empty. When present, the covariate estimators fit on these
                controls FOR PRECISION (ANCOVA-style variance reduction —
                randomization already guarantees unbiasedness) while OLS keeps
                the zero-width frame and anchors the comparison at the raw
                unadjusted difference-in-means.
            **kwargs: Additional arguments passed to estimators

        Returns:
            SelectionResult with selected estimator and comparison data
        """
        import time

        total_start = time.perf_counter()

        results: list[EstimatorResult] = []

        # Empty backdoor (zero covariates) = the correct adjustment set for a
        # randomized / exogenous treatment. Covariate-requiring estimators cannot
        # fit a 0-feature design matrix (EconML/sklearn raise "Found array with 0
        # feature(s)"), so skip them with an honest not-applicable reason rather
        # than surfacing that raw traceback per-estimator in the UI comparison.
        empty_backdoor = covariates.shape[1] == 0
        efficiency_mode = (
            empty_backdoor and efficiency_controls is not None and efficiency_controls.shape[1] > 0
        )
        if efficiency_mode:
            adjustment_type = "efficiency"
        elif not empty_backdoor:
            adjustment_type = "confounding"
        else:
            adjustment_type = "none"

        # #1392: subsampled tournament on large frames. Only a genuine
        # MULTI-estimator ranking benefits — the tournament ranks, the winner
        # is refit on the full frame, and only the full-frame fit is reported.
        # No subsampling when:
        #   * the frame is at or below the cap (today's behavior unchanged);
        #   * strategy is FIRST_SUCCESS (the first fit IS the final estimate);
        #   * fewer than 2 estimators will actually fit (the legacy
        #     explicit-method single-estimator path, or a plain empty backdoor
        #     where only OLS applies) — subsample-then-refit of a single
        #     estimator is strictly wasted work.
        n_rows_total = int(len(treatment))
        n_fitting = sum(
            1
            for w in self.estimators
            if not (
                empty_backdoor
                and not efficiency_mode
                and w.estimator_type not in _EMPTY_BACKDOOR_CAPABLE
            )
        )
        max_rows = int(self.config.selection_max_rows)
        subsampled = (
            self.config.strategy != SelectionStrategy.FIRST_SUCCESS
            and n_fitting >= 2
            and n_rows_total > max_rows
        )
        if subsampled:
            sub_idx = _stratified_subsample_indices(treatment, outcome, max_rows)
            sel_treatment = np.asarray(treatment)[sub_idx]
            sel_outcome = np.asarray(outcome)[sub_idx]
            sel_covariates = covariates.iloc[sub_idx].reset_index(drop=True)
            sel_efficiency = (
                efficiency_controls.iloc[sub_idx].reset_index(drop=True)
                if efficiency_mode and efficiency_controls is not None
                else efficiency_controls
            )
            selection_n_rows = int(len(sub_idx))
            logger.info(
                "Energy-score tournament on stratified subsample: %d/%d rows "
                "(cap=%d); winner will be refit on the full frame.",
                selection_n_rows,
                n_rows_total,
                max_rows,
            )
        else:
            sel_treatment, sel_outcome = treatment, outcome
            sel_covariates, sel_efficiency = covariates, efficiency_controls
            selection_n_rows = n_rows_total

        # #1392 (codex iter-1 MED): map each tournament result to the wrapper
        # INSTANCE that produced it, so the winner refit targets that exact
        # instance (not the first chain entry sharing its estimator_type).
        result_wrappers: dict[int, BaseEstimatorWrapper] = {}

        # Evaluate each estimator
        for wrapper in self.estimators:
            if (
                empty_backdoor
                and not efficiency_mode
                and wrapper.estimator_type not in _EMPTY_BACKDOOR_CAPABLE
            ):
                logger.info(
                    "Skipping %s: empty backdoor (0 covariates) — not applicable.",
                    wrapper.estimator_type.value,
                )
                results.append(
                    EstimatorResult(
                        estimator_type=wrapper.estimator_type,
                        success=False,
                        skipped=True,
                        error_message=_EMPTY_BACKDOOR_SKIP_REASON,
                        error_type="NotApplicable",
                    )
                )
                continue

            logger.info(f"Evaluating {wrapper.estimator_type.value}...")

            # Efficiency mode: covariate estimators absorb the baselines as
            # X=W controls (their mean CATE is the standardized MARGINAL ATE);
            # the empty-backdoor-capable anchor (OLS) keeps the zero-width
            # frame so its estimate stays the raw unadjusted contrast.
            # #1392: on a subsampled tournament these are the SUBSAMPLE views;
            # the winner is refit on the full frame after selection.
            fit_frame = sel_covariates
            if efficiency_mode and wrapper.estimator_type not in _EMPTY_BACKDOOR_CAPABLE:
                fit_frame = sel_efficiency  # type: ignore[assignment]

            result = wrapper.fit(sel_treatment, sel_outcome, fit_frame, **kwargs)
            # #1392 (codex iter-1 MED): the tournament ranks wrapper INSTANCES.
            # Record which instance produced each result so the full-frame
            # refit fits the EXACT winner — a first-match-by-type lookup would
            # pick the wrong instance when a chain holds duplicate estimator
            # types with different params.
            result_wrappers[id(result)] = wrapper

            # Compute energy score for successful estimations
            if result.success and result.cate is not None:
                # Fix 5B: Recursion guard for energy score computation
                import sys

                old_limit = sys.getrecursionlimit()
                sys.setrecursionlimit(max(old_limit, 5000))
                try:
                    energy_result = self.energy_calculator.compute(
                        treatment=sel_treatment,
                        outcome=sel_outcome,
                        # The frame the estimator actually conditioned on
                        # (efficiency mode hands baselines to the covariate
                        # estimators while OLS keeps the zero-width anchor).
                        covariates=fit_frame,
                        estimated_effects=result.cate,
                        propensity_scores=result.propensity_scores,
                        estimator_name=wrapper.estimator_type.value,
                        # #1392: on a subsampled tournament, skip the energy
                        # bootstrap CI — the pre-cap large frame never ran it
                        # (n > max_samples_for_exact), so running it on a
                        # subsample that lands exactly at the cap would ADD
                        # per-estimator latency to the very path this cap
                        # exists to shrink. Below-cap frames keep today's
                        # bootstrap behavior unchanged.
                        _skip_bootstrap=subsampled,
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

        # #1392: the tournament ranked on a subsample — the REPORTED estimate
        # must be a FULL-frame fit of the winner (the refutation node
        # reconstructs the estimator from the full estimation_data passthrough
        # and enforces a reconstructed-vs-reported ATE tolerance). On refit
        # failure the failed result propagates so the consumer fail-closes;
        # the subsample fit is never promoted to the reported estimate.
        if subsampled and selection.success:
            selection = self._refit_winner_on_full_frame(
                selection,
                results,
                winner_wrapper=result_wrappers.get(id(selection)),
                treatment=treatment,
                outcome=outcome,
                covariates=covariates,
                efficiency_mode=efficiency_mode,
                efficiency_controls=efficiency_controls,
                **kwargs,
            )

        # Build energy score comparison
        energy_scores = {r.estimator_type.value: r.energy_score for r in results if r.success}

        # Compute gap between best and second best
        sorted_scores = sorted([s for s in energy_scores.values() if np.isfinite(s)])
        energy_score_gap = 0.0
        if len(sorted_scores) >= 2:
            energy_score_gap = sorted_scores[1] - sorted_scores[0]

        total_time = (time.perf_counter() - total_start) * 1000

        return self._build_selection_result(
            selection=selection,
            results=results,
            total_time_ms=total_time,
            energy_scores=energy_scores,
            energy_score_gap=energy_score_gap,
            adjustment_type=adjustment_type,
            subsampled=subsampled,
            selection_n_rows=selection_n_rows,
            n_rows_total=n_rows_total,
        )

    def _refit_winner_on_full_frame(
        self,
        selection: EstimatorResult,
        results: list[EstimatorResult],
        *,
        winner_wrapper: Optional[BaseEstimatorWrapper],
        treatment: NDArray[np.int_],
        outcome: NDArray[np.float64],
        covariates: pd.DataFrame,
        efficiency_mode: bool,
        efficiency_controls: Optional[pd.DataFrame],
        **kwargs,
    ) -> EstimatorResult:
        """Fit ONLY the tournament winner on the full frame (#1392).

        The subsampled tournament is a ranking; the winner's subsample fit is
        NEVER reported. On refit success the tournament's energy score is
        carried over as the ranking artifact — recomputing it on the full
        frame would (a) mix scoring bases against the losers' subsample
        scores, making ``energy_score_gap`` meaningless, and (b) route through
        the energy distance's internal unseeded >5k sampling, making the
        reported score nondeterministic. ``SelectionResult`` discloses the
        subsample via ``selection_subsampled`` / ``selection_n_rows``.

        On refit failure the FAILED result replaces the winner's tournament
        entry and is returned as the selection, so the consumer fail-closes
        (``estimation.py`` raises ``EstimationError``) rather than silently
        reporting a subsample-fit ATE that the refutation node's full-frame
        reconstruction would diverge from.
        """
        # codex iter-1 MED (#1392): refit the EXACT wrapper instance that won
        # the tournament (recorded by result identity in ``select``). A
        # first-match-by-type fallback would fit the wrong instance when the
        # chain holds duplicate estimator types with different params.
        wrapper = winner_wrapper
        if wrapper is None:  # pragma: no cover — every fitted result is recorded
            logger.warning(
                "No wrapper instance recorded for tournament winner %s; "
                "skipping full-frame refit and failing closed.",
                selection.estimator_type.value,
            )
            return EstimatorResult(
                estimator_type=selection.estimator_type,
                success=False,
                error_message=(
                    "full-frame winner refit unavailable: no wrapper instance "
                    "recorded for the tournament winner "
                    f"({selection.estimator_type.value}); refusing to report "
                    "the subsample-fit estimate."
                ),
                error_type="RefitWrapperMissing",
            )
        fit_frame = covariates
        if efficiency_mode and wrapper.estimator_type not in _EMPTY_BACKDOOR_CAPABLE:
            fit_frame = efficiency_controls  # type: ignore[assignment]
        logger.info(
            "Refitting tournament winner %s on the full frame (%d rows)...",
            wrapper.estimator_type.value,
            len(treatment),
        )
        full_result = wrapper.fit(treatment, outcome, fit_frame, **kwargs)
        if full_result.success:
            full_result.energy_score_result = selection.energy_score_result
        else:
            full_result.error_message = (
                "full-frame winner refit failed after subsampled selection "
                f"(tournament winner={selection.estimator_type.value}): "
                f"{full_result.error_message}"
            )
        # Preserve the invariant that the selected result is a member of
        # all_results: replace the winner's tournament (subsample) entry.
        for i, r in enumerate(results):
            if r is selection:
                results[i] = full_result
                break
        return full_result

    def _build_selection_result(
        self,
        selection: EstimatorResult,
        results: list[EstimatorResult],
        total_time_ms: float,
        energy_scores: Optional[dict[str, float]] = None,
        energy_score_gap: float = 0.0,
        adjustment_type: str = "none",
        subsampled: bool = False,
        selection_n_rows: int = 0,
        n_rows_total: int = 0,
    ) -> SelectionResult:
        """Assemble a SelectionResult and compute the M-est3 reliability gate.

        ``exceeded_max_energy_score`` is True when the selected estimator's
        energy score is above ``config.max_acceptable_energy_score`` (only the
        warn-only branch existed before). ``requires_review`` mirrors it: a
        breach means the selected ATE must be surfaced for review, not reported
        as a clean valid estimate.
        """
        if energy_scores is None:
            energy_scores = {r.estimator_type.value: r.energy_score for r in results if r.success}
        exceeded = bool(
            selection.success and selection.energy_score > self.config.max_acceptable_energy_score
        )
        selection_reason = self._get_selection_reason(selection, results, adjustment_type)
        if subsampled:
            # #1392: user-visible disclosure (this reason is surfaced by the
            # EstimatorComparison UI block) — the ranking ran on a subsample;
            # the reported estimate is the full-frame winner fit.
            selection_reason += (
                f" Tournament ranked on a deterministic stratified subsample of "
                f"{selection_n_rows:,}/{n_rows_total:,} rows; the reported "
                f"ATE/CI come from the winner refit on the full frame."
            )
        return SelectionResult(
            selected=selection,
            selection_strategy=self.config.strategy,
            all_results=results,
            selection_reason=selection_reason,
            total_time_ms=total_time_ms,
            energy_scores=energy_scores,
            energy_score_gap=energy_score_gap,
            exceeded_max_energy_score=exceeded,
            requires_review=exceeded,
            adjustment_type=adjustment_type,
            selection_subsampled=subsampled,
            selection_n_rows=selection_n_rows,
            selection_n_rows_total=n_rows_total,
        )

    def _select_best_energy(self, results: list[EstimatorResult]) -> EstimatorResult:
        """Select the lowest-energy estimator, breaking ties by causal robustness then speed.

        We group the candidates whose energy score is within
        ``min_energy_score_gap`` of the global best (a statistically
        indistinguishable "tie band"). When scores genuinely differ by more than
        the gap, the lowest-energy estimator wins outright. Within the tie band
        the key is ``(confounding_blind?, speed_rank, energy_score)``:

        1. Confounding-robust estimators are preferred over confounding-blind
           ones (``_CONFOUNDING_BLIND_ESTIMATORS``). The energy score measures
           goodness-of-fit on the OUTCOME, not causal validity — a naive OLS can
           tie a DML/forest estimator on fit yet stay biased under confounding
           and fail the refutation gate (MEASURED on patient_journeys: OLS
           gate=BLOCK while the DML/forest family gate=PROCEED at equal energy).
           A fit-tie therefore must NOT hand the run to a confounding-blind
           estimator just because it is fastest.
        2. Among equally-robust candidates, prefer the FASTEST (#622 intent: the
           downstream refutation suite re-fits the selected estimator dozens of
           times — ~0.05s for the linear refit vs ~3.1s for CausalForestDML — so
           on a genuine tie the fast DML beats the slow forest, turning a
           ~35-60 min suite into ~30s). The slow forest still wins outright when
           its energy score is meaningfully better than the gap.
        3. Then lower energy score.

        Naive OLS only wins a tie when it is the ONLY estimator in the band.
        """
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

        # Sort by energy score (lower is better). Stable sort preserves the
        # estimator-chain order within equal scores; we override that below.
        sorted_results = sorted(successful, key=lambda r: r.energy_score)
        best_score = sorted_results[0].energy_score

        # Tie band: every estimator whose energy score is within
        # ``min_energy_score_gap`` of the best. ``inf`` scores (energy not
        # computed) never enter the band unless the best is also ``inf``.
        gap = self.config.min_energy_score_gap
        if np.isfinite(best_score):
            tie_band = [r for r in sorted_results if r.energy_score - best_score <= gap]
        else:
            # All scores are inf (energy uncomputable): the whole successful set
            # is one degenerate band — still prefer the fastest estimator rather
            # than defaulting to the slow chain-priority head.
            tie_band = list(sorted_results)

        if len(tie_band) > 1:
            # Within the tie band, prefer a CONFOUNDING-ROBUST estimator over a
            # confounding-blind one (energy score measures fit, not causal
            # validity — see _CONFOUNDING_BLIND_ESTIMATORS). Among equally-robust
            # candidates keep #622's intent: prefer the FASTEST to bound the
            # downstream refutation latency, then the lower energy score. So the
            # tiebreak key is (confounding_blind?, speed_rank, energy_score):
            # naive OLS only wins a tie when it is the ONLY estimator in the band.
            best = min(
                tie_band,
                key=lambda r: (
                    1 if r.estimator_type in _CONFOUNDING_BLIND_ESTIMATORS else 0,
                    _ESTIMATOR_SPEED_RANK.get(r.estimator_type, 99),
                    r.energy_score,
                ),
            )
            if best is not sorted_results[0]:
                logger.info(
                    "Energy-score tie within gap %.4f: selected confounding-robust "
                    "estimator %s (energy=%.4f) over %s (energy=%.4f) — energy ties "
                    "do not justify a confounding-blind estimator; fastest robust wins.",
                    gap,
                    best.estimator_type.value,
                    best.energy_score,
                    sorted_results[0].estimator_type.value,
                    sorted_results[0].energy_score,
                )
        else:
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
        self,
        selected: EstimatorResult,
        all_results: list[EstimatorResult],
        adjustment_type: str = "none",
    ) -> str:
        """Generate human-readable selection reason."""
        if not selected.success:
            return "All estimators failed; returning last attempt"

        successful = [r for r in all_results if r.success]
        skipped = [r for r in all_results if r.skipped]

        # #1188 efficiency mode: a randomized design where pre-treatment
        # baselines entered as variance-reduction controls. Frame the covariate
        # estimators as PRECISION tools — randomization already de-biases; the
        # unadjusted OLS contrast stays the unbiased anchor. (Deliberately no
        # de-biasing language here — that would misstate what adjustment does
        # on an RCT.)
        if adjustment_type == "efficiency":
            best_line = (
                f"Selected {selected.estimator_type.value} by lowest energy score. "
                if selected.estimator_type not in _EMPTY_BACKDOOR_CAPABLE
                else f"Selected the unadjusted contrast ({selected.estimator_type.value}). "
            )
            return best_line + (
                "Randomized design: baseline covariates enter only for variance "
                "reduction (ANCOVA-style precision — they tighten the interval "
                "while randomization keeps every point estimate unbiased); the "
                "unadjusted contrast (ols) remains the reference anchor."
            )

        # Empty-backdoor (randomized / exogenous treatment) case: the covariate-
        # requiring estimators were skipped as not-applicable, so the unadjusted
        # contrast (OLS) is the only — and correct — estimator. Say so plainly.
        if skipped and selected.estimator_type in _EMPTY_BACKDOOR_CAPABLE:
            return (
                "No covariates to adjust for (randomized / empty-backdoor design), so "
                f"the unadjusted contrast ({selected.estimator_type.value}) is the "
                "correct estimator; covariate-based estimators are not applicable here."
            )

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
