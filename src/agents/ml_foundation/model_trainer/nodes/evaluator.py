"""Model evaluation for model_trainer.

This module evaluates trained models on train/validation/test sets
using real sklearn metrics with bootstrap confidence intervals.

Version: 2.0.0
"""

import copy
import logging
import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np

# numpy>=2.0 removed the long-deprecated ``np.trapz`` in favour of
# ``np.trapezoid`` (identical semantics). Resolve once so the net-benefit
# integration works on both numpy 1.x and 2.x.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.lifecycle import GateLifecycleState

from .advanced_validation import (
    DEFAULT_PERMUTATION_COUNT,
    apply_post_hoc_calibration,
    check_imbalance_aware_suspicion,
    compute_calibration_analysis,
    compute_permutation_test,
    compute_stratified_cv,
    optimize_threshold_f1,
    validate_stratified_splits,
)
from .model_eval_ablation import (
    DEFAULT_MODEL_EVAL_ABLATION_MAX_FEATURES,
    DEFAULT_MODEL_EVAL_ABLATION_PERMUTATIONS,
    DEFAULT_MODEL_EVAL_PERMUTATION_PERMS,
    MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT,
    MODEL_EVAL_ABLATION_HIGH_Z,
    MODEL_EVAL_ABLATION_MODERATE_Z,
    MODEL_EVAL_ABLATION_STRONG_EFFECT_DEFAULT,
    run_model_eval_ablation,
)

logger = logging.getLogger(__name__)


# Plan v4 Gate N2 — lifecycle-state declarations for the two advisory
# emitters in this module. Both currently in ADVISORY: they emit signals
# (logs + ``validation_metrics`` keys) but do NOT mutate
# ``success_criteria_met`` or block the deployer. Transitions to
# CALIBRATING / ENFORCED require a signed doc at
# ``docs/calibration/{slug}_lifecycle_change_*.md`` per Gate N2 acceptance #3.
LIFECYCLE_STATE_T22: GateLifecycleState = GateLifecycleState.ADVISORY
LIFECYCLE_STATE_T23: GateLifecycleState = GateLifecycleState.ADVISORY


_CV_PROMOTED_METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "mcc", "f1")
_CV_PROMOTED_STATS: tuple[str, ...] = ("mean", "std")

# Plan v3 §3 Tier 1B step 1: keys promoted from the permutation_test sub-dict
# into the scalar-only ``validation_metrics`` payload for HBLP gating, deployer
# input contracts (T2.6a), and downstream observability dashboards.
_PERMUTATION_PROMOTED_KEYS: tuple[str, ...] = (
    "permutation_pvalue",
    "permutation_null_p95",
    "permutation_null_p99",
    "permutation_n_permutations",
    "permutation_n_effective",
    "permutation_auc_mean",
    "permutation_auc_std",
)

# Plan v3 §4 T2.2 — Permutation-anchored AUC floor (advisory mode).
# Default buffer above the empirical permutation null p99 that a deployable
# model's test AUC must exceed. Backlog #135: empirically calibrated to
# 0.04 via the 5-seed × 7-target-AUC sweep at
# `scripts/calibration/aggregate_t22_sweep.py` against
# `synthetic_rwd_realistic` (n=1400, prevalence=0.10). The §2.3 well-
# conditioned reading (target cells where every seed exceeds perm null
# p99) yielded: limiting cell target_auc=0.70 with P5 margin = 0.0597 →
# floor(0.0597 * 100)/100 = 0.05 → 0.05 - 0.01 safety = **0.04** buffer.
# The mechanical reading (all cells, no exclusion) clamps to 0.0 at
# small n because low-signal target cells (0.55-0.65) can produce models
# whose AUC falls below the perm-null p99 — that's a regime+sample-size
# property, not a buffer-calibration one. See
# `docs/calibration/t22_perm_anchored_synth_20260510_results.md`.
# Until T2.6c promotion, this is an OBSERVABILITY threshold only —
# violations emit a structured warning and a flag on validation_metrics,
# but do NOT enter `success_criteria` and do NOT block the deployer.
T2_2_PERMUTATION_ANCHORED_AUC_BUFFER_DEFAULT: float = 0.04

# Plan v3 §4 T2.3 — Cohort-derived honest band defaults.
# The "honest band" is the cohort-conditional range of test AUCs that a
# deployable model could plausibly achieve without leakage. Pre-T2.3 the
# band was a hardcoded `[0.62, 0.68]` literal in
# `synthetic_rwd_realistic.py` (calibrated for one synthetic regime); T2.3
# replaces it with a per-cohort derivation from `permutation_auc_std` +
# `baseline_test_auc` + the configured lift bounds.
#
# Lower bound (`honest_band_lo`) is the larger of:
#   1. baseline_test_auc + T2_3_HONEST_BAND_MIN_LIFT_DEFAULT (5pp lift over
#      the stratified-dummy baseline = "operationally meaningful").
#   2. permutation_null_p99 + T2_3_HONEST_BAND_NOISE_SIGMA_DEFAULT *
#      permutation_auc_std (statistically distinguishable from noise).
#
# Upper bound (`honest_band_hi`) is the smaller of:
#   3. baseline_test_auc + T2_3_HONEST_BAND_MAX_LIFT_DEFAULT (30pp lift =
#      "above this is suspicious for leakage on RWD claims data").
#   4. T2_3_HONEST_BAND_CEILING_DEFAULT (absolute cap; AUC > 0.95 on real
#      RWD is essentially never honest per published claims-only research).
#
# Constants are calibration-anchored defaults, NOT regulatory floors. A
# caller (T2.6c enforcement phase) can override per cohort.
T2_3_HONEST_BAND_MIN_LIFT_DEFAULT: float = 0.05
T2_3_HONEST_BAND_MAX_LIFT_DEFAULT: float = 0.30
T2_3_HONEST_BAND_CEILING_DEFAULT: float = 0.95
T2_3_HONEST_BAND_NOISE_SIGMA_DEFAULT: float = 1.0

# Plan v4 §6 G4 — T2.3 lifecycle marker.
#
# WARNING — DOCUMENTATION-ONLY MARKER.
#
# This constant is a documentation-only marker; it is NOT consumed anywhere
# in the runtime pipeline. Changing its value to "shadow" or "enforced" will
# NOT alter pipeline behavior, will NOT block the deployer, and will NOT cause
# any band-violation flag to graduate to a denial. It exists solely so that
# code-readers can see at a glance the band's intended lifecycle stage and
# correlate it with the docs at `docs/calibration/t23_cohort_bands_20260510.md`.
#
# To make this state actually load-bearing, BOTH of the following are required:
#   1. Thread the value into `_emit_cohort_derived_honest_band` and emit a
#      `honest_band_lifecycle_state` key on `validation_metrics` so downstream
#      consumers (deployer, observability dashboards) can branch on it.
#   2. Update `compute_deployer_input_metrics` (or a successor) to consume
#      `honest_band_violated` AND the lifecycle state — denying promotion only
#      when state == "enforced" AND honest_band_violated is True.
#
# Per codex-rescue 2026-05-10 G4 review (Q2): we KEEP this constant
# module-level and documentation-only until shadow promotion lands. Threading
# the state into runtime metrics now would create fake state-thread semantics
# that imply the lifecycle has runtime meaning when it does not.
#
# Why kept at all: the marker IS load-bearing as a code-review and
# documentation-correlation aid — it forces any future PR proposing band
# promotion to update this constant in lockstep with the runtime change, which
# in turn forces a code-review pass over the band-derivation constants and the
# t23_cohort_bands doc. Removing it would lose that lockstep.
#
# The honest band is currently advisory-observability-only — the band-derivation
# constants above (MIN_LIFT, MAX_LIFT, CEILING, NOISE_SIGMA) were calibrated
# against cohort metrics whose values were used to argue for the band's
# correctness (CSU n=9607 val_AUC=0.6592, Optum n=1294 cv_mean=0.6795,
# synthetic_rwd_realistic [0.62, 0.68]). Promoting them to deployer enforcement
# would be data-snooping until an un-touched cohort is onboarded.
#
# See `docs/calibration/t23_cohort_bands_20260510.md` for the advisory framing
# and `docs/calibration/t26_future_cohort_plan_20260510.md` for the graduation
# criteria. Possible future values: "advisory" (current), "shadow" (T2.6b
# integration), "enforced" (post-graduation).
T23_BAND_LIFECYCLE_STATE: Literal["advisory", "shadow", "enforced"] = "advisory"

# Backlog #20 Gap 2: F1-fallback fires when validation MCC at the
# canonically-chosen threshold falls below this floor. 0.20 is the band
# below which the canonical Youden's J / cost-optimal pick is treated as
# unreliable (extreme imbalance + model that doesn't separate classes
# well), and we attempt the F1-optimal threshold as a recovery move.
# The fallback only swaps when F1-optimal STRICTLY improves MCC, so
# raising the floor never makes the model worse — it just expands the set
# of cases where a recovery attempt is made.
_F1_FALLBACK_MCC_THRESHOLD: float = 0.20


def _promote_cv_summary_to_validation_metrics(
    metrics_result: Dict[str, Any], cv_result: Dict[str, Any]
) -> None:
    """Promote scalar CV summary metrics into ``validation_metrics``.

    The TIER0_E2E_JSON_OUT artifact filter (the dict comprehension that
    builds ``"validation_metrics"`` in ``scripts/run_tier0_test.py``'s
    ``main()`` — grep for ``state.get("validation_metrics")`` to find it)
    only retains scalar values (``int``, ``float``, ``str``, ``None``) on
    the ``validation_metrics`` payload. Without this promotion the cv
    summary lives in ``metrics_result["cv_results"]`` (a sub-dict) and
    gets dropped by the filter, forcing downstream consumers to stdout-
    scrape the ``"5-fold CV AUC: ..."`` log line.

    Closes backlog #18. Mutates ``metrics_result`` in place.

    Adds ``cv_5fold_<metric>_<stat>`` keys for {roc_auc, pr_auc, mcc, f1}
    × {mean, std} when each source key is present in ``cv_result``.

    The ``cv_5fold_`` prefix is intentionally hardcoded against the
    fixed-5-fold CV at the runner's evaluator callsite. If a future caller
    wants a different fold count, they MUST update both the helper's
    naming convention and downstream JSON consumers in lockstep — the
    ``n_folds`` assertion below fails loudly to surface the intent
    (codex pass-1 MEDIUM-2: runtime fold-count drift would silently emit
    misleading key names).
    """
    cv_n_folds = cv_result.get("n_folds")
    if cv_n_folds is not None and cv_n_folds != 5:
        raise ValueError(
            f"_promote_cv_summary_to_validation_metrics: cv_result n_folds={cv_n_folds!r}, "
            f"but the JSON-artifact key prefix is hardcoded to 'cv_5fold_'. "
            f"Update both the prefix and downstream consumers if a different "
            f"fold count is intended."
        )
    val_metrics = metrics_result.setdefault("validation_metrics", {})
    for metric in _CV_PROMOTED_METRICS:
        for stat in _CV_PROMOTED_STATS:
            src_key = f"cv_{metric}_{stat}"
            if src_key in cv_result:
                val_metrics[f"cv_5fold_{metric}_{stat}"] = cv_result[src_key]


def _emit_permutation_anchored_auc_advisory(
    metrics_result: Dict[str, Any],
    test_metrics: Dict[str, Any],
    permutation_result: Dict[str, Any],
    buffer: float = T2_2_PERMUTATION_ANCHORED_AUC_BUFFER_DEFAULT,
) -> None:
    """Plan v3 §4 T2.2 — Permutation-anchored AUC floor (advisory mode only).

    Computes ``auc_above_permutation_null = test_auc - permutation_null_p99``
    and emits an observability advisory when the lift is smaller than
    ``buffer``. Surfaces three keys on ``validation_metrics``:

      * ``auc_above_permutation_null`` — float, the signed margin (test AUC
        minus the empirical permutation null p99). Negative ⇒ test AUC is
        below the upper tail of the null; positive but small ⇒ in the
        gray zone of "barely above noise".
      * ``permutation_anchored_auc_buffer`` — the buffer (default 0.04;
        calibrated 2026-05-12 via backlog #135).
        Ships on the artifact so an operator reading
        ``validation_metrics`` can audit the threshold even when the
        advisory is not violated.
      * ``permutation_anchored_auc_advisory_violated`` — bool. True when
        ``auc_above_permutation_null < buffer`` AND both inputs are
        finite. False when the criterion is met. None when either input
        is missing (perm test was degenerate or test AUC is missing).

    Pure observability: does NOT mutate ``success_criteria_met`` and does
    NOT add a key to ``success_criteria``. Plan §6 T2.2: "emitted in
    advisory mode for one quarter before enforcement". The T2.6c
    enforcement phase (separate work) is where this graduates to a
    deployer gate.

    No-ops gracefully when ``permutation_result`` is empty or its
    ``permutation_null_p99`` key is None / absent — preserves backward
    compat with callers that have not yet adopted plan §3 Tier 1B step 1.
    """
    val_metrics = metrics_result.setdefault("validation_metrics", {})
    null_p99 = permutation_result.get("permutation_null_p99")
    test_auc = test_metrics.get("roc_auc")

    val_metrics["permutation_anchored_auc_buffer"] = float(buffer)

    if null_p99 is None or test_auc is None:
        # Degenerate perm run OR no test AUC — surface as None so
        # operators can distinguish "advisory not evaluated" from
        # "advisory evaluated and met".
        val_metrics["auc_above_permutation_null"] = None
        val_metrics["permutation_anchored_auc_advisory_violated"] = None
        return

    margin = float(test_auc) - float(null_p99)
    val_metrics["auc_above_permutation_null"] = margin
    violated = margin < float(buffer)
    val_metrics["permutation_anchored_auc_advisory_violated"] = violated

    if violated:
        logger.warning(
            "T2.2 ADVISORY: test AUC=%.4f only %+.4f above permutation "
            "null p99 (=%.4f); below the %.2f buffer. NOT enforced "
            "(advisory mode per plan v3 §4 T2.2); deployer is unaffected. "
            "If this persists across cohorts, the model's signal is "
            "indistinguishable from label permutation noise.",
            test_auc,
            margin,
            null_p99,
            buffer,
        )


def _promote_permutation_summary_to_validation_metrics(
    metrics_result: Dict[str, Any], permutation_result: Dict[str, Any]
) -> None:
    """Promote permutation-test scalar keys into ``validation_metrics``.

    Plan v3 §3 Tier 1B step 1 prerequisite for HBLP gating: the
    permutation-null distribution percentiles (``p95``, ``p99``) and shuffle
    count must be visible on the scalar-only ``validation_metrics`` payload
    so downstream HBLP variance-inflation logic, deployer T2.6 metric
    contract, and observability dashboards do not have to descend into the
    nested ``permutation_test`` sub-dict (which the JSON-artifact filter in
    ``scripts/run_tier0_test.py`` strips). Same pattern as
    ``_promote_cv_summary_to_validation_metrics``.

    Mutates ``metrics_result`` in place. None values from a degenerate
    permutation run (single-class y, missing y_proba) ARE promoted as
    None — downstream consumers must distinguish "perm test ran but null
    is degenerate" from "perm test was never executed" (key absent).

    Asymmetric with ``_promote_cv_summary_to_validation_metrics``: this
    helper guards each promotion with ``if key in permutation_result``
    (forward-compat for older callers that emit a partial dict on
    NaN-degenerate runs), whereas the CV promoter unconditionally reads
    each ``cv_result`` key (the CV result dict has a fixed contract).
    Both patterns are intentional and codex-flagged in pass-1.
    """
    val_metrics = metrics_result.setdefault("validation_metrics", {})
    for key in _PERMUTATION_PROMOTED_KEYS:
        if key in permutation_result:
            val_metrics[key] = permutation_result[key]


def _emit_cohort_derived_honest_band(
    metrics_result: Dict[str, Any],
    test_metrics: Dict[str, Any],
    permutation_result: Dict[str, Any],
    *,
    min_lift: float = T2_3_HONEST_BAND_MIN_LIFT_DEFAULT,
    max_lift: float = T2_3_HONEST_BAND_MAX_LIFT_DEFAULT,
    ceiling: float = T2_3_HONEST_BAND_CEILING_DEFAULT,
    noise_sigma: float = T2_3_HONEST_BAND_NOISE_SIGMA_DEFAULT,
) -> None:
    """Plan v3 §4 T2.3 — cohort-derived honest band (advisory mode only).

    ADVISORY OBSERVABILITY ONLY — see
    ``docs/calibration/t23_cohort_bands_20260510.md`` for the advisory
    framing and ``docs/calibration/t26_future_cohort_plan_20260510.md`` for
    the promotion-to-enforcement criteria. The band-derivation constants
    (``T2_3_HONEST_BAND_*_DEFAULT``) were calibrated against cohort metrics
    whose values were used to argue for the band's correctness (CSU n=9607
    val_AUC=0.6592, Optum n=1294 cv_mean=0.6795, synthetic_rwd_realistic
    [0.62, 0.68]); promoting them to deployer enforcement is data-snooping
    until an un-touched cohort lands. The lifecycle marker
    ``T23_BAND_LIFECYCLE_STATE`` (above) is set to ``"advisory"``.

    Surfaces the cohort-conditional honest range of test AUC values onto
    ``validation_metrics``. An "honest" AUC is large enough to be
    statistically distinguishable from random label-shuffle noise AND
    operationally meaningful relative to a stratified-dummy baseline,
    but not so large that leakage is implausible to rule out.

    Surface keys (all on ``validation_metrics``):

      * ``honest_band_lo`` — lower bound of the honest range, or None when
        either ``baseline_test_auc`` (from `_compute_baseline_test_metrics`)
        or ``permutation_null_p99`` is missing. Computed as
        ``max(baseline + min_lift, perm_null_p99 + noise_sigma * perm_auc_std)``.
      * ``honest_band_hi`` — upper bound, computed as
        ``min(ceiling, baseline + max_lift)``. None when baseline is missing.
      * ``honest_band_baseline_test_auc`` — the input from ``test_metrics``.
      * ``honest_band_perm_null_p99`` — the input from ``permutation_result``.
      * ``honest_band_perm_auc_std`` — the input from ``permutation_result``.
      * ``honest_band_min_lift`` / ``honest_band_max_lift`` /
        ``honest_band_ceiling`` / ``honest_band_noise_sigma`` — the configured
        thresholds. Always emitted for operator audit, even when the band
        cannot be evaluated.
      * ``honest_band_violated`` — bool. True iff test_auc lies OUTSIDE
        ``[lo, hi]`` AND all required inputs are present. None when
        evaluation cannot proceed.
      * ``honest_band_position`` — Literal[``"below"``, ``"in_band"``, ``"above"``],
        or None. Operator triage signal.

    Pure observability — does NOT mutate ``success_criteria``,
    ``success_criteria_met``, or ``success_criteria_results``. Plan §4 T2.3:
    this replaces the hardcoded ``[0.62, 0.68]`` literal in
    ``synthetic_rwd_realistic.py`` with a per-cohort derivation; the band
    is emitted for downstream consumption (T2.6c enforcement phase, the
    integration test ``test_csu_val_auc_measurement.py``, observability
    dashboards), not enforced.

    No-ops gracefully when any required input is missing — preserves
    backward compat with callers that pre-date the perm-null surface
    (Tier 1B step 1) or the baseline_test_auc emit.
    """
    val_metrics = metrics_result.setdefault("validation_metrics", {})

    val_metrics["honest_band_min_lift"] = float(min_lift)
    val_metrics["honest_band_max_lift"] = float(max_lift)
    val_metrics["honest_band_ceiling"] = float(ceiling)
    val_metrics["honest_band_noise_sigma"] = float(noise_sigma)

    baseline_auc = test_metrics.get("baseline_test_auc")
    perm_null_p99 = permutation_result.get("permutation_null_p99")
    perm_auc_std = permutation_result.get("permutation_auc_std")
    test_auc = test_metrics.get("roc_auc")

    val_metrics["honest_band_baseline_test_auc"] = baseline_auc
    val_metrics["honest_band_perm_null_p99"] = perm_null_p99
    val_metrics["honest_band_perm_auc_std"] = perm_auc_std

    if baseline_auc is None:
        val_metrics["honest_band_lo"] = None
        val_metrics["honest_band_hi"] = None
        val_metrics["honest_band_violated"] = None
        val_metrics["honest_band_position"] = None
        return

    baseline_auc_f = float(baseline_auc)

    # Upper bound: cap at ceiling OR baseline + max_lift, whichever is lower.
    hi = min(float(ceiling), baseline_auc_f + float(max_lift))

    # Lower bound: max of operationally-meaningful lift OR statistical
    # distinguishability above the perm-null upper tail. When the perm-null
    # inputs are missing, fall back to baseline + min_lift only.
    lo_meaningful = baseline_auc_f + float(min_lift)
    if perm_null_p99 is not None and perm_auc_std is not None:
        lo_distinguishable = float(perm_null_p99) + float(noise_sigma) * float(perm_auc_std)
        lo = max(lo_meaningful, lo_distinguishable)
    else:
        lo = lo_meaningful

    # Pathological: lo > hi (cohort with high baseline, narrow ceiling).
    # Surface as None for both bounds so downstream consumers don't see an
    # empty band.
    if lo > hi:
        val_metrics["honest_band_lo"] = None
        val_metrics["honest_band_hi"] = None
        val_metrics["honest_band_violated"] = None
        val_metrics["honest_band_position"] = None
        logger.warning(
            "T2.3 honest band collapsed: derived lo=%.4f > hi=%.4f "
            "(baseline=%.4f, perm_null_p99=%s, perm_auc_std=%s). "
            "Likely cause: very high baseline + tight ceiling. Caller "
            "may need to relax max_lift or raise ceiling for this cohort.",
            lo,
            hi,
            baseline_auc_f,
            perm_null_p99,
            perm_auc_std,
        )
        return

    val_metrics["honest_band_lo"] = float(lo)
    val_metrics["honest_band_hi"] = float(hi)

    if test_auc is None:
        val_metrics["honest_band_violated"] = None
        val_metrics["honest_band_position"] = None
        return

    test_auc_f = float(test_auc)
    if test_auc_f < lo:
        position = "below"
        violated = True
    elif test_auc_f > hi:
        position = "above"
        violated = True
    else:
        position = "in_band"
        violated = False

    val_metrics["honest_band_violated"] = violated
    val_metrics["honest_band_position"] = position

    if violated:
        logger.warning(
            "T2.3 ADVISORY: test AUC=%.4f is %s honest band [%.4f, %.4f] "
            "(baseline=%.4f, perm_null_p99=%s). NOT enforced (advisory mode "
            "per plan v3 §4 T2.3); deployer is unaffected. Position=%s — "
            "if 'below', signal is too weak relative to baseline OR noise; "
            "if 'above', leakage is harder to rule out for this cohort.",
            test_auc_f,
            "below" if position == "below" else "above",
            lo,
            hi,
            baseline_auc_f,
            perm_null_p99,
            position,
        )


def _compute_baseline_test_metrics(
    y_train: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    problem_type: str,
) -> Dict[str, float]:
    """Stratified-dummy baseline AUC for the lift-over-baseline criterion.

    The baseline is a ``DummyClassifier(strategy="stratified")`` fit on
    ``y_train`` — its test-set AUC is the random-with-class-prior reference
    against which the trained model's lift is measured. ``"stratified"`` is
    the only strategy whose test predictions span both classes, so
    ``roc_auc_score`` is well-defined; ``"most_frequent"`` and ``"prior"``
    yield single-class predictions where AUC is undefined.

    Returns an empty dict when the problem is not binary classification, or
    when either split is too small / single-class to compute a meaningful
    baseline. The caller treats an empty return as "skip the criterion"
    (see narrow exemption at ``_check_success_criteria``); this is the only
    success criterion that is exemptable, and only under these guards.

    See ``.claude/plans/pre_phase2_unblockers.md`` Section B for the design.
    """
    if problem_type != "binary_classification":
        return {}
    if y_train is None or y_test is None:
        return {}
    if len(y_train) < 10 or len(y_test) < 10:
        return {}
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)
    if np.unique(y_train_arr).size < 2 or np.unique(y_test_arr).size < 2:
        return {}

    dummy = DummyClassifier(strategy="stratified", random_state=42)  # noqa: random_state=42 — design-intentional fixed seed: DummyClassifier baseline must be reproducible across folds for variance interpretation (per cycle-14 Q2 RESOLVED 2026-05-02)
    dummy.fit(np.zeros((len(y_train_arr), 1)), y_train_arr)
    proba = dummy.predict_proba(np.zeros((len(y_test_arr), 1)))[:, 1]
    return {"baseline_test_auc": float(roc_auc_score(y_test_arr, proba))}


def _positive_class_proba(y_proba: np.ndarray) -> np.ndarray:
    """Return the positive-class probability column from a 1D or 2D proba array.

    sklearn's ``predict_proba`` returns a 2D ``(n_samples, n_classes)`` array
    where column 1 is P(class=1) for binary classifiers. Some callers
    (calibrators, custom wrappers) may already pass the 1D positive-class
    column. This helper accepts either shape and returns the 1D array.
    """
    if y_proba.ndim == 2:
        return y_proba[:, 1]
    return y_proba


async def evaluate_model(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate trained model on train/validation/test sets.

    CRITICAL EVALUATION PRINCIPLES:
    - Test set touched ONCE for final evaluation
    - Validation set already used for HPO
    - Training set evaluation for overfitting detection
    - Holdout set NOT evaluated (locked for post-deployment)

    Args:
        state: ModelTrainerState with trained_model, preprocessed data,
               problem_type, success_criteria

    Returns:
        Dictionary with train_metrics, validation_metrics, test_metrics,
        problem-specific metrics (auc_roc, precision, recall for classification;
        rmse, mae, r2 for regression), confidence_interval,
        success_criteria_met, success_criteria_results

    Raises:
        No exceptions - returns error in state if evaluation fails
    """
    # #773: an upstream node (train_model et al.) already failed — emit
    # NOTHING so the true ``error``/``error_type`` (training_failed,
    # instantiation_failed, unsupported_algorithm, ...) survives to the
    # caller (agent.py raises "Training error (<error_type>)" from the
    # merged state). The graph wires train_model -> evaluate_model
    # unconditionally, so without this guard every training failure was
    # re-masked here as "missing_trained_model" by the guard below. Same
    # F2 skip-on-upstream-error idiom as augment_training_data /
    # learning_curve; downstream conditionals route to END on the
    # pre-existing error regardless.
    if state.get("error"):
        logger.info(
            "Skipping evaluation: upstream error already set (%s) — "
            "preserving it instead of masking as missing_trained_model",
            state.get("error_type", "unknown"),
        )
        return {}

    # Extract trained model and data
    trained_model = state.get("trained_model")
    problem_type = state.get("problem_type", "binary_classification")
    success_criteria = state.get("success_criteria", {})
    # Block 5 (#10): optional dict mapping {tp,fp,fn,tn} → per-prediction
    # dollar value. None = skip business_utility computation.
    cost_matrix = state.get("cost_matrix")
    # Backlog #20 Gap 1: opt-in cost-aware threshold selection. Default
    # OFF preserves backward compatibility with synthetic baseline tests
    # that pinned Youden's J behaviour while cost_matrix was already
    # plumbed for business_utility reporting. Production callers that
    # want a utility-maximising operating point set this to True.
    use_cost_optimal_threshold = bool(state.get("use_cost_optimal_threshold", False))

    # Extract preprocessed data
    X_train_preprocessed = state.get("X_train_preprocessed")
    X_validation_preprocessed = state.get("X_validation_preprocessed")
    X_test_preprocessed = state.get("X_test_preprocessed")
    train_data = state.get("train_data", {})
    validation_data = state.get("validation_data", {})
    test_data = state.get("test_data", {})
    y_train = train_data.get("y")
    y_validation = validation_data.get("y")
    y_test = test_data.get("y")

    # Validate required inputs
    if trained_model is None:
        logger.error("No trained model available for evaluation")
        return {
            "error": "No trained model available for evaluation",
            "error_type": "missing_trained_model",
        }

    if X_test_preprocessed is None or y_test is None:
        logger.error("Missing test data for evaluation")
        return {
            "error": "Missing test data for evaluation",
            "error_type": "missing_test_data",
        }

    # The preprocessor (preprocessor.py:152) returns a numpy array, so
    # X_*_preprocessed is numpy. LightGBM 4.x stores feature_names_in_
    # at fit time even on numpy input ('Column_0..N'), and sklearn warns
    # on every predict where X has no feature names. Wrap X with the
    # preprocessor's post-encoding feature names so predict sees them
    # consistently. y_* stays numpy — metric helpers expect plain arrays.
    X_train_np = _wrap_with_feature_names(X_train_preprocessed, state)
    X_val_np = _wrap_with_feature_names(X_validation_preprocessed, state)
    X_test_np = _wrap_with_feature_names(X_test_preprocessed, state)
    y_train_np = _ensure_numpy(y_train)
    y_val_np = _ensure_numpy(y_validation)
    y_test_np = _ensure_numpy(y_test)

    logger.info(
        f"Evaluating model: problem_type={problem_type}, "
        f"X_test shape={X_test_np.shape if X_test_np is not None else 'None'}"
    )

    # Make predictions on all sets
    try:
        assert X_test_np is not None
        predictions = _make_predictions(
            model=trained_model,
            X_train=X_train_np,
            X_val=X_val_np,
            X_test=X_test_np,
            problem_type=problem_type,
        )
    except Exception as e:
        logger.error(f"Prediction failed during evaluation: {e}")
        return {
            "error": f"Prediction failed during evaluation: {str(e)}",
            "error_type": "prediction_failed",
        }

    # Get imbalance detection status
    imbalance_detected = state.get("imbalance_detected", False)
    minority_ratio = state.get("minority_ratio", 0.5)

    # y_test_np must be valid at this point (checked above)
    assert y_test_np is not None

    # Compute metrics based on problem type
    # Cycle-16 I-4 (Q2-B): thread per-fold seed into bootstrap CI to make
    # asyncio.gather n_jobs > 1 deterministic. Single-mode callers don't
    # supply fold_random_state — None preserves legacy global-RNG path.
    bootstrap_random_state = state.get("fold_random_state")
    try:
        if problem_type in ["binary_classification"]:
            metrics_result = _compute_classification_metrics(
                y_train=y_train_np,
                y_train_pred=predictions["y_train_pred"],
                y_train_proba=predictions["y_train_proba"],
                y_validation=y_val_np,
                y_validation_pred=predictions["y_val_pred"],
                y_validation_proba=predictions["y_val_proba"],
                y_test=y_test_np,
                y_test_pred=predictions["y_test_pred"],
                y_test_proba=predictions["y_test_proba"],
                imbalance_detected=imbalance_detected,
                minority_ratio=minority_ratio,
                cost_matrix=cost_matrix,
                bootstrap_random_state=bootstrap_random_state,
                success_criteria=success_criteria,
                use_cost_optimal_threshold=use_cost_optimal_threshold,
            )
        elif problem_type == "multiclass_classification":
            metrics_result = _compute_multiclass_metrics(
                y_train=y_train_np,
                y_train_pred=predictions["y_train_pred"],
                y_train_proba=predictions["y_train_proba"],
                y_validation=y_val_np,
                y_validation_pred=predictions["y_val_pred"],
                y_validation_proba=predictions["y_val_proba"],
                y_test=y_test_np,
                y_test_pred=predictions["y_test_pred"],
                y_test_proba=predictions["y_test_proba"],
            )
        elif problem_type in ["regression", "continuous"]:
            metrics_result = _compute_regression_metrics(
                y_train=y_train_np,
                y_train_pred=predictions["y_train_pred"],
                y_validation=y_val_np,
                y_validation_pred=predictions["y_val_pred"],
                y_test=y_test_np,
                y_test_pred=predictions["y_test_pred"],
                bootstrap_random_state=bootstrap_random_state,
            )
        else:
            logger.error(f"Unsupported problem type: {problem_type}")
            return {
                "error": f"Unsupported problem type: {problem_type}",
                "error_type": "unsupported_problem_type",
            }
    except Exception as e:
        logger.error(f"Metrics computation failed: {e}")
        return {
            "error": f"Metrics computation failed: {str(e)}",
            "error_type": "metrics_computation_failed",
        }

    # =========================================================================
    # ADVANCED VALIDATION (imbalance-aware)
    # =========================================================================
    # #633: the model the pipeline DEPLOYS (MLflow-logged / checkpointed /
    # returned). Defaults to the raw trained_model for every problem type;
    # the binary-classification calibration block below promotes it to the
    # calibrated estimator when post-hoc calibration is actually applied.
    deployed_model: Any = trained_model
    deployed_calibration_applied = False
    if problem_type == "binary_classification":
        y_test_proba = predictions.get("y_test_proba")

        # 1. Permutation test — confirm signal is genuine. Plan v3 §3 Tier 1B
        # step 1 raises the default shuffle count to 200 so p95/p99 percentiles
        # of the null distribution stabilize for downstream HBLP gating.
        logger.info("Running permutation test (%d shuffles)...", DEFAULT_PERMUTATION_COUNT)
        permutation_result = compute_permutation_test(
            y_test_np, y_test_proba, n_permutations=DEFAULT_PERMUTATION_COUNT
        )
        metrics_result["permutation_test"] = permutation_result
        _promote_permutation_summary_to_validation_metrics(metrics_result, permutation_result)
        # Plan v3 §4 T2.2 — Permutation-anchored AUC floor (advisory mode).
        # Emits `auc_above_permutation_null`, `permutation_anchored_auc_buffer`,
        # `permutation_anchored_auc_advisory_violated` on validation_metrics.
        # Uses the test_metrics computed above (within metrics_result). Pure
        # observability: no success_criteria mutation, no deployer impact.
        _emit_permutation_anchored_auc_advisory(
            metrics_result,
            metrics_result.get("test_metrics", {}),
            permutation_result,
        )
        # Plan v3 §4 T2.3 — Cohort-derived honest band (advisory mode).
        # Replaces the hardcoded `[0.62, 0.68]` literal in
        # `synthetic_rwd_realistic.py` with a per-cohort derivation from
        # baseline_test_auc + perm_null_p99 + perm_auc_std. Pure observability:
        # no success_criteria mutation, no deployer impact.
        _emit_cohort_derived_honest_band(
            metrics_result,
            metrics_result.get("test_metrics", {}),
            permutation_result,
        )
        if permutation_result.get("signal_genuine") is not None:
            p95 = permutation_result.get("permutation_null_p95")
            p99 = permutation_result.get("permutation_null_p99")
            logger.info(
                "Permutation test: p=%.4f, null_p95=%s, null_p99=%s, signal_genuine=%s",
                permutation_result["permutation_pvalue"],
                f"{p95:.4f}" if p95 is not None else "None",
                f"{p99:.4f}" if p99 is not None else "None",
                permutation_result["signal_genuine"],
            )

        # 1.5. Phase 3.4 — model-trainer Layer 3 ablation (advisory, opt-in).
        # Plan ref: .claude/plans/adaptive_temporal_validity_redesign.md line
        # 245. Wires ``compute_feature_ablation`` on the encoded test split
        # AFTER ``preprocessor.fit_transform`` has run (one-hot + scaler +
        # imputer). The hook catches a leak class Phase 3.3 cannot see:
        # per-category leak through OneHotEncoder. Phase 3.3 numeric ablation
        # skips categoricals; Cramér's V on the whole categorical column is
        # the data-prep first line of defense, but it misses the rare-category
        # case where ONE OHE indicator carries strong target signal but the
        # whole column's Cramér's V stays below the 0.5 threshold.
        #
        # Default OFF (``model_trainer_layer3_ablation_enabled=False``) —
        # joint-model retrain cost is O(n_encoded) × O(n_perms). Advisory
        # mode mirrors §4 T2.2 / T2.3: emits signals on
        # ``validation_metrics`` but does NOT mutate ``success_criteria_met``.
        #
        # Tuning knobs (all read from ``state`` with safe defaults):
        #   * model_trainer_layer3_ablation_enabled (bool, default False)
        #   * model_trainer_ablation_n_permutations (int, default 30)
        #   * model_trainer_ablation_z_threshold (float, default 5.0)
        #   * model_trainer_ablation_max_features (int, default 100)
        #   * model_trainer_ablation_strong_effect_threshold (float, default 0.30)
        #   * model_trainer_ablation_delta_auc_floor (float, default 0.10)
        #   * model_trainer_ablation_model_factory (callable, default None →
        #     LogisticRegression). Pass a tree-based factory to detect
        #     interaction-only leaks the linear baseline cannot learn.
        if bool(state.get("model_trainer_layer3_ablation_enabled", False)):
            _ablation_perms_raw = state.get("model_trainer_ablation_n_permutations")
            ablation_perms = (
                int(_ablation_perms_raw)
                if _ablation_perms_raw is not None
                else DEFAULT_MODEL_EVAL_ABLATION_PERMUTATIONS
            )
            # Codex LOW-1: guard against n_permutations < 1 which would
            # silently produce empty null distributions via
            # ``adversarial_leakage.py:87`` / :206 — the inner for-loop
            # would not execute, ``null_aucs`` would stay empty, and
            # severity would default to ``info`` for every feature
            # (silent miss for any real leak class). Mirror the strict
            # validation that Phase 3.3 enforces via type-coerce at the
            # read site (DEFAULT_ABLATION_PERMUTATIONS=50 with explicit
            # ``if X is not None`` guard at adaptive_validity_check.py:2624).
            if ablation_perms < 1:
                raise ValueError(
                    f"model_trainer_ablation_n_permutations must be >= 1; "
                    f"got {ablation_perms}. n_permutations < 1 would produce "
                    f"empty null distributions and silently degrade every "
                    f"feature's severity to info."
                )
            _ablation_z_raw = state.get("model_trainer_ablation_z_threshold")
            ablation_z = (
                float(_ablation_z_raw)
                if _ablation_z_raw is not None
                else MODEL_EVAL_ABLATION_HIGH_Z
            )
            _ablation_max_raw = state.get("model_trainer_ablation_max_features")
            ablation_max = (
                int(_ablation_max_raw)
                if _ablation_max_raw is not None
                else DEFAULT_MODEL_EVAL_ABLATION_MAX_FEATURES
            )
            _ablation_strong_raw = state.get("model_trainer_ablation_strong_effect_threshold")
            ablation_strong = (
                float(_ablation_strong_raw)
                if _ablation_strong_raw is not None
                else MODEL_EVAL_ABLATION_STRONG_EFFECT_DEFAULT
            )
            _ablation_floor_raw = state.get("model_trainer_ablation_delta_auc_floor")
            ablation_floor = (
                float(_ablation_floor_raw)
                if _ablation_floor_raw is not None
                else MODEL_EVAL_ABLATION_DELTA_AUC_FLOOR_DEFAULT
            )
            ablation_factory = state.get("model_trainer_ablation_model_factory")
            ablation_seed = int(state.get("model_trainer_ablation_seed", 42))
            _ablation_perm_n_raw = state.get("model_trainer_ablation_permutation_n_permutations")
            ablation_perm_n = (
                int(_ablation_perm_n_raw)
                if _ablation_perm_n_raw is not None
                else DEFAULT_MODEL_EVAL_PERMUTATION_PERMS
            )
            # Codex LOW-1: same guard for the label-shuffle perm n.
            if ablation_perm_n < 1:
                raise ValueError(
                    f"model_trainer_ablation_permutation_n_permutations must "
                    f"be >= 1; got {ablation_perm_n}. n_permutations < 1 "
                    f"produces empty null distributions and silently degrades "
                    f"every feature's permutation severity to info."
                )

            # Recover encoded feature names. _wrap_with_feature_names already
            # produces a DataFrame when names are available — use them
            # directly. If X_test_np is still a numpy array here, names were
            # unavailable / mismatched, and run_model_eval_ablation will
            # cleanly skip with a logged reason.
            try:
                import pandas as pd
            except ImportError:
                pd = None  # type: ignore[assignment]
            if pd is not None and isinstance(X_test_np, pd.DataFrame):
                encoded_names = list(X_test_np.columns)
            else:
                preprocessor = state.get("preprocessor")
                encoded_names = None
                if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                    try:
                        encoded_names = list(preprocessor.get_feature_names_out())
                    except Exception:
                        encoded_names = None

            # Phase 3.4 ablation runs on the TRAIN split, not the test
            # split. Rationale: ``compute_feature_ablation`` internally
            # retrains the joint model and measures |delta_AUC| against
            # a permutation null built from re-fits with shuffled feature
            # columns. The signal we're looking for (a feature whose
            # removal collapses the joint AUC) is most detectable where
            # the model has the most data — the training split. Running
            # on the test split would (a) under-power the per-fold
            # retraining (test is usually 10-20% of full data), and
            # (b) conflate test-set predictive power with feature
            # importance under the alternative null. This mirrors Phase
            # 3.3 which runs ablation on ``train_df`` at
            # ``adaptive_validity_check.py:2745`` (the canonical
            # data-prep call).
            logger.info(
                "Running model_trainer Phase 3.4 ablation on TRAIN split "
                "(n_perms=%d, encoded_features=%s)...",
                ablation_perms,
                "unknown" if encoded_names is None else str(len(encoded_names)),
            )
            ablation_result = run_model_eval_ablation(
                X_train_np,
                y_train_np,
                feature_names=encoded_names,
                n_permutations=ablation_perms,
                permutation_n_permutations=ablation_perm_n,
                seed=ablation_seed,
                z_threshold=ablation_z,
                max_features=ablation_max,
                strong_effect_threshold=ablation_strong,
                delta_auc_floor=ablation_floor,
                moderate_z_threshold=MODEL_EVAL_ABLATION_MODERATE_Z,
                model_factory=ablation_factory,
            )
            metrics_result["model_eval_ablation"] = ablation_result
            # Promote a compact summary onto validation_metrics so audit
            # readers can grep without unpacking the nested per_feature
            # list. Schema-uniform: when ablation_result is None or
            # ran=False, the summary keys carry None / empty list.
            validation_metrics = metrics_result.get("validation_metrics") or {}
            if isinstance(validation_metrics, dict):
                if ablation_result is None:
                    validation_metrics["model_eval_ablation_ran"] = False
                    validation_metrics["model_eval_ablation_flagged_features"] = []
                    validation_metrics["model_eval_ablation_skipped_reason"] = (
                        "ablation pass returned None (X_train or y_train unavailable; "
                        "the model-eval Phase 3.4 hook runs on the TRAIN split, not test)"
                    )
                else:
                    validation_metrics["model_eval_ablation_ran"] = bool(
                        ablation_result.get("ran", False)
                    )
                    validation_metrics["model_eval_ablation_flagged_features"] = list(
                        ablation_result.get("flagged_features", []) or []
                    )
                    validation_metrics["model_eval_ablation_skipped_reason"] = ablation_result.get(
                        "skipped_reason"
                    )
                metrics_result["validation_metrics"] = validation_metrics
            if ablation_result is not None and ablation_result.get("ran"):
                logger.info(
                    "model_eval_ablation: scored %d encoded features, "
                    "flagged %d (severity >= moderate)",
                    len(ablation_result.get("per_feature", []) or []),
                    len(ablation_result.get("flagged_features", []) or []),
                )

        # 2. Calibration analysis — ECE + calibration curve
        calibration_result = compute_calibration_analysis(y_test_np, y_test_proba)
        metrics_result["calibration_error"] = calibration_result.get("calibration_ece")
        metrics_result["calibration_analysis"] = calibration_result

        # 3. F1-optimal threshold (complements existing Youden's J)
        f1_threshold_result = optimize_threshold_f1(y_test_np, y_test_proba)
        metrics_result["f1_threshold_analysis"] = f1_threshold_result

        # 4. MCC from test metrics
        metrics_result["mcc"] = metrics_result.get("test_metrics", {}).get("mcc")

        # 5. Stratified k-fold CV — validate metric stability
        if X_train_np is not None and y_train_np is not None:
            logger.info("Running 5-fold stratified cross-validation...")
            # Concatenate while preserving feature names. If all inputs
            # are DataFrames (the post-#13 path), pd.concat keeps column
            # names so each cloned LGBM fold fit/predicts with names —
            # avoiding the 'X does not have valid feature names' warnings
            # that would re-appear if we np.vstack'd to a bare ndarray.
            try:
                import pandas as pd
            except ImportError:
                pd = None  # type: ignore[assignment]
            xs = [x for x in [X_train_np, X_val_np, X_test_np] if x is not None]
            arrays_y = [y for y in [y_train_np, y_val_np, y_test_np] if y is not None]
            if pd is not None and all(isinstance(x, pd.DataFrame) for x in xs):
                X_all = pd.concat(xs, axis=0, ignore_index=True)
            else:
                X_all = np.vstack([x.to_numpy() if hasattr(x, "to_numpy") else x for x in xs])
            y_all = np.concatenate(arrays_y)
            # Thread the per-fold seed (Day-3 W3-lite) so repeated_k10 nested-CV
            # draws diverge across folds instead of re-using random_state=42.
            from src.agents.ml_foundation.model_trainer.random_state import (
                resolve_fold_random_state,
            )

            cv_result = compute_stratified_cv(
                trained_model,
                X_all,
                y_all,
                n_folds=5,
                random_state=resolve_fold_random_state(state),
            )
            metrics_result["cv_results"] = cv_result
            if cv_result.get("cv_completed"):
                _promote_cv_summary_to_validation_metrics(metrics_result, cv_result)
                logger.info(
                    f"CV results: AUC={cv_result.get('cv_roc_auc_mean', 0):.4f}"
                    f"±{cv_result.get('cv_roc_auc_std', 0):.4f}, "
                    f"PR-AUC={cv_result.get('cv_pr_auc_mean', 0):.4f}"
                    f"±{cv_result.get('cv_pr_auc_std', 0):.4f}"
                )

        # 6. Post-hoc calibration — better probability estimates.
        # #633: when calibration is applied below, ``deployed_model`` is
        # promoted to the calibrated estimator and the v3 calibration gates
        # are judged on the DEPLOYED model's probabilities so the gate-prob
        # source matches the artifact we ship (consistent, NOT a masked gate).
        # v5 B1 (2026-05-11): default method is now "auto" (isotonic
        # vs Platt chosen at runtime from val-set positive count) via
        # ``apply_post_hoc_calibration``. ``state["calibration_method"]``
        # overrides the default.
        # Phase 1 W2 day-2 (shard 19 §A.7): calibration-native algorithms
        # (NGBoost, MAPIE-conformal) ship pre-calibrated predict_proba; layering
        # post-hoc calibration on top tends to over-fit small validation sets
        # and degrade test calibration (Duan et al. 2020 §4). Gate the block
        # on the `skip_post_hoc_calibration` flag propagated from the
        # model_selector registry entry. Default False preserves legacy behavior.
        model_candidate_meta = state.get("model_candidate") or {}
        skip_calibration = bool(model_candidate_meta.get("skip_post_hoc_calibration", False))
        if skip_calibration:
            metrics_result["post_hoc_calibration"] = {
                "calibration_applied": False,
                "skip_reason": "skip_post_hoc_calibration_flag",
            }
            # Cycle-8 codex IMPORTANT finding fix: the alias resolution
            # `maximum_calibration_error → calibrated_ece` (line 1759) would
            # otherwise return None and hard-fail the criterion at line 1818,
            # even though calibration-native algorithms produce a valid ECE
            # at metrics_result["calibration_error"] (line 265). Copy the
            # native ECE into both the outer metrics_result and the inner
            # test_metrics overlay so the alias resolves to the
            # calibration-native value (which IS the best-available estimate
            # for an algorithm that needs no isotonic).
            native_ece = metrics_result.get("calibration_error")
            metrics_result["calibrated_ece"] = native_ece
            inner_test_metrics = metrics_result.get("test_metrics")
            if isinstance(inner_test_metrics, dict):
                inner_test_metrics["calibrated_ece"] = (
                    native_ece if native_ece is not None else float("nan")
                )
            logger.info(
                "Skipping post-hoc calibration "
                "(skip_post_hoc_calibration=True from model_candidate); "
                "using native calibration_error as calibrated_ece alias"
            )
        elif X_val_np is not None and y_val_np is not None:
            # v5 Gate B1 (2026-05-11): use the auto-policy by default so
            # the method (isotonic vs Platt) is chosen from the val-set
            # positive count. ``state["calibration_method"]`` overrides
            # the default — accepted values: "auto" (= default), "isotonic",
            # "sigmoid" (Platt). A None value (e.g., YAML `null`) means
            # "use the default" and is silently coerced to "auto" without
            # a warning. Unknown string values warn + fall back to "auto".
            # NOTE: this is NOT an off-switch — the disable toggle lives at
            # ``model_candidate.skip_post_hoc_calibration`` which short-
            # circuits BEFORE this block. Setting calibration_method to a
            # falsey value cannot disable post-hoc calibration on its own.
            requested_method = state.get("calibration_method")
            if requested_method is None:
                # Intent-aware default. Commercial-targeting cohorts are rare-event
                # (low prevalence); the "auto" policy selects isotonic once the
                # validation positive count exceeds 100, which minimizes ECE but
                # can leave the recalibration SLOPE far from 1.0 (isotonic is a
                # non-parametric step function) and fail the deployer's
                # calibration_slope gate. Platt/sigmoid (2 params) yields a stable
                # slope ~1.0 at low N — the operating regime for commercial
                # targeting — so commercial defaults to sigmoid while clinical
                # keeps "auto". An explicit state["calibration_method"] override
                # (operator-set) is preserved unchanged.
                _cal_intent = (state.get("success_criteria") or {}).get(
                    "deployment_intent"
                ) or "clinical"
                requested_method = "sigmoid" if _cal_intent == "commercial" else "auto"
            elif requested_method not in ("auto", "isotonic", "sigmoid"):
                logger.warning(
                    "Unknown calibration_method=%r in state; falling back to 'auto'. "
                    "Use 'auto' (default), 'isotonic', or 'sigmoid' (Platt). To DISABLE "
                    "post-hoc calibration entirely, set the model_candidate's "
                    "skip_post_hoc_calibration=True flag.",
                    requested_method,
                )
                requested_method = "auto"
            calibrated_model, cal_info = apply_post_hoc_calibration(
                trained_model, X_val_np, y_val_np, method=requested_method
            )
            metrics_result["post_hoc_calibration"] = cal_info
            if cal_info.get("calibration_applied") and X_test_np is not None:
                cal_proba = calibrated_model.predict_proba(X_test_np)
                cal_proba_pos = _positive_class_proba(cal_proba)
                opt_thresh = metrics_result.get("optimal_threshold", 0.5)
                cal_pred = (cal_proba_pos >= opt_thresh).astype(int)
                cal_test_metrics = _compute_split_classification_metrics(
                    y_test_np, cal_pred, cal_proba
                )
                metrics_result["calibrated_test_metrics"] = cal_test_metrics
                # Compute ECE improvement
                cal_ece = compute_calibration_analysis(y_test_np, cal_proba)
                metrics_result["calibrated_ece"] = cal_ece.get("calibration_ece")
                # v3 B1 fix overlay: surface the post-isotonic ECE on the
                # inner ``test_metrics`` dict so the alias
                # ``maximum_calibration_error → calibrated_ece`` resolves
                # at criterion-check time. The outer ``metrics_result``
                # already carries the value at line above; this overlay
                # makes it reachable from ``_check_success_criteria``,
                # which only sees ``metrics_result["test_metrics"]``.
                inner_test_metrics = metrics_result.get("test_metrics")
                if isinstance(inner_test_metrics, dict):
                    inner_test_metrics["calibrated_ece"] = (
                        cal_ece.get("calibration_ece")
                        if cal_ece.get("calibration_ece") is not None
                        else float("nan")
                    )
                uncal_ece = metrics_result.get("calibration_error")
                if uncal_ece is not None and cal_ece.get("calibration_ece") is not None:
                    logger.info(
                        f"Calibration: ECE {uncal_ece:.4f} → "
                        f"{cal_ece['calibration_ece']:.4f} "
                        f"({cal_info.get('calibration_method_resolved', 'isotonic')})"
                    )

                # #633: DEPLOY the calibrated model. It is MLflow-logged /
                # checkpointed / returned downstream (via the deployed_model
                # state key), AND every v3 calibration gate is judged on ITS
                # probabilities so the gate-prob source matches the deployed
                # artifact. Post-hoc calibration is monotonic, so AUC/PR-AUC
                # (ranking metrics) are invariant; only the probability scale
                # and threshold-dependent metrics shift.
                deployed_model = calibrated_model
                deployed_calibration_applied = True
                if isinstance(inner_test_metrics, dict):
                    # (a) Threshold-independent calibration gates: recompute
                    #     slope / intercept on the calibrated test probs so
                    #     ``maximum_calibration_slope_deviation`` and
                    #     ``maximum_calibration_intercept_magnitude`` read the
                    #     DEPLOYED model (van Calster 2019). ECE was already
                    #     overlaid above (calibrated).
                    cal_slope, cal_intercept = _compute_calibration_slope_intercept(
                        np.asarray(y_test_np), cal_proba_pos
                    )
                    inner_test_metrics["calibration_slope"] = cal_slope
                    inner_test_metrics["calibration_intercept"] = cal_intercept
                    inner_test_metrics["calibration_slope_deviation"] = (
                        abs(cal_slope - 1.0) if not math.isnan(cal_slope) else float("nan")
                    )
                    inner_test_metrics["calibration_intercept_magnitude"] = (
                        abs(cal_intercept) if not math.isnan(cal_intercept) else float("nan")
                    )

                    # (a.2) Net-benefit gate (Vickers NB > 0 at the regime's p_t).
                    #     The NB grid is threshold-INDEPENDENT (each p_t uses its
                    #     OWN operating point ``proba >= p_t``), so like the
                    #     calibration-slope gate above it must be evaluated on the
                    #     DEPLOYED (calibrated) probabilities — NOT the raw grid
                    #     computed in ``_compute_classification_metrics``. The raw
                    #     grid is on the pre-calibration scale, where a balanced-
                    #     class-weight model's inflated probabilities flag nearly
                    #     every record at low p_t and UNDERSTATE net benefit (a
                    #     genuinely net-beneficial deployed model then false-fails
                    #     the gate). Recompute on the calibrated test probs so
                    #     ``minimum_net_benefit_at_p_t`` judges the artifact we ship
                    #     (same DEPLOYED-model-consistency contract as #633). NB is
                    #     monotone-calibration-SENSITIVE (unlike AUC), so this is a
                    #     genuine correction, not a no-op.
                    inner_test_metrics["net_benefit_grid"] = {
                        f"p_t={p_t:.2f}": _compute_net_benefit_at_p_t(
                            np.asarray(y_test_np), cal_proba_pos, p_t
                        )
                        for p_t in _V3_NB_GRID_P_T_VALUES
                    }

                    # (b) Threshold-dependent metrics: re-derive the operating
                    #     point on the calibrated probability scale (the raw
                    #     threshold lives on the raw prob scale and is not
                    #     transferable post-monotonic-remap), so the only thing
                    #     that changes is the prob SOURCE (calibrated):
                    #       * imbalance_detected → primary metrics are reported
                    #         at the validation-frozen threshold, re-derived on
                    #         the calibrated VAL probs via ``_select_threshold``.
                    #       * balanced → raw primary metrics are at 0.5, so keep
                    #         0.5 on the calibrated probs.
                    #     Ranking metrics (roc_auc / pr_auc) are calibration-
                    #     invariant and left untouched.
                    #
                    #     Findings #6 — provenance honesty: ``_select_threshold``
                    #     reproduces ONLY the Youden / cost-optimal arms of the
                    #     raw policy. It does NOT reproduce the raw path's
                    #     F1-FALLBACK (engaged when validation MCC < 0.20; see
                    #     ``_F1_FALLBACK_MCC_THRESHOLD`` in
                    #     ``_compute_classification_metrics``). So the deployed-
                    #     calibrated operating point can differ from the raw one
                    #     when the raw path used ``validation_f1_fallback``. We
                    #     therefore record the ACTUAL source string returned by
                    #     ``_select_threshold`` (one of ``"validation"`` /
                    #     ``"validation_cost_optimal"`` / ``"default"``) onto the
                    #     overlaid metrics rather than implying it mirrors the raw
                    #     ``chosen_threshold_source``. Re-running the full
                    #     F1-fallback policy here was deliberately NOT done: that
                    #     logic is inline in the raw helper and duplicating it on
                    #     the calibrated scale risks subtly diverging from the
                    #     well-exercised raw path (DO NO HARM on a LOW finding).
                    cal_threshold_source = "default"
                    if imbalance_detected:
                        cal_threshold = float(opt_thresh)
                        if X_val_np is not None and y_val_np is not None:
                            try:
                                cal_val_proba = calibrated_model.predict_proba(X_val_np)
                                cal_threshold, cal_threshold_source = _select_threshold(
                                    y_val_np, cal_val_proba, cost_matrix=None
                                )
                            except Exception as exc:  # pragma: no cover - defensive
                                logger.warning(
                                    "Deployed-calibrated threshold re-derivation failed "
                                    "(%s); falling back to the raw-derived threshold %.4f.",
                                    exc,
                                    cal_threshold,
                                )
                    else:
                        cal_threshold = 0.5
                    cal_pred_at_thresh = (cal_proba_pos >= cal_threshold).astype(int)
                    deployed_test_metrics = _compute_split_classification_metrics(
                        y_test_np, cal_pred_at_thresh, cal_proba
                    )
                    # Overlay only the threshold-dependent + calibration
                    # outputs the v3 gates / artifact read; keep the
                    # calibration-invariant ranking metrics (roc_auc, pr_auc,
                    # baseline lift, train_val_delta, NB grid) from the raw
                    # computation so monotonic invariants stay exact.
                    for _k in (
                        "accuracy",
                        "precision",
                        "recall",
                        "f1_score",
                        "mcc",
                        "brier_score",
                        "confusion_matrix",
                    ):
                        if _k in deployed_test_metrics:
                            inner_test_metrics[_k] = deployed_test_metrics[_k]
                    inner_test_metrics["deployed_model_is_calibrated"] = True
                    # Only the imbalanced path uses the re-derived threshold as
                    # the operating point; in the balanced path 0.5 is just the
                    # raw reporting threshold, so leave the diagnostic
                    # ``optimal_threshold`` (the Youden optimum) untouched.
                    # Findings #6: record the ACTUAL provenance of the deployed-
                    # calibrated operating point (Youden / cost-optimal only —
                    # never ``validation_f1_fallback``) so consumers don't read
                    # it as mirroring the raw ``chosen_threshold_source``.
                    if imbalance_detected:
                        inner_test_metrics["chosen_threshold"] = cal_threshold
                        inner_test_metrics["chosen_threshold_source"] = cal_threshold_source
                        metrics_result["optimal_threshold"] = cal_threshold
                    logger.info(
                        "#633 deploying calibrated model: slope_deviation %.4f, "
                        "intercept_magnitude %.4f, threshold %.4f (gates now read "
                        "the deployed model's probabilities).",
                        inner_test_metrics["calibration_slope_deviation"],
                        inner_test_metrics["calibration_intercept_magnitude"],
                        cal_threshold,
                    )

        # 7. Stratified split validation — check class ratio preservation
        if y_train_np is not None and y_val_np is not None:
            split_val = validate_stratified_splits(y_train_np, y_val_np, y_test_np)
            metrics_result["split_validation"] = split_val
            if split_val.get("stratification_warning"):
                logger.warning(split_val["stratification_warning"])

    # v3 (adaptive criteria follow-up): apply the overlay HERE so the
    # overlaid dict can be persisted into ``state["success_criteria"]``
    # downstream (Edit 3 returns it; agent.py extracts at hop 2; runner
    # + pipeline copy at hops 3 and 4). The overlay is idempotent: when
    # ``_adaptive_inputs`` or ``baseline_test_auc`` are absent (fixed
    # mode), it returns ``success_criteria`` unchanged.
    success_criteria = _apply_adaptive_criteria_overlay(
        success_criteria,
        metrics_result["test_metrics"],
        # Issue #866: thread the materialized split sizes so the overfit/
        # calibration caps reflect the noise floor of the splits the gated
        # metrics are measured on (delta: train vs val; calibration: test).
        n_train=int(len(y_train_np)) if y_train_np is not None else None,
        n_val=int(len(y_val_np)) if y_val_np is not None else None,
        n_test=int(len(y_test_np)) if y_test_np is not None else None,
    )

    # Check success criteria
    success_results = _check_success_criteria(
        metrics_result["test_metrics"],
        success_criteria,
        problem_type,
    )

    logger.info(
        f"Evaluation complete: success_criteria_met={success_results['success_criteria_met']}"
    )

    # Post-training leakage suspicion check (imbalance-aware)
    class_distribution = state.get("class_distribution")
    suspicion_result = check_imbalance_aware_suspicion(
        metrics_result, class_distribution, problem_type
    )

    if suspicion_result["leakage_suspected"]:
        logger.warning(
            f"LEAKAGE SUSPECTED: level={suspicion_result['suspicion_level']}, "
            f"reasons={suspicion_result['suspicion_reasons']}"
        )

    # Merge results — include the (possibly-overlaid) success_criteria
    # so the v3 active gates / regime overrides / deprecation pops
    # persist into LangGraph node state, then up through agent.run, then
    # into the runner's state dict and PipelineResult.success_criteria
    # (see hops 2-4 in adaptive_criteria_v3_followup plan).
    return {
        **metrics_result,
        **success_results,
        **suspicion_result,
        "success_criteria": success_criteria,
        # #633: the DEPLOYED artifact for downstream log/checkpoint/return.
        # Equals the calibrated estimator when post-hoc calibration was
        # applied, else the raw trained_model.
        "deployed_model": deployed_model,
        "calibration_applied": deployed_calibration_applied,
    }


def _is_causal_model(model: Any) -> bool:
    """Check if model is an EconML causal model.

    Args:
        model: Model instance

    Returns:
        True if model is a causal model (has .effect() but no .predict())
    """
    has_effect = hasattr(model, "effect") or hasattr(model, "const_marginal_effect")
    has_predict = hasattr(model, "predict")
    return has_effect and not has_predict


def _make_predictions(
    model: Any,
    X_train: Optional[np.ndarray],
    X_val: Optional[np.ndarray],
    X_test: np.ndarray,
    problem_type: str,
) -> Dict[str, Any]:
    """Make predictions on all data splits.

    Args:
        model: Trained model
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        problem_type: Problem type

    Returns:
        Dictionary with predictions and probabilities
    """
    is_classification = problem_type in [
        "binary_classification",
        "multiclass_classification",
    ]

    # Check if this is a causal model (EconML)
    if _is_causal_model(model):
        return _make_causal_predictions(model, X_train, X_val, X_test)

    has_proba = hasattr(model, "predict_proba")

    predictions = {}

    # Training set
    if X_train is not None:
        predictions["y_train_pred"] = model.predict(X_train)
        if is_classification and has_proba:
            predictions["y_train_proba"] = model.predict_proba(X_train)
        else:
            predictions["y_train_proba"] = None
    else:
        predictions["y_train_pred"] = None
        predictions["y_train_proba"] = None

    # Validation set
    if X_val is not None:
        predictions["y_val_pred"] = model.predict(X_val)
        if is_classification and has_proba:
            predictions["y_val_proba"] = model.predict_proba(X_val)
        else:
            predictions["y_val_proba"] = None
    else:
        predictions["y_val_pred"] = None
        predictions["y_val_proba"] = None

    # Test set (FINAL)
    predictions["y_test_pred"] = model.predict(X_test)
    if is_classification and has_proba:
        predictions["y_test_proba"] = model.predict_proba(X_test)
    else:
        predictions["y_test_proba"] = None

    return predictions


def _make_causal_predictions(
    model: Any,
    X_train: Optional[np.ndarray],
    X_val: Optional[np.ndarray],
    X_test: np.ndarray,
) -> Dict[str, Any]:
    """Make predictions for causal models (EconML).

    Causal models estimate treatment effects, not outcomes. For evaluation purposes,
    we use the CATE (Conditional Average Treatment Effect) as the "prediction".
    This allows downstream evaluation to compute metrics on effect heterogeneity.

    Args:
        model: EconML causal model
        X_train: Training features
        X_val: Validation features
        X_test: Test features

    Returns:
        Dictionary with CATE estimates as predictions (no probabilities)
    """
    logger.info("Using causal model prediction: effect() instead of predict()")

    predictions = {}

    # Determine which effect method to use
    if hasattr(model, "const_marginal_effect"):
        effect_fn = model.const_marginal_effect
    elif hasattr(model, "effect"):
        effect_fn = lambda X: model.effect(X)  # noqa: E731
    else:
        raise ValueError("Causal model has no effect() or const_marginal_effect() method")

    # Training set
    if X_train is not None:
        try:
            cate_train = effect_fn(X_train)
            # Convert CATE to binary predictions (positive effect = 1, negative = 0)
            predictions["y_train_pred"] = (cate_train.flatten() > 0).astype(int)
            # Use CATE values as "probabilities" (normalized to 0-1 for metrics)
            cate_norm = _normalize_cate(cate_train.flatten())
            predictions["y_train_proba"] = np.column_stack([1 - cate_norm, cate_norm])
        except Exception as e:
            logger.warning(f"Failed to compute CATE for training set: {e}")
            predictions["y_train_pred"] = None
            predictions["y_train_proba"] = None
    else:
        predictions["y_train_pred"] = None
        predictions["y_train_proba"] = None

    # Validation set
    if X_val is not None:
        try:
            cate_val = effect_fn(X_val)
            predictions["y_val_pred"] = (cate_val.flatten() > 0).astype(int)
            cate_norm = _normalize_cate(cate_val.flatten())
            predictions["y_val_proba"] = np.column_stack([1 - cate_norm, cate_norm])
        except Exception as e:
            logger.warning(f"Failed to compute CATE for validation set: {e}")
            predictions["y_val_pred"] = None
            predictions["y_val_proba"] = None
    else:
        predictions["y_val_pred"] = None
        predictions["y_val_proba"] = None

    # Test set (FINAL)
    try:
        cate_test = effect_fn(X_test)
        predictions["y_test_pred"] = (cate_test.flatten() > 0).astype(int)
        cate_norm = _normalize_cate(cate_test.flatten())
        predictions["y_test_proba"] = np.column_stack([1 - cate_norm, cate_norm])
    except Exception as e:
        logger.warning(f"Failed to compute CATE for test set: {e}")
        # Fall back to random predictions for evaluation to proceed
        predictions["y_test_pred"] = np.zeros(len(X_test), dtype=int)
        predictions["y_test_proba"] = np.column_stack(
            [np.ones(len(X_test)) * 0.5, np.ones(len(X_test)) * 0.5]
        )

    return predictions


def _normalize_cate(cate: np.ndarray) -> np.ndarray:
    """Normalize CATE values to 0-1 range for use as pseudo-probabilities.

    Args:
        cate: Raw CATE values

    Returns:
        Normalized values in [0, 1]
    """
    cate_min = cate.min()
    cate_max = cate.max()
    if cate_max - cate_min > 1e-8:
        return (cate - cate_min) / (cate_max - cate_min)  # type: ignore[no-any-return]
    else:
        return np.ones_like(cate) * 0.5


# v3 NB grid p_t values (Vickers 2019). The validator-set ``_adaptive_p_t``
# audit field selects the regime's p_t at criterion-check time; emitting a
# grid keeps the cost bounded (6 floats) and lets downstream tools plot
# decision-curve analyses at any p_t without re-evaluating.
_V3_NB_GRID_P_T_VALUES: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def _compute_calibration_slope_intercept(
    y_true: np.ndarray, y_proba: np.ndarray
) -> Tuple[float, float]:
    """Logistic recalibration. Returns ``(slope, intercept)`` per van Calster 2019.

    Fits ``LogisticRegression(C=1e10)`` on the logit of predicted positive-
    class probabilities. Slope == 1 and intercept == 0 mean perfect
    calibration; slope < 1 means over-confident, slope > 1 under-confident.

    Stability guard: requires ``n_pos >= 30`` AND ``n_neg >= 30``. Returns
    ``(nan, nan)`` if either count is too low (the LR fit is unstable
    below that threshold). Adverse-regime synthetic runs (``n_pos = 18`` at
    ``N=900, prev=0.02``) hit this guard, and the evaluator's NaN-guard
    branch records ``met=None`` for the calibration_*_deviation criteria.
    """
    if y_proba.ndim != 1:
        raise ValueError(
            f"y_proba must be 1d (positive-class probabilities); got shape {y_proba.shape}"
        )
    eps = 1e-9
    y_proba_clipped = np.clip(y_proba, eps, 1.0 - eps)
    logits = np.log(y_proba_clipped / (1.0 - y_proba_clipped)).reshape(-1, 1)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos < 30 or n_neg < 30:
        return (float("nan"), float("nan"))
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(logits, y_true.astype(int))
    return (float(lr.coef_[0, 0]), float(lr.intercept_[0]))


def _compute_net_benefit_at_p_t(y_true: np.ndarray, y_proba: np.ndarray, p_t: float) -> float:
    """Vickers 2006 net benefit at threshold probability ``p_t``.

    ``NB = TP/n - (FP/n) * p_t / (1 - p_t)``. Operating point: predict
    positive when ``y_proba >= p_t``. Returns ``nan`` for ``p_t`` outside
    ``(0, 1)`` or empty inputs (NB is undefined there).

    The decision-curve gate at ``NB > 0`` is operationally equivalent to
    ``precision > p_t`` (Vickers 2006 algebra), but stating it as NB makes
    the cost-ratio assumption explicit.
    """
    if not 0.0 < p_t < 1.0:
        return float("nan")
    n = len(y_true)
    if n == 0:
        return float("nan")
    y_pred = (y_proba >= p_t).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    return (tp / n) - (fp / n) * p_t / (1.0 - p_t)


# Q-W5-2 RESOLVED 2026-05-01 — anchor-point gate fires when observed
# prevalence > 0.10. See shard 20 §A.5.5 for the geometric / empirical /
# operational reasoning behind 0.10. Symmetric with Q-W5-1 RESOLVED
# `promotion_gate_metric` form-selection (raw when prev <= 0.10, relative
# when prev > 0.10).
_PREVALENCE_GATE_THRESHOLD: float = 0.10


# Section E.2 — τ-range defaults per use_case (shard 20 §E.2).
_USE_CASE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "screening": {"tau_low": 0.01, "tau_high": 0.10},
    "diagnostic": {"tau_low": 0.05, "tau_high": 0.30},
    "treatment_decision": {"tau_low": 0.20, "tau_high": 0.50},
    "critical_action": {"tau_low": 0.30, "tau_high": 0.70},
    "generic_benchmark": {"tau_low": 0.05, "tau_high": 0.30},
}

# Per-disease overrides — Novartis US commercial-analytics portfolio
# alignment per Q-W5-6 RESOLVED 2026-05-01. Citation-strength legend
# (per row): STRONG = single guideline / RCT-derived threshold;
# MODERATE = multi-source clinical risk-stratification literature;
# WEAK = no probability-threshold guideline; requires user validation.
_DISEASE_SPECIFIC_DEFAULTS: Dict[str, Dict[str, float]] = {
    # cv_risk_10y — Leqvio (inclisiran). 2018 ACC/AHA cholesterol guideline
    # statin-initiation threshold 7.5%. STRONG.
    "cv_risk_10y": {"tau_low": 0.05, "tau_high": 0.15, "primary_tau": 0.075},
    # hf_readmission_30d — Entresto (sacubitril/valsartan). LACE-HF + CMS
    # HRRP excess-readmission threshold ~22%. MODERATE.
    "hf_readmission_30d": {"tau_low": 0.10, "tau_high": 0.30, "primary_tau": 0.20},
    # breast_cancer_recurrence — Kisqali (ribociclib). TAILORx + RxPONDER
    # Oncotype DX RS 11-25 → 10-30% 10y distant recurrence. STRONG.
    "breast_cancer_recurrence": {"tau_low": 0.10, "tau_high": 0.30, "primary_tau": 0.21},
    # ms_treatment_escalation — Kesimpta + Mayzent. NEDA-3 failure /
    # Rio Score. MODERATE.
    "ms_treatment_escalation": {"tau_low": 0.20, "tau_high": 0.40, "primary_tau": 0.30},
    # psoriasis_pasi_response — Cosentyx (secukinumab). PASI 75 at 16 weeks.
    # WEAK — no probability-threshold guideline; validate before deployment.
    "psoriasis_pasi_response": {"tau_low": 0.30, "tau_high": 0.60, "primary_tau": 0.45},
    # mcrpc_progression_risk — Pluvicto (177Lu-PSMA-617). VISION + PROfound
    # post-taxane progression rates. MODERATE.
    "mcrpc_progression_risk": {"tau_low": 0.25, "tau_high": 0.55, "primary_tau": 0.40},
}


def _compute_net_benefit_area(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    tau_grid: Sequence[float],
    prevalence_gate_threshold: float = _PREVALENCE_GATE_THRESHOLD,
) -> Dict[str, Any]:
    """Trapezoidal integral of NB over ``tau_grid`` (Vickers 2006 + shard 20 §A.2).

    Computes NB_area for the model and for the treat-all reference, plus
    their difference (the operationally meaningful quantity per Vickers
    2006). Returns a dict so callers can serialize a single block. Returns
    NaN-valued fields when ``tau_grid`` is empty / non-monotonic / contains
    values outside ``(0, 1)``. The ``net_benefit_area_form`` field is the
    informational Q-W5-1 RESOLVED 2026-05-02 cycle 6 disposition: per the
    prevalence-conditional ``promotion_gate_metric`` rule, the gating form
    is raw when ``prevalence <= prevalence_gate_threshold`` and
    relative-to-treat-all otherwise. Both fields are populated regardless;
    consumers (W6 ``criteria_validator``) read ``net_benefit_area_form`` to
    decide which to gate against.

    Args:
        y_true: 1D binary labels (0/1).
        y_proba_pos: 1D positive-class probabilities.
        tau_grid: Sorted ascending list of τ values in ``(0, 1)``. Phase 1
            default is 21 evenly-spaced points across ``[tau_low, tau_high]``.
        prevalence_gate_threshold: Cutoff for ``net_benefit_area_form``
            informational selection. Defaults to ``_PREVALENCE_GATE_THRESHOLD``
            (Q-W5-2 + Q-W5-1 RESOLVED).

    Returns:
        ``{"net_benefit_area", "net_benefit_area_treat_all",
        "net_benefit_area_relative_to_treat_all", "tau_low", "tau_high",
        "n_grid_points", "prevalence", "net_benefit_area_form"}``.
    """
    n = len(y_true)
    prev = float(np.mean(y_true == 1)) if n > 0 else float("nan")
    form: Literal["raw", "relative_to_treat_all"] = (
        "relative_to_treat_all"
        if (not math.isnan(prev) and prev > prevalence_gate_threshold)
        else "raw"
    )
    nan_block: Dict[str, Any] = {
        "net_benefit_area": float("nan"),
        "net_benefit_area_treat_all": float("nan"),
        "net_benefit_area_relative_to_treat_all": float("nan"),
        "tau_low": float("nan"),
        "tau_high": float("nan"),
        "n_grid_points": int(len(tau_grid)),
        "prevalence": prev,
        "net_benefit_area_form": form,
    }
    if len(tau_grid) < 2 or n == 0:
        return nan_block
    taus = np.asarray(tau_grid, dtype=float)
    if not np.all(np.diff(taus) > 0):
        # Non-monotonic — defensive; the resolver should sort.
        taus = np.sort(taus)
    if taus[0] <= 0.0 or taus[-1] >= 1.0:
        # NB undefined at boundaries; clamp into open interval. The
        # 1e-6 buffer matches the resolver's invariant; without it a
        # caller-supplied 0.0 would leak NaN through the NB grid.
        taus = np.clip(taus, 1e-6, 1.0 - 1e-6)
    nb_model = np.array([_compute_net_benefit_at_p_t(y_true, y_proba_pos, float(t)) for t in taus])
    nb_treat_all = prev - (1.0 - prev) * taus / (1.0 - taus)
    if np.any(np.isnan(nb_model)):
        # Don't integrate over partial NaN — emit NaN for the area.
        area_model = float("nan")
    else:
        area_model = float(_trapezoid(nb_model, taus))
    area_treat_all = float(_trapezoid(nb_treat_all, taus))
    area_relative = area_model - area_treat_all if not math.isnan(area_model) else float("nan")
    return {
        "net_benefit_area": area_model,
        "net_benefit_area_treat_all": area_treat_all,
        "net_benefit_area_relative_to_treat_all": area_relative,
        "tau_low": float(taus[0]),
        "tau_high": float(taus[-1]),
        "n_grid_points": int(len(taus)),
        "prevalence": prev,
        "net_benefit_area_form": form,
    }


def _compute_dca_curves(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    tau_grid: Sequence[float],
) -> Dict[str, Any]:
    """Decision-curve analysis artifact (Vickers 2006; shard 20 §B.2).

    Returns aligned arrays so downstream consumers (MLflow, Opik, Grafana)
    can plot decision curves without recomputing. Emits Python lists, not
    numpy arrays, so the dict is JSON-serializable for ``mlflow.log_dict``.

    Args:
        y_true: 1D binary labels (0/1).
        y_proba_pos: 1D positive-class probabilities.
        tau_grid: Sorted ascending list of τ values in ``(0, 1)``.

    Returns:
        Dict with ``tau_grid``, ``nb_model``, ``nb_treat_all``,
        ``nb_treat_none`` arrays of length K, plus ``prevalence`` and
        ``tau_low`` / ``tau_high`` bounds.
    """
    taus = np.asarray(tau_grid, dtype=float) if len(tau_grid) > 0 else np.array([])
    prev = float(np.mean(y_true == 1)) if len(y_true) > 0 else float("nan")
    if len(taus) == 0:
        return {
            "tau_grid": [],
            "nb_model": [],
            "nb_treat_all": [],
            "nb_treat_none": [],
            "n_grid_points": 0,
            "prevalence": prev,
            "tau_low": float("nan"),
            "tau_high": float("nan"),
        }
    nb_model = [_compute_net_benefit_at_p_t(y_true, y_proba_pos, float(t)) for t in taus]
    nb_treat_all_arr = prev - (1.0 - prev) * taus / (1.0 - taus)
    nb_treat_all = [float(v) for v in nb_treat_all_arr]
    nb_treat_none = [0.0] * len(taus)
    return {
        "tau_grid": [float(t) for t in taus],
        "nb_model": [float(v) for v in nb_model],
        "nb_treat_all": nb_treat_all,
        "nb_treat_none": nb_treat_none,
        "n_grid_points": int(len(taus)),
        "prevalence": prev,
        "tau_low": float(taus[0]),
        "tau_high": float(taus[-1]),
    }


def _resolve_tau_grid_for_metrics(
    success_criteria: Optional[Dict[str, Any]],
    legacy_grid: Sequence[float],
    n_grid_points: int = 21,
) -> Optional[List[float]]:
    """Resolve τ_low, τ_high, and a uniform grid from ``success_criteria``.

    Resolution order per shard 16 §3 + shard 20 §E.2 (disease label is more
    specific than ``use_case=custom``, so it beats it):

      1. ``clinical_threshold_range.evaluation_grid`` (caller-supplied) wins outright.
      2. ``dataset_disease`` in ``_DISEASE_SPECIFIC_DEFAULTS`` → disease-specific bounds.
         The disease label may live at the top level of ``success_criteria``
         OR nested under ``clinical_threshold_range`` — both keys are checked.
      3. ``clinical_threshold_range.use_case == "custom"`` → explicit ``tau_low`` / ``tau_high``.
      4. ``clinical_threshold_range.use_case`` in ``_USE_CASE_DEFAULTS`` → use-case defaults.
      5. No schema or unknown use_case → ``legacy_grid``.

    Returns the resolved grid (a list of floats), or ``None`` only when the
    caller signalled they want the metrics suite skipped entirely (currently
    not used — kept in the signature for forward compatibility).
    """
    if not success_criteria:
        return list(legacy_grid)
    ctr = success_criteria.get("clinical_threshold_range") or {}
    use_case = (ctr.get("use_case") or "").strip()
    explicit_grid = ctr.get("evaluation_grid")
    if explicit_grid:  # caller-supplied grid wins outright
        return [float(t) for t in explicit_grid]

    # Disease-specific override (when dataset is labeled). Disease beats
    # use_case so a Novartis franchise default cannot be silently
    # overridden by a stale ``use_case=screening`` config — the explicit
    # disease label always wins.
    dataset_disease = (
        success_criteria.get("dataset_disease") or ctr.get("dataset_disease") or ""
    ).strip()
    if dataset_disease and dataset_disease in _DISEASE_SPECIFIC_DEFAULTS:
        ds = _DISEASE_SPECIFIC_DEFAULTS[dataset_disease]
        return [float(t) for t in np.linspace(ds["tau_low"], ds["tau_high"], n_grid_points)]

    if use_case == "custom":
        tl = ctr.get("tau_low")
        th = ctr.get("tau_high")
        try:
            tl_f = float(tl) if tl is not None else None
            th_f = float(th) if th is not None else None
        except (TypeError, ValueError):
            tl_f = th_f = None
        if tl_f is None or th_f is None or not 0.0 < tl_f < th_f < 1.0:
            logger.warning(
                "clinical_threshold_range.use_case=custom but tau_low/tau_high "
                "missing or invalid (%s, %s); falling back to legacy grid",
                tl,
                th,
            )
            return list(legacy_grid)
        return [float(t) for t in np.linspace(tl_f, th_f, n_grid_points)]

    if use_case in _USE_CASE_DEFAULTS:
        defaults = _USE_CASE_DEFAULTS[use_case]
        return [
            float(t) for t in np.linspace(defaults["tau_low"], defaults["tau_high"], n_grid_points)
        ]

    return list(legacy_grid)


def _resolve_primary_tau(
    success_criteria: Optional[Dict[str, Any]],
) -> Optional[float]:
    """Resolve the optional ``primary_tau`` clinical anchor (shard 20 §A.5.1).

    Resolution order:

      1. ``clinical_threshold_range.primary_tau`` (explicit caller config).
      2. ``_DISEASE_SPECIFIC_DEFAULTS[dataset_disease].primary_tau``.
      3. ``None`` — no anchor available; anchor-point metrics emit NaN.
    """
    if not success_criteria:
        return None
    ctr = success_criteria.get("clinical_threshold_range") or {}
    explicit = ctr.get("primary_tau")
    if explicit is not None:
        try:
            t = float(explicit)
            if 0.0 < t < 1.0:
                return t
        except (TypeError, ValueError):
            pass
    dataset_disease = (
        success_criteria.get("dataset_disease") or ctr.get("dataset_disease") or ""
    ).strip()
    if dataset_disease and dataset_disease in _DISEASE_SPECIFIC_DEFAULTS:
        primary = _DISEASE_SPECIFIC_DEFAULTS[dataset_disease].get("primary_tau")
        if primary is not None:
            try:
                t = float(primary)
                if 0.0 < t < 1.0:
                    return t
            except (TypeError, ValueError):
                pass
    return None


def _compute_anchor_point_metrics(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    primary_tau: Optional[float],
    nb_area_relative: float,
    prevalence_threshold: float = _PREVALENCE_GATE_THRESHOLD,
) -> Dict[str, Any]:
    """Single-τ anchor-point NB + Q-W5-2 RESOLVED secondary-gate trigger.

    Always emitted when ``primary_tau`` resolves. The boolean
    ``nb_anchor_secondary_gate_active`` is True iff observed prevalence on
    ``y_true`` exceeds ``prevalence_threshold`` — W6 wires this boolean
    into ``criteria_validator`` so the gate fires only when active.

    The ``nb_anchor_vs_area_disagree`` diagnostic is True when the anchor
    pass/fail differs from the NB-area-relative pass/fail. After the first
    multi-disease run (shard 22), inspect disagreement rates per regime: if
    < 1% the gate adds little signal; if > 5% it is catching real misses.
    """
    prev = float(np.mean(y_true == 1)) if len(y_true) > 0 else float("nan")
    nan_block: Dict[str, Any] = {
        "primary_tau": None,
        "net_benefit_at_primary_tau": float("nan"),
        "net_benefit_at_primary_tau_treat_all": float("nan"),
        "net_benefit_at_primary_tau_relative_to_treat_all": float("nan"),
        "nb_anchor_secondary_gate_active": False,
        "nb_anchor_passes": None,
        "nb_anchor_vs_area_disagree": False,
        "prevalence": prev,
    }
    if primary_tau is None or not (0.0 < float(primary_tau) < 1.0):
        return nan_block

    p_tau = float(primary_tau)
    nb_at_primary = _compute_net_benefit_at_p_t(y_true, y_proba_pos, p_tau)
    nb_treat_all_at_primary = float(prev - (1.0 - prev) * p_tau / (1.0 - p_tau))
    nb_relative = (
        float(nb_at_primary - nb_treat_all_at_primary)
        if not math.isnan(nb_at_primary)
        else float("nan")
    )
    gate_active = bool(not math.isnan(prev) and prev > prevalence_threshold)
    anchor_passes: Optional[bool]
    if gate_active and not math.isnan(nb_relative):
        anchor_passes = bool(nb_relative > 0.0)
    else:
        anchor_passes = None
    area_passes = bool(nb_area_relative > 0.0) if not math.isnan(nb_area_relative) else None
    disagree = bool(
        anchor_passes is not None and area_passes is not None and anchor_passes != area_passes
    )
    return {
        "primary_tau": p_tau,
        "net_benefit_at_primary_tau": float(nb_at_primary),
        "net_benefit_at_primary_tau_treat_all": nb_treat_all_at_primary,
        "net_benefit_at_primary_tau_relative_to_treat_all": nb_relative,
        "nb_anchor_secondary_gate_active": gate_active,
        "nb_anchor_passes": anchor_passes,
        "nb_anchor_vs_area_disagree": disagree,
        "prevalence": prev,
    }


def _compute_classification_metrics(
    y_train: Optional[np.ndarray],
    y_train_pred: Optional[np.ndarray],
    y_train_proba: Optional[np.ndarray],
    y_validation: Optional[np.ndarray],
    y_validation_pred: Optional[np.ndarray],
    y_validation_proba: Optional[np.ndarray],
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_proba: Optional[np.ndarray],
    imbalance_detected: bool = False,
    minority_ratio: float = 0.5,
    cost_matrix: Optional[Dict[str, float]] = None,
    bootstrap_random_state: Optional[int] = None,
    success_criteria: Optional[Dict[str, Any]] = None,
    use_cost_optimal_threshold: bool = False,
) -> Dict[str, Any]:
    """Compute binary classification metrics using sklearn.

    Threshold-selection policy (Block 1A — finding #6):
        The classification operating point (Youden's J optimum, and the
        precision-constrained alternative when ``minority_ratio < 0.05``)
        is selected on the VALIDATION arrays, frozen, and then applied to
        the test set without re-tuning. This prevents test-set leakage
        into the operating point. If validation arrays are not provided,
        the function falls back to the default 0.5 threshold rather than
        tuning on test — test-set integrity is preserved at the cost of
        a possibly off-by-default operating point.

        ``validation_metrics`` is recomputed AT the chosen threshold when
        validation probabilities are available, so its precision/recall/F1
        reflect performance at the same operating point used for test.

    Args:
        y_train: Training labels
        y_train_pred: Training predictions (at model's default threshold)
        y_train_proba: Training probabilities
        y_validation: Validation labels — used for threshold selection
        y_validation_pred: Validation predictions (at model's default)
        y_validation_proba: Validation probabilities — used for threshold
            selection. When None, threshold falls back to 0.5.
        y_test: Test labels
        y_test_pred: Test predictions (at model's default threshold)
        y_test_proba: Test probabilities. The chosen threshold is applied
            to these to produce the final reported test metrics.
        imbalance_detected: Whether class imbalance was detected. When
            True, ``test_metrics`` reports values at the chosen threshold;
            otherwise it reports values at the model's default 0.5.
        minority_ratio: Ratio of minority class (0-1). When below 0.05,
            the precision-constrained threshold is also evaluated on
            validation and may override the Youden's J optimum.
        cost_matrix: Optional Block-5 (#10) per-outcome dollar matrix —
            keys ``tp``/``fp``/``fn``/``tn`` mapped to float per-prediction
            value. When supplied, ``business_utility`` is computed at the
            chosen threshold for both validation and test and exposed via
            ``validation_metrics["business_utility"]`` /
            ``test_metrics["business_utility"]`` plus a top-level mirror.

    Returns:
        Dictionary with the standard top-level metrics keys
        (``train_metrics``, ``validation_metrics``, ``test_metrics``,
        ``test_metrics_at_05``, ``test_metrics_at_optimal``, ``auc_roc``,
        ``precision``, ``recall``, ``f1_score``, ``pr_auc``,
        ``brier_score``, ``confusion_matrix``, ``optimal_threshold``,
        ``precision_at_k``, ``confidence_interval``, ``bootstrap_samples``,
        ``precision_constrained``, ``calibration_error``, ``f1_macro``,
        ``f1_weighted``, plus minority metrics when imbalance is detected).

        Block 1A-specific keys:
            - ``optimal_threshold`` (top level): the canonical chosen
              threshold (validation-tuned, or 0.5 fallback). Existing
              cross-codebase consumers read this key.
            - ``chosen_threshold_source`` (top level): provenance flag,
              one of ``"validation"`` or ``"default"``.
            - ``validation_metrics["chosen_threshold"]``: same numeric
              value as ``optimal_threshold``, exposed at the validation
              metric level so model-registry / monitoring consumers can
              audit the operating point that produced the validation
              numbers.
            - ``validation_metrics["chosen_threshold_source"]``:
              provenance flag mirrored at the validation level.
    """
    # Training metrics
    train_metrics = {}
    if y_train is not None and y_train_pred is not None:
        train_metrics = _compute_split_classification_metrics(y_train, y_train_pred, y_train_proba)

    # =====================================================================
    # THRESHOLD TUNING — VALIDATION SET ONLY (Block 1A — finding #6)
    # =====================================================================
    # The optimal classification threshold (and any precision-constrained
    # alternative) MUST be selected on validation data, then frozen, then
    # applied to test. Tuning on test leaks test info into the operating
    # point and inflates apparent test performance.
    #
    # Step 1 (1A-I-3): pick the canonical validation-vs-default threshold
    # via `_select_threshold`. Step 2 (rare-event override) stays inline
    # because it needs `minority_ratio` and produces a `precision_constrained`
    # dict consumed by the result builder below.
    # =====================================================================
    # Backlog #20 Gap 1: cost-aware threshold is OPT-IN via
    # ``use_cost_optimal_threshold``. cost_matrix has historically been
    # a *reporting* signal (computes business_utility post-hoc) without
    # influencing threshold selection — the synthetic-baseline-invariant
    # test (test_synthetic_baseline_invariant.py) pinned that
    # behaviour. Only callers that explicitly opt in get the cost-aware
    # threshold; default OFF preserves backward compatibility.
    optimal_threshold, threshold_source = _select_threshold(
        y_validation,
        y_validation_proba,
        cost_matrix=cost_matrix if use_cost_optimal_threshold else None,
    )

    # For rare-event prediction, apply precision-constrained threshold
    # tuned ON VALIDATION (not test). This may override the Youden's J
    # optimum returned by `_select_threshold` above.
    precision_constrained: Optional[Dict[str, Any]] = None
    if minority_ratio < 0.05 and y_validation is not None and y_validation_proba is not None:
        precision_constrained = _compute_precision_constrained_threshold(
            y_validation, y_validation_proba
        )
        if precision_constrained and precision_constrained.get("target_achieved"):
            optimal_threshold = precision_constrained["precision_constrained_threshold"]
            logger.info(
                "Using precision-constrained threshold "
                f"{optimal_threshold:.4f} tuned on validation "
                f"(precision={precision_constrained['precision_at_threshold']:.4f}, "
                f"recall={precision_constrained['recall_at_threshold']:.4f})"
            )

    # Validation metrics: recomputed AT THE CHOSEN THRESHOLD when probas
    # are available so validation_metrics reflects performance at the same
    # frozen operating point used for test. We also persist
    # `chosen_threshold` and `chosen_threshold_source` here so downstream
    # consumers can audit which split produced the threshold.
    validation_metrics: Dict[str, Any] = {}
    if y_validation is not None and y_validation_pred is not None:
        if y_validation_proba is not None:
            y_val_proba_pos = _positive_class_proba(y_validation_proba)
            y_validation_pred_at_chosen = (y_val_proba_pos >= optimal_threshold).astype(int)
            validation_metrics = cast(
                Dict[str, Any],
                _compute_split_classification_metrics(
                    y_validation, y_validation_pred_at_chosen, y_validation_proba
                ),
            )

            # Backlog #20 Gap 2: F1-fallback when validation MCC is very low.
            # If the canonical pick (Youden's, cost-optimal, or precision-
            # constrained) leaves validation MCC < 0.20, retune via the
            # F1-optimal threshold from advanced_validation. F1 maximises
            # the precision/recall harmonic mean and tends to do better on
            # severely imbalanced data than Youden's J. We only switch when
            # the F1-optimal threshold actually IMPROVES MCC; otherwise the
            # original choice stays — no performative re-tuning.
            val_mcc_at_chosen = float(validation_metrics.get("mcc", 0.0) or 0.0)
            if val_mcc_at_chosen < _F1_FALLBACK_MCC_THRESHOLD:
                from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (
                    optimize_threshold_f1,
                )

                f1_result = optimize_threshold_f1(y_validation, y_validation_proba)
                f1_threshold = float(f1_result.get("f1_optimal_threshold", 0.5))
                f1_mcc_candidate = float(f1_result.get("mcc_at_f1_optimal", 0.0) or 0.0)
                if f1_mcc_candidate > val_mcc_at_chosen:
                    prior_source = threshold_source
                    logger.info(
                        "F1-fallback engaged: validation MCC %.4f < %.2f at "
                        "threshold %.4f (source=%s); switching to F1-optimal "
                        "threshold %.4f (MCC=%.4f).",
                        val_mcc_at_chosen,
                        _F1_FALLBACK_MCC_THRESHOLD,
                        optimal_threshold,
                        prior_source,
                        f1_threshold,
                        f1_mcc_candidate,
                    )
                    optimal_threshold = f1_threshold
                    threshold_source = "validation_f1_fallback"
                    y_validation_pred_at_chosen = (y_val_proba_pos >= optimal_threshold).astype(int)
                    validation_metrics = cast(
                        Dict[str, Any],
                        _compute_split_classification_metrics(
                            y_validation, y_validation_pred_at_chosen, y_validation_proba
                        ),
                    )
                    validation_metrics["f1_fallback_engaged"] = True
                    validation_metrics["f1_fallback_original_mcc"] = val_mcc_at_chosen
                    validation_metrics["f1_fallback_original_threshold_source"] = prior_source
                else:
                    logger.info(
                        "F1-fallback evaluated but not engaged: F1-optimal "
                        "MCC %.4f does not exceed current MCC %.4f; "
                        "keeping threshold %.4f (source=%s).",
                        f1_mcc_candidate,
                        val_mcc_at_chosen,
                        optimal_threshold,
                        threshold_source,
                    )
        else:
            validation_metrics = cast(
                Dict[str, Any],
                _compute_split_classification_metrics(
                    y_validation, y_validation_pred, y_validation_proba
                ),
            )
        # Commercial-targeting operating point (owner-ratified 2026-06-07). A
        # targeting model is used by its RANKING, so the decision threshold should
        # catch the commercial recall floor (false positives are cheap in
        # outreach). When deployment_intent is commercial AND the canonical
        # operating point falls short of the recall floor, switch to the highest
        # threshold that meets it (best precision subject to recall>=floor). This
        # is the FINAL operating-point decision so the F1-fallback above does not
        # revert it. deployment_intent is read off success_criteria (stamped by
        # scope_definer.define_success_criteria); the threshold is tuned on
        # VALIDATION and frozen onto test below — no test re-tuning.
        if y_validation_proba is not None:
            # success_criteria may be a plain dict OR the pydantic ScopeDefiner
            # success-criteria model (dict-like, exposes ``.get``). Duck-type on
            # ``.get`` rather than ``isinstance(dict)`` — the model is NOT a dict
            # subclass, so an isinstance check would silently skip the commercial
            # operating point even when deployment_intent IS commercial (the
            # calibration block already uses the duck-typed ``(... or {}).get``).
            _intent = "clinical"
            if success_criteria is not None and hasattr(success_criteria, "get"):
                _intent = (
                    success_criteria.get("deployment_intent")
                    or (success_criteria.get("_adaptive_inputs") or {}).get("deployment_intent")
                    or "clinical"
                )
            if _intent == "commercial":
                yv_pos = _positive_class_proba(y_validation_proba)
                cur_pred = (yv_pos >= optimal_threshold).astype(int)
                pos_mask = y_validation == 1
                n_pos = int(pos_mask.sum())
                cur_recall = float((cur_pred[pos_mask] == 1).sum()) / n_pos if n_pos else 0.0
                if cur_recall < _COMMERCIAL_RECALL_TARGET:
                    rc = _compute_recall_constrained_threshold(
                        y_validation, y_validation_proba, _COMMERCIAL_RECALL_TARGET
                    )
                    if rc and rc.get("target_achieved"):
                        optimal_threshold = float(rc["recall_constrained_threshold"])
                        threshold_source = "validation_commercial_recall"
                        y_val_pred_rc = (yv_pos >= optimal_threshold).astype(int)
                        validation_metrics = cast(
                            Dict[str, Any],
                            _compute_split_classification_metrics(
                                y_validation, y_val_pred_rc, y_validation_proba
                            ),
                        )
                        logger.info(
                            "Commercial recall-constrained threshold %.4f tuned on "
                            "validation (recall=%.4f, precision=%.4f) — operating "
                            "point tuned for targeting.",
                            optimal_threshold,
                            float(rc["recall_at_threshold"]),
                            float(rc["precision_at_threshold"]),
                        )

        validation_metrics["chosen_threshold"] = float(optimal_threshold)
        validation_metrics["chosen_threshold_source"] = threshold_source

    # CRITICAL: For imbalanced data, apply the FROZEN threshold tuned on
    # validation to test predictions. No re-tuning on test.
    # math.isclose tolerates the tiny float drift _compute_optimal_threshold
    # can return when its sklearn input lands exactly on the default — we
    # only want the rebinarisation pass when the chosen threshold is
    # meaningfully different from 0.5.
    y_test_pred_optimal = y_test_pred  # Default to model predictions
    if y_test_proba is not None and not math.isclose(optimal_threshold, 0.5):
        y_proba_pos = _positive_class_proba(y_test_proba)
        y_test_pred_optimal = (y_proba_pos >= optimal_threshold).astype(int)
        logger.info(
            f"Applying frozen threshold {optimal_threshold:.4f} "
            f"(tuned on {threshold_source}) to test predictions (vs default 0.5)"
        )

    # Test metrics at 0.5 threshold (standard)
    test_metrics_standard = _compute_split_classification_metrics(y_test, y_test_pred, y_test_proba)

    # Test metrics at the FROZEN chosen threshold (no re-tuning on test)
    test_metrics_optimal = _compute_split_classification_metrics(
        y_test, y_test_pred_optimal, y_test_proba
    )

    # Use optimal threshold metrics as primary when imbalance detected
    # This ensures we report useful metrics, not misleading ones
    if imbalance_detected:
        test_metrics = test_metrics_optimal
        logger.info(
            f"Using optimal threshold metrics for imbalanced data: "
            f"recall={test_metrics.get('recall', 0):.4f}, "
            f"precision={test_metrics.get('precision', 0):.4f}"
        )
    else:
        test_metrics = test_metrics_standard

    # Lift-over-baseline (Section B of pre_phase2_unblockers plan): inject a
    # stratified-dummy baseline AUC and the absolute lift the trained model
    # achieves over it. The criterion is read by ``_check_success_criteria``
    # via metric_aliases below; threshold (default 0.10) lives in
    # ``criteria_validator._define_classification_criteria``. Absolute lift
    # (auc - baseline_auc) is preferred over relative because: (a) it
    # matches the natural reading of the criterion name; (b) it is not
    # deflated when the baseline drifts above 0.50 on small/skewed splits.
    baseline_metrics = _compute_baseline_test_metrics(y_train, y_test, "binary_classification")
    if "baseline_test_auc" in baseline_metrics:
        baseline_auc = baseline_metrics["baseline_test_auc"]
        test_metrics["baseline_test_auc"] = baseline_auc
        test_auc = test_metrics.get("roc_auc")
        if test_auc is not None:
            test_metrics["minimum_lift_over_baseline"] = float(test_auc - baseline_auc)

    # v3 emits (task 05 of adaptive_success_criteria plan, Option C):
    # surface metrics the v3 active gates resolve against. All emits land
    # on the inner ``test_metrics`` (B4 fix) — the dict that reaches
    # ``_check_success_criteria`` via ``metrics_result["test_metrics"]``.
    #
    # 1. ``train_val_auc_delta`` (B2 fix): emit the gap between train and
    #    validation AUC so the ``maximum_train_val_delta`` adaptive gate
    #    has a metric. Without this, every adaptive run hard-fails the
    #    criterion via the missing-metric path. Use absolute value so the
    #    "max delta" gate fires regardless of direction.
    train_auc_value = train_metrics.get("roc_auc") if isinstance(train_metrics, dict) else None
    val_auc_value = (
        validation_metrics.get("roc_auc") if isinstance(validation_metrics, dict) else None
    )
    if train_auc_value is not None and val_auc_value is not None:
        try:
            gap = float(train_auc_value) - float(val_auc_value)
            test_metrics["train_val_auc_delta"] = abs(gap)
        except (TypeError, ValueError):
            logger.warning(
                "train_val_auc_delta could not be computed "
                f"(train roc_auc={train_auc_value!r}, val roc_auc={val_auc_value!r})"
            )

    # 2-4. Calibration slope/intercept (van Calster 2019, NEW v3) and
    #      net-benefit grid (Vickers 2006, NEW v3). Both require positive-
    #      class probabilities; skip the emit when y_test_proba is None.
    if y_test_proba is not None:
        y_test_proba_pos = _positive_class_proba(y_test_proba)
        slope, intercept = _compute_calibration_slope_intercept(
            np.asarray(y_test), y_test_proba_pos
        )
        test_metrics["calibration_slope"] = slope
        test_metrics["calibration_intercept"] = intercept
        test_metrics["calibration_slope_deviation"] = (
            abs(slope - 1.0) if not math.isnan(slope) else float("nan")
        )
        test_metrics["calibration_intercept_magnitude"] = (
            abs(intercept) if not math.isnan(intercept) else float("nan")
        )
        # NB grid keyed on ``p_t={p_t:.2f}`` strings; the evaluator's
        # ``_resolve_net_benefit_from_grid`` reads the regime's ``p_t``
        # from ``success_criteria['_adaptive_p_t']`` and looks up the
        # matching key. Emit the full grid for downstream DCA plotting.
        # Cast through Any to satisfy mypy — ``test_metrics`` is typed
        # ``Dict[str, Optional[float]]`` from ``_compute_split_classification_metrics``,
        # but the v3 emits add a dict-valued ``net_benefit_grid``.
        test_metrics_any: Dict[str, Any] = test_metrics  # type: ignore[assignment]
        test_metrics_any["net_benefit_grid"] = {
            f"p_t={p_t:.2f}": _compute_net_benefit_at_p_t(np.asarray(y_test), y_test_proba_pos, p_t)
            for p_t in _V3_NB_GRID_P_T_VALUES
        }

        # W1 day-2 — NB-area + DCA + anchor-point emissions per shard 20
        # §A.4 / §B / §A.5.3. Gated by ``clinical_threshold_range``
        # presence (NOT truthiness) in ``success_criteria``: when the
        # key is absent, only the legacy ``net_benefit_grid`` above is
        # emitted (single-mode parity per shard 20 §G acceptance #7).
        # An empty dict still triggers the gate — caller signalled they
        # want the metrics suite but the resolver will fall back to
        # ``_V3_NB_GRID_P_T_VALUES`` and emit the COSMETIC-2 INFO.
        ctr_present = bool(
            success_criteria is not None and "clinical_threshold_range" in success_criteria
        )
        if ctr_present:
            tau_grid_resolved = _resolve_tau_grid_for_metrics(
                success_criteria,
                legacy_grid=_V3_NB_GRID_P_T_VALUES,
            )
            if tau_grid_resolved is not None and len(tau_grid_resolved) >= 2:
                # Cycle-20 IMPORTANT-1 (Q1.B): treat-all NB blows up as
                # τ → 1 (denominator (1−τ) → 0). At τ_high ≥ 0.80 the
                # treat-all integrand exceeds −15·(1−prev), letting any
                # model dominate it on volume alone. WARN so callers can
                # validate intentionality. Cycle-21 C-1: include 0.80
                # itself in the WARNING band — 1·(1−prev)·(0.80/0.20) = 4
                # already amplifies treat-all tail noise.
                if tau_grid_resolved[-1] >= 0.80:
                    logger.warning(
                        "clinical_threshold_range emits tau_high=%.4f > 0.80 — "
                        "treat-all NB diverges near τ→1; net_benefit_area_relative_to_treat_all "
                        "is dominated by the integration tail rather than model quality. "
                        "Verify use_case=custom upper bound is intentional.",
                        tau_grid_resolved[-1],
                    )
                # Cycle-20 COSMETIC-2: a K=6 legacy grid yields a
                # ~1e-2 trapezoidal residual per shard 20 §E.3 — well
                # above the bootstrap noise floor. Emit a one-time INFO
                # so downstream consumers know to interpret NB-area on
                # legacy fallback as approximate.
                if len(tau_grid_resolved) == len(_V3_NB_GRID_P_T_VALUES) and all(
                    math.isclose(a, b)
                    for a, b in zip(tau_grid_resolved, _V3_NB_GRID_P_T_VALUES, strict=True)
                ):
                    logger.info(
                        "NB-area computed on K=6 legacy grid; trapezoidal "
                        "residual ~1e-2 per shard 20 §E.3. Treat values "
                        "as approximate — supply clinical_threshold_range "
                        "with a denser grid for higher precision."
                    )
                nb_area_block = _compute_net_benefit_area(
                    np.asarray(y_test),
                    y_test_proba_pos,
                    tau_grid_resolved,
                )
                test_metrics_any.update(nb_area_block)
                test_metrics_any["decision_curve_data"] = _compute_dca_curves(
                    np.asarray(y_test),
                    y_test_proba_pos,
                    tau_grid_resolved,
                )
                # Anchor-point + secondary-gate trigger (Q-W5-2 RESOLVED).
                primary_tau_resolved = _resolve_primary_tau(success_criteria)
                anchor_block = _compute_anchor_point_metrics(
                    np.asarray(y_test),
                    y_test_proba_pos,
                    primary_tau=primary_tau_resolved,
                    nb_area_relative=nb_area_block["net_benefit_area_relative_to_treat_all"],
                )
                test_metrics_any.update(anchor_block)

    # Extract primary metrics for state
    auc_roc = test_metrics.get("roc_auc")
    precision = test_metrics.get("precision")
    recall = test_metrics.get("recall")
    f1 = test_metrics.get("f1_score")
    pr_auc = test_metrics.get("pr_auc")
    brier = test_metrics.get("brier_score")

    # Findings #5: the headline confusion_matrix / business_utility MUST be
    # computed at the SAME operating point as the headline precision/recall/f1
    # (the metrics selected into ``test_metrics`` just above). In the
    # imbalanced path the headline is at the validation-frozen optimal
    # threshold (``y_test_pred_optimal``); in the balanced path the headline
    # is the model's default 0.5 (``y_test_pred``). Selecting the matching
    # prediction array here keeps the headline confusion matrix and business
    # utility consistent with the headline classification metrics. The
    # dedicated ``test_metrics_at_05`` / ``test_metrics_at_optimal`` keys
    # continue to carry BOTH operating points unchanged.
    if imbalance_detected:
        y_test_pred_headline = y_test_pred_optimal
        headline_threshold = optimal_threshold
    else:
        y_test_pred_headline = y_test_pred
        headline_threshold = 0.5

    # Confusion matrix at the headline operating point
    cm = confusion_matrix(y_test, y_test_pred_headline)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        confusion_dict = {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}
    else:
        confusion_dict = {"matrix": cm.tolist()}

    # Block 5 (#10): business_utility from cost_matrix at the headline
    # operating point. We compute it on BOTH validation and test using the
    # same threshold so the metric reported in validation_metrics matches the
    # operating point that produced the test number — a deployment decision
    # tool needs both. (Findings #5: ``headline_threshold`` == the operating
    # point of the headline metrics, NOT unconditionally the optimal one.)
    test_business_utility: Optional[float] = None
    val_business_utility: Optional[float] = None
    if cost_matrix is not None and cm.shape == (2, 2):
        test_business_utility = _compute_business_utility(
            int(tp), int(fp), int(fn), int(tn), cost_matrix
        )
        if y_validation is not None and y_validation_proba is not None:
            y_val_proba_pos = _positive_class_proba(y_validation_proba)
            y_val_pred_at_chosen = (y_val_proba_pos >= headline_threshold).astype(int)
            val_cm = confusion_matrix(y_validation, y_val_pred_at_chosen)
            if val_cm.shape == (2, 2):
                v_tn, v_fp, v_fn, v_tp = val_cm.ravel()
                val_business_utility = _compute_business_utility(
                    int(v_tp), int(v_fp), int(v_fn), int(v_tn), cost_matrix
                )
        if val_business_utility is not None:
            validation_metrics["business_utility"] = val_business_utility
        test_metrics["business_utility"] = test_business_utility
        logger.info(
            f"business_utility (headline_threshold={headline_threshold:.4f}): "
            f"validation={val_business_utility}, test={test_business_utility}"
        )

    # Precision at k
    precision_at_k = _compute_precision_at_k(y_test, y_test_proba, k_values=[100, 500, 1000])

    # Bootstrap confidence intervals (cycle-16 I-4: per-fold seed for
    # asyncio.gather n_jobs > 1 determinism)
    confidence_interval, bootstrap_samples = _compute_bootstrap_ci(
        y_test,
        y_test_pred_optimal,
        y_test_proba,
        problem_type="binary_classification",
        random_state=bootstrap_random_state,
    )

    # Backlog #37: when the rebinarisation guard upstream (see the
    # `not math.isclose(optimal_threshold, 0.5)` check near the top of this
    # function) skipped, ``y_test_pred`` and ``y_test_pred_optimal`` are the
    # same array, so the two _compute_split_classification_metrics calls
    # above produce equivalent core metrics. The post-call enrichment
    # (calibration_*, business_utility, baseline_test_auc,
    # minimum_lift_over_baseline, train_val_auc_delta, net_benefit_grid,
    # decision_curve_data, NB-area block, anchor-point block) attaches only
    # to ``test_metrics``, which aliases ``test_metrics_standard`` when
    # imbalance_detected=False or ``test_metrics_optimal`` when True — so
    # the unaliased dict ends up with a strict subset of keys. Mirror the
    # enriched dict into the other slot so both retain identical keysets,
    # matching their documented invariant: same y_pred → identical metrics
    # dict.
    #
    # Codex pass-1 MEDIUM-1: ``copy.deepcopy`` (not shallow ``dict(...)``)
    # so nested mutable values (``net_benefit_grid``, ``decision_curve_data``)
    # do not alias between the two output slots; without deep copy, a future
    # caller mutating a nested value via one slot would silently corrupt the
    # other. Note that ``test_metrics`` and one of the two slots remain
    # bound to the same dict object by prior assignment at line ~1755 /
    # ~1762 — that is a pre-existing identity not introduced by this fix.
    if math.isclose(optimal_threshold, 0.5):
        if imbalance_detected:
            test_metrics_standard = copy.deepcopy(test_metrics_optimal)
        else:
            test_metrics_optimal = copy.deepcopy(test_metrics_standard)

    # Build result dictionary
    result = {
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_metrics_at_05": test_metrics_standard,  # Keep standard for reference
        "test_metrics_at_optimal": test_metrics_optimal,  # Optimal threshold metrics
        "auc_roc": auc_roc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "confusion_matrix": confusion_dict,
        "optimal_threshold": optimal_threshold,
        "chosen_threshold_source": threshold_source,
        "precision_at_k": precision_at_k,
        "confidence_interval": confidence_interval,
        "bootstrap_samples": bootstrap_samples,
        "precision_constrained": precision_constrained,
        "calibration_error": None,  # Could add ECE computation
        "rmse": None,
        "mae": None,
        "r2": None,
        # Weighted metrics for imbalanced classification
        "f1_macro": test_metrics.get("f1_macro"),
        "f1_weighted": test_metrics.get("f1_weighted"),
        # Block 5 (#10): top-level mirror of test-set business_utility for
        # downstream consumers (Tier0OutputMapper, deployment decision
        # tools) that read flat metrics off the result dict. None when no
        # cost_matrix was provided.
        "business_utility": test_business_utility,
    }

    # Add minority class metrics when imbalance is detected
    # These are the key metrics for evaluating imbalanced classification
    if imbalance_detected:
        result["minority_recall"] = test_metrics.get("recall_class_1", 0.0)
        result["minority_precision"] = test_metrics.get("precision_class_1", 0.0)
        # Also report what metrics would be at 0.5 threshold for comparison
        result["minority_recall_at_05"] = test_metrics_standard.get("recall_class_1", 0.0)
        logger.info(
            f"Imbalance metrics (optimal threshold): "
            f"minority_recall={result['minority_recall']:.4f}, "
            f"minority_precision={result['minority_precision']:.4f}"
        )
        if float(result["minority_recall_at_05"]) == 0 and float(result["minority_recall"]) > 0:  # type: ignore[arg-type]
            logger.warning(
                f"Model would predict ALL negatives at 0.5 threshold! "
                f"Optimal threshold {optimal_threshold:.4f} rescues recall."
            )

    return result


def _compute_split_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    """Compute classification metrics for a single split.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    metrics: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        # Weighted metrics (critical for imbalanced data)
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        # Per-class metrics (essential for understanding imbalanced performance)
        "precision_class_0": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_class_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_class_0": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_class_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        # Matthews Correlation Coefficient — robust to class imbalance
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    # Probability-based metrics
    if y_proba is not None:
        y_proba_pos = _positive_class_proba(y_proba)

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba_pos))
        except ValueError:
            metrics["roc_auc"] = None

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba_pos))
        except ValueError:
            metrics["pr_auc"] = None

        try:
            metrics["brier_score"] = float(brier_score_loss(y_true, y_proba_pos))
        except ValueError:
            metrics["brier_score"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["brier_score"] = None

    return metrics


def _compute_multiclass_metrics(
    y_train: Optional[np.ndarray],
    y_train_pred: Optional[np.ndarray],
    y_train_proba: Optional[np.ndarray],
    y_validation: Optional[np.ndarray],
    y_validation_pred: Optional[np.ndarray],
    y_validation_proba: Optional[np.ndarray],
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_proba: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Compute multiclass classification metrics.

    Args:
        y_train: Training labels
        y_train_pred: Training predictions
        y_train_proba: Training probabilities
        y_validation: Validation labels
        y_validation_pred: Validation predictions
        y_validation_proba: Validation probabilities
        y_test: Test labels
        y_test_pred: Test predictions
        y_test_proba: Test probabilities

    Returns:
        Dictionary of metrics
    """
    # Training metrics
    train_metrics = {}
    if y_train is not None and y_train_pred is not None:
        train_metrics = {
            "accuracy": float(accuracy_score(y_train, y_train_pred)),
            "f1_macro": float(f1_score(y_train, y_train_pred, average="macro", zero_division=0)),
            "f1_weighted": float(
                f1_score(y_train, y_train_pred, average="weighted", zero_division=0)
            ),
        }

    # Validation metrics
    validation_metrics = {}
    if y_validation is not None and y_validation_pred is not None:
        validation_metrics = {
            "accuracy": float(accuracy_score(y_validation, y_validation_pred)),
            "f1_macro": float(
                f1_score(y_validation, y_validation_pred, average="macro", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_validation, y_validation_pred, average="weighted", zero_division=0)
            ),
        }

    # Test metrics
    test_metrics: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision_macro": float(
            precision_score(y_test, y_test_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_test, y_test_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_test_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0)),
    }

    # AUC for multiclass (OvR)
    if y_test_proba is not None:
        try:
            test_metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_test, y_test_proba, multi_class="ovr")
            )
        except ValueError:
            test_metrics["roc_auc_ovr"] = None

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    confusion_dict = {"matrix": cm.tolist()}

    return {
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "auc_roc": test_metrics.get("roc_auc_ovr"),
        "precision": test_metrics.get("precision_macro"),
        "recall": test_metrics.get("recall_macro"),
        "f1_score": test_metrics.get("f1_macro"),
        "pr_auc": None,
        "brier_score": None,
        "confusion_matrix": confusion_dict,
        "optimal_threshold": None,
        "precision_at_k": None,
        "confidence_interval": {},
        "bootstrap_samples": 0,
        "calibration_error": None,
        "rmse": None,
        "mae": None,
        "r2": None,
    }


def _compute_regression_metrics(
    y_train: Optional[np.ndarray],
    y_train_pred: Optional[np.ndarray],
    y_validation: Optional[np.ndarray],
    y_validation_pred: Optional[np.ndarray],
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    bootstrap_random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute regression metrics using sklearn.

    Args:
        y_train: Training labels
        y_train_pred: Training predictions
        y_validation: Validation labels
        y_validation_pred: Validation predictions
        y_test: Test labels
        y_test_pred: Test predictions
        bootstrap_random_state: Optional per-fold seed threaded into
            ``_compute_bootstrap_ci`` (cycle-16 I-4 / Q2-B). See helper docstring.

    Returns:
        Dictionary of metrics
    """
    # Training metrics
    train_metrics = {}
    if y_train is not None and y_train_pred is not None:
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_metrics = {
            "mse": float(train_mse),
            "rmse": float(np.sqrt(train_mse)),
            "mae": float(mean_absolute_error(y_train, y_train_pred)),
            "r2": float(r2_score(y_train, y_train_pred)),
        }

    # Validation metrics
    validation_metrics = {}
    if y_validation is not None and y_validation_pred is not None:
        val_mse = mean_squared_error(y_validation, y_validation_pred)
        validation_metrics = {
            "mse": float(val_mse),
            "rmse": float(np.sqrt(val_mse)),
            "mae": float(mean_absolute_error(y_validation, y_validation_pred)),
            "r2": float(r2_score(y_validation, y_validation_pred)),
        }

    # Test metrics (FINAL)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_metrics = {
        "mse": float(test_mse),
        "rmse": float(np.sqrt(test_mse)),
        "mae": float(mean_absolute_error(y_test, y_test_pred)),
        "r2": float(r2_score(y_test, y_test_pred)),
    }

    # Bootstrap confidence intervals (cycle-16 I-4: per-fold seed)
    confidence_interval, bootstrap_samples = _compute_bootstrap_ci(
        y_test,
        y_test_pred,
        None,
        problem_type="regression",
        random_state=bootstrap_random_state,
    )

    return {
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "r2": test_metrics["r2"],
        "confidence_interval": confidence_interval,
        "bootstrap_samples": bootstrap_samples,
        # Classification metrics not applicable
        "auc_roc": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "pr_auc": None,
        "brier_score": None,
        "confusion_matrix": None,
        "optimal_threshold": None,
        "precision_at_k": None,
        "calibration_error": None,
    }


def _compute_business_utility(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    cost_matrix: Dict[str, float],
) -> float:
    """Compute business_utility = sum(cost_matrix[outcome] * count[outcome]).

    Each confusion-matrix outcome is multiplied by its per-prediction
    monetary value from ``cost_matrix`` and summed. A "cost" matrix is
    typically signed: revenue from true positives is positive; the cost
    of a false positive is negative; the cost of a missed target (false
    negative) is also negative. The caller decides the sign convention —
    this helper just multiplies and sums.

    Args:
        tp/fp/fn/tn: Confusion-matrix counts at the chosen threshold.
        cost_matrix: Dict with keys ``tp``/``fp``/``fn``/``tn`` mapped to
            float dollar values per prediction.

    Returns:
        Total business utility (float).

    Raises:
        KeyError: If any of the four required keys is missing from
            ``cost_matrix``. The scope_definer's ``_validate_cost_matrix``
            normally guards against this, but the helper enforces it
            again so callers cannot silently drop a value.
    """
    return float(
        tp * cost_matrix["tp"]
        + fp * cost_matrix["fp"]
        + fn * cost_matrix["fn"]
        + tn * cost_matrix["tn"]
    )


def _select_threshold(
    y_validation: Optional[np.ndarray],
    y_validation_proba: Optional[np.ndarray],
    *,
    cost_matrix: Optional[Dict[str, float]] = None,
) -> Tuple[float, str]:
    """Pick the canonical classification threshold + provenance string.

    Block 1A — finding #6: the operating point MUST be selected on
    validation, then frozen for test. This helper encodes that policy:
    when validation labels and probabilities are available, return the
    Youden's J optimum from `_compute_optimal_threshold`; otherwise fall
    back to the default 0.5 threshold and log a warning so test-set
    integrity is preserved at the cost of an off-by-default operating
    point.

    Backlog #20 Gap 1 — when the caller supplies a ``cost_matrix`` (the
    same per-outcome dollar dict consumed by ``_compute_business_utility``),
    the helper picks the threshold that maximises validation
    business_utility instead of Youden's J. The cost matrix encodes
    asymmetric FN-vs-FP economics (typically FN >> FP for biologic-
    initiation prediction), so a cost-optimal threshold is the
    economically rational operating point. Falls through to Youden's J
    if the cost-aware sweep degenerates (constant probabilities, sklearn
    confusion-matrix shape mismatch, etc.).

    Note: the precision-constrained override (rare-event minority class)
    is NOT handled here — it requires additional caller context
    (`minority_ratio`) and produces a `precision_constrained` dict
    consumed elsewhere by the parent. The caller invokes this helper
    first and may then override the returned threshold.

    Args:
        y_validation: Validation labels. When None, falls back to default.
        y_validation_proba: Validation probabilities. When None, falls
            back to default.
        cost_matrix: Optional Block-5 (#10) per-outcome dollar matrix —
            ``{"tp": v, "fp": v, "fn": v, "tn": v}``. When provided AND
            validation arrays are present, the helper picks the threshold
            that maximises business_utility on validation (cost-aware
            selection). Falls through to Youden's J on degenerate input.

    Returns:
        Tuple of ``(chosen_threshold, chosen_threshold_source)``. The
        source string is one of these exact literals (used by
        downstream consumers — mlflow_logger, audit code, schema checkers
        — for provenance attribution):

        - ``"validation"``: validation arrays present, threshold tuned
          via Youden's J on them (canonical default).
        - ``"validation_cost_optimal"``: validation arrays + cost_matrix
          present, threshold picked to maximise business_utility on
          validation (Backlog #20 Gap 1).
        - ``"validation_f1_fallback"``: caller (``_compute_classification_metrics``)
          escalated to F1-optimal because validation MCC at the
          canonically-chosen threshold was below
          ``_F1_FALLBACK_MCC_THRESHOLD`` AND F1-optimal strictly
          improved MCC (Backlog #20 Gap 2). NOTE: ``_select_threshold``
          itself never returns this literal — the F1-fallback rewrites
          ``threshold_source`` in the parent helper. Documented here so
          consumers know the full set.
        - ``"default"``: validation arrays absent, threshold pinned to
          0.5 (test integrity preserved).

        Schema (``MetricsSchema.chosen_threshold_source``) accepts
        ``Optional[str]`` so additions don't require migration. Codex
        pass-2 LOW-4: as of this PR no consumer in src/ does an exact
        ``== "validation"`` equality check, so the new literals don't
        silently bypass downstream paths. If a future consumer adds
        such a check, it must enumerate this literal set.
    """
    if y_validation is not None and y_validation_proba is not None:
        if cost_matrix is not None:
            cost_threshold = _compute_cost_optimal_threshold(
                y_validation, y_validation_proba, cost_matrix
            )
            if cost_threshold is not None:
                return cost_threshold, "validation_cost_optimal"
            # Cost-optimal failed (degenerate input or all-equal utilities);
            # fall through to Youden's J so the model still gets a tuned
            # operating point.
        return _compute_optimal_threshold(y_validation, y_validation_proba), "validation"

    logger.warning(
        "Validation arrays unavailable for threshold tuning; "
        "falling back to default 0.5 threshold (test integrity preserved)."
    )
    return 0.5, "default"


_COST_MATRIX_REQUIRED_KEYS: frozenset[str] = frozenset({"tp", "fp", "fn", "tn"})


def _compute_cost_optimal_threshold(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    cost_matrix: Dict[str, float],
) -> Optional[float]:
    """Find the threshold that maximises business_utility on (y_true, y_proba).

    Sweeps a 99-step grid (0.01..0.99) and at each candidate computes
    ``_compute_business_utility(tp, fp, fn, tn, cost_matrix)`` from the
    confusion matrix. Returns the threshold yielding the highest utility.
    Backlog #20 Gap 1.

    Returns ``None`` on degenerate input (no probabilities, single-class
    y_true, flat utility across the grid). Caller falls through to
    Youden's J in that case.

    **Loud failures:** raises ``KeyError`` if ``cost_matrix`` is missing
    any of ``{"tp", "fp", "fn", "tn"}``. Codex pass-2 HIGH-1: a malformed
    config is a bug, NOT a fall-through condition — the previous broad
    ``except Exception`` swallowed ``KeyError`` and silently labeled the
    Youden's-J pick as the cost path's outcome, hiding the
    misconfiguration. Validating up front means the caller (and CI test
    suite) sees the bug immediately.

    Args:
        y_true: True labels (1-D, binary).
        y_proba: Predicted probabilities (1-D positive-class scores or
            2-D ``(n, 2)``). None short-circuits to None.
        cost_matrix: Per-outcome dollar matrix. Required keys
            ``{"tp", "fp", "fn", "tn"}`` — raises ``KeyError`` on missing.

    Returns:
        Threshold (float in (0, 1)) maximising validation utility, or
        None if the sweep is degenerate (no proba; flat utility; numeric
        failure during the sweep).

    Raises:
        KeyError: If any of the four required cost_matrix keys is missing.
    """
    if y_proba is None:
        return None

    # Codex pass-2 HIGH-1 fix: validate keys upfront and raise loud.
    # The previous broad ``except Exception`` would silently swallow a
    # KeyError from ``_compute_business_utility`` when the matrix is
    # malformed, falling through to Youden's J labeled as if the cost
    # path produced it. A malformed config must crash, not silently
    # mislabel.
    missing_keys = _COST_MATRIX_REQUIRED_KEYS - set(cost_matrix)
    if missing_keys:
        raise KeyError(
            f"_compute_cost_optimal_threshold: cost_matrix is missing required "
            f"keys {sorted(missing_keys)!r}; got keys {sorted(cost_matrix)!r}. "
            f"Refusing to fall through silently — fix the matrix and retry."
        )

    y_proba_pos = _positive_class_proba(y_proba)

    try:
        thresholds = np.linspace(0.01, 0.99, 99)
        best_utility = -np.inf
        worst_utility = np.inf
        best_threshold: Optional[float] = None

        for t in thresholds:
            y_pred = (y_proba_pos >= t).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape != (2, 2):
                continue
            tn, fp, fn, tp = cm.ravel()
            utility = _compute_business_utility(int(tp), int(fp), int(fn), int(tn), cost_matrix)
            if utility > best_utility:
                best_utility = utility
                best_threshold = float(t)
            if utility < worst_utility:
                worst_utility = utility

        # Codex pass-2 MEDIUM-2 fix: flat-utility sweep means every
        # threshold is equivalent. Returning the first-touched threshold
        # (~0.01) and labeling it ``"validation_cost_optimal"`` would be
        # a false claim. Reject so the caller falls through to Youden's J.
        if best_threshold is None or not np.isfinite(best_utility):
            return None
        if math.isclose(best_utility, worst_utility, rel_tol=0.0, abs_tol=1e-12):
            logger.info(
                "Cost-optimal sweep produced flat utility (best=%.6f, worst=%.6f); "
                "rejecting cost branch and falling through to Youden's J.",
                best_utility,
                worst_utility,
            )
            return None
        return best_threshold
    except Exception as e:
        # Numeric / data-degeneracy fall-through ONLY. KeyError is
        # already raised above, so a KeyError here would only arise from
        # downstream sklearn internals — still appropriate to surface.
        logger.warning(f"Cost-optimal threshold computation failed: {e}")
        return None


def _compute_optimal_threshold(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> float:
    """Compute optimal classification threshold using Youden's J statistic.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities

    Returns:
        Optimal threshold
    """
    if y_proba is None:
        return 0.5

    y_proba_pos = _positive_class_proba(y_proba)

    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos)
        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        candidate = float(thresholds[optimal_idx])
        # sklearn's roc_curve prepends a sentinel threshold of `np.inf`
        # corresponding to the (FPR=0, TPR=0) trivial point. When no
        # threshold yields a positive Youden's J (e.g., a model that is
        # worse than chance on a small held-out set) argmax returns this
        # sentinel. Treat any non-finite or out-of-range value as
        # "no useful threshold found" and fall back to 0.5.
        if not np.isfinite(candidate) or candidate < 0.0 or candidate > 1.0:
            return 0.5
        return candidate
    except Exception:
        return 0.5


def _compute_precision_constrained_threshold(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    target_precision: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """Find the lowest threshold where precision >= target.

    For rare-event prediction, Youden's J often yields very low precision.
    This finds a threshold that guarantees a minimum precision level.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        target_precision: Minimum required precision (default 5%)

    Returns:
        Dict with threshold details, or None if probabilities unavailable
    """
    if y_proba is None:
        return None

    y_proba_pos = _positive_class_proba(y_proba)

    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba_pos)
        # precision_recall_curve returns n+1 precisions but n thresholds
        # The last precision is always 1.0 with recall 0.0 (no corresponding threshold)

        # Find lowest threshold where precision >= target
        best_idx = None
        for i in range(len(thresholds)):
            if precisions[i] >= target_precision:
                if best_idx is None or thresholds[i] < thresholds[best_idx]:
                    best_idx = i

        if best_idx is not None:
            threshold = float(thresholds[best_idx])
            prec = float(precisions[best_idx])
            rec = float(recalls[best_idx])
            # Compute F1 at this threshold
            f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            return {
                "precision_constrained_threshold": threshold,
                "precision_at_threshold": prec,
                "recall_at_threshold": rec,
                "f1_at_threshold": f1_val,
                "target_precision": target_precision,
                "target_achieved": True,
                "fallback_used": False,
            }

        # Fallback: F1-optimal threshold
        f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
        f1_best_idx = int(np.argmax(f1_scores))
        threshold = float(thresholds[f1_best_idx])
        prec = float(precisions[f1_best_idx])
        rec = float(recalls[f1_best_idx])
        f1_val = float(f1_scores[f1_best_idx])
        return {
            "precision_constrained_threshold": threshold,
            "precision_at_threshold": prec,
            "recall_at_threshold": rec,
            "f1_at_threshold": f1_val,
            "target_precision": target_precision,
            "target_achieved": False,
            "fallback_used": True,
        }

    except Exception as e:
        logger.warning(f"Precision-constrained threshold computation failed: {e}")
        return None


# Commercial-targeting recall floor for the operating-point selection. Mirrors
# the commercial ``minimum_recall`` in
# ``scope_definer/nodes/criteria_validator.py`` (kept as a local constant to
# avoid a cross-agent import; both reflect the owner-ratified commercial bar).
_COMMERCIAL_RECALL_TARGET: float = 0.50


def _compute_recall_constrained_threshold(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    target_recall: float = _COMMERCIAL_RECALL_TARGET,
) -> Optional[Dict[str, Any]]:
    """Find the HIGHEST threshold where recall >= target (recall analogue of
    ``_compute_precision_constrained_threshold``).

    A COMMERCIAL targeting model is used by its RANKING (target the top scored
    decile), so its decision threshold should catch the commercial recall floor
    — false positives are cheap in outreach. Selecting the highest threshold
    whose recall >= target maximises precision SUBJECT to the recall constraint.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        target_recall: Minimum required recall (default = commercial floor).

    Returns:
        Dict with threshold details (``target_achieved``), or None if
        probabilities are unavailable.
    """
    if y_proba is None:
        return None

    y_proba_pos = _positive_class_proba(y_proba)

    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba_pos)
        # precision_recall_curve returns len(thresholds) == len(precisions) - 1;
        # recalls are monotonically non-increasing as the threshold rises.
        best_idx = None
        for i in range(len(thresholds)):
            if recalls[i] >= target_recall:
                # Highest threshold still meeting the recall floor → best precision.
                if best_idx is None or thresholds[i] > thresholds[best_idx]:
                    best_idx = i

        if best_idx is not None:
            threshold = float(thresholds[best_idx])
            prec = float(precisions[best_idx])
            rec = float(recalls[best_idx])
            f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            return {
                "recall_constrained_threshold": threshold,
                "precision_at_threshold": prec,
                "recall_at_threshold": rec,
                "f1_at_threshold": f1_val,
                "target_recall": target_recall,
                "target_achieved": True,
            }

        # No threshold meets the recall floor (degenerate model). Signal failure;
        # the caller keeps the canonical operating point.
        return {
            "recall_constrained_threshold": 0.5,
            "target_recall": target_recall,
            "target_achieved": False,
        }

    except Exception as e:
        logger.warning(f"Recall-constrained threshold computation failed: {e}")
        return None


def _compute_precision_at_k(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    k_values: List[int],
) -> Dict[int, float]:
    """Compute precision at k for different k values.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        k_values: List of k values

    Returns:
        Dictionary of {k: precision_at_k}
    """
    if y_proba is None:
        return {}

    y_proba_pos = _positive_class_proba(y_proba)

    n_samples = len(y_true)
    result = {}

    for k in k_values:
        if k > n_samples:
            continue

        # Get top k indices by probability
        top_k_indices = np.argsort(y_proba_pos)[-k:]

        # Compute precision at k
        precision_at_k = np.mean(y_true[top_k_indices])
        result[k] = float(precision_at_k)

    return result


# Q-W5-3 RESOLVED 2026-05-02 cycle 6: BCa default flipped to ``"auto"``.
# Below ``_MIN_N_FOR_BCA`` bootstrap units, BCa acceleration is unstable
# (Bengio & Grandvalet 2004 — no distribution-free unbiased k-fold CV
# variance estimator). 30 is the codex B verdict cutoff.
_MIN_N_FOR_BCA: int = 30


def _compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    problem_type: str,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
    ci_method: str = "auto",
    force_bca: bool = False,
) -> Tuple[Dict[str, Tuple[float, float]], int]:
    """Compute bootstrap confidence intervals for metrics (shard 20 §D.3).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        problem_type: Problem type
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Optional per-fold seed (cycle-16 I-4 / Q2-B). When
            provided, uses ``np.random.default_rng(random_state)`` for
            bootstrap-index generation so two folds running concurrently
            under ``asyncio.gather(n_jobs > 1)`` produce bit-identical CI
            endpoints regardless of execution order. When ``None``
            (default), falls back to numpy's global RNG via
            ``np.random.choice`` — preserves byte-identity for legacy
            callers that don't supply a fold seed (single-mode evaluation,
            external test fixtures with explicit ``np.random.seed``).
        ci_method: One of ``"auto"`` (default), ``"bca"``, ``"percentile"``.
            Q-W5-3 RESOLVED 2026-05-02 cycle 6: ``"auto"`` resolves to
            percentile when ``n_bootstrap < _MIN_N_FOR_BCA`` (=30), BCa
            otherwise. Legacy snapshot tests should pin
            ``ci_method="percentile"`` explicitly.
        force_bca: When True + ``ci_method="bca"`` + ``n_bootstrap < _MIN_N_FOR_BCA``,
            still attempt BCa (caller acknowledges small-N fragility). When
            False (default), falls back to percentile with
            ``bca_fallback_reason="below_min_n_for_bca"`` logged at INFO.

    Returns:
        Tuple of (confidence_intervals, n_bootstrap). Provenance fields
        (``ci_method_requested`` / ``ci_method_used`` / ``bca_fallback_reason``
        / ``bca_unstable_warning``) are emitted via INFO log to avoid
        breaking the legacy 2-tuple contract.
    """
    n_samples = len(y_true)
    alpha = (1 - confidence) / 2

    # Get positive class probabilities if available
    y_proba_pos = _positive_class_proba(y_proba) if y_proba is not None else None

    # Store bootstrap metrics
    bootstrap_metrics: Dict[str, List[float]] = {}

    # Cycle-16 I-4: per-fold-seeded bootstrap RNG when caller supplies seed;
    # global-RNG fallback preserves backward-compat byte-identity.
    rng = np.random.default_rng(random_state) if random_state is not None else None

    for _ in range(n_bootstrap):
        # Bootstrap sample indices
        if rng is not None:
            indices = rng.integers(low=0, high=n_samples, size=n_samples)
        else:
            indices = np.random.choice(n_samples, size=n_samples, replace=True)

        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        if problem_type == "binary_classification":
            # Accuracy
            if "accuracy" not in bootstrap_metrics:
                bootstrap_metrics["accuracy"] = []
            bootstrap_metrics["accuracy"].append(accuracy_score(y_true_boot, y_pred_boot))

            # AUC
            if y_proba_pos is not None:
                y_proba_boot = y_proba_pos[indices]
                try:
                    if "auc" not in bootstrap_metrics:
                        bootstrap_metrics["auc"] = []
                    bootstrap_metrics["auc"].append(roc_auc_score(y_true_boot, y_proba_boot))
                except ValueError:
                    pass

            # Precision, Recall, F1
            if "precision" not in bootstrap_metrics:
                bootstrap_metrics["precision"] = []
            bootstrap_metrics["precision"].append(
                precision_score(y_true_boot, y_pred_boot, zero_division=0)
            )

            if "recall" not in bootstrap_metrics:
                bootstrap_metrics["recall"] = []
            bootstrap_metrics["recall"].append(
                recall_score(y_true_boot, y_pred_boot, zero_division=0)
            )

        elif problem_type == "regression":
            y_pred_boot_reg = y_pred[indices]

            # RMSE
            if "rmse" not in bootstrap_metrics:
                bootstrap_metrics["rmse"] = []
            mse = mean_squared_error(y_true_boot, y_pred_boot_reg)
            bootstrap_metrics["rmse"].append(np.sqrt(mse))

            # MAE
            if "mae" not in bootstrap_metrics:
                bootstrap_metrics["mae"] = []
            bootstrap_metrics["mae"].append(mean_absolute_error(y_true_boot, y_pred_boot_reg))

            # R2
            if "r2" not in bootstrap_metrics:
                bootstrap_metrics["r2"] = []
            try:
                bootstrap_metrics["r2"].append(r2_score(y_true_boot, y_pred_boot_reg))
            except ValueError:
                pass

    # Resolve ci_method per Q-W5-3 RESOLVED. Below the BCa min-N cutoff
    # (Bengio & Grandvalet 2004 — k-fold variance estimator instability),
    # auto resolves to percentile and "bca" without ``force_bca`` falls back.
    method_requested = ci_method
    fallback_reason: Optional[str] = None
    if ci_method == "auto":
        method_resolved = "bca" if n_bootstrap >= _MIN_N_FOR_BCA else "percentile"
        if method_resolved == "percentile":
            fallback_reason = "auto_below_min_n_for_bca"
    elif ci_method == "bca":
        if n_bootstrap < _MIN_N_FOR_BCA and not force_bca:
            method_resolved = "percentile"
            fallback_reason = "below_min_n_for_bca"
        else:
            method_resolved = "bca"
    elif ci_method == "percentile":
        method_resolved = "percentile"
    else:
        logger.warning("Unknown ci_method=%r — falling back to percentile.", ci_method)
        method_resolved = "percentile"
        fallback_reason = f"unknown_ci_method:{ci_method}"

    confidence_intervals: Dict[str, Tuple[float, float]] = {}
    method_used: Dict[str, str] = {}

    if method_resolved == "percentile":
        for metric_name, values in bootstrap_metrics.items():
            if len(values) > 0:
                lower = float(np.percentile(values, alpha * 100))
                upper = float(np.percentile(values, (1 - alpha) * 100))
                confidence_intervals[metric_name] = (lower, upper)
                method_used[metric_name] = "percentile"
        logger.info(
            "bootstrap CI: requested=%s used=percentile fallback_reason=%s",
            method_requested,
            fallback_reason,
        )
        return confidence_intervals, n_bootstrap

    # BCa branch — compute point estimates + jackknife distributions
    # once, then call bca_ci_from_resamples per metric.
    from src.utils.bootstrap_utils import bca_ci_from_resamples

    point_estimates = _compute_point_estimates(y_true, y_pred, y_proba_pos, problem_type)
    jackknife_cache = _compute_jackknife_metrics(y_true, y_pred, y_proba_pos, problem_type)
    bca_unstable = False
    for metric_name, values in bootstrap_metrics.items():
        if len(values) == 0 or metric_name not in point_estimates:
            continue
        result = bca_ci_from_resamples(
            bootstrap_values=values,
            point_estimate=point_estimates[metric_name],
            jackknife_values=jackknife_cache.get(metric_name, np.array([])),
            confidence_level=confidence,
        )
        if result.ci_lo is None or result.ci_hi is None:
            continue
        confidence_intervals[metric_name] = (result.ci_lo, result.ci_hi)
        method_used[metric_name] = result.method
        if result.method != "bca":
            bca_unstable = True
    # Cycle-22 C-2: report which per-metric methods actually applied so
    # log consumers can distinguish "BCa requested + delivered for all"
    # from "BCa requested but every metric fell back to percentile."
    logger.info(
        "bootstrap CI: requested=%s resolved=bca per_metric_methods=%s any_fallback=%s",
        method_requested,
        method_used,
        bca_unstable,
    )
    return confidence_intervals, n_bootstrap


def _compute_point_estimates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_pos: Optional[np.ndarray],
    problem_type: str,
) -> Dict[str, float]:
    """Per-metric point estimate on the original (non-resampled) sample.

    Mirrors the metric set computed inside the bootstrap loop in
    :func:`_compute_bootstrap_ci`; required input to BCa bias-correction
    (``z0 = Φ⁻¹(P(boot < point_estimate))``).
    """
    out: Dict[str, float] = {}
    if problem_type == "binary_classification":
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        if y_proba_pos is not None:
            try:
                out["auc"] = float(roc_auc_score(y_true, y_proba_pos))
            except ValueError:
                pass
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    elif problem_type == "regression":
        mse_val = float(mean_squared_error(y_true, y_pred))
        out["rmse"] = float(np.sqrt(mse_val))
        out["mae"] = float(mean_absolute_error(y_true, y_pred))
        try:
            out["r2"] = float(r2_score(y_true, y_pred))
        except ValueError:
            pass
    return out


def _compute_jackknife_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_pos: Optional[np.ndarray],
    problem_type: str,
) -> Dict[str, np.ndarray]:
    """Per-metric leave-one-out distribution for BCa acceleration.

    Returns ``{metric_name: ndarray of length N}`` where the i-th value
    is the metric on ``(y_true, y_pred, y_proba_pos)`` with sample i
    excluded. Used by :func:`bca_ci_from_resamples` for the jackknife
    acceleration term.

    Computational cost: N metric evaluations per metric. At N=1500 this
    is ~3 seconds for all 4 binary metrics — well under the existing
    bootstrap loop's 1000-iteration cost.
    """
    n = int(len(y_true))
    if n < 2:
        return {}
    idx = np.arange(n)
    out: Dict[str, np.ndarray] = {}

    if problem_type == "binary_classification":
        accuracy_jack = np.empty(n, dtype=float)
        precision_jack = np.empty(n, dtype=float)
        recall_jack = np.empty(n, dtype=float)
        auc_jack = np.full(n, np.nan, dtype=float)
        for i in range(n):
            mask = idx != i
            yt = y_true[mask]
            yp = y_pred[mask]
            accuracy_jack[i] = accuracy_score(yt, yp)
            precision_jack[i] = precision_score(yt, yp, zero_division=0)
            recall_jack[i] = recall_score(yt, yp, zero_division=0)
            if y_proba_pos is not None:
                try:
                    auc_jack[i] = roc_auc_score(yt, y_proba_pos[mask])
                except ValueError:
                    pass
        out["accuracy"] = accuracy_jack
        out["precision"] = precision_jack
        out["recall"] = recall_jack
        if y_proba_pos is not None:
            # Cycle-22 C-1: AUC jackknife array length asymmetry. Unlike
            # accuracy/precision/recall which always have N entries, the
            # AUC array only retains entries where the leave-one-out
            # sub-sample was non-degenerate (both classes present). The
            # NaN-drop is intentional — a NaN in the BCa acceleration
            # numerator destroys the Σ(...)³ formula. Downstream
            # ``bca_ci_from_resamples`` reads ``len(jackknife) >= 2`` as
            # its sanity guard.
            valid = ~np.isnan(auc_jack)
            if valid.sum() >= 2:
                out["auc"] = auc_jack[valid]
    elif problem_type == "regression":
        rmse_jack = np.empty(n, dtype=float)
        mae_jack = np.empty(n, dtype=float)
        r2_jack = np.full(n, np.nan, dtype=float)
        for i in range(n):
            mask = idx != i
            yt = y_true[mask]
            yp = y_pred[mask]
            rmse_jack[i] = float(np.sqrt(mean_squared_error(yt, yp)))
            mae_jack[i] = mean_absolute_error(yt, yp)
            try:
                r2_jack[i] = r2_score(yt, yp)
            except ValueError:
                pass
        out["rmse"] = rmse_jack
        out["mae"] = mae_jack
        valid = ~np.isnan(r2_jack)
        if valid.sum() >= 2:
            out["r2"] = r2_jack[valid]
    return out


def _check_metric_suspicion(
    metrics_result: Dict[str, Any],
    problem_type: str,
) -> Dict[str, Any]:
    """Post-training safety net: check for implausibly perfect metrics.

    Runs after metric computation to catch models that may be tautological
    (e.g., AUC=1.0, perfect precision+recall) — a sign of data leakage
    that pre-training checks may have missed.

    Args:
        metrics_result: Full metrics dict from _compute_*_metrics
        problem_type: Problem type

    Returns:
        Dictionary with leakage_suspected, suspicion_level,
        suspicion_reasons, investigation_recommendations
    """
    reasons: List[str] = []
    recommendations: List[str] = []

    test_metrics = metrics_result.get("test_metrics", {})
    train_metrics = metrics_result.get("train_metrics", {})
    validation_metrics = metrics_result.get("validation_metrics", {})

    if problem_type in ["binary_classification", "multiclass_classification"]:
        auc = test_metrics.get("roc_auc")
        precision = test_metrics.get("precision")
        recall = test_metrics.get("recall")
        brier = test_metrics.get("brier_score")

        # Check 1: AUC >= 0.99
        if auc is not None and auc >= 0.99:
            reasons.append(f"AUC={auc:.4f} >= 0.99 is implausible on real-world data")
            recommendations.append(
                "Check features for target leakage — no real-world clinical model achieves AUC >= 0.99"
            )

        # Check 2: Perfect precision AND recall
        if precision is not None and recall is not None:
            if precision >= 0.999 and recall >= 0.999:
                reasons.append(
                    f"Perfect precision ({precision:.4f}) and recall ({recall:.4f}) "
                    f"indicates tautological model"
                )
                recommendations.append(
                    "Features likely encode the target directly — audit feature derivation pipeline"
                )

        # Check 3: All splits AUC > 0.98 with near-zero variance
        split_aucs = []
        for m in [train_metrics, validation_metrics, test_metrics]:
            a = m.get("roc_auc")
            if a is not None:
                split_aucs.append(a)
        if len(split_aucs) >= 3 and all(a > 0.98 for a in split_aucs):
            variance = float(np.var(split_aucs))
            if variance < 0.01:
                reasons.append(
                    f"All splits AUC > 0.98 (variance={variance:.6f}) — "
                    f"no generalization gap across splits"
                )
                recommendations.append(
                    "Identical performance across splits suggests the signal is trivially recoverable"
                )

        # Check 4: Brier score == 0
        if brier is not None and brier < 1e-6:
            reasons.append(f"Brier score={brier:.2e} is effectively zero — implausible calibration")
            recommendations.append(
                "Zero calibration error means predicted probabilities are perfect — "
                "this only happens with deterministic features"
            )

    elif problem_type in ["regression", "continuous"]:
        r2 = test_metrics.get("r2")
        rmse = test_metrics.get("rmse")

        if r2 is not None and r2 >= 0.999:
            reasons.append(f"R²={r2:.6f} >= 0.999 is implausible on real-world data")
            recommendations.append("Check features for target leakage")

        if rmse is not None and rmse < 1e-6:
            reasons.append(f"RMSE={rmse:.2e} is effectively zero")
            recommendations.append(
                "Near-zero RMSE suggests features deterministically encode target"
            )

    # Determine suspicion level
    if not reasons:
        return {
            "leakage_suspected": False,
            "suspicion_level": "none",
            "suspicion_reasons": [],
            "investigation_recommendations": [],
        }

    # Determine severity from the original metric values (not string matching)
    has_critical = False
    if problem_type in ["binary_classification", "multiclass_classification"]:
        auc = test_metrics.get("roc_auc")
        prec = test_metrics.get("precision")
        rec = test_metrics.get("recall")
        if auc is not None and auc >= 0.99:
            has_critical = True
        if prec is not None and rec is not None and prec >= 0.999 and rec >= 0.999:
            has_critical = True
    elif problem_type in ["regression", "continuous"]:
        r2_val = test_metrics.get("r2")
        if r2_val is not None and r2_val >= 0.999:
            has_critical = True
    suspicion_level = "critical" if has_critical else "high"

    return {
        "leakage_suspected": True,
        "suspicion_level": suspicion_level,
        "suspicion_reasons": reasons,
        "investigation_recommendations": recommendations,
    }


def _check_success_criteria(
    test_metrics: Dict[str, float],
    success_criteria: Dict[str, float],
    problem_type: str,
) -> Dict[str, Any]:
    """Check if model meets success criteria.

    Args:
        test_metrics: Test set metrics
        success_criteria: Success thresholds
        problem_type: Problem type

    Returns:
        Dictionary with success_criteria_met and success_criteria_results
    """
    if not success_criteria:
        return {
            "success_criteria_met": True,
            "success_criteria_results": {},
        }

    # v3 (adaptive criteria follow-up): the adaptive overlay is now
    # applied by ``evaluate_model`` BEFORE this function is called, so
    # the overlaid dict can persist into ``state["success_criteria"]``
    # via the LangGraph node return. ``_check_success_criteria`` is now
    # a pure check over the already-overlaid criteria — easier to unit-
    # test in isolation. Tests that previously passed a stash-shaped
    # dict here and expected internal overlay must call
    # ``_apply_adaptive_criteria_overlay`` first.

    # Per-criterion outcome: True=met, False=not met, None=soft-skipped
    # (see narrow exemption below).
    results: Dict[str, Optional[bool]] = {}
    all_met = True

    # Map metric aliases (including scope_definer naming conventions)
    metric_aliases = {
        "auc": "roc_auc",
        "roc_auc": "roc_auc",
        "minimum_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "minimum_precision": "precision",
        "recall": "recall",
        "minimum_recall": "recall",
        "f1": "f1_score",
        "f1_score": "f1_score",
        "minimum_f1": "f1_score",
        "rmse": "rmse",
        "minimum_rmse": "rmse",
        "mae": "mae",
        "r2": "r2",
        "minimum_r2": "r2",
        "mape": "mape",
        "minimum_mape": "mape",
        # Section B (pre_phase2_unblockers): self-mapping documents that
        # the criterion shares its name with the test_metrics key. Soft-skip
        # behavior on missing values is gated on this exact criterion name
        # below — see _LIFT_OVER_BASELINE_CRITERION usage.
        "minimum_lift_over_baseline": "minimum_lift_over_baseline",
        # Adaptive criteria (task 04 of adaptive_success_criteria plan v3):
        # all five criteria below are emitted by adaptive_success_criteria()
        # when ADAPTIVE_CRITERIA is on; the corresponding metrics are surfaced
        # by _compute_classification_metrics in task 05. Lower-is-better is
        # set on the resolved metric names below. ``minimum_net_benefit_at_p_t``
        # is intentionally absent — it resolves via a special-case lookup
        # against ``net_benefit_grid`` keyed on the regime's p_t (see W3 fix).
        "maximum_calibration_error": "calibrated_ece",
        "maximum_train_val_delta": "train_val_auc_delta",
        "minimum_mcc": "mcc",
        "maximum_calibration_slope_deviation": "calibration_slope_deviation",
        "maximum_calibration_intercept_magnitude": "calibration_intercept_magnitude",
    }
    # Single source of truth for the narrow exemption below; any future
    # criterion that wants soft-skip behavior must go through the same
    # explicit allowlist (do NOT accept any criterion silently).
    _LIFT_OVER_BASELINE_CRITERION = "minimum_lift_over_baseline"
    # v3 NB > 0 gate: resolved against ``net_benefit_grid[p_t=...]`` instead
    # of a metric_aliases lookup. The audit field ``_adaptive_p_t`` (set by
    # the validator when ADAPTIVE_CRITERIA is on) carries the regime's
    # threshold probability; when absent, the criterion soft-skips.
    _NB_AT_P_T_CRITERION = "minimum_net_benefit_at_p_t"

    # Metrics where lower is better
    lower_is_better = {
        "rmse",
        "mae",
        "brier_score",
        "mse",
        "mape",
        # Adaptive criteria (task 04 v3): all four resolved metrics are
        # MAXIMUM tolerable values, so lower-is-better. ``mcc`` is
        # higher-is-better (default); ``net_benefit_at_p_t`` is
        # higher-is-better and resolved via the special-case path.
        "calibrated_ece",
        "train_val_auc_delta",
        "calibration_slope_deviation",
        "calibration_intercept_magnitude",
    }

    for criterion_name, threshold in success_criteria.items():
        # v3: skip audit fields (any key starting with underscore). The
        # validator emits ``_adaptive_skipped`` (list — also caught by the
        # skip-non-numeric branch below) and ``_adaptive_p_t`` (float —
        # would otherwise be evaluated as a numeric criterion against the
        # missing ``adaptive_p_t`` test metric).
        if isinstance(criterion_name, str) and criterion_name.startswith("_"):
            continue

        # Skip non-numeric thresholds (e.g. experiment_id, baseline_model, None placeholders)
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            logger.debug(f"Skipping non-numeric success criterion: {criterion_name}={threshold}")
            continue

        # v3 W3 fix: NB > 0 gate resolves against ``net_benefit_grid`` keyed
        # on the regime's p_t, not against a single ``net_benefit_at_p_t``
        # metric. Soft-skip when ``_adaptive_p_t`` is missing or the grid
        # does not carry the requested key.
        if criterion_name == _NB_AT_P_T_CRITERION:
            actual_value = _resolve_net_benefit_from_grid(test_metrics, success_criteria)
            metric_name = "net_benefit_at_p_t"
        else:
            # Resolve metric name
            metric_name = metric_aliases.get(criterion_name, criterion_name)
            actual_value = test_metrics.get(metric_name)

        if actual_value is None:
            # Default policy: a missing metric is a hard fail of the success
            # contract — silently passing would mask a genuine gap.
            #
            # NARROW EXEMPTION (Section B of pre_phase2_unblockers plan):
            # ``minimum_lift_over_baseline`` is computed by
            # ``_compute_baseline_test_metrics`` only when the binary problem
            # has enough rows in both train and test, and both splits have
            # both classes. When those guards trip the baseline AUC is
            # legitimately undefined, so we soft-skip rather than hard-fail.
            # ``met=None`` excludes the criterion from the aggregation; the
            # log line still surfaces the skip so it isn't silent.
            if criterion_name == _LIFT_OVER_BASELINE_CRITERION:
                logger.warning(
                    "Success criterion soft-skipped (degenerate split): "
                    f"{criterion_name} (no baseline AUC available — "
                    "see _compute_baseline_test_metrics guards)"
                )
                results[criterion_name] = None
                continue
            # v3 W3 fix: the NB > 0 gate also soft-skips when the audit
            # ``_adaptive_p_t`` is unset (fixed mode) or the grid does not
            # carry the regime's p_t. The validator does not emit the gate
            # under fixed mode, but a stale config could still send it
            # through — soft-skip rather than hard-fail.
            if criterion_name == _NB_AT_P_T_CRITERION:
                logger.warning(
                    "Success criterion soft-skipped (NB grid unavailable "
                    "or _adaptive_p_t unset): %s",
                    criterion_name,
                )
                results[criterion_name] = None
                continue

            logger.warning(
                f"Success criterion metric not available: {criterion_name} "
                f"(resolved to '{metric_name}', missing from test_metrics)"
            )
            results[criterion_name] = False
            all_met = False
            continue

        # v3 B3 fix: NaN actual values record met=None instead of
        # comparing (Python's `nan <= 0.15` evaluates to False). Fires
        # for adverse-regime calibration metrics when n_pos < 30 or
        # n_neg < 30 (the LR fit is unstable and emits NaN).
        if isinstance(actual_value, float) and math.isnan(actual_value):
            logger.warning(
                "Success criterion soft-skipped (metric value is NaN): %s (resolved to '%s')",
                criterion_name,
                metric_name,
            )
            results[criterion_name] = None
            continue

        # Check if metric meets threshold
        if metric_name in lower_is_better:
            met = actual_value <= threshold
        else:
            met = actual_value >= threshold

        results[criterion_name] = met
        if not met:
            logger.info(
                f"Success criterion not met: {criterion_name}={actual_value:.4f} "
                f"(threshold={threshold})"
            )
            all_met = False
        else:
            logger.info(
                f"Success criterion met: {criterion_name}={actual_value:.4f} "
                f"(threshold={threshold})"
            )

    # v2/v3 explicit-skip set: criterion names listed in
    # ``success_criteria['_adaptive_skipped']`` are recorded as met=None
    # in results — the audit-trail recording for criteria the adaptive
    # scheme intentionally excluded from aggregation. Plain None thresholds
    # do NOT enter this path; they fall through the existing
    # skip-non-numeric branch above and the validator's S4 defense in
    # ``_define_classification_criteria`` warns + replaces them. See
    # ``.claude/plans/adaptive_success_criteria/01-design.md`` §"Skip
    # semantics" for the full contract.
    adaptive_skipped = success_criteria.get("_adaptive_skipped")
    if isinstance(adaptive_skipped, list):
        for skipped_name in adaptive_skipped:
            if isinstance(skipped_name, str):
                results[skipped_name] = None

    return {
        "success_criteria_met": all_met,
        "success_criteria_results": results,
    }


def _apply_adaptive_criteria_overlay(
    success_criteria: Dict[str, Any],
    test_metrics: Dict[str, Any],
    *,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
) -> Dict[str, Any]:
    """Recompute v3 adaptive thresholds with the live ``baseline_test_auc``.

    The scope_definer cannot compute ``baseline_auc`` — it runs before any
    model is trained. It therefore stashes the four pre-eval inputs as
    ``success_criteria['_adaptive_inputs']``; this helper reads that stash
    and the freshly-computed ``baseline_test_auc`` from ``test_metrics``,
    calls ``adaptive_success_criteria()``, and applies the resulting
    ``(thresholds, skipped)`` tuple to a SHALLOW COPY of
    ``success_criteria``:

    - Skipped criteria are REMOVED from the dict (v3 invariant).
    - The v3-deprecated fixed gates (precision / F1) are popped so only
      v3 active gates fire downstream.
    - Firing thresholds OVERWRITE the seeded fixed values.
    - The audit list is stored at ``success_criteria['_adaptive_skipped']``.
    - The regime-keyed threshold probability is stored at
      ``success_criteria['_adaptive_p_t']`` for the NB > 0 gate resolver.

    Returns the (possibly overlaid) dict. When ``_adaptive_inputs`` or
    ``baseline_test_auc`` are absent, the criteria dict is returned
    unchanged. See ``.claude/plans/adaptive_success_criteria/01-design.md``
    for the v3 design contract and ``05-data-shape-introspection.md`` for
    the two-phase hand-off rationale.
    """
    if "_adaptive_inputs" not in success_criteria:
        return success_criteria
    inputs = success_criteria.get("_adaptive_inputs")
    if not isinstance(inputs, dict):
        logger.warning(
            "_adaptive_inputs is not a dict (%s); skipping adaptive overlay",
            type(inputs).__name__,
        )
        return success_criteria
    baseline_auc = test_metrics.get("baseline_test_auc")
    if baseline_auc is None:
        return success_criteria

    # Local import avoids a top-level cross-agent dependency.
    from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
        _V3_DEPRECATED_FIXED_KEYS,
        _resolve_adaptive_p_t,
        adaptive_success_criteria,
    )

    deployment_intent = inputs.get("deployment_intent")
    try:
        thresholds, skipped = adaptive_success_criteria(
            n_samples=inputs["n_samples"],
            prevalence=float(inputs["prevalence"]),
            baseline_auc=float(baseline_auc),
            feature_count=inputs["feature_count"],
            regime=inputs.get("regime"),
            deployment_intent=deployment_intent,
            # Issue #866: the materialized split sizes (known only here, at
            # evaluation time) scale the overfit/calibration caps to the
            # sampling-noise floor of the split each metric is measured on.
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "adaptive overlay refused inputs (%s); leaving success_criteria unchanged",
            exc,
        )
        return success_criteria

    overlaid: Dict[str, Any] = dict(success_criteria)
    # Remove keys that adaptive skipped (v3 invariant: skipped ⇒ ABSENT).
    for key in skipped:
        overlaid.pop(key, None)
    # Drop v3-deprecated fixed gates (precision / F1) so only v3 active
    # gates fire downstream.
    for key in _V3_DEPRECATED_FIXED_KEYS:
        overlaid.pop(key, None)
    # Apply firing thresholds.
    overlaid.update(thresholds)
    # Audit fields.
    overlaid["_adaptive_skipped"] = sorted(skipped)
    overlaid["_adaptive_p_t"] = _resolve_adaptive_p_t(inputs.get("regime"), deployment_intent)
    return overlaid


def _resolve_net_benefit_from_grid(
    test_metrics: Dict[str, Any],
    success_criteria: Dict[str, Any],
) -> Optional[float]:
    """Resolve the v3 ``minimum_net_benefit_at_p_t`` gate via the NB grid.

    The validator emits ``_adaptive_p_t`` on ``success_criteria`` when
    ADAPTIVE_CRITERIA is on; ``_compute_classification_metrics`` (task 05)
    emits ``test_metrics['net_benefit_grid']`` keyed on canonical
    ``"p_t={p_t:.2f}"`` strings. Returns the NB value at the regime's p_t,
    or ``None`` when the grid is absent / does not carry the requested key.

    Returning ``None`` triggers the soft-skip path in the caller — the
    criterion is recorded as ``met=None`` rather than hard-failing on a
    missing metric, because the NB gate only makes sense with a paired
    ``_adaptive_p_t`` audit value.
    """
    p_t = success_criteria.get("_adaptive_p_t")
    if not isinstance(p_t, (int, float)) or isinstance(p_t, bool):
        return None
    grid = test_metrics.get("net_benefit_grid")
    if not isinstance(grid, dict):
        return None
    value = grid.get(f"p_t={float(p_t):.2f}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _ensure_numpy(data: Any) -> Optional[np.ndarray]:
    """Convert data to numpy array if needed.

    Args:
        data: Input data

    Returns:
        Numpy array or None
    """
    if data is None:
        return None

    if isinstance(data, np.ndarray):
        return data

    # Try pandas conversion
    try:
        import pandas as pd

        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values  # type: ignore[no-any-return]
    except ImportError:
        pass

    # Try list/tuple conversion
    if isinstance(data, (list, tuple)):
        return np.array(data)

    return data  # type: ignore[no-any-return]


def _wrap_with_feature_names(data: Any, state: Dict[str, Any]) -> Any:
    """Return X as a DataFrame with the preprocessor's output feature names.

    Falls back to the original `data` when the preprocessor or names are
    unavailable or the column count does not match. See the comment on
    the call site for why this matters for LightGBM 4.x.
    """
    try:
        import pandas as pd
    except ImportError:
        return data
    if data is None or isinstance(data, pd.DataFrame):
        return data
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        return data
    preprocessor = state.get("preprocessor")
    names = None
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = preprocessor.get_feature_names_out()
        except Exception:
            names = None
    if names is None or len(names) != data.shape[1]:
        return data
    return pd.DataFrame(data, columns=list(names))
