"""Success criteria definition and validation for scope_definer.

This module defines measurable success criteria and validates constraints.

ADAPTIVE_CRITERIA — PRODUCTION DEFAULT (flipped 2026-06-02). The v3 adaptive
success-criteria engine (``adaptive_success_criteria()``) is now the
platform default: ``ADAPTIVE_CRITERIA`` defaults to ``"true"``. Every
binary-classification model-quality evaluation is governed by the v3
contract unless a caller explicitly sets ``ADAPTIVE_CRITERIA=false``.

The v3 (Option C) contract DROPS ``minimum_precision`` and ``minimum_f1``
(Van Calster et al. 2025, Lancet Digital Health) and ADDS regime / N /
baseline-keyed net-benefit (DCA), MCC, and calibration slope / intercept
gates. The fixed Apr-26-baseline scheme (``minimum_auc`` /
``minimum_precision`` / ``minimum_recall`` / ``minimum_f1``) remains
reachable as the explicit opt-OUT path.

ROLLBACK: set ``ADAPTIVE_CRITERIA=false`` to revert the entire platform to
the fixed-threshold scheme. Full design contract at
``.claude/plans/adaptive_success_criteria/01-design.md``.
"""

import logging
import math
import os
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Safe defaults applied when ``performance_requirements`` carries a ``None``
# threshold for a binary-classification expected criterion. The set is the
# v2 / Apr-26-baseline contract: under flag-OFF (fixed mode) all four keys
# remain present in ``success_criteria``. Adaptive mode (v3) drops
# precision/F1 from the active gates via ``_V3_DEPRECATED_FIXED_KEYS``.
_BINARY_CLASSIFICATION_DEFAULTS: Dict[str, float] = {
    "minimum_auc": 0.75,
    "minimum_precision": 0.70,
    "minimum_recall": 0.65,
    "minimum_f1": 0.70,
}

# v3 Option C drops these gates per Van Calster et al. 2025 (Lancet Digital
# Health). When the adaptive path succeeds, these legacy keys are popped
# from ``success_criteria`` so only the v3 active gates fire downstream.
_V3_DEPRECATED_FIXED_KEYS: frozenset[str] = frozenset({"minimum_precision", "minimum_f1"})

# Regime → threshold-probability mapping for the v3 NB > 0 gate. The
# decision-maker fixes ``p_t`` once based on the cost ratio between false
# positives and false negatives (``c_FP / c_FN = (1 - p_t) / p_t``). Vickers
# 2019 calibrates these defaults against pharma-cohort scenarios:
# adverse / rare-responder ≈ 19:1, default rubric-stress ≈ 4:1, clean / RWD
# ≈ 7:3. Recorded on ``success_criteria['_adaptive_p_t']`` for audit.
_V3_REGIME_P_T: Dict[str, float] = {
    "adverse": 0.05,
    "default": 0.20,
    "clean": 0.30,
}

# --------------------------------------------------------------------------- #
# Deployment-intent axis (clinical | commercial) — ORTHOGONAL to ``regime``.   #
# --------------------------------------------------------------------------- #
# Recalibrates the deployment bar to the USE CASE rather than the data
# difficulty. A clinical-decision model (published / used at site of care)
# keeps the literature floor AUC 0.75 (Vickers 2019; Cook 2007). A COMMERCIAL
# model (e.g. HCP targeting / propensity, never implemented at site of care)
# uses a lower, separately-cited floor: AUC 0.65 — Hosmer & Lemeshow (2013,
# "Applied Logistic Regression" 3rd ed.: 0.7-0.8 "acceptable") with the
# marketing/propensity convention that >= 0.60-0.65 is operationally useful
# (advertising-propensity AUC distribution: median 0.76, range 0.60-0.95).
# The default is "clinical": the axis NEVER silently loosens the bar — a caller
# must explicitly opt into "commercial".
_VALID_DEPLOYMENT_INTENTS: frozenset[str] = frozenset({"clinical", "commercial"})
DEFAULT_DEPLOYMENT_INTENT: str = "clinical"

# (intent, auc_regime) -> (auc_floor, baseline_lift). ``auc_regime`` collapses
# every non-adverse regime to "clean" (adverse is the only looser data regime).
# Commercial floors are the literature "minimum useful discrimination for
# targeting/propensity" range (AUC >= 0.60; Hosmer & Lemeshow 2013; marketing
# convention that >= 0.60 ranks usefully better than random — advertising
# propensity AUC dist. median 0.76, range 0.60-0.95). A commercial targeting
# model is USED BY ITS RANKING (target the top scored decile), so the floor is
# the usefulness floor, not the clinical "acceptable" (0.75) threshold. Owner-
# ratified 2026-06-07 for the optum-mart commercial cohorts.
_INTENT_AUC_PARAMS: Dict[tuple[str, str], tuple[float, float]] = {
    ("clinical", "clean"): (0.75, 0.20),
    ("clinical", "adverse"): (0.70, 0.15),
    ("commercial", "clean"): (0.60, 0.05),
    ("commercial", "adverse"): (0.58, 0.03),
}
# Commercial lift-over-baseline floor (vs clinical 0.10): a useful targeting
# model need only beat a no-information baseline by a smaller, still-real margin.
_COMMERCIAL_LIFT_FLOOR: float = 0.08

# Commercial net-benefit threshold probability. A false-positive target (a
# wasted outreach touch) costs FAR less than a missed adopter: a converted
# specialist is worth many sales touches. The Vickers net-benefit threshold
# probability encodes that cost ratio as p_t = c_FP / (c_FP + c_FN); at
# p_t = 0.05 the implied ratio is c_FP : c_FN ~= 1 : 19 (one wasted touch is
# ~5% the cost of a missed adopter) — a deliberately CONSERVATIVE commercial
# value (true outreach ratios often run 1:50+). This is an economic-cost
# parameter, NOT a model-quality relaxation: the deployed model must still clear
# NB > 0 at this p_t ON ITS MERITS (the deployer evaluates net benefit on the
# calibrated/deployed probabilities). Clinical runs keep the regime-keyed
# _V3_REGIME_P_T (clean 0.30 ~= 7:3, the clinical decision-cost ratio).
_COMMERCIAL_P_T: float = 0.05


# --------------------------------------------------------------------------- #
# Issue #866 — evaluation-split-size-aware cap scaling.                        #
# --------------------------------------------------------------------------- #
# Evidence (synthetic-CSU gold-standard run, 2026-06-10, known-clean data by
# construction): at validation splits of ~590/337 rows the sampling noise of a
# validation AUC alone is ±0.022–0.030 (Hanley-McNeil), so the fixed 0.03
# train→val cap sat at ~1.3σ of pure noise and blocked statistically clean
# models (measured clean deltas 0.0397/0.0317); the fixed ECE cap (0.05) sat
# BELOW the n_test=254 perfect-calibration 10-bin noise floor (~0.079 mean).
# When the evaluator threads the materialized split sizes, the caps widen to
# the noise floor of the split each metric is measured on; the v3 floors are
# kept as minima, so the scaling can only LOOSEN, never tighten.
#
# AUC anchor for the Hanley-McNeil SE: the SE varies <±10% across the
# plausible 0.60–0.85 working range, so a fixed mid-range anchor keeps the
# cap independent of the model's own (gameable) measured AUC.
_SE_AUC_ANCHOR: float = 0.70
# k in cap = floor ∨ k·SE — a ~97.7% one-sided noise quantile.
_CAP_SE_MULTIPLIER: float = 2.0
# Mirrors the evaluator's calibration_analysis n_bins=10.
_ECE_BINS: int = 10
# van Calster 2019's moderate-calibration band (slope ±0.15, |intercept|
# ≤ 0.30) presumes ~1000+ validation rows (cf. Riley 2021 minimum-n work,
# where SE(slope) at n≈1000 ≈ 0.07–0.08, i.e. the band ≈ 2·SE). Below the
# anchor the band widens by sqrt(anchor/n) to admit the same noise quantile.
_CALIBRATION_ANCHOR_N: int = 1000
# FAIL-CLOSED guards on the scaling (codex R1 HIGH, PR #865): split_enforcer
# admits splits as small as 10 rows, where the unbounded SE scaling produced
# delta cap ≈ 0.35 / slope cap ≈ 0.47 — wide enough to deploy severely
# overfit/miscalibrated models. Below _MIN_EVAL_SUPPORT rows the gated
# metric has no statistical support, so NO loosening is applied (the strict
# v3 floors stay — the deny path); at/above it, the scaled caps are clamped
# to hard ceilings: delta ≤ 0.10 (the loosest historical fpr tier — never
# looser than the pre-#866 maximum), slope ≤ 2× the van Calster band,
# intercept ≤ 0.60, ECE ≤ 0.15.
_MIN_EVAL_SUPPORT: int = 100
_MAX_TRAIN_VAL_DELTA_CAP: float = 0.10
_MAX_SLOPE_DEV_CAP: float = 0.30
_MAX_INTERCEPT_CAP: float = 0.60
_MAX_ECE_CAP: float = 0.15


def _hanley_mcneil_se_auc(auc: float, n: int, prevalence: float) -> float:
    """Hanley & McNeil (1982) standard error of an AUC measured on ``n`` rows.

    ``n_pos``/``n_neg`` are floored at 1 so degenerate inputs return a large
    (maximally conservative-loose) SE instead of dividing by zero.
    """
    n_pos = max(float(n) * prevalence, 1.0)
    n_neg = max(float(n) * (1.0 - prevalence), 1.0)
    q1 = auc / (2.0 - auc)
    q2 = (2.0 * auc * auc) / (1.0 + auc)
    var = (
        auc * (1.0 - auc) + (n_pos - 1.0) * (q1 - auc * auc) + (n_neg - 1.0) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def _normalize_deployment_intent(value: Any) -> str:
    """Return a valid deployment-intent literal (defaults to ``"clinical"``)."""
    return value if value in _VALID_DEPLOYMENT_INTENTS else DEFAULT_DEPLOYMENT_INTENT


def _resolve_adaptive_p_t(regime: Any, deployment_intent: Any) -> float:
    """Resolve the NB>0 threshold probability for (regime, deployment_intent).

    Commercial intent pins a low p_t (cheap false positives in targeting);
    clinical intent keeps the regime-keyed mapping.
    """
    if _normalize_deployment_intent(deployment_intent) == "commercial":
        return _COMMERCIAL_P_T
    effective_regime = regime if regime in _V3_REGIME_P_T else "clean"
    return _V3_REGIME_P_T[effective_regime]


def _adaptive_criteria_enabled() -> bool:
    """Whether the ``ADAPTIVE_CRITERIA`` feature flag is on.

    Reads the env var fresh per call so test patches via ``patch.dict`` are
    observed (importing into a module-level constant would freeze the value
    at import time). The truthy-string set matches the project convention
    used in ``security_middleware.py`` for ``ENABLE_HSTS`` and similar flags.

    PRODUCTION DEFAULT (2026-06-02): the v3 adaptive engine is now the
    platform default — the flag defaults to ``"true"``. The v3 contract
    therefore governs ALL binary-classification model-quality evaluation
    unless a caller explicitly opts OUT with ``ADAPTIVE_CRITERIA=false``.
    The opt-OUT routes back to the fixed Apr-26-baseline thresholds
    (``minimum_auc 0.75`` / ``minimum_precision 0.70`` / ``minimum_recall
    0.65`` / ``minimum_f1 0.70``). ROLLBACK: set ``ADAPTIVE_CRITERIA=false``
    to revert the whole platform to the fixed scheme.
    """
    return os.getenv("ADAPTIVE_CRITERIA", "true").strip().lower() in _TRUTHY


def adaptive_success_criteria(
    n_samples: int,
    prevalence: float,
    baseline_auc: float,
    feature_count: int,
    regime: Optional[Literal["default", "clean", "adverse"]] = None,
    deployment_intent: Optional[Literal["clinical", "commercial"]] = None,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
) -> tuple[Dict[str, float], set[str]]:
    """Return ``(thresholds, skipped)`` per the v3 (Option C) design contract.

    Replaces fixed thresholds in ``_define_classification_criteria`` when
    the ``ADAPTIVE_CRITERIA`` flag is on. Full design contract — including
    formula justifications, citations, and the worked-example table — at
    ``.claude/plans/adaptive_success_criteria/01-design.md``.

    v3 (post-deep-research, Option C) drops ``minimum_precision`` and
    ``minimum_f1`` per Van Calster et al. 2025 (Lancet Digital Health) and
    replaces them with ``minimum_net_benefit_at_p_t`` (DCA-derived,
    operationally equivalent to ``precision > p_t``), ``minimum_mcc``
    (sanity gate per Chicco-Jurman 2020), ``maximum_calibration_slope_deviation``
    and ``maximum_calibration_intercept_magnitude`` (calibration quality per
    van Calster 2019).

    Skipped criteria are ABSENT from the thresholds dict, NEVER present
    with a ``None`` value. The validator stores ``skipped`` on
    ``success_criteria['_adaptive_skipped']``; the evaluator records
    ``met=None`` for those names from the explicit list, not from a
    None-value-in-dict heuristic. This closes the S4 config-typo
    silent-skip vulnerability.

    Args:
        n_samples: full-frame row count (the runner feeds ``len(df)``).
        prevalence: positive-class rate, in [0, 1].
        baseline_auc: stratified-dummy baseline AUC (consumed verbatim).
        feature_count: number of features after preprocessing.
        regime: ``"default"``, ``"clean"``, ``"adverse"``, or ``None``
            (treated as ``"clean"``).
        n_train / n_val / n_test: materialized split sizes, available only at
            evaluation time (the model_trainer overlay threads them; the
            scope-time path passes ``None``). When present, the overfit and
            calibration caps widen to the sampling-noise floor of the split
            each metric is measured on (issue #866): ``train_val_auc_delta``
            is measured train-vs-validation, the calibration metrics on the
            test split. ``None`` preserves the fixed v3 floors exactly.

    Returns:
        ``(thresholds, skipped)``. Invariant:
        ``thresholds.keys().isdisjoint(skipped)``.

    Raises:
        ValueError: when any input is out of range.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"prevalence must be in [0, 1], got {prevalence}")
    if not 0.0 <= baseline_auc <= 1.0:
        raise ValueError(f"baseline_auc must be in [0, 1], got {baseline_auc}")
    if feature_count <= 0:
        raise ValueError(f"feature_count must be > 0, got {feature_count}")

    effective_regime: str = regime or "clean"
    effective_intent: str = _normalize_deployment_intent(deployment_intent)
    thresholds: Dict[str, float] = {}
    skipped: set[str] = set()

    # AUC: intent- AND regime-keyed, baseline-aware. The clinical "default"
    # regime skips the gate entirely (rubric-stress; deployer outcome relocated
    # to a regime-keyed expectation by Codex 2026-04-30). A COMMERCIAL run always
    # carries a real discrimination floor — a targeting model with no AUC bar is
    # meaningless — so it never takes the skip branch. Floors per _INTENT_AUC_PARAMS
    # (clinical 0.75/0.70 Vickers-Cook; commercial 0.65/0.62 Hosmer-Lemeshow).
    if effective_intent == "clinical" and effective_regime == "default":
        skipped.add("minimum_auc")
    else:
        auc_regime = "adverse" if effective_regime == "adverse" else "clean"
        auc_floor, auc_lift = _INTENT_AUC_PARAMS[(effective_intent, auc_regime)]
        thresholds["minimum_auc"] = max(auc_floor, baseline_auc + auc_lift)

    # Recall: looser for commercial / adverse / low-prevalence.
    if effective_intent == "commercial" or effective_regime == "adverse" or prevalence < 0.05:
        thresholds["minimum_recall"] = 0.50
    else:
        thresholds["minimum_recall"] = 0.65

    # NB > 0 gate (v3): replaces precision per Vickers 2006 derivation
    # NB > 0 ⇔ precision > p_t. The threshold is fixed at 0.0; the
    # intent/regime-keyed cost ratio enters via the audit field ``_adaptive_p_t``
    # set by the validator (commercial p_t=0.10, clinical regime-keyed). Always
    # fires — at the low commercial p_t the gate equates to precision > 0.10
    # which a useful targeting model clears.
    thresholds["minimum_net_benefit_at_p_t"] = 0.0

    # MCC sanity gate (v3): replaces F1 per Chicco-Jurman 2020. MCC deflates at
    # low prevalence [Chen 2024]; for COMMERCIAL targeting it deflates further, so
    # the commercial floor is a weak guard (0.10) — discrimination (AUC), lift and
    # net-benefit are the load-bearing commercial gates. Clinical stays regime-keyed.
    if effective_intent == "commercial":
        thresholds["minimum_mcc"] = 0.10
    elif effective_regime == "adverse" or prevalence < 0.05:
        thresholds["minimum_mcc"] = 0.20
    elif effective_regime == "default":
        thresholds["minimum_mcc"] = 0.35
    else:  # clean (or None ⇒ clean)
        thresholds["minimum_mcc"] = 0.45

    # Calibration quality (v3, van Calster 2019 "moderate calibration"):
    # slope ∈ [0.85, 1.15] and |intercept| ≤ 0.30. Regime-independent.
    # Issue #866: both metrics are measured on the TEST split; below the
    # ~1000-row anchor the published band is narrower than its own sampling
    # noise, so it widens by sqrt(anchor/n_test). Floors preserved at/above
    # the anchor (and whenever n_test is unknown).
    # FAIL-CLOSED guard (codex R1): below _MIN_EVAL_SUPPORT test rows the
    # calibration metrics have no statistical support — keep the strict
    # floors; at/above it, clamp the widened band to the hard ceilings.
    cal_scale: float = 1.0
    if n_test is not None and _MIN_EVAL_SUPPORT <= n_test < _CALIBRATION_ANCHOR_N:
        cal_scale = math.sqrt(_CALIBRATION_ANCHOR_N / n_test)
    thresholds["maximum_calibration_slope_deviation"] = min(0.15 * cal_scale, _MAX_SLOPE_DEV_CAP)
    thresholds["maximum_calibration_intercept_magnitude"] = min(
        0.30 * cal_scale, _MAX_INTERCEPT_CAP
    )

    # Lift over baseline: skipped when AUC SE proxy is too large for the
    # lift estimate to be stable (S1 fix — Hanley-McNeil-style SE at
    # AUC=0.5, not the v1 ``1/sqrt(N*p*(1-p))`` which was ~3× too large).
    n_pos: float = n_samples * prevalence
    n_neg: float = n_samples * (1.0 - prevalence)
    se_auc: float = 0.5 / max(min(n_pos, n_neg) ** 0.5, 1.0)
    if 2.0 * se_auc < 0.10:
        thresholds["minimum_lift_over_baseline"] = (
            _COMMERCIAL_LIFT_FLOOR if effective_intent == "commercial" else 0.10
        )
    else:
        skipped.add("minimum_lift_over_baseline")

    # ECE: tighten for N >= 1000 (binomial bin-occupancy noise drops).
    # Issue #866: ECE is measured on the TEST split with B=10 bins; under
    # PERFECT calibration its expectation is already ≈ sqrt(2/π)·SE_bin with
    # SE_bin = 0.5·sqrt(B/n_test) (per-bin binomial noise, p(1−p) ≤ 0.25),
    # so the cap must sit above that noise floor: mean + k·SD, where
    # SD(ECE) = SE_bin·sqrt(1−2/π)/sqrt(B) (mean of B folded-normal |gaps|).
    # FAIL-CLOSED guard (codex R1): no loosening below _MIN_EVAL_SUPPORT
    # test rows; the widened cap is clamped to _MAX_ECE_CAP above it.
    ece_cap: float = 0.05 if n_samples >= 1000 else 0.10
    if n_test is not None and n_test >= _MIN_EVAL_SUPPORT:
        se_bin = 0.5 * math.sqrt(_ECE_BINS / n_test)
        ece_noise_mean = math.sqrt(2.0 / math.pi) * se_bin
        ece_noise_sd = se_bin * math.sqrt(1.0 - 2.0 / math.pi) / math.sqrt(_ECE_BINS)
        ece_cap = max(
            ece_cap, min(ece_noise_mean + _CAP_SE_MULTIPLIER * ece_noise_sd, _MAX_ECE_CAP)
        )
    thresholds["maximum_calibration_error"] = ece_cap

    # Train-val Δ: feature-density step function (S2 fix — replaces the
    # v1 false-Riley linear formula that clipped to a constant).
    fpr: float = feature_count / n_samples
    if fpr <= 1.0 / 50.0:
        delta_cap: float = 0.03
    elif fpr <= 1.0 / 30.0:
        delta_cap = 0.05
    elif fpr <= 1.0 / 15.0:
        delta_cap = 0.07
    else:
        delta_cap = 0.10
    # Issue #866: the delta is |AUC(train) − AUC(validation)|, two AUCs each
    # carrying Hanley-McNeil sampling noise. Under zero true overfit the
    # delta's own SE is sqrt(SE²(n_train)+SE²(n_val)); a cap below k·that SE
    # fails clean models on noise (measured: clean delta 0.0397 at n_val=591
    # vs the fixed 0.03). The feature-density tier stays as the floor.
    # FAIL-CLOSED guard (codex R1): no loosening below _MIN_EVAL_SUPPORT
    # validation rows; the widened cap is clamped to the loosest historical
    # fpr tier (_MAX_TRAIN_VAL_DELTA_CAP) above it.
    if n_val is not None and n_val >= _MIN_EVAL_SUPPORT:
        se_val = _hanley_mcneil_se_auc(_SE_AUC_ANCHOR, n_val, prevalence)
        se_train = (
            _hanley_mcneil_se_auc(_SE_AUC_ANCHOR, n_train, prevalence)
            if n_train is not None and n_train > 0
            else 0.0
        )
        delta_cap = max(
            delta_cap,
            min(_CAP_SE_MULTIPLIER * math.hypot(se_val, se_train), _MAX_TRAIN_VAL_DELTA_CAP),
        )
    thresholds["maximum_train_val_delta"] = delta_cap

    return thresholds, skipped


# v3 active gates managed by ``adaptive_success_criteria()``. Used by
# ``define_success_criteria`` to scrub fixed-mode keys when adaptive skips
# them (e.g., default-regime ``minimum_auc``).
_ADAPTIVE_MANAGED_KEYS: frozenset[str] = frozenset(
    {
        "minimum_auc",
        "minimum_recall",
        "minimum_net_benefit_at_p_t",
        "minimum_mcc",
        "maximum_calibration_slope_deviation",
        "maximum_calibration_intercept_magnitude",
        "minimum_lift_over_baseline",
        "maximum_calibration_error",
        "maximum_train_val_delta",
    }
)


async def define_success_criteria(state: Dict[str, Any]) -> Dict[str, Any]:
    """Define success criteria based on problem type and requirements.

    Creates performance thresholds that model_trainer must meet to pass validation.

    Args:
        state: ScopeDefinerState with problem_type, performance_requirements

    Returns:
        Dictionary with success_criteria, validation_passed, validation_warnings,
        validation_errors
    """
    problem_type = state.get("inferred_problem_type", "binary_classification")
    performance_reqs = state.get("performance_requirements", {})

    # Define criteria based on problem type
    if problem_type in ["binary_classification", "multiclass_classification"]:
        success_criteria = _define_classification_criteria(performance_reqs)
    elif problem_type == "regression":
        success_criteria = _define_regression_criteria(performance_reqs)
    elif problem_type == "causal_inference":
        success_criteria = _define_causal_criteria(performance_reqs)
    elif problem_type == "time_series":
        success_criteria = _define_timeseries_criteria(performance_reqs)
    else:
        success_criteria = _define_classification_criteria(performance_reqs)

    # Add common criteria
    success_criteria["experiment_id"] = state.get("experiment_id", "")
    success_criteria["baseline_model"] = _define_baseline_model(problem_type)
    success_criteria["minimum_lift_over_baseline"] = performance_reqs.get(
        "min_lift", 0.10
    )  # 10% improvement over baseline

    # Stamp deployment_intent (clinical|commercial) at the TOP LEVEL so every
    # downstream consumer can read it reliably regardless of which criteria path
    # runs below: the evaluator's commercial recall-constrained operating point
    # AND post-hoc calibration-method selector, and the model_deployer's
    # model_usefulness gate. The adaptive ``_adaptive_inputs`` stash also carries
    # it, but that stash is path-dependent (only the adaptive branch sets it) and
    # can be dropped on a success_criteria rebuild; this top-level key is not.
    success_criteria["deployment_intent"] = _normalize_deployment_intent(
        state.get("deployment_intent")
    )

    # ADAPTIVE_CRITERIA branch: when the flag is on AND the upstream pipeline
    # state carries dataset characteristics, replace fixed thresholds with
    # adaptive ones from ``adaptive_success_criteria()``. When the flag is on
    # but state is incomplete, fall back to fixed and tag the audit value
    # with the v2 third option ``"adaptive_fallback_to_fixed"`` so the gap
    # is observable.
    flag_on = _adaptive_criteria_enabled()
    criteria_source: str = "fixed"

    if flag_on and problem_type == "binary_classification":
        n_samples_raw = state.get("n_samples")
        prevalence_raw = state.get("prevalence")
        baseline_auc_raw = state.get("baseline_auc")
        feature_count_raw = state.get("feature_count")
        regime_raw = state.get("regime")
        deployment_intent = _normalize_deployment_intent(state.get("deployment_intent"))

        # The four PRE-EVAL inputs are derivable at scope-definition time
        # (the runner reads them off the synthetic / RWD dataframe). The
        # fifth input ``baseline_auc`` requires a trained dummy classifier
        # and is computed inside the evaluator. The validator therefore
        # supports two modes:
        #   - eager (4 inputs incl. baseline_auc): rare; primarily unit
        #     tests that inject baseline_auc directly. Computes thresholds
        #     here and applies the v3 invariant.
        #   - stashed (3 pre-eval inputs, no baseline_auc): production
        #     path. Validator records ``_adaptive_inputs`` and tags
        #     ``criteria_source="adaptive"``; the evaluator overlay
        #     ``_apply_adaptive_criteria_overlay`` computes thresholds at
        #     eval time using the live ``baseline_test_auc``.
        pre_eval_complete = (
            isinstance(n_samples_raw, int)
            and not isinstance(n_samples_raw, bool)
            and isinstance(prevalence_raw, (int, float))
            and not isinstance(prevalence_raw, bool)
            and isinstance(feature_count_raw, int)
            and not isinstance(feature_count_raw, bool)
        )
        regime: Optional[Literal["default", "clean", "adverse"]] = (
            regime_raw if regime_raw in ("default", "clean", "adverse") else None
        )

        if (
            pre_eval_complete
            and isinstance(baseline_auc_raw, (int, float))
            and not isinstance(baseline_auc_raw, bool)
        ):
            # Eager call (rare — only happens in unit tests that inject
            # baseline_auc directly). The pre_eval_complete guard above
            # narrows the *_raw types but mypy can't track the multi-
            # condition narrowing, so cast the validated values.
            assert isinstance(n_samples_raw, int)
            assert isinstance(prevalence_raw, (int, float))
            assert isinstance(feature_count_raw, int)
            try:
                thresholds, skipped = adaptive_success_criteria(
                    n_samples=n_samples_raw,
                    prevalence=float(prevalence_raw),
                    baseline_auc=float(baseline_auc_raw),
                    feature_count=feature_count_raw,
                    regime=regime,
                    deployment_intent=deployment_intent,
                )
                # v3 invariant: skipped criteria are ABSENT from
                # success_criteria, never None-valued.
                for key in skipped:
                    success_criteria.pop(key, None)
                # Drop the v3-deprecated fixed gates (precision / F1) so
                # only v3 active gates fire downstream when adaptive
                # succeeds. Under fixed mode (or adaptive_fallback_to_fixed)
                # these keys remain in success_criteria — preserving
                # Apr-26-baseline reproducibility.
                for key in _V3_DEPRECATED_FIXED_KEYS:
                    success_criteria.pop(key, None)
                # Apply firing thresholds (and v3 new keys: NB / MCC /
                # calibration / train_val / ECE).
                success_criteria.update(thresholds)
                success_criteria["_adaptive_skipped"] = sorted(skipped)
                success_criteria["_adaptive_p_t"] = _resolve_adaptive_p_t(regime, deployment_intent)
                criteria_source = "adaptive"
            except ValueError as exc:
                logger.warning(
                    "adaptive_success_criteria refused state inputs (%s); "
                    "falling back to fixed thresholds",
                    exc,
                )
                criteria_source = "adaptive_fallback_to_fixed"
        elif pre_eval_complete:
            # Production stash path: defer the adaptive computation to the
            # evaluator overlay (which has live baseline_test_auc). The
            # leading underscore on ``_adaptive_inputs`` marks it as an
            # audit field — the evaluator's underscore-prefix skip in
            # ``_check_success_criteria`` filters it from the per-criterion
            # loop.
            assert isinstance(n_samples_raw, int)
            assert isinstance(prevalence_raw, (int, float))
            assert isinstance(feature_count_raw, int)
            success_criteria["_adaptive_inputs"] = {
                "n_samples": n_samples_raw,
                "prevalence": float(prevalence_raw),
                "feature_count": feature_count_raw,
                "regime": regime,
                "deployment_intent": deployment_intent,
            }
            criteria_source = "adaptive"
        else:
            logger.warning(
                "ADAPTIVE_CRITERIA on but state missing pre-eval inputs "
                "(n_samples=%r, prevalence=%r, feature_count=%r); falling "
                "back to fixed thresholds. Caller (e.g., scripts/run_tier0_test.py) "
                "must inject these.",
                n_samples_raw,
                prevalence_raw,
                feature_count_raw,
            )
            criteria_source = "adaptive_fallback_to_fixed"

    # Audit tag — surfaces in pipeline reports so operators can tell
    # at-a-glance whether a deployer outcome reflects the fixed or adaptive
    # criteria scheme.
    success_criteria["criteria_source"] = criteria_source

    # Deployment-intent stamp (always present, both fixed and adaptive paths) so
    # the deployer's regulatory-eligibility audit can resolve the intent-keyed
    # literature anchor. Defaults to "clinical" — never silently loosened.
    success_criteria["deployment_intent"] = _normalize_deployment_intent(
        state.get("deployment_intent")
    )

    # Validate criteria
    validation_result = _validate_criteria(success_criteria, state)

    return {
        "success_criteria": success_criteria,
        "validation_passed": validation_result["passed"],
        "validation_warnings": validation_result["warnings"],
        "validation_errors": validation_result["errors"],
    }


def _define_classification_criteria(performance_reqs: Dict[str, float]) -> Dict[str, Any]:
    """Define success criteria for classification problems.

    v2 (S4 fix): a ``None`` value for any expected binary-classification
    criterion is treated as a config typo, NOT as an intentional skip.
    Adaptive intentional skips go through ``adaptive_success_criteria()``
    and the ``_adaptive_skipped`` audit field; plain ``None`` values from
    upstream config are warned and replaced with the safe default.
    """
    raw: Dict[str, Any] = {
        "minimum_auc": performance_reqs.get("min_auc", 0.75),
        "minimum_precision": performance_reqs.get("min_precision", 0.70),
        "minimum_recall": performance_reqs.get("min_recall", 0.65),
        "minimum_f1": performance_reqs.get("min_f1", 0.70),
    }
    for name, value in list(raw.items()):
        if value is None:
            short = name.replace("minimum_", "min_")
            logger.warning(
                "performance_requirements.%s=None for binary classification — "
                "this looks like a config typo. Falling back to the safe "
                "default (%s). Use ADAPTIVE_CRITERIA=true if you want "
                "adaptive skipping; do NOT set min_* to None directly.",
                short,
                _BINARY_CLASSIFICATION_DEFAULTS[name],
            )
            raw[name] = _BINARY_CLASSIFICATION_DEFAULTS[name]
    return {
        **raw,
        "minimum_rmse": None,  # Not applicable
        "minimum_r2": None,  # Not applicable
        "minimum_mape": None,  # Not applicable
    }


def _define_regression_criteria(performance_reqs: Dict[str, float]) -> Dict[str, Any]:
    """Define success criteria for regression problems."""
    return {
        "minimum_auc": None,  # Not applicable
        "minimum_precision": None,  # Not applicable
        "minimum_recall": None,  # Not applicable
        "minimum_f1": None,  # Not applicable
        "minimum_rmse": performance_reqs.get("max_rmse", 10.0),  # Lower is better
        "minimum_r2": performance_reqs.get("min_r2", 0.60),
        "minimum_mape": performance_reqs.get("max_mape", 0.20),  # 20% error
    }


def _define_causal_criteria(performance_reqs: Dict[str, float]) -> Dict[str, Any]:
    """Define success criteria for causal inference problems."""
    return {
        "minimum_auc": None,
        "minimum_precision": None,
        "minimum_recall": None,
        "minimum_f1": None,
        "minimum_rmse": performance_reqs.get("max_ate_se", 0.5),  # ATE std error
        "minimum_r2": performance_reqs.get("min_r2", 0.50),
        "minimum_mape": None,
    }


def _define_timeseries_criteria(performance_reqs: Dict[str, float]) -> Dict[str, Any]:
    """Define success criteria for time series problems."""
    return {
        "minimum_auc": None,
        "minimum_precision": None,
        "minimum_recall": None,
        "minimum_f1": None,
        "minimum_rmse": performance_reqs.get("max_rmse", 15.0),
        "minimum_r2": performance_reqs.get("min_r2", 0.55),
        "minimum_mape": performance_reqs.get("max_mape", 0.25),
    }


def _define_baseline_model(problem_type: str) -> str:
    """Define baseline model used by ``minimum_lift_over_baseline``.

    For binary classification, ``minimum_lift_over_baseline`` is computed
    against a stratified-dummy baseline (see
    ``model_trainer.nodes.evaluator._compute_baseline_test_metrics`` —
    Section B of pre_phase2_unblockers plan). The ``"stratified_dummy"``
    label here is metadata; actual computation is wired in the evaluator.
    Other problem types still return a placeholder name pending an actual
    baseline implementation in those evaluators.
    """
    baselines = {
        "binary_classification": "stratified_dummy",
        "multiclass_classification": "stratified_dummy",
        "regression": "linear_regression_baseline",
        "causal_inference": "ols_baseline",
        "time_series": "arima_baseline",
    }
    return baselines.get(problem_type, "random_baseline")


def _validate_criteria(criteria: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate success criteria for consistency and feasibility.

    Returns:
        Dictionary with 'passed' (bool), 'warnings' (List[str]), 'errors' (List[str])
    """
    warnings: List[str] = []
    errors: List[str] = []

    # Check if minimum samples requirement is feasible
    minimum_samples = state.get("scope_spec", {}).get("minimum_samples", 0)
    if minimum_samples < 100:
        warnings.append(
            f"Minimum sample size ({minimum_samples}) is very low. "
            "Consider requiring at least 500 samples for robust training."
        )

    # Check if performance thresholds are realistic
    min_auc = criteria.get("minimum_auc")
    if min_auc and min_auc > 0.95:
        warnings.append(
            f"Minimum AUC ({min_auc}) is very high. May be difficult to achieve in production."
        )

    min_r2 = criteria.get("minimum_r2")
    if min_r2 and min_r2 > 0.90:
        warnings.append(
            f"Minimum R² ({min_r2}) is very high. May be difficult to achieve in real-world data."
        )

    # Check for conflicting constraints
    time_budget = state.get("time_budget_hours")
    if time_budget and time_budget < 1.0:
        warnings.append(
            f"Time budget ({time_budget}h) is very low. "
            "May not be sufficient for proper hyperparameter tuning."
        )

    # Check for required fields
    if not criteria.get("experiment_id"):
        errors.append("experiment_id is required in success criteria")

    if not criteria.get("baseline_model"):
        errors.append("baseline_model must be specified")

    # Passed if no errors
    passed = len(errors) == 0

    return {
        "passed": passed,
        "warnings": warnings,
        "errors": errors,
    }
