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
        n_samples: training-split row count.
        prevalence: positive-class rate, in [0, 1].
        baseline_auc: stratified-dummy baseline AUC (consumed verbatim).
        feature_count: number of features after preprocessing.
        regime: ``"default"``, ``"clean"``, ``"adverse"``, or ``None``
            (treated as ``"clean"``).

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
    thresholds: Dict[str, float] = {}
    skipped: set[str] = set()

    # AUC: regime-keyed, baseline-aware. Default skips entirely (rubric-stress
    # regime; deployer outcome was relocated to a regime-keyed expectation by
    # Codex 2026-04-30 review of pre_phase2_unblockers).
    if effective_regime == "default":
        skipped.add("minimum_auc")
    elif effective_regime == "adverse":
        thresholds["minimum_auc"] = max(0.70, baseline_auc + 0.15)
    else:  # clean
        thresholds["minimum_auc"] = max(0.75, baseline_auc + 0.20)

    # Recall: prevalence-invariant default; looser for adverse / low-prev.
    if effective_regime == "adverse" or prevalence < 0.05:
        thresholds["minimum_recall"] = 0.50
    else:
        thresholds["minimum_recall"] = 0.65

    # NB > 0 gate (v3): replaces precision per Vickers 2006 derivation
    # NB > 0 ⇔ precision > p_t. The threshold is fixed at 0.0; the regime-
    # keyed cost ratio enters via the audit field ``_adaptive_p_t`` set by
    # the validator. Always fires — at adverse p_t=0.05 the gate equates to
    # precision > 0.05 which any non-degenerate classifier clears.
    thresholds["minimum_net_benefit_at_p_t"] = 0.0

    # MCC sanity gate (v3): replaces F1 per Chicco-Jurman 2020. Regime-keyed
    # to compensate for prevalence-deflation curve [Chen 2024].
    if effective_regime == "adverse" or prevalence < 0.05:
        thresholds["minimum_mcc"] = 0.20
    elif effective_regime == "default":
        thresholds["minimum_mcc"] = 0.35
    else:  # clean (or None ⇒ clean)
        thresholds["minimum_mcc"] = 0.45

    # Calibration quality (v3, van Calster 2019 "moderate calibration"):
    # slope ∈ [0.85, 1.15] and |intercept| ≤ 0.30. Regime-independent.
    thresholds["maximum_calibration_slope_deviation"] = 0.15
    thresholds["maximum_calibration_intercept_magnitude"] = 0.30

    # Lift over baseline: skipped when AUC SE proxy is too large for the
    # lift estimate to be stable (S1 fix — Hanley-McNeil-style SE at
    # AUC=0.5, not the v1 ``1/sqrt(N*p*(1-p))`` which was ~3× too large).
    n_pos: float = n_samples * prevalence
    n_neg: float = n_samples * (1.0 - prevalence)
    se_auc: float = 0.5 / max(min(n_pos, n_neg) ** 0.5, 1.0)
    if 2.0 * se_auc < 0.10:
        thresholds["minimum_lift_over_baseline"] = 0.10
    else:
        skipped.add("minimum_lift_over_baseline")

    # ECE: tighten for N >= 1000 (binomial bin-occupancy noise drops).
    thresholds["maximum_calibration_error"] = 0.05 if n_samples >= 1000 else 0.10

    # Train-val Δ: feature-density step function (S2 fix — replaces the
    # v1 false-Riley linear formula that clipped to a constant).
    fpr: float = feature_count / n_samples
    if fpr <= 1.0 / 50.0:
        thresholds["maximum_train_val_delta"] = 0.03
    elif fpr <= 1.0 / 30.0:
        thresholds["maximum_train_val_delta"] = 0.05
    elif fpr <= 1.0 / 15.0:
        thresholds["maximum_train_val_delta"] = 0.07
    else:
        thresholds["maximum_train_val_delta"] = 0.10

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
                effective_regime = regime if regime in _V3_REGIME_P_T else "clean"
                success_criteria["_adaptive_p_t"] = _V3_REGIME_P_T[effective_regime]
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
