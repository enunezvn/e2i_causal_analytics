"""Data sufficiency pre-flight check for the DataPreparer agent.

Implements Phase 1 of the data-sufficiency diagnostics rollout. Computes a
tiered (HARD_FAIL / SOFT_FAIL / PASS / SKIPPED / INCONCLUSIVE) verdict from
the actual data characteristics and the resolved thresholds, then writes the
verdict into the QC report so the existing gate at ``finalize_output`` picks
it up.

Verdict semantics (post PR #462 hotfix):
    HARD_FAIL    — appended to ``blocking_issues``; halts the pipeline.
                    NON-OVERRIDABLE (pharma safety, F8/D5).
    SOFT_FAIL    — appended to ``power_warnings`` (predictive paths) OR to
                    ``blocking_issues`` for causal_inference unless
                    ``scope_spec.sufficiency.force_low_power_run`` is True.
                    For causal_inference ONLY, override flips block→warn AND
                    sets `override_applied=True` + `original_verdict` on the
                    report (F7 audit trail).
    PASS         — report attached to state; no gating action.
    SKIPPED      — F10/F11: emitted when the check is deliberately skipped
                    (use_sample_data=True synthetic harness OR train_df
                    missing OR unknown problem_type). The report records WHY
                    the check skipped; the gate does NOT block.
    INCONCLUSIVE — F6: emitted when the diagnostic itself crashed
                    (uncaught exception). A VALID DataSufficiencyReport is
                    constructed with qc_status='failed' and a blocking entry,
                    so the gate HALTS the pipeline — silent passthrough on
                    a crashed pre-flight is unsafe.

Reads:
    - ``state.train_df``: row count, feature count, target column statistics
    - ``state.target_rate``: binary classification baseline rate (set by
      baseline_computer; this node MUST run after it)
    - ``state.scope_spec``: problem_type, prediction_target,
      sufficiency overrides (``scope_spec.sufficiency.*``)
    - ``state.blocking_issues``: prior blocking issues (preserved & extended)

Writes:
    - ``state.sufficiency_report``: DataSufficiencyReport.model_dump()
    - ``state.blocking_issues``: extended on HARD_FAIL / blocking SOFT_FAIL /
      INCONCLUSIVE
    - ``state.power_warnings``: SOFT_FAIL warnings for predictive paths
    - ``state.qc_status``: set to "failed" when this node adds blocking issues
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from src.utils.power_analysis_lib import (
    PowerCalculationError,
    binary_outcome_power,
    continuous_outcome_power,
    mde_for_sample_size,
    sensitivity_grid,
)
from src.utils.sufficiency_resolver import (
    resolve_absolute_floor,
    resolve_alpha,
    resolve_epv_floor,
    resolve_observational_inflation,
    resolve_power,
    resolve_regression_ratio,
    resolve_target_mde,
    resolve_timeseries_min_n,
)
from src.utils.sufficiency_schemas import (
    DataSufficiencyReport,
    SufficiencyVerdict,
    ThresholdResolution,
)

from ..state import DataPreparerState

logger = logging.getLogger(__name__)


# Verdict thresholds the node uses to classify the data.
# These are not magic numbers — they are derived from the resolver outputs
# (which themselves cite literature in sufficiency_defaults.py).
_EPV_HARD_FAIL_FLOOR = 2  # Vergouwe 2007: EPV < 2 is the severe-problems zone


async def run_sufficiency_check(state: DataPreparerState) -> Dict[str, Any]:
    """Compute the data-sufficiency verdict from loaded training data.

    Args:
        state: DataPreparerState after baseline_computer has run.

    Returns:
        State updates: sufficiency_report, possibly blocking_issues,
        power_warnings, qc_status.
    """
    experiment_id = state.get("experiment_id", "unknown")
    scope_spec = state.get("scope_spec") or {}
    problem_type = _get_scope_value(scope_spec, "problem_type", "binary_classification")

    # F10/F11 (PR #462 hotfix): SKIPPED-on-synthetic-QC-sample. D6 of the
    # rollout plan says the pre-flight "always runs (no skip flag)". The
    # previous implementation short-circuited with `{}` here when
    # ``scope_spec.use_sample_data=True`` — that satisfied the CI-failure
    # workaround that motivated the original carve-out but violated D6 by
    # collapsing skip vs. silent failure into an indistinguishable signal
    # in the audit chain. The fix is to emit a real DataSufficiencyReport
    # with verdict='SKIPPED' so:
    #   (a) the check always produces an audit-visible record (D6),
    #   (b) the gate does not block (preserving the original intent of the
    #       carve-out — synthetic QC samples HARD_FAIL by construction),
    #   (c) the rationale string explains WHY the check skipped.
    if _get_scope_value(scope_spec, "use_sample_data", False):
        logger.info(
            f"Skipping sufficiency check for experiment {experiment_id}: "
            "scope_spec.use_sample_data=True (data_preparer is running on a "
            "synthetic QC sample, not the training data)."
        )
        return _emit_skipped_report(
            state=state,
            problem_type=problem_type,
            rationale=(
                "Pre-flight skipped: scope_spec.use_sample_data=True "
                "(synthetic QC harness; the real training data is fed "
                "independently downstream)"
            ),
        )

    logger.info(f"Starting sufficiency check for experiment {experiment_id}")

    try:
        train_df = state.get("train_df")
        if train_df is None or not isinstance(train_df, pd.DataFrame):
            # F10/F11: distinct SKIPPED rationale for missing train_df.
            # Pre-fix returned `{}` so the gate couldn't tell "deliberate
            # skip" from "data not loaded due to upstream node failure".
            logger.warning("train_df missing or not a DataFrame; emitting SKIPPED verdict")
            return _emit_skipped_report(
                state=state,
                problem_type=problem_type,
                rationale=(
                    "Pre-flight skipped: state.train_df is missing or not a "
                    "DataFrame (likely upstream load_data failure; check "
                    "load_data logs)"
                ),
                n_rows=0,
                n_features=0,
            )

        target_column = _get_scope_value(scope_spec, "prediction_target", None)
        user_config = _extract_sufficiency_config(scope_spec)

        # Data characteristics
        n_rows = int(len(train_df))
        n_features = _count_features(train_df, target_column)

        minority_prevalence: Optional[float] = None
        baseline_rate: Optional[float] = None
        sigma_outcome: Optional[float] = None
        zero_event_cohort = False
        if problem_type in ("binary_classification", "multiclass_classification"):
            minority_prevalence = _compute_minority_prevalence(
                train_df, target_column, state.get("target_rate")
            )
            if problem_type == "binary_classification":
                # F12 (PR #462 hotfix): the previous expression
                # `state.get("target_rate") or minority_prevalence` hits
                # Python's truthy-of-zero gotcha — a legitimately observed
                # `target_rate=0.0` (rare-event cohort lost all positive
                # cases during join, or outcome simply never occurred)
                # silently falls through to minority_prevalence. That masks
                # a data-integrity problem as a sample-size problem. We use
                # `is not None` so the zero case is preserved AND we flag it
                # separately as a HARD_FAIL with a distinct rationale —
                # "fix the data, don't pretend sample size is the issue."
                target_rate = state.get("target_rate")
                if target_rate is not None:
                    target_rate_f = float(target_rate)
                    if target_rate_f <= 0.0:
                        zero_event_cohort = True
                    baseline_rate = target_rate_f
                else:
                    baseline_rate = minority_prevalence
        elif problem_type in ("regression", "causal_inference"):
            sigma_outcome = _compute_outcome_sigma(train_df, target_column)

        # F12 fast-fail: zero-event cohort is structurally insufficient;
        # no formula can rescue it. Emit HARD_FAIL with a rationale that
        # names the actual problem instead of misleading the operator
        # toward "increase sample size".
        if zero_event_cohort:
            return _emit_zero_event_hard_fail(
                state=state,
                problem_type=problem_type,
                n_rows=n_rows,
                n_features=n_features,
            )

        # F13 (PR #462 hotfix): resolve alpha + power as NAMED local variables
        # AND pass them as explicit kwargs to every classifier branch. The
        # previous code stuffed them into `resolved` at positions 0 and 1,
        # then read them back via `float(resolved[0].value)` and
        # `float(resolved[1].value)` deep inside each classifier — a fragile
        # contract: a future reorder/insert into `resolved` would silently
        # corrupt the MDE (alpha=0.8 power calculations don't crash, they
        # just produce nonsense). Named locals + explicit kwargs make the
        # dependency visible at every call site.
        alpha_res = resolve_alpha(user_config=user_config)
        power_res = resolve_power(user_config=user_config)
        alpha = float(alpha_res.value)
        power = float(power_res.value)
        resolved: List[ThresholdResolution] = [alpha_res, power_res]

        # Branch on problem_type
        if problem_type in ("binary_classification", "multiclass_classification"):
            verdict, rationale, required_n, mde_at_n, sens_grid, mde_assumption, mde_capped = (
                _classify_classification(
                    n_rows=n_rows,
                    n_features=n_features,
                    minority_prevalence=minority_prevalence,
                    baseline_rate=baseline_rate,
                    user_config=user_config,
                    problem_type=problem_type,
                    resolved=resolved,
                    alpha=alpha,
                    power=power,
                )
            )
        elif problem_type == "regression":
            verdict, rationale, required_n, mde_at_n, sens_grid, mde_assumption, mde_capped = (
                _classify_regression(
                    n_rows=n_rows,
                    n_features=n_features,
                    sigma_outcome=sigma_outcome,
                    user_config=user_config,
                    resolved=resolved,
                    alpha=alpha,
                    power=power,
                )
            )
        elif problem_type == "causal_inference":
            (
                verdict,
                rationale,
                required_n,
                mde_at_n,
                sens_grid,
                mde_assumption,
                mde_capped,
            ) = _classify_causal(
                n_rows=n_rows,
                n_features=n_features,
                baseline_rate=baseline_rate,
                sigma_outcome=sigma_outcome,
                user_config=user_config,
                resolved=resolved,
                alpha=alpha,
                power=power,
            )
        elif problem_type == "time_series":
            verdict, rationale, required_n, mde_at_n, sens_grid, mde_assumption, mde_capped = (
                _classify_timeseries(
                    n_rows=n_rows,
                    n_features=n_features,
                    user_config=user_config,
                    resolved=resolved,
                )
            )
        else:
            # F10/F11: distinct SKIPPED rationale for unknown problem_type.
            # Pre-fix this also returned `{}`, indistinguishable from the
            # other two skip paths in the audit chain.
            logger.warning(f"Unknown problem_type {problem_type!r} — emitting SKIPPED verdict")
            return _emit_skipped_report(
                state=state,
                problem_type=problem_type,
                rationale=(
                    f"Pre-flight skipped: unknown problem_type {problem_type!r}. "
                    f"Add a classifier branch to sufficiency_check.py for the "
                    f"new type or fix scope_spec.problem_type."
                ),
                n_rows=n_rows,
                n_features=n_features,
            )

        # MDE assumption (Strategy B) — surface what the system picked
        if mde_assumption is not None and mde_assumption.get("source") != "user_override":
            logger.warning(
                "target_mde not specified in scope_spec.sufficiency; using "
                f"{mde_assumption.get('source')} default {mde_assumption.get('value')}. "
                f"Override via scope_spec.sufficiency.target_mde."
            )

        report = DataSufficiencyReport(
            verdict=cast(SufficiencyVerdict, verdict),
            verdict_rationale=rationale,
            n_rows=n_rows,
            n_features=n_features,
            problem_type=problem_type,
            minority_prevalence=minority_prevalence,
            baseline_rate=baseline_rate,
            sigma_outcome=sigma_outcome,
            resolved_thresholds=resolved,
            required_n=required_n,
            required_n_rationale=rationale if required_n is not None else None,
            detectable_mde_at_current_n=(mde_at_n["value"] if mde_at_n is not None else None),
            detectable_mde_units=(mde_at_n.get("units") if mde_at_n is not None else None),
            # F14: surface the binary-MDE clamp flag so the report consumer
            # knows the value was statistically capped at the baseline_rate
            # boundary rather than honestly representing a detectable effect.
            detectable_mde_at_n_capped=mde_capped if mde_capped else None,
            sensitivity_grid=sens_grid,
            mde_assumption_used=mde_assumption,
            human_readable_summary=_format_summary(
                verdict, n_rows, required_n, mde_at_n, problem_type
            ),
        )

        return _apply_verdict_to_state(state, report, problem_type, user_config)

    except Exception as e:
        # F6 (PR #462 hotfix): the previous handler returned a malformed
        # ``sufficiency_report`` dict (just {error, verdict}) without
        # populating any required DataSufficiencyReport fields, AND without
        # appending to ``blocking_issues`` or setting ``qc_status='failed'``.
        # The downstream gate at ``finalize_output`` saw no blockers and
        # silently passed the pipeline through to training on a crashed
        # pre-flight — an unsafe failure mode that is exactly what the
        # sufficiency gate exists to prevent.
        #
        # Fix: emit a VALID DataSufficiencyReport with verdict=INCONCLUSIVE,
        # append a blocking_issues entry, and set qc_status='failed' so
        # finalize_output halts the pipeline. The diagnostic failure becomes
        # a HARD problem-fail rather than a silent passthrough.
        logger.error(f"Sufficiency check failed: {e}", exc_info=True)
        return _emit_inconclusive_report(
            state=state,
            problem_type=problem_type,
            exc=e,
        )


# ---------------------------------------------------------------------------
# Verdict computation per problem type
# ---------------------------------------------------------------------------


def _classify_classification(
    *,
    n_rows: int,
    n_features: int,
    minority_prevalence: Optional[float],
    baseline_rate: Optional[float],
    user_config: Optional[Dict[str, Any]],
    problem_type: str,
    resolved: List[ThresholdResolution],
    alpha: float,
    power: float,
) -> tuple[
    str,
    str,
    Optional[int],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    bool,
]:
    floor_res = resolve_absolute_floor(
        user_config=user_config,
        problem_type=problem_type,
        n_features=n_features,
        minority_prevalence=minority_prevalence,
    )
    epv_res = resolve_epv_floor(user_config=user_config, algorithm_family="unknown")
    resolved.extend([floor_res, epv_res])

    abs_floor = int(floor_res.value)
    epv_floor = int(epv_res.value)

    # EPV at current n. Falls back to None when prevalence unknown.
    if minority_prevalence is not None and minority_prevalence > 0 and n_features > 0:
        epv_at_n = (n_rows * minority_prevalence) / n_features
        required_n = math.ceil((epv_floor * n_features) / minority_prevalence)
    else:
        epv_at_n = None
        required_n = abs_floor

    # Detectable MDE (Strategy A) — only for binary
    mde_at_n: Optional[Dict[str, Any]] = None
    sens_grid: Optional[Dict[str, Any]] = None
    mde_capped = False
    if problem_type == "binary_classification" and baseline_rate is not None:
        mde_resolution = resolve_target_mde(
            user_config=user_config,
            outcome_type="binary",
            baseline_rate=baseline_rate,
        )
        resolved.append(mde_resolution)
        mde_assumption: Optional[Dict[str, Any]] = {
            "value": mde_resolution.value,
            "source": mde_resolution.source,
            "citation": mde_resolution.citation,
        }
        try:
            mde_value = mde_for_sample_size(
                n=n_rows,
                alpha=alpha,
                power=power,
                outcome_type="binary",
                baseline_rate=baseline_rate,
            )
            # F14 (PR #462 hotfix): for binary outcomes with small n + small
            # baseline_rate, the asymptotic normal-approximation can return
            # an MDE that exceeds the meaningful detection boundary. For an
            # absolute risk difference the symmetric "MDE you can detect
            # at this baseline" boundary is `min(baseline_rate, 1 -
            # baseline_rate)` — beyond this, the result describes a regime
            # where the formula's approximation has broken down (n too
            # small for CLT to hold at these probabilities). The honest
            # answer is "the asymptotic regime doesn't apply; cannot
            # detect anything meaningful." We clamp to the boundary AND
            # surface a `mde_capped=True` flag so the report consumer can
            # tell "honest small MDE" from "asymptotically invalid; capped".
            cap = min(baseline_rate, 1.0 - baseline_rate)
            if mde_value > cap:
                logger.warning(
                    f"detectable_mde={mde_value:.3f} exceeds boundary "
                    f"min(baseline_rate, 1-baseline_rate)={cap:.3f}; "
                    f"asymptotic normal approximation broke down at n={n_rows}. "
                    f"Clamping to boundary {cap:.3f}."
                )
                mde_value = cap
                mde_capped = True
            mde_at_n = {
                "value": mde_value,
                "units": "absolute_risk_difference",
            }
            sens_grid = sensitivity_grid(
                n=n_rows,
                alpha=alpha,
                power=power,
                outcome_type="binary",
                candidates=[0.05, 0.10, 0.20],
                baseline_rate=baseline_rate,
            )
        except PowerCalculationError as exc:
            logger.warning(f"MDE/sensitivity calc failed: {exc}")
    else:
        mde_assumption = None

    # Verdict
    if n_rows < abs_floor or (epv_at_n is not None and epv_at_n < _EPV_HARD_FAIL_FLOOR):
        verdict = "HARD_FAIL"
        rationale = (
            f"n={n_rows} below absolute floor {abs_floor} "
            f"(or EPV={epv_at_n:.2f} < {_EPV_HARD_FAIL_FLOOR})"
            if epv_at_n is not None
            else f"n={n_rows} below absolute floor {abs_floor}"
        )
    elif n_rows < required_n:
        verdict = "SOFT_FAIL"
        rationale = (
            f"n={n_rows} below recommended n={required_n} (EPV={epv_at_n:.2f} < {epv_floor})"
            if epv_at_n is not None
            else f"n={n_rows} below recommended {required_n}"
        )
    else:
        verdict = "PASS"
        rationale = f"n={n_rows} >= recommended {required_n}"
    # F14: annotate the rationale when the MDE was clamped so audit
    # readers see the caveat without having to inspect the report dict.
    if mde_capped and mde_at_n is not None:
        rationale = (
            f"{rationale}; detectable_mde clamped at boundary "
            f"min(baseline_rate, 1-baseline_rate) — asymptotic normal "
            "approximation invalid at this n"
        )

    return verdict, rationale, required_n, mde_at_n, sens_grid, mde_assumption, mde_capped


def _classify_regression(
    *,
    n_rows: int,
    n_features: int,
    sigma_outcome: Optional[float],
    user_config: Optional[Dict[str, Any]],
    resolved: List[ThresholdResolution],
    alpha: float,
    power: float,
) -> tuple[
    str,
    str,
    Optional[int],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    bool,
]:
    floor_res = resolve_absolute_floor(user_config=user_config, problem_type="regression")
    ratio_res = resolve_regression_ratio(user_config=user_config, algorithm_family="unknown")
    resolved.extend([floor_res, ratio_res])

    abs_floor = int(floor_res.value)
    ratio_floor = int(ratio_res.value)
    required_n = max(abs_floor, ratio_floor * n_features) if n_features > 0 else abs_floor

    mde_resolution = resolve_target_mde(
        user_config=user_config,
        outcome_type="continuous",
        sigma_outcome=sigma_outcome,
    )
    resolved.append(mde_resolution)
    mde_assumption: Optional[Dict[str, Any]] = {
        "value": mde_resolution.value,
        "source": mde_resolution.source,
        "citation": mde_resolution.citation,
    }

    mde_at_n: Optional[Dict[str, Any]] = None
    sens_grid: Optional[Dict[str, Any]] = None
    if n_rows >= 2:
        try:
            # F13: alpha/power are explicit kwargs; no positional indexing.
            mde_at_n = {
                "value": mde_for_sample_size(
                    n=n_rows, alpha=alpha, power=power, outcome_type="continuous"
                ),
                "units": "cohens_d",
            }
            sens_grid = sensitivity_grid(
                n=n_rows,
                alpha=alpha,
                power=power,
                outcome_type="continuous",
                candidates=[0.2, 0.5, 0.8],
            )
        except PowerCalculationError as exc:
            logger.warning(f"MDE calc failed: {exc}")

    sample_ratio = n_rows / n_features if n_features > 0 else float("inf")

    if n_rows < abs_floor:
        verdict = "HARD_FAIL"
        rationale = f"n={n_rows} below absolute floor {abs_floor}"
    elif n_rows < required_n:
        verdict = "SOFT_FAIL"
        rationale = (
            f"n={n_rows} below recommended {required_n} "
            f"(sample/feature ratio {sample_ratio:.1f} < {ratio_floor})"
        )
    else:
        verdict = "PASS"
        rationale = f"n={n_rows} >= recommended {required_n}"

    # F14: regression's `mde_for_sample_size` returns Cohen's d (always
    # finite, no boundary issue) so the clamp doesn't apply here — always
    # False. We thread the field for tuple-shape uniformity across helpers.
    return verdict, rationale, required_n, mde_at_n, sens_grid, mde_assumption, False


def _classify_causal(
    *,
    n_rows: int,
    n_features: int,
    baseline_rate: Optional[float],
    sigma_outcome: Optional[float],
    user_config: Optional[Dict[str, Any]],
    resolved: List[ThresholdResolution],
    alpha: float,
    power: float,
) -> tuple[
    str,
    str,
    Optional[int],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    bool,
]:
    floor_res = resolve_absolute_floor(user_config=user_config, problem_type="causal_inference")
    inflation_res = resolve_observational_inflation(user_config=user_config)
    epv_res = resolve_epv_floor(user_config=user_config, algorithm_family="unknown")
    resolved.extend([floor_res, inflation_res, epv_res])

    abs_floor = int(floor_res.value)
    inflation = float(inflation_res.value)
    epv_floor = int(epv_res.value)

    # Determine MDE + outcome type. Default to binary with synthetic-style
    # baseline_rate=0.50 when neither is detectable.
    if baseline_rate is not None:
        outcome_type = "binary"
        mde_resolution = resolve_target_mde(
            user_config=user_config, outcome_type="binary", baseline_rate=baseline_rate
        )
    elif sigma_outcome is not None:
        outcome_type = "continuous"
        mde_resolution = resolve_target_mde(
            user_config=user_config,
            outcome_type="continuous",
            sigma_outcome=sigma_outcome,
        )
    else:
        outcome_type = "binary"
        mde_resolution = resolve_target_mde(user_config=user_config, outcome_type="binary")
    resolved.append(mde_resolution)
    target_mde = float(mde_resolution.value)
    mde_assumption: Optional[Dict[str, Any]] = {
        "value": target_mde,
        "source": mde_resolution.source,
        "citation": mde_resolution.citation,
    }

    # F15 (PR #462 hotfix): the previous formula was
    # ``required_n = max(abs_floor, int(ceil(inflation*rct.n)) + 2*n_features)``
    # — additive `+ 2*n_features`. For wide panels (n_features=500) that
    # adds 1000 to required_n regardless of MDE, dominating the formula
    # and producing pessimistic "you need 1000+ more rows" warnings that
    # don't depend on the MDE the user actually cares about. The intent
    # was clearly "add headroom for covariate adjustment", but the
    # additive form was the wrong shape. The correct shape is
    # MULTIPLICATIVE via EPV (the dimension where the per-feature cost
    # actually lives): EPV × n_features. We take max(...) so the binding
    # constraint becomes whichever is largest of:
    #   - abs_floor (literature minimum for the problem type)
    #   - inflated rct_n (causal-inflation × the RCT-equivalent n)
    #   - epv_floor × n_features (EPV-driven covariate-cost floor)
    binding_label = "abs_floor"
    rct_required: Optional[int] = None
    epv_floor_required = epv_floor * n_features if n_features > 0 else 0
    try:
        if outcome_type == "binary":
            rate_for_power = baseline_rate if baseline_rate is not None else 0.50
            rct = binary_outcome_power(
                effect_size=target_mde,
                alpha=alpha,
                power=power,
                baseline_rate=rate_for_power,
            )
        else:
            rct = continuous_outcome_power(target_mde, alpha, power)
        rct_required = int(math.ceil(inflation * rct.sample_size))
        candidates_n = {
            "abs_floor": abs_floor,
            "inflated_rct_n": rct_required,
            "epv_floor*n_features": epv_floor_required,
        }
        required_n = max(candidates_n.values())
        # Name which constraint binds so the rationale can call it out.
        binding_label = max(candidates_n.items(), key=lambda kv: kv[1])[0]
    except PowerCalculationError as exc:
        logger.warning(f"Causal power calc failed: {exc}")
        candidates_n = {"abs_floor": abs_floor, "epv_floor*n_features": epv_floor_required}
        required_n = max(candidates_n.values())
        binding_label = max(candidates_n.items(), key=lambda kv: kv[1])[0]

    mde_at_n: Optional[Dict[str, Any]] = None
    sens_grid: Optional[Dict[str, Any]] = None
    mde_capped = False
    if n_rows >= 2:
        try:
            if outcome_type == "binary":
                rate_for_power = baseline_rate if baseline_rate is not None else 0.50
                # F14: apply the same clamp guard as the classification branch
                mde_value = mde_for_sample_size(
                    n=n_rows,
                    alpha=alpha,
                    power=power,
                    outcome_type="binary",
                    baseline_rate=rate_for_power,
                )
                cap = min(rate_for_power, 1.0 - rate_for_power)
                if mde_value > cap:
                    logger.warning(
                        f"causal detectable_mde={mde_value:.3f} exceeds boundary "
                        f"min(rate_for_power, 1-rate_for_power)={cap:.3f}; clamping."
                    )
                    mde_value = cap
                    mde_capped = True
                mde_at_n = {
                    "value": mde_value,
                    "units": "absolute_risk_difference",
                }
                sens_grid = sensitivity_grid(
                    n=n_rows,
                    alpha=alpha,
                    power=power,
                    outcome_type="binary",
                    candidates=[0.05, 0.10, 0.20],
                    baseline_rate=rate_for_power,
                )
            else:
                mde_at_n = {
                    "value": mde_for_sample_size(
                        n=n_rows, alpha=alpha, power=power, outcome_type="continuous"
                    ),
                    "units": "cohens_d",
                }
                sens_grid = sensitivity_grid(
                    n=n_rows,
                    alpha=alpha,
                    power=power,
                    outcome_type="continuous",
                    candidates=[0.2, 0.5, 0.8],
                )
        except PowerCalculationError as exc:
            logger.warning(f"Causal MDE/sensitivity calc failed: {exc}")

    if n_rows < abs_floor:
        verdict = "HARD_FAIL"
        rationale = f"n={n_rows} below absolute floor {abs_floor}"
    elif n_rows < required_n:
        verdict = "SOFT_FAIL"
        rationale = (
            f"n={n_rows} below recommended {required_n} "
            f"(target_mde={target_mde:.3f}, observational inflation×{inflation:.1f}; "
            f"binding constraint={binding_label})"
        )
    else:
        verdict = "PASS"
        rationale = f"n={n_rows} >= recommended {required_n} (binding constraint={binding_label})"

    if mde_capped and mde_at_n is not None:
        rationale = (
            f"{rationale}; detectable_mde clamped at boundary "
            f"min(baseline_rate, 1-baseline_rate) — asymptotic normal "
            "approximation invalid at this n"
        )

    return verdict, rationale, required_n, mde_at_n, sens_grid, mde_assumption, mde_capped


def _classify_timeseries(
    *,
    n_rows: int,
    n_features: int,
    user_config: Optional[Dict[str, Any]],
    resolved: List[ThresholdResolution],
) -> tuple[
    str,
    str,
    Optional[int],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    bool,
]:
    seasonal_period = (user_config or {}).get("seasonal_period")
    ts_res = resolve_timeseries_min_n(
        user_config=user_config,
        seasonal_period=seasonal_period,
        n_features=n_features,
    )
    floor_res = resolve_absolute_floor(user_config=user_config, problem_type="time_series")
    resolved.extend([ts_res, floor_res])

    abs_floor = int(floor_res.value)
    required_n = int(ts_res.value)

    if n_rows < abs_floor:
        verdict = "HARD_FAIL"
        rationale = f"n={n_rows} below absolute floor {abs_floor}"
    elif n_rows < required_n:
        verdict = "SOFT_FAIL"
        rationale = (
            f"n={n_rows} below recommended {required_n} "
            f"(2 seasonal cycles + ARIMA parameter headroom)"
        )
    else:
        verdict = "PASS"
        rationale = f"n={n_rows} >= recommended {required_n}"

    # time_series has no MDE calculation → mde_capped is always False
    return verdict, rationale, required_n, None, None, None, False


# ---------------------------------------------------------------------------
# State integration
# ---------------------------------------------------------------------------


def _apply_verdict_to_state(
    state: DataPreparerState,
    report: DataSufficiencyReport,
    problem_type: str,
    user_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Translate a verdict into qc_report-shaped state updates.

    Decision matrix (post-PR #462 hotfix; D5 + D6 + F7/F8/F9 locked):

    - HARD_FAIL → ALWAYS blocks. NON-OVERRIDABLE.
      Per F8 of the PR #462 hotfix brief: although the rollout-plan D5
      text could be read as "force_low_power_run applies to HARD_FAIL too",
      that reading is medically dangerous — HARD_FAIL means the data is
      STRUCTURALLY insufficient (e.g. n < absolute_floor; EPV < 2; zero
      events). Allowing operators to override that with a single flag in
      pharma regulatory contexts is the kind of safety-vs-convenience
      tradeoff regulators explicitly disallow. The actual contract is:
      force_low_power_run applies to causal_inference SOFT_FAIL ONLY.
      HARD_FAIL requires fixing the underlying data issue.
    - SOFT_FAIL + causal_inference + not force_low_power_run → BLOCKS
      (regulatory safety; observational causal inference at low power
      produces effect estimates that look credible but are not).
    - SOFT_FAIL + causal_inference + force_low_power_run → WARNS only,
      but sets `override_applied=True` + `original_verdict='SOFT_FAIL'`
      on the report (F7 audit trail — regulator/auditor must be able to
      detect the bypass).
    - SOFT_FAIL + predictive → WARNS only (predictive models at moderate
      power are productive; the warning prompts the operator to consider
      learning-curve diagnostics or more data).
    - PASS / SKIPPED → report only, no gating action.
    - INCONCLUSIVE → emitted by the exception handler; carries its own
      blocking entry and qc_status='failed' so the gate halts.
    """
    force_low_power = bool((user_config or {}).get("force_low_power_run", False))

    blocking_issues: List[str] = list(state.get("blocking_issues") or [])
    power_warnings: List[str] = list(state.get("power_warnings") or [])

    is_causal = problem_type == "causal_inference"

    # F7 (PR #462 hotfix): apply the override BEFORE constructing the dump
    # so the override audit trail (override_applied, original_verdict,
    # ` [OVERRIDDEN via force_low_power_run]` rationale suffix) lands in
    # the persisted report. Pre-fix the report's verdict still read
    # 'SOFT_FAIL' even after the override flipped the gate to warn-only,
    # and regulators had no way to detect the bypass.
    override_applied_now = report.verdict == "SOFT_FAIL" and is_causal and force_low_power
    if override_applied_now:
        report.override_applied = True
        report.original_verdict = "SOFT_FAIL"
        report.verdict_rationale = (
            f"{report.verdict_rationale} [OVERRIDDEN via force_low_power_run]"
        )
        logger.warning(
            "Sufficiency OVERRIDE: causal SOFT_FAIL bypassed via "
            "scope_spec.sufficiency.force_low_power_run=True. Original "
            "rationale: %s",
            report.verdict_rationale,
        )

    updates: Dict[str, Any] = {"sufficiency_report": report.model_dump()}

    blocks_pipeline = report.verdict == "HARD_FAIL" or (
        report.verdict == "SOFT_FAIL" and is_causal and not force_low_power
    )

    if blocks_pipeline:
        # F9 (PR #462 hotfix): make the override hint conditional on whether
        # the verdict can actually be overridden. The prior message
        # advertised `force_low_power_run` for both HARD_FAIL and causal
        # SOFT_FAIL, but per F8 the flag only applies to causal SOFT_FAIL.
        # Telling an operator "set force_low_power_run=True" on a HARD_FAIL
        # is misleading guidance — the flag will be silently ignored and
        # the pipeline will keep failing.
        if report.verdict == "HARD_FAIL":
            msg = (
                f"data_sufficiency: HARD_FAIL ({report.verdict_rationale}). "
                f"HARD_FAIL is non-overridable; review and fix the underlying "
                f"data issue (insufficient sample size / EPV / variance)."
            )
        else:
            # Blocking causal SOFT_FAIL (force_low_power_run not set).
            msg = (
                f"data_sufficiency: {report.verdict} ({report.verdict_rationale}). "
                f"Override via scope_spec.sufficiency.force_low_power_run=True "
                f"if intentional (causal SOFT_FAIL only — HARD_FAIL is "
                f"non-overridable)."
            )
        blocking_issues.append(msg)
        updates["blocking_issues"] = blocking_issues
        updates["qc_status"] = "failed"
        logger.warning(f"Sufficiency check BLOCKING: {report.verdict_rationale}")
    elif report.verdict == "SOFT_FAIL":
        msg = f"data_sufficiency: SOFT_FAIL ({report.verdict_rationale})"
        power_warnings.append(msg)
        updates["power_warnings"] = power_warnings
        logger.warning(f"Sufficiency check WARNING: {report.verdict_rationale}")
    else:
        logger.info(f"Sufficiency check PASS: {report.verdict_rationale}")

    return updates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit_skipped_report(
    *,
    state: DataPreparerState,
    problem_type: str,
    rationale: str,
    n_rows: Optional[int] = None,
    n_features: Optional[int] = None,
) -> Dict[str, Any]:
    """F10/F11: build a SKIPPED DataSufficiencyReport state update.

    Used for the three deliberate-skip cases (use_sample_data,
    missing train_df, unknown problem_type). Honors D6 (the check
    always produces an audit-visible record) while preserving the
    intent of each carve-out (gate must NOT block).
    """
    # Map unknown / non-standard problem_type values back into the
    # DataSufficiencyReport ProblemType literal — fall back to
    # binary_classification + record the actual value in the rationale,
    # which already names it. This keeps the schema valid even when the
    # caller passed a typo'd problem_type.
    safe_problem_type = (
        problem_type
        if problem_type
        in (
            "binary_classification",
            "multiclass_classification",
            "regression",
            "causal_inference",
            "time_series",
        )
        else "binary_classification"
    )
    report = DataSufficiencyReport(
        verdict=cast(SufficiencyVerdict, "SKIPPED"),
        verdict_rationale=rationale,
        n_rows=int(n_rows) if n_rows is not None else 0,
        n_features=int(n_features) if n_features is not None else 0,
        problem_type=safe_problem_type,  # type: ignore[arg-type]
        human_readable_summary=f"Verdict: SKIPPED; {rationale}",
    )
    return {"sufficiency_report": report.model_dump()}


def _emit_inconclusive_report(
    *,
    state: DataPreparerState,
    problem_type: str,
    exc: Exception,
) -> Dict[str, Any]:
    """F6: build a VALID INCONCLUSIVE report on diagnostic crash.

    Constructs a populated DataSufficiencyReport (NOT a raw
    {error, verdict} dict), appends a blocking_issues entry, and sets
    qc_status='failed' so the gate at finalize_output halts the
    pipeline. Silent passthrough on a crashed pre-flight is unsafe —
    training on data we never verified is the failure mode the
    sufficiency gate exists to prevent.
    """
    # Best-effort populate from state — these may be missing or wrong,
    # which is fine; the verdict + rationale are the load-bearing
    # signal, not the metrics.
    train_df = state.get("train_df")
    try:
        n_rows = int(len(train_df)) if isinstance(train_df, pd.DataFrame) else 0
    except Exception:
        n_rows = 0
    try:
        if isinstance(train_df, pd.DataFrame):
            n_features = max(0, len(list(train_df.columns)) - 1)
        else:
            n_features = 0
    except Exception:
        n_features = 0
    safe_problem_type = (
        problem_type
        if problem_type
        in (
            "binary_classification",
            "multiclass_classification",
            "regression",
            "causal_inference",
            "time_series",
        )
        else "binary_classification"
    )
    rationale = f"Pre-flight diagnostic failed: {type(exc).__name__}: {str(exc)}"
    report = DataSufficiencyReport(
        verdict=cast(SufficiencyVerdict, "INCONCLUSIVE"),
        verdict_rationale=rationale,
        n_rows=n_rows,
        n_features=n_features,
        problem_type=safe_problem_type,  # type: ignore[arg-type]
        human_readable_summary=f"Verdict: INCONCLUSIVE; {rationale}",
    )
    blocking_issues: List[str] = list(state.get("blocking_issues") or [])
    blocking_issues.append("data_sufficiency: INCONCLUSIVE (diagnostic failed — see logs)")
    return {
        "sufficiency_report": report.model_dump(),
        "blocking_issues": blocking_issues,
        "qc_status": "failed",
    }


def _emit_zero_event_hard_fail(
    *,
    state: DataPreparerState,
    problem_type: str,
    n_rows: int,
    n_features: int,
) -> Dict[str, Any]:
    """F12: build a HARD_FAIL report for the zero-event-cohort case.

    Used when state.target_rate is observed to be exactly 0.0 for a
    binary classification problem — i.e., zero positive cases. The
    canonical fix is to inspect the cohort join (likely lost cases),
    not to "increase sample size", so the rationale calls that out
    explicitly rather than going through the regular EPV/floor path.
    """
    rationale = (
        f"Zero positive cases observed in target column at n={n_rows} — "
        "outcome unobserved or cohort misjoined. Inspect the join "
        "logic before retrying; no sample size can rescue a "
        "structurally absent outcome."
    )
    report = DataSufficiencyReport(
        verdict=cast(SufficiencyVerdict, "HARD_FAIL"),
        verdict_rationale=rationale,
        n_rows=n_rows,
        n_features=n_features,
        problem_type=cast(Any, problem_type),
        baseline_rate=0.0,
        human_readable_summary=f"Verdict: HARD_FAIL; {rationale}",
    )
    blocking_issues: List[str] = list(state.get("blocking_issues") or [])
    blocking_issues.append(
        f"data_sufficiency: HARD_FAIL ({rationale}). HARD_FAIL is "
        f"non-overridable; review and fix the underlying data issue "
        f"(insufficient sample size / EPV / variance)."
    )
    return {
        "sufficiency_report": report.model_dump(),
        "blocking_issues": blocking_issues,
        "qc_status": "failed",
    }


def _get_scope_value(scope_spec: Any, key: str, default: Any) -> Any:
    """Read a key from scope_spec, handling both dict and pydantic shapes."""
    if isinstance(scope_spec, dict):
        return scope_spec.get(key, default)
    getter = getattr(scope_spec, "get", None)
    if callable(getter):
        try:
            value = getter(key)
            if value is not None:
                return value
        except Exception:
            pass
    return getattr(scope_spec, key, default)


def _extract_sufficiency_config(scope_spec: Any) -> Optional[Dict[str, Any]]:
    """Read scope_spec.sufficiency, accepting dict or pydantic model."""
    cfg = _get_scope_value(scope_spec, "sufficiency", None)
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg
    # Pydantic model
    if hasattr(cfg, "model_dump"):
        dumped: Dict[str, Any] = cfg.model_dump(exclude_none=True)
        return dumped
    return None


def _count_features(train_df: pd.DataFrame, target_column: Optional[str]) -> int:
    """Count predictor columns (excludes target)."""
    cols = list(train_df.columns)
    if target_column and target_column in cols:
        cols.remove(target_column)
    return len(cols)


def _compute_minority_prevalence(
    train_df: pd.DataFrame,
    target_column: Optional[str],
    target_rate: Optional[float],
) -> Optional[float]:
    """Minority-class prevalence for classification.

    For binary targets, prefer ``target_rate`` from baseline_computer (already
    normalized). Falls back to direct computation. Returns prevalence of the
    rarest class for multiclass.
    """
    if target_rate is not None:
        return min(float(target_rate), 1.0 - float(target_rate))
    if not target_column or target_column not in train_df.columns:
        return None
    target = train_df[target_column].dropna()
    if len(target) == 0:
        return None
    counts = target.value_counts(normalize=True)
    if len(counts) == 0:
        return None
    return float(counts.min())


def _compute_outcome_sigma(train_df: pd.DataFrame, target_column: Optional[str]) -> Optional[float]:
    """Standard deviation of a continuous outcome."""
    if not target_column or target_column not in train_df.columns:
        return None
    target = train_df[target_column].dropna()
    if len(target) < 2 or not np.issubdtype(target.dtype, np.number):
        return None
    sigma = float(target.std())
    return sigma if sigma > 0 else None


def _format_summary(
    verdict: str,
    n_rows: int,
    required_n: Optional[int],
    mde_at_n: Optional[Dict[str, Any]],
    problem_type: str,
) -> str:
    """Human-readable one-liner for the report consumer."""
    parts = [f"Verdict: {verdict}", f"n={n_rows}"]
    if required_n is not None:
        parts.append(f"recommended_n={required_n}")
    if mde_at_n is not None and mde_at_n.get("value") is not None:
        value = mde_at_n["value"]
        units = mde_at_n.get("units", "")
        parts.append(f"detectable_mde={value:.4f} ({units})")
    parts.append(f"problem_type={problem_type}")
    return "; ".join(parts)
