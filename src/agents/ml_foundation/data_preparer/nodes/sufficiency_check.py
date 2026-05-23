"""Data sufficiency pre-flight check for the DataPreparer agent.

Implements Phase 1 of the data-sufficiency diagnostics rollout. Computes a
tiered (HARD_FAIL / SOFT_FAIL / PASS) verdict from the actual data
characteristics and the resolved thresholds, then writes the verdict into
the QC report so the existing gate at ``finalize_output`` picks it up.

Verdict semantics:
    HARD_FAIL — appended to ``blocking_issues``; halts the pipeline.
    SOFT_FAIL — appended to ``power_warnings`` (predictive paths) OR to
                ``blocking_issues`` for causal_inference unless
                ``scope_spec.sufficiency.force_low_power_run`` is True.
    PASS     — report attached to state; no gating action.

Reads:
    - ``state.train_df``: row count, feature count, target column statistics
    - ``state.target_rate``: binary classification baseline rate (set by
      baseline_computer; this node MUST run after it)
    - ``state.scope_spec``: problem_type, prediction_target,
      sufficiency overrides (``scope_spec.sufficiency.*``)
    - ``state.blocking_issues``: prior blocking issues (preserved & extended)

Writes:
    - ``state.sufficiency_report``: DataSufficiencyReport.model_dump()
    - ``state.blocking_issues``: extended on HARD_FAIL / blocking SOFT_FAIL
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
    logger.info(f"Starting sufficiency check for experiment {experiment_id}")

    try:
        train_df = state.get("train_df")
        if train_df is None or not isinstance(train_df, pd.DataFrame):
            logger.warning("train_df missing or not a DataFrame; skipping sufficiency check")
            return {}

        scope_spec = state.get("scope_spec") or {}
        problem_type = _get_scope_value(scope_spec, "problem_type", "binary_classification")
        target_column = _get_scope_value(scope_spec, "prediction_target", None)
        user_config = _extract_sufficiency_config(scope_spec)

        # Data characteristics
        n_rows = int(len(train_df))
        n_features = _count_features(train_df, target_column)

        minority_prevalence: Optional[float] = None
        baseline_rate: Optional[float] = None
        sigma_outcome: Optional[float] = None
        if problem_type in ("binary_classification", "multiclass_classification"):
            minority_prevalence = _compute_minority_prevalence(
                train_df, target_column, state.get("target_rate")
            )
            if problem_type == "binary_classification":
                baseline_rate = state.get("target_rate") or minority_prevalence
        elif problem_type in ("regression", "causal_inference"):
            sigma_outcome = _compute_outcome_sigma(train_df, target_column)

        resolved: List[ThresholdResolution] = []
        resolved.append(resolve_alpha(user_config=user_config))
        resolved.append(resolve_power(user_config=user_config))

        # Branch on problem_type
        if problem_type in ("binary_classification", "multiclass_classification"):
            verdict, rationale, required_n, mde_at_n, sens_grid, mde_assumption = (
                _classify_classification(
                    n_rows=n_rows,
                    n_features=n_features,
                    minority_prevalence=minority_prevalence,
                    baseline_rate=baseline_rate,
                    user_config=user_config,
                    problem_type=problem_type,
                    resolved=resolved,
                )
            )
        elif problem_type == "regression":
            verdict, rationale, required_n, mde_at_n, sens_grid, mde_assumption = (
                _classify_regression(
                    n_rows=n_rows,
                    n_features=n_features,
                    sigma_outcome=sigma_outcome,
                    user_config=user_config,
                    resolved=resolved,
                )
            )
        elif problem_type == "causal_inference":
            verdict, rationale, required_n, mde_at_n, sens_grid, mde_assumption = _classify_causal(
                n_rows=n_rows,
                n_features=n_features,
                baseline_rate=baseline_rate,
                sigma_outcome=sigma_outcome,
                user_config=user_config,
                resolved=resolved,
            )
        elif problem_type == "time_series":
            verdict, rationale, required_n, mde_at_n, sens_grid, mde_assumption = (
                _classify_timeseries(
                    n_rows=n_rows,
                    n_features=n_features,
                    user_config=user_config,
                    resolved=resolved,
                )
            )
        else:
            logger.warning(f"Unknown problem_type {problem_type!r} — skipping sufficiency check")
            return {}

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
            sensitivity_grid=sens_grid,
            mde_assumption_used=mde_assumption,
            human_readable_summary=_format_summary(
                verdict, n_rows, required_n, mde_at_n, problem_type
            ),
        )

        return _apply_verdict_to_state(state, report, problem_type, user_config)

    except Exception as e:
        logger.error(f"Sufficiency check failed: {e}", exc_info=True)
        return {
            "sufficiency_report": {"error": str(e), "verdict": "INCONCLUSIVE"},
        }


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
) -> tuple[
    str,
    str,
    Optional[int],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
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
            alpha = float(resolved[0].value)
            power = float(resolved[1].value)
            mde_at_n = {
                "value": mde_for_sample_size(
                    n=n_rows,
                    alpha=alpha,
                    power=power,
                    outcome_type="binary",
                    baseline_rate=baseline_rate,
                ),
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

    return verdict, rationale, required_n, mde_at_n, sens_grid, mde_assumption


def _classify_regression(
    *,
    n_rows: int,
    n_features: int,
    sigma_outcome: Optional[float],
    user_config: Optional[Dict[str, Any]],
    resolved: List[ThresholdResolution],
) -> tuple[
    str,
    str,
    Optional[int],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
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
            alpha = float(resolved[0].value)
            power = float(resolved[1].value)
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

    return verdict, rationale, required_n, mde_at_n, sens_grid, mde_assumption


def _classify_causal(
    *,
    n_rows: int,
    n_features: int,
    baseline_rate: Optional[float],
    sigma_outcome: Optional[float],
    user_config: Optional[Dict[str, Any]],
    resolved: List[ThresholdResolution],
) -> tuple[
    str,
    str,
    Optional[int],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    floor_res = resolve_absolute_floor(user_config=user_config, problem_type="causal_inference")
    inflation_res = resolve_observational_inflation(user_config=user_config)
    resolved.extend([floor_res, inflation_res])

    abs_floor = int(floor_res.value)
    inflation = float(inflation_res.value)
    alpha = float(resolved[0].value)
    power = float(resolved[1].value)

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

    # RCT n via power lib
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
        required_n = max(abs_floor, int(math.ceil(inflation * rct.sample_size)) + 2 * n_features)
    except PowerCalculationError as exc:
        logger.warning(f"Causal power calc failed: {exc}")
        required_n = abs_floor

    mde_at_n: Optional[Dict[str, Any]] = None
    sens_grid: Optional[Dict[str, Any]] = None
    if n_rows >= 2:
        try:
            if outcome_type == "binary":
                rate_for_power = baseline_rate if baseline_rate is not None else 0.50
                mde_at_n = {
                    "value": mde_for_sample_size(
                        n=n_rows,
                        alpha=alpha,
                        power=power,
                        outcome_type="binary",
                        baseline_rate=rate_for_power,
                    ),
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
            f"(target_mde={target_mde:.3f}, observational inflation×{inflation:.1f})"
        )
    else:
        verdict = "PASS"
        rationale = f"n={n_rows} >= recommended {required_n}"

    return verdict, rationale, required_n, mde_at_n, sens_grid, mde_assumption


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

    return verdict, rationale, required_n, None, None, None


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

    Decision matrix (D5 + D6 locked):
    - HARD_FAIL → always blocks
    - SOFT_FAIL + causal_inference + not force_low_power_run → blocks
    - SOFT_FAIL + predictive (or override set) → warning, proceeds
    - PASS → report only
    """
    force_low_power = bool((user_config or {}).get("force_low_power_run", False))

    blocking_issues: List[str] = list(state.get("blocking_issues") or [])
    power_warnings: List[str] = list(state.get("power_warnings") or [])
    updates: Dict[str, Any] = {"sufficiency_report": report.model_dump()}

    is_causal = problem_type == "causal_inference"
    blocks_pipeline = report.verdict == "HARD_FAIL" or (
        report.verdict == "SOFT_FAIL" and is_causal and not force_low_power
    )

    if blocks_pipeline:
        msg = (
            f"data_sufficiency: {report.verdict} ({report.verdict_rationale}). "
            f"Override via scope_spec.sufficiency.force_low_power_run=True if intentional."
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
