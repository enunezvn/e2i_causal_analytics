#!/usr/bin/env python3
"""Manual step-by-step runner for Tier 0 MLOps workflow test.

This script executes each agent in the Tier 0 pipeline individually
with detailed output and verification between steps.

Usage:
    # Run full pipeline (synthetic data)
    python scripts/run_tier0_test.py

    # Run with real-world data
    python scripts/run_tier0_test.py --data-dir data/rwd/csu --brand competitor --target treatment_initiated --indication "Chronic Spontaneous Urticaria (CSU)"

    # Run specific step (1-8)
    python scripts/run_tier0_test.py --step 3

    # Run with MLflow enabled
    python scripts/run_tier0_test.py --enable-mlflow

    # Dry run (show what would be done)
    python scripts/run_tier0_test.py --dry-run

    # Run with BentoML model serving verification (requires step 5+7)
    python scripts/run_tier0_test.py --include-bentoml

    # Run steps 4-8 with BentoML serving (recommended for full flow validation)
    python scripts/run_tier0_test.py --step 4 --include-bentoml

Prerequisites:
    - On droplet: cd /opt/e2i_causal_analytics && source .venv/bin/activate
    - API running (port 8000)
    - MLflow running (port 5000, optional)
    - Opik running (port 5173/8080, optional)
    - BentoML installed (for --include-bentoml flag)

Author: E2I Causal Analytics Team
"""

import argparse
import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import time as time_module
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
# This provides ANTHROPIC_API_KEY, SUPABASE_ANON_KEY, and other secrets
load_dotenv(PROJECT_ROOT / ".env")

# Configure MLflow tracking URI for model artifact storage
# This ensures model_uri is properly generated during model training
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# Configure Supabase URL for database persistence
# Self-hosted Supabase runs on port 54321 (internal Docker network uses localhost)
# Always override: .env may contain a cloud URL but the Tier 0 test targets the local instance
os.environ["SUPABASE_URL"] = "http://localhost:54321"

# Disable Opik by default for local testing — OpikConnector reads OPIK_ENABLED
# from environment (defaults to "true"), so the CONFIG.enable_opik Python flag
# alone is insufficient. Override via --enable-opik flag.
if not os.environ.get("OPIK_ENABLED"):
    os.environ["OPIK_ENABLED"] = "false"


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TestConfig:
    """Test configuration."""

    brand: str = "Kisqali"
    problem_type: str = "binary_classification"
    target_outcome: str = "discontinuation_flag"
    indication: str = "HR+/HER2- breast cancer"
    hpo_trials: int = 10
    min_eligible_patients: int = 30
    # Harness cohort QC gate threshold on ``data_quality_score`` (CONFIG-driven,
    # not a hardcoded constant). Field-adaptive: a NO-OP when the frame carries
    # no ``data_quality_score`` (the adapter already did the real cohorting).
    cohort_min_data_quality: float = 0.5
    min_auc_threshold: float = 0.55
    # Dynamic AUC gate (Tier C): when True the AUC check additionally requires
    # the bootstrap CI lower bound to exceed the 0.5 no-skill floor (model is
    # SIGNIFICANTLY better than chance), not just the point estimate. Default
    # False preserves the point-floor behavior; the bootstrap CI is surfaced
    # either way. Toggle via ``--auc-significance-gate``.
    auc_gate_require_significance: bool = False
    # Minimum recall for minority class - a model that predicts all 0s is useless
    min_minority_recall: float = 0.10  # At least 10% of actual positives must be found
    min_minority_precision: float = 0.05  # At least 5% of predicted positives should be correct
    enable_mlflow: bool = True  # MLflow must be enabled for model_uri to be generated
    enable_opik: bool = False
    # Minimum viable samples per split (forwarded into ModelTrainerState).
    # Default 10 matches split_enforcer's internal default; override via
    # --min-samples-per-split for small-cohort RWD runs (e.g., Optum n=47).
    min_samples_per_split: int = 10
    # Tier D — config-exposed discovery / champion knobs (were hardcoded magic
    # numbers). Defaults preserve historical behavior.
    feature_min_non_null_frac: float = 0.5  # drop columns more than this fraction null
    feature_max_cardinality: int = 50  # categorical high-cardinality cap
    auc_tie_band: float = 0.01  # champion discrimination-tie band (mirrors _AUC_TIE_BAND)
    train_alternatives: bool = True  # False (--single-model) skips Step 5b alt training
    # Sampling-frame drift gate (Phase-1 Task 1.3): when the worst per-column
    # drift in ``sampling_frame_audit_report["max_drift_score"]`` exceeds this
    # threshold, the runner records a failed ``SAMPLING FRAME AUDIT`` step
    # alongside (and independent of) the QC gate. Mirrored into ``scope_spec``
    # as ``sampling_frame_max_drift`` so the audit node sees the same value.
    sampling_frame_max_drift: float = 0.3


CONFIG = TestConfig()


# =============================================================================
# CHAMPION SELECTION — calibration-aware tiebreak among discrimination-ties
# =============================================================================
#
# Step 5b trains the primary model_candidate plus every alternative_candidate
# from model_selector, then must pick ONE champion to deploy. The historical
# rule was a strict ``max(..., key=auc_roc)`` argmax. When several candidates
# are *discrimination-tied* (their test AUCs are within measurement noise of
# the best), a strict argmax picks an ARBITRARY tie member — which can be a
# worse-calibrated model. Calibration quality is a genuine deployment virtue
# (van Calster 2019): among models that rank cases equally well, the one whose
# predicted probabilities are closest to the truth is strictly better to ship.
#
# Policy: select the highest-AUC candidate; then AMONG candidates whose AUC is
# within ``_AUC_TIE_BAND`` of that maximum (a genuine discrimination tie),
# prefer the one with the lowest deploy-calibrated calibration slope deviation
# (``|slope - 1|``, the metric the v3 ``maximum_calibration_slope_deviation``
# gate judges). This NEVER sacrifices discrimination (the candidate must still
# be inside the AUC tie-band of the best) and always moves toward better
# calibration, so it cannot regress accuracy/ranking — it only breaks ties in
# the deployment-quality direction. Candidates lacking a finite slope
# deviation sort last so a measurable tie member always wins over an
# unmeasurable one.
#
# The band is a small absolute AUC delta: AUC differences below this are not
# statistically meaningful at tier0 test-split sizes, so treating them as a
# tie (and deciding on calibration instead) is the principled choice.
_AUC_TIE_BAND = 0.01


def _calibration_slope_deviation_of(result: dict | None) -> float:
    """Return the deploy-calibrated calibration slope deviation for a trained
    candidate's result dict, or ``+inf`` when it is missing / non-finite.

    The evaluator writes ``calibration_slope_deviation = |slope - 1|`` into
    ``test_metrics`` on the DEPLOYED (post-hoc calibrated, #633) probabilities,
    which is exactly what the v3 ``maximum_calibration_slope_deviation`` gate
    reads. Returning ``+inf`` for missing/NaN values makes such candidates sort
    LAST in the tiebreak (a measurable tie member is always preferred), so the
    tiebreak degrades gracefully to the legacy AUC argmax when no candidate
    exposes a usable slope deviation.
    """
    if not isinstance(result, dict):
        return float("inf")
    tm = result.get("test_metrics")
    if not isinstance(tm, dict):
        return float("inf")
    dev = tm.get("calibration_slope_deviation")
    try:
        dev_f = float(dev)
    except (TypeError, ValueError):
        return float("inf")
    if math.isnan(dev_f) or math.isinf(dev_f):
        return float("inf")
    return abs(dev_f)


# Deployability-aware champion selection (owner-ratified 2026-06-07): a model
# that PASSES the genuine deployment quality gates (good calibration AND not
# overfit) is preferred over a raw-AUC winner that would be BLOCKED by those
# gates. The raw-AUC winner is often an overfit gradient booster whose train-
# inflated AUC edges out a cleaner linear model that actually deploys — picking
# it strands the cohort.
#
# Synthetic-CSU persistence finding (2026-06-10): the original implementation
# read ``overfitting_severity`` — a key NO producer ever sets on trainer result
# dicts — so the deployable pool was always empty, the partition silently fell
# back to the full pool, and the calibration tiebreak picked a severely overfit
# XGBoost (train→val Δ 0.1873) over a deployable LR 0.002 AUC below it. The
# history entries now carry the candidate's own EVALUATED v3 gate results
# (``success_criteria_results['maximum_train_val_delta'/'maximum_calibration_
# slope_deviation']``, which respect the #866 split-size-scaled caps); the
# legacy fields remain as fallback for entries that lack them.
_DEPLOY_MAX_SLOPE_DEV: float = 0.15


def _candidate_history_entry(
    algorithm: str,
    result: dict | None,
    *,
    is_primary: bool,
    model_usefulness: "str | None" = None,
) -> dict:
    """Build a Step 5b comparison-history entry from a trained candidate's
    result dict, carrying the EVALUATED deployability gate outcomes
    (``_select_champion`` consumes them; see the module comment above)."""
    result = result if isinstance(result, dict) else {}
    criteria_results = result.get("success_criteria_results")
    criteria_results = criteria_results if isinstance(criteria_results, dict) else {}
    return {
        "algorithm": algorithm,
        "auc_roc": result.get("auc_roc", 0) or 0,
        # Deploy-calibrated |slope-1| for the calibration-aware tiebreak
        # among discrimination-ties (see _select_champion).
        "calibration_slope_deviation": _calibration_slope_deviation_of(result),
        "model_usefulness": (
            model_usefulness
            if model_usefulness is not None
            else result.get("model_usefulness", "unknown")
        ),
        # Evaluated v3 gate outcomes (True/False; None = not evaluated).
        "overfit_gate_met": criteria_results.get("maximum_train_val_delta"),
        "slope_gate_met": criteria_results.get("maximum_calibration_slope_deviation"),
        # Legacy field kept for fallback semantics in _is_deployable.
        "overfitting_severity": result.get("overfitting_severity"),
        "is_primary": is_primary,
    }


def _select_champion(comparison_history: list[dict]) -> dict:
    """Pick the champion from Step 5b candidates.

    Deployability-aware, then calibration-aware:

    0. Partition candidates into DEPLOYABLE (calibration_slope_deviation
       <= ``_DEPLOY_MAX_SLOPE_DEV`` AND overfitting_severity == "none") vs not.
       If ANY candidate is deployable, restrict the pool to deployable ones —
       so a deployable model that clears discrimination beats an overfit /
       miscalibrated higher-AUC one (owner-ratified 2026-06-07). If NONE are
       deployable (or candidates carry no quality fields, e.g. legacy callers),
       the pool is all candidates → legacy behaviour is preserved.
    1. Within the pool, find the maximum test AUC.
    2. Among candidates whose AUC is within ``auc_tie_band`` of that maximum
       (a genuine discrimination tie), pick the lowest deploy-calibrated
       ``calibration_slope_deviation`` (best calibration).

    Stage 1 never sacrifices discrimination *within the deployable pool*; stage
    2 decides among discrimination-equals toward better calibration. With a
    single candidate, or when quality fields are unavailable, this reduces to
    the legacy AUC argmax.
    """
    if not comparison_history:
        raise ValueError("comparison_history is empty; no champion to select")

    def _is_deployable(h: dict) -> bool:
        # Prefer the candidate's own EVALUATED v3 gate outcomes (set by
        # _candidate_history_entry from success_criteria_results — these
        # respect the #866 split-size-scaled caps). The legacy fields are
        # the fallback for entries that lack them: overfitting_severity is
        # "none" only when train-val AUC delta is within the overfit gate
        # band; absent (legacy candidate dicts) → unknown → NOT deployable,
        # which makes the pool fall back to all candidates.
        slope_gate = h.get("slope_gate_met")
        if slope_gate is not None:
            cal_ok = bool(slope_gate)
        else:
            slope_dev = h.get("calibration_slope_deviation")
            cal_ok = isinstance(slope_dev, (int, float)) and slope_dev <= _DEPLOY_MAX_SLOPE_DEV
        overfit_gate = h.get("overfit_gate_met")
        if overfit_gate is not None:
            overfit_ok = bool(overfit_gate)
        else:
            overfit_ok = h.get("overfitting_severity") == "none"
        return cal_ok and overfit_ok

    pool = [h for h in comparison_history if _is_deployable(h)] or comparison_history
    best_auc = max((h.get("auc_roc", 0) or 0) for h in pool)
    tie_band = CONFIG.auc_tie_band  # Tier D: config-exposed (default mirrors _AUC_TIE_BAND)
    tied = [h for h in pool if (best_auc - (h.get("auc_roc", 0) or 0)) <= tie_band]
    # Lowest slope deviation wins; ties on calibration fall back to highest AUC
    # (then stable order) so selection stays deterministic.
    return min(
        tied,
        key=lambda h: (
            h.get("calibration_slope_deviation", float("inf")),
            -(h.get("auc_roc", 0) or 0),
        ),
    )


def _auc_ci_from_result(result: dict) -> "tuple[float, float] | None":
    """Pull the bootstrap AUC confidence interval ``(lower, upper)`` the
    evaluator computed (``evaluator.py::_compute_bootstrap_ci`` → result
    ``confidence_interval['auc']``), or ``None`` when unavailable/malformed.

    This is the measured uncertainty the user actually wants to SEE — surfaced
    instead of trusting a single point AUC against a hardcoded constant.
    """
    ci = (result or {}).get("confidence_interval") or {}
    auc_ci = ci.get("auc")
    if (
        isinstance(auc_ci, (list, tuple))
        and len(auc_ci) == 2
        and all(isinstance(x, (int, float)) for x in auc_ci)
    ):
        return float(auc_ci[0]), float(auc_ci[1])
    return None


def _auc_gate_verdict(
    auc: "float | None",
    auc_ci: "tuple[float, float] | None",
    min_auc: float,
    *,
    require_significance: bool,
) -> "tuple[bool, str]":
    """Decide the AUC acceptance verdict — dynamic and CI-aware.

    Baseline: the point AUC must meet the configured floor ``min_auc``.
    When ``require_significance`` (the CI-mode gate, Tier C), the model must
    ALSO be SIGNIFICANTLY better than the 0.5 no-skill floor — the bootstrap CI
    LOWER bound must exceed 0.5. This ties the gate to the measured uncertainty
    rather than a point estimate alone, which is exactly the confidence-interval
    question for a rare-event cohort. Returns ``(passed, human_detail)``.
    """
    if not auc:
        return False, "no AUC"
    point_ok = auc >= min_auc
    ci_str = f" [95% CI {auc_ci[0]:.3f}-{auc_ci[1]:.3f}]" if auc_ci else ""
    if not require_significance:
        return point_ok, (f"AUC {auc:.3f}{ci_str} {'>=' if point_ok else '<'} floor {min_auc}")
    if auc_ci is None:
        return False, (f"AUC {auc:.3f}: significance gate ON but no bootstrap CI available")
    sig_ok = auc_ci[0] > 0.5
    return (point_ok and sig_ok), (
        f"AUC {auc:.3f}{ci_str}: "
        f"{'significant (CI>0.5)' if sig_ok else 'NOT significant (CI<=0.5)'}, "
        f"floor {min_auc} {'met' if point_ok else 'unmet'}"
    )


def _fit_categorical_onehot(X: "pd.DataFrame", cat_cols: list) -> "tuple[pd.DataFrame, dict]":
    """One-hot encode nominal categoricals into a faithful, calibratable feature space.

    Ordinal/integer codes impose a FALSE magnitude order on NOMINAL categories.
    The downstream ``ModelTrainerPreprocessor`` only one-hots *object*-dtype
    columns, so once the harness pre-encodes categoricals to integer codes the
    preprocessor sees them as numeric and never fixes them — distorting the
    LINEAR champion's probabilities so its post-Platt calibration slope
    deviation fails the gate (disc cohort: ~0.18 ordinal vs ~0.07 one-hot, same
    data/splits, AUC unchanged). One-hot lets the deployable linear model ship;
    tree models consume one-hot fine.

    Returns ``(X_encoded, info)``; ``info`` carries the fitted encoder + the
    produced column names so SHAP / validation re-apply reproduce the EXACT
    trained feature space via :func:`_apply_categorical_onehot`.

    Issue #773 (W2 residue beyond PR #913): ``get_feature_names_out`` embeds
    raw category VALUES in the names, so nominal values like ``"<50"`` /
    ``">65"`` (SampleDataGenerator ``age_group``) produce column names XGBoost
    hard-rejects at fit time (``feature_names must not contain [, ] or <``) —
    killing every Step-5b XGBoost alternative. PR #913 sanitized the
    data_preparer's one-hot site; THIS harness-level encoder is a second,
    independent name source and must sanitize identically (the shared
    ``sanitize_feature_names`` docstring requires all re-derivation paths to
    reuse it). ``_apply_categorical_onehot`` replays the stored
    ``onehot_columns`` so sanitizing here keeps fit/re-apply consistent.
    """
    if not cat_cols:
        return X, {"encoder": None, "columns": [], "onehot_columns": [], "method": "onehot"}
    from sklearn.preprocessing import OneHotEncoder

    from src.agents.ml_foundation.data_preparer.nodes.data_transformer import (
        sanitize_feature_names,
    )

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    arr = ohe.fit_transform(X[cat_cols].fillna("__missing__").astype(str))
    out_cols = sanitize_feature_names(ohe.get_feature_names_out(cat_cols))
    X_out = X.drop(columns=list(cat_cols)).copy()
    X_out[out_cols] = arr
    return X_out, {
        "encoder": ohe,
        "columns": list(cat_cols),
        "onehot_columns": out_cols,
        "method": "onehot",
    }


def _raw_feature_cols(feature_cols: list, info: "dict | None") -> list:
    """Map the model's (possibly one-hot-EXPANDED) ``feature_cols`` back to the
    ORIGINAL pre-encode column names (numeric + raw categorical names).

    One-hot expansion replaces ``payer`` with ``payer_HMO``/``payer_PPO``/…; the
    raw ``eligible_df`` carries the string ``payer`` column, NOT the indicators.
    The SHAP step must rebuild its frame from the raw columns and re-apply the
    fitted encoder (see :func:`_apply_categorical_onehot`) to reproduce the
    trained feature space. With no one-hot encoding this is a passthrough.
    """
    if not info or not info.get("columns"):
        return list(feature_cols)
    onehot_out = set(info.get("onehot_columns", []))
    return [c for c in feature_cols if c not in onehot_out] + list(info["columns"])


def _apply_categorical_onehot(X: "pd.DataFrame", info: "dict | None") -> "pd.DataFrame":
    """Re-apply a fitted one-hot encoding (from :func:`_fit_categorical_onehot`)
    so a downstream frame (SHAP sample, validation split) carries the IDENTICAL
    feature columns the model trained on. ``None``/empty ``info`` (or any source
    column absent) is a no-op passthrough."""
    if not info or not info.get("columns"):
        return X
    cat_cols = info["columns"]
    if not all(c in X.columns for c in cat_cols):
        return X
    ohe = info["encoder"]
    out_cols = info["onehot_columns"]
    arr = ohe.transform(X[cat_cols].fillna("__missing__").astype(str))
    X_out = X.drop(columns=list(cat_cols)).copy()
    X_out[out_cols] = arr
    return X_out


@dataclass
class StepResult:
    """Result from a pipeline step with enhanced format data."""

    step_num: int | str  # int for main steps (1-8), str for sub-steps ("2b", "2c")
    step_name: str
    status: str  # "success", "warning", "failed"
    duration_seconds: float = 0.0
    key_metrics: dict = None
    details: dict = None
    # Enhanced format fields
    input_summary: dict = None
    processing_steps: list = None  # List of (description, success, detail)
    validation_checks: list = None  # List of (name, passed, expected, actual)
    metrics_table: list = None  # List of (name, value, threshold, passed)
    interpretation: list = None  # List of observation strings
    result_message: str = ""

    def __post_init__(self):
        if self.key_metrics is None:
            self.key_metrics = {}
        if self.details is None:
            self.details = {}
        if self.input_summary is None:
            self.input_summary = {}
        if self.processing_steps is None:
            self.processing_steps = []
        if self.validation_checks is None:
            self.validation_checks = []
        if self.metrics_table is None:
            self.metrics_table = []
        if self.interpretation is None:
            self.interpretation = []


# =============================================================================
# UTILITIES
# =============================================================================


def print_header(step_num: int, title: str) -> None:
    """Print step header."""
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {title}")
    print("=" * 70)


def print_result(key: str, value: Any, indent: int = 2) -> None:
    """Print a result key-value pair."""
    prefix = " " * indent
    if isinstance(value, dict):
        print(f"{prefix}{key}:")
        for k, v in value.items():
            print_result(k, v, indent + 2)
    elif isinstance(value, list) and len(value) > 3:
        print(f"{prefix}{key}: [{len(value)} items]")
    else:
        print(f"{prefix}{key}: {value}")


# =============================================================================
# STANDARDIZED STEP OUTPUT HELPERS
# =============================================================================


def print_step_banner(step_num: int, title: str, duration: float = 0.0) -> None:
    """Print standardized step banner with duration."""
    print("\n" + "=" * 70)
    duration_str = f"Duration: {duration:.1f}s" if duration > 0 else ""
    print(f"STEP {step_num}: {title:<40} {duration_str:>20}")
    print("=" * 70)


def print_input_section(inputs: dict[str, Any]) -> None:
    """Print standardized input summary section."""
    print("\n  📥 Input Summary:")
    for key, value in inputs.items():
        if isinstance(value, (pd.DataFrame,)):
            print(f"    • {key}: DataFrame ({len(value)} rows)")
        elif isinstance(value, dict):
            print(f"    • {key}: {{{len(value)} keys}}")
        elif isinstance(value, list) and len(value) > 3:
            print(f"    • {key}: [{len(value)} items]")
        else:
            print(f"    • {key}: {value}")


def print_processing_steps(steps: list[tuple[str, bool, str | None]]) -> None:
    """Print processing steps with status.

    Args:
        steps: List of (description, success, optional_detail)
    """
    print("\n  ⚙️  Processing:")
    for desc, success, detail in steps:
        icon = "✅" if success else "❌"
        detail_str = f" ({detail})" if detail else ""
        print(f"    {icon} {desc}{detail_str}")


def print_validation_checks(checks: list[tuple[str, bool, str, str]]) -> None:
    """Print validation checks with expected vs actual.

    Args:
        checks: List of (check_name, passed, expected, actual)
    """
    print("\n  🔍 Validation Checks:")
    for name, passed, expected, actual in checks:
        icon = "✅ PASS" if passed else "❌ FAIL"
        print(f"    • {name}: {icon}")
        print(f"        Expected: {expected}")
        print(f"        Actual:   {actual}")


def print_metrics_table(metrics: list[tuple[str, Any, str | None, bool | None]]) -> None:
    """Print metrics as a formatted table.

    Args:
        metrics: List of (metric_name, value, threshold, passed)
                threshold and passed are optional (None to skip)
    """
    print("\n  📊 Key Metrics:")
    print(f"    {'Metric':<25} {'Value':<15} {'Threshold':<15} {'Status':<10}")
    print(f"    {'-' * 65}")

    for name, value, threshold, passed in metrics:
        # Format value
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        elif value is None:
            value_str = "N/A"
        else:
            value_str = str(value)

        # Format threshold and status
        if threshold is not None and passed is not None:
            threshold_str = str(threshold)
            status_str = "✅" if passed else "❌"
        else:
            threshold_str = "-"
            status_str = "-"

        print(f"    {name:<25} {value_str:<15} {threshold_str:<15} {status_str:<10}")


def print_interpretation(
    title: str, observations: list[str], recommendations: list[str] = None
) -> None:
    """Print interpretation section with observations and recommendations.

    Args:
        title: Section title
        observations: List of observation strings
        recommendations: Optional list of recommendations
    """
    print(f"\n  💡 {title}:")
    for obs in observations:
        print(f"    • {obs}")

    if recommendations:
        print("\n    Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"      {i}. {rec}")


def print_step_result(status: str, message: str) -> None:
    """Print final step result with status.

    Args:
        status: "success", "warning", or "failed"
        message: Result message
    """
    print("\n  " + "-" * 60)
    if status == "success":
        print(f"  ✅ RESULT: PASS - {message}")
    elif status == "warning":
        print(f"  ⚠️  RESULT: WARNING - {message}")
    else:
        print(f"  ❌ RESULT: FAIL - {message}")
    print("  " + "-" * 60)


def interpret_qc_scores(qc_report: dict) -> tuple[list[str], list[str]]:
    """Generate interpretation for QC dimension scores.

    Returns:
        Tuple of (observations, recommendations)
    """
    observations = []
    recommendations = []

    completeness = qc_report.get("completeness_score", 0)
    validity = qc_report.get("validity_score", 0)
    consistency = qc_report.get("consistency_score", 0)
    uniqueness = qc_report.get("uniqueness_score", 0)
    timeliness = qc_report.get("timeliness_score", 0)

    # Analyze each dimension
    if completeness < 0.9:
        observations.append(f"Completeness ({completeness:.2f}) indicates missing data")
        recommendations.append("Review data pipeline for incomplete records")

    if validity < 0.9:
        observations.append(f"Validity ({validity:.2f}) suggests data quality issues")
        recommendations.append("Check for outliers and invalid values")

    if consistency < 0.9:
        observations.append(f"Consistency ({consistency:.2f}) shows conflicting values")
        recommendations.append("Verify data source synchronization")

    if uniqueness < 0.95:
        observations.append(f"Uniqueness ({uniqueness:.2f}) indicates potential duplicates")
        recommendations.append("Run deduplication before training")

    if timeliness < 0.8:
        observations.append(f"Timeliness ({timeliness:.2f}) shows stale data")
        recommendations.append("Refresh data from source systems")

    if not observations:
        observations.append("All QC dimensions meet quality thresholds")

    return observations, recommendations


def interpret_class_imbalance(imbalance_info: dict) -> tuple[list[str], list[str]]:
    """Generate interpretation for class imbalance.

    Returns:
        Tuple of (observations, recommendations)
    """
    observations = []
    recommendations = []

    if not imbalance_info.get("imbalance_detected"):
        observations.append("No significant class imbalance detected")
        return observations, recommendations

    minority_ratio = imbalance_info.get("minority_ratio", 0)
    severity = imbalance_info.get("imbalance_severity", "unknown")
    strategy = imbalance_info.get("recommended_strategy", "none")

    observations.append(f"Class imbalance detected: {minority_ratio:.1%} minority class")
    observations.append(f"Severity: {severity.upper()}")
    observations.append(f"Applied remediation: {strategy}")

    if severity == "severe" and minority_ratio < 0.10:
        observations.append("⚠️  Severe imbalance may cause model to ignore minority class")
        recommendations.append("Consider combining SMOTE with class_weight='balanced'")
        recommendations.append("Lower prediction threshold below 0.5 for deployment")
    elif severity == "moderate":
        observations.append("Moderate imbalance handled by resampling/class_weight")

    return observations, recommendations


def interpret_model_performance(
    metrics: dict, accuracy_analysis: dict, min_recall: float, min_precision: float
) -> tuple[list[str], list[str]]:
    """Generate interpretation for model performance.

    Returns:
        Tuple of (observations, recommendations)
    """
    observations = []
    recommendations = []

    auc = metrics.get("roc_auc") or metrics.get("auc_roc", 0)
    recall = accuracy_analysis.get("val_metrics", {}).get("recall", 0)
    precision = accuracy_analysis.get("val_metrics", {}).get("precision", 0)

    # AUC interpretation
    if auc >= 0.80:
        observations.append(f"AUC-ROC ({auc:.3f}) indicates good discrimination ability")
    elif auc >= 0.70:
        observations.append(f"AUC-ROC ({auc:.3f}) indicates acceptable discrimination")
    elif auc >= 0.60:
        observations.append(f"AUC-ROC ({auc:.3f}) indicates weak discrimination")
        recommendations.append("Consider feature engineering to improve predictive power")
    else:
        observations.append(f"AUC-ROC ({auc:.3f}) indicates poor discrimination")
        recommendations.append("Review feature relevance and data quality")

    # Recall interpretation (critical for imbalanced problems)
    y_pred = accuracy_analysis.get("y_pred", [])
    n_pos = sum(y_pred) if y_pred else 0

    if n_pos == 0:
        observations.append("⚠️  CRITICAL: Model predicts ALL samples as negative")
        observations.append("    This model will miss 100% of actual discontinuation cases")
        recommendations.append("Use optimal threshold (not 0.5) for predictions")
        recommendations.append("Verify class_weight='balanced' is applied during training")
    elif recall < min_recall:
        observations.append(f"Recall ({recall:.2%}) below minimum threshold ({min_recall:.0%})")
        observations.append(f"    Model will miss {(1 - recall) * 100:.0f}% of actual positives")
        recommendations.append("Lower prediction threshold to catch more positives")
    else:
        observations.append(f"Recall ({recall:.2%}) meets threshold ({min_recall:.0%})")

    # Precision interpretation
    if precision < min_precision:
        observations.append(f"Precision ({precision:.2%}) below threshold ({min_precision:.0%})")
        recommendations.append("Consider raising threshold to reduce false positives")
    elif n_pos > 0:
        observations.append(f"Precision ({precision:.2%}) acceptable")

    return observations, recommendations


def interpret_confusion_matrix(cm_data: dict) -> list[str]:
    """Generate interpretation for confusion matrix.

    Args:
        cm_data: Dict with tn, fp, fn, tp keys

    Returns:
        List of observation strings
    """
    observations = []

    tn = cm_data.get("tn", 0)
    fp = cm_data.get("fp", 0)
    fn = cm_data.get("fn", 0)
    tp = cm_data.get("tp", 0)

    total = tn + fp + fn + tp
    if total == 0:
        return ["No predictions available for analysis"]

    # Overall accuracy
    accuracy = (tp + tn) / total
    observations.append(f"Overall accuracy: {accuracy:.1%} ({tp + tn}/{total} correct)")

    # Class-specific analysis
    actual_pos = tp + fn
    actual_neg = tn + fp
    pred_pos = tp + fp
    pred_neg = tn + fn

    if actual_pos > 0:
        recall = tp / actual_pos
        if tp == 0:
            observations.append(f"⚠️  Minority class: 0/{actual_pos} detected (0% recall)")
        else:
            observations.append(f"Minority class: {tp}/{actual_pos} detected ({recall:.1%} recall)")

    if pred_pos > 0:
        precision = tp / pred_pos
        observations.append(
            f"Of {pred_pos} predicted positives, {tp} correct ({precision:.1%} precision)"
        )
    elif tp == 0 and fn > 0:
        observations.append(f"⚠️  No positive predictions made (all {fn} positives missed)")

    return observations


def print_success(message: str) -> None:
    """Print success message."""
    print(f"\n  ✅ {message}")


def print_failure(message: str) -> None:
    """Print failure message."""
    print(f"\n  ❌ {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"\n  ⚠️  {message}")


def print_info(message: str) -> None:
    """Print info message."""
    print(f"\n  ℹ️  {message}")


def print_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    title: str = "Confusion Matrix",
) -> dict:
    """Print formatted confusion matrix with detailed metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for threshold analysis)
        title: Section title

    Returns:
        Dictionary with confusion matrix values and derived metrics
    """
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print(f"\n  {title}:")
    print(f"    ┌─────────────────────────────────────┐")
    print(f"    │           Predicted                 │")
    print(f"    │         Neg        Pos              │")
    print(f"    │  Actual ────────────────            │")
    print(f"    │    Neg   TN={tn:4d}    FP={fp:4d}          │")
    print(f"    │    Pos   FN={fn:4d}    TP={tp:4d}          │")
    print(f"    └─────────────────────────────────────┘")

    # Calculate derived metrics
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

    print(f"\n    Derived Metrics:")
    print(f"      • Accuracy:    {accuracy:.4f} ({tp + tn}/{total} correct)")
    print(f"      • Precision:   {precision:.4f} (of predicted pos, {tp}/{tp + fp} correct)")
    print(f"      • Recall/TPR:  {recall:.4f} (of actual pos, {tp}/{tp + fn} found)")
    print(f"      • Specificity: {specificity:.4f} (of actual neg, {tn}/{tn + fp} found)")
    print(f"      • F1 Score:    {f1:.4f}")
    print(f"      • NPV:         {npv:.4f} (of predicted neg, {tn}/{tn + fn} correct)")
    print(f"      • FPR:         {fpr:.4f} (false alarm rate)")
    print(f"      • FNR:         {fnr:.4f} (miss rate)")

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "npv": npv,
        "fpr": fpr,
        "fnr": fnr,
    }


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print scikit-learn classification report with formatting."""
    from sklearn.metrics import classification_report

    print("\n  Classification Report (per-class):")
    report = classification_report(
        y_true, y_pred, target_names=["Class 0 (No Discont.)", "Class 1 (Discont.)"]
    )
    for line in report.split("\n"):
        print(f"    {line}")


def print_threshold_analysis(
    y_true: np.ndarray, y_proba: np.ndarray, optimal_threshold: float = 0.5
) -> None:
    """Analyze model performance at different probability thresholds."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Include lower thresholds for imbalanced data analysis
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Add probability distribution info
    print("\n  Probability Distribution (class 1):")
    print(f"    Min: {y_proba.min():.4f}, Max: {y_proba.max():.4f}")
    print(f"    Mean: {y_proba.mean():.4f}, Median: {np.median(y_proba):.4f}")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(y_proba, percentiles)
    print(
        f"    Percentiles: " + ", ".join([f"P{p}={v:.3f}" for p, v in zip(percentiles, pct_values)])
    )

    print("\n  Threshold Analysis:")
    print(f"    {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Pred Pos':<12}")
    print(f"    {'-' * 56}")

    for thresh in thresholds:
        y_pred_at_thresh = (y_proba >= thresh).astype(int)
        prec = precision_score(y_true, y_pred_at_thresh, zero_division=0)
        rec = recall_score(y_true, y_pred_at_thresh, zero_division=0)
        f1 = f1_score(y_true, y_pred_at_thresh, zero_division=0)
        n_pred_pos = y_pred_at_thresh.sum()

        marker = ""
        if thresh == 0.5:
            marker = " ◄── default"
        elif abs(thresh - optimal_threshold) < 0.01:
            marker = " ◄── optimal"
        print(
            f"    {thresh:<12.2f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {n_pred_pos:<12}{marker}"
        )


def print_model_coefficients(model: Any, feature_names: list[str]) -> None:
    """Print model coefficients/weights for interpretability."""
    print("\n  Model Coefficients/Weights:")

    # Handle different model types
    if hasattr(model, "coef_"):
        coefs = model.coef_.flatten()
        intercept = getattr(model, "intercept_", [0])[0]

        print(f"    Intercept: {intercept:.4f}")
        print(f"    Feature Coefficients:")

        # Sort by absolute value
        coef_pairs = list(zip(feature_names, coefs))
        coef_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        for name, coef in coef_pairs:
            direction = "↑" if coef > 0 else "↓" if coef < 0 else "○"
            print(f"      {direction} {name}: {coef:+.4f}")

    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        print(f"    Feature Importances (tree-based):")

        imp_pairs = list(zip(feature_names, importances))
        imp_pairs.sort(key=lambda x: x[1], reverse=True)

        for name, imp in imp_pairs:
            bar = "█" * int(imp * 20)
            print(f"      {name}: {imp:.4f} {bar}")
    else:
        print(f"    (Model type {type(model).__name__} does not expose coefficients)")


def print_data_distribution_analysis(
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray = None
) -> None:
    """Print data distribution across splits."""
    print("\n  Data Distribution Analysis:")

    def calc_dist(y, name):
        if y is None or len(y) == 0:
            return
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        print(f"\n    {name} (n={total}):")
        for cls, cnt in zip(unique, counts):
            pct = cnt / total * 100
            bar = "█" * int(pct / 5)
            print(f"      Class {cls}: {cnt:4d} ({pct:5.1f}%) {bar}")

    calc_dist(y_train, "Training Set")
    calc_dist(y_val, "Validation Set")
    if y_test is not None:
        calc_dist(y_test, "Test Set")


def _compute_verdict(
    auc_roc: float,
    recall: float,
    precision: float,
    overfitting_severity: str | None = None,
    train_val_delta: float | None = None,
) -> tuple[str, str, str, str]:
    """Compute model usefulness verdict with overfitting gate.

    Severe overfitting (overfitting_severity=='severe' or train_val_delta > 0.15)
    downgrades EXCELLENT/GOOD/ACCEPTABLE verdicts to MARGINAL so the verdict
    honestly reflects generalization risk.

    Returns:
        (verdict, icon, description, deploy_recommendation)
    """
    severe_overfit = overfitting_severity == "severe" or (
        train_val_delta is not None and train_val_delta > 0.15
    )

    if auc_roc >= 0.85 and recall >= 0.7 and not severe_overfit:
        return (
            "EXCELLENT",
            "🌟",
            "Model has strong discrimination and high recall",
            "Ready for production deployment",
        )
    if auc_roc >= 0.75 and recall >= 0.5 and not severe_overfit:
        return (
            "GOOD",
            "✅",
            "Model has good discrimination and acceptable recall",
            "Suitable for staging/production with monitoring",
        )
    if auc_roc >= 0.65 and recall >= 0.3 and precision < 0.05:
        return (
            "THRESHOLD_NEEDED",
            "🔧",
            "Model discriminates well but operating point needs threshold optimization",
            "Apply precision-constrained threshold before deployment",
        )
    if auc_roc >= 0.65 and recall >= 0.3 and not severe_overfit:
        return (
            "ACCEPTABLE",
            "⚡",
            "Model has moderate performance, meets minimum thresholds",
            "Deploy with caution, monitor closely",
        )
    if auc_roc >= 0.55 or severe_overfit:
        if severe_overfit:
            return (
                "MARGINAL",
                "⚠️",
                "Severe overfitting detected (train-val AUC Δ > 0.15)",
                "Reduce model capacity / add regularization before deployment",
            )
        return (
            "MARGINAL",
            "⚠️",
            "Model barely exceeds random chance",
            "Consider retraining with more data or different approach",
        )
    return (
        "POOR",
        "❌",
        "Model performs near or below random chance",
        "Do not deploy, requires significant improvement",
    )


def print_detailed_summary(
    experiment_id: str, step_results: list[StepResult], state: dict[str, Any]
) -> None:
    """Print detailed results from each tier0 step using enhanced format.

    Args:
        experiment_id: The experiment identifier
        step_results: List of StepResult objects from each step
        state: Pipeline state with all collected data
    """
    print(f"\n{'=' * 70}")
    print("DETAILED STEP RESULTS")
    print(f"{'=' * 70}")

    for result in step_results:
        status_icon = (
            "✅" if result.status == "success" else "⚠️" if result.status == "warning" else "❌"
        )
        print(f"\n{'-' * 70}")
        print(f"STEP {result.step_num}: {result.step_name} [{status_icon} {result.status.upper()}]")
        print(f"{'-' * 70}")

        if result.duration_seconds > 0:
            print(f"  Duration: {result.duration_seconds:.2f}s")

        # Enhanced format: Input Summary
        if result.input_summary:
            print_input_section(result.input_summary)

        # Enhanced format: Processing Steps
        if result.processing_steps:
            print_processing_steps(result.processing_steps)

        # Enhanced format: Validation Checks
        if result.validation_checks:
            print_validation_checks(result.validation_checks)

        # Enhanced format: Metrics Table
        if result.metrics_table:
            print_metrics_table(result.metrics_table)

        # Enhanced format: Interpretation
        if result.interpretation:
            title = f"{result.step_name} Analysis"
            print_interpretation(title, result.interpretation)

        # Enhanced format: Result
        if result.result_message:
            print_step_result(
                result.status, f"{result.result_message} ({result.duration_seconds:.1f}s)"
            )

        # Fallback: Print key metrics if no enhanced data
        if not result.metrics_table and result.key_metrics:
            print("\n  Key Metrics:")
            for key, value in result.key_metrics.items():
                if isinstance(value, float):
                    print(f"    • {key}: {value:.4f}")
                else:
                    print(f"    • {key}: {value}")

        # Fallback: Print details if no enhanced data
        if not result.input_summary and result.details:
            print("\n  Details:")
            for key, value in result.details.items():
                if isinstance(value, dict):
                    print(f"    {key}:")
                    for k, v in list(value.items())[:10]:  # Limit nested items
                        if isinstance(v, float):
                            print(f"      - {k}: {v:.4f}")
                        else:
                            print(f"      - {k}: {v}")
                elif isinstance(value, list) and len(value) > 5:
                    print(f"    {key}: [{len(value)} items]")
                else:
                    print(f"    {key}: {value}")

    # Cohort Construction Analysis
    cohort_result = state.get("cohort_result")
    if cohort_result:
        print(f"\n{'=' * 70}")
        print("COHORT CONSTRUCTION ANALYSIS")
        print(f"{'=' * 70}")

        patient_df = state.get("patient_df")
        eligible_df = state.get("eligible_df")
        input_count = len(patient_df) if patient_df is not None else 0
        eligible_count = len(eligible_df) if eligible_df is not None else 0
        excluded_count = input_count - eligible_count

        print(f"\n  📊 Patient Flow:")
        print(f"    • Input Patients:    {input_count}")
        print(f"    • Eligible Patients: {eligible_count}")
        print(f"    • Excluded Patients: {excluded_count}")
        if input_count > 0:
            print(f"    • Retention Rate:    {eligible_count / input_count:.1%}")

        if hasattr(cohort_result, "eligibility_stats") and cohort_result.eligibility_stats:
            stats = cohort_result.eligibility_stats
            print(f"\n  📋 Eligibility Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"    • {key}: {value:.4f}")
                else:
                    print(f"    • {key}: {value}")

        # Show what criteria were applied
        print(f"\n  🔍 Applied Criteria:")
        print(f"    • Cohort ID: {cohort_result.cohort_id}")
        print(f"    • Execution ID: {cohort_result.execution_id}")
        print(f"    • Status: {cohort_result.status}")

    # Class Imbalance Section
    class_imbalance_info = state.get("class_imbalance_info", {})
    if class_imbalance_info.get("imbalance_detected"):
        print(f"\n{'=' * 70}")
        print("CLASS IMBALANCE REMEDIATION")
        print(f"{'=' * 70}")

        print("\n  📊 Imbalance Analysis:")
        print(f"    • Imbalance Detected: Yes")
        print(
            f"    • Severity: {class_imbalance_info.get('imbalance_severity', 'unknown').upper()}"
        )
        print(f"    • Minority Ratio: {class_imbalance_info.get('minority_ratio', 0):.2%}")
        print(f"    • Imbalance Ratio: {class_imbalance_info.get('imbalance_ratio', 1):.1f}:1")

        class_dist = class_imbalance_info.get("class_distribution", {})
        if class_dist:
            print("\n  📈 Class Distribution:")
            for cls, count in class_dist.items():
                print(f"    • Class {cls}: {count} samples")

        print("\n  🔧 Remediation Applied:")
        print(f"    • Strategy: {class_imbalance_info.get('recommended_strategy', 'none')}")
        print(f"    • Rationale: {class_imbalance_info.get('strategy_rationale', 'N/A')}")

        # Show before/after if resampling was applied
        resampling_info = state.get("resampling_info", {})
        if resampling_info.get("resampling_applied"):
            print("\n  📊 Resampling Results:")
            orig_samples = resampling_info.get("original_samples")
            resamp_samples = resampling_info.get("resampled_samples")
            print(f"    • Original Samples: {orig_samples}")
            print(f"    • Resampled Samples: {resamp_samples}")
            new_ratio = resampling_info.get("new_minority_ratio")
            if new_ratio is not None:
                print(f"    • New Minority Ratio: {new_ratio:.2%}")
            else:
                print(f"    • New Minority Ratio: N/A")
            # Show resampled distribution
            resampled_dist = resampling_info.get("resampled_distribution", {})
            if resampled_dist:
                print("\n  📈 Resampled Class Distribution:")
                for cls, count in sorted(resampled_dist.items()):
                    print(f"    • Class {cls}: {count} samples")
        else:
            # Resampling not applied even though imbalance detected (e.g., class_weight strategy)
            print("\n  📊 Resampling Results:")
            print(f"    • Resampling Applied: No")
            strategy = resampling_info.get("resampling_strategy", "none")
            if strategy == "class_weight":
                print(f"    • Strategy: class_weight (handled during training)")
            else:
                print(f"    • Strategy: {strategy}")
    elif class_imbalance_info:
        print(f"\n  ℹ️  Class Imbalance: Not detected (minority ratio >= 40%)")

    # Feature Importance Section
    feature_importance = state.get("feature_importance")
    if feature_importance:
        print(f"\n{'=' * 70}")
        print("FEATURE IMPORTANCE (SHAP)")
        print(f"{'=' * 70}")
        print("\n  Top Features:")
        for i, fi in enumerate(feature_importance[:10], 1):
            if isinstance(fi, dict):
                name = fi.get("feature", f"feature_{i}")
                importance = fi.get("importance", 0)
                print(f"    {i}. {name}: {importance:.4f}")
            else:
                print(f"    {i}. {fi}")

    # Validation Metrics Section
    validation_metrics = state.get("validation_metrics", {})
    if validation_metrics:
        print(f"\n{'=' * 70}")
        print("FINAL MODEL PERFORMANCE")
        print(f"{'=' * 70}")

        # Key metrics
        key_metrics = [
            "roc_auc",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "pr_auc",
            "brier_score",
        ]
        print("\n  Primary Metrics:")
        for metric in key_metrics:
            value = validation_metrics.get(metric)
            if value is not None:
                print(f"    • {metric}: {value:.4f}")

        # Per-class metrics
        print("\n  Per-Class Metrics:")
        for key, value in validation_metrics.items():
            if "class_" in key and value is not None:
                print(f"    • {key}: {value:.4f}")

    # Test Metrics Section (added by tier0_quality_remediation_arc Shard D,
    # 2026-05-06). Per R5 rubric (mlops_data_pipeline_engineering_distilled.md:483-489):
    # "Train on train / Tune threshold on validation / Evaluate ONCE on test." The
    # evaluator already computes test_metrics at evaluator.py:1192-1450 and the runner
    # already wires them into state at run_tier0_test.py:4782 + emits in JSON
    # artifact at L5531-5535. This printer block surfaces them to stdout for human
    # consumption (codex review I2 narrowed the gap from "consumers don't see test
    # numbers" — they do via state — to "human-visible printer block missing").
    test_metrics = state.get("test_metrics", {})
    if test_metrics:
        print(f"\n{'=' * 70}")
        print("TEST-SET HOLDOUT PERFORMANCE")
        print(f"{'=' * 70}")

        # Key metrics (same set as validation for direct comparison)
        test_key_metrics = [
            "roc_auc",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "pr_auc",
            "brier_score",
            "mcc",
            "business_utility",
        ]
        print("\n  Primary Metrics (held-out test set):")
        for metric in test_key_metrics:
            value = test_metrics.get(metric)
            if value is not None:
                if isinstance(value, (int, float)):
                    print(f"    • {metric}: {value:.4f}")
                else:
                    print(f"    • {metric}: {value}")

        # Calibration + threshold provenance
        cal_metrics = [
            "ece_pre_isotonic",
            "ece_post_isotonic",
            "calibration_slope",
            "calibration_intercept",
            "chosen_threshold",
            "chosen_threshold_source",
        ]
        print("\n  Calibration + Threshold:")
        for metric in cal_metrics:
            value = test_metrics.get(metric)
            if value is not None:
                if isinstance(value, (int, float)):
                    print(f"    • {metric}: {value:.4f}")
                else:
                    print(f"    • {metric}: {value}")

        # Train→Val→Test overfit signal (if present)
        train_val_delta = test_metrics.get("train_val_auc_delta")
        if train_val_delta is not None:
            print(f"\n  Overfit (train→val AUC delta): {train_val_delta:.4f}")

        # Per-class metrics
        print("\n  Per-Class Metrics (test):")
        for key, value in test_metrics.items():
            if "class_" in key and value is not None and isinstance(value, (int, float)):
                print(f"    • {key}: {value:.4f}")

    # =========================================================================
    # ENHANCED ACCURACY ANALYSIS SECTION
    # =========================================================================
    accuracy_data = state.get("accuracy_analysis", {})
    if accuracy_data:
        print(f"\n{'=' * 70}")
        print("ENHANCED ACCURACY ANALYSIS")
        print(f"{'=' * 70}")

        # Confusion Matrix
        if accuracy_data.get("y_true") is not None and accuracy_data.get("y_pred") is not None:
            y_true = np.array(accuracy_data["y_true"])
            y_pred = np.array(accuracy_data["y_pred"])
            y_proba = (
                np.array(accuracy_data["y_proba"])
                if accuracy_data.get("y_proba") is not None
                else None
            )

            # Print confusion matrix with all derived metrics
            print_confusion_matrix(y_true, y_pred, y_proba, "Validation Set Confusion Matrix")

            # Print full classification report
            print_classification_report(y_true, y_pred)

            # Threshold analysis (if probabilities available)
            if y_proba is not None:
                optimal_thresh = state.get("optimal_threshold", 0.5)
                print_threshold_analysis(y_true, y_proba, optimal_thresh)

        # Model coefficients/weights
        trained_model = state.get("trained_model")
        if trained_model is not None:
            feature_cols = accuracy_data.get(
                "feature_columns", ["days_on_therapy", "hcp_visits", "prior_treatments"]
            )
            print_model_coefficients(trained_model, feature_cols)

        # Data distribution across splits
        if accuracy_data.get("y_train") is not None:
            print_data_distribution_analysis(
                np.array(accuracy_data.get("y_train", [])),
                np.array(accuracy_data.get("y_val", [])),
                np.array(accuracy_data.get("y_test", [])),
            )

        # Train vs Validation comparison (overfitting check)
        train_metrics = accuracy_data.get("train_metrics", {})
        val_metrics = accuracy_data.get("val_metrics", {})
        if train_metrics and val_metrics:
            print("\n  Overfitting Analysis (Train vs Validation):")
            print(
                f"    {'Metric':<15} {'Train':<12} {'Validation':<12} {'Delta':<12} {'Status':<15}"
            )
            print(f"    {'-' * 60}")

            for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                train_val = train_metrics.get(metric)
                val_val = val_metrics.get(metric)
                if train_val is not None and val_val is not None:
                    delta = train_val - val_val
                    if delta > 0.1:
                        status = "⚠️ Overfitting"
                    elif delta > 0.05:
                        status = "⚡ Mild overfit"
                    elif delta < -0.05:
                        status = "❓ Unusual"
                    else:
                        status = "✅ Good"
                    print(
                        f"    {metric:<15} {train_val:<12.4f} {val_val:<12.4f} {delta:+<12.4f} {status:<15}"
                    )

    # =========================================================================
    # MODEL USEFULNESS VERDICT
    # =========================================================================
    model_usefulness = state.get("model_usefulness_verdict", {})
    if not model_usefulness and accuracy_data:
        # Compute from accuracy data if not explicitly set
        y_pred = accuracy_data.get("y_pred", [])
        n_pos_pred = sum(y_pred) if y_pred else 0
        eval_test_metrics = state.get("test_metrics", {})
        model_usefulness = {
            "status": "useless" if n_pos_pred == 0 else "needs_review",
            "reason": "predicts_all_negative" if n_pos_pred == 0 else "unknown",
            "minority_recall": state.get("minority_recall") or eval_test_metrics.get("recall", 0),
            "minority_precision": state.get("minority_precision")
            or eval_test_metrics.get("precision", 0),
        }

    if model_usefulness or accuracy_data:
        print(f"\n{'=' * 70}")
        print("⚠️  MODEL USEFULNESS VERDICT")
        print(f"{'=' * 70}")

        y_pred = accuracy_data.get("y_pred", []) if accuracy_data else []
        n_pos_pred = sum(y_pred) if y_pred else 0
        total_pred = len(y_pred) if y_pred else 0

        if n_pos_pred == 0 and total_pred > 0:
            print(f"\n  🚨 CRITICAL: MODEL IS USELESS FOR ITS INTENDED PURPOSE")
            print(f"\n  The model predicts EVERY sample as class 0 (no discontinuation).")
            print(f"  This means:")
            print(f"    • 0% of actual discontinuation cases will be detected")
            print(f"    • The model cannot identify any high-risk patients")
            print(f"    • It will miss 100% of the patients who actually discontinue")
            print(f"\n  Root Cause Analysis:")
            print(f"    • Severe class imbalance (minority ~9-14%) in validation data")
            print(f"    • SMOTE resampling on training data didn't generalize")
            print(f"    • Model learned majority class bias")
            print(f"\n  Recommended Actions:")
            print(f"    1. Lower prediction threshold (try 0.2-0.3 instead of 0.5)")
            print(f"    2. Use class_weight='balanced' instead of SMOTE")
            print(f"    3. Collect more minority class samples")
            print(f"    4. Try ensemble methods (RandomForest, XGBoost)")
            print(f"\n  ❌ VERDICT: FAIL - Model should NOT be deployed")
        elif total_pred > 0:
            # Model makes positive predictions - evaluate usefulness
            # Use evaluator's test metrics (authoritative) instead of recomputed val metrics
            eval_metrics = state.get("test_metrics", {})
            auc_roc = eval_metrics.get("roc_auc", 0)
            recall = eval_metrics.get("recall", 0)
            precision = eval_metrics.get("precision", 0)
            f1 = eval_metrics.get("f1_score", 0)

            # --- LEAKAGE SUSPICION CHECK (imbalance-aware) ---
            # The evaluator now uses adaptive thresholds based on class ratio,
            # PR-AUC, and MCC — no more hardcoded AUC >= 0.99 check.
            suspicion_level = state.get("suspicion_level", "none")
            leakage_severity = state.get("leakage_severity", "none")
            suspicion_reasons = state.get("suspicion_reasons", [])
            leaked_features = state.get("leaked_features", [])
            leakage_findings = state.get("leakage_findings", [])
            investigation_recs = state.get("investigation_recommendations", [])

            is_leakage_suspected = suspicion_level in ("high", "critical") or leakage_severity in (
                "high",
                "critical",
            )

            # --- VALIDATION DIAGNOSTICS ---
            permutation = state.get("permutation_test", {})
            cv_res = state.get("cv_results", {})
            cal_analysis = state.get("calibration_analysis", {})
            f1_thresh = state.get("f1_threshold_analysis", {})
            split_val = state.get("split_validation", {})
            mcc_val = state.get("mcc")
            cal_ece = state.get("calibration_error")
            cal_ece_post = state.get("calibrated_ece")
            pr_auc_val = state.get("pr_auc") or val_metrics.get("pr_auc") or 0

            print(f"\n  {'─' * 60}")
            print(f"  Validation Diagnostics:")
            print(f"  {'─' * 60}")

            # Permutation test
            if permutation.get("signal_genuine") is not None:
                sig = "GENUINE" if permutation["signal_genuine"] else "RANDOM"
                pval = permutation.get("permutation_pvalue", 0)
                shuf_mean = permutation.get("permutation_auc_mean", 0)
                print(
                    f"    Permutation test:  signal={sig} (p={pval:.4f}, shuffled AUC={shuf_mean:.4f})"
                )

            # Cross-validation
            if cv_res.get("cv_completed"):
                print(
                    f"    Stratified 5-fold: AUC={cv_res.get('cv_roc_auc_mean', 0):.4f}"
                    f"±{cv_res.get('cv_roc_auc_std', 0):.4f}, "
                    f"PR-AUC={cv_res.get('cv_pr_auc_mean', 0):.4f}"
                    f"±{cv_res.get('cv_pr_auc_std', 0):.4f}, "
                    f"MCC={cv_res.get('cv_mcc_mean', 0):.4f}"
                    f"±{cv_res.get('cv_mcc_std', 0):.4f}"
                )

            # Imbalance-robust metrics
            print(f"    PR-AUC:            {pr_auc_val:.4f} (imbalance-robust)")
            if mcc_val is not None:
                print(f"    MCC:               {mcc_val:.4f} (class-balanced)")

            # Calibration
            if cal_ece is not None:
                ece_str = f"{cal_ece:.4f}"
                if cal_ece_post is not None:
                    # v5 B1 (2026-05-11): the post-hoc method is now
                    # auto-selected (isotonic OR Platt). Read the resolved
                    # method from the post_hoc_calibration audit dict so
                    # the user-facing string reflects what actually ran.
                    cal_info_for_label = state.get("post_hoc_calibration") or {}
                    resolved_method = cal_info_for_label.get(
                        "calibration_method_resolved"
                    ) or cal_info_for_label.get("calibration_method", "post-hoc")
                    ece_str += f" → {cal_ece_post:.4f} (after {resolved_method} calibration)"
                print(f"    ECE:               {ece_str}")

            # F1-optimal threshold
            if f1_thresh.get("f1_optimal_threshold"):
                print(
                    f"    F1-optimal thresh: {f1_thresh['f1_optimal_threshold']:.4f} "
                    f"(F1={f1_thresh.get('f1_at_optimal', 0):.4f}, "
                    f"P={f1_thresh.get('precision_at_f1_optimal', 0):.4f}, "
                    f"R={f1_thresh.get('recall_at_f1_optimal', 0):.4f})"
                )

            # Split stratification
            if split_val.get("split_positive_ratios"):
                ratios = split_val["split_positive_ratios"]
                strat = "OK" if split_val.get("is_stratified") else "DRIFT"
                print(
                    f"    Split stratify:    {strat} "
                    f"(train={ratios.get('train', 0):.3f}, "
                    f"val={ratios.get('validation', 0):.3f}, "
                    f"test={ratios.get('test', 0):.3f})"
                )

            print(f"  {'─' * 60}")

            if is_leakage_suspected:
                verdict = "LEAKAGE_SUSPECTED"
                icon = "🚨"
                description = "Model metrics are implausibly perfect — data leakage suspected"
                deploy_recommendation = "DO NOT DEPLOY — investigate feature derivation pipeline"

                print(f"\n  {icon} VERDICT: {verdict}")
                print(f"\n  Assessment: {description}")

                if leaked_features:
                    print(f"\n  Leaked Features:")
                    for feat in leaked_features:
                        print(f"    • {feat}")

                if leakage_findings:
                    print(f"\n  Pre-Training Leakage Findings:")
                    for finding in leakage_findings:
                        sev = finding.get("severity", "unknown").upper()
                        desc = finding.get("description", "")
                        print(f"    [{sev}] {desc}")
                        rec = finding.get("recommendation", "")
                        if rec:
                            print(f"           → {rec}")

                if suspicion_reasons:
                    print(f"\n  Post-Training Suspicion Reasons:")
                    for reason in suspicion_reasons:
                        print(f"    • {reason}")

                if investigation_recs:
                    print(f"\n  Investigation Recommendations:")
                    for rec in investigation_recs:
                        print(f"    • {rec}")

                print(f"\n  Key Metrics:")
                print(f"    • AUC-ROC:   {auc_roc:.4f}")
                print(f"    • PR-AUC:    {pr_auc_val:.4f}")
                print(f"    • Recall:    {recall:.4f}")
                print(f"    • Precision: {precision:.4f}")
                print(f"    • F1 Score:  {f1:.4f}")
                if mcc_val is not None:
                    print(f"    • MCC:       {mcc_val:.4f}")
                print(f"\n  ❌ Recommendation: {deploy_recommendation}")

            # --- Normal verdict logic (only if leakage not suspected) ---
            else:
                overfitting_severity = state.get("overfitting_severity")
                _train_auc = state.get("train_metrics", {}).get("roc_auc")
                _val_auc = state.get("validation_metrics", {}).get("roc_auc")
                if _train_auc is None and accuracy_data:
                    _train_auc = accuracy_data.get("train_metrics", {}).get("roc_auc")
                if _val_auc is None and accuracy_data:
                    _val_auc = accuracy_data.get("val_metrics", {}).get("roc_auc")
                train_val_delta = (
                    _train_auc - _val_auc
                    if (_train_auc is not None and _val_auc is not None)
                    else None
                )
                verdict, icon, description, deploy_recommendation = _compute_verdict(
                    auc_roc=auc_roc,
                    recall=recall,
                    precision=precision,
                    overfitting_severity=overfitting_severity,
                    train_val_delta=train_val_delta,
                )

            print(f"\n  {icon} VERDICT: {verdict}")
            print(f"\n  Assessment: {description}")
            print(f"\n  Key Metrics:")
            print(f"    • AUC-ROC:   {auc_roc:.4f}")
            print(f"    • PR-AUC:    {pr_auc_val:.4f}")
            print(f"    • Recall:    {recall:.4f} ({recall * 100:.1f}% of positives detected)")
            print(
                f"    • Precision: {precision:.4f} ({precision * 100:.1f}% of predictions correct)"
            )
            print(f"    • F1 Score:  {f1:.4f}")
            if mcc_val is not None:
                print(f"    • MCC:       {mcc_val:.4f}")
            print(
                f"    • Positive Predictions: {n_pos_pred}/{total_pred} ({n_pos_pred / total_pred * 100:.1f}%)"
            )
            print(f"\n  Recommendation: {deploy_recommendation}")

    # Deployment Info
    deployment_manifest = state.get("deployment_manifest", {})
    if deployment_manifest:
        print(f"\n{'=' * 70}")
        print("DEPLOYMENT STATUS")
        print(f"{'=' * 70}")
        print(f"\n  • Deployment ID: {deployment_manifest.get('deployment_id', 'N/A')}")
        print(f"  • Environment: {deployment_manifest.get('environment', 'N/A')}")
        print(f"  • Status: {deployment_manifest.get('status', 'N/A')}")
        print(f"  • Endpoint: {deployment_manifest.get('endpoint_url', 'N/A')}")

    # Regulatory authorization manifest + advisory-vs-enforced gate map (gaps
    # G6/G12). Always printed so the compliance artifact the deployer computes
    # (frozen SHA256 payload + Gate-N1 audit) is visible in the run output — it
    # previously reached humans only via a test-only env-gated JSON, while the
    # console printed the simpler deployment_manifest above.
    from src.agents.ml_foundation.model_deployer.regulatory_report import (
        format_regulatory_report,
    )

    print(f"\n{'=' * 70}")
    print("REGULATORY AUTHORIZATION")
    print(f"{'=' * 70}\n")
    print(format_regulatory_report(state.get("regulatory_deployment_manifest")))

    print(f"\n{'=' * 70}")


def load_rwd_data(data_dir: str, target: str) -> pd.DataFrame:
    """Load real-world patient journey data from JSON or parquet.

    Prefers ``e2i_ml_v3_patient_journeys.parquet`` if present (Optum converter
    output); falls back to ``...json`` (CSU converter output). Either format
    yields a DataFrame with the same schema.

    Args:
        data_dir: Directory containing ``e2i_ml_v3_patient_journeys.*``.
        target: Target outcome column name.

    Returns:
        DataFrame with columns matching the Tier 0 pipeline schema.
    """
    base = Path(data_dir)
    parquet_path = base / "e2i_ml_v3_patient_journeys.parquet"
    json_path = base / "e2i_ml_v3_patient_journeys.json"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif json_path.exists():
        with open(json_path) as f:
            records = json.load(f)
        df = pd.DataFrame(records)
    else:
        raise FileNotFoundError(
            f"RWD data not found at {parquet_path} or {json_path}\n"
            "Ensure the patient journey file has been generated "
            "(e.g. python scripts/convert_csu_rwd.py or convert_optum_rwd.py)."
        )

    # Map age_group values to pipeline-expected buckets
    age_map = {
        "<18": "<50",
        "18-34": "<50",
        "35-49": "<50",
        "50-65": "50-65",
        "65+": ">65",
    }
    if "age_group" in df.columns:
        df["age_group"] = df["age_group"].map(age_map).fillna("<50")

    # Ensure numeric types for ML features
    for col in ["days_on_therapy", "hcp_visits", "prior_treatments"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Coerce new generic numeric columns from converter
    for col in [
        "age_continuous",
        "eligibility_duration_days",
        "medication_claim_count",
        "procedure_claim_count",
        "lab_claim_count",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Ensure treatment_initiated is int
    if "treatment_initiated" in df.columns:
        df["treatment_initiated"] = (
            pd.to_numeric(df["treatment_initiated"], errors="coerce").fillna(0).astype(int)
        )

    # Handle discontinuation_flag (may be None for non-medicated patients)
    if "discontinuation_flag" in df.columns:
        df["discontinuation_flag"] = pd.to_numeric(df["discontinuation_flag"], errors="coerce")

    # Ensure journey_status exists
    if "journey_status" not in df.columns:
        df["journey_status"] = df.apply(
            lambda r: "transitioning" if r.get("treatment_initiated", 0) == 1 else "active",
            axis=1,
        )

    # If targeting discontinuation_flag, filter to medicated patients only
    if target == "discontinuation_flag":
        pre_filter = len(df)
        df = df[(df["treatment_initiated"] == 1) & df["discontinuation_flag"].notna()].copy()
        df["discontinuation_flag"] = df["discontinuation_flag"].astype(int)
        print(f"  Filtered to medicated patients: {pre_filter} -> {len(df)}")

    print(f"  Loaded {len(df)} RWD patient records from {data_dir}")
    print(f"  Indication: {CONFIG.indication}")
    print(f"  Target: {CONFIG.target_outcome}")

    target_col = CONFIG.target_outcome
    if target_col in df.columns:
        pos = int(df[target_col].sum())
        total = len(df)
        print(f"  Class distribution: {pos}/{total} positive ({pos / total:.1%})")

    return df


_SCENARIO_REGIME_TO_NAME: Dict[str, str] = {
    "scenario_a": "A_DIAGNOSTIC_BC_IDFS",
    "scenario_a_balanced": "A_DIAGNOSTIC_BC_IDFS_BALANCED",
    "scenario_b": "B_SCREENING_IGAN_ESKD",
    "scenario_c": "C_TREATMENT_CSU_RESPONSE",
}

_SCENARIO_REGIME_TO_BRAND: Dict[str, str] = {
    "scenario_a": "Kisqali",
    "scenario_a_balanced": "Kisqali",
    "scenario_b": "Fabhalta",
    "scenario_c": "Remibrutinib",
}

_SCENARIO_REGIME_TO_ID_PREFIX: Dict[str, str] = {
    "scenario_a": "sa",
    "scenario_a_balanced": "sab",
    "scenario_b": "sb",
    "scenario_c": "sc",
}


def _scenario_to_dataframe(
    regime: str,
    seed: int,
    n_total: int | None = None,
    imbalance_ratio: float | None = None,
) -> pd.DataFrame:
    """Generate a synthetic_v2 scenario dataset and adapt to the runner contract.

    Bypasses ``ml_patients()``; calls ``synthetic_v2.api.generate_scenario`` and
    flattens train/val/test back to one DataFrame — the pipeline re-splits
    downstream, so trusting the scenario's internal split would double-split.

    Args:
        regime: One of ``scenario_a``, ``scenario_a_balanced``, ``scenario_b``,
            ``scenario_c``. Maps to a registered ``ScenarioName`` via
            ``_SCENARIO_REGIME_TO_NAME``.
        seed: Random seed forwarded to ``generate_scenario``.
        n_total: Optional override for the scenario's ``builder.default_n_total``.
            ``None`` preserves the default (6000 for all four scenarios) so
            no-flag invocations remain bit-identical to the pre-PR baseline.
        imbalance_ratio: Defense-in-depth (backlog #21.7). Must be ``None`` for
            scenario regimes. The CLI guard at lines 7168-7192 already rejects
            ``--imbalanced`` under ``--regime scenario_*`` at the argparse
            boundary; this parameter is the function-level mirror so a
            programmatic caller bypassing argparse cannot silently drop the
            ratio either. Any non-None value raises ``ValueError`` before
            generation.

    Each scenario uses a regime-specific ``brand`` + ``patient_journey_id``
    prefix so a downstream consumer that union-merges multi-scenario outputs
    can disambiguate by ``brand`` without colliding on ID.
    """
    import numpy as np

    from src.ml.synthetic_v2.api import generate_scenario
    from src.ml.synthetic_v2.scenarios import ScenarioName

    if regime not in _SCENARIO_REGIME_TO_NAME:
        raise ValueError(
            f"unknown synthetic_v2 regime {regime!r}; "
            f"expected one of {sorted(_SCENARIO_REGIME_TO_NAME.keys())}"
        )
    if imbalance_ratio is not None:
        # Scenario regimes encode signal-preservation contracts — post-hoc
        # relabel would corrupt feature ↔ target correlation. scenario_a_balanced
        # re-calibrates prevalence to 0.50 via intercept solver INSIDE the DGP,
        # preserving signal; that is the right tool for a 50:50 cohort. See
        # backlog #21.7 + CLI guard at scripts/run_tier0_test.py:7168-7192.
        if imbalance_ratio == 0.50:
            # Regime-aware redirect: only scenario_a has a balanced variant
            # (scenario_a_balanced). scenario_b / scenario_c don't, so naive
            # "use scenario_a_balanced" misleads users who wanted scenario_b/c's
            # DGP. Codex pass-2 LOW.
            if regime == "scenario_a_balanced":
                redirect = (
                    "regime='scenario_a_balanced' already produces a 50:50 "
                    "cohort via intercept-solver prevalence calibration — "
                    "drop imbalance_ratio=0.50 to use the balanced regime as-is."
                )
            elif regime == "scenario_a":
                redirect = (
                    "Use regime='scenario_a_balanced' for a signal-preserving "
                    "50:50 cohort (scenario_a DGP, prevalence re-calibrated via "
                    "intercept solver — preserves feature ↔ target correlation)."
                )
            else:  # scenario_b, scenario_c — no balanced variant
                redirect = (
                    f"regime={regime!r} has no balanced variant. Either "
                    "(a) use regime='scenario_a_balanced' for a 50:50 cohort "
                    "with scenario_a's DGP, or (b) use a legacy regime "
                    "(default/adverse/clean) with imbalance_ratio for post-hoc "
                    "relabel on top of a non-scenario data generator."
                )
        else:
            redirect = (
                "No scenario regime accepts an arbitrary prevalence ratio; "
                "use a legacy regime (default/adverse/clean) with "
                "imbalance_ratio for post-hoc relabel."
            )
        raise ValueError(
            f"imbalance_ratio={imbalance_ratio!r} is incompatible with "
            f"scenario regime {regime!r}: scenario regimes encode "
            f"signal-preservation contracts — post-hoc relabel would corrupt "
            f"feature ↔ target correlation. See backlog #21.7. " + redirect
        )
    scenario_attr = _SCENARIO_REGIME_TO_NAME[regime]
    scenario = getattr(ScenarioName, scenario_attr)

    ds = generate_scenario(scenario, seed=seed, n_total=n_total)

    X = np.vstack([ds.X_train, ds.X_val, ds.X_test])
    y = np.concatenate([ds.y_train, ds.y_val, ds.y_test])
    df = pd.DataFrame(X, columns=list(ds.metadata.feature_names))
    df["discontinuation_flag"] = y.astype(int)
    id_prefix = _SCENARIO_REGIME_TO_ID_PREFIX[regime]
    df["patient_journey_id"] = [f"{id_prefix}-{i:06d}" for i in range(len(df))]
    df["patient_id"] = df["patient_journey_id"]
    df["brand"] = _SCENARIO_REGIME_TO_BRAND[regime]
    df["geographic_region"] = "northeast"
    # journey_status must NOT correlate with the target — a target-coupled value
    # trips the leakage detector and triggers LLM remediation, which then
    # hallucinates a replacement feature set and discards the scenario's
    # clinical features. Constant "active" keeps it pipeline-shaped without leakage.
    df["journey_status"] = "active"
    today = datetime.now().isoformat()
    df["journey_start_date"] = today
    df["journey_end_date"] = None
    df["created_at"] = today
    df["data_quality_score"] = 0.95
    return df


def generate_sample_data(
    n_samples: int = 100,
    seed: int = 42,
    imbalance_ratio: float | None = None,
    positive_rate: float = 0.30,
    *,
    signal_strength: float = 1.0,
    noise_sd: float = 0.10,
    signalize_extra_features: bool = False,
    _generator: str | None = None,
    n_total: int | None = None,
) -> pd.DataFrame:
    """Generate sample patient journey data using the ML-ready generator.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        imbalance_ratio: If provided, force minority class to this ratio (e.g., 0.1 for 10%)
                        via post-hoc relabelling. None means leave the
                        generator-driven labels intact.
        positive_rate: Base positive-class rate passed into ``ml_patients``.
            Drives generator-time class balance (Block 4, Finding #8). Use
            ``0.30`` for the historical default and ``0.02`` for the adverse
            regime. Distinct from ``imbalance_ratio`` which relabels AFTER
            generation, breaking feature ↔ target correlations.
        signal_strength: Multiplier on the deterministic feature
            contributions (Section A of pre_phase2_unblockers).
        noise_sd: Standard deviation of the per-patient Gaussian noise.
        signalize_extra_features: When True, four additional features
            (age_group, geographic_region, brand, data_quality_score)
            also contribute to the risk score.
        n_total: For ``_generator`` set to a synthetic_v2 scenario (one of
            ``scenario_a``, ``scenario_a_balanced``, ``scenario_b``,
            ``scenario_c``), overrides the scenario's ``builder.default_n_total``
            (typically 6000). ``None`` preserves the default. Ignored for the
            ``ml_patients()`` (legacy) path. See
            ``.claude/plans/synthetic_cohort_growth_plan_20260509.md`` Phase 1.
    """
    if _generator in _SCENARIO_REGIME_TO_NAME:
        return _scenario_to_dataframe(
            _generator,
            seed=seed,
            n_total=n_total,
            imbalance_ratio=imbalance_ratio,
        )

    # Use the same generator as the data_preparer agent for consistency
    from src.repositories.sample_data import SampleDataGenerator
    import numpy as np

    generator = SampleDataGenerator(seed=seed)

    # Use fresh date range (last 30 days) to pass timeliness checks
    # Default range is 365 days which causes staleness warnings
    end_date = datetime.now().isoformat()
    start_date = (datetime.now() - timedelta(days=30)).isoformat()

    df = generator.ml_patients(
        n_patients=n_samples,
        start_date=start_date,
        end_date=end_date,
        positive_rate=positive_rate,
        signal_strength=signal_strength,
        noise_sd=noise_sd,
        signalize_extra_features=signalize_extra_features,
    )

    # Apply class imbalance if requested
    if imbalance_ratio is not None and 0 < imbalance_ratio < 0.5:
        np.random.seed(seed)
        target_col = CONFIG.target_outcome
        n_minority = int(n_samples * imbalance_ratio)
        n_majority = n_samples - n_minority

        # Create imbalanced target: minority class = 1 (discontinuation)
        labels = np.array([0] * n_majority + [1] * n_minority)
        np.random.shuffle(labels)
        df[target_col] = labels

        print(f"  ⚠️  Injected class imbalance: {imbalance_ratio:.1%} minority (class 1)")
        print(f"      Class 0: {n_majority} samples, Class 1: {n_minority} samples")

    # Filter to only the configured brand
    # (or keep all if testing multi-brand)
    if CONFIG.brand:
        # Keep all brands but prioritize the configured one
        pass

    return df


# =============================================================================
# BENTOML HELPER FUNCTIONS
# =============================================================================

# Docker container name for BentoML (dev overlay)
BENTOML_CONTAINER = "e2i_bentoml_dev"
BENTOML_DOCKER_ENDPOINT = "http://localhost:3000"


def _detect_bentoml_container() -> str | None:
    """Detect running BentoML Docker container.

    Returns:
        Container name if running, None otherwise.
    """
    for name in [BENTOML_CONTAINER, "e2i_bentoml"]:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip() == "true":
            return name
    return None


async def register_model_in_docker_bentoml(
    trained_model: Any,
    model_name: str,
    framework: str = "sklearn",
    metadata: dict[str, Any] | None = None,
    fitted_preprocessor: Any = None,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Register a trained model in the Docker BentoML container's model store.

    Saves the model locally via joblib, copies it into the container, and
    runs docker exec to register it in BentoML's model store.

    When fitted_preprocessor and feature_columns are provided, the model is
    bundled as a dict {"model": ..., "preprocessor": ..., "feature_columns": ...}
    so the serving service can apply preprocessing before prediction.

    Args:
        trained_model: Trained scikit-learn/xgboost/lightgbm model object.
        model_name: Name for the model in the BentoML store.
        framework: ML framework ("sklearn", "xgboost", "lightgbm").
        metadata: Optional metadata dict to attach to the model.
        fitted_preprocessor: Optional fitted preprocessor to bundle with model.
        feature_columns: Optional list of feature column names.

    Returns:
        {"success": True, "model_tag": "model_name:hash"} or
        {"success": False, "error": "..."}
    """
    import joblib

    container = _detect_bentoml_container()
    if not container:
        return {"success": False, "error": "BentoML Docker container not running"}

    # 1. Save model to temp file (bundled with preprocessor if available)
    tmp_path = f"/tmp/tier0_{model_name}.pkl"
    if fitted_preprocessor is not None:
        # Extract the internal sklearn ColumnTransformer — the custom
        # ModelTrainerPreprocessor wrapper can't be unpickled in the
        # BentoML container (it doesn't have the E2I project on PYTHONPATH).
        sklearn_pipeline = getattr(fitted_preprocessor, "_pipeline", fitted_preprocessor)
        artifact = {
            "model": trained_model,
            "preprocessor": sklearn_pipeline,
            "feature_columns": feature_columns,
        }
        joblib.dump(artifact, tmp_path)
    else:
        joblib.dump(trained_model, tmp_path)

    try:
        # 2. Copy into container
        cp_result = subprocess.run(
            ["docker", "cp", tmp_path, f"{container}:/tmp/model.pkl"],
            capture_output=True,
            text=True,
        )
        if cp_result.returncode != 0:
            return {"success": False, "error": f"docker cp failed: {cp_result.stderr.strip()}"}

        # 3. Register in BentoML model store via docker exec
        # When bundled (dict with preprocessor), always use pickle via bentoml.picklable_model
        # Framework-specific savers only work with raw model objects
        if fitted_preprocessor is not None:
            meta_dict = metadata or {}
            meta_dict["bundled"] = True
            meta_str = repr(meta_dict)
            register_script = (
                f"import joblib, bentoml; "
                f"artifact = joblib.load('/tmp/model.pkl'); "
                f"bento_model = bentoml.picklable_model.save_model('{model_name}', artifact, metadata={meta_str}); "
                f"print(str(bento_model.tag))"
            )
        else:
            save_fn_map = {
                "sklearn": "bentoml.sklearn.save_model",
                "xgboost": "bentoml.xgboost.save_model",
                "lightgbm": "bentoml.lightgbm.save_model",
            }
            save_fn = save_fn_map.get(framework, "bentoml.sklearn.save_model")
            module_path = save_fn.rsplit(".", 1)[0]  # e.g. "bentoml.sklearn"
            meta_str = repr(metadata or {})
            register_script = (
                f"import joblib, {module_path}; "
                f"model = joblib.load('/tmp/model.pkl'); "
                f"bento_model = {save_fn}('{model_name}', model, metadata={meta_str}); "
                f"print(str(bento_model.tag))"
            )

        exec_result = subprocess.run(
            ["docker", "exec", container, "python", "-c", register_script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if exec_result.returncode != 0:
            return {
                "success": False,
                "error": f"docker exec failed: {exec_result.stderr.strip()[:300]}",
            }

        model_tag = exec_result.stdout.strip()
        return {"success": True, "model_tag": model_tag}

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def restart_docker_bentoml(model_tag: str | None = None) -> dict[str, Any]:
    """Restart the Docker BentoML container and wait for health.

    Optionally sets E2I_BENTOML_MODEL_TAG so the service loads the exact model.
    If model_tag is None, the service will auto-discover the latest model.

    Args:
        model_tag: Optional BentoML model tag to load on startup.

    Returns:
        {"available": True, "endpoint": "..."} or {"available": False, "reason": "..."}
    """
    import httpx

    container = _detect_bentoml_container()
    if not container:
        return {"available": False, "reason": "BentoML Docker container not found"}

    # Set env var for model tag if provided (write to container's env at restart)
    if model_tag:
        # We cannot set env vars on a running container persistently.
        # Instead, the auto-discovery (strategy 3) will find the latest model.
        # Log the intent.
        print(f"    Model tag {model_tag} registered; relying on auto-discovery after restart")

    # Restart container
    restart = subprocess.run(
        ["docker", "restart", container],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if restart.returncode != 0:
        return {"available": False, "reason": f"restart failed: {restart.stderr.strip()}"}

    # Wait for health (BentoML uses /healthz)
    endpoint = BENTOML_DOCKER_ENDPOINT
    async with httpx.AsyncClient() as client:
        for _ in range(40):  # 40s max (BentoML start_period=10s + model load)
            await asyncio.sleep(1)
            try:
                resp = await client.get(f"{endpoint}/healthz", timeout=3.0)
                if resp.status_code == 200:
                    return {"available": True, "endpoint": endpoint}
            except Exception:
                continue

    return {"available": False, "reason": "health check timeout after 40s"}


async def verify_bentoml_predictions(
    endpoint: str,
    sample_features: list,
    service_type: str = "ephemeral",
) -> dict:
    """Verify that BentoML service returns valid predictions.

    Args:
        endpoint: BentoML service endpoint (e.g., http://localhost:3001)
        sample_features: Sample feature data to test
        service_type: Type of BentoML service ("ephemeral" or "persistent")

    Returns:
        {"health_check": True, "prediction_test": True, "predictions": [...], "latency_ms": X}
    """
    import httpx

    result = {"health_check": False, "prediction_test": False}

    # Health check (BentoML @api endpoints use POST)
    async with httpx.AsyncClient() as client:
        try:
            health_resp = await client.post(f"{endpoint}/health", timeout=5.0)
            if health_resp.status_code == 200:
                health_data = health_resp.json()
                result["health_check"] = health_data.get("status") == "healthy"
                result["model_tag"] = health_data.get("model_tag")
        except Exception as e:
            result["health_error"] = str(e)

        # Prediction test — try payload formats for compatibility
        # Ephemeral service uses {"features": ...}
        # Persistent service uses {"input_data": {"features": ...}}
        try:
            payloads = [
                {"features": sample_features},
                {"input_data": {"features": sample_features}},
            ]
            start = time_module.time()
            pred_resp = None
            for payload in payloads:
                pred_resp = await client.post(
                    f"{endpoint}/predict",
                    json=payload,
                    timeout=10.0,
                )
                if pred_resp.status_code != 400:
                    break
            elapsed = (time_module.time() - start) * 1000

            if pred_resp.status_code == 200:
                pred_data = pred_resp.json()
                result["prediction_test"] = True
                result["predictions"] = pred_data.get("predictions")
                result["probabilities"] = pred_data.get("probabilities")
                result["latency_ms"] = elapsed
                result["service_latency_ms"] = pred_data.get("latency_ms")
            else:
                result["prediction_error"] = f"HTTP {pred_resp.status_code}"
                try:
                    result["prediction_error_body"] = pred_resp.text[:500]
                except Exception:
                    pass
        except Exception as e:
            result["prediction_error"] = str(e)

    return result


async def deploy_to_persistent_service(model_tag: str) -> dict:
    """Deploy model to the Docker BentoML service.

    Registers the model in the Docker container's BentoML store (already done
    by register_model_in_docker_bentoml), restarts the container so it picks up
    the new model via auto-discovery, and waits for health.

    Args:
        model_tag: BentoML model tag (for logging; model is already registered)

    Returns:
        {"available": True, "endpoint": "http://localhost:3000"} or
        {"available": False, "reason": "..."}
    """
    container = _detect_bentoml_container()
    if not container:
        return {"available": False, "reason": "BentoML Docker container not running"}

    return await restart_docker_bentoml(model_tag)


async def stop_bentoml_service(pid: int) -> dict:
    """Stop BentoML service by PID.

    Args:
        pid: Process ID to terminate

    Returns:
        {"stopped": True/False, "pid": pid}
    """
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait briefly for graceful shutdown
        await asyncio.sleep(1)
        return {"stopped": True, "pid": pid}
    except ProcessLookupError:
        return {"stopped": True, "pid": pid, "note": "Process already terminated"}
    except Exception as e:
        return {"stopped": False, "error": str(e), "pid": pid}


# =============================================================================
# STEP IMPLEMENTATIONS
# =============================================================================


async def step_1_scope_definer(
    experiment_id: str,
    adaptive_inputs: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    """Step 1: Define ML problem scope.

    Args:
        experiment_id: Unique identifier propagated through the pipeline.
        adaptive_inputs: Optional pre-eval inputs for adaptive success
            criteria (task 05 of adaptive_success_criteria plan). When
            ``ADAPTIVE_CRITERIA=true``, these flow into
            ``criteria_validator.define_success_criteria`` which stashes
            them on ``success_criteria['_adaptive_inputs']`` for the
            evaluator overlay. Pass ``None`` (the default) when adaptive
            criteria are not desired.
    """
    import time as time_mod

    step_start = time_mod.time()

    print_header(1, "SCOPE DEFINER")

    from src.agents.ml_foundation.scope_definer import ScopeDefinerAgent

    # Input preparation
    input_data: Dict[str, Any] = {
        "problem_description": f"Predict patient discontinuation risk for {CONFIG.brand}",
        "business_objective": "Identify high-risk patients early for intervention",
        "target_outcome": CONFIG.target_outcome,
        "problem_type_hint": CONFIG.problem_type,
        "brand": CONFIG.brand,
    }
    # Merge adaptive pre-eval inputs when provided (task 05). The agent
    # forwards these into the state under the same field names; the
    # validator consumes them when ADAPTIVE_CRITERIA is on.
    if adaptive_inputs:
        input_data.update(adaptive_inputs)

    print_input_section(input_data)

    # Processing
    processing_steps = []
    processing_steps.append(("Creating ScopeDefinerAgent", True, None))

    agent = ScopeDefinerAgent()
    processing_steps.append(("Agent initialized", True, None))

    result = await agent.run(input_data)
    processing_steps.append(("Scope definition executed", True, None))

    print_processing_steps(processing_steps)

    # Validation checks
    scope_spec = result.get("scope_spec", {})
    validation_passed = result.get("validation_passed", True)

    checks = [
        (
            "Problem type defined",
            bool(scope_spec.get("problem_type")),
            "problem_type present",
            scope_spec.get("problem_type", "missing"),
        ),
        (
            "Prediction target set",
            bool(scope_spec.get("prediction_target")),
            "prediction_target present",
            scope_spec.get("prediction_target", "missing"),
        ),
        (
            "Minimum samples specified",
            bool(scope_spec.get("minimum_samples")),
            "minimum_samples > 0",
            str(scope_spec.get("minimum_samples", "missing")),
        ),
        (
            "Scope validation",
            validation_passed,
            "validation_passed = True",
            f"validation_passed = {validation_passed}",
        ),
    ]

    print_validation_checks(checks)

    # Metrics
    metrics = [
        ("experiment_id", result.get("experiment_id", experiment_id), None, None),
        ("problem_type", scope_spec.get("problem_type"), None, None),
        ("prediction_target", scope_spec.get("prediction_target"), None, None),
        ("minimum_samples", scope_spec.get("minimum_samples"), None, None),
    ]

    print_metrics_table(metrics)

    # Interpretation
    observations = []
    recommendations = []

    if scope_spec.get("problem_type") == "binary_classification":
        observations.append("Binary classification scope defined for patient risk prediction")
        observations.append(f"Target outcome: {scope_spec.get('prediction_target', 'N/A')}")
    else:
        observations.append(f"Problem type: {scope_spec.get('problem_type', 'unknown')}")

    if scope_spec.get("minimum_samples", 0) < 100:
        observations.append("⚠️  Minimum samples is low for reliable ML training")
        recommendations.append("Consider increasing minimum_samples to 500+")
    else:
        observations.append(
            f"Sample requirement ({scope_spec.get('minimum_samples')}) appropriate for ML"
        )

    print_interpretation(
        "Scope Analysis", observations, recommendations if recommendations else None
    )

    # Final result
    duration = time_mod.time() - step_start
    if validation_passed:
        print_step_result("success", f"Scope definition complete ({duration:.1f}s)")
    else:
        print_step_result("warning", f"Scope has validation warnings ({duration:.1f}s)")

    return result


async def step_2_data_preparer(
    experiment_id: str,
    scope_spec: dict,
    sample_df: pd.DataFrame,
    *,
    skip_leakage_check: bool = False,
    adaptive_fdr_enabled: bool = True,
    adaptive_declared_safe_full_immunity: bool = False,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Step 2: Load and prepare data with QC.

    ``skip_leakage_check`` bypasses the LLM-assisted leakage detector +
    remediator. Use only for clinically-grounded synthetic fixtures
    (e.g. ``--regime scenario_a``) where leakage is impossible by
    construction; the LLM otherwise name-classifies legitimate clinical
    features (e.g. ``journey_status``) as tautological and replaces the
    feature set with a hallucinated recommendation.

    ``data_dir`` routes the agent's ``data_loader`` through
    ``_load_from_files`` so the ``adaptive_validity_check`` (Layer 5)
    audits the actual cohort columns instead of the
    ``SampleDataGenerator`` synthetic schema. Required for any RWD
    cohort run where Layer 1 manifest verdicts (``layer="1"``) need to
    fire on the on-disk JSON. Backlog item #12.
    """
    import time as time_mod

    step_start = time_mod.time()

    print_header(2, "DATA PREPARER")

    from src.agents.ml_foundation.data_preparer import DataPreparerAgent

    # Override required_features with actual columns from sample data
    available_features = [
        col
        for col in sample_df.columns
        if col not in ["patient_journey_id", CONFIG.target_outcome, "brand"]
    ]

    # When data_dir is supplied (real cohort), route the agent's data_loader
    # through _load_from_files so Layer 5 sees the on-disk schema. Otherwise
    # keep the existing SampleDataGenerator path for synthetic regimes.
    is_rwd_run = data_dir is not None
    data_source: str | dict[str, Any]
    if is_rwd_run:
        data_source = {"type": "file_dir", "path": data_dir}
        sample_size = len(sample_df)
    else:
        data_source = "patient_journeys"
        sample_size = 500

    # When a feature manifest is selected on a real RWD cohort, exclude
    # every column the manifest does NOT declare (IDs, audit timestamps,
    # provenance, placeholders). Post-index forbidden manifest features
    # (e.g. ``journey_duration_days``, ``journey_status``, ``brand``)
    # are intentionally NOT excluded here — Layer 5's
    # adaptive_validity_check is responsible for catching them via
    # ``layer="1"`` verdicts. Mirrors the working
    # ``test_csu_full_data_preparer_e2e.py::csu_scope_spec`` pattern.
    excluded_features: list[str] = []
    manifest_source = scope_spec.get("feature_manifest_source")
    if is_rwd_run and manifest_source:
        try:
            if manifest_source == "csu":
                from src.data.manifests import CSU_FEATURES

                manifest_names = {c.name for c in CSU_FEATURES}
            elif manifest_source == "optum":
                from src.data.manifests import OPTUM_FEATURES

                manifest_names = {c.name for c in OPTUM_FEATURES}
            else:
                manifest_names = set()
            if manifest_names:
                excluded_features = [
                    col
                    for col in sample_df.columns
                    if col not in manifest_names and col != CONFIG.target_outcome
                ]
        except ImportError:
            pass

    # Ensure scope_spec has required fields with realistic values.
    # ``sampling_frame_max_drift`` is forwarded so the audit node and the
    # runner gate use the same threshold.
    scope_spec.update(
        {
            "experiment_id": experiment_id,
            "use_sample_data": not is_rwd_run,
            "sample_size": sample_size,
            "prediction_target": CONFIG.target_outcome,
            "problem_type": CONFIG.problem_type,
            "required_features": available_features,
            "excluded_features": excluded_features,
            "max_staleness_days": 90,
            "sampling_frame_max_drift": CONFIG.sampling_frame_max_drift,
        }
    )

    input_data = {
        "scope_spec": scope_spec,
        "data_source": data_source,
        "brand": CONFIG.brand,
        "skip_leakage_check": skip_leakage_check,
        # #594: drives adaptive_validity_check's FDR firing switch. False on
        # synthetic FIXTURE regimes (set by the caller) → static σ-band fallback.
        "adaptive_fdr_enabled": adaptive_fdr_enabled,
        # #604: grants declared-safe features full immunity from FDR auto-drop on
        # legacy synthetic fixtures (manifest leak-free by construction). False on
        # real runs preserves the "overwhelming evidence still drops" backstop.
        "adaptive_declared_safe_full_immunity": adaptive_declared_safe_full_immunity,
    }

    print_input_section(
        {
            "data_source": data_source if is_rwd_run else "patient_journeys",
            "brand": CONFIG.brand,
            "sample_size": len(sample_df),
            "features": f"{len(available_features)} available",
        }
    )

    # Processing
    processing_steps = []
    processing_steps.append(("Creating DataPreparerAgent", True, None))

    agent = DataPreparerAgent()
    processing_steps.append(("Agent initialized", True, None))

    result = await agent.run(input_data)
    processing_steps.append(("Data preparation executed", True, None))

    # Extract nested results
    qc_report = result.get("qc_report", {})
    data_readiness = result.get("data_readiness", {})
    remediation = result.get("remediation", {})

    processing_steps.append(("QC analysis complete", True, qc_report.get("status", "unknown")))

    if remediation.get("status") and remediation.get("status") != "not_needed":
        processing_steps.append(("Remediation applied", True, remediation.get("status")))

    print_processing_steps(processing_steps)

    # Validation checks
    gate_passed = result.get("gate_passed", False)
    overall_score = qc_report.get("overall_score", 0)
    train_samples = data_readiness.get("train_samples", 0)
    val_samples = data_readiness.get("validation_samples", 0)

    checks = [
        ("QC Gate", gate_passed, "gate_passed = True", f"gate_passed = {gate_passed}"),
        (
            "Overall QC Score",
            overall_score >= 0.7 if isinstance(overall_score, (int, float)) else False,
            ">= 0.70",
            f"{overall_score:.2f}"
            if isinstance(overall_score, (int, float))
            else str(overall_score),
        ),
        ("QC training samples", train_samples >= 100, ">= 100", str(train_samples)),
        ("QC validation samples", val_samples >= 30, ">= 30", str(val_samples)),
    ]

    print_validation_checks(checks)

    # Metrics table - QC dimension scores
    completeness = qc_report.get("completeness_score", 0)
    validity = qc_report.get("validity_score", 0)
    consistency = qc_report.get("consistency_score", 0)
    uniqueness = qc_report.get("uniqueness_score", 0)
    timeliness = qc_report.get("timeliness_score", 0)

    metrics = [
        (
            "overall_score",
            overall_score,
            ">= 0.70",
            overall_score >= 0.7 if isinstance(overall_score, (int, float)) else None,
        ),
        (
            "completeness",
            completeness,
            ">= 0.90",
            completeness >= 0.9 if isinstance(completeness, (int, float)) else None,
        ),
        (
            "validity",
            validity,
            ">= 0.90",
            validity >= 0.9 if isinstance(validity, (int, float)) else None,
        ),
        (
            "consistency",
            consistency,
            ">= 0.90",
            consistency >= 0.9 if isinstance(consistency, (int, float)) else None,
        ),
        (
            "uniqueness",
            uniqueness,
            ">= 0.95",
            uniqueness >= 0.95 if isinstance(uniqueness, (int, float)) else None,
        ),
        (
            "timeliness",
            timeliness,
            ">= 0.80",
            timeliness >= 0.8 if isinstance(timeliness, (int, float)) else None,
        ),
        (
            "qc_train_samples",
            train_samples,
            ">= 100",
            train_samples >= 100 if isinstance(train_samples, (int, float)) else None,
        ),
        (
            "qc_validation_samples",
            val_samples,
            ">= 30",
            val_samples >= 30 if isinstance(val_samples, (int, float)) else None,
        ),
    ]

    print_metrics_table(metrics)

    # Interpretation
    observations, recommendations = interpret_qc_scores(qc_report)

    # Add data readiness observations
    if train_samples and val_samples:
        total = train_samples + val_samples
        train_pct = train_samples / total * 100 if total > 0 else 0
        observations.insert(
            0,
            f"QC sample split: {train_samples} QC-train ({train_pct:.0f}%), {val_samples} QC-validation (full training split in Step 5)",
        )

    # Add remediation info if present
    if remediation.get("status") and remediation.get("status") != "not_needed":
        observations.append(f"Remediation was applied: {remediation.get('status')}")
        if remediation.get("actions_taken"):
            for action in remediation.get("actions_taken", [])[:2]:
                observations.append(f"  - {action}")

    print_interpretation(
        "Data Quality Analysis", observations, recommendations if recommendations else None
    )

    # Show blocking issues if gate failed
    if not gate_passed:
        blocking_issues = qc_report.get("blocking_issues", [])
        if blocking_issues:
            print("\n  🚫 Blocking Issues:")
            for issue in blocking_issues[:5]:
                print(f"    • {issue}")

    # Final result
    duration = time_mod.time() - step_start
    if gate_passed:
        print_step_result("success", f"QC Gate PASSED - Training can proceed ({duration:.1f}s)")
    else:
        print_step_result("failed", f"QC Gate FAILED - Training blocked ({duration:.1f}s)")

    return result


async def step_2b_feast_registration(experiment_id: str, state: dict[str, Any]) -> dict[str, Any]:
    """Step 2b: Register features with Feast feature store (gracefully degrading).

    NOTE (FU2 / #528): Feast is INTENTIONALLY absent from the app image — every
    Feast release (0.43–0.63) pins ``tenacity<9``, unsatisfiable against the prod
    ``tenacity==9.1.2`` required by graphiti-core (see requirements.txt, #307). So
    ``import feast`` fails in this container and these feast steps degrade BY DESIGN
    to the custom feature-store fallback — expected, not a bug. Real Feast lives in
    the ``e2i_feast`` sidecar and is exercised by ``tests/integration/test_feast_*``.
    (Whether the prod predictions route should fail-loud vs silently fall back is
    tracked separately — see #532.)
    """
    import time as time_mod

    step_start = time_mod.time()

    print_header("2b", "FEAST FEATURE REGISTRATION")

    result = {
        "status": "skipped",
        "features_registered": 0,
        "errors": [],
    }

    try:
        from src.feature_store import FeatureStoreClient, get_feature_analyzer_adapter

        processing_steps = []

        # Initialize feature store client
        fs_client = FeatureStoreClient()
        processing_steps.append(("FeatureStoreClient initialized", True, None))

        # Get adapter with Feast enabled
        adapter = get_feature_analyzer_adapter(fs_client, enable_feast=True)
        processing_steps.append(("FeatureAnalyzerAdapter created", True, "feast enabled"))

        # Build feature state from pipeline state
        feature_state = {}
        train_df = state.get("train_df")
        if train_df is not None:
            feature_state["X_train"] = train_df
            feature_state["selected_features"] = list(train_df.columns)
            feature_state["feature_importance"] = {
                col: 1.0 / len(train_df.columns) for col in train_df.columns
            }
            # Build generated_features metadata (required by adapter)
            feature_state["generated_features"] = [
                {"name": col, "description": f"Feature: {col}"} for col in train_df.columns
            ]

        # Register features
        reg_result = await adapter.register_features_from_state(
            state=feature_state,
            experiment_id=experiment_id,
            entity_key="hcp_id",
            owner="tier0_pipeline",
            tags=["tier0", "e2e_test", CONFIG.brand.lower()],
        )

        features_registered = reg_result.get("features_registered", 0)
        features_skipped = reg_result.get("features_skipped", 0)
        reg_errors = reg_result.get("errors", [])
        processing_steps.append(
            (
                "Features registered with Feast",
                features_registered > 0 or features_skipped > 0,
                f"{features_registered} registered, {features_skipped} skipped",
            )
        )

        print_input_section(
            {
                "experiment_id": experiment_id,
                "entity_key": "hcp_id",
                "feature_count": len(feature_state.get("selected_features", [])),
            }
        )
        print_processing_steps(processing_steps)

        # Validation
        checks = [
            (
                "Feature group created",
                reg_result.get("feature_group_created", False),
                "True",
                str(reg_result.get("feature_group_created", False)),
            ),
            ("Features registered", features_registered > 0, "> 0", str(features_registered)),
            (
                "No registration errors",
                len(reg_errors) == 0,
                "0 errors",
                f"{len(reg_errors)} errors",
            ),
        ]
        print_validation_checks(checks)

        result = {
            "status": "success" if features_registered > 0 else "warning",
            "features_registered": features_registered,
            "features_skipped": features_skipped,
            "feature_group_created": reg_result.get("feature_group_created", False),
            "errors": reg_errors,
        }

        duration = time_mod.time() - step_start
        if features_registered > 0:
            print_step_result(
                "success",
                f"Feast registration complete: {features_registered} features ({duration:.1f}s)",
            )
        else:
            print_step_result(
                "warning", f"Feast registration: 0 features registered ({duration:.1f}s)"
            )

    except Exception as e:
        duration = time_mod.time() - step_start
        result["status"] = "skipped"
        result["errors"] = [str(e)]
        print_step_result("warning", f"Feast registration skipped: {e} ({duration:.1f}s)")

    return result


async def step_2c_feast_freshness_check(state: dict[str, Any]) -> dict[str, Any]:
    """Step 2c: Check feature freshness in Feast (gracefully degrading).

    Feast is absent from the app image by design (tenacity conflict, #307) → degrades
    to fallback; see step_2b_feast_registration docstring + #532 (FU2 / #528).
    """
    import time as time_mod

    step_start = time_mod.time()

    print_header("2c", "FEAST FRESHNESS CHECK")

    result = {
        "status": "skipped",
        "fresh": None,
        "stale_features": [],
        "errors": [],
    }

    try:
        from src.feature_store import FeatureStoreClient, get_feature_analyzer_adapter

        processing_steps = []

        # Initialize
        fs_client = FeatureStoreClient()
        adapter = get_feature_analyzer_adapter(fs_client, enable_feast=True)
        processing_steps.append(("FeatureAnalyzerAdapter initialized", True, None))

        # Build feature refs from train_df columns
        train_df = state.get("train_df")
        if train_df is not None:
            feature_refs = [f"hcp_features:{col}" for col in train_df.columns[:20]]
        else:
            feature_refs = ["hcp_features:default_feature"]

        processing_steps.append(("Feature refs built", True, f"{len(feature_refs)} refs"))

        # Check freshness
        freshness_result = await adapter.check_feature_freshness(
            feature_refs=feature_refs,
            max_staleness_hours=24.0,
        )

        is_fresh = freshness_result.get("fresh", False)
        stale_features = freshness_result.get("stale_features", [])
        processing_steps.append(
            (
                "Freshness check completed",
                True,
                f"{'fresh' if is_fresh else f'{len(stale_features)} stale'}",
            )
        )

        print_input_section(
            {
                "feature_refs_count": len(feature_refs),
                "max_staleness_hours": 24.0,
            }
        )
        print_processing_steps(processing_steps)

        # Validation
        checks = [
            ("Freshness check executed", True, "completed", "completed"),
            (
                "Features fresh",
                is_fresh,
                "all fresh",
                f"{len(stale_features)} stale" if stale_features else "all fresh",
            ),
        ]
        print_validation_checks(checks)

        result = {
            "status": "success" if is_fresh else "warning",
            "fresh": is_fresh,
            "stale_features": stale_features,
            "checked_at": freshness_result.get("checked_at", ""),
            "errors": [],
        }

        duration = time_mod.time() - step_start
        if is_fresh:
            print_step_result("success", f"All features fresh ({duration:.1f}s)")
        else:
            print_step_result(
                "warning", f"{len(stale_features)} stale features detected ({duration:.1f}s)"
            )

    except Exception as e:
        duration = time_mod.time() - step_start
        result["status"] = "skipped"
        result["errors"] = [str(e)]
        print_step_result("warning", f"Feast freshness check skipped: {e} ({duration:.1f}s)")

    return result


def _build_cohort_config(patient_df: pd.DataFrame, min_data_quality: float) -> Any:
    """Build the harness cohort config, ADAPTING to the available quality signal.

    The mart/RWD adapters already perform the real cohorting (naive-at-index +
    transparent claim-count filter) upstream, so this harness step is a light QC
    gate, not the primary selection. The quality threshold is CONFIG-driven
    (``--cohort-min-quality`` / ``CONFIG.cohort_min_data_quality``) rather than a
    hardcoded constant, and it is FIELD-ADAPTIVE: when the frame carries
    ``data_quality_score`` the gate keeps rows at/above the threshold, but when
    the column is absent (a cohort that filtered upstream and emits no quality
    metadata) the quality criterion is a NO-OP so the step cannot erroneously
    zero out an already-constructed cohort.
    """
    from src.agents.cohort_constructor.types import (
        CohortConfig,
        Criterion,
        CriterionType,
        Operator,
    )

    has_quality = "data_quality_score" in patient_df.columns
    inclusion_criteria: list[Any] = []
    required_fields = ["patient_journey_id", "brand"]
    if has_quality:
        inclusion_criteria.append(
            Criterion(
                field="data_quality_score",
                operator=Operator.GREATER_EQUAL,
                value=min_data_quality,
                criterion_type=CriterionType.INCLUSION,
                description="Minimum data quality score",
                clinical_rationale="Ensure data quality for reliable ML predictions",
            )
        )
        required_fields.append("data_quality_score")
    return CohortConfig(
        cohort_name=f"{CONFIG.brand} Test Cohort",
        brand=CONFIG.brand.lower(),
        indication="test",
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=[],
        temporal_requirements=None,
        required_fields=required_fields,
        version="1.0.0-test",
        status="active",
        clinical_rationale="Test cohort using sample data fields - relaxed criteria for testing",
        regulatory_justification="Test configuration for MLOps workflow validation",
    )


async def step_3_cohort_constructor(patient_df: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    """Step 3: Build patient cohort."""
    import time as time_mod

    step_start = time_mod.time()

    print_header(3, "COHORT CONSTRUCTOR")

    from src.agents.cohort_constructor import CohortConstructorAgent

    quality_gate = (
        f"data_quality_score >= {CONFIG.cohort_min_data_quality}"
        if "data_quality_score" in patient_df.columns
        else "none (data_quality_score absent — cohort constructed upstream)"
    )
    print_input_section(
        {
            "input_patients": len(patient_df),
            "brand": CONFIG.brand,
            "inclusion_criteria": quality_gate,
            "exclusion_criteria": "None (maximize sample size)",
        }
    )

    # Processing
    processing_steps = []
    processing_steps.append(("Creating CohortConstructorAgent", True, None))

    agent = CohortConstructorAgent(enable_observability=CONFIG.enable_opik)
    processing_steps.append(("Agent initialized", True, None))

    # Create test config — CONFIG-driven threshold + field-adaptive quality gate.
    test_config = _build_cohort_config(patient_df, CONFIG.cohort_min_data_quality)

    processing_steps.append(("Cohort config created", True, quality_gate))

    eligible_df, result = await agent.run(
        patient_df=patient_df,
        config=test_config,
    )
    processing_steps.append(("Cohort construction executed", True, f"{len(eligible_df)} eligible"))

    print_processing_steps(processing_steps)

    # Validation checks
    eligible_count = len(eligible_df)
    input_count = len(patient_df)
    excluded_count = input_count - eligible_count
    retention_rate = eligible_count / input_count if input_count > 0 else 0

    checks = [
        (
            "Minimum cohort size",
            eligible_count >= CONFIG.min_eligible_patients,
            f">= {CONFIG.min_eligible_patients}",
            str(eligible_count),
        ),
        ("Retention rate", retention_rate >= 0.5, ">= 50%", f"{retention_rate:.1%}"),
        ("Cohort status", result.status == "completed", "completed", result.status),
    ]

    print_validation_checks(checks)

    # Metrics
    metrics = [
        ("input_patients", input_count, None, None),
        (
            "eligible_patients",
            eligible_count,
            f">= {CONFIG.min_eligible_patients}",
            eligible_count >= CONFIG.min_eligible_patients,
        ),
        ("excluded_patients", excluded_count, None, None),
        ("retention_rate", retention_rate, ">= 0.50", retention_rate >= 0.5),
        ("cohort_id", result.cohort_id, None, None),
    ]

    print_metrics_table(metrics)

    # Interpretation
    observations = []
    recommendations = []

    observations.append(
        f"Patient flow: {input_count} → {eligible_count} ({retention_rate:.1%} retention)"
    )
    observations.append(f"Excluded {excluded_count} patients based on eligibility criteria")

    if eligible_count < CONFIG.min_eligible_patients:
        observations.append(
            f"⚠️  Cohort size ({eligible_count}) below minimum ({CONFIG.min_eligible_patients})"
        )
        recommendations.append("Relax eligibility criteria or generate more sample data")
        recommendations.append("Consider lowering data_quality_score threshold")
    else:
        observations.append(f"Cohort size ({eligible_count}) sufficient for ML training")

    if retention_rate < 0.5:
        observations.append(
            f"⚠️  High exclusion rate ({1 - retention_rate:.1%}) may indicate data quality issues"
        )
        recommendations.append("Review exclusion criteria for potential over-filtering")

    # Target distribution in cohort
    if CONFIG.target_outcome in eligible_df.columns:
        target_dist = eligible_df[CONFIG.target_outcome].value_counts()
        minority_ratio = target_dist.min() / target_dist.sum() if target_dist.sum() > 0 else 0
        observations.append(f"Target class distribution: {dict(target_dist)}")
        if minority_ratio < 0.2:
            observations.append(f"⚠️  Class imbalance detected ({minority_ratio:.1%} minority)")

    print_interpretation(
        "Cohort Analysis", observations, recommendations if recommendations else None
    )

    # Final result
    duration = time_mod.time() - step_start
    if eligible_count >= CONFIG.min_eligible_patients:
        print_step_result(
            "success", f"Cohort constructed ({eligible_count} patients, {duration:.1f}s)"
        )
    else:
        print_step_result(
            "warning",
            f"Cohort below minimum size ({eligible_count}/{CONFIG.min_eligible_patients}, {duration:.1f}s)",
        )

    return eligible_df, result


async def step_4_model_selector(
    experiment_id: str,
    scope_spec: dict,
    qc_report: dict,
    feature_characteristics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Step 4: Select model candidate."""
    import time as time_mod

    step_start = time_mod.time()

    print_header(4, "MODEL SELECTOR")

    from src.agents.ml_foundation.model_selector import ModelSelectorAgent

    # D2.2 (2026-05-05): runner-side qc_report normalization shim
    # removed. The data_preparer producer now writes qc_passed/
    # qc_errors/qc_warnings directly into qc_report (see
    # data_preparer/agent.py:158-177); the runtime contract is
    # pinned by QCReportSchema in data_preparer/schemas.py.
    input_data: dict[str, Any] = {
        "scope_spec": scope_spec,
        "qc_report": qc_report,
        "skip_benchmarks": False,  # Enable benchmarks to evaluate alternatives
    }
    if feature_characteristics:
        input_data["feature_characteristics"] = feature_characteristics

    print_input_section(
        {
            "problem_type": scope_spec.get("problem_type", "binary_classification"),
            "qc_passed": qc_report.get("qc_passed"),
            "skip_benchmarks": False,
        }
    )

    # Processing
    processing_steps = []
    processing_steps.append(("Creating ModelSelectorAgent", True, None))

    agent = ModelSelectorAgent()
    processing_steps.append(("Agent initialized", True, None))

    result = await agent.run(input_data)

    # Extract candidate info - model_candidate has the structured output
    candidate = result.get("model_candidate", {})
    if candidate:
        algo_name = candidate.get("algorithm_name", "Unknown")
        # Use default_hyperparameters from agent output
        hyperparams = candidate.get("default_hyperparameters", {})
        selection_score = candidate.get("selection_score", 0)
        interpretability = candidate.get("interpretability_score", 0)
        processing_steps.append(("Model selection executed", True, algo_name))
    else:
        algo_name = "LogisticRegression (fallback)"
        hyperparams = {}
        selection_score = 0
        interpretability = 0
        processing_steps.append(("Model selection executed", False, "Using fallback"))

    # Extract alternative candidates
    alternatives = result.get("alternative_candidates", [])
    if alternatives:
        alt_names = [alt.get("algorithm_name", "Unknown") for alt in alternatives[:3]]
        processing_steps.append(("Alternatives evaluated", True, f"{len(alternatives)} candidates"))
    else:
        alt_names = []

    # Extract selection rationale
    rationale_dict = result.get("selection_rationale", {})
    primary_reason = rationale_dict.get("primary_reason", "")
    supporting_factors = rationale_dict.get("supporting_factors", [])
    alternatives_considered = rationale_dict.get("alternatives_considered", [])

    print_processing_steps(processing_steps)

    # Validation checks
    has_candidate = bool(candidate and algo_name != "Unknown")
    has_error = bool(result.get("error"))
    has_rationale = bool(primary_reason)

    checks = [
        (
            "Model candidate selected",
            has_candidate,
            "candidate present",
            algo_name if has_candidate else "missing (will use fallback)",
        ),
        (
            "Selection rationale provided",
            has_rationale,
            "rationale present",
            primary_reason[:50] if primary_reason else "none",
        ),
        (
            "Alternatives evaluated",
            len(alternatives) > 0 or len(alternatives_considered) > 0,
            "> 0 alternatives",
            f"{max(len(alternatives), len(alternatives_considered))} evaluated",
        ),
        (
            "No selection errors",
            not has_error,
            "no errors",
            result.get("error", "none")[:50] if has_error else "none",
        ),
    ]

    print_validation_checks(checks)

    # Metrics table
    metrics = [
        ("algorithm", algo_name, None, None),
        (
            "selection_score",
            f"{selection_score:.3f}" if selection_score else "N/A",
            "> 0.5",
            selection_score > 0.5 if selection_score else None,
        ),
        (
            "interpretability_score",
            f"{interpretability:.2f}" if interpretability else "N/A",
            None,
            None,
        ),
        ("hyperparameters", f"{len(hyperparams)} default params", None, None),
        (
            "alternatives_evaluated",
            len(alternatives) if alternatives else len(alternatives_considered),
            "> 0",
            len(alternatives) > 0 or len(alternatives_considered) > 0,
        ),
    ]

    print_metrics_table(metrics)

    # Print alternatives table if available
    if alternatives or alternatives_considered:
        print("\n  📋 Candidates Evaluated:")
        print("    " + "-" * 60)
        print(f"    {'Rank':<6}{'Algorithm':<25}{'Score':<12}{'Status':<15}")
        print("    " + "-" * 60)
        score_str = f"{selection_score:.3f}" if selection_score else "N/A"
        print(f"    {'1':<6}{algo_name:<25}{score_str:<12}{'✅ SELECTED':<15}")

        # Show alternatives
        alt_list = alternatives if alternatives else alternatives_considered
        for i, alt in enumerate(alt_list[:4], start=2):
            if isinstance(alt, dict):
                alt_name = alt.get("algorithm_name", alt.get("name", "Unknown"))
                alt_score = alt.get("selection_score", alt.get("score", 0))
                alt_reason = alt.get("rejection_reason", "Not selected")[:20]
            else:
                alt_name = str(alt)
                alt_score = 0
                alt_reason = "Evaluated"
            score_str = f"{alt_score:.3f}" if alt_score else "N/A"
            print(f"    {i:<6}{alt_name:<25}{score_str:<12}{alt_reason:<15}")
        print("    " + "-" * 60)

    # Interpretation with selection justification
    observations = []
    recommendations = []

    if has_candidate:
        observations.append(
            f"Selected: {algo_name} (score: {selection_score:.3f})"
            if selection_score
            else f"Selected: {algo_name}"
        )

        # Primary selection reason
        if primary_reason:
            observations.append(f"Primary reason: {primary_reason}")

        # Supporting factors
        if supporting_factors:
            factors_str = ", ".join(supporting_factors[:3])
            observations.append(f"Supporting factors: {factors_str}")

        # Hyperparameters to tune
        if hyperparams:
            param_names = list(hyperparams.keys())[:4]
            observations.append(f"HPO will tune: {', '.join(param_names)}")
        else:
            observations.append("HPO will use default search space in Step 5")
    else:
        observations.append("⚠️  No model candidate returned by selector")
        observations.append("Falling back to LogisticRegression as default")
        recommendations.append("Review selector agent logs for issues")

    # Algorithm-specific observations
    if "Logistic" in algo_name:
        observations.append("LogisticRegression: Interpretable, fast, good baseline")
    elif "RandomForest" in algo_name:
        observations.append("RandomForest: Robust ensemble, handles non-linearity")
    elif "XGB" in algo_name or "Gradient" in algo_name:
        observations.append(f"{algo_name}: High performance, may need regularization")
    elif "LightGBM" in algo_name:
        observations.append("LightGBM: Fast training, memory efficient")

    print_interpretation(
        "Model Selection Analysis", observations, recommendations if recommendations else None
    )

    # Final result
    duration = time_mod.time() - step_start
    if has_candidate:
        print_step_result("success", f"Model selected: {algo_name} ({duration:.1f}s)")
    else:
        print_step_result("warning", f"Using fallback model ({duration:.1f}s)")

    return result


async def step_5_model_trainer(
    experiment_id: str,
    model_candidate: Any,
    qc_report: dict,
    X: pd.DataFrame,
    y: pd.Series,
    success_criteria: dict | None = None,
    *,
    entity_ids: pd.Series | None = None,
    dates: pd.Series | None = None,
    split_mode: str = "auto",
    pre_assigned_splits: Dict[Any, str] | None = None,
    cost_matrix: dict | None = None,
) -> dict[str, Any]:
    """Step 5: Train model.

    Args:
        experiment_id: Experiment identifier.
        model_candidate: Model selection dict (algorithm_name, hyperparams).
        qc_report: Quality control report from data_preparer.
        X: Feature matrix, aligned row-for-row with ``y`` / ``entity_ids`` /
            ``dates``.
        y: Target vector.
        success_criteria: Optional success criteria dict propagated from
            scope_definer.
        cost_matrix: Block 5 (#10) optional cost matrix for the
            ``business_utility`` metric — see scope_definer's
            ``_validate_cost_matrix`` for the expected shape. Threaded into
            the model_trainer state so the evaluator can compute the
            metric at the chosen threshold.
        entity_ids: Optional pandas Series of entity identifiers (e.g.
            ``patient_journey_id``) aligned with ``X``. Required when
            ``split_mode`` is ``combined`` or when ``split_mode`` is ``auto``
            and ``dates`` is also supplied.
        dates: Optional pandas Series of dates aligned with ``X``. Used together
            with ``entity_ids`` to drive the combined temporal+entity split.
        split_mode: Split strategy selector.

            - ``"auto"`` (default): use ``combined`` when both ``entity_ids``
              and ``dates`` are provided; otherwise fall back to the legacy
              stratified random split.
            - ``"random"``: explicit opt-out — always run the legacy
              stratified random split.
            - ``"combined"``: explicit opt-in — error out if entity/date
              series are missing.
        pre_assigned_splits: Optional mapping of entity_id → split label
            (``"train"``/``"val"``/``"test"``/``"holdout"``). Set this when
            replaying from a cached tier0 run to **forbid re-splitting** —
            the function will reuse the cached assignments verbatim.

    Returns:
        State dict including the standard model_trainer outputs plus
        ``split_assignments`` (``Dict[entity_id, str]``) when entity_ids are
        available. ``split_assignments`` is the contract that downstream
        consumers (and the tier0 cache) must persist so re-runs do not
        re-derive splits.
    """
    import time as time_mod

    step_start = time_mod.time()

    print_header(5, "MODEL TRAINER")

    from src.agents.ml_foundation.model_trainer import ModelTrainerAgent
    from sklearn.linear_model import LogisticRegression

    # Ensure model_candidate has all required fields
    if model_candidate is None or not isinstance(model_candidate, dict):
        model_candidate = {}

    # Ensure all required fields exist
    if "algorithm_name" not in model_candidate:
        model_candidate["algorithm_name"] = "LogisticRegression"
    if "algorithm_class" not in model_candidate:
        model_candidate["algorithm_class"] = "sklearn.linear_model.LogisticRegression"
    if "hyperparameter_search_space" not in model_candidate:
        model_candidate["hyperparameter_search_space"] = {
            "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            "max_iter": {"type": "int", "low": 100, "high": 500},
        }
    if "default_hyperparameters" not in model_candidate:
        model_candidate["default_hyperparameters"] = {"C": 1.0, "max_iter": 200}

    # D2.2 (2026-05-05): runner-side qc_report normalization shim
    # removed. data_preparer/agent.py:158-177 now writes qc_passed
    # directly into qc_report; QCReportSchema pins the contract.

    # ------------------------------------------------------------------
    # Resolve split strategy
    # ------------------------------------------------------------------
    # Three branches — in priority order:
    #   1. pre_assigned_splits provided     -> reuse verbatim (cache reload)
    #   2. split_mode allows "combined"     -> combined_split (entity+temporal)
    #   3. fall back to legacy random+stratified 4-way
    # ------------------------------------------------------------------
    from sklearn.model_selection import train_test_split

    have_entity_and_date = entity_ids is not None and dates is not None
    if split_mode not in {"auto", "random", "combined"}:
        raise ValueError(
            f"split_mode must be one of 'auto'|'random'|'combined', got {split_mode!r}"
        )
    if split_mode == "combined" and not have_entity_and_date:
        raise ValueError("split_mode='combined' requires both entity_ids and dates to be passed")

    use_combined = split_mode == "combined" or (split_mode == "auto" and have_entity_and_date)

    split_assignments: Dict[Any, str] = {}
    split_strategy = "random_stratified_4way"

    if pre_assigned_splits is not None:
        # ----- branch 1: cache-replay path ---------------------------------
        # Refuse to re-split. Reuse the cached entity → label mapping.
        if entity_ids is None:
            raise ValueError(
                "pre_assigned_splits supplied without entity_ids — cannot "
                "reapply cached splits without an entity column"
            )
        # Validate label vocabulary BEFORE building masks: a typo like
        # "trian" would otherwise produce silently-empty splits and a
        # nonsensical training run (4-IMP-1).
        valid_labels = {"train", "val", "test", "holdout"}
        unique_supplied_labels = set(pre_assigned_splits.values())
        invalid_labels = unique_supplied_labels - valid_labels
        if invalid_labels:
            raise ValueError(
                f"pre_assigned_splits contains unknown split labels: "
                f"{sorted(invalid_labels)}; expected only {sorted(valid_labels)}"
            )
        # Realign labels to X.index so the row-mask comparisons line up
        # with X / y exactly (4-IMP-2: mirror branch 2's index alignment).
        labels = pd.Series(entity_ids.values, index=X.index).map(pre_assigned_splits)
        if labels.isna().any():
            missing = int(labels.isna().sum())
            raise ValueError(
                f"pre_assigned_splits is missing {missing} entity_ids present "
                "in the current data — refusing to silently re-split"
            )
        train_mask = labels == "train"
        val_mask = labels == "val"
        test_mask = labels == "test"
        holdout_mask = labels == "holdout"
        # Preserve original X indices so the model trainer's split validator
        # sees disjoint row index sets (resetting them all to [0..N-1] would
        # trigger a false-positive duplicate-indices alarm).
        train_X, train_y = X[train_mask], y[train_mask]
        val_X, val_y = X[val_mask], y[val_mask]
        test_X, test_y = X[test_mask], y[test_mask]
        holdout_X, holdout_y = X[holdout_mask], y[holdout_mask]
        split_assignments = dict(pre_assigned_splits)
        split_strategy = "cached_replay"
        print(
            f"  Reusing cached split assignments "
            f"(train={train_mask.sum()}, val={val_mask.sum()}, "
            f"test={test_mask.sum()}, holdout={holdout_mask.sum()})"
        )

    elif use_combined:
        # ----- branch 2: combined entity+temporal split --------------------
        # Step A: peel off 5% holdout (entity-isolated) via deterministic
        #   permutation, so the same entities always land in holdout across
        #   reruns.
        # Step B: try combined_split on the remaining 95% to produce
        #   train/val/test using temporal + entity-level boundaries.  If the
        #   date span is too compressed for the desired ratios, fall back
        #   to an entity-aware stratified split that still preserves entity
        #   isolation (this is what the synthetic generator typically
        #   produces — 30-day span, 1500 patients).
        # ------------------------------------------------------------------
        from src.repositories.data_splitter import DataSplitter

        # Align entity_ids and dates to X's index (caller passes them as
        # Series; we want the values keyed by X's row index so the split
        # masks can be applied directly to X / y without losing the
        # globally-unique indices that downstream split-validation needs.)
        if entity_ids is not None:
            eids_aligned = pd.Series(entity_ids.values, index=X.index, dtype=entity_ids.dtype)
        else:  # pragma: no cover - defensive
            eids_aligned = pd.Series([None] * len(X), index=X.index)
        if dates is not None:
            dates_aligned = pd.Series(pd.to_datetime(dates.values), index=X.index)
        else:  # pragma: no cover - defensive
            dates_aligned = pd.Series(pd.NaT, index=X.index)

        # Step A — entity-level holdout via deterministic permutation
        unique_entities = list(eids_aligned.unique())
        rng = np.random.default_rng(42)
        permuted = list(rng.permutation(len(unique_entities)))
        holdout_n = max(1, int(round(len(unique_entities) * 0.05)))
        holdout_entities = {unique_entities[i] for i in permuted[:holdout_n]}
        holdout_mask = eids_aligned.isin(holdout_entities)
        holdout_X = X[holdout_mask].copy()
        holdout_y = y[holdout_mask].copy()

        # Working subset (95%) — keeps original X-indices so the model
        # trainer's split-validation sees disjoint row indices.
        rest_X = X[~holdout_mask].copy()
        rest_y = y[~holdout_mask].copy()
        rest_eids = eids_aligned[~holdout_mask].copy()
        rest_dates = dates_aligned[~holdout_mask].copy()

        # Step B — try combined_split on the 95% remainder.  Construct a
        # frame keyed by X's original row index so we can pull row labels
        # back out and mask X/y directly.
        work_df = pd.DataFrame(
            {
                "__entity_id__": rest_eids,
                "__date__": rest_dates,
            },
            index=rest_X.index,
        )
        try:
            date_min = work_df["__date__"].min()
            date_max = work_df["__date__"].max()
            span_days = max((date_max - date_min).days, 1)
        except (TypeError, ValueError):
            # Narrow catch: ``.min()`` on a non-datetime column raises
            # TypeError; ``.days`` on NaT raises ValueError. Anything else
            # is a real bug and should propagate. (4-MIN-5)
            span_days = 30
        val_days = max(1, int(round(span_days * 0.20)))
        test_days = max(1, int(round(span_days * 0.15)))

        splitter = DataSplitter(random_seed=42)
        # combined_split resets indices internally — we therefore need to
        # carry the original X-index through a sentinel column so we can
        # rebuild masks after the split.
        work_df_for_split = work_df.copy()
        work_df_for_split["__row_id__"] = rest_X.index
        rest_split = splitter.combined_split(
            work_df_for_split,
            date_column="__date__",
            entity_column="__entity_id__",
            val_days=val_days,
            test_days=test_days,
        )

        train_ids = list(rest_split.train["__row_id__"]) if len(rest_split.train) else []
        val_ids = list(rest_split.val["__row_id__"]) if len(rest_split.val) else []
        test_ids = list(rest_split.test["__row_id__"]) if len(rest_split.test) else []
        # combined_split is "usable" when its output lands inside the
        # E2I split policy gates (60/20/15 over the 95% remainder of the
        # data, evaluated with the same ±2% tolerance the model trainer's
        # split_enforcer uses on the global ratios).  A skewed split
        # (e.g. 67/19/14 because the date span is too narrow for the
        # configured val_days/test_days) gets demoted to the stratified
        # fallback so the model trainer's strict ratio gates pass.
        # ----------------------------------------------------------------
        # Map global ratios → 95%-remainder ratios:
        #   train: 0.60 / 0.95 ≈ 0.6316
        #   val:   0.20 / 0.95 ≈ 0.2105
        #   test:  0.15 / 0.95 ≈ 0.1579
        # Tolerance on the remainder: 0.02 / 0.95 ≈ 0.0211
        rest_size = len(rest_X)
        target_train = 0.60 / 0.95
        target_val = 0.20 / 0.95
        target_test = 0.15 / 0.95
        ratio_tol = 0.02 / 0.95
        usable = (
            bool(train_ids)
            and bool(val_ids)
            and bool(test_ids)
            and rest_size > 0
            and abs(len(train_ids) / rest_size - target_train) <= ratio_tol
            and abs(len(val_ids) / rest_size - target_val) <= ratio_tol
            and abs(len(test_ids) / rest_size - target_test) <= ratio_tol
        )
        if usable:
            train_X = rest_X.loc[train_ids].copy()
            val_X = rest_X.loc[val_ids].copy()
            test_X = rest_X.loc[test_ids].copy()
            train_y = rest_y.loc[train_ids].copy()
            val_y = rest_y.loc[val_ids].copy()
            test_y = rest_y.loc[test_ids].copy()
            for row_id in train_ids:
                split_assignments[eids_aligned.loc[row_id]] = "train"
            for row_id in val_ids:
                split_assignments[eids_aligned.loc[row_id]] = "val"
            for row_id in test_ids:
                split_assignments[eids_aligned.loc[row_id]] = "test"
            split_strategy = "combined_temporal_entity_with_holdout"
        else:
            print(
                "  ⚠️  combined_split produced empty or unbalanced bucket(s) "
                "on the available date range — falling back to entity-aware "
                "stratified random split for train/val/test (holdout already "
                "entity-isolated)"
            )
            # Stratified random over rest_X (preserves original indices).
            stage1_X, test_X, stage1_y, test_y, stage1_eids, test_eids = train_test_split(
                rest_X,
                rest_y,
                rest_eids,
                test_size=0.15 / 0.95,
                stratify=rest_y,
                random_state=42,
            )
            train_X, val_X, train_y, val_y, train_eids, val_eids = train_test_split(
                stage1_X,
                stage1_y,
                stage1_eids,
                test_size=0.25,
                stratify=stage1_y,
                random_state=42,
            )
            for eid in train_eids:
                split_assignments[eid] = "train"
            for eid in val_eids:
                split_assignments[eid] = "val"
            for eid in test_eids:
                split_assignments[eid] = "test"
            split_strategy = "combined_fallback_stratified_with_holdout"
        for eid in holdout_entities:
            split_assignments[eid] = "holdout"
        print(
            f"  Combined entity+temporal split (strategy={split_strategy}): "
            f"train={len(train_X)}, val={len(val_X)}, "
            f"test={len(test_X)}, holdout={len(holdout_X)}"
        )

    else:
        # ----- branch 3: legacy stratified random 4-way --------------------
        # Stage 1: peel off 5% holdout, stratified on y -> trainval_test (95%) + holdout (5%)
        trainval_test_X, holdout_X, trainval_test_y, holdout_y = train_test_split(
            X,
            y,
            test_size=0.05,
            stratify=y,
            random_state=42,
        )
        # Stage 2: from trainval_test (95%), peel off test -> trainval (80%) + test (15% of total)
        trainval_X, test_X, trainval_y, test_y = train_test_split(
            trainval_test_X,
            trainval_test_y,
            test_size=0.15 / 0.95,
            stratify=trainval_test_y,
            random_state=42,
        )
        # Stage 3: from trainval (80%), split into train (60%) + val (20%)
        train_X, val_X, train_y, val_y = train_test_split(
            trainval_X,
            trainval_y,
            test_size=0.25,
            stratify=trainval_y,
            random_state=42,
        )
        if entity_ids is not None:
            # Even on the random path, record entity → split mapping so the
            # cache can persist split_assignments for re-runs.
            # Note: entity_ids is keyed by X's original index, which the
            # train_test_split splits preserve.
            eids_aligned = pd.Series(entity_ids.values, index=X.index, dtype=entity_ids.dtype)
            for idx in train_X.index:
                split_assignments[eids_aligned.loc[idx]] = "train"
            for idx in val_X.index:
                split_assignments[eids_aligned.loc[idx]] = "val"
            for idx in test_X.index:
                split_assignments[eids_aligned.loc[idx]] = "test"
            for idx in holdout_X.index:
                split_assignments[eids_aligned.loc[idx]] = "holdout"
        # Intentionally do NOT reset_index: the model trainer's split
        # validator detects "duplicate indices between splits" by
        # comparing row index sets — resetting all four to [0..N-1] would
        # produce false-positive leakage detections. Keeping the original
        # global indices guarantees disjoint sets across splits.

    train_size = len(train_X)
    val_size = len(val_X)
    test_size = len(test_X)
    holdout_size = len(holdout_X)
    n = train_size + val_size + test_size + holdout_size

    train_data = {"X": train_X, "y": train_y, "row_count": train_size}
    validation_data = {"X": val_X, "y": val_y, "row_count": val_size}
    test_data = {"X": test_X, "y": test_y, "row_count": test_size}
    holdout_data = {"X": holdout_X, "y": holdout_y, "row_count": holdout_size}

    feature_columns = list(X.columns)

    # Input section
    print_input_section(
        {
            "algorithm": model_candidate["algorithm_name"],
            "total_samples": n,
            "train_samples": f"{train_size} ({train_size / n:.0%})",
            "validation_samples": f"{val_size} ({val_size / n:.0%})",
            "test_samples": f"{test_size} ({test_size / n:.0%})",
            "holdout_samples": f"{holdout_size} ({holdout_size / n:.0%})",
            "hpo_trials": CONFIG.hpo_trials,
            "enable_mlflow": CONFIG.enable_mlflow,
        }
    )

    # Processing
    processing_steps = []
    processing_steps.append(("Creating ModelTrainerAgent", True, None))

    agent = ModelTrainerAgent()
    processing_steps.append(("Agent initialized", True, None))

    input_data = {
        "experiment_id": experiment_id,
        "model_candidate": model_candidate,
        "qc_report": qc_report,
        "enable_hpo": True,
        "hpo_trials": CONFIG.hpo_trials,
        "problem_type": CONFIG.problem_type,
        "train_data": train_data,
        "validation_data": validation_data,
        "test_data": test_data,
        "holdout_data": holdout_data,
        "enable_mlflow": CONFIG.enable_mlflow,
        "feature_columns": feature_columns,
        "success_criteria": success_criteria or {},
        "min_samples_per_split": CONFIG.min_samples_per_split,
        # Block 5 (#10): forward the cost matrix when scope_definer carried
        # one onto the spec. None when not configured — evaluator skips
        # business_utility silently.
        "cost_matrix": cost_matrix,
    }

    processing_steps.append(("HPO optimization", True, f"{CONFIG.hpo_trials} trials"))
    result = await agent.run(input_data)

    # Check class imbalance
    imbalance_detected = result.get("imbalance_detected", False)
    if imbalance_detected:
        processing_steps.append(
            ("Class imbalance detected", True, result.get("imbalance_severity", "unknown"))
        )
        processing_steps.append(
            ("Remediation applied", True, result.get("recommended_strategy", "N/A"))
        )

    processing_steps.append(
        ("Model training complete", True, f"AUC={result.get('auc_roc', 'N/A')}")
    )

    model_uri = result.get("model_artifact_uri") or result.get("mlflow_model_uri")
    if CONFIG.enable_mlflow and model_uri:
        processing_steps.append(("MLflow artifact logged", True, "model_uri available"))
    elif CONFIG.enable_mlflow:
        processing_steps.append(("MLflow artifact logged", False, "model_uri missing"))

    print_processing_steps(processing_steps)

    # =========================================================================
    # ENHANCED ACCURACY DATA COLLECTION
    # =========================================================================
    trained_model = result.get("trained_model")
    y_val_pred = []
    n_positive_predictions = 0
    optimal_threshold = result.get("optimal_threshold", 0.5)

    # Use evaluator's metrics directly instead of recomputing on validation data.
    # The evaluator already handles optimal thresholds, imbalance-aware metrics,
    # and precision-constrained thresholds correctly.
    val_metrics = result.get("validation_metrics", {})
    train_metrics = result.get("train_metrics", {})
    test_metrics = result.get("test_metrics", {})

    if trained_model is not None:
        # Generate validation predictions for confusion matrix display only
        X_val_preprocessed = result.get("X_validation_preprocessed")
        fitted_preprocessor = result.get("fitted_preprocessor")

        if X_val_preprocessed is not None:
            X_val = X_val_preprocessed
        elif fitted_preprocessor is not None:
            X_val = fitted_preprocessor.transform(validation_data["X"])
        else:
            X_val = validation_data["X"]

        # Match feature names that LightGBM 4.x sees at fit so predict
        # doesn't emit "X does not have valid feature names" UserWarning.
        # The model_trainer preprocessor returns numpy from transform();
        # wrap with engineered names if the trainer also exposes them.
        if (
            isinstance(X_val, np.ndarray)
            and X_val.ndim == 2
            and fitted_preprocessor is not None
            and hasattr(fitted_preprocessor, "get_feature_names_out")
        ):
            try:
                _names = fitted_preprocessor.get_feature_names_out()
                if _names is not None and len(_names) == X_val.shape[1]:
                    X_val = pd.DataFrame(X_val, columns=list(_names))
            except Exception:
                pass

        y_val = validation_data["y"]
        y_train = train_data["y"]
        y_test = test_data["y"]

        # Make predictions at evaluator's optimal threshold (no adaptive override)
        y_val_proba = None
        if hasattr(trained_model, "predict_proba"):
            y_val_proba = trained_model.predict_proba(X_val)[:, 1]

        if y_val_proba is not None:
            y_val_pred = (y_val_proba >= optimal_threshold).astype(int)
        else:
            y_val_pred = trained_model.predict(X_val)

        n_positive_predictions = sum(y_val_pred)

        # Store accuracy analysis data (using evaluator's metrics, val predictions for display)
        result["accuracy_analysis"] = {
            "y_true": y_val.tolist() if hasattr(y_val, "tolist") else list(y_val),
            "y_pred": y_val_pred.tolist() if hasattr(y_val_pred, "tolist") else list(y_val_pred),
            "y_proba": y_val_proba.tolist() if y_val_proba is not None else None,
            "y_train": y_train.tolist() if hasattr(y_train, "tolist") else list(y_train),
            "y_val": y_val.tolist() if hasattr(y_val, "tolist") else list(y_val),
            "y_test": y_test.tolist() if hasattr(y_test, "tolist") else list(y_test),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "feature_columns": feature_columns,
        }

    # =========================================================================
    # OVERFITTING SEVERITY
    # =========================================================================
    overfitting_severity = "none"
    max_train_test_delta = 0.0
    # ONLY use threshold-invariant, distribution-robust metrics (roc_auc, pr_auc).
    # Threshold-dependent metrics (precision, recall, F1, accuracy) are confounded by
    # two effects: (1) different thresholds between train/test, and (2) SMOTE resampling
    # changes the training distribution from e.g. 97.7/2.3 to ~50/50, so precision and
    # accuracy differ hugely even without any overfitting. AUC and PR-AUC are invariant
    # to both threshold choice and class prevalence, making them the only reliable
    # overfitting signal when resampling is applied.
    if train_metrics and test_metrics:
        for metric_key in ["roc_auc", "pr_auc"]:
            t_val = train_metrics.get(metric_key)
            c_val = test_metrics.get(metric_key)
            if t_val is not None and c_val is not None:
                max_train_test_delta = max(max_train_test_delta, t_val - c_val)
        if max_train_test_delta > 0.15:
            overfitting_severity = "severe"
        elif max_train_test_delta > 0.10:
            overfitting_severity = "moderate"
        elif max_train_test_delta > 0.05:
            overfitting_severity = "mild"
        print(
            f"\n  Overfitting severity: {overfitting_severity} (max AUC train-test delta: {max_train_test_delta:.3f})"
        )
    result["overfitting_severity"] = overfitting_severity
    result["max_train_test_delta"] = max_train_test_delta

    # =========================================================================
    # VALIDATION CHECKS
    # =========================================================================
    auc = result.get("auc_roc", 0) or test_metrics.get("roc_auc", 0)
    minority_recall = result.get("minority_recall") or test_metrics.get("recall", 0)
    minority_precision = result.get("minority_precision") or test_metrics.get("precision", 0)

    # Tier C: surface the bootstrap AUC confidence interval (the measured
    # uncertainty) and run the dynamic, CI-aware AUC gate.
    auc_ci = _auc_ci_from_result(result)
    auc_passed, auc_detail = _auc_gate_verdict(
        auc,
        auc_ci,
        CONFIG.min_auc_threshold,
        require_significance=CONFIG.auc_gate_require_significance,
    )
    if auc:
        ci_msg = f" [95% CI {auc_ci[0]:.3f}-{auc_ci[1]:.3f}]" if auc_ci else " (CI unavailable)"
        print(f"\n  AUC-ROC: {auc:.3f}{ci_msg}  →  {auc_detail}")

    checks = [
        (
            "Model trained successfully",
            trained_model is not None,
            "trained_model present",
            "present" if trained_model is not None else "missing",
        ),
        (
            "AUC-ROC significance gate"
            if CONFIG.auc_gate_require_significance
            else "AUC-ROC threshold",
            auc_passed,
            "CI lower > 0.5"
            if CONFIG.auc_gate_require_significance
            else f">= {CONFIG.min_auc_threshold}",
            f"{auc:.3f}{(' [' + format(auc_ci[0], '.3f') + '-' + format(auc_ci[1], '.3f') + ']') if auc_ci else ''}"
            if auc
            else "N/A",
        ),
        (
            "Minority recall threshold",
            minority_recall >= CONFIG.min_minority_recall,
            f">= {CONFIG.min_minority_recall:.0%}",
            f"{minority_recall:.2%}",
        ),
        (
            "Minority precision threshold",
            minority_precision >= CONFIG.min_minority_precision,
            f">= {CONFIG.min_minority_precision:.0%}",
            f"{minority_precision:.2%}",
        ),
        (
            "Positive predictions made",
            n_positive_predictions > 0,
            "> 0",
            str(n_positive_predictions),
        ),
    ]

    print_validation_checks(checks)

    # =========================================================================
    # METRICS TABLE
    # =========================================================================
    metrics_list = [
        (
            "auc_roc",
            f"{auc:.3f} [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}]" if (auc and auc_ci) else auc,
            "CI lower > 0.5"
            if CONFIG.auc_gate_require_significance
            else f">= {CONFIG.min_auc_threshold}",
            auc_passed if auc else None,
        ),
        ("accuracy", val_metrics.get("accuracy"), None, None),
        (
            "precision",
            minority_precision,
            f">= {CONFIG.min_minority_precision:.0%}",
            minority_precision >= CONFIG.min_minority_precision,
        ),
        (
            "recall",
            minority_recall,
            f">= {CONFIG.min_minority_recall:.0%}",
            minority_recall >= CONFIG.min_minority_recall,
        ),
        ("f1_score", val_metrics.get("f1_score"), None, None),
        ("optimal_threshold", optimal_threshold, None, None),
        ("positive_predictions", n_positive_predictions, "> 0", n_positive_predictions > 0),
        ("hpo_trials_run", result.get("hpo_trials_run"), None, None),
    ]

    print_metrics_table(metrics_list)

    # =========================================================================
    # CONFUSION MATRIX DISPLAY (if available)
    # =========================================================================
    if result.get("accuracy_analysis") and trained_model is not None:
        y_true_list = result["accuracy_analysis"]["y_true"]
        y_pred_list = result["accuracy_analysis"]["y_pred"]
        y_proba_list = result["accuracy_analysis"].get("y_proba")

        cm_data = print_confusion_matrix(
            np.array(y_true_list),
            np.array(y_pred_list),
            np.array(y_proba_list) if y_proba_list else None,
            "Validation Confusion Matrix",
        )

        # Threshold analysis
        if y_proba_list:
            print_threshold_analysis(
                np.array(y_true_list), np.array(y_proba_list), optimal_threshold
            )

    # =========================================================================
    # INTERPRETATION
    # =========================================================================
    observations = []
    recommendations = []

    # Model performance interpretation
    if trained_model is not None:
        perf_obs, perf_rec = interpret_model_performance(
            {"roc_auc": auc},
            result.get("accuracy_analysis", {}),
            CONFIG.min_minority_recall,
            CONFIG.min_minority_precision,
        )
        observations.extend(perf_obs)
        recommendations.extend(perf_rec)

        # Class imbalance interpretation
        if imbalance_detected:
            imb_obs, imb_rec = interpret_class_imbalance(
                {
                    "imbalance_detected": True,
                    "minority_ratio": result.get("minority_ratio", 0),
                    "imbalance_severity": result.get("imbalance_severity", "unknown"),
                    "recommended_strategy": result.get("recommended_strategy", "none"),
                }
            )
            observations.extend(imb_obs)
            recommendations.extend(imb_rec)

        # Confusion matrix interpretation
        if result.get("accuracy_analysis"):
            y_pred_list = result["accuracy_analysis"]["y_pred"]
            y_true_list = result["accuracy_analysis"]["y_true"]
            cm_obs = interpret_confusion_matrix(
                {
                    "tp": sum(1 for t, p in zip(y_true_list, y_pred_list) if t == 1 and p == 1),
                    "tn": sum(1 for t, p in zip(y_true_list, y_pred_list) if t == 0 and p == 0),
                    "fp": sum(1 for t, p in zip(y_true_list, y_pred_list) if t == 0 and p == 1),
                    "fn": sum(1 for t, p in zip(y_true_list, y_pred_list) if t == 1 and p == 0),
                }
            )
            observations.extend(cm_obs)
    else:
        observations.append("⚠️  No trained model returned - training may have failed")
        recommendations.append("Check agent logs for training errors")

    print_interpretation(
        "Model Training Analysis", observations, recommendations if recommendations else None
    )

    # =========================================================================
    # DETERMINE MODEL USEFULNESS
    # =========================================================================
    if trained_model is not None:
        if n_positive_predictions == 0:
            result["model_usefulness"] = "useless"
            result["usefulness_reason"] = "predicts_all_negative"
        elif minority_recall < CONFIG.min_minority_recall:
            result["model_usefulness"] = "poor"
            result["usefulness_reason"] = f"low_recall_{minority_recall:.2%}"
        elif minority_precision < CONFIG.min_minority_precision:
            result["model_usefulness"] = "poor"
            result["usefulness_reason"] = f"low_precision_{minority_precision:.2%}"
        else:
            result["model_usefulness"] = "acceptable"
            # Overfitting is a warning, not a deployment blocker.
            # The test set IS the generalization check — if test metrics
            # pass thresholds, the model is useful despite training overfitting.
            if overfitting_severity == "severe":
                result["usefulness_warning"] = (
                    f"severe_overfitting_delta_{max_train_test_delta:.3f}"
                )

    # =========================================================================
    # FINAL RESULT
    # =========================================================================
    duration = time_mod.time() - step_start
    model_usefulness = result.get("model_usefulness", "unknown")

    if model_usefulness == "useless":
        print_step_result("failed", f"Model USELESS - predicts all negatives ({duration:.1f}s)")
    elif model_usefulness == "poor":
        print_step_result(
            "warning",
            f"Model has poor metrics ({result.get('usefulness_reason', '')}) ({duration:.1f}s)",
        )
    elif model_usefulness == "acceptable":
        print_step_result(
            "success", f"Model trained successfully - usefulness validated ({duration:.1f}s)"
        )
    else:
        print_step_result(
            "warning", f"Model training completed with unknown status ({duration:.1f}s)"
        )

    # Persist the entity → split mapping so the tier0 cache can refuse to
    # re-split on reload (Block 4, Finding #12). Empty when caller provided
    # no entity_ids.
    result["split_assignments"] = split_assignments
    result["split_strategy"] = split_strategy

    return result


async def step_6_feature_analyzer(
    experiment_id: str,
    trained_model: Any,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    model_uri: Optional[str] = None,
) -> dict[str, Any]:
    """Step 6: Analyze feature importance."""
    import time as time_mod

    step_start = time_mod.time()

    print_header(6, "FEATURE ANALYZER")

    from src.agents.ml_foundation.feature_analyzer import FeatureAnalyzerAgent

    feature_columns = list(X_sample.columns)

    print_input_section(
        {
            "sample_size": len(X_sample),
            "features": feature_columns,
            "max_samples": min(100, len(X_sample)),
            "model_uri": model_uri[:50] + "..." if model_uri and len(model_uri) > 50 else model_uri,
        }
    )

    # Processing
    processing_steps = []
    processing_steps.append(("Creating FeatureAnalyzerAgent", True, None))

    agent = FeatureAnalyzerAgent()
    processing_steps.append(("Agent initialized", True, None))

    input_data = {
        "experiment_id": experiment_id,
        "trained_model": trained_model,
        "model_uri": model_uri,
        "X_sample": X_sample,
        "y_sample": y_sample,
        "max_samples": min(100, len(X_sample)),
        "feature_columns": feature_columns,
    }

    try:
        result = await agent.run(input_data)
        processing_steps.append(("SHAP analysis executed", True, None))
        analysis_success = True
    except Exception as e:
        result = {"feature_importance": None, "error": str(e)}
        processing_steps.append(("SHAP analysis executed", False, str(e)[:50]))
        analysis_success = False

    print_processing_steps(processing_steps)

    # Validation checks
    has_importance = result.get("feature_importance") is not None
    samples_analyzed = result.get("samples_analyzed", 0)

    checks = [
        (
            "Feature importance computed",
            has_importance,
            "feature_importance present",
            "present" if has_importance else "missing",
        ),
        (
            "Samples analyzed",
            samples_analyzed > 0 if samples_analyzed else False,
            "> 0",
            str(samples_analyzed) if samples_analyzed else "0",
        ),
    ]

    print_validation_checks(checks)

    # Metrics - Feature importance table
    if has_importance:
        print("\n  📊 Feature Importance (SHAP):")
        print(f"    {'Feature':<25} {'Importance':<15} {'Rank':<10}")
        print(f"    {'-' * 50}")

        for i, fi in enumerate(result["feature_importance"][:10], 1):
            if isinstance(fi, dict):
                name = str(fi.get("feature", f"feature_{i}"))[:25]
                imp = fi.get("importance", 0)
                print(f"    {name:<25} {imp:<15.4f} #{i:<10}")
            else:
                print(f"    {str(fi):<25} {'N/A':<15} #{i:<10}")

    # Interpretation
    observations = []
    recommendations = []

    if has_importance:
        top_features = result["feature_importance"][:3]
        if top_features:
            top_names = [
                str(fi.get("feature", "unknown")) if isinstance(fi, dict) else str(fi)
                for fi in top_features
            ]
            observations.append(f"Top predictive features: {', '.join(top_names)}")

            # Feature-specific insights
            for fi in top_features:
                if isinstance(fi, dict):
                    name = str(fi.get("feature", ""))
                    imp = fi.get("importance", 0)
                    if "days_on_therapy" in name.lower():
                        observations.append(
                            f"  • Duration on therapy ({imp:.3f}) is a strong predictor"
                        )
                    elif "hcp_visits" in name.lower():
                        observations.append(
                            f"  • HCP engagement ({imp:.3f}) influences discontinuation"
                        )
                    elif "prior_treatments" in name.lower():
                        observations.append(f"  • Treatment history ({imp:.3f}) affects outcomes")

        observations.append(f"Analysis based on {samples_analyzed} samples using SHAP explainer")
    else:
        observations.append("⚠️  Feature importance analysis failed or skipped")
        if result.get("error"):
            observations.append(f"    Error: {result['error'][:100]}")
        recommendations.append("Verify model is compatible with SHAP explainer")
        recommendations.append("Check if model_uri is valid and accessible")

    print_interpretation(
        "Feature Analysis", observations, recommendations if recommendations else None
    )

    # Final result
    duration = time_mod.time() - step_start
    if analysis_success and has_importance:
        print_step_result("success", f"Feature importance computed ({duration:.1f}s)")
    elif analysis_success:
        print_step_result("warning", f"Analysis completed but no importance data ({duration:.1f}s)")
    else:
        print_step_result("warning", f"Feature analysis failed (optional step) ({duration:.1f}s)")

    return result


async def step_7_model_deployer(
    experiment_id: str,
    model_uri: str,
    validation_metrics: dict,
    success_criteria_met: bool,
    trained_model: Any = None,
    include_bentoml: bool = True,
    fitted_preprocessor: Any = None,
    feature_columns: list[str] | None = None,
    scope_spec: Any = None,
) -> dict[str, Any]:
    """Step 7: Deploy model."""
    import time as time_mod

    step_start = time_mod.time()

    print_header(7, "MODEL DEPLOYER")

    from src.agents.ml_foundation.model_deployer import ModelDeployerAgent

    deployment_name = f"kisqali_discontinuation_{experiment_id[:8]}"

    print_input_section(
        {
            "deployment_name": deployment_name,
            "model_uri": model_uri[:50] + "..." if model_uri and len(model_uri) > 50 else model_uri,
            "success_criteria_met": success_criteria_met,
            "deployment_action": "register",
            "include_bentoml": include_bentoml,
        }
    )

    # Processing
    processing_steps = []
    processing_steps.append(("Creating ModelDeployerAgent", True, None))

    agent = ModelDeployerAgent()
    processing_steps.append(("Agent initialized", True, None))

    input_data = {
        "experiment_id": experiment_id,
        "model_uri": model_uri or f"runs:/{experiment_id}/model",
        "validation_metrics": validation_metrics,
        "success_criteria_met": success_criteria_met,
        "deployment_name": deployment_name,
        "deployment_action": "register",
    }
    # v5 Gate C1 (2026-05-11): thread cohort identity to deployer so
    # validate_promotion's regulatory_deployment_manifest builder
    # resolves the cohort authorization policy (CSU in scope; Optum
    # blocked; unknown → out_of_scope).
    if scope_spec is not None:
        input_data["scope_spec"] = scope_spec
        if isinstance(scope_spec, dict) and scope_spec.get("feature_manifest_source"):
            input_data["feature_manifest_source"] = scope_spec.get("feature_manifest_source")

    try:
        result = await agent.run(input_data)
        processing_steps.append(("Model registration", True, result.get("status", "unknown")))
        agent_success = True
    except Exception as agent_error:
        error_type = getattr(agent_error, "error_type", None) or type(agent_error).__name__
        processing_steps.append(("Model registration", False, error_type))
        agent_success = False

        result = {
            "status": "error",
            "deployment_successful": False,
            "error": str(agent_error),
            "error_type": error_type,
            "deployment_manifest": {
                "deployment_id": f"deploy_{experiment_id[:12]}",
                "environment": "staging",
                "status": "error",
            },
        }

    print_processing_steps(processing_steps)

    # Validation checks
    deployment_successful = (
        result.get("deployment_successful", False) or result.get("status") == "completed"
    )
    manifest = result.get("deployment_manifest", {})

    checks = [
        (
            "Deployment successful",
            deployment_successful,
            "deployment_successful = True",
            f"{result.get('status', 'unknown')}",
        ),
        (
            "Deployment manifest",
            bool(manifest),
            "manifest present",
            "present" if manifest else "missing",
        ),
        (
            "Success criteria met",
            success_criteria_met,
            "success_criteria_met = True",
            str(success_criteria_met),
        ),
    ]

    print_validation_checks(checks)

    # Metrics
    metrics_list = [
        ("deployment_id", manifest.get("deployment_id"), None, None),
        ("environment", manifest.get("environment"), None, None),
        ("status", manifest.get("status"), None, None),
        ("model_version", result.get("model_version"), None, None),
    ]

    print_metrics_table(metrics_list)

    # BentoML Model Serving via Docker container
    if include_bentoml and trained_model is not None:
        print("\n  " + "-" * 60)
        print("  BentoML Model Serving (Docker):")
        print("  " + "-" * 60)

        try:
            # Detect framework from model class
            model_class_name = type(trained_model).__name__
            if "XGB" in model_class_name:
                framework = "xgboost"
            elif "LGBM" in model_class_name or "LightGBM" in model_class_name:
                framework = "lightgbm"
            else:
                framework = "sklearn"

            model_name = f"tier0_{experiment_id.split('_')[-1]}"
            print(f"    Registering model in Docker BentoML: {model_name} (framework: {framework})")

            # Register model in Docker BentoML container's store
            registration = await register_model_in_docker_bentoml(
                trained_model=trained_model,
                model_name=model_name,
                framework=framework,
                metadata={
                    "experiment_id": experiment_id,
                    "tier0_test": True,
                    "algorithm": model_class_name,
                },
                fitted_preprocessor=fitted_preprocessor,
                feature_columns=feature_columns,
            )

            if registration.get("success"):
                model_tag = registration["model_tag"]
                print(f"    ✓ Model registered in Docker store: {model_tag}")

                # Restart container to pick up new model via auto-discovery
                deploy_result = await deploy_to_persistent_service(model_tag)
                if deploy_result.get("available"):
                    endpoint = deploy_result["endpoint"]
                    print(f"    ✓ Docker BentoML service ready at {endpoint}")
                    result["bentoml_persistent"] = True
                else:
                    reason = deploy_result.get("reason", "unknown")
                    print(f"    ✗ Docker BentoML restart failed: {reason}")
                    result["bentoml_serving"] = {"error": reason}
                    endpoint = None

                if endpoint:
                    # Build sample from actual feature columns (zeros as safe default)
                    n_features = len(feature_columns) if feature_columns else 3
                    sample_features = [[0.0] * n_features]

                    verification = await verify_bentoml_predictions(
                        endpoint=endpoint,
                        sample_features=sample_features,
                        service_type="persistent",
                    )

                    # Display results
                    print("\n    BentoML Serving Verification:")
                    health_icon = "✓" if verification.get("health_check") else "✗"
                    print(
                        f"      health_check: {health_icon} {'healthy' if verification.get('health_check') else 'unhealthy'}"
                    )

                    pred_icon = "✓" if verification.get("prediction_test") else "✗"
                    print(
                        f"      prediction_test: {pred_icon} {'passed' if verification.get('prediction_test') else 'failed'}"
                    )
                    if not verification.get("prediction_test"):
                        if verification.get("prediction_error"):
                            print(f"      error: {verification['prediction_error']}")
                        if verification.get("prediction_error_body"):
                            print(f"      response_body: {verification['prediction_error_body']}")

                    if verification.get("predictions"):
                        print(f"      predictions: {verification.get('predictions')}")
                    if verification.get("probabilities"):
                        print(f"      probabilities: {verification.get('probabilities')}")
                    if verification.get("latency_ms"):
                        print(f"      latency_ms: {verification.get('latency_ms'):.1f}")

                    result["bentoml_serving"] = {
                        "model_tag": model_tag,
                        "endpoint": endpoint,
                        "mode": "docker",
                        "health_check": verification.get("health_check"),
                        "prediction_test": verification.get("prediction_test"),
                        "predictions": verification.get("predictions"),
                        "probabilities": verification.get("probabilities"),
                        "latency_ms": verification.get("latency_ms"),
                    }

                    if verification.get("health_check") and verification.get("prediction_test"):
                        print_success("Real model deployed and serving verified via Docker BentoML")
                    else:
                        print_warning("Docker BentoML running but verification incomplete")
            else:
                error_msg = registration.get("error", "Registration failed")
                print(f"    ✗ Docker BentoML registration failed: {error_msg}")
                result["bentoml_serving"] = {"error": error_msg}

        except Exception as e:
            print(f"    ✗ BentoML error: {e}")
            result["bentoml_serving"] = {"error": str(e)}
            import traceback

            traceback.print_exc()

    elif include_bentoml and trained_model is None:
        result["bentoml_serving"] = {"error": "No trained model available"}

    # Interpretation
    observations = []
    recommendations = []

    if deployment_successful:
        observations.append(
            f"Model registered with deployment ID: {manifest.get('deployment_id', 'N/A')}"
        )
        observations.append(f"Environment: {manifest.get('environment', 'staging')}")
    else:
        observations.append("⚠️  Model deployment encountered issues")
        if result.get("error"):
            observations.append(f"    Error: {result['error'][:100]}")
        recommendations.append("Review deployment agent logs for details")

    if not success_criteria_met:
        observations.append("⚠️  Model did not meet success criteria")
        recommendations.append("Review model metrics before production deployment")

    # BentoML observations
    bentoml_serving = result.get("bentoml_serving", {})
    if bentoml_serving.get("health_check") and bentoml_serving.get("prediction_test"):
        observations.append("BentoML serving verified: health check passed, predictions working")
        observations.append(f"    Model tag: {bentoml_serving.get('model_tag', 'N/A')}")
        observations.append(f"    Endpoint: {bentoml_serving.get('endpoint', 'N/A')}")
        if bentoml_serving.get("latency_ms"):
            observations.append(f"    Inference latency: {bentoml_serving['latency_ms']:.1f}ms")
    elif include_bentoml and bentoml_serving.get("error"):
        observations.append(f"⚠️  BentoML serving failed: {bentoml_serving['error'][:50]}")
        recommendations.append("Check BentoML installation and model compatibility")

    print_interpretation(
        "Deployment Analysis", observations, recommendations if recommendations else None
    )

    # Final result
    duration = time_mod.time() - step_start
    bentoml_ok = (
        bentoml_serving.get("health_check") and bentoml_serving.get("prediction_test")
        if include_bentoml
        else True
    )

    if deployment_successful and bentoml_ok:
        print_step_result("success", f"Model deployed successfully ({duration:.1f}s)")
    elif deployment_successful:
        print_step_result("warning", f"Deployment OK but BentoML issues ({duration:.1f}s)")
    else:
        print_step_result("warning", f"Deployment had issues ({duration:.1f}s)")

    return result


async def step_8_observability_connector(
    experiment_id: str, stages_completed: int
) -> dict[str, Any]:
    """Step 8: Log to observability."""
    import time as time_mod

    step_start = time_mod.time()

    print_header(8, "OBSERVABILITY CONNECTOR")

    from src.agents.ml_foundation.observability_connector import ObservabilityConnectorAgent

    events = [
        {
            "event_type": "pipeline_completed",
            "agent_name": "tier0_e2e_test",
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": {
                "experiment_id": experiment_id,
                "stages_completed": stages_completed,
                "brand": CONFIG.brand,
            },
        }
    ]

    print_input_section(
        {
            "events_to_log": 1,
            "event_type": "pipeline_completed",
            "experiment_id": experiment_id,
            "stages_completed": stages_completed,
            "time_window": "1h",
        }
    )

    # Processing
    processing_steps = []
    processing_steps.append(("Creating ObservabilityConnectorAgent", True, None))

    agent = ObservabilityConnectorAgent()
    processing_steps.append(("Agent initialized", True, None))

    input_data = {
        "events_to_log": events,
        "time_window": "1h",
    }

    result = await agent.run(input_data)

    emission_successful = result.get("emission_successful", False)
    processing_steps.append(
        ("Event emission", emission_successful, f"{result.get('events_logged', 0)} events")
    )

    # Feast online feature retrieval check (gracefully degrading). Feast is absent
    # from the app image by design (tenacity conflict, #307), so this falls back to
    # the custom store — see step_2b_feast_registration docstring + #532 (FU2 / #528).
    feast_online_ok = False
    feast_online_detail = "skipped"
    try:
        from src.agents.prediction_synthesizer.nodes.feast_feature_store import FeastFeatureStore

        feast_store = FeastFeatureStore()
        sample_entity_id = f"hcp_{experiment_id[:8]}"
        online_features = await feast_store.get_online_features(entity_id=sample_entity_id)
        feast_online_ok = isinstance(online_features, dict)
        feast_online_detail = f"{len(online_features)} features" if online_features else "empty"
        processing_steps.append(("Feast online retrieval", feast_online_ok, feast_online_detail))
    except Exception as e:
        feast_online_detail = str(e)[:80]
        processing_steps.append(
            ("Feast online retrieval", False, f"skipped: {feast_online_detail}")
        )

    print_processing_steps(processing_steps)

    # Validation checks
    events_logged = result.get("events_logged", 0)
    quality_score = result.get("quality_score", 0)

    checks = [
        (
            "Emission successful",
            emission_successful,
            "emission_successful = True",
            str(emission_successful),
        ),
        (
            "Events logged",
            events_logged > 0 if events_logged else False,
            "> 0",
            str(events_logged) if events_logged else "0",
        ),
        (
            "Feast online retrieval",
            feast_online_ok,
            "accessible",
            feast_online_detail,
        ),
    ]

    print_validation_checks(checks)

    # Metrics
    metrics_list = [
        ("emission_successful", emission_successful, None, None),
        ("events_logged", events_logged, "> 0", events_logged > 0 if events_logged else None),
        ("quality_score", quality_score, None, None),
    ]

    print_metrics_table(metrics_list)

    # Interpretation
    observations = []
    recommendations = []

    if emission_successful:
        observations.append(f"Pipeline completion event logged to observability system")
        observations.append(f"Experiment {experiment_id} with {stages_completed} stages recorded")
        if quality_score:
            observations.append(f"Event quality score: {quality_score}")
    else:
        observations.append("⚠️  Observability logging encountered issues")
        recommendations.append("Check observability service connectivity")
        recommendations.append("Verify event schema compliance")

    print_interpretation(
        "Observability Analysis", observations, recommendations if recommendations else None
    )

    # Attach Feast results for pipeline StepResult access
    result["feast_online_ok"] = feast_online_ok
    result["feast_online_detail"] = feast_online_detail

    # Final result
    duration = time_mod.time() - step_start
    if emission_successful:
        print_step_result("success", f"Observability logging complete ({duration:.1f}s)")
    else:
        print_step_result("warning", f"Observability logging had issues ({duration:.1f}s)")

    return result


# =============================================================================
# MAIN RUNNER
# =============================================================================


_VALID_REGIMES: Tuple[str, ...] = (
    "default",
    "adverse",
    "clean",
    "scenario_a",
    "scenario_a_balanced",
    "scenario_b",
    "scenario_c",
)

# Codex-rescue pass-2 M2 (2026-05-09): explicit legacy-regimes set used by
# the artifact n_total guard at run_pipeline. Lifts the artifact gate from
# "is in scenario set" (positive enumeration) to "is NOT in legacy set"
# (negative enumeration of the small, stable exception). Future regime
# additions inherit the synthetic_v2 behavior automatically; only an
# author who explicitly adds a new legacy ``ml_patients()``-style regime
# needs to update this set.
_LEGACY_REGIMES: frozenset[str] = frozenset({"default", "adverse", "clean"})


def _is_synthetic_fixture_regime(regime: str) -> bool:
    """Whether *regime* is a clinically-grounded SYNTHETIC fixture with no real
    leakage by construction.

    Two families qualify: the synthetic_v2 scenario regimes
    (``_SCENARIO_REGIME_TO_NAME``) and the legacy ``ml_patients()`` regimes
    (``_LEGACY_REGIMES`` = default/adverse/clean). Both carry features
    deliberately correlated with the outcome (designed signal, NOT leakage), so
    the Layer-3 FDR confident-set firing driver (#538) false-positively flags and
    auto-drops legitimate features (e.g. ``days_on_therapy``/``prior_treatments``
    on the clean/adverse fixtures) — which degrades val_AUC below band and, post
    #556 fail-closed Feast, halts at the QC gate. The tier0 runner therefore
    disables that FDR firing for these regimes (``adaptive_fdr_enabled=False``),
    falling back to the static σ-band that still catches genuine leaks
    (e.g. ``journey_status``) WITHOUT over-dropping (#594).

    ``rwd_realistic`` and real ``--data-source`` runs are NOT fixtures — the FDR
    driver stays ON there (validated on the Optum cohort).
    """
    return regime in _SCENARIO_REGIME_TO_NAME or regime in _LEGACY_REGIMES


def _resolve_adaptive_fdr_enabled(regime: str, data_dir: str | None) -> bool:
    """Resolve the per-run FDR firing switch for the data-preparer.

    FDR firing stays ON everywhere EXCEPT synthetic ``scenario_*`` fixture
    GENERATION. #604: the LEGACY ``ml_patients()`` fixtures (default/adverse/clean)
    now keep FDR ON — their legit pre-index predictors are protected by the
    synthetic-manifest declared-safe carve-out with FULL immunity
    (``_resolve_declared_safe_full_immunity`` + ``_resolve_synthetic_manifest_source``),
    not by disabling the detector. The ``scenario_*`` (synthetic_v2) family is
    NOT manifest-wired and is not in the must-pass CI lane, so it retains the #594
    wholesale FDR-disable mitigation. The ``data_dir`` guard preserves the #594
    production-safety contract: a real ``--data-dir`` run (which IGNORES the
    possibly-fixture ``--regime`` name) keeps FDR ON.
    """
    is_unwired_scenario_fixture = not data_dir and regime in _SCENARIO_REGIME_TO_NAME
    return not is_unwired_scenario_fixture


def _resolve_declared_safe_full_immunity(
    regime: str, data_dir: str | None, manifest_source: str | None
) -> bool:
    """#604: resolve the per-run declared-safe FULL-immunity switch.

    True ONLY when ALL hold: (a) no real cohort supplied (``not data_dir``), (b) the
    regime is a legacy synthetic fixture (``_LEGACY_REGIMES``), AND (c) the EFFECTIVE
    feature manifest the node will consult is the ``"synthetic"`` one. Condition (c)
    (codex round-2) couples immunity to the manifest actually used rather than the
    regime name alone: an operator ``--feature-manifest-source csu/optum`` override
    on a legacy regime makes the node consult a fallible real-cohort manifest, so
    immunity is withheld; and an explicit ``--feature-manifest-source synthetic`` on
    a REAL ``--data-dir`` run is denied by (a). The synthetic manifest is leak-free
    BY CONSTRUCTION, so a Layer-1 declared-safe feature there must not be auto-dropped
    for being strongly outcome-correlated. Real cohorts and ``scenario_*`` keep
    immunity OFF (real runs preserve the σ!=high "overwhelming evidence" backstop;
    scenario_* has FDR disabled anyway). Genuine undeclared leaks always drop
    (immunity is additionally gated on declared-safe in the node).

    ``manifest_source`` is the EFFECTIVE source already resolved onto ``scope_spec``
    by ``_apply_synthetic_manifest_source`` (i.e. exactly what the node reads).
    """
    return not data_dir and regime in _LEGACY_REGIMES and manifest_source == "synthetic"


def _resolve_synthetic_manifest_source(
    regime: str, data_dir: str | None, override: str | None
) -> str | None:
    """#604: resolve the feature-manifest source for legacy synthetic fixtures.

    A legacy synthetic fixture run (``_LEGACY_REGIMES``, no real ``data_dir``) with
    no explicit override resolves to the ``"synthetic"`` manifest, so
    ``lookup_feature_contract`` clears the legit pre-index columns
    (days_on_therapy/hcp_visits/prior_treatments) and ``layer_1_declared_safe``
    becomes True for them — the precondition for the immunity carve-out. An explicit
    ``override`` (e.g. an operator-supplied ``--feature-manifest-source``) ALWAYS
    wins. The ``scenario_*`` family and real runs return None here (real runs resolve
    csu/optum via the separate RWD path in ``_resolve_feature_manifest_source``).
    """
    if override is not None:
        return override
    if not data_dir and regime in _LEGACY_REGIMES:
        return "synthetic"
    return None


def _apply_synthetic_manifest_source(
    scope_spec: Dict[str, Any], regime: str, data_dir: str | None, override: str | None
) -> None:
    """#604: thread the resolved feature-manifest source onto ``scope_spec``.

    Shared by the Step-1 AND Step-2 blocks of ``run_pipeline`` so a partial
    ``--step 2`` run wires the synthetic manifest identically to a full run — the
    Step-1 injection is skipped when ``steps_to_run == [2]``, which would otherwise
    leave a legacy fixture with FDR on + immunity granted but NO manifest, so
    ``layer_1_declared_safe`` stays False and the legit columns are over-dropped.
    Idempotent: re-applying with the same inputs yields the same value.
    """
    resolved = _resolve_synthetic_manifest_source(regime, data_dir, override)
    if resolved is not None:
        scope_spec["feature_manifest_source"] = resolved


def _regime_kwargs(regime: str, *, seed: int = 42) -> Dict[str, Any]:
    """Translate a regime name into kwargs for ``ml_patients()``.

    Section A of pre_phase2_unblockers — single source of truth for the
    regime → generator-knob mapping. Replaces the inline
    ``positive_rate = 0.02 if regime == "adverse" else 0.30`` ternary so
    a future regime addition only needs one edit.

    Regimes:
      - ``default``: positive_rate=0.30, signal_strength=1.0, noise_sd=0.10,
        signalize_extra_features=False — the historical balanced regime.
      - ``adverse``: positive_rate=0.02, signal_strength=1.0, noise_sd=0.10,
        signalize_extra_features=False — extreme imbalance, exercises
        remediation paths. Keeps signal config identical to ``default`` so
        the existing ``TestAdverseRegimeE2E`` contracts remain stable.
      - ``clean``: positive_rate=0.70, signal_strength=1.4, noise_sd=0.03,
        signalize_extra_features=True — Phase 2 baseline regime
        (path-D values, post-Codex review 2026-04-30). Two empirical
        adjustments diverge from the plan text:
          * positive_rate=0.70 (plan said 0.30 nominal, then 0.50 in the
            first revision). At 0.50 the realised positive share landed
            ~25%, making `precision >= 0.70` infeasible at any AUC; 0.70
            pushes realised share toward ~35% so the precision gate has
            headroom.
          * noise_sd=0.03 (plan said 0.05). Compensates for the
            scale * noise_sd interaction at ``sample_data.py:660`` —
            ``scale = positive_rate / 0.30 = 2.33`` here, so effective
            noise SD = 0.03 * 2.33 ≈ 0.07, close to the plan's intended
            ±0.05 envelope after correcting for the rescaling. At the
            old combo (positive_rate=0.50, noise_sd=0.05) the effective
            noise was 0.083 — 67% above plan, suppressing val AUC by
            ~6pp.
        See ``.claude/plans/pre_phase2_unblockers/03-section-a-synthetic.md``
        §4 (post-Codex revision) and ``08-risks.md`` #9.

    Raises:
        ValueError: when ``regime`` is unknown. Centralizing the lookup
            here means callers don't need to repeat the membership check.
    """
    if regime == "default":
        kwargs: Dict[str, Any] = {
            "seed": seed,
            "positive_rate": 0.30,
            "signal_strength": 1.0,
            "noise_sd": 0.10,
            "signalize_extra_features": False,
        }
    elif regime == "adverse":
        kwargs = {
            "seed": seed,
            "positive_rate": 0.02,
            "signal_strength": 1.0,
            "noise_sd": 0.10,
            "signalize_extra_features": False,
        }
    elif regime == "clean":
        # #633 (2026-06-02): the clean-regime v3 unblock pairs TWO levers that
        # attack the two DISJOINT gate families:
        #
        #   (A) CALIBRATION-SHAPE gates (slope_dev ≤ 0.15, |intercept| ≤ 0.30,
        #       ECE ≤ 0.05) — fixed by the deploy-calibrated machinery in
        #       evaluator.py (#639): the v3 calibration gates are judged on the
        #       post-hoc CALIBRATED estimator that the pipeline actually ships,
        #       not the under-confident raw tree. This is NOT fixture-tunable.
        #
        #   (B) RANKING / OVERFIT gates (maximum_train_val_delta ≤ 0.03, and
        #       minimum_mcc ≥ 0.45) — calibration-INVARIANT (post-hoc remap is
        #       monotonic), so only data quantity + champion regularization
        #       close them. The fixture point below keeps a calibration-ELIGIBLE
        #       TREE as champion (the #639-measured ns=0.04 point — ns=0.05 flips
        #       to a calibration-native champion that ships raw probs and busts
        #       the calibration gates; ns=0.06 flips to logistic that busts the
        #       0.92 AUC band) while the 4000-row clean cohort (see
        #       ``_REGIME_N_SAMPLES``) shrinks the tree's train↔val memorization
        #       gap below 0.03 WITH MARGIN — the lever #639 never tried.
        #
        # positive_rate=1.2 lifts the realised positive share above the 0.40
        # "none" boundary (config/imbalance_strategy.yaml) so NO oversampling
        # fires (oversampling rebalances training away from test prevalence →
        # overfit + decalibration). positive_rate is a risk-score base-rate
        # multiplier clipped to [0.05, 0.95]; > 1.0 is valid. signal_strength
        # 1.35 / noise_sd 0.04 are the #639-measured calibration-eligible point.
        # Final gate values are MEASURED on faithful AVX512 CI (slow-tests
        # Job B); local AVX2 suppresses val roc_auc ~0.10-0.18 and can flip the
        # calibration sign, so it CANNOT tune these gates. See issue #633.
        kwargs = {
            "seed": seed,
            "positive_rate": 1.2,
            "signal_strength": 1.35,
            "noise_sd": 0.04,
            "signalize_extra_features": True,
        }
    elif regime in _SCENARIO_REGIME_TO_NAME:
        # Sentinel: dispatched in generate_sample_data via synthetic_v2.api;
        # bypasses ml_patients() so each scenario's calibrated AUC band holds.
        # scenario_a:           [0.78, 0.83] — scenarios/scenario_a.py:7
        # scenario_a_balanced:  empirical (Phase 4 pin) — prevalence shifted to 0.50
        # scenario_b:           [0.72, 0.78] — scenarios/scenario_b.py:7
        # scenario_c:           [0.82, 0.88] — scenarios/scenario_c.py:7
        kwargs = {
            "_generator": regime,
            "seed": seed,
        }
    else:
        raise ValueError(f"regime must be one of {_VALID_REGIMES}, got {regime!r}")

    # Internal invariants — guard against a misroute that would silently
    # suppress signal (e.g. clean signal_strength accidentally combined
    # with a near-zero positive_rate, where the scale=positive_rate/0.30
    # rescaling at sample_data.py:602 squashes the deterministic component
    # by ~15x). These are belt-and-braces; the regime → kwargs map above
    # is correct as written.
    if regime == "clean":
        assert kwargs["positive_rate"] >= 0.20, (
            "clean regime requires positive_rate ≥ 0.20; rescaling at "
            "sample_data.py:602 suppresses signal at low rates"
        )
    if regime == "adverse":
        assert kwargs["signal_strength"] == 1.0 and not kwargs["signalize_extra_features"], (
            "adverse regime must preserve historical generator behavior"
        )
    return kwargs


# #633: per-regime synthetic cohort size. The legacy default is 1500 (chosen
# to satisfy scope_spec.minimum_samples=500 and give CausalForestDML ~500 per
# segment). The ``clean`` regime needs MORE data to honestly pass the v3
# ``maximum_train_val_delta`` overfit gate (a calibration-INVARIANT ranking
# gate that more data — not post-hoc calibration — must close), so it gets
# 4000. Every other regime keeps 1500 so its calibrated AUC band / baseline
# snapshot is unchanged. fpr = feature_count / N stays well below 1/50 at
# N=4000 (so the 0.03 train_val_delta tier is preserved — the bar does NOT
# move) and ECE stays on the N≥1000 (0.05) tier.
_REGIME_N_SAMPLES: Dict[str, int] = {"clean": 4000}
_DEFAULT_N_SAMPLES: int = 1500


def _regime_n_samples(regime: str) -> int:
    """Return the synthetic cohort size for *regime* (legacy ml_patients path).

    Defaults to ``_DEFAULT_N_SAMPLES`` (1500) for every regime except those
    listed in ``_REGIME_N_SAMPLES`` (currently only ``clean`` → 4000). See the
    #633 rationale on ``_REGIME_N_SAMPLES``.
    """
    return _REGIME_N_SAMPLES.get(regime, _DEFAULT_N_SAMPLES)


def _compute_adaptive_state_inputs(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_col: str,
    regime: str,
    deployment_intent: str = "clinical",
) -> Dict[str, Any]:
    """Compute the pre-eval inputs for the adaptive-criteria scheme.

    The fifth input (``baseline_auc``) is computed inside the evaluator
    via Section B parent branch (see
    ``model_trainer/nodes/evaluator.py::_compute_baseline_test_metrics``)
    and overlaid by ``_apply_adaptive_criteria_overlay``. This helper
    handles the inputs derivable from the dataframe, the regime label, and
    the deployment-intent (clinical | commercial) that selects the use-case
    bar (clinical AUC 0.75 vs commercial AUC 0.65).

    See ``.claude/plans/adaptive_success_criteria/05-data-shape-introspection.md``.
    """
    valid_regimes = set(_VALID_REGIMES)
    intent = deployment_intent if deployment_intent in ("clinical", "commercial") else "clinical"
    return {
        "n_samples": int(len(df)),
        "prevalence": float(df[target_col].mean()),
        "feature_count": len(feature_columns),
        "regime": regime if regime in valid_regimes else None,
        "deployment_intent": intent,
    }


_DEMO_COST_MATRIX_PATH = Path(__file__).resolve().parent.parent / "config" / "cost_matrix_demo.yaml"
_REQUIRED_DEMO_COST_KEYS = ("tp", "fn", "fp", "tn")


def _default_demo_cost_matrix() -> Dict[str, float]:
    """Return the unit-shape demo cost matrix used by the synthetic runner.

    Block 5B (#10): close the verification gap left by Block 5. The evaluator
    short-circuits ``business_utility`` whenever ``cost_matrix`` is ``None``
    (see ``evaluator.py:715``), and no current caller of this script populates
    one — so a default ``python scripts/run_tier0_test.py`` run produced no
    business_utility number, which made Block 5's metric impossible to
    verify end-to-end.

    Phase 5 Task 5.1 (tier0_evaluation_vs_distilled_mlops.md): the matrix
    used to be hardcoded here; it now lives at ``config/cost_matrix_demo.yaml``
    so the placeholder is visible, reviewable, and replaceable without
    touching this script. The caller surface (``_should_inject_demo_cost_matrix``)
    is unchanged — production callers still must supply their own matrix.

    Returns the four-key dict ``{tp, fn, fp, tn}`` parsed from the YAML.
    Raises FileNotFoundError or KeyError if the YAML is missing or malformed
    — failing loudly is preferable to silently injecting a wrong matrix.
    """
    import yaml  # local import: yaml is already in deps via Feast/MLflow

    with open(_DEMO_COST_MATRIX_PATH) as fh:
        loaded = yaml.safe_load(fh) or {}

    missing = [k for k in _REQUIRED_DEMO_COST_KEYS if k not in loaded]
    if missing:
        raise KeyError(f"config/cost_matrix_demo.yaml missing required keys: {missing}")

    return {k: float(loaded[k]) for k in _REQUIRED_DEMO_COST_KEYS}


def _should_inject_demo_cost_matrix(
    scope_spec: Dict[str, Any],
    inject: bool,
) -> bool:
    """Decide whether the run_pipeline branch should auto-inject the
    placeholder cost matrix.

    Block 5B (#10) decision rule:
      - If the caller passed ``--no-demo-cost-matrix`` (``inject=False``),
        always skip injection — even when ``scope_spec`` has no matrix.
        That is the explicit "reproduce the pre-Block-5B baseline" path.
      - If the caller did not opt out (``inject=True``), inject only
        when ``scope_spec.cost_matrix`` is falsy. ``scope_definer``
        always emits the key with a default of ``None`` (see
        ``scope_builder._validate_cost_matrix``), so ``"cost_matrix"
        not in scope_spec`` would miss that case — we treat both
        "missing" and "present-but-None" as un-set.

    Pulled out into a helper so the in-process unit test can exercise
    the real decision branch (5B-I-2).
    """
    if not inject:
        return False
    return not scope_spec.get("cost_matrix")


def _to_jsonable(value: Any) -> Any:
    """Recursively coerce ``value`` to JSON-native Python types.

    ``json.dumps(..., default=str)`` happens at write time, but ``str``-
    coercing a numpy scalar or datetime hides the actual type — readers
    parsing the artifact back would see ``"0.5"`` rather than ``0.5``,
    and assertions like ``perm["permutation_pvalue"] <= 0.01`` silently
    pass against a string. ``_to_jsonable`` walks the tree once and
    converts known non-native types to their JSON equivalents:

    - dict / list / tuple / set: recurse into elements
    - bool, int, float, str, None: passthrough
    - numpy scalars: ``.item()`` to get the native Python value
    - datetime / date: ``.isoformat()`` so they remain string-typed but
      machine-parseable
    - Pydantic models: ``.model_dump()`` if available
    - Anything else: fall back to ``str()`` (matches the existing
      ``default=str`` contract at write time)

    Codex review M4 (2026-05-08): introduced because the artifact
    extension's shallow ``dict()`` / ``list()`` coercion stopped at the
    top level and let opaque types reach the json encoder.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in value]
    # numpy scalar
    item_attr = getattr(value, "item", None)
    if callable(item_attr):
        try:
            inner = item_attr()
        except Exception:  # noqa: BLE001 — defensive normalization
            inner = None
        if isinstance(inner, (str, int, float, bool)) or inner is None:
            return inner
    # Pydantic model (v1 .dict() or v2 .model_dump())
    dump = getattr(value, "model_dump", None) or getattr(value, "dict", None)
    if callable(dump):
        try:
            return _to_jsonable(dump())
        except Exception:  # noqa: BLE001
            pass
    # datetime / date / time
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:  # noqa: BLE001
            pass
    return str(value)


# Layer 5 manifest-source resolution: the canonical strictness contract (M1
# ambiguous / M2 conflict / unknown override / M3 unmatched→None) lives in
# ``src.data.manifests.resolution`` (Phase B), shared with the
# ``MLFoundationPipeline`` scope stage and the live retraining trigger and keyed
# off the ``MANIFEST_SOURCES`` registry (single source of truth — adding a
# source is one edit there). ``_resolve_feature_manifest_source`` stays a thin
# operator-facing wrapper that delegates the strict logic and adds a stderr
# WARNING when a provided ``data_dir`` matched no known source, so an operator
# running an unrecognised RWD path sees why Layer 5 stays silent. Library
# callers use the shared resolver directly and rely on its debug log.
from src.data.manifests.resolution import (  # noqa: E402
    known_manifest_sources as _known_manifest_sources,
)
from src.data.manifests.resolution import (  # noqa: E402
    resolve_manifest_source as _shared_resolve_manifest_source,
)

_FEATURE_MANIFEST_SOURCES: tuple[str, ...] = _known_manifest_sources()


def _resolve_feature_manifest_source(
    data_dir: str | None,
    override: str | None,
) -> str | None:
    """Operator-facing wrapper around the shared manifest-source resolver.

    Delegates the M1 (ambiguous) / M2 (override conflict) / unknown-override
    strictness + path-segment auto-detection to
    ``src.data.manifests.resolution.resolve_manifest_source`` (single source of
    truth) and additionally emits a stderr WARNING when a provided ``data_dir``
    resolved to no manifest, so an operator running an unrecognised RWD path
    sees why Layer 5 verdicts stay silent. Synthetic / ad-hoc runs (no
    ``data_dir``) remain frictionless.

    Without this resolution, ``feature_manifest_source`` was never threaded into
    ``scope_spec`` for RWD runs and Layer 5's manifest-driven Layer 1 verdicts
    silently no-op'd; forbidden columns fell through to Layer 3
    (statistical-only), undercutting the deterministic post-index leak catch.
    """
    resolved = _shared_resolve_manifest_source(data_dir, override)
    if resolved is None and override is None and data_dir:
        import sys as _sys

        print(
            f"WARNING: data_dir={data_dir!r} does not contain a known "
            f"manifest source segment ({_FEATURE_MANIFEST_SOURCES}); "
            f"Layer 5 manifest verdicts will not fire for this run. Pass "
            f"--feature-manifest-source explicitly if a registry should "
            f"be consulted.",
            file=_sys.stderr,
        )
    return resolved


def _runner_exclude_set(state: "dict") -> "set[str]":
    """The runner's hard denylist of non-feature / target-proxy columns, merged
    with the scope_definer's canonical excluded_features. Single source of truth
    for Step-5 feature discovery (RC2)."""
    exclude = {
        "patient_journey_id",
        "patient_id",
        "patient_hash",
        "brand",
        "journey_start_date",
        "journey_end_date",
        "created_at",
        "updated_at",
        "source_timestamp",
        "ingestion_timestamp",
        "data_split",
        "split_config_id",
        "data_source",
        "data_sources_matched",
        "source_match_confidence",
        "source_combination_method",
        "source_stacking_flag",
        "data_lag_hours",
        # data_quality_score encodes data-source archetype (A/B/C);
        # after cohort filtering it nearly separates treated from untreated
        "data_quality_score",
        # primary_diagnosis_code removed — it's a legitimate low-cardinality
        # feature; leakage detector will vet it
        "primary_diagnosis_desc",
        "secondary_diagnosis_codes",
        "state",
        "zip_code",
        "comorbidities",
        "risk_score",
        # Derived from target — potential leakers
        "journey_stage",
        "journey_status",
        CONFIG.target_outcome,
        "discontinuation_flag",
        "treatment_initiated",
    }
    # Merge scope_definer's canonical excluded_features (PII, temporal leakage,
    # pipeline construction metadata) so Step-5 discovery honors the same policy
    # as data_preparer.
    exclude |= set((state.get("scope_spec") or {}).get("excluded_features", []) or [])
    return exclude


def _discover_model_feature_cols(
    eligible_df: "pd.DataFrame",
    exclude: "set[str]",
    min_non_null_frac: float = 0.5,
    max_categorical_cardinality: int = 50,
) -> "list[str]":
    """Stage-1 leakage-INDEPENDENT discovery: keep well-formed predictors and
    drop constants / too-sparse columns. The leakage layer may only SUBTRACT
    genuine leaks (via ``exclude``); it must never shrink the matrix to a curated
    sub-list. Sparse cardinality-2 pre-index flags (100%-non-null) are retained —
    they pass nunique>1 and the non-null cap.

    Tier D: the null-rate floor (``min_non_null_frac``) and the categorical
    high-cardinality cap (``max_categorical_cardinality``) are configurable
    (CONFIG.feature_min_non_null_frac / CONFIG.feature_max_cardinality) rather
    than hardcoded magic numbers. The ``nunique > 1`` constant-drop is NOT a
    tunable — a zero-variance column carries no signal, so it always drops.
    """
    numeric_cols = [
        c
        for c in eligible_df.columns
        if c not in exclude
        and eligible_df[c].dtype.kind in "iufb"
        and eligible_df[c].nunique() > 1
        and eligible_df[c].notna().mean() > min_non_null_frac
    ]
    categorical_cols = [
        c
        for c in eligible_df.columns
        if c not in exclude
        and eligible_df[c].dtype == object
        and 2 <= eligible_df[c].nunique() <= max_categorical_cardinality
        and eligible_df[c].notna().mean() > min_non_null_frac
    ]
    return numeric_cols + categorical_cols


def _route_leakage_outputs(
    state: dict,
    *,
    severity: str,
    leaked: list,
    findings: list,
    source: str,
    is_scenario_regime: bool,
) -> None:
    """Route PRE-TRAINING heuristic leakage outputs to live state or to diagnostics.

    Two pre-training heuristic detectors write leakage state in the runner: Step-2's
    graph (``adaptive_validity_check`` can escalate severity even when the name-based
    detector is ``skip_leakage_check``-gated) and Step-5's structural checks on the
    real feature matrix. Both can FALSE-POSITIVE on clinically-grounded synthetic
    fixtures (the documented ``journey_status`` trap). On those scenario regimes,
    writing the live ``leakage_severity``/``leaked_features``/``leakage_findings``
    would (a) trip the Step-5a LLM remediator (which then hallucinates a replacement
    feature set) and (b) block deployment at the leakage gate — neither warranted,
    since the fixture has no real leakage by construction.

    So on a scenario regime the findings are recorded under
    ``state["leakage_diagnostics"][source]`` (transparency) and the live fields are
    left untouched. On a real / RWD regime the live fields are written as before, so
    genuine leakage still remediates and blocks.

    The POST-training EMPIRICAL signal (``leakage_suspected``/``suspicion_level`` from
    ``check_imbalance_aware_suspicion``) is the genuine leakage gate and is NOT routed
    here — it stays live on every regime (see ``_deploy_blocked_by_leakage``).
    """
    if is_scenario_regime:
        diagnostics = state.setdefault("leakage_diagnostics", {})
        diagnostics[source] = {
            "severity": severity,
            "leaked_features": list(leaked),
            "findings": list(findings),
            "note": (
                "scenario regime: pre-training heuristic finding recorded for "
                "transparency but NOT applied to the live leakage gate (no real "
                "leakage by construction; the post-training empirical signal "
                "remains the live gate)"
            ),
        }
        return
    state["leakage_severity"] = severity
    state["leaked_features"] = leaked
    state["leakage_findings"] = findings


def _deploy_blocked_by_leakage(state: dict) -> bool:
    """Whether the leakage gate should block deployment.

    Mirrors the model-deployer leakage gate: the POST-training empirical suspicion
    signal (``leakage_suspected`` / ``suspicion_level``) OR a live high/critical
    ``leakage_severity`` blocks. On scenario regimes the pre-training heuristics are
    diagnostic-only (see ``_route_leakage_outputs``), so on those regimes this fires
    only on the genuine post-training empirical signal.
    """
    return bool(
        state.get("leakage_suspected", False)
        or state.get("leakage_severity", "none") in ("high", "critical")
        or state.get("suspicion_level", "none") in ("high", "critical")
    )


async def run_pipeline(
    step: int | None = None,
    dry_run: bool = False,
    imbalance_ratio: float | None = None,
    include_bentoml: bool = True,
    data_dir: str | None = None,
    *,
    regime: str = "default",
    deployment_intent: str = "clinical",
    split_mode: str = "auto",
    pre_assigned_splits: Dict[Any, str] | None = None,
    inject_demo_cost_matrix: bool = True,
    feature_manifest_source: str | None = None,
    n_total: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full pipeline or a specific step.

    Args:
        step: Run only a specific step (1-8), or None for all steps
        dry_run: Show what would be done without executing
        imbalance_ratio: If provided, create imbalanced data with this minority ratio
        include_bentoml: If True, deploy real model to BentoML and verify predictions
        data_dir: If provided, load real-world data from this directory instead of
                  generating synthetic data
        regime: Synthetic data generator regime.

            - ``"default"``: balanced base rate (positive_rate=0.30) — the
              regime that has been the historical default for tier0 runs.
            - ``"adverse"``: extreme-imbalance regime (positive_rate=0.02)
              that exercises the imbalance remediation paths
              (``recommended_strategy=combined``).

            Ignored when ``data_dir`` is set (RWD overrides synthetic).
        split_mode: Split strategy to pass through to ``step_5_model_trainer``.
            ``"auto"`` (default) lets the function decide based on whether
            entity + date columns are available; ``"random"`` is the explicit
            opt-out for the legacy stratified random 4-way split.
        pre_assigned_splits: Optional cached entity → split-label mapping to
            replay (Block 4, Finding #12). When supplied, ``step_5`` will
            refuse to re-derive splits.
        inject_demo_cost_matrix: When True (the default), a unit-shape
            placeholder ``cost_matrix`` is auto-injected onto
            ``state["scope_spec"]`` so the evaluator can emit
            ``business_utility``. See ``_default_demo_cost_matrix`` for the
            shape rationale. Set to False (CLI: ``--no-demo-cost-matrix``)
            to reproduce the pre-Block-5B baseline where
            ``business_utility`` is absent because no cost matrix flowed
            through. The auto-inject lives ONLY at the dev-script
            boundary — production tier0 runs through the LangGraph
            orchestrator, never this script.

    Returns:
        State dictionary containing all pipeline outputs. When step 5 ran,
        the returned dict also carries ``split_assignments`` so callers can
        persist them in the tier0 cache and refuse to re-split on reload.
    """
    import time

    experiment_id = f"tier0_e2e_{uuid.uuid4().hex[:8]}"
    # D1.3: mint a workflow-level audit_workflow_id once and thread it
    # through every agent.run() call below. Pre-D1.3 each agent's State
    # minted a fresh UUID via Field(default_factory=uuid4), severing the
    # audit chain across the dev-script's pipeline. After D1.4 drops
    # the default_factory, this thread becomes the only path that keeps
    # script-driven pipelines working.
    audit_workflow_id = uuid.uuid4()
    pipeline_start_time = time.time()

    if regime not in _VALID_REGIMES:
        raise ValueError(f"regime must be one of {_VALID_REGIMES}, got {regime!r}")
    if split_mode not in {"auto", "random", "combined"}:
        raise ValueError(f"split_mode must be 'auto'|'random'|'combined', got {split_mode!r}")

    print(f"\n{'=' * 70}")
    print(f"TIER 0 MLOPS WORKFLOW TEST")
    print(f"{'=' * 70}")
    print(f"  Experiment ID: {experiment_id}")
    print(f"  Brand: {CONFIG.brand}")
    print(f"  Target: {CONFIG.target_outcome}")
    print(f"  Problem Type: {CONFIG.problem_type}")
    print(f"  MLflow Enabled: {CONFIG.enable_mlflow}")
    print(f"  MLflow Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI', 'not set')}")
    print(f"  BentoML Serving: {'Enabled' if include_bentoml else 'Disabled'}")
    _regime_summary = _regime_kwargs(regime, seed=seed)
    _scenario_generator = _regime_summary.get("_generator")
    if _scenario_generator in _SCENARIO_REGIME_TO_NAME:
        n_total_disp = "default" if n_total is None else str(n_total)
        print(
            f"  Regime: {regime} "
            f"(synthetic_v2.{_scenario_generator}, n_total={n_total_disp}, seed={seed})"
        )
    else:
        print(
            f"  Regime: {regime} "
            f"(positive_rate={_regime_summary['positive_rate']:.2f}, "
            f"signal_strength={_regime_summary['signal_strength']:.2f}, "
            f"noise_sd={_regime_summary['noise_sd']:.2f}, "
            f"signalize_extras={_regime_summary['signalize_extra_features']})"
        )
    print(f"  Split mode: {split_mode}")
    if imbalance_ratio:
        print(f"  Class Imbalance: {imbalance_ratio:.1%} minority ratio (INJECTED)")
    print(f"  Started: {datetime.now().isoformat()}")

    if dry_run:
        print("\n  [DRY RUN MODE - No agents will be executed]")
        return

    # Generate sample data
    # NOTE: Generate 1500 samples to satisfy scope_spec.minimum_samples=500 and
    # provide enough data for hierarchical CATE analysis (~500 per segment in
    # CausalForestDML's 3-segment cross-fitting, preventing zero-variance leaves)
    if data_dir:
        print(f"\n  Loading real-world data from {data_dir}...")
        patient_df = load_rwd_data(data_dir, target=CONFIG.target_outcome)
    else:
        regime_kwargs = _regime_kwargs(regime, seed=seed)
        # #633: the clean regime generates a LARGER cohort (4000 vs the
        # legacy 1500). The binding v3 gate for clean is the RANKING-based
        # ``maximum_train_val_delta`` (|train_roc_auc − val_roc_auc| ≤ 0.03),
        # which post-hoc calibration CANNOT touch (it is monotonic) — so the
        # only honest levers are (a) more data (shrinks the tree's
        # train↔val memorization gap WITH MARGIN) and (b) a more-regularized
        # champion. More N also keeps the feature-density tier at fpr ≤ 1/50
        # (so the 0.03 bar does NOT move — actual overfit drops) AND gives
        # post-hoc calibration more validation positives so it over-fits the
        # small val set less (van Calster slope → 1). default/adverse/scenario
        # regimes keep 1500 so their calibrated AUC bands / baselines are
        # byte-for-byte unchanged. See issue #633.
        n_samples = _regime_n_samples(regime)
        print("\n  Generating sample patient data...")
        patient_df = generate_sample_data(
            n_samples=n_samples,
            imbalance_ratio=imbalance_ratio,
            n_total=n_total,
            **regime_kwargs,
        )
        print(f"  Generated {len(patient_df)} patient records")

    # Pipeline state
    state: dict[str, Any] = {
        "experiment_id": experiment_id,
        "patient_df": patient_df,
    }

    # Collect step results for detailed summary
    step_results: list[StepResult] = []

    steps_to_run = [step] if step else list(range(1, 9))

    try:
        # Step 1: Scope Definer
        if 1 in steps_to_run:
            step_start = time.time()
            # Adaptive success criteria pre-eval inputs (task 05 of
            # adaptive_success_criteria plan). When ADAPTIVE_CRITERIA=true,
            # the validator reads these from state and stashes them for the
            # evaluator overlay. feature_columns are estimated from the
            # source dataframe via the same exclusion list used downstream
            # (see step_2_data_preparer line ~1787); the data_preparer may
            # later drop or add columns, but the rough count is sufficient
            # for the feature-density step function.
            _adaptive_feature_columns = [
                col
                for col in patient_df.columns
                if col not in ("patient_journey_id", CONFIG.target_outcome, "brand")
            ]
            adaptive_inputs = _compute_adaptive_state_inputs(
                df=patient_df,
                feature_columns=_adaptive_feature_columns,
                target_col=CONFIG.target_outcome,
                regime=regime,
                deployment_intent=deployment_intent,
            )
            result = await step_1_scope_definer(experiment_id, adaptive_inputs=adaptive_inputs)
            state["scope_spec"] = result.get("scope_spec", {"problem_type": CONFIG.problem_type})
            state["scope_spec"]["experiment_id"] = experiment_id
            # Layer 5 manifest opt-in: thread the resolved cohort manifest
            # source so adaptive_validity_check consults the matching
            # FeatureContract registry (CSU / Optum) for layer="1" verdicts.
            # Pre-fix this was never set on RWD runs, so post-index columns
            # fell through to Layer 3 (statistical) instead of being caught
            # deterministically by Layer 1 manifest contracts.
            # #604: for legacy synthetic fixtures with no explicit override, this
            # resolves to the 'synthetic' manifest so the legit pre-index columns
            # (days_on_therapy/hcp_visits/prior_treatments) clear Layer 1 and the
            # declared-safe full-immunity carve-out can protect them under FDR-on.
            # Shared with the Step-2 block so a partial --step 2 run wires it too.
            _apply_synthetic_manifest_source(
                state["scope_spec"], regime, data_dir, feature_manifest_source
            )
            # Block 5B (#10): auto-inject the unit-shape placeholder cost
            # matrix when the caller has not explicitly opted out via
            # ``--no-demo-cost-matrix``. This closes Block 5's verification
            # gap: without a cost matrix the evaluator short-circuits
            # business_utility, so a default ``python scripts/run_tier0_test.py``
            # run produced no business_utility number to verify against.
            # Decision logic extracted to ``_should_inject_demo_cost_matrix``
            # so unit tests exercise the real branch (5B-I-2).
            if _should_inject_demo_cost_matrix(state["scope_spec"], inject_demo_cost_matrix):
                state["scope_spec"]["cost_matrix"] = _default_demo_cost_matrix()
            duration = time.time() - step_start
            scope_spec = state["scope_spec"]
            success_criteria = result.get("success_criteria", {})
            state["success_criteria"] = success_criteria
            validation_passed = result.get("validation_passed", True)

            step_results.append(
                StepResult(
                    step_num=1,
                    step_name="SCOPE DEFINER",
                    status="success" if validation_passed else "warning",
                    duration_seconds=duration,
                    key_metrics={
                        "experiment_id": result.get("experiment_id", experiment_id),
                        "problem_type": scope_spec.get("problem_type"),
                        "prediction_target": scope_spec.get("prediction_target"),
                        "minimum_samples": scope_spec.get("minimum_samples"),
                    },
                    details={
                        "brand": CONFIG.brand,
                        "success_criteria": success_criteria,
                    },
                    # Enhanced format fields
                    input_summary={
                        "problem_description": scope_spec.get(
                            "problem_description", "Predict patient discontinuation risk"
                        ),
                        "business_objective": scope_spec.get(
                            "business_objective", "Identify high-risk patients"
                        ),
                        "target_outcome": scope_spec.get(
                            "prediction_target", CONFIG.target_outcome
                        ),
                        "problem_type_hint": CONFIG.problem_type,
                        "brand": CONFIG.brand,
                    },
                    validation_checks=[
                        (
                            "Problem type defined",
                            scope_spec.get("problem_type") is not None,
                            "problem_type present",
                            scope_spec.get("problem_type", "None"),
                        ),
                        (
                            "Prediction target set",
                            scope_spec.get("prediction_target") is not None,
                            "prediction_target present",
                            scope_spec.get("prediction_target", "None"),
                        ),
                        (
                            "Minimum samples specified",
                            (scope_spec.get("minimum_samples") or 0) > 0,
                            "minimum_samples > 0",
                            scope_spec.get("minimum_samples", 0),
                        ),
                        (
                            "Scope validation",
                            validation_passed,
                            "validation_passed = True",
                            f"validation_passed = {validation_passed}",
                        ),
                    ],
                    metrics_table=[
                        ("experiment_id", result.get("experiment_id", experiment_id), None, None),
                        ("problem_type", scope_spec.get("problem_type"), None, None),
                        ("prediction_target", scope_spec.get("prediction_target"), None, None),
                        ("minimum_samples", scope_spec.get("minimum_samples"), None, None),
                    ],
                    interpretation=[
                        f"Binary classification scope defined for patient risk prediction",
                        f"Target outcome: {scope_spec.get('prediction_target', CONFIG.target_outcome)}",
                        f"Sample requirement ({scope_spec.get('minimum_samples', 'N/A')}) appropriate for ML",
                    ],
                    result_message="Scope definition complete",
                )
            )

        # Step 2: Data Preparer
        if 2 in steps_to_run:
            step_start = time.time()
            scope_spec = state.get("scope_spec", {"problem_type": CONFIG.problem_type})
            # #604: make Step 2 self-sufficient for partial (--step 2) runs — the
            # Step-1 block that injects the manifest source is skipped when Step 2
            # runs alone. Idempotent in the full-pipeline path (Step 1 already set it).
            _apply_synthetic_manifest_source(scope_spec, regime, data_dir, feature_manifest_source)
            result = await step_2_data_preparer(
                experiment_id,
                scope_spec,
                patient_df,
                # All synthetic_v2 scenario regimes are clinically-grounded
                # fixtures with no real leakage — they share the
                # journey_status="active" sentinel that confuses the LLM
                # remediator. See step_2_data_preparer docstring.
                skip_leakage_check=(regime in _SCENARIO_REGIME_TO_NAME),
                # #594/#604: the Layer-3 FDR firing driver (#538) false-positively
                # auto-drops the designed outcome-correlated predictors
                # (days_on_therapy/hcp_visits/prior_treatments) on synthetic
                # fixtures. #604: for the LEGACY ml_patients fixtures FDR stays ON
                # and those legit columns are protected by the synthetic-manifest
                # declared-safe carve-out with FULL immunity (set just below +
                # the 'synthetic' manifest threaded at the scope_spec block above).
                # scenario_* (synthetic_v2, different columns, not in must-pass CI)
                # retains the #594 wholesale FDR-disable. Real --data-dir/RWD runs
                # keep FDR ON, immunity OFF — see the three resolvers.
                adaptive_fdr_enabled=_resolve_adaptive_fdr_enabled(regime, data_dir),
                # codex round-2: gate immunity on the EFFECTIVE manifest source
                # (set on scope_spec just above), so an operator csu/optum override
                # on a legacy regime correctly withholds immunity.
                adaptive_declared_safe_full_immunity=_resolve_declared_safe_full_immunity(
                    regime, data_dir, scope_spec.get("feature_manifest_source")
                ),
                data_dir=data_dir,
            )
            state["qc_report"] = result.get("qc_report", {"gate_passed": True})
            state["gate_passed"] = result.get("gate_passed", True)

            # Store DataFrames from data_preparer if available
            if result.get("train_df") is not None:
                state["train_df"] = result["train_df"]
            if result.get("validation_df") is not None:
                state["validation_df"] = result["validation_df"]

            # Propagate leakage detection state. On scenario regimes the Step-2
            # graph's leakage outputs (incl. any adaptive_validity_check escalation,
            # which is NOT skip_leakage_check-gated) are PRE-TRAINING heuristics that
            # can false-positive on synthetic fixtures — route them to diagnostics so
            # they neither trip Step-5a remediation nor block deploy (FU1 / #528).
            _route_leakage_outputs(
                state,
                severity=result.get("leakage_severity", "none"),
                leaked=result.get("leaked_features", []),
                findings=result.get("leakage_findings", []),
                source="step2_graph",
                is_scenario_regime=(regime in _SCENARIO_REGIME_TO_NAME),
            )
            # Propagate Layer 5 adaptive-validity audit trail (PR #84+) so
            # the TIER0_E2E_JSON_OUT artifact captures per-feature verdicts
            # for the CSU val_AUC measurement test (Item A2). Without this
            # the data_preparer's adaptive_verdicts list never reaches the
            # tier0 state dict that ``run_pipeline`` returns.
            state["adaptive_verdicts"] = result.get("adaptive_verdicts", [])
            state["leakage_dropped_features"] = result.get("leakage_dropped_features", [])

            # Propagate leakage remediation state (from LLM-assisted remediation node)
            if result.get("leakage_remediation_status"):
                state["leakage_remediation_status"] = result["leakage_remediation_status"]
                state["leakage_remediated_features"] = result.get("leakage_remediated_features", [])
                state["leakage_dropped_features"] = result.get("leakage_dropped_features", [])
                state["leakage_added_features"] = result.get("leakage_added_features", [])
                state["leakage_remediation_reasoning"] = result.get("leakage_remediation_reasoning")
                state["leakage_remediation_viable"] = result.get("leakage_remediation_viable", True)

                _rem_status = result["leakage_remediation_status"]
                if _rem_status == "applied" and result.get("leakage_remediated_features"):
                    print(f"\n  🔧 Leakage remediation: {_rem_status}")
                    print(f"     Dropped: {result.get('leakage_dropped_features', [])}")
                    print(f"     Clean features: {result.get('leakage_remediated_features', [])}")
                    print(f"     Reasoning: {result.get('leakage_remediation_reasoning', 'N/A')}")
                elif _rem_status == "failed" or not result.get("leakage_remediation_viable", True):
                    print(f"\n  🚨 Leakage remediation FAILED: no viable features")
                    print(f"     Reasoning: {result.get('leakage_remediation_reasoning', 'N/A')}")
                    state["pipeline_halted"] = True
                    state["halt_reason"] = result.get(
                        "leakage_remediation_reasoning",
                        "No viable features after leakage remediation",
                    )

            qc_report = result.get("qc_report", {})
            data_readiness = result.get("data_readiness", {})
            train_samples = data_readiness.get("train_samples", 0)
            val_samples = data_readiness.get("validation_samples", 0)
            overall_score = qc_report.get("overall_score", 0)
            step_results.append(
                StepResult(
                    step_num=2,
                    step_name="DATA PREPARER",
                    status="success" if state["gate_passed"] else "failed",
                    duration_seconds=time.time() - step_start,
                    key_metrics={
                        "qc_status": qc_report.get("status", "unknown"),
                        "overall_score": overall_score,
                        "gate_passed": state["gate_passed"],
                        "qc_train_samples": train_samples,
                        "qc_validation_samples": val_samples,
                    },
                    details={
                        "completeness_score": qc_report.get("completeness_score"),
                        "validity_score": qc_report.get("validity_score"),
                        "consistency_score": qc_report.get("consistency_score"),
                        "uniqueness_score": qc_report.get("uniqueness_score"),
                        "timeliness_score": qc_report.get("timeliness_score"),
                    },
                    # Enhanced format fields
                    input_summary={
                        "experiment_id": experiment_id,
                        "scope_spec_problem_type": scope_spec.get(
                            "problem_type", CONFIG.problem_type
                        ),
                        "input_samples": len(patient_df),
                    },
                    validation_checks=[
                        (
                            "QC gate passed",
                            state["gate_passed"],
                            "gate_passed = True",
                            f"gate_passed = {state['gate_passed']}",
                        ),
                        (
                            "Overall score acceptable",
                            (overall_score or 0) >= 0.7,
                            "≥ 0.70",
                            f"{overall_score:.2f}" if overall_score else "N/A",
                        ),
                        (
                            "QC training samples sufficient",
                            train_samples >= 50,
                            "≥ 50",
                            train_samples,
                        ),
                        ("QC validation samples present", val_samples > 0, "> 0", val_samples),
                    ],
                    metrics_table=[
                        (
                            "overall_score",
                            f"{overall_score:.2f}" if overall_score else "N/A",
                            "≥ 0.70",
                            (overall_score or 0) >= 0.7,
                        ),
                        (
                            "completeness_score",
                            f"{qc_report.get('completeness_score', 0):.2f}",
                            None,
                            None,
                        ),
                        ("validity_score", f"{qc_report.get('validity_score', 0):.2f}", None, None),
                        (
                            "consistency_score",
                            f"{qc_report.get('consistency_score', 0):.2f}",
                            None,
                            None,
                        ),
                        ("qc_train_samples", train_samples, "≥ 50", train_samples >= 50),
                        ("qc_validation_samples", val_samples, "> 0", val_samples > 0),
                    ],
                    interpretation=[
                        f"Data quality score: {overall_score:.2f}"
                        if overall_score
                        else "Data quality score: N/A",
                        f"QC sample split: {train_samples} QC-train, {val_samples} QC-validation (full training split in Step 5)",
                        "QC gate PASSED - data ready for modeling"
                        if state["gate_passed"]
                        else "QC gate FAILED - data quality issues detected",
                    ],
                    result_message="Data preparation complete"
                    if state["gate_passed"]
                    else "Data preparation failed QC gate",
                )
            )

            if not state["gate_passed"]:
                print_failure("QC Gate blocked training. Pipeline stopped.")
                # Halt subsequent steps via the pipeline_halted flag (each
                # step gates on `not state.get("pipeline_halted")`) instead
                # of bailing early, so the TIER0_E2E_JSON_OUT artifact at
                # the end of run_pipeline still captures adaptive_verdicts
                # from the data_preparer's Layer 5 audit. Test fixtures
                # need this artifact even on QC halt to verify Layer 5
                # invariants on real RWD cohorts (backlog item #12).
                state["pipeline_halted"] = True
                state.setdefault("halt_reason", "qc_gate_blocked")

            # === Step 2a: Sampling-frame audit gate (Phase-1 Task 1.3) ===
            # Surface the audit's blocking decision as its own step result so
            # operators see "SAMPLING FRAME AUDIT: failed" alongside (and
            # independent of) the QC gate. The audit node itself populates
            # blocking_issues, so the QC gate above usually catches drift;
            # this branch also halts if a downstream node clobbered the
            # blocking entry but the audit report still has blocking_detail.
            sampling_frame_report = result.get("sampling_frame_audit_report", {}) or {}
            sf_threshold = CONFIG.sampling_frame_max_drift
            sf_max_drift = sampling_frame_report.get("max_drift_score")
            sf_blocking_detail = sampling_frame_report.get("blocking_detail") or {}
            sf_exceeded = bool(sf_blocking_detail) or (
                sf_max_drift is not None and float(sf_max_drift) > sf_threshold
            )
            sf_message = (
                sf_blocking_detail.get("message")
                if sf_blocking_detail
                else f"max_drift_score={sf_max_drift!r} <= {sf_threshold:.4f}"
            )
            step_results.append(
                StepResult(
                    step_num="2a",
                    step_name="SAMPLING FRAME AUDIT",
                    status="failed" if sf_exceeded else "success",
                    key_metrics={
                        "max_drift_score": sf_max_drift,
                        "threshold": sf_threshold,
                        "columns_with_drift": sampling_frame_report.get("columns_with_drift", []),
                    },
                    details={"blocking_detail": sf_blocking_detail},
                    result_message=sf_message,
                )
            )
            if sf_exceeded:
                print_failure(f"Sampling-frame audit blocked training: {sf_message}")
                # See QC-gate halt above — set pipeline_halted instead of
                # returning so the TIER0_E2E_JSON_OUT artifact at the end
                # of run_pipeline still captures adaptive_verdicts.
                state["pipeline_halted"] = True
                state.setdefault("halt_reason", "sampling_frame_audit_blocked")

        # Step 2b: Feast Feature Registration (gracefully degrading)
        if 2 in steps_to_run and not state.get("pipeline_halted"):
            step_start = time.time()
            feast_reg_result = await step_2b_feast_registration(experiment_id, state)
            feast_reg_status = feast_reg_result.get("status", "skipped")
            features_registered = feast_reg_result.get("features_registered", 0)
            step_results.append(
                StepResult(
                    step_num="2b",
                    step_name="FEAST FEATURE REGISTRATION",
                    status=feast_reg_status if feast_reg_status != "skipped" else "warning",
                    duration_seconds=time.time() - step_start,
                    key_metrics={
                        "features_registered": features_registered,
                        "features_skipped": feast_reg_result.get("features_skipped", 0),
                        "feature_group_created": feast_reg_result.get(
                            "feature_group_created", False
                        ),
                    },
                    details={"errors": feast_reg_result.get("errors", [])},
                    input_summary={
                        "experiment_id": experiment_id,
                        "entity_key": "hcp_id",
                    },
                    validation_checks=[
                        (
                            "Feast registration attempted",
                            feast_reg_status != "skipped",
                            "not skipped",
                            feast_reg_status,
                        ),
                        (
                            "Features registered",
                            features_registered > 0,
                            "> 0",
                            str(features_registered),
                        ),
                    ],
                    metrics_table=[
                        (
                            "features_registered",
                            features_registered,
                            "> 0",
                            features_registered > 0,
                        ),
                        ("status", feast_reg_status, None, None),
                    ],
                    interpretation=[
                        f"Feast feature registration: {feast_reg_status}",
                        f"{features_registered} features registered to feature store",
                    ],
                    result_message=f"Feast registration: {feast_reg_status} ({features_registered} features)"
                    if feast_reg_status != "skipped"
                    else "Feast registration skipped (service unavailable)",
                )
            )

        # Step 2c: Feast Freshness Check (gracefully degrading)
        if 2 in steps_to_run and not state.get("pipeline_halted"):
            step_start = time.time()
            freshness_result = await step_2c_feast_freshness_check(state)
            freshness_status = freshness_result.get("status", "skipped")
            is_fresh = freshness_result.get("fresh", None)
            stale_count = len(freshness_result.get("stale_features", []))
            step_results.append(
                StepResult(
                    step_num="2c",
                    step_name="FEAST FRESHNESS CHECK",
                    status=freshness_status if freshness_status != "skipped" else "warning",
                    duration_seconds=time.time() - step_start,
                    key_metrics={
                        "fresh": is_fresh,
                        "stale_features_count": stale_count,
                    },
                    details={"errors": freshness_result.get("errors", [])},
                    input_summary={
                        "max_staleness_hours": 24.0,
                    },
                    validation_checks=[
                        (
                            "Freshness check attempted",
                            freshness_status != "skipped",
                            "not skipped",
                            freshness_status,
                        ),
                        (
                            "Features fresh",
                            is_fresh is True,
                            "all fresh",
                            f"{stale_count} stale" if stale_count else "all fresh",
                        ),
                    ],
                    metrics_table=[
                        (
                            "fresh",
                            str(is_fresh) if is_fresh is not None else "N/A",
                            "True",
                            is_fresh is True,
                        ),
                        ("stale_features", stale_count, "0", stale_count == 0),
                    ],
                    interpretation=[
                        f"Feast freshness check: {freshness_status}",
                        f"{'All features fresh' if is_fresh else f'{stale_count} stale features detected'}"
                        if is_fresh is not None
                        else "Freshness check was skipped",
                    ],
                    result_message=f"Freshness: {'all fresh' if is_fresh else f'{stale_count} stale'}"
                    if freshness_status != "skipped"
                    else "Freshness check skipped (service unavailable)",
                )
            )

        # Step 3: Cohort Constructor
        if 3 in steps_to_run and not state.get("pipeline_halted"):
            step_start = time.time()
            eligible_df, cohort_result = await step_3_cohort_constructor(patient_df)
            state["eligible_df"] = eligible_df
            state["cohort_result"] = cohort_result
            input_count = len(patient_df)
            eligible_count = len(eligible_df)
            excluded_count = input_count - eligible_count
            exclusion_rate = excluded_count / input_count if input_count > 0 else 0
            step_results.append(
                StepResult(
                    step_num=3,
                    step_name="COHORT CONSTRUCTOR",
                    status="success"
                    if eligible_count >= CONFIG.min_eligible_patients
                    else "warning",
                    duration_seconds=time.time() - step_start,
                    key_metrics={
                        "cohort_id": cohort_result.cohort_id,
                        "input_patients": input_count,
                        "eligible_patients": eligible_count,
                        "excluded_patients": excluded_count,
                        "exclusion_rate": f"{exclusion_rate:.1%}",
                    },
                    details={
                        "execution_id": cohort_result.execution_id,
                        "status": cohort_result.status,
                    },
                    # Enhanced format fields
                    input_summary={
                        "input_patients": input_count,
                        "brand": CONFIG.brand,
                        "min_eligible_required": CONFIG.min_eligible_patients,
                    },
                    validation_checks=[
                        (
                            "Sufficient eligible patients",
                            eligible_count >= CONFIG.min_eligible_patients,
                            f"≥ {CONFIG.min_eligible_patients}",
                            eligible_count,
                        ),
                        (
                            "Exclusion rate reasonable",
                            exclusion_rate <= 0.5,
                            "≤ 50%",
                            f"{exclusion_rate:.1%}",
                        ),
                        (
                            "Cohort ID generated",
                            cohort_result.cohort_id is not None,
                            "cohort_id present",
                            cohort_result.cohort_id or "None",
                        ),
                        (
                            "Cohort status valid",
                            cohort_result.status in ["completed", "success"],
                            "completed/success",
                            cohort_result.status,
                        ),
                    ],
                    metrics_table=[
                        ("input_patients", input_count, None, None),
                        (
                            "eligible_patients",
                            eligible_count,
                            f"≥ {CONFIG.min_eligible_patients}",
                            eligible_count >= CONFIG.min_eligible_patients,
                        ),
                        ("excluded_patients", excluded_count, None, None),
                        ("exclusion_rate", f"{exclusion_rate:.1%}", "≤ 50%", exclusion_rate <= 0.5),
                    ],
                    interpretation=[
                        f"Cohort constructed with {eligible_count} eligible patients from {input_count} total",
                        f"Exclusion rate: {exclusion_rate:.1%} ({excluded_count} patients excluded)",
                        f"Cohort size {'meets' if eligible_count >= CONFIG.min_eligible_patients else 'below'} minimum threshold of {CONFIG.min_eligible_patients}",
                    ],
                    result_message=f"Cohort '{cohort_result.cohort_id}' constructed with {eligible_count} patients",
                )
            )

        # Step 4: Model Selector
        if 4 in steps_to_run and not state.get("pipeline_halted"):
            step_start = time.time()
            scope_spec = state.get("scope_spec", {"problem_type": CONFIG.problem_type})
            qc_report = state.get("qc_report", {"gate_passed": True})

            # Compute feature characteristics for model selector scoring
            _step4_df = state.get("eligible_df", patient_df)
            _step4_exclude = {
                "patient_journey_id",
                "patient_id",
                "patient_hash",
                "brand",
                "journey_start_date",
                "journey_end_date",
                "created_at",
                "updated_at",
                "source_timestamp",
                "ingestion_timestamp",
                "data_split",
                "split_config_id",
                "data_source",
                "data_sources_matched",
                "source_match_confidence",
                "source_combination_method",
                "source_stacking_flag",
                "data_lag_hours",
                # data_quality_score encodes data-source archetype (A/B/C);
                # after cohort filtering it nearly separates treated from untreated
                "data_quality_score",
                "primary_diagnosis_desc",
                "secondary_diagnosis_codes",
                "state",
                "zip_code",
                "comorbidities",
                "risk_score",
                "journey_stage",
                "journey_status",
                CONFIG.target_outcome,
                "discontinuation_flag",
                "treatment_initiated",
            }
            _s4_numeric = [
                c
                for c in _step4_df.columns
                if c not in _step4_exclude
                and _step4_df[c].dtype.kind in "iufb"
                and _step4_df[c].nunique() > 1
            ]
            _s4_categorical = [
                c
                for c in _step4_df.columns
                if c not in _step4_exclude
                and _step4_df[c].dtype == object
                and 2 <= _step4_df[c].nunique() <= 50
            ]
            _s4_total = len(_s4_numeric) + len(_s4_categorical)
            _cat_ratio = len(_s4_categorical) / _s4_total if _s4_total > 0 else 0.0
            _target_counts = _step4_df[CONFIG.target_outcome].value_counts()
            _min_ratio = _target_counts.min() / _target_counts.sum()
            if _min_ratio >= 0.40:
                _imb_sev = "none"
            elif _min_ratio >= 0.20:
                _imb_sev = "moderate"
            elif _min_ratio >= 0.05:
                _imb_sev = "severe"
            else:
                _imb_sev = "extreme"
            feature_characteristics = {
                "categorical_ratio": _cat_ratio,
                "num_numeric": len(_s4_numeric),
                "num_categorical": len(_s4_categorical),
                "class_imbalance_severity": _imb_sev,
            }
            state["feature_characteristics"] = feature_characteristics

            result = await step_4_model_selector(
                experiment_id,
                scope_spec,
                qc_report,
                feature_characteristics=feature_characteristics,
            )
            state["model_candidate"] = result.get("model_candidate") or result.get(
                "primary_candidate"
            )
            state["alternative_candidates"] = result.get("alternative_candidates", [])

            candidate = state["model_candidate"]
            algo_name = (
                candidate.get("algorithm_name")
                if isinstance(candidate, dict)
                else getattr(candidate, "algorithm_name", "Unknown")
            )
            # Extract selection_score from model_candidate (not selection_rationale)
            selection_score = (
                candidate.get("selection_score", 0) if isinstance(candidate, dict) else 0
            )
            # Use default_hyperparameters from agent output
            hyperparams = (
                candidate.get("default_hyperparameters", {}) if isinstance(candidate, dict) else {}
            )
            interpretability = (
                candidate.get("interpretability_score", 0) if isinstance(candidate, dict) else 0
            )

            # Extract selection rationale details
            selection_rationale = result.get("selection_rationale", {})
            primary_reason = (
                selection_rationale.get("primary_reason", "")
                if isinstance(selection_rationale, dict)
                else ""
            )
            supporting_factors = (
                selection_rationale.get("supporting_factors", [])
                if isinstance(selection_rationale, dict)
                else []
            )

            # Extract alternative candidates
            alternatives = result.get("alternative_candidates", [])
            alternatives_considered = (
                selection_rationale.get("alternatives_considered", [])
                if isinstance(selection_rationale, dict)
                else []
            )
            all_alternatives = alternatives if alternatives else alternatives_considered

            step_results.append(
                StepResult(
                    step_num=4,
                    step_name="MODEL SELECTOR",
                    status="success" if candidate else "warning",
                    duration_seconds=time.time() - step_start,
                    key_metrics={
                        "selected_algorithm": algo_name,
                        "selection_score": selection_score,
                        "alternatives_evaluated": len(all_alternatives),
                    },
                    details={
                        "selection_rationale": selection_rationale,
                        "alternative_candidates": all_alternatives,
                    },
                    # Enhanced format fields
                    input_summary={
                        "experiment_id": experiment_id,
                        "problem_type": scope_spec.get("problem_type", CONFIG.problem_type),
                        "qc_gate_passed": qc_report.get("gate_passed", True),
                    },
                    validation_checks=[
                        (
                            "Algorithm selected",
                            candidate is not None,
                            "candidate present",
                            algo_name or "None",
                        ),
                        (
                            "Selection score computed",
                            selection_score > 0,
                            "> 0",
                            f"{selection_score:.3f}" if selection_score else "N/A",
                        ),
                        (
                            "Rationale provided",
                            bool(primary_reason),
                            "reason present",
                            primary_reason[:30] if primary_reason else "None",
                        ),
                        (
                            "Alternatives evaluated",
                            len(all_alternatives) > 0,
                            "> 0",
                            f"{len(all_alternatives)} candidates",
                        ),
                    ],
                    metrics_table=[
                        ("algorithm", algo_name, None, None),
                        (
                            "selection_score",
                            f"{selection_score:.3f}" if selection_score else "N/A",
                            "> 0.5",
                            selection_score > 0.5 if selection_score else None,
                        ),
                        (
                            "interpretability",
                            f"{interpretability:.2f}" if interpretability else "N/A",
                            None,
                            None,
                        ),
                        ("default_hyperparameters", len(hyperparams), None, None),
                        (
                            "alternatives_evaluated",
                            len(all_alternatives),
                            "> 0",
                            len(all_alternatives) > 0,
                        ),
                    ],
                    interpretation=[
                        f"Selected {algo_name} (score: {selection_score:.3f})"
                        if selection_score
                        else f"Selected {algo_name}",
                        f"Reason: {primary_reason}"
                        if primary_reason
                        else "Selection based on problem type and data characteristics",
                        f"Evaluated {len(all_alternatives)} alternative{'s' if len(all_alternatives) != 1 else ''}: {', '.join([a.get('algorithm_name', str(a)) if isinstance(a, dict) else str(a) for a in all_alternatives[:3]])}"
                        if all_alternatives
                        else "No alternatives evaluated",
                        f"HPO will tune {len(hyperparams)} hyperparameters in Step 5"
                        if hyperparams
                        else "HPO will use default search space",
                    ],
                    result_message=f"Model selection complete: {algo_name} (score={selection_score:.3f})"
                    if selection_score
                    else f"Model selection complete: {algo_name}",
                )
            )

        # Step 5: Model Trainer
        if 5 in steps_to_run and not state.get("pipeline_halted"):
            step_start = time.time()
            eligible_df = state.get("eligible_df", patient_df)

            # RC2: always build the training matrix from the RETAINED set
            # (well-formed columns minus genuine leaks); never short-circuit to
            # the curated leakage_remediated_features survivor list. The leakage
            # layer may only SUBTRACT leaks; it must not define the matrix.
            _exclude = _runner_exclude_set(state)
            # Every feature any leakage layer flagged as a genuine leak.
            _genuine_leaks = set(state.get("leakage_dropped_features") or []) | set(
                state.get("leaked_features") or []
            )
            feature_cols = _discover_model_feature_cols(
                eligible_df,
                _exclude | _genuine_leaks,
                min_non_null_frac=CONFIG.feature_min_non_null_frac,
                max_categorical_cardinality=CONFIG.feature_max_cardinality,
            )
            if not feature_cols:
                # Absolute fallback to historical defaults
                feature_cols = ["days_on_therapy", "hcp_visits", "prior_treatments"]

            X = eligible_df[feature_cols].copy()
            y = eligible_df[CONFIG.target_outcome].copy()

            # === CATEGORICAL ENCODING ===
            # One-hot encode nominal categoricals so the LINEAR champion gets a
            # calibratable feature space. Integer/ordinal codes impose a false
            # magnitude order on nominal categories, and the downstream
            # ModelTrainerPreprocessor one-hots only *object*-dtype columns — so
            # pre-encoding to integers makes it skip its own encoding and the LR
            # trains on the distorted codes (disc: post-Platt slope_dev ~0.18 vs
            # ~0.07 one-hot → fails vs passes the calibration gate). One-hot
            # produces numeric data for the leakage/sklearn checks too; trees
            # consume it fine. See _fit_categorical_onehot.
            _cat_cols = [c for c in feature_cols if X[c].dtype == object]
            if _cat_cols:
                X, _enc_info = _fit_categorical_onehot(X, _cat_cols)
                feature_cols = list(X.columns)
                state["categorical_encoding"] = _enc_info
                print(
                    f"  One-hot encoded {len(_cat_cols)} categorical features "
                    f"-> {len(_enc_info['onehot_columns'])} columns: {_cat_cols}"
                )

            # === PRE-TRAINING LEAKAGE CHECK (on actual pipeline data) ===
            # The data_preparer's leakage detector runs on synthetic data,
            # so we run structural checks here on the real feature matrix + target.
            try:
                from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import (
                    check_perfect_class_separation,
                    check_zero_variance_within_class,
                    check_mutual_information,
                    check_feature_target_logical_dependency,
                    check_single_feature_auc,
                    _aggregate_severity,
                    _get_leaked_features,
                )
                import pandas as _pd

                _combined = _pd.concat([X, y], axis=1)
                _numeric_feats = [c for c in feature_cols if _combined[c].dtype.kind in "iufb"]
                _all_findings = []

                if len(_numeric_feats) > 0 and len(_combined) >= 30:
                    _all_findings.extend(
                        check_perfect_class_separation(
                            _combined, CONFIG.target_outcome, _numeric_feats
                        )
                    )
                    _all_findings.extend(
                        check_zero_variance_within_class(
                            _combined, CONFIG.target_outcome, _numeric_feats
                        )
                    )
                    _all_findings.extend(
                        check_mutual_information(_combined, CONFIG.target_outcome, _numeric_feats)
                    )
                    _all_findings.extend(
                        check_feature_target_logical_dependency(
                            _combined, CONFIG.target_outcome, _numeric_feats
                        )
                    )
                    _all_findings.extend(
                        check_single_feature_auc(_combined, CONFIG.target_outcome, _numeric_feats)
                    )

                if _all_findings:
                    _sev = _aggregate_severity(_all_findings)
                    _leaked = _get_leaked_features(_all_findings)
                    _findings_dicts = [f.to_dict() for f in _all_findings]
                    # Step-5 structural checks are PRE-TRAINING heuristics; on scenario
                    # regimes route to diagnostics (don't touch the live gate fields)
                    # so a synthetic false-positive doesn't block deploy / remediate.
                    _route_leakage_outputs(
                        state,
                        severity=_sev,
                        leaked=_leaked,
                        findings=_findings_dicts,
                        source="step5_structural",
                        is_scenario_regime=(regime in _SCENARIO_REGIME_TO_NAME),
                    )
                    print(f"\n  ⚠️  Pre-training leakage detection: {len(_all_findings)} findings")
                    print(f"     Severity: {_sev}")
                    print(f"     Leaked features: {_leaked}")
                    if regime in _SCENARIO_REGIME_TO_NAME:
                        print(
                            "     (scenario regime → recorded as diagnostic-only; "
                            "live leakage gate unaffected)"
                        )
                    for _f in _all_findings:
                        print(
                            f"     [{_f.severity.value.upper()}] {_f.check_name}: {_f.description}"
                        )
            except Exception as _e:
                print(f"  ⚠️  Pre-training leakage check error: {_e}")

            # === LEAKAGE REMEDIATION (Step 5a) ===
            # When critical/high leakage is detected, invoke the LLM-assisted
            # remediation node to reason about alternatives and rebuild features.
            if state.get("leakage_severity") in ("critical", "high") and not state.get(
                "leakage_remediated_features"
            ):
                print(
                    f"\n  🔧 LEAKAGE REMEDIATION: Analyzing {len(state.get('leaked_features', []))} leaked feature(s)..."
                )
                _rem_start = time.time()
                try:
                    from src.agents.ml_foundation.data_preparer.nodes.leakage_remediation import (
                        review_and_remediate_leakage,
                    )

                    # Build a minimal state dict for the remediation node
                    _rem_state = {
                        "experiment_id": experiment_id,
                        # Defense-in-depth: on a scenario regime this block is
                        # unreachable (live leakage_severity stays "none" — see
                        # _route_leakage_outputs), but if it ever is reached the
                        # remediator's own skip-gate must engage so it cannot
                        # hallucinate a replacement feature set on synthetic data.
                        "skip_leakage_check": (regime in _SCENARIO_REGIME_TO_NAME),
                        "leakage_severity": state["leakage_severity"],
                        "leaked_features": state["leaked_features"],
                        "leakage_findings": state["leakage_findings"],
                        "leakage_remediation_attempts": 0,
                        "train_df": eligible_df,
                        "scope_spec": state.get(
                            "scope_spec",
                            {
                                "prediction_target": CONFIG.target_outcome,
                                "problem_type": CONFIG.problem_type,
                            },
                        ),
                    }
                    _rem_result = await review_and_remediate_leakage(_rem_state)
                    _rem_status = _rem_result.get("leakage_remediation_status", "error")
                    _rem_features = _rem_result.get("leakage_remediated_features", [])
                    _rem_viable = _rem_result.get("leakage_remediation_viable", False)
                    _rem_reasoning = _rem_result.get("leakage_remediation_reasoning", "")
                    _rem_dropped = _rem_result.get("leakage_dropped_features", [])

                    # Enforce _exclude set on remediation results — the LLM may
                    # recommend metadata features (e.g. data_quality_score) that
                    # are target proxies but don't trigger single-feature AUC > 0.95
                    _force_exclude = _exclude & set(_rem_features)
                    if _force_exclude:
                        _rem_features = [f for f in _rem_features if f not in _exclude]
                        _rem_dropped = _rem_dropped + sorted(_force_exclude)
                        print(f"     Force-excluded metadata features: {sorted(_force_exclude)}")

                    _rem_duration = time.time() - _rem_start

                    print(f"\n  🔧 Remediation result: {_rem_status} ({_rem_duration:.1f}s)")
                    if _rem_dropped:
                        print(f"     Dropped: {_rem_dropped}")
                    if _rem_features:
                        print(f"     Clean features: {_rem_features}")
                    if _rem_reasoning:
                        print(f"     Reasoning: {_rem_reasoning}")

                    # Emit Step 5a result
                    step_results.append(
                        StepResult(
                            step_num="5a",
                            step_name="LEAKAGE REMEDIATION",
                            status="success" if _rem_viable else "failed",
                            duration_seconds=_rem_duration,
                            key_metrics={
                                "status": _rem_status,
                                "leaked_features_count": len(state.get("leaked_features", [])),
                                "dropped_features": _rem_dropped,
                                "clean_features": _rem_features,
                                "viable": _rem_viable,
                            },
                            input_summary={
                                "leakage_severity": state["leakage_severity"],
                                "leaked_features": state["leaked_features"],
                                "initial_feature_count": len(feature_cols),
                            },
                            validation_checks=[
                                (
                                    "Leaked features identified",
                                    bool(state.get("leaked_features")),
                                    "present",
                                    str(state.get("leaked_features", [])),
                                ),
                                (
                                    "Clean alternatives found",
                                    bool(_rem_features),
                                    "present",
                                    str(_rem_features),
                                ),
                                ("Feature set viable", _rem_viable, "True", str(_rem_viable)),
                            ],
                            interpretation=[
                                f"Leakage severity: {state['leakage_severity']}",
                                f"Dropped {len(_rem_dropped)} leaked features: {_rem_dropped}"
                                if _rem_dropped
                                else "No features dropped",
                                f"Recommended {len(_rem_features)} clean features: {_rem_features}"
                                if _rem_features
                                else "No viable replacement features found",
                                _rem_reasoning or "No reasoning provided",
                            ],
                            result_message=f"Remediated: {len(_rem_features)} clean features"
                            if _rem_viable
                            else "FAILED — no viable features",
                        )
                    )

                    if _rem_viable and _rem_features:
                        # RC2: rebuild X from the RETAINED set (discovered columns
                        # minus the genuine leaks remediation dropped), not the
                        # curated _rem_features survivor sub-list. The post-override
                        # re-check below (the safety net) re-fires if a leak slips in.
                        _step5a_leaks = (
                            set(_rem_dropped)
                            | set(state.get("leakage_dropped_features") or [])
                            | set(state.get("leaked_features") or [])
                        )
                        feature_cols = _discover_model_feature_cols(
                            eligible_df,
                            _exclude | _step5a_leaks,
                            min_non_null_frac=CONFIG.feature_min_non_null_frac,
                            max_categorical_cardinality=CONFIG.feature_max_cardinality,
                        )
                        X = eligible_df[feature_cols].copy()
                        y = eligible_df[CONFIG.target_outcome].copy()

                        # Re-encode categoricals (the original encoding was on the
                        # pre-remediation feature set which is now discarded).
                        # One-hot for the same calibration reason as the initial
                        # encode above (see _fit_categorical_onehot).
                        _cat_cols2 = [c for c in feature_cols if X[c].dtype == object]
                        if _cat_cols2:
                            X, _enc_info2 = _fit_categorical_onehot(X, _cat_cols2)
                            feature_cols = list(X.columns)
                            state["categorical_encoding"] = _enc_info2
                            print(
                                f"     One-hot re-encoded {len(_cat_cols2)} categorical "
                                f"features -> {len(_enc_info2['onehot_columns'])} columns: {_cat_cols2}"
                            )

                        # Impute numeric NaN (RWD demographics ~27% missing)
                        _nan_cols = [
                            c
                            for c in X.columns
                            if X[c].dtype.kind in "iufb" and X[c].isnull().any()
                        ]
                        if _nan_cols:
                            X[_nan_cols] = X[_nan_cols].fillna(X[_nan_cols].median())
                            print(
                                f"     Imputed NaN in {len(_nan_cols)} numeric features: {_nan_cols}"
                            )

                        # Update leakage state (re-check on the new feature set)
                        _combined2 = _pd.concat([X, y], axis=1)
                        _numeric_feats2 = [
                            c for c in feature_cols if _combined2[c].dtype.kind in "iufb"
                        ]
                        _recheck_findings = []
                        if len(_numeric_feats2) > 0 and len(_combined2) >= 30:
                            _recheck_findings.extend(
                                check_perfect_class_separation(
                                    _combined2, CONFIG.target_outcome, _numeric_feats2
                                )
                            )
                            _recheck_findings.extend(
                                check_zero_variance_within_class(
                                    _combined2, CONFIG.target_outcome, _numeric_feats2
                                )
                            )
                            _recheck_findings.extend(
                                check_mutual_information(
                                    _combined2, CONFIG.target_outcome, _numeric_feats2
                                )
                            )
                            _recheck_findings.extend(
                                check_feature_target_logical_dependency(
                                    _combined2, CONFIG.target_outcome, _numeric_feats2
                                )
                            )
                            _recheck_findings.extend(
                                check_single_feature_auc(
                                    _combined2, CONFIG.target_outcome, _numeric_feats2
                                )
                            )

                        if _recheck_findings:
                            _recheck_sev = _aggregate_severity(_recheck_findings)
                            state["leakage_severity"] = _recheck_sev
                            state["leaked_features"] = _get_leaked_features(_recheck_findings)
                            state["leakage_findings"] = [f.to_dict() for f in _recheck_findings]
                            print(
                                f"     Re-check: {_recheck_sev} severity on {len(_recheck_findings)} finding(s)"
                            )
                        else:
                            state["leakage_severity"] = "none"
                            state["leaked_features"] = []
                            state["leakage_findings"] = []
                            state["leakage_suspected"] = False
                            print(f"     ✅ Re-check: clean — no leakage in remediated features")

                        state["leakage_remediated_features"] = _rem_features
                    else:
                        # No viable features — halt pipeline
                        print(f"\n  🚨 REMEDIATION FAILED: No viable feature set found")
                        state["pipeline_halted"] = True
                        state["halt_reason"] = (
                            _rem_reasoning or "No viable features after leakage remediation"
                        )

                        step_results.append(
                            StepResult(
                                step_num=5,
                                step_name="MODEL TRAINER",
                                status="failed",
                                duration_seconds=time.time() - step_start,
                                key_metrics={"status": "skipped"},
                                interpretation=[
                                    "Training SKIPPED — no viable features after leakage remediation"
                                ],
                                result_message="Training SKIPPED — no viable features",
                            )
                        )

                except Exception as _rem_err:
                    print(f"  ⚠️  Leakage remediation error: {_rem_err}")
                    import traceback

                    traceback.print_exc()

            # Skip training if pipeline was halted by remediation
            if state.get("pipeline_halted"):
                pass  # Steps 6-8 guards will also skip
            else:
                model_candidate = state.get(
                    "model_candidate",
                    {
                        "algorithm_name": "LogisticRegression",
                        "hyperparameters": {"C": 1.0, "max_iter": 100},
                    },
                )
                qc_report = state.get("qc_report", {"gate_passed": True})

                # Resolve entity + date columns for the combined split.
                # Prefer scope_spec hints (canonical source); fall back to
                # standard synthetic schema columns (patient_journey_id /
                # journey_start_date) when present.
                _scope_spec = state.get("scope_spec", {}) or {}
                entity_col = _scope_spec.get("entity_column")
                date_col = _scope_spec.get("date_column")
                if not entity_col and "patient_journey_id" in eligible_df.columns:
                    entity_col = "patient_journey_id"
                if not date_col:
                    for _candidate in ("journey_start_date", "created_at", "index_date"):
                        if _candidate in eligible_df.columns:
                            date_col = _candidate
                            break

                _entity_ids = (
                    eligible_df.loc[X.index, entity_col]
                    if entity_col and entity_col in eligible_df.columns
                    else None
                )
                _dates = (
                    eligible_df.loc[X.index, date_col]
                    if date_col and date_col in eligible_df.columns
                    else None
                )

                result = await step_5_model_trainer(
                    experiment_id,
                    model_candidate,
                    qc_report,
                    X,
                    y,
                    success_criteria=state.get("success_criteria", {}),
                    entity_ids=_entity_ids,
                    dates=_dates,
                    split_mode=split_mode,
                    pre_assigned_splits=pre_assigned_splits,
                    cost_matrix=_scope_spec.get("cost_matrix"),
                )
                state["trained_model"] = result.get("trained_model")
                state["train_metrics"] = result.get("train_metrics", {})
                state["validation_metrics"] = result.get("validation_metrics", {})
                state["test_metrics"] = result.get("test_metrics", {})
                state["optimal_threshold"] = result.get("optimal_threshold", 0.5)
                # Persist split bookkeeping so the tier0 cache can refuse to
                # re-split on reload (Block 4, Finding #12).
                state["split_assignments"] = result.get("split_assignments", {})
                state["split_strategy"] = result.get("split_strategy")
                # Imbalance-aware evaluation fields from evaluator
                state["precision_constrained"] = result.get("precision_constrained")
                state["minority_recall"] = result.get("minority_recall")
                state["minority_precision"] = result.get("minority_precision")
                state["test_metrics_at_optimal"] = result.get("test_metrics_at_optimal", {})
                state["test_metrics_at_05"] = result.get("test_metrics_at_05", {})
                # Store feature names for downstream agents (e.g., prediction_synthesizer)
                state["feature_names"] = feature_cols
                # Store preprocessor for BentoML serving (service handles preprocessing)
                state["fitted_preprocessor"] = result.get("fitted_preprocessor")
                # Try multiple possible keys for model_uri
                state["model_uri"] = (
                    result.get("model_uri")
                    or result.get("model_artifact_uri")
                    or result.get("mlflow_model_uri")
                )
                state["success_criteria_met"] = result.get("success_criteria_met", False)
                # Hop 3 of 4 (adaptive_criteria_v3_followup): copy the
                # (possibly-overlaid) success_criteria from the agent's
                # result into runner state so the JSON artifact + deployer
                # signal see v3 active gates / regime overrides / popped
                # deprecated keys. Empty-dict guard preserves the
                # validator's pre-overlay stash when the agent (or a test
                # stub) returns no success_criteria — see
                # tests/unit/test_scripts/test_tier0_cache.py _StubAgent.
                sc_from_result = result.get("success_criteria")
                if sc_from_result:
                    state["success_criteria"] = sc_from_result
                # Hop 5 of 5 (adaptive_criteria_v3_followup): also copy the
                # per-criterion outcomes so the integration test that reads
                # ``out["success_criteria_results"]`` sees the v3 audit
                # (skipped names with met=None, NB / MCC / calibration
                # outcomes). Without this, the JSON artifact at line 5430
                # and the integration assertions at test_adaptive_criteria_e2e
                # see a stale / empty dict from scope_definer.
                state["success_criteria_results"] = result.get("success_criteria_results", {})
                state["model_usefulness"] = result.get("model_usefulness", "unknown")

                # Capture class imbalance information
                state["class_imbalance_info"] = {
                    "imbalance_detected": result.get("imbalance_detected", False),
                    "imbalance_ratio": result.get("imbalance_ratio"),
                    "minority_ratio": result.get("minority_ratio"),
                    "imbalance_severity": result.get("imbalance_severity"),
                    "class_distribution": result.get("class_distribution", {}),
                    "recommended_strategy": result.get("recommended_strategy"),
                    "strategy_rationale": result.get("strategy_rationale"),
                }

                # Capture resampling information if applied
                resampled_dist = result.get("resampled_distribution", {})
                # Calculate new minority ratio from resampled distribution
                if resampled_dist:
                    total_resampled = sum(resampled_dist.values())
                    new_minority_ratio = (
                        min(resampled_dist.values()) / total_resampled
                        if total_resampled > 0
                        else None
                    )
                else:
                    new_minority_ratio = None
                state["resampling_info"] = {
                    "resampling_applied": result.get("resampling_applied", False),
                    "original_samples": result.get("original_train_samples"),
                    "resampled_samples": result.get("resampled_train_samples"),
                    "original_distribution": result.get("original_distribution", {}),
                    "resampled_distribution": resampled_dist,
                    "new_minority_ratio": new_minority_ratio,
                    "resampling_strategy": result.get("resampling_strategy"),
                }

                # Capture enhanced accuracy analysis data
                if result.get("accuracy_analysis"):
                    state["accuracy_analysis"] = result["accuracy_analysis"]

                # Propagate post-training leakage suspicion state
                state["leakage_suspected"] = result.get("leakage_suspected", False)
                state["suspicion_level"] = result.get("suspicion_level", "none")
                state["suspicion_reasons"] = result.get("suspicion_reasons", [])
                state["investigation_recommendations"] = result.get(
                    "investigation_recommendations", []
                )

                # Propagate advanced validation results
                state["permutation_test"] = result.get("permutation_test", {})
                state["cv_results"] = result.get("cv_results", {})
                state["calibration_analysis"] = result.get("calibration_analysis", {})
                state["calibration_error"] = result.get("calibration_error")
                state["calibrated_ece"] = result.get("calibrated_ece")
                state["f1_threshold_analysis"] = result.get("f1_threshold_analysis", {})
                state["split_validation"] = result.get("split_validation", {})
                state["mcc"] = result.get("mcc")
                state["pr_auc"] = result.get("pr_auc")

                # Determine step status based on both AUC and model usefulness
                model_usefulness = result.get("model_usefulness", "unknown")
                if model_usefulness == "useless":
                    step_5_status = "failed"
                elif model_usefulness == "poor" or not result.get("success_criteria_met"):
                    step_5_status = "warning"
                else:
                    step_5_status = "success"

                auc_roc = result.get("auc_roc", 0)
                precision = result.get("precision", 0)
                recall = result.get("recall", 0)
                f1 = result.get("f1_score", 0)
                success_met = result.get("success_criteria_met", False)
                imbalance_detected = result.get("imbalance_detected", False)
                resampling_applied = result.get("resampling_applied", False)
                step_results.append(
                    StepResult(
                        step_num=5,
                        step_name="MODEL TRAINER",
                        status=step_5_status,
                        duration_seconds=time.time() - step_start,
                        key_metrics={
                            "training_run_id": result.get("training_run_id"),
                            "model_id": result.get("model_id"),
                            "auc_roc": auc_roc,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1,
                            "success_criteria_met": success_met,
                            "hpo_trials_run": result.get("hpo_trials_run"),
                            "model_usefulness": model_usefulness,
                        },
                        details={
                            "mlflow_run_id": result.get("mlflow_run_id"),
                            "model_uri": state.get("model_uri"),
                            "training_duration_seconds": result.get("training_duration_seconds"),
                            "imbalance_detected": imbalance_detected,
                            "imbalance_severity": result.get("imbalance_severity"),
                            "remediation_strategy": result.get("recommended_strategy"),
                            "usefulness_reason": result.get("usefulness_reason"),
                        },
                        # Enhanced format fields
                        input_summary={
                            "experiment_id": experiment_id,
                            "algorithm": model_candidate.get("algorithm_name")
                            if isinstance(model_candidate, dict)
                            else "Unknown",
                            "training_samples": len(X),
                            "features": list(X.columns),
                            "target": CONFIG.target_outcome,
                        },
                        validation_checks=[
                            (
                                "AUC-ROC above threshold",
                                auc_roc >= 0.6,
                                "≥ 0.60",
                                f"{auc_roc:.3f}" if auc_roc else "N/A",
                            ),
                            (
                                "Model not useless",
                                model_usefulness != "useless",
                                "not useless",
                                model_usefulness,
                            ),
                            ("Success criteria met", success_met, "True", str(success_met)),
                            (
                                "Both classes predicted",
                                model_usefulness not in ["useless", "poor"],
                                "multi-class output",
                                model_usefulness,
                            ),
                        ],
                        metrics_table=[
                            (
                                "auc_roc",
                                f"{auc_roc:.3f}" if auc_roc else "N/A",
                                "≥ 0.60",
                                auc_roc >= 0.6 if auc_roc else False,
                            ),
                            ("precision", f"{precision:.3f}" if precision else "N/A", None, None),
                            ("recall", f"{recall:.3f}" if recall else "N/A", None, None),
                            ("f1_score", f"{f1:.3f}" if f1 else "N/A", None, None),
                            (
                                "model_usefulness",
                                model_usefulness,
                                "good/acceptable",
                                model_usefulness in ["good", "acceptable"],
                            ),
                            ("imbalance_detected", imbalance_detected, None, None),
                            ("resampling_applied", resampling_applied, None, None),
                        ],
                        interpretation=[
                            f"Model trained with AUC-ROC: {auc_roc:.3f}"
                            if auc_roc
                            else "Model training completed",
                            f"Model usefulness: {model_usefulness}"
                            + (
                                f" - {result.get('usefulness_reason', '')}"
                                if result.get("usefulness_reason")
                                else ""
                            ),
                            f"Class imbalance {'detected and remediated via ' + result.get('recommended_strategy', 'resampling') if imbalance_detected else 'not detected'}",
                            f"Success criteria {'MET' if success_met else 'NOT MET'}",
                        ],
                        result_message=f"Training complete: {model_usefulness} model with AUC={auc_roc:.3f}"
                        if auc_roc
                        else "Training complete",
                    )
                )

                # ================================================================
                # Step 5b: Algorithm Comparison — always train alternatives
                # ================================================================
                # Train all alternative candidates from model_selector and pick
                # the best model by test AUC. This runs unconditionally so the
                # pipeline always compares algorithms, not just when the primary
                # model is "poor".
                # Cache primary result for comparison
                candidate_results = {}  # algorithm_name -> full result dict
                comparison_history = []
                primary_algo = (
                    state.get("model_candidate", {}).get("algorithm_name", "")
                    if isinstance(state.get("model_candidate"), dict)
                    else ""
                )
                candidate_results[primary_algo] = result
                # Entry carries the candidate's EVALUATED deployability gate
                # outcomes for _select_champion (see _candidate_history_entry).
                comparison_history.append(
                    _candidate_history_entry(
                        primary_algo,
                        result,
                        is_primary=True,
                        model_usefulness=model_usefulness,
                    )
                )

                alternatives = list(state.get("alternative_candidates", []))
                # Tier D memory lever (--single-model): skip the champion-comparison
                # alternative training. The champion is then the primary model. This
                # avoids holding multiple trained models + their bootstrap arrays at
                # once — the peak that OOMs on a memory-constrained host.
                if not CONFIG.train_alternatives and alternatives:
                    print(
                        f"  Step 5b: single-model mode — skipping {len(alternatives)} "
                        f"alternative candidate(s); champion = primary ({primary_algo})"
                    )
                    alternatives = []
                if alternatives and not state.get("pipeline_halted"):
                    from src.agents.ml_foundation.model_selector.nodes.algorithm_registry import (
                        REGULARIZATION_SEARCH_SPACE,
                    )
                    from src.agents.ml_foundation.model_selector.nodes.candidate_ranker import (
                        _get_algorithm_class,
                    )

                    # Issue #232: import the shared LR-family fixed-params helper
                    # so the Step 5b alt-train builder can't drift from the HPO
                    # dispatcher (both consume the same constant).
                    from src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner import (
                        _LR_FIXED_PARAMS,
                    )
                    from src.mlops.lr_solver_policy import reconcile_lr_solver

                    imb_sev = state.get("class_imbalance_info", {}).get("imbalance_severity")

                    for alt in alternatives:
                        if alt.get("name") == primary_algo:
                            continue  # Skip if same as primary

                        new_candidate = {
                            "algorithm_name": alt["name"],
                            "algorithm_class": _get_algorithm_class(alt),
                            "hyperparameter_search_space": {
                                **alt.get("hyperparameter_space", {}),
                                **REGULARIZATION_SEARCH_SPACE.get(alt["name"], {}),
                            },
                            "default_hyperparameters": alt.get("default_hyperparameters", {}),
                        }
                        # Propagate registry flags consumed by downstream evaluator gating.
                        # Without this, calibration-native + conformal-wrapped algorithms
                        # (NGBoost*, *_Conformal) re-enter the post-hoc isotonic path and
                        # crash predict_proba on the regressor-shaped FrozenEstimator.
                        for _flag in (
                            "skip_post_hoc_calibration",
                            "distribution_predictor",
                            "conformal_wrapper",
                            "base_estimator",
                        ):
                            if _flag in alt:
                                new_candidate[_flag] = alt[_flag]
                        # Issue #232: ALWAYS pin solver=saga + max_iter=1000 for
                        # LR-family alts (incl. LogisticRegression_Conformal) so
                        # an HPO best_params with penalty="l1" doesn't crash the
                        # alt-train builder with
                        # ``Solver lbfgs supports only 'l2' or None penalties``.
                        # The shared `_LR_FIXED_PARAMS` is the single source of
                        # truth — same constant the HPO dispatcher consumes.
                        if alt["name"] in (
                            "LogisticRegression",
                            "LogisticRegression_Conformal",
                        ):
                            new_candidate["default_hyperparameters"] = {
                                **new_candidate["default_hyperparameters"],
                                **_LR_FIXED_PARAMS,
                            }
                            # Issue #232 runtime: the candidate's penalty is known
                            # here, so downgrade the saga floor to lbfgs for l2/None
                            # (identical AUC, ~20 iters vs 1000). saga retained for
                            # l1 — see tier0_optum_mart..._disproof_20260606.md.
                            reconcile_lr_solver(new_candidate["default_hyperparameters"])
                        # Force class_weight for imbalanced tree/linear models
                        if imb_sev in ("severe", "extreme") and alt["name"] in (
                            "RandomForest",
                            "LogisticRegression",
                            "LogisticRegression_Conformal",
                        ):
                            new_candidate["default_hyperparameters"] = {
                                **new_candidate["default_hyperparameters"],
                                "class_weight": "balanced",
                            }

                        print(f"\n  {'=' * 60}")
                        print(f"  Step 5b: Algorithm comparison: training {alt['name']}")
                        print(f"  {'=' * 60}")

                        # Reuse the same split mapping from the primary run so
                        # comparison candidates train on identical data partitions.
                        _alt_pre_splits = state.get("split_assignments") or pre_assigned_splits
                        alt_result = await step_5_model_trainer(
                            experiment_id,
                            new_candidate,
                            qc_report,
                            X,
                            y,
                            success_criteria=state.get("success_criteria", {}),
                            entity_ids=_entity_ids,
                            dates=_dates,
                            split_mode=split_mode,
                            pre_assigned_splits=_alt_pre_splits,
                            cost_matrix=_scope_spec.get("cost_matrix"),
                        )

                        candidate_results[alt["name"]] = alt_result
                        comparison_history.append(
                            _candidate_history_entry(alt["name"], alt_result, is_primary=False)
                        )

                    # Pick best model across all candidates: highest AUC, with a
                    # calibration-aware tiebreak among discrimination-ties so a
                    # better-calibrated model wins over an AUC-equal but
                    # worse-calibrated one (#633; see _select_champion).
                    best = _select_champion(comparison_history)
                    if not best.get("is_primary"):
                        # An alternative won — swap it into state from cache
                        _winner = candidate_results[best["algorithm"]]
                        state["trained_model"] = _winner.get("trained_model")
                        state["train_metrics"] = _winner.get("train_metrics", {})
                        state["validation_metrics"] = _winner.get("validation_metrics", {})
                        state["test_metrics"] = _winner.get("test_metrics", {})
                        state["optimal_threshold"] = _winner.get("optimal_threshold", 0.5)
                        state["precision_constrained"] = _winner.get("precision_constrained")
                        state["minority_recall"] = _winner.get("minority_recall")
                        state["minority_precision"] = _winner.get("minority_precision")
                        state["model_usefulness"] = _winner.get("model_usefulness", "unknown")
                        state["success_criteria_met"] = _winner.get("success_criteria_met", False)
                        # Edit 9 (adaptive_criteria_v3_followup): when an
                        # alternative model wins Step 5b, also propagate its
                        # success_criteria + success_criteria_results so the
                        # downstream artifact + integration assertions see the
                        # winner's pass/fail outcomes (otherwise state has
                        # winner's test_metrics but primary's check results,
                        # producing artifacts where actual AUC > threshold yet
                        # minimum_auc=False).
                        if "success_criteria" in _winner:
                            state["success_criteria"] = _winner["success_criteria"]
                        state["success_criteria_results"] = _winner.get(
                            "success_criteria_results", {}
                        )
                        state["fitted_preprocessor"] = _winner.get("fitted_preprocessor")
                        state["model_uri"] = (
                            _winner.get("model_uri")
                            or _winner.get("model_artifact_uri")
                            or _winner.get("mlflow_model_uri")
                        )
                        state["model_candidate"] = {"algorithm_name": best["algorithm"]}
                        # Propagate advanced validation from winner
                        for _key in (
                            "permutation_test",
                            "cv_results",
                            "calibration_analysis",
                            "calibration_error",
                            "calibrated_ece",
                            "f1_threshold_analysis",
                            "split_validation",
                            "mcc",
                            "pr_auc",
                            "leakage_suspected",
                            "suspicion_level",
                            "suspicion_reasons",
                            "investigation_recommendations",
                        ):
                            if _key in _winner:
                                state[_key] = _winner[_key]
                        if _winner.get("accuracy_analysis"):
                            state["accuracy_analysis"] = _winner["accuracy_analysis"]

                # Emit Step 5b comparison result
                if len(comparison_history) > 1:
                    # Mirror the actual champion selection (calibration-aware
                    # tiebreak among discrimination-ties) so the emitted winner
                    # matches the candidate swapped into state above.
                    _best = _select_champion(comparison_history)
                    _sorted = sorted(comparison_history, key=lambda h: h["auc_roc"], reverse=True)
                    _ranking = ", ".join(f"{h['algorithm']}={h['auc_roc']:.3f}" for h in _sorted)
                    step_results.append(
                        StepResult(
                            step_num="5b",
                            step_name="ALGORITHM COMPARISON",
                            status="success",
                            duration_seconds=0.0,
                            key_metrics={
                                "candidates_trained": len(comparison_history),
                                "best_algorithm": _best["algorithm"],
                                "best_auc": _best["auc_roc"],
                                "winner_is_primary": _best.get("is_primary", False),
                            },
                            details={"comparison_history": comparison_history},
                            input_summary={
                                "candidates": [h["algorithm"] for h in comparison_history]
                            },
                            validation_checks=[
                                (
                                    "Multiple algorithms compared",
                                    len(comparison_history) > 1,
                                    "> 1",
                                    f"{len(comparison_history)} trained",
                                ),
                                (
                                    "Best model selected",
                                    True,
                                    "highest AUC",
                                    f"{_best['algorithm']} ({_best['auc_roc']:.3f})",
                                ),
                            ],
                            metrics_table=[
                                ("candidates_trained", len(comparison_history), None, None),
                                ("best_algorithm", _best["algorithm"], None, None),
                                (
                                    "best_auc",
                                    f"{_best['auc_roc']:.3f}",
                                    ">= 0.60",
                                    _best["auc_roc"] >= 0.60,
                                ),
                                ("ranking", _ranking, None, None),
                            ],
                            interpretation=[
                                f"Trained {len(comparison_history)} candidate algorithms: {', '.join(h['algorithm'] for h in comparison_history)}",
                                f"Ranking by AUC: {_ranking}",
                                # Deployability gate evidence per candidate (the
                                # pool _select_champion partitions on).
                                "Deployability gates: "
                                + "; ".join(
                                    f"{h['algorithm']}(overfit_gate={h.get('overfit_gate_met')}, "
                                    f"slope_gate={h.get('slope_gate_met')}, "
                                    f"slope_dev={h.get('calibration_slope_deviation'):.4f}, "
                                    f"severity={h.get('overfitting_severity')})"
                                    for h in comparison_history
                                ),
                                f"Selected {_best['algorithm']} as best model (AUC={_best['auc_roc']:.3f})",
                            ],
                            result_message=f"Algorithm comparison: {_best['algorithm']} wins ({_ranking})",
                        )
                    )

        # Step 6: Feature Analyzer
        if 6 in steps_to_run and not state.get("pipeline_halted"):
            step_start = time.time()
            eligible_df = state.get("eligible_df", patient_df)
            # Use whatever features the model was trained on
            feature_cols = state.get(
                "feature_names", ["days_on_therapy", "hcp_visits", "prior_treatments"]
            )
            # feature_names may be the one-hot-EXPANDED columns (e.g. payer_HMO),
            # which do NOT exist in the raw eligible_df. Select the ORIGINAL
            # pre-encode columns (numeric + raw categorical names) so the frame
            # can be rebuilt and re-encoded to the trained feature space.
            cat_enc = state.get("categorical_encoding")
            _raw_cols = _raw_feature_cols(feature_cols, cat_enc)
            X = eligible_df[[c for c in _raw_cols if c in eligible_df.columns]].copy()
            y = eligible_df[CONFIG.target_outcome].copy()

            # Apply fitted preprocessor so SHAP receives the same feature space the model expects
            X_for_shap = X.iloc[:50]

            # Re-apply the one-hot encoding used before Step 5 so SHAP sees the
            # exact feature space the model trained on (the preprocessor was
            # fitted on the encoded data, not raw strings). One-hot CHANGES the
            # column count, so this rebuilds the expanded frame via the fitted
            # encoder (see _apply_categorical_onehot).
            if cat_enc and cat_enc.get("columns"):
                X_for_shap = _apply_categorical_onehot(X_for_shap.copy(), cat_enc)

            fitted_preprocessor = state.get("fitted_preprocessor")
            if fitted_preprocessor is not None:
                try:
                    X_transformed = fitted_preprocessor.transform(X_for_shap)
                    # Check the ATTRIBUTE (sklearn convention uses trailing underscore)
                    feature_names_out = getattr(fitted_preprocessor, "feature_names_out_", None)
                    if (
                        feature_names_out is not None
                        and len(feature_names_out) == X_transformed.shape[1]
                    ):
                        X_for_shap = pd.DataFrame(
                            X_transformed, columns=feature_names_out, index=X_for_shap.index
                        )
                    else:
                        # Fallback: use the (already one-hot-expanded) input column
                        # names if the transform preserved the count (e.g. a pure
                        # scaler over the one-hot frame).
                        original_cols = list(X_for_shap.columns)
                        if len(original_cols) == X_transformed.shape[1]:
                            X_for_shap = pd.DataFrame(
                                X_transformed, columns=original_cols, index=X_for_shap.index
                            )
                        else:
                            X_for_shap = pd.DataFrame(X_transformed, index=X_for_shap.index)
                except Exception as e:
                    print(f"  ⚠ Preprocessor transform failed, using raw features: {e}")

            result = await step_6_feature_analyzer(
                experiment_id,
                state.get("trained_model"),
                X_for_shap,
                y.iloc[:50],
                model_uri=state.get("model_uri"),
            )
            state["feature_importance"] = result.get("feature_importance")

            # Extract top features for summary
            top_features = {}
            if result.get("feature_importance"):
                for fi in result["feature_importance"][:5]:
                    if isinstance(fi, dict):
                        top_features[fi.get("feature", "unknown")] = fi.get("importance", 0)

            samples_analyzed = result.get("samples_analyzed", 0)
            compute_time = result.get("computation_time_seconds", 0)
            explainer_type = result.get("explainer_type", "SHAP")
            top_feature_name = list(top_features.keys())[0] if top_features else None
            top_feature_importance = (
                top_features.get(top_feature_name, 0) if top_feature_name else 0
            )
            step_results.append(
                StepResult(
                    step_num=6,
                    step_name="FEATURE ANALYZER",
                    status="success" if result.get("feature_importance") else "warning",
                    duration_seconds=time.time() - step_start,
                    key_metrics={
                        "samples_analyzed": samples_analyzed,
                        "computation_time": compute_time,
                        "top_feature": top_feature_name,
                    },
                    details={
                        "top_features": top_features,
                        "explainer_type": explainer_type,
                    },
                    # Enhanced format fields
                    input_summary={
                        "experiment_id": experiment_id,
                        "model_uri": state.get("model_uri", "N/A"),
                        "samples_for_analysis": 50,
                        "features_analyzed": feature_cols,
                    },
                    validation_checks=[
                        (
                            "Feature importance computed",
                            result.get("feature_importance") is not None,
                            "importance present",
                            "Yes" if result.get("feature_importance") else "No",
                        ),
                        (
                            "Sufficient samples analyzed",
                            samples_analyzed >= 10,
                            "≥ 10",
                            samples_analyzed,
                        ),
                        (
                            "Computation completed",
                            compute_time > 0,
                            "compute_time > 0",
                            f"{compute_time:.2f}s",
                        ),
                    ],
                    metrics_table=[
                        ("samples_analyzed", samples_analyzed, "≥ 10", samples_analyzed >= 10),
                        ("computation_time", f"{compute_time:.2f}s", None, None),
                        ("explainer_type", explainer_type, None, None),
                        ("top_feature", top_feature_name or "N/A", None, None),
                        (
                            "top_importance",
                            f"{top_feature_importance:.3f}" if top_feature_importance else "N/A",
                            None,
                            None,
                        ),
                    ],
                    interpretation=[
                        f"SHAP analysis completed on {samples_analyzed} samples in {compute_time:.2f}s",
                        f"Top driver: {top_feature_name} (importance: {top_feature_importance:.3f})"
                        if top_feature_name
                        else "No dominant feature identified",
                        f"Feature ranking: {', '.join(str(k) for k in list(top_features.keys())[:3])}"
                        if len(top_features) >= 3
                        else f"Features analyzed: {len(top_features)}",
                    ],
                    result_message=f"Feature analysis complete: {top_feature_name} is top predictor"
                    if top_feature_name
                    else "Feature analysis complete",
                )
            )

        # Step 7: Model Deployer
        if 7 in steps_to_run and not state.get("pipeline_halted"):
            step_start = time.time()

            # Block deployment if model quality is too poor
            _model_usefulness = state.get("model_usefulness", "unknown")
            _success_criteria_met = state.get("success_criteria_met", False)
            if _model_usefulness in ("useless", "poor") or not _success_criteria_met:
                _reason_parts = []
                if _model_usefulness in ("useless", "poor"):
                    _reason_parts.append(f"model_usefulness={_model_usefulness}")
                if not _success_criteria_met:
                    _reason_parts.append("success_criteria_not_met")
                _quality_reason = ", ".join(_reason_parts)
                print(f"\n  🚨 DEPLOYMENT BLOCKED: Model quality insufficient")
                print(f"     Model usefulness: {_model_usefulness}")
                print(f"     Success criteria met: {_success_criteria_met}")
                step_results.append(
                    StepResult(
                        step_num=7,
                        step_name="MODEL DEPLOYER",
                        status="failed",
                        duration_seconds=time.time() - step_start,
                        key_metrics={
                            "deployment_id": "BLOCKED",
                            "environment": "N/A",
                            "status": "blocked_quality",
                            "deployment_successful": False,
                        },
                        details={"reason": f"Model quality insufficient — {_quality_reason}"},
                        input_summary={
                            "model_usefulness": _model_usefulness,
                            "success_criteria_met": _success_criteria_met,
                        },
                        validation_checks=[
                            ("Model quality gate", False, "acceptable+", f"{_model_usefulness}"),
                            (
                                "Success criteria",
                                _success_criteria_met,
                                "met",
                                str(_success_criteria_met),
                            ),
                        ],
                        metrics_table=[],
                        interpretation=[f"Deployment blocked: {_quality_reason}"],
                        result_message=f"Deployment BLOCKED — {_quality_reason}",
                    )
                )
            # Block deployment if leakage suspected. The leakage-gate predicate is
            # extracted so it is unit-testable and so the FU1 scenario-regime routing
            # (pre-training heuristics → diagnostics; post-training empirical signal
            # stays live) is exercised by the same condition prod uses.
            elif _deploy_blocked_by_leakage(state):
                _leakage_severity = state.get("leakage_severity", "none")
                _suspicion_level = state.get("suspicion_level", "none")
                print(f"\n  🚨 DEPLOYMENT BLOCKED: Data leakage detected")
                print(f"     Leakage severity: {_leakage_severity}")
                print(f"     Suspicion level: {_suspicion_level}")
                print(f"     Leaked features: {state.get('leaked_features', [])}")
                step_results.append(
                    StepResult(
                        step_num=7,
                        step_name="MODEL DEPLOYER",
                        status="failed",
                        duration_seconds=time.time() - step_start,
                        key_metrics={
                            "deployment_id": "BLOCKED",
                            "environment": "N/A",
                            "status": "blocked_leakage",
                            "deployment_successful": False,
                        },
                        details={"reason": "Data leakage detected — deployment blocked"},
                        input_summary={
                            "leakage_severity": _leakage_severity,
                            "suspicion_level": _suspicion_level,
                        },
                        validation_checks=[
                            ("Leakage check", False, "no leakage", f"severity={_leakage_severity}"),
                        ],
                        metrics_table=[],
                        interpretation=["Deployment blocked due to data leakage detection"],
                        result_message="Deployment BLOCKED — data leakage suspected",
                    )
                )
            else:
                result = await step_7_model_deployer(
                    experiment_id,
                    state.get("model_uri"),
                    state.get("validation_metrics", {}),
                    state.get("success_criteria_met", False),
                    trained_model=state.get("trained_model"),
                    include_bentoml=include_bentoml,
                    fitted_preprocessor=state.get("fitted_preprocessor"),
                    feature_columns=state.get("feature_names"),
                    scope_spec=state.get("scope_spec"),
                )
                state["deployment_manifest"] = result.get("deployment_manifest")
                # v5 Gate C1 (2026-05-11): surface the regulatory
                # deployment manifest from the deployer agent's output
                # onto runner state. validate_promotion populates this
                # field; the runner persists it on state so the
                # TIER0_E2E_JSON_OUT artifact carries the payload that
                # the v5 C1 integration test pins.
                state["regulatory_deployment_manifest"] = result.get(
                    "regulatory_deployment_manifest"
                )
                # Track BentoML PID for cleanup (ephemeral mode only)
                if include_bentoml and result.get("bentoml_pid"):
                    state["bentoml_pid"] = result["bentoml_pid"]
                # Track persistent mode to skip cleanup
                if include_bentoml and result.get("bentoml_persistent"):
                    state["bentoml_persistent"] = True

                manifest = result.get("deployment_manifest", {})
                bentoml_serving = result.get("bentoml_serving", {})
                step_details = {
                    "model_version": result.get("model_version"),
                    "endpoint_url": manifest.get("endpoint_url"),
                }
                # Add BentoML info if present
                if bentoml_serving:
                    step_details["bentoml_model_tag"] = bentoml_serving.get("model_tag")
                    step_details["bentoml_endpoint"] = bentoml_serving.get("endpoint")
                    step_details["bentoml_health_check"] = bentoml_serving.get("health_check")
                    step_details["bentoml_prediction_test"] = bentoml_serving.get("prediction_test")
                    step_details["bentoml_latency_ms"] = bentoml_serving.get("latency_ms")

                deployment_id = manifest.get("deployment_id", "N/A")
                environment = manifest.get("environment", "staging")
                deployment_status = manifest.get("status", "unknown")
                deployment_successful = result.get("deployment_successful", False)
                bentoml_verified = (
                    bentoml_serving.get("prediction_test", False) if bentoml_serving else None
                )
                # Overall step success requires both the deploy flag AND the validation checks to hold.
                _deployment_id_valid = bool(deployment_id) and deployment_id != "N/A"
                _status_healthy = deployment_status == "healthy"
                _all_checks_pass = (
                    deployment_successful and _deployment_id_valid and _status_healthy
                )
                step_results.append(
                    StepResult(
                        step_num=7,
                        step_name="MODEL DEPLOYER",
                        status="success" if _all_checks_pass else "warning",
                        duration_seconds=time.time() - step_start,
                        key_metrics={
                            "deployment_id": deployment_id,
                            "environment": environment,
                            "status": deployment_status,
                            "deployment_successful": deployment_successful,
                            "bentoml_verified": bentoml_verified,
                        },
                        details=step_details,
                        # Enhanced format fields
                        input_summary={
                            "experiment_id": experiment_id,
                            "model_uri": state.get("model_uri", "N/A"),
                            "validation_metrics": state.get("validation_metrics", {}),
                            "success_criteria_met": state.get("success_criteria_met", False),
                            "include_bentoml": include_bentoml,
                        },
                        validation_checks=[
                            (
                                "Deployment successful",
                                deployment_successful,
                                "True",
                                str(deployment_successful),
                            ),
                            (
                                "Deployment ID assigned",
                                _deployment_id_valid,
                                "ID present",
                                deployment_id,
                            ),
                            (
                                "Environment set",
                                environment is not None,
                                "env specified",
                                environment,
                            ),
                            (
                                "BentoML verified",
                                bentoml_verified if include_bentoml else True,
                                "True",
                                str(bentoml_verified) if include_bentoml else "N/A (not enabled)",
                            ),
                        ],
                        metrics_table=[
                            ("deployment_id", deployment_id, None, None),
                            ("environment", environment, None, None),
                            ("status", deployment_status, "healthy", _status_healthy),
                            (
                                "bentoml_verified",
                                str(bentoml_verified) if include_bentoml else "N/A",
                                None,
                                None,
                            ),
                            (
                                "latency_ms",
                                f"{bentoml_serving.get('latency_ms', 'N/A')}"
                                if bentoml_serving
                                else "N/A",
                                None,
                                None,
                            ),
                        ],
                        interpretation=[
                            f"Model deployed to {environment} environment"
                            if deployment_successful
                            else "Deployment pending or failed",
                            f"Deployment ID: {deployment_id}",
                            f"BentoML serving {'verified with live prediction test' if bentoml_verified else 'not verified' if include_bentoml else 'not enabled'}",
                        ],
                        result_message=f"Deployment complete: {deployment_id} to {environment}"
                        if deployment_successful
                        else "Deployment incomplete",
                    )
                )

        # Step 8: Observability Connector
        if 8 in steps_to_run and not state.get("pipeline_halted"):
            step_start = time.time()
            result = await step_8_observability_connector(experiment_id, len(steps_to_run))
            emission_successful = result.get("emission_successful", False)
            events_logged = result.get("events_logged", 0)
            quality_score = result.get("quality_score", 0)
            step_results.append(
                StepResult(
                    step_num=8,
                    step_name="OBSERVABILITY CONNECTOR",
                    status="success" if emission_successful else "warning",
                    duration_seconds=time.time() - step_start,
                    key_metrics={
                        "emission_successful": emission_successful,
                        "events_logged": events_logged,
                        "quality_score": quality_score,
                    },
                    details={},
                    # Enhanced format fields
                    input_summary={
                        "experiment_id": experiment_id,
                        "total_steps": len(steps_to_run),
                        "pipeline_complete": True,
                    },
                    validation_checks=[
                        ("Metrics emitted", emission_successful, "True", str(emission_successful)),
                        ("Events logged", events_logged > 0, "> 0", events_logged),
                        (
                            "Quality score computed",
                            quality_score is not None,
                            "present",
                            f"{quality_score:.2f}" if quality_score else "N/A",
                        ),
                        (
                            "Feast online retrieval",
                            result.get("feast_online_ok", False),
                            "accessible",
                            result.get("feast_online_detail", "skipped"),
                        ),
                    ],
                    metrics_table=[
                        (
                            "emission_successful",
                            str(emission_successful),
                            "True",
                            emission_successful,
                        ),
                        ("events_logged", events_logged, "> 0", events_logged > 0),
                        (
                            "quality_score",
                            f"{quality_score:.2f}" if quality_score else "N/A",
                            None,
                            None,
                        ),
                    ],
                    interpretation=[
                        f"Observability metrics {'successfully' if emission_successful else 'NOT'} emitted to monitoring systems",
                        f"{events_logged} events logged for pipeline tracking",
                        f"Overall pipeline quality score: {quality_score:.2f}"
                        if quality_score
                        else "Quality score not computed",
                    ],
                    result_message=f"Observability complete: {events_logged} events logged"
                    if emission_successful
                    else "Observability emission incomplete",
                )
            )

        # Print detailed step results
        print_detailed_summary(experiment_id, step_results, state)

        # Final summary
        pipeline_duration = time.time() - pipeline_start_time

        # Determine overall pipeline status
        failed_steps = [r for r in step_results if r.status == "failed"]
        warning_steps = [r for r in step_results if r.status == "warning"]

        print(f"\n{'=' * 70}")
        print("PIPELINE SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Experiment ID: {experiment_id}")
        print(f"  Steps Completed: {len(steps_to_run)}")
        print(f"  Total Duration: {pipeline_duration:.1f}s")
        print(f"  QC Gate: {'PASSED' if state.get('gate_passed', True) else 'FAILED'}")
        if state.get("eligible_df") is not None:
            print(f"  Cohort Size: {len(state['eligible_df'])}")
        # Feature Pipeline section
        feat_chars = state.get("feature_characteristics")
        cat_enc = state.get("categorical_encoding")
        leaked = state.get("leaked_features", [])
        leakage_sev = state.get("leakage_severity")
        remediated = state.get("leakage_remediated_features")
        leakage_dropped = state.get("leakage_dropped_features")

        has_feature_info = feat_chars or cat_enc or leaked or remediated
        if has_feature_info:
            print(f"\n  Feature Pipeline:")
            if feat_chars:
                n_num = feat_chars.get("num_numeric", 0)
                n_cat = feat_chars.get("num_categorical", 0)
                cat_ratio = feat_chars.get("categorical_ratio", 0.0)
                print(
                    f"    Features Discovered: {n_num + n_cat} ({n_num} numeric, {n_cat} categorical)"
                )
                print(f"    Categorical Ratio: {cat_ratio:.2f}")
            if cat_enc:
                enc_cols = cat_enc.get("columns", [])
                print(f"    Categorical Encoded: {len(enc_cols)} ({', '.join(enc_cols)})")
            if leaked:
                print(
                    f"    Leakage Detected: {len(leaked)} features (severity: {leakage_sev or 'unknown'})"
                )
                print(f"    Leaked: {', '.join(sorted(leaked))}")
            if leakage_dropped:
                print(f"    Dropped: {', '.join(sorted(leakage_dropped))}")
            if remediated:
                print(f"    Clean Features: {len(remediated)} ({', '.join(remediated)})")

        if state.get("validation_metrics"):
            print(f"  Validation Metrics: {state['validation_metrics']}")
        if include_bentoml and state.get("bentoml_persistent"):
            print(f"  BentoML Serving: Verified (Docker: {BENTOML_DOCKER_ENDPOINT})")
        print(f"  Completed: {datetime.now().isoformat()}")

        # Print step status summary
        success_count = len([r for r in step_results if r.status == "success"])
        print(
            f"\n  Step Status: {success_count} success, {len(warning_steps)} warnings, {len(failed_steps)} failed"
        )

        if failed_steps:
            print_failure(f"PIPELINE FAILED - {len(failed_steps)} step(s) failed:")
            for step in failed_steps:
                print(f"    • Step {step.step_num} ({step.step_name})")
        elif warning_steps:
            print_warning(f"Pipeline completed with {len(warning_steps)} warning(s)")
        else:
            print_success("Pipeline completed successfully!")

    except Exception as e:
        print_failure(f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        # Cleanup: Docker BentoML is persistent — no PID to clean up
        if state.get("bentoml_persistent"):
            print(f"\n  Docker BentoML service left running ({BENTOML_DOCKER_ENDPOINT})")

    # Test-fixture artifact (task 06 of adaptive_success_criteria plan).
    # When TIER0_E2E_JSON_OUT is set, dump a structured JSON of the key
    # outcomes for the integration tests to consume. Production runs do
    # not set this env var, so behavior is unchanged for normal CLI
    # invocations. Audit fields starting with ``_`` are filtered out of
    # the success_criteria payload (they are evaluator-internal).
    e2e_out = os.environ.get("TIER0_E2E_JSON_OUT")
    if e2e_out:
        import json

        sc_state = state.get("success_criteria") or {}
        # Codex pass-1 HIGH-2 (PR #137 v4 G1): canonical deployer
        # verdict label, computed via _compute_verdict using the same
        # inputs as print_detailed_summary (test_metrics primary,
        # validation_metrics fallback for early-halt runs). One of
        # {EXCELLENT, GOOD, ACCEPTABLE, THRESHOLD_NEEDED, MARGINAL,
        # POOR, LEAKAGE_SUSPECTED}, or None when metrics absent.
        _eval_metrics = state.get("test_metrics") or state.get("validation_metrics") or {}
        _deployer_verdict: Optional[str] = None
        _deployer_description: Optional[str] = None
        _deployer_recommendation: Optional[str] = None
        _auc_roc = _eval_metrics.get("roc_auc")
        _recall = _eval_metrics.get("recall")
        _precision = _eval_metrics.get("precision")
        if (
            isinstance(_auc_roc, (int, float))
            and isinstance(_recall, (int, float))
            and isinstance(_precision, (int, float))
        ):
            _train_auc = (state.get("train_metrics") or {}).get("roc_auc")
            _val_auc = (state.get("validation_metrics") or {}).get("roc_auc")
            _train_val_delta = (
                float(_train_auc) - float(_val_auc)
                if isinstance(_train_auc, (int, float)) and isinstance(_val_auc, (int, float))
                else None
            )
            _v, _icon, _desc, _rec = _compute_verdict(
                auc_roc=float(_auc_roc),
                recall=float(_recall),
                precision=float(_precision),
                overfitting_severity=state.get("overfitting_severity"),
                train_val_delta=_train_val_delta,
            )
            _deployer_verdict = _v
            _deployer_description = _desc
            _deployer_recommendation = _rec
        # Codex pass-1 HIGH-3 (PR #137 v4 G1): cohort_size pins the
        # assembled cohort row count so G1 integration tests can assert
        # n=9607 (CSU) / n=1294 (Optum default) without re-running
        # cohort-build. Sourced from state["eligible_df"] when present
        # (cohort-constructor's output), else state["patient_df"]; None
        # if neither.
        _cohort_size: Optional[int] = None
        _eligible = state.get("eligible_df")
        if _eligible is not None and hasattr(_eligible, "__len__"):
            try:
                _cohort_size = int(len(_eligible))
            except Exception:
                _cohort_size = None
        if _cohort_size is None:
            _patient_df = state.get("patient_df")
            if _patient_df is not None and hasattr(_patient_df, "__len__"):
                try:
                    _cohort_size = int(len(_patient_df))
                except Exception:
                    _cohort_size = None
        artifact = {
            "regime": regime,
            "seed": seed,
            # Closes ultrareview bug_002 (backlog #21.4): n_total flag is
            # ignored for legacy regimes (default/adverse/clean) which run
            # the legacy ml_patients() generator hardcoded to n_samples=1500.
            # Recording the user-supplied --n-total alongside a 1500-row
            # patient_df was misleading metadata-vs-data divergence.
            #
            # Codex-rescue pass-2 M2 (2026-05-09): inverted the gate — the
            # artifact records n_total UNLESS the regime is explicitly in
            # _LEGACY_REGIMES. Future synthetic_v2 regimes added without
            # updating _LEGACY_REGIMES inherit the correct (recorded) behavior;
            # only a NEW legacy ml_patients()-style regime would need to be
            # added to the legacy set.
            "n_total": n_total if regime not in _LEGACY_REGIMES else None,
            "criteria_source": sc_state.get("criteria_source", "fixed"),
            "success_criteria": {
                k: v for k, v in sc_state.items() if not (isinstance(k, str) and k.startswith("_"))
            },
            "success_criteria_results": dict(state.get("success_criteria_results") or {}),
            "success_criteria_met": bool(state.get("success_criteria_met", False)),
            "validation_metrics": {
                k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                for k, v in (state.get("validation_metrics") or {}).items()
                if isinstance(v, (int, float, str, bool)) or v is None
            },
            "test_metrics": {
                k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                for k, v in (state.get("test_metrics") or {}).items()
                if isinstance(v, (int, float, str, bool)) or v is None
            },
            "train_metrics": {
                k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                for k, v in (state.get("train_metrics") or {}).items()
                if isinstance(v, (int, float, str, bool)) or v is None
            },
            "experiment_id": state.get("experiment_id"),
            "pipeline_halted": bool(state.get("pipeline_halted", False)),
            "halt_reason": state.get("halt_reason"),
            "model_usefulness": state.get("model_usefulness"),
            # Codex pass-1 HIGH-2 (PR #137 v4 G1)
            "deployer_verdict": _deployer_verdict,
            "deployer_verdict_description": _deployer_description,
            "deployer_verdict_recommendation": _deployer_recommendation,
            # Codex pass-1 HIGH-3 (PR #137 v4 G1)
            "cohort_size": _cohort_size,
            "trained_model_present": state.get("trained_model") is not None,
            "class_imbalance_info": state.get("class_imbalance_info") or {},
            # Layer 3 / Layer 5 audit surface used by the CSU val_AUC
            # measurement test (Item A2 of the engineering-actionable arc).
            # ``permutation_test`` carries the model_trainer's permutation
            # null p_value; ``adaptive_verdicts`` carries per-feature Layer 1
            # / Layer 3 verdicts (with adversarial z_score top-level for
            # layer="3"). Recursively normalised via ``_to_jsonable`` (codex
            # M4) so numpy scalars / Pydantic models / datetimes round-trip
            # through ``json.dumps`` as JSON-native types instead of
            # opaque str() coercions that would defeat downstream readers.
            "permutation_test": _to_jsonable(state.get("permutation_test") or {}),
            "adaptive_verdicts": _to_jsonable(state.get("adaptive_verdicts") or []),
            "leakage_dropped_features": list(state.get("leakage_dropped_features") or []),
            "feature_manifest_source": (
                (state.get("scope_spec") or {}).get("feature_manifest_source")
            ),
            "split_assignments": {
                str(k): v
                for k, v in (state.get("split_assignments") or {}).items()
                if isinstance(v, str)
            },
            # v5 Gate C1: regulatory_deployment_manifest surfaces from
            # validate_promotion. The deployer-agent's run() composes it
            # into the final state when promotion validation fires. We
            # serialize it verbatim so integration tests can pin the
            # T2.6c authorization payload emitted by a real CSU run.
            # Codex pass-1 MED-1: real-CSU-runner coverage of the
            # manifest emission contract.
            "regulatory_deployment_manifest": _to_jsonable(
                state.get("regulatory_deployment_manifest")
            ),
        }
        e2e_path = Path(e2e_out)
        e2e_path.parent.mkdir(parents=True, exist_ok=True)
        e2e_path.write_text(json.dumps(artifact, indent=2, default=str))

        # Durable markdown twin of the regulatory authorization manifest +
        # advisory-vs-enforced gate map (gaps G6/G12), written alongside the
        # JSON so an operator capturing tier0 output also gets a human/audit-
        # readable compliance report (the console summary prints the same).
        try:
            from src.agents.ml_foundation.model_deployer.regulatory_report import (
                format_regulatory_report,
            )

            report_md = format_regulatory_report(state.get("regulatory_deployment_manifest"))
            e2e_path.with_suffix(".regulatory.md").write_text(report_md)
        except Exception as report_exc:  # best-effort artifact; never fail the run
            print(f"WARN: failed to write regulatory report markdown: {report_exc}")

    return state


def _build_parser() -> argparse.ArgumentParser:
    """Build the run_tier0_test CLI argparse parser.

    Extracted from ``main()`` so unit tests can exercise the real
    parser (5B-I-3) without spawning a subprocess just to read
    ``--help`` output. The previous subprocess-based test was
    needlessly slow and brittle to PATH / venv differences.
    """
    parser = argparse.ArgumentParser(description="Run Tier 0 MLOps workflow test")
    parser.add_argument(
        "--step", type=int, choices=range(1, 9), help="Run only a specific step (1-8)"
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow tracking (enabled by default for model_uri generation)",
    )
    parser.add_argument("--enable-opik", action="store_true", help="Enable Opik tracing")
    parser.add_argument(
        "--hpo-trials", type=int, default=10, help="Number of HPO trials (default: 10)"
    )
    parser.add_argument(
        "--min-samples-per-split",
        type=int,
        default=10,
        help=(
            "Minimum viable samples per split for split_enforcer gate "
            "(default: 10; lower for small-cohort RWD, e.g. 5 for Optum n=47)"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without executing"
    )
    parser.add_argument(
        "--imbalanced",
        type=float,
        default=None,
        metavar="RATIO",
        help="Create imbalanced data with specified minority ratio (e.g., 0.1 for 10%% minority class)",
    )
    parser.add_argument(
        "--no-bentoml",
        action="store_true",
        help="Skip BentoML model serving verification (enabled by default)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/results",
        help="Directory to save results MD file (default: docs/results)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save results to file (only print to console)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Load real-world data from this directory instead of generating synthetic data",
    )
    parser.add_argument(
        "--brand", type=str, default=None, help="Override CONFIG.brand (e.g. 'competitor')"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Override CONFIG.target_outcome (e.g. 'treatment_initiated')",
    )
    parser.add_argument(
        "--indication",
        type=str,
        default=None,
        help="Override CONFIG.indication (e.g. 'Chronic Spontaneous Urticaria (CSU)')",
    )
    parser.add_argument(
        "--feature-manifest-source",
        type=str,
        choices=("csu", "optum", "synthetic", "synthetic_csu"),
        default=None,
        help=(
            "Opt this run into a cohort-specific feature manifest so Layer 5 "
            "(adaptive_validity_check) consults the matching FeatureContract "
            "registry for layer='1' verdicts. When omitted the value is "
            "auto-detected from --data-dir ('data/rwd/csu' → 'csu', "
            "'data/rwd/optum' → 'optum', 'data/synthetic' → 'synthetic'); "
            "pass explicitly to override the auto-detection. When neither "
            "an explicit choice nor a recognizable --data-dir path segment "
            "is provided, the value stays unset, preserving the cross-cohort "
            "no-false-positive default."
        ),
    )
    parser.add_argument(
        "--regime",
        type=str,
        choices=_VALID_REGIMES,
        default="default",
        help=(
            "Synthetic data regime (Block 4 + Section A of pre_phase2_unblockers). "
            "'default': positive_rate=0.30, baseline 13-18%% positive share. "
            "'adverse': positive_rate=0.02 → extreme imbalance, exercises "
            "remediation paths (recommended_strategy=combined). "
            "'clean': positive_rate=0.50 + signal_strength=1.4 + noise_sd=0.05 "
            "+ signalized extra features → strong-signal regime intended as "
            "the Phase 2 baseline (val AUC ~0.78-0.82, deployer succeeds). "
            "'scenario_a': synthetic_v2 HR+/HER2- early BC iDFS (Kisqali "
            "franchise), 40 clinically-grounded features, n=6000, calibrated "
            "AUC band [0.78, 0.83] (9/10 seeds; scenarios/scenario_a.py:7). "
            "'scenario_a_balanced': scenario_a derivative with target "
            "prevalence shifted 0.20 → 0.50, intact feature↔target signal "
            "(see synthetic_cohort_growth_plan_20260509.md Phase 3). "
            "'scenario_b': IgAN/ESKD screening, 25 features, prev=0.05, "
            "AUC band [0.72, 0.78] (scenarios/scenario_b.py:7). "
            "'scenario_c': CSU treatment response, 60 features, prev=0.40, "
            "AUC band [0.82, 0.88] (scenarios/scenario_c.py:7). "
            "Ignored when --data-dir is set."
        ),
    )
    parser.add_argument(
        "--deployment-intent",
        type=str,
        choices=("clinical", "commercial"),
        default="clinical",
        help=(
            "Deployment use case — recalibrates the deployment AUC bar. "
            "'clinical' (default): published / site-of-care decision model, "
            "literature floor AUC 0.75 (Vickers 2019; Cook 2007). "
            "'commercial': HCP targeting / propensity model (never used at site "
            "of care), separately-cited floor AUC 0.65 + prevalence-aware "
            "operating gates (recall 0.50, MCC 0.10, net-benefit p_t 0.10). "
            "The default NEVER silently loosens the bar — opt in explicitly."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("auto", "random", "combined"),
        default="auto",
        help=(
            "Data split strategy for step_5 (Block 4, Finding #7). "
            "'auto' picks combined entity+temporal split when entity and "
            "date columns are detected, falling back to random+stratified "
            "otherwise. 'random' is the explicit opt-out — always run the "
            "legacy stratified random 4-way split. 'combined' forces the "
            "combined split and errors out if entity/date columns are "
            "absent."
        ),
    )
    parser.add_argument(
        "--no-demo-cost-matrix",
        action="store_true",
        help=(
            "Suppress the Block 5B (#10) auto-injected placeholder cost "
            "matrix. When this flag is absent (the default), a unit-shape "
            "matrix {tp:+1, fp:-0.05, fn:-1, tn:0} is set on scope_spec "
            "so the evaluator can emit business_utility for verification. "
            "Pass this flag to reproduce the pre-Block-5B baseline where "
            "business_utility is absent because no cost matrix flowed "
            "through the pipeline."
        ),
    )
    parser.add_argument(
        "--n-total",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override the synthetic_v2 scenario default cohort size "
            "(typically 6000). N must be ≥ 100 per the api.py:158 safety "
            "floor. Applies only to --regime scenario_*; ignored for "
            "default/adverse/clean (those go through the legacy "
            "ml_patients() generator with n_samples=1500). When omitted, "
            "the scenario's builder.default_n_total is used so no-flag "
            "invocations remain bit-identical to the pre-PR baseline."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help=(
            "Random seed for the synthetic_v2 generator and downstream "
            "training. Default 42 matches the pre-PR hardcoded seed so "
            "no-flag invocations remain bit-identical. Used by the "
            "synthetic_cohort_growth multi-seed sweep "
            "(.claude/plans/synthetic_cohort_growth_plan_20260509.md "
            "Phase 1.3) to measure variance across seeds at fixed n_total."
        ),
    )
    return parser


def main():
    """Main entry point."""
    import sys
    import io
    from pathlib import Path

    parser = _build_parser()
    args = parser.parse_args()

    # --n-total validation: api.py:158 enforces ≥100 at generation time, but
    # surfacing the error at the CLI boundary is friendlier to operators
    # running the multi-seed sweep, who would otherwise see the failure mid-run.
    if args.n_total is not None and args.n_total < 100:
        parser.error(
            f"--n-total must be ≥ 100 (got {args.n_total}); "
            "synthetic_v2.api.py:158 enforces this floor because stratified "
            "splits at low prevalence become degenerate below it."
        )

    # --imbalanced × scenario-regime conflict (backlog #21.7): the synthetic_v2
    # dispatch in generate_sample_data:1579 returns BEFORE the relabel block at
    # lines 1609-1621, so --imbalanced is silently ignored under any
    # --regime scenario_*. Discovered during plan Phase 3.3 contrast empirically
    # (conditions A and C produced bit-identical metrics for seed=42).
    # Fail loud at the CLI boundary instead of silently dropping the flag.
    if args.imbalanced is not None and args.regime in _SCENARIO_REGIME_TO_NAME:
        if args.imbalanced == 0.50:
            # Regime-aware redirect (codex pass-2 LOW): only scenario_a has a
            # balanced variant. scenario_b/c don't, so naive "use
            # scenario_a_balanced" misleads users who wanted scenario_b/c's DGP.
            if args.regime == "scenario_a_balanced":
                redirect = (
                    "--regime scenario_a_balanced already produces a 50:50 "
                    "cohort via intercept-solver prevalence calibration — "
                    "drop --imbalanced 0.50 to use the balanced regime as-is."
                )
            elif args.regime == "scenario_a":
                redirect = (
                    "Use --regime scenario_a_balanced for a signal-preserving "
                    "50:50 cohort (scenario_a DGP, prevalence re-calibrated via "
                    "intercept solver — preserves feature ↔ target correlation)."
                )
            else:  # scenario_b, scenario_c — no balanced variant
                redirect = (
                    f"--regime {args.regime} has no balanced variant. Either "
                    "(a) use --regime scenario_a_balanced for a 50:50 cohort "
                    "with scenario_a's DGP, or (b) use a legacy regime "
                    "(default/adverse/clean) with --imbalanced 0.50 for "
                    "post-hoc relabel on top of a non-scenario data generator."
                )
        else:
            redirect = (
                "No scenario regime accepts an arbitrary prevalence ratio; "
                "use a legacy regime (default/adverse/clean) with --imbalanced "
                "for post-hoc relabel. See backlog #21.7."
            )
        parser.error(
            f"--imbalanced {args.imbalanced} is silently ignored under "
            f"--regime {args.regime} (generate_sample_data:1579 returns before "
            f"the relabel block at lines 1609-1621). " + redirect
        )

    # Update config
    if args.disable_mlflow:
        CONFIG.enable_mlflow = False
    CONFIG.enable_opik = args.enable_opik
    if args.enable_opik:
        os.environ["OPIK_ENABLED"] = "true"
    CONFIG.hpo_trials = args.hpo_trials
    CONFIG.min_samples_per_split = args.min_samples_per_split
    if args.brand:
        CONFIG.brand = args.brand
    elif args.regime in _SCENARIO_REGIME_TO_BRAND:
        # Without this auto-sync, scenario_b/c write df["brand"]=Fabhalta/Remibrutinib
        # while CONFIG.brand stays at its default (Kisqali), creating data↔metadata
        # divergence in MLflow tags, cohort_name, scope-spec problem description,
        # and state["brand"] readers throughout the runner.
        # Gate on _SCENARIO_REGIME_TO_BRAND (the dict we read) to avoid KeyError
        # if the two maps drift out of sync (codex review MEDIUM).
        CONFIG.brand = _SCENARIO_REGIME_TO_BRAND[args.regime]
    if args.target:
        CONFIG.target_outcome = args.target
    if args.indication:
        CONFIG.indication = args.indication

    # Setup output capture if saving results
    output_buffer = None
    original_stdout = sys.stdout

    if not args.no_save:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a tee-like output that writes to both console and buffer
        class TeeOutput:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for stream in self.streams:
                    stream.write(data)
                    stream.flush()

            def flush(self):
                for stream in self.streams:
                    stream.flush()

        output_buffer = io.StringIO()
        sys.stdout = TeeOutput(original_stdout, output_buffer)

    try:
        # Run pipeline
        asyncio.run(
            run_pipeline(
                step=args.step,
                dry_run=args.dry_run,
                imbalance_ratio=args.imbalanced,
                include_bentoml=not args.no_bentoml,
                data_dir=args.data_dir,
                regime=args.regime,
                deployment_intent=args.deployment_intent,
                split_mode=args.split,
                inject_demo_cost_matrix=not args.no_demo_cost_matrix,
                feature_manifest_source=_resolve_feature_manifest_source(
                    args.data_dir, args.feature_manifest_source
                ),
                n_total=args.n_total,
                seed=args.seed,
            )
        )
    finally:
        # Restore stdout
        sys.stdout = original_stdout

        # Save results to file
        if not args.no_save and output_buffer:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "rwd" if args.data_dir else "tier0"
            output_file = Path(args.output_dir) / f"{prefix}_pipeline_run_{timestamp}.md"

            # Add markdown header
            md_content = f"# {'RWD' if args.data_dir else 'Tier 0'} Pipeline Run Results\n\n"
            md_content += f"**Generated**: {datetime.now().isoformat()}\n"
            if args.data_dir:
                md_content += f"**Data**: {args.data_dir}\n"
                md_content += f"**Target**: {CONFIG.target_outcome}\n"
                md_content += f"**Indication**: {CONFIG.indication}\n"
            md_content += "\n```\n"
            md_content += output_buffer.getvalue()
            md_content += "```\n"

            with open(output_file, "w") as f:
                f.write(md_content)

            print(f"\n📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
