#!/usr/bin/env python3
"""Manual step-by-step runner for Tier 0 MLOps workflow test.

This script executes each agent in the Tier 0 pipeline individually
with detailed output and verification between steps.

Usage:
    # Run full pipeline
    python scripts/run_tier0_test.py

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
import os
import signal
import subprocess
import sys
import time as time_module
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

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
if not os.environ.get("SUPABASE_URL"):
    os.environ["SUPABASE_URL"] = "http://localhost:54321"


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
    min_auc_threshold: float = 0.55
    # Minimum recall for minority class - a model that predicts all 0s is useless
    min_minority_recall: float = 0.10  # At least 10% of actual positives must be found
    min_minority_precision: float = 0.05  # At least 5% of predicted positives should be correct
    enable_mlflow: bool = True  # MLflow must be enabled for model_uri to be generated
    enable_opik: bool = False


CONFIG = TestConfig()


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
    print(f"    {'-'*65}")

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


def print_interpretation(title: str, observations: list[str], recommendations: list[str] = None) -> None:
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
    metrics: dict,
    accuracy_analysis: dict,
    min_recall: float,
    min_precision: float
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
        observations.append(f"    Model will miss {(1-recall)*100:.0f}% of actual positives")
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
        observations.append(f"Of {pred_pos} predicted positives, {tp} correct ({precision:.1%} precision)")
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
    title: str = "Confusion Matrix"
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
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1, "npv": npv, "fpr": fpr, "fnr": fnr
    }


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print scikit-learn classification report with formatting."""
    from sklearn.metrics import classification_report

    print("\n  Classification Report (per-class):")
    report = classification_report(y_true, y_pred, target_names=["Class 0 (No Discont.)", "Class 1 (Discont.)"])
    for line in report.split('\n'):
        print(f"    {line}")


def print_threshold_analysis(y_true: np.ndarray, y_proba: np.ndarray, optimal_threshold: float = 0.5) -> None:
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
    print(f"    Percentiles: " + ", ".join([f"P{p}={v:.3f}" for p, v in zip(percentiles, pct_values)]))

    print("\n  Threshold Analysis:")
    print(f"    {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Pred Pos':<12}")
    print(f"    {'-'*56}")

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
        print(f"    {thresh:<12.2f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {n_pred_pos:<12}{marker}")


def print_model_coefficients(model: Any, feature_names: list[str]) -> None:
    """Print model coefficients/weights for interpretability."""
    print("\n  Model Coefficients/Weights:")

    # Handle different model types
    if hasattr(model, 'coef_'):
        coefs = model.coef_.flatten()
        intercept = getattr(model, 'intercept_', [0])[0]

        print(f"    Intercept: {intercept:.4f}")
        print(f"    Feature Coefficients:")

        # Sort by absolute value
        coef_pairs = list(zip(feature_names, coefs))
        coef_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        for name, coef in coef_pairs:
            direction = "↑" if coef > 0 else "↓" if coef < 0 else "○"
            print(f"      {direction} {name}: {coef:+.4f}")

    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        print(f"    Feature Importances (tree-based):")

        imp_pairs = list(zip(feature_names, importances))
        imp_pairs.sort(key=lambda x: x[1], reverse=True)

        for name, imp in imp_pairs:
            bar = "█" * int(imp * 20)
            print(f"      {name}: {imp:.4f} {bar}")
    else:
        print(f"    (Model type {type(model).__name__} does not expose coefficients)")


def print_data_distribution_analysis(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray = None) -> None:
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


def print_detailed_summary(
    experiment_id: str,
    step_results: list[StepResult],
    state: dict[str, Any]
) -> None:
    """Print detailed results from each tier0 step using enhanced format.

    Args:
        experiment_id: The experiment identifier
        step_results: List of StepResult objects from each step
        state: Pipeline state with all collected data
    """
    print(f"\n{'='*70}")
    print("DETAILED STEP RESULTS")
    print(f"{'='*70}")

    for result in step_results:
        status_icon = "✅" if result.status == "success" else "⚠️" if result.status == "warning" else "❌"
        print(f"\n{'-'*70}")
        print(f"STEP {result.step_num}: {result.step_name} [{status_icon} {result.status.upper()}]")
        print(f"{'-'*70}")

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
            print_step_result(result.status, f"{result.result_message} ({result.duration_seconds:.1f}s)")

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
        print(f"\n{'='*70}")
        print("COHORT CONSTRUCTION ANALYSIS")
        print(f"{'='*70}")

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

        if hasattr(cohort_result, 'eligibility_stats') and cohort_result.eligibility_stats:
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
        print(f"\n{'='*70}")
        print("CLASS IMBALANCE REMEDIATION")
        print(f"{'='*70}")

        print("\n  📊 Imbalance Analysis:")
        print(f"    • Imbalance Detected: Yes")
        print(f"    • Severity: {class_imbalance_info.get('imbalance_severity', 'unknown').upper()}")
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
            orig_samples = resampling_info.get('original_samples')
            resamp_samples = resampling_info.get('resampled_samples')
            print(f"    • Original Samples: {orig_samples}")
            print(f"    • Resampled Samples: {resamp_samples}")
            new_ratio = resampling_info.get('new_minority_ratio')
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
        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE (SHAP)")
        print(f"{'='*70}")
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
        print(f"\n{'='*70}")
        print("FINAL MODEL PERFORMANCE")
        print(f"{'='*70}")

        # Key metrics
        key_metrics = ["roc_auc", "accuracy", "precision", "recall", "f1_score", "pr_auc", "brier_score"]
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

    # =========================================================================
    # ENHANCED ACCURACY ANALYSIS SECTION
    # =========================================================================
    accuracy_data = state.get("accuracy_analysis", {})
    if accuracy_data:
        print(f"\n{'='*70}")
        print("ENHANCED ACCURACY ANALYSIS")
        print(f"{'='*70}")

        # Confusion Matrix
        if accuracy_data.get("y_true") is not None and accuracy_data.get("y_pred") is not None:
            y_true = np.array(accuracy_data["y_true"])
            y_pred = np.array(accuracy_data["y_pred"])
            y_proba = np.array(accuracy_data["y_proba"]) if accuracy_data.get("y_proba") is not None else None

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
            feature_cols = accuracy_data.get("feature_columns", ["days_on_therapy", "hcp_visits", "prior_treatments"])
            print_model_coefficients(trained_model, feature_cols)

        # Data distribution across splits
        if accuracy_data.get("y_train") is not None:
            print_data_distribution_analysis(
                np.array(accuracy_data.get("y_train", [])),
                np.array(accuracy_data.get("y_val", [])),
                np.array(accuracy_data.get("y_test", []))
            )

        # Train vs Validation comparison (overfitting check)
        train_metrics = accuracy_data.get("train_metrics", {})
        val_metrics = accuracy_data.get("val_metrics", {})
        if train_metrics and val_metrics:
            print("\n  Overfitting Analysis (Train vs Validation):")
            print(f"    {'Metric':<15} {'Train':<12} {'Validation':<12} {'Delta':<12} {'Status':<15}")
            print(f"    {'-'*60}")

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
                    print(f"    {metric:<15} {train_val:<12.4f} {val_val:<12.4f} {delta:+<12.4f} {status:<15}")

    # =========================================================================
    # MODEL USEFULNESS VERDICT
    # =========================================================================
    model_usefulness = state.get("model_usefulness_verdict", {})
    if not model_usefulness and accuracy_data:
        # Compute from accuracy data if not explicitly set
        y_pred = accuracy_data.get("y_pred", [])
        n_pos_pred = sum(y_pred) if y_pred else 0
        val_metrics = accuracy_data.get("val_metrics", {})
        model_usefulness = {
            "status": "useless" if n_pos_pred == 0 else "needs_review",
            "reason": "predicts_all_negative" if n_pos_pred == 0 else "unknown",
            "minority_recall": val_metrics.get("recall", 0),
            "minority_precision": val_metrics.get("precision", 0),
        }

    if model_usefulness or accuracy_data:
        print(f"\n{'='*70}")
        print("⚠️  MODEL USEFULNESS VERDICT")
        print(f"{'='*70}")

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
            val_metrics = accuracy_data.get("val_metrics", {}) if accuracy_data else {}
            auc_roc = val_metrics.get("roc_auc", 0)
            recall = val_metrics.get("recall", 0)
            precision = val_metrics.get("precision", 0)
            f1 = val_metrics.get("f1", 0)

            # Determine verdict based on metrics
            if auc_roc >= 0.85 and recall >= 0.7:
                verdict = "EXCELLENT"
                icon = "🌟"
                description = "Model has strong discrimination and high recall"
                deploy_recommendation = "Ready for production deployment"
            elif auc_roc >= 0.75 and recall >= 0.5:
                verdict = "GOOD"
                icon = "✅"
                description = "Model has good discrimination and acceptable recall"
                deploy_recommendation = "Suitable for staging/production with monitoring"
            elif auc_roc >= 0.65 and recall >= 0.3:
                verdict = "ACCEPTABLE"
                icon = "⚡"
                description = "Model has moderate performance, meets minimum thresholds"
                deploy_recommendation = "Deploy with caution, monitor closely"
            elif auc_roc >= 0.55:
                verdict = "MARGINAL"
                icon = "⚠️"
                description = "Model barely exceeds random chance"
                deploy_recommendation = "Consider retraining with more data or different approach"
            else:
                verdict = "POOR"
                icon = "❌"
                description = "Model performs near or below random chance"
                deploy_recommendation = "Do not deploy, requires significant improvement"

            print(f"\n  {icon} VERDICT: {verdict}")
            print(f"\n  Assessment: {description}")
            print(f"\n  Key Metrics:")
            print(f"    • AUC-ROC:   {auc_roc:.4f}")
            print(f"    • Recall:    {recall:.4f} ({recall*100:.1f}% of positives detected)")
            print(f"    • Precision: {precision:.4f} ({precision*100:.1f}% of predictions correct)")
            print(f"    • F1 Score:  {f1:.4f}")
            print(f"    • Positive Predictions: {n_pos_pred}/{total_pred} ({n_pos_pred/total_pred*100:.1f}%)")
            print(f"\n  Recommendation: {deploy_recommendation}")

    # Deployment Info
    deployment_manifest = state.get("deployment_manifest", {})
    if deployment_manifest:
        print(f"\n{'='*70}")
        print("DEPLOYMENT STATUS")
        print(f"{'='*70}")
        print(f"\n  • Deployment ID: {deployment_manifest.get('deployment_id', 'N/A')}")
        print(f"  • Environment: {deployment_manifest.get('environment', 'N/A')}")
        print(f"  • Status: {deployment_manifest.get('status', 'N/A')}")
        print(f"  • Endpoint: {deployment_manifest.get('endpoint_url', 'N/A')}")

    print(f"\n{'='*70}")


def generate_sample_data(
    n_samples: int = 100,
    seed: int = 42,
    imbalance_ratio: float | None = None,
) -> pd.DataFrame:
    """Generate sample patient journey data using the ML-ready generator.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        imbalance_ratio: If provided, force minority class to this ratio (e.g., 0.1 for 10%)
                        None means balanced data (~50/50)
    """
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


async def start_bentoml_service(
    model_tag: str,
    port: int = 3001,
    preprocessor: Any = None,
    framework: str = "sklearn",
) -> dict:
    """Start BentoML service serving the real trained model.

    Args:
        model_tag: BentoML model tag from registration
        port: Port to serve on
        preprocessor: Optional fitted preprocessor to apply before prediction
        framework: ML framework used to save the model ("sklearn", "xgboost", "lightgbm")

    Returns:
        {"started": True, "endpoint": "http://localhost:3001", "pid": <pid>}
    """
    import httpx

    # Save preprocessor to temp file if provided
    preprocessor_path = ""
    if preprocessor is not None:
        try:
            import joblib as _joblib
            preprocessor_path = "/tmp/tier0_bentoml_preprocessor.pkl"
            _joblib.dump(preprocessor, preprocessor_path)
            print(f"    Saved preprocessor to {preprocessor_path}")
        except Exception as e:
            print(f"    Warning: could not save preprocessor: {e}")
            preprocessor_path = ""

    # Map framework to bentoml load function
    load_fn_map = {
        "sklearn": "bentoml.sklearn.load_model",
        "xgboost": "bentoml.xgboost.load_model",
        "lightgbm": "bentoml.lightgbm.load_model",
    }
    load_fn = load_fn_map.get(framework, "bentoml.picklable_model.load_model")

    # Build preprocessor loading code
    if preprocessor_path:
        preprocessor_code = f'''
        try:
            import joblib
            self.preprocessor = joblib.load("{preprocessor_path}")
        except Exception as e:
            print(f"Warning: could not load preprocessor: {{e}}")
            self.preprocessor = None'''
        preprocess_step = '''
        if self.preprocessor is not None:
            import pandas as pd
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            feature_names = None
            if hasattr(self.preprocessor, 'numeric_features'):
                feature_names = list(self.preprocessor.numeric_features) + list(getattr(self.preprocessor, 'categorical_features', []))
            elif hasattr(self.preprocessor, 'feature_names_in_'):
                feature_names = list(self.preprocessor.feature_names_in_)
            if feature_names and len(feature_names) == arr.shape[1]:
                arr = pd.DataFrame(arr, columns=feature_names)
            arr = self.preprocessor.transform(arr)'''
    else:
        preprocessor_code = '''
        self.preprocessor = None'''
        preprocess_step = ''

    # Generate a service file dynamically for the model
    service_code = f'''
import bentoml
import numpy as np

@bentoml.service(name="tier0_model_service")
class Tier0ModelService:
    def __init__(self):
        self.model = {load_fn}("{model_tag}")
        self.model_tag = "{model_tag}"
{preprocessor_code}

    @bentoml.api
    async def predict(self, features: list) -> dict:
        import time
        start = time.time()
        arr = np.array(features)
{preprocess_step}
        predictions = self.model.predict(arr)
        probas = self.model.predict_proba(arr) if hasattr(self.model, 'predict_proba') else None
        elapsed = (time.time() - start) * 1000
        return {{
            "predictions": predictions.tolist(),
            "probabilities": probas.tolist() if probas is not None else None,
            "latency_ms": elapsed,
            "model_tag": self.model_tag,
        }}

    @bentoml.api
    async def health(self) -> dict:
        return {{"status": "healthy", "model_tag": self.model_tag}}
'''

    # Clean up any Python files in /tmp that might shadow built-in modules
    # This prevents circular import errors when BentoML runs from /tmp
    builtin_shadow_files = ["types.py", "enum.py", "re.py", "functools.py", "dataclasses.py"]
    for shadow_file in builtin_shadow_files:
        shadow_path = Path("/tmp") / shadow_file
        if shadow_path.exists():
            shadow_path.unlink()

    # Write service file
    service_path = Path("/tmp/tier0_bentoml_service.py")
    service_path.write_text(service_code)
    print(f"    Generated service file: {service_path}")

    # Start bentoml serve in background
    # Use cwd=/tmp to avoid picking up project-level BentoML configuration
    process = subprocess.Popen(
        ["bentoml", "serve", str(service_path), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/tmp",
    )
    print(f"    Starting BentoML service on port {port} (PID: {process.pid})...")

    # Wait for service to be ready
    # BentoML @api endpoints use POST, not GET
    endpoint = f"http://localhost:{port}"
    async with httpx.AsyncClient() as client:
        for retry in range(30):  # 30 retries (30 seconds max)
            await asyncio.sleep(1)
            try:
                resp = await client.post(f"{endpoint}/health", timeout=2.0)
                if resp.status_code == 200:
                    print(f"    Service ready at {endpoint}")
                    return {"started": True, "endpoint": endpoint, "pid": process.pid}
            except Exception:
                # Check if process died
                if process.poll() is not None:
                    stderr = process.stderr.read().decode() if process.stderr else ""
                    return {
                        "started": False,
                        "error": f"Process exited with code {process.returncode}: {stderr[:500]}",
                    }

    # Timeout - kill process
    process.terminate()
    return {"started": False, "error": "Service startup timeout (30s)"}


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
    """Deploy model to the persistent e2i-bentoml systemd service.

    Checks if the systemd service is installed, updates the env file with the
    model tag, restarts the service, and waits for health. Falls back to
    returning not-available so the caller can use ephemeral mode.

    Args:
        model_tag: BentoML model tag to deploy

    Returns:
        {"available": True, "endpoint": "http://localhost:3000"} or
        {"available": False, "reason": "..."}
    """
    import httpx

    # Check if systemd service exists
    try:
        check = subprocess.run(
            ["systemctl", "is-enabled", "e2i-bentoml"],
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            return {"available": False, "reason": "e2i-bentoml.service not installed"}
    except Exception:
        return {"available": False, "reason": "systemctl not available"}

    # Update env file with the model tag
    env_path = Path("/opt/e2i_causal_analytics/deploy/e2i-bentoml.env")
    if not env_path.exists():
        # Try local path
        env_path = Path(__file__).parent.parent / "deploy" / "e2i-bentoml.env"

    if env_path.exists():
        lines = env_path.read_text().splitlines()
        new_lines = [
            ln for ln in lines
            if not ln.startswith("E2I_BENTOML_MODEL_TAG=")
            and not ln.startswith("E2I_BENTOML_MODEL_NAME=")
        ]
        new_lines.append(f"E2I_BENTOML_MODEL_TAG={model_tag}")
        env_path.write_text("\n".join(new_lines) + "\n")

    # Restart service
    try:
        restart = subprocess.run(
            ["sudo", "systemctl", "restart", "e2i-bentoml"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if restart.returncode != 0:
            return {
                "available": False,
                "reason": f"restart failed: {restart.stderr.strip()}",
            }
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": "restart timed out"}
    except Exception as e:
        return {"available": False, "reason": str(e)}

    # Wait for health
    endpoint = "http://localhost:3000"
    async with httpx.AsyncClient() as client:
        for _ in range(30):
            await asyncio.sleep(1)
            try:
                resp = await client.get(f"{endpoint}/healthz", timeout=2.0)
                if resp.status_code == 200:
                    return {
                        "available": True,
                        "endpoint": endpoint,
                        "persistent": True,
                    }
            except Exception:
                continue

    return {"available": False, "reason": "health check timeout after 30s"}


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

async def step_1_scope_definer(experiment_id: str) -> dict[str, Any]:
    """Step 1: Define ML problem scope."""
    import time as time_mod
    step_start = time_mod.time()

    print_header(1, "SCOPE DEFINER")

    from src.agents.ml_foundation.scope_definer import ScopeDefinerAgent

    # Input preparation
    input_data = {
        "problem_description": f"Predict patient discontinuation risk for {CONFIG.brand}",
        "business_objective": "Identify high-risk patients early for intervention",
        "target_outcome": CONFIG.target_outcome,
        "problem_type_hint": CONFIG.problem_type,
        "brand": CONFIG.brand,
    }

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
            scope_spec.get("problem_type", "missing")
        ),
        (
            "Prediction target set",
            bool(scope_spec.get("prediction_target")),
            "prediction_target present",
            scope_spec.get("prediction_target", "missing")
        ),
        (
            "Minimum samples specified",
            bool(scope_spec.get("minimum_samples")),
            "minimum_samples > 0",
            str(scope_spec.get("minimum_samples", "missing"))
        ),
        (
            "Scope validation",
            validation_passed,
            "validation_passed = True",
            f"validation_passed = {validation_passed}"
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
        observations.append(f"Sample requirement ({scope_spec.get('minimum_samples')}) appropriate for ML")

    print_interpretation("Scope Analysis", observations, recommendations if recommendations else None)

    # Final result
    duration = time_mod.time() - step_start
    if validation_passed:
        print_step_result("success", f"Scope definition complete ({duration:.1f}s)")
    else:
        print_step_result("warning", f"Scope has validation warnings ({duration:.1f}s)")

    return result


async def step_2_data_preparer(
    experiment_id: str, scope_spec: dict, sample_df: pd.DataFrame
) -> dict[str, Any]:
    """Step 2: Load and prepare data with QC."""
    import time as time_mod
    step_start = time_mod.time()

    print_header(2, "DATA PREPARER")

    from src.agents.ml_foundation.data_preparer import DataPreparerAgent

    # Override required_features with actual columns from sample data
    available_features = [
        col for col in sample_df.columns
        if col not in ["patient_journey_id", CONFIG.target_outcome, "brand"]
    ]

    # Ensure scope_spec has required fields with realistic values
    scope_spec.update({
        "experiment_id": experiment_id,
        "use_sample_data": True,
        "sample_size": 500,
        "prediction_target": CONFIG.target_outcome,
        "problem_type": CONFIG.problem_type,
        "required_features": available_features,
        "max_staleness_days": 90,
    })

    input_data = {
        "scope_spec": scope_spec,
        "data_source": "patient_journeys",
        "brand": CONFIG.brand,
    }

    print_input_section({
        "data_source": "patient_journeys",
        "brand": CONFIG.brand,
        "sample_size": len(sample_df),
        "features": f"{len(available_features)} available",
    })

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
        (
            "QC Gate",
            gate_passed,
            "gate_passed = True",
            f"gate_passed = {gate_passed}"
        ),
        (
            "Overall QC Score",
            overall_score >= 0.7 if isinstance(overall_score, (int, float)) else False,
            ">= 0.70",
            f"{overall_score:.2f}" if isinstance(overall_score, (int, float)) else str(overall_score)
        ),
        (
            "Training samples",
            train_samples >= 100,
            ">= 100",
            str(train_samples)
        ),
        (
            "Validation samples",
            val_samples >= 30,
            ">= 30",
            str(val_samples)
        ),
    ]

    print_validation_checks(checks)

    # Metrics table - QC dimension scores
    completeness = qc_report.get("completeness_score", 0)
    validity = qc_report.get("validity_score", 0)
    consistency = qc_report.get("consistency_score", 0)
    uniqueness = qc_report.get("uniqueness_score", 0)
    timeliness = qc_report.get("timeliness_score", 0)

    metrics = [
        ("overall_score", overall_score, ">= 0.70", overall_score >= 0.7 if isinstance(overall_score, (int, float)) else None),
        ("completeness", completeness, ">= 0.90", completeness >= 0.9 if isinstance(completeness, (int, float)) else None),
        ("validity", validity, ">= 0.90", validity >= 0.9 if isinstance(validity, (int, float)) else None),
        ("consistency", consistency, ">= 0.90", consistency >= 0.9 if isinstance(consistency, (int, float)) else None),
        ("uniqueness", uniqueness, ">= 0.95", uniqueness >= 0.95 if isinstance(uniqueness, (int, float)) else None),
        ("timeliness", timeliness, ">= 0.80", timeliness >= 0.8 if isinstance(timeliness, (int, float)) else None),
        ("train_samples", train_samples, ">= 100", train_samples >= 100 if isinstance(train_samples, (int, float)) else None),
        ("validation_samples", val_samples, ">= 30", val_samples >= 30 if isinstance(val_samples, (int, float)) else None),
    ]

    print_metrics_table(metrics)

    # Interpretation
    observations, recommendations = interpret_qc_scores(qc_report)

    # Add data readiness observations
    if train_samples and val_samples:
        total = train_samples + val_samples
        train_pct = train_samples / total * 100 if total > 0 else 0
        observations.insert(0, f"Data split: {train_samples} train ({train_pct:.0f}%), {val_samples} validation")

    # Add remediation info if present
    if remediation.get("status") and remediation.get("status") != "not_needed":
        observations.append(f"Remediation was applied: {remediation.get('status')}")
        if remediation.get("actions_taken"):
            for action in remediation.get("actions_taken", [])[:2]:
                observations.append(f"  - {action}")

    print_interpretation("Data Quality Analysis", observations, recommendations if recommendations else None)

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


async def step_2b_feast_registration(
    experiment_id: str, state: dict[str, Any]
) -> dict[str, Any]:
    """Step 2b: Register features with Feast feature store (gracefully degrading)."""
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
        processing_steps.append((
            "Features registered with Feast",
            features_registered > 0 or features_skipped > 0,
            f"{features_registered} registered, {features_skipped} skipped",
        ))

        print_input_section({
            "experiment_id": experiment_id,
            "entity_key": "hcp_id",
            "feature_count": len(feature_state.get("selected_features", [])),
        })
        print_processing_steps(processing_steps)

        # Validation
        checks = [
            ("Feature group created", reg_result.get("feature_group_created", False), "True", str(reg_result.get("feature_group_created", False))),
            ("Features registered", features_registered > 0, "> 0", str(features_registered)),
            ("No registration errors", len(reg_errors) == 0, "0 errors", f"{len(reg_errors)} errors"),
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
            print_step_result("success", f"Feast registration complete: {features_registered} features ({duration:.1f}s)")
        else:
            print_step_result("warning", f"Feast registration: 0 features registered ({duration:.1f}s)")

    except Exception as e:
        duration = time_mod.time() - step_start
        result["status"] = "skipped"
        result["errors"] = [str(e)]
        print_step_result("warning", f"Feast registration skipped: {e} ({duration:.1f}s)")

    return result


async def step_2c_feast_freshness_check(
    state: dict[str, Any]
) -> dict[str, Any]:
    """Step 2c: Check feature freshness in Feast (gracefully degrading)."""
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
        processing_steps.append((
            "Freshness check completed",
            True,
            f"{'fresh' if is_fresh else f'{len(stale_features)} stale'}",
        ))

        print_input_section({
            "feature_refs_count": len(feature_refs),
            "max_staleness_hours": 24.0,
        })
        print_processing_steps(processing_steps)

        # Validation
        checks = [
            ("Freshness check executed", True, "completed", "completed"),
            ("Features fresh", is_fresh, "all fresh", f"{len(stale_features)} stale" if stale_features else "all fresh"),
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
            print_step_result("warning", f"{len(stale_features)} stale features detected ({duration:.1f}s)")

    except Exception as e:
        duration = time_mod.time() - step_start
        result["status"] = "skipped"
        result["errors"] = [str(e)]
        print_step_result("warning", f"Feast freshness check skipped: {e} ({duration:.1f}s)")

    return result


async def step_3_cohort_constructor(patient_df: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    """Step 3: Build patient cohort."""
    import time as time_mod
    step_start = time_mod.time()

    print_header(3, "COHORT CONSTRUCTOR")

    from src.agents.cohort_constructor import CohortConstructorAgent
    from src.agents.cohort_constructor.types import (
        CohortConfig,
        Criterion,
        CriterionType,
        Operator,
        TemporalRequirements,
    )

    print_input_section({
        "input_patients": len(patient_df),
        "brand": CONFIG.brand,
        "inclusion_criteria": "data_quality_score >= 0.5",
        "exclusion_criteria": "None (maximize sample size)",
    })

    # Processing
    processing_steps = []
    processing_steps.append(("Creating CohortConstructorAgent", True, None))

    agent = CohortConstructorAgent(enable_observability=CONFIG.enable_opik)
    processing_steps.append(("Agent initialized", True, None))

    # Create test config
    test_config = CohortConfig(
        cohort_name=f"{CONFIG.brand} Test Cohort",
        brand=CONFIG.brand.lower(),
        indication="test",
        inclusion_criteria=[
            Criterion(
                field="data_quality_score",
                operator=Operator.GREATER_EQUAL,
                value=0.5,
                criterion_type=CriterionType.INCLUSION,
                description="Minimum data quality score",
                clinical_rationale="Ensure data quality for reliable ML predictions",
            ),
        ],
        exclusion_criteria=[],
        temporal_requirements=None,
        required_fields=["patient_journey_id", "brand", "data_quality_score"],
        version="1.0.0-test",
        status="active",
        clinical_rationale="Test cohort using sample data fields - relaxed criteria for testing",
        regulatory_justification="Test configuration for MLOps workflow validation",
    )

    processing_steps.append(("Cohort config created", True, "data_quality_score >= 0.5"))

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
            str(eligible_count)
        ),
        (
            "Retention rate",
            retention_rate >= 0.5,
            ">= 50%",
            f"{retention_rate:.1%}"
        ),
        (
            "Cohort status",
            result.status == "completed",
            "completed",
            result.status
        ),
    ]

    print_validation_checks(checks)

    # Metrics
    metrics = [
        ("input_patients", input_count, None, None),
        ("eligible_patients", eligible_count, f">= {CONFIG.min_eligible_patients}", eligible_count >= CONFIG.min_eligible_patients),
        ("excluded_patients", excluded_count, None, None),
        ("retention_rate", retention_rate, ">= 0.50", retention_rate >= 0.5),
        ("cohort_id", result.cohort_id, None, None),
    ]

    print_metrics_table(metrics)

    # Interpretation
    observations = []
    recommendations = []

    observations.append(f"Patient flow: {input_count} → {eligible_count} ({retention_rate:.1%} retention)")
    observations.append(f"Excluded {excluded_count} patients based on eligibility criteria")

    if eligible_count < CONFIG.min_eligible_patients:
        observations.append(f"⚠️  Cohort size ({eligible_count}) below minimum ({CONFIG.min_eligible_patients})")
        recommendations.append("Relax eligibility criteria or generate more sample data")
        recommendations.append("Consider lowering data_quality_score threshold")
    else:
        observations.append(f"Cohort size ({eligible_count}) sufficient for ML training")

    if retention_rate < 0.5:
        observations.append(f"⚠️  High exclusion rate ({1-retention_rate:.1%}) may indicate data quality issues")
        recommendations.append("Review exclusion criteria for potential over-filtering")

    # Target distribution in cohort
    if CONFIG.target_outcome in eligible_df.columns:
        target_dist = eligible_df[CONFIG.target_outcome].value_counts()
        minority_ratio = target_dist.min() / target_dist.sum() if target_dist.sum() > 0 else 0
        observations.append(f"Target class distribution: {dict(target_dist)}")
        if minority_ratio < 0.2:
            observations.append(f"⚠️  Class imbalance detected ({minority_ratio:.1%} minority)")

    print_interpretation("Cohort Analysis", observations, recommendations if recommendations else None)

    # Final result
    duration = time_mod.time() - step_start
    if eligible_count >= CONFIG.min_eligible_patients:
        print_step_result("success", f"Cohort constructed ({eligible_count} patients, {duration:.1f}s)")
    else:
        print_step_result("warning", f"Cohort below minimum size ({eligible_count}/{CONFIG.min_eligible_patients}, {duration:.1f}s)")

    return eligible_df, result


async def step_4_model_selector(experiment_id: str, scope_spec: dict, qc_report: dict) -> dict[str, Any]:
    """Step 4: Select model candidate."""
    import time as time_mod
    step_start = time_mod.time()

    print_header(4, "MODEL SELECTOR")

    from src.agents.ml_foundation.model_selector import ModelSelectorAgent

    # Normalize qc_report for compatibility
    normalized_qc_report = qc_report.copy()
    if "qc_passed" not in normalized_qc_report:
        normalized_qc_report["qc_passed"] = normalized_qc_report.get("gate_passed", True)
    if "qc_errors" not in normalized_qc_report:
        normalized_qc_report["qc_errors"] = []

    input_data = {
        "scope_spec": scope_spec,
        "qc_report": normalized_qc_report,
        "skip_benchmarks": False,  # Enable benchmarks to evaluate alternatives
    }

    print_input_section({
        "problem_type": scope_spec.get("problem_type", "binary_classification"),
        "qc_passed": normalized_qc_report.get("qc_passed"),
        "skip_benchmarks": False,
    })

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
            algo_name if has_candidate else "missing (will use fallback)"
        ),
        (
            "Selection rationale provided",
            has_rationale,
            "rationale present",
            primary_reason[:50] if primary_reason else "none"
        ),
        (
            "Alternatives evaluated",
            len(alternatives) > 0 or len(alternatives_considered) > 0,
            "> 0 alternatives",
            f"{max(len(alternatives), len(alternatives_considered))} evaluated"
        ),
        (
            "No selection errors",
            not has_error,
            "no errors",
            result.get("error", "none")[:50] if has_error else "none"
        ),
    ]

    print_validation_checks(checks)

    # Metrics table
    metrics = [
        ("algorithm", algo_name, None, None),
        ("selection_score", f"{selection_score:.3f}" if selection_score else "N/A", "> 0.5", selection_score > 0.5 if selection_score else None),
        ("interpretability_score", f"{interpretability:.2f}" if interpretability else "N/A", None, None),
        ("hyperparameters", f"{len(hyperparams)} default params", None, None),
        ("alternatives_evaluated", len(alternatives) if alternatives else len(alternatives_considered), "> 0", len(alternatives) > 0 or len(alternatives_considered) > 0),
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
        observations.append(f"Selected: {algo_name} (score: {selection_score:.3f})" if selection_score else f"Selected: {algo_name}")

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

    print_interpretation("Model Selection Analysis", observations, recommendations if recommendations else None)

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
    y: pd.Series
) -> dict[str, Any]:
    """Step 5: Train model."""
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

    # Normalize qc_report for model_trainer (expects qc_passed)
    normalized_qc_report = qc_report.copy()
    if "qc_passed" not in normalized_qc_report:
        normalized_qc_report["qc_passed"] = normalized_qc_report.get("gate_passed", True)

    # Split data using E2I required ratios: 60%/20%/15%/5%
    n = len(X)
    train_size = int(0.60 * n)
    val_size = int(0.20 * n)
    test_size = int(0.15 * n)
    holdout_size = n - train_size - val_size - test_size

    train_end = train_size
    val_end = train_end + val_size
    test_end = val_end + test_size

    train_data = {"X": X.iloc[:train_end], "y": y.iloc[:train_end], "row_count": train_size}
    validation_data = {"X": X.iloc[train_end:val_end], "y": y.iloc[train_end:val_end], "row_count": val_size}
    test_data = {"X": X.iloc[val_end:test_end], "y": y.iloc[val_end:test_end], "row_count": test_size}
    holdout_data = {"X": X.iloc[test_end:], "y": y.iloc[test_end:], "row_count": holdout_size}

    feature_columns = list(X.columns)

    # Input section
    print_input_section({
        "algorithm": model_candidate["algorithm_name"],
        "total_samples": n,
        "train_samples": f"{train_size} ({train_size / n:.0%})",
        "validation_samples": f"{val_size} ({val_size / n:.0%})",
        "test_samples": f"{test_size} ({test_size / n:.0%})",
        "holdout_samples": f"{holdout_size} ({holdout_size / n:.0%})",
        "hpo_trials": CONFIG.hpo_trials,
        "enable_mlflow": CONFIG.enable_mlflow,
    })

    # Processing
    processing_steps = []
    processing_steps.append(("Creating ModelTrainerAgent", True, None))

    agent = ModelTrainerAgent()
    processing_steps.append(("Agent initialized", True, None))

    input_data = {
        "experiment_id": experiment_id,
        "model_candidate": model_candidate,
        "qc_report": normalized_qc_report,
        "enable_hpo": True,
        "hpo_trials": CONFIG.hpo_trials,
        "problem_type": CONFIG.problem_type,
        "train_data": train_data,
        "validation_data": validation_data,
        "test_data": test_data,
        "holdout_data": holdout_data,
        "enable_mlflow": CONFIG.enable_mlflow,
        "feature_columns": feature_columns,
    }

    processing_steps.append(("HPO optimization", True, f"{CONFIG.hpo_trials} trials"))
    result = await agent.run(input_data)

    # Check class imbalance
    imbalance_detected = result.get("imbalance_detected", False)
    if imbalance_detected:
        processing_steps.append(("Class imbalance detected", True, result.get("imbalance_severity", "unknown")))
        processing_steps.append(("Remediation applied", True, result.get("recommended_strategy", "N/A")))

    processing_steps.append(("Model training complete", True, f"AUC={result.get('auc_roc', 'N/A')}"))

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
    val_metrics = {}
    train_metrics = {}
    y_val_pred = []
    n_positive_predictions = 0
    optimal_threshold = result.get("optimal_threshold", 0.5)

    if trained_model is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as sklearn_f1_score, roc_auc_score

        # Get preprocessed data
        X_val_preprocessed = result.get("X_validation_preprocessed")
        X_test_preprocessed = result.get("X_test_preprocessed")
        fitted_preprocessor = result.get("fitted_preprocessor")

        if X_val_preprocessed is not None:
            X_val = X_val_preprocessed
        elif fitted_preprocessor is not None:
            X_val = fitted_preprocessor.transform(validation_data["X"])
        else:
            X_val = validation_data["X"]

        y_val = validation_data["y"]

        if X_test_preprocessed is not None:
            X_test = X_test_preprocessed
        elif fitted_preprocessor is not None:
            X_test = fitted_preprocessor.transform(test_data["X"])
        else:
            X_test = test_data["X"]

        y_test = test_data["y"]

        if fitted_preprocessor is not None:
            X_train = fitted_preprocessor.transform(train_data["X"])
        else:
            X_train = train_data["X"]
        y_train = train_data["y"]

        # Generate probability predictions
        y_val_proba = None
        y_train_proba = None
        if hasattr(trained_model, 'predict_proba'):
            y_val_proba = trained_model.predict_proba(X_val)[:, 1]
            y_train_proba = trained_model.predict_proba(X_train)[:, 1]

        # Adaptive threshold handling
        if y_val_proba is not None:
            n_pos_at_optimal = ((y_val_proba >= optimal_threshold).astype(int)).sum()
            if n_pos_at_optimal == 0:
                target_n_pos = max(1, int(len(y_val) * 0.10))
                sorted_proba = np.sort(y_val_proba)[::-1]
                if target_n_pos <= len(sorted_proba):
                    adaptive_threshold = max(0.01, sorted_proba[target_n_pos - 1] - 0.001)
                    optimal_threshold = adaptive_threshold
                    result["optimal_threshold"] = adaptive_threshold

        # Make predictions at optimal threshold
        if y_val_proba is not None:
            y_val_pred = (y_val_proba >= optimal_threshold).astype(int)
        else:
            y_val_pred = trained_model.predict(X_val)

        if y_train_proba is not None:
            y_train_pred = (y_train_proba >= optimal_threshold).astype(int)
        else:
            y_train_pred = trained_model.predict(X_train)

        n_positive_predictions = sum(y_val_pred)

        # Calculate metrics
        train_metrics = {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred, zero_division=0),
            "recall": recall_score(y_train, y_train_pred, zero_division=0),
            "f1": sklearn_f1_score(y_train, y_train_pred, zero_division=0),
        }
        if y_train_proba is not None:
            try:
                train_metrics["roc_auc"] = roc_auc_score(y_train, y_train_proba)
            except ValueError:
                pass

        val_metrics = {
            "accuracy": accuracy_score(y_val, y_val_pred),
            "precision": precision_score(y_val, y_val_pred, zero_division=0),
            "recall": recall_score(y_val, y_val_pred, zero_division=0),
            "f1": sklearn_f1_score(y_val, y_val_pred, zero_division=0),
        }
        if y_val_proba is not None:
            try:
                val_metrics["roc_auc"] = roc_auc_score(y_val, y_val_proba)
            except ValueError:
                pass

        # Store accuracy analysis data
        result["accuracy_analysis"] = {
            "y_true": y_val.tolist() if hasattr(y_val, 'tolist') else list(y_val),
            "y_pred": y_val_pred.tolist() if hasattr(y_val_pred, 'tolist') else list(y_val_pred),
            "y_proba": y_val_proba.tolist() if y_val_proba is not None else None,
            "y_train": y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train),
            "y_val": y_val.tolist() if hasattr(y_val, 'tolist') else list(y_val),
            "y_test": y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "feature_columns": feature_columns,
        }

    # =========================================================================
    # VALIDATION CHECKS
    # =========================================================================
    auc = result.get("auc_roc", 0) or val_metrics.get("roc_auc", 0)
    minority_recall = val_metrics.get("recall", 0)
    minority_precision = val_metrics.get("precision", 0)

    checks = [
        (
            "Model trained successfully",
            trained_model is not None,
            "trained_model present",
            "present" if trained_model is not None else "missing"
        ),
        (
            "AUC-ROC threshold",
            auc >= CONFIG.min_auc_threshold if auc else False,
            f">= {CONFIG.min_auc_threshold}",
            f"{auc:.3f}" if auc else "N/A"
        ),
        (
            "Minority recall threshold",
            minority_recall >= CONFIG.min_minority_recall,
            f">= {CONFIG.min_minority_recall:.0%}",
            f"{minority_recall:.2%}"
        ),
        (
            "Minority precision threshold",
            minority_precision >= CONFIG.min_minority_precision,
            f">= {CONFIG.min_minority_precision:.0%}",
            f"{minority_precision:.2%}"
        ),
        (
            "Positive predictions made",
            n_positive_predictions > 0,
            "> 0",
            str(n_positive_predictions)
        ),
    ]

    print_validation_checks(checks)

    # =========================================================================
    # METRICS TABLE
    # =========================================================================
    metrics_list = [
        ("auc_roc", auc, f">= {CONFIG.min_auc_threshold}", auc >= CONFIG.min_auc_threshold if auc else None),
        ("accuracy", val_metrics.get("accuracy"), None, None),
        ("precision", minority_precision, f">= {CONFIG.min_minority_precision:.0%}", minority_precision >= CONFIG.min_minority_precision),
        ("recall", minority_recall, f">= {CONFIG.min_minority_recall:.0%}", minority_recall >= CONFIG.min_minority_recall),
        ("f1_score", val_metrics.get("f1"), None, None),
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
            "Validation Confusion Matrix"
        )

        # Threshold analysis
        if y_proba_list:
            print_threshold_analysis(np.array(y_true_list), np.array(y_proba_list), optimal_threshold)

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
            CONFIG.min_minority_precision
        )
        observations.extend(perf_obs)
        recommendations.extend(perf_rec)

        # Class imbalance interpretation
        if imbalance_detected:
            imb_obs, imb_rec = interpret_class_imbalance({
                "imbalance_detected": True,
                "minority_ratio": result.get("minority_ratio", 0),
                "imbalance_severity": result.get("imbalance_severity", "unknown"),
                "recommended_strategy": result.get("recommended_strategy", "none"),
            })
            observations.extend(imb_obs)
            recommendations.extend(imb_rec)

        # Confusion matrix interpretation
        if result.get("accuracy_analysis"):
            y_pred_list = result["accuracy_analysis"]["y_pred"]
            y_true_list = result["accuracy_analysis"]["y_true"]
            cm_obs = interpret_confusion_matrix({
                "tp": sum(1 for t, p in zip(y_true_list, y_pred_list) if t == 1 and p == 1),
                "tn": sum(1 for t, p in zip(y_true_list, y_pred_list) if t == 0 and p == 0),
                "fp": sum(1 for t, p in zip(y_true_list, y_pred_list) if t == 0 and p == 1),
                "fn": sum(1 for t, p in zip(y_true_list, y_pred_list) if t == 1 and p == 0),
            })
            observations.extend(cm_obs)
    else:
        observations.append("⚠️  No trained model returned - training may have failed")
        recommendations.append("Check agent logs for training errors")

    print_interpretation("Model Training Analysis", observations, recommendations if recommendations else None)

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

    # =========================================================================
    # FINAL RESULT
    # =========================================================================
    duration = time_mod.time() - step_start
    model_usefulness = result.get("model_usefulness", "unknown")

    if model_usefulness == "useless":
        print_step_result("failed", f"Model USELESS - predicts all negatives ({duration:.1f}s)")
    elif model_usefulness == "poor":
        print_step_result("warning", f"Model has poor metrics ({result.get('usefulness_reason', '')}) ({duration:.1f}s)")
    elif model_usefulness == "acceptable":
        print_step_result("success", f"Model trained successfully - usefulness validated ({duration:.1f}s)")
    else:
        print_step_result("warning", f"Model training completed with unknown status ({duration:.1f}s)")

    return result


async def step_6_feature_analyzer(
    experiment_id: str,
    trained_model: Any,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    model_uri: Optional[str] = None
) -> dict[str, Any]:
    """Step 6: Analyze feature importance."""
    import time as time_mod
    step_start = time_mod.time()

    print_header(6, "FEATURE ANALYZER")

    from src.agents.ml_foundation.feature_analyzer import FeatureAnalyzerAgent

    feature_columns = list(X_sample.columns)

    print_input_section({
        "sample_size": len(X_sample),
        "features": feature_columns,
        "max_samples": min(100, len(X_sample)),
        "model_uri": model_uri[:50] + "..." if model_uri and len(model_uri) > 50 else model_uri,
    })

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
            "present" if has_importance else "missing"
        ),
        (
            "Samples analyzed",
            samples_analyzed > 0 if samples_analyzed else False,
            "> 0",
            str(samples_analyzed) if samples_analyzed else "0"
        ),
    ]

    print_validation_checks(checks)

    # Metrics - Feature importance table
    if has_importance:
        print("\n  📊 Feature Importance (SHAP):")
        print(f"    {'Feature':<25} {'Importance':<15} {'Rank':<10}")
        print(f"    {'-'*50}")

        for i, fi in enumerate(result["feature_importance"][:10], 1):
            if isinstance(fi, dict):
                name = fi.get("feature", f"feature_{i}")[:25]
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
            top_names = [fi.get("feature", "unknown") if isinstance(fi, dict) else str(fi) for fi in top_features]
            observations.append(f"Top predictive features: {', '.join(top_names)}")

            # Feature-specific insights
            for fi in top_features:
                if isinstance(fi, dict):
                    name = fi.get("feature", "")
                    imp = fi.get("importance", 0)
                    if "days_on_therapy" in name.lower():
                        observations.append(f"  • Duration on therapy ({imp:.3f}) is a strong predictor")
                    elif "hcp_visits" in name.lower():
                        observations.append(f"  • HCP engagement ({imp:.3f}) influences discontinuation")
                    elif "prior_treatments" in name.lower():
                        observations.append(f"  • Treatment history ({imp:.3f}) affects outcomes")

        observations.append(f"Analysis based on {samples_analyzed} samples using SHAP explainer")
    else:
        observations.append("⚠️  Feature importance analysis failed or skipped")
        if result.get("error"):
            observations.append(f"    Error: {result['error'][:100]}")
        recommendations.append("Verify model is compatible with SHAP explainer")
        recommendations.append("Check if model_uri is valid and accessible")

    print_interpretation("Feature Analysis", observations, recommendations if recommendations else None)

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
    include_bentoml: bool = False,
    fitted_preprocessor: Any = None,
) -> dict[str, Any]:
    """Step 7: Deploy model."""
    import time as time_mod
    step_start = time_mod.time()

    print_header(7, "MODEL DEPLOYER")

    from src.agents.ml_foundation.model_deployer import ModelDeployerAgent

    deployment_name = f"kisqali_discontinuation_{experiment_id[:8]}"

    print_input_section({
        "deployment_name": deployment_name,
        "model_uri": model_uri[:50] + "..." if model_uri and len(model_uri) > 50 else model_uri,
        "success_criteria_met": success_criteria_met,
        "deployment_action": "register",
        "include_bentoml": include_bentoml,
    })

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
    deployment_successful = result.get("deployment_successful", False) or result.get("status") == "completed"
    manifest = result.get("deployment_manifest", {})

    checks = [
        (
            "Deployment successful",
            deployment_successful,
            "deployment_successful = True",
            f"{result.get('status', 'unknown')}"
        ),
        (
            "Deployment manifest",
            bool(manifest),
            "manifest present",
            "present" if manifest else "missing"
        ),
        (
            "Success criteria met",
            success_criteria_met,
            "success_criteria_met = True",
            str(success_criteria_met)
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

    # BentoML Model Serving (optional)
    if include_bentoml and trained_model is not None:
        print("\n  " + "-" * 60)
        print("  BentoML Model Serving:")
        print("  " + "-" * 60)

        try:
            from src.mlops.bentoml_service import register_model_for_serving

            # Detect framework from model class
            model_class_name = type(trained_model).__name__
            if "XGB" in model_class_name:
                framework = "xgboost"
            elif "LGBM" in model_class_name or "LightGBM" in model_class_name:
                framework = "lightgbm"
            else:
                framework = "sklearn"

            model_name = f"tier0_{experiment_id[:8]}"
            print(f"    Registering model: {model_name} (framework: {framework})")

            registration = await register_model_for_serving(
                model=trained_model,
                model_name=model_name,
                metadata={
                    "experiment_id": experiment_id,
                    "validation_metrics": validation_metrics,
                    "tier0_test": True,
                    "algorithm": model_class_name,
                },
                preprocessor=fitted_preprocessor,
                framework=framework,
            )

            if registration.get("registration_status") == "success":
                model_tag = registration.get("model_tag")
                print(f"    ✓ Model registered: {model_tag}")

                # Try persistent service first, fall back to ephemeral
                persistent = await deploy_to_persistent_service(model_tag)
                if persistent.get("available"):
                    endpoint = persistent["endpoint"]
                    print(f"    ✓ Using persistent service at {endpoint}")
                    result["bentoml_persistent"] = True
                else:
                    reason = persistent.get("reason", "unknown")
                    print(f"    ℹ Persistent service not available ({reason}), using ephemeral")
                    # Start BentoML service with the registered model
                    bentoml_result = await start_bentoml_service(
                        model_tag,
                        port=3001,
                        preprocessor=fitted_preprocessor,
                        framework=framework,
                    )

                    if bentoml_result.get("started"):
                        endpoint = bentoml_result.get("endpoint")
                        result["bentoml_pid"] = bentoml_result.get("pid")
                    else:
                        error_msg = bentoml_result.get("error", "Unknown error")
                        print(f"    ✗ BentoML service failed to start: {error_msg}")
                        result["bentoml_serving"] = {"error": error_msg}
                        endpoint = None

                if endpoint:
                    # Verify predictions work with sample data.
                    # Send RAW features — the service applies preprocessing internally.
                    sample_features = [[30.0, 5.0, 1.0]]  # days_on_therapy, hcp_visits, prior_treatments

                    verification = await verify_bentoml_predictions(
                        endpoint=endpoint,
                        sample_features=sample_features,
                    )

                    # Display results
                    print("\n    BentoML Serving Verification:")
                    health_icon = "✓" if verification.get("health_check") else "✗"
                    print(f"      health_check: {health_icon} {'healthy' if verification.get('health_check') else 'unhealthy'}")

                    pred_icon = "✓" if verification.get("prediction_test") else "✗"
                    print(f"      prediction_test: {pred_icon} {'passed' if verification.get('prediction_test') else 'failed'}")
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

                    mode = "persistent" if result.get("bentoml_persistent") else "ephemeral"
                    result["bentoml_serving"] = {
                        "model_tag": model_tag,
                        "endpoint": endpoint,
                        "mode": mode,
                        "health_check": verification.get("health_check"),
                        "prediction_test": verification.get("prediction_test"),
                        "predictions": verification.get("predictions"),
                        "probabilities": verification.get("probabilities"),
                        "latency_ms": verification.get("latency_ms"),
                    }

                    if verification.get("health_check") and verification.get("prediction_test"):
                        print_success(f"Real model deployed and serving verified via BentoML ({mode})")
                    else:
                        print_warning("BentoML serving started but verification incomplete")
            else:
                error_msg = registration.get("error", "Registration failed")
                print(f"    ✗ Model registration failed: {error_msg}")
                result["bentoml_serving"] = {"error": error_msg}

        except ImportError as e:
            print(f"    ✗ BentoML not available: {e}")
            result["bentoml_serving"] = {"error": f"Import error: {e}"}
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
        observations.append(f"Model registered with deployment ID: {manifest.get('deployment_id', 'N/A')}")
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

    print_interpretation("Deployment Analysis", observations, recommendations if recommendations else None)

    # Final result
    duration = time_mod.time() - step_start
    bentoml_ok = bentoml_serving.get("health_check") and bentoml_serving.get("prediction_test") if include_bentoml else True

    if deployment_successful and bentoml_ok:
        print_step_result("success", f"Model deployed successfully ({duration:.1f}s)")
    elif deployment_successful:
        print_step_result("warning", f"Deployment OK but BentoML issues ({duration:.1f}s)")
    else:
        print_step_result("warning", f"Deployment had issues ({duration:.1f}s)")

    return result


async def step_8_observability_connector(experiment_id: str, stages_completed: int) -> dict[str, Any]:
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

    print_input_section({
        "events_to_log": 1,
        "event_type": "pipeline_completed",
        "experiment_id": experiment_id,
        "stages_completed": stages_completed,
        "time_window": "1h",
    })

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
    processing_steps.append(("Event emission", emission_successful, f"{result.get('events_logged', 0)} events"))

    # Feast online feature retrieval check (gracefully degrading)
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
        processing_steps.append(("Feast online retrieval", False, f"skipped: {feast_online_detail}"))

    print_processing_steps(processing_steps)

    # Validation checks
    events_logged = result.get("events_logged", 0)
    quality_score = result.get("quality_score", 0)

    checks = [
        (
            "Emission successful",
            emission_successful,
            "emission_successful = True",
            str(emission_successful)
        ),
        (
            "Events logged",
            events_logged > 0 if events_logged else False,
            "> 0",
            str(events_logged) if events_logged else "0"
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

    print_interpretation("Observability Analysis", observations, recommendations if recommendations else None)

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

async def run_pipeline(
    step: int | None = None,
    dry_run: bool = False,
    imbalance_ratio: float | None = None,
    include_bentoml: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline or a specific step.

    Args:
        step: Run only a specific step (1-8), or None for all steps
        dry_run: Show what would be done without executing
        imbalance_ratio: If provided, create imbalanced data with this minority ratio
        include_bentoml: If True, deploy real model to BentoML and verify predictions

    Returns:
        State dictionary containing all pipeline outputs
    """
    import time

    experiment_id = f"tier0_e2e_{uuid.uuid4().hex[:8]}"
    pipeline_start_time = time.time()

    print(f"\n{'='*70}")
    print(f"TIER 0 MLOPS WORKFLOW TEST")
    print(f"{'='*70}")
    print(f"  Experiment ID: {experiment_id}")
    print(f"  Brand: {CONFIG.brand}")
    print(f"  Target: {CONFIG.target_outcome}")
    print(f"  Problem Type: {CONFIG.problem_type}")
    print(f"  MLflow Enabled: {CONFIG.enable_mlflow}")
    print(f"  MLflow Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI', 'not set')}")
    print(f"  BentoML Serving: {'Enabled' if include_bentoml else 'Disabled'}")
    if imbalance_ratio:
        print(f"  Class Imbalance: {imbalance_ratio:.1%} minority ratio (INJECTED)")
    print(f"  Started: {datetime.now().isoformat()}")

    if dry_run:
        print("\n  [DRY RUN MODE - No agents will be executed]")
        return

    # Generate sample data
    # NOTE: Generate 600 samples to satisfy scope_spec.minimum_samples=500
    # (extra samples account for potential exclusions during cohort construction)
    print("\n  Generating sample patient data...")
    patient_df = generate_sample_data(n_samples=600, imbalance_ratio=imbalance_ratio)
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
            result = await step_1_scope_definer(experiment_id)
            state["scope_spec"] = result.get("scope_spec", {"problem_type": CONFIG.problem_type})
            state["scope_spec"]["experiment_id"] = experiment_id
            duration = time.time() - step_start
            scope_spec = state["scope_spec"]
            success_criteria = result.get("success_criteria", {})
            validation_passed = result.get("validation_passed", True)

            step_results.append(StepResult(
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
                    "problem_description": scope_spec.get("problem_description", "Predict patient discontinuation risk"),
                    "business_objective": scope_spec.get("business_objective", "Identify high-risk patients"),
                    "target_outcome": scope_spec.get("prediction_target", CONFIG.target_outcome),
                    "problem_type_hint": CONFIG.problem_type,
                    "brand": CONFIG.brand,
                },
                validation_checks=[
                    ("Problem type defined", scope_spec.get("problem_type") is not None,
                     "problem_type present", scope_spec.get("problem_type", "None")),
                    ("Prediction target set", scope_spec.get("prediction_target") is not None,
                     "prediction_target present", scope_spec.get("prediction_target", "None")),
                    ("Minimum samples specified", (scope_spec.get("minimum_samples") or 0) > 0,
                     "minimum_samples > 0", scope_spec.get("minimum_samples", 0)),
                    ("Scope validation", validation_passed,
                     "validation_passed = True", f"validation_passed = {validation_passed}"),
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
            ))

        # Step 2: Data Preparer
        if 2 in steps_to_run:
            step_start = time.time()
            scope_spec = state.get("scope_spec", {"problem_type": CONFIG.problem_type})
            result = await step_2_data_preparer(experiment_id, scope_spec, patient_df)
            state["qc_report"] = result.get("qc_report", {"gate_passed": True})
            state["gate_passed"] = result.get("gate_passed", True)

            # Store DataFrames from data_preparer if available
            if result.get("train_df") is not None:
                state["train_df"] = result["train_df"]
            if result.get("validation_df") is not None:
                state["validation_df"] = result["validation_df"]

            qc_report = result.get("qc_report", {})
            data_readiness = result.get("data_readiness", {})
            train_samples = data_readiness.get("train_samples", 0)
            val_samples = data_readiness.get("validation_samples", 0)
            overall_score = qc_report.get("overall_score", 0)
            step_results.append(StepResult(
                step_num=2,
                step_name="DATA PREPARER",
                status="success" if state["gate_passed"] else "failed",
                duration_seconds=time.time() - step_start,
                key_metrics={
                    "qc_status": qc_report.get("status", "unknown"),
                    "overall_score": overall_score,
                    "gate_passed": state["gate_passed"],
                    "train_samples": train_samples,
                    "validation_samples": val_samples,
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
                    "scope_spec_problem_type": scope_spec.get("problem_type", CONFIG.problem_type),
                    "input_samples": len(patient_df),
                },
                validation_checks=[
                    ("QC gate passed", state["gate_passed"], "gate_passed = True", f"gate_passed = {state['gate_passed']}"),
                    ("Overall score acceptable", (overall_score or 0) >= 0.7, "≥ 0.70", f"{overall_score:.2f}" if overall_score else "N/A"),
                    ("Training samples sufficient", train_samples >= 50, "≥ 50", train_samples),
                    ("Validation samples present", val_samples > 0, "> 0", val_samples),
                ],
                metrics_table=[
                    ("overall_score", f"{overall_score:.2f}" if overall_score else "N/A", "≥ 0.70", (overall_score or 0) >= 0.7),
                    ("completeness_score", f"{qc_report.get('completeness_score', 0):.2f}", None, None),
                    ("validity_score", f"{qc_report.get('validity_score', 0):.2f}", None, None),
                    ("consistency_score", f"{qc_report.get('consistency_score', 0):.2f}", None, None),
                    ("train_samples", train_samples, "≥ 50", train_samples >= 50),
                    ("validation_samples", val_samples, "> 0", val_samples > 0),
                ],
                interpretation=[
                    f"Data quality score: {overall_score:.2f}" if overall_score else "Data quality score: N/A",
                    f"Training set: {train_samples} samples, validation set: {val_samples} samples",
                    "QC gate PASSED - data ready for modeling" if state["gate_passed"] else "QC gate FAILED - data quality issues detected",
                ],
                result_message="Data preparation complete" if state["gate_passed"] else "Data preparation failed QC gate",
            ))

            if not state["gate_passed"]:
                print_failure("QC Gate blocked training. Pipeline stopped.")
                return

        # Step 2b: Feast Feature Registration (gracefully degrading)
        if 2 in steps_to_run:
            step_start = time.time()
            feast_reg_result = await step_2b_feast_registration(experiment_id, state)
            feast_reg_status = feast_reg_result.get("status", "skipped")
            features_registered = feast_reg_result.get("features_registered", 0)
            step_results.append(StepResult(
                step_num="2b",
                step_name="FEAST FEATURE REGISTRATION",
                status=feast_reg_status if feast_reg_status != "skipped" else "warning",
                duration_seconds=time.time() - step_start,
                key_metrics={
                    "features_registered": features_registered,
                    "features_skipped": feast_reg_result.get("features_skipped", 0),
                    "feature_group_created": feast_reg_result.get("feature_group_created", False),
                },
                details={"errors": feast_reg_result.get("errors", [])},
                input_summary={
                    "experiment_id": experiment_id,
                    "entity_key": "hcp_id",
                },
                validation_checks=[
                    ("Feast registration attempted", feast_reg_status != "skipped", "not skipped", feast_reg_status),
                    ("Features registered", features_registered > 0, "> 0", str(features_registered)),
                ],
                metrics_table=[
                    ("features_registered", features_registered, "> 0", features_registered > 0),
                    ("status", feast_reg_status, None, None),
                ],
                interpretation=[
                    f"Feast feature registration: {feast_reg_status}",
                    f"{features_registered} features registered to feature store",
                ],
                result_message=f"Feast registration: {feast_reg_status} ({features_registered} features)"
                if feast_reg_status != "skipped" else "Feast registration skipped (service unavailable)",
            ))

        # Step 2c: Feast Freshness Check (gracefully degrading)
        if 2 in steps_to_run:
            step_start = time.time()
            freshness_result = await step_2c_feast_freshness_check(state)
            freshness_status = freshness_result.get("status", "skipped")
            is_fresh = freshness_result.get("fresh", None)
            stale_count = len(freshness_result.get("stale_features", []))
            step_results.append(StepResult(
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
                    ("Freshness check attempted", freshness_status != "skipped", "not skipped", freshness_status),
                    ("Features fresh", is_fresh is True, "all fresh", f"{stale_count} stale" if stale_count else "all fresh"),
                ],
                metrics_table=[
                    ("fresh", str(is_fresh) if is_fresh is not None else "N/A", "True", is_fresh is True),
                    ("stale_features", stale_count, "0", stale_count == 0),
                ],
                interpretation=[
                    f"Feast freshness check: {freshness_status}",
                    f"{'All features fresh' if is_fresh else f'{stale_count} stale features detected'}"
                    if is_fresh is not None else "Freshness check was skipped",
                ],
                result_message=f"Freshness: {'all fresh' if is_fresh else f'{stale_count} stale'}"
                if freshness_status != "skipped" else "Freshness check skipped (service unavailable)",
            ))

        # Step 3: Cohort Constructor
        if 3 in steps_to_run:
            step_start = time.time()
            eligible_df, cohort_result = await step_3_cohort_constructor(patient_df)
            state["eligible_df"] = eligible_df
            state["cohort_result"] = cohort_result
            input_count = len(patient_df)
            eligible_count = len(eligible_df)
            excluded_count = input_count - eligible_count
            exclusion_rate = excluded_count / input_count if input_count > 0 else 0
            step_results.append(StepResult(
                step_num=3,
                step_name="COHORT CONSTRUCTOR",
                status="success" if eligible_count >= CONFIG.min_eligible_patients else "warning",
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
                    ("Sufficient eligible patients", eligible_count >= CONFIG.min_eligible_patients,
                     f"≥ {CONFIG.min_eligible_patients}", eligible_count),
                    ("Exclusion rate reasonable", exclusion_rate <= 0.5,
                     "≤ 50%", f"{exclusion_rate:.1%}"),
                    ("Cohort ID generated", cohort_result.cohort_id is not None,
                     "cohort_id present", cohort_result.cohort_id or "None"),
                    ("Cohort status valid", cohort_result.status in ["completed", "success"],
                     "completed/success", cohort_result.status),
                ],
                metrics_table=[
                    ("input_patients", input_count, None, None),
                    ("eligible_patients", eligible_count, f"≥ {CONFIG.min_eligible_patients}", eligible_count >= CONFIG.min_eligible_patients),
                    ("excluded_patients", excluded_count, None, None),
                    ("exclusion_rate", f"{exclusion_rate:.1%}", "≤ 50%", exclusion_rate <= 0.5),
                ],
                interpretation=[
                    f"Cohort constructed with {eligible_count} eligible patients from {input_count} total",
                    f"Exclusion rate: {exclusion_rate:.1%} ({excluded_count} patients excluded)",
                    f"Cohort size {'meets' if eligible_count >= CONFIG.min_eligible_patients else 'below'} minimum threshold of {CONFIG.min_eligible_patients}",
                ],
                result_message=f"Cohort '{cohort_result.cohort_id}' constructed with {eligible_count} patients",
            ))

        # Step 4: Model Selector
        if 4 in steps_to_run:
            step_start = time.time()
            scope_spec = state.get("scope_spec", {"problem_type": CONFIG.problem_type})
            qc_report = state.get("qc_report", {"gate_passed": True})
            result = await step_4_model_selector(experiment_id, scope_spec, qc_report)
            state["model_candidate"] = result.get("model_candidate") or result.get("primary_candidate")

            candidate = state["model_candidate"]
            algo_name = candidate.get("algorithm_name") if isinstance(candidate, dict) else getattr(candidate, "algorithm_name", "Unknown")
            # Extract selection_score from model_candidate (not selection_rationale)
            selection_score = candidate.get("selection_score", 0) if isinstance(candidate, dict) else 0
            # Use default_hyperparameters from agent output
            hyperparams = candidate.get("default_hyperparameters", {}) if isinstance(candidate, dict) else {}
            interpretability = candidate.get("interpretability_score", 0) if isinstance(candidate, dict) else 0

            # Extract selection rationale details
            selection_rationale = result.get("selection_rationale", {})
            primary_reason = selection_rationale.get("primary_reason", "") if isinstance(selection_rationale, dict) else ""
            supporting_factors = selection_rationale.get("supporting_factors", []) if isinstance(selection_rationale, dict) else []

            # Extract alternative candidates
            alternatives = result.get("alternative_candidates", [])
            alternatives_considered = selection_rationale.get("alternatives_considered", []) if isinstance(selection_rationale, dict) else []
            all_alternatives = alternatives if alternatives else alternatives_considered

            step_results.append(StepResult(
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
                    ("Algorithm selected", candidate is not None, "candidate present", algo_name or "None"),
                    ("Selection score computed", selection_score > 0, "> 0", f"{selection_score:.3f}" if selection_score else "N/A"),
                    ("Rationale provided", bool(primary_reason), "reason present", primary_reason[:30] if primary_reason else "None"),
                    ("Alternatives evaluated", len(all_alternatives) > 0, "> 0", f"{len(all_alternatives)} candidates"),
                ],
                metrics_table=[
                    ("algorithm", algo_name, None, None),
                    ("selection_score", f"{selection_score:.3f}" if selection_score else "N/A", "> 0.5", selection_score > 0.5 if selection_score else None),
                    ("interpretability", f"{interpretability:.2f}" if interpretability else "N/A", None, None),
                    ("default_hyperparameters", len(hyperparams), None, None),
                    ("alternatives_evaluated", len(all_alternatives), "> 0", len(all_alternatives) > 0),
                ],
                interpretation=[
                    f"Selected {algo_name} (score: {selection_score:.3f})" if selection_score else f"Selected {algo_name}",
                    f"Reason: {primary_reason}" if primary_reason else "Selection based on problem type and data characteristics",
                    f"Evaluated {len(all_alternatives)} alternative{'s' if len(all_alternatives) != 1 else ''}: {', '.join([a.get('algorithm_name', str(a)) if isinstance(a, dict) else str(a) for a in all_alternatives[:3]])}" if all_alternatives else "No alternatives evaluated",
                    f"HPO will tune {len(hyperparams)} hyperparameters in Step 5" if hyperparams else "HPO will use default search space",
                ],
                result_message=f"Model selection complete: {algo_name} (score={selection_score:.3f})" if selection_score else f"Model selection complete: {algo_name}",
            ))

        # Step 5: Model Trainer
        if 5 in steps_to_run:
            step_start = time.time()
            eligible_df = state.get("eligible_df", patient_df)
            # Use numeric features for training
            feature_cols = ["days_on_therapy", "hcp_visits", "prior_treatments"]
            X = eligible_df[feature_cols].copy()
            y = eligible_df[CONFIG.target_outcome].copy()

            model_candidate = state.get("model_candidate", {
                "algorithm_name": "LogisticRegression",
                "hyperparameters": {"C": 1.0, "max_iter": 100}
            })
            qc_report = state.get("qc_report", {"gate_passed": True})

            result = await step_5_model_trainer(
                experiment_id, model_candidate, qc_report, X, y
            )
            state["trained_model"] = result.get("trained_model")
            state["validation_metrics"] = result.get("validation_metrics", {})
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
            state["success_criteria_met"] = result.get("success_criteria_met", True)

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
                new_minority_ratio = min(resampled_dist.values()) / total_resampled if total_resampled > 0 else None
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
            step_results.append(StepResult(
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
                    "algorithm": model_candidate.get("algorithm_name") if isinstance(model_candidate, dict) else "Unknown",
                    "training_samples": len(X),
                    "features": list(X.columns),
                    "target": CONFIG.target_outcome,
                },
                validation_checks=[
                    ("AUC-ROC above threshold", auc_roc >= 0.6, "≥ 0.60", f"{auc_roc:.3f}" if auc_roc else "N/A"),
                    ("Model not useless", model_usefulness != "useless", "not useless", model_usefulness),
                    ("Success criteria met", success_met, "True", str(success_met)),
                    ("Both classes predicted", model_usefulness not in ["useless", "poor"], "multi-class output", model_usefulness),
                ],
                metrics_table=[
                    ("auc_roc", f"{auc_roc:.3f}" if auc_roc else "N/A", "≥ 0.60", auc_roc >= 0.6 if auc_roc else False),
                    ("precision", f"{precision:.3f}" if precision else "N/A", None, None),
                    ("recall", f"{recall:.3f}" if recall else "N/A", None, None),
                    ("f1_score", f"{f1:.3f}" if f1 else "N/A", None, None),
                    ("model_usefulness", model_usefulness, "good/acceptable", model_usefulness in ["good", "acceptable"]),
                    ("imbalance_detected", imbalance_detected, None, None),
                    ("resampling_applied", resampling_applied, None, None),
                ],
                interpretation=[
                    f"Model trained with AUC-ROC: {auc_roc:.3f}" if auc_roc else "Model training completed",
                    f"Model usefulness: {model_usefulness}" + (f" - {result.get('usefulness_reason', '')}" if result.get('usefulness_reason') else ""),
                    f"Class imbalance {'detected and remediated via ' + result.get('recommended_strategy', 'resampling') if imbalance_detected else 'not detected'}",
                    f"Success criteria {'MET' if success_met else 'NOT MET'}",
                ],
                result_message=f"Training complete: {model_usefulness} model with AUC={auc_roc:.3f}" if auc_roc else "Training complete",
            ))

        # Step 6: Feature Analyzer
        if 6 in steps_to_run:
            step_start = time.time()
            eligible_df = state.get("eligible_df", patient_df)
            # Use numeric features for analysis
            feature_cols = ["days_on_therapy", "hcp_visits", "prior_treatments"]
            X = eligible_df[feature_cols].copy()
            y = eligible_df[CONFIG.target_outcome].copy()

            result = await step_6_feature_analyzer(
                experiment_id,
                state.get("trained_model"),
                X.iloc[:50],
                y.iloc[:50],
                model_uri=state.get("model_uri")
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
            top_feature_importance = top_features.get(top_feature_name, 0) if top_feature_name else 0
            step_results.append(StepResult(
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
                    ("Feature importance computed", result.get("feature_importance") is not None, "importance present", "Yes" if result.get("feature_importance") else "No"),
                    ("Sufficient samples analyzed", samples_analyzed >= 10, "≥ 10", samples_analyzed),
                    ("Computation completed", compute_time > 0, "compute_time > 0", f"{compute_time:.2f}s"),
                ],
                metrics_table=[
                    ("samples_analyzed", samples_analyzed, "≥ 10", samples_analyzed >= 10),
                    ("computation_time", f"{compute_time:.2f}s", None, None),
                    ("explainer_type", explainer_type, None, None),
                    ("top_feature", top_feature_name or "N/A", None, None),
                    ("top_importance", f"{top_feature_importance:.3f}" if top_feature_importance else "N/A", None, None),
                ],
                interpretation=[
                    f"SHAP analysis completed on {samples_analyzed} samples in {compute_time:.2f}s",
                    f"Top driver: {top_feature_name} (importance: {top_feature_importance:.3f})" if top_feature_name else "No dominant feature identified",
                    f"Feature ranking: {', '.join(list(top_features.keys())[:3])}" if len(top_features) >= 3 else f"Features analyzed: {len(top_features)}",
                ],
                result_message=f"Feature analysis complete: {top_feature_name} is top predictor" if top_feature_name else "Feature analysis complete",
            ))

        # Step 7: Model Deployer
        if 7 in steps_to_run:
            step_start = time.time()
            result = await step_7_model_deployer(
                experiment_id,
                state.get("model_uri"),
                state.get("validation_metrics", {}),
                state.get("success_criteria_met", True),
                trained_model=state.get("trained_model"),
                include_bentoml=include_bentoml,
                fitted_preprocessor=state.get("fitted_preprocessor"),
            )
            state["deployment_manifest"] = result.get("deployment_manifest")
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
            bentoml_verified = bentoml_serving.get("prediction_test", False) if bentoml_serving else None
            step_results.append(StepResult(
                step_num=7,
                step_name="MODEL DEPLOYER",
                status="success" if deployment_successful else "warning",
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
                    "success_criteria_met": state.get("success_criteria_met", True),
                    "include_bentoml": include_bentoml,
                },
                validation_checks=[
                    ("Deployment successful", deployment_successful, "True", str(deployment_successful)),
                    ("Deployment ID assigned", deployment_id != "N/A", "ID present", deployment_id),
                    ("Environment set", environment is not None, "env specified", environment),
                    ("BentoML verified", bentoml_verified if include_bentoml else True, "True", str(bentoml_verified) if include_bentoml else "N/A (not enabled)"),
                ],
                metrics_table=[
                    ("deployment_id", deployment_id, None, None),
                    ("environment", environment, None, None),
                    ("status", deployment_status, "deployed", deployment_status == "deployed"),
                    ("bentoml_verified", str(bentoml_verified) if include_bentoml else "N/A", None, None),
                    ("latency_ms", f"{bentoml_serving.get('latency_ms', 'N/A')}" if bentoml_serving else "N/A", None, None),
                ],
                interpretation=[
                    f"Model deployed to {environment} environment" if deployment_successful else "Deployment pending or failed",
                    f"Deployment ID: {deployment_id}",
                    f"BentoML serving {'verified with live prediction test' if bentoml_verified else 'not verified' if include_bentoml else 'not enabled'}",
                ],
                result_message=f"Deployment complete: {deployment_id} to {environment}" if deployment_successful else "Deployment incomplete",
            ))

        # Step 8: Observability Connector
        if 8 in steps_to_run:
            step_start = time.time()
            result = await step_8_observability_connector(
                experiment_id,
                len(steps_to_run)
            )
            emission_successful = result.get("emission_successful", False)
            events_logged = result.get("events_logged", 0)
            quality_score = result.get("quality_score", 0)
            step_results.append(StepResult(
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
                    ("Quality score computed", quality_score is not None, "present", f"{quality_score:.2f}" if quality_score else "N/A"),
                    ("Feast online retrieval", result.get("feast_online_ok", False), "accessible", result.get("feast_online_detail", "skipped")),
                ],
                metrics_table=[
                    ("emission_successful", str(emission_successful), "True", emission_successful),
                    ("events_logged", events_logged, "> 0", events_logged > 0),
                    ("quality_score", f"{quality_score:.2f}" if quality_score else "N/A", None, None),
                ],
                interpretation=[
                    f"Observability metrics {'successfully' if emission_successful else 'NOT'} emitted to monitoring systems",
                    f"{events_logged} events logged for pipeline tracking",
                    f"Overall pipeline quality score: {quality_score:.2f}" if quality_score else "Quality score not computed",
                ],
                result_message=f"Observability complete: {events_logged} events logged" if emission_successful else "Observability emission incomplete",
            ))

        # Print detailed step results
        print_detailed_summary(experiment_id, step_results, state)

        # Final summary
        pipeline_duration = time.time() - pipeline_start_time

        # Determine overall pipeline status
        failed_steps = [r for r in step_results if r.status == "failed"]
        warning_steps = [r for r in step_results if r.status == "warning"]

        print(f"\n{'='*70}")
        print("PIPELINE SUMMARY")
        print(f"{'='*70}")
        print(f"  Experiment ID: {experiment_id}")
        print(f"  Steps Completed: {len(steps_to_run)}")
        print(f"  Total Duration: {pipeline_duration:.1f}s")
        print(f"  QC Gate: {'PASSED' if state.get('gate_passed', True) else 'FAILED'}")
        if state.get("eligible_df") is not None:
            print(f"  Cohort Size: {len(state['eligible_df'])}")
        if state.get("validation_metrics"):
            print(f"  Validation Metrics: {state['validation_metrics']}")
        if include_bentoml and state.get("bentoml_pid"):
            print(f"  BentoML Serving: Verified (PID: {state['bentoml_pid']})")
        print(f"  Completed: {datetime.now().isoformat()}")

        # Print step status summary
        success_count = len([r for r in step_results if r.status == "success"])
        print(f"\n  Step Status: {success_count} success, {len(warning_steps)} warnings, {len(failed_steps)} failed")

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
        # Cleanup BentoML service if started (skip if using persistent systemd service)
        if state.get("bentoml_pid") and not state.get("bentoml_persistent"):
            print("\n  Cleaning up ephemeral BentoML service...")
            cleanup_result = await stop_bentoml_service(state["bentoml_pid"])
            if cleanup_result.get("stopped"):
                print(f"    ✓ BentoML service stopped (PID: {state['bentoml_pid']})")
            else:
                print(f"    ⚠️  BentoML cleanup issue: {cleanup_result.get('error', 'unknown')}")
        elif state.get("bentoml_persistent"):
            print("\n  Persistent BentoML service left running (e2i-bentoml.service)")

    return state


def main():
    """Main entry point."""
    import sys
    import io
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Run Tier 0 MLOps workflow test"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=range(1, 9),
        help="Run only a specific step (1-8)"
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow tracking (enabled by default for model_uri generation)"
    )
    parser.add_argument(
        "--enable-opik",
        action="store_true",
        help="Enable Opik tracing"
    )
    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=10,
        help="Number of HPO trials (default: 10)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--imbalanced",
        type=float,
        default=None,
        metavar="RATIO",
        help="Create imbalanced data with specified minority ratio (e.g., 0.1 for 10%% minority class)"
    )
    parser.add_argument(
        "--include-bentoml",
        action="store_true",
        help="Include BentoML model serving verification with the real trained model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/results",
        help="Directory to save results MD file (default: docs/results)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to file (only print to console)"
    )

    args = parser.parse_args()

    # Update config
    if args.disable_mlflow:
        CONFIG.enable_mlflow = False
    CONFIG.enable_opik = args.enable_opik
    CONFIG.hpo_trials = args.hpo_trials

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
        asyncio.run(run_pipeline(
            step=args.step,
            dry_run=args.dry_run,
            imbalance_ratio=args.imbalanced,
            include_bentoml=args.include_bentoml,
        ))
    finally:
        # Restore stdout
        sys.stdout = original_stdout

        # Save results to file
        if not args.no_save and output_buffer:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(args.output_dir) / f"tier0_pipeline_run_{timestamp}.md"

            # Add markdown header
            md_content = f"# Tier 0 Pipeline Run Results\n\n"
            md_content += f"**Generated**: {datetime.now().isoformat()}\n\n"
            md_content += "```\n"
            md_content += output_buffer.getvalue()
            md_content += "```\n"

            with open(output_file, "w") as f:
                f.write(md_content)

            print(f"\n📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
