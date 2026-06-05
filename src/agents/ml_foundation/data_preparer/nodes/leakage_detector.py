"""Leakage detector node for data_preparer agent.

This node detects multiple types of data leakage:
1. Temporal leakage: Future data leaking into training
2. Target leakage: Features that are derived from the target
3. Train-test contamination: Overlapping samples between splits
4. Perfect class separation: Features that perfectly separate target classes
5. Zero variance within class: Features with no variance in one or both classes
6. Mutual information: Features with implausibly high MI with target
7. Logical dependency: Features that are tautologically equivalent to target
8. Single-feature AUC: Individual features that nearly predict the target alone
9. Categorical class separation: Categorical features with high Cramér's V
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..state import DataPreparerState

logger = logging.getLogger(__name__)


# =============================================================================
# STRUCTURED FINDINGS
# =============================================================================


class LeakageSeverity(str, Enum):
    """Severity levels for leakage findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    INFO = "info"


@dataclass
class LeakageFinding:
    """Structured leakage detection finding."""

    check_name: str
    severity: LeakageSeverity
    feature: str
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d

    def to_issue_string(self) -> str:
        return f"[{self.severity.value.upper()}] {self.check_name}: {self.description}"


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


async def detect_leakage(state: DataPreparerState) -> Dict[str, Any]:
    """Detect data leakage across multiple types.

    This node checks for:
    1. Temporal leakage: event_date > target_date in any row
    2. Target leakage: Features with suspiciously high correlation with target
    3. Train-test contamination: Duplicate samples across splits
    4. Perfect class separation: Features that perfectly separate classes
    5. Zero variance within class: Degenerate feature distributions
    6. Mutual information: Implausibly high MI with target
    7. Logical dependency: Tautological feature-target relationships

    Args:
        state: Current agent state

    Returns:
        Updated state with leakage detection results
    """
    if state.get("skip_leakage_check", False):
        logger.warning("Leakage detection skipped per configuration")
        return {
            "leakage_detected": False,
            "leakage_issues": ["Leakage check skipped (not recommended)"],
            "leakage_findings": [],
            "leakage_severity": "none",
            "leaked_features": [],
        }

    logger.info(f"Running leakage detection for experiment {state['experiment_id']}")

    leakage_issues: List[str] = []
    findings: List[LeakageFinding] = []

    try:
        train_df = state.get("train_df")
        validation_df = state.get("validation_df")
        test_df = state.get("test_df")
        holdout_df = state.get("holdout_df")

        if train_df is None:
            raise ValueError("train_df not found in state")

        scope_spec = state.get("scope_spec", {})
        target_variable = scope_spec.get("prediction_target")
        required_features = scope_spec.get("required_features", [])

        # === 1. TEMPORAL LEAKAGE ===
        temporal_issues = check_temporal_leakage(train_df, scope_spec)
        leakage_issues.extend(temporal_issues)

        # === 2. TARGET LEAKAGE (enhanced with pointbiserialr) ===
        if target_variable and target_variable in train_df.columns:
            target_leakage_issues, target_findings = check_target_leakage(
                train_df, target_variable, required_features
            )
            leakage_issues.extend(target_leakage_issues)
            findings.extend(target_findings)

        # === 3. TRAIN-TEST CONTAMINATION ===
        contamination_issues = check_train_test_contamination(
            train_df, validation_df, test_df, holdout_df
        )
        leakage_issues.extend(contamination_issues)

        # === 4-7. NEW STRUCTURAL CHECKS ===
        if target_variable and target_variable in train_df.columns:
            # Combine train + validation for structural checks (more data = more reliable)
            combined_df = train_df
            if validation_df is not None and target_variable in validation_df.columns:
                combined_df = pd.concat([train_df, validation_df], ignore_index=True)

            # Honor scope_spec["excluded_features"] (PII, temporal leakage,
            # pipeline construction metadata) so the structural checks don't fire
            # on columns the pipeline is already committed to dropping.
            excluded_features = scope_spec.get("excluded_features", []) or []
            cols_to_drop = [
                c for c in excluded_features if c in combined_df.columns and c != target_variable
            ]
            if cols_to_drop:
                combined_df = combined_df.drop(columns=cols_to_drop)

            numeric_features = _get_numeric_features(
                combined_df, target_variable, required_features
            )

            if len(numeric_features) > 0 and len(combined_df) >= 30:
                # 4. Perfect class separation
                separation_findings = check_perfect_class_separation(
                    combined_df, target_variable, numeric_features
                )
                findings.extend(separation_findings)

                # 5. Zero variance within class
                variance_findings = check_zero_variance_within_class(
                    combined_df, target_variable, numeric_features
                )
                findings.extend(variance_findings)

                # 6. Mutual information
                mi_findings = check_mutual_information(
                    combined_df, target_variable, numeric_features
                )
                findings.extend(mi_findings)

                # 7. Logical dependency
                dependency_findings = check_feature_target_logical_dependency(
                    combined_df, target_variable, numeric_features
                )
                findings.extend(dependency_findings)

                # 8. Single-feature AUC
                auc_findings = check_single_feature_auc(
                    combined_df, target_variable, numeric_features
                )
                findings.extend(auc_findings)

            # 9. Categorical class separation (Cramér's V)
            categorical_features = _get_categorical_features(
                combined_df, target_variable, required_features
            )
            if len(categorical_features) > 0 and len(combined_df) >= 30:
                cat_findings = check_categorical_class_separation(
                    combined_df, target_variable, categorical_features
                )
                findings.extend(cat_findings)

        # Convert findings to issue strings
        for f in findings:
            leakage_issues.append(f.to_issue_string())

        # Determine overall severity
        severity = _aggregate_severity(findings)
        leaked_features = _get_leaked_features(findings)

        # Determine if leakage detected
        leakage_detected = len(leakage_issues) > 0

        # Add to blocking issues if CRITICAL or HIGH findings
        blocking_updates: Dict[str, Any] = {}
        blocking_findings = [
            f for f in findings if f.severity in (LeakageSeverity.CRITICAL, LeakageSeverity.HIGH)
        ]
        if blocking_findings or (leakage_detected and not findings):
            # Legacy leakage issues (temporal, contamination) also block
            existing_blocking = state.get("blocking_issues") or []
            new_blocking = [f.to_issue_string() for f in blocking_findings]
            # Also add legacy string issues that aren't from findings
            legacy_issues = [
                i for i in leakage_issues if not any(i == f.to_issue_string() for f in findings)
            ]
            blocking_updates["blocking_issues"] = existing_blocking + new_blocking + legacy_issues

        logger.info(
            f"Leakage detection completed: "
            f"detected={leakage_detected}, issues={len(leakage_issues)}, "
            f"severity={severity}, leaked_features={leaked_features}"
        )

        return {
            "leakage_detected": leakage_detected,
            "leakage_issues": leakage_issues,
            "leakage_findings": [f.to_dict() for f in findings],
            "leakage_severity": severity,
            "leaked_features": leaked_features,
            **blocking_updates,
        }

    except Exception as e:
        logger.error(f"Leakage detection failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_type": "leakage_detection_error",
            "leakage_detected": True,  # Assume worst case
            "leakage_issues": [f"Leakage detection error: {str(e)}"],
            "leakage_findings": [],
            "leakage_severity": "critical",
            "leaked_features": [],
        }


# =============================================================================
# HELPER: GET NUMERIC FEATURES
# =============================================================================


def _get_numeric_features(df: Any, target_variable: str, required_features: List[str]) -> List[str]:
    """Get numeric feature columns (excluding target)."""
    features = []
    candidates = (
        required_features if required_features else [c for c in df.columns if c != target_variable]
    )
    for col in candidates:
        if col == target_variable or col not in df.columns:
            continue
        if np.issubdtype(df[col].dtype, np.number):
            features.append(col)
    return features


def _get_categorical_features(
    df: Any, target_variable: str, required_features: List[str]
) -> List[str]:
    """Get categorical/object feature columns (excluding target)."""
    features = []
    candidates = (
        required_features if required_features else [c for c in df.columns if c != target_variable]
    )
    for col in candidates:
        if col == target_variable or col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            features.append(col)
    return features


def _aggregate_severity(findings: List[LeakageFinding]) -> str:
    """Get the highest severity from findings."""
    if not findings:
        return "none"
    priority = [
        LeakageSeverity.CRITICAL,
        LeakageSeverity.HIGH,
        LeakageSeverity.MODERATE,
        LeakageSeverity.INFO,
    ]
    for level in priority:
        if any(f.severity == level for f in findings):
            return level.value
    return "none"


def _get_leaked_features(findings: List[LeakageFinding]) -> List[str]:
    """Get feature names flagged at CRITICAL or HIGH severity."""
    leaked = set()
    for f in findings:
        if f.severity in (LeakageSeverity.CRITICAL, LeakageSeverity.HIGH) and f.feature:
            leaked.add(f.feature)
    return sorted(leaked)


# =============================================================================
# CHECK 1: TEMPORAL LEAKAGE (existing)
# =============================================================================


def check_temporal_leakage(df: Any, scope_spec: Dict[str, Any]) -> List[str]:
    """Check for temporal leakage.

    Temporal leakage occurs when event timestamps are after target timestamps,
    or when features contain information from the future relative to prediction time.

    Detection strategies:
    1. Explicit: event_date_column vs target_date_column comparison
    2. Split-based: feature dates vs split_date (prediction boundary)
    3. Generic: auto-detect date columns and check for future data

    Args:
        df: DataFrame to check
        scope_spec: Scope specification with temporal column hints

    Returns:
        List of temporal leakage issues
    """
    issues: List[str] = []

    if df is None or len(df) == 0:
        return issues

    try:
        # Strategy 1: Explicit event_date vs target_date comparison
        event_date_col = scope_spec.get("event_date_column")
        target_date_col = scope_spec.get("target_date_column")

        if event_date_col and target_date_col:
            if event_date_col in df.columns and target_date_col in df.columns:
                leakage_count, leakage_pct = _check_date_ordering(
                    df, event_date_col, target_date_col
                )
                if leakage_count > 0:
                    issues.append(
                        f"Temporal leakage: {leakage_count} rows ({leakage_pct:.2f}%) "
                        f"have {event_date_col} > {target_date_col}"
                    )

        # Strategy 2: Check feature date columns against split_date
        split_date_str = scope_spec.get("split_date")
        feature_date_columns = scope_spec.get("feature_date_columns", [])

        if split_date_str and feature_date_columns:
            split_date = _parse_date(split_date_str)
            if split_date:
                for col in feature_date_columns:
                    if col in df.columns:
                        future_count, future_pct = _check_future_dates(df, col, split_date)
                        if future_count > 0:
                            issues.append(
                                f"Temporal leakage: {future_count} rows ({future_pct:.2f}%) "
                                f"in '{col}' have dates after split_date ({split_date_str})"
                            )

        # Strategy 3: Generic auto-detection of date columns
        date_column = scope_spec.get("date_column")
        if split_date_str and date_column:
            split_date = _parse_date(split_date_str)
            if split_date:
                # Find all date-like columns (excluding the main date column)
                date_cols = _detect_date_columns(df, exclude=[date_column])
                for col in date_cols:
                    future_count, future_pct = _check_future_dates(df, col, split_date)
                    if future_count > 0:
                        issues.append(
                            f"Potential temporal leakage: {future_count} rows ({future_pct:.2f}%) "
                            f"in auto-detected date column '{col}' have dates after split_date"
                        )

    except Exception as e:
        logger.warning(f"Temporal leakage check failed: {e}")
        issues.append(f"Temporal leakage check incomplete: {str(e)}")

    return issues


# =============================================================================
# CHECK 2: TARGET LEAKAGE (enhanced with pointbiserialr)
# =============================================================================


def check_target_leakage(
    df: Any, target_variable: str, features: List[str]
) -> tuple[List[str], List[LeakageFinding]]:
    """Check for target leakage using point-biserial correlation.

    Enhanced: uses scipy.stats.pointbiserialr for binary targets (returns p-value),
    falls back to Pearson for continuous. Threshold lowered from 0.95 to 0.85.

    Args:
        df: DataFrame to check
        target_variable: Name of target variable
        features: List of feature names

    Returns:
        Tuple of (legacy issue strings, structured findings)
    """
    issues: List[str] = []
    findings: List[LeakageFinding] = []

    try:
        target_data = df[target_variable]
        is_binary = set(target_data.dropna().unique()).issubset({0, 1})

        for feature in features:
            if feature not in df.columns:
                continue

            feature_data = df[feature]

            # Skip non-numeric
            if not np.issubdtype(feature_data.dtype, np.number):
                continue
            if not np.issubdtype(target_data.dtype, np.number):
                continue

            # Drop NaN pairs
            valid_mask = feature_data.notna() & target_data.notna()
            feat_valid = feature_data[valid_mask]
            tgt_valid = target_data[valid_mask]

            if len(feat_valid) < 10:
                continue

            if is_binary:
                try:
                    from scipy.stats import pointbiserialr

                    corr, p_value = pointbiserialr(tgt_valid.values, feat_valid.values)
                except Exception:
                    corr = feat_valid.corr(tgt_valid)
                    p_value = None
            else:
                corr = feat_valid.corr(tgt_valid)
                p_value = None

            abs_corr = abs(corr) if not np.isnan(corr) else 0

            # Flag at thresholds
            if abs_corr > 0.85:
                severity = LeakageSeverity.CRITICAL if abs_corr > 0.95 else LeakageSeverity.HIGH
                findings.append(
                    LeakageFinding(
                        check_name="target_correlation",
                        severity=severity,
                        feature=feature,
                        description=(
                            f"Feature '{feature}' has correlation {corr:.3f} with target "
                            f"(p={p_value:.2e})"
                            if p_value is not None
                            else f"Feature '{feature}' has correlation {corr:.3f} with target"
                        ),
                        evidence={
                            "correlation": float(corr),
                            "p_value": float(p_value) if p_value else None,
                        },
                        recommendation=f"Investigate whether '{feature}' is derived from or encodes the target",
                    )
                )
                issues.append(
                    f"Potential target leakage: feature '{feature}' has "
                    f"correlation {corr:.3f} with target (threshold: 0.85)"
                )
            elif abs_corr > 0.70 and p_value is not None and p_value < 0.001:
                findings.append(
                    LeakageFinding(
                        check_name="target_correlation",
                        severity=LeakageSeverity.MODERATE,
                        feature=feature,
                        description=(
                            f"Feature '{feature}' has statistically significant correlation "
                            f"{corr:.3f} with target (p={p_value:.2e})"
                        ),
                        evidence={"correlation": float(corr), "p_value": float(p_value)},
                        recommendation=f"Review '{feature}' for potential target leakage",
                    )
                )

    except Exception as e:
        logger.warning(f"Target leakage check failed: {e}")

    return issues, findings


# =============================================================================
# CHECK 3: TRAIN-TEST CONTAMINATION (existing)
# =============================================================================


def check_train_test_contamination(
    train_df: Any,
    validation_df: Any = None,
    test_df: Any = None,
    holdout_df: Any = None,
) -> List[str]:
    """Check for train-test contamination.

    Contamination occurs when the same samples appear in multiple splits.
    This function prefers using a unique identifier column (like patient_journey_id
    or patient_id) over DataFrame index, since index-based checking gives false
    positives when DataFrames have been reset with sequential indices.

    Args:
        train_df: Training DataFrame
        validation_df: Validation DataFrame (optional)
        test_df: Test DataFrame (optional)
        holdout_df: Holdout DataFrame (optional)

    Returns:
        List of contamination issues
    """
    issues = []

    try:
        # Identify unique identifier column for more accurate contamination checking
        # Priority: patient_journey_id > patient_id > id > index
        id_column = None
        for candidate in ["patient_journey_id", "patient_id", "id"]:
            if candidate in train_df.columns:
                id_column = candidate
                break

        splits = {
            "validation": validation_df,
            "test": test_df,
            "holdout": holdout_df,
        }

        for split_name, split_df in splits.items():
            if split_df is None:
                continue

            if id_column and id_column in split_df.columns:
                # Use unique identifier column for comparison
                train_ids = set(train_df[id_column].astype(str))
                split_ids = set(split_df[id_column].astype(str))
                overlap = train_ids.intersection(split_ids)
            else:
                # Fallback to row hash comparison (more reliable than index)
                # Create hash from all columns to identify unique rows
                train_hashes = set(train_df.apply(lambda row: hash(tuple(row)), axis=1))
                split_hashes = set(split_df.apply(lambda row: hash(tuple(row)), axis=1))
                overlap = train_hashes.intersection(split_hashes)

            if len(overlap) > 0:
                overlap_pct = len(overlap) / len(train_df) * 100
                issues.append(
                    f"Train-{split_name} contamination: {len(overlap)} samples "
                    f"({overlap_pct:.2f}%) overlap between splits"
                )

    except Exception as e:
        logger.warning(f"Train-test contamination check failed: {e}")

    return issues


# =============================================================================
# CHECK 4: PERFECT CLASS SEPARATION
# =============================================================================


def check_perfect_class_separation(
    df: Any, target_variable: str, numeric_features: List[str]
) -> List[LeakageFinding]:
    """Check if any feature perfectly separates target classes.

    This is the single highest-impact check — it directly catches the CSU scenario
    where features like days_on_therapy have zero overlap between classes.

    For each numeric feature:
    1. Compute value ranges per target class: (min_0, max_0) and (min_1, max_1)
    2. Compute overlap fraction
    3. If overlap < 1% → CRITICAL

    Args:
        df: DataFrame to check
        target_variable: Target column name
        numeric_features: List of numeric feature column names

    Returns:
        List of LeakageFinding objects
    """
    findings: List[LeakageFinding] = []
    target = df[target_variable]

    for feature in numeric_features:
        try:
            feat = df[feature]
            valid = feat.notna() & target.notna()
            feat_valid = feat[valid]
            tgt_valid = target[valid]

            class_0 = feat_valid[tgt_valid == 0]
            class_1 = feat_valid[tgt_valid == 1]

            if len(class_0) < 5 or len(class_1) < 5:
                continue

            # Rare-event guard for binary/low-cardinality features.
            # With a small positive class, a binary feature where all positives
            # happen to be 0 (or all 1) trivially produces overlap_fraction=0
            # even when the feature is a legitimate pre-index predictor. Skip
            # the check when (a) feature is binary/near-binary (≤2 unique
            # values in the combined valid set) AND (b) positive class is
            # small (n < 30 or positive rate < 5%).
            n_unique = feat_valid.nunique()
            pos_rate = len(class_1) / max(len(feat_valid), 1)
            if n_unique <= 2 and (len(class_1) < 30 or pos_rate < 0.05):
                logger.debug(
                    f"Skipping perfect_class_separation for binary feature '{feature}' — "
                    f"rare-event cohort (n_pos={len(class_1)}, pos_rate={pos_rate:.2%})"
                )
                continue

            min_0, max_0 = float(class_0.min()), float(class_0.max())
            min_1, max_1 = float(class_1.min()), float(class_1.max())

            # Check for all-zero vs all-nonzero pattern
            class_0_all_zero = (class_0 == 0).all()
            class_1_all_nonzero = (class_1 != 0).all()
            class_1_all_zero = (class_1 == 0).all()
            class_0_all_nonzero = (class_0 != 0).all()

            if (class_0_all_zero and class_1_all_nonzero) or (
                class_1_all_zero and class_0_all_nonzero
            ):
                findings.append(
                    LeakageFinding(
                        check_name="perfect_class_separation",
                        severity=LeakageSeverity.CRITICAL,
                        feature=feature,
                        description=(
                            f"Feature '{feature}' has zero/nonzero split perfectly aligned with target: "
                            f"class_0 all-zero={class_0_all_zero}, class_1 all-nonzero={class_1_all_nonzero}"
                        ),
                        evidence={
                            "class_0_range": [min_0, max_0],
                            "class_1_range": [min_1, max_1],
                            "class_0_all_zero": bool(class_0_all_zero),
                            "class_1_all_nonzero": bool(class_1_all_nonzero),
                        },
                        recommendation=(
                            f"Feature '{feature}' is likely derived from the target. "
                            f"Remove it or investigate its data source."
                        ),
                    )
                )
                continue

            # Compute overlap fraction
            combined_min = min(min_0, min_1)
            combined_max = max(max_0, max_1)
            combined_range = combined_max - combined_min

            if combined_range < 1e-10:
                continue  # All same value — not informative

            overlap_start = max(min_0, min_1)
            overlap_end = min(max_0, max_1)
            overlap = max(0.0, overlap_end - overlap_start)
            overlap_fraction = overlap / combined_range

            if overlap_fraction < 0.01:
                findings.append(
                    LeakageFinding(
                        check_name="perfect_class_separation",
                        severity=LeakageSeverity.CRITICAL,
                        feature=feature,
                        description=(
                            f"Feature '{feature}' perfectly separates target classes "
                            f"(overlap={overlap_fraction:.4f}, <1%): "
                            f"class_0=[{min_0:.2f}, {max_0:.2f}], class_1=[{min_1:.2f}, {max_1:.2f}]"
                        ),
                        evidence={
                            "class_0_range": [min_0, max_0],
                            "class_1_range": [min_1, max_1],
                            "overlap_fraction": overlap_fraction,
                        },
                        recommendation=(
                            f"Feature '{feature}' has near-zero overlap between classes — "
                            f"this strongly suggests data leakage. Remove or investigate."
                        ),
                    )
                )
            elif overlap_fraction < 0.05:
                findings.append(
                    LeakageFinding(
                        check_name="perfect_class_separation",
                        severity=LeakageSeverity.HIGH,
                        feature=feature,
                        description=(
                            f"Feature '{feature}' has very low class overlap "
                            f"(overlap={overlap_fraction:.4f}, <5%)"
                        ),
                        evidence={
                            "class_0_range": [min_0, max_0],
                            "class_1_range": [min_1, max_1],
                            "overlap_fraction": overlap_fraction,
                        },
                        recommendation=f"Investigate whether '{feature}' encodes the target",
                    )
                )

        except Exception as e:
            logger.warning(f"Perfect class separation check failed for '{feature}': {e}")

    return findings


# =============================================================================
# CHECK 5: ZERO VARIANCE WITHIN CLASS
# =============================================================================


def check_zero_variance_within_class(
    df: Any, target_variable: str, numeric_features: List[str]
) -> List[LeakageFinding]:
    """Check for zero variance within target classes.

    If a feature has std=0 within one or both classes with different constant
    values, this indicates a degenerate separation pattern.

    Args:
        df: DataFrame to check
        target_variable: Target column name
        numeric_features: List of numeric feature column names

    Returns:
        List of LeakageFinding objects
    """
    findings: List[LeakageFinding] = []
    target = df[target_variable]

    for feature in numeric_features:
        try:
            feat = df[feature]
            valid = feat.notna() & target.notna()
            feat_valid = feat[valid]
            tgt_valid = target[valid]

            class_0 = feat_valid[tgt_valid == 0]
            class_1 = feat_valid[tgt_valid == 1]

            if len(class_0) < 5 or len(class_1) < 5:
                continue

            # Rare-event guard — mirror of check_perfect_class_separation (RC1).
            # A binary/near-binary feature that is constant within the tiny
            # positive class is small-sample degeneracy, NOT leakage: with a
            # rare positive class, a sparse pre-index flag whose few 1s all land
            # in the negative class is all-0 in the positive class -> std==0 ->
            # false HIGH/CRITICAL. Skip when (a) <=2 unique values AND (b) the
            # positive class is small (n < 30 or positive rate < 5%). The
            # genuine post-index leak is still caught by logical_dependency and
            # single_feature_auc, so this does not weaken leak detection.
            n_unique = feat_valid.nunique()
            pos_rate = len(class_1) / max(len(feat_valid), 1)
            if n_unique <= 2 and (len(class_1) < 30 or pos_rate < 0.05):
                logger.debug(
                    f"Skipping zero_variance_within_class for binary feature '{feature}' — "
                    f"rare-event cohort (n_pos={len(class_1)}, pos_rate={pos_rate:.2%})"
                )
                continue

            std_0 = float(class_0.std())
            std_1 = float(class_1.std())
            mean_0 = float(class_0.mean())
            mean_1 = float(class_1.mean())

            # Use tolerance for zero-std check (std() with ddof=1 can return NaN
            # for single-element series, or tiny floats for near-constant series)
            _ZERO_STD = 1e-10
            std_0_is_zero = std_0 < _ZERO_STD or np.isnan(std_0)
            std_1_is_zero = std_1 < _ZERO_STD or np.isnan(std_1)

            if std_0_is_zero and std_1_is_zero and abs(mean_0 - mean_1) > 1e-10:
                findings.append(
                    LeakageFinding(
                        check_name="zero_variance_within_class",
                        severity=LeakageSeverity.CRITICAL,
                        feature=feature,
                        description=(
                            f"Feature '{feature}' has zero variance in BOTH classes with different "
                            f"constants (class_0={mean_0:.4f}, class_1={mean_1:.4f}) — "
                            f"degenerate perfect separation"
                        ),
                        evidence={
                            "class_0_std": std_0,
                            "class_1_std": std_1,
                            "class_0_mean": mean_0,
                            "class_1_mean": mean_1,
                        },
                        recommendation=f"Feature '{feature}' is a deterministic function of the target. Remove it.",
                    )
                )
            elif (std_0_is_zero or std_1_is_zero) and abs(mean_0 - mean_1) > 1e-10:
                findings.append(
                    LeakageFinding(
                        check_name="zero_variance_within_class",
                        severity=LeakageSeverity.HIGH,
                        feature=feature,
                        description=(
                            f"Feature '{feature}' has zero variance in one class "
                            f"(std_0={std_0:.4f}, std_1={std_1:.4f}, "
                            f"mean_0={mean_0:.4f}, mean_1={mean_1:.4f})"
                        ),
                        evidence={
                            "class_0_std": std_0,
                            "class_1_std": std_1,
                            "class_0_mean": mean_0,
                            "class_1_mean": mean_1,
                        },
                        recommendation=f"Investigate '{feature}' — constant value in one class suggests leakage",
                    )
                )

        except Exception as e:
            logger.warning(f"Zero variance check failed for '{feature}': {e}")

    return findings


# =============================================================================
# CHECK 6: MUTUAL INFORMATION
# =============================================================================


def check_mutual_information(
    df: Any, target_variable: str, numeric_features: List[str]
) -> List[LeakageFinding]:
    """Check for implausibly high mutual information between features and target.

    Uses sklearn.feature_selection.mutual_info_classif. MI is normalized by
    log(n_classes) for a 0-1 scale.

    Args:
        df: DataFrame to check
        target_variable: Target column name
        numeric_features: List of numeric feature column names

    Returns:
        List of LeakageFinding objects
    """
    findings: List[LeakageFinding] = []

    if len(df) < 30 or len(numeric_features) == 0:
        return findings

    try:
        from sklearn.feature_selection import mutual_info_classif

        target = df[target_variable]
        valid_mask = target.notna()
        for feat in numeric_features:
            valid_mask = valid_mask & df[feat].notna()

        X = df.loc[valid_mask, numeric_features].values
        y = target[valid_mask].values

        if len(y) < 30:
            return findings

        n_classes = len(np.unique(y))
        if n_classes < 2:
            return findings

        mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
        normalizer = np.log(n_classes) if n_classes > 1 else 1.0

        for i, feature in enumerate(numeric_features):
            mi_raw = float(mi_scores[i])
            mi_normalized = mi_raw / normalizer if normalizer > 0 else mi_raw

            if mi_normalized > 0.9:
                findings.append(
                    LeakageFinding(
                        check_name="mutual_information",
                        severity=LeakageSeverity.CRITICAL,
                        feature=feature,
                        description=(
                            f"Feature '{feature}' has implausibly high mutual information "
                            f"with target (MI={mi_raw:.4f}, normalized={mi_normalized:.4f})"
                        ),
                        evidence={"mi_raw": mi_raw, "mi_normalized": mi_normalized},
                        recommendation=f"MI > 0.9 indicates '{feature}' nearly determines the target. Investigate data source.",
                    )
                )
            elif mi_normalized > 0.7:
                findings.append(
                    LeakageFinding(
                        check_name="mutual_information",
                        severity=LeakageSeverity.HIGH,
                        feature=feature,
                        description=(
                            f"Feature '{feature}' has suspiciously high mutual information "
                            f"with target (MI={mi_raw:.4f}, normalized={mi_normalized:.4f})"
                        ),
                        evidence={"mi_raw": mi_raw, "mi_normalized": mi_normalized},
                        recommendation=f"Review '{feature}' — high MI may indicate target leakage",
                    )
                )

    except ImportError:
        logger.warning("sklearn not available for mutual information check")
    except Exception as e:
        logger.warning(f"Mutual information check failed: {e}")

    return findings


# =============================================================================
# CHECK 7: FEATURE-TARGET LOGICAL DEPENDENCY
# =============================================================================


def check_feature_target_logical_dependency(
    df: Any, target_variable: str, numeric_features: List[str]
) -> List[LeakageFinding]:
    """Detect tautological 'if and only if' relationships.

    For each numeric feature, checks:
    - n_nonzero_when_target_0 = ((feature != 0) & (target == 0)).sum()
    - n_zero_when_target_1 = ((feature == 0) & (target == 1)).sum()
    If both counts are 0 (or < 1% of class size), the feature is logically
    equivalent to the target.

    Args:
        df: DataFrame to check
        target_variable: Target column name
        numeric_features: List of numeric feature column names

    Returns:
        List of LeakageFinding objects
    """
    findings: List[LeakageFinding] = []
    target = df[target_variable]

    for feature in numeric_features:
        try:
            feat = df[feature]
            valid = feat.notna() & target.notna()
            feat_valid = feat[valid]
            tgt_valid = target[valid]

            n_class_0 = int((tgt_valid == 0).sum())
            n_class_1 = int((tgt_valid == 1).sum())

            if n_class_0 < 5 or n_class_1 < 5:
                continue

            # Check: feature != 0 when target == 0
            n_nonzero_when_0 = int(((feat_valid != 0) & (tgt_valid == 0)).sum())
            # Check: feature == 0 when target == 1
            n_zero_when_1 = int(((feat_valid == 0) & (tgt_valid == 1)).sum())

            tolerance_0 = max(1, int(n_class_0 * 0.01))
            tolerance_1 = max(1, int(n_class_1 * 0.01))

            if n_nonzero_when_0 <= tolerance_0 and n_zero_when_1 <= tolerance_1:
                # Also check the reverse: feature == 0 IFF target == 0
                findings.append(
                    LeakageFinding(
                        check_name="logical_dependency",
                        severity=LeakageSeverity.CRITICAL,
                        feature=feature,
                        description=(
                            f"Feature '{feature}' is logically equivalent to target: "
                            f"nonzero_when_target=0: {n_nonzero_when_0}/{n_class_0} "
                            f"(<{tolerance_0}), zero_when_target=1: {n_zero_when_1}/{n_class_1} "
                            f"(<{tolerance_1})"
                        ),
                        evidence={
                            "n_nonzero_when_target_0": n_nonzero_when_0,
                            "n_zero_when_target_1": n_zero_when_1,
                            "n_class_0": n_class_0,
                            "n_class_1": n_class_1,
                            "tolerance_0": tolerance_0,
                            "tolerance_1": tolerance_1,
                        },
                        recommendation=(
                            f"Feature '{feature}' value is a tautological indicator of the target. "
                            f"This creates a model that 'predicts' rather than learns. Remove it."
                        ),
                    )
                )

        except Exception as e:
            logger.warning(f"Logical dependency check failed for '{feature}': {e}")

    return findings


# =============================================================================
# 8. SINGLE-FEATURE AUC CHECK
# =============================================================================


def check_single_feature_auc(
    df: Any, target_variable: str, numeric_features: List[str]
) -> List[LeakageFinding]:
    """Flag features where a single column yields high AUC against the target.

    Catches leakage that range-based checks miss when distributions are skewed
    but ranges overlap (e.g., disease_severity with AUC=0.964).

    Args:
        df: Combined DataFrame (train + validation)
        target_variable: Name of target column
        numeric_features: Numeric feature column names

    Returns:
        List of LeakageFinding for features with suspiciously high AUC
    """
    from sklearn.metrics import roc_auc_score

    findings = []
    target = df[target_variable].values
    unique_classes = np.unique(
        target[~np.isnan(target)] if np.issubdtype(target.dtype, np.floating) else target
    )

    if len(unique_classes) != 2:
        return findings  # AUC only defined for binary targets

    for feature in numeric_features:
        try:
            mask = df[[feature, target_variable]].notna().all(axis=1)
            if mask.sum() < 30:
                continue

            y = df.loc[mask, target_variable].values.astype(float)
            x = df.loc[mask, feature].values.astype(float)

            # AUC can be < 0.5 if relationship is inverted — check both directions
            auc = roc_auc_score(y, x)
            effective_auc = max(auc, 1 - auc)

            if effective_auc > 0.90:
                severity = LeakageSeverity.CRITICAL
            elif effective_auc > 0.80:
                severity = LeakageSeverity.HIGH
            else:
                continue

            findings.append(
                LeakageFinding(
                    check_name="single_feature_auc",
                    severity=severity,
                    feature=feature,
                    description=(
                        f"Feature '{feature}' alone achieves AUC={effective_auc:.3f} "
                        f"against target '{target_variable}'. This indicates the feature "
                        f"nearly perfectly predicts the target by itself."
                    ),
                    evidence={
                        "auc": round(effective_auc, 4),
                        "raw_auc": round(auc, 4),
                        "n_samples": int(mask.sum()),
                    },
                    recommendation=(
                        f"Feature '{feature}' is likely derived from or tautologically "
                        f"related to the target. Remove it to prevent leakage."
                    ),
                )
            )

        except Exception as e:
            logger.warning(f"Single-feature AUC check failed for '{feature}': {e}")

    return findings


# =============================================================================
# 9. CATEGORICAL CLASS SEPARATION (Cramér's V)
# =============================================================================


def check_categorical_class_separation(
    df: Any, target_variable: str, categorical_features: List[str]
) -> List[LeakageFinding]:
    """Flag categorical features with high association to the target (Cramér's V).

    Args:
        df: Combined DataFrame (train + validation)
        target_variable: Name of target column
        categorical_features: Categorical feature column names

    Returns:
        List of LeakageFinding for features with suspiciously high Cramér's V
    """
    from scipy.stats import chi2_contingency

    findings = []

    for feature in categorical_features:
        try:
            subset = df[[feature, target_variable]].dropna()
            if len(subset) < 30:
                continue
            # Skip high-cardinality categoricals (likely IDs)
            if subset[feature].nunique() > 50:
                continue

            contingency = pd.crosstab(subset[feature], subset[target_variable])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue

            chi2, p_value, dof, _ = chi2_contingency(contingency)
            n = len(subset)
            k = min(contingency.shape) - 1
            if k == 0:
                continue
            cramers_v = np.sqrt(chi2 / (n * k))

            if cramers_v > 0.7:
                severity = LeakageSeverity.CRITICAL
            elif cramers_v > 0.5:
                severity = LeakageSeverity.HIGH
            else:
                continue

            findings.append(
                LeakageFinding(
                    check_name="categorical_class_separation",
                    severity=severity,
                    feature=feature,
                    description=(
                        f"Categorical feature '{feature}' has Cramér's V={cramers_v:.3f} "
                        f"with target '{target_variable}' (p={p_value:.2e}). "
                        f"This indicates very high association."
                    ),
                    evidence={
                        "cramers_v": round(cramers_v, 4),
                        "chi2": round(chi2, 2),
                        "p_value": p_value,
                        "n_categories": int(contingency.shape[0]),
                        "n_samples": int(n),
                    },
                    recommendation=(
                        f"Categorical feature '{feature}' is strongly associated with "
                        f"the target. Investigate whether it is derived from or a proxy for the target."
                    ),
                )
            )

        except Exception as e:
            logger.warning(f"Categorical class separation check failed for '{feature}': {e}")

    return findings


# =============================================================================
# TEMPORAL LEAKAGE HELPERS (unchanged)
# =============================================================================


def _check_date_ordering(df: Any, event_col: str, target_col: str) -> tuple:
    """Check if event dates occur after target dates."""
    try:
        event_dates = pd.to_datetime(df[event_col], errors="coerce")
        target_dates = pd.to_datetime(df[target_col], errors="coerce")

        valid_mask = event_dates.notna() & target_dates.notna()
        leakage_mask = valid_mask & (event_dates > target_dates)

        leakage_count = leakage_mask.sum()
        leakage_pct = (leakage_count / len(df)) * 100 if len(df) > 0 else 0

        return leakage_count, leakage_pct
    except Exception:
        return 0, 0.0


def _check_future_dates(df: Any, col: str, reference_date: datetime) -> tuple:
    """Check for dates after a reference date."""
    try:
        dates = pd.to_datetime(df[col], errors="coerce")
        valid_mask = dates.notna()

        ref_date = pd.Timestamp(reference_date).tz_localize(None)
        dates_naive = dates.dt.tz_localize(None) if dates.dt.tz is not None else dates

        future_mask = valid_mask & (dates_naive > ref_date)
        future_count = future_mask.sum()
        future_pct = (future_count / len(df)) * 100 if len(df) > 0 else 0

        return future_count, future_pct
    except Exception:
        return 0, 0.0


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime."""
    try:
        result = pd.to_datetime(date_str).to_pydatetime()
        return result if isinstance(result, datetime) else None
    except Exception:
        return None


def _detect_date_columns(df: Any, exclude: Optional[List[str]] = None) -> List[str]:
    """Auto-detect date columns in DataFrame."""
    exclude = exclude or []
    date_cols = []

    for col in df.columns:
        if col in exclude:
            continue

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue

        date_patterns = ["_date", "_time", "_at", "_timestamp", "date_", "time_"]
        if any(pattern in col.lower() for pattern in date_patterns):
            try:
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() > len(sample) * 0.5:
                        date_cols.append(col)
            except Exception:
                pass

    return date_cols
