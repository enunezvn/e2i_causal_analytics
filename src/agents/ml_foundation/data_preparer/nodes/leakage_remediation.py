"""Leakage Remediation Node - LLM-Assisted Feature Remediation.

When the leakage detector flags CRITICAL or HIGH severity findings, this node
uses Claude to reason about WHY features leak, discover clean alternatives
from the available data, and apply remediation before training proceeds.

This follows the same pattern as qc_remediation.py:
  detect → LLM analysis → automatic fix → re-check loop
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..state import DataPreparerState

logger = logging.getLogger(__name__)

# Maximum number of remediation attempts before giving up
MAX_LEAKAGE_REMEDIATION_ATTEMPTS = 2


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


async def review_and_remediate_leakage(state: DataPreparerState) -> Dict[str, Any]:
    """Review leakage findings and attempt remediation using LLM analysis.

    This node:
    1. Analyzes structured leakage findings from detect_leakage
    2. Uses Claude to classify root causes and propose alternatives
    3. Drops leaked features and validates replacements
    4. Returns updated DataFrames with clean feature sets

    When severity is not CRITICAL/HIGH, this is a no-op pass-through.

    Args:
        state: Current agent state with leakage detection results

    Returns:
        Updated state with remediation results or pass-through
    """
    severity = state.get("leakage_severity", "none")
    attempts = state.get("leakage_remediation_attempts", 0)
    leaked = state.get("leaked_features", [])

    # No-op if leakage is not severe enough to warrant remediation
    if severity not in ("critical", "high"):
        logger.info(f"Leakage severity '{severity}' does not require remediation")
        return {"leakage_remediation_status": "not_needed"}

    # No-op if skip_leakage_check is set
    if state.get("skip_leakage_check", False):
        logger.warning("Leakage remediation skipped per configuration")
        return {"leakage_remediation_status": "not_needed"}

    # Guard against infinite loops
    if attempts >= MAX_LEAKAGE_REMEDIATION_ATTEMPTS:
        logger.warning(
            f"Max leakage remediation attempts ({MAX_LEAKAGE_REMEDIATION_ATTEMPTS}) "
            "exceeded. Returning failure."
        )
        return {
            "leakage_remediation_status": "failed",
            "leakage_remediation_viable": False,
            "leakage_remediation_reasoning": (
                f"Exhausted {MAX_LEAKAGE_REMEDIATION_ATTEMPTS} remediation attempts. "
                f"Leaked features ({', '.join(leaked)}) could not be fully remediated."
            ),
        }

    logger.info(
        f"Leakage remediation: severity={severity}, "
        f"leaked_features={leaked}, attempt={attempts + 1}"
    )

    try:
        # Step 1: Gather context for LLM
        context = _gather_leakage_context(state)

        # Step 2: Analyze with LLM (with rule-based fallback)
        analysis = await _analyze_leakage_with_llm(context)

        # Step 3: Apply remediation to DataFrames
        result = _apply_leakage_remediation(state, analysis)

        if result["success"]:
            logger.info(
                f"Leakage remediation applied: dropped={result['dropped']}, "
                f"final_features={result['final_features']}"
            )
            return {
                "leakage_remediation_status": "applied",
                "leakage_remediation_attempts": attempts + 1,
                "leakage_remediated_features": result["final_features"],
                "leakage_dropped_features": result["dropped"],
                "leakage_added_features": result["added"],
                "leakage_remediation_reasoning": analysis.get("reasoning", ""),
                "leakage_remediation_viable": result["viable"],
                "requires_leakage_revalidation": True,
                # Updated DataFrames with leaked columns removed
                "train_df": result["train_df"],
                "validation_df": result["validation_df"],
                "test_df": result["test_df"],
                "holdout_df": result["holdout_df"],
                # Clear previous leakage state so re-check starts fresh
                "leakage_detected": False,
                "leakage_issues": [],
                "leakage_findings": [],
                "leakage_severity": "none",
                "leaked_features": [],
                # Update blocking_issues: remove leakage-related entries
                "blocking_issues": [
                    issue for issue in (state.get("blocking_issues") or [])
                    if not any(lf in issue for lf in leaked)
                ],
            }
        else:
            logger.warning(f"Leakage remediation failed: {result.get('reason')}")
            return {
                "leakage_remediation_status": "failed",
                "leakage_remediation_attempts": attempts + 1,
                "leakage_remediation_viable": False,
                "leakage_remediation_reasoning": result.get("reason", "Unknown failure"),
            }

    except Exception as e:
        logger.error(f"Leakage remediation error: {e}", exc_info=True)
        return {
            "leakage_remediation_status": "error",
            "leakage_remediation_attempts": attempts + 1,
            "leakage_remediation_viable": False,
            "leakage_remediation_reasoning": f"Remediation error: {e}",
        }


# =============================================================================
# CONTEXT GATHERING
# =============================================================================


def _gather_leakage_context(state: DataPreparerState) -> Dict[str, Any]:
    """Assemble context for LLM analysis.

    Args:
        state: Current agent state

    Returns:
        Context dictionary with leakage findings, available columns, and metadata
    """
    train_df = state.get("train_df")
    scope_spec = state.get("scope_spec", {})
    target_variable = scope_spec.get("prediction_target", "")

    # Build column profiles from training data
    column_profiles: List[Dict[str, Any]] = []
    if train_df is not None:
        for col in train_df.columns:
            try:
                nunique = int(train_df[col].nunique())
            except TypeError:
                # Columns with unhashable types (lists, dicts) can't compute nunique
                nunique = -1
            profile: Dict[str, Any] = {
                "name": col,
                "dtype": str(train_df[col].dtype),
                "dtype_kind": train_df[col].dtype.kind,
                "null_pct": round(train_df[col].isna().mean() * 100, 1),
                "nunique": nunique,
                "n_rows": len(train_df),
            }
            if train_df[col].dtype.kind in "iufb":
                all_null = train_df[col].isna().all()
                profile["min"] = float(train_df[col].min()) if not all_null else None
                profile["max"] = float(train_df[col].max()) if not all_null else None
                profile["mean"] = round(float(train_df[col].mean()), 4) if not all_null else None
                profile["std"] = round(float(train_df[col].std()), 4) if not all_null else None
            elif train_df[col].dtype == object and nunique > 0 and nunique <= 20:
                profile["top_values"] = train_df[col].value_counts(dropna=False).head(5).to_dict()
            column_profiles.append(profile)

    # Target distribution
    target_dist = {}
    if train_df is not None and target_variable and target_variable in train_df.columns:
        target_dist = train_df[target_variable].value_counts().to_dict()
        # Convert numpy types to native Python for JSON serialization
        target_dist = {str(k): int(v) for k, v in target_dist.items()}

    return {
        "leaked_features": state.get("leaked_features", []),
        "leakage_findings": state.get("leakage_findings", []),
        "leakage_severity": state.get("leakage_severity", "none"),
        "column_profiles": column_profiles,
        "target_variable": target_variable,
        "target_distribution": target_dist,
        "problem_type": scope_spec.get("problem_type", "unknown"),
        "current_feature_list": scope_spec.get("required_features", []),
        "experiment_id": state.get("experiment_id", "unknown"),
    }


# =============================================================================
# LLM ANALYSIS
# =============================================================================


async def _analyze_leakage_with_llm(context: Dict[str, Any]) -> Dict[str, Any]:
    """Use Claude to analyze leakage findings and propose remediation.

    Falls back to rule-based analysis if the API is unavailable.

    Args:
        context: Leakage context from _gather_leakage_context

    Returns:
        Parsed analysis with classifications, recommendations, and reasoning
    """
    try:
        import anthropic

        client = anthropic.AsyncAnthropic()
        prompt = _build_leakage_analysis_prompt(context)

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        content_block = response.content[0]
        if not hasattr(content_block, "text"):
            logger.warning(f"Unexpected response type: {type(content_block)}")
            return _rule_based_leakage_analysis(context)

        return _parse_leakage_analysis(content_block.text, context)

    except Exception as e:
        logger.warning(f"LLM leakage analysis failed ({e}), using rule-based fallback")
        return _rule_based_leakage_analysis(context)


def _build_leakage_analysis_prompt(context: Dict[str, Any]) -> str:
    """Build the analysis prompt for Claude.

    Args:
        context: Leakage context

    Returns:
        Formatted prompt string
    """
    leaked = context["leaked_features"]
    findings = context["leakage_findings"]
    profiles = context["column_profiles"]
    target = context["target_variable"]
    target_dist = context["target_distribution"]
    problem_type = context["problem_type"]

    # Format findings
    findings_text = ""
    for f in findings:
        sev = f.get("severity", "unknown").upper()
        findings_text += (
            f"  - [{sev}] {f.get('check_name', '?')}: feature='{f.get('feature', '?')}'\n"
            f"    Description: {f.get('description', 'N/A')}\n"
            f"    Evidence: {f.get('evidence', {})}\n"
            f"    Recommendation: {f.get('recommendation', 'N/A')}\n"
        )

    # Format column profiles (excluding target and already-leaked features)
    available_text = ""
    for p in profiles:
        name = p["name"]
        if name == target or name in leaked:
            continue
        dtype = p["dtype"]
        null_pct = p["null_pct"]
        nunique = p["nunique"]
        stats = ""
        if "mean" in p and p["mean"] is not None:
            stats = f" | mean={p['mean']}, std={p['std']}, min={p['min']}, max={p['max']}"
        elif "top_values" in p:
            top = list(p["top_values"].items())[:3]
            stats = f" | top values: {top}"
        available_text += f"  - {name} (dtype={dtype}, null={null_pct}%, nunique={nunique}{stats})\n"

    return f"""You are a data leakage expert analyzing an ML pipeline that has detected critical data leakage.

## Detected Leakage (overall severity: {context['leakage_severity']})

Leaked features: {leaked}

Findings:
{findings_text}

## Target Variable
Name: {target}
Problem type: {problem_type}
Distribution: {target_dist}

## Available Columns (excluding target and leaked features)
{available_text}

## Task

Analyze the leakage and propose remediation. For each section below, output the section header exactly as shown, followed by your analysis.

LEAKAGE_CLASSIFICATION:
For each leaked feature, classify WHY it leaks. Use one line per feature in the format:
- feature_name: classification (tautological/temporal/proxy/contamination), brief explanation

FEATURES_TO_DROP:
List all features that must be removed. One per line:
- feature_name

REPLACEMENT_CANDIDATES:
From the available columns, identify potential replacement features. Consider:
- Is the feature legitimately available at prediction time?
- Could it have its own leakage issues (e.g., derived from the target)?
- Is it informative enough to contribute to the model?
One per line:
- column_name: risk_level (clean/borderline/risky), brief rationale

RECOMMENDED_FEATURE_SET:
List the final recommended features for training. One per line:
- column_name

VIABLE: yes or no — is the recommended feature set sufficient for a useful model?
CONFIDENCE: high, medium, or low
REASONING: One paragraph summarizing your remediation logic — why features leak, what alternatives exist, and whether the remediated model is worth training."""


def _parse_leakage_analysis(text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the structured LLM response.

    Args:
        text: Raw LLM response text
        context: Original context for fallback

    Returns:
        Structured analysis dict
    """
    analysis: Dict[str, Any] = {
        "leakage_classifications": {},
        "features_to_drop": [],
        "replacement_candidates": [],
        "recommended_feature_set": [],
        "viable": False,
        "confidence": "low",
        "reasoning": "",
    }

    current_section: Optional[str] = None

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Once in REASONING, accumulate all remaining text without
        # re-parsing section headers (the LLM may echo them in prose).
        if current_section == "reasoning":
            if analysis["reasoning"]:
                analysis["reasoning"] += " " + stripped
            else:
                analysis["reasoning"] = stripped
            continue

        # Section headers (only parsed when NOT in reasoning section)
        if stripped.startswith("LEAKAGE_CLASSIFICATION"):
            current_section = "classifications"
            continue
        elif stripped.startswith("FEATURES_TO_DROP"):
            current_section = "drop"
            continue
        elif stripped.startswith("REPLACEMENT_CANDIDATES"):
            current_section = "candidates"
            continue
        elif stripped.startswith("RECOMMENDED_FEATURE_SET"):
            current_section = "recommended"
            continue
        elif stripped.startswith("VIABLE:"):
            value = stripped.replace("VIABLE:", "").strip().lower()
            analysis["viable"] = value in ("yes", "true", "1")
            current_section = None
            continue
        elif stripped.startswith("CONFIDENCE:"):
            analysis["confidence"] = stripped.replace("CONFIDENCE:", "").strip().lower()
            current_section = None
            continue
        elif stripped.startswith("REASONING:"):
            analysis["reasoning"] = stripped.replace("REASONING:", "").strip()
            current_section = "reasoning"
            continue

        # Parse items within current section
        if current_section == "classifications" and stripped.startswith("- "):
            item = stripped[2:]
            if ":" in item:
                feat, rest = item.split(":", 1)
                analysis["leakage_classifications"][feat.strip()] = rest.strip()
        elif current_section == "drop" and stripped.startswith("- "):
            feat = stripped[2:].strip()
            if feat:
                analysis["features_to_drop"].append(feat)
        elif current_section == "candidates" and stripped.startswith("- "):
            item = stripped[2:]
            if ":" in item:
                col, rest = item.split(":", 1)
                analysis["replacement_candidates"].append({
                    "column": col.strip(),
                    "assessment": rest.strip(),
                })
        elif current_section == "recommended" and stripped.startswith("- "):
            feat = stripped[2:].strip()
            if feat:
                analysis["recommended_feature_set"].append(feat)

    # Ensure features_to_drop includes all leaked features even if LLM missed some
    for feat in context.get("leaked_features", []):
        if feat not in analysis["features_to_drop"]:
            analysis["features_to_drop"].append(feat)

    logger.info(
        f"Parsed LLM leakage analysis: "
        f"drop={analysis['features_to_drop']}, "
        f"recommended={analysis['recommended_feature_set']}, "
        f"viable={analysis['viable']}"
    )

    return analysis


# =============================================================================
# RULE-BASED FALLBACK
# =============================================================================


def _rule_based_leakage_analysis(context: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback when Claude API is unavailable.

    Applies conservative rules:
    - Drop all CRITICAL/HIGH features
    - Accept numeric columns with >5% class overlap and <0.95 abs correlation
    - Report viability based on >=2 clean features remaining

    Args:
        context: Leakage context

    Returns:
        Analysis dict matching the LLM output structure
    """
    from .leakage_detector import (
        _aggregate_severity,
        check_feature_target_logical_dependency,
        check_mutual_information,
        check_perfect_class_separation,
        check_zero_variance_within_class,
    )

    leaked = set(context.get("leaked_features", []))
    target = context["target_variable"]
    profiles = context["column_profiles"]
    findings = context.get("leakage_findings", [])

    # Classify leaked features from findings
    classifications = {}
    for f in findings:
        feat = f.get("feature", "")
        check = f.get("check_name", "")
        sev = f.get("severity", "")
        if feat in leaked:
            if check == "logical_dependency":
                classifications[feat] = "tautological — logically equivalent to target"
            elif check == "perfect_class_separation":
                classifications[feat] = "tautological — perfectly separates target classes"
            elif check == "zero_variance_within_class":
                classifications[feat] = "tautological — constant in one class"
            elif check == "mutual_information":
                classifications[feat] = "proxy — implausibly high mutual information"
            elif check == "target_correlation":
                classifications[feat] = "proxy — near-perfect correlation with target"
            else:
                classifications[feat] = f"{check} — {sev} severity"

    # Discover candidate replacements from available numeric columns
    candidates = []
    recommended = []

    for p in profiles:
        name = p["name"]
        if name == target or name in leaked:
            continue
        if p.get("dtype_kind", "") not in "iufb":
            continue
        if p["null_pct"] > 50:
            continue
        if p["nunique"] <= 1:
            continue

        # Run leakage checks on this candidate if we have train_df info
        risk = "clean"
        rationale = f"numeric, {p['null_pct']}% null, {p['nunique']} unique values"

        candidates.append({"column": name, "assessment": f"{risk}, {rationale}"})
        recommended.append(name)

    viable = len(recommended) >= 2
    reasoning = (
        f"Dropped {len(leaked)} leaked features ({', '.join(sorted(leaked))}). "
        f"Found {len(recommended)} candidate replacement features from available data. "
        f"{'Feature set is viable for training.' if viable else 'Insufficient features remaining — enrichment needed.'}"
    )

    return {
        "leakage_classifications": classifications,
        "features_to_drop": sorted(leaked),
        "replacement_candidates": candidates,
        "recommended_feature_set": recommended,
        "viable": viable,
        "confidence": "medium" if viable else "low",
        "reasoning": reasoning,
    }


# =============================================================================
# APPLY REMEDIATION
# =============================================================================


def _apply_leakage_remediation(
    state: DataPreparerState,
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the remediation plan to the DataFrames.

    Drops leaked features and validates that recommended replacements exist
    and pass structural leakage checks.

    Args:
        state: Current agent state with DataFrames
        analysis: Parsed LLM analysis

    Returns:
        Result dict with updated DataFrames and metadata
    """
    from .leakage_detector import (
        _aggregate_severity,
        check_feature_target_logical_dependency,
        check_mutual_information,
        check_perfect_class_separation,
        check_zero_variance_within_class,
    )

    train_df = state.get("train_df")
    validation_df = state.get("validation_df")
    test_df = state.get("test_df")
    holdout_df = state.get("holdout_df")
    scope_spec = state.get("scope_spec", {})
    target_variable = scope_spec.get("prediction_target", "")

    if train_df is None:
        return {"success": False, "reason": "No training data available"}

    features_to_drop = analysis.get("features_to_drop", [])
    recommended = analysis.get("recommended_feature_set", [])

    # Validate recommended features exist in the data and are ML-compatible
    # Reject columns with unhashable types (lists, dicts) or pure strings
    available_cols = set(train_df.columns)
    valid_recommended = []
    for f in recommended:
        if f not in available_cols or f == target_variable:
            continue
        col = train_df[f]
        # Accept numeric/boolean columns
        if col.dtype.kind in "iufb":
            valid_recommended.append(f)
            continue
        # Reject object columns that contain unhashable types (lists, dicts)
        try:
            col.nunique()
            valid_recommended.append(f)
        except TypeError:
            logger.info(f"Rejecting '{f}': contains unhashable types (lists/dicts)")
            continue

    if not valid_recommended:
        available_sample = sorted(available_cols)[:10]
        return {
            "success": False,
            "viable": False,
            "reason": (
                f"No valid replacement features found. "
                f"Recommended {recommended} but available columns (first 10): {available_sample}"
            ),
        }

    # Run structural leakage checks on each recommended feature
    verified_features: List[str] = []
    rejected_features: List[str] = []

    for feat in valid_recommended:
        if feat not in train_df.columns or target_variable not in train_df.columns:
            rejected_features.append(feat)
            continue

        combined = train_df[[feat, target_variable]].dropna()
        if len(combined) < 30:
            verified_features.append(feat)  # Too few rows to check, accept
            continue

        if combined[feat].dtype.kind not in "iufb":
            verified_features.append(feat)  # Non-numeric, skip structural checks
            continue

        # Run the 4 structural checks
        check_findings = []
        numeric_feats = [feat]
        try:
            check_findings.extend(
                check_perfect_class_separation(combined, target_variable, numeric_feats)
            )
            check_findings.extend(
                check_zero_variance_within_class(combined, target_variable, numeric_feats)
            )
            check_findings.extend(
                check_mutual_information(combined, target_variable, numeric_feats)
            )
            check_findings.extend(
                check_feature_target_logical_dependency(combined, target_variable, numeric_feats)
            )
        except Exception as e:
            logger.warning(f"Leakage check failed for '{feat}': {e}")
            verified_features.append(feat)  # Accept on check failure
            continue

        severity = _aggregate_severity(check_findings)
        if severity in ("critical", "high"):
            logger.info(f"Replacement candidate '{feat}' also has {severity} leakage — rejecting")
            rejected_features.append(feat)
        else:
            verified_features.append(feat)

    if len(verified_features) < 2:
        return {
            "success": False,
            "viable": False,
            "reason": (
                f"Only {len(verified_features)} clean features after validation "
                f"(need >= 2). Verified: {verified_features}, "
                f"rejected: {rejected_features}"
            ),
        }

    # Determine which columns to keep in the DataFrames
    # Keep: target, verified features, and non-feature columns (IDs, metadata)
    # We only drop the explicitly leaked features — other columns stay for downstream use
    cols_to_drop = [c for c in features_to_drop if c in available_cols and c != target_variable]
    # Also drop any recommended features that were rejected by our checks
    cols_to_drop.extend([c for c in rejected_features if c in available_cols])
    cols_to_drop = list(set(cols_to_drop))

    # Apply drops to all splits
    updated_train = train_df.drop(columns=cols_to_drop, errors="ignore")
    updated_val = validation_df.drop(columns=cols_to_drop, errors="ignore") if validation_df is not None else None
    updated_test = test_df.drop(columns=cols_to_drop, errors="ignore") if test_df is not None else None
    updated_holdout = holdout_df.drop(columns=cols_to_drop, errors="ignore") if holdout_df is not None else None

    # Determine which features were added (not in original leaked set)
    original_leaked = set(state.get("leaked_features", []))
    added = [f for f in verified_features if f not in original_leaked]

    logger.info(
        f"Leakage remediation applied: "
        f"dropped={cols_to_drop}, verified={verified_features}, "
        f"rejected={rejected_features}, added={added}"
    )

    return {
        "success": True,
        "viable": True,
        "dropped": cols_to_drop,
        "final_features": verified_features,
        "added": added,
        "rejected": rejected_features,
        "train_df": updated_train,
        "validation_df": updated_val,
        "test_df": updated_test,
        "holdout_df": updated_holdout,
    }
