"""QC Remediation Node - LLM-Assisted Review Loop.

This node uses Claude to analyze QC failures, diagnose root causes,
and attempt automatic remediation when possible.
"""

import logging
from typing import Any, Dict, List, Tuple

import anthropic
import pandas as pd

from ..state import DataPreparerState

logger = logging.getLogger(__name__)

# Maximum number of remediation attempts before giving up
MAX_REMEDIATION_ATTEMPTS = 2


async def review_and_remediate_qc(state: DataPreparerState) -> Dict[str, Any]:
    """Review QC failures and attempt remediation using LLM analysis.

    This node:
    1. Analyzes QC failures and blocking issues
    2. Uses Claude to diagnose root causes
    3. Attempts automatic remediation when possible
    4. Returns detailed guidance when auto-fix isn't possible

    Args:
        state: Current agent state with QC results

    Returns:
        Updated state with remediation results or failure analysis
    """
    experiment_id = state.get("experiment_id", "unknown")
    qc_status = state.get("qc_status", "unknown")
    overall_score = state.get("overall_score")
    gate_passed = state.get("gate_passed", False)
    remediation_attempts = state.get("remediation_attempts", 0)

    logger.info(
        f"QC Remediation review for experiment {experiment_id}: "
        f"status={qc_status}, score={overall_score}, gate_passed={gate_passed}"
    )

    # If QC already passed, no remediation needed
    if gate_passed and qc_status == "passed":
        logger.info("QC gate already passed, no remediation needed")
        return {"remediation_status": "not_needed"}

    # Check if we've exceeded max remediation attempts
    if remediation_attempts >= MAX_REMEDIATION_ATTEMPTS:
        logger.warning(
            f"Max remediation attempts ({MAX_REMEDIATION_ATTEMPTS}) exceeded. "
            "Returning failure analysis."
        )
        return await _generate_failure_analysis(state)

    try:
        # Gather QC context for LLM analysis
        qc_context = _gather_qc_context(state)

        # Use Claude to analyze the QC failures
        analysis = await _analyze_qc_failures_with_llm(qc_context)

        # Check if automatic remediation is possible
        if analysis.get("can_auto_remediate"):
            # Apply automatic remediation
            remediation_result = await _apply_automatic_remediation(
                state, analysis.get("remediation_actions", [])
            )

            if remediation_result.get("success"):
                return {
                    "remediation_status": "applied",
                    "remediation_attempts": remediation_attempts + 1,
                    "remediation_actions_taken": remediation_result.get("actions_taken", []),
                    "requires_revalidation": True,
                    "llm_analysis": analysis.get("root_cause_summary"),
                }
            else:
                # Remediation failed, provide guidance
                return {
                    "remediation_status": "failed",
                    "remediation_attempts": remediation_attempts + 1,
                    "remediation_error": remediation_result.get("error"),
                    "llm_analysis": analysis.get("root_cause_summary"),
                    "recommended_actions": analysis.get("manual_remediation_steps", []),
                }
        else:
            # Cannot auto-remediate, provide detailed guidance
            return {
                "remediation_status": "manual_required",
                "remediation_attempts": remediation_attempts + 1,
                "llm_analysis": analysis.get("root_cause_summary"),
                "root_causes": analysis.get("root_causes", []),
                "recommended_actions": analysis.get("manual_remediation_steps", []),
                "estimated_effort": analysis.get("estimated_effort", "unknown"),
                "blocking_issues_analysis": analysis.get("blocking_issues_analysis", []),
            }

    except Exception as e:
        logger.error(f"QC remediation review failed: {e}", exc_info=True)
        return {
            "remediation_status": "error",
            "remediation_error": str(e),
            "error_type": "remediation_review_error",
        }


def _gather_qc_context(state: DataPreparerState) -> Dict[str, Any]:
    """Gather QC context for LLM analysis.

    Args:
        state: Current agent state

    Returns:
        Context dictionary for LLM prompt
    """
    train_df = state.get("train_df")

    # Build comprehensive context
    context = {
        "experiment_id": state.get("experiment_id"),
        "qc_status": state.get("qc_status", "unknown"),
        "overall_score": state.get("overall_score"),
        "gate_passed": state.get("gate_passed", False),
        # Dimension scores
        "dimension_scores": {
            "completeness": state.get("completeness_score"),
            "validity": state.get("validity_score"),
            "consistency": state.get("consistency_score"),
            "uniqueness": state.get("uniqueness_score"),
            "timeliness": state.get("timeliness_score"),
        },
        # Issues
        "blocking_issues": state.get("blocking_issues", []),
        "failed_expectations": state.get("failed_expectations", []),
        "warnings": state.get("warnings", []),
        "existing_remediation_steps": state.get("remediation_steps", []),
        # Schema validation
        "schema_validation_status": state.get("schema_validation_status"),
        "schema_validation_errors": state.get("schema_validation_errors", []),
        # Leakage detection
        "leakage_detected": state.get("leakage_detected", False),
        "leakage_issues": state.get("leakage_issues", []),
        # Data stats
        "data_stats": {
            "train_samples": len(train_df) if train_df is not None else 0,
            "columns": list(train_df.columns) if train_df is not None else [],
            "dtypes": {
                col: str(dtype)
                for col, dtype in (train_df.dtypes.items() if train_df is not None else {})
            },
            "null_counts": train_df.isnull().sum().to_dict() if train_df is not None else {},
        },
        # Configuration
        "scope_spec": state.get("scope_spec", {}),
    }

    return context


async def _analyze_qc_failures_with_llm(context: Dict[str, Any]) -> Dict[str, Any]:
    """Use Claude to analyze QC failures and suggest remediation.

    Args:
        context: QC context dictionary

    Returns:
        Analysis results with root causes and remediation suggestions
    """
    try:
        client = anthropic.AsyncAnthropic()

        # Build prompt for QC analysis
        prompt = _build_qc_analysis_prompt(context)

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        # Parse the LLM response
        analysis_text = response.content[0].text
        return _parse_llm_analysis(analysis_text, context)

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error during QC analysis: {e}")
        # Fallback to rule-based analysis
        return _rule_based_analysis(context)
    except Exception as e:
        logger.error(f"Error in LLM QC analysis: {e}")
        return _rule_based_analysis(context)


def _build_qc_analysis_prompt(context: Dict[str, Any]) -> str:
    """Build the prompt for QC failure analysis.

    Args:
        context: QC context dictionary

    Returns:
        Formatted prompt string
    """
    blocking_issues = context.get("blocking_issues", [])
    dimension_scores = context.get("dimension_scores", {})
    data_stats = context.get("data_stats", {})

    prompt = f"""You are a data quality expert analyzing ML pipeline QC failures.

## QC Status
- Overall Status: {context.get("qc_status", "unknown")}
- Overall Score: {context.get("overall_score", "N/A")}
- Gate Passed: {context.get("gate_passed", False)}

## Dimension Scores
- Completeness: {dimension_scores.get("completeness", "N/A")}
- Validity: {dimension_scores.get("validity", "N/A")}
- Consistency: {dimension_scores.get("consistency", "N/A")}
- Uniqueness: {dimension_scores.get("uniqueness", "N/A")}
- Timeliness: {dimension_scores.get("timeliness", "N/A")}

## Blocking Issues
{chr(10).join(f"- {issue}" for issue in blocking_issues) if blocking_issues else "- None detected"}

## Data Statistics
- Training samples: {data_stats.get("train_samples", 0)}
- Columns: {len(data_stats.get("columns", []))}
- Null counts per column: {data_stats.get("null_counts", {})}

## Schema Validation
- Status: {context.get("schema_validation_status", "unknown")}
- Errors: {context.get("schema_validation_errors", [])}

## Leakage Detection
- Leakage Detected: {context.get("leakage_detected", False)}
- Issues: {context.get("leakage_issues", [])}

## Failed Expectations
{context.get("failed_expectations", [])}

## Warnings
{context.get("warnings", [])}

## Existing Remediation Suggestions
{context.get("existing_remediation_steps", [])}

---

Please analyze these QC failures and provide:

1. **ROOT CAUSES**: List the root causes of the QC failures (be specific)

2. **CAN_AUTO_REMEDIATE**: Yes/No - Can these issues be automatically fixed by:
   - Dropping columns with too many nulls
   - Imputing missing values
   - Converting data types
   - Removing duplicates

3. **REMEDIATION_ACTIONS**: If auto-remediation is possible, list specific actions:
   - Action type (drop_column, impute, convert_type, deduplicate)
   - Target column
   - Parameters

4. **MANUAL_STEPS**: If manual intervention is needed, list the steps in order of priority

5. **ESTIMATED_EFFORT**: Low/Medium/High - How much effort to fix

Format your response as:
ROOT_CAUSES:
- [cause 1]
- [cause 2]

CAN_AUTO_REMEDIATE: [Yes/No]

REMEDIATION_ACTIONS:
- action: [type], column: [name], params: [details]

MANUAL_STEPS:
1. [step 1]
2. [step 2]

ESTIMATED_EFFORT: [Low/Medium/High]

SUMMARY: [Brief summary of the issues and recommended approach]
"""
    return prompt


def _parse_llm_analysis(response_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the LLM response into structured analysis.

    Args:
        response_text: Raw LLM response
        context: Original QC context

    Returns:
        Structured analysis dictionary
    """
    analysis = {
        "root_causes": [],
        "can_auto_remediate": False,
        "remediation_actions": [],
        "manual_remediation_steps": [],
        "estimated_effort": "unknown",
        "root_cause_summary": "",
        "blocking_issues_analysis": [],
    }

    try:
        lines = response_text.strip().split("\n")
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith("ROOT_CAUSES:"):
                current_section = "root_causes"
            elif line.startswith("CAN_AUTO_REMEDIATE:"):
                value = line.replace("CAN_AUTO_REMEDIATE:", "").strip().lower()
                analysis["can_auto_remediate"] = value in ("yes", "true", "1")
            elif line.startswith("REMEDIATION_ACTIONS:"):
                current_section = "remediation_actions"
            elif line.startswith("MANUAL_STEPS:"):
                current_section = "manual_steps"
            elif line.startswith("ESTIMATED_EFFORT:"):
                analysis["estimated_effort"] = line.replace("ESTIMATED_EFFORT:", "").strip()
            elif line.startswith("SUMMARY:"):
                analysis["root_cause_summary"] = line.replace("SUMMARY:", "").strip()
            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if current_section == "root_causes":
                    analysis["root_causes"].append(item)
                elif current_section == "remediation_actions":
                    analysis["remediation_actions"].append(_parse_remediation_action(item))
            elif line and line[0].isdigit() and current_section == "manual_steps":
                step = line.split(".", 1)[-1].strip() if "." in line else line
                analysis["manual_remediation_steps"].append(step)

    except Exception as e:
        logger.warning(f"Error parsing LLM analysis: {e}")
        # Fallback to returning the raw text as summary
        analysis["root_cause_summary"] = response_text[:500]

    return analysis


def _parse_remediation_action(action_str: str) -> Dict[str, Any]:
    """Parse a remediation action string into structured format.

    Args:
        action_str: Action string like "action: drop_column, column: x, params: {}"

    Returns:
        Structured action dictionary
    """
    action = {"type": "unknown", "column": None, "params": {}}

    try:
        parts = action_str.split(",")
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "action":
                    action["type"] = value
                elif key == "column":
                    action["column"] = value
                elif key == "params":
                    action["params"] = value
    except Exception:
        pass

    return action


def _rule_based_analysis(context: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback rule-based analysis when LLM is unavailable.

    Args:
        context: QC context dictionary

    Returns:
        Analysis results based on rules
    """
    analysis = {
        "root_causes": [],
        "can_auto_remediate": False,
        "remediation_actions": [],
        "manual_remediation_steps": [],
        "estimated_effort": "unknown",
        "root_cause_summary": "",
        "blocking_issues_analysis": [],
    }

    dimension_scores = context.get("dimension_scores", {}) or {}
    data_stats = context.get("data_stats", {}) or {}
    blocking_issues = context.get("blocking_issues", []) or []

    # Analyze dimension scores
    for dim, score in dimension_scores.items():
        if score is None:
            analysis["root_causes"].append(
                f"{dim} score could not be computed - check if data loaded correctly"
            )
        elif score < 0.80:
            analysis["root_causes"].append(f"Low {dim} score ({score:.2f}) below 0.80 threshold")

    # Check for data loading issues
    if data_stats.get("train_samples", 0) == 0:
        analysis["root_causes"].append("No training data loaded - check data source connection")
        analysis["manual_remediation_steps"].append(
            "Verify SUPABASE_URL environment variable is set correctly"
        )
        analysis["manual_remediation_steps"].append(
            "Check database connectivity and table permissions"
        )

    # Analyze null counts
    null_counts = data_stats.get("null_counts", {})
    high_null_cols = [col for col, count in null_counts.items() if count > 0]
    if high_null_cols:
        analysis["root_causes"].append(f"Columns with null values: {', '.join(high_null_cols[:5])}")
        analysis["can_auto_remediate"] = True
        for col in high_null_cols[:3]:  # Limit to first 3
            analysis["remediation_actions"].append(
                {
                    "type": "impute",
                    "column": col,
                    "params": {"strategy": "median" if "score" in col.lower() else "mode"},
                }
            )

    # Check for leakage
    if context.get("leakage_detected"):
        analysis["root_causes"].append("Data leakage detected - cannot auto-remediate")
        analysis["can_auto_remediate"] = False
        for issue in context.get("leakage_issues", []):
            analysis["manual_remediation_steps"].append(f"Address leakage: {issue}")

    # Analyze blocking issues
    for issue in blocking_issues:
        analysis["blocking_issues_analysis"].append(
            {
                "issue": issue,
                "severity": "blocking",
                "can_auto_fix": False,
            }
        )

    # Determine effort
    if len(analysis["root_causes"]) > 3:
        analysis["estimated_effort"] = "High"
    elif len(analysis["root_causes"]) > 1:
        analysis["estimated_effort"] = "Medium"
    else:
        analysis["estimated_effort"] = "Low"

    analysis["root_cause_summary"] = (
        f"Found {len(analysis['root_causes'])} issues. "
        f"{'Auto-remediation possible.' if analysis['can_auto_remediate'] else 'Manual intervention required.'}"
    )

    return analysis


async def _apply_automatic_remediation(
    state: DataPreparerState, actions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply automatic remediation actions to the data.

    Args:
        state: Current agent state
        actions: List of remediation actions to apply

    Returns:
        Result of remediation attempt
    """
    actions_taken = []

    train_df = state.get("train_df")
    validation_df = state.get("validation_df")
    test_df = state.get("test_df")

    if train_df is None:
        return {"success": False, "error": "No training data available for remediation"}

    try:
        for action in actions:
            action_type = action.get("type", "unknown")
            column = action.get("column")
            params = action.get("params", {})

            if action_type == "impute" and column and column in train_df.columns:
                strategy = params.get("strategy", "median")
                train_df, msg = _impute_column(train_df, column, strategy)
                if validation_df is not None and column in validation_df.columns:
                    validation_df, _ = _impute_column(validation_df, column, strategy)
                if test_df is not None and column in test_df.columns:
                    test_df, _ = _impute_column(test_df, column, strategy)
                actions_taken.append(msg)

            elif action_type == "drop_column" and column and column in train_df.columns:
                train_df = train_df.drop(columns=[column])
                if validation_df is not None and column in validation_df.columns:
                    validation_df = validation_df.drop(columns=[column])
                if test_df is not None and column in test_df.columns:
                    test_df = test_df.drop(columns=[column])
                actions_taken.append(f"Dropped column: {column}")

            elif action_type == "deduplicate":
                before_count = len(train_df)
                train_df = train_df.drop_duplicates()
                after_count = len(train_df)
                actions_taken.append(f"Removed {before_count - after_count} duplicate rows")

        return {
            "success": True,
            "actions_taken": actions_taken,
            "train_df": train_df,
            "validation_df": validation_df,
            "test_df": test_df,
        }

    except Exception as e:
        logger.error(f"Remediation failed: {e}")
        return {"success": False, "error": str(e), "actions_taken": actions_taken}


def _impute_column(df: pd.DataFrame, column: str, strategy: str) -> Tuple[pd.DataFrame, str]:
    """Impute missing values in a column.

    Args:
        df: DataFrame to modify
        column: Column to impute
        strategy: Imputation strategy (mean, median, mode, constant)

    Returns:
        Tuple of (modified DataFrame, action message)
    """
    null_count = df[column].isnull().sum()

    if strategy == "median":
        fill_value = df[column].median()
        df[column] = df[column].fillna(fill_value)
        return df, f"Imputed {null_count} nulls in '{column}' with median ({fill_value})"
    elif strategy == "mean":
        fill_value = df[column].mean()
        df[column] = df[column].fillna(fill_value)
        return df, f"Imputed {null_count} nulls in '{column}' with mean ({fill_value:.2f})"
    elif strategy == "mode":
        fill_value = df[column].mode().iloc[0] if not df[column].mode().empty else "UNKNOWN"
        df[column] = df[column].fillna(fill_value)
        return df, f"Imputed {null_count} nulls in '{column}' with mode ({fill_value})"
    else:
        # Default: forward fill then backward fill
        df[column] = df[column].ffill().bfill()
        return df, f"Imputed {null_count} nulls in '{column}' with ffill/bfill"


async def _generate_failure_analysis(state: DataPreparerState) -> Dict[str, Any]:
    """Generate detailed failure analysis when remediation is exhausted.

    Args:
        state: Current agent state

    Returns:
        Comprehensive failure analysis
    """
    context = _gather_qc_context(state)

    return {
        "remediation_status": "exhausted",
        "gate_passed": False,
        "failure_summary": (
            f"QC gate failed after {MAX_REMEDIATION_ATTEMPTS} remediation attempts. "
            f"Status: {context.get('qc_status')}, Score: {context.get('overall_score')}"
        ),
        "blocking_issues": context.get("blocking_issues", []),
        "root_causes_identified": list(context.get("blocking_issues", [])),
        "recommended_manual_actions": [
            "Review data source connectivity (SUPABASE_URL)",
            "Check data loading node for errors",
            "Verify scope_spec configuration matches available data",
            "Examine schema validation errors",
            "Address any data leakage issues before proceeding",
        ],
        "next_steps": [
            "1. Fix the identified issues in the data source",
            "2. Re-run the data_preparer agent",
            "3. If issues persist, contact data engineering team",
        ],
    }
