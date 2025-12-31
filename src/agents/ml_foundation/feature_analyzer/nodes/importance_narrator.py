"""Importance Narrator Node - WITH LLM.

Generates natural language interpretations of SHAP analysis using Claude.
This is the LLM interpretation node in the hybrid pipeline.

V4.4: Added causal discovery integration for causal vs predictive comparison.
"""

import time
from typing import Any, Dict, List, Optional

from anthropic import Anthropic


async def narrate_importance(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate natural language interpretation of SHAP analysis.

    This node:
    1. Takes SHAP analysis results (importance + interactions)
    2. Generates executive summary for stakeholders
    3. Creates feature-specific explanations
    4. Identifies key insights and recommendations
    5. Flags cautions about model behavior
    6. (V4.4) Compares causal vs predictive importance if discovery enabled

    Args:
        state: Current agent state with SHAP analysis

    Returns:
        State updates with NL interpretation
    """
    start_time = time.time()

    try:
        # Extract inputs
        global_importance_ranked = state.get("global_importance_ranked", [])
        feature_directions = state.get("feature_directions", {})
        top_interactions_raw = state.get("top_interactions_raw", [])
        experiment_id = state.get("experiment_id", "unknown")
        model_version = state.get("model_version", "unknown")

        # V4.4: Extract causal discovery results if available
        discovery_enabled = state.get("discovery_enabled", False)
        causal_rankings = state.get("causal_rankings", [])
        discovery_gate_decision = state.get("discovery_gate_decision", None)
        discovery_gate_confidence = state.get("discovery_gate_confidence", 0.0)
        rank_correlation = state.get("rank_correlation", 0.0)
        divergent_features = state.get("divergent_features", [])
        direct_cause_features = state.get("direct_cause_features", [])
        causal_only_features = state.get("causal_only_features", [])
        predictive_only_features = state.get("predictive_only_features", [])

        if not global_importance_ranked:
            return {
                "error": "Missing SHAP analysis results for interpretation",
                "error_type": "missing_shap_results",
                "status": "failed",
            }

        # Prepare context for LLM
        context = _prepare_interpretation_context(
            global_importance_ranked,
            feature_directions,
            top_interactions_raw,
            experiment_id,
            model_version,
        )

        # V4.4: Add causal context if discovery was successful
        has_causal_results = (
            discovery_enabled
            and causal_rankings
            and discovery_gate_decision in ("accept", "review")
        )

        causal_context = ""
        if has_causal_results:
            causal_context = _prepare_causal_context(
                causal_rankings=causal_rankings,
                discovery_gate_decision=discovery_gate_decision,
                discovery_gate_confidence=discovery_gate_confidence,
                rank_correlation=rank_correlation,
                divergent_features=divergent_features,
                direct_cause_features=direct_cause_features,
                causal_only_features=causal_only_features,
                predictive_only_features=predictive_only_features,
            )

        # Call Claude for interpretation
        client = Anthropic()

        # V4.4: Build prompt with optional causal section
        causal_section = ""
        causal_output_section = ""
        if has_causal_results:
            causal_section = f"""

## Causal Discovery Results

{causal_context}
"""
            causal_output_section = """

6. **Causal vs Predictive Comparison** (if causal results available)
   - How causal importance differs from predictive importance
   - Which features are "true causes" vs correlation-based predictors
   - Implications for intervention design"""

        prompt = f"""You are an ML interpretability expert analyzing SHAP results for a pharmaceutical commercial analytics model.

## SHAP Analysis Results

{context}{causal_section}

## Your Task

Generate a comprehensive interpretability report with:

1. **Executive Summary** (2-3 sentences)
   - High-level takeaway for business stakeholders
   - What the model primarily relies on

2. **Feature Explanations** (for top 5 features)
   - Feature name
   - How it affects predictions (direction and magnitude)
   - Business interpretation

3. **Key Insights** (3-5 bullet points)
   - Most important findings
   - Notable patterns or surprises

4. **Recommendations** (2-4 actionable items)
   - How to use these insights
   - What actions to take

5. **Cautions** (1-3 warnings)
   - Limitations or risks
   - Potential confounders{causal_output_section}

## Important Context

- Domain: Pharmaceutical commercial operations (NOT clinical)
- Use cases: HCP targeting, prescription predictions, market dynamics
- Audience: Commercial strategy teams, field operations

Format your response as JSON with this structure:
{{
  "executive_summary": "...",
  "feature_explanations": {{
    "feature_name": "explanation..."
  }},
  "key_insights": ["insight 1", "insight 2", ...],
  "recommendations": ["rec 1", "rec 2", ...],
  "cautions": ["caution 1", "caution 2", ...]{', "causal_interpretation": "..."' if has_causal_results else ''}
}}"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse LLM response
        response_text = response.content[0].text

        # Extract JSON from response
        interpretation_data = _parse_interpretation_response(response_text)

        # Count tokens
        interpretation_tokens = response.usage.input_tokens + response.usage.output_tokens

        # Create structured outputs
        feature_explanations = interpretation_data.get("feature_explanations", {})
        key_insights = interpretation_data.get("key_insights", [])
        recommendations = interpretation_data.get("recommendations", [])
        cautions = interpretation_data.get("cautions", [])
        executive_summary = interpretation_data.get("executive_summary", "")

        # V4.4: Extract causal interpretation if available
        causal_interpretation = interpretation_data.get("causal_interpretation", "")

        # Generate interaction interpretations
        interaction_interpretations = _interpret_interactions(
            top_interactions_raw, feature_explanations
        )

        # Create complete interpretation text
        interpretation = _build_complete_interpretation(
            executive_summary, feature_explanations, key_insights, recommendations, cautions
        )

        computation_time = time.time() - start_time

        result = {
            "executive_summary": executive_summary,
            "feature_explanations": feature_explanations,
            "interaction_interpretations": interaction_interpretations,
            "key_insights": key_insights,
            "recommendations": recommendations,
            "cautions": cautions,
            "interpretation": interpretation,
            "interpretation_model": "claude-sonnet-4-20250514",
            "interpretation_time_seconds": computation_time,
            "interpretation_tokens": interpretation_tokens,
        }

        # V4.4: Add causal interpretation if available
        if causal_interpretation:
            result["causal_interpretation"] = causal_interpretation

        return result

    except Exception as e:
        return {
            "error": f"NL interpretation failed: {str(e)}",
            "error_type": "interpretation_error",
            "error_details": {"exception": str(e)},
            "status": "failed",
        }


def _prepare_interpretation_context(
    global_importance_ranked: List,
    feature_directions: Dict[str, str],
    top_interactions_raw: List,
    experiment_id: str,
    model_version: str,
) -> str:
    """Prepare context string for LLM interpretation.

    Args:
        global_importance_ranked: Ranked feature importance
        feature_directions: Feature effect directions
        top_interactions_raw: Top feature interactions
        experiment_id: Experiment identifier
        model_version: Model version

    Returns:
        Formatted context string
    """
    context_parts = []

    # Experiment metadata
    context_parts.append(f"**Experiment ID**: {experiment_id}")
    context_parts.append(f"**Model Version**: {model_version}")
    context_parts.append("")

    # Top features
    context_parts.append("**Top 10 Features by Importance**:")
    for i, (feature, importance) in enumerate(global_importance_ranked[:10], 1):
        direction = feature_directions.get(feature, "unknown")
        context_parts.append(f"{i}. {feature}: {importance:.4f} (direction: {direction})")

    context_parts.append("")

    # Top interactions
    if top_interactions_raw:
        context_parts.append("**Top 5 Feature Interactions**:")
        for i, (feat1, feat2, strength) in enumerate(top_interactions_raw[:5], 1):
            interaction_type = "amplifying" if strength > 0 else "opposing"
            context_parts.append(f"{i}. {feat1} × {feat2}: {strength:.4f} ({interaction_type})")

    return "\n".join(context_parts)


def _prepare_causal_context(
    causal_rankings: List[Dict[str, Any]],
    discovery_gate_decision: str,
    discovery_gate_confidence: float,
    rank_correlation: float,
    divergent_features: List[str],
    direct_cause_features: List[str],
    causal_only_features: List[str],
    predictive_only_features: List[str],
) -> str:
    """Prepare causal discovery context for LLM interpretation.

    Args:
        causal_rankings: List of feature ranking dicts from DriverRanker
        discovery_gate_decision: Gate decision (accept/review/reject)
        discovery_gate_confidence: Gate confidence score
        rank_correlation: Spearman correlation between causal and predictive ranks
        divergent_features: Features with large rank differences
        direct_cause_features: Features that are direct causes of target
        causal_only_features: Features with causal but no predictive signal
        predictive_only_features: Features with predictive but no causal signal

    Returns:
        Formatted causal context string
    """
    parts = []

    # Discovery quality
    parts.append(f"**Discovery Quality**: {discovery_gate_decision} (confidence: {discovery_gate_confidence:.2f})")
    parts.append(f"**Causal-Predictive Rank Correlation**: {rank_correlation:.3f}")
    parts.append("")

    # Top causal rankings
    if causal_rankings:
        parts.append("**Top Features by Causal vs Predictive Importance**:")
        sorted_rankings = sorted(causal_rankings, key=lambda x: x.get("causal_rank", 999))
        for r in sorted_rankings[:10]:
            feature = r.get("feature_name", "unknown")
            causal_rank = r.get("causal_rank", "N/A")
            predictive_rank = r.get("predictive_rank", "N/A")
            rank_diff = r.get("rank_difference", 0)
            is_direct = r.get("is_direct_cause", False)

            direct_marker = " [DIRECT CAUSE]" if is_direct else ""
            parts.append(
                f"  - {feature}: causal_rank={causal_rank}, predictive_rank={predictive_rank}, "
                f"diff={rank_diff:+d}{direct_marker}"
            )
        parts.append("")

    # Feature categorization
    if direct_cause_features:
        parts.append(f"**Direct Cause Features** (direct edge to target): {', '.join(direct_cause_features)}")

    if divergent_features:
        parts.append(f"**Divergent Features** (|rank_diff| > 3): {', '.join(divergent_features)}")

    if causal_only_features:
        parts.append(f"**Causal-Only Features** (causal signal, no predictive): {', '.join(causal_only_features)}")

    if predictive_only_features:
        parts.append(f"**Predictive-Only Features** (predictive signal, no causal): {', '.join(predictive_only_features)}")

    return "\n".join(parts)


def _parse_interpretation_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON response from LLM.

    Args:
        response_text: Raw LLM response text

    Returns:
        Parsed interpretation dict
    """
    import json
    import re

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Fallback: use entire response
            json_str = response_text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: return empty structure
        return {
            "executive_summary": "Interpretation parsing failed",
            "feature_explanations": {},
            "key_insights": [],
            "recommendations": [],
            "cautions": ["Failed to parse LLM response"],
        }


def _interpret_interactions(
    top_interactions_raw: List, feature_explanations: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Generate interpretations for feature interactions.

    Args:
        top_interactions_raw: Top interactions from detector
        feature_explanations: Feature explanations from LLM

    Returns:
        List of interaction interpretation dicts
    """
    interpretations = []

    for feat1, feat2, strength in top_interactions_raw[:3]:  # Top 3
        interaction_type = "amplify each other" if strength > 0 else "oppose each other"

        interpretation = {
            "features": [feat1, feat2],
            "interaction_strength": float(strength),
            "interpretation": f"{feat1} and {feat2} {interaction_type} (strength: {abs(strength):.3f})",
        }

        interpretations.append(interpretation)

    return interpretations


def _build_complete_interpretation(
    executive_summary: str,
    feature_explanations: Dict[str, str],
    key_insights: List[str],
    recommendations: List[str],
    cautions: List[str],
) -> str:
    """Build complete interpretation text from components.

    Args:
        executive_summary: Executive summary
        feature_explanations: Feature explanations
        key_insights: Key insights
        recommendations: Recommendations
        cautions: Cautions

    Returns:
        Complete interpretation text
    """
    parts = []

    parts.append("## Executive Summary")
    parts.append(executive_summary)
    parts.append("")

    if key_insights:
        parts.append("## Key Insights")
        for insight in key_insights:
            parts.append(f"- {insight}")
        parts.append("")

    if feature_explanations:
        parts.append("## Top Features")
        for feature, explanation in list(feature_explanations.items())[:5]:
            parts.append(f"**{feature}**: {explanation}")
        parts.append("")

    if recommendations:
        parts.append("## Recommendations")
        for rec in recommendations:
            parts.append(f"- {rec}")
        parts.append("")

    if cautions:
        parts.append("## Cautions")
        for caution in cautions:
            parts.append(f"- {caution}")

    return "\n".join(parts)
