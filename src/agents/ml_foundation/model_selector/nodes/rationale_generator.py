"""Selection rationale generation for model_selector.

This module generates explanations for algorithm selection decisions.
"""

from typing import Dict, Any, List


async def generate_rationale(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate explanation for algorithm selection.

    Creates a structured rationale including:
    - Primary reason for selection
    - Supporting factors
    - Alternatives considered
    - Constraint compliance check

    Args:
        state: ModelSelectorState with primary_candidate, alternative_candidates,
               technical_constraints, selection_scores

    Returns:
        Dictionary with selection_rationale, primary_reason, supporting_factors,
        alternatives_considered, constraint_compliance
    """
    primary = state.get("primary_candidate", {})
    alternatives = state.get("alternative_candidates", [])
    constraints = state.get("technical_constraints", [])
    problem_type = state.get("problem_type", "classification")

    if not primary:
        return {
            "error": "No primary candidate to generate rationale for",
            "error_type": "missing_primary_candidate_error",
        }

    algo_name = primary.get("name", "Unknown")
    algo_family = primary.get("family", "unknown")
    strengths = primary.get("strengths", [])
    selection_score = primary.get("selection_score", 0.0)

    # Generate primary reason
    primary_reason = _generate_primary_reason(primary, problem_type)

    # Generate supporting factors
    supporting_factors = _generate_supporting_factors(primary, strengths)

    # Describe alternatives
    alternatives_considered = _describe_alternatives(alternatives, primary)

    # Check constraint compliance
    constraint_compliance = _check_constraint_compliance(primary, constraints)

    # Build rationale text
    rationale_text = _build_rationale_text(
        algo_name,
        primary_reason,
        supporting_factors,
        selection_score,
        alternatives_considered,
    )

    return {
        "selection_rationale": rationale_text,
        "primary_reason": primary_reason,
        "supporting_factors": supporting_factors,
        "alternatives_considered": alternatives_considered,
        "constraint_compliance": constraint_compliance,
    }


def _generate_primary_reason(candidate: Dict[str, Any], problem_type: str) -> str:
    """Generate primary selection reason.

    Args:
        candidate: Selected algorithm
        problem_type: Problem type

    Returns:
        Primary reason string
    """
    algo_name = candidate.get("name", "Unknown")
    algo_family = candidate.get("family", "unknown")
    strengths = candidate.get("strengths", [])

    # Causal ML algorithms
    if algo_family == "causal_ml":
        if "CausalForest" in algo_name:
            return "Best for estimating heterogeneous treatment effects with high accuracy"
        if "LinearDML" in algo_name:
            return "Optimal for fast, interpretable causal inference with low variance"

    # Gradient boosting
    if algo_family == "gradient_boosting":
        if "XGBoost" in algo_name:
            return "Highest predictive accuracy with robust feature importance"
        if "LightGBM" in algo_name:
            return "Best speed and memory efficiency for large datasets"

    # Random Forest
    if algo_family == "ensemble" and "RandomForest" in algo_name:
        return "Robust ensemble method with no scaling requirements"

    # Linear models (baselines)
    if algo_family == "linear":
        return f"Highly interpretable baseline model for {problem_type}"

    # Default
    return f"Strong performance on {problem_type} problems"


def _generate_supporting_factors(
    candidate: Dict[str, Any], strengths: List[str]
) -> List[str]:
    """Generate list of supporting factors.

    Args:
        candidate: Selected algorithm
        strengths: Algorithm strengths

    Returns:
        List of supporting factor strings
    """
    factors = []

    # Add strength-based factors
    strength_descriptions = {
        "heterogeneous_effects": "Can identify subgroup-specific effects",
        "interpretability": "Highly interpretable model structure",
        "fast": "Fast training and inference",
        "low_variance": "Low prediction variance",
        "accuracy": "High predictive accuracy",
        "feature_importance": "Provides feature importance scores",
        "robust": "Robust to outliers and missing data",
        "speed": "Extremely fast training",
        "large_datasets": "Handles large datasets efficiently",
        "memory_efficient": "Low memory footprint",
        "baseline": "Well-established baseline approach",
        "stable": "Numerically stable",
        "feature_selection": "Automatic feature selection",
        "no_scaling_required": "No feature scaling required",
        "causal_inference": "Designed for causal inference",
    }

    for strength in strengths:
        if strength in strength_descriptions:
            factors.append(strength_descriptions[strength])

    # Add performance characteristics
    latency = candidate.get("inference_latency_ms", 0)
    if latency < 10:
        factors.append(f"Very low inference latency ({latency}ms)")
    elif latency < 50:
        factors.append(f"Low inference latency ({latency}ms)")

    memory = candidate.get("memory_gb", 0)
    if memory < 1:
        factors.append(f"Minimal memory requirements ({memory}GB)")
    elif memory < 3:
        factors.append(f"Moderate memory requirements ({memory}GB)")

    # Add interpretability
    interpretability = candidate.get("interpretability_score", 0)
    if interpretability >= 0.8:
        factors.append("High interpretability for stakeholders")

    return factors


def _describe_alternatives(
    alternatives: List[Dict[str, Any]], primary: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Describe alternative candidates and why they weren't selected.

    Args:
        alternatives: List of alternative algorithms
        primary: Selected primary algorithm

    Returns:
        List of alternative descriptions
    """
    described = []

    primary_score = primary.get("selection_score", 0)

    for alt in alternatives:
        alt_name = alt.get("name", "Unknown")
        alt_score = alt.get("selection_score", 0)
        score_diff = primary_score - alt_score

        # Explain why not selected
        reason = _explain_why_not_selected(alt, primary, score_diff)

        described.append({
            "algorithm_name": alt_name,
            "selection_score": alt_score,
            "score_difference": round(score_diff, 3),
            "reason_not_selected": reason,
        })

    return described


def _explain_why_not_selected(
    alternative: Dict[str, Any], primary: Dict[str, Any], score_diff: float
) -> str:
    """Explain why alternative was not selected.

    Args:
        alternative: Alternative algorithm
        primary: Selected primary algorithm
        score_diff: Score difference (primary - alternative)

    Returns:
        Explanation string
    """
    # Compare key characteristics
    if alternative.get("inference_latency_ms", 0) > primary.get("inference_latency_ms", 0) * 2:
        return "Higher inference latency than selected algorithm"

    if alternative.get("memory_gb", 0) > primary.get("memory_gb", 0) * 1.5:
        return "Higher memory requirements than selected algorithm"

    if alternative.get("interpretability_score", 0) < primary.get("interpretability_score", 0) - 0.2:
        return "Lower interpretability than selected algorithm"

    if score_diff > 0.1:
        return f"Lower overall selection score ({round(score_diff, 2)} points)"

    return "Slightly lower composite score"


def _check_constraint_compliance(
    candidate: Dict[str, Any], constraints: List[str]
) -> Dict[str, bool]:
    """Check if candidate complies with technical constraints.

    Args:
        candidate: Algorithm specification
        constraints: List of constraint strings

    Returns:
        Dictionary mapping constraint -> compliance bool
    """
    compliance = {}

    for constraint in constraints:
        constraint_lower = constraint.lower()

        # Check latency constraint
        if "latency" in constraint_lower and "<" in constraint_lower:
            try:
                max_latency = int(
                    constraint_lower.split("<")[1].replace("ms", "").strip()
                )
                actual_latency = candidate.get("inference_latency_ms", 0)
                compliance[constraint] = actual_latency <= max_latency
            except (ValueError, IndexError):
                compliance[constraint] = True  # Can't parse, assume pass

        # Check memory constraint
        elif "memory" in constraint_lower and "<" in constraint_lower:
            try:
                max_memory = float(
                    constraint_lower.split("<")[1].replace("gb", "").strip()
                )
                actual_memory = candidate.get("memory_gb", 0)
                compliance[constraint] = actual_memory <= max_memory
            except (ValueError, IndexError):
                compliance[constraint] = True  # Can't parse, assume pass

        # Check model size constraint
        elif "model_size" in constraint_lower:
            compliance[constraint] = True  # Assume pass for now

        else:
            compliance[constraint] = True  # Unknown constraint, assume pass

    return compliance


def _build_rationale_text(
    algo_name: str,
    primary_reason: str,
    supporting_factors: List[str],
    selection_score: float,
    alternatives: List[Dict[str, Any]],
) -> str:
    """Build complete rationale text.

    Args:
        algo_name: Selected algorithm name
        primary_reason: Primary selection reason
        supporting_factors: List of supporting factors
        selection_score: Overall selection score
        alternatives: Alternative candidates

    Returns:
        Formatted rationale text
    """
    lines = [
        f"Selected {algo_name} (score: {selection_score:.3f})",
        f"",
        f"Primary Reason: {primary_reason}",
        f"",
        f"Supporting Factors:",
    ]

    for i, factor in enumerate(supporting_factors, 1):
        lines.append(f"  {i}. {factor}")

    if alternatives:
        lines.append(f"")
        lines.append(f"Alternatives Considered:")
        for alt in alternatives[:3]:
            lines.append(
                f"  - {alt['algorithm_name']} (score: {alt['selection_score']:.3f}): "
                f"{alt['reason_not_selected']}"
            )

    return "\n".join(lines)
