"""
Rubric criteria definitions for E2I Causal Analytics evaluation.

Defines the 5 evaluation criteria with weights and scoring guides:
- causal_validity (0.25)
- actionability (0.25)
- evidence_chain (0.20)
- regulatory_awareness (0.15)
- uncertainty_communication (0.15)
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RubricCriterion:
    """Single rubric criterion definition."""

    name: str
    weight: float
    description: str
    scoring_guide: Dict[int, str]


# Default E2I Causal Analytics rubric criteria
DEFAULT_CRITERIA: List[RubricCriterion] = [
    RubricCriterion(
        name="causal_validity",
        weight=0.25,
        description=(
            "Response correctly distinguishes causal claims from correlations. "
            "Uses appropriate causal language (e.g., 'contributed to', 'influenced by') "
            "rather than implying direct causation without evidence."
        ),
        scoring_guide={
            5: "Precise causal claims with explicit methodology references",
            4: "Accurate causal language with appropriate caveats",
            3: "Generally correct but occasional correlation/causation conflation",
            2: "Frequent causal overstatements without supporting evidence",
            1: "Claims direct causation without any causal analysis support",
        },
    ),
    RubricCriterion(
        name="actionability",
        weight=0.25,
        description=(
            "Response provides specific, implementable recommendations. "
            "Includes concrete next steps tied to the causal insights rather than "
            "generic advice."
        ),
        scoring_guide={
            5: "Specific actions with expected outcomes and owner assignments",
            4: "Clear actionable recommendations tied to insights",
            3: "Some actionable content but lacks specificity",
            2: "Mostly descriptive with minimal actionable guidance",
            1: "Pure analysis with no actionable recommendations",
        },
    ),
    RubricCriterion(
        name="evidence_chain",
        weight=0.20,
        description=(
            "Response traces insights back to specific data sources and metrics. "
            "Shows clear provenance from query to answer with verifiable references."
        ),
        scoring_guide={
            5: "Complete data lineage with specific metrics and dates",
            4: "Clear references to data sources and time periods",
            3: "General source attribution without specifics",
            2: "Vague references to 'data shows' without details",
            1: "No evidence chain, assertions without support",
        },
    ),
    RubricCriterion(
        name="regulatory_awareness",
        weight=0.15,
        description=(
            "Response acknowledges compliance boundaries (on-label only, no medical "
            "advice). Stays within commercial operations domain appropriate for "
            "pharmaceutical analytics."
        ),
        scoring_guide={
            5: "Proactively notes compliance considerations when relevant",
            4: "Stays clearly within commercial operations scope",
            3: "Generally appropriate but could be more explicit about boundaries",
            2: "Some content approaches or crosses compliance boundaries",
            1: "Contains off-label suggestions or medical advice",
        },
    ),
    RubricCriterion(
        name="uncertainty_communication",
        weight=0.15,
        description=(
            "Response appropriately communicates confidence levels and limitations. "
            "Uses hedging language for uncertain predictions and states known unknowns."
        ),
        scoring_guide={
            5: "Explicit confidence intervals and clear limitation statements",
            4: "Appropriate uncertainty language with caveats",
            3: "Some hedging but could be more precise about confidence",
            2: "Overly confident without acknowledging limitations",
            1: "Presents uncertain predictions as definitive facts",
        },
    ),
]


def get_default_criteria() -> List[RubricCriterion]:
    """Get default E2I rubric criteria."""
    return DEFAULT_CRITERIA.copy()


def get_criterion_by_name(name: str) -> RubricCriterion:
    """Get a specific criterion by name."""
    for criterion in DEFAULT_CRITERIA:
        if criterion.name == name:
            return criterion
    raise ValueError(f"Unknown criterion: {name}")


def get_total_weight() -> float:
    """Get total weight of all criteria (should be 1.0)."""
    return sum(c.weight for c in DEFAULT_CRITERIA)


def validate_weights() -> bool:
    """Validate that weights sum to 1.0."""
    total = get_total_weight()
    return abs(total - 1.0) < 0.001


# Decision thresholds
DEFAULT_THRESHOLDS = {
    "acceptable": 4.0,
    "suggestion": 3.0,
    "auto_update": 2.0,
}

# Override conditions
DEFAULT_OVERRIDE_CONDITIONS = [
    {
        "condition": "any_criterion_below",
        "threshold": 2.0,
        "action": "suggestion",
    },
]
