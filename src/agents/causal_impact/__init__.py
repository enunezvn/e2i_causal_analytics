"""Causal Impact Agent - Tier 2 Hybrid Agent for causal effect estimation.

This package implements causal inference using DoWhy/EconML with natural language interpretation.
"""

from src.agents.causal_impact.agent import CausalImpactAgent
from src.agents.causal_impact.state import (
    CausalImpactState,
    CausalImpactInput,
    CausalImpactOutput,
    CausalGraph,
    EstimationResult,
    RefutationResults,
    RefutationTest,
    SensitivityAnalysis,
    NaturalLanguageInterpretation,
)
from src.agents.causal_impact.graph import create_causal_impact_graph

__all__ = [
    "CausalImpactAgent",
    "CausalImpactState",
    "CausalImpactInput",
    "CausalImpactOutput",
    "CausalGraph",
    "EstimationResult",
    "RefutationResults",
    "RefutationTest",
    "SensitivityAnalysis",
    "NaturalLanguageInterpretation",
    "create_causal_impact_graph",
]
