"""Causal Impact Agent - Tier 2 Hybrid Agent for causal effect estimation.

This package implements causal inference using DoWhy/EconML with natural language interpretation.
"""

from src.agents.causal_impact.agent import CausalImpactAgent
from src.agents.causal_impact.graph import create_causal_impact_graph
from src.agents.causal_impact.mlflow_tracker import (
    CausalImpactMLflowTracker,
    CausalImpactMetrics,
    AnalysisContext,
    create_tracker as create_mlflow_tracker,
)
from src.agents.causal_impact.state import (
    CausalGraph,
    CausalImpactInput,
    CausalImpactOutput,
    CausalImpactState,
    EstimationResult,
    NaturalLanguageInterpretation,
    RefutationResults,
    RefutationTest,
    SensitivityAnalysis,
)

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
    # MLflow tracking
    "CausalImpactMLflowTracker",
    "CausalImpactMetrics",
    "AnalysisContext",
    "create_mlflow_tracker",
]
