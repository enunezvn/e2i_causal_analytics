"""Causal Impact Agent - Tier 2 Hybrid Agent for causal effect estimation.

This package implements causal inference using DoWhy/EconML with natural language interpretation.
"""

from src.agents.causal_impact.agent import CausalImpactAgent
from src.agents.causal_impact.graph import create_causal_impact_graph
from src.agents.causal_impact.mlflow_tracker import (
    AnalysisContext,
    CausalImpactMetrics,
    CausalImpactMLflowTracker,
)
from src.agents.causal_impact.mlflow_tracker import (
    create_tracker as create_mlflow_tracker,
)
from src.agents.causal_impact.opik_tracer import (
    AnalysisTraceContext,
    CausalImpactOpikTracer,
    NodeSpanContext,
    get_causal_impact_tracer,
)
from src.agents.causal_impact.opik_tracer import (
    reset_tracer as reset_opik_tracer,
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
    # Opik tracing
    "CausalImpactOpikTracer",
    "AnalysisTraceContext",
    "NodeSpanContext",
    "get_causal_impact_tracer",
    "reset_opik_tracer",
]
