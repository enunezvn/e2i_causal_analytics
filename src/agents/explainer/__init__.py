"""
E2I Explainer Agent
Version: 4.2
Tier: 5 (Self-Improvement)

Natural language explanations for complex analyses.
"""

from .agent import (
    ExplainerAgent,
    ExplainerInput,
    ExplainerOutput,
    explain_analysis,
)
from .graph import build_explainer_graph, build_simple_explainer_graph
from .mlflow_tracker import (
    ExplainerMLflowTracker,
    ExplainerMetrics,
    ExplanationContext,
    create_tracker as create_mlflow_tracker,
)
from .state import (
    AnalysisContext,
    ExplainerState,
    Insight,
    NarrativeSection,
)

__all__ = [
    # Agent
    "ExplainerAgent",
    "ExplainerInput",
    "ExplainerOutput",
    "explain_analysis",
    # Graph
    "build_explainer_graph",
    "build_simple_explainer_graph",
    # MLflow Tracking
    "ExplainerMLflowTracker",
    "ExplainerMetrics",
    "ExplanationContext",
    "create_mlflow_tracker",
    # State
    "AnalysisContext",
    "ExplainerState",
    "Insight",
    "NarrativeSection",
]
