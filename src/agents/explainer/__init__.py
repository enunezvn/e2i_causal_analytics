"""
E2I Explainer Agent
Version: 4.3
Tier: 5 (Self-Improvement)

Natural language explanations for complex analyses.

Smart LLM Mode Selection (v4.3):
- Auto-detects complexity to decide LLM vs deterministic mode
- Configurable threshold and scoring weights
"""

from .agent import (
    ExplainerAgent,
    ExplainerInput,
    ExplainerOutput,
    explain_analysis,
)
from .config import (
    ComplexityScorer,
    ExplainerConfig,
    compute_complexity,
    get_default_config,
    set_default_config,
    should_use_llm,
)
from .graph import build_explainer_graph, build_simple_explainer_graph
from .mlflow_tracker import (
    ExplainerMetrics,
    ExplainerMLflowTracker,
    ExplanationContext,
)
from .mlflow_tracker import (
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
    # Configuration (v4.3)
    "ExplainerConfig",
    "ComplexityScorer",
    "compute_complexity",
    "should_use_llm",
    "get_default_config",
    "set_default_config",
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
