"""Gap Analyzer Agent - Tier 2 Standard Agent.

Identifies ROI opportunities by detecting performance gaps across segments.

Exports:
- GapAnalyzerAgent: Main agent class
- GapAnalyzerState: Workflow state type
- create_gap_analyzer_graph: Graph factory
- GapAnalyzerMLflowTracker: MLflow tracking integration
"""

from .agent import GapAnalyzerAgent
from .graph import create_gap_analyzer_graph
from .mlflow_tracker import (
    GapAnalyzerMLflowTracker,
    GapAnalyzerMetrics,
    GapAnalysisContext,
    create_tracker as create_mlflow_tracker,
)
from .state import GapAnalyzerState

__all__ = [
    "GapAnalyzerAgent",
    "GapAnalyzerState",
    "create_gap_analyzer_graph",
    # MLflow tracking
    "GapAnalyzerMLflowTracker",
    "GapAnalyzerMetrics",
    "GapAnalysisContext",
    "create_mlflow_tracker",
]
