"""Gap Analyzer Agent - Tier 2 Standard Agent.

Identifies ROI opportunities by detecting performance gaps across segments.

Exports:
- GapAnalyzerAgent: Main agent class
- GapAnalyzerState: Workflow state type
- create_gap_analyzer_graph: Graph factory
- GapAnalyzerMLflowTracker: MLflow tracking integration
- GapAnalyzerOpikTracer: Opik distributed tracing
"""

from .agent import GapAnalyzerAgent
from .graph import create_gap_analyzer_graph
from .mlflow_tracker import (
    GapAnalysisContext,
    GapAnalyzerMetrics,
    GapAnalyzerMLflowTracker,
)
from .mlflow_tracker import (
    create_tracker as create_mlflow_tracker,
)
from .opik_tracer import (
    GapAnalysisTraceContext,
    GapAnalyzerOpikTracer,
    NodeSpanContext,
    get_gap_analyzer_tracer,
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
    # Opik tracing
    "GapAnalyzerOpikTracer",
    "GapAnalysisTraceContext",
    "NodeSpanContext",
    "get_gap_analyzer_tracer",
]
