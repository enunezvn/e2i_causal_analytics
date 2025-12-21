"""Gap Analyzer Agent - Tier 2 Standard Agent.

Identifies ROI opportunities by detecting performance gaps across segments.

Exports:
- GapAnalyzerAgent: Main agent class
- GapAnalyzerState: Workflow state type
- create_gap_analyzer_graph: Graph factory
"""

from .agent import GapAnalyzerAgent
from .graph import create_gap_analyzer_graph
from .state import GapAnalyzerState

__all__ = [
    "GapAnalyzerAgent",
    "GapAnalyzerState",
    "create_gap_analyzer_graph",
]
