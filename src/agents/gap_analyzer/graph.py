"""LangGraph workflow for Gap Analyzer Agent.

Defines the 4-node linear workflow:
gap_detector → roi_calculator → prioritizer → formatter

Performance target: <20s total execution time
"""

from langgraph.graph import StateGraph, END

from .state import GapAnalyzerState
from .nodes import (
    GapDetectorNode,
    ROICalculatorNode,
    PrioritizerNode,
    FormatterNode,
)


def create_gap_analyzer_graph() -> StateGraph:
    """Create the Gap Analyzer LangGraph workflow.

    Workflow:
    1. gap_detector: Detect performance gaps across segments (parallel)
    2. roi_calculator: Calculate ROI for each gap
    3. prioritizer: Rank and categorize opportunities
    4. formatter: Generate executive summary and insights

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize nodes
    gap_detector = GapDetectorNode()
    roi_calculator = ROICalculatorNode()
    prioritizer = PrioritizerNode()
    formatter = FormatterNode()

    # Create graph
    workflow = StateGraph(GapAnalyzerState)

    # Add nodes
    workflow.add_node("gap_detector", gap_detector.execute)
    workflow.add_node("roi_calculator", roi_calculator.execute)
    workflow.add_node("prioritizer", prioritizer.execute)
    workflow.add_node("formatter", formatter.execute)

    # Define linear flow
    workflow.set_entry_point("gap_detector")
    workflow.add_edge("gap_detector", "roi_calculator")
    workflow.add_edge("roi_calculator", "prioritizer")
    workflow.add_edge("prioritizer", "formatter")
    workflow.add_edge("formatter", END)

    # Compile graph
    return workflow.compile()
