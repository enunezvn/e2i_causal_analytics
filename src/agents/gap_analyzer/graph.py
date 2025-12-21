"""LangGraph workflow for Gap Analyzer Agent.

Defines the 4-node linear workflow:
gap_detector → roi_calculator → prioritizer → formatter

Performance target: <20s total execution time

ROI Methodology:
Uses ROICalculationService for full methodology implementation:
- 6 value drivers (TRx Lift, Patient ID, Action Rate, ITP, Data Quality, Drift)
- Bootstrap confidence intervals (1,000 simulations)
- Attribution framework (Full/Partial/Shared/Minimal)
- Risk adjustment (4 factors)

Reference: docs/roi_methodology.md, src/services/roi_calculation.py
"""

from typing import Optional

from langgraph.graph import END, StateGraph

from src.services.roi_calculation import ROICalculationService

from .nodes import (
    FormatterNode,
    GapDetectorNode,
    PrioritizerNode,
    ROICalculatorNode,
)
from .state import GapAnalyzerState


def create_gap_analyzer_graph(
    roi_service: Optional[ROICalculationService] = None,
    use_bootstrap: bool = True,
    n_simulations: int = 1000,
) -> StateGraph:
    """Create the Gap Analyzer LangGraph workflow.

    Workflow:
    1. gap_detector: Detect performance gaps across segments (parallel)
    2. roi_calculator: Calculate ROI for each gap using full methodology
    3. prioritizer: Rank and categorize opportunities
    4. formatter: Generate executive summary and insights

    Args:
        roi_service: Optional injected ROICalculationService (for testing/customization)
        use_bootstrap: Whether to compute bootstrap confidence intervals
        n_simulations: Number of Monte Carlo simulations for bootstrap

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize nodes
    gap_detector = GapDetectorNode()
    roi_calculator = ROICalculatorNode(
        roi_service=roi_service,
        use_bootstrap=use_bootstrap,
        n_simulations=n_simulations,
    )
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
