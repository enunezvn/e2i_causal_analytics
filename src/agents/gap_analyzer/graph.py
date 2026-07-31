"""LangGraph workflow for Gap Analyzer Agent.

Defines the 6-node linear workflow:
audit_init → gap_detector → roi_calculator → instrument_analyzer → prioritizer → formatter

Performance target: <20s total execution time

ROI Methodology:
Uses ROICalculationService for full methodology implementation:
- 6 value drivers (TRx Lift, Patient ID, Action Rate, ITP, Data Quality, Drift)
- Bootstrap confidence intervals (1,000 simulations)
- Attribution framework (Full/Partial/Shared/Minimal)
- Risk adjustment (4 factors)

Observability:
- Audit chain recording for tamper-evident logging

Reference: docs/roi_methodology.md, src/services/roi_calculation.py
"""

from functools import partial
from typing import Optional

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.base.audit_chain_mixin import (
    add_audited_node,
    create_workflow_initializer,
)
from src.services.roi_calculation import ROICalculationService
from src.utils.audit_chain import AgentTier

from .nodes import (
    FormatterNode,
    GapDetectorNode,
    InstrumentAnalyzerNode,
    PrioritizerNode,
    ROICalculatorNode,
)
from .state import GapAnalyzerState


def create_gap_analyzer_graph(
    roi_service: Optional[ROICalculationService] = None,
    use_bootstrap: bool = True,
    n_simulations: int = 1000,
    use_mock: bool = False,
    include_synthetic: bool = False,
) -> CompiledStateGraph:
    """Create the Gap Analyzer LangGraph workflow.

    Workflow:
    0. audit_init: Initialize audit chain workflow (genesis block)
    1. gap_detector: Detect performance gaps across segments (parallel)
    2. roi_calculator: Calculate ROI for each gap using full methodology
    3. prioritizer: Rank and categorize opportunities
    4. formatter: Generate executive summary and insights

    Args:
        roi_service: Optional injected ROICalculationService (for testing/customization)
        use_bootstrap: Whether to compute bootstrap confidence intervals
        n_simulations: Number of Monte Carlo simulations for bootstrap
        use_mock: If True, use mock data connectors (for explicit testing only).
                 Default is False to use real Supabase data.
        include_synthetic: When True, the production connector opts in to reading
                 synthetic rows (the validation layer; #851). Default False keeps the
                 production read path real-mode isolated.

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize nodes
    gap_detector = GapDetectorNode(use_mock=use_mock, include_synthetic=include_synthetic)
    roi_calculator = ROICalculatorNode(
        roi_service=roi_service,
        use_bootstrap=use_bootstrap,
        n_simulations=n_simulations,
    )
    instrument_analyzer = InstrumentAnalyzerNode()
    prioritizer = PrioritizerNode()
    formatter = FormatterNode()

    # Create audit workflow initializer
    audit_initializer = create_workflow_initializer("gap_analyzer", AgentTier.CAUSAL_ANALYTICS)

    # Create graph
    workflow = StateGraph(GapAnalyzerState)

    # Add nodes. Business nodes are wrapped via add_audited_node so each emits a
    # real timed audit entry (duration_ms) -> populates the analytics latency panel.
    timed = partial(
        add_audited_node, agent_name="gap_analyzer", agent_tier=AgentTier.CAUSAL_ANALYTICS
    )
    workflow.add_node("audit_init", audit_initializer)  # type: ignore[type-var,arg-type,call-overload]  # Initialize audit chain (genesis)
    timed(workflow, "gap_detector", gap_detector.execute)
    timed(workflow, "roi_calculator", roi_calculator.execute)
    timed(workflow, "instrument_analyzer", instrument_analyzer.execute)  # #357
    timed(workflow, "prioritizer", prioritizer.execute)
    timed(workflow, "formatter", formatter.execute)

    # Define linear flow starting with audit initialization.
    # #357: instrument_analyzer runs between roi_calculator and prioritizer so that
    # instrument_strength_by_feature is populated before prioritization applies the bonus.
    workflow.set_entry_point("audit_init")
    workflow.add_edge("audit_init", "gap_detector")
    workflow.add_edge("gap_detector", "roi_calculator")
    workflow.add_edge("roi_calculator", "instrument_analyzer")
    workflow.add_edge("instrument_analyzer", "prioritizer")
    workflow.add_edge("prioritizer", "formatter")
    workflow.add_edge("formatter", END)

    # Compile graph. checkpointer=False: state carries a DataFrame passthrough
    # (tier0_data) and this graph runs as a subgraph of the checkpointed
    # chatbot graph on the chat path — a bare compile() inherits the parent's
    # Redis checkpointer whose ormsgpack serde cannot serialize DataFrames
    # (#1351 live-unmasked, same class as causal_impact).
    return workflow.compile(checkpointer=False)
