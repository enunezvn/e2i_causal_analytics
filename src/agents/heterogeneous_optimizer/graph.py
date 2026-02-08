"""LangGraph workflow for Heterogeneous Optimizer Agent.

Defines the 6-node workflow (B9.4: with hierarchical nesting):
    audit_init → estimate_cate → analyze_segments → hierarchical_analysis → learn_policy → generate_profiles

The hierarchical_analysis node (B9.4) computes segment-level CATE estimates
using EconML within CausalML uplift segments, with nested confidence intervals.

Observability:
- Audit chain recording for tamper-evident logging
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.base.audit_chain_mixin import create_workflow_initializer
from src.utils.audit_chain import AgentTier

from .nodes.cate_estimator import CATEEstimatorNode
from .nodes.hierarchical_analyzer import HierarchicalAnalyzerNode
from .nodes.policy_learner import PolicyLearnerNode
from .nodes.profile_generator import ProfileGeneratorNode
from .nodes.segment_analyzer import SegmentAnalyzerNode
from .state import HeterogeneousOptimizerState


async def error_handler_node(
    state: HeterogeneousOptimizerState,
) -> HeterogeneousOptimizerState:
    """Handle errors gracefully."""
    errors = state.get("errors", [])

    return {
        **state,
        "executive_summary": "Heterogeneous effect analysis could not be completed.",
        "key_insights": [f"Error: {e.get('error', 'Unknown')}" for e in errors],
        "status": "failed",
    }


def create_heterogeneous_optimizer_graph(
    data_connector=None,
    enable_hierarchical: bool = True,
) -> CompiledStateGraph:
    """Create the Heterogeneous Optimizer agent LangGraph workflow.

    Workflow (with hierarchical enabled - default):
        0. audit_init: Initialize audit chain workflow (genesis block)
        1. estimate_cate: Estimate CATE using EconML CausalForestDML
        2. analyze_segments: Identify high/low responder segments
        3. hierarchical_analysis: Compute segment-level CATE with nested CIs (B9.4)
        4. learn_policy: Generate optimal treatment allocation policy
        5. generate_profiles: Create visualization data and summaries

    Workflow (without hierarchical):
        0. audit_init: Initialize audit chain workflow (genesis block)
        1. estimate_cate: Estimate CATE using EconML CausalForestDML
        2. analyze_segments: Identify high/low responder segments
        3. learn_policy: Generate optimal treatment allocation policy
        4. generate_profiles: Create visualization data and summaries

    Args:
        data_connector: Data connector for fetching data (optional, uses mock if None)
        enable_hierarchical: Whether to include hierarchical analysis node (default: True)

    Returns:
        Compiled LangGraph workflow
    """

    # Initialize nodes
    cate_estimator = CATEEstimatorNode(data_connector)
    segment_analyzer = SegmentAnalyzerNode()
    hierarchical_analyzer = HierarchicalAnalyzerNode() if enable_hierarchical else None
    policy_learner = PolicyLearnerNode()
    profile_generator = ProfileGeneratorNode()

    # Create audit workflow initializer
    audit_initializer = create_workflow_initializer(
        "heterogeneous_optimizer", AgentTier.CAUSAL_ANALYTICS
    )

    # Build graph
    workflow = StateGraph(HeterogeneousOptimizerState)

    # Add nodes
    workflow.add_node("audit_init", audit_initializer)  # type: ignore[type-var,arg-type,call-overload]  # Initialize audit chain
    workflow.add_node("estimate_cate", cate_estimator.execute)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("analyze_segments", segment_analyzer.execute)  # type: ignore[type-var,arg-type,call-overload]
    if enable_hierarchical:
        workflow.add_node("hierarchical_analysis", hierarchical_analyzer.execute)  # type: ignore[type-var,arg-type,call-overload,union-attr]
    workflow.add_node("learn_policy", policy_learner.execute)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("generate_profiles", profile_generator.execute)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("error_handler", error_handler_node)  # type: ignore[type-var,arg-type,call-overload]

    # Entry point - start with audit initialization
    workflow.set_entry_point("audit_init")

    # Linear edge from audit_init to estimate_cate
    workflow.add_edge("audit_init", "estimate_cate")

    # Conditional edges for error handling
    workflow.add_conditional_edges(
        "estimate_cate",
        lambda s: "error" if s.get("status") == "failed" else "analyze_segments",
        {"analyze_segments": "analyze_segments", "error": "error_handler"},
    )

    if enable_hierarchical:
        # analyze_segments → hierarchical_analysis → learn_policy
        workflow.add_conditional_edges(
            "analyze_segments",
            lambda s: "error" if s.get("status") == "failed" else "hierarchical_analysis",
            {"hierarchical_analysis": "hierarchical_analysis", "error": "error_handler"},
        )
        # hierarchical_analysis always proceeds to learn_policy (failures are non-fatal)
        workflow.add_edge("hierarchical_analysis", "learn_policy")
    else:
        # analyze_segments → learn_policy (original flow)
        workflow.add_conditional_edges(
            "analyze_segments",
            lambda s: "error" if s.get("status") == "failed" else "learn_policy",
            {"learn_policy": "learn_policy", "error": "error_handler"},
        )

    # Direct edges
    workflow.add_edge("learn_policy", "generate_profiles")
    workflow.add_edge("generate_profiles", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()
