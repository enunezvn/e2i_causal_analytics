"""LangGraph workflow for Heterogeneous Optimizer Agent.

Defines the 6-node workflow (B9.4: with hierarchical nesting):
    audit_init → estimate_cate → analyze_segments → hierarchical_analysis → learn_policy → generate_profiles

The hierarchical_analysis node (B9.4) computes segment-level CATE estimates
using EconML within CausalML uplift segments, with nested confidence intervals.

Observability:
- Audit chain recording for tamper-evident logging
"""

import logging
from functools import partial
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.base.audit_chain_mixin import (
    add_audited_node,
    create_workflow_initializer,
)
from src.utils.audit_chain import AgentTier

from .nodes.cate_estimator import CATEEstimatorNode
from .nodes.hierarchical_analyzer import HierarchicalAnalyzerNode
from .nodes.policy_learner import PolicyLearnerNode
from .nodes.profile_generator import ProfileGeneratorNode
from .nodes.segment_analyzer import SegmentAnalyzerNode
from .nodes.uplift_analyzer import UpliftAnalyzerNode
from .state import HeterogeneousOptimizerState

logger = logging.getLogger(__name__)


async def _run_uplift_nonfatal(
    uplift_node: UpliftAnalyzerNode,
    state: HeterogeneousOptimizerState,
) -> Dict[str, Any]:
    """Run the uplift analyzer as a NON-FATAL complementary step.

    Uplift (AUUC / Qini / targeting efficiency) enriches the HTE analysis but is
    NOT required for the CATE / responder / policy outputs. So any uplift failure
    — including the node's deliberate fail-closed no-data ``RuntimeError`` (F-013)
    — is downgraded to a warning HERE, at the graph boundary, so the run still
    surfaces everything else rather than aborting. This does NOT reintroduce the
    fabrication F-013 guarded against: the node itself still never returns mock
    data unless mock is explicitly env-gated, and the page always feeds it real
    ``tier0_data``. The orchestration choice (degrade vs abort) is the graph's;
    the node's standalone fail-closed contract is unchanged for direct callers.
    """
    try:
        out = await uplift_node.execute(state)
    except Exception as exc:  # noqa: BLE001 - complementary step is never fatal
        logger.warning(
            "uplift_analysis non-fatal failure; continuing without uplift metrics: %s",
            exc,
            extra={"node": "uplift_analyzer"},
        )
        return {"warnings": [f"Uplift analysis skipped: {exc}"]}
    # The node's OWN generic handler can RETURN an errors dict (without raising).
    # Those errors accumulate into state["errors"] and would flip the agent's
    # status to "failed" via _build_output — failing the WHOLE HTE run for a
    # complementary miss. Demote any uplift errors to warnings here (no
    # fabrication: the uplift result is simply absent / honest-empty, not faked).
    errors = out.get("errors")
    if errors:
        warnings = list(out.get("warnings") or [])
        warnings.extend(
            f"Uplift analysis issue: {e.get('error', e) if isinstance(e, dict) else e}"
            for e in errors
        )
        out = {k: v for k, v in out.items() if k != "errors"}
        out["warnings"] = warnings
    return out


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
    enable_uplift: bool = True,
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
        enable_uplift: Whether to include the uplift analysis node (default: True).
            Wired between hierarchical_analysis and learn_policy as a NON-FATAL
            step (AUUC/Qini/targeting). The node existed but was never wired in,
            so overall_auuc was never populated and the page's Uplift tab was
            structurally empty regardless of substrate.

    Returns:
        Compiled LangGraph workflow
    """

    # Resolve ONE shared data connector when the caller did not supply one, so
    # both data-fetching nodes (cate_estimator + hierarchical_analyzer) read the
    # SAME live substrate. Done HERE (not in the route) so the route's
    # import-guard / mock-fallback contract is unaffected and unit tests that
    # patch this factory never trigger real connector resolution. Falls back to
    # None — the original lazy / fail-closed node behavior — when a default
    # connector is unavailable (e.g. no Supabase creds in a unit-test env), so
    # this resolution can never itself raise out of graph construction.
    if data_connector is None:
        try:
            from .nodes.cate_estimator import _get_default_data_connector

            data_connector = _get_default_data_connector()
        except Exception as exc:  # pragma: no cover - depends on env/creds
            logger.warning(
                "create_heterogeneous_optimizer_graph: default data connector "
                "unavailable (%s); nodes will resolve lazily / fail-closed.",
                exc,
            )
            data_connector = None

    # Initialize nodes. Both data-fetching nodes (cate_estimator and
    # hierarchical_analyzer) receive the SAME connector so they read the same
    # live substrate and we avoid instantiating two Supabase clients. Passing
    # data_connector to hierarchical_analyzer is required: without it the node
    # had no real source and raised RuntimeError in production (mock forbidden),
    # failing every analysis after the CATE step (#30).
    cate_estimator = CATEEstimatorNode(data_connector)
    segment_analyzer = SegmentAnalyzerNode()
    hierarchical_analyzer = (
        HierarchicalAnalyzerNode(data_connector=data_connector) if enable_hierarchical else None
    )
    policy_learner = PolicyLearnerNode()
    profile_generator = ProfileGeneratorNode()
    # Uplift consumes the SAME shared connector as cate/hierarchical (and the
    # route's tier0_data passthrough), so it reads one substrate — never
    # fabricates, never double-fetches.
    uplift_analyzer = UpliftAnalyzerNode(data_connector=data_connector) if enable_uplift else None

    # Create audit workflow initializer
    audit_initializer = create_workflow_initializer(
        "heterogeneous_optimizer", AgentTier.CAUSAL_ANALYTICS
    )

    # Build graph
    workflow = StateGraph(HeterogeneousOptimizerState)

    # Add nodes. Business nodes are wrapped via add_audited_node so each emits a
    # real timed audit entry (duration_ms) -> populates the analytics latency panel.
    timed = partial(
        add_audited_node,
        agent_name="heterogeneous_optimizer",
        agent_tier=AgentTier.CAUSAL_ANALYTICS,
    )
    workflow.add_node("audit_init", audit_initializer)  # type: ignore[type-var,arg-type,call-overload]  # Initialize audit chain (genesis)
    timed(workflow, "estimate_cate", cate_estimator.execute)
    timed(workflow, "analyze_segments", segment_analyzer.execute)
    if enable_hierarchical:
        timed(workflow, "hierarchical_analysis", hierarchical_analyzer.execute)  # type: ignore[union-attr]
    if enable_uplift:
        assert uplift_analyzer is not None  # constructed above when enable_uplift
        # Wrapped so any uplift failure degrades to a warning (NON-FATAL); the
        # CATE/responder/policy outputs must survive an uplift miss.
        timed(workflow, "uplift_analysis", partial(_run_uplift_nonfatal, uplift_analyzer))
    timed(workflow, "learn_policy", policy_learner.execute)
    timed(workflow, "generate_profiles", profile_generator.execute)
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

    # Uplift (when enabled) is inserted just before learn_policy. ``pre_policy``
    # is whichever node feeds learn_policy, so the insertion is transparent to the
    # hierarchical-enabled/disabled branches below.
    pre_policy = "learn_policy"
    if enable_uplift:
        workflow.add_edge("uplift_analysis", "learn_policy")
        pre_policy = "uplift_analysis"

    if enable_hierarchical:
        # analyze_segments → hierarchical_analysis → [uplift_analysis] → learn_policy
        workflow.add_conditional_edges(
            "analyze_segments",
            lambda s: "error" if s.get("status") == "failed" else "hierarchical_analysis",
            {"hierarchical_analysis": "hierarchical_analysis", "error": "error_handler"},
        )
        # hierarchical_analysis always proceeds (failures are non-fatal)
        workflow.add_edge("hierarchical_analysis", pre_policy)
    else:
        # analyze_segments → [uplift_analysis] → learn_policy (original flow)
        workflow.add_conditional_edges(
            "analyze_segments",
            lambda s, _t=pre_policy: "error" if s.get("status") == "failed" else _t,
            {pre_policy: pre_policy, "error": "error_handler"},
        )

    # Direct edges
    workflow.add_edge("learn_policy", "generate_profiles")
    workflow.add_edge("generate_profiles", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()
