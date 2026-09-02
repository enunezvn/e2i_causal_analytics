"""Wave-51 Gap A: route-submitted state keys must survive LangGraph's input filter.

``StateGraph(CausalImpactState)`` silently DROPS any input key that is not a
declared channel of the state TypedDict — before any node runs. Wave-50's
live-cert proved the agent-analyze route's ``"discovery_guided": True`` was
filtered this way (the key was never declared), so guided discovery NEVER ran
via the API: live runs used the unguided ges+pc ensemble and the FCI latent
diagnostic was unreachable. These tests pin the fix at the filter itself.
"""

import pytest


@pytest.mark.unit
def test_discovery_guided_survives_langgraph_input_filter():
    """The agent endpoints submit ``discovery_guided=True``; the key must be a
    declared CausalImpactState channel or graph_builder silently reads False."""
    from langgraph.graph import END, StateGraph

    from src.agents.causal_impact.state import CausalImpactState

    seen: dict = {}

    def probe(state):
        seen["auto_discover"] = state.get("auto_discover")
        seen["discovery_guided"] = state.get("discovery_guided")
        seen["undeclared_sentinel"] = state.get("wave51_undeclared_sentinel", "FILTERED")
        return {}

    g = StateGraph(CausalImpactState)
    g.add_node("probe", probe)
    g.set_entry_point("probe")
    g.add_edge("probe", END)
    g.compile().invoke(
        {
            "query": "q",
            "query_id": "t1",
            "treatment_var": "treatment_arm",
            "outcome_var": "persistent_180d",
            "confounders": [],
            "data_source": "synthetic",
            "auto_discover": True,
            "discovery_guided": True,
            "wave51_undeclared_sentinel": True,
        }
    )

    # Positive control 1: the filter is ACTIVE — an undeclared key was dropped
    # (guards against this test passing vacuously if langgraph's input
    # semantics ever change to pass-through).
    assert seen["undeclared_sentinel"] == "FILTERED"
    # Positive control 2: a declared channel reaches the node unmodified.
    assert seen["auto_discover"] is True
    # The wave-50 regression: the route's guided-discovery opt-in must arrive.
    assert seen["discovery_guided"] is True
