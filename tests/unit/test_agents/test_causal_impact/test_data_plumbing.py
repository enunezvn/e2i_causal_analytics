"""causal_impact honors an explicitly-passed DataFrame — issue #606 item D.

The Tier 1-5 harness mapper passes ``"data": <DataFrame>`` into the causal_impact
agent input (tier0_output_mapper.map_to_causal_impact), but ``_initialize_state``
previously dropped it, so the estimation node found no ``data_cache``
["estimation_data"] and (with data_source != "synthetic") failed — surfacing as
the ``semantic_validation`` quality-gate failure in CI. The fix seeds the
estimation cache from the passed data; inert when no "data" is passed (prod's
connector path populates the cache instead).

Pure state construction — no dowhy/econml run, no services.
"""

from __future__ import annotations

import pandas as pd
import pytest
from langgraph.graph import END, StateGraph

import src.agents.causal_impact.nodes.refutation as refmod
from src.agents.causal_impact.agent import CausalImpactAgent
from src.agents.causal_impact.nodes.refutation import RefutationNode
from src.agents.causal_impact.state import CausalImpactState


def _base_input() -> dict:
    return {
        "query": "What is the causal effect of hcp_visits on discontinuation_flag?",
        "query_id": "qid-606",
        "treatment_var": "hcp_visits",
        "outcome_var": "discontinuation_flag",
        "confounders": [],
        "data_source": "patient_journeys",
    }


def test_initialize_state_plumbs_passed_dataframe():
    agent = CausalImpactAgent(enable_mlflow=False)
    df = pd.DataFrame({"hcp_visits": [1, 2, 3], "discontinuation_flag": [0, 1, 0]})
    state = agent._initialize_state({**_base_input(), "data": df})
    assert state.get("data_cache", {}).get("estimation_data") is df


def test_initialize_state_without_data_leaves_cache_empty():
    """No 'data' passed -> empty cache (prod connector path fills it later)."""
    agent = CausalImpactAgent(enable_mlflow=False)
    state = agent._initialize_state(_base_input())
    assert state.get("data_cache", {}).get("estimation_data") is None


def test_estimation_data_survives_graph_state_channel():
    """Regression (#606): ``estimation_data`` MUST be a declared CausalImpactState
    field so LangGraph persists it from the estimation node to the refutation node.

    The estimation node returns ``{"estimation_data": <DataFrame>}`` (estimation.py
    success branch) and the refutation node reconstructs the DoWhy CausalModel from
    ``state["estimation_data"]`` (refutation.py). Before the fix the field was
    UNDECLARED in the TypedDict schema, so LangGraph dropped it between nodes —
    refutation fail-closed ("estimation_data passthrough is missing") and the agent
    returned an UNVALIDATED estimate for every real query that reached refutation.

    This builds a minimal 2-node graph over the real ``CausalImpactState`` schema
    and asserts the passthrough survives the channel (fast: no dowhy/econml run).
    """
    df = pd.DataFrame({"t": [0, 1, 0, 1], "y": [0, 1, 0, 1]})
    captured: dict = {}

    def producer(state: CausalImpactState) -> dict:
        return {"estimation_data": df, "status": "computing"}

    def consumer(state: CausalImpactState) -> dict:
        captured["estimation_data"] = state.get("estimation_data")
        return {"status": "completed"}

    workflow = StateGraph(CausalImpactState)
    workflow.add_node("producer", producer)
    workflow.add_node("consumer", consumer)
    workflow.set_entry_point("producer")
    workflow.add_edge("producer", "consumer")
    workflow.add_edge("consumer", END)
    app = workflow.compile()

    app.invoke(
        {
            "query": "q",
            "query_id": "1",
            "treatment_var": "t",
            "outcome_var": "y",
            "confounders": [],
            "data_source": "synthetic",
        }
    )

    assert captured["estimation_data"] is not None, (
        "estimation_data was dropped by the CausalImpactState channel — refutation "
        "would fail-close. Re-declare 'estimation_data' in CausalImpactState."
    )
    assert hasattr(captured["estimation_data"], "shape")


def test_refutation_node_merges_partial_config():
    """A partial refutation config merges onto DEFAULT_CONFIG per-key (#606).

    The harness passes only sim counts; other defaults (enabled/critical) and
    untouched test types must survive so the REAL suite still runs, just lighter.
    """
    node = RefutationNode(config={"bootstrap": {"num_bootstraps": 25}})
    cfg = node.runner.config
    assert cfg["bootstrap"]["num_bootstraps"] == 25  # overridden
    assert cfg["bootstrap"]["enabled"] is True  # default preserved
    assert cfg["placebo_treatment"]["num_simulations"] == 100  # untouched default


@pytest.mark.asyncio
async def test_refute_causal_estimate_honors_parameters_refutation_config(monkeypatch):
    """``parameters.refutation_config`` reaches RefutationNode (#606).

    This is the prod hook that lets the smoke harness run a bounded-but-REAL
    refutation suite. Inert when absent (config=None -> full DEFAULT_CONFIG).
    """
    captured: dict = {}

    class _SpyNode:
        def __init__(self, config=None, validation_repo=None, **_kw):
            captured["config"] = config

        async def execute(self, _state):
            return {"refutation_results": {}}

    monkeypatch.setattr(refmod, "RefutationNode", _SpyNode)

    light = {"bootstrap": {"num_bootstraps": 7}}
    await refmod.refute_causal_estimate({"parameters": {"refutation_config": light}})
    assert captured["config"] == light

    # Absent -> None (prod path: full default suite).
    await refmod.refute_causal_estimate({"parameters": {}})
    assert captured["config"] is None
