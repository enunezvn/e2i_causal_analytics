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

from src.agents.causal_impact.agent import CausalImpactAgent


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
