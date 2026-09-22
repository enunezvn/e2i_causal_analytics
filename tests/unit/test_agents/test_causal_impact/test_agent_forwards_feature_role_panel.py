"""Lane E item 3(d), codex r1 HIGH: the panel must reach the LIVE causal-agent path.

Before this fix ``feature_role_panel`` / ``approved_structure_roles`` were read
by graph_builder but produced by nobody: ``CausalImpactAgent._initialize_state``
did not forward them and the API request had no field for them, so in normal
agent execution every registry covariate was still adjusted blind — loadable,
not activated. These tests go through the agent's initializer and the route's
task (not ``GraphBuilderNode`` injection).

The producer stays the evidence-recorded script
(``scripts/measure_feature_role_panel.py``): building the panel inside a
request would hide a minutes-long Layer-3 run and, with Layer 4 on, paid LLM
calls; the caller passes the serialised panel instead.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.agents.causal_impact.agent import CausalImpactAgent

_PANEL = {
    "manifest_source": "optum_mart",
    "treatment": "copay_support",
    "outcome": "adherent_180d",
    "n_rows": 3,
    "features": ["insurance_access_score", "post_dx"],
    "records": {
        "insurance_access_score": {
            "feature": "insurance_access_score",
            "layer_1": {"verdict": "pre_index", "temporal_status": "pre_index"},
            "layer_2": {},
            "layer_3": {},
            "layer_4": {},
            "ensemble": {},
            "leak_verdict": False,
            "leak_source": None,
        },
        "post_dx": {
            "feature": "post_dx",
            "layer_1": {"verdict": "post_index", "temporal_status": "post_index"},
            "layer_2": {},
            "layer_3": {},
            "layer_4": {},
            "ensemble": {},
            "leak_verdict": True,
            "leak_source": "layer_1_post_index",
        },
    },
    "layer_activity": {},
    "activation_profile": {},
}


def _base_input() -> dict:
    return {
        "query": "effect of copay_support on adherent_180d",
        "treatment_var": "copay_support",
        "outcome_var": "adherent_180d",
        "confounders": ["insurance_access_score", "post_dx"],
        "data_source": "synthetic",
    }


def test_initialize_state_forwards_the_panel_and_approved_roles() -> None:
    agent = CausalImpactAgent(enable_mlflow=False)
    state = agent._initialize_state(
        {
            **_base_input(),
            "feature_role_panel": _PANEL,
            "approved_structure_roles": {"insurance_access_score": "confounder"},
        }
    )
    assert state["feature_role_panel"] == _PANEL
    assert state["approved_structure_roles"] == {"insurance_access_score": "confounder"}


def test_initialize_state_leaves_the_keys_absent_when_not_supplied() -> None:
    """Absent, not None: graph_builder keys off presence, and an absent
    ``anchored_confounders`` channel keeps pre-split callers' shape."""
    state = CausalImpactAgent(enable_mlflow=False)._initialize_state(_base_input())
    assert "feature_role_panel" not in state
    assert "approved_structure_roles" not in state


@pytest.mark.asyncio
async def test_route_task_forwards_the_request_panel_into_the_initial_state(monkeypatch) -> None:
    import src.agents.causal_impact.graph as graph_mod
    import src.api.routes.causal.agent as causal_routes
    from src.api.schemas.causal import AgentCausalAnalysisRequest

    captured: dict = {}

    class _FakeGraph:
        async def ainvoke(self, state, **kwargs):
            captured.update(state)
            raise RuntimeError("stop after capture")

    class _MemStore:
        def __init__(self) -> None:
            self._d: dict = {}

        async def get(self, key):
            return self._d.get(key)

        async def set(self, key, value):
            self._d[key] = value

    monkeypatch.setattr(graph_mod, "create_causal_impact_graph", lambda: _FakeGraph())
    monkeypatch.setattr(causal_routes, "_agent_analysis_store", _MemStore())
    df = pd.DataFrame(
        {
            "copay_support": [0.0, 1.0, 1.0],
            "adherent_180d": [0.0, 1.0, 0.0],
            "insurance_access_score": [0.2, 0.9, 0.5],
            "post_dx": [1.0, 0.0, 0.0],
        }
    )
    req = AgentCausalAnalysisRequest(
        treatment_var="copay_support",
        outcome_var="adherent_180d",
        dataset="patient_journeys",
        feature_role_panel=_PANEL,
        approved_structure_roles={"insurance_access_score": "confounder"},
    )
    await causal_routes._run_agent_analysis_task(
        "lane-e-forward",
        req,
        df,
        ["insurance_access_score", "post_dx"],
        "synthetic",
    )
    assert captured["feature_role_panel"] == _PANEL
    assert captured["approved_structure_roles"] == {"insurance_access_score": "confounder"}
    # Declared covariates are untouched at submit time: graph_builder is the
    # single place the panel narrows them (with its named warning).
    assert captured["modeled_confounders"] == ["insurance_access_score", "post_dx"]


@pytest.mark.asyncio
async def test_route_task_omits_the_keys_when_the_request_has_none(monkeypatch) -> None:
    import src.agents.causal_impact.graph as graph_mod
    import src.api.routes.causal.agent as causal_routes
    from src.api.schemas.causal import AgentCausalAnalysisRequest

    captured: dict = {}

    class _FakeGraph:
        async def ainvoke(self, state, **kwargs):
            captured.update(state)
            raise RuntimeError("stop after capture")

    class _MemStore:
        def __init__(self) -> None:
            self._d: dict = {}

        async def get(self, key):
            return self._d.get(key)

        async def set(self, key, value):
            self._d[key] = value

    monkeypatch.setattr(graph_mod, "create_causal_impact_graph", lambda: _FakeGraph())
    monkeypatch.setattr(causal_routes, "_agent_analysis_store", _MemStore())
    df = pd.DataFrame({"copay_support": [0.0, 1.0], "adherent_180d": [0.0, 1.0], "x": [0.1, 0.2]})
    req = AgentCausalAnalysisRequest(
        treatment_var="copay_support", outcome_var="adherent_180d", dataset="patient_journeys"
    )
    await causal_routes._run_agent_analysis_task("lane-e-absent", req, df, ["x"], "synthetic")
    assert "feature_role_panel" not in captured
    assert "approved_structure_roles" not in captured
