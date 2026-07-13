# tests/unit/test_api/test_causal_randomized_flag.py
"""API layer declares ``randomized_design`` from DESIGN knowledge
(post-#1217 e-value RCT-gate follow-up).

The declaration is per-TREATMENT, not per-dataset: nba_triggers carries BOTH
the randomized holdout (``control_group_flag``) and ``acceptance_status``,
which is a rep CHOICE — only the former may relax the unmeasured-confounding
gate. Fail-closed everywhere else.
"""

from __future__ import annotations

import pytest

from src.api.routes.causal import _is_randomized_treatment
from src.api.schemas.causal import AgentCausalAnalysisRequest


class TestIsRandomizedTreatment:
    def test_nba_holdout_flag_is_randomized(self):
        assert _is_randomized_treatment("nba_triggers", "control_group_flag") is True

    def test_acceptance_status_is_a_choice_not_randomized(self):
        assert _is_randomized_treatment("nba_triggers", "acceptance_status") is False

    def test_observational_datasets_are_not_randomized(self):
        assert _is_randomized_treatment("patient_journeys", "treatment_initiated") is False
        assert _is_randomized_treatment("hcp_adoption", "treatment_arm") is False

    def test_unknown_dataset_fails_closed(self):
        assert _is_randomized_treatment("nonexistent", "control_group_flag") is False


class _MemStore:
    def __init__(self) -> None:
        self._d: dict = {}

    async def get(self, key):
        return self._d.get(key)

    async def set(self, key, value):
        self._d[key] = value


@pytest.mark.asyncio
async def test_task_threads_randomized_design_into_initial_state(monkeypatch):
    """_run_agent_analysis_task must set initial_state['randomized_design']
    from the request's dataset+treatment so the graph (a declared state
    channel — LangGraph drops undeclared keys) can hand it to refutation."""
    import pandas as pd

    import src.agents.causal_impact.graph as graph_mod
    from src.api.routes import causal as causal_routes

    captured: dict = {}

    class _FakeGraph:
        async def ainvoke(self, state, **kwargs):
            captured.update(state)
            raise RuntimeError("stop after capture")

    monkeypatch.setattr(graph_mod, "create_causal_impact_graph", lambda: _FakeGraph())
    monkeypatch.setattr(causal_routes, "_agent_analysis_store", _MemStore())

    df = pd.DataFrame({"control_group_flag": [0, 1], "action_taken": [0, 1]})
    req = AgentCausalAnalysisRequest(
        treatment_var="control_group_flag",
        outcome_var="action_taken",
        dataset="nba_triggers",
    )
    await causal_routes._run_agent_analysis_task("aid-rct", req, df, [], "live")
    assert captured.get("randomized_design") is True


@pytest.mark.asyncio
async def test_task_defaults_randomized_design_false(monkeypatch):
    import pandas as pd

    import src.agents.causal_impact.graph as graph_mod
    from src.api.routes import causal as causal_routes

    captured: dict = {}

    class _FakeGraph:
        async def ainvoke(self, state, **kwargs):
            captured.update(state)
            raise RuntimeError("stop after capture")

    monkeypatch.setattr(graph_mod, "create_causal_impact_graph", lambda: _FakeGraph())
    monkeypatch.setattr(causal_routes, "_agent_analysis_store", _MemStore())

    df = pd.DataFrame({"treatment_initiated": [0, 1], "persistent_180d": [0, 1]})
    req = AgentCausalAnalysisRequest(
        treatment_var="treatment_initiated",
        outcome_var="persistent_180d",
        dataset="patient_journeys",
    )
    await causal_routes._run_agent_analysis_task("aid-obs", req, df, [], "live")
    assert captured.get("randomized_design") is False
