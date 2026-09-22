"""Lane B (real-data causal estimation, 2026-09-22) — the agent route seeds its
structural-prior channel from an APPROVED structural-author review, and only
from one. Under the unit tree's dead-Supabase pin (#1420) the default lookup
cannot reach a store: the run must proceed prior-less with a warning line,
never raise, never anchor.
"""

from __future__ import annotations

import os

import pytest

from src.api.routes.causal.agent import _apply_approved_structural_prior
from src.api.schemas.causal import AgentCausalAnalysisRequest
from src.repositories.expert_review import estimand_key_for

T, Y = "treatment_dupixent", "persistent_at_180d_g28"


def _row(status="approved"):
    from src.causal_engine.dag_hash import compute_adjustment_set_hash, compute_dag_hash

    snap = {
        "nodes": [T, Y, "age_at_index", "payer_category"],
        "edges": [["age_at_index", T], ["age_at_index", Y], ["payer_category", T], [T, Y]],
        "treatment_nodes": [T],
        "outcome_nodes": [Y],
        "adjustment_sets": [["age_at_index"]],
    }
    dag_hash = compute_dag_hash(causal_graph=snap)
    feats = [
        {
            "feature": "age_at_index",
            "fragment_role": "confounder",
            "cohort_role": "confounder",
            "leak_verdict": False,
            "leak_source": None,
            "review_required": False,
        },
        {
            "feature": "payer_category",
            "fragment_role": "instrument",
            "cohort_role": "instrument",
            "leak_verdict": False,
            "leak_source": None,
            "review_required": False,
        },
    ]
    return {
        "review_id": "33333333-3333-3333-3333-333333333333",
        "review_type": "initial_dag",
        "approval_status": status,
        "valid_until": None,
        "dag_version_hash": dag_hash,
        "adjustment_set_hash": compute_adjustment_set_hash(snap["adjustment_sets"]),
        "treatment_variable": T,
        "outcome_variable": Y,
        "estimand_key": estimand_key_for(None, T, Y),
        "dag_structure_json": dict(snap, dag_version_hash=dag_hash),
        "agent_assessment_json": {
            "structural_author": {
                "dag_version_hash": dag_hash,
                "adjustment_set_hash": compute_adjustment_set_hash(snap["adjustment_sets"]),
                "adjustment_set": ["age_at_index"],
                "features": feats,
                "manifest": "optum_mart",
                "treatment": T,
                "outcome": Y,
            }
        },
    }


class _FakeRepo:
    def __init__(self, rows):
        self.rows = rows

    async def get_reviews_for_estimand(self, estimand_key, include_expired=True):
        return [r for r in self.rows if r["estimand_key"] == estimand_key]


def _state():
    return {"anchored_confounders": [], "modeled_confounders": ["age_at_index"], "warnings": []}


def _request():
    return AgentCausalAnalysisRequest(treatment_var=T, outcome_var=Y)


@pytest.fixture
def declared_manifest(monkeypatch):
    """A dataset spec that declares the manifest the review was authored for
    (Lane A/E register ``feature_manifest_source`` on the real dataset; on
    this tree no spec declares one, so the hook is dark by construction)."""
    from src.api.routes.causal import agent as agent_mod

    specs = dict(agent_mod._CAUSAL_DATASET_SPECS)
    specs["optum_biologic_persistence"] = {
        "treatment": [T],
        "outcome": [Y],
        "covariates": ["age_at_index", "payer_category"],
        "feature_manifest_source": "optum_mart",
    }
    monkeypatch.setattr(agent_mod, "_CAUSAL_DATASET_SPECS", specs)
    return AgentCausalAnalysisRequest(
        dataset="optum_biologic_persistence", treatment_var=T, outcome_var=Y
    )


@pytest.mark.unit
async def test_approved_structural_review_anchors_confounders(declared_manifest):
    async def factory():
        return _FakeRepo([_row()])

    state = _state()
    await _apply_approved_structural_prior(
        state, declared_manifest, ["age_at_index", "payer_category"], repo_factory=factory
    )
    assert state["anchored_confounders"] == ["age_at_index"]
    assert state["approved_structure_roles"] == {
        "age_at_index": "confounder",
        "payer_category": "instrument",
    }
    assert state["warnings"][0].startswith("structural prior: approved expert review 33333333")


@pytest.mark.unit
async def test_dataset_without_a_declared_manifest_never_takes_a_prior():
    """codex r2 HIGH 3: the default (patient_journeys) request must NOT pick up
    an optum_mart review that happens to share (T, Y) names."""
    calls = []

    async def factory():
        calls.append(1)
        return _FakeRepo([_row()])

    state = _state()
    await _apply_approved_structural_prior(
        state, _request(), ["age_at_index", "payer_category"], repo_factory=factory
    )
    assert state["anchored_confounders"] == []
    assert "approved_structure_roles" not in state
    assert calls == []  # the store is not even consulted
    assert state["warnings"] == [
        "structural prior not applied: the dataset declares no feature_manifest_source, "
        "so no authored structure can be matched to it"
    ]


@pytest.mark.unit
async def test_review_for_another_manifest_is_refused(declared_manifest):
    row = _row()
    row["agent_assessment_json"]["structural_author"]["manifest"] = "csu"

    async def factory():
        return _FakeRepo([row])

    state = _state()
    await _apply_approved_structural_prior(
        state, declared_manifest, ["age_at_index"], repo_factory=factory
    )
    assert state["anchored_confounders"] == []
    assert state["warnings"] == []  # filtered out at lookup: no matching review


@pytest.mark.unit
def test_approved_structure_roles_survives_the_langgraph_boundary():
    """codex r2 MED 2: the channel must be DECLARED on CausalImpactState, or
    LangGraph drops it at the graph boundary. Proven through a compiled graph,
    not by inspecting the helper's dict."""
    from langgraph.graph import END, START, StateGraph

    from src.agents.causal_impact.state import CausalImpactState

    seen = {}

    def _node(state):
        seen["roles"] = state.get("approved_structure_roles")
        seen["anchored"] = state.get("anchored_confounders")
        seen["excluded"] = state.get("approved_leak_exclusions")
        return {}

    g = StateGraph(CausalImpactState)
    g.add_node("probe", _node)
    g.add_edge(START, "probe")
    g.add_edge("probe", END)
    out = g.compile().invoke(
        {
            "approved_structure_roles": {"age_at_index": "confounder"},
            "anchored_confounders": ["age_at_index"],
            "approved_leak_exclusions": ["post_index_visits"],
            "warnings": [],
        }
    )
    assert seen["roles"] == {"age_at_index": "confounder"}
    assert seen["anchored"] == ["age_at_index"]
    assert out["approved_structure_roles"] == {"age_at_index": "confounder"}
    assert out["approved_leak_exclusions"] == ["post_index_visits"]


@pytest.mark.unit
async def test_pending_machine_review_is_never_a_prior(declared_manifest):
    async def factory():
        return _FakeRepo([_row(status="pending")])

    state = _state()
    await _apply_approved_structural_prior(
        state, declared_manifest, ["age_at_index"], repo_factory=factory
    )
    assert state["anchored_confounders"] == []
    assert "approved_structure_roles" not in state
    assert state["warnings"] == []


@pytest.mark.unit
async def test_dead_supabase_pin_leaves_the_run_prior_less_with_a_warning(declared_manifest):
    # The unit conftest pins SUPABASE_URL to a dead endpoint (#1420): the default
    # repo factory must fail inside the lookup, not propagate.
    assert os.environ.get("SUPABASE_URL", "").startswith("http://127.0.0.1:1")
    state = _state()
    await _apply_approved_structural_prior(state, declared_manifest, ["age_at_index"])
    assert state["anchored_confounders"] == []
    assert "approved_structure_roles" not in state
    assert len(state["warnings"]) == 1
    assert state["warnings"][0].startswith("structural prior not consulted:")


@pytest.mark.unit
@pytest.mark.parametrize(
    "with_panel", [False, True], ids=["no_request_panel", "incomplete_request_panel"]
)
async def test_approved_leak_exclusion_leaves_channels_and_frame_at_the_hook(
    declared_manifest, with_panel
):
    """codex r4 HIGH 1: the approved review's leak verdict is enforced by the
    hook itself — before graph_builder, independent of whether the request
    carried a panel (and of what that panel says)."""
    import pandas as pd

    row = _row()
    row["agent_assessment_json"]["structural_author"]["features"].append(
        {
            "feature": "post_index_visits",
            "fragment_role": "confounder",
            "cohort_role": "confounder",
            "leak_verdict": True,
            "leak_source": "layer_3_high",
            "review_required": True,
        }
    )

    async def factory():
        return _FakeRepo([row])

    covs = ["age_at_index", "payer_category", "post_index_visits"]
    state = {
        "confounders": list(covs),
        "modeled_confounders": list(covs),
        "anchored_confounders": [],
        "warnings": [],
        "data_cache": {"estimation_data": pd.DataFrame({c: [0, 1] for c in [T, Y, *covs]})},
    }
    if with_panel:
        # A request panel that does NOT cover post_index_visits: it cannot
        # exclude what it never saw; the approved review still does.
        state["feature_role_panel"] = {"records": {"age_at_index": {"feature": "age_at_index"}}}
    await _apply_approved_structural_prior(state, declared_manifest, covs, repo_factory=factory)
    assert state["approved_leak_exclusions"] == ["post_index_visits"]
    for channel in ("confounders", "modeled_confounders", "anchored_confounders"):
        assert "post_index_visits" not in state[channel], channel
    assert "post_index_visits" not in state["data_cache"]["estimation_data"].columns
    assert state["anchored_confounders"] == ["age_at_index"]
