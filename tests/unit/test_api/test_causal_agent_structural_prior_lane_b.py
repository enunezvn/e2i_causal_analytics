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


@pytest.mark.unit
async def test_approved_structural_review_anchors_confounders():
    async def factory():
        return _FakeRepo([_row()])

    state = _state()
    await _apply_approved_structural_prior(
        state, _request(), ["age_at_index", "payer_category"], repo_factory=factory
    )
    assert state["anchored_confounders"] == ["age_at_index"]
    assert state["approved_structure_roles"] == {
        "age_at_index": "confounder",
        "payer_category": "instrument",
    }
    assert state["warnings"][0].startswith("structural prior: approved expert review 33333333")


@pytest.mark.unit
async def test_pending_machine_review_is_never_a_prior():
    async def factory():
        return _FakeRepo([_row(status="pending")])

    state = _state()
    await _apply_approved_structural_prior(
        state, _request(), ["age_at_index"], repo_factory=factory
    )
    assert state["anchored_confounders"] == []
    assert "approved_structure_roles" not in state
    assert state["warnings"] == []


@pytest.mark.unit
async def test_dead_supabase_pin_leaves_the_run_prior_less_with_a_warning():
    # The unit conftest pins SUPABASE_URL to a dead endpoint (#1420): the default
    # repo factory must fail inside the lookup, not propagate.
    assert os.environ.get("SUPABASE_URL", "").startswith("http://127.0.0.1:1")
    state = _state()
    await _apply_approved_structural_prior(state, _request(), ["age_at_index"])
    assert state["anchored_confounders"] == []
    assert "approved_structure_roles" not in state
    assert len(state["warnings"]) == 1
    assert state["warnings"][0].startswith("structural prior not consulted:")
