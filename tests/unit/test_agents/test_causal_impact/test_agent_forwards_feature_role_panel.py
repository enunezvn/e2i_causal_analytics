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


def _make_panel(treatment: str = "copay_support", outcome: str = "adherent_180d") -> dict:
    """A strict-valid serialised panel (codex r3: the route validates invariants)."""
    from src.causal_engine.feature_role_panel import FeatureRolePanel, FeatureRoleRecord

    def rec(name: str, *, post: bool) -> FeatureRoleRecord:
        return FeatureRoleRecord(
            feature=name,
            layer_1={
                "verdict": "post_index" if post else "pre_index",
                "temporal_status": "post_index" if post else "pre_index",
                "declared_safe": not post,
            },
            layer_2={"signal": "no_signal", "edges": [], "mode": "shadow"},
            layer_3={"ran": not post},
            layer_4={"fired": False},
            ensemble={"decided_by": "layer_1" if post else "adversarial"},
            leak_verdict=post,
            leak_source="layer_1_post_index" if post else None,
        )

    return FeatureRolePanel(
        manifest_source="optum_mart",
        treatment=treatment,
        outcome=outcome,
        n_rows=3,
        features=("insurance_access_score", "post_dx"),
        records={
            "insurance_access_score": rec("insurance_access_score", post=False),
            "post_dx": rec("post_dx", post=True),
        },
        layer_activity={},
        activation_profile={},
        built_at="2026-09-22T00:00:00+00:00",
    ).to_dict()


_PANEL = _make_panel()


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
    # The agent-level seam keeps BOTH keys: Lane B's server-side loader (an
    # approved review resolved by id) populates approved_structure_roles; the
    # public request carries only the panel.
    assert state["feature_role_panel"] == _PANEL
    assert state["approved_structure_roles"] == {"insurance_access_score": "confounder"}


def test_initialize_state_leaves_the_keys_absent_when_not_supplied() -> None:
    """Absent, not None: graph_builder keys off presence, and an absent
    ``anchored_confounders`` channel keeps pre-split callers' shape."""
    state = CausalImpactAgent(enable_mlflow=False)._initialize_state(_base_input())
    assert "feature_role_panel" not in state
    assert "approved_structure_roles" not in state


def _capture(monkeypatch) -> dict:
    import src.agents.causal_impact.graph as graph_mod
    import src.api.routes.causal.agent as causal_routes

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
    return captured


@pytest.mark.asyncio
async def test_route_task_forwards_the_request_panel_into_the_initial_state(monkeypatch) -> None:
    import src.api.routes.causal.agent as causal_routes
    from src.api.schemas.causal import AgentCausalAnalysisRequest

    captured = _capture(monkeypatch)
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
    )
    await causal_routes._run_agent_analysis_task(
        "lane-e-forward", req, df, ["insurance_access_score", "post_dx"], "synthetic"
    )
    from src.causal_engine.feature_role_panel import FeatureRolePanel

    # The agent receives the typed round-trip of the payload (codex r3), which
    # carries the panel's default keys the caller omitted.
    assert captured["feature_role_panel"] == FeatureRolePanel.from_dict(_PANEL).to_dict()
    # Declared covariates are untouched at submit time: graph_builder is the
    # single place the panel narrows them (with its named warning).
    assert captured["modeled_confounders"] == ["insurance_access_score", "post_dx"]
    # codex r2: the PUBLIC request has no "approved" channel — approval is a
    # server-side boundary (Lane B's loader resolves an approved review by id).
    assert "approved_structure_roles" not in captured
    assert "approved_structure_roles" not in AgentCausalAnalysisRequest.model_fields


@pytest.mark.asyncio
async def test_route_task_omits_the_key_when_the_request_has_no_panel(monkeypatch) -> None:
    import src.api.routes.causal.agent as causal_routes
    from src.api.schemas.causal import AgentCausalAnalysisRequest

    captured = _capture(monkeypatch)
    df = pd.DataFrame({"copay_support": [0.0, 1.0], "adherent_180d": [0.0, 1.0], "x": [0.1, 0.2]})
    req = AgentCausalAnalysisRequest(
        treatment_var="copay_support", outcome_var="adherent_180d", dataset="patient_journeys"
    )
    await causal_routes._run_agent_analysis_task("lane-e-absent", req, df, ["x"], "synthetic")
    assert "feature_role_panel" not in captured
    assert "approved_structure_roles" not in captured


def _submit_request(**panel_overrides):
    from src.api.schemas.causal import AgentCausalAnalysisRequest

    panel = _make_panel("hcp_engagement_level", "patient_conversion_rate")
    panel.update(panel_overrides)
    return AgentCausalAnalysisRequest(
        treatment_var="hcp_engagement_level",
        outcome_var="patient_conversion_rate",
        dataset="patient_journeys",
        feature_role_panel=panel,
    )


async def _expect_400(req, needle: str) -> None:
    from fastapi import BackgroundTasks, HTTPException

    import src.api.routes.causal.agent as causal_routes

    with pytest.raises(HTTPException) as exc:
        await causal_routes.run_causal_agent_analysis(req, BackgroundTasks())
    assert exc.value.status_code == 400
    assert needle in str(exc.value.detail), exc.value.detail


@pytest.mark.asyncio
async def test_submit_refuses_a_panel_built_for_another_question() -> None:
    """codex r2: the panel is a trusted causal input; one built for a different
    (treatment, outcome) must be refused at submit with a 400, not applied."""
    await _expect_400(
        _submit_request(treatment="copay_support", outcome="adherent_180d"), "copay_support"
    )


@pytest.mark.asyncio
async def test_submit_refuses_a_malformed_panel() -> None:
    """The payload must parse as a typed FeatureRolePanel, not any dict."""
    await _expect_400(_submit_request(records="not-a-mapping"), "feature_role_panel")


@pytest.mark.asyncio
async def test_submit_refuses_a_panel_covering_none_of_the_covariates(monkeypatch) -> None:
    """A panel over other columns cannot vet this question's covariates."""
    base = _make_panel("hcp_engagement_level", "patient_conversion_rate")
    rec = dict(base["records"]["insurance_access_score"])
    rec["feature"] = "unrelated_a"
    await _expect_400(
        _submit_request(features=["unrelated_a"], records={"unrelated_a": rec}), "covers none"
    )


@pytest.mark.asyncio
async def test_submit_refuses_a_panel_that_violates_the_strict_invariants() -> None:
    """codex r3: a sparse caller-authored record asserting leak_verdict=true
    without the evidence behind it is refused, not applied."""
    base = _make_panel("hcp_engagement_level", "patient_conversion_rate")
    rec = dict(base["records"]["insurance_access_score"])
    rec["leak_verdict"] = True  # no leak_source, no post-index contract, no Layer-3 high
    await _expect_400(
        _submit_request(records={**base["records"], "insurance_access_score": rec}),
        "leak_source",
    )
    await _expect_400(_submit_request(schema_version="0"), "schema_version")


@pytest.mark.asyncio
async def test_submit_refuses_a_panel_from_another_manifest_when_the_spec_declares_one(
    monkeypatch,
) -> None:
    """When the dataset spec names its manifest (Lane A's real-data entry will),
    a panel built under a different manifest is refused."""
    import src.api.routes.causal.agent as causal_routes

    spec = dict(causal_routes._CAUSAL_DATASET_SPECS["patient_journeys"])
    spec["feature_manifest_source"] = "optum"
    monkeypatch.setitem(causal_routes._CAUSAL_DATASET_SPECS, "patient_journeys", spec)
    await _expect_400(_submit_request(manifest_source="optum_mart"), "optum_mart")


@pytest.mark.asyncio
async def test_route_task_forwards_the_normalised_panel_not_the_raw_dict(monkeypatch) -> None:
    """codex r3: what the agent consumes must be exactly what was validated —
    the typed round-trip of the payload, not the caller's raw dictionary
    (unknown keys dropped, shapes normalised)."""
    import src.api.routes.causal.agent as causal_routes
    from src.api.schemas.causal import AgentCausalAnalysisRequest
    from src.causal_engine.feature_role_panel import FeatureRolePanel

    captured = _capture(monkeypatch)
    raw = {**_PANEL, "features": tuple(_PANEL["features"]), "smuggled": {"anything": 1}}
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
        feature_role_panel=raw,
    )
    await causal_routes._run_agent_analysis_task(
        "lane-e-normalised", req, df, ["insurance_access_score", "post_dx"], "synthetic"
    )
    assert captured["feature_role_panel"] == FeatureRolePanel.from_dict(raw).to_dict()
    assert "smuggled" not in captured["feature_role_panel"]
    assert captured["feature_role_panel"]["features"] == list(_PANEL["features"])


@pytest.mark.asyncio
async def test_submit_refuses_unknown_panel_fields_instead_of_normalising_them_away() -> None:
    """codex r4: reject unknown fields at submit rather than silently dropping them."""
    await _expect_400(_submit_request(smuggled={"anything": 1}), "unknown")
