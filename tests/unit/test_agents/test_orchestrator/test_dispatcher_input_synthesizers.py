"""Shard 08 T1/T2 — the dispatcher synthesizes each input-model-less agent's
required keys from the NL query + resolved brand/region so the 4 Tier-4-ish
agents (``gap_analyzer``, ``heterogeneous_optimizer``, ``resource_optimizer``,
``prediction_synthesizer``) become reachable via chat/dispatch instead of
failing closed on missing required fields.

The 4 agents declare NO ``input_model`` in ``AGENT_METHOD_MAP``, so
``_prepare_agent_input``'s generic payload (``query``/``user_context``/
``parameters``/``session_id``/``parsed_query``/``dispatch_id``/``span_id``/
``execution_mode``) never carries their required keys:

- ``gap_analyzer.run`` needs ``metrics``/``segments``/``brand`` (uses_kwargs=False).
- ``heterogeneous_optimizer.run`` needs
  ``treatment_var``/``outcome_var``/``segment_vars``/``effect_modifiers``/
  ``data_source`` (uses_kwargs=False).
- ``resource_optimizer.optimize`` needs ``allocation_targets``/``constraints``
  (uses_kwargs=True — splatted, so generic keys would TypeError).
- ``prediction_synthesizer.synthesize`` needs positional ``entity_id``/
  ``prediction_target`` (uses_kwargs=True).

Mirrors ``test_dispatcher_wrapped_input_coercion.py`` state shape + MagicMock
pattern.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agents.orchestrator.nodes.dispatcher import DispatcherNode


def _state(agent_name: str, query: str) -> dict:
    return {
        "query": query,
        "user_context": {"brand": "Kisqali", "region": "northeast"},
        "session_id": "s1",
        "parsed_query": {"entities": [{"type": "brand", "value": "Kisqali"}]},
        "dispatch_plan": [
            {
                "agent_name": agent_name,
                "priority": "critical",
                "parameters": {},
                "timeout_ms": 30000,
                "fallback_agent": None,
                "execution_mode": "parallel",
            }
        ],
        "parallel_groups": [[agent_name]],
    }


# ---------------------------------------------------------------------------
# T1 — uses_kwargs=False agents (single-dict splat, AUGMENT the generic payload)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gap_analyzer_receives_synthesized_required_keys() -> None:
    captured: dict = {}

    async def fake_run(input_data):  # gap_analyzer uses_kwargs=False -> single dict
        captured.update(input_data)
        return {"executive_summary": "ok", "prioritized_opportunities": []}

    agent = MagicMock()
    agent.run = fake_run
    del agent.analyze
    disp = DispatcherNode(agent_registry={"gap_analyzer": agent})
    res = await disp.execute(_state("gap_analyzer", "where are the gaps for Kisqali by region"))
    assert res["agent_results"][0]["success"] is True, res["agent_results"][0].get("error")
    assert isinstance(captured.get("metrics"), list) and captured["metrics"]
    assert isinstance(captured.get("segments"), list) and captured["segments"]
    assert captured.get("brand") == "Kisqali"
    # generic pass-through key still present for uses_kwargs=False agents
    assert captured.get("query")


@pytest.mark.asyncio
async def test_heterogeneous_optimizer_receives_synthesized_required_keys() -> None:
    captured: dict = {}

    async def fake_run(input_data):
        captured.update(input_data)
        return {"executive_summary": "ok", "status": "completed"}

    agent = MagicMock()
    agent.run = fake_run
    del agent.analyze
    disp = DispatcherNode(agent_registry={"heterogeneous_optimizer": agent})
    res = await disp.execute(
        _state("heterogeneous_optimizer", "which HCP segments respond best for Kisqali")
    )
    assert res["agent_results"][0]["success"] is True, res["agent_results"][0].get("error")
    # CANONICAL patient_journeys columns (INDEX §CANONICAL SSOT), not drift names.
    assert captured.get("treatment_var") == "treatment_arm"
    assert captured.get("outcome_var") == "treatment_initiated"
    assert isinstance(captured.get("segment_vars"), list)
    assert captured.get("effect_modifiers") == ["disease_severity", "age_at_diagnosis"]
    assert captured.get("data_source") is not None
    assert captured.get("query")


# ---------------------------------------------------------------------------
# T2 — uses_kwargs=True agents (splatted -> synthesizer must REPLACE the payload
# so generic keys never reach the method and TypeError)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resource_optimizer_receives_synthesized_kwargs() -> None:
    captured: dict = {}

    async def fake_optimize(**kwargs):  # uses_kwargs=True -> splatted
        captured.update(kwargs)
        return MagicMock(solver_status="optimal")

    agent = MagicMock()
    agent.optimize = fake_optimize
    del agent.analyze
    disp = DispatcherNode(agent_registry={"resource_optimizer": agent})
    res = await disp.execute(_state("resource_optimizer", "reallocate budget across Kisqali HCPs"))
    assert res["agent_results"][0]["success"] is True, res["agent_results"][0].get("error")
    assert isinstance(captured.get("allocation_targets"), list)
    assert isinstance(captured.get("constraints"), list)
    # generic keys must NOT splat (TypeError guard)
    assert "user_context" not in captured
    assert "parsed_query" not in captured
    assert "span_id" not in captured


@pytest.mark.asyncio
async def test_prediction_synthesizer_receives_synthesized_kwargs() -> None:
    captured: dict = {}

    async def fake_synthesize(**kwargs):
        captured.update(kwargs)
        return {"prediction_summary": "ok"}

    agent = MagicMock()
    agent.synthesize = fake_synthesize
    del agent.analyze
    disp = DispatcherNode(agent_registry={"prediction_synthesizer": agent})
    res = await disp.execute(
        _state_with_hcp("prediction_synthesizer", "predict conversion for HCP_123 on Kisqali")
    )
    assert res["agent_results"][0]["success"] is True, res["agent_results"][0].get("error")
    assert captured.get("entity_id")
    assert captured.get("prediction_target")
    assert "span_id" not in captured
    assert "user_context" not in captured


def _state_with_hcp(agent_name: str, query: str) -> dict:
    """State that carries an hcp_id entity so prediction_synthesizer resolves one."""
    state = _state(agent_name, query)
    state["parsed_query"] = {
        "entities": [
            {"type": "brand", "value": "Kisqali"},
            {"type": "hcp_id", "value": "HCP_123"},
        ]
    }
    return state
