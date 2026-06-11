"""Generic dynamic input-resolver registry for the dispatcher (audit F12/F13/F14).

Three Tier-2/4 agents were dispatch-dead: the orchestrator splatted/passed the
*generic* payload (``query``/``user_context``/``parsed_query``/``span_id``/...)
into a method that rejects it, so the agent crashed with a raw ``TypeError`` /
``ValueError`` and was unreachable via chat.

The fix is a single, generic ``INPUT_RESOLVERS`` registry in the dispatcher (no
per-agent ``if`` branches). Each resolver, given the prepared payload + dispatch,
returns EITHER:

* a dict of **real, data-grounded** inputs to apply, OR
* a :class:`NeedsStructuredInput` — a CLEAR, actionable fail-closed signal when
  the agent's required inputs could not be honestly grounded in real data.

Nothing is ever fabricated. The data decides per-agent:

* **heterogeneous_optimizer** — its causal spec (treatment/outcome/modifiers) is
  built from the REAL KPI substrate (``KpiFrame``: ``treatment_column`` /
  ``outcome_column`` / ``driver_columns`` over a real frame). No KPI substrate
  (or too few real rows) → fail closed.
* **resource_optimizer** / **prediction_synthesizer** — their inputs (a fully
  specified optimization problem / a specific entity + trained model) have NO
  data substrate today, so they fail closed with a precise reason and pass
  through real structured ``dispatch.parameters`` when an API caller supplies
  them.

These tests are faithful: real ``DispatcherNode`` + real agent instances. The
resolver short-circuits BEFORE the agent method, so the fail-closed assertions
do not run any heavy pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes.dispatcher import DispatcherNode, NeedsStructuredInput


def _state(agent_name: str, query: str, *, entities=None, user_context=None) -> Dict[str, Any]:
    return {
        "query": query,
        "user_context": user_context if user_context is not None else {"user_id": "u1"},
        "session_id": "sess-1",
        "parsed_query": {"intent": agent_name, "entities": entities or []},
        "dispatch_plan": [
            {
                "agent_name": agent_name,
                "priority": "high",
                "parameters": {},
                "timeout_ms": 15000,
                "fallback_agent": None,
                "execution_mode": "parallel",
            }
        ],
        "parallel_groups": [[agent_name]],
    }


def _dispatch(agent_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "agent_name": agent_name,
        "priority": "high",
        "parameters": params or {},
        "timeout_ms": 15000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_registry_contains_all_four_agents() -> None:
    """The generic registry serves tool_composer + the 3 newly-bridged agents."""
    for name in (
        "tool_composer",
        "heterogeneous_optimizer",
        "resource_optimizer",
        "prediction_synthesizer",
    ):
        assert name in disp.INPUT_RESOLVERS, f"{name} missing from INPUT_RESOLVERS"


def test_needs_structured_input_message_is_actionable() -> None:
    nsi = NeedsStructuredInput(
        agent_name="resource_optimizer",
        missing=("allocation_targets", "constraints"),
        reason="no real allocation substrate exists in the data",
        rest_endpoint="POST /resources/optimize",
    )
    msg = nsi.to_error()
    assert "resource_optimizer" in msg
    assert "allocation_targets" in msg and "constraints" in msg
    assert "/resources/optimize" in msg
    # It must be explicit that nothing was fabricated.
    assert "fabricat" in msg.lower()


# ---------------------------------------------------------------------------
# resource_optimizer (F13) — no substrate today → fail closed; params pass through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resource_optimizer_bare_chat_fails_closed_clearly() -> None:
    """Bare chat (no structured params, no data substrate) → CLEAR structured
    error naming the missing inputs; NEVER the old 'unexpected keyword
    user_context' leak and NEVER a fabricated allocation."""
    from src.agents.resource_optimizer import ResourceOptimizerAgent

    agent = ResourceOptimizerAgent()
    node = DispatcherNode(agent_registry={"resource_optimizer": agent})
    out = await node.execute(_state("resource_optimizer", "how should we allocate budget?"))

    res = out["agent_results"][0]
    assert res["success"] is False
    err = (res["error"] or "").lower()
    assert "allocation_targets" in err
    assert "user_context" not in err, f"generic-payload leak must be gone: {err}"
    assert "fabricat" in err


@pytest.mark.asyncio
async def test_resource_optimizer_passthrough_real_structured_params() -> None:
    """An API/router caller that supplies a REAL allocation problem in
    dispatch.parameters reaches ``optimize`` with a CLEAN kwarg set (no leak),
    and the agent runs the real solver."""
    params = {
        "allocation_targets": [
            {
                "entity_id": "territory_ne",
                "entity_type": "territory",
                "current_allocation": 50000.0,
                "expected_response": 1.3,
            },
            {
                "entity_id": "territory_sw",
                "entity_type": "territory",
                "current_allocation": 30000.0,
                "expected_response": 0.8,
            },
        ],
        "constraints": [{"constraint_type": "budget", "value": 90000.0}],
        "resource_type": "budget",
        "objective": "maximize_outcome",
    }
    resolved = disp.INPUT_RESOLVERS["resource_optimizer"](
        {"query": "optimize", "session_id": "s1", "user_context": {}, "parsed_query": {}},
        _dispatch("resource_optimizer", params),
    )
    assert isinstance(resolved, dict)
    # Real structured inputs preserved.
    assert resolved["allocation_targets"] == params["allocation_targets"]
    assert resolved["constraints"] == params["constraints"]
    assert resolved["query"] == "optimize"
    # The leak keys must NOT be in the kwarg set handed to optimize(**kwargs).
    for leaked in ("user_context", "parsed_query", "span_id", "dispatch_id", "execution_mode"):
        assert leaked not in resolved, f"{leaked} would leak into optimize(): {resolved.keys()}"


# ---------------------------------------------------------------------------
# prediction_synthesizer (F14) — no model substrate today → fail closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prediction_synthesizer_bare_chat_fails_closed_clearly() -> None:
    from src.agents.prediction_synthesizer import PredictionSynthesizerAgent

    agent = PredictionSynthesizerAgent()
    node = DispatcherNode(agent_registry={"prediction_synthesizer": agent})
    out = await node.execute(_state("prediction_synthesizer", "what's the forecast?"))

    res = out["agent_results"][0]
    assert res["success"] is False
    err = (res["error"] or "").lower()
    assert "entity_id" in err
    assert "user_context" not in err, f"leak must be gone: {err}"
    assert "fabricat" in err


def test_prediction_synthesizer_passthrough_real_structured_params() -> None:
    params = {"entity_id": "HCP-993", "prediction_target": "conversion", "entity_type": "hcp"}
    resolved = disp.INPUT_RESOLVERS["prediction_synthesizer"](
        {"query": "predict", "session_id": "s1", "user_context": {}, "parsed_query": {}},
        _dispatch("prediction_synthesizer", params),
    )
    assert isinstance(resolved, dict)
    assert resolved["entity_id"] == "HCP-993"
    assert resolved["prediction_target"] == "conversion"
    for leaked in ("user_context", "parsed_query", "span_id", "dispatch_id", "execution_mode"):
        assert leaked not in resolved


# ---------------------------------------------------------------------------
# heterogeneous_optimizer (F12) — BUILD substrate from real KPI frame
# ---------------------------------------------------------------------------


def _real_kpi_frame(n: int = 120):
    """A real ``KpiFrame`` (no DB) carrying a real treatment + outcome + drivers."""
    from src.services.kpi_resolution import KpiFrame

    frame = pd.DataFrame(
        {
            "accepted": [i % 2 for i in range(n)],
            "converted": [(i % 3 == 0) for i in range(n)],
            "specialty": ["onc" if i % 2 else "card" for i in range(n)],
            "decile": [i % 10 for i in range(n)],
        }
    )
    return KpiFrame(
        frame=frame,
        outcome_column="converted",
        driver_columns=["accepted", "specialty", "decile"],
        treatment_column="accepted",
        kpi_id="WS3-BI-009",
        kpi_name="Conversion Rate",
    )


def test_heterogeneous_resolver_builds_inputs_from_kpi_substrate(monkeypatch) -> None:
    """A KPI-recognized query → the resolver binds treatment/outcome/modifiers to
    the REAL columns of the resolved frame and threads the frame via tier0_data.
    Every value is a real column name from the KPI definition; nothing invented."""
    kf = _real_kpi_frame()
    monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
    monkeypatch.setattr("src.services.kpi_resolution.resolve_kpi_frame", lambda *a, **k: kf)

    agent_input = {
        "query": "which segments respond best to conversion?",
        "session_id": "s1",
        "user_context": {"brand": "Kisqali", "region": "Northeast"},
        "parsed_query": {"entities": []},
    }
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        agent_input, _dispatch("heterogeneous_optimizer")
    )
    assert isinstance(resolved, dict)
    assert resolved["outcome_var"] == "converted"
    assert resolved["treatment_var"] == "accepted"
    # effect_modifiers = real driver columns minus the treatment; non-empty.
    assert "accepted" not in resolved["effect_modifiers"]
    assert set(resolved["effect_modifiers"]) == {"specialty", "decile"}
    assert resolved["effect_modifiers"]  # het_opt requires non-empty
    # The REAL frame is threaded via tier0_data (the het_opt passthrough channel).
    assert isinstance(resolved["tier0_data"], pd.DataFrame)
    assert len(resolved["tier0_data"]) == 120
    assert "kpi_substrate" in resolved["data_source"]


def test_heterogeneous_resolver_excludes_treatment_source_from_modifiers(monkeypatch) -> None:
    """LEAKAGE GUARD: the raw column the treatment was derived from (e.g.
    ``acceptance_status`` → ``accepted``) must NEVER appear in effect_modifiers —
    it is a deterministic function of the treatment."""
    from src.services.kpi_resolution import KpiFrame

    n = 120
    frame = pd.DataFrame(
        {
            "accepted": [i % 2 for i in range(n)],
            "acceptance_status": ["accepted" if i % 2 else "rejected" for i in range(n)],
            "converted": [(i % 3 == 0) for i in range(n)],
            "trigger_type": ["email" if i % 2 else "call" for i in range(n)],
        }
    )
    kf = KpiFrame(
        frame=frame,
        outcome_column="converted",
        driver_columns=["accepted", "acceptance_status", "trigger_type"],
        treatment_column="accepted",
        treatment_source_column="acceptance_status",
        kpi_id="WS3-BI-009",
        kpi_name="Conversion Rate",
    )
    monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
    monkeypatch.setattr("src.services.kpi_resolution.resolve_kpi_frame", lambda *a, **k: kf)

    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        {
            "query": "conversion segments",
            "session_id": "s1",
            "user_context": {},
            "parsed_query": {},
        },
        _dispatch("heterogeneous_optimizer"),
    )
    assert isinstance(resolved, dict)
    assert resolved["treatment_var"] == "accepted"
    assert "acceptance_status" not in resolved["effect_modifiers"], "treatment-source leakage!"
    assert "accepted" not in resolved["effect_modifiers"]
    assert resolved["effect_modifiers"] == ["trigger_type"]


def test_heterogeneous_resolver_no_kpi_fails_closed(monkeypatch) -> None:
    """No recognized KPI → fail closed (NeedsStructuredInput), not a fabricated
    causal spec."""
    monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: None)

    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        {"query": "do some analysis", "session_id": "s1", "user_context": {}, "parsed_query": {}},
        _dispatch("heterogeneous_optimizer"),
    )
    assert isinstance(resolved, NeedsStructuredInput)
    assert "treatment_var" in resolved.missing
    assert resolved.rest_endpoint


def test_heterogeneous_resolver_too_few_rows_fails_closed(monkeypatch) -> None:
    """A KPI frame below het_opt's tier0 row floor (100) → fail closed rather than
    feed an underpowered frame the agent would silently drop for mock data."""
    kf = _real_kpi_frame(n=20)
    monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
    monkeypatch.setattr("src.services.kpi_resolution.resolve_kpi_frame", lambda *a, **k: kf)

    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        {
            "query": "conversion segments",
            "session_id": "s1",
            "user_context": {},
            "parsed_query": {},
        },
        _dispatch("heterogeneous_optimizer"),
    )
    assert isinstance(resolved, NeedsStructuredInput)


def test_heterogeneous_resolver_explicit_params_win(monkeypatch) -> None:
    """An analyst-supplied causal spec in dispatch.parameters is passed through
    verbatim (explicit choice beats substrate-building)."""
    # recognize_kpi must NOT be consulted when params already specify the spec.
    monkeypatch.setattr(
        "src.services.kpi_resolution.recognize_kpi",
        lambda _q: (_ for _ in ()).throw(AssertionError("should not build substrate")),
    )
    params = {
        "treatment_var": "rep_visits",
        "outcome_var": "trx",
        "effect_modifiers": ["specialty", "decile"],
        "segment_vars": ["region"],
        "data_source": "hcp_metrics",
    }
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        {"query": "het", "session_id": "s1", "user_context": {}, "parsed_query": {}},
        _dispatch("heterogeneous_optimizer", params),
    )
    assert isinstance(resolved, dict)
    assert resolved["treatment_var"] == "rep_visits"
    assert resolved["outcome_var"] == "trx"
    assert resolved["effect_modifiers"] == ["specialty", "decile"]


# ---------------------------------------------------------------------------
# No-fabrication regression across all three bare-chat dispatches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_agent_fabricates_on_bare_chat() -> None:
    """Parametrized guard: a bare chat dispatch to any of the three agents must
    NEVER return success with fabricated analytics — it fails closed."""
    from src.agents.heterogeneous_optimizer import HeterogeneousOptimizerAgent
    from src.agents.prediction_synthesizer import PredictionSynthesizerAgent
    from src.agents.resource_optimizer import ResourceOptimizerAgent

    agents = {
        "heterogeneous_optimizer": HeterogeneousOptimizerAgent(),
        "resource_optimizer": ResourceOptimizerAgent(),
        "prediction_synthesizer": PredictionSynthesizerAgent(),
    }
    for name, agent in agents.items():
        node = DispatcherNode(agent_registry={name: agent})
        out = await node.execute(_state(name, "tell me something useful"))
        res = out["agent_results"][0]
        assert res["success"] is False, f"{name} should fail closed on bare chat, got success"
        assert "fabricat" in (res["error"] or "").lower()


# ---------------------------------------------------------------------------
# resource_optimizer: a budget constraint is required (codex HIGH)
# ---------------------------------------------------------------------------


def test_resource_optimizer_targets_without_budget_fails_closed() -> None:
    """Targets supplied but NO budget constraint → fail closed (the optimizer's
    problem_formulator requires a budget; passing an under-specified problem would
    otherwise be laundered into a 'successful' but internally-failed dispatch)."""
    params = {
        "allocation_targets": [
            {
                "entity_id": "t1",
                "entity_type": "territory",
                "current_allocation": 1000.0,
                "expected_response": 1.0,
            }
        ],
        "constraints": [{"constraint_type": "capacity", "value": 5.0}],  # no budget
    }
    resolved = disp.INPUT_RESOLVERS["resource_optimizer"](
        {"query": "optimize", "session_id": "s1"},
        _dispatch("resource_optimizer", params),
    )
    assert isinstance(resolved, NeedsStructuredInput)
    assert any("budget" in m for m in resolved.missing)


# ---------------------------------------------------------------------------
# Domain-failure guard: an agent that runs but reports status=failed must NOT be
# laundered into a successful dispatch (codex HIGH).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_domain_failure_guard_fails_closed_on_failed_status() -> None:
    """A resolver-backed agent whose method returns ``status='failed'`` (e.g.
    prediction_synthesizer with no registered models) must yield success=False."""

    async def fake_synthesize(**kwargs):  # noqa: ANN003
        return {
            "status": "failed",
            "errors": [{"node": "orchestrator", "error": "No models available"}],
        }

    agent = MagicMock()
    agent.synthesize = fake_synthesize
    del agent.analyze

    node = DispatcherNode(agent_registry={"prediction_synthesizer": agent})
    out = await node.execute(
        _state(
            "prediction_synthesizer",
            "predict",
        )
        | {
            "dispatch_plan": [
                {
                    "agent_name": "prediction_synthesizer",
                    "priority": "high",
                    "parameters": {"entity_id": "HCP-1", "prediction_target": "conversion"},
                    "timeout_ms": 15000,
                    "fallback_agent": None,
                    "execution_mode": "parallel",
                }
            ],
        }
    )
    res = out["agent_results"][0]
    assert res["success"] is False
    err = (res["error"] or "").lower()
    assert "no models available" in err
    assert "fabricat" in err


@pytest.mark.asyncio
async def test_domain_failure_guard_allows_completed_status() -> None:
    """The guard must NOT over-fire: a ``status='completed'`` result still succeeds."""

    async def fake_synthesize(**kwargs):  # noqa: ANN003
        return {"status": "completed", "prediction_summary": "real result"}

    agent = MagicMock()
    agent.synthesize = fake_synthesize
    del agent.analyze

    node = DispatcherNode(agent_registry={"prediction_synthesizer": agent})
    out = await node.execute(
        _state("prediction_synthesizer", "predict")
        | {
            "dispatch_plan": [
                {
                    "agent_name": "prediction_synthesizer",
                    "priority": "high",
                    "parameters": {"entity_id": "HCP-1", "prediction_target": "conversion"},
                    "timeout_ms": 15000,
                    "fallback_agent": None,
                    "execution_mode": "parallel",
                }
            ],
        }
    )
    res = out["agent_results"][0]
    assert res["success"] is True, res.get("error")
    assert res["result"]["prediction_summary"] == "real result"


def test_heterogeneous_resolver_forwards_include_synthetic_opt_in(monkeypatch) -> None:
    """Validation runs opt into the synthetic substrate via filters.include_synthetic
    (the #851 plumb pattern): the resolver must forward it to resolve_kpi_frame.
    Without the flag the resolver stays real-mode (include_synthetic falsy) — on a
    clean substrate (zero untagged legacy rows) real-mode correctly fails closed,
    which is exactly why the opt-in must be explicit and default-off."""
    kf = _real_kpi_frame()
    seen: dict = {}

    def _capture(*a, **k):
        seen.update(k)
        return kf

    monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
    monkeypatch.setattr("src.services.kpi_resolution.resolve_kpi_frame", _capture)

    base_input = {
        "query": "which segments respond best to conversion?",
        "session_id": "s1",
        "user_context": {},
        "parsed_query": {"entities": []},
    }

    # opt-in via filters (direct resolver invocations): forwarded as True
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        {**base_input, "filters": {"brand": "Kisqali", "include_synthetic": True}},
        _dispatch("heterogeneous_optimizer"),
    )
    assert isinstance(resolved, dict)
    assert seen.get("include_synthetic") is True

    # opt-in via user_context (the only caller-stash field the live chat path's
    # _prepare_agent_input threads through — the state schema carries no filters)
    seen.clear()
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        {**base_input, "user_context": {"include_synthetic": True}},
        _dispatch("heterogeneous_optimizer"),
    )
    assert isinstance(resolved, dict)
    assert seen.get("include_synthetic") is True

    # default: real-mode (falsy)
    seen.clear()
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        {**base_input, "filters": {"brand": "Kisqali"}},
        _dispatch("heterogeneous_optimizer"),
    )
    assert isinstance(resolved, dict)
    assert not seen.get("include_synthetic")
