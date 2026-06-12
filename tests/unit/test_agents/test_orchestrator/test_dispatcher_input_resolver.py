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


# ---------------------------------------------------------------------------
# Issue #880: strict provenance coercion + parameter-channel parity (het)
# ---------------------------------------------------------------------------


_HET_BASE_INPUT: Dict[str, Any] = {
    "query": "which segments respond best to conversion?",
    "session_id": "s1",
    "user_context": {},
    "parsed_query": {"entities": []},
}


def _capture_kpi_resolution(monkeypatch) -> Dict[str, Any]:
    """Monkeypatch the KPI build to a real (no-DB) frame, recording kwargs."""
    kf = _real_kpi_frame()
    seen: Dict[str, Any] = {}

    def _capture(*a, **k):
        seen.update(k)
        return kf

    monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
    monkeypatch.setattr("src.services.kpi_resolution.resolve_kpi_frame", _capture)
    return seen


def test_heterogeneous_resolver_string_false_stays_real_mode(monkeypatch) -> None:
    """Issue #880 RED-FIRST: ``include_synthetic="false"`` (string, e.g. from a
    JSON-ish payload) must NOT opt into the synthetic substrate. ``bool("false")``
    is ``True``, so the loose coercion silently flipped an explicit opt-OUT into
    wrong-provenance reads — the failure class the #850/#851/#872 provenance work
    exists to prevent. Ambiguous values fail CLOSED to real-mode, mirroring the
    gap_analyzer resolver's ``_coerce_provenance_flag`` semantics (#877)."""
    seen = _capture_kpi_resolution(monkeypatch)

    # filters channel (direct resolver invocations, e.g. validation gate 6)
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        {**_HET_BASE_INPUT, "filters": {"brand": "Kisqali", "include_synthetic": "false"}},
        _dispatch("heterogeneous_optimizer"),
    )
    assert isinstance(resolved, dict)
    assert seen.get("include_synthetic") is False

    # user_context channel (the live chat path)
    seen.clear()
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        {**_HET_BASE_INPUT, "user_context": {"include_synthetic": "false"}},
        _dispatch("heterogeneous_optimizer"),
    )
    assert isinstance(resolved, dict)
    assert seen.get("include_synthetic") is False

    # other ambiguous / non-bool values likewise stay real-mode (fail CLOSED)
    for ambiguous in ("0", "no", 1, {"opt": True}):
        seen.clear()
        resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
            {**_HET_BASE_INPUT, "user_context": {"include_synthetic": ambiguous}},
            _dispatch("heterogeneous_optimizer"),
        )
        assert isinstance(resolved, dict)
        assert seen.get("include_synthetic") is False, f"value {ambiguous!r} must NOT opt in"


def test_heterogeneous_resolver_truthy_strings_opt_in(monkeypatch) -> None:
    """Per ``_coerce_provenance_flag`` semantics, "true"/"1"/"yes" (and ``True``)
    DO opt in on either channel."""
    seen = _capture_kpi_resolution(monkeypatch)
    for truthy in ("true", "1", "yes", True):
        seen.clear()
        resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
            {**_HET_BASE_INPUT, "user_context": {"include_synthetic": truthy}},
            _dispatch("heterogeneous_optimizer"),
        )
        assert isinstance(resolved, dict)
        assert seen.get("include_synthetic") is True, f"value {truthy!r} must opt in"


def test_heterogeneous_resolver_parameter_channel_parity(monkeypatch) -> None:
    """Issue #880 RED-FIRST (parity decision): the het resolver gains the gap
    resolver's parameter-level opt-in channels — ``parameters.filters.
    include_synthetic`` and the explicit ``parameters.include_synthetic``, which
    WINS when present and non-None. ``parameters`` is a live chat-path channel
    (_prepare_agent_input threads it; ``filters`` never arrives via chat) and
    gap_analyzer is het's fallback agent, so an explicitly-set parameter flag
    must not yield two different provenance modes for the same dispatch."""
    seen = _capture_kpi_resolution(monkeypatch)

    # parameters.filters channel opts in
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        dict(_HET_BASE_INPUT),
        _dispatch("heterogeneous_optimizer", {"filters": {"include_synthetic": True}}),
    )
    assert isinstance(resolved, dict)
    assert seen.get("include_synthetic") is True

    # explicit parameters.include_synthetic opts in
    seen.clear()
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        dict(_HET_BASE_INPUT),
        _dispatch("heterogeneous_optimizer", {"include_synthetic": True}),
    )
    assert isinstance(resolved, dict)
    assert seen.get("include_synthetic") is True

    # explicit "false" (string) beats an ambient True channel — strict opt-OUT
    seen.clear()
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        {**_HET_BASE_INPUT, "user_context": {"include_synthetic": True}},
        _dispatch("heterogeneous_optimizer", {"include_synthetic": "false"}),
    )
    assert isinstance(resolved, dict)
    assert seen.get("include_synthetic") is False

    # None means "unset": the ambient channel governs
    seen.clear()
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](
        {**_HET_BASE_INPUT, "user_context": {"include_synthetic": True}},
        _dispatch("heterogeneous_optimizer", {"include_synthetic": None}),
    )
    assert isinstance(resolved, dict)
    assert seen.get("include_synthetic") is True


# ---------------------------------------------------------------------------
# Issue #883 §3: the last 3 dead-via-chat ``uses_kwargs`` agents gain resolvers
# (explainer / health_score / feedback_learner). RED on the pre-#883 base:
#   explainer:        "ExplainerAgent.explain() got an unexpected keyword
#                      argument 'user_context'"
#   health_score:     "HealthScoreAgent.check_health() got an unexpected
#                      keyword argument 'user_context'"
#   feedback_learner: "FeedbackLearnerAgent.learn() got an unexpected keyword
#                      argument 'query'"
# ---------------------------------------------------------------------------


def test_registry_contains_the_883_agents() -> None:
    for name in ("explainer", "health_score", "feedback_learner"):
        assert name in disp.INPUT_RESOLVERS, f"{name} missing from INPUT_RESOLVERS"


def test_883_agents_fail_closed_on_failed_status() -> None:
    """All three report ``status='failed'`` ONLY on internal failure (explainer
    graph error / health_score exception path with placeholder 0.0-F values /
    feedback_learner node error — a zero-feedback window completes honestly),
    so a failed run must never be laundered into dispatch success."""
    for name in ("explainer", "health_score", "feedback_learner"):
        assert name in disp._FAIL_CLOSED_ON_FAILED_STATUS, name


def test_coerce_provenance_flag_is_the_shared_ssot() -> None:
    """#883 §4: the dispatcher's strict parser IS the shared repository helper
    (one contract; non-orchestrator boundaries import the same function)."""
    from src.repositories.provenance import coerce_provenance_flag

    assert disp._coerce_provenance_flag is coerce_provenance_flag


def test_prepare_agent_input_threads_agent_results() -> None:
    """The generic payload carries the state's upstream agent results so the
    explainer resolver can bind REAL analysis_results (#883 §3)."""
    node = DispatcherNode()
    upstream = [
        {"agent_name": "causal_impact", "success": True, "result": {"ate": 0.1}, "error": None}
    ]
    prepared = node._prepare_agent_input(
        {"query": "q", "agent_results": upstream},  # type: ignore[arg-type]
        _dispatch("explainer"),
    )
    assert prepared["agent_results"] == upstream


# --------------------------- explainer (#883 §3) ---------------------------


@pytest.mark.asyncio
async def test_explainer_bare_chat_fails_closed_clearly() -> None:
    """Bare 'explain' chat with NO upstream results: a clear, structured
    fail-closed error — never the raw generic-payload TypeError."""
    from src.agents.explainer import ExplainerAgent

    agent = ExplainerAgent(use_llm=False)
    node = DispatcherNode(agent_registry={"explainer": agent})
    out = await node.execute(_state("explainer", "explain the analysis"))

    res = out["agent_results"][0]
    assert res["success"] is False
    err = res["error"] or ""
    assert "analysis_results" in err
    assert "fabricat" in err.lower()
    assert "unexpected keyword argument" not in err, f"raw TypeError leak: {err}"


def test_explainer_resolver_binds_successful_upstream_results() -> None:
    """The resolver binds analysis_results from the REAL upstream AgentResults
    threaded through the payload: successes bound (with real agent attribution),
    failures and the explainer's own prior output excluded."""
    agent_input = {
        "query": "explain it",
        "session_id": "s1",
        "user_context": {},
        "parsed_query": {"entities": []},
        "agent_results": [
            {
                "agent_name": "causal_impact",
                "success": True,
                "result": {"ate": 0.12, "status": "completed"},
                "error": None,
            },
            {"agent_name": "gap_analyzer", "success": False, "result": None, "error": "boom"},
            {
                "agent_name": "explainer",
                "success": True,
                "result": {"executive_summary": "old explanation"},
                "error": None,
            },
        ],
    }
    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch("explainer"))
    assert isinstance(resolved, dict)
    assert resolved["analysis_results"] == [
        {"ate": 0.12, "status": "completed", "agent_name": "causal_impact"}
    ]
    assert resolved["query"] == "explain it"
    assert resolved["session_id"] == "s1"
    for leaked in ("user_context", "parsed_query", "span_id", "dispatch_id", "execution_mode"):
        assert leaked not in resolved, f"{leaked} would leak into explain()"


def test_explainer_resolver_explicit_params_win() -> None:
    explicit = [{"finding": "real analyst-supplied result"}]
    resolved = disp.INPUT_RESOLVERS["explainer"](
        {"query": "explain", "session_id": "s1", "agent_results": []},
        _dispatch("explainer", {"analysis_results": explicit, "user_expertise": "executive"}),
    )
    assert isinstance(resolved, dict)
    assert resolved["analysis_results"] == explicit
    assert resolved["user_expertise"] == "executive"


def test_explainer_resolver_memory_config_from_real_entities() -> None:
    """Brand/region come from the REAL parsed entities (skill loading), never
    fabricated when absent."""
    resolved = disp.INPUT_RESOLVERS["explainer"](
        {
            "query": "explain",
            "agent_results": [
                {"agent_name": "causal_impact", "success": True, "result": {"ate": 0.1}}
            ],
            "parsed_query": {
                "entities": [{"type": "brand", "value": "Kisqali", "confidence": 0.9}]
            },
        },
        _dispatch("explainer"),
    )
    assert isinstance(resolved, dict)
    assert resolved["memory_config"] == {"brand": "Kisqali"}

    resolved_no_brand = disp.INPUT_RESOLVERS["explainer"](
        {
            "query": "explain",
            "agent_results": [
                {"agent_name": "causal_impact", "success": True, "result": {"ate": 0.1}}
            ],
        },
        _dispatch("explainer"),
    )
    assert isinstance(resolved_no_brand, dict)
    assert "memory_config" not in resolved_no_brand


@pytest.mark.asyncio
async def test_explainer_fallback_binds_sibling_group_success() -> None:
    """The universal-fallback path: a failing primary with fallback_agent=
    'explainer' must hand the explainer the SUCCESSFUL results accumulated this
    turn (earlier parallel group), not crash on the generic payload. This is
    the router's standard fallback wiring (causal_effect/multi_faceted/...)."""
    from src.agents.explainer import ExplainerAgent

    class _GoodCausal:
        async def run(self, input_data):
            return {"status": "completed", "ate": 0.42, "narrative": "real upstream finding"}

    class _FailingComposer:
        async def run(self, input_data):
            raise RuntimeError("primary agent crashed")

    node = DispatcherNode(
        agent_registry={
            "causal_impact": _GoodCausal(),
            "tool_composer": _FailingComposer(),
            "explainer": ExplainerAgent(use_llm=False),
        }
    )
    state = {
        "query": "what drove conversions, and compose the follow-up?",
        "user_context": {},
        "session_id": "sess-fb",
        "parsed_query": {"entities": []},
        "dispatch_plan": [
            {
                "agent_name": "causal_impact",
                "priority": "critical",
                "parameters": {},
                "timeout_ms": 15000,
                "fallback_agent": None,
                "execution_mode": "parallel",
            },
            {
                "agent_name": "tool_composer",
                "priority": "high",
                "parameters": {},
                "timeout_ms": 15000,
                "fallback_agent": "explainer",
                "execution_mode": "parallel",
            },
        ],
        "parallel_groups": [["causal_impact"], ["tool_composer"]],
    }
    out = await node.execute(state)  # type: ignore[arg-type]

    by_agent = {r["agent_name"]: r for r in out["agent_results"]}
    assert by_agent["causal_impact"]["success"] is True
    assert by_agent["tool_composer"]["success"] is False
    explainer_res = by_agent["explainer"]
    assert explainer_res["success"] is True, explainer_res["error"]
    # The explanation is grounded in the REAL upstream result, not fabricated.
    assert explainer_res["result"] is not None
    assert explainer_res["result"].get("status") == "completed"


@pytest.mark.asyncio
async def test_explainer_fallback_with_no_upstream_success_fails_closed() -> None:
    """When NOTHING succeeded this turn, the fallback explainer fails closed
    (nothing real to explain) instead of fabricating an explanation."""
    from src.agents.explainer import ExplainerAgent

    class _FailingComposer:
        async def run(self, input_data):
            raise RuntimeError("primary agent crashed")

    node = DispatcherNode(
        agent_registry={
            "tool_composer": _FailingComposer(),
            "explainer": ExplainerAgent(use_llm=False),
        }
    )
    state = _state("tool_composer", "compose something")
    state["dispatch_plan"][0]["fallback_agent"] = "explainer"
    out = await node.execute(state)

    by_agent = {r["agent_name"]: r for r in out["agent_results"]}
    explainer_res = by_agent["explainer"]
    assert explainer_res["success"] is False
    assert "analysis_results" in (explainer_res["error"] or "")
    assert "unexpected keyword argument" not in (explainer_res["error"] or "")


# -------------------------- health_score (#883 §3) --------------------------


def test_health_score_resolver_clean_kwargs_and_session_id() -> None:
    """The resolver maps the generic payload onto check_health's exact kwarg
    set; session_id threads through for the #881 memory wiring."""
    resolved = disp.INPUT_RESOLVERS["health_score"](
        {
            "query": "how healthy is the system?",
            "session_id": "sess-hs",
            "user_context": {"user_id": "u1"},
            "parsed_query": {"entities": []},
            "span_id": "span-x",
            "dispatch_id": "disp-x",
            "execution_mode": "parallel",
            "agent_results": [],
        },
        _dispatch("health_score"),
    )
    assert isinstance(resolved, dict)
    assert resolved["scope"] == "full"
    assert resolved["query"] == "how healthy is the system?"
    assert resolved["session_id"] == "sess-hs"
    assert set(resolved) <= {"scope", "query", "experiment_name", "session_id"}


@pytest.mark.parametrize(
    ("query", "expected_scope"),
    [
        ("how healthy is the system?", "full"),
        ("are the models healthy?", "models"),
        ("check the pipeline status", "pipelines"),
        ("are all agents up?", "agents"),
        ("give me a quick health check", "quick"),
        # Multiple subsystem mentions -> the FULL check covers them all.
        ("check models and pipelines", "full"),
    ],
)
def test_health_score_scope_derivation(query: str, expected_scope: str) -> None:
    resolved = disp.INPUT_RESOLVERS["health_score"]({"query": query}, _dispatch("health_score"))
    assert isinstance(resolved, dict)
    assert resolved["scope"] == expected_scope


def test_health_score_params_scope_wins_when_valid() -> None:
    resolved = disp.INPUT_RESOLVERS["health_score"](
        {"query": "are the models healthy?"},
        _dispatch("health_score", {"scope": "quick", "experiment_name": "ops"}),
    )
    assert isinstance(resolved, dict)
    assert resolved["scope"] == "quick"
    assert resolved["experiment_name"] == "ops"

    # An invalid scope param falls back to derivation, never reaches the agent.
    resolved_bad = disp.INPUT_RESOLVERS["health_score"](
        {"query": "are the models healthy?"},
        _dispatch("health_score", {"scope": "everything"}),
    )
    assert isinstance(resolved_bad, dict)
    assert resolved_bad["scope"] == "models"


@pytest.mark.asyncio
async def test_health_score_dispatch_completes_for_real() -> None:
    """E2E through the dispatcher: the 'system_health' intent (sole agent, no
    fallback) actually RUNS now. RED on base: TypeError 'unexpected keyword
    argument user_context' made the intent 100% dead via chat."""
    from src.agents.health_score import HealthScoreAgent

    agent = HealthScoreAgent(enable_mlflow=False, enable_opik=False, enable_memory=False)
    node = DispatcherNode(agent_registry={"health_score": agent})
    out = await node.execute(_state("health_score", "how healthy is the system?"))

    res = out["agent_results"][0]
    assert res["success"] is True, res["error"]
    assert res["result"] is not None
    assert res["result"].get("status") != "failed"
    assert "health_summary" in res["result"]


# ------------------------ feedback_learner (#883 §3) ------------------------


def test_feedback_learner_default_window_mirrors_celery_beat(monkeypatch) -> None:
    """No temporal entity -> the trailing window the 6h Celery beat learns on
    (DSPY_LEARN_WINDOW_HOURS, default 24h, ending now UTC)."""
    from datetime import datetime, timedelta, timezone

    monkeypatch.delenv("DSPY_LEARN_WINDOW_HOURS", raising=False)
    resolved = disp.INPUT_RESOLVERS["feedback_learner"](
        {"query": "what have we learned from feedback?", "parsed_query": {"entities": []}},
        _dispatch("feedback_learner"),
    )
    assert isinstance(resolved, dict)
    start = datetime.fromisoformat(resolved["time_range_start"])
    end = datetime.fromisoformat(resolved["time_range_end"])
    assert abs((end - start) - timedelta(hours=24)) < timedelta(seconds=5)
    assert abs(end - datetime.now(timezone.utc)) < timedelta(minutes=1)
    # learn() accepts NO query/session/context kwargs — the clean set only.
    assert set(resolved) <= {"time_range_start", "time_range_end", "batch_id", "focus_agents"}

    # The env var governs, exactly like the beat path.
    monkeypatch.setenv("DSPY_LEARN_WINDOW_HOURS", "6")
    resolved6 = disp.INPUT_RESOLVERS["feedback_learner"](
        {"query": "feedback?", "parsed_query": {"entities": []}}, _dispatch("feedback_learner")
    )
    assert isinstance(resolved6, dict)
    start6 = datetime.fromisoformat(resolved6["time_range_start"])
    end6 = datetime.fromisoformat(resolved6["time_range_end"])
    assert abs((end6 - start6) - timedelta(hours=6)) < timedelta(seconds=5)


def test_feedback_learner_temporal_entities_bind_named_windows() -> None:
    from datetime import datetime, timezone

    def _resolve(entities):
        return disp.INPUT_RESOLVERS["feedback_learner"](
            {"query": "q", "parsed_query": {"entities": entities}},
            _dispatch("feedback_learner"),
        )

    # Quarter + year (split entities, the classifier's Q[1-4] / 20xx shapes).
    resolved = _resolve(
        [
            {"type": "time_period", "value": "Q3", "confidence": 0.9},
            {"type": "time_period", "value": "2025", "confidence": 0.9},
        ]
    )
    assert isinstance(resolved, dict)
    assert resolved["time_range_start"] == datetime(2025, 7, 1, tzinfo=timezone.utc).isoformat()
    assert resolved["time_range_end"] == datetime(2025, 10, 1, tzinfo=timezone.utc).isoformat()

    # Bare year -> calendar year.
    resolved_y = _resolve([{"type": "time_period", "value": "2025"}])
    assert isinstance(resolved_y, dict)
    assert resolved_y["time_range_start"] == datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat()
    assert resolved_y["time_range_end"] == datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()

    # Relative trailing phrase.
    resolved_r = _resolve([{"type": "time_period", "value": "last 7 days"}])
    assert isinstance(resolved_r, dict)
    start = datetime.fromisoformat(resolved_r["time_range_start"])
    end = datetime.fromisoformat(resolved_r["time_range_end"])
    assert (end - start).days == 7


def test_feedback_learner_unparseable_named_period_fails_closed() -> None:
    """A period the user NAMED but we cannot parse must fail closed — silently
    learning over a substituted window would misrepresent the result."""
    resolved = disp.INPUT_RESOLVERS["feedback_learner"](
        {
            "query": "q",
            "parsed_query": {
                "entities": [{"type": "time_period", "value": "the fortnight of yore"}]
            },
        },
        _dispatch("feedback_learner"),
    )
    assert isinstance(resolved, NeedsStructuredInput)
    assert "time_range_start" in resolved.missing
    assert "fortnight of yore" in resolved.reason


def test_feedback_learner_explicit_params_validated() -> None:
    # A real explicit window passes through verbatim (+ passthrough options).
    resolved = disp.INPUT_RESOLVERS["feedback_learner"](
        {"query": "q"},
        _dispatch(
            "feedback_learner",
            {
                "time_range_start": "2026-06-01T00:00:00+00:00",
                "time_range_end": "2026-06-08T00:00:00+00:00",
                "batch_id": "batch_x",
                "focus_agents": ["causal_impact"],
            },
        ),
    )
    assert isinstance(resolved, dict)
    assert resolved["time_range_start"] == "2026-06-01T00:00:00+00:00"
    assert resolved["time_range_end"] == "2026-06-08T00:00:00+00:00"
    assert resolved["batch_id"] == "batch_x"
    assert resolved["focus_agents"] == ["causal_impact"]

    # Garbled/half windows are ungroundable -> fail closed, never repaired.
    for bad in (
        {"time_range_start": "not-a-date", "time_range_end": "2026-06-08T00:00:00+00:00"},
        {"time_range_start": "2026-06-08T00:00:00+00:00"},  # half a window
        {
            "time_range_start": "2026-06-08T00:00:00+00:00",
            "time_range_end": "2026-06-01T00:00:00+00:00",  # start >= end
        },
    ):
        out = disp.INPUT_RESOLVERS["feedback_learner"](
            {"query": "q"}, _dispatch("feedback_learner", bad)
        )
        assert isinstance(out, NeedsStructuredInput), bad
        assert "/feedback/learn" in (out.rest_endpoint or "")


@pytest.mark.asyncio
async def test_feedback_learner_dispatch_completes_honestly() -> None:
    """E2E through the dispatcher: the 'feedback' intent (sole agent, no
    fallback) runs the REAL agent over the default window. With no stores wired
    the honest outcome is zero feedback items — completed with a warning, never
    fabricated learnings. RED on base: TypeError 'unexpected keyword argument
    query'."""
    from src.agents.feedback_learner import FeedbackLearnerAgent

    agent = FeedbackLearnerAgent()
    node = DispatcherNode(agent_registry={"feedback_learner": agent})
    out = await node.execute(_state("feedback_learner", "what have we learned from feedback?"))

    res = out["agent_results"][0]
    assert res["success"] is True, res["error"]
    assert res["result"] is not None
    assert res["result"].get("status") != "failed"
    assert res["result"].get("feedback_count") == 0  # honest no-data outcome
