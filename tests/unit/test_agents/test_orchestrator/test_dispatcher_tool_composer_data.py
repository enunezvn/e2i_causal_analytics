"""Tests for F2(a): the dispatcher threads a real cohort DataFrame into the
``tool_composer`` agent's input under ``data`` (which ``ToolComposerAgent.run``
normalizes to ``context["estimation_data"]``).

Context
-------
Two live production paths reach the Tool Composer. The CHAT path
(``chatbot_tools.tool_composer_tool``) already resolves a cohort frame for
``(brand, region)`` and threads it as ``estimation_data``. The ORCHESTRATOR
path did not: ``intent_classifier`` -> ``router.multi_faceted`` (dispatch with
``parameters={}``) -> ``DispatcherNode._prepare_agent_input`` threaded only
``query``/``user_context``/``parameters``/``session_id``/``parsed_query`` -- NO
data. ``ToolComposerAgent.run`` reads ``input_data["data"]`` but the dispatcher
never supplied it, so multi_faceted queries via the orchestrator delivered 0
data and the real causal tools all fail-closed.

F12/F13/F14 generalized this single special-case into the
``dispatcher.INPUT_RESOLVERS`` registry — the tool_composer data resolution now
lives in ``_resolve_tool_composer_input`` and is applied in ``_dispatch_agent``
after ``_prepare_agent_input`` builds the generic payload. These tests pin the
SAME wiring (no network): they patch ``resolve_cohort_frame`` (imported lazily
inside the dispatcher helper) and assert on the resolver's output, including the
fail-closed contract for unrecognized brands and resolver exceptions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import pandas as pd

from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes.dispatcher import DispatcherNode, NeedsStructuredInput


def _state_with_entities(
    brand: Optional[str] = None,
    region: Optional[str] = None,
    *,
    user_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return an OrchestratorState-like dict with brand/region parsed entities."""
    entities = []
    if brand is not None:
        entities.append({"type": "brand", "value": brand, "confidence": 0.95, "source": "exact"})
    if region is not None:
        entities.append({"type": "region", "value": region, "confidence": 0.9, "source": "exact"})
    return {
        "query": "What drives adoption and where are the gaps?",
        "user_context": user_context if user_context is not None else {"user_id": "u1"},
        "session_id": "sess-1",
        "parsed_query": {"intent": "causal_impact", "entities": entities},
    }


def _tool_composer_dispatch() -> Dict[str, Any]:
    return {
        "agent_name": "tool_composer",
        "priority": "high",
        "parameters": {},
        "timeout_ms": 90000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


def _other_dispatch(agent_name: str) -> Dict[str, Any]:
    return {
        "agent_name": agent_name,
        "priority": "high",
        "parameters": {},
        "timeout_ms": 30000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


def _resolved_input(
    node: DispatcherNode, state: Dict[str, Any], dispatch: Dict[str, Any]
) -> Union[Dict[str, Any], NeedsStructuredInput]:
    """Mirror ``_dispatch_agent``'s flow for a non-kwargs agent: build the generic
    payload, then apply the agent's INPUT_RESOLVER (if any) and merge.

    Returns the merged input dict (what the agent method would receive) or a
    ``NeedsStructuredInput`` when the resolver fails closed. tool_composer never
    fails closed, so for it this always returns a dict.
    """
    prepared = node._prepare_agent_input(state, dispatch)  # type: ignore[arg-type]
    resolver = disp.INPUT_RESOLVERS.get(dispatch["agent_name"])
    if resolver is None:
        return prepared
    resolved = resolver(prepared, dispatch)  # type: ignore[arg-type]
    if isinstance(resolved, NeedsStructuredInput):
        return resolved
    merged = dict(prepared)
    merged.update(resolved)
    return merged


def test_tool_composer_input_carries_real_dataframe(monkeypatch) -> None:
    """A tool_composer dispatch with brand/region entities -> ``data`` is a frame.

    The fake resolver stands in for the real Supabase-backed resolver; the
    assertion is on the WIRING (the dispatcher derives brand/region, calls the
    resolver, and threads the returned frame under ``data``).
    """
    fake_frame = pd.DataFrame({"engagement_score": [1, 2, 3], "treatment_initiated": [0, 1, 1]})
    captured: Dict[str, Any] = {}

    def fake_resolve(brand, region):  # noqa: ANN001
        captured["brand"] = brand
        captured["region"] = region
        return fake_frame

    monkeypatch.setattr("src.services.cohort_resolution.resolve_cohort_frame", fake_resolve)

    node = DispatcherNode()
    prepared = _resolved_input(
        node, _state_with_entities("Kisqali", "Northeast"), _tool_composer_dispatch()
    )

    assert captured == {"brand": "Kisqali", "region": "Northeast"}
    assert "data" in prepared
    assert isinstance(prepared["data"], pd.DataFrame)
    assert len(prepared["data"]) == 3
    # The contract pass-through fields remain intact.
    assert prepared["query"] == "What drives adoption and where are the gaps?"
    assert prepared["session_id"] == "sess-1"


def test_brand_region_fallback_to_user_context(monkeypatch) -> None:
    """When entities are absent, brand/region fall back to ``user_context``."""
    fake_frame = pd.DataFrame({"x": [1]})
    captured: Dict[str, Any] = {}

    def fake_resolve(brand, region):  # noqa: ANN001
        captured["brand"] = brand
        captured["region"] = region
        return fake_frame

    monkeypatch.setattr("src.services.cohort_resolution.resolve_cohort_frame", fake_resolve)

    node = DispatcherNode()
    prepared = _resolved_input(
        node,
        _state_with_entities(user_context={"brand": "Fabhalta", "region": "South"}),
        _tool_composer_dispatch(),
    )

    assert captured == {"brand": "Fabhalta", "region": "South"}
    assert isinstance(prepared["data"], pd.DataFrame)


def test_unrecognized_brand_proceeds_without_data(monkeypatch) -> None:
    """An unrecognized brand -> resolver returns None -> NO ``data`` key (fail closed).

    The dispatcher must NOT raise and must NOT add a ``data`` key (so
    ``ToolComposerAgent.run`` leaves ``estimation_data`` unset rather than being
    tripped by a ``None`` value). The real resolver returns ``None`` for an
    unrecognized brand; the fake mirrors that.
    """

    def fake_resolve(brand, region):  # noqa: ANN001
        return None

    monkeypatch.setattr("src.services.cohort_resolution.resolve_cohort_frame", fake_resolve)

    node = DispatcherNode()
    prepared = _resolved_input(
        node, _state_with_entities("NotARealBrand", "Northeast"), _tool_composer_dispatch()
    )

    assert "data" not in prepared
    # Pass-through contract still intact.
    assert prepared["query"]


def test_resolver_exception_proceeds_without_data(monkeypatch) -> None:
    """A resolver exception is logged and swallowed -> NO ``data`` key (fail closed)."""

    def fake_resolve(brand, region):  # noqa: ANN001
        raise RuntimeError("supabase down")

    monkeypatch.setattr("src.services.cohort_resolution.resolve_cohort_frame", fake_resolve)

    node = DispatcherNode()
    prepared = _resolved_input(
        node, _state_with_entities("Kisqali", "Northeast"), _tool_composer_dispatch()
    )

    assert "data" not in prepared


def test_no_brand_or_region_skips_resolution(monkeypatch) -> None:
    """With neither brand nor region, the dispatcher skips the resolver call entirely."""
    called = {"n": 0}

    def fake_resolve(brand, region):  # noqa: ANN001
        called["n"] += 1
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr("src.services.cohort_resolution.resolve_cohort_frame", fake_resolve)

    node = DispatcherNode()
    prepared = _resolved_input(node, _state_with_entities(), _tool_composer_dispatch())

    assert called["n"] == 0
    assert "data" not in prepared


def test_other_agents_have_no_tool_composer_data_resolver() -> None:
    """Scoping: non-tool_composer agents never gain a ``data`` key.

    The registry enforces this structurally — agents that do not declare a
    ``data``-threading resolver simply are not in ``INPUT_RESOLVERS`` (or their
    resolver does not emit ``data``). drift_monitor's wrapped input model would
    TypeError on an undeclared ``data`` kwarg, so it must never receive one.
    """
    for other in ("causal_impact", "gap_analyzer", "drift_monitor"):
        resolver = disp.INPUT_RESOLVERS.get(other)
        if resolver is None:
            continue  # no resolver → can never add a data key
        out = resolver({"query": "q", "session_id": "s"}, _other_dispatch(other))
        if isinstance(out, dict):
            assert "data" not in out, f"{other} resolver unexpectedly emitted a data key"


def test_extract_brand_region_prefers_entities_over_user_context() -> None:
    """Unit-level: parsed_query entities win over user_context for brand/region."""
    payload = {
        "parsed_query": {
            "entities": [
                {"type": "brand", "value": "Kisqali"},
                {"type": "region", "value": "West"},
            ]
        },
        "user_context": {"brand": "Fabhalta", "region": "South"},
    }
    assert disp._extract_brand_region(payload) == ("Kisqali", "West")


def _kpi_frame(is_truncated: bool):
    """Build a minimal real KpiFrame (no DB) for the truncation-provenance tests."""
    from src.services.kpi_resolution import KpiFrame

    return KpiFrame(
        frame=pd.DataFrame({"accepted": [0, 1], "converted": [0, 1]}),
        outcome_column="converted",
        driver_columns=["accepted"],
        treatment_column="accepted",
        kpi_id="WS3-BI-009",
        kpi_name="Conversion Rate",
        is_truncated=is_truncated,
    )


def test_tool_composer_kpi_truncation_provenance_threaded(monkeypatch) -> None:
    """#810 / codex MED: a TRUNCATED KPI substrate must surface ``kpi_truncated``
    on the orchestrator path (parity with the chatbot path) — never dropped."""
    monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
    monkeypatch.setattr(
        "src.services.kpi_resolution.resolve_kpi_frame", lambda *a, **k: _kpi_frame(True)
    )

    node = DispatcherNode()
    prepared = _resolved_input(
        node, _state_with_entities("Kisqali", "Northeast"), _tool_composer_dispatch()
    )
    assert isinstance(prepared["data"], pd.DataFrame)
    assert prepared["kpi_outcome"] == "converted"
    assert prepared["kpi_truncated"] is True


def test_tool_composer_kpi_not_truncated_omits_flag(monkeypatch) -> None:
    """A non-truncated KPI substrate must NOT add the ``kpi_truncated`` flag."""
    monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
    monkeypatch.setattr(
        "src.services.kpi_resolution.resolve_kpi_frame", lambda *a, **k: _kpi_frame(False)
    )

    node = DispatcherNode()
    prepared = _resolved_input(
        node, _state_with_entities("Kisqali", "Northeast"), _tool_composer_dispatch()
    )
    assert prepared["kpi_outcome"] == "converted"
    assert "kpi_truncated" not in prepared
