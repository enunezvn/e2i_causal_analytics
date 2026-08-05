"""#1475 target-2: the explainer resolver binds REAL evidence for the two
query classes that dead-ended on /chat/stream's multi-agent path.

Measured defect (2026-08-05, live logs):

* KPI value lookups ("What is the TRx for Kisqali?") classify as the legacy
  ``explanation`` intent — which is GOLD-CORRECT (#1337 pins 111/337 rows at
  agent=explainer) — but ``_resolve_explainer_input`` found no upstream
  ``analysis_results`` and fell straight to the #883 fail-closed return, so the
  orchestrator reported ``all agents failed - ['explainer']`` and the chat
  bridge answered instead of the multi-agent path.
* Causal asks ("What is the causal impact of rep visits on TRx for Kisqali?")
  fail-fast in the ``causal_impact`` resolver (only Conversion Rate has a KPI
  frame builder), and the explainer FALLBACK then failed closed identically:
  ``all agents failed - ['causal_impact','explainer']``.

The fix adds two REAL-evidence binding branches BEFORE that fail-closed return:

* **Branch A** — a KPI-shaped lookup binds the value the KPI engine actually
  computes (vetted registry SQL), gated by the SSOT regex the intent classifier
  already uses for this shape.
* **Branch B** — a causal ask (or an explainer fallback after a structural
  ``causal_impact`` failure) binds the curated ``causal_paths`` registry rows.

#883's anti-fabrication contract is UNCHANGED: nothing is ever invented, and a
query with no resolvable substrate still fails closed with the same message.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pytest

from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes.dispatcher import DispatcherNode, NeedsStructuredInput
from src.kpi.models import KPIResult, KPIStatus

# --------------------------------------------------------------------------
# Fixtures / fakes — mocking happens ONLY at the two service seams named in
# the design (get_kpi_calculator / the sync causal-paths helper). The resolver
# logic under test is never mocked.
# --------------------------------------------------------------------------

KPI_QUERY = "What is the TRx for Kisqali?"
CAUSAL_QUERY = "What is the causal impact of rep visits on TRx for Kisqali?"
FORECAST_QUERY = "what is the trx for next quarter expected to be?"


def _dispatch(agent_name: str = "explainer", params: Optional[Dict[str, Any]] = None):
    return {
        "agent_name": agent_name,
        "priority": "high",
        "parameters": params or {},
        "timeout_ms": 15000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


def _agent_input(query: str, *, agent_results: Optional[List[Dict[str, Any]]] = None):
    return {
        "query": query,
        "session_id": "sess-1475",
        "user_context": {},
        "parsed_query": {"entities": []},
        "agent_results": agent_results or [],
    }


def _state(query: str) -> Dict[str, Any]:
    return {
        "query": query,
        "user_context": {"user_id": "u1"},
        "session_id": "sess-1475",
        "parsed_query": {"intent": "explanation", "entities": []},
        "dispatch_plan": [_dispatch("explainer")],
        "parallel_groups": [["explainer"]],
    }


def _kpi_result(
    *,
    value: Optional[float] = 12345.0,
    error: Optional[str] = None,
    data_through: Optional[str] = "2025-04-23",
) -> KPIResult:
    """A REAL KPIResult (not a mock) shaped exactly as the engine returns one."""
    context: Dict[str, Any] = {}
    if data_through is not None:
        context["data_through"] = data_through
    return KPIResult(
        kpi_id="WS3-BI-005",
        value=value,
        status=KPIStatus.GOOD,
        error=error,
        metadata={"context": context, "include_synthetic": False},
    )


class _StubCalculator:
    """Records the (kpi_id, context) the resolver asks the engine for."""

    def __init__(self, result: Optional[KPIResult] = None, exc: Optional[Exception] = None):
        self.result = result
        self.exc = exc
        self.calls: List[tuple] = []

    def calculate(
        self,
        kpi_id: str,
        use_cache: bool = True,
        force_refresh: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> KPIResult:
        self.calls.append((kpi_id, dict(context or {})))
        if self.exc is not None:
            raise self.exc
        assert self.result is not None
        return self.result


def _install_calculator(monkeypatch, stub: _StubCalculator) -> _StubCalculator:
    monkeypatch.setattr("src.api.routes.kpi.get_kpi_calculator", lambda: stub)
    return stub


PATH_ROW = {
    "path_id": "cp-1475-a",
    "start_node": "rep_visits",
    "end_node": "trx_volume",
    "causal_effect_size": 0.18,
    "confidence_level": 0.82,
    "method_used": "dowhy",
    "validation_status": "validated",
}
PATH_ROW_2 = {
    "path_id": "cp-1475-b",
    "start_node": "speaker_programs",
    "end_node": "trx_volume",
    "causal_effect_size": 0.07,
    "confidence_level": 0.74,
    "method_used": "econml",
    "validation_status": "pending",
}


class _PathRecorder:
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, outcome_term: str, **kwargs: Any) -> List[Dict[str, Any]]:
        self.calls.append({"outcome_term": outcome_term, **kwargs})
        return list(self.rows)


def _install_paths(monkeypatch, rows: List[Dict[str, Any]]) -> _PathRecorder:
    recorder = _PathRecorder(rows)
    monkeypatch.setattr("src.repositories.causal_path.search_paths_for_outcome_sync", recorder)
    return recorder


# --------------------------------------------------------------------------
# 1. Branch A — a KPI lookup with no upstream binds the REAL computed value
# --------------------------------------------------------------------------


def test_kpi_lookup_binds_real_calculated_value(monkeypatch) -> None:
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(KPI_QUERY), _dispatch())

    assert isinstance(resolved, dict), f"expected a bound payload, got {resolved!r}"
    results = resolved["analysis_results"]
    assert len(results) == 1
    payload = results[0]
    # context_assembler._extract_context reads "agent" / "analysis_type".
    assert payload["agent"] == "kpi_calculator"
    assert payload["analysis_type"] == "kpi_lookup"
    assert payload["kpi_id"] == "WS3-BI-005"
    assert payload["value"] == 12345.0
    # key_findings MUST be non-empty and MUST carry the value — the explainer's
    # deterministic template renders "0 key finding(s)" husks otherwise.
    findings = payload["key_findings"]
    assert findings and all(isinstance(f, str) for f in findings)
    assert "12,345" in findings[0], findings
    assert "Total Prescriptions (TRx)" in findings[0], findings
    assert "Kisqali" in findings[0], findings
    assert "2025-04-23" in findings[0], "the engine's real data_through must be cited"
    # The engine was asked for the REAL brand parsed out of the query text.
    assert stub.calls == [("WS3-BI-005", {"brand": "Kisqali"})]


def test_kpi_lookup_binds_window_named_in_the_query(monkeypatch) -> None:
    """A user-named window is honored via the KPI engine's own parser."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("Show me Kisqali TRx for the last 30 days"), _dispatch()
    )

    assert isinstance(resolved, dict), resolved
    assert len(stub.calls) == 1
    _kpi_id, context = stub.calls[0]
    assert context["brand"] == "Kisqali"
    window = context["window"]
    assert set(window) == {"start", "end"}
    assert window["start"] < window["end"]


# --------------------------------------------------------------------------
# 2. A non-KPI ask with no upstream STILL fails closed (#883 untouched)
# --------------------------------------------------------------------------


def test_non_kpi_query_still_fails_closed(monkeypatch) -> None:
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input("explain the analysis"), _dispatch())

    assert isinstance(resolved, NeedsStructuredInput)
    assert resolved.missing == ("analysis_results",)
    assert stub.calls == [], "a bare chat ask must not hit the KPI engine at all"


# --------------------------------------------------------------------------
# 3. Calculator error / no value → fail closed (never a fabricated figure)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stub", "case"),
    [
        (_StubCalculator(exc=RuntimeError("supabase down")), "raises"),
        (_StubCalculator(_kpi_result(value=None, error="no rows")), "engine error"),
        (_StubCalculator(_kpi_result(value=None)), "value is None"),
    ],
)
def test_kpi_lookup_without_a_real_value_fails_closed(monkeypatch, stub, case) -> None:
    _install_calculator(monkeypatch, stub)

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(KPI_QUERY), _dispatch())

    assert isinstance(resolved, NeedsStructuredInput), f"{case}: {resolved!r}"
    assert resolved.missing == ("analysis_results",)


def test_kpi_lookup_binds_a_real_zero(monkeypatch) -> None:
    """0.0 is a REAL computed value, not a missing one — it must bind."""
    _install_calculator(monkeypatch, _StubCalculator(_kpi_result(value=0.0)))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(KPI_QUERY), _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["value"] == 0.0


# --------------------------------------------------------------------------
# 4. Branch B — the causal fallback binds curated registry paths
# --------------------------------------------------------------------------


def _failed_causal_impact_input(query: str = CAUSAL_QUERY) -> Dict[str, Any]:
    return _agent_input(
        query,
        agent_results=[
            {
                "agent_name": "causal_impact",
                "success": False,
                "result": None,
                "error": "causal_impact fails closed: no KPI frame builder",
            }
        ],
    )


def test_causal_fallback_binds_registry_paths(monkeypatch) -> None:
    recorder = _install_paths(monkeypatch, [PATH_ROW, PATH_ROW_2])

    resolved = disp.INPUT_RESOLVERS["explainer"](_failed_causal_impact_input(), _dispatch())

    assert isinstance(resolved, dict), f"expected a bound payload, got {resolved!r}"
    payload = resolved["analysis_results"][0]
    assert payload["analysis_type"] == "causal_paths_registry"
    findings = payload["key_findings"]
    assert findings and all(isinstance(f, str) for f in findings)
    joined = " | ".join(findings)
    assert "rep_visits" in joined and "trx_volume" in joined
    assert "0.18" in joined, "the real effect size must be carried"
    assert "0.82" in joined, "the real confidence must be carried"
    assert "validated" in joined, "the validation status must be carried"
    # confidence comes from the paths themselves, never a default guess.
    assert payload["confidence"] == pytest.approx(0.82)
    # The registry was asked for the recognized outcome + the real brand.
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["outcome_term"] == "Total Prescriptions (TRx)"
    assert call["brand"] == "Kisqali"


def test_causal_fallback_with_empty_registry_fails_closed(monkeypatch) -> None:
    _install_paths(monkeypatch, [])

    resolved = disp.INPUT_RESOLVERS["explainer"](_failed_causal_impact_input(), _dispatch())

    assert isinstance(resolved, NeedsStructuredInput)
    assert resolved.missing == ("analysis_results",)


def test_causal_ask_without_a_failed_sibling_also_binds(monkeypatch) -> None:
    """A directly-dispatched causal ask (no failed sibling in state) resolves
    the same curated substrate — the gate is the ASK, not only the fallback."""
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(CAUSAL_QUERY), _dispatch())

    assert isinstance(resolved, dict), resolved
    assert recorder.calls, "the registry must have been consulted"


def test_registry_is_not_consulted_for_a_bare_chat_ask(monkeypatch) -> None:
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input("explain the analysis"), _dispatch())

    assert isinstance(resolved, NeedsStructuredInput)
    assert recorder.calls == []


# --------------------------------------------------------------------------
# 5. include_synthetic follows the platform gate in BOTH states
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("env_value", "expected"), [("1", True), ("0", False)])
def test_include_synthetic_follows_the_platform_gate(monkeypatch, env_value, expected) -> None:
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", env_value)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    resolved = disp.INPUT_RESOLVERS["explainer"](_failed_causal_impact_input(), _dispatch())

    assert isinstance(resolved, dict), resolved
    assert recorder.calls[0]["include_synthetic"] is expected
    assert resolved["analysis_results"][0]["data_source"] == (
        "synthetic" if expected else "database"
    )


# --------------------------------------------------------------------------
# 6. E2E: the dispatcher reports SUCCESS and the synthesizer completes
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_and_synthesizer_complete_on_a_kpi_lookup(monkeypatch) -> None:
    """The measured live failure was 'all agents failed - [explainer]' →
    status=failed → the chat bridge answered. With the value bound, the real
    ExplainerAgent succeeds and the synthesizer reports 'completed'."""
    from src.agents.explainer import ExplainerAgent
    from src.agents.orchestrator.nodes.synthesizer import SynthesizerNode

    _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    node = DispatcherNode(agent_registry={"explainer": ExplainerAgent(use_llm=False)})
    out = await node.execute(_state(KPI_QUERY))

    res = out["agent_results"][0]
    assert res["success"] is True, res["error"]
    assert res["agent_name"] == "explainer"

    synthesized = await SynthesizerNode().execute(out)
    assert synthesized["status"] == "completed"
    assert synthesized["synthesized_response"]


# --------------------------------------------------------------------------
# 7. Explainer integration: the value survives into the narrative
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explainer_narrates_the_bound_kpi_value(monkeypatch) -> None:
    """Guards the '0 key finding(s)' husk: the deterministic template path must
    surface the REAL figure in the user-visible narrative."""
    from src.agents.explainer import ExplainerAgent

    _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(KPI_QUERY), _dispatch())
    assert isinstance(resolved, dict), resolved

    output = await ExplainerAgent(use_llm=False).explain(**resolved)

    narrative = f"{output.executive_summary}\n{output.detailed_explanation}"
    assert "12,345" in narrative, narrative
    assert "0 key finding" not in narrative, narrative


# --------------------------------------------------------------------------
# 8. Forecast-shaped asks never enter Branch A (the regex IS the gate)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        FORECAST_QUERY,
        "what is the expected TRx for Kisqali next month?",
        "show me the trx forecast for Fabhalta",
        "what is the likelihood of TRx growth for Kisqali?",
    ],
)
def test_forecast_shaped_asks_do_not_bind_a_kpi_value(monkeypatch, query) -> None:
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    _install_paths(monkeypatch, [])

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(query), _dispatch())

    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert stub.calls == [], "a forecast ask must never be answered with a current value"


# --------------------------------------------------------------------------
# SSOT guards — one regex, one token-matcher; no forked copies
# --------------------------------------------------------------------------


def test_kpi_lookup_regex_is_shared_with_the_intent_classifier() -> None:
    """The resolver's gate and INTENT_PATTERNS['explanation'] must be the SAME
    pattern object — a forked copy would let routing and binding drift apart."""
    from src.agents.orchestrator.nodes.intent_classifier import (
        KPI_VALUE_LOOKUP_PATTERN,
        KPI_VALUE_LOOKUP_RE,
        IntentClassifierNode,
    )

    assert KPI_VALUE_LOOKUP_RE.pattern == KPI_VALUE_LOOKUP_PATTERN
    explanation_patterns = IntentClassifierNode.INTENT_PATTERNS["explanation"]
    assert any(p is KPI_VALUE_LOOKUP_PATTERN for p in explanation_patterns), (
        "INTENT_PATTERNS['explanation'] must reference the module-level constant, "
        "not an inline copy of the regex"
    )
    assert KPI_VALUE_LOOKUP_RE.flags & re.IGNORECASE


class _RecordingQuery:
    """Minimal supabase-py query-builder stand-in that records the filters."""

    def __init__(self, recorder: Dict[str, Any]):
        self._recorder = recorder
        self._recorder.setdefault("filters", [])

    def select(self, *args, **kwargs):
        self._recorder["filters"].append(("select", args))
        return self

    def or_(self, expr):
        self._recorder["or_"] = expr
        return self

    def ilike(self, col, value):
        self._recorder["filters"].append(("ilike", col, value))
        return self

    def eq(self, col, value):
        self._recorder["filters"].append(("eq", col, value))
        return self

    def gte(self, col, value):
        self._recorder["filters"].append(("gte", col, value))
        return self

    def order(self, col, desc=False):
        self._recorder["filters"].append(("order", col, desc))
        return self

    def limit(self, n):
        self._recorder["filters"].append(("limit", n))
        return self


class _SyncRecordingClient:
    def __init__(self, recorder: Dict[str, Any], rows: List[Dict[str, Any]]):
        self._recorder = recorder
        self._rows = rows

    def table(self, name):
        self._recorder["table"] = name
        return _SyncRecordingQuery(self._recorder, self._rows)


class _SyncRecordingQuery(_RecordingQuery):
    def __init__(self, recorder, rows):
        super().__init__(recorder)
        self._rows = rows

    def execute(self):
        class _Res:
            data = self._rows

        return _Res()


class _AsyncRecordingClient:
    def __init__(self, recorder: Dict[str, Any], rows: List[Dict[str, Any]]):
        self._recorder = recorder
        self._rows = rows

    def table(self, name):
        self._recorder["table"] = name
        return _AsyncRecordingQuery(self._recorder, self._rows)


class _AsyncRecordingQuery(_RecordingQuery):
    def __init__(self, recorder, rows):
        super().__init__(recorder)
        self._rows = rows

    async def execute(self):
        class _Res:
            data = self._rows

        return _Res()


@pytest.mark.asyncio
async def test_sync_and_async_causal_path_search_build_the_same_filters() -> None:
    """The sync helper the resolver needs (the dispatcher contract is SYNC —
    resolvers run inside ``asyncio.to_thread``) and the async repository method
    must derive their node filters from the SAME ``outcome_match_tokens``, so
    the chat answer and the orchestrator answer can never disagree."""
    from src.repositories.causal_path import (
        CausalPathRepository,
        outcome_match_tokens,
        search_paths_for_outcome_sync,
    )

    term = "Total Prescriptions (TRx)"
    tokens = outcome_match_tokens(term)
    expected_or = ",".join(
        f"{col}.ilike.%{token}%" for token in tokens for col in ("start_node", "end_node")
    )

    sync_rec: Dict[str, Any] = {}
    sync_rows = search_paths_for_outcome_sync(
        term,
        client=_SyncRecordingClient(sync_rec, [PATH_ROW]),
        brand="Kisqali",
        min_confidence=0.7,
        limit=15,
    )

    async_rec: Dict[str, Any] = {}
    repo = CausalPathRepository(_AsyncRecordingClient(async_rec, [PATH_ROW]))
    async_rows = await repo.search_paths_for_outcome(
        term, brand="Kisqali", min_confidence=0.7, limit=15
    )

    assert sync_rec["or_"] == expected_or == async_rec["or_"]
    assert sync_rec["table"] == async_rec["table"] == "causal_paths"
    assert sync_rec["filters"] == async_rec["filters"]
    assert sync_rows == async_rows == [PATH_ROW]
