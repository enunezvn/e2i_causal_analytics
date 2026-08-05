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
    payload = _agent_input(
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
    # The prod shape: _dispatch_fallback stamps WHICH agent this dispatch
    # stands in for (pinned by test_dispatch_fallback_marks_its_origin) —
    # fallback detection is dispatch-scoped, never a scan of the accumulated
    # cross-turn agent_results channel (codex iter-4).
    payload["parameters"] = {"fallback_from": "causal_impact"}
    return payload


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


# --------------------------------------------------------------------------
# Codex iter-1 revisions — semantic notes, definition carry, governing-head
# guards (all three scenarios verified against real recognize_kpi first)
# --------------------------------------------------------------------------


def test_market_share_lookup_carries_the_semantic_note(monkeypatch) -> None:
    """[HIGH] 'market share' resolves to WS3-BI-008 TRx Share — tracked-portfolio
    share, NOT competitor market share. The chat tool pins that meaning to every
    answer via KPI_SEMANTIC_NOTES; the bound payload must carry the same note in
    key_findings (narrated) AND warnings (first-class extractor field), or a
    real number gets narrated as an answer to a question it does not answer."""
    from src.services.kpi_resolution import KPI_SEMANTIC_NOTES

    result = KPIResult(
        kpi_id="WS3-BI-008",
        value=0.341,
        status=KPIStatus.GOOD,
        metadata={"context": {"data_through": "2025-04-23"}, "include_synthetic": False},
    )
    _install_calculator(monkeypatch, _StubCalculator(result))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the market share for Kisqali compared to competitors?"),
        _dispatch(),
    )

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    note = KPI_SEMANTIC_NOTES["WS3-BI-008"]
    assert "NOT market share against external competitors" in note
    assert payload["warnings"] == [note]
    assert any("tracked portfolio" in f for f in payload["key_findings"]), payload["key_findings"]


@pytest.mark.asyncio
async def test_explainer_narrates_the_semantic_note(monkeypatch) -> None:
    """The note must survive into the user-visible narrative, not just the payload."""
    from src.agents.explainer import ExplainerAgent

    result = KPIResult(
        kpi_id="WS3-BI-008",
        value=0.341,
        status=KPIStatus.GOOD,
        metadata={"context": {"data_through": "2025-04-23"}, "include_synthetic": False},
    )
    _install_calculator(monkeypatch, _StubCalculator(result))
    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the market share for Kisqali compared to competitors?"),
        _dispatch(),
    )
    assert isinstance(resolved, dict), resolved

    output = await ExplainerAgent(use_llm=False).explain(**resolved)

    narrative = f"{output.executive_summary}\n{output.detailed_explanation}"
    assert "tracked portfolio" in narrative, narrative


def test_semantic_notes_ssot_lives_in_kpi_resolution() -> None:
    """chatbot_tools must re-export the SAME dict object — a fork would let the
    chat tool and the orchestrator disagree about what a KPI means. (The notes
    moved to kpi_resolution because importing chatbot_tools costs ~30s — it
    pulls the orchestrator/tool_composer/RAG stacks — which a sync resolver
    running inside asyncio.to_thread cannot afford on first call.)"""
    import ast
    import pathlib

    from src.services.kpi_resolution import KPI_SEMANTIC_NOTES

    assert set(KPI_SEMANTIC_NOTES) >= {"WS3-BI-008"}
    # Source-level check instead of importing chatbot_tools (30s): the module
    # must bind the name FROM kpi_resolution, not define its own dict literal.
    src_file = (
        pathlib.Path(__file__).resolve().parents[4] / "src" / "api" / "routes" / "chatbot_tools.py"
    )
    tree = ast.parse(src_file.read_text())
    defines_own = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "KPI_SEMANTIC_NOTES" for t in node.targets)
        for node in ast.walk(tree)
    )
    imports_ssot = any(
        isinstance(node, ast.ImportFrom)
        and node.module
        and "kpi_resolution" in node.module
        and any(alias.name == "KPI_SEMANTIC_NOTES" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert imports_ssot and not defines_own


def test_bare_metric_ask_binds_value_with_definition(monkeypatch) -> None:
    """[MEDIUM-rebuttal] 'What is NRx?' has no gold row; on this analytics
    platform the value reading is the measured-majority intent (bench-0113 class
    asks 'what is X' meaning the number). Bind the REAL value — and carry the
    registry definition so a definition-seeking reader is served too."""
    result = KPIResult(
        kpi_id="WS3-BI-006",
        value=4210.0,
        status=KPIStatus.GOOD,
        metadata={"context": {"data_through": "2025-04-23"}, "include_synthetic": False},
    )
    _install_calculator(monkeypatch, _StubCalculator(result))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input("What is NRx?"), _dispatch())

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["value"] == 4210.0
    assert payload["definition"], "the registry definition must ride along"


def test_kpi_as_modifier_ask_fails_closed(monkeypatch) -> None:
    """[MEDIUM] 'the cost of TRx' names TRx as a MODIFIER of a head noun the
    platform does not model (cost). Binding TRx drivers would answer a question
    the user did not ask — fail closed instead (the bridge handles it)."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _failed_causal_impact_input("what drives the cost of TRx up for Kisqali?"),
        _dispatch(),
    )

    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert stub.calls == []
    assert recorder.calls == []


def test_drivers_of_kpi_binds_causal_paths_not_a_value(monkeypatch) -> None:
    """'drivers of TRx' is a causal frame with TRx as the OUTCOME (gold
    bench-0113 pins 'what drives this NRX?' -> causal_impact): Branch A's
    value-head guard must skip it, Branch B must bind registry paths."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What are the drivers of TRx for Kisqali?"), _dispatch()
    )

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "causal_paths_registry"
    assert stub.calls == [], "a drivers ask must never bind a bare value"
    assert recorder.calls, "the registry must have been consulted"


def test_value_of_kpi_still_binds_the_value(monkeypatch) -> None:
    """'the value of TRx' is a value ask — the head guard must whitelist
    value-heads, not veto every '<head> of <kpi>' construction."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the value of TRx for Kisqali?"), _dispatch()
    )

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["value"] == 12345.0
    assert len(stub.calls) == 1


def test_causal_fallback_never_binds_a_bare_value(monkeypatch) -> None:
    """Self-audit hole: 'What is the impact of TRx on conversion rate?' after a
    failed causal_impact matched Branch A through the lookup regex's {0,3} gap
    ('impact of trx' fits) and bound a Conversion Rate VALUE — a value does not
    answer a causal ask. On a causal-fallback turn only Branch B may bind."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _failed_causal_impact_input("What is the impact of TRx on conversion rate?"),
        _dispatch(),
    )

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "causal_paths_registry"
    assert stub.calls == [], "a causal fallback must never bind a bare KPI value"
    assert recorder.calls and recorder.calls[0]["outcome_term"] == "Conversion Rate"


def test_determinants_of_brand_kpi_binds_paths_not_a_value(monkeypatch) -> None:
    """[iter-3 HIGH] A brand token between 'of' and the KPI must not strip the
    causal head: 'determinants of Kisqali NRx' is a causal ask ('determinants'
    is outside the causal_effect lexicon, so it arrives as a DIRECT explanation
    turn) — binding an NRx value would answer a question the user did not ask."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What are the determinants of Kisqali NRx?"), _dispatch()
    )

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "causal_paths_registry"
    assert stub.calls == [], "a determinants ask must never bind a bare value"
    assert recorder.calls, "the registry must have been consulted"


def test_drivers_of_brand_kpi_binds_paths_not_a_value(monkeypatch) -> None:
    """[iter-3 HIGH, same class] 'drivers of Fabhalta TRx' — routing usually
    sends this via causal_impact, but the resolver must be safe standalone
    (DSPy intent is non-deterministic): Branch A must see 'drivers' through
    the intervening brand token and skip, Branch B must bind paths."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What are the drivers of Fabhalta TRx?"), _dispatch()
    )

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "causal_paths_registry"
    assert stub.calls == [], "a drivers ask must never bind a bare value"
    assert recorder.calls, "the registry must have been consulted"


def test_cost_of_brand_kpi_fails_closed(monkeypatch) -> None:
    """[iter-3, opposite guard] 'the cost of Kisqali TRx' keeps TRx as a
    MODIFIER of an unmodeled head even with the brand in between — neither a
    value nor registry paths answer it; fail closed (the bridge handles it)."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the cost of Kisqali TRx?"), _dispatch()
    )

    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert stub.calls == []
    assert recorder.calls == []


def test_value_of_brand_kpi_still_binds_the_value(monkeypatch) -> None:
    """[iter-3, opposite guard] 'the value of Kisqali TRx' stays a value ask
    when the brand rides inside the of-chain — the widened head detector must
    still whitelist value-heads."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the value of Kisqali TRx?"), _dispatch()
    )

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["value"] == 12345.0
    assert len(stub.calls) == 1


def test_stale_causal_failure_does_not_hijack_a_fresh_value_ask(monkeypatch) -> None:
    """[iter-4 HIGH] ``agent_results`` is an operator.add channel the Redis
    checkpointer restores across turns (#1442 class): a turn-1 failed
    causal_impact must not turn turn-2's plain value ask into a causal
    fallback. Fallback detection must be current-dispatch scoped."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    stale = _agent_input(
        KPI_QUERY,
        agent_results=[
            {
                "agent_name": "causal_impact",
                "success": False,
                "result": None,
                "error": "prior turn's failure",
            },
            {"agent_name": "explainer", "success": True, "result": {}},
        ],
    )
    resolved = disp.INPUT_RESOLVERS["explainer"](stale, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["value"] == 12345.0
    assert len(stub.calls) == 1
    assert recorder.calls == [], "a stale failure must not summon registry paths"


async def test_dispatch_fallback_marks_its_origin(monkeypatch) -> None:
    """[iter-4 HIGH, prod-shape pin] the fallback dispatch must carry WHICH
    agent it stands in for, so the resolver never has to scan the accumulated
    (cross-turn) agent_results channel to reconstruct it."""
    node = disp.DispatcherNode()
    captured: Dict[str, Any] = {}

    async def fake_dispatch_agent(dispatch, state):
        captured["dispatch"] = dispatch
        return {"agent_name": dispatch["agent_name"], "success": False, "result": None}

    monkeypatch.setattr(node, "_dispatch_agent", fake_dispatch_agent)
    await node._dispatch_fallback("explainer", _state(KPI_QUERY), fallback_from="causal_impact")

    assert captured["dispatch"]["parameters"]["fallback_from"] == "causal_impact"


def test_temporal_of_phrase_still_binds_the_value(monkeypatch) -> None:
    """[iter-4 HIGH] 'end of Q2' / 'as of Q2' are TEMPORAL of-phrases, not
    governing heads — the widened of-chain must not veto a legitimate value
    ask over them (the window probe handles the period)."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the end of Q2 TRx?"), _dispatch()
    )
    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["value"] == 12345.0

    resolved2 = disp.INPUT_RESOLVERS["explainer"](_agent_input("Show me as of Q2 TRx"), _dispatch())
    assert isinstance(resolved2, dict), resolved2
    assert len(stub.calls) == 2


def test_share_of_brand_kpi_resolves_the_share_kpi(monkeypatch) -> None:
    """[iter-4 HIGH] 'the share of Kisqali TRx' is WS3-BI-008 phrasing with the
    brand riding inside the of-chain — it must resolve TRx Share, not fall to
    the bare 'trx' alias and die on the 'share' head veto."""
    result = KPIResult(
        kpi_id="WS3-BI-008",
        value=0.341,
        status=KPIStatus.GOOD,
        metadata={"context": {"data_through": "2025-04-23"}, "include_synthetic": False},
    )
    stub = _install_calculator(monkeypatch, _StubCalculator(result))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the share of Kisqali TRx?"), _dispatch()
    )

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["kpi_id"] == "WS3-BI-008"
    assert stub.calls[0][0] == "WS3-BI-008"
    assert any("tracked portfolio" in f for f in payload["key_findings"])


def test_multi_kpi_value_ask_fails_closed(monkeypatch) -> None:
    """[iter-4 HIGH] 'TRx and NRx' names TWO metrics; binding one and
    presenting it as the whole answer is a wrong answer. Fail closed (the
    bridge answers multi-KPI asks today) until multi-KPI binding exists.
    A repeated mention of the SAME KPI ('TRx ... total prescriptions') is
    not a multi-KPI ask and must still bind."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What are the TRx and NRx for Kisqali?"), _dispatch()
    )
    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert stub.calls == []
    assert recorder.calls == []

    resolved2 = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the TRx, the total prescriptions, for Kisqali?"),
        _dispatch(),
    )
    assert isinstance(resolved2, dict), resolved2
    assert len(stub.calls) == 1


def test_fresh_value_ask_outranks_stale_upstream_success(monkeypatch) -> None:
    """[iter-5 HIGH] the accumulated channel carries PRIOR turns' successes:
    turn-1's gap analysis must not be narrated as the answer to turn-2's
    'What is the TRx?' — an explicit value ask is never anaphoric."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    stale = _agent_input(
        KPI_QUERY,
        agent_results=[
            {
                "agent_name": "gap_analyzer",
                "success": True,
                "result": {"gaps": ["stale gap"], "summary": "prior turn's analysis"},
            }
        ],
    )
    resolved = disp.INPUT_RESOLVERS["explainer"](stale, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["value"] == 12345.0
    assert len(stub.calls) == 1


def test_anaphoric_ask_still_binds_upstream_results(monkeypatch) -> None:
    """[iter-5, opposite guard] 'Explain the analysis' IS anaphoric — the
    upstream-results substrate (#883 §3) must keep serving it; the value
    branch must not fire (no KPI mention)."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    payload = _agent_input(
        "Explain the analysis",
        agent_results=[
            {
                "agent_name": "gap_analyzer",
                "success": True,
                "result": {"gaps": ["gap A"], "summary": "the gap analysis"},
            }
        ],
    )
    resolved = disp.INPUT_RESOLVERS["explainer"](payload, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["summary"] == "the gap analysis"
    assert stub.calls == []


def test_half_of_kpi_fails_closed(monkeypatch) -> None:
    """[iter-5 HIGH] 'half of Kisqali TRx' asks for a TRANSFORMATION the
    platform does not model — binding the full value would answer a different
    question. 'half' is not a temporal idiom; fail closed."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is half of Kisqali TRx?"), _dispatch()
    )

    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert stub.calls == []


def test_second_metric_named_by_registry_name_fails_closed(monkeypatch) -> None:
    """[iter-5 HIGH] 33 of 45 registry KPIs have no alias — a second metric
    named by its FULL registry name ('monthly active users' = WS3-BI-001) must
    still trip the multi-KPI veto, not vanish behind the alias-only probe."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What are the TRx and monthly active users?"), _dispatch()
    )

    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert stub.calls == []


def test_multi_outcome_causal_ask_fails_closed(monkeypatch) -> None:
    """[iter-5 HIGH] 'What drives TRx and NRx?' names TWO outcomes with no
    directional grammar — a singleton path answer chosen by alias order does
    not answer it. Fail closed (the bridge handles it)."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What drives TRx and NRx for Kisqali?"), _dispatch()
    )

    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert stub.calls == []
    assert recorder.calls == []


def test_same_turn_upstream_success_outranks_value_lookup(monkeypatch) -> None:
    """[iter-6 HIGH] bench-0143 (gold PARALLEL): 'What is the current total TRx
    and which region has the largest gap opportunity?' dispatches
    ['explainer','gap_analyzer'] in ONE turn — the fresh same-turn gap answer
    must not be shadowed by a bare KPI lookup. Current-turn results ride their
    own key, separate from the accumulated cross-turn channel."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    fresh = {
        "agent_name": "gap_analyzer",
        "success": True,
        "result": {"gaps": ["west region"], "summary": "this turn's gap analysis"},
    }
    payload = _agent_input(
        "What is the current total TRx and which region has the largest gap opportunity?",
        agent_results=[fresh],
    )
    payload["current_turn_agent_results"] = [fresh]
    resolved = disp.INPUT_RESOLVERS["explainer"](payload, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["summary"] == "this turn's gap analysis"
    assert stub.calls == [], "a fresh same-turn sibling result outranks a value bind"


def test_prepare_agent_input_threads_current_turn_results(monkeypatch) -> None:
    """[iter-6 HIGH, prod-shape pin] execute()'s _state_so_far stamps the
    results accumulated THIS turn under their own key, and
    _prepare_agent_input must thread it into the agent payload — without it
    the resolver cannot tell fresh siblings from prior turns' carry."""
    node = disp.DispatcherNode()
    fresh = {"agent_name": "gap_analyzer", "success": True, "result": {"gaps": ["g"]}}
    state = dict(_state(KPI_QUERY))
    state["agent_results"] = [fresh]
    state["current_turn_agent_results"] = [fresh]

    agent_input = node._prepare_agent_input(state, _dispatch())  # type: ignore[arg-type]

    assert agent_input["current_turn_agent_results"] == [fresh]


def test_common_word_abbreviations_stay_out_of_the_metric_probe(monkeypatch) -> None:
    """[iter-6 MEDIUM] 'Average Treatment Effect (ATE)' and 'Data Lag
    (Median)' must not put the English words 'ate'/'median' into the strict
    vocabulary — 'access issues ate into field time' is not a two-metric ask.
    Real initialisms (MAU, CATE, NRx) stay."""
    from src.services.kpi_resolution import _strict_metric_vocabulary

    phrases = {p for p, _ in _strict_metric_vocabulary()}
    assert "ate" not in phrases
    assert "median" not in phrases
    assert {"mau", "cate", "nrx"} <= phrases

    recorder = _install_paths(monkeypatch, [PATH_ROW])
    resolved = disp.INPUT_RESOLVERS["explainer"](
        _failed_causal_impact_input(
            "What drives TRx for Kisqali, given that access issues ate into field time?"
        ),
        _dispatch(),
    )
    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "causal_paths_registry"
    assert recorder.calls and recorder.calls[0]["outcome_term"] == "Total Prescriptions (TRx)"


def test_uppercase_ate_still_counts_as_a_second_metric(monkeypatch) -> None:
    """[iter-7 HIGH] blocking prose 'ate' must not erase the METRIC 'ATE':
    'What are the TRx and ATE for Kisqali?' names two metrics — the
    case-sensitive form in the ORIGINAL query is the tell. Fail closed like
    every other multi-metric ask."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What are the TRx and ATE for Kisqali?"), _dispatch()
    )

    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert stub.calls == []


def test_directed_causal_ask_binds_the_on_headed_outcome(monkeypatch) -> None:
    """[iter-5, direction pin] 'impact of conversion rate on TRx' names two
    metrics but the 'on <metric>' grammar identifies TRx as the OUTCOME — the
    resolver must bind TRx paths, not follow alias-length luck to Conversion
    Rate."""
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _failed_causal_impact_input("What is the impact of conversion rate on TRx?"),
        _dispatch(),
    )

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "causal_paths_registry"
    assert recorder.calls and recorder.calls[0]["outcome_term"] == "Total Prescriptions (TRx)"


def test_recognize_kpi_span_is_the_ssot_twin() -> None:
    """recognize_kpi_span must agree with recognize_kpi on every probe (it IS
    the same matcher, refactored to expose where the vocabulary hit)."""
    from src.services.kpi_resolution import recognize_kpi, recognize_kpi_span

    for q in (
        KPI_QUERY,
        CAUSAL_QUERY,
        "What is NRx?",
        "what drives the cost of TRx up for Kisqali?",
        "explain the analysis",
    ):
        kpi = recognize_kpi(q)
        span = recognize_kpi_span(q)
        if kpi is None:
            assert span is None
        else:
            span_kpi, normalized, start, end = span
            assert span_kpi.id == kpi.id
            assert 0 <= start < end <= len(normalized)


def test_right_headed_causal_ask_binds_paths_not_a_value(monkeypatch) -> None:
    """[codex iter-2 HIGH] 'What are TRx drivers for Kisqali?' fits the lookup
    regex ('what are' + 'trx') with no of-chain, so only a RIGHT-context guard
    stops Branch A from answering a causal ask with a bare value. The realistic
    route is the causal fallback, but a direct explainer dispatch must hold too."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    recorder = _install_paths(monkeypatch, [PATH_ROW])

    for query in ("What are TRx drivers for Kisqali?", "What are the NRx determinants?"):
        resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(query), _dispatch())
        assert isinstance(resolved, dict), (query, resolved)
        assert resolved["analysis_results"][0]["analysis_type"] == "causal_paths_registry"
    assert stub.calls == [], "a drivers/determinants ask must never bind a bare value"
    assert recorder.calls


@pytest.mark.asyncio
async def test_bare_definition_shape_narrates_the_definition(monkeypatch) -> None:
    """[codex iter-2 HIGH] data_summary never reaches the narrative — for a BARE
    'What is NRx?' (no brand/region/window) the registry definition must ride in
    key_findings so the deterministic path narrates it beside the value."""
    from src.agents.explainer import ExplainerAgent
    from src.kpi.registry import get_registry

    result = KPIResult(
        kpi_id="WS3-BI-006",
        value=4210.0,
        status=KPIStatus.GOOD,
        metadata={"context": {"data_through": "2025-04-23"}, "include_synthetic": False},
    )
    _install_calculator(monkeypatch, _StubCalculator(result))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input("What is NRx?"), _dispatch())
    assert isinstance(resolved, dict), resolved
    definition = get_registry().get("WS3-BI-006").definition
    assert any(definition in f for f in resolved["analysis_results"][0]["key_findings"])

    output = await ExplainerAgent(use_llm=False).explain(**resolved)
    narrative = f"{output.executive_summary}\n{output.detailed_explanation}"
    assert "4,210" in narrative, narrative
    assert definition[:40] in narrative, narrative


def test_scoped_value_ask_keeps_a_value_only_headline(monkeypatch) -> None:
    """A brand-scoped ask is unambiguously value-seeking: the definition stays
    in the payload (data_summary) but OUT of the narrated key_findings."""
    _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(KPI_QUERY), _dispatch())

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["definition"]
    assert not any(payload["definition"] in f for f in payload["key_findings"])


def test_share_of_trx_resolves_the_share_kpi(monkeypatch) -> None:
    """[codex iter-2 MEDIUM] 'the share of TRx' is natural WS3-BI-008 phrasing —
    it must resolve TRx Share (with its semantic note), not fall to the bare
    'trx' alias and die on the head guard."""
    result = KPIResult(
        kpi_id="WS3-BI-008",
        value=0.341,
        status=KPIStatus.GOOD,
        metadata={"context": {"data_through": "2025-04-23"}, "include_synthetic": False},
    )
    stub = _install_calculator(monkeypatch, _StubCalculator(result))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the share of TRx for Kisqali?"), _dispatch()
    )

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["kpi_id"] == "WS3-BI-008"
    assert stub.calls[0][0] == "WS3-BI-008"
    assert any("tracked portfolio" in f for f in payload["key_findings"])


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
