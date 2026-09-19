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


def test_full_kpi_name_lookup_binds_the_same_real_calculated_value(monkeypatch) -> None:
    """#2130: spelling out TRx must reach the same evidence path as TRx."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("Show me total prescriptions for Kisqali"), _dispatch()
    )

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["value"] == 12345.0
    assert stub.calls == [("WS3-BI-005", {"brand": "Kisqali"})]


@pytest.mark.parametrize(
    "query",
    [
        "How many people received new prescriptions?",
        "How many individuals received new prescriptions?",
        "How many pharmacies filled new prescriptions?",
        "Show me patients receiving new prescriptions",
        "Give me pharmacies filling new prescriptions",
        "Tell me about people with new prescriptions",
    ],
)
def test_entity_count_with_full_kpi_object_never_binds_a_bare_value(
    monkeypatch, query: str
) -> None:
    """The KPI is the object, not the quantity requested by ``how many``."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(query), _dispatch())

    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert stub.calls == []


@pytest.mark.parametrize(
    ("query", "kpi_id", "expected"),
    [
        # #2130 follow-up (codex r1 HIGH): scoped asks the old 3-word gap admitted. The
        # scope-word gap must keep BINDING them, not merely classify them (measured on
        # main 2026-09-19 with the same resolver, identical calls).
        (
            "What is the Northeast TRx for Kisqali?",
            "WS3-BI-005",
            {"brand": "Kisqali", "region": "northeast"},
        ),
        ("Show me last 30 days NRx for Fabhalta", "WS3-BI-006", {"brand": "Fabhalta"}),
        ("What is Kisqali's year-to-date TRx?", "WS3-BI-005", {"brand": "Kisqali"}),
        ("What was last month's NRx for Fabhalta?", "WS3-BI-006", {"brand": "Fabhalta"}),
        ("What is Q2 TRx for Kisqali?", "WS3-BI-005", {"brand": "Kisqali"}),
        (
            "Show me the competitor comparison market share for Kisqali",
            "WS3-BI-008",
            {"brand": "Kisqali"},
        ),
        ("what is teh currnt TRx for Kisqali?", "WS3-BI-005", {"brand": "Kisqali"}),
        ("What is Remibrutinib market share?", "WS3-BI-008", {"brand": "Remibrutinib"}),
        # codex r5: value heads the governing-head guard accepts (VALUE_OF_HEADS).
        ("What is the current level of TRx for Kisqali?", "WS3-BI-005", {"brand": "Kisqali"}),
        ("Give me the latest amount of NRx for Fabhalta", "WS3-BI-006", {"brand": "Fabhalta"}),
        ("What is the current figure of NBRx for Kisqali?", "WS3-BI-007", {"brand": "Kisqali"}),
        ("Show me the sum of TRx for Kisqali", "WS3-BI-005", {"brand": "Kisqali"}),
    ],
)
def test_scoped_kpi_lookup_still_binds_its_kpi(monkeypatch, query, kpi_id, expected) -> None:
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(query), _dispatch())

    assert isinstance(resolved, dict), resolved
    assert len(stub.calls) == 1, stub.calls
    called_id, context = stub.calls[0]
    assert called_id == kpi_id
    assert {k: context.get(k) for k in expected} == expected, context


def test_full_kpi_name_as_cost_modifier_never_binds_a_bare_value(monkeypatch) -> None:
    """Widening the route vocabulary must preserve the governing-head fence."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the cost of total prescriptions?"), _dispatch()
    )

    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert stub.calls == []


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

    def range(self, start, end):
        # #1716: the paged dedup read slices with .range(); recorded like the
        # other predicates so the sync/async filter-equivalence assertion
        # covers pagination too.
        self._recorder["filters"].append(("range", start, end))
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


# --------------------------------------------------------------------------
# #1572 — a multi-region phrase ("East Coast") must end in a clarify QUESTION,
# never a silent national figure. AG-UI already clarifies (tool surface,
# #1565/#1571); this pins the /chat multi-agent path's Branch A.
# --------------------------------------------------------------------------

EAST_COAST_QUERY = "What is the TRx for Kisqali on the East Coast?"


def test_multi_region_phrase_clarifies_instead_of_national_figure(monkeypatch) -> None:
    """'East Coast' spans the northeast AND south census regions — Branch A
    must return the clarify question and must NOT compute the national KPI."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(EAST_COAST_QUERY), _dispatch())

    assert isinstance(resolved, dict), resolved
    results = resolved["analysis_results"]
    assert len(results) == 1
    payload = results[0]
    assert payload["agent"] == "kpi_calculator"
    assert payload["analysis_type"] == "kpi_lookup_clarification"
    assert payload["needs_clarification"] is True
    assert payload["unresolved_region_phrase"].lower() == "east coast"
    findings = payload["key_findings"]
    assert findings and all(isinstance(f, str) for f in findings)
    # The question names all four census regions and echoes the phrase.
    for label in ("northeast", "south", "midwest", "west"):
        assert label in findings[0], findings
    assert "East Coast" in findings[0], findings
    assert "?" in findings[0], "the clarify must be a QUESTION"
    # The silent-national defect: the engine must never be asked for the
    # unscoped figure on this ask.
    assert stub.calls == []


def test_west_coast_still_binds_the_west_scoped_figure(monkeypatch) -> None:
    """'West Coast' resolves (every west-coast state is census-west, #1565) —
    the scoped figure keeps flowing; no spurious clarify."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the TRx for Kisqali on the West Coast?"), _dispatch()
    )

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["analysis_type"] == "kpi_lookup"
    assert payload["value"] == 12345.0
    assert stub.calls == [("WS3-BI-005", {"brand": "Kisqali", "region": "west"})]


def test_guard_phrase_no_longer_widens_to_an_unscoped_figure(monkeypatch) -> None:
    """'central coast' is neither a supported census region nor a resolvable
    scope.  Returning the national figure would silently drop that qualifier
    (#2141), so the value consumer must now fail closed before calculation."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the TRx for Kisqali on the central coast?"), _dispatch()
    )

    assert isinstance(resolved, NeedsStructuredInput), resolved
    assert stub.calls == []


@pytest.mark.asyncio
async def test_explainer_narrates_the_region_clarify_question(monkeypatch) -> None:
    """The clarify question must reach the user-visible narrative verbatim,
    with NO figure beside it."""
    from src.agents.explainer import ExplainerAgent

    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(EAST_COAST_QUERY), _dispatch())
    assert isinstance(resolved, dict), resolved

    output = await ExplainerAgent(use_llm=False).explain(**resolved)

    narrative = f"{output.executive_summary}\n{output.detailed_explanation}"
    assert "census region" in narrative, narrative
    assert "East Coast" in narrative, narrative
    assert "12,345" not in narrative, "no figure may accompany the clarify"
    assert stub.calls == []


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


# --------------------------------------------------------------------------
# #2114 (canonical TRx lane, codex Task 10 r1 HIGH) — an ask that names SEVERAL
# brands must end in a clarify QUESTION, never a portfolio total presented as a
# brand figure.
#
# ``query_entities.brand_from_text`` returns None both when the text names NO
# brand and when it names TWO ("the caller must keep its honest unscoped
# behaviour rather than guess"), and Branch A only sets context["brand"] when
# it is truthy — so by the time the calculator sees the context the two cases
# are indistinguishable. Task 10 made WS3-BI-007 answer an unbranded call with
# the all-brand portfolio NBRx instead of raising, which turns "What is NBRx
# for Kisqali and Fabhalta?" into a number belonging to neither brand.
#
# Same defect class, same fix location and same shape as the #1572 region
# clarify twelve lines above: the ambiguity is resolved where the information
# still exists (the dispatcher knows the ask text), not inside the calculator.
# The AG-UI tool surface already covers its own ingress (copilotkit.py's
# "Multiple brands in play - ambiguous is not absent" rule); this pins /chat's
# multi-agent Branch A.
# --------------------------------------------------------------------------

TWO_BRAND_NBRX_QUERY = "What is NBRx for Kisqali and Fabhalta?"


def test_two_brand_volume_ask_clarifies_instead_of_the_portfolio_total(monkeypatch) -> None:
    """The exact codex case: Branch A must ask which brand and must NOT ask the
    engine for the unbranded (portfolio) figure."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(TWO_BRAND_NBRX_QUERY), _dispatch())

    assert isinstance(resolved, dict), resolved
    results = resolved["analysis_results"]
    assert len(results) == 1
    payload = results[0]
    assert payload["agent"] == "kpi_calculator"
    assert payload["analysis_type"] == "kpi_lookup_clarification"
    assert payload["needs_clarification"] is True
    assert payload["kpi_id"] == "WS3-BI-007"
    assert payload["ambiguous_brands"] == ["Fabhalta", "Kisqali"]
    assert "value" not in payload
    findings = payload["key_findings"]
    assert findings and all(isinstance(f, str) for f in findings)
    for brand in ("Kisqali", "Fabhalta"):
        assert brand in findings[0], findings
    assert "?" in findings[0], "the clarify must be a QUESTION"
    # The defect itself: the portfolio figure must never be computed.
    assert stub.calls == []


@pytest.mark.parametrize(
    "query,kpi_id",
    [
        ("What is TRx for Kisqali and Fabhalta?", "WS3-BI-005"),
        ("What is NRx for Kisqali and Fabhalta?", "WS3-BI-006"),
        ("What is NBRx for Kisqali and Fabhalta?", "WS3-BI-007"),
        ("What is TRx share for Kisqali and Fabhalta?", "WS3-BI-008"),
        # The indication pass names brands just as surely as a brand token.
        ("What is NBRx for CSU and PNH?", "WS3-BI-007"),
    ],
)
def test_every_canonical_volume_kpi_clarifies_a_multi_brand_ask(monkeypatch, query, kpi_id) -> None:
    """005 and 006 widen an unbranded ask to the portfolio total (they did
    before this lane too); 007 started doing so in Task 10; 008 refuses. All
    four now ask instead -- one rule for the family, no per-id exceptions."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(query), _dispatch())

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["analysis_type"] == "kpi_lookup_clarification"
    assert payload["kpi_id"] == kpi_id
    assert stub.calls == []


def test_intentional_portfolio_volume_ask_still_computes(monkeypatch) -> None:
    """Naming NO brand is the portfolio ask the lane deliberately serves
    (plan line 141): it keeps computing, unbranded, with no clarify."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input("What is NBRx?"), _dispatch())

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["analysis_type"] == "kpi_lookup"
    assert payload["value"] == 12345.0
    assert stub.calls == [("WS3-BI-007", {})]


def test_single_brand_volume_ask_still_computes(monkeypatch) -> None:
    """One brand named binds it exactly as before; no spurious clarify."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is NBRx for Kisqali?"), _dispatch()
    )

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["analysis_type"] == "kpi_lookup"
    assert stub.calls == [("WS3-BI-007", {"brand": "Kisqali"})]


def test_a_structured_brand_wins_over_a_multi_brand_ask(monkeypatch) -> None:
    """Mirrors the region rule: the scan is consulted ONLY when no structured
    source bound a brand -- an explicit user_context/entities brand is a
    decision already taken, so it is honoured, not questioned."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    agent_input = _agent_input(TWO_BRAND_NBRX_QUERY)
    agent_input["user_context"] = {"brand": "Kisqali"}

    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "kpi_lookup"
    assert stub.calls == [("WS3-BI-007", {"brand": "Kisqali"})]


@pytest.mark.parametrize(
    "query",
    [
        "What is the conversion rate for Kisqali and PNH?",
        "What is the conversion rate for Kisqali and Fabhalta?",
    ],
)
def test_a_structured_brand_answers_a_non_volume_multi_brand_ask(monkeypatch, query) -> None:
    """Owner decision 2026-09-15 (plan Task 10c table): an ask where entities /
    user_context supplied a brand ANSWERS -- "a structured brand is a decision
    already taken and is never re-asked, even when the text grounds two". The value
    guard reads only the text, so without the structured brand it refused these
    (codex iter10 HIGH)."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    agent_input = _agent_input(query)
    agent_input["user_context"] = {"brand": "Kisqali"}

    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "kpi_lookup"
    assert stub.calls == [("WS3-BI-009", {"brand": "Kisqali"})]


def test_non_volume_kpi_refuses_rather_than_clarifies(monkeypatch) -> None:
    """Deliberate scope: the clarify covers the Rx-VOLUME family (the ids this
    lane owns, whose unbranded read is a portfolio aggregate). Conversion Rate
    (WS3-BI-009) used to compute UNSCOPED here -- the pre-existing exposure this
    lane left alone. main's #2141 closed it independently (an unresolved brand
    scope refuses), so the boundary now reads: volume KPIs ASK, others REFUSE,
    and neither computes a figure that silently dropped both brands."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the conversion rate for Kisqali and Fabhalta?"), _dispatch()
    )

    assert not isinstance(resolved, dict), resolved
    assert stub.calls == []


@pytest.mark.asyncio
async def test_explainer_narrates_the_brand_clarify_question(monkeypatch) -> None:
    """The question must reach the user-visible narrative verbatim, with NO
    figure beside it -- the whole point is that no number is presented."""
    from src.agents.explainer import ExplainerAgent

    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(TWO_BRAND_NBRX_QUERY), _dispatch())
    assert isinstance(resolved, dict), resolved

    output = await ExplainerAgent(use_llm=False).explain(**resolved)

    narrative = f"{output.executive_summary}\n{output.detailed_explanation}"
    assert "Kisqali" in narrative, narrative
    assert "Fabhalta" in narrative, narrative
    assert "12,345" not in narrative, "no figure may accompany the clarify"
    assert stub.calls == []


def test_the_brand_clarify_gate_is_exactly_the_rx_volume_family() -> None:
    """The gate is built from the lane's SSOT, not a hand-listed set: both the
    canonical ids and their patient-panel twins. Today's KPI recognizer resolves
    every volume alias to a CANONICAL id, so the panel half is not reachable
    through Branch A yet -- it is pinned here so a recognizer that later learns
    the panel names inherits the rule instead of the defect."""
    from src.agents.orchestrator.nodes.kpi_clarify import BRAND_CLARIFY_KPI_IDS
    from src.kpi.volume_family import CANONICAL_TO_PANEL

    expected = frozenset(CANONICAL_TO_PANEL) | frozenset(CANONICAL_TO_PANEL.values())
    assert BRAND_CLARIFY_KPI_IDS == expected
    assert len(expected) == 8
    # The boundary: brand-filterable KPIs outside the volume family keep their
    # pre-existing behaviour (see the Conversion Rate test above).
    assert "WS3-BI-009" not in BRAND_CLARIFY_KPI_IDS


# --------------------------------------------------------------------------
# #2114 codex r2 HIGH — the MIXED scope: a brand NAMED plus a different brand's
# INDICATION. `brand_from_text` runs the indication pass only when no brand is
# named (#1356 precedence), so "NBRx for Kisqali and PNH" bound Kisqali, never
# reached the r1 gate (which fired only when `brand` was None), and answered a
# two-brand ask with a Kisqali-only figure. The clarify must therefore key off
# the STRUCTURED brand -- what a caller explicitly decided -- not off whatever
# the text scan happened to bind.
# --------------------------------------------------------------------------

MIXED_SCOPE_QUERY = "What is NBRx for Kisqali and PNH?"


@pytest.mark.parametrize(
    "query,kpi_id",
    [
        ("What is NBRx for Kisqali and PNH?", "WS3-BI-007"),
        ("What is NBRx for PNH and Kisqali?", "WS3-BI-007"),
        ("What is TRx for Kisqali and PNH?", "WS3-BI-005"),
        ("What is NRx for Kisqali and CSU?", "WS3-BI-006"),
        # Indication first, brand second -- the scan is order-free.
        ("What is TRx for urticaria and Kisqali?", "WS3-BI-005"),
    ],
)
def test_a_named_brand_plus_another_brands_indication_clarifies(monkeypatch, query, kpi_id) -> None:
    """Both candidates must be named in the question, no figure may be bound,
    and the engine must never be asked -- even though the text scan bound a
    single brand and the r1 gate would have waved it through."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(query), _dispatch())

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["analysis_type"] == "kpi_lookup_clarification"
    assert payload["kpi_id"] == kpi_id
    assert payload["needs_clarification"] is True
    assert "value" not in payload
    assert len(payload["ambiguous_brands"]) == 2
    for brand in payload["ambiguous_brands"]:
        assert brand in payload["key_findings"][0], payload["key_findings"]
    assert stub.calls == []


def test_the_mixed_scope_clarify_names_the_indication_brand(monkeypatch) -> None:
    """PNH -> Fabhalta, so the question must offer Fabhalta as the alternative
    -- naming only Kisqali would be the same wrong-scope answer in question
    form."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(MIXED_SCOPE_QUERY), _dispatch())

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["ambiguous_brands"] == ["Kisqali", "Fabhalta"]
    assert "Fabhalta" in payload["key_findings"][0]
    assert stub.calls == []


def test_a_brand_beside_its_own_indication_still_computes(monkeypatch) -> None:
    """The measured decision: both routes grounding the SAME brand is ONE
    scope, so the commonest clinical phrasing keeps its answer."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is TRx for Kisqali in HR+ breast cancer?"), _dispatch()
    )

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "kpi_lookup"
    assert stub.calls == [("WS3-BI-005", {"brand": "Kisqali"})]


def test_a_structured_brand_wins_over_a_mixed_scope_ask(monkeypatch) -> None:
    """The gate keys off the STRUCTURED brand: an explicit user_context brand
    is a decision already taken, so even a mixed-scope ask is honoured."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    agent_input = _agent_input(MIXED_SCOPE_QUERY)
    agent_input["user_context"] = {"brand": "Fabhalta"}

    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "kpi_lookup"
    assert stub.calls == [("WS3-BI-007", {"brand": "Fabhalta"})]


def test_a_structured_entity_brand_also_wins_over_a_mixed_scope_ask(monkeypatch) -> None:
    """Same for the NLP entities channel, the other structured source."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    agent_input = _agent_input(MIXED_SCOPE_QUERY)
    agent_input["parsed_query"] = {"entities": [{"type": "brand", "value": "Kisqali"}]}

    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "kpi_lookup"
    assert stub.calls == [("WS3-BI-007", {"brand": "Kisqali"})]


def test_a_non_volume_kpi_refuses_the_mixed_scope_too(monkeypatch) -> None:
    """The mixed shape: "Kisqali and PNH" BINDS Kisqali by name while grounding a
    second brand (Fabhalta) by indication. A volume KPI asks which; any other KPI must
    REFUSE -- answering with a Kisqali-only figure drops half the ask (codex iter9 HIGH;
    the same fail-open existed on main at 4b719abe8, measured, and was pinned here as
    'not this lane's' until the guard it lives in became this lane's)."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](
        _agent_input("What is the conversion rate for Kisqali and PNH?"), _dispatch()
    )

    assert not isinstance(resolved, dict), resolved
    assert stub.calls == []


# --------------------------------------------------------------------------
# #2114 codex r3 HIGH — a BLANK structured brand is not a decision. The gate
# skips the clarify when a structured source supplied a brand, and
# `_structured_brand` mapped "" to None but returned " " and "\t\n" unchanged,
# so a whitespace-only `user_context.brand` re-opened the two-scope leak 10c
# closed: the ask reached the calculator and was answered for one brand.
#
# Measured 2026-09-15, both ingresses: the typed-entities path was ALREADY
# correct -- it dropped every blank form before the chooser saw it. Only the
# `user_context` path had the gap. These tests pin both so the two ingresses
# can never drift apart again. (r5 later replaced `_entity_value` with
# `_entity_values`, which offers EVERY entity to `first_named_scope`; blanks
# are still absent, now because the chooser's `names_something` drops them
# rather than because the reader filtered on `.strip()`.)
# --------------------------------------------------------------------------

BLANK_BRANDS = ["", " ", "\t\n", "   \t "]


@pytest.mark.parametrize("blank", BLANK_BRANDS)
def test_a_blank_user_context_brand_is_not_a_decision(monkeypatch, blank) -> None:
    """A brand of whitespace names nobody, so it must not suppress the clarify:
    the mixed-scope ask still asks, names both candidates, binds no figure and
    makes ZERO engine calls."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    agent_input = _agent_input(MIXED_SCOPE_QUERY)
    agent_input["user_context"] = {"brand": blank}

    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch())

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["analysis_type"] == "kpi_lookup_clarification"
    assert payload["ambiguous_brands"] == ["Kisqali", "Fabhalta"]
    assert "value" not in payload
    for brand in ("Kisqali", "Fabhalta"):
        assert brand in payload["key_findings"][0], payload["key_findings"]
    assert stub.calls == []


@pytest.mark.parametrize("blank", BLANK_BRANDS)
def test_a_blank_typed_entity_brand_is_not_a_decision(monkeypatch, blank) -> None:
    """The other structured ingress, held to the SAME rule. This path was
    already correct -- a blank entity names nobody and is dropped before any
    decision; pinned here so a future edit cannot make the two ingresses
    disagree."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    agent_input = _agent_input(MIXED_SCOPE_QUERY)
    agent_input["parsed_query"] = {"entities": [{"type": "brand", "value": blank}]}

    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch())

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["analysis_type"] == "kpi_lookup_clarification"
    assert payload["ambiguous_brands"] == ["Kisqali", "Fabhalta"]
    assert stub.calls == []


@pytest.mark.parametrize("blank", BLANK_BRANDS)
def test_blank_structured_brands_agree_across_both_ingresses(blank) -> None:
    """The seam itself, stated once: every blank form reads as 'nobody decided'
    whichever structured source carried it."""
    from_ctx = {"query": MIXED_SCOPE_QUERY, "user_context": {"brand": blank}}
    from_entities = {
        "query": MIXED_SCOPE_QUERY,
        "parsed_query": {"entities": [{"type": "brand", "value": blank}]},
    }
    assert disp._structured_brand(from_ctx) is None, blank
    assert disp._structured_brand(from_entities) is None, blank


def test_a_blank_context_brand_falls_through_to_the_text_scan() -> None:
    """The consequence of fixing the seam, pinned deliberately: a blank context
    brand now behaves EXACTLY like an absent one for cohort resolution too --
    which is how "" has always behaved -- so `_extract_brand_region` reaches the
    #1351 text scan instead of handing ' ' to a case-sensitive brand predicate
    that can match no row."""
    blank_ctx = {
        "query": MIXED_SCOPE_QUERY,
        "user_context": {"brand": " "},
        "parsed_query": {"entities": []},
    }
    absent_ctx = {
        "query": MIXED_SCOPE_QUERY,
        "user_context": {},
        "parsed_query": {"entities": []},
    }
    assert disp._extract_brand_region(blank_ctx) == disp._extract_brand_region(absent_ctx)
    assert disp._extract_brand_region(blank_ctx) == ("Kisqali", None)

    brandless = {"query": "What is NBRx?", "user_context": {"brand": "\t"}, "parsed_query": {}}
    assert disp._extract_brand_region(brandless) == (None, None)


def test_a_real_structured_brand_is_still_a_decision(monkeypatch) -> None:
    """The guard must not over-reach: a genuine brand still wins the mixed-scope
    ask, exactly as the 10c pin requires."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    agent_input = _agent_input(MIXED_SCOPE_QUERY)
    agent_input["user_context"] = {"brand": "Fabhalta"}

    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "kpi_lookup"
    assert stub.calls == [("WS3-BI-007", {"brand": "Fabhalta"})]


# --------------------------------------------------------------------------
# #2114 codex r3 — two HIGHs, one root cause each.
#
# HIGH-1: `\bhr\+\b` could not match a standalone "HR+", so Kisqali's indication
# grounded nothing in "NBRx for HR+ and PNH?" and the ask was answered with a
# Fabhalta-only figure. Fixed in the alias table (see TestHrPlusIndicationAlias).
#
# HIGH-2: U+200B / U+FEFF survive `.strip()`, so they stayed truthy on the
# `user_context` path while `entities` returned None — the two ingresses 10d
# unified had drifted apart again. That was the FOURTH blank shape to leak in
# four rounds, so the predicate changed rather than the patch: `_structured_brand`
# now asks the POSITIVE question "is this a brand the substrate recognises?"
# instead of the open-ended negative "is this not blank".
# --------------------------------------------------------------------------

HR_PLUS_QUERIES = [
    ("What is NBRx for HR+ and PNH?", "WS3-BI-007"),
    ("What is NBRx for PNH and HR+?", "WS3-BI-007"),
    ("What is TRx for HR+ and PNH?", "WS3-BI-005"),
    ("What is NRx for HR+ and CSU?", "WS3-BI-006"),
]

INVISIBLE_BRANDS = ["​", "﻿", "\xa0"]


@pytest.mark.parametrize("query,kpi_id", HR_PLUS_QUERIES)
def test_standalone_hr_plus_beside_another_indication_clarifies(monkeypatch, query, kpi_id) -> None:
    """HIGH-1 at the consumer: both brands named, no figure, no engine call."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))

    resolved = disp.INPUT_RESOLVERS["explainer"](_agent_input(query), _dispatch())

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["analysis_type"] == "kpi_lookup_clarification"
    assert payload["kpi_id"] == kpi_id
    assert "value" not in payload
    assert set(payload["ambiguous_brands"]) == {"Kisqali", "Fabhalta"} or set(
        payload["ambiguous_brands"]
    ) == {"Kisqali", "Remibrutinib"}, payload["ambiguous_brands"]
    assert stub.calls == []


@pytest.mark.parametrize("invisible", INVISIBLE_BRANDS)
def test_an_invisible_character_brand_is_not_a_decision(monkeypatch, invisible) -> None:
    """HIGH-2, `user_context` ingress: a zero-width or non-breaking space names
    nobody, so it must not suppress the clarify."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    agent_input = _agent_input(MIXED_SCOPE_QUERY)
    agent_input["user_context"] = {"brand": invisible}

    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch())

    assert isinstance(resolved, dict), resolved
    payload = resolved["analysis_results"][0]
    assert payload["analysis_type"] == "kpi_lookup_clarification"
    assert payload["ambiguous_brands"] == ["Kisqali", "Fabhalta"]
    assert "value" not in payload
    assert stub.calls == []


@pytest.mark.parametrize("invisible", INVISIBLE_BRANDS)
def test_an_invisible_character_entity_is_not_a_decision(monkeypatch, invisible) -> None:
    """HIGH-2, the other ingress, held to the same rule so the two cannot drift."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    agent_input = _agent_input(MIXED_SCOPE_QUERY)
    agent_input["parsed_query"] = {"entities": [{"type": "brand", "value": invisible}]}

    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "kpi_lookup_clarification"
    assert stub.calls == []


@pytest.mark.parametrize("padded", [" Kisqali ", "kisqali", "  KISQALI\t"])
def test_a_padded_or_miscased_brand_resolves_to_a_real_scope(monkeypatch, padded) -> None:
    """The positive predicate normalises as a side effect, which closes the
    padded-brand defect: the calculator receives 'Kisqali', not a string the
    case-sensitive `brand::text = $1` predicate can never match."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    agent_input = _agent_input(MIXED_SCOPE_QUERY)
    agent_input["user_context"] = {"brand": padded}

    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "kpi_lookup"
    assert stub.calls == [("WS3-BI-007", {"brand": "Kisqali"})]


@pytest.mark.parametrize("real_enum_brand", ["competitor", "other", "COMPETITOR"])
def test_a_substrate_brand_outside_the_commercial_three_is_still_a_decision(
    monkeypatch, real_enum_brand
) -> None:
    """`brand_type` carries five labels, not three: `competitor` and `other` are
    real values with real rows (patient_journeys carries competitor RWD). The
    predicate resolves against that FULL enum, so a competitor-scoped ask keeps
    its scope instead of silently widening -- which would be this lane's own
    defect class reintroduced by the fix for it."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    agent_input = _agent_input(MIXED_SCOPE_QUERY)
    agent_input["user_context"] = {"brand": real_enum_brand}

    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "kpi_lookup"
    assert stub.calls == [("WS3-BI-007", {"brand": real_enum_brand.lower()})]


@pytest.mark.parametrize("unserveable", ["Xolair", "NotABrand", "12345", "-"])
def test_an_unrecognised_brand_string_IS_a_decision(monkeypatch, unserveable) -> None:
    """REVERSED from the r3 version of this test, deliberately (r4 HIGH-2).

    r3 treated a value outside the `brand_type` enum as nobody-decided, so the
    ambiguous ask re-asked. That conflated two different facts and, at the other
    consumer, erased the value entirely -- cohort resolution then read "no brand
    specified" and widened to every patient. A caller who sets `brand='Xolair'`
    HAS decided; the substrate simply cannot serve it. So the clarify does not
    re-ask, the value travels unchanged, and the honest failure happens where the
    scope is applied rather than being papered over with a question.
    """
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    agent_input = _agent_input(MIXED_SCOPE_QUERY)
    agent_input["user_context"] = {"brand": unserveable}

    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "kpi_lookup"
    assert stub.calls == [("WS3-BI-007", {"brand": unserveable})], (
        "the unrecognised scope must reach the calculator unchanged, not be erased"
    )


def test_an_unresolvable_entity_does_not_hide_a_valid_context_brand(monkeypatch) -> None:
    """r4 HIGH-1 at the clarify gate: validation precedes source selection, so a
    junk `entities` value no longer masks a servable `user_context` one."""
    stub = _install_calculator(monkeypatch, _StubCalculator(_kpi_result()))
    agent_input = _agent_input(MIXED_SCOPE_QUERY)
    agent_input["parsed_query"] = {"entities": [{"type": "brand", "value": "NotABrand"}]}
    agent_input["user_context"] = {"brand": "competitor"}

    resolved = disp.INPUT_RESOLVERS["explainer"](agent_input, _dispatch())

    assert isinstance(resolved, dict), resolved
    assert resolved["analysis_results"][0]["analysis_type"] == "kpi_lookup"
    assert stub.calls == [("WS3-BI-007", {"brand": "competitor"})]
