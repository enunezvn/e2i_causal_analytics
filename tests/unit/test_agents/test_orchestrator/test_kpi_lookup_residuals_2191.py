"""The two routing residuals left open when #2194 closed #2191.

Both strand a legitimate ask; neither ever served a wrong number, which is why
they were recorded rather than rushed. They share one seam — what the value-lookup
grammar admits before the metric — so they are fixed and guarded together.

1. ``How many panel TRx were recorded?`` — #2194 gave the trailing "were recorded"
   discourse a pass, but keyed it to the literal ``how many patient panel``. The
   bare ``panel TRx`` form resolves to the SAME patient-panel KPI, so the tail is
   just as much discourse there.

2. ``What is East Coast TRx?`` — #1572's whole point is that "East Coast" spans the
   northeast AND south census regions, so the ask must be ANSWERED WITH A QUESTION
   rather than a silent national figure. #2130's scope-word grammar was built from
   the canonical region phrases only, and "East Coast" is deliberately NOT one of
   them (the #1565 ruling keeps it out of the alias table). So the ask stopped
   routing at all and #1572's clarify became unreachable for its own canonical
   example — the user gets "Run an analysis first" instead of "which region?".
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.agents.orchestrator.nodes.dispatcher import _kpi_lookup_evidence
from src.agents.orchestrator.nodes.intent_classifier import KPI_VALUE_LOOKUP_RE
from src.kpi.models import KPIResult, KPIStatus
from src.services.query_entities import region_scan


class _RecordingCalculator:
    def __init__(self) -> None:
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    def calculate(self, kpi_id: str, context: Dict[str, Any]) -> KPIResult:
        self.calls.append((kpi_id, dict(context)))
        return KPIResult(
            kpi_id=kpi_id,
            value=123.0,
            status=KPIStatus.INFORMATIONAL,
            metadata={"context": {"data_through": "2026-08-31"}},
        )


@pytest.fixture
def calculator(monkeypatch: pytest.MonkeyPatch) -> _RecordingCalculator:
    recorder = _RecordingCalculator()
    monkeypatch.setattr("src.api.routes.kpi.get_kpi_calculator", lambda: recorder)
    return recorder


# --------------------------------------------------------------------------
# Residual 1 — the bare "panel <metric>" compound
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "kpi_id"),
    [
        ("How many panel TRx were recorded?", "WS3-BI-011"),
        ("How many panel NRx were recorded?", "WS3-BI-012"),
        ("How many panel NBRx were recorded?", "WS3-BI-013"),
        ("How many panel TRx was recorded?", "WS3-BI-011"),
    ],
)
def test_bare_panel_compound_completes_like_the_patient_panel_form(
    query: str, kpi_id: str, calculator: _RecordingCalculator
) -> None:
    evidence = _kpi_lookup_evidence({"query": query})
    assert evidence is not None, "the bare panel compound must bind like 'patient panel'"
    assert [call[0] for call in calculator.calls] == [kpi_id]


def test_the_patient_panel_form_still_binds(calculator: _RecordingCalculator) -> None:
    """#2194's own case must not regress while its gate is widened."""
    assert _kpi_lookup_evidence({"query": "How many patient panel TRx were recorded?"}) is not None
    assert [call[0] for call in calculator.calls] == ["WS3-BI-011"]


@pytest.mark.parametrize(
    "query",
    [
        # #2130's entity-count veto: the KPI is the AXIS, not the asked quantity.
        "How many people received new prescriptions?",
        "How many patients are in the high-severity segment?",
        "How many prescribers wrote TRx?",
        "How many pharmacies recorded TRx?",
    ],
)
def test_entity_counts_still_bind_nothing(query: str, calculator: _RecordingCalculator) -> None:
    """Widening the 'were recorded' pass must not reopen the entity-count veto."""
    assert _kpi_lookup_evidence({"query": query}) is None
    assert calculator.calls == []


# --------------------------------------------------------------------------
# Residual 2 — an ambiguous region phrase must reach #1572's clarify
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "What is East Coast TRx?",
        "What is East Coast NRx?",
        "Show me East Coast TRx.",
        "What is Eastern Seaboard TRx?",
    ],
)
def test_an_ambiguous_region_ask_routes_and_asks_which_region(
    query: str, calculator: _RecordingCalculator
) -> None:
    """It must ROUTE (else the clarify is unreachable) and then clarify, never bind."""
    assert KPI_VALUE_LOOKUP_RE.search(query), "the ask must reach the value-lookup path at all"
    assert region_scan(query).needs_clarification, "precondition: the phrase is the ambiguous kind"

    evidence = _kpi_lookup_evidence({"query": query})
    assert evidence is not None, "#1572: answer the ask with a question"
    assert evidence[0].get("analysis_type") == "kpi_lookup_clarification"
    assert calculator.calls == [], "an ambiguous region must never reach the calculator"


def test_the_clarify_names_the_phrase_the_user_typed(calculator: _RecordingCalculator) -> None:
    evidence = _kpi_lookup_evidence({"query": "What is East Coast TRx?"})
    assert evidence is not None
    blob = " ".join(str(value) for value in evidence[0].values()).lower()
    assert "east coast" in blob
    # The clarify must offer the census regions the phrase spans, or it is a dead
    # end that names no next step (cf. tests/unit/test_kpi/test_redirect_round_trip_2114.py:
    # when a refusal names a next step, following it has to actually work).
    assert "northeast" in blob and "south" in blob


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is West TRx?", {"region": "west"}),
        ("What is New England TRx?", {"region": "northeast"}),
        ("What is TRx?", {}),
    ],
)
def test_unambiguous_region_asks_still_bind(
    query: str, expected: Dict[str, Any], calculator: _RecordingCalculator
) -> None:
    """The new scope word must not disturb the phrases that already grounded."""
    assert _kpi_lookup_evidence({"query": query}) is not None
    assert len(calculator.calls) == 1
    kpi_id, context = calculator.calls[0]
    assert kpi_id == "WS3-BI-005"
    assert {key: context[key] for key in expected if key in context} == expected
    if not expected:
        assert "region" not in context


def test_a_multi_region_ask_still_clarifies_rather_than_binding(
    calculator: _RecordingCalculator,
) -> None:
    """#2191 point 2 must survive: naming two regions is still not a national ask."""
    evidence = _kpi_lookup_evidence({"query": "What is West and Midwest TRx?"})
    assert evidence is not None
    assert evidence[0].get("analysis_type") == "kpi_lookup_clarification"
    assert calculator.calls == []
