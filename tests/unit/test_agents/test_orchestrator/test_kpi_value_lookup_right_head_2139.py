"""Regression boundary for unsupported KPI value-lookup tails (#2139/#2141)."""

from typing import Any, Dict, List

import pytest

from src.agents.orchestrator.nodes.dispatcher import (
    _kpi_lookup_evidence,
    _window_from_query,
)
from src.kpi.models import KPIResult, KPIStatus


class _RecordingCalculator:
    """The real consumer seam, with calls visible even when evidence is absent."""

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
def calculator(monkeypatch) -> _RecordingCalculator:
    recorder = _RecordingCalculator()
    monkeypatch.setattr("src.api.routes.kpi.get_kpi_calculator", lambda: recorder)
    return recorder


@pytest.mark.parametrize(
    "query",
    [
        "What is TRx cost?",
        "What is NRx panel cost?",
        "What is NRx panel accuracy?",
        "What is NRx panel oncology?",
        "What is TRx patients?",
        "What is TRx segment?",
        "What is TRx therapy_line?",
        # Punctuation must not hide the first tail word from the guard.
        "What is TRx: cost?",
        "What is TRx, cost?",
        "What is TRx/cost?",
        "What is NRx panel's price?",
        "What is TRx's cost?",
        "What is NRx panel -- benchmark?",
        # A period/scope token defers the decision; it cannot license a noun.
        "What is TRx Q3 cost?",
        "What is TRx last two quarters forecast?",
        "What is TRx Kisqali target?",
        "What is TRx cost Kisqali?",
        "What is TRx Kisqali tier?",
        "What is TRx west region accuracy?",
        "What is TRx volume cost?",
        # Every repeated occurrence is checked before identical mentions mask.
        "What is TRx and TRx cost?",
        "What is TRx and the cost of TRx?",
        # A function word cannot make its unresolved object disappear (#2141).
        "What is TRx among new patients?",
        "What is TRx within the cohort?",
        "What is TRx in oncology?",
        # If upstream routing ever drifts, the scalar evidence seam must still
        # refuse this decomposition instead of returning a national number.
        "What is NRx panel by segment?",
        "What is TRx by severity?",
        "What is TRx among Kisqali patients?",
        "What is TRx across brands?",
        "What is TRx at the HCP level?",
        "What is TRx with high adherence?",
        "What is TRx per brand?",
        "What is TRx under the new plan?",
        "What is TRx compared with baseline?",
        "What is TRx when adherence is low?",
        "What is TRx if adherence drops?",
        "What is TRx for Kisqali or Fabhalta?",
        "What is TRx in west and south?",
        "What is TRx in west oncology?",
        "What is TRx Q3 for the last 30 days?",
        "What is TRx for last quarter and last year?",
        "What is TRx without Kisqali?",
        "What is TRx not Kisqali?",
        "What is TRx except west?",
        "What is TRx or Kisqali?",
        "What is TRx compared to last quarter?",
        "What is TRx relative to west?",
        "What is TRx versus last quarter?",
        "What is TRx if Kisqali?",
        "What is market share compared to last quarter?",
        # Coordinators and punctuation cannot terminate the safety walk while
        # a continuation still changes the requested quantity.
        "What is TRx and its cost?",
        "What is TRx or its price?",
        "What is TRx; specifically its cost?",
        "What is TRx? I mean its cost",
    ],
)
def test_unsupported_right_heads_fail_closed_before_calculation(query, calculator) -> None:
    assert _kpi_lookup_evidence({"query": query}) is None, query
    assert calculator.calls == [], f"{query!r} reached the calculator"


def test_a_two_brand_volume_ask_asks_which_brand_instead_of_calculating(calculator) -> None:
    """#2114 owner ruling (2026-09-15): a Rx-volume ask grounding more than one brand
    ASKS which one. It still never reaches the calculator -- this guard's purpose --
    but the answer is the clarify question, not a refusal. (Other KPIs, and a
    disjunction like "Kisqali or Fabhalta", still refuse above.)"""
    evidence = _kpi_lookup_evidence({"query": "What is TRx for Kisqali and Fabhalta?"})
    assert evidence, "the two-brand volume ask neither clarified nor answered"
    assert [e["analysis_type"] for e in evidence] == ["kpi_lookup_clarification"], evidence
    assert calculator.calls == []


@pytest.mark.parametrize(
    ("query", "expected_id", "expected_context"),
    [
        ("What is TRx?", "WS3-BI-005", {}),
        # #2114: "NRx panel" is its own KPI (the patient-panel NRx event count,
        # WS3-BI-012), no longer a qualifier on canonical NRx (WS3-BI-006).
        ("What is NRx panel for Kisqali?", "WS3-BI-012", {"brand": "Kisqali"}),
        (
            "What is NRx panel in the west region?",
            "WS3-BI-012",
            {"region": "west"},
        ),
        ("What is TRx for Kisqali?", "WS3-BI-005", {"brand": "Kisqali"}),
        ("What is the current TRx volume?", "WS3-BI-005", {}),
        ("What is TRx value?", "WS3-BI-005", {}),
        # Repeated aliases are an appositive restatement, not another quantity.
        ("What is TRx, the total prescriptions, for Kisqali?", "WS3-BI-005", {"brand": "Kisqali"}),
        ("What is TRx, the total prescriptions?", "WS3-BI-005", {}),
        # Bare scope is accepted only when the platform's resolver binds it.
        ("What is TRx Kisqali?", "WS3-BI-005", {"brand": "Kisqali"}),
        ("What is TRx west region?", "WS3-BI-005", {"region": "west"}),
        ("What is TRx? Thanks.", "WS3-BI-005", {}),
    ],
)
def test_ordinary_value_lookups_still_calculate(
    query, expected_id, expected_context, calculator
) -> None:
    evidence = _kpi_lookup_evidence({"query": query})

    assert evidence, query
    assert calculator.calls == [(expected_id, expected_context)]


@pytest.mark.parametrize(
    "query",
    [
        "What is TRx in the west region?",
        "What is TRx for the last 30 days?",
        "What is TRx in the last quarter?",
        "What is TRx for last quarter and last quarter?",
    ],
)
def test_closed_class_right_heads_do_not_over_refuse(query, calculator) -> None:
    """The guard decides the bare compound head, not a prepositional object."""
    assert _kpi_lookup_evidence({"query": query}), query
    assert len(calculator.calls) == 1


def test_bare_year_is_not_silently_treated_as_a_supported_window(calculator) -> None:
    """The consumer intentionally does not bind one-token years.

    ``_window_from_query`` omits single tokens so unrelated counts such as
    "top 2000 HCPs" cannot silently become a calendar-year scope.  The guard
    must therefore fail closed on a bare year rather than letting the KPI
    engine use its default window while implying that the year was honored.
    """
    query = "What is TRx 2025?"

    assert _window_from_query(query) is None
    assert _kpi_lookup_evidence({"query": query}) is None
    assert calculator.calls == []
