"""Regression boundary for unsupported KPI value-lookup tails (#2139/#2141)."""

from typing import Any, Dict, List

import pytest

from src.agents.orchestrator.nodes.dispatcher import _kpi_lookup_evidence
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
    ],
)
def test_unsupported_right_heads_fail_closed_before_calculation(query, calculator) -> None:
    assert _kpi_lookup_evidence({"query": query}) is None, query
    assert calculator.calls == [], f"{query!r} reached the calculator"


@pytest.mark.parametrize(
    ("query", "expected_id", "expected_context"),
    [
        ("What is TRx?", "WS3-BI-005", {}),
        ("What is NRx panel for Kisqali?", "WS3-BI-006", {"brand": "Kisqali"}),
        (
            "What is NRx panel in the west region?",
            "WS3-BI-006",
            {"region": "west"},
        ),
        ("What is TRx for Kisqali?", "WS3-BI-005", {"brand": "Kisqali"}),
        ("What is TRx compared with baseline?", "WS3-BI-005", {}),
        ("What is the current TRx volume?", "WS3-BI-005", {}),
        ("What is TRx value?", "WS3-BI-005", {}),
        # Repeated aliases are an appositive restatement, not another quantity.
        ("What is TRx, the total prescriptions, for Kisqali?", "WS3-BI-005", {"brand": "Kisqali"}),
        # Bare scope is accepted only when the platform's resolver binds it.
        ("What is TRx Kisqali?", "WS3-BI-005", {"brand": "Kisqali"}),
        ("What is TRx west region?", "WS3-BI-005", {"region": "west"}),
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
        "What is TRx by severity?",
        "What is TRx across brands?",
        "What is TRx among new patients?",
        "What is TRx at the HCP level?",
        "What is TRx with high adherence?",
        "What is TRx from Q1?",
        "What is TRx to date?",
        "What is TRx per brand?",
        "What is TRx within the cohort?",
        "What is TRx under the new plan?",
        "What is TRx between Q1 and Q2?",
        "What is TRx versus last quarter?",
        "What is TRx when adherence is low?",
        "What is TRx if adherence drops?",
    ],
)
def test_closed_class_right_heads_do_not_over_refuse(query, calculator) -> None:
    """The guard decides the bare compound head, not a prepositional object."""
    assert _kpi_lookup_evidence({"query": query}), query
    assert len(calculator.calls) == 1
