"""Regression coverage for front-loaded KPI lookup scopes (#2191 points 1/2)."""

from typing import Any, Dict, List

import pytest

from src.agents.orchestrator.nodes.dispatcher import _kpi_lookup_evidence
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
def calculator(monkeypatch) -> _RecordingCalculator:
    recorder = _RecordingCalculator()
    monkeypatch.setattr("src.api.routes.kpi.get_kpi_calculator", lambda: recorder)
    return recorder


@pytest.mark.parametrize(
    "query",
    [
        "At each pharmacy, how many new prescriptions?",
        "By pharmacy, show me new prescriptions.",
        "For every prescriber, show me total prescriptions.",
        "For high-severity patients, show me TRx.",
        "Drivers for Kisqali: show me TRx.",
        "By pharmacy, can you show me TRx?",
    ],
)
def test_unsupported_front_loaded_scope_never_calculates(query, calculator) -> None:
    assert _kpi_lookup_evidence({"query": query}) is None
    assert calculator.calls == []


@pytest.mark.parametrize(
    ("query", "expected_context"),
    [
        ("For Kisqali, show me TRx.", {"brand": "Kisqali"}),
        ("For Kisqali, can you show me TRx?", {"brand": "Kisqali"}),
        ("In New England, show me TRx.", {"region": "northeast"}),
        (
            "For Kisqali in the West, show me TRx.",
            {"brand": "Kisqali", "region": "west"},
        ),
        (
            "For the last 30 days, show me TRx.",
            None,
        ),
    ],
)
def test_supported_front_loaded_scope_still_calculates(query, expected_context, calculator) -> None:
    assert _kpi_lookup_evidence({"query": query})
    assert len(calculator.calls) == 1
    kpi_id, context = calculator.calls[0]
    assert kpi_id == "WS3-BI-005"
    if expected_context is None:
        assert set(context) == {"window"}
        assert set(context["window"]) == {"start", "end"}
    else:
        assert context == expected_context


def test_region_scan_retains_every_distinct_canonical_region() -> None:
    scan = region_scan("What is New England and West TRx?")

    assert scan.region is None
    assert scan.grounded_regions == ("northeast", "west")
    assert scan.needs_clarification


def test_front_loaded_multi_region_ask_clarifies_without_calculating(calculator) -> None:
    evidence = _kpi_lookup_evidence({"query": "What is west and south TRx?"})

    assert evidence
    assert [item["analysis_type"] for item in evidence] == ["kpi_lookup_clarification"]
    assert calculator.calls == []


def test_long_front_loaded_multi_region_ask_at_least_fails_closed(calculator) -> None:
    """The router's physical-word budget can refuse this before clarification."""
    evidence = _kpi_lookup_evidence({"query": "What is New England and West TRx?"})

    assert evidence is None or evidence[0]["analysis_type"] == "kpi_lookup_clarification"
    assert calculator.calls == []
