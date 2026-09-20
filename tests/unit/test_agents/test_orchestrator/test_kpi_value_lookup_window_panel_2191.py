"""Regressions for explicit invalid windows and patient-panel count asks (#2191)."""

from typing import Any

import pytest

from src.agents.orchestrator.nodes.dispatcher import _kpi_lookup_evidence, _window_from_query
from src.agents.orchestrator.nodes.intent_classifier import KPI_VALUE_LOOKUP_RE
from src.kpi.models import KPIResult, KPIStatus
from src.services.time_window import WindowParseError


class _RecordingCalculator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def calculate(self, kpi_id: str, context: dict[str, Any]) -> KPIResult:
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


@pytest.mark.parametrize(
    "query",
    [
        "What is 2026-05-01 to 2026-01-01 TRx?",
        "What is 2026-02-30 to 2026-03-01 TRx?",
    ],
)
def test_an_explicit_invalid_iso_window_fails_before_calculation(
    query: str, calculator: _RecordingCalculator
) -> None:
    """A requested-but-invalid period must not collapse to the default period."""
    assert KPI_VALUE_LOOKUP_RE.search(query), "exercise the routed KPI lookup path"
    assert _kpi_lookup_evidence({"query": query}) is None
    assert calculator.calls == []


@pytest.mark.parametrize(
    "query",
    [
        "What is 2026-05-01 to 2026-01-01 TRx?",
        "What is March-Jan 2025 TRx?",
    ],
)
def test_window_extraction_distinguishes_an_invalid_range_from_no_window(query: str) -> None:
    with pytest.raises(WindowParseError):
        _window_from_query(query)


def test_a_malformed_iso_window_after_the_kpi_also_fails_closed(
    calculator: _RecordingCalculator,
) -> None:
    query = "What is TRx for 2026-02-30 to 2026-03-01?"
    assert KPI_VALUE_LOOKUP_RE.search(query)
    assert _kpi_lookup_evidence({"query": query}) is None
    assert calculator.calls == []


@pytest.mark.parametrize(
    "query",
    [
        "How many patient panel TRx were recorded?",
        "How many patient-panel TRx were recorded?",
    ],
)
def test_how_many_patient_panel_trx_reaches_the_panel_kpi(
    query: str,
    calculator: _RecordingCalculator,
) -> None:
    """The recognized compound reaches routing and binds the panel KPI."""
    assert KPI_VALUE_LOOKUP_RE.search(query)
    evidence = _kpi_lookup_evidence({"query": query})
    assert evidence
    assert evidence[0]["kpi_id"] == "WS3-BI-011"
    assert [call[0] for call in calculator.calls] == ["WS3-BI-011"]


@pytest.mark.parametrize(
    "query",
    [
        "How many patients received TRx?",
        "How many high-risk patients received TRx?",
        "How many patient panel patients received TRx?",
        "How many patient panel TRx patients were counted?",
    ],
)
def test_ordinary_entity_count_asks_remain_outside_the_scalar_path(
    query: str, calculator: _RecordingCalculator
) -> None:
    assert not KPI_VALUE_LOOKUP_RE.search(query)
    assert _kpi_lookup_evidence({"query": query}) is None
    assert calculator.calls == []
