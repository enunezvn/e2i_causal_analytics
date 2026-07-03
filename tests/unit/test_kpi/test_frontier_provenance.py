"""data_through provenance from the frontier-anchored registry rows (089).

Migration 089 re-anchored the rolling-window KPI statements at their domain's
data frontier and added a ``data_through`` output column. The calculator must
surface that column into ``KPIResult.metadata["context"]`` so the chatbot
(`_kpi_result_to_response`) can cite the real as-of date instead of implying
wall-clock recency. Rows without the column (e.g. the explicit ``*_windowed*``
variants, pre-089 deployments) simply leave the key absent -- honest absence,
never a fabricated date.
"""

from typing import Any

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.registry import get_registry


class _FakeResponse:
    def __init__(self, rows: list[dict[str, Any]]):
        self.data = rows

    def execute(self):
        return self


class _FakeDB:
    """Minimal supabase-client stand-in: rpc(...).execute().data -> rows."""

    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def rpc(self, name: str, payload: dict[str, Any]) -> _FakeResponse:
        self.calls.append((name, payload))
        return _FakeResponse(self._rows)


@pytest.fixture(autouse=True)
def _no_synthetic(monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "0")
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "0")


def _nbrx_kpi():
    kpi = get_registry().get("WS3-BI-007")
    assert kpi is not None
    return kpi


@pytest.mark.unit
def test_calculator_surfaces_data_through_into_metadata_context():
    calc = BusinessImpactCalculator(
        db_client=_FakeDB([{"nbrx": 9, "data_through": "2025-04-23"}])
    )
    result = calc.calculate(_nbrx_kpi(), {"brand": "Kisqali"})
    assert result.error is None
    assert result.value == 9.0
    assert result.metadata["context"]["data_through"] == "2025-04-23"


@pytest.mark.unit
def test_calculator_honest_absence_when_row_has_no_data_through():
    # Pre-089 row shape / windowed variants: no data_through column -> no key.
    calc = BusinessImpactCalculator(db_client=_FakeDB([{"nbrx": 9}]))
    result = calc.calculate(_nbrx_kpi(), {"brand": "Kisqali"})
    assert result.error is None
    assert "data_through" not in result.metadata["context"]


@pytest.mark.unit
def test_conversion_rate_surfaces_data_through():
    calc = BusinessImpactCalculator(
        db_client=_FakeDB([{"conversion_rate": 0.36, "data_through": "2025-03-30"}])
    )
    kpi = get_registry().get("WS3-BI-009")
    assert kpi is not None
    result = calc.calculate(kpi, {})
    assert result.error is None
    assert result.metadata["context"]["data_through"] == "2025-03-30"
