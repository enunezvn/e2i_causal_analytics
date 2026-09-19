"""The chat KPI payload names the month a canonical headline covers (#2114 cert, 2026-09-19).

The canonical WS3-BI-005..008 statements (migration 143) return ``data_month`` beside
``data_through`` and the calculator stashes both into ``KPIResult.metadata``, but the chat
payload copied only ``data_through``. The live certification then answered "the most recent
complete calendar month, data through 2026-08-31" and never said which month. Home already
labels the same period ``volume_period`` ("August 2026"); the chat now carries the same label.

These run the REAL calculator on a fake registry row and the REAL response mapper, so the
seam under test is the handoff between them, not either side alone.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.api.routes.chatbot_tools import _kpi_result_to_response
from src.kpi.calculator import KPICalculator
from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.registry import get_registry


class _FakeResponse:
    def __init__(self, rows: list[dict[str, Any]]):
        self.data = rows

    def execute(self):
        return self


class _FakeDB:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def rpc(self, name: str, payload: dict[str, Any]) -> _FakeResponse:
        return _FakeResponse(self._rows)


@pytest.fixture(autouse=True)
def _no_synthetic(monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "0")
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "0")


def _payload(kpi_id: str, row: dict[str, Any], context: dict[str, Any] | None = None) -> dict:
    kpi = get_registry().get(kpi_id)
    assert kpi is not None
    ctx = dict(context or {"brand": "Kisqali"})
    result = BusinessImpactCalculator(db_client=_FakeDB([row])).calculate(kpi, ctx)
    assert result.error is None, result.error
    # The engine stamps window provenance one layer up; apply the real stamper.
    result = KPICalculator._stamp_window(result, kpi, ctx.get("window"))
    return _kpi_result_to_response(kpi, result, brand="Kisqali")


_TRX_ROW = {"trx": 784342.2, "data_month": "2026-08-01", "data_through": "2026-08-31"}


@pytest.mark.unit
def test_canonical_headline_names_its_month():
    payload = _payload("WS3-BI-005", _TRX_ROW)
    assert payload["reporting_month"] == "August 2026"
    assert payload["data_through"] == "2026-08-31"


@pytest.mark.unit
def test_default_window_note_carries_the_concrete_month():
    # The model quotes this note verbatim ("the most recent complete calendar month"),
    # so the month has to be IN it, not only in a sibling field.
    payload = _payload("WS3-BI-005", _TRX_ROW)
    assert payload["window_status"] == "default"
    assert "most recent complete calendar month" in payload["reporting_window"]
    assert "August 2026" in payload["reporting_window"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "kpi_id,key", [("WS3-BI-006", "nrx"), ("WS3-BI-007", "nbrx"), ("WS3-BI-008", "share")]
)
def test_every_canonical_kpi_names_its_month(kpi_id, key):
    row = {
        key: 0.2 if key == "share" else 1234.0,
        "data_month": "2026-08-01",
        "data_through": "2026-08-31",
    }
    assert _payload(kpi_id, row)["reporting_month"] == "August 2026"


@pytest.mark.unit
def test_date_object_from_the_driver_is_labelled_too():
    from datetime import date

    row = {"trx": 1.0, "data_month": date(2025, 12, 1), "data_through": date(2025, 12, 31)}
    assert _payload("WS3-BI-005", row)["reporting_month"] == "December 2025"


@pytest.mark.unit
def test_custom_window_gets_no_single_month_label():
    # A windowed call sums several months; its data_month is only the LAST one, so a
    # single-month label would misstate the period.
    row = {
        "trx": 5.0,
        "data_month": "2026-08-01",
        "data_through": "2026-08-31",
        "months_in_window": 3,
    }
    ctx = {"brand": "Kisqali", "window": {"start": "2026-06-01", "end": "2026-08-31"}}
    payload = _payload("WS3-BI-005", row, ctx)
    assert payload["window_status"] != "default"
    assert "reporting_month" not in payload


@pytest.mark.unit
def test_kpi_without_data_month_keeps_honest_absence():
    kpi = get_registry().get("WS3-BI-007")
    payload = _payload("WS3-BI-007", {"nbrx": 9.0, "data_through": "2026-08-31"})
    assert kpi is not None
    assert "reporting_month" not in payload
    assert "(" not in payload.get("reporting_window", "")
