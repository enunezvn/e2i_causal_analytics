"""WS3-BI-005..008 compute from the canonical business_metrics statements;
WS3-BI-011..014 keep the event-ledger statements (canonical TRx lane)."""

from typing import Any, Dict, List

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.registry import get_registry

ROW = {
    "trx": 800349.18,
    "nrx": 172926.76,
    "nbrx": 91234.5,
    "share": 0.6613,
    "data_month": "2026-08-01",
    "data_through": "2026-08-31",
}


class _Response:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data


class _Rpc:
    def __init__(self, data):
        self._data = data

    def execute(self):
        return _Response(self._data)


class _RecordingClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls: List[Dict[str, Any]] = []

    def rpc(self, name, payload):
        assert name == "kpi_query"
        self.calls.append(payload)
        return _Rpc(self.rows)


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


def _calc(kpi_id, context, rows=(ROW,)):
    client = _RecordingClient(list(rows))
    result = BusinessImpactCalculator(db_client=client).calculate(
        get_registry().get(kpi_id), dict(context)
    )
    return result, client.calls


def test_trx_headline_runs_the_canonical_statement():
    result, calls = _calc("WS3-BI-005", {"brand": "Kisqali"})
    assert calls == [{"query_id": "canonical_volume_trx", "params": ["Kisqali"]}]
    assert result.error is None and result.value == 800349.18
    assert result.metadata["context"]["data_through"] == "2026-08-31"
    assert result.metadata["context"]["data_month"] == "2026-08-01"


@pytest.mark.parametrize(
    "kpi_id,query_id,value",
    [
        ("WS3-BI-006", "canonical_volume_nrx", 172926.76),
        ("WS3-BI-007", "canonical_volume_nbrx", 91234.5),
    ],
)
def test_nrx_and_nbrx_run_their_canonical_statements(kpi_id, query_id, value):
    result, calls = _calc(kpi_id, {"brand": "Kisqali"})
    assert calls == [{"query_id": query_id, "params": ["Kisqali"]}]
    assert result.value == value


def test_portfolio_nbrx_is_defined_on_the_canonical_series():
    result, calls = _calc("WS3-BI-007", {})
    assert calls == [{"query_id": "canonical_volume_nbrx", "params": [None]}]
    assert result.value == 91234.5


def test_region_routes_to_the_region_statement_and_marks_region_routing():
    result, calls = _calc("WS3-BI-005", {"brand": "Kisqali", "region": "midwest"})
    assert calls == [{"query_id": "canonical_volume_trx_region", "params": ["Kisqali", "midwest"]}]
    assert result.metadata["context"]["_region_routed"] is True


def test_window_routes_to_the_windowed_statements():
    window = {"start": "2026-06-01", "end": "2026-08-31"}
    _, calls = _calc("WS3-BI-005", {"brand": "Kisqali", "window": window})
    assert calls == [
        {
            "query_id": "canonical_volume_trx_windowed",
            "params": ["Kisqali", "2026-06-01", "2026-08-31"],
        }
    ]
    _, calls = _calc("WS3-BI-005", {"brand": "Kisqali", "region": "west", "window": window})
    assert calls == [
        {
            "query_id": "canonical_volume_trx_windowed_region",
            "params": ["Kisqali", "west", "2026-06-01", "2026-08-31"],
        }
    ]


def test_the_showcase_flag_selects_the_synthetic_twins(monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "true")
    _, calls = _calc("WS3-BI-005", {"brand": "Kisqali"})
    assert calls[0]["query_id"] == "canonical_volume_trx_include_synthetic"
    _, calls = _calc("WS3-BI-005", {"brand": "Kisqali", "region": "west"})
    assert calls[0]["query_id"] == "canonical_volume_trx_region_include_synthetic"


@pytest.mark.parametrize(
    "axis,value",
    [
        ("segment", "high_severity"),
        ("therapy_line", "1"),
        ("biologic", "naive"),
        ("ige_tier", "high"),
    ],
)
@pytest.mark.parametrize(
    "kpi_id,panel_id",
    [
        ("WS3-BI-005", "WS3-BI-011"),
        ("WS3-BI-006", "WS3-BI-012"),
        ("WS3-BI-007", "WS3-BI-013"),
        ("WS3-BI-008", "WS3-BI-014"),
    ],
)
def test_patient_axes_are_refused_before_any_query_naming_the_panel(kpi_id, panel_id, axis, value):
    result, calls = _calc(kpi_id, {"brand": "Remibrutinib", axis: value})
    assert calls == []
    assert result.value is None
    assert panel_id in result.error and axis in result.error


def test_share_requires_a_brand_and_reads_the_share_key():
    result, calls = _calc("WS3-BI-008", {})
    assert calls == [] and "WS3-BI-008" in result.error and "brand" in result.error
    result, calls = _calc("WS3-BI-008", {"brand": "Kisqali"})
    assert calls == [{"query_id": "canonical_volume_trx_share", "params": ["Kisqali"]}]
    assert result.value == 0.6613


def test_a_null_month_fails_loud_rather_than_zero():
    result, _ = _calc(
        "WS3-BI-005",
        {"brand": "Kisqali"},
        rows=[{"trx": None, "data_month": None, "data_through": None}],
    )
    assert result.value is None and "WS3-BI-005 unavailable" in result.error


@pytest.mark.parametrize(
    "panel_id,query_id",
    [
        ("WS3-BI-011", "business_impact_trx"),
        ("WS3-BI-012", "business_impact_nrx"),
        ("WS3-BI-013", "business_impact_nbrx"),
        ("WS3-BI-014", "business_impact_trx_share"),
    ],
)
def test_panel_ids_keep_the_event_ledger_statements(panel_id, query_id):
    rows = [{"trx": 597, "nrx": 100, "nbrx": 50, "share": 0.33, "data_through": "2026-09-14"}]
    result, calls = _calc(panel_id, {"brand": "Kisqali"}, rows=rows)
    assert calls[0]["query_id"] == query_id
    assert result.error is None


def test_panel_errors_name_the_panel_id():
    result, _ = _calc("WS3-BI-011", {"brand": "Kisqali"}, rows=[])
    assert "WS3-BI-011" in result.error and "WS3-BI-005" not in result.error
