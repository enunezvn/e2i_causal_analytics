"""Home volume tiles read the canonical series and cite its complete month.

The Home landing tiles used to run the patient-panel ``business_impact_*``
statements for TRx / NRx / NBRx / TRx share. Those count ``treatment_events``
rows over a trailing window; the canonical figures live in ``business_metrics``
and are ~1,300x apart (#1640). This locks the re-point, and locks the fact that
the tiles now report a DIFFERENT period from the rest of the summary: the
canonical series' latest COMPLETE calendar month, not the event frontier.

⭐ The period must never be guessed. A failed canonical read leaves
``volume_period`` / ``volume_data_through`` None — a month label invented from
``CURRENT_DATE`` would be indistinguishable from a real one on the page.
"""

from types import SimpleNamespace

import pytest


class _Client:
    def __init__(self, rows_by_query, fail=False):
        self.rows_by_query, self.fail, self.calls = rows_by_query, fail, []

    def rpc(self, name, payload):
        assert name == "kpi_query"
        self.calls.append(payload)
        if self.fail:
            raise RuntimeError("rpc down")
        rows = self.rows_by_query.get(payload["query_id"], [])
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=rows))


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


def test_volume_tile_fields_cite_the_complete_month():
    from src.kpi.home_volume_summary import volume_tile_fields

    client = _Client(
        {
            "canonical_volume_trx": [
                {"trx": 784342.1, "data_month": "2026-08-01", "data_through": "2026-08-31"}
            ]
        }
    )
    assert volume_tile_fields(client, "2026-09-14") == {
        "data_through": "2026-09-14",
        "volume_data_through": "2026-08-31",
        "volume_period": "August 2026",
    }
    assert client.calls == [{"query_id": "canonical_volume_trx", "params": [None]}]


def test_volume_tile_fields_fail_soft_and_keep_the_event_frontier():
    from src.kpi.home_volume_summary import volume_tile_fields

    assert volume_tile_fields(_Client({}, fail=True), "2026-09-14") == {
        "data_through": "2026-09-14",
        "volume_data_through": None,
        "volume_period": None,
    }
    assert volume_tile_fields(None, None) == {
        "data_through": None,
        "volume_data_through": None,
        "volume_period": None,
    }


def test_an_empty_canonical_read_states_no_month_rather_than_todays():
    """The fail-soft path above covers a RAISING client. A reachable client that
    returns no rows (a fresh instance, or the M4 gate excluding every synthetic
    row) is the other way the period is unknown, and it must be just as silent:
    ``date.today()`` here would print a confident month for a tile showing None.
    """
    from src.kpi.home_volume_summary import volume_tile_fields

    assert volume_tile_fields(_Client({}), "2026-09-14") == {
        "data_through": "2026-09-14",
        "volume_data_through": None,
        "volume_period": None,
    }


def test_the_period_query_follows_the_synthetic_twin(monkeypatch):
    """Positive control on the resolve step. Under the demo flag every other tile
    swaps to its ``_include_synthetic`` twin; a period label read from the BASE id
    would disagree with the figures beside it — and would read as a real month.
    """
    from src.kpi.home_volume_summary import volume_tile_fields

    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "true")
    client = _Client(
        {
            "canonical_volume_trx_include_synthetic": [
                {"trx": 1.0, "data_month": "2026-07-01", "data_through": "2026-07-31"}
            ]
        }
    )
    fields = volume_tile_fields(client, "2026-09-14")
    assert client.calls[0]["query_id"] == "canonical_volume_trx_include_synthetic"
    assert fields["volume_period"] == "July 2026"


def test_volume_tiles_map_to_canonical_statements_and_the_rest_do_not():
    from src.api.routes.copilotkit import _KPI_SUMMARY_QUERIES

    assert _KPI_SUMMARY_QUERIES["trx_volume"] == ("canonical_volume_trx", "trx", True, True)
    assert _KPI_SUMMARY_QUERIES["nrx_volume"] == ("canonical_volume_nrx", "nrx", True, True)
    assert _KPI_SUMMARY_QUERIES["market_share"] == (
        "canonical_volume_trx_share",
        "share",
        True,
        False,
    )
    assert _KPI_SUMMARY_QUERIES["patient_starts"] == ("canonical_volume_nbrx", "nbrx", True, True)
    assert _KPI_SUMMARY_QUERIES["hcp_reach"][0] == "business_impact_hcp_reach"
    assert _KPI_SUMMARY_QUERIES["conversion_rate"][0] == "business_impact_conversion_rate"


def test_region_routes_a_volume_tile_to_the_canonical_region_statement():
    from src.api.routes.copilotkit import _kpi_summary_query

    assert _kpi_summary_query("canonical_volume_trx", "Kisqali", "west", True) == (
        "canonical_volume_trx_region",
        ["Kisqali", "west"],
    )


def test_the_summary_carries_the_canonical_period(monkeypatch):
    import src.api.dependencies.supabase_client as sc
    import src.api.routes.copilotkit as ck

    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "true")
    rows = {
        "canonical_volume_trx_include_synthetic": [
            {"trx": 784342.1, "data_month": "2026-08-01", "data_through": "2026-08-31"}
        ],
        "business_impact_data_through_include_synthetic": [{"data_through": "2026-09-14"}],
    }
    monkeypatch.setattr(sc, "get_supabase", lambda: _Client(rows))
    monkeypatch.setattr(ck, "query_substrates_cached", lambda ids: {})
    import asyncio

    out = asyncio.run(ck.get_kpi_summary("Kisqali"))
    assert out["metrics"]["trx_volume"] == 784342.1
    assert out["volume_period"] == "August 2026"
    assert out["volume_data_through"] == "2026-08-31"
    assert out["data_through"] == "2026-09-14"
    assert out["data_source"] == "synthetic"
