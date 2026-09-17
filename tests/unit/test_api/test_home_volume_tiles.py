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

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

#: ``database/migrations`` — the SQL this repo SHIPS. Not the deployed registry.
_MIGRATIONS = Path(__file__).resolve().parents[3] / "database" / "migrations"


def _shipped_statement(query_id: str) -> str:
    """The SQL body one APPLIED migration registers for ``query_id``.

    ``rollback_*.sql`` files are excluded — they exist to undo a migration and
    are not part of the applied sequence, so counting them would report every
    143 statement as defined twice.

    Exactly one applied migration must define the id, and that is the load
    bearing part rather than a tidiness check. Reading "the first file that
    mentions it" is the supersession trap this repo has already been bitten by
    (a files-grep once reported 8 live case-sensitive predicates where the live
    registry had 0, because a later migration had replaced them). If a future
    migration ever redefines one of these ids, this raises and names both files
    instead of silently asserting against the superseded body.
    """
    pattern = re.compile(r"\('" + re.escape(query_id) + r"', \$kpi\$(.*?)\$kpi\$", re.S)
    hits = [
        (path.name, match.group(1))
        for path in sorted(_MIGRATIONS.glob("*.sql"))
        if not path.name.startswith("rollback_")
        for match in [pattern.search(path.read_text())]
        if match
    ]
    assert len(hits) == 1, (
        f"{query_id} is defined by {[name for name, _ in hits]}, expected exactly 1 — "
        "the shipped SQL for this id is ambiguous from the files, so a Home tile may "
        "not be pointed at it (measured when a tile was re-pointed back to "
        "business_impact_trx, which 044/066/089 all touch)"
    )
    return hits[0][1]


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


class TestTheVolumeTilesShippedSqlReadsBusinessMetrics:
    """HERMETIC — no database, so this is the half of the substrate claim that
    actually runs in CI.

    ⚠ IT PROVES THE SQL WE SHIP, NEVER THE REGISTRY THAT IS DEPLOYED. A
    migration file is the source of the deployed statement; it is not evidence
    that the statement was applied. Measured 2026-09-17: the live registry on
    this box holds ZERO ``canonical_volume%`` ids while these files have shipped
    them for the whole lane. Only
    ``test_trx_substrate_fence_1640.py::TestTheTileSubstratesMatchTheLiveRegistry``
    can witness the deployed side — and that class does not run in CI, which is
    why this one exists rather than replacing it.
    """

    def test_each_volume_tile_runs_a_business_metrics_statement(self):
        """The regression this catches: re-pointing a volume tile back at a
        patient-panel statement (or at any statement over ``treatment_events``)
        while every other test still passes, because nothing else in CI reads
        the SQL behind the id."""
        from src.api.routes.copilotkit import _KPI_SUMMARY_QUERIES
        from src.kpi.measure_basis import tables_in_sql

        for field in ("trx_volume", "nrx_volume", "market_share", "patient_starts"):
            query_id = _KPI_SUMMARY_QUERIES[field][0]
            assert tables_in_sql(_shipped_statement(query_id)) == ["business_metrics"], (
                field,
                query_id,
            )

    def test_the_event_tiles_are_not_canonical_statements(self):
        """The other half, asserted at the id rather than at the substrate.

        Deriving hcp_reach / conversion substrate from the files the way the
        four above are derived would be the overclaim: measured, each is touched
        by THREE migrations (044/063, 066's twins, 089's frontier anchoring), so
        a file-order rule would be picking one of several bodies and calling it
        the answer. The live class above is where their substrate is checked.
        """
        from src.api.routes.copilotkit import _KPI_SUMMARY_QUERIES

        for field in ("hcp_reach", "conversion_rate"):
            assert not _KPI_SUMMARY_QUERIES[field][0].startswith("canonical_volume"), field

    def test_the_statement_finder_has_teeth(self):
        """Positive control. Without this, a finder that quietly returned an
        empty string for everything would make the assertion above pass against
        ``tables_in_sql("") == []``... or fail confusingly; either way the test
        would not be measuring what it claims. So: an unknown id must RAISE, and
        a known one must come back with a real body.
        """
        with pytest.raises(AssertionError):
            _shipped_statement("canonical_volume_no_such_statement")
        body = _shipped_statement("canonical_volume_trx")
        assert "FROM business_metrics" in body and len(body) > 200


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
