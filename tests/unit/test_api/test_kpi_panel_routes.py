"""Segmented history and the claims-lag nowcast are patient-panel (event) series.

The last two tests carry #2114's invariant — IF A REFUSAL NAMES A NEXT STEP,
FOLLOWING IT MUST SUCCEED — onto these two HTTP routes.
``tests/unit/test_kpi/test_redirect_round_trip_2114.py`` holds the same invariant
for the CALCULATOR surface (four dead ends shipped in one day, each one
well-formed text that led nowhere, each "covered" by a test that asserted the
wording and never took the step).
"""

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.routes.kpi import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_the_axis_and_nowcast_families_are_the_panel_ids():
    from src.kpi.nowcast.completion_factor import NOWCAST_KPI_QUERY_FAMILIES
    from src.kpi.segmented_history import SEGMENTED_KPI_QUERY_FAMILIES

    expected = {
        "WS3-BI-011": "business_impact_trx",
        "WS3-BI-012": "business_impact_nrx",
        "WS3-BI-013": "business_impact_nbrx",
    }
    assert SEGMENTED_KPI_QUERY_FAMILIES == expected
    assert NOWCAST_KPI_QUERY_FAMILIES == expected


@pytest.mark.parametrize(
    "canonical,panel",
    [("WS3-BI-005", "WS3-BI-011"), ("WS3-BI-006", "WS3-BI-012"), ("WS3-BI-007", "WS3-BI-013")],
)
def test_a_canonical_id_is_refused_with_its_panel_id_named(client, canonical, panel):
    nowcast = client.get(f"/api/kpis/{canonical}/history/nowcast")
    assert nowcast.status_code == 422 and panel in nowcast.json()["detail"]
    segmented = client.get(f"/api/kpis/{canonical}/history/segmented", params={"axis": "segment"})
    assert segmented.status_code == 422 and panel in segmented.json()["detail"]


def test_the_cohort_profiler_asks_the_panel_nrx():
    from src.agents.cohort_profiler import agent

    assert agent._NRX_KPI_ID == "WS3-BI-012"


@pytest.mark.parametrize("canonical", ["WS3-BI-005", "WS3-BI-006", "WS3-BI-007"])
def test_the_named_panel_id_actually_serves_the_route(client, canonical):
    """FOLLOW the hint, do not just match its text. A refusal that names a next
    step is only honest if that step succeeds: the id parsed OUT of the 422 detail
    must return 200 on the same route.
    """
    import re
    from unittest.mock import AsyncMock, patch

    detail = client.get(
        f"/api/kpis/{canonical}/history/segmented", params={"axis": "segment"}
    ).json()["detail"]
    named = re.search(r"use the patient-panel KPI (WS3-BI-\d+)\.", detail)
    assert named, f"no panel id named in: {detail}"

    rows = [
        {
            "month_start": "2026-01-01",
            "bucket": "low_severity",
            "value": 3,
            "data_min": "2026-01-01",
            "data_max": "2026-01-31",
        }
    ]
    # Hermetic: both the rows and the measure-basis lookup are DB reads. The
    # round trip under test is the ROUTE accepting the id it just named.
    with (
        patch("src.kpi.segmented_history.fetch_segmented_rows", new=AsyncMock(return_value=rows)),
        patch("src.kpi.measure_basis.registry_query_basis", new=AsyncMock(return_value=None)),
    ):
        followed = client.get(
            f"/api/kpis/{named.group(1)}/history/segmented", params={"axis": "segment"}
        )
    assert followed.status_code == 200, followed.text
    assert followed.json()["kpi_id"] == named.group(1)


#: Every registry KPI these two routes can be asked for that is NOT in their
#: family, so every 422 they can emit. Both share KPIs are here because dead end
#: 3 came from a share, and WS3-BI-010 because dead end 3 was in the GENERIC path.
_REFUSED_KPI_IDS = (
    "WS3-BI-005",
    "WS3-BI-006",
    "WS3-BI-007",
    "WS3-BI-008",
    "WS3-BI-010",
    "WS3-BI-014",
)


@pytest.mark.parametrize("route", ["segmented", "nowcast"])
@pytest.mark.parametrize("kpi_id", _REFUSED_KPI_IDS)
def test_every_kpi_a_422_names_is_one_this_route_serves(client, route, kpi_id):
    """THE ROUND TRIP, not the wording: read every id OUT of the detail and require
    it to be in the family this route serves.

    Reading the ids out of the message rather than asserting literals is what makes
    this a guard and not a restatement — it follows whatever the code actually said,
    so it keeps passing when a destination legitimately changes and fails the moment
    a refusal names somewhere that refuses the same request. #2114 shipped four dead
    ends in one day, every one of them covered by a test that asserted the wording
    and never took the step; this is that invariant for these two routes.
    """
    from src.kpi.nowcast.completion_factor import NOWCAST_KPI_QUERY_FAMILIES
    from src.kpi.segmented_history import SEGMENTED_KPI_QUERY_FAMILIES

    family = SEGMENTED_KPI_QUERY_FAMILIES if route == "segmented" else NOWCAST_KPI_QUERY_FAMILIES
    params = {"axis": "segment"} if route == "segmented" else {}
    resp = client.get(f"/api/kpis/{kpi_id}/history/{route}", params=params)
    assert resp.status_code == 422, resp.text

    detail = resp.json()["detail"]
    # The refused id itself is not a destination; every OTHER id named is.
    named = set(re.findall(r"WS3-BI-\d{3}", detail)) - {kpi_id}
    assert named, f"{route} refused {kpi_id} naming no KPI at all: {detail}"
    dead_ends = sorted(named - set(family))
    assert not dead_ends, (
        f"/history/{route} refused {kpi_id} and named {dead_ends}, which this route "
        f"does NOT serve — a DEAD END. It said: {detail}"
    )


@pytest.mark.parametrize("route", ["segmented", "nowcast"])
def test_the_canonical_share_is_never_redirected_to_the_panel_share(client, route):
    """WS3-BI-008 must NOT be sent to WS3-BI-014: 014 refuses every patient axis
    and has no nowcast either, so naming it is a dead-end redirect — the one
    owner #14 removed from the calculator, the chat tool and the registry YAML
    (e21e8a91f). Every redirect this route names must round-trip.
    """
    params = {"axis": "segment"} if route == "segmented" else {}
    resp = client.get(f"/api/kpis/WS3-BI-008/history/{route}", params=params)
    assert resp.status_code == 422
    assert "WS3-BI-014" not in resp.json()["detail"]
