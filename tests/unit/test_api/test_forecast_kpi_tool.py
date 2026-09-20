"""forecast_kpi_tool: the chat surface for KPI forecasting (#2115, Lane B, demo 6.5).

The tool is exercised against a REAL canonical series — the PROD rows in
tests/fixtures/forecast/canonical_trx_monthly.csv, shaped by the real Lane A shaper and
forecast by real statsmodels fits. The only thing supplied to it is the Supabase reader
it would otherwise call, which returns those same real rows.

The assertions that matter most are not about the numbers. They are about what the
payload FORCES the answer to say: that this is a univariate extrapolation of the brand's
own history, that it therefore cannot see a competitor entry or a payer change inside the
window, and that the risk half of demo 6.5 has to come from the gap and causal tools. The
planted Kisqali midwest step from 2026-10 sits inside exactly this horizon and is
invisible to every model the tool runs, so a payload that let the answer imply otherwise
would be confidently wrong about the one thing the user asked for.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date

import pytest

from src.api.routes import chat_forecast_tool as cft
from tests.unit.test_kpi.conftest_forecast_fixture import load_canonical_trx

pytestmark = pytest.mark.timeout(300)


class FixtureClient:
    """The real Supabase call shape, answered from the real PROD rows.

    Not a stand-in for the forecaster: everything under test still runs for real. This
    only replaces the network hop to a database whose contents are what the fixture is.
    """

    def __init__(self, brand_filter=True):
        self.calls = []
        self._brand_filter = brand_filter

    def rpc(self, name, payload):
        self.calls.append((name, payload))
        metric, brand, region = payload["params"]
        rows = []
        for b, series in load_canonical_trx().items():
            if self._brand_filter and brand is not None and b != brand:
                continue
            for month, value in series:
                rows.append({"metric_date": month.isoformat(), "value": value, "n_rows": 4})

        class _Exec:
            def execute(self):
                return type("R", (), {"data": rows})()

        return _Exec()


def run(_client=None, _as_of=date(2026, 9, 20), **kwargs):
    """Drive the tool's real behaviour with the fixture reader.

    ``run_forecast`` is exactly what the ``@tool`` wrapper calls, one argument wider;
    the wrapper's own wiring is pinned separately at the bottom of this file.
    """
    return asyncio.run(cft.run_forecast(client=_client or FixtureClient(), as_of=_as_of, **kwargs))


@pytest.fixture(scope="module")
def kisqali():
    return run(kpi_name="TRx", brand="Kisqali", horizon_months=6)


# ------------------------------------------------------------------------ happy path
def test_it_answers_demo_6_5_with_two_quarters_of_monthly_forecast(kisqali):
    assert kisqali["success"] is True
    assert kisqali["query_type"] == "kpi_forecast"
    assert kisqali["brand"] == "Kisqali"
    assert kisqali["horizon_months"] == 6
    assert len(kisqali["forecast"]) == 6
    assert kisqali["forecast"][0]["month"] == "2026-09"
    assert kisqali["forecast"][-1]["month"] == "2027-02"


def test_every_forecast_month_carries_a_band(kisqali):
    for point in kisqali["forecast"]:
        assert point["lower"] <= point["value"] <= point["upper"]


def test_the_two_quarter_total_is_reported_because_that_is_what_was_asked(kisqali):
    assert kisqali["horizon_total"] == pytest.approx(
        sum(p["value"] for p in kisqali["forecast"]), rel=1e-6
    )


def test_the_backtest_that_chose_the_model_is_in_the_payload(kisqali):
    backtest = kisqali["backtest"]
    assert backtest["origins"] >= 12
    assert backtest["champion_monthly_mape_pct"] < 10.0
    models = {m["model"] for m in backtest["models"]}
    assert "holt_winters_seasonal_add" in models
    assert kisqali["champion"] in models


def test_the_payload_is_json_serialisable_for_the_chat_transport(kisqali):
    json.dumps(kisqali)


# ----------------------------------------------------------------- honesty about scale
def test_the_forecast_declares_the_basis_of_the_series_it_extends(kisqali):
    """A forecast of canonical TRx is a canonical-TRx figure and must carry its basis,
    or the #1640 scale guard cannot stop it being compared with a patient-panel count."""
    basis = kisqali["measure_basis"]
    assert basis["comparison_key"] == ["business_metrics"]
    assert "business_metrics" in basis["substrate"]


def test_the_history_the_forecast_rests_on_is_named(kisqali):
    assert kisqali["data_through"] == "2026-08-31"
    assert kisqali["n_observations"] == 164
    # Synthetic mode appends a suffix to pick the twin statement; the base id is
    # what identifies WHICH vetted statement served the history.
    assert kisqali["series_query_id"].startswith("canonical_volume_monthly_series")


def test_the_forecast_is_labelled_a_projection_not_an_observation(kisqali):
    assert kisqali["is_forecast"] is True
    assert "forecast" in kisqali["note"].lower() or "project" in kisqali["note"].lower()


# --------------------------------------------------- honesty about what it cannot see
def test_the_payload_says_the_forecast_is_univariate_and_blind_to_outside_events(kisqali):
    """This is the load-bearing one for demo 6.5.

    Every model the tool runs sees ONLY the brand's own past volume. A competitor
    entry, a payer change or a regional step inside the window is invisible to all of
    them — the planted Kisqali midwest -15% from 2026-10 lands inside this very horizon.
    The payload has to say so, or the answer will present a number that cannot know
    about the risk it was asked about in the same sentence.
    """
    limits = kisqali["limitations"]
    text = " ".join(limits).lower()
    assert "univariate" in text or "own history" in text
    for expected in ("competitor", "payer"):
        assert expected in text, f"the blind spot {expected!r} must be named"


def test_the_payload_routes_the_risk_half_of_the_question_to_the_tools_that_can_see_it(
    kisqali,
):
    """6.5 asks for a forecast AND its biggest risk. The risk half is not forecastable
    from one series, so the payload names the tools that can actually answer it."""
    follow_up = " ".join(kisqali["risk_analysis_requires"]).lower()
    assert "gap" in follow_up
    assert "causal" in follow_up


def test_the_tool_tells_the_model_what_a_floored_month_means(kisqali):
    """The flag existing in the payload is not the same as the answer using it.

    Without an instruction, a floored month renders as a bare "0.0" — a precise-looking
    prediction of zero, which is the opposite of what it means. The rule has to be
    somewhere the presenting model reads.
    """
    limitations = " ".join(kisqali["limitations"]).lower()
    assert "floored_at_zero" in limitations
    assert "near zero" in limitations
    assert "not a prediction of exactly zero" in limitations or "never as a precise" in limitations

    doc = cft.forecast_kpi_tool.description.lower()
    assert "floored_at_zero" in doc, "the tool docstring must carry the rule too"


def test_the_band_says_what_it_is_made_of(kisqali):
    assert kisqali["band"]["coverage"] == 0.8
    assert "rolling-origin" in kisqali["band"]["basis"]


# --------------------------------------------------------------------------- refusals
def test_a_kpi_that_has_no_canonical_monthly_series_is_refused_by_name():
    out = run(kpi_name="conversion_rate", brand="Kisqali")
    assert out["success"] is False
    assert "conversion" in out["error"].lower()
    assert "trx" in out["error"].lower(), "the refusal must name what IS forecastable"


def test_an_unknown_brand_is_refused_rather_than_forecast_from_an_empty_series():
    out = run(kpi_name="TRx", brand="Nosuchbrand")
    assert out["success"] is False
    assert "Nosuchbrand" in out["error"]


def test_a_horizon_beyond_the_maximum_is_refused_with_the_maximum_named():
    out = run(kpi_name="TRx", brand="Kisqali", horizon_months=36)
    assert out["success"] is False
    assert "12" in out["error"]


def test_a_refusal_never_carries_a_forecast():
    out = run(kpi_name="TRx", brand="Nosuchbrand")
    assert out.get("forecast") in (None, [])


# ------------------------------------------------------------------- argument handling
@pytest.mark.parametrize(
    "name,expected",
    [
        ("TRx", "trx"),
        ("trx", "trx"),
        ("  TRx  ", "trx"),
        ("WS3-BI-005", "trx"),
        ("total prescriptions", "trx"),
        ("Total Prescriptions (TRx)", "trx"),
        ("NRx", "nrx"),
        ("WS3-BI-006", "nrx"),
        ("NBRx", "nbrx"),
        ("WS3-BI-007", "nbrx"),
    ],
)
def test_the_kpi_is_recognised_however_the_user_phrased_it(name, expected):
    """Resolution is asserted directly: running a full backtest per alias would spend
    ten seconds each to re-test one lookup, and the end-to-end path is covered below."""
    assert cft._resolve_metric(name) == expected


@pytest.mark.parametrize("name,expected", [("NRx", "nrx"), ("NBRx", "nbrx")])
def test_the_other_canonical_volume_metrics_are_forecastable_end_to_end(name, expected):
    out = run(kpi_name=name, brand="Kisqali", horizon_months=3)
    assert out["success"] is True, out.get("error")
    assert out["metric"] == expected
    assert len(out["forecast"]) == 3


def test_two_quarters_is_the_default_horizon():
    out = run(kpi_name="TRx", brand="Kisqali")
    assert out["horizon_months"] == 6


def test_the_brand_reaches_the_query_so_the_series_is_not_every_brand_summed():
    client = FixtureClient()
    run(kpi_name="TRx", brand="Kisqali", horizon_months=3, _client=client)
    payload = client.calls[0][1]
    assert payload["params"][0] == "trx"
    assert payload["params"][1] == "Kisqali"


def test_a_region_is_passed_through_to_the_series_read():
    client = FixtureClient(brand_filter=True)
    run(kpi_name="TRx", brand="Kisqali", region="Midwest", horizon_months=3, _client=client)
    # The region reaches the statement as its region_type ENUM LABEL, not the display
    # form: business_metrics.region is an enum and a non-label string raises 22P02,
    # failing the whole read (#1501).
    assert client.calls[0][1]["params"][2] == "midwest"


# ------------------------------------------------------------------- the tool wrapper
def test_the_tool_is_a_real_langchain_tool_the_model_can_be_bound_to():
    assert cft.forecast_kpi_tool.name == "forecast_kpi_tool"
    assert cft.forecast_kpi_tool.args_schema is cft.ForecastKpiInput
    fields = set(cft.ForecastKpiInput.model_fields)
    assert fields == {"kpi_name", "brand", "region", "horizon_months"}


def test_the_tool_description_tells_the_model_when_to_reach_for_it():
    """A tool the model never selects is a tool that does not exist. 6.5's phrasing
    ('forecast ... for the next two quarters') has to be recognisable in the docstring,
    and so does the boundary against the two tools that look adjacent but do not
    forecast."""
    doc = cft.forecast_kpi_tool.description.lower()
    assert "forecast" in doc and "two quarters" in doc
    assert "kpi_calculate_tool" in doc, "the model must be told which tool is NOT this"
    assert "univariate" in doc
    assert "causal_analysis_tool" in doc, "the risk half must be routed from the docstring"


def test_the_tool_delegates_to_the_same_implementation_the_tests_drive():
    """The wrapper must not be a second implementation that could drift from this one."""
    import inspect

    source = inspect.getsource(cft.forecast_kpi_tool.coroutine)
    assert "run_forecast(" in source


def test_calling_the_tool_for_real_refuses_honestly_rather_than_raising():
    """No reader is configured in a unit-test process, so this exercises the wrapper's
    own path end to end: it must come back as a refusal payload, never an exception."""
    out = asyncio.run(
        cft.forecast_kpi_tool.ainvoke(
            {"kpi_name": "conversion_rate", "brand": "Kisqali", "horizon_months": 6}
        )
    )
    assert out["success"] is False
    assert out["query_type"] == "kpi_forecast"
