"""e2i_data_query_tool honours time_range for the canonical volume KPIs (Task 15A, codex r3 HIGH).

Drives the PUBLIC tool through ``ainvoke`` with a frozen UTC clock (the one
``_get_time_filter`` reads) and a recording calculator in place of
``get_kpi_calculator``. The helper-level tests in
tests/unit/test_kpi/test_canonical_volume_stored.py can pass while the tool never
reaches the helper, or reaches it with a lookback it computed differently; only this
file pins the two together.

The eligibility rule is "the frontier month completed on or after the lookback start",
not strict month overlap. The parametrized default-range cases include 2026-10-31,
where strict overlap would refuse the tool's own default.
"""

import asyncio
import calendar
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from src.api.routes import chatbot_tools
from src.kpi import canonical_volume_stored as cvs


class _Calculator:
    def __init__(self, data_month):
        year, month = int(data_month[:4]), int(data_month[5:7])
        through = date(year, month, calendar.monthrange(year, month)[1]).isoformat()
        self.calls = []
        self.result = SimpleNamespace(
            value=800349.18,
            error=None,
            metadata={
                "context": {"data_month": data_month, "data_through": through},
                "include_synthetic": False,
            },
        )

    def calculate(self, kpi_id, use_cache=True, force_refresh=False, context=None):
        self.calls.append((kpi_id, dict(context or {})))
        return self.result


def _world(monkeypatch, today, frontier):
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(today.year, today.month, today.day, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(chatbot_tools, "datetime", _Frozen)
    monkeypatch.setattr(cvs, "_utc_today", lambda: today)
    import src.api.routes.kpi as kpi_routes

    calc = _Calculator(frontier)
    monkeypatch.setattr(kpi_routes, "get_kpi_calculator", lambda: calc)
    return calc


def _ask(**kwargs):
    args = {"query_type": "kpi", "brand": "Kisqali", "kpi_name": "TRx", **kwargs}
    return asyncio.run(chatbot_tools.e2i_data_query_tool.ainvoke(args))


def test_the_model_visible_tool_description_matches_what_the_tool_now_does():
    """codex iter1 HIGH: Task 15A landed LAST and left the SHIPPED contract describing
    pre-15A behaviour.

    ``StructuredTool.description`` is what the routing model actually reads, so it is
    asserted here rather than the source docstring — a docstring test would pass on a
    tool whose description was built some other way. Before 15A the tool returned raw
    stored rows and NBRx really was unavailable; saying so now both misroutes the model
    away from a tool that answers correctly, and re-tells the two-shape story this lane
    exists to retire.
    """
    description = chatbot_tools.e2i_data_query_tool.description
    for stale in ("raw stored rows", "not materialized here"):
        assert stale not in description, f"stale pre-15A claim in the tool description: {stale!r}"
    assert "canonical" in description.lower()


def test_last_7_days_on_sep_15_refuses_the_august_headline(monkeypatch):
    calc = _world(monkeypatch, date(2026, 9, 15), "2026-08-01")
    out = _ask(time_range="last_7_days")
    assert calc.calls == [("WS3-BI-005", {"brand": "Kisqali"})]
    assert out["success"] is False
    assert out["data"] == []
    assert out["count"] == 0
    # A refusal that still carried the number would be the two-shapes bug wearing a flag.
    assert "800349" not in repr(out)
    block = out["out_of_range"]
    assert block["requested_since"] == "2026-09-08"
    assert block["frontier_month"] == "2026-08-01"
    assert block["frontier_complete_on"] == "2026-09-01"
    assert block["use_instead"] == "kpi_calculate_tool(window=...)"


@pytest.mark.parametrize(
    "today,frontier,since",
    [
        (date(2026, 9, 15), "2026-08-01", "2026-08-16"),
        (date(2026, 10, 1), "2026-09-01", "2026-09-01"),
        (date(2026, 10, 4), "2026-09-01", "2026-09-04"),  # Sunday, before the 2026-10-05 03:00 cron
        (date(2026, 10, 31), "2026-09-01", "2026-10-01"),  # strict month overlap would refuse this
        (date(2027, 3, 31), "2027-02-01", "2027-03-01"),
    ],
)
def test_the_default_time_range_serves_the_latest_complete_month(
    monkeypatch, today, frontier, since
):
    _world(monkeypatch, today, frontier)
    out = _ask()  # no time_range: the tool default, LAST_30_DAYS
    assert out["success"] is True
    assert out["lookback_start"] == since
    assert [r["metric_date"] for r in out["data"]] == [frontier]


def test_last_30_days_on_sep_30_serves_august(monkeypatch):
    _world(monkeypatch, date(2026, 9, 30), "2026-08-01")
    out = _ask(time_range="last_30_days")
    assert out["success"] is True
    assert out["lookback_start"] == "2026-08-31"
    assert out["data"][0]["metric_date"] == "2026-08-01"
    assert out["data"][0]["value"] == 800349.18
