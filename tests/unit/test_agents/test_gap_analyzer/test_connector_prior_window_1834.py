"""#1834 — ``SupabaseDataConnector`` windows against monthly-grain rows.

Read from the local prod DB on 2026-08-30 (read-only):

    select metric_date, value from business_metrics
     where brand='Remibrutinib' and region='west' and metric_name='trx'
       and metric_date >= '2026-03-01' order by 1;

    2026-03-01|62903.46   2026-04-01|66412.17   2026-05-01|49893.14
    2026-06-01|34561.03   2026-07-01|49585.25   2026-08-01|60885.99

Rows sit on the FIRST of each month. ``fetch_prior_period`` used to shift the
window back by *day count*, so a quarter-to-date window (Jul 1–Aug 30) got a
prior of May 1/2–Jun 30 — one or two monthly rows instead of three — and the
default ``current_quarter`` never parsed at all (silent last-90-days window).

The fake repository below is the ONLY test double: it replays those real rows
and filters by the ``start_date``/``end_date`` the REAL connector computes, so
the assertions are on the frames the real pivot produces.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.agents.gap_analyzer.connectors.supabase_connector import SupabaseDataConnector
from src.utils.gap_time_period import TimePeriodError

TODAY = date(2026, 8, 30)

# Real prod rows (Remibrutinib / west / trx), monthly grain on the 1st.
WEST_TRX: Dict[str, float] = {
    "2026-01-01": 58000.00,
    "2026-02-01": 61000.00,
    "2026-03-01": 62903.46,
    "2026-04-01": 66412.17,
    "2026-05-01": 49893.14,
    "2026-06-01": 34561.03,
    "2026-07-01": 49585.25,
    "2026-08-01": 60885.99,
}


class _MonthlyRepo:
    """Replays the real monthly rows, filtered by the window the connector asks for."""

    def __init__(self) -> None:
        self.windows: List[tuple[str, str]] = []

    async def get_time_series(
        self,
        kpi_name: str,
        brand: str,
        start_date: str,
        end_date: str,
        include_synthetic: bool = False,
    ) -> List[Dict[str, Any]]:
        self.windows.append((start_date, end_date))
        return [
            {"metric_date": d, "value": v, "target": None, "region": "west"}
            for d, v in WEST_TRX.items()
            if start_date <= d <= end_date
        ]


def _connector() -> tuple[SupabaseDataConnector, _MonthlyRepo]:
    repo = _MonthlyRepo()
    connector = SupabaseDataConnector(supabase_client=MagicMock(), include_synthetic=True)
    connector._repository = repo  # bypass lazy client resolution; the repo is the double
    return connector, repo


def _mean(*dates: str) -> float:
    return sum(WEST_TRX[d] for d in dates) / len(dates)


@pytest.fixture
def frozen_today(monkeypatch):
    """Freeze the grammar's clock at the module seam the connector reads through."""
    import src.utils.gap_time_period as tp

    monkeypatch.setattr(tp, "_today", lambda: TODAY)
    return TODAY


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_quarter_window_three_monthly_rows_in_three_out(frozen_today):
    """Q2_2026: Apr/May/Jun in; prior = Q1 (Jan/Feb/Mar) — 3 rows each side."""
    connector, repo = _connector()

    current = await connector.fetch_performance_data(
        brand="Remibrutinib", metrics=["trx"], segments=["region"], time_period="Q2_2026"
    )
    prior = await connector.fetch_prior_period(
        brand="Remibrutinib", metrics=["trx"], segments=["region"], time_period="Q2_2026"
    )

    assert repo.windows == [("2026-04-01", "2026-06-30"), ("2026-01-01", "2026-03-31")]
    assert len(current) == 1 and len(prior) == 1  # pivoted: one row per region
    assert current.loc[0, "trx"] == pytest.approx(_mean("2026-04-01", "2026-05-01", "2026-06-01"))
    assert prior.loc[0, "trx"] == pytest.approx(_mean("2026-01-01", "2026-02-01", "2026-03-01"))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_current_quarter_default_is_quarter_to_date_vs_preceding_full_quarter(
    frozen_today,
):
    """The DEFAULT: Jul/Aug to date (2 rows) vs the full Q2 (3 rows) — not last-90-days."""
    connector, repo = _connector()

    current = await connector.fetch_performance_data(
        brand="Remibrutinib", metrics=["trx"], segments=["region"], time_period="current_quarter"
    )
    prior = await connector.fetch_prior_period(
        brand="Remibrutinib", metrics=["trx"], segments=["region"], time_period="current_quarter"
    )

    assert repo.windows == [("2026-07-01", "2026-08-30"), ("2026-04-01", "2026-06-30")]
    assert current.loc[0, "trx"] == pytest.approx(_mean("2026-07-01", "2026-08-01"))
    assert prior.loc[0, "trx"] == pytest.approx(_mean("2026-04-01", "2026-05-01", "2026-06-01"))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_explicit_quarter_to_date_range_prior_is_whole_months_not_day_shift(frozen_today):
    """2026-07-01_2026-08-30 explicit: prior = May 1–Jun 30 (2 rows), never May 2–Jun 30."""
    connector, repo = _connector()

    await connector.fetch_prior_period(
        brand="Remibrutinib",
        metrics=["trx"],
        segments=["region"],
        time_period="2026-07-01_2026-08-30",
    )

    assert repo.windows == [("2026-05-01", "2026-06-30")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_time_period_raises_before_any_read_and_never_falls_back(
    frozen_today, caplog
):
    """Fail closed: garbage raises ``TimePeriodError`` (a ValueError) BEFORE the
    repository is touched, with no 'using last 90 days' warning anywhere."""
    connector, repo = _connector()

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(TimePeriodError) as excinfo:
            await connector.fetch_performance_data(
                brand="Remibrutinib", metrics=["trx"], segments=["region"], time_period="bogus"
            )
        with pytest.raises(TimePeriodError):
            await connector.fetch_prior_period(
                brand="Remibrutinib", metrics=["trx"], segments=["region"], time_period="bogus"
            )

    assert repo.windows == []
    assert "current_quarter" in str(excinfo.value)
    assert not any("90 days" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
def test_parse_time_period_shim_returns_iso_strings_and_fails_closed(frozen_today):
    """The legacy ``_parse_time_period`` hook keeps its (start, end) contract but now
    speaks the shared grammar — and raises instead of defaulting."""
    connector, _ = _connector()

    assert connector._parse_time_period("current_quarter") == ("2026-07-01", "2026-08-30")
    assert connector._parse_time_period("2024-Q3") == ("2024-07-01", "2024-09-30")
    with pytest.raises(TimePeriodError):
        connector._parse_time_period("bogus")
