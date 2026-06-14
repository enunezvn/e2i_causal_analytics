"""Issue #929 (secondary) — the gap_analyzer connector's empty-fetch log must
distinguish the CURRENT-period fetch from the PRIOR-period (YoY) fetch.

``fetch_prior_period`` shifts the window back by one period length for a YoY
comparison. For any wide ``time_period`` that shifted window predates the data, so
the prior fetch is empty on EVERY successful run and used to log the same
``"No data found for brand=..."`` warning as a real current-period miss — which reads
like fabrication when it is the expected, benign case. The fix: a benign INFO for the
prior window, a genuine WARNING only for an empty current window.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.gap_analyzer.connectors.supabase_connector import SupabaseDataConnector


def _connector_with_empty_repo() -> SupabaseDataConnector:
    fake_repo = MagicMock()
    fake_repo.get_time_series = AsyncMock(return_value=[])
    connector = SupabaseDataConnector(supabase_client=MagicMock(), include_synthetic=True)
    connector._repository = fake_repo  # bypass lazy client resolution
    return connector


@pytest.mark.unit
@pytest.mark.asyncio
async def test_current_period_empty_logs_warning(caplog):
    """An empty CURRENT-period fetch logs a WARNING that names the current period."""
    connector = _connector_with_empty_repo()

    with caplog.at_level(logging.DEBUG):
        df = await connector.fetch_performance_data(
            brand="Remibrutinib",
            metrics=["trx"],
            segments=["region"],
            time_period="2024-01-01_2024-03-31",
        )

    assert df.empty
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("current" in r.getMessage().lower() for r in warnings), (
        f"expected a current-period WARNING, got: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_prior_period_empty_does_not_warn(caplog):
    """An empty PRIOR-period (YoY) fetch must NOT emit the generic 'No data found'
    WARNING — it is the expected case and should be a benign, role-labelled log."""
    connector = _connector_with_empty_repo()

    with caplog.at_level(logging.DEBUG):
        df = await connector.fetch_prior_period(
            brand="Remibrutinib",
            metrics=["trx"],
            segments=["region"],
            time_period="2024-01-01_2024-03-31",
        )

    assert df.empty
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    # No WARNING-level "no data" alarm for the benign prior-window emptiness.
    assert not any("no data found" in r.getMessage().lower() for r in warnings), (
        f"prior-period emptiness should not raise the generic warning, got: "
        f"{[(r.levelname, r.getMessage()) for r in warnings]}"
    )
    # And the benign event is still observable, labelled as the prior/YoY window.
    assert any("prior" in r.getMessage().lower() for r in caplog.records), (
        f"expected a prior-period labelled log, got: {[r.getMessage() for r in caplog.records]}"
    )
