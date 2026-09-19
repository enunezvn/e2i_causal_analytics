"""The harness runs the generated statements on planted rows, and those statements
ignore rows without both dimensions (canonical TRx lane, codex r2)."""

from datetime import date

import pytest

from tests.integration.canonical_sql_harness import PlantedBusinessMetrics, bm_row

pytestmark = pytest.mark.integration


def test_the_headline_share_and_portfolio_ignore_rows_without_both_dimensions():
    db = PlantedBusinessMetrics(
        [
            bm_row("trx", "2026-08-01", "Kisqali", "west", 10),
            bm_row("trx", "2026-08-01", "Kisqali", None, 7777),
            bm_row("trx", "2026-08-01", None, "west", 8888),
            bm_row("trx", "2026-09-01", "Kisqali", "west", 999),
        ],
        date(2026, 9, 15),
    )
    [head] = db.query("canonical_volume_trx", ["Kisqali"])
    assert float(head["trx"]) == 10.0
    assert head["data_month"] == "2026-08-01" and head["data_through"] == "2026-08-31"
    [portfolio] = db.query("canonical_volume_trx", [None])
    assert float(portfolio["trx"]) == 10.0
    [share] = db.query("canonical_volume_trx_share", ["Kisqali"])
    assert float(share["share"]) == 1.0


def test_a_null_dimension_row_cannot_move_the_frontier():
    db = PlantedBusinessMetrics(
        [
            bm_row("trx", "2026-07-01", "Kisqali", "west", 10),
            bm_row("trx", "2026-08-01", "Kisqali", None, 5),
        ],
        date(2026, 9, 15),
    )
    [head] = db.query("canonical_volume_trx", ["Kisqali"])
    assert head["data_month"] == "2026-07-01" and float(head["trx"]) == 10.0
