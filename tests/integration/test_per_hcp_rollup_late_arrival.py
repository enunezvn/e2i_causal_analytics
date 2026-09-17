"""Per-HCP rollup late arrival (canonical TRx lane, owner decision #2, 2026-09-15).

The weekly host reseed (`0 3 * * 1 scripts/reseed_synthetic.sh`) inserts a whole Tue..Mon
week of triggers on Monday ~03:00:55, each stamped 00:00 of its own day. The rollup must
write every trigger under its own metric_date, and a date whose triggers arrive in two
batches must be recomputed whole, so market_share stays a share of the COMPLETE territory
total.

Planted in January 2019. On the live DB the earliest trigger_timestamp is 2023-07-30 and the
earliest triggers.created_at is 2026-06-10 (measured 2026-09-15), so neither the arrival
windows nor the touched dates can reach a real row. Opt-in like the sibling suite; the
planted rows are deleted at module teardown.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timezone
from typing import Any

import pytest

psycopg2 = pytest.importorskip("psycopg2")

pytestmark = pytest.mark.skipif(
    not (os.getenv("SUPABASE_DB_URL") and os.getenv("E2I_DB_INTEGRATION") == "1"),
    reason="SUPABASE_DB_URL and/or E2I_DB_INTEGRATION not set; requires real Postgres + explicit opt-in.",
)

UTC = timezone.utc
TUESDAY = date(2019, 1, 1)
MONDAY = date(2019, 1, 14)
# An off-cycle batch (cf. 2026-07-30 20:04), more than ARRIVAL_WINDOW_HOURS (174 h) before the Monday run.
EARLY_BATCH = datetime(2019, 1, 2, 3, 0, 55, tzinfo=UTC)
# The weekly batch. Its Tuesday triggers arrive 13 days late, like the 2026-07-27 two-week batch.
MONDAY_BATCH = datetime(2019, 1, 14, 3, 0, 55, tzinfo=UTC)
BRAND = "Kisqali"


@pytest.fixture(scope="module")
def db_conn() -> Any:
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def planted(db_conn: Any) -> Any:
    rid = uuid.uuid4().hex[:10]
    hcps = {"a": f"hcplate_{rid}_a", "b": f"hcplate_{rid}_b"}
    with db_conn:
        with db_conn.cursor() as cur:
            for hcp_id in hcps.values():
                cur.execute(
                    "INSERT INTO hcp_profiles (hcp_id, territory_id, geographic_region, sales_rep_id) "
                    "VALUES (%s, %s, 'northeast'::region_type, NULL)",
                    (hcp_id, f"T_LATE_{rid}"),
                )
                cur.execute(
                    """
                    INSERT INTO patient_journeys (
                        patient_journey_id, patient_id, journey_start_date,
                        journey_stage, journey_status, brand, geographic_region, hcp_id
                    ) VALUES (
                        %s, %s, %s, 'diagnosis'::journey_stage_type,
                        'active'::journey_status_type, %s::brand_type, 'northeast'::region_type, %s
                    )
                    """,
                    (f"pjlate_{hcp_id}", f"patlate_{hcp_id}", date(2018, 12, 25), BRAND, hcp_id),
                )
    yield {"rid": rid, **hcps}
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM business_metrics WHERE hcp_id LIKE %s", (f"hcplate_{rid}_%",))
            cur.execute("DELETE FROM triggers WHERE trigger_id LIKE %s", (f"trlate_{rid}_%",))
            cur.execute(
                "DELETE FROM patient_journeys WHERE patient_journey_id LIKE %s",
                (f"pjlate_hcplate_{rid}_%",),
            )
            cur.execute("DELETE FROM hcp_profiles WHERE hcp_id LIKE %s", (f"hcplate_{rid}_%",))


def _land_batch(
    db_conn: Any, rid: str, arrival: datetime, triggers: list[tuple[str, str, date]]
) -> None:
    """Insert one arrival batch: (trigger suffix, hcp_id, trigger day), all created at ``arrival``."""
    with db_conn:
        with db_conn.cursor() as cur:
            for suffix, hcp_id, day in triggers:
                ts = datetime(day.year, day.month, day.day, tzinfo=UTC)
                cur.execute(
                    """
                    INSERT INTO triggers (
                        trigger_id, patient_id, hcp_id, trigger_timestamp, brand_id,
                        delivery_status, acceptance_status, delivery_timestamp, created_at
                    ) VALUES (%s, %s, %s, %s, 'UNKNOWN', 'delivered', 'pending', %s, %s)
                    """,
                    (f"trlate_{rid}_{suffix}", f"patlate_{hcp_id}", hcp_id, ts, ts, arrival),
                )


def _rows(db_conn: Any, rid: str) -> dict:
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT hcp_id, metric_date, triggers_total_count, market_share FROM business_metrics "
                "WHERE hcp_id LIKE %s AND metric_type = 'per_hcp_rollup'",
                (f"hcplate_{rid}_%",),
            )
            return {(h, d): (int(n), float(s)) for h, d, n, s in cur.fetchall()}


def test_a_late_weekly_batch_rolls_up_under_each_triggers_own_date(
    db_conn: Any, planted: dict
) -> None:
    from src.etl.business_metrics_per_hcp_etl import (
        _run_per_hcp_rollup_impl,
        preview_per_hcp_rollup,
    )

    rid, a, b = planted["rid"], planted["a"], planted["b"]

    _land_batch(db_conn, rid, EARLY_BATCH, [("b1", b, TUESDAY)])
    first = _run_per_hcp_rollup_impl(
        arrived_before="2019-01-02T03:15:00+00:00", request_id="late-arrival-1"
    )
    assert first["status"] == "completed" and first["selected_by"] == "arrival", first
    assert _rows(db_conn, rid) == {(b, TUESDAY): (1, 1.0)}

    _land_batch(
        db_conn, rid, MONDAY_BATCH, [("a1", a, TUESDAY), ("a2", a, TUESDAY), ("a3", a, MONDAY)]
    )
    # The pre-fix default (trigger_timestamp in the 24 h before the Monday 03:15 beat) touches Monday only.
    assert (
        preview_per_hcp_rollup("2019-01-13T03:15:00+00:00", "2019-01-14T03:15:00+00:00")[
            "metric_dates"
        ]
        == 1
    )

    second = _run_per_hcp_rollup_impl(
        arrived_before="2019-01-14T03:15:00+00:00", request_id="late-arrival-2"
    )
    assert second["status"] == "completed" and second["rows_affected"] == 3, second
    rows = _rows(db_conn, rid)
    assert rows[(a, TUESDAY)] == pytest.approx((2, 2 / 3))
    # b's trigger arrived in the EARLIER batch, outside this window: its date is recomputed whole.
    assert rows[(b, TUESDAY)] == pytest.approx((1, 1 / 3))
    assert rows[(a, MONDAY)] == (1, 1.0)
    assert sum(share for (_, day), (_, share) in rows.items() if day == TUESDAY) == pytest.approx(
        1.0
    )

    # The dry-run readout agrees with what the run wrote.
    after = preview_per_hcp_rollup("2019-01-01", "2019-01-15")
    assert (
        after["metric_dates"],
        after["rows_new"],
        after["rows_changed"],
        after["rows_existing"],
    ) == (2, 0, 0, 3)
