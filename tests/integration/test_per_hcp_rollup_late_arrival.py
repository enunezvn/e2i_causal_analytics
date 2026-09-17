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
            # The territory phase cross-joins EVERY territory with the planted dates and the
            # teardown deletes territory_metrics rows on those dates, so none may exist before.
            cur.execute(
                "SELECT count(*) FROM territory_metrics WHERE metric_date IN (%s, %s)",
                (TUESDAY, MONDAY),
            )
            assert cur.fetchone()[0] == 0, (
                "territory_metrics already holds rows on the planted 2019 dates"
            )
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
                        journey_stage, journey_status, brand, geographic_region, hcp_id,
                        adherence_rate, gap_days
                    ) VALUES (
                        %s, %s, %s, 'diagnosis'::journey_stage_type,
                        'active'::journey_status_type, %s::brand_type, 'northeast'::region_type, %s,
                        0.42, 7
                    )
                    """,
                    (f"pjlate_{hcp_id}", f"patlate_{hcp_id}", date(2018, 12, 25), BRAND, hcp_id),
                )
    yield {"rid": rid, **hcps}
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "DELETE FROM territory_metrics WHERE metric_date IN (%s, %s)", (TUESDAY, MONDAY)
            )
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


def _territory(db_conn: Any, rid: str) -> dict:
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT metric_date, total_trx, active_hcp_count FROM territory_metrics "
                "WHERE territory_id = %s",
                (f"T_LATE_{rid}",),
            )
            return {d: (int(trx), int(active)) for d, trx, active in cur.fetchall()}


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

    # Territory (owner decision #3): the 03:45 run rebuilds both whole dates from the per-HCP rows.
    from src.etl.territory_metrics_etl import (
        _run_territory_rollup_impl,
        preview_territory_rollup,
    )

    # The pre-fix default (metric_date in the 24 h before Monday 03:45) reaches Sunday only: no planted date.
    assert (
        preview_territory_rollup("2019-01-13T03:45:00+00:00", "2019-01-14T03:45:00+00:00")[
            "metric_dates"
        ]
        == 0
    )
    territory = _run_territory_rollup_impl(
        arrived_before="2019-01-14T03:45:00+00:00", request_id="late-arrival-territory"
    )
    assert territory["status"] == "completed" and territory["selected_by"] == "arrival", territory
    # (total_trx, active_hcp_count). Tuesday 01-01: a 2 + b 1 delivered. Monday 01-14: a3 delivered;
    # its 30-day lookback still holds a's and b's 01-01 triggers, so both HCPs are active.
    assert _territory(db_conn, rid) == {TUESDAY: (3, 2), MONDAY: (1, 2)}
    rebuilt = preview_territory_rollup("2019-01-01", "2019-01-15")
    assert (rebuilt["metric_dates"], rebuilt["rows_new"], rebuilt["rows_changed"]) == (2, 0, 0)

    # Adherence (owner decision #6, revised by codex r13-05): the 03:30 run must keep the
    # known adherence_rate (every journey has a NULL end date, so it is not computable) and
    # must not touch gap_days at all, which the synthetic DGP owns.
    from src.etl.patient_adherence_etl import _run_patient_adherence_impl

    adherence = _run_patient_adherence_impl(
        arrived_before="2019-01-14T03:30:00+00:00", request_id="late-arrival-adherence"
    )
    assert adherence["status"] == "completed" and adherence["selected_by"] == "arrival", adherence
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT hcp_id, adherence_rate, gap_days FROM patient_journeys "
                "WHERE patient_journey_id LIKE %s",
                (f"pjlate_hcplate_{rid}_%",),
            )
            journeys = {h: (float(rate), gap) for h, rate, gap in cur.fetchall()}
    # Both planted journeys keep exactly what the fixture wrote. Before the fix, a's would
    # have been (0.0, ...) from the LEAST/GREATEST clamp; the withdrawn gap rewrite would
    # have written 13 for a and 0 for b over the generator's 7.
    assert journeys[a] == (0.42, 7), journeys
    assert journeys[b] == (0.42, 7), journeys


def test_the_reconcile_deletes_our_obsolete_row_and_spares_a_foreign_one(
    db_conn: Any, planted: dict
) -> None:
    """THE DISCRIMINATING PAIR for codex r14-04's ownership predicate.

    The spec pins that predicate only by its PRESENCE in the SQL, and presence cannot fail
    for the reason the clause exists. It also cannot be exercised by live data: both
    ownership columns are NULL in all 1,840 live rows (measured 2026-09-17), so the
    "foreign row" branch of the DELETE is never reached by anything real. A predicate
    nobody exercises is untested — so this plants one.

    Two obsolete territory rows on a date with no per-HCP rows, identical except for
    ``market_potential``. After the reconcile:
      * the all-NULL row is GONE   — ours, so ours to delete
      * the valued row SURVIVES    — someone else's only copy of a budget figure
      * the preview counted them separately, one each

    ⚠ NEVER EXECUTED. Skip-only like the rest of this module; it runs in Task 31 Step 3.
    Until then the ownership predicate is pinned by text and by the partition property in
    the unit suite, and its runtime behaviour is unverified.
    """
    from src.etl.territory_metrics_etl import _run_territory_rollup_impl, preview_territory_rollup

    rid = planted["rid"]
    ours = f"T_LATE_{rid}_OURS"
    theirs = f"T_LATE_{rid}_THEIRS"
    orphan_date = date(2019, 1, 20)  # no planted trigger, so no per-HCP row can produce it

    with db_conn:
        with db_conn.cursor() as cur:
            for territory_id, market_potential in ((ours, None), (theirs, 42.0)):
                cur.execute(
                    """
                    INSERT INTO territory_metrics (
                        territory_id, metric_date, total_trx, total_nrx,
                        active_hcp_count, covered_lives, market_potential, is_synthetic
                    ) VALUES (%s, %s, 0, 0, 0, 0, %s, true)
                    """,
                    (territory_id, orphan_date, market_potential),
                )

    before = preview_territory_rollup("2019-01-20", "2019-01-21")
    assert before["rows_obsolete"] == 1, before
    assert before["rows_obsolete_foreign"] == 1, before
    assert (ours, "2019-01-20") in before["obsolete_keys"]
    assert (theirs, "2019-01-20") in before["obsolete_foreign_keys"]

    _run_territory_rollup_impl(
        start_date="2019-01-20", end_date="2019-01-21", request_id="late-arrival-ownership"
    )

    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT territory_id FROM territory_metrics WHERE metric_date = %s "
                "AND territory_id IN (%s, %s)",
                (orphan_date, ours, theirs),
            )
            survivors = {row[0] for row in cur.fetchall()}
    assert survivors == {theirs}, (
        f"expected only the foreign row to survive, got {survivors}: the ownership "
        "predicate either deleted someone else's budget figure or spared our stale row"
    )

    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "DELETE FROM territory_metrics WHERE metric_date = %s AND territory_id IN (%s, %s)",
                (orphan_date, ours, theirs),
            )
