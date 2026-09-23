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
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

from tests.integration._prod_write_guard import (
    adherence_spec,
    per_hcp_rollup_spec,
    require_isolated_windows,
    require_no_foreign_reconcile,
    require_windows_still_isolated,
    selected_metric_dates,
    territory_arrival_spec,
    territory_rollup_spec,
)

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
# Guard windows: wide enough to cover every window any phase of this file uses -- the
# arrival runs (the first at 03:15 on 2019-01-02, then 03:15 / 03:30 / 03:45 on 2019-01-14,
# each looking back the ETL's ARRIVAL_WINDOW_HOURS from its arrived_before), the explicit
# 2019-01-20 territory reconcile of
# test_the_reconcile_deletes_our_obsolete_row_and_spares_a_foreign_one, and the explicit
# per-HCP window [EXPLICIT_START, EXPLICIT_END) the #2210 guard tests run. Deliberately the
# UNION of the file's windows, not one of them: a guard that censuses a narrower range than
# the test writes over cannot fail for the reason it exists. The arrival census therefore
# starts where the FIRST run's lookback starts (codex r3 on #2212: starting it at
# EARLY_BATCH left the week before uncensused), derived in the fixture from the ETL's own
# constant so the two cannot drift.
#
# The territory ARRIVAL run (#2213) is censused separately, and inside the test rather than
# here: it selects its dates with territory_metrics_etl._TERRITORY_METRIC_DATES_BY_ARRIVAL
# (every per-HCP date written in its window, or within 30 days of a trigger that ARRIVED in
# it), so the selection depends on the triggers and per-HCP rows the earlier phases plant
# and can reach a foreign per-HCP date well outside [TUESDAY, GUARD_DATE_END). The census
# imports that CTE (territory_arrival_spec) and runs right before the run, the test records
# the selected set from the same CTE and the run's transaction id, and the teardown deletes
# the run's territory_metrics rows on every recorded date -- not just the two planted ones. The range census below still covers the
# explicit 2019-01-20 reconcile and the per-date teardown floor.
FIRST_ARRIVAL_RUN = "2019-01-02T03:15:00+00:00"
TERRITORY_ARRIVAL_RUN = "2019-01-14T03:45:00+00:00"
GUARD_ARRIVAL_END = datetime(2019, 1, 21, tzinfo=UTC)
GUARD_DATE_END = date(2019, 1, 21)
# The explicit per-HCP window: selected by trigger_timestamp, not by arrival, so it needs
# its own census (codex r2 on #2212: a foreign trigger dated inside it but created outside
# the arrival window would be rolled up and left behind by the prefix-scoped teardown).
EXPLICIT_START = "2019-01-01"
EXPLICIT_END = "2019-01-15"


@pytest.fixture(scope="module")
def db_conn() -> Any:
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def planted(db_conn: Any) -> Any:
    # 6 hex: every id built from it must fit the prod varchar(20) columns (hcp_id,
    # patient_id, patient_journey_id, territory_id); the old 10-hex ids were 27-28 chars.
    rid = uuid.uuid4().hex[:6]
    hcps = {"a": f"hl_{rid}_a", "b": f"hl_{rid}_b"}
    # Prod-write guard (owner-approved, 2026-09-17). This replaces the narrower inline
    # check this fixture used to carry -- it proved only that territory_metrics was
    # empty on the planted dates, which is the guard's third leg. The shared guard adds
    # the derivation leg (no unplanted trigger on an affected date) and the key-space
    # leg (the territory CROSS JOIN reaching real territories), and it FAILS rather than
    # raising a bare AssertionError, so the message names the window and the counts.
    # The arrival phases census on created_at from the first run's own lookback start; the
    # explicit per-HCP runs of the #2210 guard tests select by trigger_timestamp and get
    # their own census, so a trigger dated in the explicit window but created outside the
    # arrival window cannot pass the gate.
    from src.etl.business_metrics_per_hcp_etl import ARRIVAL_WINDOW_HOURS

    guard_arrival_start = datetime.fromisoformat(FIRST_ARRIVAL_RUN) - timedelta(
        hours=ARRIVAL_WINDOW_HOURS
    )
    guard_specs = (
        per_hcp_rollup_spec(
            test_file=__file__,
            start=guard_arrival_start,
            end=GUARD_ARRIVAL_END,
            hcp_like=f"hl_{rid}_%",
            trigger_like=f"trlate_{rid}_%",
            window_column="created_at",
        ),
        per_hcp_rollup_spec(
            test_file=__file__,
            start=datetime.fromisoformat(EXPLICIT_START).replace(tzinfo=UTC),
            end=datetime.fromisoformat(EXPLICIT_END).replace(tzinfo=UTC),
            hcp_like=f"hl_{rid}_%",
            trigger_like=f"trlate_{rid}_%",
            window_column="trigger_timestamp",
        ),
        territory_rollup_spec(
            test_file=__file__,
            start=TUESDAY,
            end=GUARD_DATE_END,
            territory_like=f"T_LATE_{rid}%",
            teardown_deletes_window=True,
        ),
        adherence_spec(
            test_file=__file__,
            start=guard_arrival_start,
            end=GUARD_ARRIVAL_END,
            journey_like=f"pj_hl_{rid}_%",
            window_column="created_at",
        ),
    )
    require_isolated_windows(db_conn, *guard_specs)
    with db_conn:
        with db_conn.cursor() as cur:
            for hcp_id in hcps.values():
                cur.execute(
                    "INSERT INTO hcp_profiles (hcp_id, territory_id, geographic_region, sales_rep_id) "
                    "VALUES (%s, %s, 'northeast'::region_type, NULL)",
                    (hcp_id, f"T_LATE_{rid}"),
                )
            # Three journeys, all ARRIVING with the weekly batch (created_at is what the
            # scheduled adherence run selects on; left to its now() default the journeys
            # would sit outside every 2019 window and the run would find nothing).
            # a's and b's journeys are open-ended (NULL end date): adherence_rate is not
            # computable and must be preserved. c's journey (a second patient of HCP a)
            # is computable -- 5 of 10 days -- and its stored 0.42 must become 0.5. It is
            # what proves the run reached the planted rows rather than missing them.
            journeys = (
                (f"pj_{hcps['a']}", f"pt_{hcps['a']}", hcps["a"], None, None),
                (f"pj_{hcps['b']}", f"pt_{hcps['b']}", hcps["b"], None, None),
                (f"pj_hl_{rid}_c", f"pt_hl_{rid}_c", hcps["a"], date(2019, 1, 4), 5),
            )
            for journey_id, patient_id, hcp_id, end_date, duration in journeys:
                cur.execute(
                    """
                    INSERT INTO patient_journeys (
                        patient_journey_id, patient_id, journey_start_date, journey_end_date,
                        journey_duration_days, journey_stage, journey_status, brand,
                        geographic_region, hcp_id, adherence_rate, gap_days, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, 'diagnosis'::journey_stage_type,
                        'active'::journey_status_type, %s::brand_type, 'northeast'::region_type,
                        %s, 0.42, 7, %s
                    )
                    """,
                    (
                        journey_id,
                        patient_id,
                        date(2018, 12, 25),
                        end_date,
                        duration,
                        BRAND,
                        hcp_id,
                        MONDAY_BATCH,
                    ),
                )
    # The territory ARRIVAL run writes a row for EVERY territory on EVERY date it selects;
    # the test records that set (from the run's own CTE, right before the run) here, and
    # the run's transaction id (codex r3-1: ownership by elapsed time was a proxy -- any
    # concurrent writer on a recorded date satisfied it). The run inserts and reconciles
    # in ONE transaction, so every row it wrote carries that xid as its xmin; the teardown
    # deletes exactly those rows plus our own keyed territory, and REPORTS anything else
    # on the recorded dates rather than deleting it. The two planted dates are the floor.
    territory_selected_dates: set[date] = set()
    state = {
        "rid": rid,
        "territory_selected_dates": territory_selected_dates,
        "territory_run_xid": None,
        "territory_run_created_at": None,
        # Set by the territory test: the arrival run's own selection params and the census
        # spec built on them, so the teardown can re-read the selection and the final
        # re-census can include it (#2215).
        "territory_arrival_params": None,
        "territory_extra_specs": [],
        **hcps,
    }
    yield state
    arrival_params = state["territory_arrival_params"]
    if arrival_params is not None:
        # #2215 (codex r1 on this lane): a selected date that appeared between the
        # recording and the run is reported by the test, but the run still wrote a row
        # for every territory on it. Re-read the run's own selection while the planted
        # rows that drive it still exist, and sweep the union, not just the recorded set.
        territory_selected_dates.update(selected_metric_dates(db_conn, arrival_params))
    dates = sorted({TUESDAY, MONDAY} | territory_selected_dates)
    run_xid = state["territory_run_xid"]
    run_created_at = state["territory_run_created_at"]
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "DELETE FROM territory_metrics WHERE territory_id LIKE %s", (f"T_LATE_{rid}%",)
            )
            if run_xid is not None:
                # The run's own INSERTS: its xid AND its transaction timestamp. A foreign
                # row the run overwrote takes the xid but keeps its created_at (the upsert
                # stamps NOW() on insert only), so it is reported below, not swept (#2215).
                cur.execute(
                    "DELETE FROM territory_metrics WHERE metric_date = ANY(%s) "
                    "AND xmin::text = %s AND created_at = %s",
                    (dates, run_xid, run_created_at),
                )
            cur.execute(
                "SELECT territory_id, metric_date, xmin::text, created_at FROM territory_metrics "
                " WHERE metric_date = ANY(%s) ORDER BY 2, 1",
                (dates,),
            )
            not_ours = cur.fetchall()
            cur.execute("DELETE FROM business_metrics WHERE hcp_id LIKE %s", (f"hl_{rid}_%",))
            cur.execute("DELETE FROM triggers WHERE trigger_id LIKE %s", (f"trlate_{rid}_%",))
            cur.execute(
                "DELETE FROM patient_journeys WHERE patient_journey_id LIKE %s",
                (f"pj_hl_{rid}_%",),
            )
            cur.execute("DELETE FROM hcp_profiles WHERE hcp_id LIKE %s", (f"hl_{rid}_%",))
    assert not not_ours, (
        f"territory_metrics rows on {dates} were not written by the territory run "
        f"(xid {run_xid}) and were left in place, not deleted: {not_ours}"
    )
    # #2215: the fixture census, the runs and this teardown are separate transactions.
    # With our rows gone, anything the same censuses (the four fixture specs plus the
    # arrival spec the territory test recorded) still reach landed inside a window while
    # this file was writing -- REPORTED as a teardown failure, never deleted.
    require_windows_still_isolated(db_conn, *guard_specs, *state["territory_extra_specs"])


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
                    (f"trlate_{rid}_{suffix}", f"pt_{hcp_id}", hcp_id, ts, ts, arrival),
                )


def _rows(db_conn: Any, rid: str) -> dict:
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT hcp_id, metric_date, triggers_total_count, market_share FROM business_metrics "
                "WHERE hcp_id LIKE %s AND metric_type = 'per_hcp_rollup'",
                (f"hl_{rid}_%",),
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
    first = _run_per_hcp_rollup_impl(arrived_before=FIRST_ARRIVAL_RUN, request_id="late-arrival-1")
    # #2215 (codex r2 HIGH-1): nothing on the affected dates pre-existed (censused), and
    # the second run re-rolls the same (hcp, brand, date) keys with more triggers, so
    # neither arrival run's reconcile deletes anything; a count is a foreign row that
    # landed after the census and is already gone.
    require_no_foreign_reconcile(first)
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
    require_no_foreign_reconcile(second)
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
    after = preview_per_hcp_rollup(EXPLICIT_START, EXPLICIT_END)
    assert (
        after["metric_dates"],
        after["rows_new"],
        after["rows_changed"],
        after["rows_existing"],
    ) == (2, 0, 0, 3)

    # Territory (owner decision #3): the 03:45 run rebuilds both whole dates from the per-HCP rows.
    from src.etl import territory_metrics_etl
    from src.etl.territory_metrics_etl import (
        _run_territory_rollup_impl,
        preview_territory_rollup,
    )

    # The pre-fix default (metric_date in the 24 h before Monday 03:45) reaches Sunday only: no planted date.
    assert (
        preview_territory_rollup("2019-01-13T03:45:00+00:00", TERRITORY_ARRIVAL_RUN)["metric_dates"]
        == 0
    )
    # #2213: census the ARRIVAL run by its OWN selection, now that the rows it selects on
    # exist. The window is the run's (arrived_before less the ETL's lookback); the CTE is
    # the ETL's constant, so the census cannot drift from the run. Then record the
    # selected set from the same CTE for the teardown, and pin it: with the census green,
    # every selected date is fully ours, and ours are exactly the two planted dates.
    arrival_end = datetime.fromisoformat(TERRITORY_ARRIVAL_RUN)
    arrival_start = arrival_end - timedelta(hours=territory_metrics_etl.ARRIVAL_WINDOW_HOURS)
    arrival_spec = territory_arrival_spec(
        test_file=__file__,
        start=arrival_start,
        end=arrival_end,
        hcp_like=f"hl_{rid}_%",
        trigger_like=f"trlate_{rid}_%",
        territory_like=f"T_LATE_{rid}%",
    )
    arrival_params = {
        "start_date": arrival_start,
        "end_date": arrival_end,
        "per_hcp_metric_type": territory_metrics_etl.PER_HCP_METRIC_TYPE,
    }
    (census,) = require_isolated_windows(db_conn, arrival_spec)
    # Positive control: the census COUNTED our rows (3 per-HCP rows on the two dates + the
    # 4 triggers in their lookbacks), so a green verdict is a measurement, not a census
    # that reached nothing.
    assert (census.source_rows_total, census.source_rows_planted) == (7, 7), census

    def _selected_dates() -> set[date]:
        # READ ONLY by construction (codex r2-2): the selection is the ETL's raw CTE.
        return selected_metric_dates(db_conn, arrival_params)

    # Pinned BEFORE it is recorded (codex r1-2): a date that is not one of ours fails here
    # and is never handed to the wholesale teardown.
    selected = _selected_dates()
    assert selected == {TUESDAY, MONDAY}, selected
    planted["territory_selected_dates"].update(selected)
    # Handed to the teardown: it re-reads this selection before deleting, and re-censuses
    # this spec after (#2215).
    planted["territory_arrival_params"] = arrival_params
    planted["territory_extra_specs"].append(arrival_spec)

    territory = _run_territory_rollup_impl(
        arrived_before=TERRITORY_ARRIVAL_RUN, request_id="late-arrival-territory"
    )
    # #2215: no territory row on the selected dates pre-existed (leg 3 was 0).
    require_no_foreign_reconcile(territory)
    # The run's transaction id, read off our own keyed row it just wrote: the teardown
    # deletes by it, and every row on the recorded dates must carry it -- a row that does
    # not was written by someone else and is reported, not swept.
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT xmin::text, created_at FROM territory_metrics "
                " WHERE territory_id = %s AND metric_date = %s",
                (f"T_LATE_{rid}", TUESDAY),
            )
            row = cur.fetchone()
            planted["territory_run_xid"] = row[0] if row else None
            planted["territory_run_created_at"] = row[1] if row else None
    assert territory["status"] == "completed" and territory["selected_by"] == "arrival", territory
    assert planted["territory_run_xid"] is not None
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT territory_id, metric_date, xmin::text FROM territory_metrics "
                " WHERE metric_date = ANY(%s) AND xmin::text <> %s",
                (sorted(planted["territory_selected_dates"]), planted["territory_run_xid"]),
            )
            assert cur.fetchall() == [], "a row on a selected date was not written by the run"
    # (total_trx, active_hcp_count). Tuesday 01-01: a 2 + b 1 delivered. Monday 01-14: a3 delivered;
    # its 30-day lookback still holds a's and b's 01-01 triggers, so both HCPs are active.
    assert _territory(db_conn, rid) == {TUESDAY: (3, 2), MONDAY: (1, 2)}
    # Every date the run wrote for our territory is one the teardown will sweep -- and the
    # selection re-read after the run is still the recorded set, so no date appeared
    # between the recording and the run's own evaluation of the same CTE. One that did
    # would be reported here, not swept.
    assert set(_territory(db_conn, rid)) <= planted["territory_selected_dates"]
    assert _selected_dates() == planted["territory_selected_dates"]
    rebuilt = preview_territory_rollup("2019-01-01", "2019-01-15")
    assert (rebuilt["metric_dates"], rebuilt["rows_new"], rebuilt["rows_changed"]) == (2, 0, 0)

    # Adherence (owner decision #6, revised by codex r13-05): the 03:30 run must keep the
    # known adherence_rate of the open-ended journeys (NULL end date, so it is not
    # computable) and must not touch gap_days at all, which the synthetic DGP owns.
    # After codex r14-02 a preserved row is not written (IS DISTINCT FROM), so the run's
    # own count is 1: the one computable journey. That single row is the evidence that the
    # arrival window selected the planted journeys at all -- a run that missed them would
    # also report the two preserved values untouched.
    from src.etl.patient_adherence_etl import _run_patient_adherence_impl

    adherence = _run_patient_adherence_impl(
        arrived_before="2019-01-14T03:30:00+00:00", request_id="late-arrival-adherence"
    )
    assert adherence["status"] == "completed" and adherence["selected_by"] == "arrival", adherence
    assert adherence["rows_affected"] == 1, adherence
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT patient_journey_id, adherence_rate, gap_days FROM patient_journeys "
                "WHERE patient_journey_id LIKE %s",
                (f"pj_hl_{rid}_%",),
            )
            journeys = {j: (float(rate), gap) for j, rate, gap in cur.fetchall()}
    # The open-ended journeys keep exactly what the fixture wrote. Before the fix, a's would
    # have been (0.0, ...) from the LEAST/GREATEST clamp; the withdrawn gap rewrite would
    # have written 13 for a and 0 for b over the generator's 7. The computable one is
    # recomputed (5 of 10 days) and its gap_days is still the generator's.
    assert journeys[f"pj_{a}"] == (0.42, 7), journeys
    assert journeys[f"pj_{b}"] == (0.42, 7), journeys
    assert journeys[f"pj_hl_{rid}_c"] == (0.5, 7), journeys


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


def test_the_preview_names_the_cohort_data_an_obsolete_row_still_carries(
    db_conn: Any, planted: dict
) -> None:
    """2026-09-21: the reconcile deleted obsolete rows wholesale and the Digital Twin's
    planted cohort data went with them. Before ``--execute`` the preview must say how many
    obsolete rows carry that data, so the operator knows the plant must follow. The row
    planted here is a per_hcp_rollup cell on a day with no trigger, which the recompute
    never produces (obsolete), carrying one planted channel; the fixture deletes it by hcp."""
    from src.etl.business_metrics_per_hcp_etl import preview_per_hcp_rollup

    rid, a = planted["rid"], planted["a"]
    metric_id = f"per_hcp_late_{rid}"
    baseline = preview_per_hcp_rollup(EXPLICIT_START, EXPLICIT_END)
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO business_metrics (
                    metric_id, metric_date, metric_type, brand, region, hcp_id,
                    triggers_delivered_count, triggers_accepted_count, triggers_total_count,
                    market_share, conversion_rate, is_synthetic, email_campaign_count
                ) VALUES (
                    %s, %s, 'per_hcp_rollup', %s::brand_type, 'northeast'::region_type, %s,
                    1, 0, 1, 1.0, 0.0, true, 3
                )
                """,
                (metric_id, date(2019, 1, 3), BRAND, a),
            )
    with_channel = preview_per_hcp_rollup(EXPLICIT_START, EXPLICIT_END)
    # Deltas against a baseline taken before the insert (codex r1): the explicit window is
    # wider than the fixture's arrival-scoped guard, so an unrelated obsolete row in it must
    # not fail this test -- only the row planted here is asserted on.
    assert (
        with_channel["rows_obsolete"] - baseline["rows_obsolete"],
        with_channel["rows_obsolete_with_cohort_data"] - baseline["rows_obsolete_with_cohort_data"],
    ) == (1, 1), (baseline, with_channel)
    # The same row stripped of its cohort data is still obsolete, but no longer cohort data:
    # the count is about the columns, not about obsolescence.
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "UPDATE business_metrics SET email_campaign_count = NULL WHERE metric_id = %s",
                (metric_id,),
            )
    stripped = preview_per_hcp_rollup(EXPLICIT_START, EXPLICIT_END)
    assert (
        stripped["rows_obsolete"] - baseline["rows_obsolete"],
        stripped["rows_obsolete_with_cohort_data"] - baseline["rows_obsolete_with_cohort_data"],
    ) == (1, 0), (baseline, stripped)


def test_an_explicit_window_run_refuses_to_delete_planted_cohort_data_unless_acknowledged(
    db_conn: Any, planted: dict
) -> None:
    """#2210: the preview says how much cohort data an explicit-window reconcile would delete;
    the run itself must refuse to delete it. Plant one obsolete per_hcp_rollup cell carrying a
    channel, run the explicit window: refused, row still there. Acknowledge the loss: the run
    proceeds and the reconcile deletes the row. The fixture deletes anything left by hcp."""
    from src.etl.business_metrics_per_hcp_etl import _run_per_hcp_rollup_impl

    rid, a = planted["rid"], planted["a"]
    metric_id = f"per_hcp_guard_{rid}"
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO business_metrics (
                    metric_id, metric_date, metric_type, brand, region, hcp_id,
                    triggers_delivered_count, triggers_accepted_count, triggers_total_count,
                    market_share, conversion_rate, is_synthetic, sample_volume
                ) VALUES (
                    %s, %s, 'per_hcp_rollup', %s::brand_type, 'northeast'::region_type, %s,
                    1, 0, 1, 1.0, 0.0, true, 0.5
                )
                """,
                (metric_id, date(2019, 1, 4), BRAND, a),
            )

    def _row_exists() -> bool:
        with db_conn:
            with db_conn.cursor() as cur:
                cur.execute("SELECT 1 FROM business_metrics WHERE metric_id = %s", (metric_id,))
                return cur.fetchone() is not None

    refused = _run_per_hcp_rollup_impl(
        start_date=EXPLICIT_START, end_date=EXPLICIT_END, request_id="late-arrival-guard"
    )
    assert refused["status"] == "refused", refused
    assert refused["rows_obsolete_with_cohort_data"] >= 1, refused
    assert (refused["rows_affected"], refused["rows_deleted"]) == (0, 0), refused
    assert _row_exists(), "a refused run must write nothing"

    acknowledged = _run_per_hcp_rollup_impl(
        start_date=EXPLICIT_START,
        end_date=EXPLICIT_END,
        request_id="late-arrival-guard-ack",
        allow_cohort_data_loss=True,
    )
    assert acknowledged["status"] == "completed", acknowledged
    assert acknowledged["cohort_data_loss_acknowledged"] is True
    assert acknowledged["rows_deleted"] >= 1, acknowledged
    assert not _row_exists(), "the acknowledged run's reconcile deletes the obsolete row"


class _AfterStatement:
    """A real psycopg2 connection whose cursor calls ``after()`` once, right after the run
    executes ``statement`` and before its next statement -- the seam a concurrent writer
    lands in. Everything else (``with conn:`` commit/rollback, ``close``) is the real
    connection's, so the run under test is the production function, not a re-enactment of
    its SQL (codex r2 on #2212)."""

    def __init__(self, conn: Any, statement: str, after: Any) -> None:
        self._conn, self._statement, self._after = conn, statement, after
        self.fired = 0

    def cursor(self) -> "_AfterStatementCursor":
        return _AfterStatementCursor(self, self._conn.cursor())

    def __enter__(self) -> "_AfterStatement":
        self._conn.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._conn.__exit__(*exc)

    def close(self) -> None:
        self._conn.close()


class _AfterStatementCursor:
    def __init__(self, owner: _AfterStatement, cur: Any) -> None:
        self._owner, self._cur = owner, cur

    def execute(self, sql: Any, params: Any = None) -> None:
        self._cur.execute(sql, params)
        if sql is self._owner._statement:
            self._owner.fired += 1
            self._owner._after()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cur, name)

    def __enter__(self) -> "_AfterStatementCursor":
        self._cur.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._cur.__exit__(*exc)


def _run_with_writer_after_preflight(etl: Any, after: Any, **kwargs: Any) -> tuple[dict, int]:
    """Run the real explicit-window function with ``after()`` committed from another
    connection immediately after its preflight. Returns (result, times the hook fired)."""
    from unittest.mock import patch

    hooked = _AfterStatement(etl._connect_to_db(), etl.PREFLIGHT_COHORT_DATA_SQL, after)
    with patch.object(etl, "_connect_to_db", return_value=hooked):
        result = etl._run_per_hcp_rollup_impl(
            start_date=EXPLICIT_START, end_date=EXPLICIT_END, **kwargs
        )
    return result, hooked.fired


def test_the_run_snapshot_hides_a_row_planted_after_its_preflight(
    db_conn: Any, planted: dict
) -> None:
    """codex r1 on #2212: one transaction is not one snapshot under READ COMMITTED. Prove on
    the real database, through the real run: another connection commits a cohort-bearing
    obsolete row right after the run's preflight counted zero. The run completes, its
    reconcile spares the row (invisible to the REPEATABLE READ snapshot), the row survives,
    and the next run refuses on it. Cleans its own plant so later tests start at zero."""
    from src.etl import business_metrics_per_hcp_etl as etl

    rid, a = planted["rid"], planted["a"]
    metric_id = f"per_hcp_snap_{rid}"

    def plant_after_preflight() -> None:
        with db_conn:
            with db_conn.cursor() as other:
                other.execute(
                    """
                    INSERT INTO business_metrics (
                        metric_id, metric_date, metric_type, brand, region, hcp_id,
                        triggers_delivered_count, triggers_accepted_count,
                        triggers_total_count, market_share, conversion_rate,
                        is_synthetic, rep_training_score
                    ) VALUES (
                        %s, %s, 'per_hcp_rollup', %s::brand_type,
                        'northeast'::region_type, %s, 1, 0, 1, 1.0, 0.0, true, 0.7
                    )
                    """,
                    (metric_id, date(2019, 1, 5), BRAND, a),
                )

    def _row_exists() -> bool:
        with db_conn:
            with db_conn.cursor() as cur:
                cur.execute("SELECT 1 FROM business_metrics WHERE metric_id = %s", (metric_id,))
                return cur.fetchone() is not None

    try:
        completed, fired = _run_with_writer_after_preflight(
            etl, plant_after_preflight, request_id="late-arrival-snapshot"
        )
        assert fired == 1
        assert completed["status"] == "completed", completed
        # the preflight counted zero: the payload is the pre-#2210 one
        assert "rows_obsolete_with_cohort_data" not in completed, completed
        assert completed["rows_deleted"] == 0, completed
        assert _row_exists(), "the row planted after the snapshot must survive the reconcile"
        refused = etl._run_per_hcp_rollup_impl(
            start_date=EXPLICIT_START, end_date=EXPLICIT_END, request_id="late-arrival-snapshot-2"
        )
        assert refused["status"] == "refused", refused
        assert metric_id in refused["rows_obsolete_with_cohort_data_sample"], refused
        assert _row_exists()
    finally:
        with db_conn:
            with db_conn.cursor() as cur:
                cur.execute("DELETE FROM business_metrics WHERE metric_id = %s", (metric_id,))


def test_a_row_changed_after_the_preflight_fails_the_run_and_commits_nothing(
    db_conn: Any, planted: dict
) -> None:
    """codex r2 on #2212: the snapshot's other edge. The run's INSERT ... ON CONFLICT DO
    UPDATE touches (a, TUESDAY), a row the earlier phases wrote; another connection updates
    that row right after the preflight. PostgreSQL will not let a REPEATABLE READ
    transaction update a row changed since its snapshot: the real run ends ``failed`` with
    a serialization failure, nothing it wrote is committed, and the concurrent update is
    what remains in every planted row."""
    from src.etl import business_metrics_per_hcp_etl as etl

    rid, a = planted["rid"], planted["a"]
    target = etl._build_metric_id(a, BRAND, TUESDAY)
    before = _rows(db_conn, rid)
    assert (a, TUESDAY) in before, before
    original_count, share = before[(a, TUESDAY)]
    assert original_count != 99

    def touch_after_preflight() -> None:
        with db_conn:
            with db_conn.cursor() as other:
                other.execute(
                    "UPDATE business_metrics SET triggers_total_count = 99 WHERE metric_id = %s",
                    (target,),
                )
                assert other.rowcount == 1, "the upsert target must exist before the run"

    try:
        failed, fired = _run_with_writer_after_preflight(
            etl, touch_after_preflight, request_id="late-arrival-concurrent-update"
        )
        assert fired == 1
        assert failed["status"] == "failed", failed
        assert "serialize" in failed["error"], failed
        assert (failed["rows_affected"], failed["rows_deleted"]) == (0, 0), failed
        # nothing of the run landed: the concurrent update is the only change
        assert _rows(db_conn, rid) == {**before, (a, TUESDAY): (99, share)}
    finally:
        with db_conn:
            with db_conn.cursor() as cur:
                cur.execute(
                    "UPDATE business_metrics SET triggers_total_count = %s WHERE metric_id = %s",
                    (original_count, target),
                )
