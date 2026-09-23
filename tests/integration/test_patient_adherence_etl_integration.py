"""Integration tests for ``src.etl.patient_adherence_etl``.

Exercises the full ETL against a real PostgreSQL instance reached via
``SUPABASE_DB_URL``. Tests are skipped when the env var is unset so CI
unit-only runs stay green.

Synthetic dataset
-----------------
Three patients with deliberately distinct trigger patterns. **Every journey is
planted with a known ``gap_days`` and, where the point is preservation, a known
``adherence_rate``** — this ETL must not move either except where it can compute
the ratio (canonical TRx lane, owner decision #6; see "gap_days ownership" below):

* ``patient_normal``: 5 triggers spread evenly across the window with one
  intentionally large 14-day gap. Closed journey with full coverage
  (adherence clamps to 1.0), planted ``adherence_rate = 0.11`` so a computable
  row is PROVED to be rewritten, and planted ``gap_days = 41``.
* ``patient_single``: exactly 1 trigger, planted ``gap_days = 42``. The
  withdrawn trigger walk would have written 0 here (``LAG`` NULL ->
  ``COALESCE(..., 0)``), which is what 13,272 live journeys would have suffered.
* ``patient_zero_duration``: closed journey with ``journey_end_date ==
  journey_start_date`` -- the ratio is NOT computable. Planted
  ``adherence_rate = 0.33`` and ``gap_days = 43``; both must survive the run.
  (Before the fix this row was written to 0.0: ``NULLIF(0, 0)`` is NULL, and
  ``GREATEST(0.0, NULL)`` is 0.0 because Postgres ``LEAST``/``GREATEST`` IGNORE
  NULLs rather than propagating them.)

The planted gap values (41/42/43) are deliberately values the withdrawn
computation could never produce for these patients (it would have produced
14/0/4 from the triggers above). That is what gives the preservation assertions
teeth: reintroducing the write changes them, whereas a value that happened to
coincide would not.

All rows are isolated by a unique ``test_run_id`` prefix on every primary
key so cleanup is deterministic.

gap_days ownership
------------------
This ETL stops writing ``gap_days`` entirely (owner decision #6, revised by
codex r13-05). The synthetic DGP owns the column and snaps it to the recoverable
binary ``low_gap_180d`` (``gap_days <= 30`` iff ``low_gap_180d = 1``, exact on
26,600 live rows with zero violations). The triggers planted below are therefore
no longer INPUT to the ETL — they are the counter-evidence: the ETL must ignore
them. It no longer reads the ``triggers`` table at all.

Assertions
----------

* Per-patient ``adherence_rate`` matches the expected clamp-and-divide
  result where the ratio is computable, and the PLANTED value is preserved
  where it is not.
* Per-patient ``gap_days`` is still the planted value -- the guard against
  someone reintroducing the trigger-interval write.
* Re-running the ETL yields the same per-row values (idempotency).

Run gate
--------
Two env vars are required:

* ``SUPABASE_DB_URL`` -- Postgres URL pointing at a DB that has migration
  033 applied.
* ``E2I_DB_INTEGRATION=1`` -- explicit opt-in; mirrors 6B-infra-2a's gate.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

# psycopg2 is a transitive dep of the Supabase client and the ETL itself,
# but unit-only environments may install without it. Skip the whole module
# rather than ImportError when the binary is absent.
from tests.integration._prod_write_guard import (
    adherence_spec,
    planted_prefix,
    require_isolated_windows,
    require_windows_still_isolated,
)

psycopg2 = pytest.importorskip("psycopg2")

# Module-level skip: developers must opt in AND have a reachable Postgres URL
# before the suite executes. Mirrors the 6B-infra-2a integration-test gate.
pytestmark = pytest.mark.skipif(
    not (os.getenv("SUPABASE_DB_URL") and os.getenv("E2I_DB_INTEGRATION") == "1"),
    reason=(
        "SUPABASE_DB_URL and/or E2I_DB_INTEGRATION not set; integration "
        "test requires real Postgres + explicit opt-in. Run with "
        "E2I_DB_INTEGRATION=1 once your DB has migration 033 applied."
    ),
)


@pytest.fixture(scope="module")
def db_conn() -> Any:
    """Open a single psycopg2 connection for the module's tests."""
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def test_run_id() -> str:
    """Unique prefix per pytest run so parallel suites do not collide."""
    return uuid.uuid4().hex[:10]


@pytest.fixture(scope="module")
def synthetic_dataset(db_conn: Any, test_run_id: str) -> dict:
    """Insert three patients with distinct trigger/journey patterns; tear
    down at module end.

    Returns a dict with the IDs and the ``(start_date, end_date)`` window
    spanning the synthetic data. The teardown runs inside ``try/finally``
    around the ``yield`` so an interrupted pytest still cleans synthetic
    rows (mirrors I4 from 6B-infra-2a).
    """
    base_date = date(2024, 1, 1)
    end_dt = datetime(2024, 1, 31, tzinfo=timezone.utc)
    start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Prod-write guard (owner-approved, 2026-09-17): the explicit-window variant UPDATEs
    # every journey with journey_start_date in [start, end), planted or not, so an
    # unplanted journey here is both a derivation source and a write target.
    guard_specs = (
        adherence_spec(
            test_file=__file__,
            start=start_dt,
            end=end_dt,
            journey_like=planted_prefix(f"pj_{test_run_id}_"),
        ),
    )
    require_isolated_windows(db_conn, *guard_specs)

    # Patient A ("normal"): 30-day journey, full coverage, 5 triggers with a
    # 14-day gap inserted between trigger 2 and trigger 3.
    pat_a = f"pat_{test_run_id}_a_normal"
    journey_a = f"pj_{test_run_id}_a_normal"
    a_journey_start = base_date
    a_journey_end = base_date + timedelta(days=29)
    a_duration_days = 30
    a_span_days = 29  # journey_end - journey_start (date diff)
    # Triggers: days 0, 5, 19 (14d gap from day 5), 22, 27
    a_trigger_offsets = [0, 5, 19, 22, 27]
    # Planted: a wrong-but-plausible ratio the ETL CAN compute, so the run must
    # overwrite it (the positive control for the IS DISTINCT FROM predicate).
    a_planted_adherence = 0.11
    a_planted_gap_days = 41  # the withdrawn walk would have written 14 (19 - 5)

    # Patient B ("single"): 10-day journey, full coverage, 1 trigger.
    pat_b = f"pat_{test_run_id}_b_single"
    journey_b = f"pj_{test_run_id}_b_single"
    b_journey_start = base_date
    b_journey_end = base_date + timedelta(days=9)
    b_duration_days = 10
    b_span_days = 9
    b_trigger_offsets = [3]
    b_planted_adherence = None  # computable (10 / 9 -> clamps to 1.0)
    b_planted_gap_days = 42  # the withdrawn walk would have written 0 (LAG NULL)

    # Patient C ("zero duration"): journey_end == journey_start (-> span 0
    # -> NOT computable). 2 triggers separated by 4 days.
    pat_c = f"pat_{test_run_id}_c_zero"
    journey_c = f"pj_{test_run_id}_c_zero"
    c_journey_start = base_date
    c_journey_end = base_date  # span = 0
    c_duration_days = 0
    c_trigger_offsets = [2, 6]
    # Planted: the ETL cannot compute this one, so it must keep the stored value.
    c_planted_adherence = 0.33
    c_planted_gap_days = 43  # the withdrawn walk would have written 4 (6 - 2)

    try:
        with db_conn:
            with db_conn.cursor() as cur:
                # Insert journeys, each carrying a PLANTED adherence_rate / gap_days
                # so the assertions can tell "preserved" from "recomputed".
                for journey_id, patient_id, j_start, j_end, duration, adherence, gap in (
                    (
                        journey_a,
                        pat_a,
                        a_journey_start,
                        a_journey_end,
                        a_duration_days,
                        a_planted_adherence,
                        a_planted_gap_days,
                    ),
                    (
                        journey_b,
                        pat_b,
                        b_journey_start,
                        b_journey_end,
                        b_duration_days,
                        b_planted_adherence,
                        b_planted_gap_days,
                    ),
                    (
                        journey_c,
                        pat_c,
                        c_journey_start,
                        c_journey_end,
                        c_duration_days,
                        c_planted_adherence,
                        c_planted_gap_days,
                    ),
                ):
                    cur.execute(
                        """
                        INSERT INTO patient_journeys (
                            patient_journey_id, patient_id,
                            journey_start_date, journey_end_date,
                            journey_duration_days,
                            journey_stage, journey_status, brand,
                            adherence_rate, gap_days
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            'diagnosis'::journey_stage_type,
                            'active'::journey_status_type,
                            'Remibrutinib'::brand_type,
                            %s, %s
                        )
                        ON CONFLICT (patient_journey_id) DO NOTHING
                        """,
                        (journey_id, patient_id, j_start, j_end, duration, adherence, gap),
                    )

                # Insert triggers.
                counter = 0
                trigger_plans = (
                    (pat_a, a_trigger_offsets),
                    (pat_b, b_trigger_offsets),
                    (pat_c, c_trigger_offsets),
                )
                for patient_id, offsets in trigger_plans:
                    for day_offset in offsets:
                        counter += 1
                        trigger_id = f"tr_{test_run_id}_{counter:06d}"
                        ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc) + timedelta(
                            days=day_offset
                        )
                        cur.execute(
                            """
                            INSERT INTO triggers (
                                trigger_id, patient_id, trigger_timestamp,
                                trigger_type, brand_id
                            ) VALUES (%s, %s, %s, 'engagement_gap', 'UNKNOWN')
                            ON CONFLICT (trigger_id) DO NOTHING
                            """,
                            (trigger_id, patient_id, ts),
                        )

        yield {
            "test_run_id": test_run_id,
            "patients": {
                "a_normal": {
                    "patient_id": pat_a,
                    "journey_id": journey_a,
                    "expected_adherence_rate": a_duration_days
                    / a_span_days,  # > 1 -> clamps to 1.0
                    "planted_adherence_rate": a_planted_adherence,
                    "planted_gap_days": a_planted_gap_days,
                },
                "b_single": {
                    "patient_id": pat_b,
                    "journey_id": journey_b,
                    "expected_adherence_rate": b_duration_days
                    / b_span_days,  # > 1 -> clamps to 1.0
                    "planted_adherence_rate": b_planted_adherence,
                    "planted_gap_days": b_planted_gap_days,
                },
                "c_zero": {
                    "patient_id": pat_c,
                    "journey_id": journey_c,
                    # Not computable (zero span) -> the planted value is kept.
                    "expected_adherence_rate": c_planted_adherence,
                    "planted_adherence_rate": c_planted_adherence,
                    "planted_gap_days": c_planted_gap_days,
                },
            },
            "start_date": start_dt,
            "end_date": end_dt,
        }
    finally:
        # Teardown: delete in reverse FK order. Wrapped in try/finally so an
        # interrupted pytest still leaves the DB clean (mirrors the I4
        # carry-over from 6B-infra-2a).
        with db_conn:
            with db_conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM triggers WHERE trigger_id LIKE %s ESCAPE '\\'",
                    (planted_prefix(f"tr_{test_run_id}_"),),
                )
                cur.execute(
                    "DELETE FROM patient_journeys WHERE patient_journey_id LIKE %s ESCAPE '\\'",
                    (planted_prefix(f"pj_{test_run_id}_"),),
                )
        # #2215: the census, the runs and this teardown are separate transactions. With
        # our rows gone, anything the same census still reaches landed inside the window
        # while this file was writing -- REPORTED as a teardown failure, never deleted.
        require_windows_still_isolated(db_conn, *guard_specs)


def _fetch_journey_metrics(db_conn: Any, journey_id: str) -> tuple[Any, Any, Any]:
    """Read back the one column this ETL writes, plus the two it must not.

    ``refill_count`` is documented-NULL and ``gap_days`` is the synthetic DGP's
    (owner decision #6), so both are read here as guards rather than as outputs.
    """
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT adherence_rate, refill_count, gap_days
              FROM patient_journeys
             WHERE patient_journey_id = %s
            """,
            (journey_id,),
        )
        row = cur.fetchone()
    assert row is not None, f"journey {journey_id} not found"
    return row[0], row[1], row[2]


def test_adherence_rate_clamps_to_one_for_normal_patient(
    db_conn: Any, synthetic_dataset: dict
) -> None:
    """Patient A: duration 30 / span 29 = 1.034 -> clamps to 1.0.

    Also the positive control for the ``IS DISTINCT FROM`` skip predicate: the
    journey was planted at 0.11, so a run that writes nothing would leave 0.11
    here and fail. Preservation must not become "never writes".
    """
    from src.etl.patient_adherence_etl import _run_patient_adherence_impl

    result = _run_patient_adherence_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-test-a",
    )
    assert result["status"] == "completed", f"ETL failed: {result}"
    assert result["selected_by"] == "journey_start_date", result

    a = synthetic_dataset["patients"]["a_normal"]
    adherence, refill, gap = _fetch_journey_metrics(db_conn, a["journey_id"])
    assert adherence == pytest.approx(1.0, abs=1e-9), (
        f"adherence_rate should clamp to 1.0 (planted {a['planted_adherence_rate']}); "
        f"got {adherence}"
    )
    # refill_count is intentionally left NULL (see module docstring).
    assert refill is None
    # Converted (owner decision #6): this once asserted the ETL's own trigger-walk
    # value (14). It now asserts the ETL LEFT the planted value alone -- the guard
    # against reintroducing a write of a column the synthetic DGP owns.
    assert gap == a["planted_gap_days"], (
        f"gap_days must be left at the planted {a['planted_gap_days']}; got {gap}. "
        "A 14 here means the withdrawn trigger walk is back."
    )


def test_adherence_rate_preserved_for_zero_duration_journey(
    db_conn: Any, synthetic_dataset: dict
) -> None:
    """Patient C: journey_end_date == journey_start_date -> span 0 -> NOT
    computable -> the planted value survives.

    Converted (owner decision #6). This test used to assert NULL and would have
    FAILED against the shipped code: ``NULLIF(0, 0)`` is NULL and
    ``GREATEST(0.0, NULL)`` is 0.0, because Postgres ``LEAST``/``GREATEST`` IGNORE
    NULLs. It never ran -- the module is skip-gated -- so the defect stood. With a
    planted 0.33 the assertion is now the one that matters: an uncomputable input
    must not erase a known value. A 0.0 here is the original bug returning.
    """
    from src.etl.patient_adherence_etl import _run_patient_adherence_impl

    _run_patient_adherence_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-test-c",
    )

    c = synthetic_dataset["patients"]["c_zero"]
    adherence, refill, gap = _fetch_journey_metrics(db_conn, c["journey_id"])
    assert adherence == pytest.approx(c["planted_adherence_rate"], abs=1e-9), (
        f"adherence_rate must stay at the planted {c['planted_adherence_rate']} for a "
        f"zero-duration journey; got {adherence}"
    )
    assert refill is None
    assert gap == c["planted_gap_days"], (
        f"gap_days must be left at the planted {c['planted_gap_days']}; got {gap}. "
        "A 4 here means the withdrawn trigger walk is back."
    )


def test_gap_days_untouched_for_single_event_patient(db_conn: Any, synthetic_dataset: dict) -> None:
    """Patient B: exactly one trigger.

    Converted (owner decision #6, codex r13-05). This asserted ``gap_days == 0``,
    the value the withdrawn ``LAG`` walk produced for a single-trigger patient --
    and 13,272 live journeys have exactly that shape, so re-running that
    computation would have written 0 over the generator's values and broken the
    ``gap_days <= 30`` iff ``low_gap_180d = 1`` contract on every such row whose
    binary is 0. The assertion is inverted: the planted value must survive.
    """
    from src.etl.patient_adherence_etl import _run_patient_adherence_impl

    _run_patient_adherence_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-test-b",
    )

    b = synthetic_dataset["patients"]["b_single"]
    adherence, refill, gap = _fetch_journey_metrics(db_conn, b["journey_id"])
    # Patient B has duration > span (10 / 9) so adherence clamps to 1.0.
    assert adherence == pytest.approx(1.0, abs=1e-9)
    assert refill is None
    assert gap == b["planted_gap_days"], (
        f"gap_days must be left at the planted {b['planted_gap_days']}; got {gap}. "
        "A 0 here is the single-trigger zeroing this ETL must never do again."
    )


def test_idempotent_rerun_yields_identical_values(db_conn: Any, synthetic_dataset: dict) -> None:
    """Running the ETL twice produces the same per-row values for every
    journey in the window. The UPDATE is naturally idempotent on the
    journey PK."""
    from src.etl.patient_adherence_etl import _run_patient_adherence_impl

    _run_patient_adherence_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="idempotency-1",
    )

    first_snapshot = {
        name: _fetch_journey_metrics(db_conn, payload["journey_id"])
        for name, payload in synthetic_dataset["patients"].items()
    }

    _run_patient_adherence_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="idempotency-2",
    )

    second_snapshot = {
        name: _fetch_journey_metrics(db_conn, payload["journey_id"])
        for name, payload in synthetic_dataset["patients"].items()
    }

    assert first_snapshot == second_snapshot, "second run produced different per-row values"
