"""Integration tests for ``src.etl.patient_adherence_etl``.

Exercises the full ETL against a real PostgreSQL instance reached via
``SUPABASE_DB_URL``. Tests are skipped when the env var is unset so CI
unit-only runs stay green.

Synthetic dataset
-----------------
Three patients with deliberately distinct trigger patterns:

* ``patient_normal``: 5 triggers spread evenly across the window with one
  intentionally large gap. Closed journey with full coverage (adherence ~ 1).
* ``patient_single``: exactly 1 trigger -- covers the "single-event
  patients (gap_days=0)" plan requirement.
* ``patient_zero_duration``: closed journey with ``journey_end_date ==
  journey_start_date`` -- covers the "zero-duration journeys
  (adherence_rate=NULL)" plan requirement. Has 2 triggers so gap_days is
  exercised independently.

All rows are isolated by a unique ``test_run_id`` prefix on every primary
key so cleanup is deterministic.

Assertions
----------

* Per-patient ``adherence_rate`` matches the expected clamp-and-divide
  result, including NULL for zero-duration journeys.
* Per-patient ``gap_days`` matches the expected max-consecutive-gap (in
  whole days), including 0 for the single-event patient.
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

    # Patient B ("single"): 10-day journey, full coverage, 1 trigger.
    pat_b = f"pat_{test_run_id}_b_single"
    journey_b = f"pj_{test_run_id}_b_single"
    b_journey_start = base_date
    b_journey_end = base_date + timedelta(days=9)
    b_duration_days = 10
    b_span_days = 9
    b_trigger_offsets = [3]

    # Patient C ("zero duration"): journey_end == journey_start (-> span 0
    # -> NULLIF -> NULL adherence). 2 triggers separated by 4 days.
    pat_c = f"pat_{test_run_id}_c_zero"
    journey_c = f"pj_{test_run_id}_c_zero"
    c_journey_start = base_date
    c_journey_end = base_date  # span = 0
    c_duration_days = 0
    c_trigger_offsets = [2, 6]

    try:
        with db_conn:
            with db_conn.cursor() as cur:
                # Insert journeys.
                for journey_id, patient_id, j_start, j_end, duration in (
                    (journey_a, pat_a, a_journey_start, a_journey_end, a_duration_days),
                    (journey_b, pat_b, b_journey_start, b_journey_end, b_duration_days),
                    (journey_c, pat_c, c_journey_start, c_journey_end, c_duration_days),
                ):
                    cur.execute(
                        """
                        INSERT INTO patient_journeys (
                            patient_journey_id, patient_id,
                            journey_start_date, journey_end_date,
                            journey_duration_days,
                            journey_stage, journey_status, brand
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            'diagnosis'::journey_stage_type,
                            'active'::journey_status_type,
                            'Remibrutinib'::brand_type
                        )
                        ON CONFLICT (patient_journey_id) DO NOTHING
                        """,
                        (journey_id, patient_id, j_start, j_end, duration),
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
                        ts = datetime(
                            2024, 1, 1, 12, 0, tzinfo=timezone.utc
                        ) + timedelta(days=day_offset)
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
                    "expected_adherence_rate": a_duration_days / a_span_days,  # > 1 -> clamps to 1.0
                    "expected_gap_days": 14,  # 19 - 5
                },
                "b_single": {
                    "patient_id": pat_b,
                    "journey_id": journey_b,
                    "expected_adherence_rate": b_duration_days / b_span_days,  # > 1 -> clamps to 1.0
                    "expected_gap_days": 0,  # single-event
                },
                "c_zero": {
                    "patient_id": pat_c,
                    "journey_id": journey_c,
                    "expected_adherence_rate": None,  # zero span -> NULL
                    "expected_gap_days": 4,  # 6 - 2
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
                    "DELETE FROM triggers WHERE trigger_id LIKE %s",
                    (f"tr_{test_run_id}_%",),
                )
                cur.execute(
                    "DELETE FROM patient_journeys WHERE patient_journey_id LIKE %s",
                    (f"pj_{test_run_id}_%",),
                )


def _fetch_journey_metrics(
    db_conn: Any, journey_id: str
) -> tuple[Any, Any, Any]:
    """Read back the three columns this ETL writes for a given journey."""
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
    """Patient A: duration 30 / span 29 = 1.034 -> clamps to 1.0."""
    from src.etl.patient_adherence_etl import _run_patient_adherence_impl

    result = _run_patient_adherence_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-test-a",
    )
    assert result["status"] == "completed", f"ETL failed: {result}"

    a = synthetic_dataset["patients"]["a_normal"]
    adherence, refill, gap = _fetch_journey_metrics(db_conn, a["journey_id"])
    assert adherence == pytest.approx(1.0, abs=1e-9), (
        f"adherence_rate should clamp to 1.0; got {adherence}"
    )
    # refill_count is intentionally left NULL (see module docstring).
    assert refill is None
    assert gap == a["expected_gap_days"], (
        f"gap_days mismatch for normal patient: "
        f"expected {a['expected_gap_days']}, got {gap}"
    )


def test_adherence_rate_null_for_zero_duration_journey(
    db_conn: Any, synthetic_dataset: dict
) -> None:
    """Patient C: journey_end_date == journey_start_date -> span 0 ->
    NULLIF -> adherence_rate NULL. The plan's "zero-duration journeys
    (adherence_rate=NULL)" requirement."""
    from src.etl.patient_adherence_etl import _run_patient_adherence_impl

    _run_patient_adherence_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-test-c",
    )

    c = synthetic_dataset["patients"]["c_zero"]
    adherence, refill, gap = _fetch_journey_metrics(db_conn, c["journey_id"])
    assert adherence is None, (
        f"adherence_rate must be NULL for zero-duration journey; got {adherence}"
    )
    assert refill is None
    assert gap == c["expected_gap_days"], (
        f"gap_days mismatch for zero-duration patient: "
        f"expected {c['expected_gap_days']}, got {gap}"
    )


def test_gap_days_zero_for_single_event_patient(
    db_conn: Any, synthetic_dataset: dict
) -> None:
    """Patient B: exactly one trigger -> LAG returns NULL -> COALESCE -> 0.
    The plan's "single-event patients (gap_days=0)" requirement."""
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
    assert gap == 0, f"single-event gap_days must be 0; got {gap}"


def test_idempotent_rerun_yields_identical_values(
    db_conn: Any, synthetic_dataset: dict
) -> None:
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

    assert first_snapshot == second_snapshot, (
        "second run produced different per-row values"
    )
