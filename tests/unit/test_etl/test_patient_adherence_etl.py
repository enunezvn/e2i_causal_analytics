"""Unit tests for ``src.etl.patient_adherence_etl``.

These tests do not touch a real database. They verify:

* ``_compute_adherence_rate`` mirrors the SQL clamp-and-divide for every
  edge case (zero span, NULL inputs, ratio overflow).
* The SQL string contains the load-bearing CTE, parameter placeholders,
  the ``CASE`` that returns NULL for an uncomputable ratio, the
  ``LEAST/GREATEST`` clamp, the ``COALESCE`` that keeps a stored value, the
  ``IS DISTINCT FROM`` predicate that skips unchanged rows, the
  ``UPDATE...FROM`` shape, and the explicit "refill_count NOT set" comment.
  Canonical TRx lane, owner decision #6: six tests that pinned the ``gap_days``
  trigger walk were deleted with it — the synthetic DGP owns that column, so
  there is nothing here for this ETL to assert about it beyond
  ``test_the_etl_does_not_write_gap_days_and_does_not_read_triggers``.
* ``_run_patient_adherence_impl`` orchestrates the connect/execute/commit/
  close flow correctly with mocks, and surfaces ``status`` / ``rows_affected``
  faithfully. The Celery wrapper ``run_patient_adherence_rollup`` is a thin
  one-liner over this and is exercised in the integration test.
* The shared helpers ``_resolve_db_connection_string``, ``_connect_to_db``
  and ``_resolve_window`` are re-exported here for backward compatibility
  (the canonical tests live in ``test_common.py`` since extraction in
  6B-infra-2b fix-up).

Behaviour-level assertions about adherence ratios on real synthetic data — and
about the planted ``gap_days`` this ETL must leave alone — live in
``tests/integration/test_patient_adherence_etl_integration.py``.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Importing the module triggers `from src.workers.celery_app import celery_app`,
# which only requires standard library + celery (already installed). No DB
# connection happens at import time.
from src.etl import _common
from src.etl import patient_adherence_etl as etl

# =============================================================================
# _common helper re-exports
# =============================================================================


def test_helpers_are_re_exported_from_common() -> None:
    """The three shared helpers must be accessible as module attributes on
    ``patient_adherence_etl`` so existing test imports keep working.

    The detailed behaviour tests live in ``test_common.py``; this guards the
    re-export wiring so a future cleanup that drops the shim breaks loudly.
    """
    assert etl._resolve_db_connection_string is _common._resolve_db_connection_string
    assert etl._connect_to_db is _common._connect_to_db
    assert etl._resolve_window is _common._resolve_window


# =============================================================================
# _compute_adherence_rate — mirrors SQL semantics exactly
# =============================================================================


def test_compute_adherence_rate_typical_partial_coverage() -> None:
    """A 10-day journey duration over a 20-day span -> 0.5."""
    assert etl._compute_adherence_rate(10, 20) == pytest.approx(0.5)


def test_compute_adherence_rate_full_coverage() -> None:
    """Duration equals span -> 1.0 (the upper clamp boundary)."""
    assert etl._compute_adherence_rate(20, 20) == pytest.approx(1.0)


def test_compute_adherence_rate_clamps_above_one() -> None:
    """Ratio > 1 (e.g. journey_duration_days exceeds the span) clamps to 1.0."""
    assert etl._compute_adherence_rate(30, 20) == pytest.approx(1.0)


def test_compute_adherence_rate_clamps_below_zero() -> None:
    """Negative duration (data error) clamps to 0.0; never returns negative."""
    assert etl._compute_adherence_rate(-5, 20) == pytest.approx(0.0)


def test_compute_adherence_rate_zero_span_returns_none() -> None:
    """journey_end_date == journey_start_date -> span 0 -> None.

    This is the "zero-duration journeys (adherence_rate=NULL)" case the plan
    calls out.
    """
    assert etl._compute_adherence_rate(5, 0) is None


def test_compute_adherence_rate_none_duration_returns_none() -> None:
    """journey_duration_days IS NULL propagates to None."""
    assert etl._compute_adherence_rate(None, 20) is None


def test_compute_adherence_rate_none_span_returns_none() -> None:
    """journey_end_date IS NULL (so span is None) propagates to None."""
    assert etl._compute_adherence_rate(10, None) is None


def test_compute_adherence_rate_both_none_returns_none() -> None:
    """Both args None -> None (defensive)."""
    assert etl._compute_adherence_rate(None, None) is None


# =============================================================================
# SQL string structure
# =============================================================================


class TestSQLShape:
    """Pin the load-bearing structure of UPDATE_PATIENT_ADHERENCE_SQL.

    Lean on substring presence rather than full SQL parse — a parser would
    be overkill for the small set of guarantees this query needs to keep
    across refactors.
    """

    def test_has_one_named_cte(self) -> None:
        """Changed deliberately (codex r13-05): the patient_gaps CTE is gone with the
        gap_days computation the synthetic DGP owns.

        Self-reference check: none. Both assertions name a literal CTE string, so the
        test cannot agree with a wrong implementation — and the negative half is what
        gives it teeth, since the positive half alone passes on today's two-CTE SQL.
        """
        sql = etl.UPDATE_PATIENT_ADHERENCE_SQL
        assert "journey_adherence AS" in sql
        assert "patient_gaps AS" not in sql

    def test_uses_named_parameters(self) -> None:
        """psycopg2 named-param style %()s is required so the params dict
        in run_patient_adherence_rollup matches."""
        sql = etl.UPDATE_PATIENT_ADHERENCE_SQL
        assert "%(start_date)s" in sql
        assert "%(end_date)s" in sql

    def test_clamps_adherence_rate_with_least_greatest(self) -> None:
        """adherence_rate is clamped to [0, 1] via LEAST/GREATEST.

        Whitespace is normalised so multi-line indented form still matches.
        """
        normalised = re.sub(r"\s+", " ", etl.UPDATE_PATIENT_ADHERENCE_SQL)
        # LEAST(1.0::NUMERIC, GREATEST(0.0::NUMERIC, ...))
        assert re.search(
            r"LEAST\(\s*1\.0::NUMERIC,\s*GREATEST\(\s*0\.0::NUMERIC,",
            normalised,
        ), "missing LEAST(1.0, GREATEST(0.0, ...)) clamp"

    def test_uses_nullif_to_guard_zero_span(self) -> None:
        """Division by zero span guarded via NULLIF((end - start), 0)."""
        normalised = re.sub(r"\s+", " ", etl.UPDATE_PATIENT_ADHERENCE_SQL)
        assert re.search(
            r"NULLIF\(\s*\(pj\.journey_end_date\s*-\s*pj\.journey_start_date\)::NUMERIC,\s*0\s*\)",
            normalised,
        ), "missing NULLIF guard around the journey span subtraction"

    def test_filters_journey_window_on_journey_start_date(self) -> None:
        """Window scope on patient_journeys is journey_start_date — avoids
        rewriting old static journeys on every daily run."""
        sql = etl.UPDATE_PATIENT_ADHERENCE_SQL
        assert re.search(r"pj\.journey_start_date\s*>=\s*%\(start_date\)s", sql), (
            "missing pj.journey_start_date >= start_date filter"
        )
        assert re.search(r"pj\.journey_start_date\s*<\s*%\(end_date\)s", sql), (
            "missing pj.journey_start_date < end_date filter"
        )

    def test_refill_count_left_null_with_documenting_comment(self) -> None:
        """refill_count is intentionally NOT in the SET list — and the SQL
        carries a -- comment naming the missing source so future readers
        know why."""
        sql = etl.UPDATE_PATIENT_ADHERENCE_SQL
        # Strip line comments so column-name presence in commentary doesn't
        # trip the SET-list assertion.
        stripped = re.sub(r"--[^\n]*", "", sql)
        # Slice from the SET keyword to the end of the UPDATE statement.
        set_block = stripped.split("SET", 1)[1]
        assert "refill_count" not in set_block, (
            "refill_count must not be in the UPDATE SET list (no refill source in canonical schema)"
        )
        # The original SQL still mentions it in a comment so the omission
        # is self-documenting.
        assert "refill_count" in sql
        assert "refill_reminder" in sql, "module docs the missing trigger_type by name"

    def test_set_clause_updates_only_adherence_rate_and_coalesces_it(self) -> None:
        """Changed deliberately: adherence_rate is coalesced to the stored value so an
        input this ETL cannot compute never erases one, and gap_days left the SET clause
        entirely (codex r13-05: the synthetic DGP owns it).

        Self-reference check: none. The COALESCE regex names the exact expression and the
        gap_days half is a negative over comment-stripped SQL, so neither can be satisfied
        by the implementation it is meant to constrain.
        """
        normalised = re.sub(r"\s+", " ", etl.UPDATE_PATIENT_ADHERENCE_SQL)
        assert re.search(
            r"adherence_rate\s*=\s*COALESCE\(ja\.adherence_rate, pj\.adherence_rate\)",
            normalised,
        ), "missing adherence_rate = COALESCE(ja.adherence_rate, pj.adherence_rate) in SET"
        assert "gap_days" not in re.sub(r"--[^\n]*", "", etl.UPDATE_PATIENT_ADHERENCE_SQL)


# =============================================================================
# _run_patient_adherence_impl — orchestration
# =============================================================================


def _make_mock_conn(rowcount: int = 7) -> MagicMock:
    """Build a mock psycopg2 connection that exits its `with` block cleanly."""
    cur = MagicMock()
    cur.rowcount = rowcount
    cur.execute = MagicMock()
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cur)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.close = MagicMock()
    return conn


def test_impl_completed_path() -> None:
    """Happy path: connect, execute, commit, close; status=completed."""
    conn = _make_mock_conn(rowcount=42)

    with patch.object(etl, "_connect_to_db", return_value=conn) as connect:
        result = etl._run_patient_adherence_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "completed"
    assert result["rows_affected"] == 42
    assert result["window_start"].startswith("2024-01-01T00:00:00")
    assert result["window_end"].startswith("2024-01-02T00:00:00")

    connect.assert_called_once()
    cur = conn.cursor.return_value
    args, _ = cur.execute.call_args
    assert args[0] is etl.UPDATE_PATIENT_ADHERENCE_SQL
    params = args[1]
    assert isinstance(params["start_date"], datetime)
    assert isinstance(params["end_date"], datetime)
    # Only the two window params -- no metric_id_prefix / metric_type because
    # this ETL is a straight UPDATE (idempotent on the patient_journey_id PK).
    assert set(params.keys()) == {"start_date", "end_date"}
    conn.close.assert_called_once()


def test_impl_no_data_path() -> None:
    """rowcount==0 reports status=no_data with rows_affected=0."""
    conn = _make_mock_conn(rowcount=0)

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_patient_adherence_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "no_data"
    assert result["rows_affected"] == 0


def test_impl_db_failure_returns_failed() -> None:
    """A connection / execute exception is caught and surfaced as failed."""
    with patch.object(etl, "_connect_to_db", side_effect=RuntimeError("boom")):
        result = etl._run_patient_adherence_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "failed"
    assert result["error"] == "boom"
    assert result["rows_affected"] == 0


def test_impl_invalid_window_returns_failed() -> None:
    """Inverted window short-circuits before any DB connect."""
    with patch.object(etl, "_connect_to_db") as connect:
        result = etl._run_patient_adherence_impl(
            start_date="2024-01-02T00:00:00Z",
            end_date="2024-01-01T00:00:00Z",
        )

    assert result["status"] == "failed"
    assert "must be strictly before" in result["error"]
    connect.assert_not_called()


def test_impl_default_window_is_the_arrival_window() -> None:
    """No dates supplied -> window defaults to 24 hours ending now(UTC)."""
    conn = _make_mock_conn(rowcount=1)

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_patient_adherence_impl()

    assert result["status"] == "completed"
    assert result["selected_by"] == "arrival"
    start = datetime.fromisoformat(result["window_start"])
    end = datetime.fromisoformat(result["window_end"])
    delta_hours = (end - start).total_seconds() / 3600.0
    # ⚠ SELF-REFERENTIAL, stated up front (the 22A/22B lesson): this equality moves on
    # both sides if ARRIVAL_WINDOW_HOURS is reverted, so it cannot catch a wrong constant.
    # This test's own teeth are `selected_by`; the VALUE is anchored by the literal
    # window_start in test_impl_scheduled_run_uses_the_arrival_variant and by 22A's
    # arithmetic pin.
    assert delta_hours == pytest.approx(etl.ARRIVAL_WINDOW_HOURS, abs=1e-6)


def test_impl_closes_conn_even_on_error() -> None:
    """If execute() raises, the connection is still closed."""
    conn = _make_mock_conn(rowcount=0)
    cur = conn.cursor.return_value
    cur.execute.side_effect = RuntimeError("query exploded")

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_patient_adherence_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "failed"
    conn.close.assert_called_once()


def test_impl_passes_request_id_through() -> None:
    """The request_id arg is forwarded but does not change behaviour."""
    conn = _make_mock_conn(rowcount=1)

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_patient_adherence_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
            request_id="celery-task-xyz",
        )

    assert result["status"] == "completed"


# =============================================================================
# Celery task wrapper
# =============================================================================


def test_celery_task_delegates_to_impl() -> None:
    """The Celery task ``run_patient_adherence_rollup`` is a thin shim over
    ``_run_patient_adherence_impl`` — verify it forwards args + the
    request id."""
    sentinel_result = {"status": "completed", "rows_affected": 3}
    with patch.object(etl, "_run_patient_adherence_impl", return_value=sentinel_result) as impl:
        async_result = etl.run_patient_adherence_rollup.apply(
            kwargs={
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-02T00:00:00Z",
            },
        )

    assert async_result.successful()
    assert async_result.result == sentinel_result
    impl.assert_called_once()
    call_kwargs = impl.call_args.kwargs
    assert call_kwargs["start_date"] == "2024-01-01T00:00:00Z"
    assert call_kwargs["end_date"] == "2024-01-02T00:00:00Z"
    assert isinstance(call_kwargs["request_id"], str)
    assert call_kwargs["request_id"]


# =============================================================================
# Celery registration sanity
# =============================================================================


def test_task_is_registered_with_expected_name() -> None:
    """The Celery task name string is what the beat schedule references."""
    assert (
        etl.run_patient_adherence_rollup.name
        == "src.etl.patient_adherence_etl.run_patient_adherence_rollup"
    )


def test_beat_schedule_entry_present() -> None:
    """The daily beat entry routes the task to the analytics queue.

    #1645: the cadence is a wall-clock crontab, not the old bare ``86400.0``
    interval — an interval is measured from ``last_run_at``, which every deploy
    reset, so a 24h entry never became due. Slot map in
    ``src/workers/celery_app.py``.
    """
    from celery.schedules import crontab

    from src.workers.celery_app import celery_app

    entry = celery_app.conf.beat_schedule.get("patient-adherence-rollup")
    assert entry is not None, "beat schedule entry missing"
    assert entry["task"] == "src.etl.patient_adherence_etl.run_patient_adherence_rollup"
    assert entry["schedule"] == crontab(hour=3, minute=30)
    assert entry["options"]["queue"] == "analytics"


# =============================================================================
# Preserve-on-undefined + arrival selection (owner decision #6 2026-09-15)
# =============================================================================


def test_the_sql_returns_null_when_it_cannot_compute_rather_than_zero() -> None:
    """LEAST/GREATEST IGNORE NULLs in Postgres (measured: LEAST(1.0, NULL) = 1.0,
    GREATEST(0.0, NULL) = 0.0), so the clamp alone yielded 0.0 for the NULL end dates
    every journey has. An explicit CASE now produces the NULL the docstring promises."""
    for sql in (etl.UPDATE_PATIENT_ADHERENCE_SQL, etl.UPDATE_PATIENT_ADHERENCE_BY_ARRIVAL_SQL):
        derivation = sql.split("journey_adherence AS (", 1)[1].split("UPDATE patient_journeys", 1)[
            0
        ]
        assert re.search(r"CASE\s+WHEN", derivation), "no CASE guard on the undefined inputs"
        for guard in ("pj.journey_end_date IS NULL", "pj.journey_duration_days IS NULL"):
            assert guard in derivation, guard
        assert re.search(r"THEN\s+NULL", derivation)


def test_the_update_never_overwrites_a_value_it_could_not_compute() -> None:
    for sql in (etl.UPDATE_PATIENT_ADHERENCE_SQL, etl.UPDATE_PATIENT_ADHERENCE_BY_ARRIVAL_SQL):
        # Strip -- comments first: the refill_count rationale lives inside the SET clause.
        code = "\n".join(line.split("--", 1)[0] for line in sql.splitlines())
        set_clause = code.split("SET ", 1)[1].split("FROM journey_adherence", 1)[0]
        assert "adherence_rate = COALESCE(ja.adherence_rate, pj.adherence_rate)" in set_clause
        assert "refill_count" not in set_clause


def test_the_update_skips_the_rows_it_would_not_change() -> None:
    """codex r14-02: patient_journeys has a BEFORE UPDATE trigger
    (update_patient_journeys_timestamp -> update_updated_at) that sets updated_at = NOW()
    unconditionally, so a value-preserving UPDATE is still a write. Every live journey has a
    NULL journey_end_date, so without this predicate every scheduled run would touch ~153 real
    rows and change no value."""
    for sql in (etl.UPDATE_PATIENT_ADHERENCE_SQL, etl.UPDATE_PATIENT_ADHERENCE_BY_ARRIVAL_SQL):
        code = "\n".join(line.split("--", 1)[0] for line in sql.splitlines())
        where = re.sub(r"\s+", " ", code.split("WHERE pj.patient_journey_id", 1)[1])
        assert (
            "COALESCE(ja.adherence_rate, pj.adherence_rate) IS DISTINCT FROM pj.adherence_rate"
            in where
        ), "the UPDATE must skip rows whose value would not change"
        # IS DISTINCT FROM, not <>: both sides are nullable and NULL <> NULL is NULL.
        assert not re.search(r"pj\.adherence_rate\s*(<>|!=)\s*", where)


def test_the_skip_predicate_is_the_set_expression_verbatim() -> None:
    """The guard is only correct if it tests exactly what the SET writes; a drift between the
    two would either skip a real change or write an unchanged row."""
    for sql in (etl.UPDATE_PATIENT_ADHERENCE_SQL, etl.UPDATE_PATIENT_ADHERENCE_BY_ARRIVAL_SQL):
        code = "\n".join(line.split("--", 1)[0] for line in sql.splitlines())
        written = code.split("SET adherence_rate =", 1)[1].split("FROM journey_adherence", 1)[0]
        tail = code.split("WHERE pj.patient_journey_id", 1)[1]
        guarded = tail.split("AND ", 1)[1].split(" IS DISTINCT FROM", 1)[0]
        assert (
            re.sub(r"\s+", " ", written).strip().rstrip(",") == re.sub(r"\s+", " ", guarded).strip()
        )


def test_the_etl_does_not_write_gap_days_and_does_not_read_triggers() -> None:
    """codex r13-05: the synthetic DGP owns gap_days and snaps it to low_gap_180d
    (gap_days <= 30 <=> low_gap_180d = 1, measured exact on 26,600 live rows). This ETL's
    trigger-interval quantity is a different concept, and 13,272 journeys have a
    single-trigger patient where it would compute 0 and break that contract."""
    for sql in (etl.UPDATE_PATIENT_ADHERENCE_SQL, etl.UPDATE_PATIENT_ADHERENCE_BY_ARRIVAL_SQL):
        code = "\n".join(line.split("--", 1)[0] for line in sql.splitlines())
        assert "gap_days" not in code
        assert "triggers" not in code and "patient_gaps" not in code and "LAG(" not in code


def test_the_scheduled_variant_selects_journeys_by_arrival() -> None:
    body = etl.UPDATE_PATIENT_ADHERENCE_BY_ARRIVAL_SQL.split("journey_adherence AS (", 1)[1].split(
        "UPDATE patient_journeys", 1
    )[0]
    assert re.search(r"pj\.created_at\s*>=\s*%\(start_date\)s", body)
    assert re.search(r"pj\.created_at\s*<\s*%\(end_date\)s", body)
    assert "journey_start_date >=" not in body


def test_the_explicit_variant_keeps_the_journey_start_date_window() -> None:
    body = etl.UPDATE_PATIENT_ADHERENCE_SQL.split("journey_adherence AS (", 1)[1].split(
        "UPDATE patient_journeys", 1
    )[0]
    assert re.search(r"pj\.journey_start_date\s*>=\s*%\(start_date\)s", body)
    assert "created_at" not in body


def test_the_variants_share_everything_after_the_selection() -> None:
    def after(sql: str) -> str:
        return sql.split("UPDATE patient_journeys", 1)[1]

    assert after(etl.UPDATE_PATIENT_ADHERENCE_SQL) == after(
        etl.UPDATE_PATIENT_ADHERENCE_BY_ARRIVAL_SQL
    )


def test_the_arrival_window_matches_the_other_two_rollups() -> None:
    """⚠ EQUALITY IS NOT A VALUE PIN (the 22A/22B lesson): this passes if all three
    constants drift together. The value is anchored by
    test_business_metrics_per_hcp_etl.py::test_the_arrival_window_spans_a_weekly_batch_cycle_plus_margin
    (arithmetic against the literals 7, 24, 6) and by the literal window_start in
    test_impl_scheduled_run_uses_the_arrival_variant below. Deleting either turns this
    into a proxy."""
    from src.etl import business_metrics_per_hcp_etl as per_hcp
    from src.etl import territory_metrics_etl as territory
    from src.workers.celery_app import celery_app

    assert (
        etl.ARRIVAL_WINDOW_HOURS == per_hcp.ARRIVAL_WINDOW_HOURS == territory.ARRIVAL_WINDOW_HOURS
    )
    beat = celery_app.conf.beat_schedule["patient-adherence-rollup"]["schedule"]
    assert (beat.hour, beat.minute) == ({3}, {30})  # between the per-HCP 03:15 and territory 03:45


def test_impl_scheduled_run_uses_the_arrival_variant() -> None:
    conn = _make_mock_conn(rowcount=3)
    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_patient_adherence_impl(arrived_before="2026-09-14T03:30:00+00:00")
    args, _ = conn.cursor.return_value.execute.call_args
    assert args[0] is etl.UPDATE_PATIENT_ADHERENCE_BY_ARRIVAL_SQL
    assert result["selected_by"] == "arrival"
    # A LITERAL: 03:30 - 174 h. One of the two anchors on the window constant's VALUE.
    assert result["window_start"].startswith("2026-09-06T21:30:00")


def test_impl_explicit_dates_keep_the_journey_start_date_window() -> None:
    conn = _make_mock_conn(rowcount=3)
    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_patient_adherence_impl(start_date="2026-05-01", end_date="2026-09-16")
    args, _ = conn.cursor.return_value.execute.call_args
    assert args[0] is etl.UPDATE_PATIENT_ADHERENCE_SQL
    assert result["selected_by"] == "journey_start_date"


def test_impl_refuses_arrived_before_with_explicit_dates() -> None:
    with patch.object(etl, "_connect_to_db") as connect:
        result = etl._run_patient_adherence_impl(
            start_date="2026-05-01",
            end_date="2026-09-16",
            arrived_before="2026-09-14T03:30:00+00:00",
        )
    assert result["status"] == "failed" and "arrived_before" in result["error"]
    connect.assert_not_called()


def test_a_committed_run_logs_one_self_sufficient_completion_line(caplog) -> None:  # noqa: ANN001
    """codex r14-05: certification evidence must say the run FINISHED, and must say it on one
    line — the start line may have rotated away."""
    conn = _make_mock_conn(rowcount=3)
    with caplog.at_level(logging.INFO, logger=etl.logger.name):
        with patch.object(etl, "_connect_to_db", return_value=conn):
            etl._run_patient_adherence_impl(arrived_before="2026-09-14T03:30:00+00:00")
    committed = [
        r.getMessage() for r in caplog.records if "adherence rollup committed" in r.getMessage()
    ]
    assert len(committed) == 1, committed
    assert "selected_by=arrival" in committed[0]
    assert "rows_affected=3" in committed[0]
    assert "2026-09-06T21:30:00" in committed[0] and "2026-09-14T03:30:00" in committed[0]


def test_a_run_that_changes_nothing_still_proves_it_committed(caplog) -> None:  # noqa: ANN001
    """After r14-02 the healthy scheduled outcome is rows_affected=0. That must not read as
    'the ETL did not run', or the r14-02 fix would blind the certification."""
    conn = _make_mock_conn(rowcount=0)
    with caplog.at_level(logging.INFO, logger=etl.logger.name):
        with patch.object(etl, "_connect_to_db", return_value=conn):
            result = etl._run_patient_adherence_impl(arrived_before="2026-09-14T03:30:00+00:00")
    assert result["status"] == "no_data"
    committed = [
        r.getMessage() for r in caplog.records if "adherence rollup committed" in r.getMessage()
    ]
    assert len(committed) == 1 and "rows_affected=0" in committed[0]


def test_a_failure_after_the_start_line_logs_no_completion(caplog) -> None:  # noqa: ANN001
    """The negative control: the start line alone must never be accepted as evidence."""
    with caplog.at_level(logging.INFO, logger=etl.logger.name):
        with patch.object(etl, "_connect_to_db", side_effect=RuntimeError("no route to host")):
            result = etl._run_patient_adherence_impl(arrived_before="2026-09-14T03:30:00+00:00")
    messages = [r.getMessage() for r in caplog.records]
    assert result["status"] == "failed"
    assert any("Starting per-patient adherence rollup" in m for m in messages)
    assert not any("adherence rollup committed" in m for m in messages)
