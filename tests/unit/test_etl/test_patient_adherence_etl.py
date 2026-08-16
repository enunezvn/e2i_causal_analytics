"""Unit tests for ``src.etl.patient_adherence_etl``.

These tests do not touch a real database. They verify:

* ``_compute_adherence_rate`` mirrors the SQL clamp-and-divide for every
  edge case (zero span, NULL inputs, ratio overflow).
* The SQL string contains the load-bearing CTEs, parameter placeholders,
  the ``LEAST/GREATEST`` clamp, the ``LAG``-based gap computation, the
  ``UPDATE...FROM`` shape, the in-subquery ``patient_id IN (...)`` predicate
  (so the planner filters before LAG runs), and the explicit "refill_count
  NOT set" comment.
* ``_run_patient_adherence_impl`` orchestrates the connect/execute/commit/
  close flow correctly with mocks, and surfaces ``status`` / ``rows_affected``
  faithfully. The Celery wrapper ``run_patient_adherence_rollup`` is a thin
  one-liner over this and is exercised in the integration test.
* The shared helpers ``_resolve_db_connection_string``, ``_connect_to_db``
  and ``_resolve_window`` are re-exported here for backward compatibility
  (the canonical tests live in ``test_common.py`` since extraction in
  6B-infra-2b fix-up).

Behaviour-level assertions about exact gap counts and adherence ratios on
real synthetic data live in
``tests/integration/test_patient_adherence_etl_integration.py``.
"""

from __future__ import annotations

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

    def test_has_two_named_ctes(self) -> None:
        sql = etl.UPDATE_PATIENT_ADHERENCE_SQL
        assert "journey_adherence AS" in sql
        assert "patient_gaps AS" in sql

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

    def test_gap_days_uses_lag_window_function(self) -> None:
        """gap_days computed via LAG(...) over patient-partitioned trigger
        timestamps — single-event patients yield NULL → COALESCE to 0."""
        sql = etl.UPDATE_PATIENT_ADHERENCE_SQL
        normalised = re.sub(r"\s+", " ", sql)
        assert "LAG(trigger_timestamp)" in normalised
        assert "PARTITION BY patient_id" in normalised
        assert "ORDER BY trigger_timestamp" in normalised

    def test_gap_days_coalesces_to_zero_for_single_event(self) -> None:
        """Plan: single-event patients (gap_days=0). Ensured by COALESCE."""
        normalised = re.sub(r"\s+", " ", etl.UPDATE_PATIENT_ADHERENCE_SQL)
        # COALESCE(MAX(...)::INTEGER, 0)
        assert re.search(
            r"COALESCE\(\s*MAX\(",
            normalised,
        ), "missing COALESCE around MAX(...) for single-event patients"
        assert re.search(r",\s*0\s*\)\s*AS\s+gap_days", normalised), (
            "COALESCE must default to 0 for single-event patients"
        )

    def test_gap_days_uses_epoch_division_for_seconds_to_days(self) -> None:
        """EPOCH/86400 conversion preserves sub-day precision then truncates,
        whereas EXTRACT(DAY FROM interval) drops hours+minutes."""
        normalised = re.sub(r"\s+", " ", etl.UPDATE_PATIENT_ADHERENCE_SQL)
        assert re.search(
            r"EXTRACT\(\s*EPOCH FROM gap\s*\)::BIGINT\s*/\s*86400",
            normalised,
        ), "missing EPOCH/86400 conversion"

    def test_update_uses_left_join_to_keep_no_trigger_journeys(self) -> None:
        """Journeys with no triggers must still get adherence_rate updated;
        gap_days falls out as NULL via the LEFT JOIN."""
        sql = etl.UPDATE_PATIENT_ADHERENCE_SQL
        assert "LEFT JOIN patient_gaps pg" in sql

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

    def test_filters_trigger_window_on_trigger_timestamp(self) -> None:
        """Window scope on triggers is trigger_timestamp [start, end)."""
        sql = etl.UPDATE_PATIENT_ADHERENCE_SQL
        assert re.search(r"trigger_timestamp\s*>=\s*%\(start_date\)s", sql), (
            "missing trigger_timestamp >= start_date filter"
        )
        assert re.search(r"trigger_timestamp\s*<\s*%\(end_date\)s", sql), (
            "missing trigger_timestamp < end_date filter"
        )

    def test_patient_id_predicate_lives_inside_lag_subquery(self) -> None:
        """The ``patient_id IN (SELECT patient_id FROM patient_journeys ...)``
        predicate must sit INSIDE the inner LAG-bearing subquery, alongside
        the trigger_timestamp window — not on the outer ``WHERE`` of the
        ``patient_gaps`` CTE.

        PostgreSQL is not guaranteed to push an outer predicate through a
        window function; pinning the placement here keeps the planner free
        to filter trigger rows BEFORE LAG runs (latency-safe on large
        ``triggers`` tables).
        """
        sql = etl.UPDATE_PATIENT_ADHERENCE_SQL
        normalised = re.sub(r"\s+", " ", sql)

        # The predicate must appear once.
        assert "patient_id IN ( SELECT patient_id FROM patient_journeys" in normalised, (
            "missing patient_id IN (SELECT patient_id FROM patient_journeys ...) predicate"
        )

        # And it must appear BEFORE the closing ``) lag_view`` of the inner
        # subquery — i.e. inside it, not on the outer ``patient_gaps`` WHERE.
        predicate_idx = normalised.index("patient_id IN ( SELECT patient_id FROM patient_journeys")
        lag_view_close_idx = normalised.index(") lag_view")
        assert predicate_idx < lag_view_close_idx, (
            "patient_id IN (...) predicate must live inside the inner "
            "LAG-bearing subquery (before the `) lag_view` closer), not on "
            "the outer patient_gaps WHERE"
        )

        # Belt-and-braces: the segment between ``FROM triggers`` and
        # ``) lag_view`` should contain both the trigger_timestamp window AND
        # the patient_id IN predicate (proving they're co-located inside the
        # inner subquery).
        from_triggers_idx = normalised.index("FROM triggers")
        inner_subquery = normalised[from_triggers_idx:lag_view_close_idx]
        assert "trigger_timestamp >= %(start_date)s" in inner_subquery
        assert "patient_id IN ( SELECT patient_id FROM patient_journeys" in inner_subquery

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

    def test_set_clause_updates_adherence_rate_and_gap_days(self) -> None:
        """The two columns this ETL owns are present in SET; both come from
        the joined CTE columns."""
        normalised = re.sub(r"\s+", " ", etl.UPDATE_PATIENT_ADHERENCE_SQL)
        assert re.search(r"adherence_rate\s*=\s*ja\.adherence_rate", normalised), (
            "missing adherence_rate = ja.adherence_rate in SET"
        )
        assert re.search(r"gap_days\s*=\s*pg\.gap_days", normalised), (
            "missing gap_days = pg.gap_days in SET"
        )


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


def test_impl_default_window_is_24h() -> None:
    """No dates supplied -> window defaults to 24 hours ending now(UTC)."""
    conn = _make_mock_conn(rowcount=1)

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_patient_adherence_impl()

    assert result["status"] == "completed"
    start = datetime.fromisoformat(result["window_start"])
    end = datetime.fromisoformat(result["window_end"])
    delta_hours = (end - start).total_seconds() / 3600.0
    assert delta_hours == pytest.approx(etl.DEFAULT_WINDOW_HOURS, abs=1e-6)


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
