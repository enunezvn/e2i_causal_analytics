"""Unit tests for ``src.etl.territory_metrics_etl``.

These tests do not touch a real database. They verify:

* The SQL string contains the load-bearing CTEs, parameter placeholders,
  the four expected aggregations (total_trx / total_nrx / active_hcp_count
  / covered_lives), the 30-day INTERVAL for active_hcp_count, the
  ``ON CONFLICT (territory_id, metric_date)`` clause, and that
  ``market_potential`` / ``resource_allocation_score`` are written as
  explicit NULL on INSERT (with migration 033 dropping the legacy
  NOT NULL DEFAULT 0) but stay out of the ON CONFLICT SET clause so
  pre-existing 031 random seeds survive re-runs untouched.
* ``_run_territory_rollup_impl`` orchestrates the connect/execute/commit/
  close flow correctly with mocks, and surfaces ``status`` /
  ``rows_affected`` faithfully. The Celery wrapper ``run_territory_rollup``
  is a thin one-liner over this and is exercised in the integration test.
* The shared helpers ``_resolve_db_connection_string``, ``_connect_to_db``
  and ``_resolve_window`` are re-exported here for backward compatibility
  (the canonical tests live in ``test_common.py``).

Behaviour-level assertions about exact territorial sums on real synthetic
data live in
``tests/integration/test_territory_metrics_etl_integration.py``.
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
from src.etl import territory_metrics_etl as etl

# =============================================================================
# _common helper re-exports
# =============================================================================


def test_helpers_are_re_exported_from_common() -> None:
    """The three shared helpers must be accessible as module attributes on
    ``territory_metrics_etl`` so existing test imports keep working.

    The detailed behaviour tests live in ``test_common.py``; this guards the
    re-export wiring so a future cleanup that drops the shim breaks loudly.
    """
    assert etl._resolve_db_connection_string is _common._resolve_db_connection_string
    assert etl._connect_to_db is _common._connect_to_db
    assert etl._resolve_window is _common._resolve_window


# =============================================================================
# SQL string structure
# =============================================================================


class TestSQLShape:
    """Pin the load-bearing structure of INSERT_TERRITORY_ROLLUP_SQL.

    These tests intentionally lean on substring presence rather than full
    SQL parse — a parser would be overkill for the small set of guarantees
    this query needs to keep across refactors.
    """

    def test_has_named_ctes(self) -> None:
        """All CTEs the impl relies on are present by name."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert "metric_dates AS" in sql
        assert "territories AS" in sql
        assert "territory_dates AS" in sql
        assert "per_hcp_in_territory AS" in sql
        assert "active_hcp_per_territory_date AS" in sql
        assert "territory_hcp_volume AS" in sql

    def test_uses_named_parameters(self) -> None:
        """psycopg2 named-param style %()s is required so the params dict
        in run_territory_rollup matches."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert "%(start_date)s" in sql
        assert "%(end_date)s" in sql
        assert "%(per_hcp_metric_type)s" in sql

    def test_aggregations_present(self) -> None:
        """The four aggregations the plan asks for are all in the SQL."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        normalised = re.sub(r"\s+", " ", sql)
        # SUM(triggers_delivered_count) / SUM(triggers_accepted_count) per territory+date.
        assert re.search(
            r"SUM\(\s*COALESCE\(\s*bm\.triggers_delivered_count,\s*0\s*\)\s*\)::BIGINT\s+AS\s+total_trx",
            normalised,
        ), "missing SUM(triggers_delivered_count) AS total_trx"
        assert re.search(
            r"SUM\(\s*COALESCE\(\s*bm\.triggers_accepted_count,\s*0\s*\)\s*\)::BIGINT\s+AS\s+total_nrx",
            normalised,
        ), "missing SUM(triggers_accepted_count) AS total_nrx"
        # COUNT(DISTINCT hcp_id) AS active_hcp_count.
        assert re.search(
            r"COUNT\(DISTINCT\s+t\.hcp_id\)::BIGINT\s+AS\s+active_hcp_count",
            normalised,
        ), "missing COUNT(DISTINCT hcp_id) AS active_hcp_count"
        # SUM(total_patient_volume) AS covered_lives.
        assert re.search(
            r"SUM\(\s*COALESCE\(\s*total_patient_volume,\s*0\s*\)\s*\)::BIGINT\s+AS\s+covered_lives",
            normalised,
        ), "missing SUM(total_patient_volume) AS covered_lives"

    def test_active_hcp_uses_30_day_interval(self) -> None:
        """active_hcp_count uses a 30-day backward-looking window from
        metric_date (inclusive of metric_date itself)."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        normalised = re.sub(r"\s+", " ", sql)
        # Lower bound: trigger_timestamp >= metric_date - INTERVAL '30 days'.
        assert re.search(
            r"t\.trigger_timestamp\s*>=\s*td\.metric_date\s*-\s*INTERVAL\s*'30 days'",
            normalised,
        ), "missing 30-day backward-looking lower bound"
        # Upper bound: trigger_timestamp < metric_date + INTERVAL '1 day'.
        # (Inclusive of metric_date.)
        assert re.search(
            r"t\.trigger_timestamp\s*<\s*td\.metric_date\s*\+\s*INTERVAL\s*'1 day'",
            normalised,
        ), "missing inclusive-of-metric_date upper bound"

    def test_per_hcp_aggregation_filters_to_per_hcp_rollup_rows(self) -> None:
        """``per_hcp_in_territory`` joins business_metrics restricted to
        per-HCP rollup rows only (excluding the legacy aggregate
        per-(brand, region) rows that keep hcp_id IS NULL)."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        normalised = re.sub(r"\s+", " ", sql)
        assert "bm.metric_type = %(per_hcp_metric_type)s" in normalised
        assert "bm.hcp_id IS NOT NULL" in normalised

    def test_covered_lives_sources_from_total_patient_volume(self) -> None:
        """covered_lives comes from hcp_profiles.total_patient_volume per
        the plan; the CTE name + the SUM expression both reflect that."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert "FROM hcp_profiles" in sql
        # The SUM expression appears in territory_hcp_volume CTE.
        assert "total_patient_volume" in sql

    def test_on_conflict_uses_pk(self) -> None:
        """ON CONFLICT targets (territory_id, metric_date) -- matches the
        territory_metrics PK exactly. No md5-hashing trick needed."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert "ON CONFLICT (territory_id, metric_date) DO UPDATE SET" in sql

    def test_on_conflict_updates_only_real_aggregates(self) -> None:
        """The SET clause covers only the four real aggregates; market_
        potential and resource_allocation_score are intentionally OMITTED
        so existing values (e.g. migration 031's random seed) survive."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        on_conflict_block = sql.split("ON CONFLICT (territory_id, metric_date) DO UPDATE SET", 1)[1]

        # Required: the four real aggregates.
        for col in ("total_trx", "total_nrx", "active_hcp_count", "covered_lives"):
            pattern = rf"{re.escape(col)}\s*=\s*EXCLUDED\.{re.escape(col)}"
            assert re.search(pattern, on_conflict_block), f"missing UPDATE SET clause for {col}"

        # Forbidden: the two columns that must remain untouched. Use the
        # narrower SET-clause-only block so we don't trip on the OMIT
        # comment in the INSERT column list.
        for col in ("market_potential", "resource_allocation_score"):
            pattern = rf"{re.escape(col)}\s*=\s*EXCLUDED\."
            assert not re.search(pattern, on_conflict_block), (
                f"{col} must NOT be in the ON CONFLICT SET clause "
                "(real Reltio/Veeva source not yet integrated)"
            )

    def test_market_potential_and_resource_allocation_score_written_as_null(
        self,
    ) -> None:
        """Plan: NULL otherwise (NOT random). Migration 033 drops the legacy
        NOT NULL DEFAULT 0 that 031 had set, so writing NULL explicitly is
        well-defined and matches the spec ("NULL otherwise, NOT random")
        even on fresh databases that don't already carry 031's random
        seed. The ON CONFLICT SET clause excluding them is asserted by
        ``test_on_conflict_updates_only_real_aggregates``; this test
        focuses on the INSERT side.

        Strip line comments before checking, since the explanatory comments
        in the column list and SELECT block naming both columns must NOT
        count as column-list matches.
        """
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        # Strip everything from `--` to end-of-line so comments don't
        # spuriously match column names.
        stripped = re.sub(r"--[^\n]*", "", sql)
        # Slice the INSERT INTO ... ( column-list ) block.
        insert_clause = stripped.split("INSERT INTO territory_metrics", 1)[1].split("SELECT", 1)[0]
        assert "market_potential" in insert_clause, (
            "market_potential must be in the INSERT column list "
            "(migration 033 dropped the NOT NULL so NULL is well-defined)"
        )
        assert "resource_allocation_score" in insert_clause, (
            "resource_allocation_score must be in the INSERT column list "
            "(migration 033 dropped the NOT NULL so NULL is well-defined)"
        )

        # The SELECT block following INSERT INTO must write
        # CAST(NULL AS DOUBLE PRECISION) for both columns -- explicit NULL,
        # not table default, not COALESCE'd zero. Anchor on the
        # INSERT-INTO-... slice so we don't trip on the SELECT inside
        # the upstream WITH ... metric_dates CTE.
        post_insert = stripped.split("INSERT INTO territory_metrics", 1)[1]
        normalised_post_insert = re.sub(r"\s+", " ", post_insert)
        assert re.search(
            r"CAST\(NULL AS DOUBLE PRECISION\)\s+AS\s+market_potential",
            normalised_post_insert,
        ), "market_potential must be SELECTed as CAST(NULL AS DOUBLE PRECISION)"
        assert re.search(
            r"CAST\(NULL AS DOUBLE PRECISION\)\s+AS\s+resource_allocation_score",
            normalised_post_insert,
        ), "resource_allocation_score must be SELECTed as CAST(NULL AS DOUBLE PRECISION)"

        # The original SQL still names the missing real source so future
        # readers see why these columns are NULL.
        assert "Reltio" in sql or "Veeva" in sql, "module SQL must name the missing real source"

    def test_run_window_filters_metric_dates(self) -> None:
        """The metric_dates CTE filters on the [start_date, end_date)
        half-open run window."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert re.search(r"bm\.metric_date\s*>=\s*%\(start_date\)s::DATE", sql), (
            "missing metric_date >= start_date filter"
        )
        assert re.search(r"bm\.metric_date\s*<\s*%\(end_date\)s::DATE", sql), (
            "missing metric_date < end_date filter"
        )

    def test_left_joins_anchor_at_territory_dates(self) -> None:
        """All three aggregate CTEs are LEFT JOINed onto territory_dates so
        a territory with no business_metrics for the day still gets a row
        (with COALESCE'd zeros)."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        normalised = re.sub(r"\s+", " ", sql)
        assert "LEFT JOIN per_hcp_in_territory" in normalised
        assert "LEFT JOIN active_hcp_per_territory_date" in normalised
        assert "LEFT JOIN territory_hcp_volume" in normalised


# =============================================================================
# _run_territory_rollup_impl — orchestration
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
        result = etl._run_territory_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "completed"
    assert result["rows_affected"] == 42
    assert result["window_start"].startswith("2024-01-01T00:00:00")
    assert result["window_end"].startswith("2024-01-02T00:00:00")

    connect.assert_called_once()
    cur = conn.cursor.return_value
    # ORDER-pinned, not last-call-pinned (codex r13-08): the impl now runs the reconcile
    # DELETE in the SAME transaction, so `call_args` — the LAST call — became the reconcile.
    # Measured on a bare MagicMock in 22A: after a second execute, `call_args[0][0] is
    # INSERT` is False, so the old assertion would have failed because of a statement it was
    # never about. This keeps its subject and additionally pins the order.
    args, _ = cur.execute.call_args_list[0]
    assert args[0] is etl.INSERT_TERRITORY_ROLLUP_SQL
    assert cur.execute.call_args_list[1].args[0] is etl.RECONCILE_TERRITORY_ROLLUP_SQL
    params = args[1]
    assert params["per_hcp_metric_type"] == etl.PER_HCP_METRIC_TYPE
    assert isinstance(params["start_date"], datetime)
    assert isinstance(params["end_date"], datetime)
    # The ETL only needs three params -- no metric_id_prefix because there's
    # no md5 trick (PK is the natural key already).
    assert set(params.keys()) == {"start_date", "end_date", "per_hcp_metric_type"}
    conn.close.assert_called_once()


def test_impl_no_data_path() -> None:
    """rowcount==0 reports status=no_data with rows_affected=0."""
    conn = _make_mock_conn(rowcount=0)

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_territory_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "no_data"
    assert result["rows_affected"] == 0


def test_impl_db_failure_returns_failed() -> None:
    """A connection / execute exception is caught and surfaced as failed."""
    with patch.object(etl, "_connect_to_db", side_effect=RuntimeError("boom")):
        result = etl._run_territory_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "failed"
    assert result["error"] == "boom"
    assert result["rows_affected"] == 0


def test_impl_invalid_window_returns_failed() -> None:
    """Inverted window short-circuits before any DB connect."""
    with patch.object(etl, "_connect_to_db") as connect:
        result = etl._run_territory_rollup_impl(
            start_date="2024-01-02T00:00:00Z",
            end_date="2024-01-01T00:00:00Z",
        )

    assert result["status"] == "failed"
    assert "must be strictly before" in result["error"]
    connect.assert_not_called()


def test_impl_default_window_is_the_arrival_window() -> None:
    """No dates supplied (the daily beat) -> the same ARRIVAL window as the per-HCP rollup.

    Changed deliberately (canonical TRx lane, owner decision #3): the former 24 h
    metric_date window wrote only yesterday's territory date. Measured over 56 days: 200
    rows on 5 Monday dates only, each written the NEXT day at 03:45, and 3 of the last 8
    Mondays have no territory rows at all. So the old assertion pinned defective behaviour
    as correct, and keeping it would have required the fix to fail the suite.

    ⚠ SELF-REFERENCE, STATED UP FRONT (the 22A lesson, applied before the plant rather
    than after): ``delta_hours == etl.ARRIVAL_WINDOW_HOURS`` cannot catch a reverted
    constant, because both sides move together. This test's own teeth are ``selected_by``
    — the duration half is a consistency check between the constant and the impl's
    arithmetic, nothing more. The VALUE is anchored by two other tests, named in
    ``test_the_arrival_window_and_the_beat_order_match_the_per_hcp_rollup``.
    """
    conn = _make_mock_conn(rowcount=1)

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_territory_rollup_impl()

    assert result["status"] == "completed"
    assert result["selected_by"] == "arrival"
    start = datetime.fromisoformat(result["window_start"])
    end = datetime.fromisoformat(result["window_end"])
    delta_hours = (end - start).total_seconds() / 3600.0
    assert delta_hours == pytest.approx(etl.ARRIVAL_WINDOW_HOURS, abs=1e-6)


def test_impl_closes_conn_even_on_error() -> None:
    """If execute() raises, the connection is still closed."""
    conn = _make_mock_conn(rowcount=0)
    cur = conn.cursor.return_value
    cur.execute.side_effect = RuntimeError("query exploded")

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_territory_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "failed"
    conn.close.assert_called_once()


def test_impl_passes_request_id_through() -> None:
    """The request_id arg is forwarded but does not change behaviour."""
    conn = _make_mock_conn(rowcount=1)

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_territory_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
            request_id="celery-task-xyz",
        )

    assert result["status"] == "completed"


def test_impl_delegates_window_resolution_to_resolve_window() -> None:
    """``_resolve_window`` is the single source of truth for window math.

    Pin that the impl calls the helper with the user's args (don't re-test
    the helper itself -- ``test_common.py`` covers the resolver directly).
    """
    conn = _make_mock_conn(rowcount=1)
    with (
        patch.object(etl, "_connect_to_db", return_value=conn),
        patch.object(
            etl,
            "_resolve_window",
            wraps=etl._resolve_window,
        ) as resolve,
    ):
        etl._run_territory_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    resolve.assert_called_once_with("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")


# =============================================================================
# Celery task wrapper
# =============================================================================


def test_celery_task_delegates_to_impl() -> None:
    """The Celery task ``run_territory_rollup`` is a thin shim over
    ``_run_territory_rollup_impl`` — verify it forwards args + the
    request id."""
    sentinel_result = {"status": "completed", "rows_affected": 3}
    with patch.object(etl, "_run_territory_rollup_impl", return_value=sentinel_result) as impl:
        async_result = etl.run_territory_rollup.apply(
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
    assert etl.run_territory_rollup.name == "src.etl.territory_metrics_etl.run_territory_rollup"


def test_beat_schedule_entry_present() -> None:
    """The daily beat entry routes the task to the analytics queue.

    #1645: the cadence is a wall-clock crontab, not the old bare ``86400.0``
    interval. The slot (03:45 UTC) is 30 min behind the per-HCP rollup, which is
    what finally makes this module's documented "per-HCP must run first" ordering
    real — under intervals both entries came due on the same beat tick. Slot map
    in ``src/workers/celery_app.py``.
    """
    from celery.schedules import crontab

    from src.workers.celery_app import celery_app

    entry = celery_app.conf.beat_schedule.get("territory-metrics-rollup")
    assert entry is not None, "beat schedule entry missing"
    assert entry["task"] == "src.etl.territory_metrics_etl.run_territory_rollup"
    assert entry["schedule"] == crontab(hour=3, minute=45)
    assert entry["options"]["queue"] == "analytics"


# =============================================================================
# Provenance inheritance (issue #895)
# =============================================================================


class TestProvenanceInheritance:
    """Issue #895: second-order laundering into territory_metrics.

    ``territory_metrics`` gains ``is_synthetic`` in migration 074. The
    rollup aggregates per-HCP ``business_metrics`` rows (themselves now
    provenance-tagged by 6B-infra-2a post-#895), raw ``triggers``, and
    ``hcp_profiles`` volumes -- all three carry is_synthetic, and the
    derived (territory_id, metric_date) row must inherit
    ``bool(any synthetic input)``.
    """

    def test_insert_column_list_includes_is_synthetic(self) -> None:
        """The INSERT column list must name is_synthetic explicitly so the
        migration-074 default false can never apply to a derived row."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        stripped = re.sub(r"--[^\n]*", "", sql)
        insert_clause = stripped.split("INSERT INTO territory_metrics", 1)[1].split("SELECT", 1)[0]
        assert "is_synthetic" in insert_clause, (
            "is_synthetic missing from INSERT column list -- derived rows "
            "would land with the migration-074 default false (laundering)"
        )

    def test_provenance_aggregated_via_bool_or(self) -> None:
        """Row-level provenance must be collapsed with BOOL_OR (any synthetic
        input taints the aggregate)."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert re.search(r"BOOL_OR\s*\(", sql, re.IGNORECASE), (
            "no BOOL_OR aggregation of is_synthetic -- provenance cannot "
            "be inherited per-cell without it"
        )

    def test_all_three_aggregate_ctes_contribute_provenance(self) -> None:
        """per_hcp_in_territory (business_metrics), active_hcp_per_territory_
        date (triggers) and territory_hcp_volume (hcp_profiles) each feed an
        aggregate column of the derived row, so each must carry provenance
        into the final is_synthetic."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert "bm.is_synthetic" in sql, "business_metrics provenance not read"
        assert "t.is_synthetic" in sql, "triggers provenance not read"
        for alias in ("pht.any_synthetic", "ahd.any_synthetic", "thv.any_synthetic"):
            assert alias in sql, f"{alias} not propagated to the final SELECT"

    def test_on_conflict_update_arm_preserves_provenance(self) -> None:
        """Idempotent re-runs go through the DO UPDATE arm; laundered
        semantics must not survive via that path either."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        on_conflict_block = sql.split("ON CONFLICT (territory_id, metric_date) DO UPDATE SET", 1)[1]
        assert re.search(r"is_synthetic\s*=\s*EXCLUDED\.is_synthetic", on_conflict_block), (
            "is_synthetic missing from the ON CONFLICT SET list -- a re-run "
            "would keep a stale provenance tag"
        )


def test_the_territory_rollup_reads_the_honest_trigger_columns():
    """Canonical TRx lane / migration 144.

    Two negatives, deliberately: the SQL constant must not name a legacy column
    (that is the behaviour), and neither must the MODULE (that is the prose).
    This lane has already found three doc comments asserting the inverse of the
    gate they described, so a stale sentence here is a real failure mode and not
    a tidiness preference.
    """
    import inspect
    import re as _re

    from src.etl import territory_metrics_etl as _etl
    from tests._checkout_guard import assert_same_checkout

    # Capability, not a literal path: the module must come from THIS checkout.
    # A `.worktrees/<lane>` substring was the first form and it could only ever
    # pass inside one directory on one machine -- see tests/_checkout_guard.py.
    assert_same_checkout(_etl, __file__)
    sql = _etl.INSERT_TERRITORY_ROLLUP_SQL
    assert "bm.triggers_delivered_count" in sql
    assert "bm.triggers_accepted_count" in sql
    assert not _re.search(r"\b(trx_count|nrx_count|total_rx_count)\b", sql)
    assert not _re.search(r"\b(trx_count|nrx_count|total_rx_count)\b", inspect.getsource(_etl))


# =============================================================================
# Late-arrival selection (canonical TRx lane, owner decision #3 2026-09-15)
# =============================================================================


def _cte(sql: str, name: str, following: str) -> str:
    return sql.split(f"{name} AS (", 1)[1].split(f"{following} AS (", 1)[0]


def test_the_explicit_variant_keeps_the_metric_date_window() -> None:
    body = _cte(etl.INSERT_TERRITORY_ROLLUP_SQL, "metric_dates", "territories")
    assert re.search(r"bm\.metric_date\s*>=\s*%\(start_date\)s::DATE", body)
    assert "created_at" not in body


def test_the_scheduled_variant_selects_dates_by_arrival_and_lookback_reach() -> None:
    body = _cte(etl.INSERT_TERRITORY_ROLLUP_BY_ARRIVAL_SQL, "metric_dates", "territories")
    assert re.search(r"bm\.created_at\s*>=\s*%\(start_date\)s", body)
    assert re.search(r"t\.created_at\s*>=\s*%\(start_date\)s", body)
    assert re.search(
        r"t\.trigger_timestamp\s*>=\s*per_hcp\.metric_date\s*-\s*INTERVAL '30 days'", body
    )
    assert re.search(
        r"t\.trigger_timestamp\s*<\s*per_hcp\.metric_date\s*\+\s*INTERVAL '1 day'", body
    )
    assert "::DATE" not in body


def test_every_variant_rebuilds_whole_dates() -> None:
    for sql in (
        etl.INSERT_TERRITORY_ROLLUP_SQL,
        etl.INSERT_TERRITORY_ROLLUP_BY_ARRIVAL_SQL,
        etl.PREVIEW_TERRITORY_ROLLUP_SQL,
    ):
        per_hcp = _cte(sql, "per_hcp_in_territory", "active_hcp_per_territory_date")
        assert "bm.metric_date IN (SELECT metric_date FROM metric_dates)" in per_hcp
        assert "%(start_date)s" not in per_hcp and "%(end_date)s" not in per_hcp
        assert "CROSS JOIN metric_dates" in sql


def test_the_variants_share_everything_after_the_metric_dates_cte() -> None:
    def after(sql: str) -> str:
        return sql.split("territories AS (", 1)[1]

    assert after(etl.INSERT_TERRITORY_ROLLUP_SQL) == after(
        etl.INSERT_TERRITORY_ROLLUP_BY_ARRIVAL_SQL
    )


def test_the_arrival_variant_keeps_the_seeded_columns_out_of_the_update_arm() -> None:
    block = etl.INSERT_TERRITORY_ROLLUP_BY_ARRIVAL_SQL.split(
        "ON CONFLICT (territory_id, metric_date) DO UPDATE SET", 1
    )[1]
    assert "market_potential" not in block and "resource_allocation_score" not in block


def test_the_arrival_window_and_the_beat_order_match_the_per_hcp_rollup() -> None:
    """⚠ THIS EQUALITY IS NOT A VALUE PIN ON ITS OWN. It passes if BOTH constants drift
    together — the same self-reference that made 22A's ``delta == ARRIVAL_WINDOW_HOURS``
    a tautology under a planted revert. What makes it real is a CHAIN with two
    independent anchors on literals, neither of which lives in this test:

      * ``test_business_metrics_per_hcp_etl.py::test_the_arrival_window_spans_a_weekly_batch_cycle_plus_margin``
        pins ``per_hcp.ARRIVAL_WINDOW_HOURS == 7 * 24 + ARRIVAL_MARGIN_HOURS`` with
        ``margin >= 6`` — arithmetic against the literals 7, 24 and 6.
      * ``test_impl_scheduled_run_uses_the_arrival_variant`` below pins the impl's computed
        ``window_start`` to the literal ``2026-09-06T21:45:00``.

    DELETING EITHER ANCHOR SILENTLY TURNS THIS TEST INTO A PROXY. If you are here because
    one of them is in your way, the two constants can then both be reverted to 24 with the
    whole suite green.
    """
    from src.etl import business_metrics_per_hcp_etl as per_hcp
    from src.workers.celery_app import celery_app

    assert etl.ARRIVAL_WINDOW_HOURS == per_hcp.ARRIVAL_WINDOW_HOURS
    beats = celery_app.conf.beat_schedule
    per_hcp_beat = beats["business-metrics-per-hcp-rollup"]["schedule"]
    territory_beat = beats["territory-metrics-rollup"]["schedule"]
    assert (per_hcp_beat.hour, per_hcp_beat.minute) == ({3}, {15})
    assert (territory_beat.hour, territory_beat.minute) == ({3}, {45})


def test_impl_scheduled_run_uses_the_arrival_variant() -> None:
    conn = _make_mock_conn(rowcount=3)
    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_territory_rollup_impl(arrived_before="2026-09-14T03:45:00+00:00")
    args, _ = conn.cursor.return_value.execute.call_args_list[0]  # [1] is the reconcile
    assert args[0] is etl.INSERT_TERRITORY_ROLLUP_BY_ARRIVAL_SQL
    assert result["selected_by"] == "arrival"
    # A LITERAL, deliberately: 03:45 − 174 h. This is one of the two anchors that stop the
    # window constant from drifting behind the equality pin above.
    assert result["window_start"].startswith("2026-09-06T21:45:00")


def test_impl_explicit_dates_keep_the_metric_date_window() -> None:
    conn = _make_mock_conn(rowcount=3)
    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_territory_rollup_impl(start_date="2026-05-01", end_date="2026-09-16")
    args, _ = conn.cursor.return_value.execute.call_args_list[0]  # [1] is the reconcile
    assert args[0] is etl.INSERT_TERRITORY_ROLLUP_SQL
    assert result["selected_by"] == "metric_date"


def test_impl_refuses_arrived_before_with_explicit_dates() -> None:
    with patch.object(etl, "_connect_to_db") as connect:
        result = etl._run_territory_rollup_impl(
            start_date="2026-05-01",
            end_date="2026-09-16",
            arrived_before="2026-09-14T03:45:00+00:00",
        )
    assert result["status"] == "failed" and "arrived_before" in result["error"]
    connect.assert_not_called()


def test_preview_reads_the_insert_text_and_writes_nothing() -> None:
    preview = etl.PREVIEW_TERRITORY_ROLLUP_SQL
    cte_text = etl.INSERT_TERRITORY_ROLLUP_SQL.split("INSERT INTO territory_metrics", 1)[0]
    assert preview.startswith(cte_text.rstrip())
    assert etl._TERRITORY_ROLLUP_ROWS_SELECT in preview
    assert etl._TERRITORY_ROLLUP_ROWS_SELECT in etl.INSERT_TERRITORY_ROLLUP_SQL
    code = "\n".join(line.split("--", 1)[0] for line in preview.splitlines())
    assert not re.search(r"\b(INSERT|UPDATE|DELETE|MERGE)\b|ON CONFLICT", code)


def test_preview_runs_in_a_read_only_transaction() -> None:
    from datetime import date

    conn = _make_mock_conn()
    cur = conn.cursor.return_value
    cur.fetchone.return_value = (91, 3640, 12, 1840, 40, 2, date(2026, 5, 1), date(2026, 9, 14))
    cur.fetchall.return_value = [
        ("T-07", date(2026, 6, 2), False),
        ("T-11", date(2026, 6, 3), False),
        ("T-01", date(2026, 7, 4), True),
    ]
    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl.preview_territory_rollup("2026-05-01", "2026-09-16")
    first, second, third = cur.execute.call_args_list
    assert first.args == ("SET TRANSACTION READ ONLY",)
    assert second.args[0] is etl.PREVIEW_TERRITORY_ROLLUP_SQL
    assert third.args[0] is etl.PREVIEW_TERRITORY_OBSOLETE_SQL
    assert third.args[1]["limit"] == etl.PREVIEW_KEY_LIMIT
    assert (
        result["metric_dates"],
        result["rows_new"],
        result["rows_changed"],
        result["rows_existing"],
    ) == (91, 3640, 12, 1840)
    assert result["rows_obsolete"] == 40  # codex r13-08
    # codex r14-04: the readout names the rows, and separates ours from someone else's.
    assert result["rows_obsolete_foreign"] == 2
    assert result["obsolete_keys"] == [("T-01", "2026-07-04")]
    assert result["obsolete_foreign_keys"] == [("T-07", "2026-06-02"), ("T-11", "2026-06-03")]


def test_the_reconcile_deletes_territory_rows_that_are_no_longer_produced() -> None:
    for sql, scope in (
        (etl.RECONCILE_TERRITORY_ROLLUP_SQL, "m.metric_date >= %(start_date)s::DATE"),
        (
            etl.RECONCILE_TERRITORY_ROLLUP_BY_ARRIVAL_SQL,
            "m.metric_date IN (SELECT metric_date FROM metric_dates)",
        ),
    ):
        assert "DELETE FROM territory_metrics m" in sql
        assert scope in sql
        assert "r.territory_id = m.territory_id AND r.metric_date = m.metric_date" in sql
    # A date whose per-HCP rows all vanished is not in metric_dates: only the calendar
    # range reaches its stale territory rows. Bare name, not "metric_dates)" — the spec's
    # paren-carrying literal is satisfied by `FROM metric_dates )` with one space, which is
    # exactly the too-narrow scope this asserts against (same fix as 22A).
    assert "metric_dates" not in etl.RECONCILE_TERRITORY_ROLLUP_SQL.split("DELETE FROM", 1)[1]


def test_the_reconcile_only_deletes_rows_this_etl_owns() -> None:
    """codex r14-04: territory_metrics is shared. The ON CONFLICT SET arm refuses to overwrite
    market_potential / resource_allocation_score; the DELETE must refuse the same rows, or the
    reconcile silently destroys what the upsert was written to protect. Measured 2026-09-15 and
    re-verified 2026-09-17: both columns are NULL in all 1,840 live rows, so this predicate
    deletes exactly as much as today — a guard for the first budget writer, not a behaviour
    change. This is the PRESENCE check; the discriminating behaviour is
    ``test_the_ownership_predicates_partition_every_obsolete_row`` below and the planted
    integration pair."""
    for sql in (etl.RECONCILE_TERRITORY_ROLLUP_SQL, etl.RECONCILE_TERRITORY_ROLLUP_BY_ARRIVAL_SQL):
        delete = sql.split("DELETE FROM territory_metrics m", 1)[1]
        assert "m.market_potential IS NULL" in delete
        assert "m.resource_allocation_score IS NULL" in delete
    # The same two columns, and only those two, decide ownership in the SET arm and here.
    assert etl._TERRITORY_OWNED_BY_THIS_ETL.count("IS NULL") == 2
    set_arm = etl.INSERT_TERRITORY_ROLLUP_SQL.split("DO UPDATE SET", 1)[1]
    assert "market_potential" not in set_arm and "resource_allocation_score" not in set_arm


def _flat_null_predicate(pred: str) -> tuple[str, list[tuple[str, bool]]]:
    """Parse ``<col> IS [NOT] NULL`` terms joined by a single connective.

    Refuses anything else — nesting, a comparison, an extra column — so a clause this
    evaluator cannot model raises instead of being silently ignored. That refusal is the
    positive control on the evaluator itself.
    """
    text = pred.strip().strip("()").strip()
    connective = "AND" if " AND " in text else "OR" if " OR " in text else ""
    parts = text.split(f" {connective} ") if connective else [text]
    terms: list[tuple[str, bool]] = []
    for part in parts:
        tokens = part.split()
        if len(tokens) == 3 and tokens[1].upper() == "IS" and tokens[2].upper() == "NULL":
            terms.append((tokens[0].split(".")[-1], True))
        elif (
            len(tokens) == 4
            and tokens[1].upper() == "IS"
            and tokens[2].upper() == "NOT"
            and tokens[3].upper() == "NULL"
        ):
            terms.append((tokens[0].split(".")[-1], False))
        else:
            raise AssertionError(f"evaluator cannot model {part!r} from {pred!r}")
    return connective, terms


def _holds(pred: str, row: dict) -> bool:
    connective, terms = _flat_null_predicate(pred)
    results = [(row[col] is None) == wants_null for col, wants_null in terms]
    return all(results) if connective != "OR" else any(results)


def _extracted_ownership_predicates() -> tuple[str, str]:
    """Pull both predicates OUT of the composed SQL — never restate them.

    This is the construction the lane already used to pin ``SUPPORTED_METRICS`` from
    migration 143's own ``IN ('trx','nrx','nbrx')``: read the thing under test out of the
    artefact that ships, so the test cannot agree with a copy while the server runs
    something else. Restating them here would make the partition proof below valid only
    for a predicate shape nobody guarantees.

    Both regexes assert their own match, so a restructure of the DELETE or the preview
    subquery fails loudly instead of silently extracting the wrong span.
    """
    owned_match = re.search(
        r"DELETE FROM territory_metrics m\n WHERE .*?\n   AND (.*?)\n   AND NOT EXISTS",
        etl.RECONCILE_TERRITORY_ROLLUP_SQL,
        re.S,
    )
    assert owned_match, "could not extract the ownership predicate from the DELETE"
    foreign_block = etl.PREVIEW_TERRITORY_ROLLUP_SQL.split("AS rows_obsolete_foreign", 1)[0]
    foreign_match = re.search(r"AND \(([^)]*IS NOT NULL[^)]*)\)\n", foreign_block)
    assert foreign_match, "could not extract the foreign predicate from rows_obsolete_foreign"
    return owned_match.group(1).strip(), foreign_match.group(1).strip()


def test_the_extracted_predicates_are_the_ones_the_module_declares() -> None:
    """Round trip on the extraction itself: what came out of the SQL must be what the
    module declares. If this fails, the evaluator below is reading a different predicate
    from the one the constants describe, and neither is trustworthy."""
    owned, foreign = _extracted_ownership_predicates()
    assert owned == etl._TERRITORY_OWNED_BY_THIS_ETL
    assert foreign == etl._TERRITORY_NOT_OWNED_BY_THIS_ETL


def test_the_ownership_predicates_partition_every_obsolete_row() -> None:
    """THE DISCRIMINATING PROPERTY, evaluated rather than asserted by text.

    ``rows_obsolete`` and ``rows_obsolete_foreign`` must PARTITION the obsolete set: every
    row counted exactly once, none twice and none dropped. Presence-of-clause cannot see a
    partition failure — two predicates can both be present and still overlap or leave a
    gap, which would make the Task 30 GO gate's number wrong in either direction.

    Faithful at this layer because ``IS NULL`` / ``IS NOT NULL`` are TOTAL in SQL: they
    return true or false for every value including NULL, with no three-valued UNKNOWN, so a
    Python translation of these two flat predicates cannot disagree with Postgres. Nothing
    here models the DELETE's join or scope — that behaviour is the planted integration
    pair's job, and it has never run.
    """
    owned, foreign = _extracted_ownership_predicates()
    cols = ("market_potential", "resource_allocation_score")
    for mp in (None, 1.0):
        for ras in (None, 0.5):
            row = dict(zip(cols, (mp, ras), strict=True))
            in_owned = _holds(owned.replace("m.", "").replace("o.", ""), row)
            in_foreign = _holds(foreign.replace("m.", "").replace("o.", ""), row)
            assert in_owned != in_foreign, (
                f"{row} is in {'both' if in_owned else 'neither'} bucket: "
                "rows_obsolete + rows_obsolete_foreign is not the obsolete set"
            )
    # And the owned bucket is the all-NULL row specifically — the direction that matters:
    # a row nobody else has written is ours to delete.
    assert _holds(owned.replace("m.", ""), dict.fromkeys(cols)) is True
    assert (
        _holds(
            owned.replace("m.", ""),
            {"market_potential": 1.0, "resource_allocation_score": None},
        )
        is False
    )


def test_ownership_is_not_decided_by_is_synthetic_or_created_at() -> None:
    """is_synthetic describes the INPUTS (a real-data row would legitimately be false) and
    created_at cannot tell this ETL's 2026-06-14 backfill from a future writer's insert.
    Neither is an ownership signal; using one would delete real rows or spare stale ones."""
    assert "is_synthetic" not in etl._TERRITORY_OWNED_BY_THIS_ETL
    assert "created_at" not in etl._TERRITORY_OWNED_BY_THIS_ETL


def test_the_preview_counts_exactly_what_the_reconcile_deletes() -> None:
    """A GO gate on rows_obsolete is only meaningful if the number is the deletion set."""
    counts = etl.PREVIEW_TERRITORY_ROLLUP_SQL.split("AS rows_obsolete,", 1)[0].rsplit(
        "(SELECT count(*)", 1
    )[1]
    assert "o.market_potential IS NULL AND o.resource_allocation_score IS NULL" in counts
    foreign = etl.PREVIEW_TERRITORY_ROLLUP_SQL.split("AS rows_obsolete_foreign,", 1)[0].rsplit(
        "(SELECT count(*)", 1
    )[1]
    assert "o.market_potential IS NOT NULL OR o.resource_allocation_score IS NOT NULL" in foreign


def test_the_obsolete_readout_is_read_only_and_lists_foreign_rows_first() -> None:
    code = "\n".join(
        line.split("--", 1)[0] for line in etl.PREVIEW_TERRITORY_OBSOLETE_SQL.splitlines()
    )
    assert not re.search(r"\b(INSERT|UPDATE|DELETE|MERGE)\b|ON CONFLICT", code)
    assert "ORDER BY owned_by_this_etl" in code  # false sorts first: the rows a reviewer must see
    assert "LIMIT %(limit)s" in code


def test_impl_reconciles_in_the_same_transaction_as_the_upsert() -> None:
    conn = _make_mock_conn(rowcount=2)
    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_territory_rollup_impl(arrived_before="2026-09-14T03:45:00+00:00")
    executed = [call.args[0] for call in conn.cursor.return_value.execute.call_args_list]
    assert executed == [
        etl.INSERT_TERRITORY_ROLLUP_BY_ARRIVAL_SQL,
        etl.RECONCILE_TERRITORY_ROLLUP_BY_ARRIVAL_SQL,
    ]
    assert result["status"] == "completed" and result["rows_deleted"] == 2
